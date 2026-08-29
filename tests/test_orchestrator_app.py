"""Tests for the Forge Orchestrator two-repo workflow app.

Covers session management, the validation engine, background
optimization, finalization, the XSS-safe dashboard, and concurrency.

No real network or git-clone calls are made: ``_clone_repo`` is monkey-
patched to copy a pre-seeded repo tree into the session directory, and
``evaluate_branch`` / ``run_round`` are mocked so no LLM or eval is
invoked.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from anvil.eval import EvalReport
from anvil.loop.decision import Decision
from anvil.loop.round import RoundReport
from anvil.orchestrator import app as app_module
from anvil.orchestrator.app import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_sessions() -> None:
    """Purge the in-memory session store before and after each test."""
    app_module._sessions.clear()
    yield
    app_module._sessions.clear()


@pytest.fixture
def sessions_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the session storage root to a tmp dir for isolation."""
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(app_module, "_SESSIONS_ROOT", root)
    return root


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Repo seeding helper — builds a valid (or partially-valid) agent repo
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    """Run a git command in ``repo`` with a test identity."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _seed_repo(
    dest: Path,
    *,
    mode: str = "prompt",
    with_scaffold: bool = True,
    with_golden: bool = True,
    with_config: bool = True,
    with_agents: bool = False,
    skill_files: list[str] | None = None,
    held_out_test: bool = True,
    extra_config: dict | None = None,
) -> Path:
    """Build an agent repo structure at ``dest`` and ``git init`` it.

    ``skill_files`` controls which skill .md files are created (defaults
    to ["identity.md"]). The harness.yaml ``skills`` list always
    references them, so ``scaffold_skill_files`` validation passes.
    """
    dest.mkdir(parents=True, exist_ok=True)

    if with_scaffold:
        scaffold = dest / "scaffold"
        scaffold.mkdir(exist_ok=True)
        files = skill_files or ["identity.md"]
        for f in files:
            (scaffold / f).write_text(f"# {f}\n", encoding="utf-8")
        harness = {
            "sampling": {"temperature": 0.3, "max_tokens": 2048},
            "skills": [{"file": f} for f in files],
        }
        (scaffold / "harness.yaml").write_text(
            yaml.safe_dump(harness, sort_keys=False), encoding="utf-8"
        )

    if with_golden:
        data_dir = dest / "data"
        data_dir.mkdir(exist_ok=True)
        line = json.dumps(
            {
                "example_id": "ex1",
                "query": "What is the policy?",
                "category": "direct",
                "expected": "The answer.",
            }
        )
        (data_dir / "golden_set.jsonl").write_text(line + "\n", encoding="utf-8")

    if with_config:
        harness_dir = dest / "harness"
        harness_dir.mkdir(exist_ok=True)
        config: dict[str, Any] = {
            "mode": mode,
            "runtime_endpoint": "databricks-claude-sonnet-4-6",
            "optimizer_endpoint": "databricks-claude-opus-4-7",
            "judge_endpoint": "databricks-claude-sonnet-4-6",
            "eval": {
                "default_mode": "quick",
                "modes": {"quick": {"rows": 8}, "standard": {"rows": 12}, "full": {"rows": 20}},
                "scorers": ["correctness"],
                "held_out_test": held_out_test,
            },
            "gate": {"type": "frontier"},
            "loop": {"max_optimizer_turns": 30},
        }
        if mode == "code":
            config["agent_module"] = "agents.my_agent"
        if extra_config:
            config.update(extra_config)
        (harness_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    if with_agents:
        agents_dir = dest / "agents"
        agents_dir.mkdir(exist_ok=True)
        (agents_dir / "my_agent.py").write_text("class Agent:\n    pass\n", encoding="utf-8")

    _git(dest, "init")
    _git(dest, "add", "-A")
    _git(dest, "commit", "-m", "initial")
    return dest


def _mock_clone_factory(source: Path):
    """Return a mock ``_clone_repo`` that copies ``source`` into ``dest``."""

    def _mock(repo_url: str, dest_path: Path, github_token: str | None) -> str | None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(source), str(dest_path))
        return None

    return _mock


def _fake_eval_report() -> EvalReport:
    return EvalReport(
        aggregate=0.85,
        per_judge={"correctness": 0.9},
        per_bucket={},
        failures=[],
        run_id="test-run-123",
        experiment_id="exp-1",
        n_rows=8,
        mode="quick",
        scorers=["correctness"],
        evaluated_at="2024-01-01T00:00:00",
    )


def _make_mock_run_round(calls: list[int] | None = None):
    """Return a mock ``run_round`` that writes a round JSON + frontier."""

    def _mock(
        *,
        round_id: int,
        repo_root: Path | str,
        eval_mode: str | None = None,
        max_turns: int = 30,
        **_kw: Any,
    ) -> RoundReport:
        if calls is not None:
            calls.append(round_id)
        root = Path(repo_root)
        runs = root / "eval" / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        payload = {
            "round_id": round_id,
            "branch": f"anvil/exp-round-{round_id}",
            "decision": "keep",
            "action_kind": "edit_skill",
            "parse_status": "ok",
            "baseline_score": 0.80,
            "score_delta_vs_parent": 0.05,
            "aggregate": 0.85,
        }
        (runs / f"round_{round_id:03d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        # Write a frontier so finalize + get_frontier can find one.
        frontier = {
            "best": {"aggregate": 0.85, "correctness": 0.9},
            "objectives": ["aggregate", "correctness"],
            "pareto": False,
            "directions": {},
            "sources": {},
            "epsilon": 0.0,
        }
        (runs / "frontier.json").write_text(json.dumps(frontier), encoding="utf-8")
        return RoundReport(
            round_id=round_id,
            branch=f"anvil/exp-round-{round_id}",
            decision=Decision.KEEP,
            action_kind="edit_skill",
            parse_status="ok",
            diff_summary="edited skill X",
        )

    return _mock


def _create_valid_session(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **seed_kwargs: Any,
) -> str:
    """Seed a repo, mock clone, POST /api/session, return session_id."""
    source = tmp_path / "agent-repo"
    _seed_repo(source, **seed_kwargs)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "validated", data
    return data["session_id"]


def _wait_for_status(
    client: TestClient, session_id: str, statuses: set[str], timeout: float = 10.0
) -> dict[str, Any]:
    """Poll GET /api/session until status is in ``statuses`` or timeout."""
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        resp = client.get(f"/api/session/{session_id}")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] in statuses:
            return last
        time.sleep(0.02)
    raise AssertionError(
        f"timed out waiting for status in {statuses}; last status={last.get('status')!r} data={last}"
    )


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_session_valid_repo(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "validated"
    assert data["validation"]["status"] == "valid"
    assert data["session_id"]
    assert data["config"] is not None
    assert data["config"]["mode"] == "prompt"


def test_create_session_missing_scaffold(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_scaffold=False)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "invalid"
    assert data["validation"]["status"] == "invalid"
    checks = {c["name"]: c for c in data["validation"]["checks"]}
    assert checks["scaffold_harness_yaml"]["status"] == "fail"
    assert "remediation" in checks["scaffold_harness_yaml"]


def test_create_session_missing_golden_set(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_golden=False)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    assert data["status"] == "invalid"
    checks = {c["name"]: c for c in data["validation"]["checks"]}
    assert checks["golden_set_jsonl"]["status"] == "fail"


def test_create_session_missing_config(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_config=False)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    assert data["status"] == "invalid"
    checks = {c["name"]: c for c in data["validation"]["checks"]}
    assert checks["harness_config_yaml"]["status"] == "fail"


def test_create_session_code_mode_missing_agents(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source, mode="code", with_agents=False)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    assert data["status"] == "invalid"
    checks = {c["name"]: c for c in data["validation"]["checks"]}
    assert checks["agent_code"]["status"] == "fail"


def test_create_session_clone_failure(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    def fail_clone(repo_url: str, dest_path: Path, github_token: str | None) -> str | None:
        return "fatal: repository not found"

    monkeypatch.setattr(app_module, "_clone_repo", fail_clone)
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/bad"})
    assert resp.status_code == 400
    assert "repository not found" in resp.json()["detail"]


def test_get_session(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == sid
    assert data["status"] == "validated"
    assert data["validation"]["status"] == "valid"
    assert data["repo_url"] == "https://github.com/user/repo"


def test_get_session_not_found(client: TestClient) -> None:
    resp = client.get("/api/session/nonexistent")
    assert resp.status_code == 404


def test_get_session_config(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}/config")
    assert resp.status_code == 200
    config = resp.json()
    assert config["mode"] == "prompt"
    assert config["runtime_endpoint"] == "databricks-claude-sonnet-4-6"
    assert config["eval"]["default_mode"] == "quick"


def test_get_session_config_not_validated(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_config=False)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    sid = resp.json()["session_id"]
    resp2 = client.get(f"/api/session/{sid}/config")
    assert resp2.status_code == 404


def test_config_redacts_secrets(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    extra = {
        "optimizer": {
            "backend": "omnigent",
            "server_url": "http://localhost:6767",
            "auth_token": "super-secret-token",
        },
        "my_secret_field": "shhh",
        "db_password": "hunter2",
        "api_credential": "cred-123",
        "normal_field": "visible",
    }
    _seed_repo(source, extra_config=extra)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    config = data["config"]
    assert config["optimizer"]["auth_token"] == "***"
    assert config["my_secret_field"] == "***"
    assert config["db_password"] == "***"
    assert config["api_credential"] == "***"
    assert config["normal_field"] == "visible"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_all_pass(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source, mode="code", with_agents=True, skill_files=["identity.md", "rules.md"])
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    assert data["validation"]["status"] == "valid"
    for c in data["validation"]["checks"]:
        assert c["status"] in ("pass", "warn"), c
    # parent_branch is a warn (fresh clone has no anvil/exp).
    checks = {c["name"]: c for c in data["validation"]["checks"]}
    assert checks["parent_branch"]["status"] == "warn"


def test_validation_scaffold_missing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_scaffold=False)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    checks = {c["name"]: c for c in resp.json()["validation"]["checks"]}
    assert checks["scaffold_harness_yaml"]["status"] == "fail"
    assert checks["scaffold_harness_yaml"]["remediation"]


def test_validation_golden_set_empty(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source)
    (source / "data" / "golden_set.jsonl").write_text("\n\n", encoding="utf-8")
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    checks = {c["name"]: c for c in resp.json()["validation"]["checks"]}
    assert checks["golden_set_jsonl"]["status"] == "fail"
    assert "empty" in checks["golden_set_jsonl"]["message"].lower()


def test_validation_golden_set_invalid_json(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source)
    (source / "data" / "golden_set.jsonl").write_text(
        '{"example_id": "a", "query": "q"}\nnot json{\n', encoding="utf-8"
    )
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    checks = {c["name"]: c for c in resp.json()["validation"]["checks"]}
    assert checks["golden_set_jsonl"]["status"] == "fail"
    assert "Line 2" in checks["golden_set_jsonl"]["message"]


def test_validation_missing_eval_modes(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source)
    # Remove the modes section from the config.
    config_path = source / "harness" / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del raw["eval"]["modes"]
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    checks = {c["name"]: c for c in resp.json()["validation"]["checks"]}
    assert checks["eval_modes"]["status"] == "fail"


def test_validation_code_mode_no_agents(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source, mode="code", with_agents=False)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    checks = {c["name"]: c for c in resp.json()["validation"]["checks"]}
    assert checks["agent_code"]["status"] == "fail"


def test_validation_warn_parent_branch(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    checks = {c["name"]: c for c in resp.json()["validation"]["checks"]}
    assert checks["parent_branch"]["status"] == "warn"
    assert checks["parent_branch"].get("remediation") is None


def test_validation_skill_file_missing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source, skill_files=["identity.md"])
    # Reference a non-existent skill file in harness.yaml.
    harness_path = source / "scaffold" / "harness.yaml"
    raw = yaml.safe_load(harness_path.read_text(encoding="utf-8"))
    raw["skills"].append({"file": "missing.md"})
    harness_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    checks = {c["name"]: c for c in resp.json()["validation"]["checks"]}
    assert checks["scaffold_skill_files"]["status"] == "fail"
    assert "missing.md" in checks["scaffold_skill_files"]["message"]


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def test_optimize_starts_baseline(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    resp = client.post(
        f"/api/session/{sid}/optimize", json={"max_rounds": 2, "max_turns": 5}
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "building_baseline"
    data = _wait_for_status(client, sid, {"optimizing", "optimized"})
    assert data["status"] in {"optimizing", "optimized"}


def test_optimize_runs_rounds(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 3, "max_turns": 5})
    data = _wait_for_status(client, sid, {"optimized"})
    assert data["status"] == "optimized"
    resp = client.get(f"/api/session/{sid}/rounds")
    assert resp.status_code == 200
    rounds = resp.json()
    assert len(rounds) == 3
    assert rounds[0]["round_id"] == 1
    assert rounds[2]["round_id"] == 3
    assert all(r["decision"] == "keep" for r in rounds)


def test_optimize_max_rounds(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    calls: list[int] = []
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round(calls))
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 3, "max_turns": 5})
    _wait_for_status(client, sid, {"optimized"})
    assert calls == [1, 2, 3]
    resp = client.get(f"/api/session/{sid}/rounds")
    assert len(resp.json()) == 3


def test_optimize_already_optimizing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    # Simulate an in-progress optimization.
    app_module._sessions[sid].status = "optimizing"
    resp = client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1})
    assert resp.status_code == 409


def test_optimize_finalized(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    app_module._sessions[sid].status = "finalized"
    resp = client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1})
    assert resp.status_code == 409


def test_optimize_not_validated(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_scaffold=False)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    sid = resp.json()["session_id"]
    resp2 = client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1})
    assert resp2.status_code == 409


def test_get_rounds_empty(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}/rounds")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_rounds_after_optimization(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 2, "max_turns": 5})
    _wait_for_status(client, sid, {"optimized"})
    resp = client.get(f"/api/session/{sid}/rounds")
    assert len(resp.json()) == 2


def test_get_round_by_id(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1, "max_turns": 5})
    _wait_for_status(client, sid, {"optimized"})
    resp = client.get(f"/api/session/{sid}/rounds/1")
    assert resp.status_code == 200
    assert resp.json()["round_id"] == 1


def test_get_round_not_found(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}/rounds/999")
    assert resp.status_code == 404


def test_get_baseline(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1, "max_turns": 5})
    _wait_for_status(client, sid, {"optimized"})
    resp = client.get(f"/api/session/{sid}/baseline")
    assert resp.status_code == 200
    data = resp.json()
    assert data["aggregate"] == 0.85
    assert data["mode"] == "quick"
    assert data["n_examples"] == 8


def test_get_baseline_not_found(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}/baseline")
    assert resp.status_code == 404


def test_get_frontier(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1, "max_turns": 5})
    _wait_for_status(client, sid, {"optimized"})
    resp = client.get(f"/api/session/{sid}/frontier")
    assert resp.status_code == 200
    data = resp.json()
    assert "best" in data
    assert data["best"]["aggregate"] == 0.85


def test_get_frontier_not_found(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}/frontier")
    assert resp.status_code == 404


def test_optimize_error_state(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)

    def boom(**kw: Any) -> EvalReport:
        raise RuntimeError("baseline exploded")

    monkeypatch.setattr(app_module, "evaluate_branch", boom)
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1, "max_turns": 5})
    data = _wait_for_status(client, sid, {"error"})
    assert data["status"] == "error"
    assert "baseline exploded" in data["error"]


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


def _optimize_then_finalize_setup(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sessions_root: Path,
    max_rounds: int = 1,
) -> str:
    """Run a full optimization (mocked) and return the session_id."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    client.post(
        f"/api/session/{sid}/optimize", json={"max_rounds": max_rounds, "max_turns": 5}
    )
    _wait_for_status(client, sid, {"optimized"})
    return sid


def test_finalize_success(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _optimize_then_finalize_setup(client, tmp_path, monkeypatch, sessions_root)
    resp = client.post(f"/api/session/{sid}/finalize")
    assert resp.status_code == 200
    data = resp.json()
    assert data["aggregate"] == 0.85
    assert "scaffold_commit_sha" in data
    assert "frontier" in data
    assert "finalized_at" in data
    # Session status is now finalized.
    sess_resp = client.get(f"/api/session/{sid}")
    assert sess_resp.json()["status"] == "finalized"


def test_finalize_already_finalized(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _optimize_then_finalize_setup(client, tmp_path, monkeypatch, sessions_root)
    resp = client.post(f"/api/session/{sid}/finalize")
    assert resp.status_code == 200
    resp2 = client.post(f"/api/session/{sid}/finalize")
    assert resp2.status_code == 409


def test_finalize_no_frontier(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    # Manually set status to optimized without a frontier file.
    app_module._sessions[sid].status = "optimized"
    resp = client.post(f"/api/session/{sid}/finalize")
    assert resp.status_code == 409
    assert "frontier" in resp.json()["detail"].lower()


def test_finalize_not_optimized(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.post(f"/api/session/{sid}/finalize")
    assert resp.status_code == 409


def test_get_finalize(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _optimize_then_finalize_setup(client, tmp_path, monkeypatch, sessions_root)
    client.post(f"/api/session/{sid}/finalize")
    resp = client.get(f"/api/session/{sid}/finalize")
    assert resp.status_code == 200
    assert resp.json()["aggregate"] == 0.85


def test_get_finalize_not_found(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}/finalize")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_dashboard_returns_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    html = resp.text
    assert "Forge Orchestrator" in html
    assert "Select Agent Repository" in html
    assert "Compatibility Check" in html
    assert "Configure Optimization" in html
    assert "Optimization Progress" in html


def test_dashboard_xss_safe(client: TestClient) -> None:
    """The dashboard must use textContent/createElement, never innerHTML."""
    resp = client.get("/")
    html = resp.text
    # textContent is the XSS-safe DOM API for inserting untrusted text.
    assert "textContent" in html
    # createElement builds elements programmatically (no HTML string parsing).
    assert "createElement" in html
    # innerHTML would allow XSS if fed untrusted data — must be absent.
    assert "innerHTML" not in html


def test_dashboard_has_polling(client: TestClient) -> None:
    """Dashboard polls the session endpoint for live progress."""
    resp = client.get("/")
    html = resp.text
    assert "setInterval" in html
    assert "/api/session/" in html


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_sessions(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """Two sessions optimize concurrently without interfering."""
    # Seed two repos.
    src_a = tmp_path / "repo-a"
    src_b = tmp_path / "repo-b"
    _seed_repo(src_a)
    _seed_repo(src_b)

    # Clone mock that picks the right source based on the URL.
    sources = {
        "https://github.com/user/a": src_a,
        "https://github.com/user/b": src_b,
    }

    def smart_clone(repo_url: str, dest_path: Path, github_token: str | None) -> str | None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(sources[repo_url]), str(dest_path))
        return None

    monkeypatch.setattr(app_module, "_clone_repo", smart_clone)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())

    resp_a = client.post("/api/session", json={"repo_url": "https://github.com/user/a"})
    resp_b = client.post("/api/session", json={"repo_url": "https://github.com/user/b"})
    sid_a = resp_a.json()["session_id"]
    sid_b = resp_b.json()["session_id"]
    assert sid_a != sid_b

    client.post(f"/api/session/{sid_a}/optimize", json={"max_rounds": 2, "max_turns": 5})
    client.post(f"/api/session/{sid_b}/optimize", json={"max_rounds": 3, "max_turns": 5})

    _wait_for_status(client, sid_a, {"optimized"})
    _wait_for_status(client, sid_b, {"optimized"})

    rounds_a = client.get(f"/api/session/{sid_a}/rounds").json()
    rounds_b = client.get(f"/api/session/{sid_b}/rounds").json()
    assert len(rounds_a) == 2
    assert len(rounds_b) == 3
    # No cross-contamination: round IDs belong to their own session.
    assert {r["round_id"] for r in rounds_a} == {1, 2}
    assert {r["round_id"] for r in rounds_b} == {1, 2, 3}


def test_mutation_lock_same_session(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """A second POST /optimize on the same session gets 409."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    resp1 = client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 2, "max_turns": 5})
    assert resp1.status_code == 202
    # Second optimize on the same session — status is no longer "validated".
    resp2 = client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 2, "max_turns": 5})
    assert resp2.status_code == 409


def test_get_session_validation(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}/validation")
    assert resp.status_code == 200
    report = resp.json()
    assert report["status"] == "valid"
    assert len(report["checks"]) > 0


def test_session_response_model_fields(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """GET /api/session/{id} returns all expected top-level fields."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}")
    data = resp.json()
    for key in (
        "session_id",
        "repo_url",
        "status",
        "validation",
        "config",
        "baseline",
        "rounds",
        "frontier",
        "finalized",
        "error",
    ):
        assert key in data, f"missing {key}"


def test_optimize_default_eval_mode_from_config(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """When eval_mode is omitted, the config default_mode is used."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    captured: dict[str, Any] = {}

    def capturing_eval(**kw: Any) -> EvalReport:
        captured.update(kw)
        return _fake_eval_report()

    monkeypatch.setattr(app_module, "evaluate_branch", capturing_eval)
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1, "max_turns": 5})
    _wait_for_status(client, sid, {"optimized"})
    # The baseline eval should have been called with mode="quick" (config default).
    assert captured.get("mode") == "quick"


def test_optimize_explicit_eval_mode(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """An explicit eval_mode overrides the config default."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    captured: dict[str, Any] = {}

    def capturing_eval(**kw: Any) -> EvalReport:
        captured.update(kw)
        return _fake_eval_report()

    monkeypatch.setattr(app_module, "evaluate_branch", capturing_eval)
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    client.post(
        f"/api/session/{sid}/optimize",
        json={"eval_mode": "standard", "max_rounds": 1, "max_turns": 5},
    )
    _wait_for_status(client, sid, {"optimized"})
    assert captured.get("mode") == "standard"


def test_optimize_creates_parent_branch(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """POST /optimize creates the anvil/exp parent branch if missing."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    # Before optimize, anvil/exp doesn't exist (validation warned).
    sess = app_module._sessions[sid]
    assert not app_module._branch_exists(sess.repo_path, "anvil/exp")
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1, "max_turns": 5})
    _wait_for_status(client, sid, {"optimized"})
    # After optimize, anvil/exp was created.
    assert app_module._branch_exists(sess.repo_path, "anvil/exp")
