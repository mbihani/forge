"""Tests for the Forge Orchestrator two-repo workflow app.

Covers session management, the validation engine, background
optimization, finalization, the XSS-safe dashboard, and concurrency.

No real network or git-clone calls are made: ``_clone_repo`` is monkey-
patched to copy a pre-seeded repo tree into the session directory, and
``evaluate_branch`` / ``run_round`` are mocked so no LLM or eval is
invoked.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
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

    def _mock(
        repo_url: str, dest_path: Path, github_token: str | None, **_kw: Any
    ) -> str | None:
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


def test_create_session_clone_failure_redacts_token(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """B1: the github_token must never appear in the HTTP response or sess.error."""
    secret = "ghp_supersecrettoken123"

    def fail_clone(repo_url: str, dest_path: Path, github_token: str | None) -> str | None:
        # Simulate git stderr echoing the authenticated URL.
        return (
            f"fatal: could not read Username for "
            f"https://x-access-token:{github_token}@github.com/user/bad"
        )

    monkeypatch.setattr(app_module, "_clone_repo", fail_clone)
    resp = client.post(
        "/api/session",
        json={"repo_url": "https://github.com/user/bad", "github_token": secret},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert secret not in detail
    assert "***" in detail
    # The token must also not leak into the in-memory session error.
    for sess in app_module._sessions.values():
        assert secret not in (sess.error or "")


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


def test_check_scaffold_skill_files_accepts_canonical_layout(tmp_path: Path) -> None:
    skills_dir = tmp_path / "scaffold" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "identity.md").write_text("# Identity\n", encoding="utf-8")

    result = app_module._check_scaffold_skill_files(tmp_path, [{"file": "identity.md"}])

    assert result["status"] == "pass"


def test_check_scaffold_skill_files_accepts_legacy_layout(tmp_path: Path) -> None:
    scaffold_dir = tmp_path / "scaffold"
    scaffold_dir.mkdir()
    (scaffold_dir / "identity.md").write_text("# Identity\n", encoding="utf-8")

    result = app_module._check_scaffold_skill_files(tmp_path, [{"file": "identity.md"}])

    assert result["status"] == "pass"


def test_check_scaffold_skill_files_fails_when_missing(tmp_path: Path) -> None:
    result = app_module._check_scaffold_skill_files(tmp_path, [{"file": "missing.md"}])

    assert result["status"] == "fail"
    assert "scaffold/skills/ (or scaffold/)" in result["message"]


def test_check_golden_set_warns_when_build_script_exists(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "build_golden_set.py").write_text("", encoding="utf-8")

    result = app_module._check_golden_set_jsonl(tmp_path)

    assert result["status"] == "warn"
    assert "build it locally" in result["message"]


def test_check_golden_set_fails_when_data_and_build_script_absent(tmp_path: Path) -> None:
    result = app_module._check_golden_set_jsonl(tmp_path)

    assert result["status"] == "fail"
    assert result["message"] == "data/golden_set.jsonl not found"


def test_check_golden_set_still_validates_existing_jsonl(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    golden_set = data_dir / "golden_set.jsonl"
    golden_set.write_text('{"example_id": "valid", "query": "question"}\n', encoding="utf-8")
    assert app_module._check_golden_set_jsonl(tmp_path)["status"] == "pass"

    golden_set.write_text('{"example_id": "missing-query"}\n', encoding="utf-8")
    result = app_module._check_golden_set_jsonl(tmp_path)
    assert result["status"] == "fail"
    assert "missing required field" in result["message"]


def test_validation_invalid_config_still_has_all_checks(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """W1: when harness/config.yaml is missing, the report still has all 8 checks."""
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_config=False)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    checks = {c["name"]: c for c in data["validation"]["checks"]}
    expected_names = {
        "git_repo",
        "scaffold_harness_yaml",
        "scaffold_skill_files",
        "golden_set_jsonl",
        "harness_config_yaml",
        "eval_modes",
        "agent_code",
        "parent_branch",
    }
    assert set(checks.keys()) == expected_names
    # Dependent checks should be fail with a skip message.
    assert checks["eval_modes"]["status"] == "fail"
    assert "Skipped" in checks["eval_modes"]["message"]
    assert checks["agent_code"]["status"] == "fail"
    assert "Skipped" in checks["agent_code"]["message"]


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


def test_optimize_mlflow_baseline(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """When ``mlflow_experiment_id`` is provided the baseline is seeded from
    MLflow judge results (not local eval) and becomes the round gate."""
    from anvil.eval.cache import CachedBaseline

    sid = _create_valid_session(client, tmp_path, monkeypatch)
    # evaluate_branch must NOT be called when mlflow_experiment_id is set.
    eval_calls: list[dict] = []
    monkeypatch.setattr(
        app_module,
        "evaluate_branch",
        lambda **kw: eval_calls.append(kw) or _fake_eval_report(),
    )
    mlflow_baseline = CachedBaseline(
        scaffold_commit_sha="mlflow-seed",
        evaluated_at="2024-01-01T00:00:00",
        mode="mlflow",
        scorers=["accuracy", "field_amount"],
        runtime_endpoint="",
        judge_endpoint="",
        aggregate=0.92,
        per_judge={"accuracy": 0.92, "field_amount": 0.88},
        per_bucket={},
        n_examples=300,
        mlflow_run_id=None,
        scorer_fingerprint="",
    )
    mlflow_calls: list[dict] = []
    monkeypatch.setattr(
        app_module,
        "build_mlflow_baseline",
        lambda **kw: mlflow_calls.append(kw) or mlflow_baseline,
    )
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    resp = client.post(
        f"/api/session/{sid}/optimize",
        json={"max_rounds": 2, "max_turns": 5, "mlflow_experiment_id": "967014443183055"},
    )
    assert resp.status_code == 202
    data = _wait_for_status(client, sid, {"optimizing", "optimized"})
    assert data["status"] in {"optimizing", "optimized"}
    # build_mlflow_baseline was called with the experiment ID.
    assert len(mlflow_calls) == 1
    assert mlflow_calls[0]["experiment_id"] == "967014443183055"
    # Local eval was NOT invoked.
    assert eval_calls == []
    # MLflow baseline persisted to the session.
    assert data["baseline"] is not None
    assert data["baseline"]["mode"] == "mlflow"
    assert data["baseline"]["aggregate"] == 0.92


def test_build_baseline_mlflow_resets_stale_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKING 1: building an MLflow baseline deletes a stale frontier.json
    so the gate re-seeds from the new baseline on the next scored round
    (a stale frontier would otherwise silently bypass the MLflow baseline)."""
    from anvil.eval.cache import CachedBaseline

    repo = tmp_path / "agent-repo"
    _seed_repo(repo)
    runs = repo / "eval" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    # Seed a stale frontier from a prior optimization.
    (runs / "frontier.json").write_text(
        json.dumps(
            {
                "best": {"aggregate": 0.5, "correctness": 0.4},
                "objectives": ["aggregate", "correctness"],
                "pareto": False,
                "directions": {},
                "sources": {},
                "epsilon": 0.0,
            }
        ),
        encoding="utf-8",
    )
    mlflow_baseline = CachedBaseline(
        scaffold_commit_sha="mlflow-seed",
        evaluated_at="2024-01-01T00:00:00",
        mode="mlflow",
        scorers=["accuracy"],
        runtime_endpoint="",
        judge_endpoint="",
        aggregate=0.92,
        per_judge={"accuracy": 0.92},
        per_bucket={},
        n_examples=300,
        mlflow_run_id=None,
        scorer_fingerprint="",
    )
    monkeypatch.setattr(app_module, "build_mlflow_baseline", lambda **kw: mlflow_baseline)
    result = app_module._build_baseline_sync(repo, None, "967014443183055")
    # The stale frontier was deleted so the gate will re-seed from the baseline.
    assert not (runs / "frontier.json").exists()
    assert result["mode"] == "mlflow"
    assert result["aggregate"] == 0.92


def test_build_baseline_local_eval_resets_stale_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKING 1: building a local-eval baseline also deletes a stale
    frontier.json — the fix applies to both baseline paths."""
    repo = tmp_path / "agent-repo"
    _seed_repo(repo)
    runs = repo / "eval" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "frontier.json").write_text(
        json.dumps({"best": {"aggregate": 0.5}, "objectives": ["aggregate"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    result = app_module._build_baseline_sync(repo, "quick", None)
    assert not (runs / "frontier.json").exists()
    assert result["mode"] == "quick"


def test_build_baseline_reset_frontier_safe_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frontier reset is a no-op when no frontier.json exists (no crash)."""
    from anvil.eval.cache import CachedBaseline

    repo = tmp_path / "agent-repo"
    _seed_repo(repo)
    runs = repo / "eval" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    assert not (runs / "frontier.json").exists()
    mlflow_baseline = CachedBaseline(
        scaffold_commit_sha="mlflow-seed",
        evaluated_at="2024-01-01T00:00:00",
        mode="mlflow",
        scorers=["accuracy"],
        runtime_endpoint="",
        judge_endpoint="",
        aggregate=0.9,
        per_judge={"accuracy": 0.9},
        per_bucket={},
        n_examples=10,
        mlflow_run_id=None,
        scorer_fingerprint="",
    )
    monkeypatch.setattr(app_module, "build_mlflow_baseline", lambda **kw: mlflow_baseline)
    result = app_module._build_baseline_sync(repo, None, "123")
    # No crash; baseline still built; no frontier left behind.
    assert result["mode"] == "mlflow"
    assert not (runs / "frontier.json").exists()


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


def test_optimize_invalid_max_rounds(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """W3: max_rounds=0 is rejected with 422 validation error."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 0, "max_turns": 5})
    assert resp.status_code == 422


def test_optimize_invalid_max_turns(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """W3: max_turns=0 is rejected with 422 validation error."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 5, "max_turns": 0})
    assert resp.status_code == 422


def test_optimize_parent_branch_failure(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """B7: if _ensure_parent_branch fails the session goes to 'error', not stuck."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)

    def boom_branch(repo_root: Path) -> None:
        raise RuntimeError("cannot create parent branch")

    monkeypatch.setattr(app_module, "_ensure_parent_branch", boom_branch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1, "max_turns": 5})
    data = _wait_for_status(client, sid, {"error"})
    assert data["status"] == "error"
    assert "parent branch" in data["error"]


def test_optimize_parent_branch_git_error(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """W2: nonzero git exit code in _ensure_parent_branch raises (→ error status)."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)

    # Make `git checkout -b` fail by redirecting to a bad git path.
    real_run = subprocess.run

    def fake_run(args, **kw):
        if "checkout" in args and "-b" in args:
            return subprocess.CompletedProcess(
                args, 128, stdout="", stderr="fatal: already exists\n"
            )
        return real_run(args, **kw)

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    # Remove the branch so _ensure_parent_branch actually tries to create it.
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1, "max_turns": 5})
    data = _wait_for_status(client, sid, {"error"})
    assert data["status"] == "error"


def test_run_round_failure(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """B6: if run_round raises, the session transitions to 'error'."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())

    def boom_round(**kw: Any) -> RoundReport:
        raise RuntimeError("round exploded")

    monkeypatch.setattr(app_module, "run_round", boom_round)
    client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1, "max_turns": 5})
    data = _wait_for_status(client, sid, {"error"})
    assert data["status"] == "error"
    assert "round exploded" in data["error"]


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


def test_finalize_already_finalized_on_disk(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """B2: if finalized.json exists on disk, finalize returns 409 even if
    the in-memory status is still 'optimized'."""
    sid = _optimize_then_finalize_setup(client, tmp_path, monkeypatch, sessions_root)
    sess = app_module._sessions[sid]
    # Write finalized.json to disk but leave in-memory status as 'optimized'.
    runs_dir = sess.repo_path / "eval" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "finalized.json").write_text(
        json.dumps({"aggregate": 0.99, "finalized_at": "2024-01-01T00:00:00"}), encoding="utf-8"
    )
    assert sess.status == "optimized"
    resp = client.post(f"/api/session/{sid}/finalize")
    assert resp.status_code == 409


def test_finalize_while_optimizing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """B3: finalize is rejected with 409 while optimization is running."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    app_module._sessions[sid].status = "optimizing"
    resp = client.post(f"/api/session/{sid}/finalize")
    assert resp.status_code == 409


def test_finalize_concurrent(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """B2: two concurrent finalize calls — exactly one succeeds (200), the other 409."""
    sid = _optimize_then_finalize_setup(client, tmp_path, monkeypatch, sessions_root)
    # Make _finalize_sync slow so both requests overlap in the 'finalizing' state.
    original_finalize = app_module._finalize_sync

    def slow_finalize(repo_path: Path) -> dict[str, Any]:
        time.sleep(0.3)
        return original_finalize(repo_path)

    monkeypatch.setattr(app_module, "_finalize_sync", slow_finalize)

    async def _run_concurrent() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            r1, r2 = await asyncio.gather(
                ac.post(f"/api/session/{sid}/finalize"),
                ac.post(f"/api/session/{sid}/finalize"),
            )
            return r1, r2

    r1, r2 = asyncio.run(_run_concurrent())
    statuses = sorted([r1.status_code, r2.status_code])
    assert statuses == [200, 409]


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
    """Two sessions optimize concurrently without interfering (B6: asyncio.gather)."""
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

    # B6: fire both /optimize requests concurrently via threading so the
    # two background tasks run on the TestClient's event loop (which
    # stays alive for polling).
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [
            ex.submit(
                client.post,
                f"/api/session/{sid_a}/optimize",
                json={"max_rounds": 2, "max_turns": 5},
            ),
            ex.submit(
                client.post,
                f"/api/session/{sid_b}/optimize",
                json={"max_rounds": 3, "max_turns": 5},
            ),
        ]
        results = [f.result(timeout=30) for f in futs]
    for r in results:
        assert r.status_code == 202

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


# ---------------------------------------------------------------------------
# Lifespan / shutdown (B8)
# ---------------------------------------------------------------------------


def test_lifespan_cancels_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """B8: lifespan shutdown cancels active optimization tasks and cleans up."""
    source = tmp_path / "agent-repo"
    _seed_repo(source)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    monkeypatch.setattr(app_module, "evaluate_branch", lambda **kw: _fake_eval_report())
    monkeypatch.setattr(app_module, "run_round", _make_mock_run_round())
    # Make the baseline slow so the task is still running at shutdown.
    # The function returns a dummy dict (not calling evaluate_branch) so
    # the orphaned background thread has no side effects after the test
    # ends and monkeypatch reverts.  anyio.to_thread.run_sync defaults to
    # cancellable=False, so the lifespan shutdown waits for the thread to
    # finish — keep the sleep short.
    def slow_build(repo_path: Path, eval_mode: str | None) -> dict[str, Any]:
        time.sleep(0.2)
        return {"aggregate": 0.0, "scaffold_commit_sha": "unknown", "mode": "quick"}

    monkeypatch.setattr(app_module, "_build_baseline_sync", slow_build)

    task: asyncio.Task | None = None
    repo_path: Path | None = None
    with TestClient(app) as client:
        resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
        sid = resp.json()["session_id"]
        client.post(f"/api/session/{sid}/optimize", json={"max_rounds": 1, "max_turns": 5})
        task = app_module._sessions[sid]._optimize_task
        repo_path = app_module._sessions[sid].repo_path
        assert task is not None
        assert not task.done()
    # After the context manager exits, lifespan shutdown has cancelled
    # the task and cleaned up the repo directory.
    assert task is not None
    assert task.done()
    assert repo_path is not None
    assert not repo_path.exists()


# ---------------------------------------------------------------------------
# URL parsing (subdirectory support)
# ---------------------------------------------------------------------------


def test_parse_github_url_plain() -> None:
    """Plain repo URL → no branch, no subpath."""
    clone_url, branch, subpath = app_module._parse_github_url(
        "https://github.com/mbihani/savesage"
    )
    assert clone_url == "https://github.com/mbihani/savesage"
    assert branch is None
    assert subpath is None


def test_parse_github_url_with_branch() -> None:
    """URL with /tree/main → branch=main, no subpath."""
    clone_url, branch, subpath = app_module._parse_github_url(
        "https://github.com/mbihani/savesage/tree/main"
    )
    assert clone_url == "https://github.com/mbihani/savesage"
    assert branch == "main"
    assert subpath is None


def test_parse_github_url_with_subpath() -> None:
    """URL with /tree/main/statement-agent → branch=main, subpath=statement-agent."""
    clone_url, branch, subpath = app_module._parse_github_url(
        "https://github.com/mbihani/savesage/tree/main/statement-agent"
    )
    assert clone_url == "https://github.com/mbihani/savesage"
    assert branch == "main"
    assert subpath == "statement-agent"


def test_parse_github_url_deep_subpath() -> None:
    """URL with /tree/feature/foo/bar → branch=feature/foo, subpath=bar.

    Branches can contain slashes; the last path segment is the subpath.
    """
    clone_url, branch, subpath = app_module._parse_github_url(
        "https://github.com/mbihani/savesage/tree/feature/foo/bar"
    )
    assert clone_url == "https://github.com/mbihani/savesage"
    assert branch == "feature/foo"
    assert subpath == "bar"


def test_parse_github_url_dot_git() -> None:
    """URL with .git suffix → stripped."""
    clone_url, branch, subpath = app_module._parse_github_url(
        "https://github.com/mbihani/savesage.git"
    )
    assert clone_url == "https://github.com/mbihani/savesage"
    assert branch is None
    assert subpath is None


def test_parse_github_url_trailing_slash() -> None:
    """URL with trailing slash → handled."""
    clone_url, branch, subpath = app_module._parse_github_url(
        "https://github.com/mbihani/savesage/"
    )
    assert clone_url == "https://github.com/mbihani/savesage"
    assert branch is None
    assert subpath is None


def test_parse_non_github_url() -> None:
    """Non-GitHub URL → returned as-is, no branch/subpath."""
    clone_url, branch, subpath = app_module._parse_github_url(
        "https://gitlab.com/user/repo"
    )
    assert clone_url == "https://gitlab.com/user/repo"
    assert branch is None
    assert subpath is None


# ---------------------------------------------------------------------------
# Subdirectory session tests
# ---------------------------------------------------------------------------


def test_create_session_with_subpath(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """Create a session with a subdirectory URL; validation runs against the subdir."""
    source = tmp_path / "agent-repo"
    source.mkdir(parents=True)
    # Seed a valid agent structure inside a subdirectory.
    subdir = source / "statement-agent"
    _seed_repo(subdir)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post(
        "/api/session",
        json={"repo_url": "https://github.com/user/repo/tree/main/statement-agent"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "validated", data
    assert data["validation"]["status"] == "valid"
    assert data["agent_subpath"] == "statement-agent"
    # The session's repo_path points at the subdirectory, not the clone root.
    sess = app_module._sessions[data["session_id"]]
    assert sess.repo_path == sessions_root / data["session_id"] / "statement-agent"
    assert sess._clone_root == sessions_root / data["session_id"]


def test_create_session_subpath_not_found(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """Subdirectory doesn't exist in the cloned repo → error."""
    source = tmp_path / "agent-repo"
    _seed_repo(source)  # valid structure at root, no subdirectory
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post(
        "/api/session",
        json={"repo_url": "https://github.com/user/repo/tree/main/nonexistent"},
    )
    assert resp.status_code == 400
    assert "nonexistent" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Smart remediation
# ---------------------------------------------------------------------------


def test_scan_agent_root(tmp_path: Path) -> None:
    """_scan_agent_root detects all alternative structure categories."""
    root = tmp_path / "agent"
    root.mkdir()
    # prompts/ with .txt files
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "icici.txt").write_text("rules", encoding="utf-8")
    (prompts / "hdfc.txt").write_text("rules", encoding="utf-8")
    # schema/ with .json files
    schema = root / "schema"
    schema.mkdir()
    (schema / "icici.json").write_text("{}", encoding="utf-8")
    # skills/ with .py files
    skills = root / "skills"
    skills.mkdir()
    (skills / "extract.py").write_text("# extract", encoding="utf-8")
    # judge/ with evaluator.py
    judge = root / "judge"
    judge.mkdir()
    (judge / "evaluator.py").write_text("# eval", encoding="utf-8")
    # tests/ directory
    (root / "tests").mkdir()
    # harness/ with .py but no config.yaml
    harness = root / "harness"
    harness.mkdir()
    (harness / "config_ws4.py").write_text("# config", encoding="utf-8")
    # config.py at root
    (root / "config.py").write_text("# config", encoding="utf-8")
    # data/ directory
    (root / "data").mkdir()

    findings = app_module._scan_agent_root(root)
    assert findings["prompts"] == ["hdfc.txt", "icici.txt"]
    assert findings["schemas"] == ["icici.json"]
    assert findings["python_skills"] == ["extract.py"]
    assert findings["judge"] == ["evaluator.py"]
    assert findings["tests"] == ["tests/ directory found"]
    assert findings["harness_py"] == ["config_ws4.py"]
    assert findings["config_py"] == ["config.py"]
    assert findings["data_dir"] == ["data/ directory found"]


def test_smart_remediation_prompts_found(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """scaffold/harness.yaml missing but prompts/ has .txt files → remediation
    mentions the specific prompt files found."""
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_scaffold=False)
    prompts_dir = source / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "icici.txt").write_text("rules", encoding="utf-8")
    (prompts_dir / "hdfc.txt").write_text("rules", encoding="utf-8")
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    checks = {c["name"]: c for c in data["validation"]["checks"]}
    assert checks["scaffold_harness_yaml"]["status"] == "fail"
    remediation = checks["scaffold_harness_yaml"]["remediation"]
    assert "icici.txt" in remediation
    assert "hdfc.txt" in remediation
    # Smart remediation is prepended — static remediation still present.
    assert "scaffold/harness.yaml" in remediation


def test_smart_remediation_schemas_found(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """golden_set.jsonl missing but schema/ has .json files → remediation
    mentions the schema files."""
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_golden=False)
    schema_dir = source / "schema"
    schema_dir.mkdir()
    (schema_dir / "icici.json").write_text("{}", encoding="utf-8")
    (schema_dir / "hdfc.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    checks = {c["name"]: c for c in data["validation"]["checks"]}
    assert checks["golden_set_jsonl"]["status"] == "fail"
    remediation = checks["golden_set_jsonl"]["remediation"]
    assert "icici.json" in remediation
    assert "hdfc.json" in remediation
    # Static remediation still present.
    assert "golden_set.jsonl" in remediation


def test_smart_remediation_no_findings(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """No alternative structure found → keep existing static remediation only."""
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_scaffold=False)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    checks = {c["name"]: c for c in data["validation"]["checks"]}
    assert checks["scaffold_harness_yaml"]["status"] == "fail"
    remediation = checks["scaffold_harness_yaml"]["remediation"]
    # No smart remediation — only static text.
    assert "prompts/" not in remediation
    assert "scaffold/harness.yaml" in remediation


def test_smart_remediation_harness_py_found(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """harness/config.yaml missing but harness/ has .py files → remediation
    mentions them."""
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_config=False)
    harness_dir = source / "harness"
    harness_dir.mkdir(exist_ok=True)
    (harness_dir / "config_ws4.py").write_text("config = {}", encoding="utf-8")
    (harness_dir / "auth.py").write_text("auth = {}", encoding="utf-8")
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    checks = {c["name"]: c for c in data["validation"]["checks"]}
    assert checks["harness_config_yaml"]["status"] == "fail"
    remediation = checks["harness_config_yaml"]["remediation"]
    assert "config_ws4.py" in remediation
    assert "auth.py" in remediation
    # Static remediation still present.
    assert "harness/config.yaml" in remediation


def test_session_response_includes_agent_subpath(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """GET /api/session/{id} includes agent_subpath in the response."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}")
    data = resp.json()
    assert "agent_subpath" in data
    assert data["agent_subpath"] is None  # plain URL → no subpath


# ---------------------------------------------------------------------------
# Auto-conversion (POST/GET /api/session/{id}/convert)
#
# The real conversion drives an Omnigent agent over HTTP; tests mock
# ``_run_conversion_task`` so no live server is contacted. The convert endpoint
# is only reachable for sessions whose validation found savesage-style
# alternative structures (``convertible: true``).
# ---------------------------------------------------------------------------


def _create_convertible_session(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Seed an INVALID but convertible repo (no scaffold, but prompts/*.txt
    present) and POST /api/session. Returns the session_id."""
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_scaffold=False)
    prompts_dir = source / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "icici.txt").write_text("rules", encoding="utf-8")
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "invalid", data
    assert data["validation"]["convertible"] is True
    return data["session_id"]


async def _fake_conversion_task(session_id: str, target_branch: str) -> None:
    """Test double for ``_run_conversion_task``: marks the conversion completed
    with a synthetic re-validation report. No Omnigent/git contact.

    Sets ``session_id`` / ``session_url`` on the conversion result to mirror
    the real task's ``on_session_created`` callback so the polling response
    surfaces the persisted managed-session link.
    """
    with app_module._session_lock:
        sess = app_module._sessions.get(session_id)
        if sess is not None and sess.conversion is not None:
            sess.conversion.status = "completed"
            sess.conversion.branch_name = target_branch
            sess.conversion.pr_url = (
                f"https://github.com/user/repo/compare/main...{target_branch}"
            )
            sess.conversion.session_id = "omnigent-sess-fake"
            sess.conversion.session_url = (
                "http://localhost:6767/sessions/omnigent-sess-fake"
            )
            sess.conversion.revalidation = {
                "status": "valid",
                "checks": [
                    {"name": "scaffold_harness_yaml", "status": "pass", "message": "valid"},
                ],
                "convertible": False,
                "pii_findings": [],
            }


async def _blocking_conversion_task(session_id: str, target_branch: str) -> None:
    """Test double that stays "running" forever (cancelled on shutdown) so a
    second POST /convert sees an in-progress conversion."""
    with app_module._session_lock:
        sess = app_module._sessions.get(session_id)
        if sess is not None and sess.conversion is not None:
            sess.conversion.status = "running"
    await asyncio.Event().wait()  # block until cancelled


def _wait_for_convert(
    client: TestClient, session_id: str, statuses: set[str], timeout: float = 10.0
) -> dict[str, Any]:
    """Poll GET /api/session/{id}/convert until status is in ``statuses``."""
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        resp = client.get(f"/api/session/{session_id}/convert")
        if resp.status_code == 200:
            last = resp.json()
            if last.get("status") in statuses:
                return last
        time.sleep(0.02)
    raise AssertionError(
        f"timed out waiting for convert status in {statuses}; last={last}"
    )


def test_validation_convertible_true_when_alternative_structures_found(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """A repo that fails validation but has prompts/*.txt reports
    ``convertible: true`` in the validation response."""
    sid = _create_convertible_session(client, tmp_path, monkeypatch)
    # Re-fetch the validation to confirm the flag round-trips through GET.
    resp = client.get(f"/api/session/{sid}/validation")
    assert resp.json()["convertible"] is True


def test_validation_convertible_false_for_valid_repo(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """A valid repo has no alternative-structure findings → convertible: false."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}/validation")
    assert resp.json()["convertible"] is False


def test_validation_convertible_false_when_no_alternative_structures(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """An invalid repo with NO recognizable alternative structure (just a broken
    scaffold) is not convertible — the user gets static remediation only."""
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_scaffold=False)  # no scaffold AND no prompts/
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    data = resp.json()
    assert data["status"] == "invalid"
    assert data["validation"]["convertible"] is False


def test_convert_returns_503_when_omnigent_not_configured(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_convertible_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "OMNIGENT_SERVER_URL", None)
    resp = client.post(f"/api/session/{sid}/convert", json={})
    assert resp.status_code == 503
    assert "OMNIGENT_SERVER_URL" in resp.json()["detail"]


def test_convert_starts_task_and_polls_to_completed(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_convertible_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "OMNIGENT_SERVER_URL", "http://localhost:6767")
    monkeypatch.setattr(app_module, "_run_conversion_task", _fake_conversion_task)
    resp = client.post(f"/api/session/{sid}/convert", json={})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending"
    assert body["branch_name"] == "forge-compat"
    data = _wait_for_convert(client, sid, {"completed"})
    assert data["status"] == "completed"
    assert data["branch_name"] == "forge-compat"
    assert data["pr_url"] == "https://github.com/user/repo/compare/main...forge-compat"
    assert data["revalidation"]["status"] == "valid"
    assert data["revalidation"]["pii_findings"] == []
    # The polling response surfaces the persisted managed-session link.
    assert data["session_id"] == "omnigent-sess-fake"
    assert data["session_url"] == "http://localhost:6767/sessions/omnigent-sess-fake"


def test_convert_custom_target_branch(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_convertible_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "OMNIGENT_SERVER_URL", "http://localhost:6767")
    monkeypatch.setattr(app_module, "_run_conversion_task", _fake_conversion_task)
    resp = client.post(f"/api/session/{sid}/convert", json={"target_branch": "my-branch"})
    assert resp.status_code == 202
    data = _wait_for_convert(client, sid, {"completed"})
    assert data["branch_name"] == "my-branch"
    assert "my-branch" in data["pr_url"]


def test_convert_not_convertible_returns_409(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """A valid repo (convertible: false) cannot be converted → 409."""
    sid = _create_valid_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "OMNIGENT_SERVER_URL", "http://localhost:6767")
    resp = client.post(f"/api/session/{sid}/convert", json={})
    assert resp.status_code == 409
    assert "not convertible" in resp.json()["detail"]


def test_convert_already_running_returns_409(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_convertible_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "OMNIGENT_SERVER_URL", "http://localhost:6767")
    monkeypatch.setattr(app_module, "_run_conversion_task", _blocking_conversion_task)
    resp1 = client.post(f"/api/session/{sid}/convert", json={})
    assert resp1.status_code == 202
    # A second POST while the first is still running → 409.
    resp2 = client.post(f"/api/session/{sid}/convert", json={})
    assert resp2.status_code == 409
    assert "already running" in resp2.json()["detail"]


def test_get_convert_404_when_not_started(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    sid = _create_convertible_session(client, tmp_path, monkeypatch)
    resp = client.get(f"/api/session/{sid}/convert")
    assert resp.status_code == 404


def test_get_convert_returns_progress(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """GET /convert surfaces the progress entries the task appended."""
    sid = _create_convertible_session(client, tmp_path, monkeypatch)
    monkeypatch.setattr(app_module, "OMNIGENT_SERVER_URL", "http://localhost:6767")

    async def _task_with_progress(session_id: str, target_branch: str) -> None:
        with app_module._session_lock:
            sess = app_module._sessions.get(session_id)
            if sess is not None and sess.conversion is not None:
                sess.conversion.status = "running"
                sess.conversion.progress.append(
                    {"step": "agent_running", "message": "Converting…", "timestamp": "t"}
                )

    monkeypatch.setattr(app_module, "_run_conversion_task", _task_with_progress)
    client.post(f"/api/session/{sid}/convert", json={})
    data = _wait_for_convert(client, sid, {"running"})
    assert data["status"] == "running"
    assert len(data["progress"]) >= 1
    assert data["progress"][0]["step"] == "agent_running"


def test_convert_preserves_github_token_for_agent(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """The user's GitHub token from Step 1 is kept on the session (in memory
    only) so the converter agent can clone+push a private repo — it is NEVER
    returned in any API response."""
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_scaffold=False)
    (source / "prompts").mkdir()
    (source / "prompts" / "icici.txt").write_text("rules", encoding="utf-8")
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post(
        "/api/session",
        json={"repo_url": "https://github.com/user/repo", "github_token": "ghp_secret"},
    )
    sid = resp.json()["session_id"]
    # The token is stored on the session for the converter.
    assert app_module._sessions[sid]._github_token == "ghp_secret"
    # But no API response leaks it.
    for path in (f"/api/session/{sid}", f"/api/session/{sid}/config"):
        body = client.get(path).json()
        assert "ghp_secret" not in json.dumps(body)


def test_dashboard_has_convert_ui(client: TestClient) -> None:
    """The dashboard ships the Convert button + polling functions (XSS-safe)."""
    html = client.get("/").text
    assert "Convert to forge-compatible" in html
    assert "startConvert" in html
    assert "pollConvert" in html
    assert "renderConvert" in html
    assert "conversion-panel" in html
    # Still XSS-safe — no innerHTML added by the conversion UI.
    assert "innerHTML" not in html


def test_dashboard_convert_button_gated_on_validation_convertible(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """Regression: the Convert button must read ``convertible`` from the nested
    ``validation`` object, not the top-level POST /api/session response.

    POST /api/session returns ``convertible`` nested inside ``validation``:
    ``{"validation": {"convertible": true, ...}}``. The dashboard JS previously
    read ``data.convertible`` (always ``undefined``), so the button never
    rendered even for convertible repos. The conditional now reads
    ``data.validation.convertible`` so the button appears when the API reports
    ``convertible: true`` and is hidden when ``convertible: false``.
    """
    # The served dashboard must gate the button on the *nested* path. The buggy
    # ``data.convertible`` is not a substring of the fix
    # (``data.validation.convertible`` — "validation" follows "data.", not
    # "convertible"), so the negative assertion is meaningful.
    html = client.get("/").text
    assert "data.validation.convertible" in html
    assert "data.convertible" not in html

    # Convertible repo (invalid but with prompts/*.txt): the flag the corrected
    # JS reads — data.validation.convertible — is True, and the top-level
    # ``convertible`` the old JS read is absent. → button appears.
    source = tmp_path / "agent-repo"
    _seed_repo(source, with_scaffold=False)
    (source / "prompts").mkdir()
    (source / "prompts" / "icici.txt").write_text("rules", encoding="utf-8")
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(source))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "invalid"
    assert data["validation"]["convertible"] is True  # button shows
    assert "convertible" not in data  # no top-level field (the old bug)

    # Valid repo: data.validation.convertible is False → button hidden.
    valid = tmp_path / "valid-repo"
    _seed_repo(valid)
    monkeypatch.setattr(app_module, "_clone_repo", _mock_clone_factory(valid))
    resp = client.post("/api/session", json={"repo_url": "https://github.com/user/repo"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "validated"
    assert data["validation"]["convertible"] is False  # button hidden
    assert "convertible" not in data


# ---------------------------------------------------------------------------
# Omnigent managed-session cleanup — _cleanup_omnigent_session + lifespan
# ---------------------------------------------------------------------------


def _seed_session_with_conversion(
    session_id: str, tmp_path: Path, *, omnigent_sid: str | None = "omnigent-sess-X"
) -> None:
    """Seed a forge session with a conversion result carrying a managed
    Omnigent session id (or None when ``omnigent_sid`` is None)."""
    from anvil.orchestrator.app import SessionData
    from anvil.orchestrator.conversion import ConversionResult

    sess = SessionData(
        session_id=session_id,
        repo_url="https://github.com/user/repo",
        repo_path=tmp_path / session_id / "clone",
        status="completed",
        validation={"status": "valid", "checks": [], "convertible": False},
        config=None,
        baseline=None,
        rounds=[],
        frontier=None,
        finalized=None,
        error=None,
        conversion=ConversionResult(session_id=omnigent_sid),
    )
    app_module._sessions[session_id] = sess


class _FakeOmnigentCleanupClient:
    """Fake OmnigentClient for cleanup tests — records delete_session calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.deleted: list[str] = []

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        self.deleted.append(session_id)
        return {"deleted": True}

    async def aclose(self) -> None:
        pass


def test_cleanup_omnigent_session_deletes_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """``_cleanup_omnigent_session`` deletes the persisted Omnigent managed
    conversation session when the conversion result carries a ``session_id``."""
    monkeypatch.setenv("OMNIGENT_SERVER_URL", "http://localhost:6767")
    _seed_session_with_conversion("s1", tmp_path)

    fake = _FakeOmnigentCleanupClient()
    monkeypatch.setattr(
        "anvil.optimizer.omnigent_client.OmnigentClient",
        lambda *a, **kw: fake,
    )

    asyncio.run(app_module._cleanup_omnigent_session("s1"))
    assert fake.deleted == ["omnigent-sess-X"]


def test_cleanup_omnigent_session_skips_when_no_session_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """When the conversion result has no ``session_id`` (no managed session was
    created — e.g. the conversion failed early), cleanup is a no-op."""
    monkeypatch.setenv("OMNIGENT_SERVER_URL", "http://localhost:6767")
    _seed_session_with_conversion("s2", tmp_path, omnigent_sid=None)

    fake = _FakeOmnigentCleanupClient()
    monkeypatch.setattr(
        "anvil.optimizer.omnigent_client.OmnigentClient",
        lambda *a, **kw: fake,
    )

    asyncio.run(app_module._cleanup_omnigent_session("s2"))
    assert fake.deleted == []


def test_cleanup_omnigent_session_skips_when_server_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """When OMNIGENT_SERVER_URL is not set, cleanup is a no-op (no server to
    call). The remote session is simply abandoned."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    _seed_session_with_conversion("s3", tmp_path)

    fake = _FakeOmnigentCleanupClient()
    monkeypatch.setattr(
        "anvil.optimizer.omnigent_client.OmnigentClient",
        lambda *a, **kw: fake,
    )

    asyncio.run(app_module._cleanup_omnigent_session("s3"))
    assert fake.deleted == []


def test_cleanup_omnigent_session_swallows_delete_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """A failed delete_session (server already gone) is swallowed — cleanup
    is best-effort and must not raise."""

    class _ExplodingClient(_FakeOmnigentCleanupClient):
        async def delete_session(self, session_id: str) -> dict[str, Any]:
            from anvil.optimizer.omnigent_client import OmnigentError

            raise OmnigentError("already gone", status_code=404, body="not found")

    monkeypatch.setenv("OMNIGENT_SERVER_URL", "http://localhost:6767")
    _seed_session_with_conversion("s4", tmp_path)
    monkeypatch.setattr(
        "anvil.optimizer.omnigent_client.OmnigentClient",
        lambda *a, **kw: _ExplodingClient(),
    )

    # Must not raise — the OmnigentError is suppressed inside cleanup.
    asyncio.run(app_module._cleanup_omnigent_session("s4"))


def test_shutdown_deletes_persisted_managed_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions_root: Path
) -> None:
    """When the app shuts down (lifespan teardown), persisted Omnigent managed
    sessions are deleted so they don't leak indefinitely.

    Uses a standalone ``TestClient(app)`` context so the lifespan teardown
    runs inside the test body (the ``with`` block exit) and the assertion can
    follow it.
    """
    monkeypatch.setenv("OMNIGENT_SERVER_URL", "http://localhost:6767")
    _seed_session_with_conversion("s5", tmp_path)

    fake = _FakeOmnigentCleanupClient()
    monkeypatch.setattr(
        "anvil.optimizer.omnigent_client.OmnigentClient",
        lambda *a, **kw: fake,
    )

    with TestClient(app):
        pass  # startup + immediate exit triggers the lifespan teardown

    assert fake.deleted == ["omnigent-sess-X"]
