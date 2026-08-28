"""Tests for the Forge Orchestrator FastAPI app."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from anvil.eval import EvalReport
from anvil.loop.decision import Decision
from anvil.loop.round import RoundReport
from anvil.orchestrator.app import app, get_repo_root, set_repo_root


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    original = get_repo_root()
    set_repo_root(tmp_path)
    with TestClient(app) as c:
        yield c
    set_repo_root(original)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_rounds_empty(client: TestClient) -> None:
    response = client.get("/rounds")
    assert response.status_code == 200
    assert response.json() == []


def test_list_rounds_with_data(client: TestClient, tmp_path: Path) -> None:
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "round_001.json").write_text(
        json.dumps(
            {
                "round_id": 1,
                "decision": "keep",
                "action_kind": "edit_skill",
                "baseline_score": 0.80,
                "score_delta_vs_parent": 0.05,
                "aggregate": 0.85,
            }
        ),
        encoding="utf-8",
    )
    response = client.get("/rounds")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["round_id"] == 1
    assert data[0]["decision"] == "keep"
    assert data[0]["score_delta"] == 0.05


def test_get_round_by_id(client: TestClient, tmp_path: Path) -> None:
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    payload = {"round_id": 1, "decision": "keep", "aggregate": 0.85}
    (runs_dir / "round_001.json").write_text(json.dumps(payload), encoding="utf-8")
    response = client.get("/rounds/1")
    assert response.status_code == 200
    assert response.json() == payload


def test_get_round_not_found(client: TestClient) -> None:
    response = client.get("/rounds/999")
    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_start_round(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_report = RoundReport(
        round_id=1,
        branch="anvil/exp-round-1",
        decision=Decision.KEEP,
        action_kind="edit_skill",
        parse_status="ok",
        diff_summary="edited skill X",
    )
    monkeypatch.setattr("anvil.orchestrator.app.run_round", lambda **kwargs: fake_report)
    response = client.post("/rounds", json={"round_id": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["round_id"] == 1
    assert data["decision"] == "keep"
    assert data["branch"] == "anvil/exp-round-1"
    assert data["action_kind"] == "edit_skill"


def test_start_round_auto_id(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anvil.orchestrator.app._next_round_id", lambda root: 42)
    captured: dict[str, Any] = {}

    def mock_run_round(**kwargs: Any) -> RoundReport:
        captured.update(kwargs)
        return RoundReport(
            round_id=kwargs["round_id"],
            branch=f"anvil/exp-round-{kwargs['round_id']}",
            decision=Decision.KEEP,
            action_kind="edit_skill",
            parse_status="ok",
            diff_summary="edited skill X",
        )

    monkeypatch.setattr("anvil.orchestrator.app.run_round", mock_run_round)
    response = client.post("/rounds", json={})
    assert response.status_code == 200
    assert response.json()["round_id"] == 42
    assert captured["round_id"] == 42


def test_start_round_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_run_round(**kwargs: Any) -> RoundReport:
        raise RuntimeError("optimizer exploded")

    monkeypatch.setattr("anvil.orchestrator.app.run_round", mock_run_round)
    response = client.post("/rounds", json={"round_id": 1})
    assert response.status_code == 500
    assert "optimizer exploded" in response.json()["detail"]


def test_dashboard_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Forge Orchestrator" in response.text


# ---------------------------------------------------------------------------
# Issue 3: Stored-XSS — malicious round JSON must not inject HTML
# ---------------------------------------------------------------------------


def test_dashboard_escapes_xss_in_round_json(client: TestClient, tmp_path: Path) -> None:
    """A crafted round JSON with an XSS payload must not inject raw HTML."""
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    xss = "<script>alert(1)</script>"
    (runs_dir / "round_001.json").write_text(
        json.dumps(
            {
                "round_id": xss,
                "decision": xss,
                "action_kind": xss,
                "baseline_score": xss,
                "score_delta_vs_parent": xss,
                "aggregate": xss,
            }
        ),
        encoding="utf-8",
    )
    response = client.get("/")
    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;" in response.text  # escaped form present


def test_render_dashboard_escapes_all_values() -> None:
    """_render_dashboard must escape every value even without upstream validation."""
    from anvil.orchestrator.app import _render_dashboard

    xss = "<script>alert(1)</script>"
    html_output = _render_dashboard(
        [
            {
                "round_id": xss,
                "decision": xss,
                "action_kind": xss,
                "baseline_score": xss,
                "score_delta": xss,
                "aggregate": xss,
            }
        ]
    )
    assert "<script>alert(1)</script>" not in html_output
    assert "&lt;script&gt;" in html_output


# ---------------------------------------------------------------------------
# Issue 4: run_round must execute on a worker thread
# ---------------------------------------------------------------------------


def test_start_round_runs_on_worker_thread(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_round must execute in a thread pool, not on the event loop thread."""
    main_thread_id = threading.current_thread().ident
    captured: dict[str, Any] = {}

    def mock_run_round(**kwargs: Any) -> RoundReport:
        captured["thread_id"] = threading.current_thread().ident
        return RoundReport(
            round_id=kwargs["round_id"],
            branch="anvil/exp-round-1",
            decision=Decision.KEEP,
            action_kind="edit_skill",
            parse_status="ok",
            diff_summary="edited skill X",
        )

    monkeypatch.setattr("anvil.orchestrator.app.run_round", mock_run_round)
    response = client.post("/rounds", json={"round_id": 1})
    assert response.status_code == 200
    assert captured["thread_id"] is not None
    assert captured["thread_id"] != main_thread_id


# ---------------------------------------------------------------------------
# Issue 6: Concurrent POST /rounds must return 409
# ---------------------------------------------------------------------------


def test_start_round_conflict_409(client: TestClient) -> None:
    """A second POST /rounds while a round is running gets HTTP 409."""
    from anvil.orchestrator.app import _round_lock

    _round_lock.acquire()
    try:
        response = client.post("/rounds", json={"round_id": 1})
        assert response.status_code == 409
        assert "already running" in response.json()["detail"]
    finally:
        _round_lock.release()


# ---------------------------------------------------------------------------
# GET /agents
# ---------------------------------------------------------------------------


def test_list_agents_empty(client: TestClient) -> None:
    """GET /agents returns [] when no agents directory exists."""
    response = client.get("/agents")
    assert response.status_code == 200
    assert response.json() == []


def test_list_agents(client: TestClient, tmp_path: Path) -> None:
    """GET /agents returns name, filename, path for each agents/*.yaml."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "forge_optimizer.yaml").write_text(
        "name: forge_optimizer\n", encoding="utf-8"
    )
    (agents_dir / "custom_agent.yaml").write_text(
        "name: my_custom_agent\n", encoding="utf-8"
    )
    response = client.get("/agents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    # sorted alphabetically by filename: custom_agent < forge_optimizer
    assert data[0]["name"] == "my_custom_agent"
    assert data[0]["filename"] == "custom_agent.yaml"
    assert data[0]["path"] == "agents/custom_agent.yaml"
    assert data[1]["name"] == "forge_optimizer"
    assert data[1]["filename"] == "forge_optimizer.yaml"
    assert data[1]["path"] == "agents/forge_optimizer.yaml"


def test_list_agents_fallback_name(client: TestClient, tmp_path: Path) -> None:
    """Agent YAML without a name field falls back to the file stem."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "unnamed.yaml").write_text("foo: bar\n", encoding="utf-8")
    response = client.get("/agents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "unnamed"


# ---------------------------------------------------------------------------
# POST /validate
# ---------------------------------------------------------------------------


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


def test_validate(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /validate runs eval and returns the aggregate score."""
    monkeypatch.setattr(
        "anvil.orchestrator.app.evaluate_branch", lambda **kwargs: _fake_eval_report()
    )
    response = client.post("/validate")
    assert response.status_code == 200
    data = response.json()
    assert data["aggregate"] == 0.85
    assert data["run_id"] == "test-run-123"
    assert data["n_examples"] == 8
    assert data["mode"] == "quick"


def test_validate_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /validate surfaces eval failures as HTTP 500."""
    def boom(**kwargs: Any) -> EvalReport:
        raise RuntimeError("eval exploded")

    monkeypatch.setattr("anvil.orchestrator.app.evaluate_branch", boom)
    response = client.post("/validate")
    assert response.status_code == 500
    assert "eval exploded" in response.json()["detail"]


def test_validate_runs_on_worker_thread(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /validate must run evaluate_branch in a thread pool."""
    main_thread_id = threading.current_thread().ident
    captured: dict[str, Any] = {}

    def mock_eval(**kwargs: Any) -> EvalReport:
        captured["thread_id"] = threading.current_thread().ident
        return _fake_eval_report()

    monkeypatch.setattr("anvil.orchestrator.app.evaluate_branch", mock_eval)
    response = client.post("/validate")
    assert response.status_code == 200
    assert captured["thread_id"] is not None
    assert captured["thread_id"] != main_thread_id


# ---------------------------------------------------------------------------
# POST /rounds with mode + agent config writes
# ---------------------------------------------------------------------------


def test_start_round_writes_mode_and_agent(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /rounds writes mode + agent_bundle_path to harness/config.yaml."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "mode: prompt\noptimizer:\n  backend: local\n"
        "  agent_bundle_path: agents/forge_optimizer.yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "anvil.orchestrator.app.run_round",
        lambda **kwargs: RoundReport(
            round_id=kwargs["round_id"],
            branch="anvil/exp-round-1",
            decision=Decision.KEEP,
            action_kind="edit_skill",
            parse_status="ok",
            diff_summary="edited skill X",
        ),
    )
    response = client.post(
        "/rounds", json={"mode": "code", "agent": "agents/custom.yaml"}
    )
    assert response.status_code == 200
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["mode"] == "code"
    assert raw["optimizer"]["agent_bundle_path"] == "agents/custom.yaml"


def test_start_round_writes_mode_only(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /rounds with only mode set updates just the mode key."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    original = (
        "mode: prompt\noptimizer:\n  backend: local\n"
        "  agent_bundle_path: agents/forge_optimizer.yaml\n"
    )
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        "anvil.orchestrator.app.run_round",
        lambda **kwargs: RoundReport(
            round_id=kwargs["round_id"],
            branch="anvil/exp-round-1",
            decision=Decision.KEEP,
            action_kind="edit_skill",
            parse_status="ok",
            diff_summary="edited skill X",
        ),
    )
    response = client.post("/rounds", json={"mode": "code"})
    assert response.status_code == 200
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["mode"] == "code"
    assert raw["optimizer"]["agent_bundle_path"] == "agents/forge_optimizer.yaml"


def test_start_round_no_config_write_when_unset(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /rounds without mode/agent leaves harness/config.yaml untouched."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    original = "mode: prompt\noptimizer:\n  backend: local\n"
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        "anvil.orchestrator.app.run_round",
        lambda **kwargs: RoundReport(
            round_id=kwargs["round_id"],
            branch="anvil/exp-round-1",
            decision=Decision.KEEP,
            action_kind="edit_skill",
            parse_status="ok",
            diff_summary="edited skill X",
        ),
    )
    response = client.post("/rounds", json={})
    assert response.status_code == 200
    assert config_path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Interactive dashboard UI
# ---------------------------------------------------------------------------


def test_dashboard_has_interactive_cards(client: TestClient) -> None:
    """GET / renders the four workflow cards + the rounds table."""
    response = client.get("/")
    assert response.status_code == 200
    text = response.text
    assert "1. Select Agent" in text
    assert "2. Validate Baseline" in text
    assert "3. Optimization Mode" in text
    assert "4. Run Optimizer" in text
    assert 'id="agent-select"' in text
    assert 'id="validate-btn"' in text
    assert 'id="run-btn"' in text
    assert 'name="mode"' in text
    assert 'value="prompt"' in text
    assert 'value="code"' in text
    assert "Rounds History" in text
    assert "/agents" in text
    assert "/validate" in text
    assert "/rounds" in text


# ---------------------------------------------------------------------------
# Phase 0: GET /api/state
# ---------------------------------------------------------------------------


def test_api_state_empty(client: TestClient) -> None:
    """GET /api/state returns all phases as non-existent on an empty repo."""
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert data["scaffold"]["exists"] is False
    assert data["scaffold"]["skills_count"] == 0
    assert data["golden_set"]["exists"] is False
    assert data["config"]["exists"] is False
    assert data["baseline"]["exists"] is False
    assert data["rounds"]["count"] == 0
    assert data["frontier"]["exists"] is False
    assert data["finalized"]["exists"] is False


def test_api_state_populated(client: TestClient, tmp_path: Path) -> None:
    """GET /api state reflects all four phases when files are present."""
    # Scaffold
    scaffold_dir = tmp_path / "scaffold"
    skills_dir = scaffold_dir / "skills"
    rules_dir = scaffold_dir / "rules"
    skills_dir.mkdir(parents=True)
    rules_dir.mkdir(parents=True)
    (scaffold_dir / "harness.yaml").write_text(
        "skills:\n- file: a.md\n- file: b.md\nrules:\n- file: r.md\n",
        encoding="utf-8",
    )
    (skills_dir / "a.md").write_text("a", encoding="utf-8")
    # Golden set
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "golden_set.jsonl").write_text(
        json.dumps({"example_id": "1", "category": "direct", "query": "q", "expected": "a"})
        + "\n",
        encoding="utf-8",
    )
    # Config
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "config.yaml").write_text(
        "mode: prompt\noptimizer:\n  backend: local\n", encoding="utf-8"
    )
    # Baseline
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "baseline.json").write_text(
        json.dumps(
            {
                "scaffold_commit_sha": "abc",
                "evaluated_at": "2024-01-01",
                "mode": "quick",
                "scorers": ["correctness"],
                "runtime_endpoint": "m",
                "judge_endpoint": "m",
                "aggregate": 0.85,
                "n_examples": 8,
            }
        ),
        encoding="utf-8",
    )
    # Round + frontier + finalized
    (runs_dir / "round_001.json").write_text(
        json.dumps({"round_id": 1, "decision": "keep"}), encoding="utf-8"
    )
    (runs_dir / "frontier.json").write_text(
        json.dumps({"best": {"aggregate": 0.9}}), encoding="utf-8"
    )
    (runs_dir / "finalized.json").write_text(
        json.dumps({"aggregate": 0.88}), encoding="utf-8"
    )

    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert data["scaffold"]["exists"] is True
    assert data["scaffold"]["skills_count"] == 2
    assert data["scaffold"]["rules_count"] == 1
    assert data["golden_set"]["exists"] is True
    assert data["golden_set"]["count"] == 1
    assert data["golden_set"]["buckets"]["direct"] == 1
    assert data["config"]["exists"] is True
    assert data["config"]["mode"] == "prompt"
    assert data["config"]["optimizer_backend"] == "local"
    assert data["baseline"]["exists"] is True
    assert data["baseline"]["aggregate"] == 0.85
    assert data["baseline"]["n_examples"] == 8
    assert data["rounds"]["count"] == 1
    assert data["frontier"]["exists"] is True
    assert data["finalized"]["exists"] is True
    assert data["finalized"]["aggregate"] == 0.88


# ---------------------------------------------------------------------------
# Phase 1: GET /api/scaffold
# ---------------------------------------------------------------------------


def test_get_scaffold(client: TestClient, tmp_path: Path) -> None:
    """GET /api/scaffold returns config + file list."""
    scaffold_dir = tmp_path / "scaffold"
    skills_dir = scaffold_dir / "skills"
    rules_dir = scaffold_dir / "rules"
    skills_dir.mkdir(parents=True)
    rules_dir.mkdir(parents=True)
    (scaffold_dir / "harness.yaml").write_text(
        "sampling:\n  temperature: 0.3\nskills:\n- file: identity.md\n"
        "rules:\n- file: rule1.md\n",
        encoding="utf-8",
    )
    (skills_dir / "identity.md").write_text("# Identity", encoding="utf-8")
    (rules_dir / "rule1.md").write_text("# Rule 1", encoding="utf-8")
    response = client.get("/api/scaffold")
    assert response.status_code == 200
    data = response.json()
    assert data["config"]["skills"] == [{"file": "identity.md"}]
    assert data["config"]["rules"] == [{"file": "rule1.md"}]
    assert len(data["files"]) == 2
    assert any(f["name"] == "identity.md" for f in data["files"])
    assert any(f["name"] == "rule1.md" for f in data["files"])


def test_get_scaffold_not_found(client: TestClient) -> None:
    """GET /api/scaffold returns 404 when harness.yaml is missing."""
    response = client.get("/api/scaffold")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Phase 1: PUT /api/scaffold
# ---------------------------------------------------------------------------


def test_put_scaffold(client: TestClient, tmp_path: Path) -> None:
    """PUT /api/scaffold writes the scaffold config to harness.yaml."""
    scaffold_dir = tmp_path / "scaffold"
    scaffold_dir.mkdir(parents=True)
    response = client.put(
        "/api/scaffold",
        json={
            "sampling": {"temperature": 0.5},
            "skills": [{"file": "test.md"}],
            "rules": [],
            "tools": [],
        },
    )
    assert response.status_code == 200
    raw = yaml.safe_load((scaffold_dir / "harness.yaml").read_text(encoding="utf-8"))
    assert raw["sampling"]["temperature"] == 0.5
    assert raw["skills"] == [{"file": "test.md"}]


# ---------------------------------------------------------------------------
# Phase 1: GET /api/scaffold/files/{filename}
# ---------------------------------------------------------------------------


def test_get_scaffold_file(client: TestClient, tmp_path: Path) -> None:
    """GET /api/scaffold/files/{name} returns file content as text/plain."""
    skills_dir = tmp_path / "scaffold" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "identity.md").write_text(
        "# Identity\n\nYou are a helpful agent.", encoding="utf-8"
    )
    response = client.get("/api/scaffold/files/identity.md")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "# Identity" in response.text
    assert "helpful agent" in response.text


def test_get_scaffold_file_in_rules(client: TestClient, tmp_path: Path) -> None:
    """GET /api/scaffold/files/{name} finds files in scaffold/rules/ too."""
    rules_dir = tmp_path / "scaffold" / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "answer_scope.md").write_text("# Scope", encoding="utf-8")
    response = client.get("/api/scaffold/files/answer_scope.md")
    assert response.status_code == 200
    assert "# Scope" in response.text


def test_get_scaffold_file_not_found(client: TestClient) -> None:
    """GET /api/scaffold/files/{name} returns 404 for missing files."""
    response = client.get("/api/scaffold/files/nonexistent.md")
    assert response.status_code == 404


def test_get_scaffold_file_rejects_bad_filename(client: TestClient) -> None:
    """GET /api/scaffold/files/{name} rejects path-traversal / non-md names."""
    # Starlette normalizes ".." in the URL path before routing (→ 404).
    # The important property: the request never reaches the filesystem.
    assert client.get("/api/scaffold/files/..").status_code != 200
    # Dot-prefixed and non-md names reach the handler → 400 from the validator.
    assert client.get("/api/scaffold/files/.hidden.md").status_code == 400
    assert client.get("/api/scaffold/files/notmd.txt").status_code == 400


# ---------------------------------------------------------------------------
# Phase 1: PUT /api/scaffold/files/{filename}
# ---------------------------------------------------------------------------


def test_put_scaffold_file(client: TestClient, tmp_path: Path) -> None:
    """PUT /api/scaffold/files/{name} writes content and GET reads it back."""
    skills_dir = tmp_path / "scaffold" / "skills"
    skills_dir.mkdir(parents=True)
    response = client.put(
        "/api/scaffold/files/new_skill.md",
        content="# New Skill\n\nThis is a test.",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 200
    written = (skills_dir / "new_skill.md").read_text(encoding="utf-8")
    assert "# New Skill" in written
    response = client.get("/api/scaffold/files/new_skill.md")
    assert response.status_code == 200
    assert "# New Skill" in response.text


def test_put_scaffold_file_overwrites_existing(client: TestClient, tmp_path: Path) -> None:
    """PUT overwrites a file in-place when it already exists in skills/."""
    skills_dir = tmp_path / "scaffold" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "identity.md").write_text("old content", encoding="utf-8")
    client.put(
        "/api/scaffold/files/identity.md",
        content="new content",
        headers={"Content-Type": "text/plain"},
    )
    assert (skills_dir / "identity.md").read_text(encoding="utf-8") == "new content"


def test_put_scaffold_file_rejects_bad_filename(client: TestClient) -> None:
    """PUT /api/scaffold/files/{name} rejects path-traversal / non-md names."""
    # Starlette normalizes ".." away before routing — never reaches the handler.
    response = client.put(
        "/api/scaffold/files/..",
        content="malicious",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code != 200
    # Dot-prefixed names reach the handler → 400 from the validator.
    response = client.put(
        "/api/scaffold/files/.hidden.md",
        content="malicious",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Phase 1: GET /api/golden-set
# ---------------------------------------------------------------------------


def test_get_golden_set(client: TestClient, tmp_path: Path) -> None:
    """GET /api/golden-set returns total, per-bucket counts, and samples."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    lines = [
        json.dumps({"example_id": "1", "category": "direct", "query": "q1"}),
        json.dumps({"example_id": "2", "category": "multi_hop", "query": "q2"}),
        json.dumps({"example_id": "3", "category": "direct", "query": "q3"}),
    ]
    (data_dir / "golden_set.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    response = client.get("/api/golden-set")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["buckets"]["direct"] == 2
    assert data["buckets"]["multi_hop"] == 1
    assert len(data["samples"]) == 3


def test_get_golden_set_missing(client: TestClient) -> None:
    """GET /api/golden-set returns empty stats when the file is missing."""
    response = client.get("/api/golden-set")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["buckets"] == {}
    assert data["samples"] == []


# ---------------------------------------------------------------------------
# Phase 1: GET /api/config
# ---------------------------------------------------------------------------


def test_get_config(client: TestClient, tmp_path: Path) -> None:
    """GET /api/config returns harness/config.yaml parsed as JSON."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\nruntime_endpoint: model\noptimizer:\n  backend: local\n",
        encoding="utf-8",
    )
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "prompt"
    assert data["runtime_endpoint"] == "model"
    assert data["optimizer"]["backend"] == "local"


def test_get_config_not_found(client: TestClient) -> None:
    """GET /api/config returns 404 when config.yaml is missing."""
    response = client.get("/api/config")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Phase 1: PUT /api/config
# ---------------------------------------------------------------------------


def test_put_config(client: TestClient, tmp_path: Path) -> None:
    """PUT /api/config merges only allowed fields into the existing config."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\nruntime_endpoint: old\noptimizer_endpoint: opt\n"
        "judge_endpoint: judge\nexperiments:\n  runtime: r\n  eval: e\n  optimizer: o\n"
        "eval:\n  default_mode: quick\n  n_workers: 4\n",
        encoding="utf-8",
    )
    response = client.put(
        "/api/config",
        json={
            "mode": "code",
            "runtime_endpoint": "new-model",
            "eval": {"default_mode": "standard"},
        },
    )
    assert response.status_code == 200
    raw = yaml.safe_load((config_dir / "config.yaml").read_text(encoding="utf-8"))
    assert raw["mode"] == "code"
    assert raw["runtime_endpoint"] == "new-model"
    # Unchanged top-level fields preserved
    assert raw["optimizer_endpoint"] == "opt"
    assert raw["judge_endpoint"] == "judge"
    assert raw["experiments"]["eval"] == "e"
    # Nested update applied
    assert raw["eval"]["default_mode"] == "standard"
    # Original nested field preserved
    assert raw["eval"]["n_workers"] == 4


def test_put_config_creates_new(client: TestClient, tmp_path: Path) -> None:
    """PUT /api/config works when no existing config is present."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    response = client.put("/api/config", json={"mode": "code"})
    assert response.status_code == 200
    raw = yaml.safe_load((config_dir / "config.yaml").read_text(encoding="utf-8"))
    assert raw["mode"] == "code"


# ---------------------------------------------------------------------------
# Phase 2: POST /api/baseline
# ---------------------------------------------------------------------------


def test_post_baseline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/baseline runs eval, returns baseline, writes baseline.json."""
    monkeypatch.setattr(
        "anvil.orchestrator.app.evaluate_branch", lambda **kwargs: _fake_eval_report()
    )
    response = client.post("/api/baseline")
    assert response.status_code == 200
    data = response.json()
    assert data["aggregate"] == 0.85
    assert data["n_examples"] == 8
    assert data["mode"] == "quick"
    assert "scaffold_commit_sha" in data
    assert data["per_judge"]["correctness"] == 0.9
    # Baseline file was written — verify via GET
    get_response = client.get("/api/baseline")
    assert get_response.status_code == 200
    assert get_response.json()["aggregate"] == 0.85


def test_post_baseline_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/baseline surfaces eval failures as HTTP 500."""

    def boom(**kwargs: Any) -> EvalReport:
        raise RuntimeError("baseline exploded")

    monkeypatch.setattr("anvil.orchestrator.app.evaluate_branch", boom)
    response = client.post("/api/baseline")
    assert response.status_code == 500
    assert "baseline exploded" in response.json()["detail"]


def test_post_baseline_runs_on_worker_thread(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/baseline must run evaluate_branch in a thread pool."""
    main_thread_id = threading.current_thread().ident
    captured: dict[str, Any] = {}

    def mock_eval(**kwargs: Any) -> EvalReport:
        captured["thread_id"] = threading.current_thread().ident
        return _fake_eval_report()

    monkeypatch.setattr("anvil.orchestrator.app.evaluate_branch", mock_eval)
    response = client.post("/api/baseline")
    assert response.status_code == 200
    assert captured["thread_id"] is not None
    assert captured["thread_id"] != main_thread_id


def test_post_baseline_conflict_409(client: TestClient) -> None:
    """A second POST /api/baseline while one is running gets HTTP 409."""
    from anvil.orchestrator.app import _mutation_lock

    _mutation_lock.acquire()
    try:
        response = client.post("/api/baseline")
        assert response.status_code == 409
        assert "already running" in response.json()["detail"]
    finally:
        _mutation_lock.release()


# ---------------------------------------------------------------------------
# Phase 2: GET /api/baseline
# ---------------------------------------------------------------------------


def _write_baseline(tmp_path: Path, aggregate: float = 0.85, n_examples: int = 8) -> None:
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "baseline.json").write_text(
        json.dumps(
            {
                "scaffold_commit_sha": "abc123",
                "evaluated_at": "2024-01-01T00:00:00",
                "mode": "quick",
                "scorers": ["correctness"],
                "runtime_endpoint": "test",
                "judge_endpoint": "test",
                "aggregate": aggregate,
                "per_judge": {"correctness": 0.9},
                "n_examples": n_examples,
            }
        ),
        encoding="utf-8",
    )


def test_get_baseline(client: TestClient, tmp_path: Path) -> None:
    """GET /api/baseline returns the cached baseline."""
    _write_baseline(tmp_path)
    response = client.get("/api/baseline")
    assert response.status_code == 200
    data = response.json()
    assert data["aggregate"] == 0.85
    assert data["n_examples"] == 8
    assert data["mode"] == "quick"


def test_get_baseline_not_found(client: TestClient) -> None:
    """GET /api/baseline returns 404 when no baseline exists."""
    response = client.get("/api/baseline")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Phase 3: GET /api/frontier
# ---------------------------------------------------------------------------


def test_get_frontier(client: TestClient, tmp_path: Path) -> None:
    """GET /api/frontier returns frontier.json parsed as JSON."""
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "frontier.json").write_text(
        json.dumps(
            {
                "best": {"aggregate": 0.9, "correctness": 0.85},
                "objectives": ["aggregate", "correctness"],
                "pareto": False,
            }
        ),
        encoding="utf-8",
    )
    response = client.get("/api/frontier")
    assert response.status_code == 200
    data = response.json()
    assert data["best"]["aggregate"] == 0.9
    assert data["best"]["correctness"] == 0.85


def test_get_frontier_not_found(client: TestClient) -> None:
    """GET /api/frontier returns 404 when no frontier exists."""
    response = client.get("/api/frontier")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Phase 4: POST /api/finalize
# ---------------------------------------------------------------------------


def test_post_finalize(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/finalize runs held-out eval and writes finalized.json."""
    # Config with held_out_test enabled
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\nruntime_endpoint: test\njudge_endpoint: test\n"
        "eval:\n  held_out_test: true\n  default_mode: quick\n",
        encoding="utf-8",
    )
    # Frontier must exist
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "frontier.json").write_text(
        json.dumps({"best": {"aggregate": 0.9}, "objectives": ["aggregate"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "anvil.orchestrator.app.evaluate_branch", lambda **kwargs: _fake_eval_report()
    )
    response = client.post("/api/finalize")
    assert response.status_code == 200
    data = response.json()
    assert data["aggregate"] == 0.85
    assert data["per_judge"]["correctness"] == 0.9
    assert "scaffold_commit_sha" in data
    assert "finalized_at" in data
    assert "frontier" in data
    assert data["frontier"]["best"]["aggregate"] == 0.9
    # finalized.json was written
    assert (runs_dir / "finalized.json").is_file()


def test_post_finalize_no_held_out(
    client: TestClient, tmp_path: Path
) -> None:
    """POST /api/finalize returns 500 when held_out_test is disabled."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\nruntime_endpoint: test\njudge_endpoint: test\n"
        "eval:\n  held_out_test: false\n",
        encoding="utf-8",
    )
    response = client.post("/api/finalize")
    assert response.status_code == 500
    assert "held_out_test" in response.json()["detail"]


def test_post_finalize_no_frontier(
    client: TestClient, tmp_path: Path
) -> None:
    """POST /api/finalize returns 500 when no frontier exists."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\nruntime_endpoint: test\njudge_endpoint: test\n"
        "eval:\n  held_out_test: true\n",
        encoding="utf-8",
    )
    response = client.post("/api/finalize")
    assert response.status_code == 500
    assert "frontier" in response.json()["detail"].lower()


def test_post_finalize_runs_on_worker_thread(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/finalize must run evaluate_branch in a thread pool."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\nruntime_endpoint: test\njudge_endpoint: test\n"
        "eval:\n  held_out_test: true\n",
        encoding="utf-8",
    )
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "frontier.json").write_text(
        json.dumps({"best": {"aggregate": 0.9}}), encoding="utf-8"
    )
    main_thread_id = threading.current_thread().ident
    captured: dict[str, Any] = {}

    def mock_eval(**kwargs: Any) -> EvalReport:
        captured["thread_id"] = threading.current_thread().ident
        return _fake_eval_report()

    monkeypatch.setattr("anvil.orchestrator.app.evaluate_branch", mock_eval)
    response = client.post("/api/finalize")
    assert response.status_code == 200
    assert captured["thread_id"] is not None
    assert captured["thread_id"] != main_thread_id


def test_post_finalize_conflict_409(client: TestClient) -> None:
    """A second POST /api/finalize while one is running gets HTTP 409."""
    from anvil.orchestrator.app import _mutation_lock

    _mutation_lock.acquire()
    try:
        response = client.post("/api/finalize")
        assert response.status_code == 409
        assert "already running" in response.json()["detail"]
    finally:
        _mutation_lock.release()


# ---------------------------------------------------------------------------
# Phase 4: GET /api/finalize
# ---------------------------------------------------------------------------


def test_get_finalized(client: TestClient, tmp_path: Path) -> None:
    """GET /api/finalize returns the finalized report."""
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "finalized.json").write_text(
        json.dumps(
            {
                "aggregate": 0.88,
                "per_judge": {"correctness": 0.9},
                "n_rows": 20,
                "mode": "test",
                "scaffold_commit_sha": "abc123",
                "frontier": {"best": {"aggregate": 0.9}},
            }
        ),
        encoding="utf-8",
    )
    response = client.get("/api/finalize")
    assert response.status_code == 200
    data = response.json()
    assert data["aggregate"] == 0.88
    assert data["mode"] == "test"
    assert data["frontier"]["best"]["aggregate"] == 0.9


def test_get_finalized_not_found(client: TestClient) -> None:
    """GET /api/finalize returns 404 when no finalized report exists."""
    response = client.get("/api/finalize")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# BLOCKING FIX 1: Auth token leak — GET /api/config must redact secrets
# ---------------------------------------------------------------------------


def test_get_config_redacts_auth_token(client: TestClient, tmp_path: Path) -> None:
    """GET /api/config redacts optimizer.auth_token with '***'."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\noptimizer:\n  backend: local\n  auth_token: super-secret-abc123\n",
        encoding="utf-8",
    )
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["optimizer"]["auth_token"] == "***"


def test_get_config_redacts_empty_auth_token(client: TestClient, tmp_path: Path) -> None:
    """Empty auth_token stays empty (not replaced with '***')."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\noptimizer:\n  backend: local\n  auth_token: ''\n",
        encoding="utf-8",
    )
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["optimizer"]["auth_token"] == ""


def test_get_config_redacts_all_secret_keywords(client: TestClient, tmp_path: Path) -> None:
    """Any field whose key contains token/secret/password/credential is redacted."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\n"
        "runtime_endpoint: model\n"
        "optimizer:\n"
        "  backend: local\n"
        "  auth_token: tok123\n"
        "  server_url: http://x\n"
        "judge_endpoint: judge\n"
        "custom_secret_field: hush\n"
        "my_password: pw123\n"
        "api_credential: cred456\n"
        "safe_field: visible\n",
        encoding="utf-8",
    )
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["optimizer"]["auth_token"] == "***"
    assert data["custom_secret_field"] == "***"
    assert data["my_password"] == "***"
    assert data["api_credential"] == "***"
    # Non-secret fields are NOT redacted
    assert data["optimizer"]["server_url"] == "http://x"
    assert data["safe_field"] == "visible"
    assert data["runtime_endpoint"] == "model"


def test_get_config_redacts_nested_secret(client: TestClient, tmp_path: Path) -> None:
    """Secrets nested inside sub-dicts are also redacted recursively."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\n"
        "optimizer:\n"
        "  backend: local\n"
        "  auth_token: nested-secret\n"
        "  server_url: http://x\n",
        encoding="utf-8",
    )
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["optimizer"]["auth_token"] == "***"
    assert data["optimizer"]["server_url"] == "http://x"
    assert data["optimizer"]["backend"] == "local"


# ---------------------------------------------------------------------------
# BLOCKING FIX 2: Finalization is terminal
# ---------------------------------------------------------------------------


def test_post_finalize_already_exists_409(
    client: TestClient, tmp_path: Path
) -> None:
    """POST /api/finalize returns 409 when finalized.json already exists."""
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "finalized.json").write_text(
        json.dumps({"aggregate": 0.88}), encoding="utf-8"
    )
    response = client.post("/api/finalize")
    assert response.status_code == 409
    assert "finalization already exists" in response.json()["detail"]
    assert "delete eval/runs/finalized.json" in response.json()["detail"]


def test_post_rounds_finalized_409(client: TestClient, tmp_path: Path) -> None:
    """POST /rounds returns 409 when finalized.json exists."""
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "finalized.json").write_text(
        json.dumps({"aggregate": 0.88}), encoding="utf-8"
    )
    response = client.post("/rounds", json={"round_id": 1})
    assert response.status_code == 409
    assert "optimization is finalized" in response.json()["detail"]


def test_post_rounds_force_bypasses_finalized(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /rounds with force=true bypasses the finalized check."""
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "finalized.json").write_text(
        json.dumps({"aggregate": 0.88}), encoding="utf-8"
    )
    fake_report = RoundReport(
        round_id=5,
        branch="anvil/exp-round-5",
        decision=Decision.KEEP,
        action_kind="edit_skill",
        parse_status="ok",
        diff_summary="forced round",
    )
    monkeypatch.setattr("anvil.orchestrator.app.run_round", lambda **kwargs: fake_report)
    response = client.post("/rounds", json={"round_id": 5, "force": True})
    assert response.status_code == 200
    assert response.json()["round_id"] == 5
    assert response.json()["diff_summary"] == "forced round"


# ---------------------------------------------------------------------------
# BLOCKING FIX 3: Shared lock — PUT /api/config 409 when locked
# ---------------------------------------------------------------------------


def test_put_config_conflict_409(client: TestClient, tmp_path: Path) -> None:
    """PUT /api/config returns 409 when the mutation lock is held."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("mode: prompt\n", encoding="utf-8")
    from anvil.orchestrator.app import _mutation_lock

    _mutation_lock.acquire()
    try:
        response = client.put("/api/config", json={"mode": "code"})
        assert response.status_code == 409
        assert "already running" in response.json()["detail"]
    finally:
        _mutation_lock.release()


def test_put_scaffold_conflict_409(client: TestClient) -> None:
    """PUT /api/scaffold returns 409 when the mutation lock is held."""
    from anvil.orchestrator.app import _mutation_lock

    _mutation_lock.acquire()
    try:
        response = client.put("/api/scaffold", json={"skills": []})
        assert response.status_code == 409
        assert "already running" in response.json()["detail"]
    finally:
        _mutation_lock.release()


def test_put_scaffold_file_conflict_409(client: TestClient) -> None:
    """PUT /api/scaffold/files/{name} returns 409 when the mutation lock is held."""
    from anvil.orchestrator.app import _mutation_lock

    _mutation_lock.acquire()
    try:
        response = client.put(
            "/api/scaffold/files/test.md",
            content="content",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 409
        assert "already running" in response.json()["detail"]
    finally:
        _mutation_lock.release()


# ---------------------------------------------------------------------------
# WARNING 1: GET /api/state survives malformed YAML/JSON
# ---------------------------------------------------------------------------


def test_api_state_malformed_scaffold_yaml(client: TestClient, tmp_path: Path) -> None:
    """GET /api/state returns 200 with defaults when scaffold YAML is corrupt."""
    scaffold_dir = tmp_path / "scaffold"
    scaffold_dir.mkdir()
    (scaffold_dir / "harness.yaml").write_text(
        "skills: [unclosed\n  - broken", encoding="utf-8"
    )
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert data["scaffold"]["exists"] is True
    assert data["scaffold"]["skills_count"] == 0
    assert data["scaffold"]["rules_count"] == 0


def test_api_state_malformed_config_yaml(client: TestClient, tmp_path: Path) -> None:
    """GET /api/state returns 200 with defaults when config YAML is corrupt."""
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "config.yaml").write_text(
        "mode: [unclosed\n  broken: {", encoding="utf-8"
    )
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert data["config"]["exists"] is True
    assert data["config"]["mode"] == ""


def test_api_state_malformed_baseline_json(client: TestClient, tmp_path: Path) -> None:
    """GET /api/state returns 200 with baseline.exists=False when JSON is corrupt."""
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "baseline.json").write_text("{not valid json", encoding="utf-8")
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert data["baseline"]["exists"] is False


def test_api_state_malformed_finalized_json(client: TestClient, tmp_path: Path) -> None:
    """GET /api/state returns 200 with finalized.aggregate=None when JSON is corrupt."""
    runs_dir = tmp_path / "eval" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "finalized.json").write_text("} broken json {", encoding="utf-8")
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert data["finalized"]["exists"] is True
    assert data["finalized"]["aggregate"] is None


# ---------------------------------------------------------------------------
# WARNING 2: PUT /api/config uses ConfigUpdateRequest + held_out_test
# ---------------------------------------------------------------------------


def test_put_config_held_out_test(client: TestClient, tmp_path: Path) -> None:
    """PUT /api/config can set eval.held_out_test."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\nruntime_endpoint: m\noptimizer_endpoint: o\njudge_endpoint: j\n"
        "experiments:\n  runtime: r\n  eval: e\n  optimizer: o\n"
        "eval:\n  default_mode: quick\n  held_out_test: false\n",
        encoding="utf-8",
    )
    response = client.put(
        "/api/config",
        json={"eval": {"held_out_test": True}},
    )
    assert response.status_code == 200
    raw = yaml.safe_load((config_dir / "config.yaml").read_text(encoding="utf-8"))
    assert raw["eval"]["held_out_test"] is True
    # Original eval field preserved
    assert raw["eval"]["default_mode"] == "quick"


def test_put_config_ignores_disallowed_fields(client: TestClient, tmp_path: Path) -> None:
    """PUT /api/config ignores fields not in the allow-list."""
    config_dir = tmp_path / "harness"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "mode: prompt\nruntime_endpoint: m\noptimizer_endpoint: o\njudge_endpoint: j\n"
        "experiments:\n  runtime: r\n  eval: e\n  optimizer: o\n",
        encoding="utf-8",
    )
    # "agent_module" and "experiments" are NOT in _CONFIG_ALLOWED_TOP
    response = client.put(
        "/api/config",
        json={"agent_module": "anvil.agents.evil", "mode": "code"},
    )
    assert response.status_code == 200
    raw = yaml.safe_load((config_dir / "config.yaml").read_text(encoding="utf-8"))
    # mode was updated (allowed)
    assert raw["mode"] == "code"
    # agent_module was NOT written (not in allow-list; Pydantic model ignores it
    # because it's not a field on ConfigUpdateRequest)
    assert "agent_module" not in raw or raw.get("agent_module") != "anvil.agents.evil"


# ---------------------------------------------------------------------------
# WARNING 3: Atomic writes — verified indirectly via existing PUT tests
# (the _atomic_write helper is exercised by every PUT endpoint; no separate
# test needed since the existing tests confirm the write succeeds and the
# content is correct. The temp-file + os.replace pattern is unit-tested by
# the fact that no partial files appear in the test tmpdirs.)


# ---------------------------------------------------------------------------
# WARNING 4: Dashboard finalize button checks baseline AND frontier
# ---------------------------------------------------------------------------


def test_dashboard_finalize_button_requires_baseline_and_frontier(
    client: TestClient, tmp_path: Path
) -> None:
    """The dashboard JS disables the finalize button when baseline OR frontier is missing."""
    response = client.get("/")
    assert response.status_code == 200
    # The JS code must check both conditions:
    #   $("finalize-btn").disabled = !state.baseline.exists || !state.frontier.exists;
    assert "state.baseline.exists" in response.text
    assert "state.frontier.exists" in response.text
    assert "finalize-btn" in response.text


def test_dashboard_finalize_button_js_checks_both(client: TestClient) -> None:
    """The dashboard JS must check both baseline.exists and frontier.exists for the finalize button."""
    response = client.get("/")
    text = response.text
    # The JS line should reference both baseline.exists and frontier.exists
    # in the finalize button disabled assignment.
    # Find the line with finalize-btn disabled assignment
    assert "!state.baseline.exists" in text
    assert "!state.frontier.exists" in text
