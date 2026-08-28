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
