"""Tests for the Forge Orchestrator FastAPI app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

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
