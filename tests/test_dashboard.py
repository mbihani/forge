"""Tests for the import-light frontier dashboard data layer."""

import json

from anvil.dashboard.data import (
    all_round_points,
    load_frontier,
    load_round_history,
    pareto_frontier_points,
)
from anvil.loop.frontier import Frontier, save_frontier
from anvil.loop.round import _save_dashboard_round


def test_load_frontier_and_rounds(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    raw = {"best": {"accuracy": 0.9}, "objectives": ["accuracy"]}
    (data / "frontier.json").write_text(json.dumps(raw), encoding="utf-8")
    (data / "round_010.json").write_text('{"round_id": 10}', encoding="utf-8")
    (data / "round_002.json").write_text('{"round_id": 2}', encoding="utf-8")
    (data / "round_003.json").write_text("not json", encoding="utf-8")
    assert load_frontier(tmp_path) == raw
    assert [row["round_id"] for row in load_round_history(tmp_path)] == [2, 10]
    assert load_frontier(tmp_path / "missing") is None


def test_pareto_point_extraction():
    raw = {"points": [{"round_id": 2, "scores": {"accuracy": 0.8, "tokens": 90}}]}
    assert pareto_frontier_points(raw) == [{"round_id": 2, "accuracy": 0.8, "tokens": 90}]


def test_pareto_point_extraction_from_production_frontier_shape():
    raw = Frontier.from_scores({"accuracy": 0.8, "tokens": 90}).to_dict()
    assert pareto_frontier_points(raw) == [{"round_id": None, "accuracy": 0.8, "tokens": 90}]


def test_all_round_points_marks_non_dominated_rounds():
    rounds = [
        {"round_id": 1, "aggregate": 0.8, "cost_metrics": {"total_tokens": 100}},
        {"round_id": 2, "aggregate": 0.9, "cost_metrics": {"total_tokens": 90}},
        {"round_id": 3, "aggregate": 0.95, "cost_metrics": {"total_tokens": 110}},
    ]
    objectives = [
        {"name": "accuracy", "source": "aggregate", "direction": "maximize"},
        {"name": "tokens", "source": "tokens", "direction": "minimize"},
    ]
    points = all_round_points(rounds, objectives)
    assert [point["on_frontier"] for point in points] == [False, True, True]


def test_all_round_points_excludes_rounds_with_missing_objectives():
    rounds = [
        {"round_id": 1, "aggregate": 0.9},
        {"round_id": 2},
    ]
    objectives = [{"name": "quality", "source": "aggregate"}]
    assert [point["on_frontier"] for point in all_round_points(rounds, objectives)] == [
        True,
        False,
    ]


def test_cost_objective_uses_preserved_source():
    frontier = Frontier.from_scores(
        {"cost": 1234},
        directions={"cost": "minimize"},
        sources={"cost": "context_chars"},
    ).to_dict()
    objective = {
        "name": "cost",
        "source": frontier["sources"].get("cost", "cost"),
        "direction": frontier["directions"]["cost"],
    }
    points = all_round_points(
        [{"round_id": 1, "cost_metrics": {"total_context_chars": 1234}}],
        [objective],
    )
    assert points == [{"round_id": 1, "cost": 1234, "on_frontier": True}]


def test_frontier_is_persisted_for_dashboard(tmp_path):
    frontier = Frontier.from_scores({"accuracy": 0.9})
    save_frontier(tmp_path, frontier)
    assert json.loads((tmp_path / "data" / "frontier.json").read_text()) == frontier.to_dict()


def test_round_report_is_persisted_for_dashboard(tmp_path):
    path = _save_dashboard_round(tmp_path, 7, '{"round_id": 7}\n')
    assert path == tmp_path / "data" / "round_007.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"round_id": 7}
