"""Data loading for the frontier dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_frontier(repo_root: Path | str) -> dict[str, Any] | None:
    """Load ``data/frontier.json``, or None when no frontier exists."""
    path = Path(repo_root) / "data" / "frontier.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_round_history(repo_root: Path | str) -> list[dict[str, Any]]:
    """Load round reports from ``data``, sorted by numeric round id."""
    paths = (Path(repo_root) / "data").glob("round_*.json")
    rounds = []
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(report, dict):
            rounds.append(report)
    return sorted(rounds, key=lambda report: int(report.get("round_id", 0)))


def pareto_frontier_points(frontier: dict) -> list[dict]:
    """Extract Pareto points, flattening optional nested score mappings."""
    for key in ("points", "frontier"):
        raw = frontier.get(key)
        if isinstance(raw, list):
            return [
                {"round_id": point.get("round_id"), **point.get("scores", {})}
                if isinstance(point.get("scores"), dict)
                else dict(point)
                for point in raw
                if isinstance(point, dict)
            ]
    best = frontier.get("best", {})
    return [{"round_id": frontier.get("round_id"), **best}] if best else []


def _name(objective: str | dict[str, Any]) -> str:
    return objective if isinstance(objective, str) else str(objective["name"])


def _value(report: dict[str, Any], objective: str | dict[str, Any]) -> Any:
    name = _name(objective)
    source = objective.get("source", name) if isinstance(objective, dict) else name
    if source in {"aggregate", "aggregate_score"}:
        return report.get("aggregate", report.get("aggregate_score"))
    if name in report:
        return report[name]
    if name in report.get("per_judge", {}):
        return report["per_judge"][name]
    metric = {
        "tokens": "total_tokens",
        "context_chars": "total_context_chars",
        "latency": "latency_ms_median",
        "cost": "cost_usd_per_row",
    }.get(source, source)
    return report.get("cost_metrics", {}).get(metric)


def all_round_points(round_history: list[dict], objectives: list[dict]) -> list[dict]:
    """Extract objective scores and compute non-dominated round membership."""
    points = [
        {
            "round_id": report.get("round_id"),
            **{_name(objective): _value(report, objective) for objective in objectives},
        }
        for report in round_history
    ]
    complete_points = [
        point
        for point in points
        if all(point.get(_name(objective)) is not None for objective in objectives)
    ]

    def dominates(left: dict, right: dict) -> bool:
        no_worse, better = True, False
        for objective in objectives:
            name = _name(objective)
            a, b = left.get(name), right.get(name)
            if a is None or b is None:
                return False
            direction = (
                objective.get("direction", "maximize")
                if isinstance(objective, dict)
                else "maximize"
            )
            if direction == "minimize":
                no_worse, better = no_worse and a <= b, better or a < b
            else:
                no_worse, better = no_worse and a >= b, better or a > b
        return no_worse and better

    for point in points:
        point["on_frontier"] = point in complete_points and not any(
            other is not point and dominates(other, point) for other in complete_points
        )
    return points
