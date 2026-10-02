"""The round prompt tells the optimizer which objectives the gate keeps on."""

from __future__ import annotations

import json
from pathlib import Path

from anvil.loop.builder import build_round_prompt
from anvil.runtime.models import ParetoObjective

_BASELINE = {
    "aggregate": 0.9672,
    "mode": "full",
    "n_examples": 20,
    "scorers": ["summary_quality"],
    "per_judge": {"summary_quality": 0.934},
    "per_bucket": {},
    "cost_metrics": {"n_rows": 20.0, "latency_ms_median": 41000.0},
}

_OBJECTIVES = [
    ParetoObjective(name="latency", direction="minimize", source="latency", epsilon=2000.0),
    ParetoObjective(name="quality", direction="maximize", source="aggregate", epsilon=0.01),
]


def test_default_prompt_names_aggregate_only(tmp_path: Path) -> None:
    prompt = build_round_prompt(repo_root=tmp_path, round_id=1, baseline=_BASELINE)
    assert "aggregate (maximize): the only objective" in prompt


def test_pareto_prompt_lists_objectives_with_baseline_and_frontier(tmp_path: Path) -> None:
    runs = tmp_path / "eval" / "runs"
    runs.mkdir(parents=True)
    (runs / "frontier.json").write_text(json.dumps({"best": {"latency": 38000.0, "quality": 0.97}}))
    prompt = build_round_prompt(
        repo_root=tmp_path, round_id=2, baseline=_BASELINE, objectives=_OBJECTIVES
    )
    assert "latency (minimize latency_ms_median; epsilon=2000): baseline=41000, best-so-far=38000" in prompt
    assert "quality (maximize aggregate; epsilon=0.01): baseline=0.9672, best-so-far=0.97" in prompt
    assert "Baseline cost metrics: latency_ms_median=41000, n_rows=20" in prompt
