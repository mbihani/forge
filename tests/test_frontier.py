"""Tests for the Pareto frontier gate (``anvil.loop.frontier``).

Covers:

* The :class:`Frontier` class — seeding from a baseline, the keep/revert
  gate decision (``should_keep`` / ``update``), epsilon + single-objective
  fallback, and serialization round-trip.
* The :func:`gate_decision` integration — load/init/persist the frontier
  across rounds, including the regression that the legacy frozen-baseline
  delta gate silently allowed (a round worse than a previous KEPT round
  but still above the baseline) is now REVERTED.
* Backward compatibility — ``gate.type: delta`` reproduces the old
  frozen-baseline behavior verbatim.
* Config parsing — :func:`load_gate_config` reads the ``gate`` section
  and falls back to defaults; the real ``harness/config.yaml`` parses
  with the new ``gate`` field.

No LLM calls are made — the gate is pure Python + file I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anvil.loop.decision import Decision
from anvil.loop.frontier import (
    AGGREGATE_KEY,
    Frontier,
    _scores_for_objectives,
    gate_decision,
    load_frontier,
    load_gate_config,
    save_frontier,
    scores_from_baseline,
    scores_from_eval,
)
from anvil.runtime.models import GateConfig, ParetoConfig, ParetoObjective, RuntimeYAML

# Realistic objective keys (per-judge scores + the aggregate), matching
# the live baseline.json.
BASELINE = {
    "correctness": 0.375,
    "retrieval_groundedness": 0.857,
    "refusal_appropriateness": 1.0,
    AGGREGATE_KEY: 0.744,
}


# ---------------------------------------------------------------------------
# Frontier — seeding
# ---------------------------------------------------------------------------


def test_from_scores_seeds_best_and_objectives() -> None:
    f = Frontier.from_scores(BASELINE)
    assert f.best == BASELINE
    assert f.objectives == list(BASELINE.keys())
    assert f.pareto is True
    assert f.epsilon == 0.0


def test_empty_frontier_first_update_seeds_from_scores() -> None:
    """An empty frontier accepts the first point as its seed."""
    f = Frontier()
    assert f.update(BASELINE) is True
    assert f.best == BASELINE
    assert f.objectives == list(BASELINE.keys())


# ---------------------------------------------------------------------------
# Frontier — update (keep / revert)
# ---------------------------------------------------------------------------


def test_update_dominating_scores_kept_and_folded() -> None:
    """A mutation that improves an objective without regressing others is
    KEPT and the best-so-far is updated (component-wise max)."""
    f = Frontier.from_scores(BASELINE)
    mutated = {**BASELINE, "correctness": 0.5, AGGREGATE_KEY: 0.786}
    assert f.update(mutated) is True
    # correctness + aggregate improved; the rest carry forward.
    assert f.best["correctness"] == 0.5
    assert f.best[AGGREGATE_KEY] == 0.786
    assert f.best["retrieval_groundedness"] == BASELINE["retrieval_groundedness"]


def test_update_dominated_scores_reverted_unchanged() -> None:
    """A mutation that regresses an objective without a compensating
    improvement is REVERTED and the best-so-far is left untouched."""
    f = Frontier.from_scores(BASELINE)
    snapshot = dict(f.best)
    # correctness regresses, nothing improves.
    mutated = {**BASELINE, "correctness": 0.2, AGGREGATE_KEY: 0.70}
    assert f.update(mutated) is False
    assert f.best == snapshot


def test_update_tie_reverts() -> None:
    """A tie (no objective improves) does not extend the frontier → revert."""
    f = Frontier.from_scores(BASELINE)
    snapshot = dict(f.best)
    assert f.update(dict(BASELINE)) is False
    assert f.best == snapshot


def test_update_partial_improvement_with_regression_reverts() -> None:
    """Improving one objective while regressing another is dominated → revert."""
    f = Frontier.from_scores(BASELINE)
    mutated = {
        **BASELINE,
        "correctness": 0.6,  # improves
        "retrieval_groundedness": 0.7,  # regresses
        AGGREGATE_KEY: 0.75,
    }
    assert f.update(mutated) is False


# ---------------------------------------------------------------------------
# Frontier — should_keep (static gate decision)
# ---------------------------------------------------------------------------


def test_should_keep_multi_objective_improves_one_no_regress() -> None:
    assert (
        Frontier.should_keep({**BASELINE, "correctness": 0.5, AGGREGATE_KEY: 0.786}, BASELINE)
        is True
    )


def test_should_keep_multi_objective_regresses_one_returns_false() -> None:
    mutated = {**BASELINE, "correctness": 0.6, "retrieval_groundedness": 0.7}
    assert Frontier.should_keep(mutated, BASELINE) is False


def test_should_keep_tie_returns_false() -> None:
    assert Frontier.should_keep(dict(BASELINE), BASELINE) is False


def test_should_keep_empty_frontier_extends() -> None:
    """No best-so-far yet → the first point extends the frontier."""
    assert Frontier.should_keep(BASELINE, {}) is True


def test_should_keep_epsilon_allows_small_regression() -> None:
    """With epsilon>0, a small regression is tolerated if another objective
    improves by more than epsilon."""
    mutated = {
        **BASELINE,
        "correctness": 0.55,  # +0.175
        "retrieval_groundedness": 0.85,  # -0.007, within epsilon=0.05
        AGGREGATE_KEY: 0.78,
    }
    assert Frontier.should_keep(mutated, BASELINE, epsilon=0.05) is True


def test_should_keep_epsilon_requires_improvement_beyond_epsilon() -> None:
    """An improvement of exactly epsilon does not count as 'better' (>)."""
    mutated = {**BASELINE, "correctness": BASELINE["correctness"] + 0.05}
    # epsilon == improvement → not > epsilon → no strict improvement → tie → revert.
    assert Frontier.should_keep(mutated, BASELINE, epsilon=0.05) is False


def test_should_keep_epsilon_blocks_regression_beyond_epsilon() -> None:
    """A regression larger than epsilon dominates even if another objective
    improves."""
    mutated = {
        **BASELINE,
        "correctness": 0.9,  # big improvement
        "retrieval_groundedness": 0.5,  # -0.357, far beyond epsilon=0.05
    }
    assert Frontier.should_keep(mutated, BASELINE, epsilon=0.05) is False


def test_should_keep_pareto_false_uses_aggregate() -> None:
    """pareto=False: keep iff the aggregate improves vs best-so-far aggregate."""
    better_agg = {**BASELINE, AGGREGATE_KEY: 0.80}
    worse_agg = {**BASELINE, AGGREGATE_KEY: 0.70}
    assert Frontier.should_keep(better_agg, BASELINE, pareto=False) is True
    assert Frontier.should_keep(worse_agg, BASELINE, pareto=False) is False
    # A per-judge improvement with a flat-or-worse aggregate does NOT keep
    # in single-objective mode.
    judge_only = {**BASELINE, "correctness": 0.9, AGGREGATE_KEY: 0.70}
    assert Frontier.should_keep(judge_only, BASELINE, pareto=False) is False


def test_should_keep_pareto_false_falls_back_to_first_objective() -> None:
    """Without an aggregate key, single-objective mode uses the first
    configured objective."""
    frontier = {"correctness": 0.5}
    assert Frontier.should_keep({"correctness": 0.6}, frontier, pareto=False) is True
    assert Frontier.should_keep({"correctness": 0.4}, frontier, pareto=False) is False


def test_should_keep_reverts_when_tracked_objective_missing() -> None:
    """A mutation that drops a tracked objective (e.g. safety) fails
    closed — the gate returns False (revert), not silently skipping it."""
    frontier = {"correctness": 0.5, "safety": 0.9}
    mutated = {"correctness": 0.6}  # safety absent from the mutation
    # Without explicit objectives — the normal call path
    assert Frontier.should_keep(mutated, frontier) is False
    # With explicit objectives — also fails closed
    assert Frontier.should_keep(mutated, frontier, objectives=["correctness", "safety"]) is False


def test_should_keep_new_objective_extends_frontier() -> None:
    """A mutation reporting a new objective not in the frontier
    counts as an improvement (extends the frontier)."""
    frontier = {"correctness": 0.5}
    mutated = {"correctness": 0.5, "safety": 0.9}  # safety is new
    assert Frontier.should_keep(mutated, frontier) is True


def test_should_keep_reverts_on_nan_score() -> None:
    """NaN for one objective → fail closed (NaN comparisons are always
    False, so the delta never trips regression or improvement checks)."""
    mutated = {**BASELINE, "correctness": float("nan")}
    assert Frontier.should_keep(mutated, BASELINE) is False


def test_should_keep_reverts_on_infinite_score() -> None:
    """An infinite score → fail closed."""
    mutated = {**BASELINE, "correctness": float("inf")}
    assert Frontier.should_keep(mutated, BASELINE) is False


# ---------------------------------------------------------------------------
# Frontier — serialization
# ---------------------------------------------------------------------------


def test_serialization_round_trip(tmp_path: Path) -> None:
    f = Frontier.from_scores(BASELINE, pareto=False, epsilon=0.01)
    f.update({**BASELINE, "correctness": 0.5, AGGREGATE_KEY: 0.786})
    roundtripped = Frontier.from_dict(f.to_dict())
    assert roundtripped == f
    assert roundtripped.pareto is False
    assert roundtripped.epsilon == 0.01
    assert roundtripped.best == f.best


def test_direction_aware_pareto_minimizes_cost() -> None:
    f = Frontier.from_scores(
        {"quality": 0.8, "cost": 100.0},
        directions={"quality": "maximize", "cost": "minimize"},
    )
    assert f.update({"quality": 0.8, "cost": 80.0}) is True
    assert f.best == {"quality": 0.8, "cost": 80.0}
    assert f.update({"quality": 0.81, "cost": 90.0}) is False


def test_directions_survive_serialization() -> None:
    f = Frontier.from_scores(
        {"quality": 0.8, "cost": 100.0},
        directions={"quality": "maximize", "cost": "minimize"},
    )
    assert Frontier.from_dict(f.to_dict()).directions == f.directions


def test_load_save_round_trip(tmp_path: Path) -> None:
    f = Frontier.from_scores(BASELINE)
    f.update({**BASELINE, "correctness": 0.5, AGGREGATE_KEY: 0.786})
    save_frontier(tmp_path, f)
    loaded = load_frontier(tmp_path)
    assert loaded is not None
    assert loaded == f


def test_load_frontier_missing_returns_none(tmp_path: Path) -> None:
    assert load_frontier(tmp_path) is None


# ---------------------------------------------------------------------------
# Score adapters
# ---------------------------------------------------------------------------


def test_scores_from_eval_and_baseline() -> None:
    report = SimpleNamespace(
        per_judge={"correctness": 0.5, "retrieval_groundedness": 0.9},
        aggregate=0.7,
    )
    assert scores_from_eval(report) == {
        "correctness": 0.5,
        "retrieval_groundedness": 0.9,
        AGGREGATE_KEY: 0.7,
    }
    baseline = SimpleNamespace(
        per_judge={"correctness": 0.3, "retrieval_groundedness": 0.8},
        aggregate=0.55,
    )
    assert scores_from_baseline(baseline) == {
        "correctness": 0.3,
        "retrieval_groundedness": 0.8,
        AGGREGATE_KEY: 0.55,
    }


def test_scores_from_eval_extracts_configured_cost_metrics() -> None:
    report = SimpleNamespace(
        aggregate=0.7,
        cost_metrics={"total_context_chars": 1234.0, "n_rows": 8.0},
        n_rows=8,
    )
    objectives = [
        ParetoObjective(name="quality", source="aggregate"),
        ParetoObjective(name="cost", direction="minimize", source="context_chars"),
    ]
    assert scores_from_eval(report, objectives) == {"quality": 0.7, "cost": 1234.0}


# ---------------------------------------------------------------------------
# gate_decision — the integration (load / init / decide / persist)
# ---------------------------------------------------------------------------


def _gate(
    repo_root: Path,
    *,
    gate_type: str = "frontier",
    epsilon: float = 0.0,
    pareto: bool = True,
    baseline_scores: dict | None = BASELINE,
    baseline_aggregate: float | None = BASELINE[AGGREGATE_KEY],
    mutated_scores: dict | None = None,
    mutated_aggregate: float | None = None,
    action_kind: str = "add_rule",
    eval_failed: bool = False,
    parse_status: str = "ok",
) -> tuple[Decision, Frontier | None]:
    return gate_decision(
        repo_root=repo_root,
        gate_type=gate_type,
        epsilon=epsilon,
        pareto=pareto,
        baseline_scores=baseline_scores,
        baseline_aggregate=baseline_aggregate,
        mutated_scores=mutated_scores,
        mutated_aggregate=mutated_aggregate,
        action_kind=action_kind,
        eval_failed=eval_failed,
        parse_status=parse_status,
    )


def test_gate_frontier_first_round_inits_from_baseline_then_decides(tmp_path: Path) -> None:
    """No frontier file → initialize from baseline, then decide on the mutation."""
    mutated = {**BASELINE, "correctness": 0.5, AGGREGATE_KEY: 0.786}
    decision, frontier = _gate(
        tmp_path,
        mutated_scores=mutated,
        mutated_aggregate=mutated[AGGREGATE_KEY],
    )
    assert decision == Decision.KEEP
    # frontier.json was created and reflects the folded best-so-far.
    assert (tmp_path / "eval" / "runs" / "frontier.json").is_file()
    assert frontier is not None
    assert frontier.best["correctness"] == 0.5
    assert frontier.best[AGGREGATE_KEY] == 0.786


def test_gate_frontier_reverts_regression_vs_kept_round(tmp_path: Path) -> None:
    """THE BUG: a round scoring worse than a previously KEPT round is now
    REVERTED, even though it still beats the frozen baseline.

    Legacy frozen-baseline gate would KEEP round 2 (delta vs baseline > 0);
    the frontier gate REVERTS it because round 2 regresses the
    best-so-far correctness set by round 1.
    """
    # Round 1: improves correctness + aggregate vs baseline → KEEP.
    mutated1 = {**BASELINE, "correctness": 0.5, AGGREGATE_KEY: 0.786}
    decision1, frontier1 = _gate(
        tmp_path,
        mutated_scores=mutated1,
        mutated_aggregate=mutated1[AGGREGATE_KEY],
    )
    assert decision1 == Decision.KEEP
    assert frontier1 is not None
    best_after_r1 = dict(frontier1.best)
    assert best_after_r1["correctness"] == 0.5

    # Round 2: correctness 0.4 < 0.5 (regresses best-so-far) but aggregate
    # 0.752 > baseline 0.744 (would have been KEPT by the old gate).
    mutated2 = {**BASELINE, "correctness": 0.4, AGGREGATE_KEY: 0.752}
    decision2, frontier2 = _gate(
        tmp_path,
        mutated_scores=mutated2,
        mutated_aggregate=mutated2[AGGREGATE_KEY],
    )
    assert decision2 == Decision.REVERT
    # The frontier is unchanged after a revert.
    assert frontier2 is not None
    assert frontier2.best == best_after_r1


def test_gate_frontier_persists_after_keep_and_after_revert(tmp_path: Path) -> None:
    """frontier.json is written after each scored round (keep and revert)."""
    path = tmp_path / "eval" / "runs" / "frontier.json"
    assert not path.is_file()

    # Keep round.
    mutated1 = {**BASELINE, "correctness": 0.5, AGGREGATE_KEY: 0.786}
    _gate(tmp_path, mutated_scores=mutated1, mutated_aggregate=mutated1[AGGREGATE_KEY])
    assert path.is_file()
    kept = json.loads(path.read_text())

    # Revert round — file is rewritten (present + fresh).
    mutated2 = {**BASELINE, "correctness": 0.4, AGGREGATE_KEY: 0.752}
    _gate(tmp_path, mutated_scores=mutated2, mutated_aggregate=mutated2[AGGREGATE_KEY])
    assert path.is_file()
    reverted = json.loads(path.read_text())
    # Best-so-far unchanged after the revert.
    assert reverted["best"] == kept["best"]


def test_gate_frontier_noop_returns_noop_without_frontier_io(tmp_path: Path) -> None:
    """A noop wins over the gate and creates no frontier file."""
    decision, frontier = _gate(tmp_path, action_kind="noop")
    assert decision == Decision.NOOP
    assert frontier is None
    assert not (tmp_path / "eval" / "runs" / "frontier.json").is_file()


def test_gate_frontier_eval_fail_returns_infra_fail(tmp_path: Path) -> None:
    decision, frontier = _gate(tmp_path, eval_failed=True)
    assert decision == Decision.INFRA_FAIL
    assert frontier is None
    assert not (tmp_path / "eval" / "runs" / "frontier.json").is_file()


def test_gate_frontier_missing_mutated_score_returns_infra_fail(tmp_path: Path) -> None:
    decision, frontier = _gate(tmp_path, mutated_scores=None, mutated_aggregate=None)
    assert decision == Decision.INFRA_FAIL
    assert frontier is None


def test_gate_frontier_no_baseline_no_frontier_returns_infra_fail(tmp_path: Path) -> None:
    """No baseline to seed from and no existing frontier → infra fail (we
    cannot make a comparative decision)."""
    decision, frontier = _gate(
        tmp_path,
        baseline_scores=None,
        baseline_aggregate=None,
        mutated_scores={**BASELINE, "correctness": 0.5},
        mutated_aggregate=0.786,
    )
    assert decision == Decision.INFRA_FAIL
    assert frontier is None


def test_gate_frontier_loads_existing_frontier_across_rounds(tmp_path: Path) -> None:
    """A pre-existing frontier.json is loaded (not re-initialized from baseline)."""
    seed = Frontier.from_scores({**BASELINE, "correctness": 0.6, AGGREGATE_KEY: 0.82})
    save_frontier(tmp_path, seed)

    # A mutation that only matches the seeded best → tie → revert.
    mutated = {**BASELINE, "correctness": 0.6, AGGREGATE_KEY: 0.82}
    decision, frontier = _gate(
        tmp_path,
        mutated_scores=mutated,
        mutated_aggregate=mutated[AGGREGATE_KEY],
    )
    assert decision == Decision.REVERT
    assert frontier is not None
    # The frontier kept the seeded best (not overwritten from baseline).
    assert frontier.best["correctness"] == 0.6


# ---------------------------------------------------------------------------
# gate_decision — backward compatibility (gate.type: delta)
# ---------------------------------------------------------------------------


def test_gate_delta_preserves_legacy_keeps_regression(tmp_path: Path) -> None:
    """gate.type=delta reproduces the old frozen-baseline behavior: a round
    that scores worse than a previous KEPT round but still beats the frozen
    baseline is KEPT (the legacy bug, preserved for backward compat)."""
    # Round 1: beats baseline → KEEP.
    mutated1 = {**BASELINE, AGGREGATE_KEY: 0.786}
    decision1, frontier1 = _gate(
        tmp_path,
        gate_type="delta",
        mutated_scores=mutated1,
        mutated_aggregate=mutated1[AGGREGATE_KEY],
    )
    assert decision1 == Decision.KEEP
    assert frontier1 is None  # delta gate does not use the frontier

    # Round 2: worse than round 1 (0.752 < 0.786) but still beats the
    # frozen baseline (0.752 > 0.744) → KEEP under the legacy gate.
    mutated2 = {**BASELINE, AGGREGATE_KEY: 0.752}
    decision2, _ = _gate(
        tmp_path,
        gate_type="delta",
        mutated_scores=mutated2,
        mutated_aggregate=mutated2[AGGREGATE_KEY],
    )
    assert decision2 == Decision.KEEP

    # The frontier file is never written under the delta gate.
    assert not (tmp_path / "eval" / "runs" / "frontier.json").is_file()


def test_gate_delta_reverts_when_below_baseline(tmp_path: Path) -> None:
    mutated = {**BASELINE, AGGREGATE_KEY: 0.70}  # below baseline 0.744
    decision, _ = _gate(
        tmp_path,
        gate_type="delta",
        mutated_scores=mutated,
        mutated_aggregate=mutated[AGGREGATE_KEY],
    )
    assert decision == Decision.REVERT


def test_gate_delta_tie_reverts(tmp_path: Path) -> None:
    """Legacy semantics: a zero delta is not an improvement → revert."""
    decision, _ = _gate(
        tmp_path,
        gate_type="delta",
        mutated_scores=dict(BASELINE),
        mutated_aggregate=BASELINE[AGGREGATE_KEY],
    )
    assert decision == Decision.REVERT


def test_gate_delta_noop_and_infra_fail_match_legacy(tmp_path: Path) -> None:
    assert _gate(tmp_path, gate_type="delta", action_kind="noop")[0] == Decision.NOOP
    assert _gate(tmp_path, gate_type="delta", eval_failed=True)[0] == Decision.INFRA_FAIL


def test_gate_decision_delta_works_with_aggregate_only(tmp_path: Path) -> None:
    """The delta gate only needs mutated_aggregate, not per-objective scores.
    A valid aggregate with absent per-objective scores must NOT become
    INFRA_FAIL — it should use the legacy decide()."""
    decision, frontier = _gate(
        tmp_path,
        gate_type="delta",
        mutated_scores=None,
        mutated_aggregate=0.8,
    )
    assert decision != Decision.INFRA_FAIL
    assert frontier is None
    # 0.8 > baseline 0.744 → KEEP under the legacy gate.
    assert decision == Decision.KEEP


# ---------------------------------------------------------------------------
# load_gate_config
# ---------------------------------------------------------------------------


def test_load_gate_config_reads_section(tmp_path: Path) -> None:
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "config.yaml").write_text(
        "gate:\n  type: delta\n  epsilon: 0.05\n  pareto: false\n",
        encoding="utf-8",
    )
    # load_gate_config resolves harness/config.yaml as a sibling of scaffold/.
    scaffold_root = tmp_path / "scaffold"
    scaffold_root.mkdir()
    cfg = load_gate_config(scaffold_root)
    assert cfg == GateConfig(type="delta", epsilon=0.05, pareto=False)


def test_load_gate_config_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = load_gate_config(tmp_path / "scaffold")
    assert cfg == GateConfig()
    assert cfg.type == "frontier"
    assert cfg.epsilon == 0.0
    assert cfg.pareto == ParetoConfig(enabled=False)


def test_load_gate_config_defaults_when_section_absent(tmp_path: Path) -> None:
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "config.yaml").write_text(
        "runtime_endpoint: x\noptimizer_endpoint: y\njudge_endpoint: z\n"
        "experiments: {runtime: a, eval: b, optimizer: c}\n",
        encoding="utf-8",
    )
    cfg = load_gate_config(tmp_path / "scaffold")
    assert cfg == GateConfig()


def test_load_gate_config_rejects_unknown_field(tmp_path: Path) -> None:
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "config.yaml").write_text(
        "gate:\n  type: frontier\n  bogus: 1\n", encoding="utf-8"
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_gate_config(tmp_path / "scaffold")


def test_gate_config_rejects_negative_epsilon() -> None:
    """A negative epsilon makes ties count as improvements → rejected."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GateConfig(epsilon=-0.1)


def test_gate_config_rejects_nan_epsilon() -> None:
    """A NaN epsilon breaks every comparison in the gate → rejected."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GateConfig(epsilon=float("nan"))


# ---------------------------------------------------------------------------
# Real config smoke test — guards the schema change
# ---------------------------------------------------------------------------


def test_real_harness_config_parses_with_gate() -> None:
    """The repo's harness/config.yaml must parse against the updated
    RuntimeYAML schema (with the new ``gate`` field) and default to the
    frontier gate."""
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "harness" / "config.yaml"
    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = RuntimeYAML.model_validate(raw)
    assert cfg.gate.type == "frontier"
    assert cfg.gate.epsilon == 0.0
    assert cfg.gate.pareto == ParetoConfig(enabled=False)


def test_gate_config_parses_structured_pareto_objectives() -> None:
    cfg = GateConfig.model_validate(
        {
            "pareto": {
                "enabled": True,
                "objectives": [
                    {"name": "quality", "source": "aggregate"},
                    {"name": "cost", "direction": "minimize", "source": "n_rows"},
                ],
            }
        }
    )
    assert cfg.pareto.enabled is True
    assert [o.name for o in cfg.pareto.objectives] == ["quality", "cost"]


# ---------------------------------------------------------------------------
# Latency objective + per-objective epsilon
# ---------------------------------------------------------------------------


def test_latency_source_is_valid_pareto_objective() -> None:
    """``source="latency"`` is a valid ParetoObjective (reads latency_ms_median)."""
    obj = ParetoObjective(name="latency", source="latency", direction="minimize")
    assert obj.source == "latency"
    assert obj.direction == "minimize"
    assert obj.epsilon is None  # defaults to None (fall back to global gate.epsilon)


def test_pareto_objective_accepts_per_objective_epsilon() -> None:
    """An objective can carry its own epsilon override."""
    obj = ParetoObjective(name="latency", source="latency", direction="minimize", epsilon=500.0)
    assert obj.epsilon == 500.0


def test_scores_for_objectives_maps_latency_to_latency_ms_median() -> None:
    """_scores_for_objectives reads ``latency_ms_median`` from cost_metrics
    for a ``source="latency"`` objective."""
    report = SimpleNamespace(
        aggregate=0.9,
        cost_metrics={"latency_ms_median": 1234.5, "total_tokens": 100},
    )
    objectives = [ParetoObjective(name="latency", source="latency", direction="minimize")]
    scores = _scores_for_objectives(report, objectives)
    assert scores == {"latency": 1234.5}


def test_should_keep_uses_per_objective_epsilon() -> None:
    """Per-objective epsilon allows a latency regression that the global
    epsilon (sized for accuracy) would reject.

    accuracy improves by 0.002 (above its epsilon 0.001 → improves_any);
    latency regresses by 100ms (within its epsilon 500ms → not dominated).
    With only the global epsilon 0.001 the 100ms latency regression would
    be dominated and the mutation reverted.
    """
    mutated = {"accuracy": 0.902, "latency": 1100.0}
    frontier = {"accuracy": 0.900, "latency": 1000.0}
    assert Frontier.should_keep(
        mutated,
        frontier,
        epsilon=0.001,
        epsilons={"accuracy": 0.001, "latency": 500.0},
        objectives=["accuracy", "latency"],
        directions={"accuracy": "maximize", "latency": "minimize"},
    )
    # Same mutation with only the global epsilon → reverted (100ms >> 0.001ms).
    assert not Frontier.should_keep(
        mutated,
        frontier,
        epsilon=0.001,
        objectives=["accuracy", "latency"],
        directions={"accuracy": "maximize", "latency": "minimize"},
    )


def test_should_keep_per_objective_epsilon_rejects_beyond_threshold() -> None:
    """A latency regression beyond the per-objective epsilon is rejected
    even when accuracy improves."""
    mutated = {"accuracy": 0.902, "latency": 1600.0}
    frontier = {"accuracy": 0.900, "latency": 1000.0}
    # latency regressed by 600ms, per-objective epsilon is 500ms → dominated
    assert not Frontier.should_keep(
        mutated,
        frontier,
        epsilon=0.001,
        epsilons={"accuracy": 0.001, "latency": 500.0},
        objectives=["accuracy", "latency"],
        directions={"accuracy": "maximize", "latency": "minimize"},
    )
