"""Pareto frontier gate for the ANVIL optimization loop.

Replaces the frozen-baseline delta gate. The old gate (``decide()`` in
:mod:`anvil.loop.decision`) computed ``score_delta = mutated - baseline``
against a single *cached* baseline and kept any positive delta. That
silently allowed regressions: round N+1 could score worse than a
previously KEPT round N yet still be kept, because it beat the original
frozen baseline.

The frontier gate tracks the **best-so-far score per objective** — the
per-judge scores plus the aggregate. A mutation is KEPT only if it
*extends* the frontier: it improves at least one tracked objective
without regressing any other by more than a configurable ``epsilon``.
The frontier persists to ``eval/runs/frontier.json`` and is loaded at
the start of each round, so the comparison is always against the
running best, never the stale frozen baseline.

This mirrors the Pareto frontier idea in ``meta-harness`` (the
``text_classification`` benchmark's ``compute_pareto_frontier``: a point
is kept iff it is not dominated by any point on the frontier), flattened
to "dominate the single best-so-far vector" because the loop keeps one
mutation per round.

Public surface:

* :class:`Frontier` — best-so-far per objective + the gate decision.
* :func:`gate_decision` — load/init/decide/persist, called by
  :mod:`anvil.loop.round` once per round.
* :func:`load_gate_config` — read the ``gate`` section of
  ``harness/config.yaml`` (falls back to defaults if absent).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from anvil.loop.decision import Decision, decide
from anvil.runtime.loader import default_runtime_config_path
from anvil.runtime.models import GateConfig, ParetoConfig, ParetoObjective

# Key under which the aggregate score is tracked alongside the per-judge
# scores. The objectives of the frontier are the per-judge keys plus this.
AGGREGATE_KEY = "aggregate"

# Default epsilon — a strict positive delta is required to extend the
# frontier (a tie does not extend it).
DEFAULT_EPSILON = 0.0


def _shared_objectives(a: dict[str, float], b: dict[str, float]) -> list[str]:
    """Keys present in both dicts, in ``b``'s order then ``a``'s extras."""
    out = [k for k in b if k in a]
    out.extend(k for k in a if k not in b)
    return out


def _all_finite(scores: dict[str, float]) -> bool:
    """True if every value in ``scores`` is a finite number."""
    return all(math.isfinite(v) for v in scores.values())


class Frontier:
    """Best-so-far score per objective + the keep/revert gate decision.

    Objectives are the keys of the scores dicts — typically the per-judge
    scores (``correctness``, ``retrieval_groundedness``,
    ``refusal_appropriateness``) plus the ``aggregate``. The frontier
    stores the component-wise maximum across all KEPT rounds.

    The class is pure: it holds no file handles and does no I/O.
    Persistence is handled by :func:`load_frontier` / :func:`save_frontier`
    and :func:`gate_decision`.
    """

    def __init__(
        self,
        best: dict[str, float] | None = None,
        *,
        objectives: list[str] | None = None,
        pareto: bool = True,
        directions: dict[str, str] | None = None,
        sources: dict[str, str] | None = None,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.best: dict[str, float] = {k: float(v) for k, v in best.items()} if best else {}
        if objectives is not None:
            self._objectives: list[str] = list(objectives)
        elif self.best:
            self._objectives = list(self.best.keys())
        else:
            self._objectives = []
        self.pareto = pareto
        self.directions = {obj: (directions or {}).get(obj, "maximize") for obj in self._objectives}
        self.sources = {obj: (sources or {}).get(obj, obj) for obj in self._objectives}
        self.epsilon = float(epsilon)

    @property
    def objectives(self) -> list[str]:
        return list(self._objectives)

    @classmethod
    def from_scores(
        cls,
        scores: dict[str, float],
        *,
        pareto: bool = True,
        directions: dict[str, str] | None = None,
        sources: dict[str, str] | None = None,
        epsilon: float = DEFAULT_EPSILON,
    ) -> Frontier:
        """Initialize a frontier whose best-so-far IS ``scores`` (round 1)."""
        return cls(
            best=scores,
            pareto=pareto,
            directions=directions,
            sources=sources,
            epsilon=epsilon,
        )

    @staticmethod
    def should_keep(
        mutated_scores: dict[str, float],
        current_frontier: dict[str, float],
        *,
        epsilon: float = DEFAULT_EPSILON,
        pareto: bool = True,
        objectives: list[str] | None = None,
        directions: dict[str, str] | None = None,
    ) -> bool:
        """Gate decision: keep iff the mutation extends the frontier.

        ``pareto=True`` (multi-objective): keep iff at least one objective
        improves by more than ``epsilon`` AND no objective regresses by
        more than ``epsilon``. This is Pareto dominance of the mutation
        over the best-so-far vector.

        ``pareto=False`` (single-objective): keep iff the aggregate
        improves over the best-so-far aggregate by more than ``epsilon``
        — still measured against the frontier, never the frozen baseline.

        A tie (no objective improves) does not extend the frontier and
        returns ``False`` (revert). This matches the legacy ``decide()``
        semantics where a zero gradient is not an improvement.

        An objective that the frontier has no best-so-far for counts as
        an extension (improves). An objective among the frontier's
        tracked objectives that the mutation does not report fails
        closed (revert): a mutation that drops a previously-tracked
        objective must not be KEPT while hiding the regression.

        Non-finite score values (NaN, +inf, -inf) cause the gate to fail
        closed (revert): NaN comparisons are always False in Python, so a
        NaN delta would never trip the regression or improvement checks
        and the mutation could be silently KEPT.
        """
        if objectives is not None:
            objs = list(objectives)
        else:
            # Union of frontier and mutation keys. Frontier keys first so
            # tracked objectives are always checked (missing → fail closed).
            objs = list(current_frontier.keys())
            for k in mutated_scores:
                if k not in current_frontier:
                    objs.append(k)
        if not objs:
            return False

        # Reject non-finite scores (NaN/inf) — fail closed (revert). NaN
        # comparisons are always False in Python, so a NaN delta never
        # trips the regression or improvement checks and the mutation
        # could be silently KEPT.
        if not _all_finite(mutated_scores):
            return False

        if not pareto:
            # Single-objective: the aggregate (or the first objective if
            # aggregate is absent) vs the best-so-far value for it.
            key = AGGREGATE_KEY if AGGREGATE_KEY in objs else objs[0]
            new = mutated_scores.get(key)
            if new is None:
                return False
            cur = current_frontier.get(key)
            if cur is None:
                return True  # no best-so-far yet → extends
            return new - cur > epsilon

        # Multi-objective Pareto dominance over the best-so-far vector.
        improves_any = False
        for obj in objs:
            new = mutated_scores.get(obj)
            if new is None:
                return False  # mutation dropped a tracked objective → fail closed
            cur = current_frontier.get(obj)
            if cur is None:
                improves_any = True  # frontier has no best yet → extends
                continue
            direction = (directions or {}).get(obj, "maximize")
            delta = (new - cur) if direction == "maximize" else (cur - new)
            if delta < -epsilon:
                return False  # regressed beyond epsilon → dominated → revert
            if delta > epsilon:
                improves_any = True
        return improves_any

    def update(self, scores: dict[str, float]) -> bool:
        """Apply the gate to ``scores`` and, on KEEP, fold it into best-so-far.

        Returns ``True`` if the mutation was KEPT (it extended the
        frontier), ``False`` if it was dominated (revert — best-so-far
        is left untouched).

        On the first call against an empty frontier, the incoming scores
        seed both the objectives and the best-so-far (the first point
        always extends an empty frontier).

        Non-finite score values (NaN, +inf, -inf) cause an immediate
        revert — they never enter the best-so-far.
        """
        # Reject non-finite scores before tracking objectives or
        # comparing — a NaN/inf must never enter the best-so-far.
        if not _all_finite(scores):
            return False

        # Track any newly-seen objectives so the gate compares over the
        # full set the loop has ever reported.
        if not self._objectives:
            self._objectives = list(scores.keys())
        else:
            for k in scores:
                if k not in self._objectives:
                    self._objectives.append(k)
                    self.directions[k] = "maximize"
                    self.sources[k] = k

        kept = self.should_keep(
            scores,
            self.best,
            epsilon=self.epsilon,
            pareto=self.pareto,
            objectives=self._objectives,
            directions=self.directions,
        )
        if kept:
            for obj in self._objectives:
                if obj not in scores:
                    continue
                cur = self.best.get(obj)
                better = cur is None or (
                    scores[obj] > cur
                    if self.directions.get(obj, "maximize") == "maximize"
                    else scores[obj] < cur
                )
                if better:
                    self.best[obj] = float(scores[obj])
        return kept

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": {k: float(v) for k, v in self.best.items()},
            "objectives": list(self._objectives),
            "pareto": self.pareto,
            "directions": dict(self.directions),
            "sources": dict(self.sources),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Frontier:
        return cls(
            best={k: float(v) for k, v in raw.get("best", {}).items()},
            objectives=list(raw.get("objectives", [])),
            pareto=bool(raw.get("pareto", True)),
            directions=dict(raw.get("directions", {})),
            sources=dict(raw.get("sources", {})),
            epsilon=float(raw.get("epsilon", DEFAULT_EPSILON)),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frontier):
            return NotImplemented
        return (
            self.best == other.best
            and self._objectives == other._objectives
            and self.pareto == other.pareto
            and self.directions == other.directions
            and self.sources == other.sources
            and self.epsilon == other.epsilon
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Frontier(best={self.best!r}, objectives={self._objectives!r}, "
            f"pareto={self.pareto}, epsilon={self.epsilon})"
        )


# ---------------------------------------------------------------------------
# Score-dict adapters
# ---------------------------------------------------------------------------


def _scores_for_objectives(report: Any, objectives: list[ParetoObjective]) -> dict[str, float]:
    scores: dict[str, float] = {}
    cost_metrics = getattr(report, "cost_metrics", {}) or {}
    for objective in objectives:
        if objective.source == "aggregate":
            value = report.aggregate
        else:
            metric = {
                "tokens": "total_tokens",
                "context_chars": "total_context_chars",
                "n_rows": "n_rows",
            }[objective.source]
            if metric == "n_rows" and metric not in cost_metrics:
                value = getattr(report, "n_rows", getattr(report, "n_examples", None))
            else:
                value = cost_metrics.get(metric)
            if value is None:
                raise ValueError(
                    f"Pareto objective {objective.name!r} requires unavailable cost metric {metric!r}"
                )
        scores[objective.name] = float(value)
    return scores


def scores_from_eval(
    eval_report: Any, objectives: list[ParetoObjective] | None = None
) -> dict[str, float]:
    """Extract configured objectives, or legacy judge scores + aggregate."""
    if objectives is not None:
        return _scores_for_objectives(eval_report, objectives)
    scores: dict[str, float] = {k: float(v) for k, v in eval_report.per_judge.items()}
    scores[AGGREGATE_KEY] = float(eval_report.aggregate)
    return scores


def scores_from_baseline(
    baseline: Any, objectives: list[ParetoObjective] | None = None
) -> dict[str, float]:
    """Per-judge scores + aggregate from a :class:`CachedBaseline`."""
    if objectives is not None:
        return _scores_for_objectives(baseline, objectives)
    scores: dict[str, float] = {k: float(v) for k, v in baseline.per_judge.items()}
    scores[AGGREGATE_KEY] = float(baseline.aggregate)
    return scores


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def frontier_path(repo_root: Path | str) -> Path:
    return Path(repo_root) / "eval" / "runs" / "frontier.json"


def load_frontier(repo_root: Path | str) -> Frontier | None:
    """Load the persisted frontier, or ``None`` if it does not exist yet."""
    path = frontier_path(repo_root)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Frontier.from_dict(raw)


def save_frontier(repo_root: Path | str, frontier: Frontier) -> Path:
    """Persist ``frontier`` to ``eval/runs/frontier.json``. Returns the path."""
    path = frontier_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(frontier.to_dict(), indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    dashboard_path = Path(repo_root) / "data" / "frontier.json"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(payload, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_gate_config(scaffold_root: Path | str) -> GateConfig:
    """Read the ``gate`` section of ``harness/config.yaml``.

    Falls back to :class:`GateConfig` defaults (``type=frontier``,
    ``epsilon=0.0``, Pareto disabled) when the file or the section is
    absent, so the loop keeps running on a repo that predates the gate
    config. Only the ``gate`` section is validated here; the full-file
    ``extra="forbid"`` check is enforced by the runtime loader.
    """
    path = default_runtime_config_path(Path(scaffold_root))
    if not path.is_file():
        return GateConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return GateConfig.model_validate(raw.get("gate") or {})


# ---------------------------------------------------------------------------
# Gate integration — called by anvil.loop.round once per round
# ---------------------------------------------------------------------------


def gate_decision(
    *,
    repo_root: Path | str,
    gate_type: str,
    epsilon: float,
    pareto: ParetoConfig | bool,
    baseline_scores: dict[str, float] | None,
    baseline_aggregate: float | None,
    mutated_scores: dict[str, float] | None,
    mutated_aggregate: float | None,
    action_kind: str,
    eval_failed: bool,
    parse_status: str,
) -> tuple[Decision, Frontier | None]:
    """Compute the round's terminal decision via the configured gate.

    Returns ``(decision, frontier)`` where ``frontier`` is the
    (possibly updated) :class:`Frontier` for the ``frontier`` gate, or
    ``None`` for the ``delta`` gate / noop / infra-fail paths.

    Ordering mirrors :func:`anvil.loop.decision.decide`:

    1. An explicit ``noop`` from the optimizer wins over everything.
    2. An eval-side infrastructure failure (or a missing mutated
       *aggregate*) beats any score consideration → ``INFRA_FAIL``.
    3. ``gate_type="delta"`` → legacy :func:`decide` (needs only the
       aggregate; checked before per-objective scores).
    4. ``gate_type="frontier"`` → per-objective scores required; a
       missing ``mutated_scores`` → ``INFRA_FAIL``, otherwise the
       frontier gate decides KEEP vs REVERT.

    For ``gate_type="delta"`` the legacy frozen-baseline behavior is
    reproduced exactly via :func:`decide`; the frontier is not used.

    For ``gate_type="frontier"``: load the persisted frontier (or
    initialize it from the baseline on the first scored round), apply
    :meth:`Frontier.update`, and persist. A mutation that regresses
    any tracked objective (beyond ``epsilon``) without a compensating
    improvement is REVERTED — the fix for the silent-regression bug.
    """
    # 1. noop wins over everything.
    if action_kind == "noop":
        return Decision.NOOP, None

    # 2. eval failure / missing aggregate → infra fail (no gate, no frontier
    #    I/O). Only mutated_aggregate is checked here: the delta gate needs
    #    only the aggregate, so per-objective scores are validated later for
    #    the frontier gate specifically.
    if eval_failed or mutated_aggregate is None:
        return Decision.INFRA_FAIL, None

    # 3a. Legacy frozen-baseline gate — preserved verbatim for backward compat.
    #     Placed before the mutated_scores check: the delta gate needs only
    #     mutated_aggregate, not per-objective scores.
    if gate_type == "delta":
        score_delta = (
            mutated_aggregate - baseline_aggregate if baseline_aggregate is not None else None
        )
        decision = decide(
            score_delta=score_delta,
            action_kind=action_kind,
            parse_status=parse_status,
            eval_failed=False,
        )
        return decision, None

    # 3b. Pareto frontier gate (default).
    #     Per-objective scores are required for the frontier gate.
    if mutated_scores is None:
        return Decision.INFRA_FAIL, None

    if isinstance(pareto, bool):
        pareto_enabled = pareto
        objectives = None
        directions: dict[str, str] = {}
        sources: dict[str, str] = {}
    else:
        pareto_enabled = pareto.enabled
        objectives = (
            [objective.name for objective in pareto.objectives] or [AGGREGATE_KEY]
            if pareto.enabled
            else [AGGREGATE_KEY]
        )
        directions = (
            {objective.name: objective.direction for objective in pareto.objectives}
            or {AGGREGATE_KEY: "maximize"}
            if pareto.enabled
            else {AGGREGATE_KEY: "maximize"}
        )
        sources = (
            {objective.name: objective.source for objective in pareto.objectives}
            or {AGGREGATE_KEY: AGGREGATE_KEY}
            if pareto.enabled
            else {AGGREGATE_KEY: AGGREGATE_KEY}
        )

    frontier = load_frontier(repo_root)
    if frontier is None:
        # First scored round: initialize the frontier from the baseline.
        if baseline_scores is None:
            # No baseline to seed from and no existing frontier — we cannot
            # make a comparative decision. Treat as an infra failure rather
            # than silently keeping or reverting.
            return Decision.INFRA_FAIL, None
        frontier = Frontier.from_scores(
            baseline_scores,
            pareto=pareto_enabled,
            directions=directions,
            sources=sources,
            epsilon=epsilon,
        )
        save_frontier(repo_root, frontier)
    else:
        # Adopt the current gate config rather than trusting a stale file.
        frontier.pareto = pareto_enabled
        frontier.epsilon = epsilon
        if objectives is not None:
            frontier._objectives = objectives
            frontier.directions = directions
            frontier.sources = sources

    kept = frontier.update(mutated_scores)
    # Persist after every scored round (KEEP updates the best-so-far; REVERT
    # rewrites the unchanged frontier so the file is present + fresh).
    save_frontier(repo_root, frontier)
    return (Decision.KEEP if kept else Decision.REVERT), frontier
