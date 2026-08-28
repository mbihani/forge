"""Cached baseline for fast score-delta calculation.

Running ``mlflow.genai.evaluate`` against the parent branch every
round to compute ``score_delta_vs_parent`` doubles the wall-clock and
the rate-limit hit. Instead, we cache the parent's aggregate and only
recompute when explicitly requested (``--refresh-baseline``) or when
external dependencies change (model endpoint, scorer set,
golden-set rows).

Cache file: ``eval/runs/baseline.json``.

Schema::

    {
      "scaffold_commit_sha": "<40-char SHA>",
      "evaluated_at": "<UTC ISO8601>",
      "mode": "standard",
      "scorers": ["correctness", "retrieval_groundedness", ...],
      "runtime_endpoint": "databricks-claude-sonnet-4-6",
      "judge_endpoint": "databricks-claude-sonnet-4-6",
      "aggregate": 0.861,
      "per_judge": {...},
      "per_bucket": {...},
      "n_examples": 12,
      "mlflow_run_id": "..."
    }

If any field of the requesting context (mode / scorers / endpoints)
differs from the cache header, the cache is stale and must be
refreshed before producing a delta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anvil.runtime.models import ScorerConfig

if TYPE_CHECKING:
    # EvalReport lives in the sibling runner module. Referenced under
    # TYPE_CHECKING only so this module stays import-light (no mlflow /
    # openai pulled in just for a type hint). ``report_to_baseline``
    # reads attributes off the report, so it is duck-typed at runtime.
    from anvil.eval.runner import EvalReport


def compute_scorer_fingerprint(scorer_configs: list[ScorerConfig]) -> str:
    """Compute a stable JSON fingerprint of the active scorer configs.

    Captures the full scorer specification (name, type, weight,
    check_function) so a weight change or check_function swap invalidates
    a cached baseline even when the scorer names are unchanged. The list
    is sorted by name for deterministic output.

    Storing the fingerprint in :class:`CachedBaseline` closes the
    comparability hole where a cached uniform-weight baseline stayed
    "compatible" after weights changed — the loop would then compare a
    new weighted aggregate against an old uniform-weight aggregate and
    make an invalid frontier decision.
    """
    specs = sorted(
        [
            {
                "name": c.name,
                "type": c.type,
                "weight": c.weight,
                "check_function": c.check_function,
            }
            for c in scorer_configs
        ],
        key=lambda s: s["name"],
    )
    return json.dumps(specs, sort_keys=True)


@dataclass(frozen=True)
class CachedBaseline:
    scaffold_commit_sha: str
    evaluated_at: str
    mode: str
    scorers: list[str]
    runtime_endpoint: str
    judge_endpoint: str
    aggregate: float
    per_judge: dict[str, float] = field(default_factory=dict)
    per_bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    n_examples: int = 0
    mlflow_run_id: str | None = None
    # JSON fingerprint of the scorer configs that produced this baseline
    # (see :func:`compute_scorer_fingerprint`). Empty on baselines written
    # before this field existed — :func:`is_compatible` treats an empty
    # fingerprint on either side as "not checked" for backward compat.
    scorer_fingerprint: str = ""
    cost_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "scaffold_commit_sha": self.scaffold_commit_sha,
            "evaluated_at": self.evaluated_at,
            "mode": self.mode,
            "scorers": list(self.scorers),
            "runtime_endpoint": self.runtime_endpoint,
            "judge_endpoint": self.judge_endpoint,
            "aggregate": self.aggregate,
            "per_judge": dict(self.per_judge),
            "per_bucket": {k: dict(v) for k, v in self.per_bucket.items()},
            "n_examples": self.n_examples,
            "mlflow_run_id": self.mlflow_run_id,
            "scorer_fingerprint": self.scorer_fingerprint,
        }
        # Keep the historical on-disk schema byte-for-byte compatible for
        # baselines created before cost tracking, while retaining metrics
        # whenever the eval report supplies them.
        if self.cost_metrics:
            payload["cost_metrics"] = dict(self.cost_metrics)
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CachedBaseline:
        return cls(
            scaffold_commit_sha=raw["scaffold_commit_sha"],
            evaluated_at=raw["evaluated_at"],
            mode=raw["mode"],
            scorers=list(raw["scorers"]),
            runtime_endpoint=raw["runtime_endpoint"],
            judge_endpoint=raw["judge_endpoint"],
            aggregate=float(raw["aggregate"]),
            per_judge=dict(raw.get("per_judge", {})),
            per_bucket={k: dict(v) for k, v in raw.get("per_bucket", {}).items()},
            n_examples=int(raw.get("n_examples", 0)),
            mlflow_run_id=raw.get("mlflow_run_id"),
            scorer_fingerprint=raw.get("scorer_fingerprint", ""),
            cost_metrics={k: float(v) for k, v in raw.get("cost_metrics", {}).items()},
        )


def baseline_path(repo_root: Path | str) -> Path:
    return Path(repo_root) / "eval" / "runs" / "baseline.json"


def load_baseline(repo_root: Path | str) -> CachedBaseline | None:
    path = baseline_path(repo_root)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CachedBaseline.from_dict(raw)


def save_baseline(repo_root: Path | str, baseline: CachedBaseline) -> Path:
    path = baseline_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def is_compatible(
    cached: CachedBaseline,
    *,
    mode: str,
    scorers: list[str],
    runtime_endpoint: str,
    judge_endpoint: str,
    scorer_fingerprint: str = "",
) -> bool:
    """Return True if ``cached`` is comparable with the requesting context.

    When both sides carry a non-empty ``scorer_fingerprint``, the
    fingerprints must match — a weight or check_function change
    invalidates the comparison even if the scorer names are unchanged.
    An empty fingerprint on either side (e.g. a baseline written before
    this field existed) skips the fingerprint check for backward
    compatibility.
    """
    if (
        cached.mode != mode
        or list(cached.scorers) != list(scorers)
        or cached.runtime_endpoint != runtime_endpoint
        or cached.judge_endpoint != judge_endpoint
    ):
        return False
    # When both sides carry a non-empty fingerprint, they must match; an
    # empty fingerprint on either side skips the check (backward compat).
    return not (
        cached.scorer_fingerprint
        and scorer_fingerprint
        and cached.scorer_fingerprint != scorer_fingerprint
    )


def report_to_baseline(
    report: EvalReport,
    *,
    scaffold_commit_sha: str,
    runtime_endpoint: str,
    judge_endpoint: str,
) -> CachedBaseline:
    """Convert an :class:`EvalReport` into a storable :class:`CachedBaseline`.

    ``evaluate_branch`` returns an ``EvalReport`` — the eval runner's
    own schema (``n_rows`` / ``run_id`` / ``failures`` / ``trace_ids``).
    The loop's keep/revert gate, by contrast, reads a ``CachedBaseline``
    (``n_examples`` / ``mlflow_run_id``) from
    ``eval/runs/baseline.json``. This function bridges the two schemas
    so a fresh scaffold can produce the baseline the gate needs without
    re-running the eval every round.

    The two schemas intentionally diverge on two field names:
    ``EvalReport.n_rows`` → ``CachedBaseline.n_examples`` and
    ``EvalReport.run_id`` → ``CachedBaseline.mlflow_run_id``. The
    eval-only fields (``failures`` / ``experiment_id`` / ``trace_ids``)
    are dropped — the cache header only carries what
    :func:`is_compatible` and :func:`load_baseline` consume.

    The three fields the eval does not know — ``scaffold_commit_sha``
    (git), ``runtime_endpoint`` and ``judge_endpoint``
    (``harness/config.yaml``) — are passed in by the caller, keeping
    this plane git-agnostic and config-source-agnostic (see the module
    docstring: cross-plane knowledge is forbidden here).
    """
    return CachedBaseline(
        scaffold_commit_sha=scaffold_commit_sha,
        evaluated_at=report.evaluated_at,
        mode=report.mode,
        scorers=list(report.scorers),
        runtime_endpoint=runtime_endpoint,
        judge_endpoint=judge_endpoint,
        aggregate=report.aggregate,
        per_judge=dict(report.per_judge),
        per_bucket={k: dict(v) for k, v in report.per_bucket.items()},
        n_examples=report.n_rows,
        mlflow_run_id=report.run_id,
        scorer_fingerprint=getattr(report, "scorer_fingerprint", ""),
        cost_metrics=dict(getattr(report, "cost_metrics", {})),
    )
