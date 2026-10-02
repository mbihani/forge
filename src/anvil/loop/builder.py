"""Build the optimizer's round prompt from on-disk artifacts.

Composes a single user message that contains:

* The **goal of this round** — beat the cached baseline aggregate.
* A pointer to the **scaffold tree** the optimizer must read.
* A summary of the **cached baseline**: aggregate, per-judge,
  per-bucket, plus the most-failed example_ids.
* The **last K critiques** (by default 3) so the optimizer can avoid
  re-proposing reverted vectors.
* The **action contract reminder** — the parser only accepts a single
  ```json-action`` block.

The raw round prompt at ``prompts/anvil-round.md`` is the SYSTEM-side
contract; it stays in cwd and the Claude Code subprocess reads it via
its own ``Read`` tool. The string this builder emits is the USER turn:
small, specific, links to where the rich data lives.
"""

from __future__ import annotations

import json
from pathlib import Path

from anvil.runtime.models import ParetoObjective

_PROMPT_TEMPLATE = """\
# Round {round_id}

You are running optimizer round {round_id} on the ANVIL repo at this
working directory. The round's purpose is to propose ONE structural
mutation to ``scaffold/`` that beats the cached parent baseline.

## Cached parent baseline
- aggregate: {baseline_aggregate:.3f}  (mode={baseline_mode}, n={n_examples})
- scorers: {scorers}
- per-judge: {per_judge}
- per-bucket correctness: {per_bucket_correctness}

The most-failed examples in the parent run (read the full list in
``{baseline_failures_path}``):

{failures_summary}

## Gate objectives

{objectives_block}

## What you must do

1. Read ``prompts/anvil-round.md`` — the action contract.
2. Read ``eval/runs/baseline.json`` and the failure traces it points to.
3. Read every active rule + skill from ``scaffold/harness.yaml`` (and
   the most recent ~3 critiques in ``scaffold/memory/``) so your
   mutation does not clash with what's already there.
4. Pick ONE mutation and emit its action JSON block at the end of the
   session. ``noop`` with a thoughtful rationale is preferable to a
   risky guess.

You have at most 30 turns.

## Recent critiques (last {critique_lookback})

{critiques_block}
"""


def build_round_prompt(
    *,
    repo_root: Path | str,
    round_id: int,
    baseline: dict | None,
    critique_lookback: int = 3,
    objectives: list[ParetoObjective] | None = None,
) -> str:
    repo_root = Path(repo_root)

    if baseline is None:
        baseline_aggregate = float("nan")
        baseline_mode = "none"
        n_examples = 0
        scorers_str = "(no cached baseline yet)"
        per_judge_str = "(none)"
        per_bucket_str = "(none)"
        failures_summary = "(no failures recorded — first round)"
        baseline_failures_path = "(none)"
    else:
        baseline_aggregate = float(baseline.get("aggregate", 0.0))
        baseline_mode = baseline.get("mode", "?")
        n_examples = int(baseline.get("n_examples", 0))
        scorers_str = ", ".join(baseline.get("scorers", []))
        per_judge_str = ", ".join(f"{k}={v:.3f}" for k, v in baseline.get("per_judge", {}).items())
        per_bucket_str = ", ".join(
            f"{bucket}={scores.get('correctness', 0.0):.2f}"
            for bucket, scores in baseline.get("per_bucket", {}).items()
        )
        baseline_failures_path = "eval/runs/baseline.json"
        # The baseline cache itself doesn't carry per-row failures
        # (it has aggregate + per_bucket only); point at the
        # underlying eval JSON via the mlflow_run_id, and at the
        # per-bucket aggregate as the actionable failure summary.
        failures_summary = _failure_summary_from_baseline(baseline)

    critiques_block = _format_critiques(repo_root, critique_lookback)
    objectives_block = _format_objectives(repo_root, baseline, objectives)

    return _PROMPT_TEMPLATE.format(
        round_id=round_id,
        baseline_aggregate=baseline_aggregate,
        baseline_mode=baseline_mode,
        n_examples=n_examples,
        scorers=scorers_str,
        per_judge=per_judge_str,
        per_bucket_correctness=per_bucket_str,
        baseline_failures_path=baseline_failures_path,
        failures_summary=failures_summary,
        objectives_block=objectives_block,
        critique_lookback=critique_lookback,
        critiques_block=critiques_block,
    )


def _failure_summary_from_baseline(baseline: dict) -> str:
    """Turn baseline's per_bucket into a one-bucket-per-line failure cluster summary."""
    lines: list[str] = []
    for bucket, scores in baseline.get("per_bucket", {}).items():
        worst = min(scores.items(), key=lambda kv: kv[1])
        lines.append(f"  - {bucket}: worst judge = {worst[0]}={worst[1]:.2f}")
    return "\n".join(lines) if lines else "  (no per-bucket data)"


def _format_objectives(
    repo_root: Path, baseline: dict | None, objectives: list[ParetoObjective] | None
) -> str:
    """Describe what the gate keeps on, so the optimizer targets it.

    Without Pareto objectives the gate is single-objective on the
    aggregate. With them, list each objective's direction + tolerance
    alongside the baseline value and the best-so-far from
    ``eval/runs/frontier.json`` (the gate compares against the frontier,
    not the frozen baseline).
    """
    if not objectives:
        return "- aggregate (maximize): the only objective the gate keeps on."
    cost_metrics = (baseline or {}).get("cost_metrics") or {}
    frontier_path = repo_root / "eval" / "runs" / "frontier.json"
    best = _read_json(frontier_path).get("best", {}) if frontier_path.is_file() else {}
    lines = [
        "A mutation is KEPT only if at least one objective improves by more "
        "than its epsilon AND no objective regresses by more than its epsilon:",
        "",
    ]
    for obj in objectives:
        metric = "aggregate" if obj.source == "aggregate" else _COST_METRIC[obj.source]
        base = baseline.get("aggregate") if baseline and obj.source == "aggregate" else None
        if base is None:
            base = cost_metrics.get(metric)
        eps = "gate default" if obj.epsilon is None else f"{obj.epsilon:g}"
        lines.append(
            f"- {obj.name} ({obj.direction} {metric}; epsilon={eps}): "
            f"baseline={_fmt(base)}, best-so-far={_fmt(best.get(obj.name))}"
        )
    if cost_metrics:
        lines += ["", "Baseline cost metrics: " + ", ".join(
            f"{k}={_fmt(v)}" for k, v in sorted(cost_metrics.items())
        )]
    return "\n".join(lines)


_COST_METRIC = {
    "tokens": "total_tokens",
    "context_chars": "total_context_chars",
    "n_rows": "n_rows",
    "latency": "latency_ms_median",
}


def _fmt(v: object) -> str:
    if not isinstance(v, (int, float)):
        return "n/a"
    return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def _format_critiques(repo_root: Path, k: int) -> str:
    memory_dir = repo_root / "scaffold" / "memory"
    if not memory_dir.is_dir():
        return "(no memory directory yet)"
    files = sorted(memory_dir.glob("round_*_critique.md"), reverse=True)[:k]
    if not files:
        return "(no critiques yet — this is round 1)"
    blocks: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        # Keep just the first ~30 lines per critique so the prompt stays tight.
        head = "\n".join(text.splitlines()[:30])
        blocks.append(f"### {path.name}\n\n{head}")
    return "\n\n".join(blocks)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
