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
import statistics
from pathlib import Path
from typing import Any

from anvil.runtime.models import MODEL_LEVER, LeverSpec, ParetoObjective

_PROMPT_TEMPLATE = """\
# Round {round_id}

You are running optimizer round {round_id} on the ANVIL repo at this
working directory. The round's purpose is to propose {mutation_budget}
to ``scaffold/`` that beats the cached parent baseline.

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

## Runtime levers

{levers_block}

## What the agent's existing evals say

{agent_evals_block}

## What you must do

1. Read ``prompts/anvil-round.md`` — the action contract.
2. Read ``eval/runs/baseline.json`` and the failure traces it points to.
   The round is scored by the agent's OWN judges listed above (from
   ``eval/agent_evals/review.md``) — target what they and the human
   feedback point to, not your own idea of quality.
3. Read every active rule + skill from ``scaffold/harness.yaml`` (and
   the most recent ~3 critiques in ``scaffold/memory/``) so your
   mutation does not clash with what's already there.
4. {pick_instruction} Emit its action JSON block at the end of the
   session. ``noop`` with a thoughtful rationale is preferable to a
   risky guess.

You have at most {max_turns} turns.

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
    max_mutations: int = 1,
    max_turns: int = 30,
    lever_specs: dict[str, LeverSpec] | None = None,
    lever_values: dict[str, Any] | None = None,
    model_prices: dict[str, dict[str, float]] | None = None,
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
    levers_block = _format_levers(repo_root, lever_specs or {}, lever_values or {}, model_prices)
    agent_evals_block = _format_agent_evals(repo_root)

    if max_mutations <= 1:
        mutation_budget = "ONE structural mutation"
        pick_instruction = "Pick ONE mutation."
    else:
        mutation_budget = (
            f"ONE structural mutation, or a `compound` of up to {max_mutations} "
            "mutations gated together"
        )
        pick_instruction = (
            "Pick ONE mutation — or, when you predict changes COMBINE (e.g. two "
            "independent latency cuts each smaller than the epsilon, or a cheaper "
            f"model plus a prompt edit that protects quality on it), a `compound` "
            f"of 2..{max_mutations} steps with a `synergy` explaining why the "
            "combination beats each step alone. Prefer a single step when one "
            "suffices: a compound is harder to attribute."
        )

    return _PROMPT_TEMPLATE.format(
        mutation_budget=mutation_budget,
        pick_instruction=pick_instruction,
        max_turns=max_turns,
        levers_block=levers_block,
        agent_evals_block=agent_evals_block,
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


_AGENT_EVALS_PROMPT_CHARS = 4000


def _format_agent_evals(repo_root: Path) -> str:
    """The reviewed agent evals (``eval/agent_evals/review.md``), truncated.

    Tells the optimizer which judges score the round and what they and the
    human feedback point to. Absent review → say so (the run then relies on
    the metrics agreed with the user in ``eval.agent_evals.user_metrics``).
    """
    path = repo_root / "eval" / "agent_evals" / "review.md"
    if not path.is_file():
        return (
            "(no reviewed agent evals — the round is scored with the metrics agreed with "
            "the user in harness/config.yaml > eval.agent_evals.user_metrics)"
        )
    text = path.read_text(encoding="utf-8").strip()
    if len(text) > _AGENT_EVALS_PROMPT_CHARS:
        text = (
            text[:_AGENT_EVALS_PROMPT_CHARS].rstrip()
            + "\n\n(truncated — read ``eval/agent_evals/review.md`` for the rest)"
        )
    return text


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
        lines += [
            "",
            "Baseline cost metrics: "
            + ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(cost_metrics.items())),
        ]
    return "\n".join(lines)


def _format_levers(
    repo_root: Path,
    specs: dict[str, LeverSpec],
    values: dict[str, Any],
    model_prices: dict[str, dict[str, float]] | None,
) -> str:
    """Describe each declared lever: current value, allowlist, and history.

    History is read from past ``eval/runs/round_*.json`` records (which carry
    the resolved ``levers`` each eval ran with), so the optimizer sees the
    measured median latency and aggregate per lever value — latency is NOT in
    any price list and must be measured. For the ``model`` lever, per-1M-token
    prices are shown when a model price list was supplied.
    """
    if not specs:
        return "(no levers declared in harness/config.yaml — only scaffold edits are available)"
    history = _lever_history(repo_root)
    lines = [
        "Change one with `set_lever` (name + a value from its allowed list). "
        "Values outside the list are rejected.",
        "",
    ]
    for name, spec in specs.items():
        current = values.get(name, spec.default)
        head = f"- `{name}` (current: {current!r})"
        if spec.description:
            head += f" — {spec.description}"
        lines.append(head)
        for option in spec.allowed:
            parts = [f"    - {option!r}"]
            if name == MODEL_LEVER and model_prices:
                price = model_prices.get(str(option))
                if price:
                    parts.append(
                        f"${price.get('input_per_million', float('nan')):g} in / "
                        f"${price.get('output_per_million', float('nan')):g} out per 1M tokens"
                    )
                else:
                    parts.append("price unknown")
            stats = history.get((name, str(option)))
            if stats:
                parts.append(stats)
            else:
                parts.append("not yet evaluated")
            lines.append(parts[0] + ": " + "; ".join(parts[1:]))
    return "\n".join(lines)


def _lever_history(repo_root: Path) -> dict[tuple[str, str], str]:
    """Summarize past rounds per (lever, value): n, median latency, best aggregate."""
    runs_dir = repo_root / "eval" / "runs"
    if not runs_dir.is_dir():
        return {}
    buckets: dict[tuple[str, str], dict[str, list]] = {}
    for path in sorted(runs_dir.glob("round_*.json")):
        try:
            rec = _read_json(path)
        except (OSError, ValueError):
            continue
        levers = rec.get("levers") or {}
        if not levers or rec.get("aggregate") is None:
            continue
        latency = (rec.get("cost_metrics") or {}).get("latency_ms_median")
        for name, value in levers.items():
            b = buckets.setdefault((name, str(value)), {"agg": [], "lat": [], "kept": []})
            b["agg"].append(float(rec["aggregate"]))
            if isinstance(latency, (int, float)):
                b["lat"].append(float(latency))
            b["kept"].append(rec.get("decision") == "keep")
    out: dict[tuple[str, str], str] = {}
    for key, b in buckets.items():
        text = (
            f"{len(b['agg'])} round(s), {sum(b['kept'])} kept, best aggregate {max(b['agg']):.4f}"
        )
        if b["lat"]:
            text += f", median latency {statistics.median(b['lat']):.0f}ms"
        out[key] = text
    return out


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
