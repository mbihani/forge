"""Pitcrew eval engine — re-run the mutated summarizer prompt over FINRA PDFs.

The drop-in the FORGE loop uses when ``eval.engine: pitcrew``. Structurally
identical to ``evaluate_savesage``:

    compose prompt (from scaffold) → summarize each PDF through the gateway
    (direct_pdf document block) → score (json-valid + schema + PDF-grounded
    judge) → weighted macro-aggregate → EvalReport

The optimizer edits the scaffold skills (the summarization system prompt) in
``prompt`` mode; this engine re-runs that mutated prompt each round so the
score reflects the CURRENT prompt. No MLflow run is created
(``run_id``/``experiment_id`` empty); ``scorer_fingerprint`` is empty so the
round's baseline-compat check is a no-op (both sides empty).
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anvil.catalog import cost_usd, load_catalog, model_prices
from anvil.data import select_subset
from anvil.domains.pitcrew.scoring import (
    OBJECTIVE_WEIGHTS,
    build_summary_quality_judge,
    extract_json,
    score_json_valid,
    score_schema_adherence,
)
from anvil.domains.pitcrew.summarizer import (
    DEFAULT_INPUT_MODE,
    _pdf_document_block,
    summarize,
    text_mode_available,
)
from anvil.eval.runner import EvalReport
from anvil.observability import eval_row_trace
from anvil.runtime.composer import compose_prompt
from anvil.runtime.loader import load_harness

logger = logging.getLogger(__name__)

_REQUIRED_ROW_FIELDS = ("example_id", "query", "category", "pdf_path")


def load_pitcrew_golden_set(path: str | Path) -> list[dict]:
    """Load the pitcrew golden set (one FINRA rule PDF per line).

    Each row must carry ``example_id`` (rule key), ``query`` (the rule
    title, for reporting), ``category`` (a bucket), and ``pdf_path`` (abs
    path to the rule PDF). There is no reference summary — scoring is
    deterministic structure checks + a PDF-grounded judge, so no ground
    truth is needed.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"pitcrew golden set not found: {p}")
    rows: list[dict] = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        missing = [f for f in _REQUIRED_ROW_FIELDS if f not in row]
        if missing:
            raise ValueError(
                f"pitcrew golden_set line {line_no} "
                f"({row.get('example_id', '?')}): missing fields {missing}"
            )
        rows.append(row)
    return rows


def _select(examples: list[dict], *, rows: int, buckets: dict[str, int]) -> list[dict]:
    """Deterministic subset: bucketed when configured, else first ``rows``."""
    if buckets:
        return select_subset(examples, buckets=buckets)
    return examples[:rows] if rows else examples


def _weighted_aggregate(per_objective: dict[str, float]) -> float:
    """Weighted mean over the objective keys present in ``per_objective``."""
    keys = [k for k in OBJECTIVE_WEIGHTS if k in per_objective]
    total_w = sum(OBJECTIVE_WEIGHTS[k] for k in keys)
    if not total_w:
        return 0.0
    return sum(per_objective[k] * OBJECTIVE_WEIGHTS[k] for k in keys) / total_w


def _macro(rows_scores: list[dict[str, float]]) -> dict[str, float]:
    """Per-objective macro-average across rows."""
    acc: dict[str, list[float]] = {k: [] for k in OBJECTIVE_WEIGHTS}
    for s in rows_scores:
        for k in OBJECTIVE_WEIGHTS:
            if k in s:
                acc[k].append(s[k])
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in acc.items()}


def _median(sorted_vals: list[float]) -> float:
    n = len(sorted_vals)
    mid = n // 2
    return sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def evaluate_pitcrew(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    golden_set_path: Path | str = "data/golden_set.jsonl",
    profile: str | None = None,
    mode: str | None = None,
    trace_rows: bool = False,
    predict_fn: Any | None = None,
    judge_fn: Any | None = None,
    runtime_client: Any | None = None,
    judge_client: Any | None = None,
    **_kwargs: Any,  # absorb genai-only kwargs per the engine contract
) -> EvalReport:
    """Evaluate the current scaffold's summarization prompt over a PDF subset."""
    # The engine-dispatch path in evaluate_branch returns before it binds
    # --profile into the environment (that binding is on the genai path only),
    # so honor it here: the gateway client reads DATABRICKS_CONFIG_PROFILE at
    # token-refresh time. "DEFAULT" is run_round's sentinel for ambient auth.
    if profile and profile != "DEFAULT":
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile

    scaffold_path = Path(scaffold_root)
    snapshot = load_harness(scaffold_path, runtime_config_path)
    cfg = snapshot.config.eval
    selected_mode = mode or cfg.default_mode
    if selected_mode not in cfg.modes:
        raise ValueError(
            f"mode {selected_mode!r} not in harness/config.yaml > eval.modes ({list(cfg.modes)})"
        )
    mode_cfg = cfg.modes[selected_mode]

    composed = compose_prompt(scaffold_path, audience="runtime").text
    examples = load_pitcrew_golden_set(golden_set_path)
    selected = _select(examples, rows=mode_cfg.rows, buckets=dict(mode_cfg.buckets))
    n_workers = max(1, cfg.n_workers)
    model = snapshot.config.effective_runtime_model
    # Domain lever (harness/config.yaml > levers.input_mode): how the PDF
    # reaches the model. Forge core treats the name as opaque.
    input_mode = str(snapshot.config.levers.get("input_mode", DEFAULT_INPUT_MODE))
    if input_mode == "text" and predict_fn is None and not text_mode_available():
        # Fail the whole eval loudly (an infra failure) rather than letting
        # every row error into a 0.0 that reads as a quality regression.
        raise RuntimeError(
            "levers.input_mode=text needs pymupdf in this environment (`uv pip install pymupdf`)"
        )
    # Per-1M-token price of the model this round runs, for cost_usd. From the
    # synced price sheet, LiteLLM as fallback; None = unpriced (cost omitted).
    catalog_path = scaffold_path.parent / snapshot.config.model_catalog.path
    price = model_prices(
        [model], load_catalog(catalog_path), tier=snapshot.config.model_catalog.context_tier
    ).get(model)

    # Re-run predictor + judge — injectable for offline unit tests. A
    # predict_fn may return the summary text, or (text, usage) for costing.
    if predict_fn is None:
        from anvil.runtime.client import build_gateway_client  # noqa: PLC0415

        runtime_client = runtime_client or build_gateway_client()

        def predict_fn(pdf_bytes: bytes) -> tuple[str, dict[str, int]]:  # noqa: F811
            return summarize(runtime_client, model, composed, pdf_bytes, input_mode=input_mode)

    if judge_fn is None:
        from anvil.runtime.client import build_gateway_client  # noqa: PLC0415

        judge_client = judge_client or build_gateway_client()
        judge_fn = build_summary_quality_judge(judge_client, snapshot.config.judge_endpoint)

    def _predict_and_score(row: dict) -> dict[str, Any]:
        # Wrap each row in a per-row root span (when tracing is armed by
        # evaluate_branch) so the summarize + judge gateway calls autolog as
        # CHAT_MODEL sub-spans under one coherent, tagged per-row eval trace.
        # ``nullcontext`` keeps the row untraced when called outside an eval
        # (e.g. unit tests inject predict_fn and pass trace_rows=False).
        row_cm = (
            eval_row_trace(
                example_id=row["example_id"],
                query=row["query"],
                scaffold_root=scaffold_path,
                runtime_endpoint=model,
            )
            if trace_rows
            else nullcontext()
        )
        with row_cm as span:
            try:
                # Resolve a relative pdf_path against the repo root (scaffold's
                # parent) so the vendored corpus works regardless of CWD — the
                # golden set ships repo-relative paths for portability.
                pdf_path = Path(row["pdf_path"])
                if not pdf_path.is_absolute():
                    pdf_path = scaffold_path.parent / pdf_path
                pdf_bytes = pdf_path.read_bytes()
                # Wall-clock of the summarize call only (judge excluded) — the
                # latency Pareto objective reads the median of these.
                t0 = time.perf_counter()
                result = predict_fn(pdf_bytes)
                latency_ms: float | None = (time.perf_counter() - t0) * 1000.0
                output, usage = result if isinstance(result, tuple) else (result, None)
            except Exception as exc:  # noqa: BLE001 - isolate per-row failures
                logger.warning("summarize failed for %s: %s", row.get("example_id"), exc)
                output, pdf_bytes, latency_ms, usage = "", b"", None, None
            row_cost = (
                cost_usd(price, usage["input_tokens"], usage["output_tokens"]) if usage else None
            )
            parsed = extract_json(output)
            json_valid = score_json_valid(parsed)
            scores: dict[str, float] = {
                "output_json_valid": json_valid,
                "schema_adherence": score_schema_adherence(parsed),
            }
            # Gate the quality judge behind parseability: an unparseable
            # structured output is a failure (and the judge scores prose, not
            # JSON validity), so a truncated/invalid summary must not earn
            # quality credit. Also skips the expensive PDF-grounded judge call
            # when it can't count.
            if pdf_bytes and json_valid >= 1.0:
                scores["summary_quality"] = judge_fn(_pdf_document_block(pdf_bytes), output)
            else:
                scores["summary_quality"] = 0.0
            if span is not None:
                # Span output is best-effort observability.
                with suppress(Exception):
                    span.set_outputs({"scores": scores, "n_chars": len(output)})
        return {
            "example_id": row["example_id"],
            "query": row["query"],
            "category": row["category"],
            "scores": scores,
            "latency_ms": latency_ms,
            "n_chars": len(output),
            "usage": usage,
            "cost_usd": row_cost,
        }

    results: list[dict[str, Any] | None] = [None] * len(selected)
    if n_workers <= 1:
        for i, row in enumerate(selected):
            results[i] = _predict_and_score(row)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            fut_to_idx = {ex.submit(_predict_and_score, r): i for i, r in enumerate(selected)}
            for fut in as_completed(fut_to_idx):
                results[fut_to_idx[fut]] = fut.result()
    scored_rows = [r for r in results if r is not None]

    row_scores = [r["scores"] for r in scored_rows]
    per_judge = _macro(row_scores)
    aggregate = _weighted_aggregate(per_judge)

    # Per-bucket rollup (same macro + weighting, restricted to each category).
    per_bucket: dict[str, dict[str, float]] = {}
    by_cat: dict[str, list[dict]] = {}
    for r in scored_rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rows_in in by_cat.items():
        pj = _macro([r["scores"] for r in rows_in])
        per_bucket[cat] = {**pj, "aggregate": _weighted_aggregate(pj)}

    # Failures: any row with a sub-1.0 objective, for optimizer localization.
    failures: list[dict[str, Any]] = []
    for r in scored_rows:
        misses = {k: round(v, 4) for k, v in r["scores"].items() if v < 1.0}
        if misses:
            failures.append(
                {
                    "example_id": r["example_id"],
                    "query": r["query"],
                    "category": r["category"],
                    "judge_failures": sorted(misses),
                    "per_objective": misses,
                }
            )

    # When tracing is armed, record the eval experiment id so the report
    # points at where this round's per-row traces landed. Best-effort.
    experiment_id = ""
    if trace_rows:
        try:
            import mlflow  # noqa: PLC0415

            exp = mlflow.get_experiment_by_name(snapshot.config.experiments.eval)
            if exp is not None:
                experiment_id = exp.experiment_id
        except Exception as exc:  # noqa: BLE001 - best-effort observability
            logger.warning("could not resolve eval experiment id: %s", exc)

    # Cost metrics: row count always; latency stats over rows whose summarize
    # call succeeded. The latency Pareto objective reads latency_ms_median
    # (robust to the 10-page rule's slow tail); output_chars_mean is the
    # low-noise proxy for decode time the optimizer can actually move.
    cost_metrics: dict[str, float] = {"n_rows": float(len(scored_rows))}
    latencies = sorted(r["latency_ms"] for r in scored_rows if r.get("latency_ms") is not None)
    if latencies:
        cost_metrics["latency_ms_median"] = _median(latencies)
        cost_metrics["latency_ms_mean"] = sum(latencies) / len(latencies)
        cost_metrics["latency_ms_p90"] = _percentile(latencies, 0.90)
    if scored_rows:
        cost_metrics["output_chars_mean"] = sum(r["n_chars"] for r in scored_rows) / len(
            scored_rows
        )
    # Token usage + dollar cost of the summarize calls (judge excluded, like
    # latency). Informational — shown to the optimizer, not a gate input.
    usages = [r["usage"] for r in scored_rows if r.get("usage")]
    if usages:
        cost_metrics["input_tokens_mean"] = sum(u["input_tokens"] for u in usages) / len(usages)
        cost_metrics["output_tokens_mean"] = sum(u["output_tokens"] for u in usages) / len(usages)
    costs = [r["cost_usd"] for r in scored_rows if r.get("cost_usd") is not None]
    if costs:
        cost_metrics["cost_usd_total"] = sum(costs)
        cost_metrics["cost_usd_per_row_mean"] = sum(costs) / len(costs)

    return EvalReport(
        aggregate=aggregate,
        per_judge=per_judge,
        per_bucket=per_bucket,
        failures=failures,
        run_id="",
        experiment_id=experiment_id,
        n_rows=len(scored_rows),
        mode=selected_mode,
        scorers=sorted(per_judge.keys()),
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        trace_ids=[],
        cost_metrics=cost_metrics,
        scorer_fingerprint="",
    )
