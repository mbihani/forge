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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anvil.data import select_subset
from anvil.domains.pitcrew.scoring import (
    OBJECTIVE_WEIGHTS,
    build_summary_quality_judge,
    extract_json,
    score_json_valid,
    score_schema_adherence,
)
from anvil.domains.pitcrew.summarizer import _pdf_document_block, summarize_pdf
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
    model = snapshot.config.runtime_endpoint

    # Re-run predictor + judge — injectable for offline unit tests.
    if predict_fn is None:
        from anvil.runtime.client import build_gateway_client  # noqa: PLC0415

        runtime_client = runtime_client or build_gateway_client()

        def predict_fn(pdf_bytes: bytes) -> str:  # noqa: F811
            return summarize_pdf(runtime_client, model, composed, pdf_bytes)

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
                output = predict_fn(pdf_bytes)
            except Exception as exc:  # noqa: BLE001 - isolate per-row failures
                logger.warning("summarize failed for %s: %s", row.get("example_id"), exc)
                output, pdf_bytes = "", b""
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
        cost_metrics={"n_rows": float(len(scored_rows))},
        scorer_fingerprint="",
    )
