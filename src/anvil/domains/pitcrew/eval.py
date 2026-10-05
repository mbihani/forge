"""Pitcrew eval engine — re-run the (mutated) summarizer, score with pitcrew's judges.

The drop-in the forge loop uses when ``eval.engine: pitcrew``:

    select PDFs (eval mode) → agent call per PDF (timed) → parse the JSON →
    score with pitcrew's OWN judges (``eval.agent_evals``) → macro-aggregate →
    EvalReport (+ latency / cost in ``cost_metrics``)

* **Agent.** In ``code`` mode the optimizer-mutable :class:`PitcrewAgent`
  subclass at ``agent_module`` is called per PDF; in ``prompt`` mode the
  composed scaffold prompt is sent in one production-shaped call.
* **Judges.** Every row is scored by the judges reviewed from pitcrew's
  production experiment, fed the shape production gave them: inputs
  ``{pdf_bytes, filename, prompt_version}`` (``pdf_bytes`` as its ``str``
  repr) and the parsed summary as ``[summary]``. ``aggregate`` is their
  equal-weight mean. There is no forge-side judge.
* **Latency / cost.** Only the agent call is timed (judges run after); a
  failed call still records its wall time, so every report carries latency.
  Calls are priced from the model catalog.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anvil.catalog import cost_usd, load_catalog, model_prices
from anvil.data import select_subset
from anvil.domains.pitcrew.summarizer import summarize_pdf
from anvil.eval.agent_evals import (
    fingerprint_judges,
    load_configured_agent_judges,
    score_with_agent_judges,
)
from anvil.eval.runner import EvalReport
from anvil.observability import eval_row_trace
from anvil.runtime.composer import compose_prompt
from anvil.runtime.loader import load_harness

logger = logging.getLogger(__name__)

# The genai engine's golden set is the stock demo's; this domain keeps its own
# so the demo data (and the tests coupled to it) stay intact.
GOLDEN_SET_REL = Path("data") / "pitcrew_golden_set.jsonl"
_GENERIC_GOLDEN_SET = "data/golden_set.jsonl"
_REQUIRED_ROW_FIELDS = ("example_id", "query", "category", "pdf_path")
_TRACE_TEXT_CAP = 20_000
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def load_pitcrew_golden_set(path: str | Path) -> list[dict]:
    """Load the golden set: one FINRA rule PDF per line.

    Each row carries ``example_id`` (``finra_<rule>``), ``query`` (the rule
    title), ``category`` (bucket) and ``pdf_path`` (repo-relative or
    absolute). No reference summaries — the agent's judges score directly.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"pitcrew golden set not found: {p}")
    rows: list[dict] = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        missing = [f for f in _REQUIRED_ROW_FIELDS if f not in row]
        if missing:
            raise ValueError(f"pitcrew golden set line {line_no}: missing fields {missing}")
        rows.append(row)
    return rows


def extract_json(output: str) -> dict | None:
    """Parse the model output as a JSON object, tolerating markdown fences."""
    text = _FENCE_RE.sub("", (output or "").strip()).strip()
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1] if "{" in text else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def judge_io(pdf_bytes: bytes, filename: str, parsed: dict | None, output: str) -> dict[str, Any]:
    """Judge inputs/outputs in the shape pitcrew's production traces used.

    Production traced ``summarize_document`` with inputs
    ``{pdf_bytes, filename, prompt_version}`` (``pdf_bytes`` as its ``str``
    repr — the full PDF) and returned ``(summary, run_metadata)``. The judges
    get the summary as ``[summary]`` (the run metadata is a tracing artifact,
    not agent output). An unparseable output is passed as raw text.
    """
    return {
        "inputs": {"pdf_bytes": str(pdf_bytes), "filename": filename, "prompt_version": None},
        "outputs": [parsed] if isinstance(parsed, dict) else output,
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))]


def _macro(rows_scores: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for s in rows_scores:
        for k in keys:
            if k in s:
                acc[k].append(s[k])
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in acc.items()}


def _mean(per_judge: dict[str, float]) -> float:
    return sum(per_judge.values()) / len(per_judge) if per_judge else 0.0


def _load_agent(
    agent_module: str,
    *,
    composed_prompt: str,
    repo_root: Path,
    model: str,
    allowed_models: list[str],
    client: Any | None,
):
    """Import ``agent_module`` and instantiate its one ``PitcrewAgent`` subclass."""
    from anvil.domains.pitcrew.agent_base import PitcrewAgent  # noqa: PLC0415
    from anvil.eval.runner import _import_agent_module  # noqa: PLC0415

    module_path = agent_module
    if module_path.endswith(".py"):
        p = Path(module_path)
        module_path = str(p if p.is_absolute() else repo_root / p)
    module = _import_agent_module(module_path)
    candidates = [
        obj
        for name in dir(module)
        if isinstance(obj := getattr(module, name), type)
        and issubclass(obj, PitcrewAgent)
        and obj is not PitcrewAgent
        and getattr(obj, "__module__", None) == module.__name__
        and not inspect.isabstract(obj)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one PitcrewAgent subclass in {agent_module!r}, "
            f"found {[c.__name__ for c in candidates]}"
        )
    return candidates[0](composed_prompt, model=model, allowed_models=allowed_models, client=client)


def _cost_metrics(
    latencies: list[float], call_costs: list[float | None], n: int
) -> dict[str, float]:
    """``latency_ms_*`` (always, when any row was timed) + ``cost_usd_*`` (all calls priced)."""
    metrics: dict[str, float] = {"n_rows": float(n)}
    if latencies:
        lat = sorted(latencies)
        metrics |= {
            "latency_ms_median": float(statistics.median(lat)),
            "latency_ms_mean": float(statistics.fmean(lat)),
            "latency_ms_p90": float(_percentile(lat, 0.9)),
            "n_latency": float(len(lat)),
        }
    if call_costs and all(c is not None for c in call_costs):
        total = float(sum(c for c in call_costs if c is not None))
        metrics |= {"cost_usd_total": total, "cost_usd_per_row": total / n if n else 0.0}
    return metrics


def evaluate_pitcrew(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    golden_set_path: Path | str = _GENERIC_GOLDEN_SET,
    profile: str | None = None,
    mode: str | None = None,
    trace_rows: bool = False,
    predict_fn: Any | None = None,
    runtime_client: Any | None = None,
    agent_judges: list[Any] | None = None,
    **_kwargs: Any,  # absorb genai-only kwargs per the engine contract
) -> EvalReport:
    """Evaluate the current agent over the selected FINRA PDFs."""
    # The engine-dispatch path does not bind --profile for the gateway client;
    # it reads DATABRICKS_CONFIG_PROFILE at token time. "DEFAULT" = ambient.
    if profile and profile != "DEFAULT":
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile

    scaffold_path = Path(scaffold_root)
    repo_root = scaffold_path.parent
    snapshot = load_harness(scaffold_path, runtime_config_path)
    cfg = snapshot.config
    selected_mode = mode or cfg.eval.default_mode
    if selected_mode not in cfg.eval.modes:
        raise ValueError(f"mode {selected_mode!r} not in eval.modes ({list(cfg.eval.modes)})")
    mode_cfg = cfg.eval.modes[selected_mode]

    gs = Path(golden_set_path)
    if str(golden_set_path) == _GENERIC_GOLDEN_SET:
        gs = repo_root / GOLDEN_SET_REL
    examples = load_pitcrew_golden_set(gs)
    selected = (
        select_subset(examples, buckets=dict(mode_cfg.buckets))
        if mode_cfg.buckets
        else examples[: mode_cfg.rows]
    )

    if agent_judges is None:
        agent_judges = load_configured_agent_judges(cfg.eval, repo_root)
    if not agent_judges:
        raise ValueError(
            "pitcrew scores with its own judges, but none are configured: set "
            "eval.agent_evals.experiment_id and run scripts/review_agent_evals.py"
        )
    judge_names = [j.name for j in agent_judges]

    model = cfg.effective_runtime_model
    spec = cfg.lever_specs.get("model")
    allowed = [str(m) for m in spec.allowed] if spec else [model]
    prices = model_prices(
        sorted({*allowed, model}),
        load_catalog(repo_root / cfg.model_catalog.path),
        tier=cfg.model_catalog.context_tier,
    )
    composed = compose_prompt(scaffold_path, audience="runtime").text

    if predict_fn is None:
        if cfg.mode == "code":
            agent = _load_agent(
                cfg.agent_module,
                composed_prompt=composed,
                repo_root=repo_root,
                model=model,
                allowed_models=allowed,
                client=runtime_client,
            )

            def predict_fn(pdf_bytes: bytes) -> tuple[str, dict[str, Any]]:  # noqa: F811
                return agent.predict(pdf_bytes=pdf_bytes)
        else:
            from anvil.runtime.client import build_gateway_client  # noqa: PLC0415

            runtime_client = runtime_client or build_gateway_client()

            def predict_fn(pdf_bytes: bytes) -> tuple[str, dict[str, Any]]:  # noqa: F811
                text, usage = summarize_pdf(runtime_client, model, composed, pdf_bytes)
                return text, {"calls": [{"model": model, **usage}]}

    def _row(row: dict) -> dict[str, Any]:
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
            pdf_path = Path(row["pdf_path"])
            if not pdf_path.is_absolute():
                pdf_path = repo_root / pdf_path
            output, calls, latency_ms, pdf_bytes = "", [], None, b""
            t0: float | None = None
            try:
                pdf_bytes = pdf_path.read_bytes()
                t0 = time.perf_counter()  # time the agent call only
                result = predict_fn(pdf_bytes)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                output, meta = result if isinstance(result, tuple) else (result, {})
                calls = list((meta or {}).get("calls", []))
            except Exception as exc:  # noqa: BLE001 - isolate per-row failures
                logger.warning("summarize failed for %s: %s", row["example_id"], exc)
                if t0 is not None:  # a failed call still costs its wall time
                    latency_ms = (time.perf_counter() - t0) * 1000.0
            parsed = extract_json(output)
            scores = score_with_agent_judges(
                agent_judges, **judge_io(pdf_bytes, pdf_path.name, parsed, output)
            )
            if span is not None:
                with suppress(Exception):
                    span.set_outputs(
                        {
                            "scores": scores,
                            "latency_ms": latency_ms,
                            "parsed_json": parsed is not None,
                            "summary": output[:_TRACE_TEXT_CAP],
                        }
                    )
        return {
            "example_id": row["example_id"],
            "query": row["query"],
            "category": row["category"],
            "scores": scores,
            "latency_ms": latency_ms,
            "call_costs": [
                cost_usd(
                    prices.get(str(c.get("model"))),
                    int(c.get("input_tokens", 0) or 0),
                    int(c.get("output_tokens", 0) or 0),
                )
                for c in calls
            ],
        }

    results: list[dict[str, Any] | None] = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=max(1, cfg.eval.n_workers)) as ex:
        futs = {ex.submit(_row, r): i for i, r in enumerate(selected)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    rows = [r for r in results if r is not None]

    per_judge = _macro([r["scores"] for r in rows], judge_names)
    per_bucket: dict[str, dict[str, float]] = {}
    for cat in sorted({r["category"] for r in rows}):
        pj = _macro([r["scores"] for r in rows if r["category"] == cat], judge_names)
        per_bucket[cat] = {**pj, "aggregate": _mean(pj)}
    failures = [
        {
            "example_id": r["example_id"],
            "query": r["query"],
            "category": r["category"],
            "judge_failures": sorted(k for k, v in r["scores"].items() if v < 1.0),
            "per_objective": {k: round(v, 4) for k, v in r["scores"].items() if v < 1.0},
        }
        for r in rows
        if any(v < 1.0 for v in r["scores"].values())
    ]

    experiment_id = ""
    if trace_rows:
        try:
            import mlflow  # noqa: PLC0415

            exp = mlflow.get_experiment_by_name(cfg.experiments.eval)
            experiment_id = exp.experiment_id if exp is not None else ""
        except Exception as exc:  # noqa: BLE001 - best-effort observability
            logger.warning("could not resolve eval experiment id: %s", exc)

    return EvalReport(
        aggregate=_mean(per_judge),
        per_judge=per_judge,
        per_bucket=per_bucket,
        failures=failures,
        run_id="",
        experiment_id=experiment_id,
        n_rows=len(rows),
        mode=selected_mode,
        scorers=sorted(judge_names),
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        trace_ids=[],
        cost_metrics=_cost_metrics(
            [r["latency_ms"] for r in rows if r["latency_ms"] is not None],
            [c for r in rows for c in r["call_costs"]],
            len(rows),
        ),
        scorer_fingerprint=f"agent_evals:{fingerprint_judges(agent_judges)}",
    )
