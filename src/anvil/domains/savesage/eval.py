"""Savesage ICICI eval engine — a trace-free parallel map that emits ``EvalReport``.

This is the drop-in the ANVIL loop uses instead of ``mlflow.genai.evaluate``
when ``eval.engine: savesage``. The genai path exists only to build per-row
RETRIEVER traces for ``RetrievalGroundedness``; Savesage scores deterministically
against cached Opus GT, needs no traces, and so runs as a plain parallel map:

    compose prompt (from scaffold) → Luna extract per row → score vs cached GT
    → macro-aggregate → EvalReport

The report shape (aggregate / per_judge / per_bucket / failures /
scorer_fingerprint) is exactly what ``loop/round.py`` and ``loop/frontier.py``
already consume, so the round/gate/git machinery is untouched. ``aggregate`` is
the corpus strict accuracy (reproduces ``judge.accuracy``); ``per_judge`` also
carries narration-forgiven and the seven per-field accuracies for optimizer
diagnostics. No MLflow run is created (``run_id``/``experiment_id`` empty), and
``scorer_fingerprint`` is empty so the round's baseline-compatibility check is a
no-op (both sides empty) — the reproduce-the-baseline gate validates fidelity
instead.
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

from anvil.catalog import load_catalog, model_prices
from anvil.data import select_subset
from anvil.domains.savesage.extractor import SavesageIciciExtractor
from anvil.domains.savesage.scoring import aggregate_corpus, field_slug, score_extraction
from anvil.eval.lever_probe import load_lever_probe
from anvil.eval.runner import EvalReport
from anvil.observability import eval_row_trace
from anvil.runtime.call_ledger import LEDGER, describe_error, is_infra_error, price_calls
from anvil.runtime.composer import compose_prompt
from anvil.runtime.loader import load_harness

logger = logging.getLogger(__name__)

_REQUIRED_ROW_FIELDS = ("example_id", "query", "category", "pdf_path", "expected_parsed_json")

# The genai engine's golden set is the stock demo's; this domain keeps its own
# (gitignored: cardholder PII). The loop passes the generic path (relative from
# make_baseline, absolute from run_round); either way it means this domain's.
GOLDEN_SET_REL = Path("data") / "savesage_golden_set.jsonl"
_GENERIC_GOLDEN_SET = "data/golden_set.jsonl"
INPUT_MODE_LEVER = "input_mode"


def load_savesage_golden_set(path: str | Path) -> list[dict]:
    """Load the Savesage golden set (one ICICI statement per line).

    Each row must carry ``example_id`` (the sid), ``query`` (the sid too —
    the extractor keys on ``pdf_path``, not the query text), ``category``
    (a bucket for per-bucket reporting), ``pdf_path`` (abs path to the
    statement PDF), and ``expected_parsed_json`` (the cached Opus GT
    extraction dict). Rows are returned in file order (deterministic).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"savesage golden set not found: {p}")
    rows: list[dict] = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        missing = [f for f in _REQUIRED_ROW_FIELDS if f not in row]
        if missing:
            raise ValueError(
                f"savesage golden_set line {line_no} "
                f"({row.get('example_id', '?')}): missing fields {missing}"
            )
        rows.append(row)
    return rows


def _select(examples: list[dict], *, rows: int, buckets: dict[str, int]) -> list[dict]:
    """Deterministic subset: bucketed when configured, else first ``rows``."""
    if buckets:
        return select_subset(examples, buckets=buckets)
    return examples[:rows] if rows else examples


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


def _load_savesage_agent(
    agent_module: str,
    *,
    composed_prompt: str,
    repo_root: Path,
    luna_profile: str | None = None,
    **agent_kwargs: Any,
):
    """Import the code-mode ``SavesageAgent`` module and instantiate its subclass.

    ``agent_module`` is a ``.py`` path (resolved against ``repo_root`` when
    relative) or a dotted module path. The module must define exactly one
    concrete :class:`SavesageAgent` subclass — the optimizer-mutable agent.
    Reuses the runner's ``_import_agent_module`` so code-mode agents load the
    same way the applier wrote them.
    """
    import inspect  # noqa: PLC0415

    from anvil.domains.savesage.agent_base import SavesageAgent  # noqa: PLC0415
    from anvil.eval.runner import _import_agent_module  # noqa: PLC0415

    module_path = agent_module
    if module_path.endswith(".py"):
        p = Path(module_path)
        module_path = str(p if p.is_absolute() else repo_root / p)
    module = _import_agent_module(module_path)

    candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, SavesageAgent)
            and obj is not SavesageAgent
            and getattr(obj, "__module__", None) == module.__name__
            and not inspect.isabstract(obj)
        ):
            candidates.append(obj)
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one SavesageAgent subclass in {agent_module!r}, "
            f"found {[c.__name__ for c in candidates]}"
        )
    return candidates[0](composed_prompt, luna_profile=luna_profile, **agent_kwargs)


def evaluate_savesage(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    golden_set_path: Path | str = _GENERIC_GOLDEN_SET,
    profile: str | None = None,  # noqa: ARG001 - Luna auth uses env/profile chain
    mode: str | None = None,
    cache_root: Path | str | None = None,
    trace_rows: bool = False,
    max_rows: int | None = None,
    **_kwargs: Any,  # absorb genai-only kwargs per the engine contract
) -> EvalReport:
    """Evaluate the current scaffold's ICICI prompt over a subset of the corpus.

    Composes the prompt from ``scaffold_root``, extracts each selected
    statement through Luna (prompt-addressed cache), scores against the
    cached Opus GT, and returns an ``EvalReport``.

    When ``SAVESAGE_STATEMENT_AGENT_PATH`` is not already set, the
    ``statement-agent`` subdirectory under ``scaffold_root`` is used
    if it exists — this lets the deployed Databricks App find the
    statement-agent tree when it is included in the cloned repo.
    """
    # Auto-resolve SAVESAGE_STATEMENT_AGENT_PATH from scaffold_root so the
    # deployed app can locate the statement-agent tree without an env var.
    if not os.environ.get("SAVESAGE_STATEMENT_AGENT_PATH"):
        _sa_path = Path(scaffold_root) / "statement-agent"
        if _sa_path.is_dir():
            os.environ["SAVESAGE_STATEMENT_AGENT_PATH"] = str(_sa_path)

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
    repo_root = scaffold_path.parent
    gs = Path(golden_set_path)
    if gs.name == Path(_GENERIC_GOLDEN_SET).name:
        gs = repo_root / GOLDEN_SET_REL
    examples = load_savesage_golden_set(gs)
    selected = _select(examples, rows=mode_cfg.rows, buckets=dict(mode_cfg.buckets))
    if max_rows:
        # A canary: only the first rows of the mode's subset.
        selected = selected[:max_rows]
    n_workers = max(1, cfg.n_workers)

    # Optimization mode selects the per-row predictor and whether latency is
    # captured. ``prompt`` composes the prompt and runs the (cache-aware) Luna
    # extractor. ``code`` loads the optimizer-mutable SavesageAgent and calls
    # predict() — capturing the real cold-call latency (cache disabled) that
    # the latency Pareto objective reads.
    optimization_mode = snapshot.config.mode
    model = snapshot.config.effective_runtime_model
    spec = snapshot.config.lever_specs.get("model")
    allowed = [str(m) for m in spec.allowed] if spec else [model]
    prices = model_prices(
        sorted({*allowed, model}),
        load_catalog(repo_root / snapshot.config.model_catalog.path),
        tier=snapshot.config.model_catalog.context_tier,
    )
    if optimization_mode == "code":
        agent = _load_savesage_agent(
            snapshot.config.agent_module,
            composed_prompt=composed,
            repo_root=repo_root,
            model=model,
            input_mode=str(snapshot.config.levers.get(INPUT_MODE_LEVER, "pdf")),
            allowed_models=allowed,
            lever_probe=load_lever_probe(repo_root),
        )

        max_sends = cfg.call_policy.max_sends_per_document

        def _attempt(row: dict) -> dict[str, Any]:
            # Time the whole predict (every call + any local step it takes),
            # not the agent's self-reported latency. The ledger enforces the
            # call policy and records every call made, so cost counts all of
            # them — including any still running when predict returns (end()
            # waits for those, outside the timing).
            sid = row["example_id"]
            LEDGER.begin(sid, max_sends_per_document=max_sends)
            extraction: Any = {}
            error: str | None = None
            t0 = time.perf_counter()
            try:
                extraction, _meta = agent.predict(sid=sid, pdf_path=row["pdf_path"])
            except Exception as exc:  # noqa: BLE001 - a failed row still costs its calls
                logger.warning("prediction failed for %s: %s", sid, exc)
                error = describe_error(exc)
            latency = (time.perf_counter() - t0) * 1000.0
            calls = LEDGER.end(sid)
            actual = extraction if isinstance(extraction, dict) else {}
            if error is None and not actual and calls and calls[-1].get("error"):
                # The agent swallowed the failure; surface the call's error.
                error = calls[-1]["error"]
            return {"actual": actual, "latency_ms": latency, "calls": calls, "error": error}

        def _infra_failed(attempt: dict[str, Any]) -> bool:
            # A transport failure the agent did not cause: the row raised one,
            # or produced nothing while every call it made failed on transport.
            calls = attempt["calls"]
            if attempt["actual"] and attempt["error"] is None:
                return False
            if is_infra_error(attempt["error"]):
                return True
            return bool(calls) and all(is_infra_error(c.get("error")) for c in calls)

        def _predict(row: dict) -> dict[str, Any]:
            # Re-run a row that failed for transport reasons only, up to
            # eval.infra_retries times. Every attempt's calls are priced; the
            # latency is the last attempt's. A row still failing stays failed.
            calls: list[dict] = []
            for n in range(cfg.infra_retries + 1):
                attempt = _attempt(row)
                calls.extend(attempt["calls"])
                infra = _infra_failed(attempt)
                if not infra or n == cfg.infra_retries:
                    break
                logger.warning(
                    "infra failure for %s (%s); re-running the row",
                    row["example_id"],
                    attempt["error"],
                )
            return {**attempt, "calls": calls, "infra_retries": n, "infra_failed": infra}
    else:
        if cache_root is None:
            cache_root = repo_root / "eval" / "luna_cache"
        extractor = SavesageIciciExtractor(composed, cache_root=cache_root)

        def _predict(row: dict) -> dict[str, Any]:
            extraction = extractor.extract(sid=row["example_id"], pdf_path=row["pdf_path"])
            return {"actual": extraction, "latency_ms": None, "calls": [], "error": None}

    # The accuracy FLOOR excludes the stale-GT fields ONLY on the code/latency
    # path (so "accuracy held" measures real quality); prompt mode keeps the
    # full 28-field aggregate its baselines were built on.
    exclude_fields = (
        frozenset(cfg.accuracy_exclude_fields) if optimization_mode == "code" else frozenset()
    )

    def _predict_and_score(row: dict) -> dict[str, Any]:
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
            scored = _score_row(row)
            if span is not None:
                # Scores and timings only — the extraction itself is cardholder PII.
                with suppress(Exception):
                    span.set_outputs(
                        {
                            "strict_accuracy": scored.get("accuracy"),
                            "latency_ms": scored["latency_ms"],
                            "models": scored["models"],
                            "call_costs": scored["call_costs"],
                        }
                    )
        return scored

    def _score_row(row: dict) -> dict[str, Any]:
        try:
            pred = _predict(row)
        except Exception as exc:  # noqa: BLE001 - isolate per-row failures
            logger.warning("prediction failed for %s: %s", row.get("example_id"), exc)
            # An empty extraction scores as all-DISAGREE.
            pred = {"actual": {}, "latency_ms": None, "calls": [], "error": describe_error(exc)}
        scored = score_extraction(row["expected_parsed_json"], pred["actual"])
        scored["error"] = pred["error"]
        scored["infra_retries"] = int(pred.get("infra_retries", 0))
        scored["infra_failed"] = bool(pred.get("infra_failed", False))
        scored["example_id"] = row["example_id"]
        scored["query"] = row["query"]
        scored["category"] = row["category"]
        scored["latency_ms"] = pred["latency_ms"]
        scored["call_costs"] = price_calls(pred["calls"], prices)
        scored["models"] = sorted({str(c.get("model")) for c in pred["calls"]})
        return scored

    # Preserve input order so per_bucket / failures line up with ``selected``.
    results: list[dict[str, Any] | None] = [None] * len(selected)
    if n_workers <= 1:
        for i, row in enumerate(selected):
            results[i] = _predict_and_score(row)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            fut_to_idx = {ex.submit(_predict_and_score, r): i for i, r in enumerate(selected)}
            for fut in as_completed(fut_to_idx):
                results[fut_to_idx[fut]] = fut.result()
    scored_rows: list[dict[str, Any]] = [r for r in results if r is not None]

    corpus = aggregate_corpus(scored_rows, exclude_fields=exclude_fields)

    # Per-bucket rollup (same macro-average + gating, restricted to each category).
    per_bucket: dict[str, dict[str, float]] = {}
    by_cat: dict[str, list[dict]] = {}
    for r in scored_rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rows_in in by_cat.items():
        c = aggregate_corpus(rows_in, exclude_fields=exclude_fields)
        per_bucket[cat] = {**c["per_judge"], "aggregate": c["aggregate"]}

    # Failures: any statement whose strict accuracy is below 1.0, with the
    # per-field snapshot the optimizer reads to localize the regression.
    failures: list[dict[str, Any]] = []
    for r in scored_rows:
        strict = r.get("accuracy")
        if strict is None or strict >= 1.0:
            continue
        # Surface EVERY judged field that missed (all 28, from raw_per_field),
        # not only the 7 headline ones — so the optimizer can localize the
        # laggards (network, productFamily, txnType, direction, …) that pull
        # the strict aggregate down. slugified to match production judge names.
        field_misses = {
            field_slug(path): round(stats["accuracy"], 4)
            for path, stats in (r.get("raw_per_field") or {}).items()
            if (stats or {}).get("accuracy") is not None and stats["accuracy"] < 1.0
        }
        failures.append(
            {
                "example_id": r["example_id"],
                "query": r["query"],
                "category": r["category"],
                "strict_accuracy": strict,
                "judge_failures": sorted(field_misses),
                "per_field": field_misses,
                # Why the row produced nothing (API rejection, duplicate send,
                # crash) — so the optimizer can tell a failed call from a bad
                # extraction. Capped: provider errors can be long.
                **({"error": r["error"][:500]} if r.get("error") else {}),
                **({"infra": True} if r.get("infra_failed") else {}),
            }
        )

    # Cost metrics: row count always; latency stats when the code path captured
    # per-row wall-clock. The latency Pareto objective reads latency_ms_median
    # (robust to the occasional slow tail).
    cost_metrics: dict[str, float] = {"n_rows": float(len(scored_rows))}
    latencies = sorted(r["latency_ms"] for r in scored_rows if r.get("latency_ms") is not None)
    if latencies:
        cost_metrics["latency_ms_median"] = _median(latencies)
        cost_metrics["latency_ms_mean"] = sum(latencies) / len(latencies)
        cost_metrics["latency_ms_p90"] = _percentile(latencies, 0.90)
    # Row-level failure counts: the canary and the optimizer read these.
    cost_metrics["n_row_errors"] = float(sum(1 for r in scored_rows if r.get("error")))
    cost_metrics["n_infra_retries"] = float(sum(r.get("infra_retries", 0) for r in scored_rows))
    cost_metrics["n_infra_failed"] = float(sum(1 for r in scored_rows if r.get("infra_failed")))
    # Cost only when every call was priced, so a gap never understates it.
    call_costs = [c for r in scored_rows for c in r.get("call_costs", [])]
    if call_costs and all(c is not None for c in call_costs):
        total = float(sum(call_costs))
        cost_metrics["cost_usd_total"] = total
        cost_metrics["cost_usd_per_row"] = total / len(scored_rows) if scored_rows else 0.0

    experiment_id = ""
    if trace_rows:
        try:
            import mlflow  # noqa: PLC0415

            exp = mlflow.get_experiment_by_name(snapshot.config.experiments.eval)
            experiment_id = exp.experiment_id if exp is not None else ""
        except Exception as exc:  # noqa: BLE001 - best-effort observability
            logger.warning("could not resolve eval experiment id: %s", exc)

    return EvalReport(
        aggregate=corpus["aggregate"],
        per_judge=corpus["per_judge"],
        per_bucket=per_bucket,
        failures=failures,
        run_id="",
        experiment_id=experiment_id,
        n_rows=len(scored_rows),
        mode=selected_mode,
        scorers=sorted(corpus["per_judge"].keys()),
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        trace_ids=[],
        cost_metrics=cost_metrics,
        scorer_fingerprint="",
    )
