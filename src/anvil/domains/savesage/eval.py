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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anvil.data import select_subset
from anvil.domains.savesage.extractor import SavesageIciciExtractor
from anvil.domains.savesage.scoring import aggregate_corpus, field_slug, score_extraction
from anvil.eval.runner import EvalReport
from anvil.runtime.composer import compose_prompt
from anvil.runtime.loader import load_harness

logger = logging.getLogger(__name__)

_REQUIRED_ROW_FIELDS = ("example_id", "query", "category", "pdf_path", "expected_parsed_json")


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
    agent_module: str, *, composed_prompt: str, repo_root: Path, luna_profile: str | None = None
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
    return candidates[0](composed_prompt, luna_profile=luna_profile)


def evaluate_savesage(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    golden_set_path: Path | str = "data/golden_set.jsonl",
    profile: str | None = None,  # noqa: ARG001 - Luna auth uses env/profile chain
    mode: str | None = None,
    cache_root: Path | str | None = None,
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
    examples = load_savesage_golden_set(golden_set_path)
    selected = _select(examples, rows=mode_cfg.rows, buckets=dict(mode_cfg.buckets))
    n_workers = max(1, cfg.n_workers)

    # Optimization mode selects the per-row predictor and whether latency is
    # captured. ``prompt`` composes the prompt and runs the (cache-aware) Luna
    # extractor. ``code`` loads the optimizer-mutable SavesageAgent and calls
    # predict() — capturing the real cold-call latency (cache disabled) that
    # the latency Pareto objective reads.
    optimization_mode = snapshot.config.mode
    if optimization_mode == "code":
        agent = _load_savesage_agent(
            snapshot.config.agent_module,
            composed_prompt=composed,
            repo_root=scaffold_path.parent,
        )

        def _predict(row: dict) -> tuple[dict, float | None]:
            extraction, meta = agent.predict(sid=row["example_id"], pdf_path=row["pdf_path"])
            latency = meta.get("latency_ms") if isinstance(meta, dict) else None
            actual = extraction if isinstance(extraction, dict) else {}
            return actual, (float(latency) if latency is not None else None)
    else:
        if cache_root is None:
            cache_root = scaffold_path.parent / "eval" / "luna_cache"
        extractor = SavesageIciciExtractor(composed, cache_root=cache_root)

        def _predict(row: dict) -> tuple[dict, float | None]:
            return extractor.extract(sid=row["example_id"], pdf_path=row["pdf_path"]), None

    # The accuracy FLOOR excludes the stale-GT fields ONLY on the code/latency
    # path (so "accuracy held" measures real quality); prompt mode keeps the
    # full 28-field aggregate its baselines were built on.
    exclude_fields = (
        frozenset(cfg.accuracy_exclude_fields) if optimization_mode == "code" else frozenset()
    )

    def _predict_and_score(row: dict) -> dict[str, Any]:
        try:
            actual, latency_ms = _predict(row)
        except Exception as exc:  # noqa: BLE001 - isolate per-row failures
            logger.warning("prediction failed for %s: %s", row.get("example_id"), exc)
            actual, latency_ms = {}, None  # empty extraction scores as all-DISAGREE
        scored = score_extraction(row["expected_parsed_json"], actual)
        scored["example_id"] = row["example_id"]
        scored["query"] = row["query"]
        scored["category"] = row["category"]
        scored["latency_ms"] = latency_ms
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

    return EvalReport(
        aggregate=corpus["aggregate"],
        per_judge=corpus["per_judge"],
        per_bucket=per_bucket,
        failures=failures,
        run_id="",
        experiment_id="",
        n_rows=len(scored_rows),
        mode=selected_mode,
        scorers=sorted(corpus["per_judge"].keys()),
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        trace_ids=[],
        cost_metrics=cost_metrics,
        scorer_fingerprint="",
    )
