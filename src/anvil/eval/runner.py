"""End-to-end evaluation runner — driver of ``mlflow.genai.evaluate``.

Wraps :func:`mlflow.genai.evaluate` with the active scorers (3 by
default; Safety opt-in), the golden set sub-set per mode
(``quick``/``standard``/``full``), and an :class:`AnvilAgent`
constructed with ``source=SOURCE_EVAL``.

Parallel predict execution: ``mlflow.genai.evaluate`` already runs
``predict_fn`` per row in a ``ThreadPoolExecutor`` sized by the
``MLFLOW_GENAI_EVAL_MAX_WORKERS`` env var (default 10). The harness
wires ``eval.n_workers`` from ``harness/config.yaml`` into that env
var so the configured value actually controls concurrency — and keeps
passing ``predict_fn`` (not pre-computed ``outputs``) so mlflow builds
a per-row trace carrying the ``RETRIEVER`` span that
``RetrievalGroundedness`` scores against. :func:`_run_predictions_parallel`
is anvil's own tested thread-pool primitive for direct/pre-compute
paths that do not need traces.

Public surface:

* :func:`evaluate_branch` — driver function callable from
  ``scripts/evaluate.py`` or another module.
* :class:`EvalReport` — aggregate / per-judge / per-bucket / failures.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import warnings
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.types.responses import ResponsesAgentRequest
from openai import OpenAI

try:
    from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING

    _MLFLOW_ASYNC_TRACE_LOGGING_ENV = MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.name
except ImportError:
    # Compatibility with MLflow versions that predate the env-var constant.
    _MLFLOW_ASYNC_TRACE_LOGGING_ENV = "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"

from anvil.agents.memory_system import MemorySystem
from anvil.data import load_golden_set, select_subset
from anvil.eval.cache import compute_scorer_fingerprint
from anvil.eval.engines import GENAI_ENGINE, load_engine
from anvil.eval.scorers import build_scorers
from anvil.observability import SOURCE_EVAL, enable_runtime_tracing
from anvil.runtime.agent import AnvilAgent
from anvil.runtime.client import build_gateway_client
from anvil.runtime.loader import default_runtime_config_path, load_harness
from anvil.runtime.models import EvalConfig, ScorerConfig, SplitConfig
from anvil.tools.search_knowledge_base import make_kb_executor

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Summary of one ``mlflow.genai.evaluate`` run."""

    aggregate: float
    per_judge: dict[str, float]
    per_bucket: dict[str, dict[str, float]]
    failures: list[dict[str, Any]]
    run_id: str
    experiment_id: str
    n_rows: int
    mode: str
    scorers: list[str]
    evaluated_at: str
    trace_ids: list[str] = field(default_factory=list)
    # Always-available eval cost proxies. Token usage may be added when
    # supplied by MLflow traces; context characters and row count do not
    # require another service call.
    cost_metrics: dict[str, float] = field(default_factory=dict)
    # JSON fingerprint of the aggregate scorer configs (name, type,
    # weight, check_function) that produced this report's aggregate.
    # Carried into ``CachedBaseline`` so the frontier gate can detect a
    # weight/check_function change that invalidates a cross-run
    # comparison even when scorer names are unchanged. Empty when the
    # report is built by code that predates this field.
    scorer_fingerprint: str = ""


def partition_dataset(
    examples: list[dict],
    split: SplitConfig,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition examples into (train, dev, test) by hash of example_id.

    Uses a deterministic hash plus seed so membership is stable across runs
    and independent of input ordering.
    """
    train: list[dict] = []
    dev: list[dict] = []
    test: list[dict] = []
    train_cutoff = split.train_ratio
    dev_cutoff = train_cutoff + split.dev_ratio

    for example in examples:
        digest = hashlib.md5(  # noqa: S324 - deterministic partitioning, not security
            f"{split.seed}:{example['example_id']}".encode(), usedforsecurity=False
        ).hexdigest()
        fraction = int(digest, 16) / (2**128)
        if fraction < train_cutoff:
            train.append(example)
        elif fraction < dev_cutoff:
            dev.append(example)
        else:
            test.append(example)

    return train, dev, test


def _verify_no_overlap(train: list[dict], dev: list[dict], test: list[dict]) -> None:
    """Assert no example_id appears in multiple partitions."""
    train_ids = {example["example_id"] for example in train}
    dev_ids = {example["example_id"] for example in dev}
    test_ids = {example["example_id"] for example in test}
    overlap = (train_ids & dev_ids) | (train_ids & test_ids) | (dev_ids & test_ids)
    if overlap:
        raise RuntimeError(f"partition overlap detected: {overlap}")


def _select_mode_examples(
    examples: list[dict], *, cfg: EvalConfig, selected_mode: str
) -> list[dict]:
    """Select a mode's rows while enforcing configured partition boundaries."""
    mode_config = cfg.modes[selected_mode]
    if not cfg.split.enabled:
        return select_subset(examples, buckets=mode_config.buckets)

    train, dev, test = partition_dataset(examples, cfg.split)
    _verify_no_overlap(train, dev, test)
    if selected_mode == "test":
        return test[: mode_config.rows]

    scaled_buckets = {
        bucket: max(1, round(count * cfg.split.dev_ratio))
        for bucket, count in mode_config.buckets.items()
    }
    if scaled_buckets != mode_config.buckets:
        warnings.warn(
            f"scaled {selected_mode!r} bucket counts for dev_ratio="
            f"{cfg.split.dev_ratio}: {mode_config.buckets} -> {scaled_buckets}",
            UserWarning,
            stacklevel=2,
        )
    return select_subset(dev, buckets=scaled_buckets)


def _extract_final_text(response: Any) -> str:
    """Walk ``response.output`` for the last ``message`` and concat its content."""
    output = getattr(response, "output", None) or []
    last_message: dict[str, Any] | None = None
    for item in output:
        data = item.model_dump() if hasattr(item, "model_dump") else item
        if isinstance(data, dict) and data.get("type") == "message":
            last_message = data
    if last_message is None:
        return ""
    parts = last_message.get("content", [])
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
        elif isinstance(part, str):
            chunks.append(part)
    return "".join(chunks)


def _build_dataset(examples: list[dict]) -> list[dict]:
    """Project golden-set rows into mlflow's inputs/expectations/tags shape."""
    # Correctness rejects rows that pass BOTH expected_response and
    # expected_facts. We use must_include as expected_facts; the
    # reference_answer stays in the row for human debugging via
    # mlflow.search_traces.
    #
    # ``must_include`` is ALSO projected under its golden-set name so
    # programmatic check functions (data/evaluator.py) can read the
    # familiar key directly from the expectations dict they receive as
    # ``ground_truth``. This is additive — Correctness still reads
    # ``expected_facts`` and ignores the alias.
    rows: list[dict] = []
    for ex in examples:
        expectations: dict[str, Any] = {
            "expected_facts": ex["must_include"],
            "must_include": ex["must_include"],
            "should_refuse": ex["should_refuse"],
            "expected_doc_ids": ex["expected_doc_ids"],
            "expected_citations": ex["expected_citations"],
            "must_not_include": ex["must_not_include"],
            "notes_for_judge": ex["notes_for_judge"],
            "reference_answer": ex["reference_answer"],
        }
        # Pass through json_schema, expected_fields, and any other
        # extension fields prefixed with ``json_`` or ``expected_`` so
        # programmatic check functions (json_schema_validity,
        # field_exact_match) receive their documented primary inputs
        # through the real runner. This is additive — existing scorers
        # ignore unknown keys in the expectations dict.
        for key, val in ex.items():
            if key not in expectations and (key.startswith("json_") or key.startswith("expected_")):
                expectations[key] = val
        rows.append(
            {
                "inputs": {
                    "query": ex["query"],
                    "category": ex["category"],
                },
                "expectations": expectations,
                "tags": {"example_id": ex["example_id"]},
            }
        )
    return rows


def _coerce_score(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("yes", "pass", "true", "ok"):
            return 1.0
        if s in ("no", "fail", "false"):
            return 0.0
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _row_score(row: Any, scorer_name: str) -> float | None:
    if hasattr(row, "get"):
        flat = row.get(f"{scorer_name}/value")
        coerced = _coerce_score(flat)
        if coerced is not None:
            return coerced
    assessments = row.get("assessments") if hasattr(row, "get") else None
    if not isinstance(assessments, list):
        return None
    for a in assessments:
        if not isinstance(a, dict):
            continue
        if a.get("assessment_name") != scorer_name:
            continue
        feedback = a.get("feedback")
        if isinstance(feedback, dict):
            coerced = _coerce_score(feedback.get("value"))
            if coerced is not None:
                return coerced
    return None


def _category_for_row(row: Any, examples: list[dict], idx: int) -> str:
    if hasattr(row, "get"):
        request = row.get("request")
        if isinstance(request, dict):
            cat = request.get("category")
            if isinstance(cat, str):
                return cat
    if idx < len(examples):
        cat = examples[idx].get("category")
        if isinstance(cat, str):
            return cat
    return ""


def _aggregate_report(
    *,
    result_df,
    metrics: dict[str, float],
    scorer_names: list[str],
    aggregate_scorer_names: list[str],
    weights: dict[str, float],
    examples: list[dict],
    run_id: str,
    experiment_id: str,
    mode: str,
    scorer_fingerprint: str = "",
) -> EvalReport:
    n_rows = len(result_df)

    per_judge_rows: dict[str, list[float | None]] = {
        name: [_row_score(result_df.iloc[i], name) for i in range(n_rows)] for name in scorer_names
    }

    def _mean(values: list[float | None]) -> float:
        nums = [v for v in values if v is not None]
        return sum(nums) / len(nums) if nums else 0.0

    per_judge: dict[str, float] = {}
    for name in scorer_names:
        metric_key = f"{name}/mean"
        if metric_key in metrics:
            per_judge[name] = float(metrics[metric_key])
        else:
            per_judge[name] = _mean(per_judge_rows[name])

    # Weighted average across the configured scorers. ``weights`` maps a
    # scorer name to its config weight (defaulting to 1.0); with uniform
    # weights this collapses to the legacy unweighted mean, so a shipped
    # scaffold that lists scorers as bare strings scores identically.
    total_weight = sum(weights.get(name, 1.0) for name in aggregate_scorer_names)
    if aggregate_scorer_names and total_weight > 0:
        aggregate = (
            sum(per_judge[name] * weights.get(name, 1.0) for name in aggregate_scorer_names)
            / total_weight
        )
    else:
        aggregate = 0.0

    bucket_rows: dict[str, list[int]] = defaultdict(list)
    for i in range(n_rows):
        category = _category_for_row(result_df.iloc[i], examples, i)
        if category:
            bucket_rows[category].append(i)
    per_bucket: dict[str, dict[str, float]] = {}
    for bucket, idxs in bucket_rows.items():
        per_bucket[bucket] = {
            name: _mean([per_judge_rows[name][i] for i in idxs]) for name in scorer_names
        }

    failures: list[dict[str, Any]] = []
    trace_ids: list[str] = []
    for i in range(n_rows):
        row = result_df.iloc[i]
        trace_id = row.get("trace_id") if hasattr(row, "get") else None
        if trace_id:
            trace_ids.append(str(trace_id))
        judge_failures = [
            name for name in scorer_names if (s := per_judge_rows[name][i]) is not None and s < 1.0
        ]
        if not judge_failures:
            continue
        category = _category_for_row(row, examples, i)
        example_id = examples[i]["example_id"] if i < len(examples) else ""
        query = examples[i]["query"] if i < len(examples) else ""
        failures.append(
            {
                "example_id": example_id,
                "query": query,
                "category": category,
                "judge_failures": judge_failures,
                "trace_id": trace_id,
            }
        )

    return EvalReport(
        aggregate=aggregate,
        per_judge=per_judge,
        per_bucket=per_bucket,
        failures=failures,
        run_id=run_id,
        experiment_id=experiment_id,
        n_rows=n_rows,
        mode=mode,
        scorers=list(scorer_names),
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        trace_ids=trace_ids,
        cost_metrics={
            "total_context_chars": float(
                sum(len(str(ex.get("query", ""))) for ex in examples[:n_rows])
            ),
            "n_rows": float(n_rows),
        },
        scorer_fingerprint=scorer_fingerprint,
    )


# ---------------------------------------------------------------------------
# Code-mode agent loading
# ---------------------------------------------------------------------------


def _import_agent_module(module_path: str) -> ModuleType:
    """Import an agent module from a dotted path or a ``.py`` file path.

    A dotted path (e.g. ``anvil.agents.baseline``) is resolved via
    :func:`importlib.import_module`. A path containing a separator or
    ending in ``.py`` is loaded from disk via ``spec_from_file_location``
    — this is how FORGE loads candidate modules the optimizer just wrote
    to ``agents/`` that are not yet installed packages.
    """
    if module_path.endswith(".py") or "/" in module_path or os.sep in module_path:
        path = Path(module_path)
        if not path.is_file():
            raise FileNotFoundError(f"agent module not found: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for agent module: {path}")
        module = importlib.util.module_from_spec(spec)
        # Register before exec so @dataclass, __init_subclass__, and runtime
        # type-resolution mechanisms that look up the module in sys.modules
        # work during import. Mirrors importlib.import_module's contract.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # Remove the broken module so a retry doesn't find a partially-
            # initialized entry.
            sys.modules.pop(spec.name, None)
            raise
        return module
    return importlib.import_module(module_path)


def _find_memory_system_subclass(module: ModuleType) -> type[MemorySystem]:
    """Find the concrete ``MemorySystem`` subclass defined in ``module``.

    The class must be *defined* in this module (``__module__`` match) so
    that a re-exported base class or an imported helper does not get
    mistaken for the agent. Exactly one subclass is expected; zero or
    multiple are configuration errors.
    """
    candidates: list[type[MemorySystem]] = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, MemorySystem)
            and obj is not MemorySystem
            and getattr(obj, "__module__", None) == module.__name__
            and not inspect.isabstract(obj)
        ):
            candidates.append(obj)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"no concrete MemorySystem subclass found in agent module {module.__name__!r}"
        )
    raise ValueError(
        f"multiple concrete MemorySystem subclasses found in {module.__name__!r}: "
        f"{[c.__name__ for c in candidates]}"
    )


def _load_memory_system(
    module_path: str,
    *,
    llm_client: OpenAI | None = None,
    model: str = "",
) -> MemorySystem:
    """Import an agent module and instantiate its ``MemorySystem`` subclass.

    ``module_path`` is either a dotted Python module path (e.g.
    ``anvil.agents.baseline``) or a ``.py`` file path. The module must
    define exactly one concrete ``MemorySystem`` subclass, which is
    instantiated with ``llm_client`` and ``model`` as constructor kwargs.
    """
    module = _import_agent_module(module_path)
    cls = _find_memory_system_subclass(module)
    return cls(llm_client=llm_client, model=model)


# mlflow reads this env var to size the predict/score thread pools inside
# ``mlflow.genai.evaluate`` (default 10 when unset). anvil wires
# ``eval.n_workers`` into it so the configured value controls concurrency
# rather than mlflow's default.
_MLFLOW_MAX_WORKERS_ENV = "MLFLOW_GENAI_EVAL_MAX_WORKERS"

# Name of the per-row root span ``evaluate_branch`` wraps every
# ``predict_fn`` invocation in. The span yields a real per-row trace
# carrying the ``RETRIEVER`` span that ``RetrievalGroundedness`` scores;
# ``_resilient_eval_harness`` (PR #21) is the safety net that prevents a
# crash if a row's trace is still None. See ``evaluate_branch``.
_PREDICT_SPAN_NAME = "anvil.predict"


@contextmanager
def _synchronous_trace_logging():
    """Temporarily force MLflow trace export to complete synchronously."""
    previous = os.environ.get(_MLFLOW_ASYNC_TRACE_LOGGING_ENV)
    os.environ[_MLFLOW_ASYNC_TRACE_LOGGING_ENV] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_MLFLOW_ASYNC_TRACE_LOGGING_ENV, None)
        else:
            os.environ[_MLFLOW_ASYNC_TRACE_LOGGING_ENV] = previous


@contextmanager
def _resilient_eval_harness():
    """Scope a defensive shim around ``mlflow.genai.evaluate``'s harness.

    Workaround for a known mlflow 3.11.x bug (verified against 3.11.1, the
    newest in-range release on the internal proxy — no patch bump is
    available to fix this). When ``predict_fn`` is supplied, the harness
    retrieves each row's trace via ``mlflow.get_trace(request_id,
    silent=True)`` (``harness._run_predict``, ~line 782). On the Databricks
    Tracing Server that trace is sometimes not retrievable at scoring time
    — even with synchronous export forced from process start (see
    ``anvil/__init__.py``) and PR #16's root span — leaving
    ``eval_item.trace`` None for some rows. The harness then dereferences it
    without a None check and aborts the whole run:

    * ``_get_new_expectations`` (``harness.py``:934-942) reads
      ``eval_item.trace.info.assessments`` and raises
      ``AttributeError: 'NoneType' object has no attribute 'info'`` — the
      live ``make_baseline`` crash, typically ~row 2-3 of 8.

    This context manager monkeypatches two harness symbols, scoped to the
    ``mlflow.genai.evaluate`` call (restored on exit — NOT a global
    import-time patch), so a missing per-row trace never crashes the run:

    1. ``_get_new_expectations`` → a None-safe wrapper that yields ``[]``
       (no trace-derived expectations) for a None-trace row instead of
       raising, and delegates to the original implementation otherwise.
       This directly neutralizes the confirmed crash site. Rows WITH a
       trace are scored normally — ``RetrievalGroundedness`` and the other
       scorers are NOT globally disabled; a None-trace row simply
       contributes no expectations and its scorers run as-is (scorer
       exceptions are already caught by the harness at ``run_scorer``:874
       and recorded as error feedbacks, never aborting the run).

    2. ``_run_predict`` → a wrapper that, after the original runs, falls
       back to ``create_minimal_trace(eval_item)`` when
       ``mlflow.get_trace(request_id)`` returned None. This is the SAME
       fallback the static-dataset path uses (``harness.py``:795) but the
       ``predict_fn`` path omits. ``create_minimal_trace`` fetches the
       trace by its own just-created ``trace_id`` under
       ``is_evaluate=True`` (synchronous export) — the reliable retrieval
       mechanism, not the failing request_id lookup. This ensures every
       row carries a trace so the eval COMPLETES with a real result
       DataFrame, instead of merely moving the crash one step downstream
       into ``batch_link_traces_to_run`` (``trace_utils.py``:1014, an
       unguarded ``eval_item.trace.info.trace_id`` list-comprehension) or
       ``construct_eval_result_df`` (``trace_utils.py``:925, caught but
       yields a None DataFrame that breaks ``_aggregate_report``).

    The shim (1) is the direct guard against the confirmed crash; the
    fallback (2) is the root-cause fix that prevents the crash from
    relocating. Together they bring the ``predict_fn`` path to the same
    per-row-trace reliability the production static-dataset path already
    relies on.
    """
    import mlflow.genai.evaluation.harness as _harness
    from mlflow.genai.utils.trace_utils import create_minimal_trace

    _orig_get_new_expectations = _harness._get_new_expectations
    _orig_run_predict = _harness._run_predict

    def _get_new_expectations_none_safe(eval_item):
        # mlflow 3.11.x harness.py:936 derefs ``eval_item.trace.info.assessments``
        # without a None check. A row whose trace the Databricks backend did not
        # return leaves ``eval_item.trace`` None and crashes here. Yield no
        # expectations for that row instead of raising; rows with a trace are
        # scored normally via the original implementation.
        if eval_item.trace is None:
            return []
        return _orig_get_new_expectations(eval_item)

    def _run_predict_with_minimal_trace_fallback(
        eval_item, predict_fn, run_id, rate_limiter, max_retries=0, experiment_id=None
    ):
        _orig_run_predict(eval_item, predict_fn, run_id, rate_limiter, max_retries, experiment_id)
        # harness.py:782 sets ``eval_item.trace = mlflow.get_trace(request_id)``.
        # On the Databricks backend that returns None for some rows. The
        # static-dataset path (harness.py:795) falls back to a minimal trace;
        # the predict_fn path does not, so apply the same fallback here. This
        # fetches by the just-created trace_id (reliable, sync), not request_id.
        if predict_fn is not None and eval_item.trace is None:
            eval_item.trace = create_minimal_trace(eval_item)

    _harness._get_new_expectations = _get_new_expectations_none_safe
    _harness._run_predict = _run_predict_with_minimal_trace_fallback
    try:
        yield
    finally:
        _harness._get_new_expectations = _orig_get_new_expectations
        _harness._run_predict = _orig_run_predict


def _run_predictions_parallel(
    predict_fn: Callable[[str], str],
    queries: list[str],
    n_workers: int = 1,
) -> list[str]:
    """Run ``predict_fn`` across ``queries`` in parallel.

    Uses :class:`concurrent.futures.ThreadPoolExecutor` — the runtime
    agent's work is I/O-bound (LLM / tool HTTP calls), so threads are
    sufficient and avoid the serialization overhead of processes. When
    ``n_workers <= 1`` the function runs sequentially (backward compatible
    with the pre-parallel eval path).

    Results preserve input order regardless of completion order: each
    future is keyed by its input index, so the slot it writes is fixed. A
    prediction that raises is recorded as an empty string and logged, so
    one bad row does not abort the whole eval — mirroring mlflow's own
    per-row error isolation in ``_run_predict``.

    Thread-safety: ``predict_fn`` must be safe to invoke from multiple
    threads concurrently. For prompt mode ``AnvilAgent.predict`` issues
    stateless HTTP calls against the runtime endpoint (thread-safe); for
    code mode a ``MemorySystem.predict`` subclass is thread-safe as long
    as it does not mutate shared state inside ``predict``.

    Note:
        The live ``evaluate_branch`` flow delegates predict parallelism to
        mlflow's own harness (sized via ``MLFLOW_GENAI_EVAL_MAX_WORKERS``)
        so that mlflow builds a per-row trace carrying the ``RETRIEVER``
        span that ``RetrievalGroundedness`` scores against. Pre-computing
        outputs here and passing them as a static dataset would yield a
        root-span-only trace and make ``RetrievalGroundedness`` raise, so
        this primitive is exercised directly by the unit tests and is
        available for offline/pre-compute paths that do not need traces.
    """
    if n_workers <= 1:
        # Sequential path — same per-row error isolation as the parallel
        # path so the acceptance contract ("a prediction that raises is
        # recorded as an empty string and does not abort the whole eval")
        # holds uniformly for both paths, not just the parallel one.
        results = []
        for i, q in enumerate(queries):
            try:
                results.append(predict_fn(q))
            except Exception as exc:  # noqa: BLE001 — isolate per-row failures
                logger.warning("prediction failed for row %s: %s", i, exc)
                results.append("")
        return results

    results: list[str | None] = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(predict_fn, q): i for i, q in enumerate(queries)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001 — isolate per-row failures
                logger.warning("prediction failed for row %s: %s", idx, exc)
                results[idx] = ""
    # as_completed yields every submitted future, so every slot is filled
    # — with a result on success or "" on failure — before we reach here.
    assert all(r is not None for r in results)
    return results


def evaluate_branch(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    kb_dir: Path | str = "data/kb",
    golden_set_path: Path | str = "data/golden_set.jsonl",
    evaluator_path: str | Path | None = None,
    profile: str | None = None,
    mode: str | None = None,
    allow_test: bool = False,
    include_safety: bool = False,
    runtime_client: OpenAI | None = None,
    judge_client: OpenAI | None = None,
) -> EvalReport:
    """Run the active scorers against a sub-set of the golden set."""
    scaffold_path = Path(scaffold_root)
    runtime_path = (
        Path(runtime_config_path)
        if runtime_config_path is not None
        else default_runtime_config_path(scaffold_path)
    )

    snapshot = load_harness(scaffold_path, runtime_path)

    # Engine dispatch (domain-agnostic). ``genai`` is the built-in default
    # implemented by the rest of this function; any other engine is a
    # pluggable domain resolved through the registry, which lazily imports
    # its ``anvil.domains.<name>`` package (so no domain is named here and
    # no domain tree — e.g. savesage's statement-agent — is touched on the
    # genai path). A pluggable engine returns the same ``EvalReport`` the
    # loop consumes and bypasses the mlflow.genai.evaluate path below.
    engine_name = snapshot.config.eval.engine
    if engine_name != GENAI_ENGINE:
        engine_fn = load_engine(engine_name)
        return engine_fn(
            scaffold_root=scaffold_path,
            runtime_config_path=runtime_path,
            golden_set_path=golden_set_path,
            profile=profile,
            mode=mode,
        )

    cfg: EvalConfig = snapshot.config.eval
    selected_mode = mode or cfg.default_mode
    if selected_mode == "test" and not allow_test:
        raise ValueError("test mode is held out and may only be run by explicit finalization")
    if selected_mode not in cfg.modes:
        raise ValueError(
            f"mode {selected_mode!r} not in harness/config.yaml > eval.modes ({list(cfg.modes)})"
        )
    if profile:
        mlflow.set_tracking_uri(f"databricks://{profile}")
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    mlflow.set_experiment(snapshot.config.experiments.eval)

    enable_runtime_tracing()

    # Both the runtime agent and the judge route through the AI Gateway
    # client (the sole LLM route). The gateway resolves host + token from
    # the environment; when ``profile`` is set above it is already in
    # ``DATABRICKS_CONFIG_PROFILE``, which the SDK honors at token-refresh
    # time. ``profile`` is therefore not passed to the factory.
    runtime_client = runtime_client or build_gateway_client()
    judge_client = judge_client or build_gateway_client()

    examples = load_golden_set(golden_set_path)
    selected = _select_mode_examples(examples, cfg=cfg, selected_mode=selected_mode)

    if snapshot.config.mode == "code":
        # Code mode: import the active MemorySystem subclass and call
        # predict() per row instead of the LLM agent tool-calling loop.
        # The same scorers (programmatic + LLM judges) score the output.
        memory_system = _load_memory_system(
            snapshot.config.agent_module,
            llm_client=runtime_client,
            model=snapshot.config.runtime_endpoint,
        )

        def predict_fn(query: str, **_kwargs: Any) -> str:
            # Wrap in an explicit root span so every row yields a trace.
            # See the prompt-mode predict_fn below for the full rationale.
            with mlflow.start_span(name=_PREDICT_SPAN_NAME, span_type=SpanType.CHAIN) as span:
                span.set_inputs({"query": query})
                answer, _metadata = memory_system.predict(query)
                span.set_outputs({"response": answer})
                return answer
    else:
        # Prompt mode: compose the system prompt from scaffold/ and run
        # the AnvilAgent tool-calling loop against the runtime endpoint.
        tool_executor = make_kb_executor(kb_dir)
        agent = AnvilAgent(
            scaffold_root=scaffold_path,
            runtime_config_path=runtime_path,
            source=SOURCE_EVAL,
            client=runtime_client,
            tool_executor=tool_executor,
        )

        def predict_fn(query: str, **_kwargs: Any) -> str:
            # Wrap each row in an explicit root CHAIN span so the row
            # yields a real per-row trace carrying the ``RETRIEVER`` span
            # that ``RetrievalGroundedness`` scores. The harness retrieves
            # each row's trace via ``mlflow.get_trace(request_id)``; without
            # this span a row can leave ``eval_item.trace`` None.
            #
            # ``_resilient_eval_harness`` (PR #21) is the safety net: it
            # makes ``_get_new_expectations`` None-safe and falls back to a
            # minimal trace, so a None-trace row no longer crashes the run.
            # But the minimal trace lacks the ``RETRIEVER`` span, so
            # ``RetrievalGroundedness`` is degraded for those rows — this
            # span remains the primary guarantee.
            #
            # It also supersedes the fragile ``mlflow.openai.autolog`` path
            # (``enable_runtime_tracing``): ``AnvilAgent.predict`` calls
            # ``tag_current_trace`` (``mlflow.update_current_trace``) before
            # any chat call, when no span is active — the "No active trace
            # found" warning — and on the live backend the autolog trace was
            # not retrievable by the row's request id. This root span gives
            # ``tag_current_trace`` an active trace to tag (the warning
            # disappears) and nests autolog's CHAT_MODEL spans and the
            # ``search_knowledge_base`` RETRIEVER span under one coherent
            # per-row trace. The span ends (and the trace exports) when the
            # ``with`` block exits — async logging is disabled during eval
            # (``is_evaluate=True``), so the trace is available immediately.
            with mlflow.start_span(name=_PREDICT_SPAN_NAME, span_type=SpanType.CHAIN) as span:
                span.set_inputs({"query": query})
                request = ResponsesAgentRequest(
                    input=[{"type": "message", "role": "user", "content": query}]
                )
                response = agent.predict(request)
                text = _extract_final_text(response)
                span.set_outputs({"response": text})
                return text

    aggregate_scorer_configs = list(cfg.scorers)
    aggregate_scorer_names = [c.name for c in aggregate_scorer_configs]
    weights = {c.name: c.weight for c in aggregate_scorer_configs}
    scorer_fingerprint = compute_scorer_fingerprint(aggregate_scorer_configs)
    active_scorer_configs = list(aggregate_scorer_configs)
    active_scorer_names = list(aggregate_scorer_names)
    if include_safety and "safety" not in active_scorer_names:
        active_scorer_configs.append(ScorerConfig(name="safety"))
        active_scorer_names.append("safety")

    scorers = build_scorers(
        judge_client=judge_client,
        judge_model=snapshot.config.judge_endpoint,
        scorer_configs=active_scorer_configs,
        evaluator_path=evaluator_path,
    )
    dataset = _build_dataset(selected)

    # Wire anvil's ``eval.n_workers`` into mlflow's parallel predict/score
    # pool. mlflow's harness already runs ``predict_fn`` per row in a
    # ``ThreadPoolExecutor`` sized by ``MLFLOW_GENAI_EVAL_MAX_WORKERS``
    # (default 10); setting it from the config makes the configured value
    # actually control concurrency. We keep passing ``predict_fn`` (not
    # pre-computed ``outputs``) so mlflow builds a per-row trace carrying
    # the ``RETRIEVER`` span that ``RetrievalGroundedness`` requires — a
    # static-dataset trace is root-span-only and makes that scorer raise.
    # The env var is saved/restored so the override is scoped to this call.
    # NOTE: the env var is process-global, so this override is not safe for
    # concurrent ``evaluate_branch`` calls in one process; the optimizer
    # runs rounds/evals synchronously, so this is not a live issue today.
    n_workers = max(1, cfg.n_workers)
    _prev_workers = os.environ.get(_MLFLOW_MAX_WORKERS_ENV)
    os.environ[_MLFLOW_MAX_WORKERS_ENV] = str(n_workers)
    try:
        # On the Databricks Tracing Server, async export can race the eval
        # harness's immediate per-row get_trace(request_id). A missing trace
        # then reaches scoring as None and crashes _get_new_expectations.
        # Keep export synchronous until evaluate has finished reading traces
        # (PR #17; the env var is also forced from process start in
        # anvil/__init__.py because the exporter caches the flag at construction).
        # ``_resilient_eval_harness`` is the guarantee: even if a row's trace
        # is still None despite the above, the harness shim yields no
        # expectations for that row (no crash) and the _run_predict fallback
        # synthesizes a minimal trace so the run completes with a real
        # result DataFrame. See its docstring for the exact harness.py
        # symbols and lines patched.
        with _resilient_eval_harness(), _synchronous_trace_logging():
            result = mlflow.genai.evaluate(
                data=dataset,
                scorers=scorers,
                predict_fn=predict_fn,
            )
    finally:
        if _prev_workers is None:
            os.environ.pop(_MLFLOW_MAX_WORKERS_ENV, None)
        else:
            os.environ[_MLFLOW_MAX_WORKERS_ENV] = _prev_workers

    experiment = mlflow.get_experiment_by_name(snapshot.config.experiments.eval)
    return _aggregate_report(
        result_df=result.result_df,
        metrics=result.metrics,
        scorer_names=active_scorer_names,
        aggregate_scorer_names=aggregate_scorer_names,
        weights=weights,
        examples=selected,
        run_id=result.run_id,
        experiment_id=experiment.experiment_id if experiment else "",
        mode=selected_mode,
        scorer_fingerprint=scorer_fingerprint,
    )
