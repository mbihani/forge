"""Trace eval engine — re-run the mutated agent over frozen trace inputs.

Option A (traces-as-dataset + re-run): FORGE mutates the agent each round,
so a trace's stored assessment scored the OLD agent. Replaying those stored
scores would give ``score_delta ≡ 0`` and every round would auto-revert. So
the frozen snapshot supplies the dataset + scoring TARGETS, and each round
we RE-RUN the mutated agent over the trace inputs and RE-SCORE. This is
structurally identical to ``evaluate_savesage``:

    read frozen snapshot → select mode subset → re-run agent per row
    (parallel) → score each objective → weighted macro-aggregate → EvalReport

Scoring maps each assessment to exactly one stable ``per_judge`` key:

* HUMAN Expectation  → ``ref_<name>``   — deterministic reference compare.
* HUMAN Feedback     → ``human_<name>`` — deterministic hard-label compare.
* LLM/AI_JUDGE Feedback → ``judge_<name>`` — re-run the SAME rubric judge
  (rationale-seeded) against the NEW output.

The aggregate is a WEIGHTED mean over every ``per_judge`` key; human keys
(``ref_*`` / ``human_*``) weigh ``min_human_weight`` (default 2.0) and judge
keys weigh 1.0, so humans dominate (DECISION #3). ALL judge objectives are
in the aggregate. ``scorer_fingerprint`` is a REAL hash of (sorted key set +
weights + snapshot content hash) so re-ingesting a different snapshot
invalidates the cached baseline (round.py:240-250).

The re-run and judge paths are injectable (``predict_fn`` / ``judge_fn``) so
the mapping/aggregation logic is unit-testable with no live gateway.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anvil.data import select_subset
from anvil.domains.trace.ingest import read_snapshot, snapshot_content_hash
from anvil.eval.runner import EvalReport
from anvil.runtime.loader import load_harness

logger = logging.getLogger(__name__)

# Prefixes that mark a "human" objective (weighted above judges).
_HUMAN_PREFIXES = ("ref_", "human_")

_JUDGE_PROMPT_TEMPLATE = """\
You are re-scoring an agent's NEW response using the SAME rubric a prior
judge applied. Judge only the rubric named below; do not invent new axes.

RUBRIC NAME: {rubric}
PRIOR JUDGE NOTES (rationale seed, may reference an older response):
{rationale}

USER QUERY:
{query}

NEW AGENT RESPONSE:
{response}

Decide whether the NEW response satisfies the rubric. Output JSON ONLY (no
prose, no code fences):
{{"verdict": "pass" | "fail", "rationale": "<one short sentence>"}}
"""


def _slug(name: str) -> str:
    """Stable slug for an objective name (lowercase, non-alnum → ``_``)."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return s or "unnamed"


# ---------------------------------------------------------------------------
# Pure scoring / aggregation — no mlflow, no gateway, unit-testable.
# ---------------------------------------------------------------------------


def _normalize(text: Any) -> str:
    """Casefold + collapse whitespace for a deterministic text compare."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def reference_match(output: str, reference: Any) -> float:
    """Deterministic 1.0/0.0 compare of ``output`` against a reference label.

    Used for HUMAN Expectation (expected-response reference text) and for
    categorical (string) hard labels. A string reference matches when it
    equals the output (normalized) or appears as a normalized substring of
    it (reference labels are often a key fact the answer must contain). A
    non-string reference is compared by canonical-JSON equality.
    """
    if isinstance(reference, str):
        ref = _normalize(reference)
        out = _normalize(output)
        if not ref:
            return 1.0 if not out else 0.0
        return 1.0 if (ref == out or ref in out) else 0.0
    # Non-string reference: exact structural match against the output text
    # parsed as JSON, else against its canonical form.
    ref_canon = json.dumps(reference, sort_keys=True)
    try:
        parsed = json.loads(output)
        if json.dumps(parsed, sort_keys=True) == ref_canon:
            return 1.0
    except (ValueError, TypeError):
        pass
    return 1.0 if _normalize(output) == _normalize(ref_canon) else 0.0


_TRUE_TOKENS = frozenset({"true", "yes", "y", "pass", "correct", "1"})
_FALSE_TOKENS = frozenset({"false", "no", "n", "fail", "incorrect", "0"})


def _extract_bool(output: str) -> bool | None:
    """Read a boolean verdict out of the NEW output, or None if absent."""
    try:
        val = json.loads(str(output).strip())
        if isinstance(val, bool):
            return val
    except (ValueError, TypeError):
        pass
    norm = _normalize(output)
    if norm in _TRUE_TOKENS:
        return True
    if norm in _FALSE_TOKENS:
        return False
    # First alphanumeric token (punctuation-insensitive), e.g. "yes," -> "yes".
    tokens = re.findall(r"[a-z0-9]+", norm)
    first = tokens[0] if tokens else ""
    if first in _TRUE_TOKENS:
        return True
    if first in _FALSE_TOKENS:
        return False
    return None


def _extract_number(output: str) -> float | None:
    """Read the first numeric value out of the NEW output, or None."""
    try:
        val = json.loads(str(output).strip())
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    except (ValueError, TypeError):
        pass
    match = re.search(r"-?\d+(?:\.\d+)?", str(output))
    return float(match.group()) if match else None


def hard_label_match(output: str, label: Any) -> float:
    """Programmatic 1.0/0.0 check of the NEW output against a human hard label.

    A hard label records the ground-truth verdict/value the agent should
    produce, so — unlike a reference-text Expectation — it is scored by
    reading the corresponding signal *out of the new output* and comparing
    it to the label, not by string-comparing the output to the label's
    printed form. This is what makes a ``human_*`` objective actually move
    when the mutated agent's output changes (Decision #3):

    * boolean label → parse a boolean verdict from the output;
    * numeric label → parse the first number from the output (exact match);
    * categorical / text label → :func:`reference_match`.

    ``bool`` is checked before ``int``/``float`` because ``bool`` is an
    ``int`` subclass.
    """
    if isinstance(label, bool):
        got = _extract_bool(output)
        return 1.0 if got is not None and got == label else 0.0
    if isinstance(label, (int, float)):
        got = _extract_number(output)
        if got is None:
            return 0.0
        return 1.0 if abs(got - float(label)) <= 1e-9 else 0.0
    return reference_match(output, label)


def build_key_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """Map each ``(prefix, assessment-name)`` to a UNIQUE, stable objective key.

    The base key is ``<prefix>_<slug(name)>``. Because the slug is lossy,
    two distinct names can share a base (e.g. ``correctness-v1`` and
    ``correctness v1`` both slug to ``correctness_v1``). When that happens
    every colliding name is disambiguated with a short stable hash of its
    ORIGINAL name (``<base>__<6-hex>``), so every distinct assessment keeps
    its own frontier objective (Decision #3: all objectives count
    independently) and never silently averages into another. Deriving the
    map from the frozen, deterministically-selected rows keeps the key set
    identical every round (the HARD CONSTRAINT).
    """
    by_prefix: dict[str, set[str]] = {"ref": set(), "human": set(), "judge": set()}
    for row in rows:
        for exp in row.get("expectations", []):
            by_prefix["ref"].add(str(exp["name"]))
        for lab in row.get("human_labels", []):
            by_prefix["human"].add(str(lab["name"]))
        for rub in row.get("judge_rubrics", []):
            by_prefix["judge"].add(str(rub["name"]))

    key_map: dict[tuple[str, str], str] = {}
    for prefix, names in by_prefix.items():
        groups: dict[str, list[str]] = {}
        for name in names:
            groups.setdefault(_slug(name), []).append(name)
        for slug, group in groups.items():
            if len(group) == 1:
                key_map[(prefix, group[0])] = f"{prefix}_{slug}"
            else:
                for name in group:
                    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:6]
                    key_map[(prefix, name)] = f"{prefix}_{slug}__{digest}"
    return key_map


def objective_keys(rows: list[dict[str, Any]]) -> list[str]:
    """The stable, sorted set of ``per_judge`` keys across ``rows``."""
    return sorted(build_key_map(rows).values())


def score_row(
    row: dict[str, Any],
    new_output: str,
    judge_fn: Callable[..., float],
    key_map: dict[tuple[str, str], str] | None = None,
) -> dict[str, float]:
    """Score one row's objectives against ``new_output``.

    ``judge_fn(*, query, output, rubric)`` re-runs a rubric judge and
    returns a score in ``[0, 1]``; it is injected so this stays pure.
    ``key_map`` (from :func:`build_key_map`) supplies the collision-safe
    objective keys; when omitted it is derived from this row alone (fine
    for isolated unit tests). When an objective key repeats within a row,
    the scores are averaged.

    HUMAN Expectation is scored with :func:`reference_match` (reference
    text); HUMAN Feedback with :func:`hard_label_match` (type-aware hard
    label); judge rubrics are re-run via ``judge_fn``.
    """
    if key_map is None:
        key_map = build_key_map([row])
    contributions: dict[str, list[float]] = {}

    def _add(key: str, score: float) -> None:
        contributions.setdefault(key, []).append(float(score))

    for exp in row.get("expectations", []):
        key = key_map[("ref", str(exp["name"]))]
        _add(key, reference_match(new_output, exp.get("value")))
    for lab in row.get("human_labels", []):
        key = key_map[("human", str(lab["name"]))]
        _add(key, hard_label_match(new_output, lab.get("value")))
    for rub in row.get("judge_rubrics", []):
        key = key_map[("judge", str(rub["name"]))]
        try:
            score = judge_fn(query=row.get("query", ""), output=new_output, rubric=rub)
        except Exception as exc:  # noqa: BLE001 - isolate per-rubric judge failures
            logger.warning("judge failed for rubric %r: %s", rub.get("name"), exc)
            score = 0.0
        _add(key, score)

    return {k: sum(v) / len(v) for k, v in contributions.items()}


def compute_weights(keys: list[str], min_human_weight: float) -> dict[str, float]:
    """Weight per objective: human keys ``min_human_weight``, judges ``1.0``."""
    return {
        k: (min_human_weight if k.startswith(_HUMAN_PREFIXES) else 1.0) for k in keys
    }


def aggregate_report(
    rows: list[dict[str, Any]],
    row_scores: list[dict[str, float]],
    *,
    min_human_weight: float,
    key_map: dict[tuple[str, str], str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Macro-average per objective, then weighted-mean into the aggregate.

    ``per_judge[key]`` is the mean of that objective's per-row scores over
    the rows that carry it. The aggregate is the weighted mean over the
    full (stable) key set with human objectives weighted above judges.
    ``key_map`` must be the SAME map passed to :func:`score_row` so the
    aggregated keys line up with the scored keys.
    """
    keys = sorted((key_map or build_key_map(rows)).values())
    accum: dict[str, list[float]] = {k: [] for k in keys}
    for scores in row_scores:
        for key, val in scores.items():
            if key in accum:
                accum[key].append(val)
    per_judge = {k: (sum(v) / len(v) if v else 0.0) for k, v in accum.items()}

    weights = compute_weights(keys, min_human_weight)
    total_w = sum(weights.values())
    aggregate = (
        sum(per_judge[k] * weights[k] for k in keys) / total_w if total_w else 0.0
    )
    return aggregate, per_judge


def compute_trace_fingerprint(
    keys: list[str], min_human_weight: float, snapshot_hash: str
) -> str:
    """Real fingerprint = stable hash of (sorted keys + weights + snapshot hash).

    Unlike ``evaluate_savesage`` (which zeroes the fingerprint), this makes
    ``round.py`` correctly invalidate the cached baseline whenever a
    different snapshot is ingested — the reproducibility guarantee.
    """
    payload = {
        "keys": sorted(keys),
        "weights": compute_weights(sorted(keys), min_human_weight),
        "snapshot": snapshot_hash,
    }
    return json.dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# Judge client wiring — routes through the AI Gateway (sole LLM route).
# ---------------------------------------------------------------------------


def _parse_verdict(raw: str) -> float:
    """Parse ``{"verdict": "pass"|"fail"}`` JSON → 1.0 / 0.0."""
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    verdict = obj.get("verdict") if isinstance(obj, dict) else None
    if verdict not in ("pass", "fail"):
        raise ValueError(f"judge verdict missing or invalid: {obj!r}")
    return 1.0 if verdict == "pass" else 0.0


def build_judge_fn(judge_client: Any, fallback_model: str) -> Callable[..., float]:
    """Return a ``judge_fn`` that re-runs a rubric judge via the gateway.

    DECISION #4: re-run the EXACT judge model recorded in the rubric's
    ``source_id``; if that model is not reachable through the gateway, fall
    back to the configured ``judge_endpoint`` and log a warning. All calls
    route through the AI-Gateway client (sole LLM route, fresh SP token).
    """

    def _call(model: str, prompt: str) -> str:
        response = judge_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def judge_fn(*, query: str, output: str, rubric: dict[str, Any]) -> float:
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            rubric=rubric.get("name", ""),
            rationale=rubric.get("rationale") or "(none)",
            query=query,
            response=output,
        )
        requested = rubric.get("source_id") or fallback_model
        try:
            return _parse_verdict(_call(requested, prompt))
        except Exception as exc:  # noqa: BLE001 - fall back to configured judge
            if requested == fallback_model:
                logger.warning("judge model %r failed: %s", requested, exc)
                return 0.0
            logger.warning(
                "judge model %r unreachable (%s); falling back to %r",
                requested,
                exc,
                fallback_model,
            )
            return _parse_verdict(_call(fallback_model, prompt))

    return judge_fn


# ---------------------------------------------------------------------------
# Re-run predictors — reuse the runner's code/prompt compose paths.
# ---------------------------------------------------------------------------


def _build_predictor(
    snapshot: Any,
    *,
    scaffold_path: Path,
    runtime_config_path: Path | str | None,
    runtime_client: Any,
) -> Callable[[str], tuple[str, float | None]]:
    """Return ``predict(query) -> (output_text, latency_ms)`` for the mode.

    The re-run agent (code-mode ``MemorySystem`` / prompt-mode
    ``AnvilAgent``) can hold mutable per-conversation state, so under the
    default parallel path (``n_workers > 1``) a single shared instance
    would race and leak state across rows. Each worker thread therefore
    constructs and caches its OWN agent in thread-local storage — the
    gateway ``runtime_client`` is stateless (fresh OpenAI client + token
    per request) and is safely shared.
    """
    local = threading.local()

    if snapshot.config.mode == "code":
        from anvil.eval.runner import _load_memory_system  # noqa: PLC0415

        agent_module = snapshot.config.agent_module
        model = snapshot.config.runtime_endpoint

        def _get_memory_system() -> Any:
            inst = getattr(local, "agent", None)
            if inst is None:
                inst = _load_memory_system(
                    agent_module, llm_client=runtime_client, model=model
                )
                local.agent = inst
            return inst

        def _predict(query: str) -> tuple[str, float | None]:
            memory_system = _get_memory_system()
            answer, meta = memory_system.predict(query)
            latency = meta.get("latency_ms") if isinstance(meta, dict) else None
            return (answer if isinstance(answer, str) else str(answer)), (
                float(latency) if latency is not None else None
            )

        return _predict

    from mlflow.types.responses import ResponsesAgentRequest  # noqa: PLC0415

    from anvil.eval.runner import _extract_final_text  # noqa: PLC0415
    from anvil.observability import SOURCE_EVAL  # noqa: PLC0415
    from anvil.runtime.agent import AnvilAgent  # noqa: PLC0415

    def _get_agent() -> Any:
        inst = getattr(local, "agent", None)
        if inst is None:
            inst = AnvilAgent(
                scaffold_root=scaffold_path,
                runtime_config_path=runtime_config_path,
                source=SOURCE_EVAL,
                client=runtime_client,
            )
            local.agent = inst
        return inst

    def _predict(query: str) -> tuple[str, float | None]:
        agent = _get_agent()
        request = ResponsesAgentRequest(
            input=[{"type": "message", "role": "user", "content": query}]
        )
        response = agent.predict(request)
        return _extract_final_text(response), None

    return _predict


def _select(examples: list[dict], *, rows: int, buckets: dict[str, int]) -> list[dict]:
    """Deterministic subset: bucketed when configured, else first ``rows``."""
    if buckets:
        return select_subset(examples, buckets=buckets)
    return examples[:rows] if rows else examples


def _resolve_snapshot_path(snapshot_path: str, repo_root: Path) -> Path:
    p = Path(snapshot_path)
    return p if p.is_absolute() else repo_root / p


def evaluate_trace(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    golden_set_path: Path | str = "data/golden_set.jsonl",  # noqa: ARG001 - trace uses snapshot_path
    profile: str | None = None,  # noqa: ARG001 - gateway auth uses env/profile chain
    mode: str | None = None,
    predict_fn: Callable[[str], tuple[str, float | None]] | None = None,
    judge_fn: Callable[..., float] | None = None,
    runtime_client: Any | None = None,
    judge_client: Any | None = None,
    **_kwargs: Any,  # absorb genai-only kwargs per the engine contract
) -> EvalReport:
    """Re-run the mutated agent over the frozen trace snapshot and score it.

    NEVER calls ``search_traces`` at run time (reproducibility) — reads the
    frozen snapshot at ``eval.trace.snapshot_path`` written once by
    ``scripts/ingest_traces.py``.
    """
    scaffold_path = Path(scaffold_root)
    repo_root = scaffold_path.parent
    snapshot = load_harness(scaffold_path, runtime_config_path)
    cfg = snapshot.config.eval
    trace_cfg = cfg.trace
    if trace_cfg is None:
        raise ValueError(
            "eval.engine is 'trace' but eval.trace is not configured in "
            "harness/config.yaml (need at least eval.trace.snapshot_path)"
        )

    selected_mode = mode or cfg.default_mode
    if selected_mode not in cfg.modes:
        raise ValueError(
            f"mode {selected_mode!r} not in harness/config.yaml > eval.modes ({list(cfg.modes)})"
        )
    mode_cfg = cfg.modes[selected_mode]

    snapshot_file = _resolve_snapshot_path(trace_cfg.snapshot_path, repo_root)
    all_rows = read_snapshot(snapshot_file)
    snap_hash = snapshot_content_hash(all_rows)
    selected = _select(all_rows, rows=mode_cfg.rows, buckets=dict(mode_cfg.buckets))
    # One collision-safe key map for the whole run — score_row and
    # aggregate_report MUST share it so scored keys line up with aggregated
    # keys (and stay identical every round).
    key_map = build_key_map(selected)

    # Re-run predictor (code/prompt) and rubric judge — injectable for tests.
    if predict_fn is None:
        from anvil.runtime.client import build_gateway_client  # noqa: PLC0415

        runtime_client = runtime_client or build_gateway_client()
        predict_fn = _build_predictor(
            snapshot,
            scaffold_path=scaffold_path,
            runtime_config_path=runtime_config_path,
            runtime_client=runtime_client,
        )
    if judge_fn is None:
        from anvil.runtime.client import build_gateway_client  # noqa: PLC0415

        judge_client = judge_client or build_gateway_client()
        judge_fn = build_judge_fn(judge_client, snapshot.config.judge_endpoint)

    def _predict_and_score(row: dict[str, Any]) -> dict[str, Any]:
        try:
            output, latency_ms = predict_fn(row["query"])
        except Exception as exc:  # noqa: BLE001 - isolate per-row failures
            logger.warning("prediction failed for %s: %s", row.get("example_id"), exc)
            output, latency_ms = "", None
        scores = score_row(row, output, judge_fn, key_map)
        return {
            "example_id": row["example_id"],
            "query": row["query"],
            "category": row.get("category", "trace"),
            "scores": scores,
            "latency_ms": latency_ms,
        }

    n_workers = max(1, cfg.n_workers)
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
    aggregate, per_judge = aggregate_report(
        selected,
        row_scores,
        min_human_weight=trace_cfg.min_human_weight,
        key_map=key_map,
    )

    # Single "trace" bucket mirrors the savesage per_bucket shape.
    per_bucket: dict[str, dict[str, float]] = {
        "trace": {**per_judge, "aggregate": aggregate}
    }

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

    keys = sorted(key_map.values())
    fingerprint = compute_trace_fingerprint(keys, trace_cfg.min_human_weight, snap_hash)

    cost_metrics: dict[str, float] = {"n_rows": float(len(scored_rows))}
    latencies = sorted(
        r["latency_ms"] for r in scored_rows if r.get("latency_ms") is not None
    )
    if latencies:
        mid = len(latencies) // 2
        cost_metrics["latency_ms_median"] = (
            latencies[mid]
            if len(latencies) % 2
            else (latencies[mid - 1] + latencies[mid]) / 2
        )
        cost_metrics["latency_ms_mean"] = sum(latencies) / len(latencies)

    return EvalReport(
        aggregate=aggregate,
        per_judge=per_judge,
        per_bucket=per_bucket,
        failures=failures,
        run_id="",
        experiment_id="",
        n_rows=len(scored_rows),
        mode=selected_mode,
        scorers=keys,
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        trace_ids=[r["example_id"] for r in scored_rows],
        cost_metrics=cost_metrics,
        scorer_fingerprint=fingerprint,
    )
