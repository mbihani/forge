"""Trace ingestion — turn an agent's MLflow traces into a frozen eval dataset.

This module is the mlflow-facing but **offline-unit-testable** half of the
``trace`` engine. It reads an experiment's traces (which already carry
LLM-judge and human assessments), projects each into a plain row, and
freezes the rows to a JSONL snapshot. The eval run (``eval.py``) reads that
frozen snapshot and NEVER calls the live tracing server, so a round is
reproducible and the ``per_judge`` key set is stable (the HARD CONSTRAINT
the frontier gate depends on).

Only :func:`read_traces` touches mlflow; :func:`trace_to_row`,
:func:`build_snapshot_rows`, :func:`write_snapshot`, :func:`read_snapshot`
and :func:`snapshot_content_hash` are pure and run against hand-built
``Trace`` / :class:`~mlflow.entities.Feedback` /
:class:`~mlflow.entities.Expectation` fixtures with no workspace.

Snapshot row schema (one JSON object per line)::

    {
      "example_id": "<trace_id>",     # STABLE KEY — trace.info.trace_id
      "query": "<single-turn user query>",  # parsed from trace.data.request
      "category": "trace",            # single bucket, for select_subset() compat
      "expectations": [               # HUMAN Expectation — ground-truth refs
        {"name": str, "value": Any, "rationale": str|None, "source_id": str}
      ],
      "human_labels": [               # HUMAN Feedback — hard labels
        {"name": str, "value": Any, "rationale": str|None, "source_id": str}
      ],
      "judge_rubrics": [              # LLM_JUDGE / AI_JUDGE Feedback — rerun-able
        {"name": str, "value": Any, "rationale": str|None,
         "source_id": str, "source_type": str}
      ]
    }

Each Expectation yields a ``ref_<name>`` objective, each human Feedback a
``human_<name>`` objective, and each judge Feedback a ``judge_<name>``
objective in the eval report (see :mod:`anvil.domains.trace.eval`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mlflow.entities import Expectation, Feedback

# Roles that count as a user turn when parsing a chat/Responses request.
_USER_ROLES = frozenset({"user", "human"})
# Roles / item types that mark a request as multi-turn or tool-laden — a
# request carrying any of these is NOT a clean single turn, so we skip it
# rather than guess a projection rule.
#
# TODO(trace-multiturn): multi-turn / tool-laden input projection is future
# scope. Today such a trace is skipped and counted under ``skipped_multiturn``;
# a future revision can define how to fold a conversation into one eval query.
_NON_USER_ROLES = frozenset({"assistant", "system", "tool", "developer"})
_TOOL_ITEM_TYPES = frozenset(
    {"function_call", "function_call_output", "tool_call", "tool_result"}
)
# Text part types accepted inside a structured (list) message content.
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


def read_traces(
    experiment_id: str,
    max_traces: int,
    *,
    client: Any | None = None,
) -> list[Any]:
    """Read up to ``max_traces`` labelled traces from an MLflow experiment.

    Thin wrapper over ``mlflow.search_traces`` /
    ``MlflowClient.search_traces`` scoped by EXPERIMENT_ID ONLY (assume 1
    experiment == 1 agent; ``model_id`` / ``run_id`` are out of scope and
    default to None). Returns only traces carrying at least one assessment
    — a trace with no label cannot be a scoring row.

    When ``client`` is None, uses the module-level ``mlflow.search_traces``
    (single page). When more than one page may be needed, pass an
    :class:`~mlflow.MlflowClient`; its ``search_traces`` is paginated via
    ``token`` and this walks every page up to ``max_traces``.
    """
    if max_traces <= 0:
        raise ValueError("max_traces must be > 0")

    traces: list[Any]
    if client is None:
        import mlflow  # noqa: PLC0415 - lazy so importing this module needs no server

        traces = mlflow.search_traces(
            experiment_ids=[experiment_id],
            max_results=max_traces,
            include_spans=True,
            return_type="list",
        )
    else:
        traces = _search_traces_paginated(client, experiment_id, max_traces)

    return [t for t in traces if _assessments(t)]


def _search_traces_paginated(client: Any, experiment_id: str, max_traces: int) -> list[Any]:
    """Walk ``MlflowClient.search_traces`` pages until ``max_traces`` is met."""
    out: list[Any] = []
    token: str | None = None
    while len(out) < max_traces:
        page = client.search_traces(
            experiment_ids=[experiment_id],
            max_results=max_traces - len(out),
            include_spans=True,
            page_token=token,
        )
        out.extend(list(page))
        # PagedList exposes ``token``; a plain list has none — stop after one page.
        token = getattr(page, "token", None)
        if not token or not page:
            break
    return out[:max_traces]


def _assessments(trace: Any) -> list[Any]:
    """Return ``trace.info.assessments`` defensively (empty list if absent)."""
    info = getattr(trace, "info", None)
    return list(getattr(info, "assessments", None) or [])


def _trace_id(trace: Any) -> str:
    """Return the STABLE key ``trace.info.trace_id`` (never ``request_id``)."""
    info = getattr(trace, "info", None)
    trace_id = getattr(info, "trace_id", None)
    if not trace_id:
        raise ValueError("trace.info.trace_id is missing — cannot key the row")
    return str(trace_id)


def _request_payload(trace: Any) -> Any:
    """Return ``trace.data.request`` (the raw input JSON / string), or None."""
    data = getattr(trace, "data", None)
    return getattr(data, "request", None)


def parse_single_turn_query(request: Any) -> str | None:
    """Project a trace request into a single-turn query string, or None.

    Returns the user's query for a CLEAN SINGLE-TURN request. Returns None
    (the caller counts it under ``skipped_multiturn``) when the request is
    multi-turn, tool-laden, multimodal, or otherwise not a single plain
    user message — we do NOT guess a projection rule.

    Accepted shapes: a plain string; ``{"query": "..."}``; a chat
    ``{"messages": [...]}``; or a Responses ``{"input": [...]}``.
    """
    if request is None:
        return None
    payload = request
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        # A JSON object/array is a structured request; anything else is a
        # bare single-turn query string.
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return payload
        if not isinstance(parsed, (dict, list)):
            return payload
        payload = parsed

    if isinstance(payload, list):
        return _query_from_items(payload)
    if not isinstance(payload, dict):
        return None
    if "messages" in payload and isinstance(payload["messages"], list):
        return _query_from_items(payload["messages"])
    if "input" in payload:
        inp = payload["input"]
        if isinstance(inp, str):
            return inp.strip() or None
        if isinstance(inp, list):
            return _query_from_items(inp)
        return None
    if "query" in payload and isinstance(payload["query"], str):
        return payload["query"].strip() or None
    return None


def _query_from_items(items: list[Any]) -> str | None:
    """Return the sole user message's text, or None if not a clean single turn."""
    user_texts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        if item_type in _TOOL_ITEM_TYPES:
            return None  # tool-laden → skip
        role = item.get("role")
        # An assistant message that carries tool calls (or any non-user turn)
        # makes this a conversation, not a single fresh request.
        if role in _NON_USER_ROLES or item.get("tool_calls"):
            return None
        if role is not None and role not in _USER_ROLES:
            return None
        text = _content_text(item.get("content"))
        if text is None:
            return None  # non-text (image/etc.) content → skip
        user_texts.append(text)
    if len(user_texts) != 1:
        return None  # zero or multiple user turns → not single-turn
    return user_texts[0]


def _content_text(content: Any) -> str | None:
    """Extract plain text from a message ``content``, or None if not pure text."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in _TEXT_PART_TYPES:
                parts.append(str(part.get("text", "")))
            else:
                return None  # image / tool / unknown part → not pure text
        return "".join(parts)
    return None


def _assessment_source(assessment: Any) -> tuple[str, str]:
    """Return ``(source_type, source_id)`` for an assessment, defensively."""
    source = getattr(assessment, "source", None)
    source_type = str(getattr(source, "source_type", "") or "")
    source_id = str(getattr(source, "source_id", "") or "")
    return source_type, source_id


def _assessment_record(assessment: Any, *, with_source_type: bool = False) -> dict[str, Any]:
    """Project one assessment into the JSON-safe record stored in a row."""
    source_type, source_id = _assessment_source(assessment)
    rationale = getattr(assessment, "rationale", None)
    record: dict[str, Any] = {
        "name": str(getattr(assessment, "name", "") or ""),
        "value": getattr(assessment, "value", None),
        "rationale": rationale if rationale is None else str(rationale),
        "source_id": source_id,
    }
    if with_source_type:
        record["source_type"] = source_type
    return record


def trace_to_row(
    trace: Any,
    *,
    include_source_types: list[str] | None = None,
) -> dict[str, Any] | None:
    """Project one trace into a snapshot row, or None if it must be skipped.

    Returns None when the request is not a clean single turn (the caller
    counts it under ``skipped_multiturn``). Splits assessments into
    ``expectations`` (HUMAN Expectation), ``human_labels`` (HUMAN
    Feedback) and ``judge_rubrics`` (LLM_JUDGE / AI_JUDGE Feedback),
    keeping only source types in ``include_source_types``.
    """
    allowed = (
        frozenset(include_source_types)
        if include_source_types is not None
        else frozenset({"HUMAN", "LLM_JUDGE", "AI_JUDGE"})
    )
    query = parse_single_turn_query(_request_payload(trace))
    if query is None:
        return None

    expectations: list[dict[str, Any]] = []
    human_labels: list[dict[str, Any]] = []
    judge_rubrics: list[dict[str, Any]] = []
    for assessment in _assessments(trace):
        source_type, _ = _assessment_source(assessment)
        if source_type not in allowed:
            continue
        is_expectation = isinstance(assessment, Expectation)
        is_feedback = isinstance(assessment, Feedback)
        if source_type == "HUMAN" and is_expectation:
            expectations.append(_assessment_record(assessment))
        elif source_type == "HUMAN" and is_feedback:
            human_labels.append(_assessment_record(assessment))
        elif source_type in ("LLM_JUDGE", "AI_JUDGE") and is_feedback:
            judge_rubrics.append(_assessment_record(assessment, with_source_type=True))
        # Any other combination (e.g. a judge Expectation, a CODE feedback)
        # is not a re-scorable objective under this design; drop it.

    return {
        "example_id": _trace_id(trace),
        "query": query,
        "category": "trace",
        "expectations": expectations,
        "human_labels": human_labels,
        "judge_rubrics": judge_rubrics,
    }


def _row_has_labels(row: dict[str, Any]) -> bool:
    """True if a row carries at least one scoring target."""
    return bool(row["expectations"] or row["human_labels"] or row["judge_rubrics"])


def build_snapshot_rows(
    traces: list[Any],
    *,
    include_source_types: list[str] | None = None,
    dedup: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Project traces into snapshot rows and return ``(rows, counters)``.

    Skips multi-turn/tool-laden requests (counted under
    ``skipped_multiturn``) and rows that carry no kept assessment
    (``skipped_no_labels``). When ``dedup`` is set, rows with an identical
    ``query`` collapse to the first occurrence (``deduped`` counts the
    dropped duplicates). Counters also total ``expectations`` /
    ``human_labels`` / ``judge_rubrics`` across kept rows.
    """
    counters = {
        "kept": 0,
        "skipped_multiturn": 0,
        "skipped_no_labels": 0,
        "deduped": 0,
        "expectations": 0,
        "human_labels": 0,
        "judge_rubrics": 0,
    }
    rows: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for trace in traces:
        row = trace_to_row(trace, include_source_types=include_source_types)
        if row is None:
            counters["skipped_multiturn"] += 1
            continue
        if not _row_has_labels(row):
            counters["skipped_no_labels"] += 1
            continue
        if dedup:
            if row["query"] in seen_queries:
                counters["deduped"] += 1
                continue
            seen_queries.add(row["query"])
        rows.append(row)
        counters["kept"] += 1
        counters["expectations"] += len(row["expectations"])
        counters["human_labels"] += len(row["human_labels"])
        counters["judge_rubrics"] += len(row["judge_rubrics"])
    return rows, counters


def write_snapshot(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write ``rows`` to a frozen JSONL snapshot (one object per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return p


def read_snapshot(path: str | Path) -> list[dict[str, Any]]:
    """Read a frozen JSONL snapshot written by :func:`write_snapshot`."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"trace snapshot not found: {p}")
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if "example_id" not in row or "query" not in row:
            raise ValueError(
                f"trace snapshot line {line_no}: row missing example_id/query"
            )
        # Tolerate snapshots written before a field existed by defaulting the
        # three assessment buckets and the bucket category.
        row.setdefault("category", "trace")
        row.setdefault("expectations", [])
        row.setdefault("human_labels", [])
        row.setdefault("judge_rubrics", [])
        rows.append(row)
    return rows


def snapshot_content_hash(rows: list[dict[str, Any]]) -> str:
    """Stable SHA-256 over the snapshot rows' content.

    Feeds the eval report's ``scorer_fingerprint`` so that re-ingesting a
    DIFFERENT snapshot invalidates the cached baseline (the reproducibility
    guarantee). Independent of row order and key order.
    """
    canonical = json.dumps(
        sorted(rows, key=lambda r: r.get("example_id", "")),
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
