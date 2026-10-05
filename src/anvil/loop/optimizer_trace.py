"""Replay a recorded optimizer session into an MLflow trace + a markdown log.

The optimizer runs inside the async ``claude-agent-sdk`` session, where
MLflow's async tracing interacts badly with the SDK (see ``loop/round.py``).
So the local backend only RECORDS the session as plain events
(:func:`anvil.optimizer.session.message_events`); after it returns, this
module rebuilds it synchronously, with the recorded timestamps:

* :func:`log_optimizer_trace` — one trace per round in the optimizer
  experiment: an ``AGENT`` root (the round prompt in, the chosen action
  out), a ``TOOL`` span per tool call (input + result, error status when the
  tool failed), an ``LLM`` span per assistant message / thinking block, and
  the session's turns / duration / cost / usage on the root. Tagged with the
  round, branch, decision and action, and linked to the session's MLflow run.
* :func:`render_session_markdown` — the same session as a turn-by-turn
  markdown file for the run's artifacts.

Both are best-effort: a tracing failure returns ``None`` and never fails a
round.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

TRACE_NAME = "forge_optimizer_session"


def _span_types() -> Any:
    from mlflow.entities import SpanType  # noqa: PLC0415

    return SpanType


def log_optimizer_trace(
    *,
    events: list[dict[str, Any]],
    prompt: str,
    experiment_id: str,
    started_ns: int | None,
    ended_ns: int | None,
    tags: dict[str, str] | None = None,
    outputs: dict[str, Any] | None = None,
    source_run_id: str | None = None,
) -> str | None:
    """Write one optimizer session as an MLflow trace; return its trace id.

    ``events`` come from :func:`anvil.optimizer.session.message_events`.
    ``source_run_id`` links the trace to the session's MLflow run (it then
    shows under that run's Traces). Returns ``None`` (and logs a warning)
    when there is nothing to trace or MLflow fails.
    """
    if not events or started_ns is None:
        return None
    try:
        import mlflow  # noqa: PLC0415
        from mlflow.entities import SpanStatusCode  # noqa: PLC0415

        span_type = _span_types()
        end_ns = max(ended_ns or 0, *(int(e.get("t_ns") or 0) for e in events), started_ns)
        root = mlflow.start_span_no_context(
            name=TRACE_NAME,
            span_type=span_type.AGENT,
            inputs={"prompt": prompt},
            tags={"source": "optimizer", **(tags or {})},
            metadata={"mlflow.sourceRun": source_run_id} if source_run_id else None,
            experiment_id=experiment_id,
            start_time_ns=started_ns,
        )
        open_tools: dict[str, Any] = {}
        result_stats: dict[str, Any] = {}
        for i, ev in enumerate(events):
            t = int(ev.get("t_ns") or started_ns)
            kind = ev.get("kind")
            if kind == "tool_use":
                span = mlflow.start_span_no_context(
                    name=f"tool:{ev.get('name') or 'unknown'}",
                    span_type=span_type.TOOL,
                    parent_span=root,
                    inputs=ev.get("input"),
                    start_time_ns=t,
                )
                open_tools[str(ev.get("id") or f"_{i}")] = span
            elif kind == "tool_result":
                span = open_tools.pop(str(ev.get("tool_use_id")), None)
                if span is not None:
                    span.end(
                        outputs={"content": ev.get("content")},
                        status=SpanStatusCode.ERROR if ev.get("is_error") else SpanStatusCode.OK,
                        end_time_ns=t,
                    )
            elif kind in ("text", "thinking"):
                span = mlflow.start_span_no_context(
                    name="thinking"
                    if kind == "thinking"
                    else f"{ev.get('role', 'assistant')}_message",
                    span_type=span_type.LLM,
                    parent_span=root,
                    start_time_ns=t,
                )
                span.end(outputs={"text": ev.get("text")}, end_time_ns=t)
            elif kind == "result":
                result_stats = {
                    k: ev.get(k)
                    for k in ("subtype", "is_error", "num_turns", "duration_ms", "total_cost_usd")
                    if ev.get(k) is not None
                }
                if ev.get("usage") is not None:
                    result_stats["usage"] = json.dumps(ev["usage"], default=str)
        for span in open_tools.values():  # a tool call the session never answered
            span.end(outputs={"content": "(no result recorded)"}, end_time_ns=end_ns)
        root.end(
            outputs=outputs or {},
            attributes={f"optimizer.{k}": v for k, v in result_stats.items()},
            end_time_ns=end_ns,
        )
        # Make sure the trace is exported before the round moves on (the
        # process may exit right after the last round; async export would
        # drop it).
        with contextlib.suppress(Exception):
            mlflow.flush_trace_async_logging()
        return root.trace_id
    except Exception as exc:  # noqa: BLE001 — observability must never fail a round
        logger.warning("could not log the optimizer session trace: %s", exc)
        return None


def _block(text: Any) -> str:
    body = text if isinstance(text, str) else json.dumps(text, indent=2, default=str)
    return f"```\n{body}\n```"


def render_session_markdown(events: list[dict[str, Any]], *, title: str) -> str:
    """The recorded session as a turn-by-turn markdown log."""
    lines = [f"# {title}", ""]
    names: dict[str, str] = {}
    for ev in events:
        kind = ev.get("kind")
        if kind == "text":
            lines += [f"## {ev.get('role', 'assistant')}", "", str(ev.get("text") or ""), ""]
        elif kind == "thinking":
            lines += ["## thinking", "", str(ev.get("text") or ""), ""]
        elif kind == "tool_use":
            names[str(ev.get("id"))] = str(ev.get("name"))
            lines += [f"## tool call: `{ev.get('name')}`", "", _block(ev.get("input")), ""]
        elif kind == "tool_result":
            name = names.get(str(ev.get("tool_use_id")), "?")
            status = " (error)" if ev.get("is_error") else ""
            lines += [f"### result of `{name}`{status}", "", _block(ev.get("content")), ""]
        elif kind == "result":
            stats = ", ".join(
                f"{k}={ev.get(k)}"
                for k in ("subtype", "num_turns", "duration_ms", "total_cost_usd")
                if ev.get(k) is not None
            )
            lines += ["## session result", "", stats, ""]
    return "\n".join(lines).rstrip() + "\n"
