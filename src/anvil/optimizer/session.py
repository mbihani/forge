"""The optimizer's session plane — backend protocol + local ClaudeSDKClient impl.

The optimizer is the only async piece of ANVIL because the
``claude-agent-sdk`` package is async-only. Everything else (runtime,
eval, loop) is synchronous; the loop's :func:`run_round` calls
``asyncio.run`` to bridge.

Two responsibilities, separated by the :class:`OptimizerBackend` protocol:

1. **Configure the env** — point the bundled Claude Code subprocess at
   the workspace's AI Gateway anthropic route (``ANTHROPIC_BASE_URL``,
   ``ANTHROPIC_DEFAULT_OPUS_MODEL``, the custom coding-agent header, the
   experimental-betas opt-out). Local-only; the Omnigent backend runs the
   agent on a managed server and needs none of this.
2. **Run one bounded session** — drain the agent's output into a
   transcript, parse the final action JSON, and return an
   :class:`OptimizerResult`.

The prompt is composed elsewhere (loop.builder); this module only takes a
ready-to-send ``prompt`` string. That keeps the session function
loop-side-agnostic and trivially mockable in tests.

Backends:

* :class:`LocalBackend` — the existing ``ClaudeSDKClient`` path, preserved
  byte-for-byte. Selected by ``optimizer.backend: local`` (the default).
* :class:`OmnigentBackend` (in :mod:`anvil.optimizer.omnigent_backend`) —
  runs the optimizer agent on a managed Omnigent server via REST. Selected
  by ``optimizer.backend: omnigent``.

:func:`run_optimizer_session` remains as a thin backward-compatible wrapper
over :class:`LocalBackend` so existing callers (and the loop's
``monkeypatch``-based tests) keep working unchanged.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import mlflow
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, PermissionResultAllow

from anvil.optimizer.actions import OptimizerAction
from anvil.optimizer.parser import ParseResult, parse_action

# AI Gateway host for Claude Agent SDK. Set ``ANVIL_AI_GATEWAY_URL``
# to your workspace's gateway URL — typically
# ``https://<workspace-id>.ai-gateway.cloud.databricks.com/anthropic``.
# This is NOT the workspace's ``<host>/serving-endpoints/anthropic``
# route: that one rejects the Claude Code CLI's beta flags with
# HTTP 400. The AI Gateway path implements the same Anthropic Messages
# API but speaks the Databricks-native auth header set.
ANTHROPIC_BASE_URL = os.environ.get("ANVIL_AI_GATEWAY_URL", "")


def resolve_claude_cli_path() -> str | None:
    """The Claude Code CLI the local optimizer runs.

    ``ANVIL_CLAUDE_CLI_PATH`` when set, else ``claude`` on ``PATH``, else
    ``None`` (the CLI bundled with ``claude-agent-sdk``). The bundled copy is
    pinned to the SDK release and goes stale: the gateway rejects models an old
    CLI does not know ("Claude Code 2.1.116 does not support this model"),
    which ends every session on its first turn.
    """
    return os.environ.get("ANVIL_CLAUDE_CLI_PATH") or shutil.which("claude")


def session_error(events: list[dict[str, Any]]) -> str | None:
    """The error that ended a session, or ``None`` when it ran normally.

    A session whose final result is an error (an API error on the first turn,
    a rejected model) parses to a ``NoopAction`` exactly like a deliberate
    noop; this tells them apart so the round records ``INFRA_FAIL``. Running
    out of turns (``error_max_turns``) is a normal end — the parser decides
    from whatever transcript there is.
    """
    for ev in reversed(events):
        if ev.get("kind") != "result":
            continue
        if not ev.get("is_error") or str(ev.get("subtype") or "").startswith("error_max_turns"):
            return None
        detail = ev.get("text")
        if not detail:
            detail = next(
                (
                    e.get("text")
                    for e in reversed(events)
                    if e.get("kind") == "text" and e.get("text")
                ),
                None,
            )
        return f"optimizer session ended in error: {detail or ev.get('subtype') or 'unknown'}"
    return None


# ---------------------------------------------------------------------------
# Backend protocol + result
# ---------------------------------------------------------------------------


@dataclass
class OptimizerResult:
    """Output of :meth:`OptimizerBackend.run` — always populated, never raises.

    Carries everything the loop needs to apply a mutation and persist
    diagnostics: the parsed action, the raw transcript, parse metadata,
    and the files the agent changed (for the Omnigent backend which runs
    in a remote filesystem). Local-backend-only fields (``session_url``,
    ``turns_used``, ``duration_s``) are ``None``; the Omnigent backend
    populates them.
    """

    action: OptimizerAction
    transcript: str
    parse_result: ParseResult
    # relative_path -> new_content (None = deleted). Empty for the local
    # backend, which writes nothing — the loop's applier applies the action.
    modified_files: dict[str, str | None] = field(default_factory=dict)
    session_url: str | None = None
    mlflow_trace_url: str | None = None
    turns_used: int | None = None
    duration_s: float | None = None
    # Set to a human-readable error string ONLY when the optimizer BACKEND
    # itself failed (auth/401, redirect to SSO, connection error, malformed
    # response, ...). This is distinct from a legitimate optimizer-chosen
    # ``NoopAction``: a backend failure must surface as a failure, not be
    # silently disguised as a clean noop. ``None`` on every success path.
    optimizer_error: str | None = None
    # The full turn-by-turn session as plain dicts (see
    # :func:`message_events`), with wall-clock bounds in ns. Recorded by
    # the local backend so the loop can replay the session into an MLflow
    # trace AFTER it returns (no async tracing inside the SDK session).
    events: list[dict[str, Any]] = field(default_factory=list)
    started_ns: int | None = None
    ended_ns: int | None = None


class OptimizerBackend(Protocol):
    """A pluggable optimizer session runner.

    The loop calls :meth:`run` once per round with the composed round prompt
    and the scaffold files (the Omnigent backend uploads them to its remote
    environment; the local backend ignores them — it reads from ``cwd`` via
    the Claude Code subprocess's own ``Read`` tool).
    """

    async def run(
        self,
        *,
        prompt: str,
        scaffold_files: dict[str, str],  # relative_path -> content
        max_turns: int,
        model: str | None = None,
    ) -> OptimizerResult: ...


# ---------------------------------------------------------------------------
# Environment setup (local backend only)
# ---------------------------------------------------------------------------


def setup_anthropic_env(
    profile: str | None = None,
    optimizer_endpoint: str | None = None,
) -> None:
    """Point the Claude Agent SDK at the Databricks-hosted Anthropic gateway.

    Idempotent: existing values in ``os.environ`` are left alone so a
    developer running with a direct Anthropic key locally is not
    overridden.

    ``optimizer_endpoint`` is the FMAPI model name from
    ``harness/config.yaml > optimizer_endpoint``. When set, it becomes
    the Claude Code CLI's default model (``ANTHROPIC_MODEL`` and
    ``ANTHROPIC_DEFAULT_OPUS_MODEL``). When None, falls back to the
    built-in default (``databricks-claude-opus-4-7``).

    Gateway authentication is handled automatically by Claude Code through
    the Databricks CLI (using ``DATABRICKS_CONFIG_PROFILE`` or
    ``DATABRICKS_HOST``), so no secret or token is required. An operator-set
    ``ANTHROPIC_AUTH_TOKEN`` is preserved as an optional override.
    """
    if "ANTHROPIC_BASE_URL" not in os.environ:
        if not ANTHROPIC_BASE_URL:
            raise RuntimeError(
                "ANVIL_AI_GATEWAY_URL is unset and ANTHROPIC_BASE_URL is not "
                "in the environment. Set ANVIL_AI_GATEWAY_URL to your "
                "workspace's AI Gateway endpoint, e.g. "
                "https://<workspace-id>.ai-gateway.cloud.databricks.com/anthropic"
            )
        os.environ["ANTHROPIC_BASE_URL"] = ANTHROPIC_BASE_URL
    os.environ.setdefault("CLAUDE_CODE_USE_GATEWAY", "1")
    optimizer_model = optimizer_endpoint or "databricks-claude-opus-4-7"
    os.environ.setdefault("ANTHROPIC_MODEL", optimizer_model)
    os.environ.setdefault("ANTHROPIC_DEFAULT_OPUS_MODEL", optimizer_model)
    os.environ.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", "databricks-claude-sonnet-4-6")
    os.environ.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", "databricks-claude-haiku-4-5")
    os.environ.setdefault("ANTHROPIC_CUSTOM_HEADERS", "x-databricks-use-coding-agent-mode: true")
    os.environ.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")


# ---------------------------------------------------------------------------
# Local backend (ClaudeSDKClient) — preserves the pre-refactor behavior
# ---------------------------------------------------------------------------


@dataclass
class LocalBackend:
    """The existing ``ClaudeSDKClient`` optimizer session, as a backend.

    Behavior is identical to the pre-refactor :func:`run_optimizer_session`:
    configure the env, open one bounded ``ClaudeSDKClient`` session, drain
    ``receive_response`` into a transcript, parse the action JSON. The
    ``scaffold_files`` and ``model`` args are accepted to satisfy the
    :class:`OptimizerBackend` protocol but unused — the local backend reads
    the scaffold from ``cwd`` via the subprocess's own ``Read`` tool, and
    the model is configured via :func:`setup_anthropic_env` at construction.
    """

    cwd: str
    profile: str | None = None
    setup_env: bool = True
    optimizer_endpoint: str | None = None
    experiment_name: str | None = None
    round_id: int | None = None

    async def run(
        self,
        *,
        prompt: str,
        scaffold_files: dict[str, str],  # noqa: ARG002 — unused locally
        max_turns: int,
        model: str | None = None,  # noqa: ARG002 — configured via setup_anthropic_env
    ) -> OptimizerResult:
        if self.setup_env:
            setup_anthropic_env(profile=self.profile, optimizer_endpoint=self.optimizer_endpoint)

        if self.experiment_name:
            # ``"DEFAULT"`` is the sentinel default threaded down from
            # ``run_round`` (loop/round.py) via ``run_optimizer_session``; it
            # is not a real ~/.databrickscfg profile. Binding
            # ``databricks://DEFAULT`` would break MLflow's fallback to
            # ambient DATABRICKS_HOST/DATABRICKS_TOKEN auth, so skip it and
            # let native auth apply — matching ``evaluate_branch``.
            if self.profile and self.profile != "DEFAULT":
                mlflow.set_tracking_uri(f"databricks://{self.profile}")
            mlflow.set_experiment(self.experiment_name)
            # Turn on Anthropic autolog so each LLM call inside the Claude
            # Code subprocess becomes a CHAT_MODEL child span. Idempotent.
            # autolog can fail at import time on unsupported SDK versions;
            # the trace still wraps the session, just without per-call
            # children.
            with contextlib.suppress(Exception):
                mlflow.anthropic.autolog()

        options = ClaudeAgentOptions(
            cwd=self.cwd,
            disallowed_tools=["AskUserQuestion", "ExitPlanMode"],
            setting_sources=[],
            max_turns=max_turns,
            can_use_tool=_allow_all_tool_calls,
            cli_path=resolve_claude_cli_path(),
        )

        events: list[dict[str, Any]] = []

        async def _drain_session() -> str:
            parts: list[str] = []
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    events.extend(message_events(message))
                    text = extract_message_text(message)
                    if text:
                        parts.append(text)
            return "\n\n".join(parts)

        started_ns = time.time_ns()
        start = time.monotonic()
        if self.experiment_name:
            with mlflow.start_span(name="anvil_optimizer_round") as span:
                tags: dict[str, str] = {"source": "optimizer"}
                if self.round_id is not None:
                    tags["round"] = str(self.round_id)
                tags["max_turns"] = str(max_turns)
                with contextlib.suppress(Exception):
                    mlflow.update_current_trace(tags=tags)
                span.set_inputs({"prompt_chars": len(prompt), "round_id": self.round_id})
                transcript = await _drain_session()
                span.set_outputs({"transcript_chars": len(transcript)})
        else:
            transcript = await _drain_session()
        duration_s = time.monotonic() - start

        parse_result = parse_action(transcript)
        return OptimizerResult(
            action=parse_result.action,
            transcript=transcript,
            parse_result=parse_result,
            # Local backend writes nothing; the loop's applier applies the
            # action to disk. Remote-only fields stay None.
            modified_files={},
            session_url=None,
            mlflow_trace_url=None,
            turns_used=None,
            duration_s=duration_s,
            optimizer_error=session_error(events),
            events=events,
            started_ns=started_ns,
            ended_ns=time.time_ns(),
        )


async def run_optimizer_session(
    *,
    prompt: str,
    cwd: str,
    max_turns: int = 30,
    profile: str | None = None,
    setup_env: bool = True,
    optimizer_endpoint: str | None = None,
    experiment_name: str | None = None,
    round_id: int | None = None,
    session_record: dict[str, Any] | None = None,
) -> tuple[OptimizerAction, str, ParseResult]:
    """Open one ``ClaudeSDKClient`` session, drain it, parse the action.

    Thin backward-compatible wrapper over :class:`LocalBackend` that
    preserves the original 3-tuple return shape
    ``(action, transcript, parse_result)``. Existing callers (and the
    loop's monkeypatch-based tests) keep working unchanged.

    Args:
        prompt: Fully-composed user prompt (built by ``loop.builder``).
        cwd: Working directory for the Claude Code subprocess (typically
            the repo root).
        max_turns: Hard cap on optimizer CLI turns. The session aborts
            and returns whatever transcript it has if exceeded; the
            parser then falls back to ``NoopAction``.
        profile: Databricks CLI profile used by Claude Code for gateway auth.
        setup_env: If True (default), call :func:`setup_anthropic_env`
            before opening the session. Disable in tests.
        optimizer_endpoint: FMAPI model name from
            ``harness/config.yaml > optimizer_endpoint``. Forwarded to
            :func:`setup_anthropic_env` so the Claude Code CLI uses the
            configured model. When None, the built-in default is used.
        experiment_name: When set, the session opens an MLflow trace
            under this experiment and turns on
            ``mlflow.anthropic.autolog`` so each Anthropic API call
            inside the Claude Code subprocess becomes a child span.
            Disable in tests by passing ``None`` (default).
        round_id: Tag value for the ``round`` trace tag. Optional.
        session_record: When a dict is passed, it is filled with the full
            turn-by-turn session (``events``, ``started_ns``, ``ended_ns``)
            so the caller can replay it into an MLflow trace. The 3-tuple
            return shape is unchanged.

    Returns:
        A 3-tuple ``(action, transcript, parse_result)`` where ``action``
        is the parsed ``OptimizerAction`` (always populated; ``NoopAction``
        on parse failure), ``transcript`` is the raw text Claude emitted,
        and ``parse_result`` carries diagnostic metadata about the parse.
    """
    backend = LocalBackend(
        cwd=cwd,
        profile=profile,
        setup_env=setup_env,
        optimizer_endpoint=optimizer_endpoint,
        experiment_name=experiment_name,
        round_id=round_id,
    )
    result = await backend.run(
        prompt=prompt,
        scaffold_files={},
        max_turns=max_turns,
        model=optimizer_endpoint,
    )
    if session_record is not None:
        session_record.update(
            events=result.events,
            started_ns=result.started_ns,
            ended_ns=result.ended_ns,
            optimizer_error=result.optimizer_error,
        )
    return result.action, result.transcript, result.parse_result


def extract_message_text(message) -> str:
    """Best-effort plain-text extractor for a ``claude-agent-sdk`` message.

    The SDK ships several message types — ``SystemMessage``,
    ``AssistantMessage`` (with ``content`` as a list of ``ThinkingBlock``,
    ``ToolUseBlock``, ``TextBlock`` ...), ``UserMessage`` (tool results),
    ``ResultMessage`` (final answer in ``.result``).

    The parser needs the raw text where the optimizer wrote its
    ```json-action ` fenced block. ``str(message)`` is wrong: it
    emits the Python ``repr`` with ``\\n`` escaped, breaking the
    regex's ``\\n`` matches. Instead, walk the typed shape via
    ``getattr`` (no hard import of SDK types — keeps the wrapper
    forward-compatible).

    Returns concatenated text from:

      * ``message.result`` if it's a string (``ResultMessage``).
      * ``block.text`` for every block in ``message.content`` that
        has a ``.text`` string attribute (``TextBlock`` inside an
        ``AssistantMessage``).

    Other block types (``ThinkingBlock``, ``ToolUseBlock``) are
    dropped — the optimizer's user-visible reasoning lives in
    ``TextBlock`` and the final ``ResultMessage`` only.
    """
    parts: list[str] = []

    result = getattr(message, "result", None)
    if isinstance(result, str):
        parts.append(result)

    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)

    return "\n".join(parts)


# Cap on the text kept per recorded event: tool results can be whole files
# and a written agent module can be large; traces stay readable and small.
_EVENT_TEXT_CAP = 20_000


def _cap(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _EVENT_TEXT_CAP:
        return value[:_EVENT_TEXT_CAP] + f"\n…(truncated, {len(value)} chars)"
    if isinstance(value, dict):
        return {k: _cap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cap(v) for v in value]
    return value


def _tool_result_text(content: Any) -> Any:
    """A tool result's content as text where possible (list of text blocks)."""
    if isinstance(content, list):
        texts = [
            c.get("text") if isinstance(c, dict) else getattr(c, "text", None) for c in content
        ]
        if all(isinstance(t, str) for t in texts):
            return "\n".join(texts)
    return content


def message_events(message: Any) -> list[dict[str, Any]]:
    """Plain-dict events for one ``claude-agent-sdk`` message, timestamped.

    Duck-typed like :func:`extract_message_text` (no hard SDK imports).
    Kinds: ``text`` / ``thinking`` (assistant), ``tool_use`` (id, name,
    input), ``tool_result`` (tool_use_id, content, is_error) and ``result``
    (the session's final stats: turns, duration, cost, usage). Long text is
    capped at ``_EVENT_TEXT_CAP`` chars.
    """
    t_ns = time.time_ns()
    kind = type(message).__name__
    out: list[dict[str, Any]] = []
    if kind == "ResultMessage" or (
        hasattr(message, "num_turns") and hasattr(message, "total_cost_usd")
    ):
        out.append(
            {
                "kind": "result",
                "t_ns": t_ns,
                "subtype": getattr(message, "subtype", None),
                "is_error": getattr(message, "is_error", None),
                "num_turns": getattr(message, "num_turns", None),
                "duration_ms": getattr(message, "duration_ms", None),
                "total_cost_usd": getattr(message, "total_cost_usd", None),
                "usage": _cap(getattr(message, "usage", None)),
                "text": _cap(getattr(message, "result", None)),
            }
        )
        return out
    content = getattr(message, "content", None)
    if isinstance(content, str):
        role = "user" if kind == "UserMessage" else "assistant"
        return [{"kind": "text", "role": role, "t_ns": t_ns, "text": _cap(content)}]
    if not isinstance(content, list):
        return out
    for block in content:
        if hasattr(block, "tool_use_id"):
            out.append(
                {
                    "kind": "tool_result",
                    "t_ns": t_ns,
                    "tool_use_id": getattr(block, "tool_use_id", None),
                    "is_error": bool(getattr(block, "is_error", False)),
                    "content": _cap(_tool_result_text(getattr(block, "content", None))),
                }
            )
        elif isinstance(getattr(block, "thinking", None), str):
            out.append({"kind": "thinking", "t_ns": t_ns, "text": _cap(block.thinking)})
        elif hasattr(block, "name") and hasattr(block, "input"):
            out.append(
                {
                    "kind": "tool_use",
                    "t_ns": t_ns,
                    "id": getattr(block, "id", None),
                    "name": str(getattr(block, "name", "")),
                    "input": _cap(getattr(block, "input", None)),
                }
            )
        elif isinstance(getattr(block, "text", None), str):
            out.append(
                {"kind": "text", "role": "assistant", "t_ns": t_ns, "text": _cap(block.text)}
            )
    return out


async def _allow_all_tool_calls(_tool_name: str, _tool_input: dict, _ctx) -> PermissionResultAllow:
    """Permission callback: blanket-allow every tool call.

    The CLI's filesystem sandbox blocks Write/Edit/Bash redirections
    under ``cwd`` even with ``permission_mode="bypassPermissions"`` or
    the ``--dangerously-skip-permissions`` extra arg. The Python
    callback IS honored, however. It must return the SDK's
    ``PermissionResultAllow`` — a plain ``{"behavior": "allow"}`` dict is
    rejected ("Tool permission callback must return PermissionResult"),
    which silently failed every Write and most Bash calls in a session.
    """
    return PermissionResultAllow(updated_input=_tool_input)
