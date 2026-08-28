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
import time
from dataclasses import dataclass, field
from typing import Protocol

import mlflow
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

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
            if self.profile:
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
        )

        async def _drain_session() -> str:
            parts: list[str] = []
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    text = extract_message_text(message)
                    if text:
                        parts.append(text)
            return "\n\n".join(parts)

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


async def _allow_all_tool_calls(_tool_name: str, _tool_input: dict, _ctx) -> dict:
    """Permission callback: blanket-allow every tool call.

    The CLI's filesystem sandbox blocks Write/Edit/Bash redirections
    under ``cwd`` even with ``permission_mode="bypassPermissions"`` or
    the ``--dangerously-skip-permissions`` extra arg. The Python
    callback IS honored, however; this returns ``{"behavior": "allow",
    "updatedInput": ...}`` for every call. Documented empirically.
    """
    return {"behavior": "allow", "updatedInput": _tool_input}
