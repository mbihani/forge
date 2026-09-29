"""Omnigent-backed optimizer session — runs the agent on a managed server.

Implements :class:`OptimizerBackend` by driving a managed Omnigent server
over REST (see :mod:`anvil.optimizer.omnigent_client`). One round maps to
one ephemeral Omnigent session:

1. **Build the bundle** — package ``agents/forge_optimizer.yaml`` as a
   tar.gz whose root ``config.yaml`` is the agent spec. The ``model`` and
   ``max_turns`` args are substituted into the spec so each round can pin
   them without editing the file on disk.
2. **Register, open, and EXPLICITLY bind a runner (managed flow)** — ``POST
   /v1/sessions`` (multipart: metadata + bundle) only *registers* the
   agent and returns its ``agent_id``; that registration session has no
   runner bound. So we open a SECOND session via
   ``create_session_from_agent(agent_id, host_type="managed", ...)``. Opening
   the managed session PROVISIONS a runner in the pool but does NOT bind the
   conversation to it — ``conversations.runner_id`` stays null, and the first
   resource read (``list_environments``) on an unbound conversation is
   rejected HTTP 409 "not bound to a runner; resume the session to bind a
   registered runner". So we EXPLICITLY bind before any resource read:
   discover the online runner from ``GET /v1/runners`` (preferring the
   harness the agent needs) and ``PATCH /v1/sessions/{id}`` with its
   ``runner_id`` (the resume-bind), then poll ``get_session`` until the
   snapshot reports the runner online AND bound to that id. All subsequent
   calls use the managed session id (``managed["id"]``); the throwaway
   registration session is deleted. The round's model pin rides through
   ``model_override``; ``max_turns`` rides through the baked bundle.
3. **Upload the scaffold** — the round's ``scaffold_files`` dict is written
   to the session's ``default`` environment filesystem so the agent can
   ``Read`` the same tree the local backend reads from ``cwd``.
4. **Send the prompt** — the round prompt goes in as the first user message
   via the hidden ``POST /v1/sessions/{id}/events`` ingestion route, which
   starts the agent's turn.
5. **Drain the stream** — the SSE event stream is drained into a
   transcript (assistant text deltas + completed message items). The
   drain runs across turns: ``response.completed`` marks a per-turn
   boundary (counted, not terminal); the drain stops on ``[DONE]`` /
   EOF, an inactivity timeout (no event for 120s), a max-duration
   deadline (30 min), or the turn cap. An ``idle`` session status is
   logged but NOT terminal — the inactivity timeout handles the
   "idle and never came back" case.
6. **Download modified files** — the environment's ``changes`` endpoint
   lists files the agent wrote/deleted; each is read back into
   ``OptimizerResult.modified_files`` (``None`` for deletions).
7. **Parse the action** — the transcript is parsed with the same
   :func:`parse_action` the local backend uses, so both backends share one
   action contract.

The session is left in place after the round (not auto-deleted) so the
returned ``session_url`` is inspectable. The loop may delete it via
:meth:`OmnigentClient.delete_session` when it no longer needs the transcript.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import tarfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import httpx
import yaml

from anvil.optimizer.omnigent_client import (
    OmnigentClient,
    OmnigentError,
    SessionCreateMetadata,
    build_session_url,
    resolve_omnigent_workspace_id,
)
from anvil.optimizer.parser import parse_action
from anvil.optimizer.session import OptimizerResult

# Paths under these prefixes are considered "the scaffold" when filtering
# the environment's change log — the optimizer's writable scope mirrors the
# local backend's (scaffold/, agents/, prompts/, eval/, harness/).
_SCAFFOLD_PREFIXES = ("scaffold/", "agents/", "prompts/", "eval/", "harness/")

# SSE event types that carry assistant text or mark the turn boundary.
_DELTA_TYPE = "response.output_text.delta"
_ITEM_DONE_TYPE = "response.output_item.done"
_COMPLETED_TYPE = "response.completed"
_STATUS_TYPE = "session.status"

# Bounded-drain safety timeouts. The SSE stream can stay open with only
# heartbeats and no ``response.completed`` / ``idle`` / EOF; without a bound
# the drain loops forever. ``_STREAM_INACTIVITY_TIMEOUT`` fires when no event
# arrives for this many seconds (the agent went idle and never came back, or
# the connection is silently open). ``_STREAM_MAX_DURATION`` is a hard
# ceiling on total drain time.
_STREAM_INACTIVITY_TIMEOUT = 120  # seconds
_STREAM_MAX_DURATION = 1800  # 30 minutes


@dataclass
class OmnigentBackend:
    """Optimizer backend that runs the agent on a managed Omnigent server.

    Args:
        client: An :class:`OmnigentClient` pointed at the server. Injected
            so tests pass a fake; production builds it from config.
        agent_bundle_path: Path to the agent spec YAML
            (``agents/forge_optimizer.yaml``) packaged as the bundle's
            ``config.yaml``.
        server_url: Server origin, used to build ``session_url``.
        default_model: Model used when the ``run`` arg is None.
        default_max_turns: Max turns baked into the bundle when the
            ``run`` arg is None.
    """

    client: OmnigentClient
    agent_bundle_path: Path
    server_url: str
    default_model: str = "databricks-claude-opus-4-7"
    default_max_turns: int = 70
    # Extra metadata for create_session (e.g. a host_id for a managed host).
    create_metadata: SessionCreateMetadata = field(default_factory=SessionCreateMetadata)

    async def run(
        self,
        *,
        prompt: str,
        scaffold_files: dict[str, str],
        max_turns: int,
        model: str | None = None,
    ) -> OptimizerResult:
        start = time.monotonic()
        effective_max_turns = max_turns or self.default_max_turns
        bundle = _build_agent_bundle(
            self.agent_bundle_path,
            model=model or self.default_model,
            max_turns=effective_max_turns,
        )
        session_id: str | None = None
        session_url: str | None = None
        optimizer_error: str | None = None
        registration_session_id: str | None = None

        try:
            # Step 1: register the agent via the multipart bundle upload. This
            # session has NO runner bound — it only yields the ``agent_id``.
            # Capture the registration session id FIRST so a response missing
            # ``agent_id`` (KeyError) still lets the ``finally`` tombstone the
            # throwaway registration session (no leak).
            created = await self.client.create_session(bundle, metadata=self.create_metadata)
            registration_session_id = created["session_id"]
            agent_id = created["agent_id"]

            # Step 2: open a MANAGED session bound to the registered agent —
            # ``host_type="managed"`` triggers the managed host to
            # auto-provision a runner. Without this the registration session's
            # ``list_environments`` fails HTTP 409 "not bound to a runner".
            # The round's model pin rides through ``model_override`` (the
            # managed-session level override); ``max_turns`` has no
            # create_session_from_agent parameter, so it rides through the
            # bundle baked in step 1 (``_build_agent_bundle`` set
            # ``executor.config.max_turns`` on the registered agent spec).
            managed = await self.client.create_session_from_agent(
                agent_id,
                host_type="managed",
                model_override=model or self.default_model,
                title=self.create_metadata.title,
            )
            # The managed session id is ``managed["id"]`` — NOT
            # ``created["session_id"]``. ALL subsequent calls target it.
            session_id = managed["id"]
            session_url = build_session_url(
                self.server_url, session_id, resolve_omnigent_workspace_id()
            )

            # Step 3: EXPLICITLY BIND a runner (the resume-bind). Opening the
            # managed session only PROVISIONS a runner in the pool — it does
            # NOT bind the conversation to it (``conversations.runner_id``
            # stays null). The FIRST resource read on an unbound conversation
            # (``list_environments`` below) is rejected HTTP 409 "not bound to
            # a runner; resume the session to bind a registered runner". So we
            # bind BEFORE any resource read: use the managed snapshot's
            # ``runner_id`` when it already carries one, else discover the
            # currently-registered online runner from ``/v1/runners``. A
            # missing online runner or a failed bind raises → caught below →
            # ``optimizer_error`` → ``Decision.INFRA_FAIL`` (never a silent
            # noop). The bind is wrapped in the bounded 503-retry (the runner
            # may still be provisioning).
            runner_id = managed.get("runner_id")
            if not runner_id:
                runner_id = await _retry_on_503(self._resolve_runner_id)
            await _retry_on_503(lambda: self.client.bind_runner(session_id, runner_id))

            # Confirm the bind took effect: poll ``get_session`` until the
            # snapshot reports the runner ONLINE and its ``runner_id`` equals
            # the id we bound (the server applies the bind last-write-wins).
            # A bind that never reflects within the bounded poll is a genuine
            # infra failure — ``raise_on_timeout=True`` surfaces it as
            # ``optimizer_error`` → ``Decision.INFRA_FAIL``, never a silent
            # success/noop.
            await _wait_for_runner(
                self.client,
                session_id,
                raise_on_timeout=True,
                expected_runner_id=runner_id,
            )

            # Steps 4+: EVERY post-bind call targets the MANAGED session id and
            # is retried on transient 503 — the runner may still be
            # provisioning for a short window even after it reports online, so
            # the entire post-bind path (env resolve, upload, send, stream
            # drain, items fallback, changes/file reads) is 503-resilient.
            env_id = await _retry_on_503(lambda: self._resolve_environment(session_id))
            await _retry_on_503(
                lambda: self._upload_scaffold(session_id, env_id, scaffold_files)
            )
            await _send_with_retry(self.client, session_id, prompt)
            stream_text, turns_used, stream_drop = await _retry_on_503(
                lambda: self._drain_stream(session_id, max_turns=effective_max_turns)
            )

            if stream_drop is not None:
                # The SSE read was cut MID-STREAM (the front-door proxy kills a
                # long-lived stream after ~5-6 min). The partial stream text is
                # NOT trustworthy even when it happens to contain a parseable
                # fence — that action may be stale or truncated by the drop. The
                # durably-persisted TERMINAL transcript is authoritative, so on
                # ANY incomplete stream we IGNORE the partial parse and instead
                # (a) wait for the turn to reach terminal, then (b) reconstruct
                # from persisted items and (c) parse THAT. The surfaced note
                # reflects the ACTUAL caught exception (RemoteProtocolError /
                # ReadError / StreamError), not a hard-coded type.
                drop_desc = f"{type(stream_drop).__name__}: {stream_drop}"
                if await self._wait_for_turn_terminal(session_id):
                    transcript = await _retry_on_503(
                        lambda: self._transcript_from_items(session_id)
                    )
                    # Terminal reached but no parseable action recovered from
                    # the persisted items → a GENUINE failure, surfaced as
                    # INFRA_FAIL rather than an indistinguishable "optimized,
                    # noop".
                    if parse_action(transcript).parse_status not in (
                        "ok",
                        "ok_last_of_many",
                    ):
                        optimizer_error = (
                            f"optimizer SSE stream disconnected mid-turn ({drop_desc}); "
                            "turn reached terminal but no parseable action could be "
                            "recovered from the persisted conversation items"
                        )
                else:
                    # The bounded terminal-wait NEVER observed a terminal turn
                    # within budget. Reading items now could capture a
                    # STILL-RUNNING turn's partial state, so we do NOT read
                    # them; keep the partial stream text for the record and
                    # surface INFRA_FAIL.
                    transcript = stream_text
                    optimizer_error = (
                        f"optimizer SSE stream disconnected mid-turn ({drop_desc}); "
                        "turn did not reach a terminal state within the bounded "
                        "recovery budget"
                    )
            else:
                # Clean stream end. The stream is the primary transcript source;
                # fall back to the persisted conversation items only if the
                # stream text did NOT yield a parseable action. Checking
                # ``parse_action`` (not a raw triple-backtick scan) avoids
                # letting an unrelated or malformed fence in the stream suppress
                # a valid items fallback — a ``bad_json`` / ``schema_mismatch``
                # fence is just as much a non-success as no fence at all.
                transcript = stream_text
                if parse_action(transcript).parse_status not in ("ok", "ok_last_of_many"):
                    items_text = await _retry_on_503(
                        lambda: self._transcript_from_items(session_id)
                    )
                    if items_text:
                        transcript = items_text

            modified_files = await self._collect_modified_files(session_id, env_id)
        except Exception as exc:
            # Degrade to a noop-producing transcript on ANY backend failure so
            # the loop never crashes — same posture as the local backend's
            # parse-failure path. ``OmnigentError`` carries a response body;
            # transport failures (httpx.ConnectError, httpx.TimeoutException)
            # and malformed-JSON / unexpected-shape errors do not, so the
            # ``body`` is pulled defensively via getattr.
            body = getattr(exc, "body", "")
            note = f"[omnigent backend error: {exc}]"
            if body:
                note += f"\n{body}"
            transcript = note
            modified_files = {}
            turns_used = None
            # Preserve the failure as a DISTINGUISHABLE marker so the round
            # surfaces it as a backend failure rather than an
            # indistinguishable "optimized, noop". ``parse_action`` below
            # will still yield a NoopAction from this transcript, but
            # ``optimizer_error`` lets the loop tell the two apart. Keep the
            # marker to a single line so it reads cleanly in round summaries.
            optimizer_error = f"{type(exc).__name__}: {exc}"
            if body:
                optimizer_error += f" | {body}"
        finally:
            # Clean up ONLY the throwaway registration session; the managed
            # conversation session is KEPT alive so ``session_url`` stays
            # inspectable. Best-effort — a failed delete must never mask the
            # round result or overwrite ``optimizer_error``.
            if registration_session_id is not None:
                with contextlib.suppress(Exception):
                    await self.client.delete_session(registration_session_id)

        parse_result = parse_action(transcript)
        return OptimizerResult(
            action=parse_result.action,
            transcript=transcript,
            parse_result=parse_result,
            modified_files=modified_files,
            session_url=session_url,
            mlflow_trace_url=None,
            turns_used=turns_used,
            duration_s=time.monotonic() - start,
            optimizer_error=optimizer_error,
        )

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    async def _resolve_runner_id(self) -> str:
        """Discover a currently-registered ONLINE runner to bind, or RAISE.

        Reads the runner pool (``GET /v1/runners``) and picks an online
        runner, PREFERRING one whose ``harnesses`` includes the harness the
        optimizer agent needs (``executor.config.harness`` in the bundle spec
        — ``claude-sdk`` / ``claude-native``). When the needed harness cannot
        be determined, or no online runner advertises it, the first online
        runner is used.

        Raises :class:`OmnigentError` when NO online runner is available (or
        none carries a ``runner_id``) so the backend surfaces it as an
        infrastructure failure (``optimizer_error`` → ``Decision.INFRA_FAIL``)
        rather than a silent noop.
        """
        runners = await self.client.list_runners()
        online = [r for r in runners if r.get("online")]
        if not online:
            raise OmnigentError(
                "no online Omnigent runner available to bind the managed session "
                f"(pool size {len(runners)}); cannot resume the conversation",
                status_code=None,
            )
        needed = _needed_harness(self.agent_bundle_path)
        if needed:
            for runner in online:
                if needed in (runner.get("harnesses") or []) and runner.get("runner_id"):
                    return runner["runner_id"]
        for runner in online:
            if runner.get("runner_id"):
                return runner["runner_id"]
        raise OmnigentError(
            "online Omnigent runner(s) present but none carry a runner_id; "
            "cannot bind the managed session",
            status_code=None,
        )

    async def _resolve_environment(self, session_id: str) -> str:
        envs = await self.client.list_environments(session_id)
        if not envs:
            raise OmnigentError(f"session {session_id} has no environments")
        # Prefer the primary/default environment; fall back to the first.
        for env in envs:
            meta = env.get("metadata", {}) or {}
            if meta.get("role") == "primary" or env.get("id") == "default":
                return env["id"]
        return envs[0]["id"]

    async def _upload_scaffold(
        self,
        session_id: str,
        env_id: str,
        scaffold_files: dict[str, str],
    ) -> None:
        for relative_path, content in scaffold_files.items():
            if content is None:
                continue  # deletions are not uploaded
            await self.client.upload_file(session_id, env_id, relative_path, content)

    async def _drain_stream(
        self,
        session_id: str,
        *,
        max_turns: int = 70,
        inactivity_timeout: float = _STREAM_INACTIVITY_TIMEOUT,
        max_duration: float = _STREAM_MAX_DURATION,
    ) -> tuple[str, int | None, BaseException | None]:
        """Drain the SSE stream; return (text, turns_used, stream_drop).

        Collects ``response.output_text.delta`` fragments and the text of
        completed assistant messages (``response.output_item.done``).
        ``response.completed`` marks a per-TURN boundary (counted, not
        terminal) — the agent runs autonomously across turns on a single
        stream, so the drain must keep consuming past each turn.

        The drain is bounded by two safety timeouts:

        * ``inactivity_timeout`` — if no stream event arrives for this many
          seconds the drain breaks (the agent went idle and never came back,
          or the connection is silently open with only heartbeats).
        * ``max_duration`` — a hard ceiling on total drain time.

        An ``idle`` session status is NOT terminal — an inter-turn idle may
        be followed by another autonomous turn, and the inactivity timeout
        is the reliable terminal signal for "idle and never came back." The
        drain stops on: ``[DONE]`` / EOF, the inactivity timeout, the
        max-duration deadline, or the turn cap.

        ``stream_drop`` is the caught exception when the SSE read was cut
        MID-STREAM by the transport — the Databricks front-door proxy kills a
        long-lived stream after ~5-6 min, surfacing ``httpx.RemoteProtocolError``
        (and, pragmatically, ``httpx.ReadError`` / ``httpx.StreamError``) from
        ``stream.__anext__()``. The turn has very likely COMPLETED server-side
        by then, so instead of letting the disconnect propagate to ``run()``
        (→ ``INFRA_FAIL``) we break, return the partial transcript, and hand
        the caller the exception so it can (a) recover the full transcript from
        the durably-persisted conversation items and (b) surface the ACTUAL
        exception type/message if recovery fails. A clean end (``[DONE]`` /
        EOF / a bounded timeout / the turn cap) returns ``None``.
        """
        parts: list[str] = []
        turns = 0
        stream_drop: BaseException | None = None
        start = time.monotonic()
        last_event = start
        stream = self.client.stream_session(session_id)
        try:
            while True:
                now = time.monotonic()
                if now - start >= max_duration:
                    break
                # Bound the read by BOTH deadlines — whichever comes first.
                # Using only the inactivity remaining lets a late read block
                # past the max_duration ceiling; ``min()`` ensures the read
                # can never run past either deadline.
                inactivity_remaining = inactivity_timeout - (now - last_event)
                max_duration_remaining = max_duration - (now - start)
                remaining = min(inactivity_remaining, max_duration_remaining)
                if remaining <= 0:
                    break
                try:
                    async with asyncio.timeout(remaining):
                        event_type, data = await stream.__anext__()
                except (TimeoutError, StopAsyncIteration):
                    break
                except (httpx.RemoteProtocolError, httpx.ReadError, httpx.StreamError) as exc:
                    # Mid-stream disconnect: the front-door proxy cut our
                    # long-lived SSE read. Capture the exception (its type +
                    # message flow into the caller's recovery note) and break
                    # with the partial transcript so run() can recover the
                    # completed transcript from persisted items (rather than
                    # raising → INFRA_FAIL).
                    stream_drop = exc
                    break
                last_event = time.monotonic()
                if event_type == _DELTA_TYPE:
                    delta = data.get("delta")
                    if isinstance(delta, str):
                        parts.append(delta)
                elif event_type == _ITEM_DONE_TYPE:
                    parts.append(_text_from_item(data.get("item")))
                elif event_type == _STATUS_TYPE:
                    # idle is NOT terminal — the inactivity timeout handles
                    # the "idle and never came back" case. An inter-turn
                    # idle may be followed by another autonomous turn.
                    pass
                elif event_type == _COMPLETED_TYPE:
                    # Per-turn boundary, not whole-conversation completion.
                    turns += 1
                    if turns >= max_turns:
                        break
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
        transcript = "".join(parts).strip()
        return transcript, (turns or None), stream_drop

    async def _wait_for_turn_terminal(
        self,
        session_id: str,
        *,
        max_attempts: int = 30,
        delay: float = 2.0,
        max_duration: float = _STREAM_MAX_DURATION,
    ) -> bool:
        """Poll ``get_session`` until the agent's turn is TERMINAL, or give up.

        Called after a mid-stream SSE disconnect: the turn very likely
        completed server-side after our read dropped, and we must let it
        finish before reconstructing the transcript from persisted items (so
        the assistant's final message — carrying the fenced action — is fully
        durable). "Terminal" == the snapshot reports ``status == "idle"`` AND
        the runner is still online (an idle-but-offline snapshot is not a
        clean completion).

        Bounded so it can NEVER hang, reusing :func:`_wait_for_runner`'s
        bounded-poll style: at most ``max_attempts`` polls spaced by
        ``delay``, and additionally capped by the ``_STREAM_MAX_DURATION``
        budget. Returns True when the turn went terminal, False on exhaustion
        — a False is best-effort: the caller still attempts the items
        recovery, and a genuinely empty recovery surfaces INFRA_FAIL anyway.
        A transient :class:`OmnigentError` on a single poll is suppressed so a
        blip never aborts the wait.
        """
        start = time.monotonic()
        for attempt in range(1, max_attempts + 1):
            if time.monotonic() - start >= max_duration:
                break
            with contextlib.suppress(OmnigentError):
                snapshot = await self.client.get_session(session_id)
                if snapshot.get("status") == "idle" and snapshot.get("runner_online"):
                    return True
            if attempt < max_attempts:
                await anyio.sleep(delay)
        return False

    async def _transcript_from_items(self, session_id: str) -> str:
        """Build a transcript from the persisted conversation items.

        Fallback when the stream produced no fenced block. Walks the
        assistant ``message`` items in order and concatenates their
        ``output_text`` content blocks.
        """
        parts: list[str] = []
        after: str | None = None
        while True:
            page = await self.client.list_items(session_id, limit=100, after=after)
            for item in page.get("data", []):
                if item.get("type") == "message" and item.get("role") == "assistant":
                    parts.append(_text_from_content(item.get("content")))
            if not page.get("has_more"):
                break
            after = page.get("last_id")
            if after is None:
                break
        return "\n".join(p for p in parts if p).strip()

    async def _collect_modified_files(
        self,
        session_id: str,
        env_id: str,
    ) -> dict[str, str | None]:
        """Read back files the agent changed, scoped to the scaffold tree.

        Uses the environment's ``changes`` endpoint (each entry carries a
        ``status`` of ``created`` / ``modified`` / ``deleted``). Only paths
        under the scaffold prefixes are kept — the environment holds many
        unrelated files (logs, caches) we must not surface as mutations.

        Both post-bind reads (``list_changes`` and each ``read_file``) go
        through :func:`_retry_on_503` so a runner still provisioning is
        retried before a persistent failure is swallowed — the ``changes``
        listing degrades to empty and an unreadable file is skipped rather
        than crashing the round.
        """
        modified: dict[str, str | None] = {}
        try:
            changes = await _retry_on_503(lambda: self.client.list_changes(session_id, env_id))
        except OmnigentError:
            changes = []
        for entry in changes:
            path = entry.get("path")
            status = entry.get("status")
            if not isinstance(path, str) or not _is_scaffold_path(path):
                continue
            if status == "deleted":
                modified[path] = None
            else:  # created / modified
                with contextlib.suppress(OmnigentError):
                    content = await _retry_on_503(
                        lambda p=path: self.client.read_file(session_id, env_id, p)
                    )
                    text = content.get("content")
                    if isinstance(text, str):
                        modified[path] = text
        return modified


# ---------------------------------------------------------------------------
# Managed-host session helpers
#
# The managed Omnigent server requires a two-step flow: the multipart bundle
# upload only REGISTERS the agent; a runner is bound only when a session is
# opened with ``host_type="managed"``. During the brief provisioning window
# post-bind calls can transiently return 503, so they are retried. These
# helpers are shared with :mod:`anvil.orchestrator.conversion` (the converter
# runs the identical managed flow) — extracted here, the lower-level module,
# to avoid a circular import (conversion already imports from this module).
# ``progress`` is optional so the optimizer backend (no live progress feed)
# can call them with the default no-op while conversion passes its UI feed.
# ---------------------------------------------------------------------------


def _noop_progress(*_args: Any, **_kwargs: Any) -> None:
    """A progress callback that does nothing (the backend has no UI feed)."""


async def _wait_for_runner(
    client: Any,
    session_id: str,
    progress: Callable[..., None] | None = None,
    *,
    max_attempts: int = 6,
    delay: float = 2.0,
    raise_on_timeout: bool = False,
    expected_runner_id: str | None = None,
) -> None:
    """Poll ``get_session`` until the runner is online (or give up).

    The managed host usually provisions the runner synchronously in the
    create response, but a brief async window is possible. This polls
    rather than blocking indefinitely (bounded by ``max_attempts``) so a
    stuck host surfaces promptly instead of hanging.

    The poll succeeds when the snapshot reports ``runner_online`` True and,
    when ``expected_runner_id`` is given, the snapshot's ``runner_id``
    equals it — i.e. the explicit bind (resume-bind) has been applied by the
    server. When ``expected_runner_id`` is None the poll only checks
    ``runner_online`` — this preserves the converter's
    (:mod:`anvil.orchestrator.conversion`) existing behavior, which does not
    bind explicitly.

    On exhaustion (the runner never came online within ``max_attempts``):

    * ``raise_on_timeout=True`` — raise :class:`OmnigentError` so a
      permanently-offline (or never-bound) runner becomes a DISTINGUISHABLE
      backend failure (the optimizer backend maps this to ``optimizer_error``
      → ``Decision.INFRA_FAIL`` rather than a silent success/noop).
    * ``raise_on_timeout=False`` (default) — log and return, letting the
      caller attempt the message anyway (the ``runner_online`` snapshot can
      be stale; the send's own 503-retry is the safety net). This preserves
      the converter's (:mod:`anvil.orchestrator.conversion`) existing flow.
    """
    progress = progress or _noop_progress
    for attempt in range(1, max_attempts + 1):
        with contextlib.suppress(OmnigentError):
            snapshot = await client.get_session(session_id)
            if snapshot.get("runner_online") and (
                expected_runner_id is None
                or snapshot.get("runner_id") == expected_runner_id
            ):
                return
        if attempt < max_attempts:
            progress("agent_session", f"Waiting for runner… ({attempt}/{max_attempts})")
            await anyio.sleep(delay)
    if raise_on_timeout:
        raise OmnigentError(
            f"managed runner for session {session_id} never came online after "
            f"{max_attempts} attempts",
            status_code=None,
        )
    progress("agent_session", "Runner not yet online; attempting message anyway.")


async def _send_with_retry(
    client: Any,
    session_id: str,
    text: str,
    progress: Callable[..., None] | None = None,
    *,
    max_attempts: int = 3,
    delay: float = 3.0,
) -> None:
    """Send a message, retrying on transient 503 (runner still provisioning)."""
    progress = progress or _noop_progress
    last_err: OmnigentError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            await client.send_message(session_id, text)
            return
        except OmnigentError as exc:
            last_err = exc
            if exc.status_code == 503 and attempt < max_attempts:
                progress("agent_session", f"Runner busy, retrying… ({attempt}/{max_attempts})")
                await anyio.sleep(delay)
                continue
            raise
    assert last_err is not None
    raise last_err


async def _retry_on_503(
    op: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int = 3,
    delay: float = 3.0,
) -> Any:
    """Await ``op()``, retrying on transient HTTP 503 (runner provisioning).

    Generic sibling of :func:`_send_with_retry` for the other post-bind
    calls (``list_environments`` / scaffold upload) whose 503-during-
    provisioning must retry rather than hard-fail. A non-503
    :class:`OmnigentError` (or any other exception) propagates immediately
    so a genuine backend failure still surfaces as an ``optimizer_error``.
    """
    last_err: OmnigentError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await op()
        except OmnigentError as exc:
            last_err = exc
            if exc.status_code == 503 and attempt < max_attempts:
                await anyio.sleep(delay)
                continue
            raise
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------


def _needed_harness(agent_yaml_path: Path) -> str | None:
    """The harness the optimizer agent needs (``executor.config.harness``).

    Read best-effort from the agent spec so the runner-bind can PREFER a
    runner whose ``harnesses`` advertises it (e.g. ``claude-sdk``). Returns
    ``None`` on any read/parse failure or when the field is absent/blank —
    the caller then binds the first online runner regardless of harness.
    """
    try:
        raw = yaml.safe_load(Path(agent_yaml_path).read_text(encoding="utf-8")) or {}
        harness = (raw.get("executor", {}) or {}).get("config", {}).get("harness")
    except Exception:  # noqa: BLE001 — best-effort; harness preference is optional
        return None
    return harness if isinstance(harness, str) and harness else None


def _build_agent_bundle(agent_yaml_path: Path, *, model: str, max_turns: int) -> bytes:
    """Package ``agent_yaml_path`` as a tar.gz whose root is ``config.yaml``.

    Substitutes ``executor.model`` and ``executor.config.max_turns`` so the
    round can pin them per-session without mutating the file on disk, and
    strips ``guardrails`` (the managed server's policy registry has no custom
    handlers; the prompt is the authoritative contract). The bundle format
    matches the Omnigent agent spec (a tar.gz with ``config.yaml`` at the
    root — the same shape as the bundled ``polly`` agent downloaded from
    ``GET /v1/sessions/{id}/agent/contents``).
    """
    raw = yaml.safe_load(Path(agent_yaml_path).read_text(encoding="utf-8")) or {}
    # Strip guardrails before upload: the managed Omnigent server's policy
    # registry does not include custom handlers like
    # ``omnigent.inner.nessie.policies.cel_policy``, so bundles carrying
    # ``guardrails.policies`` are rejected with HTTP 400. The prompt is the
    # authoritative contract; guardrails are defense-in-depth on top of it.
    raw.pop("guardrails", None)
    executor = raw.setdefault("executor", {})
    executor["model"] = model
    config = executor.setdefault("config", {})
    config["max_turns"] = max_turns
    config_bytes = yaml.safe_dump(raw, sort_keys=False).encode("utf-8")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="config.yaml")
        info.size = len(config_bytes)
        tar.addfile(info, io.BytesIO(config_bytes))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Item text helpers
# ---------------------------------------------------------------------------


def _text_from_item(item: Any) -> str:
    """Extract assistant text from a ``response.output_item.done`` item."""
    if not isinstance(item, dict):
        return ""
    if item.get("type") != "message" or item.get("role") != "assistant":
        return ""
    return _text_from_content(item.get("content"))


def _text_from_content(content: Any) -> str:
    """Concatenate ``output_text`` blocks from a message content list."""
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "output_text":
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "".join(texts)


def _is_scaffold_path(path: str) -> bool:
    """True if ``path`` is under one of the optimizer's writable prefixes."""
    return any(path == p.rstrip("/") or path.startswith(p) for p in _SCAFFOLD_PREFIXES)
