"""Omnigent-backed optimizer session — runs the agent on a managed server.

Implements :class:`OptimizerBackend` by driving a managed Omnigent server
over REST (see :mod:`anvil.optimizer.omnigent_client`). One round maps to
one ephemeral Omnigent session:

1. **Build the bundle** — package ``agents/forge_optimizer.yaml`` as a
   tar.gz whose root ``config.yaml`` is the agent spec. The ``model`` and
   ``max_turns`` args are substituted into the spec so each round can pin
   them without editing the file on disk.
2. **Create the session** — ``POST /v1/sessions`` (multipart: metadata +
   bundle) returns a session-scoped agent id.
3. **Upload the scaffold** — the round's ``scaffold_files`` dict is written
   to the session's ``default`` environment filesystem so the agent can
   ``Read`` the same tree the local backend reads from ``cwd``.
4. **Send the prompt** — the round prompt goes in as the first user message
   via the hidden ``POST /v1/sessions/{id}/events`` ingestion route, which
   starts the agent's turn.
5. **Drain the stream** — the SSE event stream is drained into a
   transcript (assistant text deltas + completed message items). The
   stream also gates completion: the drain stops on
   ``response.completed`` / ``[DONE]``.
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

import contextlib
import io
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from anvil.optimizer.omnigent_client import (
    OmnigentClient,
    OmnigentError,
    SessionCreateMetadata,
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
        bundle = _build_agent_bundle(
            self.agent_bundle_path,
            model=model or self.default_model,
            max_turns=max_turns or self.default_max_turns,
        )
        session_id: str | None = None
        session_url: str | None = None

        try:
            created = await self.client.create_session(bundle, metadata=self.create_metadata)
            session_id = created["session_id"]
            session_url = f"{self.server_url.rstrip('/')}/sessions/{session_id}"

            env_id = await self._resolve_environment(session_id)
            await self._upload_scaffold(session_id, env_id, scaffold_files)
            await self.client.send_message(session_id, prompt)
            stream_text, turns_used = await self._drain_stream(session_id)

            # The stream is the primary transcript source; fall back to the
            # persisted conversation items only if the stream text did NOT
            # yield a parseable action. Checking ``parse_action`` (not a raw
            # triple-backtick scan) avoids letting an unrelated or malformed
            # fence in the stream suppress a valid items fallback — a
            # ``bad_json`` / ``schema_mismatch`` fence is just as much a
            # non-success as no fence at all.
            transcript = stream_text
            stream_parse = parse_action(transcript)
            if stream_parse.parse_status not in ("ok", "ok_last_of_many"):
                items_text = await self._transcript_from_items(session_id)
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
        )

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

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

    async def _drain_stream(self, session_id: str) -> tuple[str, int | None]:
        """Drain the SSE stream into a transcript; return (text, turns_used).

        Collects ``response.output_text.delta`` fragments and the text of
        completed assistant messages (``response.output_item.done``). Stops
        on ``response.completed`` (each such event counts as one turn) — the
        underlying generator also returns on the ``[DONE]`` sentinel.
        """
        parts: list[str] = []
        turns = 0
        async for event_type, data in self.client.stream_session(session_id):
            if event_type == _DELTA_TYPE:
                delta = data.get("delta")
                if isinstance(delta, str):
                    parts.append(delta)
            elif event_type == _ITEM_DONE_TYPE:
                parts.append(_text_from_item(data.get("item")))
            elif event_type == _STATUS_TYPE:
                # An idle status after a running turn means the agent is
                # done; stop draining so we don't block on the live stream.
                if data.get("status") == "idle" and parts:
                    break
            elif event_type == _COMPLETED_TYPE:
                turns += 1
                break
        transcript = "".join(parts).strip()
        return transcript, (turns or None)

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
        """
        modified: dict[str, str | None] = {}
        try:
            changes = await self.client.list_changes(session_id, env_id)
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
                    content = await self.client.read_file(session_id, env_id, path)
                    text = content.get("content")
                    if isinstance(text, str):
                        modified[path] = text
        return modified


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------


def _build_agent_bundle(agent_yaml_path: Path, *, model: str, max_turns: int) -> bytes:
    """Package ``agent_yaml_path`` as a tar.gz whose root is ``config.yaml``.

    Substitutes ``executor.model`` and ``executor.config.max_turns`` so the
    round can pin them per-session without mutating the file on disk. The
    bundle format matches the Omnigent agent spec (a tar.gz with
    ``config.yaml`` at the root — the same shape as the bundled ``polly``
    agent downloaded from ``GET /v1/sessions/{id}/agent/contents``).
    """
    raw = yaml.safe_load(Path(agent_yaml_path).read_text(encoding="utf-8")) or {}
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
