"""Thin async HTTP client for the Omnigent server REST API.

The Omnigent server (``http://localhost:6767`` for local dev) hosts a
FastAPI app with an OpenAPI spec at ``/openapi.json`` and Swagger UI at
``/docs``. This module wraps the subset of endpoints the optimizer
backend needs: session lifecycle, environment filesystem, the live SSE
event stream, conversation items, and the (hidden-from-schema) event
ingestion endpoint that sends a user message to the agent.

Design notes:

* **httpx** is the only HTTP dependency (declared explicitly in
  ``pyproject.toml``). The client is injected (``client=``) so tests can
  pass a fake :class:`httpx.AsyncClient`-shaped object without a live
  server.
* **SSE** frames are ``event: <type>\\ndata: <json>\\n\\n``; the stream
  generator yields parsed ``(event_type, data_dict)`` tuples and stops on
  ``[DONE]`` or when the source closes.
* **Errors** raise :class:`OmnigentError` carrying the status code and
  response body so the backend can surface a useful transcript-on-failure.

The request/response shapes were verified against the live server's
OpenAPI spec and by probing ``localhost:6767`` directly (see the session
creation multipart contract at ``POST /v1/sessions`` and the hidden
``POST /v1/sessions/{id}/events`` ingestion route in the omnigent package).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx


def _default_token_fn() -> str:
    """Mint a fresh Databricks OAuth/SP token for the app's own identity.

    Reuses the EXACT same token-minting mechanism the AI Gateway client
    uses (:func:`anvil.runtime.client._get_fresh_sp_token`) rather than
    duplicating the SDK resolution logic. Imported lazily so importing
    this module never requires the Databricks SDK — the token is only
    needed on the default (app-identity) auth path, which is exercised at
    request time, not at import.
    """
    from anvil.runtime.client import _get_fresh_sp_token

    return _get_fresh_sp_token()


def resolve_omnigent_server_url() -> str | None:
    """Resolve the Omnigent control-plane base URL for this deployment.

    Resolution precedence (evaluated lazily, at call time — never frozen at
    import — so a deployed App picks up its own workspace and local dev with an
    explicit env still works):

    1. **Explicit override** — if ``OMNIGENT_SERVER_URL`` is set and non-empty,
       it is returned verbatim (local dev / explicit pin; nothing is appended).
    2. **Derived from the workspace host** — otherwise
       ``f"{host}/api/2.0/omnigent"`` where ``host`` is ``DATABRICKS_HOST``
       normalized: a trailing slash and a trailing ``/omnigent`` or
       ``/api/2.0/omnigent`` segment are stripped before exactly one
       ``/api/2.0/omnigent`` is appended, so a host that already carries either
       suffix does not produce a doubled ``.../api/2.0/omnigent/api/2.0/omnigent``
       or ``.../omnigent/api/2.0/omnigent`` path. Databricks Apps inject
       ``DATABRICKS_HOST`` as a BARE hostname with no scheme (e.g.
       ``fevm-stable-classic-7ppxjq.cloud.databricks.com``), so a derived value
       that lacks an ``http://``/``https://`` scheme is normalized to
       ``https://`` (an httpx request without a scheme raises
       ``UnsupportedProtocol``). A host that already carries a scheme keeps it,
       untouched — never doubled. When the app runs as a Databricks App the
       Omnigent REST API is mounted at ``https://<workspace-host>/api/2.0/omnigent``
       (the workspace-hosted API service — NOT ``<host>/omnigent``, which is the
       interactive browser UI surface and has no ``/v1/sessions`` REST route).

    Returns ``None`` when neither source yields a value (Omnigent unconfigured
    — callers surface a 503 / skip best-effort work).
    """
    explicit = os.getenv("OMNIGENT_SERVER_URL")
    if explicit:
        return explicit
    host = os.getenv("DATABRICKS_HOST")
    if host:
        host = host.rstrip("/")
        # Strip an already-present Omnigent suffix so we never double it up
        # (mirrors ``build_session_url``'s host recovery). Check the longer
        # legacy suffix first.
        for suffix in ("/api/2.0/omnigent", "/omnigent"):
            if host.endswith(suffix):
                host = host[: -len(suffix)].rstrip("/")
                break
        if host:
            # Databricks Apps inject a scheme-less bare hostname; guarantee an
            # http(s):// scheme on the DERIVED value so the httpx client never
            # raises ``UnsupportedProtocol``. A host that already carries a
            # scheme is left untouched (never doubled). URI schemes are
            # case-insensitive (RFC 3986), so lowercase only the CHECK — the
            # host's original casing is preserved in the returned URL.
            if not host.lower().startswith(("http://", "https://")):
                host = f"https://{host}"
            return f"{host}/api/2.0/omnigent"
    return None


def _sdk_workspace_id() -> str | None:
    """Best-effort numeric workspace/org id via the Databricks SDK.

    Returns ``None`` on any failure (SDK missing, no auth, network error) so
    the caller can fall back to the ``DATABRICKS_WORKSPACE_ID`` env — the
    ``?o=`` UI-link param is optional and the link resolves without it.
    """
    try:
        from databricks.sdk import WorkspaceClient  # noqa: PLC0415 - lazy, optional

        wsid = WorkspaceClient().get_workspace_id()
    except Exception:  # noqa: BLE001 — best-effort; ?o= is optional
        return None
    return str(wsid) if wsid else None


def resolve_omnigent_workspace_id() -> str | None:
    """Resolve the numeric workspace/org id for the Omnigent ``?o=`` UI link.

    Derives the id dynamically via the Databricks SDK where feasible
    (best-effort), falling back to the ``DATABRICKS_WORKSPACE_ID`` env. No
    workspace id is ever hard-coded. Returns ``None`` when neither yields a
    value; the ``?o=`` param is then omitted and the link still resolves.
    """
    wsid = _sdk_workspace_id()
    if wsid:
        return wsid
    return os.getenv("DATABRICKS_WORKSPACE_ID") or None


def build_session_url(server_url: str, session_id: str, workspace_id: str | None = None) -> str:
    """Build a navigable Databricks UI URL for an Omnigent session.

    ``server_url`` may be either the legacy API surface
    (``…/api/2.0/omnigent``) or the workspace-derived base
    (``<workspace_host>/omnigent`` — see :func:`resolve_omnigent_server_url`);
    the navigable UI path is ``<workspace_host>/omnigent/c/<session_id>``. The
    trailing ``/api/2.0/omnigent`` or ``/omnigent`` suffix is stripped to
    recover the workspace host. A ``?o=<workspace_id>`` query param is appended
    when a workspace id is supplied, and omitted otherwise — the link still
    resolves without it.
    """
    host = server_url.rstrip("/")
    for suffix in ("/api/2.0/omnigent", "/omnigent"):
        if host.endswith(suffix):
            host = host[: -len(suffix)]
            break
    url = f"{host}/omnigent/c/{session_id}"
    if workspace_id:
        url = f"{url}?o={workspace_id}"
    return url


class OmnigentError(RuntimeError):
    """A non-2xx response from the Omnigent server."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class SessionCreateMetadata:
    """Metadata JSON part for multipart ``POST /v1/sessions``.

    Mirrors the server's ``SessionCreateMetadata`` model. Only
    session-level metadata lives here; the agent spec comes from the
    bundle. ``extra="forbid"`` is enforced server-side.
    """

    title: str | None = None
    labels: dict[str, str] | None = None
    reasoning_effort: str | None = None
    host_id: str | None = None
    workspace: str | None = None
    terminal_launch_args: list[str] | None = None
    parent_session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.title is not None:
            out["title"] = self.title
        if self.labels is not None:
            out["labels"] = self.labels
        if self.reasoning_effort is not None:
            out["reasoning_effort"] = self.reasoning_effort
        if self.host_id is not None:
            out["host_id"] = self.host_id
        if self.workspace is not None:
            out["workspace"] = self.workspace
        if self.terminal_launch_args is not None:
            out["terminal_launch_args"] = self.terminal_launch_args
        if self.parent_session_id is not None:
            out["parent_session_id"] = self.parent_session_id
        return out


class OmnigentClient:
    """Async client for the Omnigent server REST API.

    Authentication precedence (see :meth:`_resolve_token`):

    * If ``auth_token`` (or ``OMNIGENT_AUTH_TOKEN`` upstream) is set and
      non-empty, it is sent verbatim as a static ``Authorization: Bearer
      <token>`` header — the local/dev / explicit-override path.
    * OTHERWISE the client authenticates as the deployed app's own
      Databricks identity: it mints a fresh OAuth/SP token via
      ``token_fn`` on EVERY request (a request event hook), because SP
      tokens expire (~1h). This matches the deployment intent — the app
      reaches the in-workspace Omnigent control-plane with its own
      identity. An empty/None ``Authorization`` header is never sent.

    Args:
        base_url: Server origin, e.g. ``http://localhost:6767``.
        auth_token: Optional explicit Bearer token. When set (non-empty) it
            wins and is used verbatim; when ``None`` the app-identity token
            minter is used per-request instead. An explicitly-provided but
            empty/blank value raises — it is a misconfiguration, not a
            request to skip auth (pass ``None`` for app-identity auth).
        client: Injected :class:`httpx.AsyncClient` (or a test fake). When
            omitted a fresh client is created and closed with this wrapper.
        timeout: Default request timeout in seconds.
        token_fn: Zero-arg callable returning a fresh bearer token, used
            only when no explicit ``auth_token`` is set. Defaults to the
            app's Databricks OAuth/SP minter (:func:`_default_token_fn`).
            Injected in tests.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
        token_fn: Callable[[], str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._owns_client = client is None
        # An explicitly-provided but empty/blank token is a misconfiguration:
        # fail loudly at construction rather than silently sending an
        # unauthenticated request or falling through to app-identity minting.
        # ``None`` is the sentinel for "use the app's own Databricks identity".
        if auth_token is not None and not str(auth_token).strip():
            raise OmnigentError(
                "OmnigentClient received an explicit but empty auth token; pass "
                "a non-empty token, or pass None to authenticate with the app's "
                "own Databricks OAuth/SP identity."
            )
        # An explicit token wins and is sent verbatim. When it is unset we
        # fall back to minting the app's own Databricks OAuth token per
        # request (SP tokens expire ~1h), so we must NOT freeze a header.
        self._explicit_token = auth_token or None
        self._token_fn = token_fn or _default_token_fn
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._explicit_token:
            headers["Authorization"] = f"Bearer {self._explicit_token}"
        # Register the per-request auth hook only on the app-identity path
        # (no explicit token). The hook refreshes the bearer on every
        # request so a token minted at construction time can never expire
        # out from under a long-running optimization.
        event_hooks: dict[str, list[Any]] = {}
        if not self._explicit_token:
            event_hooks["request"] = [self._inject_auth_header]
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
            event_hooks=event_hooks,
        )
        if client is not None:
            # Respect caller-provided client auth headers if not already set.
            for key, value in headers.items():
                self._client.headers.setdefault(key, value)
            # Wire the per-request auth hook onto a caller-provided real
            # httpx client too, when it exposes ``event_hooks`` (test fakes
            # generally do not — they run no requests, so this is skipped).
            if not self._explicit_token:
                hooks = getattr(self._client, "event_hooks", None)
                if isinstance(hooks, dict):
                    hooks.setdefault("request", []).append(self._inject_auth_header)

    def _resolve_token(self) -> str:
        """Resolve the non-empty bearer token for a request, or RAISE.

        Explicit token wins verbatim; otherwise mint the app's Databricks
        OAuth/SP token fresh. Never returns an empty/None token: an empty
        resolved/minted token raises :class:`OmnigentError` so we fail
        loudly and locally rather than sending an unauthenticated request
        (which the workspace bounces to the SSO front door — the exact
        failure mode this client exists to prevent).
        """
        if self._explicit_token:
            return self._explicit_token
        token = self._token_fn()
        if not token or not str(token).strip():
            raise OmnigentError(
                "refusing to send an unauthenticated Omnigent request: the "
                "Databricks OAuth/SP token minter returned an empty token. "
                "Check the app's identity / OMNIGENT_AUTH_TOKEN configuration."
            )
        return token

    async def _inject_auth_header(self, request: httpx.Request) -> None:
        """httpx request event hook: set a fresh ``Authorization`` bearer.

        Runs per request on the app-identity path. :meth:`_resolve_token`
        guarantees a non-empty token or raises BEFORE the request reaches
        the transport — a request is never sent without a valid bearer.
        """
        request.headers["Authorization"] = f"Bearer {self._resolve_token()}"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OmnigentClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        bundle_bytes: bytes,
        metadata: SessionCreateMetadata | None = None,
        *,
        bundle_filename: str = "agent.tar.gz",
    ) -> dict[str, Any]:
        """Create a session by uploading an agent bundle (multipart).

        ``POST /v1/sessions`` with ``multipart/form-data``: a ``metadata``
        JSON part and a ``bundle`` file part (the agent tar.gz). Returns the
        server's ``CreatedSessionResponse``:
        ``{session_id, agent_id, agent_name}``.
        """
        meta = (metadata or SessionCreateMetadata()).to_dict()
        files = {"bundle": (bundle_filename, bundle_bytes, "application/gzip")}
        data = {"metadata": json.dumps(meta)}
        resp = await self._client.post("/v1/sessions", data=data, files=files)
        self._raise_for_status(resp, "create_session")
        return resp.json()

    async def create_session_from_agent(
        self,
        agent_id: str,
        *,
        title: str | None = None,
        initial_items: list[dict[str, Any]] | None = None,
        host_id: str | None = None,
        host_type: str | None = None,
        workspace: str | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Create a session bound to an already-registered agent (JSON).

        ``POST /v1/sessions`` with ``application/json`` (``SessionCreateRequest``).
        ``initial_items`` seeds the input queue — typically a single user
        ``"message"`` — so the first turn starts without a separate send.
        Returns the full ``SessionResponse`` snapshot.
        """
        payload: dict[str, Any] = {"agent_id": agent_id}
        if title is not None:
            payload["title"] = title
        if initial_items is not None:
            payload["initial_items"] = initial_items
        if host_id is not None:
            payload["host_id"] = host_id
        if host_type is not None:
            payload["host_type"] = host_type
        if workspace is not None:
            payload["workspace"] = workspace
        if model_override is not None:
            payload["model_override"] = model_override
        resp = await self._client.post("/v1/sessions", json=payload)
        self._raise_for_status(resp, "create_session_from_agent")
        return resp.json()

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """``GET /v1/sessions/{id}`` — the full session snapshot.

        The ``SessionResponse`` carries the runner-affinity fields the
        managed flow needs: ``runner_id`` (null on an unbound conversation),
        ``runner_online``, ``host_id``, ``host_online``, ``host_resumable``.
        """
        resp = await self._client.get(f"/v1/sessions/{session_id}")
        self._raise_for_status(resp, "get_session")
        return resp.json()

    async def list_runners(self) -> list[dict[str, Any]]:
        """``GET /v1/runners`` — the registered runner pool.

        Returns the ``data`` list of runner dicts, each carrying
        ``runner_id``, ``online`` (bool), and ``harnesses`` (the list of
        harnesses that runner can execute, e.g. ``["claude-native",
        "claude-sdk", ...]``). Used to discover the currently-registered
        online runner a managed conversation must be bound to.
        """
        resp = await self._client.get("/v1/runners")
        self._raise_for_status(resp, "list_runners")
        body = resp.json()
        return list(body.get("data", []))

    async def bind_runner(self, session_id: str, runner_id: str) -> dict[str, Any]:
        """``PATCH /v1/sessions/{id}`` with ``{"runner_id": ...}`` — bind a runner.

        This is the mutable affinity primitive (``UpdateSessionRequest`` with
        ``runner_id``): create-bind, resume-bind, and recover-bind all send
        the currently-registered runner id, and the server atomically replaces
        ``conversations.runner_id`` with that value (last-write-wins). It is
        the "resume the session to bind a registered runner" the HTTP 409
        error on an unbound managed conversation demands. Returns the updated
        ``SessionResponse`` snapshot.
        """
        resp = await self._client.patch(f"/v1/sessions/{session_id}", json={"runner_id": runner_id})
        self._raise_for_status(resp, "bind_runner")
        return resp.json()

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        """``DELETE /v1/sessions/{id}`` — tombstone the conversation."""
        resp = await self._client.delete(f"/v1/sessions/{session_id}")
        self._raise_for_status(resp, "delete_session")
        return resp.json()

    # ------------------------------------------------------------------
    # Environments + filesystem
    # ------------------------------------------------------------------

    async def list_environments(self, session_id: str) -> list[dict[str, Any]]:
        """``GET /v1/sessions/{id}/resources/environments`` — the env list."""
        resp = await self._client.get(f"/v1/sessions/{session_id}/resources/environments")
        self._raise_for_status(resp, "list_environments")
        body = resp.json()
        return list(body.get("data", []))

    async def upload_file(
        self,
        session_id: str,
        env_id: str,
        relative_path: str,
        content: str,
    ) -> dict[str, Any]:
        """``PUT /v1/sessions/{id}/resources/environments/{env}/filesystem/{path}``.

        Body is JSON ``{"content": <str>}``. Returns the write result.
        """
        path = (
            f"/v1/sessions/{session_id}/resources/environments/{env_id}/filesystem/{relative_path}"
        )
        resp = await self._client.put(path, json={"content": content})
        self._raise_for_status(resp, "upload_file")
        return resp.json()

    async def read_file(self, session_id: str, env_id: str, relative_path: str) -> dict[str, Any]:
        """``GET .../filesystem/{path}`` — file content (``file_content``) or a dir listing."""
        path = (
            f"/v1/sessions/{session_id}/resources/environments/{env_id}/filesystem/{relative_path}"
        )
        resp = await self._client.get(path)
        self._raise_for_status(resp, "read_file")
        return resp.json()

    async def delete_file(self, session_id: str, env_id: str, relative_path: str) -> dict[str, Any]:
        """``DELETE .../filesystem/{path}`` — remove a file or directory."""
        path = (
            f"/v1/sessions/{session_id}/resources/environments/{env_id}/filesystem/{relative_path}"
        )
        resp = await self._client.delete(path)
        self._raise_for_status(resp, "delete_file")
        return resp.json()

    async def list_changes(
        self,
        session_id: str,
        env_id: str,
    ) -> list[dict[str, Any]]:
        """``GET .../environments/{env}/changes`` — files changed in the env.

        Returns the ``data`` list of change entries, each carrying
        ``path``, ``status`` (``"created"`` / ``"modified"`` /
        ``"deleted"``), ``bytes``, and ``modified_at``. Used by the
        optimizer backend to populate ``OptimizerResult.modified_files``
        after the agent's turn.
        """
        path = f"/v1/sessions/{session_id}/resources/environments/{env_id}/changes"
        resp = await self._client.get(path)
        self._raise_for_status(resp, "list_changes")
        body = resp.json()
        return list(body.get("data", []))

    async def list_files(
        self,
        session_id: str,
        env_id: str,
        path: str = "",
        *,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """List a directory's entries.

        ``GET .../filesystem`` (root) or ``GET .../filesystem/{path}`` (a
        directory). Returns the ``data`` list of entry dicts
        (``{id, name, path, type, bytes, modified_at}``).
        """
        base = f"/v1/sessions/{session_id}/resources/environments/{env_id}/filesystem"
        url = f"{base}/{path}" if path else base
        resp = await self._client.get(url, params={"limit": limit, "order": "asc"})
        self._raise_for_status(resp, "list_files")
        body = resp.json()
        # A file read returns a ``file_content`` object, not a listing.
        if body.get("object") == "session.environment.filesystem.file_content":
            return []
        return list(body.get("data", []))

    # ------------------------------------------------------------------
    # Conversation items + event ingestion
    # ------------------------------------------------------------------

    async def list_items(
        self,
        session_id: str,
        *,
        limit: int = 100,
        after: str | None = None,
    ) -> dict[str, Any]:
        """``GET /v1/sessions/{id}/items`` — paginated conversation items.

        Returns ``{data, first_id, last_id, has_more}``. Each item is a
        ``ConversationItem`` (message / function_call / function_call_output
        / resource_event / ...).
        """
        params: dict[str, Any] = {"limit": limit}
        if after is not None:
            params["after"] = after
        resp = await self._client.get(f"/v1/sessions/{session_id}/items", params=params)
        self._raise_for_status(resp, "list_items")
        return resp.json()

    async def send_message(self, session_id: str, text: str) -> dict[str, Any]:
        """Send a user message to the session's agent.

        ``POST /v1/sessions/{id}/events`` (hidden from the public OpenAPI
        schema) with a ``SessionEventInput`` of ``type: "message"``. The
        server starts (or steers) a turn and streams the agent's response
        over the SSE stream. Returns ``{queued, item_id}``.
        """
        event = {
            "type": "message",
            "data": {"role": "user", "content": [{"type": "input_text", "text": text}]},
        }
        resp = await self._client.post(f"/v1/sessions/{session_id}/events", json=event)
        self._raise_for_status(resp, "send_message")
        return resp.json()

    # ------------------------------------------------------------------
    # SSE stream
    # ------------------------------------------------------------------

    async def stream_session(self, session_id: str) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """``GET /v1/sessions/{id}/stream`` — the live SSE event stream.

        Yields parsed ``(event_type, data_dict)`` tuples. Frames are
        ``event: <type>\\ndata: <json>\\n\\n``. The generator stops on the
        ``[DONE]`` sentinel or when the source closes; ``response.completed``
        is yielded like any other event (it marks a per-turn boundary, not
        stream completion — callers decide when to stop draining). ``data``
        that is not valid JSON is yielded as a dict
        ``{"_raw": <text>}`` so callers never crash on a malformed frame.
        """
        url = f"/v1/sessions/{session_id}/stream"
        async with self._client.stream("GET", url) as response:
            self._raise_for_status(response, "stream_session")
            event_type = "message"
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                line = line.rstrip("\r\n")
                if line == "":
                    # End of a frame; emit if we accumulated data.
                    if data_lines:
                        raw = "\n".join(data_lines)
                        yield event_type, _parse_sse_data(raw)
                    event_type = "message"
                    data_lines = []
                    continue
                if line == "data: [DONE]":
                    return
                if line.startswith("event:"):
                    event_type = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].lstrip())
            # Flush a trailing frame if the stream closed mid-frame.
            if data_lines:
                raw = "\n".join(data_lines)
                yield event_type, _parse_sse_data(raw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _raise_for_status(resp: httpx.Response, op: str) -> None:
        if resp.status_code >= 400:
            body = resp.text
            raise OmnigentError(
                f"Omnigent {op} failed: HTTP {resp.status_code}",
                status_code=resp.status_code,
                body=body,
            )


def _parse_sse_data(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": raw}
