"""Tests for :class:`OmnigentClient` — the thin async Omnigent REST wrapper.

No live server and no ``respx``/``pytest-httpx`` (pypi is blocked on this
Mac). Instead we inject a fake :class:`httpx.AsyncClient`-shaped object
that records every call and returns canned responses. The fake is
deliberately minimal — it only implements the surface
:class:`OmnigentClient` touches (``post``/``get``/``put``/``delete``/
``stream``/``headers``/``aclose``).

Async client methods are driven via ``asyncio.run`` (sync test
functions) because ``pytest-asyncio`` is not a project dependency.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from anvil.optimizer.omnigent_client import (
    OmnigentClient,
    OmnigentError,
    SessionCreateMetadata,
    _parse_sse_data,
    build_session_url,
    resolve_omnigent_server_url,
    resolve_omnigent_workspace_id,
)

# ---------------------------------------------------------------------------
# Fake httpx layer
# ---------------------------------------------------------------------------


@dataclass
class FakeResponse:
    """Mimics ``httpx.Response`` for the methods OmnigentClient reads."""

    status_code: int = 200
    _body: Any = None
    text: str = ""

    def json(self) -> Any:
        if isinstance(self._body, (dict, list)):
            return self._body
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body


@dataclass
class FakeStreamResponse:
    """Mimics the response yielded by ``httpx.AsyncClient.stream``."""

    status_code: int = 200
    lines: list[str] = field(default_factory=list)
    text: str = ""

    async def aiter_lines(self):  # type: ignore[no-untyped-def] — async generator
        for line in self.lines:
            yield line


class FakeAsyncClient:
    """Records calls and returns canned :class:`FakeResponse` objects.

    ``responses`` maps ``(method, url_substring)`` → response. Unmatched
    calls get a default 200 with an empty JSON body. The request methods
    (``post``/``get``/``put``/``delete``) are async — they await in the
    real ``httpx.AsyncClient``. ``stream`` is sync and returns an async
    context manager, matching httpx.
    """

    def __init__(self, responses: dict[tuple[str, str], Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = responses or {}
        self.headers: dict[str, str] = {}

    def _lookup(self, method: str, url: str) -> Any:
        for (m, sub), resp in self.responses.items():
            if m == method and sub in url:
                return resp
        return FakeResponse(status_code=200, _body={})

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return self._lookup("POST", url)

    async def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": "GET", "url": url, **kwargs})
        return self._lookup("GET", url)

    async def put(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": "PUT", "url": url, **kwargs})
        return self._lookup("PUT", url)

    async def patch(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": "PATCH", "url": url, **kwargs})
        return self._lookup("PATCH", url)

    async def delete(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": "DELETE", "url": url, **kwargs})
        return self._lookup("DELETE", url)

    def stream(self, method: str, url: str, **kwargs: Any):
        self.calls.append({"method": "STREAM", "url": url, "http_method": method, **kwargs})

        class _Ctx:
            def __init__(self, response: Any) -> None:
                self._response = response

            async def __aenter__(self) -> Any:
                return self._response

            async def __aexit__(self, *exc: object) -> None:
                pass

        return _Ctx(self._lookup("STREAM", url))

    async def aclose(self) -> None:
        pass


def _client_with(responses: dict[tuple[str, str], Any], token: str | None = None) -> OmnigentClient:
    return OmnigentClient("http://localhost:6767", token, client=FakeAsyncClient(responses))


def _last_call(c: OmnigentClient, method: str) -> dict[str, Any]:
    fake = c._client  # type: ignore[attr-defined]
    calls = [x for x in fake.calls if x["method"] == method]
    assert calls, f"no {method} call recorded"
    return calls[-1]


async def _drain_stream(c: OmnigentClient, session_id: str) -> list[tuple[str, dict]]:
    return [(e, d) async for e, d in c.stream_session(session_id)]


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_create_session_multipart_upload() -> None:
    resp = FakeResponse(
        status_code=201, _body={"session_id": "s1", "agent_id": "a1", "agent_name": "forge"}
    )
    c = _client_with({("POST", "/v1/sessions"): resp})
    meta = SessionCreateMetadata(title="round 7", labels={"round": "7"})
    out = asyncio.run(c.create_session(b"\x1f\x8bbundle", metadata=meta))

    assert out == {"session_id": "s1", "agent_id": "a1", "agent_name": "forge"}
    call = _last_call(c, "POST")
    assert "files" in call and "data" in call
    # The metadata is JSON-encoded in the form ``data`` field.
    assert json.loads(call["data"]["metadata"])["title"] == "round 7"
    # The bundle bytes travel in the ``files`` multipart part.
    assert call["files"]["bundle"][1] == b"\x1f\x8bbundle"


def test_create_session_from_agent_json_body() -> None:
    resp = FakeResponse(status_code=201, _body={"session_id": "s2"})
    c = _client_with({("POST", "/v1/sessions"): resp})
    out = asyncio.run(
        c.create_session_from_agent(
            "agent-xyz",
            title="t",
            initial_items=[{"type": "message", "data": {"role": "user", "content": []}}],
        )
    )
    assert out["session_id"] == "s2"
    call = _last_call(c, "POST")
    assert call["json"]["agent_id"] == "agent-xyz"
    assert call["json"]["title"] == "t"
    assert len(call["json"]["initial_items"]) == 1


def test_get_session() -> None:
    c = _client_with({("GET", "/v1/sessions/s1"): FakeResponse(_body={"id": "s1"})})
    out = asyncio.run(c.get_session("s1"))
    assert out["id"] == "s1"
    assert _last_call(c, "GET")["url"] == "/v1/sessions/s1"


def test_delete_session() -> None:
    c = _client_with({("DELETE", "/v1/sessions/s1"): FakeResponse(_body={"ok": True})})
    out = asyncio.run(c.delete_session("s1"))
    assert out == {"ok": True}
    assert _last_call(c, "DELETE")["url"] == "/v1/sessions/s1"


# ---------------------------------------------------------------------------
# Runner discovery + binding (managed-flow HTTP 409 fix)
# ---------------------------------------------------------------------------


def test_list_runners_returns_data_list() -> None:
    """``GET /v1/runners`` returns the pool's ``data`` list of runner dicts."""
    body = {
        "data": [
            {
                "runner_id": "runner_token_abc",
                "online": True,
                "harnesses": ["claude-native", "claude-sdk", "codex"],
            }
        ]
    }
    c = _client_with({("GET", "/v1/runners"): FakeResponse(_body=body)})
    runners = asyncio.run(c.list_runners())
    assert len(runners) == 1
    assert runners[0]["runner_id"] == "runner_token_abc"
    assert runners[0]["online"] is True
    assert "claude-sdk" in runners[0]["harnesses"]
    assert _last_call(c, "GET")["url"] == "/v1/runners"


def test_list_runners_empty_pool() -> None:
    """An empty pool degrades to an empty list, not a crash."""
    c = _client_with({("GET", "/v1/runners"): FakeResponse(_body={"data": []})})
    assert asyncio.run(c.list_runners()) == []


def test_bind_runner_patches_session_with_runner_id() -> None:
    """``bind_runner`` PATCHes /v1/sessions/{id} with a ``{"runner_id": ...}``
    body and returns the updated SessionResponse snapshot."""
    resp = FakeResponse(
        _body={"id": "sess-9", "runner_id": "runner_token_abc", "runner_online": True}
    )
    c = _client_with({("PATCH", "/v1/sessions/sess-9"): resp})
    out = asyncio.run(c.bind_runner("sess-9", "runner_token_abc"))
    assert out["runner_id"] == "runner_token_abc"
    call = _last_call(c, "PATCH")
    assert call["url"] == "/v1/sessions/sess-9"
    assert call["json"] == {"runner_id": "runner_token_abc"}


def test_bind_runner_raises_on_error() -> None:
    """A non-2xx from the bind PATCH raises OmnigentError with status + body."""
    c = _client_with(
        {("PATCH", "/v1/sessions/s1"): FakeResponse(status_code=409, text="not bound")}
    )
    with pytest.raises(OmnigentError) as exc_info:
        asyncio.run(c.bind_runner("s1", "runner-x"))
    assert exc_info.value.status_code == 409


# ---------------------------------------------------------------------------
# Environments + filesystem
# ---------------------------------------------------------------------------


def test_list_environments() -> None:
    c = _client_with(
        {
            ("GET", "environments"): FakeResponse(
                _body={"data": [{"id": "default"}, {"id": "other"}]}
            )
        }
    )
    envs = asyncio.run(c.list_environments("s1"))
    assert [e["id"] for e in envs] == ["default", "other"]


def test_upload_file_json_content_body() -> None:
    c = _client_with({("PUT", "filesystem"): FakeResponse(_body={"ok": True})})
    asyncio.run(c.upload_file("s1", "default", "scaffold/harness.yaml", "content here"))
    call = _last_call(c, "PUT")
    assert call["json"] == {"content": "content here"}
    assert "filesystem/scaffold/harness.yaml" in call["url"]


def test_read_file() -> None:
    c = _client_with(
        {("GET", "filesystem/config.yaml"): FakeResponse(_body={"content": "raw text"})}
    )
    out = asyncio.run(c.read_file("s1", "default", "config.yaml"))
    assert out["content"] == "raw text"


def test_delete_file() -> None:
    c = _client_with({("DELETE", "filesystem"): FakeResponse(_body={"ok": True})})
    asyncio.run(c.delete_file("s1", "default", "scaffold/old.md"))
    assert _last_call(c, "DELETE")["method"] == "DELETE"


def test_list_changes() -> None:
    body = {
        "data": [
            {"path": "scaffold/rules/x.md", "status": "created"},
            {"path": "scaffold/rules/y.md", "status": "deleted"},
        ]
    }
    c = _client_with({("GET", "changes"): FakeResponse(_body=body)})
    changes = asyncio.run(c.list_changes("s1", "default"))
    assert len(changes) == 2
    assert changes[1]["status"] == "deleted"


def test_list_files_directory() -> None:
    body = {"data": [{"name": "harness.yaml", "type": "file"}]}
    c = _client_with({("GET", "filesystem"): FakeResponse(_body=body)})
    entries = asyncio.run(c.list_files("s1", "default"))
    assert entries[0]["name"] == "harness.yaml"


def test_list_files_returns_empty_for_file_content() -> None:
    """A file read (file_content object) must not be mistaken for a listing."""
    body = {"object": "session.environment.filesystem.file_content", "content": "x"}
    c = _client_with({("GET", "filesystem"): FakeResponse(_body=body)})
    entries = asyncio.run(c.list_files("s1", "default", "harness.yaml"))
    assert entries == []


# ---------------------------------------------------------------------------
# Items + event ingestion
# ---------------------------------------------------------------------------


def test_list_items_pagination_params() -> None:
    c = _client_with({("GET", "/items"): FakeResponse(_body={"data": [], "has_more": False})})
    asyncio.run(c.list_items("s1", limit=50, after="item-3"))
    call = _last_call(c, "GET")
    assert call["params"] == {"limit": 50, "after": "item-3"}


def test_send_message_event_shape() -> None:
    c = _client_with({("POST", "/events"): FakeResponse(_body={"queued": True, "item_id": "i1"})})
    out = asyncio.run(c.send_message("s1", "do the thing"))
    assert out == {"queued": True, "item_id": "i1"}
    call = _last_call(c, "POST")
    assert call["url"] == "/v1/sessions/s1/events"
    event = call["json"]
    assert event["type"] == "message"
    assert event["data"]["role"] == "user"
    assert event["data"]["content"][0] == {"type": "input_text", "text": "do the thing"}


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------


def test_stream_session_parses_frames() -> None:
    sse_lines = [
        "event: response.output_text.delta",
        'data: {"delta": "hello "}',
        "",
        "event: response.output_text.delta",
        'data: {"delta": "world"}',
        "",
        "event: response.completed",
        'data: {"done": true}',
        "",
    ]
    c = _client_with({("STREAM", "/stream"): FakeStreamResponse(lines=sse_lines)})
    events = asyncio.run(_drain_stream(c, "s1"))
    assert events[0] == ("response.output_text.delta", {"delta": "hello "})
    assert events[1] == ("response.output_text.delta", {"delta": "world"})
    assert events[2][0] == "response.completed"


def test_stream_session_stops_on_done_sentinel() -> None:
    sse_lines = [
        'data: {"delta": "a"}',
        "",
        "data: [DONE]",
    ]
    c = _client_with({("STREAM", "/stream"): FakeStreamResponse(lines=sse_lines)})
    events = asyncio.run(_drain_stream(c, "s1"))
    assert len(events) == 1
    assert events[0][1] == {"delta": "a"}


def test_stream_session_malformed_data_yields_raw() -> None:
    sse_lines = [
        "event: weird",
        "data: not-json",
        "",
    ]
    c = _client_with({("STREAM", "/stream"): FakeStreamResponse(lines=sse_lines)})
    events = asyncio.run(_drain_stream(c, "s1"))
    assert events[0] == ("weird", {"_raw": "not-json"})


def test_parse_sse_data_valid_json() -> None:
    assert _parse_sse_data('{"x": 1}') == {"x": 1}


def test_parse_sse_data_invalid_json_returns_raw() -> None:
    assert _parse_sse_data("garbage") == {"_raw": "garbage"}


def test_parse_sse_data_empty_returns_empty_dict() -> None:
    assert _parse_sse_data("") == {}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_error_raises_omnigent_error_with_status_and_body() -> None:
    c = _client_with(
        {("GET", "/v1/sessions/s1"): FakeResponse(status_code=500, _body=None, text="boom")}
    )
    with pytest.raises(OmnigentError) as exc_info:
        asyncio.run(c.get_session("s1"))
    assert exc_info.value.status_code == 500
    assert "boom" in exc_info.value.body


def test_stream_error_raises() -> None:
    c = _client_with({("STREAM", "/stream"): FakeStreamResponse(status_code=403, lines=[])})
    with pytest.raises(OmnigentError) as exc_info:
        asyncio.run(_drain_stream(c, "s1"))
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------


def test_auth_token_sets_bearer_header() -> None:
    fake = FakeAsyncClient()
    OmnigentClient("http://x", "secret-token", client=fake)
    assert fake.headers["Authorization"] == "Bearer secret-token"
    assert fake.headers["Accept"] == "application/json"


def test_no_auth_token_leaves_header_unset() -> None:
    fake = FakeAsyncClient()
    OmnigentClient("http://x", None, client=fake)
    assert "Authorization" not in fake.headers


def test_timeout_is_wired_to_async_client() -> None:
    """The timeout arg reaches httpx.AsyncClient, not stored unused."""
    c = OmnigentClient("http://localhost:6767", None, timeout=42.0)
    try:
        # When no client is injected, a real httpx.AsyncClient is built and
        # the timeout must be configured on it (not just stored on self).
        assert c._client.timeout.read == 42.0
        assert c._client.timeout.connect == 42.0
    finally:
        asyncio.run(c.aclose())


def test_session_create_metadata_to_dict_omits_none() -> None:
    meta = SessionCreateMetadata(title="t")
    d = meta.to_dict()
    assert d == {"title": "t"}


# ---------------------------------------------------------------------------
# Omnigent server-URL / workspace-id resolution
#
# The orchestrator derives the Omnigent control-plane URL from the workspace
# it is deployed in, instead of a static ``OMNIGENT_SERVER_URL`` that pins one
# workspace. Env is mocked via ``monkeypatch``; nothing here touches the
# network (the SDK path is monkeypatched out).
# ---------------------------------------------------------------------------


def test_resolve_server_url_explicit_env_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """(a) A non-empty ``OMNIGENT_SERVER_URL`` is returned verbatim — nothing
    is appended — and it wins even when ``DATABRICKS_HOST`` is also set."""
    monkeypatch.setenv("OMNIGENT_SERVER_URL", "http://localhost:6767")
    monkeypatch.setenv("DATABRICKS_HOST", "https://foo.cloud.databricks.com")
    assert resolve_omnigent_server_url() == "http://localhost:6767"


def test_resolve_server_url_derived_from_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """(b) With the env empty, the URL is derived as the REST API base
    ``<host>/api/2.0/omnigent`` (the workspace-hosted API service — NOT the
    ``<host>/omnigent`` browser UI surface, which has no ``/v1/sessions``
    route and 404s)."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://foo.cloud.databricks.com")
    assert (
        resolve_omnigent_server_url()
        == "https://foo.cloud.databricks.com/api/2.0/omnigent"
    )


def test_resolve_server_url_derived_strips_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) A trailing slash on ``DATABRICKS_HOST`` is handled (no double slash)."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://foo.cloud.databricks.com/")
    assert (
        resolve_omnigent_server_url()
        == "https://foo.cloud.databricks.com/api/2.0/omnigent"
    )


def test_resolve_server_url_host_already_has_omnigent_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) A ``DATABRICKS_HOST`` already ending in the UI ``/omnigent`` suffix is
    de-duped (stripped) before the single ``/api/2.0/omnigent`` REST base is
    appended — never ``.../omnigent/api/2.0/omnigent`` (a trailing slash on it
    is fine too)."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    for host in (
        "https://foo.cloud.databricks.com/omnigent",
        "https://foo.cloud.databricks.com/omnigent/",
    ):
        monkeypatch.setenv("DATABRICKS_HOST", host)
        assert (
            resolve_omnigent_server_url()
            == "https://foo.cloud.databricks.com/api/2.0/omnigent"
        )


def test_resolve_server_url_host_has_legacy_api_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(d) A ``DATABRICKS_HOST`` already ending in ``/api/2.0/omnigent`` is
    de-duped (stripped) before the single ``/api/2.0/omnigent`` base is
    appended — never doubled into
    ``.../api/2.0/omnigent/api/2.0/omnigent``."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    monkeypatch.setenv(
        "DATABRICKS_HOST", "https://foo.cloud.databricks.com/api/2.0/omnigent"
    )
    assert (
        resolve_omnigent_server_url()
        == "https://foo.cloud.databricks.com/api/2.0/omnigent"
    )


def test_resolve_server_url_bare_host_gets_https_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) Databricks Apps inject ``DATABRICKS_HOST`` as a BARE hostname with no
    scheme — the derived URL MUST carry an ``https://`` scheme AND target the
    REST API base ``/api/2.0/omnigent`` (the workspace-hosted API service), or
    the request fails: no scheme raises ``UnsupportedProtocol``, and the
    ``/omnigent`` UI surface 404s on ``/v1/sessions``. This is the live prod
    failure this fix targets.

    Regression-bite: reverting the ``/api/2.0/omnigent`` change makes this
    assertion fail (the derived value comes back as
    ``https://fevm-...databricks.com/omnigent``, the UI surface)."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    monkeypatch.setenv(
        "DATABRICKS_HOST", "fevm-stable-classic-7ppxjq.cloud.databricks.com"
    )
    assert resolve_omnigent_server_url() == (
        "https://fevm-stable-classic-7ppxjq.cloud.databricks.com/api/2.0/omnigent"
    )


def test_resolve_server_url_scheme_host_not_doubled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) A ``DATABRICKS_HOST`` that ALREADY carries an ``https://`` (or
    ``http://``) scheme keeps it verbatim — the scheme is never doubled."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    for host, expected in (
        ("https://foo.cloud.databricks.com", "https://foo.cloud.databricks.com/api/2.0/omnigent"),
        ("http://foo.cloud.databricks.com", "http://foo.cloud.databricks.com/api/2.0/omnigent"),
    ):
        monkeypatch.setenv("DATABRICKS_HOST", host)
        assert resolve_omnigent_server_url() == expected


def test_resolve_server_url_uppercase_scheme_not_double_prefixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) URI schemes are case-insensitive (RFC 3986): a ``DATABRICKS_HOST``
    with an UPPERCASE or mixed-case scheme is detected as already-scheme-ful and
    NOT double-prefixed into ``https://HTTPS://...``. The host's ORIGINAL casing
    is preserved verbatim in the returned URL (only the CHECK is lowercased).

    Regression-bite: reverting the ``.lower()`` on the scheme check makes the
    uppercase case fail (it double-prefixes to
    ``https://HTTPS://host/api/2.0/omnigent``)."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    for host, expected in (
        ("HTTPS://foo.cloud.databricks.com", "HTTPS://foo.cloud.databricks.com/api/2.0/omnigent"),
        ("HtTp://foo.cloud.databricks.com", "HtTp://foo.cloud.databricks.com/api/2.0/omnigent"),
    ):
        monkeypatch.setenv("DATABRICKS_HOST", host)
        assert resolve_omnigent_server_url() == expected


def test_resolve_server_url_bare_host_with_suffix_dedup_and_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) A scheme-less host that ALSO carries a trailing ``/omnigent`` or
    ``/api/2.0/omnigent`` suffix is both de-duped (a single ``/api/2.0/omnigent``
    base, no doubling) AND scheme-normalized to ``https://`` — the fixes
    compose."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    for host in (
        "fevm-stable-classic-7ppxjq.cloud.databricks.com/omnigent",
        "fevm-stable-classic-7ppxjq.cloud.databricks.com/api/2.0/omnigent",
    ):
        monkeypatch.setenv("DATABRICKS_HOST", host)
        assert resolve_omnigent_server_url() == (
            "https://fevm-stable-classic-7ppxjq.cloud.databricks.com/api/2.0/omnigent"
        )


def test_resolve_server_url_explicit_override_scheme_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) The explicit ``OMNIGENT_SERVER_URL`` override is returned VERBATIM —
    the scheme-normalization applies only to the DERIVED value, so a local-dev
    ``http://localhost:6767`` is never forced to ``https://`` and nothing is
    appended."""
    monkeypatch.setenv("OMNIGENT_SERVER_URL", "http://localhost:6767")
    monkeypatch.setenv(
        "DATABRICKS_HOST", "fevm-stable-classic-7ppxjq.cloud.databricks.com"
    )
    assert resolve_omnigent_server_url() == "http://localhost:6767"


def test_build_session_url_from_derived_bare_host_is_navigable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) End-to-end: a BARE (scheme-less) ``DATABRICKS_HOST`` flows through
    the resolver → :func:`build_session_url` and yields a valid, scheme-ful
    ``https://.../omnigent/c/<id>`` navigable link (with ``?o=`` when a
    workspace id resolves)."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    monkeypatch.setenv(
        "DATABRICKS_HOST", "fevm-stable-classic-7ppxjq.cloud.databricks.com"
    )
    monkeypatch.setattr(
        "anvil.optimizer.omnigent_client._sdk_workspace_id", lambda: None
    )
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "7474660648944264")
    server_url = resolve_omnigent_server_url()
    wsid = resolve_omnigent_workspace_id()
    url = build_session_url(server_url, "sess-123", wsid)
    assert url == (
        "https://fevm-stable-classic-7ppxjq.cloud.databricks.com"
        "/omnigent/c/sess-123?o=7474660648944264"
    )


def test_derived_api_base_composes_to_ui_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(6) End-to-end COMPOSE proof — the exact string
    :func:`resolve_omnigent_server_url` DERIVES is one
    :func:`build_session_url` can strip back to the correct navigable UI link.

    Unlike ``test_build_session_url_strips_api_suffix`` (which feeds a
    hard-coded base), this chains the resolver's REAL output — so it proves the
    two functions interlock: the derived REST API base
    ``.../api/2.0/omnigent`` round-trips to ``.../omnigent/c/<id>`` and NEVER
    leaks the API suffix into the UI link as ``.../api/2.0/omnigent/c/<id>``.

    Regression-bite: reverting the production ``/api/2.0/omnigent`` append back
    to ``/omnigent`` makes the ``derived ==`` assertion fail (the resolver would
    return the UI surface), while the UI-link assertion still holds — which
    documents the coupling between the two functions.
    """
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    monkeypatch.setenv(
        "DATABRICKS_HOST", "fevm-stable-classic-7ppxjq.cloud.databricks.com"
    )
    derived = resolve_omnigent_server_url()
    assert derived == (
        "https://fevm-stable-classic-7ppxjq.cloud.databricks.com/api/2.0/omnigent"
    )
    assert build_session_url(derived, "sid", "42") == (
        "https://fevm-stable-classic-7ppxjq.cloud.databricks.com/omnigent/c/sid?o=42"
    )


def test_resolve_server_url_empty_env_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty (not just absent) ``OMNIGENT_SERVER_URL`` falls through to
    derivation — deploy configs set it to ``""`` to enable derivation."""
    monkeypatch.setenv("OMNIGENT_SERVER_URL", "")
    monkeypatch.setenv("DATABRICKS_HOST", "https://bar.cloud.databricks.com")
    assert (
        resolve_omnigent_server_url()
        == "https://bar.cloud.databricks.com/api/2.0/omnigent"
    )


def test_resolve_server_url_none_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither source set → ``None`` (Omnigent unconfigured)."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    assert resolve_omnigent_server_url() is None


def test_resolve_workspace_id_sdk_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dynamically-derived (SDK) workspace id is preferred when available."""
    monkeypatch.setattr(
        "anvil.optimizer.omnigent_client._sdk_workspace_id",
        lambda: "7474660648944264",
    )
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "9999999999")
    assert resolve_omnigent_workspace_id() == "7474660648944264"


def test_resolve_workspace_id_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """(c) When the SDK yields nothing, fall back to ``DATABRICKS_WORKSPACE_ID``."""
    monkeypatch.setattr(
        "anvil.optimizer.omnigent_client._sdk_workspace_id", lambda: None
    )
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "7474660648944264")
    assert resolve_omnigent_workspace_id() == "7474660648944264"


def test_resolve_workspace_id_none_when_nothing_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No SDK id and no env → ``None`` (the ``?o=`` param is omitted)."""
    monkeypatch.setattr(
        "anvil.optimizer.omnigent_client._sdk_workspace_id", lambda: None
    )
    monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)
    assert resolve_omnigent_workspace_id() is None


def test_sdk_workspace_id_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_sdk_workspace_id`` is best-effort: any SDK failure yields ``None``."""
    from anvil.optimizer import omnigent_client as mod

    class _Boom:
        def __init__(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("no auth")

    monkeypatch.setitem(
        __import__("sys").modules,
        "databricks.sdk",
        type("m", (), {"WorkspaceClient": _Boom}),
    )
    assert mod._sdk_workspace_id() is None


def test_build_session_url_derived_base_with_o_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) End-to-end: a workspace-derived server base + resolved workspace id
    yield the canonical navigable link (single ``/omnigent``, ``?o=`` set)."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://fevm-stable-classic-7ppxjq.cloud.databricks.com")
    monkeypatch.setattr(
        "anvil.optimizer.omnigent_client._sdk_workspace_id", lambda: None
    )
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "7474660648944264")
    server_url = resolve_omnigent_server_url()
    wsid = resolve_omnigent_workspace_id()
    url = build_session_url(server_url, "sess-123", wsid)
    assert url == (
        "https://fevm-stable-classic-7ppxjq.cloud.databricks.com"
        "/omnigent/c/sess-123?o=7474660648944264"
    )


def test_build_session_url_strips_omnigent_suffix() -> None:
    """A ``<host>/omnigent`` base is not doubled up into ``/omnigent/omnigent``."""
    url = build_session_url("https://foo.cloud.databricks.com/omnigent", "sid")
    assert url == "https://foo.cloud.databricks.com/omnigent/c/sid"


def test_build_session_url_strips_api_suffix() -> None:
    """Cross-check: fed the CURRENT derived REST API base
    ``…/api/2.0/omnigent`` (what :func:`resolve_omnigent_server_url` now
    returns), ``build_session_url`` still recovers the workspace host and
    yields the navigable UI link ``<host>/omnigent/c/<id>`` — NOT
    ``…/api/2.0/omnigent/c/<id>``."""
    url = build_session_url(
        "https://foo.cloud.databricks.com/api/2.0/omnigent", "sid", "42"
    )
    assert url == "https://foo.cloud.databricks.com/omnigent/c/sid?o=42"


# ---------------------------------------------------------------------------
# Authentication precedence (FIX 1)
# ---------------------------------------------------------------------------
#
# The deployed app authenticates to the in-workspace Omnigent server with
# its OWN Databricks identity when no explicit ``OMNIGENT_AUTH_TOKEN`` is
# set. Precedence: an explicit token wins verbatim; otherwise the app's
# Databricks OAuth/SP token is minted fresh per request. An empty/None
# ``Authorization`` header must NEVER be sent.


class _FakeRequest:
    """Minimal stand-in for ``httpx.Request`` — just a mutable headers dict."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}


def test_explicit_token_used_verbatim_as_static_header() -> None:
    """An explicit token is sent verbatim as a static Bearer header."""
    c = OmnigentClient("http://localhost:6767", "explicit-tok", client=FakeAsyncClient({}))
    # Static header set on the (injected) client.
    assert c._client.headers.get("Authorization") == "Bearer explicit-tok"  # type: ignore[attr-defined]
    # And resolved verbatim, without consulting any minter.
    assert c._resolve_token() == "explicit-tok"


def test_explicit_token_never_calls_the_minter() -> None:
    """With an explicit token set, the SP/OAuth minter is never invoked."""

    def _boom() -> str:
        raise AssertionError("token_fn must not be called when an explicit token is set")

    c = OmnigentClient(
        "http://localhost:6767", "explicit-tok", client=FakeAsyncClient({}), token_fn=_boom
    )
    assert c._resolve_token() == "explicit-tok"


def test_empty_token_mints_bearer_from_sp_minter() -> None:
    """No explicit token → the per-request hook injects a freshly minted Bearer."""
    minted: list[int] = []

    def _fake_minter() -> str:
        minted.append(1)
        return "minted-sp-token"

    c = OmnigentClient(
        "http://localhost:6767", None, client=FakeAsyncClient({}), token_fn=_fake_minter
    )
    # No static Authorization header is frozen on the app-identity path.
    assert "Authorization" not in c._client.headers  # type: ignore[attr-defined]

    # The per-request event hook mints and sets the header.
    req = _FakeRequest()
    asyncio.run(c._inject_auth_header(req))  # type: ignore[arg-type]
    assert req.headers["Authorization"] == "Bearer minted-sp-token"
    assert minted, "the SP/OAuth minter was not called"


def test_empty_token_refreshes_per_request() -> None:
    """The bearer is refreshed on EVERY request (tokens expire ~1h)."""
    tokens = iter(["tok-1", "tok-2"])

    c = OmnigentClient(
        "http://localhost:6767", None, client=FakeAsyncClient({}), token_fn=lambda: next(tokens)
    )
    r1 = _FakeRequest()
    r2 = _FakeRequest()
    asyncio.run(c._inject_auth_header(r1))  # type: ignore[arg-type]
    asyncio.run(c._inject_auth_header(r2))  # type: ignore[arg-type]
    assert r1.headers["Authorization"] == "Bearer tok-1"
    assert r2.headers["Authorization"] == "Bearer tok-2"


def test_empty_minted_token_raises_not_silent_noauth() -> None:
    """An empty/None minted token RAISES rather than sending no auth header.

    Sending an unauthenticated request recreates the original failure mode
    (SSO bounce → silently swallowed noop), so an empty resolved token is a
    loud, local error — never a header-less request.
    """
    for bad in ("", "   ", None):
        c = OmnigentClient(
            "http://localhost:6767",
            None,
            client=FakeAsyncClient({}),
            token_fn=lambda b=bad: b,  # type: ignore[return-value]
        )
        with pytest.raises(OmnigentError):
            c._resolve_token()
        req = _FakeRequest()
        with pytest.raises(OmnigentError):
            asyncio.run(c._inject_auth_header(req))  # type: ignore[arg-type]
        # And crucially the header was never set to an empty bearer.
        assert "Authorization" not in req.headers


def test_explicit_but_empty_token_raises_at_construction() -> None:
    """An explicitly-provided but empty/blank token raises immediately.

    ``None`` means "use app identity"; an explicit ``""``/whitespace is a
    misconfiguration, not a request to skip auth.
    """
    for bad in ("", "   "):
        with pytest.raises(OmnigentError):
            OmnigentClient("http://localhost:6767", bad, client=FakeAsyncClient({}))


def test_default_token_fn_reuses_gateway_sp_minter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default minter reuses ``runtime.client._get_fresh_sp_token`` (no duplication)."""
    import anvil.runtime.client as rc

    monkeypatch.setattr(rc, "_get_fresh_sp_token", lambda: "gateway-minted")
    c = OmnigentClient("http://localhost:6767", None, client=FakeAsyncClient({}))
    assert c._resolve_token() == "gateway-minted"


def test_owned_client_registers_auth_hook_on_app_identity_path() -> None:
    """When we own the httpx client and have no explicit token, the hook is wired."""
    c = OmnigentClient("http://localhost:6767", None, token_fn=lambda: "t")
    try:
        hooks = c._client.event_hooks.get("request", [])  # type: ignore[attr-defined]
        assert c._inject_auth_header in hooks
    finally:
        asyncio.run(c.aclose())


def test_owned_client_no_hook_when_explicit_token() -> None:
    """An explicit token uses a static header — no per-request hook needed."""
    c = OmnigentClient("http://localhost:6767", "explicit-tok")
    try:
        hooks = c._client.event_hooks.get("request", [])  # type: ignore[attr-defined]
        assert c._inject_auth_header not in hooks
        assert c._client.headers.get("Authorization") == "Bearer explicit-tok"  # type: ignore[attr-defined]
    finally:
        asyncio.run(c.aclose())


# ---------------------------------------------------------------------------
# Request-level auth (FIX 1) — exercised through ACTUAL client API calls
# ---------------------------------------------------------------------------
#
# The tests above call ``_inject_auth_header`` directly. These drive a real
# ``httpx.AsyncClient`` (via ``httpx.MockTransport``) through real
# ``OmnigentClient`` methods and assert the header that ACTUALLY lands on
# the wire — proving the request event hook fires per real request.


def _mock_client(base_url: str, seen: list[str | None]) -> httpx.AsyncClient:
    """A real httpx.AsyncClient whose transport records each request's
    ``Authorization`` header and returns a canned 200 JSON body."""

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={"id": "s1", "data": []})

    return httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(_handler))


def test_request_mints_fresh_bearer_per_actual_call() -> None:
    """(a) With no explicit token, EACH real request carries a freshly
    minted ``Authorization: Bearer <token>`` and the minter is invoked
    per request — two calls carry two DIFFERENT fresh tokens."""
    seen: list[str | None] = []
    tokens = iter(["tok-1", "tok-2"])
    mint_calls: list[int] = []

    def _minter() -> str:
        mint_calls.append(1)
        return next(tokens)

    http_client = _mock_client("http://localhost:6767", seen)
    c = OmnigentClient("http://localhost:6767", None, client=http_client, token_fn=_minter)
    try:
        asyncio.run(c.get_session("s1"))
        asyncio.run(c.get_session("s1"))
    finally:
        asyncio.run(c.aclose())

    # Both real requests carried their own freshly minted bearer.
    assert seen == ["Bearer tok-1", "Bearer tok-2"]
    # The minter was actually invoked once per request (per-request refresh).
    assert len(mint_calls) == 2


def test_request_carries_explicit_token_verbatim() -> None:
    """(b) With an explicit token, every real request carries it verbatim."""
    seen: list[str | None] = []
    http_client = _mock_client("http://localhost:6767", seen)
    c = OmnigentClient("http://localhost:6767", "explicit-tok", client=http_client)
    try:
        asyncio.run(c.get_session("s1"))
        asyncio.run(c.list_environments("s1"))
    finally:
        asyncio.run(c.aclose())

    assert seen == ["Bearer explicit-tok", "Bearer explicit-tok"]


def test_no_request_is_sent_without_a_bearer() -> None:
    """(c) No real request is ever sent with an empty/absent Authorization
    header when the minter yields a token — every request carries a
    non-empty ``Bearer <token>``."""
    seen: list[str | None] = []
    http_client = _mock_client("http://localhost:6767", seen)
    c = OmnigentClient("http://localhost:6767", None, client=http_client, token_fn=lambda: "live-tok")
    try:
        asyncio.run(c.get_session("s1"))
        asyncio.run(c.list_environments("s1"))
        asyncio.run(c.delete_session("s1"))
    finally:
        asyncio.run(c.aclose())

    assert len(seen) == 3
    for auth in seen:
        assert auth is not None and auth.startswith("Bearer ")
        assert auth.removeprefix("Bearer ").strip() != ""


def test_empty_minted_token_raises_before_reaching_transport() -> None:
    """When the minter returns ""/None, the client RAISES before any request
    reaches the transport — the transport handler is never invoked."""
    for bad in ("", None):
        hit: list[int] = []

        def _handler(request: httpx.Request, _hit: list[int] = hit) -> httpx.Response:
            _hit.append(1)
            raise AssertionError("request reached the transport without a Bearer")

        http_client = httpx.AsyncClient(
            base_url="http://localhost:6767", transport=httpx.MockTransport(_handler)
        )
        c = OmnigentClient(
            "http://localhost:6767",
            None,
            client=http_client,
            token_fn=lambda b=bad: b,  # type: ignore[return-value]
        )
        try:
            with pytest.raises(OmnigentError):
                asyncio.run(c.get_session("s1"))
        finally:
            asyncio.run(http_client.aclose())
        # The transport handler was never called — the request never went out.
        assert hit == []
