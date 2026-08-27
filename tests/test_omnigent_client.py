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

import pytest

from anvil.optimizer.omnigent_client import (
    OmnigentClient,
    OmnigentError,
    SessionCreateMetadata,
    _parse_sse_data,
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


def test_session_create_metadata_to_dict_omits_none() -> None:
    meta = SessionCreateMetadata(title="t")
    d = meta.to_dict()
    assert d == {"title": "t"}
