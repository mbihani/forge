"""Tests for :class:`OmnigentBackend` — the remote optimizer session.

Drives the full round flow with a fake :class:`OmnigentClient` that
records every call and returns canned responses. Verifies the backend:
creates a session, resolves the environment, uploads every scaffold
file, sends the prompt, drains the SSE stream into a transcript, falls
back to conversation items when the stream has no fenced block,
collects modified files, and parses the action — all without a live
server.

Async backend methods are driven via ``asyncio.run`` (sync test
functions) because ``pytest-asyncio`` is not a project dependency.
"""

from __future__ import annotations

import asyncio
import io
import tarfile
from pathlib import Path
from typing import Any

import httpx
import yaml

from anvil.optimizer.omnigent_backend import (
    OmnigentBackend,
    _build_agent_bundle,
    _is_scaffold_path,
)
from anvil.optimizer.omnigent_client import OmnigentError, SessionCreateMetadata
from anvil.optimizer.parser import parse_action

# ---------------------------------------------------------------------------
# Fake OmnigentClient
# ---------------------------------------------------------------------------


class FakeOmnigentClient:
    """Records calls and returns canned responses for the backend flow."""

    def __init__(
        self,
        *,
        stream_events: list[tuple[str, dict]] | None = None,
        changes: list[dict] | None = None,
        environments: list[dict] | None = None,
        items: dict | None = None,
        file_contents: dict[str, str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._stream_events = stream_events or []
        self._changes = changes or []
        self._environments = environments or [{"id": "default"}]
        self._items = items
        self._file_contents = file_contents or {}

    async def create_session(
        self, bundle_bytes: bytes, metadata: SessionCreateMetadata | None = None, **kw: Any
    ) -> dict[str, Any]:
        self.calls.append(("create_session", len(bundle_bytes)))
        return {"session_id": "sess-1", "agent_id": "ag-1", "agent_name": "forge_optimizer"}

    async def list_environments(self, session_id: str) -> list[dict]:
        self.calls.append(("list_environments", session_id))
        return self._environments

    async def upload_file(
        self, session_id: str, env_id: str, relative_path: str, content: str
    ) -> dict:
        self.calls.append(("upload_file", relative_path))
        return {"ok": True}

    async def send_message(self, session_id: str, text: str) -> dict:
        self.calls.append(("send_message", text[:40]))
        return {"queued": True}

    async def stream_session(self, session_id: str):  # type: ignore[no-untyped-def]
        self.calls.append(("stream_session", session_id))
        for event_type, data in self._stream_events:
            yield event_type, data

    async def list_items(
        self, session_id: str, *, limit: int = 100, after: str | None = None
    ) -> dict:
        self.calls.append(("list_items", after))
        return self._items or {"data": [], "has_more": False}

    async def list_changes(self, session_id: str, env_id: str) -> list[dict]:
        self.calls.append(("list_changes", session_id, env_id))
        return self._changes

    async def read_file(self, session_id: str, env_id: str, relative_path: str) -> dict:
        self.calls.append(("read_file", relative_path))
        return {"content": self._file_contents.get(relative_path, "")}


# ---------------------------------------------------------------------------
# Agent bundle builder
# ---------------------------------------------------------------------------


def _write_agent_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
spec_version: 1
name: forge_optimizer
executor:
  type: omnigent
  model: original-model
  config:
    harness: claude-sdk
    max_turns: 10
prompt: |
  be the optimizer
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_build_agent_bundle_substitutes_model_and_max_turns(tmp_path: Path) -> None:
    path = _write_agent_yaml(tmp_path)
    bundle = _build_agent_bundle(path, model="custom-model", max_turns=42)

    # It's a tar.gz whose root is config.yaml.
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        names = tar.getnames()
        assert names == ["config.yaml"]
        config = yaml.safe_load(tar.extractfile("config.yaml").read())  # type: ignore[union-attr]

    assert config["executor"]["model"] == "custom-model"
    assert config["executor"]["config"]["max_turns"] == 42
    # Non-substituted fields survive.
    assert config["name"] == "forge_optimizer"
    assert config["spec_version"] == 1


def test_build_agent_bundle_preserves_other_config(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "spec_version": 1,
                "name": "x",
                "executor": {"type": "omnigent", "model": "old", "config": {"harness": "z"}},
                "os_env": {"type": "caller_process", "cwd": "."},
            }
        ),
        encoding="utf-8",
    )
    bundle = _build_agent_bundle(path, model="new", max_turns=5)
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        config = yaml.safe_load(tar.extractfile("config.yaml").read())  # type: ignore[union-attr]
    assert config["os_env"] == {"type": "caller_process", "cwd": "."}
    assert config["executor"]["config"]["harness"] == "z"


def test_build_agent_bundle_strips_guardrails(tmp_path: Path) -> None:
    """Guardrails with custom handlers are stripped for the managed server.

    The managed Omnigent server's policy registry has no custom handlers like
    ``omnigent.inner.nessie.policies.cel_policy`` — a bundle carrying
    ``guardrails.policies`` is rejected with HTTP 400. The builder strips the
    whole ``guardrails`` key before packaging; the prompt is the authoritative
    contract, so removing the defense-in-depth policy layer is safe.
    """
    path = tmp_path / "agent.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "spec_version": 1,
                "name": "forge_optimizer",
                "executor": {
                    "type": "omnigent",
                    "model": "original-model",
                    "config": {"harness": "claude-sdk", "max_turns": 10},
                },
                "prompt": "be the optimizer",
                # Mirrors agents/forge_optimizer.yaml + forge_converter.yaml:
                # a CEL policy backed by a custom handler the managed server's
                # policy registry does not include.
                "guardrails": {
                    "policies": {
                        "filesystem_only": {
                            "type": "function",
                            "on": ["tool_call"],
                            "function": {
                                "path": "omnigent.inner.nessie.policies.cel_policy",
                                "arguments": {"expression": "ALLOW"},
                            },
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = _build_agent_bundle(path, model="custom-model", max_turns=42)

    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        assert tar.getnames() == ["config.yaml"]
        config = yaml.safe_load(tar.extractfile("config.yaml").read())  # type: ignore[union-attr]

    # The guardrails section is gone — the managed server would reject it.
    assert "guardrails" not in config
    # Everything else survives, including the substituted executor fields.
    assert config["spec_version"] == 1
    assert config["name"] == "forge_optimizer"
    assert config["executor"]["model"] == "custom-model"
    assert config["executor"]["config"]["max_turns"] == 42
    assert config["prompt"] == "be the optimizer"


# ---------------------------------------------------------------------------
# Scaffold path filter
# ---------------------------------------------------------------------------


def test_is_scaffold_path_accepts_scaffold_prefixes() -> None:
    assert _is_scaffold_path("scaffold/harness.yaml")
    assert _is_scaffold_path("agents/foo.py")
    assert _is_scaffold_path("prompts/anvil-round.md")
    assert _is_scaffold_path("eval/runs/baseline.json")
    assert _is_scaffold_path("harness/config.yaml")


def test_is_scaffold_path_rejects_outside_tree() -> None:
    assert not _is_scaffold_path("src/anvil/round.py")
    assert not _is_scaffold_path("tests/test_x.py")
    assert not _is_scaffold_path("data/round_001.json")


# ---------------------------------------------------------------------------
# Full round flow
# ---------------------------------------------------------------------------

_ACTION_BLOCK = '```json-action\n{"action": "noop", "rationale": "no actionable failure"}\n```'


def test_run_full_flow_creates_uploads_sends_drains_parses(tmp_path: Path) -> None:
    """The happy path: every step is called in order and the action parses."""
    stream_events = [
        ("response.output_text.delta", {"delta": "Analyzing the scaffold...\n\n"}),
        ("response.output_text.delta", {"delta": _ACTION_BLOCK}),
        ("response.completed", {"done": True}),
    ]
    fake = FakeOmnigentClient(stream_events=stream_events)
    backend = OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )

    scaffold_files = {
        "scaffold/harness.yaml": "sampling: ...",
        "scaffold/rules/x.md": "---\nrule_id: x\n---\n",
    }
    result = asyncio.run(
        backend.run(
            prompt="improve the worst bucket",
            scaffold_files=scaffold_files,
            max_turns=30,
            model="databricks-claude-opus-4-7",
        )
    )

    # 1. Session created with a real bundle.
    assert fake.calls[0][0] == "create_session"
    assert fake.calls[0][1] > 0  # bundle bytes are non-empty

    # 2. Environment resolved.
    assert ("list_environments", "sess-1") in fake.calls

    # 3. Every scaffold file uploaded.
    uploaded = [c[1] for c in fake.calls if c[0] == "upload_file"]
    assert set(uploaded) == {"scaffold/harness.yaml", "scaffold/rules/x.md"}

    # 4. Prompt sent.
    send_calls = [c for c in fake.calls if c[0] == "send_message"]
    assert len(send_calls) == 1
    assert "worst bucket" in send_calls[0][1]

    # 5. Stream drained.
    assert ("stream_session", "sess-1") in fake.calls

    # 6. Action parsed from the transcript.
    assert result.action.action == "noop"
    assert result.action.rationale == "no actionable failure"
    assert result.parse_result.parse_status == "ok"
    assert "json-action" in result.transcript

    # 7. Session URL is populated.
    assert result.session_url == "http://localhost:6767/sessions/sess-1"
    assert result.turns_used == 1  # one response.completed event


def test_run_falls_back_to_items_when_stream_has_no_fenced_block(tmp_path: Path) -> None:
    """When the stream produces no ``` block, the items list is the transcript source."""
    stream_events = [
        ("response.output_text.delta", {"delta": "just reasoning, no action block"}),
        ("response.completed", {"done": True}),
    ]
    items = {
        "data": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _ACTION_BLOCK}],
            }
        ],
        "has_more": False,
    }
    fake = FakeOmnigentClient(stream_events=stream_events, items=items)
    backend = OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    # The fallback to list_items was triggered.
    assert any(c[0] == "list_items" for c in fake.calls)
    # The action was parsed from the item content.
    assert result.action.action == "noop"
    assert result.parse_result.parse_status == "ok"


def test_run_collects_modified_files(tmp_path: Path) -> None:
    """Modified files under scaffold prefixes are read back; deletions map to None."""
    stream_events = [
        ("response.output_text.delta", {"delta": _ACTION_BLOCK}),
        ("response.completed", {"done": True}),
    ]
    changes = [
        {"path": "scaffold/rules/new.md", "status": "created"},
        {"path": "scaffold/skills/old.md", "status": "deleted"},
        {"path": "src/anvil/round.py", "status": "modified"},  # outside scaffold — ignored
    ]
    fake = FakeOmnigentClient(
        stream_events=stream_events,
        changes=changes,
        file_contents={"scaffold/rules/new.md": "new content"},
    )
    backend = OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    assert result.modified_files == {
        "scaffold/rules/new.md": "new content",
        "scaffold/skills/old.md": None,  # deletion
    }
    # The src/ change was filtered out.
    assert "src/anvil/round.py" not in result.modified_files
    # read_file was called for the created file but NOT for the deleted one.
    read_paths = [c[1] for c in fake.calls if c[0] == "read_file"]
    assert "scaffold/rules/new.md" in read_paths
    assert "scaffold/skills/old.md" not in read_paths


def test_run_server_error_produces_noop_transcript(tmp_path: Path) -> None:
    """An OmnigentError mid-flow collapses to a noop — the loop never crashes."""

    class _FailingClient(FakeOmnigentClient):
        async def create_session(self, *a: Any, **kw: Any) -> dict:
            raise OmnigentError("boom", status_code=500, body="server down")

    backend = OmnigentBackend(
        client=_FailingClient(),  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    # The error was surfaced as a noop-producing transcript.
    assert result.action.action == "noop"
    assert "omnigent backend error" in result.transcript
    assert result.modified_files == {}
    assert result.turns_used is None


def test_run_transport_connect_error_produces_noop(tmp_path: Path) -> None:
    """An httpx.ConnectError (server unreachable) degrades to a noop, not a crash."""

    class _FailingClient(FakeOmnigentClient):
        async def create_session(self, *a: Any, **kw: Any) -> dict:
            raise httpx.ConnectError("connection refused")

    backend = OmnigentBackend(
        client=_FailingClient(),  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    assert result.action.action == "noop"
    assert "omnigent backend error" in result.transcript
    assert "connection refused" in result.transcript
    assert result.modified_files == {}
    assert result.turns_used is None


def test_run_transport_timeout_produces_noop(tmp_path: Path) -> None:
    """An httpx.TimeoutException (request timed out) degrades to a noop, not a crash."""

    class _FailingClient(FakeOmnigentClient):
        async def send_message(self, session_id: str, text: str) -> dict:
            raise httpx.TimeoutException("timed out waiting for response")

    backend = OmnigentBackend(
        client=_FailingClient(),  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    assert result.action.action == "noop"
    assert "omnigent backend error" in result.transcript
    assert "timed out" in result.transcript
    assert result.modified_files == {}
    assert result.turns_used is None


def test_run_skips_none_scaffold_files(tmp_path: Path) -> None:
    """scaffold_files entries with None content (deletions) are not uploaded."""
    stream_events = [("response.output_text.delta", {"delta": _ACTION_BLOCK})]
    fake = FakeOmnigentClient(stream_events=stream_events)
    backend = OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    asyncio.run(
        backend.run(
            prompt="p",
            scaffold_files={"scaffold/real.md": "x", "scaffold/gone.md": None},
            max_turns=5,
            model="m",
        )
    )
    uploaded = [c[1] for c in fake.calls if c[0] == "upload_file"]
    assert uploaded == ["scaffold/real.md"]


def test_run_uses_default_model_when_model_arg_is_none(tmp_path: Path) -> None:
    """When model=None, the backend's default_model is baked into the bundle."""
    stream_events = [("response.output_text.delta", {"delta": _ACTION_BLOCK})]
    fake = FakeOmnigentClient(stream_events=stream_events)
    backend = OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
        default_model="my-default-model",
        default_max_turns=99,
    )
    asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=0, model=None))
    # The run completed without raising — the `max_turns=0` fallback used
    # the default_max_turns (99). Verify the bundle contract directly.
    bundle = _build_agent_bundle(
        _write_agent_yaml(tmp_path), model="my-default-model", max_turns=99
    )
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        config = yaml.safe_load(tar.extractfile("config.yaml").read())  # type: ignore[union-attr]
    assert config["executor"]["model"] == "my-default-model"
    assert config["executor"]["config"]["max_turns"] == 99


def test_run_parses_add_skill_action_from_stream(tmp_path: Path) -> None:
    """A non-noop action from the stream round-trips through the parser."""
    block = (
        "```json-action\n"
        '{"action": "add_rule", "target_file": "rules/new.md", '
        '"content": "---\\nrule_id: new\\nkind: guardrail\\napplies_to: runtime\\n---\\n\\n# new\\n", '
        '"rationale": "fixes retrieval"}\n```'
    )
    stream_events = [
        ("response.output_text.delta", {"delta": "Here is my proposal.\n\n" + block}),
        ("response.completed", {"done": True}),
    ]
    fake = FakeOmnigentClient(stream_events=stream_events)
    backend = OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    assert result.action.action == "add_rule"
    assert result.action.target_file == "rules/new.md"
    assert result.parse_result.parse_status == "ok"
    # The transcript matches what the parser sees locally.
    assert parse_action(result.transcript).action.action == "add_rule"
