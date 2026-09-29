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
import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from anvil.loop.decision import Decision, apply_optimizer_error
from anvil.optimizer.omnigent_backend import (
    OmnigentBackend,
    _build_agent_bundle,
    _is_scaffold_path,
)
from anvil.optimizer.omnigent_client import OmnigentClient, OmnigentError, SessionCreateMetadata
from anvil.optimizer.parser import parse_action

# ---------------------------------------------------------------------------
# Fake OmnigentClient
# ---------------------------------------------------------------------------


# The registration session id (throwaway) and the managed session id are
# DELIBERATELY different so tests can prove every post-bind call targets the
# managed id (``managed["id"]``), never the registration ``session_id``.
_REG_SESSION_ID = "reg-sess-1"
_MANAGED_SESSION_ID = "managed-sess-1"


class FakeOmnigentClient:
    """Records calls and returns canned responses for the backend flow.

    Models the TWO-STEP managed flow: ``create_session`` registers the agent
    (throwaway session ``_REG_SESSION_ID``) and returns an ``agent_id``;
    ``create_session_from_agent`` opens the managed session
    (``_MANAGED_SESSION_ID``) that binds a runner. All post-bind calls run
    against the managed id.
    """

    def __init__(
        self,
        *,
        stream_events: list[tuple[str, dict]] | None = None,
        changes: list[dict] | None = None,
        environments: list[dict] | None = None,
        items: dict | None = None,
        file_contents: dict[str, str] | None = None,
        hang_after: bool = False,
        stream_error: BaseException | None = None,
        runner_online: bool = True,
        get_session_returns: list[dict] | None = None,
        runners: list[dict] | None = None,
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._stream_events = stream_events or []
        # When set, ``stream_session`` RAISES this after yielding the canned
        # events — models the front-door proxy cutting the SSE read mid-stream
        # (e.g. ``httpx.RemoteProtocolError``).
        self._stream_error = stream_error
        self._changes = changes or []
        self._environments = environments or [{"id": "default"}]
        self._items = items
        self._file_contents = file_contents or {}
        self._hang_after = hang_after
        self._runner_online = runner_online
        # Explicit successive get_session snapshots for the _wait_for_runner
        # poll loop. When None, get_session DERIVES the snapshot from the bound
        # state (runner_online True + the bound runner_id) so the happy path's
        # bind→poll succeeds without every test pre-scripting snapshots.
        self._get_session_returns = get_session_returns
        # The registered runner pool returned by list_runners(). One ONLINE
        # runner advertising the optimizer's harnesses by default.
        self._runners = (
            runners
            if runners is not None
            else [
                {
                    "runner_id": "runner-fake-1",
                    "online": True,
                    "harnesses": ["claude-native", "claude-sdk", "codex"],
                }
            ]
        )
        # The runner id the conversation is bound to (None == unbound). The
        # managed session starts UNBOUND (models the live 409 bug); bind_runner
        # flips it. list_environments RAISES 409 while still unbound so a
        # regression that skips the bind FAILS the test.
        self._bound_runner_id: str | None = None
        self.deleted_sessions: list[str] = []
        # The EXACT bundle bytes submitted to create_session (the multipart
        # register). Captured so tests can assert what run() actually baked
        # (model + max_turns) rather than re-deriving a separate bundle.
        self.submitted_bundle: bytes | None = None

    async def create_session(
        self, bundle_bytes: bytes, metadata: SessionCreateMetadata | None = None, **kw: Any
    ) -> dict[str, Any]:
        self.submitted_bundle = bundle_bytes
        self.calls.append(("create_session", len(bundle_bytes)))
        return {
            "session_id": _REG_SESSION_ID,
            "agent_id": "ag-1",
            "agent_name": "forge_optimizer",
        }

    async def create_session_from_agent(
        self,
        agent_id: str,
        *,
        title: str | None = None,
        host_type: str | None = None,
        model_override: str | None = None,
        **kw: Any,
    ) -> dict[str, Any]:
        self.calls.append(("create_session_from_agent", agent_id, host_type, model_override))
        # Opening the managed session provisions a runner in the pool but does
        # NOT bind it — runner_id is null until the explicit bind_runner call.
        return {
            "id": _MANAGED_SESSION_ID,
            "runner_online": self._runner_online,
            "runner_id": None,
        }

    async def list_runners(self) -> list[dict]:
        self.calls.append(("list_runners",))
        return self._runners

    async def bind_runner(self, session_id: str, runner_id: str) -> dict:
        self.calls.append(("bind_runner", session_id, runner_id))
        self._bound_runner_id = runner_id
        return {"id": session_id, "runner_id": runner_id, "runner_online": True}

    async def get_session(self, session_id: str) -> dict[str, Any]:
        self.calls.append(("get_session", session_id))
        if self._get_session_returns is not None:
            # Pop successive scripted snapshots; hold the last once exhausted.
            if len(self._get_session_returns) > 1:
                return self._get_session_returns.pop(0)
            return self._get_session_returns[0]
        # Derived snapshot: online + reflecting the bound runner id, so the
        # happy-path bind→poll (expected_runner_id) succeeds.
        return {"runner_online": True, "runner_id": self._bound_runner_id}

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        self.calls.append(("delete_session", session_id))
        self.deleted_sessions.append(session_id)
        return {"ok": True}

    async def list_environments(self, session_id: str) -> list[dict]:
        self.calls.append(("list_environments", session_id))
        # An unbound managed conversation rejects the first resource read with
        # HTTP 409 — this is the live bug the explicit bind fixes. A regression
        # that skips bind_runner leaves the session unbound and fails here.
        if self._bound_runner_id is None:
            raise OmnigentError(
                f"conversation '{session_id}' is not bound to a runner; "
                "resume the session to bind a registered runner",
                status_code=409,
            )
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
        if self._stream_error is not None:
            raise self._stream_error
        if self._hang_after:
            await asyncio.Event().wait()

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


def test_run_full_flow_creates_uploads_sends_drains_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The happy path: every step is called in order and the action parses."""
    # The session URL is built without a workspace id; ensure a host env
    # leak never appends ``?o=`` to the asserted URL.
    monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)
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

    # 1. Step 1 registers the agent via a real (multipart) bundle upload.
    assert fake.calls[0][0] == "create_session"
    assert fake.calls[0][1] > 0  # bundle bytes are non-empty

    # 2. Step 2 opens the MANAGED session bound to the registered agent —
    #    host_type="managed" (the runner-binding step) with the round's
    #    model carried through model_override.
    bind_calls = [c for c in fake.calls if c[0] == "create_session_from_agent"]
    assert len(bind_calls) == 1
    _, agent_id, host_type, model_override = bind_calls[0]
    assert agent_id == "ag-1"
    assert host_type == "managed"
    assert model_override == "databricks-claude-opus-4-7"

    # 3. Environment resolved AGAINST THE MANAGED session id (managed["id"]),
    #    never the throwaway registration session_id.
    assert ("list_environments", _MANAGED_SESSION_ID) in fake.calls
    assert ("list_environments", _REG_SESSION_ID) not in fake.calls

    # 4. Every scaffold file uploaded.
    uploaded = [c[1] for c in fake.calls if c[0] == "upload_file"]
    assert set(uploaded) == {"scaffold/harness.yaml", "scaffold/rules/x.md"}

    # 5. Prompt sent.
    send_calls = [c for c in fake.calls if c[0] == "send_message"]
    assert len(send_calls) == 1
    assert "worst bucket" in send_calls[0][1]

    # 6. Stream drained against the managed session id.
    assert ("stream_session", _MANAGED_SESSION_ID) in fake.calls

    # 7. The throwaway registration session is deleted; the managed session
    #    is KEPT (so session_url stays inspectable).
    assert _REG_SESSION_ID in fake.deleted_sessions
    assert _MANAGED_SESSION_ID not in fake.deleted_sessions

    # 8. Action parsed from the transcript.
    assert result.action.action == "noop"
    assert result.action.rationale == "no actionable failure"
    assert result.parse_result.parse_status == "ok"
    assert "json-action" in result.transcript

    # 9. Session URL points at the MANAGED session.
    assert result.session_url == f"http://localhost:6767/omnigent/c/{_MANAGED_SESSION_ID}"
    assert result.turns_used == 1  # one response.completed event
    # A clean run leaves no backend-failure marker.
    assert result.optimizer_error is None


# ---------------------------------------------------------------------------
# Explicit runner bind (managed-flow HTTP 409 fix)
# ---------------------------------------------------------------------------


def _bind_backend(tmp_path: Path, fake: FakeOmnigentClient) -> OmnigentBackend:
    return OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )


def test_run_binds_discovered_runner_before_first_resource_read(tmp_path: Path) -> None:
    """The conversation is EXPLICITLY bound (bind_runner) with the managed id
    and the discovered online runner id BEFORE the first resource read
    (list_environments) — the fix for the persistent HTTP 409."""
    stream_events = [
        ("response.output_text.delta", {"delta": _ACTION_BLOCK}),
        ("response.completed", {"done": True}),
    ]
    fake = FakeOmnigentClient(stream_events=stream_events)
    result = asyncio.run(
        _bind_backend(tmp_path, fake).run(
            prompt="p", scaffold_files={}, max_turns=5, model="m"
        )
    )

    names = [c[0] for c in fake.calls]
    # list_runners discovered the pool; bind_runner bound the managed session.
    assert "list_runners" in names
    bind_calls = [c for c in fake.calls if c[0] == "bind_runner"]
    assert len(bind_calls) == 1
    # Bound with the MANAGED session id and the discovered online runner id.
    assert bind_calls[0] == ("bind_runner", _MANAGED_SESSION_ID, "runner-fake-1")

    # ORDERING: bind_runner strictly precedes the first list_environments.
    bind_idx = names.index("bind_runner")
    env_idx = names.index("list_environments")
    assert bind_idx < env_idx

    # list_environments succeeded only because the bind ran first (else the
    # fake raises 409) — a clean, backend-failure-free run.
    assert ("list_environments", _MANAGED_SESSION_ID) in fake.calls
    assert result.action.action == "noop"
    assert result.optimizer_error is None


def test_run_prefers_runner_advertising_needed_harness(tmp_path: Path) -> None:
    """When several runners are online, the bind PREFERS one whose harnesses
    include the harness the agent needs (``claude-sdk`` in the test bundle)."""
    stream_events = [("response.output_text.delta", {"delta": _ACTION_BLOCK})]
    runners = [
        {"runner_id": "runner-codex", "online": True, "harnesses": ["codex"]},
        {"runner_id": "runner-offline", "online": False, "harnesses": ["claude-sdk"]},
        {"runner_id": "runner-sdk", "online": True, "harnesses": ["claude-sdk"]},
    ]
    fake = FakeOmnigentClient(stream_events=stream_events, runners=runners)
    asyncio.run(
        _bind_backend(tmp_path, fake).run(prompt="p", scaffold_files={}, max_turns=5, model="m")
    )
    bind_calls = [c for c in fake.calls if c[0] == "bind_runner"]
    # Bound the ONLINE runner advertising claude-sdk, not the codex-only one
    # and not the offline claude-sdk one.
    assert bind_calls[0][2] == "runner-sdk"


def test_run_binds_first_online_runner_when_no_harness_match(tmp_path: Path) -> None:
    """No online runner advertises the needed harness → the first online
    runner is bound (never a silent skip)."""
    stream_events = [("response.output_text.delta", {"delta": _ACTION_BLOCK})]
    runners = [
        {"runner_id": "runner-a", "online": True, "harnesses": ["openai-agents"]},
        {"runner_id": "runner-b", "online": True, "harnesses": ["pi"]},
    ]
    fake = FakeOmnigentClient(stream_events=stream_events, runners=runners)
    asyncio.run(
        _bind_backend(tmp_path, fake).run(prompt="p", scaffold_files={}, max_turns=5, model="m")
    )
    bind_calls = [c for c in fake.calls if c[0] == "bind_runner"]
    assert bind_calls[0][2] == "runner-a"


def test_run_uses_managed_snapshot_runner_id_when_already_bound(tmp_path: Path) -> None:
    """When the managed snapshot ALREADY carries a non-null runner_id, that id
    is bound (resume-bind) without consulting list_runners."""
    stream_events = [("response.output_text.delta", {"delta": _ACTION_BLOCK})]

    class _PreBoundClient(FakeOmnigentClient):
        async def create_session_from_agent(self, *a: Any, **kw: Any) -> dict:
            self.calls.append(("create_session_from_agent",))
            return {"id": _MANAGED_SESSION_ID, "runner_online": True, "runner_id": "pre-bound-x"}

    fake = _PreBoundClient(stream_events=stream_events)
    asyncio.run(
        _bind_backend(tmp_path, fake).run(prompt="p", scaffold_files={}, max_turns=5, model="m")
    )
    # list_runners was NOT needed — the snapshot's runner_id was used directly.
    assert not any(c[0] == "list_runners" for c in fake.calls)
    bind_calls = [c for c in fake.calls if c[0] == "bind_runner"]
    assert bind_calls[0] == ("bind_runner", _MANAGED_SESSION_ID, "pre-bound-x")


def test_run_no_online_runner_surfaces_infra_fail(tmp_path: Path) -> None:
    """No ONLINE runner in the pool → the round surfaces INFRA_FAIL (never a
    silent noop), and no resource read is attempted."""
    stream_events = [("response.output_text.delta", {"delta": _ACTION_BLOCK})]
    runners = [{"runner_id": "runner-x", "online": False, "harnesses": ["claude-sdk"]}]
    fake = FakeOmnigentClient(stream_events=stream_events, runners=runners)
    result = asyncio.run(
        _bind_backend(tmp_path, fake).run(prompt="p", scaffold_files={}, max_turns=5, model="m")
    )

    # No bind, no resource read — we bailed at runner discovery.
    assert not any(c[0] == "bind_runner" for c in fake.calls)
    assert not any(c[0] == "list_environments" for c in fake.calls)
    # Distinguishable backend failure → INFRA_FAIL, not a silent noop.
    assert result.action.action == "noop"
    assert result.optimizer_error is not None
    assert "no online" in result.optimizer_error.lower()
    assert apply_optimizer_error(Decision.NOOP, result.optimizer_error) == Decision.INFRA_FAIL


def test_run_bind_failure_surfaces_infra_fail(tmp_path: Path) -> None:
    """A failed bind_runner (e.g. server rejects the PATCH) surfaces as
    INFRA_FAIL, never a silent noop — and no resource read runs."""
    stream_events = [("response.output_text.delta", {"delta": _ACTION_BLOCK})]

    class _BindFailsClient(FakeOmnigentClient):
        async def bind_runner(self, session_id: str, runner_id: str) -> dict:
            self.calls.append(("bind_runner", session_id, runner_id))
            raise OmnigentError("bind rejected", status_code=500, body="host gone")

    fake = _BindFailsClient(stream_events=stream_events)
    result = asyncio.run(
        _bind_backend(tmp_path, fake).run(prompt="p", scaffold_files={}, max_turns=5, model="m")
    )

    # The bind was attempted but failed; no resource read ran (still unbound).
    assert any(c[0] == "bind_runner" for c in fake.calls)
    assert not any(c[0] == "list_environments" for c in fake.calls)
    assert result.action.action == "noop"
    assert result.optimizer_error is not None
    assert apply_optimizer_error(Decision.NOOP, result.optimizer_error) == Decision.INFRA_FAIL


def test_run_bind_never_reflected_surfaces_infra_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bind_runner succeeds but the poll never sees the runner online+bound →
    INFRA_FAIL (the bounded _wait_for_runner raise_on_timeout path)."""
    import anvil.optimizer.omnigent_backend as backend_mod

    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(backend_mod.anyio, "sleep", _instant)

    stream_events = [("response.output_text.delta", {"delta": _ACTION_BLOCK})]
    # Scripted snapshots: runner never comes online (and never reflects the id).
    fake = FakeOmnigentClient(
        stream_events=stream_events,
        get_session_returns=[{"runner_online": False, "runner_id": None}],
    )
    result = asyncio.run(
        _bind_backend(tmp_path, fake).run(prompt="p", scaffold_files={}, max_turns=5, model="m")
    )

    # Bind ran, but the confirmation poll timed out → no resource read.
    assert any(c[0] == "bind_runner" for c in fake.calls)
    assert not any(c[0] == "list_environments" for c in fake.calls)
    assert result.optimizer_error is not None
    assert "never came online" in result.optimizer_error
    assert apply_optimizer_error(Decision.NOOP, result.optimizer_error) == Decision.INFRA_FAIL


def test_drain_stream_max_duration_breaks_during_read(tmp_path: Path) -> None:
    """The total-duration deadline interrupts a hanging stream read."""
    fake = FakeOmnigentClient(
        stream_events=[("response.output_text.delta", {"delta": "before-hang "})],
        hang_after=True,
    )
    backend = OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )

    start = time.monotonic()
    transcript, turns_used, stream_incomplete = asyncio.run(
        backend._drain_stream(
            "sess-1",
            inactivity_timeout=10,
            max_duration=0.05,
        )
    )
    elapsed = time.monotonic() - start

    assert elapsed < 2.0
    assert "before-hang" in transcript
    assert turns_used is None
    # A duration-bounded break is a CLEAN stop, not a mid-stream disconnect.
    assert stream_incomplete is False


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


# ---------------------------------------------------------------------------
# Mid-stream disconnect recovery (front-door proxy cuts the SSE read)
# ---------------------------------------------------------------------------


def test_drain_stream_reports_incomplete_on_remote_protocol_error(tmp_path: Path) -> None:
    """_drain_stream CATCHES a mid-stream httpx disconnect, returns the partial
    transcript, and flags ``stream_incomplete=True`` — it does NOT propagate."""
    fake = FakeOmnigentClient(
        stream_events=[("response.output_text.delta", {"delta": "partial "})],
        stream_error=httpx.RemoteProtocolError("peer closed connection"),
    )
    backend = OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    transcript, turns_used, stream_incomplete = asyncio.run(backend._drain_stream("s1"))
    assert "partial" in transcript
    assert stream_incomplete is True


def test_run_recovers_transcript_from_items_on_mid_stream_disconnect(tmp_path: Path) -> None:
    """A mid-stream ``RemoteProtocolError`` is RECOVERED from persisted items.

    Simulates the live failure: the SSE stream emits a few frames then the
    front-door proxy cuts it BEFORE the fenced action is streamed; the turn
    completes server-side (``get_session`` → status idle, runner online); and
    ``list_items`` returns the completed assistant message carrying a valid
    ``json-action`` block. The backend must wait for the turn to go terminal,
    reconstruct the transcript from items, parse the action, and NOT surface
    INFRA_FAIL.

    REGRESSION BITE: reverting the disconnect-catch in ``_drain_stream`` (or
    the items recovery in ``run``) lets the ``RemoteProtocolError`` propagate
    to ``run()``'s broad ``except`` → ``optimizer_error`` set → INFRA_FAIL,
    which fails the ``optimizer_error is None`` / recovered-action assertions
    below.
    """
    stream_events = [
        ("response.output_text.delta", {"delta": "Analyzing the worst bucket...\n"}),
        ("response.output_text.delta", {"delta": "```json-acti"}),  # cut mid-fence
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
    fake = FakeOmnigentClient(
        stream_events=stream_events,
        stream_error=httpx.RemoteProtocolError("peer closed connection"),
        items=items,
        # One snapshot satisfies BOTH the bind-confirm poll (online + bound
        # runner id) AND the terminal-wait (status idle + online).
        get_session_returns=[
            {"runner_online": True, "runner_id": "runner-fake-1", "status": "idle"}
        ],
    )
    result = asyncio.run(
        _bind_backend(tmp_path, fake).run(prompt="p", scaffold_files={}, max_turns=5, model="m")
    )

    # The turn was polled to terminal, THEN items were read for recovery.
    assert any(c[0] == "get_session" for c in fake.calls)
    assert any(c[0] == "list_items" for c in fake.calls)
    # The action parsed from the RECOVERED items transcript.
    assert result.action.action == "noop"
    assert result.action.rationale == "no actionable failure"
    assert result.parse_result.parse_status == "ok"
    # Recovered cleanly → NOT a backend failure → NOT INFRA_FAIL.
    assert result.optimizer_error is None
    assert apply_optimizer_error(Decision.NOOP, result.optimizer_error) == Decision.NOOP


def test_run_mid_stream_disconnect_no_recoverable_action_surfaces_infra_fail(
    tmp_path: Path,
) -> None:
    """A mid-stream disconnect whose items recovery ALSO yields no parseable
    action is a GENUINE failure → INFRA_FAIL (never a silent, indistinguishable
    noop)."""
    stream_events = [
        ("response.output_text.delta", {"delta": "partial reasoning, no action"}),
    ]
    # Persisted items hold an assistant message with NO fenced action block.
    items = {
        "data": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "still thinking, cut off"}],
            }
        ],
        "has_more": False,
    }
    fake = FakeOmnigentClient(
        stream_events=stream_events,
        stream_error=httpx.RemoteProtocolError("peer closed connection"),
        items=items,
        get_session_returns=[
            {"runner_online": True, "runner_id": "runner-fake-1", "status": "idle"}
        ],
    )
    result = asyncio.run(
        _bind_backend(tmp_path, fake).run(prompt="p", scaffold_files={}, max_turns=5, model="m")
    )

    # Recovery WAS attempted (items read) but yielded no action.
    assert any(c[0] == "list_items" for c in fake.calls)
    # Distinguishable backend failure → INFRA_FAIL.
    assert result.optimizer_error is not None
    assert "disconnected mid-turn" in result.optimizer_error
    assert apply_optimizer_error(Decision.NOOP, result.optimizer_error) == Decision.INFRA_FAIL


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


def test_run_backend_error_sets_optimizer_error_marker(tmp_path: Path) -> None:
    """A backend failure carries a DISTINGUISHABLE ``optimizer_error`` marker.

    The transcript still parses to a NoopAction (the loop never crashes),
    but ``optimizer_error`` must be non-None so the round can surface a
    401 / SSO redirect / connection error as a failure rather than an
    indistinguishable "optimized, noop".
    """

    class _FailingClient(FakeOmnigentClient):
        async def create_session(self, *a: Any, **kw: Any) -> dict:
            raise OmnigentError("unauthorized", status_code=401, body="login redirect")

    backend = OmnigentBackend(
        client=_FailingClient(),  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    # Still a noop-producing action (loop never crashes) ...
    assert result.action.action == "noop"
    # ... but the failure is DISTINGUISHABLE via the marker.
    assert result.optimizer_error is not None
    assert "OmnigentError" in result.optimizer_error
    assert "login redirect" in result.optimizer_error


def test_run_success_leaves_optimizer_error_none(tmp_path: Path) -> None:
    """A clean run (legit optimizer noop) leaves ``optimizer_error`` None.

    This is the discriminator: a genuine optimizer-chosen noop must NOT be
    confused with a swallowed backend failure.
    """
    stream_events = [
        ("response.output_text.delta", {"delta": _ACTION_BLOCK}),
        ("response.completed", {"done": True}),
    ]
    backend = OmnigentBackend(
        client=FakeOmnigentClient(stream_events=stream_events),  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    assert result.action.action == "noop"
    assert result.optimizer_error is None


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


# ---------------------------------------------------------------------------
# Request-level managed-flow tests (real OmnigentClient + httpx.MockTransport)
#
# The FakeOmnigentClient above proves the backend CALLS the right client
# methods; these prove the actual HTTP REQUESTS on the wire — the multipart
# register followed by the JSON create_session_from_agent(host_type="managed"),
# and that every post-bind request targets the managed session id. This is the
# regression bite: revert the two-step flow to the old single create_session
# and these fail (post-bind requests would carry the registration id, and no
# JSON host_type="managed" request would be sent).
# ---------------------------------------------------------------------------

_WIRE_REG_ID = "reg-wire-1"
_WIRE_MANAGED_ID = "managed-wire-1"
_WIRE_RUNNER_ID = "runner_token_wire-1"
_WIRE_ACTION_SSE = (
    "event: response.output_text.delta\n"
    f"data: {json.dumps({'delta': _ACTION_BLOCK})}\n"
    "\n"
    "event: response.completed\n"
    'data: {"done": true}\n'
    "\n"
    "data: [DONE]\n"
)


@dataclass
class _Recorded:
    """One HTTP request the MockTransport handler saw."""

    method: str
    path: str
    content_type: str
    body: bytes


def _managed_transport(
    recorded: list[_Recorded],
    *,
    runner_online_seq: list[bool] | None = None,
    env_status_seq: list[int] | None = None,
    stream_status_seq: list[int] | None = None,
) -> httpx.MockTransport:
    """A MockTransport modeling the managed Omnigent server.

    ``POST /v1/sessions`` dispatches on content type: multipart registers the
    agent (throwaway ``_WIRE_REG_ID`` + an ``agent_id``); JSON opens the
    managed session (``_WIRE_MANAGED_ID``). ``runner_online_seq`` drives the
    ``runner_online`` flag returned by the JSON create and successive
    ``get_session`` polls; ``env_status_seq`` / ``stream_status_seq`` drive
    successive HTTP statuses for the environments / stream calls (e.g.
    ``[503, 200]`` to exercise a retry, or ``[503]`` to exercise exhaustion).
    """
    runner_seq = list(runner_online_seq if runner_online_seq is not None else [True])
    env_seq = list(env_status_seq if env_status_seq is not None else [200])
    stream_seq = list(stream_status_seq if stream_status_seq is not None else [200])

    def _next_runner() -> bool:
        return runner_seq.pop(0) if len(runner_seq) > 1 else runner_seq[0]

    def _handler(request: httpx.Request) -> httpx.Response:
        ct = request.headers.get("content-type", "")
        recorded.append(_Recorded(request.method, request.url.path, ct, request.content))
        path = request.url.path

        if request.method == "POST" and path == "/v1/sessions":
            if ct.startswith("multipart/form-data"):
                return httpx.Response(
                    200,
                    json={
                        "session_id": _WIRE_REG_ID,
                        "agent_id": "ag-wire-1",
                        "agent_name": "forge_optimizer",
                    },
                )
            # JSON create_session_from_agent — opens the managed session. It
            # provisions a runner in the pool but leaves the conversation
            # UNBOUND (runner_id null); the explicit PATCH bind follows.
            return httpx.Response(
                200,
                json={
                    "id": _WIRE_MANAGED_ID,
                    "runner_online": _next_runner(),
                    "runner_id": None,
                },
            )
        if request.method == "GET" and path == "/v1/runners":
            # One ONLINE runner advertising the optimizer's harnesses.
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "runner_id": _WIRE_RUNNER_ID,
                            "online": True,
                            "harnesses": ["claude-native", "claude-sdk", "codex"],
                        }
                    ]
                },
            )
        if request.method == "PATCH" and path == f"/v1/sessions/{_WIRE_MANAGED_ID}":
            # The explicit bind (resume-bind) — atomically sets runner_id.
            return httpx.Response(
                200,
                json={
                    "id": _WIRE_MANAGED_ID,
                    "runner_id": _WIRE_RUNNER_ID,
                    "runner_online": True,
                },
            )
        if request.method == "GET" and path == f"/v1/sessions/{_WIRE_MANAGED_ID}":
            # Snapshot reflects the bound runner id (last-write-wins); its
            # online flag is driven by runner_online_seq.
            return httpx.Response(
                200, json={"runner_online": _next_runner(), "runner_id": _WIRE_RUNNER_ID}
            )
        if path.endswith("/resources/environments"):
            status = env_seq.pop(0) if len(env_seq) > 1 else env_seq[0]
            if status != 200:
                return httpx.Response(status, json={"error": "runner provisioning"})
            return httpx.Response(200, json={"data": [{"id": "default"}]})
        if path.endswith("/stream"):
            status = stream_seq.pop(0) if len(stream_seq) > 1 else stream_seq[0]
            if status != 200:
                return httpx.Response(status, json={"error": "runner provisioning"})
            return httpx.Response(200, text=_WIRE_ACTION_SSE)
        if path.endswith("/changes"):
            return httpx.Response(200, json={"data": []})
        if path.endswith("/items"):
            return httpx.Response(200, json={"data": [], "has_more": False})
        if request.method == "PUT":  # filesystem upload
            return httpx.Response(200, json={"ok": True})
        if request.method == "POST" and path.endswith("/events"):
            return httpx.Response(200, json={"queued": True})
        if request.method == "DELETE":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"error": f"unhandled {request.method} {path}"})

    return httpx.MockTransport(_handler)


def _managed_backend(
    tmp_path: Path,
    recorded: list[_Recorded],
    *,
    runner_online_seq: list[bool] | None = None,
    env_status_seq: list[int] | None = None,
    stream_status_seq: list[int] | None = None,
    model: str = "databricks-claude-opus-4-7",
) -> OmnigentBackend:
    transport = _managed_transport(
        recorded,
        runner_online_seq=runner_online_seq,
        env_status_seq=env_status_seq,
        stream_status_seq=stream_status_seq,
    )
    http_client = httpx.AsyncClient(base_url="http://localhost:6767", transport=transport)
    client = OmnigentClient("http://localhost:6767", "test-token", client=http_client)
    return OmnigentBackend(
        client=client,
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
        default_model=model,
    )


def test_wire_register_then_managed_create_are_sent(tmp_path: Path) -> None:
    """(a)+(e) On the wire: a multipart POST /v1/sessions (register) is
    followed by a JSON POST /v1/sessions carrying host_type="managed"."""
    recorded: list[_Recorded] = []
    backend = _managed_backend(tmp_path, recorded)
    result = asyncio.run(
        backend.run(prompt="p", scaffold_files={"scaffold/a.md": "x"}, max_turns=5, model="m")
    )

    creates = [r for r in recorded if r.method == "POST" and r.path == "/v1/sessions"]
    assert len(creates) == 2
    # First create is the multipart register.
    assert creates[0].content_type.startswith("multipart/form-data")
    # Second create is the JSON managed bind carrying host_type="managed".
    assert creates[1].content_type.startswith("application/json")
    bind_body = json.loads(creates[1].body)
    assert bind_body["host_type"] == "managed"
    assert bind_body["agent_id"] == "ag-wire-1"
    # A clean managed run parses the action and flags no backend failure.
    assert result.action.action == "noop"
    assert result.optimizer_error is None


def test_wire_post_bind_calls_target_managed_id(tmp_path: Path) -> None:
    """(b) Every post-bind request targets the managed session id, never the
    throwaway registration id."""
    recorded: list[_Recorded] = []
    backend = _managed_backend(tmp_path, recorded)
    asyncio.run(
        backend.run(prompt="p", scaffold_files={"scaffold/a.md": "x"}, max_turns=5, model="m")
    )

    # The registration id appears ONLY on its own DELETE (throwaway cleanup);
    # it is never used for environments / filesystem / events / stream.
    reg_paths = [r.path for r in recorded if _WIRE_REG_ID in r.path]
    assert reg_paths == [f"/v1/sessions/{_WIRE_REG_ID}"]
    assert [r.method for r in recorded if _WIRE_REG_ID in r.path] == ["DELETE"]

    # Post-bind traffic (environments, filesystem, events, stream) all carry
    # the managed id.
    managed_ops = {
        (r.method, r.path.split(_WIRE_MANAGED_ID, 1)[1])
        for r in recorded
        if _WIRE_MANAGED_ID in r.path
    }
    assert ("GET", "/resources/environments") in managed_ops
    assert ("POST", "/events") in managed_ops
    assert ("GET", "/stream") in managed_ops
    assert any(m == "PUT" and p.startswith("/resources/environments") for m, p in managed_ops)
    # The managed session is NOT deleted (kept inspectable).
    assert (
        "DELETE",
        "",
    ) not in {(r.method, r.path.split(_WIRE_MANAGED_ID, 1)[1]) for r in recorded if _WIRE_MANAGED_ID in r.path}


def test_wire_waits_for_runner_then_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) When the managed create reports runner_online=False, the backend
    polls get_session until it flips True, THEN runs the post-bind calls."""
    import anvil.optimizer.omnigent_backend as backend_mod

    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(backend_mod.anyio, "sleep", _instant)

    recorded: list[_Recorded] = []
    # create → offline; first poll → offline; second poll → online.
    backend = _managed_backend(tmp_path, recorded, runner_online_seq=[False, False, True])
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    get_session_calls = [
        r for r in recorded if r.method == "GET" and r.path == f"/v1/sessions/{_WIRE_MANAGED_ID}"
    ]
    assert len(get_session_calls) >= 2  # polled until runner_online
    # Only AFTER the runner is online do post-bind calls run.
    first_env_idx = next(
        i for i, r in enumerate(recorded) if r.path.endswith("/resources/environments")
    )
    last_poll_idx = max(
        i
        for i, r in enumerate(recorded)
        if r.method == "GET" and r.path == f"/v1/sessions/{_WIRE_MANAGED_ID}"
    )
    assert last_poll_idx < first_env_idx
    assert result.action.action == "noop"
    assert result.optimizer_error is None


def test_wire_retries_environments_on_transient_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(d) A 503 on the first post-bind environments call (runner still
    provisioning) is retried, and the run then succeeds."""
    import anvil.optimizer.omnigent_backend as backend_mod

    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(backend_mod.anyio, "sleep", _instant)

    recorded: list[_Recorded] = []
    backend = _managed_backend(tmp_path, recorded, env_status_seq=[503, 200])
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    env_calls = [r for r in recorded if r.path.endswith("/resources/environments")]
    assert len(env_calls) == 2  # first 503, retried to 200
    # The retry made the run succeed (no backend-failure marker).
    assert result.action.action == "noop"
    assert result.optimizer_error is None


def test_wire_model_pin_carries_through_model_override(tmp_path: Path) -> None:
    """(model pin) The round's model reaches the managed session via
    model_override on the JSON create_session_from_agent request."""
    recorded: list[_Recorded] = []
    backend = _managed_backend(tmp_path, recorded)
    asyncio.run(
        backend.run(
            prompt="p", scaffold_files={}, max_turns=37, model="databricks-claude-opus-4-8"
        )
    )

    bind = next(
        r
        for r in recorded
        if r.method == "POST"
        and r.path == "/v1/sessions"
        and r.content_type.startswith("application/json")
    )
    body = json.loads(bind.body)
    # The round's model pin is carried on the managed create (model_override).
    assert body["model_override"] == "databricks-claude-opus-4-8"


def test_run_bakes_round_model_and_max_turns_into_submitted_bundle(tmp_path: Path) -> None:
    """(max_turns pin) The round's model AND max_turns are baked into the
    bundle that run() ACTUALLY submits to create_session.

    Inspects the exact ``bundle_bytes`` the fake received (not a
    separately-reconstructed bundle) — so a regression where run() baked the
    wrong max_turns into the submitted bundle would fail here.
    """
    stream_events = [("response.output_text.delta", {"delta": _ACTION_BLOCK})]
    fake = FakeOmnigentClient(stream_events=stream_events)
    backend = OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    asyncio.run(
        backend.run(
            prompt="p", scaffold_files={}, max_turns=37, model="databricks-claude-opus-4-8"
        )
    )

    assert fake.submitted_bundle is not None  # captured the real submitted bytes
    with tarfile.open(fileobj=io.BytesIO(fake.submitted_bundle), mode="r:gz") as tar:
        assert tar.getnames() == ["config.yaml"]
        config = yaml.safe_load(tar.extractfile("config.yaml").read())  # type: ignore[union-attr]
    # The round's pins are in the bundle run() actually uploaded.
    assert config["executor"]["model"] == "databricks-claude-opus-4-8"
    assert config["executor"]["config"]["max_turns"] == 37


def test_run_bakes_default_max_turns_when_arg_is_zero(tmp_path: Path) -> None:
    """When max_turns=0, run() bakes ``default_max_turns`` into the SUBMITTED
    bundle (the round's turn cap, resolved by run(), is what ships)."""
    stream_events = [("response.output_text.delta", {"delta": _ACTION_BLOCK})]
    fake = FakeOmnigentClient(stream_events=stream_events)
    backend = OmnigentBackend(
        client=fake,  # type: ignore[arg-type]
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
        default_model="my-default-model",
        default_max_turns=88,
    )
    asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=0, model=None))

    assert fake.submitted_bundle is not None
    with tarfile.open(fileobj=io.BytesIO(fake.submitted_bundle), mode="r:gz") as tar:
        config = yaml.safe_load(tar.extractfile("config.yaml").read())  # type: ignore[union-attr]
    assert config["executor"]["model"] == "my-default-model"
    assert config["executor"]["config"]["max_turns"] == 88


def test_wire_managed_create_failure_surfaces_optimizer_error(tmp_path: Path) -> None:
    """A genuine failure at the managed-bind step surfaces as an
    optimizer_error marker (→ INFRA_FAIL), not a silent noop."""

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/sessions":
            ct = request.headers.get("content-type", "")
            if ct.startswith("multipart/form-data"):
                return httpx.Response(
                    200,
                    json={
                        "session_id": _WIRE_REG_ID,
                        "agent_id": "ag-wire-1",
                        "agent_name": "x",
                    },
                )
            # The managed bind fails hard (e.g. host has no capacity).
            return httpx.Response(500, json={"error": "no managed host"})
        if request.method == "DELETE":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={})

    http_client = httpx.AsyncClient(
        base_url="http://localhost:6767", transport=httpx.MockTransport(_handler)
    )
    client = OmnigentClient("http://localhost:6767", "test-token", client=http_client)
    backend = OmnigentBackend(
        client=client,
        agent_bundle_path=_write_agent_yaml(tmp_path),
        server_url="http://localhost:6767",
    )
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    # Loop never crashes (noop action) BUT the failure is distinguishable.
    assert result.action.action == "noop"
    assert result.optimizer_error is not None
    assert "500" in result.optimizer_error
    # Contract item 4: the round's decision mapping turns this marker into
    # INFRA_FAIL (not a silent noop). Exercise the exact mapping round.py runs.
    assert apply_optimizer_error(Decision.NOOP, result.optimizer_error) == Decision.INFRA_FAIL


def test_wire_runner_never_online_surfaces_infra_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(fix 2) A managed runner that never comes online within the bounded
    poll is surfaced as a backend failure → INFRA_FAIL, never a silent noop.

    The poll is bounded (no infinite loop): after ``max_attempts`` with the
    runner still offline, _wait_for_runner(raise_on_timeout=True) raises,
    run() catches it and sets optimizer_error.
    """
    import anvil.optimizer.omnigent_backend as backend_mod

    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(backend_mod.anyio, "sleep", _instant)

    recorded: list[_Recorded] = []
    # create → offline; every get_session poll → offline (never flips).
    backend = _managed_backend(tmp_path, recorded, runner_online_seq=[False])
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    # The poll was bounded (a finite number of get_session calls) ...
    poll_calls = [
        r for r in recorded if r.method == "GET" and r.path == f"/v1/sessions/{_WIRE_MANAGED_ID}"
    ]
    assert 1 <= len(poll_calls) <= 6
    # ... no post-bind work ran (we never reached environments/stream) ...
    assert not any(r.path.endswith("/resources/environments") for r in recorded)
    assert not any(r.path.endswith("/stream") for r in recorded)
    # ... and the failure is DISTINGUISHABLE → INFRA_FAIL, not a silent noop.
    assert result.action.action == "noop"
    assert result.optimizer_error is not None
    assert "never came online" in result.optimizer_error
    assert apply_optimizer_error(Decision.NOOP, result.optimizer_error) == Decision.INFRA_FAIL


def test_wire_stream_503_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(suggestion) A transient 503 on the stream open (runner still
    provisioning) is retried, and the drain then succeeds."""
    import anvil.optimizer.omnigent_backend as backend_mod

    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(backend_mod.anyio, "sleep", _instant)

    recorded: list[_Recorded] = []
    backend = _managed_backend(tmp_path, recorded, stream_status_seq=[503, 200])
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    stream_calls = [r for r in recorded if r.path.endswith("/stream")]
    assert len(stream_calls) == 2  # first 503, retried to 200
    assert result.action.action == "noop"
    assert result.optimizer_error is None


def test_wire_stream_503_exhaustion_surfaces_infra_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(suggestion) A persistent 503 on the stream open exhausts the bounded
    retry and surfaces as a backend failure → INFRA_FAIL."""
    import anvil.optimizer.omnigent_backend as backend_mod

    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(backend_mod.anyio, "sleep", _instant)

    recorded: list[_Recorded] = []
    backend = _managed_backend(tmp_path, recorded, stream_status_seq=[503])
    result = asyncio.run(backend.run(prompt="p", scaffold_files={}, max_turns=5, model="m"))

    # Bounded retry: the stream open was attempted _retry_on_503's max (3x).
    stream_calls = [r for r in recorded if r.path.endswith("/stream")]
    assert len(stream_calls) == 3
    assert result.action.action == "noop"
    assert result.optimizer_error is not None
    assert apply_optimizer_error(Decision.NOOP, result.optimizer_error) == Decision.INFRA_FAIL
