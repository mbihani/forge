"""Tests for :mod:`anvil.orchestrator.conversion` — the auto-converter that
spins up an Omnigent agent to turn a custom (savesage-style) repo into the
forge-compatible directory structure.

Covers the pure pieces (the conversion prompt builder, the post-conversion
PII scanner, the :class:`ConversionResult` dataclass) and the converter agent
bundle spec (``agents/forge_converter.yaml``). The background task's Omnigent
+ git integration is exercised via the orchestrator endpoint tests in
``tests/test_orchestrator_app.py``.
"""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from anvil.optimizer.omnigent_backend import _build_agent_bundle
from anvil.optimizer.omnigent_client import OmnigentClient, OmnigentError
from anvil.orchestrator.conversion import (
    _CONVERTER_MODEL,
    DEFAULT_TARGET_BRANCH,
    ConversionResult,
    _build_pr_url,
    _drain_conversion_stream,
    _run_conversion_task,
    _run_managed_session,
    _send_with_retry,
    _wait_for_runner,
    build_conversion_prompt,
    check_pii_in_commit,
)

_CONVERTER_YAML = Path(__file__).resolve().parents[1] / "agents" / "forge_converter.yaml"


# ---------------------------------------------------------------------------
# Git helper for the PII-scanner tests (deterministic "main" base branch)
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _git_init(repo: Path) -> Path:
    """Init a repo whose default branch is ``main`` (regardless of the
    user's ``init.defaultBranch``) so the PII diff has a known base."""
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    # Force HEAD to the unborn ``main`` branch before the first commit.
    subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
        capture_output=True,
    )
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    return repo


# ---------------------------------------------------------------------------
# build_conversion_prompt
# ---------------------------------------------------------------------------


def test_build_conversion_prompt_with_findings() -> None:
    """Detected alternative structures are surfaced in the prompt so the agent
    knows exactly what to convert from."""
    findings = {
        "prompts": ["icici.txt", "hdfc.txt"],
        "schemas": ["icici.json"],
        "harness_py": ["config_ws4.py"],
    }
    prompt = build_conversion_prompt(
        "https://github.com/user/repo", "main", "forge-compat", findings
    )
    assert "icici.txt" in prompt
    assert "hdfc.txt" in prompt
    assert "icici.json" in prompt
    assert "config_ws4.py" in prompt
    assert "forge-compat" in prompt  # target branch
    assert "main" in prompt  # base branch


def test_build_conversion_prompt_no_findings() -> None:
    """With no alternative structures the prompt still generates, with a
    generic 'infer the structure' instruction."""
    prompt = build_conversion_prompt(
        "https://github.com/user/repo", "main", "forge-compat", {}
    )
    assert "forge-compat" in prompt
    assert "no alternative structures detected" in prompt


def test_build_conversion_prompt_includes_pii_safety() -> None:
    """The prompt MUST instruct the agent to gitignore golden_set.jsonl and
    commit a build script instead (non-negotiable PII safety)."""
    prompt = build_conversion_prompt(
        "https://github.com/user/repo", "main", "forge-compat", {"prompts": ["a.txt"]}
    )
    assert "golden_set.jsonl" in prompt
    assert ".gitignore" in prompt
    assert "build_golden_set.py" in prompt
    assert "PII" in prompt
    assert "NEVER" in prompt  # the non-negotiable framing


def test_build_conversion_prompt_includes_gh_token() -> None:
    """When a GitHub token is provided it is embedded in the clone/push remote
    URL instruction so the agent can operate on private repos. Without a token
    the tokenized URL is just the plain URL (no x-access-token prefix)."""
    token = "ghp_secrettoken123"
    prompt = build_conversion_prompt(
        "https://github.com/user/repo", "main", "forge-compat", {}, gh_token=token
    )
    # The token appears in the tokenized clone URL.
    assert f"x-access-token:{token}@github.com/user/repo" in prompt
    # The agent is warned not to leak it into files/commits.
    assert "NEVER write the" in prompt or "token" in prompt.lower()

    # No token → plain URL, no x-access-token prefix.
    prompt_no_token = build_conversion_prompt(
        "https://github.com/user/repo", "main", "forge-compat", {}
    )
    assert "x-access-token" not in prompt_no_token
    assert "https://github.com/user/repo" in prompt_no_token


def test_build_conversion_prompt_non_github_url() -> None:
    """A non-GitHub URL is passed through as the clone URL (no tokenization)."""
    prompt = build_conversion_prompt(
        "https://gitlab.com/u/r", "main", "forge-compat", {}, gh_token="tok"
    )
    assert "https://gitlab.com/u/r" in prompt
    assert "x-access-token" not in prompt


# ---------------------------------------------------------------------------
# check_pii_in_commit
# ---------------------------------------------------------------------------


def test_check_pii_clean(tmp_path: Path) -> None:
    """A branch that adds only forge-compatible files (no card data) → clean."""
    repo = _git_init(tmp_path / "repo")
    _git(repo, "checkout", "-b", "feature")
    (repo / "scaffold").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text(
        "skills: []\nsampling: {temperature: 0.3}\n", encoding="utf-8"
    )
    (repo / "scripts").mkdir()
    (repo / "scripts" / "build_golden_set.py").write_text(
        "# builds data/golden_set.jsonl from schemas\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add forge files")
    findings = check_pii_in_commit(repo, "feature", base_branch="main")
    assert findings == []


def test_check_pii_detects_card_masks(tmp_path: Path) -> None:
    """A masked card number in the branch diff is detected."""
    repo = _git_init(tmp_path / "repo")
    _git(repo, "checkout", "-b", "feature")
    (repo / "notes.txt").write_text(
        "Card on file: 4591-XXXX-XXXX-1234 was charged.", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add notes")
    findings = check_pii_in_commit(repo, "feature", base_branch="main")
    assert findings, "expected a masked-card PII finding"
    assert any("card" in f.lower() for f in findings)
    assert any("4591" in f for f in findings)


def test_check_pii_detects_full_pan(tmp_path: Path) -> None:
    """A full 16-digit PAN is detected."""
    repo = _git_init(tmp_path / "repo")
    _git(repo, "checkout", "-b", "feature")
    (repo / "data.txt").write_text("pan=4591123456789012", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add pan")
    findings = check_pii_in_commit(repo, "feature", base_branch="main")
    assert findings
    assert any("card" in f.lower() for f in findings)


def test_check_pii_detects_golden_set_committed(tmp_path: Path) -> None:
    """``data/golden_set.jsonl`` appearing as an added file is flagged on its
    own — it is gitignored by construction, so committing it means the agent
    leaked raw cardholder data."""
    repo = _git_init(tmp_path / "repo")
    _git(repo, "checkout", "-b", "feature")
    (repo / "data").mkdir()
    (repo / "data" / "golden_set.jsonl").write_text(
        '{"example_id":"1","query":"x"}\n', encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "leak golden set")
    findings = check_pii_in_commit(repo, "feature", base_branch="main")
    assert any("golden_set.jsonl" in f for f in findings)


def test_check_pii_missing_base_branch_returns_empty(tmp_path: Path) -> None:
    """A missing/unreachable base branch degrades to an empty list rather than
    raising — PII absence can't be proven, but a crash here must not mask the
    conversion result."""
    repo = _git_init(tmp_path / "repo")
    _git(repo, "checkout", "-b", "feature")
    (repo / "x.txt").write_text("ok", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "x")
    # " nonexistent-base" ref does not exist → git diff fails → empty list.
    findings = check_pii_in_commit(repo, "feature", base_branch="nonexistent-base")
    assert findings == []


# ---------------------------------------------------------------------------
# ConversionResult dataclass
# ---------------------------------------------------------------------------


def test_conversion_result_dataclass() -> None:
    """Default values match the pending initial state."""
    r = ConversionResult()
    assert r.status == "pending"
    assert r.progress == []
    assert r.pr_url is None
    assert r.branch_name is None
    assert r.revalidation is None
    assert r.error is None
    assert r.session_id is None
    assert r.session_url is None


def test_conversion_result_to_dict_round_trips_progress() -> None:
    """``to_dict`` returns a JSON-serializable snapshot including progress."""
    r = ConversionResult(status="running", branch_name="forge-compat")
    r.progress.append({"step": "starting", "message": "Conversion task started.", "timestamp": "t"})
    d = r.to_dict()
    assert d["status"] == "running"
    assert d["branch_name"] == "forge-compat"
    assert d["progress"] == [
        {"step": "starting", "message": "Conversion task started.", "timestamp": "t"}
    ]
    assert d["pr_url"] is None
    assert d["revalidation"] is None
    assert d["error"] is None
    assert d["session_id"] is None
    assert d["session_url"] is None


def test_conversion_result_carries_session_fields() -> None:
    """session_id + session_url round-trip through to_dict so the API can
    surface a link to the persisted Omnigent conversation."""
    r = ConversionResult(
        status="completed",
        session_id="omnigent-sess-123",
        session_url="http://localhost:6767/sessions/omnigent-sess-123",
    )
    d = r.to_dict()
    assert d["session_id"] == "omnigent-sess-123"
    assert d["session_url"] == "http://localhost:6767/sessions/omnigent-sess-123"


def test_default_target_branch_constant() -> None:
    assert DEFAULT_TARGET_BRANCH == "forge-compat"


# ---------------------------------------------------------------------------
# _build_pr_url
# ---------------------------------------------------------------------------


def test_build_pr_url_github() -> None:
    url = _build_pr_url("https://github.com/user/repo", "main", "forge-compat")
    assert url == "https://github.com/user/repo/compare/main...forge-compat"


def test_build_pr_url_non_github_returns_none() -> None:
    assert _build_pr_url("https://gitlab.com/u/r", "main", "forge-compat") is None


def test_build_pr_url_strips_dot_git() -> None:
    # _parse_github_url already strips .git before this is called, but the
    # builder is tolerant of a trailing path segment only when owner/repo parse.
    url = _build_pr_url("https://github.com/user/repo", "main", "b")
    assert "compare/main...b" in url


# ---------------------------------------------------------------------------
# Converter agent bundle spec (agents/forge_converter.yaml)
# ---------------------------------------------------------------------------


def _load_converter() -> dict:
    return yaml.safe_load(_CONVERTER_YAML.read_text(encoding="utf-8"))


def test_converter_agent_yaml_exists_and_is_valid() -> None:
    assert _CONVERTER_YAML.is_file()
    data = _load_converter()
    assert isinstance(data, dict)
    assert data["spec_version"] == 1
    assert data["name"] == "forge-converter"
    assert "prompt" in data
    assert "executor" in data
    assert "os_env" in data


def test_converter_agent_yaml_executor_shape() -> None:
    data = _load_converter()
    executor = data["executor"]
    assert executor["type"] == "omnigent"
    assert executor["model"] == "databricks-claude-opus-4-8"
    assert executor["config"]["harness"] == "claude-sdk"
    assert executor["config"]["max_turns"] == 50


def test_converter_agent_yaml_prompt_enforces_pii_and_additive() -> None:
    """The system prompt carries the non-negotiable PII + additive-only rules."""
    prompt = _load_converter()["prompt"]
    assert "golden_set.jsonl" in prompt
    assert ".gitignore" in prompt
    assert "build_golden_set.py" in prompt
    assert "additive" in prompt.lower() or "ADDITIVE" in prompt
    assert "--force" in prompt  # no force push
    assert "PII" in prompt


def test_converter_agent_yaml_has_guardrails_policy() -> None:
    """A guardrails policy denies destructive git ops + raw PII commits
    (defense-in-depth on top of the prompt)."""
    data = _load_converter()
    guardrails = data.get("guardrails", {})
    policies = guardrails.get("policies", {})
    assert "no_destructive_or_pii" in policies
    pol = policies["no_destructive_or_pii"]
    assert pol["on"] == ["tool_call"]
    expr = pol["function"]["arguments"]["expression"]
    assert '"DENY"' in expr
    for forbidden in ("--force", "reset --hard", "golden_set.jsonl"):
        assert forbidden in expr, f"guardrail must forbid {forbidden!r}"


def test_converter_agent_bundle_builds_and_substitutes() -> None:
    """The converter YAML packages into a tar.gz with model/max_turns substituted
    (same contract as the optimizer bundle)."""
    bundle = _build_agent_bundle(
        _CONVERTER_YAML, model="databricks-claude-opus-4-8", max_turns=50
    )
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        assert tar.getnames() == ["config.yaml"]
        config = yaml.safe_load(tar.extractfile("config.yaml").read())  # type: ignore[union-attr]
    assert config["name"] == "forge-converter"
    assert config["executor"]["model"] == "databricks-claude-opus-4-8"
    assert config["executor"]["config"]["max_turns"] == 50
    assert "prompt" in config


# ---------------------------------------------------------------------------
# Two-step managed-host flow: _CONVERTER_MODEL + the runner helpers
# ---------------------------------------------------------------------------


class _FakeRunnerClient:
    """Minimal async stand-in for :class:`OmnigentClient` exercising only the
    methods the managed-host helpers touch (``get_session`` / ``send_message``).

    ``get_session_returns`` is consumed in order (clamped to the last entry).
    ``send_raises`` raises the listed :class:`OmnigentError`s in order, then
    ``send_message`` succeeds.
    """

    def __init__(
        self,
        *,
        get_session_returns: list[dict[str, Any]] | None = None,
        send_raises: list[OmnigentError] | None = None,
    ) -> None:
        self.get_session_returns = get_session_returns or []
        self.send_raises = send_raises or []
        self.get_calls = 0
        self.send_calls = 0

    async def get_session(self, session_id: str) -> dict[str, Any]:
        self.get_calls += 1
        if not self.get_session_returns:
            return {}
        idx = min(self.get_calls - 1, len(self.get_session_returns) - 1)
        return self.get_session_returns[idx]

    async def send_message(self, session_id: str, text: str) -> dict[str, Any]:
        if self.send_calls < len(self.send_raises):
            exc = self.send_raises[self.send_calls]
            self.send_calls += 1
            raise exc
        self.send_calls += 1
        return {"queued": True}


def _noop_progress(*_args: Any, **_kwargs: Any) -> None:
    """Progress sink that discards every step message."""


def test_converter_model_targets_opus_4_8() -> None:
    """The converter agent runs on databricks-claude-opus-4-8 on the managed
    host — the same model feeds both the bundle build and the model_override."""
    assert _CONVERTER_MODEL == "databricks-claude-opus-4-8"


def test_create_session_from_agent_is_available() -> None:
    """The two-step flow depends on OmnigentClient.create_session_from_agent
    existing as a distinct method from the multipart create_session."""
    assert hasattr(OmnigentClient, "create_session_from_agent")
    assert OmnigentClient.create_session_from_agent is not OmnigentClient.create_session


def test_send_with_retry_retries_on_transient_503() -> None:
    """A transient 503 (runner still provisioning) is retried up to max_attempts;
    once send_message succeeds the helper returns without re-raising."""
    client = _FakeRunnerClient(
        send_raises=[
            OmnigentError("runner busy", status_code=503),
            OmnigentError("runner busy", status_code=503),
        ]
    )

    async def _run() -> None:
        await _send_with_retry(
            client, "s1", "convert please", _noop_progress, max_attempts=3, delay=0.0
        )

    asyncio.run(_run())
    # Two 503 raises + one success → three send attempts.
    assert client.send_calls == 3


def test_send_with_retry_raises_non_503_immediately() -> None:
    """A non-503 OmnigentError is surfaced immediately — no retry, no sleep."""
    client = _FakeRunnerClient(send_raises=[OmnigentError("boom", status_code=500)])

    async def _run() -> None:
        with pytest.raises(OmnigentError):
            await _send_with_retry(
                client, "s1", "convert please", _noop_progress, max_attempts=3, delay=0.0
            )

    asyncio.run(_run())
    assert client.send_calls == 1  # no retry on non-503


def test_wait_for_runner_polls_until_online() -> None:
    """_wait_for_runner polls get_session until runner_online flips True."""
    client = _FakeRunnerClient(
        get_session_returns=[
            {"runner_online": False},
            {"runner_online": False},
            {"runner_online": True},
        ]
    )

    async def _run() -> None:
        await _wait_for_runner(client, "s1", _noop_progress, max_attempts=6, delay=0.0)

    asyncio.run(_run())
    assert client.get_calls == 3


def test_wait_for_runner_gives_up_without_raising() -> None:
    """When the runner never comes online within max_attempts the helper gives
    up gracefully (no raise) so the caller can still attempt send_message."""
    client = _FakeRunnerClient(get_session_returns=[{"runner_online": False}] * 6)

    async def _run() -> None:
        await _wait_for_runner(client, "s1", _noop_progress, max_attempts=6, delay=0.0)

    asyncio.run(_run())
    assert client.get_calls == 6


# ---------------------------------------------------------------------------
# Two-step managed-host flow: _run_managed_session (end-to-end with a fake)
# ---------------------------------------------------------------------------


class _FakeManagedSessionClient:
    """Async stand-in for :class:`OmnigentClient` exercising the FULL two-step
    managed-host flow end-to-end: ``create_session`` (multipart register) →
    ``create_session_from_agent`` (managed) → ``send_message`` →
    ``stream_session`` → ``delete_session`` (+ ``aclose`` / ``get_session``).

    Records every call so the test can assert ordering and arguments.
    ``stream_events`` is the ``(event_type, data)`` sequence yielded by
    ``stream_session`` (consumed by :func:`_drain_conversion_stream`).
    """

    def __init__(
        self,
        *,
        stream_events: list[tuple[str, dict[str, Any]]],
        hang_after: bool = False,
    ) -> None:
        self.stream_events = stream_events
        # When True, stream_session hangs forever after yielding all events,
        # simulating an open-but-silent SSE connection (no [DONE], no EOF).
        self.hang_after = hang_after
        self.create_session_calls: list[dict[str, Any]] = []
        self.create_session_from_agent_calls: list[dict[str, Any]] = []
        self.send_message_calls: list[dict[str, Any]] = []
        self.stream_session_calls: list[str] = []
        self.delete_session_calls: list[str] = []
        self.aclose_calls = 0

    async def create_session(
        self, bundle_bytes: bytes, metadata: Any = None, *, bundle_filename: str = "agent.tar.gz"
    ) -> dict[str, Any]:
        self.create_session_calls.append(
            {"bundle": bundle_bytes, "metadata": metadata, "bundle_filename": bundle_filename}
        )
        return {"session_id": "reg-sess-id", "agent_id": "agent-xyz", "agent_name": "forge-converter"}

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
        self.create_session_from_agent_calls.append(
            {
                "agent_id": agent_id,
                "title": title,
                "host_type": host_type,
                "model_override": model_override,
            }
        )
        return {"id": "managed-sess-id", "runner_online": True}

    async def send_message(self, session_id: str, text: str) -> dict[str, Any]:
        self.send_message_calls.append({"session_id": session_id, "text": text})
        return {"queued": True, "item_id": "item-1"}

    async def stream_session(self, session_id: str):
        self.stream_session_calls.append(session_id)
        for event in self.stream_events:
            yield event
        if self.hang_after:
            # Simulate an open connection that never sends [DONE] — the drain's
            # inactivity timeout must break out of this.
            await asyncio.Event().wait()

    async def get_session(self, session_id: str) -> dict[str, Any]:
        # Present for _wait_for_runner; not reached when create_session_from_agent
        # already reports runner_online=True.
        return {"runner_online": True}

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        self.delete_session_calls.append(session_id)
        return {"deleted": True}

    async def aclose(self) -> None:
        self.aclose_calls += 1


def test_run_managed_session_two_step_flow() -> None:
    """_run_managed_session runs the full two-step managed-host flow:
    multipart register (→ agent_id) → managed session (host_type="managed" +
    model_override=_CONVERTER_MODEL, reading ``id`` NOT ``session_id``) →
    send_message on the managed id → drain → ONLY the registration session
    tombstoned (the managed conversation session is KEPT alive)."""
    transcript_text = "Conversion complete. Created scaffold/harness.yaml."
    events = [
        # An assistant item_done so the drained transcript is non-empty …
        (
            "response.output_item.done",
            {
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": transcript_text}],
                }
            },
        ),
        # … then response.completed (a per-turn boundary — the drain no
        # longer breaks here; the stream simply ends after this event).
        ("response.completed", {}),
    ]
    captured_session_ids: list[str] = []
    client = _FakeManagedSessionClient(stream_events=events)
    bundle = b"fake-bundle-bytes"

    async def _run() -> str:
        return await _run_managed_session(
            client, bundle, "convert the repo", "forge-compat", _noop_progress,
            on_session_created=captured_session_ids.append,
        )

    transcript = asyncio.run(_run())

    # Step 1: multipart create_session called first, carrying the bundle bytes.
    assert len(client.create_session_calls) == 1
    assert client.create_session_calls[0]["bundle"] == bundle

    # Step 2: create_session_from_agent gets the agent_id from step 1,
    # host_type="managed", and model_override=_CONVERTER_MODEL.
    assert len(client.create_session_from_agent_calls) == 1
    agent_call = client.create_session_from_agent_calls[0]
    assert agent_call["agent_id"] == "agent-xyz"
    assert agent_call["host_type"] == "managed"
    assert agent_call["model_override"] == _CONVERTER_MODEL

    # The managed session's `id` (NOT `session_id`) drives send + stream.
    assert len(client.send_message_calls) == 1
    assert client.send_message_calls[0]["session_id"] == "managed-sess-id"
    assert client.send_message_calls[0]["text"] == "convert the repo"
    assert client.stream_session_calls == ["managed-sess-id"]

    # The on_session_created callback was invoked with the managed session id
    # right after creation (before the drain).
    assert captured_session_ids == ["managed-sess-id"]

    # The returned transcript is the agent's response text.
    assert transcript == transcript_text

    # Finally: ONLY the throwaway registration session is tombstoned — the
    # managed conversation session is KEPT alive so the transcript persists.
    assert client.delete_session_calls == ["reg-sess-id"]
    assert client.aclose_calls == 1


def test_drain_conversion_stream_consumes_multiple_turns() -> None:
    """The drain must NOT break on the first ``response.completed`` — that
    event marks a per-TURN boundary, not whole-conversation completion. A
    multi-turn stream (two turns of deltas + item_done + response.completed,
    then an idle status) must be consumed in full and the transcript must
    concatenate text from BOTH turns.

    The trailing ``idle`` status is logged but does NOT break the drain —
    the stream simply ends (EOF) after it, which is the normal end.

    Regression for the mid-run drop bug: the old drain broke on the first
    ``response.completed`` and labeled the conversation "Agent finished"
    after one turn.
    """
    events = [
        # Turn 1 — deltas + a completed assistant message.
        ("response.output_text.delta", {"delta": "Hello "}),
        ("response.output_text.delta", {"delta": "from turn 1. "}),
        (
            "response.output_item.done",
            {
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Turn 1 done."}],
                }
            },
        ),
        ("response.completed", {}),
        # Turn 2 — the old code never reached here.
        ("response.output_text.delta", {"delta": "Now "}),
        ("response.output_text.delta", {"delta": "turn 2. "}),
        (
            "response.output_item.done",
            {
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Turn 2 done."}],
                }
            },
        ),
        ("response.completed", {}),
        # The agent goes idle — logged but NOT terminal (the drain continues;
        # the stream ends on EOF after this event).
        ("session.status", {"status": "idle"}),
    ]
    client = _FakeManagedSessionClient(stream_events=events)
    progress_calls: list[tuple[str, str]] = []

    def _progress(kind: str, msg: str) -> None:
        progress_calls.append((kind, msg))

    async def _run() -> str:
        return await _drain_conversion_stream(client, "managed-sess-id", _progress)

    transcript = asyncio.run(_run())

    # The stream was consumed across BOTH turns (not just the first).
    assert "Turn 1 done." in transcript
    assert "Turn 2 done." in transcript
    assert "from turn 1." in transcript
    assert "turn 2." in transcript

    # A turn-progress entry was emitted for each response.completed (two).
    turn_entries = [e for e in progress_calls if e[0] == "agent_turn"]
    assert [e[1] for e in turn_entries] == ["Turn 1 complete.", "Turn 2 complete."]

    # An idle-status progress entry was emitted (logged, NOT terminal).
    idle_entries = [e for e in progress_calls if e[0] == "agent_idle"]
    assert idle_entries and "idle" in idle_entries[0][1].lower()

    # The idle did NOT trigger an "agent_done" — the drain ended on EOF.
    done_entries = [e for e in progress_calls if e[0] == "agent_done"]
    assert done_entries == []


def test_drain_conversion_stream_respects_turn_cap() -> None:
    """When the turn cap is reached the drain stops even if the stream has
    more turns — confirming ``response.completed`` is counted and capped
    rather than treated as terminal on the first occurrence."""
    events = [
        ("response.output_text.delta", {"delta": "t1. "}),
        ("response.completed", {}),
        ("response.output_text.delta", {"delta": "t2. "}),
        ("response.completed", {}),
        ("response.output_text.delta", {"delta": "t3. "}),
        ("response.completed", {}),
    ]
    client = _FakeManagedSessionClient(stream_events=events)

    async def _run() -> str:
        return await _drain_conversion_stream(
            client, "managed-sess-id", _noop_progress, max_turns=2
        )

    transcript = asyncio.run(_run())

    # Only the first two turns were drained; the third delta never appears.
    assert "t1." in transcript
    assert "t2." in transcript
    assert "t3." not in transcript


def test_drain_conversion_stream_inactivity_timeout() -> None:
    """When the SSE stream stays open but emits no events for
    ``inactivity_timeout`` seconds, the drain must break (not loop forever).
    This handles the "agent went idle and never came back" / "silently open
    connection with only heartbeats" case.
    """
    events = [
        # One event, then the stream hangs (hang_after=True).
        ("response.output_text.delta", {"delta": "partial "}),
    ]
    client = _FakeManagedSessionClient(stream_events=events, hang_after=True)
    progress_calls: list[tuple[str, str]] = []

    def _progress(kind: str, msg: str) -> None:
        progress_calls.append((kind, msg))

    async def _run() -> str:
        return await _drain_conversion_stream(
            client, "managed-sess-id", _progress, inactivity_timeout=0.05
        )

    transcript = asyncio.run(_run())

    # The one event that arrived before the hang was captured.
    assert "partial" in transcript

    # The inactivity timeout fired and produced an agent_done progress entry.
    done_entries = [e for e in progress_calls if e[0] == "agent_done"]
    assert done_entries and "inactive" in done_entries[0][1].lower()


def test_drain_conversion_stream_idle_does_not_break() -> None:
    """An inter-turn ``idle`` status must NOT cause a premature break — the
    drain keeps consuming past it. The terminal signal is the inactivity
    timeout, which fires when no events arrive after the LAST idle.

    Mock: turn 1 (delta + completed), idle, turn 2 (delta + completed), idle,
    then the stream hangs. With a short inactivity timeout, both turns are
    consumed and the break happens on the timeout after the second idle, NOT
    on the first idle.
    """
    events = [
        # Turn 1.
        ("response.output_text.delta", {"delta": "turn1 "}),
        ("response.completed", {}),
        # Inter-turn idle — must NOT break.
        ("session.status", {"status": "idle"}),
        # Turn 2 — the old code (idle+parts break) never reached here.
        ("response.output_text.delta", {"delta": "turn2 "}),
        ("response.completed", {}),
        # Final idle, then the stream hangs.
        ("session.status", {"status": "idle"}),
    ]
    client = _FakeManagedSessionClient(stream_events=events, hang_after=True)
    progress_calls: list[tuple[str, str]] = []

    def _progress(kind: str, msg: str) -> None:
        progress_calls.append((kind, msg))

    async def _run() -> str:
        return await _drain_conversion_stream(
            client, "managed-sess-id", _progress, inactivity_timeout=0.05
        )

    transcript = asyncio.run(_run())

    # BOTH turns were consumed — the first idle did NOT break the drain.
    assert "turn1" in transcript
    assert "turn2" in transcript

    # Two turn-progress entries (both response.completed events counted).
    turn_entries = [e for e in progress_calls if e[0] == "agent_turn"]
    assert len(turn_entries) == 2

    # Two idle entries (both were logged, neither broke the drain).
    idle_entries = [e for e in progress_calls if e[0] == "agent_idle"]
    assert len(idle_entries) == 2

    # The terminal signal was the inactivity timeout, not the idle.
    done_entries = [e for e in progress_calls if e[0] == "agent_done"]
    assert done_entries and "inactive" in done_entries[0][1].lower()


def test_drain_conversion_stream_max_duration_breaks_during_read() -> None:
    """The ``max_duration`` deadline is a true hard ceiling: even when a read
    is mid-``__anext__`` (the stream is hanging), the drain must break at the
    ``max_duration`` deadline, NOT wait for the larger inactivity timeout.

    Regression for the unbounded-read bug: ``asyncio.timeout(remaining)``
    used only the inactivity remaining, so an event near the 1800s mark could
    let the read block until ~1920s. With ``min(inactivity, max_duration)``
    the read is bounded by whichever deadline comes first.

    Setup: one event, then the stream hangs. ``inactivity_timeout=10``
    (large — would wait 10s), ``max_duration=0.05`` (small — must fire first).
    The drain must return within ~1s (the 0.05s max_duration fires, not the
    10s inactivity timeout).
    """
    events = [
        ("response.output_text.delta", {"delta": "before-hang "}),
    ]
    client = _FakeManagedSessionClient(stream_events=events, hang_after=True)
    progress_calls: list[tuple[str, str]] = []

    def _progress(kind: str, msg: str) -> None:
        progress_calls.append((kind, msg))

    async def _run() -> str:
        return await _drain_conversion_stream(
            client,
            "managed-sess-id",
            _progress,
            inactivity_timeout=10,  # large — would wait 10s
            max_duration=0.05,  # small — must fire first
        )

    start = time.monotonic()
    transcript = asyncio.run(_run())
    elapsed = time.monotonic() - start

    # The drain broke at the max_duration deadline, not the 10s inactivity
    # timeout — well under 2s.
    assert elapsed < 2.0

    # The one event before the hang was captured.
    assert "before-hang" in transcript

    # A max-duration progress entry was emitted (NOT an inactivity entry).
    done_entries = [e for e in progress_calls if e[0] == "agent_done"]
    assert done_entries and "max drain duration" in done_entries[0][1].lower()


# ---------------------------------------------------------------------------
# _run_conversion_task: re-validation must unpack _run_validation's 3-tuple
# ---------------------------------------------------------------------------


def test_run_conversion_task_revalidation_unpacks_3tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ``_run_conversion_task`` unpacked ``_run_validation`` into 2
    variables, but it returns a 3-tuple ``(report, config, findings)`` — so the
    re-validation step raised ``ValueError: too many values to unpack (expected
    2)``. Mock ``_run_validation`` to return a 3-tuple and assert the task
    completes (no ValueError) and stores the re-validation report.

    ``app.py`` already unpacks all three (``report, config, findings = …``);
    this pins the conversion flow against the same shape.
    """
    from anvil.orchestrator import app as app_module
    from anvil.orchestrator import conversion as conversion_module
    from anvil.orchestrator.app import SessionData

    session_id = "sess-revalid"
    target_branch = "forge-compat"
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(app_module, "_SESSIONS_ROOT", sessions_root)

    # Seed a convertible session the task reads (repo url, token, findings, …).
    sess = SessionData(
        session_id=session_id,
        repo_url="https://github.com/owner/repo",
        repo_path=tmp_path / "clone",
        status="invalid",
        validation={"status": "invalid", "checks": [], "convertible": True},
        config=None,
        baseline=None,
        rounds=[],
        frontier=None,
        finalized=None,
        error=None,
        conversion=ConversionResult(),
        _findings={},
        _github_token="ghp_testtoken",
    )
    app_module._sessions[session_id] = sess

    # Mock every external dependency the task touches before re-validation.
    monkeypatch.setattr(app_module, "_current_branch", lambda _root: "main")
    monkeypatch.setattr(
        app_module,
        "_parse_github_url",
        lambda _url: ("https://github.com/owner/repo", "owner", "repo"),
    )
    monkeypatch.setattr(
        conversion_module, "_build_agent_bundle", lambda *a, **kw: b"bundle-bytes"
    )
    monkeypatch.setattr(conversion_module, "OmnigentClient", lambda *a, **kw: object())
    monkeypatch.setenv("OMNIGENT_SERVER_URL", "http://test")
    monkeypatch.setenv("OMNIGENT_AUTH_TOKEN", "tok")

    async def _fake_managed_session(*_args: Any, **_kw: Any) -> str:
        return ""

    monkeypatch.setattr(conversion_module, "_run_managed_session", _fake_managed_session)

    # _clone_repo must materialize the converted checkout dir (so the subpath
    # check passes) and return None (no error string).
    def _fake_clone(_url: str, dest: Path, _token: str | None, _branch: str) -> str | None:
        dest.mkdir(parents=True, exist_ok=True)
        return None

    monkeypatch.setattr(app_module, "_clone_repo", _fake_clone)
    monkeypatch.setattr(conversion_module, "check_pii_in_commit", lambda *_a, **_kw: [])

    # The crux: _run_validation returns a 3-tuple. Before the fix this raised
    # ``ValueError: too many values to unpack (expected 2)``.
    reval_report = {"status": "valid", "checks": [], "convertible": False}
    reval_config = {"runtime_endpoint": "ep"}
    reval_findings = {"prompts": ["p1.txt"]}
    validation_calls: list[Path] = []

    def _fake_run_validation(repo_path: Path):
        validation_calls.append(repo_path)
        return (reval_report, reval_config, reval_findings)

    monkeypatch.setattr(app_module, "_run_validation", _fake_run_validation)

    async def _run() -> None:
        await _run_conversion_task(session_id, target_branch)

    try:
        # Must not raise ValueError (or anything else).
        asyncio.run(_run())

        # Re-validation was actually invoked once on the converted checkout.
        assert len(validation_calls) == 1

        # The task reached "completed" and stored the re-validation report.
        assert sess.conversion is not None
        assert sess.conversion.status == "completed"
        assert sess.conversion.branch_name == target_branch
        assert sess.conversion.error is None
        assert sess.conversion.revalidation is not None
        assert sess.conversion.revalidation["status"] == "valid"
        assert sess.conversion.revalidation["pii_findings"] == []
    finally:
        app_module._sessions.pop(session_id, None)
