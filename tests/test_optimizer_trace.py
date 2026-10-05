"""Optimizer session → MLflow trace replay, session log, and run cleanup.

Offline: SDK messages are SimpleNamespace stand-ins named like the real
``claude-agent-sdk`` types, traces go to a throwaway local SQLite MLflow
store, and the MLflow client used by the orphan reaper is a fake.
"""

from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path
from types import ModuleType, SimpleNamespace

import mlflow
import pytest

from anvil.loop.optimizer_artifacts import reap_orphaned_cli_sessions
from anvil.loop.optimizer_trace import log_optimizer_trace, render_session_markdown
from anvil.optimizer.session import message_events


def _msg(cls_name: str, **attrs):
    return type(cls_name, (), {})() if not attrs else _obj(cls_name, **attrs)


def _obj(cls_name: str, **attrs):
    obj = type(cls_name, (), {})()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _session_messages():
    return [
        _obj(
            "AssistantMessage",
            content=[
                _obj("ThinkingBlock", thinking="look at the baseline first", signature="s"),
                _obj("TextBlock", text="Reading the scaffold."),
                _obj("ToolUseBlock", id="t1", name="Read", input={"file_path": "scaffold/a.md"}),
            ],
        ),
        _obj(
            "UserMessage",
            content=[_obj("ToolResultBlock", tool_use_id="t1", content="file body", is_error=None)],
        ),
        _obj(
            "AssistantMessage",
            content=[_obj("ToolUseBlock", id="t2", name="Bash", input={"command": "false"})],
        ),
        _obj(
            "UserMessage",
            content=[
                _obj(
                    "ToolResultBlock",
                    tool_use_id="t2",
                    content=[{"type": "text", "text": "exit 1"}],
                    is_error=True,
                )
            ],
        ),
        _obj(
            "ResultMessage",
            subtype="success",
            is_error=False,
            num_turns=3,
            duration_ms=4200,
            total_cost_usd=0.12,
            usage={"input_tokens": 100, "output_tokens": 20},
            result='```json-action\n{"action": "noop", "rationale": "r"}\n```',
        ),
    ]


def _events():
    return [ev for m in _session_messages() for ev in message_events(m)]


def test_message_events_capture_every_turn() -> None:
    kinds = [e["kind"] for e in _events()]
    assert kinds == [
        "thinking",
        "text",
        "tool_use",
        "tool_result",
        "tool_use",
        "tool_result",
        "result",
    ]
    ev = _events()
    assert ev[2] == {**ev[2], "name": "Read", "id": "t1", "input": {"file_path": "scaffold/a.md"}}
    assert ev[5]["content"] == "exit 1" and ev[5]["is_error"] is True
    assert ev[6]["num_turns"] == 3 and ev[6]["total_cost_usd"] == 0.12


def test_message_events_cap_large_payloads() -> None:
    big = "x" * 50_000
    (ev,) = message_events(
        _obj("UserMessage", content=[_obj("ToolResultBlock", tool_use_id="t", content=big)])
    )
    assert len(ev["content"]) < 21_000 and "truncated, 50000 chars" in ev["content"]


def test_render_session_markdown_is_turn_by_turn() -> None:
    md = render_session_markdown(_events(), title="Optimizer session — round 7")
    assert md.startswith("# Optimizer session — round 7")
    assert "## tool call: `Read`" in md and "### result of `Read`" in md
    assert "### result of `Bash` (error)" in md
    assert "num_turns=3" in md and "total_cost_usd=0.12" in md


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # MLflow's tracer caches its export destination when first used, so
    # reset it around the URI switch — otherwise traces go to whatever store
    # an earlier test pointed it at.
    # Earlier tests can leave an active experiment from another store behind.
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    prev = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.tracing.reset()
    try:
        yield mlflow.MlflowClient()
    finally:
        mlflow.set_tracking_uri(prev)
        mlflow.tracing.reset()


def test_log_optimizer_trace_replays_the_session(local_mlflow) -> None:
    exp_id = local_mlflow.create_experiment("optimizer")
    run_id = local_mlflow.create_run(exp_id).info.run_id
    events = _events()
    started = min(e["t_ns"] for e in events) - 1_000
    trace_id = log_optimizer_trace(
        events=events,
        prompt="# Round 7",
        experiment_id=exp_id,
        started_ns=started,
        ended_ns=max(e["t_ns"] for e in events) + 1_000,
        tags={"round": "7", "decision": "revert"},
        outputs={"action": "noop"},
        source_run_id=run_id,
    )
    assert trace_id
    trace = mlflow.get_trace(trace_id)
    assert trace.info.tags["round"] == "7" and trace.info.tags["source"] == "optimizer"
    assert trace.info.trace_metadata.get("mlflow.sourceRun") == run_id
    spans = {s.name: s for s in trace.data.spans}
    root = spans["forge_optimizer_session"]
    assert root.inputs == {"prompt": "# Round 7"} and root.outputs == {"action": "noop"}
    assert root.attributes["optimizer.num_turns"] == 3
    assert spans["tool:Read"].outputs == {"content": "file body"}
    assert spans["tool:Read"].inputs == {"file_path": "scaffold/a.md"}
    assert str(spans["tool:Bash"].status.status_code).endswith("ERROR")
    assert {"thinking", "assistant_message"} <= set(spans)


def test_log_optimizer_trace_is_best_effort() -> None:
    assert (
        log_optimizer_trace(events=[], prompt="p", experiment_id="1", started_ns=1, ended_ns=2)
        is None
    )
    # An MLflow failure never raises.
    assert (
        log_optimizer_trace(
            events=_events(),
            prompt="p",
            experiment_id="does-not-exist",
            started_ns=None,
            ended_ns=None,
        )
        is None
    )


# ---------------------------------------------------------------- run close


class _FakeClient:
    def __init__(self, runs):
        self.runs = runs
        self.terminated: dict[str, str] = {}
        self.filters: list[str] = []

    def search_runs(self, _exps, filter_string, max_results):  # noqa: ARG002
        self.filters.append(filter_string)
        return self.runs

    def set_tag(self, *_a):
        pass

    def set_terminated(self, run_id, status):
        self.terminated[run_id] = status


def _run(run_id, pid):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id), data=SimpleNamespace(tags={"forge.pid": pid})
    )


def test_reaper_kills_only_dead_local_sessions() -> None:
    dead = 2**22 + 12345  # far above any live pid
    client = _FakeClient(
        [
            _run("dead", str(dead)),
            _run("live", str(os.getpid())),
            _run("nopid", ""),
            _run("me", str(dead)),
        ]
    )
    reaped = reap_orphaned_cli_sessions(client, "1", keep_run_id="me")
    assert reaped == ["dead"]
    assert client.terminated == {"dead": "KILLED"}
    assert socket.gethostname() in client.filters[0] and "RUNNING" in client.filters[0]


def _load_run_round_script() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / "run_round.py"
    spec = importlib.util.spec_from_file_location("run_round_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stop", ["sigterm", "ctrl_c"])
def test_stopped_session_closes_its_run_as_killed(monkeypatch: pytest.MonkeyPatch, stop) -> None:
    module = _load_run_round_script()
    handlers: dict = {}
    closed: list[str] = []
    monkeypatch.setattr(module.signal, "signal", lambda sig, fn: handlers.__setitem__(sig, fn))
    monkeypatch.setattr(module, "check_clean_worktree", lambda: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(module, "ensure_agent_evals_ready", lambda *a, **k: None)
    monkeypatch.setattr(module, "configure_tracking_uri", lambda *_a: "databricks")
    monkeypatch.setattr(
        module,
        "open_cli_session_sink",
        lambda *a, **k: SimpleNamespace(run_id="r", experiment_id="e"),
    )
    monkeypatch.setattr(
        module, "close_cli_session_sink", lambda *a, status, **k: closed.append(status)
    )

    def _fake_round(**_kw):
        if stop == "sigterm":
            handlers[module.signal.SIGTERM](module.signal.SIGTERM, None)
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "run_round", _fake_round)
    with pytest.raises((SystemExit, KeyboardInterrupt)):
        module.main(["--rounds", "1"])
    assert closed == ["KILLED"]
    assert module.signal.SIGHUP in handlers
