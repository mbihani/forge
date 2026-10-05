"""A local optimizer session that ends in an API error is an INFRA_FAIL, not a noop.

Also: which Claude Code CLI the local backend runs, and the run loop stopping
after repeated infra failures. Offline — no SDK subprocess is started.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from anvil.loop.decision import Decision
from anvil.optimizer.session import resolve_claude_cli_path, session_error

_STALE_CLI = (
    'API Error: 400 {"message":"Claude Code 2.1.116 does not support this model; '
    'version 2.1.280 or newer is required."}'
)


def _result(**kw):
    return {
        "kind": "result",
        "t_ns": 1,
        "subtype": "success",
        "is_error": False,
        "text": None,
        **kw,
    }


def test_api_error_on_first_turn_is_a_session_error() -> None:
    events = [{"kind": "text", "t_ns": 0, "text": _STALE_CLI}, _result(is_error=True)]
    err = session_error(events)
    assert err is not None and "2.1.116 does not support this model" in err


def test_result_text_is_preferred_for_the_error() -> None:
    assert "boom" in session_error([_result(is_error=True, text="boom")])


def test_normal_and_max_turns_endings_are_not_errors() -> None:
    assert session_error([{"kind": "text", "t_ns": 0, "text": "ok"}, _result()]) is None
    assert session_error([_result(is_error=True, subtype="error_max_turns")]) is None
    assert session_error([]) is None


def test_cli_path_env_override_then_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import anvil.optimizer.session as session_mod

    monkeypatch.setattr(session_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("ANVIL_CLAUDE_CLI_PATH", "/opt/claude")
    assert resolve_claude_cli_path() == "/opt/claude"
    monkeypatch.delenv("ANVIL_CLAUDE_CLI_PATH")
    assert resolve_claude_cli_path() == "/usr/bin/claude"
    monkeypatch.setattr(session_mod.shutil, "which", lambda name: None)  # noqa: ARG005
    assert resolve_claude_cli_path() is None  # the SDK's bundled CLI


def _load_run_round_script() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / "run_round.py"
    spec = importlib.util.spec_from_file_location("run_round_script_errors", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_stops_after_consecutive_infra_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_run_round_script()
    closed: list[str] = []
    calls: list[int] = []
    monkeypatch.setattr(module.signal, "signal", lambda *a: None)
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

    def _fake_round(*, round_id, **_kw):
        calls.append(round_id)
        return SimpleNamespace(decision=Decision.INFRA_FAIL, action_kind="noop", score_delta=0.0)

    monkeypatch.setattr(module, "run_round", _fake_round)
    module.main(["--rounds", "10"])
    assert len(calls) == module.MAX_CONSECUTIVE_INFRA_FAILS == 2
    assert closed == ["FAILED"]
