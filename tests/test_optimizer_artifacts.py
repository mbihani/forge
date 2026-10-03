"""Durable optimizer-artifact persistence — sink, summary, config, guard.

Covers the acceptance contract for persisting optimizer logs + the run
improvement summary to a parent MLflow run under ``experiments.optimizer``:

* (a) per-round transcript + critique are logged to the parent run at the
  expected artifact paths (``rounds/round_NNN_{transcript,critique}.md``);
* (b) the improvement summary is assembled correctly from a representative
  rounds + frontier + finalized + baseline fixture (per-round decisions,
  net improvement, kept/reverted counts);
* (c) BEST-EFFORT guard — a raising MLflow client never fails a round
  (driven end-to-end through ``run_round``) nor the sink helpers;
* (d) config — the ``persistence`` block defaults parse, the env override
  wins, and a pre-existing config WITHOUT the block still validates.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

# Import ``anvil.eval`` before ``anvil.loop.*`` — mirrors the import-ordering
# side effect documented in tests/test_noop_commit.py (circular import between
# anvil.loop.frontier and anvil.eval.cache via anvil.runtime.models).
import anvil.eval  # noqa: F401
from anvil.eval.cache import CachedBaseline, save_baseline
from anvil.loop.decision import Decision
from anvil.loop.optimizer_artifacts import (
    OptimizerArtifactSink,
    build_improvement_summary,
    persist_improvement_summary,
    render_improvement_summary_md,
    resolve_persistence_settings,
)
from anvil.loop.round import run_round
from anvil.optimizer.actions import NoopAction
from anvil.optimizer.parser import ParseResult
from anvil.runtime.models import OptimizerPersistenceConfig, RuntimeYAML

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Fake MlflowClient recording ``log_text`` + ``set_terminated`` calls."""

    def __init__(self) -> None:
        self.texts: list[tuple[str, str, str]] = []  # (run_id, text, artifact_file)
        self.terminated: list[tuple[str, str]] = []  # (run_id, status)

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        self.texts.append((run_id, text, artifact_file))

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
        self.terminated.append((run_id, status))


class _RaisingClient:
    """Fake MlflowClient whose every method raises — the dead-sink case."""

    def log_text(self, *_a: object, **_k: object) -> None:
        raise RuntimeError("tracking server unreachable")

    def set_terminated(self, *_a: object, **_k: object) -> None:
        raise RuntimeError("tracking server unreachable")


# ---------------------------------------------------------------------------
# (a) per-round logs to the parent run at the expected paths
# ---------------------------------------------------------------------------


def test_log_round_writes_expected_artifact_paths() -> None:
    client = _RecordingClient()
    sink = OptimizerArtifactSink(client=client, run_id="run-1", experiment_id="exp-1")

    sink.log_round(round_id=7, transcript="the transcript", critique_md="the critique")

    paths = {c[2] for c in client.texts}
    assert paths == {"rounds/round_007_transcript.md", "rounds/round_007_critique.md"}
    by_path = {c[2]: c[1] for c in client.texts}
    assert by_path["rounds/round_007_transcript.md"] == "the transcript"
    assert by_path["rounds/round_007_critique.md"] == "the critique"
    assert all(c[0] == "run-1" for c in client.texts)


def test_log_improvement_summary_writes_json_and_md() -> None:
    client = _RecordingClient()
    sink = OptimizerArtifactSink(client=client, run_id="run-1", experiment_id="exp-1")

    sink.log_improvement_summary(summary={"net_improvement": 0.1}, summary_md="# hi")

    paths = {c[2] for c in client.texts}
    assert paths == {"improvement_summary.json", "improvement_summary.md"}
    by_path = {c[2]: c[1] for c in client.texts}
    assert json.loads(by_path["improvement_summary.json"]) == {"net_improvement": 0.1}
    assert by_path["improvement_summary.md"] == "# hi"


def test_close_terminates_the_run() -> None:
    client = _RecordingClient()
    sink = OptimizerArtifactSink(client=client, run_id="run-1", experiment_id="exp-1")
    sink.close()
    assert client.terminated == [("run-1", "FINISHED")]


# ---------------------------------------------------------------------------
# (b) improvement summary assembly
# ---------------------------------------------------------------------------


def _rounds_fixture() -> list[dict]:
    """Three rounds: a KEEP that improves, a REVERT that regresses, a NOOP."""
    return [
        {
            "round_id": 1,
            "decision": "keep",
            "action_kind": "edit_skill",
            "baseline_score": 0.50,
            "aggregate": 0.60,
            "score_delta_vs_parent": 0.10,
            "rationale": "tightened the retrieval skill",
            "mlflow": {"run_id": "eval-run-1", "experiment_id": "e"},
            "optimizer_error": None,
        },
        {
            "round_id": 2,
            "decision": "revert",
            "action_kind": "edit_rule",
            "baseline_score": 0.50,
            "aggregate": 0.45,
            "score_delta_vs_parent": -0.05,
            "rationale": "loosened a rule — regressed",
            "mlflow": {"run_id": "eval-run-2", "experiment_id": "e"},
            "optimizer_error": None,
        },
        {
            "round_id": 3,
            "decision": "noop",
            "action_kind": "noop",
            "baseline_score": 0.60,
            "aggregate": None,
            "score_delta_vs_parent": None,
            "rationale": "nothing worth changing",
            "mlflow": None,
            "optimizer_error": None,
        },
    ]


def test_build_improvement_summary_assembles_correctly() -> None:
    summary = build_improvement_summary(
        session_meta={"session_id": "s1"},
        rounds=_rounds_fixture(),
        frontier={"best": {"aggregate": 0.60}},
        finalized={"aggregate": 0.58},
        baseline={"aggregate": 0.50},
    )

    assert summary["baseline_aggregate"] == 0.50
    # Final aggregate prefers the held-out finalized aggregate.
    assert summary["final_aggregate"] == 0.58
    assert summary["net_improvement"] == pytest.approx(0.08)
    assert summary["rounds_total"] == 3
    assert summary["decision_counts"] == {"keep": 1, "revert": 1, "noop": 0 + 1, "infra_fail": 0}

    # Per-round rows map decision/action/eval-run correctly.
    rows = {r["round"]: r for r in summary["per_round"]}
    assert rows[1]["decision"] == "keep"
    assert rows[1]["eval_run_id"] == "eval-run-1"
    assert rows[2]["mutated_aggregate"] == 0.45
    # A noop inherits the parent's aggregate (baseline_score) as its score.
    assert rows[3]["mutated_aggregate"] == 0.60

    # Trajectory carries one entry per round.
    assert [t["round"] for t in summary["score_trajectory"]] == [1, 2, 3]


def test_build_improvement_summary_falls_back_to_frontier_without_finalized() -> None:
    """Without a finalized report, net improvement uses the frontier best."""
    summary = build_improvement_summary(
        session_meta={},
        rounds=_rounds_fixture(),
        frontier={"best": {"aggregate": 0.60}},
        finalized=None,
        baseline={"aggregate": 0.50},
    )
    assert summary["finalized_present"] is False
    assert summary["final_aggregate"] == 0.60
    assert summary["net_improvement"] == pytest.approx(0.10)


def test_render_improvement_summary_md_is_tabular() -> None:
    summary = build_improvement_summary(
        session_meta={
            "session_id": "s1",
            "repo_url": "https://example/repo",
            "agent_ref": "abc123",
            "mode": "prompt",
            "backend": "local",
            "started_at": "2026-10-02T00:00:00+00:00",
            "finished_at": "2026-10-02T00:10:00+00:00",
        },
        rounds=_rounds_fixture(),
        frontier={"best": {"aggregate": 0.60}},
        finalized={"aggregate": 0.58},
        baseline={"aggregate": 0.50},
    )
    md = render_improvement_summary_md(summary)
    assert "# Optimize session s1 — improvement summary" in md
    assert "Net improvement:** +0.0800" in md
    assert "| Round | Decision | Action |" in md
    # Each fixture round appears as a table row.
    assert "| 1 | keep | edit_skill |" in md
    assert "kept 1 · reverted 1 · noop 1 · infra_fail 0" in md


def test_final_aggregate_fallback_uses_last_kept_not_last_round() -> None:
    """Without finalized/frontier, the 'final' aggregate must reflect the
    RETAINED agent — the last KEPT round — NOT a later reverted candidate
    whose mutation was discarded.

    Rounds: R1 KEEP@0.60, R2 REVERT@0.70 (a tempting-but-discarded higher
    score). The retained agent is R1's 0.60, so net = 0.60 - 0.50 = +0.10.
    If the fallback naively took the last evaluated round it would report
    0.70 / +0.20 — the bite this test guards.
    """
    rounds = [
        {"round_id": 1, "decision": "keep", "action_kind": "edit_skill",
         "baseline_score": 0.50, "aggregate": 0.60, "score_delta_vs_parent": 0.10},
        {"round_id": 2, "decision": "revert", "action_kind": "edit_rule",
         "baseline_score": 0.60, "aggregate": 0.70, "score_delta_vs_parent": 0.10},
    ]
    summary = build_improvement_summary(
        session_meta={},
        rounds=rounds,
        frontier=None,
        finalized=None,
        baseline={"aggregate": 0.50},
    )
    assert summary["final_aggregate"] == 0.60
    assert summary["net_improvement"] == pytest.approx(0.10)


def test_final_aggregate_fallback_all_reverted_is_baseline() -> None:
    """When NO round was kept (all reverted/noop/infra_fail) and there is no
    finalized/frontier data, the agent is still the baseline: final ==
    baseline and net_improvement == 0 — never a discarded candidate's score."""
    rounds = [
        {"round_id": 1, "decision": "revert", "action_kind": "edit_skill",
         "baseline_score": 0.50, "aggregate": 0.70, "score_delta_vs_parent": 0.20},
        {"round_id": 2, "decision": "infra_fail", "action_kind": "noop",
         "baseline_score": 0.50, "aggregate": None, "score_delta_vs_parent": None},
    ]
    summary = build_improvement_summary(
        session_meta={},
        rounds=rounds,
        frontier=None,
        finalized=None,
        baseline={"aggregate": 0.50},
    )
    assert summary["final_aggregate"] == 0.50
    assert summary["net_improvement"] == 0.0


# ---------------------------------------------------------------------------
# (c) best-effort guard
# ---------------------------------------------------------------------------


def test_sink_methods_swallow_a_raising_client() -> None:
    """Every sink write/close swallows a raising client — no exception."""
    sink = OptimizerArtifactSink(client=_RaisingClient(), run_id="r", experiment_id="e")
    # None of these may raise.
    sink.log_round(round_id=1, transcript="t", critique_md="c")
    sink.log_improvement_summary(summary={"x": 1}, summary_md="md")
    sink.close()


def test_persist_improvement_summary_swallows_failure() -> None:
    """persist_improvement_summary never raises — None sink or raising sink."""
    persist_improvement_summary(
        None,
        session_meta={},
        rounds=[],
        frontier=None,
        finalized=None,
        baseline=None,
    )
    sink = OptimizerArtifactSink(client=_RaisingClient(), run_id="r", experiment_id="e")
    persist_improvement_summary(
        sink,
        session_meta={"session_id": "s"},
        rounds=_rounds_fixture(),
        frontier={"best": {"aggregate": 0.6}},
        finalized={"aggregate": 0.58},
        baseline={"aggregate": 0.5},
    )


def test_open_returns_none_when_mlflow_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tracking server that rejects the client construction => None sink
    (persistence skipped), never an exception that aborts the session."""
    import mlflow.tracking as mlflow_tracking

    class _Boom:
        def __init__(self, *_a: object, **_k: object) -> None:
            raise RuntimeError("no tracking server")

    monkeypatch.setattr(mlflow_tracking, "MlflowClient", _Boom)
    assert OptimizerArtifactSink.open(experiment_name="/Shared/x", session_id="s") is None


# -- run_round end-to-end still succeeds with a dead sink --------------------


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    """Minimal ANVIL repo on ``anvil/exp`` (mirrors test_noop_commit)."""
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "scaffold").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text("tools: []\n", encoding="utf-8")
    (repo / "harness").mkdir()
    (repo / "harness" / "config.yaml").write_text("mode: prompt\n", encoding="utf-8")
    (repo / "eval" / "runs").mkdir(parents=True)
    save_baseline(
        repo,
        CachedBaseline(
            scaffold_commit_sha="a" * 40,
            evaluated_at="2026-08-16T12:00:00+00:00",
            mode="test",
            scorers=["correctness"],
            runtime_endpoint="runtime",
            judge_endpoint="judge",
            aggregate=0.5,
            per_judge={"correctness": 0.5},
            per_bucket={"direct": {"correctness": 0.5}},
            n_examples=10,
        ),
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "checkout", "-b", "anvil/exp")
    return repo


def test_run_round_with_raising_sink_still_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sink whose MLflow client RAISES on every call must NOT fail the
    round. The round completes and the parent-run pointer still lands in the
    round JSON (discoverability), proving the write is attempted but guarded.
    """
    repo = _init_repo(tmp_path)

    import anvil.loop.round as round_mod

    action = NoopAction(rationale="nothing worth mutating this round")
    parse_result = ParseResult(action=action, parse_status="ok")

    async def _fake_session(**_kwargs: object):  # noqa: ANN003
        return (action, "(clean noop)\n", parse_result)

    monkeypatch.setattr(round_mod, "run_optimizer_session", _fake_session)

    sink = OptimizerArtifactSink(client=_RaisingClient(), run_id="run-xyz", experiment_id="exp-9")
    report = run_round(
        round_id=1,
        repo_root=repo,
        parent_branch="anvil/exp",
        max_turns=1,
        artifact_sink=sink,
    )

    assert report.decision is Decision.NOOP  # round completed despite dead sink
    round_json = json.loads(
        (repo / "eval" / "runs" / "round_001.json").read_text(encoding="utf-8")
    )
    assert round_json["optimizer_run"] == {"run_id": "run-xyz", "experiment_id": "exp-9"}
    assert "rationale" in round_json


# ---------------------------------------------------------------------------
# (d) config — defaults, env override, backward compatibility
# ---------------------------------------------------------------------------


_MINIMAL_RUNTIME = {
    "runtime_endpoint": "r",
    "optimizer_endpoint": "o",
    "judge_endpoint": "j",
    "experiments": {"root": "/Shared/forge"},
}


def test_runtime_yaml_defaults_persistence_enabled() -> None:
    """A config WITHOUT a ``persistence:`` block validates and defaults ON."""
    cfg = RuntimeYAML(**_MINIMAL_RUNTIME)
    assert isinstance(cfg.persistence, OptimizerPersistenceConfig)
    assert cfg.persistence.enabled is True
    assert cfg.persistence.experiment is None


def test_runtime_yaml_accepts_persistence_block() -> None:
    cfg = RuntimeYAML(**_MINIMAL_RUNTIME, persistence={"enabled": False, "experiment": "/Shared/forge/genai/custom"})
    assert cfg.persistence.enabled is False
    assert cfg.persistence.experiment == "/Shared/forge/genai/custom"


def test_persistence_block_forbids_extra_keys() -> None:
    with pytest.raises(ValueError):
        OptimizerPersistenceConfig(enabled=True, bogus=1)  # type: ignore[call-arg]


def _write_config(repo: Path, body: str) -> Path:
    (repo / "scaffold").mkdir(parents=True, exist_ok=True)
    (repo / "harness").mkdir(parents=True, exist_ok=True)
    (repo / "harness" / "config.yaml").write_text(body, encoding="utf-8")
    return repo / "scaffold"


def test_resolve_persistence_settings_defaults(tmp_path: Path) -> None:
    """Config with ``experiments.optimizer`` but no persistence block →
    enabled by default, experiment taken from experiments.optimizer."""
    scaffold = _write_config(
        tmp_path,
        "mode: prompt\nexperiments:\n  optimizer: /Shared/forge/genai/opt-x\n",
    )
    enabled, experiment = resolve_persistence_settings(scaffold)
    assert enabled is True
    assert experiment == "/Shared/forge/genai/opt-x"


def test_resolve_persistence_settings_no_config_file(tmp_path: Path) -> None:
    """No config.yaml at all → enabled default + the default domain's experiment."""
    enabled, experiment = resolve_persistence_settings(tmp_path / "scaffold")
    assert enabled is True
    assert experiment == "/Shared/forge/genai/optimizer"


def test_resolve_persistence_settings_file_disables(tmp_path: Path) -> None:
    scaffold = _write_config(tmp_path, "persistence:\n  enabled: false\n")
    enabled, _ = resolve_persistence_settings(scaffold)
    assert enabled is False


def test_env_override_wins_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ANVIL_PERSIST_OPTIMIZER_ARTIFACTS is authoritative over the file —
    both directions (env enables a file-disabled block, env disables a
    file-enabled one)."""
    scaffold = _write_config(tmp_path, "persistence:\n  enabled: false\n")

    monkeypatch.setenv("ANVIL_PERSIST_OPTIMIZER_ARTIFACTS", "1")
    enabled, _ = resolve_persistence_settings(scaffold)
    assert enabled is True

    monkeypatch.setenv("ANVIL_PERSIST_OPTIMIZER_ARTIFACTS", "0")
    enabled, _ = resolve_persistence_settings(scaffold)
    assert enabled is False

    # An unrecognized value is treated as unset (file value preserved).
    monkeypatch.setenv("ANVIL_PERSIST_OPTIMIZER_ARTIFACTS", "maybe")
    enabled, _ = resolve_persistence_settings(scaffold)
    assert enabled is False


def test_persistence_experiment_overrides_experiments_optimizer(tmp_path: Path) -> None:
    scaffold = _write_config(
        tmp_path,
        "persistence:\n  experiment: /Shared/forge/genai/custom-opt\n"
        "experiments:\n  optimizer: /Shared/forge/genai/optimizer\n",
    )
    _, experiment = resolve_persistence_settings(scaffold)
    assert experiment == "/Shared/forge/genai/custom-opt"


# ---------------------------------------------------------------------------
# finalize re-log preserves the ORIGINAL session start metadata
# ---------------------------------------------------------------------------


def test_finalize_relog_preserves_started_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The held-out summary re-logged at finalize must preserve the original
    ``started_at`` (captured at task start into SessionData), not blank it.

    Reverting the fix (passing ``started_at=""``) makes this fail: the
    re-logged ``improvement_summary.json`` would carry an empty start time.
    """
    from anvil.orchestrator import app as app_mod

    recording = _RecordingClient()

    def _fake_bind(**kwargs: object) -> OptimizerArtifactSink:
        return OptimizerArtifactSink(
            client=recording,
            run_id=str(kwargs["run_id"]),
            experiment_id=str(kwargs["experiment_id"]),
        )

    monkeypatch.setattr(app_mod.OptimizerArtifactSink, "bind", staticmethod(_fake_bind))

    start_ts = "2026-10-02T00:00:00+00:00"
    sess = app_mod.SessionData(
        session_id="s1",
        repo_url="https://example/repo",
        repo_path=tmp_path,
        status="finalized",
        validation={},
        config={},
        baseline={"aggregate": 0.50},
        rounds=[
            {"round_id": 1, "decision": "keep", "action_kind": "edit_skill",
             "baseline_score": 0.50, "aggregate": 0.60, "score_delta_vs_parent": 0.10},
        ],
        frontier=None,
        finalized={"aggregate": 0.58},
        error=None,
        optimizer_run_id="run-1",
        optimizer_experiment_id="exp-1",
        optimizer_started_at=start_ts,
    )

    app_mod._relog_summary_after_finalize_sync(sess)

    by_path = {c[2]: c[1] for c in recording.texts}
    summary = json.loads(by_path["improvement_summary.json"])
    assert summary["session"]["started_at"] == start_ts
    # The held-out net improvement is also present (finalized aggregate used).
    assert summary["net_improvement"] == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# finally-block parent-run close must never mask the real task outcome
# ---------------------------------------------------------------------------


def test_task_finally_close_failure_does_not_mask_real_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a round raises AND the finally-block ``sink.close`` ALSO raises,
    ``_run_optimization_task`` must still complete normally (no exception
    escapes) and record the REAL failure (the round error) — the close error
    must be swallowed, never mask the outcome.

    Reverting the close guard (removing the try/except around
    ``anyio.to_thread.run_sync(sink.close)``) makes the close exception
    escape the finally, so the task raises instead of recording the round
    error — this test bites that.
    """
    import asyncio

    from anvil.orchestrator import app as app_mod

    class _ClosesBoom:
        run_id = "run-1"
        experiment_id = "exp-1"

        def close(self) -> None:
            raise RuntimeError("close boom")

    closed: list[bool] = []

    def _fake_close() -> None:
        closed.append(True)
        raise RuntimeError("close boom")

    sink = _ClosesBoom()
    sink.close = _fake_close  # type: ignore[method-assign]

    monkeypatch.setattr(app_mod, "_ensure_parent_branch", lambda *_a, **_k: None)
    monkeypatch.setattr(app_mod, "_build_baseline_sync", lambda *_a, **_k: {"aggregate": 0.5})
    monkeypatch.setattr(app_mod, "_open_optimizer_sink_sync", lambda *_a, **_k: sink)

    def _boom_round(**_kwargs: object) -> None:
        raise RuntimeError("round boom")

    monkeypatch.setattr(app_mod, "run_round", _boom_round)

    session_id = "sess-close-guard"
    app_mod._sessions[session_id] = app_mod.SessionData(
        session_id=session_id,
        repo_url="https://example/repo",
        repo_path=tmp_path,
        status="optimizing",
        validation={},
        config={},
        baseline=None,
        rounds=[],
        frontier=None,
        finalized=None,
        error=None,
    )
    try:
        # Must NOT raise — the close error is swallowed in the finally.
        asyncio.run(
            app_mod._run_optimization_task(session_id, None, max_rounds=1, max_turns=1)
        )
        sess = app_mod._sessions[session_id]
        # The REAL failure (the round error) is what the session records.
        assert sess.status == "error"
        assert "round boom" in (sess.error or "")
        assert "close boom" not in (sess.error or "")
        # The finally-block close was attempted (and its exception swallowed).
        assert closed == [True]
    finally:
        app_mod._sessions.pop(session_id, None)


def test_task_finally_close_is_shielded_from_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation arriving while the task runs must NOT abort the
    finally-block parent-run close: the close is SHIELDED so it runs to
    completion, and cancellation is NOT suppressed — ``CancelledError`` still
    propagates afterward (the task ends cancelled).

    Reverting the shield makes the cancellation abort the close await before
    it completes (``closed`` stays empty) — this test bites that.
    """
    import asyncio
    import threading
    import time

    from anvil.orchestrator import app as app_mod

    started = threading.Event()
    closed: list[bool] = []

    def _blocking_round(**_kwargs: object) -> None:
        # Signal that the task is inside the (cancellable) round await, then
        # block briefly so the test can cancel the task at this await point.
        started.set()
        time.sleep(0.3)

    def _slow_close() -> None:
        # Non-trivial so "ran to completion" is observable; must finish even
        # though the surrounding task is being cancelled (shield).
        time.sleep(0.05)
        closed.append(True)

    class _Sink:
        run_id = "run-1"
        experiment_id = "exp-1"
        close = staticmethod(_slow_close)

    monkeypatch.setattr(app_mod, "_ensure_parent_branch", lambda *_a, **_k: None)
    monkeypatch.setattr(app_mod, "_build_baseline_sync", lambda *_a, **_k: {"aggregate": 0.5})
    monkeypatch.setattr(app_mod, "_open_optimizer_sink_sync", lambda *_a, **_k: _Sink())
    monkeypatch.setattr(app_mod, "run_round", _blocking_round)

    session_id = "sess-cancel-shield"
    app_mod._sessions[session_id] = app_mod.SessionData(
        session_id=session_id,
        repo_url="https://example/repo",
        repo_path=tmp_path,
        status="optimizing",
        validation={},
        config={},
        baseline=None,
        rounds=[],
        frontier=None,
        finalized=None,
        error=None,
    )

    async def _drive() -> None:
        task = asyncio.create_task(
            app_mod._run_optimization_task(session_id, None, max_rounds=1, max_turns=1)
        )
        # Wait until the task is parked inside the round await, then cancel.
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        # Cancellation must NOT be suppressed — it propagates after the
        # shielded close completes.
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(_drive())
        # The shielded close ran to completion despite the cancellation.
        assert closed == [True]
    finally:
        app_mod._sessions.pop(session_id, None)
