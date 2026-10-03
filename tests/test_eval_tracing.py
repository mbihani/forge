"""Tests for the GENERIC eval-tracing seam (observability + engine adoption).

No gateway, no workspace: MLflow is pointed at an isolated local file store so
``setup_eval_tracing`` / ``eval_row_trace`` create real traces we can read back.
This proves that any engine wrapping its rows in ``eval_row_trace`` yields a
per-row trace in the eval experiment — the mechanism that makes round-by-round
optimization activity visible for ALL engines, not just the builtin genai one.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest

from anvil.observability import eval_row_trace, setup_eval_tracing

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def local_mlruns(tmp_path: Path):
    """Point mlflow at an isolated local file store; restore on teardown."""
    store = tmp_path / "mlruns"
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{store}")
    yield store
    mlflow.set_tracking_uri(old_uri)


def test_setup_eval_tracing_sets_experiment(local_mlruns: Path) -> None:
    exp_id = setup_eval_tracing(experiment="eval-tracing-test", profile=None)
    assert exp_id is not None
    assert mlflow.get_experiment_by_name("eval-tracing-test").experiment_id == exp_id


def test_eval_row_trace_opens_active_span(local_mlruns: Path) -> None:
    setup_eval_tracing(experiment="row-span-test", profile=None)
    with eval_row_trace(
        example_id="r1",
        query="q",
        scaffold_root=REPO_ROOT / "scaffold",
        runtime_endpoint="databricks-claude-sonnet-4-6",
    ) as span:
        assert span is not None
        assert mlflow.get_current_active_span() is not None


def _toy_engine(rows: list[dict], *, trace_rows: bool) -> int:
    """Minimal engine following the per-row contract (docs/onboarding.md §15):
    each row runs inside ``eval_row_trace`` only when tracing is armed."""
    from contextlib import nullcontext

    done = 0
    for row in rows:
        cm = (
            eval_row_trace(
                example_id=row["example_id"],
                query=row["query"],
                scaffold_root=REPO_ROOT / "scaffold",
                runtime_endpoint="databricks-claude-sonnet-4-6",
            )
            if trace_rows
            else nullcontext()
        )
        with cm as span:
            if span is not None:
                span.set_outputs({"score": 1.0})
            done += 1
    return done


_ROWS = [{"example_id": f"ex_{i}", "query": f"q{i}"} for i in range(8)]


def test_engine_trace_rows_emits_one_trace_per_row(local_mlruns: Path) -> None:
    """With tracing armed, a custom engine yields one trace per row in the eval experiment."""
    exp_id = setup_eval_tracing(experiment="engine-trace-test", profile=None)
    assert _toy_engine(_ROWS, trace_rows=True) == 8
    traces = mlflow.search_traces(experiment_ids=[exp_id], return_type="list")
    assert len(traces) == 8
    assert {t.info.tags.get("source") for t in traces} == {"eval"}


def test_engine_trace_rows_false_creates_no_trace(local_mlruns: Path) -> None:
    """Default (trace_rows=False) stays hermetic — no traces created."""
    exp_id = setup_eval_tracing(experiment="engine-notrace-test", profile=None)
    assert _toy_engine(_ROWS, trace_rows=False) == 8
    traces = mlflow.search_traces(experiment_ids=[exp_id], return_type="list")
    assert len(traces) == 0


# --- tracking URI + CLI optimizer sink -------------------------------------


def test_configure_tracking_uri_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Named profile → databricks://p; unset + creds → databricks; explicit wins."""
    from mlflow.tracking._tracking_service import utils as tracking_utils

    from anvil import observability

    calls: list[str] = []
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: calls.append(uri))
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: calls[-1] if calls else "sqlite:///x")

    observability.configure_tracking_uri("fevm-stable")
    assert calls[-1] == "databricks://fevm-stable"

    calls.clear()
    monkeypatch.setattr(tracking_utils, "is_tracking_uri_set", lambda: False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    assert observability.configure_tracking_uri(None) == "databricks"

    calls.clear()
    # An explicitly set URI (e.g. MLFLOW_TRACKING_URI) is never overridden.
    monkeypatch.setattr(tracking_utils, "is_tracking_uri_set", lambda: True)
    observability.configure_tracking_uri("DEFAULT")
    assert calls == []  # an explicit URI is never overridden


def test_cli_session_sink_logs_rounds_and_summary(local_mlruns: Path, tmp_path: Path) -> None:
    from anvil.loop.optimizer_artifacts import close_cli_session_sink, open_cli_session_sink

    repo = tmp_path / "repo"
    (repo / "scaffold").mkdir(parents=True)
    (repo / "harness").mkdir()
    (repo / "harness" / "config.yaml").write_text(
        "eval: {engine: clitest}\nexperiments: {optimizer: /Shared/forge/clitest/cli-optimizer-test}\n"
    )
    runs = repo / "eval" / "runs"
    runs.mkdir(parents=True)
    (runs / "round_001.json").write_text(
        json.dumps({"round_id": 1, "decision": "keep", "aggregate": 0.9})
    )

    sink = open_cli_session_sink(repo, session_id="cli-test")
    assert sink is not None
    sink.log_round(round_id=1, transcript="t", critique_md="c")
    close_cli_session_sink(sink, repo, round_ids=[1], session_meta={"session_id": "cli-test"})

    client = mlflow.MlflowClient()
    run = client.get_run(sink.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["anvil.surface"] == "cli"
    files = {a.path for a in client.list_artifacts(sink.run_id)} | {
        a.path for a in client.list_artifacts(sink.run_id, "rounds")
    }
    assert {"improvement_summary.json", "rounds/round_001_transcript.md"} <= files
