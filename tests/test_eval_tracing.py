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

from anvil.domains.pitcrew.eval import evaluate_pitcrew
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


@pytest.fixture
def tmp_golden(tmp_path: Path) -> Path:
    rows = []
    for i in range(8):
        pdf = tmp_path / f"rule_{i}.pdf"
        pdf.write_bytes(b"%PDF-1.4 placeholder")
        rows.append(
            {
                "example_id": f"finra_{i}",
                "query": f"Rule {i}",
                "category": "finra",
                "pdf_path": str(pdf),
            }
        )
    gs = tmp_path / "golden.jsonl"
    gs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return gs


def test_engine_trace_rows_emits_one_trace_per_row(
    local_mlruns: Path, tmp_golden: Path
) -> None:
    """With trace_rows=True, each row yields a per-row trace + report carries the exp id.

    Mirrors the real flow: evaluate_branch arms tracing on
    ``config.experiments.eval`` before dispatch, and the engine resolves that
    same experiment name — so set it here too.
    """
    from anvil.runtime.loader import load_harness

    snap = load_harness(REPO_ROOT / "scaffold", REPO_ROOT / "harness" / "config.yaml")
    exp_id = setup_eval_tracing(experiment=snap.config.experiments.eval, profile=None)

    good = json.dumps(
        {
            "document_title": "t",
            "sections": [
                {
                    "id": "a",
                    "title": "T",
                    "level": 2,
                    "summary": "s",
                    "citations": [{"page_numbers": [1], "source_text": "q"}],
                }
            ],
        }
    )

    report = evaluate_pitcrew(
        scaffold_root=REPO_ROOT / "scaffold",
        runtime_config_path=REPO_ROOT / "harness" / "config.yaml",
        golden_set_path=tmp_golden,
        mode="quick",
        trace_rows=True,
        predict_fn=lambda _pdf: good,
        judge_fn=lambda _blk, _txt: 1.0,
    )
    assert report.n_rows == 8
    assert report.experiment_id == exp_id
    traces = mlflow.search_traces(experiment_ids=[exp_id], return_type="list")
    # One root per-row trace per example (8). Allow >= in case the store counts
    # differently across mlflow versions, but it must be at least the row count.
    assert len(traces) >= 8


def test_engine_trace_rows_false_creates_no_trace(
    local_mlruns: Path, tmp_golden: Path
) -> None:
    """Default (trace_rows=False) stays hermetic — no traces created."""
    exp_id = setup_eval_tracing(experiment="pitcrew-notrace-test", profile=None)
    evaluate_pitcrew(
        scaffold_root=REPO_ROOT / "scaffold",
        runtime_config_path=REPO_ROOT / "harness" / "config.yaml",
        golden_set_path=tmp_golden,
        mode="quick",
        predict_fn=lambda _pdf: "garbage",
        judge_fn=lambda _blk, _txt: 1.0,
    )
    traces = mlflow.search_traces(experiment_ids=[exp_id], return_type="list")
    assert len(traces) == 0
