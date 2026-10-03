"""Tests for parallel eval (Phase 4 task 4.4).

Covers the acceptance contract:

* :func:`_run_predictions_parallel` runs ``predict_fn`` across queries:
  sequential when ``n_workers <= 1`` (backward compatible), parallel via
  :class:`ThreadPoolExecutor` otherwise.
* Results preserve input order regardless of completion order.
* A prediction that raises is recorded as an empty string and logged —
  one bad row does not abort the whole eval.
* :func:`evaluate_branch` wires ``eval.n_workers`` from
  ``harness/config.yaml`` into the ``MLFLOW_GENAI_EVAL_MAX_WORKERS`` env
  var so the configured value controls mlflow's predict/score thread
  pool, while still passing ``predict_fn`` (so per-row ``RETRIEVER``
  traces are preserved for ``RetrievalGroundedness``). The env-var
  override is scoped to the ``mlflow.genai.evaluate`` call.

No LLM calls and no Databricks calls are made — ``mlflow.genai.evaluate``
and the runtime agent are mocked.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _gold(example_id: str, answer: str) -> dict:
    return {
        "example_id": example_id,
        "query": f"q-{example_id}",
        "category": "direct",
        "expected_doc_ids": [],
        "reference_answer": answer,
        "should_refuse": False,
        "expected_citations": [],
        "must_include": [answer],
        "must_not_include": [],
        "notes_for_judge": "",
    }


def _patch_runner_common(
    monkeypatch: pytest.MonkeyPatch,
    config,
) -> None:
    """Patch the runner's external dependencies for a mocked eval run.

    Mirrors the harness in ``test_code_mode.py`` so ``evaluate_branch``
    can run without an LLM, a knowledge base, or a live Databricks
    workspace.
    """
    from anvil.eval import runner

    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(
        runner, "load_golden_set", lambda _p: [_gold("g1", "hello"), _gold("g2", "world")]
    )
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "AnvilAgent", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr("anvil.observability.enable_runtime_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "get_experiment_by_name", lambda *a, **kw: None)


def _result_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "correctness/value": [1.0, 1.0],
            "trace_id": ["t0", "t1"],
        }
    )


# ---------------------------------------------------------------------------
# 1. _run_predictions_parallel — sequential mode (n_workers <= 1)
# ---------------------------------------------------------------------------


def test_run_predictions_sequential_default() -> None:
    """With the default n_workers=1, predictions run sequentially and
    in input order — the backward-compatible path."""
    from anvil.eval.runner import _run_predictions_parallel

    calls: list[str] = []

    def predict_fn(q: str) -> str:
        calls.append(q)
        return f"r-{q}"

    queries = ["a", "b", "c"]
    out = _run_predictions_parallel(predict_fn, queries)
    assert out == ["r-a", "r-b", "r-c"]
    # Called exactly once per query, in input order.
    assert calls == ["a", "b", "c"]


def test_run_predictions_sequential_n_zero_and_negative() -> None:
    """n_workers of 0 or a negative value collapses to the sequential
    path (the ``<= 1`` guard), never to a zero/negative-sized pool."""
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        return q.upper()

    for n in (0, -1, -5):
        out = _run_predictions_parallel(predict_fn, ["a", "b"], n_workers=n)
        assert out == ["A", "B"]


def test_run_predictions_empty_queries() -> None:
    """An empty query list yields an empty result list (no pool work)."""
    from anvil.eval.runner import _run_predictions_parallel

    assert _run_predictions_parallel(lambda q: q, [], n_workers=4) == []
    assert _run_predictions_parallel(lambda q: q, []) == []


# ---------------------------------------------------------------------------
# 2. _run_predictions_parallel — parallel mode (n_workers > 1)
# ---------------------------------------------------------------------------


def test_run_predictions_parallel_preserves_order() -> None:
    """Parallel execution returns results in INPUT order even when the
    futures complete in a different order. Earlier queries sleep longer
    so they finish last; the result list must still match the input."""
    from anvil.eval.runner import _run_predictions_parallel

    n = 6
    queries = [f"q{i}" for i in range(n)]
    completion_order: list[int] = []

    def predict_fn(q: str) -> str:
        i = int(q[1:])
        # q0 sleeps the longest → finishes last; q(n-1) finishes first.
        time.sleep(0.02 * (n - i))
        completion_order.append(i)
        return f"r{i}"

    out = _run_predictions_parallel(predict_fn, queries, n_workers=4)

    # Results are in input order regardless of completion order.
    assert out == [f"r{i}" for i in range(n)]
    # And parallelism actually happened: completion order is NOT the
    # input order (the longest-sleeping q0 finished after the quick ones).
    assert completion_order != list(range(n))


def test_run_predictions_parallel_more_workers_than_queries() -> None:
    """A pool larger than the query count still produces every result in
    order (excess workers simply idle)."""
    from anvil.eval.runner import _run_predictions_parallel

    out = _run_predictions_parallel(lambda q: f"x{q}", ["a", "b"], n_workers=16)
    assert out == ["xa", "xb"]


def test_run_predictions_parallel_single_worker_equivalent_to_sequential() -> None:
    """n_workers=1 takes the sequential branch (no executor), so the
    output is identical to a plain list comprehension."""
    from anvil.eval.runner import _run_predictions_parallel

    queries = [f"q{i}" for i in range(5)]
    expected = [f"r{q}" for q in queries]
    out = _run_predictions_parallel(lambda q: f"r{q}", queries, n_workers=1)
    assert out == expected


# ---------------------------------------------------------------------------
# 3. _run_predictions_parallel — error handling
# ---------------------------------------------------------------------------


def test_run_predictions_failed_row_recorded_empty_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A prediction that raises is recorded as an empty string and
    logged; the surrounding rows still complete and the call does not
    propagate the exception."""
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        if q == "boom":
            raise RuntimeError("synthetic failure")
        return f"ok-{q}"

    with caplog.at_level(logging.WARNING, logger="anvil.eval.runner"):
        out = _run_predictions_parallel(predict_fn, ["a", "boom", "c"], n_workers=3)
    assert out == ["ok-a", "", "ok-c"]

    assert "prediction failed for row 1" in caplog.text
    assert "synthetic failure" in caplog.text


def test_run_predictions_all_fail_returns_all_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every row failing still yields a full-length list of empty
    strings — the eval is not aborted by a uniformly broken agent."""
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        raise ValueError("nope")

    with caplog.at_level(logging.WARNING, logger="anvil.eval.runner"):
        out = _run_predictions_parallel(predict_fn, ["a", "b", "c"], n_workers=2)
    assert out == ["", "", ""]
    assert caplog.text.count("prediction failed") == 3


def test_run_predictions_sequential_failure_isolated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The sequential path (n_workers <= 1) isolates failures the same way
    the parallel path does: a row that raises is recorded as an empty
    string and logged, and the surrounding rows still complete — the call
    does not propagate the exception. This unifies the acceptance contract
    ("a prediction that raises is recorded as an empty string and does not
    abort the whole eval") across both paths."""
    from anvil.eval.runner import _run_predictions_parallel

    def predict_fn(q: str) -> str:
        if q == "boom":
            raise RuntimeError("synthetic failure")
        return q

    with caplog.at_level(logging.WARNING, logger="anvil.eval.runner"):
        out = _run_predictions_parallel(predict_fn, ["a", "boom", "c"], n_workers=1)
    assert out == ["a", "", "c"]
    assert "prediction failed for row 1" in caplog.text
    assert "synthetic failure" in caplog.text


# ---------------------------------------------------------------------------
# 4. Config model — EvalConfig.n_workers default + real config
# ---------------------------------------------------------------------------


def test_eval_config_n_workers_default() -> None:
    """EvalConfig defaults n_workers to 4 (the shipped config value)."""
    from anvil.runtime.models import EvalConfig

    cfg = EvalConfig()
    assert cfg.n_workers == 4


def test_real_config_has_n_workers() -> None:
    """The repo's harness/config.yaml carries eval.n_workers so the
    wiring has a real source to read from."""
    import yaml

    from anvil.runtime.models import RuntimeYAML

    config_path = REPO_ROOT / "harness" / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "n_workers" in raw["eval"]
    cfg = RuntimeYAML.model_validate(raw)
    assert cfg.eval.n_workers == 4


# ---------------------------------------------------------------------------
# 5. evaluate_branch wiring — n_workers -> MLFLOW_GENAI_EVAL_MAX_WORKERS
# ---------------------------------------------------------------------------


def _wiring_config(n_workers: int):
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    return HarnessConfig(
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="correctness", type="llm", weight=1.0)],
            modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
            n_workers=n_workers,
        ),
    )


def test_evaluate_branch_sets_max_workers_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_branch sets MLFLOW_GENAI_EVAL_MAX_WORKERS to cfg.n_workers
    for the duration of the mlflow.genai.evaluate call, then restores
    the prior value (here: unset before → unset after)."""
    from anvil.eval import runner

    _patch_runner_common(monkeypatch, _wiring_config(n_workers=3))

    # Clean baseline: the env var is unset before the call.
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)

    captured: dict[str, str | None] = {}

    def fake_evaluate(**kwargs: object) -> object:
        # The env var must be set to the configured n_workers DURING the
        # call (the try-block sets it; finally restores it after).
        captured["workers"] = os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS")
        captured["predict_fn_passed"] = kwargs.get("predict_fn") is not None
        return SimpleNamespace(result_df=_result_df(), metrics={}, run_id="run-1")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    assert captured["workers"] == "3"
    # predict_fn is still passed (trace-preserving path), not pre-computed.
    assert captured["predict_fn_passed"] is True
    # Restored: unset before → unset after.
    assert "MLFLOW_GENAI_EVAL_MAX_WORKERS" not in os.environ


def test_evaluate_branch_restores_prior_max_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the env var was already set before the call, evaluate_branch
    restores that prior value rather than deleting it."""
    from anvil.eval import runner

    _patch_runner_common(monkeypatch, _wiring_config(n_workers=8))
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", "2")

    monkeypatch.setattr(
        runner.mlflow.genai,
        "evaluate",
        lambda **kw: SimpleNamespace(result_df=_result_df(), metrics={}, run_id="run-1"),
    )

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    # The pre-call value "2" is restored, not the override "8".
    assert os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS") == "2"
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)


@pytest.mark.parametrize("prior", [None, "unexpected-value"])
def test_evaluate_branch_scopes_synchronous_trace_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior: str | None
) -> None:
    """Trace logging is synchronous only during evaluate, then restored."""
    from anvil.eval import runner

    _patch_runner_common(monkeypatch, _wiring_config(n_workers=2))
    env_name = "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"
    if prior is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, prior)

    captured: dict[str, str | None] = {}

    def fake_evaluate(**_kwargs: object) -> object:
        captured["async_trace_logging"] = os.environ.get(env_name)
        return SimpleNamespace(result_df=_result_df(), metrics={}, run_id="run-1")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    assert captured["async_trace_logging"] == "false"
    assert os.environ.get(env_name) == prior


def test_evaluate_branch_restores_env_on_evaluate_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``mlflow.genai.evaluate`` raises, the ``finally`` block still
    restores the prior ``MLFLOW_GENAI_EVAL_MAX_WORKERS`` value — the
    override never leaks past the call, even on the error path."""
    from anvil.eval import runner

    _patch_runner_common(monkeypatch, _wiring_config(n_workers=3))
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)

    def boom_evaluate(**_kwargs: object) -> object:
        raise RuntimeError("evaluate blew up")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", boom_evaluate)

    with pytest.raises(RuntimeError, match="evaluate blew up"):
        runner.evaluate_branch(
            scaffold_root=tmp_path / "scaffold",
            runtime_config_path=tmp_path / "config.yaml",
            runtime_client=SimpleNamespace(),
            judge_client=SimpleNamespace(),
        )

    # finally restored the env var: unset before the call → unset after.
    assert "MLFLOW_GENAI_EVAL_MAX_WORKERS" not in os.environ


def test_evaluate_branch_sequential_when_n_workers_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_workers <= 1 maps to MLFLOW_GENAI_EVAL_MAX_WORKERS=1 during the
    call, forcing mlflow's pool to a single worker (sequential)."""
    from anvil.eval import runner

    _patch_runner_common(monkeypatch, _wiring_config(n_workers=1))
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)

    captured: dict[str, str | None] = {}

    def fake_evaluate(**kwargs: object) -> object:
        captured["workers"] = os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS")
        return SimpleNamespace(result_df=_result_df(), metrics={}, run_id="run-1")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    assert captured["workers"] == "1"
    assert "MLFLOW_GENAI_EVAL_MAX_WORKERS" not in os.environ


def test_evaluate_branch_default_n_workers_is_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An EvalConfig that does not set n_workers inherits the default 4,
    which evaluate_branch forwards to the env var — backward compatible
    with configs written before this wiring landed."""
    from anvil.eval import runner
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    config = HarnessConfig(
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="correctness", type="llm", weight=1.0)],
            modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
            # n_workers intentionally unset → default 4.
        ),
    )
    _patch_runner_common(monkeypatch, config)
    monkeypatch.delenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", raising=False)

    captured: dict[str, str | None] = {}

    def fake_evaluate(**kwargs: object) -> object:
        captured["workers"] = os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS")
        return SimpleNamespace(result_df=_result_df(), metrics={}, run_id="run-1")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    assert captured["workers"] == "4"
    assert "MLFLOW_GENAI_EVAL_MAX_WORKERS" not in os.environ
