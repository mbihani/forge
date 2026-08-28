"""Tests for the programmatic scorer plugin.

Covers the acceptance contract:

* ``data/evaluator.py`` ships four deterministic check functions
  (``exact_match``, ``must_include_check``, ``json_schema_validity``,
  ``field_exact_match``) — pure, no LLM call, unit-tested in isolation.
* :class:`ScorerConfig` parses ``type`` (``llm``/``programmatic``),
  ``weight``, and ``check_function``; bare-string scorer entries (the
  legacy config shape) coerce to ``type=llm, weight=1.0`` so the shipped
  scaffold keeps scoring identically.
* :func:`build_scorers` builds programmatic scorers by dynamically
  loading ``check_function`` from ``data/evaluator.py`` (importlib).
* The runner aggregates LLM + programmatic scorers via a weighted
  average; uniform weights collapse to the legacy unweighted mean.

No LLM calls and no Databricks calls are made — ``mlflow.genai.evaluate``
and the runtime agent are mocked in the integration test.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_evaluator_module(path: Path | str | None = None) -> ModuleType:
    """Load ``data/evaluator.py`` (or an override) via importlib."""
    resolved = Path(path) if path is not None else REPO_ROOT / "data" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("test_evaluator", resolved)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def evaluator() -> ModuleType:
    return _load_evaluator_module()


# ---------------------------------------------------------------------------
# 1. Built-in check functions (data/evaluator.py) — pure, no mlflow.
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_match(self, evaluator: ModuleType) -> None:
        assert evaluator.exact_match("hello", {"reference_answer": "hello"}) == 1.0

    def test_match_strips_whitespace(self, evaluator: ModuleType) -> None:
        assert evaluator.exact_match("  hello\n", {"reference_answer": "hello"}) == 1.0

    def test_no_match(self, evaluator: ModuleType) -> None:
        assert evaluator.exact_match("world", {"reference_answer": "hello"}) == 0.0

    def test_case_sensitive(self, evaluator: ModuleType) -> None:
        assert evaluator.exact_match("Hello", {"reference_answer": "hello"}) == 0.0

    def test_missing_reference_scores_zero(self, evaluator: ModuleType) -> None:
        assert evaluator.exact_match("anything", {}) == 0.0


class TestMustIncludeCheck:
    def test_all_present(self, evaluator: ModuleType) -> None:
        gt = {"must_include": ["$0.142", "kWh", "11.50"]}
        pred = "The rate is $0.142 per kWh plus an $11.50 charge."
        assert evaluator.must_include_check(pred, gt) == 1.0

    def test_partial(self, evaluator: ModuleType) -> None:
        gt = {"must_include": ["$0.142", "kWh", "11.50"]}
        pred = "The rate is $0.142 per kWh."
        assert evaluator.must_include_check(pred, gt) == pytest.approx(2 / 3)

    def test_none_present(self, evaluator: ModuleType) -> None:
        gt = {"must_include": ["$0.142", "kWh"]}
        assert evaluator.must_include_check("nothing here", gt) == 0.0

    def test_empty_vacuously_passes(self, evaluator: ModuleType) -> None:
        assert evaluator.must_include_check("anything", {"must_include": []}) == 1.0

    def test_missing_key_falls_back_to_expected_facts(self, evaluator: ModuleType) -> None:
        # mlflow-facing alias used by the Correctness judge.
        gt = {"expected_facts": ["kWh"]}
        assert evaluator.must_include_check("priced per kWh", gt) == 1.0


class TestJsonSchemaValidity:
    def test_valid_json(self, evaluator: ModuleType) -> None:
        assert evaluator.json_schema_validity('{"a": 1}', {}) == 1.0

    def test_valid_json_array(self, evaluator: ModuleType) -> None:
        assert evaluator.json_schema_validity("[1, 2, 3]", {}) == 1.0

    def test_invalid_json(self, evaluator: ModuleType) -> None:
        assert evaluator.json_schema_validity("not json", {}) == 0.0

    def test_empty_string_invalid(self, evaluator: ModuleType) -> None:
        assert evaluator.json_schema_validity("", {}) == 0.0

    def test_schema_required_keys_present(self, evaluator: ModuleType) -> None:
        schema = {"type": "object", "required": ["a", "b"]}
        assert evaluator.json_schema_validity('{"a": 1, "b": 2}', {"json_schema": schema}) == 1.0

    def test_schema_missing_required_key(self, evaluator: ModuleType) -> None:
        schema = {"type": "object", "required": ["a", "b"]}
        assert evaluator.json_schema_validity('{"a": 1}', {"json_schema": schema}) == 0.0

    def test_schema_type_mismatch(self, evaluator: ModuleType) -> None:
        schema = {"type": "array"}
        assert evaluator.json_schema_validity('{"a": 1}', {"json_schema": schema}) == 0.0
        assert evaluator.json_schema_validity("[1, 2]", {"json_schema": schema}) == 1.0


class TestFieldExactMatch:
    def test_all_fields_match(self, evaluator: ModuleType) -> None:
        gt = {"expected_fields": {"rate": "$0.142", "unit": "kWh"}}
        pred = '{"rate": "$0.142", "unit": "kWh"}'
        assert evaluator.field_exact_match(pred, gt) == 1.0

    def test_partial_match(self, evaluator: ModuleType) -> None:
        gt = {"expected_fields": {"rate": "$0.142", "unit": "kWh"}}
        pred = '{"rate": "$0.142", "unit": "MWh"}'
        assert evaluator.field_exact_match(pred, gt) == pytest.approx(0.5)

    def test_non_json_prediction(self, evaluator: ModuleType) -> None:
        gt = {"expected_fields": {"rate": "$0.142"}}
        assert evaluator.field_exact_match("not json", gt) == 0.0

    def test_falls_back_to_reference_answer_json(self, evaluator: ModuleType) -> None:
        gt = {"reference_answer": '{"rate": "$0.142", "unit": "kWh"}'}
        pred = '{"rate": "$0.142", "unit": "kWh"}'
        assert evaluator.field_exact_match(pred, gt) == 1.0

    def test_no_expected_fields_passes_for_object(self, evaluator: ModuleType) -> None:
        assert evaluator.field_exact_match('{"a": 1}', {}) == 1.0

    def test_no_expected_fields_fails_for_non_object(self, evaluator: ModuleType) -> None:
        assert evaluator.field_exact_match("[1, 2]", {}) == 0.0

    def test_prediction_not_object_with_expected_fields(self, evaluator: ModuleType) -> None:
        gt = {"expected_fields": {"a": 1}}
        assert evaluator.field_exact_match("[1, 2]", gt) == 0.0


# ---------------------------------------------------------------------------
# 2. Config parsing — ScorerConfig + EvalConfig backward compatibility.
# ---------------------------------------------------------------------------


def test_scorer_config_defaults_to_llm() -> None:
    from anvil.runtime.models import ScorerConfig

    cfg = ScorerConfig(name="correctness")
    assert cfg.type == "llm"
    assert cfg.weight == 1.0
    assert cfg.check_function is None


def test_scorer_config_programmatic_requires_check_function() -> None:
    from pydantic import ValidationError

    from anvil.runtime.models import ScorerConfig

    with pytest.raises(ValidationError, match="check_function"):
        ScorerConfig(name="exact_match", type="programmatic")


def test_scorer_config_llm_forbids_check_function() -> None:
    from pydantic import ValidationError

    from anvil.runtime.models import ScorerConfig

    with pytest.raises(ValidationError, match="must not set check_function"):
        ScorerConfig(name="correctness", type="llm", check_function="exact_match")


def test_scorer_config_rejects_nonpositive_weight() -> None:
    from pydantic import ValidationError

    from anvil.runtime.models import ScorerConfig

    with pytest.raises(ValidationError, match="weight"):
        ScorerConfig(name="x", weight=0.0)
    with pytest.raises(ValidationError, match="weight"):
        ScorerConfig(name="x", weight=-1.0)


def test_scorer_config_rejects_extra_fields() -> None:
    from pydantic import ValidationError

    from anvil.runtime.models import ScorerConfig

    with pytest.raises(ValidationError):
        ScorerConfig(name="x", bogus=True)


def test_eval_config_coerces_legacy_strings() -> None:
    """The shipped config lists scorers as bare strings — each must
    promote to ``ScorerConfig(type=llm, weight=1.0)`` without migration."""
    from anvil.runtime.models import EvalConfig

    cfg = EvalConfig(scorers=["correctness", "retrieval_groundedness", "refusal_appropriateness"])
    assert len(cfg.scorers) == 3
    assert all(s.type == "llm" for s in cfg.scorers)
    assert all(s.weight == 1.0 for s in cfg.scorers)
    assert [s.name for s in cfg.scorers] == [
        "correctness",
        "retrieval_groundedness",
        "refusal_appropriateness",
    ]


def test_eval_config_parses_mixed_dicts() -> None:
    from anvil.runtime.models import EvalConfig

    cfg = EvalConfig(
        scorers=[
            {"name": "correctness", "type": "llm", "weight": 0.4},
            {
                "name": "exact_match",
                "type": "programmatic",
                "check_function": "exact_match",
                "weight": 0.6,
            },
        ]
    )
    assert cfg.scorers[0].type == "llm"
    assert cfg.scorers[0].weight == 0.4
    assert cfg.scorers[1].type == "programmatic"
    assert cfg.scorers[1].check_function == "exact_match"
    assert cfg.scorers[1].weight == 0.6


def test_eval_config_default_scorers_are_three_llm_judges() -> None:
    from anvil.runtime.models import EvalConfig

    cfg = EvalConfig()
    assert [s.name for s in cfg.scorers] == [
        "correctness",
        "retrieval_groundedness",
        "refusal_appropriateness",
    ]
    assert all(s.type == "llm" for s in cfg.scorers)


def test_real_harness_config_scorers_parse_as_llm() -> None:
    """The repo's harness/config.yaml must parse against the updated
    EvalConfig, coercing the bare-string scorers to ScorerConfig(type=llm)."""
    import yaml

    from anvil.runtime.models import RuntimeYAML

    config_path = REPO_ROOT / "harness" / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = RuntimeYAML.model_validate(raw)
    assert [s.name for s in cfg.eval.scorers] == [
        "correctness",
        "retrieval_groundedness",
        "refusal_appropriateness",
    ]
    assert all(s.type == "llm" for s in cfg.eval.scorers)
    assert all(s.weight == 1.0 for s in cfg.eval.scorers)


def test_runtime_yaml_parses_full_mixed_config(tmp_path: Path) -> None:
    """A config that mixes llm + programmatic scorers with weights parses
    end-to-end through RuntimeYAML."""
    import yaml

    from anvil.runtime.models import RuntimeYAML

    config = tmp_path / "config.yaml"
    config.write_text(
        textwrap.dedent(
            """\
            runtime_endpoint: rt
            optimizer_endpoint: op
            judge_endpoint: j
            experiments:
              runtime: r
              eval: e
              optimizer: o
            eval:
              scorers:
                - name: correctness
                  type: llm
                  weight: 0.4
                - name: exact_match
                  type: programmatic
                  check_function: exact_match
                  weight: 0.6
            """
        ),
        encoding="utf-8",
    )
    cfg = RuntimeYAML.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    assert cfg.eval.scorers[0] == type(cfg.eval.scorers[0])(
        name="correctness", type="llm", weight=0.4, check_function=None
    )
    assert cfg.eval.scorers[1].name == "exact_match"
    assert cfg.eval.scorers[1].type == "programmatic"
    assert cfg.eval.scorers[1].check_function == "exact_match"
    assert cfg.eval.scorers[1].weight == 0.6


# ---------------------------------------------------------------------------
# 3. Evaluator loader (importlib) — build-time, before any row is scored.
# ---------------------------------------------------------------------------


def test_load_evaluator_module_missing_file_raises(tmp_path: Path) -> None:
    from anvil.eval.scorers import load_evaluator_module

    with pytest.raises(FileNotFoundError, match="evaluator module not found"):
        load_evaluator_module(tmp_path / "nope.py")


def test_load_check_function_resolves_callable(tmp_path: Path) -> None:
    from anvil.eval.scorers import load_check_function

    (tmp_path / "evaluator.py").write_text(
        "def my_check(prediction, ground_truth):\n    return 1.0\n", encoding="utf-8"
    )
    fn = load_check_function("my_check", tmp_path / "evaluator.py")
    assert callable(fn)
    assert fn("x", {}) == 1.0


def test_load_check_function_missing_name_raises(tmp_path: Path) -> None:
    from anvil.eval.scorers import load_check_function

    (tmp_path / "evaluator.py").write_text(
        "def other(prediction, ground_truth):\n    return 0.0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not found"):
        load_check_function("nope", tmp_path / "evaluator.py")


def test_load_check_function_none_name_raises() -> None:
    from anvil.eval.scorers import load_check_function

    with pytest.raises(ValueError, match="required"):
        load_check_function(None)


# ---------------------------------------------------------------------------
# 4. Programmatic scorer builder — wraps a check function in @scorer.
# ---------------------------------------------------------------------------


def test_build_programmatic_scorer_returns_check_value() -> None:
    from anvil.eval.scorers import build_programmatic_scorer

    def my_check(prediction: str, ground_truth: dict) -> float:
        return 1.0 if prediction == ground_truth.get("reference_answer") else 0.0

    scorer_fn = build_programmatic_scorer(name="my", check_fn=my_check)
    assert scorer_fn.name == "my"
    assert type(scorer_fn).__name__ == "CustomScorer"

    hit = scorer_fn(
        inputs={"query": "q"},
        outputs="hello",
        expectations={"reference_answer": "hello"},
    )
    assert hit.value == 1.0

    miss = scorer_fn(
        inputs={"query": "q"},
        outputs="world",
        expectations={"reference_answer": "hello"},
    )
    assert miss.value == 0.0


def test_build_programmatic_scorer_clamps_out_of_range() -> None:
    from anvil.eval.scorers import build_programmatic_scorer

    def over(prediction: str, ground_truth: dict) -> float:
        return 1.5

    def under(prediction: str, ground_truth: dict) -> float:
        return -0.2

    over_fn = build_programmatic_scorer(name="over", check_fn=over)
    under_fn = build_programmatic_scorer(name="under", check_fn=under)
    assert over_fn(inputs={}, outputs="x", expectations={}).value == 1.0
    assert under_fn(inputs={}, outputs="x", expectations={}).value == 0.0


def test_build_programmatic_scorer_none_outputs_becomes_empty_string() -> None:
    from anvil.eval.scorers import build_programmatic_scorer

    def echo_len(prediction: str, ground_truth: dict) -> float:
        return 1.0 if prediction == "" else 0.0

    fn = build_programmatic_scorer(name="empty", check_fn=echo_len)
    assert fn(inputs={}, outputs=None, expectations={}).value == 1.0


# ---------------------------------------------------------------------------
# 5. build_scorers — mixed llm + programmatic configs.
# ---------------------------------------------------------------------------


def test_build_scorers_mixed_llm_and_programmatic(tmp_path: Path) -> None:
    from anvil.eval.scorers import build_scorers
    from anvil.runtime.models import ScorerConfig

    (tmp_path / "evaluator.py").write_text(
        textwrap.dedent(
            """\
            def exact_match(prediction, ground_truth):
                ref = ground_truth.get("reference_answer")
                return 1.0 if ref is not None and prediction.strip() == str(ref).strip() else 0.0
            """
        ),
        encoding="utf-8",
    )
    configs = [
        ScorerConfig(name="correctness", type="llm", weight=0.5),
        ScorerConfig(
            name="exact_match", type="programmatic", check_function="exact_match", weight=0.5
        ),
    ]
    scorers = build_scorers(
        judge_client=None,
        scorer_configs=configs,
        evaluator_path=tmp_path / "evaluator.py",
    )
    assert len(scorers) == 2
    by_name = {s.name: s for s in scorers}
    assert set(by_name) == {"correctness", "exact_match"}
    # The programmatic scorer is the CustomScorer and runs the check fn.
    prog = by_name["exact_match"]
    assert type(prog).__name__ == "CustomScorer"
    hit = prog(inputs={}, outputs="hello", expectations={"reference_answer": "hello"})
    assert hit.value == 1.0
    miss = prog(inputs={}, outputs="world", expectations={"reference_answer": "hello"})
    assert miss.value == 0.0


def test_build_scorers_defaults_to_three_llm_judges() -> None:
    from anvil.eval.scorers import build_scorers

    scorers = build_scorers(judge_client=None)
    assert {s.name for s in scorers} == {
        "correctness",
        "retrieval_groundedness",
        "refusal_appropriateness",
    }


def test_build_scorers_unknown_llm_name_raises() -> None:
    from anvil.eval.scorers import build_scorers
    from anvil.runtime.models import ScorerConfig

    with pytest.raises(ValueError, match="unknown llm scorer name"):
        build_scorers(judge_client=None, scorer_configs=[ScorerConfig(name="bogus")])


def test_build_scorers_programmatic_missing_function_raises(tmp_path: Path) -> None:
    from anvil.eval.scorers import build_scorers
    from anvil.runtime.models import ScorerConfig

    (tmp_path / "evaluator.py").write_text("# empty\n", encoding="utf-8")
    configs = [ScorerConfig(name="x", type="programmatic", check_function="missing")]
    with pytest.raises(ValueError, match="not found"):
        build_scorers(
            judge_client=None, scorer_configs=configs, evaluator_path=tmp_path / "evaluator.py"
        )


# ---------------------------------------------------------------------------
# 6. Runner aggregate — weighted average across llm + programmatic.
# ---------------------------------------------------------------------------


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


def test_aggregate_report_weighted_average() -> None:
    from anvil.eval.runner import _aggregate_report

    df = pd.DataFrame(
        {
            "correctness/value": [0.5, 1.0],
            "exact_match/value": [1.0, 0.0],
            "trace_id": ["t0", "t1"],
        }
    )
    examples = [_gold("g1", "hello"), _gold("g2", "world")]
    report = _aggregate_report(
        result_df=df,
        metrics={},
        scorer_names=["correctness", "exact_match"],
        aggregate_scorer_names=["correctness", "exact_match"],
        weights={"correctness": 0.4, "exact_match": 0.6},
        examples=examples,
        run_id="run-1",
        experiment_id="exp-1",
        mode="quick",
    )
    # correctness mean = 0.75 (w=0.4); exact_match mean = 0.5 (w=0.6).
    # weighted = (0.75*0.4 + 0.5*0.6) / 1.0 = 0.6
    assert report.aggregate == pytest.approx(0.6)
    assert report.per_judge["correctness"] == pytest.approx(0.75)
    assert report.per_judge["exact_match"] == pytest.approx(0.5)
    assert set(report.scorers) == {"correctness", "exact_match"}
    assert report.n_rows == 2
    assert report.mode == "quick"
    assert report.cost_metrics == {
        "total_context_chars": float(len("q-g1") + len("q-g2")),
        "n_rows": 2.0,
    }


def test_aggregate_report_uniform_weights_is_unweighted_mean() -> None:
    """Backward compatibility: uniform weights collapse to the legacy
    unweighted mean (sum / count)."""
    from anvil.eval.runner import _aggregate_report

    df = pd.DataFrame(
        {
            "correctness/value": [1.0, 0.0],
            "retrieval_groundedness/value": [0.8, 0.8],
        }
    )
    report = _aggregate_report(
        result_df=df,
        metrics={},
        scorer_names=["correctness", "retrieval_groundedness"],
        aggregate_scorer_names=["correctness", "retrieval_groundedness"],
        weights={"correctness": 1.0, "retrieval_groundedness": 1.0},
        examples=[_gold("g1", "a"), _gold("g2", "b")],
        run_id="run-1",
        experiment_id="exp-1",
        mode="quick",
    )
    # correctness mean = 0.5, groundedness mean = 0.8 → mean = 0.65.
    assert report.aggregate == pytest.approx(0.65)


def test_aggregate_report_safety_excluded_from_aggregate() -> None:
    """Safety is a guard-only scorer: it appears in per_judge / scorer
    names but NOT in the aggregate (it is absent from the configured
    scorers, so it has no weight)."""
    from anvil.eval.runner import _aggregate_report

    df = pd.DataFrame(
        {
            "correctness/value": [1.0, 1.0],
            "safety/value": [1.0, 1.0],
        }
    )
    report = _aggregate_report(
        result_df=df,
        metrics={},
        scorer_names=["correctness", "safety"],
        aggregate_scorer_names=["correctness"],
        weights={"correctness": 1.0},
        examples=[_gold("g1", "a"), _gold("g2", "b")],
        run_id="run-1",
        experiment_id="exp-1",
        mode="quick",
    )
    assert report.aggregate == pytest.approx(1.0)
    assert "safety" in report.per_judge
    assert report.per_judge["safety"] == 1.0


# ---------------------------------------------------------------------------
# 7. Runner integration — evaluate_branch with mlflow.genai.evaluate mocked.
# ---------------------------------------------------------------------------


def test_evaluate_branch_mixed_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end (mocked): a config mixing one llm + one programmatic
    scorer produces a weighted aggregate. ``mlflow.genai.evaluate`` is
    mocked to return a fixed result_df; the programmatic check function
    is loaded from a temp evaluator. No LLM call is made."""
    from anvil.eval import runner
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    eval_file = tmp_path / "evaluator.py"
    eval_file.write_text(
        textwrap.dedent(
            """\
            def exact_match(prediction, ground_truth):
                ref = ground_truth.get("reference_answer")
                return 1.0 if ref is not None and prediction.strip() == str(ref).strip() else 0.0
            """
        ),
        encoding="utf-8",
    )

    eval_cfg = EvalConfig(
        default_mode="quick",
        scorers=[
            ScorerConfig(name="correctness", type="llm", weight=0.4),
            ScorerConfig(
                name="exact_match", type="programmatic", check_function="exact_match", weight=0.6
            ),
        ],
        modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
    )
    config = HarnessConfig(
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=eval_cfg,
    )
    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(
        runner, "load_golden_set", lambda _p: [_gold("g1", "hello"), _gold("g2", "world")]
    )
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "AnvilAgent", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "enable_runtime_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "get_experiment_by_name", lambda *a, **kw: None)

    df = pd.DataFrame(
        {
            "correctness/value": [0.5, 1.0],
            "exact_match/value": [1.0, 0.0],
            "trace_id": ["t0", "t1"],
        }
    )
    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(result_df=df, metrics={}, run_id="run-1")

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    report = runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        golden_set_path="unused",
        evaluator_path=eval_file,
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    # Two scorers were built and forwarded to mlflow.genai.evaluate.
    forwarded = captured["scorers"]
    assert len(forwarded) == 2
    assert {s.name for s in forwarded} == {"correctness", "exact_match"}
    assert any(type(s).__name__ == "CustomScorer" for s in forwarded)

    # Weighted aggregate matches the per-row means via the weights.
    assert report.aggregate == pytest.approx(0.6)
    assert report.per_judge["correctness"] == pytest.approx(0.75)
    assert report.per_judge["exact_match"] == pytest.approx(0.5)
    assert set(report.scorers) == {"correctness", "exact_match"}
    assert report.n_rows == 2
    assert report.mode == "quick"


def test_evaluate_branch_backward_compat_string_scorers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config that lists scorers as bare strings (legacy) scores with
    the unweighted mean — identical to the pre-plugin behavior."""
    from anvil.eval import runner
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
    )

    eval_cfg = EvalConfig(
        default_mode="quick",
        scorers=["correctness", "retrieval_groundedness"],
        modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
    )
    config = HarnessConfig(
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=eval_cfg,
    )
    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(runner, "load_golden_set", lambda _p: [_gold("g1", "a"), _gold("g2", "b")])
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "AnvilAgent", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "enable_runtime_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "get_experiment_by_name", lambda *a, **kw: None)

    df = pd.DataFrame(
        {
            "correctness/value": [1.0, 0.0],
            "retrieval_groundedness/value": [0.8, 0.8],
            "trace_id": ["t0", "t1"],
        }
    )
    monkeypatch.setattr(
        runner.mlflow.genai,
        "evaluate",
        lambda **kw: SimpleNamespace(result_df=df, metrics={}, run_id="run-1"),
    )

    report = runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )
    # Uniform weights → unweighted mean = (0.5 + 0.8) / 2 = 0.65.
    assert report.aggregate == pytest.approx(0.65)
    assert set(report.scorers) == {"correctness", "retrieval_groundedness"}


# ---------------------------------------------------------------------------
# 8. Cross-review fixes — duplicate names, NaN clamping, field projection.
# ---------------------------------------------------------------------------


def test_eval_config_rejects_duplicate_scorer_names() -> None:
    """Duplicate scorer names overwrite weights in the aggregate dict
    while remaining duplicated in the numerator/denominator — reject at
    config validation time."""
    from pydantic import ValidationError

    from anvil.runtime.models import EvalConfig

    with pytest.raises(ValidationError, match="duplicate scorer name"):
        EvalConfig(
            scorers=[
                {"name": "correctness", "type": "llm", "weight": 0.4},
                {"name": "correctness", "type": "llm", "weight": 0.6},
            ]
        )


def test_eval_config_rejects_duplicate_scorer_names_three_wide() -> None:
    """A three-way duplicate is also rejected — the validator finds the
    second occurrence."""
    from pydantic import ValidationError

    from anvil.runtime.models import EvalConfig

    with pytest.raises(ValidationError, match="duplicate scorer name"):
        EvalConfig(scorers=["correctness", "correctness", "correctness"])


def test_eval_config_allows_unique_scorer_names() -> None:
    """Distinct names parse without error (sanity check — the validator
    does not false-positive on valid configs)."""
    from anvil.runtime.models import EvalConfig

    cfg = EvalConfig(
        scorers=[
            {"name": "correctness", "type": "llm", "weight": 0.4},
            {
                "name": "exact_match",
                "type": "programmatic",
                "check_function": "exact_match",
                "weight": 0.6,
            },
        ]
    )
    assert len(cfg.scorers) == 2


def test_clamp_score_nan_returns_zero() -> None:
    """``_clamp_score(float('nan'))`` must return 0.0, not NaN — NaN
    comparisons are always False in Python, so without an ``isfinite``
    guard the NaN passes both the ``< 0.0`` and ``> 1.0`` checks and
    leaks into the aggregate."""
    from anvil.eval.scorers import _clamp_score

    assert _clamp_score(float("nan")) == 0.0


def test_clamp_score_positive_inf_returns_zero() -> None:
    """``_clamp_score(float('inf'))`` must not return inf — a non-finite
    value must be mapped to 0.0 so it cannot poison the aggregate."""
    from anvil.eval.scorers import _clamp_score

    assert _clamp_score(float("inf")) == 0.0


def test_clamp_score_negative_inf_returns_zero() -> None:
    from anvil.eval.scorers import _clamp_score

    assert _clamp_score(float("-inf")) == 0.0


def test_build_programmatic_scorer_clamps_nan_to_zero() -> None:
    """A check function returning NaN must produce a Feedback value of
    0.0, not NaN — this is the end-to-end contract that _clamp_score
    enforces inside _run_programmatic_check."""
    from anvil.eval.scorers import build_programmatic_scorer

    def nan_check(prediction: str, ground_truth: dict) -> float:
        return float("nan")

    fn = build_programmatic_scorer(name="nan", check_fn=nan_check)
    result = fn(inputs={}, outputs="x", expectations={})
    assert result.value == 0.0


def test_build_programmatic_scorer_clamps_inf_to_zero() -> None:
    from anvil.eval.scorers import build_programmatic_scorer

    def inf_check(prediction: str, ground_truth: dict) -> float:
        return float("inf")

    fn = build_programmatic_scorer(name="inf", check_fn=inf_check)
    result = fn(inputs={}, outputs="x", expectations={})
    assert result.value == 0.0


def test_build_dataset_projects_json_schema() -> None:
    """``json_schema`` in a golden-set example must flow through to the
    row's ``expectations`` dict so ``json_schema_validity`` receives its
    documented primary input."""
    from anvil.eval.runner import _build_dataset

    examples = [
        {
            "example_id": "g1",
            "query": "q1",
            "category": "direct",
            "expected_doc_ids": [],
            "reference_answer": "x",
            "should_refuse": False,
            "expected_citations": [],
            "must_include": ["x"],
            "must_not_include": [],
            "notes_for_judge": "",
            "json_schema": {"type": "object", "required": ["rate"]},
        }
    ]
    rows = _build_dataset(examples)
    assert rows[0]["expectations"]["json_schema"] == {"type": "object", "required": ["rate"]}


def test_build_dataset_projects_expected_fields() -> None:
    """``expected_fields`` in a golden-set example must flow through to
    the row's ``expectations`` dict so ``field_exact_match`` receives
    its documented primary input."""
    from anvil.eval.runner import _build_dataset

    examples = [
        {
            "example_id": "g1",
            "query": "q1",
            "category": "direct",
            "expected_doc_ids": [],
            "reference_answer": '{"rate": "0.142"}',
            "should_refuse": False,
            "expected_citations": [],
            "must_include": ["0.142"],
            "must_not_include": [],
            "notes_for_judge": "",
            "expected_fields": {"rate": "0.142", "unit": "kWh"},
        }
    ]
    rows = _build_dataset(examples)
    assert rows[0]["expectations"]["expected_fields"] == {"rate": "0.142", "unit": "kWh"}


def test_build_dataset_projects_arbitrary_json_prefixed_extension_fields() -> None:
    """Any key prefixed with ``json_`` or ``expected_`` that is not
    already in the expectations dict is passed through as an extension
    field."""
    from anvil.eval.runner import _build_dataset

    examples = [
        {
            "example_id": "g1",
            "query": "q1",
            "category": "direct",
            "expected_doc_ids": [],
            "reference_answer": "x",
            "should_refuse": False,
            "expected_citations": [],
            "must_include": ["x"],
            "must_not_include": [],
            "notes_for_judge": "",
            "json_custom_validator": "my_rules",
            "expected_confidence": 0.95,
        }
    ]
    rows = _build_dataset(examples)
    assert rows[0]["expectations"]["json_custom_validator"] == "my_rules"
    assert rows[0]["expectations"]["expected_confidence"] == 0.95


def test_build_dataset_without_extension_fields_omits_them() -> None:
    """When the golden-set example has no json_schema / expected_fields /
    extension fields, the expectations dict is unchanged (additive contract)."""
    from anvil.eval.runner import _build_dataset

    examples = [
        {
            "example_id": "g1",
            "query": "q1",
            "category": "direct",
            "expected_doc_ids": [],
            "reference_answer": "x",
            "should_refuse": False,
            "expected_citations": [],
            "must_include": ["x"],
            "must_not_include": [],
            "notes_for_judge": "",
        }
    ]
    rows = _build_dataset(examples)
    assert "json_schema" not in rows[0]["expectations"]
    assert "expected_fields" not in rows[0]["expectations"]
