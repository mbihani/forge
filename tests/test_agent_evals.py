"""Offline tests for reviewing + re-using the agent's existing evals.

No tracking server: traces are SimpleNamespace stand-ins carrying real
MLflow ``Feedback`` assessments, scorer payloads are built from real MLflow
scorers (then given a field a newer MLflow writes), and judges are fakes
injected into :class:`AgentJudge`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.entities import AssessmentSource, Feedback

from anvil.eval.agent_evals import (
    AgentEvalsNotReady,
    AgentJudge,
    JudgeSpec,
    deserialize_scorer,
    discover_agent_evals,
    ensure_agent_evals_ready,
    fetch_registered_scorers,
    load_agent_judges,
    read_agent_evals,
    render_review,
    score_with_agent_judges,
    summarize_assessments,
    write_agent_evals,
)
from anvil.loop.builder import _format_agent_evals
from anvil.runtime.models import AgentEvalsConfig, UserMetric


def _fb(name, value, source_type="LLM_JUDGE", source_id="databricks", rationale="", metadata=None):
    return Feedback(
        name=name,
        value=value,
        rationale=rationale,
        source=AssessmentSource(source_type=source_type, source_id=source_id),
        metadata=metadata,
    )


def _trace(*assessments, request=None, response=None):
    return SimpleNamespace(
        info=SimpleNamespace(assessments=list(assessments)),
        data=SimpleNamespace(request=request, response=response),
    )


def _guidelines_payload(name="summary_quality", guidelines=("Must not omit sections.",)):
    from mlflow.genai.scorers import Guidelines

    payload = Guidelines(name=name, guidelines=list(guidelines)).model_dump()
    # A newer MLflow writes fields this version does not know.
    payload["timeout"] = 0
    payload["third_party_scorer_data"] = None
    return payload


TRACES = [
    _trace(
        _fb(
            "summary_quality",
            "no",
            rationale="omits section (b)",
            metadata={"guideline": "Must not omit sections.\nNo fabrication."},
        ),
        _fb("output_json_valid", False, source_type="CODE", source_id="default", rationale="bad"),
        _fb("thumbs", "no", source_type="HUMAN", source_id="sme@x", rationale="too terse"),
        request='{"pdf_bytes": "...", "filename": "a.pdf"}',
        response=[{"document_title": "x"}],
    ),
    _trace(
        _fb("summary_quality", "yes"),
        _fb("output_json_valid", False, source_type="CODE", source_id="default", rationale="bad"),
        _fb(
            "issue-1",
            True,
            source_type="CODE",
            source_id="mlflow.issue_detection",
            rationale="ids inconsistent",
            metadata={"mlflow.issue.name": "Inconsistent ids", "mlflow.issue.severity": "low"},
        ),
    ),
]


# ----------------------------------------------------------------- discovery


def test_summarize_assessments_scores_humans_issues_and_guidelines() -> None:
    summaries, issues, guidelines, io_shape, n_with = summarize_assessments(TRACES)
    by = {(s.name, s.source_type): s for s in summaries}
    assert n_with == 2
    assert by[("summary_quality", "LLM_JUDGE")].mean == pytest.approx(0.5)
    assert by[("summary_quality", "LLM_JUDGE")].failing_rationales == ["omits section (b)"]
    # Duplicate rationales are recorded once.
    assert by[("output_json_valid", "CODE")].failing_rationales == ["bad"]
    assert by[("thumbs", "HUMAN")].failing_rationales == ["too terse"]
    # Issue-detection findings are not scored assessments.
    assert ("issue-1", "CODE") not in by
    assert issues == [
        {"name": "Inconsistent ids", "severity": "low", "rationale": "ids inconsistent", "n": 1}
    ]
    assert guidelines == {"summary_quality": ["Must not omit sections.", "No fabrication."]}
    assert io_shape == {"request": ["filename", "pdf_bytes"], "response": "list[dict]"}


def test_discover_prefers_registered_scorers_over_assessment_guidelines() -> None:
    inv = discover_agent_evals(
        "123", traces=TRACES, scorer_payloads=[_guidelines_payload("summary_quality")]
    )
    assert inv.judge_names() == ["summary_quality"]
    assert inv.judges[0].source == "registered_scorer"
    assert inv.judges[0].kind == "guidelines"


def test_discover_rebuilds_guidelines_judge_when_not_registered() -> None:
    inv = discover_agent_evals("123", traces=TRACES, scorer_payloads=[])
    (judge,) = inv.judges
    assert (judge.name, judge.source, judge.kind) == (
        "summary_quality",
        "assessment_guideline",
        "guidelines",
    )
    assert judge.guidelines == ["Must not omit sections.", "No fabrication."]


def test_fetch_registered_scorers_falls_through_and_records_warnings() -> None:
    warnings: list[str] = []

    def broken(_exp):
        raise RuntimeError("not databricks")

    def ok(_exp):
        return [{"name": "safety"}]

    assert fetch_registered_scorers("1", fetchers=[broken, ok], warnings=warnings) == [
        {"name": "safety"}
    ]
    assert "not databricks" in warnings[0]
    assert fetch_registered_scorers("1", fetchers=[broken], warnings=[]) == []


def test_deserialize_scorer_drops_fields_from_newer_mlflow() -> None:
    scorer = deserialize_scorer(_guidelines_payload())
    assert type(scorer).__name__ == "Guidelines"
    assert scorer.name == "summary_quality"


# -------------------------------------------------------------------- review


def test_review_flags_no_signal_judges_and_humans() -> None:
    inv = discover_agent_evals("123", traces=TRACES, scorer_payloads=[_guidelines_payload()])
    md = render_review(inv)
    assert "Judges forge re-uses" in md and "`summary_quality`" in md
    # output_json_valid fails on both traces → flagged as possibly stale.
    assert "`output_json_valid` fails on every trace (n=2)" in md
    assert "too terse" in md  # human feedback surfaced
    assert "Inconsistent ids" in md  # issue surfaced
    assert "`['filename', 'pdf_bytes']`" in md  # io shape for faithful replay


def test_review_without_judges_tells_operator_to_ask_for_metrics() -> None:
    inv = discover_agent_evals("123", traces=[], scorer_payloads=[])
    assert "No judges found" in render_review(inv)


def test_write_read_roundtrip_and_prompt_block(tmp_path: Path) -> None:
    inv = discover_agent_evals("123", traces=TRACES, scorer_payloads=[_guidelines_payload()])
    write_agent_evals(tmp_path, inv)
    back = read_agent_evals(tmp_path)
    assert back is not None and back.judge_names() == ["summary_quality"]
    assert back.fingerprint() == inv.fingerprint()
    assert "Judges forge re-uses" in _format_agent_evals(tmp_path)
    assert "no reviewed agent evals" in _format_agent_evals(tmp_path / "missing")


# -------------------------------------------------------------------- judges


class _FakeScorer:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.value, Exception):
            raise self.value
        return Feedback(name="j", value=self.value, rationale="why")


def test_agent_judge_scores_and_normalizes() -> None:
    spec = JudgeSpec(name="q", source="registered_scorer", kind="guidelines")
    assert AgentJudge(spec, scorer=_FakeScorer("yes")).score(outputs="o") == (1.0, "why")
    assert AgentJudge(spec, scorer=_FakeScorer(False)).score(outputs="o")[0] == 0.0
    assert AgentJudge(spec, scorer=_FakeScorer(0.4)).score(outputs="o")[0] == pytest.approx(0.4)


def test_score_with_agent_judges_failure_counts_as_zero() -> None:
    good = AgentJudge(JudgeSpec("a", "registered_scorer", "builtin"), scorer=_FakeScorer("yes"))
    bad = AgentJudge(
        JudgeSpec("b", "registered_scorer", "builtin"), scorer=_FakeScorer(RuntimeError("x"))
    )
    assert score_with_agent_judges([good, bad], inputs={"q": 1}, outputs="o") == {
        "a": 1.0,
        "b": 0.0,
    }
    assert good._scorer.calls[0]["inputs"] == {"q": 1}


def test_load_agent_judges_subset_and_unknown_name(tmp_path: Path) -> None:
    inv = discover_agent_evals("123", traces=TRACES, scorer_payloads=[_guidelines_payload()])
    write_agent_evals(tmp_path, inv)
    assert [j.name for j in load_agent_judges(tmp_path, names=["summary_quality"])] == [
        "summary_quality"
    ]
    with pytest.raises(ValueError, match="not found"):
        load_agent_judges(tmp_path, names=["nope"])
    assert load_agent_judges(tmp_path / "missing") == []


def test_guidelines_judge_rebuilt_from_assessment_metadata() -> None:
    spec = JudgeSpec(
        name="summary_quality",
        source="assessment_guideline",
        kind="guidelines",
        guidelines=["Must not omit sections."],
    )
    assert type(AgentJudge(spec).scorer).__name__ == "Guidelines"


# ---------------------------------------------------------------------- gate


def _cfg(**kw):
    return SimpleNamespace(agent_evals=AgentEvalsConfig(**kw) if kw is not None else None)


def test_gate_requires_agent_evals_section(tmp_path: Path) -> None:
    with pytest.raises(AgentEvalsNotReady, match="EXISTING evals"):
        ensure_agent_evals_ready(SimpleNamespace(agent_evals=None), tmp_path)


def test_gate_requires_review_for_the_configured_experiment(tmp_path: Path) -> None:
    with pytest.raises(AgentEvalsNotReady, match="have not been reviewed"):
        ensure_agent_evals_ready(_cfg(experiment_id="123"), tmp_path)
    write_agent_evals(tmp_path, discover_agent_evals("999", traces=TRACES, scorer_payloads=[]))
    with pytest.raises(AgentEvalsNotReady, match="have not been reviewed"):
        ensure_agent_evals_ready(_cfg(experiment_id="123"), tmp_path)


def test_gate_passes_with_reviewed_judges(tmp_path: Path) -> None:
    write_agent_evals(
        tmp_path,
        discover_agent_evals("123", traces=TRACES, scorer_payloads=[_guidelines_payload()]),
    )
    inv = ensure_agent_evals_ready(_cfg(experiment_id="123"), tmp_path)
    assert inv is not None and inv.judge_names() == ["summary_quality"]
    with pytest.raises(ValueError, match="not found"):
        ensure_agent_evals_ready(_cfg(experiment_id="123", judges=["nope"]), tmp_path)


def test_gate_no_judges_requires_user_metrics(tmp_path: Path) -> None:
    write_agent_evals(tmp_path, discover_agent_evals("123", traces=[], scorer_payloads=[]))
    with pytest.raises(AgentEvalsNotReady, match="user_metrics"):
        ensure_agent_evals_ready(_cfg(experiment_id="123"), tmp_path)
    metrics = [UserMetric(name="accuracy", description="d")]
    assert (
        ensure_agent_evals_ready(_cfg(experiment_id="123", user_metrics=metrics), tmp_path)
        is not None
    )
    # No experiment at all: only agreed user metrics let the run proceed.
    with pytest.raises(AgentEvalsNotReady, match="neither"):
        ensure_agent_evals_ready(_cfg(), tmp_path)
    assert ensure_agent_evals_ready(_cfg(user_metrics=metrics), tmp_path) is None


def test_shipped_demo_config_passes_gate() -> None:
    from anvil.runtime.loader import load_harness

    repo = Path(__file__).resolve().parent.parent
    cfg = load_harness(repo / "scaffold").config.eval
    assert ensure_agent_evals_ready(cfg, repo) is None
    assert [m.name for m in cfg.agent_evals.user_metrics] == [
        "correctness",
        "retrieval_groundedness",
        "refusal_appropriateness",
    ]


def test_inventory_json_is_plain_json(tmp_path: Path) -> None:
    inv = discover_agent_evals("123", traces=TRACES, scorer_payloads=[_guidelines_payload()])
    inv_path, _ = write_agent_evals(tmp_path, inv)
    raw = json.loads(inv_path.read_text())
    assert raw["experiment_id"] == "123"
    assert raw["judges"][0]["payload"]["name"] == "summary_quality"


# ------------------------------------------------------------- integrations


def test_trace_engine_judge_fn_prefers_agent_judge() -> None:
    from anvil.domains.trace.eval import build_judge_fn

    fake = _FakeScorer("no")
    judge = AgentJudge(JudgeSpec("summary_quality", "registered_scorer", "guidelines"), scorer=fake)

    class _NoClient:  # the prompt-rebuilt path must not be taken
        chat = None

    judge_fn = build_judge_fn(_NoClient(), "fallback", agent_judges=[judge])
    assert judge_fn(query="q", output="o", rubric={"name": "summary_quality"}) == 0.0
    assert fake.calls[0]["inputs"] == {"query": "q"} and fake.calls[0]["outputs"] == "o"


def test_round_prompt_carries_the_review(tmp_path: Path) -> None:
    from anvil.loop.builder import build_round_prompt

    write_agent_evals(
        tmp_path,
        discover_agent_evals("123", traces=TRACES, scorer_payloads=[_guidelines_payload()]),
    )
    prompt = build_round_prompt(repo_root=tmp_path, round_id=1, baseline=None)
    assert "What the agent's existing evals say" in prompt
    assert "omits section (b)" in prompt
    assert "agent's OWN judges" in prompt


def test_orchestrator_agent_evals_check(tmp_path: Path) -> None:
    from anvil.orchestrator import app as app_module

    assert app_module._check_agent_evals(tmp_path, {"eval": {}})["status"] == "fail"
    ok = app_module._check_agent_evals(
        tmp_path, {"eval": {"agent_evals": {"user_metrics": [{"name": "accuracy"}]}}}
    )
    assert ok["status"] == "pass" and "agreed with the user" in ok["message"]
    write_agent_evals(
        tmp_path,
        discover_agent_evals("123", traces=TRACES, scorer_payloads=[_guidelines_payload()]),
    )
    reused = app_module._check_agent_evals(
        tmp_path, {"eval": {"agent_evals": {"experiment_id": "123"}}}
    )
    assert reused["status"] == "pass" and "summary_quality" in reused["message"]


def test_runner_uses_agent_judges_only_with_an_experiment(tmp_path: Path) -> None:
    from anvil.eval.runner import _agent_judges_for
    from anvil.runtime.models import EvalConfig

    write_agent_evals(
        tmp_path,
        discover_agent_evals("123", traces=TRACES, scorer_payloads=[_guidelines_payload()]),
    )
    no_exp = EvalConfig(agent_evals=AgentEvalsConfig(user_metrics=[UserMetric(name="a")]))
    assert _agent_judges_for(no_exp, tmp_path) == []
    with_exp = EvalConfig(agent_evals=AgentEvalsConfig(experiment_id="123"))
    assert [j.name for j in _agent_judges_for(with_exp, tmp_path)] == ["summary_quality"]
