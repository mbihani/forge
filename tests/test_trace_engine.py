"""Offline unit tests for the ``trace`` eval engine (Option A).

No live Databricks workspace, no gateway, no MLflow tracking server: the
ingest path is exercised against hand-built ``Trace`` /
:class:`~mlflow.entities.Feedback` / :class:`~mlflow.entities.Expectation`
fixtures, and the scoring/aggregation path with an injected judge function.

Covers:

* ``trace_to_row`` splits HUMAN vs LLM_JUDGE vs AI_JUDGE and Expectation
  vs Feedback into the right buckets;
* traces with zero assessments are dropped and multi-turn / tool-laden
  requests are skipped and counted;
* the JSONL snapshot round-trips (write then read);
* the weighted aggregate ranks human objectives above judge objectives;
* the ``scorer_fingerprint`` changes when the snapshot content changes;
* the judge client is injectable and falls back off an unreachable model;
* the engine self-registers under ``"trace"``.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from mlflow.entities import AssessmentSource, Expectation, Feedback

from anvil.domains.trace.eval import (
    aggregate_report,
    build_judge_fn,
    build_key_map,
    compute_trace_fingerprint,
    compute_weights,
    hard_label_match,
    objective_keys,
    reference_match,
    score_row,
)
from anvil.domains.trace.ingest import (
    build_snapshot_rows,
    parse_single_turn_query,
    read_snapshot,
    snapshot_content_hash,
    trace_to_row,
    write_snapshot,
)

_HUMAN = AssessmentSource(source_type="HUMAN", source_id="alice")
_JUDGE = AssessmentSource(source_type="LLM_JUDGE", source_id="databricks-claude-sonnet-4-6")


def _mk_trace(trace_id: str, request, assessments: list) -> SimpleNamespace:
    """A duck-typed Trace fixture — ingest reads only these attributes."""
    return SimpleNamespace(
        info=SimpleNamespace(trace_id=trace_id, assessments=assessments),
        data=SimpleNamespace(request=request, response=None),
    )


def _ai_feedback(name: str, value, *, source_id: str = "ai-model") -> Feedback:
    """A real Feedback whose source stays AI_JUDGE (the SDK normalizes the
    canonical AssessmentSource(AI_JUDGE) to LLM_JUDGE, so we set a
    duck-typed source to exercise the distinct AI_JUDGE branch)."""
    fb = Feedback(name=name, value=value, source=_HUMAN, rationale="ai note")
    fb.source = SimpleNamespace(source_type="AI_JUDGE", source_id=source_id)
    return fb


# ---------------------------------------------------------------------------
# ingest: assessment splitting
# ---------------------------------------------------------------------------


def test_trace_to_row_splits_assessment_kinds() -> None:
    trace = _mk_trace(
        "tr-1",
        '{"messages": [{"role": "user", "content": "What is X?"}]}',
        [
            Expectation(name="answer", value="42", source=_HUMAN),
            Feedback(name="helpful", value="yes", source=_HUMAN, rationale="good"),
            Feedback(name="correctness", value="pass", source=_JUDGE, rationale="right"),
            _ai_feedback("relevance", 0.9),
        ],
    )
    row = trace_to_row(trace)

    assert row is not None
    assert row["example_id"] == "tr-1"
    assert row["query"] == "What is X?"
    # HUMAN Expectation -> expectations
    assert [e["name"] for e in row["expectations"]] == ["answer"]
    assert row["expectations"][0]["source_id"] == "alice"
    # HUMAN Feedback -> human_labels
    assert [h["name"] for h in row["human_labels"]] == ["helpful"]
    # LLM_JUDGE + AI_JUDGE Feedback -> judge_rubrics (both kinds)
    judge_types = {r["name"]: r["source_type"] for r in row["judge_rubrics"]}
    assert judge_types == {"correctness": "LLM_JUDGE", "relevance": "AI_JUDGE"}
    assert row["judge_rubrics"][0]["rationale"] == "right"


def test_trace_to_row_respects_include_source_types() -> None:
    trace = _mk_trace(
        "tr-2",
        "plain single turn query",
        [
            Feedback(name="human_score", value="yes", source=_HUMAN),
            Feedback(name="judge_score", value="pass", source=_JUDGE),
        ],
    )
    row = trace_to_row(trace, include_source_types=["HUMAN"])
    assert row is not None
    assert [h["name"] for h in row["human_labels"]] == ["human_score"]
    assert row["judge_rubrics"] == []  # LLM_JUDGE excluded


def test_judge_expectation_is_dropped() -> None:
    # A judge-sourced Expectation is not a re-scorable objective; drop it.
    trace = _mk_trace(
        "tr-3",
        "q",
        [Expectation(name="ref", value="x", source=_JUDGE)],
    )
    row = trace_to_row(trace)
    assert row is not None
    assert row["expectations"] == []
    assert row["judge_rubrics"] == []


# ---------------------------------------------------------------------------
# ingest: dropping / skipping / counting
# ---------------------------------------------------------------------------


def test_zero_assessment_trace_dropped_by_build() -> None:
    t_labelled = _mk_trace("a", "q1", [Feedback(name="x", value="y", source=_HUMAN)])
    t_bare = _mk_trace("b", "q2", [])
    rows, counters = build_snapshot_rows([t_labelled, t_bare])
    assert [r["example_id"] for r in rows] == ["a"]
    assert counters["kept"] == 1
    assert counters["skipped_no_labels"] == 1


def test_multiturn_and_tool_requests_skipped_and_counted() -> None:
    lab = [Feedback(name="x", value="y", source=_HUMAN)]
    single = _mk_trace("s", '{"messages": [{"role": "user", "content": "hi"}]}', lab)
    multi = _mk_trace(
        "m",
        '{"messages": [{"role": "user", "content": "a"},'
        ' {"role": "assistant", "content": "b"},'
        ' {"role": "user", "content": "c"}]}',
        lab,
    )
    tool = _mk_trace("t", '{"input": [{"type": "function_call", "name": "f"}]}', lab)
    rows, counters = build_snapshot_rows([single, multi, tool])
    assert [r["example_id"] for r in rows] == ["s"]
    assert counters["skipped_multiturn"] == 2


def test_dedup_collapses_identical_queries() -> None:
    lab = [Feedback(name="x", value="y", source=_HUMAN)]
    t1 = _mk_trace("1", "same query", lab)
    t2 = _mk_trace("2", "same query", lab)
    t3 = _mk_trace("3", "other query", lab)
    rows, counters = build_snapshot_rows([t1, t2, t3], dedup=True)
    assert [r["example_id"] for r in rows] == ["1", "3"]
    assert counters["deduped"] == 1
    rows_nodedup, counters_nodedup = build_snapshot_rows([t1, t2, t3], dedup=False)
    assert len(rows_nodedup) == 3
    assert counters_nodedup["deduped"] == 0


def test_parse_single_turn_query_shapes() -> None:
    assert parse_single_turn_query("bare text") == "bare text"
    assert parse_single_turn_query('{"query": "hello"}') == "hello"
    assert parse_single_turn_query('{"input": [{"role": "user", "content": "hi"}]}') == "hi"
    # multimodal content -> skip
    assert (
        parse_single_turn_query(
            '{"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}'
        )
        is None
    )
    # structured text parts join
    assert (
        parse_single_turn_query(
            '{"messages": [{"role": "user", "content":'
            ' [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}'
        )
        == "ab"
    )
    assert parse_single_turn_query(None) is None
    assert parse_single_turn_query("") is None


# ---------------------------------------------------------------------------
# ingest: snapshot round-trip
# ---------------------------------------------------------------------------


def test_snapshot_round_trips(tmp_path: Path) -> None:
    trace = _mk_trace(
        "tr",
        "q",
        [
            Expectation(name="answer", value="42", source=_HUMAN),
            Feedback(name="rubric", value="pass", source=_JUDGE, rationale="ok"),
        ],
    )
    rows, _ = build_snapshot_rows([trace])
    path = tmp_path / "snap.jsonl"
    write_snapshot(path, rows)
    loaded = read_snapshot(path)
    assert loaded == rows


def test_read_snapshot_defaults_missing_buckets(tmp_path: Path) -> None:
    path = tmp_path / "snap.jsonl"
    path.write_text('{"example_id": "x", "query": "q"}\n', encoding="utf-8")
    rows = read_snapshot(path)
    assert rows[0]["expectations"] == []
    assert rows[0]["human_labels"] == []
    assert rows[0]["judge_rubrics"] == []
    assert rows[0]["category"] == "trace"


# ---------------------------------------------------------------------------
# scoring / aggregation
# ---------------------------------------------------------------------------


def test_reference_match_exact_substring_and_miss() -> None:
    assert reference_match("42", "42") == 1.0
    assert reference_match("the answer is 42.", "42") == 1.0  # substring
    assert reference_match("nope", "42") == 0.0
    assert reference_match("[1, 2]", [1, 2]) == 1.0  # structural JSON compare


def test_weighted_aggregate_ranks_human_above_judge() -> None:
    rows = [
        {
            "query": "q1",
            "expectations": [{"name": "answer", "value": "42"}],
            "human_labels": [],
            "judge_rubrics": [{"name": "correctness", "value": "pass", "source_id": "m"}],
        }
    ]

    # Human objective PASSES, judge objective FAILS. With min_human_weight=2
    # the aggregate must lie above 0.5 (the unweighted mean), proving humans
    # dominate.
    def judge_fail(*, query, output, rubric):
        return 0.0

    scores = score_row(rows[0], "the answer is 42", judge_fail)
    agg, per_judge = aggregate_report(rows, [scores], min_human_weight=2.0)
    assert per_judge == {"ref_answer": 1.0, "judge_correctness": 0.0}
    # weighted: (2*1.0 + 1*0.0) / 3 = 0.6667 > 0.5 unweighted
    assert round(agg, 4) == 0.6667

    # Invert (human fails, judge passes) -> aggregate below 0.5.
    def judge_pass(*, query, output, rubric):
        return 1.0

    scores2 = score_row(rows[0], "wrong", judge_pass)
    agg2, _ = aggregate_report(rows, [scores2], min_human_weight=2.0)
    assert round(agg2, 4) == 0.3333


def test_objective_keys_stable_and_sorted() -> None:
    rows = [
        {"expectations": [{"name": "A"}], "human_labels": [], "judge_rubrics": []},
        {
            "expectations": [],
            "human_labels": [{"name": "b label"}],
            "judge_rubrics": [{"name": "C-rubric"}],
        },
    ]
    assert objective_keys(rows) == ["human_b_label", "judge_c_rubric", "ref_a"]
    assert compute_weights(objective_keys(rows), 2.0) == {
        "human_b_label": 2.0,
        "judge_c_rubric": 1.0,
        "ref_a": 2.0,
    }


# ---------------------------------------------------------------------------
# BLOCKER 1: human hard labels are genuinely re-scored against the new output
# ---------------------------------------------------------------------------


def test_hard_label_match_bool_and_numeric() -> None:
    # Boolean label — reads a boolean verdict OUT OF the output.
    assert hard_label_match("true", True) == 1.0
    assert hard_label_match("yes, this is correct", True) == 1.0
    assert hard_label_match("false", True) == 0.0
    assert hard_label_match("no", True) == 0.0
    assert hard_label_match("false", False) == 1.0
    assert hard_label_match("nothing boolean here", True) == 0.0
    # Numeric label — reads the first number out of the output.
    assert hard_label_match("the rating is 4", 4) == 1.0
    assert hard_label_match("3", 4) == 0.0
    assert hard_label_match("no number", 4) == 0.0
    assert hard_label_match("score: 4.0", 4.0) == 1.0
    # Categorical (string) label falls back to reference matching.
    assert hard_label_match("sentiment: positive", "positive") == 1.0
    assert hard_label_match("negative", "positive") == 0.0


def test_human_bool_objective_moves_with_output() -> None:
    # A REAL boolean human feedback (helpful=True). The human_* objective
    # must move when the mutated agent's output changes — proving it scores
    # the NEW output, not a literal text compare against "True".
    row = {
        "query": "q",
        "expectations": [],
        "human_labels": [{"name": "helpful", "value": True}],
        "judge_rubrics": [],
    }

    def _no_judge(**_kwargs):
        raise AssertionError("judge must not run for a human label")

    good = score_row(row, "yes", _no_judge)
    bad = score_row(row, "no", _no_judge)
    assert good == {"human_helpful": 1.0}
    assert bad == {"human_helpful": 0.0}
    assert good != bad  # objective genuinely moves with the output


def test_human_numeric_objective_moves_with_output() -> None:
    row = {
        "query": "q",
        "expectations": [],
        "human_labels": [{"name": "rating", "value": 4}],
        "judge_rubrics": [],
    }
    assert score_row(row, "I'd rate this a 4", lambda **k: 0.0) == {"human_rating": 1.0}
    assert score_row(row, "I'd rate this a 2", lambda **k: 0.0) == {"human_rating": 0.0}


# ---------------------------------------------------------------------------
# BLOCKER 2: colliding slugs keep separate objectives
# ---------------------------------------------------------------------------


def test_slug_collisions_stay_distinct_objectives() -> None:
    # Two judge names that slugify identically must NOT collapse into one
    # objective (they would silently average together otherwise).
    rows = [
        {
            "query": "q",
            "expectations": [],
            "human_labels": [],
            "judge_rubrics": [
                {"name": "correctness-v1", "value": "pass", "source_id": "m"},
                {"name": "correctness v1", "value": "pass", "source_id": "m"},
            ],
        }
    ]
    keys = objective_keys(rows)
    assert len(keys) == 2  # both survive as separate objectives
    assert len(set(keys)) == 2

    # Both are scored independently (one passes, one fails) — neither is lost.
    call_log: list[str] = []

    def judge_fn(*, query, output, rubric):
        call_log.append(rubric["name"])
        return 1.0 if rubric["name"] == "correctness-v1" else 0.0

    key_map = build_key_map(rows)
    scores = score_row(rows[0], "out", judge_fn, key_map)
    assert sorted(call_log) == ["correctness v1", "correctness-v1"]
    assert set(scores.keys()) == set(keys)
    assert sorted(scores.values()) == [0.0, 1.0]  # distinct, not averaged to 0.5


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_changes_with_snapshot_content() -> None:
    rows = [{"example_id": "a", "query": "q", "expectations": [{"name": "x"}],
             "human_labels": [], "judge_rubrics": []}]
    keys = objective_keys(rows)
    fp_a = compute_trace_fingerprint(keys, 2.0, snapshot_content_hash(rows))
    # Same content -> identical fingerprint (reproducible).
    assert fp_a == compute_trace_fingerprint(keys, 2.0, snapshot_content_hash(rows))
    # Different snapshot content -> different fingerprint (invalidates baseline).
    rows2 = rows + [{"example_id": "b", "query": "q2", "expectations": [],
                     "human_labels": [], "judge_rubrics": []}]
    fp_b = compute_trace_fingerprint(
        objective_keys(rows2), 2.0, snapshot_content_hash(rows2)
    )
    assert fp_a != fp_b
    # Fingerprint is non-empty (unlike savesage's zeroed one).
    assert fp_a


def test_snapshot_content_hash_order_independent() -> None:
    a = {"example_id": "a", "query": "1"}
    b = {"example_id": "b", "query": "2"}
    assert snapshot_content_hash([a, b]) == snapshot_content_hash([b, a])


# ---------------------------------------------------------------------------
# judge client wiring (injectable, with fallback)
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, fail_models: set[str], content: str) -> None:
        self._fail = fail_models
        self._content = content
        self.calls: list[str] = []

    def create(self, *, model: str, messages, **kwargs):
        self.calls.append(model)
        if model in self._fail:
            raise RuntimeError(f"model {model} unreachable")
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, fail_models: set[str], content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(fail_models, content))


def test_judge_uses_exact_model_then_falls_back() -> None:
    client = _FakeClient(fail_models={"exact-model"}, content='{"verdict": "pass"}')
    judge_fn = build_judge_fn(client, fallback_model="fallback-model")
    rubric = {"name": "c", "source_id": "exact-model", "rationale": "seed"}
    score = judge_fn(query="q", output="o", rubric=rubric)
    assert score == 1.0
    # Tried the exact recorded model first, then the configured fallback.
    assert client.chat.completions.calls == ["exact-model", "fallback-model"]


def test_judge_parses_fail_verdict() -> None:
    client = _FakeClient(fail_models=set(), content='noise {"verdict": "fail"} tail')
    judge_fn = build_judge_fn(client, fallback_model="fb")
    score = judge_fn(query="q", output="o", rubric={"name": "c", "source_id": "m1"})
    assert score == 0.0
    assert client.chat.completions.calls == ["m1"]  # no fallback on a clean call


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_engine_registers_under_trace() -> None:
    import anvil.domains.trace  # noqa: F401 - import registers the engine
    from anvil.eval.engines import load_engine

    fn = load_engine("trace")
    assert fn.__name__ == "evaluate_trace"


# ---------------------------------------------------------------------------
# evaluate_trace end-to-end (injected predict + judge; no gateway, no server)
# ---------------------------------------------------------------------------


def _write_trace_scaffold(root: Path, snapshot_rel: str) -> tuple[Path, Path]:
    """Write a minimal scaffold/ + harness/config.yaml wired to engine: trace."""
    scaffold = root / "scaffold"
    (scaffold / "skills").mkdir(parents=True)
    (scaffold / "harness.yaml").write_text(
        "sampling: {}\nskills:\n  - file: identity.md\n", encoding="utf-8"
    )
    (scaffold / "skills" / "identity.md").write_text(
        "---\nskill_id: identity\nkind: identity\napplies_to: runtime\n---\n\n# Role\nAgent.\n",
        encoding="utf-8",
    )
    harness = root / "harness"
    harness.mkdir()
    (harness / "config.yaml").write_text(
        "runtime_endpoint: model-r\n"
        "optimizer_endpoint: model-o\n"
        "judge_endpoint: model-j\n"
        "experiments:\n"
        "  runtime: /exp/runtime\n"
        "  eval: /exp/eval\n"
        "  optimizer: /exp/optimizer\n"
        "eval:\n"
        "  engine: trace\n"
        "  default_mode: quick\n"
        # Exercise the DEFAULT parallel path (n_workers > 1), not a forced 1.
        "  n_workers: 4\n"
        "  modes:\n"
        "    quick: {rows: 2}\n"
        f"  trace:\n"
        f"    snapshot_path: {snapshot_rel}\n"
        "    min_human_weight: 2.0\n",
        encoding="utf-8",
    )
    return scaffold, harness / "config.yaml"


def test_evaluate_trace_end_to_end(tmp_path: Path) -> None:
    from anvil.domains.trace.eval import evaluate_trace

    # Freeze a two-row snapshot at the repo-relative path the config names.
    snapshot_rel = "data/trace_snapshot.jsonl"
    scaffold, config_path = _write_trace_scaffold(tmp_path, snapshot_rel)
    rows = [
        {
            "example_id": "tr-1",
            "query": "what is the answer",
            "category": "trace",
            "expectations": [{"name": "answer", "value": "42"}],
            "human_labels": [],
            "judge_rubrics": [{"name": "correctness", "value": "pass", "source_id": "m"}],
        },
        {
            "example_id": "tr-2",
            "query": "second",
            "category": "trace",
            "expectations": [],
            "human_labels": [{"name": "tone", "value": "polite"}],
            "judge_rubrics": [],
        },
    ]
    write_snapshot(tmp_path / snapshot_rel, rows)

    # Deterministic agent: echoes back a good answer for tr-1's query.
    def predict_fn(query: str):
        if "answer" in query:
            return "the answer is 42", 12.5
        return "polite reply", 8.0

    def judge_fn(*, query, output, rubric):
        return 1.0  # judge passes

    report = evaluate_trace(
        scaffold_root=scaffold,
        runtime_config_path=config_path,
        mode="quick",
        predict_fn=predict_fn,
        judge_fn=judge_fn,
    )

    assert report.n_rows == 2
    assert report.mode == "quick"
    # Stable per_judge key set across the two rows' objectives.
    assert set(report.per_judge) == {"ref_answer", "human_tone", "judge_correctness"}
    # tr-1: ref_answer=1 (substring), judge_correctness=1; tr-2 human_tone: "polite"
    # is a substring of "polite reply" -> 1.0. All pass -> aggregate 1.0.
    assert report.per_judge["ref_answer"] == 1.0
    assert report.per_judge["judge_correctness"] == 1.0
    assert report.per_judge["human_tone"] == 1.0
    assert report.aggregate == 1.0
    assert report.failures == []
    # Real (non-empty) fingerprint and captured latency.
    assert report.scorer_fingerprint
    assert report.run_id == "" and report.experiment_id == ""
    assert set(report.trace_ids) == {"tr-1", "tr-2"}
    assert report.cost_metrics["n_rows"] == 2.0
    assert "latency_ms_median" in report.cost_metrics


def test_evaluate_trace_requires_trace_config(tmp_path: Path) -> None:
    import pytest

    from anvil.domains.trace.eval import evaluate_trace

    scaffold = tmp_path / "scaffold"
    (scaffold / "skills").mkdir(parents=True)
    (scaffold / "harness.yaml").write_text(
        "sampling: {}\nskills:\n  - file: identity.md\n", encoding="utf-8"
    )
    (scaffold / "skills" / "identity.md").write_text(
        "---\nskill_id: identity\nkind: identity\napplies_to: runtime\n---\n\n# Role\nAgent.\n",
        encoding="utf-8",
    )
    harness = tmp_path / "harness"
    harness.mkdir()
    # engine: trace but NO trace: block -> must fail loudly.
    (harness / "config.yaml").write_text(
        "runtime_endpoint: r\noptimizer_endpoint: o\njudge_endpoint: j\n"
        "experiments: {runtime: /r, eval: /e, optimizer: /o}\n"
        "eval:\n  engine: trace\n  modes:\n    quick: {rows: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="eval.trace is not configured"):
        evaluate_trace(
            scaffold_root=scaffold,
            runtime_config_path=harness / "config.yaml",
            mode="quick",
            predict_fn=lambda q: ("x", None),
            judge_fn=lambda **k: 1.0,
        )


# ---------------------------------------------------------------------------
# BLOCKER 3: the default parallel re-run path isolates agent state per thread
# ---------------------------------------------------------------------------


def test_build_predictor_isolates_agent_per_thread(monkeypatch) -> None:
    """Each worker thread on the default parallel path gets its OWN agent.

    A single shared memory/agent instance across ``n_workers > 1`` would
    race and leak per-conversation state between concurrent rows. The
    predictor must construct (and cache) one instance per worker thread.
    """
    import anvil.eval.runner as runner_mod
    from anvil.domains.trace.eval import _build_predictor

    constructed_on: list[int] = []
    lock = threading.Lock()

    class _FakeMemorySystem:
        def __init__(self, **_kwargs) -> None:
            self._owner = threading.get_ident()
            with lock:
                constructed_on.append(self._owner)

        def predict(self, query: str):
            # A foreign thread using this instance would trip this assert.
            assert self._owner == threading.get_ident()
            time.sleep(0.005)  # widen the race window
            return "ok", {"latency_ms": 1.0}

    monkeypatch.setattr(
        runner_mod, "_load_memory_system", lambda *a, **k: _FakeMemorySystem()
    )
    snapshot = SimpleNamespace(
        config=SimpleNamespace(mode="code", agent_module="x", runtime_endpoint="")
    )
    predict = _build_predictor(
        snapshot,
        scaffold_path=Path("."),
        runtime_config_path=None,
        runtime_client=None,
    )

    with ThreadPoolExecutor(max_workers=4) as ex:
        outputs = list(ex.map(lambda _i: predict("q"), range(40)))

    assert all(out == ("ok", 1.0) for out in outputs)
    # Exactly one construction per distinct worker thread (thread-local
    # cache): no thread built twice, so nothing is shared across threads.
    assert len(constructed_on) == len(set(constructed_on))
    assert len(set(constructed_on)) >= 2  # the pool really used >1 thread
