"""Offline tests for the interstell domain helpers (no App, no judges, no Interstell repo)."""

from __future__ import annotations

from anvil.domains.interstell.eval import (
    _failure,
    _turns,
    quality_aggregate,
    reasoning_mode,
    request_extra,
)
from anvil.domains.interstell.probe import _classify


def _probe(*results):
    return {"results": [{"model": m, "settings": s, "ok": ok} for m, s, ok in results]}


def test_reasoning_mode_defaults_without_probe() -> None:
    assert reasoning_mode("databricks-gpt-5-4-mini", None) == "none"
    assert reasoning_mode("databricks-gemini-3-1-flash-lite", None) == "profile"


def test_reasoning_mode_prefers_none_then_omit() -> None:
    probe = _probe(
        ("m1", {"reasoning": "none"}, True),
        ("m2", {"reasoning": "none"}, False),
        ("m2", {"reasoning": "omit"}, True),
        ("m3", {"reasoning": "none"}, False),
        ("m3", {"reasoning": "omit"}, False),
        ("m1", {"temperature": 0.0}, True),
    )
    assert reasoning_mode("m1", probe) == "none"
    assert reasoning_mode("m2", probe) == "omit"
    assert reasoning_mode("m3", probe) is None
    # A model the probe never tried falls back to the default.
    assert reasoning_mode("m4", probe) == "none"


def test_request_extra_maps_reasoning_modes() -> None:
    levers = {"temperature": 0.7, "max_completion_tokens": 256, "route": "serving"}
    assert request_extra(levers, "none") == {
        "temperature": 0.7,
        "max_completion_tokens": 256,
        "reasoning_effort": "none",
    }
    # None tells the App to drop the key.
    assert request_extra(levers, "omit")["reasoning_effort"] is None
    assert "reasoning_effort" not in request_extra(levers, "profile")


def test_quality_aggregate_weights_and_missing_metrics() -> None:
    m = {"action_accuracy": 0.5, "speak_and_act_rate": 1.0, "reply_quality_pass_rate": 0.0}
    assert quality_aggregate(m) == (2 * 0.5 + 1.0 + 0.0) / 4
    # Metrics with no applicable rows are left out, not counted as zero.
    assert quality_aggregate({"action_accuracy": 0.75, "speak_and_act_rate": None}) == 0.75
    assert quality_aggregate({}) == 0.0


def test_turns_repeat_samples_and_canary_truncates() -> None:
    samples = [{"id": f"S{i}"} for i in range(1, 37)]
    turns = _turns(samples, 108, None)
    assert len(turns) == 108
    assert turns[0] == (samples[0], 1) and turns[36] == (samples[0], 2)
    assert len(_turns(samples, 36, 3)) == 3


def test_failure_rows_carry_scorer_rationales_and_errors() -> None:
    rec = {
        "sample_id": "S15",
        "round": 2,
        "category": "transfer",
        "expected_action": "speak",
        "outputs": {"action": "transfer_call", "text": "Great.", "first_action_ms": 700},
        "scores": {"action_correct": False, "transfer_gate_safety": False, "speak_and_act": None},
        "rationales": {"action_correct": "agent=transfer_call", "speak_and_act": "n/a"},
    }
    f = _failure(rec)
    assert f["example_id"] == "S15#r2"
    assert f["judge_failures"] == ["action_correct", "transfer_gate_safety"]
    assert f["rationales"] == {"action_correct": "agent=transfer_call"}
    ok = {**rec, "scores": {"action_correct": True}, "outputs": {"action": "speak"}}
    assert _failure(ok) is None


def test_probe_classify() -> None:
    assert _classify({"error": None, "reasoning_tokens": 0}, "none") == (True, None)
    ok, err = _classify({"error": "BadRequestError: Error code: 400 - bad"}, "none")
    assert ok is False and "400" in err
    ok, err = _classify({"reasoning_tokens": 12}, "omit")
    assert ok is False and "reasons by default" in err
    ok, _ = _classify({"error": "App call failed: ReadTimeout (timeout: after 3 attempts)"}, "none")
    assert ok is None


def test_token_param_from_probe_and_max_tokens_extra() -> None:
    from anvil.domains.interstell.eval import token_param

    probe = _probe(
        ("llama", {"reasoning": "none"}, False),
        ("llama", {"reasoning": "omit", "token_param": "max_tokens"}, True),
        ("gpt", {"reasoning": "none"}, True),
    )
    assert token_param("llama", probe) == "max_tokens"
    assert token_param("gpt", probe) == "max_completion_tokens"
    assert token_param("other", None) == "max_completion_tokens"
    extra = request_extra({"max_completion_tokens": 512}, "omit", "max_tokens")
    assert extra["max_completion_tokens"] is None and extra["max_tokens"] == 512
