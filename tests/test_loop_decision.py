"""Tests for the keep/revert/noop/infra_fail decision logic."""

from __future__ import annotations

from anvil.loop.decision import Decision, decide


def test_strict_positive_delta_keeps() -> None:
    assert decide(score_delta=0.001, action_kind="add_rule", parse_status="ok") == Decision.KEEP


def test_zero_delta_reverts() -> None:
    """A tie counts as a revert. Neutral mutations clutter the loop."""
    assert decide(score_delta=0.0, action_kind="add_rule", parse_status="ok") == Decision.REVERT


def test_negative_delta_reverts() -> None:
    assert decide(score_delta=-0.001, action_kind="edit_rule", parse_status="ok") == Decision.REVERT


def test_noop_action_returns_noop_regardless_of_delta() -> None:
    """The optimizer's choice not to mutate wins over score considerations."""
    assert decide(score_delta=None, action_kind="noop", parse_status="ok") == Decision.NOOP
    assert decide(score_delta=0.5, action_kind="noop", parse_status="ok") == Decision.NOOP


def test_eval_failure_returns_infra_fail() -> None:
    assert (
        decide(
            score_delta=None,
            action_kind="add_rule",
            parse_status="ok",
            eval_failed=True,
        )
        == Decision.INFRA_FAIL
    )


def test_missing_score_returns_infra_fail_for_non_noop() -> None:
    """Score=None on a non-noop action means the eval did not return — infra issue."""
    assert (
        decide(
            score_delta=None,
            action_kind="add_rule",
            parse_status="ok",
        )
        == Decision.INFRA_FAIL
    )


def test_parse_failure_with_noop_action_still_noop() -> None:
    """If the parser falls back to noop, the decision is noop (action wins)."""
    assert (
        decide(
            score_delta=None,
            action_kind="noop",
            parse_status="schema_mismatch",
        )
        == Decision.NOOP
    )
