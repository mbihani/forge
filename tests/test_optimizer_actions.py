"""Tests for the ``OptimizerAction`` model + parser.

These tests pin the contract every other plane (loop applier, mutations
Delta row, critique md generator) reads, so they stay green even when
the optimizer prompt evolves.
"""

from __future__ import annotations

import json

import pytest

from anvil.optimizer.actions import (
    AddRuleAction,
    AddSkillAction,
    ChangeSamplingAction,
    DeleteRuleAction,
    DeleteSkillAction,
    EditSkillAction,
    NoopAction,
)
from anvil.optimizer.parser import parse_action

# ---------------------------------------------------------------------------
# Action model validation
# ---------------------------------------------------------------------------


def test_add_rule_minimum_valid() -> None:
    a = AddRuleAction(
        target_file="rules/answer_scope_discipline.md",
        content="# rule\n",
        rationale="seed",
    )
    assert a.action == "add_rule"


def test_path_rejects_absolute() -> None:
    with pytest.raises(ValueError, match="must be relative"):
        AddRuleAction(target_file="/rules/foo.md", content="x", rationale="r")


def test_path_rejects_traversal() -> None:
    with pytest.raises(ValueError, match="must be relative"):
        AddRuleAction(target_file="rules/../sneaky.md", content="x", rationale="r")


def test_path_rejects_wrong_dir_for_rule() -> None:
    with pytest.raises(ValueError, match=r"under scaffold/rules"):
        AddRuleAction(target_file="skills/foo.md", content="x", rationale="r")


def test_path_rejects_wrong_dir_for_skill() -> None:
    with pytest.raises(ValueError, match=r"under scaffold/skills"):
        AddSkillAction(target_file="rules/foo.md", content="x", rationale="r")


def test_path_requires_md_extension() -> None:
    with pytest.raises(ValueError, match=".md"):
        AddRuleAction(target_file="rules/foo.txt", content="x", rationale="r")


def test_change_sampling_field_enum() -> None:
    a = ChangeSamplingAction(field="temperature", value=0.2, rationale="explore")
    assert a.field == "temperature"
    assert a.value == 0.2

    with pytest.raises(ValueError):
        ChangeSamplingAction(field="not_a_field", value=0.5, rationale="r")


def test_noop_rationale_required() -> None:
    a = NoopAction(rationale="nothing to do")
    assert a.action == "noop"

    with pytest.raises(ValueError):
        NoopAction(rationale="")


# ---------------------------------------------------------------------------
# Parser — happy paths
# ---------------------------------------------------------------------------


def _wrap_block(payload: dict | str, fence: str = "json-action") -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    return f"some prose\n```{fence}\n{body}\n```\nmore prose"


def test_parse_add_rule() -> None:
    transcript = _wrap_block(
        {
            "action": "add_rule",
            "target_file": "rules/foo.md",
            "content": "# foo\nbody\n",
            "rationale": "seed",
        }
    )
    result = parse_action(transcript)
    assert result.parse_status == "ok"
    assert isinstance(result.action, AddRuleAction)


def test_parse_edit_skill() -> None:
    transcript = _wrap_block(
        {
            "action": "edit_skill",
            "target_file": "skills/identity.md",
            "content": "# new identity\n",
            "rationale": "tighten",
        }
    )
    result = parse_action(transcript)
    assert result.parse_status == "ok"
    assert isinstance(result.action, EditSkillAction)


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        (
            {"action": "delete_skill", "target": "skills/retrieval.md", "rationale": "harmful"},
            DeleteSkillAction,
        ),
        (
            {"action": "delete_rule", "target": "rules/specificity.md", "rationale": "too strict"},
            DeleteRuleAction,
        ),
    ],
)
def test_parse_delete_actions(payload: dict, expected_type: type) -> None:
    result = parse_action(_wrap_block(payload))
    assert result.parse_status == "ok"
    assert isinstance(result.action, expected_type)


def test_delete_path_security() -> None:
    with pytest.raises(ValueError, match="must be relative"):
        DeleteSkillAction(target="skills/../identity.md", rationale="escape")
    with pytest.raises(ValueError, match=r"under scaffold/rules"):
        DeleteRuleAction(target="skills/identity.md", rationale="wrong tree")


def test_parse_change_sampling() -> None:
    transcript = _wrap_block(
        {
            "action": "change_sampling",
            "field": "max_tool_calls",
            "value": 5,
            "rationale": "headroom",
        }
    )
    result = parse_action(transcript)
    assert result.parse_status == "ok"
    assert isinstance(result.action, ChangeSamplingAction)


def test_parse_noop_explicit() -> None:
    transcript = _wrap_block({"action": "noop", "rationale": "no failures worth chasing"})
    result = parse_action(transcript)
    assert result.parse_status == "ok"
    assert isinstance(result.action, NoopAction)


def test_parse_accepts_plain_json_fence() -> None:
    """Tolerate the model emitting ```json instead of ```json-action."""
    transcript = _wrap_block({"action": "noop", "rationale": "explicit"}, fence="json")
    result = parse_action(transcript)
    assert result.parse_status == "ok"
    assert isinstance(result.action, NoopAction)


# ---------------------------------------------------------------------------
# Parser — defensive fallbacks
# ---------------------------------------------------------------------------


def test_parse_no_block_returns_noop() -> None:
    result = parse_action("the optimizer rambled and never produced a block")
    assert result.parse_status == "no_block"
    assert isinstance(result.action, NoopAction)
    assert "no `json-action`" in result.action.rationale


def test_parse_multiple_blocks_picks_last() -> None:
    """When the optimizer emits more than one block (e.g. a worked example
    plus the final decision), the parser takes the LAST block as the real
    decision and marks the status as ``ok_last_of_many`` for diagnostics."""
    transcript = (
        _wrap_block({"action": "noop", "rationale": "earlier exploration"})
        + "\n"
        + _wrap_block({"action": "noop", "rationale": "the real decision"})
    )
    result = parse_action(transcript)
    assert result.parse_status == "ok_last_of_many"
    assert isinstance(result.action, NoopAction)
    assert result.action.rationale == "the real decision"
    assert result.n_blocks_found == 2


def test_parse_bad_json_returns_noop() -> None:
    transcript = "```json-action\n{this is not json}\n```"
    result = parse_action(transcript)
    assert result.parse_status == "bad_json"
    assert isinstance(result.action, NoopAction)


def test_parse_schema_mismatch_returns_noop() -> None:
    """Action field unknown → noop."""
    transcript = _wrap_block({"action": "delete_universe", "rationale": "burn it down"})
    result = parse_action(transcript)
    assert result.parse_status == "schema_mismatch"
    assert isinstance(result.action, NoopAction)


def test_parse_missing_rationale_returns_noop() -> None:
    transcript = _wrap_block({"action": "add_rule", "target_file": "rules/foo.md", "content": "x"})
    result = parse_action(transcript)
    assert result.parse_status == "schema_mismatch"
    assert isinstance(result.action, NoopAction)


def test_parse_extra_field_returns_noop() -> None:
    """extra='forbid' on _ActionBase rejects unknown keys."""
    transcript = _wrap_block(
        {
            "action": "noop",
            "rationale": "fine",
            "secret_handshake": "no",
        }
    )
    result = parse_action(transcript)
    assert result.parse_status == "schema_mismatch"
    assert isinstance(result.action, NoopAction)
