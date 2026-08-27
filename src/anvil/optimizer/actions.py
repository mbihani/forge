"""Structured actions emitted by the optimizer.

Replaces the legacy "free-reign Claude Code" pattern with a constrained
action menu. The optimizer reasons freely during its session, but its
final decision is a single JSON object matching one of the discriminated
``OptimizerAction`` variants below. The :func:`anvil.optimizer.parser.parse_action`
function validates this with ``Pydantic`` and falls back to ``NoopAction``
on any malformed output.

Why this shape (vs. arbitrary ``Edit`` / ``Write`` / ``Bash``):

* **Reduces variance.** The legacy optimizer could touch any file with
  any content; round 6's regression came from a skill that clashed with
  an existing rule. With the menu, the loop's applier can lint and
  validate before committing.
* **Mockable.** Tests for the loop can inject a fake action without
  spinning a real Claude session.
* **Auditable.** ``rationale`` is mandatory on every variant; the
  mutation Delta row carries it as ``diff_summary``.

Eight actions (today):

* ``add_skill`` / ``edit_skill``      — add or edit a markdown skill
* ``add_rule``  / ``edit_rule``       — add or edit a markdown rule
* ``delete_skill`` / ``delete_rule``   — remove a markdown skill or rule
* ``change_sampling``                  — tweak ``scaffold/harness.yaml > sampling.*``
* ``noop``                             — the optimizer chose to do nothing

Each file mutation carries a relative target path and a rationale. Add/edit
actions also carry the full replacement ``content``; delete actions do not.

The existing add/edit variants name their path field ``target_file``; delete
variants use ``target``:

* ``target_file``: relative path under ``scaffold/`` (without the
  ``scaffold/`` prefix; e.g. ``"skills/identity.md"``).
* ``content``: full file content for add/edit skills and rules.
* ``rationale``: short string explaining the why. Goes into the
  critique md and the mutations Delta row.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _ActionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str = Field(min_length=1, max_length=2000)


class AddSkillAction(_ActionBase):
    action: Literal["add_skill"] = "add_skill"
    target_file: str = Field(min_length=1)
    content: str = Field(min_length=1)

    @field_validator("target_file")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _check_relative_path(v, expected_dir="skills")


class EditSkillAction(_ActionBase):
    action: Literal["edit_skill"] = "edit_skill"
    target_file: str = Field(min_length=1)
    content: str = Field(min_length=1)

    @field_validator("target_file")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _check_relative_path(v, expected_dir="skills")


class DeleteSkillAction(_ActionBase):
    action: Literal["delete_skill"] = "delete_skill"
    target: str = Field(min_length=1)

    @field_validator("target")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _check_relative_path(v, expected_dir="skills")


class AddRuleAction(_ActionBase):
    action: Literal["add_rule"] = "add_rule"
    target_file: str = Field(min_length=1)
    content: str = Field(min_length=1)

    @field_validator("target_file")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _check_relative_path(v, expected_dir="rules")


class EditRuleAction(_ActionBase):
    action: Literal["edit_rule"] = "edit_rule"
    target_file: str = Field(min_length=1)
    content: str = Field(min_length=1)

    @field_validator("target_file")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _check_relative_path(v, expected_dir="rules")


class DeleteRuleAction(_ActionBase):
    action: Literal["delete_rule"] = "delete_rule"
    target: str = Field(min_length=1)

    @field_validator("target")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _check_relative_path(v, expected_dir="rules")


class ChangeSamplingAction(_ActionBase):
    """Edit a single field under ``scaffold/harness.yaml > sampling.*``."""

    action: Literal["change_sampling"] = "change_sampling"
    field: Literal["temperature", "top_p", "max_tokens", "tool_choice", "max_tool_calls"]
    value: float | int | str | None


class NoopAction(_ActionBase):
    """The optimizer chose to make no change. Always valid; never a parse failure."""

    action: Literal["noop"] = "noop"


class WriteAgentAction(_ActionBase):
    """Write or replace a Python agent module in ``agents/``.

    In code mode, the optimizer writes a complete ``MemorySystem``
    subclass as a Python file. The file path is relative to the repo
    root and must be inside the ``agents/`` directory. The applier
    validates the code (AST denylist + isolated import) BEFORE writing
    it to disk — unvalidated code is never persisted.
    """

    action: Literal["write_agent"] = "write_agent"
    target_file: str = Field(min_length=1)
    content: str = Field(min_length=1)

    @field_validator("target_file")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _check_agent_path(v)


class DeleteAgentAction(_ActionBase):
    """Delete a Python agent module from ``agents/``.

    Removes a candidate agent file. The active ``agent_module``
    (configured in ``harness/config.yaml``) is protected from deletion.
    """

    action: Literal["delete_agent"] = "delete_agent"
    target: str = Field(min_length=1)

    @field_validator("target")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _check_agent_path(v)


# Discriminated union over the literal ``action`` field.
OptimizerAction = Annotated[
    AddSkillAction
    | EditSkillAction
    | DeleteSkillAction
    | AddRuleAction
    | EditRuleAction
    | DeleteRuleAction
    | ChangeSamplingAction
    | WriteAgentAction
    | DeleteAgentAction
    | NoopAction,
    Field(discriminator="action"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_relative_path(path: str, *, expected_dir: str) -> str:
    """Reject path traversal and require the right scaffold sub-directory."""
    if path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"target_file must be relative under scaffold/: got {path!r}")
    parts = path.split("/")
    if parts[0] != expected_dir:
        raise ValueError(
            f"target_file must live under scaffold/{expected_dir}/, got first segment {parts[0]!r}"
        )
    if not path.endswith(".md"):
        raise ValueError(f"target_file must be a .md file, got {path!r}")
    return path


def _check_agent_path(path: str) -> str:
    """Reject path traversal and require a ``.py`` file inside ``agents/``.

    Same security model as :func:`_check_relative_path` but for code-mode
    agent modules: the path is relative to the repo root (not scaffold/),
    must start with ``agents/``, and must end with ``.py``.
    """
    if path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"target must be relative, got {path!r}")
    parts = path.split("/")
    if parts[0] != "agents":
        raise ValueError(f"target must live under agents/, got first segment {parts[0]!r}")
    if not path.endswith(".py"):
        raise ValueError(f"target must be a .py file, got {path!r}")
    return path
