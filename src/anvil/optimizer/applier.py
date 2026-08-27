"""Apply an :class:`OptimizerAction` to the on-disk scaffold.

The optimizer plane never writes files itself — it returns a structured
action and the loop's applier writes it. This separation is what makes
the optimizer mockable (loop tests inject a fake action) and the loop
auditable (every write goes through one place that can lint, validate,
and git-add).

For ``add_*`` and ``edit_*``, this module writes the markdown file under
``scaffold/`` and registers it in ``scaffold/harness.yaml`` if it isn't
already (``add_*`` only). For ``change_sampling``, this module updates
``scaffold/harness.yaml > sampling.<field>``. For ``noop``, no-op.

Returns an :class:`ApplyResult` listing the files touched + an
``action_summary`` that goes into the mutations Delta row's
``diff_summary``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from anvil.optimizer.actions import (
    AddRuleAction,
    AddSkillAction,
    ChangeSamplingAction,
    DeleteAgentAction,
    DeleteRuleAction,
    DeleteSkillAction,
    EditRuleAction,
    EditSkillAction,
    NoopAction,
    OptimizerAction,
    WriteAgentAction,
)
from anvil.optimizer.code_validation import validate_code_candidate


@dataclass
class ApplyResult:
    files_added: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)
    action_summary: str = ""


class ApplyError(Exception):
    """Raised when the action's pre-conditions are violated.

    Examples: edit_* targeting a non-existent file; add_* targeting a
    file that already exists; change_sampling on a value Pydantic
    didn't already constrain.
    """


# Actions that belong to each optimization mode. ``noop`` is always
# valid regardless of mode. These sets drive the mode-aware validation
# that prevents the optimizer from mixing prompt and code mutations in
# the same round.
_PROMPT_ACTIONS: frozenset[str] = frozenset(
    {
        "add_skill",
        "edit_skill",
        "delete_skill",
        "add_rule",
        "edit_rule",
        "delete_rule",
        "change_sampling",
    }
)
_CODE_ACTIONS: frozenset[str] = frozenset({"write_agent", "delete_agent"})


def apply_action(
    action: OptimizerAction,
    scaffold_root: Path | str,
    *,
    mode: str = "prompt",
    repo_root: Path | str | None = None,
) -> ApplyResult:
    """Apply ``action`` to ``scaffold_root``. Returns paths touched.

    ``mode`` selects which action vocabulary is allowed: ``prompt``
    mode accepts skill/rule/sampling mutations; ``code`` mode accepts
    write_agent/delete_agent. Mismatched actions raise :class:`ApplyError`.

    ``repo_root`` is the repo root for code-mode file paths (agents/
    lives at the repo root, not under scaffold/). When ``None`` it is
    derived from ``scaffold_root.parent`` (the convention: scaffold/
    and harness/ are siblings under the repo root).
    """
    root = Path(scaffold_root)
    repo_root = root.parent if repo_root is None else Path(repo_root)

    _validate_action_mode(action.action, mode)

    if isinstance(action, NoopAction):
        return ApplyResult(action_summary=f"noop: {action.rationale}")

    if isinstance(action, AddSkillAction):
        return _apply_add_file(
            root,
            role="skill",
            target=action.target_file,
            content=action.content,
            rationale=action.rationale,
        )
    if isinstance(action, EditSkillAction):
        return _apply_edit_file(
            root,
            role="skill",
            target=action.target_file,
            content=action.content,
            rationale=action.rationale,
        )
    if isinstance(action, DeleteSkillAction):
        return _apply_delete_file(
            root,
            role="skill",
            target=action.target,
            rationale=action.rationale,
        )
    if isinstance(action, AddRuleAction):
        return _apply_add_file(
            root,
            role="rule",
            target=action.target_file,
            content=action.content,
            rationale=action.rationale,
        )
    if isinstance(action, EditRuleAction):
        return _apply_edit_file(
            root,
            role="rule",
            target=action.target_file,
            content=action.content,
            rationale=action.rationale,
        )
    if isinstance(action, DeleteRuleAction):
        return _apply_delete_file(
            root,
            role="rule",
            target=action.target,
            rationale=action.rationale,
        )
    if isinstance(action, ChangeSamplingAction):
        return _apply_change_sampling(
            root,
            field_name=action.field,
            value=action.value,
            rationale=action.rationale,
        )
    if isinstance(action, WriteAgentAction):
        return _apply_write_agent(action, repo_root)
    if isinstance(action, DeleteAgentAction):
        return _apply_delete_agent(action, repo_root, root)

    raise ApplyError(f"unknown action type: {type(action).__name__}")


def _validate_action_mode(action_kind: str, mode: str) -> None:
    """Reject actions that do not match the active optimization mode.

    * ``prompt`` mode → only skill/rule/sampling actions + ``noop``.
    * ``code``   mode → only ``write_agent`` / ``delete_agent`` + ``noop``.
    * anything else → rejected (fail-closed).

    This prevents the optimizer from accidentally mixing prompt and
    code mutations in the same round — e.g. editing a skill while the
    eval expects a code-mode agent module — and ensures an unknown
    mode value (typo, empty string, ``hybrid``) cannot silently permit
    every action because neither branch fires.
    """
    if mode == "prompt":
        if action_kind in _CODE_ACTIONS:
            raise ApplyError(
                f"action {action_kind!r} is only valid in code mode "
                f"(harness/config.yaml > mode: code)"
            )
    elif mode == "code":
        if action_kind in _PROMPT_ACTIONS:
            raise ApplyError(
                f"action {action_kind!r} is only valid in prompt mode "
                f"(harness/config.yaml > mode: prompt)"
            )
    else:
        raise ApplyError(f"unknown optimization mode {mode!r}; expected 'prompt' or 'code'")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_add_file(
    root: Path,
    *,
    role: str,
    target: str,
    content: str,
    rationale: str,
) -> ApplyResult:
    path = root / target
    if path.exists():
        raise ApplyError(f"{role} '{target}' already exists; use edit_{role} instead")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ensure_trailing_newline(content), encoding="utf-8")

    # Register in harness.yaml if not already present.
    harness_path = root / "harness.yaml"
    harness = _load_yaml(harness_path)
    list_key = "skills" if role == "skill" else "rules"
    entries: list[dict] = list(harness.get(list_key) or [])
    filename = target.split("/", 1)[1]  # strip "skills/" or "rules/"
    if not any((e.get("file") == filename) for e in entries if isinstance(e, dict)):
        entries.append({"file": filename})
        harness[list_key] = entries
        _dump_yaml(harness_path, harness)
        files_changed = [str(harness_path.relative_to(root.parent))]
    else:
        files_changed = []

    return ApplyResult(
        files_added=[f"scaffold/{target}"],
        files_changed=files_changed,
        action_summary=f"add_{role} {target}: {rationale[:120]}",
    )


def _apply_edit_file(
    root: Path,
    *,
    role: str,
    target: str,
    content: str,
    rationale: str,
) -> ApplyResult:
    path = root / target
    if not path.is_file():
        raise ApplyError(f"{role} '{target}' does not exist; use add_{role} instead")
    path.write_text(_ensure_trailing_newline(content), encoding="utf-8")
    return ApplyResult(
        files_changed=[f"scaffold/{target}"],
        action_summary=f"edit_{role} {target}: {rationale[:120]}",
    )


def _apply_delete_file(
    root: Path,
    *,
    role: str,
    target: str,
    rationale: str,
) -> ApplyResult:
    path = root / target
    if not path.is_file():
        raise ApplyError(f"{role} '{target}' does not exist and cannot be deleted")

    if role == "skill" and _frontmatter(path).get("kind") == "identity":
        raise ApplyError(f"cannot delete identity skill '{target}'")

    harness_path = root / "harness.yaml"
    harness = _load_yaml(harness_path)
    list_key = "skills" if role == "skill" else "rules"
    entries = list(harness.get(list_key) or [])
    filename = target.split("/", 1)[1]
    remaining = [
        entry
        for entry in entries
        if not (isinstance(entry, dict) and entry.get("file") == filename)
    ]
    path.unlink()

    files_changed = []
    if remaining != entries:
        harness[list_key] = remaining
        _dump_yaml(harness_path, harness)
        files_changed.append(str(harness_path.relative_to(root.parent)))

    return ApplyResult(
        files_changed=files_changed,
        files_removed=[f"scaffold/{target}"],
        action_summary=f"delete_{role} {target}: {rationale[:120]}",
    )


def _apply_change_sampling(
    root: Path,
    *,
    field_name: str,
    value: float | int | str | None,
    rationale: str,
) -> ApplyResult:
    harness_path = root / "harness.yaml"
    harness = _load_yaml(harness_path)
    sampling = dict(harness.get("sampling") or {})
    old = sampling.get(field_name)
    sampling[field_name] = value
    harness["sampling"] = sampling
    _dump_yaml(harness_path, harness)
    return ApplyResult(
        files_changed=[str(harness_path.relative_to(root.parent))],
        action_summary=f"change_sampling {field_name}: {old!r} → {value!r}: {rationale[:120]}",
    )


def _check_path_safe(target_path: Path, repo_root: Path) -> None:
    """Verify the resolved target path is beneath the resolved agents/ dir.

    The model-level :func:`anvil.optimizer.actions._check_agent_path` only
    inspects the *string* (no ``..``, starts with ``agents/``, ``.py``
    suffix). It cannot catch a symlink inside ``agents/`` that points
    outside the repo, which would allow arbitrary writes/deletes. This
    resolver-level check closes that gap by resolving symlinks on both
    the target and the ``agents/`` directory before the containment test.
    """
    agents_dir = (repo_root / "agents").resolve()
    resolved = target_path.resolve()
    try:
        resolved.relative_to(agents_dir)
    except ValueError:
        raise ApplyError(f"target path {target_path} resolves outside agents/") from None


def _apply_write_agent(action: WriteAgentAction, repo_root: Path) -> ApplyResult:
    """Write a new agent module, validating it first.

    The code is written to a **temp file**, run through
    :func:`validate_code_candidate` (AST denylist + isolated import),
    and only then written to its final location. Unvalidated code is
    never persisted to disk — a validation failure raises
    :class:`CodeValidationError` and leaves the target untouched.
    """
    target_path = repo_root / action.target_file
    _check_path_safe(target_path, repo_root)
    target_existed = target_path.is_file()

    # Write content to a temp file for validation. We never write
    # unvalidated code to the final location.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="anvil_candidate_",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(action.content)
        temp_path = Path(f.name)

    try:
        validate_code_candidate(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    # Validation passed — write to the final location.
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(action.content, encoding="utf-8")

    return ApplyResult(
        files_added=[action.target_file] if not target_existed else [],
        files_changed=[action.target_file] if target_existed else [],
        action_summary=f"write_agent {action.target_file}: {action.rationale[:120]}",
    )


def _apply_delete_agent(
    action: DeleteAgentAction,
    repo_root: Path,
    scaffold_root: Path,
) -> ApplyResult:
    """Delete an agent module, protecting the configured baseline.

    The active ``agent_module`` (read from ``harness/config.yaml``) is
    protected from deletion — removing the file the eval would load
    next round would break the loop.
    """
    target_path = repo_root / action.target
    _check_path_safe(target_path, repo_root)

    # Protect the configured agent module.
    agent_module = _read_agent_module(scaffold_root)
    protected = _agent_module_to_agents_path(agent_module)
    if protected is not None and protected == action.target:
        raise ApplyError(
            f"cannot delete active agent module {action.target!r} "
            f"(configured as agent_module: {agent_module!r})"
        )

    if not target_path.is_file():
        raise ApplyError(f"agent {action.target!r} does not exist and cannot be deleted")

    target_path.unlink()

    return ApplyResult(
        files_removed=[action.target],
        action_summary=f"delete_agent {action.target}: {action.rationale[:120]}",
    )


def _read_agent_module(scaffold_root: Path) -> str:
    """Read ``agent_module`` from ``harness/config.yaml``.

    Returns the default (``"anvil.agents.baseline"``) when the file or
    field is absent, so the protection check works on repos that
    predate the ``agent_module`` config field.
    """
    config_path = scaffold_root.parent / "harness" / "config.yaml"
    if not config_path.is_file():
        return "anvil.agents.baseline"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return raw.get("agent_module", "anvil.agents.baseline")


def _agent_module_to_agents_path(agent_module: str) -> str | None:
    """Convert an ``agent_module`` config value to a repo-root-relative
    path under ``agents/``, or ``None`` if the module is not a file in
    ``agents/``.

    The ``agent_module`` config accepts two forms:

    * A dotted Python path (e.g. ``"anvil.agents.baseline"``) — a
      package module, not in ``agents/``. Returns ``None``.
    * A ``.py`` file path (e.g. ``"agents/extractor_v2.py"``) — already
      a path. Returned as-is when it starts with ``agents/``.

    Dotted paths starting with ``agents.`` (e.g.
    ``"agents.extractor_v2"``) are converted to ``agents/extractor_v2.py``.
    """
    if agent_module.endswith(".py") and "/" in agent_module:
        parts = agent_module.split("/")
        if parts[0] == "agents":
            return agent_module
        return None
    # Dotted path.
    parts = agent_module.split(".")
    if len(parts) >= 2 and parts[0] == "agents":
        return "/".join(parts) + ".py"
    return None


def _load_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ApplyError(f"{path} did not parse as a YAML mapping")
    return raw


def _dump_yaml(path: Path, data: dict) -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8")


def _frontmatter(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        return {}
    end = raw.find("\n---", 3)
    if end == -1:
        return {}
    metadata = yaml.safe_load(raw[3:end].strip()) or {}
    return metadata if isinstance(metadata, dict) else {}


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"
