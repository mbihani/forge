"""Tests for code-mutation actions (Phase 3 task 3.3).

Covers the acceptance contract:

* ``WriteAgentAction`` / ``DeleteAgentAction`` Pydantic models exist.
* Path validation rejects absolute, ``..`` traversal, wrong dir, non-``.py``.
* ``_apply_write_agent`` validates with ``validate_code_candidate`` BEFORE
  writing to disk (temp file → validate → write). On failure the target
  file is NOT written.
* ``_apply_delete_agent`` protects the configured ``agent_module``.
* Mode-aware validation rejects code actions in prompt mode and vice versa.
* Parser can parse ``write_agent`` / ``delete_agent`` JSON blocks.
* Optimizer prompt has examples for both new actions.

No LLM calls, no Databricks calls — all mocked or offline.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from anvil.optimizer.actions import (
    DeleteAgentAction,
    WriteAgentAction,
)
from anvil.optimizer.applier import ApplyError, apply_action
from anvil.optimizer.code_validation import CodeValidationError
from anvil.optimizer.parser import parse_action

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_AGENT_SOURCE = textwrap.dedent("""\
    from anvil.agents.memory_system import MemorySystem

    class TestAgent(MemorySystem):
        def __init__(self, **kwargs):
            pass

        def predict(self, input):
            return input, {"context_chars": len(input)}

        def learn_from_batch(self, batch_results):
            pass
    """)


def _bare_scaffold(tmp_path: Path) -> Path:
    """Build a minimal valid prompt-mode scaffold tree under ``tmp_path/scaffold``."""
    root = tmp_path / "scaffold"
    (root / "skills").mkdir(parents=True)
    (root / "rules").mkdir(parents=True)
    (root / "memory").mkdir(parents=True)
    (root / "skills" / "identity.md").write_text(
        "---\nskill_id: identity\nkind: identity\napplies_to: runtime\n---\n\n# role\ntest agent\n",
        encoding="utf-8",
    )
    (root / "rules" / "existing.md").write_text(
        "---\nrule_id: existing\napplies_to: runtime\n---\n\n# existing\nbody\n",
        encoding="utf-8",
    )
    (root / "harness.yaml").write_text(
        "sampling:\n  temperature: 0.7\n  max_tool_calls: 3\n"
        "  tool_choice: auto\n  max_tokens: 2048\n"
        "skills:\n  - file: identity.md\n"
        "rules:\n  - file: existing.md\n"
        "tools: []\n",
        encoding="utf-8",
    )
    return root


def _code_mode_scaffold(tmp_path: Path, agent_module: str = "anvil.agents.baseline") -> Path:
    """Build a minimal code-mode scaffold + config under ``tmp_path``.

    ``scaffold/`` lives at ``tmp_path/scaffold``; ``harness/config.yaml``
    lives at ``tmp_path/harness/config.yaml`` (sibling of scaffold/).
    ``agents/`` lives at ``tmp_path/agents`` (repo root).
    """
    scaffold_root = _bare_scaffold(tmp_path)
    config = tmp_path / "harness" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f"mode: code\n"
        f"agent_module: {agent_module}\n"
        f"runtime_endpoint: rt\n"
        f"optimizer_endpoint: op\n"
        f"judge_endpoint: j\n"
        f"experiments:\n"
        f"  runtime: r\n  eval: e\n  optimizer: o\n",
        encoding="utf-8",
    )
    (tmp_path / "agents").mkdir(parents=True, exist_ok=True)
    return scaffold_root


def _wrap_block(payload: dict | str, fence: str = "json-action") -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    return f"some prose\n```{fence}\n{body}\n```\nmore prose"


# ---------------------------------------------------------------------------
# 1. Action model validation
# ---------------------------------------------------------------------------


def test_write_agent_minimum_valid() -> None:
    a = WriteAgentAction(
        target_file="agents/extractor_v2.py",
        content="x = 1\n",
        rationale="add agent",
    )
    assert a.action == "write_agent"
    assert a.content == "x = 1\n"


def test_delete_agent_minimum_valid() -> None:
    a = DeleteAgentAction(
        target="agents/extractor_v2.py",
        rationale="remove agent",
    )
    assert a.action == "delete_agent"


# ---------------------------------------------------------------------------
# 2. Path validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_file",
    ["/agents/foo.py", "agents/../sneaky.py", "agents/../../etc/passwd"],
)
def test_write_agent_rejects_unsafe_path(target_file: str) -> None:
    with pytest.raises(ValueError, match="must be relative"):
        WriteAgentAction(target_file=target_file, content="x = 1\n", rationale="r")


def test_write_agent_rejects_wrong_dir() -> None:
    with pytest.raises(ValueError, match="under agents"):
        WriteAgentAction(target_file="skills/foo.py", content="x = 1\n", rationale="r")


def test_write_agent_rejects_non_py_extension() -> None:
    with pytest.raises(ValueError, match=".py"):
        WriteAgentAction(target_file="agents/foo.md", content="x = 1\n", rationale="r")


def test_write_agent_requires_content() -> None:
    with pytest.raises(ValueError):
        WriteAgentAction(target_file="agents/foo.py", content="", rationale="r")


@pytest.mark.parametrize(
    "target",
    ["/agents/foo.py", "agents/../sneaky.py"],
)
def test_delete_agent_rejects_unsafe_path(target: str) -> None:
    with pytest.raises(ValueError, match="must be relative"):
        DeleteAgentAction(target=target, rationale="r")


def test_delete_agent_rejects_wrong_dir() -> None:
    with pytest.raises(ValueError, match="under agents"):
        DeleteAgentAction(target="skills/foo.py", rationale="r")


def test_delete_agent_rejects_non_py_extension() -> None:
    with pytest.raises(ValueError, match=".py"):
        DeleteAgentAction(target="agents/foo.md", rationale="r")


# ---------------------------------------------------------------------------
# 3. Parser — write_agent / delete_agent
# ---------------------------------------------------------------------------


def test_parse_write_agent() -> None:
    transcript = _wrap_block(
        {
            "action": "write_agent",
            "target_file": "agents/extractor_v2.py",
            "content": "from anvil.agents.memory_system import MemorySystem\n",
            "rationale": "add retrieval agent",
        }
    )
    result = parse_action(transcript)
    assert result.parse_status == "ok"
    assert isinstance(result.action, WriteAgentAction)
    assert result.action.target_file == "agents/extractor_v2.py"


def test_parse_delete_agent() -> None:
    transcript = _wrap_block(
        {
            "action": "delete_agent",
            "target": "agents/old_agent.py",
            "rationale": "remove",
        }
    )
    result = parse_action(transcript)
    assert result.parse_status == "ok"
    assert isinstance(result.action, DeleteAgentAction)
    assert result.action.target == "agents/old_agent.py"


def test_parse_write_agent_bad_path_returns_noop() -> None:
    """A path traversal in write_agent collapses to noop (schema mismatch)."""
    transcript = _wrap_block(
        {
            "action": "write_agent",
            "target_file": "/etc/passwd",
            "content": "x = 1\n",
            "rationale": "escape",
        }
    )
    result = parse_action(transcript)
    assert result.parse_status == "schema_mismatch"
    assert result.action.action == "noop"


# ---------------------------------------------------------------------------
# 4. _apply_write_agent — happy paths
# ---------------------------------------------------------------------------


def test_apply_write_agent_creates_new_file(tmp_path: Path) -> None:
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    action = WriteAgentAction(
        target_file="agents/test_agent.py",
        content=_AGENT_SOURCE,
        rationale="add retrieval agent",
    )
    result = apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    agent_path = repo_root / "agents" / "test_agent.py"
    assert agent_path.is_file()
    assert "class TestAgent" in agent_path.read_text(encoding="utf-8")
    assert "agents/test_agent.py" in result.files_added
    assert result.files_changed == []
    assert "write_agent" in result.action_summary


def test_apply_write_agent_overwrites_existing_file(tmp_path: Path) -> None:
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    agent_path = repo_root / "agents" / "test_agent.py"
    agent_path.write_text("# old content\n", encoding="utf-8")

    action = WriteAgentAction(
        target_file="agents/test_agent.py",
        content=_AGENT_SOURCE,
        rationale="rewrite agent",
    )
    result = apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    assert "agents/test_agent.py" in result.files_changed
    assert "agents/test_agent.py" not in result.files_added
    assert "class TestAgent" in agent_path.read_text(encoding="utf-8")


def test_apply_write_agent_creates_nested_directory(tmp_path: Path) -> None:
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    action = WriteAgentAction(
        target_file="agents/retrieval/v2.py",
        content=_AGENT_SOURCE,
        rationale="nested agent",
    )
    result = apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    assert (repo_root / "agents" / "retrieval" / "v2.py").is_file()
    assert "agents/retrieval/v2.py" in result.files_added


# ---------------------------------------------------------------------------
# 5. _apply_write_agent — validation failure (no file written)
# ---------------------------------------------------------------------------


def test_apply_write_agent_ast_denylist_failure_no_file(tmp_path: Path) -> None:
    """Content with a forbidden string literal fails AST denylist.
    The target file must NOT be written."""
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    action = WriteAgentAction(
        target_file="agents/bad_agent.py",
        content='DATA = "golden_set.json"\n',
        rationale="should fail",
    )
    with pytest.raises(CodeValidationError, match="forbidden reference"):
        apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    assert not (repo_root / "agents" / "bad_agent.py").exists()


def test_apply_write_agent_import_failure_no_file(tmp_path: Path) -> None:
    """Content that fails to import is rejected. The target file must
    NOT be written."""
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    action = WriteAgentAction(
        target_file="agents/broken_agent.py",
        content="import nonexistent_module_xyz_123\n",
        rationale="should fail import",
    )
    with pytest.raises(CodeValidationError, match="failed to import"):
        apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    assert not (repo_root / "agents" / "broken_agent.py").exists()


def test_apply_write_agent_validation_failure_leaves_existing_untouched(
    tmp_path: Path,
) -> None:
    """If validation fails and the target file already exists, the
    old content must remain untouched."""
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    agent_path = repo_root / "agents" / "test_agent.py"
    original = "# original safe content\n"
    agent_path.write_text(original, encoding="utf-8")

    action = WriteAgentAction(
        target_file="agents/test_agent.py",
        content='DATA = "answer_key.yaml"\n',
        rationale="bad rewrite",
    )
    with pytest.raises(CodeValidationError):
        apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    assert agent_path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# 6. _apply_delete_agent — happy paths
# ---------------------------------------------------------------------------


def test_apply_delete_agent_removes_file(tmp_path: Path) -> None:
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    agent_path = repo_root / "agents" / "candidate.py"
    agent_path.write_text("# candidate\n", encoding="utf-8")

    action = DeleteAgentAction(
        target="agents/candidate.py",
        rationale="remove failed candidate",
    )
    result = apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    assert not agent_path.exists()
    assert result.files_removed == ["agents/candidate.py"]
    assert "delete_agent" in result.action_summary


def test_apply_delete_agent_missing_rejected(tmp_path: Path) -> None:
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    action = DeleteAgentAction(
        target="agents/nonexistent.py",
        rationale="gone",
    )
    with pytest.raises(ApplyError, match="does not exist"):
        apply_action(action, scaffold_root, mode="code", repo_root=repo_root)


# ---------------------------------------------------------------------------
# 7. _apply_delete_agent — agent_module protection
# ---------------------------------------------------------------------------


def test_apply_delete_agent_protected_file_path(tmp_path: Path) -> None:
    """When agent_module is a .py file path matching the delete target,
    the delete is rejected."""
    scaffold_root = _code_mode_scaffold(tmp_path, agent_module="agents/active_agent.py")
    repo_root = tmp_path

    agent_path = repo_root / "agents" / "active_agent.py"
    agent_path.write_text("# active\n", encoding="utf-8")

    action = DeleteAgentAction(
        target="agents/active_agent.py",
        rationale="oops",
    )
    with pytest.raises(ApplyError, match="cannot delete active agent"):
        apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    assert agent_path.is_file()


def test_apply_delete_agent_protected_dotted_path(tmp_path: Path) -> None:
    """When agent_module is a dotted path like agents.foo, the
    corresponding agents/foo.py is protected."""
    scaffold_root = _code_mode_scaffold(tmp_path, agent_module="agents.active_agent")
    repo_root = tmp_path

    agent_path = repo_root / "agents" / "active_agent.py"
    agent_path.write_text("# active\n", encoding="utf-8")

    action = DeleteAgentAction(
        target="agents/active_agent.py",
        rationale="try dotted",
    )
    with pytest.raises(ApplyError, match="cannot delete active agent"):
        apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    assert agent_path.is_file()


def test_apply_delete_agent_package_module_not_protected(tmp_path: Path) -> None:
    """agent_module=anvil.agents.baseline (a package module, not in
    agents/) does NOT protect agents/baseline.py — they're different
    files at different paths."""
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    agent_path = repo_root / "agents" / "baseline.py"
    agent_path.write_text("# not the package baseline\n", encoding="utf-8")

    action = DeleteAgentAction(
        target="agents/baseline.py",
        rationale="remove local copy",
    )
    result = apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    assert not agent_path.exists()
    assert "agents/baseline.py" in result.files_removed


def test_apply_delete_agent_protected_default_module(tmp_path: Path) -> None:
    """The default agent_module (anvil.agents.baseline) is a package
    module — it does not protect any file in agents/ by path, but the
    agents/baseline.py file (if it exists) is a separate file that CAN
    be deleted. This confirms the protection logic is path-based, not
    name-based."""
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    # Default config has agent_module: anvil.agents.baseline
    agent_path = repo_root / "agents" / "other.py"
    agent_path.write_text("# other\n", encoding="utf-8")

    action = DeleteAgentAction(
        target="agents/other.py",
        rationale="not protected",
    )
    apply_action(action, scaffold_root, mode="code", repo_root=repo_root)
    assert not agent_path.exists()


# ---------------------------------------------------------------------------
# 8. Mode-aware action validation
# ---------------------------------------------------------------------------


def test_write_agent_rejected_in_prompt_mode(tmp_path: Path) -> None:
    scaffold_root = _bare_scaffold(tmp_path)
    repo_root = tmp_path

    action = WriteAgentAction(
        target_file="agents/foo.py",
        content="x = 1\n",
        rationale="should fail",
    )
    with pytest.raises(ApplyError, match="only valid in code mode"):
        apply_action(action, scaffold_root, mode="prompt", repo_root=repo_root)


def test_delete_agent_rejected_in_prompt_mode(tmp_path: Path) -> None:
    scaffold_root = _bare_scaffold(tmp_path)
    repo_root = tmp_path

    action = DeleteAgentAction(
        target="agents/foo.py",
        rationale="should fail",
    )
    with pytest.raises(ApplyError, match="only valid in code mode"):
        apply_action(action, scaffold_root, mode="prompt", repo_root=repo_root)


def test_add_rule_rejected_in_code_mode(tmp_path: Path) -> None:
    from anvil.optimizer.actions import AddRuleAction

    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    action = AddRuleAction(
        target_file="rules/foo.md",
        content="# foo\n",
        rationale="should fail",
    )
    with pytest.raises(ApplyError, match="only valid in prompt mode"):
        apply_action(action, scaffold_root, mode="code", repo_root=repo_root)


def test_change_sampling_rejected_in_code_mode(tmp_path: Path) -> None:
    from anvil.optimizer.actions import ChangeSamplingAction

    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    action = ChangeSamplingAction(
        field="temperature",
        value=0.2,
        rationale="should fail",
    )
    with pytest.raises(ApplyError, match="only valid in prompt mode"):
        apply_action(action, scaffold_root, mode="code", repo_root=repo_root)


def test_noop_valid_in_code_mode(tmp_path: Path) -> None:
    from anvil.optimizer.actions import NoopAction

    scaffold_root = _code_mode_scaffold(tmp_path)
    result = apply_action(
        NoopAction(rationale="rest in code mode"),
        scaffold_root,
        mode="code",
        repo_root=tmp_path,
    )
    assert "noop" in result.action_summary


def test_noop_valid_in_prompt_mode(tmp_path: Path) -> None:
    from anvil.optimizer.actions import NoopAction

    scaffold_root = _bare_scaffold(tmp_path)
    result = apply_action(
        NoopAction(rationale="rest in prompt mode"),
        scaffold_root,
        mode="prompt",
    )
    assert "noop" in result.action_summary


def test_default_mode_is_prompt_rejects_code_actions(tmp_path: Path) -> None:
    """When no mode is passed, the default is prompt mode, so code
    actions are rejected."""
    scaffold_root = _bare_scaffold(tmp_path)
    repo_root = tmp_path

    action = WriteAgentAction(
        target_file="agents/foo.py",
        content="x = 1\n",
        rationale="should fail",
    )
    with pytest.raises(ApplyError, match="only valid in code mode"):
        apply_action(action, scaffold_root, repo_root=repo_root)


# ---------------------------------------------------------------------------
# 9. repo_root derivation
# ---------------------------------------------------------------------------


def test_write_agent_derives_repo_root_from_scaffold_parent(tmp_path: Path) -> None:
    """When repo_root is not passed, it is derived from
    scaffold_root.parent (the convention: scaffold/ and agents/ are
    siblings under the repo root)."""
    scaffold_root = _code_mode_scaffold(tmp_path)
    # Do NOT pass repo_root — let the applier derive it.

    action = WriteAgentAction(
        target_file="agents/derived.py",
        content=_AGENT_SOURCE,
        rationale="derive repo root",
    )
    result = apply_action(action, scaffold_root, mode="code")

    assert (tmp_path / "agents" / "derived.py").is_file()
    assert "agents/derived.py" in result.files_added


# ---------------------------------------------------------------------------
# 10. Code-mode commit includes agents/ files (B1)
# ---------------------------------------------------------------------------


def test_commit_includes_agents_directory(tmp_path: Path) -> None:
    """commit_all() must stage agents/ as well as scaffold/ so that
    code-mode mutations (write_agent) land in the commit. Without
    this, the round branch's merge/revert would miss or lose the
    mutation, since code-mode writes live under agents/ not scaffold/.
    """
    import subprocess

    from anvil.loop.git_ops import commit_all

    repo = tmp_path
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    # Seed scaffold/ + an initial commit so HEAD exists.
    (repo / "scaffold").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text("tools: []\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)

    # Simulate a code-mode mutation: write a file under agents/.
    (repo / "agents").mkdir()
    (repo / "agents" / "candidate.py").write_text("# new agent\n", encoding="utf-8")

    sha = commit_all(repo, message="round 001: write_agent agents/candidate.py")

    changed = subprocess.run(
        ["git", "-C", str(repo), "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "agents/candidate.py" in changed


def test_commit_all_noop_without_agents_dir(tmp_path: Path) -> None:
    """commit_all() must not blow up when agents/ is absent (prompt
    mode). ``git add`` on a missing pathspec is fatal, so the
    existence guard is load-bearing here."""
    import subprocess

    from anvil.loop.git_ops import commit_all

    repo = tmp_path
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "scaffold").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text("tools: []\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)

    # No agents/ dir, no changes — commit_all returns the current SHA.
    sha = commit_all(repo, message="noop round")
    assert (
        sha
        == subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


# ---------------------------------------------------------------------------
# 11. Symlink containment (B2)
# ---------------------------------------------------------------------------


def test_apply_write_agent_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlink inside agents/ pointing outside the repo must be
    rejected — the resolved path would escape agents/, allowing an
    arbitrary write. The lexical _check_agent_path cannot see this;
    the resolver-level _check_path_safe in the applier must catch it.
    """
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    external = repo_root / "external"
    external.mkdir()
    (repo_root / "agents" / "evil").symlink_to(external)

    action = WriteAgentAction(
        target_file="agents/evil/foo.py",
        content=_AGENT_SOURCE,
        rationale="escape via symlink",
    )
    with pytest.raises(ApplyError, match="resolves outside agents/"):
        apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    # No file was written outside agents/.
    assert not (external / "foo.py").exists()


def test_apply_delete_agent_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlink under agents/ pointing outside the repo must not let
    delete_agent erase an external file."""
    scaffold_root = _code_mode_scaffold(tmp_path)
    repo_root = tmp_path

    external = repo_root / "external"
    external.mkdir()
    (repo_root / "agents" / "evil").symlink_to(external)
    (external / "victim.py").write_text("# secret\n", encoding="utf-8")

    action = DeleteAgentAction(
        target="agents/evil/victim.py",
        rationale="escape via symlink",
    )
    with pytest.raises(ApplyError, match="resolves outside agents/"):
        apply_action(action, scaffold_root, mode="code", repo_root=repo_root)

    # The external file survives.
    assert (external / "victim.py").is_file()
    assert (external / "victim.py").read_text(encoding="utf-8") == "# secret\n"


# ---------------------------------------------------------------------------
# 12. Mode validation fails closed (B3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["hybrid", "", "cod", "CODE", "code "])
def test_validate_action_mode_rejects_unknown_mode(mode: str) -> None:
    """Unknown mode values must raise, not silently permit every
    action (the old code only fired on exact 'prompt'/'code' matches,
    so a typo let everything through)."""
    from anvil.optimizer.applier import _validate_action_mode

    with pytest.raises(ApplyError, match="unknown optimization mode"):
        _validate_action_mode("write_agent", mode)


def test_validate_action_mode_unknown_mode_rejects_prompt_action_too() -> None:
    """An unknown mode must reject even prompt-mode actions, so a
    typo can't widen the vocabulary either direction."""
    from anvil.optimizer.applier import _validate_action_mode

    with pytest.raises(ApplyError, match="unknown optimization mode"):
        _validate_action_mode("add_skill", "hybrid")


def test_read_optimization_mode_rejects_invalid_value(tmp_path: Path) -> None:
    """An invalid mode in config.yaml must fail at the source rather
    than reach the applier as a permissive unknown mode."""
    from anvil.loop.round import _read_optimization_mode

    scaffold_root = tmp_path / "scaffold"
    config = tmp_path / "harness" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("mode: hybrid\nruntime_endpoint: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown optimization mode"):
        _read_optimization_mode(scaffold_root)
