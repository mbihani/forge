"""Tests for the rigorous prompt composer.

These tests pin the contracts the composer enforces and guard against
the bug we found in the legacy composer (rules with
``applies_to: optimizer`` leaking into the runtime prompt; missing
identity skill silently OK).
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from anvil.runtime.composer import (
    MissingIdentitySkillError,
    MissingScaffoldFileError,
    MultipleIdentitySkillsError,
    compose_prompt,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip("\n"), encoding="utf-8")


def _identity_skill(role: str = "You are a helpful test agent.") -> str:
    return f"""
        ---
        skill_id: identity
        kind: identity
        applies_to: runtime
        ---

        # Role
        {role}
    """


def _plain_skill(name: str, body: str) -> str:
    return f"""
        ---
        skill_id: {name}
        applies_to: runtime
        ---

        # {name}
        {body}
    """


def _rule(name: str, applies_to: str, body: str) -> str:
    return f"""
        ---
        rule_id: {name}
        applies_to: {applies_to}
        ---

        # {name}
        {body}
    """


def _scaffold(tmp_path: Path, harness: str, files: dict[str, str]) -> Path:
    root = tmp_path / "scaffold"
    _write(root / "harness.yaml", harness)
    for rel, content in files.items():
        _write(root / rel, content)
    return root


# ---------------------------------------------------------------------------
# Identity skill enforcement
# ---------------------------------------------------------------------------


def test_compose_emits_identity_first_even_when_listed_late(tmp_path: Path) -> None:
    """The identity skill is emitted first regardless of harness.yaml order."""
    harness = """
        skills:
          - file: greeting.md
          - file: identity.md
        rules: []
    """
    files = {
        "skills/greeting.md": _plain_skill("greeting", "Greet politely."),
        "skills/identity.md": _identity_skill("You are a NeoVolt assistant."),
    }
    root = _scaffold(tmp_path, harness, files)
    composed = compose_prompt(root, audience="runtime")

    # Identity body appears before the greeting body in the prompt text.
    assert composed.text.index("NeoVolt assistant") < composed.text.index("Greet politely")

    # Manifest records identity first.
    assert composed.manifest.files[0].kind == "identity"


def test_compose_fails_when_no_identity_skill(tmp_path: Path) -> None:
    """A scaffold without any identity skill must fail loudly."""
    harness = """
        skills:
          - file: greeting.md
        rules: []
    """
    files = {"skills/greeting.md": _plain_skill("greeting", "Greet politely.")}
    root = _scaffold(tmp_path, harness, files)

    with pytest.raises(MissingIdentitySkillError):
        compose_prompt(root, audience="runtime")


def test_compose_fails_with_multiple_identity_skills(tmp_path: Path) -> None:
    """Two skills both declaring kind=identity is a fatal error."""
    harness = """
        skills:
          - file: identity_a.md
          - file: identity_b.md
        rules: []
    """
    files = {
        "skills/identity_a.md": _identity_skill("Identity A."),
        "skills/identity_b.md": _identity_skill("Identity B."),
    }
    root = _scaffold(tmp_path, harness, files)

    with pytest.raises(MultipleIdentitySkillsError):
        compose_prompt(root, audience="runtime")


# ---------------------------------------------------------------------------
# applies_to filtering on rules
# ---------------------------------------------------------------------------


def test_runtime_audience_excludes_optimizer_only_rules(tmp_path: Path) -> None:
    """Rules with applies_to=optimizer must NOT enter the runtime prompt.

    This is the regression that ate our optimizer signal: the
    `no_repeat_failed_mutations` rule (applies_to: optimizer) was
    leaking into runtime turns.
    """
    harness = """
        skills:
          - file: identity.md
        rules:
          - file: optimizer_only.md
          - file: runtime_only.md
          - file: both_audiences.md
    """
    files = {
        "skills/identity.md": _identity_skill(),
        "rules/optimizer_only.md": _rule("optimizer_only", "optimizer", "DO NOT REPROPOSE."),
        "rules/runtime_only.md": _rule("runtime_only", "runtime", "BE CONCISE."),
        "rules/both_audiences.md": _rule("both_audiences", "both", "BE HONEST."),
    }
    root = _scaffold(tmp_path, harness, files)
    composed = compose_prompt(root, audience="runtime")

    assert "DO NOT REPROPOSE" not in composed.text
    assert "BE CONCISE" in composed.text
    assert "BE HONEST" in composed.text


def test_optimizer_audience_excludes_runtime_only_rules(tmp_path: Path) -> None:
    """Symmetric: composing for the optimizer drops runtime-only rules."""
    harness = """
        skills:
          - file: identity.md
        rules:
          - file: optimizer_only.md
          - file: runtime_only.md
          - file: both_audiences.md
    """
    files = {
        "skills/identity.md": _identity_skill(),
        "rules/optimizer_only.md": _rule("optimizer_only", "optimizer", "DO NOT REPROPOSE."),
        "rules/runtime_only.md": _rule("runtime_only", "runtime", "BE CONCISE."),
        "rules/both_audiences.md": _rule("both_audiences", "both", "BE HONEST."),
    }
    root = _scaffold(tmp_path, harness, files)
    composed = compose_prompt(root, audience="optimizer")

    assert "DO NOT REPROPOSE" in composed.text
    assert "BE CONCISE" not in composed.text
    assert "BE HONEST" in composed.text


def test_rule_without_applies_to_defaults_to_runtime(tmp_path: Path) -> None:
    """A rule that omits applies_to is treated as applies_to: runtime."""
    harness = """
        skills:
          - file: identity.md
        rules:
          - file: ambiguous.md
    """
    # Frontmatter present but no applies_to field.
    ambiguous = """
        ---
        rule_id: ambiguous
        ---

        # ambiguous
        DEFAULT TO RUNTIME.
    """
    files = {"skills/identity.md": _identity_skill(), "rules/ambiguous.md": ambiguous}
    root = _scaffold(tmp_path, harness, files)

    runtime = compose_prompt(root, audience="runtime").text
    optimizer = compose_prompt(root, audience="optimizer").text
    assert "DEFAULT TO RUNTIME" in runtime
    assert "DEFAULT TO RUNTIME" not in optimizer


# ---------------------------------------------------------------------------
# Manifest + missing-file errors
# ---------------------------------------------------------------------------


def test_manifest_records_files_in_emission_order(tmp_path: Path) -> None:
    harness = """
        skills:
          - file: greeting.md
          - file: identity.md
        rules:
          - file: runtime_rule.md
          - file: optimizer_rule.md
    """
    files = {
        "skills/greeting.md": _plain_skill("greeting", "G."),
        "skills/identity.md": _identity_skill(),
        "rules/runtime_rule.md": _rule("runtime_rule", "runtime", "R."),
        "rules/optimizer_rule.md": _rule("optimizer_rule", "optimizer", "O."),
    }
    root = _scaffold(tmp_path, harness, files)
    composed = compose_prompt(root, audience="runtime")

    paths = [f.path for f in composed.manifest.files]
    # identity first; then other skills in harness order; then runtime-applicable rules only.
    assert paths == [
        "scaffold/skills/identity.md",
        "scaffold/skills/greeting.md",
        "scaffold/rules/runtime_rule.md",
    ]
    assert all(len(f.sha256) == 64 for f in composed.manifest.files)


def test_missing_referenced_file_raises(tmp_path: Path) -> None:
    harness = """
        skills:
          - file: identity.md
          - file: not_there.md
        rules: []
    """
    files = {"skills/identity.md": _identity_skill()}
    root = _scaffold(tmp_path, harness, files)

    with pytest.raises(MissingScaffoldFileError):
        compose_prompt(root, audience="runtime")


# ---------------------------------------------------------------------------
# Real scaffold smoke test
# ---------------------------------------------------------------------------


def test_real_scaffold_composes_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke: the actual repo scaffold composes for both audiences.

    Uses the live ``scaffold/`` directory at the repo root, so this
    test acts as a guard against regressions in the shipping scaffold.
    """
    repo_root = Path(__file__).resolve().parent.parent
    scaffold_root = repo_root / "scaffold"

    runtime = compose_prompt(scaffold_root, audience="runtime")
    optimizer = compose_prompt(scaffold_root, audience="optimizer")

    # Identity skill is the first manifest entry for both audiences.
    assert runtime.manifest.files[0].kind == "identity"
    assert optimizer.manifest.files[0].kind == "identity"

    # The runtime prompt must NOT contain the meta-optimizer rule body.
    assert (
        "no_repeat_failed_mutations" not in runtime.text.lower()
        or "Loop-detection" not in runtime.text
    )

    # answer_scope_discipline is runtime-only — present on runtime, absent on optimizer.
    assert "Answer-scope discipline" in runtime.text
    assert "Answer-scope discipline" not in optimizer.text
