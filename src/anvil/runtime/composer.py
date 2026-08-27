"""Rigorous prompt composer for the ANVIL runtime.

Reads ``scaffold/harness.yaml`` (skills order + rules order),
parses each referenced markdown file's frontmatter, filters by
``applies_to``, validates the identity skill is registered, and
composes the system prompt deterministically.

Three contracts the composer enforces (and that the legacy composer
violated):

1. **An identity skill must be registered.** A skill is the identity
   skill if its frontmatter declares ``kind: identity``. The composer
   raises ``MissingIdentitySkillError`` if zero or multiple identity
   skills are present.

2. **The identity skill is emitted first** in the prompt regardless
   of its position in ``harness.yaml`` (so the optimizer cannot
   accidentally bury it).

3. **Rules are filtered by ``applies_to``.** When composing for
   ``runtime``, only rules with ``applies_to ∈ {runtime, both}`` (or
   missing, default ``runtime``) are included. When composing for
   ``optimizer``, only ``{optimizer, both}``. This is what fixes the
   bug where ``no_repeat_failed_mutations`` (applies_to: optimizer)
   was leaking into the runtime prompt.

The composer also returns a **manifest** so callers can log it as a
trace tag: which files were composed, in what order, with what
sha256. ``git checkout <scaffold_commit_sha>`` reproduces the prompt
byte-for-byte.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

Audience = Literal["runtime", "optimizer"]

_RUNTIME_AUDIENCES = {"runtime", "both"}
_OPTIMIZER_AUDIENCES = {"optimizer", "both"}
_DEFAULT_RULE_APPLIES_TO = "runtime"


class ComposerError(Exception):
    """Base for composer-side validation errors."""


class MissingIdentitySkillError(ComposerError):
    """No skill with ``kind: identity`` was registered."""


class MultipleIdentitySkillsError(ComposerError):
    """More than one skill with ``kind: identity`` was registered."""


class MissingScaffoldFileError(ComposerError):
    """A file referenced from ``harness.yaml`` does not exist."""


@dataclass(frozen=True)
class ComposedFile:
    """One file that contributed to the composed prompt."""

    path: str
    role: Literal["skill", "rule"]
    kind: str | None
    applies_to: str | None
    sha256: str


@dataclass(frozen=True)
class ComposeManifest:
    """Manifest returned alongside the composed prompt."""

    audience: Audience
    files: list[ComposedFile] = field(default_factory=list)


@dataclass(frozen=True)
class ComposedPrompt:
    text: str
    manifest: ComposeManifest


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """Split a markdown file's YAML frontmatter from its body.

    Frontmatter is delimited by ``---`` lines at the very top. Returns
    ``({}, raw)`` if no frontmatter is present.
    """
    if not raw.startswith("---"):
        return {}, raw
    # Find the closing fence.
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    yaml_block = raw[3:end].strip()
    body_start = end + 4  # skip "\n---"
    if body_start < len(raw) and raw[body_start] == "\n":
        body_start += 1
    body = raw[body_start:]
    metadata = yaml.safe_load(yaml_block) or {}
    if not isinstance(metadata, dict):
        # Malformed frontmatter is treated as no frontmatter.
        return {}, raw
    return metadata, body


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_scaffold_file(
    scaffold_root: Path, kind: Literal["skill", "rule"], filename: str
) -> tuple[dict, str, str]:
    """Read one scaffold file. Return (frontmatter, body, sha256)."""
    subdir = "skills" if kind == "skill" else "rules"
    path = scaffold_root / subdir / filename
    if not path.is_file():
        raise MissingScaffoldFileError(
            f"{kind} '{filename}' referenced from harness.yaml not found at {path}"
        )
    raw = path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(raw)
    return fm, body, _hash(raw)


def _select_skills(
    scaffold_root: Path,
    skill_entries: list[dict],
) -> tuple[ComposedFile, list[ComposedFile], list[str]]:
    """Resolve the registered skills.

    Returns ``(identity_file, other_files, body_strings)`` where
    ``body_strings`` is in emission order: identity first, then the
    rest in the order they appear in ``harness.yaml``.

    Raises:
        MissingIdentitySkillError: if zero skills declare
            ``kind: identity``.
        MultipleIdentitySkillsError: if more than one does.
    """
    identity: tuple[ComposedFile, str] | None = None
    others: list[tuple[ComposedFile, str]] = []

    for entry in skill_entries:
        filename = entry["file"] if isinstance(entry, dict) else str(entry)
        fm, body, sha = _read_scaffold_file(scaffold_root, "skill", filename)
        kind = fm.get("kind")
        composed = ComposedFile(
            path=f"scaffold/skills/{filename}",
            role="skill",
            kind=kind,
            applies_to=fm.get("applies_to"),
            sha256=sha,
        )
        if kind == "identity":
            if identity is not None:
                raise MultipleIdentitySkillsError(
                    f"more than one identity skill registered: "
                    f"{identity[0].path} and {composed.path}"
                )
            identity = (composed, body)
        else:
            others.append((composed, body))

    if identity is None:
        registered = ", ".join(e["file"] for e in skill_entries)
        raise MissingIdentitySkillError(
            "no skill with `kind: identity` is registered in "
            f"harness.yaml > skills (registered: {registered})"
        )

    bodies = [identity[1], *(b for _, b in others)]
    return identity[0], [c for c, _ in others], bodies


def _select_rules(
    scaffold_root: Path,
    rule_entries: list[dict],
    audience: Audience,
) -> tuple[list[ComposedFile], list[str]]:
    """Resolve and filter the registered rules by ``applies_to``."""
    audiences = _RUNTIME_AUDIENCES if audience == "runtime" else _OPTIMIZER_AUDIENCES
    files: list[ComposedFile] = []
    bodies: list[str] = []
    for entry in rule_entries:
        filename = entry["file"] if isinstance(entry, dict) else str(entry)
        fm, body, sha = _read_scaffold_file(scaffold_root, "rule", filename)
        applies_to = fm.get("applies_to") or _DEFAULT_RULE_APPLIES_TO
        if applies_to not in audiences:
            continue
        files.append(
            ComposedFile(
                path=f"scaffold/rules/{filename}",
                role="rule",
                kind=fm.get("kind"),
                applies_to=applies_to,
                sha256=sha,
            )
        )
        bodies.append(body)
    return files, bodies


def compose_prompt(
    scaffold_root: Path | str,
    *,
    audience: Audience = "runtime",
) -> ComposedPrompt:
    """Compose the system prompt for the given audience.

    Reads ``<scaffold_root>/harness.yaml`` for skills + rules order,
    resolves each referenced file, validates the identity skill,
    filters rules by ``applies_to``, and concatenates in order:
    identity skill → other skills → applicable rules.

    Args:
        scaffold_root: Path to the ``scaffold/`` directory.
        audience: Either ``"runtime"`` or ``"optimizer"``.

    Returns:
        :class:`ComposedPrompt` with ``text`` (the prompt) and
        ``manifest`` (which files contributed, in order, with hashes).

    Raises:
        MissingIdentitySkillError: see :func:`_select_skills`.
        MultipleIdentitySkillsError: see :func:`_select_skills`.
        MissingScaffoldFileError: a referenced file is missing.
    """
    root = Path(scaffold_root)
    harness = yaml.safe_load((root / "harness.yaml").read_text(encoding="utf-8"))
    skill_entries = harness.get("skills") or []
    rule_entries = harness.get("rules") or []

    identity_file, other_skill_files, skill_bodies = _select_skills(root, skill_entries)
    rule_files, rule_bodies = _select_rules(root, rule_entries, audience)

    parts: list[str] = []
    parts.extend(b.strip() for b in skill_bodies if b.strip())
    if rule_bodies:
        parts.append("# Workflow rules")
        parts.extend(b.strip() for b in rule_bodies if b.strip())
    text = "\n\n".join(parts) + "\n"

    manifest = ComposeManifest(
        audience=audience,
        files=[identity_file, *other_skill_files, *rule_files],
    )
    return ComposedPrompt(text=text, manifest=manifest)
