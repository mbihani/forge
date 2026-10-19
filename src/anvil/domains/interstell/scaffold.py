"""Decompose the client's ``system_prompt.txt`` into forge scaffold skills.

The prompt is split at its top-level ``##`` section headers (in file order). The text
before the first ``##`` (role, mission, call-flow diagram, TRANSFER GATE) becomes the
identity skill; each section becomes one skill. ``compose_prompt`` re-joins the bodies with
blank lines, so the composed prompt equals the original up to blank-line runs at section
boundaries (:func:`same_up_to_blank_lines`). Skill files are forge's copy; the client's
file is never written.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

_SECTION = re.compile(r"^##(?!#)", re.MULTILINE)


def _slug(header: str, used: set[str]) -> str:
    words = re.findall(r"[a-z0-9]+", header.lower())[:5]
    base = "_".join(words) or "section"
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}_{n}", n + 1
    used.add(slug)
    return slug


def split_prompt(text: str) -> list[tuple[str, str]]:
    """``[(skill_id, body), ...]``: identity first, then one per ``##`` section, in order."""
    starts = [m.start() for m in _SECTION.finditer(text)]
    if not starts:
        return [("identity", text.strip())]
    chunks = [text[: starts[0]]] + [
        text[a:b] for a, b in zip(starts, [*starts[1:], len(text)], strict=True)
    ]
    used = {"identity"}
    out = [("identity", chunks[0].strip())]
    for chunk in chunks[1:]:
        header = chunk.strip().splitlines()[0].strip("# ")
        out.append((_slug(header, used), chunk.strip()))
    return out


def write_scaffold(
    scaffold_root: Path | str, prompt_text: str, *, levers: dict | None = None
) -> list[str]:
    """Write one skill file per section and point ``harness.yaml`` at them, in order."""
    root = Path(scaffold_root)
    skills_dir = root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for old in skills_dir.glob("*.md"):
        old.unlink()
    today = date.today().isoformat()
    files = []
    for skill_id, body in split_prompt(prompt_text):
        front = {"skill_id": skill_id, "applies_to": "runtime", "created_at": today}
        if skill_id == "identity":
            front = {"skill_id": "identity", "kind": "identity", "required": True, **front}
        name = f"{skill_id}.md"
        (skills_dir / name).write_text(
            "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n\n" + body + "\n",
            encoding="utf-8",
        )
        files.append(name)
    harness = {
        "sampling": {"temperature": 0.3, "max_tokens": 1024, "tool_choice": "auto"},
        "skills": [{"file": f} for f in files],
        "rules": [],
        "tools": [],
        **({"levers": levers} if levers else {}),
    }
    (root / "harness.yaml").write_text(
        "# Interstell prompt + levers session: skills = forge's copy of the client's\n"
        "# system_prompt.txt split at its ## sections (composed in this order and sent to\n"
        "# the App as a per-turn system_prompt override). The client's file is untouched.\n"
        + yaml.safe_dump(harness, sort_keys=False),
        encoding="utf-8",
    )
    return files


def same_up_to_blank_lines(a: str, b: str) -> bool:
    """Equal after collapsing runs of blank lines and trimming the ends."""

    def norm(s: str) -> str:
        return re.sub(r"\n\s*\n+", "\n\n", s.strip())

    return norm(a) == norm(b)
