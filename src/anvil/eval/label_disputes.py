"""Ground-truth labels the optimizer believes are wrong, collected for the user.

A golden set can be stale or inconsistent (a label the agent's own rules
forbid, two identical cases labelled differently). The optimizer notices,
but without a place to say so it re-analyzes the same labels every round —
or worse, games them. Instead, any action may carry ``label_disputes``
(example, field, reason); the round appends them to
``eval/label_disputes.jsonl`` (one record per example+field, first report
wins) and every later round prompt lists them as already reported. The
user reviews ``scripts/label_disputes.py`` output and fixes the golden set.

Records hold ids, field names and reasons only — never label values (the
golden set may hold PII).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DISPUTES_REL = Path("eval") / "label_disputes.jsonl"
_PROMPT_MAX_FIELDS = 30
_PROMPT_MAX_IDS = 8


def load_label_disputes(repo_root: Path | str) -> list[dict[str, Any]]:
    path = Path(repo_root) / DISPUTES_REL
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("example_id") and rec.get("field"):
            out.append(rec)
    return out


def append_label_disputes(repo_root: Path | str, round_id: int, disputes: list[Any]) -> int:
    """Record new disputes from round ``round_id``; return how many were new."""
    if not disputes:
        return 0
    seen = {(d["example_id"], d["field"]) for d in load_label_disputes(repo_root)}
    path = Path(repo_root) / DISPUTES_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    new = 0
    with path.open("a", encoding="utf-8") as fh:
        for d in disputes:
            rec = d if isinstance(d, dict) else d.model_dump()
            key = (str(rec["example_id"]), str(rec["field"]))
            if key in seen:
                continue
            seen.add(key)
            fh.write(
                json.dumps(
                    {
                        "round_id": round_id,
                        "example_id": key[0],
                        "field": key[1],
                        "reason": str(rec.get("reason", "")),
                        "reported_at": now,
                    }
                )
                + "\n"
            )
            new += 1
    return new


def _by_field(disputes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for d in disputes:
        out.setdefault(str(d["field"]), []).append(d)
    return dict(sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def render_disputes_md(disputes: list[dict[str, Any]]) -> str:
    """Round-prompt section: labels already reported, so they are not re-analyzed."""
    if not disputes:
        return (
            "## Disputed labels\n\n"
            "If a golden label looks wrong (it contradicts the agent's own rules, or "
            "identical cases are labelled differently), do not tune the agent to it: "
            "add it to `label_disputes` in your action (see `prompts/anvil-round.md`). "
            "The user reviews and fixes the golden set.\n"
        )
    groups = _by_field(disputes)
    lines = [
        f"## Disputed labels already reported ({len(disputes)})",
        "",
        "Already flagged for the user's review — do not re-analyze them or tune the "
        "agent to them. Report NEW ones in `label_disputes` in your action.",
        "",
    ]
    for field, items in list(groups.items())[:_PROMPT_MAX_FIELDS]:
        ids = [str(d["example_id"]) for d in items]
        shown = ", ".join(ids[:_PROMPT_MAX_IDS]) + (
            f" (+{len(ids) - _PROMPT_MAX_IDS} more)" if len(ids) > _PROMPT_MAX_IDS else ""
        )
        lines.append(f"- `{field}` ({len(items)}): {shown} — {items[0].get('reason', '')}")
    if len(groups) > _PROMPT_MAX_FIELDS:
        lines.append(f"- … {len(groups) - _PROMPT_MAX_FIELDS} more fields")
    return "\n".join(lines) + "\n"


def render_disputes_report_md(disputes: list[dict[str, Any]]) -> str:
    """Full report for the user: every disputed label, grouped by field."""
    if not disputes:
        return "# Disputed labels\n\nNone reported.\n"
    lines = [f"# Disputed labels ({len(disputes)})", ""]
    for field, items in _by_field(disputes).items():
        lines += [
            f"## `{field}` ({len(items)})",
            "",
            "| example | round | reason |",
            "|---|---|---|",
        ]
        for d in items:
            reason = str(d.get("reason", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{d['example_id']}` | {d.get('round_id', '')} | {reason} |")
        lines.append("")
    return "\n".join(lines)
