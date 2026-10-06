#!/usr/bin/env python
"""One table for a whole run: every round's change, decision and numbers.

Reads the committed records (``eval/runs/baseline.json``,
``eval/runs/round_NNN.json``, ``eval/runs/round_NNN.patch``) — no MLflow
needed — and prints markdown:

    uv run python scripts/round_summary.py            # all rounds
    uv run python scripts/round_summary.py --since 11 # rounds 11+

Columns: decision, action, what changed (the agent docstring's first line
for a code change, else the applier's summary), the gate objectives
(aggregate, median / p90 latency, cost per row), the levers it ran with,
and why it was decided (notes, e.g. a canary failure or an applier
rejection). Reverted rounds' code is in their ``.patch`` file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "eval" / "runs"


def _first_docstring_line(patch: str) -> str:
    """The first line of the module docstring a code change wrote, if any."""
    added = [ln[1:] for ln in patch.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    text = "\n".join(added)
    m = re.search(r'^\s*(?:"""|\'\'\')\s*(.+)', text, re.MULTILINE)
    return m.group(1).strip().rstrip('"').strip() if m else ""


def _cell(text: Any, limit: int = 110) -> str:
    s = str(text or "—").replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _ms(v: Any) -> str:
    return f"{v / 1000:.1f}s" if isinstance(v, int | float) else "—"


def _num(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, int | float) else "—"


def _usd(v: Any) -> str:
    return f"${v:.5f}" if isinstance(v, int | float) else "—"


def _row(name: str, rec: dict[str, Any], change: str) -> str:
    cm = rec.get("cost_metrics") or {}
    levers = ", ".join(f"{k}={v}" for k, v in sorted((rec.get("levers") or {}).items()))
    return (
        f"| {name} | {rec.get('decision', '—')} | {rec.get('action_kind', '—')} "
        f"| {_cell(change)} | {_num(rec.get('aggregate'))} "
        f"| {_ms(cm.get('latency_ms_median'))} | {_ms(cm.get('latency_ms_p90'))} "
        f"| {_usd(cm.get('cost_usd_per_row'))} | {_cell(levers, 60)} "
        f"| {_cell(rec.get('notes'), 80)} |"
    )


def render(runs_dir: Path, *, since: int = 0) -> str:
    lines = [
        "| Round | Decision | Action | Change | Aggregate | Median | p90 | $/row | Levers | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    baseline = runs_dir / "baseline.json"
    if baseline.is_file():
        b = json.loads(baseline.read_text(encoding="utf-8"))
        lines.append(
            _row("baseline", {**b, "decision": "—", "action_kind": "—"}, "(starting agent)")
        )
    for path in sorted(runs_dir.glob("round_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        rid = int(rec.get("round_id", 0) or 0)
        if rid < since:
            continue
        patch_path = path.with_suffix(".patch")
        patch = patch_path.read_text(encoding="utf-8") if patch_path.is_file() else ""
        change = _first_docstring_line(patch) or (rec.get("rationale") or "")[:200]
        lines.append(_row(str(rid), rec, change))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", default=str(RUNS), help="eval/runs directory")
    p.add_argument("--since", type=int, default=0, help="first round id to show")
    args = p.parse_args(argv)
    runs = Path(args.runs)
    if not runs.is_dir():
        print(f"no {runs}", file=sys.stderr)
        return 2
    print(render(runs, since=args.since))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
