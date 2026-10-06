#!/usr/bin/env python
"""Golden labels the optimizer flagged as wrong, for review.

The optimizer reports labels it believes are stale or inconsistent in its
actions (``label_disputes``); rounds collect them in
``eval/label_disputes.jsonl``. This prints them grouped by field:

    uv run python scripts/label_disputes.py > label_disputes.md

Records carry example ids, field names and reasons only — look the values up
in the golden set. Fix the labels there, regenerate the baseline, and delete
the resolved records.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anvil.eval.label_disputes import (  # noqa: E402
    load_label_disputes,
    render_disputes_report_md,
)


def main() -> int:
    print(render_disputes_report_md(load_label_disputes(REPO_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
