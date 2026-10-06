#!/usr/bin/env python3
"""Reconstruct missing critique markdown files from round JSONs + mutations log.

The legacy ``run_round`` (in this repo through commit ``d94b9e9``)
``commit_all`` only happens after ``apply_action`` and BEFORE the
critique md is written. So the critique md is left as untracked in
the working tree, never enters the round branch, and gets clobbered
by subsequent ``git checkout`` operations during the next round's
keep — so the critiques of R4, R6, R8 in the original 10-round
chain ended up missing.

The data still survives across two artifacts:

* ``eval/runs/round_NNN.json`` — decision, action_kind, parse_status,
  baseline_score, score_delta_vs_parent, aggregate, per_judge,
  per_bucket, failures, mlflow run_id.
* ``eval/mutations.jsonl`` — diff_summary (the optimizer's rationale,
  truncated at 120 chars).

This script regenerates a ``round_NNN_critique.md`` for any round
that has a JSON but no critique md, in the same format that
``run_round`` emits. Run idempotently — existing critique mds are
left alone.

Usage::

    uv run python scripts/reconstruct_critiques.py
    uv run python scripts/reconstruct_critiques.py --round-id 6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "eval" / "runs"
MEMORY_DIR = REPO_ROOT / "scaffold" / "memory"
MUTATIONS_PATH = REPO_ROOT / "eval" / "mutations.jsonl"

_ROUND_JSON_RE = re.compile(r"^round_(\d+)\.json$")


def _load_mutations_by_round() -> dict[int, dict]:
    if not MUTATIONS_PATH.is_file():
        return {}
    out: dict[int, dict] = {}
    for line in MUTATIONS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[int(r["round_id"])] = r
    return out


def _format_critique(round_data: dict, mutation: dict | None) -> str:
    decision = round_data.get("decision", "?")
    action_kind = round_data.get("action_kind", "?")
    parse_status = round_data.get("parse_status", "?")
    baseline_score = round_data.get("baseline_score")
    aggregate = round_data.get("aggregate")
    score_delta = round_data.get("score_delta_vs_parent")
    branch = round_data.get("branch", f"anvil/exp-round-{round_data.get('round_id')}")

    bs = f"{baseline_score:.4f}" if baseline_score is not None else "null"
    ms = f"{aggregate:.4f}" if aggregate is not None else "null"
    sd = f"{score_delta:+.4f}" if score_delta is not None else "null"

    summary = (mutation or {}).get("diff_summary") or "(no diff_summary in mutations log)"

    return f"""---
round: {round_data.get("round_id")}
branch: {branch}
decision: {decision}
action_kind: {action_kind}
parse_status: {parse_status}
baseline_score: {bs}
mutated_score: {ms}
score_delta: {sd}
reconstructed: true
reconstructed_from: [eval/runs/round_NNN.json, eval/mutations.jsonl]
---

# Round {round_data.get("round_id")} critique (reconstructed)

This file was reconstructed from the round JSON + mutations log
because the original critique md was orphaned by the ``run_round``
commit-ordering bug (the writer ran AFTER the only ``commit_all``,
leaving the file untracked in the working tree until a subsequent
round's ``git checkout`` clobbered it).

## Action applied (from mutations.diff_summary)

{summary}

## Outcome

Decision: **{decision.upper()}**. Score delta vs cached baseline: {sd}.

For the full optimizer reasoning, open the transcript at
``scaffold/memory/round_{round_data.get("round_id"):03d}_transcript.md``
(if present) — that file contains the verbatim Claude session.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round-id",
        type=int,
        default=None,
        help="reconstruct only this round (default: all missing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing critique mds",
    )
    args = parser.parse_args(argv)

    mutations = _load_mutations_by_round()

    json_files = sorted(
        (p for p in RUNS_DIR.glob("round_*.json")),
        key=lambda p: (
            int(_ROUND_JSON_RE.match(p.name).group(1)) if _ROUND_JSON_RE.match(p.name) else 0
        ),
    )

    n_written = 0
    for json_path in json_files:
        m = _ROUND_JSON_RE.match(json_path.name)
        if not m:
            continue
        round_id = int(m.group(1))
        if args.round_id is not None and round_id != args.round_id:
            continue

        critique_path = MEMORY_DIR / f"round_{round_id:03d}_critique.md"
        if critique_path.exists() and not args.force:
            print(f"R{round_id}: critique already present, skipping")
            continue

        try:
            round_data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"R{round_id}: ERROR parsing JSON: {exc}", file=sys.stderr)
            continue

        round_data["round_id"] = round_id
        mutation = mutations.get(round_id)
        text = _format_critique(round_data, mutation)
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        critique_path.write_text(text, encoding="utf-8")
        n_written += 1
        print(f"R{round_id}: wrote {critique_path.relative_to(REPO_ROOT)}")

    print(f"\n{n_written} critique md(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
