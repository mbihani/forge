#!/usr/bin/env python3
"""CLI driver for ``anvil.loop.round.run_round``.

Usage::

    # Run one round (auto-detects round_id from existing branches)
    uv run python scripts/run_round.py

    # Run a specific round id
    uv run python scripts/run_round.py --round-id 1

    # Run N consecutive rounds
    uv run python scripts/run_round.py --rounds 3

    # Choose eval mode (default = harness/config.yaml > eval.default_mode)
    uv run python scripts/run_round.py --eval-mode quick

The runner expects:

* The repo's parent branch (default ``anvil/exp``) to exist.
* ``eval/runs/baseline.json`` to be cached (run
  ``scripts/evaluate.py --mode full`` first if missing).
* Optimizer auth is handled automatically by Claude Code via the
  Databricks CLI (``CLAUDE_CODE_USE_GATEWAY=1``), using the active
  profile / ``DATABRICKS_HOST`` — no secret is required. An operator
  may set ``ANTHROPIC_AUTH_TOKEN`` in the env as an optional override.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from datetime import UTC, datetime  # noqa: E402

from anvil.loop.git_ops import check_clean_worktree  # noqa: E402
from anvil.loop.optimizer_artifacts import (  # noqa: E402
    close_cli_session_sink,
    open_cli_session_sink,
)
from anvil.loop.round import run_round  # noqa: E402
from anvil.observability import configure_tracking_uri  # noqa: E402


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round-id", type=int, default=None, help="explicit round id")
    p.add_argument("--rounds", type=int, default=1, help="number of consecutive rounds")
    p.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    p.add_argument("--parent-branch", default="anvil/exp", help="branch to fork from + ff-merge to")
    p.add_argument(
        "--eval-mode",
        choices=["quick", "standard", "full"],
        default=None,
        help="eval mode for the post-mutation eval",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="hard cap on optimizer CLI turns per round",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="skip the clean-worktree safety check",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="run even when eval/runs/finalized.json exists",
    )
    return p


def _next_round_id(repo_root: Path) -> int:
    """Highest existing eval/runs/round_NNN.json + 1, or 1 if none."""
    runs = (repo_root / "eval" / "runs").glob("round_*.json")
    nums = []
    for p in runs:
        m = re.search(r"round_(\d+)\.json$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)

    finalized_path = REPO_ROOT / "eval" / "runs" / "finalized.json"
    if finalized_path.is_file() and not args.force:
        print(f"ERROR: optimization is finalized ({finalized_path}); pass --force to override.")
        return 1

    if args.allow_dirty:
        print("WARNING: --allow-dirty specified; skipping clean-worktree safety check.")
    else:
        check_clean_worktree()

    # Verify parent branch exists.
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", args.parent_branch],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"ERROR: parent branch {args.parent_branch!r} does not exist.")
        print(f"Create it first: git -C {REPO_ROOT} checkout -b {args.parent_branch} main")
        return 2

    next_id = args.round_id if args.round_id is not None else _next_round_id(REPO_ROOT)

    # Send traces + optimizer transcripts to the Databricks workspace (not a
    # local mlflow.db that a sandbox wipes), then open ONE parent run for this
    # session's optimizer transcripts/critiques — the same durable sink the
    # orchestrator uses. Best-effort: a failed open just skips persistence.
    print(f"MLflow tracking URI: {configure_tracking_uri(args.profile)}")
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    session_id = f"cli-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    sink = open_cli_session_sink(REPO_ROOT, session_id=session_id)
    if sink is not None:
        print(f"Optimizer transcripts → MLflow run {sink.run_id} (experiment {sink.experiment_id})")

    round_ids: list[int] = []
    status = "FAILED"
    try:
        for i in range(args.rounds):
            rid = next_id + i
            print(f"\n=== round {rid} ===")
            report = run_round(
                round_id=rid,
                repo_root=REPO_ROOT,
                profile=args.profile,
                parent_branch=args.parent_branch,
                eval_mode=args.eval_mode,
                max_turns=args.max_turns,
                artifact_sink=sink,
            )
            round_ids.append(rid)
            print(
                f"=== round {rid} done · {report.decision} · "
                f"action={report.action_kind} · Δ={report.score_delta}\n"
            )
        status = "FINISHED"
    finally:
        close_cli_session_sink(
            sink,
            REPO_ROOT,
            round_ids=round_ids,
            session_meta={
                "session_id": session_id,
                "surface": "cli",
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            status=status,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
