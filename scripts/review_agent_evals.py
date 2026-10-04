#!/usr/bin/env python3
"""Review the agent's EXISTING evals before optimizing (step one of a session).

Reads the agent's own MLflow experiment — the judges / scorers registered on
it, the judge, code and human assessments on its traces, and MLflow issue
findings — and writes:

* ``eval/agent_evals/inventory.json`` — the judges forge will re-use to score
  every round (exact registered scorer definitions), plus per-assessment stats.
* ``eval/agent_evals/review.md`` — what those judges and the human feedback
  point towards. Walk the user through it, then commit both files: the
  baseline and every round are scored with these judges, and the optimizer
  reads the review each round.

If the experiment has no judges forge can re-run, ask the user which metrics
the evals should use and record them under ``eval.agent_evals.user_metrics``.

Usage::

    uv run python scripts/review_agent_evals.py --profile my-workspace
    uv run python scripts/review_agent_evals.py --experiment-id 1234 --max-traces 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anvil.eval.agent_evals import discover_agent_evals, write_agent_evals  # noqa: E402
from anvil.runtime.loader import default_runtime_config_path, load_harness  # noqa: E402


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--experiment-id",
        default=None,
        help="the agent's eval experiment (default: eval.agent_evals.experiment_id)",
    )
    p.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="cap on traces read (default: eval.agent_evals.max_traces)",
    )
    p.add_argument(
        "--scaffold",
        default=str(REPO_ROOT / "scaffold"),
        help="path to scaffold/ directory",
    )
    p.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)

    scaffold_path = Path(args.scaffold)
    snapshot = load_harness(scaffold_path, default_runtime_config_path(scaffold_path))
    cfg = snapshot.config.eval.agent_evals
    experiment_id = args.experiment_id or (cfg.experiment_id if cfg else None)
    if not experiment_id:
        print(
            "ERROR: no experiment id. Pass --experiment-id or set "
            "eval.agent_evals.experiment_id in harness/config.yaml. If the agent has no "
            "evals at all, ask the user which metrics matter and record them under "
            "eval.agent_evals.user_metrics instead.",
            file=sys.stderr,
        )
        return 2
    max_traces = args.max_traces or (cfg.max_traces if cfg else 200)

    # Point mlflow / the SDK at the selected profile. A Databricks tracking
    # URI is also what lets MLflow load the agent's custom-code scorers.
    import os  # noqa: PLC0415

    import mlflow  # noqa: PLC0415

    if args.profile and args.profile != "DEFAULT":
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
        mlflow.set_tracking_uri(f"databricks://{args.profile}")
    else:
        mlflow.set_tracking_uri("databricks")

    print(f"Reviewing evals in experiment {experiment_id} (max {max_traces} traces)...")
    inv = discover_agent_evals(str(experiment_id), max_traces=max_traces)
    overrides = dict(cfg.guideline_overrides) if cfg else {}
    inv_path, review_path = write_agent_evals(REPO_ROOT, inv, overrides)

    print(review_path.read_text(encoding="utf-8"))
    print(f"Wrote {inv_path}\nWrote {review_path}")
    if not inv.judges:
        print(
            "\nNo judges found. Ask the user which metrics the evals should use and record "
            "them under eval.agent_evals.user_metrics before priming a baseline."
        )
    elif cfg is None or str(cfg.experiment_id or "") != str(experiment_id):
        print(
            f"\nSet eval.agent_evals.experiment_id: '{experiment_id}' in harness/config.yaml "
            "so the baseline and rounds use these judges."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
