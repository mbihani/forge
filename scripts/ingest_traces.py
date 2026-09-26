#!/usr/bin/env python3
"""Freeze an experiment's labelled MLflow traces into an eval snapshot.

This is the one-time "freeze the dataset once per optimizer session"
primitive for the ``trace`` eval engine (``eval.engine: trace``). It reads
an MLflow experiment's traces — which already carry LLM-judge and human
assessments — projects each into a plain row, and writes a frozen JSONL
snapshot to ``eval.trace.snapshot_path``. The eval run
(:func:`anvil.domains.trace.eval.evaluate_trace`) reads that frozen
snapshot and NEVER calls the live tracing server, so every round is
reproducible and the ``per_judge`` key set is stable.

Multi-turn / tool-laden requests are skipped (counted, not guessed at).

Usage::

    uv run python scripts/ingest_traces.py --experiment-id 1234567890
    uv run python scripts/ingest_traces.py --experiment-id 1234 --max-traces 500
    uv run python scripts/ingest_traces.py --profile my-workspace --out data/snap.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anvil.domains.trace.ingest import (  # noqa: E402
    build_snapshot_rows,
    read_traces,
    write_snapshot,
)
from anvil.runtime.loader import default_runtime_config_path, load_harness  # noqa: E402


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--experiment-id",
        default=None,
        help="MLflow experiment id to ingest (default: eval.trace.experiment_id)",
    )
    p.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="cap on traces pulled (default: eval.trace.max_traces)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output snapshot path (default: eval.trace.snapshot_path, repo-relative)",
    )
    p.add_argument(
        "--scaffold",
        default=str(REPO_ROOT / "scaffold"),
        help="path to scaffold/ directory",
    )
    p.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="keep rows with identical query text (default: dedup to first)",
    )
    return p


def _resolve_out(out: str, repo_root: Path) -> Path:
    p = Path(out)
    return p if p.is_absolute() else repo_root / p


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)

    scaffold_path = Path(args.scaffold)
    runtime_config_path = default_runtime_config_path(scaffold_path)
    snapshot = load_harness(scaffold_path, runtime_config_path)
    trace_cfg = snapshot.config.eval.trace
    if trace_cfg is None:
        print(
            "ERROR: eval.trace is not configured in harness/config.yaml. "
            "Add an `eval: { engine: trace, trace: { experiment_id: ... } }` block.",
            file=sys.stderr,
        )
        return 2

    experiment_id = args.experiment_id or trace_cfg.experiment_id
    if not experiment_id:
        print(
            "ERROR: no experiment id. Pass --experiment-id or set "
            "eval.trace.experiment_id in harness/config.yaml.",
            file=sys.stderr,
        )
        return 2
    max_traces = args.max_traces or trace_cfg.max_traces

    # Point mlflow / the SDK at the selected profile.
    import os  # noqa: PLC0415

    import mlflow  # noqa: PLC0415

    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
        mlflow.set_tracking_uri(f"databricks://{args.profile}")

    print(f"Reading traces from experiment {experiment_id} (max {max_traces})...")
    traces = read_traces(experiment_id, max_traces)
    rows, counters = build_snapshot_rows(
        traces,
        include_source_types=list(trace_cfg.include_source_types),
        dedup=not args.no_dedup,
    )

    out_path = _resolve_out(args.out or trace_cfg.snapshot_path, REPO_ROOT)
    write_snapshot(out_path, rows)

    print(
        f"Snapshot written to {out_path}.\n"
        f"  rows kept:          {counters['kept']}\n"
        f"  expectations total: {counters['expectations']}\n"
        f"  human labels total: {counters['human_labels']}\n"
        f"  judge rubrics total:{counters['judge_rubrics']}\n"
        f"  skipped_multiturn:  {counters['skipped_multiturn']}\n"
        f"  skipped_no_labels:  {counters['skipped_no_labels']}\n"
        f"  deduped:            {counters['deduped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
