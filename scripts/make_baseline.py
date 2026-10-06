#!/usr/bin/env python3
"""Generate ``eval/runs/baseline.json`` for the current scaffold.

``scripts/evaluate.py --out`` writes an ``EvalReport`` — the eval
runner's own schema. The ANVIL loop's keep/revert gate, however,
reads a ``CachedBaseline`` from ``eval/runs/baseline.json`` (see
:func:`anvil.eval.load_baseline`). There was no supported path from a
fresh eval to that cache file, which blocked running ANVIL on a new
agent domain: without a baseline the gate has nothing to delta
against.

This script closes that gap. It runs the eval on the current
scaffold (the same logic as ``evaluate.py --mode quick``) and converts
the resulting ``EvalReport`` into a ``CachedBaseline``, sourcing the
three fields the eval does not know itself — ``scaffold_commit_sha``
from ``git rev-parse HEAD`` and ``runtime_endpoint`` /
``judge_endpoint`` from ``harness/config.yaml``.

Usage::

    uv run python scripts/make_baseline.py --mode quick
    uv run python scripts/make_baseline.py --mode standard --out eval/runs/baseline.json
    uv run python scripts/make_baseline.py --profile my-workspace --include-safety
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from anvil.eval.agent_evals import AgentEvalsNotReady, ensure_agent_evals_ready  # noqa: E402
from anvil.eval.cache import CachedBaseline, report_to_baseline  # noqa: E402
from anvil.eval.runner import evaluate_branch  # noqa: E402
from anvil.runtime.loader import default_runtime_config_path  # noqa: E402
from anvil.runtime.models import RuntimeYAML  # noqa: E402

# Default cache location the loop's gate reads. Relative to the repo
# root so ``--help`` shows a stable, machine-independent string.
DEFAULT_OUT_REL = "eval/runs/baseline.json"


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default=None,
        help="row sub-set size (default: harness/config.yaml > eval.default_mode)",
    )
    p.add_argument(
        "--out",
        default=None,
        help=f"output JSON path. Default: {DEFAULT_OUT_REL}",
    )
    p.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    p.add_argument(
        "--scaffold",
        default=str(REPO_ROOT / "scaffold"),
        help="path to scaffold/ directory",
    )
    p.add_argument(
        "--include-safety",
        action="store_true",
        help="evaluate Safety per-row (still excluded from aggregate)",
    )
    return p


def _load_endpoints(runtime_config_path: Path) -> tuple[str, str]:
    """Read ``runtime_endpoint`` + ``judge_endpoint`` from harness/config.yaml.

    Uses the same :class:`RuntimeYAML` schema the loader validates
    against, so a malformed config fails the same way ``evaluate.py``
    would — but without composing the runtime prompt, since the
    baseline only needs the two endpoint strings.
    """
    raw = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8")) or {}
    runtime = RuntimeYAML.model_validate(raw)
    return runtime.runtime_endpoint, runtime.judge_endpoint


def _git_head_sha(scaffold_root: Path) -> str:
    """``git rev-parse HEAD`` of the scaffold's repo (full 40-char SHA).

    Falls back to ``"unknown"`` if git is unavailable or the repo has
    no commits yet, mirroring ``evaluate.py``'s defensive SHA lookup.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(scaffold_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def build_baseline(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    profile: str = "DEFAULT",
    mode: str | None = None,
    include_safety: bool = False,
) -> CachedBaseline:
    """Run the eval on the current scaffold and build a ``CachedBaseline``.

    This is the programmatic entry point ``main()`` wraps; tests call
    it with ``evaluate_branch`` monkeypatched so no LLM is invoked.
    """
    scaffold_path = Path(scaffold_root)
    runtime_path = (
        Path(runtime_config_path)
        if runtime_config_path is not None
        else default_runtime_config_path(scaffold_path)
    )
    runtime_endpoint, judge_endpoint = _load_endpoints(runtime_path)

    # Forward the resolved config path so the eval runs against the SAME
    # harness/config.yaml the endpoints above were read from. Without this
    # evaluate_branch() falls back to default_runtime_config_path() and the
    # recorded endpoints can diverge from the config that produced the eval
    # — a misleading baseline cache.
    report = evaluate_branch(
        scaffold_root=scaffold_path,
        runtime_config_path=runtime_path,
        profile=profile,
        mode=mode,
        include_safety=include_safety,
    )
    scaffold_commit_sha = _git_head_sha(scaffold_path)
    return report_to_baseline(
        report,
        scaffold_commit_sha=scaffold_commit_sha,
        runtime_endpoint=runtime_endpoint,
        judge_endpoint=judge_endpoint,
    )


def main(argv: list[str] | None = None) -> int:
    # Line-buffer so a run redirected to a log file shows progress live.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(line_buffering=True)
    args = _arg_parser().parse_args(argv)

    # Forge starts from the agent's existing evals: refuse to prime a
    # baseline until they were reviewed (or metrics agreed with the user).
    scaffold_path = Path(args.scaffold)
    config_path = default_runtime_config_path(scaffold_path)
    runtime = RuntimeYAML.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    )
    try:
        ensure_agent_evals_ready(runtime.eval, scaffold_path.parent)
    except AgentEvalsNotReady as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    baseline = build_baseline(
        scaffold_root=args.scaffold,
        profile=args.profile,
        mode=args.mode,
        include_safety=args.include_safety,
    )

    out_path = Path(args.out) if args.out else REPO_ROOT / DEFAULT_OUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline.to_dict(), indent=2) + "\n", encoding="utf-8")

    print(
        f"Baseline written to {out_path}. Aggregate: {baseline.aggregate:.4f}. "
        f"{baseline.n_examples} examples, {baseline.mode} mode."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
