#!/usr/bin/env python3
"""Show what happened in one ANVIL round (or the current state).

Usage::

    # Current state (no round yet) — scaffold tree + cached baseline + manifest
    uv run python scripts/round_show.py

    # A specific round (after Phase 4)
    uv run python scripts/round_show.py 1

    # An arbitrary eval JSON (e.g. one of the bare evals)
    uv run python scripts/round_show.py --eval eval/runs/eval_standard_70fe6609.json

Output combines five artifacts so a reader can answer five questions at once:

  Q1 — what changed?           → git diff against main
  Q2 — why?                    → scaffold/memory/round_NNN_critique.md
  Q3 — what numbers?           → eval/runs/round_NNN.json
  Q4 — vs baseline?            → eval/runs/baseline.json comparison
  Q5 — which files in prompt?  → compose manifest (sha256 + path)

Phase-4 will add the optimizer trace URL and the
``anvil.default.mutations`` Delta row.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anvil.eval.cache import baseline_path  # noqa: E402
from anvil.runtime.composer import compose_prompt  # noqa: E402
from anvil.runtime.loader import load_harness  # noqa: E402

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _h1(text: str) -> str:
    return f"{BOLD}{CYAN}== {text}{RESET}"


def _h2(text: str) -> str:
    return f"{BOLD}-- {text}{RESET}"


def _color_delta(delta: float) -> str:
    if delta > 0:
        return f"{GREEN}+{delta:.3f}{RESET}"
    if delta < 0:
        return f"{RED}{delta:.3f}{RESET}"
    return f"{DIM}±0.000{RESET}"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def show_scaffold_tree(scaffold_root: Path) -> None:
    """Show the scaffold/ directory contents (files + sizes)."""
    print(_h1("Scaffold tree"))
    files = sorted(scaffold_root.rglob("*.md")) + sorted(scaffold_root.rglob("*.yaml"))
    files = [f for f in files if f.is_file()]
    for f in files:
        rel = f.relative_to(scaffold_root)
        size = f.stat().st_size
        print(f"  {DIM}{size:6d}{RESET}  scaffold/{rel}")
    print()


def show_compose_manifest(scaffold_root: Path) -> None:
    """Show which files contribute to the runtime + optimizer prompts."""
    print(_h1("Compose manifest (what enters each prompt)"))
    for audience in ("runtime", "optimizer"):
        print(_h2(audience))
        composed = compose_prompt(scaffold_root, audience=audience)
        for f in composed.manifest.files:
            kind = f"[{f.kind}]" if f.kind else ""
            applies = f"applies_to={f.applies_to}" if f.applies_to else ""
            extras = " ".join(x for x in (kind, applies) if x)
            print(f"  {f.sha256[:8]}  {f.role:5}  {f.path}  {DIM}{extras}{RESET}")
        n_chars = len(composed.text)
        print(f"  {DIM}prompt size: {n_chars} chars / ~{n_chars // 4} tokens{RESET}")
    print()


def show_baseline(repo_root: Path) -> dict | None:
    """Show the cached baseline aggregate + per-judge + per-bucket."""
    bp = baseline_path(repo_root)
    if not bp.is_file():
        print(_h1("Baseline cache"))
        print(f"  {DIM}none yet (run --refresh-baseline to create){RESET}")
        print()
        return None
    raw = json.loads(bp.read_text(encoding="utf-8"))
    print(_h1(f"Baseline cache · {raw['mode']} · sha {raw['scaffold_commit_sha'][:8]}"))
    print(
        f"  {BOLD}aggregate: {raw['aggregate']:.3f}{RESET}  "
        f"(n={raw['n_examples']}, scorers={','.join(raw['scorers'])})"
    )
    print(_h2("per-judge"))
    for name, value in raw["per_judge"].items():
        print(f"  {value:5.3f}  {name}")
    print(_h2("per-bucket"))
    judge_names = list(raw["per_judge"].keys())
    header = "  bucket".ljust(18) + "".join(j[:10].rjust(11) for j in judge_names)
    print(f"  {DIM}{header}{RESET}")
    for bucket, scores in raw["per_bucket"].items():
        line = f"  {bucket:14}"
        for j in judge_names:
            v = scores.get(j, 0.0)
            line += f"  {v:5.3f}    "
        print(line)
    print()
    return raw


def show_eval_report(eval_path: Path, baseline: dict | None) -> None:
    """Show one eval JSON: aggregate, per-judge, per-bucket, failures, vs baseline.

    Tolerant to two slightly different shapes:
    * eval-only JSONs (from ``scripts/evaluate.py``) carry ``n_rows``.
    * round JSONs (from ``loop.run_round``) carry ``n_examples`` instead.
    """
    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    label = raw.get("label") or eval_path.stem
    print(_h1(f"Eval report · {label}"))

    if "aggregate" not in raw or raw.get("aggregate") is None:
        print(f"  {YELLOW}no aggregate in JSON (likely a noop round){RESET}")
        decision = raw.get("decision", "?")
        action_kind = raw.get("action_kind", "?")
        parse_status = raw.get("parse_status", "?")
        print(f"  decision={decision} · action={action_kind} · parse_status={parse_status}")
        return

    agg = float(raw["aggregate"])
    n = raw.get("n_examples", raw.get("n_rows", "?"))
    mode = raw.get("mode", "?")
    line = f"  {BOLD}aggregate: {agg:.3f}{RESET}  (n={n}, mode={mode})"
    if baseline:
        delta = agg - float(baseline["aggregate"])
        line += f"  vs baseline: {_color_delta(delta)}"
    print(line)

    print(_h2("per-judge"))
    for name, value in raw["per_judge"].items():
        delta_str = ""
        if baseline and name in baseline.get("per_judge", {}):
            d = value - baseline["per_judge"][name]
            delta_str = f"  ({_color_delta(d)})"
        print(f"  {value:5.3f}  {name}{delta_str}")

    print(_h2("per-bucket"))
    judge_names = list(raw["per_judge"].keys())
    header = "  bucket".ljust(18) + "".join(j[:10].rjust(11) for j in judge_names)
    print(f"  {DIM}{header}{RESET}")
    for bucket, scores in raw["per_bucket"].items():
        line = f"  {bucket:14}"
        for j in judge_names:
            v = scores.get(j, 0.0)
            line += f"  {v:5.3f}    "
        print(line)

    failures = raw.get("failures", [])
    print(_h2(f"failures ({len(failures)})"))
    for f in failures:
        judges = ",".join(f.get("judge_failures", []))
        print(
            f"  {RED}{f['example_id']:11}{RESET}  {f['category']:12}  {DIM}fails:{RESET} {judges}"
        )
        print(f"               {DIM}{f.get('query', '')[:80]}{RESET}")
        print(f"               {DIM}trace_id: {f.get('trace_id')}{RESET}")

    print(_h2("mlflow"))
    # Round JSONs nest mlflow under "mlflow"; eval-only JSONs put run_id flat.
    mlflow_block = raw.get("mlflow") or {}
    run_id = mlflow_block.get("run_id") if mlflow_block else raw.get("run_id")
    experiment_id = mlflow_block.get("experiment_id") if mlflow_block else raw.get("experiment_id")
    print(f"  run_id: {run_id}")
    print(f"  experiment_id: {experiment_id}")
    print()


def show_round(round_id: int, repo_root: Path, baseline: dict | None) -> None:
    """Show one round end-to-end: diff + critique + eval + delta."""
    branch = f"anvil/exp-round-{round_id}"
    print(_h1(f"Round {round_id} · branch {branch}"))

    # 1. Decision marker (look up Delta if we wire it later; for now from JSON).
    json_path = repo_root / "eval" / "runs" / f"round_{round_id:03d}.json"
    if not json_path.is_file():
        print(f"  {YELLOW}no eval JSON at {json_path} — round may not have completed yet{RESET}")
        return

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    decision = raw.get("decision", "unknown")
    delta = raw.get("score_delta_vs_parent")
    line = f"  decision: {BOLD}{decision.upper()}{RESET}"
    if delta is not None:
        line += f"  Δ={_color_delta(float(delta))}"
    print(line)

    # 2. Mutation: git diff between main and the round branch.
    print(_h2("mutation (git diff)"))
    try:
        diff = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--stat", f"main..{branch}", "--", "scaffold/"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if diff.stdout.strip():
            for line in diff.stdout.splitlines():
                print(f"  {line}")
        else:
            print(f"  {DIM}(no scaffold changes — pure noop round?){RESET}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  {DIM}git not available{RESET}")

    # 3. Critique md. If missing, fall back to the rationale stored in
    # mutations.jsonl (the round.py bug that orphaned critique mds for
    # keep-rounds is a known issue; data still survives in mutations).
    critique_path = repo_root / "scaffold" / "memory" / f"round_{round_id:03d}_critique.md"
    print(_h2("critique"))
    if critique_path.is_file():
        text = critique_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for line in lines[:35]:
            print(f"  {line}")
        if len(lines) > 35:
            print(f"  {DIM}... ({len(lines) - 35} more lines){RESET}")
    else:
        print(f"  {YELLOW}critique md missing on disk — falling back to mutations log{RESET}")
        mutations_path = repo_root / "eval" / "mutations.jsonl"
        rationale = None
        if mutations_path.is_file():
            for line in mutations_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("round_id") == round_id:
                    rationale = record.get("diff_summary", "")
                    break
        if rationale:
            print(f"  decision: {raw.get('decision', '?')}")
            print(f"  action:   {raw.get('action_kind', '?')}")
            print(f"  parse:    {raw.get('parse_status', '?')}")
            print(f"  summary:  {rationale}")
        else:
            print(f"  {DIM}also no row in mutations.jsonl{RESET}")

    # 4. Eval body.
    print()
    show_eval_report(json_path, baseline)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Show ANVIL round (or current state) — diff, critique, eval, delta.",
    )
    p.add_argument(
        "round_id",
        nargs="?",
        type=int,
        default=None,
        help="round number to show (e.g. 1). If omitted, shows current state.",
    )
    p.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="path to an eval JSON to render (overrides round_id lookup)",
    )
    p.add_argument(
        "--scaffold",
        type=Path,
        default=REPO_ROOT / "scaffold",
        help="path to scaffold/ directory",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    scaffold_root = args.scaffold

    # Always show current state header.
    snapshot = load_harness(scaffold_root)
    print(_h1(f"ANVIL — repo {REPO_ROOT.name}"))
    sha_proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    branch_proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    print(f"  branch: {branch_proc.stdout.strip() or '?'}")
    print(f"  HEAD:   {sha_proc.stdout.strip()[:12] or '?'}")
    print(f"  runtime endpoint: {snapshot.runtime_endpoint}")
    print(f"  judge   endpoint: {snapshot.judge_endpoint}")
    print(f"  experiments:     {snapshot.config.experiments.eval}")
    print()

    show_scaffold_tree(scaffold_root)
    show_compose_manifest(scaffold_root)
    baseline = show_baseline(REPO_ROOT)

    if args.eval:
        show_eval_report(args.eval, baseline)
    elif args.round_id is not None:
        show_round(args.round_id, REPO_ROOT, baseline)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
