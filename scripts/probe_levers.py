#!/usr/bin/env python
"""Probe which runtime settings the serving endpoints accept, before a run.

Makes one tiny real call per combination of the engine's knobs (e.g. model
x input mode x reasoning_effort for savesage) through the engine's
registered lever probe, and writes ``eval/lever_probe.json``. The round
prompt then lists only verified combinations, and the engine refuses a
rejected one before calling the endpoint.

    uv run python scripts/probe_levers.py --profile fevm-stable

Re-run when the model allowlist or an endpoint changes. Commit the file
with the baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from anvil.eval.engines import load_lever_probe_fn  # noqa: E402
from anvil.eval.lever_probe import render_probe_md, write_lever_probe  # noqa: E402
from anvil.runtime.loader import default_runtime_config_path, load_harness  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scaffold", default=str(REPO_ROOT / "scaffold"), help="scaffold/ directory")
    p.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    args = p.parse_args(argv)

    scaffold = Path(args.scaffold)
    cfg = load_harness(scaffold, default_runtime_config_path(scaffold)).config
    engine = cfg.eval.engine
    probe = load_lever_probe_fn(engine)
    if probe is None:
        print(f"engine {engine!r} registers no lever probe; nothing to do.", file=sys.stderr)
        return 2
    if args.profile and args.profile != "DEFAULT":
        os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", args.profile)

    results = probe(
        scaffold_root=scaffold,
        runtime_config_path=default_runtime_config_path(scaffold),
        golden_set_path=str(REPO_ROOT / "data" / "golden_set.jsonl"),
        profile=args.profile,
    )
    path = write_lever_probe(REPO_ROOT, results, engine=engine)
    from anvil.eval.lever_probe import load_lever_probe  # noqa: PLC0415

    print(render_probe_md(load_lever_probe(REPO_ROOT)))
    n_ok = sum(1 for r in results if r.ok)
    print(f"{n_ok}/{len(results)} combinations accepted. Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
