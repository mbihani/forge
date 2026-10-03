#!/usr/bin/env python3
"""Sync the model price sheet into ``harness/model_catalog.csv``.

The loop never reads the Google Sheet directly — it reads the committed CSV
this script writes, so rounds are reproducible and run without Google auth.
Re-run after the sheet changes, review the diff, and commit it.

Auth: a Google OAuth token from gcloud Application Default Credentials
(``gcloud auth application-default login``), sent with the quota project
``ANVIL_GOOGLE_QUOTA_PROJECT`` (default ``gcp-dev-field-eng-aiapiquota``).

Usage::

    uv run python scripts/sync_model_catalog.py              # sheet from harness/config.yaml
    uv run python scripts/sync_model_catalog.py --sheet-id <id> --range Sheet1
    uv run python scripts/sync_model_catalog.py --from-json values.json   # offline
    uv run python scripts/sync_model_catalog.py --check databricks-claude-sonnet-4-6 ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from anvil.catalog import (  # noqa: E402
    fetch_sheet_values,
    load_catalog,
    model_prices,
    parse_sheet_rows,
    write_catalog_csv,
)
from anvil.runtime.models import ModelCatalogConfig  # noqa: E402

_DEFAULT_QUOTA_PROJECT = "gcp-dev-field-eng-aiapiquota"


def _catalog_config() -> ModelCatalogConfig:
    raw = yaml.safe_load((REPO_ROOT / "harness" / "config.yaml").read_text(encoding="utf-8")) or {}
    return ModelCatalogConfig.model_validate(raw.get("model_catalog") or {})


def _gcloud_token() -> str:
    proc = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    token = proc.stdout.strip()
    if proc.returncode != 0 or not token:
        raise SystemExit(
            "ERROR: no Google token. Run `gcloud auth application-default login` "
            "(or pass --from-json with an exported values file)."
        )
    return token


def main(argv: list[str] | None = None) -> int:
    cfg = _catalog_config()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sheet-id", default=cfg.sheet_id, help="Google Sheet id (default: config)")
    p.add_argument("--range", dest="sheet_range", default=cfg.sheet_range, help="A1 range / tab")
    p.add_argument("--out", default=cfg.path, help="output CSV (repo-relative)")
    p.add_argument("--from-json", help="parse a saved Sheets API values JSON instead of fetching")
    p.add_argument(
        "--check",
        nargs="*",
        default=None,
        help="print the price matched for these models from the committed CSV (no sync)",
    )
    args = p.parse_args(argv)

    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out

    if args.check is not None and not args.from_json:
        entries = load_catalog(out)
    else:
        if args.from_json:
            values = json.loads(Path(args.from_json).read_text(encoding="utf-8")).get("values", [])
            source = f"file {args.from_json}"
        else:
            if not args.sheet_id:
                print(
                    "ERROR: no sheet id (set model_catalog.sheet_id or pass --sheet-id)",
                    file=sys.stderr,
                )
                return 2
            quota = os.environ.get("ANVIL_GOOGLE_QUOTA_PROJECT", _DEFAULT_QUOTA_PROJECT)
            values = fetch_sheet_values(
                args.sheet_id, args.sheet_range, token=_gcloud_token(), quota_project=quota
            )
            source = f"Google Sheet {args.sheet_id} ({args.sheet_range})"
        entries = parse_sheet_rows(values)
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        write_catalog_csv(out, entries, header_note=f"synced from {source} at {stamp}")
        print(f"Wrote {len(entries)} priced rows to {out.relative_to(REPO_ROOT)}")

    if args.check:
        prices = model_prices(args.check, entries, tier=cfg.context_tier)
        for model in args.check:
            pr = prices.get(model)
            if pr is None:
                print(f"  {model}: NOT PRICED (add it to the sheet)")
            else:
                print(
                    f"  {model}: ${pr['input_per_million']:g} in / ${pr['output_per_million']:g} out "
                    f"per 1M  [{pr['source']}{': ' + pr['family'] if pr.get('family') else ''}]"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
