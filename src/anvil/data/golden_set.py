"""Loader for the synthetic NeoVolt golden evaluation set.

Reads ``data/golden_set.jsonl`` line-by-line and returns the list of
example dicts. The eval runner consumes them directly — every
example is required to carry: ``example_id``, ``query``, ``category``,
``expected_doc_ids``, ``reference_answer``, ``should_refuse``,
``expected_citations``, ``must_include``, ``must_not_include``,
``notes_for_judge``.

Includes :func:`select_subset` — bucket-aware deterministic sub-set
for the eval modes (quick/standard/full). Determinism is critical:
the same mode must always select the same rows so cached baselines
remain comparable across runs.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FIELDS: tuple[str, ...] = (
    "example_id",
    "query",
    "category",
    "expected_doc_ids",
    "reference_answer",
    "should_refuse",
    "expected_citations",
    "must_include",
    "must_not_include",
    "notes_for_judge",
)


def load_golden_set(golden_set_path: Path | str) -> list[dict]:
    """Parse ``golden_set_path`` and return the list of example dicts.

    Lightweight validation: every line must parse as JSON and carry
    the required fields. Cross-reference validation (KB doc_ids) is
    intentionally NOT done here — it belongs in a dedicated validator
    script that runs in CI; eval-side load stays fast.
    """
    path = Path(golden_set_path)
    if not path.is_file():
        raise FileNotFoundError(f"golden set not found: {path}")

    examples: list[dict] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"golden_set line {line_no}: invalid JSON: {exc}") from exc
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(
                f"golden_set line {line_no} ({row.get('example_id', '?')}): "
                f"missing fields {missing}"
            )
        examples.append(row)
    return examples


def select_subset(
    examples: list[dict],
    *,
    buckets: dict[str, int],
) -> list[dict]:
    """Return a deterministic sub-set respecting per-bucket counts.

    Args:
        examples: full list from :func:`load_golden_set`.
        buckets: mapping from category name (``direct``,
            ``multi_hop``, ``distractor``, ``out_of_scope``) to the
            number of rows to keep from that bucket.

    The selection is the **first N rows of each bucket in their
    natural order**. Rows are not shuffled; the JSONL file ordering
    is the canonical reference. This guarantees the same mode always
    produces the same rows (so cached baselines remain valid).
    """
    by_bucket: dict[str, list[dict]] = {}
    for ex in examples:
        by_bucket.setdefault(ex["category"], []).append(ex)

    out: list[dict] = []
    for bucket, count in buckets.items():
        rows = by_bucket.get(bucket, [])
        if len(rows) < count:
            raise ValueError(
                f"select_subset: bucket {bucket!r} has only {len(rows)} rows, requested {count}"
            )
        out.extend(rows[:count])
    return out
