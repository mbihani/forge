"""Deterministic per-field accuracy scoring for the Savesage ICICI domain.

This module is the eval *gradient* for the ANVIL loop on Savesage. It
reuses the SAME judge modules the production LLM-as-judge uses —
``judge.matching`` / ``judge.comparison`` / ``judge.aggregation`` — but
against a **cached Opus ground truth** instead of a fresh Opus call. No
LLM is invoked here, so the per-round signal is deterministic and free.

Faithfulness to the production metric is the whole point: the Savesage
judge (``judge/scorer.py``) is itself a *Luna-vs-Opus-GT per-field diff*
via exactly these modules, and its corpus aggregate
(``_aggregate_results``) is the **macro** mean over statements of each
statement's ``aggregate()`` reading. :func:`aggregate_corpus` reproduces
that arithmetic, so scoring cached outputs here reproduces the MLflow
``judge.*`` numbers (validated by ``scripts/reproduce_baseline_check.py``).

Two levels:

* :func:`score_extraction` — one statement: ``(expected, actual)`` parsed
  extraction dicts → strict / narration-forgiven / per-field accuracies.
* :func:`aggregate_corpus` — many statements → the corpus aggregate in
  the MLflow-compatible key shape the ANVIL report + baseline share.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from anvil.domains.savesage._statement_agent import ensure_importable

# The seven per-field accuracies the production judge reports per trace
# (``judge/scorer.py`` JUDGED_FIELDS). We surface the same seven so the
# deterministic report lines up 1:1 with the MLflow ``judge.<slug>``
# metrics. ``direction`` is scored inside strict accuracy but the judge
# does not report it as its own metric, so it is omitted here to keep
# the reproduce-the-baseline check exact. (raw judge path -> report slug,
# where the slug is MLflow's ``path.replace("[]","").replace(".","_")``.)
REPORTED_FIELDS: dict[str, str] = {
    "cards[].cardMeta.cardDisplayName": "cards_cardMeta_cardDisplayName",
    "cards[].cardMeta.lastFourDigit": "cards_cardMeta_lastFourDigit",
    "rewards.pointsEarnedThisCycle": "rewards_pointsEarnedThisCycle",
    "rewards.closingPoints": "rewards_closingPoints",
    "transactions[].date": "transactions_date",
    "transactions[].description": "transactions_description",
    "transactions[].amount": "transactions_amount",
}

# Report keys for the two overall readings (mirror MLflow ``judge.accuracy``
# / ``judge.accuracy_forgiven`` with the ``judge.`` prefix dropped).
STRICT_KEY = "accuracy"
FORGIVEN_KEY = "accuracy_forgiven"


def field_slug(judge_path: str) -> str:
    """MLflow-style slug for a raw judge field path.

    ``transactions[].description`` -> ``transactions_description``. Matches
    ``verdict_to_metrics``'s ``path.replace("[]","").replace(".","_")`` so a
    per-field diagnostic lines up 1:1 with the production ``judge.<slug>`` name.
    """
    return judge_path.replace("[]", "").replace(".", "_")


def _extraction(record_or_parsed: Any) -> dict:
    """Return the parsed-extraction dict from a record or a bare payload.

    The cached corpus stores the extraction under ``parsed_json``; a
    caller may also pass the already-unwrapped dict (e.g. a live Luna
    result). Anything else (None, a non-dict) becomes ``{}`` so the
    judge sees an empty statement rather than crashing — that scores as
    all-DISAGREE, the correct outcome for an unparseable extraction.
    """
    if isinstance(record_or_parsed, dict):
        if isinstance(record_or_parsed.get("parsed_json"), dict):
            return record_or_parsed["parsed_json"]
        return record_or_parsed
    return {}


def score_extraction(expected: Any, actual: Any) -> dict[str, Any]:
    """Score one statement's ``actual`` extraction against ``expected`` GT.

    Both arguments may be a full cached record (``{"parsed_json": ...}``)
    or the bare extraction dict. Returns::

        {
          "accuracy": float | None,            # strict, == judge.accuracy
          "accuracy_forgiven": float | None,   # == judge.accuracy_forgiven
          "per_field": {report_slug: float | None, ...},  # the 7 reported
          "raw_per_field": {judge_path: {"accuracy", "correct", "scored"}},
        }

    ``None`` accuracies mean "nothing scoreable" (every comparison was
    ABSENT_IN_PDF) — propagated, never coerced to 0, so the corpus
    macro-average skips them exactly as the production judge does.
    """
    ensure_importable()
    from contracts.models import Bank  # noqa: PLC0415 - lazy, see bridge
    from judge.aggregation import aggregate  # noqa: PLC0415
    from judge.comparison import build_comparisons  # noqa: PLC0415

    expected_dict = _extraction(expected)
    actual_dict = _extraction(actual)

    # build_comparisons only reads ``request.bank`` — a lightweight shim
    # avoids constructing a full ParseRequest (which needs pdf bytes).
    request = SimpleNamespace(bank=Bank.ICICI)
    comparisons = build_comparisons(request, expected_dict, actual_dict)
    rolled = aggregate(comparisons)

    per_field_raw = rolled["per_field"]
    per_field = {
        slug: (per_field_raw.get(path) or {}).get("accuracy")
        for path, slug in REPORTED_FIELDS.items()
    }
    return {
        STRICT_KEY: rolled["strict"]["accuracy"],
        FORGIVEN_KEY: rolled["narration_forgiven"]["accuracy"],
        "per_field": per_field,
        "raw_per_field": per_field_raw,
    }


def _mean(values: list[float]) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def gated_accuracy(scored: dict[str, Any], exclude_fields: frozenset[str] | None) -> float | None:
    """One statement's strict accuracy, optionally excluding some judged fields.

    Recomputes ``Σcorrect / Σscored`` from the statement's ``raw_per_field``
    counts (the same arithmetic the production judge's strict accuracy uses),
    dropping any path whose :func:`field_slug` is in ``exclude_fields``. This
    is how the code-mode/latency gate gets an accuracy FLOOR that measures
    real quality — the two stale-GT fields (``rewards.programType``,
    ``cards.cardMeta.productFamily``) are excluded so "accuracy held" is not
    partly measuring corrupt ground truth.

    Returns ``None`` when nothing remains scoreable (every non-excluded field
    was ABSENT_IN_PDF), so the corpus macro-average skips it. Falls back to
    the statement's plain ``accuracy`` when it carries no ``raw_per_field``
    (a hand-built input) or when ``exclude_fields`` is empty.
    """
    if not exclude_fields:
        return scored.get(STRICT_KEY)
    raw = scored.get("raw_per_field") or {}
    if not raw:
        return scored.get(STRICT_KEY)
    correct = scored_count = 0
    for path, stats in raw.items():
        if field_slug(path) in exclude_fields:
            continue
        correct += int((stats or {}).get("correct", 0) or 0)
        scored_count += int((stats or {}).get("scored", 0) or 0)
    return correct / scored_count if scored_count else None


def aggregate_corpus(
    per_statement: list[dict[str, Any]],
    *,
    exclude_fields: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Macro-average per-statement scores exactly like the production judge.

    ``per_statement`` is a list of :func:`score_extraction` results. The
    overall strict/forgiven and each per-field metric are the mean over
    statements that have a non-``None`` value for that metric — the
    arithmetic in ``judge/scorer.py::_aggregate_results``. Returns::

        {
          "aggregate": float,        # mean strict accuracy (the frontier objective)
          "per_judge": {             # diagnostics + potential objectives
             "accuracy": float, "accuracy_forgiven": float,
             "field_<slug>": float, ...
          },
          "per_field_counts": {slug: int, ...},
          "n_scored": int,
        }

    ``aggregate`` is the headline strict accuracy (``judge.accuracy``
    mean) — the single objective the milestone-1 frontier gate compares.
    A corpus with nothing scoreable yields ``aggregate = 0.0`` (a
    degenerate scaffold should not look like a perfect one).

    Per-field means cover EVERY judged field the statements scored — not
    only the seven MLflow-reported ones — so the optimizer can see the real
    laggards (network, productFamily, txnType, direction, …) that drag the
    28-field strict aggregate even when the seven headline fields are ~1.0.
    Sourced from each statement's ``raw_per_field`` (all judged paths) with
    the ``per_field`` slugs (the 7) merged in for hand-built inputs.

    ``exclude_fields`` (slugs) drops those judged fields from the headline
    ``aggregate`` only — used by the code-mode/latency gate so its accuracy
    FLOOR measures real quality (the stale-GT fields excluded). The full,
    un-gated strict number is always kept in ``per_judge["accuracy"]`` and
    the per-field means are unchanged, so diagnostics still show everything.
    With no exclusions ``aggregate`` is the plain strict mean (unchanged).
    """
    strict_vals = [s.get(STRICT_KEY) for s in per_statement]
    forgiven_vals = [s.get(FORGIVEN_KEY) for s in per_statement]

    overall_strict = _mean(strict_vals)
    overall_forgiven = _mean(forgiven_vals)

    per_judge: dict[str, float] = {}
    if overall_strict is not None:
        per_judge[STRICT_KEY] = overall_strict
    if overall_forgiven is not None:
        per_judge[FORGIVEN_KEY] = overall_forgiven

    # Gated accuracy = strict accuracy over the non-excluded judged fields,
    # macro-averaged. Becomes the headline ``aggregate`` (the frontier floor)
    # when fields are excluded; recorded under ``accuracy_gated`` for clarity.
    overall_gated = overall_strict
    if exclude_fields:
        overall_gated = _mean([gated_accuracy(s, exclude_fields) for s in per_statement])
        if overall_gated is not None:
            per_judge["accuracy_gated"] = overall_gated

    # Per-statement {slug: accuracy}, unioning the 7 reported slugs (present on
    # hand-built inputs) with every raw judged path (present on real scores).
    per_stmt_fields: list[dict[str, float | None]] = []
    all_slugs: set[str] = set()
    for s in per_statement:
        merged: dict[str, float | None] = dict(s.get("per_field") or {})
        for path, stats in (s.get("raw_per_field") or {}).items():
            merged.setdefault(field_slug(path), (stats or {}).get("accuracy"))
        per_stmt_fields.append(merged)
        all_slugs.update(merged)

    per_field_counts: dict[str, int] = {}
    for slug in sorted(all_slugs):
        vals = [f.get(slug) for f in per_stmt_fields]
        per_field_counts[slug] = sum(1 for v in vals if v is not None)
        mean = _mean(vals)
        if mean is not None:
            per_judge[f"field_{slug}"] = mean

    return {
        "aggregate": overall_gated if overall_gated is not None else 0.0,
        "per_judge": per_judge,
        "per_field_counts": per_field_counts,
        "n_scored": sum(1 for v in strict_vals if v is not None),
    }
