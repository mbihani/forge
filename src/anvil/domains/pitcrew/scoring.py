"""Pitcrew scoring — deterministic structure checks + a PDF-grounded judge.

Three objectives, mirroring the assessments pitcrew already logs on its
own traces (``output_json_valid``, ``schema_adherence``,
``summary_quality``):

* ``output_json_valid`` — deterministic 1.0/0.0: the output parses as JSON
  and carries the top-level schema (``document_title`` + ``sections`` list).
* ``schema_adherence`` — deterministic [0,1]: the FRACTION of sections that
  are well-formed (id/title/level/summary/citations all present and
  correctly typed). A continuous gradient, not a pass/fail.
* ``summary_quality`` — an LLM judge GROUNDED on the source PDF (the judge
  sees the same ``document`` block the summarizer saw), scoring accuracy,
  completeness, faithfulness, and conciseness 1-5 each, normalized to [0,1].

The aggregate weights ``summary_quality`` above the two structural checks
(it is the real optimization target); weights live in
:data:`OBJECTIVE_WEIGHTS`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# Objective weights for the weighted-mean aggregate. summary_quality is the
# quality target; the structural checks are guards with lower weight.
OBJECTIVE_WEIGHTS: dict[str, float] = {
    "summary_quality": 2.0,
    "schema_adherence": 1.0,
    "output_json_valid": 1.0,
}

OBJECTIVE_KEYS: tuple[str, ...] = tuple(sorted(OBJECTIVE_WEIGHTS))

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json(output: str) -> dict | None:
    """Parse the model output into a dict, tolerating markdown fences.

    Returns None if no JSON object can be recovered.
    """
    text = (output or "").strip()
    if not text:
        return None
    # Strip a leading/trailing ``` or ```json fence if present.
    text = _FENCE_RE.sub("", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass
    # Fall back to the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def score_json_valid(parsed: dict | None) -> float:
    """1.0 iff the output parsed and carries the top-level schema."""
    if not isinstance(parsed, dict):
        return 0.0
    if not isinstance(parsed.get("document_title"), str):
        return 0.0
    if not isinstance(parsed.get("sections"), list):
        return 0.0
    return 1.0


def _section_well_formed(section: Any) -> bool:
    """True iff one section object has all fields present and correctly typed."""
    if not isinstance(section, dict):
        return False
    if not isinstance(section.get("id"), str) or not section["id"].strip():
        return False
    if not isinstance(section.get("title"), str) or not section["title"].strip():
        return False
    level = section.get("level")
    if not isinstance(level, int) or isinstance(level, bool) or not (1 <= level <= 5):
        return False
    if not isinstance(section.get("summary"), str) or not section["summary"].strip():
        return False
    citations = section.get("citations")
    if not isinstance(citations, list) or not citations:
        return False
    for cite in citations:
        if not isinstance(cite, dict):
            return False
        if not isinstance(cite.get("page_numbers"), list):
            return False
        if not isinstance(cite.get("source_text"), str) or not cite["source_text"].strip():
            return False
    return True


def score_schema_adherence(parsed: dict | None) -> float:
    """Fraction of sections that are well-formed (0.0 if no valid sections)."""
    if not isinstance(parsed, dict):
        return 0.0
    sections = parsed.get("sections")
    if not isinstance(sections, list) or not sections:
        return 0.0
    good = sum(1 for s in sections if _section_well_formed(s))
    return good / len(sections)


_JUDGE_PROMPT = """\
You are evaluating an AI-generated structured summary of a FINRA regulatory
rule. The source PDF is attached above. Judge the summary ONLY against the
source document.

Score each criterion from 1 (poor) to 5 (excellent):
- accuracy: are the stated requirements/definitions correct per the source?
- completeness: are the document's substantive sections and rules covered?
- faithfulness: is everything grounded in the source, with no fabrication?
- conciseness: are section summaries tight (1-3 sentences) and non-redundant?

GENERATED SUMMARY (JSON):
{summary}

Output JSON ONLY (no prose, no code fences):
{{"accuracy": <1-5>, "completeness": <1-5>, "faithfulness": <1-5>, "conciseness": <1-5>}}
"""


def _parse_scores(raw: str) -> float:
    """Parse the 4-criterion judge JSON and normalize the mean to [0,1]."""
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    crits = ("accuracy", "completeness", "faithfulness", "conciseness")
    vals = []
    for c in crits:
        v = obj.get(c)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"judge score {c!r} missing or non-numeric: {obj!r}")
        vals.append(max(1.0, min(5.0, float(v))))
    mean = sum(vals) / len(vals)
    return (mean - 1.0) / 4.0  # 1-5 -> 0-1


_JUDGE_RATE_LIMIT_RETRIES = 4
_JUDGE_BACKOFF_S = 10.0


def _is_rate_limited(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "REQUEST_LIMIT_EXCEEDED" in str(exc)


def build_summary_quality_judge(judge_client: Any, model: str):
    """Return ``judge(pdf_document_block, summary_text) -> float`` in [0,1].

    The judge is grounded on the source PDF — it receives the same
    Anthropic-style ``document`` block the summarizer saw, so faithfulness
    and completeness are judged against the actual document.
    """

    def judge(pdf_document_block: dict[str, Any], summary_text: str) -> float:
        prompt = _JUDGE_PROMPT.format(summary=summary_text[:20000])
        content = [pdf_document_block, {"type": "text", "text": prompt}]
        for attempt in range(_JUDGE_RATE_LIMIT_RETRIES + 1):
            try:
                response = judge_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=200,
                    temperature=0,
                )
                return _parse_scores(response.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001 - isolate judge failures per row
                # A rate-limited judge call scoring 0.0 costs ~0.025 aggregate on
                # a 20-row run (as much as the gate's noise band), so back off and
                # retry 429s instead of letting quota pressure decide the round.
                if _is_rate_limited(exc) and attempt < _JUDGE_RATE_LIMIT_RETRIES:
                    time.sleep(_JUDGE_BACKOFF_S * 2**attempt)
                    continue
                logger.warning("summary_quality judge failed: %s", exc)
                return 0.0
        return 0.0

    return judge
