"""Savesage ICICI extraction agent (code-mode), round 1: low reasoning effort.

Same model / input mode as production (the round's levers: GPT-5.6 Luna,
native PDF) but the extraction runs at ``reasoning_effort='low'`` to cut
reasoning-token wall time. Statement extraction is mostly transcription
(dates, amounts, descriptions) that does not need medium reasoning.

Safety net: if the low-effort call fails (truncation / refusal / parse
error raises) or returns an obviously empty extraction, the statement is
re-run once at the production ``medium`` effort, so a rare failure costs
latency on that row instead of zeroing its accuracy.
"""

from anvil.domains.savesage.agent_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT,
    SavesageAgent,
)

FAST_EFFORT = "low"


def _looks_empty(parsed) -> bool:
    if not isinstance(parsed, dict) or not parsed:
        return True
    meta = parsed.get("statementMeta")
    summary = parsed.get("statementLevelSummary")
    return not meta and not summary


class LowEffortSavesageAgent(SavesageAgent):
    """Luna native PDF at low reasoning effort, medium-effort fallback."""

    def predict(self, *, sid: str, pdf_path: str) -> tuple[dict, dict]:
        first_meta = None
        try:
            parsed, first_meta = self.extract(
                sid=sid,
                pdf_path=pdf_path,
                reasoning_effort=FAST_EFFORT,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            if not _looks_empty(parsed):
                return parsed, first_meta
        except Exception:
            first_meta = None
        parsed, meta = self.extract(
            sid=sid,
            pdf_path=pdf_path,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        if first_meta is not None:
            meta = self.merge_meta(first_meta, meta)
        return parsed, meta
