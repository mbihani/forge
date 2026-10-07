"""Baseline Savesage ICICI extraction agent (code-mode).

The ANVIL optimizer mutates this file to cut extraction latency while
holding field accuracy: the model (``self.allowed_models``), the input mode
(native PDF or pdftotext text), ``reasoning_effort`` and ``max_tokens`` —
statically or per statement. The baseline is production: the round's
model / input mode levers (GPT-5.6 Luna, native PDF) at medium reasoning
effort and 96K max tokens.
"""

from anvil.domains.savesage.agent_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT,
    SavesageAgent,
)


class BaselineSavesageAgent(SavesageAgent):
    """Baseline: production defaults for every statement."""

    def predict(self, *, sid: str, pdf_path: str) -> tuple[dict, dict]:
        return self.extract(
            sid=sid,
            pdf_path=pdf_path,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
