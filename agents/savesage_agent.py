"""Savesage ICICI extraction agent (code-mode), round 1.

Single-variable change from the production baseline: GPT-5.6 Luna on the
native PDF, same prompt, same 96K max_tokens, but reasoning_effort="low"
instead of "medium". One call per statement (no extra sends, so the
per-document ledger and the cost gate are respected).
"""

from anvil.domains.savesage.agent_base import (
    DEFAULT_MAX_TOKENS,
    SavesageAgent,
)


class LowEffortSavesageAgent(SavesageAgent):
    """Production extraction with reduced reasoning effort for latency/cost."""

    REASONING_EFFORT = "low"

    def predict(self, *, sid: str, pdf_path: str) -> tuple[dict, dict]:
        return self.extract(
            sid=sid,
            pdf_path=pdf_path,
            reasoning_effort=self.REASONING_EFFORT,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
