"""Savesage ICICI extraction agent — low-reasoning variant.

Keeps the round's lever values (GPT-6 Luna + native PDF) but drops the
``reasoning_effort`` from the production default of ``medium`` to
``low``. ICICI statements are structured, highly-templated documents and
the prompt scaffold already encodes the field extraction rules; medium
chain-of-thought is largely spent on bookkeeping tokens that the output
schema already enforces. Dropping to ``low`` should materially cut both
the reasoning-output tokens (price-dominant on Luna: $0.50/M out vs
$0.10/M in) and the wall-clock latency (output tokens are serial), while
leaving the structured-extraction answers intact.

The lever probe accepts ``input_mode=pdf, reasoning_effort=low`` for
``databricks-gpt-6-luna`` so this routes cleanly without a fallback
branch.
"""

from anvil.domains.savesage.agent_base import (
    DEFAULT_MAX_TOKENS,
    SavesageAgent,
)


class LowReasoningSavesageAgent(SavesageAgent):
    """GPT-6 Luna + native PDF + ``reasoning_effort='low'`` for every statement."""

    def predict(self, *, sid: str, pdf_path: str) -> tuple[dict, dict]:
        return self.extract(
            sid=sid,
            pdf_path=pdf_path,
            reasoning_effort="low",
            max_tokens=DEFAULT_MAX_TOKENS,
        )
