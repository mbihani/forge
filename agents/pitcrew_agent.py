"""Active code-mode pitcrew agent — the optimizer rewrites this module.

Baseline: production's ``direct_pdf`` path — one call on the round's model
with the composed scaffold prompt and production's ``max_tokens`` (32000).
"""

from __future__ import annotations

from typing import Any

from anvil.domains.pitcrew.agent_base import PitcrewAgent


class BaselinePitcrewAgent(PitcrewAgent):
    def predict(self, *, pdf_bytes: bytes) -> tuple[str, dict[str, Any]]:
        return self.summarize(pdf_bytes)
