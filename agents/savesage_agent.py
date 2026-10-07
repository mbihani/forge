"""Savesage ICICI extraction agent — zero-reasoning variant.

Round 3: parent is GPT-6 Luna + native PDF + ``reasoning_effort='low'``
(aggregate 0.9807, latency median 14931ms, cost $0.002441/row). Frontier
best-so-far per objective is {accuracy=0.983, latency=14931ms,
cost=$0.002441/row}; to clear the gate, at least one objective must
improve by more than its epsilon (accuracy 0.005 / latency 1000ms /
cost $0.0003) without any other regressing more than its epsilon.

This variant drops the GPT-6 Luna ``reasoning_effort`` from ``low`` to
``none``, keeping every other knob the same. The lever-probe accepts
``input_mode='pdf', reasoning_effort='none'`` for
``databricks-gpt-6-luna`` so this routes cleanly without a fallback.

Why this should combine latency AND cost wins without blowing the
accuracy floor:

* **Cost / latency.** Output tokens are price- and latency-dominant on
  Luna ($0.50/M out vs $0.10/M in; output generation is serial on the
  wire). Reasoning tokens are counted as output and still make up a
  non-trivial slice at ``low`` — the drop from medium to low already
  cut cost ~9% and latency ~2.9s between rounds 1 and 2. Removing the
  reasoning phase entirely eliminates the remaining reasoning-output
  slice: predicted Δ ~−$0.0003 to −$0.0007 per row (clears the
  $0.0003 cost epsilon) and ~−1500 to −2500ms median latency (clears
  the 1000ms latency epsilon).
* **Accuracy floor.** The remaining hard failures are structural, not
  reasoning-shaped. ``field_cards_cardMeta_network``=0.0 and
  ``field_rewards_programType``=0.0 (excluded) are enum/format
  mismatches that medium CoT could not fix either. ``cardCreditLimit``
  / ``cardAvailableCreditLimit`` were already 0.633 at
  ``reasoning_effort='medium'`` in round 1 — that is a GPT-6-Luna
  card-art parsing ceiling, not a reasoning headroom problem. Low →
  none should mostly affect long free-text fields
  (``field_transactions_description``, period-end dates) which already
  sit at 0.97–1.00 with slack within the 0.005 accuracy epsilon.

Single-step mutation over compound: clean attribution (either latency
or cost alone already clears a gate objective, and both are predicted
to move), and the parent (round 2 code) only changes one line.
"""

from anvil.domains.savesage.agent_base import (
    DEFAULT_MAX_TOKENS,
    SavesageAgent,
)


class ZeroReasoningSavesageAgent(SavesageAgent):
    """GPT-6 Luna + native PDF + ``reasoning_effort='none'`` for every statement."""

    def predict(self, *, sid: str, pdf_path: str) -> tuple[dict, dict]:
        return self.extract(
            sid=sid,
            pdf_path=pdf_path,
            reasoning_effort="none",
            max_tokens=DEFAULT_MAX_TOKENS,
        )
