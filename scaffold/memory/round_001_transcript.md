Looking at this round, I need to understand the setup carefully before proposing a mutation.

Good — GPT-6 Luna is the newer-generation OpenAI Luna model (newer than GPT-5.6 Luna) and is roughly 50% cheaper per token in both input and output. It's verified to accept native PDF input and reasoning_effort ∈ {none, low, medium, high, xhigh}.

Let me verify the baseline agent is a clean passthrough and that `set_lever` for `model` will route through properly:

The baseline agent doesn't pass `model` explicitly, so `extract()` falls back to `self.model` (the lever value). A `set_lever` change for `model` will route cleanly through.

## My analysis

**Baseline state:**
- aggregate 0.983, latency median 18635ms, cost $0.005371/row
- Gated accuracy excludes `rewards_programType` and `cards_cardMeta_productFamily` (stale GT — must not tune for them)
- The gating field failures (`cardAvailableCreditLimit` 0.776, `cardCreditLimit` 0.776, `network` 0.000, etc.) are mostly layout/parsing issues that aren't straightforwardly fixable by swapping a knob in a code-mode agent — the prompt skills are already in place and I'm not allowed to touch src/prompts.

**Lever opportunity: `model=databricks-gpt-6-luna`.**
From the model catalog (`harness/model_catalog.csv`):
- GPT-5.6 Luna (current): $0.20 in / $1.20 out per 1M tokens
- GPT-6 Luna: $0.10 in / $0.50 out per 1M tokens — newer OpenAI Luna generation (6 > 5.6), ~50% cheaper on input, ~58% cheaper on output

**Projected impact:**
- **Cost**: $0.005371 → ~$0.0025/row (Δ ≈ -$0.0028 ≫ epsilon $0.0003). Clear win.
- **Latency**: Unknown but Luna is OpenAI's small/fast tier; GPT-6 Luna is a newer gen and the probe accepted PDF+medium cleanly. Likely within the 1000ms epsilon.
- **Accuracy**: Primary risk. Baseline 0.983 → floor 0.978 (0.005 epsilon). As a newer-generation model (not a stripped-down variant), odds favor parity or improvement.

This is a clean single-lever change (no code rewrite required), targets the cost objective most directly, and provides data for future rounds on whether GPT-6 Luna preserves quality on this task. I prefer this over a `compound` because adding a second step (e.g., bumping `reasoning_effort=high` to compensate) would both increase latency/cost and muddle attribution if the lever change alone already clears the gate.

```json-action
{
  "action": "set_lever",
  "name": "model",
  "value": "databricks-gpt-6-luna",
  "rationale": "Round 1 opens with the baseline at 0.983 aggregate, $0.005371/row cost, 18635ms median latency. GPT-6 Luna is OpenAI's newer-generation Luna variant (6 > 5.6) that is ~50% cheaper per input token ($0.10 vs $0.20 per 1M) and ~58% cheaper per output token ($0.50 vs $1.20 per 1M) per harness/model_catalog.csv. The lever-probe accepts `input_mode=pdf, reasoning_effort=medium` for GPT-6 Luna, so the baseline agent (which passes no explicit model and falls back to self.model) will route cleanly without any code change. Predicted Δ: cost drops ~50% (−$0.0028 ≫ cost epsilon $0.0003 = clear win); latency likely ≈ or slightly below current median (Luna family is the small/fast tier); accuracy expected within the 0.005 epsilon since a newer-gen model of the same family typically preserves or improves quality. Main risk is an accuracy regression on the already-struggling card-level fields (cardAvailableCreditLimit 0.776, network 0.000) — but those failures are structural/prompt-level, not solved by GPT-5.6 either, so the quality floor is intact. Preferred over a compound because the lever change alone should clear the gate on cost without pulling in an attribution-muddying second step."
}
```

The baseline agent doesn't pass `model` explicitly, so `extract()` falls back to `self.model` (the lever value). A `set_lever` change for `model` will route cleanly through.

## My analysis

**Baseline state:**
- aggregate 0.983, latency median 18635ms, cost $0.005371/row
- Gated accuracy excludes `rewards_programType` and `cards_cardMeta_productFamily` (stale GT — must not tune for them)
- The gating field failures (`cardAvailableCreditLimit` 0.776, `cardCreditLimit` 0.776, `network` 0.000, etc.) are mostly layout/parsing issues that aren't straightforwardly fixable by swapping a knob in a code-mode agent — the prompt skills are already in place and I'm not allowed to touch src/prompts.

**Lever opportunity: `model=databricks-gpt-6-luna`.**
From the model catalog (`harness/model_catalog.csv`):
- GPT-5.6 Luna (current): $0.20 in / $1.20 out per 1M tokens
- GPT-6 Luna: $0.10 in / $0.50 out per 1M tokens — newer OpenAI Luna generation (6 > 5.6), ~50% cheaper on input, ~58% cheaper on output

**Projected impact:**
- **Cost**: $0.005371 → ~$0.0025/row (Δ ≈ -$0.0028 ≫ epsilon $0.0003). Clear win.
- **Latency**: Unknown but Luna is OpenAI's small/fast tier; GPT-6 Luna is a newer gen and the probe accepted PDF+medium cleanly. Likely within the 1000ms epsilon.
- **Accuracy**: Primary risk. Baseline 0.983 → floor 0.978 (0.005 epsilon). As a newer-generation model (not a stripped-down variant), odds favor parity or improvement.

This is a clean single-lever change (no code rewrite required), targets the cost objective most directly, and provides data for future rounds on whether GPT-6 Luna preserves quality on this task. I prefer this over a `compound` because adding a second step (e.g., bumping `reasoning_effort=high` to compensate) would both increase latency/cost and muddle attribution if the lever change alone already clears the gate.

```json-action
{
  "action": "set_lever",
  "name": "model",
  "value": "databricks-gpt-6-luna",
  "rationale": "Round 1 opens with the baseline at 0.983 aggregate, $0.005371/row cost, 18635ms median latency. GPT-6 Luna is OpenAI's newer-generation Luna variant (6 > 5.6) that is ~50% cheaper per input token ($0.10 vs $0.20 per 1M) and ~58% cheaper per output token ($0.50 vs $1.20 per 1M) per harness/model_catalog.csv. The lever-probe accepts `input_mode=pdf, reasoning_effort=medium` for GPT-6 Luna, so the baseline agent (which passes no explicit model and falls back to self.model) will route cleanly without any code change. Predicted Δ: cost drops ~50% (−$0.0028 ≫ cost epsilon $0.0003 = clear win); latency likely ≈ or slightly below current median (Luna family is the small/fast tier); accuracy expected within the 0.005 epsilon since a newer-gen model of the same family typically preserves or improves quality. Main risk is an accuracy regression on the already-struggling card-level fields (cardAvailableCreditLimit 0.776, network 0.000) — but those failures are structural/prompt-level, not solved by GPT-5.6 either, so the quality floor is intact. Preferred over a compound because the lever change alone should clear the gate on cost without pulling in an attribution-muddying second step."
}
```