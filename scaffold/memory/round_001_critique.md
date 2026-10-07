---
round: 1
branch: anvil/exp-round-1
decision: keep
action_kind: set_lever
parse_status: ok_last_of_many
baseline_score: 0.9830
mutated_score: 0.9830
score_delta: +0.0000
---

# Round 1 critique

## Action applied
set_lever model: None → 'databricks-gpt-6-luna': Round 1 opens with the baseline at 0.983 aggregate, $0.005371/row cost, 18635ms median latency. GPT-6 Luna is OpenAI's n

## Rationale (from optimizer)
Round 1 opens with the baseline at 0.983 aggregate, $0.005371/row cost, 18635ms median latency. GPT-6 Luna is OpenAI's newer-generation Luna variant (6 > 5.6) that is ~50% cheaper per input token ($0.10 vs $0.20 per 1M) and ~58% cheaper per output token ($0.50 vs $1.20 per 1M) per harness/model_catalog.csv. The lever-probe accepts `input_mode=pdf, reasoning_effort=medium` for GPT-6 Luna, so the baseline agent (which passes no explicit model and falls back to self.model) will route cleanly without any code change. Predicted Δ: cost drops ~50% (−$0.0028 ≫ cost epsilon $0.0003 = clear win); latency likely ≈ or slightly below current median (Luna family is the small/fast tier); accuracy expected within the 0.005 epsilon since a newer-gen model of the same family typically preserves or improves quality. Main risk is an accuracy regression on the already-struggling card-level fields (cardAvailableCreditLimit 0.776, network 0.000) — but those failures are structural/prompt-level, not solved by GPT-5.6 either, so the quality floor is intact. Preferred over a compound because the lever change alone should clear the gate on cost without pulling in an attribution-muddying second step.

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
+0.0000.
