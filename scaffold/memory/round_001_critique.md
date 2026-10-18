---
round: 1
branch: anvil/exp-interstell-round-1
decision: keep
action_kind: set_lever
parse_status: ok_last_of_many
baseline_score: 0.5086
mutated_score: 0.5155
score_delta: +0.0069
---

# Round 1 critique

## Action applied
set_lever route: 'gateway' → 'serving': Primary objective is latency (first-action p50 682ms vs Interstell's <450ms target). Interstell measured the AI Gateway 

## Rationale (from optimizer)
Primary objective is latency (first-action p50 682ms vs Interstell's <450ms target). Interstell measured the AI Gateway adding 40-130ms per call; routing the same Gemini 3.1 Flash-Lite request directly to <host>/serving-endpoints should save most of that, which beats the 50ms epsilon at the middle of the range and above. Model, prompt, temperature (0.3) and token cap are unchanged, so action_correct/speak_and_act/reply_quality (aggregate 0.509) and premature_transfers (12) should stay within sampling noise; this is the lowest-risk way to satisfy the epsilon-0 premature-transfer gate while improving an objective. Predicted: latency_ms_median ~580-640ms, quality ~0.51 ±0.02, premature_transfers ~12. Model swaps were deferred: they are the only lever for the quality failures but carry a high risk of extra premature transfers, which would revert the round. Caveat: the probe accepted route=serving only with temperature unset; if every row errors, check whether the serving route rejects temperature=0.3.

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
+0.0069.
