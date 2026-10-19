---
round: 3
branch: anvil/exp-interstell-sa-round-3
decision: keep
action_kind: edit_skill
parse_status: ok_last_of_many
baseline_score: 0.5236
mutated_score: 0.6708
score_delta: +0.1472
---

# Round 3 critique

## Action applied
edit_skill skills/ending_the_call.md: After round 2, two end turns still fail on all 3 repeats and have undisputed labels. S01: the caller line is TV/radio br

## Rationale (from optimizer)
After round 2, two end turns still fail on all 3 repeats and have undisputed labels. S01: the caller line is TV/radio broadcast. The agent said 'I'm having trouble understanding.' at line 18 and 'I didn't catch that.' at line 53, then keys off its latest line and repeats rung 2 instead of closing with 'I'm unable to assist at this time. Goodbye'. The edit tells it to scan ALL earlier lines for 'I'm having trouble understanding.' (even if it came before 'I didn't catch that.'), say rung 3 and call end_call in the same response. It also adds TV/radio/broadcast audio to the not-engaging triggers. S14: after complaining about being called at night, the caller says 'call me due the date' (meaning during the day). The agent reads it as gibberish. The edit says objecting to the call time and naming a better time, even in misheard words, is a callback request: say 'Sure, I will arrange a callback, goodbye' plus end_call, with no confirmation question. Collateral check against samples.jsonl: every sample with 'trouble understanding' in its history (S01/S06/S11/S12/S35) is labelled end_call and accepts the not-engaging closing, and only S14 contains time-of-call phrases, so no expected-speak turn should flip to an early end. Predicted: speak_and_act +0.03 to +0.06. S01 alone is 3 of 108 rows, about +0.028, so at least one S14 repeat must also flip to clear the 0.03 bar. action_correct should rise as well. About +110 prompt tokens (~1%), so first-action p50 should stay near 630ms, under the 700ms ceiling. Unlike round 1's broad closing table, this leaves the instruction-change and transfer lines untouched.

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
+0.1472.
