---
round: 7
branch: anvil/exp-interstell-q-round-7
decision: keep
action_kind: edit_skill
parse_status: ok_last_of_many
baseline_score: 0.6646
mutated_score: 0.8301
score_delta: +0.1655
---

# Round 7 critique

## Action applied
edit_skill skills/ending_the_call.md: Parent = round 5 (0.8347; speak_and_act 0.889). Two examples have failed on all 3 repeats in every round (1-6) and no ro

## Rationale (from optimizer)
Parent = round 5 (0.8347; speak_and_act 0.889). Two examples have failed on all 3 repeats in every round (1-6) and no round has targeted either. S35 (end): the caller says 'f*** off' three times. The agent reads it as a do-not-call request and says 'Sure, I will remove you from the list, goodbye'. The label accepts only rung 3 'I'm unable to assist at this time. Goodbye', which fits because the agent already said 'I'm having trouble understanding.' S09 (reply): the caller asks a crude off-topic question ('what color are your *******'). The agent uses the instruction-change line 'I'm sorry, I don't understand.', or says a bare 'I didn't catch that.' with no re-ask. Every judge rationale asks for rung 1 plus a re-ask of the current question. The edit adds one paragraph under the not-engaging ladder. Swearing, insults and crude off-topic remarks that say nothing about calling go on the not-engaging ladder. They are not a do-not-call request (that needs words about calling) and not an instruction change. Rungs 1-2 must always include the re-ask. Collateral check over all samples with profanity: S12 is labelled end_call and accepts both rung 3 and the do-not-call line, so it is safe either way. S18 (transfer) already uses rung 1 correctly. S06 is already disputed. 'Go away' is deliberately left out so it does not clash with the existing stop-bothering-me do-not-call trigger. Predicted: speak_and_act from 0.889 to about 0.97 if S35 flips 3/3, clearing the +0.03 bar on its own. reply_quality up about +0.06 if S09 flips 3/3. Quality up about +0.03 to +0.035. action_correct and transfer_gate_safety unchanged. About +85 words (<1% of the prompt), so first-action p50 should stay near 590ms (limit 700) and p90 near 720ms (limit 1000).

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
+0.1655.
