# Existing evals for this agent (experiment `3145915622642643`)

Read 200 traces (98 with assessments) on 2026-10-18T09:53:09+00:00.

## Judges forge re-uses to score every round

**No judges found.** Ask the user which metrics the evals should use and record them under `eval.agent_evals.user_metrics` before optimizing.

## What the judges point towards (lowest score first)

| Assessment | Source | n | Mean (1 = pass) |
| --- | --- | --- | --- |
| `first_action_under_450ms` | CODE | 98 | 0.00 |
| `cost_usd` | CODE | 98 | 0.00 |
| `reply_quality` | LLM_JUDGE | 52 | 0.08 |
| `action_correct` | CODE | 98 | 0.61 |
| `speak_and_act` | CODE | 46 | 0.65 |
| `transfer_gate_safety` | CODE | 98 | 0.88 |
| `first_action_ms` | CODE | 98 | 733.65 |
| `total_ms` | CODE | 98 | 934.04 |

**`reply_quality` failures:**
- on_script=❌, concise_for_voice=✅, no_hallucination=✅, no_private_leak=✅, pass=False. The caller stated a credit card balance under 7 thousand dollars, so the script requires the GATE 2 personal-loans check before any under-limit ending. The caller's 'already done this with my credit cards' is the 'already handled' objection, and its rebuttal has not been used yet. The reply ends the call with a ge
- on_script=❌, concise_for_voice=✅, no_hallucination=✅, no_private_leak=✅, pass=False. The caller answered the clarification with 'wrong number'. The script routes that to the Special Situations rebuttal ('We provide free advice on lowering credit card interest rates and balances...'). Only an explicit 'no debt' reaches the 'I am sorry to bother you, goodbye' ending, and the caller only said they ha
- on_script=❌, concise_for_voice=✅, no_hallucination=✅, no_private_leak=✅, pass=False. The caller clarified it was a wrong number. This is the first wrong-number objection of the call, so the script calls for the rebuttal: 'We provide free advice on lowering credit card interest rates and balances. If you owe money on credit cards, I'm happy to help.' Instead the agent ended the call with the no-deb
- on_script=❌, concise_for_voice=✅, no_hallucination=✅, no_private_leak=✅, pass=False. The caller is raising the 'already in a program' objection, and the script's rebuttal for it has not yet been properly delivered. Turn 5 asked a different, improvised question and never asked how long they've been working with the other company. Ending the call now skips the available rebuttal and abandons a still
- on_script=❌, concise_for_voice=✅, no_hallucination=✅, no_private_leak=✅, pass=False. The script says only an explicit 'no debt' reaches the no-debt ending. The caller said only that they have no credit cards and that the cards are their son's, which leaves personal loans or other debt unresolved. The agent's clarification ('Just to confirm, no…') was cut off, so the script calls for finishing that

**`action_correct` failures:**
- agent=end_call expected=speak (prod=end_call)
- agent=speak expected=end_call (prod=end_call)
- agent=transfer_call expected=speak (prod=transfer_call)
- agent=end_call expected=speak (prod=transfer_call)
- agent=speak expected=end_call (prod=speak)

**`speak_and_act` failures:**
- both; tool=yes (end_call), spoke=yes; spoken "I'm sorry, I don't understand. Thank you for your time. Goodbye." is not an acceptable scripted line ["I'll reach out later, goodbye", 'Thank you for your time, goodbye']
- spoke_only; tool=no (speak), spoke=yes; spoken "I'm sorry, I don't understand. I'm calling because the information we received shows you've been making payments on your credit cards on time, but you're still carrying over 7 thousand dollars in balances, is that right?" is not an acceptable scripted line ["I'll try again later, goodbye", "I'll reach out later, goodbye"]
- acted_only; tool=yes (transfer_call), spoke=no
- spoke_only; tool=no (speak), spoke=yes; spoken "I understand. I'm calling because our records show you've been making payments on your credit cards on time, but you're still carrying over 7 thousand dollars in balances, is that right?" is not an acceptable scripted line ['Sure, I will arrange a callback, goodbye']
- both; tool=yes (end_call), spoke=yes; spoken "I'm sorry, I don't understand. Thank you for your time. Goodbye.\n\n" is not an acceptable scripted line ["I'm unable to assist at this time. Goodbye"]

**`transfer_gate_safety` failures:**
- PREMATURE TRANSFER: gate not satisfied (debt_amount_stated, debt_in_range, unsecured_confirmed, decision_maker_confirmed)
- PREMATURE TRANSFER: gate not satisfied (unsecured_confirmed)
- PREMATURE TRANSFER: gate not satisfied (debt_in_range, unsecured_confirmed); total_debt '15,000' does not match caller's $4,000 (not within 15%)
- PREMATURE TRANSFER: gate not satisfied (debt_in_range, unsecured_confirmed); total_debt '15,000' does not match caller's $4,000 (>15% off)

## Judges with no signal — confirm with the user

- `first_action_under_450ms` fails on every trace (n=98). Check its definition against the agent's real output — it may be stale or miscalibrated rather than the agent being wrong. It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).

## Human feedback

- `acceptable_lines`: n=72, mean=—
- `call_state`: n=98, mean=—
- `category`: n=98, mean=—
- `confidence`: n=98, mean=—
- `expected_action`: n=98, mean=—
- `expected_line`: n=98, mean=—
- `gate`: n=98, mean=—
- `prod_action`: n=98, mean=—
- `stated_debt`: n=72, mean=—
- `transfer_allowed`: n=98, mean=0.18

## Trace input/output shape the judges were run on

- request: `['messages', 'sample_id']`
- response: `['action', 'cost_usd', 'error', 'first_action_ms', 'first_tool_ms', 'first_word_ms', 'text', 'tool_calls', 'total_ms']`

Pass the judges inputs/outputs in this shape so their verdicts stay comparable.
