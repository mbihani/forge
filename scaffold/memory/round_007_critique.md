---
round: 7
branch: anvil/exp-round-7
decision: keep
action_kind: write_agent
parse_status: ok_last_of_many
baseline_score: 0.9845
mutated_score: 0.9940
score_delta: +0.0095
---

# Round 7 critique

## Action applied
write_agent agents/savesage_agent.py: Targets latency: median must fall below about 12744ms (best so far 13744ms, round 6) with gated accuracy at or above abo

## Rationale (from optimizer)
Targets latency: median must fall below about 12744ms (best so far 13744ms, round 6) with gated accuracy at or above about 0.9795. Accuracy is near its ceiling (round 4 0.9864; round 6 lost only about one row of transactions and one card-name miss; network and limits fail the same way in native-PDF mode too), so latency is the objective with room. Round 3 showed the Luna text pass alone runs about 12.9s median, and rounds 5/6 showed the page-1 identity pass lands in about the same window. Both critical-path calls must get faster. Changes: (1) the main text-mode pass moves to GLM 5.3 Flash at low effort (not yet evaluated; a text-only flash model, expected to generate the transaction-heavy output faster). Card name, network and limits still come from the Luna page-1 identity pass, so GLM's lack of card art does not matter. (2) The output-free page-1 Luna identity call is sent twice at once and the first usable result wins, cutting Luna's wide latency tail (p90 about 2x median). (3) The identity wait is now max(main_done+6s, t0+13s), so a fast GLM pass does not cut identity results that land at round 6's timing. Fallbacks: if GLM raises or returns no summary or no transactions, the agent runs Luna text (low), then the native-PDF chain, so a GLM failure costs latency on that row, not accuracy. Predicted: median about 10-12s, gated accuracy about 0.980-0.985 (card fields as in round 6; transactions and summary depend on GLM's text quality). Risks: GLM's quality on ICICI text (backtick rupee sign) is unknown; if it is poor the gate reverts and we learn whether GLM is usable. If GLM rejects reasoning_effort, every row falls back and the round reverts on latency. Three concurrent calls per row could queue at the endpoint. Not import-checked locally because Write and Python execution were blocked by permissions this session. Helpers are round 6's validated code; new code uses only the stdlib concurrent.futures wait/FIRST_COMPLETED.

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
+0.0095.
