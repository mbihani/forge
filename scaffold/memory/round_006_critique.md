---
round: 6
branch: anvil/exp-round-6
decision: keep
action_kind: write_agent
parse_status: ok_last_of_many
baseline_score: 0.9845
mutated_score: 0.9812
score_delta: -0.0033
---

# Round 6 critique

## Action applied
write_agent agents/savesage_agent.py: Targets latency: the median must fall below about 15857ms (best so far 16857ms) with gated accuracy at or above about 0.

## Rationale (from optimizer)
Targets latency: the median must fall below about 15857ms (best so far 16857ms) with gated accuracy at or above about 0.9795. Two earlier rounds bracket the answer. Round 4 waited unconditionally for a page-1 PDF identity call that still listed transactions: accuracy 0.9864, median 16.0s. Round 5 used this exact structure (page-1 identity call that outputs no transactions) but gave it only 2.5s after the text pass: median 13.1s, accuracy 0.963. Its failures list shows cardDisplayName misses rising from 7 rows (round 4) to 17, with card limits falling 0.73 -> 0.63. All other fields were at 1.0, so the 2.5s cutoff dropped the identity result on about 10 rows that arrived just after the text pass. This round changes only the grace window, 2.5s -> 6s: the identity pass gets up to 6s after the text pass returns. Predicted: most of those 10 rows recover (cardDisplayName ~0.83-0.86, limits ~0.72), gated accuracy ~0.980-0.985, within epsilon of 0.9845. Median latency ~13.5-15s, because most rows already finished inside 2.5s and only the late tail waits longer, still under the 15857ms target. Risk: if round 5's misses were really parse or quality failures of the identity pass rather than timing, accuracy stays near 0.963 and the gate reverts, which itself shows the no-transactions suffix is the problem. Single-parameter change to an already-validated module, so the result is attributable. Not import-checked locally because Bash redirection was blocked this session. The code is round 5's validated module with only the constant and docstring changed.

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
-0.0033.
