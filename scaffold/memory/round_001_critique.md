---
round: 1
branch: anvil/exp-round-1
decision: keep
action_kind: set_lever
parse_status: ok_last_of_many
baseline_score: 0.9500
mutated_score: 0.9250
score_delta: -0.0250
---

# Round 1 critique

## Action applied
set_lever model: None → 'databricks-claude-haiku-4-5': Baseline latency is median 44.7s / p90 191s on Sonnet 4-6 with quality already at 0.95 (summary_quality=0.90, safety=1.0

## Rationale (from optimizer)
Baseline latency is median 44.7s / p90 191s on Sonnet 4-6 with quality already at 0.95 (summary_quality=0.90, safety=1.00). The quality floor permits a 0.03 drop, and Haiku 4-5 — same Claude family, 3× cheaper — should preserve JSON schema fidelity and the summary-quality judge's fact-faithfulness criterion while cutting the summarize call latency by a wide margin (far above the 2000ms epsilon). This is the single highest-leverage mutation for the latency objective and gives a clean attribution for round 1 before any compound work. If quality holds, later rounds can probe the cheaper GPT models or trim the prompt; if it drops >0.03, we have a clean revert signal pointing to prompt specialization next.

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
-0.0250.
