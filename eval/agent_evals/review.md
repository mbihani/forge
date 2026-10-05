# Existing evals for this agent (experiment `967014443183055`)

Read 300 traces (186 with assessments) on 2026-10-05T11:51:41+00:00.

## Judges forge re-uses to score every round

**No judges found.** Ask the user which metrics the evals should use and record them under `eval.agent_evals.user_metrics` before optimizing.

## What the judges point towards (lowest score first)

| Assessment | Source | n | Mean (1 = pass) |
| --- | --- | --- | --- |
| `judge_programType` | LLM_JUDGE | 152 | 0.11 |
| `judge_bonusPointsThisCycle` | LLM_JUDGE | 152 | 0.67 |
| `judge_closingPoints` | LLM_JUDGE | 277 | 0.70 |
| `judge_cardNetwork` | LLM_JUDGE | 152 | 0.75 |
| `judge_cardProductFamily` | LLM_JUDGE | 152 | 0.85 |
| `judge_statementPeriodEnd` | LLM_JUDGE | 152 | 0.96 |
| `judge_statementPeriodStart` | LLM_JUDGE | 152 | 0.96 |
| `judge_cardDisplayName` | LLM_JUDGE | 277 | 0.96 |
| `judge_issuerName` | LLM_JUDGE | 152 | 0.97 |
| `judge_stmt_totalAmountDue` | LLM_JUDGE | 152 | 0.98 |
| `judge_overall_strict` | LLM_JUDGE | 277 | 0.98 |
| `judge_overall_forgiven` | LLM_JUDGE | 276 | 0.98 |
| `judge_transactions_description` | LLM_JUDGE | 276 | 0.99 |
| `judge_pointsEarnedThisCycle` | LLM_JUDGE | 277 | 0.99 |
| `judge_lastFourDigit` | LLM_JUDGE | 277 | 1.00 |
| `judge_transactions_direction` | LLM_JUDGE | 152 | 1.00 |
| `judge_transactions_date` | LLM_JUDGE | 277 | 1.00 |
| `judge_transactions_rewardPointsOnThisTransaction` | LLM_JUDGE | 151 | 1.00 |
| `judge_cardAvailableCreditLimit` | LLM_JUDGE | 152 | 1.00 |
| `judge_cardCreditLimit` | LLM_JUDGE | 152 | 1.00 |
| `judge_dueDate` | LLM_JUDGE | 152 | 1.00 |
| `judge_openingPoints` | LLM_JUDGE | 152 | 1.00 |
| `judge_pointsExpiringNext30Days` | LLM_JUDGE | 152 | 1.00 |
| `judge_pointsExpiringNext60Days` | LLM_JUDGE | 152 | 1.00 |
| `judge_pointsRedeemedThisCycle` | LLM_JUDGE | 152 | 1.00 |
| `judge_statementDate` | LLM_JUDGE | 152 | 1.00 |
| `judge_stmt_availableCreditLimit` | LLM_JUDGE | 152 | 1.00 |
| `judge_stmt_totalCreditLimit` | LLM_JUDGE | 152 | 1.00 |
| `judge_stmt_totalMinimumAmountDue` | LLM_JUDGE | 152 | 1.00 |
| `judge_transactions_amount` | LLM_JUDGE | 276 | 1.00 |

## Judges with no signal — confirm with the user

- `judge_cardAvailableCreditLimit` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_cardCreditLimit` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_dueDate` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_openingPoints` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_pointsExpiringNext30Days` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_pointsExpiringNext60Days` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_pointsRedeemedThisCycle` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_statementDate` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_stmt_availableCreditLimit` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_stmt_totalCreditLimit` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_stmt_totalMinimumAmountDue` passes on every trace (n=152). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `judge_transactions_amount` passes on every trace (n=276). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).

## Human feedback

- `field_feedback`: n=10, mean=1.00
  - ACCEPT: transactions.2.date
  - ACCEPT: transactions.2.description
  - ACCEPT: rewards.closingPoints
  - ACCEPT: cards.0.cardMeta.lastFourDigit
  - ACCEPT: cards.1.cardMeta.cardDisplayName
- `human_feedback_statementMeta_issuerName`: n=1, mean=1.00
- `human_feedback_statementMeta_rawStatementId`: n=1, mean=0.00

## Issues MLflow detected

- **Incomplete LLM judge evaluation coverage for KOTAK and batch ICICI traces** (severity medium, n=136): 44 of 100 traces (44%) have no LLM_JUDGE assessment scores, leaving their extraction quality unevaluated. All 3 KOTAK bank traces receive only CODE-type assessments with N/A values, suggesting the evaluation pipeline lacks expected-output ground truth for KOTAK statements. Additionally, 41 ICICI traces from the Aug 21 04:47-04:55 batch window were never scored by any evaluation run, while other IC
- **Systematic programType extraction failure across ICICI and one HDFC statement** (severity medium, n=76): The extract span produces completely wrong programType (judge_programType=0) across 37 ICICI and 1 HDFC trace. Same root cause as previously reported but now at 38 vs original 3 traces, and extends to HDFC.
- **Incorrect issuerName and programType extraction for ICICI and GENERIC banks** (severity medium, n=50): The extract span produces wrong values for the issuerName and programType metadata fields on non-HDFC bank statements. LLM judges score programType at 0 (completely wrong) across ICICI traces, and issuerName at 0 on at least one ICICI and one GENERIC trace. The issue also affects one HDFC trace (tr-75e35c57) where programType is scored 0, extending beyond the originally reported scope.
- **Rewards points closing balance arithmetic violation in HDFC statements** (severity medium, n=28): The extract span produces reward point values where closingPoints does not equal openingPoints + earnedThisCycle + bonusThisCycle minus redeemedThisCycle. The validate span catches this with an explicit arithmetic-violation exception, but the trace still completes with OK status, meaning incorrect extraction data is persisted downstream.
- **Card and reward metadata extraction failures in ICICI and multi-card statements** (severity medium, n=26): The extract span produces incorrect values for card-level and reward-level metadata fields (closingPoints, cardDisplayName, cardProductFamily, pointsEarnedThisCycle, cardNetwork) in certain ICICI and HDFC layouts. These are distinct from the known issuerName/programType and transaction-field failures, clustering around multi-card statements.
- **Transaction description extraction degradation across banks** (severity medium, n=4): The extract span partially misidentifies transaction descriptions across multiple bank formats with LLM judge scores dropping to 0.50-0.86, correlated with statements containing many transactions or multi-line descriptions. Two additional traces (GENERIC tr-569a1dea and HDFC tr-c74646f0) also exhibit this but could not be linked here due to assessment limits.
- **Degraded transaction field extraction accuracy across banks** (severity medium, n=3): The extract span partially misidentifies transaction-level fields including descriptions, directions, and bonus points across GENERIC and HDFC formats. LLM judge scores drop to 0.57-0.89 on transaction description and direction fields while statement-level metadata scores remain at 1.0. Additional affected traces (tr-569a1dea GENERIC, tr-c74646f0 HDFC) could not be linked due to assessment limits.

## Trace input/output shape the judges were run on

- request: `['bank', 'filename', 'request_id']`
- response: `['extraction', 'outcome', 'schema_valid']`

Pass the judges inputs/outputs in this shape so their verdicts stay comparable.
