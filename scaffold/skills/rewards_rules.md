---
kind: rewards_rules
---
- Extract rewards ONLY from statement-level rewards sections.
- NEVER infer, compute, aggregate, or roll up rewards from transactions.
- Reward Points Earned at transaction level have no relation to total reward Earned at statement Level.
- transaction.rewardPointsOnThisTransaction MUST NEVER be summed or used to populate rewards.* fields.
- Identify rewards.programType first.
- rewards.programType:
    - Understand the rewards section and classify the program type.
    - ONLY these values are allowed: "Cashback", "Reward Points", "Membership Rewards".
    - If cashback/wallet credit/balance credited → "Cashback"
    - DO NOT copy payment methods, wallet names, or the programme's brand name as programType.
      See ICICI_REWARDS_LAYOUTS for the specific ICICI brand names that must NOT be emitted here.
- closingPoints is a REWARDS BALANCE. Populate it only from a numeric POINTS balance printed inside a
  rewards/points block — a figure whose own column or row label is about points, not about money.
- Set closingPoints = null if no numeric POINTS balance is explicitly shown in a rewards block.
- RewardPoints can be negative.
- For cashback cards: cashback earned = pointsEarnedThisCycle; cashback credited or transferred = pointsRedeemedThisCycle.
  Cashback credited MUST NOT be treated as closingPoints.
  Cashback earned in transactions should not be included in total cashback earned at statement level.
  If the description contains "Cashback Credit", do not derive rewards from the cashback amount;
  set rewardPointsOnThisTransaction only to an explicitly stated reward value in the description, otherwise null.

MISSING_DATA_RULE:
