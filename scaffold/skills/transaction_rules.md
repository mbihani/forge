---
kind: transaction_rules
---
TRANSACTION RULES:
- For any Amount except for transactions->"amount" in the JSON Schema, if it is a credit transaction then add - before that Amount.
- transactions->amount is ALWAYS a positive number. Never negate the amount field regardless of the transaction direction.
- Extract every transaction exactly as shown in the statement.
- Copy transaction descriptions EXACTLY as they appear in the statement. Do NOT shorten, summarize, paraphrase, or omit any characters from the description.
- Do NOT infer, fabricate, aggregate, or compute missing transaction fields.
- rewardPointsOnThisTransaction:
  - MUST NOT be used outside the transaction object.
  - Do NOT convert an explicit 0 to null, and do NOT convert a missing/unstated value to 0.
  - Only use null when the transaction row has no reward points column/value shown at all for that specific transaction.
  - For co-brand cards that display a CASHBACK EARNED or equivalent per-transaction column, populate rewardPointsOnThisTransaction from that column value for each transaction row. Do not leave it null when the column is present and shows a value.
- For direction
  - ONLY allowed values are "DEBIT" or "CREDIT". No other values are acceptable.
  - Classify using these exact markers:
      - CREDIT: CR, C, +, credit, cashback, refund, reversal
      - DEBIT: DR, D, -, debit, purchase, payment
  - If the marker is ambiguous or missing, infer from transaction context:
      - Payments TO the bank → DEBIT
      - Refunds, reversals, cashback credited → CREDIT
  - NEVER output raw statement markers (e.g., "DR", "CR", "+") as the direction value.
- For currency:
  - If the transaction row explicitly states a currency code (e.g., INR, USD, GBP) → use it.
  - If the transaction amount has a currency symbol (e.g., ₹ → INR, $ → USD) → map the symbol to its ISO 4217 currency code and use it.
  - If no currency is stated per transaction, inherit the default currency from the statement header or account summary section.
  - If currency cannot be determined by any of the above → set to null.

REWARDS_RULES (AUTHORITATIVE):
