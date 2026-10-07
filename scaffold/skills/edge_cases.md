---
kind: edge_cases
---
- If due date is non-date text (e.g., IMMEDIATELY, PAY IMMEDIATELY, DUE IMMEDIATELY),
  set dueDate to the exact text as shown — do not infer, normalize, or replace it with statementDate.
- For cardDisplayName: extract the credit card product name as printed in the page-1 identity block.
  See ICICI_CARD_IDENTITY for the exact form to use on this bank.
  Do NOT include the cardholder's name or account holder name.
- For lastFourDigit: extract ONLY the last 4 characters of the card number as a string.
  Preserve leading zeros (e.g., 0576 → "0576").
  Do NOT pad, expand, or perform numeric conversion.
  Only output actual digits for positions where the source PDF shows a real digit. Do not invent,
  infer, or backfill digits for masked positions; write "X" only for a position that is itself
  masked in the print.
  See ICICI_BANK_RULES for how this bank prints the mask — on this corpus the mask sits in the
  MIDDLE of the card number and the final four characters are real digits.
- For availableCreditLimit: if the specific phrase "available credit limit" (or synonymous label)
  and its value are not explicitly stated, set to null.
  DO NOT calculate this by subtracting totalAmountDue from totalCreditLimit.
ICICI_BANK_RULES (this corpus is ICICI Bank; these override anything above on conflict):
