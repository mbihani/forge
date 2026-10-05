---
kind: bank_rules
---
- issuerName is ALWAYS "ICICI Bank". These cards are co-branded (Amazon Pay, Coral, Sapphiro, Rubyx,
  Emeralde, Platinum, MakeMyTrip, Adani One, HPCL, Expressions, Mine, Manu). The co-brand partner and
  the product name are NOT the issuer. Never return "Amazon", "MakeMyTrip", "Adani One", "HPCL",
  "Coral", "Sapphiro" or any other product/partner name in issuerName.
- THE RUPEE SIGN ON THIS BANK IS A BACKTICK. ICICI encodes the rupee symbol as the single character
  ` (grave accent) in a dedicated rupee font, so the amount column header prints as "Amount (in `)"
  and money values may be preceded by a bare `. Treat ` in a money position as the rupee sign and
  report currency "INR"; never copy it into a description or read it as a digit.
  A capital "C" on this bank is NOT a currency glyph — it is a genuine CREDIT marker, so a "C" next
  to an amount means direction = "CREDIT".
- Amounts use Indian digit grouping (e.g. 1,23,456.78). Read the whole grouped number; never truncate
  at a comma.
- IGNORE the "Closing Balance" line inside the illustrative Minimum-Amount-Due worked example
  (the numbered "SL. No / Transaction" table whose money column is headed by a bare ` and which
  begins "On statement dated <Month> <DD>, <YYYY>, following Minimum Amount Due is calculated").
  That whole table is PRE-PRINTED BOILERPLATE with fixed specimen values (e.g. Closing Balance
  26,958.20, Tax on Over-limit Fee 99.00, Overdue of previous statement 1,200.00) and hardcoded
  specimen dates; it belongs to no real cardholder. Never use any figure from it for closingPoints,
  for any statementLevelSummary field, or as a transaction.
  The "Closing Balance" INSIDE that Minimum-Amount-Due worked example is a MONEY specimen: never use
  it for closingPoints. This does NOT apply to a "Closing Balance" printed INSIDE a rewards strip
  (see ICICI_REWARDS_LAYOUTS Layout 5, the Mine Cash strip), which is a genuine rewards balance and
  DOES populate closingPoints. Distinguish the two by location: the money specimen sits in the
  numbered SL.No worked-example table; the rewards balance sits under a rewards heading beside
  "Earned" / "Redeemed" cells.
- network — EVIDENCE-FIRST, DEFAULT NULL. ICICI statements do NOT state the card's own network.
  Return a network ONLY when the statement visibly prints THIS card's own network as its own label or
  logo caption in or beside that card's own section. If you cannot point to such a printed label for
  that specific card, network MUST be null. The only place a network name appears as text in this
  corpus is the fuel-surcharge disclaimer "For RuPay/American Express/ Visa/Mastercard Credit Cards:
  Fuel surcharge ...", which lists ALL FOUR networks and identifies nothing — it is never evidence
  for any card. Never infer network from the card BIN/first digits (a leading "4" is not evidence of
  Visa), from the product or co-brand name, from marketing or cross-sell copy, from merchant names in
  the transaction table, from rewards/offer wording, from that disclaimer, or from the network of
  another card on the same statement. When in doubt, null.
  Two further reasons a BIN can never settle this on ICICI: the leading four printed characters are
  themselves frequently MASKED (a heading may read "0000XXXXXXXX6043", so there is no BIN to read at
  all), and two cards on the SAME statement routinely carry different leading digits (e.g. one
  heading "3747XXXXXXXX4004" beside another "5241XXXXXXXX8004"), so no single network describes the
  statement.
  A network name that appears only INSIDE A PICTURE — a card photograph or marketing banner, such as
  an Aadhaar-update or iMobile-Pay promo showing a specimen card with a network logo on it — is
  artwork, not this cardholder's card. It is never evidence. network stays null.
- FOREIGN-CURRENCY ROWS: the transaction table has a separate "Intl.# amount" column printed to the LEFT
  of the "Amount (in `)" column, e.g. "CLAUDE.AI SUBSCRIPTION ANTHROPIC.COM US* | 0 | 118 USD | 11,632.05".
  Here 118 USD is the original foreign amount and 11,632.05 is the rupee amount billed. Report the RUPEE
  amount with currency "INR". Do NOT pair the rupee amount with the foreign currency code, and do NOT
  report the foreign amount. A row is only non-INR if the amount you report is itself in that currency.
- The "Reward Points" column sits between the description and the amount and may be negative (e.g. -45 on
  a reversal). It is rewardPointsOnThisTransaction. NEVER read it as the transaction amount, and never let
  it shift which amount belongs to which row: each row's amount is the value in the "Amount (in `)" column
  on that row's own line. Re-check every row whose amount equals a neighbouring row's amount.
- Transactions may be grouped under a masked card number heading (e.g. "5431XXXXXXXX5000") with more rows
  for a second card further down, and the same merchant can legitimately repeat with identical amounts on
  the same date. Extract EVERY row of EVERY table on EVERY page, including exact duplicates; do not
  de-duplicate and do not skip a repeated row.
- lastFourDigit — ON THIS BANK THE MASK IS IN THE MIDDLE. ICICI prints the card number as
  NNNNXXXXXXXXNNNN (e.g. "4315XXXXXXXX5002", "0000XXXXXXXX3225"): the masked run sits in the MIDDLE
  and the four characters at the RIGHT END are real digits. lastFourDigit = those four rightmost
  printed characters of that card's own masked card-number heading — "4315XXXXXXXX5002" → "5002",
  "0000XXXXXXXX3225" → "3225". NEVER return an "X" when the rightmost four printed characters are
  digits (not "XX02", not "X001", not "XX21"); return "X" only for a position that is itself masked
  within those rightmost four. Do not slice the window from the middle of the number, and do not read
  the leading 4-digit BIN fragment as the last four.
  The same rule applies when the mask is printed in SPACE-SEPARATED groups — "4375 XXXX XXXX 4008" →
  "4008", "0000 XXXX XXXX 0599" → "0599" — and when the number sits in a labelled cell such as
  "Card Number : 4375 XXXX XXXX 4008 - <CARDHOLDER NAME>" or a "Card Account No" box. Ignore the
  trailing " - <NAME>" part; it is the cardholder, not part of the number.
  Bind the value to the SAME card section the heading belongs to: never take it from another card's
  heading, from the credit-card account number, from the "Invoice No", or from a transaction
  reference number.
  NEVER take a card number from a PICTURE. The marketing banners on page 1 display specimen cards
  bearing numbers such as "4378 XXXX XXXX XXXX" and "4375 5174 1234 5678"; those are stock artwork
  and belong to nobody. Use only card numbers printed as text in a card heading or a card-number cell.
- TRAILING COUNTRY CODE IS PART OF THE DESCRIPTION. ICICI prints a terminal country-code token
  (IN, US, ...) as the last token of many transaction narrations, e.g. "UPI-570397032082-Babasahe b IN",
  "GOOGLE *Discovery Plus g.co/helppay# US", "UPI-...-PARASHRA IN". It is often laid out flush to the
  right edge of the description cell and can look like a separate column — it is NOT a separate column,
  it belongs to the narration. Keep it, with its separating space, exactly as printed: a description
  printed "... PARASHRA IN" must end in " IN".
  GUARD: never supply a country code that is not visibly printed for that row. Do not infer one from
  the merchant's identity, from the billing currency, or from a neighbouring row that has one.
- NARRATION IS TRANSCRIBED, NOT INTERPRETED. Copy each description character-for-character from that
  row as printed:
    - preserve the printed capitalisation exactly ("McDonald's", "Myntra", "IGST-CI@18%") — do not
      title-case, upper-case, lower-case or otherwise re-case it;
    - copy every reference/UPI number digit-for-digit from that row's own text; never re-type it from
      memory or from a neighbouring row;
    - do not spell-correct, clean up, expand, abbreviate or substitute a merchant name (a row printed
      "Myntra BANGALORE IN" is not "MYNTRA DESIGNS PRIVATE L Bangalore IN"; "SHAMBHU" is not "SHAMBU");
    - ICICI truncates long narrations mid-word at a fixed cell width ("...UPI-133114539429-Amazo",
      "...REALME MOBILE TELECOMMU"): stop exactly where the print stops — do not complete the word, do
      not truncate it further, and do not add or drop a trailing character.
- isPrimaryCard: ICICI does not label cards "primary" or "add-on". When several cards appear, the card
  whose transactions are listed first under the Statement Summary is the primary card (true) and the
  others are false. With exactly one card, isPrimaryCard = true.
- "INFINITY PAYMENT RECEIVED", "BBPS Payment received" and similar are payments TO the bank: they print
  with a "CR" marker, so direction = "CREDIT" and txnType = "PAYMENT".
- Amortization rows ("Principal Amount Amortization - <n/m>MERCHANT", "Interest Amount Amortization -
  <n/m>MERCHANT") are real EMI transaction rows: include them. Keep the "<n/m>" fragment in the
  description exactly as printed.
- If both points and cashback are present in the statement, select points.
- Ensure the lastFourDigit field strictly corresponds to the card associated with the selected points;
  do not aggregate points of all cards if there are multiple cards.
- Transaction Date must not exceed the Statement Date, nor fall more than two months prior to it.
- If statement is not a credit card statement, set all fields to null.
- For all date fields (statementDate, dueDate, and every transaction date):
  - always format the value as DD/MM/YYYY, regardless of how the date appears in the statement.
  - Exception: if dueDate is non-date text (e.g., "PAY IMMEDIATELY"), preserve it as-is per the existing rule.
  - Sanity check before output: a transaction date must not exceed statementDate, and must not fall more
    than two months before it. If a parsed date fails this check, swap day/month and re-validate before
    finalizing. Use the statement period printed on the page only as a reading aid for this check; it is
    not itself an output field.
- If the transaction amount carries a "+", "Cr", "C" or "CREDIT" marker, set direction to "CREDIT".

ICICI_CARD_IDENTITY (cardDisplayName / productFamily):
