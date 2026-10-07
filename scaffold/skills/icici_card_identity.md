---
kind: card_identity
---
- The product name is printed ONCE, in the page-1 identity block at the TOP RIGHT, immediately beside
  the "ICICI Bank Credit Cards" logo — for example "Sapphiro", "Coral", "Rubyx", or the co-brand
  wordmark "amazon pay". On the older "CREDIT CARD E-STATEMENT" template there is no such block.
- USE THE BARE PRINTED FORM, exactly as it appears there: "Sapphiro", not "ICICI Bank Sapphiro Credit
  Card"; "Coral", not "ICICI Bank Coral Credit Card"; "Rubyx", not "ICICI Bank RubiX Credit Card".
  Do NOT expand it, do not prepend the issuer, and do not append "Credit Card".
- If that identity block shows only the "ICICI Bank Credit Cards" logo and NO product name, then this
  statement prints no product name: cardDisplayName = null and productFamily = null. Do not substitute
  the logo text "ICICI Bank Credit Card" as if it were a product.
- Do NOT take the product name from anywhere else on the page. In particular the offers and rewards
  footnotes print whole marketing sentences containing an expanded product name — e.g. "*My Cash
  earned on qualifying expenditure using MakeMyTrip ICICI Bank Credit Card will be added to your
  MakeMyTrip account ..." — and a product name inside such a sentence, or inside a fee table listing
  many products, is marketing copy, not this card's identity label.
- Do NOT read a product name off a card PHOTOGRAPH in a marketing banner. Those show stock artwork.
- All cards listed on one statement belong to the same product unless the identity block itself shows
  otherwise; give each card the same cardDisplayName.

ICICI_REWARDS_LAYOUTS (authoritative for rewards.*; bind every value to its own label by COLUMN):
