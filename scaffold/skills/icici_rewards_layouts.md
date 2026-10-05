---
kind: rewards_layouts
---
ICICI prints its rewards block low on page 1. Four layouts occur. In every one of them the LABELS sit
  on one or two lines and the VALUES sit BELOW them on a single shared line, separated only by their x
  position — so you must bind each value to the label that sits in the SAME COLUMN, not to the nearest
  text in reading order.

  Layout 1 — heading "ICICI Bank Rewards" (the letterforms may render as "ICICl Bank Rewards"):
      Total Points earned*        <value>
      Points earned on iShop      <value>
    pointsEarnedThisCycle = the "Total Points earned" value.
    The footnote states "*The total points earned are inclusive of points earned on iShop", so iShop is
    a SUBSET of the total: NEVER add the two together, and never report iShop as its own field.
    This layout prints NO redeemed, NO opening and NO closing figure →
    pointsRedeemedThisCycle = null, openingPoints = null, closingPoints = null.
    programType = "Reward Points".

  Layout 2 — heading "MakeMyTrip My Cash":
      My Cash earned      |  My Cash transferred to MakeMyTrip*
      <value>             |  <value>
    pointsEarnedThisCycle = the "My Cash earned" value (left column).
    pointsRedeemedThisCycle = the "My Cash transferred to MakeMyTrip" value (right column).
    programType = "Cashback".  "MakeMyTrip My Cash" and "My Cash" are the programme's BRAND name —
    never emit them as programType.

  Layout 3 — heading "EARNINGS" (Amazon Pay co-brand):
      Earned              |  Earnings transfered to Amazon Pay balance*
      <value>             |  <value>
    (the statement itself misspells "transfered"; match it loosely.)
    pointsEarnedThisCycle = the "Earned" value (left column).
    pointsRedeemedThisCycle = the "Earnings transfered to Amazon Pay balance" value (right column).
    programType = "Cashback".  "Amazon Pay balance" is a WALLET name — never emit it as programType.

  Layout 4 — older "CREDIT CARD E-STATEMENT" template, two labels side by side:
      Points Earned       |  Points Transferred to PAYBACK (Acc:<number>)
      <value>             |  <value>
    pointsEarnedThisCycle = the "Points Earned" value.
    pointsRedeemedThisCycle = the "Points Transferred to PAYBACK" value.
    programType = "Reward Points".  "PAYBACK" is the loyalty programme's BRAND name — never emit it as
    programType, and never emit the PAYBACK account number as a points figure.

  Layout 5 — heading "Mine Cash" (co-brand cash card), a FOUR-column strip. The labels sit on one
  line and the four values share the line below, bound by COLUMN:
      Opening Balance   |  Earned   |  Redeemed/Expired   |  Closing Balance
      <value>           |  <value>  |  <value>            |  <value>
    (the print may misspell "Opening Balace".)
    openingPoints           = the "Opening Balance" value.
    pointsEarnedThisCycle   = the "Earned" value.
    pointsRedeemedThisCycle = the "Redeemed/Expired" value (this IS a printed cell — report it even
                              when it is 0; do not null it).
    closingPoints           = the "Closing Balance" value (a GENUINE rewards balance printed inside
                              the Mine Cash strip — NOT the money specimen from the Minimum-Amount-Due
                              worked example).
    programType = "Cashback".  "Mine Cash" is the programme's BRAND name — never emit it as programType.

- EARNED AND TRANSFERRED MAY LEGITIMATELY BE EQUAL. When the statement prints two SEPARATE cells and
  both happen to show the same number (a cycle in which everything earned was transferred out), report
  BOTH — that is the printed truth, not a duplicate. Only refuse to fill the second field when the
  statement prints just ONE cell; never copy a single printed cell into two different fields.
- "Points Transferred", "transferred to", and "Earnings transfered to" all mean REDEEMED. They are
  pointsRedeemedThisCycle. They are never closingPoints.
- closingPoints on ICICI: Layouts 1–4 print no closing or available POINTS balance — "Total Points
  earned" and "Points earned on iShop" are CYCLE EARN figures, not a balance — so on those layouts
  closingPoints = null. The EXCEPTION is Layout 5 (Mine Cash), which DOES print a labelled "Closing
  Balance" cell inside its rewards strip: there, closingPoints = that value. In every case, do NOT
  compute closingPoints from earned minus redeemed, do NOT copy the earned figure into it, and do NOT
  take the money "Closing Balance" specimen from the Minimum-Amount-Due worked example.
- openingPoints on ICICI: null on Layouts 1–4 (no opening balance printed). The EXCEPTION is Layout 5
  (Mine Cash), which prints a labelled "Opening Balance" cell: there, openingPoints = that value.
  Never derive it, and never use the money "Previous Balance" from the Statement Summary.
- No ICICI layout in this corpus prints a points-expiry figure → pointsExpiringNext30Days and
  pointsExpiringNext60Days are null. A terms-and-conditions sentence describing when points expire is
  a policy, not a value.
