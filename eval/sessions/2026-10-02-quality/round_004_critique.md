---
round: 4
branch: anvil/exp-round-4
decision: keep
action_kind: edit_skill
parse_status: ok_last_of_many
baseline_score: 0.9672
mutated_score: 0.9688
score_delta: +0.0016
---

# Round 4 critique

## Action applied
edit_skill skills/section_summary.md: Only non-perfect judge is summary_quality=0.934 (headroom 0.033) and 14/20 examples sit at 0.8125–0.9375 — i.e. 1-3 of t

## Rationale (from optimizer)
Only non-perfect judge is summary_quality=0.934 (headroom 0.033) and 14/20 examples sit at 0.8125–0.9375 — i.e. 1-3 of the four 1-5 sub-criteria (accuracy, completeness, faithfulness, conciseness) are dropping by one point. The one sub-criterion that NO active skill speaks to is conciseness: section_summary.md says 'exactly a 1-3 sentence summary' with no brevity bias, and section_summary + citations + output_schema all pull the writer toward using the full 3-sentence budget (verbatim legal terms, cross-references, defined-term preservation). A judge rewarding '1-3 sentences and non-redundant' will dock conciseness whenever a 2-sentence idea is padded to 3. This edit injects a single parenthetical ('prefer 1 or 2 sentences; use a third only when a threshold/exception/cross-reference would otherwise be omitted') + a one-line 'do not pad' close, leaving all three existing bullets untouched. Materially different from Round 1 (−0.106): Round 1 added coverage + no-paraphrase + normative-obligation and inflated output length to the point of 32k-truncation on two examples (both scored 0/0/0 on all judges) — a cascade that only triggers when guidance pushes output longer. This change strictly SHORTENS output, so it cannot reproduce that failure. Does not clash: output_schema.md allows 1-3 sentences (1-2 is a subset), identity.md says 'faithful short summary' (aligned), citations/hierarchy/exclusions are structural and orthogonal. Materially different from Round 3 (0.0): sampling-level, not content. Predicted Δ: +0.005 to +0.015 on summary_quality (two or three of the 0.9375 examples flipping conciseness from 4→5); downside is a true noop if the model already produces short summaries at T=0.0, or at worst a small completeness regression of ~0.005 — asymmetric upside given the brevity-cannot-truncate floor.

## Outcome
Decision: **KEEP**. Score delta vs cached baseline:
+0.0016.
