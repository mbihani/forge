Now I have a complete picture. Let me summarize before deciding:

**State analysis:**
- Baseline: aggregate=0.967; only summary_quality=0.934 is non-perfect; schema_adherence & output_json_valid are perfect
- Judge criteria (hidden sub-scores): accuracy, completeness, faithfulness, conciseness, each 1-5, mean normalized to [0,1]
- 0.9375 example = exactly one criterion dropped to 4; 0.8125 example = three criteria at 4
- **14 of 20 examples still fail summary_quality** at 0.8125–0.9375 in round 3

**Lessons from prior rounds:**
- **Round 1 (−0.106):** Edited section_summary.md with 3 bullets pushing coverage + verbatim + normative-obligation. Caused output inflation → 2 examples truncated to 0/0/0. **Pushing length UP is catastrophic.**
- **Round 2 (noop):** Punted citing no telemetry.
- **Round 3 (0.0):** Dropped temperature 0.1→0.0; endpoint already treated them identically.

**Un-tried direction:** The one criterion that is almost certainly leaking across many examples but no scaffold mentions it is **conciseness** — the judge rewards 1-3 sentence tight summaries, and the current `section_summary.md` says "exactly a 1-3 sentence summary" with no bias toward brevity. A model instructed "1-3 sentences" with a verbatim-preservation push tends to use the whole budget (3 sentences) by default. Round 1's failure was *expansion*; nudging the model to *prefer shorter* is a strictly length-reducing change — it cannot cause the Round-1-class truncation failure, and 1-2 sentences stays inside the output_schema's "1-3 sentences" bound (no clash).

I'll make a surgical, length-reducing edit to `section_summary.md`: add a single parenthetical that tilts the model toward 1-2 sentences while still permitting 3. Bullets stay unchanged (so no faithfulness/accuracy regression vector). This targets the conciseness sub-criterion without touching coverage.

```json-action
{
  "action": "edit_skill",
  "target_file": "skills/section_summary.md",
  "content": "---\nskill_id: section_summary\nkind: domain\nrequired: true\napplies_to: runtime\npriority: high\ncreated_at: 2026-10-02\n---\n\n# Per-section summary\n\nFor each section, write a 1-3 sentence summary (prefer 1 or 2 sentences;\nuse a third sentence only when strictly necessary to cover a numeric\nthreshold, exception, or cross-reference that would otherwise be\nomitted) that:\n\n- Captures the core requirement or definition.\n- Preserves key legal terms, rule cross-references (e.g. \"Rule 3110(b)\",\n  \"SEA Rule 17a-4(b)\"), dates, thresholds, and party names verbatim.\n- Uses precise language appropriate for compliance professionals.\n\nDo not pad. If the one-sentence form is faithful and complete, stop there.\n",
  "rationale": "Only non-perfect judge is summary_quality=0.934 (headroom 0.033) and 14/20 examples sit at 0.8125–0.9375 — i.e. 1-3 of the four 1-5 sub-criteria (accuracy, completeness, faithfulness, conciseness) are dropping by one point. The one sub-criterion that NO active skill speaks to is conciseness: section_summary.md says 'exactly a 1-3 sentence summary' with no brevity bias, and section_summary + citations + output_schema all pull the writer toward using the full 3-sentence budget (verbatim legal terms, cross-references, defined-term preservation). A judge rewarding '1-3 sentences and non-redundant' will dock conciseness whenever a 2-sentence idea is padded to 3. This edit injects a single parenthetical ('prefer 1 or 2 sentences; use a third only when a threshold/exception/cross-reference would otherwise be omitted') + a one-line 'do not pad' close, leaving all three existing bullets untouched. Materially different from Round 1 (−0.106): Round 1 added coverage + no-paraphrase + normative-obligation and inflated output length to the point of 32k-truncation on two examples (both scored 0/0/0 on all judges) — a cascade that only triggers when guidance pushes output longer. This change strictly SHORTENS output, so it cannot reproduce that failure. Does not clash: output_schema.md allows 1-3 sentences (1-2 is a subset), identity.md says 'faithful short summary' (aligned), citations/hierarchy/exclusions are structural and orthogonal. Materially different from Round 3 (0.0): sampling-level, not content. Predicted Δ: +0.005 to +0.015 on summary_quality (two or three of the 0.9375 examples flipping conciseness from 4→5); downside is a true noop if the model already produces short summaries at T=0.0, or at worst a small completeness regression of ~0.005 — asymmetric upside given the brevity-cannot-truncate floor."
}
```

Now I have a complete picture. Let me summarize before deciding:

**State analysis:**
- Baseline: aggregate=0.967; only summary_quality=0.934 is non-perfect; schema_adherence & output_json_valid are perfect
- Judge criteria (hidden sub-scores): accuracy, completeness, faithfulness, conciseness, each 1-5, mean normalized to [0,1]
- 0.9375 example = exactly one criterion dropped to 4; 0.8125 example = three criteria at 4
- **14 of 20 examples still fail summary_quality** at 0.8125–0.9375 in round 3

**Lessons from prior rounds:**
- **Round 1 (−0.106):** Edited section_summary.md with 3 bullets pushing coverage + verbatim + normative-obligation. Caused output inflation → 2 examples truncated to 0/0/0. **Pushing length UP is catastrophic.**
- **Round 2 (noop):** Punted citing no telemetry.
- **Round 3 (0.0):** Dropped temperature 0.1→0.0; endpoint already treated them identically.

**Un-tried direction:** The one criterion that is almost certainly leaking across many examples but no scaffold mentions it is **conciseness** — the judge rewards 1-3 sentence tight summaries, and the current `section_summary.md` says "exactly a 1-3 sentence summary" with no bias toward brevity. A model instructed "1-3 sentences" with a verbatim-preservation push tends to use the whole budget (3 sentences) by default. Round 1's failure was *expansion*; nudging the model to *prefer shorter* is a strictly length-reducing change — it cannot cause the Round-1-class truncation failure, and 1-2 sentences stays inside the output_schema's "1-3 sentences" bound (no clash).

I'll make a surgical, length-reducing edit to `section_summary.md`: add a single parenthetical that tilts the model toward 1-2 sentences while still permitting 3. Bullets stay unchanged (so no faithfulness/accuracy regression vector). This targets the conciseness sub-criterion without touching coverage.

```json-action
{
  "action": "edit_skill",
  "target_file": "skills/section_summary.md",
  "content": "---\nskill_id: section_summary\nkind: domain\nrequired: true\napplies_to: runtime\npriority: high\ncreated_at: 2026-10-02\n---\n\n# Per-section summary\n\nFor each section, write a 1-3 sentence summary (prefer 1 or 2 sentences;\nuse a third sentence only when strictly necessary to cover a numeric\nthreshold, exception, or cross-reference that would otherwise be\nomitted) that:\n\n- Captures the core requirement or definition.\n- Preserves key legal terms, rule cross-references (e.g. \"Rule 3110(b)\",\n  \"SEA Rule 17a-4(b)\"), dates, thresholds, and party names verbatim.\n- Uses precise language appropriate for compliance professionals.\n\nDo not pad. If the one-sentence form is faithful and complete, stop there.\n",
  "rationale": "Only non-perfect judge is summary_quality=0.934 (headroom 0.033) and 14/20 examples sit at 0.8125–0.9375 — i.e. 1-3 of the four 1-5 sub-criteria (accuracy, completeness, faithfulness, conciseness) are dropping by one point. The one sub-criterion that NO active skill speaks to is conciseness: section_summary.md says 'exactly a 1-3 sentence summary' with no brevity bias, and section_summary + citations + output_schema all pull the writer toward using the full 3-sentence budget (verbatim legal terms, cross-references, defined-term preservation). A judge rewarding '1-3 sentences and non-redundant' will dock conciseness whenever a 2-sentence idea is padded to 3. This edit injects a single parenthetical ('prefer 1 or 2 sentences; use a third only when a threshold/exception/cross-reference would otherwise be omitted') + a one-line 'do not pad' close, leaving all three existing bullets untouched. Materially different from Round 1 (−0.106): Round 1 added coverage + no-paraphrase + normative-obligation and inflated output length to the point of 32k-truncation on two examples (both scored 0/0/0 on all judges) — a cascade that only triggers when guidance pushes output longer. This change strictly SHORTENS output, so it cannot reproduce that failure. Does not clash: output_schema.md allows 1-3 sentences (1-2 is a subset), identity.md says 'faithful short summary' (aligned), citations/hierarchy/exclusions are structural and orthogonal. Materially different from Round 3 (0.0): sampling-level, not content. Predicted Δ: +0.005 to +0.015 on summary_quality (two or three of the 0.9375 examples flipping conciseness from 4→5); downside is a true noop if the model already produces short summaries at T=0.0, or at worst a small completeness regression of ~0.005 — asymmetric upside given the brevity-cannot-truncate floor."
}
```