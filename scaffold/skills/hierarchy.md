---
skill_id: hierarchy
kind: domain
required: true
applies_to: runtime
priority: high
created_at: 2026-10-02
---

# Preserve the full section hierarchy

Preserve the full section hierarchy exactly as it appears in the
document:

- Level 1: The document/rule title (e.g. "2210. Communications with the
  Public").
- Level 2: Major sections identified by lowercase letters (e.g.
  "(a) Definitions", "(b) Approval, Review and Recordkeeping").
- Level 3: Numbered subsections (e.g. "(1) Retail Communications",
  "(2) Correspondence").
- Level 4: Lettered sub-subsections (e.g. "(A) Principal Approval",
  "(B) Supervisory Analyst Alternative").
- Level 5: Further nesting if present (e.g. "(i)", "(ii)").

# Section IDs

Section IDs use dot notation reflecting hierarchy: e.g. "a", "a.1",
"a.1.A", "b.1.D.iii".
