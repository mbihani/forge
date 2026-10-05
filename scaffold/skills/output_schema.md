---
skill_id: output_schema
kind: domain
applies_to: runtime
priority: high
created_at: 2026-10-05
---

# Output format

Respond ONLY with valid JSON matching this EXACT schema (use these exact field names):
{
  "document_title": "string",
  "sections": [
    {
      "id": "string (e.g. 'a', 'a.1', 'a.1.A')",
      "title": "string",
      "level": integer (1-5),
      "summary": "string (2 sentences)",
      "citations": [{"page_numbers": [1], "source_text": "quoted phrase"}]
    }
  ]
}

No markdown fences, no commentary.
