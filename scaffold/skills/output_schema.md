---
skill_id: output_schema
kind: domain
required: true
applies_to: runtime
priority: critical
created_at: 2026-10-02
---

# Output schema

Respond ONLY with valid JSON matching this EXACT schema (use these exact
field names):

```
{
  "document_title": "string",
  "sections": [
    {
      "id": "string (e.g. 'a', 'a.1', 'a.1.A')",
      "title": "string",
      "level": integer (1-5),
      "summary": "string (1-3 sentences)",
      "citations": [{"page_numbers": [1], "source_text": "quoted phrase"}]
    }
  ]
}
```

No markdown fences, no commentary.
