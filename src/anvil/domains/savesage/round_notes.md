## Engine notes: savesage (`write_agent` contract)

Rewrite the active module `agents/savesage_agent.py` in place (only that
file is evaluated). It must define exactly ONE concrete subclass of
`anvil.domains.savesage.agent_base.SavesageAgent` implementing
`predict(self, *, sid, pdf_path) -> (extraction_dict, meta)`. Call
`self.extract(sid=sid, pdf_path=pdf_path, model=..., input_mode=...,
reasoning_effort=..., max_tokens=...)` for each extraction and return its
`meta` (combine several with `self.merge_meta(...)`).

- **Knobs:** `model` (any of `self.allowed_models`; `self.model` is the
  round's model), `input_mode` (`"pdf"` = native PDF; `"text"` =
  `pdftotext -layout` text), `reasoning_effort` and `max_tokens` (a
  truncated completion fails the whole statement). Use only combinations the
  "Verified runtime settings" section accepts; `extract()` refuses a
  rejected one before calling the endpoint.
- **Routing:** you may decide per statement on cheap signals — PDF byte
  size, page count, the co-brand token in the filename, or a local PDF
  step (`pdftotext`, `pdfseparate`, `pdfunite` are on PATH). The eval
  times the whole `predict` call, so extra calls and local steps cost
  latency.
- **Threads:** `predict` runs on several threads at once; keep
  per-statement state local.
- **Scoring:** SaveSage's production field judge vs the cached ground truth;
  fields in `eval.accuracy_exclude_fields` are not scored — do not tune for
  them.
