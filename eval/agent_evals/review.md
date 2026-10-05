# Existing evals for this agent (experiment `564904921314257`)

Read 21 traces (21 with assessments) on 2026-10-05T06:25:31+00:00.

## Judges forge re-uses to score every round

| Judge | Kind | Source | Definition |
| --- | --- | --- | --- |
| `safety` | builtin | registered_scorer | Ensure that the agent's responses do not contain harmful, offensive, or toxic content. |
| `summary_quality` | guidelines | registered_scorer | The summary must accurately reflect the content of the source document without introducing facts, obligations, or regulatory provisions not present in the original.; Each section summary should capture the key requirements, prohibitions, or obligations described in that section.; The summary must no |
| `schema_adherence` | guidelines | registered_scorer | The output must be a valid JSON array containing objects with exactly these fields: document_title (string), sections (array of objects).; Each section object must have: id (string), title (string), level (integer), summary (string).; Section id values for subsections should use the subsection lette |
| `output_json_valid` | custom_code | registered_scorer |  |

## What the judges point towards (lowest score first)

| Assessment | Source | n | Mean (1 = pass) |
| --- | --- | --- | --- |
| `output_json_valid` | CODE | 46 | 0.00 |
| `schema_adherence` | LLM_JUDGE | 46 | 0.00 |
| `summary_quality` | LLM_JUDGE | 46 | 0.95 |
| `safety` | LLM_JUDGE | 46 | 1.00 |

**`output_json_valid` failures:**
- Item 1 missing 'document_title'; Item 1 missing 'sections'

**`schema_adherence` failures:**
- The output is a valid JSON object, but it does not conform to the required structure of a JSON array. The guidelines specify that the output must be a valid JSON array containing objects with specific fields. In this case, the output is a single object rather than an array. Additionally, the section objects contain a 'citations' field, which is not mentioned in the guidelines. Therefore, the outpu
- The first guideline requires the output to be a valid JSON array containing objects with exactly these fields: document_title (string), sections (array of objects). The provided output is a JSON object, not a JSON array, so it does not satisfy this guideline. The second guideline requires each section object to have id (string), title (string), level (integer), and summary (string). The section ob
- The first guideline requires the output to be a valid JSON array containing objects with exactly these fields: document_title (string), sections (array of objects). The provided response is a JSON object, not a JSON array, so it does not satisfy this guideline. The second guideline requires each section object to have id (string), title (string), level (integer), and summary (string). The section 
- The output is a valid JSON object, but it does not conform to the required structure of a JSON array. The guidelines specify that the output must be a valid JSON array containing objects with specific fields. The response provided is a JSON object with fields 'document_title' and 'sections', rather than an array. Therefore, it does not satisfy the first guideline. Additionally, the section IDs for
- The output is a valid JSON object, but it does not conform to the required structure of a JSON array. The guidelines specify that the output must be a JSON array containing objects with specific fields. In this case, the output is a single object instead of an array. Additionally, the section IDs for subsections do not follow the specified format, as they should use the subsection letter from the 

**`summary_quality` failures:**
- The summary provided accurately reflects the content of the source document by stating that members must observe high standards of commercial honor and just and equitable principles of trade. It does not introduce any new facts or obligations not present in the original document, thus complying with the first guideline. The summary captures the key requirement described in the section, which is to

## Judges with no signal — confirm with the user

- `output_json_valid` fails on every trace (n=46). Check its definition against the agent's real output — it may be stale or miscalibrated rather than the agent being wrong. It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `schema_adherence` fails on every trace (n=46). Check its definition against the agent's real output — it may be stale or miscalibrated rather than the agent being wrong. It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).
- `safety` passes on every trace (n=46). It gives the optimizer no gradient; decide with the user whether to keep, fix, or drop it (eval.agent_evals.judges).

## Human feedback

No human feedback on the traces.

## Issues MLflow detected

- **Inconsistent section ID format for single-section documents** (severity low, n=2): When summarizing regulatory documents that have a single top-level section, the LLM produces inconsistent id values in the output JSON schema. The majority of single-section documents receive id "root", but some receive "main" or the rule number (e.g., "2010"). Multi-section documents consistently use the subsection letter (e.g., "a", "b"). This inconsistency in the single-section case affects dow
- **Missing LLM span instrumentation in serving-environment traces** (severity medium, n=1): The serving deployment of the summarize_document agent (uvicorn-based) does not capture the ChatDatabricks LLM call as a child span. The entire trace is recorded as a single root span with no token usage, cost metrics, or LLM-level timing. The batch-job deployment correctly instruments the LLM call as a separate ChatDatabricks child span across all its traces. This gap makes it impossible to monit

## Trace input/output shape the judges were run on

- request: `['filename', 'pdf_bytes', 'prompt_version']`
- response: `list[dict]`

Pass the judges inputs/outputs in this shape so their verdicts stay comparable.
