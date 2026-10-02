"""Pitcrew domain: auto-optimize the FINRA document-summarizer prompt.

Binds FORGE to the pitcrew ``document-summarizer`` task:

* **scaffold** — the summarization system prompt decomposed into per-concern
  skills (identity / hierarchy / section_summary / citations / exclusions /
  output_schema) the optimizer edits one at a time.
* **golden set** — one FINRA rule PDF per row (``data/golden_set.jsonl``),
  pointing at the 20-doc corpus; no reference summary needed.
* **eval** — :func:`evaluate_pitcrew` composes the prompt, re-summarizes each
  PDF through the AI Gateway (direct_pdf document block), and scores with two
  deterministic structure checks plus a PDF-grounded quality judge.

Self-registers under ``"pitcrew"`` on import; the domain-agnostic core
resolves it by config name. Importing this package is cheap — it only
registers a callable; mlflow / the gateway are touched only at run time.
"""

from anvil.eval.engines import register_engine


def _register() -> None:
    """Register the pitcrew eval engine (lazy import keeps this cheap)."""
    from anvil.domains.pitcrew.eval import evaluate_pitcrew

    register_engine("pitcrew", evaluate_pitcrew)


_register()
