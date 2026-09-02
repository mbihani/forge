"""Savesage ICICI domain: auto-optimize the credit-card statement prompt.

Binds ANVIL to the Savesage ``statement-agent`` extraction task:

* **scaffold** — ``prompts/icici.txt`` decomposed into per-section skills
  the optimizer edits one at a time.
* **golden set** — one ICICI statement per row, carrying the cached Opus GT.
* **eval** — :func:`evaluate_savesage` composes the prompt, runs Luna, and
  scores per-field accuracy against the GT with the production judge
  modules (no LLM per round); reproduces the MLflow ``judge.*`` baseline.

This package self-registers its eval engine under the name ``"savesage"``
on import (see below), so the domain-agnostic core (``anvil.eval.engines``
/ ``anvil.eval.runner``) never names this domain — it resolves the engine
by config name and imports this package by convention. Importing this
package does NOT touch the statement-agent tree: ``evaluate_savesage`` and
the extractor call ``ensure_importable()`` only at run time, so
registration is safe even where the statement-agent checkout is absent.
"""

from anvil.eval.engines import register_engine


def _register() -> None:
    """Register the savesage eval engine (lazy import keeps this cheap)."""
    from anvil.domains.savesage.eval import evaluate_savesage

    register_engine("savesage", evaluate_savesage)


_register()
