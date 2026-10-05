"""Pitcrew domain: the FINRA regulatory-rule PDF summarizer.

Re-runs the (mutated) summarizer over a vendored 20-PDF FINRA corpus and
scores every output with pitcrew's OWN judges, reviewed from its production
MLflow experiment into ``eval/agent_evals/`` (``eval.agent_evals``):

* ``summarizer`` — the one LLM call production makes: composed system prompt
  + the PDF as a content block (``document`` for Claude, ``file`` for GPT).
* ``agent_base`` — the code-mode :class:`PitcrewAgent` the optimizer subclasses.
* ``eval`` — the engine: times each agent call, prices it, scores it with the
  agent's judges in production's input/output shape.

Self-registers its eval engine as ``"pitcrew"`` on import.
"""

from anvil.eval.engines import register_engine


def _register() -> None:
    """Register the pitcrew eval engine (lazy import keeps this cheap)."""
    from anvil.domains.pitcrew.eval import evaluate_pitcrew

    register_engine("pitcrew", evaluate_pitcrew)


_register()
