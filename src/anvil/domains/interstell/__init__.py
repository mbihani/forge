"""Interstell domain: optimize the Interstell debt-relief voice agent's runtime levers.

The agent (``mayanck-bihani_data/interstell-ai``) is one streaming tool-calling Chat
Completions turn with the client's fixed system prompt. forge searches its runtime levers
(model, route, temperature, max_completion_tokens; reasoning fixed at none) for lower
first-action latency without losing correctness, scoring every round with Interstell's
own scorers and Opus judges (``evals/``) through the deployed App. See
:mod:`anvil.domains.interstell.eval`.

Self-registers its eval engine and lever probe under ``"interstell"`` on import; the
Interstell repo is only imported at run time.
"""

from anvil.eval.engines import register_engine, register_lever_probe


def _register() -> None:
    from anvil.domains.interstell.eval import evaluate_interstell
    from anvil.domains.interstell.probe import probe_interstell

    register_engine("interstell", evaluate_interstell)
    register_lever_probe("interstell", probe_interstell)


_register()
