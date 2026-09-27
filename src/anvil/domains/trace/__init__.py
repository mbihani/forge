"""Trace domain: use an agent's MLflow traces as the eval dataset (Option A).

Binds FORGE to any agent whose MLflow traces already carry LLM-judge and
human assessments. Because FORGE mutates the agent every round, a trace's
stored assessment scored the OLD agent — so the traces supply the dataset +
scoring TARGETS, and each round RE-RUNS the mutated agent over the trace
inputs and RE-SCORES:

* **ingest** (``anvil.domains.trace.ingest`` + ``scripts/ingest_traces.py``)
  — freeze an experiment's labelled traces into a JSONL snapshot once per
  optimizer session.
* **eval** (:func:`anvil.domains.trace.eval.evaluate_trace`) — read the
  frozen snapshot, re-run the mutated agent per row, re-score each
  assessment, and macro-aggregate into an ``EvalReport``.

This package self-registers its eval engine under the name ``"trace"`` on
import (see below), so the domain-agnostic core never names this domain —
it resolves the engine by config name and imports this package by
convention. Importing this package is cheap: it only registers a callable;
mlflow / the gateway are touched only at run time.
"""

from anvil.eval.engines import register_engine


def _register() -> None:
    """Register the trace eval engine (lazy import keeps this cheap)."""
    from anvil.domains.trace.eval import evaluate_trace

    register_engine("trace", evaluate_trace)


_register()
