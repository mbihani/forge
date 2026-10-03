"""ANVIL — self-mutating agent harness on Databricks."""

import os

# Force MLflow trace export to be synchronous BEFORE mlflow is imported
# anywhere in the process. MLflow's v3 trace exporter
# (mlflow/tracing/export/mlflow_v3.py) caches the async-logging flag ONCE in
# __init__ via ``self._is_async_enabled = self._should_enable_async_logging()``,
# so flipping ``MLFLOW_ENABLE_ASYNC_TRACE_LOGGING`` after the exporter is
# constructed (e.g. inside ``evaluate_branch``) is too late — the cached flag
# stays whatever it was at first construction. On the Databricks Tracing
# Server, async export races ``mlflow.genai.evaluate``'s immediate per-row
# ``get_trace(request_id)``; a trace not yet persisted leaves
# ``eval_item.trace`` None and crashes the harness in
# ``_get_new_expectations`` (``AttributeError: 'NoneType' ... has no
# attribute 'info'``). Setting it here — the first executable statement,
# before any anvil submodule imports mlflow — ensures the exporter is built
# with async disabled, so traces are available by the time the harness
# scores them. ``setdefault`` honors an explicit user/process override.
#
# This reduces how often a per-row trace is missing; the harness-level shim
# in ``anvil.eval.runner._resilient_eval_harness`` is the guarantee that a
# missing trace never crashes the run.
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

# LiteLLM (imported by MLflow's eval harness and its trace-cost lookup, and by
# anvil.catalog's price fallback) calls ``dotenv.load_dotenv()`` on import
# unless LITELLM_MODE is PRODUCTION. That walks up from the installed package
# and loads the first ``.env`` it finds — e.g. a stray ``~/.env`` — into
# os.environ mid-run, which can silently switch DATABRICKS_CONFIG_PROFILE (and
# so the workspace the gateway and MLflow talk to) halfway through a round.
# ``setdefault`` honors an explicit override.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

__version__ = "0.2.0"
