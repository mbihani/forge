"""MLflow tracing helpers — shared across runtime, eval, optimizer.

Two responsibilities:

* :func:`enable_runtime_tracing` — flip on
  :func:`mlflow.openai.autolog` so the OpenAI
  ``chat.completions.create`` call inside ``AnvilAgent.predict``
  becomes a ``CHAT_MODEL`` sub-span under the ``predict`` root span.
  Idempotent.

* :func:`tag_current_trace` — attach the standard ANVIL operational
  tags (``source``, ``runtime_endpoint``, ``scaffold_branch``,
  ``scaffold_commit_sha``, optional ``round``) to the active trace.
  Every runtime trace is queried by these tags from the optimizer
  side, so they are not cosmetic.

The git-SHA tag intentionally substitutes for an MLflow Prompt
Registry link: the scaffold commit pinpoints the exact mutable state
that produced the response, and ``git checkout <sha>`` reproduces it
byte-for-byte.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import mlflow
import mlflow.openai
from mlflow.entities import SpanType
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

# Shared root-span name for a single eval row's trace. The genai runner and
# any pluggable domain engine use the SAME name so the optimizer side can
# query per-row eval traces uniformly regardless of which engine produced
# them.
PREDICT_SPAN_NAME = "anvil.predict"

SOURCE_PRODUCTION = "production"
SOURCE_EVAL = "eval"
SOURCE_OPTIMIZER = "optimizer"
SourceTag = Literal["production", "eval", "optimizer"]

_DETACHED_HEAD_LITERAL = "HEAD"

# Env-var fallbacks for runtime contexts where the scaffold lives at
# a path that is not a git repo (Databricks Apps containers,
# Volume-synced multi-pod). Populated by the deploy wrapper from
# ``git rev-parse`` at deploy time so the SHA tag still pins the
# exact scaffold the optimizer needs to query.
_ENV_SCAFFOLD_COMMIT_SHA = "ANVIL_SCAFFOLD_COMMIT_SHA"
_ENV_SCAFFOLD_BRANCH = "ANVIL_SCAFFOLD_BRANCH"


def configure_tracking_uri(profile: str | None = None) -> str:
    """Point MLflow at the Databricks workspace, not a local file. Returns the URI.

    * A named profile (not the ``"DEFAULT"`` sentinel) → ``databricks://<profile>``,
      and ``DATABRICKS_CONFIG_PROFILE`` is bound for the gateway client.
    * Otherwise, if nothing set a tracking URI (no ``MLFLOW_TRACKING_URI``, no
      earlier ``mlflow.set_tracking_uri``) but Databricks credentials are present
      (``DATABRICKS_HOST``, ``DATABRICKS_CONFIG_PROFILE``, or ``~/.databrickscfg``)
      → ``databricks``. Without this, MLflow 3.x silently defaults to a local
      ``sqlite:///<cwd>/mlflow.db`` — in a sandbox that file is wiped with the
      sandbox, taking every eval trace and optimizer transcript with it.
    * An explicitly set URI is always respected (tests, local-file setups).

    Warns when the result is still a local store, so it is visible in the log.
    """
    if profile and profile != "DEFAULT":
        mlflow.set_tracking_uri(f"databricks://{profile}")
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    else:
        from mlflow.tracking._tracking_service.utils import is_tracking_uri_set  # noqa: PLC0415

        has_databricks_creds = bool(
            os.environ.get("DATABRICKS_HOST")
            or os.environ.get("DATABRICKS_CONFIG_PROFILE")
            or Path("~/.databrickscfg").expanduser().is_file()
        )
        if not is_tracking_uri_set() and has_databricks_creds:
            mlflow.set_tracking_uri("databricks")
    uri = mlflow.get_tracking_uri()
    if not uri.startswith("databricks"):
        logger.warning(
            "MLflow tracking URI is %r — traces and optimizer transcripts stay LOCAL and "
            "do not reach the Databricks workspace. Set MLFLOW_TRACKING_URI=databricks "
            "(with DATABRICKS_HOST/TOKEN or a profile) to keep them.",
            uri,
        )
    return uri


def ensure_experiment_folder(experiment: str) -> None:
    """Create the workspace folder an experiment lives in, if it is missing.

    Databricks refuses to create ``/Shared/forge/<domain>/eval`` until
    ``/Shared/forge/<domain>`` exists ("Parent directory does not exist"), so
    a domain's first run would silently get no experiment. Only acts when
    MLflow points at Databricks; uses the profile in the tracking URI
    (``databricks://<profile>``) or the ambient SDK config. Best-effort — a
    failure is logged and the experiment creation that follows reports it.
    """
    uri = mlflow.get_tracking_uri()
    if not uri.startswith("databricks"):
        return
    parent = experiment.rsplit("/", 1)[0]
    if not parent:
        return
    try:
        from databricks.sdk import WorkspaceClient  # noqa: PLC0415

        profile = uri.split("://", 1)[1] if "://" in uri else None
        WorkspaceClient(profile=profile or None).workspace.mkdirs(parent)
    except Exception as exc:  # noqa: BLE001 - best-effort; set_experiment reports the real error
        logger.warning("could not create experiment folder %r: %s", parent, exc)


def setup_eval_tracing(*, experiment: str, profile: str | None = None) -> str | None:
    """Arm MLflow tracing for an eval run — GENERIC across every engine.

    Binds the Databricks profile (so the gateway client's host/token resolve),
    selects the eval ``experiment``, and enables ``mlflow.openai.autolog`` so
    any gateway ``chat.completions.create`` call — the builtin genai agent OR a
    pluggable domain engine (e.g. savesage) — emits a trace to that
    experiment. Call this ONCE before dispatching to the engine; previously
    this setup lived only on the genai path, so custom engines produced no
    per-row traces of the optimization activity.

    ``"DEFAULT"`` is ``run_round``'s sentinel for ambient/native auth and is
    NOT bound as a profile (binding ``databricks://DEFAULT`` would disable the
    SDK's env-var fallback). Returns the experiment id when resolvable, else
    None — best-effort, never raises on a tracking-server hiccup.
    """
    configure_tracking_uri(profile)
    experiment_id: str | None = None
    try:
        # Creates the experiment when it does not exist yet (needs write access
        # to its parent workspace folder, e.g. /Shared or /Users/<you>).
        if mlflow.get_experiment_by_name(experiment) is None:
            ensure_experiment_folder(experiment)
        exp = mlflow.set_experiment(experiment)
        experiment_id = getattr(exp, "experiment_id", None)
    except Exception as exc:  # noqa: BLE001 - best-effort: tracing is observability
        logger.warning("setup_eval_tracing: could not set experiment %r: %s", experiment, exc)
    enable_runtime_tracing()
    return experiment_id


@contextmanager
def eval_row_trace(
    *,
    example_id: str,
    query: str,
    scaffold_root: Path | str,
    runtime_endpoint: str,
    round: int | None = None,
) -> Iterator[Any]:
    """Open a per-row root ``CHAIN`` span so one eval row yields ONE trace.

    An engine wraps each row's work in this context so the autologged
    ``CHAT_MODEL`` sub-spans (the summarize call, the judge call, …) nest
    under a single per-row trace, tagged with the standard operational tags
    (``source=eval`` + scaffold branch/sha, which identify the round). This
    is the opt-in grouping layer on top of :func:`setup_eval_tracing`'s
    autolog; an engine that does not adopt it still gets per-call autolog
    traces, just not grouped.

    Best-effort: if MLflow cannot open a span (no tracking context, server
    down), yields ``None`` and the caller's body still runs untraced. Errors
    inside the caller's body propagate normally.
    """
    span_cm = None
    try:
        span_cm = mlflow.start_span(name=PREDICT_SPAN_NAME, span_type=SpanType.CHAIN)
    except Exception as exc:  # noqa: BLE001 - tracing must never fail a row
        logger.warning("eval_row_trace: could not start span for %s: %s", example_id, exc)
        span_cm = None
    if span_cm is None:
        yield None
        return
    with span_cm as span:
        try:
            span.set_inputs({"example_id": example_id, "query": query})
            tag_current_trace(
                source=SOURCE_EVAL,
                scaffold_root=scaffold_root,
                runtime_endpoint=runtime_endpoint,
                round=round,
            )
        except Exception as exc:  # noqa: BLE001 - tag/inputs are best-effort
            logger.warning("eval_row_trace: tagging failed for %s: %s", example_id, exc)
        yield span


def enable_runtime_tracing() -> None:
    """Enable ``mlflow.openai.autolog`` so chat completions become CHAT_MODEL spans.

    Idempotent. Caller is responsible for setting the tracking URI and
    experiment beforehand. Must be invoked **before** any OpenAI
    client makes its first ``chat.completions.create`` call (autolog
    patches the method on import-time symbols of
    ``openai.resources.chat.completions``).
    """
    mlflow.openai.autolog()


def tag_current_trace(
    *,
    source: SourceTag,
    scaffold_root: Path | str,
    runtime_endpoint: str,
    round: int | None = None,
) -> None:
    """Attach ANVIL's standard operational tag set to the active trace."""
    tags: dict[str, str] = {
        "source": source,
        "runtime_endpoint": runtime_endpoint,
    }
    branch = _resolve_scaffold_branch(scaffold_root) or _env(_ENV_SCAFFOLD_BRANCH)
    if branch is not None:
        tags["scaffold_branch"] = branch
    sha = _resolve_scaffold_commit_sha(scaffold_root) or _env(_ENV_SCAFFOLD_COMMIT_SHA)
    if sha is not None:
        tags["scaffold_commit_sha"] = sha
    if round is not None:
        tags["round"] = str(round)

    try:
        mlflow.update_current_trace(tags=tags)
    except MlflowException:
        return


def _resolve_scaffold_commit_sha(scaffold_root: Path | str) -> str | None:
    return _git_call(scaffold_root, ["rev-parse", "HEAD"])


def _resolve_scaffold_branch(scaffold_root: Path | str) -> str | None:
    branch = _git_call(scaffold_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch == _DETACHED_HEAD_LITERAL:
        return None
    return branch


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _git_call(scaffold_root: Path | str, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(scaffold_root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None
