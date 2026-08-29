"""Build a :class:`CachedBaseline` from MLflow judge results via the Databricks REST API.

Ports the proven seed logic from ``savesage-forge/scripts/seed_baseline_from_mlflow.py``
into a reusable module so the Forge Orchestrator can seed the optimization
baseline from a real MLflow experiment instead of a local golden-set eval.
The resulting :class:`CachedBaseline` is written to
``eval/runs/baseline.json`` and picked up by the round gate exactly like a
locally-evaluated baseline — the only compatibility check the round loop
performs (``scorer_fingerprint``) is a no-op because the MLflow baseline
carries an empty fingerprint.

Method (matches ``judge/scorer.py`` provenance — the bank lives in the trace
inputs, not a run tag):

  1. Paginate ``GET /api/2.0/mlflow/traces`` → optionally filter traces whose
     ``mlflow.traceInputs.bank == <bank_filter>``; collect their
     ``mlflow.sourceRun``.
  2. Paginate ``POST /api/2.0/mlflow/runs/search`` → for each run that carries
     ``judge.accuracy``, read its ``judge.*`` metrics.
  3. Macro-average across runs exactly like ``judge/scorer.py::_aggregate_results``:
     ``aggregate`` = mean ``judge.accuracy``; per-field = mean over runs having it.

Uses the Databricks CLI (mlflow is not pip-installable on the deploy host) via
the newer binary, stripping its version banner from stdout. Both the CLI path
and the active profile are configurable with sensible defaults so the same
code runs on the local Mac and the forge deploy.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

from anvil.eval.cache import CachedBaseline

# Default Databricks CLI binary — the newer one that supports ``api``. The
# local Mac blackholes pypi/npm but the CLI is fine; on the forge deploy the
# path may differ, so it is overridable via the ``cli`` parameter.
_DEFAULT_CLI = "/usr/local/bin/databricks"

# Non-per-field judge metrics to exclude when mapping to per-field slugs.
# ``accuracy`` / ``accuracy_forgiven`` are top-level aggregates, ``comparisons``
# / ``scored`` / ``correct`` are counts — none of them are per-field scores.
_DEFAULT_NON_FIELD = frozenset(
    {"accuracy", "accuracy_forgiven", "comparisons", "scored", "correct"}
)


def _api(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    cli: str,
    profile: str,
) -> dict:
    """Call the Databricks REST API via the CLI; parse JSON after the banner.

    The Databricks CLI prints a version banner to stdout before the JSON
    payload. We find the first ``{`` and parse from there — robust against
    banner changes across CLI versions.
    """
    cmd = [cli, "api", method, path, "--profile", profile]
    if body is not None:
        cmd += ["--json", json.dumps(body)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout
    brace = out.find("{")
    if brace < 0:
        raise RuntimeError(f"no JSON in CLI output for {path}: {out[:200]}")
    return json.loads(out[brace:])


def _collect_run_ids(
    experiment_id: str,
    *,
    cli: str,
    profile: str,
    bank_filter: str | None = None,
    page_size: int = 500,
    max_pages: int = 40,
) -> set[str]:
    """Paginate traces; return ``sourceRun`` ids, optionally filtered by bank.

    The bank lives in ``mlflow.traceInputs`` (a JSON-encoded trace metadata
    field), not a run tag — so we must walk the traces to discover which runs
    belong to a given bank. When ``bank_filter`` is ``None`` every trace's
    source run is collected (no bank restriction).
    """
    run_ids: set[str] = set()
    token: str | None = None
    for _ in range(max_pages):
        q = f"/api/2.0/mlflow/traces?experiment_ids={experiment_id}&max_results={page_size}"
        if token:
            q += f"&page_token={token}"
        data = _api("get", q, cli=cli, profile=profile)
        traces = data.get("traces", [])
        for t in traces:
            md = {m["key"]: m["value"] for m in t.get("request_metadata", [])}
            try:
                bank = json.loads(md.get("mlflow.traceInputs", "{}")).get("bank")
            except json.JSONDecodeError:
                bank = None
            run = md.get("mlflow.sourceRun")
            if bank_filter and bank != bank_filter:
                continue
            if run:
                run_ids.add(run)
        token = data.get("next_page_token")
        if not token or not traces:
            break
    return run_ids


def _collect_run_metrics(
    experiment_id: str,
    *,
    cli: str,
    profile: str,
    page_size: int = 1000,
    max_pages: int = 40,
) -> dict[str, dict]:
    """Paginate ``runs/search``; return ``{run_id: {"metrics": {...}}}``."""
    runs: dict[str, dict] = {}
    token: str | None = None
    for _ in range(max_pages):
        body: dict = {"experiment_ids": [experiment_id], "max_results": page_size}
        if token:
            body["page_token"] = token
        data = _api("post", "/api/2.0/mlflow/runs/search", body, cli=cli, profile=profile)
        page = data.get("runs", [])
        for r in page:
            info = r.get("info", {})
            rdata = r.get("data", {})
            metrics = {m["key"]: m["value"] for m in rdata.get("metrics", [])}
            runs[info.get("run_id")] = {"metrics": metrics}
        token = data.get("next_page_token")
        if not token or not page:
            break
    return runs


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_mlflow_baseline(
    experiment_id: str,
    bank_filter: str | None = None,
    profile: str = "full",
    *,
    cli: str = _DEFAULT_CLI,
    non_field_metrics: frozenset[str] = _DEFAULT_NON_FIELD,
) -> CachedBaseline:
    """Build a :class:`CachedBaseline` from an MLflow experiment's judge results.

    Parameters
    ----------
    experiment_id:
        Numeric MLflow experiment ID (e.g. ``"967014443183055"``).
    bank_filter:
        Optional bank name (e.g. ``"ICICI"``). When set, only runs that
        appear as a ``sourceRun`` on a trace whose ``mlflow.traceInputs.bank``
        matches are scored. When ``None``, every run carrying
        ``judge.accuracy`` is scored regardless of bank.
    profile:
        Databricks CLI profile to use for the REST calls (default ``"full"``).
    cli:
        Path to the Databricks CLI binary (default ``/usr/local/bin/databricks``
        — the newer one that supports ``api``). Override when the deploy host
        has the CLI elsewhere.
    non_field_metrics:
        Judge metric slugs (without the ``judge.`` prefix) that are NOT
        per-field scores and should be excluded from the per-field mapping.

    Returns
    -------
    CachedBaseline
        A baseline with ``mode="mlflow"``, ``scaffold_commit_sha="mlflow-seed"``,
        and an empty ``scorer_fingerprint`` (so the round loop's compatibility
        check is a no-op). The ``per_judge`` dict carries the macro-averaged
        ``accuracy``, ``accuracy_forgiven``, and ``field_<slug>`` scores.

    Raises
    ------
    RuntimeError
        If no scored judge runs are found in the experiment (optionally
        filtered by bank).
    """
    if bank_filter:
        filtered_run_ids = _collect_run_ids(
            experiment_id, cli=cli, profile=profile, bank_filter=bank_filter
        )
        candidate_ids = filtered_run_ids
    else:
        # No bank filter: score every run in the experiment that has
        # judge.accuracy — skip the trace walk entirely.
        candidate_ids = None

    all_runs = _collect_run_metrics(experiment_id, cli=cli, profile=profile)

    if candidate_ids is not None:
        candidate_run_ids = candidate_ids
    else:
        candidate_run_ids = set(all_runs.keys())

    scored: list[dict[str, float]] = []
    for run_id in candidate_run_ids:
        rec = all_runs.get(run_id)
        if not rec:
            continue
        metrics = rec["metrics"]
        # A scored run has judge.accuracy (true or error — an error run still
        # produced a verdict). Require it present.
        if "judge.accuracy" not in metrics:
            continue
        scored.append(metrics)

    if not scored:
        bank_msg = f" for bank {bank_filter!r}" if bank_filter else ""
        raise RuntimeError(
            f"no scored judge runs found in experiment {experiment_id}{bank_msg}"
            " — cannot seed baseline"
        )

    accuracy = _mean([m["judge.accuracy"] for m in scored if "judge.accuracy" in m])
    forgiven = _mean(
        [m["judge.accuracy_forgiven"] for m in scored if "judge.accuracy_forgiven" in m]
    )

    per_judge: dict[str, float] = {}
    if accuracy is not None:
        per_judge["accuracy"] = accuracy
    if forgiven is not None:
        per_judge["accuracy_forgiven"] = forgiven

    # Per-field means (keyed as field_<slug> to line up with the eval report).
    field_keys = {
        k.removeprefix("judge.")
        for m in scored
        for k in m
        if k.startswith("judge.") and k.removeprefix("judge.") not in non_field_metrics
    }
    for slug in sorted(field_keys):
        vals = [m[f"judge.{slug}"] for m in scored if f"judge.{slug}" in m]
        mean = _mean(vals)
        if mean is not None:
            per_judge[f"field_{slug}"] = mean

    return CachedBaseline(
        scaffold_commit_sha="mlflow-seed",
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        mode="mlflow",
        scorers=sorted(per_judge.keys()),
        runtime_endpoint="",
        judge_endpoint="",
        aggregate=accuracy if accuracy is not None else 0.0,
        per_judge=per_judge,
        per_bucket={},
        n_examples=len(scored),
        mlflow_run_id=None,
        scorer_fingerprint="",  # empty → round.py compatibility check is a no-op
    )
