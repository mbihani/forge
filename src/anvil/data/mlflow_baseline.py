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
import math
import os
import shutil
import subprocess
from datetime import UTC, datetime
from urllib.parse import quote

from anvil.eval.cache import CachedBaseline

# Hardcoded fallback for the Databricks CLI binary — the newer one that
# supports ``api``. Used only when the CLI is not found on PATH. The local
# Mac blackholes pypi/npm but the CLI is fine; on the forge deploy the path
# may differ, so the binary is resolved at call time via
# :func:`_resolve_cli` (PATH lookup first, this fallback second) and is
# overridable via the ``cli`` parameter.
_FALLBACK_CLI = "/usr/local/bin/databricks"

# Non-per-field judge metrics to exclude when mapping to per-field slugs.
# ``accuracy`` / ``accuracy_forgiven`` are top-level aggregates, ``comparisons``
# / ``scored`` / ``correct`` are counts — none of them are per-field scores.
_DEFAULT_NON_FIELD = frozenset(
    {"accuracy", "accuracy_forgiven", "comparisons", "scored", "correct"}
)


def _resolve_cli(cli: str | None = None) -> str:
    """Resolve the Databricks CLI binary path.

    Prefers an explicit ``cli`` arg, then ``shutil.which`` (PATH lookup, so
    the same code works on the local Mac and the forge deploy), then the
    hardcoded :data:`_FALLBACK_CLI` path.
    """
    if cli:
        return cli
    found = shutil.which("databricks")
    return found if found else _FALLBACK_CLI


def _resolve_profile(profile: str | None = None) -> str:
    """Resolve the Databricks CLI profile.

    Prefers an explicit ``profile`` arg, then the ``DATABRICKS_PROFILE`` env
    var, then ``"DEFAULT"`` (the conventional profile ``databricks configure``
    sets up). Resolved at call time so env changes take effect without a
    re-import.
    """
    if profile:
        return profile
    return os.environ.get("DATABRICKS_PROFILE") or "DEFAULT"


def _api(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    cli: str,
    profile: str,
    timeout: int = 120,
) -> dict:
    """Call the Databricks REST API via the CLI; parse JSON after the banner.

    The Databricks CLI prints a version banner to stdout before the JSON
    payload. We find the first ``{`` and parse from there — robust against
    banner changes across CLI versions.

    ``timeout`` (seconds) guards against a hung CLI call; a
    :class:`subprocess.TimeoutExpired` is re-raised as a
    :class:`RuntimeError` with a clear message.
    """
    cmd = [cli, "api", method, path, "--profile", profile]
    if body is not None:
        cmd += ["--json", json.dumps(body)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Databricks CLI call to {path} timed out after {timeout}s"
        ) from exc
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
    max_pages: int = 200,
) -> set[str]:
    """Paginate traces; return ``sourceRun`` ids, optionally filtered by bank.

    The bank lives in ``mlflow.traceInputs`` (a JSON-encoded trace metadata
    field), not a run tag — so we must walk the traces to discover which runs
    belong to a given bank. When ``bank_filter`` is ``None`` every trace's
    source run is collected (no bank restriction).

    Pagination continues until ``next_page_token`` is absent/empty (no
    silent truncation). ``max_pages`` is a safety limit only — if reached
    the experiment has too many pages and a :class:`RuntimeError` is raised
    rather than silently dropping data.
    """
    run_ids: set[str] = set()
    token: str | None = None
    page = 0
    while True:
        page += 1
        if page > max_pages:
            raise RuntimeError(
                f"experiment {experiment_id} has too many trace pages to "
                f"process (safety limit {max_pages} reached)"
            )
        q = (
            f"/api/2.0/mlflow/traces?experiment_ids={quote(experiment_id)}"
            f"&max_results={page_size}"
        )
        if token:
            q += f"&page_token={quote(token)}"
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
    max_pages: int = 200,
) -> dict[str, dict]:
    """Paginate ``runs/search``; return ``{run_id: {"metrics": {...}}}``.

    Pagination continues until ``next_page_token`` is absent/empty (no
    silent truncation). ``max_pages`` is a safety limit only — if reached
    a :class:`RuntimeError` is raised rather than silently dropping runs.
    """
    runs: dict[str, dict] = {}
    token: str | None = None
    page_num = 0
    while True:
        page_num += 1
        if page_num > max_pages:
            raise RuntimeError(
                f"experiment {experiment_id} has too many run pages to "
                f"process (safety limit {max_pages} reached)"
            )
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
    profile: str | None = None,
    *,
    cli: str | None = None,
    non_field_metrics: frozenset[str] = _DEFAULT_NON_FIELD,
) -> CachedBaseline:
    """Build a :class:`CachedBaseline` from an MLflow experiment's judge results.

    Parameters
    ----------
    experiment_id:
        Numeric MLflow experiment ID (e.g. ``"967014443183055"``). Must be a
        non-empty string of digits (whitespace is stripped); a
        :class:`ValueError` is raised otherwise.
    bank_filter:
        Optional bank name (e.g. ``"ICICI"``). When set, only runs that
        appear as a ``sourceRun`` on a trace whose ``mlflow.traceInputs.bank``
        matches are scored. When ``None``, every run carrying
        ``judge.accuracy`` is scored regardless of bank.
    profile:
        Databricks CLI profile to use for the REST calls. When ``None``
        (default) the profile is resolved by :func:`_resolve_profile`: the
        ``DATABRICKS_PROFILE`` env var, then ``"DEFAULT"``.
    cli:
        Path to the Databricks CLI binary. When ``None`` (default) the binary
        is resolved by :func:`_resolve_cli`: ``shutil.which("databricks")``
        (PATH lookup), then the hardcoded ``/usr/local/bin/databricks``
        fallback. Override when the deploy host has the CLI elsewhere.
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
    ValueError
        If ``experiment_id`` is not a non-empty numeric string.
    RuntimeError
        If no scored judge runs are found in the experiment (optionally
        filtered by bank), if ``judge.accuracy`` has no finite values, or if
        the experiment has too many pages to process.
    """
    # Validate experiment_id: a non-empty numeric string (digits only).
    eid = experiment_id.strip()
    if not eid or not eid.isdigit():
        raise ValueError(
            f"experiment_id must be a non-empty numeric string (digits only), "
            f"got {experiment_id!r}"
        )

    cli = _resolve_cli(cli)
    profile = _resolve_profile(profile)

    if bank_filter:
        filtered_run_ids = _collect_run_ids(
            eid, cli=cli, profile=profile, bank_filter=bank_filter
        )
        candidate_ids = filtered_run_ids
    else:
        # No bank filter: score every run in the experiment that has
        # judge.accuracy — skip the trace walk entirely.
        candidate_ids = None

    all_runs = _collect_run_metrics(eid, cli=cli, profile=profile)

    candidate_run_ids = candidate_ids if candidate_ids is not None else set(all_runs.keys())

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
            f"no scored judge runs found in experiment {eid}{bank_msg}"
            " — cannot seed baseline"
        )

    # Collect only finite metric values (skip NaN/inf). judge.accuracy must
    # have at least one finite value or we cannot seed a baseline.
    acc_vals = [
        m["judge.accuracy"]
        for m in scored
        if "judge.accuracy" in m and math.isfinite(m["judge.accuracy"])
    ]
    if not acc_vals:
        raise RuntimeError(
            f"no finite judge.accuracy values found in experiment {eid}"
            " — cannot seed baseline"
        )
    accuracy = _mean(acc_vals)

    forgiven_vals = [
        m["judge.accuracy_forgiven"]
        for m in scored
        if "judge.accuracy_forgiven" in m and math.isfinite(m["judge.accuracy_forgiven"])
    ]
    forgiven = _mean(forgiven_vals)

    per_judge: dict[str, float] = {"accuracy": accuracy}
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
        vals = [
            m[f"judge.{slug}"]
            for m in scored
            if f"judge.{slug}" in m and math.isfinite(m[f"judge.{slug}"])
        ]
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
        aggregate=accuracy,
        per_judge=per_judge,
        per_bucket={},
        n_examples=len(scored),
        mlflow_run_id=None,
        scorer_fingerprint="",  # empty → round.py compatibility check is a no-op
    )
