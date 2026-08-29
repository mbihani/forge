"""Tests for :mod:`anvil.data.mlflow_baseline`.

Covers the cross-vendor review fixes:

* WARNING 1 — ``experiment_id`` validation + URL encoding.
* WARNING 2 — pagination no longer silently truncates (safety limit raises).
* WARNING 3 — CLI call timeout raises a clear ``RuntimeError``.
* WARNING 4 — non-finite metric values are skipped; all-NaN ``judge.accuracy``
  raises; all-NaN per-field metrics are excluded.
* BLOCKING 2 — CLI/profile resolution (PATH + env var) replaces hardcoded
  defaults.

No real CLI or network calls are made: ``_api`` / ``_collect_run_metrics`` /
``_collect_run_ids`` are monkeypatched so the pagination/finite/timeout logic
is exercised in isolation.
"""

from __future__ import annotations

import subprocess

import pytest

from anvil.data import mlflow_baseline as mb
from anvil.eval.cache import CachedBaseline

# ---------------------------------------------------------------------------
# BLOCKING 2 — CLI / profile resolution
# ---------------------------------------------------------------------------


def test_resolve_cli_prefers_explicit_arg() -> None:
    assert mb._resolve_cli("/custom/databricks") == "/custom/databricks"


def test_resolve_cli_falls_back_to_hardcoded_when_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mb.shutil, "which", lambda name: None)
    assert mb._resolve_cli() == mb._FALLBACK_CLI


def test_resolve_cli_uses_path_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mb.shutil, "which", lambda name: "/usr/bin/databricks")
    assert mb._resolve_cli() == "/usr/bin/databricks"


def test_resolve_profile_prefers_explicit_arg() -> None:
    assert mb._resolve_profile("custom") == "custom"


def test_resolve_profile_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_PROFILE", "fevm-stable")
    assert mb._resolve_profile() == "fevm-stable"


def test_resolve_profile_defaults_to_DEFAULT(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_PROFILE", raising=False)
    assert mb._resolve_profile() == "DEFAULT"


# ---------------------------------------------------------------------------
# WARNING 1 — experiment_id validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "abc", "12a3", "12.3", "-123", "exp-1"])
def test_build_rejects_non_numeric_experiment_id(bad: str) -> None:
    with pytest.raises(ValueError, match="non-empty numeric string"):
        mb.build_mlflow_baseline(experiment_id=bad)


def test_build_accepts_whitespace_padded_numeric_experiment_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace is stripped before the digits-only check, so a padded
    numeric ID is accepted."""
    monkeypatch.setattr(
        mb, "_collect_run_metrics", lambda eid, **kw: {"r": {"metrics": {"judge.accuracy": 0.9}}}
    )
    baseline = mb.build_mlflow_baseline(experiment_id="  123  ")
    assert baseline.aggregate == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# WARNING 4 — non-finite metric handling
# ---------------------------------------------------------------------------


def test_build_skips_nonfinite_field_keeps_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NaN field value on one run is skipped; the finite value on the
    other run still contributes to the macro-average."""
    runs = {
        "r1": {"metrics": {"judge.accuracy": 0.9, "judge.amount": float("nan")}},
        "r2": {"metrics": {"judge.accuracy": 0.8, "judge.amount": 0.7}},
    }
    monkeypatch.setattr(mb, "_collect_run_metrics", lambda eid, **kw: runs)
    baseline = mb.build_mlflow_baseline(experiment_id="123")
    assert baseline.aggregate == pytest.approx(0.85)
    assert baseline.per_judge["field_amount"] == pytest.approx(0.7)


def test_build_excludes_field_with_all_nonfinite_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every value for a per-field metric is non-finite, the metric is
    excluded from ``per_judge`` (no NaN leaks into the baseline)."""
    runs = {
        "r1": {"metrics": {"judge.accuracy": 0.9, "judge.amount": float("nan")}},
        "r2": {"metrics": {"judge.accuracy": 0.8, "judge.amount": float("inf")}},
    }
    monkeypatch.setattr(mb, "_collect_run_metrics", lambda eid, **kw: runs)
    baseline = mb.build_mlflow_baseline(experiment_id="123")
    assert "field_amount" not in baseline.per_judge
    assert baseline.aggregate == pytest.approx(0.85)


def test_build_raises_when_all_accuracy_nonfinite(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {"r1": {"metrics": {"judge.accuracy": float("nan"), "judge.amount": 0.7}}}
    monkeypatch.setattr(mb, "_collect_run_metrics", lambda eid, **kw: runs)
    with pytest.raises(RuntimeError, match="no finite judge.accuracy"):
        mb.build_mlflow_baseline(experiment_id="123")


def test_build_raises_when_no_scored_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mb, "_collect_run_metrics", lambda eid, **kw: {})
    with pytest.raises(RuntimeError, match="no scored judge runs"):
        mb.build_mlflow_baseline(experiment_id="123")


def test_build_returns_cached_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {"r1": {"metrics": {"judge.accuracy": 0.9, "judge.amount": 0.8}}}
    monkeypatch.setattr(mb, "_collect_run_metrics", lambda eid, **kw: runs)
    baseline = mb.build_mlflow_baseline(experiment_id="123")
    assert isinstance(baseline, CachedBaseline)
    assert baseline.mode == "mlflow"
    assert baseline.scorer_fingerprint == ""
    assert baseline.n_examples == 1


# ---------------------------------------------------------------------------
# WARNING 2 — pagination safety limit
# ---------------------------------------------------------------------------


def test_collect_run_metrics_raises_on_too_many_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the API keeps returning a next_page_token, the safety limit trips
    instead of silently truncating or looping forever."""

    def fake_api(method: str, path: str, body: dict | None = None, **kw) -> dict:
        return {
            "runs": [{"info": {"run_id": "r"}, "data": {"metrics": []}}],
            "next_page_token": "tok",
        }

    monkeypatch.setattr(mb, "_api", fake_api)
    with pytest.raises(RuntimeError, match="too many run pages"):
        mb._collect_run_metrics("123", cli="databricks", profile="DEFAULT", max_pages=3)


def test_collect_run_ids_raises_on_too_many_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_api(method: str, path: str, body: dict | None = None, **kw) -> dict:
        # Non-empty traces so the ``not traces`` break guard doesn't trip
        # before the safety limit — the pagination must keep going.
        return {"traces": [{"request_metadata": []}], "next_page_token": "tok"}

    monkeypatch.setattr(mb, "_api", fake_api)
    with pytest.raises(RuntimeError, match="too many trace pages"):
        mb._collect_run_ids("123", cli="databricks", profile="DEFAULT", max_pages=3)


def test_collect_run_metrics_stops_when_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pagination stops cleanly when next_page_token is absent."""

    def fake_api(method: str, path: str, body: dict | None = None, **kw) -> dict:
        return {"runs": [{"info": {"run_id": "r"}, "data": {"metrics": []}}]}

    monkeypatch.setattr(mb, "_api", fake_api)
    runs = mb._collect_run_metrics("123", cli="databricks", profile="DEFAULT")
    assert "r" in runs


# ---------------------------------------------------------------------------
# WARNING 3 — CLI call timeout
# ---------------------------------------------------------------------------


def test_api_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr(mb.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="timed out after 1s"):
        mb._api("get", "/api/2.0/x", cli="databricks", profile="DEFAULT", timeout=1)


# ---------------------------------------------------------------------------
# WARNING 1 — URL encoding of experiment_id in the trace query
# ---------------------------------------------------------------------------


def test_collect_run_ids_url_encodes_experiment_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The experiment_id is URL-encoded into the trace query path (defensive
    against any non-digit characters even though validation rejects them at
    the build entry point)."""
    captured: list[str] = []

    def fake_api(method: str, path: str, body: dict | None = None, **kw) -> dict:
        captured.append(path)
        return {"traces": []}  # empty → loop breaks immediately

    monkeypatch.setattr(mb, "_api", fake_api)
    mb._collect_run_ids("123", cli="databricks", profile="DEFAULT")
    assert captured, "expected at least one _api call"
    assert "experiment_ids=123" in captured[0]
    assert "max_results=" in captured[0]
