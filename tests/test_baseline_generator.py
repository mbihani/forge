"""Tests for the EvalReport → CachedBaseline baseline generator.

Covers the three contracts the accept/reject gate depends on:

* :func:`report_to_baseline` maps every ``EvalReport`` field onto the
  ``CachedBaseline`` schema (including the two deliberate renames:
  ``n_rows`` → ``n_examples`` and ``run_id`` → ``mlflow_run_id``).
* The generated file is loadable by :func:`anvil.eval.load_baseline`
  — the exact reader ``round.py`` calls before every round.
* Every required ``CachedBaseline`` field is populated.

The CLI integration test exercises ``scripts/make_baseline.py``
end-to-end with ``evaluate_branch`` monkeypatched, so **no LLM and no
git** are invoked.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from anvil.eval.cache import (
    CachedBaseline,
    load_baseline,
    report_to_baseline,
    save_baseline,
)
from anvil.eval.runner import EvalReport

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FIELDS = [
    "scaffold_commit_sha",
    "evaluated_at",
    "mode",
    "scorers",
    "runtime_endpoint",
    "judge_endpoint",
    "aggregate",
    "per_judge",
    "per_bucket",
    "n_examples",
    "mlflow_run_id",
    "scorer_fingerprint",
    "cost_metrics",
]

_SHA = "a" * 40
_RUNTIME = "databricks-claude-sonnet-4-6"
_JUDGE = "databricks-claude-sonnet-4-6"


def _fake_report() -> EvalReport:
    """A mock eval result — no LLM, no mlflow, just the dataclass."""
    return EvalReport(
        aggregate=0.75,
        per_judge={
            "correctness": 0.5,
            "retrieval_groundedness": 0.875,
            "refusal_appropriateness": 1.0,
        },
        per_bucket={
            "direct": {
                "correctness": 0.5,
                "retrieval_groundedness": 1.0,
                "refusal_appropriateness": 1.0,
            },
            "out_of_scope": {
                "correctness": 0.0,
                "retrieval_groundedness": 0.0,
                "refusal_appropriateness": 1.0,
            },
        },
        failures=[],
        run_id="abc123def456",
        experiment_id="exp_1",
        n_rows=8,
        mode="quick",
        scorers=["correctness", "retrieval_groundedness", "refusal_appropriateness"],
        evaluated_at="2026-08-16T12:00:00+00:00",
        trace_ids=["t0", "t1"],
        cost_metrics={"total_context_chars": 128.0, "n_rows": 8.0},
        scorer_fingerprint=(
            '[{"check_function": null, "name": "correctness", "type": "llm", "weight": 1.0}]'
        ),
    )


def _baseline_from_fake() -> CachedBaseline:
    return report_to_baseline(
        _fake_report(),
        scaffold_commit_sha=_SHA,
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
    )


# ---------------------------------------------------------------------------
# 1. Field mapping (EvalReport → CachedBaseline) using a mock eval result.
# ---------------------------------------------------------------------------


def test_report_to_baseline_maps_all_fields() -> None:
    report = _fake_report()
    baseline = report_to_baseline(
        report,
        scaffold_commit_sha=_SHA,
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
    )

    # Direct-copy fields.
    assert baseline.evaluated_at == report.evaluated_at
    assert baseline.mode == report.mode
    assert baseline.scorers == list(report.scorers)
    assert baseline.aggregate == report.aggregate
    assert baseline.per_judge == report.per_judge
    assert baseline.per_bucket == report.per_bucket
    assert baseline.scorer_fingerprint == report.scorer_fingerprint
    assert baseline.cost_metrics == report.cost_metrics

    # Fields sourced from the caller (git + config), not the report.
    assert baseline.scaffold_commit_sha == _SHA
    assert baseline.runtime_endpoint == _RUNTIME
    assert baseline.judge_endpoint == _JUDGE

    # The conversion MUST drop the eval-only fields — the cache header
    # only carries what is_compatible() / load_baseline() consume.
    dumped = baseline.to_dict()
    assert set(dumped.keys()) == set(REQUIRED_FIELDS)
    for dropped in ("failures", "experiment_id", "trace_ids", "n_rows", "run_id"):
        assert dropped not in dumped


def test_report_to_baseline_renames_n_rows_and_run_id() -> None:
    """The two schemas intentionally diverge on these two field names."""
    report = _fake_report()
    baseline = report_to_baseline(
        report,
        scaffold_commit_sha=_SHA,
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
    )

    assert baseline.n_examples == report.n_rows
    assert baseline.n_examples != 0  # not the dataclass default
    assert baseline.mlflow_run_id == report.run_id
    assert baseline.mlflow_run_id is not None


def test_report_to_baseline_copies_not_aliases() -> None:
    """Mutating the report's containers must not leak into the baseline."""
    report = _fake_report()
    baseline = report_to_baseline(
        report,
        scaffold_commit_sha=_SHA,
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
    )

    report.scorers.append("safety")
    report.per_judge["correctness"] = 0.0
    report.per_bucket["direct"]["correctness"] = 0.0

    assert "safety" not in baseline.scorers
    assert baseline.per_judge["correctness"] == 0.5
    assert baseline.per_bucket["direct"]["correctness"] == 0.5


# ---------------------------------------------------------------------------
# 2. load_baseline() can load the generated file (the round.py reader).
# ---------------------------------------------------------------------------


def test_generated_baseline_loads_via_load_baseline(tmp_path: Path) -> None:
    baseline = _baseline_from_fake()
    save_baseline(tmp_path, baseline)

    loaded = load_baseline(tmp_path)
    assert loaded is not None
    assert loaded.scaffold_commit_sha == baseline.scaffold_commit_sha
    assert loaded.evaluated_at == baseline.evaluated_at
    assert loaded.mode == baseline.mode
    assert loaded.scorers == baseline.scorers
    assert loaded.runtime_endpoint == baseline.runtime_endpoint
    assert loaded.judge_endpoint == baseline.judge_endpoint
    assert loaded.aggregate == baseline.aggregate
    assert loaded.per_judge == baseline.per_judge
    assert loaded.per_bucket == baseline.per_bucket
    assert loaded.n_examples == baseline.n_examples
    assert loaded.mlflow_run_id == baseline.mlflow_run_id


def test_load_baseline_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_baseline(tmp_path) is None


# ---------------------------------------------------------------------------
# 3. All required CachedBaseline fields are populated.
# ---------------------------------------------------------------------------


def test_generated_baseline_has_all_required_fields() -> None:
    baseline = _baseline_from_fake()
    dumped = baseline.to_dict()

    for field in REQUIRED_FIELDS:
        assert field in dumped, f"missing required field: {field}"

    assert dumped["scaffold_commit_sha"]
    assert dumped["evaluated_at"]
    assert dumped["mode"]
    assert dumped["scorers"]
    assert dumped["runtime_endpoint"]
    assert dumped["judge_endpoint"]
    assert dumped["aggregate"] == pytest.approx(0.75)
    assert dumped["per_judge"]
    assert dumped["per_bucket"]
    assert dumped["n_examples"] == 8
    assert dumped["mlflow_run_id"]
    assert dumped["scorer_fingerprint"]


def test_generated_baseline_json_round_trips(tmp_path: Path) -> None:
    """The on-disk JSON (as make_baseline.py writes it) re-parses cleanly."""
    baseline = _baseline_from_fake()
    path = tmp_path / "eval" / "runs" / "baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline.to_dict(), indent=2) + "\n", encoding="utf-8")

    raw = json.loads(path.read_text(encoding="utf-8"))
    reborn = CachedBaseline.from_dict(raw)
    assert reborn == baseline


# ---------------------------------------------------------------------------
# 4. CLI integration: scripts/make_baseline.py writes a loadable file.
#    evaluate_branch + git are monkeypatched — no LLM, no git.
# ---------------------------------------------------------------------------


_MIN_CONFIG_YAML = """\
runtime_endpoint: databricks-claude-sonnet-4-6
optimizer_endpoint: databricks-claude-opus-4-7
judge_endpoint: databricks-claude-sonnet-4-6
experiments:
  runtime: "/Shared/anvil-runtime"
  eval: "/Shared/anvil-eval"
  optimizer: "/Shared/anvil-optimizer"
"""


def _import_make_baseline():
    """Load ``scripts/make_baseline.py`` as a module without touching sys.path.

    ``scripts/`` is not a package, so the script is loaded directly from
    its file via :mod:`importlib`. Each call returns a fresh module
    object, which keeps per-test ``monkeypatch`` of ``evaluate_branch``
    / ``_git_head_sha`` fully isolated.
    """
    path = REPO_ROOT / "scripts" / "make_baseline.py"
    spec = importlib.util.spec_from_file_location("make_baseline", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs the script's top-level imports
    return mod


def test_make_baseline_cli_writes_loadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_baseline = _import_make_baseline()

    # Minimal harness/config.yaml so _load_endpoints validates via RuntimeYAML.
    (tmp_path / "harness").mkdir(parents=True)
    (tmp_path / "harness" / "config.yaml").write_text(_MIN_CONFIG_YAML, encoding="utf-8")
    (tmp_path / "scaffold").mkdir()

    # Mock the eval (no LLM) and the git sha lookup (no git repo).
    monkeypatch.setattr(make_baseline, "evaluate_branch", lambda **_kw: _fake_report())
    monkeypatch.setattr(make_baseline, "_git_head_sha", lambda _root: _SHA)

    out_path = tmp_path / "eval" / "runs" / "baseline.json"
    rc = make_baseline.main(["--scaffold", str(tmp_path / "scaffold"), "--out", str(out_path)])
    assert rc == 0
    assert out_path.is_file()

    # load_baseline (the round.py reader) must consume the script's output.
    loaded = load_baseline(tmp_path)
    assert loaded is not None
    assert loaded.scaffold_commit_sha == _SHA
    assert loaded.runtime_endpoint == "databricks-claude-sonnet-4-6"
    assert loaded.judge_endpoint == "databricks-claude-sonnet-4-6"
    assert loaded.n_examples == 8
    assert loaded.mlflow_run_id == "abc123def456"
    assert loaded.mode == "quick"
    assert loaded.aggregate == pytest.approx(0.75)


def test_make_baseline_help_lists_options(
    capsys: pytest.CaptureFixture,
) -> None:
    make_baseline = _import_make_baseline()

    with pytest.raises(SystemExit) as exc:
        make_baseline._arg_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--mode" in out
    assert "--out" in out
    assert "--scaffold" in out
    assert "--profile" in out
    assert "--include-safety" in out


# ---------------------------------------------------------------------------
# 5. build_baseline() forwards an explicit runtime_config_path to
#    evaluate_branch(). Without the forwarding the eval runs against the
#    default config while the baseline records endpoints read from a
#    different config — a misleading cache (cross-review of PR #1).
# ---------------------------------------------------------------------------


def test_build_baseline_forwards_explicit_runtime_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_baseline = _import_make_baseline()

    # A custom config at a NON-default location, with endpoints distinct
    # from the default so a silent fall-back would be detectable.
    custom_runtime = "databricks-claude-opus-4-7"
    custom_judge = "databricks-claude-haiku-4-5"
    custom_cfg = tmp_path / "custom" / "config.yaml"
    custom_cfg.parent.mkdir(parents=True)
    custom_cfg.write_text(
        f"runtime_endpoint: {custom_runtime}\n"
        f"optimizer_endpoint: {custom_runtime}\n"
        f"judge_endpoint: {custom_judge}\n"
        "experiments:\n"
        '  runtime: "/Shared/anvil-runtime"\n'
        '  eval: "/Shared/anvil-eval"\n'
        '  optimizer: "/Shared/anvil-optimizer"\n',
        encoding="utf-8",
    )

    # The default config (sibling of scaffold/) carries DIFFERENT endpoints.
    (tmp_path / "harness").mkdir(parents=True)
    (tmp_path / "harness" / "config.yaml").write_text(_MIN_CONFIG_YAML, encoding="utf-8")
    (tmp_path / "scaffold").mkdir()

    captured: dict[str, object] = {}

    def _capture_evaluate(**kwargs: object) -> EvalReport:
        captured.update(kwargs)
        return _fake_report()

    monkeypatch.setattr(make_baseline, "evaluate_branch", _capture_evaluate)
    monkeypatch.setattr(make_baseline, "_git_head_sha", lambda _root: _SHA)

    baseline = make_baseline.build_baseline(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=custom_cfg,
    )

    # The eval must run against the SAME config the baseline read endpoints
    # from — not the default sibling of scaffold/.
    assert "runtime_config_path" in captured
    assert Path(captured["runtime_config_path"]) == custom_cfg  # type: ignore[arg-type]
    # And the recorded endpoints match the custom config end-to-end, proving
    # the cache header and the forwarded eval config are consistent.
    assert baseline.runtime_endpoint == custom_runtime
    assert baseline.judge_endpoint == custom_judge


# ---------------------------------------------------------------------------
# 6. Scorer-config fingerprint — baseline invalidation on config change.
# ---------------------------------------------------------------------------


def test_compute_scorer_fingerprint_is_stable() -> None:
    """The same scorer configs always produce the same fingerprint
    (sorted by name, deterministic JSON)."""
    from anvil.eval.cache import compute_scorer_fingerprint
    from anvil.runtime.models import ScorerConfig

    configs = [
        ScorerConfig(name="retrieval_groundedness"),
        ScorerConfig(name="correctness"),
        ScorerConfig(name="refusal_appropriateness"),
    ]
    fp1 = compute_scorer_fingerprint(configs)
    fp2 = compute_scorer_fingerprint(list(reversed(configs)))
    # Order-independent — the list is sorted by name internally.
    assert fp1 == fp2


def test_compute_scorer_fingerprint_changes_with_weight() -> None:
    """A weight change must invalidate the fingerprint — this is the
    core fix for the comparability hole."""
    from anvil.eval.cache import compute_scorer_fingerprint
    from anvil.runtime.models import ScorerConfig

    uniform = [ScorerConfig(name="correctness", weight=1.0)]
    weighted = [ScorerConfig(name="correctness", weight=0.5)]
    assert compute_scorer_fingerprint(uniform) != compute_scorer_fingerprint(weighted)


def test_compute_scorer_fingerprint_changes_with_check_function() -> None:
    """A check_function swap must invalidate the fingerprint."""
    from anvil.eval.cache import compute_scorer_fingerprint
    from anvil.runtime.models import ScorerConfig

    a = [ScorerConfig(name="exact_match", type="programmatic", check_function="exact_match")]
    b = [ScorerConfig(name="exact_match", type="programmatic", check_function="must_include_check")]
    assert compute_scorer_fingerprint(a) != compute_scorer_fingerprint(b)


def test_compute_scorer_fingerprint_changes_with_type() -> None:
    """A type change (llm → programmatic) must invalidate the fingerprint."""
    from anvil.eval.cache import compute_scorer_fingerprint
    from anvil.runtime.models import ScorerConfig

    llm = [ScorerConfig(name="x", type="llm")]
    prog = [ScorerConfig(name="x", type="programmatic", check_function="exact_match")]
    assert compute_scorer_fingerprint(llm) != compute_scorer_fingerprint(prog)


def test_compute_scorer_fingerprint_same_for_identical_configs() -> None:
    """Two configs with identical specs produce the same fingerprint."""
    from anvil.eval.cache import compute_scorer_fingerprint
    from anvil.runtime.models import ScorerConfig

    a = [ScorerConfig(name="correctness", weight=0.4)]
    b = [ScorerConfig(name="correctness", weight=0.4)]
    assert compute_scorer_fingerprint(a) == compute_scorer_fingerprint(b)


def test_is_compatible_rejects_fingerprint_mismatch() -> None:
    """A baseline cached with uniform weights must NOT be compatible
    with a run using weighted scorers, even if the names match."""
    from anvil.eval.cache import CachedBaseline, is_compatible

    baseline = CachedBaseline(
        scaffold_commit_sha=_SHA,
        evaluated_at="2026-01-01T00:00:00+00:00",
        mode="quick",
        scorers=["correctness"],
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
        aggregate=0.8,
        scorer_fingerprint='[{"name": "correctness", "type": "llm", "weight": 1.0, "check_function": null}]',
    )
    assert not is_compatible(
        baseline,
        mode="quick",
        scorers=["correctness"],
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
        scorer_fingerprint='[{"name": "correctness", "type": "llm", "weight": 0.5, "check_function": null}]',
    )


def test_is_compatible_accepts_matching_fingerprint() -> None:
    from anvil.eval.cache import CachedBaseline, is_compatible

    fp = '[{"name": "correctness", "type": "llm", "weight": 1.0, "check_function": null}]'
    baseline = CachedBaseline(
        scaffold_commit_sha=_SHA,
        evaluated_at="2026-01-01T00:00:00+00:00",
        mode="quick",
        scorers=["correctness"],
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
        aggregate=0.8,
        scorer_fingerprint=fp,
    )
    assert is_compatible(
        baseline,
        mode="quick",
        scorers=["correctness"],
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
        scorer_fingerprint=fp,
    )


def test_is_compatible_skips_fingerprint_when_cached_empty() -> None:
    """A baseline with an empty fingerprint (written before the field
    existed) skips the fingerprint check for backward compat."""
    from anvil.eval.cache import CachedBaseline, is_compatible

    baseline = CachedBaseline(
        scaffold_commit_sha=_SHA,
        evaluated_at="2026-01-01T00:00:00+00:00",
        mode="quick",
        scorers=["correctness"],
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
        aggregate=0.8,
        scorer_fingerprint="",
    )
    assert is_compatible(
        baseline,
        mode="quick",
        scorers=["correctness"],
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
        scorer_fingerprint='[{"name": "correctness", "type": "llm", "weight": 0.5, "check_function": null}]',
    )


def test_is_compatible_skips_fingerprint_when_current_empty() -> None:
    """An empty current fingerprint also skips the check."""
    from anvil.eval.cache import CachedBaseline, is_compatible

    baseline = CachedBaseline(
        scaffold_commit_sha=_SHA,
        evaluated_at="2026-01-01T00:00:00+00:00",
        mode="quick",
        scorers=["correctness"],
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
        aggregate=0.8,
        scorer_fingerprint='[{"name": "correctness", "type": "llm", "weight": 1.0, "check_function": null}]',
    )
    assert is_compatible(
        baseline,
        mode="quick",
        scorers=["correctness"],
        runtime_endpoint=_RUNTIME,
        judge_endpoint=_JUDGE,
        scorer_fingerprint="",
    )


def test_cached_baseline_fingerprint_round_trips(tmp_path: Path) -> None:
    """The fingerprint survives a save → load cycle."""
    from anvil.eval.cache import load_baseline, save_baseline

    fp = '[{"name": "correctness", "type": "llm", "weight": 1.0, "check_function": null}]'
    baseline = _baseline_from_fake()
    baseline = CachedBaseline(
        scaffold_commit_sha=baseline.scaffold_commit_sha,
        evaluated_at=baseline.evaluated_at,
        mode=baseline.mode,
        scorers=baseline.scorers,
        runtime_endpoint=baseline.runtime_endpoint,
        judge_endpoint=baseline.judge_endpoint,
        aggregate=baseline.aggregate,
        per_judge=baseline.per_judge,
        per_bucket=baseline.per_bucket,
        n_examples=baseline.n_examples,
        mlflow_run_id=baseline.mlflow_run_id,
        scorer_fingerprint=fp,
    )
    save_baseline(tmp_path, baseline)
    loaded = load_baseline(tmp_path)
    assert loaded is not None
    assert loaded.scorer_fingerprint == fp


def test_cached_baseline_from_dict_handles_missing_fingerprint() -> None:
    """A baseline JSON written before the fingerprint field existed
    loads with an empty fingerprint (backward compat)."""
    from anvil.eval.cache import CachedBaseline

    raw = {
        "scaffold_commit_sha": _SHA,
        "evaluated_at": "2026-01-01T00:00:00+00:00",
        "mode": "quick",
        "scorers": ["correctness"],
        "runtime_endpoint": _RUNTIME,
        "judge_endpoint": _JUDGE,
        "aggregate": 0.8,
    }
    baseline = CachedBaseline.from_dict(raw)
    assert baseline.scorer_fingerprint == ""
