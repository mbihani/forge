"""Held-out test selection, mocked finalization, and optimization lock tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from anvil.data import load_golden_set, select_subset
from anvil.eval.runner import EvalReport
from anvil.loop.frontier import Frontier

REPO_ROOT = Path(__file__).resolve().parent.parent
SHA = "b" * 40


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report() -> EvalReport:
    return EvalReport(
        aggregate=0.8,
        per_judge={"correctness": 0.7, "groundedness": 0.9},
        per_bucket={"direct": {"correctness": 0.7, "groundedness": 0.9}},
        failures=[],
        run_id="run-test",
        experiment_id="exp-test",
        n_rows=20,
        mode="test",
        scorers=["correctness", "groundedness"],
        evaluated_at="2026-08-16T12:00:00+00:00",
    )


def _write_config(root: Path, *, enabled: bool = True) -> None:
    (root / "harness").mkdir(parents=True)
    (root / "harness" / "config.yaml").write_text(
        "runtime_endpoint: runtime\n"
        "optimizer_endpoint: optimizer\n"
        "judge_endpoint: judge\n"
        "experiments:\n"
        "  root: /Shared/forge\n"
        "eval:\n"
        f"  held_out_test: {str(enabled).lower()}\n"
        "  modes:\n"
        "    test: {rows: 20, buckets: {direct: 6, multi_hop: 6, distractor: 4, out_of_scope: 4}}\n",
        encoding="utf-8",
    )


def test_test_mode_selects_all_golden_rows() -> None:
    examples = load_golden_set(REPO_ROOT / "data" / "golden_set.jsonl")
    selected = select_subset(
        examples,
        buckets={"direct": 6, "multi_hop": 6, "distractor": 4, "out_of_scope": 4},
    )
    assert len(examples) == 20
    assert len(selected) == 20
    assert {row["example_id"] for row in selected} == {row["example_id"] for row in examples}


def test_finalize_cli_writes_expected_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("finalize")
    _write_config(tmp_path)
    (tmp_path / "scaffold").mkdir()
    frontier = Frontier.from_scores({"correctness": 0.7, "aggregate": 0.8})
    calls: dict[str, object] = {}

    def fake_evaluate(**kwargs: object) -> EvalReport:
        calls.update(kwargs)
        return _report()

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "load_frontier", lambda _root: frontier)
    monkeypatch.setattr(module, "evaluate_branch", fake_evaluate)
    monkeypatch.setattr(module, "_git_head_sha", lambda _root: SHA)

    out = tmp_path / "eval" / "runs" / "finalized.json"
    assert module.main(["--scaffold", str(tmp_path / "scaffold")]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["aggregate"] == pytest.approx(0.8)
    assert payload["per_judge"] == _report().per_judge
    assert payload["per_bucket"] == _report().per_bucket
    assert payload["scaffold_commit_sha"] == SHA
    assert payload["finalized_at"]
    assert payload["frontier"] == frontier.to_dict()
    assert calls["mode"] == "test"
    assert calls["allow_test"] is True


def test_finalize_refuses_when_disabled(tmp_path: Path) -> None:
    module = _load_script("finalize")
    _write_config(tmp_path, enabled=False)
    with pytest.raises(RuntimeError, match="held-out finalization is disabled"):
        module.finalize(repo_root=tmp_path, scaffold_root=tmp_path / "scaffold")


def test_run_round_lock_and_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("run_round")
    finalized = tmp_path / "eval" / "runs" / "finalized.json"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    assert module.main([]) == 1

    checked = False

    def check_clean() -> None:
        nonlocal checked
        checked = True

    monkeypatch.setattr(module, "check_clean_worktree", check_clean)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 1})(),
    )
    assert module.main(["--force"]) == 2
    assert checked
