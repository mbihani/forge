"""Run-control and trust features added after the savesage v3 runs.

* canary: a mutation whose first rows all fail is reverted without the full eval;
* every round (kept or reverted) is recorded on the parent branch, with the
  round's code change saved as a patch;
* disputed golden labels are collected from any action and shown to later rounds;
* the verified-settings probe file, and the round prompt's extra sections.

Offline: a throwaway git repo, fake optimizer sessions and a fake evaluate_branch.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from anvil.eval.cache import CachedBaseline, save_baseline
from anvil.eval.label_disputes import (
    append_label_disputes,
    load_label_disputes,
    render_disputes_md,
    render_disputes_report_md,
)
from anvil.eval.lever_probe import (
    ProbeResult,
    load_lever_probe,
    render_probe_md,
    setting_rejection,
    write_lever_probe,
)
from anvil.eval.runner import EvalReport
from anvil.loop.decision import Decision
from anvil.loop.round import run_round
from anvil.optimizer.actions import AddSkillAction, NoopAction
from anvil.optimizer.parser import ParseResult, parse_action

LUNA = "databricks-gpt-5-6-luna"
GLM = "databricks-glm-5-3-flash"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _init_repo(tmp_path: Path, config: str = "mode: prompt\n") -> Path:
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "scaffold" / "skills").mkdir(parents=True)
    (repo / "scaffold" / "skills" / "identity.md").write_text("# identity\n", encoding="utf-8")
    (repo / "scaffold" / "harness.yaml").write_text(
        "skills:\n  - file: identity.md\n", encoding="utf-8"
    )
    (repo / "harness").mkdir()
    (repo / "harness" / "config.yaml").write_text(config, encoding="utf-8")
    (repo / "eval" / "runs").mkdir(parents=True)
    save_baseline(
        repo,
        CachedBaseline(
            scaffold_commit_sha="a" * 40,
            evaluated_at="2026-10-06T00:00:00+00:00",
            mode="quick",
            scorers=["correctness"],
            runtime_endpoint="runtime",
            judge_endpoint="judge",
            aggregate=0.5,
            per_judge={"correctness": 0.5},
            per_bucket={"direct": {"correctness": 0.5}},
            n_examples=4,
        ),
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "checkout", "-q", "-b", "anvil/exp")
    return repo


def _session(monkeypatch: pytest.MonkeyPatch, action) -> None:
    import anvil.loop.round as round_mod

    async def _fake(**_kw):  # noqa: ANN003
        return action, "(transcript)\n", ParseResult(action=action, parse_status="ok")

    monkeypatch.setattr(round_mod, "run_optimizer_session", _fake)


def _report(*, aggregate: float, n: int, errors: int) -> EvalReport:
    return EvalReport(
        aggregate=aggregate,
        per_judge={"correctness": aggregate},
        per_bucket={"direct": {"correctness": aggregate}},
        failures=[{"example_id": f"e{i}", "error": "HTTP 400: bad param"} for i in range(errors)],
        run_id="",
        experiment_id="",
        n_rows=n,
        mode="quick",
        scorers=["correctness"],
        evaluated_at="2026-10-06T00:00:00+00:00",
        trace_ids=[],
        cost_metrics={"n_rows": float(n)},
        scorer_fingerprint="",
    )


def _fake_eval(monkeypatch: pytest.MonkeyPatch, *, canary_errors: int) -> list[int | None]:
    import anvil.loop.round as round_mod

    calls: list[int | None] = []

    def _evaluate(**kw):  # noqa: ANN003
        calls.append(kw.get("max_rows"))
        if kw.get("max_rows"):
            return _report(aggregate=0.0, n=kw["max_rows"], errors=canary_errors)
        return _report(aggregate=0.6, n=4, errors=0)

    monkeypatch.setattr(round_mod, "evaluate_branch", _evaluate)
    return calls


_SKILL = AddSkillAction(target_file="skills/extra.md", content="# extra\n", rationale="try it")


# ------------------------------------------------------------------ canary


def test_canary_reverts_when_every_row_fails_and_skips_the_full_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, "mode: prompt\nloop:\n  canary_rows: 2\n")
    _session(monkeypatch, _SKILL)
    calls = _fake_eval(monkeypatch, canary_errors=2)

    report = run_round(round_id=1, repo_root=repo, parent_branch="anvil/exp", max_turns=1)

    assert report.decision is Decision.REVERT
    assert calls == [2]  # the canary only; the full eval never ran
    rec = json.loads((repo / "eval" / "runs" / "round_001.json").read_text())
    assert rec["notes"].startswith("canary: all 2 rows failed") and "HTTP 400" in rec["notes"]


def test_canary_passes_when_any_row_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, "mode: prompt\nloop:\n  canary_rows: 2\n")
    _session(monkeypatch, _SKILL)
    calls = _fake_eval(monkeypatch, canary_errors=1)

    report = run_round(round_id=1, repo_root=repo, parent_branch="anvil/exp", max_turns=1)

    assert calls == [2, None]  # canary, then the full eval
    assert report.decision is Decision.KEEP


def test_no_canary_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    _session(monkeypatch, _SKILL)
    calls = _fake_eval(monkeypatch, canary_errors=0)
    run_round(round_id=1, repo_root=repo, parent_branch="anvil/exp", max_turns=1)
    assert calls == [None]


# ------------------------------------------------------------------ records on the parent


def test_reverted_round_is_recorded_on_the_parent_with_its_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path, "mode: prompt\nloop:\n  canary_rows: 2\n")
    _session(monkeypatch, _SKILL)
    _fake_eval(monkeypatch, canary_errors=2)

    run_round(round_id=1, repo_root=repo, parent_branch="anvil/exp", max_turns=1)

    assert _git(repo, "branch", "--show-current") == "anvil/exp"
    assert "round 001: record (revert)" in _git(repo, "log", "--format=%s", "-1")
    tracked = _git(repo, "ls-files", "eval")
    assert "eval/runs/round_001.json" in tracked and "eval/runs/round_001.patch" in tracked
    patch = (repo / "eval" / "runs" / "round_001.patch").read_text()
    assert "+# extra" in patch  # the reverted code change survives the branch delete
    assert not (repo / "scaffold" / "skills" / "extra.md").exists()  # but is not applied
    assert _git(repo, "status", "--porcelain", "--", "eval") == ""


def test_kept_round_is_recorded_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    _session(monkeypatch, _SKILL)
    _fake_eval(monkeypatch, canary_errors=0)

    report = run_round(round_id=1, repo_root=repo, parent_branch="anvil/exp", max_turns=1)

    assert report.decision is Decision.KEEP
    assert "round 001: record (keep)" in _git(repo, "log", "--format=%s", "-1")
    assert (repo / "scaffold" / "skills" / "extra.md").exists()
    assert _git(repo, "status", "--porcelain", "--", "eval") == ""


# ------------------------------------------------------------------ disputed labels


def _dispute_noop() -> NoopAction:
    body = {
        "action": "noop",
        "rationale": "nothing clears a margin",
        "label_disputes": [
            {"example_id": "e7", "field": "cards_cardMeta_network", "reason": "rules forbid it"}
        ],
    }
    return parse_action(f"```json-action\n{json.dumps(body)}\n```").action


def test_disputes_on_a_noop_are_recorded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    _session(monkeypatch, _dispute_noop())

    for rid in (1, 2):
        run_round(round_id=rid, repo_root=repo, parent_branch="anvil/exp", max_turns=1)

    disputes = load_label_disputes(repo)
    assert [(d["example_id"], d["field"], d["round_id"]) for d in disputes] == [
        ("e7", "cards_cardMeta_network", 1)
    ]
    assert "eval/label_disputes.jsonl" in _git(repo, "ls-files", "eval")


def test_dispute_rendering(tmp_path: Path) -> None:
    assert "label_disputes" in render_disputes_md([])  # explains how to report
    append_label_disputes(
        tmp_path,
        3,
        [
            {"example_id": "a", "field": "network", "reason": "contradicts rule"},
            {"example_id": "b", "field": "network", "reason": "same"},
        ],
    )
    md = render_disputes_md(load_label_disputes(tmp_path))
    assert "already reported (2)" in md and "`network` (2): a, b" in md
    report = render_disputes_report_md(load_label_disputes(tmp_path))
    assert "| `a` | 3 | contradicts rule |" in report


def test_actions_accept_disputes_and_cap_the_reason() -> None:
    action = NoopAction(
        rationale="r",
        label_disputes=[{"example_id": "x", "field": "f", "reason": "y" * 400}],
    )
    assert len(action.label_disputes[0].reason) == 300
    assert NoopAction(rationale="r").label_disputes == []


# ------------------------------------------------------------------ lever probe


def _probe(tmp_path: Path) -> dict:
    write_lever_probe(
        tmp_path,
        [
            ProbeResult(LUNA, {"input_mode": "pdf", "reasoning_effort": "low"}, ok=True),
            ProbeResult(
                LUNA,
                {"input_mode": "pdf", "reasoning_effort": "minimal"},
                ok=False,
                error="HTTP 400: 'reasoning_effort' does not support 'minimal'",
            ),
            ProbeResult(
                GLM,
                {"input_mode": "pdf", "reasoning_effort": "low"},
                ok=False,
                error="HTTP 400: unsupported content item type: file",
            ),
            ProbeResult(GLM, {"input_mode": "text", "reasoning_effort": "low"}, ok=True),
        ],
        engine="savesage",
    )
    return load_lever_probe(tmp_path)


def test_setting_rejection_only_for_combinations_seen_failing(tmp_path: Path) -> None:
    probe = _probe(tmp_path)
    assert "minimal" in setting_rejection(probe, LUNA, input_mode="pdf", reasoning_effort="minimal")
    assert setting_rejection(probe, LUNA, input_mode="pdf", reasoning_effort="low") is None
    # Never probed: unknown, so allowed.
    assert setting_rejection(probe, LUNA, input_mode="pdf", reasoning_effort="xhigh") is None
    # Every pdf attempt on GLM failed: the input mode itself is rejected.
    assert "content item" in setting_rejection(probe, GLM, input_mode="pdf")
    assert setting_rejection(None, LUNA, input_mode="pdf") is None


def test_probe_renders_accepted_and_rejected(tmp_path: Path) -> None:
    md = render_probe_md(_probe(tmp_path))
    assert "## Verified runtime settings" in md
    assert "accepted: input_mode=pdf, reasoning_effort=low" in md
    assert "rejected: input_mode=pdf, reasoning_effort=minimal" in md


def test_savesage_agent_refuses_a_probe_rejected_combination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anvil.domains.savesage.extractor as extractor_mod
    from anvil.domains.savesage.agent_base import SavesageAgent

    built: list[dict] = []
    monkeypatch.setattr(extractor_mod, "SavesageIciciExtractor", lambda *_a, **kw: built.append(kw))

    class _A(SavesageAgent):
        def predict(self, *, sid, pdf_path):
            return self.extract(sid=sid, pdf_path=pdf_path, reasoning_effort="minimal")

    agent = _A("p", allowed_models=[LUNA, GLM], lever_probe=_probe(tmp_path))
    with pytest.raises(ValueError, match="does not support 'minimal'"):
        agent.predict(sid="s", pdf_path="x.pdf")
    with pytest.raises(ValueError, match="content item"):
        agent.extract(sid="s", pdf_path="x.pdf", model=GLM, input_mode="pdf")
    assert built == []  # refused before any endpoint call


def test_savesage_probe_classifies_endpoint_answers() -> None:
    from anvil.domains.savesage.probe import _classify

    assert _classify("ExtractionError: did not finish cleanly: finish_reason='length'") is True
    assert _classify("ExtractionError: HTTP 400: 'reasoning_effort' does not support") is False
    assert _classify("ExtractionError: after 2 attempts: TimeoutError: timed out") is None


# ------------------------------------------------------------------ round prompt


def test_round_prompt_carries_probe_policy_notes_and_disputes(tmp_path: Path) -> None:
    from anvil.loop.builder import build_round_prompt
    from anvil.runtime.models import CallPolicyConfig

    _probe(tmp_path)
    append_label_disputes(tmp_path, 1, [{"example_id": "e1", "field": "network", "reason": "r"}])
    prompt = build_round_prompt(
        repo_root=tmp_path,
        round_id=2,
        baseline=None,
        engine="savesage",
        call_policy=CallPolicyConfig(max_sends_per_document=1),
    )
    assert "## Verified runtime settings" in prompt
    assert "## Call rules (enforced by the eval)" in prompt and "per row" in prompt
    assert "## Engine notes: savesage" in prompt
    assert "## Disputed labels already reported (1)" in prompt


def test_round_prompt_without_extras_is_unchanged_in_shape(tmp_path: Path) -> None:
    from anvil.loop.builder import build_round_prompt

    prompt = build_round_prompt(repo_root=tmp_path, round_id=1, baseline=None)
    assert "Call rules" not in prompt and "Verified runtime settings" not in prompt
    assert "## Disputed labels" in prompt  # how-to-report hint is always there


def test_canary_outage_is_an_infra_failure_not_a_revert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anvil.loop.round as round_mod

    repo = _init_repo(tmp_path, "mode: prompt\nloop:\n  canary_rows: 2\n")
    _session(monkeypatch, _SKILL)
    report = _report(aggregate=0.0, n=2, errors=0)
    report.failures = [
        {"example_id": f"e{i}", "error": "ExtractionError: HTTP 503: unavailable"} for i in range(2)
    ]
    monkeypatch.setattr(round_mod, "evaluate_branch", lambda **_kw: report)

    result = run_round(round_id=1, repo_root=repo, parent_branch="anvil/exp", max_turns=1)
    assert result.decision is Decision.INFRA_FAIL
