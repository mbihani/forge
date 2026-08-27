"""run_round must tolerate a no-op / no-change mutation without crashing.

Regression: when the optimizer produced a PARSE-FAILURE noop
(``parse_status=no_block`` — the transcript had no ``json-action``
fenced block), the applier writes no files. ``run_round`` then called
``commit_all`` on an empty index. The old ``commit_all`` guard
(:func:`has_changes`) inspected the *entire* working tree, so any
unrelated dirty/untracked file outside ``scaffold/``/``agents/`` (e.g.
``eval/runs/round_N.json`` / ``data/round_N.json`` left over from a
previous round) made it return ``True`` while nothing was staged.
``git commit`` then exited 1 ("no changes added to commit") and raised
``GitError``, aborting the whole multi-round ``--rounds N`` run.

The fix has two layers:

* :func:`anvil.loop.round.run_round` skips the step-4 commit entirely
  when the applier's ``ApplyResult`` reports no files touched, and
  records the parent SHA instead.
* :func:`anvil.loop.git_ops.commit_all` checks the *index*
  (:func:`has_staged_changes`) rather than the whole tree, so it can no
  longer run ``git commit`` on an empty index.

These tests reproduce the live crash condition (parse-fail noop + a
working tree dirtied by unrelated files) and assert neither layer
crashes and that a ``--rounds 2`` loop proceeds past the noop.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from anvil.eval.cache import CachedBaseline, save_baseline
from anvil.loop.decision import Decision
from anvil.loop.git_ops import commit_all, current_sha
from anvil.loop.round import run_round
from anvil.optimizer.actions import NoopAction
from anvil.optimizer.parser import ParseResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    """A minimal ANVIL repo: scaffold/ + harness/config.yaml + baseline.

    Creates ``anvil/exp`` (the parent branch) off the initial commit and
    leaves the working tree checked out on it. ``run_round`` forks
    ``anvil/exp-round-<N>`` from here.
    """
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")

    # scaffold/ — tracked so HEAD knows about it (commit_all stages it).
    (repo / "scaffold").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text("tools: []\n", encoding="utf-8")

    # harness/config.yaml — read by _read_optimization_mode / load_gate_config.
    # mode: prompt so a NoopAction passes mode validation; no `gate` section
    # so GateConfig falls back to defaults (frontier gate; noop short-circuits
    # it before any frontier I/O).
    (repo / "harness").mkdir()
    (repo / "harness" / "config.yaml").write_text("mode: prompt\n", encoding="utf-8")

    # Cached baseline — load_baseline reads eval/runs/baseline.json.
    (repo / "eval" / "runs").mkdir(parents=True)
    save_baseline(
        repo,
        CachedBaseline(
            scaffold_commit_sha="a" * 40,
            evaluated_at="2026-08-16T12:00:00+00:00",
            mode="test",
            scorers=["correctness"],
            runtime_endpoint="runtime",
            judge_endpoint="judge",
            aggregate=0.5,
            per_judge={"correctness": 0.5},
            per_bucket={"direct": {"correctness": 0.5}},
            n_examples=10,
        ),
    )

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "checkout", "-b", "anvil/exp")
    return repo


def _parse_fail_noop() -> tuple[NoopAction, str, ParseResult]:
    """A parse-failure noop: no json-action block in the transcript."""
    action = NoopAction(rationale="parser: no `json-action` fenced block in transcript")
    parse_result = ParseResult(action=action, parse_status="no_block", n_blocks_found=0)
    return action, "(no json-action block)\n", parse_result


def _patch_noop_session(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Monkeypatch run_optimizer_session to always return a parse-fail noop.

    Returns a mutable call counter so a test can assert how many rounds
    actually executed (the loop must proceed past the first noop).
    """
    import anvil.loop.round as round_mod

    calls: list[int] = []

    async def _fake_session(**_kwargs: object):  # noqa: ANN003
        calls.append(1)
        return _parse_fail_noop()

    monkeypatch.setattr(round_mod, "run_optimizer_session", _fake_session)
    return calls


# ---------------------------------------------------------------------------
# round.py — run_round tolerates a parse-fail noop
# ---------------------------------------------------------------------------


def test_run_round_parse_fail_noop_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parse-fail noop must not raise GitError, even when the working
    tree is dirtied by unrelated files (the live crash condition: by
    round N, eval/runs/ and data/ accumulate untracked artifacts).

    The round records decision=NOOP and makes no scaffold commit — the
    commit SHA equals the parent (no empty-index commit was attempted).
    """
    repo = _init_repo(tmp_path)
    _patch_noop_session(monkeypatch)

    # Reproduce the live condition: an untracked file outside scaffold/
    # (e.g. a leftover round artifact). The OLD commit_all guard
    # (has_changes on the whole tree) returned True here while nothing
    # was staged → `git commit` exited 1 → GitError.
    (repo / "eval" / "runs" / "round_000.json").write_text("{}", encoding="utf-8")

    parent_sha = _git(repo, "rev-parse", "anvil/exp")

    report = run_round(
        round_id=1,
        repo_root=repo,
        parent_branch="anvil/exp",
        max_turns=1,
    )

    assert report.decision is Decision.NOOP
    assert report.parse_status == "no_block"
    assert report.action_kind == "noop"
    # No scaffold mutation commit was made for the noop: the recorded
    # SHA is the parent (anvil/exp) HEAD, not a new commit.
    assert report.git_commit_sha == parent_sha
    # The branch was cleaned up (NOOP → delete_branch).
    branches = _git(repo, "branch", "--list")
    assert "anvil/exp-round-1" not in branches


def test_run_round_loop_continues_past_parse_fail_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--rounds 2` loop must CONTINUE past a parse-fail noop to the
    next round rather than aborting. Round 1's artifacts
    (eval/runs/round_001.json, data/round_001.json, eval/mutations.jsonl)
    are untracked and persist into round 2's working tree — i.e. round 2
    runs under exactly the dirty-tree condition that crashed the old
    code. Both rounds must complete without raising.
    """
    repo = _init_repo(tmp_path)
    calls = _patch_noop_session(monkeypatch)

    reports = []
    for rid in (1, 2):
        report = run_round(
            round_id=rid,
            repo_root=repo,
            parent_branch="anvil/exp",
            max_turns=1,
        )
        reports.append(report)

    # Both rounds executed (the loop did not abort after round 1).
    assert len(calls) == 2
    assert all(r.decision is Decision.NOOP for r in reports)
    assert all(r.parse_status == "no_block" for r in reports)
    # No round branch lingers.
    assert "anvil/exp-round-1" not in _git(repo, "branch", "--list")
    assert "anvil/exp-round-2" not in _git(repo, "branch", "--list")


def test_run_round_clean_noop_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A VALID noop (optimizer emits a proper `noop` action) on a clean
    tree must behave exactly as before: decision=NOOP, no crash, parent
    SHA recorded. Guards the 'keep clean-noop behavior unchanged' rule.
    """
    repo = _init_repo(tmp_path)

    import anvil.loop.round as round_mod

    action = NoopAction(rationale="nothing worth mutating this round")
    parse_result = ParseResult(action=action, parse_status="ok")

    async def _fake_session(**_kwargs: object):  # noqa: ANN003
        return (action, "(clean noop)\n", parse_result)

    monkeypatch.setattr(round_mod, "run_optimizer_session", _fake_session)

    parent_sha = _git(repo, "rev-parse", "anvil/exp")
    report = run_round(
        round_id=1,
        repo_root=repo,
        parent_branch="anvil/exp",
        max_turns=1,
    )

    assert report.decision is Decision.NOOP
    assert report.parse_status == "ok"
    assert report.git_commit_sha == parent_sha


# ---------------------------------------------------------------------------
# git_ops.py — commit_all safe against an empty index
# ---------------------------------------------------------------------------


def test_commit_all_no_raise_when_tree_dirty_but_nothing_staged(
    tmp_path: Path,
) -> None:
    """commit_all must NOT raise when the working tree is dirty outside
    scaffold/agents but nothing is staged. This is the direct root
    cause of the live crash: the old has_changes guard saw the dirty
    tree, ran `git commit` on an empty index, and exited 1.
    """
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "scaffold").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text("tools: []\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")

    head_before = current_sha(repo)

    # Dirty the tree with an UNTRACKED file outside scaffold/agents —
    # mimicking leftover eval/runs/round_N.json artifacts. Nothing is
    # staged under scaffold/.
    (repo / "eval" / "runs").mkdir(parents=True)
    (repo / "eval" / "runs" / "round_010.json").write_text("{}", encoding="utf-8")

    # Must not raise GitError; returns the unchanged HEAD.
    sha = commit_all(repo, message="round 010: noop")
    assert sha == head_before


def test_commit_all_commits_when_scaffold_actually_staged(tmp_path: Path) -> None:
    """commit_all still commits a real scaffold mutation (the fix must
    not change behavior when there IS something staged). Regression
    guard for the has_staged_changes switch.
    """
    repo = tmp_path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "scaffold").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text("tools: []\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")

    head_before = current_sha(repo)

    # A real mutation: edit scaffold/harness.yaml.
    (repo / "scaffold" / "harness.yaml").write_text("tools: [t]\n", encoding="utf-8")
    sha = commit_all(repo, message="round 001: edit scaffold/harness.yaml")

    assert sha != head_before
    changed = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
    assert "scaffold/harness.yaml" in changed
