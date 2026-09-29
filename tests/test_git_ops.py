"""Tests for :mod:`anvil.loop.git_ops` — the round-orchestration git wrappers.

Focus: :func:`commit_all` must succeed in a freshly-cloned working repo that
has NO configured ``user.name`` / ``user.email``. The round-artifacts commit
(``anvil.loop.round``) runs in exactly such a /tmp clone, where a bare
``git commit`` fails exit 128 ("unable to auto-detect email address"). The fix
injects a committer identity as per-invocation ``-c`` overrides at the single
``_run`` choke point.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from anvil.loop.git_ops import (
    _DEFAULT_COMMITTER_EMAIL,
    _DEFAULT_COMMITTER_NAME,
    commit_all,
    current_sha,
)


def _raw_git(repo: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``git`` directly (NOT through the wrapper) and return the result."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _isolate_git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ALL ambient committer identity from the environment.

    Points ``GIT_CONFIG_GLOBAL`` / ``GIT_CONFIG_SYSTEM`` at ``/dev/null`` so
    the machine's own ``~/.gitconfig`` identity cannot leak in, and clears the
    ``GIT_AUTHOR_*`` / ``GIT_COMMITTER_*`` and ``ANVIL_GIT_COMMITTER_*`` env
    vars so neither an ambient identity nor a project override is present.
    This reproduces the freshly-cloned /tmp repo the round commits run in.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    for var in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "ANVIL_GIT_COMMITTER_NAME",
        "ANVIL_GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.delenv(var, raising=False)


def _init_repo_no_identity(repo: Path) -> None:
    """A git repo with a HEAD commit but NO stored ``user.*`` identity.

    The seed commit is made with an EPHEMERAL ``-c`` identity (never written
    to any config) so the repo has a HEAD to commit onto while
    ``git config user.email`` stays unset — exactly the state a fresh clone is
    in.
    """
    _raw_git(repo, "init", "-q")
    # Local (repo) config only — disable gpg signing so the seed/commit paths
    # never block on a key. This is NOT an identity.
    _raw_git(repo, "config", "commit.gpgsign", "false")
    # Force git to REQUIRE an explicit configured identity: without this git
    # may auto-detect ``user@hostname`` from GECOS and let a bare commit
    # succeed on some machines, which would defeat the exit-128 bite. With it,
    # a bare commit fails deterministically while ``commit_all``'s per-
    # invocation ``-c`` identity still satisfies the requirement.
    _raw_git(repo, "config", "user.useConfigOnly", "true")
    (repo / "scaffold").mkdir()
    (repo / "scaffold" / "seed.md").write_text("seed\n", encoding="utf-8")
    _raw_git(repo, "add", ".")
    seed = _raw_git(
        repo,
        "-c",
        "user.name=Seed",
        "-c",
        "user.email=seed@example.com",
        "commit",
        "-q",
        "-m",
        "init",
    )
    assert seed.returncode == 0, seed.stderr


def test_commit_all_succeeds_without_configured_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """commit_all commits in a repo with NO configured identity.

    REGRESSION BITE: the test first proves a BARE ``git commit`` in this same
    repo + environment fails exit 128 (so the environment genuinely lacks any
    committer identity). ``commit_all``'s ONLY difference from that bare commit
    is the injected identity — so it succeeding proves the injection is
    load-bearing. Reverting the identity injection makes ``commit_all`` take
    the same exit-128 path and fail this test.
    """
    _isolate_git_identity(monkeypatch)
    repo = tmp_path / "clone"
    repo.mkdir()
    _init_repo_no_identity(repo)

    # A new scaffold change to commit.
    (repo / "scaffold" / "change.md").write_text("round artifact\n", encoding="utf-8")

    # --- bite: a bare `git commit` (no identity) fails exit 128 ---
    stage = _raw_git(repo, "add", "scaffold/")
    assert stage.returncode == 0, stage.stderr
    bare = _raw_git(repo, "commit", "-m", "would fail")
    assert bare.returncode == 128, (
        "expected exit 128 (no identity) but bare git commit succeeded — the "
        "test environment is not actually identity-free, so it cannot prove "
        f"the injection bites. stderr:\n{bare.stderr}"
    )
    assert "empty ident" in bare.stderr.lower() or "user.email" in bare.stderr.lower()

    # --- commit_all injects the identity and succeeds ---
    sha = commit_all(repo, message="round artifacts")
    assert sha == current_sha(repo)
    # A NEW commit was actually created (HEAD advanced past the seed).
    assert sha != _raw_git(repo, "rev-parse", "HEAD~1").stdout.strip()


def test_commit_all_uses_default_bot_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no env override, the injected committer is the safe bot default."""
    _isolate_git_identity(monkeypatch)
    repo = tmp_path / "clone"
    repo.mkdir()
    _init_repo_no_identity(repo)

    (repo / "scaffold" / "change.md").write_text("round artifact\n", encoding="utf-8")
    commit_all(repo, message="round artifacts")

    name = _raw_git(repo, "log", "-1", "--format=%cn").stdout.strip()
    email = _raw_git(repo, "log", "-1", "--format=%ce").stdout.strip()
    assert name == _DEFAULT_COMMITTER_NAME
    assert email == _DEFAULT_COMMITTER_EMAIL


def test_commit_all_honors_env_override_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``ANVIL_GIT_COMMITTER_*`` override is used for the commit identity."""
    _isolate_git_identity(monkeypatch)
    monkeypatch.setenv("ANVIL_GIT_COMMITTER_NAME", "custom-bot")
    monkeypatch.setenv("ANVIL_GIT_COMMITTER_EMAIL", "custom@bot.example")
    repo = tmp_path / "clone"
    repo.mkdir()
    _init_repo_no_identity(repo)

    (repo / "scaffold" / "change.md").write_text("round artifact\n", encoding="utf-8")
    commit_all(repo, message="round artifacts")

    name = _raw_git(repo, "log", "-1", "--format=%cn").stdout.strip()
    email = _raw_git(repo, "log", "-1", "--format=%ce").stdout.strip()
    assert name == "custom-bot"
    assert email == "custom@bot.example"


def test_commit_all_noop_returns_current_sha_without_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty index still short-circuits to the current SHA (no commit run),
    even with no configured identity — the noop path never shells a commit."""
    _isolate_git_identity(monkeypatch)
    repo = tmp_path / "clone"
    repo.mkdir()
    _init_repo_no_identity(repo)

    before = current_sha(repo)
    # Nothing new staged under scaffold/ → commit_all must return HEAD, not
    # attempt (and fail) an empty commit.
    sha = commit_all(repo, message="nothing to do")
    assert sha == before
