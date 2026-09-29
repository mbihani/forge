"""Thin wrappers over ``git`` for round orchestration.

The loop is the only plane that knows about git. Everything else
(runtime, eval, optimizer) is git-agnostic. Keeping these wrappers
small + checked makes the loop's intent legible and the failure
modes explicit.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Committer identity injected into every ``git`` invocation. Round-artifact
# commits run in a freshly-cloned /tmp working repo that has NO configured
# ``user.name`` / ``user.email``, so a bare ``git commit`` fails exit 128
# ("unable to auto-detect email address"). We inject a SAFE BOT identity via
# ``git -c ...`` (per-invocation only — global/user git config is never
# mutated), overridable through the environment for a project that wants its
# own automation identity. A ``users.noreply.github.com`` address is a valid,
# non-personal default that GitHub accepts.
_DEFAULT_COMMITTER_NAME = "anvil-bot"
_DEFAULT_COMMITTER_EMAIL = "anvil-bot@users.noreply.github.com"


def _committer_identity_args() -> list[str]:
    """``-c user.name=… -c user.email=…`` for the committer identity.

    Read from ``ANVIL_GIT_COMMITTER_NAME`` / ``ANVIL_GIT_COMMITTER_EMAIL`` when
    set (and non-empty), else the safe bot default. Passed as per-invocation
    ``-c`` overrides so no global/user/repo git config is written.
    """
    name = os.environ.get("ANVIL_GIT_COMMITTER_NAME") or _DEFAULT_COMMITTER_NAME
    email = os.environ.get("ANVIL_GIT_COMMITTER_EMAIL") or _DEFAULT_COMMITTER_EMAIL
    return ["-c", f"user.name={name}", "-c", f"user.email={email}"]


class GitError(RuntimeError):
    """Raised when a git operation returns non-zero."""


@dataclass(frozen=True)
class GitResult:
    stdout: str
    stderr: str
    returncode: int


def _run(repo_root: Path | str, args: list[str], *, check: bool = True) -> GitResult:
    # Inject the committer identity as per-invocation ``-c`` overrides at this
    # single choke point so a freshly-cloned repo with no configured identity
    # can still ``git commit`` (the round-artifact commits). ``-c`` is a
    # top-level option accepted by every subcommand and is a no-op for
    # non-commit calls, so it is safe to prepend unconditionally.
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *_committer_identity_args(), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if check and proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed (exit {proc.returncode}):\n"
            f"  stdout: {proc.stdout.strip()}\n"
            f"  stderr: {proc.stderr.strip()}"
        )
    return GitResult(
        stdout=proc.stdout.strip(),
        stderr=proc.stderr.strip(),
        returncode=proc.returncode,
    )


def current_branch(repo_root: Path | str) -> str:
    return _run(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout


def current_sha(repo_root: Path | str) -> str:
    return _run(repo_root, ["rev-parse", "HEAD"]).stdout


def has_changes(repo_root: Path | str) -> bool:
    """True if there are tracked or untracked changes in the working tree."""
    res = _run(repo_root, ["status", "--porcelain"], check=False)
    return bool(res.stdout)


def has_staged_changes(repo_root: Path | str) -> bool:
    """True if the index has changes staged and ready to commit.

    Unlike :func:`has_changes` (which inspects the *entire* working
    tree — staged and unstaged), this inspects only the index. It stays
    ``False`` when the applier wrote nothing under ``scaffold/``/
    ``agents/`` even if unrelated files elsewhere in the tree are dirty,
    so :func:`commit_all` can decide whether ``git commit`` has anything
    to commit without risking a non-zero exit on an empty index
    (``"no changes added to commit"``).

    Uses ``git diff --cached --quiet``, whose exit code distinguishes
    the three outcomes: ``0`` = no staged changes (index matches HEAD),
    ``1`` = staged changes present, ``>1`` = a real git error. Only the
    legitimate empty-index case (exit ``0`` / ``1``) is mapped to a bool;
    any higher exit is raised as :class:`GitError` so a real git failure
    is not silently masked as "no staged changes" (the risk of the
    previous ``check=False`` + ``bool(stdout)`` form).
    """
    res = _run(repo_root, ["diff", "--cached", "--quiet"], check=False)
    if res.returncode == 0:
        return False  # index matches HEAD — nothing staged
    if res.returncode == 1:
        return True  # staged differences present
    raise GitError(
        f"git diff --cached --quiet failed (exit {res.returncode}):\n"
        f"  stdout: {res.stdout}\n  stderr: {res.stderr}"
    )


def check_clean_worktree() -> None:
    """Raise if the Git working tree containing the current directory is dirty."""
    status = _run(Path.cwd(), ["status", "--porcelain"])
    if status.stdout:
        raise RuntimeError(
            "Working tree is dirty. Commit or stash your changes before running an "
            "optimization round. Uncommitted files:\n"
            f"{status.stdout}"
        )


def create_round_branch(
    repo_root: Path | str, round_id: int, *, parent_branch: str = "anvil/exp"
) -> str:
    """Create ``anvil/exp-round-<N>`` from ``parent_branch`` and check it out.

    Idempotent on a clean tree: if the branch already exists, raises
    ``GitError`` (the loop should not silently reuse a stale branch).
    """
    branch = f"anvil/exp-round-{round_id}"
    _run(repo_root, ["checkout", parent_branch])
    _run(repo_root, ["checkout", "-b", branch])
    return branch


def commit_all(repo_root: Path | str, *, message: str) -> str:
    """Stage ``scaffold/`` (and ``agents/`` when present) and commit.

    Returns the new SHA, or the current SHA when nothing is staged to
    commit (e.g. a ``noop`` action, or content identical to HEAD).
    Checking the *index* (:func:`has_staged_changes`) rather than the
    whole working tree means unrelated dirty files outside
    ``scaffold/``/``agents/`` can no longer trick this into running
    ``git commit`` on an empty index — which exits non-zero
    (``"no changes added to commit"``) and aborts the multi-round run.

    In code mode the optimizer writes agent modules under ``agents/``
    (a repo-root sibling of ``scaffold/``); staging only ``scaffold/``
    would leave those mutations uncommitted and break the round's
    keep/revert branch operations. ``git add`` on a missing pathspec is
    fatal, so ``agents/`` is staged only when the directory exists.
    """
    _run(repo_root, ["add", "scaffold/"])
    if (Path(repo_root) / "agents").is_dir():
        _run(repo_root, ["add", "agents/"])
    if not has_staged_changes(repo_root):
        # Nothing staged to commit (e.g. noop action, or applier wrote
        # content identical to HEAD). Return the current SHA so no
        # caller can crash on an empty ``git commit``.
        return current_sha(repo_root)
    _run(repo_root, ["commit", "-m", message])
    return current_sha(repo_root)


def ff_merge(repo_root: Path | str, *, branch: str, target: str = "anvil/exp") -> None:
    _run(repo_root, ["checkout", target])
    _run(repo_root, ["merge", "--ff-only", branch])


def delete_branch(repo_root: Path | str, *, branch: str, target: str = "anvil/exp") -> None:
    _run(repo_root, ["checkout", target])
    _run(repo_root, ["branch", "-D", branch])
