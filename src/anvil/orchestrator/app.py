"""Forge Orchestrator FastAPI app — two-repo workflow orchestrator.

Bridges TWO separate GitHub repos:

1. **Forge** (deployed as this app) — provides the ``anvil`` engine on
   ``PYTHONPATH=src``. The eval runner, loop, optimizer, etc. live here.
2. **Custom agent repo** (user provides a GitHub URL) — provides
   ``scaffold/``, ``data/``, ``harness/``, ``agents/``. The orchestrator
   clones it to a temp dir, validates it, and runs the ANVIL engine
   against it.

Workflow::

    user inputs repo URL
      → POST /api/session clones + validates
      → if valid: GET / shows editable optimization params (pre-populated
        from harness/config.yaml)
      → POST /api/session/{id}/optimize runs baseline + rounds in a
        background asyncio task
      → POST /api/session/{id}/finalize runs held-out eval, locks the run

Session state machine::

    cloning → validating → validated (ready for optimize)
                           → invalid (show remediation)
    validated → [POST /optimize] → building_baseline → optimizing → optimized
    optimized → [POST /finalize] → finalizing → finalized
                                     ↘ optimized (on failure, retry)
    any state → error (on failure; finalized is terminal on disk)

Sessions live in an in-memory ``_sessions`` dict guarded by a single
``_session_lock`` (state mutations only). Each session owns one
background optimization task; concurrent sessions are fine but a single
session can only optimize once at a time. Finalization is gated by an
atomic ``optimized → finalizing`` transition under the lock plus a
disk check for ``finalized.json`` so concurrent finalize calls cannot
both proceed.

ALL blocking calls (git clone/branch, ``evaluate_branch``,
``run_round``, file I/O) run via ``anyio.to_thread.run_sync`` so the
single uvicorn event loop stays responsive — calling them directly in
an async handler freezes the loop and the Databricks App 502s.

Security: ``GET /api/session/{id}/config`` and the session response's
``config`` field redact any key containing ``token``, ``secret``,
``password``, or ``credential`` (case-insensitive) via
:func:`_redact_secrets`.

The dashboard (``GET /``) is XSS-safe: dynamic values are inserted
with ``textContent`` / ``createElement`` (never ``innerHTML``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import traceback
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

import anyio
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Heavy imports — wrapped so the app still starts when optional deps
# (mlflow, openai, …) are missing on serverless compute. On failure each
# name is set to None and the error is captured in ``_STARTUP_ERROR``;
# endpoints that need these imports return 503 with the error message.
_STARTUP_ERROR: Exception | None = None
try:
    from anvil.data.mlflow_baseline import build_mlflow_baseline
    from anvil.eval import evaluate_branch
    from anvil.eval.cache import report_to_baseline, save_baseline
    from anvil.loop.frontier import load_frontier
    from anvil.loop.round import run_round
    from anvil.orchestrator.conversion import (
        DEFAULT_TARGET_BRANCH,
        ConversionResult,
        _run_conversion_task,
    )
except Exception as exc:  # noqa: BLE001 — capture any import failure
    _STARTUP_ERROR = exc
    build_mlflow_baseline = None  # type: ignore[assignment]
    evaluate_branch = None  # type: ignore[assignment]
    report_to_baseline = None  # type: ignore[assignment]
    save_baseline = None  # type: ignore[assignment]
    load_frontier = None  # type: ignore[assignment]
    run_round = None  # type: ignore[assignment]
    DEFAULT_TARGET_BRANCH = None  # type: ignore[assignment]
    ConversionResult = None  # type: ignore[assignment]
    _run_conversion_task = None  # type: ignore[assignment]

logger = logging.getLogger("anvil.orchestrator")

# Omnigent server connection for the auto-conversion feature. Read at import
# time so the convert endpoint can return 503 early when the agent server is
# not configured. ``OMNIGENT_AUTH_TOKEN`` is optional (some deployments run the
# server without auth); ``OMNIGENT_SERVER_URL`` is required to run a conversion.
OMNIGENT_SERVER_URL = os.getenv("OMNIGENT_SERVER_URL")
OMNIGENT_AUTH_TOKEN = os.getenv("OMNIGENT_AUTH_TOKEN")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Substrings that mark a config field as secret — any key containing one of
# these (case-insensitive) is redacted by ``_redact_secrets`` before the
# config JSON is returned to the client.
_REDACT_KEYWORDS = frozenset({"token", "secret", "password", "credential"})

# Parent branch the ANVIL loop forks round branches off.
_PARENT_BRANCH = "anvil/exp"

# Valid decision values, for sanitizing persisted round JSON.
_VALID_DECISIONS = frozenset({"keep", "revert", "noop", "infra_fail"})

# Session storage root.
_SESSIONS_ROOT = Path("/tmp/forge-sessions")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    repo_url: str
    github_token: str | None = None


class OptimizeRequest(BaseModel):
    eval_mode: str | None = None
    max_rounds: int = Field(default=10, ge=1)
    max_turns: int = Field(default=30, ge=1)
    mlflow_experiment_id: str | None = None
    # Optimization mode: "prompt" (mutate scaffold skills/rules/sampling in
    # markdown + YAML) or "code" (mutate agent Python code). When omitted the
    # session's harness/config.yaml ``mode`` is used (defaulting to "prompt").
    mode: Literal["prompt", "code"] | None = None


class ConvertRequest(BaseModel):
    # Target branch the converter agent creates + pushes. Defaults to
    # "forge-compat" when omitted.
    target_branch: str | None = None


class CheckResult(BaseModel):
    name: str
    status: str  # pass/fail/warn
    message: str
    remediation: str | None = None


class ValidationReport(BaseModel):
    status: str  # valid/invalid
    checks: list[CheckResult]
    # True when the repo failed validation but has a recognizable savesage-style
    # alternative structure (prompts/ + schema/ + harness/ + skills/) that the
    # auto-converter can transform into the forge-compatible layout. Gates the
    # "Convert to forge-compatible" button in the UI.
    convertible: bool = False


class ConversionStatus(BaseModel):
    """Pollable conversion state (GET /api/session/{id}/convert)."""

    status: str  # pending, running, completed, failed
    progress: list[dict[str, Any]]
    pr_url: str | None = None
    branch_name: str | None = None
    revalidation: dict[str, Any] | None = None
    error: str | None = None
    # Managed Omnigent conversation session — kept alive after conversion so
    # the transcript is inspectable. Surfaced as a link in the UI.
    session_id: str | None = None
    session_url: str | None = None


class RoundSummary(BaseModel):
    round_id: int
    decision: str
    action_kind: str | None = None
    baseline_score: float | None = None
    score_delta: float | None = None
    aggregate: float | None = None


class SessionResponse(BaseModel):
    session_id: str
    status: str
    repo_url: str
    agent_subpath: str | None = None
    validation: ValidationReport
    config: dict | None = None
    baseline: dict | None = None
    rounds: list[RoundSummary] = Field(default_factory=list)
    frontier: dict | None = None
    finalized: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Session data model
# ---------------------------------------------------------------------------


@dataclass
class SessionData:
    session_id: str
    repo_url: str
    repo_path: Path
    status: str  # state machine: cloning/validating/validated/invalid/...
    validation: dict  # ValidationReport as dict
    config: dict | None  # harness/config.yaml as dict (secrets redacted)
    baseline: dict | None
    rounds: list[dict]
    frontier: dict | None
    finalized: dict | None
    error: str | None
    agent_subpath: str | None = None  # subdirectory within the cloned repo
    _clone_root: Path | None = field(default=None, repr=False)  # full clone path for cleanup
    _optimize_task: Any = field(default=None, repr=False)  # asyncio.Task | None
    # Auto-conversion state. ``conversion`` is the pollable result; ``_findings``
    # is the alternative-structure scan from validation (fed to the converter
    # prompt); ``_github_token`` is the user's token from Step 1, kept in memory
    # only so the converter agent can clone+push a private repo (never returned
    # in any API response — ``_session_to_response`` does not serialize it).
    conversion: ConversionResult | None = None
    _findings: dict[str, list[str]] | None = field(default=None, repr=False)
    _github_token: str | None = field(default=None, repr=False)
    _convert_task: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------


def _redact_secrets(obj: Any) -> Any:
    """Recursively redact secret-bearing fields from a config structure.

    Any dict key containing ``token``, ``secret``, ``password``, or
    ``credential`` (case-insensitive) whose value is non-empty is replaced
    with ``"***"``. Empty values (``""``, ``None``, ``0``) are left as-is
    so the caller can distinguish "not set" from "set but redacted".
    """
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for key, val in obj.items():
            key_lower = str(key).lower()
            if any(kw in key_lower for kw in _REDACT_KEYWORDS) and val:
                result[key] = "***"
            else:
                result[key] = _redact_secrets(val)
        return result
    if isinstance(obj, list):
        return [_redact_secrets(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Git helpers (subprocess-based, run in thread pool)
# ---------------------------------------------------------------------------


def _git_head_sha(repo_root: Path) -> str:
    """``git rev-parse HEAD`` of the repo (falls back to 'unknown').

    Defensive: returns ``"unknown"`` if git is unavailable, the repo has
    no commits, or the call times out.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _current_branch(repo_root: Path) -> str | None:
    """Return the current git branch name, or ``None`` on failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    name = proc.stdout.strip()
    return name or None


def _branch_exists(repo_root: Path, branch: str) -> bool:
    """Return True if ``branch`` exists in the repo."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", branch],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _ensure_parent_branch(repo_root: Path) -> None:
    """Create ``anvil/exp`` from the current branch if it doesn't exist.

    The ANVIL loop forks round branches off ``anvil/exp``. When a fresh
    clone lacks that branch, create it pointing at the current HEAD,
    then restore the original checkout branch so the working tree is
    unchanged.

    Raises ``RuntimeError`` if the branch creation fails (W2: nonzero
    exit code is no longer silently ignored).
    """
    if _branch_exists(repo_root, _PARENT_BRANCH):
        return
    original = _current_branch(repo_root)
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "checkout", "-b", _PARENT_BRANCH],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to create branch '{_PARENT_BRANCH}': "
            f"{proc.stderr.strip() or 'unknown git error'}"
        )
    if original and original != _PARENT_BRANCH:
        subprocess.run(
            ["git", "-C", str(repo_root), "checkout", original],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )


def _parse_github_url(url: str) -> tuple[str, str | None, str | None]:
    """Extract ``(clone_url, branch, subpath)`` from a GitHub URL.

    Handles subdirectory URLs like
    ``https://github.com/user/repo/tree/main/statement-agent`` by splitting
    the path after ``/tree/``: the last segment is the subdirectory, everything
    between ``tree/`` and the last segment is the branch (branches can contain
    slashes).

    Examples::

        https://github.com/user/repo
            → ("https://github.com/user/repo", None, None)
        https://github.com/user/repo/tree/main
            → ("https://github.com/user/repo", "main", None)
        https://github.com/user/repo/tree/main/statement-agent
            → ("https://github.com/user/repo", "main", "statement-agent")
        https://github.com/user/repo/tree/feature/foo/bar
            → ("https://github.com/user/repo", "feature/foo", "bar")

    Non-GitHub URLs are returned as-is with no branch or subpath.
    ``.git`` suffixes are stripped from the clone URL.
    """
    if not url.startswith("https://github.com/"):
        return (url, None, None)

    clean = url.rstrip("/")
    tree_marker = "/tree/"
    tree_idx = clean.find(tree_marker)
    if tree_idx == -1:
        clone_url = clean
        if clone_url.endswith(".git"):
            clone_url = clone_url[:-4]
        return (clone_url, None, None)

    clone_url = clean[:tree_idx]
    if clone_url.endswith(".git"):
        clone_url = clone_url[:-4]

    after_tree = clean[tree_idx + len(tree_marker):]
    parts = after_tree.split("/")
    if len(parts) <= 1:
        branch = parts[0] if parts and parts[0] else None
        return (clone_url, branch, None)
    # Last segment is the subpath; the rest is the branch (may have slashes).
    subpath = parts[-1]
    branch = "/".join(parts[:-1])
    return (clone_url, branch or None, subpath or None)


def _clone_repo(
    repo_url: str, dest_path: Path, github_token: str | None, branch: str | None = None
) -> str | None:
    """Clone ``repo_url`` into ``dest_path``. Return ``None`` on success or
    an error message on failure (run in a thread pool).

    When ``branch`` is not ``None``, passes ``--branch <branch>`` to
    ``git clone`` so a specific branch is checked out.
    """
    url = repo_url
    if github_token and url.startswith("https://github.com/"):
        url = f"https://x-access-token:{github_token}@github.com/" + url[len("https://github.com/"):]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if branch is not None:
        cmd += ["--branch", branch]
    cmd += [url, str(dest_path)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return f"git clone failed: {exc}"
    if proc.returncode != 0:
        return proc.stderr.strip() or "git clone failed"
    return None


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Smart remediation — scan alternative structures and map them to Forge
# expectations so the user gets actionable guidance instead of a generic
# "create scaffold/harness.yaml" message.
# ---------------------------------------------------------------------------

# Finding categories that represent a recognizable savesage-style structure
# the forge-converter agent can transform additively (prompts/, schema/,
# skills/*.py, judge/, harness/*.py, config.py). ``tests`` and ``data_dir``
# are excluded — they are benign remediation hints and a *valid* forge repo
# has them too, so they must NOT light up the "Convert" button on their own.
_CONVERTIBLE_FINDINGS = frozenset(
    {"prompts", "schemas", "python_skills", "judge", "harness_py", "config_py"}
)


def _scan_agent_root(repo_path: Path) -> dict[str, list[str]]:
    """Scan the agent root for common alternative structures.

    Returns a dict mapping finding categories to lists of filenames (or
    descriptive strings).  Used by :func:`_build_smart_remediation` to
    prepend specific, actionable guidance when a validation check fails.

    Categories scanned:

    - ``prompts`` — ``prompts/`` dir with ``.txt`` files
    - ``schemas`` — ``schema/`` dir with ``.json`` files
    - ``python_skills`` — ``skills/`` dir with ``.py`` files
    - ``judge`` — ``judge/evaluator.py`` or ``judge/scorer.py``
    - ``tests`` — ``tests/`` directory
    - ``harness_py`` — ``harness/`` dir with ``.py`` files but no ``config.yaml``
    - ``config_py`` — ``config.py`` or ``config*.py`` at the repo root
    - ``data_dir`` — ``data/`` directory (exists, maybe no ``golden_set.jsonl``)
    """
    findings: dict[str, list[str]] = {}

    prompts_dir = repo_path / "prompts"
    if prompts_dir.is_dir():
        txt_files = sorted(p.name for p in prompts_dir.glob("*.txt"))
        if txt_files:
            findings["prompts"] = txt_files

    schema_dir = repo_path / "schema"
    if schema_dir.is_dir():
        json_files = sorted(p.name for p in schema_dir.glob("*.json"))
        if json_files:
            findings["schemas"] = json_files

    skills_dir = repo_path / "skills"
    if skills_dir.is_dir():
        py_files = sorted(p.name for p in skills_dir.glob("*.py"))
        if py_files:
            findings["python_skills"] = py_files

    judge_dir = repo_path / "judge"
    if judge_dir.is_dir():
        judge_files = [n for n in ("evaluator.py", "scorer.py") if (judge_dir / n).is_file()]
        if judge_files:
            findings["judge"] = judge_files

    if (repo_path / "tests").is_dir():
        findings["tests"] = ["tests/ directory found"]

    harness_dir = repo_path / "harness"
    if harness_dir.is_dir():
        py_files = sorted(p.name for p in harness_dir.glob("*.py"))
        if py_files and not (harness_dir / "config.yaml").is_file():
            findings["harness_py"] = py_files

    config_py = sorted(p.name for p in repo_path.glob("config*.py") if p.is_file())
    if config_py:
        findings["config_py"] = config_py

    if (repo_path / "data").is_dir():
        findings["data_dir"] = ["data/ directory found"]

    return findings


def _build_smart_remediation(
    check_name: str, repo_path: Path, findings: dict[str, list[str]]
) -> str | None:
    """Return specific remediation text based on the failed check and what
    was found in the repo.

    Returns ``None`` when no relevant findings exist for the given check,
    so the caller keeps the existing static remediation unchanged.
    """
    if check_name == "scaffold_harness_yaml":
        prompts = findings.get("prompts")
        if prompts:
            files_str = ", ".join(prompts)
            return (
                f"Found prompts/ with {files_str}. Decompose each prompt file's "
                f"sections (transaction rules, rewards rules, edge cases, etc.) into "
                f"separate scaffold/*.md skill files, then reference them in "
                f"scaffold/harness.yaml under the 'skills' list."
            )

    elif check_name == "golden_set_jsonl":
        schemas = findings.get("schemas")
        if schemas:
            files_str = ", ".join(schemas)
            return (
                f"Found schema/ with {files_str}. Use these schema definitions as "
                f"the expected_parsed_json structure for your golden_set.jsonl test "
                f"cases. Each golden set line should have: example_id, query (or "
                f"input), and expected_parsed_json matching the relevant bank's schema."
            )

    elif check_name == "harness_config_yaml":
        harness_py = findings.get("harness_py")
        if harness_py:
            files_str = ", ".join(harness_py)
            return (
                f"Found harness/ with {files_str}. Create harness/config.yaml (Forge's "
                f"config format) with: mode (prompt or code), runtime_endpoint, "
                f"optimizer_endpoint, eval section, gate section. Reference your "
                f"existing config_ws4.py for endpoint names and eval settings."
            )

    elif check_name == "agent_code":
        python_skills = findings.get("python_skills")
        if python_skills:
            files_str = ", ".join(python_skills)
            return (
                f"Found skills/ with {files_str}. For code mode, move or wrap these in "
                f"an agents/ directory. Create agents/<name>.py with a class implementing "
                f"a predict() method that delegates to your existing skill functions."
            )

    return None


def _prepend_smart_remediation(
    result: dict, repo_path: Path, findings: dict[str, list[str]] | None
) -> dict:
    """Prepend smart remediation to a failed check's static remediation.

    Smart remediation is only added when the check failed AND relevant
    alternative structures were found.  When no relevant findings exist the
    existing static remediation is kept unchanged.  The smart text is
    prepended (separated by a newline) so the user sees the specific
    guidance first, then the generic fallback.
    """
    if findings is None or result.get("status") != "fail":
        return result
    smart = _build_smart_remediation(result["name"], repo_path, findings)
    if smart is None:
        return result
    existing = result.get("remediation") or ""
    result["remediation"] = f"{smart}\n{existing}" if existing else smart
    return result


def _check_git_repo(repo_path: Path) -> dict:
    if _git_head_sha(repo_path) == "unknown":
        return {
            "name": "git_repo",
            "status": "fail",
            "message": "Not a git repository or no commits found.",
            "remediation": "Initialize the repo with `git init && git add -A && "
            "git commit -m 'initial'` and push to GitHub.",
        }
    return {"name": "git_repo", "status": "pass", "message": "Valid git repository."}


def _check_scaffold_harness_yaml(
    repo_path: Path, findings: dict[str, list[str]] | None = None
) -> dict:
    path = repo_path / "scaffold" / "harness.yaml"
    if not path.is_file():
        return _prepend_smart_remediation(
            {
                "name": "scaffold_harness_yaml",
                "status": "fail",
                "message": "scaffold/harness.yaml not found",
                "remediation": "Create scaffold/harness.yaml with a 'skills' list and "
                "'sampling' dict. Each skill has a 'file' field pointing to a markdown "
                "file in scaffold/skills/ using its basename (for example, identity.md "
                "resolves to scaffold/skills/identity.md). Decompose your agent's prompt "
                "into sections — each section becomes one skill. Example from the Savesage "
                "ICICI integration (files live in scaffold/skills/): "
                "skills = [identity.md, transaction_rules.md, rewards_rules.md, "
                "missing_data.md, edge_cases.md, icici_bank_rules.md, "
                "icici_card_identity.md, icici_rewards_layouts.md]",
            },
            repo_path,
            findings,
        )
    try:
        raw = _load_yaml(path)
    except yaml.YAMLError:
        return {
            "name": "scaffold_harness_yaml",
            "status": "fail",
            "message": "Invalid YAML",
            "remediation": "Fix the YAML syntax in scaffold/harness.yaml.",
        }
    if not isinstance(raw, dict) or not isinstance(raw.get("skills"), list) or not isinstance(
        raw.get("sampling"), dict
    ):
        return {
            "name": "scaffold_harness_yaml",
            "status": "fail",
            "message": "Missing 'skills' or 'sampling' section",
            "remediation": "Create scaffold/harness.yaml with a 'skills' list and "
            "'sampling' dict. Each skill's 'file' value should be the basename of a file "
            "in scaffold/skills/ (for example, identity.md resolves to "
            "scaffold/skills/identity.md).",
        }
    return {
        "name": "scaffold_harness_yaml",
        "status": "pass",
        "message": f"scaffold/harness.yaml valid ({len(raw['skills'])} skills).",
    }


def _check_scaffold_skill_files(repo_path: Path, skills: list[Any]) -> dict:
    for entry in skills:
        if not isinstance(entry, dict):
            continue
        rel = entry.get("file")
        if not rel:
            continue
        canonical_path = repo_path / "scaffold" / "skills" / rel
        legacy_path = repo_path / "scaffold" / rel
        if not canonical_path.is_file() and not legacy_path.is_file():
            return {
                "name": "scaffold_skill_files",
                "status": "fail",
                "message": f"Skill file '{rel}' referenced in harness.yaml not found in "
                "scaffold/skills/ (or scaffold/)",
                "remediation": "Create the missing skill markdown file in scaffold/skills/. "
                "Each skill file is a section of the agent's system prompt; scaffold/ is "
                "also supported for backward compatibility.",
            }
    return {
        "name": "scaffold_skill_files",
        "status": "pass",
        "message": "All skill files present.",
    }


def _check_golden_set_jsonl(
    repo_path: Path, findings: dict[str, list[str]] | None = None
) -> dict:
    path = repo_path / "data" / "golden_set.jsonl"
    if not path.is_file():
        if (repo_path / "scripts" / "build_golden_set.py").is_file():
            return {
                "name": "golden_set_jsonl",
                "status": "warn",
                "message": "data/golden_set.jsonl not found — build it locally",
                "remediation": "Build the golden set locally with scripts/build_golden_set.py. "
                "The data is cardholder PII and is gitignored by design — the repo is "
                "structurally forge-compatible; the optimizer will need the golden set "
                "before the first scored round.",
            }
        return _prepend_smart_remediation(
            {
                "name": "golden_set_jsonl",
                "status": "fail",
                "message": "data/golden_set.jsonl not found",
                "remediation": "Create data/golden_set.jsonl with test examples. Each line is a "
                "JSON object with at minimum: example_id (unique string), query or input "
                "(the user's request), category (classification like "
                "direct/multi_hop/distractor/out_of_scope), and expected or expectations "
                "(the ground truth answer).",
            },
            repo_path,
            findings,
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {
            "name": "golden_set_jsonl",
            "status": "fail",
            "message": f"Cannot read golden_set.jsonl: {exc}",
            "remediation": "Create data/golden_set.jsonl with test examples.",
        }
    real_lines = [ln for ln in lines if ln.strip()]
    if not real_lines:
        return {
            "name": "golden_set_jsonl",
            "status": "fail",
            "message": "File is empty",
            "remediation": "Add at least one JSON example line to data/golden_set.jsonl.",
        }
    for i, ln in enumerate(real_lines, start=1):
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            return {
                "name": "golden_set_jsonl",
                "status": "fail",
                "message": f"Line {i} is not valid JSON",
                "remediation": "Fix the JSON on the indicated line of data/golden_set.jsonl.",
            }
        if not isinstance(obj, dict) or not (
            "example_id" in obj and ("query" in obj or "input" in obj)
        ):
            return {
                "name": "golden_set_jsonl",
                "status": "fail",
                "message": f"Line {i} missing required field (example_id + query/input)",
                "remediation": "Each line needs example_id and query (or input).",
            }
    return {
        "name": "golden_set_jsonl",
        "status": "pass",
        "message": f"{len(real_lines)} golden examples.",
    }


def _check_harness_config_yaml(
    repo_path: Path, findings: dict[str, list[str]] | None = None
) -> tuple[dict, dict | None]:
    """Return (check_result, parsed_config_dict_or_None)."""
    path = repo_path / "harness" / "config.yaml"
    if not path.is_file():
        return (
            _prepend_smart_remediation(
                {
                    "name": "harness_config_yaml",
                    "status": "fail",
                    "message": "harness/config.yaml not found",
                    "remediation": "Create harness/config.yaml with: mode (prompt or code), "
                    "runtime_endpoint (FMAPI model name like databricks-claude-sonnet-4-6), "
                    "optimizer_endpoint (like databricks-claude-opus-4-7), eval section "
                    "(default_mode, modes with row counts, scorers list), gate section "
                    "(type: frontier recommended). Copy the forge repo's harness/config.yaml "
                    "as a starting template.",
                },
                repo_path,
                findings,
            ),
            None,
        )
    try:
        raw = _load_yaml(path)
    except yaml.YAMLError:
        return (
            {
                "name": "harness_config_yaml",
                "status": "fail",
                "message": "Invalid YAML",
                "remediation": "Fix the YAML syntax in harness/config.yaml.",
            },
            None,
        )
    if not isinstance(raw, dict):
        return (
            {
                "name": "harness_config_yaml",
                "status": "fail",
                "message": "harness/config.yaml is not a mapping",
                "remediation": "harness/config.yaml must be a YAML mapping at the top level.",
            },
            None,
        )
    missing: list[str] = []
    if "mode" not in raw:
        missing.append("mode")
    eval_section = raw.get("eval")
    if not isinstance(eval_section, dict):
        missing.append("eval section")
    else:
        if "default_mode" not in eval_section:
            missing.append("eval.default_mode")
        if not isinstance(eval_section.get("scorers"), list):
            missing.append("eval.scorers")
    gate = raw.get("gate")
    if not isinstance(gate, dict) or "type" not in gate:
        missing.append("gate.type")
    if missing:
        return (
            {
                "name": "harness_config_yaml",
                "status": "fail",
                "message": f"Missing: {', '.join(missing)}",
                "remediation": "Add the missing fields to harness/config.yaml.",
            },
            None,
        )
    return (
        {"name": "harness_config_yaml", "status": "pass", "message": "harness/config.yaml valid."},
        raw,
    )


def _check_eval_modes(config: dict) -> dict:
    eval_section = config.get("eval") or {}
    modes = eval_section.get("modes")
    if not isinstance(modes, dict) or not any(
        isinstance(m, dict) and isinstance(m.get("rows"), int) and m["rows"] > 0
        for m in modes.values()
    ):
        return {
            "name": "eval_modes",
            "status": "fail",
            "message": "eval.modes section missing or has no modes with 'rows'",
            "remediation": "Add an eval.modes section to harness/config.yaml. Example: "
            "modes: {quick: {rows: 12}, standard: {rows: 24}, full: {rows: 304}}",
        }
    return {"name": "eval_modes", "status": "pass", "message": "eval.modes present."}


def _check_agent_code(
    repo_path: Path, config: dict, findings: dict[str, list[str]] | None = None
) -> dict:
    mode = config.get("mode")
    if mode != "code":
        return {
            "name": "agent_code",
            "status": "pass",
            "message": f"mode is '{mode}' — agent_code check skipped.",
        }
    agents_dir = repo_path / "agents"
    has_py = agents_dir.is_dir() and any(agents_dir.glob("*.py"))
    if not has_py:
        return _prepend_smart_remediation(
            {
                "name": "agent_code",
                "status": "fail",
                "message": "mode is 'code' but no Python files found in agents/",
                "remediation": "For code mode, create a Python file in agents/ with a class "
                "implementing a predict() method.",
            },
            repo_path,
            findings,
        )
    if not config.get("agent_module"):
        return {
            "name": "agent_code",
            "status": "fail",
            "message": "agent_module not set in harness/config.yaml",
            "remediation": "Set agent_module in harness/config.yaml to point to your agent "
            "module (e.g. agents/my_agent.py).",
        }
    return {
        "name": "agent_code",
        "status": "pass",
        "message": "agents/ has Python code and agent_module is set.",
    }


def _check_parent_branch(repo_path: Path) -> dict:
    if _branch_exists(repo_path, _PARENT_BRANCH):
        return {
            "name": "parent_branch",
            "status": "pass",
            "message": f"Branch '{_PARENT_BRANCH}' exists.",
        }
    return {
        "name": "parent_branch",
        "status": "warn",
        "message": "Branch 'anvil/exp' not found. The app will create it "
        "automatically before optimization.",
    }


def _run_validation(repo_path: Path) -> tuple[dict, dict | None, dict[str, list[str]]]:
    """Run all validation checks.

    Return ``(ValidationReport_dict, config_or_None, findings)``. The config
    is returned (unredacted) so the session can store a redacted copy; ``None``
    when the config check failed. ``findings`` is the alternative-structure
    scan from :func:`_scan_agent_root` — the caller stores it on the session
    so the auto-converter can feed it to :func:`build_conversion_prompt`.

    The report dict carries a ``convertible`` flag (True when the repo failed
    but has a recognizable savesage-style structure the converter can handle)
    that gates the "Convert to forge-compatible" button in the UI.
    """
    # Scan for alternative structures once — used by smart remediation.
    findings = _scan_agent_root(repo_path)

    checks: list[dict] = []
    checks.append(_check_git_repo(repo_path))
    checks.append(_check_scaffold_harness_yaml(repo_path, findings))

    skills: list[Any] = []
    harness_path = repo_path / "scaffold" / "harness.yaml"
    if harness_path.is_file():
        try:
            raw = _load_yaml(harness_path)
            if isinstance(raw, dict) and isinstance(raw.get("skills"), list):
                skills = raw["skills"]
        except yaml.YAMLError:
            pass

    checks.append(_check_scaffold_skill_files(repo_path, skills))
    checks.append(_check_golden_set_jsonl(repo_path, findings))
    config_check, config = _check_harness_config_yaml(repo_path, findings)
    checks.append(config_check)
    if config is not None:
        checks.append(_check_eval_modes(config))
        checks.append(_check_agent_code(repo_path, config, findings))
    else:
        # W1: Always include all 8 checks. When the config prerequisite
        # failed, mark the dependent checks as skipped/fail so the
        # report always has exactly 8 entries.
        for dep_name in ("eval_modes", "agent_code"):
            checks.append(
                {
                    "name": dep_name,
                    "status": "fail",
                    "message": "Skipped: prerequisite check 'harness_config_yaml' failed",
                }
            )
    checks.append(_check_parent_branch(repo_path))

    any_fail = any(c["status"] == "fail" for c in checks)
    # A repo is "convertible" when it failed validation but has at least one
    # recognizable alternative structure (prompts/, schema/, harness/*.py,
    # skills/*.py, …) the forge-converter agent can transform additively.
    # Only the transformable categories count — ``tests``/``data_dir`` are
    # benign (a valid repo has them) and would false-trigger the button.
    convertible = any(findings.get(c) for c in _CONVERTIBLE_FINDINGS)
    report = {
        "status": "invalid" if any_fail else "valid",
        "checks": checks,
        "convertible": convertible,
    }
    return (report, config, findings)


# ---------------------------------------------------------------------------
# Round / baseline / frontier helpers (read persisted JSON)
# ---------------------------------------------------------------------------


def _round_json_path(repo_root: Path, round_id: int) -> Path:
    return repo_root / "eval" / "runs" / f"round_{round_id:03d}.json"


def _list_round_summaries(repo_root: Path) -> list[dict[str, Any]]:
    """Scan eval/runs/round_*.json and return summaries sorted by round_id."""
    runs_dir = repo_root / "eval" / "runs"
    if not runs_dir.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for p in sorted(runs_dir.glob("round_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        decision = data.get("decision")
        if decision not in _VALID_DECISIONS:
            decision = None
        round_id = data.get("round_id")
        try:
            round_id = int(round_id)
        except (TypeError, ValueError):
            round_id = None
        summaries.append(
            {
                "round_id": round_id,
                "decision": decision,
                "action_kind": data.get("action_kind"),
                "baseline_score": data.get("baseline_score"),
                "score_delta": data.get("score_delta_vs_parent"),
                "aggregate": data.get("aggregate"),
            }
        )
    summaries.sort(key=lambda r: r.get("round_id") or 0)
    return summaries


def _read_json_file_sync(path: Path) -> dict[str, Any]:
    """Read a JSON file synchronously, raising FileNotFoundError if absent."""
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _read_round_json_sync(path: Path) -> dict[str, Any] | None:
    """Read a round JSON file. Return ``None`` if absent or corrupt.

    B4: designed to run in a thread pool so file I/O doesn't block the
    event loop.
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_artifacts_sync(repo_path: Path) -> dict[str, Any]:
    """Read frontier/finalized/baseline JSON from disk.

    B4: runs entirely in a thread pool so all file I/O is offloaded.
    Returns a dict whose ``frontier`` key is always present (``None`` if
    not found); ``finalized`` and ``baseline`` keys are only present when
    the file exists and parsed successfully.
    """
    frontier = load_frontier(repo_path)
    result: dict[str, Any] = {"frontier": frontier.to_dict() if frontier else None}
    fin_path = repo_path / "eval" / "runs" / "finalized.json"
    if fin_path.is_file():
        with suppress(json.JSONDecodeError, OSError):
            result["finalized"] = json.loads(fin_path.read_text(encoding="utf-8"))
    base_path = repo_path / "eval" / "runs" / "baseline.json"
    if base_path.is_file():
        with suppress(json.JSONDecodeError, OSError):
            result["baseline"] = json.loads(base_path.read_text(encoding="utf-8"))
    return result


# ---------------------------------------------------------------------------
# Synchronous baseline + finalize (run in thread pool)
# ---------------------------------------------------------------------------


def _reset_frontier(repo_path: Path) -> None:
    """Delete any existing ``eval/runs/frontier.json``.

    The frontier gate (:func:`anvil.loop.frontier.gate_decision`) only seeds
    from the baseline when no ``frontier.json`` exists — a stale frontier
    left over from a prior optimization would silently ignore a freshly
    built baseline. Called after every ``save_baseline`` (both the MLflow
    and local-eval paths) so the gate re-seeds from the new baseline on the
    next scored round.

    Safe to call when the file does not exist (no-op).
    """
    frontier_file = repo_path / "eval" / "runs" / "frontier.json"
    if frontier_file.is_file():
        frontier_file.unlink()


def _ensure_synthetic_golden_set(repo_path: Path) -> Path:
    """Ensure a golden set file exists, synthesizing a minimal fallback.

    The real ``data/golden_set.jsonl`` is gitignored (cardholder PII) and
    absent from a fresh clone of the agent repo. On the Databricks App the
    process CWD is the app source dir — not the cloned repo — so the
    relative ``"data/golden_set.jsonl"`` default resolves to the wrong
    place and :func:`load_golden_set` raises ``FileNotFoundError`` before
    the first eval can run.

    When the file is missing this writes a minimal 3-row synthetic golden
    set carrying every required field (see
    :data:`anvil.data.golden_set.REQUIRED_FIELDS`) so the eval runner has
    something to score. The real PII golden set should be provisioned
    out-of-band; this fallback exists so the orchestrator does not
    hard-crash on a fresh clone.

    Returns the path to the golden set file (existing or newly written).
    """
    golden_set_path = repo_path / "data" / "golden_set.jsonl"
    if golden_set_path.is_file():
        return golden_set_path
    golden_set_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "example_id": f"test-{i:03d}",
            "query": "What are the transaction details?",
            "category": "test",
            "expected_doc_ids": [],
            "reference_answer": "Sample answer",
            "should_refuse": False,
            "expected_citations": [],
            "must_include": [],
            "must_not_include": [],
            "notes_for_judge": "",
        }
        for i in range(1, 4)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    golden_set_path.write_text(payload, encoding="utf-8")
    return golden_set_path


def _build_baseline_sync(
    repo_path: Path,
    eval_mode: str | None,
    mlflow_experiment_id: str | None = None,
    mode: str | None = None,
) -> dict:
    """Run eval on the current scaffold and write eval/runs/baseline.json.

    Mirrors ``scripts/make_baseline.build_baseline``: calls
    ``evaluate_branch``, reads endpoints from harness/config.yaml, gets the
    scaffold commit SHA from git, converts the EvalReport to a
    CachedBaseline via ``report_to_baseline``, and persists it.

    When ``mlflow_experiment_id`` is provided, the baseline is seeded from
    that MLflow experiment's judge results (via :func:`build_mlflow_baseline`)
    instead of a local golden-set eval. The MLflow baseline carries an empty
    ``scorer_fingerprint`` so the round loop's compatibility check is a
    no-op, and it becomes the actual round gate once saved.

    When ``mode`` (the optimization mode: "prompt" or "code") is provided it
    overrides the ``mode`` key in ``harness/config.yaml`` for the baseline
    eval so :func:`evaluate_branch` runs in the requested mode — the same
    override :func:`run_round` applies per round. The config file is rewritten
    in place so the eval's ``load_harness`` picks it up.

    In both paths the frontier is reset (:func:`_reset_frontier`) after the
    baseline is saved so the gate re-seeds from it on the next scored round
    instead of trusting a stale ``frontier.json``.
    """
    if mlflow_experiment_id:
        baseline = build_mlflow_baseline(experiment_id=mlflow_experiment_id)
        save_baseline(repo_path, baseline)
        _reset_frontier(repo_path)
        return baseline.to_dict()

    # The real golden set is gitignored (cardholder PII) and absent from a
    # fresh clone; synthesize a minimal fallback so the baseline eval does
    # not FileNotFoundError before the first scored round.
    golden_set_path = _ensure_synthetic_golden_set(repo_path)

    scaffold_root = repo_path / "scaffold"
    config_path = repo_path / "harness" / "config.yaml"
    runtime_endpoint = ""
    judge_endpoint = ""
    if config_path.is_file():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        runtime_endpoint = raw.get("runtime_endpoint", "")
        judge_endpoint = raw.get("judge_endpoint", "")
        # Override the optimization mode on disk when an explicit mode is
        # requested, so the baseline eval (which reads mode via load_harness)
        # runs in the same mode the optimizer will use for the rounds.
        if mode is not None and raw.get("mode") != mode:
            raw["mode"] = mode
            config_path.write_text(
                yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
            )
    report = evaluate_branch(
        scaffold_root=scaffold_root,
        runtime_config_path=config_path if config_path.is_file() else None,
        golden_set_path=str(golden_set_path),
        mode=eval_mode,
    )
    baseline = report_to_baseline(
        report,
        scaffold_commit_sha=_git_head_sha(repo_path),
        runtime_endpoint=runtime_endpoint,
        judge_endpoint=judge_endpoint,
    )
    save_baseline(repo_path, baseline)
    _reset_frontier(repo_path)
    return baseline.to_dict()


def _finalize_sync(repo_path: Path) -> dict[str, Any]:
    """Evaluate HEAD on the held-out set and write eval/runs/finalized.json.

    Mirrors ``scripts/finalize.finalize``.
    """
    config_path = repo_path / "harness" / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError("harness/config.yaml not found; cannot finalize")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not (raw.get("eval") or {}).get("held_out_test"):
        raise RuntimeError("held-out finalization is disabled; set eval.held_out_test: true")
    frontier = load_frontier(repo_path)
    if frontier is None:
        raise RuntimeError("cannot finalize without eval/runs/frontier.json")
    report = evaluate_branch(
        scaffold_root=repo_path / "scaffold",
        runtime_config_path=config_path,
        mode="test",
        allow_test=True,
    )
    payload = {
        **dataclasses.asdict(report),
        "scaffold_commit_sha": _git_head_sha(repo_path),
        "finalized_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "frontier": frontier.to_dict(),
    }
    out_path = repo_path / "eval" / "runs" / "finalized.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_sessions: dict[str, SessionData] = {}
_session_lock = threading.Lock()


def _get_session(session_id: str) -> SessionData | None:
    return _sessions.get(session_id)


def _require_session(session_id: str) -> SessionData:
    sess = _get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    return sess


def _session_to_response(sess: SessionData, rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the JSON-serializable session response dict.

    ``rounds`` is read from disk by the caller (in a thread pool) and
    passed in so no file I/O happens under ``_session_lock``.
    """
    return {
        "session_id": sess.session_id,
        "repo_url": sess.repo_url,
        "agent_subpath": sess.agent_subpath,
        "status": sess.status,
        "validation": sess.validation,
        "config": sess.config,
        "baseline": sess.baseline,
        "rounds": rounds,
        "frontier": sess.frontier,
        "finalized": sess.finalized,
        "error": sess.error,
    }


def _apply_artifacts(sess: SessionData, artifacts: dict[str, Any]) -> None:
    """Apply disk-read artifacts to the session (caller holds ``_session_lock``).

    ``frontier`` is always overwritten (``None`` if not found).
    ``finalized`` / ``baseline`` are only overwritten when present in the
    dict (i.e. the file existed and parsed).
    """
    sess.frontier = artifacts["frontier"]
    if "finalized" in artifacts:
        sess.finalized = artifacts["finalized"]
    if "baseline" in artifacts:
        sess.baseline = artifacts["baseline"]


# ---------------------------------------------------------------------------
# Async background optimization task
# ---------------------------------------------------------------------------


async def _run_optimization_task(
    session_id: str,
    eval_mode: str | None,
    max_rounds: int,
    max_turns: int,
    mlflow_experiment_id: str | None = None,
    mode: str | None = None,
) -> None:
    """Background asyncio task that runs baseline + rounds.

    Each blocking step (parent-branch creation, baseline, each round,
    file reads) runs via ``anyio.to_thread.run_sync`` so the event loop
    stays responsive and a polling client sees the status transition
    building_baseline → optimizing → optimized, with rounds appearing
    one at a time.

    B7: ``_ensure_parent_branch`` runs *inside* the task (not in the
    request handler) so a git failure sets the session to ``error``
    instead of leaving it stuck in ``building_baseline``.
    """
    try:
        sess = _get_session(session_id)
        if sess is None:
            return
        # B7: Ensure the parent branch exists — inside the task so a
        # failure transitions to 'error' instead of a stuck session.
        await anyio.to_thread.run_sync(partial(_ensure_parent_branch, sess.repo_path))
        # Baseline (blocking) in a thread pool.
        baseline = await anyio.to_thread.run_sync(
            partial(
                _build_baseline_sync,
                sess.repo_path,
                eval_mode,
                mlflow_experiment_id,
                mode,
            )
        )
        with _session_lock:
            sess.baseline = baseline
            sess.status = "optimizing"
        # Rounds — each in its own thread-pool call so the event loop can
        # update status + the rounds list between rounds.
        for i in range(1, max_rounds + 1):
            await anyio.to_thread.run_sync(
                partial(
                    run_round,
                    round_id=i,
                    repo_root=sess.repo_path,
                    eval_mode=eval_mode,
                    max_turns=max_turns,
                    mode=mode,
                )
            )
            # B4: Read round JSON in a thread pool (not under the lock).
            round_data = await anyio.to_thread.run_sync(
                _read_round_json_sync, _round_json_path(sess.repo_path, i)
            )
            if round_data is not None:
                with _session_lock:
                    sess.rounds.append(round_data)
            # B4: Check finalized.json in a thread pool.
            fin_path = sess.repo_path / "eval" / "runs" / "finalized.json"
            if await anyio.to_thread.run_sync(fin_path.is_file):
                break
        # B4: Read artifacts in a thread pool, then update under lock.
        artifacts = await anyio.to_thread.run_sync(_read_artifacts_sync, sess.repo_path)
        with _session_lock:
            _apply_artifacts(sess, artifacts)
            sess.status = "finalized" if sess.finalized else "optimized"
    except Exception as exc:  # noqa: BLE001 — surface any failure
        with _session_lock:
            sess = _get_session(session_id)
            if sess is not None:
                sess.status = "error"
                sess.error = str(exc)
        logger.exception("optimization task for session %s failed", session_id)
        logger.exception("optimization task for session %s failed", session_id)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def _cleanup_session(session_id: str) -> None:
    """Remove a session's cloned repo directory (run in thread pool).

    B8: called during shutdown for every session. Does NOT cancel the
    optimization task — that is handled separately in the lifespan.

    When a subdirectory URL was used, ``repo_path`` points at the
    subdirectory but the full clone lives at ``_clone_root`` — clean up
    the clone root so no temp files leak.
    """
    sess = _sessions.get(session_id)
    if sess is None:
        return
    root = sess._clone_root or sess.repo_path
    shutil.rmtree(root, ignore_errors=True)


async def _cleanup_omnigent_session(session_id: str) -> None:
    """Best-effort delete the persisted Omnigent managed conversation session
    for a forge session, if a conversion run created one.

    The conversion keeps the managed session alive so the transcript is
    inspectable (see :func:`_run_managed_session`); on shutdown we release
    it so remote sessions don't leak indefinitely. All failures are swallowed
    — the remote server may already be gone or unreachable.
    """
    with _session_lock:
        sess = _sessions.get(session_id)
        if sess is None or sess.conversion is None:
            return
        omnigent_sid = sess.conversion.session_id
    if not omnigent_sid:
        return
    server_url = os.getenv("OMNIGENT_SERVER_URL")
    if not server_url:
        return
    auth_token = os.getenv("OMNIGENT_AUTH_TOKEN")
    try:
        from anvil.optimizer.omnigent_client import OmnigentClient, OmnigentError

        client = OmnigentClient(server_url, auth_token)
        try:
            with suppress(OmnigentError):
                await client.delete_session(omnigent_sid)
        finally:
            await client.aclose()
    except Exception:  # noqa: BLE001 — best-effort, swallow all failures
        logger.debug(
            "omnigent session cleanup failed for %s", omnigent_sid, exc_info=True
        )


# ---------------------------------------------------------------------------
# Crash diagnostics — signal handlers + excepthook write to
# /tmp/forge-crash-log.txt and stderr (captured by the platform) so we can
# diagnose serverless crashes that kill the process ~1 min after startup.
# Also best-effort mirrors the log to a Databricks workspace file via REST.
# ---------------------------------------------------------------------------

_CRASH_LOG_FILE = Path("/tmp/forge-crash-log.txt")

# Workspace path for the crash-log mirror. Configurable via env var;
# when unset the mirror is skipped (avoids hard-coding a user email).
_CRASH_LOG_WORKSPACE_PATH = os.getenv("FORGE_CRASH_LOG_WORKSPACE_PATH")


def _write_crash_log_to_databricks(entry: str) -> None:
    """Best-effort write the crash log to a Databricks workspace file.

    Reads ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` from the environment.
    On serverless compute these are often unset — falls back to deriving the
    host from ``OMNIGENT_SERVER_URL`` (stripping the ``/api/2.0/omnigent``
    suffix) and using ``OMNIGENT_AUTH_TOKEN`` as the token. Any failure is
    swallowed silently (best-effort only).
    """
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    if not host or not token:
        omnigent_url = os.getenv("OMNIGENT_SERVER_URL")
        omnigent_token = os.getenv("OMNIGENT_AUTH_TOKEN")
        if omnigent_url and omnigent_token:
            host = omnigent_url
            suffix = "/api/2.0/omnigent"
            if host.endswith(suffix):
                host = host[: -len(suffix)]
            token = omnigent_token
    if not host or not token:
        return
    ws_path = _CRASH_LOG_WORKSPACE_PATH
    if not ws_path:
        return
    try:
        import httpx  # lazy import — best-effort, may not be installed
    except ImportError:
        return
    url = (
        f"{host}/api/2.0/workspace-files/write"
        f"?path={ws_path}&overwrite=true"
    )
    try:
        resp = httpx.put(
            url,
            content=entry,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception:  # noqa: BLE001 — best-effort, swallow all failures
        pass


def _write_crash_log(source: str, exc: BaseException | None = None) -> None:
    """Append a crash record to the crash log file + stderr.

    ``source`` is a short label (e.g. ``"signal: SIGTERM"``);
    ``exc``, when given, is formatted with its traceback.
    """
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        entry = f"[{ts}] {source}\n{tb}\n"
    else:
        entry = f"[{ts}] {source}\n"
    # Local file (append so successive crashes accumulate).
    try:
        with _CRASH_LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except OSError:
        pass
    # stderr is captured by the Databricks Apps log infrastructure.
    print(entry, file=sys.stderr, flush=True)
    # Best-effort mirror to a Databricks workspace file.
    _write_crash_log_to_databricks(entry)


def _signal_handler(signum: int, frame: Any) -> None:
    """Write signal info to the crash log on SIGTERM/SIGINT, then chain to
    the previous handler so Uvicorn's graceful shutdown still works."""
    try:
        sig_name = signal.Signals(signum).name
    except ValueError:
        sig_name = f"signal-{signum}"
    _write_crash_log(f"signal: {sig_name} ({signum})")
    # Chain to the prior handler (Uvicorn's graceful-shutdown handler)
    # so the process still terminates instead of silently swallowing the
    # signal and requiring SIGKILL.
    prev = _prev_sigterm if signum == signal.SIGTERM else _prev_sigint
    if callable(prev):
        prev(signum, frame)
    elif prev == signal.SIG_DFL:
        # Default disposition — re-raise as KeyboardInterrupt (SIGINT semantics).
        signal.default_int_handler(signum, frame)


def _async_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Custom asyncio exception handler — writes unhandled async exceptions
    to the crash log, then delegates to the default handler."""
    exc = context.get("exception")
    msg = context.get("message", "unhandled async exception")
    source = f"asyncio: {msg}"
    _write_crash_log(source, exc if isinstance(exc, BaseException) else None)
    loop.default_exception_handler(context)


def _excepthook(
    exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any
) -> None:
    """Global excepthook — writes unhandled exceptions to the crash log."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Let the default handler deal with Ctrl-C.
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    _write_crash_log("unhandled exception", exc_value)


# Register the signal handlers + excepthook at import time so they are
# active before the event loop starts. Capture the existing handlers first
# so ``_signal_handler`` can chain to them (Uvicorn installs its own
# SIGTERM/SIGINT handlers for graceful shutdown — replacing them without
# chaining would leave the process requiring SIGKILL).
_prev_sigterm = signal.getsignal(signal.SIGTERM)
_prev_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
sys.excepthook = _excepthook


def _require_imports() -> None:
    """Raise 503 if the heavy imports failed at startup.

    Called by endpoints that depend on the optional heavy imports
    (mlflow, openai, the anvil eval/loop/conversion modules) so the user
    gets a clear error instead of a ``NoneType is not callable`` crash.
    """
    if _STARTUP_ERROR is not None:
        raise HTTPException(status_code=503, detail=str(_STARTUP_ERROR))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Catch unhandled async exceptions and write them to the crash log.
    asyncio.get_event_loop().set_exception_handler(_async_exception_handler)
    logger.info("forge orchestrator startup — sessions root: %s", _SESSIONS_ROOT)
    yield
    # B8: On shutdown, cancel active optimization tasks and clean up
    # cloned repos so we don't leave temp directories behind.
    logger.info("forge orchestrator shutdown — cancelling tasks + cleaning up")
    tasks_to_cancel: list[asyncio.Task] = []
    session_ids: list[str] = []
    with _session_lock:
        for sid, sess in _sessions.items():
            session_ids.append(sid)
            if sess._optimize_task is not None:
                tasks_to_cancel.append(sess._optimize_task)
            if sess._convert_task is not None:
                tasks_to_cancel.append(sess._convert_task)
    for t in tasks_to_cancel:
        t.cancel()
    if tasks_to_cancel:
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
    # Delete any persisted Omnigent managed sessions (best-effort) before the
    # local file cleanup. The conversion keeps the managed session alive so
    # the transcript is inspectable; on shutdown we release it so remote
    # sessions don't leak indefinitely.
    for sid in session_ids:
        await _cleanup_omnigent_session(sid)
    for sid in session_ids:
        await anyio.to_thread.run_sync(partial(_cleanup_session, sid))
    with _session_lock:
        _sessions.clear()


app = FastAPI(title="Forge Orchestrator", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    if _STARTUP_ERROR is not None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "error": str(_STARTUP_ERROR)},
        )
    return {"status": "ok"}


@app.get("/crash-log")
async def crash_log() -> dict[str, Any]:
    """Return the contents of the crash log file.

    When the heavy imports failed at startup, the ``_STARTUP_ERROR`` is
    included alongside the crash log so a single request surfaces both.
    """
    try:
        content = _CRASH_LOG_FILE.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        content = "No crash logged"
    result: dict[str, Any] = {"crash_log": content}
    if _STARTUP_ERROR is not None:
        result["startup_error"] = str(_STARTUP_ERROR)
    return result


@app.post("/api/session")
async def create_session(req: CreateSessionRequest) -> dict[str, Any]:
    """Load the agent repo, validate it, return a new session.

    Local filesystem paths (absolute or ``~``-prefixed) are used in place
    without cloning. Supports subdirectory URLs like
    ``https://github.com/user/repo/tree/main/statement-agent`` — the
    repo is cloned at the root and validation/optimization run against
    the specified subdirectory.
    """
    session_id = uuid.uuid4().hex[:12]
    is_local_path = req.repo_url.startswith(("/", "~"))
    if is_local_path:
        dest_path = Path(req.repo_url).expanduser()
        clone_url, branch, subpath = req.repo_url, None, None
    else:
        dest_path = _SESSIONS_ROOT / session_id
        # Parse the URL for subdirectory + branch support.
        clone_url, branch, subpath = _parse_github_url(req.repo_url)

    sess = SessionData(
        session_id=session_id,
        repo_url=req.repo_url,
        repo_path=dest_path,
        status="cloning",
        validation={"status": "invalid", "checks": []},
        config=None,
        baseline=None,
        rounds=[],
        frontier=None,
        finalized=None,
        error=None,
        agent_subpath=subpath,
        _clone_root=dest_path,
        # Keep the user's GitHub token in memory only — the auto-converter
        # needs it to clone+push a private repo. Never serialized in any API
        # response (see ``_session_to_response`` / ``_redact_secrets``).
        _github_token=req.github_token,
    )
    with _session_lock:
        _sessions[session_id] = sess

    if not is_local_path:
        # Clone (blocking) in a thread pool. Only pass --branch when a
        # branch was extracted from the URL so existing callers (plain URLs)
        # are unaffected.
        clone_kwargs: dict[str, Any] = {}
        if branch is not None:
            clone_kwargs["branch"] = branch
        err = await anyio.to_thread.run_sync(
            partial(_clone_repo, clone_url, dest_path, req.github_token, **clone_kwargs)
        )
        if err is not None:
            # B1: Redact any embedded token from the git error message
            # before storing it in the session or returning it to the client.
            token = req.github_token
            if token:
                err = err.replace(token, "***")
            with _session_lock:
                sess.status = "invalid"
                sess.error = err
            raise HTTPException(status_code=400, detail=err)

    # Resolve the agent root — the subdirectory if a subpath was given.
    agent_root = dest_path / subpath if subpath else dest_path
    if subpath and not agent_root.is_dir():
        with _session_lock:
            sess.status = "invalid"
            sess.error = f"subdirectory '{subpath}' not found in repository"
        raise HTTPException(
            status_code=400, detail=f"subdirectory '{subpath}' not found in repository"
        )

    with _session_lock:
        sess.repo_path = agent_root
        sess.status = "validating"

    # Validate (file I/O + git) in a thread pool.
    report, config, findings = await anyio.to_thread.run_sync(partial(_run_validation, agent_root))
    with _session_lock:
        sess.validation = report
        sess.status = "validated" if report["status"] == "valid" else "invalid"
        sess._findings = findings
        if config is not None:
            sess.config = _redact_secrets(config)

    return {
        "session_id": session_id,
        "status": sess.status,
        "validation": report,
        "config": sess.config,
        "agent_subpath": subpath,
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    _require_imports()
    sess = _require_session(session_id)
    # B4: Read rounds + artifacts in a thread pool (not under the lock).
    rounds = await anyio.to_thread.run_sync(_list_round_summaries, sess.repo_path)
    artifacts = await anyio.to_thread.run_sync(_read_artifacts_sync, sess.repo_path)
    with _session_lock:
        _apply_artifacts(sess, artifacts)
        return _session_to_response(sess, rounds)


@app.get("/api/session/{session_id}/validation")
async def get_session_validation(session_id: str) -> dict[str, Any]:
    sess = _require_session(session_id)
    return sess.validation


@app.get("/api/session/{session_id}/config")
async def get_session_config(session_id: str) -> dict[str, Any]:
    sess = _require_session(session_id)
    if sess.config is None:
        raise HTTPException(status_code=404, detail="session not validated; no config available")
    return sess.config


# ---------------------------------------------------------------------------
# Optimization endpoints
# ---------------------------------------------------------------------------


@app.post("/api/session/{session_id}/optimize", status_code=202)
async def start_optimize(session_id: str, req: OptimizeRequest) -> dict[str, str]:
    _require_imports()
    sess = _require_session(session_id)
    # Resolve + validate the optimization mode BEFORE the status transition so
    # an invalid mode (e.g. a typo in harness/config.yaml) returns 400 without
    # leaving the session stuck in "building_baseline". Resolution chain:
    # explicit request > session config > "prompt" default. The mode selects
    # what the optimizer mutates (prompt scaffolds vs agent Python code) and
    # overrides the value on disk for the duration of the run.
    mode = req.mode
    if mode is None and sess.config:
        mode = sess.config.get("mode")
    if mode is None:
        mode = "prompt"
    if mode not in ("prompt", "code"):
        raise HTTPException(
            status_code=400,
            detail=f"invalid optimization mode {mode!r}; expected 'prompt' or 'code'",
        )
    # Check-and-set under the lock: mark in-progress atomically so a
    # concurrent POST gets 409. The lock is released BEFORE any await.
    with _session_lock:
        if sess.status in ("building_baseline", "optimizing"):
            raise HTTPException(status_code=409, detail="optimization is already running")
        if sess.status == "finalized":
            raise HTTPException(status_code=409, detail="session is finalized; cannot optimize")
        if sess.status != "validated":
            raise HTTPException(
                status_code=409,
                detail=f"session is in '{sess.status}' state; must be 'validated'",
            )
        sess.status = "building_baseline"
    # Resolve eval_mode: explicit > config default > None.
    eval_mode = req.eval_mode
    if eval_mode is None and sess.config:
        eval_mode = (sess.config.get("eval") or {}).get("default_mode")
    # B7: _ensure_parent_branch now runs INSIDE the optimization task
    # so a git failure sets status to 'error' instead of leaving the
    # session stuck in 'building_baseline'.
    task = asyncio.create_task(
        _run_optimization_task(
            session_id,
            eval_mode,
            req.max_rounds,
            req.max_turns,
            req.mlflow_experiment_id,
            mode,
        )
    )
    with _session_lock:
        sess._optimize_task = task
    return {"status": "building_baseline"}


@app.get("/api/session/{session_id}/rounds")
async def list_rounds(session_id: str) -> list[dict[str, Any]]:
    sess = _require_session(session_id)
    # B4: file I/O offloaded to a thread pool.
    return await anyio.to_thread.run_sync(_list_round_summaries, sess.repo_path)


@app.get("/api/session/{session_id}/rounds/{round_id}")
async def get_round(session_id: str, round_id: int) -> dict[str, Any]:
    sess = _require_session(session_id)
    path = _round_json_path(sess.repo_path, round_id)
    try:
        return await anyio.to_thread.run_sync(_read_json_file_sync, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"round {round_id} not found") from None
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"round {round_id} JSON is corrupt: {exc}"
        ) from exc


@app.get("/api/session/{session_id}/baseline")
async def get_baseline(session_id: str) -> dict[str, Any]:
    sess = _require_session(session_id)
    path = sess.repo_path / "eval" / "runs" / "baseline.json"
    try:
        return await anyio.to_thread.run_sync(_read_json_file_sync, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="baseline not found") from None
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"baseline JSON is corrupt: {exc}") from exc


@app.get("/api/session/{session_id}/frontier")
async def get_frontier(session_id: str) -> dict[str, Any]:
    _require_imports()
    sess = _require_session(session_id)
    frontier = await anyio.to_thread.run_sync(load_frontier, sess.repo_path)
    if frontier is None:
        raise HTTPException(status_code=404, detail="frontier not found")
    return frontier.to_dict()


# ---------------------------------------------------------------------------
# Finalize endpoints
# ---------------------------------------------------------------------------


@app.post("/api/session/{session_id}/finalize")
async def finalize(session_id: str) -> dict[str, Any]:
    _require_imports()
    sess = _require_session(session_id)
    # B2: FIRST check if finalized.json already exists on disk — if it
    # does, the session is terminal regardless of in-memory status.
    fin_path = sess.repo_path / "eval" / "runs" / "finalized.json"
    if await anyio.to_thread.run_sync(fin_path.is_file):
        with _session_lock:
            if sess.status != "finalized":
                sess.status = "finalized"
        raise HTTPException(status_code=409, detail="session already finalized")
    # B2 + B3: Atomic state transition — only allow finalize from the
    # 'optimized' state. Optimizing/building_baseline/finalizing all get
    # 409 so finalization and rounds never mutate the repo concurrently.
    with _session_lock:
        if sess.status == "finalized":
            raise HTTPException(status_code=409, detail="session already finalized")
        if sess.status in ("optimizing", "building_baseline", "finalizing"):
            raise HTTPException(
                status_code=409,
                detail=f"session is in '{sess.status}' state; must be 'optimized'",
            )
        if sess.status != "optimized":
            raise HTTPException(
                status_code=409,
                detail=f"session is in '{sess.status}' state; must be 'optimized'",
            )
        # Atomically transition to 'finalizing' so a concurrent finalize
        # call gets 409 before the eval starts.
        sess.status = "finalizing"
        sess.error = None
    # Frontier check + finalize run in a thread pool — lock is NOT held
    # across the await so the event loop stays responsive.
    frontier = await anyio.to_thread.run_sync(load_frontier, sess.repo_path)
    if frontier is None:
        # B2: revert to 'optimized' so the user can retry.
        with _session_lock:
            sess.status = "optimized"
        raise HTTPException(
            status_code=409, detail="no frontier; run optimization before finalizing"
        )
    try:
        result = await anyio.to_thread.run_sync(partial(_finalize_sync, sess.repo_path))
    except Exception as exc:  # noqa: BLE001 — surface any finalize failure
        # B2: revert to 'optimized' so the user can retry.
        with _session_lock:
            sess.status = "optimized"
            sess.error = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    with _session_lock:
        sess.finalized = result
        sess.status = "finalized"
    return result


@app.get("/api/session/{session_id}/finalize")
async def get_finalize(session_id: str) -> dict[str, Any]:
    sess = _require_session(session_id)
    path = sess.repo_path / "eval" / "runs" / "finalized.json"
    try:
        return await anyio.to_thread.run_sync(_read_json_file_sync, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="finalized report not found") from None
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"finalized JSON is corrupt: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Conversion endpoints — auto-convert a custom (savesage-style) repo into the
# forge-compatible structure via a managed Omnigent agent. See
# :mod:`anvil.orchestrator.conversion` for the agent flow + PII safety.
# ---------------------------------------------------------------------------


@app.post("/api/session/{session_id}/convert", status_code=202)
async def start_convert(session_id: str, req: ConvertRequest) -> dict[str, Any]:
    """Start a background conversion task. Returns 202 + the initial
    (pending) :class:`ConversionStatus`; poll ``GET /convert`` for progress.

    Guards:

    * 503 when ``OMNIGENT_SERVER_URL`` is not configured (the agent cannot run).
    * 409 when the repo is not convertible (no alternative structures were
      detected by validation) — the UI hides the button in this case, so a 409
      here means the caller bypassed the UI.
    * 409 when a conversion is already running/pending on this session.
    """
    _require_imports()
    sess = _require_session(session_id)
    if not OMNIGENT_SERVER_URL:
        raise HTTPException(
            status_code=503,
            detail="OMNIGENT_SERVER_URL is not configured; the conversion agent cannot run.",
        )
    target_branch = req.target_branch or DEFAULT_TARGET_BRANCH
    with _session_lock:
        if not sess.validation.get("convertible"):
            raise HTTPException(
                status_code=409,
                detail="repo is not convertible — no savesage-style alternative structures detected",
            )
        if sess.conversion is not None and sess.conversion.status in ("running", "pending"):
            raise HTTPException(status_code=409, detail="conversion is already running")
        # Initialize the pollable result before the task starts so the first
        # GET /convert (and the 202 body) sees a pending state.
        sess.conversion = ConversionResult(status="pending", branch_name=target_branch)
    task = asyncio.create_task(_run_conversion_task(session_id, target_branch))
    with _session_lock:
        sess._convert_task = task
        return sess.conversion.to_dict()


@app.get("/api/session/{session_id}/convert")
async def get_convert(session_id: str) -> dict[str, Any]:
    """Return the current :class:`ConversionStatus` for polling.

    404 when no conversion has been started for this session.
    """
    sess = _require_session(session_id)
    with _session_lock:
        if sess.conversion is None:
            raise HTTPException(
                status_code=404, detail="no conversion has been started for this session"
            )
        return sess.conversion.to_dict()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return _DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Dashboard HTML (single-page wizard, XSS-safe via textContent/createElement)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forge Orchestrator</title>
<style>
  :root {
    --bg: #f7f7f8; --card: #fff; --border: #e2e2e4; --text: #1a1a1a;
    --muted: #6b6b6e; --accent: #4f46e5; --pass: #16a34a; --fail: #dc2626;
    --warn: #d97706; --keep: #16a34a; --revert: #dc2626; --noop: #6b6b6e;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1a1a1a; --card: #242426; --border: #3a3a3c; --text: #e8e8e8;
      --muted: #9a9a9d; --accent: #818cf8;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; }
  .wrap { max-width: 760px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 1.4rem; margin: 0 0 8px; }
  .sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 10px; padding: 20px; margin-bottom: 16px; }
  .card h2 { font-size: 1.05rem; margin: 0 0 12px; }
  .card.step { display: none; }
  .card.active { display: block; }
  label { display: block; font-size: 0.85rem; margin-bottom: 4px; font-weight: 600; }
  .hint { display: block; color: var(--muted); font-size: 0.78rem; margin: -6px 0 12px; }
  input, select { width: 100%; padding: 8px 10px; border: 1px solid var(--border);
          border-radius: 6px; background: var(--bg); color: var(--text);
          font-size: 0.9rem; margin-bottom: 12px; }
  button { padding: 9px 16px; border: none; border-radius: 6px; cursor: pointer;
          font-size: 0.9rem; font-weight: 600; background: var(--accent); color: #fff; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button.secondary { background: var(--border); color: var(--text); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
          font-size: 0.75rem; font-weight: 600; }
  .badge.pass { background: var(--pass); color: #fff; }
  .badge.fail { background: var(--fail); color: #fff; }
  .badge.warn { background: var(--warn); color: #fff; }
  .badge.status { background: var(--accent); color: #fff; }
  .check { display: flex; gap: 10px; align-items: flex-start; padding: 8px 0;
          border-bottom: 1px solid var(--border); }
  .check:last-child { border-bottom: none; }
  .check .msg { flex: 1; font-size: 0.88rem; }
  .remediation { background: rgba(220,38,38,0.08); border: 1px solid var(--fail);
          border-radius: 6px; padding: 10px 12px; margin: 6px 0 0;
          font-size: 0.82rem; }
  .config-box { background: var(--bg); border: 1px solid var(--border);
          border-radius: 6px; padding: 12px; font-size: 0.82rem;
          font-family: ui-monospace, monospace; white-space: pre-wrap;
          word-break: break-word; margin-top: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  th { font-weight: 600; color: var(--muted); }
  tr.keep { color: var(--keep); }
  tr.revert { color: var(--revert); }
  tr.noop { color: var(--noop); }
  .error-box { background: rgba(220,38,38,0.1); border: 1px solid var(--fail);
          border-radius: 6px; padding: 12px; color: var(--fail); }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border);
          border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .hidden { display: none !important; }
  .conv-panel { margin-top: 16px; border: 1px solid var(--border);
          border-radius: 8px; padding: 14px; background: var(--bg); }
  .conv-panel h3 { font-size: 0.95rem; margin: 0 0 8px; }
  .conv-step { display: flex; gap: 8px; align-items: flex-start; padding: 4px 0;
          font-size: 0.82rem; border-bottom: 1px solid var(--border); }
  .conv-step:last-child { border-bottom: none; }
  .conv-step .ts { color: var(--muted); font-size: 0.72rem; white-space: nowrap; }
  .conv-step .body { flex: 1; }
  .conv-step .tag { font-weight: 600; color: var(--accent); }
  .link { color: var(--accent); text-decoration: underline; word-break: break-all; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Forge Orchestrator</h1>
  <div class="sub">Two-repo ANVIL optimization workflow</div>

  <div class="card step active" id="step1">
    <h2>Step 1 · Select Agent Repository</h2>
    <label for="repo-url">GitHub repo URL or local path</label>
    <input id="repo-url" type="text" placeholder="https://github.com/user/repo or /path/to/local/repo" autocomplete="off">
    <label for="gh-token">GitHub token (optional, for private repos)</label>
    <input id="gh-token" type="password" placeholder="optional, for private repos" autocomplete="off">
    <button id="btn-validate" onclick="validateRepo()">Validate Repository</button>
    <span id="step1-status"></span>
  </div>

  <div class="card step" id="step2">
    <h2>Step 2 · Compatibility Check</h2>
    <div id="validation-list"></div>
    <div id="validation-config" class="config-box hidden"></div>
    <div id="validation-summary" class="row" style="margin-top:12px"></div>
    <div id="conversion-panel" class="conv-panel hidden"></div>
  </div>

  <div class="card step" id="step3">
    <h2>Step 3 · Configure Optimization</h2>
    <label for="opt-mode">Optimization mode</label>
    <select id="opt-mode">
      <option value="prompt">Prompt</option>
      <option value="code">Code</option>
    </select>
    <small class="hint">prompt = optimize prompt scaffolds; code = optimize agent Python code</small>
    <label for="eval-mode">Eval mode</label>
    <select id="eval-mode"></select>
    <label for="mlflow-experiment-id">MLflow Experiment ID (optional)</label>
    <input id="mlflow-experiment-id" type="text" placeholder="e.g. 967014443183055 — seed baseline from MLflow judge results" autocomplete="off">
    <label for="max-rounds">Max rounds</label>
    <input id="max-rounds" type="number" value="10" min="1" max="200">
    <label for="max-turns">Max turns per round</label>
    <input id="max-turns" type="number" value="30" min="1" max="200">
    <button onclick="startOptimize()">Start Optimization</button>
  </div>

  <div class="card step" id="step4">
    <h2>Step 4 · Optimization Progress</h2>
    <div id="opt-status" class="row" style="margin-bottom:12px"></div>
    <div id="opt-body"></div>
    <div id="opt-error" class="error-box hidden"></div>
    <div id="opt-finalize" class="row" style="margin-top:12px"></div>
  </div>
</div>

<script>
let sessionId = null;
let pollTimer = null;

function show(stepId) {
  document.querySelectorAll('.card.step').forEach(c => c.classList.remove('active'));
  document.getElementById(stepId).classList.add('active');
}

function setBtn(id, text, disabled) {
  const b = document.getElementById(id);
  if (b) { b.textContent = text; b.disabled = disabled; }
}

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

async function validateRepo() {
  const repoUrl = document.getElementById('repo-url').value.trim();
  const token = document.getElementById('gh-token').value.trim() || null;
  if (!repoUrl) { alert('Enter a repo URL'); return; }
  // Reset any prior conversion panel + stop its poll before a fresh validation.
  if (convertTimer) { clearInterval(convertTimer); convertTimer = null; }
  const convPanel = document.getElementById('conversion-panel');
  convPanel.classList.add('hidden');
  convPanel.textContent = '';
  setBtn('btn-validate', 'Validating...', true);
  try {
    const resp = await fetch('/api/session', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ repo_url: repoUrl, github_token: token })
    });
    const data = await resp.json();
    if (resp.status !== 200) {
      setBtn('btn-validate', 'Validate Repository', false);
      alert(data.detail || 'Validation failed');
      return;
    }
    sessionId = data.session_id;
    renderValidation(data.validation);
    renderConfigSummary(data.config);
    const summary = document.getElementById('validation-summary');
    summary.textContent = '';
    if (data.status === 'validated') {
      summary.appendChild(el('span', 'All checks passed. ', null));
      const proceed = el('button', 'Proceed to Optimization', null);
      proceed.onclick = () => { show('step3'); };
      summary.appendChild(proceed);
    } else {
      summary.appendChild(el('span', 'Fix the issues above, then re-validate.', null));
      // The "Convert to forge-compatible" button appears only when the repo
      // failed validation BUT has a recognizable savesage-style alternative
      // structure the auto-converter can transform. Gated on `convertible`,
      // which the POST /api/session response nests inside `validation`.
      if (data.validation && data.validation.convertible === true) {
        const cv = el('button', 'Convert to forge-compatible', null);
        cv.onclick = startConvert;
        summary.appendChild(cv);
      }
      const re = el('button', 'Re-validate', 'secondary');
      re.onclick = () => { show('step1'); setBtn('btn-validate','Validate Repository',false); };
      summary.appendChild(re);
    }
    show('step2');
  } catch (e) {
    alert('Request failed: ' + e.message);
  } finally {
    setBtn('btn-validate', 'Validate Repository', false);
  }
}

function renderValidation(validation) {
  const list = document.getElementById('validation-list');
  list.textContent = '';
  if (!validation || !validation.checks) return;
  validation.checks.forEach(c => {
    const row = el('div', null, 'check');
    row.appendChild(el('span', c.status.toUpperCase(), 'badge ' + c.status));
    const msg = el('div', null, 'msg');
    msg.appendChild(el('div', c.name + ' — ' + c.message, null));
    if (c.remediation) {
      msg.appendChild(el('div', c.remediation, 'remediation'));
    }
    row.appendChild(msg);
    list.appendChild(row);
  });
}

function renderConfigSummary(config) {
  const box = document.getElementById('validation-config');
  box.textContent = '';
  if (!config) { box.classList.add('hidden'); return; }
  const mode = config.mode || '—';
  const rt = config.runtime_endpoint || '—';
  const opt = config.optimizer_endpoint || '—';
  const evalModes = config.eval && config.eval.modes
    ? Object.keys(config.eval.modes).join(', ') : '—';
  const scorers = config.eval && config.eval.scorers
    ? config.eval.scorers.join(', ') : '—';
  box.textContent = 'mode: ' + mode + '\\nruntime_endpoint: ' + rt
    + '\\noptimizer_endpoint: ' + opt + '\\neval modes: ' + evalModes
    + '\\nscorers: ' + scorers;
  box.classList.remove('hidden');
  const sel = document.getElementById('eval-mode');
  sel.textContent = '';
  if (config.eval && config.eval.modes) {
    Object.keys(config.eval.modes).forEach(m => {
      const o = el('option', m, null);
      o.value = m;
      sel.appendChild(o);
    });
  }
  if (config.eval && config.eval.default_mode) {
    sel.value = config.eval.default_mode;
  }
  // Pre-populate the optimization-mode selector from the config's mode.
  if (config.mode) {
    document.getElementById('opt-mode').value = config.mode;
  }
  if (config.loop && config.loop.max_optimizer_turns) {
    document.getElementById('max-turns').value = config.loop.max_optimizer_turns;
  }
}

async function startOptimize() {
  if (!sessionId) { alert('No active session'); return; }
  const evalMode = document.getElementById('eval-mode').value || null;
  const mode = document.getElementById('opt-mode').value || null;
  const mlflowExperimentId = document.getElementById('mlflow-experiment-id').value.trim() || null;
  const maxRounds = parseInt(document.getElementById('max-rounds').value, 10) || 10;
  const maxTurns = parseInt(document.getElementById('max-turns').value, 10) || 30;
  show('step4');
  document.getElementById('opt-status').textContent = '';
  document.getElementById('opt-body').textContent = '';
  document.getElementById('opt-error').classList.add('hidden');
  try {
    const resp = await fetch('/api/session/' + sessionId + '/optimize', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ eval_mode: evalMode, mlflow_experiment_id: mlflowExperimentId, max_rounds: maxRounds, max_turns: maxTurns, mode: mode })
    });
    const data = await resp.json();
    if (resp.status !== 202) {
      document.getElementById('opt-error').textContent = data.detail || 'Failed to start';
      document.getElementById('opt-error').classList.remove('hidden');
      return;
    }
    pollProgress();
  } catch (e) {
    document.getElementById('opt-error').textContent = 'Request failed: ' + e.message;
    document.getElementById('opt-error').classList.remove('hidden');
  }
}

function pollProgress() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(fetchProgress, 3000);
  fetchProgress();
}

async function fetchProgress() {
  if (!sessionId) return;
  try {
    const resp = await fetch('/api/session/' + sessionId);
    const data = await resp.json();
    renderProgress(data);
  } catch (e) {
    // transient; keep polling
  }
}

function renderProgress(data) {
  const status = data.status || '—';
  const statusEl = document.getElementById('opt-status');
  statusEl.textContent = '';
  statusEl.appendChild(el('span', 'Status: ' + status, 'badge status'));
  if (status === 'building_baseline') {
    statusEl.appendChild(el('span', null, 'spinner'));
    statusEl.appendChild(el('span', 'Building baseline...', null));
  } else if (status === 'optimizing') {
    statusEl.appendChild(el('span', null, 'spinner'));
    statusEl.appendChild(el('span', 'Optimizing...', null));
  }

  const body = document.getElementById('opt-body');
  body.textContent = '';
  const errBox = document.getElementById('opt-error');
  if (data.error) {
    errBox.textContent = data.error;
    errBox.classList.remove('hidden');
  } else {
    errBox.classList.add('hidden');
  }

  if (data.rounds && data.rounds.length > 0) {
    const tbl = el('table', null, null);
    const thead = el('thead', null, null);
    const hr = el('tr', null, null);
    ['Round','Decision','Score Δ','Aggregate','Action','Files Changed'].forEach(h => {
      hr.appendChild(el('th', h, null));
    });
    thead.appendChild(hr);
    tbl.appendChild(thead);
    const tbody = el('tbody', null, null);
    data.rounds.forEach(r => {
      const cls = (r.decision || '').toLowerCase();
      const tr = el('tr', null, cls || null);
      tr.appendChild(el('td', String(r.round_id ?? '—'), null));
      tr.appendChild(el('td', r.decision || '—', null));
      const sd = r.score_delta;
      tr.appendChild(el('td', typeof sd === 'number' ? sd.toFixed(4) : '—', null));
      const agg = r.aggregate;
      tr.appendChild(el('td', typeof agg === 'number' ? agg.toFixed(4) : '—', null));
      tr.appendChild(el('td', r.action_kind || '—', null));
      tr.appendChild(el('td', '—', null));
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    body.appendChild(tbl);
  }

  const finDiv = document.getElementById('opt-finalize');
  finDiv.textContent = '';
  if (status === 'optimized') {
    const kept = (data.rounds || []).filter(r => r.decision === 'keep').length;
    const rev = (data.rounds || []).filter(r => r.decision === 'revert').length;
    finDiv.appendChild(el('span',
      'All ' + (data.rounds || []).length + ' rounds complete. ' + kept + ' kept, ' + rev + ' reverted.', null));
    const btn = el('button', 'Finalize', null);
    btn.onclick = doFinalize;
    finDiv.appendChild(btn);
  } else if (status === 'finalized' && data.finalized) {
    renderFinalized(data.finalized, body);
  } else if (status === 'error') {
    const btn = el('button', 'Retry', null);
    btn.onclick = () => { show('step3'); };
    finDiv.appendChild(btn);
  }
}

function renderFinalized(fin, parent) {
  const card = el('div', null, 'config-box');
  card.appendChild(el('div', 'Final Result', null));
  const agg = typeof fin.aggregate === 'number' ? fin.aggregate.toFixed(4) : '—';
  card.appendChild(el('div', 'Aggregate: ' + agg, null));
  if (fin.per_judge) {
    Object.keys(fin.per_judge).forEach(k => {
      const v = fin.per_judge[k];
      card.appendChild(el('div', '  ' + k + ': ' + (typeof v === 'number' ? v.toFixed(4) : v), null));
    });
  }
  if (fin.frontier) {
    card.appendChild(el('div', 'Frontier: ' + JSON.stringify(fin.frontier.best || fin.frontier), null));
  }
  parent.appendChild(card);
}

async function doFinalize() {
  if (!sessionId) return;
  try {
    const resp = await fetch('/api/session/' + sessionId + '/finalize', { method: 'POST' });
    const data = await resp.json();
    if (resp.status !== 200) {
      document.getElementById('opt-error').textContent = data.detail || 'Finalize failed';
      document.getElementById('opt-error').classList.remove('hidden');
      return;
    }
    fetchProgress();
  } catch (e) {
    document.getElementById('opt-error').textContent = 'Request failed: ' + e.message;
    document.getElementById('opt-error').classList.remove('hidden');
  }
}

// ---------------------------------------------------------------------------
// Auto-conversion (Convert to forge-compatible) — spins up an Omnigent agent
// that additively creates the forge files on a new branch, pushes it, then the
// orchestrator re-validates. Polls GET /convert every 3s (same cadence as the
// optimization poll). All XSS-safe: textContent/createElement, no raw HTML.
// ---------------------------------------------------------------------------
async function startConvert() {
  if (!sessionId) { alert('No active session'); return; }
  const panel = document.getElementById('conversion-panel');
  panel.classList.remove('hidden');
  panel.textContent = '';
  panel.appendChild(el('h3', 'Convert to forge-compatible', null));
  panel.appendChild(el('div', 'Starting conversion agent…', 'badge status'));
  try {
    const resp = await fetch('/api/session/' + sessionId + '/convert', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({})
    });
    const data = await resp.json();
    if (resp.status !== 202) {
      panel.textContent = '';
      panel.appendChild(el('h3', 'Convert to forge-compatible', null));
      panel.appendChild(el('div', data.detail || 'Failed to start conversion', 'error-box'));
      return;
    }
    pollConvert();
  } catch (e) {
    panel.textContent = '';
    panel.appendChild(el('h3', 'Convert to forge-compatible', null));
    panel.appendChild(el('div', 'Request failed: ' + e.message, 'error-box'));
  }
}

let convertTimer = null;
function pollConvert() {
  if (convertTimer) clearInterval(convertTimer);
  convertTimer = setInterval(fetchConvert, 3000);
  fetchConvert();
}

async function fetchConvert() {
  if (!sessionId) return;
  try {
    const resp = await fetch('/api/session/' + sessionId + '/convert');
    const data = await resp.json();
    if (resp.status === 404) return;  // no conversion started / cleared
    renderConvert(data);
  } catch (e) {
    // transient fetch error; keep polling
  }
}

function renderConvert(data) {
  const panel = document.getElementById('conversion-panel');
  panel.classList.remove('hidden');
  panel.textContent = '';
  panel.appendChild(el('h3', 'Convert to forge-compatible', null));

  const statusRow = el('div', null, 'row');
  statusRow.appendChild(el('span', 'Status: ' + (data.status || '—'), 'badge status'));
  if (data.status === 'running' || data.status === 'pending') {
    statusRow.appendChild(el('span', null, 'spinner'));
    statusRow.appendChild(el('span', data.status === 'running' ? 'Converting…' : 'Queued…', null));
  }
  panel.appendChild(statusRow);

  if (data.error) {
    panel.appendChild(el('div', data.error, 'error-box'));
  }

  // Agent session link — visible as soon as the managed session is created
  // (while the agent is still working), so the user can open the transcript.
  if (data.session_url) {
    const srow = el('div', null, 'row');
    srow.appendChild(el('span', 'Agent session:', null));
    const slink = el('a', data.session_id || 'view transcript', 'link');
    slink.href = data.session_url;
    slink.target = '_blank';
    slink.rel = 'noopener noreferrer';
    srow.appendChild(slink);
    panel.appendChild(srow);
  }

  if (data.branch_name) {
    const row = el('div', null, 'row');
    row.appendChild(el('span', 'Branch: ' + data.branch_name, null));
    if (data.pr_url) {
      const a = el('a', 'Open PR / compare', 'link');
      a.href = data.pr_url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      row.appendChild(a);
    }
    panel.appendChild(row);
  }

  if (data.progress && data.progress.length) {
    const list = el('div', null, null);
    data.progress.forEach(step => {
      const row = el('div', null, 'conv-step');
      row.appendChild(el('span', step.timestamp || '', 'ts'));
      const body = el('div', null, 'body');
      body.appendChild(el('span', step.step || '', 'tag'));
      body.appendChild(el('span', ' ' + (step.message || ''), null));
      row.appendChild(body);
      list.appendChild(row);
    });
    panel.appendChild(list);
  }

  if (data.revalidation) {
    const sub = el('div', null, 'config-box');
    sub.style.marginTop = '12px';
    sub.appendChild(el('div', 'Re-validation on converted branch:', null));
    sub.appendChild(el('div', 'Status: ' + (data.revalidation.status || '—'), null));
    const pii = data.revalidation.pii_findings;
    if (pii && pii.length) {
      sub.appendChild(el('div', 'PII findings: ' + pii.join('; '), 'remediation'));
    } else if (pii) {
      sub.appendChild(el('div', 'No PII patterns detected in the branch diff.', null));
    }
    if (data.revalidation.checks) {
      data.revalidation.checks.forEach(c => {
        const row = el('div', null, 'check');
        row.appendChild(el('span', c.status.toUpperCase(), 'badge ' + c.status));
        const msg = el('div', null, 'msg');
        msg.appendChild(el('div', c.name + ' — ' + c.message, null));
        if (c.remediation) {
          msg.appendChild(el('div', c.remediation, 'remediation'));
        }
        row.appendChild(msg);
        sub.appendChild(row);
      });
    }
    panel.appendChild(sub);
  }

  // Terminal state: stop polling + offer Re-validate (re-runs the 8 checks on
  // the now-forge-compatible branch the user re-submits) and an Open PR link.
  if (data.status === 'completed' || data.status === 'failed') {
    if (convertTimer) { clearInterval(convertTimer); convertTimer = null; }
    const actions = el('div', null, 'row');
    actions.style.marginTop = '8px';
    const re = el('button', 'Re-validate', null);
    re.onclick = () => {
      panel.classList.add('hidden');
      panel.textContent = '';
      show('step1'); setBtn('btn-validate', 'Validate Repository', false);
    };
    actions.appendChild(re);
    if (data.status === 'completed' && data.pr_url) {
      const open = el('button', 'Open PR / compare', 'secondary');
      open.onclick = () => { window.open(data.pr_url, '_blank', 'noopener'); };
      actions.appendChild(open);
    }
    panel.appendChild(actions);
  }
}
</script>
</body>
</html>
"""
