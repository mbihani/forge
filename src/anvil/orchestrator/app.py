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
    optimized → [POST /finalize] → finalized

Sessions live in an in-memory ``_sessions`` dict guarded by a single
``_session_lock`` (state mutations only). Each session owns one
background optimization task; concurrent sessions are fine but a single
session can only optimize once at a time.

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
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from anvil.eval import evaluate_branch
from anvil.eval.cache import report_to_baseline, save_baseline
from anvil.loop.frontier import load_frontier
from anvil.loop.round import run_round

logger = logging.getLogger("anvil.orchestrator")

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
    max_rounds: int = 10
    max_turns: int = 30
    optimizer_backend: str | None = None


class CheckResult(BaseModel):
    name: str
    status: str  # pass/fail/warn
    message: str
    remediation: str | None = None


class ValidationReport(BaseModel):
    status: str  # valid/invalid
    checks: list[CheckResult]


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
    _optimize_task: Any = field(default=None, repr=False)  # asyncio.Task | None


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
    """
    if _branch_exists(repo_root, _PARENT_BRANCH):
        return
    original = _current_branch(repo_root)
    subprocess.run(
        ["git", "-C", str(repo_root), "checkout", "-b", _PARENT_BRANCH],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if original and original != _PARENT_BRANCH:
        subprocess.run(
            ["git", "-C", str(repo_root), "checkout", original],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )


def _clone_repo(repo_url: str, dest_path: Path, github_token: str | None) -> str | None:
    """Clone ``repo_url`` into ``dest_path``. Return ``None`` on success or
    an error message on failure (run in a thread pool).
    """
    url = repo_url
    if github_token and url.startswith("https://github.com/"):
        url = f"https://x-access-token:{github_token}@github.com/" + url[len("https://github.com/"):]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["git", "clone", url, str(dest_path)],
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


def _check_scaffold_harness_yaml(repo_path: Path) -> dict:
    path = repo_path / "scaffold" / "harness.yaml"
    if not path.is_file():
        return {
            "name": "scaffold_harness_yaml",
            "status": "fail",
            "message": "scaffold/harness.yaml not found",
            "remediation": "Create scaffold/harness.yaml with a 'skills' list and "
            "'sampling' dict. Each skill has a 'file' field pointing to a markdown "
            "file in scaffold/. Decompose your agent's prompt into sections — each "
            "section becomes one skill. Example from the Savesage ICICI integration: "
            "skills = [identity.md, transaction_rules.md, rewards_rules.md, "
            "missing_data.md, edge_cases.md, icici_bank_rules.md, "
            "icici_card_identity.md, icici_rewards_layouts.md]",
        }
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
            "'sampling' dict.",
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
        if not (repo_path / "scaffold" / rel).is_file():
            return {
                "name": "scaffold_skill_files",
                "status": "fail",
                "message": f"Skill file '{rel}' referenced in harness.yaml not found in scaffold/",
                "remediation": "Create the missing skill markdown file in scaffold/. "
                "Each skill file is a section of the agent's system prompt.",
            }
    return {
        "name": "scaffold_skill_files",
        "status": "pass",
        "message": "All skill files present.",
    }


def _check_golden_set_jsonl(repo_path: Path) -> dict:
    path = repo_path / "data" / "golden_set.jsonl"
    if not path.is_file():
        return {
            "name": "golden_set_jsonl",
            "status": "fail",
            "message": "data/golden_set.jsonl not found",
            "remediation": "Create data/golden_set.jsonl with test examples. Each line is a "
            "JSON object with at minimum: example_id (unique string), query or input "
            "(the user's request), category (classification like "
            "direct/multi_hop/distractor/out_of_scope), and expected or expectations "
            "(the ground truth answer).",
        }
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


def _check_harness_config_yaml(repo_path: Path) -> tuple[dict, dict | None]:
    """Return (check_result, parsed_config_dict_or_None)."""
    path = repo_path / "harness" / "config.yaml"
    if not path.is_file():
        return (
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


def _check_agent_code(repo_path: Path, config: dict) -> dict:
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
        return {
            "name": "agent_code",
            "status": "fail",
            "message": "mode is 'code' but no Python files found in agents/",
            "remediation": "For code mode, create a Python file in agents/ with a class "
            "implementing a predict() method.",
        }
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


def _run_validation(repo_path: Path) -> tuple[dict, dict | None]:
    """Run all validation checks. Return (ValidationReport_dict, config_or_None).

    The config is returned (unredacted) so the session can store a redacted
    copy; ``None`` when the config check failed.
    """
    checks: list[dict] = []
    checks.append(_check_git_repo(repo_path))
    checks.append(_check_scaffold_harness_yaml(repo_path))

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
    checks.append(_check_golden_set_jsonl(repo_path))
    config_check, config = _check_harness_config_yaml(repo_path)
    checks.append(config_check)
    if config is not None:
        checks.append(_check_eval_modes(config))
        checks.append(_check_agent_code(repo_path, config))
    checks.append(_check_parent_branch(repo_path))

    any_fail = any(c["status"] == "fail" for c in checks)
    return (
        {"status": "invalid" if any_fail else "valid", "checks": checks},
        config,
    )


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


# ---------------------------------------------------------------------------
# Synchronous baseline + finalize (run in thread pool)
# ---------------------------------------------------------------------------


def _build_baseline_sync(repo_path: Path, eval_mode: str | None) -> dict:
    """Run eval on the current scaffold and write eval/runs/baseline.json.

    Mirrors ``scripts/make_baseline.build_baseline``: calls
    ``evaluate_branch``, reads endpoints from harness/config.yaml, gets the
    scaffold commit SHA from git, converts the EvalReport to a
    CachedBaseline via ``report_to_baseline``, and persists it.
    """
    scaffold_root = repo_path / "scaffold"
    config_path = repo_path / "harness" / "config.yaml"
    runtime_endpoint = ""
    judge_endpoint = ""
    if config_path.is_file():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        runtime_endpoint = raw.get("runtime_endpoint", "")
        judge_endpoint = raw.get("judge_endpoint", "")
    report = evaluate_branch(
        scaffold_root=scaffold_root,
        runtime_config_path=config_path if config_path.is_file() else None,
        mode=eval_mode,
    )
    baseline = report_to_baseline(
        report,
        scaffold_commit_sha=_git_head_sha(repo_path),
        runtime_endpoint=runtime_endpoint,
        judge_endpoint=judge_endpoint,
    )
    save_baseline(repo_path, baseline)
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


def _session_to_response(sess: SessionData) -> dict[str, Any]:
    """Build the JSON-serializable session response dict.

    Reads the live round/baseline/frontier/finalized state from disk so a
    polling client sees progress without depending on the background task
    having updated the in-memory copy yet.
    """
    return {
        "session_id": sess.session_id,
        "repo_url": sess.repo_url,
        "status": sess.status,
        "validation": sess.validation,
        "config": sess.config,
        "baseline": sess.baseline,
        "rounds": _list_round_summaries(sess.repo_path),
        "frontier": sess.frontier,
        "finalized": sess.finalized,
        "error": sess.error,
    }


def _refresh_session_artifacts(sess: SessionData) -> None:
    """Refresh frontier/finalized/baseline from disk into the session."""
    frontier = load_frontier(sess.repo_path)
    sess.frontier = frontier.to_dict() if frontier else None
    fin_path = sess.repo_path / "eval" / "runs" / "finalized.json"
    if fin_path.is_file():
        with suppress(json.JSONDecodeError, OSError):
            sess.finalized = json.loads(fin_path.read_text(encoding="utf-8"))
    base_path = sess.repo_path / "eval" / "runs" / "baseline.json"
    if base_path.is_file():
        with suppress(json.JSONDecodeError, OSError):
            sess.baseline = json.loads(base_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Async background optimization task
# ---------------------------------------------------------------------------


async def _run_optimization_task(
    session_id: str, eval_mode: str | None, max_rounds: int, max_turns: int
) -> None:
    """Background asyncio task that runs baseline + rounds.

    Each blocking step (baseline, each round) runs via
    ``anyio.to_thread.run_sync`` so the event loop stays responsive and a
    polling client sees the status transition building_baseline →
    optimizing → optimized, with rounds appearing one at a time.
    """
    try:
        sess = _get_session(session_id)
        if sess is None:
            return
        # Baseline (blocking) in a thread pool.
        baseline = await anyio.to_thread.run_sync(
            partial(_build_baseline_sync, sess.repo_path, eval_mode)
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
                )
            )
            path = _round_json_path(sess.repo_path, i)
            if path.is_file():
                try:
                    round_data = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    round_data = None
                if round_data is not None:
                    with _session_lock:
                        sess.rounds.append(round_data)
            # Stop if a finalized.json appeared (a round triggered finalization).
            if (sess.repo_path / "eval" / "runs" / "finalized.json").is_file():
                break
        with _session_lock:
            _refresh_session_artifacts(sess)
            sess.status = "finalized" if sess.finalized else "optimized"
    except Exception as exc:  # noqa: BLE001 — surface any failure
        with _session_lock:
            sess = _get_session(session_id)
            if sess is not None:
                sess.status = "error"
                sess.error = str(exc)
        logger.exception("optimization task for session %s failed", session_id)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("forge orchestrator startup — sessions root: %s", _SESSIONS_ROOT)
    yield


app = FastAPI(title="Forge Orchestrator", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session")
async def create_session(req: CreateSessionRequest) -> dict[str, Any]:
    """Clone the agent repo, validate it, return a new session."""
    session_id = uuid.uuid4().hex[:12]
    dest_path = _SESSIONS_ROOT / session_id

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
    )
    with _session_lock:
        _sessions[session_id] = sess

    # Clone (blocking) in a thread pool.
    err = await anyio.to_thread.run_sync(
        partial(_clone_repo, req.repo_url, dest_path, req.github_token)
    )
    if err is not None:
        with _session_lock:
            sess.status = "invalid"
            sess.error = err
        raise HTTPException(status_code=400, detail=err)

    with _session_lock:
        sess.status = "validating"

    # Validate (file I/O + git) in a thread pool.
    report, config = await anyio.to_thread.run_sync(partial(_run_validation, dest_path))
    with _session_lock:
        sess.validation = report
        sess.status = "validated" if report["status"] == "valid" else "invalid"
        if config is not None:
            sess.config = _redact_secrets(config)

    return {
        "session_id": session_id,
        "status": sess.status,
        "validation": report,
        "config": sess.config,
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    sess = _require_session(session_id)
    with _session_lock:
        _refresh_session_artifacts(sess)
        return _session_to_response(sess)


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
    sess = _require_session(session_id)
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
    # Create the parent branch (blocking git) in a thread pool — outside
    # the lock so the event loop stays responsive.
    await anyio.to_thread.run_sync(partial(_ensure_parent_branch, sess.repo_path))
    task = asyncio.create_task(
        _run_optimization_task(session_id, eval_mode, req.max_rounds, req.max_turns)
    )
    with _session_lock:
        sess._optimize_task = task
    return {"status": "building_baseline"}


@app.get("/api/session/{session_id}/rounds")
async def list_rounds(session_id: str) -> list[dict[str, Any]]:
    sess = _require_session(session_id)
    return _list_round_summaries(sess.repo_path)


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
    sess = _require_session(session_id)
    with _session_lock:
        if sess.status == "finalized":
            raise HTTPException(status_code=409, detail="session already finalized")
        if sess.status not in ("optimized", "optimizing"):
            raise HTTPException(
                status_code=409,
                detail=f"session is in '{sess.status}' state; must be 'optimized'",
            )
    # Frontier check + finalize run in a thread pool — lock is NOT held
    # across the await so the event loop stays responsive.
    frontier = await anyio.to_thread.run_sync(load_frontier, sess.repo_path)
    if frontier is None:
        raise HTTPException(
            status_code=409, detail="no frontier; run optimization before finalizing"
        )
    try:
        result = await anyio.to_thread.run_sync(partial(_finalize_sync, sess.repo_path))
    except Exception as exc:  # noqa: BLE001 — surface any finalize failure
        with _session_lock:
            sess.status = "error"
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
</style>
</head>
<body>
<div class="wrap">
  <h1>Forge Orchestrator</h1>
  <div class="sub">Two-repo ANVIL optimization workflow</div>

  <div class="card step active" id="step1">
    <h2>Step 1 · Select Agent Repository</h2>
    <label for="repo-url">GitHub repo URL</label>
    <input id="repo-url" type="text" placeholder="https://github.com/user/savesage-agent" autocomplete="off">
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
  </div>

  <div class="card step" id="step3">
    <h2>Step 3 · Configure Optimization</h2>
    <label for="eval-mode">Eval mode</label>
    <select id="eval-mode"></select>
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
  if (config.loop && config.loop.max_optimizer_turns) {
    document.getElementById('max-turns').value = config.loop.max_optimizer_turns;
  }
}

async function startOptimize() {
  if (!sessionId) { alert('No active session'); return; }
  const evalMode = document.getElementById('eval-mode').value || null;
  const maxRounds = parseInt(document.getElementById('max-rounds').value, 10) || 10;
  const maxTurns = parseInt(document.getElementById('max-turns').value, 10) || 30;
  show('step4');
  document.getElementById('opt-status').textContent = '';
  document.getElementById('opt-body').textContent = '';
  document.getElementById('opt-error').classList.add('hidden');
  try {
    const resp = await fetch('/api/session/' + sessionId + '/optimize', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ eval_mode: evalMode, max_rounds: maxRounds, max_turns: maxTurns })
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
</script>
</body>
</html>
"""
