"""Forge Orchestrator FastAPI app — HTTP control plane for the full ANVIL optimization workflow.

Serves all four phases of the optimization workflow:

Phase 0 — workflow state:
  * GET  /api/state                 — status of all four phases.

Phase 1 — setup:
  * GET  /api/scaffold              — scaffold config + skill/rule file list.
  * PUT  /api/scaffold              — write scaffold/harness.yaml.
  * GET  /api/scaffold/files/{name} — raw text of a skill/rule markdown file.
  * PUT  /api/scaffold/files/{name} — write a skill/rule markdown file.
  * GET  /api/golden-set            — golden dataset stats + sample entries.
  * GET  /api/config                — harness runtime config as JSON (secrets redacted).
  * PUT  /api/config                — update harness/config.yaml (allowed fields only).

Phase 2 — baseline:
  * POST /api/baseline              — build baseline (eval → CachedBaseline → baseline.json).
  * GET  /api/baseline              — read cached baseline.

Phase 3 — optimize:
  * POST /rounds                    — start a new optimization round (existing, kept).
  * GET  /rounds                    — list completed rounds (existing, kept).
  * GET  /rounds/{id}               — read a single round JSON (existing, kept).
  * GET  /api/frontier              — read eval/runs/frontier.json.

Phase 4 — finalize:
  * POST /api/finalize              — run held-out eval → eval/runs/finalized.json (terminal).
  * GET  /api/finalize              — read finalized report.

Supporting:
  * GET  /health       — liveness probe (required by deploy/app.yaml).
  * GET  /agents       — list agent YAML bundles from ``agents/``.
  * POST /validate     — read-only baseline eval (diagnostic).
  * GET  /             — interactive 4-phase wizard dashboard.

Security: ``GET /api/config`` redacts any field whose key contains
``token``, ``secret``, ``password``, or ``credential`` (case-insensitive)
before returning the config JSON.

Terminal finalization: once ``eval/runs/finalized.json`` exists,
``POST /api/finalize`` returns 409 and ``POST /rounds`` returns 409
(unless ``force: true`` is passed in the request body).

Shared mutation lock: a single ``_mutation_lock`` serializes ALL
mutating operations (``POST /rounds``, ``POST /api/baseline``,
``POST /api/finalize``, ``PUT /api/scaffold``,
``PUT /api/scaffold/files/{name}``, ``PUT /api/config``) so concurrent
requests don't race on shared scaffold/config/eval artifacts.

``run_round``, ``evaluate_branch``, ``_build_baseline_sync``,
``_finalize_sync`` and all synchronous file I/O MUST run in a thread pool
(``anyio.to_thread.run_sync``) so the single uvicorn event loop stays
responsive — calling them directly in an async handler freezes the loop
and the Databricks App 502s.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from functools import partial
from html import escape
from pathlib import Path
from typing import Any, Literal

import anyio
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from anvil.eval import evaluate_branch
from anvil.eval.cache import (
    CachedBaseline,
    load_baseline,
    report_to_baseline,
    save_baseline,
)
from anvil.loop.decision import Decision
from anvil.loop.frontier import load_frontier
from anvil.loop.round import RoundReport, run_round

logger = logging.getLogger("anvil.orchestrator")

# Env vars passed by deploy/databricks.yml — logged at startup for
# debugging. Most are consumed by run_round() / harness/config.yaml,
# not by this app directly.
_ENV_VARS = [
    "OMNIGENT_SERVER_URL",
    "OMNIGENT_AUTH_TOKEN",
    "GIT_TOKEN",
    "GIT_REMOTE_URL",
    "ANVIL_AI_GATEWAY_URL",
    "ANVIL_EVAL_ENGINE",
    "MLFLOW_EXPERIMENT_NAME",
    "ANVIL_OPTIMIZER_MODEL",
    "ANVIL_DOMAIN_CONFIG",
]

# Env vars whose raw values are credentials — must never be logged.
_SECRET_ENV_VARS = frozenset({"OMNIGENT_AUTH_TOKEN", "GIT_TOKEN"})

# Valid decision values (for sanitizing persisted round JSON before
# it reaches the HTML dashboard or API responses).
_VALID_DECISIONS = frozenset(d.value for d in Decision)

# Allowed filename pattern for scaffold skill/rule markdown files.
# Only alphanumeric + dash + underscore + .md extension. Rejects path
# traversal (/, ..) and dot-prefixed names.
_SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_-]+\.md$")

# Substrings that mark a config field as secret — any key containing
# one of these (case-insensitive) is redacted by ``_redact_secrets``
# before the config JSON is returned to the client.
_REDACT_KEYWORDS = frozenset({"token", "secret", "password", "credential"})

# Top-level config fields the PUT /api/config endpoint may update.
_CONFIG_ALLOWED_TOP = frozenset({
    "mode",
    "runtime_endpoint",
    "optimizer_endpoint",
    "judge_endpoint",
})

# Nested config sections + the sub-fields the PUT /api/config endpoint
# may update within each.
_CONFIG_ALLOWED_NESTED: dict[str, frozenset[str]] = {
    "optimizer": frozenset({"backend", "server_url", "auth_token", "agent_bundle_path"}),
    "eval": frozenset({"default_mode", "n_workers", "scorers", "held_out_test"}),
    "gate": frozenset({"type", "epsilon"}),
    "loop": frozenset({"target_rounds", "max_optimizer_turns"}),
}

# ---------------------------------------------------------------------------
# Module-level config — repo_root
# ---------------------------------------------------------------------------

# Default: the repo root that contains this source tree. For local dev
# that's the worktree root; for a Databricks App it's the deployed app
# root (source_code_path: .. in the bundle). Tests override via
# ``set_repo_root()`` to point at a tmpdir.
_repo_root: Path = Path(__file__).resolve().parents[3]


def get_repo_root() -> Path:
    """Return the repo root the orchestrator operates on."""
    return _repo_root


def set_repo_root(path: Path | str) -> None:
    """Override the repo root (used by tests to point at a tmpdir)."""
    global _repo_root
    _repo_root = Path(path).resolve()


# ---------------------------------------------------------------------------
# Round-id detection (mirrors scripts/run_round.py:_next_round_id)
# ---------------------------------------------------------------------------


def _next_round_id(repo_root: Path) -> int:
    """Highest existing eval/runs/round_NNN.json + 1, or 1 if none."""
    runs_dir = repo_root / "eval" / "runs"
    if not runs_dir.is_dir():
        return 1
    nums: list[int] = []
    for p in runs_dir.glob("round_*.json"):
        m = re.search(r"round_(\d+)\.json$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RoundCreateRequest(BaseModel):
    round_id: int | None = None
    eval_mode: Literal["quick", "standard", "full"] | None = None
    max_turns: int | None = None
    mode: Literal["prompt", "code"] | None = None
    agent: str | None = None
    force: bool = False


class RoundReportResponse(BaseModel):
    round_id: int
    branch: str
    decision: str
    action_kind: str
    parse_status: str
    diff_summary: str
    mode: str = "prompt"
    files_added: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    files_removed: list[str] = Field(default_factory=list)
    baseline_score: float | None = None
    mutated_score: float | None = None
    score_delta: float | None = None
    eval_run_id: str | None = None
    git_commit_sha: str | None = None
    notes: str = ""


class RoundSummaryResponse(BaseModel):
    round_id: int | None = None
    decision: str | None = None
    action_kind: str | None = None
    baseline_score: float | None = None
    score_delta: float | None = None
    aggregate: float | None = None


class ValidationResponse(BaseModel):
    aggregate: float
    mode: str = ""
    n_examples: int = 0
    run_id: str = ""


class AgentInfo(BaseModel):
    name: str
    filename: str
    path: str


class ScaffoldFileEntry(BaseModel):
    """One skill or rule markdown file in scaffold/."""
    name: str
    type: str  # "skills" or "rules"
    path: str


class ScaffoldResponse(BaseModel):
    """Response for GET /api/scaffold — scaffold config + file list."""
    config: dict[str, Any]
    files: list[ScaffoldFileEntry]


class GoldenSetResponse(BaseModel):
    """Response for GET /api/golden-set — stats + sample entries."""
    total: int
    buckets: dict[str, int]
    samples: list[dict[str, Any]]


class ConfigUpdateRequest(BaseModel):
    """Body for PUT /api/config — partial config update (allowed fields only).

    Top-level: ``mode``, ``runtime_endpoint``, ``optimizer_endpoint``,
    ``judge_endpoint``. Nested sections: ``optimizer``, ``eval``,
    ``gate``, ``loop`` — each accepts a dict whose sub-fields are
    allow-listed by ``_CONFIG_ALLOWED_NESTED``.
    """
    mode: str | None = None
    runtime_endpoint: str | None = None
    optimizer_endpoint: str | None = None
    judge_endpoint: str | None = None
    optimizer: dict[str, Any] | None = None
    eval: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    loop: dict[str, Any] | None = None


def _round_report_to_response(report: RoundReport) -> RoundReportResponse:
    d = dataclasses.asdict(report)
    d["decision"] = report.decision.value
    return RoundReportResponse(**d)


# ---------------------------------------------------------------------------
# Round JSON helpers
# ---------------------------------------------------------------------------


def _round_json_path(repo_root: Path, round_id: int) -> Path:
    return repo_root / "eval" / "runs" / f"round_{round_id:03d}.json"


def _list_round_summaries(repo_root: Path) -> list[dict[str, Any]]:
    """Scan eval/runs/round_*.json and return summaries sorted by round_id.

    Fields read from untrusted persisted JSON are validated: ``decision``
    must be a known :class:`Decision` value, ``round_id`` is coerced to
    ``int``. Arbitrary keys are never forwarded to the dashboard or API.
    """
    runs_dir = repo_root / "eval" / "runs"
    if not runs_dir.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for p in sorted(runs_dir.glob("round_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # Validate decision — must be a known Decision value.
        decision = data.get("decision")
        if decision not in _VALID_DECISIONS:
            decision = None
        # Coerce round_id to int — don't trust arbitrary JSON values.
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


# ---------------------------------------------------------------------------
# Agent + harness-config helpers
# ---------------------------------------------------------------------------


def _harness_config_path(repo_root: Path) -> Path:
    """Path to the immutable harness runtime config (``harness/config.yaml``).

    Mirrors :func:`anvil.runtime.loader.default_runtime_config_path` —
    ``scaffold_root.parent / harness / config.yaml`` — which resolves to
    ``repo_root / harness / config.yaml`` for the default scaffold layout.
    """
    return repo_root / "harness" / "config.yaml"


def _update_harness_config(
    repo_root: Path, *, mode: str | None = None, agent: str | None = None
) -> None:
    """Update ``mode`` / ``optimizer.agent_bundle_path`` in harness/config.yaml.

    Reads the YAML, updates only the requested keys, writes it back. Called
    before ``run_round`` so the optimizer loop picks up the new values.
    Comments are not preserved by ``yaml.safe_dump`` — acceptable here because
    the orchestrator is the authorized control surface for these fields and
    the full commented template lives in version control.
    """
    path = _harness_config_path(repo_root)
    if not path.is_file():
        return
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    changed = False
    if mode is not None and raw.get("mode") != mode:
        raw["mode"] = mode
        changed = True
    if agent is not None:
        optimizer = raw.get("optimizer") or {}
        if optimizer.get("agent_bundle_path") != agent:
            optimizer["agent_bundle_path"] = agent
            raw["optimizer"] = optimizer
            changed = True
    if changed:
        _atomic_write(
            path,
            yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
        )


def _list_agents(repo_root: Path) -> list[dict[str, str]]:
    """List agent YAML files from ``agents/*.yaml``.

    Returns ``[{name, filename, path}]`` sorted by filename. ``name`` is read
    from the YAML's top-level ``name:`` field; falls back to the file stem
    when the file has no name field or fails to parse.
    """
    agents_dir = repo_root / "agents"
    if not agents_dir.is_dir():
        return []
    result: list[dict[str, str]] = []
    for p in sorted(agents_dir.glob("*.yaml")):
        name = p.stem
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict) and raw.get("name"):
                name = str(raw["name"])
        except (yaml.YAMLError, OSError):
            pass
        result.append(
            {
                "name": name,
                "filename": p.name,
                "path": p.relative_to(repo_root).as_posix(),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Scaffold helpers (Phase 1)
# ---------------------------------------------------------------------------


def _scaffold_dir(repo_root: Path) -> Path:
    """Path to the scaffold directory."""
    return repo_root / "scaffold"


def _scaffold_config_path(repo_root: Path) -> Path:
    """Path to scaffold/harness.yaml (the mutable optimizer config)."""
    return repo_root / "scaffold" / "harness.yaml"


def _list_scaffold_files(repo_root: Path) -> list[dict[str, str]]:
    """List skill + rule markdown files from scaffold/skills/ and scaffold/rules/.

    Returns ``[{name, type, path}]`` sorted by type then name. ``type`` is
    ``"skills"`` or ``"rules"`` (matching the subdirectory name).
    """
    result: list[dict[str, str]] = []
    scaffold = _scaffold_dir(repo_root)
    for subdir in ("skills", "rules"):
        d = scaffold / subdir
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            result.append(
                {
                    "name": p.name,
                    "type": subdir,
                    "path": p.relative_to(repo_root).as_posix(),
                }
            )
    return result


def _find_scaffold_file(repo_root: Path, filename: str) -> Path | None:
    """Search scaffold/skills/ and scaffold/rules/ for ``filename``.

    Returns the first match, or ``None`` if not found.
    """
    for subdir in ("skills", "rules"):
        path = _scaffold_dir(repo_root) / subdir / filename
        if path.is_file():
            return path
    return None


def _scaffold_file_write_path(repo_root: Path, filename: str) -> Path:
    """Determine where to write a scaffold file.

    If the file already exists in skills/ or rules/, overwrite it in place.
    Otherwise default to scaffold/skills/ (the common editable case).
    """
    for subdir in ("skills", "rules"):
        path = _scaffold_dir(repo_root) / subdir / filename
        if path.is_file():
            return path
    return _scaffold_dir(repo_root) / "skills" / filename


def _validate_scaffold_filename(filename: str) -> str:
    """Validate a scaffold-file filename for safe filesystem access.

    Only allows ``[a-zA-Z0-9_-]+.md`` — rejects path traversal (``/``,
    ``..``), dot-prefixed names, and non-markdown extensions.
    """
    if not _SAFE_FILENAME_RE.match(filename):
        raise HTTPException(
            status_code=400,
            detail=f"invalid filename: {filename!r} — must be [a-zA-Z0-9_-]+.md",
        )
    return filename


# ---------------------------------------------------------------------------
# Golden set helpers (Phase 1)
# ---------------------------------------------------------------------------


def _golden_set_path(repo_root: Path) -> Path:
    return repo_root / "data" / "golden_set.jsonl"


def _golden_set_stats(repo_root: Path) -> dict[str, Any]:
    """Read data/golden_set.jsonl and return stats + first 5 samples.

    Returns ``{"total": int, "buckets": {category: count}, "samples": [...]}``.
    Handles a missing file gracefully (returns empty stats).
    """
    path = _golden_set_path(repo_root)
    if not path.is_file():
        return {"total": 0, "buckets": {}, "samples": []}
    examples: list[dict[str, Any]] = []
    buckets: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ex, dict):
            continue
        examples.append(ex)
        bucket = ex.get("category") or ex.get("bucket") or "unknown"
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return {
        "total": len(examples),
        "buckets": buckets,
        "samples": examples[:5],
    }


# ---------------------------------------------------------------------------
# Config merge helper (Phase 1 — PUT /api/config)
# ---------------------------------------------------------------------------


def _merge_config(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Merge ``update`` into ``existing`` config, only touching allowed fields.

    Top-level: ``mode``, ``runtime_endpoint``, ``optimizer_endpoint``,
    ``judge_endpoint``.
    Nested sections (only listed sub-fields are updated):
      ``optimizer`` → ``backend``, ``server_url``, ``auth_token``,
      ``agent_bundle_path``.
      ``eval`` → ``default_mode``, ``n_workers``, ``scorers``,
      ``held_out_test``.
      ``gate`` → ``type``, ``epsilon``.
      ``loop`` → ``target_rounds``, ``max_optimizer_turns``.
    """
    result = dict(existing)
    for key in _CONFIG_ALLOWED_TOP:
        if key in update:
            result[key] = update[key]
    for section, allowed in _CONFIG_ALLOWED_NESTED.items():
        section_update = update.get(section)
        if isinstance(section_update, dict):
            current = dict(result.get(section) or {})
            for field in allowed:
                if field in section_update:
                    current[field] = section_update[field]
            result[section] = current
    return result


# ---------------------------------------------------------------------------
# Secret redaction (GET /api/config)
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
# Atomic write helper
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via a temp file + ``os.replace``.

    The temp file is created in the same directory as the target so the
    rename is guaranteed to be atomic (same filesystem). If any step
    fails the temp file is cleaned up and the original is left intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, str(path))
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


def _read_json_file_sync(path: Path) -> dict[str, Any]:
    """Read a JSON file synchronously, raising FileNotFoundError if absent."""
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Sync I/O helpers for GET endpoints (run in thread pool)
# ---------------------------------------------------------------------------


def _get_scaffold_sync(repo_root: Path) -> dict[str, Any]:
    """Read scaffold config + list skill/rule files (sync I/O)."""
    config_path = _scaffold_config_path(repo_root)
    if not config_path.is_file():
        raise FileNotFoundError("scaffold/harness.yaml not found")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    files = _list_scaffold_files(repo_root)
    return {"config": config, "files": files}


def _get_config_sync(repo_root: Path) -> dict[str, Any]:
    """Read harness config + redact secrets (sync I/O)."""
    path = _harness_config_path(repo_root)
    if not path.is_file():
        raise FileNotFoundError("harness/config.yaml not found")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _redact_secrets(raw)


def _put_config_sync(repo_root: Path, update: dict[str, Any]) -> None:
    """Merge + atomically write harness config (sync I/O)."""
    path = _harness_config_path(repo_root)
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = _merge_config(existing, update)
    _atomic_write(
        path, yaml.safe_dump(merged, sort_keys=False, default_flow_style=False)
    )


# ---------------------------------------------------------------------------
# Git SHA helper
# ---------------------------------------------------------------------------


def _git_head_sha(repo_root: Path) -> str:
    """``git rev-parse HEAD`` of the repo (falls back to 'unknown').

    Defensive: returns ``"unknown"`` if git is unavailable, the repo has
    no commits, or the call times out. Mirrors the scripts' defensive SHA
    lookup.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


# ---------------------------------------------------------------------------
# Baseline builder (Phase 2 — POST /api/baseline)
# ---------------------------------------------------------------------------


def _build_baseline_sync(repo_root: Path) -> CachedBaseline:
    """Run the eval on the current scaffold and build a CachedBaseline.

    Mirrors ``scripts/make_baseline.build_baseline``: calls
    ``evaluate_branch``, reads endpoints from harness/config.yaml, gets
    the scaffold commit SHA from git, converts the EvalReport to a
    CachedBaseline via ``report_to_baseline``, and persists it via
    ``save_baseline``.
    """
    scaffold_root = repo_root / "scaffold"
    config_path = _harness_config_path(repo_root)
    runtime_endpoint = ""
    judge_endpoint = ""
    if config_path.is_file():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        runtime_endpoint = raw.get("runtime_endpoint", "")
        judge_endpoint = raw.get("judge_endpoint", "")
    report = evaluate_branch(
        scaffold_root=scaffold_root,
        runtime_config_path=config_path if config_path.is_file() else None,
    )
    baseline = report_to_baseline(
        report,
        scaffold_commit_sha=_git_head_sha(repo_root),
        runtime_endpoint=runtime_endpoint,
        judge_endpoint=judge_endpoint,
    )
    save_baseline(repo_root, baseline)
    return baseline


# ---------------------------------------------------------------------------
# Finalize helper (Phase 4 — POST /api/finalize)
# ---------------------------------------------------------------------------


def _finalize_sync(repo_root: Path) -> dict[str, Any]:
    """Evaluate HEAD on the held-out set and return the finalized payload.

    Mirrors ``scripts/finalize.finalize``: checks that held-out test is
    enabled and the frontier exists, runs ``evaluate_branch`` with
    ``mode="test"`` and ``allow_test=True``, and writes the result to
    ``eval/runs/finalized.json``.
    """
    config_path = _harness_config_path(repo_root)
    if not config_path.is_file():
        raise RuntimeError(
            "harness/config.yaml not found; cannot finalize without runtime config"
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    eval_config = raw.get("eval") or {}
    if not eval_config.get("held_out_test"):
        raise RuntimeError(
            "held-out finalization is disabled; set eval.held_out_test: true"
        )

    frontier = load_frontier(repo_root)
    if frontier is None:
        raise RuntimeError("cannot finalize without eval/runs/frontier.json")

    scaffold_root = repo_root / "scaffold"
    report = evaluate_branch(
        scaffold_root=scaffold_root,
        runtime_config_path=config_path,
        mode="test",
        allow_test=True,
    )

    payload = {
        **dataclasses.asdict(report),
        "scaffold_commit_sha": _git_head_sha(repo_root),
        "finalized_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "frontier": frontier.to_dict(),
    }

    finalized_path = repo_root / "eval" / "runs" / "finalized.json"
    _atomic_write(finalized_path, json.dumps(payload, indent=2) + "\n")
    return payload


# ---------------------------------------------------------------------------
# Workflow state helper (Phase 0 — GET /api/state)
# ---------------------------------------------------------------------------


def _workflow_state(repo_root: Path) -> dict[str, Any]:
    """Return the status of all four workflow phases.

    Each file read + YAML/JSON parse is individually wrapped in
    try/except so a single malformed file never crashes the whole
    state response — the affected phase just reports defaults.
    """
    # Phase 1: Scaffold
    scaffold_path = _scaffold_config_path(repo_root)
    scaffold_exists = scaffold_path.is_file()
    skills_count = 0
    rules_count = 0
    if scaffold_exists:
        try:
            raw = yaml.safe_load(scaffold_path.read_text(encoding="utf-8")) or {}
            skills_count = len(raw.get("skills") or [])
            rules_count = len(raw.get("rules") or [])
        except (yaml.YAMLError, OSError):
            pass

    # Phase 1: Golden set
    golden_path = _golden_set_path(repo_root)
    golden_exists = golden_path.is_file()
    golden_count = 0
    golden_buckets: dict[str, int] = {}
    if golden_exists:
        try:
            for line in golden_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(ex, dict):
                    continue
                golden_count += 1
                bucket = ex.get("category") or ex.get("bucket") or "unknown"
                golden_buckets[bucket] = golden_buckets.get(bucket, 0) + 1
        except (json.JSONDecodeError, OSError):
            pass

    # Phase 1: Config
    config_path = _harness_config_path(repo_root)
    config_exists = config_path.is_file()
    config_mode = ""
    config_backend = ""
    if config_exists:
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            config_mode = str(raw.get("mode", ""))
            config_backend = str((raw.get("optimizer") or {}).get("backend", ""))
        except (yaml.YAMLError, OSError):
            pass

    # Phase 2: Baseline
    baseline: CachedBaseline | None = None
    with suppress(json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        baseline = load_baseline(repo_root)
    baseline_exists = baseline is not None
    baseline_aggregate = baseline.aggregate if baseline else None
    baseline_n = baseline.n_examples if baseline else 0

    # Phase 3: Rounds
    runs_dir = repo_root / "eval" / "runs"
    rounds_count = 0
    if runs_dir.is_dir():
        with suppress(OSError):
            rounds_count = sum(1 for _ in runs_dir.glob("round_*.json"))

    # Phase 3: Frontier
    frontier_path = runs_dir / "frontier.json"
    frontier_exists = frontier_path.is_file()

    # Phase 4: Finalized
    finalized_path = runs_dir / "finalized.json"
    finalized_exists = finalized_path.is_file()
    finalized_aggregate: float | None = None
    if finalized_exists:
        try:
            finalized_data = json.loads(finalized_path.read_text(encoding="utf-8"))
            finalized_aggregate = finalized_data.get("aggregate")
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "scaffold": {
            "exists": scaffold_exists,
            "skills_count": skills_count,
            "rules_count": rules_count,
        },
        "golden_set": {
            "exists": golden_exists,
            "count": golden_count,
            "buckets": golden_buckets,
        },
        "config": {
            "exists": config_exists,
            "mode": config_mode,
            "optimizer_backend": config_backend,
        },
        "baseline": {
            "exists": baseline_exists,
            "aggregate": baseline_aggregate,
            "n_examples": baseline_n,
        },
        "rounds": {"count": rounds_count},
        "frontier": {"exists": frontier_exists},
        "finalized": {
            "exists": finalized_exists,
            "aggregate": finalized_aggregate,
        },
    }


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forge Orchestrator</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f8fafc; --surface: #ffffff; --text: #0f172a;
    --muted: #64748b; --border: #cbd5e1; --radius: 0.5rem;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f172a; --surface: #1e293b; --text: #f8fafc;
      --muted: #94a3b8; --border: #475569;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    background: var(--bg); color: var(--text); padding: 2rem;
    max-width: 64rem; margin-left: auto; margin-right: auto;
  }
  h1 { font-size: 1.5rem; margin: 0 0 0.25rem; }
  p.subtitle { color: var(--muted); margin: 0 0 2rem; font-size: 0.9rem; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.25rem 1.5rem; margin-bottom: 1rem;
  }
  .card h2 { font-size: 1.1rem; margin: 0 0 0.75rem; }
  .card h3 { font-size: 0.95rem; margin: 1rem 0 0.5rem; }
  .card label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.25rem; }
  select {
    width: 100%; padding: 0.5rem; border: 1px solid var(--border);
    border-radius: var(--radius); background: var(--surface); color: var(--text);
    font-size: 0.9rem;
  }
  button {
    padding: 0.5rem 1.25rem; border: 1px solid var(--border);
    border-radius: var(--radius); background: var(--surface); color: var(--text);
    font-size: 0.9rem; cursor: pointer; font-weight: 500;
  }
  button:hover:not(:disabled) { border-color: var(--text); }
  button:disabled { opacity: 0.5; cursor: wait; }
  .toggle-btn { font-size: 0.85rem; padding: 0.35rem 0.8rem; }
  .radio-label {
    display: inline-flex; align-items: center; gap: 0.35rem;
    margin-right: 1.5rem; font-size: 0.9rem; cursor: pointer;
  }
  .radio-label input { margin: 0; }
  .result-area { margin-top: 0.75rem; min-height: 1.5rem; font-size: 0.9rem; }
  .error-msg { color: #991b1b; }
  @media (prefers-color-scheme: dark) { .error-msg { color: #fca5a5; } }
  .spinner {
    display: inline-block; width: 1rem; height: 1rem;
    border: 2px solid var(--border); border-top-color: var(--text);
    border-radius: 50%; animation: spin 0.8s linear infinite;
    vertical-align: middle; margin-right: 0.4rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .meta { color: var(--muted); font-size: 0.85rem; }
  .score-row { margin-top: 0.5rem; font-size: 0.9rem; }
  .score-item { font-weight: 500; }
  .delta { font-weight: 600; }
  .delta.positive { color: #166534; }
  .delta.negative { color: #991b1b; }
  @media (prefers-color-scheme: dark) {
    .delta.positive { color: #86efac; }
    .delta.negative { color: #fca5a5; }
  }
  .files-section { margin-top: 0.5rem; font-size: 0.85rem; }
  .file-list { margin-top: 0.25rem; }
  .diff-summary {
    margin-top: 0.5rem; padding: 0.5rem; background: var(--bg);
    border: 1px solid var(--border); border-radius: var(--radius);
    font-size: 0.85rem; color: var(--muted); white-space: pre-wrap;
  }
  .round-link { display: inline-block; margin-top: 0.5rem; font-size: 0.85rem; }
  .section-title { font-size: 1.1rem; margin: 1.5rem 0 0.75rem; }
  table { width: 100%; border-collapse: collapse; background: var(--surface);
    border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  th, td { padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }
  th { background: var(--surface); font-weight: 600; color: var(--muted);
    font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
  td.empty { text-align: center; color: var(--muted); padding: 2rem; }
  .badge { padding: 0.15rem 0.5rem; border-radius: var(--radius); font-size: 0.8rem; font-weight: 600; }
  .badge.keep { background: #dcfce7; color: #166534; }
  .badge.revert { background: #fee2e2; color: #991b1b; }
  .badge.noop { background: #f1f5f9; color: #475569; }
  .badge.infra_fail { background: #fef3c7; color: #92400e; }
  .positive { color: #166534; }
  .negative { color: #991b1b; }
  @media (prefers-color-scheme: dark) {
    .badge.keep { background: #052e16; color: #86efac; }
    .badge.revert { background: #450a0a; color: #fca5a5; }
    .badge.noop { background: #1e293b; color: #94a3b8; }
    .badge.infra_fail { background: #422006; color: #fcd34d; }
    .positive { color: #86efac; }
    .negative { color: #fca5a5; }
  }
  .status-badge { font-size: 0.8rem; font-weight: 600; margin-left: 0.5rem; }
  .phase-header { display: flex; align-items: center; }
  .subsection { margin-bottom: 0.75rem; }
  .subsection h3 { margin: 0.5rem 0 0.25rem; font-size: 0.9rem; }
  .detail-section { margin-top: 0.5rem; }
  .config-preview {
    margin: 0.5rem 0; padding: 0.5rem; background: var(--bg);
    border: 1px solid var(--border); border-radius: var(--radius);
    font-size: 0.8rem; overflow-x: auto; white-space: pre-wrap;
    max-height: 20rem; overflow-y: auto;
  }
  .file-item {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.35rem 0; font-size: 0.85rem;
  }
  .file-item button { font-size: 0.8rem; padding: 0.25rem 0.6rem; }
  .edit-area {
    width: 100%; min-height: 8rem; padding: 0.5rem; margin-top: 0.5rem;
    border: 1px solid var(--border); border-radius: var(--radius);
    background: var(--surface); color: var(--text); font-size: 0.85rem;
    font-family: ui-monospace, monospace; resize: vertical;
  }
  .golden-sample {
    margin-top: 0.5rem; padding: 0.5rem; background: var(--bg);
    border: 1px solid var(--border); border-radius: var(--radius);
    font-size: 0.8rem; overflow-x: auto;
  }
</style>
</head>
<body>
<h1>Forge Orchestrator</h1>
<p class="subtitle">Optimization workflow wizard \\u2014 setup, baseline, optimize, finalize.</p>

<!-- Phase 1: Setup -->
<div class="card" id="phase-setup">
  <div class="phase-header"><h2>Phase 1: Setup</h2><span class="status-badge" id="setup-status"></span></div>
  <div class="subsection">
    <h3>Scaffold <span class="meta" id="scaffold-meta"></span></h3>
    <button id="scaffold-toggle" class="toggle-btn" type="button">View/Edit</button>
    <div id="scaffold-detail" class="detail-section" style="display:none;"></div>
  </div>
  <div class="subsection">
    <h3>Golden Dataset <span class="meta" id="golden-meta"></span></h3>
    <button id="golden-toggle" class="toggle-btn" type="button">View</button>
    <div id="golden-detail" class="detail-section" style="display:none;"></div>
  </div>
  <div class="subsection">
    <h3>Harness Config <span class="meta" id="config-meta"></span></h3>
    <button id="config-toggle" class="toggle-btn" type="button">View/Edit</button>
    <div id="config-detail" class="detail-section" style="display:none;"></div>
  </div>
</div>

<!-- Phase 2: Baseline -->
<div class="card" id="phase-baseline">
  <div class="phase-header"><h2>Phase 2: Baseline</h2><span class="status-badge" id="baseline-status"></span></div>
  <div id="baseline-info" class="meta"></div>
  <h3>2. Validate Baseline</h3>
  <button id="validate-btn" type="button">Run Validation</button>
  <div id="validate-result" class="result-area"></div>
  <h3>Build Baseline</h3>
  <button id="baseline-btn" type="button">Build Baseline</button>
  <div id="baseline-result" class="result-area"></div>
</div>

<!-- Phase 3: Optimize -->
<div class="card" id="phase-optimize">
  <div class="phase-header"><h2>Phase 3: Optimize</h2><span class="status-badge" id="optimize-status"></span></div>
  <h3>1. Select Agent</h3>
  <label for="agent-select">Agent bundle</label>
  <select id="agent-select"><option value="">Loading agents\\u2026</option></select>
  <h3>3. Optimization Mode</h3>
  <label class="radio-label"><input type="radio" name="mode" value="prompt" checked> Prompt Mode</label>
  <label class="radio-label"><input type="radio" name="mode" value="code"> Code Mode</label>
  <h3>4. Run Optimizer</h3>
  <button id="run-btn" type="button">Run Optimizer</button>
  <div id="run-result" class="result-area"></div>
  <h3>Frontier Summary</h3>
  <div id="frontier-summary" class="result-area"></div>
</div>

<!-- Phase 4: Finalize -->
<div class="card" id="phase-finalize">
  <div class="phase-header"><h2>Phase 4: Finalize</h2><span class="status-badge" id="finalize-status"></span></div>
  <div id="finalize-info" class="meta"></div>
  <button id="finalize-btn" type="button">Finalize</button>
  <div id="finalize-result" class="result-area"></div>
</div>

<h2 class="section-title">Rounds History</h2>
<table>
<thead><tr>
  <th>Round</th><th>Decision</th><th>Action</th>
  <th>Baseline</th><th>&Delta; Score</th><th>Aggregate</th>
</tr></thead>
<tbody>
"""

_DASHBOARD_TAIL = """
</tbody>
</table>
<script>
"use strict";
function $(id) { return document.getElementById(id); }

function fmt(val) {
  return val != null ? val.toFixed(4) : "\\u2014";
}

function showLoading(el, msg) {
  el.innerHTML = "";
  var spinner = document.createElement("span");
  spinner.className = "spinner";
  el.appendChild(spinner);
  el.appendChild(document.createTextNode(msg));
}

function showError(el, msg) {
  el.innerHTML = "";
  var span = document.createElement("span");
  span.className = "error-msg";
  span.textContent = msg;
  el.appendChild(span);
}

function statusBadge(exists, readyText, incompleteText) {
  var span = document.createElement("span");
  if (exists) {
    span.className = "badge keep";
    span.textContent = readyText || "\\u2705 Ready";
  } else {
    span.className = "badge noop";
    span.textContent = incompleteText || "\\u26a0\\ufe0f Incomplete";
  }
  return span;
}

// --- Phase 0: State loading ---

async function loadState() {
  try {
    var res = await fetch("/api/state");
    var state = await res.json();
    renderState(state);
  } catch (e) {
    console.error("Failed to load state:", e);
  }
}

function renderState(state) {
  // Phase 1: Setup
  var setupStatus = $("setup-status");
  setupStatus.innerHTML = "";
  var setupReady = state.scaffold.exists && state.golden_set.exists && state.config.exists;
  setupStatus.appendChild(statusBadge(setupReady, "\\u2705 Ready", "\\u26a0\\ufe0f Incomplete"));
  $("scaffold-meta").textContent = state.scaffold.exists
    ? state.scaffold.skills_count + " skills, " + state.scaffold.rules_count + " rules"
    : "not configured";
  $("golden-meta").textContent = state.golden_set.exists
    ? state.golden_set.count + " examples"
    : "not configured";
  $("config-meta").textContent = state.config.exists
    ? "mode: " + (state.config.mode || "\\u2014") + ", backend: " + (state.config.optimizer_backend || "\\u2014")
    : "not configured";

  // Phase 2: Baseline
  var baselineStatus = $("baseline-status");
  baselineStatus.innerHTML = "";
  baselineStatus.appendChild(statusBadge(state.baseline.exists, "\\u2705 Ready", "\\u26a0\\ufe0f Incomplete"));
  var baselineInfo = $("baseline-info");
  baselineInfo.innerHTML = "";
  if (state.baseline.exists) {
    baselineInfo.textContent = "Aggregate: " + fmt(state.baseline.aggregate)
      + ", " + state.baseline.n_examples + " examples";
  } else {
    baselineInfo.textContent = "No baseline built yet.";
  }

  // Phase 3: Optimize
  var optimizeStatus = $("optimize-status");
  optimizeStatus.innerHTML = "";
  optimizeStatus.appendChild(statusBadge(
    state.rounds.count > 0,
    "\\u2705 " + state.rounds.count + " rounds",
    "\\u26a0\\ufe0f No rounds yet"
  ));
  loadFrontier();

  // Phase 4: Finalize
  var finalizeStatus = $("finalize-status");
  finalizeStatus.innerHTML = "";
  finalizeStatus.appendChild(statusBadge(state.finalized.exists, "\\u2705 Done", "\\u26a0\\ufe0f Pending"));
  var finalizeInfo = $("finalize-info");
  finalizeInfo.innerHTML = "";
  if (state.finalized.exists) {
    finalizeInfo.textContent = "Aggregate: " + fmt(state.finalized.aggregate);
  } else {
    finalizeInfo.textContent = "Not finalized yet.";
  }
  // Disable finalize button if baseline OR frontier doesn't exist.
  $("finalize-btn").disabled = !state.baseline.exists || !state.frontier.exists;
}

// --- Agent loading (existing) ---

async function loadAgents() {
  var sel = $("agent-select");
  try {
    var res = await fetch("/agents");
    var agents = await res.json();
    sel.innerHTML = "";
    if (!agents.length) {
      var opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No agents found";
      sel.appendChild(opt);
      return;
    }
    agents.forEach(function(a) {
      var opt = document.createElement("option");
      opt.value = a.path;
      opt.textContent = a.name + " (" + a.filename + ")";
      sel.appendChild(opt);
    });
  } catch (e) {
    sel.innerHTML = "";
    var opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Failed to load agents";
    sel.appendChild(opt);
  }
}

// --- Validation (existing) ---

async function runValidation() {
  var btn = $("validate-btn");
  var result = $("validate-result");
  btn.disabled = true;
  showLoading(result, "Running validation\\u2026");
  try {
    var res = await fetch("/validate", { method: "POST" });
    var data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Validation failed");
    result.innerHTML = "";
    var badge = document.createElement("span");
    badge.className = "badge keep";
    badge.textContent = "Score: " + fmt(data.aggregate);
    result.appendChild(badge);
    if (data.n_examples) {
      var meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = "  \\u00b7  " + data.n_examples + " examples  \\u00b7  mode: " + (data.mode || "\\u2014");
      result.appendChild(meta);
    }
  } catch (e) {
    showError(result, e.message);
  } finally {
    btn.disabled = false;
  }
}

// --- Baseline build (Phase 2) ---

async function buildBaseline() {
  var btn = $("baseline-btn");
  var result = $("baseline-result");
  btn.disabled = true;
  showLoading(result, "Building baseline\\u2026");
  try {
    var res = await fetch("/api/baseline", { method: "POST" });
    var data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Baseline failed");
    result.innerHTML = "";
    var badge = document.createElement("span");
    badge.className = "badge keep";
    badge.textContent = "Aggregate: " + fmt(data.aggregate);
    result.appendChild(badge);
    var meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = "  \\u00b7  " + data.n_examples + " examples  \\u00b7  mode: " + (data.mode || "\\u2014");
    result.appendChild(meta);
    loadState();
  } catch (e) {
    showError(result, e.message);
  } finally {
    btn.disabled = false;
  }
}

// --- Scaffold detail (Phase 1) ---

async function loadScaffoldDetail() {
  var detail = $("scaffold-detail");
  showLoading(detail, "Loading scaffold\\u2026");
  try {
    var res = await fetch("/api/scaffold");
    var data = await res.json();
    detail.innerHTML = "";
    var pre = document.createElement("pre");
    pre.className = "config-preview";
    pre.textContent = JSON.stringify(data.config, null, 2);
    detail.appendChild(pre);
    if (data.files && data.files.length) {
      var title = document.createElement("h4");
      title.textContent = "Skill/Rule Files";
      detail.appendChild(title);
      data.files.forEach(function(f) {
        var fileDiv = document.createElement("div");
        fileDiv.className = "file-item";
        var nameSpan = document.createElement("span");
        nameSpan.textContent = f.name + " (" + f.type + ")";
        fileDiv.appendChild(nameSpan);
        var editBtn = document.createElement("button");
        editBtn.textContent = "Edit";
        editBtn.addEventListener("click", function() { editScaffoldFile(f.name); });
        fileDiv.appendChild(editBtn);
        detail.appendChild(fileDiv);
      });
    }
  } catch (e) {
    showError(detail, e.message);
  }
}

async function editScaffoldFile(name) {
  var detail = $("scaffold-detail");
  detail.innerHTML = "";
  var title = document.createElement("h4");
  title.textContent = "Editing: " + name;
  detail.appendChild(title);
  try {
    var res = await fetch("/api/scaffold/files/" + name);
    var content = await res.text();
    var area = document.createElement("textarea");
    area.className = "edit-area";
    area.id = "scaffold-file-edit";
    area.value = content;
    detail.appendChild(area);
    var saveBtn = document.createElement("button");
    saveBtn.textContent = "Save";
    saveBtn.addEventListener("click", function() { saveScaffoldFile(name); });
    detail.appendChild(saveBtn);
    var backBtn = document.createElement("button");
    backBtn.textContent = "Back";
    backBtn.addEventListener("click", loadScaffoldDetail);
    detail.appendChild(backBtn);
  } catch (e) {
    showError(detail, e.message);
  }
}

async function saveScaffoldFile(name) {
  var area = $("scaffold-file-edit");
  if (!area) return;
  try {
    var res = await fetch("/api/scaffold/files/" + name, {
      method: "PUT",
      headers: { "Content-Type": "text/plain" },
      body: area.value
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Save failed");
    loadScaffoldDetail();
  } catch (e) {
    showError($("scaffold-detail"), e.message);
  }
}

// --- Golden set detail (Phase 1) ---

async function loadGoldenDetail() {
  var detail = $("golden-detail");
  showLoading(detail, "Loading golden set\\u2026");
  try {
    var res = await fetch("/api/golden-set");
    var data = await res.json();
    detail.innerHTML = "";
    var summary = document.createElement("div");
    summary.className = "meta";
    summary.textContent = "Total: " + data.total + " examples";
    detail.appendChild(summary);
    var bucketsDiv = document.createElement("div");
    bucketsDiv.className = "meta";
    var bucketParts = [];
    for (var key in data.buckets) {
      bucketParts.push(key + ": " + data.buckets[key]);
    }
    bucketsDiv.textContent = "Buckets: " + (bucketParts.join(", ") || "\\u2014");
    detail.appendChild(bucketsDiv);
    if (data.samples && data.samples.length) {
      var title = document.createElement("h4");
      title.textContent = "First " + data.samples.length + " Examples";
      detail.appendChild(title);
      data.samples.forEach(function(ex) {
        var sampleDiv = document.createElement("div");
        sampleDiv.className = "golden-sample";
        sampleDiv.textContent = JSON.stringify(ex, null, 2);
        detail.appendChild(sampleDiv);
      });
    }
  } catch (e) {
    showError(detail, e.message);
  }
}

// --- Config detail (Phase 1) ---

async function loadConfigDetail() {
  var detail = $("config-detail");
  showLoading(detail, "Loading config\\u2026");
  try {
    var res = await fetch("/api/config");
    var data = await res.json();
    detail.innerHTML = "";
    var pre = document.createElement("pre");
    pre.className = "config-preview";
    pre.textContent = JSON.stringify(data, null, 2);
    detail.appendChild(pre);
  } catch (e) {
    showError(detail, e.message);
  }
}

// --- Frontier (Phase 3) ---

async function loadFrontier() {
  var container = $("frontier-summary");
  try {
    var res = await fetch("/api/frontier");
    if (!res.ok) {
      container.textContent = "No frontier yet.";
      return;
    }
    var data = await res.json();
    container.innerHTML = "";
    var best = data.best || {};
    var keys = Object.keys(best);
    if (!keys.length) {
      container.textContent = "No frontier yet.";
      return;
    }
    keys.forEach(function(key) {
      var row = document.createElement("div");
      row.className = "score-row";
      row.textContent = key + ": " + fmt(best[key]);
      container.appendChild(row);
    });
  } catch (e) {
    container.textContent = "No frontier yet.";
  }
}

// --- Finalize (Phase 4) ---

async function runFinalize() {
  var btn = $("finalize-btn");
  var result = $("finalize-result");
  btn.disabled = true;
  showLoading(result, "Finalizing\\u2026");
  try {
    var res = await fetch("/api/finalize", { method: "POST" });
    var data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Finalize failed");
    result.innerHTML = "";
    var badge = document.createElement("span");
    badge.className = "badge keep";
    badge.textContent = "Aggregate: " + fmt(data.aggregate);
    result.appendChild(badge);
    if (data.per_judge) {
      var judges = document.createElement("div");
      judges.className = "score-row";
      for (var key in data.per_judge) {
        judges.appendChild(document.createTextNode(
          key + ": " + fmt(data.per_judge[key]) + "  "
        ));
      }
      result.appendChild(judges);
    }
    loadState();
  } catch (e) {
    showError(result, e.message);
  } finally {
    btn.disabled = false;
  }
}

// --- Round result rendering (existing) ---

function scoreSpan(label, val) {
  var span = document.createElement("span");
  span.className = "score-item";
  span.textContent = label + ": " + fmt(val);
  return span;
}

function fileList(label, items) {
  var div = document.createElement("div");
  div.className = "file-list";
  var strong = document.createElement("strong");
  strong.textContent = label + ": ";
  div.appendChild(strong);
  div.appendChild(document.createTextNode(items && items.length ? items.join(", ") : "\\u2014"));
  return div;
}

function renderRoundResult(data) {
  var container = $("run-result");
  container.innerHTML = "";

  var decision = (data.decision || "").toLowerCase();
  var badge = document.createElement("span");
  badge.className = "badge " + decision;
  badge.textContent = data.decision || "\\u2014";
  container.appendChild(badge);

  var scores = document.createElement("div");
  scores.className = "score-row";
  scores.appendChild(scoreSpan("Baseline", data.baseline_score));
  scores.appendChild(document.createTextNode(" \\u2192 "));
  scores.appendChild(scoreSpan("Mutated", data.mutated_score));
  var delta = data.score_delta;
  var deltaSpan = document.createElement("span");
  deltaSpan.className = "delta " + (delta > 0 ? "positive" : delta < 0 ? "negative" : "");
  deltaSpan.textContent = "  \\u0394 " + fmt(delta);
  scores.appendChild(deltaSpan);
  container.appendChild(scores);

  if ((data.files_added && data.files_added.length) ||
      (data.files_changed && data.files_changed.length) ||
      (data.files_removed && data.files_removed.length)) {
    var files = document.createElement("div");
    files.className = "files-section";
    files.appendChild(fileList("Added", data.files_added));
    files.appendChild(fileList("Changed", data.files_changed));
    files.appendChild(fileList("Removed", data.files_removed));
    container.appendChild(files);
  }

  if (data.diff_summary) {
    var summary = document.createElement("div");
    summary.className = "diff-summary";
    summary.textContent = data.diff_summary;
    container.appendChild(summary);
  }

  if (data.round_id != null) {
    var link = document.createElement("a");
    link.href = "/rounds/" + data.round_id;
    link.className = "round-link";
    link.textContent = "View round " + data.round_id + " JSON \\u2192";
    container.appendChild(link);
  }
}

async function runOptimizer() {
  var btn = $("run-btn");
  var result = $("run-result");
  var agent = $("agent-select").value;
  var modeEl = document.querySelector("input[name=mode]:checked");
  var mode = modeEl ? modeEl.value : "prompt";
  btn.disabled = true;
  showLoading(result, "Running optimizer\\u2026 this may take several minutes");
  var body = {};
  if (agent) body.agent = agent;
  if (mode) body.mode = mode;
  try {
    var res = await fetch("/rounds", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    var data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Optimizer failed");
    renderRoundResult(data);
    loadState();
  } catch (e) {
    showError(result, e.message);
  } finally {
    btn.disabled = false;
  }
}

// --- Toggle helpers ---

function toggleSection(detailId, toggleId, loadFn) {
  var detail = $(detailId);
  var btn = $(toggleId);
  if (detail.style.display === "none") {
    detail.style.display = "block";
    btn.textContent = "Hide";
    loadFn();
  } else {
    detail.style.display = "none";
    btn.textContent = "View/Edit";
  }
}

// --- Init ---

document.addEventListener("DOMContentLoaded", function() {
  loadState();
  loadAgents();
  $("validate-btn").addEventListener("click", runValidation);
  $("baseline-btn").addEventListener("click", buildBaseline);
  $("run-btn").addEventListener("click", runOptimizer);
  $("finalize-btn").addEventListener("click", runFinalize);
  $("scaffold-toggle").addEventListener("click", function() {
    toggleSection("scaffold-detail", "scaffold-toggle", loadScaffoldDetail);
  });
  $("golden-toggle").addEventListener("click", function() {
    toggleSection("golden-detail", "golden-toggle", loadGoldenDetail);
  });
  $("config-toggle").addEventListener("click", function() {
    toggleSection("config-detail", "config-toggle", loadConfigDetail);
  });
});
</script>
</body>
</html>"""


def _render_dashboard(rounds: list[dict[str, Any]]) -> str:
    """Render the interactive 4-phase wizard dashboard from round summaries.

    The page embeds four phase cards (setup, baseline, optimize, finalize)
    above the rounds history table. Dynamic data is fetched client-side via
    ``fetch()`` and inserted with ``textContent`` / ``createElement`` (never
    ``innerHTML``) to prevent XSS. Server-rendered values in the rounds
    table are escaped with :func:`html.escape`.
    """
    if not rounds:
        rows_html = '<tr><td colspan="6" class="empty">No rounds yet.</td></tr>'
    else:
        row_parts: list[str] = []
        for r in rounds:
            decision = escape(str(r.get("decision") or "—"))
            decision_cls = escape((r.get("decision") or "").lower())
            sd = r.get("score_delta")
            if isinstance(sd, int | float):
                sd_str = f"{sd:+.4f}"
                sd_cls = "positive" if sd > 0 else ("negative" if sd < 0 else "")
            else:
                sd_str = "—"
                sd_cls = ""
            bs = r.get("baseline_score")
            bs_str = f"{bs:.4f}" if isinstance(bs, int | float) else "—"
            agg = r.get("aggregate")
            agg_str = f"{agg:.4f}" if isinstance(agg, int | float) else "—"
            row_parts.append(
                "<tr>"
                f"<td>{escape(str(r.get('round_id', '—')))}</td>"
                f'<td><span class="badge {decision_cls}">{decision}</span></td>'
                f"<td>{escape(str(r.get('action_kind') or '—'))}</td>"
                f"<td>{escape(bs_str)}</td>"
                f'<td class="{escape(sd_cls)}">{escape(sd_str)}</td>'
                f"<td>{escape(agg_str)}</td>"
                "</tr>"
            )
        rows_html = "\n".join(row_parts)
    return _DASHBOARD_HEAD + rows_html + _DASHBOARD_TAIL


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    for var in _ENV_VARS:
        val = os.environ.get(var)
        if var in _SECRET_ENV_VARS:
            logger.info("startup %s=%s", var, "set" if val else "(not set)")
        else:
            logger.info("startup %s=%s", var, val or "<unset>")
    yield


app = FastAPI(title="Forge Orchestrator", lifespan=_lifespan)

# Single shared lock for ALL mutating operations — prevents concurrent
# mutations to shared scaffold/config/eval artifacts. The alias
# ``_round_lock`` is kept for backward compat with existing tests that
# import it directly.
_mutation_lock = threading.Lock()
_round_lock = _mutation_lock  # backward-compat alias for existing tests


# ---------------------------------------------------------------------------
# Supporting endpoints (existing, kept)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agents", response_model=list[AgentInfo])
async def list_agents() -> list[AgentInfo]:
    return [AgentInfo(**a) for a in _list_agents(get_repo_root())]


@app.post("/validate", response_model=ValidationResponse)
async def validate() -> ValidationResponse:
    """Run a read-only baseline eval and return the aggregate score.

    Does NOT write a baseline cache file — purely diagnostic. Runs
    ``evaluate_branch`` in a thread pool (same pattern as ``POST /rounds``)
    so the event loop stays responsive.
    """
    repo_root = get_repo_root()
    scaffold_root = repo_root / "scaffold"
    try:
        report = await anyio.to_thread.run_sync(
            partial(evaluate_branch, scaffold_root=scaffold_root)
        )
    except Exception as exc:  # noqa: BLE001 — surface any eval failure
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ValidationResponse(
        aggregate=report.aggregate,
        mode=report.mode,
        n_examples=report.n_rows,
        run_id=report.run_id,
    )


# ---------------------------------------------------------------------------
# Phase 3: Optimize (existing, kept — with terminal-finalization guard)
# ---------------------------------------------------------------------------


@app.post("/rounds", response_model=RoundReportResponse)
async def start_round(req: RoundCreateRequest) -> RoundReportResponse:
    repo_root = get_repo_root()
    finalized_path = repo_root / "eval" / "runs" / "finalized.json"
    if finalized_path.is_file() and not req.force:
        raise HTTPException(
            status_code=409,
            detail="optimization is finalized; cannot run more rounds",
        )
    if not _mutation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a round is already running")
    try:
        if req.mode is not None or req.agent is not None:
            _update_harness_config(repo_root, mode=req.mode, agent=req.agent)
        round_id = req.round_id if req.round_id is not None else _next_round_id(repo_root)
        max_turns = req.max_turns if req.max_turns is not None else 30
        try:
            report = await anyio.to_thread.run_sync(
                partial(
                    run_round,
                    round_id=round_id,
                    repo_root=repo_root,
                    eval_mode=req.eval_mode,
                    max_turns=max_turns,
                )
            )
        except Exception as exc:  # noqa: BLE001 — surface any round failure
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _round_report_to_response(report)
    finally:
        _mutation_lock.release()


@app.get("/rounds", response_model=list[RoundSummaryResponse])
async def list_rounds() -> list[RoundSummaryResponse]:
    return [RoundSummaryResponse(**s) for s in _list_round_summaries(get_repo_root())]


@app.get("/rounds/{round_id}")
async def get_round(round_id: int) -> dict[str, Any]:
    path = _round_json_path(get_repo_root(), round_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"round {round_id} not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"round {round_id} JSON is corrupt: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Phase 0: Workflow state
# ---------------------------------------------------------------------------


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    """Return the status of all four workflow phases."""
    return await anyio.to_thread.run_sync(_workflow_state, get_repo_root())


# ---------------------------------------------------------------------------
# Phase 1: Setup
# ---------------------------------------------------------------------------


@app.get("/api/scaffold", response_model=ScaffoldResponse)
async def get_scaffold() -> ScaffoldResponse:
    """Return scaffold/harness.yaml parsed as JSON + list of skill/rule files."""
    try:
        data = await anyio.to_thread.run_sync(_get_scaffold_sync, get_repo_root())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ScaffoldResponse(
        config=data["config"],
        files=[ScaffoldFileEntry(**f) for f in data["files"]],
    )


@app.put("/api/scaffold")
async def put_scaffold(body: dict[str, Any]) -> dict[str, str]:
    """Write scaffold/harness.yaml from a JSON body.

    Accepts the full scaffold config (sampling, skills, rules, tools).
    Comments are not preserved by ``yaml.safe_dump`` — acceptable because
    the full commented template lives in version control.
    """
    if not _mutation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a scaffold update is already running")
    try:
        repo_root = get_repo_root()
        path = _scaffold_config_path(repo_root)
        content = yaml.safe_dump(body, sort_keys=False, default_flow_style=False)
        try:
            await anyio.to_thread.run_sync(partial(_atomic_write, path, content))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "ok"}
    finally:
        _mutation_lock.release()


@app.get("/api/scaffold/files/{filename}", response_class=PlainTextResponse)
async def get_scaffold_file(filename: str) -> str:
    """Return the raw text content of a skill/rule markdown file from scaffold/."""
    _validate_scaffold_filename(filename)
    repo_root = get_repo_root()
    path = _find_scaffold_file(repo_root, filename)
    if path is None:
        raise HTTPException(status_code=404, detail=f"file {filename!r} not found in scaffold/")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/api/scaffold/files/{filename}")
async def put_scaffold_file(filename: str, request: Request) -> dict[str, str]:
    """Write raw text to a scaffold skill/rule markdown file."""
    _validate_scaffold_filename(filename)
    content = (await request.body()).decode("utf-8")
    if not _mutation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a scaffold file update is already running")
    try:
        repo_root = get_repo_root()
        path = _scaffold_file_write_path(repo_root, filename)
        try:
            await anyio.to_thread.run_sync(partial(_atomic_write, path, content))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "ok", "file": filename}
    finally:
        _mutation_lock.release()


@app.get("/api/golden-set", response_model=GoldenSetResponse)
async def get_golden_set() -> GoldenSetResponse:
    """Return golden dataset stats (total, per-bucket counts) + first 5 samples."""
    stats = await anyio.to_thread.run_sync(_golden_set_stats, get_repo_root())
    return GoldenSetResponse(**stats)


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    """Return harness/config.yaml parsed as JSON, with secrets redacted."""
    try:
        return await anyio.to_thread.run_sync(_get_config_sync, get_repo_root())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/api/config")
async def put_config(body: ConfigUpdateRequest) -> dict[str, str]:
    """Update harness/config.yaml — only allowed fields are merged.

    Top-level: ``mode``, ``runtime_endpoint``, ``optimizer_endpoint``,
    ``judge_endpoint``. Nested sections (only listed sub-fields updated):
    ``optimizer``, ``eval``, ``gate``, ``loop``.
    """
    if not _mutation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a config update is already running")
    try:
        repo_root = get_repo_root()
        update = body.model_dump(exclude_none=True)
        try:
            await anyio.to_thread.run_sync(
                partial(_put_config_sync, repo_root, update)
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "ok"}
    finally:
        _mutation_lock.release()


# ---------------------------------------------------------------------------
# Phase 2: Baseline
# ---------------------------------------------------------------------------


@app.post("/api/baseline")
async def build_baseline() -> dict[str, Any]:
    """Build the baseline: run eval → CachedBaseline → eval/runs/baseline.json.

    Runs ``_build_baseline_sync`` in a thread pool so the event loop
    stays responsive. Overwrites any existing baseline.
    """
    if not _mutation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a baseline build is already running")
    try:
        repo_root = get_repo_root()
        try:
            baseline = await anyio.to_thread.run_sync(
                partial(_build_baseline_sync, repo_root)
            )
        except Exception as exc:  # noqa: BLE001 — surface any baseline failure
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return baseline.to_dict()
    finally:
        _mutation_lock.release()


@app.get("/api/baseline")
async def get_baseline() -> dict[str, Any]:
    """Return the cached baseline from eval/runs/baseline.json, or 404."""
    try:
        baseline = await anyio.to_thread.run_sync(load_baseline, get_repo_root())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if baseline is None:
        raise HTTPException(status_code=404, detail="baseline not found")
    return baseline.to_dict()


# ---------------------------------------------------------------------------
# Phase 3: Optimize — frontier
# ---------------------------------------------------------------------------


@app.get("/api/frontier")
async def get_frontier() -> dict[str, Any]:
    """Return eval/runs/frontier.json parsed as JSON, or 404 if not found."""
    path = get_repo_root() / "eval" / "runs" / "frontier.json"
    try:
        return await anyio.to_thread.run_sync(_read_json_file_sync, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="frontier not found") from None
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"frontier JSON is corrupt: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Phase 4: Finalize
# ---------------------------------------------------------------------------


@app.post("/api/finalize")
async def finalize() -> dict[str, Any]:
    """Run the held-out evaluation and write eval/runs/finalized.json.

    Terminal: returns 409 if ``finalized.json`` already exists (delete
    the file to re-run). Checks that ``eval/runs/frontier.json`` exists
    and ``harness/config.yaml`` has ``eval.held_out_test: true``, then
    runs ``_finalize_sync`` in a thread pool.
    """
    repo_root = get_repo_root()
    finalized_path = repo_root / "eval" / "runs" / "finalized.json"
    if finalized_path.is_file():
        raise HTTPException(
            status_code=409,
            detail="finalization already exists; delete eval/runs/finalized.json to re-run",
        )
    if not _mutation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="finalization is already running")
    try:
        try:
            result = await anyio.to_thread.run_sync(
                partial(_finalize_sync, repo_root)
            )
        except Exception as exc:  # noqa: BLE001 — surface any finalize failure
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result
    finally:
        _mutation_lock.release()


@app.get("/api/finalize")
async def get_finalized() -> dict[str, Any]:
    """Return eval/runs/finalized.json, or 404 if not found."""
    path = get_repo_root() / "eval" / "runs" / "finalized.json"
    try:
        return await anyio.to_thread.run_sync(_read_json_file_sync, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="finalized report not found") from None
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"finalized JSON is corrupt: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Dashboard (must be last — route ordering)
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    rounds = _list_round_summaries(get_repo_root())
    return _render_dashboard(rounds)
