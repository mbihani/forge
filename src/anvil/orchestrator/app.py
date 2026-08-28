"""Forge Orchestrator FastAPI app — HTTP control plane for ANVIL rounds.

Serves:
  * GET  /health       — liveness probe (required by deploy/app.yaml).
  * GET  /agents       — list agent YAML bundles from ``agents/``.
  * POST /validate     — run a read-only baseline eval, return aggregate.
  * POST /rounds       — start a new optimization round (optional mode + agent).
  * GET  /rounds       — list completed rounds.
  * GET  /rounds/{id}  — read a single round JSON.
  * GET  /             — interactive HTML dashboard (agent selection →
                         validation → mode choice → run optimizer + history).

``run_round`` and ``evaluate_branch`` are sync + blocking. Both MUST run
in a thread pool (``anyio.to_thread.run_sync``) so the single uvicorn event
loop stays responsive — calling them directly in an async handler freezes
the loop and the Databricks App 502s.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from functools import partial
from html import escape
from pathlib import Path
from typing import Any, Literal

import anyio
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from anvil.eval import evaluate_branch
from anvil.loop.decision import Decision
from anvil.loop.round import RoundReport, run_round

logger = logging.getLogger("anvil.orchestrator")

# Env vars passed by deploy/databricks.yml — logged at startup for
# debugging. Most are consumed by run_round() / harness/config.yaml,
# not by this app directly.
_ENV_VARS = [
    "OMNIGENT_SERVER_URL",
    "OMNIGENT_AUTH_TOKEN",
    "GIT_REMOTE_URL",
    "GIT_TOKEN",
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
        path.write_text(
            yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
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
</style>
</head>
<body>
<h1>Forge Orchestrator</h1>
<p class="subtitle">Iterate, evaluate, and improve agent scaffolds.</p>

<div class="card">
  <h2>1. Select Agent</h2>
  <label for="agent-select">Agent bundle</label>
  <select id="agent-select"><option value="">Loading agents…</option></select>
</div>

<div class="card">
  <h2>2. Validate Baseline</h2>
  <button id="validate-btn" type="button">Run Validation</button>
  <div id="validate-result" class="result-area"></div>
</div>

<div class="card">
  <h2>3. Optimization Mode</h2>
  <label class="radio-label"><input type="radio" name="mode" value="prompt" checked> Prompt Mode</label>
  <label class="radio-label"><input type="radio" name="mode" value="code"> Code Mode</label>
</div>

<div class="card">
  <h2>4. Run Optimizer</h2>
  <button id="run-btn" type="button">Run Optimizer</button>
  <div id="run-result" class="result-area"></div>
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

function fmt(val) {
  return val != null ? val.toFixed(4) : "\\u2014";
}

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
  } catch (e) {
    showError(result, e.message);
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", function() {
  loadAgents();
  $("validate-btn").addEventListener("click", runValidation);
  $("run-btn").addEventListener("click", runOptimizer);
});
</script>
</body>
</html>"""


def _render_dashboard(rounds: list[dict[str, Any]]) -> str:
    """Render the interactive orchestrator dashboard from round summaries.

    The page embeds four workflow cards (agent selection, validation,
    mode choice, run optimizer) above the rounds history table. Dynamic
    data is fetched client-side via ``fetch()`` and inserted with
    ``textContent`` / ``createElement`` (never ``innerHTML``) to prevent
    XSS. Server-rendered values in the rounds table are escaped with
    :func:`html.escape`.
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

# Prevents two concurrent POST /rounds from racing on the same auto-
# detected round ID and performing conflicting git operations.
_round_lock = threading.Lock()


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


@app.post("/rounds", response_model=RoundReportResponse)
async def start_round(req: RoundCreateRequest) -> RoundReportResponse:
    if not _round_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a round is already running")
    try:
        repo_root = get_repo_root()
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
        _round_lock.release()


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


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    rounds = _list_round_summaries(get_repo_root())
    return _render_dashboard(rounds)
