"""Forge Orchestrator FastAPI app — HTTP control plane for ANVIL rounds.

Serves:
  * GET  /health       — liveness probe (required by deploy/app.yaml).
  * POST /rounds       — start a new optimization round.
  * GET  /rounds       — list completed rounds.
  * GET  /rounds/{id}  — read a single round JSON.
  * GET  /             — HTML status dashboard.

``run_round`` is sync + blocking (spawns a Claude Code subprocess, runs
MLflow eval, does git operations). It MUST run in a thread pool so the
single uvicorn event loop stays responsive — calling it directly in an
async handler freezes the loop and the Databricks App 502s.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from functools import partial
from html import escape
from pathlib import Path
from typing import Any, Literal

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

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


def _round_report_to_response(report: RoundReport) -> RoundReportResponse:
    d = dataclasses.asdict(report)
    d["decision"] = str(report.decision)
    return RoundReportResponse(**d)


# ---------------------------------------------------------------------------
# Round JSON helpers
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
        summaries.append(
            {
                "round_id": data.get("round_id"),
                "decision": data.get("decision"),
                "action_kind": data.get("action_kind"),
                "baseline_score": data.get("baseline_score"),
                "score_delta": data.get("score_delta_vs_parent"),
                "aggregate": data.get("aggregate"),
            }
        )
    summaries.sort(key=lambda r: r.get("round_id") or 0)
    return summaries


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
  }
  h1 { font-size: 1.5rem; margin: 0 0 0.25rem; }
  p.subtitle { color: var(--muted); margin: 0 0 2rem; font-size: 0.9rem; }
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
</body>
</html>"""


def _render_dashboard(rounds: list[dict[str, Any]]) -> str:
    """Render a simple HTML status page from round summaries."""
    if not rounds:
        rows_html = '<tr><td colspan="6" class="empty">No rounds yet.</td></tr>'
    else:
        row_parts: list[str] = []
        for r in rounds:
            decision = escape(str(r.get("decision") or "—"))
            decision_cls = (r.get("decision") or "").lower()
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
                f"<td>{r.get('round_id', '—')}</td>"
                f'<td><span class="badge {decision_cls}">{decision}</span></td>'
                f"<td>{escape(str(r.get('action_kind') or '—'))}</td>"
                f"<td>{bs_str}</td>"
                f'<td class="{sd_cls}">{sd_str}</td>'
                f"<td>{agg_str}</td>"
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
        logger.info("startup %s=%s", var, os.environ.get(var, "<unset>"))
    yield


app = FastAPI(title="Forge Orchestrator", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rounds", response_model=RoundReportResponse)
async def start_round(req: RoundCreateRequest) -> RoundReportResponse:
    repo_root = get_repo_root()
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
