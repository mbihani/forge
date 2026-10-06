"""Streamlit app: ANVIL round explorer.

Single-file local dashboard. Pick a round in the sidebar; the main
area shows tabs for overview, mutation diff, failures, critique,
and the cross-round curve.

Run::

    uv run streamlit run scripts/round_dashboard.py

Reads (read-only):

* ``eval/runs/round_NNN.json`` — per-round summary.
* ``eval/runs/baseline.json`` — cached baseline.
* ``eval/mutations.jsonl`` — append-only mutation log.
* ``scaffold/memory/round_NNN_critique.md`` — optimizer rationale.
* ``scaffold/memory/round_NNN_transcript.md`` — full session.
* ``git diff main..anvil/exp-round-N -- scaffold/`` for the diff.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "eval" / "runs"
MEMORY_DIR = REPO_ROOT / "scaffold" / "memory"
MUTATIONS_PATH = REPO_ROOT / "eval" / "mutations.jsonl"
BASELINE_PATH = RUNS_DIR / "baseline.json"

WORKSPACE_HOST = os.environ.get("DATABRICKS_HOST", "")

DECISION_COLOR = {
    "keep": "#28a745",
    "noop": "#6c757d",
    "revert": "#dc3545",
    "infra_fail": "#fd7e14",
}

_ROUND_JSON_RE = re.compile(r"^round_(\d+)\.json$")


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_baseline() -> dict | None:
    if not BASELINE_PATH.is_file():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_mutations() -> list[dict]:
    if not MUTATIONS_PATH.is_file():
        return []
    rows: list[dict] = []
    for line in MUTATIONS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@st.cache_data(show_spinner=False)
def load_round_json(round_id: int) -> dict | None:
    path = RUNS_DIR / f"round_{round_id:03d}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def list_round_ids() -> list[int]:
    out: list[int] = []
    for p in sorted(RUNS_DIR.glob("round_*.json")):
        m = _ROUND_JSON_RE.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return out


@st.cache_data(show_spinner=False)
def load_critique(round_id: int) -> str | None:
    path = MEMORY_DIR / f"round_{round_id:03d}_critique.md"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_transcript_head(round_id: int, max_chars: int = 50_000) -> str | None:
    path = MEMORY_DIR / f"round_{round_id:03d}_transcript.md"
    if not path.is_file():
        return None
    size = path.stat().st_size
    if size > 5_000_000:
        return f"(transcript is {size:,} bytes — too large to render; open the file at {path})"
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... ({len(text) - max_chars:,} more chars truncated)"
    return text


@st.cache_data(show_spinner=False)
def get_diff_stat(round_id: int) -> tuple[str, str]:
    """Return (stat_text, full_diff). Empty strings on git failure or no branch."""
    branch = f"anvil/exp-round-{round_id}"
    try:
        stat = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--stat", f"main..{branch}", "--", "scaffold/"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if stat.returncode != 0:
            return "", ""
        full = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "diff",
                f"main..{branch}",
                "--",
                "scaffold/skills/",
                "scaffold/rules/",
                "scaffold/harness.yaml",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        return stat.stdout, full.stdout
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "", ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_score(value: float | None, places: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.3f}"


def decision_pill(decision: str) -> str:
    color = DECISION_COLOR.get(decision, "#888")
    return (
        f"<span style='background:{color};color:white;padding:3px 12px;"
        f"border-radius:12px;font-weight:600;font-size:0.9em;'>"
        f"{decision.upper()}</span>"
    )


def mlflow_run_url(experiment_id: str | None, run_id: str | None) -> str | None:
    if not (experiment_id and run_id):
        return None
    return (
        f"{WORKSPACE_HOST}/ml/experiments/{experiment_id}/evaluation-runs?selectedRunUuid={run_id}"
    )


def mlflow_trace_url(experiment_id: str | None, trace_id: str | None) -> str | None:
    if not (experiment_id and trace_id):
        return None
    return f"{WORKSPACE_HOST}/ml/experiments/{experiment_id}/traces?selectedTraceId={trace_id}"


@dataclass
class RoundSummary:
    round_id: int
    decision: str
    action_kind: str
    aggregate: float | None
    score_delta: float | None
    parse_status: str


def all_round_summaries() -> list[RoundSummary]:
    out: list[RoundSummary] = []
    for rid in list_round_ids():
        raw = load_round_json(rid)
        if not raw:
            continue
        out.append(
            RoundSummary(
                round_id=rid,
                decision=raw.get("decision", "?"),
                action_kind=raw.get("action_kind", "?"),
                aggregate=raw.get("aggregate"),
                score_delta=raw.get("score_delta_vs_parent"),
                parse_status=raw.get("parse_status", "?"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


st.set_page_config(layout="wide", page_title="ANVIL — round explorer")

baseline = load_baseline()
mutations = load_mutations()
round_ids = list_round_ids()
summaries = all_round_summaries()

# ---- Sidebar -------------------------------------------------------------

st.sidebar.title("ANVIL — round explorer")

if not round_ids:
    st.sidebar.warning("No rounds found in eval/runs/.")
    st.stop()


def _label(rid: int) -> str:
    summary = next((s for s in summaries if s.round_id == rid), None)
    if summary is None:
        return f"R{rid}"
    delta_str = fmt_delta(summary.score_delta)
    return f"R{rid} · {summary.decision} · Δ {delta_str}"


selected_round = st.sidebar.selectbox(
    "Round",
    options=round_ids,
    index=len(round_ids) - 1,
    format_func=_label,
)

st.sidebar.markdown("---")

# KPI strip in the sidebar
n_total = len(mutations)
counts = {"keep": 0, "noop": 0, "revert": 0, "infra_fail": 0}
for r in mutations:
    counts[r.get("decision", "?")] = counts.get(r.get("decision", "?"), 0) + 1

st.sidebar.markdown("### Loop totals")
st.sidebar.metric("rounds run", n_total)
col_k, col_n = st.sidebar.columns(2)
col_k.metric("keeps", counts.get("keep", 0))
col_n.metric("noops", counts.get("noop", 0))
col_r, col_i = st.sidebar.columns(2)
col_r.metric("reverts", counts.get("revert", 0))
col_i.metric("infra fails", counts.get("infra_fail", 0))

if baseline:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Cached baseline")
    st.sidebar.metric(
        f"{baseline.get('mode', '?')} · n={baseline.get('n_examples', '?')}",
        fmt_score(baseline.get("aggregate")),
    )

# ---- Main area -----------------------------------------------------------

raw = load_round_json(selected_round) or {}

# Hero header.
header_cols = st.columns([1, 2, 1, 1])
header_cols[0].markdown(f"# Round {selected_round}")
header_cols[1].markdown(decision_pill(raw.get("decision", "?")), unsafe_allow_html=True)
header_cols[2].metric("aggregate", fmt_score(raw.get("aggregate")))
header_cols[3].metric(
    "Δ vs baseline",
    fmt_delta(raw.get("score_delta_vs_parent")),
)

st.markdown(
    f"**branch** `{raw.get('branch', '?')}` · "
    f"**action** `{raw.get('action_kind', '?')}` · "
    f"**parse_status** `{raw.get('parse_status', '?')}` · "
    f"**evaluated_at** {raw.get('evaluated_at', '—')}"
)

tabs = st.tabs(["Overview", "Mutation", "Failures", "Critique", "Curve"])

# === Overview tab =========================================================
with tabs[0]:
    per_judge = raw.get("per_judge") or {}
    cols = st.columns(4)
    cols[0].metric("aggregate", fmt_score(raw.get("aggregate")))
    cols[1].metric("correctness", fmt_score(per_judge.get("correctness")))
    cols[2].metric("retrieval_groundedness", fmt_score(per_judge.get("retrieval_groundedness")))
    cols[3].metric(
        "refusal_appropriateness",
        fmt_score(per_judge.get("refusal_appropriateness")),
    )

    per_bucket = raw.get("per_bucket") or {}
    if per_bucket:
        bucket_rows = []
        for bucket, scores in per_bucket.items():
            bucket_rows.append(
                {
                    "bucket": bucket,
                    "correctness": scores.get("correctness"),
                    "retrieval_groundedness": scores.get("retrieval_groundedness"),
                    "refusal_appropriateness": scores.get("refusal_appropriateness"),
                }
            )
        st.markdown("### Per-bucket")
        st.dataframe(pd.DataFrame(bucket_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No per-bucket data (likely a noop round).")

    st.markdown("### Action applied")
    mutation_row = next((m for m in mutations if m.get("round_id") == selected_round), None)
    if mutation_row:
        st.write(f"**diff_summary:** {mutation_row.get('diff_summary', '—')}")
        st.write(f"**files_added:** {mutation_row.get('files_added') or '—'}")
        st.write(f"**files_changed:** {mutation_row.get('files_changed') or '—'}")
        st.write(
            f"**git_commit_sha:** `{(mutation_row.get('git_commit_sha') or '')[:12]}` · "
            f"**parent_sha:** `{(mutation_row.get('parent_commit_sha') or '')[:12]}`"
        )
    else:
        st.info("No mutations row for this round.")

    st.markdown("### MLflow links")
    mlflow_block = raw.get("mlflow") or {}
    run_id = mlflow_block.get("run_id") or raw.get("run_id")
    experiment_id = mlflow_block.get("experiment_id") or raw.get("experiment_id")
    run_url = mlflow_run_url(experiment_id, run_id)
    if run_url:
        st.write(f"[Open eval run in MLflow]({run_url})")
        st.write(f"`run_id` `{run_id}` · `experiment_id` `{experiment_id}`")
    else:
        st.info("No MLflow run id recorded (likely a noop round without eval).")

# === Mutation tab =========================================================
with tabs[1]:
    stat, full_diff = get_diff_stat(selected_round)
    if not stat and not full_diff:
        st.info(
            f"No diff available — branch `anvil/exp-round-{selected_round}` may "
            "have been deleted (noop / revert) or git is unavailable."
        )
    else:
        if stat:
            st.markdown("### Files changed")
            st.code(stat, language="diff")
        if full_diff:
            st.markdown("### Patch")
            st.code(full_diff, language="diff")

# === Failures tab =========================================================
with tabs[2]:
    failures = raw.get("failures") or []
    if not failures:
        st.info("No failures recorded for this round.")
    else:
        rows = []
        for f in failures:
            tid = f.get("trace_id")
            tu = mlflow_trace_url(experiment_id, tid)
            rows.append(
                {
                    "example_id": f.get("example_id", ""),
                    "category": f.get("category", ""),
                    "query": f.get("query", "")[:120],
                    "judges_failed": ", ".join(f.get("judge_failures") or []),
                    "trace": f"[{tid[:12]}…]({tu})" if (tid and tu) else (tid or ""),
                }
            )
        st.markdown(f"### {len(rows)} failure(s)")
        # use markdown for the trace links
        try:
            st.write(pd.DataFrame(rows).to_markdown(index=False), unsafe_allow_html=False)
        except Exception:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# === Critique tab =========================================================
with tabs[3]:
    critique_text = load_critique(selected_round)
    if critique_text:
        st.markdown(critique_text)
    else:
        st.info("Critique md missing on disk. Falling back to mutation summary:")
        if mutation_row:
            st.write(mutation_row.get("diff_summary", "—"))

    transcript = load_transcript_head(selected_round)
    if transcript:
        with st.expander("Optimizer transcript (raw, truncated to 50KB)"):
            st.code(transcript)

# === Curve tab ============================================================
with tabs[4]:
    if not summaries:
        st.info("No round summaries.")
    else:
        df_curve = pd.DataFrame(
            [
                {
                    "round_id": s.round_id,
                    "decision": s.decision,
                    "aggregate": s.aggregate,
                    "score_delta": s.score_delta,
                    "action_kind": s.action_kind,
                    "parse_status": s.parse_status,
                }
                for s in summaries
            ]
        )

        st.markdown("### Aggregate per round")
        chart_df = df_curve.dropna(subset=["aggregate"]).copy()
        if not chart_df.empty:
            try:
                import altair as alt

                base = alt.Chart(chart_df).encode(
                    x=alt.X("round_id:Q", title="round"),
                    y=alt.Y("aggregate:Q", title="aggregate", scale=alt.Scale(domain=[0.7, 0.92])),
                )
                line = base.mark_line(strokeWidth=2, color="#888")
                points = base.mark_point(size=140, filled=True).encode(
                    color=alt.Color(
                        "decision:N",
                        scale=alt.Scale(
                            domain=list(DECISION_COLOR),
                            range=list(DECISION_COLOR.values()),
                        ),
                    ),
                    tooltip=["round_id", "decision", "aggregate", "score_delta", "action_kind"],
                )
                if baseline and baseline.get("aggregate") is not None:
                    rule = (
                        alt.Chart(pd.DataFrame({"y": [baseline["aggregate"]]}))
                        .mark_rule(strokeDash=[6, 4], color="#ff3621")
                        .encode(y="y:Q")
                    )
                    chart = (rule + line + points).properties(height=380)
                else:
                    chart = (line + points).properties(height=380)
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.line_chart(chart_df.set_index("round_id")["aggregate"])
        else:
            st.info("No measurable aggregates yet — all rounds are noops.")

        st.markdown("### All rounds")
        st.dataframe(df_curve, hide_index=True, use_container_width=True)
