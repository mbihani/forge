"""Durable persistence of optimizer logs + the run improvement summary.

Everything an ANVIL optimize session writes — the per-round optimizer
transcript, the per-round critique rationale, and (at the end of the
run) a human- and machine-readable improvement summary — otherwise lives
only inside the ephemeral ``/tmp/forge-sessions/<id>`` clone, which is
``shutil.rmtree``'d on shutdown. This module attaches those artifacts to
a single parent MLflow run under the ``experiments.optimizer`` experiment
(``/Shared/forge/<domain>/optimizer`` by default) so they SURVIVE session
shutdown.

The parent run is DISTINCT from the per-round eval runs under
``experiments.eval`` (``/Shared/forge/<domain>/eval``): those are created by
``mlflow.genai.evaluate`` and must not be disturbed. We never make the
optimizer run the *active* MLflow run — all writes go through an explicit
``MlflowClient`` keyed on the run id, so the eval runs' active-run stack
is untouched.

BEST-EFFORT POSTURE (critical): every MLflow call here is wrapped so that
ANY failure — unreachable tracking server, auth error, API error — logs a
warning and is swallowed. Persistence being down must NEVER abort or fail
an optimize round or finalize. This matches the repo's best-effort MLflow
stance (``MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=false`` + the resilient eval
harness).
"""

from __future__ import annotations

import json
import logging
import os
import socket
from pathlib import Path
from typing import Any

import yaml

from anvil.runtime.loader import default_runtime_config_path
from anvil.runtime.models import (
    ExperimentsConfig,
    check_domain_experiment,
    resolve_domain_experiments,
)

logger = logging.getLogger(__name__)

# Env override for the persistence toggle, env-authoritative over the
# file's ``persistence.enabled`` — mirrors ``ANVIL_OPTIMIZER_BACKEND`` for
# the optimizer backend (``round.py``'s ``_read_optimizer_config``). A
# deployment (e.g. the Databricks App) can force-enable or force-disable
# the sink without editing the cloned target repo's config.yaml.
_ENV_TOGGLE = "ANVIL_PERSIST_OPTIMIZER_ARTIFACTS"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _env_toggle() -> bool | None:
    """Parse ``ANVIL_PERSIST_OPTIMIZER_ARTIFACTS``.

    Returns ``True``/``False`` when set to a recognized value, ``None``
    when unset/empty (so the file value is preserved). An unrecognized
    value is treated as unset (``None``) so a typo cannot silently flip
    the toggle.
    """
    raw = os.getenv(_ENV_TOGGLE, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return None


def resolve_persistence_settings(scaffold_root: Path | str) -> tuple[bool, str]:
    """Resolve ``(enabled, experiment_name)`` for the optimizer artifact sink.

    * ``enabled`` — the ``ANVIL_PERSIST_OPTIMIZER_ARTIFACTS`` env var WINS
      when set (env-authoritative, matching the optimizer-backend
      precedence); otherwise the file's ``persistence.enabled`` (default
      ``True`` — enabled by default so a config without the block keeps
      persisting).
    * ``experiment_name`` — ``persistence.experiment`` when set, else the
      domain's optimizer experiment (``experiments.optimizer``, default
      ``<experiments.root>/<eval.engine>/optimizer``). Both must live in the
      domain's own folder — the same rule the config loader enforces — so a
      session never writes into another domain's experiment (``ValueError``).

    Only these keys are read here; full-file ``extra="forbid"`` validation
    is enforced by the runtime loader.
    """
    enabled = True
    raw: dict[str, Any] = {}
    path = default_runtime_config_path(Path(scaffold_root))
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    persistence = raw.get("persistence")
    persistence = persistence if isinstance(persistence, dict) else {}
    if "enabled" in persistence:
        enabled = bool(persistence["enabled"])
    eval_cfg = raw.get("eval") if isinstance(raw.get("eval"), dict) else {}
    domain = str(eval_cfg.get("engine") or "genai")
    experiments = resolve_domain_experiments(
        ExperimentsConfig.model_validate(raw.get("experiments") or {}), domain
    )
    experiment = experiments.optimizer
    if persistence.get("experiment"):
        experiment = check_domain_experiment(
            str(persistence["experiment"]),
            root=experiments.root,
            domain=domain,
            field="persistence.experiment",
        )
    env = _env_toggle()
    if env is not None:
        enabled = env
    return enabled, experiment


# ---------------------------------------------------------------------------
# The sink
# ---------------------------------------------------------------------------


class OptimizerArtifactSink:
    """Logs optimizer artifacts to one parent MLflow run, best-effort.

    Construct via :meth:`open` (creates a fresh parent run) or :meth:`bind`
    (attaches to an existing run id — used by finalize to append the
    held-out summary). Every method swallows exceptions and logs a
    warning; none raises, so a caller can wire these in without a
    try/except and the optimize round/finalize never fails on a dead sink.
    """

    def __init__(
        self,
        *,
        client: Any,
        run_id: str,
        experiment_id: str,
        session_url: str | None = None,
    ) -> None:
        self.client = client
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.session_url = session_url

    # ------------------------------------------------------------------
    # Construction (best-effort — return None rather than raise)
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        *,
        experiment_name: str,
        session_id: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> OptimizerArtifactSink | None:
        """Create a parent run under ``experiment_name``.

        Returns ``None`` on ANY failure (mlflow missing, tracking server
        unreachable, auth error) so persistence is simply skipped for the
        session — never fatal.
        """
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                from anvil.observability import ensure_experiment_folder  # noqa: PLC0415

                ensure_experiment_folder(experiment_name)
                experiment_id = client.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            run_tags = {
                "anvil.session_id": session_id,
                "anvil.kind": "optimizer-session",
            }
            if tags:
                run_tags.update(tags)
            run = client.create_run(
                experiment_id=experiment_id,
                run_name=run_name or f"optimize-{session_id}",
                tags=run_tags,
            )
            return cls(
                client=client,
                run_id=run.info.run_id,
                experiment_id=experiment_id,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; never fatal
            logger.warning(
                "optimizer artifact sink: could not open parent run under %r "
                "(persistence disabled for this session): %s",
                experiment_name,
                exc,
            )
            return None

    @classmethod
    def bind(
        cls,
        *,
        run_id: str,
        experiment_id: str,
        session_url: str | None = None,
    ) -> OptimizerArtifactSink | None:
        """Attach to an EXISTING parent run id (does not create a run).

        Used by finalize to append the held-out improvement summary to the
        run opened during optimization. Returns ``None`` on failure.
        """
        try:
            from mlflow.tracking import MlflowClient

            return cls(
                client=MlflowClient(),
                run_id=run_id,
                experiment_id=experiment_id,
                session_url=session_url,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; never fatal
            logger.warning("optimizer artifact sink: could not bind to run %s: %s", run_id, exc)
            return None

    # ------------------------------------------------------------------
    # Writes (best-effort — swallow + warn)
    # ------------------------------------------------------------------

    def log_round(self, *, round_id: int, transcript: str, critique_md: str) -> None:
        """Persist one round's optimizer transcript + critique rationale.

        Artifact layout under the parent run:
          ``rounds/round_NNN_transcript.md`` — the optimizer transcript
          ``rounds/round_NNN_critique.md``   — the critique rationale
        """
        self._log_text(transcript or "(empty)\n", f"rounds/round_{round_id:03d}_transcript.md")
        self._log_text(critique_md or "(empty)\n", f"rounds/round_{round_id:03d}_critique.md")

    def log_session(self, *, round_id: int, session_md: str) -> None:
        """Persist one round's full turn-by-turn optimizer session.

        ``rounds/round_NNN_session.md`` — every assistant message, tool call
        (input + result) and the session's final stats, unlike
        ``round_NNN_transcript.md`` which holds only the final text.
        """
        self._log_text(session_md or "(empty)\n", f"rounds/round_{round_id:03d}_session.md")

    def log_patch(self, *, round_id: int, patch: str) -> None:
        """The round's code change, kept for reverted rounds whose branch is deleted."""
        self._log_text(patch, f"rounds/round_{round_id:03d}.patch")

    def log_improvement_summary(self, *, summary: dict[str, Any], summary_md: str) -> None:
        """Persist the run-level improvement summary (JSON + markdown)."""
        self._log_text(json.dumps(summary, indent=2) + "\n", "improvement_summary.json")
        self._log_text(summary_md, "improvement_summary.md")

    def close(self, *, status: str = "FINISHED") -> None:
        """Terminate the parent run. Best-effort; a dead sink is a no-op."""
        try:
            self.client.set_terminated(self.run_id, status=status)
        except Exception as exc:  # noqa: BLE001 — best-effort; never fatal
            logger.warning(
                "optimizer artifact sink: could not terminate run %s: %s", self.run_id, exc
            )

    def _log_text(self, text: str, artifact_file: str) -> None:
        try:
            self.client.log_text(self.run_id, text, artifact_file)
        except Exception as exc:  # noqa: BLE001 — best-effort; never fatal
            logger.warning(
                "optimizer artifact sink: could not log %s to run %s: %s",
                artifact_file,
                self.run_id,
                exc,
            )


# ---------------------------------------------------------------------------
# Improvement summary assembly (pure — no MLflow, trivially testable)
# ---------------------------------------------------------------------------

_DECISION_KEYS = ("keep", "revert", "noop", "infra_fail")


def _as_aggregate(obj: Any) -> float | None:
    """Pull an ``aggregate`` float from a baseline/report dict, defensively."""
    if isinstance(obj, dict):
        val = obj.get("aggregate")
        if isinstance(val, int | float):
            return float(val)
    return None


def build_improvement_summary(
    *,
    session_meta: dict[str, Any],
    rounds: list[dict[str, Any]],
    frontier: dict[str, Any] | None,
    finalized: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the structured improvement summary from in-memory state.

    ``rounds`` are the per-round JSON dicts (``eval/runs/round_NNN.json``
    shape built by :func:`anvil.loop.round._build_round_json`), ``frontier``
    / ``finalized`` / ``baseline`` are their respective ``to_dict`` /
    on-disk shapes. Pure function — assembles, never does any I/O.

    ``net_improvement`` is ``baseline aggregate -> final aggregate`` where
    the final aggregate prefers the held-out finalized aggregate, then the
    frontier's best aggregate, then the last scored round's aggregate.
    """
    baseline_aggregate = _as_aggregate(baseline)

    decision_counts = dict.fromkeys(_DECISION_KEYS, 0)
    per_round: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for r in rounds:
        decision = str(r.get("decision") or "").lower()
        if decision in decision_counts:
            decision_counts[decision] += 1
        mutated = r.get("aggregate")
        # A noop is not evaluated; its effective score is the parent's.
        if mutated is None and decision == "noop":
            mutated = r.get("baseline_score")
        eval_run_id = (
            (r.get("mlflow") or {}).get("run_id") if isinstance(r.get("mlflow"), dict) else None
        )
        per_round.append(
            {
                "round": r.get("round_id"),
                "decision": r.get("decision"),
                "action_kind": r.get("action_kind"),
                "baseline_aggregate": r.get("baseline_score"),
                "mutated_aggregate": mutated,
                "delta": r.get("score_delta_vs_parent"),
                "rationale": r.get("rationale"),
                "eval_run_id": eval_run_id,
                "optimizer_error": r.get("optimizer_error"),
                "notes": r.get("notes"),
                "levers": r.get("levers"),
                **_round_cost_fields(r),
            }
        )
        trajectory.append({"round": r.get("round_id"), "aggregate": mutated})

    final_aggregate: float | None = _as_aggregate(finalized)
    if final_aggregate is None and isinstance(frontier, dict):
        best = frontier.get("best")
        if isinstance(best, dict) and isinstance(best.get("aggregate"), int | float):
            final_aggregate = float(best["aggregate"])
    if final_aggregate is None:
        # Without finalized/frontier data, the final outcome must reflect the
        # RETAINED agent, not merely the last evaluated mutation: walk back to
        # the last KEPT round's aggregate. A reverted/noop/infra_fail round's
        # score is NOT the final state (its mutation was discarded). When no
        # round was kept, the agent is still the baseline => final == baseline
        # (net improvement 0).
        for r in reversed(rounds):
            if str(r.get("decision") or "").lower() != "keep":
                continue
            agg = r.get("aggregate")
            if isinstance(agg, int | float):
                final_aggregate = float(agg)
                break
        if final_aggregate is None:
            final_aggregate = baseline_aggregate

    net_improvement = (
        final_aggregate - baseline_aggregate
        if (final_aggregate is not None and baseline_aggregate is not None)
        else None
    )

    return {
        "session": session_meta,
        "baseline_aggregate": baseline_aggregate,
        "final_aggregate": final_aggregate,
        "net_improvement": net_improvement,
        "rounds_total": len(rounds),
        "decision_counts": decision_counts,
        "score_trajectory": trajectory,
        "per_round": per_round,
        "frontier": frontier,
        "finalized_present": isinstance(finalized, dict),
    }


def _fmt(x: Any) -> str:
    """Format a numeric cell for the markdown table; '—' when absent."""
    if isinstance(x, int | float):
        return f"{x:.4f}"
    return "—"


def _round_cost_fields(r: dict[str, Any]) -> dict[str, Any]:
    cm = r.get("cost_metrics") if isinstance(r.get("cost_metrics"), dict) else {}
    return {
        k: cm.get(k)
        for k in ("latency_ms_median", "latency_ms_p90", "cost_usd_per_row", "n_row_errors")
        if cm.get(k) is not None
    }


def _ms(v: Any) -> str:
    return f"{v / 1000:.1f}s" if isinstance(v, int | float) else "—"


def _usd(v: Any) -> str:
    return f"${v:.5f}" if isinstance(v, int | float) else "—"


def render_improvement_summary_md(summary: dict[str, Any]) -> str:
    """Render the improvement summary dict as human-readable markdown."""
    s = summary.get("session", {})
    counts = summary.get("decision_counts", {})
    lines: list[str] = []
    lines.append(f"# Optimize session {s.get('session_id', '?')} — improvement summary")
    lines.append("")
    lines.append("## Session")
    lines.append("")
    lines.append(f"- **Repo:** {s.get('repo_url', '—')} @ `{s.get('agent_ref', '—')}`")
    lines.append(f"- **Mode:** {s.get('mode', '—')}")
    lines.append(f"- **Optimizer backend:** {s.get('backend', '—')}")
    lines.append(f"- **Started:** {s.get('started_at', '—')}")
    lines.append(f"- **Finished:** {s.get('finished_at', '—')}")
    if s.get("session_url"):
        lines.append(f"- **Optimizer session:** {s['session_url']}")
    lines.append("")
    lines.append("## Outcome")
    lines.append("")
    lines.append(f"- **Baseline aggregate:** {_fmt(summary.get('baseline_aggregate'))}")
    lines.append(f"- **Final aggregate:** {_fmt(summary.get('final_aggregate'))}")
    net = summary.get("net_improvement")
    net_str = f"{net:+.4f}" if isinstance(net, int | float) else "—"
    lines.append(f"- **Net improvement:** {net_str}")
    lines.append(f"- **Rounds:** {summary.get('rounds_total', 0)}")
    lines.append(
        f"- **Decisions:** kept {counts.get('keep', 0)} · reverted {counts.get('revert', 0)} · "
        f"noop {counts.get('noop', 0)} · infra_fail {counts.get('infra_fail', 0)}"
    )
    frontier = summary.get("frontier")
    if isinstance(frontier, dict) and isinstance(frontier.get("best"), dict):
        best = ", ".join(f"{k}={_fmt(v)}" for k, v in frontier["best"].items())
        lines.append(f"- **Frontier best:** {best}")
    lines.append("")
    lines.append("## Per-round")
    lines.append("")
    lines.append(
        "| Round | Decision | Action | Baseline | Mutated | Δ | Median | p90 | $/row "
        "| Eval run | Rationale |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary.get("per_round", []):
        delta = row.get("delta")
        delta_str = f"{delta:+.4f}" if isinstance(delta, int | float) else "—"
        text = row.get("notes") or row.get("rationale") or ""
        rationale = text.replace("\n", " ").replace("|", "\\|")
        if len(rationale) > 120:
            rationale = rationale[:117] + "..."
        lines.append(
            f"| {row.get('round', '?')} "
            f"| {row.get('decision', '—')} "
            f"| {row.get('action_kind', '—')} "
            f"| {_fmt(row.get('baseline_aggregate'))} "
            f"| {_fmt(row.get('mutated_aggregate'))} "
            f"| {delta_str} "
            f"| {_ms(row.get('latency_ms_median'))} "
            f"| {_ms(row.get('latency_ms_p90'))} "
            f"| {_usd(row.get('cost_usd_per_row'))} "
            f"| {row.get('eval_run_id') or '—'} "
            f"| {rationale or '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def persist_improvement_summary(
    sink: OptimizerArtifactSink | None,
    *,
    session_meta: dict[str, Any],
    rounds: list[dict[str, Any]],
    frontier: dict[str, Any] | None,
    finalized: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> None:
    """Assemble + log the improvement summary to ``sink``. Fully guarded.

    A ``None`` sink (persistence disabled or failed to open) is a no-op;
    any assembly/logging error is swallowed so finalize / the optimize
    task never fails on persistence.
    """
    if sink is None:
        return
    try:
        summary = build_improvement_summary(
            session_meta=session_meta,
            rounds=rounds,
            frontier=frontier,
            finalized=finalized,
            baseline=baseline,
        )
        sink.log_improvement_summary(
            summary=summary, summary_md=render_improvement_summary_md(summary)
        )
    except Exception as exc:  # noqa: BLE001 — best-effort; never fatal
        logger.warning("optimizer artifact sink: could not persist improvement summary: %s", exc)


# ---------------------------------------------------------------------------
# CLI sessions (scripts/run_round.py) — same durable sink as the orchestrator
# ---------------------------------------------------------------------------


def open_cli_session_sink(
    repo_root: Path | str, *, session_id: str, tags: dict[str, str] | None = None
) -> OptimizerArtifactSink | None:
    """Open the session's parent run for a scripts-driven optimize session.

    Mirrors the orchestrator: honours ``persistence.enabled`` (env
    ``ANVIL_PERSIST_OPTIMIZER_ARTIFACTS`` authoritative) and the optimizer
    experiment, creating the experiment if needed. Returns ``None`` when
    persistence is off or the sink cannot open — never fatal. The caller
    must point MLflow at the workspace first
    (:func:`anvil.observability.configure_tracking_uri`).
    """
    try:
        enabled, experiment = resolve_persistence_settings(Path(repo_root) / "scaffold")
    except Exception as exc:  # noqa: BLE001 — best-effort; never fatal
        logger.warning("could not resolve optimizer persistence settings: %s", exc)
        return None
    if not enabled:
        logger.info("optimizer-artifact persistence disabled (persistence.enabled / env)")
        return None
    sink = OptimizerArtifactSink.open(
        experiment_name=experiment,
        session_id=session_id,
        tags={
            "anvil.surface": "cli",
            _HOST_TAG: socket.gethostname(),
            _PID_TAG: str(os.getpid()),
            **(tags or {}),
        },
    )
    if sink is not None:
        reap_orphaned_cli_sessions(sink.client, sink.experiment_id, keep_run_id=sink.run_id)
    return sink


_HOST_TAG = "forge.host"
_PID_TAG = "forge.pid"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def reap_orphaned_cli_sessions(
    client: Any, experiment_id: str, *, keep_run_id: str | None = None
) -> list[str]:
    """Mark CLI session runs whose process is gone as ``KILLED``.

    A session killed with SIGKILL (or a crashed host process) never reaches
    its ``finally`` and leaves its parent run ``RUNNING`` with no artifacts.
    At the next session start, any ``RUNNING`` CLI run recorded on THIS host
    whose pid no longer exists is terminated (runs from other hosts, or
    without a pid tag, are left alone — they may be live). Best-effort;
    returns the reaped run ids.
    """
    reaped: list[str] = []
    try:
        host = socket.gethostname()
        runs = client.search_runs(
            [experiment_id],
            filter_string=(
                "attributes.status = 'RUNNING' and tags.`anvil.surface` = 'cli' "
                f"and tags.`{_HOST_TAG}` = '{host}'"
            ),
            max_results=100,
        )
        for run in runs:
            if run.info.run_id == keep_run_id:
                continue
            pid = str(run.data.tags.get(_PID_TAG, ""))
            if not pid.isdigit() or _pid_alive(int(pid)):
                continue
            client.set_tag(run.info.run_id, "forge.reaped", "orphaned session (process gone)")
            client.set_terminated(run.info.run_id, status="KILLED")
            reaped.append(run.info.run_id)
    except Exception as exc:  # noqa: BLE001 — best-effort; never fatal
        logger.warning("could not reap orphaned optimizer session runs: %s", exc)
    if reaped:
        logger.info("marked %d orphaned optimizer session run(s) KILLED", len(reaped))
    return reaped


def close_cli_session_sink(
    sink: OptimizerArtifactSink | None,
    repo_root: Path | str,
    *,
    round_ids: list[int],
    session_meta: dict[str, Any],
    status: str = "FINISHED",
) -> None:
    """Log the improvement summary from on-disk round records, then close the run.

    Reads ``eval/runs/round_NNN.json`` for the session's rounds plus
    ``frontier.json`` / ``baseline.json``. Best-effort; a ``None`` sink is a
    no-op.
    """
    if sink is None:
        return
    runs = Path(repo_root) / "eval" / "runs"

    def _read(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    rounds = [r for rid in round_ids if (r := _read(runs / f"round_{rid:03d}.json"))]
    persist_improvement_summary(
        sink,
        session_meta=session_meta,
        rounds=rounds,
        frontier=_read(runs / "frontier.json"),
        finalized=_read(runs / "finalized.json"),
        baseline=_read(runs / "baseline.json"),
    )
    sink.close(status=status)
