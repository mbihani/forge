"""Run one ANVIL round end-to-end.

The orchestrator. Reads cached baseline → spawns optimizer session →
applies the action → runs eval on the mutated branch → computes
score_delta → writes critique md + round JSON + mutations log row →
applies the keep/revert decision via the configured gate
(``harness/config.yaml > gate``; see :mod:`anvil.loop.frontier`).

Sync function. Internally calls ``asyncio.run`` to bridge to the
async ``run_optimizer_session``. Everything else is plain Python.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from anvil.loop.optimizer_artifacts import OptimizerArtifactSink

from anvil.catalog import load_catalog, model_prices
from anvil.eval import evaluate_branch, load_baseline
from anvil.eval.label_disputes import append_label_disputes
from anvil.loop.builder import build_round_prompt
from anvil.loop.decision import Decision, apply_optimizer_error
from anvil.loop.frontier import (
    gate_decision,
    load_gate_config,
    scores_from_baseline,
    scores_from_eval,
)
from anvil.loop.git_ops import (
    commit_all,
    commit_paths,
    create_round_branch,
    current_branch,
    current_sha,
    delete_branch,
    diff_patch,
    ff_merge,
)
from anvil.loop.mutations_log import MutationRecord, append_mutation
from anvil.loop.optimizer_trace import log_optimizer_trace, render_session_markdown
from anvil.optimizer import (
    BackendConfig,
    NoopAction,
    apply_action,
    get_backend,
    run_optimizer_session,
)
from anvil.optimizer.applier import ApplyError
from anvil.optimizer.code_validation import CodeValidationError
from anvil.optimizer.omnigent_backend import OmnigentBackend
from anvil.optimizer.omnigent_client import resolve_omnigent_server_url
from anvil.runtime.call_ledger import is_infra_error
from anvil.runtime.loader import default_runtime_config_path
from anvil.runtime.models import (
    CallPolicyConfig,
    LeverSpec,
    LoopConfig,
    ModelCatalogConfig,
    resolve_levers,
)


@dataclass
class RoundReport:
    round_id: int
    branch: str
    decision: Decision
    action_kind: str
    parse_status: str
    diff_summary: str
    mode: str = "prompt"
    files_added: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)
    baseline_score: float | None = None
    mutated_score: float | None = None
    score_delta: float | None = None
    eval_run_id: str | None = None
    git_commit_sha: str | None = None
    notes: str = ""


def run_round(
    *,
    round_id: int,
    repo_root: Path | str,
    scaffold_root: Path | str | None = None,
    profile: str = "DEFAULT",
    parent_branch: str = "anvil/exp",
    eval_mode: str | None = None,
    max_turns: int = 30,
    mode: str | None = None,
    artifact_sink: OptimizerArtifactSink | None = None,
) -> RoundReport:
    """Run round ``round_id`` end-to-end. Returns a report.

    Side effects:
      * Creates ``anvil/exp-round-<N>`` from ``parent_branch``.
      * May write under ``scaffold/`` (the applier).
      * Commits the change to the round branch.
      * Runs eval against the mutated branch.
      * Writes ``scaffold/memory/round_<N>_critique.md``,
        ``eval/runs/round_<N>.json``, appends to
        ``eval/mutations.jsonl``.
      * On KEEP: ff-merges round branch into ``parent_branch``.
        On REVERT/INFRA_FAIL: deletes the round branch.

    ``mode`` (optimization mode: "prompt" or "code") overrides the value
    read from ``harness/config.yaml`` for this round when provided. When
    ``None`` (the default) the mode is read from disk via
    :func:`_read_optimization_mode`, preserving backward compatibility.

    ``artifact_sink`` (optional) is a per-session
    :class:`~anvil.loop.optimizer_artifacts.OptimizerArtifactSink`. When
    supplied, the round ADDITIONALLY logs the optimizer transcript + the
    critique rationale to the session's durable parent MLflow run (the
    clone-local writes are kept — the MLflow write is additive and
    best-effort, so a dead sink never fails the round). ``None`` (the
    default, e.g. a bare ``run_round`` CLI call) skips durable logging.
    """
    repo_root = Path(repo_root).resolve()
    scaffold_root = Path(scaffold_root or (repo_root / "scaffold")).resolve()

    mode = mode if mode is not None else _read_optimization_mode(scaffold_root)
    optimizer_endpoint = _read_optimizer_endpoint(scaffold_root)
    print(f"[round {round_id}] mode={mode}")

    _starting_branch = current_branch(repo_root)
    parent_sha = current_sha(repo_root)
    baseline = load_baseline(repo_root)

    # 1. Branch off the parent.
    branch = create_round_branch(repo_root, round_id, parent_branch=parent_branch)

    # 2. Build prompt and run the optimizer session.
    prompt_gate_cfg = load_gate_config(scaffold_root)
    loop_cfg = _read_loop_config(scaffold_root)
    lever_specs = _read_lever_specs_for_prompt(scaffold_root)
    prompt = build_round_prompt(
        repo_root=repo_root,
        round_id=round_id,
        baseline=asdict_baseline(baseline) if baseline else None,
        objectives=(
            prompt_gate_cfg.pareto.objectives or None if prompt_gate_cfg.pareto.enabled else None
        ),
        max_mutations=loop_cfg.max_mutations_per_round,
        max_turns=max_turns,
        lever_specs=lever_specs,
        lever_values=_effective_levers(scaffold_root),
        model_prices=_model_prices_for_prompt(scaffold_root, lever_specs),
        engine=_read_eval_engine(scaffold_root),
        call_policy=_read_call_policy(scaffold_root),
    )

    # Optimizer-side MLflow tracing does NOT wrap the async SDK session
    # (MLflow's async tracing interacts badly with ``claude-agent-sdk``; see
    # https://docs.databricks.com/aws/en/mlflow3/genai/tracing/integrations/claude-code).
    # Instead the local backend records the session as plain events
    # (``session_record``) and step 8c replays them into a trace after the
    # session returns. The final text also lives at
    # ``scaffold/memory/round_NNN_transcript.md``.
    optimizer_cfg = _read_optimizer_config(scaffold_root)
    # ``optimizer_error`` is set only when the optimizer BACKEND itself
    # failed (auth/401, SSO redirect, connection error, ...). The local
    # backend reports a session that ended in an API error the same way.
    optimizer_error: str | None = None
    # Selected backend + (omnigent only) resolved server URL, recorded into
    # the round JSON so a 'local noop' is distinguishable from an
    # 'omnigent noop/INFRA_FAIL' WITHOUT reading the transcript file.
    selected_backend = optimizer_cfg.get("backend") or "local"
    resolved_optimizer_server_url: str | None = None
    # Managed-omnigent conversation URL — a durable pointer to the
    # optimizer session's transcript on the Omnigent server. None on the
    # local backend. Recorded into the round JSON for discoverability.
    optimizer_session_url: str | None = None
    # The local backend fills this with the full turn-by-turn session, which
    # step 8b replays into an MLflow trace (see loop/optimizer_trace.py).
    session_record: dict = {}
    if selected_backend == "omnigent":
        resolved_optimizer_server_url = _resolve_omnigent_backend_url(optimizer_cfg)
        action, transcript, parse_result, optimizer_error, optimizer_session_url = asyncio.run(
            _run_omnigent_session(
                prompt=prompt,
                repo_root=repo_root,
                optimizer_cfg=optimizer_cfg,
                max_turns=max_turns,
                optimizer_endpoint=optimizer_endpoint,
                server_url=resolved_optimizer_server_url,
            )
        )
    else:
        action, transcript, parse_result = asyncio.run(
            run_optimizer_session(
                prompt=prompt,
                cwd=str(repo_root),
                max_turns=max_turns,
                profile=profile,
                optimizer_endpoint=optimizer_endpoint,
                session_record=session_record,
            )
        )
        optimizer_error = session_record.get("optimizer_error")

    # 3. Apply the action (writes scaffold files, edits harness.yaml).
    #
    # An action the applier rejects (a lever value outside its allowlist, a
    # compound over loop.max_mutations_per_round, an edit of a missing file,
    # agent code that fails the AST denylist / isolated import) is recorded
    # as a noop with the reason, instead of crashing the whole multi-round
    # run. Compound actions restore scaffold/ + agents/ on
    # failure, so nothing half-applied is left behind.
    # Disputed golden labels ride on any action (read before an applier
    # rejection replaces the action with a noop).
    label_disputes = list(getattr(action, "label_disputes", None) or [])
    apply_error: str | None = None
    try:
        apply_result = apply_action(action, scaffold_root, mode=mode, repo_root=repo_root)
    except (ApplyError, CodeValidationError) as exc:
        apply_error = f"{exc}"
        print(f"[round {round_id}] applier rejected {action.action}: {apply_error}")
        action = NoopAction(rationale=f"applier rejected {action.action}: {apply_error}"[:2000])
        apply_result = apply_action(action, scaffold_root, mode=mode, repo_root=repo_root)
    # Lever values THIS round's eval runs with (e.g. which model). Captured
    # now, before a revert can check out the parent, for the round JSON.
    round_levers = _effective_levers(scaffold_root)
    new_disputes = 0
    try:
        new_disputes = append_label_disputes(repo_root, round_id, label_disputes)
    except Exception as exc:  # noqa: BLE001 - bookkeeping must never fail a round
        print(f"[round {round_id}] warning: recording label disputes failed: {exc}")
    if new_disputes:
        print(f"[round {round_id}] {new_disputes} new disputed label(s) recorded")

    # 4. Commit the mutation — but only when the applier actually wrote
    # something. A parse-failure noop (parse_status=no_block) collapses
    # to a NoopAction whose applier writes no files; committing on the
    # resulting empty index exits non-zero ("no changes added to commit")
    # and aborts the whole multi-round run. Detect no-change explicitly
    # from the applier's file lists (the robust signal) rather than
    # relying on the commit failing, and record the parent SHA instead.
    # ``commit_all`` is itself hardened against an empty index, so a
    # real mutation whose written content is byte-identical to HEAD
    # (files_changed populated but nothing staged) still returns the
    # current SHA instead of raising.
    applied_change = bool(
        apply_result.files_added or apply_result.files_changed or apply_result.files_removed
    )
    if applied_change:
        commit_message = f"round {round_id:03d}: {apply_result.action_summary or 'noop'}"
        commit_sha = commit_all(repo_root, message=commit_message)
    else:
        commit_sha = current_sha(repo_root)

    # 5. Persist transcript (debug aid; not the critique md yet).
    transcript_path = repo_root / "scaffold" / "memory" / f"round_{round_id:03d}_transcript.md"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(transcript or "(empty)\n", encoding="utf-8")

    # 6. Run eval on the mutated branch (or skip if noop).
    mutated_score: float | None = None
    eval_run_id: str | None = None
    eval_failed = False
    eval_report = None
    canary_note: str | None = None
    if isinstance(action, NoopAction):
        # No need to evaluate for a noop; the score is the parent's.
        if baseline:
            mutated_score = float(baseline.aggregate)
    else:
        # 6a. Canary: the first rows alone. If every one fails with an error
        # (a rejected parameter, a crash in the new code), revert now with
        # that error instead of spending the full eval on it.
        if loop_cfg.canary_rows:
            canary_note = _run_canary(
                scaffold_root=scaffold_root,
                repo_root=repo_root,
                profile=profile,
                eval_mode=eval_mode,
                n_rows=loop_cfg.canary_rows,
            )
            if canary_note:
                print(f"[round {round_id}] {canary_note}")
        if canary_note is None:
            try:
                eval_report = evaluate_branch(
                    scaffold_root=scaffold_root,
                    profile=profile,
                    golden_set_path=str(Path(repo_root) / "data" / "golden_set.jsonl"),
                    mode=eval_mode,
                )
                mutated_score = eval_report.aggregate
                eval_run_id = eval_report.run_id
            except Exception as exc:  # noqa: BLE001 — surface any eval failure
                eval_failed = True
                mutated_score = None
                notes = f"eval failed: {exc.__class__.__name__}: {exc}"
                print(f"[round {round_id}] eval failure: {notes}")

    # 7. Compute score delta + decision.
    #
    # ``score_delta`` is reported vs the cached (frozen) baseline for the
    # human-facing artifacts (round JSON, mutations log) — it shows how the
    # mutation compares to the *original* baseline. The keep/revert DECISION
    # is delegated to the configured gate (``harness/config.yaml > gate``):
    #
    # * ``gate.type: frontier`` (default) — Pareto frontier. The decision
    #   compares the mutation against the best-so-far per objective
    #   (per-judge scores + aggregate), persisted to
    #   ``eval/runs/frontier.json``. A round that scores worse than a
    #   previously KEPT round is REVERTED even if it still beats the frozen
    #   baseline — the fix for the silent regression the frozen-baseline
    #   delta gate allowed. On the first scored round the frontier is
    #   initialized from the baseline.
    # * ``gate.type: delta`` — legacy frozen-baseline behavior (kept for
    #   backward compatibility); the decision reproduces the old
    #   ``score_delta > 0`` check exactly and does not touch the frontier.
    baseline_aggregate = float(baseline.aggregate) if baseline else None
    score_delta = (
        (mutated_score - baseline_aggregate)
        if (mutated_score is not None and baseline_aggregate is not None)
        else None
    )

    # Validate scorer-config compatibility between the cached baseline
    # and the current eval run. A weight or check_function change
    # invalidates the comparison even when scorer names are unchanged —
    # the cached aggregate has a different meaning under a different
    # weighting, so the frontier gate could make an invalid decision.
    # An empty fingerprint on either side (e.g. a baseline written
    # before this field existed) skips the check for backward compat.
    if (
        baseline is not None
        and eval_report is not None
        and baseline.scorer_fingerprint
        and eval_report.scorer_fingerprint
        and baseline.scorer_fingerprint != eval_report.scorer_fingerprint
    ):
        raise RuntimeError(
            "scorer configuration has changed since baseline was cached — "
            "regenerate the baseline with scripts/make_baseline.py"
        )

    gate_cfg = load_gate_config(scaffold_root)
    configured_objectives = gate_cfg.pareto.objectives if gate_cfg.pareto.enabled else None
    if gate_cfg.pareto.enabled and not configured_objectives:
        configured_objectives = None
    baseline_scores = scores_from_baseline(baseline, configured_objectives) if baseline else None
    mutated_scores = (
        scores_from_eval(eval_report, configured_objectives) if eval_report is not None else None
    )
    canary_decision = (
        Decision.INFRA_FAIL
        if canary_note and "transport errors" in canary_note
        else Decision.REVERT
    )
    decision, frontier = (
        (canary_decision, None)
        if canary_note
        else gate_decision(
            repo_root=repo_root,
            gate_type=gate_cfg.type,
            epsilon=gate_cfg.epsilon,
            pareto=gate_cfg.pareto,
            baseline_scores=baseline_scores,
            baseline_aggregate=baseline_aggregate,
            mutated_scores=mutated_scores,
            mutated_aggregate=mutated_score,
            action_kind=action.action,
            eval_failed=eval_failed,
            parse_status=parse_result.parse_status,
        )
    )

    # 7b. A swallowed optimizer-backend failure must NOT be reported as a
    # clean noop. When the backend errored, the transcript parsed to a
    # NoopAction (so ``gate_decision`` returned NOOP with no frontier I/O);
    # override to INFRA_FAIL and carry the error into the round artifacts so
    # a 401 / SSO redirect / connection error is visibly a failure, not
    # "Status: optimized, noop". Distinct from a legitimate optimizer noop,
    # which leaves ``optimizer_error`` None and keeps the NOOP decision.
    decision = apply_optimizer_error(decision, optimizer_error)
    if optimizer_error:
        print(f"[round {round_id}] optimizer backend failure: {optimizer_error}")

    # 8. Write critique md.
    critique_md = _build_critique_md(
        round_id=round_id,
        branch=branch,
        decision=decision,
        action_kind=action.action,
        apply_summary=apply_result.action_summary,
        rationale=action.rationale,
        baseline_score=baseline_aggregate,
        mutated_score=mutated_score,
        score_delta=score_delta,
        parse_status=parse_result.parse_status,
    )
    critique_path = repo_root / "scaffold" / "memory" / f"round_{round_id:03d}_critique.md"
    critique_path.write_text(critique_md, encoding="utf-8")

    # 8b. Durably persist the per-round LOGS (transcript + critique) to the
    # session's parent MLflow run, when a sink is supplied. This is the
    # survives-shutdown write: the clone-local files above are wiped when
    # ``/tmp/forge-sessions/<id>`` is rmtree'd. We log HERE — while the
    # transcript + critique are in hand and BEFORE the round branch may be
    # deleted on REVERT/INFRA_FAIL (which would otherwise drop the artifacts
    # for exactly the failing rounds). Fully best-effort: ``log_round``
    # swallows any MLflow failure internally, and we ALSO guard the
    # invocation here so a future sink implementation error can never fail
    # the round either.
    if artifact_sink is not None:
        try:
            artifact_sink.log_round(
                round_id=round_id,
                transcript=transcript or "(empty)\n",
                critique_md=critique_md,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; never fail the round
            print(f"[round {round_id}] warning: optimizer artifact log_round failed: {exc}")

    # 8c. Replay the recorded optimizer session (local backend) into an
    # MLflow trace in the optimizer experiment, linked to the session run,
    # plus a turn-by-turn markdown artifact. Synchronous and after the
    # session — no async tracing inside the SDK. Best-effort.
    optimizer_trace_id: str | None = None
    events = session_record.get("events") or []
    decision_str = str(getattr(decision, "value", decision))
    if artifact_sink is not None and events:
        try:
            artifact_sink.log_session(
                round_id=round_id,
                session_md=render_session_markdown(
                    events, title=f"Optimizer session — round {round_id} ({branch})"
                ),
            )
            optimizer_trace_id = log_optimizer_trace(
                events=events,
                prompt=prompt,
                experiment_id=artifact_sink.experiment_id,
                started_ns=session_record.get("started_ns"),
                ended_ns=session_record.get("ended_ns"),
                tags={
                    "round": str(round_id),
                    "scaffold_branch": branch,
                    "decision": decision_str,
                    "action": action.action,
                    "mode": str(mode),
                },
                outputs={
                    "decision": decision_str,
                    "action": action.action,
                    "rationale": action.rationale,
                    "parse_status": parse_result.parse_status,
                },
                source_run_id=artifact_sink.run_id,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; never fail the round
            print(f"[round {round_id}] warning: optimizer session trace failed: {exc}")

    # 9. Write round JSON (combines aggregate + decision + delta).
    round_json_path = repo_root / "eval" / "runs" / f"round_{round_id:03d}.json"
    round_json_path.parent.mkdir(parents=True, exist_ok=True)
    round_payload = (
        json.dumps(
            _build_round_json(
                round_id=round_id,
                branch=branch,
                commit_sha=commit_sha,
                parent_sha=parent_sha,
                decision=decision,
                action_kind=action.action,
                eval_report=eval_report,
                baseline_score=baseline_aggregate,
                score_delta=score_delta,
                parse_status=("apply_rejected" if apply_error else parse_result.parse_status),
                notes=(
                    f"optimizer backend failed: {optimizer_error}"
                    if optimizer_error
                    else (
                        f"applier rejected action: {apply_error}"
                        if apply_error
                        else (canary_note or "")
                    )
                ),
                levers=round_levers,
                frontier_best=frontier.best if frontier else None,
                optimizer_error=optimizer_error,
                optimizer_backend=selected_backend,
                optimizer_server_url=resolved_optimizer_server_url,
                rationale=action.rationale,
                optimizer_run_id=artifact_sink.run_id if artifact_sink else None,
                optimizer_experiment_id=artifact_sink.experiment_id if artifact_sink else None,
                optimizer_session_url=optimizer_session_url,
                optimizer_trace_id=optimizer_trace_id,
            ),
            indent=2,
        )
        + "\n"
    )
    round_json_path.write_text(round_payload, encoding="utf-8")
    _save_dashboard_round(repo_root, round_id, round_payload)

    # 10. Append mutations log.
    record = MutationRecord.new(
        round_id=round_id,
        git_branch=branch,
        git_commit_sha=commit_sha,
        parent_commit_sha=parent_sha,
        files_added=apply_result.files_added,
        files_changed=apply_result.files_changed,
        files_removed=apply_result.files_removed,
        diff_summary=apply_result.action_summary,
        baseline_score=baseline_aggregate,
        mutated_score=mutated_score,
        score_delta=score_delta,
        decision=decision.value,
        mlflow_eval_run_id=eval_run_id,
        parse_status=parse_result.parse_status,
    )
    append_mutation(repo_root, record)

    # 10b. Commit the round artifacts (critique md, transcript md,
    # round JSON) onto the round branch BEFORE we ff-merge or delete
    # it. Without this step the critique md was orphaned: the only
    # commit_all (step 4) ran before the critique was written, so the
    # file lived as untracked in the working tree and got clobbered
    # by the next round's git checkout. Fixed in this commit; the
    # legacy artifacts for R4/R6/R8 were reconstructed via
    # ``scripts/reconstruct_critiques.py``.
    try:
        commit_all(repo_root, message=f"round {round_id:03d}: critique + eval artifacts")
    except Exception as exc:  # noqa: BLE001 — defensive, don't break the round
        print(f"[round {round_id}] warning: artifacts commit failed: {exc}")

    # 10c. Save the round's code change as a patch BEFORE a revert deletes
    # the branch, so reverted proposals stay inspectable (and re-appliable).
    patch_path = repo_root / "eval" / "runs" / f"round_{round_id:03d}.patch"
    try:
        patch = diff_patch(repo_root, parent_sha, commit_sha) if applied_change else ""
        if patch:
            patch_path.write_text(patch, encoding="utf-8")
            if artifact_sink is not None:
                artifact_sink.log_patch(round_id=round_id, patch=patch)
    except Exception as exc:  # noqa: BLE001 — best-effort; never fail the round
        print(f"[round {round_id}] warning: saving the round patch failed: {exc}")

    # 11. Apply git verdict.
    if decision == Decision.KEEP:
        ff_merge(repo_root, branch=branch, target=parent_branch)
    elif decision in (Decision.REVERT, Decision.INFRA_FAIL):
        delete_branch(repo_root, branch=branch, target=parent_branch)
    else:
        # NOOP: keep the (empty) branch around? Cheaper to delete.
        delete_branch(repo_root, branch=branch, target=parent_branch)

    # 12. Record the round on the parent branch — kept AND reverted rounds —
    # so history holds every round's JSON / patch / frontier / mutation log
    # and the tree is clean for the next round (no --allow-dirty to resume).
    try:
        commit_paths(
            repo_root,
            [
                f"eval/runs/round_{round_id:03d}.json",
                f"eval/runs/round_{round_id:03d}.patch",
                "eval/runs/frontier.json",
                "eval/mutations.jsonl",
                "eval/label_disputes.jsonl",
            ],
            message=f"round {round_id:03d}: record ({decision.value})",
        )
    except Exception as exc:  # noqa: BLE001 — best-effort; never fail the round
        print(f"[round {round_id}] warning: recording the round on {parent_branch} failed: {exc}")

    # We don't hard-restore the starting branch — caller's session is now on
    # parent_branch, which is correct: that's where the kept mutation
    # lives. Worth printing for clarity.
    print(
        f"[round {round_id}] {decision} · action={action.action} · "
        f"baseline={baseline_aggregate} · mutated={mutated_score} · Δ={score_delta}"
    )

    return RoundReport(
        round_id=round_id,
        branch=branch,
        decision=decision,
        action_kind=action.action,
        parse_status=parse_result.parse_status,
        diff_summary=apply_result.action_summary,
        mode=mode,
        files_added=apply_result.files_added,
        files_changed=apply_result.files_changed,
        files_removed=apply_result.files_removed,
        baseline_score=baseline_aggregate,
        mutated_score=mutated_score,
        score_delta=score_delta,
        eval_run_id=eval_run_id,
        git_commit_sha=commit_sha,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def asdict_baseline(baseline) -> dict:
    """Convert a CachedBaseline to a plain dict for the prompt builder."""
    return baseline.to_dict()


def _save_dashboard_round(repo_root: Path | str, round_id: int, payload: str) -> Path:
    """Persist serialized round JSON in the dashboard data directory."""
    path = Path(repo_root) / "data" / f"round_{round_id:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _effective_levers(scaffold_root: Path | str) -> dict[str, Any]:
    """Resolved lever values for the scaffold as it is on disk right now.

    Merges ``scaffold/harness.yaml > levers`` with the declared specs'
    defaults; for a declared ``model`` lever with no value, reports the base
    ``runtime_endpoint``. Best-effort: returns ``{}`` on any read/validation
    problem — this is a record, not a gate.
    """
    scaffold_root = Path(scaffold_root)
    try:
        raw_cfg = (
            yaml.safe_load(default_runtime_config_path(scaffold_root).read_text(encoding="utf-8"))
            or {}
        )
        raw_scaffold = (
            yaml.safe_load((scaffold_root / "harness.yaml").read_text(encoding="utf-8")) or {}
        )
        specs = {n: LeverSpec.model_validate(s) for n, s in (raw_cfg.get("levers") or {}).items()}
        resolved: dict[str, Any] = resolve_levers(specs, raw_scaffold.get("levers") or {})
    except Exception:  # noqa: BLE001 - a record, never fail the round
        return {}
    if "model" in specs and "model" not in resolved and raw_cfg.get("runtime_endpoint"):
        resolved["model"] = raw_cfg["runtime_endpoint"]
    return resolved


def _read_loop_config(scaffold_root: Path | str) -> LoopConfig:
    """``harness/config.yaml > loop`` (defaults when absent/invalid)."""
    try:
        raw = (
            yaml.safe_load(default_runtime_config_path(scaffold_root).read_text(encoding="utf-8"))
            or {}
        )
        return LoopConfig.model_validate(raw.get("loop") or {})
    except Exception:  # noqa: BLE001 - fall back to the classic defaults
        return LoopConfig()


def _read_eval_section(scaffold_root: Path | str) -> dict[str, Any]:
    try:
        raw = (
            yaml.safe_load(default_runtime_config_path(scaffold_root).read_text(encoding="utf-8"))
            or {}
        )
        return raw.get("eval") or {}
    except Exception:  # noqa: BLE001 - absent/invalid config: defaults
        return {}


def _read_eval_engine(scaffold_root: Path | str) -> str:
    return str(_read_eval_section(scaffold_root).get("engine") or "genai")


def _read_call_policy(scaffold_root: Path | str) -> CallPolicyConfig:
    try:
        return CallPolicyConfig.model_validate(
            _read_eval_section(scaffold_root).get("call_policy") or {}
        )
    except Exception:  # noqa: BLE001 - invalid section: no policy
        return CallPolicyConfig()


def _run_canary(
    *,
    scaffold_root: Path,
    repo_root: Path,
    profile: str,
    eval_mode: str | None,
    n_rows: int,
) -> str | None:
    """Evaluate the first ``n_rows``; a note when EVERY one failed with an error.

    Returns ``None`` (go on to the full eval) when any row produced a result,
    when the engine reports no row errors, or when the canary itself raised.
    """
    try:
        report = evaluate_branch(
            scaffold_root=scaffold_root,
            profile=profile,
            golden_set_path=str(Path(repo_root) / "data" / "golden_set.jsonl"),
            mode=eval_mode,
            max_rows=n_rows,
        )
    except Exception as exc:  # noqa: BLE001 - let the full eval surface it
        print(f"canary raised {exc.__class__.__name__}: {exc}; running the full eval")
        return None
    errors = [f["error"] for f in (report.failures or []) if f.get("error")]
    n = int(report.n_rows or 0)
    if n == 0 or len(errors) < n:
        return None
    if all(is_infra_error(e) for e in errors):
        # An outage, not the mutation: the round is an infra failure.
        return f"canary: all {n} rows failed on transport errors — {errors[0][:400]}"
    return f"canary: all {n} rows failed — {errors[0][:400]}"


def _read_lever_specs_for_prompt(scaffold_root: Path | str) -> dict[str, LeverSpec]:
    """Declared lever specs for the round prompt (``{}`` when absent/invalid)."""
    try:
        raw = (
            yaml.safe_load(default_runtime_config_path(scaffold_root).read_text(encoding="utf-8"))
            or {}
        )
        return {n: LeverSpec.model_validate(s) for n, s in (raw.get("levers") or {}).items()}
    except Exception:  # noqa: BLE001
        return {}


def _model_prices_for_prompt(
    scaffold_root: Path | str, lever_specs: dict[str, LeverSpec]
) -> dict[str, dict[str, Any]] | None:
    """Prices for the ``model`` lever's allowed values (None when no model lever).

    Read from the synced ``model_catalog`` CSV (LiteLLM fallback). Best-effort:
    a missing/unreadable catalog just shows "price unknown" in the prompt.
    """
    spec = lever_specs.get("model")
    if spec is None:
        return None
    try:
        raw = (
            yaml.safe_load(default_runtime_config_path(scaffold_root).read_text(encoding="utf-8"))
            or {}
        )
        cfg = ModelCatalogConfig.model_validate(raw.get("model_catalog") or {})
        entries = load_catalog(Path(scaffold_root).parent / cfg.path)
        return model_prices([str(m) for m in spec.allowed], entries, tier=cfg.context_tier)
    except Exception:  # noqa: BLE001 - prices are context for the optimizer, not a gate
        return None


def _read_optimization_mode(scaffold_root: Path | str) -> str:
    """Read the optimization mode from ``harness/config.yaml``.

    Returns ``"prompt"`` (the default) when the file or field is absent,
    so the loop keeps running on a repo that predates the mode field.
    Only the ``mode`` key is read here; the full-file ``extra="forbid"``
    check is enforced by the runtime loader.

    Invalid values (e.g. ``hybrid``, a typo, or an empty string) raise
    :class:`ValueError` so the loop fails closed at the source rather
    than silently permitting every action downstream.
    """
    path = default_runtime_config_path(Path(scaffold_root))
    if not path.is_file():
        return "prompt"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mode = raw.get("mode", "prompt")
    if mode not in ("prompt", "code"):
        raise ValueError(f"unknown optimization mode {mode!r}; expected 'prompt' or 'code'")
    return mode


def _read_optimizer_endpoint(scaffold_root: Path | str) -> str | None:
    """Read the ``optimizer_endpoint`` FMAPI model from harness/config.yaml.

    Returns None when the file or field is absent, so the optimizer
    session falls back to its built-in default model. Only this field is
    read here; the full-file ``extra="forbid"`` validation is enforced
    by the runtime loader.
    """
    path = default_runtime_config_path(Path(scaffold_root))
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("optimizer_endpoint")


def _read_optimizer_config(scaffold_root: Path | str) -> dict[str, Any]:
    """Read the ``optimizer`` section from harness/config.yaml.

    Returns an empty dict when the file or section is absent — backward
    compatible: no ``optimizer`` block means the local backend (the
    existing ``run_optimizer_session`` path). Only the ``optimizer`` key
    is read here; full-file validation is enforced by the runtime loader.

    The ``backend`` key can be overridden via the
    ``ANVIL_OPTIMIZER_BACKEND`` env var, so a deployment (e.g. a
    Databricks App) can force the omnigent backend without the cloned
    target repo needing an ``optimizer:`` section in its config.yaml.

    Precedence for ``backend``: the ``ANVIL_OPTIMIZER_BACKEND`` env var
    WINS over the file's value whenever it is set and non-empty. forge is
    domain-agnostic and clones arbitrary target repos whose
    ``harness/config.yaml`` we do not control — the optimizer backend is
    a DEPLOYMENT/INFRA concern, so the deployment env must be
    authoritative over a file-pinned ``optimizer.backend`` (e.g. a repo
    that pins ``backend: local`` must still run omnigent when the App
    sets ``ANVIL_OPTIMIZER_BACKEND=omnigent``). When the env var is
    unset or empty, the file value is preserved unchanged so local dev
    keeps whatever ``harness/config.yaml`` pins.
    """
    path = default_runtime_config_path(Path(scaffold_root))
    if not path.is_file():
        optimizer_cfg: dict[str, Any] = {}
    else:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        optimizer = raw.get("optimizer")
        optimizer_cfg = optimizer if isinstance(optimizer, dict) else {}
    # Env-var override lets a deployment select the backend authoritatively
    # over the cloned repo's file config. When set and non-empty, the env
    # var WINS (overriding even a truthy file value); when unset/empty the
    # file value is preserved.
    env_backend = os.getenv("ANVIL_OPTIMIZER_BACKEND", "")
    if env_backend:
        optimizer_cfg["backend"] = env_backend
    return optimizer_cfg


def _resolve_omnigent_backend_url(optimizer_cfg: dict[str, Any]) -> str:
    """Resolve the Omnigent server URL for the backend, env-authoritative.

    forge clones arbitrary target repos whose ``harness/config.yaml`` may
    pin an ``optimizer.server_url`` (e.g. ``http://localhost:6767``) that
    is meaningless in a deployed App. The server URL is a DEPLOYMENT/INFRA
    concern, so precedence is:

    1. ``OMNIGENT_SERVER_URL`` env (non-empty) — explicit deployment pin;
    2. derived ``{DATABRICKS_HOST}/omnigent`` when ``DATABRICKS_HOST`` is
       present — the workspace-native Omnigent surface;
    3. the file-pinned ``optimizer.server_url`` — local dev fallback only;
    4. ``http://localhost:6767`` — last-resort default.

    Steps 1 + 2 are exactly what :func:`resolve_omnigent_server_url`
    returns, so a file ``server_url`` NEVER shadows the env/derived URL
    when either source yields a value.
    """
    return (
        resolve_omnigent_server_url() or optimizer_cfg.get("server_url") or "http://localhost:6767"
    )


async def _run_omnigent_session(
    *,
    prompt: str,
    repo_root: Path,
    optimizer_cfg: dict[str, Any],
    max_turns: int,
    optimizer_endpoint: str | None,
    server_url: str,
) -> tuple[Any, str, Any, str | None, str | None]:
    """Run the optimizer on a managed Omnigent server.

    Builds the backend from the ``optimizer:`` config section, collects
    the scaffold tree into a flat ``relative_path -> content`` dict, and
    delegates to :func:`get_backend`. Returns a 5-tuple
    ``(action, transcript, parse_result, optimizer_error, session_url)`` —
    the same three values as :func:`run_optimizer_session`, plus the
    backend-failure marker (``None`` on success) so the round can
    distinguish a swallowed backend error from a legitimate
    optimizer-chosen noop, plus the managed conversation ``session_url``
    (a durable pointer to the optimizer transcript; ``None`` on failure).
    """
    scaffold_files = _collect_scaffold_files(repo_root)
    bundle_rel = optimizer_cfg.get("agent_bundle_path", "agents/forge_optimizer.yaml")
    bundle_path = Path(bundle_rel)
    if not bundle_path.is_absolute():
        bundle_path = (repo_root / bundle_rel).resolve()
    # Fallback: if the bundle doesn't exist in the cloned repo, use the
    # forge package's own copy so a target repo without an agents/
    # directory still works.
    if not bundle_path.exists():
        forge_agents = Path(__file__).resolve().parent.parent.parent.parent / "agents"
        forge_bundle = forge_agents / "forge_optimizer.yaml"
        if forge_bundle.exists():
            bundle_path = forge_bundle
    cfg = BackendConfig(
        backend="omnigent",
        # Env-authoritative URL resolved by the caller (see
        # ``_resolve_omnigent_backend_url``): OMNIGENT_SERVER_URL env >
        # derived {DATABRICKS_HOST}/omnigent > file server_url > localhost.
        server_url=server_url,
        auth_token=optimizer_cfg.get("auth_token") or os.getenv("OMNIGENT_AUTH_TOKEN") or None,
        agent_bundle_path=str(bundle_path),
        model=optimizer_endpoint,
        max_turns=max_turns,
    )
    backend = get_backend(cfg)
    try:
        result = await backend.run(
            prompt=prompt,
            scaffold_files=scaffold_files,
            max_turns=max_turns,
            model=optimizer_endpoint,
        )
    finally:
        # Close the factory-created OmnigentClient so its underlying
        # ``httpx.AsyncClient`` connection pool does not leak. The local
        # backend has no ``client`` attribute; the isinstance guard keeps
        # the cleanup omnigent-specific without widening the Protocol.
        if isinstance(backend, OmnigentBackend):
            await backend.client.aclose()
    return (
        result.action,
        result.transcript,
        result.parse_result,
        result.optimizer_error,
        result.session_url,
    )


def _collect_scaffold_files(repo_root: Path) -> dict[str, str]:
    """Collect the scaffold + config + prompts + baseline tree for upload.

    Walks the directories the optimizer agent reads from
    (``scaffold/``, ``prompts/``, ``harness/``, ``agents/``) and returns a
    flat ``{relative_path: content}`` dict of UTF-8 text files. The
    cached eval baseline + frontier JSONs are included so the agent can
    diagnose the score to beat. Binary and unreadable files are skipped.
    """
    files: dict[str, str] = {}
    for sub in ("scaffold", "prompts", "harness", "agents"):
        root = repo_root / sub
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            files[path.relative_to(repo_root).as_posix()] = content
    for name in ("baseline.json", "frontier.json"):
        p = repo_root / "eval" / "runs" / name
        if not p.is_file():
            continue
        with contextlib.suppress(UnicodeDecodeError, OSError):
            files[p.relative_to(repo_root).as_posix()] = p.read_text(encoding="utf-8")
    return files


def _build_critique_md(
    *,
    round_id: int,
    branch: str,
    decision: Decision,
    action_kind: str,
    apply_summary: str,
    rationale: str,
    baseline_score: float | None,
    mutated_score: float | None,
    score_delta: float | None,
    parse_status: str,
) -> str:
    bs = f"{baseline_score:.4f}" if baseline_score is not None else "null"
    ms = f"{mutated_score:.4f}" if mutated_score is not None else "null"
    sd = f"{score_delta:+.4f}" if score_delta is not None else "null"
    return f"""---
round: {round_id}
branch: {branch}
decision: {decision}
action_kind: {action_kind}
parse_status: {parse_status}
baseline_score: {bs}
mutated_score: {ms}
score_delta: {sd}
---

# Round {round_id} critique

## Action applied
{apply_summary or "(no change)"}

## Rationale (from optimizer)
{rationale}

## Outcome
Decision: **{decision.value.upper()}**. Score delta vs cached baseline:
{sd}.
"""


def _build_round_json(
    *,
    round_id: int,
    branch: str,
    commit_sha: str,
    parent_sha: str,
    decision: Decision,
    action_kind: str,
    eval_report,
    baseline_score: float | None,
    score_delta: float | None,
    parse_status: str,
    notes: str,
    frontier_best: dict[str, float] | None = None,
    optimizer_error: str | None = None,
    optimizer_backend: str | None = None,
    optimizer_server_url: str | None = None,
    rationale: str | None = None,
    optimizer_run_id: str | None = None,
    optimizer_experiment_id: str | None = None,
    optimizer_session_url: str | None = None,
    optimizer_trace_id: str | None = None,
    levers: dict[str, Any] | None = None,
) -> dict:
    payload: dict = {
        "round_id": round_id,
        "branch": branch,
        "scaffold_commit_sha": commit_sha,
        "parent_commit_sha": parent_sha,
        "decision": decision.value,
        "action_kind": action_kind,
        "parse_status": parse_status,
        # Resolved lever values the mutated eval ran with (e.g. the model).
        # Empty when no levers are declared. Lets per-model latency/quality
        # history be read straight from the round records.
        "levers": levers or {},
        "baseline_score": baseline_score,
        "score_delta_vs_parent": score_delta,
        "notes": notes,
        # Non-None ONLY on an optimizer-backend failure (auth/redirect/
        # connection error). Lets the orchestrator + UI distinguish a
        # swallowed backend error from a legitimate optimizer-chosen noop.
        "optimizer_error": optimizer_error,
        # The optimizer backend that ACTUALLY ran this round
        # ('local' | 'omnigent'), after env-var precedence is applied. Lets
        # a 'local noop' be told apart from an 'omnigent noop/INFRA_FAIL'
        # from the round record alone (no transcript needed). When
        # 'omnigent', ``optimizer_server_url`` carries the resolved URL the
        # backend was pointed at; None otherwise.
        "optimizer_backend": optimizer_backend,
        "optimizer_server_url": optimizer_server_url,
        # Optimizer's own rationale for the action it chose this round
        # (``OptimizerAction.rationale``). Carried so the run-level
        # improvement summary can show the per-round diagnosis without
        # re-reading the critique md. May be empty.
        "rationale": rationale,
        # Durable sink pointers: the SESSION-level parent MLflow run that
        # holds this round's persisted transcript + critique (distinct from
        # the per-round eval run under ``mlflow`` below), and — on the
        # omnigent backend — the managed conversation URL. All None when
        # persistence is off / local backend. Surfaced alongside the eval
        # ``mlflow`` pointer so the stored logs/summary are findable.
        "optimizer_run": (
            {"run_id": optimizer_run_id, "experiment_id": optimizer_experiment_id}
            if optimizer_run_id
            else None
        ),
        "optimizer_session_url": optimizer_session_url,
        # MLflow trace of the full optimizer session (local backend; None
        # when persistence is off or the omnigent backend ran the round).
        "optimizer_trace_id": optimizer_trace_id,
        # Best-so-far per objective after this round's decision (frontier
        # gate only; None for the legacy delta gate / noop / infra-fail).
        # The decision is driven by this, not by ``score_delta_vs_parent``.
        "frontier_best": frontier_best,
    }
    if eval_report is not None:
        payload.update(
            {
                "evaluated_at": eval_report.evaluated_at,
                "n_examples": eval_report.n_rows,
                "mode": eval_report.mode,
                "scorers": list(eval_report.scorers),
                "aggregate": eval_report.aggregate,
                "per_judge": dict(eval_report.per_judge),
                "per_bucket": {k: dict(v) for k, v in eval_report.per_bucket.items()},
                "cost_metrics": dict(getattr(eval_report, "cost_metrics", {})),
                "failures": list(eval_report.failures),
                "trace_ids": list(eval_report.trace_ids),
                "mlflow": {
                    "run_id": eval_report.run_id,
                    "experiment_id": eval_report.experiment_id,
                },
            }
        )
    return payload
