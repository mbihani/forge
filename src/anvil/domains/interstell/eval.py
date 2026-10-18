"""Interstell eval engine: run every sample turn through the deployed App, score with Interstell's judges.

One eval = the 36 client samples x ``ceil(rows / 36)`` repeats (``eval.modes.<mode>.rows``
counts TURNS, so ``rows: 108`` is 36 x 3). Each turn is one ``POST /api/turn`` on the
deployed Databricks App with the round's levers (``model``, ``route``, ``temperature``,
``max_completion_tokens``; ``reasoning_effort`` fixed at ``none``), so latency is measured
inside the workspace exactly as ``evals.run_eval --target app`` measures it. The prompt
(the client's ``system_prompt.txt``) and tools are fixed: this domain optimizes levers only.

Scoring re-uses Interstell's own scorers and labels unchanged (see ``_interstell``):
``action_correct``, ``transfer_gate_safety`` (+ Opus args judge), ``speak_and_act`` and
``reply_quality`` (Opus judge), aggregated by ``evals.summary.compute`` with its
provisional-row exclusions. The report's:

* ``aggregate`` = weighted mean of action accuracy (x2), speak-and-act (x1) and reply
  quality (x1);
* ``cost_metrics.latency_ms_median`` = first-action p50 (Interstell's target metric);
* ``cost_metrics.premature_transfers`` = transfers the TRANSFER GATE does not allow (the
  safety veto objective).
"""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anvil.catalog import load_catalog, model_prices
from anvil.domains.interstell import _interstell
from anvil.eval.lever_probe import load_lever_probe, setting_rejection
from anvil.eval.runner import EvalReport
from anvil.observability import eval_row_trace
from anvil.runtime.call_ledger import is_infra_error, price_calls
from anvil.runtime.loader import load_harness

logger = logging.getLogger(__name__)

# Mirrors evals/run_eval.py EXPECTATION_KEYS (run_eval itself imports MLflow 3.17-only tracing).
EXPECTATION_KEYS = (
    "expected_action",
    "transfer_allowed",
    "gate",
    "expected_line",
    "acceptable_lines",
    "stated_debt",
    "call_state",
    "prod_action",
    "confidence",
)
OUTPUT_KEYS = (
    "action",
    "text",
    "tool_calls",
    "first_word_ms",
    "first_tool_ms",
    "first_action_ms",
    "total_ms",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "cost_usd",
    "model",
    "route",
    "error",
    "trace_id",
)
# Quality aggregate weights over evals.summary.compute metric names.
QUALITY_WEIGHTS = {
    "action_accuracy": 2.0,
    "speak_and_act_rate": 1.0,
    "reply_quality_pass_rate": 1.0,
}
PER_JUDGE_NAMES = {
    "action_accuracy": "action_correct",
    "speak_and_act_rate": "speak_and_act",
    "reply_quality_pass_rate": "reply_quality",
}
# Models whose App profile already sets reasoning_effort=minimal (they reject "none").
PROFILE_REASONING_PREFIXES = ("databricks-gemini-",)
REASONING_MODES = ("none", "omit", "profile")
DEFAULT_LEVERS = {"route": "gateway", "temperature": 0.3, "max_completion_tokens": 1024}
APP_TIMEOUT_S = 120.0
APP_ATTEMPTS = 3


# Transport failures in App/openai wording that the core pattern does not name.
_APP_INFRA = re.compile(
    r"Error code: (5\d\d|429)|ConnectError|ConnectTimeout|ReadTimeout|RemoteProtocolError|"
    r"APIConnectionError|APITimeoutError"
)


def is_transport_error(error: str | None) -> bool:
    """True when a turn failed on transport (App or model endpoint), not on its settings."""
    return bool(error) and (is_infra_error(error) or bool(_APP_INFRA.search(error)))


def reasoning_mode(model: str, probe: dict | None) -> str | None:
    """How ``reasoning_effort=none`` is expressed for ``model``.

    ``none`` sends ``reasoning_effort: "none"``; ``omit`` sends no key (a model with no
    reasoning setting that the probe saw produce zero reasoning tokens); ``profile`` leaves
    the App's per-model default (Gemini: ``minimal``, the baseline incumbent). ``None`` =
    the probe found no way to run this model without reasoning.
    """
    if probe:
        tried = [
            r
            for r in probe.get("results", [])
            if r.get("model") == model and "reasoning" in (r.get("settings") or {})
        ]
        accepted = {r["settings"]["reasoning"] for r in tried if r.get("ok")}
        for mode in REASONING_MODES:
            if mode in accepted:
                return mode
        if tried:
            return None
    return "profile" if model.startswith(PROFILE_REASONING_PREFIXES) else "none"


def request_extra(levers: dict[str, Any], reasoning: str) -> dict[str, Any]:
    """The ``/api/turn`` ``extra`` overrides for this round (None drops a key in the App)."""
    extra: dict[str, Any] = {
        "temperature": float(levers.get("temperature", DEFAULT_LEVERS["temperature"])),
        "max_completion_tokens": int(
            levers.get("max_completion_tokens", DEFAULT_LEVERS["max_completion_tokens"])
        ),
    }
    if reasoning == "none":
        extra["reasoning_effort"] = "none"
    elif reasoning == "omit":
        extra["reasoning_effort"] = None
    return extra


def call_app(
    messages: list[dict], sample_id: str, *, model: str, route: str, extra: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    """One agent turn on the deployed App. Returns (outputs, transport retries used).

    Transport failures (timeouts, 5xx, 429) are retried; a turn the model rejected comes
    back as ``outputs["error"]`` from the App and is not retried.
    """
    import httpx  # noqa: PLC0415

    body = {
        "messages": messages,
        "sample_id": sample_id,
        "model": model,
        "route": route,
        "extra": extra,
    }
    url = _interstell.app_url() + "/api/turn"
    error = "no attempt"
    for attempt in range(APP_ATTEMPTS):
        try:
            r = httpx.post(url, json=body, headers=_interstell.app_headers(), timeout=APP_TIMEOUT_S)
            if r.status_code >= 500 or r.status_code == 429:
                error = f"App HTTP {r.status_code}: {r.text[:300]}"
            else:
                r.raise_for_status()
                res = r.json()
                return {k: res.get(k) for k in OUTPUT_KEYS}, attempt
        except httpx.HTTPStatusError as exc:
            return {
                "error": f"App HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            }, attempt
        except Exception as exc:  # noqa: BLE001 - transport failure, retried
            error = f"App call failed: {type(exc).__name__}: {exc}"[:400]
        time.sleep(2 * (attempt + 1))
    return {"error": f"{error} (timeout: after {APP_ATTEMPTS} attempts)"}, APP_ATTEMPTS - 1


def _turns(samples: list[dict], rows: int, max_rows: int | None) -> list[tuple[dict, int]]:
    repeats = max(1, math.ceil(rows / len(samples))) if rows else 1
    turns = [(s, rnd) for rnd in range(1, repeats + 1) for s in samples]
    return turns[:max_rows] if max_rows else turns


def _score(messages, outputs, expectations, judge: str) -> tuple[dict, dict]:
    """Interstell's four scorers on one turn, exactly as evals/run_eval.py wires them."""
    from evals import judges, scorers  # noqa: PLC0415

    scores: dict[str, Any] = {}
    why: dict[str, str] = {}
    if outputs.get("error"):
        return scores, {"error": outputs["error"]}
    scores["action_correct"], why["action_correct"] = scorers.action_correct(outputs, expectations)
    v, w, status = judges.score_transfer_gate(
        messages,
        outputs,
        expectations,
        scorers.transfer_gate_safety,
        lambda msgs, args: judges.transfer_args_faithful(msgs, args, judge),
    )
    scores["transfer_gate_safety"], why["transfer_gate_safety"] = v, w
    scores["transfer_judge_status"] = status
    scores["speak_and_act"], why["speak_and_act"] = scorers.speak_and_act(outputs, expectations)
    scores["speak_and_act_mode"] = scorers.speak_and_act_mode(outputs, expectations)
    v, w, err = judges.score_reply_quality(
        messages,
        outputs,
        expectations,
        lambda msgs, text, exp: judges.reply_quality(msgs, text, exp, judge),
    )
    scores["reply_quality"], why["reply_quality"] = v, w
    scores["reply_quality_eval_error"] = err
    return scores, why


def quality_aggregate(m: dict[str, Any]) -> float:
    """Weighted mean over the quality metrics present (None = no applicable rows)."""
    parts = [(w, m.get(k)) for k, w in QUALITY_WEIGHTS.items() if m.get(k) is not None]
    total = sum(w for w, _ in parts)
    return sum(w * float(v) for w, v in parts) / total if total else 0.0


def _per_judge(m: dict[str, Any]) -> dict[str, float]:
    out = {name: float(m[k]) for k, name in PER_JUDGE_NAMES.items() if m.get(k) is not None}
    if m.get("n_scored"):
        out["transfer_gate_safety"] = 1.0 - m["gate_violations"] / m["n_scored"]
    return out


def _failure(rec: dict) -> dict | None:
    sc = rec["scores"]
    failed = sorted(
        k
        for k in ("action_correct", "transfer_gate_safety", "speak_and_act", "reply_quality")
        if sc.get(k) is False
    )
    err = rec["outputs"].get("error")
    if not failed and not err:
        return None
    out = rec["outputs"]
    return {
        "example_id": f"{rec['sample_id']}#r{rec['round']}",
        "query": rec["sample_id"],
        "category": rec["category"],
        "judge_failures": failed,
        "expected_action": rec["expected_action"],
        "agent_action": out.get("action"),
        "spoken": (out.get("text") or "")[:200],
        "first_action_ms": out.get("first_action_ms"),
        "rationales": {k: str(v)[:240] for k, v in rec["rationales"].items() if k in failed},
        **({"provisional_label": True} if rec.get("provisional") else {}),
        **({"error": str(err)[:500]} if err else {}),
        **({"infra": True} if rec.get("infra_failed") else {}),
    }


def evaluate_interstell(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    mode: str | None = None,
    trace_rows: bool = False,
    max_rows: int | None = None,
    **_kwargs: Any,  # absorb genai-only kwargs (golden_set_path, profile, ...) per the engine contract
) -> EvalReport:
    """Evaluate the round's levers over the Interstell samples and return an ``EvalReport``."""
    _interstell.ensure_importable()
    from agent import assets  # noqa: PLC0415
    from evals import summary  # noqa: PLC0415
    from evals.label import load_labels  # noqa: PLC0415

    scaffold_path = Path(scaffold_root)
    repo_root = scaffold_path.parent
    snapshot = load_harness(scaffold_path, runtime_config_path)
    cfg = snapshot.config.eval
    selected_mode = mode or cfg.default_mode
    if selected_mode not in cfg.modes:
        raise ValueError(f"mode {selected_mode!r} not in harness/config.yaml > eval.modes")
    levers = {**DEFAULT_LEVERS, **snapshot.config.levers}
    model = snapshot.config.effective_runtime_model
    route = str(levers["route"])
    judge = snapshot.config.judge_endpoint
    probe = load_lever_probe(repo_root)
    reasoning = reasoning_mode(model, probe)

    rejection = None
    if reasoning is None:
        rejection = f"{model} rejects reasoning_effort=none and cannot run without reasoning"
    else:
        for key in ("route", "temperature", "max_completion_tokens"):
            why = setting_rejection(probe, model, **{key: levers[key]})
            if why:
                rejection = f"{model} rejects {key}={levers[key]!r}: {why}"
                break
    extra = request_extra(levers, reasoning or "none")

    labels = load_labels()
    turns = _turns(assets.samples(), cfg.modes[selected_mode].rows, max_rows)

    def _run(turn: tuple[dict, int]) -> dict[str, Any]:
        sample, rnd = turn
        sid = sample["id"]
        lab = labels[sid]
        expectations = {**{k: lab.get(k) for k in EXPECTATION_KEYS}, "category": sample["category"]}
        row_cm = (
            eval_row_trace(
                example_id=f"{sid}#r{rnd}",
                query=sid,
                scaffold_root=scaffold_path,
                runtime_endpoint=model,
            )
            if trace_rows
            else nullcontext()
        )
        with row_cm as span:
            retries = 0
            if rejection:
                outputs: dict[str, Any] = {"error": f"setting rejected by lever probe: {rejection}"}
            else:
                outputs, retries = call_app(
                    sample["messages"], sid, model=model, route=route, extra=extra
                )
            scores, why = _score(sample["messages"], outputs, expectations, judge)
            if span is not None:
                with suppress(Exception):
                    span.set_outputs({"outputs": outputs, "scores": scores})
        return {
            "sample_id": sid,
            "round": rnd,
            "category": sample["category"],
            "expected_action": expectations["expected_action"],
            "prod_action": expectations["prod_action"],
            "transfer_allowed": expectations["transfer_allowed"],
            "outputs": outputs,
            "scores": scores,
            "rationales": why,
            "provisional": lab.get("judges_agree") is False,
            "closing_provisional": bool(lab.get("closing_provisional")),
            "infra_retries": retries,
            "infra_failed": is_transport_error(outputs.get("error")),
        }

    with ThreadPoolExecutor(max_workers=max(1, cfg.n_workers)) as ex:
        records = list(ex.map(_run, turns))

    m = summary.compute(records, judges_enabled=True)
    per_bucket: dict[str, dict[str, float]] = {}
    for cat in sorted({r["category"] for r in records}):
        mc = summary.compute([r for r in records if r["category"] == cat], judges_enabled=True)
        per_bucket[cat] = {**_per_judge(mc), "aggregate": quality_aggregate(mc)}

    prices = model_prices(
        [model],
        load_catalog(repo_root / snapshot.config.model_catalog.path),
        tier=snapshot.config.model_catalog.context_tier,
    )
    ok = [r for r in records if not r["outputs"].get("error")]
    call_costs = price_calls(
        [
            {
                "model": model,
                "input_tokens": r["outputs"].get("prompt_tokens") or 0,
                "output_tokens": r["outputs"].get("completion_tokens") or 0,
            }
            for r in ok
        ],
        prices,
    )
    cost_metrics: dict[str, float] = {
        "n_rows": float(len(records)),
        "n_row_errors": float(len(records) - len(ok)),
        "n_infra_retries": float(sum(r["infra_retries"] for r in records)),
        "n_infra_failed": float(sum(1 for r in records if r["infra_failed"])),
        "premature_transfers": float(m["premature_transfers"]),
        "gate_violations": float(m["gate_violations"]),
        "transfers_made": float(m["transfers_made"]),
        "reasoning_tokens_total": float(m["reasoning_tokens_total"]),
    }
    for key, name in (
        ("first_action_p50_ms", "latency_ms_median"),
        ("first_action_p90_ms", "latency_ms_p90"),
        ("first_action_max_ms", "latency_ms_max"),
        ("total_p50_ms", "total_ms_p50"),
        ("total_p90_ms", "total_ms_p90"),
        ("action_vs_prod", "action_vs_prod"),
        ("prompt_tokens_mean", "prompt_tokens_mean"),
        ("completion_tokens_mean", "completion_tokens_mean"),
    ):
        if m.get(key) is not None:
            cost_metrics[name] = float(m[key])
    if call_costs and all(c is not None for c in call_costs):
        cost_metrics["cost_usd_total"] = float(sum(call_costs))
        cost_metrics["cost_usd_per_row"] = cost_metrics["cost_usd_total"] / len(call_costs)

    experiment_id = ""
    if trace_rows:
        try:
            import mlflow  # noqa: PLC0415

            exp = mlflow.get_experiment_by_name(snapshot.config.experiments.eval)
            experiment_id = exp.experiment_id if exp is not None else ""
        except Exception as exc:  # noqa: BLE001 - best-effort observability
            logger.warning("could not resolve eval experiment id: %s", exc)

    per_judge = _per_judge(m)
    return EvalReport(
        aggregate=quality_aggregate(m),
        per_judge=per_judge,
        per_bucket=per_bucket,
        failures=[f for f in (_failure(r) for r in records) if f],
        run_id="",
        experiment_id=experiment_id,
        n_rows=len(records),
        mode=selected_mode,
        scorers=sorted(per_judge),
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        trace_ids=[],
        cost_metrics=cost_metrics,
        scorer_fingerprint="",
    )
