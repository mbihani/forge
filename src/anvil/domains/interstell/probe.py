"""Lever probe for the interstell engine: which model x reasoning x temperature x route work.

One real turn per combination — sample S17 (a transfer-gate turn with the full 11.5K-token
system prompt and both tools) through the deployed App's ``/api/turn``, the same path the
eval uses. ``reasoning`` is probed first per model (``none``, then ``omit`` for a model with
no reasoning setting, accepted only if it produced zero reasoning tokens; Gemini, the
baseline incumbent, runs its App profile default ``minimal``). Temperature and route are
then probed with the model's working reasoning mode. Transport failures leave a
combination unknown rather than rejected.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from anvil.domains.interstell import _interstell
from anvil.domains.interstell.eval import (
    DEFAULT_LEVERS,
    PROFILE_REASONING_PREFIXES,
    TOKEN_PARAMS,
    call_app,
    is_transport_error,
    request_extra,
)
from anvil.eval.lever_probe import ProbeResult
from anvil.runtime.loader import load_harness

PROBE_SAMPLE = "S17"
_PROBE_MAX_TOKENS = 64


def _classify(outputs: dict[str, Any], reasoning: str) -> tuple[bool | None, str | None]:
    """(ok, error): ok None = unknown (transport failure)."""
    err = outputs.get("error")
    if err:
        if is_transport_error(err):
            return None, f"unknown: {err}"
        return False, str(err)[:400]
    if reasoning == "omit" and (outputs.get("reasoning_tokens") or 0) > 0:
        return (
            False,
            f"reasons by default ({outputs['reasoning_tokens']} reasoning tokens) with no setting",
        )
    return True, None


def _allowed(cfg, name: str) -> list[Any]:
    spec = cfg.lever_specs.get(name)
    return list(spec.allowed) if spec else [DEFAULT_LEVERS[name]]


def probe_interstell(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    max_workers: int = 6,
    **_kwargs: Any,
) -> list[ProbeResult]:
    _interstell.ensure_importable()
    from agent import assets  # noqa: PLC0415

    cfg = load_harness(Path(scaffold_root), runtime_config_path).config
    spec = cfg.lever_specs.get("model")
    models = [str(m) for m in spec.allowed] if spec else [cfg.effective_runtime_model]
    messages = assets.sample(PROBE_SAMPLE)["messages"]

    def _one(
        model: str, settings: dict[str, Any], reasoning: str, cap_field: str = TOKEN_PARAMS[0]
    ) -> ProbeResult:
        levers = {**DEFAULT_LEVERS, "max_completion_tokens": _PROBE_MAX_TOKENS}
        levers.update({k: v for k, v in settings.items() if k in DEFAULT_LEVERS})
        outputs, _ = call_app(
            messages,
            PROBE_SAMPLE,
            model=model,
            route=str(levers["route"]),
            extra=request_extra(levers, reasoning, cap_field),
        )
        ok, error = _classify(outputs, reasoning)
        return ProbeResult(model=model, settings=settings, ok=bool(ok), error=error)

    def _reasoning_with(model: str, mode: str) -> list[ProbeResult]:
        # Production sends max_completion_tokens; retry with max_tokens when the
        # endpoint rejects the field itself, recording which one worked.
        first = _one(model, {"reasoning": mode}, mode)
        if first.ok or 'unknown field "max_completion_tokens"' not in (first.error or ""):
            return [first]
        return [
            first,
            _one(model, {"reasoning": mode, "token_param": "max_tokens"}, mode, "max_tokens"),
        ]

    def _reasoning(model: str) -> list[ProbeResult]:
        if model.startswith(PROFILE_REASONING_PREFIXES):
            return _reasoning_with(model, "profile")
        first = _reasoning_with(model, "none")
        if first[-1].ok or (first[-1].error or "").startswith("unknown:"):
            return first
        return first + _reasoning_with(model, "omit")

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        per_model = dict(zip(models, ex.map(_reasoning, models), strict=True))
        combos = []
        for model, results in per_model.items():
            working = next((r.settings for r in results if r.ok), None)
            if working is None:
                continue
            mode = working["reasoning"]
            cap_field = working.get("token_param", TOKEN_PARAMS[0])
            combos += [
                (model, {"temperature": t}, mode, cap_field) for t in _allowed(cfg, "temperature")
            ]
            combos += [(model, {"route": r}, mode, cap_field) for r in _allowed(cfg, "route")]
        rest = list(ex.map(lambda c: _one(*c), combos))
    results = [r for rs in per_model.values() for r in rs] + rest
    # Unknown (transport) outcomes are not evidence either way: drop them.
    return [r for r in results if r.ok or not (r.error or "").startswith("unknown:")]
