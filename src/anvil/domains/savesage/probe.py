"""Lever probe for the savesage engine: which model x input mode x effort work.

One tiny real extraction per combination — page 1 of the first golden
statement, the composed prompt, the production schema, a 16-token output
cap — through the same transport the eval uses. The endpoint either takes
the settings (the call then usually stops at the token cap, which counts as
accepted) or rejects them with a 4xx naming the bad parameter
("'reasoning_effort' does not support 'minimal'", "unsupported content item
type: file"). Transport failures (timeouts, 5xx) leave the combination
unknown rather than rejected.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from anvil.domains.savesage.extractor import INPUT_MODES, SavesageIciciExtractor
from anvil.eval.lever_probe import ProbeResult, short_error
from anvil.runtime.call_ledger import describe_error, is_infra_error
from anvil.runtime.composer import compose_prompt
from anvil.runtime.loader import load_harness

# Every reasoning_effort value any of the supported endpoints has been seen to
# name; the probe finds out which each model takes.
EFFORT_CANDIDATES = ("none", "minimal", "low", "medium", "high", "xhigh")
_PROBE_MAX_TOKENS = 16
# The endpoint accepted the request; only the 16-token output was cut short.
_ACCEPTED_MARKERS = ("finish_reason='length'", "not valid JSON", "empty content")


def _classify(error: str) -> bool | None:
    """True = accepted, False = rejected, None = unknown (transport failure)."""
    if any(m in error for m in _ACCEPTED_MARKERS):
        return True
    if is_infra_error(error):
        return None
    if "HTTP 4" in error:
        return False
    return None


def _first_page(pdf_path: Path, workdir: Path) -> Path:
    exe = shutil.which("pdfseparate")
    if exe is None:
        return pdf_path
    out = workdir / pdf_path.name
    proc = subprocess.run(
        [exe, "-f", "1", "-l", "1", str(pdf_path), str(out)],
        capture_output=True,
        check=False,
        timeout=30,
    )
    return out if proc.returncode == 0 and out.is_file() else pdf_path


def probe_savesage(
    *,
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
    golden_set_path: Path | str = "data/golden_set.jsonl",
    profile: str | None = None,  # noqa: ARG001 - endpoints auth via the Luna profile chain
    efforts: tuple[str, ...] = EFFORT_CANDIDATES,
    input_modes: tuple[str, ...] = INPUT_MODES,
    max_workers: int = 6,
    **_kwargs: Any,
) -> list[ProbeResult]:
    """Probe every allowed model x input mode x reasoning_effort combination."""
    from anvil.domains.savesage.eval import (  # noqa: PLC0415
        _GENERIC_GOLDEN_SET,
        GOLDEN_SET_REL,
        load_savesage_golden_set,
    )

    if not os.environ.get("SAVESAGE_STATEMENT_AGENT_PATH"):
        sa = Path(scaffold_root) / "statement-agent"
        if sa.is_dir():
            os.environ["SAVESAGE_STATEMENT_AGENT_PATH"] = str(sa)
    scaffold_path = Path(scaffold_root)
    repo_root = scaffold_path.parent
    cfg = load_harness(scaffold_path, runtime_config_path).config
    spec = cfg.lever_specs.get("model")
    models = [str(m) for m in spec.allowed] if spec else [cfg.effective_runtime_model]
    gs = Path(golden_set_path)
    if gs.name == Path(_GENERIC_GOLDEN_SET).name:
        gs = repo_root / GOLDEN_SET_REL
    row = load_savesage_golden_set(gs)[0]
    composed = compose_prompt(scaffold_path, audience="runtime").text

    combos = [(m, i, e) for m in models for i in input_modes for e in efforts]
    with tempfile.TemporaryDirectory(prefix="ss-probe-") as tmp:
        page1 = _first_page(Path(row["pdf_path"]), Path(tmp))

        def _one(combo: tuple[str, str, str]) -> ProbeResult:
            model, input_mode, effort = combo
            settings = {"input_mode": input_mode, "reasoning_effort": effort}
            try:
                SavesageIciciExtractor(
                    composed,
                    cache_root=None,
                    reasoning_effort=effort,
                    max_tokens=_PROBE_MAX_TOKENS,
                    model=model,
                    input_mode=input_mode,
                ).extract_with_usage(sid="lever-probe", pdf_path=page1)
            except Exception as exc:  # noqa: BLE001 - the outcome IS the result
                error = describe_error(exc)
                ok = _classify(error)
                return ProbeResult(
                    model=model,
                    settings=settings,
                    ok=bool(ok),
                    error=None
                    if ok
                    else (short_error(error) if ok is False else f"unknown: {error}")[:400],
                )
            return ProbeResult(model=model, settings=settings, ok=True)

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            results = list(ex.map(_one, combos))
    # Unknown (transport) outcomes are not evidence either way: drop them.
    return [r for r in results if r.ok or not (r.error or "").startswith("unknown:")]
