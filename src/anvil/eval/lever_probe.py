"""Verified runtime settings: which model x setting combinations really work.

An agent's runtime knobs are only useful if the serving endpoints accept
them, and the accepted values differ per model (GPT-5.6 Luna rejects
``reasoning_effort="minimal"``; GLM and DeepSeek reject native-PDF input).
A wrong list in the round prompt sends the optimizer after settings that
fail every row. So before a run, ``scripts/probe_levers.py`` makes one tiny
real call per combination through the engine's registered probe
(:func:`anvil.eval.engines.register_lever_probe`) and records the outcome
in ``eval/lever_probe.json``:

* the round prompt shows the accepted and rejected combinations
  (:func:`render_probe_md`), so the optimizer only picks verified ones;
* an engine can refuse a rejected combination up front
  (:func:`setting_rejection`) instead of spending an eval on it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROBE_REL = Path("eval") / "lever_probe.json"


@dataclass
class ProbeResult:
    """One probed combination: a model plus engine settings, accepted or not."""

    model: str
    settings: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    error: str | None = None


def write_lever_probe(repo_root: Path | str, results: list[ProbeResult], *, engine: str) -> Path:
    path = Path(repo_root) / PROBE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine": engine,
        "probed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_lever_probe(repo_root: Path | str) -> dict[str, Any] | None:
    """The recorded probe, or ``None`` when none was run (or it is unreadable)."""
    path = Path(repo_root) / PROBE_REL
    if not path.is_file():
        return None
    try:
        probe = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return probe if isinstance(probe, dict) and isinstance(probe.get("results"), list) else None


def _matches(result: dict[str, Any], model: str, settings: dict[str, Any]) -> bool:
    return result.get("model") == model and all(
        result.get("settings", {}).get(k) == v for k, v in settings.items()
    )


def setting_rejection(probe: dict[str, Any] | None, model: str, **settings: Any) -> str | None:
    """Why ``model`` + ``settings`` is known NOT to work, or ``None``.

    A combination is rejected only when the probe tried exactly these
    values (a superset of keys is fine) and every such attempt failed.
    Combinations the probe never tried return ``None`` (unknown, allowed).
    """
    if not probe:
        return None
    tried = [r for r in probe["results"] if _matches(r, model, settings)]
    if not tried or any(r.get("ok") for r in tried):
        return None
    return str(tried[0].get("error") or "rejected by the endpoint")


def short_error(error: str | None) -> str:
    """The endpoint's own message from a wrapped error, e.g. "Unsupported value: ...".

    Transport errors arrive wrapped ("ExtractionError: ... after 2 attempts:
    HTTP 400: {"error_code": ..., "message": "{\"error\": {...}}"}"). Unwrap
    the nested JSON to the innermost message; fall back to the raw text.
    """
    text = str(error or "")
    m = re.search(r"HTTP (\d{3}):\s*(\{.*)", text, re.DOTALL)
    if not m:
        return text
    payload: Any = m.group(2)
    for _ in range(4):  # unwrap up to a few nested JSON layers
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                break
        if isinstance(payload, dict):
            inner = payload.get("error", payload.get("message"))
            if isinstance(inner, dict):
                inner = inner.get("message", inner)
            if inner is None:
                break
            payload = inner
        else:
            break
    if not isinstance(payload, str) or payload.lstrip().startswith("{"):
        # Truncated or odd JSON: drop the escaping, take the innermost "message".
        raw = str(payload if isinstance(payload, str) else m.group(2))
        flat = re.sub(r"\\+", "", re.sub(r"\\+n", " ", raw))
        found = re.findall(r'"message"\s*:\s*"([^"]+)', flat)
        payload = found[-1] if found else flat
    return f"HTTP {m.group(1)}: {str(payload).strip()}"


def render_probe_md(probe: dict[str, Any] | None, *, max_error_chars: int = 220) -> str:
    """Round-prompt section: accepted / rejected combinations per model.

    Combinations that differ in only one setting and share an outcome are
    collapsed into one line (``reasoning_effort ∈ {low, medium}``).
    """
    if not probe:
        return ""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in probe["results"]:
        by_model.setdefault(str(r.get("model")), []).append(r)

    lines = [
        "## Verified runtime settings",
        "",
        f"Probed {probe.get('probed_at', '?')} with one real call per combination. "
        "Use only accepted combinations — a rejected one fails every row.",
        "",
    ]
    for model, results in sorted(by_model.items()):
        lines.append(f"- `{model}`")
        for verdict, group, err in _grouped(results):
            err_s = f" — {short_error(err)[:max_error_chars]}" if err else ""
            lines.append(f"  - {verdict}: {group}{err_s}".replace("\n", " "))
    return "\n".join(lines) + "\n"


def _grouped(results: list[dict[str, Any]]) -> list[tuple[str, str, str | None]]:
    """(verdict, settings text, error) lines, one setting collapsed per line."""
    keys = sorted({k for r in results for k in r.get("settings", {})})
    # Collapse the setting with the most distinct values (e.g. reasoning_effort);
    # on a tie, the last one (finer-grained knobs sort after input_mode).
    vary = max(
        keys,
        key=lambda k: (len({str(r.get("settings", {}).get(k)) for r in results}), keys.index(k)),
        default=None,
    )
    order = {str(r.get("settings", {}).get(vary)): i for i, r in enumerate(results)} if vary else {}
    groups: dict[tuple, list[str]] = {}
    for r in results:
        st = r.get("settings", {})
        fixed = tuple((k, str(st.get(k))) for k in keys if k != vary)
        err = None if r.get("ok") else short_error(r.get("error"))
        groups.setdefault((bool(r.get("ok")), fixed, err), []).append(str(st.get(vary)))
    out: list[tuple[str, str, str | None]] = []
    for (ok, fixed, err), values in sorted(groups.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
        parts = [f"{k}={v}" for k, v in fixed]
        if vary is not None:
            vals = sorted(set(values), key=lambda v: order.get(v, 0))
            parts.append(
                f"{vary}={vals[0]}" if len(vals) == 1 else f"{vary} ∈ {{{', '.join(vals)}}}"
            )
        out.append(("accepted" if ok else "rejected", ", ".join(parts) or "(defaults)", err))
    return out
