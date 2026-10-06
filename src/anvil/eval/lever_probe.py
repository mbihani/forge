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


def render_probe_md(probe: dict[str, Any] | None, *, max_error_chars: int = 160) -> str:
    """Round-prompt section: accepted / rejected combinations per model."""
    if not probe:
        return ""
    by_model: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in probe["results"]:
        slot = by_model.setdefault(str(r.get("model")), {"ok": [], "rejected": []})
        slot["ok" if r.get("ok") else "rejected"].append(r)

    def _fmt(settings: dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(settings.items())) or "(defaults)"

    lines = [
        "## Verified runtime settings",
        "",
        f"Probed {probe.get('probed_at', '?')} with one real call per combination. "
        "Use only accepted combinations — a rejected one fails every row.",
        "",
    ]
    for model, slot in sorted(by_model.items()):
        lines.append(f"- `{model}`")
        if slot["ok"]:
            lines.append("  - accepted: " + "; ".join(_fmt(r["settings"]) for r in slot["ok"]))
        for r in slot["rejected"]:
            err = str(r.get("error") or "rejected")[:max_error_chars].replace("\n", " ")
            lines.append(f"  - rejected: {_fmt(r['settings'])} — {err}")
    return "\n".join(lines) + "\n"
