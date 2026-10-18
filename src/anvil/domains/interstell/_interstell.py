"""Bridge to the Interstell repo (``mayanck-bihani_data/interstell-ai``).

The engine re-uses Interstell's OWN eval code — ``evals.scorers`` (deterministic
checks), ``evals.judges`` (the Opus LLM judges), ``evals.summary`` (headline
metrics, provisional-row exclusions) and ``evals/labels.jsonl`` — so a round is
scored exactly as ``python -m evals.run_eval`` scores the agent. Nothing is
copied: the repo is put on ``sys.path`` at run time.

Environment (all optional):

* ``INTERSTELL_REPO_PATH`` — checkout of the Interstell repo (default ``~/interstell-ai``).
* ``INTERSTELL_APP_URL`` — the deployed App whose ``/api/turn`` runs each turn.
* ``INTERSTELL_APP_PROFILE`` — CLI profile with a USER OAuth token for the App
  (Apps reject PATs; default ``fevm-stable-oauth``).
* ``INTERSTELL_JUDGE_PROFILE`` — profile the judges authenticate with against the
  workspace AI Gateway (default ``fevm-stable``).
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

REPO_ENV = "INTERSTELL_REPO_PATH"
DEFAULT_REPO = Path.home() / "interstell-ai"
APP_URL_ENV = "INTERSTELL_APP_URL"
DEFAULT_APP_URL = "https://interstell-voice-agent-7474660648944264.aws.databricksapps.com"
APP_PROFILE_ENV = "INTERSTELL_APP_PROFILE"
DEFAULT_APP_PROFILE = "fevm-stable-oauth"
JUDGE_PROFILE_ENV = "INTERSTELL_JUDGE_PROFILE"
DEFAULT_JUDGE_PROFILE = "fevm-stable"

_lock = threading.Lock()
_app_cfg = None


def repo_path() -> Path:
    return Path(os.environ.get(REPO_ENV) or DEFAULT_REPO).expanduser()


def ensure_importable() -> Path:
    """Put the Interstell repo on ``sys.path`` and bind the judges' auth profile."""
    root = repo_path()
    if not (root / "evals" / "scorers.py").is_file() or not (root / "system_prompt.txt").is_file():
        raise FileNotFoundError(
            f"Interstell repo not found at {root} (set {REPO_ENV} to a checkout of interstell-ai)"
        )
    with _lock:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        os.environ.setdefault("INTERSTELL_ASSETS_DIR", str(root))
        from agent import auth  # noqa: PLC0415 - the Interstell package, importable only now

        if auth._cfg is None:
            # The judges authenticate through agent.auth's cached Config. Bind it to an
            # explicit profile: the forge launch env unsets DATABRICKS_CONFIG_PROFILE so the
            # optimizer's gateway auth is not hijacked.
            from databricks.sdk.core import Config  # noqa: PLC0415

            auth._cfg = Config(profile=os.environ.get(JUDGE_PROFILE_ENV) or DEFAULT_JUDGE_PROFILE)
    return root


def app_url() -> str:
    return (os.environ.get(APP_URL_ENV) or DEFAULT_APP_URL).rstrip("/")


def app_headers() -> dict[str, str]:
    """A fresh user OAuth bearer for the App (the SDK caches and refreshes it)."""
    global _app_cfg
    with _lock:
        if _app_cfg is None:
            from databricks.sdk.core import Config  # noqa: PLC0415

            _app_cfg = Config(profile=os.environ.get(APP_PROFILE_ENV) or DEFAULT_APP_PROFILE)
        cfg = _app_cfg
    return cfg.authenticate()
