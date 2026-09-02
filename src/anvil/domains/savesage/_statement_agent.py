"""Bridge to the local Savesage ``statement-agent`` package.

The Savesage domain reuses — never re-implements — the production
extraction adapter and the field-accuracy judge from the sibling
``statement-agent`` repo. Those modules are top-level packages
(``config``, ``contracts``, ``judge``, ``harness``, ``rules``,
``graph``) rather than an installed distribution, so this helper puts
their repo root on ``sys.path`` exactly once, on first use.

The path is resolved from the ``SAVESAGE_STATEMENT_AGENT_PATH`` env var
when set, else the known local checkout. It is appended (not inserted at
the front) so ``anvil.*`` always wins; the Savesage module names are
unique to that repo, so there is no collision. Import is lazy: nothing
here runs at ANVIL import time, so the NeoVolt path and the offline
unit tests never touch the statement-agent tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Default local checkout on this machine. Overridable so the domain is
# not pinned to one path (CI / another clone can point elsewhere).
_DEFAULT_STATEMENT_AGENT_PATH = Path.home() / "savesage-build" / "main" / "statement-agent"

_ADDED = False


def statement_agent_root() -> Path:
    """Return the resolved ``statement-agent`` repo root."""
    raw = os.environ.get("SAVESAGE_STATEMENT_AGENT_PATH")
    return Path(raw) if raw else _DEFAULT_STATEMENT_AGENT_PATH


def ensure_importable() -> Path:
    """Put the statement-agent root on ``sys.path`` (idempotent).

    Returns the root path. Raises ``FileNotFoundError`` with an
    actionable message if the checkout is missing — the Savesage domain
    cannot score or extract without it.
    """
    global _ADDED
    root = statement_agent_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"statement-agent checkout not found at {root}. Set "
            "SAVESAGE_STATEMENT_AGENT_PATH to the repo root that contains "
            "judge/, harness/, contracts/, rules/, config.py."
        )
    if not _ADDED:
        path_str = str(root)
        if path_str not in sys.path:
            sys.path.append(path_str)
        _ADDED = True
    return root
