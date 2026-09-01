"""Eval-engine registry — the domain-agnostic dispatch seam.

``evaluate_branch`` scores a branch through an *engine*. ``genai`` is
ANVIL's built-in default (``mlflow.genai.evaluate`` over a golden set);
any other engine is a **pluggable domain** that registers itself here.
The core never names a specific domain — it looks an engine up by name,
lazily importing the domain package by convention when needed. Adding a
new domain (hdfc, sbi, a non-savesage task) is therefore a pure add: a
new ``anvil.domains.<name>`` package that calls :func:`register_engine`
on import, and zero edits to this module or the runner.

Contract for a registered engine callable::

    def engine(*, scaffold_root, runtime_config_path, golden_set_path,
               profile, mode, **_kwargs) -> EvalReport

It must accept ``**_kwargs`` so the runner can pass a superset of
arguments (genai-only kwargs like ``include_safety``) without each
engine having to declare them.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from typing import Any

# ANVIL's built-in engine. Handled inline in ``evaluate_branch`` (its
# implementation IS the runner body), so it is NOT registered here — the
# runner treats this one name as the default and dispatches everything
# else through the registry. Named as a constant so the runner does not
# carry a bare string literal.
GENAI_ENGINE = "genai"

# Engine name shape. Also the trailing segment of the convention import
# path ``anvil.domains.<name>``, so validating it here doubles as a guard
# against import-path injection through a crafted config value.
_ENGINE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\Z")

_ENGINES: dict[str, Callable[..., Any]] = {}


def is_valid_engine_name(name: str) -> bool:
    """True if ``name`` is a safe engine identifier (lowercase, no dots/slashes)."""
    return bool(_ENGINE_NAME_RE.match(name))


def register_engine(name: str, fn: Callable[..., Any]) -> None:
    """Register a pluggable eval engine under ``name``.

    Called by a domain package's ``__init__`` on import (e.g.
    ``anvil.domains.savesage`` registers ``"savesage"``). Re-registering
    the same name overwrites — harmless on a module re-import.
    """
    if not is_valid_engine_name(name):
        raise ValueError(f"invalid engine name {name!r}: must match {_ENGINE_NAME_RE.pattern}")
    _ENGINES[name] = fn


def load_engine(name: str) -> Callable[..., Any]:
    """Return the engine callable for ``name`` (never ``genai`` — that's inline).

    Resolution order:

    1. Already registered → return it.
    2. Not registered → import ``anvil.domains.<name>`` by convention (the
       package self-registers on import) and look again.

    Raises :class:`ValueError` for an unsafe name, a missing domain
    package, or a package that imported but did not register an engine —
    so a misconfigured ``eval.engine`` fails loudly rather than silently
    falling back to genai.
    """
    if not is_valid_engine_name(name):
        raise ValueError(f"invalid engine name {name!r}: must match {_ENGINE_NAME_RE.pattern}")
    fn = _ENGINES.get(name)
    if fn is not None:
        return fn
    # Convention: a domain named <name> lives at anvil.domains.<name> and
    # registers its engine on import. Import is deferred to here so the
    # genai path and the offline core never import any domain package.
    try:
        importlib.import_module(f"anvil.domains.{name}")
    except ModuleNotFoundError as exc:
        # The missing module is the engine package itself or a parent
        # path component (e.g. ``anvil.domains`` not yet created) → the
        # engine is unknown. A missing dependency *inside* the engine
        # package has a different ``exc.name`` and should surface as-is.
        target = f"anvil.domains.{name}"
        if exc.name == target or (exc.name is not None and target.startswith(f"{exc.name}.")):
            raise ValueError(
                f"unknown eval engine {name!r}: not registered and no "
                f"anvil.domains.{name} package to import"
            ) from exc
        raise  # missing dependency inside the engine package — re-raise
    fn = _ENGINES.get(name)
    if fn is None:
        raise ValueError(
            f"eval engine {name!r}: anvil.domains.{name} imported but did not "
            f"register an engine (expected a register_engine({name!r}, ...) call)"
        )
    return fn


def registered_engines() -> list[str]:
    """Names currently in the registry (excludes the built-in genai). Test aid."""
    return sorted(_ENGINES)
