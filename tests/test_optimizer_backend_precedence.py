"""Deployment env must be authoritative over a cloned repo's file config.

forge is domain-agnostic: it clones ARBITRARY target repos whose
``harness/config.yaml`` we do not control, then runs the optimizer round
against that clone. The optimizer backend, its Omnigent server URL, and
its auth are DEPLOYMENT/INFRA concerns — so the deployment environment
(env vars set by the Databricks App) must WIN over whatever the cloned
repo pinned.

Before this fix the deployed App set ``ANVIL_OPTIMIZER_BACKEND=omnigent``
but the round silently ran the LOCAL backend, producing a phantom noop
with zero Omnigent sessions. Root cause: ``_read_optimizer_config`` only
consulted the env var when the file's ``optimizer.backend`` was FALSY, and
forge's own ``harness/config.yaml`` pins ``backend: local`` (truthy). A
secondary bug preferred the file's ``server_url`` (``localhost:6767``)
over the workspace-derived URL.

These tests prove:

* env ``ANVIL_OPTIMIZER_BACKEND`` OVERRIDES a file ``backend: local``;
* env unset / empty-string preserves the file value;
* the omnigent server URL follows env/derived-over-file precedence;
* the selected backend + resolved URL land in the round JSON.

Each test's docstring names the production line it bites: reverting the
fix makes the assertion fail (verified by inspection of the guard/order).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# NOTE: import ``anvil.eval`` before ``anvil.loop.*`` — the package has a
# circular import between ``anvil.loop.frontier`` and ``anvil.eval.cache``
# (both reach ``anvil.runtime.models``) that only resolves cleanly when
# ``anvil.eval`` is fully initialized first. Matches tests/test_noop_commit.py.
import anvil.eval  # noqa: F401  (import-ordering side effect)
from anvil.loop.decision import Decision
from anvil.loop.round import (
    _build_round_json,
    _read_optimizer_config,
    _resolve_omnigent_backend_url,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(repo: Path, optimizer_block: str) -> Path:
    """Create ``<repo>/scaffold`` + ``<repo>/harness/config.yaml``.

    Returns the scaffold_root that ``_read_optimizer_config`` expects
    (config.yaml is its sibling under ``harness/``).
    """
    scaffold_root = repo / "scaffold"
    scaffold_root.mkdir(parents=True, exist_ok=True)
    harness = repo / "harness"
    harness.mkdir(parents=True, exist_ok=True)
    (harness / "config.yaml").write_text(optimizer_block, encoding="utf-8")
    return scaffold_root


# A minimal but realistic config that mirrors forge's own pinned values:
# a truthy ``backend: local`` plus a localhost server_url. This is exactly
# the shape that shadowed the env var before the fix.
_LOCAL_PINNED_CONFIG = """\
mode: prompt
optimizer:
  backend: local
  server_url: http://localhost:6767
"""


# ---------------------------------------------------------------------------
# (a) env ANVIL_OPTIMIZER_BACKEND=omnigent OVERRIDES file backend: local
# ---------------------------------------------------------------------------


def test_env_backend_overrides_truthy_file_backend(tmp_path, monkeypatch):
    """Bites round.py ``_read_optimizer_config`` env-precedence.

    Reverting to the old ``if not optimizer_cfg.get('backend')`` guard
    makes this fail: the truthy file ``local`` would block the env var.
    """
    scaffold_root = _write_config(tmp_path, _LOCAL_PINNED_CONFIG)
    monkeypatch.setenv("ANVIL_OPTIMIZER_BACKEND", "omnigent")

    cfg = _read_optimizer_config(scaffold_root)

    assert cfg["backend"] == "omnigent"


# ---------------------------------------------------------------------------
# (b) env unset -> file backend: local preserved
# ---------------------------------------------------------------------------


def test_env_unset_preserves_file_backend(tmp_path, monkeypatch):
    """Local dev keeps its file-pinned backend when the env is unset."""
    scaffold_root = _write_config(tmp_path, _LOCAL_PINNED_CONFIG)
    monkeypatch.delenv("ANVIL_OPTIMIZER_BACKEND", raising=False)

    cfg = _read_optimizer_config(scaffold_root)

    assert cfg["backend"] == "local"


# ---------------------------------------------------------------------------
# (c) env empty-string -> treated as unset (file preserved)
# ---------------------------------------------------------------------------


def test_env_empty_string_treated_as_unset(tmp_path, monkeypatch):
    """An empty env var must not clobber the file value with ''."""
    scaffold_root = _write_config(tmp_path, _LOCAL_PINNED_CONFIG)
    monkeypatch.setenv("ANVIL_OPTIMIZER_BACKEND", "")

    cfg = _read_optimizer_config(scaffold_root)

    assert cfg["backend"] == "local"


def test_env_backend_applies_when_file_has_no_optimizer_section(tmp_path, monkeypatch):
    """A cloned repo with NO optimizer block still gets the env backend."""
    scaffold_root = _write_config(tmp_path, "mode: prompt\n")
    monkeypatch.setenv("ANVIL_OPTIMIZER_BACKEND", "omnigent")

    cfg = _read_optimizer_config(scaffold_root)

    assert cfg["backend"] == "omnigent"


# ---------------------------------------------------------------------------
# (d) omnigent server-url precedence
# ---------------------------------------------------------------------------


def test_derived_url_beats_file_server_url(monkeypatch):
    """Bites round.py ``_resolve_omnigent_backend_url`` ordering.

    Backend forced omnigent, OMNIGENT_SERVER_URL unset, DATABRICKS_HOST
    set -> the resolved URL is the derived ``{host}/omnigent`` NOT the
    file's ``localhost:6767``. Reverting to the old
    ``optimizer_cfg.get('server_url') or resolve_omnigent_server_url()``
    order makes this fail (localhost would win).
    """
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://myworkspace.cloud.databricks.com")
    optimizer_cfg: dict[str, Any] = {"server_url": "http://localhost:6767"}

    url = _resolve_omnigent_backend_url(optimizer_cfg)

    assert url == "https://myworkspace.cloud.databricks.com/omnigent"


def test_explicit_env_url_wins(monkeypatch):
    """OMNIGENT_SERVER_URL env (non-empty) beats both derived and file."""
    monkeypatch.setenv("OMNIGENT_SERVER_URL", "https://explicit.example/omnigent")
    monkeypatch.setenv("DATABRICKS_HOST", "https://myworkspace.cloud.databricks.com")
    optimizer_cfg: dict[str, Any] = {"server_url": "http://localhost:6767"}

    url = _resolve_omnigent_backend_url(optimizer_cfg)

    assert url == "https://explicit.example/omnigent"


def test_file_server_url_used_when_no_env_or_host(monkeypatch):
    """With neither env nor DATABRICKS_HOST, the file value is the fallback."""
    monkeypatch.delenv("OMNIGENT_SERVER_URL", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    optimizer_cfg: dict[str, Any] = {"server_url": "http://localhost:6767"}

    url = _resolve_omnigent_backend_url(optimizer_cfg)

    assert url == "http://localhost:6767"


# ---------------------------------------------------------------------------
# (e) round JSON records the selected backend + resolved URL
# ---------------------------------------------------------------------------


def test_round_json_records_selected_backend_and_url():
    """Bites round.py ``_build_round_json`` payload additions.

    The round record must carry ``optimizer_backend`` (and, for omnigent,
    ``optimizer_server_url``) so a 'local noop' is distinguishable from an
    'omnigent noop/INFRA_FAIL' without reading the transcript.
    """
    payload = _build_round_json(
        round_id=7,
        branch="anvil/round-7",
        commit_sha="abc123",
        parent_sha="def456",
        decision=Decision.NOOP,
        action_kind="noop",
        eval_report=None,
        baseline_score=0.5,
        score_delta=None,
        parse_status="ok",
        notes="",
        optimizer_backend="omnigent",
        optimizer_server_url="https://myworkspace.cloud.databricks.com/omnigent",
    )

    assert payload["optimizer_backend"] == "omnigent"
    assert (
        payload["optimizer_server_url"]
        == "https://myworkspace.cloud.databricks.com/omnigent"
    )


def test_round_json_backend_fields_default_none_for_local():
    """Local rounds leave the URL None (backward-compatible, additive)."""
    payload = _build_round_json(
        round_id=1,
        branch="anvil/round-1",
        commit_sha="abc",
        parent_sha="def",
        decision=Decision.NOOP,
        action_kind="noop",
        eval_report=None,
        baseline_score=None,
        score_delta=None,
        parse_status="ok",
        notes="",
        optimizer_backend="local",
    )

    assert payload["optimizer_backend"] == "local"
    assert payload["optimizer_server_url"] is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
