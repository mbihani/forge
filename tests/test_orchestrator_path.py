"""The orchestrator extends ``anvil.__path__`` with the cloned agent
repo's ``src/anvil/`` so domain packages shipped in the agent repo
(e.g. ``anvil.domains.<name>``) become importable. Forge itself stays
domain-agnostic — no domain code lives in the forge repo.

These tests verify the generic path-extension mechanism: it works for
any agent repo that ships a ``src/anvil/domains/<name>/`` package that
registers an eval engine on import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from anvil.eval.engines import load_engine, registered_engines
from anvil.orchestrator.app import _extend_anvil_path

# A unique domain name per test module so import caching in
# ``sys.modules`` can't leak between this test and any other.
_DOMAIN = "fake_domain_path_test"


# ---------------------------------------------------------------------------
# Isolation — anvil.__path__ and the engine registry are process-global.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_anvil_path():
    """Snapshot and restore ``anvil.__path__`` and clean up any module
    and registry entry this test creates, so tests don't pollute each
    other or the rest of the suite."""
    import anvil

    saved_path = list(anvil.__path__)
    # Remove any stale registration from a previous run.
    saved_engines = set(registered_engines())
    yield
    anvil.__path__[:] = saved_path
    # Reset the lazily-captured forge-path snapshot so the next test
    # re-captures from its (restored) pristine ``anvil.__path__``.
    import anvil.orchestrator.app as _app_mod

    _app_mod._FORGE_ANVIL_PATH = None
    # Purge the fake domain module from the import cache. We must also
    # evict ``anvil.domains`` itself: it's a namespace package whose
    # ``__path__`` was built from a previous test's tmp_path, and leaving
    # it cached lets a later test's ``import anvil.domains.<name>`` find
    # files in a stale path instead of failing as expected. Forge has no
    # ``anvil/domains/`` dir, so any ``anvil.domains*`` in sys.modules
    # was created by these tests.
    for mod_name in list(sys.modules):
        if mod_name.startswith("anvil.domains"):
            del sys.modules[mod_name]
    # Restore the engine registry to its pre-test state.
    from anvil.eval.engines import _ENGINES

    current = set(_ENGINES)
    for name in current - saved_engines:
        del _ENGINES[name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_agent_repo(root: Path, domain: str = _DOMAIN, result: str = "fake-result") -> Path:
    """Build a minimal agent repo tree with a domain package that
    registers a fake eval engine on import.

    Structure::

        <root>/src/anvil/domains/<domain>/__init__.py

    The ``__init__.py`` calls ``register_engine`` so ``load_engine``
    finds it after ``_extend_anvil_path`` makes the package importable.
    ``result`` is what the registered engine returns, so a test can tell
    two clones that ship the SAME ``domain`` name apart by their code.
    """
    pkg_dir = root / "src" / "anvil" / "domains" / domain
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text(
        "from anvil.eval.engines import register_engine\n"
        "\n"
        "def fake_engine(**_kwargs):\n"
        f"    return {result!r}\n"
        "\n"
        f"register_engine({domain!r}, fake_engine)\n"
    )
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extend_path_makes_domain_importable(tmp_path: Path):
    """After ``_extend_anvil_path``, ``load_engine`` resolves the
    domain package from the cloned agent repo and returns its callable."""
    repo = _make_fake_agent_repo(tmp_path)
    _extend_anvil_path(repo)

    engine = load_engine(_DOMAIN)
    assert engine() == "fake-result"
    assert _DOMAIN in registered_engines()


def test_extend_path_is_idempotent(tmp_path: Path):
    """Calling twice must not duplicate the path entry — the dedup
    check uses the resolved string path."""
    import anvil

    repo = _make_fake_agent_repo(tmp_path)
    _extend_anvil_path(repo)
    _extend_anvil_path(repo)

    candidate = str((repo / "src" / "anvil").resolve())
    assert list(anvil.__path__).count(candidate) == 1


def test_extend_path_noop_without_src_anvil(tmp_path: Path):
    """A plain agent repo without ``src/anvil/`` is unaffected — no
    path entry added, and the domain remains unresolvable."""
    import anvil

    # A repo with no src/anvil/ at all.
    (tmp_path / "scaffold").mkdir()
    before = list(anvil.__path__)
    _extend_anvil_path(tmp_path)
    assert list(anvil.__path__) == before

    with pytest.raises(ValueError, match="unknown eval engine"):
        load_engine("nonexistent_domain_xyz")


def test_extend_path_resolves_symlinks(tmp_path: Path):
    """The dedup uses ``.resolve()`` so the same physical directory
    reached via a symlink is not added twice."""
    import anvil

    repo = _make_fake_agent_repo(tmp_path)
    # Create a symlink alias to the same repo root.
    alias = tmp_path / "alias"
    alias.symlink_to(repo)

    _extend_anvil_path(repo)
    _extend_anvil_path(alias)

    candidate = str((repo / "src" / "anvil").resolve())
    assert list(anvil.__path__).count(candidate) == 1


def test_load_engine_fails_without_extend(tmp_path: Path):
    """Sanity check: without ``_extend_anvil_path``, the domain package
    in the agent repo is NOT importable (forge has no domains dir)."""
    _make_fake_agent_repo(tmp_path)
    # Deliberately do NOT call _extend_anvil_path.
    with pytest.raises(ValueError, match="unknown eval engine"):
        load_engine(_DOMAIN)


def test_switching_sessions_loads_active_clone_not_stale(tmp_path: Path):
    """Regression: two sessions ship the SAME ``anvil.domains.<name>``
    from DIFFERENT clone paths with DIFFERENT engine code. After the
    orchestrator switches to the second session, ``load_engine`` must
    resolve the SECOND clone's code — not silently reuse the first
    session's cached module/engine (the process-global import-state
    bleed this guards against).
    """
    shared = "shared_domain_switch_test"
    repo1 = _make_fake_agent_repo(tmp_path / "s1", domain=shared, result="from-session-1")
    repo2 = _make_fake_agent_repo(tmp_path / "s2", domain=shared, result="from-session-2")

    # Session 1 loads and caches its engine (module + registry).
    _extend_anvil_path(repo1)
    assert load_engine(shared)() == "from-session-1"

    # Session 2 points at a different clone shipping the same domain.
    _extend_anvil_path(repo2)
    engine = load_engine(shared)
    assert engine() == "from-session-2", "must load session 2's code, not the cached session-1 code"

    # And the resolved module now comes from session 2's clone path.
    mod = sys.modules[f"anvil.domains.{shared}"]
    assert str(repo2.resolve()) in (mod.__file__ or "")


def test_switching_sessions_preserves_other_domain(tmp_path: Path):
    """Switching to a second session that ships a DIFFERENT domain must
    not evict or clobber the first session's still-valid domain — a
    concurrent session with a distinct domain keeps working."""
    repo_a = _make_fake_agent_repo(tmp_path / "a", domain="domain_a_test", result="a-result")
    repo_b = _make_fake_agent_repo(tmp_path / "b", domain="domain_b_test", result="b-result")

    _extend_anvil_path(repo_a)
    assert load_engine("domain_a_test")() == "a-result"

    _extend_anvil_path(repo_b)
    assert load_engine("domain_b_test")() == "b-result"
    # domain_a is untouched — still resolvable and unchanged.
    assert load_engine("domain_a_test")() == "a-result"


def test_clone_path_never_precedes_forge_path(tmp_path: Path):
    """A cloned agent repo is a full fork shipping the whole
    ``src/anvil/`` tree; its path must stay AFTER forge's own path so it
    can never shadow forge's core modules (only ``anvil.domains.<name>``
    falls through)."""
    import anvil

    repo = _make_fake_agent_repo(tmp_path)
    forge_entries = [os.path.realpath(p) for p in anvil.__path__]
    _extend_anvil_path(repo)

    resolved = [os.path.realpath(p) for p in anvil.__path__]
    candidate = str((repo / "src" / "anvil").resolve())
    # Every original forge entry precedes the newly added clone entry.
    clone_idx = resolved.index(candidate)
    for entry in forge_entries:
        assert resolved.index(entry) < clone_idx
