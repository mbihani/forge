"""The eval-engine registry is domain-agnostic: the core dispatches by
name, resolves pluggable engines through the registry (lazily importing
``anvil.domains.<name>`` by convention), and names no specific domain."""

from __future__ import annotations

import pytest

from anvil.eval.engines import (
    GENAI_ENGINE,
    is_valid_engine_name,
    load_engine,
    register_engine,
    registered_engines,
)
from anvil.runtime.models import EvalConfig

# ---------------------------------------------------------------------------
# Name validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["genai", "savesage", "hdfc", "sbi", "a", "x_2"])
def test_valid_engine_names(name):
    assert is_valid_engine_name(name)


@pytest.mark.parametrize("name", ["Savesage", "save-sage", "save.sage", "..", "1x", "", "os/x"])
def test_invalid_engine_names(name):
    assert not is_valid_engine_name(name)


# ---------------------------------------------------------------------------
# register / load.
# ---------------------------------------------------------------------------


def test_register_then_load_returns_the_callable():
    sentinel = object()

    def _fake_engine(**_kwargs):
        return sentinel

    register_engine("dummy_engine", _fake_engine)
    assert load_engine("dummy_engine") is _fake_engine
    assert "dummy_engine" in registered_engines()
    assert load_engine("dummy_engine")() is sentinel


def test_register_rejects_unsafe_name():
    with pytest.raises(ValueError, match="invalid engine name"):
        register_engine("Bad-Name", lambda **_k: None)


def test_load_unknown_engine_raises_not_falls_back():
    # No registered engine and no anvil.domains.<name> package → loud failure,
    # never a silent fallback to genai.
    with pytest.raises(ValueError, match="unknown eval engine"):
        load_engine("no_such_engine_xyz")


def test_load_rejects_unsafe_name_before_import():
    with pytest.raises(ValueError, match="invalid engine name"):
        load_engine("../evil")


# ---------------------------------------------------------------------------
# Config: engine is a validated free identifier, not a hardcoded enum.
# ---------------------------------------------------------------------------


def test_config_defaults_to_genai():
    assert EvalConfig().engine == GENAI_ENGINE


def test_config_accepts_any_identifier_engine():
    # A future domain name the core has never heard of is a valid config value;
    # existence is checked at dispatch time, not by the schema.
    assert EvalConfig(engine="hdfc").engine == "hdfc"
    assert EvalConfig(engine="savesage").engine == "savesage"


def test_config_rejects_non_identifier_engine():
    with pytest.raises(ValueError, match="lowercase identifier"):
        EvalConfig(engine="Bad-Name")
