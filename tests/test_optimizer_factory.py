"""Tests for :func:`get_backend` — the optimizer backend factory.

Verifies the factory selects the right backend from a ``BackendConfig``
or a plain ``dict``, that the default (no config) is the local backend
(backward compatible), and that omnigent config builds an
:class:`OmnigentBackend` with a fresh :class:`OmnigentClient`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anvil.optimizer import BackendConfig, LocalBackend, OmnigentBackend, get_backend


def test_get_backend_none_returns_local() -> None:
    """No config at all → LocalBackend (backward compatible)."""
    backend = get_backend(None)
    assert isinstance(backend, LocalBackend)


def test_get_backend_default_returns_local() -> None:
    backend = get_backend(BackendConfig())
    assert isinstance(backend, LocalBackend)


def test_get_backend_local_from_dict() -> None:
    backend = get_backend({"backend": "local", "cwd": "/tmp/anvil"})
    assert isinstance(backend, LocalBackend)
    assert backend.cwd == "/tmp/anvil"


def test_get_backend_local_is_default_when_backend_absent() -> None:
    """A dict without a ``backend`` key defaults to local."""
    backend = get_backend({"cwd": "/tmp"})
    assert isinstance(backend, LocalBackend)


def test_get_backend_omnigent_from_dict(tmp_path: Path) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        "spec_version: 1\nname: x\nexecutor: {type: omnigent}\n", encoding="utf-8"
    )
    backend = get_backend(
        {
            "backend": "omnigent",
            "server_url": "http://omni:9999",
            "auth_token": "tok-123",
            "agent_bundle_path": str(agent_yaml),
            "model": "custom-model",
            "max_turns": 42,
        }
    )
    assert isinstance(backend, OmnigentBackend)
    assert backend.server_url == "http://omni:9999"
    assert backend.default_model == "custom-model"
    assert backend.default_max_turns == 42
    assert backend.agent_bundle_path == agent_yaml


def test_get_backend_omnigent_uses_defaults_for_missing_fields(tmp_path: Path) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text("spec_version: 1\nname: x\n", encoding="utf-8")
    backend = get_backend({"backend": "omnigent", "agent_bundle_path": str(agent_yaml)})
    assert isinstance(backend, OmnigentBackend)
    # Defaults from the module constants.
    assert backend.server_url == "http://localhost:6767"
    assert backend.default_model == "databricks-claude-opus-4-7"
    assert backend.default_max_turns == 70


def test_get_backend_omnigent_client_has_auth_header(tmp_path: Path) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text("spec_version: 1\nname: x\n", encoding="utf-8")
    backend = get_backend(
        {"backend": "omnigent", "auth_token": "secret", "agent_bundle_path": str(agent_yaml)}
    )
    assert backend.client._client.headers["Authorization"] == "Bearer secret"  # type: ignore[attr-defined]


def test_get_backend_omnigent_no_token_leaves_auth_unset(tmp_path: Path) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text("spec_version: 1\nname: x\n", encoding="utf-8")
    backend = get_backend({"backend": "omnigent", "agent_bundle_path": str(agent_yaml)})
    assert "Authorization" not in backend.client._client.headers  # type: ignore[attr-defined]


def test_get_backend_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown optimizer backend"):
        get_backend({"backend": "cloud"})


def test_get_backend_drops_unknown_dict_keys() -> None:
    """A dict with extra keys does not raise — unknown keys are silently dropped."""
    backend = get_backend({"backend": "local", "cwd": "/tmp", "bogus_field": 123})
    assert isinstance(backend, LocalBackend)
    assert backend.cwd == "/tmp"


def test_get_backend_accepts_backend_config_dataclass(tmp_path: Path) -> None:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text("spec_version: 1\nname: x\n", encoding="utf-8")
    cfg = BackendConfig(
        backend="omnigent",
        server_url="http://omni:8080",
        agent_bundle_path=str(agent_yaml),
    )
    backend = get_backend(cfg)
    assert isinstance(backend, OmnigentBackend)
    assert backend.server_url == "http://omni:8080"


def test_get_backend_invalid_type_raises() -> None:
    with pytest.raises(TypeError, match="config must be BackendConfig"):
        get_backend("not-a-config")  # type: ignore[arg-type]


def test_backend_config_defaults() -> None:
    cfg = BackendConfig()
    assert cfg.backend == "local"
    assert cfg.server_url == "http://localhost:6767"
    assert cfg.agent_bundle_path == "agents/forge_optimizer.yaml"
    assert cfg.max_turns == 70
    assert cfg.model is None
    assert cfg.auth_token is None
