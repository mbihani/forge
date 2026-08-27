"""Tests for ``agents/forge_optimizer.yaml`` — the Omnigent agent bundle spec.

Validates the YAML is well-formed, has the required fields for the
Omnigent agent spec, and can be packaged by ``_build_agent_bundle``
into a tar.gz whose root ``config.yaml`` preserves the structure with
the round-specific model/max_turns substituted.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import yaml

from anvil.optimizer.omnigent_backend import _build_agent_bundle

_AGENT_YAML = Path(__file__).resolve().parents[1] / "agents" / "forge_optimizer.yaml"


def _load() -> dict:
    return yaml.safe_load(_AGENT_YAML.read_text(encoding="utf-8"))


def test_agent_yaml_exists_and_is_valid_yaml() -> None:
    assert _AGENT_YAML.is_file()
    data = _load()
    assert isinstance(data, dict)


def test_agent_yaml_has_required_top_level_fields() -> None:
    data = _load()
    assert data["spec_version"] == 1
    assert data["name"] == "forge_optimizer"
    assert "description" in data
    assert "prompt" in data
    assert "executor" in data
    assert "os_env" in data


def test_agent_yaml_executor_shape() -> None:
    data = _load()
    executor = data["executor"]
    assert executor["type"] == "omnigent"
    assert "model" in executor
    assert isinstance(executor["model"], str) and executor["model"]
    config = executor["config"]
    assert config["harness"] == "claude-sdk"
    assert isinstance(config["max_turns"], int)
    assert config["max_turns"] > 0


def test_agent_yaml_default_model_is_configurable() -> None:
    """The default model is a named FMAPI model (not a hardcoded Anthropic ID)."""
    data = _load()
    model = data["executor"]["model"]
    assert model.startswith("databricks-")


def test_agent_yaml_default_max_turns_is_70() -> None:
    data = _load()
    assert data["executor"]["config"]["max_turns"] == 70


def test_agent_yaml_prompt_forbids_shell_and_git() -> None:
    """The system prompt must enforce the no-shell/no-git contract."""
    prompt = _load()["prompt"]
    assert "Bash" in prompt or "shell" in prompt.lower()
    assert "git" in prompt.lower()
    assert "json-action" in prompt


def test_agent_yaml_os_env_is_caller_process() -> None:
    data = _load()
    assert data["os_env"]["type"] == "caller_process"


def test_build_agent_bundle_round_trips_the_real_yaml() -> None:
    """The real YAML can be packaged and the substitution is surgical."""
    bundle = _build_agent_bundle(_AGENT_YAML, model="databricks-claude-opus-5", max_turns=55)
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        assert tar.getnames() == ["config.yaml"]
        config = yaml.safe_load(tar.extractfile("config.yaml").read())  # type: ignore[union-attr]

    # Substituted fields.
    assert config["executor"]["model"] == "databricks-claude-opus-5"
    assert config["executor"]["config"]["max_turns"] == 55
    # Preserved fields.
    assert config["spec_version"] == 1
    assert config["name"] == "forge_optimizer"
    assert config["executor"]["type"] == "omnigent"
    assert "prompt" in config
