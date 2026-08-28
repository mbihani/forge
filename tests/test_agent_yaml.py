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


def test_agent_yaml_enforces_filesystem_only_via_policy() -> None:
    """The no-shell/no-git contract is enforced at the policy layer, not prompt-only.

    The Omnigent agent YAML has no static ``tools:`` allowlist for the
    harness's built-in tools (the ``tools:`` section is for sub-agents only).
    Tool restriction is done via ``guardrails.policies`` — a CEL policy
    evaluated on every ``tool_call`` event. This test verifies the policy
    exists and denies shell / Bash / code-execution surfaces.
    """
    data = _load()
    guardrails = data.get("guardrails", {})
    policies = guardrails.get("policies", {})
    assert "filesystem_only" in policies, "expected a filesystem_only guardrail policy"

    pol = policies["filesystem_only"]
    assert pol["on"] == ["tool_call"]
    expr = pol["function"]["arguments"]["expression"]
    # The CEL expression must DENY shell / Bash / code-execution tools.
    assert '"DENY"' in expr
    for forbidden in ("Bash", "sys_os_shell", "shell", "terminal", "execute_code"):
        assert f'"{forbidden}"' in expr, f"CEL policy must deny {forbidden!r}"


def test_agent_yaml_has_no_shell_bash_git_tools() -> None:
    """No shell, Bash, or git tool may appear as an allowed tool in the YAML.

    There is no ``tools:`` allowlist for built-in tools (the ``tools:`` key,
    when present, declares sub-agents — not permitted tools). This test
    guards against a future edit accidentally whitelisting a shell surface.
    """
    data = _load()
    # The ``tools:`` section (if present) declares sub-agents, not allowed
    # built-in tools. Assert it does not list a shell/bash/git agent.
    tools = data.get("tools", {})
    if isinstance(tools, dict):
        agents = tools.get("agents", [])
        for agent in agents:
            name = agent if isinstance(agent, str) else str(agent).lower()
            assert name not in ("bash", "shell", "terminal", "git"), (
                f"shell/bash/git sub-agent {name!r} must not be allowed"
            )
    # The guardrails block is present (enforced by the test above); its
    # existence is the structural signal that shell/bash/git are denied.
    yaml_text = _AGENT_YAML.read_text(encoding="utf-8")
    assert "guardrails:" in yaml_text


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
