"""Every optimized domain gets its own MLflow experiments under ``experiments.root``."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from anvil.loop.optimizer_artifacts import resolve_persistence_settings
from anvil.runtime.models import RuntimeYAML

_BASE = {
    "runtime_endpoint": "m",
    "optimizer_endpoint": "m",
    "judge_endpoint": "m",
}


def _cfg(engine: str = "agent_a", **extra) -> RuntimeYAML:
    return RuntimeYAML.model_validate({**_BASE, "eval": {"engine": engine}, **extra})


def test_defaults_are_per_domain() -> None:
    e = _cfg().experiments
    assert (e.eval, e.optimizer, e.runtime) == (
        "/Shared/forge/agent_a/eval",
        "/Shared/forge/agent_a/optimizer",
        "/Shared/forge/agent_a/runtime",
    )


def test_two_domains_never_share() -> None:
    a, b = _cfg("agent_a").experiments, _cfg("agent_b").experiments
    assert {a.eval, a.optimizer, a.runtime}.isdisjoint({b.eval, b.optimizer, b.runtime})


def test_custom_root_and_explicit_path_inside_domain_folder() -> None:
    e = _cfg(
        experiments={
            "root": "/Users/me@x.com/forge",
            "eval": "/Users/me@x.com/forge/agent_a/eval-v2",
        }
    ).experiments
    assert e.eval == "/Users/me@x.com/forge/agent_a/eval-v2"
    assert e.optimizer == "/Users/me@x.com/forge/agent_a/optimizer"


@pytest.mark.parametrize(
    "experiments",
    [
        {"eval": "/Shared/anvil-eval"},  # the old shared default
        {"optimizer": "/Shared/forge/agent_b/optimizer"},  # another domain's folder
        {"runtime": "/Shared/forge/agent_a"},  # the folder itself, not inside it
    ],
)
def test_paths_outside_the_domain_folder_fail(experiments: dict) -> None:
    with pytest.raises(ValidationError, match="outside domain 'agent_a'"):
        _cfg(experiments=experiments)


def test_relative_root_rejected() -> None:
    with pytest.raises(ValidationError, match="absolute"):
        _cfg(experiments={"root": "forge"})


def test_persistence_experiment_must_stay_in_domain() -> None:
    with pytest.raises(ValidationError, match="persistence.experiment"):
        _cfg(persistence={"experiment": "/Shared/anvil-optimizer"})


def test_sink_uses_the_domain_optimizer_experiment(tmp_path: Path) -> None:
    (tmp_path / "scaffold").mkdir()
    (tmp_path / "harness").mkdir()
    cfg = tmp_path / "harness" / "config.yaml"
    cfg.write_text("eval: {engine: agent_a}\n")
    assert resolve_persistence_settings(tmp_path / "scaffold") == (
        True,
        "/Shared/forge/agent_a/optimizer",
    )
    cfg.write_text("eval: {engine: agent_a}\npersistence: {experiment: /Shared/anvil-optimizer}\n")
    with pytest.raises(ValueError, match="outside domain"):
        resolve_persistence_settings(tmp_path / "scaffold")
