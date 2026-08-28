"""ANVIL optimizer plane.

The optimizer proposes one structural mutation per round, structured as
an ``OptimizerAction`` (Pydantic discriminated union). The session emits
exactly one fenced JSON action block which the parser validates; any
malformed output collapses to ``NoopAction`` so the loop is never blocked
by a bad transcript.

Two pluggable backends implement :class:`OptimizerBackend`:

* :class:`LocalBackend` — the existing async ``ClaudeSDKClient`` session
  (default; selected by ``optimizer.backend: local`` or no backend field).
* :class:`OmnigentBackend` — runs the optimizer agent on a managed
  Omnigent server over REST (selected by ``optimizer.backend: omnigent``).

:func:`get_backend` selects between them from a :class:`BackendConfig`.
:func:`run_optimizer_session` remains as a thin backward-compatible wrapper
over :class:`LocalBackend` (3-tuple return) so existing callers and the
loop's monkeypatch-based tests keep working unchanged.

The optimizer plane never runs git commands and never writes files itself
— the loop's applier does that. This separation makes the optimizer
mockable (loop tests inject a fake action) and the loop auditable (every
write goes through one place).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anvil.optimizer.actions import (
    AddRuleAction,
    AddSkillAction,
    ChangeSamplingAction,
    DeleteAgentAction,
    DeleteRuleAction,
    DeleteSkillAction,
    EditRuleAction,
    EditSkillAction,
    NoopAction,
    OptimizerAction,
    WriteAgentAction,
)
from anvil.optimizer.applier import ApplyError, ApplyResult, apply_action
from anvil.optimizer.omnigent_backend import OmnigentBackend
from anvil.optimizer.omnigent_client import (
    OmnigentClient,
    OmnigentError,
    SessionCreateMetadata,
)
from anvil.optimizer.parser import ParseResult, parse_action
from anvil.optimizer.session import (
    LocalBackend,
    OptimizerBackend,
    OptimizerResult,
    extract_message_text,
    run_optimizer_session,
    setup_anthropic_env,
)

__all__ = [
    "AddRuleAction",
    "AddSkillAction",
    "ApplyError",
    "ApplyResult",
    "BackendConfig",
    "ChangeSamplingAction",
    "DeleteAgentAction",
    "DeleteRuleAction",
    "DeleteSkillAction",
    "EditRuleAction",
    "EditSkillAction",
    "LocalBackend",
    "NoopAction",
    "OmnigentBackend",
    "OmnigentClient",
    "OmnigentError",
    "OptimizerAction",
    "OptimizerBackend",
    "OptimizerResult",
    "ParseResult",
    "SessionCreateMetadata",
    "WriteAgentAction",
    "apply_action",
    "extract_message_text",
    "get_backend",
    "parse_action",
    "run_optimizer_session",
    "setup_anthropic_env",
]

_DEFAULT_AGENT_BUNDLE = "agents/forge_optimizer.yaml"
_DEFAULT_SERVER_URL = "http://localhost:6767"
_DEFAULT_MODEL = "databricks-claude-opus-4-7"
_DEFAULT_MAX_TURNS = 70


@dataclass
class BackendConfig:
    """Configuration for :func:`get_backend`.

    The ``backend`` field selects the implementation; the remaining fields
    are passed to the chosen backend. Local-backend fields are read by the
    loop from the round context; omnigent fields are read from the
    ``optimizer:`` section of ``harness/config.yaml``.

    A plain ``dict`` is accepted by :func:`get_backend` and coerced —
    unknown keys are dropped so a partial config (e.g. only ``backend``)
    selects the default backend without raising.
    """

    backend: str = "local"
    # LocalBackend (round context).
    cwd: str | None = None
    profile: str | None = None
    optimizer_endpoint: str | None = None
    experiment_name: str | None = None
    round_id: int | None = None
    setup_env: bool = True
    # OmnigentBackend (harness/config.yaml > optimizer).
    server_url: str = _DEFAULT_SERVER_URL
    auth_token: str | None = None
    agent_bundle_path: str = _DEFAULT_AGENT_BUNDLE
    model: str | None = None
    max_turns: int = _DEFAULT_MAX_TURNS
    # Extra session-create metadata for the Omnigent backend (e.g. a
    # host_id for a managed host). Opaque to the local backend.
    omnigent_metadata: SessionCreateMetadata = field(default_factory=SessionCreateMetadata)


def get_backend(config: BackendConfig | dict[str, Any] | None = None) -> OptimizerBackend:
    """Select and build an :class:`OptimizerBackend` from config.

    ``config`` may be a :class:`BackendConfig` or a plain ``dict`` (unknown
    keys are dropped). ``backend`` defaults to ``"local"`` (backward
    compatible — no ``optimizer.backend`` field means the existing
    ClaudeSDKClient path).

    * ``backend: local`` (or unset) → :class:`LocalBackend` configured from
      the round-context fields (``cwd`` is required for a usable backend).
    * ``backend: omnigent`` → :class:`OmnigentBackend` with a fresh
      :class:`OmnigentClient` built from ``server_url`` / ``auth_token``.
    """
    cfg = _coerce_config(config)
    if cfg.backend == "omnigent":
        client = OmnigentClient(cfg.server_url, cfg.auth_token)
        return OmnigentBackend(
            client=client,
            agent_bundle_path=Path(cfg.agent_bundle_path),
            server_url=cfg.server_url,
            default_model=cfg.model or _DEFAULT_MODEL,
            default_max_turns=cfg.max_turns,
            create_metadata=cfg.omnigent_metadata,
        )
    if cfg.backend != "local":
        raise ValueError(
            f"unknown optimizer backend {cfg.backend!r}; expected 'local' or 'omnigent'"
        )
    return LocalBackend(
        cwd=cfg.cwd or "",
        profile=cfg.profile,
        setup_env=cfg.setup_env,
        optimizer_endpoint=cfg.optimizer_endpoint,
        experiment_name=cfg.experiment_name,
        round_id=cfg.round_id,
    )


def _coerce_config(config: BackendConfig | dict[str, Any] | None) -> BackendConfig:
    """Normalize a dict/None/config into a BackendConfig, dropping unknown keys."""
    if config is None:
        return BackendConfig()
    if isinstance(config, BackendConfig):
        return config
    if isinstance(config, dict):
        known = {f for f in BackendConfig.__dataclass_fields__}
        return BackendConfig(**{k: v for k, v in config.items() if k in known})
    raise TypeError(f"config must be BackendConfig, dict, or None; got {type(config).__name__}")
