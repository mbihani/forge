"""Pydantic models for the ANVIL harness configuration.

The harness configuration is split across two YAML files by
mutability:

* ``scaffold/harness.yaml`` — **mutable** by the optimizer.
  Sampling, skills, rules, tools.
* ``harness/config.yaml`` — **immutable** at runtime. Endpoints,
  experiment paths, eval modes, loop meta-config.

Both files are validated with ``extra="forbid"`` so a misplaced
field fails loudly. The loader catches those errors and reraises
with a domain-specific message.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from anvil.eval.engines import is_valid_engine_name


class SamplingConfig(BaseModel):
    """Sampling parameters for a model call."""

    temperature: float | None = 0.7
    top_p: float | None = None
    max_tokens: int = 2048
    tool_choice: Literal["auto", "required", "none"] = "auto"
    max_tool_calls: int = 3


class SkillRef(BaseModel):
    file: str
    sampling: SamplingConfig | None = None

    @property
    def name(self) -> str:
        return Path(self.file).stem


class RuleRef(BaseModel):
    file: str

    @property
    def name(self) -> str:
        return Path(self.file).stem


class ToolRef(BaseModel):
    """Tool entry registered in the runtime agent's tool list."""

    name: str
    description: str | None = None


class LoopConfig(BaseModel):
    """Configuration for the ANVIL optimizer loop."""

    target_rounds: int = 50
    stretch_rounds: int = 100
    cost_budget_usd_per_round: float = 5.0
    early_stop_after_stalled_rounds: int = 10
    critique_lookback: int = 3
    revert_lookback: int = 20
    max_optimizer_turns: int = 30


class ParetoObjective(BaseModel):
    """One named objective used by the frontier gate."""

    model_config = ConfigDict(extra="forbid")

    name: str
    direction: Literal["maximize", "minimize"] = "maximize"
    # ``latency`` reads ``cost_metrics["latency_ms_median"]`` — the savesage
    # code-mode/latency objective; the others read the aggregate or a token/
    # context/row cost proxy.
    source: Literal["aggregate", "tokens", "context_chars", "n_rows", "latency"] = "aggregate"
    # Optional per-objective epsilon. A single scalar ``gate.epsilon`` cannot
    # serve objectives on different scales (accuracy ∈ [0,1], epsilon ~0.005 vs
    # latency in ms, epsilon ~hundreds). When set, this objective uses its own
    # epsilon in the frontier's keep/regress test; when None it falls back to
    # the global ``gate.epsilon``.
    epsilon: float | None = None


class ParetoConfig(BaseModel):
    """Optional multi-objective configuration for the frontier gate."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    objectives: list[ParetoObjective] = Field(default_factory=list)


class GateConfig(BaseModel):
    """Configuration for the round keep/revert gate.

    The gate decides whether a round's mutation is KEPT (fast-forward
    merged into the parent branch) or REVERTED (branch deleted).

    ``type`` selects the strategy:

    * ``frontier`` (default) — Pareto frontier. A mutation is KEPT only
      if it improves at least one tracked objective (per-judge scores +
      the aggregate) without regressing any other by more than
      ``epsilon``. The frontier — the best-so-far score per objective —
      persists to ``eval/runs/frontier.json`` and is loaded at the start
      of each round. On the first scored round (no frontier file) the
      frontier is initialized from the cached baseline. This closes
      the silent-regression hole in the legacy gate: a round that
      scores worse than a previous KEPT round is REVERTED, even if it
      still beats the original frozen baseline.
    * ``delta`` — Legacy frozen-baseline behavior, preserved for
      backward compatibility. A mutation is KEPT iff its aggregate beats
      the cached baseline aggregate (``score_delta > 0``). The frontier
      file is neither written nor read.

    ``pareto`` configures named, direction-aware objectives. When it is
    disabled the frontier uses only the aggregate score, preserving the
    single-objective behavior.

    ``epsilon`` is the minimum improvement on an objective to count as
    "better". With ``0.0`` a strict positive delta is required, so a tie
    (no objective improves) does not extend the frontier and is reverted.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["frontier", "delta"] = "frontier"
    epsilon: float = 0.0
    pareto: ParetoConfig = Field(default_factory=ParetoConfig)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_pareto_bool(cls, data):
        """Accept the pre-structured ``pareto: bool`` YAML shape."""
        if isinstance(data, dict) and isinstance(data.get("pareto"), bool):
            data = dict(data)
            data["pareto"] = {"enabled": data["pareto"]}
        return data

    @field_validator("epsilon")
    @classmethod
    def _epsilon_must_be_nonneg_finite(cls, v: float) -> float:
        """Reject negative or non-finite epsilon.

        A negative epsilon makes ties count as improvements (``delta >
        epsilon`` is True when epsilon < 0 and delta == 0). NaN/inf
        epsilon breaks every comparison in the gate.
        """
        if not math.isfinite(v) or v < 0:
            raise ValueError("epsilon must be >= 0 and finite")
        return v


class ExperimentsConfig(BaseModel):
    """MLflow experiment paths. Stable, declared in config.yaml."""

    runtime: str
    eval: str
    optimizer: str


class OptimizerBackendConfig(BaseModel):
    """Schema of ``harness/config.yaml > optimizer`` — backend selection.

    Selects where the optimizer agent runs (local ClaudeSDKClient vs a
    managed Omnigent server). All fields are optional so a config
    without the ``optimizer:`` section validates (defaults to ``local``;
    backward compatible). Read by the loop's ``_read_optimizer_config``
    helper, NOT by the runtime loader — the optimizer backend is a
    loop-plane concern.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["local", "omnigent"] = "local"
    # --- omnigent-only fields (ignored when backend: local) ---
    server_url: str = "http://localhost:6767"
    auth_token: str = ""
    agent_bundle_path: str = "agents/forge_optimizer.yaml"


class EvalModeConfig(BaseModel):
    rows: int
    buckets: dict[str, int] = Field(default_factory=dict)


class SplitConfig(BaseModel):
    """Data split configuration for anti-overfit enforcement.

    The golden set is partitioned into train (reserved for optimizer-visible
    use; currently excluded from scoring), dev (round-by-round eval), and test
    (held-out finalization only).
    Split is by hash of example_id — deterministic and stable across runs.
    """

    enabled: bool = False
    train_ratio: float = 0.6
    dev_ratio: float = 0.2
    seed: int = 42

    @model_validator(mode="after")
    def _ratios_must_define_three_partitions(self) -> SplitConfig:
        if not math.isfinite(self.train_ratio) or not 0 < self.train_ratio < 1:
            raise ValueError("split train_ratio must be finite and between 0 and 1")
        if not math.isfinite(self.dev_ratio) or not 0 < self.dev_ratio < 1:
            raise ValueError("split dev_ratio must be finite and between 0 and 1")
        if self.train_ratio + self.dev_ratio >= 1:
            raise ValueError("split train_ratio + dev_ratio must be less than 1")
        return self


class ScorerConfig(BaseModel):
    """Configuration for one eval scorer — LLM judge or programmatic.

    Two scorer ``type`` values contribute to the aggregate:

    * ``llm`` (default) — an MLflow judge scorer (Correctness,
      RetrievalGroundedness, the custom refusal judge, Safety). Scored
      by ``mlflow.genai.evaluate`` exactly as before.
    * ``programmatic`` — a deterministic check function loaded from
      ``data/evaluator.py`` and referenced by ``check_function``. Makes
      no LLM call; the check runs inside the same evaluate pipeline and
      returns a ``Feedback`` with the function's score.

    ``weight`` scales the scorer's contribution to the aggregate. The
    aggregate is a weighted average across all configured scorers; with
    the default weight of 1.0 for every scorer it reduces to the
    unweighted mean (the legacy behavior), so shipped scaffolds that
    list scorers as bare strings keep scoring identically.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["llm", "programmatic"] = "llm"
    weight: float = 1.0
    check_function: str | None = None

    @field_validator("weight")
    @classmethod
    def _weight_must_be_positive(cls, v: float) -> float:
        """Reject non-positive or non-finite weights.

        A zero or negative weight in a weighted average is a config
        mistake (use weight=1.0 or drop the scorer). NaN/inf breaks the
        aggregate normalization.
        """
        if not math.isfinite(v) or v <= 0:
            raise ValueError("scorer weight must be > 0 and finite")
        return v

    @model_validator(mode="after")
    def _check_function_only_on_programmatic(self) -> ScorerConfig:
        """``check_function`` is required for ``programmatic`` and
        forbidden for ``llm`` — a stray ``check_function`` on an llm
        scorer is a misplaced config that should fail loudly."""
        if self.type == "programmatic" and not self.check_function:
            raise ValueError(f"scorer {self.name!r}: type=programmatic requires a check_function")
        if self.type == "llm" and self.check_function is not None:
            raise ValueError(f"scorer {self.name!r}: type=llm must not set check_function")
        return self


class EvalConfig(BaseModel):
    """Eval-side configuration."""

    # Which eval engine scores a branch.
    #   genai — the built-in default: ``mlflow.genai.evaluate`` with
    #           LLM/programmatic scorers over the golden set (builds per-row
    #           RETRIEVER traces for RetrievalGroundedness).
    #   <name> — any pluggable domain engine registered under that name (see
    #            anvil.eval.engines); the runner resolves it through the
    #            registry, which lazily imports ``anvil.domains.<name>``. The
    #            core never enumerates domains here, so a new domain is a pure
    #            add. Validated as a lowercase identifier (it also becomes the
    #            trailing segment of that import path). Existence is checked at
    #            dispatch time by the registry, which fails loudly on an
    #            unknown engine rather than falling back to genai.
    engine: str = "genai"
    default_mode: Literal["quick", "standard", "full"] = "standard"
    # Judged-field slugs (MLflow style, e.g. ``rewards_programType``) dropped
    # from the headline ``aggregate`` on the savesage CODE-mode/latency path so
    # the accuracy FLOOR the latency gate protects measures real quality, not
    # stale-GT artifacts. Ignored by the prompt-mode path (full 28-field
    # aggregate) and by the genai engine.
    accuracy_exclude_fields: list[str] = Field(default_factory=list)
    held_out_test: bool = False
    split: SplitConfig = Field(default_factory=SplitConfig)
    modes: dict[str, EvalModeConfig] = Field(default_factory=dict)
    n_workers: int = 4
    inter_row_cooldown_s: float = 0.0
    scorers: list[ScorerConfig] = Field(
        default_factory=lambda: [
            ScorerConfig(name="correctness"),
            ScorerConfig(name="retrieval_groundedness"),
            ScorerConfig(name="refusal_appropriateness"),
        ]
    )
    safety_guard_threshold: float = 0.95

    @field_validator("engine")
    @classmethod
    def _engine_name_is_safe_identifier(cls, v: str) -> str:
        """Reject an engine name that is not a lowercase identifier.

        The name is dispatched by :mod:`anvil.eval.engines` and, for a
        pluggable engine, becomes the trailing segment of the import path
        ``anvil.domains.<name>`` — so an unsafe value (dots, slashes,
        ``..``) could redirect the import. Delegates to the shared
        :func:`anvil.eval.engines.is_valid_engine_name` so the rule stays
        in sync with the registry; whether the engine actually exists is
        checked at dispatch time by the registry.
        """
        if not is_valid_engine_name(v):
            raise ValueError(
                f"eval.engine {v!r} must be a lowercase identifier "
                r"matching ^[a-z][a-z0-9_]*$"
            )
        return v

    @field_validator("scorers", mode="before")
    @classmethod
    def _coerce_scorers(cls, v: object) -> object:
        """Backward compatibility: accept a list of bare scorer-name
        strings (the legacy config shape) by promoting each to a
        ``{name: <str>}`` dict, which :class:`ScorerConfig` then parses
        with ``type=llm`` and ``weight=1.0``. A list of dicts (the new
        shape) and a list of already-built ``ScorerConfig`` objects pass
        through unchanged. The shipped scaffold lists scorers as
        strings, so this keeps it scoring identically without a config
        migration."""
        if not isinstance(v, list):
            return v
        out: list[object] = []
        for item in v:
            if isinstance(item, str):
                out.append({"name": item})
            else:
                out.append(item)
        return out

    @model_validator(mode="after")
    def _scorer_names_must_be_unique(self) -> EvalConfig:
        """Reject duplicate scorer names.

        The aggregate stores weights by name (``weights = {c.name:
        c.weight ...}``), so a duplicate name silently overwrites the
        earlier weight while both entries remain in the
        numerator/denominator. This produces incorrect weighting and
        ambiguous MLflow columns (``{name}/value``, ``{name}/mean``).
        """
        seen: set[str] = set()
        for s in self.scorers:
            if s.name in seen:
                raise ValueError(
                    f"duplicate scorer name {s.name!r} — each scorer must have a unique name"
                )
            seen.add(s.name)
        return self


class ScaffoldYAML(BaseModel):
    """Schema of ``scaffold/harness.yaml`` — the optimizer-mutable file."""

    model_config = ConfigDict(extra="forbid")

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    skills: list[SkillRef] = Field(default_factory=list)
    rules: list[RuleRef] = Field(default_factory=list)
    tools: list[ToolRef] = Field(default_factory=list)


class RuntimeYAML(BaseModel):
    """Schema of ``harness/config.yaml`` — the immutable runtime file."""

    model_config = ConfigDict(extra="forbid")

    # Optimization mode — what FORGE mutates.
    #   prompt — prompt scaffolds (skills/rules/sampling in markdown + YAML).
    #            The default; backward compatible with all existing rounds.
    #   code   — agent Python code (MemorySystem subclasses in agents/).
    #            The eval imports the active agent module instead of
    #            composing a prompt from scaffold/.
    mode: Literal["prompt", "code"] = "prompt"
    # Dotted Python module path or .py file path of the active agent in
    # code mode. Ignored in prompt mode. Default: the passthrough baseline.
    agent_module: str = "anvil.agents.baseline"
    # LLM model names for the AI Gateway. These are FMAPI model names
    # (e.g. ``databricks-claude-sonnet-4-6``), not serving-endpoint URLs.
    # All three routes go through the same AI Gateway unified URL; the
    # model name selects which FMAPI model the gateway routes to. The
    # optimizer path uses the Claude Agent SDK against the gateway's
    # Anthropic route (``optimizer_endpoint`` is the FMAPI model name
    # for that route too).
    runtime_endpoint: str  # FMAPI model for the runtime agent
    optimizer_endpoint: str  # FMAPI model for the optimizer
    judge_endpoint: str  # FMAPI model for the judge
    experiments: ExperimentsConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    optimizer: OptimizerBackendConfig = Field(default_factory=OptimizerBackendConfig)


class HarnessConfig(BaseModel):
    """Merged view of both YAML files. Built by the loader."""

    mode: Literal["prompt", "code"] = "prompt"
    agent_module: str = "anvil.agents.baseline"
    runtime_endpoint: str
    optimizer_endpoint: str
    judge_endpoint: str
    experiments: ExperimentsConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    skills: list[SkillRef] = Field(default_factory=list)
    rules: list[RuleRef] = Field(default_factory=list)
    tools: list[ToolRef] = Field(default_factory=list)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    gate: GateConfig = Field(default_factory=GateConfig)

    @classmethod
    def from_split(cls, scaffold: ScaffoldYAML, runtime: RuntimeYAML) -> HarnessConfig:
        return cls(
            mode=runtime.mode,
            agent_module=runtime.agent_module,
            runtime_endpoint=runtime.runtime_endpoint,
            optimizer_endpoint=runtime.optimizer_endpoint,
            judge_endpoint=runtime.judge_endpoint,
            experiments=runtime.experiments,
            sampling=scaffold.sampling,
            skills=list(scaffold.skills),
            rules=list(scaffold.rules),
            tools=list(scaffold.tools),
            loop=runtime.loop,
            eval=runtime.eval,
            gate=runtime.gate,
        )


# Field names that belong in the *other* file. Used by the loader to
# generate domain-specific errors when an extra field happens to be
# one of the canonical fields on the opposite side of the split.
RUNTIME_FIELDS: frozenset[str] = frozenset(
    {
        "mode",
        "agent_module",
        "runtime_endpoint",
        "optimizer_endpoint",
        "judge_endpoint",
        "experiments",
        "loop",
        "eval",
        "gate",
    }
)
SCAFFOLD_FIELDS: frozenset[str] = frozenset({"sampling", "skills", "rules", "tools"})
