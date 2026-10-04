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
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A lever value: a scalar the optimizer may select from an allowlist.
LeverValue = str | int | float | bool

# The one lever forge core understands: it selects the runtime model for
# every engine (see :meth:`HarnessConfig.effective_runtime_model`). Every
# other lever name is opaque to core and interpreted by the domain engine.
MODEL_LEVER = "model"

_LEVER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\Z")


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
    # Upper bound on mutations one round may apply together via a
    # ``compound`` action. 1 (the default) keeps the classic
    # one-mutation-per-round loop and rejects compound actions.
    max_mutations_per_round: int = Field(default=1, ge=1, le=5)


class LeverSpec(BaseModel):
    """One optimizer-selectable runtime lever, declared in ``harness/config.yaml``.

    The immutable config owns the allowlist; the mutable
    ``scaffold/harness.yaml > levers`` holds the current choice. The
    optimizer changes a lever with a ``set_lever`` action, which the
    applier validates against ``allowed`` — so the optimizer can never
    select a value the operator did not list.
    """

    model_config = ConfigDict(extra="forbid")

    allowed: list[LeverValue] = Field(min_length=1)
    # Value used when the scaffold does not set the lever. Must be one of
    # ``allowed``. For the ``model`` lever, omitting it means
    # ``runtime_endpoint``.
    default: LeverValue | None = None
    # Shown to the optimizer in the round prompt — what the lever does.
    description: str = ""

    @model_validator(mode="after")
    def _default_is_allowed(self) -> LeverSpec:
        if self.default is not None and self.default not in self.allowed:
            raise ValueError(f"lever default {self.default!r} is not in allowed {self.allowed!r}")
        return self


class ModelCatalogConfig(BaseModel):
    """Where model prices come from (``harness/config.yaml > model_catalog``).

    ``scripts/sync_model_catalog.py`` exports the operator's Google Sheet
    (``sheet_id`` / ``sheet_range``) into ``path``; the loop reads only
    that committed file, never the sheet. ``context_tier`` picks the
    ``Short (<=200k)`` or ``Long (>200k)`` row for models priced per tier.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = "harness/model_catalog.csv"
    sheet_id: str | None = None
    sheet_range: str = "Sheet1"
    context_tier: Literal["short", "long"] = "short"


def _validate_lever_names(names: Any) -> None:
    for name in names:
        if not isinstance(name, str) or not _LEVER_NAME_RE.match(name):
            raise ValueError(f"invalid lever name {name!r}: must match {_LEVER_NAME_RE.pattern}")


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


DEFAULT_EXPERIMENTS_ROOT = "/Shared/forge"
EXPERIMENT_KINDS = ("runtime", "eval", "optimizer")


class ExperimentsConfig(BaseModel):
    """MLflow experiment paths — one set PER DOMAIN, under ``root``.

    Every optimized domain gets its own experiments at
    ``<root>/<domain>/<kind>`` (domain = ``eval.engine``), e.g.
    ``/Shared/forge/<domain>/eval``, so two agents' traces never mix. Leave a
    kind unset to use that path; an explicit path must still live under
    ``<root>/<domain>/`` or the config fails to load. Resolved by
    :func:`resolve_domain_experiments` when ``harness/config.yaml`` is parsed.
    """

    model_config = ConfigDict(extra="forbid")

    root: str = DEFAULT_EXPERIMENTS_ROOT
    runtime: str | None = None
    eval: str | None = None
    optimizer: str | None = None

    @field_validator("root")
    @classmethod
    def _absolute_root(cls, v: str) -> str:
        if not v.startswith("/") or v.rstrip("/") == "":
            raise ValueError(f"experiments.root must be an absolute workspace folder, got {v!r}")
        return v.rstrip("/")


def domain_experiment_folder(root: str, domain: str) -> str:
    """``<root>/<domain>`` — the folder holding one domain's experiments."""
    return f"{root.rstrip('/')}/{domain}"


def check_domain_experiment(path: str, *, root: str, domain: str, field: str) -> str:
    """Return ``path`` if it lives under this domain's folder, else raise ``ValueError``."""
    folder = domain_experiment_folder(root, domain)
    if not path.rstrip("/").startswith(folder + "/"):
        raise ValueError(
            f"{field} = {path!r} is outside domain {domain!r}'s experiment folder {folder}/. "
            f"Every optimized domain gets its own experiments: remove {field} to use "
            f"{folder}/<kind>, put it under {folder}/, or change experiments.root."
        )
    return path


def resolve_domain_experiments(cfg: ExperimentsConfig, domain: str) -> ExperimentsConfig:
    """Fill unset kinds with ``<root>/<domain>/<kind>``; validate explicit ones."""
    folder = domain_experiment_folder(cfg.root, domain)
    resolved: dict[str, str] = {}
    for kind in EXPERIMENT_KINDS:
        value = getattr(cfg, kind)
        resolved[kind] = (
            f"{folder}/{kind}"
            if value is None
            else check_domain_experiment(
                value, root=cfg.root, domain=domain, field=f"experiments.{kind}"
            )
        )
    return ExperimentsConfig(root=cfg.root, **resolved)


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


class OptimizerPersistenceConfig(BaseModel):
    """Schema of ``harness/config.yaml > persistence`` — durable optimizer
    artifact sink toggle.

    Controls whether an optimize session's per-round optimizer logs
    (transcript + critique) and run-level improvement summary are durably
    persisted to a parent MLflow run under ``experiments.optimizer``. All
    fields are optional so a config WITHOUT the ``persistence:`` section
    validates (backward compatible) — and ``enabled`` defaults to ``True``,
    so persistence is ON by default. Read by the loop/orchestrator plane
    (:func:`anvil.loop.optimizer_artifacts.resolve_persistence_settings`),
    NOT merged into :class:`HarnessConfig` — like ``optimizer`` backend
    selection, this is a loop/deployment concern.

    The ``ANVIL_PERSIST_OPTIMIZER_ARTIFACTS`` env var is authoritative over
    ``enabled`` (env wins when set), matching the ``ANVIL_OPTIMIZER_BACKEND``
    precedence for the optimizer backend.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Experiment to attach the parent run to. When None, falls back to
    # ``experiments.optimizer`` (then ``/Shared/anvil-optimizer``).
    experiment: str | None = None


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


# Assessment source types the trace engine understands. Mirrors
# ``mlflow.entities.assessment_source.AssessmentSourceType`` without
# importing mlflow here (this module must stay import-light — see the
# module docstring). ``HUMAN`` labels are the ground-truth signal;
# ``LLM_JUDGE``/``AI_JUDGE`` are re-runnable rubric judges. ``CODE`` is
# accepted so a config may opt into deterministic code assessments, but
# it is not in the default include set.
_KNOWN_ASSESSMENT_SOURCE_TYPES = frozenset(
    {"HUMAN", "LLM_JUDGE", "AI_JUDGE", "CODE"}
)


class TraceEvalConfig(BaseModel):
    """Configuration for the ``trace`` eval engine (Option A).

    The trace engine turns an agent's MLflow traces — which already carry
    LLM-judge and human assessments — into a frozen eval dataset. Each
    round RE-RUNS the mutated agent over the trace inputs and RE-SCORES
    against those assessments (replaying stored scores would give
    ``score_delta ≡ 0``, so the traces supply the dataset + scoring
    TARGETS, not the scores). See :mod:`anvil.domains.trace`.

    * ``experiment_id`` — the MLflow experiment to ingest (1 experiment ==
      1 agent). Only consulted by the one-time ingest CLI; the eval run
      itself reads the frozen snapshot, never the live server.
    * ``max_traces`` — cap on traces pulled during ingest.
    * ``snapshot_path`` — frozen JSONL the eval run reads (relative paths
      resolve against the repo root).
    * ``min_human_weight`` — weight applied to human objectives
      (``ref_*`` / ``human_*``) in the aggregate; judge objectives
      (``judge_*``) weigh 1.0, so humans are weighted strictly above
      judges (must be ``> 1.0``).
    * ``include_source_types`` — assessment source types kept during
      ingest; others are dropped.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str | None = None
    max_traces: int = 200
    snapshot_path: str = "data/trace_snapshot.jsonl"
    min_human_weight: float = 2.0
    include_source_types: list[str] = Field(
        default_factory=lambda: ["HUMAN", "LLM_JUDGE", "AI_JUDGE"]
    )

    @field_validator("max_traces")
    @classmethod
    def _max_traces_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("trace.max_traces must be > 0")
        return v

    @field_validator("min_human_weight")
    @classmethod
    def _min_human_weight_above_judges(cls, v: float) -> float:
        """Human objectives must weigh strictly above the judge weight (1.0).

        The whole point of DECISION #3 is that human labels dominate the
        aggregate; a value ``<= 1.0`` would silently equalize (or invert)
        that, so reject it loudly.
        """
        if not math.isfinite(v) or v <= 1.0:
            raise ValueError("trace.min_human_weight must be finite and > 1.0")
        return v

    @field_validator("include_source_types")
    @classmethod
    def _known_source_types(cls, v: list[str]) -> list[str]:
        """Reject an unknown source type so a typo fails at config-load."""
        unknown = [s for s in v if s not in _KNOWN_ASSESSMENT_SOURCE_TYPES]
        if unknown:
            raise ValueError(
                f"trace.include_source_types has unknown entries {unknown}; "
                f"valid values are {sorted(_KNOWN_ASSESSMENT_SOURCE_TYPES)}"
            )
        return v


class UserMetric(BaseModel):
    """A metric the user chose because the agent had no judges of its own."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""


class AgentEvalsConfig(BaseModel):
    """The agent's EXISTING evals, which forge reviews and re-uses.

    ``experiment_id`` is the MLflow experiment holding the agent's own
    traces, judges and human feedback (not forge's ``experiments.*``).
    ``scripts/review_agent_evals.py`` reads it into
    ``eval/agent_evals/{inventory.json,review.md}``; every round is then
    scored with those judges (``judges`` narrows them; empty = all).
    ``user_metrics`` records the metrics agreed with the user — required
    only when the agent has no judges forge can re-run.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str | None = None
    max_traces: int = Field(default=200, ge=1)
    judges: list[str] = Field(default_factory=list)
    user_metrics: list[UserMetric] = Field(default_factory=list)


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
    # Config for the ``trace`` eval engine (``engine: trace``). Ignored by
    # the genai and savesage engines. Optional so existing configs that
    # never set it validate unchanged.
    trace: TraceEvalConfig | None = None
    # The agent's own evals (judges + human feedback) that forge reviews
    # first and re-uses to score every round. Required by the baseline and
    # round entrypoints (``anvil.eval.agent_evals.ensure_agent_evals_ready``).
    agent_evals: AgentEvalsConfig | None = None
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
        # Imported lazily (not at module top) to break an import cycle:
        # ``anvil.eval.__init__`` → ``anvil.eval.cache`` imports ``ScorerConfig``
        # from this module, so a top-level ``anvil.eval.engines`` import here
        # makes the two packages' initialization order-dependent (and it fails
        # when this module is imported before ``anvil.eval`` is warm).
        from anvil.eval.engines import is_valid_engine_name

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
    # Current lever choices (name -> value). Each name must be declared in
    # ``harness/config.yaml > levers`` and each value must be in that
    # lever's ``allowed`` list; validated when the two files are merged.
    levers: dict[str, LeverValue] = Field(default_factory=dict)

    @field_validator("levers")
    @classmethod
    def _lever_names(cls, v: dict[str, LeverValue]) -> dict[str, LeverValue]:
        _validate_lever_names(v)
        return v


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
    # Per-domain MLflow experiments; unset kinds resolve to
    # <root>/<eval.engine>/<kind> (see ExperimentsConfig).
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    optimizer: OptimizerBackendConfig = Field(default_factory=OptimizerBackendConfig)
    # Durable optimizer-artifact persistence (per-round logs + run summary
    # to MLflow). Optional + enabled-by-default so existing configs stay
    # valid; a loop/orchestrator-plane concern, not merged into HarnessConfig.
    persistence: OptimizerPersistenceConfig = Field(default_factory=OptimizerPersistenceConfig)
    # Optimizer-selectable runtime levers (name -> allowlist). ``model`` is
    # understood by forge core and swaps the runtime model for every engine;
    # any other name is opaque to core and interpreted by the domain engine
    # (e.g. an input-format switch). Empty = no levers (backward compatible).
    # ``judge_endpoint`` is deliberately NOT a lever: the grader stays fixed
    # so scores remain comparable across rounds.
    levers: dict[str, LeverSpec] = Field(default_factory=dict)
    # Model price list source + synced file. Informational: prices are shown
    # to the optimizer and used for cost_usd metrics, never as a gate input.
    model_catalog: ModelCatalogConfig = Field(default_factory=ModelCatalogConfig)

    @field_validator("levers")
    @classmethod
    def _lever_names(cls, v: dict[str, LeverSpec]) -> dict[str, LeverSpec]:
        _validate_lever_names(v)
        return v

    @model_validator(mode="after")
    def _per_domain_experiments(self) -> RuntimeYAML:
        """Give this domain its own experiments (and keep persistence inside them)."""
        domain = self.eval.engine
        self.experiments = resolve_domain_experiments(self.experiments, domain)
        if self.persistence.experiment:
            check_domain_experiment(
                self.persistence.experiment,
                root=self.experiments.root,
                domain=domain,
                field="persistence.experiment",
            )
        return self


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
    # Declared lever allowlists (from harness/config.yaml).
    lever_specs: dict[str, LeverSpec] = Field(default_factory=dict)
    # Resolved lever values: the scaffold's choice, else the lever's
    # default. A lever with neither is absent here (the engine decides).
    levers: dict[str, LeverValue] = Field(default_factory=dict)
    model_catalog: ModelCatalogConfig = Field(default_factory=ModelCatalogConfig)

    @property
    def effective_runtime_model(self) -> str:
        """The model the runtime agent actually calls this round.

        The ``model`` lever when set, else the immutable
        ``runtime_endpoint``. ``runtime_endpoint`` itself stays the BASE
        model — it is what the baseline cache records.
        """
        chosen = self.levers.get(MODEL_LEVER)
        return str(chosen) if chosen is not None else self.runtime_endpoint

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
            lever_specs=dict(runtime.levers),
            levers=resolve_levers(runtime.levers, scaffold.levers),
            model_catalog=runtime.model_catalog,
        )


def resolve_levers(
    specs: dict[str, LeverSpec], chosen: dict[str, LeverValue]
) -> dict[str, LeverValue]:
    """Merge scaffold lever choices with their declared specs.

    Raises ``ValueError`` when the scaffold sets an undeclared lever or a
    value outside its allowlist — a hand-edited or stale scaffold fails
    loudly at load time instead of silently running an unlisted model.
    """
    unknown = sorted(set(chosen) - set(specs))
    if unknown:
        raise ValueError(
            f"scaffold/harness.yaml > levers sets undeclared lever(s) {unknown}; "
            f"declare them in harness/config.yaml > levers (declared: {sorted(specs)})"
        )
    resolved: dict[str, LeverValue] = {}
    for name, spec in specs.items():
        if name in chosen:
            value = chosen[name]
            if value not in spec.allowed:
                raise ValueError(
                    f"lever {name!r} = {value!r} is not in its allowlist {spec.allowed!r}"
                )
            resolved[name] = value
        elif spec.default is not None:
            resolved[name] = spec.default
    return resolved


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
        "optimizer",
        "persistence",
        "model_catalog",
    }
)
SCAFFOLD_FIELDS: frozenset[str] = frozenset({"sampling", "skills", "rules", "tools"})
