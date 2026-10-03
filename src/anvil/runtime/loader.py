"""Load harness/config.yaml + scaffold/harness.yaml into a snapshot.

The loader is intentionally thin: parse both YAMLs, validate against
their Pydantic schemas, build a merged ``HarnessConfig``, then call
the rigorous :func:`anvil.runtime.composer.compose_prompt` to produce
the runtime system prompt. The composer is what enforces
``applies_to`` filtering and identity-skill validation — this loader
just wires the inputs.

Memory loading (per-round critiques) is **not** the runtime's
concern; it lives on the optimizer side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from anvil.runtime.composer import ComposeManifest, compose_prompt
from anvil.runtime.models import (
    RUNTIME_FIELDS,
    SCAFFOLD_FIELDS,
    HarnessConfig,
    RuntimeYAML,
    SamplingConfig,
    ScaffoldYAML,
    ToolRef,
)


@dataclass(frozen=True)
class HarnessSnapshot:
    """Everything the runtime agent needs for a turn, ready to consume."""

    system_prompt: str
    sampling: SamplingConfig
    tools: list[ToolRef]
    runtime_endpoint: str
    judge_endpoint: str
    config: HarnessConfig
    manifest: ComposeManifest = field(default=None)  # type: ignore[assignment]


def default_runtime_config_path(scaffold_root: Path | str) -> Path:
    """Convention: ``<repo>/scaffold/`` and ``<repo>/harness/config.yaml``
    are siblings."""
    return Path(scaffold_root).parent / "harness" / "config.yaml"


def load_harness(
    scaffold_root: Path | str,
    runtime_config_path: Path | str | None = None,
) -> HarnessSnapshot:
    """Load both YAML files, validate, and compose the runtime prompt."""
    scaffold_dir = Path(scaffold_root)
    runtime_path = (
        Path(runtime_config_path)
        if runtime_config_path is not None
        else default_runtime_config_path(scaffold_dir)
    )

    scaffold = _load_scaffold_yaml(scaffold_dir / "harness.yaml")
    runtime = _load_runtime_yaml(runtime_path)
    config = HarnessConfig.from_split(scaffold, runtime)

    composed = compose_prompt(scaffold_dir, audience="runtime")

    return HarnessSnapshot(
        system_prompt=composed.text,
        sampling=config.sampling,
        tools=list(config.tools),
        # The model the runtime agent calls: the ``model`` lever when the
        # optimizer set one, else the immutable base ``runtime_endpoint``.
        runtime_endpoint=config.effective_runtime_model,
        judge_endpoint=config.judge_endpoint,
        config=config,
        manifest=composed.manifest,
    )


def _load_scaffold_yaml(path: Path) -> ScaffoldYAML:
    if not path.is_file():
        raise FileNotFoundError(f"scaffold harness.yaml not found at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return ScaffoldYAML.model_validate(raw)
    except ValidationError as exc:
        _reraise_with_misplaced_field_hint(
            exc,
            file_path=path,
            wrong_side_fields=RUNTIME_FIELDS,
            wrong_side_filename="harness/config.yaml",
            this_filename="scaffold/harness.yaml",
        )
        raise


def _load_runtime_yaml(path: Path) -> RuntimeYAML:
    if not path.is_file():
        raise FileNotFoundError(f"runtime config.yaml not found at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return RuntimeYAML.model_validate(raw)
    except ValidationError as exc:
        _reraise_with_misplaced_field_hint(
            exc,
            file_path=path,
            wrong_side_fields=SCAFFOLD_FIELDS,
            wrong_side_filename="scaffold/harness.yaml",
            this_filename="harness/config.yaml",
        )
        raise


def _reraise_with_misplaced_field_hint(
    exc: ValidationError,
    *,
    file_path: Path,
    wrong_side_fields: frozenset[str],
    wrong_side_filename: str,
    this_filename: str,
) -> None:
    for err in exc.errors():
        if err.get("type") != "extra_forbidden":
            continue
        loc = err.get("loc", ())
        if not loc:
            continue
        field_name = loc[0]
        if field_name in wrong_side_fields:
            raise ValueError(
                f"{this_filename} contains '{field_name}', which belongs in "
                f"{wrong_side_filename}. The harness config is split: "
                f"mutable knobs (sampling/skills/rules/tools) live in "
                f"scaffold/harness.yaml; immutable runtime identity "
                f"(endpoints/experiments/eval/loop) lives in harness/config.yaml. "
                f"File parsed: {file_path}"
            ) from exc
