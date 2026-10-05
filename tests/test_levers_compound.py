"""Tests for runtime levers, compound mutations, and the model price list.

Offline: no gateway, no Google, no workspace. Uses a throwaway repo layout
(``scaffold/`` + ``harness/config.yaml``) under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from anvil.catalog import (
    CatalogEntry,
    cost_usd,
    endpoint_slug,
    expand_family,
    load_catalog,
    lookup_price,
    model_prices,
    parse_sheet_rows,
    write_catalog_csv,
)
from anvil.loop.builder import build_round_prompt
from anvil.optimizer.actions import (
    CompoundAction,
    EditSkillAction,
    OptimizerAction,
    SetLeverAction,
)
from anvil.optimizer.applier import ApplyError, apply_action
from anvil.runtime.client import _drop_rejected_params
from anvil.runtime.loader import load_harness
from anvil.runtime.models import LeverSpec, resolve_levers

_adapter = TypeAdapter(OptimizerAction)

_CONFIG = """\
runtime_endpoint: databricks-claude-sonnet-4-6
optimizer_endpoint: databricks-claude-opus-4-7
judge_endpoint: databricks-claude-sonnet-4-6
experiments: {root: /Shared/forge}
loop:
  max_mutations_per_round: %(max_mut)d
levers:
  model:
    allowed: [databricks-claude-sonnet-4-6, databricks-claude-haiku-4-5]
  max_chunk:
    allowed: [4000, 8000]
    default: 8000
"""


def _repo(tmp_path: Path, *, max_mut: int = 3) -> Path:
    """Minimal repo: scaffold (identity + one extra skill) + immutable config."""
    root = tmp_path / "scaffold"
    (root / "skills").mkdir(parents=True)
    (root / "rules").mkdir()
    (root / "skills" / "identity.md").write_text(
        "---\nskill_id: identity\nkind: identity\napplies_to: runtime\n---\n\n# role\nagent\n",
        encoding="utf-8",
    )
    (root / "skills" / "style.md").write_text(
        "---\nskill_id: style\nkind: domain\n---\n\n# style\nv1\n", encoding="utf-8"
    )
    (root / "harness.yaml").write_text(
        dedent(
            """\
            skills:
              - file: identity.md
              - file: style.md
            rules: []
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "config.yaml").write_text(
        _CONFIG % {"max_mut": max_mut}, encoding="utf-8"
    )
    return root


def _harness(root: Path) -> dict:
    return yaml.safe_load((root / "harness.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------- models


def test_resolve_levers_defaults_and_choices() -> None:
    specs = {
        "model": LeverSpec(allowed=["a", "b"]),
        "mode": LeverSpec(allowed=["x", "y"], default="x"),
    }
    assert resolve_levers(specs, {}) == {"mode": "x"}  # model: no default -> engine decides
    assert resolve_levers(specs, {"model": "b"}) == {"model": "b", "mode": "x"}


def test_resolve_levers_rejects_unknown_and_disallowed() -> None:
    specs = {"model": LeverSpec(allowed=["a"])}
    with pytest.raises(ValueError, match="undeclared"):
        resolve_levers(specs, {"temperature_hack": 1})
    with pytest.raises(ValueError, match="allowlist"):
        resolve_levers(specs, {"model": "z"})


def test_lever_default_must_be_allowed() -> None:
    with pytest.raises(ValidationError):
        LeverSpec(allowed=["a"], default="b")


def test_effective_runtime_model_follows_lever(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    snap = load_harness(root)
    assert snap.runtime_endpoint == "databricks-claude-sonnet-4-6"  # base model
    h = _harness(root)
    h["levers"] = {"model": "databricks-claude-haiku-4-5"}
    (root / "harness.yaml").write_text(yaml.safe_dump(h), encoding="utf-8")
    snap = load_harness(root)
    assert snap.runtime_endpoint == "databricks-claude-haiku-4-5"
    assert snap.config.effective_runtime_model == "databricks-claude-haiku-4-5"
    # The base stays the immutable config value (what the baseline records).
    assert snap.config.runtime_endpoint == "databricks-claude-sonnet-4-6"
    assert snap.config.levers["max_chunk"] == 8000


# -------------------------------------------------------------------- actions


def test_compound_schema() -> None:
    edit = {
        "action": "edit_skill",
        "target_file": "skills/style.md",
        "content": "x",
        "rationale": "r",
    }
    lever = {"action": "set_lever", "name": "model", "value": "m", "rationale": "r"}
    ok = _adapter.validate_python(
        {"action": "compound", "synergy": "s", "rationale": "r", "steps": [edit, lever]}
    )
    assert isinstance(ok, CompoundAction) and len(ok.steps) == 2
    for bad_steps in ([edit], [edit, {"action": "noop", "rationale": "r"}]):
        with pytest.raises(ValidationError):
            _adapter.validate_python(
                {"action": "compound", "synergy": "s", "rationale": "r", "steps": bad_steps}
            )
    with pytest.raises(ValidationError):  # synergy is mandatory
        _adapter.validate_python({"action": "compound", "rationale": "r", "steps": [edit, lever]})


# -------------------------------------------------------------------- applier


def test_set_lever_writes_canonical_value(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    apply_action(SetLeverAction(name="max_chunk", value="4000", rationale="r"), root)
    assert _harness(root)["levers"] == {"max_chunk": 4000}  # canonical int, not "4000"
    load_harness(root)  # scaffold still validates


def test_set_lever_rejects_undeclared_and_disallowed(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(ApplyError, match="not declared"):
        apply_action(SetLeverAction(name="judge", value="x", rationale="r"), root)
    with pytest.raises(ApplyError, match="not in allowed"):
        apply_action(SetLeverAction(name="model", value="databricks-gpt-6", rationale="r"), root)
    assert "levers" not in _harness(root)


def _compound(*steps) -> CompoundAction:
    return CompoundAction(synergy="combine", rationale="r", steps=list(steps))


def test_compound_applies_all_steps(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    res = apply_action(
        _compound(
            EditSkillAction(target_file="skills/style.md", content="# style\nv2\n", rationale="r"),
            SetLeverAction(name="model", value="databricks-claude-haiku-4-5", rationale="r"),
        ),
        root,
    )
    assert "v2" in (root / "skills" / "style.md").read_text()
    assert _harness(root)["levers"]["model"] == "databricks-claude-haiku-4-5"
    assert res.action_summary.startswith("compound[2]")
    assert set(res.files_changed) == {"scaffold/skills/style.md", "scaffold/harness.yaml"}


def test_compound_over_cap_is_rejected_before_writing(tmp_path: Path) -> None:
    root = _repo(tmp_path, max_mut=1)
    before = (root / "skills" / "style.md").read_text()
    with pytest.raises(ApplyError, match="max_mutations_per_round"):
        apply_action(
            _compound(
                EditSkillAction(target_file="skills/style.md", content="v2", rationale="r"),
                SetLeverAction(name="model", value="databricks-claude-haiku-4-5", rationale="r"),
            ),
            root,
        )
    assert (root / "skills" / "style.md").read_text() == before


def test_compound_rejects_duplicate_targets(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(ApplyError, match="more than once"):
        apply_action(
            _compound(
                SetLeverAction(name="model", value="databricks-claude-haiku-4-5", rationale="r"),
                SetLeverAction(name="model", value="databricks-claude-sonnet-4-6", rationale="r"),
            ),
            root,
        )


def test_compound_is_atomic(tmp_path: Path) -> None:
    """A failing later step rolls back the earlier step's write."""
    root = _repo(tmp_path)
    before = (root / "skills" / "style.md").read_text()
    with pytest.raises(ApplyError):
        apply_action(
            _compound(
                EditSkillAction(target_file="skills/style.md", content="v2", rationale="r"),
                EditSkillAction(target_file="skills/missing.md", content="x", rationale="r"),
            ),
            root,
        )
    assert (root / "skills" / "style.md").read_text() == before


# -------------------------------------------------------------------- builder


def test_builder_shows_levers_prices_history_and_budget(tmp_path: Path) -> None:
    runs = tmp_path / "eval" / "runs"
    runs.mkdir(parents=True)
    (runs / "round_001.json").write_text(
        json.dumps(
            {
                "decision": "revert",
                "aggregate": 0.95,
                "levers": {"model": "databricks-claude-haiku-4-5"},
                "cost_metrics": {"latency_ms_median": 21000.0},
            }
        )
    )
    prompt = build_round_prompt(
        repo_root=tmp_path,
        round_id=2,
        baseline=None,
        max_mutations=3,
        max_turns=45,
        lever_specs={
            "model": LeverSpec(
                allowed=["databricks-claude-sonnet-4-6", "databricks-claude-haiku-4-5"]
            )
        },
        lever_values={"model": "databricks-claude-sonnet-4-6"},
        model_prices={
            "databricks-claude-haiku-4-5": {"input_per_million": 1.0, "output_per_million": 5.0}
        },
    )
    assert "`compound` of up to 3" in prompt
    assert "at most 45 turns" in prompt
    assert "$1 in / $5 out per 1M" in prompt
    assert "median latency 21000ms" in prompt
    assert "price unknown" in prompt  # sonnet-4-6 not in the supplied prices
    single = build_round_prompt(repo_root=tmp_path, round_id=2, baseline=None)
    assert "propose ONE structural mutation\n" in single and "compound" not in single


# -------------------------------------------------------------------- catalog


def test_expand_family() -> None:
    assert expand_family("Claude Opus 4.5 / 4.6") == ("claude-opus-4-5", "claude-opus-4-6")
    assert expand_family("GPT-5.4 Pro / 5.5 Pro") == ("gpt-5-4-pro", "gpt-5-5-pro")
    assert expand_family("Gemini 3.0 / 3.1 Pro") == (
        "gemini-3-0-pro",
        "gemini-3-pro",
        "gemini-3-1-pro",
    )
    assert expand_family("GTE (embedding)") == ("gte",)
    assert endpoint_slug("databricks-claude-sonnet-4-6") == "claude-sonnet-4-6"


def _entries() -> list[CatalogEntry]:
    return [
        CatalogEntry("Claude Sonnet 4", "All lengths", 3.0, 15.0),
        CatalogEntry("Claude Sonnet 4.5 / 4.6", "All lengths", 3.0, 15.0),
        CatalogEntry("GPT-5.4", "Long (>200k)", 5.0, 22.5),
        CatalogEntry("GPT-5.4", "Short (<=200k)", 2.5, 15.0),
        CatalogEntry("Llama 3.3 70B", "All lengths", 0.53, 1.59),
        CatalogEntry("DeepSeek V4 Flash", "All lengths", 0.15, 0.3),
    ]


def test_lookup_price_matching_rules() -> None:
    e = _entries()
    assert lookup_price("databricks-claude-sonnet-4-6", e).family == "Claude Sonnet 4.5 / 4.6"
    assert lookup_price("databricks-claude-sonnet-4", e).family == "Claude Sonnet 4"
    assert lookup_price("databricks-gpt-5-4", e).context.startswith("Short")
    assert lookup_price("databricks-gpt-5-4", e, tier="long").context.startswith("Long")
    assert lookup_price("databricks-meta-llama-3-3-70b-instruct", e).family == "Llama 3.3 70B"
    assert lookup_price("databricks-deepseek-v4-flash-0731", e).family == "DeepSeek V4 Flash"
    # A version suffix is a different model, not a substring match.
    assert lookup_price("databricks-claude-sonnet-4-9", e) is None


def test_parse_sheet_rows_and_csv_roundtrip(tmp_path: Path) -> None:
    values = [
        ["", "Title row"],
        [
            "",
            "Model",
            "Price/Input Token ($/1M)",
            "Price/Output Token ($/1M)",
            "Context",
            "Cache read ($/1M)",
        ],
        ["", "Claude Haiku 4.5", 1.00002, 5.00003, "All lengths", 0.1],
        ["", "GTE (embedding)", 0.137, "n/a", "All lengths", "n/a"],
        ["", "Notes", "free text"],
    ]
    entries = parse_sheet_rows(values)
    assert [x.family for x in entries] == ["Claude Haiku 4.5", "GTE (embedding)"]
    assert entries[1].output_per_million is None
    out = write_catalog_csv(tmp_path / "c.csv", entries, header_note="test")
    assert load_catalog(out) == entries
    with pytest.raises(ValueError, match="header"):
        parse_sheet_rows([["no", "header"]])


def test_model_prices_and_cost() -> None:
    prices = model_prices(["databricks-claude-sonnet-4-6", "databricks-unknown-x"], _entries())
    assert prices["databricks-claude-sonnet-4-6"]["source"] == "sheet"
    assert "databricks-unknown-x" not in prices
    assert cost_usd(prices["databricks-claude-sonnet-4-6"], 10_000, 2_000) == pytest.approx(0.06)
    assert cost_usd(None, 1, 1) is None


# --------------------------------------------------------------------- client


def test_drop_rejected_params() -> None:
    exc = Exception("BAD_REQUEST: Model x does not support the temperature parameter.")
    assert _drop_rejected_params(exc, {"temperature": 0, "max_tokens": 5}) == {"max_tokens": 5}
    assert _drop_rejected_params(Exception("rate limited"), {"temperature": 0}) is None
    assert _drop_rejected_params(exc, {"max_tokens": 5}) is None  # nothing we sent


def test_drop_rejected_params_openai_unsupported_value() -> None:
    # GPT reasoning models reject temperature=0 with OpenAI's "Unsupported
    # value" wording; the live SDK error is a dict repr with escaped quotes.
    live = Exception(
        "{'message': 'Unsupported value: \\'temperature\\' does not support 0.0 with this model.'}"
    )
    assert _drop_rejected_params(live, {"temperature": 0.0, "max_tokens": 5}) == {"max_tokens": 5}
    plain = Exception("Unsupported value: 'temperature' does not support 0.0 with this model.")
    assert _drop_rejected_params(plain, {"temperature": 0.0}) == {}
    other = Exception("Unsupported value: 'top_k' does not support 0.0 with this model.")
    assert _drop_rejected_params(other, {"temperature": 0.0}) is None
