"""Offline tests for the pitcrew domain (no gateway, no workspace).

The gateway client and pitcrew's judges are fakes; the engine reads the
vendored corpus (or tiny placeholder PDFs) off disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from anvil.domains.pitcrew.agent_base import PitcrewAgent
from anvil.domains.pitcrew.eval import (
    GOLDEN_SET_REL,
    evaluate_pitcrew,
    extract_json,
    judge_io,
    load_pitcrew_golden_set,
)
from anvil.domains.pitcrew.summarizer import pdf_content_block, summarize_pdf
from anvil.eval.agent_evals import AgentJudge, JudgeSpec

REPO_ROOT = Path(__file__).resolve().parent.parent

_GOOD = {
    "document_title": "2210. Communications with the Public",
    "sections": [
        {
            "id": "a",
            "title": "Definitions",
            "level": 2,
            "summary": "Defines communications.",
            "citations": [{"page_numbers": [1], "source_text": "communications means"}],
        }
    ],
}


class _FakeClient:
    """Gateway-client stand-in: records each call, returns canned text + usage."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500),
        )


class _Scorer:
    def __init__(self, value):
        self.value = value
        self.calls: list[dict] = []

    def run(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(value=self.value, rationale="r", error=None)


def _judges(quality="yes", safety="yes"):
    return [
        AgentJudge(
            JudgeSpec("summary_quality", "registered_scorer", "guidelines"), _Scorer(quality)
        ),
        AgentJudge(JudgeSpec("safety", "registered_scorer", "builtin"), _Scorer(safety)),
    ]


@pytest.fixture
def tiny_golden(tmp_path: Path) -> Path:
    rows = []
    for i in range(8):
        pdf = tmp_path / f"rule_{i}.pdf"
        pdf.write_bytes(b"%PDF-1.4 placeholder")
        rows.append(
            {
                "example_id": f"finra_{i}",
                "query": f"Rule {i}",
                "category": "finra",
                "pdf_path": str(pdf),
            }
        )
    gs = tmp_path / "golden.jsonl"
    gs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return gs


def _config(tmp_path: Path, **overrides) -> Path:
    cfg = yaml.safe_load((REPO_ROOT / "harness" / "config.yaml").read_text())
    cfg.update(overrides)
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


# ------------------------------------------------------------------ pieces


def test_vendored_golden_set_covers_the_corpus() -> None:
    rows = load_pitcrew_golden_set(REPO_ROOT / GOLDEN_SET_REL)
    assert len(rows) == 20
    assert all((REPO_ROOT / r["pdf_path"]).is_file() for r in rows)
    assert {r["category"] for r in rows} == {"finra"}


def test_golden_set_requires_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"example_id": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing fields"):
        load_pitcrew_golden_set(bad)


def test_extract_json_tolerates_fences_and_prose() -> None:
    assert extract_json("```json\n" + json.dumps(_GOOD) + "\n```") == _GOOD
    assert extract_json("here: " + json.dumps(_GOOD) + " done") == _GOOD
    assert extract_json("not json") is None


def test_pdf_block_per_model_family() -> None:
    assert pdf_content_block(b"%PDF", "databricks-claude-haiku-4-5")["type"] == "document"
    block = pdf_content_block(b"%PDF", "databricks-gpt-6-luna")
    assert block["type"] == "file" and block["file"]["filename"] == "document.pdf"


def test_summarize_pdf_sends_production_shape_and_returns_usage() -> None:
    client = _FakeClient("{}")
    text, usage = summarize_pdf(client, "databricks-claude-sonnet-4-6", "sys", b"%PDF")
    assert (text, usage) == ("{}", {"input_tokens": 1000, "output_tokens": 500})
    call = client.calls[0]
    assert call["max_tokens"] == 32000 and call["messages"][0] == {
        "role": "system",
        "content": "sys",
    }
    assert call["messages"][1]["content"][0]["type"] == "document"


def test_judge_io_matches_production_trace_shape() -> None:
    io = judge_io(b"%PDF-1.4", "2210.pdf", _GOOD, json.dumps(_GOOD))
    assert io["inputs"] == {
        "pdf_bytes": "b'%PDF-1.4'",
        "filename": "2210.pdf",
        "prompt_version": None,
    }
    assert io["outputs"] == [_GOOD]
    assert judge_io(b"x", "f.pdf", None, "garbage")["outputs"] == "garbage"


def test_active_agent_passes_code_validation() -> None:
    from anvil.optimizer.code_validation import validate_code_candidate

    validate_code_candidate(REPO_ROOT / "agents" / "pitcrew_agent.py")


def test_agent_rejects_models_outside_the_allowlist() -> None:
    class _A(PitcrewAgent):
        def predict(self, *, pdf_bytes):
            return self.summarize(pdf_bytes)

    agent = _A("p", model="databricks-claude-sonnet-4-6", client=_FakeClient("{}"))
    with pytest.raises(ValueError, match="not in allowed_models"):
        agent.summarize(b"%PDF", model="databricks-claude-opus-4-7")


# ------------------------------------------------------------------ engine


def test_engine_runs_active_agent_scores_with_agent_judges(
    tiny_golden: Path, tmp_path: Path
) -> None:
    client = _FakeClient(json.dumps(_GOOD))
    judges = _judges(quality="yes", safety="yes")
    report = evaluate_pitcrew(
        scaffold_root=REPO_ROOT / "scaffold",
        runtime_config_path=_config(tmp_path),
        golden_set_path=tiny_golden,
        mode="quick",
        runtime_client=client,
        agent_judges=judges,
    )
    assert report.n_rows == 8 and len(client.calls) == 8
    assert {c["model"] for c in client.calls} == {"databricks-claude-sonnet-4-6"}
    assert report.per_judge == {"summary_quality": 1.0, "safety": 1.0}
    assert report.aggregate == pytest.approx(1.0)
    assert report.scorer_fingerprint.startswith("agent_evals:")
    cm = report.cost_metrics
    assert cm["n_latency"] == 8.0 and cm["latency_ms_p90"] >= cm["latency_ms_median"] >= 0.0
    # Sonnet 4.6: 1000 in @ ~$3/M + 500 out @ ~$15/M per PDF.
    assert cm["cost_usd_per_row"] == pytest.approx(0.0105, rel=1e-3)
    seen = judges[0].scorer.calls[0]
    assert set(seen["inputs"]) == {"pdf_bytes", "filename", "prompt_version"}
    assert seen["outputs"] == [_GOOD]


def test_engine_aggregate_is_mean_of_agent_judges(tiny_golden: Path, tmp_path: Path) -> None:
    report = evaluate_pitcrew(
        scaffold_root=REPO_ROOT / "scaffold",
        runtime_config_path=_config(tmp_path),
        golden_set_path=tiny_golden,
        mode="quick",
        predict_fn=lambda pdf_bytes: json.dumps(_GOOD),  # noqa: ARG005
        agent_judges=_judges(quality="no", safety="yes"),
    )
    assert report.aggregate == pytest.approx(0.5)
    assert len(report.failures) == 8
    assert report.failures[0]["judge_failures"] == ["summary_quality"]


def test_agent_can_route_to_an_allowed_gpt_model(tiny_golden: Path, tmp_path: Path) -> None:
    agent = tmp_path / "luna_agent.py"
    agent.write_text(
        "from anvil.domains.pitcrew.agent_base import PitcrewAgent\n"
        "class LunaAgent(PitcrewAgent):\n"
        "    def predict(self, *, pdf_bytes):\n"
        "        return self.summarize(pdf_bytes, model='databricks-gpt-6-luna', max_tokens=8000)\n"
    )
    client = _FakeClient(json.dumps(_GOOD))
    report = evaluate_pitcrew(
        scaffold_root=REPO_ROOT / "scaffold",
        runtime_config_path=_config(tmp_path, agent_module=str(agent)),
        golden_set_path=tiny_golden,
        mode="quick",
        runtime_client=client,
        agent_judges=_judges(),
    )
    assert {c["model"] for c in client.calls} == {"databricks-gpt-6-luna"}
    assert {c["max_tokens"] for c in client.calls} == {8000}
    assert client.calls[0]["messages"][1]["content"][0]["type"] == "file"
    assert report.cost_metrics["cost_usd_total"] > 0.0


def test_failed_rows_are_still_timed_and_fail_the_judges(tiny_golden: Path, tmp_path: Path) -> None:
    def predict_fn(pdf_bytes: bytes) -> str:  # noqa: ARG001
        predict_fn.n = getattr(predict_fn, "n", 0) + 1
        if predict_fn.n == 1:
            raise RuntimeError("gateway down")
        return json.dumps(_GOOD)

    report = evaluate_pitcrew(
        scaffold_root=REPO_ROOT / "scaffold",
        runtime_config_path=_config(tmp_path),
        golden_set_path=tiny_golden,
        mode="quick",
        predict_fn=predict_fn,
        agent_judges=_judges(),
    )
    assert report.cost_metrics["n_latency"] == 8.0
    # Bare-text predictors carry no usage, so cost is omitted rather than understated.
    assert "cost_usd_total" not in report.cost_metrics


def test_engine_refuses_to_run_without_agent_judges(tiny_golden: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="own judges"):
        evaluate_pitcrew(
            scaffold_root=REPO_ROOT / "scaffold",
            runtime_config_path=_config(tmp_path),
            golden_set_path=tiny_golden,
            mode="quick",
            predict_fn=lambda b: "x",  # noqa: ARG005
            agent_judges=[],
        )


def test_shipped_config_is_the_pitcrew_instance() -> None:
    from anvil.runtime.loader import load_harness

    cfg = load_harness(REPO_ROOT / "scaffold").config
    assert (cfg.mode, cfg.agent_module, cfg.eval.engine) == (
        "code",
        "agents/pitcrew_agent.py",
        "pitcrew",
    )
    assert cfg.eval.agent_evals.judges == ["summary_quality", "safety"]
    assert cfg.experiments.eval == "/Shared/forge-v3/pitcrew/eval"


@pytest.mark.parametrize(
    "generic", ["data/golden_set.jsonl", str(REPO_ROOT / "data" / "golden_set.jsonl")]
)
def test_generic_golden_set_path_resolves_to_pitcrews_own(generic, tmp_path: Path) -> None:
    """make_baseline passes the generic path relative, run_round passes it absolute."""
    seen: list[bytes] = []

    def predict_fn(pdf_bytes: bytes) -> str:
        seen.append(pdf_bytes)
        return json.dumps(_GOOD)

    report = evaluate_pitcrew(
        scaffold_root=REPO_ROOT / "scaffold",
        runtime_config_path=_config(tmp_path),
        golden_set_path=generic,
        mode="quick",
        predict_fn=predict_fn,
        agent_judges=_judges(),
    )
    assert report.n_rows == 8
    assert all(b.startswith(b"%PDF") for b in seen)
