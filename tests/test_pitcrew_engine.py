"""Offline tests for the pitcrew domain (no gateway, no workspace).

Covers the pure scoring logic and the engine's map/aggregate wiring via
injected ``predict_fn`` / ``judge_fn`` — the same injection seam
``evaluate_trace`` / ``evaluate_savesage`` expose. The engine reads each
row's ``pdf_path`` off disk, so the fixture writes tiny placeholder files;
``predict_fn`` ignores the bytes and returns a canned summary string.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.domains.pitcrew.eval import evaluate_pitcrew, load_pitcrew_golden_set
from anvil.domains.pitcrew.scoring import (
    extract_json,
    score_json_valid,
    score_schema_adherence,
)

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


def test_extract_json_strips_fences() -> None:
    assert extract_json("```json\n" + json.dumps(_GOOD) + "\n```") == _GOOD
    assert extract_json("prose " + json.dumps(_GOOD) + " trailing") == _GOOD
    assert extract_json("not json at all") is None


def test_score_json_valid() -> None:
    assert score_json_valid(_GOOD) == 1.0
    assert score_json_valid({"document_title": "x"}) == 0.0  # no sections
    assert score_json_valid({"sections": []}) == 0.0  # no title
    assert score_json_valid(None) == 0.0


def test_schema_adherence_fraction() -> None:
    assert score_schema_adherence(_GOOD) == 1.0
    half = {
        "document_title": "t",
        "sections": [
            _GOOD["sections"][0],
            {"id": "b", "title": "T", "level": 9, "summary": "s", "citations": []},
        ],
    }
    assert score_schema_adherence(half) == 0.5  # second section: bad level + empty citations
    assert score_schema_adherence({"document_title": "t", "sections": []}) == 0.0


@pytest.fixture
def tmp_golden(tmp_path: Path) -> Path:
    """Write 8 placeholder PDFs + a golden set (quick mode needs 8 finra rows)."""
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
    gs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return gs


def test_load_golden_set_requires_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"example_id": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing fields"):
        load_pitcrew_golden_set(bad)


def test_engine_aggregates_injected_scores(tmp_golden: Path) -> None:
    """Half the rows return valid JSON (quality 1.0), half return junk (0.0)."""
    good_text = json.dumps(_GOOD)

    def predict_fn(pdf_bytes: bytes) -> str:  # noqa: ARG001
        # Alternate good/bad by a counter stashed on the function.
        predict_fn.n = getattr(predict_fn, "n", 0) + 1
        return good_text if predict_fn.n % 2 == 1 else "garbage not json"

    def judge_fn(doc_block, summary_text) -> float:  # noqa: ARG001
        return 1.0  # only called for parseable rows (gated behind json_valid)

    report = evaluate_pitcrew(
        scaffold_root=REPO_ROOT / "scaffold",
        runtime_config_path=REPO_ROOT / "harness" / "config.yaml",
        golden_set_path=tmp_golden,
        mode="quick",
        predict_fn=predict_fn,
        judge_fn=judge_fn,
    )
    assert report.n_rows == 8
    assert set(report.per_judge) == {"output_json_valid", "schema_adherence", "summary_quality"}
    # 4 of 8 parse → each objective macro-averages to 0.5.
    assert report.per_judge["output_json_valid"] == pytest.approx(0.5)
    assert report.per_judge["summary_quality"] == pytest.approx(0.5)
    assert 0.0 < report.aggregate < 1.0
    # Junk rows surface as failures.
    assert len(report.failures) == 4
