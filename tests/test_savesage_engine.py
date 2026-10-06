"""Offline tests for the savesage domain's model / input-mode levers.

No gateway, no workspace: the extractor is a fake. Scoring reuses the
production field judge from the statement-agent tree, so the engine tests
skip when that tree is not on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anvil.domains.savesage import agent_base
from anvil.domains.savesage.agent_base import NATIVE_PDF_MODELS, SavesageAgent

REPO_ROOT = Path(__file__).resolve().parent.parent
LUNA = "databricks-gpt-5-6-luna"
GLM = "databricks-glm-5-3-flash"


def _statement_agent_available() -> bool:
    try:
        from anvil.domains.savesage._statement_agent import ensure_importable

        ensure_importable()
        import judge.matching  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


needs_statement_agent = pytest.mark.skipif(
    not _statement_agent_available(), reason="statement-agent tree not available"
)


class _FakeExtractor:
    """SavesageIciciExtractor stand-in: records its knobs, returns canned output."""

    built: list[dict] = []

    def __init__(self, _prompt, **kwargs):
        type(self).built.append(kwargs)
        self.kwargs = kwargs

    def extract_with_usage(self, *, sid, pdf_path):
        from anvil.domains.savesage.extractor import LEDGER

        usage = {"input_tokens": 1000, "output_tokens": 500}
        info = {k: self.kwargs.get(k) for k in ("model", "input_mode")}
        with LEDGER.call(sid, str(pdf_path).encode(), info) as record:
            record.update(usage)
        return {"sid": sid}, 12.5, usage


@pytest.fixture
def fake_extractor(monkeypatch: pytest.MonkeyPatch):
    import anvil.domains.savesage.extractor as extractor_mod

    _FakeExtractor.built = []
    monkeypatch.setattr(extractor_mod, "SavesageIciciExtractor", _FakeExtractor)
    return _FakeExtractor


class _Agent(SavesageAgent):
    def predict(self, *, sid, pdf_path):
        return self.extract(sid=sid, pdf_path=pdf_path)


# ------------------------------------------------------------------ agent base


def test_extract_uses_round_levers_and_reports_the_call(fake_extractor) -> None:
    agent = _Agent("p", model=GLM, input_mode="text", allowed_models=[LUNA, GLM])
    parsed, meta = agent.predict(sid="s1", pdf_path="x.pdf")
    assert parsed == {"sid": "s1"}
    assert fake_extractor.built[0]["model"] == GLM
    assert fake_extractor.built[0]["input_mode"] == "text"
    assert meta["latency_ms"] == 12.5
    (call,) = meta["calls"]
    assert call["model"] == GLM and call["input_tokens"] == 1000 and call["output_tokens"] == 500


def test_extract_reuses_one_extractor_per_combo(fake_extractor) -> None:
    agent = _Agent("p", allowed_models=[LUNA])
    for sid in ("a", "b", "c"):
        agent.predict(sid=sid, pdf_path="x.pdf")
    assert len(fake_extractor.built) == 1


def test_extract_rejects_models_outside_the_allowlist(fake_extractor) -> None:  # noqa: ARG001
    agent = _Agent("p", allowed_models=[LUNA])
    with pytest.raises(ValueError, match="not in allowed_models"):
        agent.extract(sid="s", pdf_path="x.pdf", model=GLM, input_mode="text")


def test_text_only_models_cannot_take_native_pdf(fake_extractor) -> None:  # noqa: ARG001
    assert GLM not in NATIVE_PDF_MODELS and LUNA in NATIVE_PDF_MODELS
    agent = _Agent("p", allowed_models=[LUNA, GLM])
    with pytest.raises(ValueError, match="cannot read native PDFs"):
        agent.extract(sid="s", pdf_path="x.pdf", model=GLM, input_mode="pdf")


def test_merge_meta_sums_latency_and_keeps_every_call(fake_extractor) -> None:  # noqa: ARG001
    agent = _Agent("p", allowed_models=[LUNA])
    _, m1 = agent.extract(sid="a", pdf_path="x.pdf")
    _, m2 = agent.extract(sid="a", pdf_path="x.pdf")
    merged = agent.merge_meta(m1, m2)
    assert merged["latency_ms"] == 25.0 and len(merged["calls"]) == 2


def test_run_luna_back_compat_is_native_pdf_luna(fake_extractor) -> None:
    parsed, latency = _Agent("p").run_luna(sid="s", pdf_path="x.pdf")
    assert (parsed, latency) == ({"sid": "s"}, 12.5)
    assert fake_extractor.built[0]["model"] == LUNA
    assert fake_extractor.built[0]["input_mode"] == "pdf"


def test_active_agent_passes_code_validation() -> None:
    from anvil.optimizer.code_validation import validate_code_candidate

    validate_code_candidate(REPO_ROOT / "agents" / "savesage_agent.py")


def test_active_agent_extracts_through_the_base(fake_extractor) -> None:
    """Whatever the optimizer last kept, the active agent loads and extracts via the base."""
    from anvil.domains.savesage.eval import _load_savesage_agent

    agent = _load_savesage_agent(
        "agents/savesage_agent.py", composed_prompt="p", repo_root=REPO_ROOT
    )
    agent.predict(sid="s", pdf_path="x.pdf")
    assert fake_extractor.built, "predict() made no extraction"
    assert agent_base.DEFAULT_REASONING_EFFORT == "medium"
    assert agent_base.DEFAULT_MAX_TOKENS == 96_000


# ------------------------------------------------------------------ text mode


@needs_statement_agent
def test_text_mode_swaps_only_the_pdf_block(monkeypatch: pytest.MonkeyPatch) -> None:
    from anvil.domains.savesage import extractor as extractor_mod

    monkeypatch.setattr(extractor_mod, "pdf_to_text", lambda pdf: f"TEXT<{len(pdf)}>")
    request = type(
        "Req", (), {"pdf": b"%PDF-1.4", "filename": "s.pdf", "bank": None, "request_id": "r"}
    )()

    def payload(**kwargs):
        adapter = extractor_mod._build_adapter("PROMPT", lambda: "tok", **kwargs)
        req = adapter._build_request(request, "PROMPT", {"type": "object"})
        return req.full_url, json.loads(req.data)

    url, pdf_body = payload(model=LUNA, input_mode="pdf")
    assert url.endswith(f"/serving-endpoints/{LUNA}/invocations")
    assert pdf_body["messages"][0]["content"][0]["type"] == "file"

    url, text_body = payload(model=GLM, input_mode="text", reasoning_effort=None)
    assert url.endswith(f"/serving-endpoints/{GLM}/invocations")
    content = text_body["messages"][0]["content"]
    assert content[0] == {
        "type": "text",
        "text": "Statement text (extracted from the PDF):\n\nTEXT<8>",
    }
    assert content[1] == {"type": "text", "text": "PROMPT"}
    assert "reasoning_effort" not in text_body
    assert text_body["response_format"] == pdf_body["response_format"]


def test_unknown_input_mode_is_rejected() -> None:
    from anvil.domains.savesage.extractor import _build_adapter

    with pytest.raises(ValueError, match="input_mode"):
        _build_adapter("p", input_mode="ocr")


# ------------------------------------------------------------------ engine


def _config(tmp_path: Path, **levers) -> Path:
    cfg = yaml.safe_load((REPO_ROOT / "harness" / "config.yaml").read_text())
    cfg["eval"]["n_workers"] = 2
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(cfg))
    scaffold = tmp_path / "scaffold"
    import shutil

    shutil.copytree(REPO_ROOT / "scaffold", scaffold)
    if levers:
        h = yaml.safe_load((scaffold / "harness.yaml").read_text())
        h["levers"] = levers
        (scaffold / "harness.yaml").write_text(yaml.safe_dump(h))
    (tmp_path / "harness").mkdir()
    shutil.copy(REPO_ROOT / "harness" / "model_catalog.csv", tmp_path / "harness")
    (tmp_path / "agents").mkdir()
    shutil.copy(REPO_ROOT / "agents" / "savesage_agent.py", tmp_path / "agents")
    return out


def _golden(tmp_path: Path, n: int = 4) -> Path:
    gs = tmp_path / "data" / "savesage_golden_set.jsonl"
    gs.parent.mkdir(parents=True, exist_ok=True)
    gs.write_text(
        "".join(
            json.dumps(
                {
                    "example_id": f"s{i}",
                    "query": f"s{i}",
                    "category": "amazon",
                    "pdf_path": str(tmp_path / f"s{i}.pdf"),
                    "expected_parsed_json": {},
                }
            )
            + "\n"
            for i in range(n)
        )
    )
    return gs


@needs_statement_agent
def test_engine_threads_levers_and_prices_calls(tmp_path: Path, fake_extractor) -> None:
    from anvil.domains.savesage.eval import evaluate_savesage

    _golden(tmp_path)
    report = evaluate_savesage(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=_config(tmp_path, model=GLM, input_mode="text"),
        golden_set_path="data/golden_set.jsonl",  # the loop's generic path
        mode="quick",
    )
    assert report.n_rows == 4
    assert {(b["model"], b["input_mode"]) for b in fake_extractor.built} == {(GLM, "text")}
    cm = report.cost_metrics
    assert cm["n_rows"] == 4.0 and cm["latency_ms_median"] >= 0.0
    # GLM-5.3 Flash: 1000 in @ ~$0.159/M + 500 out @ ~$0.529/M per statement.
    assert cm["cost_usd_per_row"] == pytest.approx(0.000423, rel=1e-2)


def test_shipped_config_is_the_savesage_instance() -> None:
    from anvil.runtime.loader import load_harness

    cfg = load_harness(REPO_ROOT / "scaffold").config
    assert (cfg.mode, cfg.agent_module, cfg.eval.engine) == (
        "code",
        "agents/savesage_agent.py",
        "savesage",
    )
    assert cfg.effective_runtime_model == LUNA and cfg.levers["input_mode"] == "pdf"
    assert cfg.lever_specs["model"].allowed == [LUNA, GLM, "databricks-deepseek-v4-1-flash"]
    assert cfg.eval.agent_evals.experiment_id == "967014443183055"
    assert [m.name for m in cfg.eval.agent_evals.user_metrics] == ["accuracy"]
    assert cfg.eval.modes["standard"].rows == sum(cfg.eval.modes["standard"].buckets.values())
    assert cfg.experiments.eval == "/Shared/forge-v3/savesage/eval"


# ------------------------------------------------------------------ call ledger


def test_ledger_refuses_a_second_send_of_the_same_document() -> None:
    from anvil.domains.savesage.extractor import CallLedger, DuplicateCallError

    ledger = CallLedger()
    ledger.begin("s")
    with ledger.call("s", b"PDF", {"model": GLM}):
        pass
    with pytest.raises(DuplicateCallError), ledger.call("s", b"PDF", {"model": LUNA}):
        pass  # fallback / race on another model: refused
    with ledger.call("s", b"PAGE-1", {"model": LUNA}):
        pass  # a different document (page split): allowed
    assert [c["model"] for c in ledger.end("s")] == [GLM, LUNA]


def test_ledger_waits_for_calls_still_in_flight() -> None:
    import threading
    import time

    from anvil.domains.savesage.extractor import CallLedger

    ledger = CallLedger()
    ledger.begin("s")
    started = threading.Event()

    def _slow():
        with ledger.call("s", b"PAGE-1", {"model": LUNA}) as rec:
            started.set()
            time.sleep(0.2)
            rec["input_tokens"] = 7

    t = threading.Thread(target=_slow)
    t.start()
    started.wait()
    calls = ledger.end("s")  # the agent returned; the call is still running
    t.join()
    assert calls == [{"model": LUNA, "input_tokens": 7}]


def test_unarmed_statements_pass_through() -> None:
    from anvil.domains.savesage.extractor import CallLedger

    ledger = CallLedger()
    for _ in range(2):
        with ledger.call("never-armed", b"PDF", {}):
            pass
    assert ledger.end("never-armed") == []


@needs_statement_agent
def test_engine_prices_every_call_and_fails_duplicates(
    tmp_path: Path,
    fake_extractor,  # noqa: ARG001
    caplog: pytest.LogCaptureFixture,
) -> None:
    from anvil.domains.savesage.eval import evaluate_savesage

    racer = tmp_path / "agents" / "racer.py"
    _golden(tmp_path)
    cfg = _config(tmp_path)
    racer.write_text(
        "from anvil.domains.savesage.agent_base import SavesageAgent\n"
        "class Racer(SavesageAgent):\n"
        "    def predict(self, *, sid, pdf_path):\n"
        "        self.extract(sid=sid, pdf_path=pdf_path)\n"
        "        return self.extract(sid=sid, pdf_path=pdf_path, reasoning_effort='low')\n"
    )
    raw = yaml.safe_load(cfg.read_text())
    raw["agent_module"] = str(racer)
    cfg.write_text(yaml.safe_dump(raw))
    report = evaluate_savesage(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=cfg,
        golden_set_path="data/golden_set.jsonl",
        mode="quick",
    )
    # The second (duplicate) send was refused, so each row failed — but the
    # first call was still made and is priced.
    assert report.n_rows == 4
    assert sum("already sent once" in r.getMessage() for r in caplog.records) == 4
    # Luna: 1000 in @ ~$0.2/M + 500 out @ ~$1.2/M = ~$0.0008 per row (one call).
    assert report.cost_metrics["cost_usd_per_row"] == pytest.approx(0.0008, rel=1e-2)
