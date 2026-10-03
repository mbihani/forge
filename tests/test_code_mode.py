"""Tests for the code-mode scaffold (Phase 3 task 3.1).

Covers the acceptance contract:

* ``MemorySystem`` ABC — abstract, requires ``predict`` + ``learn_from_batch``.
* ``BaselineExtractor`` — passthrough without an LLM client, no learning.
* ``RuntimeYAML.mode`` — ``Literal["prompt", "code"]``, default ``"prompt"``,
  plus ``agent_module`` with a sensible default.
* Code-mode eval path — ``_load_memory_system`` imports a module (dotted
  path or ``.py`` file) and finds the single ``MemorySystem`` subclass;
  ``evaluate_branch`` routes to it instead of ``AnvilAgent`` when
  ``mode: code``.
* Round-loop awareness — ``_read_optimization_mode`` reads the mode from
  ``harness/config.yaml``; ``RoundReport`` carries a ``mode`` field.

No LLM calls and no Databricks calls are made — ``mlflow.genai.evaluate``
and the runtime agent are mocked.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _gold(example_id: str, answer: str) -> dict:
    return {
        "example_id": example_id,
        "query": f"q-{example_id}",
        "category": "direct",
        "expected_doc_ids": [],
        "reference_answer": answer,
        "should_refuse": False,
        "expected_citations": [],
        "must_include": [answer],
        "must_not_include": [],
        "notes_for_judge": "",
    }


# ---------------------------------------------------------------------------
# 1. MemorySystem ABC
# ---------------------------------------------------------------------------


def test_memory_system_cannot_be_instantiated() -> None:
    """MemorySystem is abstract — direct instantiation must fail."""
    from anvil.agents.memory_system import MemorySystem

    with pytest.raises(TypeError, match="abstract"):
        MemorySystem()


def test_memory_system_requires_predict() -> None:
    from anvil.agents.memory_system import MemorySystem

    class NoPredict(MemorySystem):
        def learn_from_batch(self, batch_results: list[dict]) -> None:
            pass

    with pytest.raises(TypeError, match="predict"):
        NoPredict()


def test_memory_system_requires_learn_from_batch() -> None:
    from anvil.agents.memory_system import MemorySystem

    class NoLearn(MemorySystem):
        def predict(self, input: str) -> tuple[str, dict]:
            return input, {}

    with pytest.raises(TypeError, match="learn_from_batch"):
        NoLearn()


def test_memory_system_default_get_state() -> None:
    """get_state returns '{}' by default; set_state is a no-op."""
    from anvil.agents.memory_system import MemorySystem

    class Minimal(MemorySystem):
        def predict(self, input: str) -> tuple[str, dict]:
            return input, {}

        def learn_from_batch(self, batch_results: list[dict]) -> None:
            pass

    m = Minimal()
    assert m.get_state() == "{}"
    # set_state should not raise.
    m.set_state('{"key": "value"}')


# ---------------------------------------------------------------------------
# 2. BaselineExtractor
# ---------------------------------------------------------------------------


def test_baseline_is_memory_system() -> None:
    from anvil.agents.baseline import BaselineExtractor
    from anvil.agents.memory_system import MemorySystem

    assert issubclass(BaselineExtractor, MemorySystem)


def test_baseline_passthrough_without_client() -> None:
    """Without an LLM client, predict echoes the input (testable, no
    external service)."""
    from anvil.agents.baseline import BaselineExtractor

    agent = BaselineExtractor()
    answer, metadata = agent.predict("hello world")
    assert answer == "hello world"
    assert metadata == {"context_chars": 11}


def test_baseline_learn_from_batch_is_noop() -> None:
    from anvil.agents.baseline import BaselineExtractor

    agent = BaselineExtractor()
    # Should not raise and should not change state.
    agent.learn_from_batch([{"ground_truth": "x", "prediction": "y"}])
    assert agent.get_state() == "{}"


def test_baseline_default_state_methods() -> None:
    from anvil.agents.baseline import BaselineExtractor

    agent = BaselineExtractor()
    assert agent.get_state() == "{}"
    agent.set_state('{"learned": true}')
    assert agent.get_state() == "{}"  # baseline has no state


def test_baseline_with_mock_llm_client() -> None:
    """When an LLM client is injected, predict calls it and returns the
    response content."""
    from anvil.agents.baseline import BaselineExtractor

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="LLM reply"))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))
    )
    agent = BaselineExtractor(llm_client=client, model="test-model")
    answer, metadata = agent.predict("question")
    assert answer == "LLM reply"
    assert metadata == {"context_chars": len("question")}


# ---------------------------------------------------------------------------
# 3. Mode config parsing
# ---------------------------------------------------------------------------


def test_runtime_yaml_mode_defaults_to_prompt() -> None:
    from anvil.runtime.models import RuntimeYAML

    cfg = RuntimeYAML(
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments={"root": "/Shared/forge"},
    )
    assert cfg.mode == "prompt"
    assert cfg.agent_module == "anvil.agents.baseline"


def test_runtime_yaml_accepts_code_mode() -> None:
    from anvil.runtime.models import RuntimeYAML

    cfg = RuntimeYAML(
        mode="code",
        agent_module="my.custom.agent",
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments={"root": "/Shared/forge"},
    )
    assert cfg.mode == "code"
    assert cfg.agent_module == "my.custom.agent"


def test_runtime_yaml_rejects_invalid_mode() -> None:
    from pydantic import ValidationError

    from anvil.runtime.models import RuntimeYAML

    with pytest.raises(ValidationError):
        RuntimeYAML(
            mode="hybrid",
            runtime_endpoint="rt",
            optimizer_endpoint="op",
            judge_endpoint="j",
            experiments={"root": "/Shared/forge"},
        )


def test_harness_config_mode_flows_through_from_split() -> None:
    """from_split carries mode + agent_module from RuntimeYAML into the
    merged HarnessConfig."""
    from anvil.runtime.models import (
        ExperimentsConfig,
        HarnessConfig,
        RuntimeYAML,
        ScaffoldYAML,
    )

    runtime = RuntimeYAML(
        mode="code",
        agent_module="custom.agent",
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(),
    )
    scaffold = ScaffoldYAML()
    merged = HarnessConfig.from_split(scaffold, runtime)
    assert merged.mode == "code"
    assert merged.agent_module == "custom.agent"


def test_harness_config_defaults_to_prompt() -> None:
    from anvil.runtime.models import HarnessConfig

    cfg = HarnessConfig(
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments={"root": "/Shared/forge"},
    )
    assert cfg.mode == "prompt"
    assert cfg.agent_module == "anvil.agents.baseline"


def test_real_config_has_mode_prompt() -> None:
    """The repo's harness/config.yaml must have mode: prompt (backward
    compatible with all existing rounds)."""
    import yaml

    from anvil.runtime.models import RuntimeYAML

    config_path = REPO_ROOT / "harness" / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["mode"] == "prompt"
    cfg = RuntimeYAML.model_validate(raw)
    assert cfg.mode == "prompt"
    assert cfg.agent_module == "anvil.agents.baseline"


def test_mode_in_runtime_fields() -> None:
    """mode + agent_module belong to config.yaml, not scaffold/harness.yaml.
    The loader uses RUNTIME_FIELDS to produce a helpful error when they
    appear on the wrong side of the split."""
    from anvil.runtime.models import RUNTIME_FIELDS

    assert "mode" in RUNTIME_FIELDS
    assert "agent_module" in RUNTIME_FIELDS


# ---------------------------------------------------------------------------
# 4. Code-mode agent loading — _load_memory_system
# ---------------------------------------------------------------------------


def test_load_memory_system_from_dotted_path() -> None:
    """_load_memory_system imports anvil.agents.baseline by dotted path
    and finds the BaselineExtractor subclass."""
    from anvil.agents.baseline import BaselineExtractor
    from anvil.agents.memory_system import MemorySystem
    from anvil.eval.runner import _load_memory_system

    ms = _load_memory_system("anvil.agents.baseline")
    assert isinstance(ms, MemorySystem)
    assert isinstance(ms, BaselineExtractor)


def test_load_memory_system_from_file_path(tmp_path: Path) -> None:
    """_load_memory_system loads a .py file and finds the MemorySystem
    subclass defined in it."""
    from anvil.agents.memory_system import MemorySystem
    from anvil.eval.runner import _load_memory_system

    agent_file = tmp_path / "custom_agent.py"
    agent_file.write_text(
        textwrap.dedent(
            """\
            from anvil.agents.memory_system import MemorySystem

            class CustomAgent(MemorySystem):
                def __init__(self, **kwargs):
                    pass
                def predict(self, input):
                    return "custom", {}
                def learn_from_batch(self, batch_results):
                    pass
            """
        ),
        encoding="utf-8",
    )
    ms = _load_memory_system(str(agent_file))
    assert isinstance(ms, MemorySystem)
    assert ms.__class__.__name__ == "CustomAgent"


def test_load_memory_system_passes_llm_client_and_model() -> None:
    """Constructor kwargs (llm_client, model) are forwarded to the
    MemorySystem subclass."""
    from anvil.eval.runner import _load_memory_system

    sentinel_client = object()
    ms = _load_memory_system(
        "anvil.agents.baseline",
        llm_client=sentinel_client,
        model="test-model",
    )
    assert ms.llm_client is sentinel_client
    assert ms.model == "test-model"


def test_load_memory_system_no_subclass_raises(tmp_path: Path) -> None:
    """A module that imports MemorySystem but defines no subclass raises
    ValueError."""
    from anvil.eval.runner import _load_memory_system

    agent_file = tmp_path / "empty_agent.py"
    agent_file.write_text(
        "from anvil.agents.memory_system import MemorySystem\n# no subclass\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no concrete MemorySystem subclass"):
        _load_memory_system(str(agent_file))


def test_load_memory_system_multiple_subclasses_raises(tmp_path: Path) -> None:
    """A module with two MemorySystem subclasses raises ValueError — the
    loader cannot disambiguate."""
    from anvil.eval.runner import _load_memory_system

    agent_file = tmp_path / "multi_agent.py"
    agent_file.write_text(
        textwrap.dedent(
            """\
            from anvil.agents.memory_system import MemorySystem

            class Agent1(MemorySystem):
                def predict(self, input):
                    return "x", {}
                def learn_from_batch(self, batch_results):
                    pass

            class Agent2(MemorySystem):
                def predict(self, input):
                    return "y", {}
                def learn_from_batch(self, batch_results):
                    pass
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple"):
        _load_memory_system(str(agent_file))


def test_load_memory_system_missing_file_raises(tmp_path: Path) -> None:
    from anvil.eval.runner import _load_memory_system

    with pytest.raises(FileNotFoundError, match="agent module not found"):
        _load_memory_system(str(tmp_path / "nope.py"))


def test_find_subclass_ignores_imported_base_class(tmp_path: Path) -> None:
    """_find_memory_system_subclass ignores the MemorySystem base class
    itself (which is imported, not defined, in the agent module)."""
    from anvil.eval.runner import _find_memory_system_subclass, _import_agent_module

    agent_file = tmp_path / "agent.py"
    agent_file.write_text(
        textwrap.dedent(
            """\
            from anvil.agents.memory_system import MemorySystem

            class MyAgent(MemorySystem):
                def predict(self, input):
                    return "x", {}
                def learn_from_batch(self, batch_results):
                    pass
            """
        ),
        encoding="utf-8",
    )
    module = _import_agent_module(str(agent_file))
    cls = _find_memory_system_subclass(module)
    assert cls.__name__ == "MyAgent"


def test_load_memory_system_file_path_with_dataclass(tmp_path: Path) -> None:
    """A MemorySystem subclass using @dataclass must import correctly
    when loaded from a .py file path. This requires the module to be
    registered in sys.modules before exec_module — @dataclass and other
    runtime type-resolution mechanisms look up the defining module there."""
    from anvil.agents.memory_system import MemorySystem
    from anvil.eval.runner import _load_memory_system

    agent_file = tmp_path / "dataclass_agent.py"
    agent_file.write_text(
        textwrap.dedent(
            """\
            from dataclasses import dataclass
            from anvil.agents.memory_system import MemorySystem

            @dataclass
            class DataclassAgent(MemorySystem):
                llm_client: object = None
                model: str = ""

                def predict(self, input: str):
                    return "dataclass", {}

                def learn_from_batch(self, batch_results: list):
                    pass
            """
        ),
        encoding="utf-8",
    )
    ms = _load_memory_system(str(agent_file))
    assert isinstance(ms, MemorySystem)
    assert ms.__class__.__name__ == "DataclassAgent"


def test_find_subclass_ignores_abstract_helpers(tmp_path: Path) -> None:
    """An abstract intermediate subclass plus one concrete implementation
    should find only the concrete class — abstract helpers are filtered
    out via inspect.isabstract so they don't trigger the 'multiple
    subclasses' error."""
    from anvil.eval.runner import _find_memory_system_subclass, _import_agent_module

    agent_file = tmp_path / "abstract_agent.py"
    agent_file.write_text(
        textwrap.dedent(
            """\
            from abc import abstractmethod
            from anvil.agents.memory_system import MemorySystem

            class AbstractHelper(MemorySystem):
                @abstractmethod
                def extra_method(self) -> str:
                    ...

            class ConcreteAgent(AbstractHelper):
                def predict(self, input: str):
                    return "x", {}
                def learn_from_batch(self, batch_results: list):
                    pass
                def extra_method(self) -> str:
                    return "done"
            """
        ),
        encoding="utf-8",
    )
    module = _import_agent_module(str(agent_file))
    cls = _find_memory_system_subclass(module)
    assert cls.__name__ == "ConcreteAgent"


def test_find_subclass_all_abstract_raises(tmp_path: Path) -> None:
    """A module with only abstract subclasses should raise a clear
    ValueError, not an opaque TypeError at instantiation time."""
    from anvil.eval.runner import _find_memory_system_subclass, _import_agent_module

    agent_file = tmp_path / "only_abstract.py"
    agent_file.write_text(
        textwrap.dedent(
            """\
            from abc import abstractmethod
            from anvil.agents.memory_system import MemorySystem

            class StillAbstract(MemorySystem):
                @abstractmethod
                def extra_method(self) -> str:
                    ...
            """
        ),
        encoding="utf-8",
    )
    module = _import_agent_module(str(agent_file))
    with pytest.raises(ValueError, match="no concrete MemorySystem subclass"):
        _find_memory_system_subclass(module)


# ---------------------------------------------------------------------------
# 5. Code-mode eval path — evaluate_branch routing
# ---------------------------------------------------------------------------


def _patch_runner_common(
    monkeypatch: pytest.MonkeyPatch,
    config,
) -> None:
    """Patch the runner's external dependencies for a mocked eval run."""
    from anvil.eval import runner

    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(
        runner, "load_golden_set", lambda _p: [_gold("g1", "hello"), _gold("g2", "world")]
    )
    monkeypatch.setattr(runner, "select_subset", lambda exs, **_k: exs)
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr("anvil.observability.enable_runtime_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "get_experiment_by_name", lambda *a, **kw: None)


def test_evaluate_branch_code_mode_routes_to_memory_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In code mode, evaluate_branch loads the MemorySystem and the
    predict_fn calls predict() per row. AnvilAgent is never constructed."""
    from anvil.agents.memory_system import MemorySystem
    from anvil.eval import runner
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    class FakeMemorySystem(MemorySystem):
        def __init__(self, llm_client=None, model="") -> None:
            self.predict_calls: list[str] = []

        def predict(self, input: str) -> tuple[str, dict]:
            self.predict_calls.append(input)
            return f"answer-{input}", {"context_chars": len(input)}

        def learn_from_batch(self, batch_results: list[dict]) -> None:
            pass

    fake_ms = FakeMemorySystem()
    loaded: dict[str, object] = {}

    def fake_load(module_path: str, **kwargs: object) -> MemorySystem:
        loaded["module_path"] = module_path
        loaded["kwargs"] = kwargs
        return fake_ms

    monkeypatch.setattr(runner, "_load_memory_system", fake_load)

    agent_constructed: list[bool] = []

    def fake_agent(*a: object, **kw: object) -> SimpleNamespace:
        agent_constructed.append(True)
        return SimpleNamespace()

    monkeypatch.setattr(runner, "AnvilAgent", fake_agent)

    config = HarnessConfig(
        mode="code",
        agent_module="my.custom.agent",
        runtime_endpoint="rt-model",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="correctness", type="llm", weight=1.0)],
            modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
        ),
    )
    _patch_runner_common(monkeypatch, config)

    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs: object) -> SimpleNamespace:
        captured["predict_fn"] = kwargs["predict_fn"]
        captured["data"] = kwargs["data"]
        # Exercise the predict_fn to verify it routes to the MemorySystem.
        captured["results"] = [
            kwargs["predict_fn"](row["inputs"]["query"]) for row in kwargs["data"]
        ]
        return SimpleNamespace(
            result_df=pd.DataFrame({"correctness/value": [1.0, 1.0]}),
            metrics={},
            run_id="run-1",
        )

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    # _load_memory_system was called with the configured module + model.
    assert loaded["module_path"] == "my.custom.agent"
    assert loaded["kwargs"]["model"] == "rt-model"
    # AnvilAgent was NOT constructed in code mode.
    assert agent_constructed == []
    # predict_fn routed through the MemorySystem.predict().
    assert captured["results"] == ["answer-q-g1", "answer-q-g2"]
    assert fake_ms.predict_calls == ["q-g1", "q-g2"]


def test_evaluate_branch_prompt_mode_routes_to_anvil_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In prompt mode, evaluate_branch constructs AnvilAgent and does
    NOT call _load_memory_system."""
    from anvil.eval import runner
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    load_called: list[bool] = []

    def fake_load(*a: object, **kw: object) -> None:
        load_called.append(True)

    monkeypatch.setattr(runner, "_load_memory_system", fake_load)

    agent_constructed: list[bool] = []

    def fake_agent(*a: object, **kw: object) -> SimpleNamespace:
        agent_constructed.append(True)
        return SimpleNamespace(predict=lambda req: SimpleNamespace(output=[]))

    monkeypatch.setattr(runner, "AnvilAgent", fake_agent)

    config = HarnessConfig(
        mode="prompt",
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="correctness", type="llm", weight=1.0)],
            modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
        ),
    )
    _patch_runner_common(monkeypatch, config)

    monkeypatch.setattr(
        runner.mlflow.genai,
        "evaluate",
        lambda **kw: SimpleNamespace(
            result_df=pd.DataFrame({"correctness/value": [1.0, 1.0]}),
            metrics={},
            run_id="run-1",
        ),
    )

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=SimpleNamespace(),
    )

    # AnvilAgent WAS constructed.
    assert agent_constructed == [True]
    # _load_memory_system was NOT called.
    assert load_called == []


def test_evaluate_branch_code_mode_real_baseline_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end with the REAL BaselineExtractor (no monkeypatch on
    _load_memory_system): the passthrough agent echoes the query. No
    LLM client is injected, so no external call is made."""
    from anvil.eval import runner
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    # Build a real config that points code mode at the baseline agent.
    # Pass runtime_client=None so the BaselineExtractor does passthrough.
    config = HarnessConfig(
        mode="code",
        agent_module="anvil.agents.baseline",
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="correctness", type="llm", weight=1.0)],
            modes={"quick": EvalModeConfig(rows=2, buckets={"direct": 2})},
        ),
    )
    _patch_runner_common(monkeypatch, config)
    monkeypatch.setattr(runner, "build_gateway_client", lambda **kw: None)

    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs: object) -> SimpleNamespace:
        captured["results"] = [
            kwargs["predict_fn"](row["inputs"]["query"]) for row in kwargs["data"]
        ]
        return SimpleNamespace(
            result_df=pd.DataFrame({"correctness/value": [1.0, 1.0]}),
            metrics={},
            run_id="run-1",
        )

    monkeypatch.setattr(runner.mlflow.genai, "evaluate", fake_evaluate)

    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=None,
        judge_client=SimpleNamespace(),
    )

    # BaselineExtractor with no client echoes the query.
    assert captured["results"] == ["q-g1", "q-g2"]


# ---------------------------------------------------------------------------
# 6. Round-loop mode awareness
# ---------------------------------------------------------------------------


def test_read_optimization_mode_no_config_file(tmp_path: Path) -> None:
    """When harness/config.yaml is absent, mode defaults to prompt."""
    from anvil.loop.round import _read_optimization_mode

    assert _read_optimization_mode(tmp_path / "scaffold") == "prompt"


def test_read_optimization_mode_prompt(tmp_path: Path) -> None:
    from anvil.loop.round import _read_optimization_mode

    scaffold_root = tmp_path / "scaffold"
    config = tmp_path / "harness" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("mode: prompt\nruntime_endpoint: x\n", encoding="utf-8")
    assert _read_optimization_mode(scaffold_root) == "prompt"


def test_read_optimization_mode_code(tmp_path: Path) -> None:
    from anvil.loop.round import _read_optimization_mode

    scaffold_root = tmp_path / "scaffold"
    config = tmp_path / "harness" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("mode: code\nruntime_endpoint: x\n", encoding="utf-8")
    assert _read_optimization_mode(scaffold_root) == "code"


def test_read_optimization_mode_absent_defaults_to_prompt(tmp_path: Path) -> None:
    """When the mode key is missing from the config, default to prompt."""
    from anvil.loop.round import _read_optimization_mode

    scaffold_root = tmp_path / "scaffold"
    config = tmp_path / "harness" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("runtime_endpoint: x\n", encoding="utf-8")
    assert _read_optimization_mode(scaffold_root) == "prompt"


def test_round_report_has_mode_field() -> None:
    """RoundReport carries a mode field that defaults to prompt."""
    from anvil.loop.decision import Decision
    from anvil.loop.round import RoundReport

    report = RoundReport(
        round_id=1,
        branch="anvil/exp-round-1",
        decision=Decision.KEEP,
        action_kind="add_rule",
        parse_status="ok",
        diff_summary="add_rule rules/foo.md",
    )
    assert report.mode == "prompt"

    report_code = RoundReport(
        round_id=2,
        branch="anvil/exp-round-2",
        decision=Decision.KEEP,
        action_kind="add_skill",
        parse_status="ok",
        diff_summary="add_skill skills/bar.md",
        mode="code",
    )
    assert report_code.mode == "code"
