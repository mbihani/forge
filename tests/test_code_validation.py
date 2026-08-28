"""Tests for generated Python candidate validation."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from anvil.optimizer.code_validation import (
    CodeValidationError,
    check_ast_denylist,
    validate_code_candidate,
    validate_imports,
)


def _candidate(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "candidate.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_clean_import_passes_without_polluting_sys_modules(tmp_path: Path) -> None:
    path = _candidate(tmp_path, "VALUE = 42\n")
    before = set(sys.modules)

    validate_imports(path)

    assert set(sys.modules) == before


def test_import_failure_raises_validation_error(tmp_path: Path) -> None:
    path = _candidate(tmp_path, "raise RuntimeError('broken candidate')\n")

    with pytest.raises(CodeValidationError, match="failed to import.*broken candidate"):
        validate_imports(path)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_validate_imports_propagates_process_control_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception: type[BaseException]
) -> None:
    path = _candidate(tmp_path, "VALUE = 42\n")

    class FakeLoader:
        def create_module(self, spec: ModuleSpec) -> None:
            return None

        def exec_module(self, module: object) -> None:
            raise exception

    spec = ModuleSpec("candidate", FakeLoader())
    monkeypatch.setattr(
        "anvil.optimizer.code_validation.importlib.util.spec_from_file_location",
        lambda *_args: spec,
    )

    with pytest.raises(exception):
        validate_imports(path)


def test_import_side_effect_is_isolated_to_temp_directory(tmp_path: Path) -> None:
    path = _candidate(tmp_path, "from pathlib import Path\nPath('side-effect').write_text('x')\n")

    validate_imports(path)

    assert not (tmp_path / "side-effect").exists()


@pytest.mark.parametrize(
    "literal",
    [
        "test_private.py",
        "eval/results.json",
        "solution.py",
        "golden_set.json",
        "answer_key.yaml",
        "ground_truth.csv",
    ],
)
def test_ast_denylist_rejects_forbidden_string_literals(tmp_path: Path, literal: str) -> None:
    path = _candidate(tmp_path, f"REFERENCE = {literal!r}\n")

    with pytest.raises(CodeValidationError, match="forbidden reference"):
        check_ast_denylist(path)


@pytest.mark.parametrize(
    "source",
    [
        "import test_helpers\n",
        "from eval_runner import score\n",
        "import solution\n",
    ],
)
def test_ast_denylist_rejects_forbidden_imports(tmp_path: Path, source: str) -> None:
    with pytest.raises(CodeValidationError, match="forbidden import"):
        check_ast_denylist(_candidate(tmp_path, source))


@pytest.mark.parametrize(
    "source",
    ["import resolution\n", "from absolution import x\n", "import dissolution\n"],
)
def test_ast_denylist_import_terms_use_word_boundaries(tmp_path: Path, source: str) -> None:
    check_ast_denylist(_candidate(tmp_path, source))


@pytest.mark.parametrize(
    "source",
    [
        "open('eval/held_out.json')\n",
        "from pathlib import Path\nPath('test_answers.json').open()\n",
        "open('solution/output.txt', 'w')\n",
    ],
)
def test_ast_denylist_rejects_file_calls_with_forbidden_paths(tmp_path: Path, source: str) -> None:
    with pytest.raises(CodeValidationError, match="forbidden reference"):
        check_ast_denylist(_candidate(tmp_path, source))


@pytest.mark.parametrize(
    "literal",
    ["evaluation", "evaluate", "contest_results", "resolution", "golden retriever"],
)
def test_ast_denylist_avoids_substring_false_positives(tmp_path: Path, literal: str) -> None:
    check_ast_denylist(_candidate(tmp_path, f"DESCRIPTION = {literal!r}\n"))


def test_validate_code_candidate_runs_ast_check_before_import(tmp_path: Path) -> None:
    path = _candidate(
        tmp_path,
        "open('golden_set.json')\nraise RuntimeError('import should not run')\n",
    )

    with pytest.raises(CodeValidationError, match="forbidden reference"):
        validate_code_candidate(path)


def test_validate_code_candidate_accepts_clean_candidate(tmp_path: Path) -> None:
    validate_code_candidate(_candidate(tmp_path, "def remember(value):\n    return value\n"))
