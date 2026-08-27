"""Safety checks for optimizer-generated Python candidates."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Final


class CodeValidationError(Exception):
    """Raised when a code candidate fails validation."""


# Kept as public module-level data so later code-mode work can extend the
# policy without changing the AST walker.
FORBIDDEN_STRING_PATTERNS: Final[tuple[str, ...]] = (
    r"(?<![A-Za-z0-9_])test_",
    r"(?<![A-Za-z0-9_])eval(?=$|[/\\])",
    r"(?<![A-Za-z0-9_])solution(?![A-Za-z0-9_])",
    r"(?<![A-Za-z0-9_])golden_set(?![A-Za-z0-9_])",
    r"(?<![A-Za-z0-9_])answer_key(?![A-Za-z0-9_])",
    r"(?<![A-Za-z0-9_])ground_truth(?![A-Za-z0-9_])",
)
FORBIDDEN_IMPORT_PREFIXES: Final[tuple[str, ...]] = ("test_", "eval_")
FORBIDDEN_IMPORT_TERMS: Final[tuple[str, ...]] = ("solution",)


def _forbidden_string(value: str) -> str | None:
    for pattern in FORBIDDEN_STRING_PATTERNS:
        if re.search(pattern, value, flags=re.IGNORECASE):
            return pattern
    return None


def _forbidden_import(module_name: str) -> bool:
    name = module_name.lower()
    if name.startswith(FORBIDDEN_IMPORT_PREFIXES):
        return True
    return any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", name)
        for term in FORBIDDEN_IMPORT_TERMS
    )


def check_ast_denylist(file_path: Path | str) -> None:
    """Reject references to test, evaluation, solution, or answer data."""
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise CodeValidationError(f"could not parse code candidate {path}: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _forbidden_string(node.value) is not None:
                raise CodeValidationError(
                    f"forbidden reference in string literal at line {node.lineno}"
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _forbidden_import(alias.name):
                    raise CodeValidationError(
                        f"forbidden import {alias.name!r} at line {node.lineno}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _forbidden_import(module):
                raise CodeValidationError(f"forbidden import {module!r} at line {node.lineno}")


def validate_imports(file_path: Path | str) -> None:
    """Load a candidate from an isolated temporary working directory."""
    path = Path(file_path)
    if not path.is_file():
        raise CodeValidationError(f"code candidate is not a file: {path}")

    module_name = f"_anvil_candidate_{uuid.uuid4().hex}"
    previous_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="anvil-code-validation-") as temp_dir:
            isolated_path = Path(temp_dir) / path.name
            shutil.copy2(path, isolated_path)
            os.chdir(temp_dir)
            spec = importlib.util.spec_from_file_location(module_name, isolated_path)
            if spec is None or spec.loader is None:
                raise CodeValidationError(f"could not create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            # Deliberately do not register the candidate in sys.modules.
            spec.loader.exec_module(module)
    except CodeValidationError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise CodeValidationError(
            f"code candidate {path} failed to import: {exc.__class__.__name__}: {exc}"
        ) from exc
    finally:
        os.chdir(previous_cwd)


def validate_code_candidate(file_path: Path | str) -> None:
    """Run the cheap AST policy check before attempting an isolated import."""
    check_ast_denylist(file_path)
    validate_imports(file_path)
