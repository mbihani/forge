"""Code-mode agent base for the Savesage ICICI domain.

In *code* mode the optimizer writes a Python ``SavesageAgent`` subclass
(instead of editing prompt skills). The subclass decides, per statement,
how to run the extraction — the levers available for **latency** (and
cost) are:

* ``model`` — any of ``self.allowed_models`` (the ``model`` lever's
  allowlist; ``self.model`` is the round's lever value);
* ``input_mode`` — ``"pdf"`` sends the production native-PDF ``file``
  block (only :data:`NATIVE_PDF_MODELS` accept it); ``"text"`` sends the
  ``pdftotext -layout`` extraction instead (works for every model; the
  local text step counts toward latency);
* ``reasoning_effort`` / ``max_tokens`` of the extraction payload.

A subclass may pick these statically or adaptively from a cheap
per-statement signal (PDF byte size, page count, the co-brand token in
the filename).

During an eval every extraction passes through
:data:`anvil.domains.savesage.extractor.LEDGER`: a document goes to a
model at most once per statement (no racing, no fallback re-runs —
:class:`~anvil.domains.savesage.extractor.DuplicateCallError`), and every
call is priced into the row's cost.

The base ships :meth:`extract`, the single call a subclass makes to
extract one statement and get back ``(parsed_json, meta)``. It reuses the
production transport via
:class:`anvil.domains.savesage.extractor.SavesageIciciExtractor` with the
cache DISABLED — a cache hit returns in ~0 ms and would fabricate the
latency signal the code-mode gate optimizes, so every code-mode call is a
live, cold extraction.

Import is light on purpose (only ``abc`` + ``typing``): the extractor —
and through it the statement-agent tree — is imported lazily inside
:meth:`extract`. This keeps a subclass module import-safe under the
optimizer's isolated-import validation (``code_validation``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# The production defaults the extraction payload ships with today. A
# subclass that calls extract() with these reproduces current behavior.
DEFAULT_MODEL = "databricks-gpt-5-6-luna"
DEFAULT_INPUT_MODE = "pdf"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 96_000

# Endpoints that accept the native-PDF ``file`` block. GLM and DeepSeek
# reject it ("unsupported content item type"), so they run ``text`` only.
NATIVE_PDF_MODELS = frozenset({"databricks-gpt-5-6-luna", "databricks-gpt-6-luna"})


class SavesageAgent(ABC):
    """Base class for a code-mode Savesage ICICI extraction agent.

    The eval harness constructs one instance per round with the
    composed ICICI prompt, the round's ``model`` / ``input_mode`` lever
    values and the model allowlist, then calls :meth:`predict` once per
    statement (from several threads at once — keep per-statement state
    local). A subclass implements only :meth:`predict`; it calls
    :meth:`extract` to do each extraction.

    ``composed_prompt`` is the ICICI system prompt composed from the
    scaffold (identical to the prompt-mode prompt). ``luna_profile`` is
    the Databricks profile the serving endpoints authenticate against
    (threaded to the extractor; ``None`` uses the extractor's default).
    """

    def __init__(
        self,
        composed_prompt: str,
        *,
        luna_profile: str | None = None,
        model: str = DEFAULT_MODEL,
        input_mode: str = DEFAULT_INPUT_MODE,
        allowed_models: list[str] | None = None,
    ) -> None:
        self._composed_prompt = composed_prompt
        self._luna_profile = luna_profile
        self.model = model
        self.input_mode = input_mode
        self.allowed_models = list(allowed_models or [model])
        # Extractors cached by (model, input_mode, reasoning_effort, max_tokens)
        # so an adaptive agent builds one per distinct combo and reuses it.
        self._extractors: dict[tuple[str, str, str | None, int], Any] = {}

    @property
    def composed_prompt(self) -> str:
        return self._composed_prompt

    def extract(
        self,
        *,
        sid: str,
        pdf_path: str | Path,
        model: str | None = None,
        input_mode: str | None = None,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract one statement; return ``(parsed, meta)``.

        ``model`` / ``input_mode`` default to the round's lever values.
        ``meta`` carries ``latency_ms`` (real cold-call wall time) and
        ``calls`` (one record with the model, knobs and token usage, which
        the eval prices). Return it from :meth:`predict` as-is, or merge
        several with :meth:`merge_meta`.
        """
        model = model or self.model
        input_mode = input_mode or self.input_mode
        if model not in self.allowed_models:
            raise ValueError(f"model {model!r} not in allowed_models {self.allowed_models}")
        if input_mode == "pdf" and model not in NATIVE_PDF_MODELS:
            raise ValueError(f"model {model!r} cannot read native PDFs; use input_mode='text'")
        key = (model, input_mode, reasoning_effort, int(max_tokens))
        extractor = self._extractors.get(key)
        if extractor is None:
            from anvil.domains.savesage.extractor import SavesageIciciExtractor

            extractor = SavesageIciciExtractor(
                self._composed_prompt,
                cache_root=None,
                luna_profile=self._luna_profile,
                reasoning_effort=reasoning_effort,
                max_tokens=int(max_tokens),
                model=model,
                input_mode=input_mode,
            )
            self._extractors[key] = extractor
        parsed, latency_ms, usage = extractor.extract_with_usage(sid=sid, pdf_path=pdf_path)
        call = {
            "model": model,
            "input_mode": input_mode,
            "reasoning_effort": reasoning_effort,
            "max_tokens": int(max_tokens),
            "latency_ms": latency_ms,
            **usage,
        }
        return parsed, {"latency_ms": latency_ms, "calls": [call]}

    @staticmethod
    def merge_meta(*metas: dict[str, Any]) -> dict[str, Any]:
        """Combine the ``meta`` of several :meth:`extract` calls (sequential)."""
        return {
            "latency_ms": sum(float(m.get("latency_ms", 0.0)) for m in metas),
            "calls": [c for m in metas for c in m.get("calls", [])],
        }

    def run_luna(
        self,
        *,
        sid: str,
        pdf_path: str | Path,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[dict[str, Any], float]:
        """Back-compat: native-PDF Luna extraction; return ``(parsed, latency_ms)``."""
        parsed, meta = self.extract(
            sid=sid,
            pdf_path=pdf_path,
            model=DEFAULT_MODEL,
            input_mode="pdf",
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        return parsed, meta["latency_ms"]

    @abstractmethod
    def predict(self, *, sid: str, pdf_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract one ICICI statement.

        Returns ``(extraction, meta)`` where ``extraction`` is the parsed
        statement dict (scored against the cached Opus GT) and ``meta``
        MUST carry ``latency_ms`` (float) and SHOULD carry the ``calls``
        from :meth:`extract` (they price the row).
        """
        ...
