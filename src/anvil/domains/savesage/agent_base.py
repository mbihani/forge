"""Code-mode agent base for the Savesage ICICI domain.

In *code* mode the optimizer writes a Python ``SavesageAgent`` subclass
(instead of editing prompt skills). The subclass decides, per statement,
how to call Luna — the ONE lever available for **latency** is the
extraction payload's ``reasoning_effort`` / ``max_tokens`` (native-PDF
input is mandatory and the LLM inference dominates wall-clock, so there
is nothing else a Python strategy can change). A subclass may pick those
knobs statically or adaptively from a cheap per-statement signal (PDF
byte size, page count, the co-brand token in the filename).

The base ships :meth:`run_luna`, the single call a subclass makes to
extract one statement and get back ``(parsed_json, latency_ms)``. It
reuses the production Luna transport via
:class:`anvil.domains.savesage.extractor.SavesageIciciExtractor` with the
Luna cache DISABLED — a cache hit returns in ~0 ms and would fabricate
the latency signal the code-mode gate optimizes, so every code-mode call
is a live, cold extraction.

Import is light on purpose (only ``abc`` + ``typing``): the extractor —
and through it the statement-agent tree — is imported lazily inside
:meth:`run_luna`. This keeps a subclass module import-safe under the
optimizer's isolated-import validation (``code_validation``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# The production defaults the extraction payload ships with today. A
# subclass that calls run_luna with these reproduces current behavior.
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 96_000


class SavesageAgent(ABC):
    """Base class for a code-mode Savesage ICICI extraction agent.

    The eval harness constructs one instance per round with the
    composed ICICI prompt and the Luna workspace profile, then calls
    :meth:`predict` once per statement. A subclass implements only
    :meth:`predict`; it calls :meth:`run_luna` to do the extraction.

    ``composed_prompt`` is the ICICI system prompt composed from the
    scaffold (identical to the prompt-mode prompt). ``luna_profile`` is
    the Databricks profile Luna authenticates against (threaded to the
    extractor; ``None`` uses the extractor's default).
    """

    def __init__(self, composed_prompt: str, *, luna_profile: str | None = None) -> None:
        self._composed_prompt = composed_prompt
        self._luna_profile = luna_profile
        # Extractors cached by (reasoning_effort, max_tokens) so an adaptive
        # agent that varies the knobs builds one extractor per distinct combo
        # and reuses it across statements. Cache-free extraction (live latency).
        self._extractors: dict[tuple[str, int], Any] = {}

    def run_luna(
        self,
        *,
        sid: str,
        pdf_path: str | Path,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[dict[str, Any], float]:
        """Extract one statement through Luna; return ``(parsed, latency_ms)``.

        No caching — the returned ``latency_ms`` is the real cold-call
        wall time, which is the signal the latency gate optimizes.
        """
        key = (reasoning_effort, int(max_tokens))
        extractor = self._extractors.get(key)
        if extractor is None:
            from anvil.domains.savesage.extractor import SavesageIciciExtractor

            extractor = SavesageIciciExtractor(
                self._composed_prompt,
                cache_root=None,
                luna_profile=self._luna_profile,
                reasoning_effort=reasoning_effort,
                max_tokens=int(max_tokens),
            )
            self._extractors[key] = extractor
        return extractor.extract_with_latency(sid=sid, pdf_path=pdf_path)

    @abstractmethod
    def predict(self, *, sid: str, pdf_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract one ICICI statement.

        Returns ``(extraction, meta)`` where ``extraction`` is the parsed
        statement dict (scored against the cached Opus GT) and ``meta``
        MUST carry ``latency_ms`` (float). ``meta`` may also record the
        ``reasoning_effort`` / ``max_tokens`` chosen, for provenance.
        """
        ...
