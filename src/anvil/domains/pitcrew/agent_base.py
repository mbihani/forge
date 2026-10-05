"""Code-mode agent base for the pitcrew domain.

The optimizer writes one :class:`PitcrewAgent` subclass in
``agents/pitcrew_agent.py``. Its ``predict`` decides, per PDF, how to call the
summarizer. The latency levers a strategy controls are the **model** (any of
``allowed_models``), ``max_tokens`` and the **system prompt**
(``composed_prompt``, which it may trim or restructure), statically or from a
cheap per-PDF signal (byte size, page count).

:meth:`PitcrewAgent.summarize` is the one call a subclass makes. The eval
times the whole ``predict`` (so extra calls cost latency) and prices
``meta["calls"]``.

Kept import-light (no gateway client at import time) so a subclass module
passes the optimizer's isolated-import validation.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from anvil.domains.pitcrew.summarizer import MAX_TOKENS, TEMPERATURE


class PitcrewAgent(ABC):
    """Base for a code-mode pitcrew summarization agent.

    One instance per round; ``predict`` is called once per PDF from several
    threads at once — keep per-PDF state local. ``model`` is the round's model
    (the ``model`` lever, else ``runtime_endpoint``); :meth:`summarize`
    rejects any model not in ``allowed_models``.
    """

    def __init__(
        self,
        composed_prompt: str,
        *,
        model: str,
        allowed_models: list[str] | None = None,
        client: Any | None = None,
    ) -> None:
        self.composed_prompt = composed_prompt
        self.model = model
        self.allowed_models = list(dict.fromkeys([*(allowed_models or []), model]))
        self._client = client
        self._client_lock = threading.Lock()

    def _gateway_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                from anvil.runtime.client import build_gateway_client  # noqa: PLC0415

                self._client = build_gateway_client()
            return self._client

    def summarize(
        self,
        pdf_bytes: bytes,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> tuple[str, dict[str, Any]]:
        """Summarize one PDF; return ``(text, {"calls": [call]})``.

        ``call`` = ``{model, input_tokens, output_tokens, latency_ms}``.
        Return the meta from ``predict`` (combine several with
        :meth:`merge_meta`) so the eval can price it.
        """
        from anvil.domains.pitcrew.summarizer import summarize_pdf  # noqa: PLC0415

        chosen = model or self.model
        if chosen not in self.allowed_models:
            raise ValueError(f"model {chosen!r} is not in allowed_models {self.allowed_models}")
        t0 = time.perf_counter()
        text, usage = summarize_pdf(
            self._gateway_client(),
            chosen,
            self.composed_prompt if system_prompt is None else system_prompt,
            pdf_bytes,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        call = {"model": chosen, **usage, "latency_ms": (time.perf_counter() - t0) * 1000.0}
        return text, {"calls": [call]}

    @staticmethod
    def merge_meta(*metas: dict[str, Any]) -> dict[str, Any]:
        """Concatenate the ``calls`` of several :meth:`summarize` metas."""
        return {"calls": [c for m in metas for c in (m or {}).get("calls", [])]}

    @abstractmethod
    def predict(self, *, pdf_bytes: bytes) -> tuple[str, dict[str, Any]]:
        """Return ``(summary_json_text, meta)`` for one FINRA rule PDF."""
