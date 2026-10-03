"""Pitcrew re-run — summarize one FINRA PDF through the AI Gateway.

FORGE's stack is the ``openai`` SDK pointed at the Databricks AI Gateway
(``build_gateway_client``), NOT the production app's ``ChatDatabricks``
chain — which cannot be imported here (``databricks-langchain`` is not in
the offline lock). The one LLM call pitcrew's chain makes is reproduced
faithfully: the mutated system prompt (composed from the scaffold) plus a
single user turn carrying the PDF as an Anthropic-style base64 ``document``
content block. That block is accepted by ``databricks-claude-sonnet-4-6``
through the gateway's OpenAI-compatible route (verified live), so the
re-run matches the production ``direct_pdf`` path (``server/pdf_handler.py``).

Decode params are fixed module constants (temperature 0, generous
max_tokens): in prompt mode the optimizer mutates the PROMPT, so holding
decode fixed keeps every round's score delta attributable to the prompt
edit rather than sampling noise.
"""

from __future__ import annotations

import base64
import importlib.util
import re
from typing import Any

# Values of the domain's ``input_mode`` lever (harness/config.yaml > levers).
#   direct_pdf — the PDF as a base64 ``document`` block (production default).
#   text       — text extracted locally first; the model reads plain text
#                instead of rendered pages (far fewer input tokens).
INPUT_MODES = ("direct_pdf", "text")
DEFAULT_INPUT_MODE = "direct_pdf"

# Fixed decode params for the summarizer re-run. Temperature 0 for a
# deterministic re-run; max_tokens matches the production app
# (``prompt_registry.py``, 32000) — a lower cap truncates the largest rules'
# structured JSON mid-object, which silently fails the json-valid gate.
TEMPERATURE = 0.0
MAX_TOKENS = 32000

_USER_INSTRUCTION = (
    "Please analyze the PDF document above and produce a structured summary."
)
_TEXT_INSTRUCTION = (
    "Please analyze the following document text and produce a structured summary."
)


def text_mode_available() -> bool:
    """True when the PDF text extractor (pymupdf) is importable."""
    return importlib.util.find_spec("fitz") is not None


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Page-marked text, mirroring production ``extract_text_from_pdf``.

    ``[Page N]`` markers keep page citations possible; the trailing
    amendment-history block is stripped like the production fallback.
    """
    try:
        import fitz  # noqa: PLC0415 - pymupdf, optional (text input mode only)
    except ImportError as exc:
        raise RuntimeError(
            "input_mode=text needs pymupdf (`uv pip install pymupdf`)"
        ) from exc
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = [
            f"[Page {i + 1}]\n{text.strip()}"
            for i in range(len(doc))
            if (text := doc[i].get_text("text")).strip()
        ]
    finally:
        doc.close()
    full = "\n\n".join(pages)
    return re.split(r"\nAmended by SR-", full, maxsplit=1)[0].strip()


def _pdf_document_block(pdf_bytes: bytes) -> dict[str, Any]:
    """Anthropic-style base64 PDF ``document`` content block.

    Mirrors ``server/pdf_handler.py::_prepare_direct_pdf`` so the re-run
    sends the same input shape the production app does.
    """
    encoded = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": encoded,
        },
    }


def summarize(
    client: Any,
    model: str,
    system_prompt: str,
    pdf_bytes: bytes,
    *,
    input_mode: str = DEFAULT_INPUT_MODE,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> tuple[str, dict[str, int]]:
    """Return ``(summary_text, usage)`` for one PDF.

    ``usage`` is ``{"input_tokens", "output_tokens"}`` from the response
    (zeros when the gateway reports none) — the engine prices it into
    ``cost_usd``. ``client`` is a gateway client (``build_gateway_client``).
    """
    if input_mode == "text":
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"{_TEXT_INSTRUCTION}\n\n{extract_pdf_text(pdf_bytes)}"}
        ]
    elif input_mode == "direct_pdf":
        content = [
            _pdf_document_block(pdf_bytes),
            {"type": "text", "text": _USER_INSTRUCTION},
        ]
    else:
        raise ValueError(f"unknown input_mode {input_mode!r}; expected one of {INPUT_MODES}")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    usage = getattr(response, "usage", None)
    tokens = {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    return response.choices[0].message.content or "", tokens


def summarize_pdf(
    client: Any,
    model: str,
    system_prompt: str,
    pdf_bytes: bytes,
    *,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Return the model's raw summary text for one PDF (direct_pdf mode)."""
    text, _usage = summarize(
        client,
        model,
        system_prompt,
        pdf_bytes,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return text
