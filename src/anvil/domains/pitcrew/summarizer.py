"""One pitcrew summarize call through the AI Gateway.

Reproduces production's ``direct_pdf`` path (``server/pdf_handler.py``): the
composed system prompt plus one user turn carrying the PDF as a base64
content block and production's instruction. Claude models take an
Anthropic-style ``document`` block; GPT models reject it (400 "Invalid value:
'document'") and take an OpenAI ``file`` block instead.

Decode defaults match production (``prompt_registry.py``): ``max_tokens``
32000 — lower caps truncate the largest rules' JSON mid-object. Temperature 0
for a deterministic re-run; models that only accept the default temperature
(GPT reasoning models) are retried without it by the gateway client.
"""

from __future__ import annotations

import base64
from typing import Any

TEMPERATURE = 0.0
MAX_TOKENS = 32000

USER_INSTRUCTION = "Please analyze the PDF document above and produce a structured summary."


def pdf_document_block(pdf_bytes: bytes) -> dict[str, Any]:
    """Anthropic-style base64 PDF ``document`` block (production's shape)."""
    encoded = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": encoded},
    }


def pdf_file_block(pdf_bytes: bytes) -> dict[str, Any]:
    """OpenAI-style base64 PDF ``file`` block, for GPT models.

    The filename is a fixed placeholder so the rule number never reaches the
    model through it.
    """
    encoded = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    return {
        "type": "file",
        "file": {"filename": "document.pdf", "file_data": f"data:application/pdf;base64,{encoded}"},
    }


def pdf_content_block(pdf_bytes: bytes, model: str) -> dict[str, Any]:
    """The PDF block ``model`` accepts: ``file`` for GPT, else ``document``."""
    if "gpt" in model.lower():
        return pdf_file_block(pdf_bytes)
    return pdf_document_block(pdf_bytes)


def summarize_pdf(
    client: Any,
    model: str,
    system_prompt: str,
    pdf_bytes: bytes,
    *,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> tuple[str, dict[str, int]]:
    """Summarize one PDF; return ``(text, {"input_tokens", "output_tokens"})``.

    ``client`` exposes ``chat.completions.create`` (``build_gateway_client``).
    Token counts are 0 when the response carries no usage.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    pdf_content_block(pdf_bytes, model),
                    {"type": "text", "text": USER_INSTRUCTION},
                ],
            },
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
