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
from typing import Any

# Fixed decode params for the summarizer re-run. Temperature 0 for a
# deterministic re-run; max_tokens matches the production app
# (``prompt_registry.py``, 32000) — a lower cap truncates the largest rules'
# structured JSON mid-object, which silently fails the json-valid gate.
TEMPERATURE = 0.0
MAX_TOKENS = 32000

_USER_INSTRUCTION = (
    "Please analyze the PDF document above and produce a structured summary."
)


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


def summarize_pdf(
    client: Any,
    model: str,
    system_prompt: str,
    pdf_bytes: bytes,
    *,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Return the model's raw summary text for one PDF.

    ``client`` is a gateway client exposing ``chat.completions.create``
    (``build_gateway_client``). The caller scores the returned text.
    """
    content = [
        _pdf_document_block(pdf_bytes),
        {"type": "text", "text": _USER_INSTRUCTION},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
