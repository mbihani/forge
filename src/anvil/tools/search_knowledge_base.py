"""``search_knowledge_base`` tool: BM25 in-memory retriever over ``data/kb/``.

Schema (locked, must stay identical when a UC Function + Vector
Search backend swaps in for Phase 4)::

    search_knowledge_base(query: str, k: int = 3)
        -> list[{doc_id, title, snippet}]

Backend = ``"bm25"`` for the demo: zero dependencies, fully offline,
deterministic. ``"vector_search"`` raises ``NotImplementedError`` and
is the documented Phase-4 plug-in point.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlflow
import yaml
from mlflow.entities import SpanType

from anvil.runtime.agent import ToolExecutor

SEARCH_TOOL_NAME = "search_knowledge_base"
DEFAULT_K = 3
SNIPPET_CHAR_LIMIT = 500
EMPTY_RESULT_TEXT = "No matching policy documents."

_BM25_K1 = 1.5
_BM25_B = 0.75
_MIN_TOKEN_LEN = 3
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FRONTMATTER_DELIM = "---"

Backend = Literal["bm25", "vector_search"]


@dataclass(frozen=True)
class KbHit:
    doc_id: str
    title: str
    snippet: str
    score: float


@dataclass(frozen=True)
class _IndexedDoc:
    doc_id: str
    title: str
    body: str
    tokens: list[str]
    token_counts: Counter[str]
    length: int


def _strip_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file's optional YAML frontmatter from its body."""
    if not text.startswith(_FRONTMATTER_DELIM):
        return {}, text
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip() != _FRONTMATTER_DELIM:
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].rstrip() == _FRONTMATTER_DELIM:
            fm_text = "".join(lines[1:i])
            body = "".join(lines[i + 1 :]).lstrip("\n")
            fm = yaml.safe_load(fm_text) or {}
            if not isinstance(fm, dict):
                fm = {}
            return fm, body
    return {}, text


def _tokenise(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) >= _MIN_TOKEN_LEN]


def _load_kb_index(kb_dir: Path) -> list[_IndexedDoc]:
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"KB directory not found: {kb_dir}")

    docs: list[_IndexedDoc] = []
    for path in sorted(kb_dir.glob("*.md")):
        fm, body = _strip_frontmatter(path.read_text(encoding="utf-8"))
        doc_id = fm.get("doc_id") if isinstance(fm.get("doc_id"), str) else path.stem
        title = fm.get("title") if isinstance(fm.get("title"), str) else doc_id
        tokens = _tokenise(f"{title}\n{body}")
        if not tokens:
            continue
        docs.append(
            _IndexedDoc(
                doc_id=doc_id,
                title=title,
                body=body.strip(),
                tokens=tokens,
                token_counts=Counter(tokens),
                length=len(tokens),
            )
        )
    if not docs:
        raise ValueError(f"KB directory has no usable markdown docs: {kb_dir}")
    return docs


def _bm25_scores(query_tokens: list[str], docs: list[_IndexedDoc]) -> list[float]:
    n_docs = len(docs)
    avgdl = sum(d.length for d in docs) / n_docs

    df: dict[str, int] = {}
    for term in set(query_tokens):
        df[term] = sum(1 for d in docs if term in d.token_counts)

    scores: list[float] = []
    for d in docs:
        score = 0.0
        norm = 1 - _BM25_B + _BM25_B * d.length / avgdl
        for term in query_tokens:
            tf = d.token_counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(((n_docs - df[term] + 0.5) / (df[term] + 0.5)) + 1.0)
            score += idf * (tf * (_BM25_K1 + 1)) / (tf + _BM25_K1 * norm)
        scores.append(score)
    return scores


def _make_snippet(body: str) -> str:
    body = body.strip()
    if len(body) <= SNIPPET_CHAR_LIMIT:
        return body
    return body[:SNIPPET_CHAR_LIMIT].rstrip() + "..."


def _search(docs: list[_IndexedDoc], query: str, k: int) -> list[KbHit]:
    query_tokens = _tokenise(query)
    if not query_tokens:
        return []
    scores = _bm25_scores(query_tokens, docs)
    ranked = sorted(
        ((score, doc) for score, doc in zip(scores, docs, strict=False) if score > 0),
        key=lambda pair: pair[0],
        reverse=True,
    )
    hits: list[KbHit] = []
    for score, doc in ranked[:k]:
        hits.append(
            KbHit(
                doc_id=doc.doc_id,
                title=doc.title,
                snippet=_make_snippet(doc.body),
                score=score,
            )
        )
    return hits


def format_hits(hits: list[KbHit]) -> str:
    if not hits:
        return EMPTY_RESULT_TEXT
    blocks: list[str] = []
    for h in hits:
        blocks.append(f"=== doc_id: {h.doc_id} ===\ntitle: {h.title}\n\n{h.snippet}")
    return "\n\n".join(blocks)


class _KbToolExecutor:
    """Callable that dispatches ``search_knowledge_base`` calls."""

    def __init__(self, docs: list[_IndexedDoc]) -> None:
        self._docs = docs

    def __call__(self, name: str, arguments_json: str) -> str:
        if name != SEARCH_TOOL_NAME:
            raise RuntimeError(
                f"KbToolExecutor cannot dispatch tool {name!r}. "
                f"This executor only handles {SEARCH_TOOL_NAME!r}."
            )
        args = json.loads(arguments_json) if arguments_json else {}
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"{SEARCH_TOOL_NAME}: 'query' is required and must be a non-empty string"
            )
        k_raw = args.get("k", DEFAULT_K)
        try:
            k = int(k_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{SEARCH_TOOL_NAME}: 'k' must be an integer (got {k_raw!r})") from exc
        if k <= 0:
            raise ValueError(f"{SEARCH_TOOL_NAME}: 'k' must be > 0 (got {k})")
        # Emit a RETRIEVER span so MLflow's RetrievalGroundedness
        # scorer (and any future trace-level citation extractor) can
        # read the retrieved chunks. ``mlflow.start_span`` is a no-op
        # outside an active trace in MLflow 3.10.
        # NOTE: ``extract_retrieval_context_from_trace`` reads
        # ``chunk["metadata"]["doc_uri"]`` (NOT ``doc_id``) for source
        # attribution — keep the key as ``doc_uri``.
        with mlflow.start_span(name=SEARCH_TOOL_NAME, span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": query, "k": k})
            hits = _search(self._docs, query, k)
            span.set_outputs(
                [
                    {
                        "page_content": h.snippet,
                        "metadata": {
                            "doc_uri": h.doc_id,
                            "title": h.title,
                            "score": h.score,
                        },
                    }
                    for h in hits
                ]
            )
        return format_hits(hits)


def make_kb_executor(kb_dir: Path | str, backend: Backend = "bm25") -> ToolExecutor:
    """Build a ToolExecutor that dispatches ``search_knowledge_base`` calls."""
    if backend == "vector_search":
        raise NotImplementedError(
            "vector_search backend lands in Phase 4. "
            "Use backend='bm25' for now — schema is identical, swap is later transparent."
        )
    if backend != "bm25":
        raise ValueError(f"Unknown backend: {backend!r}")
    docs = _load_kb_index(Path(kb_dir))
    return _KbToolExecutor(docs)
