"""Savesage ICICI extraction agent (code-mode), round 6: text pass + output-free identity pass, wider grace.

Two concurrent low-effort Luna calls per statement:

* text pass: input_mode='text' over the full statement, used for every field
  (round 3: 12.9s median, all non-card fields at or above native PDF).
* identity pass: native PDF of page 1 only (pdfseparate, same filename) with
  the composed prompt plus a suffix that asks for EMPTY transaction arrays.
  Output is just statementMeta + card blocks.

Round 4 (wait for identity unconditionally, identity still listed page-1
transactions): accuracy 0.9864, median 16.0s. Round 5 (this structure, 2.5s
grace): median 13.1s but cardDisplayName misses rose 7 -> 17 rows, i.e. the
identity pass usually lands just after the text pass and a 2.5s grace cut it
off on ~10 rows. This round widens the grace to 6s so those rows get their
card identity back while the wait stays bounded.

Card identity (cardDisplayName / network / productFamily) and non-null card
limits are copied from the identity pass by last-four match. If the text
result fails or is empty, a native-PDF full extraction (low, then medium)
runs instead.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anvil.domains.savesage.agent_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT,
    NATIVE_PDF_MODELS,
    SavesageAgent,
)

FAST_EFFORT = 'low'
PDF_MODEL = 'databricks-gpt-5-6-luna'
IDENTITY_MAX_TOKENS = 32_000
IDENTITY_GRACE_S = 6.0

IDENTITY_SUFFIX = (
    '\n\n## IDENTITY-ONLY PASS (overrides any earlier instruction about listing transactions)\n'
    'This request is a fast identity pass over the first page of the statement only. '
    'Fill statementMeta and, for every card, cardMeta (cardDisplayName, lastFourDigit, '
    'network, productFamily) and bigPicture (credit limit, available credit limit) exactly '
    'as the rules above describe, reading the card name or wordmark, the network logo and '
    'the limits from the page. Every transactions array MUST be empty: []. Do not list any '
    'transaction. Use null for any other field not visible on this page.\n'
)


def _looks_empty(parsed) -> bool:
    if not isinstance(parsed, dict) or not parsed:
        return True
    return not parsed.get('statementMeta') and not parsed.get('statementLevelSummary')


def _cards(parsed) -> list:
    cards = parsed.get('cards') if isinstance(parsed, dict) else None
    if not isinstance(cards, list):
        return []
    return [c for c in cards if isinstance(c, dict)]


def _last4(card):
    meta = card.get('cardMeta')
    if not isinstance(meta, dict):
        return None
    v = meta.get('lastFourDigit')
    if v is None:
        return None
    s = ''.join(ch for ch in str(v) if ch.isdigit())
    return s[-4:] if len(s) >= 4 else None


def _present(v) -> bool:
    return v is not None and v != ''


def _merge_identity(base: dict, ident: dict) -> dict:
    bcards, icards = _cards(base), _cards(ident)
    if not bcards or not icards:
        return base
    by4 = {}
    for c in icards:
        k = _last4(c)
        if k and k not in by4:
            by4[k] = c
    for bc in bcards:
        k = _last4(bc)
        src = by4.get(k) if k else None
        if src is None and len(bcards) == 1 and len(icards) == 1:
            src = icards[0]
        if src is None:
            continue
        bm, sm = bc.get('cardMeta'), src.get('cardMeta')
        if isinstance(bm, dict) and isinstance(sm, dict):
            for f in ('cardDisplayName', 'network', 'productFamily'):
                if _present(sm.get(f)):
                    bm[f] = sm[f]
        bb, sb = bc.get('bigPicture'), src.get('bigPicture')
        if isinstance(sb, dict):
            if not isinstance(bb, dict):
                bb = {}
                bc['bigPicture'] = bb
            for f in ('cardCreditLimit', 'cardAvailableCreditLimit'):
                if _present(sb.get(f)):
                    bb[f] = sb[f]
    return base


def _first_page(pdf_path, tmpdir: str):
    exe = shutil.which('pdfseparate')
    if exe is None:
        return None
    out = Path(tmpdir) / Path(pdf_path).name
    subprocess.run(
        [exe, '-f', '1', '-l', '1', str(pdf_path), str(out)],
        check=True,
        capture_output=True,
        timeout=20,
    )
    if out.is_file() and out.stat().st_size > 0:
        return out
    return None


class TextPlusIdentitySavesageAgent(SavesageAgent):
    """Low-effort text extraction, with card identity from an output-free page-1 PDF pass."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._id_lock = threading.Lock()
        self._id_extractor = None

    def _identity_extractor(self):
        with self._id_lock:
            if self._id_extractor is None:
                from anvil.domains.savesage.extractor import SavesageIciciExtractor

                self._id_extractor = SavesageIciciExtractor(
                    self.composed_prompt + IDENTITY_SUFFIX,
                    cache_root=None,
                    luna_profile=self._luna_profile,
                    reasoning_effort=FAST_EFFORT,
                    max_tokens=IDENTITY_MAX_TOKENS,
                    model=PDF_MODEL,
                    input_mode='pdf',
                )
            return self._id_extractor

    def _identity_pass(self, sid, pdf_path):
        tmpdir = tempfile.mkdtemp(prefix='ss_id_')
        try:
            src = pdf_path
            try:
                p1 = _first_page(pdf_path, tmpdir)
                if p1 is not None:
                    src = p1
            except Exception:
                src = pdf_path
            parsed, latency_ms, usage = self._identity_extractor().extract_with_usage(
                sid=sid, pdf_path=src
            )
            call = {
                'model': PDF_MODEL,
                'input_mode': 'pdf',
                'reasoning_effort': FAST_EFFORT,
                'max_tokens': IDENTITY_MAX_TOKENS,
                'latency_ms': latency_ms,
                **usage,
            }
            return parsed, call
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _text_pass(self, sid, pdf_path):
        return self.extract(
            sid=sid,
            pdf_path=pdf_path,
            input_mode='text',
            reasoning_effort=FAST_EFFORT,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

    def _pdf_fallback(self, sid, pdf_path, calls: list):
        model = PDF_MODEL if PDF_MODEL in self.allowed_models else self.model
        for effort in (FAST_EFFORT, DEFAULT_REASONING_EFFORT):
            try:
                parsed, meta = self.extract(
                    sid=sid,
                    pdf_path=pdf_path,
                    model=model,
                    input_mode='pdf',
                    reasoning_effort=effort,
                    max_tokens=DEFAULT_MAX_TOKENS,
                )
            except Exception:
                if effort == DEFAULT_REASONING_EFFORT:
                    raise
                continue
            calls.extend(meta.get('calls', []))
            if not _looks_empty(parsed) or effort == DEFAULT_REASONING_EFFORT:
                return parsed
        return {}

    def predict(self, *, sid: str, pdf_path: str) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        calls: list = []
        can_identity = PDF_MODEL in self.allowed_models and PDF_MODEL in NATIVE_PDF_MODELS
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            id_fut = pool.submit(self._identity_pass, sid, pdf_path) if can_identity else None
            text_fut = pool.submit(self._text_pass, sid, pdf_path)
            parsed = None
            try:
                parsed, tmeta = text_fut.result()
                calls.extend(tmeta.get('calls', []))
            except Exception:
                parsed = None
            if parsed is None or _looks_empty(parsed):
                parsed = self._pdf_fallback(sid, pdf_path, calls)
            elif id_fut is not None:
                try:
                    ident, call = id_fut.result(timeout=IDENTITY_GRACE_S)
                    calls.append(call)
                    if isinstance(ident, dict) and ident:
                        parsed = _merge_identity(parsed, ident)
                except Exception:
                    pass
        finally:
            pool.shutdown(wait=False)
        return parsed, {'latency_ms': (time.perf_counter() - t0) * 1000.0, 'calls': calls}
