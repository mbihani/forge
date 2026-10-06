"""Savesage ICICI extraction agent (code-mode), round 6.

Round 6 change (on top of round 5 below): after the single extraction call,
attribute the account-level credit limits to the cards. ICICI prints ONE
credit limit / available credit limit per statement account; every card on
the account (primary + add-ons) shares it and no per-card limit is printed.
So each card's bigPicture.cardCreditLimit / cardAvailableCreditLimit is set
to statementLevelSummary.totalCreditLimit / availableCreditLimit whenever
those statement-level values were extracted. Pure local post-processing: no
extra call, no extra document send, no cost/latency change.

Round 5 notes:

Parent (round 1): GPT-5.6 Luna, native PDF, reasoning_effort="low", 96K
max_tokens, one call per statement.

Change: before the single call, drop the trailing ICICI boilerplate pages
(MITC / "Important information" / fee & MAD illustration pages) and send a
trimmed PDF holding only the statement pages that come before them. ICICI
statements average ~8 pages, and ~83% of those pages are this fixed
boilerplate. It carries no statement data, and its illustration tables list
fake transactions that act as distractors. Fewer pages means fewer input
tokens (cost) and less prefill/reasoning (latency).

Still exactly one send per statement: either the trimmed PDF or, if the trim
can't be done safely (no marker, poppler missing, any error), the original
PDF. Never both. The trimmed file keeps the original basename, so the
co-brand token in the filename is unchanged.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from anvil.domains.savesage.agent_base import (
    DEFAULT_MAX_TOKENS,
    SavesageAgent,
)

# A page that opens the boilerplate block; it and every later page are dropped.
_BOILERPLATE_START = re.compile(
    r"MOST\s+IMPORTANT\s+TERMS\s+AND\s+CONDITIONS|IMPORTANT\s+INFORMATION\s+ON\s+YOUR\s+CREDIT\s+CARD",
    re.IGNORECASE,
)


def _page_texts(pdf_path: Path) -> list[str]:
    exe = shutil.which("pdftotext")
    if exe is None:
        return []
    out = subprocess.run(
        [exe, "-enc", "UTF-8", str(pdf_path), "-"],
        capture_output=True,
        check=True,
        timeout=30,
    )
    pages = out.stdout.decode("utf-8", errors="replace").split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return pages


def _keep_count(pages: list[str]) -> int:
    """Number of leading pages to keep (0 = don't trim)."""
    n = len(pages)
    for k in range(1, n):  # page 1 is always kept
        if _BOILERPLATE_START.search(pages[k]):
            return k
    return 0


def _write_trimmed(pdf_path: Path, keep: int, workdir: Path) -> Path | None:
    sep = shutil.which("pdfseparate")
    unite = shutil.which("pdfunite")
    if sep is None or unite is None:
        return None
    parts = workdir / "parts"
    parts.mkdir()
    subprocess.run(
        [sep, "-f", "1", "-l", str(keep), str(pdf_path), str(parts / "pg-%d.pdf")],
        capture_output=True,
        check=True,
        timeout=30,
    )
    files = [parts / f"pg-{i}.pdf" for i in range(1, keep + 1)]
    if not all(f.is_file() and f.stat().st_size > 0 for f in files):
        return None
    out = workdir / pdf_path.name
    if keep == 1:
        shutil.copyfile(files[0], out)
    else:
        subprocess.run(
            [unite, *map(str, files), str(out)],
            capture_output=True,
            check=True,
            timeout=30,
        )
    if not out.is_file() or out.stat().st_size == 0:
        return None
    return out


def _share_account_limits(parsed: dict) -> dict:
    """Copy the statement-level (account) limits onto every card's bigPicture."""
    if not isinstance(parsed, dict):
        return parsed
    root = parsed.get("parsed_json") if isinstance(parsed.get("parsed_json"), dict) else parsed
    summary = root.get("statementLevelSummary")
    cards = root.get("cards")
    if not isinstance(summary, dict) or not isinstance(cards, list):
        return parsed
    pairs = (
        ("cardCreditLimit", summary.get("totalCreditLimit")),
        ("cardAvailableCreditLimit", summary.get("availableCreditLimit")),
    )
    for card in cards:
        if not isinstance(card, dict):
            continue
        bp = card.get("bigPicture")
        if not isinstance(bp, dict):
            bp = {}
            card["bigPicture"] = bp
        for key, value in pairs:
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            bp[key] = value
    return parsed


class TrimmedPdfSavesageAgent(SavesageAgent):
    """Low-effort Luna on a PDF trimmed of the trailing ICICI boilerplate pages."""

    REASONING_EFFORT = "low"

    def predict(self, *, sid: str, pdf_path: str) -> tuple[dict, dict]:
        src = Path(pdf_path)
        with tempfile.TemporaryDirectory(prefix="ss-trim-") as tmp:
            send_path: Path = src
            try:
                pages = _page_texts(src)
                keep = _keep_count(pages)
                if 0 < keep < len(pages):
                    trimmed = _write_trimmed(src, keep, Path(tmp))
                    if trimmed is not None:
                        send_path = trimmed
            except Exception:  # noqa: BLE001 - any trim failure -> send the original once
                send_path = src
            parsed, meta = self.extract(
                sid=sid,
                pdf_path=send_path,
                reasoning_effort=self.REASONING_EFFORT,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
        try:
            parsed = _share_account_limits(parsed)
        except Exception:  # noqa: BLE001 - post-processing must never fail a row
            pass
        return parsed, meta
