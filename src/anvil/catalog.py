"""Model price list — per-model token prices for the optimizer and cost metrics.

Prices come from an operator-maintained Google Sheet that is SYNCED INTO THE
REPO (``scripts/sync_model_catalog.py`` → ``harness/model_catalog.csv``). The
loop reads only that committed file: rounds never call Google (the sandbox has
no Google auth), and a sheet edit mid-run cannot change prices between rounds.
Re-run the sync when the sheet changes.

The sheet keys rows by human model-family names (``Claude Sonnet 4.5 / 4.6``),
not FMAPI endpoint names (``databricks-claude-sonnet-4-6``). Each family is
expanded into endpoint-style slugs (``claude-sonnet-4-5``, ``claude-sonnet-4-6``)
and an endpoint is matched against them (see :func:`lookup_price`). Models the
sheet does not cover fall back to LiteLLM's bundled table (the source MLflow
uses for trace cost) under its ``databricks/<name>`` key.

Latency is deliberately NOT here: no price source publishes it. It is measured
by the eval and summarized per lever value from past round records.

Everything except :func:`litellm_price` is pure and offline-testable.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Sheet header names (the operator's sheet) → our field names.
_HEADER_MAP = {
    "model": "family",
    "price/input token ($/1m)": "input_per_million",
    "price/output token ($/1m)": "output_per_million",
    "cache read ($/1m)": "cache_read_per_million",
    "provider": "provider",
    "context": "context",
    "notes": "notes",
}

CSV_FIELDS = (
    "family",
    "context",
    "input_per_million",
    "output_per_million",
    "cache_read_per_million",
    "provider",
    "notes",
)

_ENDPOINT_PREFIX = "databricks-"


@dataclass(frozen=True)
class CatalogEntry:
    """One priced row: a model family at one context tier."""

    family: str
    context: str
    input_per_million: float
    output_per_million: float | None
    cache_read_per_million: float | None = None
    provider: str = ""
    notes: str = ""

    @property
    def aliases(self) -> tuple[str, ...]:
        return expand_family(self.family)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_sheet_rows(values: list[list[Any]]) -> list[CatalogEntry]:
    """Turn raw sheet values (rows of cells) into priced entries.

    Finds the header row by its ``Model`` + input-price columns, maps columns
    by header name (so column order can change), and keeps only rows with a
    numeric input price — notes/footer rows are skipped. Raises ``ValueError``
    when no header row is found (the sheet layout changed).
    """
    header_idx = None
    col: dict[str, int] = {}
    for i, row in enumerate(values):
        names = [str(c).strip().lower() for c in row]
        if "model" in names and "price/input token ($/1m)" in names:
            header_idx = i
            col = {_HEADER_MAP[n]: j for j, n in enumerate(names) if n in _HEADER_MAP}
            break
    if header_idx is None:
        raise ValueError(
            "model price sheet: no header row with 'Model' and 'Price/Input Token ($/1M)'"
        )

    def cell(row: list[Any], field: str) -> Any:
        j = col.get(field)
        return row[j] if j is not None and j < len(row) else ""

    entries: list[CatalogEntry] = []
    for row in values[header_idx + 1 :]:
        family = str(cell(row, "family")).strip()
        price_in = _to_float(cell(row, "input_per_million"))
        if not family or price_in is None:
            continue
        entries.append(
            CatalogEntry(
                family=family,
                context=str(cell(row, "context")).strip(),
                input_per_million=price_in,
                output_per_million=_to_float(cell(row, "output_per_million")),
                cache_read_per_million=_to_float(cell(row, "cache_read_per_million")),
                provider=str(cell(row, "provider")).strip(),
                notes=str(cell(row, "notes")).strip(),
            )
        )
    return entries


def _tokens(text: str) -> list[str]:
    """Lowercase words; hyphens/underscores split words (``GPT-5.4`` → gpt, 5.4)."""
    return re.sub(r"[-_]+", " ", text.lower()).split()


def _split_version(tokens: list[str]) -> tuple[list[str], str | None, list[str]]:
    """(prefix words, version token, suffix words); version = first digit-led token."""
    for i, tok in enumerate(tokens):
        if tok[:1].isdigit():
            return tokens[:i], tok, tokens[i + 1 :]
    return tokens, None, []


def _slug(words: list[str]) -> str:
    return "-".join(w.replace(".", "-") for w in words if w)


def expand_family(family: str) -> tuple[str, ...]:
    """Expand a sheet family name into endpoint-style slugs.

    ``Claude Opus 4.5 / 4.6``  → claude-opus-4-5, claude-opus-4-6
    ``GPT-5.4 Pro / 5.5 Pro``  → gpt-5-4-pro, gpt-5-5-pro
    ``Gemini 3.0 / 3.1 Pro``   → gemini-3-0-pro, gemini-3-pro, gemini-3-1-pro

    A bare-version alternative (``4.6``) inherits the first alternative's
    prefix words; an alternative with no suffix inherits the LAST
    alternative's suffix (``Gemini 3.0 / 3.1 Pro``: both are Pro). A ``.0``
    version also yields the short form (``3.0`` → ``3``). Parenthesized
    qualifiers (``(embedding)``) are dropped.
    """
    clean = re.sub(r"\(.*?\)", " ", family)
    alts = [a.strip() for a in clean.split("/") if a.strip()]
    if not alts:
        return ()
    first_prefix, _, _ = _split_version(_tokens(alts[0]))
    _, _, last_suffix = _split_version(_tokens(alts[-1]))
    slugs: list[str] = []
    for alt in alts:
        prefix, version, suffix = _split_version(_tokens(alt))
        if not prefix:
            prefix = first_prefix
        if version is not None and not suffix:
            suffix = last_suffix
        versions = [version] if version is not None else [None]
        if version is not None and version.endswith(".0"):
            versions.append(version[:-2])
        for v in versions:
            slug = _slug([*prefix, *([v] if v else []), *suffix])
            if slug and slug not in slugs:
                slugs.append(slug)
    return tuple(slugs)


def endpoint_slug(model: str) -> str:
    """``databricks-claude-sonnet-4-6`` → ``claude-sonnet-4-6``."""
    name = model.strip().lower()
    if name.startswith(_ENDPOINT_PREFIX):
        name = name[len(_ENDPOINT_PREFIX) :]
    return _slug(_tokens(name))


def _tier_rank(context: str, tier: str) -> int:
    """Lower is better: exact tier first, then 'all lengths', then anything."""
    c = context.lower()
    if tier and c.startswith(tier.lower()):
        return 0
    if c.startswith("all"):
        return 1
    return 2


def _alias_matches(alias: str, slug: str) -> bool:
    """Exact, or alias is a token-run inside slug not followed by a version digit.

    ``llama-3-3-70b`` matches ``meta-llama-3-3-70b-instruct``, but
    ``claude-sonnet-4`` must NOT match ``claude-sonnet-4-5`` (the next token
    is a version number — a different model). A longer all-digit token is a
    release-date code, not a version (``deepseek-v4-flash-0731``), so it does
    not block the match.
    """
    if alias == slug:
        return True
    s_toks, a_toks = slug.split("-"), alias.split("-")
    n = len(a_toks)
    for i in range(len(s_toks) - n + 1):
        if s_toks[i : i + n] == a_toks:
            nxt = s_toks[i + n] if i + n < len(s_toks) else ""
            is_version = nxt.isdigit() and len(nxt) <= 2
            if not is_version:
                return True
    return False


def lookup_price(
    model: str, entries: list[CatalogEntry], *, tier: str = "short"
) -> CatalogEntry | None:
    """Best catalog row for an FMAPI model name, or None.

    Prefers an exact alias match over a substring match, a longer (more
    specific) alias over a shorter one, then the requested context ``tier``
    (``short`` / ``long``) over ``All lengths``.
    """
    slug = endpoint_slug(model)
    best: tuple[tuple[int, int, int], CatalogEntry] | None = None
    for entry in entries:
        for alias in entry.aliases:
            if not _alias_matches(alias, slug):
                continue
            key = (0 if alias == slug else 1, -len(alias), _tier_rank(entry.context, tier))
            if best is None or key < best[0]:
                best = (key, entry)
    return best[1] if best else None


def litellm_price(model: str) -> dict[str, float] | None:
    """Fallback: LiteLLM's bundled table (MLflow's trace-cost source).

    Keyed ``databricks/<endpoint>``; frozen at the installed litellm version
    and list-priced (not workspace rates). None when litellm is absent or the
    model is not in its table.
    """
    try:
        import litellm  # noqa: PLC0415
    except ImportError:
        return None
    entry = (getattr(litellm, "model_cost", None) or {}).get(f"databricks/{model}")
    if not entry or entry.get("input_cost_per_token") is None:
        return None
    out = entry.get("output_cost_per_token")
    return {
        "input_per_million": float(entry["input_cost_per_token"]) * 1e6,
        "output_per_million": float(out) * 1e6 if out is not None else float("nan"),
    }


def model_prices(
    models: list[str], entries: list[CatalogEntry], *, tier: str = "short"
) -> dict[str, dict[str, Any]]:
    """Price each model: the synced sheet first, LiteLLM as fallback.

    Returns ``{model: {input_per_million, output_per_million, source, family}}``;
    a model with no price anywhere is omitted (the prompt shows it as unknown).
    """
    prices: dict[str, dict[str, Any]] = {}
    for model in models:
        entry = lookup_price(model, entries, tier=tier)
        if entry is not None:
            prices[model] = {
                "input_per_million": entry.input_per_million,
                "output_per_million": entry.output_per_million,
                "source": "sheet",
                "family": entry.family,
                "context": entry.context,
            }
            continue
        fallback = litellm_price(model)
        if fallback is not None:
            prices[model] = {**fallback, "source": "litellm"}
    return prices


def cost_usd(price: dict[str, Any] | None, input_tokens: int, output_tokens: int) -> float | None:
    """Dollar cost of one call from per-1M prices; None when unpriced."""
    if not price:
        return None
    p_in = price.get("input_per_million")
    p_out = price.get("output_per_million")
    if p_in is None or p_out is None or p_out != p_out:  # NaN check
        return None
    return (input_tokens * float(p_in) + output_tokens * float(p_out)) / 1e6


def fetch_sheet_values(
    sheet_id: str,
    sheet_range: str,
    *,
    token: str,
    quota_project: str | None = None,
) -> list[list[Any]]:
    """Read raw cell values via the Google Sheets API (unformatted numbers).

    ``token`` is a Google OAuth access token (e.g. gcloud ADC);
    ``quota_project`` is sent as ``x-goog-user-project`` when the token's
    project needs one. Only the sync script calls this — never the loop.
    """
    import json  # noqa: PLC0415
    import urllib.parse  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    url = (
        f"https://sheets.googleapis.com/v4/spreadsheets/{urllib.parse.quote(sheet_id)}"
        f"/values/{urllib.parse.quote(sheet_range)}?valueRenderOption=UNFORMATTED_VALUE"
    )
    headers = {"Authorization": f"Bearer {token}"}
    if quota_project:
        headers["x-goog-user-project"] = quota_project
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https host
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("values", [])


def write_catalog_csv(path: Path | str, entries: list[CatalogEntry], *, header_note: str) -> Path:
    """Write the synced catalog. ``header_note`` (source + time) goes in a comment line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {header_note}\n")
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for e in entries:
            writer.writerow(
                {
                    "family": e.family,
                    "context": e.context,
                    "input_per_million": e.input_per_million,
                    "output_per_million": ""
                    if e.output_per_million is None
                    else e.output_per_million,
                    "cache_read_per_million": (
                        "" if e.cache_read_per_million is None else e.cache_read_per_million
                    ),
                    "provider": e.provider,
                    "notes": e.notes,
                }
            )
    return p


def load_catalog(path: Path | str) -> list[CatalogEntry]:
    """Read ``harness/model_catalog.csv``; ``[]`` when the file is absent."""
    p = Path(path)
    if not p.is_file():
        return []
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    entries: list[CatalogEntry] = []
    for row in csv.DictReader(lines):
        price_in = _to_float(row.get("input_per_million"))
        if not row.get("family") or price_in is None:
            continue
        entries.append(
            CatalogEntry(
                family=row["family"],
                context=row.get("context", ""),
                input_per_million=price_in,
                output_per_million=_to_float(row.get("output_per_million")),
                cache_read_per_million=_to_float(row.get("cache_read_per_million")),
                provider=row.get("provider", ""),
                notes=row.get("notes", ""),
            )
        )
    return entries
