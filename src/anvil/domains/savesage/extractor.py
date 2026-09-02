"""Runtime predict for the Savesage ICICI domain: compose prompt → Luna → JSON.

The ANVIL scaffold composes into one ICICI extraction prompt; this module
runs that prompt through the SAME Luna endpoint + transport the production
agent uses (``harness.extraction_adapter.LunaExtractionAdapter``), injecting
the composed prompt in place of the on-disk ``prompts/icici.txt``.

Two reuse seams keep this thin and faithful:

* :class:`_ComposedPromptLunaAdapter` subclasses the production adapter and
  overrides only ``extract`` to swap the prompt source. Everything else —
  auth, stdlib ``urllib`` transport, retry policy, response mapping,
  truncation/refusal guards — is inherited unchanged. The per-bank JSON
  **schema** is still the production ``schema/icici.json`` (loaded via
  ``rules.routing.load_schema_for_bank``), so only the prompt varies.
* A content-addressed Luna cache keyed by ``(sha256(prompt), sid)`` means an
  unchanged or reverted prompt never re-extracts — reverts and repeated sids
  are free, and a resumed round reuses prior work.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anvil.domains.savesage._statement_agent import ensure_importable

logger = logging.getLogger(__name__)

# Databricks profile Luna authenticates against. Luna lives on ONE workspace
# (fevm-stable) regardless of where the optimizer's gateway runs, so the
# extractor mints its own workspace-scoped token via this profile rather than
# reading the process-global DATABRICKS_TOKEN / DATABRICKS_CONFIG_PROFILE.
# That decoupling is deliberate: the ANVIL optimizer's Claude Code gateway
# auth ALSO consumes DATABRICKS_TOKEN, so a shared token env would send Luna's
# workspace token to the optimizer's (different) gateway and break it. Override
# with SAVESAGE_LUNA_PROFILE; set it empty to fall back to the statement-agent
# default env chain (DATABRICKS_TOKEN, then the SDK default profile).
_DEFAULT_LUNA_PROFILE = "fevm-stable"  # CONFIGURE(luna-profile)


def prompt_hash(prompt: str) -> str:
    """Short content hash of a composed prompt (cache key + trace tag)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _profile_token_provider(profile: str) -> Callable[[], str]:
    """A token provider that mints a bearer token for a specific DB profile.

    Uses the Databricks SDK ``Config(profile=...)`` so the token is scoped to
    ``profile`` independent of any process-global ``DATABRICKS_TOKEN`` /
    ``DATABRICKS_CONFIG_PROFILE`` (the SDK caches + refreshes internally, so a
    long batch does not re-mint on every call). Import is lazy so this module
    stays importable without the SDK.
    """

    def _provide() -> str:
        from databricks.sdk.core import Config  # noqa: PLC0415 - lazy

        headers = Config(profile=profile).authenticate()
        auth = headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise RuntimeError(f"profile {profile!r} did not yield a bearer token")
        return auth.removeprefix("Bearer ")

    return _provide


def _build_adapter(
    composed_prompt: str,
    token_provider: Callable[[], str] | None = None,
    *,
    reasoning_effort: str = "medium",
    max_tokens: int = 96_000,
):
    """Construct the prompt-injected Luna adapter (lazy import of the bridge).

    ``token_provider`` (when given) overrides the statement-agent default
    ``acquire_token`` so Luna authenticates to its own workspace regardless of
    the ambient token env — see :data:`_DEFAULT_LUNA_PROFILE`.

    ``reasoning_effort`` / ``max_tokens`` are the two latency levers the
    code-mode agent tunes. They override the production payload's hardcoded
    ``reasoning_effort="medium"`` / ``max_tokens=96_000`` for THIS adapter only
    (no edit to the shared ``statement-agent`` transport); prompt-mode callers
    keep the production defaults.
    """
    ensure_importable()
    from harness.extraction_adapter import (  # noqa: PLC0415 - lazy, see bridge
        ExtractionError,
        LunaExtractionAdapter,
        _read_pdf,
        map_response,
    )
    from harness.transports import extraction_payload  # noqa: PLC0415

    class _ComposedPromptLunaAdapter(LunaExtractionAdapter):
        """Luna adapter that injects a composed prompt instead of resolve_prompt.

        Only ``extract`` (+ the request build) differs from the parent: it
        builds the request with the composed prompt (and the production
        per-bank schema), then runs the parent's exact retry/auth/timeout
        loop. The loop is reproduced here rather than shared because the parent
        hardcodes ``resolve_prompt`` inside ``extract`` and exposes no
        prompt-injection seam; keeping the override self-contained is also
        thread-safe (no module-global patch), which matters because eval runs
        rows in a thread pool.
        """

        def __init__(self, prompt: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._composed_prompt = prompt
            self._reasoning_effort = reasoning_effort
            self._max_tokens = int(max_tokens)

        def _build_request(self, request, prompt, schema):  # type: ignore[override]
            # Mirror the parent's request build (same URL/headers) but override
            # the two latency knobs in the payload. extraction_payload takes
            # max_tokens as an arg and hardcodes reasoning_effort="medium", so
            # we set the max_tokens arg and patch reasoning_effort on the dict.
            settings = self._settings_obj()
            url = settings.endpoint_url(settings.extraction_endpoint)
            pdf = _read_pdf(request)
            payload = extraction_payload(
                pdf, request.filename, prompt, schema, max_tokens=self._max_tokens
            )
            payload["reasoning_effort"] = self._reasoning_effort
            body = json.dumps(payload).encode()
            token = self._token_provider()
            return urllib.request.Request(
                url,
                data=body,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST",
            )

        def extract(self, request):  # type: ignore[override]
            from rules.routing import load_schema_for_bank  # noqa: PLC0415

            schema = load_schema_for_bank(request.bank)
            req = self._build_request(request, self._composed_prompt, schema)
            timeout = self._policy.timeout_seconds

            last_error = ""
            t0 = time.perf_counter()
            for attempt in range(1, self._policy.max_attempts + 1):
                try:
                    with self._urlopen(req, timeout=timeout) as r:
                        raw = r.read().decode()
                    resp = json.loads(raw)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    return map_response(resp, request, latency_ms)
                except urllib.error.HTTPError as exc:
                    last_error = f"HTTP {exc.code}: {self._read_err(exc)[:500]}"
                    retryable = exc.code in self._policy.retry_statuses
                    if retryable and attempt < self._policy.max_attempts:
                        time.sleep(self._policy.backoff_for_attempt(attempt))
                        if exc.code in (401, 403):  # token expired mid-batch
                            token = self._token_provider()
                            req.add_header("Authorization", f"Bearer {token}")
                        continue
                    break
                except Exception as exc:  # noqa: BLE001 - timeout / socket reset
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < self._policy.max_attempts:
                        time.sleep(self._policy.backoff_for_attempt(attempt))
                        continue
                    break
            raise ExtractionError(
                f"extraction failed for {request.request_id} after "
                f"{self._policy.max_attempts} attempts: {last_error}"
            )

    kwargs = {} if token_provider is None else {"token_provider": token_provider}
    return _ComposedPromptLunaAdapter(composed_prompt, **kwargs)


def _read_pdf(pdf_path: str | Path) -> bytes:
    return Path(pdf_path).read_bytes()


class LunaCache:
    """Content-addressed cache of parsed Luna extractions.

    Layout: ``<root>/<prompt_hash>/<sid>.json`` holding the parsed_json
    dict. Keyed by the composed-prompt hash so a changed prompt gets a
    fresh namespace and an unchanged/reverted prompt hits every row.
    """

    def __init__(self, root: str | Path, composed_prompt: str) -> None:
        self._dir = Path(root) / prompt_hash(composed_prompt)

    def get(self, sid: str) -> dict | None:
        path = self._dir / f"{sid}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, sid: str, parsed_json: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / f"{sid}.json").write_text(
            json.dumps(parsed_json, ensure_ascii=False), encoding="utf-8"
        )


class SavesageIciciExtractor:
    """Compose-prompt → Luna extractor for one ICICI statement.

    ``composed_prompt`` is the text produced by ``compose_prompt`` over the
    scaffold. ``cache_root`` (when given) enables the prompt-addressed Luna
    cache. The extractor is stateless per call and safe to share across
    threads (the underlying adapter issues stateless HTTP calls with a
    per-request token).

    ``luna_profile`` selects the Databricks profile Luna authenticates against
    (default :data:`_DEFAULT_LUNA_PROFILE`, overridable via the
    ``SAVESAGE_LUNA_PROFILE`` env var). Pass an empty string to fall back to
    the statement-agent default token chain (``DATABRICKS_TOKEN`` / SDK default
    profile) — useful when the whole process already targets Luna's workspace.
    """

    def __init__(
        self,
        composed_prompt: str,
        *,
        cache_root: str | Path | None = None,
        luna_profile: str | None = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 96_000,
    ) -> None:
        self._prompt = composed_prompt
        profile = (
            luna_profile
            if luna_profile is not None
            else os.getenv("SAVESAGE_LUNA_PROFILE", _DEFAULT_LUNA_PROFILE)
        )
        token_provider = _profile_token_provider(profile) if profile else None
        self._adapter = _build_adapter(
            composed_prompt,
            token_provider=token_provider,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        self._cache = LunaCache(cache_root, composed_prompt) if cache_root else None

    def extract(self, *, sid: str, pdf_path: str | Path) -> dict:
        """Return the parsed extraction dict for one statement (cache-aware)."""
        if self._cache is not None:
            hit = self._cache.get(sid)
            if hit is not None:
                return hit

        ensure_importable()
        from contracts.models import Bank, ParseRequest  # noqa: PLC0415

        request = ParseRequest(
            pdf=_read_pdf(pdf_path),
            filename=Path(pdf_path).name,
            bank=Bank.ICICI,
            request_id=f"savesage-{sid}",
        )
        result = self._adapter.extract(request)
        parsed = result.payload if isinstance(result.payload, dict) else {}
        if self._cache is not None:
            self._cache.put(sid, parsed)
        return parsed

    def extract_with_latency(self, *, sid: str, pdf_path: str | Path) -> tuple[dict, float]:
        """Extract one statement and return ``(parsed, latency_ms)``.

        The code-mode / latency path: NEVER cache-served (a cache hit returns
        in ~0 ms and would fabricate the latency signal), so this always issues
        a live cold Luna call and reports its real wall-clock latency (measured
        by the adapter and carried on ``ExtractionResult.latency_ms``).
        """
        ensure_importable()
        from contracts.models import Bank, ParseRequest  # noqa: PLC0415

        request = ParseRequest(
            pdf=_read_pdf(pdf_path),
            filename=Path(pdf_path).name,
            bank=Bank.ICICI,
            request_id=f"savesage-{sid}",
        )
        result = self._adapter.extract(request)
        parsed = result.payload if isinstance(result.payload, dict) else {}
        return parsed, float(result.latency_ms)
