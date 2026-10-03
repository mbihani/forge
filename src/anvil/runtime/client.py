"""Databricks AI Gateway client — the sole LLM route for FORGE.

All three LLM call paths (runtime agent, optimizer, judge) route through
a single OpenAI-compatible endpoint on the Databricks workspace. The
endpoint routes by FMAPI model name, so a single client serves all
models — no per-endpoint clients needed.

Base URL resolution (lowest to highest precedence):

* Default: ``https://<workspace_host>/serving-endpoints`` — the
  OpenAI-compatible route of standard Databricks FMAPI serving endpoints.
* ``ANVIL_GATEWAY_BASE_URL`` env var — used verbatim (only a trailing
  slash is stripped). Set this to point at a dedicated AI Gateway proxy
  on workspaces that have one.
* An explicit ``base_url=`` argument to ``build_gateway_client``.

The workspace host is read from ``DATABRICKS_HOST`` or, when unset, the
Databricks SDK config (``~/.databrickscfg`` profile /
``DATABRICKS_CONFIG_PROFILE``).

SP token refresh: Databricks SP tokens expire (~1h). The client refreshes
the token on each request to avoid expiry failures.

``build_gateway_client()`` builds the client; ``build_databricks_client()``
is kept as a backward-compatible wrapper that delegates to it.

The ``databricks`` SDK import (for token refresh when no static
``DATABRICKS_TOKEN`` is set) is lazy — inside the function — so importing
this module never requires the SDK to be installed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def _get_fresh_sp_token() -> str:
    """Get a fresh Databricks SP token.

    SP tokens expire (~1h), so we refresh on each call. Resolution order:

    1. ``DATABRICKS_TOKEN`` env var (static token; fine for testing/dev
       and for long-lived PATs).
    2. The Databricks SDK config chain (``~/.databrickscfg`` profile, env
       vars such as ``DATABRICKS_CONFIG_PROFILE``). The SDK mints a fresh
       OAuth token on demand.
    """
    token = os.environ.get("DATABRICKS_TOKEN")
    if token:
        return token
    # Lazy import so importing this module never requires the SDK. The
    # SDK is a core dependency, but keeping the import lazy also lets
    # the gateway-client unit tests run without exercising the SDK path.
    from databricks.sdk import WorkspaceClient

    ws = WorkspaceClient()
    # ``config.authenticate()`` returns a ``Dict[str, str]`` of header
    # name → header value (e.g. ``{"Authorization": "Bearer <token>"}``).
    # Resolving via ``config.token`` first covers the case where a static
    # PAT is configured on the profile.
    token = ws.config.token
    if not token:
        headers = ws.config.authenticate()  # Dict[str, str]
        auth = headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth.removeprefix("Bearer ")
    if not token:
        raise RuntimeError(
            "Could not resolve a Databricks token. Set DATABRICKS_TOKEN or "
            "configure a profile in ~/.databrickscfg."
        )
    return token


def _gateway_base_url() -> str:
    """Resolve the OpenAI-compatible base URL for LLM calls.

    This is only consulted when no explicit ``base_url`` was passed to
    ``build_gateway_client`` (an explicit argument always wins). Within
    this function:

    1. ``ANVIL_GATEWAY_BASE_URL`` env var — used verbatim (only a
       trailing slash is stripped). Set this to point at a dedicated AI
       Gateway proxy on workspaces that have one; it bypasses host
       resolution entirely.
    2. Default: ``https://<workspace_host>/serving-endpoints`` — the
       OpenAI-compatible route of standard Databricks FMAPI serving
       endpoints. The ``model`` parameter in ``chat.completions.create``
       selects which endpoint the request routes to.

    The workspace host is read from ``DATABRICKS_HOST``. When not set,
    falls back to the Databricks SDK config (which reads ~/.databrickscfg
    profiles and DATABRICKS_CONFIG_PROFILE).
    """
    env_url = os.environ.get("ANVIL_GATEWAY_BASE_URL", "")
    if env_url:
        # Verbatim override — only strip a trailing slash, append nothing.
        return env_url.rstrip("/")
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if not host:
        from databricks.sdk import WorkspaceClient

        ws = WorkspaceClient()
        host = str(ws.config.host or "").rstrip("/")
    if not host:
        raise RuntimeError(
            "Could not resolve Databricks host. Set DATABRICKS_HOST or "
            "configure a profile in ~/.databrickscfg."
        )
    return f"{host}/serving-endpoints"


class GatewayClient:
    """OpenAI-compatible client backed by the Databricks AI Gateway.

    Exposes ``chat.completions.create()`` with the same interface as
    ``openai.OpenAI``, but routes through the AI Gateway unified URL.
    Refreshes the SP token on each request to avoid expiry failures.

    The ``model`` parameter in ``chat.completions.create()`` should be an
    FMAPI model name (e.g. ``databricks-claude-sonnet-4-6``) — the gateway
    routes by that name, not by a per-endpoint URL.
    """

    class _Chat:
        class _Completions:
            def __init__(self, parent: GatewayClient) -> None:
                self._parent = parent

            def create(
                self,
                *,
                model: str,
                messages: list[dict[str, Any]],
                **kwargs: Any,
            ) -> Any:
                # Lazy import: ``openai`` is a project dependency, but
                # importing it here keeps the module side-effect-free and
                # lets tests stub ``openai.OpenAI`` without a real client.
                from openai import OpenAI

                # Fresh token on each call — SP tokens expire (~1h).
                token = self._parent._get_token()
                client = OpenAI(
                    api_key=token,
                    base_url=self._parent._base_url,
                )
                try:
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **kwargs,
                    )
                except Exception as exc:
                    # Newer models reject some sampling params (e.g. Claude
                    # Sonnet 5: "does not support the temperature parameter").
                    # A ``model`` lever can swap to such a model mid-run, so
                    # drop the rejected param(s) and retry ONCE rather than
                    # failing every row. Any other error propagates.
                    retry_kwargs = _drop_rejected_params(exc, kwargs)
                    if retry_kwargs is None:
                        raise
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **retry_kwargs,
                    )

        def __init__(self, parent: GatewayClient) -> None:
            self.completions = self._Completions(parent)

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token_fn: Callable[[], str] | None = None,
    ) -> None:
        self._base_url = base_url or _gateway_base_url()
        self._token_fn = token_fn or _get_fresh_sp_token
        self.chat = self._Chat(self)

    def _get_token(self) -> str:
        # Always refresh — SP tokens expire, and a long-running app can
        # outlive a token minted at construction time.
        return self._token_fn()


# Sampling params a model may reject outright. Dropping them only changes
# sampling, never the request's content, so a retry without them is safe.
_DROPPABLE_PARAMS = ("temperature", "top_p")


def _drop_rejected_params(exc: Exception, kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """kwargs minus the param(s) a 400 says the model does not support, else None.

    Matches the gateway's "does not support the <param> parameter" message.
    Returns None when the error is not that, or names no param we sent — the
    caller then re-raises the original error.
    """
    message = str(exc).lower()
    if "does not support" not in message:
        return None
    dropped = {p for p in _DROPPABLE_PARAMS if p in kwargs and f"{p} parameter" in message}
    if not dropped:
        return None
    return {k: v for k, v in kwargs.items() if k not in dropped}


def build_gateway_client(
    *,
    base_url: str | None = None,
    token_fn: Callable[[], str] | None = None,
) -> GatewayClient:
    """Build an AI Gateway client for LLM calls.

    The client is OpenAI-compatible (``chat.completions.create``) but
    routes through the AI Gateway unified URL with per-request SP token
    refresh. The ``model`` parameter should be an FMAPI model name.
    """
    return GatewayClient(base_url=base_url, token_fn=token_fn)


def build_databricks_client(
    profile: str | None = None,
    **kwargs: Any,
) -> GatewayClient:
    """Backward-compatible wrapper — delegates to :func:`build_gateway_client`.

    The gateway client resolves host and token from the environment /
    Databricks SDK config (``DATABRICKS_HOST``, ``~/.databrickscfg``,
    ``DATABRICKS_CONFIG_PROFILE``), so production callers use
    :func:`build_gateway_client` directly. This wrapper is retained for
    callers (and tests) that select a profile positionally: when
    ``profile`` is set it writes it to ``DATABRICKS_CONFIG_PROFILE``
    before delegating, so the SDK reads that profile's host and
    credentials from ~/.databrickscfg.
    """
    if profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    return build_gateway_client(**kwargs)
