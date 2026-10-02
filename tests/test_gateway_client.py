"""Tests for the Databricks AI Gateway client — the sole LLM route.

Covers the acceptance contract:

* ``GatewayClient`` constructs a fresh ``openai.OpenAI`` per call with a
  per-request refreshed SP token.
* ``_get_fresh_sp_token`` resolves from ``DATABRICKS_TOKEN`` or the SDK.
* ``_gateway_base_url`` builds the unified gateway URL.
* ``build_databricks_client`` stays as a backward-compatible wrapper that
  delegates to ``build_gateway_client``.
* The judge path in ``evaluate_branch`` builds its client via
  ``build_gateway_client``.
* The runtime agent builds its client via ``build_gateway_client`` when
  none is injected.

No real LLM, Databricks, or gateway calls are made — ``openai.OpenAI``
and the Databricks SDK are stubbed.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 1. GatewayClient: per-call fresh OpenAI client + per-request token refresh
# ---------------------------------------------------------------------------


def _fake_openai_factory(constructed: list, response: object) -> object:
    """Return a callable that mimics ``openai.OpenAI`` and records calls."""

    def _ctor(**kwargs: object) -> object:
        constructed.append(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_k: response))
        )

    return _ctor


def test_gateway_client_creates_fresh_openai_per_call_with_fresh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each ``chat.completions.create`` builds a new ``OpenAI`` with a
    token fetched on that call — SP tokens expire, so we never reuse a
    stale token or client."""
    from anvil.runtime.client import build_gateway_client

    constructed: list[dict] = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    monkeypatch.setattr("openai.OpenAI", _fake_openai_factory(constructed, response))

    token_calls: list[str] = []

    def token_fn() -> str:
        token_calls.append(f"tok-{len(token_calls)}")
        return token_calls[-1]

    client = build_gateway_client(
        base_url="https://gw.example/ai-proxy-api/llm/v1",
        token_fn=token_fn,
    )

    r1 = client.chat.completions.create(
        model="databricks-claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
    )
    r2 = client.chat.completions.create(
        model="databricks-claude-sonnet-4-6",
        messages=[{"role": "user", "content": "bye"}],
    )

    # Both calls return the delegated response.
    assert r1 is response
    assert r2 is response
    # A fresh OpenAI client is constructed per call (not cached).
    assert len(constructed) == 2
    # The token is refreshed on every call — never reused.
    assert len(token_calls) == 2
    # Each constructed client got the token from its own refresh.
    assert constructed[0]["api_key"] == "tok-0"
    assert constructed[1]["api_key"] == "tok-1"
    # The unified gateway base URL is forwarded to every client.
    assert constructed[0]["base_url"] == "https://gw.example/ai-proxy-api/llm/v1"
    assert constructed[1]["base_url"] == "https://gw.example/ai-proxy-api/llm/v1"


def test_gateway_client_forwards_model_and_kwargs_to_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``model`` + extra kwargs flow through to the underlying call."""
    from anvil.runtime.client import build_gateway_client

    captured: dict = {}
    response = SimpleNamespace()

    def _ctor(**kwargs: object) -> object:
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **k: (captured.update(k), response)[1])
            )
        )

    monkeypatch.setattr("openai.OpenAI", _ctor)

    client = build_gateway_client(
        base_url="https://gw.example/v1",
        token_fn=lambda: "t",
    )
    out = client.chat.completions.create(
        model="databricks-claude-sonnet-4-6",
        messages=[{"role": "user", "content": "q"}],
        max_tokens=400,
        temperature=0,
    )
    assert out is response
    assert captured["model"] == "databricks-claude-sonnet-4-6"
    assert captured["messages"] == [{"role": "user", "content": "q"}]
    assert captured["max_tokens"] == 400
    assert captured["temperature"] == 0


# ---------------------------------------------------------------------------
# 2. _get_fresh_sp_token
# ---------------------------------------------------------------------------


def test_token_prefers_databricks_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A static ``DATABRICKS_TOKEN`` short-circuits the SDK path."""
    from anvil.runtime.client import _get_fresh_sp_token

    monkeypatch.setenv("DATABRICKS_TOKEN", "static-pat")

    # If the SDK were touched, this would blow up.
    import databricks.sdk as dbsdk

    def _boom(*a, **k):
        raise AssertionError("SDK must not be used when DATABRICKS_TOKEN is set")

    monkeypatch.setattr(dbsdk, "WorkspaceClient", _boom)

    assert _get_fresh_sp_token() == "static-pat"


def test_token_uses_sdk_config_token_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env token, the SDK config's static token is used."""
    from anvil.runtime.client import _get_fresh_sp_token

    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    fake_ws = SimpleNamespace(
        config=SimpleNamespace(
            token="cfg-token",
            authenticate=lambda: {"Authorization": "Bearer should-not-be-used"},
        )
    )
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: fake_ws)

    assert _get_fresh_sp_token() == "cfg-token"


def test_token_uses_sdk_authenticate_when_config_token_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no static token anywhere, the SDK mints a fresh bearer.

    ``config.authenticate()`` returns a ``Dict[str, str]``; the token is
    extracted from the ``Authorization: Bearer <token>`` header.
    """
    from anvil.runtime.client import _get_fresh_sp_token

    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    fake_ws = SimpleNamespace(
        config=SimpleNamespace(
            token="", authenticate=lambda: {"Authorization": "Bearer fresh-bearer"}
        )
    )
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: fake_ws)

    assert _get_fresh_sp_token() == "fresh-bearer"


def test_token_raises_when_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    from anvil.runtime.client import _get_fresh_sp_token

    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    fake_ws = SimpleNamespace(
        config=SimpleNamespace(token="", authenticate=lambda: {"Authorization": ""})
    )
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: fake_ws)

    with pytest.raises(RuntimeError, match="Could not resolve a Databricks token"):
        _get_fresh_sp_token()


# ---------------------------------------------------------------------------
# 3. _gateway_base_url
# ---------------------------------------------------------------------------


def test_gateway_base_url_built_from_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from anvil.runtime.client import _gateway_base_url

    monkeypatch.setenv("DATABRICKS_HOST", "https://foo.cloud.databricks.com")
    monkeypatch.delenv("ANVIL_GATEWAY_BASE_URL", raising=False)
    assert _gateway_base_url() == "https://foo.cloud.databricks.com/serving-endpoints"


def test_gateway_base_url_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    from anvil.runtime.client import _gateway_base_url

    monkeypatch.setenv("DATABRICKS_HOST", "https://foo.cloud.databricks.com/")
    monkeypatch.delenv("ANVIL_GATEWAY_BASE_URL", raising=False)
    assert _gateway_base_url() == "https://foo.cloud.databricks.com/serving-endpoints"


def test_gateway_base_url_falls_back_to_sdk_config_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``DATABRICKS_HOST`` is unset (profile-only config), the host is
    resolved from the SDK config (which reads ~/.databrickscfg profiles and
    DATABRICKS_CONFIG_PROFILE)."""
    from anvil.runtime.client import _gateway_base_url

    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("ANVIL_GATEWAY_BASE_URL", raising=False)

    fake_ws = SimpleNamespace(config=SimpleNamespace(host="https://profile.cloud.databricks.com"))
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: fake_ws)

    assert _gateway_base_url() == "https://profile.cloud.databricks.com/serving-endpoints"


def test_gateway_base_url_raises_when_host_unresolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no host can be resolved from env or SDK config, fail loudly
    rather than producing an empty-host gateway URL."""
    from anvil.runtime.client import _gateway_base_url

    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("ANVIL_GATEWAY_BASE_URL", raising=False)

    fake_ws = SimpleNamespace(config=SimpleNamespace(host=None))
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda *a, **k: fake_ws)

    with pytest.raises(RuntimeError, match="Could not resolve Databricks host"):
        _gateway_base_url()


def test_gateway_base_url_env_override_used_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ANVIL_GATEWAY_BASE_URL`` (when set, non-empty) is used verbatim —
    only a trailing slash is stripped, no path is appended, and the
    workspace host is not consulted."""
    from anvil.runtime.client import _gateway_base_url

    monkeypatch.setenv("DATABRICKS_HOST", "https://should-not-be-used.cloud.databricks.com")
    monkeypatch.setenv("ANVIL_GATEWAY_BASE_URL", "https://gw-proxy.example/ai-proxy-api/llm/v1")

    assert _gateway_base_url() == "https://gw-proxy.example/ai-proxy-api/llm/v1"


def test_gateway_base_url_env_override_strips_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trailing slash on ``ANVIL_GATEWAY_BASE_URL`` is stripped; the rest
    of the value is used verbatim."""
    from anvil.runtime.client import _gateway_base_url

    monkeypatch.setenv("DATABRICKS_HOST", "https://should-not-be-used.cloud.databricks.com")
    monkeypatch.setenv("ANVIL_GATEWAY_BASE_URL", "https://gw-proxy.example/proxy/")

    assert _gateway_base_url() == "https://gw-proxy.example/proxy"


def test_gateway_base_url_env_override_empty_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``ANVIL_GATEWAY_BASE_URL`` is treated as unset, so the
    default ``/serving-endpoints`` route is built from the resolved host."""
    from anvil.runtime.client import _gateway_base_url

    monkeypatch.setenv("DATABRICKS_HOST", "https://foo.cloud.databricks.com")
    monkeypatch.setenv("ANVIL_GATEWAY_BASE_URL", "")

    assert _gateway_base_url() == "https://foo.cloud.databricks.com/serving-endpoints"


def test_gateway_base_url_env_override_beats_sdk_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The env override wins even when the host would otherwise come from
    the SDK config (``DATABRICKS_HOST`` unset). The SDK is not consulted
    at all when the override is set."""
    from anvil.runtime.client import _gateway_base_url

    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    # If the SDK were consulted, this would blow up — proving it isn't.
    import databricks.sdk as dbsdk

    def _boom(*a, **k):
        raise AssertionError("SDK must not be used when ANVIL_GATEWAY_BASE_URL is set")

    monkeypatch.setattr(dbsdk, "WorkspaceClient", _boom)
    monkeypatch.setenv("ANVIL_GATEWAY_BASE_URL", "https://gw-proxy.example/proxy")

    assert _gateway_base_url() == "https://gw-proxy.example/proxy"


# ---------------------------------------------------------------------------
# 4. build_gateway_client defaults + backward-compat build_databricks_client
# ---------------------------------------------------------------------------


def test_build_gateway_client_defaults_use_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without overrides, base_url + token_fn come from the environment."""
    from anvil.runtime.client import build_gateway_client

    monkeypatch.setenv("DATABRICKS_HOST", "https://bar.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "env-token")
    monkeypatch.delenv("ANVIL_GATEWAY_BASE_URL", raising=False)

    client = build_gateway_client()
    assert client._base_url == "https://bar.cloud.databricks.com/serving-endpoints"
    # token_fn defaults to _get_fresh_sp_token, which reads the env token.
    assert client._get_token() == "env-token"


def test_build_gateway_client_explicit_base_url_beats_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``base_url=`` argument to ``build_gateway_client`` takes
    precedence over the ``ANVIL_GATEWAY_BASE_URL`` env var, which in turn
    beats the default ``/serving-endpoints`` route."""
    from anvil.runtime.client import build_gateway_client

    monkeypatch.setenv("DATABRICKS_HOST", "https://foo.cloud.databricks.com")
    monkeypatch.setenv("ANVIL_GATEWAY_BASE_URL", "https://env.example/proxy")
    monkeypatch.setenv("DATABRICKS_TOKEN", "env-token")

    client = build_gateway_client(base_url="https://explicit.example/v1")
    assert client._base_url == "https://explicit.example/v1"


def test_build_gateway_client_env_base_url_beats_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no explicit ``base_url=`` argument, ``ANVIL_GATEWAY_BASE_URL``
    wins over the default ``/serving-endpoints`` route built from the host."""
    from anvil.runtime.client import build_gateway_client

    monkeypatch.setenv("DATABRICKS_HOST", "https://foo.cloud.databricks.com")
    monkeypatch.setenv("ANVIL_GATEWAY_BASE_URL", "https://env.example/proxy")
    monkeypatch.setenv("DATABRICKS_TOKEN", "env-token")

    client = build_gateway_client()
    assert client._base_url == "https://env.example/proxy"


def test_build_databricks_client_delegates_to_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_databricks_client`` is a backward-compat wrapper returning a
    ``GatewayClient``. The ``profile`` kwarg is honored — it sets
    ``DATABRICKS_CONFIG_PROFILE`` so the SDK reads the profile."""
    from anvil.runtime.client import GatewayClient, build_databricks_client, build_gateway_client

    monkeypatch.setenv("DATABRICKS_HOST", "https://test.cloud.databricks.com")

    assert isinstance(build_databricks_client(), GatewayClient)
    # The legacy ``profile`` kwarg is accepted and sets DATABRICKS_CONFIG_PROFILE.
    assert isinstance(build_databricks_client(profile="some-profile"), GatewayClient)
    # Extra kwargs (base_url, token_fn) pass through to build_gateway_client.
    delegating = build_databricks_client(base_url="https://custom/v1", token_fn=lambda: "t")
    assert isinstance(delegating, GatewayClient)
    assert delegating._base_url == "https://custom/v1"
    # The wrapper and the direct factory produce the same type.
    assert type(build_databricks_client()) is type(build_gateway_client())


def test_build_databricks_client_sets_config_profile_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing ``profile=`` sets ``DATABRICKS_CONFIG_PROFILE`` so the SDK
    reads the profile's host + credentials from ~/.databrickscfg. This
    preserves the behavioral contract of the legacy call — the profile
    must actually be used, not silently ignored."""
    from anvil.runtime.client import build_databricks_client

    monkeypatch.setenv("DATABRICKS_HOST", "https://test.cloud.databricks.com")
    # Record a known starting value so monkeypatch restores it on teardown.
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "__pre_test__")

    build_databricks_client(profile="my-profile")
    assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "my-profile"


# ---------------------------------------------------------------------------
# 5. Judge path routes through build_gateway_client (runner wiring)
# ---------------------------------------------------------------------------


def _judge_wiring_config():
    from anvil.runtime.models import (
        EvalConfig,
        EvalModeConfig,
        ExperimentsConfig,
        HarnessConfig,
        ScorerConfig,
    )

    return HarnessConfig(
        mode="code",
        agent_module="anvil.agents.baseline",
        runtime_endpoint="rt",
        optimizer_endpoint="op",
        judge_endpoint="j",
        experiments=ExperimentsConfig(runtime="r", eval="e", optimizer="o"),
        eval=EvalConfig(
            default_mode="quick",
            scorers=[ScorerConfig(name="correctness", type="llm", weight=1.0)],
            modes={"quick": EvalModeConfig(rows=1, buckets={"direct": 1})},
        ),
    )


def test_judge_client_built_via_build_gateway_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no ``judge_client`` is passed, ``evaluate_branch`` builds it via
    ``build_gateway_client`` and hands it to ``build_scorers``."""
    from anvil.eval import runner
    from anvil.runtime.client import GatewayClient

    config = _judge_wiring_config()
    monkeypatch.setattr(runner, "load_harness", lambda *a, **kw: SimpleNamespace(config=config))
    monkeypatch.setattr(runner, "load_golden_set", lambda _p: [])
    monkeypatch.setattr(runner, "select_subset", lambda *a, **k: [])
    monkeypatch.setattr(runner, "make_kb_executor", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runner, "setup_eval_tracing", lambda **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(runner.mlflow, "get_experiment_by_name", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_load_memory_system", lambda *a, **kw: SimpleNamespace())

    sentinel_gateway = GatewayClient(base_url="https://gw/v1", token_fn=lambda: "t")
    monkeypatch.setattr(runner, "build_gateway_client", lambda **kw: sentinel_gateway)

    captured: dict = {}
    monkeypatch.setattr(
        runner,
        "build_scorers",
        lambda **kw: captured.update(judge_client=kw["judge_client"]) or [],
    )
    monkeypatch.setattr(
        runner.mlflow.genai,
        "evaluate",
        lambda **kw: SimpleNamespace(result_df=pd.DataFrame(), metrics={}, run_id="r"),
    )

    # Pass a truthy runtime_client so the runtime factory call is skipped;
    # pass judge_client=None so the judge factory (build_gateway_client) runs.
    runner.evaluate_branch(
        scaffold_root=tmp_path / "scaffold",
        runtime_config_path=tmp_path / "config.yaml",
        runtime_client=SimpleNamespace(),
        judge_client=None,
    )

    assert captured["judge_client"] is sentinel_gateway


# ---------------------------------------------------------------------------
# 6. Runtime agent builds its client via build_gateway_client
# ---------------------------------------------------------------------------


def test_runtime_agent_uses_gateway_client_when_none_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anvil.runtime import agent as agent_mod

    monkeypatch.setattr(agent_mod, "load_harness", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(
        agent_mod, "default_runtime_config_path", lambda root: tmp_path / "config.yaml"
    )

    sentinel = SimpleNamespace(name="gateway-client")
    monkeypatch.setattr(agent_mod, "build_gateway_client", lambda **kw: sentinel)

    agent = agent_mod.AnvilAgent(scaffold_root=tmp_path / "scaffold")
    assert agent._client is sentinel
