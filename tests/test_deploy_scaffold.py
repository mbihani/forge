from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
DEPLOY = ROOT / "deploy"
CONFIGURE_RE = re.compile(r"CONFIGURE\(([a-z][a-z0-9_]*)\)")

EXPECTED_SLUGS = {
    "ai_gateway_url",
    "domain_config",
    "eval_engine",
    "git_remote_url",
    "git_token",
    "mlflow_experiment",
    "omnigent_auth_token",
    "omnigent_server_url",
    "optimizer_model",
}


def _load_yaml(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_yaml_files_are_valid_and_describe_the_serverless_app() -> None:
    agent = _load_yaml(ROOT / "agents" / "forge_optimizer.yaml")
    bundle = _load_yaml(DEPLOY / "databricks.yml")
    app = _load_yaml(DEPLOY / "app.yaml")

    assert agent["executor"]["model"] == "databricks-claude-opus-4-7"
    assert agent["executor"]["extra"]["max_turns"] == 70
    assert set(agent) & {"mcp_servers", "tools"} == set()
    assert app["compute"] == "serverless"
    assert app["health_check"]["path"] == "/health"

    resource = bundle["resources"]["apps"]["forge_orchestrator"]
    assert resource["config"]["command"][:2] == [
        "uvicorn",
        "anvil.orchestrator.app:app",
    ]
    assert set(bundle["variables"]) == EXPECTED_SLUGS


def test_manifesto_configure_catalog_matches_deployment_markers() -> None:
    source_slugs = set(CONFIGURE_RE.findall((DEPLOY / "databricks.yml").read_text()))
    manifesto = (DEPLOY / "MANIFESTO.md").read_text(encoding="utf-8")
    manifesto_slugs = set(CONFIGURE_RE.findall(manifesto))

    assert source_slugs == EXPECTED_SLUGS
    assert manifesto_slugs == source_slugs
    assert manifesto.count("## Required") == 1
    assert manifesto.count("## Customize") == 1
    assert manifesto.count("## Optional") == 1


def test_branding_defaults_support_light_and_dark_themes() -> None:
    css = (DEPLOY / "branding.css").read_text(encoding="utf-8")
    variables = set(re.findall(r"--([a-z0-9-]+):", css))

    assert ":root" in css
    assert "prefers-color-scheme: dark" in css
    assert {"brand-color-primary", "brand-color-background", "brand-color-text"} <= variables

    spec = importlib.util.spec_from_file_location("forge_branding", DEPLOY / "branding.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.APP_NAME == "Forge Orchestrator"
    assert module.TAGLINE
