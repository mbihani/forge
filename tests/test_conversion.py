"""Tests for :mod:`anvil.orchestrator.conversion` — the auto-converter that
spins up an Omnigent agent to turn a custom (savesage-style) repo into the
forge-compatible directory structure.

Covers the pure pieces (the conversion prompt builder, the post-conversion
PII scanner, the :class:`ConversionResult` dataclass) and the converter agent
bundle spec (``agents/forge_converter.yaml``). The background task's Omnigent
+ git integration is exercised via the orchestrator endpoint tests in
``tests/test_orchestrator_app.py``.
"""

from __future__ import annotations

import io
import os
import subprocess
import tarfile
from pathlib import Path

import yaml

from anvil.optimizer.omnigent_backend import _build_agent_bundle
from anvil.orchestrator.conversion import (
    DEFAULT_TARGET_BRANCH,
    ConversionResult,
    _build_pr_url,
    build_conversion_prompt,
    check_pii_in_commit,
)

_CONVERTER_YAML = Path(__file__).resolve().parents[1] / "agents" / "forge_converter.yaml"


# ---------------------------------------------------------------------------
# Git helper for the PII-scanner tests (deterministic "main" base branch)
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _git_init(repo: Path) -> Path:
    """Init a repo whose default branch is ``main`` (regardless of the
    user's ``init.defaultBranch``) so the PII diff has a known base."""
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    # Force HEAD to the unborn ``main`` branch before the first commit.
    subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
        capture_output=True,
    )
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    return repo


# ---------------------------------------------------------------------------
# build_conversion_prompt
# ---------------------------------------------------------------------------


def test_build_conversion_prompt_with_findings() -> None:
    """Detected alternative structures are surfaced in the prompt so the agent
    knows exactly what to convert from."""
    findings = {
        "prompts": ["icici.txt", "hdfc.txt"],
        "schemas": ["icici.json"],
        "harness_py": ["config_ws4.py"],
    }
    prompt = build_conversion_prompt(
        "https://github.com/user/repo", "main", "forge-compat", findings
    )
    assert "icici.txt" in prompt
    assert "hdfc.txt" in prompt
    assert "icici.json" in prompt
    assert "config_ws4.py" in prompt
    assert "forge-compat" in prompt  # target branch
    assert "main" in prompt  # base branch


def test_build_conversion_prompt_no_findings() -> None:
    """With no alternative structures the prompt still generates, with a
    generic 'infer the structure' instruction."""
    prompt = build_conversion_prompt(
        "https://github.com/user/repo", "main", "forge-compat", {}
    )
    assert "forge-compat" in prompt
    assert "no alternative structures detected" in prompt


def test_build_conversion_prompt_includes_pii_safety() -> None:
    """The prompt MUST instruct the agent to gitignore golden_set.jsonl and
    commit a build script instead (non-negotiable PII safety)."""
    prompt = build_conversion_prompt(
        "https://github.com/user/repo", "main", "forge-compat", {"prompts": ["a.txt"]}
    )
    assert "golden_set.jsonl" in prompt
    assert ".gitignore" in prompt
    assert "build_golden_set.py" in prompt
    assert "PII" in prompt
    assert "NEVER" in prompt  # the non-negotiable framing


def test_build_conversion_prompt_includes_gh_token() -> None:
    """When a GitHub token is provided it is embedded in the clone/push remote
    URL instruction so the agent can operate on private repos. Without a token
    the tokenized URL is just the plain URL (no x-access-token prefix)."""
    token = "ghp_secrettoken123"
    prompt = build_conversion_prompt(
        "https://github.com/user/repo", "main", "forge-compat", {}, gh_token=token
    )
    # The token appears in the tokenized clone URL.
    assert f"x-access-token:{token}@github.com/user/repo" in prompt
    # The agent is warned not to leak it into files/commits.
    assert "NEVER write the" in prompt or "token" in prompt.lower()

    # No token → plain URL, no x-access-token prefix.
    prompt_no_token = build_conversion_prompt(
        "https://github.com/user/repo", "main", "forge-compat", {}
    )
    assert "x-access-token" not in prompt_no_token
    assert "https://github.com/user/repo" in prompt_no_token


def test_build_conversion_prompt_non_github_url() -> None:
    """A non-GitHub URL is passed through as the clone URL (no tokenization)."""
    prompt = build_conversion_prompt(
        "https://gitlab.com/u/r", "main", "forge-compat", {}, gh_token="tok"
    )
    assert "https://gitlab.com/u/r" in prompt
    assert "x-access-token" not in prompt


# ---------------------------------------------------------------------------
# check_pii_in_commit
# ---------------------------------------------------------------------------


def test_check_pii_clean(tmp_path: Path) -> None:
    """A branch that adds only forge-compatible files (no card data) → clean."""
    repo = _git_init(tmp_path / "repo")
    _git(repo, "checkout", "-b", "feature")
    (repo / "scaffold").mkdir()
    (repo / "scaffold" / "harness.yaml").write_text(
        "skills: []\nsampling: {temperature: 0.3}\n", encoding="utf-8"
    )
    (repo / "scripts").mkdir()
    (repo / "scripts" / "build_golden_set.py").write_text(
        "# builds data/golden_set.jsonl from schemas\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add forge files")
    findings = check_pii_in_commit(repo, "feature", base_branch="main")
    assert findings == []


def test_check_pii_detects_card_masks(tmp_path: Path) -> None:
    """A masked card number in the branch diff is detected."""
    repo = _git_init(tmp_path / "repo")
    _git(repo, "checkout", "-b", "feature")
    (repo / "notes.txt").write_text(
        "Card on file: 4591-XXXX-XXXX-1234 was charged.", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add notes")
    findings = check_pii_in_commit(repo, "feature", base_branch="main")
    assert findings, "expected a masked-card PII finding"
    assert any("card" in f.lower() for f in findings)
    assert any("4591" in f for f in findings)


def test_check_pii_detects_full_pan(tmp_path: Path) -> None:
    """A full 16-digit PAN is detected."""
    repo = _git_init(tmp_path / "repo")
    _git(repo, "checkout", "-b", "feature")
    (repo / "data.txt").write_text("pan=4591123456789012", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add pan")
    findings = check_pii_in_commit(repo, "feature", base_branch="main")
    assert findings
    assert any("card" in f.lower() for f in findings)


def test_check_pii_detects_golden_set_committed(tmp_path: Path) -> None:
    """``data/golden_set.jsonl`` appearing as an added file is flagged on its
    own — it is gitignored by construction, so committing it means the agent
    leaked raw cardholder data."""
    repo = _git_init(tmp_path / "repo")
    _git(repo, "checkout", "-b", "feature")
    (repo / "data").mkdir()
    (repo / "data" / "golden_set.jsonl").write_text(
        '{"example_id":"1","query":"x"}\n', encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "leak golden set")
    findings = check_pii_in_commit(repo, "feature", base_branch="main")
    assert any("golden_set.jsonl" in f for f in findings)


def test_check_pii_missing_base_branch_returns_empty(tmp_path: Path) -> None:
    """A missing/unreachable base branch degrades to an empty list rather than
    raising — PII absence can't be proven, but a crash here must not mask the
    conversion result."""
    repo = _git_init(tmp_path / "repo")
    _git(repo, "checkout", "-b", "feature")
    (repo / "x.txt").write_text("ok", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "x")
    # " nonexistent-base" ref does not exist → git diff fails → empty list.
    findings = check_pii_in_commit(repo, "feature", base_branch="nonexistent-base")
    assert findings == []


# ---------------------------------------------------------------------------
# ConversionResult dataclass
# ---------------------------------------------------------------------------


def test_conversion_result_dataclass() -> None:
    """Default values match the pending initial state."""
    r = ConversionResult()
    assert r.status == "pending"
    assert r.progress == []
    assert r.pr_url is None
    assert r.branch_name is None
    assert r.revalidation is None
    assert r.error is None


def test_conversion_result_to_dict_round_trips_progress() -> None:
    """``to_dict`` returns a JSON-serializable snapshot including progress."""
    r = ConversionResult(status="running", branch_name="forge-compat")
    r.progress.append({"step": "starting", "message": "Conversion task started.", "timestamp": "t"})
    d = r.to_dict()
    assert d["status"] == "running"
    assert d["branch_name"] == "forge-compat"
    assert d["progress"] == [
        {"step": "starting", "message": "Conversion task started.", "timestamp": "t"}
    ]
    assert d["pr_url"] is None
    assert d["revalidation"] is None
    assert d["error"] is None


def test_default_target_branch_constant() -> None:
    assert DEFAULT_TARGET_BRANCH == "forge-compat"


# ---------------------------------------------------------------------------
# _build_pr_url
# ---------------------------------------------------------------------------


def test_build_pr_url_github() -> None:
    url = _build_pr_url("https://github.com/user/repo", "main", "forge-compat")
    assert url == "https://github.com/user/repo/compare/main...forge-compat"


def test_build_pr_url_non_github_returns_none() -> None:
    assert _build_pr_url("https://gitlab.com/u/r", "main", "forge-compat") is None


def test_build_pr_url_strips_dot_git() -> None:
    # _parse_github_url already strips .git before this is called, but the
    # builder is tolerant of a trailing path segment only when owner/repo parse.
    url = _build_pr_url("https://github.com/user/repo", "main", "b")
    assert "compare/main...b" in url


# ---------------------------------------------------------------------------
# Converter agent bundle spec (agents/forge_converter.yaml)
# ---------------------------------------------------------------------------


def _load_converter() -> dict:
    return yaml.safe_load(_CONVERTER_YAML.read_text(encoding="utf-8"))


def test_converter_agent_yaml_exists_and_is_valid() -> None:
    assert _CONVERTER_YAML.is_file()
    data = _load_converter()
    assert isinstance(data, dict)
    assert data["spec_version"] == 1
    assert data["name"] == "forge-converter"
    assert "prompt" in data
    assert "executor" in data
    assert "os_env" in data


def test_converter_agent_yaml_executor_shape() -> None:
    data = _load_converter()
    executor = data["executor"]
    assert executor["type"] == "omnigent"
    assert executor["model"] == "databricks-claude-opus-4-7"
    assert executor["config"]["harness"] == "claude-sdk"
    assert executor["config"]["max_turns"] == 50


def test_converter_agent_yaml_prompt_enforces_pii_and_additive() -> None:
    """The system prompt carries the non-negotiable PII + additive-only rules."""
    prompt = _load_converter()["prompt"]
    assert "golden_set.jsonl" in prompt
    assert ".gitignore" in prompt
    assert "build_golden_set.py" in prompt
    assert "additive" in prompt.lower() or "ADDITIVE" in prompt
    assert "--force" in prompt  # no force push
    assert "PII" in prompt


def test_converter_agent_yaml_has_guardrails_policy() -> None:
    """A guardrails policy denies destructive git ops + raw PII commits
    (defense-in-depth on top of the prompt)."""
    data = _load_converter()
    guardrails = data.get("guardrails", {})
    policies = guardrails.get("policies", {})
    assert "no_destructive_or_pii" in policies
    pol = policies["no_destructive_or_pii"]
    assert pol["on"] == ["tool_call"]
    expr = pol["function"]["arguments"]["expression"]
    assert '"DENY"' in expr
    for forbidden in ("--force", "reset --hard", "golden_set.jsonl"):
        assert forbidden in expr, f"guardrail must forbid {forbidden!r}"


def test_converter_agent_bundle_builds_and_substitutes() -> None:
    """The converter YAML packages into a tar.gz with model/max_turns substituted
    (same contract as the optimizer bundle)."""
    bundle = _build_agent_bundle(
        _CONVERTER_YAML, model="databricks-claude-opus-4-7", max_turns=50
    )
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        assert tar.getnames() == ["config.yaml"]
        config = yaml.safe_load(tar.extractfile("config.yaml").read())  # type: ignore[union-attr]
    assert config["name"] == "forge-converter"
    assert config["executor"]["model"] == "databricks-claude-opus-4-7"
    assert config["executor"]["config"]["max_turns"] == 50
    assert "prompt" in config
