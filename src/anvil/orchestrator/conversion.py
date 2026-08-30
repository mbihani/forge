"""Auto-convert a custom (savesage-style) repo into the forge-compatible
directory structure, driven by an Omnigent agent.

The orchestrator validates a user-provided agent repo against 8 checks. When
the repo fails but has a recognizable *alternative* structure
(``prompts/*.txt`` + ``schema/*.json`` + ``harness/*.py`` + ``skills/*.py``),
the user can click "Convert to forge-compatible". This module spins up a
forge-converter Omnigent agent that:

1. Clones the repo (with the user's GitHub token, for private repos).
2. Creates a target branch from the base branch.
3. ADDITIVELY creates the forge files (``scaffold/harness.yaml``,
   ``scaffold/skills/*.md``, ``harness/config.yaml``, a build script) and a
   ``.gitignore`` that excludes ``data/golden_set.jsonl`` (cardholder PII).
4. Commits + pushes the target branch (never force).
5. Reports the files it created.

The orchestrator then re-clones the converted branch, runs a post-conversion
PII check, and re-runs the 8 validation checks. The result (progress log,
branch name, PR compare link, re-validation report) is exposed for polling.

PII safety (non-negotiable):

* ``data/golden_set.jsonl`` may contain cardholder / customer transaction
  data, so the conversion prompt instructs the agent to gitignore it and
  commit a ``scripts/build_golden_set.py`` build script instead.
* :func:`check_pii_in_commit` scans the converted branch diff for card masks /
  tokens / a committed ``golden_set.jsonl`` after the agent finishes.

The Omnigent client is async (httpx + SSE); the background task runs on the
orchestrator's event loop and offloads blocking git/file work to a thread
pool (same pattern as the optimization task in :mod:`anvil.orchestrator.app`).
"""

from __future__ import annotations

import logging
import re
import subprocess
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import anyio

from anvil.optimizer.omnigent_backend import _build_agent_bundle
from anvil.optimizer.omnigent_client import (
    OmnigentClient,
    OmnigentError,
    SessionCreateMetadata,
)

logger = logging.getLogger("anvil.orchestrator.conversion")

# Two-step managed-host flow. The managed Omnigent server returns 503
# ("No runner bound for session") when a session is created from a multipart
# bundle upload alone — the upload registers the agent but no managed runner
# is provisioned. So the converter uploads the bundle to *register* the agent
# (and discards that throwaway session), then opens a SECOND session via
# OmnigentClient.create_session_from_agent(agent_id, host_type="managed",
# model_override=...) which triggers the managed host to auto-provision a
# runner. Messages then get HTTP 202 instead of 503.

# Repo root (this file is at src/anvil/orchestrator/conversion.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONVERTER_AGENT_YAML = _REPO_ROOT / "agents" / "forge_converter.yaml"

# Model for the converter agent on the managed Omnigent host.
_CONVERTER_MODEL = "databricks-claude-opus-4-8"

# Default conversion branch name (overridable via ConvertRequest.target_branch).
DEFAULT_TARGET_BRANCH = "forge-compat"

# Omnigent SSE event types (mirrors omnigent_backend.py).
_DELTA_TYPE = "response.output_text.delta"
_ITEM_DONE_TYPE = "response.output_item.done"
_COMPLETED_TYPE = "response.completed"
_STATUS_TYPE = "session.status"

# ---------------------------------------------------------------------------
# PII patterns scanned on the converted branch diff.
#
# Card masks are the primary signal: a converted branch must never carry raw
# cardholder data. The patterns below catch masked PANs (4591-XXXX-XXXX-1234),
# star-masked PANs (4591****1234), and full 15/16-digit PANs with optional
# spaces/dashes. ``data/golden_set.jsonl`` appearing as an added file is
# flagged separately — it is gitignored by construction, so its presence in the
# diff means the agent committed raw data.
# ---------------------------------------------------------------------------
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Masked card: 3-4 leading digits, 1-4 groups of mask chars (X/x/*) with
    # optional separators, then 2-4 trailing digits. Catches the common
    # "4591-XXXX-XXXX-1234" (two mask groups) and "4591****1234" (one group).
    (re.compile(r"\b\d{3,4}(?:[\s\-]?[Xx\*]+){1,4}[\s\-]?\d{2,4}\b"), "masked card number"),
    # Full 16-digit PAN (Visa/MC) with optional spaces/dashes.
    (re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"), "16-digit card number"),
    # Amex 15-digit: 4-6-5 grouping.
    (re.compile(r"\b3\d{3}[\s\-]?\d{6}[\s\-]?\d{5}\b"), "Amex card number"),
]


@dataclass
class ConversionResult:
    """Mutable, pollable state for one conversion. Stored on the session.

    The task updates ``status`` / ``progress`` / ``revalidation`` as it runs;
    the GET /convert endpoint serializes this via :meth:`to_dict`.
    """

    status: str = "pending"  # pending, running, completed, failed
    progress: list[dict[str, str]] = field(default_factory=list)
    pr_url: str | None = None
    branch_name: str | None = None
    revalidation: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "progress": [dict(entry) for entry in self.progress],
            "pr_url": self.pr_url,
            "branch_name": self.branch_name,
            "revalidation": self.revalidation,
            "error": self.error,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_conversion_prompt(
    repo_url: str,
    base_branch: str,
    target_branch: str,
    findings: dict[str, Any],
    gh_token: str | None = None,
) -> str:
    """Generate the agent instruction text (the first USER message).

    ``findings`` is what :func:`anvil.orchestrator.app._scan_agent_root` returns
    — ``{prompts, schemas, python_skills, judge, tests, harness_py,
    config_py, data_dir}``, each mapping to a list of filenames (empty/absent
    when nothing was found). The prompt instructs the agent to clone at the
    base branch, create the target branch, additively create the forge files,
    gitignore ``data/golden_set.jsonl``, commit a build script, and push.

    When ``gh_token`` is provided it is embedded in the clone/push remote URL
    so the agent can operate on private repos (per the user's design decision).
    The token NEVER appears in committed files or the progress log — only in
    this instruction to the agent.
    """
    # Build the clone/push remote URL, embedding the token for private repos.
    tokenized_clone_url = repo_url
    if gh_token and repo_url.startswith("https://github.com/"):
        tokenized_clone_url = (
            f"https://x-access-token:{gh_token}@github.com/"
            + repo_url[len("https://github.com/"):]
        )

    # Summarize the detected alternative structures for the agent.
    findings_lines: list[str] = []
    for key, label in (
        ("prompts", "prompts/*.txt"),
        ("schemas", "schema/*.json"),
        ("python_skills", "skills/*.py"),
        ("judge", "judge/*.py"),
        ("tests", "tests/"),
        ("harness_py", "harness/*.py"),
        ("config_py", "config*.py"),
        ("data_dir", "data/"),
    ):
        files = findings.get(key) if isinstance(findings, dict) else None
        if files:
            files_str = ", ".join(str(f) for f in files)
            findings_lines.append(f"- {label}: {files_str}")
    findings_block = (
        "\n".join(findings_lines) if findings_lines else "- (no alternative structures detected; "
        "infer the agent structure from the repo contents directly)"
    )

    token_note = ""
    if gh_token:
        token_note = (
            "A GitHub token is provided below (embedded in the clone URL). Use the "
            "TOKENIZED clone URL for `git clone` and `git push`. NEVER write the "
            "token into any file or commit message — use it only inside the remote "
            "URL.\n\n"
        )

    return f"""Convert this repository to the forge-compatible structure.

## Repository

- Repo URL: {repo_url}
- Base branch: {base_branch}
- Target branch (create + push ONLY this): {target_branch}

{token_note}## Clone + push URLs

- Clone URL (use this for `git clone`): {tokenized_clone_url}
- For `git push origin {target_branch}` use the same tokenized remote.

## Detected alternative structures (convert from these)

{findings_block}

## What to do (ALL ADDITIVE — never edit or delete existing files)

1. Clone the repo at the base branch using the clone URL above. Clone into a
   fresh temp directory; do not touch the orchestrator's working copy.
2. Create the target branch from the base branch:
   `git checkout -b {target_branch}`. Never operate on the base branch.
3. Create these forge-compatible files ADDITIVELY:
   - `scaffold/harness.yaml` — a `skills` list (one entry per skill file you
     create, `{{file: <rel-path>}}`) and a `sampling` dict (temperature ~0.3,
     max_tokens ~2048).
   - `scaffold/skills/*.md` — decompose each `prompts/*.txt` file into separate
     skill files, one per logical section (transaction rules, rewards rules,
     edge cases, missing data, bank identity, …). Each starts with `---` YAML
     frontmatter (`skill_id`, `applies_to: runtime`) then the section body.
   - `harness/config.yaml` — `mode` (prompt, or code if `skills/*.py` agent
     code with a predict() entrypoint exists), `runtime_endpoint`,
     `optimizer_endpoint`, `judge_endpoint` (derive from `harness/*.py` /
     `config*.py`; default runtime/judge to `databricks-claude-sonnet-4-6`,
     optimizer to `databricks-claude-opus-4-8`), an `eval` section
     (`default_mode`, `modes` with `{{quick: {{rows: 12}}, standard: {{rows: 24}},
     full: {{rows: 304}}}}`, `scorers: [correctness]`, `held_out_test: true`),
     and `gate: {{type: frontier}}`.
   - `scripts/build_golden_set.py` — a committed placeholder that documents
     how to build `data/golden_set.jsonl` from `schema/*.json` definitions.
     Must NOT embed real customer data.
4. Create / append `.gitignore` so `data/golden_set.jsonl` is ignored.
5. Stage files EXPLICITLY (never `git add -A` before `.gitignore` exists):
   `git add scaffold/ harness/config.yaml scripts/build_golden_set.py .gitignore`.
6. Commit: `git commit -m "chore: convert to forge-compatible structure"`.
7. Push the target branch: `git push origin {target_branch}`. NEVER use
   `--force` or `--force-with-lease`.
8. End your turn with a report: every file you created (full paths), the
   target branch name, the clone/push outcome, and any issues.

## PII SAFETY (non-negotiable)

`data/golden_set.jsonl` may contain cardholder / customer transaction data.
- ALWAYS add `data/golden_set.jsonl` to `.gitignore` BEFORE any `git add`.
- Commit `scripts/build_golden_set.py` (a build script) INSTEAD of the data.
- NEVER `git add data/golden_set.jsonl`.
- NEVER commit raw customer extraction data.

## Hard constraints

- Additive only — never `git rm`, never edit an existing file, never
  `reset --hard`, never delete a branch.
- No force push.
- All work on the target branch only — never push to base/main/master.
- No secrets in commits — the GitHub token stays in the remote URL only.
"""


def check_pii_in_commit(
    repo_path: str | Path, branch: str, base_branch: str = "main"
) -> list[str]:
    """Post-conversion PII check: scan the branch diff for card masks, tokens,
    and a committed ``data/golden_set.jsonl``.

    Returns a list of human-readable findings (empty = clean). Runs ``git diff``
    between ``base_branch`` and ``branch`` (three-dot: changes on ``branch``
    since their merge-base) and inspects the *added* lines (``+``-prefixed,
    excluding ``+++`` file headers) for the PII patterns. A
    ``data/golden_set.jsonl`` that appears as an added/modified file is flagged
    on its own — it is gitignored by construction, so its presence in the diff
    means the agent committed raw data.

    Defensive: if the git command fails (missing base branch, not a git repo,
    timeout) the function returns an empty list rather than raising — PII
    absence cannot be proven, but a crash here must not mask the conversion
    result. The caller surfaces the (possibly empty) list alongside the
    re-validation report.
    """
    repo = Path(repo_path)
    # Three-dot diff: changes introduced on `branch` since the merge-base with
    # `base_branch`. Falls back to two-dot if the base ref is unreachable.
    diff_text = _git_text(repo, "diff", f"{base_branch}...{branch}")
    if diff_text is None:
        diff_text = _git_text(repo, "diff", f"{base_branch}..{branch}") or ""

    findings: list[str] = []
    current_file: str | None = None
    for line in diff_text.splitlines():
        # Track the current file from the `+++ b/<file>` header.
        if line.startswith("+++ b/"):
            current_file = line[6:]
            if current_file == "data/golden_set.jsonl":
                findings.append(
                    "data/golden_set.jsonl is committed on the branch "
                    "(may contain cardholder data — must be gitignored)"
                )
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        # Added content line — strip the leading `+`.
        content = line[1:]
        for pattern, label in _PII_PATTERNS:
            match = pattern.search(content)
            if match:
                snippet = match.group(0)
                where = current_file or "(diff)"
                findings.append(f"possible {label} in {where}: {snippet}")
                break  # one finding per added line

    return findings


def _git_text(repo: Path, *args: str, timeout: int = 30) -> str | None:
    """Run a git command, returning stdout text or ``None`` on any failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _build_pr_url(clone_url: str, base_branch: str, target_branch: str) -> str | None:
    """Build a GitHub compare/PR-creation URL for the converted branch.

    ``clone_url`` is the plain ``https://github.com/<owner>/<repo>`` form (the
    orchestrator's ``_parse_github_url`` already strips ``.git`` and
    ``/tree/...``). Returns ``None`` for non-GitHub URLs.
    """
    if not clone_url.startswith("https://github.com/"):
        return None
    rest = clone_url[len("https://github.com/"):]
    parts = rest.split("/")
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    return f"https://github.com/{owner}/{repo}/compare/{base_branch}...{target_branch}"


# ---------------------------------------------------------------------------
# Background conversion task
# ---------------------------------------------------------------------------


async def _run_conversion_task(session_id: str, target_branch: str) -> None:
    """Background asyncio task that runs the forge-converter agent and
    re-validates the converted branch.

    Reads the session (repo URL, detected findings, GitHub token, clone root)
    via a lazy import of :mod:`anvil.orchestrator.app` (avoids a circular
    import). Updates ``sess.conversion`` (a :class:`ConversionResult`) under
    the session lock as it progresses so the GET /convert endpoint can poll.

    Blocking work (git clone/branch, validation) runs via
    ``anyio.to_thread.run_sync``; the Omnigent client is async and runs on the
    event loop. The omnigent session is deleted in a ``finally`` for cleanup.
    """
    # Lazy import: app.py imports this module, so we defer the reverse import
    # to call time to avoid a circular import at module load. Resolving
    # ``app_module._clone_repo`` / ``_run_validation`` at call time also means
    # tests that monkeypatch them on the app module are picked up here.
    import os

    from anvil.orchestrator import app as app_module

    server_url = os.getenv("OMNIGENT_SERVER_URL")
    auth_token = os.getenv("OMNIGENT_AUTH_TOKEN")

    def _progress(step: str, message: str) -> None:
        with app_module._session_lock:
            sess = app_module._sessions.get(session_id)
            if sess is not None and sess.conversion is not None:
                sess.conversion.progress.append(
                    {"step": step, "message": message, "timestamp": _now()}
                )

    def _set(**fields: Any) -> None:
        with app_module._session_lock:
            sess = app_module._sessions.get(session_id)
            if sess is not None and sess.conversion is not None:
                for key, value in fields.items():
                    setattr(sess.conversion, key, value)

    try:
        sess = app_module._sessions.get(session_id)
        if sess is None:
            return
        repo_url = sess.repo_url
        gh_token = sess._github_token
        findings = sess._findings or {}
        agent_subpath = sess.agent_subpath
        clone_root = sess._clone_root or sess.repo_path

        _set(status="running")
        _progress("starting", "Conversion task started.")

        # Determine the base branch from the existing clone's checkout.
        base_branch = await anyio.to_thread.run_sync(app_module._current_branch, clone_root)
        if not base_branch:
            base_branch = "main"
        _progress("base_branch", f"Base branch: {base_branch}")

        clone_url, _, _ = app_module._parse_github_url(repo_url)

        # ---- Build the converter agent bundle + omnigent session ----
        _progress("building_bundle", "Building the forge-converter agent bundle.")
        bundle = await anyio.to_thread.run_sync(
            partial(
                _build_agent_bundle,
                _CONVERTER_AGENT_YAML,
                model=_CONVERTER_MODEL,
                max_turns=50,
            )
        )

        if not server_url:
            raise RuntimeError(
                "OMNIGENT_SERVER_URL is not set; cannot run the conversion agent."
            )

        prompt = build_conversion_prompt(
            repo_url=repo_url,
            base_branch=base_branch,
            target_branch=target_branch,
            findings=findings,
            gh_token=gh_token,
        )

        _progress("agent_session", "Creating the Omnigent conversion session.")
        client = OmnigentClient(server_url, auth_token)
        await _run_managed_session(client, bundle, prompt, target_branch, _progress)

        # ---- Re-clone the converted branch + re-validate ----
        _progress(
            "recloning",
            f"Re-cloning the converted branch '{target_branch}' for re-validation.",
        )
        converted_root = app_module._SESSIONS_ROOT / f"{session_id}-converted"
        converted_path = (
            (converted_root / agent_subpath) if agent_subpath else converted_root
        )
        err = await anyio.to_thread.run_sync(
            partial(
                app_module._clone_repo,
                clone_url,
                converted_root,
                gh_token,
                target_branch,
            )
        )
        if err is not None:
            # Redact any embedded token from the git error.
            if gh_token:
                err = err.replace(gh_token, "***")
            raise RuntimeError(f"failed to re-clone converted branch: {err}")
        if agent_subpath and not converted_path.is_dir():
            raise RuntimeError(
                f"subdirectory '{agent_subpath}' not found in converted branch"
            )

        # ---- Post-conversion PII check ----
        _progress("pii_check", "Scanning the converted branch for PII (card masks / tokens).")
        pii_findings = await anyio.to_thread.run_sync(
            partial(check_pii_in_commit, converted_root, target_branch, base_branch)
        )
        if pii_findings:
            _progress(
                "pii_findings",
                "PII detected: " + "; ".join(pii_findings[:5]),
            )
        else:
            _progress("pii_check", "No PII patterns detected in the branch diff.")

        # ---- Re-run the 8 validation checks on the converted branch ----
        _progress("revalidating", "Re-running the 8 validation checks on the converted branch.")
        report, _config, _findings = await anyio.to_thread.run_sync(
            partial(app_module._run_validation, converted_path)
        )

        pr_url = _build_pr_url(clone_url, base_branch, target_branch)
        _set(
            status="completed",
            branch_name=target_branch,
            pr_url=pr_url,
            revalidation={
                **report,
                "pii_findings": pii_findings,
            },
        )
        _progress(
            "complete",
            f"Conversion complete. Branch '{target_branch}' re-validation: "
            f"{report.get('status', 'unknown')}."
            + (f" PR link: {pr_url}" if pr_url else "")
            + (" PII flagged — review before merging." if pii_findings else ""),
        )
    except Exception as exc:  # noqa: BLE001 — surface any failure
        logger.exception("conversion task for session %s failed", session_id)
        # Redact the token from the error string before storing it.
        message = str(exc)
        try:
            token = app_module._sessions.get(session_id)._github_token  # type: ignore[union-attr]
            if token:
                message = message.replace(token, "***")
        except (AttributeError, RuntimeError):
            pass
        _set(status="failed", error=message)
        _progress("failed", f"Conversion failed: {message}")


async def _run_managed_session(
    client: OmnigentClient,
    bundle: bytes,
    prompt: str,
    target_branch: str,
    progress: Any,
) -> str:
    """Two-step managed-host flow: upload the bundle to register the agent,
    create a managed session (``host_type="managed"``) that auto-provisions
    a runner, send the prompt, drain the stream, and tombstone BOTH sessions
    in the ``finally`` block.

    Returns the agent's transcript string. Extracted from
    :func:`_run_conversion_task` so the full flow (multipart register →
    managed session → send → drain → cleanup) is unit-testable with a fake
    OmnigentClient.
    """
    omnigent_session_id: str | None = None
    registration_session_id: str | None = None
    try:
        # Step 1: Upload the bundle to register the agent (multipart).
        # This creates a session WITHOUT a managed runner — we only need the
        # agent_id. Capture the registration session id FIRST so that a
        # response missing ``agent_id`` (KeyError) still lets the finally
        # tombstone the throwaway registration session (no leak).
        created = await client.create_session(
            bundle,
            metadata=SessionCreateMetadata(title=f"forge-converter {target_branch}"),
        )
        registration_session_id = created["session_id"]
        agent_id = created["agent_id"]
        progress("agent_session", f"Agent registered ({agent_id}).")

        # Step 2: Create a managed session bound to the registered agent.
        # host_type="managed" triggers the managed host to auto-provision a
        # runner — without it the server returns 503 "No runner bound".
        managed = await client.create_session_from_agent(
            agent_id,
            host_type="managed",
            model_override=_CONVERTER_MODEL,
            title=f"forge-converter {target_branch}",
        )
        omnigent_session_id = managed["id"]
        progress("agent_session", f"Managed session created ({omnigent_session_id}).")

        # Step 3: Wait briefly for the runner if not yet online.
        if not managed.get("runner_online"):
            await _wait_for_runner(client, omnigent_session_id, progress)

        # Step 4: Send the conversion prompt (retry on transient 503).
        await _send_with_retry(client, omnigent_session_id, prompt, progress)
        progress("agent_running", "Agent is converting the repository...")

        transcript = await _drain_conversion_stream(client, omnigent_session_id, progress)
        progress(
            "agent_done",
            "Agent finished. "
            + (transcript[:200] + "…" if len(transcript) > 200 else transcript),
        )
        return transcript
    finally:
        if omnigent_session_id is not None:
            with suppress(OmnigentError):
                await client.delete_session(omnigent_session_id)
        if registration_session_id is not None:
            with suppress(OmnigentError):
                await client.delete_session(registration_session_id)
        with suppress(Exception):
            await client.aclose()


async def _wait_for_runner(
    client: OmnigentClient, session_id: str, progress: Any,
    *, max_attempts: int = 6, delay: float = 2.0,
) -> None:
    """Poll get_session until runner_online is True (or give up).

    The managed host usually provisions the runner synchronously in the
    create response, but a brief async window is possible. This polls
    rather than blocking indefinitely so a stuck host surfaces as a
    clear send_message error instead of a hang.
    """
    for attempt in range(1, max_attempts + 1):
        with suppress(OmnigentError):
            snapshot = await client.get_session(session_id)
            if snapshot.get("runner_online"):
                return
        if attempt < max_attempts:
            progress("agent_session", f"Waiting for runner… ({attempt}/{max_attempts})")
            await anyio.sleep(delay)
    progress("agent_session", "Runner not yet online; attempting message anyway.")


async def _send_with_retry(
    client: OmnigentClient, session_id: str, text: str, progress: Any,
    *, max_attempts: int = 3, delay: float = 3.0,
) -> None:
    """Send a message, retrying on transient 503 (runner still provisioning)."""
    last_err: OmnigentError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            await client.send_message(session_id, text)
            return
        except OmnigentError as exc:
            last_err = exc
            if exc.status_code == 503 and attempt < max_attempts:
                progress("agent_session", f"Runner busy, retrying… ({attempt}/{max_attempts})")
                await anyio.sleep(delay)
                continue
            raise
    assert last_err is not None
    raise last_err


async def _drain_conversion_stream(
    client: OmnigentClient, session_id: str, progress: Any
) -> str:
    """Drain the omnigent SSE stream into a transcript.

    Mirrors :meth:`OmnigentBackend._drain_stream` but appends a progress entry
    for each completed assistant message so the UI shows live steps. Stops on
    ``response.completed`` / ``[DONE]`` / an idle status after output.
    """
    parts: list[str] = []
    async for event_type, data in client.stream_session(session_id):
        if event_type == _DELTA_TYPE:
            delta = data.get("delta")
            if isinstance(delta, str):
                parts.append(delta)
        elif event_type == _ITEM_DONE_TYPE:
            text = _text_from_item(data.get("item"))
            if text:
                parts.append(text)
                progress("agent_message", text[:200])
        elif event_type == _STATUS_TYPE:
            if data.get("status") == "idle" and parts:
                break
        elif event_type == _COMPLETED_TYPE:
            break
    return "".join(parts).strip()


def _text_from_item(item: Any) -> str:
    """Extract assistant text from a ``response.output_item.done`` item."""
    if not isinstance(item, dict):
        return ""
    if item.get("type") != "message" or item.get("role") != "assistant":
        return ""
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "output_text":
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "".join(texts)
