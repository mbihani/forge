---
name: forge-onboarding
description: >-
  Start or set up an optimization session with forge, the self-mutating agent
  harness. Use when a user in the forge repo wants to optimize or improve an
  agent, says "start an optimizer session", "run forge on my agent", "optimize
  for latency/cost/quality", "take me through optimization", wants to set up a
  forge domain / eval engine, prime a baseline, or kick off rounds. ALWAYS runs
  an intake first: asks the user every decision the run needs (objective,
  models, levers, mutations per round, eval size, budget, workspace, where
  results go), writes the config, runs preflight checks, and only then starts.
---

# Starting a forge optimization session

forge optimizes an agent by treating its **scaffold** (skills, rules, sampling,
levers) — or its agent code — as data an **optimizer** rewrites round by round,
keeping or reverting each change with `git` based on an **eval score** and a
**frontier gate**. This skill takes a user from "I want to optimize my agent"
to a correctly configured, running session.

**The rule: no round, baseline, or config edit before the intake (Step 1) is
answered and the preflight (Step 3) passes.** A session started on a wrong
assumption burns hours: every question below exists because skipping it once
produced a wasted or misleading run.

## How rounds behave (tell the user up front)

Once started, forge is **fire-and-forget**: one kickoff runs the baseline plus
all rounds autonomously, and the gate decides keep/revert without asking. So
every decision the user cares about must be made in the intake, not mid-run.
forge's optimizer internally runs its own Claude Agent SDK session (the
`local` backend) or a managed Omnigent session (`omnigent`); that is forge's
engine, not a session the user types into.

## Step 1 — Intake: ask before doing anything

First read what already exists so questions offer real choices, not blanks:
`harness/config.yaml` (engine, mode, `levers`, `gate`, `loop`, `model_catalog`),
`scaffold/harness.yaml`, `eval/runs/baseline.json`, `eval/runs/frontier.json`,
the latest `eval/runs/round_*.json`, `git branch -a`, and `git remote -v`.
Where a current value exists, offer it as the default.

Ask with the harness's structured-question tool when it has one (in Claude
Code: `AskUserQuestion`, at most 4 questions per call, so ask in the groups
below); otherwise ask in plain text and wait. Record every answer.

### Group A — What to optimize

1. **Which agent / domain?** An existing domain in `src/anvil/domains/<name>/`,
   the built-in `trace` engine (agent already has MLflow traces with
   assessments), or a new domain to build (see Step 2). If the agent is
   traced at the *function* level or takes non-text input (PDFs, images), the
   `trace` engine cannot ingest it — say so and plan a domain instead.
2. **Objective?** This decides the gate, so never assume it:
   - *Quality* — maximize the aggregate (`gate.pareto.enabled: false`).
   - *Latency (or cost) with a quality floor* — Pareto gate: minimize
     `latency` (`cost_metrics.latency_ms_median`) and keep `aggregate` within an
     epsilon of best.
   - *Both* — Pareto on quality and latency, each with its own epsilon.
   Point out where the current agent stands (e.g. already ~0.97 quality but
   30s/doc ⇒ latency is likely the higher-value target).
3. **What may the optimizer change?** `prompt` mode (skills/rules/sampling
   markdown) or `code` mode (an agent Python module in `agents/`). Note that
   most latency/cost wins come from **levers** (Group B), which work in both.

### Group B — Search space

4. **Which runtime models may it try?** (multi-select) These become the
   `levers.model.allowed` list; the first should be the current
   `runtime_endpoint`. Show each candidate's price from the price list
   (`uv run python scripts/sync_model_catalog.py --check <models...>`). Only
   models that pass the Step 3 probe stay in the list. The judge model is not
   a lever and stays fixed.
5. **Other levers to expose?** Domain switches the engine implements (e.g.
   pitcrew `input_mode: [direct_pdf, text]`). Ask the user which to allow and
   warn about any extra dependency a value needs (e.g. `text` needs pymupdf).
6. **Mutations per round?** `loop.max_mutations_per_round`: 1 (classic, easiest
   to attribute), 2, or 3 (lets the optimizer combine near-misses — two
   changes each smaller than the epsilon — in one `compound` round). Max 5.

### Group C — Measurement and budget

7. **Eval size?** `quick` / `standard` / `full` (row counts in `eval.modes`).
   Explain the trade-off: small sets are fast but noisy — LLM-judge scores and
   latency medians wobble enough at ~8 rows that real improvements revert. Use
   the largest size the budget allows for latency objectives.
8. **How many rounds, and the gate margins?** Rounds (forge targets 50+;
   rounds are resumable) and each objective's `epsilon` (e.g. latency 2000ms,
   quality 0.03). Show the measured noise if earlier rounds exist; an epsilon
   below the noise keeps noise, one far above it blocks real wins.

### Group D — Where it runs and where results go

9. **Workspace and auth?** Databricks profile (`--profile`) for local runs, or
   `DATABRICKS_HOST` + `DATABRICKS_TOKEN` in a sandbox. Optimizer backend
   `local` or `omnigent` (`optimizer.backend` + `server_url`).
10. **Which repo/branch receives results?** Confirm the git remote the session
    pushes to (`git remote -v`) — clones have pushed to a stale repo before.
    Confirm the parent branch (default `anvil/exp`).
11. **Refresh model prices?** If `model_catalog.sheet_id` is set, offer to
    re-sync now (`scripts/sync_model_catalog.py`, needs
    `gcloud auth application-default login`) and commit the CSV diff.

End the intake with a summary table of every answer and get an explicit
"go" before writing anything.

## Step 2 — Write the configuration

Make the edits the answers imply, then show the user the diff:

- `harness/config.yaml` (immutable during the run): `mode`, `eval.engine`,
  `eval.default_mode`, `gate` (`type: frontier`, `pareto.enabled`,
  `objectives` with `direction` / `source` / `epsilon`), `loop`
  (`max_mutations_per_round`, `max_optimizer_turns`), `levers` (allowlists +
  defaults + descriptions), `optimizer` backend, `model_catalog`.
- `scaffold/harness.yaml`: leave `levers:` unset (defaults apply) unless the
  user wants a non-default starting point — that starting point must then be
  what the baseline is measured with.

If the agent has no domain yet, build one first — `docs/onboarding.md` §2–§7
(minimal skeleton, `EvalReport` contract, name-coupling rule). A latency or
cost objective needs the engine to fill `cost_metrics.latency_ms_median`
(and optionally `cost_usd_*` via `anvil.catalog`); a domain lever needs the
engine to read `snapshot.config.levers`; a model lever needs it to call
`snapshot.config.effective_runtime_model`.

## Step 3 — Preflight (all must pass before kickoff)

Run each check and report pass/fail; fix or go back to the user on any fail.

1. **Clean tree, right branch, right remote** — `git status --short` empty;
   `git remote -v` points at the repo from question 10.
2. **Engine loads and config is valid** —
   `uv run python -c "from anvil.eval.engines import load_engine; from anvil.runtime.loader import load_harness; load_engine('<engine>'); s = load_harness('scaffold'); print(s.config.effective_runtime_model, s.config.levers)"`.
   This also validates every lever value against its allowlist.
3. **Every allowed model works with the agent's real input** — make one real
   call per model with the agent's actual input shape (not a text-only ping).
   Drop any model that errors and tell the user why. Known traps: GPT and
   Gemini reject Anthropic-style PDF `document` blocks; some newer models reject
   `temperature` (the gateway client retries without it — confirm it succeeds).
4. **Lever dependencies present** — e.g. `uv run python -c "import fitz"` before
   allowing `input_mode: text`.
5. **Prices known** — `scripts/sync_model_catalog.py --check <allowed models>`;
   unpriced models still run, but tell the user their cost will be blank.
6. **Baseline matches the run** — regenerate it now, live, at the chosen eval
   size: `uv run python scripts/make_baseline.py --mode <mode>` and commit
   `eval/runs/baseline.json`. Its `mode` must equal the rounds' `--eval-mode`,
   it must be produced by the same scaffold/levers the rounds start from, and
   for a latency objective its `cost_metrics` must contain
   `latency_ms_median`. Never reuse a cached or different-size baseline: the
   gate would compare unlike runs and keep or revert on noise.
7. **Parent branch** — `anvil/exp` exists at the baseline commit
   (`git branch -f anvil/exp HEAD` after committing the baseline).

## Step 4 — Kick off and report

Local / sandbox scripts:

```bash
uv run python scripts/run_round.py --parent-branch anvil/exp \
  --eval-mode <mode> --rounds <N> --max-turns <turns> [--profile <profile>]
```

Wrap long local runs in `caffeinate -dimsu` on macOS (sleep kills them and
looks like timeouts). Re-invoking continues from the next round id.

After (or during) the run, report per round: decision (keep / revert / noop /
infra_fail / apply_rejected), action (for `compound`, each step), the levers it
ran with (`round_NNN.json > levers`), objective values vs best-so-far, and
cost. Per-row traces are in the `experiments.eval` MLflow experiment (tagged
`scaffold_branch=anvil/round-N`); optimizer transcripts in
`experiments.optimizer`. Then push the results to the remote from question 10.

## Other surfaces

The intake still applies — ask it before using either.

**Web wizard** — the `forge-orchestrator` Databricks App serves a guided UI at
`GET /`: select repo → **Validate** → compatibility check → configure → **Start
Optimization** → watch progress → **Finalize**. It is a browser flow; point the
user at the app URL.

**REST API** (fire-and-forget):

```
POST /api/session            {"repo_url": "...", "github_token": "<optional>"}
POST /api/session/{id}/optimize   {"eval_mode": "quick", "max_rounds": 10, "max_turns": 30, "mode": "prompt"}
GET  /api/session/{id}            # poll: status + rounds[] + frontier + baseline
POST /api/session/{id}/finalize   # held-out eval, locks the run
```

`repo_url` may be a GitHub URL (cloned to `/tmp/forge-sessions/<id>`, wiped on
shutdown), a `.../tree/<branch>/<subpath>` URL, or a local path.

> **Flag-name gotcha:** row-set size is `--mode` in `evaluate.py` /
> `make_baseline.py` but `--eval-mode` in `run_round.py`. `prompt` vs `code` is
> not a `run_round.py` flag — it is `mode:` in `harness/config.yaml`.

## See also

- `docs/onboarding.md` — domain skeleton and contract (§2–§7), trace engine
  (§10), per-row tracing (§12), levers / compound rounds / prices (§13), and
  the session intake summary.
- `CLAUDE.md` — invariants (plane separation, immutable `harness/config.yaml`).
- `src/anvil/domains/pitcrew/` — worked domain with a latency objective, a
  model lever, a domain lever (`input_mode`), and cost metrics.
