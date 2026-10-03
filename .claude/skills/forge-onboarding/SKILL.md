---
name: forge-onboarding
description: >-
  Start or set up an optimization session with forge, the self-mutating agent
  harness. Use when a user in the forge repo wants to optimize or improve an
  agent, says "start an optimizer session", "run forge on my agent", "optimize
  for latency/cost/quality", "take me through optimization", wants to set up a
  forge domain / eval engine, prime a baseline, or kick off rounds. ALWAYS runs
  an intake first: asks the user every decision the run needs (agent repo,
  objective, models, levers, mutations per round, eval size, budget,
  workspace, forge repo/branch/account, environment variables), writes the
  config, runs preflight checks, and only then starts.
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

1. **Which agent, and where does its code live?** Ask for:
   - the **agent's source repo** — `owner/name` or URL (e.g. `mbihani/pitcrew`)
     or a local path, plus the **branch** and **subdirectory** holding the
     agent (e.g. `document-summarizer/`), and whether it is private (the REST
     path then needs a `github_token`, held in memory only);
   - the **domain / engine** — an existing domain in
     `src/anvil/domains/<name>/`, the built-in `trace` engine (agent already
     has MLflow traces with assessments — ask for the `experiment_id`), or a
     new domain to build (Step 2). If the agent is traced at the *function*
     level or takes non-text input (PDFs, images), the `trace` engine cannot
     ingest it — say so and plan a domain instead.
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

9. **Where does it run?** This machine (local scripts), a sandbox (e.g. an
   Omnigent sandbox — env vars only, no `~/.databrickscfg`, no browser
   logins), or the `forge-orchestrator` app. Which Databricks **workspace**
   (host or profile name) serves the runtime model, the judge, and MLflow?
   Which optimizer **backend** — `local` (Claude Agent SDK subprocess) or
   `omnigent` (managed server; ask for its URL)? The answers decide which
   environment variables Group E must collect.
10. **Which forge repo, branch, and account?** Ask for each by name — do not
    infer them from `git remote -v`, which has pointed at a stale repo before:
    - the **forge repo** the session runs from and pushes results to
      (`owner/name`; list every remote if results go to more than one);
    - the **branch** to start from (the one carrying this domain's code and
      config) and the **parent branch** rounds fork from (default `anvil/exp`);
    - the **GitHub account** used to push (and, for a managed/EMU account,
      that git credentials are set for it — e.g. a repo-local credential
      helper — since the active `gh` account may be a different one);
    - the **git identity** for round commits (`ANVIL_GIT_COMMITTER_NAME` /
      `ANVIL_GIT_COMMITTER_EMAIL`; unset, rounds commit as `anvil-bot`).
11. **Refresh model prices?** If `model_catalog.sheet_id` is set, offer to
    re-sync now (`scripts/sync_model_catalog.py`, needs
    `gcloud auth application-default login`) and commit the CSV diff.
12. **Where do this domain's MLflow experiments live?** Every domain gets
    its own set, derived from `experiments.root` (default `/Shared/forge`) and
    the engine name — e.g. for `pitcrew`:
    `/Shared/forge/pitcrew/eval` (one trace per eval row, tagged with the
    round's branch), `/Shared/forge/pitcrew/optimizer` (one run per session:
    optimizer transcripts, critiques, improvement summary) and
    `/Shared/forge/pitcrew/runtime` (deployed agent). Ask only:
    - **the root** — keep `/Shared/forge`, or a per-user folder such as
      `/Users/<email>/forge`. They are created on first use if the identity
      can write that folder. An explicit path outside `<root>/<domain>/`
      makes the config fail to load, so domains can never share experiments.
    - **optimizer transcripts on or off** (`persistence.enabled`, default on;
      `ANVIL_PERSIST_OPTIMIZER_ARTIFACTS` overrides it).
    - For the `trace` engine, the agent's OWN experiment (the dataset) is a
      separate, pre-existing one — ask for its id.

### Group E — Environment variables

Check which variables are set **by name only** — never print, log, or commit
a value; `env | cut -d= -f1 | sort` or `[ -n "$VAR" ] && echo set`. Ask the
user only for what is missing for the chosen path. For secrets, ask them to
set the variable themselves (in Claude Code: `! export NAME=...`, or their
sandbox's secret store) rather than pasting it into the chat.

| Variable | Needed when | What it does |
|---|---|---|
| `DATABRICKS_CONFIG_PROFILE` | local runs (or pass `--profile`) | Profile in `~/.databrickscfg` for the gateway (runtime + judge) and MLflow |
| `DATABRICKS_HOST` + `DATABRICKS_TOKEN` | sandbox / no profile | Workspace URL + token for the same. `DATABRICKS_TOKEN` is read by **every** path (runtime, judge, MLflow, and the optimizer's gateway auth). Fine when everything is on one workspace. If the optimizer's gateway is on a different workspace from the agent, do **not** set it — give each side its own auth (profiles); a token for workspace A sent to workspace B's gateway fails silently as an empty optimizer transcript |
| `MLFLOW_TRACKING_URI` | sandbox without a profile — set `databricks` | Where traces and runs go. forge sets `databricks` itself when it sees Databricks credentials and nothing else chose a URI; if it is set to anything else (`sqlite:`, `file:`, a path), traces stay **local** — in a sandbox they are lost when it is wiped. Ask before accepting a non-Databricks value |
| `ANVIL_AI_GATEWAY_URL` | `local` optimizer backend — **required** | The optimizer's Anthropic route, `https://<workspace-id>.ai-gateway.cloud.databricks.com/anthropic` (not `<host>/serving-endpoints/anthropic`). Unset ⇒ every round raises at start, unless `ANTHROPIC_BASE_URL` is already set — which it often is, **inherited from the coding harness you are running in** (e.g. Claude Code), silently sending the optimizer to that harness's workspace. If it is set, ask which workspace it points at and whether that is intended |
| `ANTHROPIC_AUTH_TOKEN` | optional | Overrides the optimizer's gateway auth (normally from the Databricks profile / host) |
| `OMNIGENT_SERVER_URL`, `OMNIGENT_AUTH_TOKEN` | `omnigent` backend | Server URL and bearer token; override `optimizer.server_url` / `auth_token` |
| `ANVIL_OPTIMIZER_BACKEND` | optional — **overrides config** | `local` / `omnigent`; wins over `optimizer.backend`. If set to something other than the user's Q9 answer, flag it |
| `ANVIL_PERSIST_OPTIMIZER_ARTIFACTS` | optional — **overrides config** | `0` / `1`; wins over `persistence.enabled` (optimizer transcripts in MLflow) |
| `ANVIL_GATEWAY_BASE_URL` | optional | Runtime + judge base URL; default `<DATABRICKS_HOST>/serving-endpoints` |
| `ANVIL_GIT_COMMITTER_NAME`, `ANVIL_GIT_COMMITTER_EMAIL` | optional | Author of round commits; unset ⇒ `anvil-bot <anvil-bot@users.noreply.github.com>` |
| `ANVIL_GOOGLE_QUOTA_PROJECT` | price re-sync only | Google quota project for the Sheets API (default `gcp-dev-field-eng-aiapiquota`) |
| Domain-specific (e.g. `SAVESAGE_STATEMENT_AGENT_PATH`, `SAVESAGE_LUNA_PROFILE`) | that domain | Find them with `grep -rn "environ\|getenv" src/anvil/domains/<name>/` and ask for each |

Leave alone (forge sets these itself): `MLFLOW_GENAI_EVAL_MAX_WORKERS` (from
`eval.n_workers`), `MLFLOW_ENABLE_ASYNC_TRACE_LOGGING`, the `ANTHROPIC_*`
model defaults (from `optimizer_endpoint`), and the deploy-time
`ANVIL_SCAFFOLD_*` / `FORGE_CRASH_LOG_WORKSPACE_PATH`.

End the intake with a summary table of every answer — repos and branches by
name, environment variables by name with set / missing (never values) — and
get an explicit "go" before writing anything.

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
   `git branch --show-current` is the branch from question 10; `git remote -v`
   points at the repo(s) from question 10, and `git ls-remote <remote>`
   succeeds with the push account's credentials.
2. **Environment complete and consistent** (names only, never values) —
   every variable Group E marked as needed is set; no override variable
   (`ANVIL_OPTIMIZER_BACKEND`, `ANVIL_PERSIST_OPTIMIZER_ARTIFACTS`,
   `OMNIGENT_SERVER_URL`) contradicts the user's answers; you can say which
   workspace the runtime gateway, the optimizer gateway (`ANVIL_AI_GATEWAY_URL`
   or an inherited `ANTHROPIC_BASE_URL`), and MLflow each point at — and if
   the optimizer's differs from the agent's, `DATABRICKS_TOKEN` is unset.
   Confirm reachability with one cheap call each: a one-token gateway
   completion on `runtime_endpoint`, and
   `mlflow.get_experiment_by_name(<experiments.eval>)`.
3. **Engine loads and config is valid** —
   `uv run python -c "from anvil.eval.engines import load_engine; from anvil.runtime.loader import load_harness; load_engine('<engine>'); s = load_harness('scaffold'); print(s.config.effective_runtime_model, s.config.levers)"`.
   This also validates every lever value against its allowlist.
4. **Every allowed model works with the agent's real input** — make one real
   call per model with the agent's actual input shape (not a text-only ping).
   Drop any model that errors and tell the user why. Known traps: GPT and
   Gemini reject Anthropic-style PDF `document` blocks; some newer models reject
   `temperature` (the gateway client retries without it — confirm it succeeds).
5. **Lever dependencies present** — e.g. `uv run python -c "import fitz"` before
   allowing `input_mode: text`.
6. **Prices known** — `scripts/sync_model_catalog.py --check <allowed models>`;
   unpriced models still run, but tell the user their cost will be blank.
7. **Traces land in the workspace** — `uv run python -c "from anvil.observability
   import configure_tracking_uri as c; print(c('<profile or DEFAULT>'))"` must
   print a URI starting with `databricks` (`run_round.py` prints it too; any
   path warns when it is local). After the baseline (step 8), confirm in the
   workspace that the
   `<root>/<domain>/eval` experiment exists and holds new traces (a local
   SQLite experiment id is a small integer like `1`; a workspace id is a
   long number) and, once rounds start, that the `<root>/<domain>/optimizer`
   experiment has a run tagged `anvil.surface=cli`.
8. **Baseline matches the run** — regenerate it now, live, at the chosen eval
   size: `uv run python scripts/make_baseline.py --mode <mode>` and commit
   `eval/runs/baseline.json`. Its `mode` must equal the rounds' `--eval-mode`,
   it must be produced by the same scaffold/levers the rounds start from, and
   for a latency objective its `cost_metrics` must contain
   `latency_ms_median`. Never reuse a cached or different-size baseline: the
   gate would compare unlike runs and keep or revert on noise.
9. **Parent branch** — `anvil/exp` exists at the baseline commit
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

- `docs/onboarding.md` — domain skeleton and contract (§2–§7), baseline
  conditions (§8), trace engine (§10), failure modes (§11), per-row tracing
  (§12), levers / compound rounds / prices (§13), per-domain MLflow
  experiments and where traces land (§14), and the session intake summary.
- `CLAUDE.md` — invariants (plane separation, immutable `harness/config.yaml`).
- `src/anvil/domains/pitcrew/` — worked domain with a latency objective, a
  model lever, a domain lever (`input_mode`), and cost metrics.
