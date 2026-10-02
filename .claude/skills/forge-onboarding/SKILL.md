---
name: forge-onboarding
description: >-
  Optimize your own agent with forge, the self-mutating agent harness. Use when
  a user working in the forge repo wants to get started optimizing or improving
  an agent, asks "how do I run forge", "optimize my agent with forge", "take me
  through optimization", wants to set up a forge domain / eval engine, prime a
  baseline, or kick off optimization rounds. Explains the three ways to drive
  forge (guided web wizard, REST API, local scripts), the prerequisites, and the
  exact commands to run.
---

# Onboarding an agent into forge

forge optimizes an agent by treating its **scaffold** (skills, rules, memory,
sampling, tool registry) — or its agent code — as data that an **optimizer**
rewrites one mutation at a time, keeping or reverting each change with `git`
based on an **eval score**. This skill walks a user from "I have an agent" to a
running optimization.

## First, clear up the "Claude Code" confusion

There is **no prompt you type into your own Claude Code that runs an
optimization round.** forge is driven through a **web wizard, a REST API, or
local shell scripts** (below). This skill's job is to drive those for the user.

Separately, forge's **optimizer** *internally* runs a Claude Agent SDK session
as its mutation engine (the `local` backend) or runs that same agent on a
managed Omnigent server (the `omnigent` backend, which the deployed app forces).
That internal Claude session is forge's engine proposing one mutation per round
— it is **not** a session the user types into. If a user says "I want to use
Claude Code to optimize my agent," they almost always mean "help me drive forge"
— use this skill to do that.

## The interaction model: fire-and-forget, the gate decides

- The **web wizard is the only guided, wait-between-steps surface** (and it is a
  browser UI, not a terminal). It walks the user: validate repo → review
  compatibility checks → configure → **Start** → watch progress → **Finalize**.
- The **REST API and the scripts are fire-and-forget**: one kickoff runs the
  baseline plus all `--rounds N` autonomously to completion while the user
  watches progress. forge **never pauses to ask the user to approve a
  mutation** — each round's keep/revert is decided automatically by the frontier
  gate on the eval score (`EvalReport.aggregate`).

Set this expectation explicitly so the user is not waiting for a prompt that
never comes.

## Step 1 — Confirm the prerequisites

Before any round can run, the agent must be a valid forge **domain** and a
**baseline** must exist. Verify (or help the user create) all of:

1. **A registered domain** — `src/anvil/domains/<name>/__init__.py` calls
   `register_engine("<name>", <eval_fn>)`, and `src/anvil/domains/<name>/eval.py`
   returns an `EvalReport` with a *meaningful, varying* `aggregate` (a constant
   score reverts every round). See `docs/onboarding.md` for the minimal
   skeleton and the full contract.
2. **Config selects the engine** — `harness/config.yaml` has `eval.engine:
   <name>` (plus `mode: prompt|code`, endpoints, and `eval.modes` row counts).
   The three names (`eval.engine`, the `register_engine` string, the package
   directory) must be **byte-identical** and match `^[a-z][a-z0-9_]*$`.
3. **A valid scaffold** — `scaffold/harness.yaml` plus **≥1
   `scaffold/skills/*.md` with `kind: identity`** (else prompt composition
   raises `MissingIdentitySkillError`).
4. **Parent branch `anvil/exp`** — the loop forks round branches from it. The
   scripts expect it; the orchestrator auto-creates it if missing.
5. **A primed baseline** — `eval/runs/baseline.json`, generated once by
   `scripts/make_baseline.py` (Step 2a). The frontier gate needs this seed.

**Shortcut for agents that already emit MLflow traces:** if the agent has an
MLflow experiment with LLM-judge and/or human assessments, skip hand-writing
`eval.py` and use the built-in **`trace`** engine instead:

```yaml
eval:
  engine: trace
  trace:
    experiment_id: "<the experiment holding the agent's traces>"
    max_traces: 200
```

It freezes a trace snapshot once, then each round re-runs the mutated agent over
those inputs and re-scores against the recorded assessments. See
`docs/onboarding.md` §10.

## Step 2 — Pick a surface and drive it

Ask the user which they want; default to the **guided web wizard** when they say
"take me through it," and to the **local scripts** when they are working in a
terminal / coding agent.

### 2a. Local scripts (the terminal / coding-agent path)

Environment setup (once):

```bash
uv venv --python 3.12
uv sync --extra dev --extra optimizer
```

Sanity-check the runtime + scorers without mutating anything:

```bash
uv run python scripts/evaluate.py --mode quick
```

Prime the baseline (**required once before rounds** — writes
`eval/runs/baseline.json`):

```bash
uv run python scripts/make_baseline.py --mode quick
```

Run the optimization rounds (this is the kickoff; it runs to completion):

```bash
uv run python scripts/run_round.py --rounds 10 --eval-mode quick --parent-branch anvil/exp
```

Each round forks `anvil/round-N`, the optimizer proposes one mutation, your
engine scores it, and the frontier gate ff-merges (keep) or deletes the branch
(revert).

> **Flag-name gotcha:** the row-set size is `--mode` in `evaluate.py` and
> `make_baseline.py`, but `--eval-mode` in `run_round.py`. The `prompt` vs
> `code` optimization mode is **not** a `run_round.py` flag — set it via
> `harness/config.yaml` `mode:` (or the REST/wizard `mode` field).

Useful `run_round.py` flags: `--rounds` (default 1), `--parent-branch` (default
`anvil/exp`), `--eval-mode {quick,standard,full}`, `--max-turns` (default 30),
`--profile`, `--allow-dirty`, `--force`.

Finalize (held-out eval; requires `eval.held_out_test: true` in config):

```bash
uv run python scripts/finalize.py --mode test
```

### 2b. Guided web wizard (the "take me through it step-by-step" surface)

The orchestrator app serves a single-page wizard at `GET /` (deployed as the
Databricks App `forge-orchestrator`). Point the user at the running app URL and
walk them through:

1. **Select Agent Repository** → paste the agent repo URL (or a local path) →
   **Validate Repository**.
2. **Compatibility Check** → review the validation checks → **Proceed to
   Optimization** (a **Convert to forge-compatible** button appears instead if
   the repo needs conversion).
3. **Configure Optimization** → edit params (mode, eval mode, max rounds, max
   turns, MLflow experiment) → **Start Optimization**.
4. **Optimization Progress** → rounds appear one at a time (the page polls every
   few seconds).
5. **Finalize** when satisfied.

This is a browser flow — the user clicks through it; you cannot drive it from a
terminal.

### 2c. REST API (programmatic / fire-and-forget)

```
POST /api/session
    {"repo_url": "https://github.com/you/your-agent", "github_token": "<optional>"}
    → clones + validates; returns {session_id, status, validation, config}

POST /api/session/{id}/optimize            (returns 202, runs in background)
    {"eval_mode": "quick", "max_rounds": 10, "max_turns": 30, "mode": "prompt"}
    # all fields optional; empty {} uses defaults (10 rounds, prompt mode)

GET  /api/session/{id}                     # poll: status + rounds[] + frontier + baseline
POST /api/session/{id}/finalize            # held-out eval, locks the run
```

`repo_url` may be a GitHub URL (cloned to an ephemeral `/tmp/forge-sessions/<id>`
dir, wiped on shutdown), a `.../tree/<branch>/<subpath>` URL, or a local path
(used in place, not cloned). `github_token` is held in memory only.

## Step 3 — Report what happened

After kickoff, surface the per-round outcomes (decision keep/revert/noop/
infra_fail, action, baseline vs mutated aggregate, delta) and the finalize
result. The durable record of each round's eval lives in MLflow; per-round
artifacts (`eval/runs/round_NNN.json`, `scaffold/memory/round_NNN_*.md`) are
written into the working tree.

## See also

- `docs/onboarding.md` — the full "cost of entry": the minimal domain skeleton,
  the `EvalReport` contract, the name-coupling rule, Pattern A vs B, and the
  trace engine.
- `CLAUDE.md` — the invariants (plane separation, immutable `harness/config.yaml`).
- `src/anvil/domains/savesage/` — a worked domain example;
  `src/anvil/domains/trace/` — the trace-driven engine.
