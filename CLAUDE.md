# ANVIL — project-specific rules for Claude

Minimal rules that prevent recurring mistakes in this repo. For full
architectural context, read `docs/decisions.md`. For current state and
next steps, read `docs/plan.md`.

## Stack constraints

- **LLM client:** `openai` SDK pointed at Databricks serving endpoints.
  Never import or add `anthropic` (or any other provider SDK) to runtime
  dependencies. Model choice is config (`runtime_endpoint`,
  `optimizer_endpoint` in `scaffold/harness.yaml`), not code.
- **Agent interface:** `mlflow.pyfunc.ResponsesAgent`. Never use or
  propose `ChatAgent` — it is legacy-ish at Databricks.
- **No agent-framework abstraction** inside the runtime agent. No
  LangGraph, no DSPy, no CrewAI. Tool-calling loop is plain Python. The
  whole point of ANVIL is that the optimizer can read and mutate the
  scaffold — framework abstractions fight that.

## Storage rules

- **Scaffold (skills, memory, workflow rules, harness config)** →
  files in `scaffold/` (markdown + YAML), git-tracked, synced to
  `anvil.default.scaffold` UC Volume on deploy. Never put scaffold in a
  Delta table. Matches the MiniMax M2.7 reference pattern.
- **Operational artifacts** → Delta. Only two today:
  1. `anvil.default.mutations` — append-only round log.
  2. MLflow-managed trace Delta (native sync). Eval aggregates are a
     view over it, not a custom table.
- **Lakebase is not used** for anything in the current scope. Only
  reconsider if we add high-QPS per-conversation runtime memory.

## Workflow rules

- **Catalog is `anvil`** (created with a managed location on the
  workspace's default external storage). Schema is `default`. When
  writing SQL or DAB resources, use `anvil.default.*` — do not use
  `main.anvil.*` or hive-style namespacing.
- **Databricks profile:** the harness reads the active profile from
  the `--profile` flag (default `DEFAULT`). Set it once per workspace;
  never switch mid-run without explicit user instruction.
- **Iteration target** is 50+ rounds (stretch 100). Don't propose
  10–20 rounds for the loop — M2.7 needed 100+ for a visible curve.
- **Keep/revert uses git branches,** not Delta `RESTORE TABLE`. Each
  round is `anvil/round-N`; keep = fast-forward merge; revert =
  `git branch -D`.

## Starting an optimization session

- When the user wants to start, set up, or resume an optimization run, use
  the **`forge-onboarding`** skill (`.claude/skills/forge-onboarding/`). Run
  its intake — agent repo, objective, models, levers, mutations per round,
  eval size, budget, workspace, forge repo/branch/account, environment
  variables (checked by name, never echoed) — and its preflight before writing config
  or starting any baseline or round. Never assume the objective (quality vs
  latency/cost); ask.

## When proposing architecture changes

- Cite MiniMax M2.7 (`research/minimax-m27-*.md`) if relevant — those
  files are the authoritative reference the design is calibrated
  against.
- Call out explicitly when something would diverge from M2.7's pattern
  and why. Divergence is allowed; it just needs justification.

## What not to auto-create

- Do **not** create a `README.md`, `LICENSE`, or `CONTRIBUTING.md` on
  your own. These are queued as "Reusability queue" items in
  `docs/plan.md` and should be written deliberately when the user is
  ready to share the repo.
- Do **not** create executive briefs, demo scripts, or blog posts
  unless the user explicitly asks via the matching skill.
