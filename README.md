# ANVIL

A self-mutating agent harness on Databricks. The runtime is an
`mlflow.pyfunc.ResponsesAgent`; the optimizer is a Claude Agent SDK
session that mutates the agent's **scaffold** (skills, rules, memory,
sampling, tool registry) round by round, with `git` + MLflow evals
as gate.

## Architecture (5 planes, physically separated)

| Plane | Path | Knows about | Output |
|---|---|---|---|
| Runtime | `src/anvil/runtime/` | how to compose a prompt and answer | trace + response |
| Eval | `src/anvil/eval/` | how to run `mlflow.genai.evaluate` | `EvalReport` + JSON |
| Optimizer | `src/anvil/optimizer/` | how to propose a mutation | `OptimizerAction` + critique |
| Loop | `src/anvil/loop/` | git, branches, baselines, decisions | round artifacts + Delta row |
| Observability | `src/anvil/observability.py` | autolog + standard tag set | tagged traces |

The runtime never imports from the optimizer. The eval never imports
from git. The loop is the only orchestrator.

## Quickstart

```bash
uv venv --python 3.12
uv sync --extra dev --extra optimizer

# Quick eval, no mutation — sanity-check the runtime + scorers (~3-5 min)
uv run python scripts/evaluate.py --mode quick

# Prime the baseline (required once before rounds) → eval/runs/baseline.json
uv run python scripts/make_baseline.py --mode quick

# One optimization round end-to-end
uv run python scripts/run_round.py --rounds 1 --eval-mode quick --parent-branch anvil/exp
```

> Row-set size is `--mode` in `evaluate.py` / `make_baseline.py` but
> `--eval-mode` in `run_round.py`.

Prefer a guided, step-by-step flow? Open the orchestrator **web wizard** (the
`forge-orchestrator` Databricks App) and click through: select repo →
compatibility check → configure → **Start** → watch progress → **Finalize**.
To onboard **your own** agent, see [`docs/onboarding.md`](docs/onboarding.md),
or run the `forge-onboarding` skill from Claude Code in this repo.

## Storage

- **Scaffold** → `scaffold/` (markdown + YAML, git-tracked).
- **Mutations log** → `anvil.default.mutations` (Delta append-only).
- **Traces** → MLflow in the Databricks workspace, one experiment set per
  domain: `/Shared/forge/<domain>/{eval,optimizer,runtime}`
  (`harness/config.yaml > experiments`).
- **Per-round eval JSON** → `eval/runs/round_NNN.json`.
- **Per-round critique** → `scaffold/memory/round_NNN_critique.md`.
