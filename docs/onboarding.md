# Onboarding an agent into forge

This guide walks through everything you must author to optimize **your own
agent** with forge — the smallest valid domain package, the contract each file
must satisfy, and the two ways your agent's code can plug in. It is the "cost of
entry" reference: what is genuinely *required* by the code versus what is merely
convention.

> **TL;DR** — You author **2 Python files + 1 config edit + a valid scaffold**.
> forge locates, loads, runs, and scores your agent by config name alone across
> 50–100 autonomous rounds, keeping/reverting each mutation with git. The only
> semantically-required output field is `EvalReport.aggregate`; the real work is
> entirely inside your `eval.py`.

---

## How you drive forge (three surfaces)

forge has **no Claude Code slash command or prompt** that runs an optimization —
you drive it one of three ways. Only the web wizard is guided step-by-step; the
REST API and the scripts are fire-and-forget (one kickoff runs all rounds
autonomously, and the frontier gate decides keep/revert without asking you).

| Surface | What it is | Guided? |
|---|---|---|
| **Web wizard** | The `forge-orchestrator` Databricks App's single-page UI: select repo → compatibility check → configure → **Start** → watch progress → **Finalize** | ✅ Yes — waits between steps 1–3 |
| **REST API** | `POST /api/session` → `POST /optimize` → poll `GET /api/session/{id}` → `POST /finalize` | ❌ Fire-and-forget |
| **Local scripts** | `scripts/make_baseline.py` then `scripts/run_round.py --rounds N` (§8–§9) | ❌ Fire-and-forget |

> From **Claude Code** in this repo, run the **`forge-onboarding`** skill — it
> verifies the prerequisites below and drives the scripts (or points you at the
> wizard) for you.

Note: forge's *optimizer* internally runs a Claude Agent SDK session as its
mutation engine, but that is forge's own engine — not a session you type into.

---

## 1. Mental model: your agent is a "domain"

forge is **domain-agnostic**. It never hard-codes what your agent does. Instead
your agent is packaged as a **domain** that registers an **eval engine**, and
forge's loop reuses the same machinery for every domain:

```
branch anvil/round-N  →  optimizer proposes ONE mutation  →  commit
                      →  YOUR engine evaluates the mutated agent  →  EvalReport
                      →  frontier gate reads EvalReport.aggregate  →  KEEP (ff-merge) or REVERT (branch -D)
```

The optimizer mutates the **scaffold** (skills / rules / harness YAML) in
`prompt` mode, or an agent Python class in `code` mode. Your engine's one job
each round is: **run the mutated agent over your inputs, score it, and return a
meaningful `aggregate`.**

Three planes stay strictly separated: `runtime/` never imports `optimizer/`,
`eval/` never imports git, and the `loop/` is the only orchestrator. Your domain
lives on the `eval/` side of that line.

---

## 2. The cost of entry

To onboard your agent as engine `<name>`, you author exactly this. Everything
else is convention, not contract:

| # | Artifact | Why it is mandatory |
|---|---|---|
| 1 | `src/anvil/domains/<name>/__init__.py` | Self-registers the engine on import (`register_engine`). |
| 2 | `src/anvil/domains/<name>/eval.py` | The engine function; returns an `EvalReport`. |
| 3 | `harness/config.yaml` → `eval.engine: <name>` (+ endpoints / experiments) | Resolves and loads your engine. |
| 4 | `scaffold/harness.yaml` + at least one `skills/*.md` with `kind: identity` | The runner composes the prompt **before** dispatch and requires ≥1 identity skill. |
| 5 | `eval/runs/baseline.json` (loop only) | Seeds the frontier. Generated "for free" by running `scripts/make_baseline.py` once through your engine. |

Notes:

- **`scaffold/` and `data/golden_set.jsonl` live at the repo root**, not inside
  your domain package. They are passed to your engine as *paths*. Your domain
  package holds only engine code.
- forge never opens `golden_set.jsonl` for you — it just hands you the path.
  Whether you even use a golden set is up to your engine (the trace engine, for
  example, reads a frozen trace snapshot instead).

---

## 3. The minimal valid skeleton

This is the literal smallest domain that loads, registers, and passes the gate —
an `echo` stub. Copy it, rename `echo` → your `<name>`, and replace the body of
`eval.py` with your real run-and-score logic.

### `src/anvil/domains/echo/__init__.py`

```python
from anvil.eval.engines import register_engine        # [CONTRACT] the registration API
from anvil.domains.echo.eval import evaluate_echo      # [OPTIONAL] a lazy-import wrapper is also fine

register_engine("echo", evaluate_echo)                 # [CONTRACT] "echo" MUST equal eval.engine
                                                       #  AND the package directory name
```

A bare top-level `register_engine(...)` call is valid — it runs on import, which
is exactly when forge imports your package.

### `src/anvil/domains/echo/eval.py`

```python
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anvil.eval.runner import EvalReport               # [CONTRACT] required return type


def evaluate_echo(
    *,                                                  # [CONTRACT] keyword-only
    scaffold_root: Path | str,                          # [CONTRACT] always passed — may ignore
    runtime_config_path: Path | str | None = None,      # [CONTRACT] always passed (non-None) — may ignore
    golden_set_path: Path | str = "data/golden_set.jsonl",  # [CONTRACT] path only; core never opens it
    profile: str | None = None,                         # [CONTRACT] may be None — may ignore
    mode: str | None = None,                            # [CONTRACT] always passed (non-None) — may ignore
    **_kwargs: Any,                                     # [CONTRACT] MUST absorb a superset of kwargs
) -> EvalReport:
    return EvalReport(
        aggregate=1.0,          # [CONTRACT] the ONLY semantically-required field — the gate's input
        per_judge={},           # [STUB OK] not read when the Pareto gate is disabled
        per_bucket={},          # [STUB OK] recorded only
        failures=[],            # [STUB OK]
        run_id="",              # [STUB OK] no MLflow run needed
        experiment_id="",       # [STUB OK]
        n_rows=0,               # [STUB OK] recorded only
        mode=mode or "standard",# [STUB OK] echo it back
        scorers=[],             # [STUB OK]
        evaluated_at=datetime.now(UTC).isoformat(timespec="seconds"),  # [CONTRACT] required str
        # trace_ids / cost_metrics / scorer_fingerprint → optional, defaults apply
    )
```

**The engine may ignore all five arguments.** The runner pre-resolves
`runtime_config_path` and `mode`, so you never actually receive `None` for them.

> ⚠️ **This stub is not useful, on purpose.** A constant `aggregate=1.0` ties
> every round against the baseline, and a tie does **not** extend the frontier —
> so a constant-score engine **reverts every round after the first**. The
> skeleton proves what *loads*; the value comes entirely from computing a
> *varying, meaningful* `aggregate` from your mutated agent's behavior.

---

## 4. The `EvalReport` contract

`EvalReport` has ~10 fields, but with the **default gate** (`pareto.enabled =
false`) only one is read for the keep/revert decision:

- **`aggregate` (float) — the decision input.** The frontier compares this
  round's `aggregate` against the best-kept-so-far. Higher must mean better.
- `evaluated_at` (str) — required, but any ISO timestamp works.

Everything else can be empty / dummy for a passing gate:

| Field | Role | Stub? |
|---|---|---|
| `per_judge` | per-objective scores (only read when a **Pareto** gate is enabled) | `{}` |
| `per_bucket` | per-slice breakdown, recorded for inspection | `{}` |
| `failures` | per-row failure records (fuel for a future critic pass) | `[]` |
| `run_id`, `experiment_id` | MLflow linkage; not required — `savesage` ships them empty | `""` |
| `n_rows`, `scorers` | recorded only | `0` / `[]` |
| `trace_ids`, `cost_metrics`, `scorer_fingerprint` | optional; defaults apply | omit |

**The one exception:** if you enable a **Pareto** gate with token/latency
objectives, you must also populate the matching `cost_metrics` key (e.g.
`latency_ms_median`) or the gate raises `ValueError`. Populate `per_judge` too,
since that is what a Pareto gate reads per objective.

---

## 5. The scaffold and config are non-negotiable — even for a no-op engine

Before the runner ever looks at your engine, it calls `load_harness` →
`compose_prompt`. That composition step needs a valid scaffold, so even a stub
engine that ignores everything still requires:

1. **`harness/config.yaml`** with the endpoint / experiment fields, plus your
   `eval.engine: <name>` selector. This file is **immutable** — the optimizer
   must never rewrite it, because it defines the comparison's fixed conditions.
2. **`scaffold/harness.yaml`** — the mutable scaffold the optimizer edits.
3. **At least one skill in `scaffold/skills/*.md` with `kind: identity`.** With
   zero identity skills, composition raises `MissingIdentitySkillError`.

Minimal `harness/config.yaml` engine selector:

```yaml
eval:
  engine: <name>        # defaults to the "genie" builtin if omitted
  # mode: prompt        # optimizer rewrites scaffold markdown/YAML  (default)
  # mode: code          # optimizer rewrites an agent Python class
```

---

## 6. The name-coupling rule (enforced twice)

Three names must be **byte-identical**:

1. `eval.engine` in `harness/config.yaml`
2. the string in `register_engine("<name>", ...)`
3. the package directory `src/anvil/domains/<name>/`

Resolution is by convention: forge does
`importlib.import_module(f"anvil.domains.{name}")` and then expects that import
to have registered `<name>`. The name must match `^[a-z][a-z0-9_]*$`, validated
at **config-parse time** *and* at **load time** (it becomes an import-path
segment, so this is an injection guard). An unknown or misnamed engine fails
**loudly** with `ValueError` — it never silently falls back to the builtin.

---

## 7. Where your agent's code physically lives — two patterns

The import path is **identical** in both patterns (`anvil.domains.<name>`). Only
the physical location of the package differs.

### Pattern A — domain inside forge (works on `main` today)

Your package sits in forge's tree at `src/anvil/domains/<name>/` and loads via a
direct `import_module`. This is how the `savesage` example ships today.

- ✅ Simplest to run right now.
- ❌ Your agent's code is committed into forge — forge is no longer purely
  domain-agnostic for your case.

### Pattern B — domain in your own repo, clone-loaded at runtime (PR #46)

forge **clones your repo** to a per-session temp dir
(`/tmp/forge-sessions/<session-id>`, wiped on shutdown) and appends its
`src/anvil` onto `anvil.__path__` at runtime, so `import anvil.domains.<name>`
resolves *into your clone*. **Nothing is ever written into forge's tree.**

- You point forge at your agent per-session by POSTing `repo_url`
  (+ optional `github_token`, kept in memory only) to the orchestrator. The URL
  may be a `.../tree/<branch>/<subpath>` form; forge parses out the ref and
  optional subdirectory.
- This is the intended end state: **your agent stays in its own repo.**
- Status: delivered on branch `polly/load-domain-from-clone` / **PR #46** (the
  runtime `anvil.__path__` splice with cross-session import isolation). Confirm
  it is merged to `main` before relying on it; on plain `main` without it, use
  Pattern A.

Your own-repo package layout for Pattern B:

```
your-agent-repo/
  src/anvil/domains/<name>/
    __init__.py        # register_engine("<name>", <eval_fn>)
    eval.py            # the engine
  # (+ whatever runtime/scoring modules your engine imports)
```

---

## 8. Priming the baseline (loop only)

The frontier gate needs a starting point. Before running rounds, generate it
once through *your* engine:

```bash
uv run python scripts/make_baseline.py   # → eval/runs/baseline.json
```

The baseline's scorer fingerprint pins the eval conditions. If you later change
the eval set or scorers (e.g. re-ingest a new trace snapshot), the fingerprint
changes, forge's staleness gate fires, and you must re-run `make_baseline.py` —
this is the reproducibility guarantee working as intended, not a bug.

---

## 9. The end-to-end walkthrough

Pattern A (works on merged `main` today):

1. **Create the package.** Copy the skeleton in §3 into
   `src/anvil/domains/<name>/`, renaming `echo` → `<name>`.
2. **Write the real `eval.py`.** Replace the stub body: compose the prompt from
   the scaffold, run your mutated agent over your inputs, score the outputs, and
   return an `EvalReport` whose `aggregate` genuinely varies with quality.
3. **Wire the config.** Set `eval.engine: <name>` (and `mode:`) in
   `harness/config.yaml`, plus the endpoint / experiment fields.
4. **Provide the scaffold.** Ensure `scaffold/harness.yaml` exists with ≥1
   `kind: identity` skill, and `data/golden_set.jsonl` (or your engine's own
   data source) is in place.
5. **Prime the baseline.** `uv run python scripts/make_baseline.py`.
6. **Run rounds.**
   `uv run python scripts/run_round.py --rounds N --eval-mode quick --parent-branch anvil/exp`.

Each round forks `anvil/round-N`, the optimizer mutates the scaffold (or agent
code), your engine scores it, and the frontier gate ff-merges (keep) or deletes
the branch (revert).

---

## 10. Batteries-included: the trace eval engine

Writing a meaningful `eval.py` from scratch is the non-trivial part. If your
agent **already has MLflow traces** carrying LLM-judge and human assessments,
the built-in **`trace`** engine (`src/anvil/domains/trace/`, PR #50) is a
ready-made `eval.py`:

- Ingests once from a single `experiment_id` (keeping only traces with ≥1
  assessment) and **freezes a snapshot** for the whole optimizer session.
- Each round **re-runs the mutated agent** over those trace inputs and re-scores
  against the traces' assessments — so the score reflects the *current* agent,
  not the one that produced the traces.
- Human hard-labels are weighted above LLM-judge rubrics; the exact judge
  recorded on each trace is re-run (with a gateway fallback); single-turn inputs
  are supported (multi-turn is skipped and counted, reserved for future scope).

Config:

```yaml
eval:
  engine: trace
  trace:
    experiment_id: "<the one experiment holding your agent's traces>"
    max_traces: 200
```

This is the natural on-ramp for any agent that already emits traces: you get
forge's rigid, machine-enforced engine seam **and** a live-trace-derived gradient
without hand-writing a scorer.

---

## 11. Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Every round reverts | `aggregate` is constant (a tie never extends the frontier) | Make `aggregate` vary with the mutated agent's quality. |
| `ValueError` at load: unknown engine | `eval.engine`, `register_engine` name, and dir name disagree, or bad chars | Make all three byte-identical and match `^[a-z][a-z0-9_]*$`. |
| `MissingIdentitySkillError` | No `kind: identity` skill in `scaffold/skills/` | Add at least one identity skill. |
| Pareto gate `ValueError` | Objective declared but matching `cost_metrics` / `per_judge` key missing | Populate the matching keys, or use the default (non-Pareto) gate. |
| Staleness gate fires unexpectedly | Eval set / scorers changed → fingerprint changed | Re-run `scripts/make_baseline.py`. |
| `import anvil.domains.<name>` fails on `main` for a clone | The `anvil.__path__` splice (PR #46) is not merged | Use Pattern A, or merge PR #46 / run its branch. |

---

## 12. Seeing the optimization activity — per-round eval traces

forge arms MLflow tracing for **every** engine before it dispatches
(`evaluate_branch` → `anvil.observability.setup_eval_tracing`): it selects the
eval experiment (`experiments.eval`, default `/Shared/anvil-eval`) and enables
`mlflow.openai.autolog`. Because the gateway client (`build_gateway_client`) is
openai-backed, **any** `chat.completions.create` your engine makes is captured
as a `CHAT_MODEL` span in that experiment — no engine code required.

For a clean **one-trace-per-row** view (the summarize call, the judge call,
etc. grouped under a single tagged trace), wrap each row in the helper:

```python
from anvil.observability import eval_row_trace

def _predict_and_score(row):
    with eval_row_trace(
        example_id=row["example_id"], query=row["query"],
        scaffold_root=scaffold_path, runtime_endpoint=model,
    ):
        ...  # your gateway calls autolog as sub-spans under this root CHAIN span
```

The runner passes `trace_rows=True` to your engine when tracing is armed (absorb
it via `**_kwargs`, or declare it and gate the wrap on it so direct/unit calls
stay untraced). Traces are tagged `source=eval` + `scaffold_branch` /
`scaffold_commit_sha`, so a round's traces are identifiable by its
`anvil/round-N` branch. Record the eval `experiment_id` on your `EvalReport`
(best-effort `mlflow.get_experiment_by_name(config.experiments.eval)`) so the
report points at where the traces landed. See `anvil.domains.pitcrew.eval` for
the worked adoption.

## 13. Runtime levers, compound rounds, and model prices

**Levers** let the optimizer change runtime settings that are not prompt
text — the model, or a domain switch like an input format — but only to
values you allow. Declare them in the immutable `harness/config.yaml`:

```yaml
levers:
  model:                      # forge core: swaps the runtime model for every engine
    allowed: [databricks-claude-sonnet-4-6, databricks-claude-haiku-4-5]
  input_mode:                 # any other name: opaque to core, read by your engine
    allowed: [direct_pdf, text]
    default: direct_pdf
```

The optimizer picks a value with a `set_lever` action; the choice is written
to `scaffold/harness.yaml > levers` and validated against `allowed`. Your
engine reads resolved values from `snapshot.config.levers` and the model from
`snapshot.config.effective_runtime_model` (never `runtime_endpoint`, which
stays the base model the baseline records). The judge model is not a lever.
Only list models you have checked work with your agent's inputs (e.g. not
every model accepts a PDF `document` block). The gateway client drops a
`temperature`/`top_p` a model rejects and retries once.

**Compound rounds**: set `loop.max_mutations_per_round` (default 1, max 5) to
let one round apply several mutations together via a `compound` action with a
mandatory `synergy` rationale. Steps apply all-or-nothing; the gate judges the
combined result once.

**Model prices**: point `model_catalog.sheet_id` at a price sheet (columns
`Model`, `Price/Input Token ($/1M)`, `Price/Output Token ($/1M)`, optional
`Context` / `Cache read ($/1M)` / `Provider` / `Notes`) and run
`uv run python scripts/sync_model_catalog.py` to write
`harness/model_catalog.csv`; commit it. Rounds read only the CSV. Prices are
shown to the optimizer next to each `model` value and can be turned into
`cost_metrics.cost_usd_*` by your engine via `anvil.catalog`; latency is never
in a price list, so the prompt shows each value's measured history from past
rounds instead.

## See also

- `CLAUDE.md` — the invariants (plane separation, immutable `harness/config.yaml`).
- `src/anvil/domains/savesage/` — the worked example for Pattern A.
- `src/anvil/domains/trace/` — the trace-driven engine (§10).
