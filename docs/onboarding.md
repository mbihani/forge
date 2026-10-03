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
> asks you the session intake below, writes the config, runs the preflight
> checks, and drives the scripts (or points you at the wizard) for you.

Note: forge's *optimizer* internally runs a Claude Agent SDK session as its
mutation engine, but that is forge's own engine — not a session you type into.

### Session intake — decide these before any round runs

Rounds run unattended once started, so every decision is made up front. Any
coding harness driving forge should ask these and wait for answers
(`.claude/skills/forge-onboarding/SKILL.md` has the full wording and how each
answer maps to config):

| # | Decision | Sets |
|---|---|---|
| 1 | Which agent: source repo (`owner/name` or path, branch, subdirectory, private?) and domain (existing, `trace` engine, or new) | `eval.engine`, domain code |
| 2 | Objective: quality, latency/cost with a quality floor, or both | `gate.pareto` objectives |
| 3 | What may change: prompt scaffold or agent code | `mode` |
| 4 | Which runtime models may be tried (each verified on the agent's real input) | `levers.model.allowed` |
| 5 | Other domain levers to allow (and their dependencies) | `levers.<name>` |
| 6 | Mutations per round (1–5) | `loop.max_mutations_per_round` |
| 7 | Eval size (quick / standard / full) | `eval.default_mode`, `--eval-mode` |
| 8 | Rounds and per-objective gate margins | `--rounds`, `gate…epsilon` |
| 9 | Where it runs (local / sandbox / app), which workspace, optimizer backend | `--profile`, `optimizer.backend` |
| 10 | Forge repo(s), starting branch, parent branch, and push account — each by name | git remotes + credentials, `--parent-branch` |
| 11 | Re-sync model prices now? | `harness/model_catalog.csv` |
| 12 | Root folder for the domain's own MLflow experiments (`<root>/<domain>/{eval,optimizer,runtime}`, auto-created) and whether to keep optimizer transcripts | `experiments.root`, `persistence.enabled` |
| 13 | Environment variables for that path (checked by name, never echoed) | see below |

**Environment variables** (the skill's Group E has the full table):

- **Always:** Databricks auth — `DATABRICKS_CONFIG_PROFILE` locally, or
  `DATABRICKS_HOST` + `DATABRICKS_TOKEN` in a sandbox. Runtime, judge, MLflow,
  and the optimizer all read it: fine on one workspace, but if the
  optimizer's gateway is on another workspace, leave `DATABRICKS_TOKEN` unset
  and use per-side auth (a shared token fails as an empty optimizer transcript).
- **MLflow destination:** `MLFLOW_TRACKING_URI=databricks` — forge sets it itself
  when it finds Databricks credentials and nothing else picked a URI; any other
  value (`sqlite:`, `file:`) keeps traces in a local file, which a sandbox
  loses when it is wiped.
- **`local` optimizer backend:** `ANVIL_AI_GATEWAY_URL`
  (`https://<workspace-id>.ai-gateway.cloud.databricks.com/anthropic`) —
  required; without it every round fails at start, unless `ANTHROPIC_BASE_URL`
  is set — often inherited from the coding harness you run forge from, which
  silently points the optimizer at that harness's workspace.
- **`omnigent` backend:** `OMNIGENT_SERVER_URL`, `OMNIGENT_AUTH_TOKEN`.
- **Overrides that silently beat `harness/config.yaml`:**
  `ANVIL_OPTIMIZER_BACKEND`, `ANVIL_PERSIST_OPTIMIZER_ARTIFACTS`.
- **Optional:** `ANVIL_GATEWAY_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`,
  `ANVIL_GIT_COMMITTER_NAME` / `_EMAIL`, `ANVIL_GOOGLE_QUOTA_PROJECT`
  (price sync), plus any your domain reads (`grep -rn "environ\|getenv"
  src/anvil/domains/<name>/`).

Then the preflight: clean tree on the right remote, engine + levers load,
every allowed model accepts the agent's real input, lever dependencies are
installed, prices are known, and the baseline is **freshly generated at the
same eval size and starting levers as the rounds** (a mismatched or cached
baseline makes the gate compare unlike runs).

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

Generate the baseline under the **same conditions as the rounds**, or the gate
compares unlike runs and keeps or reverts on noise:

- **Same eval size:** `make_baseline.py --mode <m>` must match
  `run_round.py --eval-mode <m>`. A `quick` baseline gated against `full`
  rounds compares 8 rows to 20.
- **Fresh and live:** produce it by actually running the current agent, not
  from cached outputs. A cached baseline has drifted from a live re-run and
  once produced a false keep.
- **Same starting levers:** the baseline must run with the lever values the
  rounds start from (e.g. the base model).
- **Latency objectives:** its `cost_metrics` must include
  `latency_ms_median`.

---

## 9. The end-to-end walkthrough

Pattern A (works on merged `main` today):

1. **Create the package.** Copy the skeleton in §3 into
   `src/anvil/domains/<name>/`, renaming `echo` → `<name>`.
2. **Write the real `eval.py`.** Replace the stub body: compose the prompt from
   the scaffold, run your mutated agent over your inputs, score the outputs, and
   return an `EvalReport` whose `aggregate` genuinely varies with quality.
3. **Wire the config.** Set `eval.engine: <name>` (and `mode:`) in
   `harness/config.yaml`, plus the endpoints, the gate objectives, and
   optionally `levers` / `loop.max_mutations_per_round` / `model_catalog`
   (§13). Experiments need no entry — they default to
   `/Shared/forge/<name>/{eval,optimizer,runtime}` (§14).
4. **Provide the scaffold.** Ensure `scaffold/harness.yaml` exists with ≥1
   `kind: identity` skill, and `data/golden_set.jsonl` (or your engine's own
   data source) is in place.
5. **Run the session intake + preflight** (the "Session intake" checklist
   above, or the `forge-onboarding` skill): environment, models verified on real
   inputs, prices, where traces land.
6. **Prime the baseline** at the size you will run rounds with (§8):
   `uv run python scripts/make_baseline.py --mode <mode>`, then commit it and
   point `anvil/exp` at that commit.
7. **Run rounds.**
   `uv run python scripts/run_round.py --rounds N --eval-mode <mode> --parent-branch anvil/exp`.

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
| Config fails to load: `experiments.<kind> = … is outside domain '<name>'s experiment folder` | An experiment path (or `persistence.experiment`) is not under `<experiments.root>/<domain>/` | Remove it to use the default, move it under the domain folder, or change `experiments.root` (§14). |
| Every round raises at start: `ANVIL_AI_GATEWAY_URL is unset …` | `local` optimizer backend with no Anthropic route configured | Set `ANVIL_AI_GATEWAY_URL=https://<workspace-id>.ai-gateway.cloud.databricks.com/anthropic`. If `ANTHROPIC_BASE_URL` is inherited from your coding harness instead, check which workspace it points at. |
| Optimizer rounds end as `noop` with an empty transcript | The optimizer's gateway is on a different workspace from the agent and a shared `DATABRICKS_TOKEN` was sent to it | Unset `DATABRICKS_TOKEN` and give each side its own auth (profiles). |
| Traces missing from the workspace; round records show eval experiment id `1` | MLflow wrote to a local `sqlite:///…/mlflow.db` (no profile, no credentials found, or `MLFLOW_TRACKING_URI` set to a local store) | Provide Databricks credentials or set `MLFLOW_TRACKING_URI=databricks`; `run_round.py` prints the URI it uses (§14). |
| Round recorded as `noop` with `parse_status: apply_rejected` | The applier refused the action — lever value not in its allowlist, undeclared lever, compound over `loop.max_mutations_per_round`, two steps on one target, or an edit of a missing file | Read `notes` in the round JSON; widen the allowlist / cap if the action was reasonable. |
| Every row fails after the optimizer picks a new model | That model does not accept the agent's input (e.g. GPT / Gemini reject an Anthropic-style PDF `document` block) | Remove it from `levers.model.allowed`; verify each candidate on a real input before listing it. (A rejected `temperature`/`top_p` is retried automatically.) |
| Eval raises `input_mode=text needs pymupdf` | A domain lever value whose dependency is not installed | Install it (`uv pip install pymupdf`) or drop the value from the allowlist. |
| Sound improvements keep reverting | The gain is smaller than the objective's `epsilon` or the eval noise (LLM-judge scores and latency medians wobble on small eval sets) | Use a larger `--eval-mode`, check the epsilon against measured noise, or let near-misses combine via `compound` (§13). |
| `ingest_traces.py` keeps 0 rows, all `skipped_multiturn` | The agent is traced at the function level, or its input is a PDF/image | The `trace` engine cannot use these traces; write a domain (§2–§9). |

---

## 12. Seeing the optimization activity — per-round eval traces

forge arms MLflow tracing for **every** engine before it dispatches
(`evaluate_branch` → `anvil.observability.setup_eval_tracing`): it selects the
eval experiment (`experiments.eval`, default `/Shared/forge/<domain>/eval`) and enables
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

Which experiment and which workspace these traces land in — and why a sandbox
run used to lose them — is covered in §14.

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
mandatory `synergy` rationale. Steps apply all-or-nothing (`scaffold/` and
`agents/` are restored if any step fails), no two steps may touch the same
file / sampling field / lever, and the gate judges the combined result once.
Use it for changes that are each too small to clear an epsilon but should add
up (two ~1s latency cuts against a 2s epsilon), or a cheaper model plus a
prompt edit that protects quality on it. The `no_repeat_failed_mutations` rule
lets a mutation that was reverted for falling *short* (not for regressing) be
retried only as one step of such a compound; retried alone it is still a
repeat.

**Rejected actions** no longer crash a run. If the applier refuses an action —
a lever value outside its allowlist, an undeclared lever, a compound over the
cap, an edit of a missing file — the round is recorded as a `noop` with
`parse_status: apply_rejected` and the reason in `notes`, and the loop moves on.

**What the optimizer sees.** The round prompt gains a *Runtime levers*
section: each lever, its current value, its allowed values, the price of each
`model` value, and how each value has done in past rounds (rounds run, kept,
best aggregate, median latency). That history comes from the round records:
every `eval/runs/round_NNN.json` now has a `levers` field with the values the
eval ran with. The prompt also states the mutation budget (one mutation, or a
compound of up to N) and the real `--max-turns`.

**Model prices**: point `model_catalog.sheet_id` at a price sheet (columns
`Model`, `Price/Input Token ($/1M)`, `Price/Output Token ($/1M)`, optional
`Context` / `Cache read ($/1M)` / `Provider` / `Notes`) and run
`uv run python scripts/sync_model_catalog.py` to write
`harness/model_catalog.csv`; commit it. Rounds read only the CSV. Prices are
shown to the optimizer next to each `model` value and can be turned into
`cost_metrics.cost_usd_*` by your engine via `anvil.catalog`; latency is never
in a price list, so the prompt shows each value's measured history from past
rounds instead.

How names are matched: the sheet lists model *families* (`Claude Sonnet 4.5 /
4.6`), which are expanded into endpoint-style names (`claude-sonnet-4-5`,
`claude-sonnet-4-6`) and matched against the FMAPI endpoint with its
`databricks-` prefix dropped. Rows priced per context length use
`model_catalog.context_tier` (`short` by default). A model the sheet lacks
falls back to LiteLLM's bundled table — the source MLflow itself uses for
trace cost, which is list-priced and frozen at the installed version — and is
otherwise shown as "price unknown". Check coverage with
`scripts/sync_model_catalog.py --check <model> ...` (reads the committed CSV,
no sync). Syncing needs `gcloud auth application-default login`; the quota
project defaults to `gcp-dev-field-eng-aiapiquota`
(`ANVIL_GOOGLE_QUOTA_PROJECT`).

## 14. MLflow experiments — one set per domain, in the workspace

Everything forge records about an optimization lives in MLflow in the
Databricks workspace, in experiments that belong to **one domain only**:

| Experiment | Path (default) | What it holds |
|---|---|---|
| `experiments.eval` | `/Shared/forge/<domain>/eval` | One trace per eval row (the agent call + judge calls as sub-spans), tagged `source=eval`, `scaffold_branch=anvil/round-N`, `scaffold_commit_sha` |
| `experiments.optimizer` | `/Shared/forge/<domain>/optimizer` | One parent run per optimize session: `rounds/round_NNN_transcript.md`, `rounds/round_NNN_critique.md`, `improvement_summary.{json,md}` |
| `experiments.runtime` | `/Shared/forge/<domain>/runtime` | Traces from the deployed agent (not written by the loop) |

`<domain>` is `eval.engine`. Configure only the root:

```yaml
experiments:
  root: /Shared/forge          # or a per-user folder, e.g. /Users/<email>/forge
```

**The rule is enforced.** A kind left unset resolves to
`<root>/<domain>/<kind>`. A kind set explicitly must still be inside
`<root>/<domain>/` (e.g. `/Shared/forge/pitcrew/eval-v2`); any other path —
the old shared `/Shared/anvil-eval`, or another domain's folder — makes
`harness/config.yaml` fail to load with a message saying how to fix it. The
same applies to `persistence.experiment`. So two domains can never write into
the same experiment.

**Nothing to create by hand.** Each experiment is created on first use, and
forge creates its workspace folder first (Databricks will not create an
experiment whose parent folder is missing). The identity needs write access
to `root`; `/Shared/...` usually works, `/Users/<you>/...` always does.

**Where MLflow writes.** With a named `--profile`, MLflow uses
`databricks://<profile>`. Without one, forge sets `databricks` itself when it
finds Databricks credentials (`DATABRICKS_HOST`, `DATABRICKS_CONFIG_PROFILE`,
or `~/.databrickscfg`) and nothing else chose a tracking URI. This matters in
sandboxes: MLflow 3.x otherwise defaults to a local `sqlite:///<cwd>/mlflow.db`,
which is wiped with the sandbox — an earlier sandbox run lost every trace
that way (its round records show eval experiment id `1`, a local id). An
explicit `MLFLOW_TRACKING_URI` is always respected; if it points at a local
store, forge logs a warning. `scripts/run_round.py` prints the tracking URI it
uses at start.

**Optimizer transcripts from the scripts path.** `scripts/run_round.py` opens
the same durable sink as the orchestrator app: one parent run per invocation
(tagged `anvil.surface=cli`), each round's transcript + critique logged as it
finishes, and the improvement summary written when the session ends (status
`FAILED` if a round raised). Turn it off with `persistence.enabled: false` or
`ANVIL_PERSIST_OPTIMIZER_ARTIFACTS=0`. Before this, only the orchestrator
persisted transcripts; scripts kept them in the working tree only.

The `trace` engine's *input* experiment (§10, `eval.trace.experiment_id`) is
separate: it is the agent's own, pre-existing experiment that supplies the
dataset, and forge only reads it.

## See also

- `CLAUDE.md` — the invariants (plane separation, immutable `harness/config.yaml`).
- `.claude/skills/forge-onboarding/SKILL.md` — the session intake + preflight
  a coding harness runs before starting rounds.
- `src/anvil/domains/savesage/` — the worked example for Pattern A.
- `src/anvil/domains/pitcrew/` — a second worked domain, a FINRA-rule PDF
  summarizer. It shows: re-running a PDF agent through the gateway with an
  Anthropic-style base64 `document` block (no provider SDK); deterministic
  structure scorers plus a PDF-grounded judge gated on valid JSON; a latency
  objective (`cost_metrics.latency_ms_median` from timing the agent call only);
  a `model` lever and a domain lever (`input_mode: direct_pdf | text`, where
  `text` needs pymupdf and fails the eval loudly without it); `cost_usd_*`
  from token usage; and per-row tracing (§12). Its corpus is vendored in
  `data/pitcrew_corpus/` with repo-relative paths so it runs in any clone.
- `src/anvil/domains/trace/` — the trace-driven engine (§10). It cannot ingest
  traces recorded at the *function* level (the request is the function's
  kwargs, not a chat message) or with non-text input such as PDFs — every such
  trace is counted under `skipped_multiturn`. Write a domain for those agents.
