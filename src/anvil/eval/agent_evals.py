"""Start from the agent's EXISTING evals: review them, then re-use their judges.

An agent brought to forge has usually been evaluated already — LLM judges
and code scorers registered on its MLflow experiment, judge / code / human
assessments on its traces, MLflow issue-detection findings. Forge must not
invent its own idea of "quality" when the agent's owner already has one:

1. :func:`discover_agent_evals` reads the agent's experiment once and builds
   an :class:`AgentEvalInventory` — every judge it can re-run (registered
   scorers, plus Guidelines judges recoverable from assessment metadata) and
   a per-assessment summary (pass rate / mean, failing rationales, human
   feedback, issues).
2. :func:`render_review` turns that into ``eval/agent_evals/review.md`` —
   what the judges and human feedback point towards. The operator reads it
   in the intake and the optimizer reads it every round.
3. :func:`load_agent_judges` rebuilds THOSE judges so every round is scored
   by the same standard the agent was evaluated with. The genai and trace
   engines use them directly; a domain engine calls
   :func:`score_with_agent_judges`.
4. :func:`ensure_agent_evals_ready` is the gate the baseline / round
   entrypoints run: the review must exist, and if the agent has no judges
   the metrics must have been agreed with the user
   (``eval.agent_evals.user_metrics``).

Heavy imports (mlflow, databricks-sdk) are lazy so the module imports with
no server, and every network touchpoint is injectable for offline tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AGENT_EVALS_DIR = Path("eval") / "agent_evals"
INVENTORY_FILENAME = "inventory.json"
REVIEW_FILENAME = "review.md"

# Assessment source types (MLflow ``AssessmentSourceType``).
_JUDGE_SOURCES = frozenset({"LLM_JUDGE", "AI_JUDGE"})
_HUMAN_SOURCE = "HUMAN"
_CODE_SOURCE = "CODE"
# MLflow issue detection logs its findings as CODE assessments from this source.
_ISSUE_SOURCE_ID = "mlflow.issue_detection"
_MAX_RATIONALE_CHARS = 400


class AgentEvalsNotReady(RuntimeError):  # noqa: N818 - reads as a sentence at raise sites
    """The agent's existing evals have not been reviewed (or no metrics agreed)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class JudgeSpec:
    """One judge forge can re-run.

    ``source`` is ``registered_scorer`` (the exact scorer registered on the
    agent's experiment; ``payload`` is its serialized form) or
    ``assessment_guideline`` (a Guidelines judge rebuilt from the
    ``guideline`` metadata its assessments carry, used only when no scorer of
    that name is registered). ``kind`` is ``builtin`` / ``guidelines`` /
    ``custom_code`` / ``instructions_judge`` / ``unknown``.
    """

    name: str
    source: str
    kind: str
    description: str = ""
    guidelines: list[str] = field(default_factory=list)
    payload: dict[str, Any] | None = None


@dataclass
class AssessmentSummary:
    """How one named assessment scored across the agent's traces."""

    name: str
    source_type: str
    n: int
    mean: float | None
    failing_rationales: list[str] = field(default_factory=list)


@dataclass
class AgentEvalInventory:
    """Everything forge learned from the agent's existing evals."""

    experiment_id: str
    fetched_at: str
    n_traces: int
    n_traces_with_assessments: int
    judges: list[JudgeSpec] = field(default_factory=list)
    assessments: list[AssessmentSummary] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    io_shape: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AgentEvalInventory:
        data = dict(raw)
        data["judges"] = [JudgeSpec(**j) for j in data.get("judges", [])]
        data["assessments"] = [AssessmentSummary(**a) for a in data.get("assessments", [])]
        return cls(**data)

    def judge_names(self) -> list[str]:
        return [j.name for j in self.judges]

    def fingerprint(self) -> str:
        """Hash of the judge definitions — changes when a judge changes."""
        blob = json.dumps(
            [dataclasses.asdict(j) for j in sorted(self.judges, key=lambda j: j.name)],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Registered scorers
# ---------------------------------------------------------------------------


def _fetch_databricks_scheduled_scorers(experiment_id: str) -> list[dict[str, Any]]:
    """Serialized scorers registered on a Databricks experiment (REST)."""
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    response = WorkspaceClient().api_client.do(
        "GET", f"/api/2.0/managed-evals/scheduled-scorers/{experiment_id}"
    )
    payloads: list[dict[str, Any]] = []
    for entry in (response.get("scheduled_scorers") or {}).get("scorers") or []:
        serialized = entry.get("serialized_scorer")
        if isinstance(serialized, str):
            serialized = json.loads(serialized)
        if isinstance(serialized, dict):
            payloads.append(serialized)
    return payloads


def _fetch_mlflow_registered_scorers(experiment_id: str) -> list[dict[str, Any]]:
    """Serialized scorers from the OSS MLflow scorer registry."""
    from mlflow.genai.scorers import list_scorers  # noqa: PLC0415

    return [s.model_dump() for s in list_scorers(experiment_id=experiment_id)]


def fetch_registered_scorers(
    experiment_id: str,
    *,
    fetchers: Iterable[Callable[[str], list[dict[str, Any]]]] | None = None,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Serialized scorer payloads registered on ``experiment_id``.

    Tries the Databricks scheduled-scorers API first (it returns the raw
    serialized form even when the local MLflow is too old to deserialize
    it), then the OSS MLflow registry. Each failure is recorded in
    ``warnings``; an experiment with no registered scorers returns ``[]``.
    """
    chain = (
        list(fetchers)
        if fetchers is not None
        else [_fetch_databricks_scheduled_scorers, _fetch_mlflow_registered_scorers]
    )
    for fetch in chain:
        try:
            return fetch(experiment_id)
        except Exception as exc:  # noqa: BLE001 - fall through to the next source
            msg = f"{getattr(fetch, '__name__', 'fetch')}: {type(exc).__name__}: {exc}"
            logger.info("registered-scorer lookup failed: %s", msg)
            if warnings is not None:
                warnings.append(msg[:300])
    return []


def _classify_payload(payload: dict[str, Any]) -> str:
    builtin = payload.get("builtin_scorer_class")
    if builtin == "Guidelines":
        return "guidelines"
    if builtin:
        return "builtin"
    if payload.get("call_source"):
        return "custom_code"
    if payload.get("instructions_judge_pydantic_data"):
        return "instructions_judge"
    return "unknown"


def _spec_from_payload(payload: dict[str, Any]) -> JudgeSpec:
    builtin_data = payload.get("builtin_scorer_pydantic_data") or {}
    guidelines = builtin_data.get("guidelines") or []
    if isinstance(guidelines, str):
        guidelines = [guidelines]
    return JudgeSpec(
        name=str(payload.get("name") or ""),
        source="registered_scorer",
        kind=_classify_payload(payload),
        description=str(payload.get("description") or ""),
        guidelines=[str(g) for g in guidelines],
        payload=payload,
    )


def deserialize_scorer(payload: dict[str, Any]) -> Any:
    """Rebuild an MLflow ``Scorer`` from its serialized payload.

    Forward-compatible: a payload written by a NEWER MLflow can carry fields
    this version's ``SerializedScorer`` does not know (e.g. ``timeout``,
    ``third_party_scorer_data``); those are dropped before validation, the
    same way an older reader ignores unknown JSON. A custom-code scorer only
    loads under a Databricks tracking URI (an MLflow security rule).
    """
    from mlflow.genai.scorers.base import Scorer, SerializedScorer  # noqa: PLC0415

    known = {f.name for f in dataclasses.fields(SerializedScorer)}
    return Scorer.model_validate({k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Trace assessments
# ---------------------------------------------------------------------------


def _assessment_value(assessment: Any) -> Any:
    feedback = getattr(assessment, "feedback", None)
    if feedback is not None and hasattr(feedback, "value"):
        return feedback.value
    expectation = getattr(assessment, "expectation", None)
    if expectation is not None and hasattr(expectation, "value"):
        return expectation.value
    return getattr(assessment, "value", None)


def to_score(raw: Any) -> float | None:
    """Map a judge value (bool / yes-no / pass-fail / number) to a float."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("yes", "pass", "true", "ok"):
            return 1.0
        if s in ("no", "fail", "false"):
            return 0.0
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _source(assessment: Any) -> tuple[str, str]:
    src = getattr(assessment, "source", None)
    return (
        str(getattr(src, "source_type", "") or ""),
        str(getattr(src, "source_id", "") or ""),
    )


def _shape(payload: Any) -> Any:
    """Coarse shape of a trace request/response: dict keys or the type name."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return "str"
    if isinstance(payload, dict):
        return sorted(str(k) for k in payload)
    if isinstance(payload, list):
        return f"list[{type(payload[0]).__name__ if payload else ''}]"
    return type(payload).__name__


def summarize_assessments(
    traces: Iterable[Any], *, max_rationales: int = 5
) -> tuple[
    list[AssessmentSummary], list[dict[str, Any]], dict[str, list[str]], dict[str, Any], int
]:
    """Summarize the assessments on ``traces``.

    Returns ``(summaries, issues, guidelines_by_judge, io_shape,
    n_with_assessments)``. Issue-detection findings are returned separately
    from scored assessments; ``guidelines_by_judge`` is the ``guideline``
    metadata LLM-judge assessments carry, so a Guidelines judge can be
    rebuilt when it is not registered as a scorer.
    """
    acc: dict[tuple[str, str], dict[str, Any]] = {}
    issues: dict[str, dict[str, Any]] = {}
    guidelines: dict[str, list[str]] = {}
    io_shape: dict[str, Any] = {}
    n_with = 0
    for trace in traces:
        info = getattr(trace, "info", None)
        assessments = list(getattr(info, "assessments", None) or [])
        if assessments:
            n_with += 1
        if not io_shape:
            data = getattr(trace, "data", None)
            io_shape = {
                "request": _shape(getattr(data, "request", None)),
                "response": _shape(getattr(data, "response", None)),
            }
        for a in assessments:
            name = str(getattr(a, "name", "") or "")
            source_type, source_id = _source(a)
            metadata = getattr(a, "metadata", None) or {}
            rationale = str(getattr(a, "rationale", None) or "")[:_MAX_RATIONALE_CHARS]
            if source_id == _ISSUE_SOURCE_ID:
                issue_name = str(metadata.get("mlflow.issue.name") or name)
                entry = issues.setdefault(
                    issue_name,
                    {
                        "name": issue_name,
                        "severity": metadata.get("mlflow.issue.severity", ""),
                        "rationale": rationale,
                        "n": 0,
                    },
                )
                entry["n"] += 1
                continue
            if source_type in _JUDGE_SOURCES and metadata.get("guideline"):
                lines = [ln.strip() for ln in str(metadata["guideline"]).splitlines()]
                guidelines.setdefault(name, [ln for ln in lines if ln])
            slot = acc.setdefault((name, source_type), {"scores": [], "n": 0, "failing": []})
            slot["n"] += 1
            score = to_score(_assessment_value(a))
            if score is not None:
                slot["scores"].append(score)
            failing = score is not None and score < 1.0
            if (
                (failing or source_type == _HUMAN_SOURCE)
                and rationale
                and rationale not in slot["failing"]
                and len(slot["failing"]) < max_rationales
            ):
                slot["failing"].append(rationale)
    summaries = [
        AssessmentSummary(
            name=name,
            source_type=source_type,
            n=slot["n"],
            mean=(sum(slot["scores"]) / len(slot["scores"])) if slot["scores"] else None,
            failing_rationales=slot["failing"],
        )
        for (name, source_type), slot in sorted(acc.items())
    ]
    return summaries, sorted(issues.values(), key=lambda i: -i["n"]), guidelines, io_shape, n_with


def _search_traces(experiment_id: str, max_traces: int) -> list[Any]:
    from mlflow import MlflowClient  # noqa: PLC0415

    client = MlflowClient()
    out: list[Any] = []
    token: str | None = None
    while len(out) < max_traces:
        page = client.search_traces(
            experiment_ids=[experiment_id],
            max_results=min(100, max_traces - len(out)),
            page_token=token,
        )
        out.extend(list(page))
        token = getattr(page, "token", None)
        if not token or not page:
            break
    return out[:max_traces]


# ---------------------------------------------------------------------------
# Discovery + review
# ---------------------------------------------------------------------------


def discover_agent_evals(
    experiment_id: str,
    *,
    max_traces: int = 200,
    traces: list[Any] | None = None,
    scorer_payloads: list[dict[str, Any]] | None = None,
) -> AgentEvalInventory:
    """Build the inventory of the agent's existing evals.

    ``traces`` / ``scorer_payloads`` are injectable (offline tests); by
    default the experiment's traces and registered scorers are read live.
    Registered scorers win; a Guidelines judge recovered from assessment
    metadata is added only for a name with no registered scorer.
    """
    warnings: list[str] = []
    if scorer_payloads is None:
        scorer_payloads = fetch_registered_scorers(experiment_id, warnings=warnings)
    if traces is None:
        try:
            traces = _search_traces(experiment_id, max_traces)
        except Exception as exc:  # noqa: BLE001 - review still works from scorers
            warnings.append(f"search_traces: {type(exc).__name__}: {exc}"[:300])
            traces = []

    judges = [_spec_from_payload(p) for p in scorer_payloads if p.get("name")]
    summaries, issues, guidelines, io_shape, n_with = summarize_assessments(traces)
    registered = {j.name for j in judges}
    for name, lines in sorted(guidelines.items()):
        if name not in registered:
            judges.append(
                JudgeSpec(
                    name=name,
                    source="assessment_guideline",
                    kind="guidelines",
                    guidelines=lines,
                )
            )
    return AgentEvalInventory(
        experiment_id=str(experiment_id),
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_traces=len(traces),
        n_traces_with_assessments=n_with,
        judges=judges,
        assessments=summaries,
        issues=issues,
        io_shape=io_shape,
        warnings=warnings,
    )


def _fmt_mean(mean: float | None) -> str:
    return "—" if mean is None else f"{mean:.2f}"


def render_review(inv: AgentEvalInventory, overrides: dict[str, Any] | None = None) -> str:
    """Markdown review: which judges exist and what they + humans point to.

    ``overrides`` (``eval.agent_evals.guideline_overrides``) are listed so
    the reader — and the optimizer — know which guideline text actually
    scores the rounds.
    """
    lines = [
        f"# Existing evals for this agent (experiment `{inv.experiment_id}`)",
        "",
        f"Read {inv.n_traces} traces ({inv.n_traces_with_assessments} with assessments) "
        f"on {inv.fetched_at}.",
        "",
        "## Judges forge re-uses to score every round",
        "",
    ]
    if inv.judges:
        lines += ["| Judge | Kind | Source | Definition |", "| --- | --- | --- | --- |"]
        for j in inv.judges:
            definition = "; ".join(j.guidelines) if j.guidelines else j.description
            lines.append(
                f"| `{j.name}` | {j.kind} | {j.source} | {definition[:300].replace('|', '/')} |"
            )
    else:
        lines.append(
            "**No judges found.** Ask the user which metrics the evals should use and "
            "record them under `eval.agent_evals.user_metrics` before optimizing."
        )

    if overrides:
        lines += ["", "### Forge-side guideline overrides (score the rounds instead)", ""]
        for name, ov in sorted(overrides.items()):
            lines.append(f"- `{name}` — {_override_field(ov, 'reason')}")
            lines += [f"  - {g}" for g in _override_field(ov, "guidelines")]

    worst = sorted(
        (a for a in inv.assessments if a.mean is not None and a.source_type != _HUMAN_SOURCE),
        key=lambda a: a.mean,
    )
    lines += ["", "## What the judges point towards (lowest score first)", ""]
    if worst:
        lines += ["| Assessment | Source | n | Mean (1 = pass) |", "| --- | --- | --- | --- |"]
        lines += [f"| `{a.name}` | {a.source_type} | {a.n} | {_fmt_mean(a.mean)} |" for a in worst]
        for a in worst:
            if a.failing_rationales and (a.mean or 0.0) < 1.0:
                lines += ["", f"**`{a.name}` failures:**"]
                lines += [f"- {r}" for r in a.failing_rationales]
    else:
        lines.append("No judge or code assessments on the traces.")

    flat = [a for a in worst if a.n >= 2 and a.mean in (0.0, 1.0)]
    if flat:
        lines += ["", "## Judges with no signal — confirm with the user", ""]
        for a in flat:
            verdict = "fails" if a.mean == 0.0 else "passes"
            lines.append(
                f"- `{a.name}` {verdict} on every trace (n={a.n}). "
                + (
                    "Check its definition against the agent's real output — it may be stale "
                    "or miscalibrated rather than the agent being wrong. "
                    if a.mean == 0.0
                    else ""
                )
                + "It gives the optimizer no gradient; decide with the user whether to keep, "
                "fix, or drop it (eval.agent_evals.judges)."
            )

    human = [a for a in inv.assessments if a.source_type == _HUMAN_SOURCE]
    lines += ["", "## Human feedback", ""]
    if human:
        for a in human:
            lines.append(f"- `{a.name}`: n={a.n}, mean={_fmt_mean(a.mean)}")
            lines += [f"  - {r}" for r in a.failing_rationales]
    else:
        lines.append("No human feedback on the traces.")

    if inv.issues:
        lines += ["", "## Issues MLflow detected", ""]
        lines += [
            f"- **{i['name']}** (severity {i.get('severity') or '?'}, n={i['n']}): {i['rationale']}"
            for i in inv.issues
        ]

    if inv.io_shape:
        lines += [
            "",
            "## Trace input/output shape the judges were run on",
            "",
            f"- request: `{inv.io_shape.get('request')}`",
            f"- response: `{inv.io_shape.get('response')}`",
            "",
            "Pass the judges inputs/outputs in this shape so their verdicts stay comparable.",
        ]
    if inv.warnings:
        lines += ["", "## Lookup warnings", ""] + [f"- {w}" for w in inv.warnings]
    return "\n".join(lines) + "\n"


def agent_evals_dir(repo_root: Path | str) -> Path:
    return Path(repo_root) / AGENT_EVALS_DIR


def write_agent_evals(
    repo_root: Path | str,
    inv: AgentEvalInventory,
    overrides: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write ``inventory.json`` + ``review.md`` under ``eval/agent_evals/``."""
    out = agent_evals_dir(repo_root)
    out.mkdir(parents=True, exist_ok=True)
    inv_path = out / INVENTORY_FILENAME
    inv_path.write_text(json.dumps(inv.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    review_path = out / REVIEW_FILENAME
    review_path.write_text(render_review(inv, overrides), encoding="utf-8")
    return inv_path, review_path


def read_agent_evals(repo_root: Path | str) -> AgentEvalInventory | None:
    path = agent_evals_dir(repo_root) / INVENTORY_FILENAME
    if not path.is_file():
        return None
    return AgentEvalInventory.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Re-running the agent's judges
# ---------------------------------------------------------------------------


class AgentJudge:
    """One of the agent's own judges, rebuilt and ready to score a round."""

    def __init__(
        self, spec: JudgeSpec, scorer: Any | None = None, model: str | None = None
    ) -> None:
        self.spec = spec
        self.name = spec.name
        self._scorer = scorer
        # Judge model for a Guidelines judge rebuilt from text (None = the
        # MLflow default judge, as for a registered scorer with no model set).
        self._model = model

    @property
    def scorer(self) -> Any:
        """The MLflow ``Scorer`` (built lazily so loading needs no server)."""
        if self._scorer is None:
            if self.spec.payload is not None:
                self._scorer = deserialize_scorer(self.spec.payload)
            elif self.spec.kind == "guidelines" and self.spec.guidelines:
                from mlflow.genai.scorers import Guidelines  # noqa: PLC0415

                kwargs = {"model": self._model} if self._model else {}
                self._scorer = Guidelines(
                    name=self.spec.name, guidelines=self.spec.guidelines, **kwargs
                )
            else:
                raise ValueError(f"judge {self.spec.name!r} has no definition to rebuild")
        return self._scorer

    def score(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: Any = None,
        trace: Any = None,
    ) -> tuple[float | None, str | None]:
        """Run the judge; return ``(score in [0,1] or None, rationale)``.

        ``Scorer.run`` passes only the arguments this judge accepts (e.g.
        ``Safety`` takes outputs only). A list of feedbacks averages.
        """
        result = self.scorer.run(
            inputs=inputs, outputs=outputs, expectations=expectations, trace=trace
        )
        items = result if isinstance(result, list) else [result]
        scores: list[float] = []
        rationale: str | None = None
        for item in items:
            if getattr(item, "error", None):
                continue
            value = to_score(getattr(item, "value", item))
            if value is not None:
                scores.append(value)
            rationale = rationale or getattr(item, "rationale", None)
        return (sum(scores) / len(scores) if scores else None), rationale


def load_agent_judges(
    repo_root: Path | str,
    *,
    names: list[str] | None = None,
    inventory: AgentEvalInventory | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[AgentJudge]:
    """The agent's judges from the reviewed inventory (optionally a subset).

    ``overrides`` maps a Guidelines judge's name to a corrected
    ``guidelines`` list (``eval.agent_evals.guideline_overrides``); that judge
    is rebuilt with the corrected text and the same name and judge model.
    """
    inv = inventory if inventory is not None else read_agent_evals(repo_root)
    if inv is None:
        return []
    wanted = set(names or [])
    unknown = (wanted | set(overrides or {})) - set(inv.judge_names())
    if unknown:
        raise ValueError(
            f"eval.agent_evals judge names {sorted(unknown)} not found in the reviewed "
            f"inventory (available: {inv.judge_names()})"
        )
    judges: list[AgentJudge] = []
    for spec in inv.judges:
        if wanted and spec.name not in wanted:
            continue
        override = (overrides or {}).get(spec.name)
        if override is not None:
            if spec.kind != "guidelines":
                raise ValueError(
                    f"guideline_overrides[{spec.name!r}]: only a Guidelines judge can be "
                    f"overridden (this one is {spec.kind!r})"
                )
            model = ((spec.payload or {}).get("builtin_scorer_pydantic_data") or {}).get("model")
            spec = dataclasses.replace(
                spec,
                source=f"{spec.source}+forge_override",
                guidelines=list(_override_field(override, "guidelines")),
                payload=None,
                description=f"forge override: {_override_field(override, 'reason')}",
            )
            judges.append(AgentJudge(spec, scorer=None, model=model))
            continue
        judges.append(AgentJudge(spec))
    return judges


def _override_field(override: Any, name: str) -> Any:
    return override[name] if isinstance(override, dict) else getattr(override, name)


def load_configured_agent_judges(eval_cfg: Any, repo_root: Path | str) -> list[AgentJudge]:
    """The judges ``eval.agent_evals`` selects (``[]`` without an experiment)."""
    cfg = getattr(eval_cfg, "agent_evals", None)
    if cfg is None or not cfg.experiment_id:
        return []
    return load_agent_judges(
        repo_root, names=list(cfg.judges), overrides=dict(cfg.guideline_overrides)
    )


def fingerprint_judges(judges: Iterable[AgentJudge]) -> str:
    """Hash of the effective judge definitions (overrides included)."""
    blob = json.dumps(
        sorted((dataclasses.asdict(j.spec) for j in judges), key=lambda d: d["name"]),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def score_with_agent_judges(
    judges: Iterable[AgentJudge],
    *,
    inputs: Any = None,
    outputs: Any = None,
    expectations: Any = None,
    trace: Any = None,
) -> dict[str, float]:
    """Score one row with every agent judge: ``{judge_name: score}``.

    A judge that errors or returns no usable value scores ``0.0`` (a failed
    verdict must not count as a pass) and is logged.
    """
    out: dict[str, float] = {}
    for judge in judges:
        try:
            score, _ = judge.score(
                inputs=inputs, outputs=outputs, expectations=expectations, trace=trace
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-judge failures
            logger.warning("agent judge %r failed: %s", judge.name, exc)
            score = None
        out[judge.name] = 0.0 if score is None else float(score)
    return out


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

_REVIEW_HOW = (
    "run `uv run python scripts/review_agent_evals.py --profile <profile>` and walk the "
    "user through eval/agent_evals/review.md"
)


def ensure_agent_evals_ready(eval_cfg: Any, repo_root: Path | str) -> AgentEvalInventory | None:
    """Refuse to baseline / optimize before the agent's evals were reviewed.

    * ``eval.agent_evals`` must be declared.
    * With an ``experiment_id``: the reviewed inventory for THAT experiment
      must exist; if it found no judges, ``user_metrics`` (agreed with the
      user) are required; any ``judges`` subset must exist in it.
    * Without one (the agent truly has no evals): ``user_metrics`` are
      required.

    Returns the inventory (``None`` when the run relies on user metrics).
    """
    cfg = getattr(eval_cfg, "agent_evals", None)
    if cfg is None:
        raise AgentEvalsNotReady(
            "harness/config.yaml has no eval.agent_evals. Forge starts from the agent's "
            "EXISTING evals: set eval.agent_evals.experiment_id to the MLflow experiment "
            f"holding the agent's traces/judges, then {_REVIEW_HOW}. If the agent has no "
            "evals, ask the user which metrics matter and record them under "
            "eval.agent_evals.user_metrics."
        )
    if not cfg.experiment_id:
        if not cfg.user_metrics:
            raise AgentEvalsNotReady(
                "eval.agent_evals has neither experiment_id nor user_metrics. Point it at "
                "the agent's eval experiment, or — only if the agent has no evals — ask the "
                "user which metrics to use and list them under user_metrics."
            )
        return None
    inv = read_agent_evals(repo_root)
    if inv is None or inv.experiment_id != str(cfg.experiment_id):
        raise AgentEvalsNotReady(
            f"the agent's evals (experiment {cfg.experiment_id}) have not been reviewed: "
            f"{_REVIEW_HOW}."
        )
    if not inv.judges and not cfg.user_metrics:
        raise AgentEvalsNotReady(
            f"experiment {cfg.experiment_id} has no judges forge can re-run. Ask the user "
            "which metrics the evals should use and record them under "
            "eval.agent_evals.user_metrics."
        )
    load_agent_judges(  # validates the subset + overrides
        repo_root,
        names=list(cfg.judges),
        inventory=inv,
        overrides=dict(cfg.guideline_overrides),
    )
    return inv
