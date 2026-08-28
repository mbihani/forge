"""Append-only log of round mutations.

Defaults to a local JSONL at ``eval/mutations.jsonl`` for hassle-free
local runs. The Delta-table backend (``anvil.default.mutations``) is
the production target and lands in a follow-up commit; the schema is
the same so a `COPY INTO` from JSONL is the migration path.

Schema per row::

    {
      "mutation_id":      "mut_<12-hex>",
      "round_id":         <int>,
      "git_branch":       "anvil/exp-round-<N>",
      "git_commit_sha":   "<40-hex>",
      "parent_commit_sha":"<40-hex>",
      "files_added":      ["scaffold/..."],
      "files_changed":    ["scaffold/..."],
      "files_removed":    ["scaffold/..."],
      "diff_summary":     "<applier action_summary>",
      "proposed_by":      "claude-opus-4-7",
      "baseline_score":   0.7833,
      "mutated_score":    0.78 | null,
      "score_delta":      -0.003 | null,
      "decision":         "keep" | "revert" | "noop" | "infra_fail",
      "decided_at":       "<ISO8601 UTC>",
      "mlflow_eval_run_id": "<run id> | null",
      "parse_status":     "ok" | "no_block" | "schema_mismatch" | ...,
      "notes":            "<free-form>"
    }
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class MutationRecord:
    mutation_id: str
    round_id: int
    git_branch: str
    git_commit_sha: str
    parent_commit_sha: str
    files_added: list[str]
    files_changed: list[str]
    files_removed: list[str]
    diff_summary: str
    proposed_by: str
    baseline_score: float | None
    mutated_score: float | None
    score_delta: float | None
    decision: str
    decided_at: str
    mlflow_eval_run_id: str | None
    parse_status: str
    notes: str = ""

    @classmethod
    def new(
        cls,
        *,
        round_id: int,
        git_branch: str,
        git_commit_sha: str,
        parent_commit_sha: str,
        files_added: list[str] | None = None,
        files_changed: list[str] | None = None,
        files_removed: list[str] | None = None,
        diff_summary: str = "",
        proposed_by: str = "claude-opus-4-7",
        baseline_score: float | None = None,
        mutated_score: float | None = None,
        score_delta: float | None = None,
        decision: str = "noop",
        mlflow_eval_run_id: str | None = None,
        parse_status: str = "ok",
        notes: str = "",
    ) -> MutationRecord:
        return cls(
            mutation_id=f"mut_{uuid.uuid4().hex[:12]}",
            round_id=round_id,
            git_branch=git_branch,
            git_commit_sha=git_commit_sha,
            parent_commit_sha=parent_commit_sha,
            files_added=list(files_added or []),
            files_changed=list(files_changed or []),
            files_removed=list(files_removed or []),
            diff_summary=diff_summary,
            proposed_by=proposed_by,
            baseline_score=baseline_score,
            mutated_score=mutated_score,
            score_delta=score_delta,
            decision=decision,
            decided_at=datetime.now(UTC).isoformat(timespec="seconds"),
            mlflow_eval_run_id=mlflow_eval_run_id,
            parse_status=parse_status,
            notes=notes,
        )

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


def mutations_log_path(repo_root: Path | str) -> Path:
    return Path(repo_root) / "eval" / "mutations.jsonl"


def append_mutation(repo_root: Path | str, record: MutationRecord) -> Path:
    """Append a record to the JSONL file. Creates the file if missing."""
    path = mutations_log_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.to_json_line() + "\n")
    return path


def load_mutations(repo_root: Path | str) -> list[MutationRecord]:
    """Read all rows back. Used by the round_show CLI + tests."""
    path = mutations_log_path(repo_root)
    if not path.is_file():
        return []
    out: list[MutationRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        out.append(MutationRecord(**raw))
    return out
