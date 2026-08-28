"""Deterministic train/dev/test partition enforcement tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from anvil.data import load_golden_set
from anvil.eval import runner
from anvil.runtime.loader import load_harness
from anvil.runtime.models import EvalConfig, EvalModeConfig, SplitConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _examples(count: int = 10_000) -> list[dict]:
    return [
        {"example_id": f"example-{i}", "category": "all", "query": f"query {i}"}
        for i in range(count)
    ]


def _ids(rows: list[dict]) -> set[str]:
    return {row["example_id"] for row in rows}


def _config(*, enabled: bool) -> EvalConfig:
    return EvalConfig(
        split=SplitConfig(enabled=enabled),
        modes={
            "quick": EvalModeConfig(rows=1, buckets={"all": 1}),
            "full": EvalModeConfig(rows=2, buckets={"all": 2}),
            "test": EvalModeConfig(rows=999, buckets={"all": 999}),
        },
    )


def test_partition_is_deterministic_and_order_independent() -> None:
    examples = _examples(200)
    split = SplitConfig(enabled=True, seed=73)
    first = runner.partition_dataset(examples, split)
    second = runner.partition_dataset(list(reversed(examples)), split)
    assert tuple(map(_ids, first)) == tuple(map(_ids, second))


def test_partition_ratios_are_close_to_configuration() -> None:
    partitions = runner.partition_dataset(_examples(), SplitConfig(enabled=True))
    ratios = [len(rows) / 10_000 for rows in partitions]
    assert ratios == pytest.approx([0.6, 0.2, 0.2], abs=0.015)


def test_partitions_have_no_overlap_or_missing_examples() -> None:
    examples = _examples(500)
    train, dev, test = runner.partition_dataset(examples, SplitConfig(enabled=True))
    runner._verify_no_overlap(train, dev, test)
    memberships = Counter(row["example_id"] for rows in (train, dev, test) for row in rows)
    assert set(memberships) == _ids(examples)
    assert set(memberships.values()) == {1}


def test_contamination_guard_rejects_overlap() -> None:
    duplicate = {"example_id": "leaked"}
    with pytest.raises(RuntimeError, match=r"partition overlap detected.*leaked"):
        runner._verify_no_overlap([duplicate], [], [duplicate])


def test_split_enabled_routes_dev_modes_to_dev_and_test_mode_to_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = [{"example_id": "train", "category": "all"}]
    dev = [
        {"example_id": "dev-1", "category": "all"},
        {"example_id": "dev-2", "category": "all"},
    ]
    test = [{"example_id": "test", "category": "all"}]
    monkeypatch.setattr(runner, "partition_dataset", lambda _rows, _split: (train, dev, test))
    cfg = _config(enabled=True)

    assert runner._select_mode_examples([], cfg=cfg, selected_mode="quick") == dev[:1]
    with pytest.warns(UserWarning, match="scaled 'full' bucket counts"):
        assert runner._select_mode_examples([], cfg=cfg, selected_mode="full") == dev[:1]
    assert runner._select_mode_examples([], cfg=cfg, selected_mode="test") == test


def test_split_modes_run_against_shipped_golden_set() -> None:
    examples = load_golden_set(REPO_ROOT / "data" / "golden_set.jsonl")
    cfg = load_harness(REPO_ROOT / "scaffold").config.eval.model_copy(
        update={"split": SplitConfig(enabled=True)}
    )

    with pytest.warns(UserWarning, match="scaled 'quick' bucket counts"):
        quick = runner._select_mode_examples(examples, cfg=cfg, selected_mode="quick")
    test = runner._select_mode_examples(examples, cfg=cfg, selected_mode="test")

    assert Counter(row["category"] for row in quick) == {
        "direct": 1,
        "multi_hop": 1,
        "distractor": 1,
        "out_of_scope": 1,
    }
    assert len(test) == 5


def test_split_disabled_preserves_full_dataset_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = [
        {"example_id": "first", "category": "all"},
        {"example_id": "second", "category": "all"},
    ]
    monkeypatch.setattr(
        runner,
        "partition_dataset",
        lambda *_args: pytest.fail("disabled split must not partition the golden set"),
    )
    selected = runner._select_mode_examples(
        examples, cfg=_config(enabled=False), selected_mode="quick"
    )
    assert selected == examples[:1]


def test_split_ratio_validation() -> None:
    with pytest.raises(ValueError, match="must be less than 1"):
        SplitConfig(train_ratio=0.8, dev_ratio=0.2)
