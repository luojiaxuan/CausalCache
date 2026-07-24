from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train_hgkv_set_selector_v1 import (
    build_conditional_groups,
    build_group_arrays,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_conditional_groups_restore_edge_marginals_and_budgets(
    tmp_path: Path,
) -> None:
    pair_group = "episode:6"
    render = tmp_path / "samples-shard000.jsonl"
    _write(
        render,
        [
            {
                "conditional_anchor_set": [1],
                "conditional_candidate": None,
                "memory_config": {"restored_event_step_ids": [1]},
                "pair_group": pair_group,
                "restored_set_key": "1",
                "variant": "cond_base",
            },
            {
                "conditional_anchor_set": [1],
                "conditional_candidate": 2,
                "memory_config": {"restored_event_step_ids": [1, 2]},
                "pair_group": pair_group,
                "restored_set_key": "1-2",
                "variant": "cond_edge1",
            },
            {
                "conditional_anchor_set": [1, 2],
                "conditional_candidate": None,
                "memory_config": {"restored_event_step_ids": [1, 2]},
                "pair_group": pair_group,
                "restored_set_key": "1-2",
                "variant": "cond_base",
            },
            {
                "conditional_anchor_set": [1, 2],
                "conditional_candidate": 3,
                "memory_config": {"restored_event_step_ids": [1, 2, 3]},
                "pair_group": pair_group,
                "restored_set_key": "1-2-3",
                "variant": "cond_edge2",
            },
        ],
    )
    conditional_scores = {
        (pair_group, "cond_base", "1"): -0.4,
        (pair_group, "cond_edge1", "1-2"): -0.1,
        (pair_group, "cond_base", "1-2"): -0.1,
        (pair_group, "cond_edge2", "1-2-3"): -0.25,
    }

    groups, audit = build_conditional_groups(
        [render],
        conditional_scores=conditional_scores,
        singleton_scores={(pair_group, 1): -0.4},
        budget_replication={"cond_edge1": [2, 4], "cond_edge2": [4]},
    )

    assert groups[(pair_group, (1,), 2)][2] == pytest.approx(0.3)
    assert groups[(pair_group, (1,), 4)][2] == pytest.approx(0.3)
    assert groups[(pair_group, (1, 2), 4)][3] == pytest.approx(-0.15)
    assert audit["cond_base_max_abs_singleton_delta"] == 0.0
    assert audit["cond_base_pair_count"] == 1
    assert audit["cond_base_pair_max_abs_edge1_delta"] == 0.0


def test_conditional_groups_reject_pair_base_parity_drift(
    tmp_path: Path,
) -> None:
    pair_group = "episode:6"
    render = tmp_path / "samples-shard000.jsonl"
    _write(
        render,
        [
            {
                "conditional_anchor_set": [1, 2],
                "conditional_candidate": None,
                "memory_config": {"restored_event_step_ids": [1, 2]},
                "pair_group": pair_group,
                "restored_set_key": "1-2",
                "variant": "cond_base",
            }
        ],
    )
    with pytest.raises(ValueError, match="parity drifted"):
        build_conditional_groups(
            [render],
            conditional_scores={
                (pair_group, "cond_base", "1-2"): -0.2,
                (pair_group, "cond_edge1", "1-2"): -0.1,
            },
            singleton_scores={},
            budget_replication={
                "cond_edge1": [2, 4],
                "cond_edge2": [4],
            },
        )


def test_group_arrays_reference_singleton_feature_rows() -> None:
    pair_group = "episode:6"
    groups = {
        (pair_group, (1,), 2): {3: 0.1, 2: -0.2},
        (pair_group, (1, 2), 4): {3: 0.05},
    }
    feature_indices = {
        (pair_group, 1): 0,
        (pair_group, 2): 1,
        (pair_group, 3): 2,
    }

    arrays = build_group_arrays(
        groups,
        feature_indices=feature_indices,
        episode_by_state={pair_group: "episode"},
    )

    assert arrays["candidate_mask"].sum() == 3
    assert arrays["selected_mask"].sum() == 3
    assert arrays["remaining_budget"].tolist() == [1, 2]
