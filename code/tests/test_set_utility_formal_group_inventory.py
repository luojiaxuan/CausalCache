from __future__ import annotations

import copy

import pytest

from causalcache.set_utility_structured_training import formal_group_inventory


def _example(state_id: str, trajectory_id: str, selected: tuple[int, ...]) -> dict:
    candidates = (1, 2, 3, 4, 5)
    action_mask = [True]
    normalized = [0.0]
    for event_id in candidates:
        keep = event_id not in selected
        action_mask.append(keep)
        normalized.append(-0.1 if keep else 0.0)
    return {
        "candidate_event_step_ids": candidates,
        "conditional_groups": (
            {
                "action_mask": tuple(action_mask),
                "balance": {"stop_all_negative": True},
                "normalized_marginals": tuple(normalized),
                "selected_event_step_ids": selected,
            },
        ),
        "state_id": state_id,
        "trajectory_id": trajectory_id,
    }


def test_formal_group_inventory_binds_exact_ids_and_joint_strata() -> None:
    examples = (
        _example("t0:decision:006", "t0", ()),
        _example("t0:decision:006", "t0", (1,)),
        _example("t1:decision:006", "t1", (2, 3)),
    )
    first = formal_group_inventory(examples)
    second = formal_group_inventory(tuple(reversed(examples)))
    assert first == second
    assert first["complete_group_count"] == 3
    assert first["optimizer_state_count"] == 2
    assert first["optimizer_trajectory_count"] == 2
    assert first["base_cardinality_counts"] == {"0": 1, "1": 1, "2": 1}
    assert first["stop_all_negative_counts"] == {"true": 3}
    assert first["joint_stratum_counts"] == {
        "0|short|true": 1,
        "1|short|true": 1,
        "2|short|true": 1,
    }
    assert len(first["content_sha256"]) == 64


def test_formal_group_inventory_rejects_duplicate_group_identity() -> None:
    example = _example("t0:decision:006", "t0", ())
    with pytest.raises(ValueError, match="duplicated"):
        formal_group_inventory((example, copy.deepcopy(example)))


def test_formal_group_inventory_rejects_stop_metadata_drift() -> None:
    example = _example("t0:decision:006", "t0", ())
    example["conditional_groups"][0]["balance"]["stop_all_negative"] = False
    with pytest.raises(ValueError, match="STOP balance"):
        formal_group_inventory((example,))
