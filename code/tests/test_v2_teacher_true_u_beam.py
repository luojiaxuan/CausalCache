"""True-utility teacher beam planning and reduction invariants."""

from __future__ import annotations

from scripts.plan_hgkv_teacher_beam_v2 import build_plan_rows
from scripts.reduce_hgkv_teacher_beam_v2 import reduce_depth


def _state():
    return {
        "episode": "episode",
        "pair_group": "episode:9",
        "decision_step": 9,
        "history_length": 8,
        "candidate_event_step_ids": [1, 2, 3, 4, 5],
        "recent_candidate_cap": None,
    }


def _cache_row(pair_group: str, key: str, utility: float):
    return {
        "pair_group": pair_group,
        "restored_set_key": key,
        "target_action_sha256": "a" * 64,
        "u_act": utility,
    }


def test_depth0_beam_uses_true_singleton_utility():
    pair_group = "episode:9"
    cache = {(pair_group, ""): _cache_row(pair_group, "", 0.0)}
    for candidate, utility in {1: 0.1, 2: 0.5, 3: -0.2, 4: 0.4, 5: 0.3}.items():
        cache[(pair_group, str(candidate))] = _cache_row(
            pair_group, str(candidate), utility
        )
    plan = build_plan_rows(
        states={pair_group: _state()},
        cache=cache,
        prefixes_by_state={},
        depth=0,
    )
    labels, beam = reduce_depth(
        plan_rows=plan,
        cache=cache,
        depth=0,
        beam_width=4,
    )
    assert len(plan) == 5
    assert len(labels) == 1
    assert labels[0]["marginal_targets"] == [0.1, 0.5, -0.2, 0.4, 0.3]
    assert labels[0]["stop_is_optimal"] is False
    assert beam[0]["prefixes"] == [[2], [4], [5], [1]]


def test_depth1_renders_unique_pairs_but_keeps_all_prefix_edges():
    pair_group = "episode:9"
    singleton_prefixes = [[1], [2], [3], [4]]
    plan = build_plan_rows(
        states={pair_group: _state()},
        cache={},
        prefixes_by_state={pair_group: singleton_prefixes},
        depth=1,
    )
    assert len(plan) == 16
    assert len({row["restored_set_key"] for row in plan}) == 10

    cache = {
        (pair_group, str(candidate)): _cache_row(
            pair_group, str(candidate), candidate / 100
        )
        for candidate in range(1, 5)
    }
    for row in plan:
        key = row["restored_set_key"]
        values = [int(value) for value in key.split("-")]
        cache[(pair_group, key)] = _cache_row(
            pair_group, key, sum(values) / 10
        )
    labels, beam = reduce_depth(
        plan_rows=plan,
        cache=cache,
        depth=1,
        beam_width=4,
    )
    assert len(labels) == 4
    assert all(len(row["marginal_targets"]) == 4 for row in labels)
    assert beam[0]["prefixes"] == [[4, 5], [3, 5], [2, 5], [3, 4]]
