"""Exact-search validation for the frozen n<=8 V2 development subset."""

from __future__ import annotations

import json

import pytest

from scripts.plan_hgkv_exact_search_v2 import build_exact_plan_rows
from scripts.reduce_hgkv_exact_search_v2 import (
    _best_at_most,
    load_exact_plan,
    reduce_exact_search,
)


def _state(*, split: str = "dev", candidates: int = 4):
    return {
        "episode": "episode",
        "pair_group": "episode:9",
        "decision_step": 9,
        "history_length": 8,
        "candidate_event_step_ids": list(range(1, candidates + 1)),
        "recent_candidate_cap": None,
        "split": split,
    }


def _cache_row(pair_group: str, key: str, utility: float):
    return {
        "pair_group": pair_group,
        "restored_set_key": key,
        "target_action_sha256": "a" * 64,
        "u_act": utility,
    }


def test_exact_plan_is_dev_only_and_enumerates_all_sets_through_b4():
    pair_group = "episode:9"
    states = {
        pair_group: _state(),
        "train:9": _state(split="train"),
        "large:9": _state(candidates=9),
    }
    rows = build_exact_plan_rows(states=states, cache={})
    assert len(rows) == 4 + 6 + 4 + 1
    assert {row["pair_group"] for row in rows} == {pair_group}
    assert {row["depth"] for row in rows} == {0, 1, 2, 3}
    assert len({row["restored_set_key"] for row in rows}) == len(rows)


def test_exact_reducer_reports_beam_recovery_regret_and_jaccard(tmp_path):
    pair_group = "episode:9"
    state = _state(candidates=2)
    rows = build_exact_plan_rows(states={pair_group: state}, cache={})
    path = tmp_path / "plan.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    states = load_exact_plan([path])
    utilities = {"": 0.0, "1": 0.1, "2": 0.2, "1-2": 0.15}
    cache = {
        (pair_group, key): _cache_row(pair_group, key, utility)
        for key, utility in utilities.items()
    }
    teacher = {
        (pair_group, "teacher_beam4", budget): {
            "selected_event_step_ids": selection
        }
        for budget, selection in ((1, [1]), (2, [2]), (4, [2]))
    }
    selections, diagnostic = reduce_exact_search(
        states=states,
        cache=cache,
        teacher=teacher,
    )
    assert [row["selected_event_step_ids"] for row in selections] == [
        [2],
        [2],
        [2],
    ]
    b1 = diagnostic["budgets"]["B1"]
    assert b1["teacher_beam4_oracle_recovery"] == 0.0
    assert b1["teacher_beam4_mean_regret"] == 0.1
    assert b1["teacher_beam4_mean_top_b_jaccard"] == 0.0
    assert diagnostic["budgets"]["B2"]["teacher_beam4_oracle_recovery"] == 1.0


def test_exact_recovery_is_order_invariant_for_the_same_teacher_set(tmp_path):
    pair_group = "episode:9"
    state = _state(candidates=2)
    rows = build_exact_plan_rows(states={pair_group: state}, cache={})
    path = tmp_path / "plan.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    states = load_exact_plan([path])
    utilities = {"": 0.0, "1": 0.1, "2": 0.2, "1-2": 0.4}
    cache = {
        (pair_group, key): _cache_row(pair_group, key, utility)
        for key, utility in utilities.items()
    }
    teacher = {
        (pair_group, "teacher_beam4", budget): {
            "selected_event_step_ids": selection
        }
        for budget, selection in ((1, [2]), (2, [2, 1]), (4, [2, 1]))
    }
    _selections, diagnostic = reduce_exact_search(
        states=states,
        cache=cache,
        teacher=teacher,
    )
    assert diagnostic["budgets"]["B2"]["teacher_beam4_oracle_recovery"] == 1.0
    assert diagnostic["budgets"]["B4"]["teacher_beam4_oracle_recovery"] == 1.0


def test_exact_at_most_budget_can_stop_at_empty_and_requires_b0(tmp_path):
    pair_group = "episode:9"
    state = _state(candidates=2)
    rows = build_exact_plan_rows(states={pair_group: state}, cache={})
    path = tmp_path / "plan.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    states = load_exact_plan([path])
    utilities = {"": 0.0, "1": -0.1, "2": -0.2, "1-2": -0.3}
    cache = {
        (pair_group, key): _cache_row(pair_group, key, utility)
        for key, utility in utilities.items()
    }
    teacher = {
        (pair_group, "teacher_beam4", budget): {
            "selected_event_step_ids": []
        }
        for budget in (1, 2, 4)
    }
    selections, diagnostic = reduce_exact_search(
        states=states,
        cache=cache,
        teacher=teacher,
    )
    assert all(row["selected_event_step_ids"] == [] for row in selections)
    assert all(
        row["teacher_beam4_oracle_recovery"] == 1.0
        for row in diagnostic["budgets"].values()
    )
    del cache[(pair_group, "")]
    with pytest.raises(KeyError, match="misses 1 exact utilities"):
        _best_at_most(
            pair_group=pair_group,
            candidates=(1, 2),
            budget=1,
            cache=cache,
        )
