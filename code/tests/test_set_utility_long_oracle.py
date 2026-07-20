"""Tests for the long-history oracle wave diagnostic."""

from __future__ import annotations

import pytest

from causalcache.set_utility_long_oracle import (
    BUDGETS,
    WAVES,
    additive_top_k,
    greedy_selection_from_distances,
    next_wave_coalitions,
    recent_prefix,
    reduce_long_oracle_metrics,
    select_long_oracle_states,
    wave_one_coalitions,
)


def _assignments() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(40):
        rows.append(
            {
                "trajectory_id": f"train-long-{index:03d}",
                "role": "train",
                "decision_count": 24,
            }
        )
    for index in range(12):
        rows.append(
            {
                "trajectory_id": f"train-verylong-{index:03d}",
                "role": "train",
                "decision_count": 40,
            }
        )
    rows.append(
        {"trajectory_id": "tune-000", "role": "tune", "decision_count": 40}
    )
    rows.append(
        {
            "trajectory_id": "evaluation-000",
            "role": "evaluation",
            "decision_count": 40,
        }
    )
    return rows


def test_selection_is_deterministic_with_quotas_and_trajectory_cap() -> None:
    kwargs = {
        "bin_targets": {"long": 30, "very_long": 8},
        "maximum_states_per_trajectory": 2,
        "salt": "unit-salt",
    }
    first = select_long_oracle_states(_assignments(), **kwargs)
    second = select_long_oracle_states(_assignments(), **kwargs)
    assert first == second
    assert len(first) == 38
    bins = {row["history_bin"] for row in first}
    assert bins == {"long", "very_long"}
    assert all(row["role"] == "train" for row in first)
    per_trajectory: dict[str, int] = {}
    for row in first:
        per_trajectory[row["trajectory_id"]] = (
            per_trajectory.get(row["trajectory_id"], 0) + 1
        )
        count = len(row["candidate_event_ids"])
        assert row["candidate_event_ids"] == list(range(1, count + 1))
    assert max(per_trajectory.values()) <= 2
    assert [row["state_id"] for row in first] == sorted(
        row["state_id"] for row in first
    )


def test_selection_rejects_unreachable_targets() -> None:
    with pytest.raises(ValueError):
        select_long_oracle_states(
            _assignments(),
            bin_targets={"long": 10, "very_long": 200},
            maximum_states_per_trajectory=2,
            salt="unit-salt",
        )


def test_wave_one_covers_singletons_recent_random_and_anchors() -> None:
    candidates = tuple(range(1, 21))
    rows = wave_one_coalitions(
        state_id="trajectory:decision:021",
        candidate_event_ids=candidates,
        random_subsets_per_budget=1,
        random_seed=7,
    )
    subsets = {tuple(row["event_ids"]) for row in rows}
    assert () in subsets
    assert candidates in subsets
    for event in candidates:
        assert (event,) in subsets
    for budget in BUDGETS:
        assert recent_prefix(candidates, budget) in subsets
    sources = [row["source"] for row in rows]
    assert len(subsets) == len(rows)
    for budget in BUDGETS:
        matching = [
            tuple(row["event_ids"])
            for row in rows
            if row["source"] == f"random_b{budget}"
        ]
        for subset in matching:
            assert len(subset) == budget
    repeat = wave_one_coalitions(
        state_id="trajectory:decision:021",
        candidate_event_ids=candidates,
        random_subsets_per_budget=1,
        random_seed=7,
    )
    assert rows == repeat
    assert sources[0] == "anchor_empty"


def _additive_distance(subset: tuple[int, ...], weights: dict[int, float]) -> float:
    return round(1.0 - sum(weights.get(event, 0.0) for event in subset), 12)


def test_greedy_selection_and_additive_top_k() -> None:
    candidates = tuple(range(1, 18))
    weights = {2: 0.3, 5: 0.2, 17: 0.05, 16: 0.04}
    rows = [
        {
            "coalition_event_step_ids": list(subset),
            "distance": _additive_distance(subset, weights),
        }
        for subset in [(), *((event,) for event in candidates)]
    ]
    first = greedy_selection_from_distances(
        candidate_event_ids=candidates,
        distance_rows=rows,
        previous_selected=(),
    )
    assert first == (2,)
    assert additive_top_k(
        candidate_event_ids=candidates, distance_rows=rows, k=2
    ) == (2, 5)
    with pytest.raises(ValueError):
        greedy_selection_from_distances(
            candidate_event_ids=candidates,
            distance_rows=rows[:5],
            previous_selected=(),
        )


def test_greedy_tie_break_is_lexicographic() -> None:
    candidates = tuple(range(1, 18))
    rows = [
        {"coalition_event_step_ids": [event], "distance": 0.5}
        for event in candidates
    ]
    selected = greedy_selection_from_distances(
        candidate_event_ids=candidates,
        distance_rows=rows,
        previous_selected=(),
    )
    assert selected == (1,)


def test_next_wave_expands_prefix_and_binds_additive_size() -> None:
    candidates = tuple(range(1, 18))
    rows = next_wave_coalitions(
        candidate_event_ids=candidates,
        previous_selected=(2, 5),
        additive_subset=(2, 5, 17),
    )
    subsets = {tuple(row["event_ids"]) for row in rows}
    assert () in subsets and candidates in subsets
    for event in candidates:
        if event not in (2, 5):
            assert tuple(sorted((2, 5, event))) in subsets
    with pytest.raises(ValueError):
        next_wave_coalitions(
            candidate_event_ids=candidates,
            previous_selected=(2, 5),
            additive_subset=(2, 5),
        )
    with pytest.raises(ValueError):
        next_wave_coalitions(
            candidate_event_ids=candidates,
            previous_selected=(1, 2, 3, 4),
            additive_subset=None,
        )


def _synthetic_wave_terminals(
    states: tuple[dict[str, object], ...],
    weights_by_state: dict[str, dict[int, float]],
    *,
    random_seed: int,
) -> dict[int, dict[str, dict[str, object]]]:
    terminals: dict[int, dict[str, dict[str, object]]] = {
        wave: {} for wave in WAVES
    }
    for state in states:
        state_id = str(state["state_id"])
        candidates = tuple(state["candidate_event_ids"])
        weights = weights_by_state[state_id]

        def rows_for(coalitions: tuple[dict[str, object], ...]) -> list[dict[str, object]]:
            return [
                {
                    "coalition_event_step_ids": list(row["event_ids"]),
                    "distance": _additive_distance(
                        tuple(row["event_ids"]), weights
                    ),
                }
                for row in coalitions
            ]

        wave_rows = rows_for(
            wave_one_coalitions(
                state_id=state_id,
                candidate_event_ids=candidates,
                random_subsets_per_budget=1,
                random_seed=random_seed,
            )
        )
        prefix: tuple[int, ...] = ()
        for wave in WAVES:
            if wave > 1:
                additive = (
                    additive_top_k(
                        candidate_event_ids=candidates,
                        distance_rows=terminals[1][state_id]["distance_rows"],
                        k=wave,
                    )
                    if wave >= 3
                    else None
                )
                wave_rows = rows_for(
                    next_wave_coalitions(
                        candidate_event_ids=candidates,
                        previous_selected=prefix,
                        additive_subset=additive,
                    )
                )
            terminals[wave][state_id] = {
                "distance_rows": wave_rows,
                "reference": {"serialized_action": "action"},
                "state_id": state_id,
                "status": "COMPLETED_VARIABLE_HISTORY_LABEL_STATE",
            }
            prefix = greedy_selection_from_distances(
                candidate_event_ids=candidates,
                distance_rows=wave_rows,
                previous_selected=prefix,
            )
    return terminals


def test_reducer_recovers_additive_ground_truth() -> None:
    states = select_long_oracle_states(
        _assignments(),
        bin_targets={"long": 4, "very_long": 2},
        maximum_states_per_trajectory=1,
        salt="unit-salt",
    )
    weights_by_state = {}
    for state in states:
        candidates = tuple(state["candidate_event_ids"])
        weights_by_state[str(state["state_id"])] = {
            2: 0.3,
            5: 0.2,
            candidates[-1]: 0.05,
            candidates[-2]: 0.04,
        }
    terminals = _synthetic_wave_terminals(
        states, weights_by_state, random_seed=11
    )
    summary = reduce_long_oracle_metrics(
        states=states,
        wave_terminals=terminals,
        bootstrap_resamples=200,
        bootstrap_seed=3,
        random_subsets_per_budget=1,
        random_seed=11,
    )
    assert summary["completed_state_count"] == len(states)
    oracle = summary["methods"]["oracle_greedy"]
    recent = summary["methods"]["recent"]
    assert oracle["B1"] == pytest.approx(0.3)
    assert oracle["B2"] == pytest.approx(0.5)
    assert recent["B1"] == pytest.approx(0.05)
    assert recent["B2"] == pytest.approx(0.09)
    assert summary["methods"]["additive"]["B4"] == pytest.approx(oracle["B4"])
    delta = summary["comparisons"]["oracle_greedy_minus_recent_macro"]
    assert delta["point_estimate"] > 0.2
    assert delta["lower_95"] <= delta["point_estimate"] <= delta["upper_95"]
    assert summary["best_singleton_outside_recent4_fraction"] == pytest.approx(1.0)


def test_reducer_skips_states_missing_late_waves() -> None:
    states = select_long_oracle_states(
        _assignments(),
        bin_targets={"long": 4, "very_long": 2},
        maximum_states_per_trajectory=1,
        salt="unit-salt",
    )
    weights_by_state = {}
    for state in states:
        candidates = tuple(state["candidate_event_ids"])
        weights_by_state[str(state["state_id"])] = {
            2: 0.3,
            5: 0.2,
            candidates[-1]: 0.05,
            candidates[-2]: 0.04,
        }
    terminals = _synthetic_wave_terminals(
        states, weights_by_state, random_seed=11
    )
    dropped = str(states[0]["state_id"])
    del terminals[4][dropped]
    summary = reduce_long_oracle_metrics(
        states=states,
        wave_terminals=terminals,
        bootstrap_resamples=50,
        bootstrap_seed=3,
        random_subsets_per_budget=1,
        random_seed=11,
    )
    assert summary["completed_state_count"] == len(states) - 1
    assert summary["skipped_states"] == {dropped: "missing_wave_4"}
