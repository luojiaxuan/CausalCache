from __future__ import annotations

import itertools

import pytest

from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.long_horizon_evaluation import (
    ComparatorPairUnionDistanceTable,
    StateSelectionMetric,
    build_comparator_pair_union_distance_table,
    build_sparse_reference_distance_table,
    evaluate_pair_union_table,
    evaluate_sparse_tables_after_seal,
    exact_at_most_b_oracle,
    horizon_incidence,
    paired_trajectory_bootstrap,
    paired_trajectory_deltas,
    trajectory_equal_selector_mean,
    type7_quantile,
)
from causalcache.long_horizon_selectors import (
    independent_selection,
    recent_selection,
    seal_label_blind_selections,
)


def _state(source: str, state_id: str, n: int) -> FeatureState:
    event_ids = tuple(range(1, n + 1))
    return FeatureState(
        source_id=source,
        state_id=state_id,
        decision_step_id=n + 2,
        candidate_event_step_ids=event_ids,
        q64=(0.0,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=(0.0,) * 64,
                g8=(0.0,) * 8,
            )
            for event in event_ids
        ),
    )


def _n8_distances() -> dict[tuple[int, ...], float]:
    events = tuple(range(1, 9))
    distances = {
        coalition: 100.0 - sum(coalition)
        for size in range(5)
        for coalition in itertools.combinations(events, size)
    }
    return distances


def test_n8_sparse_table_and_exact_b2_b4_oracles() -> None:
    events = tuple(range(1, 9))
    table = build_sparse_reference_distance_table(
        events,
        _n8_distances(),
        maximum_enumerated_budget=4,
    )
    b2 = exact_at_most_b_oracle(table, budget_event_capacity=2)
    b4 = exact_at_most_b_oracle(table, budget_event_capacity=4)
    assert b2.coalition == (7, 8)
    assert b2.evaluated_coalition_count == 1 + 8 + 28
    assert b4.coalition == (5, 6, 7, 8)
    assert b4.evaluated_coalition_count == 1 + 8 + 28 + 56 + 70
    assert len(table.rows) == 163
    assert table.distance(events) == 0.0
    assert b4.exact_within_budget
    assert not b4.full_power_set_enumerated
    assert b4.raw_utility == 26.0
    assert b4.normalized_recovery == 0.26

    missing = _n8_distances()
    missing.pop((1, 2, 3, 4))
    with pytest.raises(ValueError, match="missing"):
        build_sparse_reference_distance_table(
            events,
            missing,
            maximum_enumerated_budget=4,
        )
    with pytest.raises(ValueError, match="one of"):
        exact_at_most_b_oracle(table, budget_event_capacity=3)


def test_sparse_evaluation_requires_a_prebuilt_selection_seal() -> None:
    state = _state("trajectory", "n8", 8)
    decisions = (
        recent_selection(state, budget_event_capacity=4),
        independent_selection(
            state,
            lambda _state, event, _coalition: float(event),
            budget_event_capacity=4,
        ),
    )
    seal = seal_label_blind_selections((state,), decisions)
    table = build_sparse_reference_distance_table(
        state.candidate_event_step_ids,
        _n8_distances(),
        maximum_enumerated_budget=4,
    )
    records = evaluate_sparse_tables_after_seal(seal, {state.state_id: table})
    assert len(records) == 2
    assert all(record.raw_utility == 26.0 for record in records)
    assert all(record.normalized_recovery == 0.26 for record in records)

    with pytest.raises(TypeError, match="SelectionSeal"):
        evaluate_sparse_tables_after_seal(object(), {state.state_id: table})


def test_normalization_uses_full_reference_and_can_be_ineligible() -> None:
    events = tuple(range(1, 9))
    distances = _n8_distances()
    distances[()] = 110.0
    table = build_sparse_reference_distance_table(
        events, distances, maximum_enumerated_budget=4
    )
    assert table.raw_utility((7, 8)) == 25.0
    assert table.normalized_recovery((7, 8)) == 25.0 / 110.0

    flat = {coalition: 0.0 for coalition in _n8_distances()}
    flat_table = build_sparse_reference_distance_table(
        events, flat, maximum_enumerated_budget=4
    )
    assert flat_table.normalized_recovery((7, 8)) == 0.0

    with pytest.raises(ValueError, match="self-reference"):
        build_sparse_reference_distance_table(
            events,
            _n8_distances(),
            maximum_enumerated_budget=4,
            full_reference_distance=1e-9,
        )


def _pair_union_distances(
    events: tuple[int, ...], selections: dict[str, tuple[int, ...]]
) -> dict[tuple[int, ...], float]:
    assert len(selections) == 2
    left, right = sorted(selections)
    reference = tuple(sorted(set(selections[left]) | set(selections[right])))
    required = {(), *selections.values(), reference}
    result = {coalition: 100.0 - sum(coalition) for coalition in required}
    result[reference] = 0.0
    return result


def test_n16_pair_union_table_is_explicitly_not_an_oracle() -> None:
    events = tuple(range(1, 17))
    selections = {
        "restoration_independent_gate": (1, 4, 8, 16),
        "recent": (13, 14, 15, 16),
    }
    table = build_comparator_pair_union_distance_table(
        source_id="trajectory",
        state_id="n16",
        event_ids=events,
        selections=selections,
        distances=_pair_union_distances(events, selections),
    )
    assert isinstance(table, ComparatorPairUnionDistanceTable)
    assert not table.supports_exact_subset_oracle
    assert table.pair_union_reference_coalition == (1, 4, 8, 13, 14, 15, 16)
    assert table.pair_union_distance("restoration_independent_gate", "recent") == 0.0
    assert events not in dict(table.rows)
    metrics = evaluate_pair_union_table(table)
    assert {record.selector_name for record in metrics} == set(selections)
    assert all(record.budget_event_capacity == 4 for record in metrics)
    independent = next(
        row for row in metrics if row.selector_name == "restoration_independent_gate"
    )
    assert independent.raw_utility == 29.0
    assert independent.normalized_recovery == 0.29

    with pytest.raises(TypeError, match="SparseReferenceDistanceTable"):
        exact_at_most_b_oracle(table, budget_event_capacity=4)
    distances = _pair_union_distances(events, selections)
    distances[events] = 0.0
    with pytest.raises(ValueError, match="exactly"):
        build_comparator_pair_union_distance_table(
            source_id="trajectory",
            state_id="n16",
            event_ids=events,
            selections=selections,
            distances=distances,
        )

    with pytest.raises(ValueError, match="exactly two"):
        build_comparator_pair_union_distance_table(
            source_id="trajectory",
            state_id="n16",
            event_ids=events,
            selections={**selections, "ocr_rgb_v2": (2, 4, 6, 8)},
            distances=distances,
        )


def _metric(
    selector: str,
    source: str,
    state: str,
    value: float | None,
) -> StateSelectionMetric:
    return StateSelectionMetric(
        selector_name=selector,
        source_id=source,
        state_id=state,
        horizon_n=8,
        budget_event_capacity=4,
        selected_event_step_ids=(1, 2, 3, 4),
        distance=0.0,
        raw_utility=0.0 if value is None else value,
        normalized_recovery=value,
    )


def test_trajectory_equal_aggregation_and_paired_bootstrap_are_deterministic() -> None:
    records = (
        _metric("left", "a", "a1", 10.0),
        _metric("right", "a", "a1", 8.0),
        _metric("left", "a", "a2", 12.0),
        _metric("right", "a", "a2", 8.0),
        _metric("left", "b", "b1", 1.0),
        _metric("right", "b", "b1", 2.0),
    )
    mean, excluded_states, excluded_trajectories = trajectory_equal_selector_mean(
        records,
        selector_name="left",
        metric="normalized_recovery",
    )
    assert mean == 6.0
    assert excluded_states == 0
    assert excluded_trajectories == 0
    deltas = paired_trajectory_deltas(
        records,
        left_selector="left",
        right_selector="right",
        metric="normalized_recovery",
    )
    assert deltas == {"a": 3.0, "b": -1.0}
    first = paired_trajectory_bootstrap(deltas, resamples=1000)
    second = paired_trajectory_bootstrap(dict(reversed(tuple(deltas.items()))), resamples=1000)
    assert first == second
    assert first.estimate == 1.0
    assert first.seed == 271828
    assert first.quantile_method == "type7_linear"
    assert type7_quantile((0.0, 10.0), 0.25) == 2.5


def test_paired_metrics_exclude_only_ineligible_state_pairs() -> None:
    records = (
        _metric("left", "a", "a1", None),
        _metric("right", "a", "a1", 1.0),
        _metric("left", "a", "a2", 3.0),
        _metric("right", "a", "a2", 1.0),
    )
    deltas = paired_trajectory_deltas(
        records,
        left_selector="left",
        right_selector="right",
        metric="normalized_recovery",
    )
    assert deltas == {"a": 2.0}


def test_trajectory_reducers_require_budget_filter_when_cells_are_combined() -> None:
    b4 = _metric("left", "a", "a1", 2.0)
    b2 = StateSelectionMetric(
        **{
            **b4.__dict__,
            "budget_event_capacity": 2,
            "normalized_recovery": 1.0,
        }
    )
    right_b4 = _metric("right", "a", "a1", 1.0)
    right_b2 = StateSelectionMetric(
        **{**right_b4.__dict__, "budget_event_capacity": 2, "normalized_recovery": 0.5}
    )
    records = (b4, b2, right_b4, right_b2)
    with pytest.raises(ValueError, match="explicit budget"):
        trajectory_equal_selector_mean(
            records, selector_name="left", metric="normalized_recovery"
        )
    mean, _, _ = trajectory_equal_selector_mean(
        records,
        selector_name="left",
        metric="normalized_recovery",
        budget_event_capacity=4,
    )
    assert mean == 2.0
    with pytest.raises(ValueError, match="duplicate selector/state"):
        paired_trajectory_deltas(
            records,
            left_selector="left",
            right_selector="right",
            metric="normalized_recovery",
        )
    assert paired_trajectory_deltas(
        records,
        left_selector="left",
        right_selector="right",
        metric="normalized_recovery",
        budget_event_capacity=2,
    ) == {"a": 0.5}


def test_horizon_incidence_reports_decision_and_trajectory_exposure() -> None:
    states = (
        _state("a", "a4", 4),
        _state("a", "a8", 8),
        _state("a", "a16", 16),
        _state("b", "b16", 16),
    )
    report = horizon_incidence(states)
    assert report.total_decision_count == 4
    assert report.total_trajectory_count == 2
    assert report.minimum_n == 4
    assert report.maximum_n == 16
    assert report.exact_n_decision_counts == ((4, 1), (8, 1), (16, 2))
    bins = {row.name: row for row in report.bins}
    assert bins["n_le_4"].decision_count == 1
    assert bins["n_5_to_8"].trajectory_count == 1
    assert bins["n_9_to_16"].decision_fraction == 0.5
    assert bins["n_9_to_16"].trajectory_fraction == 1.0
