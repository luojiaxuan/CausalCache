from __future__ import annotations

import itertools
import math

import pytest

from causalcache.set_utility_label_table import (
    EQUAL_WEIGHTING_SCHEME,
    EXACT_ORACLE_TIE_BREAK,
    MAXIMUM_CANDIDATE_EVENT_COUNT,
    SetUtilityDistanceInputRow,
    build_equal_weighting_metadata,
    exact_at_most_budget_oracle,
    expected_capped_row_count,
    validate_cardinality_capped_distance_table,
    validate_cardinality_capped_label_rows,
)


def _distances(
    event_ids: tuple[int, ...],
    cap: int,
    *,
    empty_distance: float = 100.0,
) -> dict[tuple[int, ...], float]:
    return {
        coalition: empty_distance - sum(event_ids.index(item) + 1 for item in coalition)
        for cardinality in range(min(len(event_ids), cap) + 1)
        for coalition in itertools.combinations(event_ids, cardinality)
    }


def _input_rows(
    *,
    split: str,
    state_id: str,
    event_ids: tuple[int, ...],
    cap: int,
) -> list[SetUtilityDistanceInputRow]:
    return [
        SetUtilityDistanceInputRow(
            split=split,
            state_id=state_id,
            candidate_event_step_ids=event_ids,
            maximum_labeled_cardinality=cap,
            coalition_event_step_ids=coalition,
            distance=distance,
        )
        for coalition, distance in _distances(event_ids, cap).items()
    ]


@pytest.mark.parametrize(
    ("candidate_count", "cap", "expected"),
    ((16, 2, 137), (6, 3, 42), (8, 4, 163), (1, 4, 2), (0, 2, 1)),
)
def test_arbitrary_n_capped_inventory_counts(
    candidate_count: int,
    cap: int,
    expected: int,
) -> None:
    event_ids = tuple(range(1, candidate_count + 1))
    table = validate_cardinality_capped_distance_table(
        split="train",
        state_id=f"state-{candidate_count}-{cap}",
        candidate_event_step_ids=event_ids,
        maximum_labeled_cardinality=cap,
        distances=_distances(event_ids, cap),
    )
    assert len(table.rows) == expected
    assert expected_capped_row_count(candidate_count, cap) == expected
    assert max((row.cardinality for row in table.rows), default=0) <= cap


def test_capped_table_does_not_require_or_accept_the_rest_of_power_set() -> None:
    event_ids = (1, 2, 3, 4)
    capped = _distances(event_ids, 2)
    table = validate_cardinality_capped_distance_table(
        split="train",
        state_id="state-capped",
        candidate_event_step_ids=event_ids,
        maximum_labeled_cardinality=2,
        distances=capped,
    )
    assert len(table.rows) == 11
    with pytest.raises(ValueError, match="labeled-cardinality cap"):
        validate_cardinality_capped_distance_table(
            split="train",
            state_id="state-extra",
            candidate_event_step_ids=event_ids,
            maximum_labeled_cardinality=2,
            distances={**capped, (1, 2, 3): 0.0},
        )


def test_every_capped_subset_is_required_exactly_once() -> None:
    event_ids = (1, 2, 3)
    valid = _distances(event_ids, 2)
    missing = dict(valid)
    missing.pop((1, 3))
    with pytest.raises(ValueError, match="exactly every cardinality-capped subset"):
        validate_cardinality_capped_distance_table(
            split="train",
            state_id="state-missing",
            candidate_event_step_ids=event_ids,
            maximum_labeled_cardinality=2,
            distances=missing,
        )
    with pytest.raises(ValueError, match="duplicate normalized coalition"):
        validate_cardinality_capped_distance_table(
            split="train",
            state_id="state-duplicate",
            candidate_event_step_ids=event_ids,
            maximum_labeled_cardinality=2,
            distances={**valid, frozenset({1, 2}): valid[(1, 2)]},
        )


def test_nonnegative_distances_produce_signed_utilities_without_clipping() -> None:
    table = validate_cardinality_capped_distance_table(
        split="evaluation",
        state_id="state-signed",
        candidate_event_step_ids=(4, 8),
        maximum_labeled_cardinality=2,
        distances={
            (): 2.0,
            (4,): 0.0,
            (8,): 5.5,
            (4, 8): 3.0,
        },
    )
    assert table.distance(()) == 2.0
    assert table.utility(()) == 0.0
    assert table.utility((4,)) == 2.0
    assert table.utility((8,)) == -3.5
    assert table.utility((4, 8)) == -1.0
    assert [row.utility for row in table.rows] == [0.0, 2.0, -3.5, -1.0]


def test_exact_at_most_budget_oracle_uses_frozen_tie_break_and_empty() -> None:
    table = validate_cardinality_capped_distance_table(
        split="tune",
        state_id="state-ties",
        candidate_event_step_ids=(1, 2, 3),
        maximum_labeled_cardinality=2,
        distances={
            (): 10.0,
            (1,): 5.0,
            (2,): 5.0,
            (3,): 8.0,
            (1, 2): 5.0,
            (1, 3): 6.0,
            (2, 3): 6.0,
        },
    )
    oracle = exact_at_most_budget_oracle(table, budget=2)
    assert oracle.tie_break == EXACT_ORACLE_TIE_BREAK
    assert oracle.coalition_event_step_ids == (1,)
    assert oracle.utility == 5.0
    assert oracle.evaluated_coalition_count == 7

    harmful = validate_cardinality_capped_distance_table(
        split="tune",
        state_id="state-empty-best",
        candidate_event_step_ids=(1, 2),
        maximum_labeled_cardinality=2,
        distances={(): 0.0, (1,): 1.0, (2,): 3.0, (1, 2): 7.0},
    )
    assert exact_at_most_budget_oracle(
        harmful,
        budget=2,
    ).coalition_event_step_ids == ()


def test_serialized_rows_cannot_cross_state_or_split() -> None:
    rows = _input_rows(
        split="train",
        state_id="shared-state",
        event_ids=(1, 2),
        cap=2,
    )
    rows.append(
        SetUtilityDistanceInputRow(
            split="evaluation",
            state_id="shared-state",
            candidate_event_step_ids=(3, 4),
            maximum_labeled_cardinality=2,
            coalition_event_step_ids=(),
            distance=0.0,
        )
    )
    with pytest.raises(ValueError, match="multiple splits"):
        validate_cardinality_capped_label_rows(rows)

    disagrees = _input_rows(
        split="train",
        state_id="one-state",
        event_ids=(1, 2),
        cap=2,
    )
    disagrees[-1] = SetUtilityDistanceInputRow(
        split="train",
        state_id="one-state",
        candidate_event_step_ids=(1, 3),
        maximum_labeled_cardinality=2,
        coalition_event_step_ids=(1, 3),
        distance=0.0,
    )
    with pytest.raises(ValueError, match="disagree on candidate events"):
        validate_cardinality_capped_label_rows(disagrees)


def test_serialized_rows_reject_subset_from_another_state_candidate_pool() -> None:
    rows = _input_rows(
        split="train",
        state_id="state-a",
        event_ids=(1, 2),
        cap=2,
    )
    rows[-1] = SetUtilityDistanceInputRow(
        split="train",
        state_id="state-a",
        candidate_event_step_ids=(1, 2),
        maximum_labeled_cardinality=2,
        coalition_event_step_ids=(1, 9),
        distance=0.0,
    )
    with pytest.raises(ValueError, match="outside its state"):
        validate_cardinality_capped_label_rows(rows)


def test_equal_state_and_cardinality_weighting_metadata_is_split_local() -> None:
    rows = [
        *_input_rows(
            split="train",
            state_id="train-a",
            event_ids=(1, 2),
            cap=2,
        ),
        *_input_rows(
            split="train",
            state_id="train-b",
            event_ids=(1, 2, 3),
            cap=2,
        ),
        *_input_rows(
            split="evaluation",
            state_id="eval-a",
            event_ids=(1, 2, 3, 4),
            cap=2,
        ),
    ]
    dataset = validate_cardinality_capped_label_rows(rows)
    metadata = dataset.weighting
    assert metadata.scheme == EQUAL_WEIGHTING_SCHEME
    assert metadata.normalization_scope == "independently_within_each_split"
    assert metadata.split_state_counts == (("evaluation", 1), ("train", 2))

    for split in ("train", "evaluation"):
        assert sum(
            row.normalized_row_weight for row in metadata.rows if row.split == split
        ) == pytest.approx(1.0)
    for state_id, expected_state_mass in (
        ("train-a", 0.5),
        ("train-b", 0.5),
        ("eval-a", 1.0),
    ):
        state_rows = [row for row in metadata.rows if row.state_id == state_id]
        assert sum(row.normalized_row_weight for row in state_rows) == pytest.approx(
            expected_state_mass
        )
        by_cardinality = {
            cardinality: sum(
                row.normalized_row_weight
                for row in state_rows
                if row.cardinality == cardinality
            )
            for cardinality in (0, 1, 2)
        }
        assert tuple(by_cardinality.values()) == pytest.approx(
            (expected_state_mass / 3.0,) * 3
        )


def test_limits_finiteness_oracle_cap_and_weighting_duplicates_fail_closed() -> None:
    with pytest.raises(ValueError, match="between 0 and 16"):
        expected_capped_row_count(MAXIMUM_CANDIDATE_EVENT_COUNT + 1, 2)
    with pytest.raises(ValueError, match="exactly 2, 3, or 4"):
        expected_capped_row_count(4, 1)
    invalid = _distances((1, 2), 2)
    invalid[(1,)] = math.nan
    with pytest.raises(ValueError, match="finite"):
        validate_cardinality_capped_distance_table(
            split="train",
            state_id="nonfinite",
            candidate_event_step_ids=(1, 2),
            maximum_labeled_cardinality=2,
            distances=invalid,
        )
    negative = _distances((1, 2), 2)
    negative[(1,)] = -1e-12
    with pytest.raises(ValueError, match="non-negative"):
        validate_cardinality_capped_distance_table(
            split="train",
            state_id="negative-distance",
            candidate_event_step_ids=(1, 2),
            maximum_labeled_cardinality=2,
            distances=negative,
        )
    table = validate_cardinality_capped_distance_table(
        split="train",
        state_id="budget",
        candidate_event_step_ids=(1, 2),
        maximum_labeled_cardinality=2,
        distances=_distances((1, 2), 2),
    )
    with pytest.raises(ValueError, match="exceeds"):
        exact_at_most_budget_oracle(table, budget=3)
    with pytest.raises(ValueError, match="duplicate state"):
        build_equal_weighting_metadata((table, table))
