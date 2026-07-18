from __future__ import annotations

import itertools
import math

import pytest

from causalcache.gate_v1_data import CandidateFeatures, GateState
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from causalcache.set_conditioned_v4_data import (
    PAIR_FEATURE_DIMENSION,
    build_training_batch,
    feasible_coalitions,
)


def _state(
    source: str,
    event_count: int,
    *,
    baseline: float = 20.0,
    utility_by_coalition: dict[tuple[int, ...], float] | None = None,
) -> GateState:
    event_ids = tuple(range(1, event_count + 1))
    utilities = {
        coalition: (
            utility_by_coalition[coalition]
            if utility_by_coalition is not None
            else math.fsum(coalition)
            + math.fsum(
                left * right / 5.0
                for left, right in itertools.combinations(coalition, 2)
            )
        )
        for size in range(event_count + 1)
        for coalition in itertools.combinations(event_ids, size)
    }
    table = validate_complete_distance_table(
        event_ids,
        {coalition: baseline - utility for coalition, utility in utilities.items()},
    )
    return GateState(
        source_id=source,
        state_id=f"{source}:n{event_count}",
        decision_step_id=event_count + 2,
        candidate_event_step_ids=event_ids,
        q64=(0.125,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=tuple(1.0 if index == event - 1 else 0.0 for index in range(64)),
                g8=(event / 10.0,) * 8,
            )
            for event in event_ids
        ),
        table=table,
    )


def test_pair_feature_dimension_binds_frozen_88d_base() -> None:
    assert PAIR_FEATURE_DIMENSION == 4 * 88 + 64 == 416


def test_batch_contains_only_pair_regression_and_pair_involving_ranking() -> None:
    states = (_state("source-a", 2), _state("source-b", 3))
    batch = build_training_batch(states)
    assert len(batch.event_rows) == 5
    assert len(batch.pair_targets) == 1 + 3
    assert batch.eligible_pair_count == 4
    assert batch.regression_weight_sum == pytest.approx(1.0, abs=1e-12)
    assert batch.ranking_weight_sum == pytest.approx(1.0, abs=1e-12)
    assert all(
        len(row.left_coalition) == 2 or len(row.right_coalition) == 2
        for row in batch.ranking_pairs
    )
    assert all(row.left_coalition != row.right_coalition for row in batch.ranking_pairs)


def test_pair_target_is_normalized_full_set_utility_not_raw_interaction() -> None:
    utilities = {(): 0.0, (1,): 3.0, (2,): 4.0, (1, 2): 10.0}
    batch = build_training_batch(
        (_state("source-a", 2, baseline=20.0, utility_by_coalition=utilities),)
    )
    target = batch.pair_targets[0]
    assert target.raw_pair_utility == 10.0
    assert target.normalized_pair_utility == 0.5
    assert target.regression_weight == 1.0


def test_near_zero_baseline_is_ranking_only_and_never_divided() -> None:
    eligible = _state("source-a", 2)
    zero = _state(
        "source-b",
        2,
        baseline=0.0,
        utility_by_coalition={(): 0.0, (1,): 0.0, (2,): 0.0, (1, 2): 0.0},
    )
    batch = build_training_batch((eligible, zero))
    zero_pair = next(row for row in batch.pair_targets if row.source_id == "source-b")
    assert zero_pair.normalized_pair_utility is None
    assert zero_pair.regression_weight == 0.0
    assert batch.regression_weight_sum == 1.0


def test_geometry_and_duplicate_state_ids_fail_closed() -> None:
    state = _state("source-a", 2)
    assert feasible_coalitions(state) == ((), (1,), (2,), (1, 2))
    with pytest.raises(ValueError, match="duplicate state"):
        build_training_batch((state, state))
