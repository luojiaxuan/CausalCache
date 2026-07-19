from __future__ import annotations

import itertools

from causalcache.set_utility_features import SetUtilityFeatureState
from causalcache.set_utility_label_producer import (
    CandidateLengthAttempt,
    FrozenCandidateContext,
)
from causalcache.set_utility_mvp import (
    COMPLETED_STATE_STATUS,
    feature_state_from_payload,
    feature_state_to_payload,
    select_worker_candidate_records,
    state_from_completed_record,
)
from causalcache.set_utility_processor_freeze import FrozenQueryCandidateRecord


def _candidate(role: str, index: int, *, count: int = 4) -> dict:
    event_ids = tuple(range(1, count + 1))
    candidate = FrozenQueryCandidateRecord(
        state_id=f"{role}-{index}:decision:010",
        trajectory_id=f"{role}-{index}",
        source_id=f"{role}-{index}",
        role=role,
        query_kind="stratum_anchor",
        decision_step_id=10,
        maximum_labeled_cardinality=2,
        candidate_context=FrozenCandidateContext(
            initial_candidate_event_step_ids=event_ids,
            candidate_event_step_ids=event_ids,
            dropped_event_step_ids=(),
            processor_input_token_count=100,
            reserved_action_tokens=256,
            context_limit=32_768,
            attempts=(CandidateLengthAttempt(event_ids, 100),),
        ),
    )
    return candidate.to_payload()


def test_role_balanced_selection_is_deterministic_and_filters_n() -> None:
    records = [
        _candidate(role, index)
        for role in ("train", "tune", "evaluation")
        for index in range(3)
    ]
    records.append(_candidate("train", 99, count=5))

    first = select_worker_candidate_records(
        records,
        candidate_count=4,
        query_kind="stratum_anchor",
        quota_by_role={"train": 2, "tune": 1, "evaluation": 1},
        selection_salt="fixture",
    )
    second = select_worker_candidate_records(
        tuple(reversed(records)),
        candidate_count=4,
        query_kind="stratum_anchor",
        quota_by_role={"train": 2, "tune": 1, "evaluation": 1},
        selection_salt="fixture",
    )

    assert first == second
    assert len(first) == 4
    assert {len(item.candidate_context.candidate_event_step_ids) for item in first} == {4}
    assert [item.role for item in first].count("train") == 2


def test_completed_state_round_trip_builds_exact_utility_table() -> None:
    event_ids = (1, 2, 3, 4)
    feature = SetUtilityFeatureState(
        source_id="trajectory",
        state_id="trajectory:decision:010",
        decision_step_id=10,
        event_step_ids=event_ids,
        query_features=(0.1, 0.2),
        context_features=(0.3,),
        event_features=tuple((step / 10.0, 1.0) for step in event_ids),
        pair_features=tuple(
            tuple((float(left == right),) for right in event_ids)
            for left in event_ids
        ),
        event_mask=(True, True, True, True),
    )
    payload = feature_state_to_payload(feature)
    assert feature_state_from_payload(payload) == feature
    coalitions = tuple(
        coalition
        for cardinality in range(3)
        for coalition in itertools.combinations(event_ids, cardinality)
    )
    record = {
        "candidate_event_step_ids": list(event_ids),
        "distance_rows": [
            {
                "coalition_event_step_ids": list(coalition),
                "distance": 1.0 - 0.1 * len(coalition),
            }
            for coalition in coalitions
        ],
        "feature": payload,
        "maximum_labeled_cardinality": 2,
        "role": "train",
        "state_id": feature.state_id,
        "status": COMPLETED_STATE_STATUS,
    }

    state = state_from_completed_record(record)

    assert state.utility(()) == 0.0
    assert abs(state.utility((1, 2)) - 0.2) < 1e-12
    assert state.normalization_scale == 1.0
