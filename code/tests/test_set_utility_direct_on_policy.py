from __future__ import annotations

from causalcache.set_utility_direct_on_policy import (
    candidate_complete_coalitions,
    collection_bases,
    complete_group_count,
    missing_candidate_complete_coalitions,
    runner_schedule_coalitions,
)
from scripts.run_set_utility_variable_history_labels import _microbatches


def _record() -> dict:
    return {
        "candidate_event_ids": [1, 2, 3, 4, 5],
        "hybrid": {"1": [1], "2": [1, 5], "3": [1, 4, 5], "4": [1, 3, 4, 5]},
        "learned": {"1": [1], "2": [1, 2], "3": [1, 2, 3], "4": [1, 2, 3, 4]},
        "recent": {"1": [5], "2": [4, 5], "3": [3, 4, 5], "4": [2, 3, 4, 5]},
    }


def test_collection_bases_cover_three_nested_policies() -> None:
    bases = collection_bases(_record(), maximum_base_cardinality=3)
    assert () in bases
    assert (1, 2) in bases
    assert (4, 5) in bases
    assert (1, 5) in bases
    assert all(len(base) <= 3 for base in bases)


def test_candidate_complete_collection_expands_every_base() -> None:
    record = _record()
    desired = candidate_complete_coalitions(record, maximum_base_cardinality=3)
    for base in collection_bases(record, maximum_base_cardinality=3):
        assert base in desired
        for event_id in record["candidate_event_ids"]:
            if event_id not in base:
                assert tuple(sorted((*base, event_id))) in desired


def test_missing_collection_reuses_existing_truth() -> None:
    record = _record()
    desired = candidate_complete_coalitions(record, maximum_base_cardinality=3)
    existing = [(), (1,), (2,), (3,), (4,), (5,)]
    missing = missing_candidate_complete_coalitions(
        record, existing, maximum_base_cardinality=3
    )
    assert not {tuple(row["event_ids"]) for row in missing}.intersection(existing)
    resulting = [*existing, *(row["event_ids"] for row in missing)]
    assert complete_group_count(
        record["candidate_event_ids"], resulting, maximum_base_cardinality=3
    ) >= len(collection_bases(record, maximum_base_cardinality=3))
    assert len(desired) == len(set(map(tuple, resulting)))


def test_complete_group_count_excludes_terminal_full_set() -> None:
    assert complete_group_count(
        [1, 2], [(1, 2)], maximum_base_cardinality=2
    ) == 0


def test_runner_schedule_restores_zero_cost_full_anchor() -> None:
    record = _record()
    candidates = record["candidate_event_ids"]
    missing = missing_candidate_complete_coalitions(
        record,
        [candidates],
        maximum_base_cardinality=3,
    )
    assert tuple(candidates) not in {
        tuple(row["event_ids"]) for row in missing
    }
    scheduled = runner_schedule_coalitions(candidates, missing)
    assert tuple(candidates) in {
        tuple(row["event_ids"]) for row in scheduled
    }
    assert len(scheduled) == len(missing) + 1


def test_runner_schedule_does_not_duplicate_existing_full_anchor() -> None:
    candidates = _record()["candidate_event_ids"]
    scheduled = runner_schedule_coalitions(
        candidates,
        ({"event_ids": candidates, "source": "full_anchor"},),
    )
    assert scheduled == ({"event_ids": candidates, "source": "full_anchor"},)


def test_runner_schedule_exactly_matches_measured_and_anchor_distances() -> None:
    record = _record()
    candidates = record["candidate_event_ids"]
    missing = missing_candidate_complete_coalitions(
        record,
        [candidates],
        maximum_base_cardinality=3,
    )
    scheduled = runner_schedule_coalitions(candidates, missing)
    schedule = {
        "candidate_event_ids": candidates,
        "coalitions": list(scheduled),
    }
    distances = {tuple(candidates)}
    for batch in _microbatches(schedule, microbatch_size=16):
        distances.update(batch)
    assert distances == {tuple(row["event_ids"]) for row in scheduled}
