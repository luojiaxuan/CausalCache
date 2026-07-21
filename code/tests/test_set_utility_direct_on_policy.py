from __future__ import annotations

from causalcache.set_utility_direct_on_policy import (
    candidate_complete_coalitions,
    collection_bases,
    complete_group_count,
    missing_candidate_complete_coalitions,
)


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
