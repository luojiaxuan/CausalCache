"""Train-only direct/recent/hybrid candidate-complete data collection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


POLICY_KEYS = ("learned", "recent", "hybrid")


def _canonical_subset(
    value: Any,
    *,
    candidates: tuple[int, ...],
    label: str,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be an event-id sequence")
    subset = tuple(value)
    if (
        any(type(item) is not int for item in subset)
        or subset != tuple(sorted(subset))
        or len(subset) != len(set(subset))
        or not set(subset).issubset(candidates)
    ):
        raise ValueError(f"{label} escaped the candidate universe")
    return subset


def validate_direct_train_selection_record(record: Mapping[str, Any]) -> None:
    candidates = tuple(record.get("candidate_event_ids", ()))
    if (
        not candidates
        or candidates != tuple(sorted(candidates))
        or len(candidates) != len(set(candidates))
        or any(type(item) is not int for item in candidates)
    ):
        raise ValueError("direct train selection candidate universe is invalid")
    for policy in POLICY_KEYS:
        path = record.get(policy)
        if not isinstance(path, Mapping) or set(path) != {"1", "2", "3", "4"}:
            raise ValueError(f"direct train selection is missing {policy} path")
        previous: tuple[int, ...] = ()
        for budget in range(1, 5):
            subset = _canonical_subset(
                path[str(budget)], candidates=candidates, label=f"{policy} B{budget}"
            )
            if len(subset) > budget or not set(previous).issubset(subset):
                raise ValueError(f"{policy} path is not nested at-most-B")
            previous = subset


def collection_bases(
    record: Mapping[str, Any],
    *,
    maximum_base_cardinality: int,
) -> dict[tuple[int, ...], tuple[str, ...]]:
    """Return unique bases visited by direct, recent, and hybrid paths."""
    if maximum_base_cardinality < 0:
        raise ValueError("maximum base cardinality must be non-negative")
    validate_direct_train_selection_record(record)
    candidates = tuple(record["candidate_event_ids"])
    sources: dict[tuple[int, ...], set[str]] = defaultdict(set)
    sources[()].add("shared_empty")
    for policy in POLICY_KEYS:
        path = record[policy]
        for budget in range(1, maximum_base_cardinality + 1):
            subset = _canonical_subset(
                path[str(budget)], candidates=candidates, label=f"{policy} B{budget}"
            )
            if len(subset) <= maximum_base_cardinality:
                sources[subset].add(f"{policy}_b{budget}")
    return {
        subset: tuple(sorted(values))
        for subset, values in sorted(
            sources.items(), key=lambda item: (len(item[0]), item[0])
        )
    }


def candidate_complete_coalitions(
    record: Mapping[str, Any],
    *,
    maximum_base_cardinality: int,
) -> dict[tuple[int, ...], tuple[str, ...]]:
    """Expand every collected base by every eligible one-event action."""
    candidates = tuple(record["candidate_event_ids"])
    bases = collection_bases(
        record, maximum_base_cardinality=maximum_base_cardinality
    )
    sources: dict[tuple[int, ...], set[str]] = defaultdict(set)
    for base, base_sources in bases.items():
        sources[base].update(f"{source}_base" for source in base_sources)
        for event_id in candidates:
            if event_id in base:
                continue
            expanded = tuple(sorted((*base, event_id)))
            sources[expanded].update(
                f"{source}_expand" for source in base_sources
            )
    sources[candidates].add("full_anchor")
    return {
        subset: tuple(sorted(values))
        for subset, values in sorted(
            sources.items(), key=lambda item: (len(item[0]), item[0])
        )
    }


def missing_candidate_complete_coalitions(
    record: Mapping[str, Any],
    existing_coalitions: Sequence[Sequence[int]],
    *,
    maximum_base_cardinality: int,
) -> tuple[dict[str, Any], ...]:
    desired = candidate_complete_coalitions(
        record, maximum_base_cardinality=maximum_base_cardinality
    )
    existing = {tuple(value) for value in existing_coalitions}
    return tuple(
        {
            "event_ids": list(subset),
            "source": "+".join(sources),
        }
        for subset, sources in desired.items()
        if subset not in existing
    )


def runner_schedule_coalitions(
    candidates: Sequence[int],
    missing: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Add the zero-cost full-history anchor required by the label runner."""
    candidate_tuple = tuple(candidates)
    if (
        not candidate_tuple
        or candidate_tuple != tuple(sorted(candidate_tuple))
        or len(candidate_tuple) != len(set(candidate_tuple))
        or any(type(item) is not int for item in candidate_tuple)
    ):
        raise ValueError("runner schedule candidate universe is invalid")
    rows: dict[tuple[int, ...], dict[str, Any]] = {}
    for index, row in enumerate(missing):
        if not isinstance(row, Mapping) or not isinstance(row.get("source"), str):
            raise ValueError("runner schedule coalition row is invalid")
        subset = _canonical_subset(
            row.get("event_ids"),
            candidates=candidate_tuple,
            label=f"runner schedule coalition {index}",
        )
        if subset in rows:
            raise ValueError("runner schedule coalition is duplicated")
        rows[subset] = {"event_ids": list(subset), "source": row["source"]}
    rows.setdefault(
        candidate_tuple,
        {"event_ids": list(candidate_tuple), "source": "full_anchor"},
    )
    return tuple(
        rows[subset]
        for subset in sorted(rows, key=lambda value: (len(value), value))
    )


def complete_group_count(
    candidates: Sequence[int],
    coalitions: Sequence[Sequence[int]],
    *,
    maximum_base_cardinality: int,
) -> int:
    candidate_tuple = tuple(candidates)
    available = {tuple(value) for value in coalitions}
    result = 0
    for base in available:
        if len(base) > maximum_base_cardinality:
            continue
        remaining = tuple(
            event_id for event_id in candidate_tuple if event_id not in base
        )
        if not remaining:
            continue
        if all(
            tuple(sorted((*base, event_id))) in available
            for event_id in remaining
        ):
            result += 1
    return result


__all__ = [
    "POLICY_KEYS",
    "candidate_complete_coalitions",
    "collection_bases",
    "complete_group_count",
    "missing_candidate_complete_coalitions",
    "runner_schedule_coalitions",
    "validate_direct_train_selection_record",
]
