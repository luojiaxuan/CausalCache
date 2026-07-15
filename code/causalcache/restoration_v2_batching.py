"""Deterministic coalition microbatch planning for restoration-v2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


FROZEN_COALITION_MICROBATCH_SIZE = 2
GROUPING_FIELDS = ("image_count", "sequence_length")
PLANNER_SCHEMA_VERSION = "0.1.0"


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        comparator = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {comparator}")
    return value


@dataclass(frozen=True)
class CoalitionMicrobatchRecord:
    """Minimal scheduling identity for one coalition forward."""

    coalition_id: str
    image_count: int
    sequence_length: int
    input_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.coalition_id, str):
            raise TypeError("coalition_id must be a string")
        if not self.coalition_id.strip():
            raise ValueError("coalition_id must not be empty")
        _integer(self.image_count, name="image_count", minimum=1)
        _integer(self.sequence_length, name="sequence_length", minimum=1)
        _integer(self.input_index, name="input_index", minimum=0)


@dataclass(frozen=True)
class CoalitionMicrobatch:
    """One exact-shape batch in deterministic execution order."""

    microbatch_index: int
    group_index: int
    image_count: int
    sequence_length: int
    records: tuple[CoalitionMicrobatchRecord, ...]

    @property
    def coalition_ids(self) -> tuple[str, ...]:
        return tuple(record.coalition_id for record in self.records)

    @property
    def input_indices(self) -> tuple[int, ...]:
        return tuple(record.input_index for record in self.records)


@dataclass(frozen=True)
class CoalitionMicrobatchGroupAudit:
    """Auditable summary for one exact image/sequence shape group."""

    group_index: int
    image_count: int
    sequence_length: int
    record_count: int
    coalition_ids: tuple[str, ...]
    input_indices: tuple[int, ...]
    microbatch_indices: tuple[int, ...]
    microbatch_sizes: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_index": self.group_index,
            "image_count": self.image_count,
            "sequence_length": self.sequence_length,
            "record_count": self.record_count,
            "coalition_ids": list(self.coalition_ids),
            "input_indices": list(self.input_indices),
            "microbatch_indices": list(self.microbatch_indices),
            "microbatch_sizes": list(self.microbatch_sizes),
        }


@dataclass(frozen=True)
class CoalitionMicrobatchAudit:
    """JSON-ready provenance for a deterministic planning decision."""

    input_record_count: int
    group_count: int
    microbatch_count: int
    groups: tuple[CoalitionMicrobatchGroupAudit, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLANNER_SCHEMA_VERSION,
            "microbatch_size": FROZEN_COALITION_MICROBATCH_SIZE,
            "microbatch_size_source": "explicit_argument",
            "automatic_oom_fallback": False,
            "grouping_fields": list(GROUPING_FIELDS),
            "group_order": "ascending_image_count_then_sequence_length",
            "within_group_order": "ascending_input_index",
            "input_record_count": self.input_record_count,
            "group_count": self.group_count,
            "microbatch_count": self.microbatch_count,
            "groups": [group.as_dict() for group in self.groups],
        }


@dataclass(frozen=True)
class CoalitionMicrobatchPlan:
    """Immutable execution plan and its audit metadata."""

    microbatches: tuple[CoalitionMicrobatch, ...]
    audit: CoalitionMicrobatchAudit


def plan_coalition_microbatches(
    records: Iterable[CoalitionMicrobatchRecord],
    *,
    microbatch_size: int,
) -> CoalitionMicrobatchPlan:
    """Group exact shapes and split them into frozen-size deterministic batches."""
    requested_size = _integer(
        microbatch_size,
        name="microbatch_size",
        minimum=1,
    )
    if requested_size != FROZEN_COALITION_MICROBATCH_SIZE:
        raise ValueError(
            "microbatch_size must equal the frozen restoration-v2 value "
            f"{FROZEN_COALITION_MICROBATCH_SIZE}"
        )
    if isinstance(records, (str, bytes, bytearray, Mapping)):
        raise TypeError("records must be an iterable of CoalitionMicrobatchRecord")
    try:
        materialized = tuple(records)
    except TypeError as error:
        raise TypeError(
            "records must be an iterable of CoalitionMicrobatchRecord"
        ) from error
    if not materialized:
        raise ValueError("records must not be empty")
    if any(not isinstance(record, CoalitionMicrobatchRecord) for record in materialized):
        raise TypeError("records must contain only CoalitionMicrobatchRecord values")

    coalition_ids = [record.coalition_id for record in materialized]
    input_indices = [record.input_index for record in materialized]
    if len(set(coalition_ids)) != len(coalition_ids):
        raise ValueError("coalition_id values must be unique")
    if len(set(input_indices)) != len(input_indices):
        raise ValueError("input_index values must be unique")

    # note (luojiaxuan): input_index is the canonical caller-supplied order. Sorting
    # before grouping makes the same logical input independent of iterable traversal.
    canonical_records = tuple(sorted(materialized, key=lambda record: record.input_index))
    grouped: dict[tuple[int, int], list[CoalitionMicrobatchRecord]] = defaultdict(list)
    for record in canonical_records:
        grouped[(record.image_count, record.sequence_length)].append(record)

    microbatches: list[CoalitionMicrobatch] = []
    group_audits: list[CoalitionMicrobatchGroupAudit] = []
    for group_index, (group_key, group_records) in enumerate(sorted(grouped.items())):
        image_count, sequence_length = group_key
        first_microbatch_index = len(microbatches)
        for offset in range(0, len(group_records), requested_size):
            batch_records = tuple(group_records[offset : offset + requested_size])
            microbatches.append(
                CoalitionMicrobatch(
                    microbatch_index=len(microbatches),
                    group_index=group_index,
                    image_count=image_count,
                    sequence_length=sequence_length,
                    records=batch_records,
                )
            )
        group_microbatches = microbatches[first_microbatch_index:]
        group_audits.append(
            CoalitionMicrobatchGroupAudit(
                group_index=group_index,
                image_count=image_count,
                sequence_length=sequence_length,
                record_count=len(group_records),
                coalition_ids=tuple(record.coalition_id for record in group_records),
                input_indices=tuple(record.input_index for record in group_records),
                microbatch_indices=tuple(
                    batch.microbatch_index for batch in group_microbatches
                ),
                microbatch_sizes=tuple(len(batch.records) for batch in group_microbatches),
            )
        )

    batches = tuple(microbatches)
    audit = CoalitionMicrobatchAudit(
        input_record_count=len(canonical_records),
        group_count=len(group_audits),
        microbatch_count=len(batches),
        groups=tuple(group_audits),
    )
    return CoalitionMicrobatchPlan(microbatches=batches, audit=audit)
