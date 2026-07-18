"""Pure-CPU scheduling for variable-n cardinality-capped utility labels."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "1.0.0"
MAXIMUM_CANDIDATE_COUNT = 16
SUPPORTED_MAXIMUM_CARDINALITIES = (2, 3, 4)
MAXIMUM_WORKER_COUNT = 4


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_canonical_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class LabelStateRequest:
    """Frozen identity and candidate universe for one exact label table."""

    state_id: str
    candidate_event_ids: tuple[int, ...]
    maximum_cardinality: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.state_id, str)
            or not self.state_id
            or self.state_id != self.state_id.strip()
        ):
            raise ValueError("label state id must be non-empty canonical text")
        if (
            not self.candidate_event_ids
            or len(self.candidate_event_ids) > MAXIMUM_CANDIDATE_COUNT
            or any(
                type(event_id) is not int or event_id <= 0
                for event_id in self.candidate_event_ids
            )
            or self.candidate_event_ids
            != tuple(sorted(self.candidate_event_ids))
            or len(set(self.candidate_event_ids))
            != len(self.candidate_event_ids)
        ):
            raise ValueError(
                "candidate event ids must contain 1..16 sorted unique positive integers"
            )
        if (
            type(self.maximum_cardinality) is not int
            or self.maximum_cardinality
            not in SUPPORTED_MAXIMUM_CARDINALITIES
        ):
            raise ValueError("maximum cardinality must be exactly 2, 3, or 4")


@dataclass(frozen=True)
class ScheduledSubsetForward:
    state_id: str
    subset_index: int
    coalition_event_ids: tuple[int, ...]

    @property
    def cardinality(self) -> int:
        return len(self.coalition_event_ids)


@dataclass(frozen=True)
class TeacherOperationBudget:
    canonical_action_generations: int
    reference_teacher_forwards: int
    identical_reference_repeat_teacher_forwards: int
    capped_candidate_teacher_forwards: int
    total_teacher_forwards: int
    kl_measurements: int
    raw_label_rows: int
    total_model_operations: int


@dataclass(frozen=True)
class StateLabelSchedule:
    request: LabelStateRequest
    subsets: tuple[ScheduledSubsetForward, ...]
    operations: TeacherOperationBudget


@dataclass(frozen=True)
class WorkerStateShard:
    worker_index: int
    states: tuple[StateLabelSchedule, ...]
    operations: TeacherOperationBudget

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(state.request.state_id for state in self.states)


@dataclass(frozen=True)
class VariableNLabelSchedule:
    schema_version: str
    worker_count: int
    states: tuple[StateLabelSchedule, ...]
    workers: tuple[WorkerStateShard, ...]
    operations: TeacherOperationBudget
    subset_count_by_cardinality: tuple[tuple[int, int], ...]
    state_count_by_candidate_count: tuple[tuple[int, int], ...]
    state_count_by_maximum_cardinality: tuple[tuple[int, int], ...]
    subset_identity_sha256: str
    inventory_sha256: str
    execution_sha256: str


@dataclass(frozen=True)
class CompletedSubsetIdentity:
    state_id: str
    coalition_event_ids: tuple[int, ...]


@dataclass(frozen=True)
class CompletedInventoryValidation:
    row_count: int
    subset_identity_sha256: str
    complete: bool


def expected_subset_forward_count(
    candidate_count: int,
    maximum_cardinality: int,
) -> int:
    if (
        type(candidate_count) is not int
        or not 1 <= candidate_count <= MAXIMUM_CANDIDATE_COUNT
    ):
        raise ValueError("candidate count must be an integer in [1, 16]")
    if (
        type(maximum_cardinality) is not int
        or maximum_cardinality not in SUPPORTED_MAXIMUM_CARDINALITIES
    ):
        raise ValueError("maximum cardinality must be exactly 2, 3, or 4")
    return sum(
        math.comb(candidate_count, cardinality)
        for cardinality in range(min(candidate_count, maximum_cardinality) + 1)
    )


def enumerate_cardinality_capped_subsets(
    candidate_event_ids: Sequence[int],
    *,
    maximum_cardinality: int,
) -> tuple[tuple[int, ...], ...]:
    """Return all |S|<=K subsets in cardinality then lexicographic order."""
    request = LabelStateRequest(
        state_id="__inventory_validation__",
        candidate_event_ids=tuple(candidate_event_ids),
        maximum_cardinality=maximum_cardinality,
    )
    maximum = min(len(request.candidate_event_ids), maximum_cardinality)
    return tuple(
        coalition
        for cardinality in range(maximum + 1)
        for coalition in itertools.combinations(
            request.candidate_event_ids, cardinality
        )
    )


def _operation_budget(
    state_count: int,
    subset_count: int,
    labeled_full_candidate_count: int,
) -> TeacherOperationBudget:
    if state_count < 0 or subset_count < 0 or labeled_full_candidate_count < 0:
        raise ValueError("operation counts must be non-negative")
    if labeled_full_candidate_count > state_count:
        raise ValueError("labeled full-candidate count cannot exceed state count")
    candidate_forwards = subset_count - labeled_full_candidate_count
    if candidate_forwards < 0:
        raise ValueError("candidate teacher forward count cannot be negative")
    teacher_forwards = 2 * state_count + candidate_forwards
    generations = 2 * state_count
    return TeacherOperationBudget(
        canonical_action_generations=generations,
        reference_teacher_forwards=state_count,
        identical_reference_repeat_teacher_forwards=state_count,
        capped_candidate_teacher_forwards=candidate_forwards,
        total_teacher_forwards=teacher_forwards,
        kl_measurements=state_count + candidate_forwards,
        raw_label_rows=subset_count,
        total_model_operations=generations + teacher_forwards,
    )


def build_state_label_schedule(request: LabelStateRequest) -> StateLabelSchedule:
    if not isinstance(request, LabelStateRequest):
        raise TypeError("state label request must be LabelStateRequest")
    coalitions = enumerate_cardinality_capped_subsets(
        request.candidate_event_ids,
        maximum_cardinality=request.maximum_cardinality,
    )
    subsets = tuple(
        ScheduledSubsetForward(
            state_id=request.state_id,
            subset_index=index,
            coalition_event_ids=coalition,
        )
        for index, coalition in enumerate(coalitions)
    )
    expected = expected_subset_forward_count(
        len(request.candidate_event_ids), request.maximum_cardinality
    )
    if len(subsets) != expected:
        raise RuntimeError("cardinality-capped subset count drifted")
    return StateLabelSchedule(
        request=request,
        subsets=subsets,
        operations=_operation_budget(
            1,
            len(subsets),
            int(len(request.candidate_event_ids) <= request.maximum_cardinality),
        ),
    )


def _subset_identity_payload(
    states: Sequence[StateLabelSchedule],
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": [
            {
                "state_id": state.request.state_id,
                "coalition_event_ids": list(subset.coalition_event_ids),
            }
            for state in states
            for subset in state.subsets
        ],
    }


def _inventory_payload(
    states: Sequence[StateLabelSchedule],
    *,
    operations: TeacherOperationBudget,
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "states": [
            {
                "state_id": state.request.state_id,
                "candidate_event_ids": list(state.request.candidate_event_ids),
                "maximum_cardinality": state.request.maximum_cardinality,
                "subsets": [
                    {
                        "subset_index": subset.subset_index,
                        "coalition_event_ids": list(subset.coalition_event_ids),
                    }
                    for subset in state.subsets
                ],
                "operations": _operations_payload(state.operations),
            }
            for state in states
        ],
        "operations": _operations_payload(operations),
    }


def _operations_payload(value: TeacherOperationBudget) -> Mapping[str, int]:
    return {
        "canonical_action_generations": value.canonical_action_generations,
        "reference_teacher_forwards": value.reference_teacher_forwards,
        "identical_reference_repeat_teacher_forwards": (
            value.identical_reference_repeat_teacher_forwards
        ),
        "capped_candidate_teacher_forwards": (
            value.capped_candidate_teacher_forwards
        ),
        "total_teacher_forwards": value.total_teacher_forwards,
        "kl_measurements": value.kl_measurements,
        "raw_label_rows": value.raw_label_rows,
        "total_model_operations": value.total_model_operations,
    }


def _execution_payload(
    *,
    inventory_sha256: str,
    workers: Sequence[WorkerStateShard],
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "inventory_sha256": inventory_sha256,
        "worker_count": len(workers),
        "workers": [
            {
                "worker_index": worker.worker_index,
                "state_ids": list(worker.state_ids),
                "operations": _operations_payload(worker.operations),
            }
            for worker in workers
        ],
    }


def build_variable_n_label_schedule(
    requests: Sequence[LabelStateRequest],
    *,
    worker_count: int,
) -> VariableNLabelSchedule:
    """Build a deterministic state-level schedule for at most four workers."""
    if (
        isinstance(requests, (str, bytes, bytearray, Mapping))
        or not isinstance(requests, Sequence)
        or not requests
    ):
        raise ValueError("label schedule requires a non-empty request sequence")
    if any(not isinstance(request, LabelStateRequest) for request in requests):
        raise TypeError("label schedule requests must be LabelStateRequest values")
    if type(worker_count) is not int or not 1 <= worker_count <= MAXIMUM_WORKER_COUNT:
        raise ValueError("worker count must be an integer in [1, 4]")
    state_ids = tuple(request.state_id for request in requests)
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("label schedule contains a duplicate state id")

    ordered_requests = tuple(sorted(requests, key=lambda item: item.state_id))
    states = tuple(build_state_label_schedule(request) for request in ordered_requests)
    state_by_id = {state.request.state_id: state for state in states}
    worker_state_ids: list[list[str]] = [[] for _ in range(worker_count)]
    worker_operation_loads = [0] * worker_count
    for state in sorted(
        states,
        key=lambda item: (
            -item.operations.total_model_operations,
            item.request.state_id,
        ),
    ):
        worker_index = min(
            range(worker_count),
            key=lambda index: (worker_operation_loads[index], index),
        )
        worker_state_ids[worker_index].append(state.request.state_id)
        worker_operation_loads[worker_index] += (
            state.operations.total_model_operations
        )
    for assigned_ids in worker_state_ids:
        assigned_ids.sort()
    workers = tuple(
        WorkerStateShard(
            worker_index=worker_index,
            states=tuple(state_by_id[state_id] for state_id in assigned_ids),
            operations=_operation_budget(
                len(assigned_ids),
                sum(
                    len(state_by_id[state_id].subsets)
                    for state_id in assigned_ids
                ),
                sum(
                    int(
                        len(state_by_id[state_id].request.candidate_event_ids)
                        <= state_by_id[state_id].request.maximum_cardinality
                    )
                    for state_id in assigned_ids
                ),
            ),
        )
        for worker_index, assigned_ids in enumerate(worker_state_ids)
    )
    operations = _operation_budget(
        len(states),
        sum(len(state.subsets) for state in states),
        sum(
            int(
                len(state.request.candidate_event_ids)
                <= state.request.maximum_cardinality
            )
            for state in states
        ),
    )
    subset_cardinalities = Counter(
        subset.cardinality for state in states for subset in state.subsets
    )
    candidate_counts = Counter(len(state.request.candidate_event_ids) for state in states)
    maximum_cardinalities = Counter(
        state.request.maximum_cardinality for state in states
    )
    identity_sha = sha256_canonical_json(_subset_identity_payload(states))
    inventory_sha = sha256_canonical_json(
        _inventory_payload(states, operations=operations)
    )
    execution_sha = sha256_canonical_json(
        _execution_payload(inventory_sha256=inventory_sha, workers=workers)
    )
    return VariableNLabelSchedule(
        schema_version=SCHEMA_VERSION,
        worker_count=worker_count,
        states=states,
        workers=workers,
        operations=operations,
        subset_count_by_cardinality=tuple(sorted(subset_cardinalities.items())),
        state_count_by_candidate_count=tuple(sorted(candidate_counts.items())),
        state_count_by_maximum_cardinality=tuple(
            sorted(maximum_cardinalities.items())
        ),
        subset_identity_sha256=identity_sha,
        inventory_sha256=inventory_sha,
        execution_sha256=execution_sha,
    )


def validate_variable_n_label_schedule(
    schedule: VariableNLabelSchedule,
) -> VariableNLabelSchedule:
    """Rebuild every row, count, shard, and digest and require exact equality."""
    if not isinstance(schedule, VariableNLabelSchedule):
        raise TypeError("schedule must be VariableNLabelSchedule")
    expected = build_variable_n_label_schedule(
        tuple(state.request for state in schedule.states),
        worker_count=schedule.worker_count,
    )
    if schedule != expected:
        raise ValueError("variable-n label schedule is incomplete or drifted")
    return schedule


def validate_completed_subset_inventory(
    schedule: VariableNLabelSchedule,
    completed_rows: Sequence[CompletedSubsetIdentity],
) -> CompletedInventoryValidation:
    """Reject duplicate, missing, or unexpected completed subset identities."""
    validate_variable_n_label_schedule(schedule)
    if (
        isinstance(completed_rows, (str, bytes, bytearray, Mapping))
        or not isinstance(completed_rows, Sequence)
    ):
        raise TypeError("completed subset rows must be a sequence")
    if any(not isinstance(row, CompletedSubsetIdentity) for row in completed_rows):
        raise TypeError("completed rows must be CompletedSubsetIdentity values")
    expected = tuple(
        CompletedSubsetIdentity(
            state_id=state.request.state_id,
            coalition_event_ids=subset.coalition_event_ids,
        )
        for state in schedule.states
        for subset in state.subsets
    )
    expected_keys = {
        (row.state_id, row.coalition_event_ids) for row in expected
    }
    observed_keys: set[tuple[str, tuple[int, ...]]] = set()
    for row in completed_rows:
        key = (row.state_id, tuple(row.coalition_event_ids))
        if key in observed_keys:
            raise ValueError("completed subset inventory contains a duplicate row")
        observed_keys.add(key)
    missing = expected_keys - observed_keys
    unexpected = observed_keys - expected_keys
    if missing or unexpected:
        raise ValueError(
            "completed subset inventory differs from the frozen schedule: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    canonical_completed = tuple(
        row for row in expected if (row.state_id, row.coalition_event_ids) in observed_keys
    )
    digest = sha256_canonical_json(
        {
            "schema_version": SCHEMA_VERSION,
            "rows": [
                {
                    "state_id": row.state_id,
                    "coalition_event_ids": list(row.coalition_event_ids),
                }
                for row in canonical_completed
            ],
        }
    )
    if digest != schedule.subset_identity_sha256:
        raise RuntimeError("completed subset identity digest drifted")
    return CompletedInventoryValidation(
        row_count=len(canonical_completed),
        subset_identity_sha256=digest,
        complete=True,
    )


__all__ = [
    "CompletedInventoryValidation",
    "CompletedSubsetIdentity",
    "LabelStateRequest",
    "MAXIMUM_CANDIDATE_COUNT",
    "MAXIMUM_WORKER_COUNT",
    "SCHEMA_VERSION",
    "SUPPORTED_MAXIMUM_CARDINALITIES",
    "ScheduledSubsetForward",
    "StateLabelSchedule",
    "TeacherOperationBudget",
    "VariableNLabelSchedule",
    "WorkerStateShard",
    "build_state_label_schedule",
    "build_variable_n_label_schedule",
    "canonical_json_bytes",
    "enumerate_cardinality_capped_subsets",
    "expected_subset_forward_count",
    "sha256_canonical_json",
    "validate_completed_subset_inventory",
    "validate_variable_n_label_schedule",
]
