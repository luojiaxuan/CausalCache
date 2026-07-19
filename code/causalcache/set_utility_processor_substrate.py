"""Selected-row and low-fidelity substrate for set-utility processor runs."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.data.guiodyssey import build_pilot_manifest
from causalcache.data.guiodyssey_independent import (
    SourceFileSpec,
    inspect_candidate,
    normalize_app_label,
)
from causalcache.data.guiodyssey_restoration_v2 import (
    _build_derived_event,
    generate_ocr_records,
    validate_ocr_record,
)
from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_full_pool import (
    _first_instruction_text,
    candidate_capacity_stratum,
)
from causalcache.set_utility_label_producer import UtilityHistoryEvent
from causalcache.set_utility_long_pool import instruction_app_group_sha256


ASSIGNMENT_KEYS = {
    "candidate_capacity_stratum",
    "decision_count",
    "instruction_app_group_sha256",
    "p0_selection_sha256",
    "role",
    "source_id",
    "terminal_decision_step_id",
    "trajectory_id",
    "transport_file",
    "transport_row_index",
}
QUERY_STATE_KEYS = {
    "candidate_capacity_stratum",
    "current_equivalent_event_step_id",
    "decision_step_id",
    "initial_candidate_count",
    "initial_candidate_event_step_ids",
    "maximum_labeled_cardinality",
    "processor_candidate_freeze_status",
    "query_kind",
    "role",
    "source_id",
    "state_id",
    "trajectory_id",
}
SPLIT_ROLES = ("train", "tune", "evaluation")
QUERY_KINDS = ("stratum_anchor", "terminal")
STRATUM_ANCHOR_DECISION_STEPS = {
    "decisions_6_9": 6,
    "decisions_10_17": 10,
    "decisions_18_plus": 18,
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _identity(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _ordered_positive_ids(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(f"{label} must be an ordered sequence")
    result = tuple(value)
    if (
        not result
        or any(type(item) is not int or item <= 0 for item in result)
        or result != tuple(sorted(result))
        or len(result) != len(set(result))
    ):
        raise ValueError(f"{label} must contain sorted unique positive integers")
    return result


@dataclass(frozen=True)
class SelectedTrajectoryAssignment:
    trajectory_id: str
    source_id: str
    instruction_app_group_sha256: str
    role: str
    candidate_capacity_stratum: str
    decision_count: int
    terminal_decision_step_id: int
    transport_file: str
    transport_row_index: int
    p0_selection_sha256: str

    def __post_init__(self) -> None:
        trajectory_id = _identity(self.trajectory_id, label="trajectory id")
        source_id = _identity(self.source_id, label="source id")
        if trajectory_id != source_id:
            raise ValueError("GUIOdyssey trajectory and source identities must match")
        _sha256(
            self.instruction_app_group_sha256,
            label="instruction-app group SHA256",
        )
        _sha256(self.p0_selection_sha256, label="P0 selection SHA256")
        if self.role not in SPLIT_ROLES:
            raise ValueError(f"assignment role must be one of {SPLIT_ROLES!r}")
        decision_count = _positive_int(self.decision_count, label="decision count")
        if decision_count < 6:
            raise ValueError("selected trajectory is below the source minimum")
        if self.terminal_decision_step_id != decision_count + 1:
            raise ValueError("terminal decision step must equal decision_count + 1")
        if self.candidate_capacity_stratum != candidate_capacity_stratum(
            decision_count
        ):
            raise ValueError("assignment candidate-capacity stratum drifted")
        _identity(self.transport_file, label="transport file")
        _nonnegative_int(self.transport_row_index, label="transport row index")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SelectedTrajectoryAssignment:
        record = _mapping(value, label="selected assignment")
        if set(record) != ASSIGNMENT_KEYS:
            raise ValueError("selected assignment fields drifted")
        return cls(
            trajectory_id=record["trajectory_id"],
            source_id=record["source_id"],
            instruction_app_group_sha256=record[
                "instruction_app_group_sha256"
            ],
            role=record["role"],
            candidate_capacity_stratum=record["candidate_capacity_stratum"],
            decision_count=record["decision_count"],
            terminal_decision_step_id=record["terminal_decision_step_id"],
            transport_file=record["transport_file"],
            transport_row_index=record["transport_row_index"],
            p0_selection_sha256=record["p0_selection_sha256"],
        )


@dataclass(frozen=True)
class SelectedShardRead:
    source_file: SourceFileSpec
    source_row_count: int
    assignments: tuple[SelectedTrajectoryAssignment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_file, SourceFileSpec):
            raise TypeError("selected shard source file must be SourceFileSpec")
        row_count = _positive_int(self.source_row_count, label="source row count")
        if not self.assignments:
            raise ValueError("selected shard must contain at least one assignment")
        indices = tuple(item.transport_row_index for item in self.assignments)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("selected shard row indices must be sorted and unique")
        if any(
            item.transport_file != self.source_file.transport_file
            for item in self.assignments
        ):
            raise ValueError("selected assignment transport file drifted")
        if indices[-1] >= row_count:
            raise ValueError("selected transport row is outside the frozen shard")


@dataclass(frozen=True)
class SelectedRowReadPlan:
    shards: tuple[SelectedShardRead, ...]
    assignment_count: int

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("selected-row read plan must contain at least one shard")
        paths = tuple(item.source_file.transport_file for item in self.shards)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("selected-row shards must be path-sorted and unique")
        if self.assignment_count != sum(
            len(item.assignments) for item in self.shards
        ):
            raise ValueError("selected-row assignment count drifted")


@dataclass(frozen=True)
class LoadedSelectedRow:
    assignment: SelectedTrajectoryAssignment
    source_file: SourceFileSpec
    row: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, SelectedTrajectoryAssignment):
            raise TypeError("loaded row assignment is invalid")
        if not isinstance(self.source_file, SourceFileSpec):
            raise TypeError("loaded row source file is invalid")
        _mapping(self.row, label="loaded selected row")
        if self.assignment.transport_file != self.source_file.transport_file:
            raise ValueError("loaded row source-file identity drifted")


@dataclass(frozen=True)
class SourceProvenance:
    upstream_repo: str
    upstream_revision: str
    transport_repo: str
    transport_revision: str
    derived_repo: str
    coordinate_grid_size: int = 10

    def __post_init__(self) -> None:
        for value, label in (
            (self.upstream_repo, "upstream repo"),
            (self.upstream_revision, "upstream revision"),
            (self.transport_repo, "transport repo"),
            (self.transport_revision, "transport revision"),
            (self.derived_repo, "derived repo"),
        ):
            _identity(value, label=label)
        _positive_int(self.coordinate_grid_size, label="coordinate grid size")


@dataclass(frozen=True)
class SelectedPilot:
    assignment: SelectedTrajectoryAssignment
    source_file: SourceFileSpec
    manifest: Mapping[str, Any]
    image_payloads: Mapping[str, bytes]

    def __post_init__(self) -> None:
        _mapping(self.manifest, label="selected pilot manifest")
        if not isinstance(self.image_payloads, Mapping) or not self.image_payloads:
            raise ValueError("selected pilot image payloads must be non-empty")


@dataclass(frozen=True)
class ValidatedOcrBatch:
    records_by_path: Mapping[str, Mapping[str, Any]]
    prepared_by_path: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.records_by_path or set(self.records_by_path) != set(
            self.prepared_by_path
        ):
            raise ValueError("OCR record and prepared-image inventories differ")


@dataclass(frozen=True)
class QuerySubstrateRef:
    state_id: str
    query_kind: str
    decision_step_id: int
    current_equivalent_event_step_id: int
    initial_candidate_event_step_ids: tuple[int, ...]
    maximum_labeled_cardinality: int
    current_observation_ref: str

    def __post_init__(self) -> None:
        _identity(self.state_id, label="query state id")
        if self.query_kind not in QUERY_KINDS:
            raise ValueError(f"query kind must be one of {QUERY_KINDS!r}")
        step = _positive_int(self.decision_step_id, label="query decision step")
        if self.current_equivalent_event_step_id != step - 1:
            raise ValueError("query current-equivalent event is not decision_step - 1")
        candidates = _ordered_positive_ids(
            self.initial_candidate_event_step_ids,
            label="initial candidate event ids",
        )
        if candidates[-1] >= self.current_equivalent_event_step_id:
            raise ValueError("current-equivalent event leaked into query candidates")
        if type(self.maximum_labeled_cardinality) is not int or (
            self.maximum_labeled_cardinality not in (2, 3, 4)
        ):
            raise ValueError("maximum labeled cardinality must be 2, 3, or 4")
        _identity(self.current_observation_ref, label="current observation reference")

    def to_payload(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "query_kind": self.query_kind,
            "decision_step_id": self.decision_step_id,
            "current_equivalent_event_step_id": (
                self.current_equivalent_event_step_id
            ),
            "initial_candidate_event_step_ids": list(
                self.initial_candidate_event_step_ids
            ),
            "maximum_labeled_cardinality": self.maximum_labeled_cardinality,
            "current_observation_ref": self.current_observation_ref,
        }


@dataclass(frozen=True)
class TrajectoryProcessorSubstrate:
    assignment: SelectedTrajectoryAssignment
    transport_file_sha256: str
    task_instruction: str
    history_events: tuple[UtilityHistoryEvent, ...]
    derived_event_records: tuple[Mapping[str, Any], ...]
    query_states: tuple[QuerySubstrateRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, SelectedTrajectoryAssignment):
            raise TypeError("trajectory substrate assignment is invalid")
        _sha256(self.transport_file_sha256, label="transport file SHA256")
        _identity(self.task_instruction, label="task instruction")
        if not self.history_events or any(
            not isinstance(event, UtilityHistoryEvent)
            for event in self.history_events
        ):
            raise ValueError("trajectory substrate history is invalid")
        event_ids = tuple(event.event_step_id for event in self.history_events)
        if event_ids != tuple(range(1, self.assignment.decision_count + 1)):
            raise ValueError("trajectory substrate does not cover the full event prefix")
        if len(self.derived_event_records) != len(self.history_events):
            raise ValueError("derived event records do not match utility history")
        if not self.query_states:
            raise ValueError("trajectory substrate must contain query states")
        if len({query.state_id for query in self.query_states}) != len(
            self.query_states
        ):
            raise ValueError("trajectory substrate query state IDs are duplicated")

    def utility_history_for_query(
        self, query: QuerySubstrateRef
    ) -> tuple[UtilityHistoryEvent, ...]:
        if query not in self.query_states:
            raise ValueError("query does not belong to this trajectory substrate")
        return self.history_events[: query.current_equivalent_event_step_id]

    def to_payload(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.assignment.trajectory_id,
            "source_id": self.assignment.source_id,
            "role": self.assignment.role,
            "instruction_app_group_sha256": (
                self.assignment.instruction_app_group_sha256
            ),
            "decision_count": self.assignment.decision_count,
            "terminal_decision_step_id": (
                self.assignment.terminal_decision_step_id
            ),
            "transport_file": self.assignment.transport_file,
            "transport_row_index": self.assignment.transport_row_index,
            "transport_file_sha256": self.transport_file_sha256,
            "task_instruction": self.task_instruction,
            "history_events": [
                {
                    "event_step_id": event.event_step_id,
                    "low_fidelity_summary": (
                        event.low_fidelity_summary.to_ordered_dict()
                    ),
                    "high_fidelity_observation_ref": (
                        event.high_fidelity_observation_ref
                    ),
                }
                for event in self.history_events
            ],
            "derived_event_records": [dict(record) for record in self.derived_event_records],
            "query_states": [query.to_payload() for query in self.query_states],
        }


RowIteratorFactory = Callable[
    [SourceFileSpec], Iterable[tuple[int, Mapping[str, Any]]]
]
OcrBatchBuilder = Callable[[Mapping[str, bytes]], ValidatedOcrBatch]
DerivedEventBuilder = Callable[..., Mapping[str, Any]]


def build_selected_row_read_plan(
    assignments: Sequence[Mapping[str, Any]],
    *,
    source_files: Sequence[SourceFileSpec],
    source_row_counts: Mapping[str, int],
) -> SelectedRowReadPlan:
    """Group exact selected row indices by pinned shard in canonical order."""
    if isinstance(assignments, (str, bytes, bytearray, Mapping)) or not assignments:
        raise TypeError("selected assignments must be a non-empty ordered sequence")
    specs = tuple(source_files)
    if not specs or any(not isinstance(spec, SourceFileSpec) for spec in specs):
        raise TypeError("source files must be a non-empty SourceFileSpec sequence")
    spec_by_path = {spec.transport_file: spec for spec in specs}
    if len(spec_by_path) != len(specs):
        raise ValueError("source-file transport paths must be unique")
    if set(source_row_counts) != set(spec_by_path):
        raise ValueError("source row-count inventory differs from source files")
    normalized = tuple(
        SelectedTrajectoryAssignment.from_mapping(value) for value in assignments
    )
    if len({item.source_id for item in normalized}) != len(normalized):
        raise ValueError("selected assignment source IDs are not unique")
    if len(
        {(item.transport_file, item.transport_row_index) for item in normalized}
    ) != len(normalized):
        raise ValueError("selected assignment transport rows are not unique")

    grouped: dict[str, list[SelectedTrajectoryAssignment]] = defaultdict(list)
    for assignment in normalized:
        if assignment.transport_file not in spec_by_path:
            raise ValueError("selected transport file is absent from source inventory")
        grouped[assignment.transport_file].append(assignment)
    shards = tuple(
        SelectedShardRead(
            source_file=spec_by_path[path],
            source_row_count=_positive_int(
                source_row_counts[path], label=f"row count for {path}"
            ),
            assignments=tuple(
                sorted(grouped[path], key=lambda item: item.transport_row_index)
            ),
        )
        for path in sorted(grouped)
    )
    return SelectedRowReadPlan(shards=shards, assignment_count=len(normalized))


def load_selected_rows_once(
    plan: SelectedRowReadPlan,
    *,
    row_iterator_factory: RowIteratorFactory,
    inspection_config: Mapping[str, Any],
) -> tuple[LoadedSelectedRow, ...]:
    """Read every selected shard once, sequentially, and fail on any identity drift."""
    if not isinstance(plan, SelectedRowReadPlan):
        raise TypeError("plan must be SelectedRowReadPlan")
    if not callable(row_iterator_factory):
        raise TypeError("row_iterator_factory must be callable")
    loaded: list[LoadedSelectedRow] = []
    for shard in plan.shards:
        wanted = {
            item.transport_row_index: item for item in shard.assignments
        }
        found: set[int] = set()
        expected_row_index = 0
        for row_index, row in row_iterator_factory(shard.source_file):
            if type(row_index) is not int or row_index != expected_row_index:
                raise ValueError(
                    f"row indices for {shard.source_file.transport_file} must be "
                    "contiguous from zero"
                )
            if not isinstance(row, Mapping):
                raise TypeError("decoded selected-source row must be a mapping")
            assignment = wanted.get(row_index)
            if assignment is not None:
                inspection = inspect_candidate(
                    row,
                    transport_file=shard.source_file.transport_file,
                    transport_row_index=row_index,
                    config=inspection_config,
                )
                if inspection.candidate is None:
                    reason = (
                        inspection.exclusion_reason.value
                        if inspection.exclusion_reason is not None
                        else "unknown"
                    )
                    raise ValueError(f"selected source row is no longer eligible: {reason}")
                candidate = inspection.candidate
                if (
                    candidate.source_id != assignment.source_id
                    or candidate.transport_file != assignment.transport_file
                    or candidate.transport_row_index != assignment.transport_row_index
                    or candidate.selection_sha256 != assignment.p0_selection_sha256
                    or candidate.decision_count != assignment.decision_count
                ):
                    raise ValueError("selected raw-row/P0 assignment identity drifted")
                observed_group = instruction_app_group_sha256(
                    _first_instruction_text(row),
                    normalized_app_labels=tuple(
                        normalize_app_label(value)
                        for value in candidate.normalized_app_labels
                    ),
                )
                if observed_group != assignment.instruction_app_group_sha256:
                    raise ValueError("selected instruction-app group identity drifted")
                found.add(row_index)
                loaded.append(
                    LoadedSelectedRow(
                        assignment=assignment,
                        source_file=shard.source_file,
                        row=dict(row),
                    )
                )
            expected_row_index += 1
        if expected_row_index != shard.source_row_count:
            raise ValueError(
                f"source row count drifted: {shard.source_file.transport_file}"
            )
        if found != set(wanted):
            raise ValueError(
                f"failed to load every selected row: {shard.source_file.transport_file}"
            )
    if len(loaded) != plan.assignment_count:
        raise RuntimeError("loaded selected-row denominator drifted")
    loaded.sort(key=lambda item: item.assignment.source_id)
    return tuple(loaded)


def build_selected_pilot(
    loaded: LoadedSelectedRow,
    *,
    provenance: SourceProvenance,
) -> SelectedPilot:
    """Convert one strict selected raw row with the canonical GUIOdyssey parser."""
    if not isinstance(loaded, LoadedSelectedRow):
        raise TypeError("loaded must be LoadedSelectedRow")
    if not isinstance(provenance, SourceProvenance):
        raise TypeError("provenance must be SourceProvenance")
    assignment = loaded.assignment
    pilot, images = build_pilot_manifest(
        loaded.row,
        row_index=assignment.transport_row_index,
        upstream_repo=provenance.upstream_repo,
        upstream_revision=provenance.upstream_revision,
        transport_repo=provenance.transport_repo,
        transport_revision=provenance.transport_revision,
        transport_file=assignment.transport_file,
        transport_file_sha256=loaded.source_file.sha256,
        hf_destination=provenance.derived_repo,
        grid_size=provenance.coordinate_grid_size,
    )
    source = pilot.get("source")
    trajectory = pilot.get("trajectory")
    if not isinstance(source, Mapping) or not isinstance(trajectory, Mapping):
        raise ValueError("selected pilot schema drifted")
    if source != {
        "upstream_repo": provenance.upstream_repo,
        "upstream_revision": provenance.upstream_revision,
        "transport_repo": provenance.transport_repo,
        "transport_revision": provenance.transport_revision,
        "transport_file": assignment.transport_file,
        "transport_file_sha256": loaded.source_file.sha256,
        "transport_row_index": assignment.transport_row_index,
        "license": "cc-by-4.0",
    }:
        raise ValueError("selected pilot source identity drifted")
    decisions = trajectory.get("decisions")
    events = trajectory.get("events")
    if (
        trajectory.get("source_id") != assignment.source_id
        or not isinstance(decisions, list)
        or len(decisions) != assignment.decision_count
        or not isinstance(events, list)
        or len(events) != assignment.decision_count
        or decisions[-1].get("decision_step_id")
        != assignment.terminal_decision_step_id
    ):
        raise ValueError("selected pilot trajectory/terminal identity drifted")
    return SelectedPilot(
        assignment=assignment,
        source_file=loaded.source_file,
        manifest=pilot,
        image_payloads=images,
    )


def build_validated_ocr_batch(
    image_payloads: Mapping[str, bytes],
    *,
    engine: Any,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    record_runner: Callable[..., Mapping[str, Any]],
) -> ValidatedOcrBatch:
    """Run the injected OCR engine and validate the exact reusable OCR records."""
    records, _ = generate_ocr_records(
        engine=engine,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
        image_payloads=image_payloads,
        record_runner=record_runner,
    )
    prepared = {
        path: validate_ocr_record(
            record,
            image_bytes=image_payloads[path],
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
        )
        for path, record in records.items()
    }
    return ValidatedOcrBatch(records_by_path=records, prepared_by_path=prepared)


def _low_fidelity_from_record(record: Mapping[str, Any]) -> LowFidelityEventV2:
    value = _mapping(record.get("low_fidelity_v2"), label="low-fidelity v2")
    return LowFidelityEventV2(
        step_id=value.get("step_id"),
        action_type=value.get("action_type"),
        action_argument=value.get("action_argument"),
        foreground_app=value.get("foreground_app"),
        screen_text_added=tuple(value.get("screen_text_added", ())),
        screen_text_removed=tuple(value.get("screen_text_removed", ())),
        screen_change=value.get("screen_change"),
        executor_result=value.get("executor_result"),
    )


def derive_utility_history_events(
    pilot: SelectedPilot,
    *,
    ocr_batch_builder: OcrBatchBuilder,
    derived_event_builder: DerivedEventBuilder = _build_derived_event,
) -> tuple[tuple[UtilityHistoryEvent, ...], tuple[dict[str, Any], ...]]:
    """Derive the full low/high-fidelity history without policy or utility access."""
    if not isinstance(pilot, SelectedPilot):
        raise TypeError("pilot must be SelectedPilot")
    if not callable(ocr_batch_builder) or not callable(derived_event_builder):
        raise TypeError("OCR and derived-event builders must be callable")
    trajectory = _mapping(pilot.manifest.get("trajectory"), label="pilot trajectory")
    raw_events = trajectory.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("pilot events must be an array")
    expected_ids = tuple(range(1, pilot.assignment.decision_count + 1))
    if tuple(event.get("step_id") for event in raw_events) != expected_ids:
        raise ValueError("pilot event prefix does not cover arbitrary terminal history")

    required_paths = {
        str(event[field])
        for event in raw_events
        for field in ("observation_before_path", "observation_after_path")
    }
    if not required_paths.issubset(pilot.image_payloads):
        raise ValueError("pilot image payloads are missing a required event image")
    required_images = {
        path: pilot.image_payloads[path] for path in sorted(required_paths)
    }
    batch = ocr_batch_builder(required_images)
    if not isinstance(batch, ValidatedOcrBatch):
        raise TypeError("OCR batch builder must return ValidatedOcrBatch")
    if set(batch.records_by_path) != required_paths:
        raise ValueError("OCR batch does not exactly cover required history images")

    derived_records: list[dict[str, Any]] = []
    history: list[UtilityHistoryEvent] = []
    for raw_event in raw_events:
        derived_value = derived_event_builder(
            raw_event,
            image_payloads=required_images,
            ocr_records_by_path=batch.records_by_path,
            prepared_by_path=batch.prepared_by_path,
        )
        if not isinstance(derived_value, Mapping):
            raise TypeError("derived-event builder must return a mapping")
        derived = dict(derived_value)
        step_id = raw_event["step_id"]
        high = _mapping(derived.get("high_fidelity_v2"), label="high-fidelity v2")
        after_path = raw_event["observation_after_path"]
        if (
            derived.get("step_id") != step_id
            or high.get("content_type") != "image"
            or high.get("image_role") != "post_action_state"
            or high.get("image_member_path") != after_path
        ):
            raise ValueError("derived event high-fidelity post-state identity drifted")
        low_fidelity = _low_fidelity_from_record(derived)
        history.append(
            UtilityHistoryEvent(
                event_step_id=step_id,
                low_fidelity_summary=low_fidelity,
                high_fidelity_observation_ref=after_path,
            )
        )
        derived_records.append(derived)
    return tuple(history), tuple(derived_records)


def build_trajectory_processor_substrate(
    pilot: SelectedPilot,
    *,
    query_states: Sequence[Mapping[str, Any]],
    ocr_batch_builder: OcrBatchBuilder,
    derived_event_builder: DerivedEventBuilder = _build_derived_event,
) -> TrajectoryProcessorSubstrate:
    """Bind exact v2 query records to one arbitrary-length selected trajectory."""
    if isinstance(query_states, (str, bytes, bytearray, Mapping)) or not query_states:
        raise TypeError("query states must be a non-empty ordered sequence")
    assignment = pilot.assignment
    history, derived = derive_utility_history_events(
        pilot,
        ocr_batch_builder=ocr_batch_builder,
        derived_event_builder=derived_event_builder,
    )
    trajectory = _mapping(pilot.manifest["trajectory"], label="pilot trajectory")
    decisions = {
        decision["decision_step_id"]: decision
        for decision in trajectory["decisions"]
    }
    normalized_queries: list[QuerySubstrateRef] = []
    query_kind_counts: dict[str, int] = {kind: 0 for kind in QUERY_KINDS}
    terminal_count = 0
    for raw_value in query_states:
        raw = _mapping(raw_value, label="query state")
        if set(raw) != QUERY_STATE_KEYS:
            raise ValueError("query state fields drifted")
        if (
            raw.get("trajectory_id") != assignment.trajectory_id
            or raw.get("source_id") != assignment.source_id
            or raw.get("role") != assignment.role
            or raw.get("candidate_capacity_stratum")
            != assignment.candidate_capacity_stratum
            or raw.get("processor_candidate_freeze_status")
            != "PENDING_SEPARATE_EXECUTION"
        ):
            raise ValueError("query state/assignment identity drifted")
        step = _positive_int(raw.get("decision_step_id"), label="decision step")
        decision = decisions.get(step)
        if decision is None:
            raise ValueError("query decision is absent from the selected raw row")
        current_equivalent = _positive_int(
            raw.get("current_equivalent_event_step_id"),
            label="current-equivalent event step",
        )
        candidates = _ordered_positive_ids(
            raw.get("initial_candidate_event_step_ids"),
            label="initial candidate event step ids",
        )
        expected_candidates = tuple(range(1, current_equivalent))[-16:]
        if (
            raw.get("initial_candidate_count") != len(candidates)
            or candidates != expected_candidates
        ):
            raise ValueError("query initial-candidate suffix identity drifted")
        current_path = decision.get("current_observation_path")
        if current_path != history[current_equivalent - 1].high_fidelity_observation_ref:
            raise ValueError("query current observation/post-state identity drifted")
        query = QuerySubstrateRef(
            state_id=raw.get("state_id"),
            query_kind=raw.get("query_kind"),
            decision_step_id=step,
            current_equivalent_event_step_id=current_equivalent,
            initial_candidate_event_step_ids=candidates,
            maximum_labeled_cardinality=raw.get("maximum_labeled_cardinality"),
            current_observation_ref=current_path,
        )
        expected_state_id = f"{assignment.source_id}:decision:{step:03d}"
        if query.state_id != expected_state_id:
            raise ValueError("query state ID/decision identity drifted")
        if query.maximum_labeled_cardinality != 2:
            raise ValueError("processor-freeze query cardinality must equal two")
        query_kind_counts[query.query_kind] += 1
        if query.query_kind == "stratum_anchor" and query.decision_step_id != (
            STRATUM_ANCHOR_DECISION_STEPS[assignment.candidate_capacity_stratum]
        ):
            raise ValueError("stratum-anchor decision step drifted")
        if query.query_kind == "terminal":
            terminal_count += 1
            if query.decision_step_id != assignment.decision_count + 1:
                raise ValueError("terminal query is not decision_count + 1")
        normalized_queries.append(query)
    if len(normalized_queries) != 2 or query_kind_counts != {
        "stratum_anchor": 1,
        "terminal": 1,
    }:
        raise ValueError(
            "trajectory substrate requires exactly one anchor and one terminal query"
        )
    if terminal_count != 1:
        raise RuntimeError("terminal query accounting drifted")
    normalized_queries.sort(key=lambda item: item.state_id)
    return TrajectoryProcessorSubstrate(
        assignment=assignment,
        transport_file_sha256=pilot.source_file.sha256,
        task_instruction=trajectory["instruction"],
        history_events=history,
        derived_event_records=derived,
        query_states=tuple(normalized_queries),
    )


__all__ = [
    "LoadedSelectedRow",
    "QuerySubstrateRef",
    "SelectedPilot",
    "SelectedRowReadPlan",
    "SelectedShardRead",
    "SelectedTrajectoryAssignment",
    "SourceProvenance",
    "TrajectoryProcessorSubstrate",
    "ValidatedOcrBatch",
    "build_selected_pilot",
    "build_selected_row_read_plan",
    "build_trajectory_processor_substrate",
    "build_validated_ocr_batch",
    "derive_utility_history_events",
    "load_selected_rows_once",
]
