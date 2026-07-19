"""Deterministic, prefix-safe artifacts for processor-only candidate freezing."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_processor_freeze import (
    FrozenQueryCandidateRecord,
    WORKER_COUNT,
    build_final_label_schedule,
    label_schedule_payload,
    operation_budget_payload,
)
from causalcache.set_utility_label_schedule import VariableNLabelSchedule
from causalcache.set_utility_processor_substrate import (
    QUERY_KINDS,
    SPLIT_ROLES,
    SelectedRowReadPlan,
    SelectedTrajectoryAssignment,
    TrajectoryProcessorSubstrate,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_PROTOCOL_ID = "causalcache_set_utility_processor_prefix_substrate_shards"
CANDIDATE_PROTOCOL_ID = "causalcache_set_utility_final_candidate_schedule"
GIT_SUMMARY_PROTOCOL_ID = "causalcache_set_utility_processor_freeze_git_summary"
ARTIFACT_FILENAME_TEMPLATE = "processor-substrate-worker-{worker_index:02d}.tar"
CANDIDATE_SCHEDULE_FILENAME = "processor-candidate-freeze-schedule.json"
EXPECTED_QUERY_COUNT_PER_TRAJECTORY = 2

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_SANITIZED_KEYS = frozenset(
    {
        "canonical_action",
        "decisions",
        "derived_event_records",
        "js",
        "kl",
        "label",
        "labels",
        "logit",
        "logits",
        "manifest",
        "outcome",
        "policy_output",
        "policy_outputs",
        "raw_row",
        "reward",
        "selected_pilot",
        "target_action",
        "terminal_outcome",
        "utilities",
        "utility",
        "validated_target_action",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _event_ids(value: Any, *, label: str) -> tuple[int, ...]:
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


def _reject_forbidden_sanitized_keys(
    value: Any,
    *,
    path: str,
    inspect_ocr_children: bool = True,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} JSON keys must be strings")
            folded = key.casefold()
            if folded in _FORBIDDEN_SANITIZED_KEYS:
                raise ValueError(f"{path} contains forbidden field {key!r}")
            if folded == "ocr_records_by_path" and not inspect_ocr_children:
                continue
            _reject_forbidden_sanitized_keys(
                child,
                path=f"{path}.{key}",
                inspect_ocr_children=inspect_ocr_children,
            )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_forbidden_sanitized_keys(
                child,
                path=f"{path}[{index}]",
                inspect_ocr_children=inspect_ocr_children,
            )


@dataclass(frozen=True)
class TrajectoryWorkItem:
    trajectory_id: str
    role: str
    observation_count: int
    transport_file: str
    transport_row_index: int

    def __post_init__(self) -> None:
        _canonical_text(self.trajectory_id, label="trajectory id")
        if self.role not in SPLIT_ROLES:
            raise ValueError(f"trajectory role must be one of {SPLIT_ROLES!r}")
        _positive_int(self.observation_count, label="observation count")
        _canonical_text(self.transport_file, label="transport file")
        if type(self.transport_row_index) is not int or self.transport_row_index < 0:
            raise ValueError("transport row index must be a non-negative integer")

    @classmethod
    def from_assignment(
        cls,
        assignment: SelectedTrajectoryAssignment,
    ) -> TrajectoryWorkItem:
        if not isinstance(assignment, SelectedTrajectoryAssignment):
            raise TypeError("work assignment is invalid")
        return cls(
            trajectory_id=assignment.trajectory_id,
            role=assignment.role,
            observation_count=assignment.decision_count + 1,
            transport_file=assignment.transport_file,
            transport_row_index=assignment.transport_row_index,
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessorWorkerShard:
    worker_index: int
    trajectories: tuple[TrajectoryWorkItem, ...]
    transport_files: tuple[str, ...]
    observation_count: int

    def __post_init__(self) -> None:
        if type(self.worker_index) is not int or not 0 <= self.worker_index < WORKER_COUNT:
            raise ValueError("worker index must be in [0, 3]")
        if not self.trajectories or any(
            not isinstance(item, TrajectoryWorkItem) for item in self.trajectories
        ):
            raise ValueError("worker shard must contain trajectory work items")
        expected_order = tuple(
            sorted(
                self.trajectories,
                key=lambda item: (item.transport_file, item.transport_row_index),
            )
        )
        if self.trajectories != expected_order:
            raise ValueError("worker trajectories must follow shard-and-row stream order")
        identities = tuple(
            (item.transport_file, item.transport_row_index)
            for item in self.trajectories
        )
        if len(identities) != len(set(identities)):
            raise ValueError("worker trajectory transport identities are duplicated")
        expected_files = tuple(sorted({item.transport_file for item in self.trajectories}))
        if self.transport_files != expected_files:
            raise ValueError("worker transport-file inventory drifted")
        if self.observation_count != sum(
            item.observation_count for item in self.trajectories
        ):
            raise ValueError("worker observation count drifted")

    @property
    def filename(self) -> str:
        return ARTIFACT_FILENAME_TEMPLATE.format(worker_index=self.worker_index)

    def to_payload(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "trajectories": [item.to_payload() for item in self.trajectories],
            "transport_files": list(self.transport_files),
            "worker_index": self.worker_index,
        }


@dataclass(frozen=True)
class ProcessorWorkerSchedule:
    workers: tuple[ProcessorWorkerShard, ...]
    selected_shard_count: int
    trajectory_count: int
    observation_count: int
    inventory_sha256: str
    execution_sha256: str

    def __post_init__(self) -> None:
        if len(self.workers) != WORKER_COUNT or tuple(
            worker.worker_index for worker in self.workers
        ) != tuple(range(WORKER_COUNT)):
            raise ValueError("processor schedule must contain four ordered workers")
        files = tuple(path for worker in self.workers for path in worker.transport_files)
        if len(files) != self.selected_shard_count or len(files) != len(set(files)):
            raise ValueError("processor schedule split or duplicated a transport shard")
        items = tuple(item for worker in self.workers for item in worker.trajectories)
        if len(items) != self.trajectory_count:
            raise ValueError("processor schedule trajectory count drifted")
        if len({item.trajectory_id for item in items}) != len(items):
            raise ValueError("processor schedule duplicated a trajectory")
        if self.observation_count != sum(item.observation_count for item in items):
            raise ValueError("processor schedule observation count drifted")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in (self.inventory_sha256, self.execution_sha256)
        ):
            raise ValueError("processor schedule SHA256 is invalid")


def build_processor_worker_schedule(
    read_plan: SelectedRowReadPlan,
) -> ProcessorWorkerSchedule:
    """Assign whole transport shards by LPT and preserve sequential row streaming."""
    if not isinstance(read_plan, SelectedRowReadPlan):
        raise TypeError("read_plan must be SelectedRowReadPlan")
    buckets: list[list[Any]] = [[] for _ in range(WORKER_COUNT)]
    loads = [0] * WORKER_COUNT
    weighted = tuple(
        (
            sum(item.decision_count + 1 for item in shard.assignments),
            shard.source_file.transport_file,
            shard,
        )
        for shard in read_plan.shards
    )
    for weight, _, shard in sorted(weighted, key=lambda value: (-value[0], value[1])):
        index = min(range(WORKER_COUNT), key=lambda worker: (loads[worker], worker))
        buckets[index].append(shard)
        loads[index] += weight
    workers: list[ProcessorWorkerShard] = []
    for index, shards in enumerate(buckets):
        ordered_shards = tuple(
            sorted(shards, key=lambda shard: shard.source_file.transport_file)
        )
        trajectories = tuple(
            TrajectoryWorkItem.from_assignment(assignment)
            for shard in ordered_shards
            for assignment in shard.assignments
        )
        workers.append(
            ProcessorWorkerShard(
                worker_index=index,
                trajectories=trajectories,
                transport_files=tuple(
                    shard.source_file.transport_file for shard in ordered_shards
                ),
                observation_count=loads[index],
            )
        )
    inventory_payload = {
        "schema_version": SCHEMA_VERSION,
        "trajectories": [
            TrajectoryWorkItem.from_assignment(assignment).to_payload()
            for shard in read_plan.shards
            for assignment in shard.assignments
        ],
    }
    inventory_sha = sha256_bytes(canonical_json_bytes(inventory_payload))
    execution_payload = {
        "inventory_sha256": inventory_sha,
        "schema_version": SCHEMA_VERSION,
        "workers": [worker.to_payload() for worker in workers],
    }
    return ProcessorWorkerSchedule(
        workers=tuple(workers),
        selected_shard_count=len(read_plan.shards),
        trajectory_count=read_plan.assignment_count,
        observation_count=sum(loads),
        inventory_sha256=inventory_sha,
        execution_sha256=sha256_bytes(canonical_json_bytes(execution_payload)),
    )


@dataclass(frozen=True)
class ProcessorQueryArtifactRecord:
    state_id: str
    trajectory_id: str
    role: str
    query_kind: str
    decision_step_id: int
    current_equivalent_event_step_id: int
    maximum_labeled_cardinality: int
    initial_candidate_event_step_ids: tuple[int, ...]
    task_instruction: str
    history_events: tuple[Mapping[str, Any], ...]
    current_observation_ref: str
    ocr_records_by_path: Mapping[str, Mapping[str, Any]]
    image_payloads: Mapping[str, bytes]

    def __post_init__(self) -> None:
        _canonical_text(self.state_id, label="query state id")
        _canonical_text(self.trajectory_id, label="query trajectory id")
        if self.role not in SPLIT_ROLES:
            raise ValueError(f"query role must be one of {SPLIT_ROLES!r}")
        if self.query_kind not in QUERY_KINDS:
            raise ValueError(f"query kind must be one of {QUERY_KINDS!r}")
        decision_step = _positive_int(self.decision_step_id, label="decision step")
        if self.current_equivalent_event_step_id != decision_step - 1:
            raise ValueError("query current-equivalent event must be decision step minus one")
        if self.maximum_labeled_cardinality != 2:
            raise ValueError("Execution-CF maximum labeled cardinality must be two")
        candidates = _event_ids(
            self.initial_candidate_event_step_ids,
            label="query initial candidates",
        )
        expected_candidates = tuple(range(1, self.current_equivalent_event_step_id))[-16:]
        if candidates != expected_candidates:
            raise ValueError("query initial candidates are not the newest non-current suffix")
        _canonical_text(self.task_instruction, label="task instruction")
        if not isinstance(self.history_events, tuple) or not self.history_events:
            raise ValueError("query history must be a non-empty tuple")
        expected_history_ids = tuple(range(1, self.current_equivalent_event_step_id + 1))
        observed_history_ids: list[int] = []
        post_ref_by_step: dict[int, str] = {}
        for event in self.history_events:
            if not isinstance(event, Mapping) or set(event) != {
                "event_step_id",
                "high_fidelity_observation_ref",
                "low_fidelity_summary",
                "observation_after_ref",
                "observation_before_ref",
            }:
                raise ValueError("prefix history event fields drifted")
            step = event["event_step_id"]
            observed_history_ids.append(step)
            low = LowFidelityEventV2.from_mapping(event["low_fidelity_summary"])
            if low.step_id != step or low.to_ordered_dict() != event["low_fidelity_summary"]:
                raise ValueError("prefix low-fidelity summary is not canonical")
            before_ref = _canonical_text(
                event["observation_before_ref"],
                label="prefix before-observation reference",
            )
            after_ref = _canonical_text(
                event["observation_after_ref"],
                label="prefix after-observation reference",
            )
            high_ref = _canonical_text(
                event["high_fidelity_observation_ref"],
                label="prefix high-fidelity observation reference",
            )
            if after_ref != high_ref:
                raise ValueError("prefix high-fidelity reference is not the after state")
            if step > 1 and before_ref != post_ref_by_step[step - 1]:
                raise ValueError("prefix observation chain is discontinuous")
            post_ref_by_step[step] = after_ref
        if tuple(observed_history_ids) != expected_history_ids:
            raise ValueError("query artifact history is not an exact causal prefix")
        current_ref = _canonical_text(
            self.current_observation_ref,
            label="query current observation reference",
        )
        if current_ref != post_ref_by_step[self.current_equivalent_event_step_id]:
            raise ValueError("current observation is not the current-equivalent post state")
        expected_images = {
            current_ref,
            *(post_ref_by_step[step] for step in candidates),
        }
        if (
            not isinstance(self.image_payloads, Mapping)
            or set(self.image_payloads) != expected_images
        ):
            raise ValueError("query image inventory is not exactly candidates plus current")
        for reference, payload in self.image_payloads.items():
            _canonical_text(reference, label="query image reference")
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("query image payloads must be non-empty bytes")
        if not isinstance(self.ocr_records_by_path, Mapping) or not self.ocr_records_by_path:
            raise ValueError("query OCR records must be a non-empty mapping")
        ocr_paths = tuple(self.ocr_records_by_path)
        if ocr_paths != tuple(sorted(ocr_paths)):
            raise ValueError("query OCR record paths must be sorted")
        expected_ocr_paths = {
            str(event[field])
            for event in self.history_events
            for field in ("observation_before_ref", "observation_after_ref")
        }
        if set(self.ocr_records_by_path) != expected_ocr_paths:
            raise ValueError("query OCR records are not exactly the observation prefix")
        for path, record in self.ocr_records_by_path.items():
            _canonical_text(path, label="query OCR path")
            if not isinstance(record, Mapping) or record.get("image_member_path") != path:
                raise ValueError("query OCR record path identity drifted")
            canonical_json_bytes(record)
        sanitized = self.to_payload(include_image_inventory=False)
        _reject_forbidden_sanitized_keys(
            sanitized,
            path="query_record",
            inspect_ocr_children=False,
        )

    def to_payload(self, *, include_image_inventory: bool) -> dict[str, Any]:
        value = {
            "current_equivalent_event_step_id": self.current_equivalent_event_step_id,
            "current_observation_ref": self.current_observation_ref,
            "decision_step_id": self.decision_step_id,
            "history_events": [dict(event) for event in self.history_events],
            "initial_candidate_event_step_ids": list(
                self.initial_candidate_event_step_ids
            ),
            "maximum_labeled_cardinality": self.maximum_labeled_cardinality,
            "ocr_records_by_path": {
                path: dict(record) for path, record in self.ocr_records_by_path.items()
            },
            "query_kind": self.query_kind,
            "role": self.role,
            "schema_version": SCHEMA_VERSION,
            "state_id": self.state_id,
            "task_instruction": self.task_instruction,
            "trajectory_id": self.trajectory_id,
        }
        if include_image_inventory:
            value["image_inventory"] = []
        return value


@dataclass(frozen=True)
class ProcessorTrajectoryArtifactRecord:
    trajectory_id: str
    role: str
    observation_count: int
    transport_file: str
    transport_row_index: int
    queries: tuple[ProcessorQueryArtifactRecord, ...]

    def __post_init__(self) -> None:
        item = TrajectoryWorkItem(
            trajectory_id=self.trajectory_id,
            role=self.role,
            observation_count=self.observation_count,
            transport_file=self.transport_file,
            transport_row_index=self.transport_row_index,
        )
        if len(self.queries) != EXPECTED_QUERY_COUNT_PER_TRAJECTORY or any(
            not isinstance(query, ProcessorQueryArtifactRecord) for query in self.queries
        ):
            raise ValueError("trajectory artifact requires exactly two valid queries")
        if tuple(query.decision_step_id for query in self.queries) != tuple(
            sorted(query.decision_step_id for query in self.queries)
        ):
            raise ValueError("trajectory artifact queries must be decision-step sorted")
        if {query.query_kind for query in self.queries} != set(QUERY_KINDS):
            raise ValueError("trajectory artifact requires anchor and terminal queries")
        if any(
            query.trajectory_id != item.trajectory_id or query.role != item.role
            for query in self.queries
        ):
            raise ValueError("trajectory/query artifact identity drifted")
        for query in self.queries:
            if query.state_id != (
                f"{self.trajectory_id}:decision:{query.decision_step_id:03d}"
            ):
                raise ValueError("trajectory/query state identity drifted")
        decision_count = self.observation_count - 1
        if 6 <= decision_count <= 9:
            expected_anchor_step = 6
        elif 10 <= decision_count <= 17:
            expected_anchor_step = 10
        elif decision_count >= 18:
            expected_anchor_step = 18
        else:
            raise ValueError("trajectory artifact is below the frozen decision minimum")
        anchor = next(
            query for query in self.queries if query.query_kind == "stratum_anchor"
        )
        if anchor.decision_step_id != expected_anchor_step:
            raise ValueError("trajectory anchor decision step drifted")
        terminal = next(query for query in self.queries if query.query_kind == "terminal")
        if terminal.decision_step_id != self.observation_count:
            raise ValueError("terminal decision step must equal trajectory observation count")


def build_prefix_safe_processor_artifact_record(
    substrate: TrajectoryProcessorSubstrate,
    *,
    image_payloads: Mapping[str, bytes],
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
) -> ProcessorTrajectoryArtifactRecord:
    """Project a full in-memory substrate into two independently prefix-safe queries."""
    if not isinstance(substrate, TrajectoryProcessorSubstrate):
        raise TypeError("substrate must be TrajectoryProcessorSubstrate")
    if not isinstance(image_payloads, Mapping) or not isinstance(
        ocr_records_by_path, Mapping
    ):
        raise TypeError("images and OCR records must be mappings")
    derived_by_step = {
        record["step_id"]: record for record in substrate.derived_event_records
    }
    queries: list[ProcessorQueryArtifactRecord] = []
    for query in sorted(substrate.query_states, key=lambda item: item.decision_step_id):
        prefix = substrate.utility_history_for_query(query)
        history = tuple(
            {
                "event_step_id": event.event_step_id,
                "high_fidelity_observation_ref": event.high_fidelity_observation_ref,
                "low_fidelity_summary": event.low_fidelity_summary.to_ordered_dict(),
                "observation_after_ref": derived_by_step[event.event_step_id][
                    "observation_after_path"
                ],
                "observation_before_ref": derived_by_step[event.event_step_id][
                    "observation_before_path"
                ],
            }
            for event in prefix
        )
        prefix_ocr_paths = tuple(
            sorted(
                {
                    str(derived_by_step[step][field])
                    for step in range(1, query.current_equivalent_event_step_id + 1)
                    for field in ("observation_before_path", "observation_after_path")
                }
            )
        )
        if not set(prefix_ocr_paths).issubset(ocr_records_by_path):
            raise ValueError("validated OCR inventory does not cover a query prefix")
        required_image_refs = tuple(
            sorted(
                {
                    query.current_observation_ref,
                    *(
                        substrate.history_events[step - 1].high_fidelity_observation_ref
                        for step in query.initial_candidate_event_step_ids
                    ),
                }
            )
        )
        if not set(required_image_refs).issubset(image_payloads):
            raise ValueError("image inventory does not cover query candidates and current")
        queries.append(
            ProcessorQueryArtifactRecord(
                state_id=query.state_id,
                trajectory_id=substrate.assignment.trajectory_id,
                role=substrate.assignment.role,
                query_kind=query.query_kind,
                decision_step_id=query.decision_step_id,
                current_equivalent_event_step_id=(
                    query.current_equivalent_event_step_id
                ),
                maximum_labeled_cardinality=query.maximum_labeled_cardinality,
                initial_candidate_event_step_ids=(
                    query.initial_candidate_event_step_ids
                ),
                task_instruction=substrate.task_instruction,
                history_events=history,
                current_observation_ref=query.current_observation_ref,
                ocr_records_by_path={
                    path: dict(ocr_records_by_path[path]) for path in prefix_ocr_paths
                },
                image_payloads={
                    path: image_payloads[path] for path in required_image_refs
                },
            )
        )
    assignment = substrate.assignment
    return ProcessorTrajectoryArtifactRecord(
        trajectory_id=assignment.trajectory_id,
        role=assignment.role,
        observation_count=assignment.decision_count + 1,
        transport_file=assignment.transport_file,
        transport_row_index=assignment.transport_row_index,
        queries=tuple(queries),
    )


@dataclass(frozen=True)
class ProcessorArtifactShardDescriptor:
    worker_index: int
    filename: str
    sha256: str
    byte_count: int
    trajectory_count: int
    query_count: int
    observation_count: int
    image_member_count: int
    image_byte_count: int
    content_inventory_sha256: str

    def __post_init__(self) -> None:
        if self.filename != ARTIFACT_FILENAME_TEMPLATE.format(
            worker_index=self.worker_index
        ):
            raise ValueError("artifact filename drifted")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in (self.sha256, self.content_inventory_sha256)
        ):
            raise ValueError("artifact SHA256 is invalid")
        for value, label in (
            (self.byte_count, "artifact byte count"),
            (self.trajectory_count, "artifact trajectory count"),
            (self.query_count, "artifact query count"),
            (self.observation_count, "artifact observation count"),
            (self.image_member_count, "artifact image member count"),
            (self.image_byte_count, "artifact image byte count"),
        ):
            _positive_int(value, label=label)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.mode = 0o444
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))


def _manifest_payload(worker: ProcessorWorkerShard) -> dict[str, Any]:
    return {
        "artifact_protocol_id": ARTIFACT_PROTOCOL_ID,
        "contains_generated_labels": False,
        "contains_policy_forward_outputs": False,
        "contains_raw_rows_or_selected_pilots": False,
        "schema_version": SCHEMA_VERSION,
        "worker": worker.to_payload(),
    }


def _query_payload_and_images(
    query: ProcessorQueryArtifactRecord,
    *,
    trajectory_index: int,
    query_index: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    images: list[dict[str, Any]] = []
    for image_index, reference in enumerate(sorted(query.image_payloads)):
        payload = query.image_payloads[reference]
        member = (
            f"trajectories/{trajectory_index:04d}/queries/{query_index:02d}/"
            f"images/{image_index:04d}.bin"
        )
        images.append(
            {
                "byte_count": len(payload),
                "member": member,
                "reference": reference,
                "sha256": sha256_bytes(payload),
            }
        )
    record = query.to_payload(include_image_inventory=True)
    record["image_inventory"] = images
    return record, tuple(images)


def _output_path(output_root: str | Path, filename: str) -> Path:
    root = Path(output_root)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("artifact output root must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("artifact output root must not be a symlink")
    return root / filename


def _open_partial(destination: Path, *, resume: bool) -> tuple[Path, Any]:
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists() or partial.is_symlink():
        if not resume:
            raise FileExistsError(f"partial artifact exists: {partial}")
        if partial.is_symlink() or not partial.is_file():
            raise ValueError("resumable partial artifact must be a regular file")
        partial.unlink()
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return partial, os.fdopen(descriptor, "wb")


def _publish_no_overwrite(partial: Path, destination: Path) -> None:
    os.chmod(partial, 0o444)
    try:
        os.link(partial, destination)
    except FileExistsError:
        raise FileExistsError(f"artifact destination exists: {destination}")
    finally:
        if partial.exists() and partial.is_file() and not partial.is_symlink():
            partial.unlink()
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def materialize_processor_artifact_shard(
    output_root: str | Path,
    *,
    worker: ProcessorWorkerShard,
    records: Iterable[ProcessorTrajectoryArtifactRecord],
    resume: bool = False,
) -> ProcessorArtifactShardDescriptor:
    """Stream one worker to an immutable tar; resume reuses complete shards only."""
    if not isinstance(worker, ProcessorWorkerShard):
        raise TypeError("worker must be ProcessorWorkerShard")
    destination = _output_path(output_root, worker.filename)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("existing artifact must be a regular file")
        if not resume:
            raise FileExistsError(f"artifact destination exists: {destination}")
        return inspect_processor_artifact_shard(destination, expected_worker=worker)
    partial, handle = _open_partial(destination, resume=resume)
    receipts: list[dict[str, Any]] = []
    image_count = 0
    image_bytes = 0
    try:
        with handle, tarfile.open(
            fileobj=handle,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive:
            _add_tar_bytes(
                archive,
                "shard-manifest.json",
                canonical_json_bytes(_manifest_payload(worker)),
            )
            iterator = iter(records)
            for trajectory_index, expected in enumerate(worker.trajectories):
                try:
                    record = next(iterator)
                except StopIteration as error:
                    raise ValueError("trajectory stream ended before worker schedule") from error
                if not isinstance(record, ProcessorTrajectoryArtifactRecord):
                    raise TypeError("trajectory stream contains an invalid record")
                if (
                    record.trajectory_id != expected.trajectory_id
                    or record.role != expected.role
                    or record.observation_count != expected.observation_count
                    or record.transport_file != expected.transport_file
                    or record.transport_row_index != expected.transport_row_index
                ):
                    raise ValueError("trajectory stream order or identity drifted")
                for query_index, query in enumerate(record.queries):
                    value, images = _query_payload_and_images(
                        query,
                        trajectory_index=trajectory_index,
                        query_index=query_index,
                    )
                    record_bytes = canonical_json_bytes(value)
                    record_member = (
                        f"trajectories/{trajectory_index:04d}/queries/"
                        f"{query_index:02d}/record.json"
                    )
                    _add_tar_bytes(archive, record_member, record_bytes)
                    for image in images:
                        payload = query.image_payloads[image["reference"]]
                        _add_tar_bytes(archive, image["member"], payload)
                        image_count += 1
                        image_bytes += len(payload)
                    receipts.append(
                        {
                            "image_inventory": list(images),
                            "query_kind": query.query_kind,
                            "record_sha256": sha256_bytes(record_bytes),
                            "state_id": query.state_id,
                            "trajectory_id": query.trajectory_id,
                        }
                    )
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("trajectory stream contains unexpected extra records")
            content_sha = sha256_bytes(
                canonical_json_bytes(
                    {"receipts": receipts, "schema_version": SCHEMA_VERSION}
                )
            )
            completion = {
                "content_inventory_sha256": content_sha,
                "image_byte_count": image_bytes,
                "image_member_count": image_count,
                "observation_count": worker.observation_count,
                "query_count": len(receipts),
                "schema_version": SCHEMA_VERSION,
                "trajectory_count": len(worker.trajectories),
                "worker_index": worker.worker_index,
            }
            _add_tar_bytes(
                archive,
                "completion.json",
                canonical_json_bytes(completion),
            )
        with partial.open("rb") as readable:
            os.fsync(readable.fileno())
        _publish_no_overwrite(partial, destination)
    except Exception:
        if not handle.closed:
            handle.close()
        if partial.exists() and partial.is_file() and not partial.is_symlink():
            partial.unlink()
        raise
    return inspect_processor_artifact_shard(destination, expected_worker=worker)


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _member_bytes(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    if not member.isfile():
        raise ValueError(f"tar member is not regular: {member.name}")
    if (
        member.mtime != 0
        or member.mode != 0o444
        or member.uid != 0
        or member.gid != 0
        or member.uname != ""
        or member.gname != ""
    ):
        raise ValueError(f"tar member metadata drifted: {member.name}")
    if maximum_bytes is not None and member.size > maximum_bytes:
        raise ValueError(f"tar member is too large: {member.name}")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"tar member could not be opened: {member.name}")
    payload = source.read()
    if len(payload) != member.size:
        raise ValueError(f"tar member ended early: {member.name}")
    return payload


def _worker_from_payload(value: Any) -> ProcessorWorkerShard:
    if not isinstance(value, Mapping) or set(value) != {
        "observation_count",
        "trajectories",
        "transport_files",
        "worker_index",
    }:
        raise ValueError("artifact worker fields drifted")
    raw_items = value["trajectories"]
    if not isinstance(raw_items, list):
        raise ValueError("artifact worker trajectories must be an array")
    items: list[TrajectoryWorkItem] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping) or set(raw) != {
            "observation_count",
            "role",
            "trajectory_id",
            "transport_file",
            "transport_row_index",
        }:
            raise ValueError("artifact trajectory work-item fields drifted")
        items.append(TrajectoryWorkItem(**raw))
    return ProcessorWorkerShard(
        worker_index=value["worker_index"],
        trajectories=tuple(items),
        transport_files=tuple(value["transport_files"]),
        observation_count=value["observation_count"],
    )


def _read_manifest(
    archive: tarfile.TarFile,
    *,
    expected_worker: ProcessorWorkerShard | None,
) -> ProcessorWorkerShard:
    member = archive.next()
    if member is None or member.name != "shard-manifest.json":
        raise ValueError("artifact is missing its leading manifest")
    value = _strict_json(
        _member_bytes(archive, member, maximum_bytes=64 * 1024 * 1024),
        label="artifact manifest",
    )
    if set(value) != {
        "artifact_protocol_id",
        "contains_generated_labels",
        "contains_policy_forward_outputs",
        "contains_raw_rows_or_selected_pilots",
        "schema_version",
        "worker",
    } or value != {
        "artifact_protocol_id": ARTIFACT_PROTOCOL_ID,
        "contains_generated_labels": False,
        "contains_policy_forward_outputs": False,
        "contains_raw_rows_or_selected_pilots": False,
        "schema_version": SCHEMA_VERSION,
        "worker": value["worker"],
    }:
        raise ValueError("artifact manifest contract drifted")
    worker = _worker_from_payload(value["worker"])
    if expected_worker is not None and worker != expected_worker:
        raise ValueError("artifact worker schedule drifted")
    return worker


def _query_from_payload(
    value: Any,
    *,
    expected: TrajectoryWorkItem,
) -> tuple[ProcessorQueryArtifactRecord, tuple[dict[str, Any], ...]]:
    expected_keys = {
        "current_equivalent_event_step_id",
        "current_observation_ref",
        "decision_step_id",
        "history_events",
        "image_inventory",
        "initial_candidate_event_step_ids",
        "maximum_labeled_cardinality",
        "ocr_records_by_path",
        "query_kind",
        "role",
        "schema_version",
        "state_id",
        "task_instruction",
        "trajectory_id",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("query artifact fields drifted")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("query artifact schema drifted")
    raw_images = value["image_inventory"]
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("query image inventory must be a non-empty array")
    images: list[dict[str, Any]] = []
    for raw in raw_images:
        if not isinstance(raw, Mapping) or set(raw) != {
            "byte_count",
            "member",
            "reference",
            "sha256",
        }:
            raise ValueError("query image binding fields drifted")
        _positive_int(raw["byte_count"], label="query image byte count")
        if not isinstance(raw["sha256"], str) or _SHA256.fullmatch(raw["sha256"]) is None:
            raise ValueError("query image SHA256 is invalid")
        images.append(dict(raw))
    references = tuple(item["reference"] for item in images)
    if references != tuple(sorted(references)) or len(references) != len(set(references)):
        raise ValueError("query image references are not sorted unique")
    query = ProcessorQueryArtifactRecord(
        state_id=value["state_id"],
        trajectory_id=value["trajectory_id"],
        role=value["role"],
        query_kind=value["query_kind"],
        decision_step_id=value["decision_step_id"],
        current_equivalent_event_step_id=value[
            "current_equivalent_event_step_id"
        ],
        maximum_labeled_cardinality=value["maximum_labeled_cardinality"],
        initial_candidate_event_step_ids=tuple(
            value["initial_candidate_event_step_ids"]
        ),
        task_instruction=value["task_instruction"],
        history_events=tuple(value["history_events"]),
        current_observation_ref=value["current_observation_ref"],
        ocr_records_by_path=value["ocr_records_by_path"],
        image_payloads={reference: b"placeholder" for reference in references},
    )
    if query.trajectory_id != expected.trajectory_id or query.role != expected.role:
        raise ValueError("query artifact/work-item identity drifted")
    return query, tuple(images)


def _read_query_header_and_images(
    archive: tarfile.TarFile,
    *,
    expected: TrajectoryWorkItem,
    trajectory_index: int,
    query_index: int,
    load_images: bool,
) -> tuple[ProcessorQueryArtifactRecord, dict[str, bytes], dict[str, Any]]:
    member_name = (
        f"trajectories/{trajectory_index:04d}/queries/{query_index:02d}/record.json"
    )
    member = archive.next()
    if member is None or member.name != member_name:
        raise ValueError("query artifact record order drifted")
    record_bytes = _member_bytes(archive, member, maximum_bytes=256 * 1024 * 1024)
    query, images = _query_from_payload(
        _strict_json(record_bytes, label="query artifact record"),
        expected=expected,
    )
    payloads: dict[str, bytes] = {}
    observed_image_bytes = 0
    for image_index, image in enumerate(images):
        expected_member = (
            f"trajectories/{trajectory_index:04d}/queries/{query_index:02d}/"
            f"images/{image_index:04d}.bin"
        )
        if image["member"] != expected_member:
            raise ValueError("query image member escaped its query prefix")
        image_member = archive.next()
        if image_member is None or image_member.name != image["member"]:
            raise ValueError("query image member order drifted")
        payload = _member_bytes(archive, image_member)
        if len(payload) != image["byte_count"] or sha256_bytes(payload) != image["sha256"]:
            raise ValueError("query image byte binding drifted")
        if load_images:
            payloads[image["reference"]] = payload
        observed_image_bytes += len(payload)
    receipt = {
        "image_inventory": list(images),
        "query_kind": query.query_kind,
        "record_sha256": sha256_bytes(record_bytes),
        "state_id": query.state_id,
        "trajectory_id": query.trajectory_id,
        "image_member_count": len(images),
        "image_byte_count": observed_image_bytes,
    }
    if load_images:
        query = ProcessorQueryArtifactRecord(
            state_id=query.state_id,
            trajectory_id=query.trajectory_id,
            role=query.role,
            query_kind=query.query_kind,
            decision_step_id=query.decision_step_id,
            current_equivalent_event_step_id=query.current_equivalent_event_step_id,
            maximum_labeled_cardinality=query.maximum_labeled_cardinality,
            initial_candidate_event_step_ids=query.initial_candidate_event_step_ids,
            task_instruction=query.task_instruction,
            history_events=query.history_events,
            current_observation_ref=query.current_observation_ref,
            ocr_records_by_path=query.ocr_records_by_path,
            image_payloads=payloads,
        )
    return query, payloads, receipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_processor_artifact_shard(
    path: str | Path,
    *,
    expected_worker: ProcessorWorkerShard | None = None,
) -> ProcessorArtifactShardDescriptor:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("processor artifact shard must be a regular file")
    receipts: list[dict[str, Any]] = []
    image_count = 0
    image_bytes = 0
    try:
        context = tarfile.open(source, mode="r:")
    except tarfile.TarError as error:
        raise ValueError("processor artifact is not a valid uncompressed tar") from error
    with context as archive:
        worker = _read_manifest(archive, expected_worker=expected_worker)
        for trajectory_index, expected in enumerate(worker.trajectories):
            query_kinds: set[str] = set()
            for query_index in range(EXPECTED_QUERY_COUNT_PER_TRAJECTORY):
                query, _, receipt = _read_query_header_and_images(
                    archive,
                    expected=expected,
                    trajectory_index=trajectory_index,
                    query_index=query_index,
                    load_images=False,
                )
                query_kinds.add(query.query_kind)
                image_count += receipt.pop("image_member_count")
                image_bytes += receipt.pop("image_byte_count")
                receipts.append(receipt)
            if query_kinds != set(QUERY_KINDS):
                raise ValueError("artifact trajectory lacks anchor or terminal query")
        member = archive.next()
        if member is None or member.name != "completion.json":
            raise ValueError("artifact is missing its completion receipt")
        completion = _strict_json(
            _member_bytes(archive, member, maximum_bytes=1024 * 1024),
            label="artifact completion",
        )
        content_sha = sha256_bytes(
            canonical_json_bytes({"receipts": receipts, "schema_version": SCHEMA_VERSION})
        )
        expected_completion = {
            "content_inventory_sha256": content_sha,
            "image_byte_count": image_bytes,
            "image_member_count": image_count,
            "observation_count": worker.observation_count,
            "query_count": len(receipts),
            "schema_version": SCHEMA_VERSION,
            "trajectory_count": len(worker.trajectories),
            "worker_index": worker.worker_index,
        }
        if completion != expected_completion:
            raise ValueError("artifact completion receipt drifted")
        if archive.next() is not None:
            raise ValueError("artifact contains an unreferenced trailing member")
    return ProcessorArtifactShardDescriptor(
        worker_index=worker.worker_index,
        filename=source.name,
        sha256=_sha256_file(source),
        byte_count=source.stat().st_size,
        trajectory_count=len(worker.trajectories),
        query_count=len(receipts),
        observation_count=worker.observation_count,
        image_member_count=image_count,
        image_byte_count=image_bytes,
        content_inventory_sha256=content_sha,
    )


def iter_processor_query_artifact_records(
    path: str | Path,
    *,
    expected_worker: ProcessorWorkerShard | None = None,
) -> Iterator[ProcessorQueryArtifactRecord]:
    """Yield one prefix-safe query at a time; no future query data shares the record."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("processor artifact shard must be a regular file")
    with tarfile.open(source, mode="r:") as archive:
        worker = _read_manifest(archive, expected_worker=expected_worker)
        receipts: list[dict[str, Any]] = []
        image_count = 0
        image_bytes = 0
        for trajectory_index, expected in enumerate(worker.trajectories):
            for query_index in range(EXPECTED_QUERY_COUNT_PER_TRAJECTORY):
                query, _, receipt = _read_query_header_and_images(
                    archive,
                    expected=expected,
                    trajectory_index=trajectory_index,
                    query_index=query_index,
                    load_images=True,
                )
                image_count += receipt.pop("image_member_count")
                image_bytes += receipt.pop("image_byte_count")
                receipts.append(receipt)
                yield query
        member = archive.next()
        if member is None or member.name != "completion.json":
            raise ValueError("artifact is missing its completion receipt")
        completion = _strict_json(
            _member_bytes(archive, member, maximum_bytes=1024 * 1024),
            label="artifact completion",
        )
        content_sha = sha256_bytes(
            canonical_json_bytes({"receipts": receipts, "schema_version": SCHEMA_VERSION})
        )
        expected_completion = {
            "content_inventory_sha256": content_sha,
            "image_byte_count": image_bytes,
            "image_member_count": image_count,
            "observation_count": worker.observation_count,
            "query_count": len(receipts),
            "schema_version": SCHEMA_VERSION,
            "trajectory_count": len(worker.trajectories),
            "worker_index": worker.worker_index,
        }
        if completion != expected_completion or archive.next() is not None:
            raise ValueError("artifact completion or member inventory drifted")


@dataclass(frozen=True)
class FinalCandidateSchedule:
    records: tuple[FrozenQueryCandidateRecord, ...]
    label_schedule: VariableNLabelSchedule
    candidate_inventory_sha256: str

    def __post_init__(self) -> None:
        if not self.records or any(
            not isinstance(record, FrozenQueryCandidateRecord) for record in self.records
        ):
            raise ValueError("final candidate schedule requires frozen query records")
        ids = tuple(record.state_id for record in self.records)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("final candidate records must be ID-sorted and unique")
        if not isinstance(self.label_schedule, VariableNLabelSchedule):
            raise TypeError("final label schedule is invalid")
        if _SHA256.fullmatch(self.candidate_inventory_sha256) is None:
            raise ValueError("candidate inventory SHA256 is invalid")


def build_final_candidate_schedule(
    records: Sequence[FrozenQueryCandidateRecord],
) -> FinalCandidateSchedule:
    if (
        isinstance(records, (str, bytes, bytearray, Mapping))
        or not isinstance(records, Sequence)
        or not records
    ):
        raise ValueError("candidate schedule requires a non-empty record sequence")
    if any(not isinstance(record, FrozenQueryCandidateRecord) for record in records):
        raise TypeError("candidate schedule contains an invalid frozen query record")
    ordered = tuple(sorted(records, key=lambda record: record.state_id))
    if len({record.state_id for record in ordered}) != len(ordered):
        raise ValueError("candidate schedule contains duplicate state IDs")
    inventory = {
        "records": [record.to_payload() for record in ordered],
        "schema_version": SCHEMA_VERSION,
    }
    return FinalCandidateSchedule(
        records=ordered,
        label_schedule=build_final_label_schedule(ordered),
        candidate_inventory_sha256=sha256_bytes(canonical_json_bytes(inventory)),
    )


def final_candidate_schedule_payload(schedule: FinalCandidateSchedule) -> dict[str, Any]:
    if not isinstance(schedule, FinalCandidateSchedule):
        raise TypeError("schedule must be FinalCandidateSchedule")
    return {
        "candidate_inventory": {
            "records": [record.to_payload() for record in schedule.records],
            "schema_version": SCHEMA_VERSION,
        },
        "candidate_inventory_sha256": schedule.candidate_inventory_sha256,
        "candidate_protocol_id": CANDIDATE_PROTOCOL_ID,
        "contains_generated_labels": False,
        "contains_policy_forward_outputs": False,
        "exact_label_schedule": label_schedule_payload(schedule.label_schedule),
        "schema_version": SCHEMA_VERSION,
    }


@dataclass(frozen=True)
class CandidateScheduleDescriptor:
    filename: str
    sha256: str
    byte_count: int
    state_count: int
    subset_forward_count: int
    total_model_operations: int
    candidate_inventory_sha256: str

    def __post_init__(self) -> None:
        if self.filename != CANDIDATE_SCHEDULE_FILENAME:
            raise ValueError("candidate schedule filename drifted")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in (self.sha256, self.candidate_inventory_sha256)
        ):
            raise ValueError("candidate schedule SHA256 is invalid")
        for value, label in (
            (self.byte_count, "candidate byte count"),
            (self.state_count, "candidate state count"),
            (self.subset_forward_count, "candidate subset count"),
            (self.total_model_operations, "candidate model-operation count"),
        ):
            _positive_int(value, label=label)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def materialize_final_candidate_schedule(
    output_root: str | Path,
    *,
    schedule: FinalCandidateSchedule,
    resume: bool = False,
) -> CandidateScheduleDescriptor:
    destination = _output_path(output_root, CANDIDATE_SCHEDULE_FILENAME)
    payload = canonical_pretty_json_bytes(final_candidate_schedule_payload(schedule))
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("existing candidate schedule must be a regular file")
        if not resume:
            raise FileExistsError(f"artifact destination exists: {destination}")
        if destination.read_bytes() != payload:
            raise ValueError("existing candidate schedule differs from requested schedule")
    else:
        partial, handle = _open_partial(destination, resume=resume)
        try:
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _publish_no_overwrite(partial, destination)
        except Exception:
            if not handle.closed:
                handle.close()
            if partial.exists() and partial.is_file() and not partial.is_symlink():
                partial.unlink()
            raise
    operations = schedule.label_schedule.operations
    return CandidateScheduleDescriptor(
        filename=destination.name,
        sha256=sha256_bytes(payload),
        byte_count=len(payload),
        state_count=len(schedule.records),
        subset_forward_count=operations.raw_label_rows,
        total_model_operations=operations.total_model_operations,
        candidate_inventory_sha256=schedule.candidate_inventory_sha256,
    )


def build_git_safe_processor_freeze_summary(
    *,
    worker_schedule: ProcessorWorkerSchedule,
    artifact_shards: Sequence[ProcessorArtifactShardDescriptor],
    candidate_schedule: FinalCandidateSchedule,
    candidate_artifact: CandidateScheduleDescriptor,
) -> dict[str, Any]:
    """Aggregate only counts and digests; raw text, OCR, images, and state IDs stay out."""
    if not isinstance(worker_schedule, ProcessorWorkerSchedule):
        raise TypeError("worker schedule is invalid")
    if (
        isinstance(artifact_shards, (str, bytes, bytearray, Mapping))
        or not isinstance(artifact_shards, Sequence)
        or len(artifact_shards) != WORKER_COUNT
        or any(
            not isinstance(item, ProcessorArtifactShardDescriptor)
            for item in artifact_shards
        )
    ):
        raise ValueError("Git summary requires exactly four artifact shards")
    shards = tuple(sorted(artifact_shards, key=lambda item: item.worker_index))
    if tuple(item.worker_index for item in shards) != tuple(range(WORKER_COUNT)):
        raise ValueError("Git summary worker indices drifted")
    for descriptor, worker in zip(shards, worker_schedule.workers, strict=True):
        if (
            descriptor.trajectory_count != len(worker.trajectories)
            or descriptor.observation_count != worker.observation_count
        ):
            raise ValueError("Git summary shard/schedule counts drifted")
    if candidate_artifact.candidate_inventory_sha256 != (
        candidate_schedule.candidate_inventory_sha256
    ):
        raise ValueError("Git summary candidate identity drifted")
    final_counts = Counter(
        len(record.candidate_context.candidate_event_step_ids)
        for record in candidate_schedule.records
    )
    drop_counts = Counter(
        len(record.candidate_context.dropped_event_step_ids)
        for record in candidate_schedule.records
    )
    role_counts = Counter(record.role for record in candidate_schedule.records)
    token_counts = [
        attempt.processor_input_token_count
        for record in candidate_schedule.records
        for attempt in record.candidate_context.attempts
    ]
    return {
        "artifact_shards": [item.to_payload() for item in shards],
        "candidate_artifact": candidate_artifact.to_payload(),
        "candidate_freeze": {
            "candidate_inventory_sha256": candidate_schedule.candidate_inventory_sha256,
            "drop_count_histogram": {
                str(key): value for key, value in sorted(drop_counts.items())
            },
            "final_candidate_count_histogram": {
                str(key): value for key, value in sorted(final_counts.items())
            },
            "label_execution_sha256": candidate_schedule.label_schedule.execution_sha256,
            "label_inventory_sha256": candidate_schedule.label_schedule.inventory_sha256,
            "label_subset_identity_sha256": (
                candidate_schedule.label_schedule.subset_identity_sha256
            ),
            "operation_budget": operation_budget_payload(
                candidate_schedule.label_schedule
            ),
            "processor_attempt_count": len(token_counts),
            "processor_token_length_maximum": max(token_counts),
            "processor_token_length_minimum": min(token_counts),
            "role_state_counts": {
                role: role_counts.get(role, 0) for role in SPLIT_ROLES
            },
            "state_count": len(candidate_schedule.records),
        },
        "contains_generated_labels": False,
        "contains_policy_forward_outputs": False,
        "contains_raw_text_ocr_or_images": False,
        "git_summary_protocol_id": GIT_SUMMARY_PROTOCOL_ID,
        "processor_workers": {
            "execution_sha256": worker_schedule.execution_sha256,
            "inventory_sha256": worker_schedule.inventory_sha256,
            "observation_count": worker_schedule.observation_count,
            "selected_shard_count": worker_schedule.selected_shard_count,
            "trajectory_count": worker_schedule.trajectory_count,
            "worker_count": WORKER_COUNT,
            "worker_observation_counts": [
                worker.observation_count for worker in worker_schedule.workers
            ],
            "worker_trajectory_counts": [
                len(worker.trajectories) for worker in worker_schedule.workers
            ],
        },
        "schema_version": SCHEMA_VERSION,
    }


__all__ = [
    "ARTIFACT_FILENAME_TEMPLATE",
    "ARTIFACT_PROTOCOL_ID",
    "CANDIDATE_SCHEDULE_FILENAME",
    "CandidateScheduleDescriptor",
    "FinalCandidateSchedule",
    "ProcessorArtifactShardDescriptor",
    "ProcessorQueryArtifactRecord",
    "ProcessorTrajectoryArtifactRecord",
    "ProcessorWorkerSchedule",
    "ProcessorWorkerShard",
    "TrajectoryWorkItem",
    "build_final_candidate_schedule",
    "build_git_safe_processor_freeze_summary",
    "build_prefix_safe_processor_artifact_record",
    "build_processor_worker_schedule",
    "canonical_json_bytes",
    "canonical_pretty_json_bytes",
    "final_candidate_schedule_payload",
    "inspect_processor_artifact_shard",
    "iter_processor_query_artifact_records",
    "materialize_final_candidate_schedule",
    "materialize_processor_artifact_shard",
    "sha256_bytes",
]
