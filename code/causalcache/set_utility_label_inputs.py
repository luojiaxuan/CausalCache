"""Firewall-safe processor inputs for set-utility label execution."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache import set_utility_processor_artifacts as _artifacts
from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_label_producer import (
    UtilityHistoryEvent,
    UtilityQuerySpec,
)
from causalcache.set_utility_processor_artifacts import (
    EXPECTED_QUERY_COUNT_PER_TRAJECTORY,
    ProcessorQueryArtifactRecord,
    ProcessorWorkerShard,
    TrajectoryWorkItem,
    canonical_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_processor_freeze import FrozenQueryCandidateRecord
from causalcache.set_utility_processor_substrate import (
    SPLIT_ROLES,
    SelectedTrajectoryAssignment,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAXIMUM_QUERY_RECORD_BYTES = 256 * 1024 * 1024


def _canonical_text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _open_hashed_artifact(
    path: Path,
) -> tuple[Any, os.stat_result, str]:
    if not path.is_absolute():
        raise ValueError("processor artifact shard path must be absolute")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("processor artifact shard must be a regular file")
        handle = os.fdopen(descriptor, "rb")
        descriptor = None
    except OSError as error:
        raise ValueError(
            "processor artifact shard could not be opened no-follow"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)

    digest = hashlib.sha256()
    byte_count = 0
    try:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        if byte_count != before.st_size or handle.seek(0, os.SEEK_SET) != 0:
            raise ValueError("processor artifact shard changed during hashing")
    except BaseException:
        handle.close()
        raise
    return handle, before, digest.hexdigest()


def _stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_stable_artifact(
    path: Path,
    *,
    handle: Any,
    before: os.stat_result,
) -> None:
    after = os.fstat(handle.fileno())
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError(
            "processor artifact shard path disappeared during its read"
        ) from error
    if (
        _stat_fingerprint(after) != _stat_fingerprint(before)
        or not stat.S_ISREG(current.st_mode)
        or _stat_fingerprint(current) != _stat_fingerprint(after)
    ):
        raise ValueError("processor artifact shard identity changed during its read")


def _anchor_decision_step(observation_count: int) -> int:
    decision_count = observation_count - 1
    if 6 <= decision_count <= 9:
        return 6
    if 10 <= decision_count <= 17:
        return 10
    if decision_count >= 18:
        return 18
    raise ValueError("processor trajectory is below the frozen decision minimum")


def _expected_query_identities(
    item: TrajectoryWorkItem,
) -> tuple[tuple[str, str, int], ...]:
    anchor = _anchor_decision_step(item.observation_count)
    terminal = item.observation_count
    return (
        (
            f"{item.trajectory_id}:decision:{anchor:03d}",
            "stratum_anchor",
            anchor,
        ),
        (
            f"{item.trajectory_id}:decision:{terminal:03d}",
            "terminal",
            terminal,
        ),
    )


def _expected_image_count(decision_step_id: int) -> int:
    return min(16, decision_step_id - 2) + 1


@dataclass(frozen=True)
class SelectedProcessorQuery:
    query: ProcessorQueryArtifactRecord
    processor_worker_index: int
    role_partition: str
    processor_artifact_sha256: str
    query_record_sha256: str
    query_record_member: str
    transport_file: str
    transport_row_index: int
    observation_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.query, ProcessorQueryArtifactRecord):
            raise TypeError("selected processor query is invalid")
        if (
            type(self.processor_worker_index) is not int
            or not 0 <= self.processor_worker_index < 4
        ):
            raise ValueError("processor worker index must be in [0, 3]")
        if self.role_partition not in SPLIT_ROLES:
            raise ValueError("selected query role partition is invalid")
        if self.query.role != self.role_partition:
            raise ValueError("selected query escaped its role partition")
        _sha256(
            self.processor_artifact_sha256,
            label="processor artifact SHA256",
        )
        _sha256(self.query_record_sha256, label="query record SHA256")
        _canonical_text(self.query_record_member, label="query record member")
        _canonical_text(self.transport_file, label="transport file")
        if type(self.transport_row_index) is not int or self.transport_row_index < 0:
            raise ValueError("transport row index must be non-negative")
        if type(self.observation_count) is not int or self.observation_count <= 0:
            raise ValueError("observation count must be positive")


@dataclass(frozen=True)
class LabelExecutionPartition:
    state_id: str
    worker_index: int
    role_partition: str

    def __post_init__(self) -> None:
        _canonical_text(self.state_id, label="label execution state id")
        if type(self.worker_index) is not int or not 0 <= self.worker_index < 4:
            raise ValueError("label execution worker index must be in [0, 3]")
        if self.role_partition not in SPLIT_ROLES:
            raise ValueError("label execution role partition is invalid")


@dataclass(frozen=True)
class JoinedUtilityQueryInput:
    query: UtilityQuerySpec
    processor_worker_index: int
    execution_worker_index: int
    role_partition: str
    processor_artifact_sha256: str
    query_record_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.query, UtilityQuerySpec):
            raise TypeError("joined utility query is invalid")
        for value, label in (
            (self.processor_worker_index, "processor worker index"),
            (self.execution_worker_index, "execution worker index"),
        ):
            if type(value) is not int or not 0 <= value < 4:
                raise ValueError(f"{label} must be in [0, 3]")
        if self.role_partition != self.query.split:
            raise ValueError("joined utility query escaped its role partition")
        if (
            self.processor_artifact_sha256
            != self.query.source_artifact_sha256
            or self.query_record_sha256 != self.query.slice_witness_sha256
        ):
            raise ValueError("joined utility query provenance drifted")
        _sha256(
            self.processor_artifact_sha256,
            label="processor artifact SHA256",
        )
        _sha256(self.query_record_sha256, label="query record SHA256")


def _normalized_allowlist(
    state_ids: Sequence[str],
    *,
    expected_worker: ProcessorWorkerShard,
) -> frozenset[str]:
    if (
        isinstance(state_ids, (str, bytes, bytearray, Mapping))
        or not isinstance(state_ids, Sequence)
        or not state_ids
    ):
        raise ValueError("train state allowlist must be a non-empty sequence")
    normalized = tuple(
        _canonical_text(state_id, label="train state allowlist item")
        for state_id in state_ids
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("train state allowlist contains a duplicate")
    known: dict[str, str] = {}
    for item in expected_worker.trajectories:
        for state_id, _, _ in _expected_query_identities(item):
            if state_id in known:
                raise ValueError("processor worker has a duplicate derived state id")
            known[state_id] = item.role
    unknown = set(normalized) - set(known)
    if unknown:
        raise ValueError("train state allowlist escaped the processor worker shard")
    if any(known[state_id] != "train" for state_id in normalized):
        raise ValueError("train state allowlist contains a tune or evaluation state")
    return frozenset(normalized)


def _require_member(
    archive: tarfile.TarFile,
    *,
    expected_name: str,
    label: str,
) -> tarfile.TarInfo:
    member = archive.next()
    if member is None or member.name != expected_name or not member.isfile():
        raise ValueError(f"{label} member order or type drifted")
    if (
        member.mtime != 0
        or member.mode != 0o444
        or member.uid != 0
        or member.gid != 0
        or member.uname != ""
        or member.gname != ""
    ):
        raise ValueError(f"{label} member metadata drifted")
    return member


def _skip_query_images(
    archive: tarfile.TarFile,
    *,
    trajectory_index: int,
    query_index: int,
    image_count: int,
) -> None:
    for image_index in range(image_count):
        member = _require_member(
            archive,
            expected_name=(
                f"trajectories/{trajectory_index:04d}/queries/{query_index:02d}/"
                f"images/{image_index:04d}.bin"
            ),
            label="skipped query image",
        )
        if member.size <= 0:
            raise ValueError("skipped query image must be non-empty")


def _read_selected_query(
    archive: tarfile.TarFile,
    *,
    member: tarfile.TarInfo,
    item: TrajectoryWorkItem,
    trajectory_index: int,
    query_index: int,
    expected_state_id: str,
    expected_query_kind: str,
    expected_decision_step_id: int,
    processor_worker_index: int,
    processor_artifact_sha256: str,
) -> SelectedProcessorQuery:
    record_bytes = _artifacts._member_bytes(
        archive,
        member,
        maximum_bytes=_MAXIMUM_QUERY_RECORD_BYTES,
    )
    value = _artifacts._strict_json(record_bytes, label="allowlisted query record")
    if record_bytes != canonical_json_bytes(value):
        raise ValueError("allowlisted query record is not canonical JSON")
    query, images = _artifacts._query_from_payload(value, expected=item)
    if (
        query.state_id != expected_state_id
        or query.query_kind != expected_query_kind
        or query.decision_step_id != expected_decision_step_id
        or query.role != "train"
    ):
        raise ValueError("allowlisted query differs from its header-derived identity")
    expected_count = _expected_image_count(expected_decision_step_id)
    if len(images) != expected_count:
        raise ValueError("allowlisted query image count drifted")
    payloads: dict[str, bytes] = {}
    for image_index, image in enumerate(images):
        expected_member = (
            f"trajectories/{trajectory_index:04d}/queries/{query_index:02d}/"
            f"images/{image_index:04d}.bin"
        )
        if image["member"] != expected_member:
            raise ValueError("allowlisted query image binding drifted")
        image_member = _require_member(
            archive,
            expected_name=expected_member,
            label="allowlisted query image",
        )
        payload = _artifacts._member_bytes(archive, image_member)
        if (
            len(payload) != image["byte_count"]
            or sha256_bytes(payload) != image["sha256"]
        ):
            raise ValueError("allowlisted query image bytes drifted")
        payloads[image["reference"]] = payload
    loaded = ProcessorQueryArtifactRecord(
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
    return SelectedProcessorQuery(
        query=loaded,
        processor_worker_index=processor_worker_index,
        role_partition=loaded.role,
        processor_artifact_sha256=processor_artifact_sha256,
        query_record_sha256=sha256_bytes(record_bytes),
        query_record_member=member.name,
        transport_file=item.transport_file,
        transport_row_index=item.transport_row_index,
        observation_count=item.observation_count,
    )


def read_allowlisted_train_processor_queries(
    path: str | Path,
    *,
    expected_worker: ProcessorWorkerShard,
    expected_artifact_sha256: str,
    state_ids: Sequence[str],
) -> tuple[SelectedProcessorQuery, ...]:
    """Read only explicitly allowlisted train records from one processor shard."""
    if not isinstance(expected_worker, ProcessorWorkerShard):
        raise TypeError("expected worker must be a ProcessorWorkerShard")
    expected_sha = _sha256(
        expected_artifact_sha256,
        label="expected processor artifact SHA256",
    )
    allowlist = _normalized_allowlist(state_ids, expected_worker=expected_worker)
    source = Path(path)
    handle, before, observed_sha = _open_hashed_artifact(source)
    if observed_sha != expected_sha:
        handle.close()
        raise ValueError("processor artifact shard SHA256 drifted")

    selected: list[SelectedProcessorQuery] = []
    with handle:
        with tarfile.open(fileobj=handle, mode="r:") as archive:
            worker = _artifacts._read_manifest(
                archive,
                expected_worker=expected_worker,
            )
            if worker.worker_index != expected_worker.worker_index:
                raise ValueError("processor artifact worker identity drifted")
            for trajectory_index, item in enumerate(worker.trajectories):
                identities = _expected_query_identities(item)
                if len(identities) != EXPECTED_QUERY_COUNT_PER_TRAJECTORY:
                    raise RuntimeError("derived query count drifted")
                for query_index, (
                    state_id,
                    query_kind,
                    decision_step_id,
                ) in enumerate(identities):
                    record_name = (
                        f"trajectories/{trajectory_index:04d}/queries/"
                        f"{query_index:02d}/record.json"
                    )
                    member = _require_member(
                        archive,
                        expected_name=record_name,
                        label="processor query record",
                    )
                    if state_id in allowlist:
                        selected.append(
                            _read_selected_query(
                                archive,
                                member=member,
                                item=item,
                                trajectory_index=trajectory_index,
                                query_index=query_index,
                                expected_state_id=state_id,
                                expected_query_kind=query_kind,
                                expected_decision_step_id=decision_step_id,
                                processor_worker_index=worker.worker_index,
                                processor_artifact_sha256=observed_sha,
                            )
                        )
                    else:
                        _skip_query_images(
                            archive,
                            trajectory_index=trajectory_index,
                            query_index=query_index,
                            image_count=_expected_image_count(decision_step_id),
                        )
            completion = _require_member(
                archive,
                expected_name="completion.json",
                label="processor artifact completion",
            )
            if completion.size <= 0 or archive.next() is not None:
                raise ValueError(
                    "processor artifact completion or trailing inventory drifted"
                )
        _validate_stable_artifact(source, handle=handle, before=before)
    if {item.query.state_id for item in selected} != set(allowlist):
        raise ValueError("allowlisted train query inventory is incomplete")
    return tuple(sorted(selected, key=lambda item: item.query.state_id))


def build_joined_utility_query_input(
    selected: SelectedProcessorQuery,
    *,
    candidate: FrozenQueryCandidateRecord,
    assignment: SelectedTrajectoryAssignment,
    execution: LabelExecutionPartition,
    request_manifest_sha256: str,
) -> JoinedUtilityQueryInput:
    """Strictly join one selected artifact record with frozen roster and candidates."""
    if not isinstance(selected, SelectedProcessorQuery):
        raise TypeError("selected query must be a SelectedProcessorQuery")
    if not isinstance(candidate, FrozenQueryCandidateRecord):
        raise TypeError("candidate must be a FrozenQueryCandidateRecord")
    if not isinstance(assignment, SelectedTrajectoryAssignment):
        raise TypeError("assignment must be a SelectedTrajectoryAssignment")
    if not isinstance(execution, LabelExecutionPartition):
        raise TypeError("execution must be a LabelExecutionPartition")
    request_sha = _sha256(request_manifest_sha256, label="request manifest SHA256")
    query = selected.query
    if (
        assignment.trajectory_id != query.trajectory_id
        or assignment.source_id != candidate.source_id
        or assignment.role != query.role
        or assignment.transport_file != selected.transport_file
        or assignment.transport_row_index != selected.transport_row_index
        or assignment.decision_count + 1 != selected.observation_count
    ):
        raise ValueError("processor query differs from its frozen assignment")
    if (
        candidate.state_id != query.state_id
        or candidate.trajectory_id != query.trajectory_id
        or candidate.role != query.role
        or candidate.query_kind != query.query_kind
        or candidate.decision_step_id != query.decision_step_id
        or candidate.maximum_labeled_cardinality
        != query.maximum_labeled_cardinality
        or candidate.candidate_context.initial_candidate_event_step_ids
        != query.initial_candidate_event_step_ids
    ):
        raise ValueError("processor query differs from its frozen candidate record")
    if (
        execution.state_id != query.state_id
        or execution.role_partition != query.role
    ):
        raise ValueError("processor query differs from its label execution partition")
    history = tuple(
        UtilityHistoryEvent(
            event_step_id=event["event_step_id"],
            low_fidelity_summary=LowFidelityEventV2.from_mapping(
                event["low_fidelity_summary"]
            ),
            high_fidelity_observation_ref=event[
                "high_fidelity_observation_ref"
            ],
        )
        for event in query.history_events
    )
    utility_query = UtilityQuerySpec(
        split=query.role,
        trajectory_id=query.trajectory_id,
        source_id=assignment.source_id,
        state_id=query.state_id,
        instruction_app_group_sha256=(
            assignment.instruction_app_group_sha256
        ),
        task_instruction=query.task_instruction,
        decision_step_id=query.decision_step_id,
        history_events=history,
        current_equivalent_event_step_id=query.current_equivalent_event_step_id,
        candidate_context=candidate.candidate_context,
        maximum_labeled_cardinality=query.maximum_labeled_cardinality,
        current_observation_ref=query.current_observation_ref,
        source_artifact_sha256=selected.processor_artifact_sha256,
        request_manifest_sha256=request_sha,
        slice_witness_sha256=selected.query_record_sha256,
    )
    return JoinedUtilityQueryInput(
        query=utility_query,
        processor_worker_index=selected.processor_worker_index,
        execution_worker_index=execution.worker_index,
        role_partition=execution.role_partition,
        processor_artifact_sha256=selected.processor_artifact_sha256,
        query_record_sha256=selected.query_record_sha256,
    )


__all__ = [
    "JoinedUtilityQueryInput",
    "LabelExecutionPartition",
    "SelectedProcessorQuery",
    "build_joined_utility_query_input",
    "read_allowlisted_train_processor_queries",
]
