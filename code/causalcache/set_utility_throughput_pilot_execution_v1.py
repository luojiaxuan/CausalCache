"""Fail-closed four-worker execution for the train-only throughput pilot.

The public orchestration boundary is dependency-injected so source/config and
filesystem-envelope validation can finish before semantic artifact decoding,
PyTorch import, or model construction.  Production dependencies are imported
lazily only after the durable no-retry attempt marker exists.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from causalcache.set_utility_throughput_pilot_contract_v1 import (
    CANDIDATE_SCHEDULE_INTEGER_KEY_PATH,
    CANONICAL_CONFIG_PATH,
    EXPECTED_STATE_IDS,
    WORKER_COUNT,
    TrainOnlyThroughputPilotSourceContractV1,
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    load_train_only_throughput_pilot_v1_contract,
    sha256_bytes,
)


SCHEMA_VERSION = "1.0.0"
EXECUTION_PROTOCOL_ID = "causalcache_set_utility_throughput_pilot_execution_v1"
WORKER_TERMINAL_PROTOCOL_ID = (
    "causalcache_set_utility_throughput_pilot_worker_terminal_v1"
)
AGGREGATE_PROTOCOL_ID = (
    "causalcache_set_utility_throughput_pilot_execution_aggregate_v1"
)
ATTEMPT_FILENAME = "attempt.json"
TERMINAL_FILENAME = "terminal.json"
AGGREGATE_FILENAME = "aggregate.json"
RESERVED_MEMORY_NUMERATOR = 80
RESERVED_MEMORY_DENOMINATOR = 100
MB2_SPEED_NUMERATOR = 95
MB2_SPEED_DENOMINATOR = 100

_SHA256 = re.compile(r"[0-9a-f]{64}")
_DEVICE = re.compile(r"cuda:[0-9]+")
_SAFE_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_PAIR_REQUIRED_KEYS = {
    "counts",
    "cross_variant_reference_action_equal",
    "expected_success_counts_for_12_states",
    "expected_success_counts_per_state",
    "failure_class",
    "latency_seconds",
    "metric_only",
    "peak_memory_bytes",
    "protocol_id",
    "retry_count",
    "schema_version",
    "state_id",
    "variant_order",
    "variants",
}
_PROJECTED_PAIR_KEYS = {
    "counts",
    "cross_variant_equal",
    "failure_class",
    "latency_seconds",
    "metric_only",
    "peak_memory_bytes",
    "retry_count",
    "state_id",
    "variant_order",
    "variants",
}
_ATTEMPT_KEYS = {
    "candidate_schedule_sha256",
    "config_sha256",
    "device_id",
    "freeze_manifest_sha256",
    "model_inventory_sha256",
    "processor_inventory_sha256",
    "protocol_id",
    "retry_count",
    "schema_version",
    "source_inventory_sha256",
    "state_ids",
    "status",
    "worker_index",
}
_TERMINAL_KEYS = {
    "attempt_sha256",
    "candidate_schedule_sha256",
    "config_sha256",
    "counts",
    "device_id",
    "device_total_memory_bytes",
    "failure_class",
    "freeze_manifest_sha256",
    "metric_only",
    "model_inventory_sha256",
    "pair_results",
    "processor_inventory_sha256",
    "protocol_id",
    "runtime_identity_sha256",
    "schema_version",
    "source_inventory_sha256",
    "state_ids",
    "status",
    "worker_index",
}
_ATTEMPT_TERMINAL_IDENTITY_KEYS = (
    "candidate_schedule_sha256",
    "config_sha256",
    "device_id",
    "freeze_manifest_sha256",
    "model_inventory_sha256",
    "processor_inventory_sha256",
    "source_inventory_sha256",
    "state_ids",
    "worker_index",
)


class PilotRuntimeProtocol(Protocol):
    def generate_reference_action(self, reference_input: object) -> object: ...

    def reference_actions_equal(self, left: object, right: object) -> bool: ...

    def teacher_force_reference(
        self,
        reference_inputs: tuple[object, ...],
        action_handles: tuple[object, ...],
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class ExecutionLaunchEnvelopeV1:
    repository_root: Path
    config_path: str
    processor_root: Path
    model_dir: Path
    output_root: Path
    worker_index: int
    device: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.repository_root, "repository root"),
            (self.processor_root, "processor root"),
            (self.model_dir, "model directory"),
            (self.output_root, "output root"),
        ):
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{label} must be an absolute Path")
        if self.config_path != CANONICAL_CONFIG_PATH:
            raise ValueError("execution config path must remain canonical")
        if type(self.worker_index) is not int or not 0 <= self.worker_index < WORKER_COUNT:
            raise ValueError("execution worker index must be in [0, 3]")
        if not isinstance(self.device, str) or _DEVICE.fullmatch(self.device) is None:
            raise ValueError("execution device must be one explicit CUDA device")


@dataclass(frozen=True, slots=True)
class ValidatedExecutionEnvelopeV1:
    launch: ExecutionLaunchEnvelopeV1
    freeze_manifest_bytes: bytes
    candidate_schedule_bytes: bytes
    processor_inventory_sha256: str
    model_inventory_sha256: str
    processor_artifact_sha256_by_worker: Mapping[int, str]

    def __post_init__(self) -> None:
        if not isinstance(self.launch, ExecutionLaunchEnvelopeV1):
            raise TypeError("validated launch envelope is invalid")
        if not self.freeze_manifest_bytes or not self.candidate_schedule_bytes:
            raise ValueError("validated envelope lost a bound JSON input")
        for value, label in (
            (self.processor_inventory_sha256, "processor inventory SHA256"),
            (self.model_inventory_sha256, "model inventory SHA256"),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{label} is invalid")
        if set(self.processor_artifact_sha256_by_worker) != set(range(WORKER_COUNT)):
            raise ValueError("processor artifact worker inventory is incomplete")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in self.processor_artifact_sha256_by_worker.values()
        ):
            raise ValueError("processor artifact worker SHA256 is invalid")


@dataclass(frozen=True, slots=True)
class LoadedPilotStateV1:
    state_id: str
    query: object
    joined_input: object

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError("loaded pilot state id is invalid")


@dataclass(frozen=True, slots=True)
class WorkerRuntimeBundleV1:
    runtime: PilotRuntimeProtocol
    reference_input_builder_for: Callable[[object], Callable[[object], object]]
    device_total_memory_bytes: int
    runtime_identity_sha256: str

    def __post_init__(self) -> None:
        if not callable(self.reference_input_builder_for):
            raise TypeError("runtime bundle input-builder factory is not callable")
        if type(self.device_total_memory_bytes) is not int or (
            self.device_total_memory_bytes <= 0
        ):
            raise ValueError("runtime bundle device memory must be positive")
        if _SHA256.fullmatch(self.runtime_identity_sha256) is None:
            raise ValueError("runtime identity SHA256 is invalid")


ContractLoader = Callable[..., TrainOnlyThroughputPilotSourceContractV1]
EnvelopeValidator = Callable[
    [ExecutionLaunchEnvelopeV1, TrainOnlyThroughputPilotSourceContractV1],
    ValidatedExecutionEnvelopeV1,
]
SemanticLoader = Callable[
    [ValidatedExecutionEnvelopeV1, TrainOnlyThroughputPilotSourceContractV1],
    tuple[LoadedPilotStateV1, ...],
]
RuntimeFactory = Callable[
    [ValidatedExecutionEnvelopeV1, TrainOnlyThroughputPilotSourceContractV1],
    WorkerRuntimeBundleV1,
]
PairRunner = Callable[..., Mapping[str, object]]


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
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


def canonical_candidate_schedule_producer_bytes_v2(
    schedule: Mapping[str, Any],
) -> bytes:
    """Reconstruct the producer's one typed integer-key mapping exactly."""
    root = dict(_mapping(schedule, label="candidate schedule"))
    exact = dict(
        _mapping(
            root.get(CANDIDATE_SCHEDULE_INTEGER_KEY_PATH[0]),
            label="candidate exact label schedule",
        )
    )
    counts = _mapping(
        exact.get(CANDIDATE_SCHEDULE_INTEGER_KEY_PATH[1]),
        label="candidate state counts by candidate count",
    )
    typed_counts: dict[int, Any] = {}
    for key, value in counts.items():
        if not isinstance(key, str) or re.fullmatch(r"[1-9][0-9]*", key) is None:
            raise ValueError(
                "candidate-count key must be canonical positive base-10 text"
            )
        integer_key = int(key)
        if integer_key in typed_counts:
            raise ValueError("candidate-count key conversion collided")
        typed_counts[integer_key] = value
    if not typed_counts:
        raise ValueError("candidate state-count mapping must not be empty")
    exact[CANDIDATE_SCHEDULE_INTEGER_KEY_PATH[1]] = typed_counts
    root[CANDIDATE_SCHEDULE_INTEGER_KEY_PATH[0]] = exact
    return canonical_pretty_json_bytes(root)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _safe_failure(error: BaseException) -> str:
    name = error.__class__.__name__
    return name if _SAFE_FAILURE.fullmatch(name) is not None else "UnexpectedException"


def _state_ids_for_worker(config: Mapping[str, Any], worker_index: int) -> tuple[str, ...]:
    pilot = _mapping(config.get("pilot"), label="pilot config")
    roster = _sequence(pilot.get("roster"), label="pilot roster")
    state_ids = tuple(
        str(_mapping(item, label="pilot roster item")["state_id"])
        for index, item in enumerate(roster)
        if index % WORKER_COUNT == worker_index
    )
    expected = tuple(EXPECTED_STATE_IDS[worker_index::WORKER_COUNT])
    mapping = _sequence(pilot.get("worker_mapping"), label="pilot worker mapping")
    configured = _mapping(mapping[worker_index], label="pilot worker")
    if (
        pilot.get("worker_count") != WORKER_COUNT
        or configured.get("worker_index") != worker_index
        or tuple(configured.get("state_ids", ())) != expected
        or state_ids != expected
        or len(state_ids) != 3
    ):
        raise ValueError("pilot worker mapping differs from state_ids[index::4]")
    return state_ids


def _regular_file_payload(path: Path, *, label: str) -> bytes:
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
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} could not be read safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
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
            raise ValueError("bound artifact must be a regular file")
        byte_count = 0
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        byte_count != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("bound artifact changed while being hashed")
    return byte_count, digest.hexdigest()


def _safe_tree_files(root: Path, *, label: str) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be one real directory")
    files: dict[str, Path] = {}
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if directory_path.is_symlink():
            raise ValueError(f"{label} must not contain a symlink directory")
        for name in directories:
            if (directory_path / name).is_symlink():
                raise ValueError(f"{label} must not contain a symlink directory")
        for name in filenames:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{label} must contain regular files only")
            files[relative] = path
    return files


def _validate_bound_tree(
    root: Path,
    inventory: Sequence[Any],
    *,
    label: str,
    hash_workers: int = 4,
) -> tuple[str, dict[str, bytes]]:
    records = tuple(_mapping(item, label=f"{label} record") for item in inventory)
    expected_paths = tuple(str(item.get("path")) for item in records)
    if (
        not records
        or expected_paths != tuple(sorted(expected_paths))
        or len(expected_paths) != len(set(expected_paths))
    ):
        raise ValueError(f"{label} inventory paths are invalid")
    observed = _safe_tree_files(root, label=label)
    if set(observed) != set(expected_paths):
        raise ValueError(f"{label} file inventory drifted")
    worker_count = min(max(1, hash_workers), len(records))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        hashes = tuple(pool.map(lambda item: _sha256_file(observed[str(item["path"])]), records))
    normalized: list[dict[str, Any]] = []
    retained: dict[str, bytes] = {}
    for record, (size, digest) in zip(records, hashes, strict=True):
        path = str(record["path"])
        expected_size = record.get("size_bytes", record.get("size"))
        if size != expected_size or digest != record.get("sha256"):
            raise ValueError(f"{label} file binding drifted: {path}")
        normalized.append({"path": path, "sha256": digest, "size_bytes": size})
        if path.endswith(".json") and size <= 32 * 1024 * 1024:
            retained[path] = _regular_file_payload(observed[path], label=path)
    return sha256_bytes(canonical_json_bytes(normalized)), retained


def _validate_output_root(launch: ExecutionLaunchEnvelopeV1) -> None:
    for source in (launch.repository_root, launch.processor_root, launch.model_dir):
        if launch.output_root == source or source in launch.output_root.parents:
            raise ValueError("execution output root must not mutate a bound input tree")
    launch.output_root.mkdir(parents=True, exist_ok=True)
    if launch.output_root.is_symlink() or not launch.output_root.is_dir():
        raise ValueError("execution output root must be one real directory")


def validate_execution_envelope_v1(
    launch: ExecutionLaunchEnvelopeV1,
    contract: TrainOnlyThroughputPilotSourceContractV1,
) -> ValidatedExecutionEnvelopeV1:
    """Hash every bound processor/model byte before semantic or torch access."""
    if not isinstance(launch, ExecutionLaunchEnvelopeV1):
        raise TypeError("launch envelope is invalid")
    if not isinstance(contract, TrainOnlyThroughputPilotSourceContractV1):
        raise TypeError("source contract is invalid")
    if launch.repository_root.resolve() != contract.repository_root.resolve():
        raise ValueError("launch repository differs from the validated source contract")
    _state_ids_for_worker(contract.data, launch.worker_index)
    _validate_output_root(launch)

    inputs = _mapping(contract.data.get("inputs"), label="contract inputs")
    processor = _mapping(
        inputs.get("processor_publication"), label="processor publication"
    )
    formal_inventory = _sequence(
        processor.get("formal_file_inventory"), label="formal file inventory"
    )
    processor_inventory_sha, retained = _validate_bound_tree(
        launch.processor_root,
        formal_inventory,
        label="processor root",
    )
    if processor_inventory_sha != processor.get("formal_file_inventory_sha256"):
        raise ValueError("processor root inventory SHA256 drifted")
    candidate_binding = _mapping(
        processor.get("candidate_schedule"), label="candidate schedule binding"
    )
    candidate_path = str(candidate_binding.get("path"))
    candidate_bytes = retained.get(candidate_path)
    if (
        candidate_bytes is None
        or len(candidate_bytes) != candidate_binding.get("byte_count")
        or sha256_bytes(candidate_bytes) != candidate_binding.get("sha256")
    ):
        raise ValueError("candidate schedule envelope drifted")

    model_binding = _mapping(
        inputs.get("model_snapshot_manifest"), label="model snapshot binding"
    )
    repository_manifest = _regular_file_payload(
        contract.repository_root / str(model_binding["path"]),
        label="repository model snapshot manifest",
    )
    if sha256_bytes(repository_manifest) != model_binding.get("sha256"):
        raise ValueError("repository model snapshot manifest drifted")
    model_manifest = _strict_json_object(repository_manifest, label="model manifest")
    model_records = _sequence(model_manifest.get("files"), label="model files")
    model_inventory_sha, _ = _validate_bound_tree(
        launch.model_dir,
        (
            *model_records,
            {
                "path": ".snapshot.json",
                "sha256": sha256_bytes(repository_manifest),
                "size": len(repository_manifest),
            },
        ),
        label="model directory",
    )
    snapshot_bytes = _regular_file_payload(
        launch.model_dir / ".snapshot.json", label="local snapshot manifest"
    )
    if snapshot_bytes != repository_manifest:
        raise ValueError("local model snapshot manifest differs from repository bytes")

    freeze_binding = _mapping(
        inputs.get("freeze_b_v2_manifest"), label="Freeze-B binding"
    )
    freeze_bytes = _regular_file_payload(
        contract.repository_root / str(freeze_binding["path"]),
        label="Freeze-B manifest",
    )
    if (
        len(freeze_bytes) != freeze_binding.get("byte_count")
        or sha256_bytes(freeze_bytes) != freeze_binding.get("sha256")
    ):
        raise ValueError("Freeze-B manifest envelope drifted")

    artifact_sha: dict[int, str] = {}
    formal_by_path = {str(record["path"]): record for record in formal_inventory}
    for worker_index in range(WORKER_COUNT):
        path = f"substrate/processor-substrate-worker-{worker_index:02d}.tar"
        record = _mapping(formal_by_path.get(path), label="processor tar binding")
        artifact_sha[worker_index] = str(record["sha256"])
    return ValidatedExecutionEnvelopeV1(
        launch=launch,
        freeze_manifest_bytes=freeze_bytes,
        candidate_schedule_bytes=candidate_bytes,
        processor_inventory_sha256=processor_inventory_sha,
        model_inventory_sha256=model_inventory_sha,
        processor_artifact_sha256_by_worker=artifact_sha,
    )


def validated_execution_from_authorized_projection_v1(
    launch: ExecutionLaunchEnvelopeV1,
    contract: TrainOnlyThroughputPilotSourceContractV1,
    projection: Mapping[str, Any],
) -> ValidatedExecutionEnvelopeV1:
    """Consume an already byte-verified formal envelope without rehashing 36 GB."""
    if not isinstance(projection, Mapping):
        raise TypeError("authorized execution projection must be a mapping")
    envelope = _mapping(projection.get("envelope"), label="authorized envelope")
    validation = _mapping(
        projection.get("validation"), label="authorized envelope validation"
    )
    if (
        projection.get("repository_root") != str(launch.repository_root)
        or projection.get("processor_root") != str(launch.processor_root)
        or projection.get("model_dir") != str(launch.model_dir)
        or projection.get("run_root") != str(launch.output_root)
        or validation.get("source_config_sha256") != contract.config_sha256
        or validation.get("worker_count") != WORKER_COUNT
    ):
        raise ValueError("authorized execution projection differs from launch/source")
    artifacts = _mapping(envelope.get("artifacts"), label="envelope artifacts")
    processor = _mapping(artifacts.get("processor"), label="envelope processor")
    model = _mapping(artifacts.get("model"), label="envelope model")
    processor_inventory = _sequence(
        processor.get("file_inventory"), label="envelope processor inventory"
    )
    processor_by_path = {
        str(_mapping(item, label="processor file")["path"]): item
        for item in processor_inventory
    }
    artifact_sha = {
        worker_index: str(
            _mapping(
                processor_by_path[
                    f"substrate/processor-substrate-worker-{worker_index:02d}.tar"
                ],
                label="processor tar",
            )["sha256"]
        )
        for worker_index in range(WORKER_COUNT)
    }
    inputs = _mapping(contract.data.get("inputs"), label="contract inputs")
    freeze_binding = _mapping(
        inputs.get("freeze_b_v2_manifest"), label="Freeze-B binding"
    )
    freeze_bytes = _regular_file_payload(
        contract.repository_root / str(freeze_binding["path"]),
        label="Freeze-B manifest",
    )
    candidate_binding = _mapping(
        _mapping(
            inputs.get("processor_publication"), label="processor publication"
        ).get("candidate_schedule"),
        label="candidate schedule binding",
    )
    candidate_bytes = _regular_file_payload(
        launch.processor_root / str(candidate_binding["path"]),
        label="candidate schedule",
    )
    if (
        sha256_bytes(freeze_bytes) != freeze_binding.get("sha256")
        or sha256_bytes(candidate_bytes) != candidate_binding.get("sha256")
    ):
        raise ValueError("authorized small bound input bytes drifted")
    return ValidatedExecutionEnvelopeV1(
        launch=launch,
        freeze_manifest_bytes=freeze_bytes,
        candidate_schedule_bytes=candidate_bytes,
        processor_inventory_sha256=str(processor["formal_inventory_sha256"]),
        model_inventory_sha256=str(model["file_inventory_sha256"]),
        processor_artifact_sha256_by_worker=artifact_sha,
    )


def _processor_worker_manifests(validated: ValidatedExecutionEnvelopeV1) -> tuple[Any, ...]:
    import tarfile

    from causalcache import set_utility_processor_artifacts as artifacts

    workers: list[Any] = []
    for worker_index in range(WORKER_COUNT):
        path = (
            validated.launch.processor_root
            / "substrate"
            / f"processor-substrate-worker-{worker_index:02d}.tar"
        )
        with tarfile.open(path, mode="r:") as archive:
            worker = artifacts._read_manifest(archive, expected_worker=None)
        if worker.worker_index != worker_index:
            raise ValueError("processor tar leading worker identity drifted")
        workers.append(worker)
    return tuple(workers)


def load_worker_semantic_inputs_v1(
    validated: ValidatedExecutionEnvelopeV1,
    contract: TrainOnlyThroughputPilotSourceContractV1,
) -> tuple[LoadedPilotStateV1, ...]:
    """Decode only the three preregistered train records assigned to this worker."""
    from causalcache.set_utility_label_inputs import (
        LabelExecutionPartition,
        build_joined_utility_query_input,
        read_allowlisted_train_processor_queries,
    )
    from causalcache.set_utility_processor_freeze import FrozenQueryCandidateRecord
    from causalcache.set_utility_processor_substrate import (
        SelectedTrajectoryAssignment,
    )

    launch = validated.launch
    state_ids = _state_ids_for_worker(contract.data, launch.worker_index)
    state_set = set(state_ids)
    source_ids = {state_id.split(":", 1)[0] for state_id in state_ids}

    freeze = _strict_json_object(
        validated.freeze_manifest_bytes, label="Freeze-B manifest"
    )
    if canonical_pretty_json_bytes(freeze) != validated.freeze_manifest_bytes:
        raise ValueError("Freeze-B manifest is not canonical pretty JSON")
    assignments: dict[str, Any] = {}
    for raw in _sequence(freeze.get("assignments"), label="Freeze-B assignments"):
        record = _mapping(raw, label="Freeze-B assignment")
        if record.get("source_id") in source_ids:
            assignment = SelectedTrajectoryAssignment.from_mapping(record)
            if assignment.source_id in assignments:
                raise ValueError("preregistered Freeze-B assignment is duplicated")
            assignments[assignment.source_id] = assignment
    if set(assignments) != source_ids or any(
        assignment.role != "train" for assignment in assignments.values()
    ):
        raise ValueError("preregistered train assignment inventory drifted")

    schedule = _strict_json_object(
        validated.candidate_schedule_bytes, label="candidate schedule"
    )
    if (
        canonical_candidate_schedule_producer_bytes_v2(schedule)
        != validated.candidate_schedule_bytes
    ):
        raise ValueError(
            "candidate schedule differs from producer-typed canonical JSON"
        )
    processor_config = _mapping(
        _mapping(contract.data["inputs"], label="contract inputs")[
            "processor_publication"
        ],
        label="processor config",
    )
    candidate_binding = _mapping(
        processor_config["candidate_schedule"], label="candidate binding"
    )
    inventory = _mapping(
        schedule.get("candidate_inventory"), label="candidate inventory"
    )
    raw_records = _sequence(inventory.get("records"), label="candidate records")
    if (
        schedule.get("candidate_inventory_sha256")
        != candidate_binding.get("candidate_inventory_sha256")
        or sha256_bytes(canonical_json_bytes(dict(inventory)))
        != candidate_binding.get("candidate_inventory_sha256")
    ):
        raise ValueError("candidate inventory SHA256 drifted")
    candidates: dict[str, Any] = {}
    for raw in raw_records:
        record = _mapping(raw, label="candidate record")
        state_id = record.get("state_id")
        if state_id in state_set:
            candidate = FrozenQueryCandidateRecord.from_payload(record)
            if candidate.state_id in candidates:
                raise ValueError("preregistered candidate record is duplicated")
            candidates[candidate.state_id] = candidate
    if set(candidates) != state_set or any(
        candidate.role != "train" or candidate.query_kind != "stratum_anchor"
        for candidate in candidates.values()
    ):
        raise ValueError("preregistered train candidate inventory drifted")

    workers = _processor_worker_manifests(validated)
    source_to_worker: dict[str, int] = {}
    for worker in workers:
        for trajectory in worker.trajectories:
            if trajectory.trajectory_id in source_ids:
                if trajectory.trajectory_id in source_to_worker:
                    raise ValueError("preregistered trajectory spans processor workers")
                source_to_worker[trajectory.trajectory_id] = worker.worker_index
    if set(source_to_worker) != source_ids:
        raise ValueError("preregistered trajectories are absent from processor tars")

    selected_by_state: dict[str, Any] = {}
    for processor_worker_index, worker in enumerate(workers):
        allowlist = tuple(
            state_id
            for state_id in state_ids
            if source_to_worker[state_id.split(":", 1)[0]] == processor_worker_index
        )
        if not allowlist:
            continue
        path = (
            launch.processor_root
            / "substrate"
            / f"processor-substrate-worker-{processor_worker_index:02d}.tar"
        )
        selected = read_allowlisted_train_processor_queries(
            path,
            expected_worker=worker,
            expected_artifact_sha256=(
                validated.processor_artifact_sha256_by_worker[
                    processor_worker_index
                ]
            ),
            state_ids=allowlist,
        )
        for item in selected:
            if item.query.state_id in selected_by_state:
                raise ValueError("preregistered processor query is duplicated")
            selected_by_state[item.query.state_id] = item
    if set(selected_by_state) != state_set:
        raise ValueError("preregistered processor query inventory is incomplete")

    request_sha = str(
        _mapping(contract.data["inputs"]["freeze_b_v2_manifest"], label="freeze")
        ["sha256"]
    )
    loaded: list[LoadedPilotStateV1] = []
    for state_id in state_ids:
        source_id = state_id.split(":", 1)[0]
        selected = selected_by_state[state_id]
        joined = build_joined_utility_query_input(
            selected,
            candidate=candidates[state_id],
            assignment=assignments[source_id],
            execution=LabelExecutionPartition(
                state_id=state_id,
                worker_index=launch.worker_index,
                role_partition="train",
            ),
            request_manifest_sha256=request_sha,
        )
        if joined.image_payloads is not selected.image_payloads:
            raise RuntimeError("strict join replaced the selected image payload mapping")
        loaded.append(
            LoadedPilotStateV1(
                state_id=state_id,
                query=joined.query,
                joined_input=joined,
            )
        )
    return tuple(loaded)


def build_production_runtime_v1(
    validated: ValidatedExecutionEnvelopeV1,
    contract: TrainOnlyThroughputPilotSourceContractV1,
) -> WorkerRuntimeBundleV1:
    """Lazily construct one real GUI-Owl v2.1 runtime on one explicit GPU."""
    from PIL import Image

    from causalcache.policy.gui_owl_v2_runtime import (
        FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from causalcache.policy.gui_owl_v2_1_throughput_runtime import (
        GUIOwlV21ThroughputRuntime,
    )
    from causalcache.set_utility_gui_owl_v2_1_throughput_adapter import (
        GUIOwlV21SetUtilityThroughputAdapter,
        build_gui_owl_v2_1_throughput_reference_input,
    )

    launch = validated.launch
    model_binding = _mapping(
        _mapping(contract.data["inputs"], label="contract inputs")[
            "model_snapshot_manifest"
        ],
        label="model binding",
    )
    runtime = GUIOwlV21ThroughputRuntime(
        model_dir=launch.model_dir,
        expected_snapshot_manifest=(
            contract.repository_root / str(model_binding["path"])
        ),
        device=launch.device,
        target_effective_visual_tokens_per_image=(
            FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ),
    )
    adapter = GUIOwlV21SetUtilityThroughputAdapter(runtime)
    properties = runtime.torch.cuda.get_device_properties(runtime.device)
    total_memory = getattr(properties, "total_memory", None)
    if type(total_memory) is not int or total_memory <= 0:
        raise RuntimeError("CUDA device total memory is unavailable")

    def decode_rgb(payload: bytes) -> Any:
        source = Image.open(io.BytesIO(payload))
        try:
            return source.convert("RGB")
        finally:
            source.close()

    def builder_for(joined: object) -> Callable[[object], object]:
        return lambda plan: build_gui_owl_v2_1_throughput_reference_input(
            joined,
            plan,
            image_decoder=decode_rgb,
        )

    identity = {
        "device": launch.device,
        "model_inventory_sha256": validated.model_inventory_sha256,
        "model_repo": model_binding["repo"],
        "model_revision": model_binding["revision"],
        "runtime_metadata_sha256": sha256_bytes(
            canonical_json_bytes(dict(runtime.metadata))
        ),
    }
    return WorkerRuntimeBundleV1(
        runtime=adapter,
        reference_input_builder_for=builder_for,
        device_total_memory_bytes=total_memory,
        runtime_identity_sha256=sha256_bytes(canonical_json_bytes(identity)),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent must be one real directory")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("exclusive write made no progress")
            view = view[count:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_atomic_no_clobber(path: Path, payload: bytes) -> None:
    partial = path.with_name(f".{path.name}.partial")
    _write_exclusive(partial, payload)
    try:
        os.link(partial, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        partial.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _attempt_payload(
    validated: ValidatedExecutionEnvelopeV1,
    contract: TrainOnlyThroughputPilotSourceContractV1,
) -> dict[str, Any]:
    launch = validated.launch
    return {
        "candidate_schedule_sha256": sha256_bytes(
            validated.candidate_schedule_bytes
        ),
        "config_sha256": contract.config_sha256,
        "device_id": launch.device,
        "freeze_manifest_sha256": sha256_bytes(validated.freeze_manifest_bytes),
        "model_inventory_sha256": validated.model_inventory_sha256,
        "processor_inventory_sha256": validated.processor_inventory_sha256,
        "protocol_id": EXECUTION_PROTOCOL_ID,
        "retry_count": 0,
        "schema_version": SCHEMA_VERSION,
        "source_inventory_sha256": contract.data["source"]["inventory_sha256"],
        "state_ids": list(_state_ids_for_worker(contract.data, launch.worker_index)),
        "status": "CLAIMED_NO_RETRY_THROUGHPUT_PILOT_WORKER_V1",
        "worker_index": launch.worker_index,
    }


def _project_pair_for_terminal(pair: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(pair, Mapping) or set(pair) != _PAIR_REQUIRED_KEYS:
        raise ValueError("paired pilot output fields drifted")
    projected = {
        "counts": pair["counts"],
        "cross_variant_equal": pair["cross_variant_reference_action_equal"],
        "failure_class": pair["failure_class"],
        "latency_seconds": pair["latency_seconds"],
        "metric_only": pair["metric_only"],
        "peak_memory_bytes": pair["peak_memory_bytes"],
        "retry_count": pair["retry_count"],
        "state_id": pair["state_id"],
        "variant_order": pair["variant_order"],
        "variants": pair["variants"],
    }
    if set(projected) != _PROJECTED_PAIR_KEYS:
        raise AssertionError("terminal pair projection drifted")
    canonical_json_bytes(projected)
    return projected


def _inflate_terminal_pair(pair: Mapping[str, object]) -> dict[str, object]:
    from causalcache.set_utility_throughput_pilot_pair_v1 import (
        EXPECTED_SUCCESS_COUNTS_FOR_12_STATES,
        EXPECTED_SUCCESS_COUNTS_PER_STATE,
        PROTOCOL_ID,
    )

    if not isinstance(pair, Mapping) or set(pair) != _PROJECTED_PAIR_KEYS:
        raise ValueError("terminal paired metric fields drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "state_id": pair["state_id"],
        "variant_order": pair["variant_order"],
        "variants": pair["variants"],
        "cross_variant_reference_action_equal": pair["cross_variant_equal"],
        "failure_class": pair["failure_class"],
        "retry_count": pair["retry_count"],
        "metric_only": pair["metric_only"],
        "latency_seconds": pair["latency_seconds"],
        "peak_memory_bytes": pair["peak_memory_bytes"],
        "counts": pair["counts"],
        "expected_success_counts_per_state": dict(
            EXPECTED_SUCCESS_COUNTS_PER_STATE
        ),
        "expected_success_counts_for_12_states": dict(
            EXPECTED_SUCCESS_COUNTS_FOR_12_STATES
        ),
    }


def run_throughput_pilot_worker_v1(
    launch: ExecutionLaunchEnvelopeV1,
    *,
    contract_loader: ContractLoader = load_train_only_throughput_pilot_v1_contract,
    envelope_validator: EnvelopeValidator = validate_execution_envelope_v1,
    semantic_loader: SemanticLoader = load_worker_semantic_inputs_v1,
    runtime_factory: RuntimeFactory = build_production_runtime_v1,
    pair_runner: PairRunner | None = None,
) -> dict[str, Any]:
    """Validate, claim once, execute three states, and publish one terminal."""
    contract = contract_loader(
        repository_root=launch.repository_root,
        config_path=launch.config_path,
    )
    validated = envelope_validator(launch, contract)
    worker_root = launch.output_root / "workers" / f"worker-{launch.worker_index:02d}"
    attempt = _attempt_payload(validated, contract)
    attempt_bytes = canonical_pretty_json_bytes(attempt)
    _write_exclusive(worker_root / ATTEMPT_FILENAME, attempt_bytes)
    attempt_sha = sha256_bytes(attempt_bytes)

    pair_results: list[dict[str, object]] = []
    runtime_identity_sha: str | None = None
    device_total_memory = 0
    failure_class: str | None = None
    try:
        states = semantic_loader(validated, contract)
        expected_states = _state_ids_for_worker(contract.data, launch.worker_index)
        if tuple(item.state_id for item in states) != expected_states:
            raise ValueError("semantic loader returned a non-preregistered state roster")
        bundle = runtime_factory(validated, contract)
        runtime_identity_sha = bundle.runtime_identity_sha256
        device_total_memory = bundle.device_total_memory_bytes
        selected_pair_runner = pair_runner
        if selected_pair_runner is None:
            from causalcache.set_utility_throughput_pilot_pair_v1 import (
                run_train_only_set_utility_throughput_pilot_pair_v1,
            )

            selected_pair_runner = run_train_only_set_utility_throughput_pilot_pair_v1
        for state in states:
            raw = selected_pair_runner(
                state.query,
                reference_input_builder=(
                    bundle.reference_input_builder_for(state.joined_input)
                ),
                runtime=bundle.runtime,
            )
            projected = _project_pair_for_terminal(raw)
            if projected["state_id"] != state.state_id:
                raise ValueError("paired pilot result state identity drifted")
            pair_results.append(projected)
    except BaseException as error:
        failure_class = _safe_failure(error)

    completed = failure_class is None and len(pair_results) == 3
    terminal = {
        "attempt_sha256": attempt_sha,
        "candidate_schedule_sha256": sha256_bytes(
            validated.candidate_schedule_bytes
        ),
        "config_sha256": contract.config_sha256,
        "counts": {
            "pair_completed_count": len(pair_results),
            "pair_expected_count": 3,
            "retry_count": 0,
        },
        "device_id": launch.device,
        "device_total_memory_bytes": device_total_memory,
        "failure_class": failure_class,
        "freeze_manifest_sha256": sha256_bytes(validated.freeze_manifest_bytes),
        "metric_only": True,
        "model_inventory_sha256": validated.model_inventory_sha256,
        "pair_results": pair_results,
        "processor_inventory_sha256": validated.processor_inventory_sha256,
        "protocol_id": WORKER_TERMINAL_PROTOCOL_ID,
        "runtime_identity_sha256": runtime_identity_sha,
        "schema_version": SCHEMA_VERSION,
        "source_inventory_sha256": contract.data["source"]["inventory_sha256"],
        "state_ids": list(_state_ids_for_worker(contract.data, launch.worker_index)),
        "status": (
            "COMPLETED_THROUGHPUT_PILOT_WORKER_V1"
            if completed
            else "FAILED_THROUGHPUT_PILOT_WORKER_V1"
        ),
        "worker_index": launch.worker_index,
    }
    terminal_bytes = canonical_pretty_json_bytes(terminal)
    _write_atomic_no_clobber(worker_root / TERMINAL_FILENAME, terminal_bytes)
    return terminal


def _variant(pair: Mapping[str, object], microbatch: int) -> Mapping[str, object]:
    variants = _mapping(pair.get("variants"), label="paired variants")
    variant = variants.get(str(microbatch))
    if not isinstance(variant, Mapping):
        raise ValueError(f"microbatch {microbatch} result is missing")
    return variant


def _optional_variant(
    pair: Mapping[str, object], microbatch: int
) -> Mapping[str, object] | None:
    variants = _mapping(pair.get("variants"), label="paired variants")
    variant = variants.get(str(microbatch))
    if variant is None:
        return None
    if not isinstance(variant, Mapping):
        raise ValueError(f"microbatch {microbatch} result is invalid")
    return variant


def _selection_no_go(reason: str) -> dict[str, Any]:
    return {
        "outcome": "NO_GO",
        "reason": reason,
        "selected_reference_teacher_microbatch_size": None,
    }


def select_reference_teacher_microbatch_v1(
    pairs: Sequence[Mapping[str, object]],
    *,
    worker_index_by_state: Mapping[str, int],
    device_total_memory_bytes_by_worker: Mapping[int, int],
) -> dict[str, Any]:
    """Apply the preregistered memory/failure/speed decision without retries."""
    if (
        isinstance(pairs, (str, bytes, bytearray, Mapping))
        or not isinstance(pairs, Sequence)
        or len(pairs) != len(EXPECTED_STATE_IDS)
    ):
        raise ValueError("microbatch selection requires the exact 12-state roster")
    if tuple(pair.get("state_id") for pair in pairs) != EXPECTED_STATE_IDS:
        raise ValueError("microbatch selection state order drifted")
    if set(worker_index_by_state) != set(EXPECTED_STATE_IDS):
        raise ValueError("microbatch selection worker map is incomplete")
    if set(device_total_memory_bytes_by_worker) != set(range(WORKER_COUNT)) or any(
        type(value) is not int or value <= 0
        for value in device_total_memory_bytes_by_worker.values()
    ):
        raise ValueError("microbatch selection device-memory map is invalid")

    teacher_totals = {1: [], 2: []}
    mb2_failures: list[str] = []
    cross_equal: list[bool] = []
    for pair in pairs:
        state_id = str(pair["state_id"])
        worker_index = worker_index_by_state[state_id]
        if type(worker_index) is not int or not 0 <= worker_index < WORKER_COUNT:
            raise ValueError("microbatch selection worker index is invalid")
        total_memory = device_total_memory_bytes_by_worker[worker_index]
        memory_ceiling = (
            total_memory * RESERVED_MEMORY_NUMERATOR // RESERVED_MEMORY_DENOMINATOR
        )
        mb1 = _optional_variant(pair, 1)
        if mb1 is None:
            if pair.get("failure_class") != "MICROBATCH_1__INVALID_CORE_PROJECTION":
                raise ValueError("microbatch one result is missing without its failure")
            return _selection_no_go("MICROBATCH_1_INVALID_CORE_PROJECTION")
        mb1_peak = _mapping(
            mb1.get("peak_memory_bytes"), label="variant peak memory"
        ).get("full_call_cuda_reserved_max")
        if type(mb1_peak) is not int or mb1_peak < 0:
            raise ValueError("variant reserved peak memory is invalid")
        if mb1_peak > memory_ceiling:
            return _selection_no_go(
                "MICROBATCH_1_RESERVED_PEAK_EXCEEDS_80_PERCENT"
            )
        if mb1.get("failure_class") is not None:
            return _selection_no_go("MICROBATCH_1_FAILURE")
        mb2 = _optional_variant(pair, 2)
        if mb2 is None:
            if pair.get("failure_class") != "MICROBATCH_2__INVALID_CORE_PROJECTION":
                raise ValueError("microbatch two result is missing without its failure")
            return _selection_no_go("MICROBATCH_2_INVALID_CORE_PROJECTION")
        for microbatch, variant in ((1, mb1), (2, mb2)):
            peak = _mapping(
                variant.get("peak_memory_bytes"), label="variant peak memory"
            ).get("full_call_cuda_reserved_max")
            if type(peak) is not int or peak < 0:
                raise ValueError("variant reserved peak memory is invalid")
            if peak > memory_ceiling:
                return _selection_no_go(
                    f"MICROBATCH_{microbatch}_RESERVED_PEAK_EXCEEDS_80_PERCENT"
                )
            latency = _mapping(
                variant.get("latency_seconds"), label="variant latency"
            ).get("reference_teacher_forward_end_to_end_wall_total")
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(float(latency))
                or float(latency) < 0.0
            ):
                raise ValueError("variant teacher latency is invalid")
            teacher_totals[microbatch].append(float(latency))
        mb2_failure = mb2.get("failure_class")
        if mb2_failure is not None:
            if not isinstance(mb2_failure, str):
                raise ValueError("microbatch two failure class is invalid")
            mb2_failures.append(mb2_failure)
        cross_equal.append(pair.get("cross_variant_equal") is True)

    mb1_teacher = math.fsum(teacher_totals[1])
    mb2_teacher = math.fsum(teacher_totals[2])
    metrics = {
        "mb1_reference_teacher_wall_seconds": mb1_teacher,
        "mb2_reference_teacher_wall_seconds": mb2_teacher,
        "speed_threshold_denominator": MB2_SPEED_DENOMINATOR,
        "speed_threshold_numerator": MB2_SPEED_NUMERATOR,
    }
    if mb2_failures:
        if all(cross_equal) and all(
            value.startswith("REFERENCE_TEACHER_FORWARD:")
            for value in mb2_failures
        ):
            return {
                **metrics,
                "outcome": "GO_MB1",
                "reason": "MICROBATCH_2_TEACHER_FAILURE_FALLBACK",
                "selected_reference_teacher_microbatch_size": 1,
            }
        return {**metrics, **_selection_no_go("MICROBATCH_2_NONFALLBACK_FAILURE")}
    if not all(cross_equal):
        return {**metrics, **_selection_no_go("CROSS_VARIANT_MISMATCH")}
    mb2_fast_enough = (
        mb2_teacher * MB2_SPEED_DENOMINATOR
        <= mb1_teacher * MB2_SPEED_NUMERATOR
    )
    selected = 2 if mb2_fast_enough else 1
    return {
        **metrics,
        "outcome": f"GO_MB{selected}",
        "reason": (
            "MICROBATCH_2_AT_LEAST_FIVE_PERCENT_FASTER"
            if selected == 2
            else "MICROBATCH_2_BELOW_FIVE_PERCENT_SPEEDUP"
        ),
        "selected_reference_teacher_microbatch_size": selected,
    }


def _read_canonical_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload = _regular_file_payload(path, label=label)
    value = _strict_json_object(payload, label=label)
    if payload != canonical_pretty_json_bytes(value):
        raise ValueError(f"{label} is not canonical pretty JSON")
    return value, payload


def aggregate_throughput_pilot_workers_v1(
    *,
    repository_root: str | Path,
    output_root: str | Path,
    config_path: str = CANONICAL_CONFIG_PATH,
    contract_loader: ContractLoader = load_train_only_throughput_pilot_v1_contract,
) -> dict[str, Any]:
    """Read exactly four durable terminals, decide, and atomically publish once."""
    root = Path(repository_root).resolve()
    output = Path(output_root)
    if not output.is_absolute():
        raise ValueError("aggregate output root must be absolute")
    contract = contract_loader(repository_root=root, config_path=config_path)
    worker_container = output / "workers"
    if not worker_container.is_dir() or worker_container.is_symlink():
        raise ValueError("worker output container is missing or unsafe")
    expected_worker_dirs = {f"worker-{index:02d}" for index in range(WORKER_COUNT)}
    if {path.name for path in worker_container.iterdir()} != expected_worker_dirs:
        raise ValueError("worker output directory inventory drifted")

    terminals: list[dict[str, Any]] = []
    terminal_hashes: list[dict[str, Any]] = []
    memory_by_worker: dict[int, int] = {}
    worker_by_state: dict[str, int] = {}
    common: dict[str, Any] | None = None
    common_runtime_identity: str | None = None
    for worker_index in range(WORKER_COUNT):
        worker_root = worker_container / f"worker-{worker_index:02d}"
        if worker_root.is_symlink() or {path.name for path in worker_root.iterdir()} != {
            ATTEMPT_FILENAME,
            TERMINAL_FILENAME,
        }:
            raise ValueError("worker terminal directory inventory drifted")
        attempt, attempt_bytes = _read_canonical_json(
            worker_root / ATTEMPT_FILENAME, label="worker attempt"
        )
        terminal, terminal_bytes = _read_canonical_json(
            worker_root / TERMINAL_FILENAME, label="worker terminal"
        )
        expected_state_ids = _state_ids_for_worker(contract.data, worker_index)
        if (
            set(attempt) != _ATTEMPT_KEYS
            or set(terminal) != _TERMINAL_KEYS
            or attempt.get("schema_version") != SCHEMA_VERSION
            or attempt.get("protocol_id") != EXECUTION_PROTOCOL_ID
            or attempt.get("status")
            != "CLAIMED_NO_RETRY_THROUGHPUT_PILOT_WORKER_V1"
            or attempt.get("worker_index") != worker_index
            or tuple(attempt.get("state_ids", ())) != expected_state_ids
            or attempt.get("retry_count") != 0
            or not isinstance(attempt.get("device_id"), str)
            or _DEVICE.fullmatch(attempt["device_id"]) is None
            or terminal.get("schema_version") != SCHEMA_VERSION
            or terminal.get("protocol_id") != WORKER_TERMINAL_PROTOCOL_ID
            or terminal.get("status") != "COMPLETED_THROUGHPUT_PILOT_WORKER_V1"
            or terminal.get("worker_index") != worker_index
            or tuple(terminal.get("state_ids", ())) != expected_state_ids
            or terminal.get("attempt_sha256") != sha256_bytes(attempt_bytes)
            or terminal.get("failure_class") is not None
            or terminal.get("metric_only") is not True
            or terminal.get("counts")
            != {
                "pair_completed_count": 3,
                "pair_expected_count": 3,
                "retry_count": 0,
            }
        ):
            raise ValueError("worker attempt/terminal identity drifted")
        if any(
            terminal.get(key) != attempt.get(key)
            for key in _ATTEMPT_TERMINAL_IDENTITY_KEYS
        ):
            raise ValueError("worker terminal provenance differs from its attempt")
        runtime_identity = terminal.get("runtime_identity_sha256")
        if (
            not isinstance(runtime_identity, str)
            or _SHA256.fullmatch(runtime_identity) is None
        ):
            raise ValueError("worker runtime identity is invalid")
        if common_runtime_identity is None:
            common_runtime_identity = runtime_identity
        elif runtime_identity != common_runtime_identity:
            raise ValueError("workers disagree on the frozen runtime identity")
        projection = {
            key: terminal[key]
            for key in (
                "candidate_schedule_sha256",
                "config_sha256",
                "freeze_manifest_sha256",
                "model_inventory_sha256",
                "processor_inventory_sha256",
                "source_inventory_sha256",
            )
        }
        if common is None:
            common = projection
        elif projection != common:
            raise ValueError("worker terminals disagree on bound input identities")
        if (
            terminal["config_sha256"] != contract.config_sha256
            or terminal["source_inventory_sha256"]
            != contract.data["source"]["inventory_sha256"]
        ):
            raise ValueError("worker terminal differs from live validated source")
        total_memory = terminal.get("device_total_memory_bytes")
        if type(total_memory) is not int or total_memory <= 0:
            raise ValueError("worker terminal device memory is invalid")
        memory_by_worker[worker_index] = total_memory
        pair_results = _sequence(
            terminal.get("pair_results"), label="worker pair results"
        )
        if tuple(pair.get("state_id") for pair in pair_results) != expected_state_ids:
            raise ValueError("worker paired result roster drifted")
        for state_id in expected_state_ids:
            worker_by_state[state_id] = worker_index
        terminals.extend(pair_results)
        terminal_hashes.append(
            {
                "attempt_sha256": sha256_bytes(attempt_bytes),
                "terminal_sha256": sha256_bytes(terminal_bytes),
                "worker_index": worker_index,
            }
        )
    if common is None:
        raise AssertionError("four-worker aggregate lost its common identity")

    by_state = {str(pair["state_id"]): pair for pair in terminals}
    if len(by_state) != len(EXPECTED_STATE_IDS) or set(by_state) != set(
        EXPECTED_STATE_IDS
    ):
        raise ValueError("four-worker paired result inventory is not exact")
    ordered = tuple(by_state[state_id] for state_id in EXPECTED_STATE_IDS)
    inflated = tuple(_inflate_terminal_pair(pair) for pair in ordered)
    from causalcache.set_utility_throughput_pilot_pair_v1 import (
        aggregate_train_only_set_utility_throughput_pilot_pairs_v1,
    )

    metrics = aggregate_train_only_set_utility_throughput_pilot_pairs_v1(inflated)
    selection = select_reference_teacher_microbatch_v1(
        ordered,
        worker_index_by_state=worker_by_state,
        device_total_memory_bytes_by_worker=memory_by_worker,
    )
    aggregate = {
        **common,
        "device_total_memory_bytes_by_worker": {
            str(key): value for key, value in sorted(memory_by_worker.items())
        },
        "metric_only": True,
        "metrics": metrics,
        "protocol_id": AGGREGATE_PROTOCOL_ID,
        "retry_count": 0,
        "schema_version": SCHEMA_VERSION,
        "selection": selection,
        "state_ids": list(EXPECTED_STATE_IDS),
        "status": "COMPLETED_THROUGHPUT_PILOT_EXECUTION_AGGREGATE_V1",
        "worker_count": WORKER_COUNT,
        "worker_terminal_hashes": terminal_hashes,
    }
    _write_atomic_no_clobber(
        output / AGGREGATE_FILENAME,
        canonical_pretty_json_bytes(aggregate),
    )
    return aggregate


__all__ = [
    "AGGREGATE_FILENAME",
    "AGGREGATE_PROTOCOL_ID",
    "ATTEMPT_FILENAME",
    "EXECUTION_PROTOCOL_ID",
    "ExecutionLaunchEnvelopeV1",
    "LoadedPilotStateV1",
    "RESERVED_MEMORY_DENOMINATOR",
    "RESERVED_MEMORY_NUMERATOR",
    "SCHEMA_VERSION",
    "TERMINAL_FILENAME",
    "ValidatedExecutionEnvelopeV1",
    "WORKER_TERMINAL_PROTOCOL_ID",
    "WorkerRuntimeBundleV1",
    "aggregate_throughput_pilot_workers_v1",
    "build_production_runtime_v1",
    "canonical_candidate_schedule_producer_bytes_v2",
    "load_worker_semantic_inputs_v1",
    "run_throughput_pilot_worker_v1",
    "select_reference_teacher_microbatch_v1",
    "validated_execution_from_authorized_projection_v1",
    "validate_execution_envelope_v1",
]
