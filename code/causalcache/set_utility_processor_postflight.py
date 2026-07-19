"""Independent read-only postflight validation for completed processor freezes."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.set_utility_full_pool import build_full_pool_inspection_config
from causalcache.set_utility_processor_artifacts import (
    ARTIFACT_FILENAME_TEMPLATE,
    CANDIDATE_SCHEDULE_FILENAME,
    EXPECTED_QUERY_COUNT_PER_TRAJECTORY,
    CandidateScheduleDescriptor,
    ProcessorArtifactShardDescriptor,
    ProcessorWorkerSchedule,
    build_final_candidate_schedule,
    build_git_safe_processor_freeze_summary,
    build_processor_worker_schedule,
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    final_candidate_schedule_payload,
    inspect_processor_artifact_shard,
    sha256_bytes,
)
from causalcache.set_utility_processor_freeze import (
    CONTEXT_LIMIT_TOKENS,
    FrozenQueryCandidateRecord,
    MAXIMUM_FINAL_CANDIDATE_COUNT,
    MINIMUM_FINAL_CANDIDATE_COUNT,
    RESERVED_ACTION_TOKENS,
    WORKER_COUNT,
)
from causalcache.set_utility_processor_freeze import (
    operation_budget_payload,
    validate_freeze_query_topology,
    validate_processor_only_source,
)
from causalcache.set_utility_processor_freeze_contract import (
    CANONICAL_EXECUTION_CONFIG_PATH,
    FULL_POOL_INVENTORY_MANIFEST_PATH,
    INDEPENDENT_REFERENCE_GATE_CONFIG_PATH,
    OCR_BACKEND_CONFIG_PATH,
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_PATH_ARGUMENTS,
    REQUIRED_VERSION_ARGUMENTS_BY_PHASE,
    RUNNER_PATH,
    SNAPSHOT_MANIFEST_PATH,
    ProcessorFreezeExecutionContract,
)
from causalcache.set_utility_processor_substrate import build_selected_row_read_plan


SCHEMA_VERSION = "1.0.0"
VALIDATION_STATUS = "VALID_COMPLETED_SET_UTILITY_PROCESSOR_FREEZE"
COMPLETED_STATUS = "PROCESSOR_ONLY_CANDIDATE_FREEZE_COMPLETED"
EXPECTED_TOP_LEVEL_FILES = frozenset(
    {"manifest.json", "run-identity.json", CANDIDATE_SCHEDULE_FILENAME}
)
EXPECTED_TOP_LEVEL_DIRECTORIES = frozenset(
    {"candidate-parts", "logs", "receipts", "substrate"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ExpectedFrozenQuery:
    state_id: str
    trajectory_id: str
    source_id: str
    role: str
    query_kind: str
    decision_step_id: int
    maximum_labeled_cardinality: int
    initial_candidate_event_step_ids: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExpectedFrozenQuery:
        return cls(
            state_id=value["state_id"],
            trajectory_id=value["trajectory_id"],
            source_id=value["source_id"],
            role=value["role"],
            query_kind=value["query_kind"],
            decision_step_id=value["decision_step_id"],
            maximum_labeled_cardinality=value["maximum_labeled_cardinality"],
            initial_candidate_event_step_ids=tuple(
                value["initial_candidate_event_step_ids"]
            ),
        )


@dataclass(frozen=True)
class ProcessorFreezePostflightContext:
    repository_root: Path
    execution_config_path: Path
    execution_config_sha256: str
    expected_git_revision: str
    worker_schedule: ProcessorWorkerSchedule
    expected_queries: Mapping[str, ExpectedFrozenQuery]
    source_audit: Mapping[str, Any]
    snapshot_identity_without_model_dir: Mapping[str, Any]
    ocr_backend_config_sha256: str
    ocr_runtime_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.worker_schedule, ProcessorWorkerSchedule):
            raise TypeError("postflight worker schedule is invalid")
        if _SHA256.fullmatch(self.execution_config_sha256) is None:
            raise ValueError("postflight execution-config SHA256 is invalid")
        if _GIT_REVISION.fullmatch(self.expected_git_revision) is None:
            raise ValueError("postflight Git revision is invalid")
        if _SHA256.fullmatch(self.ocr_backend_config_sha256) is None:
            raise ValueError("postflight OCR-config SHA256 is invalid")
        if not self.expected_queries or any(
            not isinstance(value, ExpectedFrozenQuery)
            for value in self.expected_queries.values()
        ):
            raise ValueError("postflight expected-query inventory is invalid")
        if set(self.expected_queries) != {
            value.state_id for value in self.expected_queries.values()
        }:
            raise ValueError("postflight expected-query keys drifted")
        trajectories = {
            item.trajectory_id
            for worker in self.worker_schedule.workers
            for item in worker.trajectories
        }
        if Counter(
            query.trajectory_id for query in self.expected_queries.values()
        ) != Counter({trajectory_id: 2 for trajectory_id in trajectories}):
            raise ValueError("postflight requires exactly two queries per trajectory")


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


def _load_bound_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular file")
    return _strict_json_object(path.read_bytes(), label=label)


def _ocr_runtime_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    recognizer = config["models"]["recognizer"]
    package_files = config["upstream_package_files"]
    return {
        "model_sha256": {
            role: record["sha256"] for role, record in config["models"].items()
        },
        "rapidocr_package_file_sha256": {
            "ch_ppocr_rec/main.py": package_files["recognizer_main_py_sha256"],
            "config.yaml": package_files["config_yaml_sha256"],
            "default_models.yaml": package_files["default_models_yaml_sha256"],
            "inference_engine/onnxruntime/main.py": package_files[
                "onnxruntime_main_py_sha256"
            ],
            "utils/load_image.py": package_files["load_image_py_sha256"],
        },
        "recognizer_character_inventory": {
            "canonical_json_sha256": recognizer[
                "embedded_character_canonical_json_sha256"
            ],
            "entry_count": recognizer["embedded_character_entry_count"],
            "metadata_key": recognizer["embedded_character_metadata_key"],
            "utf8_sha256": recognizer["embedded_character_utf8_sha256"],
        },
        "runtime_packages": dict(config["runtime_packages"]),
        "wheel_sha256": {
            distribution: record["sha256"]
            for distribution, record in config["wheels"].items()
        },
    }


def _snapshot_identity_without_model_dir(
    repository_root: Path,
) -> dict[str, Any]:
    path = repository_root / SNAPSHOT_MANIFEST_PATH
    payload = path.read_bytes()
    manifest = _strict_json_object(payload, label="snapshot manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("snapshot manifest file inventory is invalid")
    records = [
        {"path": item["path"], "sha256": item["sha256"], "size": item["size"]}
        for item in files
    ]
    return {
        "file_inventory_sha256": sha256_bytes(canonical_json_bytes(records)),
        "maximum_context_tokens": CONTEXT_LIMIT_TOKENS,
        "model_or_policy_loaded": False,
        "model_repo": manifest["repo"],
        "model_revision": manifest["revision"],
        "snapshot_manifest_sha256": sha256_bytes(payload),
        "verified_file_count": len(records),
        "verified_total_bytes": sum(item["size"] for item in records),
    }


def build_processor_freeze_postflight_context(
    contract: ProcessorFreezeExecutionContract,
    *,
    expected_git_revision: str,
) -> ProcessorFreezePostflightContext:
    """Rebuild formal expectations solely from the byte-bound repository inputs."""
    if not isinstance(contract, ProcessorFreezeExecutionContract):
        raise TypeError("postflight requires a validated Execution-CF contract")
    root = contract.repository_root
    inventory = _load_bound_json(
        root / FULL_POOL_INVENTORY_MANIFEST_PATH,
        label="full-pool inventory",
    )
    census_path = root / "data/manifests/set_utility_full_pool_census_v2.json"
    census = _load_bound_json(census_path, label="full-pool census")
    base = _load_bound_json(
        root / INDEPENDENT_REFERENCE_GATE_CONFIG_PATH,
        label="independent base config",
    )
    files = tuple(
        SourceFileSpec(item["path"], item["size_bytes"], item["lfs_sha256"])
        for item in inventory["inventory"]["files"]
    )
    build_full_pool_inspection_config(
        base,
        transport_files=tuple(item.transport_file for item in files),
        format_safety_maximum_decisions=None,
    )
    freeze = contract.freeze_b_v2_manifest
    _, grouped_queries = validate_freeze_query_topology(
        freeze["assignments"],
        freeze["query_states"],
        anchor_step_by_stratum=freeze["repair"]["stratum_anchor_decision_step"],
    )
    plan = build_selected_row_read_plan(
        freeze["assignments"],
        source_files=files,
        source_row_counts=census["source"]["row_counts_by_file"],
    )
    schedule = build_processor_worker_schedule(plan)
    expected_queries = {
        query["state_id"]: ExpectedFrozenQuery.from_mapping(query)
        for values in grouped_queries.values()
        for query in values
    }
    ocr_path = root / OCR_BACKEND_CONFIG_PATH
    ocr_payload = ocr_path.read_bytes()
    ocr_config = _strict_json_object(ocr_payload, label="OCR backend config")
    return ProcessorFreezePostflightContext(
        repository_root=root,
        execution_config_path=root / CANONICAL_EXECUTION_CONFIG_PATH,
        execution_config_sha256=contract.config_sha256,
        expected_git_revision=expected_git_revision,
        worker_schedule=schedule,
        expected_queries=expected_queries,
        source_audit=validate_processor_only_source(root / RUNNER_PATH),
        snapshot_identity_without_model_dir=_snapshot_identity_without_model_dir(root),
        ocr_backend_config_sha256=sha256_bytes(ocr_payload),
        ocr_runtime_identity=_ocr_runtime_identity(ocr_config),
    )


def _regular_file(path: Path, *, label: str) -> None:
    mode = path.lstat().st_mode
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be one real regular file")


def _real_directory(path: Path, *, label: str) -> None:
    mode = path.lstat().st_mode
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be one real directory")


def _require_exact_children(
    root: Path,
    *,
    files: set[str] | frozenset[str],
    directories: set[str] | frozenset[str] = frozenset(),
    label: str,
) -> None:
    _real_directory(root, label=label)
    entries = {entry.name: entry for entry in os.scandir(root)}
    expected = set(files) | set(directories)
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise ValueError(f"{label} inventory drifted: missing={missing}, extra={extra}")
    for name in files:
        if ".partial" in name:
            raise ValueError(f"{label} contains a partial artifact")
        _regular_file(root / name, label=f"{label}/{name}")
    for name in directories:
        _real_directory(root / name, label=f"{label}/{name}")


def _validate_tree(root: Path) -> None:
    _require_exact_children(
        root,
        files=EXPECTED_TOP_LEVEL_FILES,
        directories=EXPECTED_TOP_LEVEL_DIRECTORIES,
        label="processor-freeze output root",
    )
    _require_exact_children(
        root / "candidate-parts",
        files={
            f"worker-{worker:02d}.jsonl" for worker in range(WORKER_COUNT)
        },
        label="candidate-parts",
    )
    _require_exact_children(
        root / "logs",
        files={
            f"{phase}-worker-{worker:02d}.log"
            for phase in ("ocr", "processor")
            for worker in range(WORKER_COUNT)
        },
        label="worker logs",
    )
    _require_exact_children(
        root / "receipts",
        files={
            f"ocr-worker-{worker:02d}.json" for worker in range(WORKER_COUNT)
        },
        label="OCR receipts",
    )
    _require_exact_children(
        root / "substrate",
        files={
            ARTIFACT_FILENAME_TEMPLATE.format(worker_index=worker)
            for worker in range(WORKER_COUNT)
        },
        label="processor substrate",
    )


def _candidate_records(
    root: Path,
    context: ProcessorFreezePostflightContext,
) -> tuple[FrozenQueryCandidateRecord, ...]:
    worker_by_trajectory = {
        item.trajectory_id: worker.worker_index
        for worker in context.worker_schedule.workers
        for item in worker.trajectories
    }
    observed: list[FrozenQueryCandidateRecord] = []
    for worker_index in range(WORKER_COUNT):
        path = root / "candidate-parts" / f"worker-{worker_index:02d}.jsonl"
        payload = path.read_bytes()
        if not payload or not payload.endswith(b"\n"):
            raise ValueError("candidate part must be non-empty canonical JSONL")
        records: list[FrozenQueryCandidateRecord] = []
        for line_index, line in enumerate(payload.splitlines()):
            value = _strict_json_object(
                line,
                label=f"candidate part {worker_index} line {line_index}",
            )
            record = FrozenQueryCandidateRecord.from_payload(value)
            if value != record.to_payload() or line != canonical_json_bytes(value):
                raise ValueError("candidate part record is not canonical")
            records.append(record)
        if tuple(record.state_id for record in records) != tuple(
            sorted(record.state_id for record in records)
        ):
            raise ValueError("candidate part state order drifted")
        if any(
            worker_by_trajectory.get(record.trajectory_id) != worker_index
            for record in records
        ):
            raise ValueError("candidate part escaped its processor worker")
        expected_payload = b"".join(
            canonical_json_bytes(record.to_payload()) + b"\n" for record in records
        )
        if payload != expected_payload:
            raise ValueError("candidate part bytes are not canonical")
        observed.extend(records)
    if len({record.state_id for record in observed}) != len(observed):
        raise ValueError("candidate parts contain duplicate state IDs")
    if {record.state_id for record in observed} != set(context.expected_queries):
        raise ValueError("candidate state inventory differs from the frozen roster")
    for record in observed:
        expected = context.expected_queries[record.state_id]
        actual_identity = (
            record.state_id,
            record.trajectory_id,
            record.source_id,
            record.role,
            record.query_kind,
            record.decision_step_id,
            record.maximum_labeled_cardinality,
            record.candidate_context.initial_candidate_event_step_ids,
        )
        expected_identity = (
            expected.state_id,
            expected.trajectory_id,
            expected.source_id,
            expected.role,
            expected.query_kind,
            expected.decision_step_id,
            expected.maximum_labeled_cardinality,
            expected.initial_candidate_event_step_ids,
        )
        if actual_identity != expected_identity:
            raise ValueError("candidate record differs from its frozen query")
        candidate_count = len(record.candidate_context.candidate_event_step_ids)
        if (
            record.maximum_labeled_cardinality != 2
            or not MINIMUM_FINAL_CANDIDATE_COUNT
            <= candidate_count
            <= MAXIMUM_FINAL_CANDIDATE_COUNT
            or record.candidate_context.reserved_action_tokens
            != RESERVED_ACTION_TOKENS
            or record.candidate_context.context_limit != CONTEXT_LIMIT_TOKENS
        ):
            raise ValueError("candidate record violates exact Execution-CF limits")
    return tuple(sorted(observed, key=lambda record: record.state_id))


def _exact_operation_budget(
    records: Sequence[FrozenQueryCandidateRecord],
) -> dict[str, int]:
    state_count = len(records)
    raw_rows = sum(
        1 + len(record.candidate_context.candidate_event_step_ids)
        + math.comb(len(record.candidate_context.candidate_event_step_ids), 2)
        for record in records
    )
    return {
        "canonical_action_generations": 2 * state_count,
        "reference_teacher_forwards": state_count,
        "identical_reference_repeat_teacher_forwards": state_count,
        "capped_candidate_teacher_forwards": raw_rows,
        "total_teacher_forwards": 2 * state_count + raw_rows,
        "kl_measurements": state_count + raw_rows,
        "raw_label_rows": raw_rows,
        "total_model_operations": 4 * state_count + raw_rows,
    }


def _candidate_schedule(
    root: Path,
    records: tuple[FrozenQueryCandidateRecord, ...],
) -> tuple[Any, CandidateScheduleDescriptor]:
    schedule = build_final_candidate_schedule(records)
    expected_payload = canonical_pretty_json_bytes(
        final_candidate_schedule_payload(schedule)
    )
    path = root / CANDIDATE_SCHEDULE_FILENAME
    if path.read_bytes() != expected_payload:
        raise ValueError("candidate schedule differs from canonical reconstruction")
    exact_operations = _exact_operation_budget(records)
    if operation_budget_payload(schedule.label_schedule) != exact_operations:
        raise ValueError("candidate schedule exact operation invariants drifted")
    if dict(schedule.label_schedule.subset_count_by_cardinality) != {
        0: len(records),
        1: sum(
            len(record.candidate_context.candidate_event_step_ids)
            for record in records
        ),
        2: sum(
            math.comb(len(record.candidate_context.candidate_event_step_ids), 2)
            for record in records
        ),
    }:
        raise ValueError("candidate schedule cardinality inventory drifted")
    return schedule, CandidateScheduleDescriptor(
        filename=CANDIDATE_SCHEDULE_FILENAME,
        sha256=sha256_bytes(expected_payload),
        byte_count=len(expected_payload),
        state_count=len(records),
        subset_forward_count=exact_operations["raw_label_rows"],
        total_model_operations=exact_operations["total_model_operations"],
        candidate_inventory_sha256=schedule.candidate_inventory_sha256,
    )


def _canonical_tar_json(
    archive: tarfile.TarFile,
    *,
    expected_name: str,
    label: str,
) -> dict[str, Any]:
    member = archive.next()
    if member is None or member.name != expected_name or not member.isfile():
        raise ValueError(f"{label} member order or type drifted")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"{label} could not be opened")
    payload = source.read()
    if len(payload) != member.size:
        raise ValueError(f"{label} ended early")
    value = _strict_json_object(payload, label=label)
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical compact JSON")
    return value


def _audit_canonical_shard_and_format_tally(
    path: Path,
    *,
    worker: Any,
) -> Mapping[str, int]:
    """Check canonical JSON members and rebuild source formats from terminal OCR."""
    tally: Counter[str] = Counter()
    try:
        opened = tarfile.open(path, mode="r:")
    except tarfile.TarError as error:
        raise ValueError(
            "processor artifact is not a valid uncompressed tar"
        ) from error
    with opened as archive:
        _canonical_tar_json(
            archive,
            expected_name="shard-manifest.json",
            label=f"worker {worker.worker_index} shard manifest",
        )
        for trajectory_index, expected in enumerate(worker.trajectories):
            for query_index in range(EXPECTED_QUERY_COUNT_PER_TRAJECTORY):
                prefix = (
                    f"trajectories/{trajectory_index:04d}/queries/"
                    f"{query_index:02d}"
                )
                record = _canonical_tar_json(
                    archive,
                    expected_name=f"{prefix}/record.json",
                    label=f"worker {worker.worker_index} query record",
                )
                images = record.get("image_inventory")
                if not isinstance(images, list) or not images:
                    raise ValueError(
                        "query image inventory drifted during canonical audit"
                    )
                for image_index, image in enumerate(images):
                    expected_member = f"{prefix}/images/{image_index:04d}.bin"
                    if (
                        not isinstance(image, Mapping)
                        or image.get("member") != expected_member
                    ):
                        raise ValueError(
                            "query image binding drifted during canonical audit"
                        )
                    member = archive.next()
                    if (
                        member is None
                        or member.name != expected_member
                        or not member.isfile()
                    ):
                        raise ValueError("query image member order or type drifted")
                if record.get("query_kind") != "terminal":
                    continue
                if record.get("trajectory_id") != expected.trajectory_id:
                    raise ValueError("terminal OCR trajectory identity drifted")
                ocr_records = record.get("ocr_records_by_path")
                if not isinstance(ocr_records, Mapping) or not ocr_records:
                    raise ValueError("terminal OCR inventory is invalid")
                for ocr_record in ocr_records.values():
                    if not isinstance(ocr_record, Mapping):
                        raise ValueError("terminal OCR record is invalid")
                    source_format = ocr_record.get("source_format")
                    source_mode = ocr_record.get("source_mode")
                    for value, label in (
                        (source_format, "OCR source format"),
                        (source_mode, "OCR source mode"),
                    ):
                        if (
                            not isinstance(value, str)
                            or not value
                            or value != value.strip()
                            or any(
                                ord(character) < 32 or ord(character) == 127
                                for character in value
                            )
                        ):
                            raise ValueError(f"{label} is not canonical text")
                    tally[f"{source_format}:{source_mode}"] += 1
        _canonical_tar_json(
            archive,
            expected_name="completion.json",
            label=f"worker {worker.worker_index} completion receipt",
        )
        if archive.next() is not None:
            raise ValueError("processor artifact contains a trailing tar member")
    if not tally:
        raise ValueError("processor artifact has an empty terminal OCR format tally")
    return dict(sorted(tally.items()))


def _artifact_shards(
    root: Path,
    schedule: ProcessorWorkerSchedule,
) -> tuple[
    tuple[ProcessorArtifactShardDescriptor, ...],
    tuple[Mapping[str, int], ...],
]:
    def inspect(
        worker: Any,
    ) -> tuple[ProcessorArtifactShardDescriptor, Mapping[str, int]]:
        path = root / "substrate" / worker.filename
        descriptor = inspect_processor_artifact_shard(
            path,
            expected_worker=worker,
        )
        return descriptor, _audit_canonical_shard_and_format_tally(
            path,
            worker=worker,
        )

    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
        inspected = tuple(executor.map(inspect, schedule.workers))
    descriptors = tuple(item[0] for item in inspected)
    tallies = tuple(item[1] for item in inspected)
    if tuple(item.worker_index for item in descriptors) != tuple(range(WORKER_COUNT)):
        raise ValueError("processor shard worker order drifted")
    return descriptors, tallies


def _validate_ocr_receipts(
    root: Path,
    context: ProcessorFreezePostflightContext,
    descriptors: Sequence[ProcessorArtifactShardDescriptor],
    format_tallies: Sequence[Mapping[str, int]],
) -> None:
    if len(descriptors) != len(format_tallies):
        raise ValueError("OCR receipt/shard inventory length drifted")
    for worker_index, (descriptor, format_tally) in enumerate(
        zip(descriptors, format_tallies, strict=True)
    ):
        path = root / "receipts" / f"ocr-worker-{worker_index:02d}.json"
        payload = path.read_bytes()
        receipt = _strict_json_object(payload, label=f"OCR receipt {worker_index}")
        if payload != canonical_json_bytes(receipt):
            raise ValueError("OCR receipt is not canonical compact JSON")
        if set(receipt) != {
            "backend_config_sha256",
            "format_tally",
            "ocr_runtime_identity",
            "shard",
            "worker_index",
        }:
            raise ValueError("OCR receipt fields drifted")
        if (
            receipt["backend_config_sha256"]
            != context.ocr_backend_config_sha256
            or receipt["ocr_runtime_identity"] != context.ocr_runtime_identity
            or receipt["shard"] != descriptor.to_payload()
            or receipt["worker_index"] != worker_index
            or receipt["format_tally"] != format_tally
        ):
            raise ValueError("OCR receipt identity or count drifted")


def _runtime_cli_keys() -> set[str]:
    result = {
        item.removeprefix("--").replace("-", "_")
        for item in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }
    result.update(
        item.removeprefix("--").replace("-", "_")
        for values in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.values()
        for item in values
    )
    return result


def _run_identity(
    root: Path,
    context: ProcessorFreezePostflightContext,
) -> tuple[dict[str, Any], str]:
    path = root / "run-identity.json"
    payload = path.read_bytes()
    value = _strict_json_object(payload, label="run identity")
    if payload != canonical_json_bytes(value):
        raise ValueError("run identity is not canonical compact JSON")
    if set(value) != {
        "config_sha256",
        "git_revision",
        "runtime_cli",
        "source_audit",
        "snapshot",
        "worker_execution_sha256",
    }:
        raise ValueError("run identity fields drifted")
    runtime = value["runtime_cli"]
    if not isinstance(runtime, Mapping) or set(runtime) != _runtime_cli_keys():
        raise ValueError("run identity runtime-CLI fields drifted")
    if any(
        not isinstance(item, str)
        or not item
        or item != item.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in item)
        for item in runtime.values()
    ):
        raise ValueError("run identity runtime-CLI value is not canonical text")
    path_keys = {
        item.removeprefix("--").replace("-", "_") for item in REQUIRED_PATH_ARGUMENTS
    }
    if any(not Path(runtime[key]).is_absolute() for key in path_keys):
        raise ValueError("run identity contains a non-absolute runtime path")
    if (
        runtime["repository_root"] != str(context.repository_root.resolve())
        or runtime["execution_config"]
        != str(context.execution_config_path.resolve())
        or runtime["snapshot_manifest"]
        != str((context.repository_root / SNAPSHOT_MANIFEST_PATH).resolve())
        or runtime["output_root"] != str(root.resolve())
        or runtime["worker_count"] != "4"
        or runtime["git_revision"] != context.expected_git_revision
        or not runtime["container_image_digest"].startswith("sha256:")
        or _SHA256.fullmatch(runtime["container_image_digest"][7:]) is None
    ):
        raise ValueError("run identity fixed runtime values drifted")
    expected_snapshot = {
        "model_dir": str(Path(runtime["model_dir"]).resolve()),
        **context.snapshot_identity_without_model_dir,
    }
    if (
        value["config_sha256"] != context.execution_config_sha256
        or value["git_revision"] != context.expected_git_revision
        or value["source_audit"] != context.source_audit
        or value["snapshot"] != expected_snapshot
        or value["worker_execution_sha256"]
        != context.worker_schedule.execution_sha256
    ):
        raise ValueError("run identity fixed fields drifted")
    return value, sha256_bytes(canonical_json_bytes(value))


def validate_completed_processor_freeze_root(
    output_root: str | Path,
    *,
    context: ProcessorFreezePostflightContext,
) -> dict[str, Any]:
    """Validate one atomically published processor-freeze root without writing it."""
    if not isinstance(context, ProcessorFreezePostflightContext):
        raise TypeError("postflight context is invalid")
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("postflight output root must be absolute")
    _validate_tree(root)
    records = _candidate_records(root, context)
    candidate_schedule, candidate_descriptor = _candidate_schedule(root, records)
    descriptors, format_tallies = _artifact_shards(root, context.worker_schedule)
    _validate_ocr_receipts(root, context, descriptors, format_tallies)
    _, run_identity_sha256 = _run_identity(root, context)
    summary = build_git_safe_processor_freeze_summary(
        worker_schedule=context.worker_schedule,
        artifact_shards=descriptors,
        candidate_schedule=candidate_schedule,
        candidate_artifact=candidate_descriptor,
    )
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = _strict_json_object(manifest_bytes, label="completion manifest")
    expected_manifest = {
        "contains_model_or_policy_output": False,
        "contains_restoration_labels": False,
        "execution_config_sha256": context.execution_config_sha256,
        "git_revision": context.expected_git_revision,
        "processor_freeze_summary": summary,
        "run_identity_sha256": run_identity_sha256,
        "status": COMPLETED_STATUS,
    }
    if manifest != expected_manifest:
        raise ValueError("completion manifest differs from whole-root reconstruction")
    if manifest_bytes != canonical_pretty_json_bytes(expected_manifest):
        raise ValueError("completion manifest is not canonical pretty JSON")
    return {
        "artifact_shards": [item.to_payload() for item in descriptors],
        "candidate_artifact": candidate_descriptor.to_payload(),
        "execution_config_sha256": context.execution_config_sha256,
        "git_revision": context.expected_git_revision,
        "operation_budget": operation_budget_payload(
            candidate_schedule.label_schedule
        ),
        "processor_freeze_summary_sha256": sha256_bytes(
            canonical_json_bytes(summary)
        ),
        "run_identity_sha256": run_identity_sha256,
        "schema_version": SCHEMA_VERSION,
        "status": VALIDATION_STATUS,
    }


__all__ = [
    "COMPLETED_STATUS",
    "ExpectedFrozenQuery",
    "ProcessorFreezePostflightContext",
    "VALIDATION_STATUS",
    "build_processor_freeze_postflight_context",
    "validate_completed_processor_freeze_root",
]
