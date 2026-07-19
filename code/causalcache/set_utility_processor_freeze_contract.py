"""Fail-closed Execution-CF contract for processor-only candidate freezing."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_processor_freeze_execution_cf"
STATUS = "PROCESSOR_ONLY_CANDIDATE_FREEZE_EXECUTION_AUTHORIZED"
VALIDATION_STATUS = "VALID_SET_UTILITY_PROCESSOR_FREEZE_EXECUTION_CF"

FREEZE_B_V2_MANIFEST_PATH = (
    "data/manifests/set_utility_freeze_b_v2_terminal_index_repair.json"
)
FREEZE_B_V2_MANIFEST_SHA256 = (
    "915892ef2e0f1495da4b9e409b3e7a112dc86cda0b06384e8a1b7f8053581d30"
)
FREEZE_B_V2_MANIFEST_BYTE_COUNT = 2_242_268
FREEZE_B_V2_PROTOCOL_ID = (
    "causalcache_set_utility_freeze_b_v2_terminal_index_repair"
)
FREEZE_B_V2_STATUS = "POLICY_BLIND_FREEZE_B_V2_TERMINAL_INDEX_REPAIR_COMPLETED"

CANONICAL_EXECUTION_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_processor_freeze_execution_cf.json"
)
RUNNER_PATH = "code/scripts/run_set_utility_processor_freeze.py"
SNAPSHOT_MANIFEST_PATH = "code/configs/gui_owl_1_5_8b_snapshot.json"
OCR_BACKEND_CONFIG_PATH = "code/configs/restoration_v2_ocr_backend.json"
OCR_BACKEND_COMPLETION_MANIFEST_PATH = (
    "data/manifests/restoration_v2_ocr_backend.json"
)
P0_CENSUS_MANIFEST_PATH = "data/manifests/set_utility_full_pool_census_v2.json"
FULL_POOL_INVENTORY_MANIFEST_PATH = (
    "data/manifests/set_utility_full_pool_inventory_v1.json"
)
INDEPENDENT_REFERENCE_GATE_CONFIG_PATH = (
    "code/configs/independent_reference_gate_v1.json"
)

REQUIRED_FROZEN_INPUT_PATHS = {
    "full_pool_inventory_manifest": FULL_POOL_INVENTORY_MANIFEST_PATH,
    "independent_reference_gate_base_config": (
        INDEPENDENT_REFERENCE_GATE_CONFIG_PATH
    ),
    "ocr_backend_completion_manifest": OCR_BACKEND_COMPLETION_MANIFEST_PATH,
    "ocr_backend_config": OCR_BACKEND_CONFIG_PATH,
    "p0_census_manifest": P0_CENSUS_MANIFEST_PATH,
    "teacher_snapshot_manifest": SNAPSHOT_MANIFEST_PATH,
}

REQUIRED_REPOSITORY_SOURCE_PATHS = (
    "code/causalcache/data/guiodyssey.py",
    "code/causalcache/data/guiodyssey_independent.py",
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/causalcache/data/restoration_v2_selection.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_1.py",
    "code/causalcache/policy/prompt.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/schema.py",
    "code/causalcache/set_utility_consumed_ledger.py",
    "code/causalcache/set_utility_consumed_ledger_contract.py",
    "code/causalcache/set_utility_full_pool.py",
    "code/causalcache/set_utility_full_pool_inventory_v1.py",
    "code/causalcache/set_utility_label_producer.py",
    "code/causalcache/set_utility_label_schedule.py",
    "code/causalcache/set_utility_long_pool.py",
    "code/causalcache/set_utility_processor_artifacts.py",
    "code/causalcache/set_utility_processor_freeze.py",
    "code/causalcache/set_utility_processor_freeze_contract.py",
    "code/causalcache/set_utility_processor_prompt.py",
    "code/causalcache/set_utility_processor_substrate.py",
    "code/causalcache/set_utility_split_validation.py",
    SNAPSHOT_MANIFEST_PATH,
    OCR_BACKEND_CONFIG_PATH,
    RUNNER_PATH,
)

REQUIRED_PATH_ARGUMENTS = (
    "--repository-root",
    "--execution-config",
    "--source-root",
    "--model-dir",
    "--snapshot-manifest",
    "--ocr-model-dir",
    "--ocr-wheel-dir",
    "--output-root",
    "--ocr-python-executable",
    "--processor-python-executable",
)
REQUIRED_IDENTITY_ARGUMENTS = (
    "--git-revision",
    "--host-alias",
    "--host-hostname",
    "--container-id",
    "--container-image-digest",
    "--worker-count",
)
REQUIRED_VERSION_ARGUMENTS_BY_PHASE = {
    "raw_decode_and_ocr": (
        "--ocr-python-version",
        "--ocr-runtime-version",
        "--pyarrow-version",
    ),
    "auto_processor": (
        "--processor-python-version",
        "--transformers-version",
        "--torch-version",
        "--pillow-version",
    ),
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _canonical_compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be one JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields drifted")


def _lower_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _safe_repository_file(
    root: Path,
    relative_path: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{label} path must be a non-empty string")
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or relative_path != pure.as_posix()
        or relative_path.startswith("./")
    ):
        raise ValueError(f"{label} path must be normalized and repository-relative")
    path = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} path must not traverse a symlink")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escaped the repository") from error
    if not path.is_file():
        raise ValueError(f"{label} must bind one existing regular file")
    return path


def _file_binding(root: Path, relative_path: str) -> dict[str, Any]:
    path = _safe_repository_file(root, relative_path, label=relative_path)
    payload = path.read_bytes()
    return {
        "byte_count": len(payload),
        "path": relative_path,
        "sha256": sha256_bytes(payload),
    }


def _validate_file_binding(
    root: Path,
    value: Any,
    *,
    label: str,
    expected_path: str | None = None,
) -> bytes:
    binding = _mapping(value, label=label)
    _exact_keys(binding, {"byte_count", "path", "sha256"}, label=label)
    relative_path = binding["path"]
    if expected_path is not None and relative_path != expected_path:
        raise ValueError(f"{label} path drifted")
    expected_sha256 = _lower_sha256(binding["sha256"], label=f"{label} SHA256")
    expected_bytes = binding["byte_count"]
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError(f"{label} byte_count must be a positive integer")
    payload = _safe_repository_file(
        root,
        relative_path,
        label=label,
    ).read_bytes()
    if len(payload) != expected_bytes or sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"{label} byte binding drifted")
    return payload


def _load_bound_json(
    root: Path,
    value: Any,
    *,
    label: str,
    expected_path: str,
) -> tuple[dict[str, Any], str]:
    payload = _validate_file_binding(
        root,
        value,
        label=label,
        expected_path=expected_path,
    )
    return _strict_json_object(payload, label=label), sha256_bytes(payload)


def _teacher_source_hash(
    freeze_manifest: Mapping[str, Any],
    *,
    expected_path: str,
) -> str:
    teacher = _mapping(freeze_manifest.get("teacher"), label="Freeze-B v2 teacher")
    source_files = _sequence(
        teacher.get("source_files"), label="Freeze-B v2 teacher source files"
    )
    matches: list[str] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(source_files):
        source = _mapping(raw, label=f"Freeze-B v2 teacher source {index}")
        _exact_keys(source, {"path", "sha256"}, label="Freeze-B v2 teacher source")
        path = source.get("path")
        if not isinstance(path, str) or path in seen_paths:
            raise ValueError("Freeze-B v2 teacher source path inventory drifted")
        seen_paths.add(path)
        digest = _lower_sha256(
            source.get("sha256"), label="Freeze-B v2 teacher source SHA256"
        )
        if path == expected_path:
            matches.append(digest)
    if len(matches) != 1:
        raise ValueError(f"Freeze-B v2 teacher must bind {expected_path} exactly once")
    return matches[0]


def _validate_input_hash_chain(
    *,
    freeze_manifest: Mapping[str, Any],
    input_documents: Mapping[str, Mapping[str, Any]],
    input_sha256: Mapping[str, str],
) -> None:
    if set(input_documents) != set(REQUIRED_FROZEN_INPUT_PATHS) or set(
        input_sha256
    ) != set(REQUIRED_FROZEN_INPUT_PATHS):
        raise ValueError("frozen input hash-chain inventory drifted")

    p0 = input_documents["p0_census_manifest"]
    inventory_manifest = input_documents["full_pool_inventory_manifest"]
    base_config = input_documents["independent_reference_gate_base_config"]
    snapshot = input_documents["teacher_snapshot_manifest"]
    ocr_config = input_documents["ocr_backend_config"]
    ocr_completion = input_documents["ocr_backend_completion_manifest"]

    freeze_source = _mapping(
        freeze_manifest.get("source"), label="Freeze-B v2 source"
    )
    if (
        freeze_source.get("p0_census_manifest_path") != P0_CENSUS_MANIFEST_PATH
        or freeze_source.get("p0_census_manifest_sha256")
        != input_sha256["p0_census_manifest"]
    ):
        raise ValueError("Freeze-B v2 to live P0 census hash chain drifted")

    if (
        p0.get("schema_version") != "1.0.0"
        or p0.get("protocol_id") != "causalcache_set_utility_full_pool_census_v2"
        or p0.get("status")
        != "POLICY_BLIND_UNCONSUMED_FULL_POOL_CENSUS_COMPLETED"
    ):
        raise ValueError("P0 census identity drifted")
    p0_source = _mapping(p0.get("source"), label="P0 census source")
    if (
        p0_source.get("source_manifest_sha256")
        != input_sha256["full_pool_inventory_manifest"]
        or p0_source.get("base_config_sha256")
        != input_sha256["independent_reference_gate_base_config"]
    ):
        raise ValueError("P0 census input hash chain drifted")

    if (
        inventory_manifest.get("schema_version") != "1.0.0"
        or inventory_manifest.get("protocol_id")
        != "causalcache_set_utility_full_pool_inventory_v1"
        or inventory_manifest.get("status")
        != "COMPLETE_METADATA_ONLY_FULL_POOL_INVENTORY_V1"
    ):
        raise ValueError("full-pool inventory identity drifted")
    inventory_source = _mapping(
        inventory_manifest.get("source"), label="full-pool inventory source"
    )
    inventory = _mapping(
        inventory_manifest.get("inventory"), label="full-pool inventory"
    )
    _exact_keys(
        inventory,
        {"file_count", "files", "files_sha256", "total_size_bytes"},
        label="full-pool inventory",
    )
    raw_files = _sequence(inventory.get("files"), label="full-pool inventory files")
    files: list[dict[str, Any]] = []
    paths: list[str] = []
    total_size_bytes = 0
    for index, raw in enumerate(raw_files):
        record = _mapping(raw, label=f"full-pool inventory file {index}")
        _exact_keys(
            record,
            {"lfs_sha256", "path", "size_bytes"},
            label=f"full-pool inventory file {index}",
        )
        path = record.get("path")
        size = record.get("size_bytes")
        digest = record.get("lfs_sha256")
        if (
            not isinstance(path, str)
            or not path.startswith("mobile/use/train/")
            or type(size) is not int
            or size <= 0
        ):
            raise ValueError("full-pool inventory file identity drifted")
        _lower_sha256(digest, label="full-pool inventory LFS SHA256")
        paths.append(path)
        total_size_bytes += size
        files.append(dict(record))
    if paths != sorted(set(paths)):
        raise ValueError("full-pool inventory file path order drifted")
    if (
        inventory.get("file_count") != len(files)
        or inventory.get("total_size_bytes") != total_size_bytes
        or inventory.get("files_sha256")
        != sha256_bytes(_canonical_compact_json_bytes(files))
    ):
        raise ValueError("full-pool inventory file or size digest drifted")

    if base_config.get("protocol_id") != "independent_reference_gate_v1":
        raise ValueError("independent reference-gate base config identity drifted")
    base_source = _mapping(
        base_config.get("source_pool"), label="independent base source pool"
    )
    shared_source_identity = {
        "repo_id": base_source.get("transport_repo"),
        "repo_type": "dataset",
        "revision": base_source.get("transport_revision"),
    }
    for key, expected in shared_source_identity.items():
        if inventory_source.get(key) != expected or p0_source.get(key) != expected:
            raise ValueError(f"inventory/census {key} identity drifted")
    if (
        inventory_source.get("target_directory")
        != p0_source.get("target_directory")
        or inventory_source.get("target_directory") != "mobile/use/train"
    ):
        raise ValueError("inventory/census target directory drifted")
    base_transport_files = _sequence(
        base_source.get("transport_files"), label="independent base transport files"
    )
    if (
        len(set(base_transport_files)) != len(base_transport_files)
        or any(path not in set(paths) for path in base_transport_files)
    ):
        raise ValueError("independent base transport files left the full inventory")

    row_counts = _mapping(
        p0_source.get("row_counts_by_file"), label="P0 row counts by file"
    )
    if list(row_counts) != paths:
        raise ValueError("inventory file paths and P0 row-count keys drifted")
    if any(type(count) is not int or count <= 0 for count in row_counts.values()):
        raise ValueError("P0 row count values drifted")
    if (
        p0_source.get("file_count") != len(files)
        or p0_source.get("total_size_bytes") != total_size_bytes
        or p0_source.get("total_rows") != sum(row_counts.values())
        or p0_source.get("source_inventory_files_sha256")
        != inventory.get("files_sha256")
    ):
        raise ValueError("inventory/census file, size, or row aggregate drifted")

    p0_selection = _mapping(p0.get("selection"), label="P0 census selection")
    if (
        freeze_source.get("p0_candidate_count")
        != p0_selection.get("candidate_count")
        or freeze_source.get("p0_candidate_inventory_sha256")
        != p0_selection.get("candidate_inventory_sha256")
    ):
        raise ValueError("Freeze-B v2 P0 candidate identity drifted")

    teacher = _mapping(freeze_manifest.get("teacher"), label="Freeze-B v2 teacher")
    teacher_inputs = {
        SNAPSHOT_MANIFEST_PATH: "teacher_snapshot_manifest",
        OCR_BACKEND_CONFIG_PATH: "ocr_backend_config",
        OCR_BACKEND_COMPLETION_MANIFEST_PATH: "ocr_backend_completion_manifest",
    }
    for path, input_name in teacher_inputs.items():
        if _teacher_source_hash(freeze_manifest, expected_path=path) != input_sha256[
            input_name
        ]:
            raise ValueError(f"Freeze-B v2 teacher hash chain drifted for {path}")
    if (
        snapshot.get("repo") != teacher.get("policy_repo")
        or snapshot.get("revision") != teacher.get("policy_revision")
    ):
        raise ValueError("Freeze-B v2 teacher snapshot identity drifted")
    if (
        ocr_config.get("schema_version") != "1.0.0"
        or ocr_config.get("protocol_id") != "causalcache_restoration_v2"
        or ocr_completion.get("schema_version") != "1.0.0"
        or ocr_completion.get("protocol_id") != "causalcache_restoration_v2"
        or ocr_completion.get("status")
        != "hf_model_immutable_verified_synthetic_and_real_screen_golden_passed"
    ):
        raise ValueError("OCR config or completion identity drifted")
    completion_sources = _sequence(
        ocr_completion.get("source_files"), label="OCR completion source files"
    )
    completion_config_hashes = [
        _lower_sha256(
            _mapping(value, label="OCR completion source").get("sha256"),
            label="OCR completion source SHA256",
        )
        for value in completion_sources
        if _mapping(value, label="OCR completion source").get("path")
        == OCR_BACKEND_CONFIG_PATH
    ]
    if completion_config_hashes != [input_sha256["ocr_backend_config"]]:
        raise ValueError("OCR completion to config hash chain drifted")
    ocr_artifact = _mapping(
        ocr_config.get("hf_model_artifact"), label="OCR config HF artifact"
    )
    completion_artifact = _mapping(
        ocr_completion.get("hf_model_artifact"), label="OCR completion HF artifact"
    )
    if (
        ocr_artifact.get("repo") != completion_artifact.get("repo")
        or ocr_artifact.get("repo_type") != completion_artifact.get("repo_type")
        or not isinstance(completion_artifact.get("immutable_revision"), str)
    ):
        raise ValueError("OCR completion model artifact identity drifted")


def _expected_selection() -> dict[str, Any]:
    return {
        "evaluation_access": (
            "raw_input_and_candidate_topology_only_without_label_outcome_or_utility"
        ),
        "query_state_count": 2400,
        "role_query_state_counts": {
            "evaluation": 200,
            "train": 2000,
            "tune": 200,
        },
        "role_trajectory_counts": {
            "evaluation": 100,
            "train": 1000,
            "tune": 100,
        },
        "row_scope": "only_rows_named_by_the_frozen_v2_assignments",
        "trajectory_count": 1200,
    }


def _expected_phases() -> dict[str, Any]:
    return {
        "auto_processor": {
            "context_limit_tokens": 32768,
            "device": "cpu",
            "drop_rule": "drop_one_oldest_candidate_until_full_reference_fits",
            "initial_candidate_rule": (
                "newest_16_non_current_equivalent_history_events"
            ),
            "interface": "transformers.AutoProcessor_only",
            "maximum_final_candidate_count": 16,
            "minimum_final_candidate_count": 4,
            "model_or_policy_forward_count": 0,
            "reserved_action_tokens": 256,
        },
        "raw_decode_and_ocr": {
            "ocr_backend": "frozen_restoration_v2_ocr_backend",
            "python_environment": "cli_bound_and_may_differ_from_auto_processor",
            "raw_payload_retention_scope": "selected_rows_only",
            "row_identity": "transport_file_plus_transport_row_index",
        },
        "schedule_materialization": {
            "exact_subset_cardinalities": [0, 1, 2],
            "materialize_exact_operation_budget": True,
            "materialize_final_candidate_sets": True,
            "materialize_four_worker_schedule": True,
            "worker_count": 4,
        },
        "throughput_pilot": {
            "policy_forward_allowed": False,
            "status": "EXCLUDED_REQUIRES_SEPARATE_POST_FREEZE_TRAIN_ONLY_CONTRACT",
        },
    }


def _expected_runtime_cli() -> dict[str, Any]:
    return {
        "ambient_environment_as_science_input_allowed": False,
        "ocr_and_processor_may_use_distinct_python_environments": True,
        "paths_must_be_explicit_cli_arguments": True,
        "python_executables_must_be_absolute_with_regular_executable_targets": True,
        "required_identity_arguments": list(REQUIRED_IDENTITY_ARGUMENTS),
        "required_path_arguments": list(REQUIRED_PATH_ARGUMENTS),
        "required_version_arguments_by_phase": {
            key: list(value)
            for key, value in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.items()
        },
        "runner_path": RUNNER_PATH,
        "versions_must_be_explicit_cli_arguments": True,
    }


def _expected_output() -> dict[str, Any]:
    return {
        "atomic_publish_required": True,
        "canonical_json": True,
        "existing_output_root_allowed": False,
        "manifest_name": "manifest.json",
        "output_root_argument": "--output-root",
        "output_root_must_be_absolute": True,
        "output_root_must_be_outside_repository": True,
        "overwrite_allowed": False,
        "role_partitioning_required": True,
    }


def _expected_authorization() -> dict[str, bool]:
    allowed = {
        "apply_auto_processor_allowed",
        "decode_selected_raw_rows_allowed",
        "freeze_recent_suffix_candidates_allowed",
        "load_auto_processor_allowed",
        "materialize_exact_operation_budget_allowed",
        "materialize_four_worker_schedule_allowed",
        "read_frozen_v2_manifest_allowed",
        "read_selected_raw_shards_allowed",
        "run_ocr_allowed",
        "write_external_processor_freeze_artifact_allowed",
        "write_ocr_low_fidelity_substrate_allowed",
    }
    all_keys = allowed | {
        "access_outcome_or_utility_allowed",
        "access_sealed_androidworld_test_allowed",
        "auto_model_load_allowed",
        "closed_loop_allowed",
        "generate_restoration_labels_allowed",
        "hugging_face_mutation_allowed",
        "kl_measurement_allowed",
        "one_shot_evaluation_label_access_allowed",
        "policy_or_vision_forward_allowed",
        "policy_throughput_pilot_allowed",
        "predictor_training_allowed",
        "teacher_forced_action_forward_allowed",
        "unselected_row_semantic_use_allowed",
    }
    return {key: key in allowed for key in sorted(all_keys)}


def _validate_freeze_b_v2_manifest(payload: bytes) -> Mapping[str, Any]:
    if len(payload) != FREEZE_B_V2_MANIFEST_BYTE_COUNT:
        raise ValueError("Freeze-B v2 manifest byte count drifted")
    if sha256_bytes(payload) != FREEZE_B_V2_MANIFEST_SHA256:
        raise ValueError("Freeze-B v2 manifest SHA256 drifted")
    manifest = _strict_json_object(payload, label="Freeze-B v2 manifest")
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("protocol_id") != FREEZE_B_V2_PROTOCOL_ID
        or manifest.get("status") != FREEZE_B_V2_STATUS
    ):
        raise ValueError("Freeze-B v2 manifest identity drifted")
    summary = _mapping(manifest.get("summary"), label="Freeze-B v2 summary")
    expected = _expected_selection()
    for key in (
        "trajectory_count",
        "query_state_count",
        "role_trajectory_counts",
        "role_query_state_counts",
    ):
        if summary.get(key) != expected[key]:
            raise ValueError(f"Freeze-B v2 {key} drifted")
    if (
        summary.get("contains_raw_instruction") is not False
        or summary.get("processor_candidate_freeze_status")
        != "PENDING_SEPARATE_EXECUTION"
        or summary.get("exact_operation_budget_status")
        != "PENDING_PROCESSOR_CANDIDATE_FREEZE"
    ):
        raise ValueError("Freeze-B v2 candidate-freeze boundary drifted")
    if len(_sequence(manifest.get("assignments"), label="Freeze-B assignments")) != 1200:
        raise ValueError("Freeze-B v2 assignment denominator drifted")
    queries = _sequence(manifest.get("query_states"), label="Freeze-B query states")
    if len(queries) != 2400:
        raise ValueError("Freeze-B v2 query denominator drifted")
    source_counts: Counter[str] = Counter()
    for value in queries:
        query = _mapping(value, label="Freeze-B query state")
        source_id = query.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Freeze-B v2 query source identity drifted")
        source_counts[source_id] += 1
        if query.get("processor_candidate_freeze_status") != (
            "PENDING_SEPARATE_EXECUTION"
        ):
            raise ValueError("Freeze-B v2 query was already processor-consumed")
    if set(source_counts.values()) != {2} or len(source_counts) != 1200:
        raise ValueError("Freeze-B v2 states-per-trajectory drifted")
    throughput = _mapping(
        manifest.get("throughput_pilot"), label="Freeze-B throughput pilot"
    )
    if throughput.get("policy_throughput_requires_separate_post_freeze_contract") is not True:
        raise ValueError("Freeze-B v2 throughput separation drifted")
    return manifest


def build_execution_config_skeleton(*, repository_root: str | Path) -> dict[str, Any]:
    """Build, but never write, a byte-bound future Execution-CF config."""
    root = Path(repository_root).resolve()
    freeze_binding = _file_binding(root, FREEZE_B_V2_MANIFEST_PATH)
    if freeze_binding != {
        "byte_count": FREEZE_B_V2_MANIFEST_BYTE_COUNT,
        "path": FREEZE_B_V2_MANIFEST_PATH,
        "sha256": FREEZE_B_V2_MANIFEST_SHA256,
    }:
        raise ValueError("Freeze-B v2 manifest binding drifted")
    freeze_manifest = _validate_freeze_b_v2_manifest(
        (root / FREEZE_B_V2_MANIFEST_PATH).read_bytes()
    )
    frozen_input_bindings = {
        name: _file_binding(root, path)
        for name, path in REQUIRED_FROZEN_INPUT_PATHS.items()
    }
    frozen_input_documents: dict[str, Mapping[str, Any]] = {}
    frozen_input_sha256: dict[str, str] = {}
    for name, path in REQUIRED_FROZEN_INPUT_PATHS.items():
        payload = (root / path).read_bytes()
        frozen_input_documents[name] = _strict_json_object(
            payload, label=f"frozen input {name}"
        )
        frozen_input_sha256[name] = sha256_bytes(payload)
    _validate_input_hash_chain(
        freeze_manifest=freeze_manifest,
        input_documents=frozen_input_documents,
        input_sha256=frozen_input_sha256,
    )
    source_bindings = [
        _file_binding(root, path) for path in REQUIRED_REPOSITORY_SOURCE_PATHS
    ]
    return {
        "authorization": _expected_authorization(),
        "bindings": {
            "freeze_b_v2_manifest": freeze_binding,
            "frozen_inputs": frozen_input_bindings,
            "repository_sources": source_bindings,
        },
        "output": _expected_output(),
        "phases": _expected_phases(),
        "protocol_id": PROTOCOL_ID,
        "runtime_cli": _expected_runtime_cli(),
        "schema_version": SCHEMA_VERSION,
        "selection": _expected_selection(),
        "status": STATUS,
    }


@dataclass(frozen=True)
class ProcessorFreezeExecutionContract:
    data: Mapping[str, Any]
    repository_root: Path
    config_sha256: str
    freeze_b_v2_manifest: Mapping[str, Any]


def validate_execution_config(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> ProcessorFreezeExecutionContract:
    root = Path(repository_root).resolve()
    _exact_keys(
        config,
        {
            "authorization",
            "bindings",
            "output",
            "phases",
            "protocol_id",
            "runtime_cli",
            "schema_version",
            "selection",
            "status",
        },
        label="processor-freeze execution config",
    )
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("protocol_id") != PROTOCOL_ID
        or config.get("status") != STATUS
    ):
        raise ValueError("processor-freeze execution config identity drifted")

    bindings = _mapping(config.get("bindings"), label="execution bindings")
    _exact_keys(
        bindings,
        {"freeze_b_v2_manifest", "frozen_inputs", "repository_sources"},
        label="execution bindings",
    )
    freeze_binding = _mapping(
        bindings["freeze_b_v2_manifest"], label="Freeze-B v2 binding"
    )
    if dict(freeze_binding) != {
        "byte_count": FREEZE_B_V2_MANIFEST_BYTE_COUNT,
        "path": FREEZE_B_V2_MANIFEST_PATH,
        "sha256": FREEZE_B_V2_MANIFEST_SHA256,
    }:
        raise ValueError("Freeze-B v2 manifest binding drifted")
    freeze_payload = _validate_file_binding(
        root,
        freeze_binding,
        label="Freeze-B v2 binding",
        expected_path=FREEZE_B_V2_MANIFEST_PATH,
    )
    freeze_manifest = _validate_freeze_b_v2_manifest(freeze_payload)

    frozen_inputs = _mapping(
        bindings["frozen_inputs"], label="frozen input bindings"
    )
    _exact_keys(
        frozen_inputs,
        set(REQUIRED_FROZEN_INPUT_PATHS),
        label="frozen input bindings",
    )
    frozen_input_documents: dict[str, Mapping[str, Any]] = {}
    frozen_input_sha256: dict[str, str] = {}
    for name, path in REQUIRED_FROZEN_INPUT_PATHS.items():
        document, digest = _load_bound_json(
            root,
            frozen_inputs[name],
            label=f"frozen input {name}",
            expected_path=path,
        )
        frozen_input_documents[name] = document
        frozen_input_sha256[name] = digest
    _validate_input_hash_chain(
        freeze_manifest=freeze_manifest,
        input_documents=frozen_input_documents,
        input_sha256=frozen_input_sha256,
    )

    source_values = _sequence(
        bindings["repository_sources"], label="repository source bindings"
    )
    if len(source_values) != len(REQUIRED_REPOSITORY_SOURCE_PATHS):
        raise ValueError("repository source binding count drifted")
    observed_paths: list[str] = []
    for index, value in enumerate(source_values):
        binding = _mapping(value, label=f"repository source binding {index}")
        expected_path = REQUIRED_REPOSITORY_SOURCE_PATHS[index]
        _validate_file_binding(
            root,
            binding,
            label=f"repository source binding {index}",
            expected_path=expected_path,
        )
        observed_paths.append(str(binding["path"]))
    if tuple(observed_paths) != REQUIRED_REPOSITORY_SOURCE_PATHS:
        raise ValueError("repository source path inventory drifted")

    expected_sections = {
        "selection": _expected_selection(),
        "phases": _expected_phases(),
        "runtime_cli": _expected_runtime_cli(),
        "output": _expected_output(),
        "authorization": _expected_authorization(),
    }
    for name, expected in expected_sections.items():
        actual = _mapping(config.get(name), label=name)
        if dict(actual) != expected:
            raise ValueError(f"processor-freeze {name} contract drifted")

    config_sha256 = sha256_bytes(canonical_pretty_json_bytes(dict(config)))
    return ProcessorFreezeExecutionContract(
        data=config,
        repository_root=root,
        config_sha256=config_sha256,
        freeze_b_v2_manifest=freeze_manifest,
    )


def load_execution_contract(
    *,
    repository_root: str | Path,
    execution_config_path: str | Path,
) -> ProcessorFreezeExecutionContract:
    root = Path(repository_root).resolve()
    supplied = Path(execution_config_path)
    path = supplied if supplied.is_absolute() else root / supplied
    expected = (root / CANONICAL_EXECUTION_CONFIG_PATH).resolve()
    if path.resolve() != expected:
        raise ValueError(
            f"execution config must be the canonical repository path {expected}"
        )
    if path.is_symlink() or not path.is_file():
        raise ValueError("processor-freeze execution config must be a regular file")
    payload = path.read_bytes()
    config = _strict_json_object(payload, label="processor-freeze execution config")
    if payload != canonical_pretty_json_bytes(config):
        raise ValueError("processor-freeze execution config must be canonical pretty JSON")
    contract = validate_execution_config(config, repository_root=root)
    return ProcessorFreezeExecutionContract(
        data=contract.data,
        repository_root=contract.repository_root,
        config_sha256=sha256_bytes(payload),
        freeze_b_v2_manifest=contract.freeze_b_v2_manifest,
    )


def validate_runtime_cli_values(
    contract: ProcessorFreezeExecutionContract,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate explicit runner CLI values without importing either runtime."""
    expected_keys = {
        argument.removeprefix("--").replace("-", "_")
        for argument in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }
    expected_keys.update(
        argument.removeprefix("--").replace("-", "_")
        for arguments in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.values()
        for argument in arguments
    )
    if set(values) != expected_keys:
        raise ValueError("processor-freeze runtime CLI fields drifted")

    root = contract.repository_root.resolve()
    path_keys = {
        argument.removeprefix("--").replace("-", "_")
        for argument in REQUIRED_PATH_ARGUMENTS
    }
    paths: dict[str, Path] = {}
    for key in path_keys:
        raw = values[key]
        if not isinstance(raw, (str, Path)) or not Path(raw).is_absolute():
            raise ValueError(f"--{key.replace('_', '-')} must be an explicit absolute path")
        paths[key] = Path(raw)
    if paths["repository_root"].resolve() != root:
        raise ValueError("--repository-root differs from the validated contract root")
    expected_config = (root / CANONICAL_EXECUTION_CONFIG_PATH).resolve()
    if paths["execution_config"].resolve() != expected_config:
        raise ValueError("--execution-config differs from the canonical config")
    expected_snapshot = (root / SNAPSHOT_MANIFEST_PATH).resolve()
    if paths["snapshot_manifest"].resolve() != expected_snapshot:
        raise ValueError("--snapshot-manifest differs from the byte-bound repository file")
    for key in ("ocr_python_executable", "processor_python_executable"):
        executable = paths[key]
        if (
            not executable.exists()
            or not executable.resolve().is_file()
            or not os.access(executable, os.X_OK)
        ):
            raise ValueError(
                f"--{key.replace('_', '-')} must resolve to one regular executable file"
            )
    for key in ("source_root", "model_dir", "ocr_model_dir", "ocr_wheel_dir"):
        if not paths[key].is_dir():
            raise ValueError(f"--{key.replace('_', '-')} must be one existing directory")
    if not paths["snapshot_manifest"].is_file():
        raise ValueError("--snapshot-manifest must be one existing regular file")

    output = paths["output_root"]
    if not output.parent.is_dir():
        raise ValueError("--output-root parent directory must already exist")
    if output.exists() or output.is_symlink():
        raise FileExistsError("--output-root already exists; overwrite is forbidden")
    resolved_output = output.resolve()
    protected = [
        root,
        paths["source_root"].resolve(),
        paths["model_dir"].resolve(),
        paths["ocr_model_dir"].resolve(),
        paths["ocr_wheel_dir"].resolve(),
    ]
    for protected_root in protected:
        if resolved_output == protected_root or protected_root in resolved_output.parents:
            raise ValueError("--output-root must be outside all protected input roots")

    worker_count = values["worker_count"]
    if type(worker_count) is not int or worker_count != 4:
        raise ValueError("--worker-count must equal the frozen four-worker schedule")
    git_revision = values["git_revision"]
    if not isinstance(git_revision, str) or _GIT_REVISION.fullmatch(git_revision) is None:
        raise ValueError("--git-revision must be one full lowercase Git commit")
    string_keys = expected_keys - path_keys - {"worker_count", "git_revision"}
    for key in string_keys:
        value = values[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"--{key.replace('_', '-')} must be explicit and non-empty")

    return {
        "config_sha256": contract.config_sha256,
        "ocr_and_processor_python_may_differ": True,
        "ocr_python_executable": str(paths["ocr_python_executable"]),
        "output_overwrite_allowed": False,
        "output_root": str(output),
        "processor_python_executable": str(paths["processor_python_executable"]),
        "required_worker_count": 4,
        "status": "VALID_PROCESSOR_FREEZE_RUNTIME_CLI",
    }


def validation_summary(contract: ProcessorFreezeExecutionContract) -> dict[str, Any]:
    authorization = _mapping(
        contract.data["authorization"], label="processor-freeze authorization"
    )
    return {
        "config_sha256": contract.config_sha256,
        "forbidden_authorization_count": sum(
            value is False for value in authorization.values()
        ),
        "freeze_b_v2_manifest_sha256": FREEZE_B_V2_MANIFEST_SHA256,
        "frozen_input_binding_count": len(REQUIRED_FROZEN_INPUT_PATHS),
        "policy_or_vision_forward_authorized": False,
        "repository_source_binding_count": len(REQUIRED_REPOSITORY_SOURCE_PATHS),
        "required_runner_cli": {
            "identity": list(REQUIRED_IDENTITY_ARGUMENTS),
            "paths": list(REQUIRED_PATH_ARGUMENTS),
            "versions_by_phase": {
                key: list(value)
                for key, value in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.items()
            },
        },
        "status": VALIDATION_STATUS,
        "throughput_pilot_authorized": False,
    }


__all__ = [
    "CANONICAL_EXECUTION_CONFIG_PATH",
    "FREEZE_B_V2_MANIFEST_PATH",
    "FREEZE_B_V2_MANIFEST_SHA256",
    "FULL_POOL_INVENTORY_MANIFEST_PATH",
    "INDEPENDENT_REFERENCE_GATE_CONFIG_PATH",
    "OCR_BACKEND_COMPLETION_MANIFEST_PATH",
    "P0_CENSUS_MANIFEST_PATH",
    "ProcessorFreezeExecutionContract",
    "REQUIRED_FROZEN_INPUT_PATHS",
    "REQUIRED_IDENTITY_ARGUMENTS",
    "REQUIRED_PATH_ARGUMENTS",
    "REQUIRED_REPOSITORY_SOURCE_PATHS",
    "REQUIRED_VERSION_ARGUMENTS_BY_PHASE",
    "RUNNER_PATH",
    "build_execution_config_skeleton",
    "canonical_pretty_json_bytes",
    "load_execution_contract",
    "sha256_bytes",
    "validate_execution_config",
    "validate_runtime_cli_values",
    "validation_summary",
]
