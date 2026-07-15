"""Deterministic raw packaging and independent validation for the v2.1 pilot."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.restoration_v2_screening import ScreeningState
from causalcache.policy.gui_owl_v2_1 import GUI_OWL_V2_1_PROTOCOL_ID
from scripts.run_restoration_v2_1_interface_pilot import (
    AGGREGATE_FILENAME,
    ATTEMPT_DIRECTORY,
    ATTEMPT_STATUS,
    CANONICAL_PILOT_CONTAINER_ID,
    CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
    CANONICAL_PILOT_DEVICE,
    CANONICAL_PILOT_HOST_ALIAS,
    CANONICAL_PILOT_HOST_HOSTNAME,
    CANONICAL_REMOTE_URL,
    EXPECTED_STATE_COUNT,
    GLOBAL_ATTEMPT_STATUS,
    INVALID_OUTCOME,
    NEGATIVE_OPERATION_COUNTS,
    NO_GO_OUTCOME,
    PASS_OUTCOME,
    PROTOCOL_ID,
    RUN_MANIFEST_FILENAME,
    RUN_STATUS,
    RUNTIME_GENERATION_BINDING_KEYS,
    RUNTIME_IDENTITY_FILENAME,
    RUNTIME_METADATA_KEYS,
    SCHEMA_VERSION as PILOT_SCHEMA_VERSION,
    SOURCE_STATE_START,
    STATE_DIRECTORY,
    PilotGate,
    _pilot_projection,
    _duration_seconds,
    _validate_invalid_aggregate,
    _validate_runtime_identity_record,
    _validate_terminal_state_record,
    aggregate_pilot_gate,
    validate_clean_pushed_main,
)


ARTIFACT_SCHEMA_VERSION = "0.1.0"
ARTIFACT_TYPE = "private_hf_restoration_v2_1_interface_pilot_raw_evidence"
CANONICAL_ATTEMPT_ID = "restoration-v2-1-interface-pilot-v1"
CANONICAL_RAW_OUTPUT_DIR = Path(
    "/data/experiments/causalcache/restoration-v2-1-interface-pilot-v1"
)
CANONICAL_GLOBAL_LEDGER_PATH = Path(
    "/data/experiments/causalcache/"
    ".restoration-v2-1-interface-pilot-v1.attempt.json"
)
CANONICAL_RAW_ARCHIVE_PATH = Path(
    "/data/experiments/causalcache/restoration-v2-1-interface-pilot-v1.raw.tar"
)
CANONICAL_GIT_ARTIFACT_PATH = (
    "data/results/restoration_v2_1_interface_pilot/artifact.json"
)
CANONICAL_HF_REPO = (
    "gavinlaw/causalcache-restoration-v2-1-interface-pilot-mobile"
)
CANONICAL_HF_TAG = "v2.1-interface-pilot-v1"
CANONICAL_HF_PATH = "raw/restoration-v2-1-interface-pilot-v1.tar"
ARCHIVE_FORMAT = "ustar"
ARCHIVE_MEMBER_PREFIX = CANONICAL_ATTEMPT_ID
ARCHIVE_LEDGER_NAME = "global_attempt_ledger.json"
V2_1_CONFIG_PATH = "code/configs/causalcache_restoration_v2_1_pilot.json"
V2_CONFIG_PATH = "code/configs/causalcache_restoration_v2.json"
SELECTION_MANIFEST_PATH = "data/manifests/restoration_v2_selection.json"
OCR_CONFIG_PATH = "code/configs/restoration_v2_ocr_backend.json"
SNAPSHOT_MANIFEST_PATH = "code/configs/gui_owl_1_5_8b_snapshot.json"
PROCESSOR_ARTIFACT_PATH = (
    "data/results/restoration_v2_1_processor_preflight/artifact.json"
)
EXPECTED_RUN_SOURCE_PATHS = (
    "code/scripts/run_restoration_v2_1_interface_pilot.py",
    "code/causalcache/restoration_v2_1_processor_audit.py",
    "code/scripts/audit_gui_owl_v2_1_processor.py",
    PROCESSOR_ARTIFACT_PATH,
    "code/causalcache/restoration_v2_1_contract.py",
    "code/causalcache/policy/gui_owl_v2_1.py",
    "code/causalcache/policy/gui_owl_v2_1_runtime.py",
    "code/causalcache/data/restoration_v2_screening.py",
    "code/causalcache/data/restoration_v2_1_processor_inputs.py",
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/causalcache/data/restoration_v2_selection.py",
    "code/causalcache/data/guiodyssey.py",
    "code/causalcache/data/guiodyssey_independent.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/restoration_v2_contract.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/schema.py",
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_runtime.py",
    "code/causalcache/policy/gui_owl_v2_vision.py",
    "code/causalcache/restoration_v2_baselines.py",
    "code/causalcache/restoration_v2_1_pilot_artifact.py",
    "code/scripts/manage_restoration_v2_1_pilot_artifact.py",
    V2_1_CONFIG_PATH,
    V2_CONFIG_PATH,
    SELECTION_MANIFEST_PATH,
    OCR_CONFIG_PATH,
    SNAPSHOT_MANIFEST_PATH,
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class PilotEvidence:
    files: Mapping[str, bytes]
    run_contract: Mapping[str, Any]
    source_git_commit: str
    run_contract_sha256: str
    outcome: str
    status: str
    aggregate_sha256: str
    run_manifest_sha256: str
    global_attempt_ledger_sha256: str
    completed_state_count: int
    attempted_state_count: int
    inventory: tuple[Mapping[str, Any], ...]
    tree_inventory_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value,
        Sequence,
    ):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def _safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} is unsafe")
    normalized = path.as_posix()
    if normalized != value or normalized.startswith("/"):
        raise ValueError(f"{label} is not canonical POSIX syntax")
    return normalized


def _fixed_gate() -> PilotGate:
    return PilotGate.from_mapping(
        {
            "fixed_state_denominator": EXPECTED_STATE_COUNT,
            "required_exact_whole_output_parse_count": EXPECTED_STATE_COUNT,
            "required_model_emitted_closer_count": EXPECTED_STATE_COUNT,
            "required_androidworld_bridge_count": EXPECTED_STATE_COUNT,
            "maximum_max_token_truncation_count": 0,
            "maximum_surrounding_prose_count": 0,
            "maximum_action_line_count": 0,
            "maximum_observation_count": 0,
            "maximum_second_json_count": 0,
            "maximum_second_tool_call_count": 0,
            "maximum_retry_count": 0,
            "maximum_top_up_count": 0,
            "pass_outcome": PASS_OUTCOME,
            "fail_outcome": NO_GO_OUTCOME,
            "invalid_outcome": INVALID_OUTCOME,
        }
    )


def _canonical_attempt_identity() -> dict[str, Any]:
    return {
        "attempt_id": CANONICAL_ATTEMPT_ID,
        "canonical_persistent_output_dir": str(CANONICAL_RAW_OUTPUT_DIR),
        "global_attempt_ledger": str(CANONICAL_GLOBAL_LEDGER_PATH),
        "canonical_host_alias": CANONICAL_PILOT_HOST_ALIAS,
        "canonical_host_hostname": CANONICAL_PILOT_HOST_HOSTNAME,
        "canonical_container_id": CANONICAL_PILOT_CONTAINER_ID,
        "canonical_container_image_digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        "canonical_device": CANONICAL_PILOT_DEVICE,
        "cross_host_attempt_allowed": False,
        "alternate_output_dir_allowed": False,
        "output_directory_deletion_after_first_attempt_allowed": False,
    }


def _validate_source_identity(
    run_contract: Mapping[str, Any],
    *,
    expected_source_git_commit: str | None,
) -> str:
    git = _mapping(run_contract.get("git_identity"), "run contract git_identity")
    _exact_keys(
        git,
        {
            "branch",
            "commit",
            "origin_main",
            "remote_main",
            "remote_url",
            "worktree",
        },
        "run contract git_identity",
    )
    commit = git.get("commit")
    if (
        not isinstance(commit, str)
        or GIT_SHA_PATTERN.fullmatch(commit) is None
        or git.get("branch") != "main"
        or git.get("origin_main") != commit
        or git.get("remote_main") != commit
        or git.get("remote_url") != CANONICAL_REMOTE_URL
        or git.get("worktree") != "clean_including_untracked"
    ):
        raise ValueError("run contract clean pushed Git identity drifted")
    if expected_source_git_commit is not None and commit != expected_source_git_commit:
        raise ValueError("pilot source Git commit differs from the expected commit")
    inventory = _sequence(
        run_contract.get("source_inventory"),
        "run contract source_inventory",
    )
    if not inventory:
        raise ValueError("run contract source inventory is empty")
    paths: list[str] = []
    for index, item in enumerate(inventory):
        record = _mapping(item, f"run contract source_inventory[{index}]")
        _exact_keys(record, {"path", "sha256", "git_commit"}, "source record")
        path = _safe_relative_path(record.get("path"), label="source record path")
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            or record.get("git_commit") != commit
        ):
            raise ValueError("run contract source record binding drifted")
        paths.append(path)
    if paths != list(EXPECTED_RUN_SOURCE_PATHS):
        raise ValueError("run contract source inventory path/order drifted")
    return commit


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA256")
    return value


def _validate_processor_audit(value: Any, *, source_git_commit: str) -> None:
    audit = _mapping(value, "processor_audit")
    _exact_keys(
        audit,
        {
            "status",
            "prompt_count",
            "prompt_records_sha256",
            "shape_records_sha256",
            "teacher_golden_records_sha256",
            "processor_classes_sha256",
            "evidence_git_commit",
            "current_git_commit",
            "validation_mode",
        },
        "processor_audit",
    )
    if (
        audit.get("status")
        != "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT"
        or audit.get("prompt_count") != 90
        or audit.get("current_git_commit") != source_git_commit
        or audit.get("validation_mode") != "reuse"
        or not isinstance(audit.get("evidence_git_commit"), str)
        or GIT_SHA_PATTERN.fullmatch(audit["evidence_git_commit"]) is None
    ):
        raise ValueError("processor audit authorization drifted")
    for name in (
        "prompt_records_sha256",
        "shape_records_sha256",
        "teacher_golden_records_sha256",
        "processor_classes_sha256",
    ):
        _require_sha256(audit.get(name), name=f"processor_audit.{name}")


def _validate_canonical_inputs(value: Any, *, contract_sha256: str) -> None:
    inputs = _mapping(value, "canonical_inputs")
    _exact_keys(
        inputs,
        {
            "contract",
            "processor_preflight",
            "scientific_config",
            "selection_manifest",
            "ocr_backend_config",
            "snapshot_manifest",
        },
        "canonical_inputs",
    )
    expected_paths = {
        "contract": V2_1_CONFIG_PATH,
        "scientific_config": V2_CONFIG_PATH,
        "selection_manifest": SELECTION_MANIFEST_PATH,
        "ocr_backend_config": OCR_CONFIG_PATH,
        "snapshot_manifest": SNAPSHOT_MANIFEST_PATH,
    }
    for label, path in expected_paths.items():
        record = _mapping(inputs.get(label), f"canonical_inputs.{label}")
        _exact_keys(record, {"path", "sha256"}, f"canonical_inputs.{label}")
        if record.get("path") != path:
            raise ValueError(f"canonical input {label} path drifted")
        digest = _require_sha256(
            record.get("sha256"),
            name=f"canonical_inputs.{label}.sha256",
        )
        if label == "contract" and digest != contract_sha256:
            raise ValueError("canonical contract SHA differs from run contract")
    processor = _mapping(
        inputs.get("processor_preflight"),
        "canonical_inputs.processor_preflight",
    )
    _exact_keys(
        processor,
        {
            "external_path",
            "sha256",
            "size_bytes",
            "artifact_manifest_path",
            "artifact_manifest_sha256",
            "artifact_manifest",
        },
        "canonical_inputs.processor_preflight",
    )
    external_path = processor.get("external_path")
    if (
        not isinstance(external_path, str)
        or not Path(external_path).is_absolute()
        or processor.get("artifact_manifest_path") != PROCESSOR_ARTIFACT_PATH
        or type(processor.get("size_bytes")) is not int
        or processor["size_bytes"] <= 0
        or not isinstance(processor.get("artifact_manifest"), Mapping)
    ):
        raise ValueError("canonical processor preflight binding drifted")
    _require_sha256(processor.get("sha256"), name="processor raw SHA256")
    _require_sha256(
        processor.get("artifact_manifest_sha256"),
        name="processor artifact manifest SHA256",
    )


def _validate_policy_and_artifact(run_contract: Mapping[str, Any]) -> None:
    artifact = _mapping(run_contract.get("artifact"), "artifact")
    _exact_keys(
        artifact,
        {
            "artifact_tree_sha256",
            "artifact_manifest_sha256",
            "screening_manifest_sha256",
            "derived_repo",
        },
        "artifact",
    )
    for key in (
        "artifact_tree_sha256",
        "artifact_manifest_sha256",
        "screening_manifest_sha256",
    ):
        _require_sha256(artifact.get(key), name=f"artifact.{key}")
    derived = _mapping(artifact.get("derived_repo"), "artifact.derived_repo")
    if dict(derived) != {
        "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
        "immutable_revision": "89f136abaff797e14fe758a198996e51032a10a6",
        "artifact_tree_sha256": (
            "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e"
        ),
    } or artifact.get("artifact_tree_sha256") != derived["artifact_tree_sha256"]:
        raise ValueError("derived artifact immutable identity drifted")
    policy = _mapping(run_contract.get("policy"), "policy")
    _exact_keys(
        policy,
        {"repo", "revision", "model_dir", "runtime_metadata_requirement"},
        "policy",
    )
    model_dir = policy.get("model_dir")
    if (
        policy.get("repo") != "mPLUG/GUI-Owl-1.5-8B-Instruct"
        or policy.get("revision") != "06d5faecff74840bab2be2425e9c42667a5d04fc"
        or not isinstance(model_dir, str)
        or not Path(model_dir).is_absolute()
        or Path(model_dir).name != "GUI-Owl-1.5-8B-Instruct"
    ):
        raise ValueError("pilot policy snapshot identity drifted")
    requirement = _mapping(
        policy.get("runtime_metadata_requirement"),
        "policy runtime_metadata_requirement",
    )
    expected_requirement = {
        "required_protocol_id": PROTOCOL_ID,
        "validated_after_claim_before_generation": True,
        "native_generation_metadata_persisted_per_state": True,
        "required_metadata_keys": sorted(RUNTIME_METADATA_KEYS),
        "model_dir": model_dir,
        "model_repo": policy["repo"],
        "model_revision": policy["revision"],
        "snapshot_manifest_sha256": requirement.get("snapshot_manifest_sha256"),
        "device": CANONICAL_PILOT_DEVICE,
        "target_effective_visual_tokens_per_image": 2560,
    }
    if (
        not isinstance(requirement.get("snapshot_manifest_sha256"), str)
        or SHA256_PATTERN.fullmatch(requirement["snapshot_manifest_sha256"]) is None
        or dict(requirement) != expected_requirement
    ):
        raise ValueError("pilot post-claim runtime metadata requirement drifted")


def _validate_runtime_identity(value: Any) -> None:
    runtime = _mapping(value, "runtime_identity")
    _exact_keys(
        runtime,
        {
            "declared_host",
            "verified_container_runtime",
            "declared_container_image",
            "selected_device",
            "live",
        },
        "runtime_identity",
    )
    if runtime.get("declared_host") != {
        "alias": CANONICAL_PILOT_HOST_ALIAS,
        "hostname": CANONICAL_PILOT_HOST_HOSTNAME,
        "verification": "requires_independent_host_preflight",
    } or runtime.get("declared_container_image") != {
        "digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        "verification": "requires_independent_host_preflight",
    } or runtime.get("selected_device") != CANONICAL_PILOT_DEVICE:
        raise ValueError("pilot declared host/image/device identity drifted")
    container = _mapping(
        runtime.get("verified_container_runtime"),
        "verified_container_runtime",
    )
    if (
        container.get("container_id") != CANONICAL_PILOT_CONTAINER_ID
        or container.get("verification")
        != "live_hostname_prefix_of_full_container_id"
        or not isinstance(container.get("container_hostname"), str)
        or not CANONICAL_PILOT_CONTAINER_ID.startswith(container["container_hostname"])
    ):
        raise ValueError("pilot verified container identity drifted")
    live = _mapping(runtime.get("live"), "runtime_identity.live")
    _exact_keys(
        live,
        {
            "platform_machine",
            "gpu_name",
            "gpu_uuid",
            "nvidia_smi_gpu_uuid",
            "nvidia_driver_version",
            "gpu_compute_capability",
            "gpu_multiprocessor_count",
            "visible_cuda_device_count",
            "selected_device",
            "python_version",
            "torch_version",
            "torch_cuda_build_version",
            "cudnn_version",
            "transformers_version",
        },
        "runtime_identity.live",
    )
    if (
        live.get("visible_cuda_device_count") != 1
        or live.get("selected_device") != CANONICAL_PILOT_DEVICE
        or not isinstance(live.get("gpu_name"), str)
        or "H200" not in live["gpu_name"]
        or not isinstance(live.get("gpu_uuid"), str)
        or live.get("nvidia_smi_gpu_uuid") != f"GPU-{live.get('gpu_uuid')}"
        or live.get("gpu_compute_capability") != [9, 0]
        or type(live.get("gpu_multiprocessor_count")) is not int
        or live["gpu_multiprocessor_count"] <= 0
    ):
        raise ValueError("pilot live single-H200 identity drifted")
    for name in (
        "platform_machine",
        "nvidia_driver_version",
        "python_version",
        "torch_version",
        "torch_cuda_build_version",
        "transformers_version",
    ):
        if not isinstance(live.get(name), str) or not live[name]:
            raise ValueError(f"pilot live runtime field {name} is invalid")


def _validate_execution_argv(
    value: Any,
    *,
    run_contract: Mapping[str, Any],
) -> None:
    argv = list(_sequence(value, "execution_argv"))
    if (
        len(argv) < 2
        or any(not isinstance(item, str) or not item for item in argv)
        or not Path(argv[0]).is_absolute()
        or argv[1]
        != "/data/CausalCache/code/scripts/run_restoration_v2_1_interface_pilot.py"
        or "--resume" in argv
    ):
        raise ValueError("pilot execution argv prefix or resume policy drifted")
    tokens = argv[2:]
    if len(tokens) % 2:
        raise ValueError("pilot execution argv must contain exact flag-value pairs")
    pairs = dict(zip(tokens[::2], tokens[1::2], strict=True))
    expected_flags = {
        "--repository-root",
        "--contract",
        "--processor-preflight",
        "--derived-artifact-root",
        "--scientific-config",
        "--selection-manifest",
        "--ocr-backend-config",
        "--model-dir",
        "--device",
        "--host-alias",
        "--host-hostname",
        "--container-id",
        "--container-image-digest",
        "--output-dir",
    }
    if len(pairs) != len(tokens) // 2 or set(pairs) != expected_flags:
        raise ValueError("pilot execution argv flags drifted or duplicated")
    expected_values = {
        "--repository-root": "/data/CausalCache",
        "--contract": f"/data/CausalCache/{V2_1_CONFIG_PATH}",
        "--scientific-config": f"/data/CausalCache/{V2_CONFIG_PATH}",
        "--selection-manifest": f"/data/CausalCache/{SELECTION_MANIFEST_PATH}",
        "--ocr-backend-config": f"/data/CausalCache/{OCR_CONFIG_PATH}",
        "--processor-preflight": str(
            _mapping(
                _mapping(run_contract.get("canonical_inputs"), "canonical_inputs").get(
                    "processor_preflight"
                ),
                "processor preflight",
            ).get("external_path")
        ),
        "--model-dir": str(
            _mapping(run_contract.get("policy"), "policy").get("model_dir")
        ),
        "--device": CANONICAL_PILOT_DEVICE,
        "--host-alias": CANONICAL_PILOT_HOST_ALIAS,
        "--host-hostname": CANONICAL_PILOT_HOST_HOSTNAME,
        "--container-id": CANONICAL_PILOT_CONTAINER_ID,
        "--container-image-digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        "--output-dir": str(CANONICAL_RAW_OUTPUT_DIR),
    }
    for flag, expected in expected_values.items():
        supplied = pairs[flag]
        if supplied != expected:
            raise ValueError(f"pilot execution argv {flag} drifted")
    for flag in (
        "--derived-artifact-root",
    ):
        if not Path(pairs[flag]).is_absolute():
            raise ValueError(f"pilot execution argv {flag} must be absolute")


def _validate_run_contract(
    value: Mapping[str, Any],
    *,
    expected_source_git_commit: str | None,
) -> tuple[str, tuple[ScreeningState, ...]]:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "contract_sha256",
        "git_identity",
        "source_inventory",
        "processor_audit",
        "canonical_inputs",
        "artifact",
        "policy",
        "runtime_identity",
        "execution_argv",
        "operational_argv_policy",
        "seed_policy",
        "output_dir",
        "attempt_identity",
        "states",
        "generation_plan",
        "negative_operation_counts",
        "confirm_state",
    }
    _exact_keys(value, expected_keys, "pilot run contract")
    contract_sha256 = _require_sha256(
        value.get("contract_sha256"),
        name="run contract contract_sha256",
    )
    if (
        value.get("schema_version") != PILOT_SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("output_dir") != str(CANONICAL_RAW_OUTPUT_DIR)
        or value.get("attempt_identity") != _canonical_attempt_identity()
        or value.get("negative_operation_counts") != NEGATIVE_OPERATION_COUNTS
        or value.get("confirm_state") != "LOCKED"
    ):
        raise ValueError("pilot run contract identity or lock drifted")
    generation = _mapping(value.get("generation_plan"), "generation_plan")
    if dict(generation) != {
        "fidelity": "full_history_reference",
        "generation_calls_per_state": 1,
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "automatic_retry": False,
        "top_up": False,
    }:
        raise ValueError("pilot generation plan drifted")
    commit = _validate_source_identity(
        value,
        expected_source_git_commit=expected_source_git_commit,
    )
    _validate_processor_audit(value.get("processor_audit"), source_git_commit=commit)
    _validate_canonical_inputs(
        value.get("canonical_inputs"),
        contract_sha256=contract_sha256,
    )
    _validate_policy_and_artifact(value)
    _validate_runtime_identity(value.get("runtime_identity"))
    _validate_execution_argv(value.get("execution_argv"), run_contract=value)
    if value.get("operational_argv_policy") != {
        "resume_flag_excluded_from_scientific_run_identity": True,
        "initial_invocation_is_recorded_without_resume": True,
    } or value.get("seed_policy") != {
        "decoding": "greedy_do_sample_false",
        "random_seed": None,
        "sampling_seed_not_applicable": True,
    }:
        raise ValueError("pilot argv or decoding seed policy drifted")
    projections = _sequence(value.get("states"), "pilot states")
    if len(projections) != EXPECTED_STATE_COUNT:
        raise ValueError("pilot state denominator is not fixed at 15")
    states: list[ScreeningState] = []
    for pilot_index, item in enumerate(projections):
        projection = _mapping(item, f"pilot state {pilot_index}")
        expected_projection_keys = {
            "pilot_index",
            "source_state_index",
            "role",
            "trajectory_id",
            "decision_step_id",
            "state_id",
            "candidate_event_step_ids",
        }
        _exact_keys(projection, expected_projection_keys, "pilot state projection")
        trajectory_id = projection.get("trajectory_id")
        decision_step_id = projection.get("decision_step_id")
        candidates = projection.get("candidate_event_step_ids")
        if (
            projection.get("pilot_index") != pilot_index
            or projection.get("source_state_index") != SOURCE_STATE_START + pilot_index
            or projection.get("role") != "v2_development"
            or not isinstance(trajectory_id, str)
            or not trajectory_id
            or decision_step_id not in {4, 5, 6}
            or candidates != list(range(1, int(decision_step_id) - 1))
        ):
            raise ValueError("pilot state order, role, or dependency set drifted")
        state = ScreeningState(
            index=SOURCE_STATE_START + pilot_index,
            role="v2_development",
            trajectory_id=trajectory_id,
            decision_step_id=int(decision_step_id),
            candidate_event_step_ids=tuple(candidates),
        )
        if dict(projection) != _pilot_projection(state, pilot_index):
            raise ValueError("pilot state projection is not canonical")
        states.append(state)
    return commit, tuple(states)


def _default_committed_blob_reader(
    repository_root: Path,
    source_git_commit: str,
    relative_path: str,
) -> bytes:
    return _git_bytes(
        repository_root,
        "show",
        f"{source_git_commit}:{relative_path}",
    )


def _default_ancestor_validator(
    repository_root: Path,
    ancestor: str,
    descendant: str,
) -> None:
    require_source_commit_ancestor(
        repository_root=repository_root,
        source_git_commit=ancestor,
        current_git_commit=descendant,
    )


def validate_source_x_run_contract(
    run_contract: Mapping[str, Any],
    *,
    repository_root: str | Path,
    source_git_commit: str,
    committed_blob_reader: Callable[[Path, str, str], bytes] = (
        _default_committed_blob_reader
    ),
    ancestor_validator: Callable[[Path, str, str], None] = (
        _default_ancestor_validator
    ),
) -> None:
    """Replay config, denominator, policy, and source bindings from commit X."""
    root = Path(repository_root).resolve()
    inventory = _sequence(run_contract.get("source_inventory"), "source_inventory")
    for relative, item in zip(EXPECTED_RUN_SOURCE_PATHS, inventory, strict=True):
        record = _mapping(item, f"source inventory {relative}")
        payload = committed_blob_reader(root, source_git_commit, relative)
        if (
            record.get("path") != relative
            or record.get("git_commit") != source_git_commit
            or record.get("sha256") != sha256_bytes(payload)
        ):
            raise ValueError(f"source-X committed blob binding drifted: {relative}")

    committed: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for relative in (
        V2_1_CONFIG_PATH,
        V2_CONFIG_PATH,
        SELECTION_MANIFEST_PATH,
        OCR_CONFIG_PATH,
        SNAPSHOT_MANIFEST_PATH,
        PROCESSOR_ARTIFACT_PATH,
    ):
        payload = committed_blob_reader(root, source_git_commit, relative)
        committed[relative] = (
            payload,
            strict_json_object_bytes(payload, label=f"source-X {relative}"),
        )
    v21_bytes, v21 = committed[V2_1_CONFIG_PATH]
    _, selection = committed[SELECTION_MANIFEST_PATH]
    source_contract_sha256 = sha256_bytes(v21_bytes)
    if run_contract.get("contract_sha256") != source_contract_sha256:
        raise ValueError("run contract does not bind source-X v2.1 config bytes")
    if (
        v21.get("protocol_id") != PROTOCOL_ID
        or _mapping(v21.get("pilot_execution"), "v2.1 pilot_execution").get(
            "attempt_id"
        )
        != CANONICAL_ATTEMPT_ID
        or _mapping(v21.get("pilot_execution"), "v2.1 pilot_execution").get(
            "canonical_persistent_output_dir"
        )
        != str(CANONICAL_RAW_OUTPUT_DIR)
    ):
        raise ValueError("source-X v2.1 execution identity drifted")
    PilotGate.from_mapping(_mapping(v21.get("pilot_gate"), "v2.1 pilot_gate"))

    data = _mapping(v21.get("data"), "v2.1 data")
    selection_record = _mapping(data.get("selection_manifest"), "selection record")
    selection_bytes = committed[SELECTION_MANIFEST_PATH][0]
    if selection_record != {
        "path": SELECTION_MANIFEST_PATH,
        "sha256": sha256_bytes(selection_bytes),
    }:
        raise ValueError("source-X selection manifest binding drifted")
    roles = _mapping(selection.get("roles"), "selection roles")
    development = _mapping(
        roles.get("v2_development"),
        "selection v2_development",
    )
    selected = _sequence(development.get("states"), "selection development states")
    if len(selected) != EXPECTED_STATE_COUNT:
        raise ValueError("source-X selection development denominator drifted")
    projection: list[dict[str, Any]] = []
    for pilot_index, item in enumerate(selected):
        state = _mapping(item, f"selection development state {pilot_index}")
        source_id = state.get("source_id")
        decision_step_id = state.get("decision_step_id")
        candidates = state.get("candidate_event_step_ids")
        if (
            not isinstance(source_id, str)
            or decision_step_id not in {4, 5, 6}
            or candidates != list(range(1, int(decision_step_id) - 1))
            or state.get("state_id")
            != f"{source_id}:decision_step:{int(decision_step_id):03d}"
        ):
            raise ValueError("source-X selection development state drifted")
        projection.append(
            {
                "pilot_index": pilot_index,
                "source_state_index": SOURCE_STATE_START + pilot_index,
                "role": "v2_development",
                "trajectory_id": source_id,
                "decision_step_id": int(decision_step_id),
                "state_id": state["state_id"],
                "candidate_event_step_ids": list(candidates),
            }
        )
    if (
        run_contract.get("states") != projection
        or data.get("pilot_state_count") != EXPECTED_STATE_COUNT
        or data.get("source_state_indices")
        != list(range(SOURCE_STATE_START, SOURCE_STATE_START + EXPECTED_STATE_COUNT))
        or data.get("pilot_state_ids") != [item["state_id"] for item in projection]
        or data.get("pilot_state_projection_sha256")
        != sha256_bytes(canonical_json_bytes(projection))
    ):
        raise ValueError("run denominator differs from frozen source-X projection")

    canonical_inputs = _mapping(run_contract.get("canonical_inputs"), "canonical_inputs")
    input_paths = {
        "contract": V2_1_CONFIG_PATH,
        "scientific_config": V2_CONFIG_PATH,
        "selection_manifest": SELECTION_MANIFEST_PATH,
        "ocr_backend_config": OCR_CONFIG_PATH,
        "snapshot_manifest": SNAPSHOT_MANIFEST_PATH,
    }
    for label, relative in input_paths.items():
        expected = {
            "path": relative,
            "sha256": sha256_bytes(committed[relative][0]),
        }
        if canonical_inputs.get(label) != expected:
            raise ValueError(f"run canonical input differs from source-X {label}")
    processor_input = _mapping(
        canonical_inputs.get("processor_preflight"),
        "canonical processor preflight",
    )
    processor_manifest_bytes, processor_manifest = committed[PROCESSOR_ARTIFACT_PATH]
    _exact_keys(
        processor_manifest,
        {
            "schema_version",
            "protocol_id",
            "artifact_type",
            "hf_artifact",
            "raw_evidence",
            "compact_reduction",
        },
        "processor artifact manifest",
    )
    processor_hf = _mapping(
        processor_manifest.get("hf_artifact"),
        "processor artifact hf_artifact",
    )
    _exact_keys(
        processor_hf,
        {
            "repo",
            "repo_type",
            "visibility",
            "tag",
            "immutable_revision",
            "path",
        },
        "processor artifact hf_artifact",
    )
    if (
        processor_manifest.get("protocol_id") != PROTOCOL_ID
        or processor_manifest.get("artifact_type")
        != "private_hf_processor_preflight_evidence"
        or processor_hf.get("repo")
        != "gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile"
        or processor_hf.get("repo_type") != "dataset"
        or processor_hf.get("visibility") != "private"
        or processor_hf.get("tag") != "v2.1-processor-preflight-v1"
        or not isinstance(processor_hf.get("immutable_revision"), str)
        or GIT_SHA_PATTERN.fullmatch(processor_hf["immutable_revision"]) is None
        or not _safe_relative_path(
            processor_hf.get("path"),
            label="processor artifact HF path",
        ).endswith(".json")
    ):
        raise ValueError("processor artifact immutable HF identity drifted")
    if (
        processor_input.get("artifact_manifest_path") != PROCESSOR_ARTIFACT_PATH
        or processor_input.get("artifact_manifest_sha256")
        != sha256_bytes(processor_manifest_bytes)
        or processor_input.get("artifact_manifest") != processor_manifest
    ):
        raise ValueError("run processor artifact differs from source-X Git blob")
    audit = _mapping(run_contract.get("processor_audit"), "processor_audit")
    compact = _mapping(
        processor_manifest.get("compact_reduction"),
        "processor artifact compact_reduction",
    )
    _exact_keys(
        compact,
        {
            "status",
            "contract_sha256",
            "selection_manifest_sha256",
            "policy_interface_source_sha256",
            "runtime_source_sha256",
            "state_count",
            "prompt_count",
            "official_tools_injected_prompt_count",
            "image_count_distribution",
            "context_overflow_count",
            "shape_records_sha256",
            "prompt_records_sha256",
            "teacher_golden_records_sha256",
            "processor_classes_sha256",
        },
        "processor artifact compact_reduction",
    )
    compact_pairs = {
        "status": "status",
        "prompt_count": "prompt_count",
        "prompt_records_sha256": "prompt_records_sha256",
        "shape_records_sha256": "shape_records_sha256",
        "teacher_golden_records_sha256": "teacher_golden_records_sha256",
        "processor_classes_sha256": "processor_classes_sha256",
    }
    if any(audit.get(left) != compact.get(right) for left, right in compact_pairs.items()):
        raise ValueError("run processor audit differs from committed compact reduction")
    expected_compact_bindings = {
        "contract_sha256": source_contract_sha256,
        "selection_manifest_sha256": sha256_bytes(selection_bytes),
        "policy_interface_source_sha256": sha256_bytes(
            committed_blob_reader(
                root,
                source_git_commit,
                "code/causalcache/policy/gui_owl_v2_1.py",
            )
        ),
        "runtime_source_sha256": sha256_bytes(
            committed_blob_reader(
                root,
                source_git_commit,
                "code/causalcache/policy/gui_owl_v2_1_runtime.py",
            )
        ),
        "state_count": 45,
        "prompt_count": 90,
        "official_tools_injected_prompt_count": 90,
        "image_count_distribution": {"1": 45, "3": 15, "4": 15, "5": 15},
        "context_overflow_count": 0,
    }
    if any(
        compact.get(key) != expected
        for key, expected in expected_compact_bindings.items()
    ):
        raise ValueError("processor compact source-X binding drifted")
    raw_processor = _mapping(
        processor_manifest.get("raw_evidence"),
        "processor artifact raw_evidence",
    )
    _exact_keys(
        raw_processor,
        {"sha256", "size_bytes", "source_git_commit"},
        "processor artifact raw_evidence",
    )
    _require_sha256(raw_processor.get("sha256"), name="processor raw SHA256")
    if type(raw_processor.get("size_bytes")) is not int or raw_processor["size_bytes"] <= 0:
        raise ValueError("processor raw evidence size is invalid")
    evidence_commit = audit.get("evidence_git_commit")
    if raw_processor.get("source_git_commit") != evidence_commit:
        raise ValueError("processor evidence source commit binding drifted")
    if (
        processor_input.get("sha256") != raw_processor.get("sha256")
        or processor_input.get("size_bytes") != raw_processor.get("size_bytes")
    ):
        raise ValueError("processor raw evidence hash/size binding drifted")
    ancestor_validator(root, str(evidence_commit), source_git_commit)

    primary = _mapping(v21.get("primary_policy"), "v2.1 primary_policy")
    policy = _mapping(run_contract.get("policy"), "policy")
    if (
        policy.get("repo") != primary.get("repo")
        or policy.get("revision") != primary.get("revision")
        or primary.get("snapshot_manifest") != canonical_inputs.get("snapshot_manifest")
        or _mapping(
            policy.get("runtime_metadata_requirement"),
            "runtime metadata requirement",
        ).get("snapshot_manifest_sha256")
        != canonical_inputs["snapshot_manifest"]["sha256"]
    ):
        raise ValueError("run policy differs from source-X primary policy")
    interface = _mapping(v21.get("policy_interface"), "v2.1 policy_interface")
    interface_source = _mapping(interface.get("source"), "policy interface source")
    runtime_source = _mapping(
        _mapping(interface.get("immutable_identity"), "interface immutable identity").get(
            "runtime_source"
        ),
        "runtime source",
    )
    for record in (interface_source, runtime_source):
        relative = _safe_relative_path(record.get("path"), label="policy source path")
        payload = committed_blob_reader(root, source_git_commit, relative)
        if record.get("sha256") != sha256_bytes(payload):
            raise ValueError("source-X policy interface/runtime source SHA drifted")
    preflight = _mapping(v21.get("processor_preflight"), "v2.1 processor_preflight")
    if (
        audit.get("status") != preflight.get("required_success_status")
        or audit.get("prompt_count") != preflight.get("prompt_count")
    ):
        raise ValueError("run processor/runtime identity differs from source-X contract")
    derived = _mapping(data.get("derived_artifact"), "v2.1 derived_artifact")
    if _mapping(run_contract.get("artifact"), "artifact").get("derived_repo") != derived:
        raise ValueError("run derived artifact differs from source-X contract")


def _inventory(files: Mapping[str, bytes]) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    for relative in sorted(files):
        _safe_relative_path(relative, label="evidence inventory path")
        payload = files[relative]
        records.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    return tuple(records)


def _expected_complete_file_names() -> set[str]:
    return {
        ARCHIVE_LEDGER_NAME,
        RUN_MANIFEST_FILENAME,
        RUNTIME_IDENTITY_FILENAME,
        AGGREGATE_FILENAME,
        *{
            f"{STATE_DIRECTORY}/{index:03d}.json"
            for index in range(EXPECTED_STATE_COUNT)
        },
        *{
            f"{ATTEMPT_DIRECTORY}/{index:03d}.json"
            for index in range(EXPECTED_STATE_COUNT)
        },
    }


def _validate_attempt_and_state_records(
    files: Mapping[str, bytes],
    *,
    states: Sequence[ScreeningState],
    run_contract_sha256: str,
    completed_state_count: int,
    attempted_state_count: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pilot_index in range(attempted_state_count):
        attempt_name = f"{ATTEMPT_DIRECTORY}/{pilot_index:03d}.json"
        attempt = strict_json_object_bytes(files[attempt_name], label=attempt_name)
        projection = _pilot_projection(states[pilot_index], pilot_index)
        expected_attempt_keys = {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "state",
            "attempt_ordinal",
            "retry_count",
            "top_up_count",
            "created_at_utc",
        }
        _exact_keys(attempt, expected_attempt_keys, "pilot attempt marker")
        if (
            attempt.get("schema_version") != PILOT_SCHEMA_VERSION
            or attempt.get("protocol_id") != PROTOCOL_ID
            or attempt.get("status") != ATTEMPT_STATUS
            or attempt.get("run_contract_sha256") != run_contract_sha256
            or attempt.get("state") != projection
            or attempt.get("attempt_ordinal") != 1
            or attempt.get("retry_count") != 0
            or attempt.get("top_up_count") != 0
            or not isinstance(attempt.get("created_at_utc"), str)
            or not attempt["created_at_utc"]
        ):
            raise ValueError("pilot attempt marker identity drifted")
        if pilot_index < completed_state_count:
            state_name = f"{STATE_DIRECTORY}/{pilot_index:03d}.json"
            record = strict_json_object_bytes(files[state_name], label=state_name)
            _validate_terminal_state_record(
                record,
                projection=projection,
                run_contract_sha256=run_contract_sha256,
            )
            records.append(record)
    return records


def validate_pilot_evidence_files(
    files: Mapping[str, bytes],
    *,
    expected_source_git_commit: str | None = None,
) -> PilotEvidence:
    """Recompute the formal outcome from an exact logical evidence tree."""
    copied = dict(files)
    if any(not isinstance(name, str) or not isinstance(data, bytes) for name, data in copied.items()):
        raise TypeError("pilot evidence files must map POSIX paths to bytes")
    required = {ARCHIVE_LEDGER_NAME, RUN_MANIFEST_FILENAME, AGGREGATE_FILENAME}
    if not required.issubset(copied):
        raise ValueError("pilot evidence lacks manifest, ledger, or aggregate")
    manifest = strict_json_object_bytes(
        copied[RUN_MANIFEST_FILENAME],
        label="pilot run manifest",
    )
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "run_contract",
            "created_at_utc",
        },
        "pilot run manifest",
    )
    run_contract = _mapping(manifest.get("run_contract"), "pilot run contract")
    contract_sha256 = sha256_bytes(canonical_json_bytes(run_contract))
    created_at_utc = manifest.get("created_at_utc")
    if (
        manifest.get("schema_version") != PILOT_SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != RUN_STATUS
        or manifest.get("run_contract_sha256") != contract_sha256
        or not isinstance(created_at_utc, str)
    ):
        raise ValueError("pilot run manifest identity or contract SHA drifted")
    _duration_seconds(created_at_utc, created_at_utc)
    source_commit, states = _validate_run_contract(
        run_contract,
        expected_source_git_commit=expected_source_git_commit,
    )
    ledger = strict_json_object_bytes(
        copied[ARCHIVE_LEDGER_NAME],
        label="global attempt ledger",
    )
    _exact_keys(
        ledger,
        {
            "schema_version",
            "protocol_id",
            "status",
            "attempt_identity",
            "run_contract_sha256",
            "created_at_utc",
        },
        "global attempt ledger",
    )
    if (
        ledger.get("schema_version") != PILOT_SCHEMA_VERSION
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("status") != GLOBAL_ATTEMPT_STATUS
        or ledger.get("attempt_identity") != _canonical_attempt_identity()
        or ledger.get("run_contract_sha256") != contract_sha256
        or not isinstance(ledger.get("created_at_utc"), str)
        or not ledger["created_at_utc"]
    ):
        raise ValueError("global attempt ledger identity drifted")

    aggregate = strict_json_object_bytes(
        copied[AGGREGATE_FILENAME],
        label="pilot aggregate",
    )
    outcome = aggregate.get("outcome")
    runtime_metadata: Mapping[str, Any] | None = None
    if RUNTIME_IDENTITY_FILENAME in copied:
        runtime_record = strict_json_object_bytes(
            copied[RUNTIME_IDENTITY_FILENAME],
            label="pilot runtime identity",
        )
        runtime_metadata = _validate_runtime_identity_record(
            runtime_record,
            run_contract=run_contract,
            run_contract_sha256=contract_sha256,
        )
    if outcome in {PASS_OUTCOME, NO_GO_OUTCOME}:
        expected_files = _expected_complete_file_names()
        if set(copied) != expected_files:
            raise ValueError("completed pilot evidence inventory is not exact fixed-15")
        records = _validate_attempt_and_state_records(
            copied,
            states=states,
            run_contract_sha256=contract_sha256,
            completed_state_count=EXPECTED_STATE_COUNT,
            attempted_state_count=EXPECTED_STATE_COUNT,
        )
        if runtime_metadata is None:
            raise ValueError("completed pilot evidence lacks runtime identity")
        for record in records:
            metadata = _mapping(
                record.get("generation_metadata"),
                "state generation metadata",
            )
            if any(
                metadata.get(key) != runtime_metadata[key]
                for key in RUNTIME_GENERATION_BINDING_KEYS
            ):
                raise ValueError(
                    "state generation metadata differs from runtime identity"
                )
        recomputed = aggregate_pilot_gate(
            records,
            expected_states=states,
            gate=_fixed_gate(),
            run_contract_sha256=contract_sha256,
            started_at_utc=created_at_utc,
            ended_at_utc=str(aggregate.get("ended_at_utc")),
        )
        if aggregate != recomputed:
            raise ValueError("stored pilot aggregate differs from independent reduction")
        completed_state_count = EXPECTED_STATE_COUNT
        attempted_state_count = EXPECTED_STATE_COUNT
    elif outcome == INVALID_OUTCOME:
        _validate_invalid_aggregate(
            aggregate,
            run_contract_sha256=contract_sha256,
            started_at_utc=created_at_utc,
        )
        completed_state_count = int(aggregate["completed_state_count"])
        attempted_state_count = int(aggregate["attempted_state_count"])
        failure = _mapping(aggregate.get("invalid_failure"), "invalid failure")
        stage = failure.get("stage")
        runtime_forbidden_stages = {
            "durable_claim_bootstrap_failure",
            "policy_runtime_initialization_after_durable_claim",
        }
        runtime_required = (
            attempted_state_count > 0
            or stage not in {
                *runtime_forbidden_stages,
                "resume_recovered_nonterminal_no_retry_claim",
            }
        )
        if stage in runtime_forbidden_stages and runtime_metadata is not None:
            raise ValueError("pre-runtime INVALID evidence contains runtime identity")
        if runtime_required and runtime_metadata is None:
            raise ValueError("post-runtime INVALID evidence lacks runtime identity")
        expected_files = {
            ARCHIVE_LEDGER_NAME,
            RUN_MANIFEST_FILENAME,
            AGGREGATE_FILENAME,
            *{
                f"{STATE_DIRECTORY}/{index:03d}.json"
                for index in range(completed_state_count)
            },
            *{
                f"{ATTEMPT_DIRECTORY}/{index:03d}.json"
                for index in range(attempted_state_count)
            },
        }
        if runtime_metadata is not None:
            expected_files.add(RUNTIME_IDENTITY_FILENAME)
        if set(copied) != expected_files:
            raise ValueError("INVALID pilot evidence inventory drifted")
        invalid_records = _validate_attempt_and_state_records(
            copied,
            states=states,
            run_contract_sha256=contract_sha256,
            completed_state_count=completed_state_count,
            attempted_state_count=attempted_state_count,
        )
        if runtime_metadata is not None:
            for record in invalid_records:
                metadata = _mapping(
                    record.get("generation_metadata"),
                    "INVALID state generation metadata",
                )
                if any(
                    metadata.get(key) != runtime_metadata[key]
                    for key in RUNTIME_GENERATION_BINDING_KEYS
                ):
                    raise ValueError(
                        "INVALID state metadata differs from runtime identity"
                    )
    else:
        raise ValueError("pilot aggregate outcome is not PASS, NO_GO, or INVALID")
    inventory = _inventory(copied)
    return PilotEvidence(
        files=copied,
        run_contract=dict(run_contract),
        source_git_commit=source_commit,
        run_contract_sha256=contract_sha256,
        outcome=str(outcome),
        status=str(aggregate.get("status")),
        aggregate_sha256=sha256_bytes(copied[AGGREGATE_FILENAME]),
        run_manifest_sha256=sha256_bytes(copied[RUN_MANIFEST_FILENAME]),
        global_attempt_ledger_sha256=sha256_bytes(copied[ARCHIVE_LEDGER_NAME]),
        completed_state_count=completed_state_count,
        attempted_state_count=attempted_state_count,
        inventory=inventory,
        tree_inventory_sha256=sha256_bytes(canonical_json_bytes(inventory)),
    )


def collect_raw_pilot_evidence(
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    *,
    expected_source_git_commit: str | None = None,
    require_canonical_location: bool = True,
) -> PilotEvidence:
    root = Path(raw_output_dir)
    ledger = Path(global_attempt_ledger)
    if require_canonical_location and (
        root.resolve() != CANONICAL_RAW_OUTPUT_DIR
        or ledger.resolve() != CANONICAL_GLOBAL_LEDGER_PATH
    ):
        raise ValueError("raw pilot directory or global ledger is not canonical")
    if root.is_symlink() or ledger.is_symlink() or not root.is_dir() or not ledger.is_file():
        raise ValueError("raw pilot directory or global ledger is missing or symlinked")
    root_entries = tuple(root.iterdir())
    root_names = {path.name for path in root_entries}
    required_root_names = {
        RUN_MANIFEST_FILENAME,
        AGGREGATE_FILENAME,
        STATE_DIRECTORY,
        ATTEMPT_DIRECTORY,
    }
    allowed_root_names = {*required_root_names, RUNTIME_IDENTITY_FILENAME}
    if (
        not required_root_names.issubset(root_names)
        or not root_names.issubset(allowed_root_names)
        or any(path.is_symlink() for path in root_entries)
        or not (root / STATE_DIRECTORY).is_dir()
        or not (root / ATTEMPT_DIRECTORY).is_dir()
        or not (root / RUN_MANIFEST_FILENAME).is_file()
        or not (root / AGGREGATE_FILENAME).is_file()
    ):
        raise ValueError("raw pilot root inventory drifted")
    files: dict[str, bytes] = {ARCHIVE_LEDGER_NAME: ledger.read_bytes()}
    for name in (RUN_MANIFEST_FILENAME, AGGREGATE_FILENAME):
        files[name] = (root / name).read_bytes()
    runtime_identity = root / RUNTIME_IDENTITY_FILENAME
    if runtime_identity.exists() and (
        runtime_identity.is_symlink() or not runtime_identity.is_file()
    ):
        raise ValueError("raw pilot runtime identity is not a regular file")
    if runtime_identity.is_file():
        files[RUNTIME_IDENTITY_FILENAME] = runtime_identity.read_bytes()
    for directory in (STATE_DIRECTORY, ATTEMPT_DIRECTORY):
        for path in sorted((root / directory).iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"raw pilot {directory} inventory contains a non-file")
            relative = f"{directory}/{path.name}"
            _safe_relative_path(relative, label=f"raw pilot {directory} path")
            files[relative] = path.read_bytes()
    return validate_pilot_evidence_files(
        files,
        expected_source_git_commit=expected_source_git_commit,
    )


def _write_deterministic_tar(path: Path, files: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        with tarfile.open(
            fileobj=destination,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as archive:
            for relative in sorted(files):
                member_name = f"{ARCHIVE_MEMBER_PREFIX}/{relative}"
                payload = files[relative]
                info = tarfile.TarInfo(member_name)
                info.size = len(payload)
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))
        destination.flush()
        os.fsync(destination.fileno())


def read_pilot_evidence_archive(
    archive_path: str | Path,
    *,
    expected_source_git_commit: str | None = None,
) -> PilotEvidence:
    path = Path(archive_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("pilot raw archive is missing or symlinked")
    files: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("pilot archive member ordering or uniqueness drifted")
        for member in members:
            prefix = f"{ARCHIVE_MEMBER_PREFIX}/"
            if (
                not member.isfile()
                or not member.name.startswith(prefix)
                or member.mode != 0o644
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != 0
                or member.pax_headers
            ):
                raise ValueError("pilot archive contains a non-canonical member")
            relative = member.name[len(prefix) :]
            _safe_relative_path(relative, label="pilot archive member path")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("pilot archive regular file cannot be read")
            payload = source.read()
            if len(payload) != member.size:
                raise ValueError("pilot archive member size drifted")
            files[relative] = payload
    return validate_pilot_evidence_files(
        files,
        expected_source_git_commit=expected_source_git_commit,
    )


def read_extracted_pilot_evidence(
    extracted_path: str | Path,
    *,
    expected_source_git_commit: str | None = None,
) -> PilotEvidence:
    supplied = Path(extracted_path)
    root = supplied / ARCHIVE_MEMBER_PREFIX if (supplied / ARCHIVE_MEMBER_PREFIX).is_dir() else supplied
    if root.is_symlink() or not root.is_dir() or root.name != ARCHIVE_MEMBER_PREFIX:
        raise ValueError("extracted pilot evidence root is not canonical")
    files: dict[str, bytes] = {}
    observed_directories: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("extracted pilot evidence contains a symlink")
        if path.is_dir():
            observed_directories.add(path.relative_to(root).as_posix())
            continue
        if not path.is_file():
            raise ValueError("extracted pilot evidence contains a non-file")
        relative = path.relative_to(root).as_posix()
        _safe_relative_path(relative, label="extracted pilot evidence path")
        files[relative] = path.read_bytes()
    if observed_directories != {STATE_DIRECTORY, ATTEMPT_DIRECTORY}:
        raise ValueError("extracted pilot evidence directory inventory drifted")
    return validate_pilot_evidence_files(
        files,
        expected_source_git_commit=expected_source_git_commit,
    )


def package_raw_pilot_evidence(
    *,
    repository_root: str | Path,
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    output_archive: str | Path,
    source_git_commit: str,
    require_canonical_location: bool = True,
    git_identity_validator: Callable[[str | Path], Mapping[str, Any]] = (
        validate_clean_pushed_main
    ),
    source_binding_validator: Callable[..., None] = validate_source_x_run_contract,
) -> dict[str, Any]:
    """Validate a completed/invalid run on clean source X and write one raw tar."""
    if GIT_SHA_PATTERN.fullmatch(source_git_commit) is None:
        raise ValueError("source Git commit must be a full lowercase commit SHA")
    archive = Path(output_archive)
    if require_canonical_location and archive.resolve() != CANONICAL_RAW_ARCHIVE_PATH:
        raise ValueError("pilot archive output path is not canonical")
    git = dict(git_identity_validator(Path(repository_root).resolve()))
    if git.get("commit") != source_git_commit:
        raise ValueError("packaging Git identity differs from pilot source commit")
    evidence = collect_raw_pilot_evidence(
        raw_output_dir,
        global_attempt_ledger,
        expected_source_git_commit=source_git_commit,
        require_canonical_location=require_canonical_location,
    )
    source_binding_validator(
        evidence.run_contract,
        repository_root=repository_root,
        source_git_commit=source_git_commit,
    )
    _write_deterministic_tar(archive, evidence.files)
    reread = read_pilot_evidence_archive(
        archive,
        expected_source_git_commit=source_git_commit,
    )
    if reread.inventory != evidence.inventory or reread.outcome != evidence.outcome:
        raise ValueError("written pilot archive differs from validated raw tree")
    return {
        "status": "PACKAGED_RESTORATION_V2_1_INTERFACE_PILOT_RAW_EVIDENCE",
        "source_git_commit": source_git_commit,
        "outcome": evidence.outcome,
        "archive_path": str(archive.resolve()),
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "file_count": len(evidence.inventory),
    }


def _validate_hf_identity(
    *,
    repo: str,
    immutable_revision: str,
    path: str,
) -> dict[str, str]:
    if repo != CANONICAL_HF_REPO:
        raise ValueError("pilot raw evidence HF repo differs from the canonical repo")
    if GIT_SHA_PATTERN.fullmatch(immutable_revision) is None:
        raise ValueError("pilot HF revision must be an immutable 40-hex commit")
    safe_path = _safe_relative_path(path, label="pilot HF artifact path")
    if safe_path != CANONICAL_HF_PATH or not safe_path.endswith(".tar"):
        raise ValueError("pilot HF artifact path differs from the canonical tar path")
    return {
        "repo": CANONICAL_HF_REPO,
        "repo_type": "dataset",
        "visibility": "private",
        "tag": CANONICAL_HF_TAG,
        "immutable_revision": immutable_revision,
        "path": safe_path,
    }


def build_pilot_artifact_manifest(
    *,
    repository_root: str | Path,
    raw_archive: str | Path,
    source_git_commit: str,
    hf_repo: str,
    hf_immutable_revision: str,
    hf_path: str,
    source_binding_validator: Callable[..., None] = validate_source_x_run_contract,
) -> dict[str, Any]:
    archive = Path(raw_archive)
    evidence = read_pilot_evidence_archive(
        archive,
        expected_source_git_commit=source_git_commit,
    )
    source_binding_validator(
        evidence.run_contract,
        repository_root=repository_root,
        source_git_commit=source_git_commit,
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_id": GUI_OWL_V2_1_PROTOCOL_ID,
        "artifact_type": ARTIFACT_TYPE,
        "source_execution": {
            "attempt_id": CANONICAL_ATTEMPT_ID,
            "canonical_raw_output_dir": str(CANONICAL_RAW_OUTPUT_DIR),
            "canonical_global_attempt_ledger": str(CANONICAL_GLOBAL_LEDGER_PATH),
            "source_git_commit": evidence.source_git_commit,
            "run_contract_sha256": evidence.run_contract_sha256,
        },
        "hf_artifact": _validate_hf_identity(
            repo=hf_repo,
            immutable_revision=hf_immutable_revision,
            path=hf_path,
        ),
        "raw_archive": {
            "format": ARCHIVE_FORMAT,
            "member_prefix": ARCHIVE_MEMBER_PREFIX,
            "sha256": sha256_file(archive),
            "size_bytes": archive.stat().st_size,
            "file_count": len(evidence.inventory),
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
            "inventory": list(evidence.inventory),
        },
        "result": {
            "outcome": evidence.outcome,
            "status": evidence.status,
            "fixed_state_denominator": EXPECTED_STATE_COUNT,
            "completed_state_count": evidence.completed_state_count,
            "attempted_state_count": evidence.attempted_state_count,
            "aggregate_sha256": evidence.aggregate_sha256,
            "run_manifest_sha256": evidence.run_manifest_sha256,
            "global_attempt_ledger_sha256": (
                evidence.global_attempt_ledger_sha256
            ),
        },
    }


def validate_pilot_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    evidence: PilotEvidence,
    archive_path: str | Path | None,
) -> dict[str, Any]:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "artifact_type",
            "source_execution",
            "hf_artifact",
            "raw_archive",
            "result",
        },
        "pilot artifact manifest",
    )
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("protocol_id") != GUI_OWL_V2_1_PROTOCOL_ID
        or manifest.get("artifact_type") != ARTIFACT_TYPE
    ):
        raise ValueError("pilot artifact manifest identity drifted")
    source = _mapping(manifest.get("source_execution"), "source_execution")
    expected_source = {
        "attempt_id": CANONICAL_ATTEMPT_ID,
        "canonical_raw_output_dir": str(CANONICAL_RAW_OUTPUT_DIR),
        "canonical_global_attempt_ledger": str(CANONICAL_GLOBAL_LEDGER_PATH),
        "source_git_commit": evidence.source_git_commit,
        "run_contract_sha256": evidence.run_contract_sha256,
    }
    if dict(source) != expected_source:
        raise ValueError("pilot source execution binding drifted")
    hf = _mapping(manifest.get("hf_artifact"), "hf_artifact")
    expected_hf = _validate_hf_identity(
        repo=str(hf.get("repo")),
        immutable_revision=str(hf.get("immutable_revision")),
        path=str(hf.get("path")),
    )
    if dict(hf) != expected_hf:
        raise ValueError("pilot HF artifact metadata drifted")
    raw = _mapping(manifest.get("raw_archive"), "raw_archive")
    expected_raw = {
        "format": ARCHIVE_FORMAT,
        "member_prefix": ARCHIVE_MEMBER_PREFIX,
        "sha256": raw.get("sha256"),
        "size_bytes": raw.get("size_bytes"),
        "file_count": len(evidence.inventory),
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "inventory": list(evidence.inventory),
    }
    if (
        not isinstance(raw.get("sha256"), str)
        or SHA256_PATTERN.fullmatch(raw["sha256"]) is None
        or type(raw.get("size_bytes")) is not int
        or raw["size_bytes"] <= 0
        or dict(raw) != expected_raw
    ):
        raise ValueError("pilot raw archive inventory binding drifted")
    archive_hash_verified = False
    if archive_path is not None:
        archive = Path(archive_path)
        if (
            sha256_file(archive) != raw["sha256"]
            or archive.stat().st_size != raw["size_bytes"]
        ):
            raise ValueError("pilot raw archive hash or size drifted")
        archive_hash_verified = True
    result = _mapping(manifest.get("result"), "result")
    expected_result = {
        "outcome": evidence.outcome,
        "status": evidence.status,
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "completed_state_count": evidence.completed_state_count,
        "attempted_state_count": evidence.attempted_state_count,
        "aggregate_sha256": evidence.aggregate_sha256,
        "run_manifest_sha256": evidence.run_manifest_sha256,
        "global_attempt_ledger_sha256": evidence.global_attempt_ledger_sha256,
    }
    if dict(result) != expected_result:
        raise ValueError("pilot compact result differs from raw evidence")
    return {
        "status": "VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT",
        "source_git_commit": evidence.source_git_commit,
        "outcome": evidence.outcome,
        "file_count": len(evidence.inventory),
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "archive_hash_verified": archive_hash_verified,
    }


def write_pilot_artifact_manifest(
    *,
    repository_root: str | Path,
    raw_archive: str | Path,
    source_git_commit: str,
    hf_repo: str,
    hf_immutable_revision: str,
    hf_path: str,
    output: str | Path,
    git_identity_validator: Callable[[str | Path], Mapping[str, Any]] = (
        validate_clean_pushed_main
    ),
    source_binding_validator: Callable[..., None] = validate_source_x_run_contract,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(output).resolve()
    canonical = (root / CANONICAL_GIT_ARTIFACT_PATH).resolve()
    if destination != canonical:
        raise ValueError("pilot Git artifact output path is not canonical")
    if destination.exists():
        raise FileExistsError("pilot Git artifact manifest already exists")
    git = dict(git_identity_validator(root))
    if git.get("commit") != source_git_commit:
        raise ValueError("post-upload manifest must be created from clean source commit X")
    manifest = build_pilot_artifact_manifest(
        repository_root=root,
        raw_archive=raw_archive,
        source_git_commit=source_git_commit,
        hf_repo=hf_repo,
        hf_immutable_revision=hf_immutable_revision,
        hf_path=hf_path,
        source_binding_validator=source_binding_validator,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(pretty_json_bytes(manifest))
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def require_source_commit_ancestor(
    *,
    repository_root: str | Path,
    source_git_commit: str,
    current_git_commit: str,
) -> None:
    if (
        GIT_SHA_PATTERN.fullmatch(source_git_commit) is None
        or GIT_SHA_PATTERN.fullmatch(current_git_commit) is None
    ):
        raise ValueError("source/current Git commit is invalid")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(Path(repository_root).resolve()),
                "merge-base",
                "--is-ancestor",
                source_git_commit,
                current_git_commit,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("pilot source commit is not an ancestor of current main") from error


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git command failed: {message or arguments}") from error


def validate_committed_pilot_artifact(
    *,
    repository_root: str | Path,
    current_git_commit: str,
    evidence_path: str | Path,
    artifact_path: str | Path = CANONICAL_GIT_ARTIFACT_PATH,
    source_binding_validator: Callable[..., None] = validate_source_x_run_contract,
) -> dict[str, Any]:
    """Reuse evidence X from a clean descendant main commit Y."""
    root = Path(repository_root).resolve()
    canonical = (root / CANONICAL_GIT_ARTIFACT_PATH).resolve()
    supplied_artifact = Path(artifact_path)
    supplied_artifact = (
        supplied_artifact.resolve()
        if supplied_artifact.is_absolute()
        else (root / supplied_artifact).resolve()
    )
    if supplied_artifact != canonical or not canonical.is_file():
        raise ValueError("canonical pilot Git artifact manifest is missing")
    live = canonical.read_bytes()
    committed = _git_bytes(
        root,
        "show",
        f"{current_git_commit}:{CANONICAL_GIT_ARTIFACT_PATH}",
    )
    if live != committed:
        raise ValueError("pilot artifact manifest differs from current Git blob")
    manifest = strict_json_object_bytes(committed, label="pilot artifact manifest")
    source = _mapping(manifest.get("source_execution"), "source_execution")
    source_commit = source.get("source_git_commit")
    if not isinstance(source_commit, str):
        raise ValueError("pilot source Git commit is missing")
    require_source_commit_ancestor(
        repository_root=root,
        source_git_commit=source_commit,
        current_git_commit=current_git_commit,
    )
    evidence_source = Path(evidence_path)
    if evidence_source.is_dir():
        evidence = read_extracted_pilot_evidence(
            evidence_source,
            expected_source_git_commit=source_commit,
        )
        archive_path: Path | None = None
        source_kind = "extracted_tree"
    else:
        evidence = read_pilot_evidence_archive(
            evidence_source,
            expected_source_git_commit=source_commit,
        )
        archive_path = evidence_source
        source_kind = "raw_archive"
    source_binding_validator(
        evidence.run_contract,
        repository_root=root,
        source_git_commit=source_commit,
    )
    result = validate_pilot_artifact_manifest(
        manifest,
        evidence=evidence,
        archive_path=archive_path,
    )
    return {
        **result,
        "current_git_commit": current_git_commit,
        "evidence_source_kind": source_kind,
    }
