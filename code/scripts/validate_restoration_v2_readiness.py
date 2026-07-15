"""Fail-closed pre-policy authorization for restoration-v2 screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_restoration_v2"
EXECUTION_STATUS = "FROZEN_PREOUTPUT_EXECUTION"
READINESS_STATUS = "SCREENING_ALLOWED"
CONFIRM_STATE = "CONFIRM_LOCKED"
SCIENTIFIC_SHA256 = (
    "9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc"
)
CANONICAL_REMOTE = "https://github.com/luojiaxuan/CausalCache.git"
ALLOWED_ROLES = ("v2_label_train", "v2_development")
READINESS_MANIFEST_PATH = "data/manifests/restoration_v2_readiness.json"
PROCESSOR_AUDIT_SOURCE_PATH = "code/scripts/audit_gui_owl_v2_processor.py"
PROCESSOR_AUDIT_SUMMARY_PATH = (
    "data/results/restoration_v2_processor_audit/summary.json"
)
PROCESSOR_AUDIT_SUMMARY_SHA256 = (
    "7d5ac1bd13ba5def46dfb2ca419d59bb0da1ff970f186e9fd092d4006d8b43b8"
)
PROCESSOR_AUDIT_SOURCE_PATHS = (
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_runtime.py",
    "code/causalcache/policy/gui_owl_v2_vision.py",
    PROCESSOR_AUDIT_SOURCE_PATH,
)
TRANSFORMERS_SOURCE_SHA256 = {
    "modeling_qwen3_vl.py": (
        "c37abc40a3744dc116cf1e06eceb7ff79bd384cfbfe0d41773f5efc074f31523"
    ),
    "processing_qwen3_vl.py": (
        "b51f77d783b4a88d503bf9198828d46f71e3ead3833da2631748188b159253ba"
    ),
    "image_processing_qwen2_vl.py": (
        "de5859892f04c64f3e79c9d318842ac0040000788e43870997bd1ec6565c02bb"
    ),
}
DETERMINISTIC_PROCESSOR_IMAGES = {
    "generator": "stdlib_rgb_scanlines_png_zlib_level_9",
    "portrait": {
        "width": 108,
        "height": 240,
        "sha256": "e9e529dfc0d8c881053a3d81dde0802dae2c52cc1a4821f95c4ded117ed34b37",
        "size_bytes": 55_579,
    },
    "landscape": {
        "width": 240,
        "height": 108,
        "sha256": "3af4878fdbba4f543cddd68f06ad8b56eb51d3735b89c0f30b96c5f31ed5c75a",
        "size_bytes": 77_956,
    },
}


def _expected_cpu_tensor(shape: list[int], dtype: str) -> dict[str, Any]:
    return {
        "shape": shape,
        "dtype": dtype,
        "device": "cpu",
        "requires_grad": False,
    }


def _expected_processor_case(
    *,
    name: str,
    image_counts: list[int],
    samples: list[dict[str, Any]],
    sequence_length: int,
    raw_patch_count: int,
) -> dict[str, Any]:
    batch_size = len(image_counts)
    return {
        "status": "passed",
        "case": name,
        "batch_size": batch_size,
        "image_counts": image_counts,
        "padding": False,
        "no_padding": True,
        "equal_image_count": True,
        "equal_sequence_length": True,
        "attention_mask_all_one": True,
        "assistant_prefix_tail_exact": True,
        "tensor_inventory": {
            "attention_mask": _expected_cpu_tensor(
                [batch_size, sequence_length], "torch.int64"
            ),
            "image_grid_thw": _expected_cpu_tensor(
                [sum(image_counts), 3], "torch.int64"
            ),
            "input_ids": _expected_cpu_tensor(
                [batch_size, sequence_length], "torch.int64"
            ),
            "mm_token_type_ids": _expected_cpu_tensor(
                [batch_size, sequence_length], "torch.int64"
            ),
            "pixel_values": _expected_cpu_tensor(
                [raw_patch_count, 1536], "torch.float32"
            ),
        },
        "samples": samples,
    }


EXPECTED_PROCESSOR_TOKENIZER_BOUNDARY = {
    "status": "passed",
    "action": "wait",
    "assistant_prefix_text": "<|im_start|>assistant\n",
    "carrier_text": "Action: Execute the selected mobile action.\n",
    "distance_text": (
        '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n'
        "</tool_call>"
    ),
    "assistant_prefix_tokens": 3,
    "carrier_tokens": 8,
    "distance_tokens": 15,
    "joint_boundary_exact": True,
}
EXPECTED_PROCESSOR_CASES = {
    "single_conversation_one_image": _expected_processor_case(
        name="single_conversation_one_image",
        image_counts=[1],
        samples=[
            {
                "image_count": 1,
                "image_grid_thw": [[1, 152, 68]],
                "effective_visual_tokens_per_image": [2584],
                "effective_visual_tokens": 2584,
                "policy_visible_text_tokens": 359,
                "sequence_length": 2943,
            }
        ],
        sequence_length=2943,
        raw_patch_count=10_336,
    ),
    "single_conversation_five_images": _expected_processor_case(
        name="single_conversation_five_images",
        image_counts=[5],
        samples=[
            {
                "image_count": 5,
                "image_grid_thw": [
                    [1, 152, 68],
                    [1, 68, 152],
                    [1, 152, 68],
                    [1, 68, 152],
                    [1, 152, 68],
                ],
                "effective_visual_tokens_per_image": [2584, 2584, 2584, 2584, 2584],
                "effective_visual_tokens": 12_920,
                "policy_visible_text_tokens": 366,
                "sequence_length": 13_286,
            }
        ],
        sequence_length=13_286,
        raw_patch_count=51_680,
    ),
    "nested_batch_two_equal_shape": _expected_processor_case(
        name="nested_batch_two_equal_shape",
        image_counts=[1, 1],
        samples=[
            {
                "image_count": 1,
                "image_grid_thw": [[1, 152, 68]],
                "effective_visual_tokens_per_image": [2584],
                "effective_visual_tokens": 2584,
                "policy_visible_text_tokens": 362,
                "sequence_length": 2946,
            },
            {
                "image_count": 1,
                "image_grid_thw": [[1, 68, 152]],
                "effective_visual_tokens_per_image": [2584],
                "effective_visual_tokens": 2584,
                "policy_visible_text_tokens": 362,
                "sequence_length": 2946,
            },
        ],
        sequence_length=2946,
        raw_patch_count=20_672,
    ),
}
EXPECTED_PROCESSOR_GEOMETRY = {
    "evidence": {
        "path": PROCESSOR_AUDIT_SUMMARY_PATH,
        "sha256": PROCESSOR_AUDIT_SUMMARY_SHA256,
    },
    "processor_class": "Qwen3VLProcessor",
    "tokenizer_class": "Qwen2Tokenizer",
    "image_processor_class": "Qwen2VLImageProcessor",
    "pixel_target_pixels_per_image": 2_621_440,
    "pixel_target_runtime_representation": (
        "image_processor.size.shortest_edge_longest_edge"
    ),
    "actual_effective_visual_tokens_per_image": 2584,
    "portrait_image_grid_thw": [1, 152, 68],
    "landscape_image_grid_thw": [1, 68, 152],
    "single_image_sequence_length": 2943,
    "five_image_sequence_length": 13_286,
    "nested_batch_two_sequence_length": 2946,
    "teacher_boundary_tokens": {
        "assistant_prefix": 3,
        "carrier": 8,
        "distance": 15,
    },
}
EXPECTED_SOURCE_PATHS = {
    "derived_artifact_validator": (
        "code/causalcache/data/guiodyssey_restoration_v2.py"
    ),
    "selection_validator": (
        "code/causalcache/data/restoration_v2_selection.py"
    ),
    "low_fidelity_serializer": "code/causalcache/low_fidelity_v2.py",
    "policy_interface": "code/causalcache/policy/gui_owl_v2.py",
    "policy_runtime": "code/causalcache/policy/gui_owl_v2_runtime.py",
    "vision_runtime_verifier": "code/causalcache/policy/gui_owl_v2_vision.py",
    "gpu_kl": "code/causalcache/restoration_v2_gpu_kl.py",
    "microbatch_planner": "code/causalcache/restoration_v2_batching.py",
    "gpu_compute_audit": "code/scripts/audit_restoration_v2_gpu_compute.py",
    "gpu_compute_audit_validator": (
        "code/scripts/validate_restoration_v2_gpu_compute_audit.py"
    ),
    "readiness_validator": "code/scripts/validate_restoration_v2_readiness.py",
    "screening_artifact_loader": (
        "code/causalcache/data/restoration_v2_screening.py"
    ),
    "substrate_screening_runner": (
        "code/scripts/run_restoration_v2_substrate_screening.py"
    ),
    "real_processor_audit": PROCESSOR_AUDIT_SOURCE_PATH,
}
EXPECTED_DEPENDENCY_NAMES = {
    1: "derived_artifact_immutable_hf_revision_and_file_hashes",
    2: "exact_confirm_trajectory_and_state_ids",
    3: "exposure_ledger",
    4: "restricted_prompt_parser_bridge_executor_round_trip_fixture",
    5: "pinned_ocr_or_accessibility_backend_identity",
    6: "baseline_specification_and_source_hashes",
    7: "v2_interface_source_hashes",
    8: "execution_config_referencing_scientific_config_sha256",
}
EXPECTED_DEPENDENCY_EVIDENCE_PATHS = {
    1: ("data/manifests/restoration_v2_derived_artifact.json",),
    2: ("data/manifests/restoration_v2_selection.json",),
    3: ("data/manifests/restoration_v2_exposure.json",),
    4: (
        "data/manifests/restoration_v2_interfaces.json",
        "data/results/restoration_v2_executor_dispatch/summary-rv2-20260715T101814Z-53016a40.json",
    ),
    5: ("data/manifests/restoration_v2_ocr_backend.json",),
    6: ("data/manifests/restoration_v2_baselines.json",),
    7: ("data/manifests/restoration_v2_interfaces.json",),
    8: (
        "data/results/restoration_v2_gpu_compute_audit/summary.json",
        "data/results/restoration_v2_gpu_compute_audit/validation.json",
        PROCESSOR_AUDIT_SUMMARY_PATH,
    ),
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def _resolve(repository_root: Path, value: str, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a non-empty repository-relative path")
    root = repository_root.resolve()
    result = (root / value).resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"{name} escapes the repository root")
    return result


def _validate_file_reference(
    value: Any,
    *,
    repository_root: Path,
    expected_path: str,
    name: str,
) -> dict[str, str]:
    record = _object(value, name=name)
    _exact_keys(record, {"path", "sha256"}, name=name)
    if record["path"] != expected_path:
        raise ValueError(f"{name} path drifted")
    sha256 = record["sha256"]
    if not isinstance(sha256, str) or SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError(f"{name} SHA256 is invalid")
    path = _resolve(repository_root, expected_path, name=f"{name}.path")
    if not path.is_file() or _sha256_file(path) != sha256:
        raise ValueError(f"{name} file hash drifted")
    return {"path": expected_path, "sha256": sha256}


def _dependency_map(value: Any) -> dict[int, Mapping[str, Any]]:
    records = _list(value, name="dependencies")
    if len(records) != 8:
        raise ValueError("execution config must contain exactly eight dependencies")
    result: dict[int, Mapping[str, Any]] = {}
    for raw_record in records:
        record = _object(raw_record, name="dependency")
        _exact_keys(record, {"id", "name", "status", "evidence"}, name="dependency")
        identifier = record["id"]
        if type(identifier) is not int or identifier not in EXPECTED_DEPENDENCY_NAMES:
            raise ValueError("dependency id drifted")
        if identifier in result:
            raise ValueError("duplicate dependency id")
        if (
            record["name"] != EXPECTED_DEPENDENCY_NAMES[identifier]
            or record["status"] != "passed"
        ):
            raise ValueError(f"dependency {identifier} did not pass exactly")
        evidence = _list(record["evidence"], name=f"dependency {identifier} evidence")
        if not evidence:
            raise ValueError(f"dependency {identifier} has no evidence")
        result[identifier] = record
    if tuple(sorted(result)) != tuple(range(1, 9)):
        raise ValueError("dependency ids must be exactly 1 through 8")
    return result


def _evidence_by_path(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for value in _list(record["evidence"], name="dependency evidence"):
        evidence = _object(value, name="dependency evidence record")
        _exact_keys(evidence, {"path", "sha256"}, name="dependency evidence record")
        path = evidence["path"]
        if not isinstance(path, str) or path in result:
            raise ValueError("dependency evidence path is invalid or duplicated")
        result[path] = evidence
    return result


def _require_evidence(
    dependency: Mapping[str, Any],
    *,
    path: str,
    repository_root: Path,
) -> dict[str, str]:
    records = _evidence_by_path(dependency)
    if path not in records:
        raise ValueError(f"dependency evidence is missing {path}")
    return _validate_file_reference(
        records[path],
        repository_root=repository_root,
        expected_path=path,
        name=f"dependency evidence {path}",
    )


def _validate_dependency_evidence_inventory(
    dependencies: Mapping[int, Mapping[str, Any]],
) -> None:
    for identifier, expected_paths in EXPECTED_DEPENDENCY_EVIDENCE_PATHS.items():
        actual_paths = set(_evidence_by_path(dependencies[identifier]))
        if actual_paths != set(expected_paths):
            raise ValueError(f"dependency {identifier} evidence inventory drifted")


def _validate_preclosure_gpu_compute_audit(
    gpu_summary: Mapping[str, Any],
    gpu_validation: Mapping[str, Any],
    *,
    summary_sha256: str,
) -> None:
    """Validate the earlier GPU evidence without reinterpreting it as closure."""
    compute = _object(
        gpu_validation.get("compute_validation"), name="GPU compute validation"
    )
    planner = _object(
        gpu_validation.get("planner_validation"), name="GPU planner validation"
    )
    if (
        gpu_summary.get("outcome") != "PASSED_RESTORATION_V2_GPU_COMPUTE_AUDIT"
        or gpu_summary.get("policy_loaded") is not False
        or gpu_summary.get("policy_output_generated") is not False
        or gpu_summary.get("restoration_output_generated") is not False
        or gpu_validation.get("outcome")
        != "PASSED_INDEPENDENT_RESTORATION_V2_GPU_COMPUTE_AUDIT_VALIDATION"
        or gpu_validation.get("summary_sha256") != summary_sha256
        or gpu_validation.get("policy_loaded") is not False
        or gpu_validation.get("policy_output_generated") is not False
        or gpu_validation.get("restoration_output_generated") is not False
        or gpu_validation.get("dependency_8_closed") is not False
        or gpu_validation.get("screening_unlocked") is not False
        or compute.get("validation_scalar_host_reads") != 0
        or compute.get("full_tensor_host_transfers") != 0
        or compute.get("invalid_numeric_output") != "nan_final_distance"
        or planner
        != {
            "automatic_oom_fallback": False,
            "microbatch_size": 2,
            "planning_scope": "single_decision_state",
        }
    ):
        raise ValueError("dependency 8 GPU audit contract drifted")


def _validate_dependencies(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
) -> list[dict[str, object]]:
    dependencies = _dependency_map(config["dependencies"])
    paths = EXPECTED_DEPENDENCY_EVIDENCE_PATHS
    _validate_dependency_evidence_inventory(dependencies)
    loaded: dict[str, Mapping[str, Any]] = {}
    result: list[dict[str, object]] = []
    for identifier, expected_paths in paths.items():
        evidence = []
        for path in expected_paths:
            evidence.append(
                _require_evidence(
                    dependencies[identifier],
                    path=path,
                    repository_root=repository_root,
                )
            )
            loaded.setdefault(path, _load_json_object(repository_root / path))
        result.append(
            {
                "id": identifier,
                "name": EXPECTED_DEPENDENCY_NAMES[identifier],
                "status": "passed",
                "evidence": evidence,
            }
        )

    derived = loaded[paths[1][0]]
    artifact = _object(derived.get("hf_dataset_artifact"), name="derived artifact")
    if (
        derived.get("protocol_id") != PROTOCOL_ID
        or derived.get("dependency_1_closed") is not True
        or derived.get("status")
        != "hf_immutable_verified_two_materializations_and_three_exact_replays_passed"
        or artifact.get("repo")
        != "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile"
        or artifact.get("immutable_revision")
        != "89f136abaff797e14fe758a198996e51032a10a6"
        or artifact.get("artifact_tree_sha256")
        != "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e"
        or any(
            derived.get(key) is not False
            for key in (
                "policy_loaded_before_manifest",
                "policy_output_generated_before_manifest",
                "restoration_output_generated_before_manifest",
            )
        )
    ):
        raise ValueError("dependency 1 derived artifact contract drifted")

    selection = loaded[paths[2][0]]
    roles = _object(selection.get("roles"), name="selection roles")
    role_counts = {
        role: len(_list(_object(value, name=role).get("states"), name=f"{role}.states"))
        for role, value in roles.items()
        if role in {"v2_label_train", "v2_development", "v2_confirm_primary"}
    }
    if (
        selection.get("protocol_id") != PROTOCOL_ID
        or selection.get("status") != "PASSED_PREOUTPUT_SELECTION"
        or selection.get("policy_output_generated") is not False
        or selection.get("restoration_output_generated") is not False
        or role_counts
        != {"v2_label_train": 30, "v2_development": 15, "v2_confirm_primary": 20}
    ):
        raise ValueError("dependency 2 selection contract drifted")

    exposure = loaded[paths[3][0]]
    reduced_roles = _object(exposure.get("reduced_roles"), name="exposure roles")
    if (
        exposure.get("protocol_id") != PROTOCOL_ID
        or exposure.get("status") != "FROZEN_PREOUTPUT_EXPOSURE"
        or exposure.get("selection_manifest_sha256")
        != _evidence_by_path(dependencies[2])[paths[2][0]]["sha256"]
    ):
        raise ValueError("dependency 3 exposure contract drifted")
    for role in ALLOWED_ROLES + ("v2_confirm_primary",):
        record = _object(reduced_roles.get(role), name=f"exposure role {role}")
        if record.get("known_policy_outputs") != [] or record.get(
            "known_restoration_outputs"
        ) != []:
            raise ValueError(f"dependency 3 role {role} has forbidden output exposure")

    interface = loaded[paths[4][0]]
    cpu = _object(interface.get("local_cpu_validation"), name="interface CPU validation")
    if (
        interface.get("protocol_id") != PROTOCOL_ID
        or interface.get("scientific_contract_sha256") != SCIENTIFIC_SHA256
        or interface.get("materialized_before_any_v2_policy_output") is not True
        or interface.get("policy_output_generated_by_this_validation") is not False
        or cpu
        != {
            "valid_action_cases": 14,
            "invalid_action_cases": 23,
            "coordinate_scalar_checks": 6000,
            "prompt_coalitions": 28,
            "confirm_prompt_coalitions": 16,
            "status": "passed",
        }
    ):
        raise ValueError("dependency 4/7 interface contract drifted")
    for source in _list(interface.get("files"), name="interface files"):
        source_record = _object(source, name="interface source")
        source_path = _resolve(repository_root, source_record["path"], name="interface source")
        if _sha256_file(source_path) != source_record.get("sha256"):
            raise ValueError("dependency 7 interface source hash drifted")

    executor = loaded[paths[4][1]]
    offline = _object(executor.get("offline_reduction"), name="executor reduction")
    interface_validation = _object(
        executor.get("interface_validation"), name="executor interface validation"
    )
    constructor = _object(
        interface_validation.get("androidworld_json_action_constructor_validation"),
        name="executor constructor validation",
    )
    if (
        executor.get("protocol_id") != PROTOCOL_ID
        or executor.get("status") != "passed"
        or executor.get("policy_output_generated") is not False
        or executor.get("restoration_label_generated") is not False
        or offline
        != {
            "negative_actuation_control_status": 500,
            "status": "PASSED_EXECUTOR_DISPATCH",
            "validated_case_count": 14,
        }
        or interface_validation.get("scientific_contract_sha256") != SCIENTIFIC_SHA256
        or _object(interface_validation.get("interface_manifest"), name="interface link").get(
            "sha256"
        )
        != _evidence_by_path(dependencies[4])[paths[4][0]]["sha256"]
        or constructor.get("source_revision")
        != "11cea575561fb7800b5fb6b6cafa56f7a91de11f"
        or constructor.get("module_sha256")
        != "14ca00cabf3d5b83e4d55cb683a09a4beccbbc658e21039ca5cf8cef3f543e3f"
    ):
        raise ValueError("dependency 4 executor contract drifted")

    ocr = loaded[paths[5][0]]
    real_screen = _object(ocr.get("real_screen_golden"), name="OCR real-screen golden")
    model = _object(ocr.get("hf_model_artifact"), name="OCR model")
    if (
        ocr.get("protocol_id") != PROTOCOL_ID
        or ocr.get("dependency_5_closed") is not True
        or ocr.get("status")
        != "hf_model_immutable_verified_synthetic_and_real_screen_golden_passed"
        or ocr.get("policy_output_generated_before_manifest") is not False
        or ocr.get("restoration_output_generated_before_manifest") is not False
        or model.get("immutable_revision")
        != "0dbc766a73ee88d10d52285d434dbfec58617835"
        or real_screen.get("immutable_revision")
        != "9ebbbbbc4666e8a065f4ecb5240491c70f05e21b"
        or real_screen.get("artifact_tree_sha256")
        != "605d6396b0cde84697ff3f2407a3630d7ccdb822f55c2bdae56be82e942a7e25"
        or real_screen.get("confirm_images_used") is not False
    ):
        raise ValueError("dependency 5 OCR contract drifted")

    baselines = loaded[paths[6][0]]
    negative = _object(baselines.get("negative_declarations"), name="baseline negatives")
    if (
        baselines.get("protocol_id") != PROTOCOL_ID
        or baselines.get("dependency_6_closed") is not True
        or baselines.get("status") != "PASSED_PREOUTPUT_BASELINE_IMPLEMENTATION"
        or _object(baselines.get("scientific_contract"), name="baseline contract").get(
            "sha256"
        )
        != SCIENTIFIC_SHA256
        or any(value is not False for value in negative.values())
    ):
        raise ValueError("dependency 6 baseline contract drifted")

    gpu_summary = loaded[paths[8][0]]
    gpu_validation = loaded[paths[8][1]]
    _validate_preclosure_gpu_compute_audit(
        gpu_summary,
        gpu_validation,
        summary_sha256=_evidence_by_path(dependencies[8])[paths[8][0]]["sha256"],
    )
    configured_sources = {
        record["path"]: record
        for raw_record in _list(config["source_files"], name="source_files")
        for record in (_object(raw_record, name="source file"),)
    }
    if (
        _evidence_by_path(dependencies[8])[PROCESSOR_AUDIT_SUMMARY_PATH]["sha256"]
        != PROCESSOR_AUDIT_SUMMARY_SHA256
    ):
        raise ValueError("dependency 8 processor summary SHA256 drifted")
    _validate_processor_audit_summary(
        loaded[PROCESSOR_AUDIT_SUMMARY_PATH],
        configured_sources=configured_sources,
        execution_runtime=_object(
            config["execution_runtime"], name="execution_runtime"
        ),
    )
    _validate_processor_audit_git_binding(
        loaded[PROCESSOR_AUDIT_SUMMARY_PATH],
        repository_root=repository_root,
    )
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_processor_case(
    value: Any,
    *,
    name: str,
    expected_image_counts: list[int],
) -> None:
    case = _object(value, name=name)
    expected_batch_size = len(expected_image_counts)
    if (
        case.get("status") != "passed"
        or case.get("case") != name
        or case.get("batch_size") != expected_batch_size
        or case.get("image_counts") != expected_image_counts
        or case.get("padding") is not False
        or case.get("no_padding") is not True
        or case.get("attention_mask_all_one") is not True
        or case.get("assistant_prefix_tail_exact") is not True
    ):
        raise ValueError(f"processor audit case {name} boundary drifted")
    expected_equal = len(set(expected_image_counts)) == 1
    if (
        case.get("equal_image_count") is not expected_equal
        or case.get("equal_sequence_length") is not True
    ):
        raise ValueError(f"processor audit case {name} batching drifted")
    tensors = _object(case.get("tensor_inventory"), name=f"{name} tensors")
    required_tensors = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if not required_tensors.issubset(tensors) or set(tensors).difference(
        required_tensors | {"mm_token_type_ids"}
    ):
        raise ValueError(f"processor audit case {name} tensor inventory drifted")
    tensor_metadata: dict[str, tuple[list[Any], str]] = {}
    for tensor_name, raw_tensor in tensors.items():
        tensor = _object(raw_tensor, name=f"{name}.{tensor_name}")
        shape = _list(tensor.get("shape"), name=f"{name}.{tensor_name}.shape")
        if (
            not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or tensor.get("device") != "cpu"
            or tensor.get("requires_grad") is not False
            or not isinstance(tensor.get("dtype"), str)
            or not tensor.get("dtype")
        ):
            raise ValueError(f"processor audit case {name} tensor metadata drifted")
        tensor_metadata[tensor_name] = (shape, str(tensor["dtype"]))
    samples = _list(case.get("samples"), name=f"{name}.samples")
    if len(samples) != expected_batch_size:
        raise ValueError(f"processor audit case {name} sample count drifted")
    sequence_lengths: list[int] = []
    flattened_grids: list[list[int]] = []
    for index, (raw_sample, image_count) in enumerate(
        zip(samples, expected_image_counts, strict=True)
    ):
        sample = _object(raw_sample, name=f"{name}.samples[{index}]")
        grids = _list(
            sample.get("image_grid_thw"),
            name=f"{name}.samples[{index}].image_grid_thw",
        )
        per_image = _list(
            sample.get("effective_visual_tokens_per_image"),
            name=f"{name}.samples[{index}].effective_visual_tokens_per_image",
        )
        if (
            sample.get("image_count") != image_count
            or len(grids) != image_count
            or len(per_image) != image_count
            or any(
                not isinstance(grid, list)
                or len(grid) != 3
                or any(type(dimension) is not int or dimension <= 0 for dimension in grid)
                for grid in grids
            )
            or any(type(count) is not int or count <= 0 for count in per_image)
            or sample.get("effective_visual_tokens") != sum(per_image)
        ):
            raise ValueError(f"processor audit case {name} image geometry drifted")
        for grid, effective_count in zip(grids, per_image, strict=True):
            temporal, height, width = grid
            if (
                height % 2
                or width % 2
                or effective_count != temporal * height * width // 4
            ):
                raise ValueError(f"processor audit case {name} grid accounting drifted")
            flattened_grids.append(grid)
        text_tokens = _positive_int(
            sample.get("policy_visible_text_tokens"),
            name=f"{name}.samples[{index}].policy_visible_text_tokens",
        )
        sequence_length = _positive_int(
            sample.get("sequence_length"),
            name=f"{name}.samples[{index}].sequence_length",
        )
        if sequence_length != sum(per_image) + text_tokens:
            raise ValueError(f"processor audit case {name} token accounting drifted")
        sequence_lengths.append(sequence_length)
    if expected_batch_size == 2 and len(set(sequence_lengths)) != 1:
        raise ValueError("processor audit nested batch-2 sequence lengths drifted")
    common_sequence_length = sequence_lengths[0]
    expected_shapes_and_dtypes = {
        "input_ids": ([expected_batch_size, common_sequence_length], "torch.int64"),
        "attention_mask": (
            [expected_batch_size, common_sequence_length],
            "torch.int64",
        ),
        "pixel_values": (
            [
                sum(t * h * w for t, h, w in flattened_grids),
                1536,
            ],
            "torch.float32",
        ),
        "image_grid_thw": ([sum(expected_image_counts), 3], "torch.int64"),
    }
    if "mm_token_type_ids" in tensor_metadata:
        expected_shapes_and_dtypes["mm_token_type_ids"] = (
            [expected_batch_size, common_sequence_length],
            "torch.int64",
        )
    if tensor_metadata != expected_shapes_and_dtypes:
        raise ValueError(f"processor audit case {name} tensor shape/dtype drifted")
    if case != EXPECTED_PROCESSOR_CASES[name]:
        raise ValueError(f"processor audit case {name} exact evidence drifted")


def _validate_processor_audit_summary(
    summary: Mapping[str, Any],
    *,
    configured_sources: Mapping[str, Mapping[str, Any]],
    execution_runtime: Mapping[str, Any],
) -> None:
    """Validate policy-output-free use of the real pinned AutoProcessor."""
    negative_flags = (
        "policy_model_loaded",
        "policy_loaded",
        "policy_forward_executed",
        "policy_generate_executed",
        "policy_output",
        "policy_output_generated",
        "restoration_output_generated",
    )
    if (
        summary.get("schema_version") != "0.1.0"
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("evidence_type")
        != "gui_owl_v2_policy_output_free_processor_audit"
        or summary.get("outcome") != "PASSED_GUI_OWL_V2_PROCESSOR_AUDIT"
        or summary.get("auto_processor_only") is not True
        or summary.get("auto_processor_loaded") is not True
        or summary.get("model_weights_loaded") is not False
        or summary.get("pretrained_loader_calls")
        != ["AutoProcessor.from_pretrained"]
        or summary.get("model_weight_files_sha256_verified") is not True
        or summary.get("model_weights_materialized_as_tensors") is not False
        or any(summary.get(flag) is not False for flag in negative_flags)
    ):
        raise ValueError("dependency 8 real processor execution boundary drifted")

    repository = _object(summary.get("repository"), name="processor audit repository")
    if (
        not isinstance(repository.get("repository_root"), str)
        or not repository.get("repository_root")
        or not isinstance(repository.get("run_git_commit"), str)
        or GIT_SHA_PATTERN.fullmatch(repository["run_git_commit"]) is None
        or repository.get("head_verified_as_commit") is not True
        or repository.get("worktree_clean") is not True
        or repository.get("untracked_files_checked") is not True
        or repository.get("submodules_checked") is not True
    ):
        raise ValueError("dependency 8 processor audit repository identity drifted")

    source_inventory = _object(
        summary.get("source_files"), name="processor audit source files"
    )
    if set(source_inventory) != set(PROCESSOR_AUDIT_SOURCE_PATHS):
        raise ValueError("dependency 8 processor audit source inventory drifted")
    configured_by_path: dict[str, Mapping[str, Any]] = {}
    for raw_record in configured_sources.values():
        record = _object(raw_record, name="configured source")
        path = record.get("path")
        if not isinstance(path, str) or path in configured_by_path:
            raise ValueError("configured source path is invalid or duplicated")
        configured_by_path[path] = record
    for path in PROCESSOR_AUDIT_SOURCE_PATHS:
        audited = _object(source_inventory[path], name=f"processor source {path}")
        configured = configured_by_path.get(path)
        if (
            configured is None
            or audited.get("sha256") != configured.get("sha256")
            or SHA256_PATTERN.fullmatch(str(audited.get("sha256"))) is None
            or type(audited.get("size_bytes")) is not int
            or audited.get("size_bytes") <= 0
        ):
            raise ValueError(f"dependency 8 processor source hash drifted for {path}")

    runtime = _object(summary.get("runtime_identity"), name="processor runtime")
    shared_runtime_fields = (
        "host_alias",
        "host_hostname",
        "container_id",
        "container_hostname",
        "container_image_digest",
        "python_version",
        "platform_machine",
        "transformers_version",
    )
    if (
        any(runtime.get(field) != execution_runtime.get(field) for field in shared_runtime_fields)
        or runtime.get("platform_system") != "Linux"
        or runtime.get("processor_device") != "cpu"
        or runtime.get("torch_distribution_version") != "2.11.0+cu130"
        or runtime.get("pillow_version") != "12.2.0"
    ):
        raise ValueError("dependency 8 processor runtime identity drifted")

    audit = _object(summary.get("processor_audit"), name="processor audit")
    if (
        audit.get("status") != "passed"
        or audit.get("auto_processor_only") is not True
        or audit.get("auto_processor_loaded") is not True
        or audit.get("model_weights_loaded") is not False
        or audit.get("pretrained_loader_calls")
        != ["AutoProcessor.from_pretrained"]
        or audit.get("model_weight_files_sha256_verified") is not True
        or audit.get("model_weights_materialized_as_tensors") is not False
        or any(audit.get(flag) is not False for flag in negative_flags)
    ):
        raise ValueError("dependency 8 nested processor execution boundary drifted")
    model = _object(audit.get("model_identity"), name="processor model identity")
    transformer_sources = _object(
        model.get("transformers_source_sha256"),
        name="processor Transformers source inventory",
    )
    if (
        not isinstance(model.get("model_dir"), str)
        or not Path(model["model_dir"]).is_absolute()
        or model.get("model_repo") != "mPLUG/GUI-Owl-1.5-8B-Instruct"
        or model.get("model_revision")
        != "06d5faecff74840bab2be2425e9c42667a5d04fc"
        or model.get("snapshot_manifest_sha256")
        != "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3"
        or model.get("verified_model_file_count") != 14
        or model.get("verified_model_total_bytes") != 17_545_907_171
        or model.get("transformers_version") != "5.6.0"
        or transformer_sources != TRANSFORMERS_SOURCE_SHA256
    ):
        raise ValueError("dependency 8 processor model snapshot identity drifted")

    deterministic_images = _object(
        audit.get("deterministic_images"), name="processor deterministic images"
    )
    if deterministic_images != DETERMINISTIC_PROCESSOR_IMAGES:
        raise ValueError("dependency 8 processor deterministic image identity drifted")

    processor = _object(audit.get("processor_identity"), name="processor identity")
    if (
        processor.get("processor_class") != "Qwen3VLProcessor"
        or processor.get("tokenizer_class") != "Qwen2Tokenizer"
        or processor.get("image_processor_class") != "Qwen2VLImageProcessor"
        or processor.get("image_processor_module")
        != "transformers.models.qwen2_vl.image_processing_qwen2_vl"
        or processor.get("pixel_target_runtime_representation")
        != "image_processor.size.shortest_edge_longest_edge"
        or processor.get("size_class") != "SizeDict"
        or processor.get("size_module") != "transformers.image_utils"
        or processor.get("direct_min_pixels_attribute_present") is not False
        or processor.get("direct_max_pixels_attribute_present") is not False
        or processor.get("size_non_edge_fields")
        != {
            "height": None,
            "width": None,
            "max_height": None,
            "max_width": None,
        }
        or processor.get("target_effective_visual_tokens_per_image") != 2560
        or processor.get("target_pixels_per_image") != 2_621_440
        or processor.get("actual_min_pixels") != 2_621_440
        or processor.get("actual_max_pixels") != 2_621_440
        or processor.get("actual_min_equals_max_equals_target") is not True
        or processor.get("spatial_merge_size") != 2
        or processor.get("local_files_only") is not True
    ):
        raise ValueError("dependency 8 real processor pixel target drifted")

    tokenizer = _object(audit.get("tokenizer_boundary"), name="tokenizer boundary")
    if tokenizer != EXPECTED_PROCESSOR_TOKENIZER_BOUNDARY:
        raise ValueError("dependency 8 processor tokenizer boundary drifted")

    _validate_processor_case(
        audit.get("single_conversation_one_image"),
        name="single_conversation_one_image",
        expected_image_counts=[1],
    )
    _validate_processor_case(
        audit.get("single_conversation_five_images"),
        name="single_conversation_five_images",
        expected_image_counts=[5],
    )
    _validate_processor_case(
        audit.get("nested_batch_two_equal_shape"),
        name="nested_batch_two_equal_shape",
        expected_image_counts=[1, 1],
    )


def _validate_processor_audit_git_binding(
    summary: Mapping[str, Any],
    *,
    repository_root: Path,
) -> str:
    repository = _object(summary.get("repository"), name="processor audit repository")
    run_commit = repository.get("run_git_commit")
    if not isinstance(run_commit, str) or GIT_SHA_PATTERN.fullmatch(run_commit) is None:
        raise ValueError("processor audit run Git commit is invalid")
    verified = _git_output(
        repository_root,
        "rev-parse",
        "--verify",
        f"{run_commit}^{{commit}}",
    )
    if verified != run_commit:
        raise ValueError("processor audit run Git commit is unavailable")
    current_head = _git_output(repository_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "merge-base",
            "--is-ancestor",
            run_commit,
            current_head,
        ],
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("processor audit run commit is not an ancestor of current main")
    source_inventory = _object(
        summary.get("source_files"), name="processor audit source files"
    )
    for path in PROCESSOR_AUDIT_SOURCE_PATHS:
        source = _object(source_inventory.get(path), name=f"processor source {path}")
        if _sha256_bytes(_git_bytes(repository_root, run_commit, path)) != source.get(
            "sha256"
        ):
            raise ValueError(f"processor audit committed source hash drifted for {path}")
    return run_commit


def _validate_execution_config(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "execution_id",
        "status",
        "materialized_before_any_v2_policy_output",
        "scientific_contract",
        "dependencies",
        "canonical_policy",
        "canonical_data",
        "execution_runtime",
        "distance_compute",
        "batching",
        "source_files",
        "readiness_boundary",
    }
    _exact_keys(config, expected_keys, name="execution config")
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["protocol_id"] != PROTOCOL_ID
        or config["execution_id"] != "causalcache-restoration-v2-hyper00-h200-v1"
        or config["status"] != EXECUTION_STATUS
        or config["materialized_before_any_v2_policy_output"] is not True
    ):
        raise ValueError("execution config identity or pre-output status drifted")
    scientific = _validate_file_reference(
        config["scientific_contract"],
        repository_root=repository_root,
        expected_path="code/configs/causalcache_restoration_v2.json",
        name="scientific contract",
    )
    if scientific["sha256"] != SCIENTIFIC_SHA256:
        raise ValueError("scientific contract SHA256 drifted")

    policy = _object(config["canonical_policy"], name="canonical_policy")
    _exact_keys(
        policy,
        {
            "repo",
            "revision",
            "model_class",
            "dtype",
            "target_effective_visual_tokens_per_image",
            "generation",
            "snapshot_manifest",
            "processor_geometry",
        },
        name="canonical_policy",
    )
    snapshot = _object(policy.get("snapshot_manifest"), name="policy snapshot")
    snapshot_reference = _validate_file_reference(
        snapshot,
        repository_root=repository_root,
        expected_path="code/configs/gui_owl_1_5_8b_snapshot.json",
        name="policy snapshot manifest",
    )
    snapshot_payload = _load_json_object(repository_root / snapshot_reference["path"])
    if (
        policy.get("repo") != "mPLUG/GUI-Owl-1.5-8B-Instruct"
        or policy.get("revision")
        != "06d5faecff74840bab2be2425e9c42667a5d04fc"
        or policy.get("model_class") != "Qwen3VLForConditionalGeneration"
        or policy.get("dtype") != "torch.bfloat16"
        or policy.get("target_effective_visual_tokens_per_image") != 2560
        or policy.get("generation")
        != {"batch_size": 1, "do_sample": False, "max_new_tokens": 256}
        or snapshot_payload.get("repo") != policy.get("repo")
        or snapshot_payload.get("revision") != policy.get("revision")
        or len(_list(snapshot_payload.get("files"), name="snapshot files")) != 14
    ):
        raise ValueError("canonical policy contract drifted")
    geometry = _object(policy.get("processor_geometry"), name="processor_geometry")
    geometry_evidence = _object(
        geometry.get("evidence"), name="processor_geometry evidence"
    )
    _validate_file_reference(
        geometry_evidence,
        repository_root=repository_root,
        expected_path=PROCESSOR_AUDIT_SUMMARY_PATH,
        name="processor geometry evidence",
    )
    if geometry != EXPECTED_PROCESSOR_GEOMETRY:
        raise ValueError("canonical processor geometry drifted")

    data = _object(config["canonical_data"], name="canonical_data")
    if data != {
        "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
        "immutable_revision": "89f136abaff797e14fe758a198996e51032a10a6",
        "artifact_tree_sha256": (
            "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e"
        ),
        "payload_prefix": "derived/restoration-v2-v1",
        "counts": {
            "trajectory_count": 35,
            "event_count": 175,
            "state_count": 65,
            "image_member_count": 210,
            "ocr_record_count": 210,
        },
    }:
        raise ValueError("canonical derived data identity drifted")

    runtime = _object(config["execution_runtime"], name="execution_runtime")
    gpu_validation = _load_json_object(
        repository_root / "data/results/restoration_v2_gpu_compute_audit/validation.json"
    )
    audited_runtime = _object(
        gpu_validation.get("runtime_identity"), name="audited runtime identity"
    )
    expected_runtime = {
        "host_alias": audited_runtime["host_alias"],
        "host_hostname": audited_runtime["host_hostname"],
        "platform_machine": audited_runtime["platform_machine"],
        "gpu_name": audited_runtime["gpu_name"],
        "gpu_uuid": audited_runtime["gpu_uuid"],
        "nvidia_smi_gpu_uuid": audited_runtime["nvidia_smi_gpu_uuid"],
        "nvidia_driver_version": audited_runtime["nvidia_driver_version"],
        "gpu_compute_capability": audited_runtime["gpu_compute_capability"],
        "gpu_multiprocessor_count": audited_runtime["gpu_multiprocessor_count"],
        "visible_cuda_device_count": audited_runtime["visible_cuda_device_count"],
        "selected_device": audited_runtime["selected_device"],
        "container_id": audited_runtime["container_id"],
        "container_hostname": audited_runtime["container_hostname"],
        "container_image": "hongccc/sglang-omni:dev",
        "container_image_digest": audited_runtime["container_image_digest"],
        "python_version": "3.12.3",
        "torch_version": audited_runtime["torch_version"],
        "torch_cuda_build_version": audited_runtime["torch_cuda_build_version"],
        "cudnn_version": 91900,
        "transformers_version": "5.6.0",
    }
    if runtime != expected_runtime:
        raise ValueError("canonical Hyper00 runtime identity drifted")

    distance = _object(config["distance_compute"], name="distance_compute")
    if distance != {
        "operation": "teacher_forced_full_vocabulary_mean_kl_on_distance_token_span",
        "reference_log_probs_dtype": "torch.float32",
        "candidate_logits_dtype": "torch.bfloat16",
        "compute_dtype": "torch.float32",
        "output_dtype": "torch.float32",
        "numeric_validation": "gpu_resident_per_example_predicates",
        "invalid_numeric_output": "nan_final_distance",
        "validation_scalar_host_reads": 0,
        "full_tensor_host_transfers": 0,
        "host_transfer": "final_distance_scalars_only",
        "equivalence_atol": 1e-6,
        "equivalence_rtol": 1e-5,
    }:
        raise ValueError("distance compute contract drifted")

    batching = _object(config["batching"], name="batching")
    if batching != {
        "reference_generation_batch_size": 1,
        "reference_repeat_forwards": 2,
        "coalition_microbatch_size": 2,
        "planning_scope": "single_decision_state",
        "grouping_fields": ["image_count", "sequence_length"],
        "group_order": "ascending_image_count_then_sequence_length",
        "within_group_order": "ascending_input_index",
        "automatic_oom_fallback": False,
        "cross_shape_padding": False,
    }:
        raise ValueError("batching contract drifted")

    sources = _list(config["source_files"], name="source_files")
    if len(sources) != len(EXPECTED_SOURCE_PATHS):
        raise ValueError("source file inventory drifted")
    observed_roles: dict[str, dict[str, str]] = {}
    for value in sources:
        record = _object(value, name="source file")
        _exact_keys(record, {"role", "path", "sha256"}, name="source file")
        role = record["role"]
        if not isinstance(role, str) or role in observed_roles:
            raise ValueError("source role is invalid or duplicated")
        if role not in EXPECTED_SOURCE_PATHS:
            raise ValueError(f"unknown source role: {role}")
        observed_roles[role] = _validate_file_reference(
            {"path": record["path"], "sha256": record["sha256"]},
            repository_root=repository_root,
            expected_path=EXPECTED_SOURCE_PATHS[role],
            name=f"source {role}",
        )
    if set(observed_roles) != set(EXPECTED_SOURCE_PATHS):
        raise ValueError("required runtime/audit source roles drifted")

    boundary = _object(config["readiness_boundary"], name="readiness_boundary")
    if boundary != {
        "readiness_manifest_path": READINESS_MANIFEST_PATH,
        "implementation_git_commit_source": "external_readiness_manifest",
        "dependency_count": 8,
        "state_after_completion": READINESS_STATUS,
        "confirm_state_after_completion": CONFIRM_STATE,
        "allowed_roles": list(ALLOWED_ROLES),
        "forbidden_roles": ["v2_confirm_primary"],
        "policy_output_generated_before_config": False,
        "restoration_output_generated_before_config": False,
    }:
        raise ValueError("readiness boundary drifted")
    dependencies = _validate_dependencies(config, repository_root=repository_root)
    return {
        "dependencies": dependencies,
        "source_files": observed_roles,
    }


def _validate_readiness_manifest(
    manifest: Mapping[str, Any],
    *,
    config_path: str,
    config_sha256: str,
    source_files: Mapping[str, Mapping[str, str]],
) -> str:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "confirm_locked",
        "dependency_count",
        "passed_dependency_count",
        "execution_config",
        "source_files",
        "implementation_git_commit",
        "canonical_remote",
        "branch",
        "materialized_before_any_v2_policy_output",
        "policy_output_generated_before_manifest",
        "restoration_output_generated_before_manifest",
    }
    _exact_keys(manifest, expected_keys, name="readiness manifest")
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["protocol_id"] != PROTOCOL_ID
        or manifest["status"] != READINESS_STATUS
        or manifest["confirm_locked"] is not True
        or manifest["dependency_count"] != 8
        or manifest["passed_dependency_count"] != 8
        or manifest["canonical_remote"] != CANONICAL_REMOTE
        or manifest["branch"] != "main"
        or manifest["materialized_before_any_v2_policy_output"] is not True
        or manifest["policy_output_generated_before_manifest"] is not False
        or manifest["restoration_output_generated_before_manifest"] is not False
    ):
        raise ValueError("readiness manifest state or pre-output boundary drifted")
    config_record = _object(manifest["execution_config"], name="manifest config")
    if config_record != {"path": config_path, "sha256": config_sha256}:
        raise ValueError("readiness manifest execution config binding drifted")
    manifest_sources = _list(manifest["source_files"], name="manifest source_files")
    expected_sources = [
        {"role": role, **source_files[role]} for role in sorted(source_files)
    ]
    if manifest_sources != expected_sources:
        raise ValueError("readiness manifest source hashes differ from execution config")
    commit = manifest["implementation_git_commit"]
    if not isinstance(commit, str) or GIT_SHA_PATTERN.fullmatch(commit) is None:
        raise ValueError("readiness implementation Git commit is invalid")
    return commit


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_bytes(repository_root: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository_root), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def _validate_repository_binding(
    repository_root: Path,
    *,
    implementation_commit: str,
    execution_config_path: str,
    execution_config_sha256: str,
    source_files: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    root = repository_root.resolve()
    status = _git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise ValueError("screening authorization requires a clean Git worktree")
    head = _git_output(root, "rev-parse", "HEAD")
    origin_main = _git_output(root, "rev-parse", "origin/main")
    if head != origin_main:
        raise ValueError("screening authorization requires HEAD == origin/main")
    if _git_output(root, "remote", "get-url", "origin") != CANONICAL_REMOTE:
        raise ValueError("canonical Git remote drifted")
    verified = _git_output(root, "rev-parse", "--verify", f"{implementation_commit}^{{commit}}")
    if verified != implementation_commit:
        raise ValueError("implementation Git commit is not locally available")
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", implementation_commit, head],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("implementation Git commit is not an ancestor of current main")
    if _sha256_bytes(_git_bytes(root, implementation_commit, execution_config_path)) != (
        execution_config_sha256
    ):
        raise ValueError("implementation commit does not contain the bound execution config")
    for role, record in source_files.items():
        if _sha256_bytes(_git_bytes(root, implementation_commit, record["path"])) != record[
            "sha256"
        ]:
            raise ValueError(f"implementation commit source hash drifted for {role}")
    return {
        "implementation_git_commit": implementation_commit,
        "current_git_commit": head,
        "origin_main_git_commit": origin_main,
    }


def validate_screening_authorization(
    execution_config_path: str | Path,
    readiness_manifest_path: str | Path,
    repository_root: str | Path,
    require_role: str = "screening",
) -> dict[str, object]:
    """Return SCREENING_ALLOWED only after all eight dependencies and Git bind pass."""
    root = Path(repository_root).resolve()
    config_file = Path(execution_config_path)
    manifest_file = Path(readiness_manifest_path)
    if not config_file.is_absolute():
        config_file = _resolve(root, str(config_file), name="execution_config_path")
    if not manifest_file.is_absolute():
        manifest_file = _resolve(root, str(manifest_file), name="readiness_manifest_path")
    if require_role != "screening":
        raise ValueError("readiness validator authorizes screening only; confirm remains locked")
    if manifest_file.resolve() != (root / READINESS_MANIFEST_PATH).resolve():
        raise ValueError("readiness manifest path drifted")
    config_relative = config_file.resolve().relative_to(root).as_posix()
    config = _load_json_object(config_file)
    validated = _validate_execution_config(config, repository_root=root)
    config_sha256 = _sha256_file(config_file)
    manifest = _load_json_object(manifest_file)
    implementation_commit = _validate_readiness_manifest(
        manifest,
        config_path=config_relative,
        config_sha256=config_sha256,
        source_files=validated["source_files"],
    )
    git_binding = _validate_repository_binding(
        root,
        implementation_commit=implementation_commit,
        execution_config_path=config_relative,
        execution_config_sha256=config_sha256,
        source_files=validated["source_files"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "state": READINESS_STATUS,
        "confirm_state": CONFIRM_STATE,
        "confirm_locked": True,
        "allowed_roles": list(ALLOWED_ROLES),
        "dependency_count": 8,
        "passed_dependency_count": 8,
        "dependencies": validated["dependencies"],
        "execution_config": {"path": config_relative, "sha256": config_sha256},
        "git": git_binding,
        "policy_imported_by_validator": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-config", required=True, type=Path)
    parser.add_argument("--readiness-manifest", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--require-role", default="screening", choices=("screening",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = validate_screening_authorization(
        args.execution_config,
        args.readiness_manifest,
        args.repository_root,
        require_role=args.require_role,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
