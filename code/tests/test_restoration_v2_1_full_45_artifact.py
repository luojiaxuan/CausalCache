from __future__ import annotations

import copy
import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import causalcache.restoration_v2_1_full_45_artifact as artifact_module
from causalcache.data.restoration_v2_screening import ScreeningState
from causalcache.policy.gui_owl_v2 import gui_owl_v2_action_to_androidworld
from causalcache.policy.gui_owl_v2_1 import parse_gui_owl_v2_1_output
from causalcache.restoration_v2_1_full_45_artifact import (
    AGGREGATE_FILENAME,
    ARCHIVE_LEDGER_NAME,
    ARCHIVE_MEMBER_PREFIX,
    BFLOAT16_MIN_FINITE,
    CANONICAL_GLOBAL_LEDGER_PATH,
    CANONICAL_HF_PATH,
    CANONICAL_HF_REPO,
    CANONICAL_RAW_OUTPUT_DIR,
    EXPECTED_COMPUTATION_SCHEDULE,
    EXPECTED_PROMOTION,
    EXPECTED_RUN_SOURCE_PATHS,
    FULL_45_CONFIG_PATH,
    INVALID_OUTCOME,
    NO_GO_OUTCOME,
    PASS_OUTCOME,
    PILOT_ARTIFACT_PATH,
    PILOT_CONTRACT_PATH,
    PROCESSOR_ARTIFACT_PATH,
    RUN_MANIFEST_FILENAME,
    RUNTIME_IDENTITY_FILENAME,
    SELECTION_MANIFEST_PATH,
    V2_CONFIG_PATH,
    _write_deterministic_tar,
    build_full_45_artifact_manifest,
    canonical_json_bytes,
    package_raw_full_45_evidence,
    pretty_json_bytes,
    read_extracted_full_45_evidence,
    read_full_45_evidence_archive,
    sha256_bytes,
    validate_full_45_artifact_manifest,
    validate_full_45_evidence_files,
    validate_source_x_run_contract,
)
from causalcache.restoration_v2_1_full_45_contract import (
    FROZEN_RESTORATION_V2_1_FULL_45_SHA256,
    _full_45_projection,
)
from scripts.manage_restoration_v2_1_full_45_artifact import _build_parser
from scripts.run_restoration_v2_1_full_45_substrate import (
    ATTEMPT_STATUS,
    FORBIDDEN_OPERATION_COUNTS,
    GLOBAL_ATTEMPT_STATUS,
    PROTOCOL_ID,
    RUN_STATUS,
    SCHEMA_VERSION,
    STATE_OUTCOME_FAILED,
    STATE_OUTCOME_VALID,
    Full45Gate,
    _base_state_record,
    _full45_projection,
    _prepare_layout,
    _runtime_identity_record,
    aggregate_full45_gate,
)


SOURCE_COMMIT = "a" * 40
HF_REVISION = "b" * 40
RUN_STARTED = "2026-07-15T00:00:00Z"
RUN_ENDED = "2026-07-15T00:01:00Z"
VALID_OUTPUT = (
    '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n'
    "</tool_call>"
)
INVALID_OUTPUT = (
    '<tool_call>\n{"name":"mobile_use","arguments":{"action":"bogus"}}\n'
    "</tool_call>"
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _source_blobs(repository_root: Path) -> dict[str, bytes]:
    return {
        relative: (repository_root / relative).read_bytes()
        for relative in EXPECTED_RUN_SOURCE_PATHS
    }


def _runtime_metadata(repository_root: Path) -> dict[str, Any]:
    pilot = _json(repository_root / PILOT_CONTRACT_PATH)
    preflight = pilot["processor_preflight"]
    return {
        "protocol_id": "causalcache_restoration_v2_1_official_tool_interface",
        "generation_interface": "processor_apply_chat_template_official_tools_kwarg",
        "official_tools_argument_count": 1,
        "official_tool_schema_sha256": preflight["canonical_tool_schema_sha256"],
        "chat_template_file_sha256": preflight["chat_template_file_sha256"],
        "chat_template_text_sha256": preflight["chat_template_text_sha256"],
        "assistant_prefix_token_ids": [151644, 77091, 198],
        "tool_call_open_token_id": 151657,
        "tool_call_close_token_id": 151658,
        "generation_eos_token_id": 151658,
        "suppressed_standard_eos_token_ids": [151645, 151643],
        "generation_pad_token_id": 151643,
        "generation_num_beams": 1,
        "generation_num_return_sequences": 1,
        "generation_standard_eos_suppression": (
            "negative_infinity_via_transformers_suppress_tokens"
        ),
        "host_injected_tool_call_closer": False,
        "output_recovery_or_normalization": False,
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "frozen": True,
        "max_pixels": 2_621_440,
        "min_pixels": 2_621_440,
        "model_class": "GUIOwlForConditionalGeneration",
        "model_dir": "/data/artifacts/models/GUI-Owl-1.5-8B-Instruct",
        "model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
        "model_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
        "processor_class": "GUIOwlProcessor",
        "single_device": True,
        "snapshot_manifest_sha256": pilot["primary_policy"]["snapshot_manifest"][
            "sha256"
        ],
        "target_effective_visual_tokens_per_image": 2560,
        "torch_version": "2.11.0+cu130",
        "transformers_source_sha256": "8" * 64,
        "transformers_version": "5.6.0",
        "verified_model_file_count": 10,
        "verified_model_total_bytes": 1_000_000,
    }


def _host_runtime_identity() -> dict[str, Any]:
    from scripts.run_restoration_v2_1_interface_pilot import (
        CANONICAL_PILOT_CONTAINER_ID,
        CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        CANONICAL_PILOT_HOST_ALIAS,
        CANONICAL_PILOT_HOST_HOSTNAME,
    )

    uuid = "TEST-UUID"
    return {
        "declared_host": {
            "alias": CANONICAL_PILOT_HOST_ALIAS,
            "hostname": CANONICAL_PILOT_HOST_HOSTNAME,
            "verification": "requires_independent_host_preflight",
        },
        "verified_container_runtime": {
            "container_id": CANONICAL_PILOT_CONTAINER_ID,
            "container_hostname": CANONICAL_PILOT_CONTAINER_ID[:12],
            "verification": "live_hostname_prefix_of_full_container_id",
        },
        "declared_container_image": {
            "digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
            "verification": "requires_independent_host_preflight",
        },
        "selected_device": "cuda:0",
        "live": {
            "platform_machine": "x86_64",
            "gpu_name": "NVIDIA H200",
            "gpu_uuid": uuid,
            "nvidia_smi_gpu_uuid": f"GPU-{uuid}",
            "nvidia_driver_version": "570.172.08",
            "gpu_compute_capability": [9, 0],
            "gpu_multiprocessor_count": 132,
            "visible_cuda_device_count": 1,
            "selected_device": "cuda:0",
            "python_version": "3.12.11",
            "torch_version": "2.11.0+cu130",
            "torch_cuda_build_version": "13.0",
            "cudnn_version": 91000,
            "transformers_version": "5.6.0",
        },
    }


def _attempt_identity() -> dict[str, Any]:
    from scripts.run_restoration_v2_1_interface_pilot import (
        CANONICAL_PILOT_CONTAINER_ID,
        CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        CANONICAL_PILOT_HOST_ALIAS,
        CANONICAL_PILOT_HOST_HOSTNAME,
    )

    return {
        "attempt_id": "restoration-v2-1-full-45-substrate-v1",
        "canonical_persistent_output_dir": str(CANONICAL_RAW_OUTPUT_DIR),
        "canonical_global_attempt_ledger": str(CANONICAL_GLOBAL_LEDGER_PATH),
        "canonical_host_alias": CANONICAL_PILOT_HOST_ALIAS,
        "canonical_host_hostname": CANONICAL_PILOT_HOST_HOSTNAME,
        "canonical_container_id": CANONICAL_PILOT_CONTAINER_ID,
        "canonical_container_image_digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        "canonical_device": "cuda:0",
        "cross_host_attempt_allowed": False,
        "alternate_output_or_ledger_allowed": False,
        "output_or_ledger_deletion_after_first_attempt_allowed": False,
    }


def _run_contract(repository_root: Path, blobs: dict[str, bytes]) -> dict[str, Any]:
    child = json.loads(blobs[FULL_45_CONFIG_PATH])
    pilot_manifest = json.loads(blobs[PILOT_ARTIFACT_PATH])
    processor_manifest = json.loads(blobs[PROCESSOR_ARTIFACT_PATH])
    pilot_raw = pilot_manifest["raw_archive"]
    pilot_source = pilot_manifest["source_execution"]
    processor_raw = processor_manifest["raw_evidence"]
    compact = processor_manifest["compact_reduction"]
    selection = json.loads(blobs[SELECTION_MANIFEST_PATH])
    states = _full_45_projection(selection)
    runtime = _runtime_metadata(repository_root)
    canonical_inputs = {
        "contract": {
            "path": FULL_45_CONFIG_PATH,
            "sha256": FROZEN_RESTORATION_V2_1_FULL_45_SHA256,
        },
        "pilot_contract": {
            "path": PILOT_CONTRACT_PATH,
            "sha256": sha256_bytes(blobs[PILOT_CONTRACT_PATH]),
        },
        "pilot_evidence": {
            "external": {
                "path": "/data/evidence/pilot.tar",
                "sha256": pilot_raw["sha256"],
                "size_bytes": pilot_raw["size_bytes"],
            },
            "artifact_manifest_path": PILOT_ARTIFACT_PATH,
            "artifact_manifest_sha256": sha256_bytes(blobs[PILOT_ARTIFACT_PATH]),
        },
        "processor_preflight": {
            "external": {
                "path": "/data/evidence/processor.json",
                "sha256": processor_raw["sha256"],
                "size_bytes": processor_raw["size_bytes"],
            },
            "artifact_manifest_path": PROCESSOR_ARTIFACT_PATH,
            "artifact_manifest_sha256": sha256_bytes(
                blobs[PROCESSOR_ARTIFACT_PATH]
            ),
        },
        "scientific_config": {
            "path": V2_CONFIG_PATH,
            "sha256": sha256_bytes(blobs[V2_CONFIG_PATH]),
        },
        "selection_manifest": {
            "path": SELECTION_MANIFEST_PATH,
            "sha256": sha256_bytes(blobs[SELECTION_MANIFEST_PATH]),
        },
        "ocr_backend_config": {
            "path": "code/configs/restoration_v2_ocr_backend.json",
            "sha256": sha256_bytes(
                blobs["code/configs/restoration_v2_ocr_backend.json"]
            ),
        },
        "snapshot_manifest": {
            "path": "code/configs/gui_owl_1_5_8b_snapshot.json",
            "sha256": sha256_bytes(
                blobs["code/configs/gui_owl_1_5_8b_snapshot.json"]
            ),
        },
    }
    parent = {
        "pilot_artifact_validation": {
            "status": "VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT",
            "source_git_commit": pilot_source["source_git_commit"],
            "outcome": pilot_manifest["result"]["outcome"],
            "file_count": pilot_raw["file_count"],
            "tree_inventory_sha256": pilot_raw["tree_inventory_sha256"],
            "archive_hash_verified": True,
            "current_git_commit": SOURCE_COMMIT,
            "evidence_source_kind": "raw_archive",
        },
        "processor_audit": {
            "status": compact["status"],
            "prompt_count": compact["prompt_count"],
            "prompt_records_sha256": compact["prompt_records_sha256"],
            "shape_records_sha256": compact["shape_records_sha256"],
            "teacher_golden_records_sha256": compact[
                "teacher_golden_records_sha256"
            ],
            "processor_classes_sha256": compact["processor_classes_sha256"],
            "evidence_git_commit": processor_raw["source_git_commit"],
            "current_git_commit": SOURCE_COMMIT,
            "validation_mode": "reuse",
        },
    }
    policy = {
        "repo": runtime["model_repo"],
        "revision": runtime["model_revision"],
        "model_dir": runtime["model_dir"],
        "runtime_metadata_requirement": {
            "required_policy_protocol_id": runtime["protocol_id"],
            "validated_after_claim_before_generation": True,
            "native_generation_metadata_persisted_twice_per_state": True,
            "teacher_metadata_persisted_three_times_per_eligible_state": True,
            "required_metadata_keys": sorted(runtime),
            "model_dir": runtime["model_dir"],
            "model_repo": runtime["model_repo"],
            "model_revision": runtime["model_revision"],
            "snapshot_manifest_sha256": runtime["snapshot_manifest_sha256"],
            "device": "cuda:0",
            "target_effective_visual_tokens_per_image": 2560,
        },
    }
    argv_values = {
        "--repository-root": "/data/CausalCache",
        "--contract": f"/data/CausalCache/{FULL_45_CONFIG_PATH}",
        "--pilot-evidence": "/data/evidence/pilot.tar",
        "--processor-preflight": "/data/evidence/processor.json",
        "--derived-artifact-root": "/data/derived/restoration-v2-v1",
        "--scientific-config": f"/data/CausalCache/{V2_CONFIG_PATH}",
        "--selection-manifest": f"/data/CausalCache/{SELECTION_MANIFEST_PATH}",
        "--ocr-backend-config": (
            "/data/CausalCache/code/configs/restoration_v2_ocr_backend.json"
        ),
        "--model-dir": runtime["model_dir"],
        "--device": "cuda:0",
        "--host-alias": "hyper00",
        "--host-hostname": "node-radixark-16-0000",
        "--container-id": _attempt_identity()["canonical_container_id"],
        "--container-image-digest": _attempt_identity()[
            "canonical_container_image_digest"
        ],
        "--output-dir": str(CANONICAL_RAW_OUTPUT_DIR),
    }
    argv = [
        "/usr/bin/python3",
        "/data/CausalCache/code/scripts/run_restoration_v2_1_full_45_substrate.py",
    ]
    for flag, value in argv_values.items():
        argv.extend([flag, value])
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_sha256": FROZEN_RESTORATION_V2_1_FULL_45_SHA256,
        "git_identity": {
            "branch": "main",
            "commit": SOURCE_COMMIT,
            "origin_main": SOURCE_COMMIT,
            "remote_main": SOURCE_COMMIT,
            "remote_url": "https://github.com/luojiaxuan/CausalCache.git",
            "worktree": "clean_including_untracked",
        },
        "source_inventory": [
            {
                "path": path,
                "sha256": sha256_bytes(blobs[path]),
                "git_commit": SOURCE_COMMIT,
            }
            for path in EXPECTED_RUN_SOURCE_PATHS
        ],
        "parent_authorization": parent,
        "canonical_inputs": canonical_inputs,
        "artifact": {
            "artifact_tree_sha256": child["data"]["derived_artifact"][
                "artifact_tree_sha256"
            ],
            "artifact_manifest_sha256": "6" * 64,
            "screening_manifest_sha256": "7" * 64,
            "derived_repo": dict(child["data"]["derived_artifact"]),
        },
        "policy": policy,
        "runtime_identity": _host_runtime_identity(),
        "execution_argv": argv,
        "operational_argv_policy": {
            "resume_flag_excluded_from_scientific_run_identity": True,
            "initial_invocation_is_recorded_without_resume": True,
            "resume_only_skips_terminal_prefix": True,
        },
        "seed_policy": {
            "decoding": "greedy_do_sample_false",
            "random_seed": None,
            "sampling_seed_not_applicable": True,
        },
        "output_dir": str(CANONICAL_RAW_OUTPUT_DIR),
        "attempt_identity": _attempt_identity(),
        "states": states,
        "computation_schedule": copy.deepcopy(EXPECTED_COMPUTATION_SCHEDULE),
        "prohibited_operation_counts": dict(FORBIDDEN_OPERATION_COUNTS),
        "confirm_state": "LOCKED_NO_PROMPT_IMAGE_DECODER_OR_POLICY_ACCESS",
        "promotion": dict(EXPECTED_PROMOTION),
    }


def _generation_metadata(repository_root: Path, output: str) -> dict[str, Any]:
    runtime = _runtime_metadata(repository_root)
    return {
        "protocol_id": runtime["protocol_id"],
        "generation_interface": runtime["generation_interface"],
        "official_tools_argument_count": 1,
        "official_tool_schema_sha256": runtime["official_tool_schema_sha256"],
        "chat_template_file_sha256": runtime["chat_template_file_sha256"],
        "chat_template_text_sha256": runtime["chat_template_text_sha256"],
        "assistant_prefix_token_ids": runtime["assistant_prefix_token_ids"],
        "tool_call_open_token_id": 151657,
        "tool_call_close_token_id": 151658,
        "generation_eos_token_id": 151658,
        "suppressed_standard_eos_token_ids": [151645, 151643],
        "generation_pad_token_id": 151643,
        "generation_num_beams": 1,
        "generation_num_return_sequences": 1,
        "generation_standard_eos_suppression": runtime[
            "generation_standard_eos_suppression"
        ],
        "host_injected_tool_call_closer": False,
        "output_recovery_or_normalization": False,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "max_new_tokens": 256,
        "decoded_output_utf8_sha256": sha256_bytes(output.encode()),
        "generated_tool_call_close_token_count": 1,
        "final_generated_token_id": 151658,
        "termination_reason": "model_emitted_tool_call_close",
        "model_emitted_tool_call_close": True,
    }


def _teacher_metadata(repository_root: Path) -> dict[str, Any]:
    runtime = _runtime_metadata(repository_root)
    interface_keys = {
        "protocol_id",
        "generation_interface",
        "official_tools_argument_count",
        "official_tool_schema_sha256",
        "chat_template_file_sha256",
        "chat_template_text_sha256",
        "assistant_prefix_token_ids",
        "tool_call_open_token_id",
        "tool_call_close_token_id",
        "generation_eos_token_id",
        "suppressed_standard_eos_token_ids",
        "generation_pad_token_id",
        "generation_num_beams",
        "generation_num_return_sequences",
        "generation_standard_eos_suppression",
        "host_injected_tool_call_closer",
        "output_recovery_or_normalization",
    }
    return {
        **{key: runtime[key] for key in interface_keys},
        "batch_size": 1,
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "vocabulary_size": 151_669,
        "distance_span": "official_tool_call_open_through_close_inclusive",
        "teacher_context": "official_tools_prompt_plus_assistant_prefix_direct",
        "teacher_carrier": None,
        "teacher_target_json_separators": [", ", ": "],
        "teacher_target_ends_with_model_generation_eos": True,
        "teacher_target_disjoint_from_suppressed_standard_eos": True,
        "teacher_standard_eos_suppressed_token_ids": [151645, 151643],
        "teacher_standard_eos_suppression_value": BFLOAT16_MIN_FINITE,
        "teacher_standard_eos_suppression_semantics": (
            "torch_finfo_bfloat16_min_finite_generation_alignment"
        ),
        "teacher_standard_eos_mask_application": (
            "same_mask_on_every_reference_and_candidate_action_path_position_"
            "before_float32_log_softmax"
        ),
        "teacher_raw_logits_mutated": False,
        "finite_logits_validation": (
            "deferred_to_gpu_kl_invalid_to_nan_final_distance"
        ),
        "extended_prompt_aligned_inputs": [],
        "logits_to_keep": 5,
        "latency_seconds": 0.1,
        "peak_gpu_memory_allocated_bytes": 100,
        "peak_gpu_memory_reserved_bytes": 200,
        "full_logit_tensor_host_transfers": 0,
        "samples": [{"distance_action_tokens": 5}],
    }


def _distance_audit() -> dict[str, Any]:
    return {
        "operation": "teacher_forced_full_vocabulary_mean_kl_on_distance_token_span",
        "candidate_representation": "logits",
        "batch_size": 1,
        "distance_tokens": 5,
        "vocabulary_size": 151_669,
        "device": "cuda:0",
        "reference_input_dtype": "torch.float32",
        "candidate_input_dtype": "torch.bfloat16",
        "compute_dtype": "torch.float32",
        "output_dtype": "torch.float32",
        "reference_batch_stride": 758_345,
        "reference_zero_copy_batch_expansion": False,
        "reference_compute_batch_size": 1,
        "reduction": "full_vocabulary_sum_then_distance_token_mean_per_example",
        "log_normalization_atol": 5e-4,
        "negative_kl_atol": 1e-5,
        "numeric_validation": "gpu_resident_per_example_predicates",
        "invalid_numeric_output": "nan_final_distance",
        "device_validation_category_count": 4,
        "validation_scalar_host_reads": 0,
        "full_tensor_host_transfers": 0,
    }


def _state_record(
    repository_root: Path,
    projection: dict[str, Any],
    contract_sha256: str,
    *,
    parse_failure: bool,
) -> dict[str, Any]:
    record = _base_state_record(
        projection=projection,
        run_contract_sha256=contract_sha256,
    )
    output = INVALID_OUTPUT if parse_failure else VALID_OUTPUT
    generation_metadata = _generation_metadata(repository_root, output)
    native = [
        {
            "repeat_index": repeat,
            "output_text": output,
            "metadata": generation_metadata,
            "canonical_action": None,
            "androidworld_bridge": None,
            "parse_error_type": "ValueError",
            "parse_error_message": "unsupported action",
        }
        for repeat in (1, 2)
    ]
    record.update(
        {
            "screen_dimensions": {"width": 1080, "height": 1920},
            "native_generations": native,
            "model_emitted_closer_count": 2,
            "started_at_utc": "2026-07-15T00:00:01Z",
            "ended_at_utc": "2026-07-15T00:00:02Z",
            "duration_seconds": 1.0,
            "operation_counts": {
                "generation_call_count": 2,
                "teacher_forward_count": 0,
                "kl_measurement_count": 0,
                **FORBIDDEN_OPERATION_COUNTS,
            },
        }
    )
    if parse_failure:
        record.update(
            {
                "outcome": STATE_OUTCOME_FAILED,
                "failure": {
                    "stage": "reference_generations",
                    "category": "PARSE_FAILURE",
                    "exception_type": "ValueError",
                    "message": "2 of 2 strict reference generations failed",
                },
            }
        )
        return record
    parsed = parse_gui_owl_v2_1_output(VALID_OUTPUT)
    action = parsed.canonical_action.arguments()
    bridge = gui_owl_v2_action_to_androidworld(
        parsed.canonical_action,
        screen_width=1080,
        screen_height=1920,
    )
    for generation in native:
        generation.update(
            {
                "canonical_action": action,
                "androidworld_bridge": bridge,
                "parse_error_type": None,
                "parse_error_message": None,
            }
        )
    teacher = _teacher_metadata(repository_root)
    record.update(
        {
            "outcome": STATE_OUTCOME_VALID,
            "failure": None,
            "parse_success": True,
            "parse_success_count": 2,
            "androidworld_bridge_count": 2,
            "repeat_canonical_action_agreement": True,
            "finite_logit_distances": True,
            "canonical_action": action,
            "androidworld_bridge": bridge,
            "teacher_forwards": {
                "reference_1": copy.deepcopy(teacher),
                "reference_2": copy.deepcopy(teacher),
                "summary_only": copy.deepcopy(teacher),
            },
            "distances": {
                "repeat_reference_kl": 1e-6,
                "summary_reference_kl": 0.01,
            },
            "distance_audits": {
                "repeat_reference_kl": _distance_audit(),
                "summary_reference_kl": _distance_audit(),
            },
            "operation_counts": {
                "generation_call_count": 2,
                "teacher_forward_count": 3,
                "kl_measurement_count": 2,
                **FORBIDDEN_OPERATION_COUNTS,
            },
        }
    )
    return record


def _gate() -> Full45Gate:
    return Full45Gate(
        minimum_screening_states=20,
        minimum_parse_coverage=0.99,
        minimum_finite_logit_coverage=1.0,
        minimum_repeat_canonical_action_agreement=1.0,
        minimum_memory_sensitive_states=8,
        pass_outcome=PASS_OUTCOME,
        fail_outcome=NO_GO_OUTCOME,
        invalid_outcome=INVALID_OUTCOME,
    )


def _states(projections: list[dict[str, Any]]) -> tuple[ScreeningState, ...]:
    return tuple(
        ScreeningState(
            index=record["index"],
            role=record["role"],
            trajectory_id=record["trajectory_id"],
            decision_step_id=record["decision_step_id"],
            candidate_event_step_ids=tuple(record["candidate_event_step_ids"]),
        )
        for record in projections
    )


def _ledger(
    run_contract_sha256: str,
    *,
    completed: int,
    attempted: int,
) -> dict[str, Any]:
    journal = [
        {
            "sequence": 0,
            "event": "GLOBAL_ATTEMPT_CLAIMED",
            "state_index": None,
            "attempted_state_count": 0,
            "completed_state_count": 0,
            "created_at_utc": RUN_STARTED,
        }
    ]
    attempted_high_water = 0
    completed_high_water = 0
    while attempted_high_water < attempted:
        index = attempted_high_water
        attempted_high_water += 1
        journal.append(
            {
                "sequence": len(journal),
                "event": "STATE_ATTEMPT_CLAIMED",
                "state_index": index,
                "attempted_state_count": attempted_high_water,
                "completed_state_count": completed_high_water,
                "created_at_utc": RUN_STARTED,
            }
        )
        if completed_high_water < completed:
            completed_high_water += 1
            journal.append(
                {
                    "sequence": len(journal),
                    "event": "STATE_TERMINAL_PERSISTED",
                    "state_index": index,
                    "attempted_state_count": attempted_high_water,
                    "completed_state_count": completed_high_water,
                    "created_at_utc": RUN_STARTED,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": GLOBAL_ATTEMPT_STATUS,
        "attempt_identity": _attempt_identity(),
        "run_contract_sha256": run_contract_sha256,
        "created_at_utc": RUN_STARTED,
        "attempted_state_count": attempted,
        "completed_state_count": completed,
        "journal": journal,
    }


def _attempt_record(
    projection: dict[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": ATTEMPT_STATUS,
        "run_contract_sha256": run_contract_sha256,
        "state": projection,
        "attempt_ordinal": 1,
        "retry_count": 0,
        "top_up_count": 0,
        "created_at_utc": "2026-07-15T00:00:01Z",
    }


def _complete_files(
    repository_root: Path,
    *,
    no_go: bool = False,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    blobs = _source_blobs(repository_root)
    run_contract = _run_contract(repository_root, blobs)
    contract_sha256 = sha256_bytes(canonical_json_bytes(run_contract))
    projections = run_contract["states"]
    states = _states(projections)
    runtime_metadata = _runtime_metadata(repository_root)
    records: list[dict[str, Any]] = []
    files: dict[str, bytes] = {}
    for index, projection in enumerate(projections):
        record = _state_record(
            repository_root,
            projection,
            contract_sha256,
            parse_failure=no_go and index == 0,
        )
        records.append(record)
        files[f"states/{index:03d}.json"] = pretty_json_bytes(record)
        files[f"attempts/{index:03d}.json"] = pretty_json_bytes(
            _attempt_record(projection, contract_sha256)
        )
    aggregate = aggregate_full45_gate(
        records,
        expected_states=states,
        gate=_gate(),
        run_contract_sha256=contract_sha256,
        started_at_utc=RUN_STARTED,
        ended_at_utc=RUN_ENDED,
        runtime_metadata=runtime_metadata,
    )
    files[RUN_MANIFEST_FILENAME] = pretty_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": RUN_STATUS,
            "run_contract_sha256": contract_sha256,
            "run_contract": run_contract,
            "created_at_utc": RUN_STARTED,
        }
    )
    files[AGGREGATE_FILENAME] = pretty_json_bytes(aggregate)
    layout = _prepare_layout(
        states=states,
        run_contract=run_contract,
        output_dir=CANONICAL_RAW_OUTPUT_DIR,
    )
    files[RUNTIME_IDENTITY_FILENAME] = pretty_json_bytes(
        _runtime_identity_record(
            layout=layout,
            runtime_metadata=runtime_metadata,
        )
    )
    files[ARCHIVE_LEDGER_NAME] = pretty_json_bytes(
        _ledger(contract_sha256, completed=45, attempted=45)
    )
    return files, blobs


def _invalid_files(
    repository_root: Path,
    *,
    completed: int,
    attempted: int,
    observed_completed: int,
    observed_attempted: int,
    stage: str,
    include_runtime: bool,
) -> dict[str, bytes]:
    complete, _ = _complete_files(repository_root)
    manifest = json.loads(complete[RUN_MANIFEST_FILENAME])
    contract_sha256 = manifest["run_contract_sha256"]
    files = {
        RUN_MANIFEST_FILENAME: complete[RUN_MANIFEST_FILENAME],
        ARCHIVE_LEDGER_NAME: pretty_json_bytes(
            _ledger(contract_sha256, completed=completed, attempted=attempted)
        ),
    }
    for index in range(observed_attempted):
        files[f"attempts/{index:03d}.json"] = complete[
            f"attempts/{index:03d}.json"
        ]
    for index in range(observed_completed):
        files[f"states/{index:03d}.json"] = complete[f"states/{index:03d}.json"]
    if include_runtime:
        files[RUNTIME_IDENTITY_FILENAME] = complete[RUNTIME_IDENTITY_FILENAME]
    observed = {
        "state_entries": [f"{index:03d}.json" for index in range(observed_completed)],
        "attempt_entries": [
            f"{index:03d}.json" for index in range(observed_attempted)
        ],
        "state_file_count": observed_completed,
        "attempt_marker_file_count": observed_attempted,
        "all_entries_are_regular_files": True,
        "contiguous_prefix": (
            observed_completed <= observed_attempted
            and observed_attempted - observed_completed <= 1
        ),
    }
    files[AGGREGATE_FILENAME] = pretty_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "run_contract_sha256": contract_sha256,
            "status": "TERMINATED_INVALID_V2_1_FULL_45_SUBSTRATE",
            "outcome": INVALID_OUTCOME,
            "gate_passed": False,
            "started_at_utc": RUN_STARTED,
            "ended_at_utc": RUN_ENDED,
            "duration_seconds": 60.0,
            "invalid_failure": {
                "stage": stage,
                "category": "CONTRACT_OR_RUNTIME",
                "exception_type": "RuntimeError",
                "message": "synthetic canonical INVALID",
            },
            "completed_state_count": completed,
            "attempted_state_count": attempted,
            "observed_inventory": observed,
            "retry_count": 0,
            "top_up_count": 0,
            "prohibited_operation_counts": {
                key: 0
                for key in FORBIDDEN_OPERATION_COUNTS
                if key not in {"retry_count", "top_up_count"}
            },
        }
    )
    return files


def _write_raw_tree(root: Path, files: dict[str, bytes]) -> tuple[Path, Path]:
    run = root / ARCHIVE_MEMBER_PREFIX
    (run / "states").mkdir(parents=True)
    (run / "attempts").mkdir()
    ledger = root / ".full-45-ledger.json"
    ledger.write_bytes(files[ARCHIVE_LEDGER_NAME])
    for relative, payload in files.items():
        if relative == ARCHIVE_LEDGER_NAME:
            continue
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return run, ledger


class RestorationV21Full45ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = Path(__file__).resolve().parents[2]

    def test_pass_and_no_go_exact_inventory_and_operation_counts(self) -> None:
        passed_files, _ = _complete_files(self.repository_root)
        passed = validate_full_45_evidence_files(passed_files)
        self.assertEqual(passed.outcome, PASS_OUTCOME)
        self.assertEqual((passed.completed_state_count, passed.attempted_state_count), (45, 45))
        self.assertEqual(
            (
                passed.generation_call_count,
                passed.teacher_forward_count,
                passed.kl_measurement_count,
            ),
            (90, 135, 90),
        )
        self.assertEqual(len(passed.inventory), 94)

        no_go_files, _ = _complete_files(self.repository_root, no_go=True)
        no_go = validate_full_45_evidence_files(no_go_files)
        self.assertEqual(no_go.outcome, NO_GO_OUTCOME)
        self.assertEqual(
            (
                no_go.generation_call_count,
                no_go.teacher_forward_count,
                no_go.kl_measurement_count,
            ),
            (90, 132, 88),
        )
        missing = dict(passed_files)
        del missing["attempts/044.json"]
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_full_45_evidence_files(missing)

    def test_state_attempt_schedule_runtime_suppression_and_ledger_tamper(self) -> None:
        files, _ = _complete_files(self.repository_root)
        teacher = copy.deepcopy(files)
        state = json.loads(teacher["states/000.json"])
        state["teacher_forwards"]["summary_only"][
            "teacher_standard_eos_suppression_value"
        ] = -1.0
        teacher["states/000.json"] = pretty_json_bytes(state)
        with self.assertRaisesRegex(ValueError, "suppression"):
            validate_full_45_evidence_files(teacher)

        schedule = copy.deepcopy(files)
        state = json.loads(schedule["states/000.json"])
        state["operation_counts"]["teacher_forward_count"] = 2
        schedule["states/000.json"] = pretty_json_bytes(state)
        with self.assertRaisesRegex(ValueError, "operation|teacher"):
            validate_full_45_evidence_files(schedule)

        marker = copy.deepcopy(files)
        value = json.loads(marker["attempts/000.json"])
        value["state"] = json.loads(marker["attempts/001.json"])["state"]
        marker["attempts/000.json"] = pretty_json_bytes(value)
        with self.assertRaisesRegex(ValueError, "attempt marker"):
            validate_full_45_evidence_files(marker)

        ledger = copy.deepcopy(files)
        value = json.loads(ledger[ARCHIVE_LEDGER_NAME])
        value["journal"][-1]["completed_state_count"] = 44
        ledger[ARCHIVE_LEDGER_NAME] = pretty_json_bytes(value)
        with self.assertRaisesRegex(ValueError, "journal|high-water"):
            validate_full_45_evidence_files(ledger)

    def test_stale_pass_missing_chain_cannot_validate_or_package(self) -> None:
        files, _ = _complete_files(self.repository_root)
        for relative in (
            RUNTIME_IDENTITY_FILENAME,
            RUN_MANIFEST_FILENAME,
            ARCHIVE_LEDGER_NAME,
            "states/044.json",
            "attempts/044.json",
        ):
            with self.subTest(relative=relative):
                incomplete = dict(files)
                del incomplete[relative]
                with self.assertRaises(ValueError):
                    validate_full_45_evidence_files(incomplete)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, ledger = _write_raw_tree(root, files)
            shutil.rmtree(raw / "states")
            with self.assertRaisesRegex(ValueError, "inventory"):
                package_raw_full_45_evidence(
                    repository_root=self.repository_root,
                    raw_output_dir=raw,
                    global_attempt_ledger=ledger,
                    output_archive=root / "forbidden.tar",
                    source_git_commit=SOURCE_COMMIT,
                    require_canonical_location=False,
                    git_identity_validator=lambda _: {"commit": SOURCE_COMMIT},
                    source_binding_validator=lambda *args, **kwargs: None,
                )

    def test_invalid_partial_missing_runtime_and_deleted_inventory(self) -> None:
        partial = _invalid_files(
            self.repository_root,
            completed=2,
            attempted=3,
            observed_completed=2,
            observed_attempted=3,
            stage="state_002",
            include_runtime=True,
        )
        evidence = validate_full_45_evidence_files(partial)
        self.assertEqual(evidence.outcome, INVALID_OUTCOME)
        self.assertEqual((evidence.completed_state_count, evidence.attempted_state_count), (2, 3))
        self.assertEqual(
            (
                evidence.generation_call_count,
                evidence.teacher_forward_count,
                evidence.kl_measurement_count,
            ),
            (4, 6, 4),
        )

        missing_runtime = _invalid_files(
            self.repository_root,
            completed=2,
            attempted=2,
            observed_completed=2,
            observed_attempted=2,
            stage="resume_detected_missing_runtime_identity",
            include_runtime=False,
        )
        self.assertEqual(
            validate_full_45_evidence_files(missing_runtime).completed_state_count,
            2,
        )

        deleted = _invalid_files(
            self.repository_root,
            completed=2,
            attempted=2,
            observed_completed=0,
            observed_attempted=0,
            stage="resume_detected_deleted_durable_claim_component",
            include_runtime=False,
        )
        deleted_evidence = validate_full_45_evidence_files(deleted)
        self.assertEqual(deleted_evidence.completed_state_count, 2)
        self.assertEqual(deleted_evidence.generation_call_count, 0)

        forbidden = copy.deepcopy(deleted)
        aggregate = json.loads(forbidden[AGGREGATE_FILENAME])
        aggregate["invalid_failure"]["stage"] = "state_002"
        forbidden[AGGREGATE_FILENAME] = pretty_json_bytes(aggregate)
        with self.assertRaisesRegex(ValueError, "requires persisted runtime"):
            validate_full_45_evidence_files(forbidden)

    def test_invalid_deleted_terminal_aggregate_preserves_full_high_water(self) -> None:
        invalid = _invalid_files(
            self.repository_root,
            completed=45,
            attempted=45,
            observed_completed=45,
            observed_attempted=45,
            stage="resume_detected_deleted_terminal_aggregate",
            include_runtime=True,
        )
        evidence = validate_full_45_evidence_files(invalid)
        self.assertEqual(evidence.outcome, INVALID_OUTCOME)
        self.assertEqual((evidence.completed_state_count, evidence.attempted_state_count), (45, 45))
        self.assertEqual(evidence.generation_call_count, 90)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, ledger = _write_raw_tree(root, invalid)
            packaged = package_raw_full_45_evidence(
                repository_root=self.repository_root,
                raw_output_dir=raw,
                global_attempt_ledger=ledger,
                output_archive=root / "invalid.tar",
                source_git_commit=SOURCE_COMMIT,
                require_canonical_location=False,
                git_identity_validator=lambda _: {"commit": SOURCE_COMMIT},
                source_binding_validator=lambda *args, **kwargs: None,
            )
            self.assertEqual(packaged["outcome"], INVALID_OUTCOME)

    def test_deterministic_ustar_package_and_extracted_tree(self) -> None:
        files, _ = _complete_files(self.repository_root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.tar"
            second = root / "second.tar"
            _write_deterministic_tar(first, files)
            _write_deterministic_tar(second, files)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            evidence = read_full_45_evidence_archive(first)
            self.assertEqual(evidence.outcome, PASS_OUTCOME)

            gnu = root / "gnu.tar"
            with tarfile.open(gnu, "w", format=tarfile.GNU_FORMAT) as archive:
                for relative in sorted(files):
                    payload = files[relative]
                    info = tarfile.TarInfo(f"{ARCHIVE_MEMBER_PREFIX}/{relative}")
                    info.size = len(payload)
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(ValueError, "canonical deterministic USTAR"):
                read_full_45_evidence_archive(gnu)

            trailing = root / "trailing.tar"
            trailing.write_bytes(first.read_bytes() + b"trailing-junk")
            with self.assertRaisesRegex(ValueError, "canonical deterministic USTAR"):
                read_full_45_evidence_archive(trailing)
            with tarfile.open(first, "r:") as archive:
                archive.extractall(root / "extracted", filter="data")
            extracted = read_extracted_full_45_evidence(root / "extracted")
            self.assertEqual(extracted.tree_inventory_sha256, evidence.tree_inventory_sha256)

            raw, ledger = _write_raw_tree(root / "raw", files)
            packaged = root / "packaged.tar"
            result = package_raw_full_45_evidence(
                repository_root=self.repository_root,
                raw_output_dir=raw,
                global_attempt_ledger=ledger,
                output_archive=packaged,
                source_git_commit=SOURCE_COMMIT,
                require_canonical_location=False,
                git_identity_validator=lambda _: {"commit": SOURCE_COMMIT},
                source_binding_validator=lambda *args, **kwargs: None,
            )
            self.assertEqual(result["archive_sha256"], artifact_module.sha256_file(packaged))

    def test_source_x_replay_rejects_config_parent_and_source_tamper(self) -> None:
        files, blobs = _complete_files(self.repository_root)
        run_contract = json.loads(files[RUN_MANIFEST_FILENAME])["run_contract"]

        def reader(root: Path, commit: str, relative: str) -> bytes:
            del root, commit
            return blobs[relative]

        validate_source_x_run_contract(
            run_contract,
            repository_root=self.repository_root,
            source_git_commit=SOURCE_COMMIT,
            committed_blob_reader=reader,
            ancestor_validator=lambda *_: None,
        )
        for relative in (FULL_45_CONFIG_PATH, PILOT_ARTIFACT_PATH, PROCESSOR_ARTIFACT_PATH):
            tampered = dict(blobs)
            tampered[relative] += b"\n"
            with self.assertRaisesRegex(ValueError, "source-X|blob|binding"):
                validate_source_x_run_contract(
                    run_contract,
                    repository_root=self.repository_root,
                    source_git_commit=SOURCE_COMMIT,
                    committed_blob_reader=lambda root, commit, path, values=tampered: values[path],
                    ancestor_validator=lambda *_: None,
                )

        source_record = copy.deepcopy(run_contract)
        source_record["source_inventory"][0]["sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "source-X.*blob"):
            validate_source_x_run_contract(
                source_record,
                repository_root=self.repository_root,
                source_git_commit=SOURCE_COMMIT,
                committed_blob_reader=reader,
                ancestor_validator=lambda *_: None,
            )

    def test_fresh_immutable_manifest_hash_and_clean_descendant_binding(self) -> None:
        files, _ = _complete_files(self.repository_root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.tar"
            fresh = root / "fresh-download.tar"
            _write_deterministic_tar(source, files)
            shutil.copyfile(source, fresh)
            manifest = build_full_45_artifact_manifest(
                repository_root=self.repository_root,
                fresh_immutable_archive=fresh,
                source_git_commit=SOURCE_COMMIT,
                hf_repo=CANONICAL_HF_REPO,
                hf_immutable_revision=HF_REVISION,
                hf_path=CANONICAL_HF_PATH,
                source_binding_validator=lambda *args, **kwargs: None,
            )
            evidence = read_full_45_evidence_archive(fresh)
            result = validate_full_45_artifact_manifest(
                manifest,
                evidence=evidence,
                archive_path=fresh,
            )
            self.assertIs(result["archive_hash_verified"], True)
            self.assertIs(
                manifest["raw_archive"]["fresh_immutable_download_hash_verified"],
                True,
            )
            with mock.patch.object(
                artifact_module,
                "CANONICAL_RAW_ARCHIVE_PATH",
                source.resolve(),
            ):
                with self.assertRaisesRegex(ValueError, "fresh immutable"):
                    build_full_45_artifact_manifest(
                        repository_root=self.repository_root,
                        fresh_immutable_archive=source,
                        source_git_commit=SOURCE_COMMIT,
                        hf_repo=CANONICAL_HF_REPO,
                        hf_immutable_revision=HF_REVISION,
                        hf_path=CANONICAL_HF_PATH,
                        source_binding_validator=lambda *args, **kwargs: None,
                    )
            tampered = copy.deepcopy(manifest)
            tampered["raw_archive"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash|size"):
                validate_full_45_artifact_manifest(
                    tampered,
                    evidence=evidence,
                    archive_path=fresh,
                )

            git_root = root / "repo"
            artifact_path = git_root / artifact_module.CANONICAL_GIT_ARTIFACT_PATH
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(pretty_json_bytes(manifest))
            current_commit = "c" * 40
            with mock.patch.object(
                artifact_module,
                "_git_bytes",
                return_value=artifact_path.read_bytes(),
            ), mock.patch.object(
                artifact_module,
                "require_source_commit_ancestor",
            ) as ancestor:
                validated = artifact_module.validate_committed_full_45_artifact(
                    repository_root=git_root,
                    current_git_commit=current_commit,
                    evidence_path=fresh,
                    source_binding_validator=lambda *args, **kwargs: None,
                )
            ancestor.assert_called_once()
            self.assertEqual(validated["current_git_commit"], current_commit)

    def test_cli_requires_fresh_immutable_archive(self) -> None:
        parser = _build_parser()
        help_text = parser.format_help()
        self.assertIn("create-manifest", help_text)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "create-manifest",
                    "--repository-root",
                    "/repo",
                    "--source-git-commit",
                    SOURCE_COMMIT,
                    "--hf-immutable-revision",
                    HF_REVISION,
                    "--output",
                    "/repo/artifact.json",
                ]
            )
        parsed = parser.parse_args(
            [
                "create-manifest",
                "--repository-root",
                "/repo",
                "--fresh-immutable-archive",
                "/tmp/fresh.tar",
                "--source-git-commit",
                SOURCE_COMMIT,
                "--hf-immutable-revision",
                HF_REVISION,
                "--output",
                "/repo/artifact.json",
            ]
        )
        self.assertEqual(parsed.fresh_immutable_archive, Path("/tmp/fresh.tar"))
