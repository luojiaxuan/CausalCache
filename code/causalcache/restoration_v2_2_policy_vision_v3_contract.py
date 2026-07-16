"""Fail-closed source contract for the policy-vision SizeDict repair."""

from __future__ import annotations

import copy
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.restoration_v2_2_policy_vision_contract import (
    EXPECTED_OUTPUT_FILES,
    sha256_bytes,
    sha256_file,
    strict_json_object_bytes,
)
from causalcache.restoration_v2_2_policy_vision_v2_contract import (
    CANONICAL_CONFIG_PATH as PARENT_CONFIG_PATH,
    FROZEN_CONFIG_SHA256 as PARENT_CONFIG_SHA256,
    GPU_UUID_RUNTIME_TYPE_PROFILE,
    GPU_UUID_VALUE_TYPE_PROFILE,
    GPU_UUID_VALUE_TYPE_PROFILE_MAPPING,
    PROTOCOL_ID as PARENT_PROTOCOL_ID,
    RestorationV22PolicyVisionV2Contract,
)


PROTOCOL_ID = (
    "causalcache_restoration_v2_2_policy_vision_baseline_"
    "v3_size_dict_interface_repair"
)
CANONICAL_CONFIG_PATH = (
    "code/configs/"
    "causalcache_restoration_v2_2_policy_vision_baseline_"
    "v3_size_dict_interface_repair.json"
)
FROZEN_CONFIG_SHA256 = (
    "794474d8bc60463ba10fdd772691461f5910ca5e5501f7cccff4c53542b84b7f"
)
PASS_STATUS = (
    "VALID_RESTORATION_V2_2_POLICY_VISION_BASELINE_"
    "V3_SIZE_DICT_INTERFACE_REPAIR_SOURCE_CONTRACT"
)
RUN_STATUS = (
    "COMPLETED_RESTORATION_V2_2_POLICY_VISION_BASELINE_"
    "V3_SIZE_DICT_INTERFACE_REPAIR"
)
VALID_STATUS = (
    "VALID_RESTORATION_V2_2_POLICY_VISION_BASELINE_"
    "V3_SIZE_DICT_INTERFACE_REPAIR"
)
CANONICAL_OUTPUT_DIRECTORY = (
    "data/results/"
    "restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair"
)
FORMAL_ATTEMPT_LEDGER_PATH = (
    "/data/experiments/causalcache/"
    "restoration-v2-2-policy-vision-v3-size-dict-interface-repair-attempt.json"
)
IMAGE_PROCESSOR_SIZE_VALUE_TYPE_PROFILE = (
    "transformers_image_utils_SizeDict_exact_type_v3"
)
IMAGE_PROCESSOR_SIZE_RUNTIME_TYPE_PROFILE = (
    "transformers_image_utils_size_dict_exact_edges_v3"
)
IMAGE_PROCESSOR_SIZE_VALUE_TYPE_PROFILE_MAPPING = (
    f"{IMAGE_PROCESSOR_SIZE_VALUE_TYPE_PROFILE} -> "
    f"{IMAGE_PROCESSOR_SIZE_RUNTIME_TYPE_PROFILE}"
)
COMBINED_RUNTIME_PROFILE_ID = (
    "causalcache_restoration_v2_2_policy_vision_feature_only_runtime_"
    "uuid_type_v2_size_dict_interface_v3"
)
PREREGISTRATION_STATUS = (
    "source_only_frozen_before_any_v3_size_dict_interface_repair_feature_or_score"
)
PARENT_SOURCE_GIT_COMMIT = "fe7640395d3b6aea2e5e3a8cc34a49efc5ba2d2f"
FAILURE_GIT_COMMIT = "b1d07551aace6f9242d9754cc0e2d0ce84bbdd90"
SOURCE_DIFF_BASELINE_GIT_COMMIT = FAILURE_GIT_COMMIT
FAILURE_DIRECTORY = (
    "data/results/"
    "restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt"
)
V1_CANONICAL_OUTPUT_DIRECTORY = (
    "data/results/restoration_v2_2_policy_vision_baseline_v1"
)
V2_CANONICAL_OUTPUT_DIRECTORY = (
    "data/results/"
    "restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair"
)

EXPECTED_PARENT = {
    "path": PARENT_CONFIG_PATH,
    "sha256": PARENT_CONFIG_SHA256,
    "protocol_id": PARENT_PROTOCOL_ID,
    "source_git_commit": PARENT_SOURCE_GIT_COMMIT,
}
EXPECTED_FAILURE_FILES = (
    {
        "path": "README.md",
        "size_bytes": 1624,
        "sha256": (
            "9ddcf6eb8374a76c053622d3e16b7722"
            "46f4c9aea17bc2578fe43f73762e3e90"
        ),
    },
    {
        "path": "failure.json",
        "size_bytes": 4910,
        "sha256": (
            "3fe5c7fd6ea082501627fcbe52d4ca27"
            "51dc00f6f9a2270a9fdf20987097e5b7"
        ),
    },
)
EXPECTED_FAILED_ATTEMPT = {
    "git_ref": "main",
    "git_commit": FAILURE_GIT_COMMIT,
    "directory": FAILURE_DIRECTORY,
    "protocol_id": PARENT_PROTOCOL_ID,
    "status": "INVALID_POLICY_VISION_V2_ZERO_FEATURE_SIZE_DICT_INTERFACE",
    "source_git_commit": PARENT_SOURCE_GIT_COMMIT,
    "contract_sha256": PARENT_CONFIG_SHA256,
    "failure_stage": (
        "after_gpu_uuid_snapshot_verification_and_image_processor_construction_"
        "before_policy_model_load_or_feature_forward"
    ),
    "observed_python_type": "transformers.image_utils.SizeDict",
    "observed_mapping_membership": False,
    "observed_dict_conversion_equals_expected": True,
    "policy_model_load_count": 0,
    "image_processor_batch_count": 0,
    "policy_vision_feature_forward_count": 0,
    "result_file_count": 0,
    "same_protocol_retry_allowed": False,
    "attempt_ledger_path": (
        "/data/experiments/causalcache/"
        "restoration-v2-2-policy-vision-v2-gpu-uuid-repair-attempt.json"
    ),
    "attempt_ledger_sha256": (
        "492e7c7aea0539fc5bc65c0d0be41b5659563a140d8e411a3b347b4c18584502"
    ),
    "exact_files": list(EXPECTED_FAILURE_FILES),
}
BASELINE_EXISTING_SOURCES = (
    {
        "path": "code/causalcache/policy/gui_owl_v2_2_vision_runtime.py",
        "sha256": (
            "ed5efae5b1201181f3dec1e56eb6e49d"
            "e7921355dc903478d6c3b6a8a4990a00"
        ),
    },
    {
        "path": "code/causalcache/restoration_v2_2_policy_vision.py",
        "sha256": (
            "38c90f17da63999399ea21d11e63591c"
            "016e2818feb5dd7a041ded92021f5af1"
        ),
    },
    {
        "path": "code/scripts/run_restoration_v2_2_policy_vision_baseline.py",
        "sha256": (
            "6b90b7f48e99c293fe519bb4c5315b3e"
            "680c3f54a5a99a891c85bb7280b60708"
        ),
    },
    {
        "path": "code/scripts/run_restoration_v2_2_policy_vision_baseline_v2.py",
        "sha256": (
            "18fa616c55e8c5dad89981de40bf1ccdf"
            "570b4bba6217b25894f82f854f3ba4d"
        ),
    },
)
EXISTING_SYMBOLS_ALLOWED_TO_CHANGE = {
    "code/causalcache/policy/gui_owl_v2_2_vision_runtime.py": [
        "image processor size profile constants",
        "combined runtime profile constant",
        "_runtime_profile_id",
        "_normalize_image_processor_size",
        "GUIOwlV22VisionFeatureRuntime.__init__",
        "GUIOwlV22VisionFeatureRuntime.metadata.runtime_profile_id",
        "GUIOwlV22VisionFeatureRuntime.metadata.processor_size",
    ],
    "code/causalcache/restoration_v2_2_policy_vision.py": [
        "DEFAULT_IMAGE_PROCESSOR_SIZE_PROFILE",
        "run_policy_vision_worker",
    ],
    "code/scripts/run_restoration_v2_2_policy_vision_baseline.py": [
        "PolicyVisionRunnerProtocol",
        "V1_RUN_TOMBSTONE_STATUS",
        "V1_RUNNER_PROTOCOL",
        "_run_feature_workers",
        "_validate_runner_protocol",
        "run_protocol_main",
    ],
    "code/scripts/run_restoration_v2_2_policy_vision_baseline_v2.py": [
        "V2_RUN_TOMBSTONE_STATUS",
        "V2_RUNNER_PROTOCOL.formal_run_allowed",
        "V2_RUNNER_PROTOCOL.image_processor_size_profile",
        "V2_RUNNER_PROTOCOL.run_tombstone_status",
    ],
}
NEW_PROTOCOL_PLUMBING_PATHS_ALLOWED = (
    "code/causalcache/restoration_v2_2_policy_vision_v3_contract.py",
    CANONICAL_CONFIG_PATH,
    "code/scripts/run_restoration_v2_2_policy_vision_baseline_v3.py",
    "code/scripts/validate_restoration_v2_2_policy_vision_v3_contract.py",
    "code/tests/test_restoration_v2_2_policy_vision_v3_contract.py",
    "code/tests/test_run_restoration_v2_2_policy_vision_baseline_v3.py",
    "code/tests/test_validate_restoration_v2_2_policy_vision_v3_contract.py",
)
FORMAL_SOURCE_CHANGED_PATHS_EXACT = (
    "README.md",
    "code/README.md",
    "code/causalcache/policy/gui_owl_v2_2_vision_runtime.py",
    "code/causalcache/restoration_v2_2_policy_vision.py",
    "code/causalcache/restoration_v2_2_policy_vision_v3_contract.py",
    CANONICAL_CONFIG_PATH,
    "code/scripts/run_restoration_v2_2_policy_vision_baseline.py",
    "code/scripts/run_restoration_v2_2_policy_vision_baseline_v2.py",
    "code/scripts/run_restoration_v2_2_policy_vision_baseline_v3.py",
    "code/scripts/validate_restoration_v2_2_policy_vision_v3_contract.py",
    "code/tests/test_gui_owl_v2_2_vision_runtime.py",
    "code/tests/test_restoration_v2_2_policy_vision.py",
    "code/tests/test_restoration_v2_2_policy_vision_v3_contract.py",
    "code/tests/test_run_restoration_v2_2_policy_vision_baseline.py",
    "code/tests/test_run_restoration_v2_2_policy_vision_baseline_v2.py",
    "code/tests/test_run_restoration_v2_2_policy_vision_baseline_v3.py",
    "code/tests/test_validate_restoration_v2_2_policy_vision_v3_contract.py",
    "data/README.md",
    "docs/progress.md",
    "docs/restoration_v2_2_policy_vision_baseline.md",
)
EXPECTED_SEMANTIC_REPAIR = {
    "newly_accepted_runtime_type": "transformers.image_utils.SizeDict",
    "acceptance_predicate": (
        "type(value) is loaded transformers.image_utils.SizeDict"
    ),
    "mapping_membership_required": False,
    "mapping_implementation_explicitly_rejected_for_v3": True,
    "non_edge_fields_required_none": [
        "height",
        "width",
        "max_height",
        "max_width",
    ],
    "edge_attributes_required_exact_int": [
        "shortest_edge",
        "longest_edge",
    ],
    "conversion": "normalized = dict(value)",
    "conversion_after_exact_type_and_attribute_checks": True,
    "exact_normalized_keys": ["longest_edge", "shortest_edge"],
    "exact_normalized_values": {
        "longest_edge": 2621440,
        "shortest_edge": 2621440,
    },
    "existing_v1_mapping_profile_unchanged": True,
    "existing_v2_gpu_uuid_profile_unchanged": True,
    "all_other_v3_runtime_types_rejected": True,
    "metadata_processor_size_remains_plain_json_object": True,
}
EXPECTED_IMPLEMENTATION_BOUNDARY = {
    "source_diff_baseline_git_commit": SOURCE_DIFF_BASELINE_GIT_COMMIT,
    "baseline_existing_sources": list(BASELINE_EXISTING_SOURCES),
    "existing_symbols_allowed_to_change": EXISTING_SYMBOLS_ALLOWED_TO_CHANGE,
    "new_protocol_plumbing_paths_allowed": list(
        NEW_PROTOCOL_PLUMBING_PATHS_ALLOWED
    ),
    "formal_runner_change_limited_to": [
        (
            "exact v1 tombstone status identity without changing its prior "
            "run prohibition"
        ),
        "permanent v2 run tombstone",
        "v3 protocol output and status constants",
        "reuse exact v2 GPU UUID runtime profile",
        "explicit v3 image processor SizeDict runtime profile threading",
        "exact v3 runtime metadata profile validation",
        "repair identity summary provenance",
        "durable exclusive v3 attempt claim before model or feature access",
    ],
    "formal_source_changed_paths_exact": list(
        FORMAL_SOURCE_CHANGED_PATHS_EXACT
    ),
    "source_freeze_must_match_exact_changed_paths_before_formal_run": True,
}
EXPECTED_REPAIR_SCOPE = {
    "image_processor_size_interface_only": True,
    "gpu_uuid_value_type_profile": GPU_UUID_VALUE_TYPE_PROFILE,
    "gpu_uuid_runtime_type_profile": GPU_UUID_RUNTIME_TYPE_PROFILE,
    "gpu_uuid_contract_to_runtime_profile_mapping": (
        GPU_UUID_VALUE_TYPE_PROFILE_MAPPING
    ),
    "image_processor_size_value_type_profile": (
        IMAGE_PROCESSOR_SIZE_VALUE_TYPE_PROFILE
    ),
    "image_processor_size_runtime_type_profile": (
        IMAGE_PROCESSOR_SIZE_RUNTIME_TYPE_PROFILE
    ),
    "image_processor_size_contract_to_runtime_profile_mapping": (
        IMAGE_PROCESSOR_SIZE_VALUE_TYPE_PROFILE_MAPPING
    ),
    "combined_runtime_profile_id": COMBINED_RUNTIME_PROFILE_ID,
    "pinned_torch_version": "2.11.0+cu130",
    "pinned_transformers_version": "5.6.0",
    "semantic_repair": EXPECTED_SEMANTIC_REPAIR,
    "implementation_boundary": EXPECTED_IMPLEMENTATION_BOUNDARY,
    "parent_scientific_contract_changed": False,
    "authorization_changed": False,
    "immutable_inputs_changed": False,
    "scope_or_denominator_changed": False,
    "preprocessing_geometry_or_values_changed": False,
    "runtime_stack_or_worker_schedule_changed": False,
    "gpu_uuid_semantics_changed": False,
    "verification_schedule_or_tolerance_changed": False,
    "feature_pooling_similarity_or_selection_changed": False,
    "operation_ceiling_changed": False,
    "statistics_contract_changed": False,
    "confirm_or_sealed_test_access_changed": False,
}
EXPECTED_OUTPUT = {
    "canonical_result_directory": CANONICAL_OUTPUT_DIRECTORY,
    "exact_files": list(EXPECTED_OUTPUT_FILES),
    "run_status": RUN_STATUS,
    "valid_status": VALID_STATUS,
    "invalid_v1_canonical_result_directory": V1_CANONICAL_OUTPUT_DIRECTORY,
    "invalid_v2_canonical_result_directory": V2_CANONICAL_OUTPUT_DIRECTORY,
    "invalid_v1_and_v2_output_and_staging_must_remain_absent": True,
    "failed_attempt_directory_must_remain_byte_unchanged": True,
    "raw_labels_images_features_or_model_parameters_copied_to_git": False,
    "formal_run_requires_clean_pushed_main": True,
    "run_requires_exclusive_new_output_directory": True,
    "atomic_directory_publish_required": True,
    (
        "validate_rebuilds_every_output_byte_from_recorded_feature_rows_"
        "and_immutable_labels_witness"
    ): True,
    "validate_recomputes_policy_vision_features": False,
    "formal_attempt_ledger_path": FORMAL_ATTEMPT_LEDGER_PATH,
    "formal_attempt_ledger_created_with_o_excl_before_model_or_feature_access": True,
    "retry_allowed": False,
    "resume_allowed": False,
    "formal_result_status_before_execution": "not_generated",
}
UNCHANGED_PARENT_SECTIONS = (
    "schema_version",
    "authorization",
    "immutable_inputs",
    "scope",
    "execution_contract",
    "feature_contract",
    "operation_ceiling",
    "statistics_contract",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} key schema drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _git_bytes(root: Path, revision: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _git_text(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _require_commit(root: Path, revision: str) -> None:
    subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _validate_overlay_semantics(data: Mapping[str, Any]) -> None:
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol_id",
            "preregistration_status",
            "parent_contract",
            "failed_attempt",
            "repair_scope",
            "output_contract",
        },
        "policy-vision v3 SizeDict repair contract",
    )
    if (
        data["schema_version"] != "0.2.0"
        or data["protocol_id"] != PROTOCOL_ID
        or data["preregistration_status"] != PREREGISTRATION_STATUS
    ):
        raise ValueError("policy-vision v3 SizeDict repair identity drifted")
    if dict(_mapping(data["parent_contract"], "parent contract")) != EXPECTED_PARENT:
        raise ValueError("policy-vision v3 parent contract drifted")
    if dict(_mapping(data["failed_attempt"], "failed attempt")) != (
        EXPECTED_FAILED_ATTEMPT
    ):
        raise ValueError("policy-vision v3 failed-attempt binding drifted")
    if dict(_mapping(data["repair_scope"], "repair scope")) != (
        EXPECTED_REPAIR_SCOPE
    ):
        raise ValueError("policy-vision v3 repair scope drifted")
    if dict(_mapping(data["output_contract"], "output contract")) != (
        EXPECTED_OUTPUT
    ):
        raise ValueError("policy-vision v3 output contract drifted")


def _validate_failure_artifact(root: Path) -> None:
    directory = root.joinpath(*PurePosixPath(FAILURE_DIRECTORY).parts)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("policy-vision v2 failure directory is missing or symlinked")
    children = tuple(directory.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in children):
        raise ValueError("policy-vision v2 failure directory is not regular-file only")
    if sorted(path.name for path in children) != sorted(
        record["path"] for record in EXPECTED_FAILURE_FILES
    ):
        raise ValueError("policy-vision v2 failure file inventory drifted")
    for record in EXPECTED_FAILURE_FILES:
        relative = f"{FAILURE_DIRECTORY}/{record['path']}"
        path = root.joinpath(*PurePosixPath(relative).parts)
        payload = path.read_bytes()
        if (
            len(payload) != record["size_bytes"]
            or sha256_bytes(payload) != record["sha256"]
            or _git_bytes(root, FAILURE_GIT_COMMIT, relative) != payload
        ):
            raise ValueError(
                f"policy-vision v2 failure artifact drifted: {record['path']}"
            )
    failure = strict_json_object_bytes(
        (directory / "failure.json").read_bytes(),
        label="policy-vision v2 failure evidence",
    )
    expected_scalars = {
        "status": EXPECTED_FAILED_ATTEMPT["status"],
        "protocol_id": PARENT_PROTOCOL_ID,
        "source_git_commit": PARENT_SOURCE_GIT_COMMIT,
        "contract_sha256": PARENT_CONFIG_SHA256,
        "failure_stage": EXPECTED_FAILED_ATTEMPT["failure_stage"],
    }
    if any(failure.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError("policy-vision v2 failure scalar identity drifted")
    exception = _mapping(failure.get("exception"), "v2 failure exception")
    if dict(exception) != {
        "type": "ValueError",
        "message": "policy-vision image processor size drifted",
        "origin": (
            "GUIOwlV22VisionFeatureRuntime.__init__:"
            "image_processor_size_interface_check"
        ),
    }:
        raise ValueError("policy-vision v2 failure exception drifted")
    root_cause = _mapping(failure.get("root_cause"), "v2 failure root cause")
    if (
        root_cause.get("size_python_type") != "transformers.image_utils.SizeDict"
        or root_cause.get("is_collections_abc_mapping") is not False
        or root_cause.get("dict_conversion_equals_expected") is not True
        or root_cause.get("scientific_preprocessing_value_drift_observed")
        is not False
        or root_cause.get("dict_conversion")
        != {"longest_edge": 2621440, "shortest_edge": 2621440}
    ):
        raise ValueError("policy-vision v2 SizeDict diagnosis drifted")
    counts = _mapping(
        failure.get("formal_operation_bounds_before_failure"),
        "v2 failure operation bounds",
    )
    for key in (
        "policy_model_load_count",
        "image_processor_batch_count",
        "policy_vision_feature_forward_count",
        "cosine_scalar_transfer_count",
        "canonical_state_score_count",
        "selection_count",
        "restoration_label_semantic_load_count",
        "result_file_count",
        "gate_training_example_count",
        "matched_nll_evaluation_count",
        "closed_loop_episode_count",
        "confirm_state_access_count",
        "sealed_test_state_access_count",
    ):
        if counts.get(key) != 0:
            raise ValueError("policy-vision v2 failure is not zero-feature")
    retry = _mapping(failure.get("retry_boundary"), "v2 retry boundary")
    if (
        retry.get("same_protocol_retry_allowed") is not False
        or retry.get("attempt_ledger_must_be_preserved") is not True
    ):
        raise ValueError("policy-vision v2 retry boundary drifted")


def _repair_identity(config: Mapping[str, Any], digest: str) -> dict[str, Any]:
    failed = _mapping(config["failed_attempt"], "failed attempt")
    repair = _mapping(config["repair_scope"], "repair scope")
    semantic = _mapping(repair["semantic_repair"], "semantic repair")
    implementation = _mapping(
        repair["implementation_boundary"],
        "implementation boundary",
    )
    output = _mapping(config["output_contract"], "output contract")
    return {
        "repair_config": {
            "path": CANONICAL_CONFIG_PATH,
            "sha256": digest,
            "protocol_id": PROTOCOL_ID,
        },
        "parent_contract": copy.deepcopy(dict(config["parent_contract"])),
        "failed_attempt": {
            "git_ref": failed["git_ref"],
            "git_commit": failed["git_commit"],
            "directory": failed["directory"],
            "status": failed["status"],
            "attempt_ledger_path": failed["attempt_ledger_path"],
            "attempt_ledger_sha256": failed["attempt_ledger_sha256"],
            "exact_files": copy.deepcopy(failed["exact_files"]),
        },
        "gpu_uuid_value_type_profile": repair["gpu_uuid_value_type_profile"],
        "gpu_uuid_runtime_type_profile": repair[
            "gpu_uuid_runtime_type_profile"
        ],
        "gpu_uuid_contract_to_runtime_profile_mapping": repair[
            "gpu_uuid_contract_to_runtime_profile_mapping"
        ],
        "image_processor_size_value_type_profile": repair[
            "image_processor_size_value_type_profile"
        ],
        "image_processor_size_runtime_type_profile": repair[
            "image_processor_size_runtime_type_profile"
        ],
        "image_processor_size_contract_to_runtime_profile_mapping": repair[
            "image_processor_size_contract_to_runtime_profile_mapping"
        ],
        "combined_runtime_profile_id": repair["combined_runtime_profile_id"],
        "newly_accepted_processor_size_runtime_type": semantic[
            "newly_accepted_runtime_type"
        ],
        "processor_size_acceptance_predicate": semantic[
            "acceptance_predicate"
        ],
        "processor_size_conversion": semantic["conversion"],
        "exact_processor_size": copy.deepcopy(
            semantic["exact_normalized_values"]
        ),
        "source_diff_baseline_git_commit": implementation[
            "source_diff_baseline_git_commit"
        ],
        "implementation_changed_paths_allowlist": copy.deepcopy(
            list(implementation["existing_symbols_allowed_to_change"])
            + implementation["new_protocol_plumbing_paths_allowed"]
        ),
        "formal_source_changed_paths_exact": copy.deepcopy(
            implementation["formal_source_changed_paths_exact"]
        ),
        "canonical_output_directory": output["canonical_result_directory"],
        "formal_attempt_ledger_path": output["formal_attempt_ledger_path"],
        "run_status": output["run_status"],
        "valid_status": output["valid_status"],
    }


def _runner_data(
    parent: RestorationV22PolicyVisionV2Contract,
    config: Mapping[str, Any],
    digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = copy.deepcopy(dict(parent.data))
    data["protocol_id"] = PROTOCOL_ID
    data["preregistration_status"] = PREREGISTRATION_STATUS
    data["output_contract"] = copy.deepcopy(dict(config["output_contract"]))
    identity = _repair_identity(config, digest)
    data["repair_identity"] = copy.deepcopy(identity)
    for section in UNCHANGED_PARENT_SECTIONS:
        if data[section] != parent.data[section]:
            raise ValueError(f"policy-vision v3 inherited section drifted: {section}")
    return data, identity


@dataclass(frozen=True)
class RestorationV22PolicyVisionV3Contract:
    path: Path
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    repair_identity: Mapping[str, Any]
    parent: RestorationV22PolicyVisionV2Contract

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        validate_bound_sources: bool = True,
    ) -> "RestorationV22PolicyVisionV3Contract":
        root = Path(repository_root).resolve()
        resolved = Path(path).resolve()
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file() or resolved.is_symlink():
            raise ValueError("policy-vision v3 contract must use the canonical config")
        digest = sha256_file(resolved)
        if digest != FROZEN_CONFIG_SHA256:
            raise ValueError("policy-vision v3 frozen config SHA256 drifted")
        config = strict_json_object_bytes(
            resolved.read_bytes(),
            label="policy-vision v3 SizeDict repair contract",
        )
        _validate_overlay_semantics(config)
        parent = RestorationV22PolicyVisionV2Contract.load(
            root / PARENT_CONFIG_PATH,
            repository_root=root,
            validate_bound_sources=False,
        )
        if parent.sha256 != PARENT_CONFIG_SHA256:
            raise ValueError("policy-vision v3 parent config SHA256 drifted")
        data, identity = _runner_data(parent, config, digest)
        contract = cls(
            path=resolved,
            data=data,
            sha256=digest,
            repository_root=root,
            repair_identity=identity,
            parent=parent,
        )
        if validate_bound_sources:
            contract.validate_bound_sources()
        return contract

    def validate_bound_sources(self) -> None:
        self.parent.validate_bound_sources()
        for revision in (PARENT_SOURCE_GIT_COMMIT, FAILURE_GIT_COMMIT):
            _require_commit(self.repository_root, revision)
        _require_ancestor(
            self.repository_root,
            PARENT_SOURCE_GIT_COMMIT,
            FAILURE_GIT_COMMIT,
        )
        for descendant in (
            _git_text(self.repository_root, "rev-parse", "HEAD"),
            _git_text(self.repository_root, "rev-parse", "origin/main"),
        ):
            _require_ancestor(self.repository_root, FAILURE_GIT_COMMIT, descendant)
        parent_config_bytes = _git_bytes(
            self.repository_root,
            PARENT_SOURCE_GIT_COMMIT,
            PARENT_CONFIG_PATH,
        )
        if (
            sha256_bytes(parent_config_bytes) != PARENT_CONFIG_SHA256
            or parent_config_bytes != self.parent.path.read_bytes()
        ):
            raise ValueError("policy-vision v3 parent config source bytes drifted")
        for record in BASELINE_EXISTING_SOURCES:
            payload = _git_bytes(
                self.repository_root,
                SOURCE_DIFF_BASELINE_GIT_COMMIT,
                record["path"],
            )
            if sha256_bytes(payload) != record["sha256"]:
                raise ValueError(
                    "policy-vision v3 baseline implementation source drifted: "
                    + record["path"]
                )
        _validate_failure_artifact(self.repository_root)
        for relative in (
            V1_CANONICAL_OUTPUT_DIRECTORY,
            V2_CANONICAL_OUTPUT_DIRECTORY,
        ):
            output = self.repository_root / relative
            staging = output.with_name(f".{output.name}.staging")
            if output.exists() or staging.exists():
                raise ValueError("invalid policy-vision v1/v2 output or staging exists")

    def validate_formal_source_diff(self, source_git_commit: str) -> None:
        if (
            not isinstance(source_git_commit, str)
            or len(source_git_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in source_git_commit
            )
        ):
            raise ValueError("policy-vision v3 source commit must be full lowercase SHA")
        _require_commit(self.repository_root, source_git_commit)
        _require_ancestor(
            self.repository_root,
            SOURCE_DIFF_BASELINE_GIT_COMMIT,
            source_git_commit,
        )
        observed = tuple(
            sorted(
                line
                for line in _git_text(
                    self.repository_root,
                    "diff",
                    "--name-only",
                    SOURCE_DIFF_BASELINE_GIT_COMMIT,
                    source_git_commit,
                    "--",
                ).splitlines()
                if line
            )
        )
        repair = _mapping(
            self.data["repair_identity"],
            "policy-vision v3 repair identity",
        )
        expected = tuple(sorted(repair["formal_source_changed_paths_exact"]))
        if observed != expected:
            raise ValueError(
                "policy-vision v3 formal source changed-path boundary drifted"
            )


def validate_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
    require_output_absent: bool = True,
) -> dict[str, Any]:
    contract = RestorationV22PolicyVisionV3Contract.load(
        path,
        repository_root=repository_root,
    )
    output = contract.repository_root / CANONICAL_OUTPUT_DIRECTORY
    staging = output.with_name(f".{output.name}.staging")
    if require_output_absent and (output.exists() or staging.exists()):
        raise ValueError("policy-vision v3 output or staging directory already exists")
    return {
        "status": PASS_STATUS,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": contract.sha256,
        "parent_config_sha256": contract.parent.sha256,
        "failure_git_commit": FAILURE_GIT_COMMIT,
        "source_diff_baseline_git_commit": SOURCE_DIFF_BASELINE_GIT_COMMIT,
        "gpu_uuid_runtime_type_profile": GPU_UUID_RUNTIME_TYPE_PROFILE,
        "image_processor_size_runtime_type_profile": (
            IMAGE_PROCESSOR_SIZE_RUNTIME_TYPE_PROFILE
        ),
        "combined_runtime_profile_id": COMBINED_RUNTIME_PROFILE_ID,
        "canonical_output_directory": CANONICAL_OUTPUT_DIRECTORY,
        "run_status": RUN_STATUS,
        "valid_status": VALID_STATUS,
        "scientific_contract_changed": False,
        "primary_state_count": 15,
        "gpu_count": 2,
        "policy_vision_feature_forward_count": 31,
        "confirm_locked": True,
        "sealed_test_locked": True,
        "prior_outputs_absent": all(
            not (contract.repository_root / relative).exists()
            for relative in (
                V1_CANONICAL_OUTPUT_DIRECTORY,
                V2_CANONICAL_OUTPUT_DIRECTORY,
            )
        ),
        "output_absent": not output.exists(),
        "staging_output_absent": not staging.exists(),
    }
