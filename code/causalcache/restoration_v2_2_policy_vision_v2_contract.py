"""Fail-closed source contract for the policy-vision GPU UUID type repair."""

from __future__ import annotations

import copy
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.restoration_v2_2_policy_vision_contract import (
    CANONICAL_CONFIG_PATH as PARENT_CONFIG_PATH,
    EXPECTED_OUTPUT_FILES,
    FROZEN_CONFIG_SHA256 as PARENT_CONFIG_SHA256,
    PROTOCOL_ID as PARENT_PROTOCOL_ID,
    RestorationV22PolicyVisionContract,
    sha256_bytes,
    sha256_file,
    strict_json_object_bytes,
)


PROTOCOL_ID = (
    "causalcache_restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair"
)
CANONICAL_CONFIG_PATH = (
    "code/configs/"
    "causalcache_restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair.json"
)
FROZEN_CONFIG_SHA256 = (
    "23733169ef5ba60a84f4447080ef12e1893858aa375ff50d8d4e3595699e774e"
)
PASS_STATUS = (
    "VALID_RESTORATION_V2_2_POLICY_VISION_BASELINE_V2_GPU_UUID_REPAIR_"
    "SOURCE_CONTRACT"
)
RUN_STATUS = (
    "COMPLETED_RESTORATION_V2_2_POLICY_VISION_BASELINE_V2_GPU_UUID_REPAIR"
)
VALID_STATUS = (
    "VALID_RESTORATION_V2_2_POLICY_VISION_BASELINE_V2_GPU_UUID_REPAIR"
)
CANONICAL_OUTPUT_DIRECTORY = (
    "data/results/"
    "restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair"
)
FORMAL_ATTEMPT_LEDGER_PATH = (
    "/data/experiments/causalcache/"
    "restoration-v2-2-policy-vision-v2-gpu-uuid-repair-attempt.json"
)
GPU_UUID_VALUE_TYPE_PROFILE = "torch_C_CUuuid_v2"
GPU_UUID_RUNTIME_TYPE_PROFILE = "cuda_device_property_uuid_torch_c_cuuuid_v2"
GPU_UUID_VALUE_TYPE_PROFILE_MAPPING = (
    f"{GPU_UUID_VALUE_TYPE_PROFILE} -> {GPU_UUID_RUNTIME_TYPE_PROFILE}"
)
PREREGISTRATION_STATUS = (
    "source_only_frozen_before_any_v2_gpu_uuid_repair_feature_or_score"
)
PARENT_SOURCE_GIT_COMMIT = "c0937056e94d110cd67e593288f9e0c3a3b24809"
FAILURE_GIT_COMMIT = "a809a908207768759f6a15874aeac552a7fd5e08"
FAILURE_DIRECTORY = (
    "data/results/restoration_v2_2_policy_vision_baseline_v1_attempt"
)
V1_CANONICAL_OUTPUT_DIRECTORY = (
    "data/results/restoration_v2_2_policy_vision_baseline_v1"
)
PARENT_RUNTIME_PATH = (
    "code/causalcache/policy/gui_owl_v2_2_vision_runtime.py"
)
PARENT_RUNTIME_SHA256 = (
    "59d3527a7ec066e112330f1b8c112fafa093f3ef52030f563a67e2c371feb55d"
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
        "size_bytes": 1386,
        "sha256": (
            "50dada41703c45523839c2904ec5d32f8"
            "6553a71d48b46c815dd5d4dbfb9e144"
        ),
    },
    {
        "path": "failure.json",
        "size_bytes": 4308,
        "sha256": (
            "1cea24506b14b9dcd4bdfff77c820244"
            "17f65e1416a418caeec0d1192c2b0311"
        ),
    },
)
EXPECTED_FAILED_ATTEMPT = {
    "git_ref": "main",
    "git_commit": FAILURE_GIT_COMMIT,
    "directory": FAILURE_DIRECTORY,
    "protocol_id": PARENT_PROTOCOL_ID,
    "status": "INVALID_POLICY_VISION_V1_ZERO_FEATURE_GPU_UUID_TYPE",
    "source_git_commit": PARENT_SOURCE_GIT_COMMIT,
    "contract_sha256": PARENT_CONFIG_SHA256,
    "failure_stage": (
        "runtime_gpu_identity_before_model_snapshot_hash_processor_or_model_load"
    ),
    "observed_python_type": "torch._C._CUuuid",
    "policy_vision_feature_forward_count": 0,
    "result_file_count": 0,
    "same_protocol_retry_allowed": False,
    "exact_files": list(EXPECTED_FAILURE_FILES),
}
EXPECTED_REPAIR_SCOPE = {
    "gpu_uuid_value_type_only": True,
    "value_type_profile": GPU_UUID_VALUE_TYPE_PROFILE,
    "runtime_type_profile": GPU_UUID_RUNTIME_TYPE_PROFILE,
    "contract_to_runtime_profile_mapping": GPU_UUID_VALUE_TYPE_PROFILE_MAPPING,
    "pinned_torch_version": "2.11.0+cu130",
    "semantic_repair": {
        "newly_accepted_runtime_type": "torch._C._CUuuid",
        "acceptance_predicate": "type(value) is loaded torch._C._CUuuid",
        "conversion": "value = str(value)",
        "conversion_before_existing_parser": True,
        "existing_str_and_bytes_paths_unchanged": True,
        "all_other_runtime_types_rejected": True,
        "post_conversion_checks_reused_exactly": [
            "canonical_gpu_uuid_format_and_lowercase_GPU_prefix",
            "observed_uuid_equals_expected_gpu_uuid",
            "nvidia_smi_uuid_matches_exactly_one_row",
            "nvidia_smi_pci_bus_id_matches_frozen_format",
            "nvidia_smi_index_is_nonnegative_integer",
        ],
    },
    "implementation_boundary": {
        "parent_existing_sources": [
            {"path": PARENT_RUNTIME_PATH, "sha256": PARENT_RUNTIME_SHA256},
            {
                "path": "code/causalcache/restoration_v2_2_policy_vision.py",
                "sha256": (
                    "d3208c0c8e694076b4a54a4c80abf12"
                    "ef1fbb5d8eba1384532200024d8d79ae0"
                ),
            },
            {
                "path": (
                    "code/scripts/"
                    "run_restoration_v2_2_policy_vision_baseline.py"
                ),
                "sha256": (
                    "a5cf7b198702b4ad56ef1aedadbbaf3c"
                    "9053996ea63f0eb8d3260d8b91980b73"
                ),
            },
        ],
        "existing_symbols_allowed_to_change": {
            PARENT_RUNTIME_PATH: [
                "GPU_UUID type profile constants",
                "_runtime_profile_id",
                "_canonical_gpu_uuid",
                "_validated_gpu_identity",
                "GUIOwlV22VisionFeatureRuntime.__init__",
                "GUIOwlV22VisionFeatureRuntime.metadata.runtime_profile_id",
            ],
            "code/causalcache/restoration_v2_2_policy_vision.py": [
                "DEFAULT_GPU_UUID_TYPE_PROFILE",
                "run_policy_vision_worker",
            ],
            (
                "code/scripts/"
                "run_restoration_v2_2_policy_vision_baseline.py"
            ): [
                "PolicyVisionRunnerProtocol",
                "V1_RUNNER_PROTOCOL",
                "_claim_formal_attempt",
                "_validate_run_source_snapshot",
                "_validate_source_unchanged",
                "_run_feature_workers",
                "_expected_runtime_metadata",
                "_validate_runtime_metadata",
                "_validate_worker_outputs",
                "_execution_record",
                "_validate_recorded_execution",
                "_validate_formal_attempt_claim",
                "_readme",
                "_source_execution",
                "_expected_files_from_features",
                "_write_new_result",
                "_validate_existing_result",
                "_validate_mode",
                "_parser",
                "_validate_runner_protocol",
                "run_protocol_main",
                "main",
            ],
        },
        "new_protocol_plumbing_paths_allowed": [
            CANONICAL_CONFIG_PATH,
            "code/causalcache/restoration_v2_2_policy_vision_v2_contract.py",
            (
                "code/scripts/"
                "validate_restoration_v2_2_policy_vision_v2_contract.py"
            ),
            (
                "code/scripts/"
                "run_restoration_v2_2_policy_vision_baseline_v2.py"
            ),
            "code/tests/test_restoration_v2_2_policy_vision_v2_contract.py",
            (
                "code/tests/"
                "test_run_restoration_v2_2_policy_vision_baseline_v2.py"
            ),
        ],
        "formal_runner_change_limited_to": [
            "permanent v1 run tombstone",
            "v2 protocol output and status constants",
            "explicit v2 GPU UUID runtime profile threading",
            "exact v2 runtime metadata profile validation",
            "repair identity summary provenance",
            "durable exclusive v2 attempt claim before feature or model access",
        ],
        "formal_source_changed_paths_exact": [
            "README.md",
            "code/README.md",
            "code/causalcache/policy/gui_owl_v2_2_vision_runtime.py",
            "code/causalcache/restoration_v2_2_policy_vision.py",
            "code/causalcache/restoration_v2_2_policy_vision_v2_contract.py",
            (
                "code/configs/"
                "causalcache_restoration_v2_2_policy_vision_baseline_"
                "v2_gpu_uuid_repair.json"
            ),
            "code/scripts/run_restoration_v2_2_policy_vision_baseline.py",
            "code/scripts/run_restoration_v2_2_policy_vision_baseline_v2.py",
            "code/scripts/validate_restoration_v2_2_policy_vision_v2_contract.py",
            "code/tests/test_gui_owl_v2_2_vision_runtime.py",
            "code/tests/test_restoration_v2_2_policy_vision.py",
            "code/tests/test_restoration_v2_2_policy_vision_v2_contract.py",
            "code/tests/test_run_restoration_v2_2_policy_vision_baseline.py",
            (
                "code/tests/"
                "test_run_restoration_v2_2_policy_vision_baseline_v2.py"
            ),
            "data/README.md",
            "docs/progress.md",
            "docs/restoration_v2_2_policy_vision_baseline.md",
        ],
        "source_freeze_must_match_exact_changed_paths_before_formal_run": True,
    },
    "parent_scientific_contract_changed": False,
    "authorization_changed": False,
    "immutable_inputs_changed": False,
    "scope_or_denominator_changed": False,
    "preprocessing_changed": False,
    "runtime_stack_or_worker_schedule_changed": False,
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
    "invalid_v1_output_and_staging_must_remain_absent": True,
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
        "policy-vision v2 GPU UUID repair contract",
    )
    if (
        data["schema_version"] != "0.2.0"
        or data["protocol_id"] != PROTOCOL_ID
        or data["preregistration_status"] != PREREGISTRATION_STATUS
    ):
        raise ValueError("policy-vision v2 GPU UUID repair identity drifted")
    if dict(_mapping(data["parent_contract"], "parent contract")) != EXPECTED_PARENT:
        raise ValueError("policy-vision v2 parent contract drifted")
    if dict(_mapping(data["failed_attempt"], "failed attempt")) != (
        EXPECTED_FAILED_ATTEMPT
    ):
        raise ValueError("policy-vision v2 failed-attempt binding drifted")
    if dict(_mapping(data["repair_scope"], "repair scope")) != (
        EXPECTED_REPAIR_SCOPE
    ):
        raise ValueError("policy-vision v2 repair scope drifted")
    if dict(_mapping(data["output_contract"], "output contract")) != (
        EXPECTED_OUTPUT
    ):
        raise ValueError("policy-vision v2 output contract drifted")


def _validate_failure_artifact(root: Path) -> None:
    directory = root.joinpath(*PurePosixPath(FAILURE_DIRECTORY).parts)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("policy-vision v1 failure directory is missing or symlinked")
    children = tuple(directory.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in children):
        raise ValueError("policy-vision v1 failure directory is not regular-file only")
    if sorted(path.name for path in children) != sorted(
        record["path"] for record in EXPECTED_FAILURE_FILES
    ):
        raise ValueError("policy-vision v1 failure file inventory drifted")
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
                f"policy-vision v1 failure artifact drifted: {record['path']}"
            )
    failure = strict_json_object_bytes(
        (directory / "failure.json").read_bytes(),
        label="policy-vision v1 failure evidence",
    )
    expected_scalars = {
        "status": EXPECTED_FAILED_ATTEMPT["status"],
        "protocol_id": PARENT_PROTOCOL_ID,
        "source_git_commit": PARENT_SOURCE_GIT_COMMIT,
        "contract_sha256": PARENT_CONFIG_SHA256,
        "failure_stage": EXPECTED_FAILED_ATTEMPT["failure_stage"],
    }
    if any(failure.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError("policy-vision v1 failure scalar identity drifted")
    exception = _mapping(failure.get("exception"), "v1 failure exception")
    root_cause = _mapping(failure.get("root_cause"), "v1 failure root cause")
    if dict(exception) != {
        "type": "ValueError",
        "message": "GPU UUID must be a non-empty string",
        "origin": "GUIOwlV22VisionFeatureRuntime._validated_gpu_identity",
    } or root_cause.get("observed_python_type") != "torch._C._CUuuid":
        raise ValueError("policy-vision v1 GPU UUID diagnosis drifted")
    counts = _mapping(
        failure.get("operation_counts_before_failure"),
        "v1 failure operation counts",
    )
    for key in (
        "model_snapshot_full_hash_count",
        "image_processor_load_count",
        "policy_model_load_count",
        "image_processor_batch_count",
        "vision_feature_forward_count",
        "cosine_scalar_transfer_count",
        "canonical_state_score_count",
        "selection_count",
        "restoration_label_semantic_load_count",
        "gate_training_example_count",
        "matched_nll_evaluation_count",
        "closed_loop_episode_count",
        "confirm_state_access_count",
        "sealed_test_state_access_count",
        "result_file_count",
    ):
        if counts.get(key) != 0:
            raise ValueError("policy-vision v1 failure is not zero-feature")
    retry = _mapping(failure.get("retry_boundary"), "v1 retry boundary")
    if retry.get("same_protocol_retry_allowed") is not False:
        raise ValueError("policy-vision v1 same-protocol retry became allowed")


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
            "exact_files": copy.deepcopy(failed["exact_files"]),
        },
        "gpu_uuid_value_type_profile": repair["value_type_profile"],
        "gpu_uuid_runtime_type_profile": repair["runtime_type_profile"],
        "contract_to_runtime_profile_mapping": repair[
            "contract_to_runtime_profile_mapping"
        ],
        "newly_accepted_runtime_type": semantic["newly_accepted_runtime_type"],
        "conversion": semantic["conversion"],
        "post_conversion_checks_reused_exactly": copy.deepcopy(
            semantic["post_conversion_checks_reused_exactly"]
        ),
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
    parent: RestorationV22PolicyVisionContract,
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
            raise ValueError(f"policy-vision v2 inherited section drifted: {section}")
    return data, identity


@dataclass(frozen=True)
class RestorationV22PolicyVisionV2Contract:
    path: Path
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    repair_identity: Mapping[str, Any]
    parent: RestorationV22PolicyVisionContract

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        validate_bound_sources: bool = True,
    ) -> "RestorationV22PolicyVisionV2Contract":
        root = Path(repository_root).resolve()
        resolved = Path(path).resolve()
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file() or resolved.is_symlink():
            raise ValueError("policy-vision v2 contract must use the canonical config")
        digest = sha256_file(resolved)
        if digest != FROZEN_CONFIG_SHA256:
            raise ValueError("policy-vision v2 frozen config SHA256 drifted")
        config = strict_json_object_bytes(
            resolved.read_bytes(),
            label="policy-vision v2 GPU UUID repair contract",
        )
        _validate_overlay_semantics(config)
        parent = RestorationV22PolicyVisionContract.load(
            root / PARENT_CONFIG_PATH,
            repository_root=root,
            validate_bound_sources=False,
        )
        if parent.sha256 != PARENT_CONFIG_SHA256:
            raise ValueError("policy-vision v2 parent config SHA256 drifted")
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
            raise ValueError("policy-vision v2 parent config source bytes drifted")
        parent_sources = EXPECTED_REPAIR_SCOPE["implementation_boundary"][
            "parent_existing_sources"
        ]
        for record in parent_sources:
            payload = _git_bytes(
                self.repository_root,
                PARENT_SOURCE_GIT_COMMIT,
                record["path"],
            )
            if sha256_bytes(payload) != record["sha256"]:
                raise ValueError(
                    "policy-vision v2 parent implementation source drifted: "
                    + record["path"]
                )
        _validate_failure_artifact(self.repository_root)
        v1_output = self.repository_root / V1_CANONICAL_OUTPUT_DIRECTORY
        v1_staging = v1_output.with_name(f".{v1_output.name}.staging")
        if v1_output.exists() or v1_staging.exists():
            raise ValueError("invalid policy-vision v1 output or staging exists")

    def validate_formal_source_diff(self, source_git_commit: str) -> None:
        if (
            not isinstance(source_git_commit, str)
            or len(source_git_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_git_commit)
        ):
            raise ValueError("policy-vision v2 source commit must be full lowercase SHA")
        _require_commit(self.repository_root, source_git_commit)
        _require_ancestor(
            self.repository_root,
            FAILURE_GIT_COMMIT,
            source_git_commit,
        )
        observed = tuple(
            sorted(
                line
                for line in _git_text(
                    self.repository_root,
                    "diff",
                    "--name-only",
                    FAILURE_GIT_COMMIT,
                    source_git_commit,
                    "--",
                ).splitlines()
                if line
            )
        )
        repair = _mapping(
            self.data["repair_identity"],
            "policy-vision v2 repair identity",
        )
        expected = tuple(sorted(repair["formal_source_changed_paths_exact"]))
        if observed != expected:
            raise ValueError(
                "policy-vision v2 formal source changed-path boundary drifted"
            )


def validate_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
    require_output_absent: bool = True,
) -> dict[str, Any]:
    contract = RestorationV22PolicyVisionV2Contract.load(
        path,
        repository_root=repository_root,
    )
    output = contract.repository_root / CANONICAL_OUTPUT_DIRECTORY
    staging = output.with_name(f".{output.name}.staging")
    if require_output_absent and (output.exists() or staging.exists()):
        raise ValueError("policy-vision v2 output or staging directory already exists")
    return {
        "status": PASS_STATUS,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": contract.sha256,
        "parent_config_sha256": contract.parent.sha256,
        "failure_git_commit": FAILURE_GIT_COMMIT,
        "gpu_uuid_value_type_profile": GPU_UUID_VALUE_TYPE_PROFILE,
        "canonical_output_directory": CANONICAL_OUTPUT_DIRECTORY,
        "run_status": RUN_STATUS,
        "valid_status": VALID_STATUS,
        "scientific_contract_changed": False,
        "primary_state_count": 15,
        "gpu_count": 2,
        "policy_vision_feature_forward_count": 31,
        "confirm_locked": True,
        "sealed_test_locked": True,
        "output_absent": not output.exists(),
        "staging_output_absent": not staging.exists(),
    }
