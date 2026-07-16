"""Fail-closed source contract for the v2.2 policy-vision comparator."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL_ID = "causalcache_restoration_v2_2_policy_vision_baseline_v1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json"
)
FROZEN_CONFIG_SHA256 = (
    "a2319f8ea52d53fa01487cdbcbfef20b86ac81dcfab1c8a36ce4583b0b503023"
)
PASS_STATUS = "VALID_RESTORATION_V2_2_POLICY_VISION_BASELINE_SOURCE_CONTRACT"
EXPECTED_OUTPUT_FILES = ("README.md", "state_scores.jsonl", "summary.json")
EXPECTED_PRIMARY_STATE_INDICES = tuple(range(2, 45, 3))
OCR_RGB_EXECUTION_SOURCE_COMMIT = "a9bede85ab8bd10623c5755b944b3c26865c6485"
OCR_RGB_ARTIFACT_COMMIT = "7e59591573cb31f178dfd07422cc2e3c8aeff573"
OCR_RGB_VALIDATION_COMMIT = "09ff71c5e24ed4a4fee7bda14ed888c5ddf1db2a"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")

EXPECTED_OPERATION_CEILING = {
    "validated_label_state_count": 45,
    "validated_label_distance_row_count": 420,
    "primary_label_state_count": 15,
    "primary_label_distance_row_count": 240,
    "geometry_primary_record_count": 15,
    "identity_witness_state_count": 15,
    "identity_witness_unique_image_count": 75,
    "policy_model_load_count": 2,
    "image_processor_load_count": 2,
    "canonical_image_processor_batch_count": 15,
    "verification_image_processor_batch_count": 1,
    "image_processor_batch_count": 16,
    "canonical_image_processing_assignment_count": 75,
    "verification_image_processing_assignment_count": 5,
    "image_processing_assignment_count": 80,
    "canonical_image_payload_extract_count": 75,
    "verification_image_payload_extract_count": 5,
    "image_payload_extract_count": 80,
    "canonical_image_decode_count": 75,
    "verification_image_decode_count": 5,
    "image_decode_count": 80,
    "canonical_vision_feature_forward_count": 15,
    "same_device_replay_vision_feature_forward_count": 15,
    "cross_device_sentinel_vision_feature_forward_count": 1,
    "policy_vision_feature_forward_count": 31,
    "canonical_cosine_scalar_transfer_count": 60,
    "verification_cosine_scalar_transfer_count": 64,
    "cosine_scalar_transfer_count": 124,
    "canonical_state_score_count": 15,
    "restoration_label_post_selection_evaluation_count": 15,
    "policy_forward_count": 0,
    "language_model_forward_count": 0,
    "lm_head_forward_count": 0,
    "generation_count": 0,
    "teacher_forward_count": 0,
    "kl_measurement_count": 0,
    "gate_training_example_count": 0,
    "gate_model_forward_count": 0,
    "matched_nll_evaluation_count": 0,
    "closed_loop_episode_count": 0,
    "confirm_state_access_count": 0,
    "sealed_test_state_access_count": 0,
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def strict_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=pairs_hook)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} key schema drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or str(parsed) != value
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _require_git_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or GIT_SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git SHA")
    return value


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


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


def _validate_file_record(
    root: Path,
    raw_record: Any,
    *,
    prefix: str = "",
    revision: str | None = None,
) -> None:
    record = _mapping(raw_record, "file record")
    _exact_keys(record, {"path", "size_bytes", "sha256"}, "file record")
    relative = _safe_relative_path(record["path"], "file path")
    size = record["size_bytes"]
    digest = _require_sha256(record["sha256"], "file SHA256")
    if type(size) is not int or size < 0:
        raise ValueError("file size must be a non-negative integer")
    repository_path = f"{prefix}/{relative}" if prefix else relative
    path = root.joinpath(*PurePosixPath(repository_path).parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"bound file is missing or symlinked: {repository_path}")
    payload = path.read_bytes()
    if len(payload) != size or sha256_bytes(payload) != digest:
        raise ValueError(f"bound file identity drifted: {repository_path}")
    if revision is not None and _git_bytes(root, revision, repository_path) != payload:
        raise ValueError(f"bound file differs from frozen commit: {repository_path}")


def _validate_source_record(root: Path, raw_record: Any, label: str) -> None:
    record = _mapping(raw_record, label)
    _exact_keys(record, {"path", "sha256"}, label)
    relative = _safe_relative_path(record["path"], f"{label} path")
    digest = _require_sha256(record["sha256"], f"{label} SHA256")
    path = root.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
        raise ValueError(f"{label} identity drifted")


def _validate_semantics(data: Mapping[str, Any]) -> None:
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol_id",
            "preregistration_status",
            "authorization",
            "immutable_inputs",
            "scope",
            "execution_contract",
            "feature_contract",
            "statistics_contract",
            "operation_ceiling",
            "output_contract",
        },
        "policy-vision contract",
    )
    if data["schema_version"] != "0.1.0" or data["protocol_id"] != PROTOCOL_ID:
        raise ValueError("policy-vision protocol identity drifted")
    if data["preregistration_status"] != (
        "source_only_frozen_before_any_formal_policy_vision_feature_or_score"
    ):
        raise ValueError("policy-vision preregistration status drifted")

    authorization = _mapping(data["authorization"], "authorization")
    allowed_true = {
        "gpu_vision_feature_extraction_allowed",
        "direct_image_processor_allowed",
        "restoration_label_evaluation_after_selection_allowed",
    }
    if set(authorization) != allowed_true | {
        "goal_or_text_to_feature_worker_allowed",
        "ocr_to_feature_worker_allowed",
        "restoration_labels_to_feature_worker_allowed",
        "language_model_forward_allowed",
        "lm_head_allowed",
        "generation_allowed",
        "restoration_teacher_forward_allowed",
        "new_kl_measurement_allowed",
        "gate_training_allowed",
        "matched_nll_allowed",
        "closed_loop_allowed",
        "confirm_access_allowed",
        "sealed_test_access_allowed",
    }:
        raise ValueError("policy-vision authorization key schema drifted")
    if any(authorization[key] is not True for key in allowed_true) or any(
        authorization[key] is not False
        for key in authorization
        if key not in allowed_true
    ):
        raise ValueError("policy-vision authorization boundary drifted")

    inputs = _mapping(data["immutable_inputs"], "immutable inputs")
    _exact_keys(
        inputs,
        {
            "selector_geometry_result",
            "restoration_labels",
            "derived_dataset",
            "ocr_rgb_identity_witness",
            "baseline_implementation",
            "model_snapshot",
        },
        "immutable inputs",
    )
    scope = _mapping(data["scope"], "scope")
    expected_scope = {
        "roles_in_order": ["v2_label_train", "v2_development"],
        "role_state_counts": {"v2_label_train": 10, "v2_development": 5},
        "primary_candidate_event_count": 4,
        "primary_budget_event_capacity": 2,
        "primary_decision_step_id": 6,
        "candidate_event_step_ids": [1, 2, 3, 4],
        "candidate_image_role": "event_observation_after_post_action_state",
        "current_equivalent_event_step_id": 5,
        "query_image_role": "decision_step_6_current_observation",
        "state_count": 15,
        "primary_state_indices": list(EXPECTED_PRIMARY_STATE_INDICES),
        "candidate_comparison_count": 60,
        "unique_image_count": 75,
        "confirm_or_test_in_scope": False,
    }
    if dict(scope) != expected_scope:
        raise ValueError("policy-vision primary scope drifted")

    feature = _mapping(data["feature_contract"], "feature contract")
    required_feature_values = {
        "model_class": "Qwen3VLForConditionalGeneration",
        "feature_call": (
            "model.get_image_features(pixel_values=pixel_values,"
            "image_grid_thw=image_grid_thw,return_dict=True)"
        ),
        "feature_field": "pooler_output",
        "feature_semantics": "final_main_vision_merger_output",
        "pre_merger_last_hidden_state_excluded": True,
        "all_deepstack_features_excluded": True,
        "deepstack_visual_indexes": [8, 16, 24],
        "feature_dtype": "torch.bfloat16",
        "vision_output_size_after_merger": 4096,
        "merged_token_formula": "t*h*w/(spatial_merge_size^2)",
        "spatial_merge_size": 2,
        "reduction": "per_image_FP32_mean_then_L2_normalize",
        "l2_norm_epsilon": 1e-12,
        "similarity": (
            "accelerator_cosine_between_normalized_event_and_current_embeddings"
        ),
        "selection": "top_2_by_cosine_similarity",
        "score_tie_rel_tol": 1e-9,
        "score_tie_abs_tol": 1e-12,
        "score_tie_secondary_key": "lower_event_step_id",
        "full_visual_features_move_to_cpu": False,
        "cpu_transfer": (
            "grid_metadata_norm_validation_scalars_and_exactly_frozen_final_"
            "cosine_scalars_only"
        ),
        "language_model_called": False,
        "lm_head_called": False,
        "generate_called": False,
    }
    if dict(feature) != required_feature_values:
        raise ValueError("policy-vision feature contract drifted")
    if dict(_mapping(data["operation_ceiling"], "operation ceiling")) != (
        EXPECTED_OPERATION_CEILING
    ):
        raise ValueError("policy-vision operation ceiling drifted")
    _validate_execution(data["execution_contract"])
    _validate_statistics(data["statistics_contract"])

    output = _mapping(data["output_contract"], "output contract")
    expected_output = {
        "canonical_result_directory": (
            "data/results/restoration_v2_2_policy_vision_baseline_v1"
        ),
        "exact_files": list(EXPECTED_OUTPUT_FILES),
        "formal_run_requires_clean_pushed_main": True,
        "run_requires_exclusive_new_output_directory": True,
        "atomic_directory_publish_required": True,
        "retry_allowed": False,
        "resume_allowed": False,
        "raw_labels_images_features_or_model_parameters_copied_to_git": False,
        "formal_result_status_before_execution": "not_generated",
    }
    if dict(output) != expected_output:
        raise ValueError("policy-vision output contract drifted")


def _validate_execution(raw_execution: Any) -> None:
    execution = _mapping(raw_execution, "execution contract")
    _exact_keys(execution, {"preprocessing", "runtime", "verification"}, "execution")
    preprocessing = _mapping(execution["preprocessing"], "preprocessing")
    expected_preprocessing = {
        "loader": "transformers.AutoImageProcessor.from_pretrained",
        "processor_interface": (
            "image_processor(images=five_decoded_RGB_PIL_images,return_tensors=pt)"
        ),
        "auto_processor_or_tokenizer_allowed": False,
        "inputs_to_feature_worker": [
            "five_image_payloads_in_event_step_order_1_2_3_4_5",
            "image_sha256",
            "state_index",
            "worker_assignment",
        ],
        "forbidden_inputs_to_feature_worker": [
            "goal",
            "text",
            "ocr",
            "restoration_labels",
            "geometry_utilities",
            "selected_coalitions",
        ],
        "output_keys_exact": ["pixel_values", "image_grid_thw"],
        "grid_thw_histogram_over_75_canonical_images": {
            "1,136,76": 15,
            "1,148,68": 15,
            "1,150,66": 5,
            "1,152,68": 20,
            "1,80,128": 20,
        },
        "state_pixel_value_row_histogram_over_15_canonical_batches": {
            "49500": 1,
            "50320": 3,
            "51200": 4,
            "51680": 7,
        },
        "canonical_raw_patch_rows": 767020,
        "canonical_merged_visual_tokens": 191755,
        "input_geometry_observed_on": (
            "Hyper00_pinned_stack_processor_only_no_model_forward"
        ),
    }
    if dict(preprocessing) != expected_preprocessing:
        raise ValueError("policy-vision preprocessing identity drifted")

    runtime = _mapping(execution["runtime"], "runtime")
    expected_workers = [
        {
            "worker_id": "even",
            "logical_device": "cuda:0",
            "state_indices": list(range(2, 45, 6)),
        },
        {
            "worker_id": "odd",
            "logical_device": "cuda:1",
            "state_indices": list(range(5, 42, 6)),
        },
    ]
    expected_runtime = {
        "required_host_class": "Hyper_H200",
        "gpu_count": 2,
        "dtype": "bfloat16",
        "attention_implementation_requested": "eager",
        "attention_implementation_observed_must_equal": "eager",
        "eager_control_flags": {
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "seed": 0,
        },
        "strict_cuda_determinism_claimed": False,
        "expected_execution_stack": {
            "container_image_digest": (
                "sha256:6a8f60af7ca868dc266c118249d12fc7"
                "3ba85e2e8075e5e31473bd25d349acfa"
            ),
            "python_version": "3.12.3",
            "torch_version": "2.11.0+cu130",
            "torch_cuda_version": "13.0",
            "cudnn_version": 91900,
            "transformers_version": "5.6.0",
            "pillow_version": "12.2.0",
            "nvidia_driver_version": "570.172.08",
            "gpu_name": "NVIDIA H200",
        },
        "workers": expected_workers,
        "worker_identity_fields_required": [
            "device",
            "gpu_name",
            "gpu_uuid",
            "gpu_pci_bus_id",
            "logical_device_index",
            "nvidia_smi_index",
        ],
    }
    if dict(runtime) != expected_runtime:
        raise ValueError("policy-vision runtime contract drifted")

    verification = _mapping(execution["verification"], "verification")
    expected_verification = {
        "canonical_pass": {
            "state_count": 15,
            "processor_batch_count": 15,
            "image_processing_assignment_count": 75,
            "vision_feature_forward_count": 15,
            "cosine_scalar_transfer_count": 60,
        },
        "same_device_replay": {
            "state_count": 15,
            "reuse_canonical_processed_tensors": True,
            "processor_batch_count": 0,
            "image_processing_assignment_count": 0,
            "vision_feature_forward_count": 15,
            "cosine_scalar_transfer_count": 60,
        },
        "cross_device_sentinel": {
            "primary_ordinal": 0,
            "state_index": 2,
            "state_id": "0131649930078879:decision_step:006",
            "source_worker": "even",
            "verification_worker": "odd",
            "processor_batch_count": 1,
            "image_processing_assignment_count": 5,
            "vision_feature_forward_count": 1,
            "cosine_scalar_transfer_count": 4,
        },
        "score_absolute_tolerance": 1e-6,
        "candidate_ranking_must_match_exactly": True,
        "selected_coalition_must_match_exactly": True,
        "strict_bitwise_cuda_determinism_claimed": False,
    }
    if dict(verification) != expected_verification:
        raise ValueError("policy-vision verification schedule drifted")


def _validate_statistics(raw_statistics: Any) -> None:
    statistics = _mapping(raw_statistics, "statistics contract")
    expected = {
        "selection_must_complete_before_labels_are_loaded": True,
        "distance_source": "immutable_restoration_labels_only",
        "utility": "D_empty_minus_D_selected",
        "normalized_recovery": (
            "(D_empty_minus_D_selected)/D_empty_without_clamping"
        ),
        "statistics_inherit_from": (
            "causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair"
        ),
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 271828,
        "bootstrap_confidence": 0.9,
        "resampling_unit": "trajectory_id",
        "overall_resampling": "stratified_within_role",
    }
    if dict(statistics) != expected:
        raise ValueError("policy-vision statistics contract drifted")


@dataclass(frozen=True)
class RestorationV22PolicyVisionContract:
    path: Path
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        validate_bound_sources: bool = True,
    ) -> "RestorationV22PolicyVisionContract":
        root = Path(repository_root).resolve()
        resolved = Path(path).resolve()
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file() or resolved.is_symlink():
            raise ValueError("policy-vision contract must use the canonical config")
        digest = sha256_file(resolved)
        if digest != FROZEN_CONFIG_SHA256:
            raise ValueError("policy-vision frozen config SHA256 drifted")
        data = strict_json_object_bytes(
            resolved.read_bytes(),
            label="policy-vision contract",
        )
        _validate_semantics(data)
        contract = cls(resolved, data, digest, root)
        if validate_bound_sources:
            contract.validate_bound_sources()
        return contract

    def validate_bound_sources(self) -> None:
        inputs = _mapping(self.data["immutable_inputs"], "immutable inputs")
        self._validate_geometry(inputs["selector_geometry_result"])
        self._validate_labels(inputs["restoration_labels"])
        self._validate_derived(inputs["derived_dataset"])
        self._validate_identity_witness(inputs["ocr_rgb_identity_witness"])
        self._validate_baseline(inputs["baseline_implementation"])
        self._validate_model_snapshot(inputs["model_snapshot"])

    def _read_bound_json(
        self,
        path_value: Any,
        digest_value: Any,
        label: str,
    ) -> dict[str, Any]:
        relative = _safe_relative_path(path_value, f"{label} path")
        digest = _require_sha256(digest_value, f"{label} SHA256")
        path = self.repository_root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"{label} identity drifted")
        return strict_json_object_bytes(path.read_bytes(), label=label)

    def _validate_geometry(self, raw_geometry: Any) -> None:
        geometry = _mapping(raw_geometry, "selector geometry result")
        _exact_keys(
            geometry,
            {
                "git_commit",
                "execution_source_git_commit",
                "artifact_git_commit",
                "validation_git_commit",
                "directory",
                "protocol_id",
                "scientific_payload_sha256",
                "repair_contract_sha256",
                "parent_contract_sha256",
                "files",
            },
            "selector geometry result",
        )
        commits = tuple(
            _require_git_sha(geometry[key], f"geometry {key}")
            for key in (
                "execution_source_git_commit",
                "artifact_git_commit",
                "validation_git_commit",
            )
        )
        if geometry["git_commit"] != commits[1]:
            raise ValueError("geometry artifact Git identity drifted")
        for revision in commits:
            _require_commit(self.repository_root, revision)
        _require_ancestor(self.repository_root, commits[0], commits[1])
        _require_ancestor(self.repository_root, commits[1], commits[2])
        if geometry["protocol_id"] != (
            "causalcache_restoration_v2_2_selector_geometry_v2_repair"
        ) or geometry["scientific_payload_sha256"] != (
            "cc505443a7efdc68c8eeca754f24c9143cabf72a090f024fde05011e783cbb21"
        ):
            raise ValueError("geometry scientific identity drifted")
        directory = _safe_relative_path(geometry["directory"], "geometry directory")
        files = _sequence(geometry["files"], "geometry files")
        expected_files = (
            "README.md",
            "state_budget_records.jsonl",
            "summary.json",
        )
        if tuple(record.get("path") for record in files) != expected_files:
            raise ValueError("geometry file inventory drifted")
        for record in files:
            _validate_file_record(
                self.repository_root,
                record,
                prefix=directory,
                revision=commits[1],
            )
            path = f"{directory}/{record['path']}"
            if _git_bytes(self.repository_root, commits[2], path) != _git_bytes(
                self.repository_root, commits[1], path
            ):
                raise ValueError("geometry validation commit changed artifact bytes")
        repair_path = (
            "code/configs/"
            "causalcache_restoration_v2_2_selector_geometry_v2_repair.json"
        )
        parent_path = (
            "code/configs/causalcache_restoration_v2_2_selector_geometry.json"
        )
        frozen_contracts = {
            repair_path: geometry["repair_contract_sha256"],
            parent_path: geometry["parent_contract_sha256"],
        }
        for path, digest in frozen_contracts.items():
            payload = _git_bytes(self.repository_root, commits[0], path)
            if sha256_bytes(payload) != digest:
                raise ValueError(f"geometry frozen contract identity drifted: {path}")

    def _validate_labels(self, raw_labels: Any) -> None:
        labels = _mapping(raw_labels, "restoration labels")
        binding = self._read_bound_json(
            labels["binding_path"], labels["binding_sha256"], "label binding"
        )
        raw = _mapping(binding.get("raw_archive"), "label raw archive")
        hf = _mapping(binding.get("hf_artifact"), "label HF artifact")
        expected = {
            "hf_repo": hf.get("repo"),
            "hf_revision": hf.get("immutable_revision"),
            "hf_tag": hf.get("tag"),
            "hf_path": hf.get("path"),
            "raw_archive_sha256": raw.get("sha256"),
            "raw_archive_size_bytes": raw.get("size_bytes"),
            "raw_archive_file_count": raw.get("file_count"),
            "raw_tree_inventory_sha256": raw.get("tree_inventory_sha256"),
        }
        if any(labels.get(key) != value for key, value in expected.items()):
            raise ValueError("restoration-label immutable identity drifted")
        if binding.get("status") != "VERIFIED_RESTORATION_V2_2_EAGER_LABEL_ARTIFACT":
            raise ValueError("restoration-label artifact is not verified")

    def _validate_derived(self, raw_derived: Any) -> None:
        derived = _mapping(raw_derived, "derived dataset")
        binding = self._read_bound_json(
            derived["binding_path"], derived["binding_sha256"], "derived binding"
        )
        hf = _mapping(binding.get("hf_dataset_artifact"), "derived HF artifact")
        expected = {
            "hf_repo": hf.get("repo"),
            "hf_revision": hf.get("immutable_revision"),
            "hf_tag": hf.get("tag"),
            "payload_prefix": hf.get("payload_prefix"),
            "artifact_tree_sha256": hf.get("artifact_tree_sha256"),
            "exact_files": hf.get("files"),
        }
        if any(derived.get(key) != value for key, value in expected.items()):
            raise ValueError("derived dataset immutable identity drifted")
        files = _sequence(derived.get("exact_files"), "derived exact files")
        if len(files) != 6:
            raise ValueError("derived dataset must bind exactly six files")
        for record in files:
            mapped = _mapping(record, "derived file")
            _exact_keys(mapped, {"path", "size_bytes", "sha256"}, "derived file")
            _safe_relative_path(mapped["path"], "derived file path")
            _require_sha256(mapped["sha256"], "derived file SHA256")
            if type(mapped["size_bytes"]) is not int or mapped["size_bytes"] < 0:
                raise ValueError("derived file size is invalid")

    def _validate_identity_witness(self, raw_witness: Any) -> None:
        witness = _mapping(raw_witness, "OCR/RGB identity witness")
        _exact_keys(
            witness,
            {
                "directory",
                "git_commit",
                "protocol_id",
                "status",
                "scientific_payload_sha256",
                "files",
            },
            "OCR/RGB identity witness",
        )
        if witness["git_commit"] != OCR_RGB_ARTIFACT_COMMIT:
            raise ValueError("OCR/RGB artifact commit drifted")
        for revision in (
            OCR_RGB_EXECUTION_SOURCE_COMMIT,
            OCR_RGB_ARTIFACT_COMMIT,
            OCR_RGB_VALIDATION_COMMIT,
        ):
            _require_commit(self.repository_root, revision)
        _require_ancestor(
            self.repository_root,
            OCR_RGB_EXECUTION_SOURCE_COMMIT,
            OCR_RGB_ARTIFACT_COMMIT,
        )
        _require_ancestor(
            self.repository_root,
            OCR_RGB_ARTIFACT_COMMIT,
            OCR_RGB_VALIDATION_COMMIT,
        )
        expected_identity = {
            "protocol_id": (
                "causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair"
            ),
            "status": (
                "COMPLETED_RESTORATION_V2_2_OCR_RGB_BASELINE_V2_IDENTITY_REPAIR"
            ),
            "scientific_payload_sha256": (
                "5942519bdff8f3e8a64bbc8b32a5d42a64abff37daf0804fcce0f098a63765b2"
            ),
        }
        if any(witness.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("OCR/RGB scientific identity drifted")
        directory = _safe_relative_path(witness["directory"], "OCR/RGB directory")
        files = _sequence(witness["files"], "OCR/RGB files")
        if tuple(record.get("path") for record in files) != EXPECTED_OUTPUT_FILES:
            raise ValueError("OCR/RGB exact file inventory drifted")
        for record in files:
            _validate_file_record(
                self.repository_root,
                record,
                prefix=directory,
                revision=OCR_RGB_ARTIFACT_COMMIT,
            )
            path = f"{directory}/{record['path']}"
            if _git_bytes(self.repository_root, OCR_RGB_VALIDATION_COMMIT, path) != (
                _git_bytes(self.repository_root, OCR_RGB_ARTIFACT_COMMIT, path)
            ):
                raise ValueError("OCR/RGB validation commit changed artifact bytes")
        summary = strict_json_object_bytes(
            (self.repository_root / directory / "summary.json").read_bytes(),
            label="OCR/RGB summary",
        )
        if any(summary.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("OCR/RGB summary identity drifted")
        self._validate_identity_rows(
            self.repository_root / directory / "state_scores.jsonl"
        )

    def _validate_identity_rows(self, path: Path) -> None:
        rows = [
            strict_json_object_bytes(line, label=f"OCR/RGB state row {index}")
            for index, line in enumerate(path.read_bytes().splitlines())
        ]
        if len(rows) != 15:
            raise ValueError("OCR/RGB identity witness must contain 15 state rows")
        state_indices: list[int] = []
        image_identities: set[tuple[str, str]] = set()
        roles: list[str] = []
        for row in rows:
            state = _mapping(row.get("state"), "OCR/RGB row state")
            state_indices.append(state.get("index"))
            roles.append(state.get("role"))
            candidates = _sequence(row.get("candidate_scores"), "candidate scores")
            if [item.get("event_step_id") for item in candidates] != [1, 2, 3, 4]:
                raise ValueError("OCR/RGB candidate identity order drifted")
            for item in candidates:
                image_identities.add(
                    (item.get("post_image_member_path"), item.get("post_image_sha256"))
                )
            current = _mapping(row.get("current_image"), "current image")
            image_identities.add(
                (current.get("image_member_path"), current.get("image_sha256"))
            )
        if tuple(state_indices) != EXPECTED_PRIMARY_STATE_INDICES:
            raise ValueError("OCR/RGB primary state identity drifted")
        if roles != ["v2_label_train"] * 10 + ["v2_development"] * 5:
            raise ValueError("OCR/RGB primary role identity drifted")
        if len(image_identities) != 75 or any(
            not isinstance(path_value, str)
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            for path_value, digest in image_identities
        ):
            raise ValueError("OCR/RGB 75-image identity witness drifted")

    def _validate_baseline(self, raw_baseline: Any) -> None:
        baseline = _mapping(raw_baseline, "baseline implementation")
        _exact_keys(
            baseline,
            {
                "manifest_path",
                "manifest_sha256",
                "artifact_id",
                "required_status",
                "vision_extractor_source",
                "eager_runtime_source",
            },
            "baseline implementation",
        )
        manifest = self._read_bound_json(
            baseline["manifest_path"],
            baseline["manifest_sha256"],
            "baseline manifest",
        )
        if (
            manifest.get("artifact_id") != baseline["artifact_id"]
            or manifest.get("status") != baseline["required_status"]
            or manifest.get("dependency_6_closed") is not True
        ):
            raise ValueError("baseline manifest status drifted")
        _validate_source_record(
            self.repository_root,
            baseline["vision_extractor_source"],
            "vision extractor source",
        )
        _validate_source_record(
            self.repository_root,
            baseline["eager_runtime_source"],
            "eager runtime source",
        )
        source_map = {
            record.get("path"): record.get("sha256")
            for record in _sequence(manifest.get("source_files"), "baseline sources")
        }
        vision_source = _mapping(
            baseline["vision_extractor_source"], "vision extractor source"
        )
        if source_map.get(vision_source["path"]) != vision_source["sha256"]:
            raise ValueError("vision extractor is not bound by baseline manifest")
        policy_vision = _mapping(
            _mapping(manifest.get("implementation_contract"), "implementation")
            .get("policy_vision"),
            "manifest policy vision",
        )
        required_manifest_values = {
            "model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
            "model_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
            "snapshot_file_count": 14,
            "snapshot_total_bytes": 17545907171,
            "feature_field": "pooler_output",
            "feature_semantics": "final_main_vision_merger_output",
            "language_model_called": False,
            "lm_head_called": False,
            "generate_called": False,
        }
        if any(
            policy_vision.get(key) != value
            for key, value in required_manifest_values.items()
        ):
            raise ValueError("baseline policy-vision declaration drifted")

    def _validate_model_snapshot(self, raw_snapshot: Any) -> None:
        snapshot = _mapping(raw_snapshot, "model snapshot")
        _exact_keys(
            snapshot,
            {
                "manifest_path",
                "manifest_sha256",
                "repo",
                "revision",
                "file_count",
                "total_bytes",
            },
            "model snapshot",
        )
        manifest = self._read_bound_json(
            snapshot["manifest_path"],
            snapshot["manifest_sha256"],
            "model snapshot manifest",
        )
        files = _sequence(manifest.get("files"), "model snapshot files")
        if (
            manifest.get("repo") != snapshot["repo"]
            or manifest.get("revision") != snapshot["revision"]
            or len(files) != snapshot["file_count"]
            or sum(record.get("size", -1) for record in files)
            != snapshot["total_bytes"]
        ):
            raise ValueError("model snapshot identity drifted")
        paths: list[str] = []
        for raw_record in files:
            record = _mapping(raw_record, "model snapshot file")
            _exact_keys(record, {"path", "size", "sha256"}, "model snapshot file")
            paths.append(_safe_relative_path(record["path"], "snapshot file path"))
            _require_sha256(record["sha256"], "snapshot file SHA256")
            if type(record["size"]) is not int or record["size"] < 0:
                raise ValueError("snapshot file size is invalid")
        if len(paths) != len(set(paths)) or paths != sorted(paths):
            raise ValueError("model snapshot file inventory is not unique and sorted")


def validate_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
    require_output_absent: bool = True,
) -> dict[str, Any]:
    contract = RestorationV22PolicyVisionContract.load(
        path,
        repository_root=repository_root,
    )
    output = _mapping(contract.data["output_contract"], "output contract")
    result_path = contract.repository_root.joinpath(
        *PurePosixPath(output["canonical_result_directory"]).parts
    )
    staging_path = result_path.with_name(f".{result_path.name}.staging")
    if require_output_absent and (result_path.exists() or staging_path.exists()):
        raise ValueError("policy-vision output or staging directory already exists")
    return {
        "status": PASS_STATUS,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": contract.sha256,
        "primary_state_count": 15,
        "unique_image_count": 75,
        "gpu_count": 2,
        "policy_vision_feature_forward_count": 31,
        "image_processor_batch_count": 16,
        "image_processing_assignment_count": 80,
        "cosine_scalar_transfer_count": 124,
        "language_model_forward_count": 0,
        "generation_count": 0,
        "confirm_locked": True,
        "sealed_test_locked": True,
        "output_absent": not result_path.exists(),
        "staging_output_absent": not staging_path.exists(),
    }
