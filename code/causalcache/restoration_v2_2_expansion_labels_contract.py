"""Fail-closed source contract for the 192-state expansion exact labels."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_restoration_v2_2_expansion_exact_labels_v1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_expansion_labels_v1.json"
)
CANONICAL_CONTRACT_PATH = (
    "code/causalcache/restoration_v2_2_expansion_labels_contract.py"
)
SOURCE_PARENT_GIT_COMMIT = "8ce24f511a4b6dbcbcd9d05c2d58abb5327f1baf"
FROZEN_CONFIG_SHA256 = (
    "65f7fa1d35a0b1fdd4fa09fe09120e858252406a3b850d85d9415ab34d6feed5"
)

SUBSTRATE_ARTIFACT_PATH = (
    "data/results/restoration_v2_2_label_expansion_substrate_v1/artifact.json"
)
SUBSTRATE_ARTIFACT_SHA256 = (
    "e7007edcf408805ee355e647f037063538897c1f6906bf5e98734f41293dcd02"
)
SUBSTRATE_HF_REVISION = "25ac19cf6ef98adc243d421cd0039ac104ddb539"
DERIVED_HF_REVISION = "630363a6adb692d72774f16dd0653a50216313ff"

ATTEMPT_ID = "restoration-v2-2-expansion-exact-labels-v1"
CANONICAL_OUTPUT_DIR = Path(f"/data/experiments/causalcache/{ATTEMPT_ID}")
CANONICAL_LEDGER_PATH = Path(
    f"/data/experiments/causalcache/.{ATTEMPT_ID}.attempt.json"
)
CANONICAL_ARCHIVE_PATH = Path(f"/data/experiments/causalcache/{ATTEMPT_ID}.tar")
CANONICAL_HF_REPO = (
    "gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-mobile"
)
CANONICAL_HF_TAG = "v2.2-expansion-exact-labels-v1"
CANONICAL_HF_PATH = f"raw/{CANONICAL_HF_TAG}.tar"

EXPECTED_TRAJECTORIES = 64
EXPECTED_STATES = 192
EXPECTED_DISTANCE_ROWS = 1792
EXPECTED_DEPLOYMENT_EDGES = 1856
EXPECTED_FULL_EDGES = 3072
EXPECTED_INTERACTIONS = 1984
EXPECTED_ATTRIBUTIONS = 576
EXPECTED_ORACLES = 192
EXPECTED_TEACHER_FORWARDS = 1984

SUBSTRATE_RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_restoration_v2_2_expansion_substrate_runner_v1.json"
)
SUBSTRATE_RUNNER_FREEZE_SHA256 = (
    "91a768202f653f4e2ca2960adf27a6c6c4f3a288d79c44345a865a1d53010040"
)
EXPECTED_SUBSTRATE_SOURCE_COUNT = 67
ADDITIONAL_PARENT_SOURCE_PATHS = (
    SUBSTRATE_RUNNER_FREEZE_PATH,
    SUBSTRATE_ARTIFACT_PATH,
)
RESERVED_EXECUTION_SOURCE_PATHS = (
    "code/causalcache/data/restoration_v2_2_expansion_label_parent.py",
    "code/causalcache/data/restoration_v2_2_expansion_label_inputs.py",
    "code/causalcache/restoration_v2_2_expansion_labels_contract.py",
    "code/causalcache/restoration_v2_2_expansion_labels_artifact.py",
    "code/scripts/run_restoration_v2_2_expansion_labels.py",
    "code/scripts/manage_restoration_v2_2_expansion_labels_artifact.py",
)

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json_object(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    payload = Path(path).read_bytes()
    value = json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload, value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _safe_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{name} must be a canonical repository-relative path")
    if path.as_posix() != value:
        raise ValueError(f"{name} must use canonical POSIX syntax")
    return value


def _repo_file(root: Path, relative: str) -> Path:
    canonical = _safe_relative_path(relative, "repository file")
    path = root.joinpath(*PurePosixPath(canonical).parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"repository file is missing or unsafe: {canonical}")
    return path


def _file_witness(root: Path, relative: str) -> dict[str, Any]:
    path = _repo_file(root, relative)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_file_witness(value: Any, name: str) -> dict[str, Any]:
    record = dict(_mapping(value, name))
    if set(record) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{name} field inventory drifted")
    _safe_relative_path(record.get("path"), f"{name}.path")
    if (
        not isinstance(record.get("sha256"), str)
        or SHA256_PATTERN.fullmatch(record["sha256"]) is None
    ):
        raise ValueError(f"{name}.sha256 must be a lowercase SHA256")
    if type(record.get("size_bytes")) is not int or record["size_bytes"] < 0:
        raise ValueError(f"{name}.size_bytes must be a non-negative integer")
    return record


def _inherited_source_files(root: Path) -> list[dict[str, Any]]:
    freeze_path = _repo_file(root, SUBSTRATE_RUNNER_FREEZE_PATH)
    freeze_payload, freeze = load_json_object(freeze_path)
    if (
        sha256_bytes(freeze_payload) != SUBSTRATE_RUNNER_FREEZE_SHA256
        or freeze_payload != pretty_json_bytes(freeze)
        or freeze.get("protocol_id")
        != "causalcache_restoration_v2_2_label_expansion_substrate_runner_freeze_v1"
        or freeze.get("status")
        != "FROZEN_COMMITTED_PUSHED_EXPANSION_SUBSTRATE_RUNNER_SOURCE"
    ):
        raise ValueError("substrate runner freeze identity drifted")
    inherited = [
        _validate_file_witness(value, f"substrate source {index}")
        for index, value in enumerate(
            _sequence(freeze.get("source_inventory"), "substrate source inventory")
        )
    ]
    if (
        len(inherited) != EXPECTED_SUBSTRATE_SOURCE_COUNT
        or [record["path"] for record in inherited]
        != sorted(record["path"] for record in inherited)
        or len({record["path"] for record in inherited}) != len(inherited)
    ):
        raise ValueError("substrate source inventory denominator drifted")
    for record in inherited:
        if _file_witness(root, record["path"]) != record:
            raise ValueError(f"substrate source drifted: {record['path']}")

    records = [
        *inherited,
        *(_file_witness(root, path) for path in ADDITIONAL_PARENT_SOURCE_PATHS),
    ]
    by_path = {record["path"]: record for record in records}
    if len(by_path) != EXPECTED_SUBSTRATE_SOURCE_COUNT + len(
        ADDITIONAL_PARENT_SOURCE_PATHS
    ):
        raise ValueError("exact-label inherited source inventory overlaps")
    return [by_path[path] for path in sorted(by_path)]


def _git_blob(root: Path, revision: str, relative: str) -> bytes:
    if revision != "HEAD" and GIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("Git revision must be HEAD or a full lowercase SHA")
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"{relative} is not committed at {revision}")
    return result.stdout


def _worker_topology() -> dict[str, Any]:
    return {
        "worker_count": 2,
        "gpu_model": "NVIDIA H200",
        "one_process_per_device": True,
        "workers": [
            {
                "worker_id": "even",
                "device": "cuda:0",
                "index_parity": 0,
                "state_index_rule": "range(0,192,2)",
                "state_count": 96,
            },
            {
                "worker_id": "odd",
                "device": "cuda:1",
                "index_parity": 1,
                "state_index_rule": "range(1,192,2)",
                "state_count": 96,
            },
        ],
        "cross_worker_state_stealing_allowed": False,
        "worker_failure_invalidates_entire_attempt": True,
        "worker_outputs_must_be_disjoint": True,
        "worker_union_must_equal_fixed_denominator": True,
    }


def _substrate_binding() -> dict[str, Any]:
    return {
        "path": SUBSTRATE_ARTIFACT_PATH,
        "sha256": SUBSTRATE_ARTIFACT_SHA256,
        "required_schema_version": "1.0.0",
        "required_protocol_id": (
            "causalcache_restoration_v2_2_label_expansion_substrate_artifact_v1"
        ),
        "required_status": (
            "VERIFIED_IMMUTABLE_LABEL_EXPANSION_SUBSTRATE_ARTIFACT"
        ),
        "required_outcome": "PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1",
        "required_state_count": EXPECTED_STATES,
        "required_generation_call_count": 384,
        "required_teacher_forward_count": 576,
        "required_kl_measurement_count": 384,
        "required_hf_repo": (
            "gavinlaw/causalcache-restoration-v2-2-label-expansion-substrate-mobile"
        ),
        "required_hf_tag": "v2.2-label-expansion-substrate-v1",
        "required_hf_revision": SUBSTRATE_HF_REVISION,
        "required_hf_path": "raw/v2.2-label-expansion-substrate-v1.tar",
        "required_raw_sha256": (
            "4e77a38be34cb2f3c084a13abd47c0530e6729ff6cca72793977c62fa78ff47d"
        ),
        "required_raw_size_bytes": 7475200,
        "required_raw_file_count": 406,
        "required_raw_tree_sha256": (
            "56f291053121ccf813698a48beb6269b1fa1d096b7e974f6eb9424f55bc45543"
        ),
        "required_execution_git_commit": (
            "642feb28b4f7ce4e7bf9f7791f7fb0f6919c1839"
        ),
        "required_runner_source_git_commit": (
            "1a3833d6951c768ce1bdd5f976d1044c291d002e"
        ),
        "required_runner_freeze_sha256": (
            "91a768202f653f4e2ca2960adf27a6c6c4f3a288d79c44345a865a1d53010040"
        ),
        "required_run_contract_sha256": (
            "b467113130da74d943a00c2f11d7cad242204e475b7b806e76f666e670a1ed6a"
        ),
    }


def build_contract(*, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "preregistration_status": (
            "source_only_frozen_after_substrate_pass_before_any_expansion_label_forward"
        ),
        "freeze": {
            "source_parent_git_commit": SOURCE_PARENT_GIT_COMMIT,
            "substrate_pass_precedes_this_freeze": True,
            "restoration_output_accessed_to_choose_contract": False,
            "confirm_state_or_output_accessed": False,
            "source_only_validator_authorizes_label_execution": False,
            "source_only_validator_authorizes_policy_or_gpu_execution": False,
        },
        "immutable_inputs": {
            "substrate_artifact": _substrate_binding(),
            "derived_artifact": {
                "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
                "tag": "restoration-v2-label-expansion-v1.0.0",
                "immutable_revision": DERIVED_HF_REVISION,
                "payload_prefix": "derived/restoration-v2-label-expansion-v1",
                "artifact_tree_sha256": (
                    "9394b369e2b741e6aacf9ece4fc5dae3e6337b25a7e65402307e4e7862b94abc"
                ),
            },
        },
        "data_projection": {
            "role_order": [
                "gate_train_expansion",
                "gate_development_expansion",
            ],
            "role_trajectory_counts": {
                "gate_train_expansion": 48,
                "gate_development_expansion": 16,
            },
            "role_state_counts": {
                "gate_train_expansion": 144,
                "gate_development_expansion": 48,
            },
            "fixed_trajectory_denominator": EXPECTED_TRAJECTORIES,
            "fixed_state_denominator": EXPECTED_STATES,
            "decision_steps_per_trajectory": [4, 5, 6],
            "candidate_event_counts_by_decision_step": {"4": 2, "5": 3, "6": 4},
            "state_projection_sha256": (
                "fc1c6bed6f069e4df28a12a835fa2a9264b532f206cc30c211e6e8f0e5f21be7"
            ),
            "fresh_state_output_required": True,
            "state_filtering_top_up_or_replacement_allowed": False,
            "confirm_state_prompt_image_or_action_access_allowed": False,
            "all_coalitions_from_one_state_must_remain_in_one_split": True,
        },
        "estimand": {
            "reference_mode": (
                "substrate_canonical_action_frozen_fresh_teacher_recompute"
            ),
            "reference_policy_runtime": (
                "causalcache_restoration_v2_2_eager_runtime"
            ),
            "distance": (
                "teacher_forced_full_vocabulary_mean_kl_on_complete_official_tool_call_span"
            ),
            "reference_distribution": "fresh_full_history_teacher_log_probs",
            "candidate_distribution": "fresh_mixed_fidelity_teacher_logits",
            "high_fidelity_event_representation": "post_state_image_only",
            "low_fidelity_event_representation": "frozen_eight_field_strong_summary",
            "full_history_reference_coalition": "all_candidate_event_step_ids",
            "utility": "D(empty)-D(S)",
            "conditional_marginal": "D(S)-D(S_union_j)",
            "negative_utility_or_marginal_clipped": False,
            "full_tensor_logit_or_probability_host_transfer_allowed": False,
            "distance_scalar_host_transfer_per_measurement": 1,
        },
        "enumeration": {
            "raw_distance_domain": "complete_power_set",
            "event_identity": "event_step_id",
            "event_step_ids_sorted_ascending": True,
            "unit_slot_cost_per_event": 1,
            "primary_deployment_budget_slots": 2,
            "budget_is_injected_into_policy_prompt": False,
            "coalition_order": "cardinality_then_lexicographic_event_step_ids",
            "raw_distance_rows_by_decision_step": {"4": 4, "5": 8, "6": 16},
            "raw_distance_row_counts": {
                "gate_train_expansion": 1344,
                "gate_development_expansion": 448,
                "total": EXPECTED_DISTANCE_ROWS,
            },
            "deployment_conditional_edges_by_decision_step": {
                "4": 4,
                "5": 9,
                "6": 16,
            },
            "deployment_conditional_edge_counts": {
                "gate_train_expansion": 1392,
                "gate_development_expansion": 464,
                "total": EXPECTED_DEPLOYMENT_EDGES,
            },
            "full_hypercube_edges_by_decision_step": {"4": 4, "5": 12, "6": 32},
            "full_hypercube_edge_counts": {
                "gate_train_expansion": 2304,
                "gate_development_expansion": 768,
                "total": EXPECTED_FULL_EDGES,
            },
            "pair_interactions_by_decision_step": {"4": 1, "5": 6, "6": 24},
            "full_pair_interaction_counts": {
                "gate_train_expansion": 1488,
                "gate_development_expansion": 496,
                "total": EXPECTED_INTERACTIONS,
            },
            "exact_permutation_attributions_by_decision_step": {
                "4": 2,
                "5": 3,
                "6": 4,
            },
            "exact_permutation_attribution_counts": {
                "gate_train_expansion": 432,
                "gate_development_expansion": 144,
                "total": EXPECTED_ATTRIBUTIONS,
            },
            "primary_exact_subset_oracle_counts": {
                "gate_train_expansion": 144,
                "gate_development_expansion": 48,
                "total": EXPECTED_ORACLES,
            },
        },
        "reduction": {
            "canonical_truth": "raw_state_coalition_distance_table",
            "all_derived_rows_recomputed_policy_free_from_canonical_truth": True,
            "exact_oracle_domain": (
                "all_coalitions_with_cardinality_at_most_primary_budget_including_empty"
            ),
            "exact_oracle_tie_epsilon": 0.0,
            "exact_oracle_tie_break": [
                "lower_distance",
                "lower_cardinality",
                "lexicographically_lower_sorted_event_step_ids",
            ],
            "negative_utility_or_marginal_clipped": False,
            "negative_conditional_marginals_are_retained_as_signed_values": True,
            "greedy_stopping_threshold": 0.0,
            "greedy_stops_when_maximum_conditional_marginal_is_not_positive": True,
            "state_normalization_epsilon": 1e-12,
            "zero_or_near_zero_oracle_utility_ratio": None,
            "conditional_edge_weighting_for_gate_training": (
                "deferred_to_separate_gate_training_contract"
            ),
            "exact_permutation_attribution": "uniform_over_all_event_permutations",
            "required_algebraic_checks": [
                "U(empty)=0",
                "Delta=D(S)-D(S_union_j)",
                "edge_utility_difference_equals_D_difference",
                "path_telescoping",
                "pair_interaction_symmetry",
                "exact_oracle_recomputed_from_raw_table",
            ],
        },
        "runtime": {
            "dtype": "bfloat16",
            "attention_implementation": "eager",
            "seed": 0,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "strict_cuda_determinism_claimed": False,
            "microbatch_size": 1,
            "expected_execution_stack": {
                "container_image_digest": (
                    "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
                ),
                "python_version": "3.12.3",
                "torch_version": "2.11.0+cu130",
                "torch_cuda_version": "13.0",
                "cudnn_version": 91900,
                "transformers_version": "5.6.0",
                "nvidia_driver_version": "570.172.08",
                "gpu_name": "NVIDIA H200",
            },
        },
        "operation_schedule": {
            "generation_call_count": 0,
            "reference_teacher_forward_count": 192,
            "reference_repeat_teacher_forward_count": 192,
            "non_full_coalition_teacher_forward_count": 1600,
            "total_teacher_forward_call_count": EXPECTED_TEACHER_FORWARDS,
            "total_teacher_forward_example_count": EXPECTED_TEACHER_FORWARDS,
            "non_full_coalition_kl_measurement_count": 1600,
            "reference_repeat_kl_measurement_count": 192,
            "total_kl_measurement_count": EXPECTED_DISTANCE_ROWS,
            "maximum_reference_repeat_kl": 0.0001,
            "raw_distance_row_count": EXPECTED_DISTANCE_ROWS,
            "deployment_conditional_label_count": EXPECTED_DEPLOYMENT_EDGES,
            "full_hypercube_edge_count": EXPECTED_FULL_EDGES,
            "full_pair_interaction_count": EXPECTED_INTERACTIONS,
            "exact_permutation_attribution_count": EXPECTED_ATTRIBUTIONS,
            "primary_exact_subset_oracle_count": EXPECTED_ORACLES,
            "retry_count": 0,
            "top_up_count": 0,
            "resume_allowed": False,
            "completed_state_regeneration_allowed": False,
            "incomplete_attempt_retry_allowed": False,
        },
        "worker_topology": _worker_topology(),
        "prohibited_work": {
            "maximum_generation_call_count": 0,
            "maximum_confirm_state_access_count": 0,
            "maximum_confirm_processor_prompt_count": 0,
            "maximum_confirm_decoder_input_count": 0,
            "maximum_confirm_generation_count": 0,
            "maximum_confirm_teacher_forward_count": 0,
            "maximum_expert_action_read_count": 0,
            "maximum_gate_training_example_count": 0,
            "maximum_gate_model_forward_count": 0,
            "maximum_gate_selection_count": 0,
            "maximum_matched_nll_evaluation_count": 0,
            "maximum_closed_loop_episode_count": 0,
            "androidworld_test_split_access_allowed": False,
        },
        "scientific_source_lock": {
            "source_files": _inherited_source_files(root),
            "reserved_execution_source_paths": list(
                RESERVED_EXECUTION_SOURCE_PATHS
            ),
            "prompt_policy_parser_teacher_kl_semantics_inherited_from_v2_2_eager": True,
            "distance_reduction_semantics_inherited_from_legacy_exact_labels": True,
            "formal_execution_requires_separate_committed_pushed_runner_freeze": True,
            "scientific_environment_variables_allowed": False,
        },
        "execution": {
            "attempt_id": ATTEMPT_ID,
            "canonical_persistent_output_dir": str(CANONICAL_OUTPUT_DIR),
            "canonical_global_attempt_ledger": str(CANONICAL_LEDGER_PATH),
            "canonical_raw_archive": str(CANONICAL_ARCHIVE_PATH),
            "required_host_class": "Hyper_H200",
            "same_host_and_container_required_for_both_workers": True,
            "alternate_output_or_ledger_allowed": False,
            "output_or_ledger_deletion_after_first_attempt_allowed": False,
            "runtime_import_requires_fresh_substrate_and_derived_validation": True,
            "execution_locked_until_runner_source_is_committed_and_pushed": True,
            "source_only_contract_authorizes_execution": False,
        },
        "artifact_destination": {
            "repo": CANONICAL_HF_REPO,
            "repo_type": "dataset",
            "visibility": "private",
            "tag": CANONICAL_HF_TAG,
            "raw_path": CANONICAL_HF_PATH,
            "immutable_revision": None,
            "git_result_dir": (
                "data/results/restoration_v2_2_expansion_exact_labels_v1"
            ),
            "git_tracks_raw_artifact": False,
        },
        "promotion": {
            "substrate_pass_is_necessary_but_not_execution_authorization": True,
            "source_contract_authorizes_only_runner_source_implementation": True,
            "separate_runner_freeze_required_before_gpu_execution": True,
            "gate_training_matched_nll_closed_loop_and_confirm_remain_locked": True,
            "source_only_validator_authorizes_policy_or_gpu_execution": False,
        },
    }


def _validate_substrate_artifact(
    root: Path,
    *,
    require_git_blobs: bool,
) -> None:
    path = _repo_file(root, SUBSTRATE_ARTIFACT_PATH)
    payload, artifact = load_json_object(path)
    if sha256_bytes(payload) != SUBSTRATE_ARTIFACT_SHA256:
        raise ValueError("substrate artifact SHA256 drifted")
    if require_git_blobs:
        if _git_blob(root, "HEAD", SUBSTRATE_ARTIFACT_PATH) != payload:
            raise ValueError("substrate artifact differs from HEAD")
        if _git_blob(root, SOURCE_PARENT_GIT_COMMIT, SUBSTRATE_ARTIFACT_PATH) != payload:
            raise ValueError("substrate artifact differs from source parent")

    binding = _substrate_binding()
    result = _mapping(artifact.get("result"), "substrate result")
    actual = _mapping(result.get("actual_counts"), "substrate actual counts")
    hf = _mapping(artifact.get("hf_artifact"), "substrate HF artifact")
    raw = _mapping(artifact.get("raw_archive"), "substrate raw archive")
    source = _mapping(artifact.get("source"), "substrate source")
    inputs = _mapping(artifact.get("input_artifacts"), "substrate inputs")
    negatives = _mapping(
        artifact.get("negative_declarations"), "substrate negative declarations"
    )
    observed = {
        "schema_version": artifact.get("schema_version"),
        "protocol_id": artifact.get("protocol_id"),
        "status": artifact.get("status"),
        "outcome": result.get("outcome"),
        "state_count": result.get("completed_state_count"),
        "generation_call_count": actual.get("generation_call_count"),
        "teacher_forward_count": actual.get("teacher_forward_count"),
        "kl_measurement_count": actual.get("kl_measurement_count"),
        "hf_repo": hf.get("repo"),
        "hf_tag": hf.get("tag"),
        "hf_revision": hf.get("immutable_revision"),
        "hf_path": hf.get("path"),
        "raw_sha256": raw.get("sha256"),
        "raw_size_bytes": raw.get("size_bytes"),
        "raw_file_count": raw.get("file_count"),
        "raw_tree_sha256": raw.get("tree_inventory_sha256"),
        "execution_git_commit": source.get("execution_git_commit"),
        "runner_source_git_commit": source.get("runner_source_git_commit"),
        "runner_freeze_sha256": source.get("runner_freeze_sha256"),
        "run_contract_sha256": source.get("run_contract_sha256"),
        "derived_revision": inputs.get("derived_immutable_revision"),
        "derived_tree_sha256": inputs.get("derived_artifact_tree_sha256"),
    }
    expected = {
        "schema_version": binding["required_schema_version"],
        "protocol_id": binding["required_protocol_id"],
        "status": binding["required_status"],
        "outcome": binding["required_outcome"],
        "state_count": binding["required_state_count"],
        "generation_call_count": binding["required_generation_call_count"],
        "teacher_forward_count": binding["required_teacher_forward_count"],
        "kl_measurement_count": binding["required_kl_measurement_count"],
        "hf_repo": binding["required_hf_repo"],
        "hf_tag": binding["required_hf_tag"],
        "hf_revision": binding["required_hf_revision"],
        "hf_path": binding["required_hf_path"],
        "raw_sha256": binding["required_raw_sha256"],
        "raw_size_bytes": binding["required_raw_size_bytes"],
        "raw_file_count": binding["required_raw_file_count"],
        "raw_tree_sha256": binding["required_raw_tree_sha256"],
        "execution_git_commit": binding["required_execution_git_commit"],
        "runner_source_git_commit": binding["required_runner_source_git_commit"],
        "runner_freeze_sha256": binding["required_runner_freeze_sha256"],
        "run_contract_sha256": binding["required_run_contract_sha256"],
        "derived_revision": DERIVED_HF_REVISION,
        "derived_tree_sha256": (
            "9394b369e2b741e6aacf9ece4fc5dae3e6337b25a7e65402307e4e7862b94abc"
        ),
    }
    if observed != expected:
        raise ValueError("substrate artifact identity drifted")
    if set(negatives.values()) != {False}:
        raise ValueError("substrate negative declarations must all remain false")


def validate_contract_data(
    data: Mapping[str, Any],
    *,
    repository_root: str | Path,
    require_git_blobs: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if dict(data) != build_contract(repository_root=root):
        raise ValueError("expansion exact-label contract differs from deterministic freeze")
    _validate_substrate_artifact(root, require_git_blobs=require_git_blobs)

    if require_git_blobs:
        source_lock = _mapping(data["scientific_source_lock"], "source lock")
        source_files = _sequence(source_lock.get("source_files"), "source files")
        for index, raw_record in enumerate(source_files):
            record = _validate_file_witness(raw_record, f"source file {index}")
            relative = record["path"]
            path = _repo_file(root, relative)
            payload = path.read_bytes()
            if _git_blob(root, "HEAD", relative) != payload:
                raise ValueError(f"{relative} differs from HEAD")
            if _git_blob(root, SOURCE_PARENT_GIT_COMMIT, relative) != payload:
                raise ValueError(f"{relative} differs from source parent")

    workers = _sequence(data["worker_topology"]["workers"], "workers")
    if [worker["state_count"] for worker in workers] != [96, 96]:
        raise ValueError("worker state counts drifted")
    prohibited = _mapping(data["prohibited_work"], "prohibited work")
    if any(value not in {0, False} for value in prohibited.values()):
        raise ValueError("all prohibited-work counts and permissions must be zero/false")
    return {
        "contract_valid": True,
        "protocol_id": PROTOCOL_ID,
        "trajectory_count": EXPECTED_TRAJECTORIES,
        "state_count": EXPECTED_STATES,
        "raw_distance_row_count": EXPECTED_DISTANCE_ROWS,
        "deployment_conditional_label_count": EXPECTED_DEPLOYMENT_EDGES,
        "full_hypercube_edge_count": EXPECTED_FULL_EDGES,
        "full_pair_interaction_count": EXPECTED_INTERACTIONS,
        "exact_permutation_attribution_count": EXPECTED_ATTRIBUTIONS,
        "primary_exact_subset_oracle_count": EXPECTED_ORACLES,
        "teacher_forward_count": EXPECTED_TEACHER_FORWARDS,
        "worker_state_counts": {"even": 96, "odd": 96},
        "policy_or_gpu_execution_authorized_by_this_validator": False,
        "label_execution_authorized_by_this_validator": False,
        "confirm_access_authorized_by_this_validator": False,
    }


def load_and_validate_contract(
    config_path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    supplied = Path(config_path)
    actual = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    expected = (root / CANONICAL_CONFIG_PATH).resolve()
    if actual != expected or actual.is_symlink() or not actual.is_file():
        raise ValueError(f"expansion exact-label config must be {CANONICAL_CONFIG_PATH}")
    payload, data = load_json_object(actual)
    if payload != pretty_json_bytes(data):
        raise ValueError("expansion exact-label config must be canonical pretty JSON")
    if sha256_bytes(payload) != FROZEN_CONFIG_SHA256:
        raise ValueError("expansion exact-label config SHA256 drifted")
    if _git_blob(root, "HEAD", CANONICAL_CONFIG_PATH) != payload:
        raise ValueError("expansion exact-label config differs from HEAD")
    contract_path = _repo_file(root, CANONICAL_CONTRACT_PATH)
    if _git_blob(root, "HEAD", CANONICAL_CONTRACT_PATH) != contract_path.read_bytes():
        raise ValueError("expansion exact-label contract source differs from HEAD")
    validation = validate_contract_data(
        data,
        repository_root=root,
        require_git_blobs=True,
    )
    validation["config_sha256"] = sha256_bytes(payload)
    validation["source_parent_git_commit"] = SOURCE_PARENT_GIT_COMMIT
    return data, validation


__all__ = [
    "ATTEMPT_ID",
    "CANONICAL_CONFIG_PATH",
    "CANONICAL_HF_PATH",
    "CANONICAL_HF_REPO",
    "CANONICAL_HF_TAG",
    "DERIVED_HF_REVISION",
    "EXPECTED_ATTRIBUTIONS",
    "EXPECTED_DEPLOYMENT_EDGES",
    "EXPECTED_DISTANCE_ROWS",
    "EXPECTED_FULL_EDGES",
    "EXPECTED_INTERACTIONS",
    "EXPECTED_ORACLES",
    "EXPECTED_STATES",
    "EXPECTED_TEACHER_FORWARDS",
    "FROZEN_CONFIG_SHA256",
    "PROTOCOL_ID",
    "SOURCE_PARENT_GIT_COMMIT",
    "SUBSTRATE_ARTIFACT_PATH",
    "SUBSTRATE_HF_REVISION",
    "build_contract",
    "load_and_validate_contract",
    "load_json_object",
    "pretty_json_bytes",
    "sha256_file",
    "validate_contract_data",
]
