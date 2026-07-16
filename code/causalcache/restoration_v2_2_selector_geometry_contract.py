"""Fail-closed source contract for restoration-v2.2 selector geometry."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "causalcache_restoration_v2_2_selector_geometry_v1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_selector_geometry.json"
)
FROZEN_CONFIG_SHA256 = (
    "8022dcdec272916b7975d696a3ce6b54022c7414cd348a715c55b0d3d694dad5"
)
PASS_STATUS = "VALID_RESTORATION_V2_2_SELECTOR_GEOMETRY_SOURCE_CONTRACT"

EXPECTED_STATE_COUNT = 45
EXPECTED_TRAJECTORY_COUNT = 15
PRIMARY_EVENT_COUNT = 4
PRIMARY_BUDGET = 2
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 271_828
BOOTSTRAP_CONFIDENCE = 0.9

LABEL_BINDING_PATH = (
    "data/results/restoration_v2_2_eager_labels_v2/artifact.json"
)
LABEL_BINDING_SHA256 = (
    "de2960a73399327871a6472315d543b3de60ce2ad13136166ce5bbeb609f1777"
)
LABEL_SUMMARY_PATH = (
    "data/results/restoration_v2_2_eager_labels_v2/summary.json"
)
LABEL_SUMMARY_SHA256 = (
    "0bacfe08b487e9ddc37a7214dde6d3b8e8f4b6523f64d0b420731655ddd2ffa0"
)
DERIVED_BINDING_PATH = "data/manifests/restoration_v2_derived_artifact.json"
DERIVED_BINDING_SHA256 = (
    "1f155a90af33da21947c89ac12f30f873f18e15e4efc82e5759f648114f25f7c"
)
BASELINE_BINDING_PATH = "data/manifests/restoration_v2_baselines.json"
BASELINE_BINDING_SHA256 = (
    "bd6f4d620d02bde478bf765c6fa3605bb86e20ebf449e999654e7a6766ba5710"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
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


def load_strict_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_bytes(),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} key schema drifted: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{label} is not a canonical repository-relative path")
    if path.as_posix() != value:
        raise ValueError(f"{label} must use canonical POSIX syntax")
    return value


def _bound_json(
    repository_root: Path,
    *,
    relative_path: Any,
    expected_sha256: Any,
    label: str,
) -> Mapping[str, Any]:
    relative = _safe_relative_path(relative_path, f"{label}.path")
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label}.sha256 is invalid")
    path = (repository_root / relative).resolve()
    if repository_root not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing, symlinked, or escapes the repository")
    payload = path.read_bytes()
    _equal(sha256_bytes(payload), expected_sha256, f"{label} SHA256")
    value = json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    return _mapping(value, f"{label} value")


EXPECTED_AUTHORIZATION = {
    "policy_free_selector_geometry_allowed": True,
    "restoration_teacher_forward_allowed": False,
    "new_kl_measurement_allowed": False,
    "gate_training_allowed": False,
    "matched_nll_allowed": False,
    "closed_loop_allowed": False,
    "confirm_access_allowed": False,
    "sealed_test_access_allowed": False,
    "policy_vision_feature_extraction_allowed_in_this_stage": False,
    "policy_vision_requires_separate_feature_only_source_freeze": True,
}

EXPECTED_ANALYSIS_DENOMINATOR = {
    "fixed_state_count": 45,
    "fixed_trajectory_count": 15,
    "states_per_trajectory": 3,
    "role_state_counts": {"v2_label_train": 30, "v2_development": 15},
    "role_trajectory_counts": {"v2_label_train": 10, "v2_development": 5},
    "candidate_event_count_histogram": {"2": 15, "3": 15, "4": 15},
    "decision_step_histogram": {"4": 15, "5": 15, "6": 15},
    "candidate_count_identity": "n_equals_decision_step_id_minus_2",
    "canonical_scientific_truth": "raw_complete_power_set_D_of_S",
    "raw_distance_row_count": 420,
    "confirm_state_count": 0,
    "sealed_test_state_count": 0,
}

EXPECTED_NORMALIZATION = {
    "utility": "D_empty_minus_D_S",
    "epsilon": 1e-12,
    "normalized_recovery": (
        "utility_divided_by_D_empty_when_D_empty_gt_epsilon_else_null"
    ),
    "negative_utility_clamped": False,
    "negative_marginal_clamped": False,
    "state_weighting": "equal_within_each_reported_stratum",
    "trajectory_is_the_inference_unit": True,
}

EXPECTED_ANALYSIS_GRID = {
    "primary_setting": {
        "candidate_event_count": 4,
        "decision_step_id": 6,
        "budget_event_capacity": 2,
        "compression_ratio": 0.5,
        "required_role_reports": ["v2_label_train", "v2_development"],
    },
    "candidate_event_counts": [2, 3, 4],
    "decision_step_ids": [4, 5, 6],
    "budget_curve": {
        "per_state_integer_range": "0_through_n_inclusive",
        "budgets_are_event_slot_capacities": True,
        "report_by_candidate_count_and_role": True,
        "n_equals_2_B_equals_2_is_sanity_ceiling": True,
        "n_greater_than_or_equal_to_3_B_equals_2_is_hard_case_aggregate": True,
    },
}

EXPECTED_SELECTORS = {
    "exact_subset_oracle": {
        "eligible_coalitions": "all_S_with_cardinality_at_most_B",
        "allows_early_stop": True,
        "objective": "minimum_D_S",
        "tie_epsilon": 0.0,
        "tie_break": [
            "lower_D_S",
            "smaller_cardinality",
            "lexicographically_smaller_sorted_event_step_ids",
        ],
    },
    "exact_cardinality_oracle": {
        "role": "forced_fill_nonmonotonicity_sensitivity",
        "eligible_coalitions": "all_S_with_cardinality_exactly_B",
        "objective": "minimum_D_S",
        "tie_break": "lexicographically_smaller_sorted_event_step_ids",
    },
    "true_conditional_marginal_greedy": {
        "initial_coalition": [],
        "marginal_gain": "D_S_minus_D_S_union_j",
        "candidate_rescoring_after_every_selection": True,
        "selection": "maximum_true_conditional_marginal_gain",
        "event_tie_break": "lower_event_step_id",
        "continue_condition": (
            "strict_gain_greater_than_0_and_cardinality_less_than_B"
        ),
        "allows_early_stop": True,
    },
    "forced_fill_true_conditional_marginal_greedy": {
        "role": "early_stop_sensitivity",
        "selection": (
            "maximum_true_conditional_marginal_gain_recomputed_after_every_selection"
        ),
        "event_tie_break": "lower_event_step_id",
        "stop_condition": "cardinality_equals_B_even_if_gain_is_nonpositive",
    },
    "budget_conditioned_independent": {
        "role": "primary_independent_objective_projection_baseline",
        "event_score": (
            "mean_true_marginal_over_all_base_coalitions_of_cardinality_"
            "B_minus_1_that_exclude_event"
        ),
        "budget_zero_rule": "all_scores_zero_and_select_empty",
        "score_uses_only_same_state_D_table": True,
        "selection": "up_to_B_strictly_positive_scores",
        "score_tie_break": "lower_event_step_id",
        "selected_coalition_serialization": "chronological_sorted_event_step_ids",
        "allows_early_stop": True,
    },
    "forced_fill_budget_conditioned_independent": {
        "role": "early_stop_sensitivity",
        "score_source": "same_budget_conditioned_independent_scores",
        "selection": "exactly_B_highest_additive_score_events",
        "score_tie_break": "lower_event_step_id",
    },
    "full_shapley_independent": {
        "role": "secondary_independent_ablation",
        "event_score": (
            "exact_uniform_average_of_predecessor_marginals_over_all_n_"
            "factorial_permutations"
        ),
        "selection": "up_to_B_strictly_positive_scores",
        "score_tie_break": "lower_event_step_id",
        "allows_early_stop": True,
    },
    "forced_fill_full_shapley_independent": {
        "role": "early_stop_sensitivity",
        "score_source": "same_full_shapley_independent_scores",
        "selection": "exactly_B_highest_additive_score_events",
        "score_tie_break": "lower_event_step_id",
    },
    "dynamic_recent": {
        "selection": "last_min_B_n_chronological_candidate_event_step_ids",
        "exact_cardinality": "min_B_n",
        "allows_early_stop": False,
    },
    "uniform_random_exact_expectation": {
        "selection_domain": "all_exact_k_subsets_where_k_equals_min_B_n",
        "aggregation": (
            "arithmetic_mean_of_actual_normalized_recovery_within_state"
        ),
        "sampled": False,
        "seed": None,
        "allows_early_stop": False,
    },
    "ocr_and_rgb_similarity": {
        "scope": "n_equals_4_B_equals_2_only",
        "selection": "top_2_by_frozen_similarity_then_lower_event_step_id",
        "requires_immutable_derived_dataset": True,
        "new_policy_forward_required": False,
    },
    "frozen_policy_vision_similarity": {
        "scope": "n_equals_4_B_equals_2_only",
        "selection": "top_2_by_frozen_similarity_then_lower_event_step_id",
        "feature": (
            "mean_L2_normalized_final_main_vision_merger_pooler_output"
        ),
        "language_model_called": False,
        "generation_called": False,
        "embedding_cache_present_in_labels_or_derived_artifact": False,
        "execution": (
            "separate_feature_only_stage_after_its_own_source_freeze"
        ),
    },
}

EXPECTED_PRIMARY_METRICS = {
    "per_selector": [
        "trajectory_weighted_mean_actual_utility",
        "trajectory_weighted_mean_normalized_recovery",
        "trajectory_weighted_recovery_ratio_of_means_to_exact_subset_oracle",
        "exact_coalition_match_rate",
        "selected_cardinality_histogram",
    ],
    "paired_gaps": {
        "search_gap": (
            "exact_subset_oracle_minus_true_conditional_marginal_greedy"
        ),
        "objective_projection_gap": (
            "true_conditional_marginal_greedy_minus_budget_conditioned_independent"
        ),
        "full_shapley_projection_gap": (
            "true_conditional_marginal_greedy_minus_full_shapley_independent"
        ),
    },
    "aggregation_order": (
        "state_equal_within_trajectory_then_trajectory_equal_within_reported_stratum"
    ),
    "ratio_definition": (
        "trajectory_weighted_mean_selector_recovery_divided_by_trajectory_"
        "weighted_mean_exact_recovery"
    ),
    "utility_ratio_zero_denominator_rule": (
        "null_when_trajectory_weighted_mean_exact_recovery_is_at_most_epsilon"
    ),
    "per_state_ratio_is_primary": False,
    "per_state_nonpositive_exact_ratio_rule": (
        "null_and_excluded_from_any_secondary_ratio_summary"
    ),
    "score_sum_may_replace_actual_set_utility": False,
}

EXPECTED_INTERACTION_ANALYSIS = {
    "pair_interaction": "Delta_i_S_union_j_minus_Delta_i_S",
    "pair_interaction_equivalent": (
        "D_S_union_j_minus_D_S_union_i_j_minus_D_S_plus_D_S_union_i"
    ),
    "per_state_metrics": [
        "positive_interaction_count",
        "negative_interaction_count",
        "mean_absolute_interaction",
        "normalized_mean_absolute_interaction",
        "redundancy_share",
        "strict_negative_deployment_marginal_count_and_rate",
        "epsilon_1e12_negative_deployment_marginal_count_and_rate",
    ],
    "normalized_interaction_mass": (
        "mean_abs_pair_interactions_divided_by_max_D_empty_epsilon"
    ),
    "strength_strata": {
        "cutpoint_source": (
            "v2_label_train_tertiles_computed_separately_within_each_"
            "candidate_event_count"
        ),
        "development_application": (
            "apply_frozen_train_cutpoints_within_matching_candidate_event_count"
        ),
        "labels": ["low", "mid", "high"],
    },
    "required_selector_reports_by": [
        "role",
        "candidate_event_count",
        "budget",
        "interaction_strength_stratum",
        "has_strict_negative_deployment_marginal",
    ],
}

EXPECTED_UNCERTAINTY = {
    "method": "paired_cluster_bootstrap",
    "resampling_unit": "trajectory_id",
    "replicate_count": 10_000,
    "seed": 271_828,
    "confidence_level": 0.9,
    "interval": "percentile",
    "quantiles": [0.05, 0.95],
    "pairing": (
        "reuse_identical_sampled_trajectory_multiplicities_for_all_selectors_"
        "in_a_comparison"
    ),
    "role_handling": (
        "resample_trajectories_with_replacement_within_role_and_keep_all_"
        "eligible_states"
    ),
    "primary_inference_role": "v2_development",
    "training_role_is_descriptive": True,
}

EXPECTED_INTERNAL_METHOD_SHAPING = {
    "paper_claim_gate": False,
    "confirm_unlock_gate": False,
    "set_conditioning_materiality": {
        "scope": "v2_development_n_equals_4_B_equals_2",
        "green_minimum_mean_normalized_recovery_advantage_over_budget_conditioned_independent": 0.05,
        "green_minimum_same_direction_trajectory_count_out_of_5": 4,
        "paired_90_percent_bootstrap_lower_bound_must_exceed": 0.0,
        "green_requires_all_conditions": True,
    },
    "set_conditioning_yellow_band": {
        "definition": "neither_green_nor_independent_simplification",
    },
    "independent_simplification": {
        "maximum_mean_normalized_recovery_advantage_for_simplification": 0.01,
    },
    "greedy_search_sufficiency": {
        "scope": "v2_development_n_equals_4_B_equals_2",
        "green_minimum_mean_true_greedy_to_exact_recovery_ratio": 0.9,
        "green_maximum_mean_normalized_recovery_search_gap": 0.05,
        "strong_diagnosis_ratio_strictly_less_than": 0.85,
        "strong_diagnosis_gap_strictly_greater_than": 0.1,
        "intermediate_region": "yellow",
        "strong_diagnosis_action": (
            "add_existing_swap_and_beam_only_as_train_development_offline_"
            "diagnostics"
        ),
    },
    "visual_baseline_requirement": {
        "scope": "n_equals_4_B_equals_2",
        "required_before_gate_training_claims": True,
        "policy_vision_result_not_required_for_policy_free_core_reduction": True,
    },
    "decision_labels": {
        "set_conditioned_main_candidate": "green_materiality_passes",
        "dual_architecture_ablation_required": "yellow_materiality",
        "prefer_independent_gate": "independent_simplification_applies",
        "online_greedy_sufficient": "greedy_search_sufficiency_green",
        "search_gap_yellow": "greedy_search_sufficiency_intermediate",
        "search_gap_requires_strong_diagnosis": (
            "greedy_search_strong_diagnosis"
        ),
    },
}

EXPECTED_OPERATION_CEILING = {
    "policy_free_reducer_runs_unbounded_but_deterministic": True,
    "restoration_teacher_forward_count": 0,
    "kl_measurement_count": 0,
    "generation_count": 0,
    "gate_training_example_count": 0,
    "gate_model_forward_count": 0,
    "matched_nll_evaluation_count": 0,
    "closed_loop_episode_count": 0,
    "confirm_state_access_count": 0,
    "sealed_test_state_access_count": 0,
    "policy_vision_feature_forward_count": 0,
}

EXPECTED_FUTURE_POLICY_VISION_STAGE = {
    "input_scope": "label_train_and_development_n_equals_4_B_equals_2_images",
    "expected_unique_image_upper_bound": 75,
    "cache_must_bind_image_sha_model_revision_extractor_source_and_tensor_schema": True,
    "feature_only_no_language_model_or_generation": True,
    "must_not_read_confirm_or_test": True,
}

EXPECTED_OUTPUT_CONTRACT = {
    "canonical_result_directory": (
        "data/results/restoration_v2_2_selector_geometry_v1"
    ),
    "raw_labels_or_images_copied_to_git": False,
    "result_must_bind_input_revisions_and_contract_sha256": True,
    "result_status_before_execution": "not_generated",
}


@dataclass(frozen=True)
class RestorationV22SelectorGeometryContract:
    path: Path
    data: Mapping[str, Any]
    sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "RestorationV22SelectorGeometryContract":
        root = Path(repository_root).resolve()
        resolved = Path(path).resolve()
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file() or resolved.is_symlink():
            raise ValueError("selector-geometry contract must use the canonical config")
        digest = sha256_file(resolved)
        _equal(digest, FROZEN_CONFIG_SHA256, "selector-geometry config SHA256")
        contract = cls(
            path=resolved,
            data=load_strict_json_object(resolved),
            sha256=digest,
        )
        contract.validate(repository_root=root)
        return contract

    def validate(self, *, repository_root: str | Path) -> None:
        root = Path(repository_root).resolve()
        _equal(self.sha256, FROZEN_CONFIG_SHA256, "selector-geometry config SHA256")
        data = _mapping(self.data, "contract")
        _exact_keys(
            data,
            {
                "schema_version",
                "protocol_id",
                "preregistration_status",
                "authorization",
                "immutable_inputs",
                "analysis_denominator",
                "normalization",
                "analysis_grid",
                "selectors",
                "primary_metrics",
                "interaction_analysis",
                "uncertainty",
                "internal_method_shaping",
                "operation_ceiling",
                "future_policy_vision_stage",
                "output_contract",
            },
            "contract",
        )
        _equal(data.get("schema_version"), "0.1.0", "schema version")
        _equal(data.get("protocol_id"), PROTOCOL_ID, "protocol id")
        _equal(
            data.get("preregistration_status"),
            "source_only_frozen_before_selector_geometry_reduction_or_policy_vision_feature_extraction",
            "preregistration status",
        )
        _equal(data.get("authorization"), EXPECTED_AUTHORIZATION, "authorization")
        _equal(
            data.get("analysis_denominator"),
            EXPECTED_ANALYSIS_DENOMINATOR,
            "analysis denominator",
        )
        _equal(data.get("normalization"), EXPECTED_NORMALIZATION, "normalization")
        _equal(data.get("analysis_grid"), EXPECTED_ANALYSIS_GRID, "analysis grid")
        _equal(data.get("selectors"), EXPECTED_SELECTORS, "selectors")
        _equal(data.get("primary_metrics"), EXPECTED_PRIMARY_METRICS, "primary metrics")
        _equal(
            data.get("interaction_analysis"),
            EXPECTED_INTERACTION_ANALYSIS,
            "interaction analysis",
        )
        _equal(data.get("uncertainty"), EXPECTED_UNCERTAINTY, "uncertainty")
        _equal(
            data.get("internal_method_shaping"),
            EXPECTED_INTERNAL_METHOD_SHAPING,
            "internal method shaping",
        )
        _equal(
            data.get("operation_ceiling"),
            EXPECTED_OPERATION_CEILING,
            "operation ceiling",
        )
        _equal(
            data.get("future_policy_vision_stage"),
            EXPECTED_FUTURE_POLICY_VISION_STAGE,
            "future policy-vision stage",
        )
        _equal(
            data.get("output_contract"),
            EXPECTED_OUTPUT_CONTRACT,
            "output contract",
        )
        self._validate_immutable_inputs(root)
        self._validate_internal_invariants()

    def _validate_immutable_inputs(self, root: Path) -> None:
        inputs = _mapping(self.data.get("immutable_inputs"), "immutable inputs")
        _exact_keys(
            inputs,
            {"restoration_labels", "derived_dataset", "visual_baseline_identity"},
            "immutable inputs",
        )
        labels = _mapping(inputs.get("restoration_labels"), "restoration labels")
        expected_labels = {
            "binding_path": LABEL_BINDING_PATH,
            "binding_sha256": LABEL_BINDING_SHA256,
            "summary_path": LABEL_SUMMARY_PATH,
            "summary_sha256": LABEL_SUMMARY_SHA256,
            "attempt_id": "restoration-v2-2-eager-labels-v2",
            "attempt_revision": "v2_preclaim_repair",
            "required_status": "VERIFIED_RESTORATION_V2_2_EAGER_LABEL_ARTIFACT",
            "required_outcome": "PASS_RESTORATION_V2_2_EAGER_LABELS_V2",
            "hf_repo": "gavinlaw/causalcache-restoration-labels-mobile",
            "hf_revision": "8f6baae5c0b23b08915fa1b0fb848dd519b4c8db",
            "hf_tag": "v2.2-eager-train-dev-exact-v2",
            "hf_path": "raw/v2.2-eager-train-dev-exact-v2.tar",
            "raw_archive_sha256": (
                "99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e"
            ),
            "raw_archive_size_bytes": 3_747_840,
            "raw_archive_file_count": 101,
            "raw_tree_inventory_sha256": (
                "c5104594b0741810f3d49d007a63a74f16ee4236dd137d1dea92b9373065c45f"
            ),
            "source_git_commit": "5ae40d4aed4eb20b931216776b379bc6ae55629d",
            "run_contract_sha256": (
                "b78ca1e70652c7eef68efc3472adf78cec4510e507a0c00cfcee2c2cf05285d9"
            ),
        }
        _equal(labels, expected_labels, "restoration-label input identity")
        label_artifact = _bound_json(
            root,
            relative_path=labels["binding_path"],
            expected_sha256=labels["binding_sha256"],
            label="restoration-label binding",
        )
        label_summary = _bound_json(
            root,
            relative_path=labels["summary_path"],
            expected_sha256=labels["summary_sha256"],
            label="restoration-label summary",
        )
        self._validate_label_artifact(label_artifact, label_summary, labels)

        derived = _mapping(inputs.get("derived_dataset"), "derived dataset")
        expected_derived = {
            "binding_path": DERIVED_BINDING_PATH,
            "binding_sha256": DERIVED_BINDING_SHA256,
            "required_status": (
                "hf_immutable_verified_two_materializations_and_three_exact_replays_passed"
            ),
            "hf_repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
            "hf_revision": "89f136abaff797e14fe758a198996e51032a10a6",
            "hf_tag": "restoration-v2-derived-v1.0.0",
            "artifact_tree_sha256": (
                "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e"
            ),
            "trajectory_count": 35,
            "state_count": 65,
            "event_count": 175,
            "image_member_count": 210,
            "ocr_record_count": 210,
        }
        _equal(derived, expected_derived, "derived-dataset input identity")
        derived_manifest = _bound_json(
            root,
            relative_path=derived["binding_path"],
            expected_sha256=derived["binding_sha256"],
            label="derived-dataset binding",
        )
        self._validate_derived_manifest(derived_manifest, derived)

        baseline = _mapping(
            inputs.get("visual_baseline_identity"), "visual baseline identity"
        )
        expected_baseline = {
            "binding_path": BASELINE_BINDING_PATH,
            "binding_sha256": BASELINE_BINDING_SHA256,
            "artifact_id": "causalcache-restoration-v2-baselines-v1",
            "model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
            "model_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
            "snapshot_manifest_sha256": (
                "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3"
            ),
        }
        _equal(baseline, expected_baseline, "visual-baseline input identity")
        baseline_manifest = _bound_json(
            root,
            relative_path=baseline["binding_path"],
            expected_sha256=baseline["binding_sha256"],
            label="visual-baseline binding",
        )
        self._validate_baseline_manifest(baseline_manifest, baseline)

    @staticmethod
    def _validate_label_artifact(
        artifact: Mapping[str, Any],
        summary: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> None:
        result = _mapping(artifact.get("result"), "label artifact result")
        execution = _mapping(
            artifact.get("source_execution"), "label source execution"
        )
        hf = _mapping(artifact.get("hf_artifact"), "label HF artifact")
        raw = _mapping(artifact.get("raw_archive"), "label raw archive")
        observed = {
            "attempt_id": artifact.get("attempt_id"),
            "attempt_revision": artifact.get("attempt_revision"),
            "required_status": artifact.get("status"),
            "required_outcome": result.get("outcome"),
            "hf_repo": hf.get("repo"),
            "hf_revision": hf.get("immutable_revision"),
            "hf_tag": hf.get("tag"),
            "hf_path": hf.get("path"),
            "raw_archive_sha256": raw.get("sha256"),
            "raw_archive_size_bytes": raw.get("size_bytes"),
            "raw_archive_file_count": raw.get("file_count"),
            "raw_tree_inventory_sha256": raw.get("tree_inventory_sha256"),
            "source_git_commit": execution.get("source_git_commit"),
            "run_contract_sha256": execution.get("run_contract_sha256"),
        }
        _equal(
            observed,
            {key: expected[key] for key in observed},
            "restoration-label artifact content",
        )
        _equal(result.get("fixed_state_denominator"), 45, "label state count")
        _equal(result.get("raw_distance_row_count"), 420, "label distance rows")
        overall = _mapping(summary.get("overall"), "label summary overall")
        _equal(summary.get("attempt_id"), expected["attempt_id"], "summary attempt")
        _equal(summary.get("outcome"), expected["required_outcome"], "summary outcome")
        _equal(overall.get("state_count"), 45, "summary state count")
        _equal(
            overall.get("candidate_event_count_histogram"),
            {"2": 15, "3": 15, "4": 15},
            "summary event-count histogram",
        )

    @staticmethod
    def _validate_derived_manifest(
        manifest: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> None:
        artifact = _mapping(
            manifest.get("hf_dataset_artifact"), "derived HF artifact"
        )
        counts = _mapping(artifact.get("counts"), "derived counts")
        observed = {
            "required_status": manifest.get("status"),
            "hf_repo": artifact.get("repo"),
            "hf_revision": artifact.get("immutable_revision"),
            "hf_tag": artifact.get("tag"),
            "artifact_tree_sha256": artifact.get("artifact_tree_sha256"),
            "trajectory_count": counts.get("trajectory_count"),
            "state_count": counts.get("state_count"),
            "event_count": counts.get("event_count"),
            "image_member_count": counts.get("image_member_count"),
            "ocr_record_count": counts.get("ocr_record_count"),
        }
        _equal(
            observed,
            {key: expected[key] for key in observed},
            "derived-dataset artifact content",
        )

    @staticmethod
    def _validate_baseline_manifest(
        manifest: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> None:
        implementation = _mapping(
            manifest.get("implementation_contract"), "baseline implementation"
        )
        vision = _mapping(implementation.get("policy_vision"), "policy vision")
        observed = {
            "artifact_id": manifest.get("artifact_id"),
            "model_repo": vision.get("model_repo"),
            "model_revision": vision.get("model_revision"),
            "snapshot_manifest_sha256": vision.get("snapshot_manifest_sha256"),
        }
        _equal(
            observed,
            {key: expected[key] for key in observed},
            "visual-baseline artifact content",
        )
        _equal(manifest.get("dependency_6_closed"), True, "baseline dependency")
        _equal(
            implementation.get("candidate_event_step_ids"),
            [1, 2, 3, 4],
            "frozen visual baseline candidate scope",
        )
        _equal(
            implementation.get("primary_budget_event_capacity"),
            2,
            "frozen visual baseline budget",
        )

    def _validate_internal_invariants(self) -> None:
        denominator = _mapping(
            self.data.get("analysis_denominator"), "analysis denominator"
        )
        if denominator["states_per_trajectory"] * denominator["fixed_trajectory_count"] != denominator["fixed_state_count"]:
            raise ValueError("trajectory/state denominator is not internally consistent")
        if sum(denominator["role_state_counts"].values()) != EXPECTED_STATE_COUNT:
            raise ValueError("role state counts do not cover the denominator")
        if sum(denominator["role_trajectory_counts"].values()) != EXPECTED_TRAJECTORY_COUNT:
            raise ValueError("role trajectory counts do not cover the denominator")
        if sum(denominator["candidate_event_count_histogram"].values()) != EXPECTED_STATE_COUNT:
            raise ValueError("candidate-event histogram does not cover the denominator")
        grid = _mapping(self.data.get("analysis_grid"), "analysis grid")
        primary = _mapping(grid.get("primary_setting"), "primary setting")
        _equal(primary["candidate_event_count"], PRIMARY_EVENT_COUNT, "primary n")
        _equal(primary["budget_event_capacity"], PRIMARY_BUDGET, "primary B")
        _equal(
            primary["decision_step_id"] - 2,
            primary["candidate_event_count"],
            "primary decision-step/event-count identity",
        )
        uncertainty = _mapping(self.data.get("uncertainty"), "uncertainty")
        _equal(uncertainty["replicate_count"], BOOTSTRAP_REPLICATES, "bootstrap N")
        _equal(uncertainty["seed"], BOOTSTRAP_SEED, "bootstrap seed")
        _equal(
            uncertainty["confidence_level"],
            BOOTSTRAP_CONFIDENCE,
            "bootstrap confidence",
        )
        quantiles = _sequence(uncertainty["quantiles"], "bootstrap quantiles")
        if len(quantiles) != 2 or not math.isclose(
            float(quantiles[0]) + float(quantiles[1]), 1.0
        ):
            raise ValueError("bootstrap percentile endpoints are inconsistent")
        operation = _mapping(self.data.get("operation_ceiling"), "operation ceiling")
        prohibited_counts = [
            value
            for key, value in operation.items()
            if key != "policy_free_reducer_runs_unbounded_but_deterministic"
        ]
        if any(type(value) is not int or value != 0 for value in prohibited_counts):
            raise ValueError("prohibited operation count must remain zero")
        labels = _mapping(
            _mapping(self.data.get("immutable_inputs"), "immutable inputs").get(
                "restoration_labels"
            ),
            "restoration labels",
        )
        for key in ("source_git_commit", "hf_revision"):
            if _GIT_SHA.fullmatch(str(labels[key])) is None:
                raise ValueError(f"restoration label {key} is not a full revision")


def validate_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    contract = RestorationV22SelectorGeometryContract.load(
        path,
        repository_root=repository_root,
    )
    return {
        "status": PASS_STATUS,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": contract.sha256,
        "fixed_state_count": EXPECTED_STATE_COUNT,
        "fixed_trajectory_count": EXPECTED_TRAJECTORY_COUNT,
        "primary_candidate_event_count": PRIMARY_EVENT_COUNT,
        "primary_budget_event_capacity": PRIMARY_BUDGET,
        "budget_curve": "B=0..n",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirm_locked": True,
        "sealed_test_locked": True,
        "gate_training_locked": True,
        "policy_vision_requires_separate_feature_only_source_freeze": True,
    }
