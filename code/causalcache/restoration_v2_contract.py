"""Fail-closed validation for the restoration-v2 scientific contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FROZEN_RESTORATION_V2_SHA256 = "9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must equal {expected!r}; got {actual!r}")


def _true(value: Any, name: str) -> None:
    if value is not True:
        raise ValueError(f"{name} must be true")


def _false(value: Any, name: str) -> None:
    if value is not False:
        raise ValueError(f"{name} must be false")


def _unique_strings(value: Any, name: str) -> tuple[str, ...]:
    items = tuple(str(item) for item in _list(value, name))
    if not items or len(items) != len(set(items)):
        raise ValueError(f"{name} must contain unique non-empty values")
    return items


def _rate(value: Any, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _validate_reference(data: Mapping[str, Any]) -> None:
    reference = _mapping(data["reference_definition"], "reference_definition")
    _equal(reference["mode"], "stable_full_history_self_behavior", "reference_definition.mode")
    _false(reference["expert_top1_match_required"], "reference_definition.expert_top1_match_required")
    _true(reference["summary_present_for_every_history_event"], "reference_definition.summary_present_for_every_history_event")
    _equal(reference["confirm_decision_step_id"], 6, "reference_definition.confirm_decision_step_id")

    history = tuple(_list(reference["history_event_step_ids"], "reference_definition.history_event_step_ids"))
    high_fidelity = tuple(
        _list(
            reference["reference_high_fidelity_event_step_ids"],
            "reference_definition.reference_high_fidelity_event_step_ids",
        )
    )
    current_equivalent = reference["current_equivalent_event_step_id"]
    _equal(
        reference["observation_identity"],
        "artifact_image_member_path_and_sha256",
        "reference_definition.observation_identity",
    )
    _equal(history, (1, 2, 3, 4, 5), "reference_definition.history_event_step_ids")
    if set(high_fidelity) != set(history) - {current_equivalent}:
        raise ValueError("reference high-fidelity events must cover every non-current distinct post-state")
    _equal(high_fidelity, (1, 2, 3, 4), "reference_definition.reference_high_fidelity_event_step_ids")
    _equal(reference["reference_history_images"], len(high_fidelity), "reference_definition.reference_history_images")
    _equal(reference["current_observation_images"], 1, "reference_definition.current_observation_images")
    _equal(reference["total_reference_images"], len(history), "reference_definition.total_reference_images")
    _true(
        reference["reference_contains_every_distinct_post_state_image"],
        "reference_definition.reference_contains_every_distinct_post_state_image",
    )
    _equal(
        reference["canonical_action_path"],
        "native_envelope_with_fixed_nonsemantic_action_carrier_and_normalized_mobile_use_tool_call",
        "reference_definition.canonical_action_path",
    )
    _equal(
        reference["teacher_forced_action_carrier"],
        "Action: Execute the selected mobile action.\n",
        "reference_definition.teacher_forced_action_carrier",
    )
    _equal(
        reference["teacher_forced_context"],
        "processor_native_assistant_prefix_plus_fixed_action_carrier",
        "reference_definition.teacher_forced_context",
    )
    _equal(
        reference["canonical_tool_call_serialization"],
        {
            "opening": "<tool_call>\n",
            "top_level_key_order": ["name", "arguments"],
            "name": "mobile_use",
            "argument_key_order": ["action", "coordinate", "coordinate2", "text", "button", "status"],
            "omit_absent_arguments": True,
            "text_normalization": "Unicode_NFKC",
            "coordinate_number_format": "base10_integer",
            "json": "ensure_ascii_false_separators_comma_colon",
            "closing": "\n</tool_call>",
        },
        "reference_definition.canonical_tool_call_serialization",
    )
    _equal(
        reference["distance_token_span"],
        "first_token_of_<tool_call>_through_final_token_of_</tool_call>_inclusive",
        "reference_definition.distance_token_span",
    )
    _equal(
        reference["distance"],
        "teacher_forced_full_vocabulary_mean_kl_on_distance_token_span",
        "reference_definition.distance",
    )
    _true(reference["parse_required"], "reference_definition.parse_required")
    _true(reference["finite_logits_required"], "reference_definition.finite_logits_required")
    _equal(reference["repeat_forwards"], 2, "reference_definition.repeat_forwards")
    _true(
        reference["repeat_canonical_action_match_required"],
        "reference_definition.repeat_canonical_action_match_required",
    )
    _equal(
        reference["repeat_noise_epsilon"],
        "max(1e-4,10*mean_repeat_kl)",
        "reference_definition.repeat_noise_epsilon",
    )


def _validate_action_contract(data: Mapping[str, Any]) -> None:
    action = _mapping(data["action_contract"], "action_contract")
    canonical = (
        "click",
        "long_press",
        "swipe",
        "type",
        "system_button",
        "open",
        "wait",
        "answer",
        "terminate",
    )
    _equal(tuple(action["canonical_prompt_actions"]), canonical, "action_contract.canonical_prompt_actions")
    _equal(tuple(action["system_buttons"]), ("Back", "Home", "Enter"), "action_contract.system_buttons")
    _equal(action["accepted_model_aliases"], {"tap": "click", "open_app": "open"}, "action_contract.accepted_model_aliases")
    _equal(
        set(action["removed_upstream_actions"]),
        {"key", "system_button:Menu"},
        "action_contract.removed_upstream_actions",
    )
    _true(
        action["prompt_parser_bridge_effective_executor_inventory_equal"],
        "action_contract.prompt_parser_bridge_effective_executor_inventory_equal",
    )
    _equal(
        action["inventory_equality_scope"],
        "normalized_policy_visible_effective_inventory_after_bridge",
        "action_contract.inventory_equality_scope",
    )
    _false(action["unsupported_prompt_actions_allowed"], "action_contract.unsupported_prompt_actions_allowed")

    coordinates = _mapping(action["coordinate_contract"], "action_contract.coordinate_contract")
    _equal(coordinates["normalized_minimum"], 0, "action_contract.coordinate_contract.normalized_minimum")
    _equal(coordinates["normalized_maximum"], 999, "action_contract.coordinate_contract.normalized_maximum")
    _equal(
        coordinates["pixel_scaling"],
        "floor(value*(extent-1)/999+0.5)",
        "action_contract.coordinate_contract.pixel_scaling",
    )
    _true(
        coordinates["pixel_output_must_be_strictly_inside_extent"],
        "action_contract.coordinate_contract.pixel_output_must_be_strictly_inside_extent",
    )

    parameters = _mapping(action["parameter_contract"], "action_contract.parameter_contract")
    _equal(set(parameters), set(canonical), "action_contract.parameter_contract keys")
    _equal(
        parameters,
        {
            "click": ["coordinate"],
            "long_press": ["coordinate"],
            "swipe": ["coordinate", "coordinate2"],
            "type": ["text"],
            "system_button": ["button"],
            "open": ["text"],
            "wait": [],
            "answer": ["text"],
            "terminate": ["status=success"],
        },
        "action_contract.parameter_contract",
    )
    _true(action["exact_native_tool_call_preserved_in_archive"], "action_contract.exact_native_tool_call_preserved_in_archive")
    _true(
        action["exact_androidworld_payload_preserved_in_archive"],
        "action_contract.exact_androidworld_payload_preserved_in_archive",
    )
    _true(action["exhaustive_round_trip_fixture_required"], "action_contract.exhaustive_round_trip_fixture_required")
    _equal(action["exhaustive_round_trip_pass_rate"], 1.0, "action_contract.exhaustive_round_trip_pass_rate")
    _rate(action["empirical_development_parse_coverage_minimum"], "action_contract.empirical_development_parse_coverage_minimum")


def _validate_fidelity_and_budget(data: Mapping[str, Any]) -> None:
    low = _mapping(data["low_fidelity_event"], "low_fidelity_event")
    expected_keys = (
        "step_id",
        "action_type",
        "action_argument",
        "foreground_app",
        "screen_text_added",
        "screen_text_removed",
        "screen_change",
        "executor_result",
    )
    _equal(tuple(low["serialized_key_order"]), expected_keys, "low_fidelity_event.serialized_key_order")
    _equal(
        low["serialization"],
        "compact_utf8_json_ensure_ascii_false_separators_comma_colon_then_newline",
        "low_fidelity_event.serialization",
    )
    _equal(low["missing_scalar_value"], "unknown", "low_fidelity_event.missing_scalar_value")
    _true(low["present_for_every_event_in_every_memory"], "low_fidelity_event.present_for_every_event_in_every_memory")
    _true(low["byte_identical_between_low_and_high_fidelity"], "low_fidelity_event.byte_identical_between_low_and_high_fidelity")
    _true(low["report_policy_visible_text_tokens"], "low_fidelity_event.report_policy_visible_text_tokens")
    _equal(
        low["action_argument"],
        {
            "spatial_action": "10x10_coordinate_bin",
            "swipe": "viewport_direction_and_displacement_bin",
            "text_open_answer": "exact_nfkc_text",
            "system_button": "exact_button",
            "wait": "wait",
        },
        "low_fidelity_event.action_argument",
    )
    text_delta = _mapping(low["screen_text_delta"], "low_fidelity_event.screen_text_delta")
    _equal(text_delta["source"], "accessibility_tree_else_pinned_ocr", "low_fidelity_event.screen_text_delta.source")
    _equal(
        text_delta["normalization"],
        "Unicode_NFKC_collapse_whitespace_strip",
        "low_fidelity_event.screen_text_delta.normalization",
    )
    _equal(
        text_delta["ordering"],
        "top_to_bottom_then_left_to_right",
        "low_fidelity_event.screen_text_delta.ordering",
    )
    if int(text_delta["maximum_added_tokens"]) <= 0 or int(text_delta["maximum_removed_tokens"]) <= 0:
        raise ValueError("screen text delta token limits must be positive")
    _equal(text_delta["maximum_added_tokens"], 32, "low_fidelity_event.screen_text_delta.maximum_added_tokens")
    _equal(text_delta["maximum_removed_tokens"], 32, "low_fidelity_event.screen_text_delta.maximum_removed_tokens")
    _equal(
        text_delta["truncation"],
        "keep_first_tokens_in_frozen_spatial_order_and_record_discarded_count_in_artifact_metadata",
        "low_fidelity_event.screen_text_delta.truncation",
    )
    _true(
        text_delta["backend_revision_and_model_hash_required_in_execution_config"],
        "low_fidelity_event.screen_text_delta.backend_revision_and_model_hash_required_in_execution_config",
    )
    _equal(
        low["foreground_app"],
        {
            "source": "source_app_label_else_executor_package_name_else_unknown",
            "normalization": "Unicode_NFKC_collapse_whitespace_strip_casefold",
        },
        "low_fidelity_event.foreground_app",
    )
    _equal(
        low["screen_change"],
        {
            "metric": "mean_absolute_rgb_difference_after_256x256_bilinear_resize_in_[0,1]",
            "bins": {
                "none": "value<=0.005",
                "low": "0.005<value<=0.05",
                "medium": "0.05<value<=0.20",
                "high": "value>0.20",
            },
            "library_version_and_resampling_identity_required_in_execution_config": True,
        },
        "low_fidelity_event.screen_change",
    )
    _equal(low["executor_result_values"], ["accepted", "failed", "unknown"], "low_fidelity_event.executor_result_values")
    _equal(
        low["executor_result_provenance"],
        "accepted_or_failed_only_from_executor_record_else_unknown_never_inferred_from_pixels",
        "low_fidelity_event.executor_result_provenance",
    )
    _true(low["exact_text_argument_is_not_truncated"], "low_fidelity_event.exact_text_argument_is_not_truncated")
    _equal(low["context_overflow_outcome"], "INVALID_BEFORE_POLICY_FORWARD", "low_fidelity_event.context_overflow_outcome")

    high = _mapping(data["high_fidelity_event"], "high_fidelity_event")
    _equal(high["images_per_event"], 1, "high_fidelity_event.images_per_event")
    _equal(high["image"], "post_action_state", "high_fidelity_event.image")
    _false(high["include_before_image"], "high_fidelity_event.include_before_image")
    _false(high["include_additional_action_text"], "high_fidelity_event.include_additional_action_text")
    _true(high["adds_only_post_state_image_to_shared_summary"], "high_fidelity_event.adds_only_post_state_image_to_shared_summary")
    _true(
        high["latest_event_whose_post_state_is_current_is_not_a_candidate"],
        "high_fidelity_event.latest_event_whose_post_state_is_current_is_not_a_candidate",
    )
    _true(
        high["candidate_image_must_not_equal_current_observation_path"],
        "high_fidelity_event.candidate_image_must_not_equal_current_observation_path",
    )
    _true(high["persistent_archive_retains_before_and_after_images"], "high_fidelity_event.persistent_archive_retains_before_and_after_images")

    budget = _mapping(data["memory_budget"], "memory_budget")
    _equal(budget["cost_unit"], "effective_visual_tokens", "memory_budget.cost_unit")
    _equal(budget["budget_semantics"], "maximum_cap_not_required_consumption", "memory_budget.budget_semantics")
    _equal(budget["candidate_event_count_at_confirm_state"], 4, "memory_budget.candidate_event_count_at_confirm_state")
    _equal(budget["reference_high_fidelity_event_count"], 4, "memory_budget.reference_high_fidelity_event_count")
    _equal(budget["primary_selected_event_capacity"], 2, "memory_budget.primary_selected_event_capacity")
    _equal(budget["secondary_selected_event_capacities"], [1, 3], "memory_budget.secondary_selected_event_capacities")
    _true(budget["candidate_costs_must_be_equal_within_state"], "memory_budget.candidate_costs_must_be_equal_within_state")
    _true(budget["same_summary_and_visual_budget_for_every_selector"], "memory_budget.same_summary_and_visual_budget_for_every_selector")
    _true(budget["report_actual_visual_tokens"], "memory_budget.report_actual_visual_tokens")
    _true(budget["cardinality_matched_ablation_required"], "memory_budget.cardinality_matched_ablation_required")


def _validate_data_roles(data: Mapping[str, Any]) -> None:
    dataset = _mapping(data["data"], "data")
    parent = _mapping(dataset["parent_artifact"], "data.parent_artifact")
    _equal(parent["repo"], "gavinlaw/causalcache-guiodyssey-independent-mobile", "data.parent_artifact.repo")
    _equal(parent["revision"], "84c9f5a335e9612ccb4bd566f977574f359b2485", "data.parent_artifact.revision")
    _equal(
        parent["manifest_sha256"],
        "3870900442dd0c8f037c9127e0c61c53c9d08c3735164c3c57c857f2c65eb949",
        "data.parent_artifact.manifest_sha256",
    )
    _equal(
        parent["eligible_pool_sha256"],
        "84d6855b20227d7f884d4ce26cfae7188c1075bb7751cab78a3cf325eef192cb",
        "data.parent_artifact.eligible_pool_sha256",
    )
    derived = _mapping(dataset["derived_artifact"], "data.derived_artifact")
    _equal(
        derived["repo"],
        "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
        "data.derived_artifact.repo",
    )
    _equal(derived["visibility"], "private", "data.derived_artifact.visibility")
    _true(derived["immutable_revision_required_before_policy_output"], "data.derived_artifact.immutable_revision_required_before_policy_output")

    roles = _mapping(dataset["roles"], "data.roles")
    reference = _mapping(roles["v1_reference_contract_audit_only"], "data.roles.v1_reference_contract_audit_only")
    train = _mapping(roles["v2_label_train"], "data.roles.v2_label_train")
    development = _mapping(roles["v2_development"], "data.roles.v2_development")
    confirm = _mapping(roles["v2_confirm_primary"], "data.roles.v2_confirm_primary")
    role_ids: list[tuple[str, tuple[str, ...], int]] = [
        ("v1_reference_contract_audit_only", _unique_strings(reference["source_ids"], "v1 reference source_ids"), 8),
        ("v2_label_train", _unique_strings(train["source_ids"], "v2 label-train source_ids"), 10),
        ("v2_development", _unique_strings(development["source_ids"], "v2 development source_ids"), 5),
    ]
    all_ids: list[str] = []
    for name, source_ids, expected_count in role_ids:
        _equal(len(source_ids), expected_count, f"data.roles.{name}.source_ids count")
        _equal(roles[name]["trajectory_count"], expected_count, f"data.roles.{name}.trajectory_count")
        all_ids.extend(source_ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("v1 reference, v2 label-train, and v2 development trajectories must be disjoint")
    _true(reference["policy_output_seen"], "data.roles.v1_reference_contract_audit_only.policy_output_seen")
    _false(reference["may_train_or_confirm"], "data.roles.v1_reference_contract_audit_only.may_train_or_confirm")
    _false(train["policy_output_seen"], "data.roles.v2_label_train.policy_output_seen")
    _false(development["policy_output_seen"], "data.roles.v2_development.policy_output_seen")
    expected_state_selection = "all_available_requested_step_ids_in_source_order_without_top_up"
    _equal(train["state_selection"], expected_state_selection, "data.roles.v2_label_train.state_selection")
    _equal(development["state_selection"], expected_state_selection, "data.roles.v2_development.state_selection")

    _equal(confirm["trajectory_order"], "continue_frozen_v1_sha256_order_without_new_salt", "data.roles.v2_confirm_primary.trajectory_order")
    _equal(confirm["trajectory_count"], 20, "data.roles.v2_confirm_primary.trajectory_count")
    _equal(confirm["decision_step_id"], 6, "data.roles.v2_confirm_primary.decision_step_id")
    _equal(confirm["states_per_trajectory"], 1, "data.roles.v2_confirm_primary.states_per_trajectory")
    _equal(confirm["state_count"], 20, "data.roles.v2_confirm_primary.state_count")
    _equal(
        confirm["insufficient_prefix_diversity_outcome"],
        "INVALID_BEFORE_POLICY_OUTPUT",
        "data.roles.v2_confirm_primary.insufficient_prefix_diversity_outcome",
    )
    _false(confirm["policy_output_seen"], "data.roles.v2_confirm_primary.policy_output_seen")
    _false(confirm["restoration_output_seen"], "data.roles.v2_confirm_primary.restoration_output_seen")
    _true(confirm["exact_ids_manifest_required_before_policy_output"], "data.roles.v2_confirm_primary.exact_ids_manifest_required_before_policy_output")
    _false(confirm["top_up_after_policy_output_allowed"], "data.roles.v2_confirm_primary.top_up_after_policy_output_allowed")
    _false(
        confirm["filter_by_parse_stability_quality_or_sensitivity_allowed"],
        "data.roles.v2_confirm_primary.filter_by_parse_stability_quality_or_sensitivity_allowed",
    )
    _true(dataset["exposure_ledger_required"], "data.exposure_ledger_required")
    _true(dataset["raw_data_seen_is_distinct_from_policy_output_seen"], "data.raw_data_seen_is_distinct_from_policy_output_seen")


def _validate_gates_and_compute(data: Mapping[str, Any]) -> None:
    quality = _mapping(data["quality_reporting"], "quality_reporting")
    _false(quality["may_filter_label_or_confirm_states"], "quality_reporting.may_filter_label_or_confirm_states")
    _true(quality["semantic_match_requires_preoutput_ui_target_annotation"], "quality_reporting.semantic_match_requires_preoutput_ui_target_annotation")
    _equal(
        quality["rollout_outcome_axis"],
        ["successful_self_rollout", "failed_self_rollout", "not_available"],
        "quality_reporting.rollout_outcome_axis",
    )
    _equal(
        quality["source_trajectory_axis"],
        ["expert_success_trajectory", "self_rollout_trajectory", "not_available"],
        "quality_reporting.source_trajectory_axis",
    )
    required_diagnostics = {
        "action_type_match",
        "canonical_bin_match",
        "normalized_coordinate_distance",
        "text_argument_match",
    }
    _equal(set(quality["always_report"]), required_diagnostics, "quality_reporting.always_report")

    substrate = _mapping(data["substrate_gate"], "substrate_gate")
    _equal(substrate["screening_roles"], ["v2_label_train", "v2_development"], "substrate_gate.screening_roles")
    _equal(substrate["minimum_screening_states"], 20, "substrate_gate.minimum_screening_states")
    _equal(_rate(substrate["minimum_parse_coverage"], "substrate_gate.minimum_parse_coverage"), 0.99, "substrate_gate.minimum_parse_coverage")
    _equal(_rate(substrate["minimum_finite_logit_coverage"], "substrate_gate.minimum_finite_logit_coverage"), 1.0, "substrate_gate.minimum_finite_logit_coverage")
    _equal(
        _rate(substrate["minimum_repeat_canonical_action_agreement"], "substrate_gate.minimum_repeat_canonical_action_agreement"),
        1.0,
        "substrate_gate.minimum_repeat_canonical_action_agreement",
    )
    _equal(substrate["minimum_memory_sensitive_states"], 8, "substrate_gate.minimum_memory_sensitive_states")

    gate = _mapping(data["restoration_gate"], "restoration_gate")
    _equal(gate["primary_states"], 20, "restoration_gate.primary_states")
    _equal(gate["primary_budget_event_capacity"], 2, "restoration_gate.primary_budget_event_capacity")
    _equal(gate["confirmatory_permutation_samples"], 16, "restoration_gate.confirmatory_permutation_samples")
    _equal(gate["confirmatory_seed"], 20270715, "restoration_gate.confirmatory_seed")
    _true(gate["run_full_fixed_state_denominator"], "restoration_gate.run_full_fixed_state_denominator")
    _true(gate["parse_or_stability_failure_counts_as_gate_failure"], "restoration_gate.parse_or_stability_failure_counts_as_gate_failure")
    _equal(gate["minimum_memory_sensitive_states"], 8, "restoration_gate.minimum_memory_sensitive_states")
    _equal(
        gate["normalized_recovery_per_state"],
        "(D(empty)-D(selected))/max(D(empty),epsilon)",
        "restoration_gate.normalized_recovery_per_state",
    )
    _equal(gate["mean_recovery_denominator"], "all_20_fixed_confirm_states", "restoration_gate.mean_recovery_denominator")
    _equal(
        gate["oracle_selection"],
        "minimum_distance_subset_with_cardinality_0_1_or_2_of_4_candidates",
        "restoration_gate.oracle_selection",
    )
    _equal(
        gate["oracle_tie_break"],
        "lower_distance_then_lower_cardinality_then_lower_event_step_ids",
        "restoration_gate.oracle_tie_break",
    )
    _equal(
        gate["strongest_baseline_definition"],
        "maximum_dataset_mean_normalized_recovery_across_nonoracle_baselines",
        "restoration_gate.strongest_baseline_definition",
    )
    frozen_rates = {
        "minimum_mean_oracle_recovery": 0.3,
        "minimum_mean_gain_over_strongest_baseline": 0.1,
        "minimum_median_spearman_to_exact": 0.8,
        "minimum_median_top_budget_jaccard_to_exact": 0.75,
        "minimum_median_exact_selector_utility_ratio": 0.9,
    }
    for key, expected in frozen_rates.items():
        _equal(_rate(gate[key], f"restoration_gate.{key}"), expected, f"restoration_gate.{key}")
    _equal(
        gate["stability_metric_contract"],
        {
            "state_denominator": "memory_sensitive_confirm_states",
            "aggregation": "deterministic_median_average_two_middle_values",
            "exact_attribution": "mean_marginal_over_all_singleton_near_budget_coalitions_for_each_candidate",
            "score_tie": "math_isclose_rel_tol_1e-9_abs_tol_1e-12_then_lower_event_step_id",
            "spearman": "average_ranks;both_constant_and_same_selected_set=1;one_constant=0",
            "sampled_and_exact_selection": "positive_gain_knapsack_with_capacity_2_and_lower_event_step_id_tie_break",
            "jaccard": "set_jaccard_empty_empty_equals_1",
            "utility_ratio": "sampled_selection_utility_divided_by_exact_attribution_selection_utility;if_both_abs_le_epsilon_then_1;if_only_denominator_abs_le_epsilon_then_0",
        },
        "restoration_gate.stability_metric_contract",
    )
    bootstrap = _mapping(gate["paired_bootstrap"], "restoration_gate.paired_bootstrap")
    _equal(bootstrap["confidence"], 0.9, "restoration_gate.paired_bootstrap.confidence")
    _equal(bootstrap["resamples"], 10000, "restoration_gate.paired_bootstrap.resamples")
    _equal(bootstrap["seed"], 271828, "restoration_gate.paired_bootstrap.seed")
    _equal(
        bootstrap["resampling_unit"],
        "confirm_state_with_replacement_fixed_20_draws",
        "restoration_gate.paired_bootstrap.resampling_unit",
    )
    _equal(
        bootstrap["interval"],
        "percentile_two_sided_90_percent_lower_fifth_percentile",
        "restoration_gate.paired_bootstrap.interval",
    )
    _true(bootstrap["lower_bound_must_be_positive"], "restoration_gate.paired_bootstrap.lower_bound_must_be_positive")
    _equal(
        bootstrap["comparison_scope"],
        "oracle_minus_each_nonoracle_baseline_all_lower_bounds_must_be_positive",
        "restoration_gate.paired_bootstrap.comparison_scope",
    )
    expected_baselines = [
        "summary_only",
        "recent",
        "uniform_random_exact_expectation",
        "ocr_and_rgb_similarity",
        "frozen_policy_vision_embedding_similarity",
    ]
    _equal(gate["baselines"], expected_baselines, "restoration_gate.baselines")
    _equal(
        gate["baseline_contract"],
        {
            "summary_only": "select_empty",
            "recent": "select_events_3_and_4",
            "uniform_random_exact_expectation": "mean_normalized_recovery_over_all_2_of_4_subsets",
            "ocr_and_rgb_similarity": "score=0.5*set_jaccard_of_whitespace_token_sets_after_frozen_ocr_normalization(post,current)+0.5*cosine_joint_rgb_histogram_16x16x16(post,current);select_top_2",
            "frozen_policy_vision_embedding_similarity": "cosine_of_l2_normalized_mean_last_visual_encoder_output_after_spatial_merger(post,current);select_top_2",
            "tie_break": "higher_score_then_lower_event_step_id",
            "formula_source_hashes_required_before_confirm": True,
        },
        "restoration_gate.baseline_contract",
    )
    _true(
        gate["exact_capacity_2_cardinality_ablation_required"],
        "restoration_gate.exact_capacity_2_cardinality_ablation_required",
    )
    _equal(gate["pass_outcome"], "GO_TO_GATE_TRAINING", "restoration_gate.pass_outcome")
    _equal(gate["hard_no_go_outcome"], "NO_GO_RESTORATION_V2", "restoration_gate.hard_no_go_outcome")
    _equal(
        gate["hard_no_go_if"],
        ["memory_sensitive_states_less_than_4", "mean_oracle_recovery_less_than_or_equal_to_0.10"],
        "restoration_gate.hard_no_go_if",
    )
    _equal(gate["other_valid_outcome"], "INCONCLUSIVE_V2", "restoration_gate.other_valid_outcome")
    _equal(
        gate["parse_or_stability_failure_outcome"],
        "INCONCLUSIVE_V2",
        "restoration_gate.parse_or_stability_failure_outcome",
    )

    compute = _mapping(data["attribution_compute"], "attribution_compute")
    _equal(compute["development_permutation_samples"], 4, "attribution_compute.development_permutation_samples")
    _equal(compute["confirmatory_permutation_samples"], gate["confirmatory_permutation_samples"], "attribution_compute.confirmatory_permutation_samples")
    _equal(compute["stability_audit_permutation_samples"], 32, "attribution_compute.stability_audit_permutation_samples")
    _true(compute["screening_may_stop_confirmatory_run"], "attribution_compute.screening_may_stop_confirmatory_run")
    for key in (
        "reference_log_probs_remain_on_gpu",
        "kl_reduction_on_gpu",
        "only_scalar_distances_move_to_cpu",
        "coalitions_grouped_by_image_count_and_sequence_length",
        "microbatch_size_frozen_in_execution_config_before_confirm",
        "batch_1_equivalence_to_audited_cpu_kl_required",
    ):
        _true(compute[key], f"attribution_compute.{key}")


def validate_restoration_v2_contract(data: Mapping[str, Any]) -> None:
    _equal(data["schema_version"], "0.2.0", "schema_version")
    _equal(data["protocol_id"], "causalcache_restoration_v2", "protocol_id")
    _equal(
        data["preregistration_status"],
        "scientific_contract_frozen_before_any_v2_policy_output",
        "preregistration_status",
    )
    change = _mapping(data["change_control"], "change_control")
    _equal(change["v1_outcome_must_remain"], "NO_GO_CURRENT_REFERENCE_STACK", "change_control.v1_outcome_must_remain")
    _true(change["v2_is_a_new_estimand"], "change_control.v2_is_a_new_estimand")
    _true(change["expert_top1_threshold_removed_not_lowered"], "change_control.expert_top1_threshold_removed_not_lowered")
    _false(
        change["scientific_fields_may_change_after_first_v2_policy_output"],
        "change_control.scientific_fields_may_change_after_first_v2_policy_output",
    )
    _true(
        change["implementation_identity_filled_by_separate_execution_config_before_output"],
        "change_control.implementation_identity_filled_by_separate_execution_config_before_output",
    )

    policy = _mapping(data["primary_policy"], "primary_policy")
    _equal(policy["repo"], "mPLUG/GUI-Owl-1.5-8B-Instruct", "primary_policy.repo")
    _equal(policy["revision"], "06d5faecff74840bab2be2425e9c42667a5d04fc", "primary_policy.revision")
    _true(policy["frozen"], "primary_policy.frozen")
    _equal(policy["dtype"], "bfloat16", "primary_policy.dtype")
    _false(policy["decoding"]["do_sample"], "primary_policy.decoding.do_sample")
    _equal(policy["decoding"]["max_new_tokens"], 256, "primary_policy.decoding.max_new_tokens")
    _equal(policy["mixed_fidelity_prompt_mode"], "single_user_post_state_v2", "primary_policy.mixed_fidelity_prompt_mode")
    _equal(policy["visual_preprocessing"]["mode"], "fixed_token_target", "primary_policy.visual_preprocessing.mode")
    _equal(
        policy["visual_preprocessing"]["target_effective_tokens_per_image"],
        2560,
        "primary_policy.visual_preprocessing.target_effective_tokens_per_image",
    )
    _true(
        policy["visual_preprocessing"]["record_actual_image_grid_and_tokens"],
        "primary_policy.visual_preprocessing.record_actual_image_grid_and_tokens",
    )

    _validate_reference(data)
    _validate_action_contract(data)
    _validate_fidelity_and_budget(data)
    _validate_data_roles(data)
    _validate_gates_and_compute(data)

    androidworld = _mapping(data["androidworld_evaluation"], "androidworld_evaluation")
    _equal(androidworld["validation_role"], "development_only_due_to_prior_gui_owl_outputs", "androidworld_evaluation.validation_role")
    _equal(androidworld["test_templates_and_instances"], "25_templates_75_instances_sealed", "androidworld_evaluation.test_templates_and_instances")
    _true(androidworld["test_plan_generated_only_after_gate_checkpoint_freeze"], "androidworld_evaluation.test_plan_generated_only_after_gate_checkpoint_freeze")

    required_dependencies = {
        "derived_artifact_immutable_hf_revision_and_file_hashes",
        "exact_confirm_trajectory_and_state_ids",
        "exposure_ledger",
        "restricted_prompt_parser_bridge_executor_round_trip_fixture",
        "pinned_ocr_or_accessibility_backend_identity",
        "baseline_specification_and_source_hashes",
        "v2_interface_source_hashes",
        "execution_config_referencing_this_scientific_config_sha256",
    }
    dependencies = set(_unique_strings(data["implementation_dependencies_before_any_v2_policy_output"], "implementation dependencies"))
    _equal(dependencies, required_dependencies, "implementation_dependencies_before_any_v2_policy_output")


@dataclass(frozen=True)
class RestorationV2Contract:
    protocol_id: str
    source_sha256: str
    data: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = FROZEN_RESTORATION_V2_SHA256,
    ) -> "RestorationV2Contract":
        source = Path(path).read_bytes()
        source_sha256 = hashlib.sha256(source).hexdigest()
        if expected_sha256 is not None and source_sha256 != expected_sha256:
            raise ValueError(
                "restoration-v2 scientific contract SHA256 mismatch: "
                f"expected {expected_sha256}, got {source_sha256}"
            )
        parsed = _mapping(json.loads(source), "contract")
        validate_restoration_v2_contract(parsed)
        return cls(
            protocol_id=str(parsed["protocol_id"]),
            source_sha256=source_sha256,
            data=parsed,
        )
