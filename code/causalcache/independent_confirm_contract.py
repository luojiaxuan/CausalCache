"""Fail-closed Source-A contract for independent confirm and closed-loop routing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_independent_confirm_closed_loop_v1"
SOURCE_STATUS = "source_only_frozen_before_confirm_policy_or_restoration_output_access"
VALIDATION_STATUS = "VALID_CAUSALCACHE_INDEPENDENT_CONFIRM_CLOSED_LOOP_SOURCE_A_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_independent_confirm_closed_loop_v1.json"
)
RUNNER_FREEZE_PATH = "code/configs/causalcache_independent_confirm_runner_v1.json"
FROZEN_CONFIG_SHA256 = (
    "33e5c0f856a9074cd5460f0d6f7cdda2108b120d50b208234253ff3136d27b63"
)
CONFIRM_SOURCE_IDS_SHA256 = (
    "c84ba8b9b705abc7ba7d6d7230b868fd4600e3a835b1c763a8aaf1f93feb052e"
)
REQUIRED_EXECUTION_MODULES = (
    "causalcache.independent_confirm_contract",
    "causalcache.independent_confirm_data",
    "causalcache.independent_confirm_evaluation",
    "causalcache.independent_confirm_runner",
    "causalcache.independent_confirm_artifact",
    "causalcache.independent_confirm_execution",
    "causalcache.independent_closed_loop_features",
    "causalcache.independent_closed_loop_evaluation",
    "scripts.run_independent_confirm",
    "scripts.smoke_independent_confirm_gpu_topology",
    "scripts.manage_independent_confirm",
    "scripts.validate_independent_confirm_contract",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "paper_decision",
    "lineage",
    "source_freeze",
    "independent_model",
    "confirm_input",
    "confirm_geometry",
    "access_firewall",
    "selection_contract",
    "restoration_contract",
    "evaluation_contract",
    "closed_loop_contract",
    "matched_nll_contract",
    "output_contract",
    "runtime_contract",
    "authorization",
}


@dataclass(frozen=True)
class IndependentConfirmContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def source(self) -> Mapping[str, Any]:
        return _mapping(self.data["source_freeze"], "source_freeze")

    @property
    def model(self) -> Mapping[str, Any]:
        return _mapping(self.data["independent_model"], "independent_model")

    @property
    def confirm(self) -> Mapping[str, Any]:
        return _mapping(self.data["confirm_geometry"], "confirm_geometry")

    @property
    def destination(self) -> Mapping[str, Any]:
        output = _mapping(self.data["output_contract"], "output_contract")
        return _mapping(output["destination"], "output destination")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        require_runner_absent: bool = True,
    ) -> "IndependentConfirmContract":
        root = Path(repository_root).resolve()
        source = Path(path).resolve()
        expected = (root / CANONICAL_CONFIG_PATH).resolve()
        if source != expected:
            raise ValueError("independent confirm contract path is not canonical")
        payload = _regular_file_bytes(source, label="independent confirm contract")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != FROZEN_CONFIG_SHA256:
            raise ValueError("independent confirm contract bytes drifted")
        value = _strict_json(payload, label="independent confirm contract")
        _validate_contract(value)
        runner_path = root / RUNNER_FREEZE_PATH
        runner = runner_path.resolve()
        if require_runner_absent and (
            runner.exists() or runner_path.is_symlink()
        ):
            raise PermissionError("Execution-B runner freeze exists during Source-A validation")
        return cls(
            data=value,
            sha256=digest,
            repository_root=root,
            source_path=source,
        )


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    fingerprint = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if fingerprint(before) != fingerprint(after) or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(value, label)


def load_runner_freeze_file(path: str | Path) -> Mapping[str, Any]:
    payload = _regular_file_bytes(Path(path), label="Execution-B runner freeze")
    value = _strict_json(payload, label="Execution-B runner freeze")
    if payload != pretty_json_bytes(value):
        raise ValueError("Execution-B runner freeze is not canonical pretty JSON")
    return value


def _validate_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _validate_contract(config: Mapping[str, Any]) -> None:
    if set(config) != _TOP_LEVEL_KEYS:
        raise ValueError("independent confirm top-level key inventory drifted")
    _equal(config.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(config.get("protocol_id"), PROTOCOL_ID, "protocol id")
    _equal(config.get("status"), SOURCE_STATUS, "source status")
    _validate_paper_decision(_mapping(config["paper_decision"], "paper decision"))
    _validate_source(_mapping(config["source_freeze"], "source freeze"))
    _validate_model(_mapping(config["independent_model"], "independent model"))
    _validate_confirm(_mapping(config["confirm_geometry"], "confirm geometry"))
    _validate_firewall(_mapping(config["access_firewall"], "access firewall"))
    _validate_evaluation(_mapping(config["evaluation_contract"], "evaluation"))
    _validate_closed_loop(_mapping(config["closed_loop_contract"], "closed loop"))
    _validate_matched_nll(_mapping(config["matched_nll_contract"], "matched NLL"))
    _validate_runtime(_mapping(config["runtime_contract"], "runtime"))
    authorization = _mapping(config["authorization"], "authorization")
    if any(value is not False for value in authorization.values()):
        raise PermissionError("Source-A cannot authorize execution or data access")


def _validate_paper_decision(value: Mapping[str, Any]) -> None:
    _equal(value.get("primary_method"), "restoration_guided_independent_gate", "primary method")
    _equal(value.get("frozen_negative_result"), "NO_V2_CONDITIONAL_RESCUE", "negative result")
    _equal(value.get("conditional_gate_role"), "negative_ablation_only", "conditional role")
    _equal(value.get("retraining_allowed"), False, "retraining authorization")
    _equal(value.get("fresh16_reuse"), "published_development_aggregate_only", "fresh16 reuse")


def _validate_source(value: Mapping[str, Any]) -> None:
    _equal(value.get("branch"), "main", "source branch")
    _equal(value.get("origin_url"), "https://github.com/luojiaxuan/CausalCache.git", "origin URL")
    paths = tuple(
        _safe_relative(path, "required Source-A path")
        for path in _sequence(value.get("required_source_a_paths"), "required Source-A paths")
    )
    if (
        not paths
        or len(paths) != len(set(paths))
        or CANONICAL_CONFIG_PATH not in paths
        or RUNNER_FREEZE_PATH in paths
    ):
        raise ValueError("Source-A path inventory is empty, duplicated, or contaminated")
    modules = tuple(
        _sequence(value.get("required_execution_modules"), "required execution modules")
    )
    _equal(modules, REQUIRED_EXECUTION_MODULES, "required execution modules")
    expected_module_paths = {
        f"code/{name.replace('.', '/')}.py" for name in REQUIRED_EXECUTION_MODULES
    }
    if not expected_module_paths.issubset(paths):
        raise ValueError("required execution module is absent from Source-A paths")
    freeze = _mapping(value.get("execution_b_runner_freeze"), "runner freeze")
    _equal(freeze.get("path"), RUNNER_FREEZE_PATH, "runner freeze path")
    for key in (
        "must_be_absent_during_source_only_validation",
        "only_allowed_execution_b_source_tree_diff",
        "direct_single_parent_child_of_source_a",
        "bind_source_inventory",
        "bind_loaded_module_inventory",
    ):
        _equal(freeze.get(key), True, f"runner freeze {key}")


def _validate_model(value: Mapping[str, Any]) -> None:
    _equal(value.get("repo"), "gavinlaw/causalcache-gate-v1-formal58-selector-mobile", "model repo")
    _equal(value.get("payload_commit"), "a6c9e7f6dab6bc27794438b5b66da07fd59b2889", "model payload commit")
    _equal(value.get("manifest_commit"), "23f6786075c7bff91f93fd7e8a878e070efb72a9", "model manifest commit")
    manifest = _mapping(value.get("ensemble_manifest"), "ensemble manifest")
    _equal(manifest.get("provenance_sha256"), "238c2e64bdb22ccebd261961200021e74a688f912d530f486b00bbb853a8ea57", "independent provenance")
    checkpoints = _sequence(value.get("checkpoints"), "independent checkpoints")
    if len(checkpoints) != 5:
        raise ValueError("independent model must bind exactly five checkpoints")
    for seed, raw in enumerate(checkpoints):
        item = _mapping(raw, f"checkpoint {seed}")
        _equal(item.get("seed"), seed, f"checkpoint {seed} seed")
        _validate_sha(item.get("sha256"), f"checkpoint {seed} SHA256")
        _validate_sha(item.get("model_state_sha256"), f"checkpoint {seed} state SHA256")
    _equal(value.get("training_or_optimizer_allowed"), False, "optimizer authorization")


def _validate_confirm(value: Mapping[str, Any]) -> None:
    expected = {
        "trajectory_count": 20,
        "state_count": 20,
        "decision_step_id": 6,
        "candidate_event_step_ids": [1, 2, 3, 4],
        "budget_event_capacity": 2,
        "source_ids_sha256": CONFIRM_SOURCE_IDS_SHA256,
        "fixed_denominator": True,
        "retry_count": 0,
        "top_up_count": 0,
        "filter_count": 0,
    }
    for key, expected_value in expected.items():
        _equal(value.get(key), expected_value, f"confirm {key}")
    source_ids = _sequence(value.get("source_ids"), "confirm source ids")
    if (
        len(source_ids) != 20
        or len(set(source_ids)) != 20
        or hashlib.sha256(canonical_json_bytes(list(source_ids))).hexdigest()
        != CONFIRM_SOURCE_IDS_SHA256
    ):
        raise ValueError("confirm source roster or order drifted")


def _validate_firewall(value: Mapping[str, Any]) -> None:
    for key in (
        "source_a_confirm_semantic_decode_count",
        "source_a_policy_generation_count",
        "source_a_restoration_teacher_forward_count",
        "source_a_model_load_count",
    ):
        _equal(value.get(key), 0, f"firewall {key}")
    for key in (
        "all_selector_scores_and_selections_sealed_before_any_reference_generation",
        "payload_commit_must_precede_any_restoration_access",
        "confirm_result_may_not_change_thresholds_or_closed_loop_contract",
    ):
        _equal(value.get(key), True, f"firewall {key}")
    order = tuple(_sequence(value.get("execution_b_order"), "execution order"))
    if order[:3] != (
        "verify_transport",
        "verify_gpu_topology_smoke_receipt",
        "build_features",
    ):
        raise ValueError("GPU topology smoke receipt is not verified before semantic decode")
    if order.index("publish_payload_commit") >= order.index("claim_restoration_access"):
        raise ValueError("restoration access is not ordered after the payload commit")


def _validate_evaluation(value: Mapping[str, Any]) -> None:
    _equal(value.get("primary_aggregation"), "sum_raw_utility_over_all_20_fixed_states", "primary aggregation")
    _equal(value.get("normalization_epsilon"), 1e-12, "normalization epsilon")
    _equal(
        dict(_mapping(value.get("descriptive_normalized_reporting"), "descriptive normalized reporting")),
        {
            "affects_go": False,
            "fixed20_zero_imputed_mean": "utility_over_Dempty_when_Dempty_gt_epsilon_else_zero_then_mean_over_all_20",
            "legacy_eligible_only_mean": "utility_over_Dempty_mean_over_Dempty_gt_epsilon_states_only",
            "report_independent_exact_heuristics_and_five_seeds": True,
            "direct_mean_comparison_to_fresh16_allowed": False,
        },
        "descriptive normalized reporting",
    )
    bootstrap = _mapping(value.get("bootstrap"), "bootstrap")
    expected_bootstrap = {
        "unit": "trajectory",
        "resamples": 10000,
        "seed": 271828,
        "confidence": 0.9,
        "interval": "percentile",
        "quantile": "Hyndman_Fan_type_7",
    }
    _equal(dict(bootstrap), expected_bootstrap, "bootstrap contract")
    gates = _mapping(value.get("go_all"), "confirm GO thresholds")
    expected_gates = {
        "valid_reference_state_count_minimum": 20,
        "memory_sensitive_state_count_minimum": 8,
        "ensemble_raw_utility_over_exact_raw_minimum": 0.85,
        "ensemble_raw_utility_over_baseline_distance_minimum": 0.5,
        "mean_raw_delta_vs_each_heuristic_strictly_greater_than": 0.0,
        "strongest_heuristic_positive_trajectory_count_minimum": 12,
        "strongest_heuristic_paired_bootstrap_lower_strictly_greater_than": 0.0,
        "individual_seed_exact_raw_ratio_minimum": 0.75,
        "individual_seed_pass_count_minimum": 4,
        "seed_exact_raw_ratio_population_std_maximum": 0.1,
    }
    _equal(dict(gates), expected_gates, "confirm GO thresholds")
    _equal(value.get("threshold_or_denominator_change_after_source_a_allowed"), False, "post-freeze threshold changes")


def _validate_closed_loop(value: Mapping[str, Any]) -> None:
    _equal(
        set(value),
        {
            "unlock_semantics",
            "estimand",
            "validated_teacher_claim_allowed",
            "backbone",
            "prompt_and_parser",
            "task_partition_binding",
            "topology",
            "memory",
            "online_feature_binding",
            "primary_arms",
            "secondary_arms",
            "unequal_budget_reference",
            "development",
            "arm_counterbalance",
            "deterministic_random_B2",
            "primary_estimator",
            "primary_bootstrap",
            "development_go_all",
            "sealed_test",
            "validation_partition_reuse_for_primary_evidence_allowed",
            "actual_visual_tokens_latency_horizon_and_failure_classification_required",
        },
        "closed-loop field inventory",
    )
    _equal(value.get("estimand"), "paired_memory_controller_effect_under_a_fixed_weak_backbone", "closed-loop estimand")
    _equal(value.get("validated_teacher_claim_allowed"), False, "teacher claim")
    _equal(value.get("primary_arms"), ["independent_B2", "recent_B2"], "primary arms")
    _equal(
        dict(_mapping(value.get("task_partition_binding"), "task partition binding")),
        {
            "path": "code/configs/androidworld_task_partition.json",
            "sha256": "edb3ba4b03339db7c8580f58052d63813d4456030686493161f288230c787683",
            "sorted_task_types_sha256": "185ae2019706693bd32ecc25ffd0c8f87be87331cae6f7d7e31c91674c962b89",
        },
        "task partition binding",
    )
    topology = _mapping(value.get("topology"), "closed-loop topology")
    _equal(topology.get("policy"), "Hyper_H200", "policy host class")
    _equal(topology.get("emulator"), "Aries_AndroidWorld", "emulator host")
    _equal(topology.get("silent_A6000_policy_fallback_allowed"), False, "A6000 fallback")
    _equal(
        dict(_mapping(value.get("memory"), "closed-loop memory")),
        {
            "budget_event_capacity": 2,
            "current_observation_budget_cost": 0,
            "exclude_immediately_previous_post_state_equivalent_to_current": True,
            "all_older_events_keep_fixed_summary": True,
            "independent_rule": "five_seed_mean_strictly_positive_top2",
            "score_threshold": 0.0,
            "tie_break": "ascending_step_id",
        },
        "closed-loop memory",
    )
    _equal(
        dict(_mapping(value.get("online_feature_binding"), "online feature binding")),
        {
            "feature_contract": "formal_gate_v1_q64_h64_q64_times_h64_g8_200d",
            "gate_preregistration_sha256": "37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b",
            "transition_builder": "causalcache.independent_closed_loop_features.live_history_event_from_transition",
            "feature_state_adapter": "causalcache.independent_closed_loop_features.feature_state_from_live_history",
            "feature_function": "causalcache.independent_closed_loop_features.independent_input_from_feature_state",
            "history_order": "oldest_to_newest",
            "age_rule": "history_length_minus_one_minus_chronological_index",
            "candidate_rule": "exclude_exactly_the_newest_current_equivalent_event_then_score_all_older_events",
            "current_equivalence_audit": "newest_post_ocr_spatial_tokens_must_equal_current_ocr_spatial_tokens",
            "transition_summary_rule": "canonical_GUIOwlV2Action_plus_exact_OCR_delta_plus_generic_256x256_RGB_MAD_with_foreground_app_and_executor_result_fixed_unknown",
            "live_ocr_record_rule": "exact_build_ocr_record_canonical_rebuild_over_original_RGB_or_RGBA_encoded_bytes_no_GUIOdyssey_RGBA_input_contract",
            "requested_budget_is_model_input": False,
            "policy_vision_feature_is_model_input": False,
            "ocr_backend_config_sha256": "51e08a9565f804ecb1ee7f12887cc12742b84c04996fffbbde7c0f304e4bd036",
            "ocr_backend_manifest_sha256": "107478672438b52e9c2ccc9ef8de5d13d4e16329ab1afbe95772e2d3df5bd2a9",
            "ocr_model_repo": "gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en",
            "ocr_model_revision": "0dbc766a73ee88d10d52285d434dbfec58617835",
            "summary_and_normalization": "exact_gate_v1_preregistration_label_blind_feature_contract",
            "feature_or_ocr_fallback_allowed": False,
        },
        "online feature binding",
    )
    development = _mapping(value.get("development"), "closed-loop development")
    _equal(
        dict(development),
        {
            "partition": "train",
            "task_indices_per_template": [0],
            "template_count": 60,
            "paired_instance_count": 60,
            "primary_arm_episode_count": 120,
            "seed": 314159,
            "first_12_are_infrastructure_smoke_and_remain_in_denominator": True,
            "hidden_retry_allowed": False,
            "failure_policy": "intention_to_treat_zero_plus_classification",
        },
        "closed-loop development",
    )
    _equal(
        dict(_mapping(value.get("arm_counterbalance"), "arm counterbalance")),
        {
            "key_utf8": "protocol_id+NUL+partition+NUL+suite_seed_decimal+NUL+task_type+NUL+task_index_decimal",
            "digest": "sha256",
            "bit": "digest_byte_0_bit_0",
            "bit_0_order": ["independent_B2", "recent_B2"],
            "bit_1_order": ["recent_B2", "independent_B2"],
            "each_arm_requires_independent_initialize_reset_teardown": True,
        },
        "arm counterbalance",
    )
    _equal(
        dict(_mapping(value.get("deterministic_random_B2"), "random B2")),
        {
            "candidate_key": "sha256(arm_counterbalance_key+NUL+deterministic_random_B2+NUL+decision_index_decimal+NUL+event_step_id_decimal)",
            "rank_order": "ascending_digest_then_ascending_event_step_id",
            "selection": "first_min_2_n_then_return_ascending_event_step_ids",
            "python_rng_allowed": False,
        },
        "random B2",
    )
    _equal(
        dict(_mapping(value.get("primary_estimator"), "primary estimator")),
        {
            "development_instance_indices": [0],
            "sealed_test_instance_indices": [0, 1, 2],
            "instance_pair_effect": "official_terminal_success_independent_minus_recent",
            "failed_or_missing_arm_success_value": 0.0,
            "template_effect": "arithmetic_mean_over_all_preregistered_instance_pair_effects",
            "overall_effect": "arithmetic_mean_over_template_effects_in_androidworld_task_partition_order",
            "paired_unit": "partition_task_type_task_index_with_two_independently_initialized_arms_on_the_identical_frozen_instance_record",
            "template_or_instance_exclusion_allowed": False,
        },
        "primary estimator",
    )
    expected_bootstrap = {
        "unit": "template_cluster",
        "template_order": "androidworld_task_partition_split_task_types",
        "resample": "sample_T_template_indices_with_replacement_and_carry_all_preregistered_instance_pairs_for_each_sampled_template",
        "sample_size": "T_equals_60_train_or_25_test",
        "rng": "python_random_Random_271828_randrange_T_exactly_T_calls_per_replicate",
        "resamples": 10000,
        "seed": 271828,
        "confidence": 0.9,
        "interval": "percentile",
        "quantile": "Hyndman_Fan_type_7",
        "lower_quantile": 0.05,
    }
    _equal(
        dict(_mapping(value.get("primary_bootstrap"), "closed-loop bootstrap")),
        expected_bootstrap,
        "closed-loop bootstrap",
    )
    sealed = _mapping(value.get("sealed_test"), "sealed test")
    _equal(
        dict(sealed),
        {
            "partition": "test",
            "template_count": 25,
            "task_indices_per_template": [0, 1, 2],
            "paired_instance_count": 75,
            "primary_arm_episode_count": 150,
            "seed": 161803,
            "access_before_development_go_allowed": False,
            "top_up_allowed": False,
            "primary_metric": "equal_template_mean_of_within_template_mean_official_terminal_success_paired_differences",
            "cluster_unit": "template_carrying_all_three_instance_pairs",
        },
        "sealed test",
    )
    _equal(value.get("validation_partition_reuse_for_primary_evidence_allowed"), False, "validation reuse")


def _validate_matched_nll(value: Mapping[str, Any]) -> None:
    expected = {
        "frozen_before_first_formal_closed_loop_episode": True,
        "execution_stage": "after_each_authorized_partition_closed_loop_primary_report",
        "primary_claim_partition": "sealed_test75_only",
        "partition_roles": {
            "train60": "development_diagnostic_only_no_primary_paper_claim_or_contract_change",
            "sealed_test75": "primary_mechanism_evidence_only_after_development_go_authorizes_test_access",
        },
        "state_distribution": "per_instance_pair_two_origin_policy_decision_state_multiset_without_cross_origin_deduplication",
        "reference": "pair_union_reference_M_ind_union_M_recent_max_four_history_images_plus_current",
        "reference_generation_repeats": 2,
        "reference_requires_exact_canonical_agreement": True,
        "state_evaluation": "at_every_state_from_both_origin_episodes_evaluate_both_memories_on_same_pair_union_canonical_action",
        "metric": "mean_teacher_forced_action_token_NLL_for_each_memory",
        "restoration_mass": "R_m_equals_D_empty_minus_D_M_m_under_pair_union_reference_raw_may_be_negative",
        "normalized_restoration_mass_role": "descriptive_only",
        "instance_aggregation": "equal_state_within_each_origin_episode_then_equal_half_weight_over_two_origin_episodes",
        "template_aggregation": "arithmetic_mean_over_all_preregistered_instances_index0_for_train_and_indices0_1_2_for_test",
        "success_aggregation": "per_arm_arithmetic_mean_of_official_intention_to_treat_success_over_all_preregistered_instances",
        "instance_eligibility": "both_origin_episodes_present_nonempty_and_every_policy_decision_state_has_exact_repeat_reference_finite_NLL_and_KL_and_complete_selection_budget_audit",
        "template_eligibility": "all_preregistered_instance_pairs_are_eligible_one_for_train_three_for_test",
        "ineligible_policy": "exclude_from_mechanism_only_record_reason_no_retry_no_top_up_retain_in_closed_loop_ITT",
        "matching": "eligible_template_enters_caliper_subset_when_abs_template_mean_NLL_independent_minus_recent_lte_caliper",
        "primary_caliper_nats_per_token": 0.05,
        "sensitivity_calipers_nats_per_token": [0.02, 0.1],
        "minimum_matched_template_clusters_for_claim": 15,
        "primary_statistic": "mean_over_matched_templates_of_sign_eps_Rind_minus_Rrecent_times_success_independent_minus_success_recent",
        "restoration_tie_epsilon": 1e-12,
        "restoration_tie_contribution": 0.0,
        "bootstrap": {
            "unit": "matched_template_cluster",
            "resample": "sample_m_matched_template_records_with_replacement",
            "sample_size": "m_equals_frozen_primary_caliper_matched_template_count",
            "rng": "python_random_Random_271828_randrange_m_exactly_m_calls_per_replicate",
            "resamples": 10000,
            "seed": 271828,
            "confidence": 0.9,
            "interval": "percentile",
            "quantile": "Hyndman_Fan_type_7",
            "lower_quantile": 0.05,
        },
        "claim_requires": "sealed_test75_primary_caliper_matched_template_count_at_least_15_and_primary_statistic_bootstrap_lower_strictly_greater_than_zero",
        "interpretation": "post_treatment_mechanism_association_on_controller_conditioned_state_distributions_not_a_mediated_causal_effect",
        "sensitivity_may_replace_primary": False,
        "outcome_or_result_based_caliper_selection_allowed": False,
    }
    _equal(dict(value), expected, "matched-NLL contract")


def _validate_runtime(value: Mapping[str, Any]) -> None:
    _equal(value.get("confirm_host"), "hyper00_or_hyper01_h200", "confirm host")
    _equal(value.get("gpu_count"), 4, "confirm GPU count")
    _equal(value.get("worker_count"), 4, "confirm worker count")
    _equal(value.get("states_per_worker"), 5, "states per worker")
    _equal(
        value.get("gpu_topology_smoke_protocol_id"),
        "causalcache_independent_confirm_gpu_topology_smoke_v1",
        "GPU topology smoke protocol",
    )
    _equal(
        value.get("gpu_topology_smoke_required_before_confirm_semantic_decode"),
        True,
        "GPU topology smoke order",
    )
    _equal(
        value.get("gpu_topology_smoke_phase_timeout_seconds"),
        7200,
        "GPU topology smoke timeout",
    )
    _equal(
        value.get(
            "gpu_topology_smoke_receipt_must_bind_same_allocation_model_and_snapshot"
        ),
        True,
        "GPU topology smoke binding",
    )
    _equal(value.get("policy_vision_phase_timeout_seconds"), 7200, "policy-vision timeout")
    _equal(value.get("restoration_phase_timeout_seconds"), 43200, "restoration timeout")
    _equal(value.get("worker_termination_grace_seconds"), 30, "worker termination grace")
    _equal(value.get("dtype"), "bfloat16", "confirm dtype")


def source_inventory(contract: IndependentConfirmContract) -> tuple[Mapping[str, Any], ...]:
    paths = tuple(contract.source["required_source_a_paths"])
    records: list[Mapping[str, Any]] = []
    for relative in paths:
        path = contract.repository_root / relative
        payload = _regular_file_bytes(path, label=f"Source-A path {relative}")
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return tuple(records)


def validate_lineage_files(contract: IndependentConfirmContract) -> tuple[Mapping[str, Any], ...]:
    records = []
    for name, raw in _mapping(contract.data["lineage"], "lineage").items():
        item = _mapping(raw, f"lineage {name}")
        relative = _safe_relative(item.get("path"), f"lineage {name} path")
        payload = _regular_file_bytes(
            contract.repository_root / relative,
            label=f"lineage {name}",
        )
        if (
            len(payload) != item.get("size_bytes")
            or hashlib.sha256(payload).hexdigest() != item.get("sha256")
        ):
            raise ValueError(f"lineage bytes drifted: {name}")
        records.append(
            {
                "name": name,
                "path": relative,
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
        )
    return tuple(records)


def build_source_validation(
    contract: IndependentConfirmContract,
) -> Mapping[str, Any]:
    lineage = validate_lineage_files(contract)
    inventory = source_inventory(contract)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": VALIDATION_STATUS,
        "contract_sha256": contract.sha256,
        "source_a_path_count": len(inventory),
        "source_a_inventory_sha256": sha256_bytes(canonical_json_bytes(list(inventory))),
        "source_a_inventory": list(inventory),
        "lineage_count": len(lineage),
        "lineage_inventory_sha256": sha256_bytes(canonical_json_bytes(list(lineage))),
        "lineage_inventory": list(lineage),
        "runner_freeze_path": RUNNER_FREEZE_PATH,
        "runner_freeze_present": False,
        "confirm_semantic_decode_count": 0,
        "policy_generation_count": 0,
        "restoration_teacher_forward_count": 0,
        "model_load_count": 0,
        "network_call_count": 0,
        "file_write_count": 0,
        "execution_authorized": False,
    }


def validate_runner_freeze(
    contract: IndependentConfirmContract,
    value: Mapping[str, Any],
    *,
    source_a_commit: str,
) -> None:
    if not isinstance(source_a_commit, str) or _COMMIT.fullmatch(source_a_commit) is None:
        raise ValueError("Source-A commit is invalid")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "source_a_commit",
        "source_a_parent_count",
        "source_a_remote_main",
        "contract_sha256",
        "source_a_inventory",
        "source_a_inventory_sha256",
        "loaded_module_inventory",
        "loaded_module_inventory_sha256",
        "confirm_access_authorized",
        "restoration_access_requires_payload_commit",
        "closed_loop_train_requires_confirm_go",
        "sealed_test_requires_development_go",
    }
    if set(value) != expected_keys:
        raise ValueError("Execution-B runner-freeze key inventory drifted")
    _equal(value.get("schema_version"), SCHEMA_VERSION, "runner-freeze schema")
    _equal(value.get("protocol_id"), PROTOCOL_ID, "runner-freeze protocol")
    _equal(value.get("status"), "FROZEN_CAUSALCACHE_INDEPENDENT_CONFIRM_EXECUTION_B_V1", "runner-freeze status")
    _equal(value.get("source_a_commit"), source_a_commit, "runner-freeze Source-A commit")
    _equal(value.get("source_a_parent_count"), 1, "Source-A parent count")
    _equal(value.get("source_a_remote_main"), source_a_commit, "Source-A remote main")
    _equal(value.get("contract_sha256"), contract.sha256, "runner-freeze contract")
    for name in ("source_a", "loaded_module"):
        raw_inventory = _sequence(
            value.get(f"{name}_inventory"), f"runner-freeze {name} inventory"
        )
        inventory = []
        seen_paths: set[str] = set()
        for index, raw in enumerate(raw_inventory):
            item = _mapping(raw, f"runner-freeze {name} inventory item {index}")
            if set(item) != {"path", "sha256", "size_bytes"}:
                raise ValueError(f"runner-freeze {name} inventory schema drifted")
            path = _safe_relative(item.get("path"), f"runner-freeze {name} path")
            if path in seen_paths:
                raise ValueError(f"runner-freeze {name} inventory has duplicate paths")
            seen_paths.add(path)
            digest = _validate_sha(
                item.get("sha256"), f"runner-freeze {name} item SHA256"
            )
            size = item.get("size_bytes")
            if type(size) is not int or size <= 0:
                raise ValueError(f"runner-freeze {name} item size is invalid")
            inventory.append({"path": path, "sha256": digest, "size_bytes": size})
        if not inventory:
            raise ValueError(f"runner-freeze {name} inventory is empty")
        expected_digest = sha256_bytes(canonical_json_bytes(inventory))
        _equal(
            value.get(f"{name}_inventory_sha256"),
            expected_digest,
            f"runner-freeze {name} inventory SHA256",
        )
    _equal(value.get("confirm_access_authorized"), True, "confirm authorization")
    _equal(value.get("restoration_access_requires_payload_commit"), True, "restoration barrier")
    _equal(value.get("closed_loop_train_requires_confirm_go"), True, "closed-loop barrier")
    _equal(value.get("sealed_test_requires_development_go"), True, "test barrier")


__all__ = [
    "CANONICAL_CONFIG_PATH",
    "CONFIRM_SOURCE_IDS_SHA256",
    "FROZEN_CONFIG_SHA256",
    "IndependentConfirmContract",
    "PROTOCOL_ID",
    "REQUIRED_EXECUTION_MODULES",
    "RUNNER_FREEZE_PATH",
    "SCHEMA_VERSION",
    "VALIDATION_STATUS",
    "build_source_validation",
    "canonical_json_bytes",
    "load_runner_freeze_file",
    "pretty_json_bytes",
    "sha256_bytes",
    "source_inventory",
    "validate_lineage_files",
    "validate_runner_freeze",
]
