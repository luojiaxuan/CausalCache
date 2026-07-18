"""Fail-closed Source-A contract for the budget-agnostic set utility route."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_predictor_v1"
SOURCE_STATUS = "source_only_frozen_before_new_development_data_access"
VALIDATION_STATUS = "VALID_SOURCE_ONLY_SET_UTILITY_PREDICTOR_V1"
CANONICAL_CONFIG_PATH = "code/configs/causalcache_set_utility_predictor_v1.json"
FROZEN_CONFIG_SHA256 = (
    "ac4333055a0cc78fa229f44a6981d7804d753bb72b466872540473c85074a3b1"
)
MODEL_FAMILIES = ("pairwise_additive", "deepsets")
COMPARATOR_ORDER = (
    "pairwise_additive",
    "deepsets",
    "recent",
    "ocr_rgb",
    "oracle_independent_J",
    "exact_subset_oracle",
)
PHASE2_TRACKS = (
    ("B3_n6", 6, 3, 42),
    ("B4_n8", 8, 4, 163),
)
EXPECTED_SECTION_SHA256 = {
    "scientific_scope": "93788d092764309819d74f2ff71054695d9e06f1cda1a0bd75194c8ae86844a9",
    "utility_contract": "57fd1cba793287a6b89409788918df02b1760b79abeb26db499a50c8bc29f98f",
    "teacher_and_representation": "3385f85f88da819c6456a6067cd5779c9ab31765e22badde92f4f36cf26a61dc",
    "new_development_data": "8b16a00d91dda855a26856a67e207f4259f11dd8d392061e0d7f400e86b1456b",
    "p0_policy_blind_roster_discovery": "82c1591cb03f39c13da48adb397ba3b1d7c21d99c2563db23d19d6ceeef88d56",
    "phase1_exact_b2": "d558a53fc17fce4c50fbef8f6f2ebb2ef3c787c68756d3f2491738b9b4336e6b",
    "model_families": "8626b0924d4cee042c62cfa90544955209f70528a23d7afc053688b1b1939ada",
    "phase1_comparators_and_evaluation": "855df7753eb865f6230202fa0d416806f7e75392913e8e4c7cd1b7c231767b59",
    "phase2_cardinality_transfer": "81b4de991f63912171449f1e28da03aadbf5a82521a8f18e99b1b2e905e9b2b4",
    "locked_followups": "f92f0e88e3c9e02190ff8fd9a000ccf64c688006995a8ee4708c5b42d955a8b7",
    "source_only_operation_contract": "e45043745018c2a8898e1bc0652159465534f18f101ee8ee6622d7cc7b216e8b",
    "authorization": "5f7ab664fbe928741735259b422003202068e675244f96ec87813c33c6bb73b8",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    *EXPECTED_SECTION_SHA256,
}


@dataclass(frozen=True)
class SetUtilityPredictorContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def utility(self) -> Mapping[str, Any]:
        return _mapping(self.data["utility_contract"], "utility contract")

    @property
    def development_data(self) -> Mapping[str, Any]:
        return _mapping(self.data["new_development_data"], "development data")

    @property
    def phase1(self) -> Mapping[str, Any]:
        return _mapping(self.data["phase1_exact_b2"], "phase 1")

    @property
    def models(self) -> Mapping[str, Any]:
        return _mapping(self.data["model_families"], "model families")

    @property
    def phase2(self) -> Mapping[str, Any]:
        return _mapping(self.data["phase2_cardinality_transfer"], "phase 2")

    @property
    def locks(self) -> Mapping[str, Any]:
        return _mapping(self.data["locked_followups"], "locked followups")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
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
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _strict_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
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


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = _mapping(config[name], name.replace("_", " "))
    observed = sha256_bytes(canonical_json_bytes(value))
    expected = EXPECTED_SECTION_SHA256[name]
    if observed != expected:
        raise ValueError(
            f"{name} section drifted: expected {expected}, got {observed}"
        )
    return value


def _validate_scope_and_utility(config: Mapping[str, Any]) -> None:
    scope = _section(config, "scientific_scope")
    _equal(scope["development_only"], True, "development-only scope")
    _equal(scope["paper_primary_evidence"], False, "paper evidence lock")
    _equal(
        scope["historical_independent_and_conditional_no_go_preserved"],
        True,
        "historical no-go preservation",
    )

    utility = _section(config, "utility_contract")
    _equal(
        utility["target_definition"],
        "U_t(S)=D_t(empty)-D_t(S)",
        "utility target",
    )
    _equal(utility["empty_set_utility"], 0.0, "empty utility")
    for key in (
        "predictor_receives_budget",
        "predictor_receives_remaining_budget",
        "predictor_receives_selection_order",
        "iterative_conditional_greedy_is_primary",
        "learned_roundwise_stop_threshold_used",
    ):
        _equal(utility[key], False, f"utility exclusion {key}")
    for key in (
        "utility_is_budget_agnostic",
        "budget_enters_search_constraint_only",
        "empty_set_is_always_search_eligible",
        "selection_is_joint_subset_scoring",
    ):
        _equal(utility[key], True, f"utility invariant {key}")


def _validate_teacher_and_data_firewall(config: Mapping[str, Any]) -> None:
    teacher = _section(config, "teacher_and_representation")
    _equal(
        teacher["policy_revision"],
        "06d5faecff74840bab2be2425e9c42667a5d04fc",
        "teacher revision",
    )
    _equal(teacher["policy_frozen"], True, "teacher freeze")
    _equal(teacher["policy_training_allowed"], False, "teacher training lock")
    _equal(
        teacher["cross_layer_kv_surgery_allowed"],
        False,
        "KV surgery lock",
    )

    data = _section(config, "new_development_data")
    _equal(data["new_data_required"], True, "new data requirement")
    _equal(
        data["split_unit"],
        "trajectory_id_grouped_by_instruction_app_group_sha256",
        "split unit",
    )
    _equal(
        data["instruction_app_group_definition"],
        "SHA256(canonical_json({normalized_instruction,sorted_normalized_apps}))",
        "instruction-app group definition",
    )
    _equal(
        data["instruction_app_group_overlap_across_new_roles_allowed"],
        False,
        "instruction-app group overlap",
    )
    _equal(
        data["instruction_app_group_key_is_model_feature"],
        False,
        "instruction-app group feature firewall",
    )
    _equal(
        data["training_or_tuning_eligible_source"],
        "new_development_v1_plus_historical_formal58_train_only",
        "eligible training source",
    )
    _equal(
        data["historical_formal58_training_rows_eligible"],
        True,
        "formal58 legacy train eligibility",
    )
    _equal(
        data["historical_formal58_role"],
        "legacy_train_only_never_new_evaluation",
        "formal58 role",
    )
    for key in (
        "consumed_old_dev5_training_rows_eligible",
        "consumed_old_dev5_tuning_rows_eligible",
        "consumed_fresh16_training_rows_eligible",
        "consumed_fresh16_tuning_rows_eligible",
        "consumed_confirm20_training_rows_eligible",
        "consumed_confirm20_tuning_rows_eligible",
        "data_access_before_freeze_b_allowed",
    ):
        _equal(data[key], False, f"data firewall {key}")
    for key in (
        "roster_status",
        "state_count_status",
        "source_identity_manifest_status",
    ):
        _equal(
            data[key],
            "UNBOUND_REQUIRES_SEPARATE_PRE_ACCESS_FREEZE_B",
            f"pre-access freeze {key}",
        )


def _validate_policy_blind_discovery(config: Mapping[str, Any]) -> None:
    discovery = _section(config, "p0_policy_blind_roster_discovery")
    _equal(
        discovery["status"],
        "LOCKED_REQUIRES_SEPARATE_P0_RUNNER_FREEZE",
        "P0 discovery lock",
    )
    _equal(
        discovery["separate_p0_config_sha256"],
        "d10484f53f579bf26012ba6fdd3e7e701c90d7e760db6667ebf7e74c75a98957",
        "P0 source config",
    )
    historical = _mapping(discovery["historical_observation"], "P0 observation")
    _equal(
        historical["decision_count_above_12_exclusion_count"],
        81,
        "historical long-trajectory exclusion count",
    )
    inherited = _mapping(
        discovery["eligibility_inherited_unchanged"], "P0 inherited eligibility"
    )
    _equal(inherited["minimum_decisions_per_trajectory"], 13, "P0 minimum length")
    _equal(inherited["terminal_status"], "success", "P0 terminal status")
    _equal(inherited["terminal_signal_must_be_last"], True, "P0 terminal order")
    _equal(inherited["require_embedded_supported_images"], True, "P0 images")
    _equal(
        tuple(_sequence(inherited["parser_compatible_action_types"], "P0 actions")),
        ("back", "home", "long_press", "stop", "swipe", "tap", "type_text", "wait"),
        "P0 parser-compatible actions",
    )
    changed = _mapping(discovery["long_trajectory_change_only"], "P0 long change")
    _equal(changed["old_maximum_decisions_per_trajectory"], 12, "old P0 cap")
    _equal(
        changed["discovery_minimum_decisions_per_trajectory"],
        13,
        "P0 discovery minimum",
    )
    _equal(
        changed["discovery_maximum_decisions_per_trajectory"],
        64,
        "P0 discovery maximum",
    )
    _equal(
        changed["decision_count_above_12_is_an_exclusion"],
        False,
        "P0 long-trajectory inclusion",
    )
    _equal(
        changed["all_non_length_eligibility_rules_must_be_byte_equivalent_to_inherited_parser"],
        True,
        "P0 single eligibility change",
    )
    blindness = _mapping(discovery["policy_blindness"], "P0 policy blindness")
    for key in (
        "policy_model_load_count",
        "policy_forward_count",
        "restoration_distance_access_count",
        "restoration_label_access_count",
        "ocr_score_access_count",
        "learned_gate_score_access_count",
    ):
        _equal(blindness[key], 0, f"P0 forbidden operation {key}")
    _equal(
        blindness["policy_or_restoration_output_may_determine_roster"],
        False,
        "P0 output-blind roster",
    )
    output = _mapping(discovery["allowed_output"], "P0 allowed output")
    _equal(
        output["instruction_app_group_sha256_without_instruction_content"],
        True,
        "P0 instruction-app group output",
    )
    for key in (
        "train_tune_evaluation_assignment",
        "query_state_selection",
        "restoration_or_utility_labels",
    ):
        _equal(output[key], False, f"P0 output exclusion {key}")
    stages = _mapping(discovery["stage_boundaries"], "P0 stage boundaries")
    for key in (
        "concrete_split_counts_require_new_committed_freeze",
        "label_execution_A_source_and_roster_freeze_is_separate",
        "label_execution_B_teacher_run_is_separate",
        "p0_may_not_authorize_label_execution_A_or_B",
    ):
        _equal(stages[key], True, f"P0 stage boundary {key}")


def _validate_phase1_and_models(config: Mapping[str, Any]) -> None:
    phase1 = _section(config, "phase1_exact_b2")
    _equal(tuple(_sequence(phase1["allowed_budgets"], "phase 1 budgets")), (1, 2), "phase 1 budgets")
    _equal(
        tuple(_sequence(phase1["labeled_cardinalities"], "phase 1 cardinalities")),
        (0, 1, 2),
        "phase 1 cardinalities",
    )
    _equal(phase1["maximum_labeled_cardinality"], 2, "phase 1 maximum cardinality")
    _equal(phase1["subset_sampling_allowed"], False, "phase 1 sampling lock")
    _equal(
        phase1["missing_exact_subset_label_allowed"],
        False,
        "phase 1 completeness",
    )
    candidate_count = phase1["candidate_count_maximum"]
    if type(candidate_count) is not int or candidate_count < 2:
        raise ValueError("phase 1 candidate maximum must be an integer >= 2")
    expected_count = 1 + candidate_count + math.comb(candidate_count, 2)
    _equal(
        phase1["label_count_at_candidate_maximum"],
        expected_count,
        "phase 1 exact label count",
    )

    models = _section(config, "model_families")
    if set(models) != {"shared_requirements", *MODEL_FAMILIES}:
        raise ValueError("model family inventory drifted")
    shared = _mapping(models["shared_requirements"], "shared model requirements")
    _equal(shared["budget_feature_allowed"], False, "model budget feature lock")
    _equal(shared["permutation_invariant"], True, "permutation invariance")
    _equal(shared["variable_cardinality_input"], True, "variable cardinality")
    _equal(shared["empty_set_output_constrained_to_zero"], True, "empty output")
    _equal(
        shared["student_feature_schema_status"],
        "UNBOUND_REQUIRES_SEPARATE_PRE_TRAIN_FREEZE_B",
        "student feature freeze",
    )
    _equal(
        tuple(_sequence(shared["student_feature_candidates"], "student features")),
        (
            "lightweight_q64_h64_ocr_rgb_recency",
            "plus_frozen_gui_owl_final_main_normalized_visual_embedding",
        ),
        "student feature candidates",
    )
    _equal(
        shared["feature_choice_after_new_evaluation_label_access_allowed"],
        False,
        "student feature outcome firewall",
    )
    pairwise = _mapping(models["pairwise_additive"], "pairwise model")
    _equal(pairwise["maximum_explicit_interaction_order"], 2, "pairwise order")
    _equal(pairwise["higher_order_interactions_representable"], False, "pairwise limit")
    deepsets = _mapping(models["deepsets"], "DeepSets model")
    _equal(deepsets["role"], "main_candidate", "DeepSets role")
    _equal(deepsets["set_transformer_allowed_in_v1"], False, "Set Transformer lock")


def _validate_comparators_and_phase2(config: Mapping[str, Any]) -> None:
    evaluation = _section(config, "phase1_comparators_and_evaluation")
    ordered = tuple(
        _sequence(evaluation["ordered_methods"], "phase 1 ordered methods")
    )
    _equal(ordered, COMPARATOR_ORDER, "phase 1 comparator order")
    recent = _mapping(evaluation["recent"], "recent comparator")
    _equal(
        recent["implementation"],
        "most_recent_event_ids_fill_up_to_B",
        "recent comparator implementation",
    )
    _equal(
        recent["may_be_retuned_on_new_evaluation_labels"],
        False,
        "recent comparator retuning",
    )
    oracle_j = _mapping(
        evaluation["oracle_independent_J"], "oracle independent J"
    )
    _equal(
        oracle_j["event_score"],
        "0.5*(Delta_j(empty)+mean_{i_not_equal_j}Delta_j({i}))",
        "oracle-independent J target",
    )
    _equal(oracle_j["selection"], "strictly_positive_top_B", "J selection")
    _equal(
        evaluation["phase1_result_may_unlock_closed_loop_directly"],
        False,
        "closed-loop route lock",
    )

    phase2 = _section(config, "phase2_cardinality_transfer")
    _equal(
        phase2["status"],
        "LOCKED_UNTIL_PHASE1_ENTRY_RULE_PASSES_AND_NEW_FREEZE_IS_PUSHED",
        "phase 2 lock",
    )
    tracks = tuple(_sequence(phase2["tracks"], "phase 2 tracks"))
    if len(tracks) != len(PHASE2_TRACKS):
        raise ValueError("phase 2 track count drifted")
    for raw, expected in zip(tracks, PHASE2_TRACKS, strict=True):
        track = _mapping(raw, "phase 2 track")
        track_id, candidate_count, maximum_cardinality, label_count = expected
        _equal(track["track_id"], track_id, f"{track_id} id")
        _equal(track["candidate_count"], candidate_count, f"{track_id} n")
        _equal(
            track["maximum_labeled_cardinality"],
            maximum_cardinality,
            f"{track_id} maximum cardinality",
        )
        expected_labels = sum(
            math.comb(candidate_count, cardinality)
            for cardinality in range(maximum_cardinality + 1)
        )
        _equal(
            track["exact_label_count_per_state"],
            expected_labels,
            f"{track_id} label count",
        )
    zero_shot = _mapping(phase2["zero_shot"], "zero-shot transfer")
    _equal(zero_shot["phase1_checkpoint_byte_identical"], True, "zero-shot checkpoint")
    _equal(zero_shot["higher_cardinality_training_label_count"], 0, "zero-shot labels")
    _equal(zero_shot["hyperparameter_change_allowed"], False, "zero-shot hyperparameters")
    few_shot = _mapping(phase2["few_shot"], "few-shot transfer")
    _equal(
        few_shot["calibration_and_evaluation_trajectory_disjoint"],
        True,
        "few-shot split firewall",
    )
    _equal(few_shot["new_architecture_search_allowed"], False, "few-shot architecture lock")


def _validate_locks_and_source_operations(config: Mapping[str, Any]) -> None:
    locks = _section(config, "locked_followups")
    for key in (
        "closed_loop_status",
        "matched_nll_status",
        "sealed_androidworld_test_status",
        "paper_primary_table_status",
    ):
        _equal(locks[key], "LOCKED", f"follow-up lock {key}")
    _equal(
        locks["exploratory_closed_loop_validation12_execution_status"],
        "SUPERSEDED_AND_NOT_AUTHORIZED",
        "superseded closed-loop protocol",
    )

    operations = _section(config, "source_only_operation_contract")
    if not operations:
        raise ValueError("source-only operation inventory is empty")
    _equal(operations["config_read_count"], 1, "source config read count")
    for key, value in operations.items():
        if key == "config_read_count":
            continue
        if type(value) is not int or value != 0:
            raise ValueError(f"source-only operation must be exact zero: {key}")

    authorization = _section(config, "authorization")
    if not authorization:
        raise ValueError("source-only authorization inventory is empty")
    for key, value in authorization.items():
        if value is not False:
            raise PermissionError(f"source-only authorization must remain false: {key}")


def validate_set_utility_predictor_contract(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if set(config) != TOP_LEVEL_KEYS:
        raise ValueError("set utility predictor top-level key inventory drifted")
    _equal(config["schema_version"], SCHEMA_VERSION, "schema version")
    _equal(config["protocol_id"], PROTOCOL_ID, "protocol id")
    _equal(config["status"], SOURCE_STATUS, "source status")
    _validate_scope_and_utility(config)
    _validate_teacher_and_data_firewall(config)
    _validate_policy_blind_discovery(config)
    _validate_phase1_and_models(config)
    _validate_comparators_and_phase2(config)
    _validate_locks_and_source_operations(config)
    return {
        "status": VALIDATION_STATUS,
        "protocol_id": PROTOCOL_ID,
        "utility_target": "U_t(S)=D_t(empty)-D_t(S)",
        "utility_is_budget_agnostic": True,
        "budget_enters_search_constraint_only": True,
        "phase1_allowed_budgets": [1, 2],
        "phase1_exact_label_count_at_n16": 137,
        "model_families": list(MODEL_FAMILIES),
        "ordered_comparators": list(COMPARATOR_ORDER),
        "phase2_tracks": [track[0] for track in PHASE2_TRACKS],
        "new_development_data_required": True,
        "p0_policy_blind_roster_discovery_required": True,
        "p0_discovery_decision_range": [13, 64],
        "fresh16_training_or_tuning_allowed": False,
        "confirm20_training_or_tuning_allowed": False,
        "closed_loop_locked": True,
        "matched_nll_locked": True,
        "sealed_test_locked": True,
        "execution_authorized": False,
    }


def _canonical_contract_path(root: Path, supplied: str | Path) -> Path:
    raw = Path(supplied)
    candidate = raw if raw.is_absolute() else root / raw
    canonical_raw = root / CANONICAL_CONFIG_PATH
    canonical = canonical_raw.resolve()
    if (
        candidate.resolve() != canonical
        or candidate.is_symlink()
        or canonical_raw.is_symlink()
    ):
        raise ValueError("set utility predictor contract path is not canonical")
    return canonical


def load_frozen_set_utility_predictor_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> SetUtilityPredictorContract:
    root = Path(repository_root).resolve()
    source = _canonical_contract_path(root, path)
    payload = _regular_file_bytes(source, label="set utility predictor contract")
    observed = sha256_bytes(payload)
    if observed != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "set utility predictor contract bytes drifted: "
            f"expected {FROZEN_CONFIG_SHA256}, got {observed}"
        )
    config = _strict_json_bytes(payload, label="set utility predictor contract")
    validate_set_utility_predictor_contract(config)
    return SetUtilityPredictorContract(
        data=config,
        sha256=observed,
        repository_root=root,
        source_path=source,
    )


def validate_set_utility_predictor_source_only(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    contract = load_frozen_set_utility_predictor_contract(
        path,
        repository_root=repository_root,
    )
    result = validate_set_utility_predictor_contract(contract.data)
    operations = _mapping(
        contract.data["source_only_operation_contract"],
        "source-only operation contract",
    )
    return {
        **result,
        "config_path": CANONICAL_CONFIG_PATH,
        "config_sha256": contract.sha256,
        "source_contract_read_count": 1,
        "only_read_path": CANONICAL_CONFIG_PATH,
        "artifact_access_count": 0,
        "bound_artifacts_opened_or_verified": False,
        "source_only_operation_counts": dict(operations),
    }


__all__ = [
    "CANONICAL_CONFIG_PATH",
    "COMPARATOR_ORDER",
    "EXPECTED_SECTION_SHA256",
    "FROZEN_CONFIG_SHA256",
    "MODEL_FAMILIES",
    "PHASE2_TRACKS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SOURCE_STATUS",
    "SetUtilityPredictorContract",
    "VALIDATION_STATUS",
    "canonical_json_bytes",
    "load_frozen_set_utility_predictor_contract",
    "sha256_bytes",
    "validate_set_utility_predictor_contract",
    "validate_set_utility_predictor_source_only",
]
