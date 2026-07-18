from __future__ import annotations

import copy
import hashlib
import json
import shutil
import socket
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from unittest import mock

import pytest

import causalcache.set_utility_predictor_contract as contract_module
from causalcache.set_utility_predictor_contract import (
    CANONICAL_CONFIG_PATH,
    COMPARATOR_ORDER,
    FROZEN_CONFIG_SHA256,
    MODEL_FAMILIES,
    PHASE2_TRACKS,
    VALIDATION_STATUS,
    load_frozen_set_utility_predictor_contract,
    validate_set_utility_predictor_contract,
    validate_set_utility_predictor_source_only,
)
from scripts.validate_set_utility_predictor_contract import _parser


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _copy_canonical_config(destination_root: Path) -> Path:
    target = destination_root / CANONICAL_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, target)
    return target


def test_frozen_identity_and_budget_agnostic_public_contract() -> None:
    contract = load_frozen_set_utility_predictor_contract(repository_root=ROOT)
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == FROZEN_CONFIG_SHA256
    assert contract.sha256 == FROZEN_CONFIG_SHA256
    assert contract.utility["target_definition"] == "U_t(S)=D_t(empty)-D_t(S)"
    assert contract.utility["utility_is_budget_agnostic"]
    assert not contract.utility["predictor_receives_budget"]
    assert contract.utility["budget_enters_search_constraint_only"]
    assert contract.utility["empty_set_is_always_search_eligible"]
    assert not contract.utility["learned_roundwise_stop_threshold_used"]


def test_policy_blind_long_trajectory_discovery_is_a_separate_locked_stage() -> None:
    discovery = _config()["p0_policy_blind_roster_discovery"]
    assert discovery["historical_observation"][
        "decision_count_above_12_exclusion_count"
    ] == 81
    assert discovery["long_trajectory_change_only"][
        "discovery_minimum_decisions_per_trajectory"
    ] == 13
    assert discovery["long_trajectory_change_only"][
        "discovery_maximum_decisions_per_trajectory"
    ] == 64
    assert not discovery["long_trajectory_change_only"][
        "decision_count_above_12_is_an_exclusion"
    ]
    assert discovery["eligibility_inherited_unchanged"][
        "require_embedded_supported_images"
    ]
    blindness = discovery["policy_blindness"]
    for key in (
        "policy_model_load_count",
        "policy_forward_count",
        "restoration_distance_access_count",
        "restoration_label_access_count",
        "ocr_score_access_count",
        "learned_gate_score_access_count",
    ):
        assert blindness[key] == 0
    assert not blindness["policy_or_restoration_output_may_determine_roster"]
    stages = discovery["stage_boundaries"]
    assert stages["label_execution_A_source_and_roster_freeze_is_separate"]
    assert stages["label_execution_B_teacher_run_is_separate"]
    assert stages["p0_may_not_authorize_label_execution_A_or_B"]


def test_phase1_is_complete_b2_exact_and_compares_the_five_frozen_methods() -> None:
    contract = load_frozen_set_utility_predictor_contract(repository_root=ROOT)
    assert contract.phase1["allowed_budgets"] == [1, 2]
    assert contract.phase1["labeled_cardinalities"] == [0, 1, 2]
    assert contract.phase1["label_inventory_per_state"] == (
        "all_subsets_with_cardinality_at_most_2"
    )
    assert not contract.phase1["subset_sampling_allowed"]
    assert not contract.phase1["missing_exact_subset_label_allowed"]
    assert contract.phase1["candidate_count_maximum"] == 16
    assert contract.phase1["label_count_at_candidate_maximum"] == 137
    assert tuple(contract.models) == ("shared_requirements", *MODEL_FAMILIES)
    evaluation = contract.data["phase1_comparators_and_evaluation"]
    assert tuple(evaluation["ordered_methods"]) == COMPARATOR_ORDER
    assert evaluation["oracle_independent_J"]["event_score"] == (
        "0.5*(Delta_j(empty)+mean_{i_not_equal_j}Delta_j({i}))"
    )
    assert not evaluation["phase1_result_may_unlock_closed_loop_directly"]


def test_pairwise_and_deepsets_support_variable_cardinality_without_budget_input() -> None:
    models = _config()["model_families"]
    shared = models["shared_requirements"]
    assert shared["permutation_invariant"]
    assert shared["variable_cardinality_input"]
    assert not shared["budget_feature_allowed"]
    assert shared["empty_set_output_constrained_to_zero"]
    pairwise = models["pairwise_additive"]
    assert pairwise["supports_arbitrary_cardinality"]
    assert pairwise["maximum_explicit_interaction_order"] == 2
    assert not pairwise["higher_order_interactions_representable"]
    deepsets = models["deepsets"]
    assert deepsets["role"] == "main_candidate"
    assert deepsets["supports_arbitrary_cardinality"]
    assert not deepsets["set_transformer_allowed_in_v1"]


def test_phase2_freezes_zero_shot_and_few_shot_b3_b4_transfer() -> None:
    contract = load_frozen_set_utility_predictor_contract(repository_root=ROOT)
    assert contract.phase2["status"].startswith("LOCKED_")
    tracks = contract.phase2["tracks"]
    observed = tuple(
        (
            track["track_id"],
            track["candidate_count"],
            track["maximum_labeled_cardinality"],
            track["exact_label_count_per_state"],
        )
        for track in tracks
    )
    assert observed == PHASE2_TRACKS
    assert contract.phase2["zero_shot"]["phase1_checkpoint_byte_identical"]
    assert contract.phase2["zero_shot"][
        "higher_cardinality_training_label_count"
    ] == 0
    assert not contract.phase2["zero_shot"]["hyperparameter_change_allowed"]
    assert contract.phase2["few_shot"][
        "calibration_and_evaluation_trajectory_disjoint"
    ]
    assert not contract.phase2["few_shot"]["new_architecture_search_allowed"]


def test_new_data_firewall_and_downstream_locks_are_exact() -> None:
    contract = load_frozen_set_utility_predictor_contract(repository_root=ROOT)
    data = contract.development_data
    assert data["training_or_tuning_eligible_source"] == (
        "new_development_v1_plus_historical_formal58_train_only"
    )
    assert data["historical_formal58_training_rows_eligible"]
    assert data["historical_formal58_role"] == (
        "legacy_train_only_never_new_evaluation"
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
        assert not data[key]
    assert contract.locks["closed_loop_status"] == "LOCKED"
    assert contract.locks["matched_nll_status"] == "LOCKED"
    assert contract.locks["sealed_androidworld_test_status"] == "LOCKED"
    assert contract.locks["exploratory_closed_loop_validation12_execution_status"] == (
        "SUPERSEDED_AND_NOT_AUTHORIZED"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["utility_contract"].update(
            {"predictor_receives_budget": True}
        ),
        lambda value: value["new_development_data"].update(
            {"consumed_fresh16_training_rows_eligible": True}
        ),
        lambda value: value["p0_policy_blind_roster_discovery"][
            "long_trajectory_change_only"
        ].update({"discovery_maximum_decisions_per_trajectory": 12}),
        lambda value: value["p0_policy_blind_roster_discovery"][
            "policy_blindness"
        ].update({"policy_forward_count": 1}),
        lambda value: value["phase1_exact_b2"].update(
            {"subset_sampling_allowed": True}
        ),
        lambda value: value["model_families"]["deepsets"].update(
            {"set_transformer_allowed_in_v1": True}
        ),
        lambda value: value["phase1_comparators_and_evaluation"][
            "ordered_methods"
        ].reverse(),
        lambda value: value["phase2_cardinality_transfer"]["zero_shot"].update(
            {"higher_cardinality_training_label_count": 1}
        ),
        lambda value: value["locked_followups"].update(
            {"closed_loop_status": "OPEN"}
        ),
        lambda value: value["authorization"].update(
            {"training_authorized": True}
        ),
    ),
)
def test_any_scientific_firewall_or_authority_drift_fails_closed(mutation) -> None:
    changed = copy.deepcopy(_config())
    mutation(changed)
    with pytest.raises((PermissionError, ValueError), match="drifted|false|zero"):
        validate_set_utility_predictor_contract(changed)


def test_unknown_top_level_key_fails_closed() -> None:
    changed = _config()
    changed["unbound_override"] = False
    with pytest.raises(ValueError, match="top-level key inventory drifted"):
        validate_set_utility_predictor_contract(changed)


def test_source_only_validator_reads_one_config_and_has_no_side_effects() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture_root = Path(directory)
        canonical = _copy_canonical_config(fixture_root)
        write_methods = (
            "write_bytes",
            "write_text",
            "touch",
            "mkdir",
            "rename",
            "replace",
            "unlink",
        )
        patches = [
            mock.patch.object(Path, name, side_effect=AssertionError(f"write: {name}"))
            for name in write_methods
        ]
        reader_patch = mock.patch.object(
            contract_module,
            "_regular_file_bytes",
            wraps=contract_module._regular_file_bytes,
        )
        patches.extend(
            (
                mock.patch.object(
                    socket, "socket", side_effect=AssertionError("network")
                ),
                mock.patch.object(
                    urllib.request,
                    "urlopen",
                    side_effect=AssertionError("network"),
                ),
                mock.patch.object(
                    subprocess, "run", side_effect=AssertionError("subprocess")
                ),
                reader_patch,
            )
        )
        for patch in patches:
            patch.start()
        try:
            result = validate_set_utility_predictor_source_only(
                repository_root=fixture_root
            )
            reader = reader_patch.target._regular_file_bytes
        finally:
            for patch in reversed(patches):
                patch.stop()

    assert reader.call_count == 1
    assert reader.call_args.args[0] == canonical.resolve()
    assert result["status"] == VALIDATION_STATUS
    assert result["only_read_path"] == CANONICAL_CONFIG_PATH
    assert result["source_contract_read_count"] == 1
    assert result["artifact_access_count"] == 0
    assert not result["bound_artifacts_opened_or_verified"]
    assert not result["execution_authorized"]
    operations = result["source_only_operation_counts"]
    assert operations["config_read_count"] == 1
    assert all(
        value == 0 for key, value in operations.items() if key != "config_read_count"
    )


def test_loader_rejects_noncanonical_path_byte_drift_and_symlink(tmp_path: Path) -> None:
    canonical = _copy_canonical_config(tmp_path)
    alternative = tmp_path / "alternative.json"
    shutil.copyfile(CONFIG, alternative)
    with pytest.raises(ValueError, match="path is not canonical"):
        load_frozen_set_utility_predictor_contract(
            alternative,
            repository_root=tmp_path,
        )

    canonical.write_bytes(canonical.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes drifted"):
        load_frozen_set_utility_predictor_contract(repository_root=tmp_path)

    canonical.unlink()
    canonical.symlink_to(CONFIG)
    with pytest.raises(ValueError, match="path is not canonical"):
        load_frozen_set_utility_predictor_contract(repository_root=tmp_path)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        contract_module._strict_json_bytes(b'{"x":1,"x":2}', label="fixture")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        contract_module._strict_json_bytes(b'{"x":NaN}', label="fixture")


def test_cli_defaults_to_canonical_source_only_contract() -> None:
    args = _parser().parse_args([])
    assert args.contract == Path(CANONICAL_CONFIG_PATH)
    assert args.repository_root == Path(".")
