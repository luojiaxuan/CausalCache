from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import causalcache.set_conditioned_v4_contract as contract_module
from causalcache.set_conditioned_v4_contract import (
    BASE_GIT_COMMIT,
    CANONICAL_CONFIG_PATH,
    EXECUTION_B_RUNNER_FREEZE_PATH,
    EXPECTED_BASE_EPOCHS,
    EXPECTED_BASE_OOF_RATIOS,
    EXPECTED_CHECKPOINTS,
    EXPECTED_STATES,
    FRESH16_FEATURE_SHA256,
    FRESH16_LABEL_SHA256,
    FROZEN_BASE_REVISION,
    FROZEN_CONFIG_SHA256_PLACEHOLDER,
    HISTORICAL_INDEPENDENT_SHA256,
    LABEL_BLIND_SEAL_STATUS,
    PROTOCOL_ID,
    REQUIRED_BRANCH,
    execution_b_freeze_payload,
    load_frozen_set_conditioned_v4_contract,
    sha256_bytes,
    validate_contract_data,
    validate_source_a,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


def _data() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


class SetConditionedV4ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_frozen_set_conditioned_v4_contract(
            repository_root=ROOT
        )

    def test_lineage_isolated_branch_and_v3_result_are_frozen(self) -> None:
        self.assertEqual(self.contract.data["protocol_id"], PROTOCOL_ID)
        lineage = self.contract.data["lineage"]
        self.assertEqual(lineage["branch"], REQUIRED_BRANCH)
        self.assertEqual(lineage["base_git_commit"], BASE_GIT_COMMIT)
        self.assertTrue(lineage["v3_result_is_immutable"])
        self.assertEqual(
            lineage["v3_result"]["required_outcome"],
            "stop_after_consumed_development_no_go",
        )
        source = self.contract.data["source_freeze"]
        self.assertEqual(
            source["execution_b_runner_freeze_path"],
            EXECUTION_B_RUNNER_FREEZE_PATH,
        )
        self.assertTrue(
            source[
                "execution_b_runner_freeze_must_be_absent_during_source_a_validation"
            ]
        )

    def test_formal_and_fresh_inputs_are_immutable_and_roles_are_strict(self) -> None:
        formal = self.contract.formal_input
        self.assertEqual(formal["trajectory_count"], 58)
        self.assertEqual(formal["state_count"], 174)
        self.assertEqual(formal["fold_sizes"], [12, 12, 12, 11, 11])
        fresh = self.contract.fresh_input
        files = {item["kind"]: item for item in fresh["exact_files"]}
        self.assertEqual(files["feature_states"]["sha256"], FRESH16_FEATURE_SHA256)
        self.assertEqual(files["label_states"]["sha256"], FRESH16_LABEL_SHA256)
        self.assertEqual(
            files["historical_independent_decisions"]["sha256"],
            HISTORICAL_INDEPENDENT_SHA256,
        )
        self.assertEqual(
            files["historical_independent_decisions"]["access_stage"],
            "evaluate_after_v4_label_blind_seal",
        )
        self.assertFalse(fresh["may_be_called_holdout"])
        self.assertFalse(fresh["may_be_called_test"])
        self.assertFalse(fresh["may_authorize_go"])
        self.assertFalse(fresh["may_tune_after_label_join"])

    def test_five_strong_base_checkpoints_are_exact_and_immutable(self) -> None:
        base = self.contract.frozen_base
        self.assertEqual(base["immutable_revision"], FROZEN_BASE_REVISION)
        self.assertEqual(tuple(base["selected_epochs_by_seed"]), EXPECTED_BASE_EPOCHS)
        self.assertEqual(
            tuple(base["oof_report"]["seed_best_raw_ratios"]),
            EXPECTED_BASE_OOF_RATIOS,
        )
        observed = tuple(
            (
                item["seed"],
                item["selected_epoch"],
                item["sha256"],
                item["model_state_sha256"],
            )
            for item in base["checkpoints"]
        )
        self.assertEqual(observed, EXPECTED_CHECKPOINTS)
        self.assertFalse(base["all_parameters_require_grad"])
        self.assertFalse(base["optimizer_may_contain_base_parameter"])
        self.assertTrue(base["training_pre_post_checkpoint_bytes_must_match"])
        self.assertTrue(base["zero_residual_must_replay_canonical_independent_selector"])

    def test_residual_target_has_same_units_and_is_not_overclaimed(self) -> None:
        science = self.contract.data["scientific_contract"]
        self.assertEqual(
            science["pair_correction_target"],
            "r_star_m_ij=U({i,j})/D(empty)-(g_m_i+g_m_j)",
        )
        self.assertEqual(science["target_eligibility"], "D(empty)>1e-12")
        self.assertEqual(science["target_name"], "set_utility_correction_residual")
        self.assertFalse(
            science["target_is_claimed_to_be_pure_second_order_interaction"]
        )
        self.assertTrue(science["target_may_absorb_frozen_base_prediction_error"])

    def test_model_is_nested_at_epoch_zero(self) -> None:
        model = self.contract.data["model_contract"]
        self.assertEqual(model["event_embedding_dimension"], 88)
        self.assertEqual(model["pair_features"]["dimension"], 416)
        self.assertEqual(
            model["pair_features"]["concatenation"],
            ["z_i", "z_j", "z_i_times_z_j", "abs_z_i_minus_z_j", "q64"],
        )
        self.assertEqual(model["trainable_parameter_count"], 26753)
        self.assertEqual(model["final_linear_weight_initialization"], "all_zeros")
        self.assertEqual(model["final_linear_bias_initialization"], "all_zeros")
        self.assertEqual(model["epoch_zero_output"], "exact_zero_for_every_pair")
        training = self.contract.training
        self.assertTrue(training["epoch_zero_is_a_model_selection_candidate"])
        self.assertTrue(training["epoch_zero_is_exact_frozen_base"])

    def test_fold_specific_base_replay_prevents_full_fit_leakage(self) -> None:
        crossfit = self.contract.data["cross_fitting_contract"]
        self.assertEqual(crossfit["fold_sizes"], [12, 12, 12, 11, 11])
        self.assertEqual(crossfit["base_replay_learning_rate"], 0.0003)
        self.assertEqual(
            tuple(crossfit["base_replay_epochs_by_seed"]), EXPECTED_BASE_EPOCHS
        )
        self.assertTrue(
            crossfit[
                "base_replay_must_reproduce_five_historical_raw_oof_ratios_and_mean_within_1e_12"
            ]
        )
        self.assertTrue(crossfit["base_replay_is_frozen_before_residual_optimization"])
        self.assertFalse(crossfit["full_fit_base_may_score_oof_heldout_fold"])
        self.assertFalse(crossfit["heldout_or_fresh_label_may_update_base"])

    def test_training_grid_loss_and_optimizer_scope_are_frozen(self) -> None:
        training = self.contract.training
        self.assertEqual(training["learning_rates_in_grid_order"], [0.0003, 0.001])
        self.assertEqual(training["seeds_in_grid_order"], [0, 1, 2, 3, 4])
        self.assertEqual(training["maximum_epochs"], 500)
        self.assertEqual(training["patience_epochs"], 50)
        self.assertEqual(training["epoch_improvement_epsilon"], 0.0001)
        self.assertEqual(training["loss"]["ranking_weight"], 0.25)
        self.assertEqual(training["loss"]["singleton_regression_weight"], 0.0)
        self.assertEqual(training["optimizer"]["parameter_scope"], "residual_head_only")

    def test_safe_guard_is_inherited_without_new_threshold(self) -> None:
        selection = self.contract.selection
        self.assertEqual(
            selection["variants"],
            [
                "frozen_base",
                "unguarded_frozen_base_residual",
                "safe_frozen_base_residual",
            ],
        )
        self.assertEqual(selection["primary_selector"], "safe_frozen_base_residual")
        self.assertEqual(selection["safe_vote_threshold"], 4)
        self.assertEqual(selection["safe_vote_total"], 5)
        self.assertEqual(selection["safe_positive_margin_threshold"], 4)
        self.assertEqual(selection["tie_epsilon"], 0.0)
        self.assertEqual(
            selection["fallback"],
            "canonical_five_seed_mean_frozen_independent_selection",
        )

    def test_fresh_repeated_read_accounting_is_explicit_and_not_global(self) -> None:
        firewall = self.contract.data["access_firewall"]
        evaluate = firewall["evaluate_maximum_new_counts"]
        self.assertEqual(evaluate["fresh16_label_access_claim"], 1)
        self.assertEqual(evaluate["fresh16_label_semantic_decode_attempt"], 1)
        self.assertEqual(evaluate["fresh16_decoded_state"], 48)
        self.assertEqual(evaluate["fresh16_development_join"], 1)
        self.assertEqual(evaluate["fresh16_development_report"], 1)
        self.assertTrue(
            all(
                value == 0
                for value in firewall[
                    "post_report_validation_maximum_new_counts"
                ].values()
            )
        )
        historical = firewall["historical_accounting"]
        self.assertEqual(historical["v1_primary_unit"], "decoded_state_rows")
        self.assertEqual(historical["v1_primary_value"], 48)
        self.assertEqual(historical["v3_parent_semantic_decode_attempt_count"], 2)
        self.assertEqual(
            historical["v3_to_v4_lineage_semantic_decode_attempt_count_after_success"],
            3,
        )
        self.assertFalse(historical["lineage_totals_may_be_called_project_global"])

    def test_label_blind_state_order_and_report_only_validation_are_frozen(self) -> None:
        machine = self.contract.data["execution_state_machine"]
        self.assertEqual(tuple(machine["ordered_states"]), EXPECTED_STATES)
        self.assertLess(
            machine["ordered_states"].index("label_blind_seal"),
            machine["ordered_states"].index(
                "consumed_development_label_access_claim"
            ),
        )
        self.assertLess(
            machine["ordered_states"].index(
                "consumed_development_label_access_claim"
            ),
            machine["ordered_states"].index("single_fresh_label_decode_and_join"),
        )
        self.assertEqual(machine["label_blind_seal"]["status"], LABEL_BLIND_SEAL_STATUS)
        self.assertTrue(machine["validate_reads_report_only"])
        self.assertEqual(
            machine["ordered_states"][-1],
            "report_only_self_consistency_validation",
        )

        output = self.contract.data["output_contract"]
        self.assertTrue(
            output[
                "tag_resolution_and_fresh_download_hash_replay_required_before_git_completion"
            ]
        )

    def test_route_keeps_all_five_checks_and_cannot_open_confirm(self) -> None:
        evaluation = self.contract.data["evaluation_contract"]
        rule = evaluation["development_interpretation_rule"]
        self.assertTrue(rule["all_conditions_required_for_promising"])
        self.assertEqual(rule["minimum_mean_normalized_delta"], 0.01)
        self.assertEqual(
            rule["normalized_paired_bootstrap_lower_strictly_greater_than"], 0.0
        )
        self.assertEqual(rule["mean_raw_delta_strictly_greater_than"], 0.0)
        self.assertEqual(
            rule["n4_mean_normalized_delta_strictly_greater_than"], 0.0
        )
        self.assertEqual(rule["minimum_positive_trajectory_count"], 8)
        self.assertFalse(evaluation["result_may_automatically_open_confirm20"])
        self.assertFalse(evaluation["same_formal58_fresh16_may_support_another_v5_retune"])
        self.assertEqual(
            set(evaluation["forbidden_interpretations"]),
            {"GO", "CONFIRMED", "HELDOUT_PASS", "TEST_PASS"},
        )

    def test_source_operations_confirm_policy_and_gpu_stay_zero(self) -> None:
        source_counts = self.contract.data["access_firewall"]["source_a_counts"]
        self.assertTrue(all(value == 0 for value in source_counts.values()))
        operations = self.contract.data["source_only_operation_contract"]
        self.assertTrue(all(value == 0 for value in operations.values()))
        authorization = self.contract.data["authorization"]
        for field in (
            "fresh16_may_authorize_confirm",
            "confirm20_access_authorized",
            "legacy_dev5_access_authorized",
            "matched_nll_authorized",
            "closed_loop_authorized",
            "gpu_authorized",
            "policy_forward_authorized",
            "mainline_or_v3_artifact_mutation_authorized",
        ):
            self.assertFalse(authorization[field], field)

    def test_scientific_or_firewall_mutations_fail_closed(self) -> None:
        changed = copy.deepcopy(self.contract.data)
        changed["scientific_contract"]["pair_correction_target"] = "raw_residual"
        with self.assertRaisesRegex(ValueError, "pair correction target"):
            validate_contract_data(changed)

        changed = copy.deepcopy(self.contract.data)
        changed["cross_fitting_contract"][
            "full_fit_base_may_score_oof_heldout_fold"
        ] = True
        with self.assertRaisesRegex(ValueError, "OOF leakage"):
            validate_contract_data(changed)

        changed = copy.deepcopy(self.contract.data)
        changed["selection_contract"]["safe_vote_threshold"] = 3
        with self.assertRaisesRegex(ValueError, "safe vote"):
            validate_contract_data(changed)

        changed = copy.deepcopy(self.contract.data)
        changed["access_firewall"]["evaluate_maximum_new_counts"][
            "fresh16_label_semantic_decode_attempt"
        ] = 2
        with self.assertRaisesRegex(ValueError, "semantic_decode_attempt"):
            validate_contract_data(changed)

        changed = copy.deepcopy(self.contract.data)
        changed["authorization"]["confirm20_access_authorized"] = True
        with self.assertRaisesRegex(ValueError, "confirm20"):
            validate_contract_data(changed)

    def test_placeholder_loads_for_tests_but_cannot_validate_source_a(self) -> None:
        if contract_module.FROZEN_CONFIG_SHA256 != FROZEN_CONFIG_SHA256_PLACEHOLDER:
            self.skipTest("Source-A config SHA has already been mechanically frozen")
        self.assertFalse(self.contract.mechanically_frozen)
        with self.assertRaisesRegex(ValueError, "not been mechanically frozen"):
            validate_source_a(repository_root=ROOT)

    def test_canonical_execution_b_payload_after_mechanical_hash_freeze(self) -> None:
        digest = sha256_bytes(CONFIG.read_bytes())
        with patch.object(contract_module, "FROZEN_CONFIG_SHA256", digest):
            frozen = load_frozen_set_conditioned_v4_contract(repository_root=ROOT)
            payload = execution_b_freeze_payload(
                frozen,
                source_a_git_commit="1" * 40,
                source_inventory_sha256="2" * 64,
            )
        self.assertEqual(payload["runner_freeze_path"], EXECUTION_B_RUNNER_FREEZE_PATH)
        self.assertEqual(payload["runtime_device"], "cpu")
        self.assertEqual(payload["gpu_count"], 0)
        self.assertEqual(payload["frozen_base_revision"], FROZEN_BASE_REVISION)
        self.assertEqual(
            payload["fresh16_maximum_new_semantic_decode_attempt_count"], 1
        )
        self.assertEqual(payload["confirm20_access_count"], 0)

    def test_frozen_loader_rejects_byte_drift_after_hash_freeze(self) -> None:
        digest = sha256_bytes(CONFIG.read_bytes())
        changed = _data()
        changed["training_contract"]["patience_epochs"] = 49
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with patch.object(contract_module, "FROZEN_CONFIG_SHA256", digest):
                with self.assertRaisesRegex(ValueError, "contract hash drifted"):
                    load_frozen_set_conditioned_v4_contract(
                        path,
                        repository_root=ROOT,
                    )


if __name__ == "__main__":
    unittest.main()
