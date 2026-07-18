from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from causalcache.set_conditioned_v3_contract import (
    BASE_GIT_COMMIT,
    CANONICAL_CONFIG_PATH,
    EXPECTED_FORMAL_FILES,
    EXPECTED_STATES,
    FRESH16_FEATURE_SHA256,
    FRESH16_LABEL_SHA256,
    FROZEN_CONFIG_SHA256,
    LABEL_BLIND_SEAL_STATUS,
    PROTOCOL_ID,
    REQUIRED_BRANCH,
    load_frozen_set_conditioned_v3_contract,
    validate_contract_data,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


def _data() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


class SetConditionedV3ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_frozen_set_conditioned_v3_contract(
            repository_root=ROOT
        )

    def test_frozen_identity_and_non_mainline_lineage(self) -> None:
        self.assertEqual(self.contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(self.contract.data["protocol_id"], PROTOCOL_ID)
        lineage = self.contract.data["lineage"]
        self.assertEqual(lineage["branch"], REQUIRED_BRANCH)
        self.assertEqual(lineage["base_git_commit"], BASE_GIT_COMMIT)
        self.assertEqual(
            lineage["study_role"], "non_mainline_set_conditioning_exploration"
        )
        self.assertTrue(lineage["v1_failure_is_immutable"])
        self.assertEqual(
            lineage["v1_failure_summary"]["required_outcome"],
            "NO_V2_CONDITIONAL_RESCUE",
        )

    def test_formal_and_consumed_development_bytes_are_exact(self) -> None:
        self.assertEqual(
            tuple(self.contract.formal_input["exact_files"]),
            EXPECTED_FORMAL_FILES,
        )
        development = self.contract.development_input
        by_kind = {item["kind"]: item for item in development["exact_files"]}
        self.assertEqual(
            by_kind["feature_states"]["sha256"], FRESH16_FEATURE_SHA256
        )
        self.assertEqual(by_kind["label_states"]["sha256"], FRESH16_LABEL_SHA256)
        self.assertEqual(by_kind["feature_states"]["access_stage"], "train_seal")
        self.assertEqual(
            by_kind["label_states"]["access_stage"],
            "evaluate_after_label_blind_seal",
        )
        self.assertFalse(development["may_be_called_holdout"])
        self.assertFalse(development["may_be_called_test"])
        self.assertFalse(development["may_authorize_go"])
        self.assertFalse(development["may_tune_after_label_join"])

    def test_pair_residual_is_structural_and_chronologically_canonical(self) -> None:
        science = self.contract.data["scientific_contract"]
        self.assertEqual(science["singleton_target"], "U_i=U({i})")
        self.assertEqual(
            science["pair_residual_target"],
            "R_ij=U({i,j})-U({i})-U({j})",
        )
        self.assertEqual(science["maximum_feasible_subset_count"], 11)
        model = self.contract.data["model_contract"]
        self.assertEqual(model["event_encoder"]["output_dimension"], 64)
        self.assertEqual(model["pair_features"]["dimension"], 320)
        self.assertEqual(
            model["pair_features"]["concatenation"],
            ["z_i", "z_j", "z_i_times_z_j", "abs_z_i_minus_z_j", "q64"],
        )
        self.assertTrue(
            model[
                "chronological_pair_features_are_intentionally_not_permutation_invariant"
            ]
        )
        self.assertFalse(model["iterative_conditional_mlp_reused"])

    def test_train_only_rms_loss_and_n4_model_selection_are_frozen(self) -> None:
        training = self.contract.training
        self.assertEqual(training["learning_rates_in_grid_order"], [0.0003, 0.001])
        self.assertEqual(training["seeds_in_grid_order"], [0, 1, 2, 3, 4])
        self.assertEqual(training["maximum_epochs"], 500)
        self.assertEqual(training["patience_epochs"], 50)
        self.assertEqual(
            training["target_scaling"]["scope"], "fold_training_partition_only"
        )
        self.assertFalse(
            training["target_scaling"]["heldout_or_fresh_values_may_fit_scale"]
        )
        loss = training["loss"]
        self.assertEqual(loss["ranking_weight"], 0.25)
        self.assertEqual(loss["normalized_regression_weight"], 0.0)
        self.assertEqual(loss["extra_sign_loss_weight"], 0.0)
        self.assertEqual(
            training["single_seed_early_stop_and_lr_selection_variant"],
            "unguarded_pair_residual",
        )
        self.assertEqual(
            training["epoch_selection_metric"],
            "minimum_of_n4_raw_utility_over_exact_and_n4_normalized_recovery_over_exact",
        )
        self.assertEqual(
            training["bootstrap"],
            {
                "unit": "trajectory",
                "resamples": 10000,
                "confidence": 0.9,
                "interval": "percentile",
                "seed": 271828,
            },
        )

    def test_safe_selector_requires_four_votes_or_additive_fallback(self) -> None:
        selection = self.contract.selection
        self.assertEqual(
            [item["name"] for item in selection["variants"]],
            ["additive", "unguarded_pair_residual", "safe_pair_residual"],
        )
        self.assertEqual(selection["safe_vote_threshold"], 4)
        self.assertEqual(selection["safe_vote_total"], 5)
        self.assertEqual(selection["safe_positive_margin_threshold"], 4)
        self.assertEqual(
            selection["safe_positive_margin_definition"],
            "individual_seed_F_m(pair_candidate)-F_m(base_candidate)_strictly_greater_than_zero",
        )
        self.assertEqual(
            selection["fallback"], "five_seed_mean_additive_exact_enumeration"
        )
        self.assertEqual(selection["primary_selector"], "safe_pair_residual")
        self.assertTrue(
            selection[
                "safe_selector_does_not_participate_in_single_seed_early_stopping"
            ]
        )
        self.assertEqual(
            selection["safe_trace_required_fields"],
            [
                "pair_candidate",
                "base_candidate",
                "seedwise_unguarded_argmax",
                "seedwise_pair_minus_base_margins",
                "pair_candidate_vote_count",
                "strictly_positive_margin_count",
                "used_pair_candidate",
            ],
        )
        self.assertEqual(
            selection["tie_break"],
            [
                "higher_predicted_utility",
                "lower_cardinality",
                "lexicographically_lower_event_step_ids",
            ],
        )

    def test_label_blind_seal_strictly_precedes_semantic_label_access(self) -> None:
        machine = self.contract.data["execution_state_machine"]
        self.assertEqual(tuple(machine["ordered_states"]), EXPECTED_STATES)
        self.assertLess(
            machine["ordered_states"].index("label_blind_seal"),
            machine["ordered_states"].index("label_access_claim"),
        )
        self.assertLess(
            machine["ordered_states"].index("label_access_claim"),
            machine["ordered_states"].index("one_time_consumed_development_join"),
        )
        self.assertEqual(
            machine["label_blind_seal"]["status"], LABEL_BLIND_SEAL_STATUS
        )
        firewall = self.contract.data["access_firewall"]
        self.assertTrue(
            firewall[
                "label_access_claim_must_precede_label_open_parse_or_semantic_decode"
            ]
        )
        self.assertTrue(firewall["transport_byte_possession_is_not_semantic_access"])
        self.assertTrue(firewall["no_post_label_training_or_prediction_change"])

    def test_confirm_closed_loop_policy_gpu_and_source_operations_stay_zero(self) -> None:
        source_counts = self.contract.data["access_firewall"]["source_a_counts"]
        self.assertTrue(source_counts)
        self.assertTrue(all(value == 0 for value in source_counts.values()))
        operations = self.contract.data["source_only_operation_contract"]
        self.assertTrue(operations)
        self.assertTrue(all(value == 0 for value in operations.values()))
        authorization = self.contract.data["authorization"]
        for field in (
            "confirm20_access_authorized",
            "legacy_dev5_access_authorized",
            "matched_nll_authorized",
            "closed_loop_authorized",
            "gpu_authorized",
            "policy_forward_authorized",
            "mainline_artifact_mutation_authorized",
        ):
            self.assertFalse(authorization[field], field)
        self.assertEqual(self.contract.data["runtime_contract"]["gpu_count"], 0)

    def test_fresh16_interpretation_cannot_be_upgraded_to_go(self) -> None:
        evaluation = self.contract.data["evaluation_contract"]
        self.assertTrue(evaluation["fresh16_is_development_only"])
        self.assertEqual(
            set(evaluation["forbidden_interpretations"]),
            {"GO", "CONFIRMED", "HELDOUT_PASS", "TEST_PASS"},
        )
        changed = copy.deepcopy(self.contract.data)
        changed["fresh16_consumed_development_input"]["may_authorize_go"] = True
        with self.assertRaisesRegex(ValueError, "may_authorize_go"):
            validate_contract_data(changed)

    def test_mutations_of_model_loss_or_firewall_fail_closed(self) -> None:
        changed = copy.deepcopy(self.contract.data)
        changed["model_contract"]["pair_features"]["dimension"] = 256
        with self.assertRaisesRegex(ValueError, "pair feature dimension"):
            validate_contract_data(changed)

        changed = copy.deepcopy(self.contract.data)
        changed["training_contract"]["loss"]["normalized_regression_weight"] = 1.0
        with self.assertRaisesRegex(ValueError, "loss contract"):
            validate_contract_data(changed)

        changed = copy.deepcopy(self.contract.data)
        changed["selection_contract"]["safe_vote_threshold"] = 3
        with self.assertRaisesRegex(ValueError, "safe vote threshold"):
            validate_contract_data(changed)

        changed = copy.deepcopy(self.contract.data)
        changed["access_firewall"][
            "checkpoint_and_prediction_seal_must_precede_label_access_claim"
        ] = False
        with self.assertRaisesRegex(ValueError, "checkpoint_and_prediction"):
            validate_contract_data(changed)

        changed = copy.deepcopy(self.contract.data)
        changed["authorization"]["confirm20_access_authorized"] = True
        with self.assertRaisesRegex(ValueError, "confirm20"):
            validate_contract_data(changed)

    def test_frozen_loader_rejects_byte_drift(self) -> None:
        changed = _data()
        changed["training_contract"]["patience_epochs"] = 49
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract hash drifted"):
                load_frozen_set_conditioned_v3_contract(
                    path,
                    repository_root=ROOT,
                )


if __name__ == "__main__":
    unittest.main()
