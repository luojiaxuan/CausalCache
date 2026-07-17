from __future__ import annotations

import copy
import unittest
from pathlib import Path

from causalcache.gate_v1_contract import (
    CANONICAL_CONFIG_PATH,
    EXPECTED_ASSIGNMENT_DIGEST,
    EXPECTED_FOLD_DIGESTS,
    EXPECTED_FOLD_SIZES,
    STATUS,
    canonical_json_bytes,
    derive_oof_folds,
    derive_rosters,
    hierarchical_weight_audit,
    independent_weight_audit,
    load_strict_json_object,
    mlp_parameter_count,
    sha256_bytes,
    validate_contract,
    validate_gate_v1_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH
LEGACY = ROOT / "data/manifests/restoration_v2_selection.json"
EXPANSION = (
    ROOT / "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
)


class GateV1ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_strict_json_object(CONFIG)
        cls.legacy = load_strict_json_object(LEGACY)
        cls.expansion = load_strict_json_object(EXPANSION)
        cls.rosters = derive_rosters(cls.legacy, cls.expansion)

    def test_canonical_contract_replays_rosters_models_and_firewalls(self) -> None:
        result = validate_contract(CONFIG, repository_root=ROOT)
        self.assertEqual(result["status"], STATUS)
        self.assertEqual(result["formal_train_trajectory_count"], 58)
        self.assertEqual(result["formal_train_conditional_edge_count"], 1682)
        self.assertEqual(result["combined_development_trajectory_count"], 21)
        self.assertEqual(result["combined_development_conditional_edge_count"], 609)
        self.assertEqual(result["fresh_development_trajectory_count"], 16)
        self.assertEqual(result["fresh_development_conditional_edge_count"], 464)
        self.assertEqual(result["sealed_confirm_trajectory_count"], 20)
        self.assertEqual(result["conditional_parameter_count"], 25409)
        self.assertEqual(result["independent_parameter_count"], 25609)
        self.assertEqual(result["formal_train_independent_target_count"], 522)
        self.assertEqual(result["combined_development_independent_target_count"], 189)
        self.assertEqual(result["fresh_development_independent_target_count"], 144)
        self.assertTrue(result["fresh_development_is_only_formal_go_slice"])
        self.assertFalse(result["confirm_access_authorized"])
        self.assertFalse(result["matched_nll_authorized"])
        self.assertFalse(result["closed_loop_authorized"])
        self.assertEqual(set(result["hierarchical_weight_sums"].values()), {"1/1"})
        self.assertEqual(set(result["independent_weight_sums"].values()), {"1/1"})

    def test_five_fold_assignment_is_exact_and_train_only(self) -> None:
        assignments, folds = derive_oof_folds(self.rosters["formal_train"])
        self.assertEqual(tuple(len(fold) for fold in folds), EXPECTED_FOLD_SIZES)
        self.assertEqual(
            tuple(sha256_bytes(canonical_json_bytes(list(fold))) for fold in folds),
            EXPECTED_FOLD_DIGESTS,
        )
        assignment_payload = [
            {"source_id": source_id, "fold": fold} for source_id, fold in assignments
        ]
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(assignment_payload)),
            EXPECTED_ASSIGNMENT_DIGEST,
        )
        self.assertEqual(
            set(source_id for fold in folds for source_id in fold),
            set(self.rosters["formal_train"]),
        )
        self.assertFalse(
            set(source_id for fold in folds for source_id in fold)
            & set(self.rosters["combined_development"])
        )

    def test_hierarchical_edge_weights_sum_to_one_for_all_formal_rosters(self) -> None:
        for trajectory_count, expected_edges in (
            (10, 290),
            (48, 1392),
            (58, 1682),
            (5, 145),
            (16, 464),
            (21, 609),
        ):
            audit = hierarchical_weight_audit(trajectory_count)
            self.assertEqual(audit["edge_count"], expected_edges)
            self.assertEqual(audit["weight_sum_numerator"], 1)
            self.assertEqual(audit["weight_sum_denominator"], 1)
            independent = independent_weight_audit(trajectory_count)
            self.assertEqual(independent["target_count"], trajectory_count * 9)
            self.assertEqual(independent["weight_sum_numerator"], 1)
            self.assertEqual(independent["weight_sum_denominator"], 1)

    def test_parameter_counts_include_all_biases(self) -> None:
        self.assertEqual(mlp_parameter_count(330, 64), 25409)
        self.assertEqual(mlp_parameter_count(200, 88), 25609)

    def test_projection_normalization_smoke_and_go_thresholds_are_frozen(self) -> None:
        labels = self.config["label_contract"]
        self.assertEqual(
            labels["normalization"],
            "raw_conditional_target/max(D(empty),1e-12)",
        )
        self.assertFalse(labels["target_clipping"])
        self.assertEqual(
            labels["independent_raw_projection"],
            "0.5*(raw_Delta_j(empty)+mean_i_not_j(raw_Delta_j({i})))",
        )
        self.assertEqual(labels["independent_targets_per_state_candidate"], 1)
        self.assertFalse(labels["independent_projection_repeated_on_conditional_edges"])
        self.assertEqual(labels["normalization_eligibility"], "D(empty)>1e-12")
        smoke = self.config["current_trainer_smoke"]
        self.assertEqual(smoke["roster"], "legacy_train")
        self.assertEqual(smoke["seed"], 0)
        self.assertEqual(smoke["maximum_optimization_steps"], 2)
        self.assertEqual(smoke["paper_metric_count"], 0)
        self.assertEqual(smoke["persistent_checkpoint_count"], 0)
        self.assertEqual(smoke["development_semantic_access_count"], 0)
        self.assertIn("negative_target_not_clipped", smoke["allowed_checks"])
        self.assertIn("conditional_rescoring", smoke["allowed_checks"])
        self.assertIn("independent_one_shot", smoke["allowed_checks"])
        self.assertIn(
            "temporary_checkpoint_save_load_exact_score", smoke["allowed_checks"]
        )
        optimization = self.config["optimization_contract"]
        self.assertEqual(optimization["weight_decay"], 0.0001)
        self.assertEqual(optimization["gradient_clipping"], "global_grad_norm_1.0")
        self.assertEqual(self.config["loss_contract"]["label_tie_epsilon"], 1e-12)
        oof = self.config["oof_contract"]
        self.assertEqual(oof["epoch_update_minimum_strict_improvement"], 0.0001)
        self.assertIn("raw_utility", oof["epoch_rollout_metric"])
        self.assertIn("five_seed_mean", oof["learning_rate_selection"])
        inference = self.config["inference_contract"]
        self.assertTrue(
            inference["conditional_rescore_all_remaining_candidates_after_each_selection"]
        )
        self.assertFalse(inference["independent_rescoring_allowed"])
        evaluation = self.config["evaluation_contract"]
        self.assertEqual(evaluation["primary_slice"], "fresh_development")
        self.assertEqual(
            evaluation["go_selector_all"][
                "ensemble_normalized_recovery_over_exact_minimum"
            ],
            0.8,
        )
        self.assertEqual(
            evaluation["go_set_conditioning_all"][
                "conditional_minus_independent_mean_normalized_delta_minimum"
            ],
            0.02,
        )

    def test_any_static_section_or_roster_digest_drift_fails_closed(self) -> None:
        changed_threshold = copy.deepcopy(self.config)
        changed_threshold["evaluation_contract"]["go_selector_all"][
            "ensemble_normalized_recovery_over_exact_minimum"
        ] = 0.79
        with self.assertRaises(ValueError):
            validate_gate_v1_config(changed_threshold, repository_root=ROOT)

        changed_roster = copy.deepcopy(self.config)
        changed_roster["rosters"]["formal_train"]["conditional_edge_count"] = 1681
        with self.assertRaises(ValueError):
            validate_gate_v1_config(changed_roster, repository_root=ROOT)

        relaxed_firewall = copy.deepcopy(self.config)
        relaxed_firewall["prohibited_work"]["matched_nll_evaluation_count"] = 1
        with self.assertRaises(ValueError):
            validate_gate_v1_config(relaxed_firewall, repository_root=ROOT)


if __name__ == "__main__":
    unittest.main()
