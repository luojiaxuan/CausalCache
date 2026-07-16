from __future__ import annotations

import copy
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_selector_geometry_contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CANONICAL_CONFIG_PATH,
    EXPECTED_STATE_COUNT,
    EXPECTED_TRAJECTORY_COUNT,
    FROZEN_CONFIG_SHA256,
    PRIMARY_BUDGET,
    PRIMARY_EVENT_COUNT,
    RestorationV22SelectorGeometryContract,
    load_strict_json_object,
    sha256_file,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


class RestorationV22SelectorGeometryContractTest(unittest.TestCase):
    def _contract(self, data=None) -> RestorationV22SelectorGeometryContract:
        return RestorationV22SelectorGeometryContract(
            path=CONFIG,
            data=load_strict_json_object(CONFIG) if data is None else data,
            sha256=FROZEN_CONFIG_SHA256,
        )

    def test_frozen_contract_and_cli_reducer_boundary_validate(self) -> None:
        self.assertEqual(sha256_file(CONFIG), FROZEN_CONFIG_SHA256)
        self._contract().validate(repository_root=ROOT)
        result = validate_contract(CONFIG, repository_root=ROOT)
        self.assertEqual(result["fixed_state_count"], EXPECTED_STATE_COUNT)
        self.assertEqual(
            result["fixed_trajectory_count"], EXPECTED_TRAJECTORY_COUNT
        )
        self.assertEqual(
            result["primary_candidate_event_count"], PRIMARY_EVENT_COUNT
        )
        self.assertEqual(result["primary_budget_event_capacity"], PRIMARY_BUDGET)
        self.assertEqual(result["bootstrap_replicates"], BOOTSTRAP_REPLICATES)
        self.assertEqual(result["bootstrap_seed"], BOOTSTRAP_SEED)
        self.assertTrue(result["confirm_locked"])
        self.assertTrue(result["sealed_test_locked"])
        self.assertTrue(result["gate_training_locked"])

    def test_immutable_label_and_derived_artifact_identities_are_exact(self) -> None:
        data = load_strict_json_object(CONFIG)
        labels = data["immutable_inputs"]["restoration_labels"]
        self.assertEqual(
            labels["hf_revision"],
            "8f6baae5c0b23b08915fa1b0fb848dd519b4c8db",
        )
        self.assertEqual(
            labels["raw_archive_sha256"],
            "99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e",
        )
        derived = data["immutable_inputs"]["derived_dataset"]
        self.assertEqual(
            derived["hf_revision"],
            "89f136abaff797e14fe758a198996e51032a10a6",
        )
        self.assertEqual(
            derived["artifact_tree_sha256"],
            "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e",
        )

    def test_primary_and_secondary_selector_contracts_are_distinct(self) -> None:
        selectors = load_strict_json_object(CONFIG)["selectors"]
        self.assertEqual(
            selectors["exact_subset_oracle"]["eligible_coalitions"],
            "all_S_with_cardinality_at_most_B",
        )
        self.assertEqual(
            selectors["true_conditional_marginal_greedy"]["continue_condition"],
            "strict_gain_greater_than_0_and_cardinality_less_than_B",
        )
        self.assertEqual(
            selectors["budget_conditioned_independent"]["role"],
            "primary_independent_objective_projection_baseline",
        )
        self.assertEqual(
            selectors["full_shapley_independent"]["role"],
            "secondary_independent_ablation",
        )
        self.assertEqual(
            selectors["exact_cardinality_oracle"]["role"],
            "forced_fill_nonmonotonicity_sensitivity",
        )
        for key in (
            "forced_fill_true_conditional_marginal_greedy",
            "forced_fill_budget_conditioned_independent",
            "forced_fill_full_shapley_independent",
        ):
            self.assertEqual(selectors[key]["role"], "early_stop_sensitivity")
        self.assertEqual(
            selectors["ocr_and_rgb_similarity"]["scope"],
            "n_equals_4_B_equals_2_only",
        )
        self.assertFalse(
            selectors["frozen_policy_vision_similarity"]
            ["embedding_cache_present_in_labels_or_derived_artifact"]
        )
        metrics = load_strict_json_object(CONFIG)["primary_metrics"]
        self.assertFalse(metrics["per_state_ratio_is_primary"])
        self.assertEqual(
            metrics["per_state_nonpositive_exact_ratio_rule"],
            "null_and_excluded_from_any_secondary_ratio_summary",
        )

    def test_denominator_selector_bootstrap_or_threshold_drift_fails_closed(self) -> None:
        original = load_strict_json_object(CONFIG)
        mutations = (
            (
                "state denominator",
                lambda value: value["analysis_denominator"].__setitem__(
                    "fixed_state_count", 44
                ),
            ),
            (
                "primary n",
                lambda value: value["analysis_grid"]["primary_setting"].__setitem__(
                    "candidate_event_count", 3
                ),
            ),
            (
                "greedy stop",
                lambda value: value["selectors"]
                ["true_conditional_marginal_greedy"].__setitem__(
                    "continue_condition", "gain_greater_than_or_equal_to_0"
                ),
            ),
            (
                "independent attribution",
                lambda value: value["selectors"]
                ["budget_conditioned_independent"].__setitem__(
                    "role", "full_shapley"
                ),
            ),
            (
                "bootstrap seed",
                lambda value: value["uncertainty"].__setitem__("seed", 271829),
            ),
            (
                "shaping threshold",
                lambda value: value["internal_method_shaping"]
                ["set_conditioning_materiality"].__setitem__(
                    "green_minimum_mean_normalized_recovery_advantage_over_budget_conditioned_independent",
                    0.04,
                ),
            ),
            (
                "interaction strata",
                lambda value: value["interaction_analysis"]
                ["strength_strata"].__setitem__(
                    "cutpoint_source", "fixed_global_thresholds"
                ),
            ),
            (
                "ratio aggregation",
                lambda value: value["primary_metrics"].__setitem__(
                    "ratio_definition", "mean_of_state_ratios"
                ),
            ),
        )
        for label, mutate in mutations:
            data = copy.deepcopy(original)
            mutate(data)
            with self.subTest(label=label), self.assertRaises(ValueError):
                self._contract(data).validate(repository_root=ROOT)

    def test_confirm_test_training_or_policy_vision_forward_unlock_fails(self) -> None:
        original = load_strict_json_object(CONFIG)
        authorization_keys = (
            "gate_training_allowed",
            "matched_nll_allowed",
            "closed_loop_allowed",
            "confirm_access_allowed",
            "sealed_test_access_allowed",
            "policy_vision_feature_extraction_allowed_in_this_stage",
        )
        for key in authorization_keys:
            data = copy.deepcopy(original)
            data["authorization"][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._contract(data).validate(repository_root=ROOT)

        for key in (
            "restoration_teacher_forward_count",
            "kl_measurement_count",
            "gate_model_forward_count",
            "confirm_state_access_count",
            "sealed_test_state_access_count",
            "policy_vision_feature_forward_count",
        ):
            data = copy.deepcopy(original)
            data["operation_ceiling"][key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._contract(data).validate(repository_root=ROOT)

    def test_extra_contract_key_and_binding_revision_drift_fail_closed(self) -> None:
        original = load_strict_json_object(CONFIG)
        data = copy.deepcopy(original)
        data["unregistered_analysis"] = {}
        with self.assertRaisesRegex(ValueError, "key schema"):
            self._contract(data).validate(repository_root=ROOT)

        data = copy.deepcopy(original)
        data["immutable_inputs"]["restoration_labels"]["hf_revision"] = "0" * 40
        with self.assertRaises(ValueError):
            self._contract(data).validate(repository_root=ROOT)


if __name__ == "__main__":
    unittest.main()
