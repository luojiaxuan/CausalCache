import json
import tempfile
import unittest
from pathlib import Path

from causalcache.subset_search_ablation import (
    SubsetSearchAblationConfig,
    build_scale_scenarios,
    evaluate_scenario,
    load_controlled_scenarios,
    load_phase0_scenario,
    load_real_table_scenarios,
    verify_input_bindings,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
CONFIG_PATH = CODE_ROOT / "configs" / "subset_search_ablation_v1.json"


def method(result: dict, name: str) -> dict:
    return next(item for item in result["methods"] if item["method"] == name)


class SubsetSearchAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = SubsetSearchAblationConfig.load(CONFIG_PATH)

    def test_config_and_input_hashes_are_frozen(self) -> None:
        bindings = verify_input_bindings(REPOSITORY_ROOT, self.config)
        self.assertEqual(
            bindings["real_policy_coalition_table"]["sha256"],
            "3679d990166c6907e20f1eb141152fee221fcb2bf72f5392f6c8210a505dbed2",
        )
        self.assertFalse(self.config.claim_boundary["can_reopen_v2_1_no_go"])
        self.assertFalse(self.config.claim_boundary["learned_gate_evaluated"])

    def test_config_rejects_relaxed_claim_boundary(self) -> None:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        data["claim_boundary"]["can_reopen_v2_1_no_go"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "claim boundary"):
                SubsetSearchAblationConfig.load(path)

    def test_controlled_complementarity_trap_separates_searchers(self) -> None:
        scenarios = load_controlled_scenarios(
            REPOSITORY_ROOT / self.config.inputs["controlled_scenarios"].path
        )
        scenario = next(item for item in scenarios if item.scenario_id == "controlled_complementary_trap")
        result = evaluate_scenario(scenario, self.config)
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(method(result, "exact_subset")["selected_coalition"], [2, 3])
        self.assertEqual(method(result, "true_conditional_greedy_raw_gain")["selected_coalition"], [0, 1])
        self.assertEqual(method(result, "true_conditional_greedy_raw_gain")["utility_ratio_to_exact_subset"], 0.5)
        self.assertEqual(method(result, "true_conditional_greedy_raw_gain_exchange_2x2")["selected_coalition"], [2, 3])
        self.assertEqual(method(result, "true_utility_beam_2")["utility_ratio_to_exact_subset"], 0.5)
        self.assertEqual(method(result, "true_utility_beam_4")["utility_ratio_to_exact_subset"], 1.0)

    def test_phase0_names_projection_gap_separately(self) -> None:
        scenario = load_phase0_scenario(REPOSITORY_ROOT / self.config.inputs["phase0_scenario"].path)
        result = evaluate_scenario(scenario, self.config)
        static = method(result, "exact_average_marginal_positive_value_knapsack")
        greedy = method(result, "true_conditional_greedy_raw_gain")
        self.assertEqual(static["interpretation"], "projection_objective_gap_not_greedy_search_gap")
        self.assertAlmostEqual(static["utility_ratio_to_exact_subset"], 0.858594, places=6)
        self.assertEqual(greedy["utility_ratio_to_exact_subset"], 1.0)

    def test_real_cached_tables_are_complete_and_greedy_optimal(self) -> None:
        scenarios = load_real_table_scenarios(
            REPOSITORY_ROOT / self.config.inputs["real_policy_coalition_table"].path,
            REPOSITORY_ROOT / self.config.inputs["real_policy_compact_summary"].path,
            self.config,
        )
        self.assertEqual(len(scenarios), 4)
        for scenario in scenarios:
            result = evaluate_scenario(scenario, self.config)
            exact = method(result, "exact_subset")
            greedy = method(result, "true_conditional_greedy_raw_gain")
            self.assertEqual(greedy["selected_coalition"], exact["selected_coalition"])
            self.assertEqual(greedy["utility_ratio_to_exact_subset"], 1.0)
            self.assertEqual(result["metadata"]["new_policy_forwards"], 0)
        step8_budget1024 = next(
            scenario for scenario in scenarios if scenario.scenario_id == "real_v1_step_8_budget_1024"
        )
        interaction = evaluate_scenario(step8_budget1024, self.config)["interaction_at_empty"]
        self.assertEqual(interaction["positive_pair_count"], 2)
        self.assertEqual(interaction["negative_pair_count"], 19)

    def test_scale_sweep_freezes_feasible_counts(self) -> None:
        scenarios = build_scale_scenarios(self.config)
        self.assertEqual([scenario.scenario_id for scenario in scenarios], [
            "scale_n8_slots3",
            "scale_n12_slots3",
            "scale_n16_slots3",
            "scale_n24_slots3",
        ])
        counts = [
            evaluate_scenario(scenario, self.config)["feasible_coalition_count"]
            for scenario in scenarios[:3]
        ]
        self.assertEqual(counts, [93, 299, 697])


if __name__ == "__main__":
    unittest.main()
