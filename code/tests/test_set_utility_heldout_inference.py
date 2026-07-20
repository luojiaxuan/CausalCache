from __future__ import annotations

import unittest

from causalcache.set_utility_heldout_inference import (
    beam_budget_path,
    conditional_greedy_budget_path,
    merge_model_selection_payloads,
    ocr_rgb_budget_selections,
    random_budget_selections,
    recent_budget_selections,
)


class HeldoutInferenceTest(unittest.TestCase):
    def test_baselines_are_nested_and_budget_separated(self) -> None:
        events = (1, 2, 3, 4, 5)
        self.assertEqual(recent_budget_selections(events)["2"], [4, 5])
        self.assertEqual(
            ocr_rgb_budget_selections(events, (0.1, 0.9, 0.0, 0.8, 0.2))["3"],
            [2, 4, 5],
        )
        left = random_budget_selections(events, state_id="state", seed=7)
        right = random_budget_selections(events, state_id="state", seed=7)
        self.assertEqual(left, right)
        self.assertTrue(set(left["1"]).issubset(left["4"]))

    def test_greedy_runs_one_path_and_stops_at_nonpositive_extension(self) -> None:
        utility = {
            (): 0.0,
            (1,): 2.0,
            (2,): 1.0,
            (3,): 0.5,
            (1, 2): 3.0,
            (1, 3): 2.5,
            (1, 2, 3): 2.9,
        }
        selections, values, count = conditional_greedy_budget_path(
            (1, 2, 3),
            score_batch=lambda subsets: (utility[subset] for subset in subsets),
        )
        self.assertEqual(selections, {"1": [1], "2": [1, 2], "3": [1, 2], "4": [1, 2]})
        self.assertEqual(values["4"], 3.0)
        self.assertEqual(count, 7)

    def test_beam_supports_nonnested_budget_optima_and_exposes_bases(self) -> None:
        utilities = {
            (): 0.0,
            (1,): 10.0,
            (2,): 1.0,
            (3,): 1.0,
            (1, 2): 10.5,
            (1, 3): 10.5,
            (2, 3): 20.0,
            (1, 2, 3): 20.0,
        }
        selections, values, count, trace = beam_budget_path(
            (1, 2, 3),
            score_batch=lambda subsets: [utilities[subset] for subset in subsets],
            width=3,
        )
        self.assertEqual(selections["1"], [1])
        self.assertEqual(selections["2"], [2, 3])
        self.assertEqual(values["2"], 20.0)
        self.assertEqual(trace[1]["base_subsets"], [[1], [2], [3]])
        self.assertGreater(count, 1)

    def test_merge_requires_identical_baselines_and_adds_models(self) -> None:
        def payload(name: str, selected: list[int]) -> dict[str, object]:
            result: dict[str, object] = {
                "cache_content_sha256": "cache",
                "checkpoint_sha256": f"checkpoint-{name}",
                "config_sha256": "config",
                "content_sha256": f"content-{name}",
                "input_content_sha256": "input",
                "inventory_sha256": "inventory",
                "model_name": name,
                "status": "COMPLETED_SET_UTILITY_HELDOUT_MODEL_SELECTIONS",
                "variant": f"variant-{name}",
                "records": [
                    {
                        "candidate_event_ids": [1, 2],
                        "latency_ms": {"search": 1.0},
                        "logical_shard": 3,
                        "methods": {
                            name: {str(b): selected for b in range(1, 5)},
                            "recent": {str(b): [2] for b in range(1, 5)},
                            "ocr_rgb": {str(b): [1] for b in range(1, 5)},
                            "random": {str(b): [1] for b in range(1, 5)},
                        },
                        "predicted_utilities": {str(b): 1.0 for b in range(1, 5)},
                        "state_id": "trajectory:decision:006",
                        "subset_score_count": 3,
                        "tracks": ["exact_oracle"],
                        "trajectory_id": "trajectory",
                    }
                ],
            }
            return result

        merged = merge_model_selection_payloads(
            (payload("deepsets", [1]), payload("set_transformer", [2]))
        )
        self.assertEqual(merged["status"], "SEALED_SET_UTILITY_HELDOUT_SELECTIONS")
        self.assertEqual(
            set(merged["records"][0]["methods"]),
            {"deepsets", "set_transformer", "recent", "ocr_rgb", "random"},
        )


if __name__ == "__main__":
    unittest.main()
