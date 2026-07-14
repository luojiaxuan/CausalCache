import unittest

from scripts.run_gui_owl_androidworld_validation import (
    aggregate_validation,
    build_assignments,
    episode_filename,
)


class GUIOwlAndroidWorldValidationTest(unittest.TestCase):
    def test_round_robin_assignment_preserves_plan_indices(self) -> None:
        instances = [
            {"task_type": f"Task{index}", "task_index": 0}
            for index in range(5)
        ]
        assignments = build_assignments(instances, ["http://one", "http://two"])
        self.assertEqual([index for index, _ in assignments[0]], [0, 2, 4])
        self.assertEqual([index for index, _ in assignments[1]], [1, 3])
        self.assertEqual(episode_filename(2, instances[2]), "002-Task2-0.json")

    def test_aggregation_keeps_failures_in_success_denominator(self) -> None:
        plan = {
            "split": "validation",
            "instance_records_sha256": "abc",
            "task_instance_count": 2,
        }
        episodes = [
            {
                "run_status": "complete",
                "model_step_count": 2,
                "parse_success_count": 2,
                "environment_success": True,
                "official_success": True,
                "termination_reason": "policy_terminated",
            },
            {
                "run_status": "exception",
                "model_step_count": 0,
                "parse_success_count": 0,
                "environment_success": False,
                "official_success": False,
                "termination_reason": "exception",
            },
        ]
        summary = aggregate_validation(
            plan=plan,
            episodes=episodes,
            minimum_parse_coverage=0.95,
            minimum_official_success=0.5,
        )
        self.assertEqual(summary["parse_coverage"], 1.0)
        self.assertEqual(summary["official_success_rate"], 0.5)
        self.assertEqual(summary["outcomes"]["infrastructure_failure"], 1)
        self.assertTrue(summary["gates"]["validation_gate_passed"])

    def test_parse_failure_fails_parse_gate(self) -> None:
        plan = {
            "split": "validation",
            "instance_records_sha256": "abc",
            "task_instance_count": 1,
        }
        summary = aggregate_validation(
            plan=plan,
            episodes=[
                {
                    "run_status": "complete",
                    "model_step_count": 1,
                    "parse_success_count": 0,
                    "official_success": False,
                    "environment_success": False,
                    "termination_reason": "parse_error",
                }
            ],
            minimum_parse_coverage=0.95,
            minimum_official_success=0.5,
        )
        self.assertFalse(summary["gates"]["parse_gate_passed"])
        self.assertEqual(summary["outcomes"], {"parse_failure": 1})


if __name__ == "__main__":
    unittest.main()
