import json
import unittest

from scripts.run_gui_owl_androidworld_validation import (
    aggregate_early_stopped_validation,
    aggregate_validation,
    build_run_contract,
    build_assignments,
    episode_filename,
    success_gate_is_mathematically_impossible,
    validate_resume_checkpoint,
)


class GUIOwlAndroidWorldValidationTest(unittest.TestCase):
    def test_resume_checkpoint_requires_exact_run_contract(self) -> None:
        instance = {"task_type": "Task", "task_index": 0}
        runtime_metadata = {
            "snapshot": {"repo": "owner/model", "revision": "abc", "files": []},
            "model_class": "Model",
            "processor_class": "Processor",
            "dtype": "bfloat16",
            "torch_version": "2.0",
            "transformers_version": "5.0",
            "visual_preprocessing": {"mode": "model_default"},
        }
        contract = build_run_contract(
            git_commit="a" * 40,
            plan={"instance_records_sha256": "plan"},
            runtime_metadata=runtime_metadata,
            maximum_visible_images=5,
            max_new_tokens=256,
            server_image="server:image",
            server_image_sha256="image-sha",
        )
        episode = {
            "plan_index": 0,
            "instance": instance,
            "run_contract": contract,
            "environment_runtime": {
                "base_url": "http://worker",
                "server_image": "server:image",
                "server_image_sha256": "image-sha",
            },
        }
        validate_resume_checkpoint(
            episode,
            plan_index=0,
            instance=instance,
            base_url="http://worker",
            run_contract=contract,
        )

        different_contract = json.loads(json.dumps(contract))
        different_contract["model"]["revision"] = "different"
        with self.assertRaisesRegex(ValueError, "run contract"):
            validate_resume_checkpoint(
                episode,
                plan_index=0,
                instance=instance,
                base_url="http://worker",
                run_contract=different_contract,
            )

    def test_early_stop_boundary_is_strictly_below_required_successes(self) -> None:
        self.assertFalse(
            success_gate_is_mathematically_impossible(
                planned_count=62,
                checkpoint_count=47,
                official_successes=16,
                minimum_official_success=0.5,
            )
        )
        self.assertTrue(
            success_gate_is_mathematically_impossible(
                planned_count=62,
                checkpoint_count=47,
                official_successes=15,
                minimum_official_success=0.5,
            )
        )

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

    def test_early_stop_requires_mathematically_impossible_gate(self) -> None:
        instances = [
            {"task_type": f"Task{index}", "task_index": 0}
            for index in range(4)
        ]
        plan = {
            "split": "validation",
            "instance_records_sha256": "abc",
            "task_instance_count": 4,
            "instances": instances,
        }
        episodes = [
            {
                "plan_index": index,
                "instance": instances[index],
                "run_status": "complete",
                "model_step_count": 1,
                "parse_success_count": 1,
                "official_success": False,
                "termination_reason": "policy_terminated",
            }
            for index in range(3)
        ]
        summary = aggregate_early_stopped_validation(
            plan=plan,
            episodes=episodes,
            minimum_parse_coverage=0.95,
            minimum_official_success=0.5,
        )
        self.assertEqual(summary["maximum_possible_official_success_count"], 1)
        self.assertEqual(summary["minimum_required_official_success_count"], 2)
        self.assertFalse(summary["gates"]["validation_gate_passed"])

        episodes[0]["official_success"] = True
        with self.assertRaisesRegex(ValueError, "not yet mathematically impossible"):
            aggregate_early_stopped_validation(
                plan=plan,
                episodes=episodes,
                minimum_parse_coverage=0.95,
                minimum_official_success=0.5,
            )


if __name__ == "__main__":
    unittest.main()
