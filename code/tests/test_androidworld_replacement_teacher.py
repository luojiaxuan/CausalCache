import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class AndroidWorldReplacementTeacherTest(unittest.TestCase):
    def setUp(self) -> None:
        config_dir = REPOSITORY_ROOT / "code" / "configs"
        self.replacement = json.loads(
            (config_dir / "androidworld_replacement_teacher_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.think_snapshot = json.loads(
            (config_dir / "gui_owl_1_5_8b_think_snapshot.json").read_text(
                encoding="utf-8"
            )
        )
        self.instruct_snapshot = json.loads(
            (config_dir / "gui_owl_1_5_8b_snapshot.json").read_text(encoding="utf-8")
        )

    def test_candidate_and_snapshot_revision_match(self) -> None:
        candidate = self.replacement["candidate"]
        self.assertEqual(candidate["repo"], self.think_snapshot["repo"])
        self.assertEqual(candidate["revision"], self.think_snapshot["revision"])
        self.assertEqual(len(self.think_snapshot["files"]), 14)

    def test_runtime_files_match_instruct_and_weights_differ(self) -> None:
        think_files = {item["path"]: item for item in self.think_snapshot["files"]}
        instruct_files = {item["path"]: item for item in self.instruct_snapshot["files"]}
        self.assertEqual(set(think_files), set(instruct_files))
        for path in think_files:
            if path.endswith(".safetensors") and path.startswith("model-"):
                self.assertNotEqual(
                    think_files[path]["sha256"], instruct_files[path]["sha256"]
                )
            else:
                self.assertEqual(think_files[path], instruct_files[path])

    def test_gate_reuses_frozen_validation_contract(self) -> None:
        gate = self.replacement["validation_gate"]
        self.assertEqual(gate["plan_instance_count"], 62)
        self.assertEqual(gate["minimum_action_parse_coverage"], 0.95)
        self.assertEqual(gate["minimum_task_success"], 0.5)
        self.assertTrue(gate["test_partition_must_remain_sealed"])
        self.assertTrue(
            self.replacement["change_control"][
                "no_prompt_equivalence_threshold_or_visual_preprocessing_changes_after_validation_starts"
            ]
        )

    def test_smoke_uses_true_five_image_history_and_frozen_generation(self) -> None:
        smoke_gate = self.replacement["smoke_gate"]
        generation = self.replacement["frozen_interface"]["generation"]
        self.assertEqual(smoke_gate["decision_step_id"], 6)
        self.assertEqual(generation["max_new_tokens"], 256)
        self.assertFalse(generation["do_sample"])

    def test_smoke_result_is_interface_only(self) -> None:
        result = self.replacement["smoke_gate"]["result"]
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["scope"], "interface_only_not_policy_coverage")
        self.assertEqual(result["parsed_variants"], result["required_variants"])

    def test_validation_failure_rejects_candidate(self) -> None:
        result = self.replacement["validation_gate"]["result"]
        self.assertEqual(
            self.replacement["status"],
            "rejected_by_androidworld_validation_gate",
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["artifact_status"],
            "valid_early_stopped_policy_rejection",
        )
        self.assertGreaterEqual(
            result["parse_coverage"],
            self.replacement["validation_gate"]["minimum_action_parse_coverage"],
        )
        self.assertLess(
            result["maximum_possible_official_success_rate"],
            self.replacement["validation_gate"]["minimum_task_success"],
        )
        self.assertEqual(
            result["checkpoint_count"] + result["unobserved_instance_count"],
            result["plan_instance_count"],
        )

        summary = json.loads(
            (REPOSITORY_ROOT / result["summary"]).read_text(encoding="utf-8")
        )
        for key in (
            "artifact_status",
            "checkpoint_count",
            "plan_instance_count",
            "unobserved_instance_count",
            "parse_success_count",
            "model_step_count",
            "parse_coverage",
            "official_success_count",
            "official_success_rate_lower_bound",
            "maximum_possible_official_success_count",
            "maximum_possible_official_success_rate",
            "early_stop_reason",
        ):
            self.assertEqual(result[key], summary[key])
        self.assertEqual(result["run_git_commit"], summary["run_contract"]["git_commit"])

    def test_validation_artifact_provenance_is_unambiguous(self) -> None:
        result_dir = (
            REPOSITORY_ROOT
            / "data"
            / "results"
            / "gui_owl_1_5_8b_think_androidworld_validation"
        )
        run_manifest = json.loads(
            (result_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        container = run_manifest["container"]
        self.assertEqual(
            container["digest"],
            "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa",
        )
        self.assertEqual(
            container["image_id"],
            "sha256:81b5df11b32ad8460be270a67066196cb7c6d4fb92cb5d05a44fb06d1ec88d21",
        )

        dataset_manifest = json.loads(
            (result_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        self.assertIn("canonical_revision", dataset_manifest["manifest_scope"])
        self.assertIn("hf_payload_layout", dataset_manifest)
        self.assertIn("hf_payload_files", dataset_manifest)
        self.assertNotIn("files", dataset_manifest)


if __name__ == "__main__":
    unittest.main()
