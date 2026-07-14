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


if __name__ == "__main__":
    unittest.main()
