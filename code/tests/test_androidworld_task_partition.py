import json
import unittest
from pathlib import Path

from scripts.build_androidworld_task_partition import (
    build_manifest,
    registry_sha256,
    task_bucket,
)


class AndroidWorldTaskPartitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = {
            "benchmark": {"canonical_repo": "fixture", "canonical_revision": "abc"},
            "task_partition": {
                "assignment": "int(sha256(task_name).hexdigest(), 16) % 10",
                "train_buckets": [0, 1, 2, 3, 4, 5],
                "validation_buckets": [6, 7],
                "test_buckets": [8, 9],
                "train_suite_seed": 1,
                "validation_suite_seed": 2,
                "test_suite_seed": 3,
                "train_task_combinations": 3,
                "validation_task_combinations": 2,
                "test_task_combinations": 3,
            },
        }

    def test_manifest_is_order_invariant_and_covers_all_tasks(self) -> None:
        names = ["TaskZulu", "TaskAlpha", "TaskBeta"]
        forward = build_manifest(names, self.stack)
        reverse = build_manifest(list(reversed(names)), self.stack)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["registry"]["task_type_count"], 3)
        self.assertEqual(forward["registry"]["sorted_task_types_sha256"], registry_sha256(names))
        self.assertEqual({task["task_type"] for task in forward["tasks"]}, set(names))
        for task in forward["tasks"]:
            self.assertEqual(task["bucket"], task_bucket(task["task_type"]))

    def test_duplicate_task_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_manifest(["TaskAlpha", "TaskAlpha"], self.stack)

    def test_overlapping_bucket_is_rejected(self) -> None:
        self.stack["task_partition"]["validation_buckets"] = [5, 6, 7]
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            build_manifest(["TaskAlpha"], self.stack)

    def test_committed_manifest_rebuilds_from_its_task_types(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        stack = json.loads(
            (project_root / "configs/androidworld_stack.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = json.loads(
            (project_root / "configs/androidworld_task_partition.json").read_text(
                encoding="utf-8"
            )
        )
        task_types = [task["task_type"] for task in manifest["tasks"]]
        self.assertEqual(build_manifest(task_types, stack), manifest)


if __name__ == "__main__":
    unittest.main()
