import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_androidworld_validation_plan import records_sha256


class AndroidWorldValidationPlanTest(unittest.TestCase):
    def test_records_hash_is_deterministic_and_content_sensitive(self) -> None:
        records = [
            {
                "task_type": "SystemWifiTurnOn",
                "task_index": 0,
                "goal": "Turn wifi on.",
                "max_steps": 10,
            }
        ]
        self.assertEqual(records_sha256(records), records_sha256(list(records)))
        changed = [dict(records[0], max_steps=20)]
        self.assertNotEqual(records_sha256(records), records_sha256(changed))

    def test_committed_plan_matches_frozen_validation_split(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        partition = json.loads(
            (project_root / "configs/androidworld_task_partition.json").read_text()
        )
        plan = json.loads(
            (project_root / "configs/androidworld_validation_plan.json").read_text()
        )
        self.assertEqual(plan["split"], "validation")
        self.assertEqual(plan["task_instance_count"], 62)
        self.assertEqual(
            plan["instance_records_sha256"], records_sha256(plan["instances"])
        )
        counts = Counter(item["task_type"] for item in plan["instances"])
        self.assertEqual(
            set(counts), set(partition["splits"]["validation"]["task_types"])
        )
        self.assertEqual(set(counts.values()), {2})
        for instance in plan["instances"]:
            self.assertEqual(instance["max_steps"], int(10 * instance["complexity"]))


if __name__ == "__main__":
    unittest.main()
