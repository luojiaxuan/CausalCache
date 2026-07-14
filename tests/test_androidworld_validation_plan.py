import unittest

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


if __name__ == "__main__":
    unittest.main()
