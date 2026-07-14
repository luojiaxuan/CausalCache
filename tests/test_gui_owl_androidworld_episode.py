import unittest

from scripts.run_gui_owl_androidworld_episode import select_instance


class GUIOwlAndroidWorldEpisodeTest(unittest.TestCase):
    def test_selects_exact_validation_instance(self) -> None:
        plan = {
            "split": "validation",
            "instances": [
                {"task_type": "Example", "task_index": 0, "goal": "first"},
                {"task_type": "Example", "task_index": 1, "goal": "second"},
            ],
        }
        self.assertEqual(
            select_instance(plan, task_type="Example", task_index=1)["goal"],
            "second",
        )

    def test_rejects_nonvalidation_or_missing_instance(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation plan"):
            select_instance(
                {"split": "test", "instances": []},
                task_type="Example",
                task_index=0,
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_instance(
                {"split": "validation", "instances": []},
                task_type="Example",
                task_index=0,
            )


if __name__ == "__main__":
    unittest.main()
