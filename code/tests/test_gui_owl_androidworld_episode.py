import unittest
import urllib.error
from io import BytesIO

from scripts.run_gui_owl_androidworld_episode import (
    apply_answer_followup_override,
    http_error_record,
    select_instance,
)


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

    def test_http_error_record_preserves_executor_response(self) -> None:
        error = urllib.error.HTTPError(
            "http://example/execute_action",
            500,
            "Internal Server Error",
            {},
            BytesIO(b'{"detail":"bad action"}'),
        )
        try:
            self.assertEqual(
                http_error_record(error),
                {
                    "status_code": 500,
                    "reason": "Internal Server Error",
                    "response_body": '{"detail":"bad action"}',
                },
            )
        finally:
            error.close()

    def test_answer_forces_next_generated_action_to_terminate(self) -> None:
        generated = {"action_type": "answer", "text": "done"}
        effective, overridden = apply_answer_followup_override(
            generated,
            previous_executed_action={"action_type": "answer", "text": "done"},
        )
        self.assertTrue(overridden)
        self.assertEqual(
            effective,
            {"action_type": "status", "goal_status": "task_complete"},
        )
        unchanged, overridden = apply_answer_followup_override(
            generated,
            previous_executed_action={"action_type": "click", "x": 1, "y": 2},
        )
        self.assertFalse(overridden)
        self.assertEqual(unchanged, generated)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_instance(
                {"split": "validation", "instances": []},
                task_type="Example",
                task_index=0,
            )


if __name__ == "__main__":
    unittest.main()
