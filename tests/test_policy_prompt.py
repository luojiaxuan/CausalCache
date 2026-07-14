import unittest

from causalcache.policy import build_policy_messages, parse_policy_action
from causalcache.schema import ActionType


class PolicyPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "trajectory": {
                "instruction": "Open the saved item.",
                "events": [
                    {
                        "step_id": 1,
                        "observation_before_path": "before.png",
                        "observation_after_path": "after.png",
                        "executed_action": {"action_type": "tap", "target": "coordinate_bin:x1_y2"},
                        "source_tool_call": {
                            "function": {"name": "tap", "arguments": {"coordinate": [100, 200]}}
                        },
                        "low_fidelity": {
                            "step_id": 1,
                            "action_type": "tap",
                            "target_text_or_coordinate_bin": "coordinate_bin:x1_y2",
                            "deterministic_ui_delta": "not_available",
                            "result_status": "unknown",
                        },
                    }
                ],
                "decisions": [
                    {
                        "decision_step_id": 2,
                        "history_event_step_ids": [1],
                        "current_observation_path": "current.png",
                    }
                ],
            }
        }

    def test_mixed_fidelity_changes_only_restored_image_blocks(self) -> None:
        summary_messages = build_policy_messages(
            self.manifest,
            decision_step_id=2,
            restored_event_step_ids=[],
            image_loader=lambda path: path,
        )
        restored_messages = build_policy_messages(
            self.manifest,
            decision_step_id=2,
            restored_event_step_ids=[1],
            image_loader=lambda path: path,
        )
        summary_images = [item for item in summary_messages[1]["content"] if item["type"] == "image"]
        restored_images = [item for item in restored_messages[1]["content"] if item["type"] == "image"]
        self.assertEqual([item["image"] for item in summary_images], ["current.png"])
        self.assertEqual(
            [item["image"] for item in restored_images],
            ["before.png", "after.png", "current.png"],
        )

    def test_policy_action_parser_uses_executable_canonicalization(self) -> None:
        action = parse_policy_action('answer: {"action_type":"tap","coordinate":[1000,0]}')
        self.assertEqual(action.action_type, ActionType.TAP)
        self.assertEqual(action.target, "coordinate_bin:x9_y0")


if __name__ == "__main__":
    unittest.main()
