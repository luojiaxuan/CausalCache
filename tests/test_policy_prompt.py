import unittest

from causalcache.policy import build_policy_messages, parse_policy_action
from causalcache.policy.open_cua import build_open_cua_messages, parse_open_cua_action
from causalcache.policy.showui import build_showui_messages, parse_showui_action
from causalcache.policy.ui_tars import build_ui_tars_messages, parse_ui_tars_action
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

    def test_ui_tars_native_prompt_and_coordinate_parser(self) -> None:
        messages = build_ui_tars_messages(
            self.manifest,
            decision_step_id=2,
            restored_event_step_ids=[1],
            image_loader=lambda path: path,
        )
        self.assertIn("scroll(point=", messages[1]["content"][0]["text"])
        action = parse_ui_tars_action(
            "Thought: tap the center\nAction: click(point='<point>170 300</point>')",
            {"image_grid_thw": [[1, 42, 24]]},
        )
        self.assertEqual(action.action_type, ActionType.TAP)
        self.assertEqual(action.target, "coordinate_bin:x5_y5")
        swipe = parse_ui_tars_action(
            "Thought: move down\nAction: scroll(point='<point>170 300</point>', direction='down')",
            {"image_grid_thw": [[1, 42, 24]]},
        )
        self.assertEqual(swipe.target, "scroll:down")

    def test_open_cua_prompt_and_pyautogui_parser(self) -> None:
        messages = build_open_cua_messages(
            self.manifest,
            decision_step_id=2,
            restored_event_step_ids=[1],
            image_loader=lambda path: path,
        )
        self.assertIn("PyAutoGUI", messages[0]["content"][0]["text"])
        self.assertEqual(messages[0]["content"][0]["type"], "text")
        inputs = {"image_grid_thw": [[1, 42, 24]]}
        action = parse_open_cua_action(
            "Thought: use the center\nCode:\n```python\npyautogui.click(x=170, y=300)\n```",
            inputs,
        )
        self.assertEqual(action.action_type, ActionType.TAP)
        self.assertEqual(action.target, "coordinate_bin:x5_y5")
        text_action = parse_open_cua_action("Code:\npyautogui.write('hello')", inputs)
        self.assertEqual(text_action.action_type, ActionType.TYPE_TEXT)
        self.assertEqual(text_action.text_argument, "hello")
        scroll = parse_open_cua_action("Code:\npyautogui.scroll(-4)", inputs)
        self.assertEqual(scroll.target, "scroll:down")
        with self.assertRaisesRegex(ValueError, "2 executable code lines"):
            parse_open_cua_action(
                "Code:\npyautogui.write('hello')\npyautogui.press('enter')",
                inputs,
            )

    def test_showui_phone_prompt_and_dictionary_parser(self) -> None:
        messages = build_showui_messages(
            self.manifest,
            decision_step_id=2,
            restored_event_step_ids=[1],
            image_loader=lambda path: path,
        )
        self.assertIn("`INPUT`", messages[0]["content"][0]["text"])
        self.assertIn("`SWIPE`", messages[0]["content"][0]["text"])
        self.assertEqual(
            parse_showui_action(
                "{'action': 'TAP', 'value': None, 'position': [0.55, 0.42]}",
                {},
            ).target,
            "coordinate_bin:x5_y4",
        )
        text_action = parse_showui_action(
            "{'action': 'INPUT', 'value': 'Hello', 'position': [0.5, 0.5]}",
            {},
        )
        self.assertEqual(text_action.action_type, ActionType.TYPE_TEXT)
        self.assertEqual(text_action.text_argument, "Hello")
        swipe = parse_showui_action(
            "{'action': 'SWIPE', 'value': None, 'position': [[0.5, 0.8], [0.5, 0.2]]}",
            {},
        )
        self.assertEqual(swipe.target, "scroll:down")
        self.assertEqual(
            parse_showui_action(
                "```python\n{'action': 'ENTER', 'value': None, 'position': None}\n```",
                {},
            ).action_type,
            ActionType.ENTER,
        )

    def test_showui_parser_rejects_noncanonical_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one dictionary"):
            parse_showui_action(
                "{'action': 'TAP', 'value': None, 'position': [0.5, 0.5]}\n"
                "{'action': 'ENTER', 'value': None, 'position': None}",
                {},
            )
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            parse_showui_action(
                "{'action': 'TAP', 'value': None, 'position': [500, 500]}",
                {},
            )


if __name__ == "__main__":
    unittest.main()
