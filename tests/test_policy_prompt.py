import unittest

from causalcache.policy import build_policy_messages, parse_policy_action
from causalcache.policy.open_cua import build_open_cua_messages, parse_open_cua_action
from causalcache.policy.gui_owl import (
    build_gui_owl_native_messages,
    gui_owl_action_to_androidworld,
    parse_gui_owl_action,
    render_gui_owl_action,
)
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
            "{'action': 'INPUT', 'value': 'Hello', 'position': None}",
            {},
        )
        self.assertEqual(text_action.action_type, ActionType.TYPE_TEXT)
        self.assertEqual(text_action.text_argument, "Hello")
        positioned_text_action = parse_showui_action(
            "{'action': 'INPUT', 'value': 'World', 'position': [0.5, 0.5]}",
            {},
        )
        self.assertEqual(positioned_text_action.text_argument, "World")
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

    def test_gui_owl_native_history_and_parser(self) -> None:
        action_outputs = [
            render_gui_owl_action(
                {"function": {"name": "tap", "arguments": {"coordinate": [100, 200]}}},
                description=f"Tap item {index}",
            )
            for index in range(5)
        ]
        messages = build_gui_owl_native_messages(
            instruction="Open the saved item.",
            screenshots=[f"screen-{index}.png" for index in range(6)],
            action_outputs=action_outputs,
        )
        images = [
            item["image"]
            for message in messages
            for item in message["content"]
            if item["type"] == "image"
        ]
        self.assertEqual(images, [f"screen-{index}.png" for index in range(1, 6)])
        self.assertIn("Step1: Tap item 0", messages[1]["content"][0]["text"])

        tap = parse_gui_owl_action(action_outputs[0], {})
        self.assertEqual(tap.action_type, ActionType.TAP)
        self.assertEqual(tap.target, "coordinate_bin:x0_y1")
        answer = parse_gui_owl_action(
            'Action: Return the result\n<tool_call>\n'
            '{"name":"mobile_use","arguments":{"action":"answer","text":"42"}}\n'
            '</tool_call>',
            {},
        )
        self.assertEqual(answer.action_type, ActionType.ANSWER)
        self.assertEqual(answer.text_argument, "42")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_gui_owl_action(action_outputs[0] + "\n" + action_outputs[1], {})

    def test_gui_owl_androidworld_converter_matches_pinned_mobileagent(self) -> None:
        click = gui_owl_action_to_androidworld(
            'Action: Tap the item\n<tool_call>{"name":"mobile_use","arguments":'
            '{"action":"click","coordinate":[500,250]}}</tool_call>',
            screen_width=1080,
            screen_height=2400,
        )
        self.assertEqual(click, {"action_type": "click", "x": 540, "y": 600})
        swipe = gui_owl_action_to_androidworld(
            'Action: Scroll down\n<tool_call>{"name":"mobile_use","arguments":'
            '{"action":"swipe","coordinate":[500,800],"coordinate2":[500,200]}}'
            '</tool_call>',
            screen_width=1080,
            screen_height=2400,
        )
        self.assertEqual(swipe, {"action_type": "swipe", "direction": [540, 1921, 540, 480]})
        typed = gui_owl_action_to_androidworld(
            'Action: Enter text\n<tool_call>{"name":"mobile_use","arguments":'
            '{"action":"type","text":"hello"}}</tool_call>',
            screen_width=1080,
            screen_height=2400,
        )
        self.assertEqual(typed, {"action_type": "input_text", "text": "hello"})
        terminated = gui_owl_action_to_androidworld(
            'Action: Finish\n<tool_call>{"name":"mobile_use","arguments":'
            '{"action":"terminate","status":"success"}}</tool_call>',
            screen_width=1080,
            screen_height=2400,
        )
        self.assertEqual(
            terminated,
            {"action_type": "status", "goal_status": "task_complete"},
        )

    def test_gui_owl_androidworld_converter_rejects_unimplemented_official_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "system button"):
            gui_owl_action_to_androidworld(
                'Action: Show apps\n<tool_call>{"name":"mobile_use","arguments":'
                '{"action":"system_button","button":"Menu"}}</tool_call>',
                screen_width=1080,
                screen_height=2400,
            )
        with self.assertRaisesRegex(ValueError, "key actions"):
            gui_owl_action_to_androidworld(
                'Action: Clear\n<tool_call>{"name":"mobile_use","arguments":'
                '{"action":"key","text":"clear"}}</tool_call>',
                screen_width=1080,
                screen_height=2400,
            )


if __name__ == "__main__":
    unittest.main()
