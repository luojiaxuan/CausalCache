from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from causalcache.osworld_gui_owl import (
    build_gui_owl_osworld_messages,
    parse_gui_owl_osworld_action,
)


def _png_base64(color: str) -> str:
    if color not in {"white", "black"}:
        raise ValueError("test fixture supports only white and black")
    return (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAusB9Y9Zl9sAAAAASUVORK5CYII="
    )


class OSWorldGUIOwlPromptTests(unittest.TestCase):
    def test_builds_recent_mixed_fidelity_messages(self) -> None:
        request = {
            "task": {"instruction": "Open Settings"},
            "screen_size": [1920, 1080],
            "current_screenshot_png_base64": _png_base64("white"),
            "selected_event_step_ids": [2],
            "history": [
                {
                    "step_id": 1,
                    "action": {"type": "wait"},
                    "result_status": "executed",
                    "screen_changed": False,
                    "restored_post_screenshot_png_base64": None,
                },
                {
                    "step_id": 2,
                    "action": {"type": "click", "x": 10, "y": 20},
                    "result_status": "executed",
                    "screen_changed": True,
                    "restored_post_screenshot_png_base64": _png_base64("black"),
                },
            ],
        }
        with patch("causalcache.osworld_gui_owl._decode_png", return_value=object()):
            messages = build_gui_owl_osworld_messages(request)
        images = [
            part
            for message in messages
            for part in message["content"]
            if part["type"] == "image"
        ]
        self.assertEqual(len(images), 2)
        self.assertIn("Previous event summaries", messages[1]["content"][0]["text"])

    def test_rejects_selected_event_without_image(self) -> None:
        request = {
            "task": {"instruction": "Open Settings"},
            "screen_size": [1920, 1080],
            "current_screenshot_png_base64": _png_base64("white"),
            "selected_event_step_ids": [1],
            "history": [
                {
                    "step_id": 1,
                    "action": {"type": "wait"},
                    "result_status": "executed",
                    "screen_changed": False,
                    "restored_post_screenshot_png_base64": None,
                }
            ],
        }
        with patch("causalcache.osworld_gui_owl._decode_png", return_value=object()):
            with self.assertRaisesRegex(ValueError, "lacks its restored screenshot"):
                build_gui_owl_osworld_messages(request)


class OSWorldGUIOwlActionTests(unittest.TestCase):
    def _output(self, arguments: dict[str, object]) -> str:
        return (
            "Action: execute\n<tool_call>"
            + json.dumps({"name": "computer_use", "arguments": arguments})
            + "</tool_call>"
        )

    def test_maps_normalized_click_to_desktop_pixels(self) -> None:
        action = parse_gui_owl_osworld_action(
            self._output({"action": "left_click", "coordinate": [500, 500]}),
            screen_size=(1920, 1080),
        )
        self.assertEqual(action.type, "click")
        self.assertEqual((action.x, action.y), (960, 540))
        alias = parse_gui_owl_osworld_action(
            self._output({"action": "click", "coordinate": [0, 999]}),
            screen_size=(1920, 1080),
        )
        self.assertEqual((alias.type, alias.x, alias.y), ("click", 0, 1079))

    def test_maps_keyboard_scroll_and_terminal_actions(self) -> None:
        hotkey = parse_gui_owl_osworld_action(
            self._output({"action": "key", "keys": ["CTRL", "L"]}),
            screen_size=(1920, 1080),
        )
        scroll = parse_gui_owl_osworld_action(
            self._output({"action": "scroll", "pixels": -4}),
            screen_size=(1920, 1080),
        )
        done = parse_gui_owl_osworld_action(
            self._output({"action": "terminate", "status": "success"}),
            screen_size=(1920, 1080),
        )
        self.assertEqual(hotkey.keys, ("ctrl", "l"))
        self.assertEqual(scroll.dy, -4)
        self.assertEqual(done.type, "done")


if __name__ == "__main__":
    unittest.main()
