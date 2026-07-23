from __future__ import annotations

import json
import unittest
from pathlib import Path

from causalcache.osworld_lora_leakage import (
    classify_profile,
    inspect_output,
    summarize_records,
)
from scripts.run_osworld_lora_leakage_profiles import parse_assignment


def _tool(name: str, action: str, extra: dict[str, object] | None = None) -> str:
    arguments = {"action": action, **(extra or {})}
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


class OSWorldLoRALeakageTests(unittest.TestCase):
    def test_parses_explicit_profile_assignment(self) -> None:
        self.assertEqual(
            parse_assignment("2:adapted:/data/checkpoint.pt"),
            ("2", "adapted", Path("/data/checkpoint.pt")),
        )
        self.assertEqual(parse_assignment("0:frozen:-"), ("0", "frozen", None))

    def test_detects_valid_click_alias_without_hard_leakage(self) -> None:
        value = inspect_output(
            _tool("computer_use", "click", {"coordinate": [500, 500]}),
            screen_size=(1920, 1080),
        )
        self.assertTrue(value["parser_valid"])
        self.assertTrue(value["click_alias"])
        self.assertFalse(value["hard_leakage"])

    def test_detects_mobile_tool_and_action_leakage(self) -> None:
        mobile_tool = inspect_output(
            _tool("mobile_use", "click", {"coordinate": [500, 500]}),
            screen_size=(1920, 1080),
        )
        mobile_action = inspect_output(
            _tool("computer_use", "swipe"), screen_size=(1920, 1080)
        )
        self.assertTrue(mobile_tool["hard_leakage"])
        self.assertTrue(mobile_action["hard_leakage"])
        self.assertFalse(mobile_tool["parser_valid"])

    def test_summarizes_and_applies_preregistered_drop_gate(self) -> None:
        records = [
            {
                "inspection": {
                    "parser_valid": True,
                    "hard_leakage": False,
                    "raw_action": "left_click",
                    "normalized_action": "click",
                }
            },
            {
                "inspection": {
                    "parser_valid": False,
                    "hard_leakage": True,
                    "raw_action": "swipe",
                    "normalized_action": None,
                }
            },
        ]
        adapted = summarize_records(records)
        frozen = {
            **adapted,
            "parser_valid_rate": 1.0,
            "hard_leakage_count": 0,
            "normalized_action_counts": {"click": 2},
        }
        verdict = classify_profile(
            frozen=frozen,
            adapted=adapted,
            thresholds={"pass_drop_points": 2, "hard_fail_drop_points": 10},
        )
        self.assertEqual(adapted["parser_valid_count"], 1)
        self.assertEqual(verdict["verdict"], "FAIL_HARD_LEAKAGE")


if __name__ == "__main__":
    unittest.main()
