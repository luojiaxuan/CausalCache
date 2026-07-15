import json
import unittest
from pathlib import Path

from causalcache.policy.gui_owl_v2 import (
    CANONICAL_GUI_OWL_V2_ACTIONS,
    GUI_OWL_V2_ACTION_ALIASES,
    GUI_OWL_V2_SYSTEM_BUTTONS,
    GUI_OWL_V2_SYSTEM_PROMPT,
    GUI_OWL_V2_TEACHER_CARRIER,
    gui_owl_v2_action_to_androidworld,
    normalized_coordinate_to_pixel,
    parse_gui_owl_v2_output,
    serialize_gui_owl_v2_teacher_target,
    serialize_gui_owl_v2_tool_call,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPOSITORY_ROOT / "data/fixtures/gui_owl_v2_action_roundtrip.json"


class GUIOwlV2ActionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_metadata_and_coverage_are_exhaustive(self) -> None:
        self.assertEqual(self.fixture["contract_id"], "gui_owl_androidworld_v2")
        parsed = [
            parse_gui_owl_v2_output(case["native_output"])
            for case in self.fixture["valid_cases"]
        ]
        canonical_actions = {item.canonical_action.action for item in parsed}
        source_actions = {item.source_action_name for item in parsed}
        buttons = {
            item.canonical_action.button
            for item in parsed
            if item.canonical_action.button is not None
        }
        self.assertEqual(canonical_actions, set(CANONICAL_GUI_OWL_V2_ACTIONS))
        self.assertTrue(set(GUI_OWL_V2_ACTION_ALIASES).issubset(source_actions))
        self.assertEqual(buttons, set(GUI_OWL_V2_SYSTEM_BUTTONS))

    def test_valid_fixture_round_trips_through_canonical_target_and_bridge(self) -> None:
        screen = self.fixture["screen"]
        for case in self.fixture["valid_cases"]:
            with self.subTest(case=case["id"]):
                parsed = parse_gui_owl_v2_output(case["native_output"])
                self.assertEqual(parsed.canonical_action.arguments(), case["canonical_arguments"])
                self.assertEqual(
                    serialize_gui_owl_v2_tool_call(parsed.canonical_action),
                    case["canonical_tool_call"],
                )
                teacher_target = serialize_gui_owl_v2_teacher_target(parsed.canonical_action)
                self.assertEqual(
                    teacher_target,
                    GUI_OWL_V2_TEACHER_CARRIER + case["canonical_tool_call"],
                )
                reparsed = parse_gui_owl_v2_output(teacher_target)
                self.assertEqual(reparsed.canonical_action, parsed.canonical_action)
                self.assertEqual(
                    gui_owl_v2_action_to_androidworld(
                        parsed.canonical_action,
                        screen_width=screen["width"],
                        screen_height=screen["height"],
                    ),
                    case["androidworld_payload"],
                )
                self.assertEqual(parsed.raw_native_output, case["native_output"])

    def test_invalid_fixture_is_rejected_fail_closed(self) -> None:
        for case in self.fixture["invalid_cases"]:
            with self.subTest(case=case["id"]):
                with self.assertRaisesRegex(ValueError, case["error_contains"]):
                    parse_gui_owl_v2_output(case["native_output"])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            parse_gui_owl_v2_output(
                'Action:   \n<tool_call>{"name":"mobile_use","arguments":{"action":"wait"}}</tool_call>'
            )

    def test_every_coordinate_value_is_monotonic_and_inside_extent(self) -> None:
        for extent in (1, 2, 432, 768, 1080, 2400):
            with self.subTest(extent=extent):
                pixels = [normalized_coordinate_to_pixel(value, extent) for value in range(1000)]
                self.assertEqual(pixels[0], 0)
                self.assertEqual(pixels[-1], extent - 1)
                self.assertTrue(all(0 <= pixel < extent for pixel in pixels))
                self.assertEqual(pixels, sorted(pixels))

    def test_prompt_exposes_only_the_frozen_effective_inventory(self) -> None:
        expected_enum = '"enum":["click","long_press","swipe","type","system_button","open","wait","answer","terminate"]'
        self.assertIn(expected_enum, GUI_OWL_V2_SYSTEM_PROMPT)
        self.assertNotIn('"Menu"', GUI_OWL_V2_SYSTEM_PROMPT)
        self.assertNotIn('"time"', GUI_OWL_V2_SYSTEM_PROMPT)
        self.assertNotIn('"open_app"', GUI_OWL_V2_SYSTEM_PROMPT)
        self.assertNotIn('<think>', GUI_OWL_V2_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
