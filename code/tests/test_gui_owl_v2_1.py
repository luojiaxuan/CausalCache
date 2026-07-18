from __future__ import annotations

import json
import unittest
from pathlib import Path

from causalcache.policy.gui_owl_v2 import (
    CANONICAL_GUI_OWL_V2_ACTIONS,
    GUIOwlV2Action,
    gui_owl_v2_action_to_androidworld,
)
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_MOBILE_USE_TOOL,
    GUI_OWL_V2_1_PROTOCOL_ID,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    build_gui_owl_v2_1_mixed_fidelity_messages,
    canonical_json_sha256,
    parse_gui_owl_v2_1_output,
    serialize_gui_owl_v2_1_teacher_target,
    validate_gui_owl_v2_1_native_messages,
    validate_gui_owl_v2_1_tool_schema,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROMPT_FIXTURE = REPOSITORY_ROOT / "data/fixtures/restoration_v2_prompt_low_fidelity.json"


def _actions() -> tuple[GUIOwlV2Action, ...]:
    return (
        GUIOwlV2Action(action="click", coordinate=(0, 999)),
        GUIOwlV2Action(action="long_press", coordinate=(500, 500)),
        GUIOwlV2Action(action="swipe", coordinate=(1, 2), coordinate2=(3, 4)),
        GUIOwlV2Action(action="type", text="Café"),
        GUIOwlV2Action(action="system_button", button="Back"),
        GUIOwlV2Action(action="open", text="Clock"),
        GUIOwlV2Action(action="wait"),
        GUIOwlV2Action(action="answer", text="done"),
        GUIOwlV2Action(action="terminate", status="success"),
    )


class GUIOwlV21InterfaceTest(unittest.TestCase):
    def test_protocol_prompt_and_schema_are_official_tool_only(self) -> None:
        self.assertEqual(
            GUI_OWL_V2_1_PROTOCOL_ID,
            "causalcache_restoration_v2_1_official_tool_interface",
        )
        self.assertNotIn("Action:", GUI_OWL_V2_1_SYSTEM_PROMPT)
        self.assertNotIn("<tools>", GUI_OWL_V2_1_SYSTEM_PROMPT)
        self.assertIn("exactly once", GUI_OWL_V2_1_SYSTEM_PROMPT)
        self.assertIn("Unicode NFKC", GUI_OWL_V2_1_SYSTEM_PROMPT)
        validate_gui_owl_v2_1_tool_schema()
        variants = GUI_OWL_V2_1_MOBILE_USE_TOOL["function"]["parameters"]["oneOf"]
        self.assertEqual(
            tuple(variant["properties"]["action"]["const"] for variant in variants),
            CANONICAL_GUI_OWL_V2_ACTIONS,
        )
        self.assertTrue(all(variant["additionalProperties"] is False for variant in variants))
        text_variants = {
            variant["properties"]["action"]["const"]: variant["properties"]["text"]
            for variant in variants
            if "text" in variant["properties"]
        }
        self.assertEqual(set(text_variants), {"type", "open", "answer"})
        self.assertTrue(
            all("Unicode NFKC" in schema["description"] for schema in text_variants.values())
        )
        self.assertEqual(len(canonical_json_sha256(GUI_OWL_V2_1_MOBILE_USE_TOOL)), 64)

    def test_every_action_round_trips_through_exact_whole_output_and_bridge(self) -> None:
        for action in _actions():
            with self.subTest(action=action.action):
                target = serialize_gui_owl_v2_1_teacher_target(action)
                self.assertTrue(target.startswith("<tool_call>\n{"))
                self.assertTrue(target.endswith("\n</tool_call>"))
                self.assertIn('"name": "mobile_use"', target)
                parsed = parse_gui_owl_v2_1_output(target)
                self.assertEqual(parsed.canonical_action, action)
                self.assertEqual(parsed.raw_native_output, target)
                gui_owl_v2_action_to_androidworld(
                    action,
                    screen_width=1080,
                    screen_height=2400,
                )

    def test_teacher_target_matches_official_tojson_unicode_serialization(self) -> None:
        target = serialize_gui_owl_v2_1_teacher_target(
            GUIOwlV2Action(action="type", text="Café")
        )
        self.assertEqual(
            target,
            '<tool_call>\n{"name": "mobile_use", "arguments": '
            '{"action": "type", "text": "Café"}}\n</tool_call>',
        )
        self.assertEqual(parse_gui_owl_v2_1_output(target).canonical_action.text, "Café")

    def test_parser_rejects_recovery_aliases_and_any_extra_output(self) -> None:
        valid = serialize_gui_owl_v2_1_teacher_target(GUIOwlV2Action(action="wait"))
        cases = (
            " " + valid,
            valid + "\n",
            "Action: Wait\n" + valid,
            "analysis\n" + valid,
            valid + "\n<observation>x</observation>",
            valid + "\n" + valid,
            valid.replace("</tool_call>", ""),
            valid.replace("mobile_use", "computer_use"),
            valid.replace('"wait"', '"open_app"').replace(
                "}", ', "text": "Clock"}', 1
            ),
            '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"},"extra":1}\n</tool_call>',
            '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait","action":"click"}}\n</tool_call>',
            '<tool_call>\n{"name":"mobile_use","arguments":{"action":"click","coordinate":[NaN,2]}}\n</tool_call>',
            '<tool_call>\r\n{"name":"mobile_use","arguments":{"action":"wait"}}\r\n</tool_call>',
            '<tool_call>\n{\n"name":"mobile_use","arguments":{"action":"wait"}\n}\n</tool_call>',
            '<tool_call>\n{"name":"mobile_use","arguments":{"action":"answer","text":"\ud800"}}\n</tool_call>',
            '<tool_call>\n{"name":"mobile_use","arguments":{"action":"type","text":"Ａ"}}\n</tool_call>',
        )
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_gui_owl_v2_1_output(text)

    def test_parser_accepts_single_line_json_key_order_and_whitespace(self) -> None:
        text = (
            "<tool_call>\n"
            '{ "arguments": {"coordinate": [123, 456], "action": "click"}, '
            '"name": "mobile_use" }\n'
            "</tool_call>"
        )
        parsed = parse_gui_owl_v2_1_output(text)
        self.assertEqual(parsed.canonical_action.coordinate, (123, 456))
        self.assertEqual(parsed.raw_native_output, text)

    def test_v21_builder_changes_only_interface_text_not_fidelity_blocks(self) -> None:
        manifest = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))
        messages = build_gui_owl_v2_1_mixed_fidelity_messages(
            manifest,
            trajectory_id="fixture-step-6",
            decision_step_id=6,
            restored_event_step_ids=(1, 2, 3, 4),
            image_bytes_loader=lambda path: path.encode("utf-8"),
            image_decoder=lambda raw: raw.decode("utf-8"),
        )
        self.assertEqual(validate_gui_owl_v2_1_native_messages(messages), 5)
        self.assertEqual(messages[0]["content"][0]["text"], GUI_OWL_V2_1_SYSTEM_PROMPT)
        self.assertEqual(
            messages[1]["content"][-1],
            {"type": "text", "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION},
        )
        images = [
            item["image"]
            for item in messages[1]["content"]
            if item["type"] == "image"
        ]
        self.assertEqual(
            images,
            [
                "fixture://post-001.png",
                "fixture://post-002.png",
                "fixture://post-003.png",
                "fixture://post-004.png",
                "fixture://current.png",
            ],
        )

    def test_v21_default_allowlist_output_is_identical_to_explicit_default(self) -> None:
        manifest = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))
        kwargs = {
            "trajectory_id": "fixture-step-6",
            "decision_step_id": 6,
            "restored_event_step_ids": (1, 3),
            "image_bytes_loader": lambda path: path.encode("utf-8"),
            "image_decoder": lambda raw: raw.decode("utf-8"),
        }
        implicit = build_gui_owl_v2_1_mixed_fidelity_messages(manifest, **kwargs)
        explicit = build_gui_owl_v2_1_mixed_fidelity_messages(
            manifest,
            **kwargs,
            allowed_decision_steps=(4, 5, 6),
        )
        self.assertEqual(implicit, explicit)

    def test_native_message_validation_is_fail_closed(self) -> None:
        manifest = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))
        messages = build_gui_owl_v2_1_mixed_fidelity_messages(
            manifest,
            trajectory_id="fixture-step-6",
            decision_step_id=4,
            restored_event_step_ids=(1, 2),
            image_bytes_loader=lambda path: path.encode("utf-8"),
            image_decoder=lambda raw: raw.decode("utf-8"),
        )
        messages[0]["content"][0]["text"] += " drift"
        with self.assertRaisesRegex(ValueError, "system message drifted"):
            validate_gui_owl_v2_1_native_messages(messages)


if __name__ == "__main__":
    unittest.main()
