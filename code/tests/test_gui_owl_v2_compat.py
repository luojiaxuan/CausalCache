from __future__ import annotations

import json
import unittest

from causalcache.policy.gui_owl_v2 import (
    GUIOwlV2Action,
    gui_owl_v2_action_to_androidworld,
    serialize_gui_owl_v2_teacher_target,
)
from causalcache.policy.gui_owl_v2_compat import (
    GUI_OWL_V2_COMPATIBILITY_PARSER_CONTRACT_ID,
    inspect_gui_owl_v2_output_compatibility,
    parse_gui_owl_v2_output_compat,
)


def _native(payload: object, suffix: str = "") -> str:
    return (
        "Action: Execute one action.\n<tool_call>\n"
        + json.dumps(payload, ensure_ascii=False)
        + suffix
    )


class GUIOwlV2CompatibilityParserTest(unittest.TestCase):
    def test_strict_parser_remains_the_first_path(self) -> None:
        action = GUIOwlV2Action(action="click", coordinate=(123, 456))
        text = serialize_gui_owl_v2_teacher_target(action)
        parsed = parse_gui_owl_v2_output_compat(text)
        self.assertEqual(parsed.parser_path, "strict")
        self.assertEqual(
            parsed.parser_contract_id,
            GUI_OWL_V2_COMPATIBILITY_PARSER_CONTRACT_ID,
        )
        self.assertEqual(parsed.parsed_output.canonical_action, action)
        self.assertEqual(parsed.wrapper_variant, "name_and_arguments")
        self.assertEqual(parsed.suffix_variant, "canonical_closer")

    def test_four_observed_wrappers_and_safe_suffixes_canonicalize(self) -> None:
        cases = (
            ({"action": "click", "coordinate": [100, 200]}, "", "raw_arguments", "eof"),
            (
                {"mobile_use": {"action": "system_button", "button": "Home"}},
                "\n<tool_call>",
                "mobile_use_key",
                "bare_tool_call_echo",
            ),
            (
                {
                    "name": "mobile_use",
                    "arguments": {"action": "type", "text": "a}b"},
                },
                "\n}",
                "name_and_arguments",
                "extra_closing_brace",
            ),
            (
                {
                    "function": "mobile_use",
                    "arguments": {"action": "swipe", "coordinate": [1, 2], "coordinate2": [3, 4]},
                },
                "\n}\n<tool_call>",
                "function_and_arguments",
                "extra_brace_then_tool_call_echo",
            ),
        )
        for payload, suffix, wrapper, suffix_variant in cases:
            with self.subTest(wrapper=wrapper, suffix=suffix_variant):
                text = _native(payload, suffix)
                parsed = parse_gui_owl_v2_output_compat(text)
                self.assertEqual(parsed.parser_path, "compatibility")
                self.assertEqual(parsed.wrapper_variant, wrapper)
                self.assertEqual(parsed.suffix_variant, suffix_variant)
                self.assertEqual(parsed.parsed_output.raw_native_output, text)
                gui_owl_v2_action_to_androidworld(
                    parsed.parsed_output.canonical_action,
                    screen_width=1080,
                    screen_height=2400,
                )

    def test_compatibility_output_serializes_back_into_strict_form(self) -> None:
        parsed = parse_gui_owl_v2_output_compat(
            _native({"action": "open_app", "text": "Clock"})
        )
        strict = parse_gui_owl_v2_output_compat(
            serialize_gui_owl_v2_teacher_target(parsed.parsed_output.canonical_action)
        )
        self.assertEqual(strict.parser_path, "strict")
        self.assertEqual(strict.parsed_output.canonical_action.action, "open")
        self.assertEqual(
            strict.canonical_tool_call_sha256,
            parsed.canonical_tool_call_sha256,
        )

    def test_unsafe_real_output_shapes_remain_rejected(self) -> None:
        valid = _native({"action": "click", "coordinate": [100, 200]})
        cases = {
            "EXTRA_OBSERVATION": valid + "\n<observation>",
            "SECOND_JSON": valid + "\n{\"coordinate\":[3,4]}",
            "MULTIPLE_ACTIONS": valid + "\nAction: Again.\n<tool_call>\n{\"action\":\"wait\"}",
            "TRUNCATED_OR_INVALID_JSON": "Action: Click.\n<tool_call>\n{\"action\":\"click\",\"coordinate\":[1,2]",
            "PREFIX_MISMATCH": "<think>no</think>\n" + valid,
            "EMPTY_ACTION_DESCRIPTION": "Action:   \n<tool_call>\n{\"action\":\"wait\"}",
        }
        for code, text in cases.items():
            with self.subTest(code=code):
                inspection = inspect_gui_owl_v2_output_compatibility(text)
                self.assertFalse(inspection.compatibility_parse_success)
                self.assertEqual(inspection.rejection_code, code)
                with self.assertRaisesRegex(ValueError, code):
                    parse_gui_owl_v2_output_compat(text)

    def test_duplicate_nonfinite_unknown_and_semantic_drift_fail_closed(self) -> None:
        cases = (
            "Action: Click.\n<tool_call>\n{\"action\":\"click\",\"action\":\"wait\"}",
            "Action: Click.\n<tool_call>\n{\"action\":\"click\",\"coordinate\":[NaN,2]}",
            _native({"tool_calls": [{"action": "wait"}]}),
            _native({"function": {"name": "mobile_use"}, "arguments": {"action": "wait"}}),
            _native({"action": "click", "coordinate": [True, 2]}),
            _native({"action": "terminate", "status": "failure"}),
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(
                    inspect_gui_owl_v2_output_compatibility(text).compatibility_parse_success
                )
                with self.assertRaises(ValueError):
                    parse_gui_owl_v2_output_compat(text)

    def test_markdown_crlf_and_extra_prose_are_not_new_prefix_or_suffix_variants(self) -> None:
        valid = _native({"action": "wait"})
        cases = (
            "```json\n" + valid + "\n```",
            valid.replace("\n", "\r\n"),
            valid + "\nDone.",
        )
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_gui_owl_v2_output_compat(text)


if __name__ == "__main__":
    unittest.main()
