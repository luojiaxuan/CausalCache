"""Versioned, format-only compatibility parser for recorded GUI-Owl v2 outputs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from causalcache.policy.gui_owl_v2 import (
    ParsedGUIOwlV2Output,
    _canonical_action,
    _reject_constant,
    _unique_object,
    parse_gui_owl_v2_output,
    serialize_gui_owl_v2_tool_call,
)


GUI_OWL_V2_COMPATIBILITY_PARSER_CONTRACT_ID = (
    "gui_owl_v2_recorded_envelope_compatibility_v1"
)

_COMPATIBLE_PREFIX = re.compile(
    r"Action:[ \t]*(?P<description>[^\r\n]+)\n"
    r"<tool_call>\n"
)
_JSON_DECODER = json.JSONDecoder(
    object_pairs_hook=_unique_object,
    parse_constant=_reject_constant,
)


@dataclass(frozen=True)
class GUIOwlV2CompatibilityInspection:
    strict_parse_success: bool
    compatibility_parse_success: bool
    prefix_variant: str | None
    wrapper_variant: str | None
    suffix_variant: str | None
    first_balanced_json: bool
    canonical_first_payload: bool
    rejection_code: str | None
    extracted_json: str | None
    parsed_output: ParsedGUIOwlV2Output | None


@dataclass(frozen=True)
class ParsedGUIOwlV2CompatibilityOutput:
    parsed_output: ParsedGUIOwlV2Output
    parser_path: str
    parser_contract_id: str
    wrapper_variant: str
    suffix_variant: str
    raw_output_sha256: str
    extracted_json_sha256: str
    canonical_tool_call_sha256: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _arguments_from_wrapper(value: Any) -> tuple[Mapping[str, Any], str]:
    if not isinstance(value, Mapping):
        raise ValueError("tool call JSON must be an object")
    if "action" in value:
        return value, "raw_arguments"
    if set(value) == {"mobile_use"}:
        arguments = value["mobile_use"]
        if not isinstance(arguments, Mapping):
            raise ValueError("mobile_use wrapper must contain an object")
        return arguments, "mobile_use_key"
    if set(value) == {"name", "arguments"} and value.get("name") == "mobile_use":
        arguments = value["arguments"]
        if not isinstance(arguments, Mapping):
            raise ValueError("name wrapper arguments must be an object")
        return arguments, "name_and_arguments"
    if (
        set(value) == {"function", "arguments"}
        and value.get("function") == "mobile_use"
    ):
        arguments = value["arguments"]
        if not isinstance(arguments, Mapping):
            raise ValueError("function wrapper arguments must be an object")
        return arguments, "function_and_arguments"
    raise ValueError("unsupported GUI-Owl v2 compatibility wrapper")


def _suffix_variant(remainder: str) -> str | None:
    tail = remainder.strip(" \t\r\n")
    if tail == "":
        return "eof"
    if tail == "</tool_call>":
        return "canonical_closer"
    if tail == "<tool_call>":
        return "bare_tool_call_echo"
    if tail == "}":
        return "extra_closing_brace"
    if re.fullmatch(r"}[ \t\r\n]*<tool_call>", tail) is not None:
        return "extra_brace_then_tool_call_echo"
    return None


def _unsafe_suffix_rejection_code(remainder: str) -> str:
    tail = remainder.strip(" \t\r\n")
    if "\nAction:" in f"\n{tail}" or tail.startswith("Action:"):
        return "MULTIPLE_ACTIONS"
    if tail.startswith("<observation>"):
        return "EXTRA_OBSERVATION"
    if tail.startswith("{"):
        return "SECOND_JSON"
    return "UNSAFE_SUFFIX"


def inspect_gui_owl_v2_output_compatibility(
    text: str,
) -> GUIOwlV2CompatibilityInspection:
    if not isinstance(text, str):
        raise TypeError("GUI-Owl v2 output must be a string")
    try:
        strict = parse_gui_owl_v2_output(text)
    except ValueError:
        strict = None
    if strict is not None:
        return GUIOwlV2CompatibilityInspection(
            strict_parse_success=True,
            compatibility_parse_success=True,
            prefix_variant="strict_canonical",
            wrapper_variant="name_and_arguments",
            suffix_variant="canonical_closer",
            first_balanced_json=True,
            canonical_first_payload=True,
            rejection_code=None,
            extracted_json=strict.raw_tool_call_json,
            parsed_output=strict,
        )

    stripped = text.strip(" \t\r\n")
    prefix = _COMPATIBLE_PREFIX.match(stripped)
    if prefix is None:
        return GUIOwlV2CompatibilityInspection(
            strict_parse_success=False,
            compatibility_parse_success=False,
            prefix_variant=None,
            wrapper_variant=None,
            suffix_variant=None,
            first_balanced_json=False,
            canonical_first_payload=False,
            rejection_code="PREFIX_MISMATCH",
            extracted_json=None,
            parsed_output=None,
        )
    action_description = prefix.group("description").strip()
    if not action_description:
        return GUIOwlV2CompatibilityInspection(
            strict_parse_success=False,
            compatibility_parse_success=False,
            prefix_variant="action_line_and_tool_call",
            wrapper_variant=None,
            suffix_variant=None,
            first_balanced_json=False,
            canonical_first_payload=False,
            rejection_code="EMPTY_ACTION_DESCRIPTION",
            extracted_json=None,
            parsed_output=None,
        )

    action_line_count = sum(
        line.startswith("Action:") for line in stripped.splitlines()
    )
    if action_line_count != 1:
        return GUIOwlV2CompatibilityInspection(
            strict_parse_success=False,
            compatibility_parse_success=False,
            prefix_variant="action_line_and_tool_call",
            wrapper_variant=None,
            suffix_variant=None,
            first_balanced_json=False,
            canonical_first_payload=False,
            rejection_code="MULTIPLE_ACTIONS",
            extracted_json=None,
            parsed_output=None,
        )

    body = stripped[prefix.end() :]
    try:
        value, end = _JSON_DECODER.raw_decode(body)
    except (TypeError, ValueError):
        return GUIOwlV2CompatibilityInspection(
            strict_parse_success=False,
            compatibility_parse_success=False,
            prefix_variant="action_line_and_tool_call",
            wrapper_variant=None,
            suffix_variant=None,
            first_balanced_json=False,
            canonical_first_payload=False,
            rejection_code="TRUNCATED_OR_INVALID_JSON",
            extracted_json=None,
            parsed_output=None,
        )

    raw_json = body[:end]
    try:
        arguments, wrapper_variant = _arguments_from_wrapper(value)
    except ValueError:
        return GUIOwlV2CompatibilityInspection(
            strict_parse_success=False,
            compatibility_parse_success=False,
            prefix_variant="action_line_and_tool_call",
            wrapper_variant=None,
            suffix_variant=None,
            first_balanced_json=True,
            canonical_first_payload=False,
            rejection_code="UNSUPPORTED_WRAPPER",
            extracted_json=raw_json,
            parsed_output=None,
        )
    try:
        canonical_action, source_action = _canonical_action(arguments)
    except (TypeError, ValueError):
        return GUIOwlV2CompatibilityInspection(
            strict_parse_success=False,
            compatibility_parse_success=False,
            prefix_variant="action_line_and_tool_call",
            wrapper_variant=wrapper_variant,
            suffix_variant=None,
            first_balanced_json=True,
            canonical_first_payload=False,
            rejection_code="INVALID_ACTION_PAYLOAD",
            extracted_json=raw_json,
            parsed_output=None,
        )

    remainder = body[end:]
    suffix_variant = _suffix_variant(remainder)
    if suffix_variant is None:
        return GUIOwlV2CompatibilityInspection(
            strict_parse_success=False,
            compatibility_parse_success=False,
            prefix_variant="action_line_and_tool_call",
            wrapper_variant=wrapper_variant,
            suffix_variant=None,
            first_balanced_json=True,
            canonical_first_payload=True,
            rejection_code=_unsafe_suffix_rejection_code(remainder),
            extracted_json=raw_json,
            parsed_output=None,
        )

    parsed = ParsedGUIOwlV2Output(
        canonical_action=canonical_action,
        action_description=action_description,
        raw_native_output=text,
        raw_tool_call_json=raw_json,
        source_action_name=source_action,
    )
    return GUIOwlV2CompatibilityInspection(
        strict_parse_success=False,
        compatibility_parse_success=True,
        prefix_variant="action_line_and_tool_call",
        wrapper_variant=wrapper_variant,
        suffix_variant=suffix_variant,
        first_balanced_json=True,
        canonical_first_payload=True,
        rejection_code=None,
        extracted_json=raw_json,
        parsed_output=parsed,
    )


def parse_gui_owl_v2_output_compat(
    text: str,
) -> ParsedGUIOwlV2CompatibilityOutput:
    inspection = inspect_gui_owl_v2_output_compatibility(text)
    if not inspection.compatibility_parse_success or inspection.parsed_output is None:
        code = inspection.rejection_code or "UNKNOWN_REJECTION"
        raise ValueError(f"GUI-Owl v2 compatibility parse rejected output: {code}")
    if inspection.extracted_json is None:
        raise AssertionError("accepted compatibility output lacks extracted JSON")
    parsed = inspection.parsed_output
    canonical_tool_call = serialize_gui_owl_v2_tool_call(parsed.canonical_action)
    return ParsedGUIOwlV2CompatibilityOutput(
        parsed_output=parsed,
        parser_path="strict" if inspection.strict_parse_success else "compatibility",
        parser_contract_id=GUI_OWL_V2_COMPATIBILITY_PARSER_CONTRACT_ID,
        wrapper_variant=inspection.wrapper_variant or "unknown",
        suffix_variant=inspection.suffix_variant or "unknown",
        raw_output_sha256=_sha256_text(text),
        extracted_json_sha256=_sha256_text(inspection.extracted_json),
        canonical_tool_call_sha256=_sha256_text(canonical_tool_call),
    )
