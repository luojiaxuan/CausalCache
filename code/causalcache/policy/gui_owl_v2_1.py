"""Versioned official-tool interface for the restoration v2.1 rescue."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.policy.gui_owl_v2 import (
    CANONICAL_GUI_OWL_V2_ACTIONS,
    GUI_OWL_V2_PARAMETERS_BY_ACTION,
    GUI_OWL_V2_SYSTEM_BUTTONS,
    GUIOwlV2Action,
    _canonical_action,
    _parse_tool_call_json,
    build_gui_owl_v2_mixed_fidelity_messages,
)


GUI_OWL_V2_1_PROTOCOL_ID = "causalcache_restoration_v2_1_official_tool_interface"
GUI_OWL_V2_1_SYSTEM_PROMPT = (
    "You are a frozen single-step GUI action policy. For this assistant turn, call "
    "mobile_use exactly once with the next executable action. Return only that tool "
    "call: no Action line, analysis, observation, tool response, or second call. "
    "Coordinates are base-10 integers in [0,999]. Use exactly the frozen "
    "action-dependent arguments. Text arguments must already use Unicode NFKC "
    "normalization."
)
GUI_OWL_V2_1_FINAL_USER_INSTRUCTION = (
    "Call mobile_use exactly once with the next executable action."
)


def _coordinate_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": 999},
        "minItems": 2,
        "maxItems": 2,
    }


def _nfkc_text_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "description": "Text already normalized with Unicode NFKC.",
    }


def _action_variant(
    action: str,
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "const": action},
            **dict(properties),
        },
        "required": ["action", *properties],
        "additionalProperties": False,
    }


GUI_OWL_V2_1_MOBILE_USE_TOOL = {
    "type": "function",
    "function": {
        "name": "mobile_use",
        "description": "Execute exactly one action on the visible mobile UI.",
        "parameters": {
            "type": "object",
            "oneOf": [
                _action_variant("click", {"coordinate": _coordinate_schema()}),
                _action_variant("long_press", {"coordinate": _coordinate_schema()}),
                _action_variant(
                    "swipe",
                    {
                        "coordinate": _coordinate_schema(),
                        "coordinate2": _coordinate_schema(),
                    },
                ),
                _action_variant("type", {"text": _nfkc_text_schema()}),
                _action_variant(
                    "system_button",
                    {
                        "button": {
                            "type": "string",
                            "enum": list(GUI_OWL_V2_SYSTEM_BUTTONS),
                        }
                    },
                ),
                _action_variant("open", {"text": _nfkc_text_schema()}),
                _action_variant("wait", {}),
                _action_variant("answer", {"text": _nfkc_text_schema()}),
                _action_variant(
                    "terminate",
                    {"status": {"type": "string", "const": "success"}},
                ),
            ]
        },
    },
}


@dataclass(frozen=True)
class ParsedGUIOwlV21Output:
    canonical_action: GUIOwlV2Action
    raw_native_output: str
    raw_tool_call_json: str
    source_action_name: str


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_gui_owl_v2_1_output(text: str) -> ParsedGUIOwlV21Output:
    """Require one complete official tool call and no surrounding output."""
    if not isinstance(text, str):
        raise TypeError("GUI-Owl v2.1 output must be a string")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("GUI-Owl v2.1 output must be valid Unicode text") from error
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        raise ValueError("GUI-Owl v2.1 output requires exactly one complete tool call")
    match = re.fullmatch(
        r"<tool_call>\n(?P<call>\{[^\r\n]*\})\n</tool_call>",
        text,
    )
    if match is None:
        raise ValueError(
            "GUI-Owl v2.1 output must exactly match the official tool-call envelope"
        )
    raw_json = match.group("call")
    arguments = _parse_tool_call_json(raw_json)
    if "text" in arguments:
        if type(arguments["text"]) is not str:
            raise ValueError("GUI-Owl v2.1 text arguments must be JSON strings")
        if arguments["text"] != unicodedata.normalize("NFKC", arguments["text"]):
            raise ValueError("GUI-Owl v2.1 text arguments must already use Unicode NFKC")
    canonical_action, source_action = _canonical_action(arguments)
    if source_action != canonical_action.action:
        raise ValueError("GUI-Owl v2.1 requires a canonical action name")
    return ParsedGUIOwlV21Output(
        canonical_action=canonical_action,
        raw_native_output=text,
        raw_tool_call_json=raw_json,
        source_action_name=source_action,
    )


def serialize_gui_owl_v2_1_teacher_target(action: GUIOwlV2Action) -> str:
    if not isinstance(action, GUIOwlV2Action):
        raise TypeError("action must be a GUIOwlV2Action")
    arguments = json.dumps(
        action.arguments(),
        ensure_ascii=False,
        sort_keys=False,
        separators=(", ", ": "),
        allow_nan=False,
    )
    payload = f'{{"name": "mobile_use", "arguments": {arguments}}}'
    return f"<tool_call>\n{payload}\n</tool_call>"


def validate_gui_owl_v2_1_native_messages(messages: Any) -> int:
    if isinstance(messages, (str, bytes, bytearray, Mapping)):
        raise TypeError("GUI-Owl v2.1 messages must be a two-message sequence")
    try:
        records = tuple(messages)
    except TypeError as error:
        raise TypeError("GUI-Owl v2.1 messages must be a two-message sequence") from error
    if len(records) != 2 or any(not isinstance(record, Mapping) for record in records):
        raise ValueError("GUI-Owl v2.1 requires exactly one system and one user message")
    system, user = records
    if system != {
        "role": "system",
        "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
    }:
        raise ValueError("GUI-Owl v2.1 system message drifted")
    if set(user) != {"role", "content"} or user.get("role") != "user":
        raise ValueError("GUI-Owl v2.1 user message envelope drifted")
    content = user.get("content")
    if isinstance(content, (str, bytes, bytearray, Mapping)):
        raise TypeError("GUI-Owl v2.1 user content must be a non-empty block sequence")
    try:
        blocks = tuple(content)
    except TypeError as error:
        raise TypeError(
            "GUI-Owl v2.1 user content must be a non-empty block sequence"
        ) from error
    if not blocks or any(not isinstance(block, Mapping) for block in blocks):
        raise ValueError("GUI-Owl v2.1 user content must contain mapping blocks")
    image_count = 0
    for block in blocks:
        if block.get("type") == "text":
            if set(block) != {"type", "text"} or not isinstance(block["text"], str):
                raise ValueError("GUI-Owl v2.1 text block schema drifted")
        elif block.get("type") == "image":
            if set(block) != {"type", "image"}:
                raise ValueError("GUI-Owl v2.1 image block schema drifted")
            image_count += 1
        else:
            raise ValueError("GUI-Owl v2.1 messages allow only text and image blocks")
    if image_count <= 0:
        raise ValueError("GUI-Owl v2.1 messages require the current observation image")
    if blocks[-1] != {
        "type": "text",
        "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    }:
        raise ValueError("GUI-Owl v2.1 final user instruction drifted")
    return image_count


def build_gui_owl_v2_1_mixed_fidelity_messages(
    manifest: Mapping[str, Any],
    *,
    trajectory_id: str,
    decision_step_id: int,
    restored_event_step_ids: Sequence[int],
    image_bytes_loader: Callable[[str], bytes],
    image_decoder: Callable[[bytes], Any],
) -> list[dict[str, Any]]:
    messages = build_gui_owl_v2_mixed_fidelity_messages(
        manifest,
        trajectory_id=trajectory_id,
        decision_step_id=decision_step_id,
        restored_event_step_ids=restored_event_step_ids,
        image_bytes_loader=image_bytes_loader,
        image_decoder=image_decoder,
    )
    user_content = messages[1]["content"]
    if not isinstance(user_content, list) or not user_content:
        raise ValueError("frozen v2 builder returned invalid user content")
    upgraded_content = [dict(block) for block in user_content]
    upgraded_content[-1] = {
        "type": "text",
        "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    }
    upgraded = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": upgraded_content},
    ]
    validate_gui_owl_v2_1_native_messages(upgraded)
    return upgraded


def validate_gui_owl_v2_1_tool_schema() -> None:
    function = GUI_OWL_V2_1_MOBILE_USE_TOOL.get("function")
    if (
        set(GUI_OWL_V2_1_MOBILE_USE_TOOL) != {"type", "function"}
        or GUI_OWL_V2_1_MOBILE_USE_TOOL["type"] != "function"
        or not isinstance(function, Mapping)
        or set(function) != {"name", "description", "parameters"}
        or function.get("name") != "mobile_use"
    ):
        raise ValueError("GUI-Owl v2.1 tool envelope drifted")
    parameters = function.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or set(parameters) != {"type", "oneOf"}
        or parameters.get("type") != "object"
    ):
        raise ValueError("GUI-Owl v2.1 tool parameters drifted")
    variants = parameters["oneOf"]
    if not isinstance(variants, list) or len(variants) != len(
        CANONICAL_GUI_OWL_V2_ACTIONS
    ):
        raise ValueError("GUI-Owl v2.1 tool action count drifted")
    observed: list[str] = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("GUI-Owl v2.1 action schema must be an object")
        properties = variant.get("properties")
        if (
            variant.get("type") != "object"
            or variant.get("additionalProperties") is not False
            or not isinstance(properties, Mapping)
            or not isinstance(variant.get("required"), list)
        ):
            raise ValueError("GUI-Owl v2.1 action variant drifted")
        action_schema = properties.get("action")
        if not isinstance(action_schema, Mapping):
            raise ValueError("GUI-Owl v2.1 action variant lacks action")
        action = action_schema.get("const")
        if action not in CANONICAL_GUI_OWL_V2_ACTIONS:
            raise ValueError("GUI-Owl v2.1 action variant is not canonical")
        expected = ["action", *GUI_OWL_V2_PARAMETERS_BY_ACTION[action]]
        if variant["required"] != expected or set(properties) != set(expected):
            raise ValueError("GUI-Owl v2.1 action arguments drifted")
        observed.append(action)
    if tuple(observed) != CANONICAL_GUI_OWL_V2_ACTIONS:
        raise ValueError("GUI-Owl v2.1 action schema ordering drifted")


validate_gui_owl_v2_1_tool_schema()
