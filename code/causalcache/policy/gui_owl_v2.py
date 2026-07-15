"""Frozen GUI-Owl v2 action and post-state-only prompt interfaces."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.low_fidelity_v2 import (
    LowFidelityEventV2,
    serialize_low_fidelity_v2,
    sha256_bytes,
)


CANONICAL_GUI_OWL_V2_ACTIONS = (
    "click",
    "long_press",
    "swipe",
    "type",
    "system_button",
    "open",
    "wait",
    "answer",
    "terminate",
)
GUI_OWL_V2_ACTION_ALIASES = {"tap": "click", "open_app": "open"}
GUI_OWL_V2_SYSTEM_BUTTONS = ("Back", "Home", "Enter")
GUI_OWL_V2_ARGUMENT_KEY_ORDER = (
    "action",
    "coordinate",
    "coordinate2",
    "text",
    "button",
    "status",
)
GUI_OWL_V2_TEACHER_CARRIER = "Action: Execute the selected mobile action.\n"
GUI_OWL_V2_DECISION_STEPS = (4, 5, 6)


_GUI_OWL_V2_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name_for_human": "mobile_use",
        "name": "mobile_use",
        "description": "Use the visible mobile UI and return exactly one executable action.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(CANONICAL_GUI_OWL_V2_ACTIONS),
                },
                "coordinate": {
                    "type": "array",
                    "description": "Two integer coordinates in [0,999].",
                },
                "coordinate2": {
                    "type": "array",
                    "description": "Swipe endpoint as two integers in [0,999].",
                },
                "text": {"type": "string"},
                "button": {
                    "type": "string",
                    "enum": list(GUI_OWL_V2_SYSTEM_BUTTONS),
                },
                "status": {"type": "string", "enum": ["success"]},
            },
            "required": ["action"],
        },
        "args_format": "Format the arguments as a JSON object.",
    },
}

GUI_OWL_V2_SYSTEM_PROMPT = """# Tools

You are provided with one function signature within <tools></tools> XML tags:
<tools>
{tool_spec}
</tools>

Return exactly two parts in this order:
1) One non-empty line beginning with `Action:`.
2) One <tool_call> block containing compact JSON with name `mobile_use` and its arguments.

Use only the canonical actions shown in the tool schema. Coordinates are integer values from 0
through 999. After `action`, use exactly these arguments:
- click or long_press: coordinate
- swipe: coordinate and coordinate2
- type, open, or answer: text
- system_button: button, which must be Back, Home, or Enter
- wait: no additional argument
- terminate: status, which must be success
Do not add duration or time. Return no thinking block or other prose.""".format(
    tool_spec=json.dumps(_GUI_OWL_V2_TOOL_SPEC, ensure_ascii=False, separators=(",", ":"))
)


GUI_OWL_V2_PARAMETERS_BY_ACTION = {
    "click": ("coordinate",),
    "long_press": ("coordinate",),
    "swipe": ("coordinate", "coordinate2"),
    "type": ("text",),
    "system_button": ("button",),
    "open": ("text",),
    "wait": (),
    "answer": ("text",),
    "terminate": ("status",),
}


def _coordinate(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a JSON list containing exactly two integers")
    if any(type(item) is not int for item in value):
        raise ValueError(f"{name} values must be base-10 JSON integers")
    if any(item < 0 or item > 999 for item in value):
        raise ValueError(f"{name} values must be in [0, 999]")
    return value[0], value[1]


@dataclass(frozen=True)
class GUIOwlV2Action:
    action: str
    coordinate: tuple[int, int] | None = None
    coordinate2: tuple[int, int] | None = None
    text: str | None = None
    button: str | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        if self.action not in CANONICAL_GUI_OWL_V2_ACTIONS:
            raise ValueError(f"unsupported canonical GUI-Owl v2 action: {self.action}")
        expected = GUI_OWL_V2_PARAMETERS_BY_ACTION[self.action]
        values = {
            "coordinate": self.coordinate,
            "coordinate2": self.coordinate2,
            "text": self.text,
            "button": self.button,
            "status": self.status,
        }
        present = tuple(name for name in GUI_OWL_V2_ARGUMENT_KEY_ORDER[1:] if values[name] is not None)
        if present != expected:
            raise ValueError(f"{self.action} requires exactly {expected}; got {present}")
        if self.coordinate is not None:
            _coordinate(list(self.coordinate), "coordinate")
        if self.coordinate2 is not None:
            _coordinate(list(self.coordinate2), "coordinate2")
        if self.text is not None:
            if not isinstance(self.text, str):
                raise TypeError("text must be a string")
            if self.text != unicodedata.normalize("NFKC", self.text):
                raise ValueError("text must already use Unicode NFKC normalization")
        if self.button is not None and self.button not in GUI_OWL_V2_SYSTEM_BUTTONS:
            raise ValueError("system_button accepts only Back, Home, or Enter")
        if self.status is not None and self.status != "success":
            raise ValueError("terminate accepts only status=success")

    def arguments(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "action": self.action,
            "coordinate": list(self.coordinate) if self.coordinate is not None else None,
            "coordinate2": list(self.coordinate2) if self.coordinate2 is not None else None,
            "text": self.text,
            "button": self.button,
            "status": self.status,
        }
        return {
            key: values[key]
            for key in GUI_OWL_V2_ARGUMENT_KEY_ORDER
            if values[key] is not None
        }


@dataclass(frozen=True)
class ParsedGUIOwlV2Output:
    canonical_action: GUIOwlV2Action
    action_description: str
    raw_native_output: str
    raw_tool_call_json: str
    source_action_name: str


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _parse_tool_call_json(raw_json: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw_json,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("GUI-Owl v2 tool call must contain valid JSON") from error
    if not isinstance(value, Mapping) or set(value) != {"name", "arguments"}:
        raise ValueError("GUI-Owl v2 tool call requires exactly name and arguments")
    if value["name"] != "mobile_use":
        raise ValueError("GUI-Owl v2 tool call must target mobile_use")
    arguments = value["arguments"]
    if not isinstance(arguments, Mapping):
        raise ValueError("GUI-Owl v2 arguments must be a JSON object")
    return arguments


def _canonical_action(arguments: Mapping[str, Any]) -> tuple[GUIOwlV2Action, str]:
    if "action" not in arguments or type(arguments["action"]) is not str:
        raise ValueError("GUI-Owl v2 arguments require a string action")
    source_action = arguments["action"]
    action_name = GUI_OWL_V2_ACTION_ALIASES.get(source_action, source_action)
    if action_name not in CANONICAL_GUI_OWL_V2_ACTIONS:
        raise ValueError(f"unsupported GUI-Owl v2 action: {source_action}")
    expected_keys = {"action", *GUI_OWL_V2_PARAMETERS_BY_ACTION[action_name]}
    if set(arguments) != expected_keys:
        raise ValueError(f"{source_action} arguments must contain exactly {sorted(expected_keys)}")

    coordinate = None
    coordinate2 = None
    text = None
    button = None
    status = None
    if "coordinate" in arguments:
        coordinate = _coordinate(arguments["coordinate"], "coordinate")
    if "coordinate2" in arguments:
        coordinate2 = _coordinate(arguments["coordinate2"], "coordinate2")
    if "text" in arguments:
        if type(arguments["text"]) is not str:
            raise ValueError("text must be a JSON string")
        text = unicodedata.normalize("NFKC", arguments["text"])
    if "button" in arguments:
        if type(arguments["button"]) is not str or arguments["button"] not in GUI_OWL_V2_SYSTEM_BUTTONS:
            raise ValueError("system_button accepts only Back, Home, or Enter")
        button = arguments["button"]
    if "status" in arguments:
        if arguments["status"] != "success":
            raise ValueError("terminate accepts only status=success")
        status = "success"
    return (
        GUIOwlV2Action(
            action=action_name,
            coordinate=coordinate,
            coordinate2=coordinate2,
            text=text,
            button=button,
            status=status,
        ),
        source_action,
    )


def parse_gui_owl_v2_output(text: str) -> ParsedGUIOwlV2Output:
    if not isinstance(text, str):
        raise TypeError("GUI-Owl v2 output must be a string")
    stripped = text.strip()
    match = re.fullmatch(
        r"Action:[ \t]*(?P<description>[^\r\n]+)\r?\n"
        r"<tool_call>[ \t]*\r?\n?(?P<call>\{.*\})\r?\n?</tool_call>",
        stripped,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("GUI-Owl v2 output must be one Action line followed by one tool call")
    description = match.group("description").strip()
    if not description:
        raise ValueError("GUI-Owl v2 output must contain a non-empty Action description")
    raw_json = match.group("call")
    arguments = _parse_tool_call_json(raw_json)
    canonical_action, source_action = _canonical_action(arguments)
    return ParsedGUIOwlV2Output(
        canonical_action=canonical_action,
        action_description=description,
        raw_native_output=text,
        raw_tool_call_json=raw_json,
        source_action_name=source_action,
    )


def serialize_gui_owl_v2_tool_call(action: GUIOwlV2Action) -> str:
    tool_call = {"name": "mobile_use", "arguments": action.arguments()}
    payload = json.dumps(tool_call, ensure_ascii=False, separators=(",", ":"))
    return f"<tool_call>\n{payload}\n</tool_call>"


def serialize_gui_owl_v2_teacher_target(action: GUIOwlV2Action) -> str:
    return GUI_OWL_V2_TEACHER_CARRIER + serialize_gui_owl_v2_tool_call(action)


def normalized_coordinate_to_pixel(value: int, extent: int) -> int:
    if type(value) is not int or value < 0 or value > 999:
        raise ValueError("normalized coordinate must be an integer in [0, 999]")
    if type(extent) is not int or extent <= 0:
        raise ValueError("pixel extent must be a positive integer")
    pixel = (2 * value * (extent - 1) + 999) // 1998
    if pixel < 0 or pixel >= extent:
        raise AssertionError("normalized coordinate scaling produced an out-of-bounds pixel")
    return pixel


def _pixel_coordinate(
    coordinate: tuple[int, int],
    *,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    return (
        normalized_coordinate_to_pixel(coordinate[0], screen_width),
        normalized_coordinate_to_pixel(coordinate[1], screen_height),
    )


def gui_owl_v2_action_to_androidworld(
    action: GUIOwlV2Action,
    *,
    screen_width: int,
    screen_height: int,
) -> dict[str, Any]:
    if action.action in {"click", "long_press"}:
        assert action.coordinate is not None
        x, y = _pixel_coordinate(
            action.coordinate,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        return {"action_type": action.action, "x": x, "y": y}
    if action.action == "swipe":
        assert action.coordinate is not None and action.coordinate2 is not None
        start = _pixel_coordinate(
            action.coordinate,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        end = _pixel_coordinate(
            action.coordinate2,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        return {"action_type": "swipe", "direction": [*start, *end]}
    if action.action == "type":
        return {"action_type": "input_text", "text": action.text}
    if action.action == "system_button":
        button_actions = {
            "Back": "navigate_back",
            "Home": "navigate_home",
            "Enter": "keyboard_enter",
        }
        assert action.button is not None
        return {"action_type": button_actions[action.button]}
    if action.action == "open":
        return {"action_type": "open_app", "app_name": action.text}
    if action.action == "wait":
        return {"action_type": "wait"}
    if action.action == "answer":
        return {"action_type": "answer", "text": action.text}
    if action.action == "terminate":
        return {"action_type": "status", "goal_status": "task_complete"}
    raise AssertionError(f"unhandled canonical GUI-Owl v2 action: {action.action}")


def _decision_by_step(trajectory: Mapping[str, Any], decision_step_id: int) -> Mapping[str, Any]:
    if not isinstance(trajectory["decisions"], list):
        raise ValueError("v2 decisions must be a JSON array")
    decision_records = tuple(trajectory["decisions"])
    decisions = {}
    for decision in decision_records:
        step_id = decision["decision_step_id"]
        if type(step_id) is not int or step_id < 1:
            raise ValueError("v2 decision step ids must be positive integers")
        decisions[step_id] = decision
    if len(decisions) != len(decision_records):
        raise ValueError("v2 decision step ids must be unique")
    if decision_step_id not in decisions:
        raise ValueError(f"unknown v2 decision step: {decision_step_id}")
    return decisions[decision_step_id]


def _validated_summary(event: Mapping[str, Any], *, expected_step_id: int) -> str:
    summary = LowFidelityEventV2.from_mapping(event["low_fidelity_v2"])
    if summary.step_id != expected_step_id:
        raise ValueError("v2 low-fidelity step id must match its archived event")
    serialized = serialize_low_fidelity_v2(summary)
    stored = event.get("low_fidelity_v2_serialized")
    if stored != serialized.decode("utf-8"):
        raise ValueError("stored v2 low-fidelity serialization does not match its object")
    if event.get("low_fidelity_v2_sha256") != sha256_bytes(serialized):
        raise ValueError("stored v2 low-fidelity SHA256 does not match its serialization")
    return stored


def build_gui_owl_v2_mixed_fidelity_messages(
    manifest: Mapping[str, Any],
    *,
    trajectory_id: str,
    decision_step_id: int,
    restored_event_step_ids: Sequence[int],
    image_bytes_loader: Callable[[str], bytes],
    image_decoder: Callable[[bytes], Any],
) -> list[dict[str, Any]]:
    if decision_step_id not in GUI_OWL_V2_DECISION_STEPS:
        raise ValueError("v2 decision step must be one of 4, 5, or 6")
    if not isinstance(manifest["trajectories"], list):
        raise ValueError("v2 trajectories must be a JSON array")
    trajectory_records = tuple(manifest["trajectories"])
    trajectories = {}
    for trajectory in trajectory_records:
        source_id = trajectory["source_id"]
        if type(source_id) is not str or not source_id:
            raise ValueError("v2 trajectory ids must be non-empty strings")
        trajectories[source_id] = trajectory
    if len(trajectories) != len(trajectory_records):
        raise ValueError("v2 trajectory ids must be unique")
    if trajectory_id not in trajectories:
        raise ValueError(f"unknown v2 trajectory: {trajectory_id}")
    trajectory = trajectories[trajectory_id]
    decision = _decision_by_step(trajectory, decision_step_id)
    if not isinstance(decision["history_event_step_ids"], list) or not isinstance(
        decision["candidate_event_step_ids"], list
    ):
        raise ValueError("v2 history and candidate ids must be JSON arrays")
    history_ids = tuple(decision["history_event_step_ids"])
    candidate_ids = tuple(decision["candidate_event_step_ids"])
    current_equivalent = decision["current_equivalent_event_step_id"]
    if any(type(step_id) is not int or step_id < 1 for step_id in (*history_ids, *candidate_ids)):
        raise ValueError("v2 history and candidate ids must be positive integers")
    if type(current_equivalent) is not int or current_equivalent < 1:
        raise ValueError("v2 current-equivalent event id must be a positive integer")
    if history_ids != tuple(range(1, decision_step_id)):
        raise ValueError("v2 history step ids must be the complete ordered prefix")
    if current_equivalent != history_ids[-1] or candidate_ids != history_ids[:-1]:
        raise ValueError("v2 candidates must exclude only the current-equivalent latest event")
    restored_sequence = tuple(restored_event_step_ids)
    if any(type(step_id) is not int for step_id in restored_sequence):
        raise TypeError("restored v2 event ids must be integers")
    if len(restored_sequence) != len(set(restored_sequence)):
        raise ValueError("restored v2 event ids must be unique")
    restored_ids = frozenset(restored_sequence)
    if not restored_ids.issubset(candidate_ids):
        raise ValueError("restored v2 events must belong to the candidate set")

    if not isinstance(trajectory["events"], list):
        raise ValueError("v2 events must be a JSON array")
    event_records = tuple(trajectory["events"])
    events = {}
    for event in event_records:
        step_id = event["step_id"]
        if type(step_id) is not int or step_id < 1:
            raise ValueError("v2 event step ids must be positive integers")
        events[step_id] = event
    if len(events) != len(event_records):
        raise ValueError("v2 event step ids must be unique")
    if any(step_id not in events for step_id in history_ids):
        raise ValueError("v2 trajectory is missing a history event")
    current_path = decision["current_observation_path"]
    current_sha256 = decision["current_observation_sha256"]
    if type(current_path) is not str or not current_path:
        raise ValueError("current observation path must be a non-empty string")
    if type(current_sha256) is not str:
        raise ValueError("current observation SHA256 must be a string")
    if re.fullmatch(r"[0-9a-f]{64}", current_sha256) is None:
        raise ValueError("current observation SHA256 must be lowercase hexadecimal")
    latest = events[current_equivalent]
    if (
        latest["observation_after_path"],
        latest["observation_after_sha256"],
    ) != (current_path, current_sha256):
        raise ValueError("latest event post-state must be identical to the current observation")
    for step_id in candidate_ids:
        event = events[step_id]
        event_path = event["observation_after_path"]
        event_sha256 = event["observation_after_sha256"]
        if type(event_path) is not str or not event_path:
            raise ValueError("candidate post-state path must be a non-empty string")
        if type(event_sha256) is not str:
            raise ValueError("candidate post-state SHA256 must be a string")
        if re.fullmatch(r"[0-9a-f]{64}", event_sha256) is None:
            raise ValueError("candidate post-state SHA256 must be lowercase hexadecimal")
        if event_path == current_path:
            raise ValueError("candidate post-state must not equal the current observation")

    summaries = {
        step_id: _validated_summary(events[step_id], expected_step_id=step_id)
        for step_id in history_ids
    }

    def load_verified_image(path: str, expected_sha256: str) -> Any:
        raw = image_bytes_loader(path)
        if not isinstance(raw, bytes):
            raise TypeError("v2 image_bytes_loader must return bytes")
        if sha256_bytes(raw) != expected_sha256:
            raise ValueError(f"artifact image SHA256 mismatch: {path}")
        return image_decoder(raw)

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Please generate the next move from the task, event summaries, restored "
                f"post-action states, and current observation.\n\nInstruction: {trajectory['instruction']}"
            ),
        }
    ]
    for step_id in history_ids:
        event = events[step_id]
        summary = summaries[step_id]
        content.append({"type": "text", "text": f"Event summary:\n{summary}"})
        if step_id in restored_ids:
            content.append(
                {
                    "type": "image",
                    "image": load_verified_image(
                        event["observation_after_path"],
                        event["observation_after_sha256"],
                    ),
                }
            )
    content.extend(
        [
            {"type": "text", "text": "Current observation:"},
            {
                "type": "image",
                "image": load_verified_image(current_path, current_sha256),
            },
            {
                "type": "text",
                "text": "Return the next action using the required native mobile_use format.",
            },
        ]
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]
