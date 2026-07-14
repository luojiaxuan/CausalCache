"""Policy-visible prompt contract for mixed-fidelity GUI history."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from causalcache.data.guiodyssey import canonicalize_tool_call
from causalcache.schema import ActionType, ExecutableAction


SYSTEM_PROMPT = """You are a frozen GUI action policy. Predict exactly one next executable action from the task, history, and current screenshot. Screenshot coordinates are normalized to integers from 0 to 1000. Return exactly one compact JSON object and no explanation.

Allowed objects:
{"action_type":"tap","coordinate":[x,y]}
{"action_type":"long_press","coordinate":[x,y]}
{"action_type":"type_text","text":"exact text"}
{"action_type":"swipe","start_coordinate":[x,y],"coordinate":[x,y]}
{"action_type":"back"}
{"action_type":"home"}
{"action_type":"enter"}
{"action_type":"wait"}
{"action_type":"stop"}"""


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_mixed_fidelity_content(
    manifest: Mapping[str, Any],
    *,
    decision_step_id: int,
    restored_event_step_ids: Sequence[int],
    image_loader: Callable[[str], Any],
    initial_text: str,
    final_text: str,
) -> list[dict[str, Any]]:
    trajectory = manifest["trajectory"]
    decisions = {
        int(decision["decision_step_id"]): decision for decision in trajectory["decisions"]
    }
    if decision_step_id not in decisions:
        raise ValueError(f"unknown decision step: {decision_step_id}")
    decision = decisions[decision_step_id]
    history_ids = tuple(int(step_id) for step_id in decision["history_event_step_ids"])
    restored_ids = frozenset(int(step_id) for step_id in restored_event_step_ids)
    if not restored_ids.issubset(history_ids):
        raise ValueError("restored events must belong to the decision history")
    events = {int(event["step_id"]): event for event in trajectory["events"]}

    content: list[dict[str, Any]] = [{"type": "text", "text": initial_text}]
    for step_id in history_ids:
        event = events[step_id]
        if step_id in restored_ids:
            content.extend(
                [
                    {"type": "text", "text": f"High-fidelity event {step_id}, before action:"},
                    {"type": "image", "image": image_loader(event["observation_before_path"])},
                    {
                        "type": "text",
                        "text": f"Executed action: {_json_text(event['source_tool_call'])}. After action:",
                    },
                    {"type": "image", "image": image_loader(event["observation_after_path"])},
                ]
            )
        else:
            content.append(
                {
                    "type": "text",
                    "text": f"Low-fidelity event: {_json_text(event['low_fidelity'])}",
                }
            )
    content.extend(
        [
            {"type": "text", "text": "Current observation:"},
            {"type": "image", "image": image_loader(decision["current_observation_path"])},
            {"type": "text", "text": final_text},
        ]
    )
    return content


def build_policy_messages(
    manifest: Mapping[str, Any],
    *,
    decision_step_id: int,
    restored_event_step_ids: Sequence[int],
    image_loader: Callable[[str], Any],
) -> list[dict[str, Any]]:
    content = build_mixed_fidelity_content(
        manifest,
        decision_step_id=decision_step_id,
        restored_event_step_ids=restored_event_step_ids,
        image_loader=image_loader,
        initial_text=f"Task instruction: {manifest['trajectory']['instruction']}",
        final_text="Predict the next executable action.",
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def _first_json_object(text: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise ValueError("policy output does not contain a JSON object")


def parse_policy_action(text: str) -> ExecutableAction:
    value = _first_json_object(text)
    action_type = str(value["action_type"])
    if action_type in {"tap", "long_press"}:
        return canonicalize_tool_call(
            {"function": {"name": action_type, "arguments": {"coordinate": value["coordinate"]}}}
        )[0]
    if action_type == "type_text":
        return ExecutableAction(ActionType.TYPE_TEXT, target="text_field_unknown", text_argument=str(value["text"]))
    if action_type == "swipe":
        return canonicalize_tool_call(
            {
                "function": {
                    "name": "swipe",
                    "arguments": {
                        "start_coordinate": value["start_coordinate"],
                        "coordinate": value["coordinate"],
                    },
                }
            }
        )[0]
    simple_actions = {
        "back": ActionType.BACK,
        "home": ActionType.HOME,
        "enter": ActionType.ENTER,
        "wait": ActionType.WAIT,
        "stop": ActionType.STOP,
    }
    if action_type not in simple_actions:
        raise ValueError(f"unsupported policy action type: {action_type}")
    return ExecutableAction(simple_actions[action_type], target=action_type if action_type not in {"wait", "stop"} else None)
