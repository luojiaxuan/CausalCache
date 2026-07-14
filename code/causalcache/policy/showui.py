"""ShowUI native phone-navigation prompt and executable action parser."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any

from causalcache.data.guiodyssey import canonicalize_tool_call
from causalcache.policy.prompt import build_mixed_fidelity_content
from causalcache.schema import ActionType, ExecutableAction


SHOWUI_PHONE_PROMPT = """You are an assistant trained to navigate the phone screen.
Given a task instruction, a screen observation, and an action history sequence,
output the next action and wait for the next observation.

Here is the action space:
1. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required.
2. `SWIPE`: Swipe the screen, value is not applicable and the position [[x1,y1], [x2,y2]] is the start and end position of the swipe operation.
3. `TAP`: Tap on an element, value is not applicable and the position [x,y] is required.
4. `ANSWER`: Answer the question, value is the status (e.g., 'task complete') and the position is not applicable.
5. `ENTER`: Enter operation, value and position are not applicable.

Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents relative coordinates on the screenshot and should be scaled to a range of 0-1."""


def build_showui_messages(
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
        initial_text=(
            f"{SHOWUI_PHONE_PROMPT}\n\n"
            f"Task: {manifest['trajectory']['instruction']}"
        ),
        final_text="Output exactly one next action dictionary and wait for the next observation.",
    )
    return [{"role": "user", "content": content}]


def _single_action_dict(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3:
            raise ValueError("ShowUI fenced output is incomplete")
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        value = ast.literal_eval(candidate)
    except (SyntaxError, ValueError) as error:
        raise ValueError("ShowUI output must be exactly one dictionary") from error
    if not isinstance(value, Mapping):
        raise ValueError("ShowUI output must be a dictionary")
    return value


def _normalized_point(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("ShowUI position must contain x and y")
    if any(isinstance(coordinate, bool) or not isinstance(coordinate, Real) for coordinate in value):
        raise ValueError("ShowUI position coordinates must be numeric")
    x, y = (float(coordinate) for coordinate in value)
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError("ShowUI position coordinates must be in [0, 1]")
    return [round(x * 1000), round(y * 1000)]


def parse_showui_action(text: str, model_inputs: Any) -> ExecutableAction:
    del model_inputs
    value = _single_action_dict(text)
    action = str(value["action"]).upper()
    position = value.get("position")
    if action == "TAP":
        return canonicalize_tool_call(
            {
                "function": {
                    "name": "tap",
                    "arguments": {"coordinate": _normalized_point(position)},
                }
            }
        )[0]
    if action == "INPUT":
        if position is not None:
            _normalized_point(position)
        text_argument = value.get("value")
        if not isinstance(text_argument, str):
            raise ValueError("ShowUI INPUT requires a string value")
        return ExecutableAction(
            ActionType.TYPE_TEXT,
            target="text_field_unknown",
            text_argument=text_argument,
        )
    if action == "SWIPE":
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ValueError("ShowUI SWIPE requires start and end positions")
        start, end = (_normalized_point(point) for point in position)
        return canonicalize_tool_call(
            {
                "function": {
                    "name": "swipe",
                    "arguments": {"start_coordinate": start, "coordinate": end},
                }
            }
        )[0]
    if action == "ANSWER":
        return ExecutableAction(ActionType.STOP)
    if action == "ENTER":
        return ExecutableAction(ActionType.ENTER, target="enter")
    raise ValueError(f"unsupported ShowUI action: {action}")
