"""UI-TARS native mobile prompt and executable action parser."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from causalcache.data.guiodyssey import canonicalize_tool_call
from causalcache.policy.prompt import build_mixed_fidelity_content
from causalcache.schema import ActionType, ExecutableAction


MOBILE_USE_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
Thought: ...
Action: ...

## Action Space
click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='')
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name='')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
wait()
finished(content='')

## User Instruction
{instruction}"""


def build_ui_tars_messages(
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
        initial_text=MOBILE_USE_PROMPT.format(instruction=manifest["trajectory"]["instruction"]),
        final_text="Return the next Thought and Action using the mobile action space above.",
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {"role": "user", "content": content},
    ]


def _action_expression(text: str) -> ast.Call:
    matches = re.findall(r"^\s*Action:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not matches:
        raise ValueError("UI-TARS output does not contain an Action line")
    expression = ast.parse(matches[-1].strip().strip("`"), mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("UI-TARS Action must be a function call")
    return expression


def _keyword_values(call: ast.Call) -> dict[str, Any]:
    values = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("UI-TARS Action cannot use expanded keyword arguments")
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return values


def _point(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    coordinates = re.findall(r"-?\d+", str(value))
    if len(coordinates) < 2:
        raise ValueError(f"cannot parse UI-TARS point: {value}")
    return int(coordinates[0]), int(coordinates[1])


def _current_resized_shape(model_inputs: Any) -> tuple[int, int]:
    grids = model_inputs["image_grid_thw"]
    grid = grids[-1]
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    resized_height = int(grid[1]) * 14
    resized_width = int(grid[2]) * 14
    return resized_width, resized_height


def _normalized_point(value: Any, model_inputs: Any) -> list[int]:
    x, y = _point(value)
    width, height = _current_resized_shape(model_inputs)
    normalized_x = min(1000, max(0, round(x / width * 1000)))
    normalized_y = min(1000, max(0, round(y / height * 1000)))
    return [normalized_x, normalized_y]


def parse_ui_tars_action(text: str, model_inputs: Any) -> ExecutableAction:
    call = _action_expression(text)
    name = call.func.id
    values = _keyword_values(call)
    if name in {"click", "long_press"}:
        point_value = values.get("point", values.get("start_box"))
        if point_value is None:
            raise ValueError(f"{name} requires a point")
        tool_name = "tap" if name == "click" else "long_press"
        return canonicalize_tool_call(
            {
                "function": {
                    "name": tool_name,
                    "arguments": {"coordinate": _normalized_point(point_value, model_inputs)},
                }
            }
        )[0]
    if name == "type":
        return ExecutableAction(
            ActionType.TYPE_TEXT,
            target="text_field_unknown",
            text_argument=str(values["content"]).removesuffix("\n"),
        )
    if name == "scroll":
        direction = str(values["direction"]).casefold()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"unsupported scroll direction: {direction}")
        return ExecutableAction(ActionType.SWIPE, target=f"scroll:{direction}")
    if name == "drag":
        start = _normalized_point(values["start_point"], model_inputs)
        end = _normalized_point(values["end_point"], model_inputs)
        return canonicalize_tool_call(
            {
                "function": {
                    "name": "swipe",
                    "arguments": {"start_coordinate": start, "coordinate": end},
                }
            }
        )[0]
    simple_actions = {
        "press_home": ActionType.HOME,
        "press_back": ActionType.BACK,
        "wait": ActionType.WAIT,
        "finished": ActionType.STOP,
    }
    if name not in simple_actions:
        raise ValueError(f"unsupported UI-TARS action: {name}")
    target = name.removeprefix("press_") if name.startswith("press_") else None
    return ExecutableAction(simple_actions[name], target=target)
