"""OpenCUA prompt and safe PyAutoGUI-to-executable parser."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from causalcache.data.guiodyssey import canonicalize_tool_call
from causalcache.policy.coordinates import normalized_resized_point
from causalcache.policy.prompt import build_mixed_fidelity_content
from causalcache.schema import ActionType, ExecutableAction


OPEN_CUA_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and screenshots of the screen. You need to perform a series of PyAutoGUI actions to complete the task.

For each step, provide your response in this format:

Thought: briefly assess progress and choose the next action.
Action: describe the target and intended interaction.
Code: output exactly one executable pyautogui.* call, or computer.terminate(status='success') when the task is complete."""


def build_open_cua_messages(
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
        initial_text=f"# Task Instruction:\n{manifest['trajectory']['instruction']}",
        final_text=(
            "Generate the next move from the screenshots, task instruction, and previous "
            "events. Return exactly one action."
        ),
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": OPEN_CUA_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]


def _action_call(text: str) -> tuple[str, ast.Call]:
    candidates = re.findall(
        r"^\s*((?:pyautogui|computer)\.[A-Za-z_]+\s*\(.*\))\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not candidates:
        raise ValueError("OpenCUA output does not contain a supported code line")
    expression = ast.parse(candidates[-1].strip().strip("`"), mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Attribute):
        raise ValueError("OpenCUA code must be a function call")
    namespace = expression.func.value
    if not isinstance(namespace, ast.Name) or namespace.id not in {"pyautogui", "computer"}:
        raise ValueError("OpenCUA code uses an unsupported namespace")
    return f"{namespace.id}.{expression.func.attr}", expression


def _arguments(call: ast.Call) -> tuple[list[Any], dict[str, Any]]:
    positional = [ast.literal_eval(value) for value in call.args]
    keyword = {}
    for item in call.keywords:
        if item.arg is None:
            raise ValueError("OpenCUA action cannot expand keyword arguments")
        keyword[item.arg] = ast.literal_eval(item.value)
    return positional, keyword


def _value(positional: list[Any], keyword: dict[str, Any], name: str, index: int) -> Any:
    if name in keyword:
        return keyword[name]
    if len(positional) > index:
        return positional[index]
    raise ValueError(f"OpenCUA action is missing {name}")


def parse_open_cua_action(text: str, model_inputs: Any) -> ExecutableAction:
    name, call = _action_call(text)
    positional, keyword = _arguments(call)
    if name == "pyautogui.click":
        point = [
            _value(positional, keyword, "x", 0),
            _value(positional, keyword, "y", 1),
        ]
        return canonicalize_tool_call(
            {
                "function": {
                    "name": "tap",
                    "arguments": {"coordinate": normalized_resized_point(point, model_inputs)},
                }
            }
        )[0]
    if name == "pyautogui.write":
        content = _value(positional, keyword, "message", 0)
        return ExecutableAction(
            ActionType.TYPE_TEXT,
            target="text_field_unknown",
            text_argument=str(content),
        )
    if name == "pyautogui.press":
        key = str(_value(positional, keyword, "key", 0)).casefold()
        key_actions = {
            "enter": ActionType.ENTER,
            "return": ActionType.ENTER,
            "home": ActionType.HOME,
            "back": ActionType.BACK,
            "esc": ActionType.BACK,
        }
        if key not in key_actions:
            raise ValueError(f"unsupported OpenCUA key: {key}")
        return ExecutableAction(key_actions[key], target=key)
    if name == "pyautogui.scroll":
        amount = float(_value(positional, keyword, "clicks", 0))
        if amount == 0:
            raise ValueError("zero scroll has no executable direction")
        direction = "up" if amount > 0 else "down"
        return ExecutableAction(ActionType.SWIPE, target=f"scroll:{direction}")
    if name == "computer.terminate":
        return ExecutableAction(ActionType.STOP)
    raise ValueError(f"unsupported OpenCUA action: {name}")
