"""Pinned GUI-Owl-1.5 AndroidWorld prompt, history, and action adapter."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.data.guiodyssey import canonicalize_tool_call
from causalcache.schema import ActionType, ExecutableAction


GUI_OWL_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `key`: Perform a key event on the mobile device.\n    - This supports adb's `keyevent` syntax.\n    - Examples: \"volume_up\", \"volume_down\", \"power\", \"camera\", \"clear\".\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `system_button`: Press the system button.\n* `open`: Open an app on the device.\n* `wait`: Wait specified seconds for the change to happen.\n* `answer`: Terminate the current task and output the answer.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one for Action.
- Do not output anything else outside those two parts.
- If finishing, use action=terminate in the tool call.'''


def _action_description(output: str) -> str:
    match = re.fullmatch(
        r"\s*Action:\s*(?P<description>[^\n]+)\n\s*<tool_call>.*</tool_call>\s*",
        output,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("GUI-Owl history output must use the native response format")
    return match.group("description").strip()


def build_gui_owl_native_messages(
    *,
    instruction: str,
    screenshots: Sequence[Any],
    action_outputs: Sequence[str],
    maximum_visible_images: int = 5,
) -> list[dict[str, Any]]:
    if not instruction:
        raise ValueError("GUI-Owl instruction cannot be empty")
    if len(screenshots) != len(action_outputs) + 1:
        raise ValueError("native history requires one more screenshot than executed actions")
    if maximum_visible_images < 1:
        raise ValueError("maximum_visible_images must be positive")

    first_visible = max(0, len(screenshots) - maximum_visible_images)
    older_actions = [
        f"Step{index + 1}: {_action_description(action_outputs[index])}"
        for index in range(first_visible)
    ]
    previous_actions = "\n".join(older_actions) if older_actions else "No previous action."
    instruction_prompt = (
        "Please generate the next move according to the UI screenshot, instruction "
        "and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n{previous_actions}"
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_SYSTEM_PROMPT}],
        }
    ]
    for screenshot_index in range(first_visible, len(screenshots)):
        content: list[dict[str, Any]] = []
        if screenshot_index == first_visible:
            content.append({"type": "text", "text": instruction_prompt})
        content.append({"type": "image", "image": screenshots[screenshot_index]})
        messages.append({"role": "user", "content": content})
        if screenshot_index < len(action_outputs):
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": action_outputs[screenshot_index]}],
                }
            )
    return messages


def render_gui_owl_action(
    source_tool_call: Mapping[str, Any],
    *,
    description: str,
) -> str:
    function = source_tool_call.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("source tool call requires a function object")
    name = str(function["name"])
    source_arguments = function.get("arguments", {})
    if not isinstance(source_arguments, Mapping):
        raise ValueError("source tool arguments must be an object")
    action_names = {"tap": "click", "type": "type", "swipe": "swipe"}
    arguments: dict[str, Any] = {"action": action_names.get(name, name)}
    if name in {"tap", "long_press"}:
        arguments["coordinate"] = source_arguments["coordinate"]
    elif name == "swipe":
        arguments["coordinate"] = source_arguments["start_coordinate"]
        arguments["coordinate2"] = source_arguments["coordinate"]
    elif name in {"type", "open", "answer", "key"}:
        arguments["text"] = source_arguments["text"]
    elif name == "system_button":
        arguments["button"] = source_arguments["button"]
    elif name == "wait":
        if "time" in source_arguments:
            arguments["time"] = source_arguments["time"]
    elif name == "stop":
        arguments = {"action": "terminate", "status": "success"}
    else:
        raise ValueError(f"unsupported history action for GUI-Owl: {name}")
    tool_call = {"name": "mobile_use", "arguments": arguments}
    return (
        f"Action: {description.strip()}\n"
        f"<tool_call>\n{json.dumps(tool_call, ensure_ascii=False, separators=(',', ':'))}\n"
        "</tool_call>"
    )


def _native_tool_call(text: str) -> Mapping[str, Any]:
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        raise ValueError("GUI-Owl output must contain exactly one native action and tool call")
    match = re.fullmatch(
        r"\s*Action:\s*[^\n]+\n\s*<tool_call>\s*(?P<call>\{.*\})\s*</tool_call>\s*",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("GUI-Owl output must contain exactly one native action and tool call")
    value = json.loads(match.group("call"))
    if not isinstance(value, Mapping) or value.get("name") != "mobile_use":
        raise ValueError("GUI-Owl tool call must target mobile_use")
    arguments = value.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("GUI-Owl mobile_use arguments must be an object")
    return arguments


def parse_gui_owl_action(text: str, model_inputs: Any) -> ExecutableAction:
    del model_inputs
    arguments = _native_tool_call(text)
    action = str(arguments["action"]).casefold()
    if action == "tap":
        action = "click"
    if action == "click":
        return canonicalize_tool_call(
            {"function": {"name": "tap", "arguments": {"coordinate": arguments["coordinate"]}}}
        )[0]
    if action == "long_press":
        return canonicalize_tool_call(
            {
                "function": {
                    "name": "long_press",
                    "arguments": {"coordinate": arguments["coordinate"]},
                }
            }
        )[0]
    if action == "swipe":
        return canonicalize_tool_call(
            {
                "function": {
                    "name": "swipe",
                    "arguments": {
                        "start_coordinate": arguments["coordinate"],
                        "coordinate": arguments["coordinate2"],
                    },
                }
            }
        )[0]
    if action == "type":
        return ExecutableAction(
            ActionType.TYPE_TEXT,
            target="text_field_unknown",
            text_argument=str(arguments["text"]),
        )
    if action == "system_button":
        return canonicalize_tool_call(
            {
                "function": {
                    "name": "system_button",
                    "arguments": {"button": arguments["button"]},
                }
            }
        )[0]
    if action in {"open", "answer", "key"}:
        return canonicalize_tool_call(
            {"function": {"name": action, "arguments": {"text": arguments["text"]}}}
        )[0]
    if action == "wait":
        return ExecutableAction(ActionType.WAIT)
    if action == "terminate":
        return ExecutableAction(ActionType.STOP)
    raise ValueError(f"unsupported GUI-Owl action: {action}")
