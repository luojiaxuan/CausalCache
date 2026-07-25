#!/usr/bin/env python3
"""Official-faithful GUI-Owl interaction protocol (reproduction harness).

# note (luojiaxuan): 逐字复刻 X-PLUG/MobileAgent 的 Mobile-Agent-v3.5 单体
# ``gui_owl.py`` wrapper(commit main),用于复现论文报告的 AndroidWorld 分数,
# 并作为记忆预算消融的公平底座。与我们此前的 v2.1 私有协议的关键差异:
#   1. 原生响应格式 ``Action: <一句话>`` + ``<tool_call>{...}</tool_call>``;
#      v2.1 曾禁止 Action 行,把模型逼离其 RL 训练分布。
#   2. 历史 = 模型自己写的 Action 文本,渲染为 ``Previous actions: StepN: ...``;
#      v2.1 用外部构造的 Event summary。
#   3. 解析宽松(split 提取 tool_call),失败转 UNKNOWN 动作并消耗一步,而非判死整局。
#   4. ``last_image`` 控制保留几张高保真截图(含当前),更早的步骤只留文本。
#      记忆预算映射: last_image = B + 1,B=0 即仅当前截图。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action, _canonical_action

OFFICIAL_PROTOCOL_ID = "mobile_agent_v3_5_gui_owl_official_faithful"

OFFICIAL_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `answer`: Terminate the current task and output the answer.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
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
- If finishing, use action=terminate in the tool call."""

OFFICIAL_FIRST_USER_TEMPLATE = (
    "Please generate the next move according to the UI screenshot, "
    "instruction and previous actions.\n\nInstruction: {goal}\n\n"
    "Previous actions:\n{history}"
)
NO_PREVIOUS_ACTION = "No previous action."


class OfficialParseError(ValueError):
    """Raised when no tool call can be recovered; caller emits UNKNOWN."""


def extract_action_line(response: str) -> str:
    """Official history text: everything after 'Action:' and before the tool call."""
    return response.split("Action:")[-1].split("<tool_call>")[0].strip()


def parse_official_output(text: str) -> tuple[GUIOwlV2Action, dict[str, Any]]:
    """Permissive extraction matching the official wrapper's split/json.loads."""
    if not isinstance(text, str):
        raise OfficialParseError("model output must be text")
    if "<tool_call>" not in text or "</tool_call>" not in text:
        raise OfficialParseError("no tool call present")
    raw = text.split("<tool_call>")[-1].split("</tool_call>")[0].strip()
    try:
        call = json.loads(raw)
    except json.JSONDecodeError as error:
        raise OfficialParseError(f"tool call is not valid JSON: {error}") from error
    if not isinstance(call, Mapping) or "arguments" not in call:
        raise OfficialParseError("tool call missing arguments")
    arguments = dict(call["arguments"])
    if isinstance(arguments.get("action"), str):
        # note (luojiaxuan): 官方同款别名归一(tap → click)。
        arguments["action"] = arguments["action"].replace("tap", "click")
    # note (luojiaxuan): 官方 mobile_use schema 给 wait / long_press 定义了 ``time``
    # 参数,而本仓库冻结的 GUIOwlV2Action 不建模它(2026-07-25 实测:模型合规输出
    # {"action":"wait","time":2} 被判非法,占 UNKNOWN 步的 78%)。冻结契约不动,
    # 在官方 parser 侧剥离该字段并原样保留在 dropped_arguments 供审计。
    dropped = {k: arguments.pop(k) for k in ("time",) if k in arguments}
    # 官方允许 terminate 以 status=failure 收尾;冻结契约只接受 success,这里
    # 归一为 success 以完成解析,真实状态记入 dropped,评分一律来自环境。
    if arguments.get("action") == "terminate" and arguments.get("status") != "success":
        dropped["status"] = arguments.get("status")
        arguments["status"] = "success"
    try:
        canonical, _ = _canonical_action(arguments)
    except Exception as error:  # noqa: BLE001
        raise OfficialParseError(f"uncanonical action: {error}") from error
    return canonical, dropped


def build_official_messages(
    *,
    goal: str,
    past_action_texts: Sequence[str],
    recent_images: Sequence[Any],
    current_image: Any,
    past_full_responses: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the official multi-turn prompt.

    ``past_action_texts`` covers every completed step in order. ``recent_images``
    are the post-action screenshots kept at high fidelity, oldest first; each is
    paired with the trailing action texts so the transcript reads exactly like the
    official rolling conversation. Steps older than that survive only as text.

    # note (luojiaxuan): 保真要点——官方**保留轮**的 assistant 内容是模型的**完整
    # 响应**(``Action: ...`` + ``<tool_call>{...}</tool_call>``),只有被折叠掉的
    # 更早步骤才抽成纯描述。此前本函数在保留轮里只放裸描述,导致上下文里的历史
    # 示范不含 tool_call,与目标输出格式不一致(实测 teacher-forced 目标 logprob
    # 低 0.133 nats)。传入 ``past_full_responses`` 即可还原官方行为;缺省时退回
    # 裸描述以兼容旧调用。
    """
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be non-empty text")
    kept = len(recent_images)
    if kept > len(past_action_texts):
        raise ValueError("more retained images than completed steps")
    folded = past_action_texts[: len(past_action_texts) - kept] if kept else list(
        past_action_texts
    )
    history = (
        "\n".join(f"Step{i + 1}: {t}" for i, t in enumerate(folded))
        if folded
        else NO_PREVIOUS_ACTION
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": OFFICIAL_SYSTEM_PROMPT}],
        }
    ]
    first_text = OFFICIAL_FIRST_USER_TEMPLATE.format(goal=goal, history=history)
    if kept:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": first_text},
                    {"type": "image", "image": recent_images[0]},
                ],
            }
        )
        source = list(past_full_responses) if past_full_responses else list(past_action_texts)
        if len(source) != len(past_action_texts):
            raise ValueError("past_full_responses must align with past_action_texts")
        tail = source[len(source) - kept :]
        for index in range(1, kept):
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": tail[index - 1]}],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image", "image": recent_images[index]}],
                }
            )
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": tail[-1]}]}
        )
        messages.append(
            {"role": "user", "content": [{"type": "image", "image": current_image}]}
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": first_text},
                    {"type": "image", "image": current_image},
                ],
            }
        )
    return messages
