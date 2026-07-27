"""Official-faithful multi-turn desktop protocol for AgentNet decision points.

# note (luojiaxuan): 桌面版官方多轮结构,镜像 policy/gui_owl_official.py 的
# Mobile-Agent-v3.5 协议(2026-07-27 用户裁定:单轮 osworld_gui_owl 路线保留
# 不删,本模块是**新建**的官方结构 builder,v4 语料与重训走这里)。要点:
#   1. system prompt 内嵌 <tools> computer_use 签名(编码时**不得**再传 tools
#      实参,走 trainer 的 chat_template 分支而非 osworld_chat_template 分支);
#   2. 历史 = 官方响应格式 ``Action: <一句话>`` + ``<tool_call>{...}</tool_call>``,
#      gold AgentNet 动作由冻结的 parse_agentnet_step 重解析为 [0,999] 归一
#      computer_use 调用后确定性渲染 —— 杜绝单轮结构里"历史像素坐标 + 输出
#      [0,999]"的量纲混用(实测 31.8% 历史动作 x/y>999);
#   3. 保留轮(占高保真图预算的事件)是完整的 user(图)→assistant(完整响应)
#      交替;更早步骤折叠成 ``StepN: <Action 行>`` 纯文本;
#   4. **gap-fold 泛化**:被选事件可以不连续 —— 每张保留图落在其真实时间位置,
#      与"看到该图后模型做出的动作"配对,相邻被选事件之间的步骤折叠成
#      continuation 文本轮。连续 Recent-B 窗口时逐字节退化为官方滚动结构,
#      B=0 退化为官方 kept=0 单轮;
#   5. 图序 = 被选事件升序 post 帧 + 当前帧,保持 HGKV mask"前 K 张为历史、
#      末张为当前"的冻结约定。
# 帧语义:事件 j 的 post 帧 == 步骤 j+1 的决策前观测(AgentNet 每步 image 是
# 动作前观测),故被选事件 j 的保留轮 = user(post_j) → assistant(step j+1 响应)。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.agentnet_actions import parse_agentnet_step

DESKTOP_OFFICIAL_PROTOCOL_ID = "mobile_agent_v3_5_desktop_official_faithful"

# note (luojiaxuan): 与 OFFICIAL_SYSTEM_PROMPT 同构:<tools> 段描述冻结的
# computer_use 合约(枚举与参数字段对齐 osworld_gui_owl._TOOL_SPEC 的目标子集,
# 即 parse_agentnet_step 会产出的动作),坐标口径 1000x1000([0,999] 归一),
# "# Response format" 段与官方 mobile 版逐字相同(协议层规则与设备无关)。
DESKTOP_OFFICIAL_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You can perform actions like clicking, typing, scrolling, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at the specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor from a start coordinate to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at the specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at the specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at the specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a vertical scroll of the mouse scroll wheel by the specified amount of pixels.
* `hscroll`: Performs a horizontal scroll of the mouse scroll wheel by the specified amount of pixels.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Terminate the current task and output the answer.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "hscroll", "wait", "terminate", "answer"], "type": "string"}, "keys": {"description": "Required only by `action=key`.", "type": "array", "items": {"type": "string"}}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click`, `action=right_click`, `action=middle_click`, `action=double_click`, and `action=left_click_drag`.", "type": "array", "items": {"type": "integer"}}, "coordinate2": {"description": "(x, y): The target coordinate for `action=left_click_drag`.", "type": "array", "items": {"type": "integer"}}, "pixels": {"description": "The amount of scrolling. Positive values scroll up (or right for hscroll), negative values scroll down (or left). Required only by `action=scroll` and `action=hscroll`.", "type": "integer"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
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

DESKTOP_FIRST_USER_TEMPLATE = (
    "Please generate the next move according to the UI screenshot, "
    "instruction and previous actions.\n\nInstruction: {goal}\n\n"
    "Previous actions:\n{history}"
)
DESKTOP_NO_PREVIOUS_ACTION = "No previous action."
# note (luojiaxuan): gap-fold 桥接轮的文本头。官方结构没有中段折叠的先例(它只
# 折叠前缀),这是本仓库对"非连续保留图"的最小泛化:延续第一轮的
# "Previous actions:" 语汇,步号保持全局编号。
DESKTOP_CONTINUED_TEMPLATE = "Previous actions (continued):\n{history}"


def _coordinate_pair(arguments: Mapping[str, Any], key: str) -> tuple[int, int]:
    value = arguments.get(key)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"tool call {key} must be a 2-element coordinate")
    x, y = value
    return int(x), int(y)


def render_official_action_line(tool_call: Mapping[str, Any]) -> str:
    """确定性祈使句:官方响应的 ``Action:`` 行(gold 动作没有模型自述,用模板)。"""
    arguments = tool_call["arguments"]
    action = arguments["action"]
    if action in {"left_click", "right_click", "middle_click", "double_click",
                  "mouse_move"}:
        x, y = _coordinate_pair(arguments, "coordinate")
        verb = {
            "left_click": "Click",
            "right_click": "Right-click",
            "middle_click": "Middle-click",
            "double_click": "Double-click",
            "mouse_move": "Move the cursor to",
        }[action]
        if action == "mouse_move":
            return f"{verb} ({x}, {y})."
        return f"{verb} at ({x}, {y})."
    if action == "left_click_drag":
        x2, y2 = _coordinate_pair(arguments, "coordinate2")
        if "coordinate" in arguments:
            x, y = _coordinate_pair(arguments, "coordinate")
            return f"Drag from ({x}, {y}) to ({x2}, {y2})."
        return f"Drag to ({x2}, {y2})."
    if action == "type":
        return f"Type {json.dumps(arguments.get('text', ''), ensure_ascii=False)}."
    if action == "key":
        keys = arguments.get("keys")
        if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)) or not keys:
            raise ValueError("key action requires a non-empty keys array")
        return f"Press {'+'.join(str(key) for key in keys)}."
    if action == "scroll":
        pixels = int(arguments.get("pixels", 0))
        return "Scroll up." if pixels > 0 else "Scroll down."
    if action == "hscroll":
        pixels = int(arguments.get("pixels", 0))
        return "Scroll right." if pixels > 0 else "Scroll left."
    if action == "wait":
        return "Wait for the screen to update."
    if action == "terminate":
        status = arguments.get("status", "success")
        return "Task completed." if status == "success" else "Task failed."
    if action == "answer":
        return f"Answer {json.dumps(arguments.get('text', ''), ensure_ascii=False)}."
    raise ValueError(f"unsupported computer_use action {action!r}")


def render_official_response(tool_call: Mapping[str, Any]) -> str:
    """完整官方响应:``Action:`` 行 + <tool_call> JSON(与 mobile 版逐段同构)。"""
    line = render_official_action_line(tool_call)
    payload = json.dumps(
        {"name": tool_call["name"], "arguments": dict(tool_call["arguments"])},
        ensure_ascii=False,
    )
    return f"Action: {line}\n<tool_call>\n{payload}\n</tool_call>"


@dataclass(frozen=True)
class OfficialStepForms:
    """一步历史的官方双形态:折叠文本行 + 保留轮完整响应(可缺)。"""

    step_id: int
    action_line: str
    full_response: str | None
    target_unavailable_reason: str | None


def official_step_forms(
    history_entry: Mapping[str, Any], *, screen_size: tuple[int, int]
) -> OfficialStepForms:
    """从 manifest history 条目(带 raw_code)重解析出官方形态。

    # note (luojiaxuan): 复用冻结的 parse_agentnet_step 而不是另写一份动作映射,
    # 保证历史轮的 [0,999] 归一与 teacher target 完全同一条代码路径。manifest 已
    # 保证 history 形态可解析;target 形态缺失(如 tripleClick)时该步仍可折叠
    # 进文本,但**不能**作保留轮 —— 调用方 fail-closed。
    """
    parsed = parse_agentnet_step(str(history_entry["raw_code"]), screen_size=screen_size)
    if parsed.unparseable_reason is not None:
        raise ValueError(
            f"history step {history_entry.get('step_id')} unparseable: "
            f"{parsed.unparseable_reason}"
        )
    if parsed.target_tool_call is not None:
        return OfficialStepForms(
            step_id=int(history_entry["step_id"]),
            action_line=render_official_action_line(parsed.target_tool_call),
            full_response=render_official_response(parsed.target_tool_call),
            target_unavailable_reason=None,
        )
    # 折叠文本兜底:从 history 像素映射归一回 [0,999] 渲染祈使句。
    mapping = dict(parsed.history_action or {})
    width, height = screen_size
    line = _fallback_action_line(mapping, width=width, height=height)
    return OfficialStepForms(
        step_id=int(history_entry["step_id"]),
        action_line=line,
        full_response=None,
        target_unavailable_reason=parsed.target_unavailable_reason,
    )


def _norm999_from_pixel(value: Any, size: int) -> int:
    return round(float(value) / max(size - 1, 1) * 999)


def _fallback_action_line(
    mapping: Mapping[str, Any], *, width: int, height: int
) -> str:
    kind = mapping.get("type")
    if kind in {"click", "double_click", "right_click", "middle_click", "move",
                "drag"}:
        x = _norm999_from_pixel(mapping["x"], width)
        y = _norm999_from_pixel(mapping["y"], height)
        verb = {
            "click": "Click at",
            "double_click": "Double-click at",
            "right_click": "Right-click at",
            "middle_click": "Middle-click at",
            "move": "Move the cursor to",
            "drag": "Drag to",
        }[kind]
        if kind == "click" and int(mapping.get("clicks", 1)) == 3:
            verb = "Triple-click at"
        return f"{verb} ({x}, {y})."
    if kind == "type_text":
        return f"Type {json.dumps(mapping.get('text', ''), ensure_ascii=False)}."
    if kind == "press":
        return f"Press {mapping.get('key')}."
    if kind == "hotkey":
        keys = mapping.get("keys") or []
        return f"Press {'+'.join(str(key) for key in keys)}."
    if kind == "scroll":
        dy = int(mapping.get("dy", 0))
        dx = int(mapping.get("dx", 0))
        if dy:
            return "Scroll up." if dy > 0 else "Scroll down."
        return "Scroll right." if dx > 0 else "Scroll left."
    if kind == "wait":
        return "Wait for the screen to update."
    if kind in {"terminal", "terminate"}:
        return "Task completed."
    raise ValueError(f"unsupported history action type {kind!r}")


def _fold_lines(steps: Sequence[OfficialStepForms], lo: int, hi: int) -> list[str]:
    """全局步号折叠文本行,覆盖步骤 lo..hi(1-based,含端点)。"""
    return [f"Step{t}: {steps[t - 1].action_line}" for t in range(lo, hi + 1)]


def build_desktop_official_messages(
    *,
    goal: str,
    steps: Sequence[OfficialStepForms],
    shown_events: Sequence[int],
    event_images: Mapping[int, Any],
    current_image: Any,
) -> list[dict[str, Any]]:
    """官方滚动结构的 gap-fold 泛化(见模块 docstring 第 4 点)。

    ``steps``:步骤 1..s-1 的官方形态(steps[i] 即步骤 i+1);
    ``shown_events``:严格递增的被选事件号 ⊆ [1, s-2],事件 j 的 post 帧作为
    步骤 j+1 的观测进保留轮;``event_images[j]`` 为该帧载荷;当前帧恒在末轮。
    每步恰好出现一次:或作折叠文本,或作保留轮完整响应 —— 违反即 ValueError。
    """
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be non-empty text")
    total_steps = len(steps)
    for index, form in enumerate(steps, start=1):
        if form.step_id != index:
            raise ValueError("steps must be contiguous 1-based history forms")
    events = [int(event) for event in shown_events]
    if events != sorted(set(events)):
        raise ValueError("shown_events must be strictly increasing and unique")
    if events and (events[0] < 1 or events[-1] > total_steps - 1):
        # 事件 s-1 的 post 帧就是当前观测,不允许作为历史保留图重复出现。
        raise ValueError(
            "shown_events must lie in [1, s-2]; the current frame is always shown"
        )
    for event in events:
        if event not in event_images:
            raise ValueError(f"missing image payload for shown event {event}")
        exhibited = steps[event]  # 步骤 event+1
        if not exhibited.full_response:
            raise ValueError(
                f"retained turn for step {event + 1} has no official full response "
                f"({exhibited.target_unavailable_reason})"
            )

    prefix_hi = events[0] if events else total_steps
    prefix = _fold_lines(steps, 1, prefix_hi)
    history_text = "\n".join(prefix) if prefix else DESKTOP_NO_PREVIOUS_ACTION
    first_text = DESKTOP_FIRST_USER_TEMPLATE.format(goal=goal, history=history_text)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": DESKTOP_OFFICIAL_SYSTEM_PROMPT}],
        }
    ]
    if not events:
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

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": first_text},
                {"type": "image", "image": event_images[events[0]]},
            ],
        }
    )
    for previous, event in zip(events, events[1:]):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": steps[previous].full_response}
                ],
            }
        )
        gap = _fold_lines(steps, previous + 2, event)
        content: list[dict[str, Any]] = []
        if gap:
            content.append(
                {
                    "type": "text",
                    "text": DESKTOP_CONTINUED_TEMPLATE.format(
                        history="\n".join(gap)
                    ),
                }
            )
        content.append({"type": "image", "image": event_images[event]})
        messages.append({"role": "user", "content": content})
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": steps[events[-1]].full_response}],
        }
    )
    tail_gap = _fold_lines(steps, events[-1] + 2, total_steps)
    tail_content: list[dict[str, Any]] = []
    if tail_gap:
        tail_content.append(
            {
                "type": "text",
                "text": DESKTOP_CONTINUED_TEMPLATE.format(
                    history="\n".join(tail_gap)
                ),
            }
        )
    tail_content.append({"type": "image", "image": current_image})
    messages.append({"role": "user", "content": tail_content})
    return messages


def build_official_forms_for_record(
    record: Mapping[str, Any]
) -> list[OfficialStepForms]:
    """screening manifest 决策点记录 → 步骤 1..s-1 的官方形态列表。"""
    screen_size = tuple(record["screen_size"])
    return [
        official_step_forms(entry, screen_size=screen_size)
        for entry in record["history"]
    ]


def render_official_target_text(target_tool_call: Mapping[str, Any]) -> str:
    """teacher-forced 目标:与历史保留轮同一渲染(官方完整响应格式)。"""
    return render_official_response(target_tool_call)


__all__ = [
    "DESKTOP_CONTINUED_TEMPLATE",
    "DESKTOP_FIRST_USER_TEMPLATE",
    "DESKTOP_NO_PREVIOUS_ACTION",
    "DESKTOP_OFFICIAL_PROTOCOL_ID",
    "DESKTOP_OFFICIAL_SYSTEM_PROMPT",
    "OfficialStepForms",
    "build_desktop_official_messages",
    "build_official_forms_for_record",
    "official_step_forms",
    "render_official_action_line",
    "render_official_response",
    "render_official_target_text",
]
