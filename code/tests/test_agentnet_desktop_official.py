"""官方多轮桌面 builder 的冻结不变量。

# note (luojiaxuan): 四组锁:(1) 连续 Recent-B 窗口逐消息退化为
# policy/gui_owl_official.build_official_messages 的官方滚动结构(system 之后
# 完全相等);(2) gap-fold 覆盖不变量 —— 每步恰好出现一次(折叠文本或保留轮),
# 图序 = 事件升序 + 当前帧;(3) 官方形态渲染:[0,999] 归一、<tool_call> 载荷与
# 冻结 render_target_text 逐字节一致、target 缺失步可折叠不可保留;(4) fail-closed
# 边界(事件越界/乱序/缺图/缺完整响应)。
"""

from __future__ import annotations

import json

import pytest

from causalcache.agentnet_actions import render_target_text
from causalcache.agentnet_desktop_official import (
    DESKTOP_NO_PREVIOUS_ACTION,
    OfficialStepForms,
    build_desktop_official_messages,
    build_official_forms_for_record,
    official_step_forms,
    render_official_action_line,
    render_official_response,
)
from causalcache.policy.gui_owl_official import build_official_messages


def form(step_id: int, *, retained_ok: bool = True) -> OfficialStepForms:
    line = f"Click at ({step_id}, {step_id})."
    return OfficialStepForms(
        step_id=step_id,
        action_line=line,
        full_response=(
            f"Action: {line}\n<tool_call>\n"
            + json.dumps({"name": "computer_use", "arguments": {
                "action": "left_click", "coordinate": [step_id, step_id]}},
                ensure_ascii=False)
            + "\n</tool_call>"
            if retained_ok
            else None
        ),
        target_unavailable_reason=None if retained_ok else "triple_click",
    )


def image_map(events):
    return {event: f"img-{event}" for event in events}


def test_contiguous_window_degenerates_to_official_rolling_structure() -> None:
    # s=10:步骤 1..9;B=3 窗口 = 事件 {6,7,8}(post 帧为步骤 7,8,9 的观测)。
    steps = [form(i) for i in range(1, 10)]
    window = [6, 7, 8]
    ours = build_desktop_official_messages(
        goal="open the settings panel",
        steps=steps,
        shown_events=window,
        event_images=image_map(window),
        current_image="img-current",
    )
    official = build_official_messages(
        goal="open the settings panel",
        past_action_texts=[step.action_line for step in steps],
        recent_images=[f"img-{event}" for event in window],
        current_image="img-current",
        past_full_responses=[step.full_response for step in steps],
    )
    # system prompt 本身是 mobile/desktop 两份不同常量,结构对齐从第二条起。
    assert ours[0]["role"] == official[0]["role"] == "system"
    assert ours[1:] == official[1:]


def test_b0_matches_official_kept_zero() -> None:
    steps = [form(i) for i in range(1, 5)]
    ours = build_desktop_official_messages(
        goal="g",
        steps=steps,
        shown_events=[],
        event_images={},
        current_image="cur",
    )
    official = build_official_messages(
        goal="g",
        past_action_texts=[step.action_line for step in steps],
        recent_images=[],
        current_image="cur",
        past_full_responses=None,
    )
    assert ours[1:] == official[1:]
    assert len(ours) == 2  # system + 单条 user


def test_empty_history_uses_no_previous_action() -> None:
    ours = build_desktop_official_messages(
        goal="g", steps=[], shown_events=[], event_images={}, current_image="cur"
    )
    assert DESKTOP_NO_PREVIOUS_ACTION in ours[1]["content"][0]["text"]


def test_gap_fold_covers_every_step_exactly_once() -> None:
    # s=12(11 步),shown = {2, 7, 9}:前缀折 1..2,中段折 4..7 与 9,尾折 11。
    steps = [form(i) for i in range(1, 12)]
    shown = [2, 7, 9]
    messages = build_desktop_official_messages(
        goal="g",
        steps=steps,
        shown_events=shown,
        event_images=image_map(shown),
        current_image="cur",
    )
    folded = []
    retained = []
    images = []
    for message in messages[1:]:
        for part in message["content"]:
            if part["type"] == "text" and message["role"] == "user":
                for line in part["text"].splitlines():
                    if line.startswith("Step") and ":" in line:
                        folded.append(int(line.split(":")[0][4:]))
            if message["role"] == "assistant":
                retained.append(part["text"])
            if part["type"] == "image":
                images.append(part["image"])
    exhibited_steps = [event + 1 for event in shown]
    assert sorted(folded + exhibited_steps) == list(range(1, 12))
    assert retained == [steps[event].full_response for event in shown]
    # 图序 = 被选事件升序 post 帧 + 当前帧(HGKV mask 约定:前 K 张历史)。
    assert images == [f"img-{event}" for event in shown] + ["cur"]
    # 交替合法性:user/assistant 严格交替,首尾都是 user。
    roles = [message["role"] for message in messages[1:]]
    assert roles[0] == roles[-1] == "user"
    assert all(a != b for a, b in zip(roles, roles[1:]))


def test_adjacent_events_have_no_gap_text() -> None:
    steps = [form(i) for i in range(1, 8)]
    shown = [4, 5]
    messages = build_desktop_official_messages(
        goal="g",
        steps=steps,
        shown_events=shown,
        event_images=image_map(shown),
        current_image="cur",
    )
    # 事件 4 与 5 相邻:第二条 user 轮只有图,无 continuation 文本。
    second_user = messages[3]
    assert second_user["role"] == "user"
    assert [part["type"] for part in second_user["content"]] == ["image"]
    # 尾段:事件 5 之后是步骤 7 折叠(6+2=7 == total=7)。
    tail_user = messages[-1]
    assert tail_user["content"][0]["type"] == "text"
    assert "Step7:" in tail_user["content"][0]["text"]


def test_min_age_edge_yields_single_step_middle_fold() -> None:
    # age = B+2 恰好合格:s=10,B=3,替换槽位后 shown={5,7,8}(5 = s-1-B-1)。
    # 事件 5 与 7 之间只折步骤 7(5+2..7)。
    steps = [form(i) for i in range(1, 10)]
    shown = [5, 7, 8]
    messages = build_desktop_official_messages(
        goal="g",
        steps=steps,
        shown_events=shown,
        event_images=image_map(shown),
        current_image="cur",
    )
    bridge = messages[3]
    assert bridge["role"] == "user"
    assert bridge["content"][0]["type"] == "text"
    assert bridge["content"][0]["text"].splitlines()[1:] == [
        f"Step7: {steps[6].action_line}"
    ]


def test_fail_closed_boundaries() -> None:
    steps = [form(i) for i in range(1, 6)]
    kwargs = dict(goal="g", steps=steps, current_image="cur")
    with pytest.raises(ValueError, match="strictly increasing"):
        build_desktop_official_messages(
            shown_events=[3, 2], event_images=image_map([2, 3]), **kwargs
        )
    with pytest.raises(ValueError, match=r"\[1, s-2\]"):
        build_desktop_official_messages(
            shown_events=[5], event_images=image_map([5]), **kwargs
        )
    with pytest.raises(ValueError, match="missing image"):
        build_desktop_official_messages(
            shown_events=[2], event_images={}, **kwargs
        )
    broken = [form(1), form(2, retained_ok=False), form(3), form(4), form(5)]
    # 事件 1 的保留轮展示步骤 2(缺官方响应)→ fail-closed。
    with pytest.raises(ValueError, match="no official full response"):
        build_desktop_official_messages(
            goal="g",
            steps=broken,
            shown_events=[1],
            event_images=image_map([1]),
            current_image="cur",
        )
    # 同一步作折叠文本则合法。
    build_desktop_official_messages(
        goal="g",
        steps=broken,
        shown_events=[3],
        event_images=image_map([3]),
        current_image="cur",
    )
    with pytest.raises(ValueError, match="contiguous"):
        build_desktop_official_messages(
            goal="g",
            steps=[form(2)],
            shown_events=[],
            event_images={},
            current_image="cur",
        )


def test_official_forms_from_raw_code_normalize_to_999() -> None:
    entry = {"step_id": 1, "raw_code": "pyautogui.click(x=0.5, y=0.25)"}
    forms = official_step_forms(entry, screen_size=(1920, 1080))
    assert forms.action_line == "Click at (500, 250)."
    payload = forms.full_response.split("<tool_call>\n")[1].split("\n</tool_call>")[0]
    call = json.loads(payload)
    assert call == {
        "name": "computer_use",
        "arguments": {"action": "left_click", "coordinate": [500, 250]},
    }
    # <tool_call> 段与冻结 render_target_text 逐字节一致(Action 行之外无漂移)。
    assert forms.full_response.split("Action: ")[1].split("\n", 1)[1] == (
        render_target_text(call)
    )


def test_official_forms_scroll_and_hotkey_lines() -> None:
    scroll = official_step_forms(
        {"step_id": 1, "raw_code": "pyautogui.scroll(-3)"}, screen_size=(800, 600)
    )
    assert scroll.action_line == "Scroll down."
    assert '"pixels": -3' in scroll.full_response
    hotkey = official_step_forms(
        {"step_id": 2, "raw_code": "pyautogui.hotkey('ctrl', 's')"},
        screen_size=(800, 600),
    )
    assert hotkey.action_line == "Press ctrl+s."


def test_target_unavailable_step_folds_but_cannot_be_retained() -> None:
    entry = {
        "step_id": 1,
        "raw_code": "pyautogui.click(x=0.5, y=0.5, clicks=3)",
    }
    forms = official_step_forms(entry, screen_size=(1000, 1000))
    assert forms.full_response is None
    assert forms.target_unavailable_reason == "triple_click_not_in_computer_use"
    assert forms.action_line.startswith("Triple-click at")  # 仍可折叠进文字历史


def test_build_official_forms_for_record_round_trip() -> None:
    record = {
        "screen_size": [1000, 1000],
        "history": [
            {"step_id": 1, "raw_code": "pyautogui.click(x=0.1, y=0.1)"},
            {"step_id": 2, "raw_code": "pyautogui.write('hello')"},
        ],
    }
    forms = build_official_forms_for_record(record)
    assert [f.step_id for f in forms] == [1, 2]
    assert forms[1].action_line == 'Type "hello".'


def test_action_line_families() -> None:
    def call(action, **kwargs):
        return {"name": "computer_use", "arguments": {"action": action, **kwargs}}

    assert render_official_action_line(
        call("left_click_drag", coordinate=[1, 2], coordinate2=[3, 4])
    ) == "Drag from (1, 2) to (3, 4)."
    assert render_official_action_line(
        call("left_click_drag", coordinate2=[3, 4])
    ) == "Drag to (3, 4)."
    assert render_official_action_line(call("wait")) == (
        "Wait for the screen to update."
    )
    assert render_official_action_line(
        call("terminate", status="success")
    ) == "Task completed."
    assert render_official_response(call("terminate", status="success")).startswith(
        "Action: Task completed.\n<tool_call>"
    )
    with pytest.raises(ValueError, match="unsupported"):
        render_official_action_line(call("fly"))
