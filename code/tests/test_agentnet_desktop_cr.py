"""AgentNet 桌面 C_r 适配层的冻结不变量(与 mobile 版 sparse-history 测试同构)。

# note (luojiaxuan): 核心是 r=0 逐字节一致 —— 适配层装配的 request 必须与在线
# ``GUIOwlOSWorldV2Agent.predict`` 手工复刻的原生构造完全相同(json 序列化后
# 逐字节比较),消息层再比一次(图片换成解码后的像素字节摘要)。其余锁:
# post 帧对齐(事件 j 的恢复图 = step j+1 的决策前截图)、screen_changed 按字节
# 比较、Recent-r 语义 = 冻结 select_osworld_memory、候选恢复帧并入 selected。
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from causalcache.agentnet_desktop_cr import (
    build_agentnet_cr_request,
    build_agentnet_history_events,
    render_agentnet_cr_messages,
)
from causalcache.osworld import (
    OSWorldHistoryEvent,
    OSWorldTask,
    build_osworld_policy_request,
)
from causalcache.osworld_gui_owl import build_gui_owl_osworld_messages

SCREEN = (1440, 900)


def png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def synthetic_trajectory(steps: int = 8):
    """steps 张决策前截图 + steps-1 个历史动作(决策点在第 steps 步)。"""
    screenshots = [png_bytes((i * 20 % 255, 30, 60)) for i in range(steps)]
    history = [
        {
            "step_id": j,
            "action": {"type": "click", "x": 10 * j, "y": 20 * j,
                       "button": "left", "clicks": 1, "duration": 0.1},
            "osworld_action": f"pyautogui.click({10 * j}, {20 * j})",
        }
        for j in range(1, steps)
    ]
    return screenshots, history


def message_digest(messages) -> list:
    """图片不可直接比等,换成像素字节 sha256 后整棵树可比。"""
    rendered = []
    for message in messages:
        parts = []
        for part in message["content"]:
            if part.get("type") == "image":
                parts.append(
                    ("image", hashlib.sha256(part["image"].tobytes()).hexdigest())
                )
            else:
                parts.append(("text", part["text"]))
        rendered.append((message["role"], tuple(parts)))
    return rendered


def native_request(screenshots, history, *, selected, instruction="demo task"):
    events = [
        OSWorldHistoryEvent(
            step_id=record["step_id"],
            action=dict(record["action"]),
            osworld_action=record["osworld_action"],
            result_status="executed",
            screen_changed=screenshots[record["step_id"]]
            != screenshots[record["step_id"] - 1],
            post_screenshot=screenshots[record["step_id"]],
        )
        for record in history
    ]
    task = OSWorldTask(
        domain="tasks",
        task_id=hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16],
        config_path=Path("gated-task-class"),
        config={"instruction": instruction},
    )
    return build_osworld_policy_request(
        task=task,
        step_id=len(events) + 1,
        screenshot=screenshots[len(events)],
        history=events,
        selected_event_step_ids=selected,
        screen_size=SCREEN,
    )


def test_r0_request_matches_native_construction_byte_for_byte() -> None:
    screenshots, history = synthetic_trajectory()
    adapted = build_agentnet_cr_request(
        instruction="demo task",
        screenshots=screenshots,
        history_actions=history,
        current_step=len(screenshots),
        recent_budget=0,
        screen_size=SCREEN,
    )
    native = native_request(screenshots, history, selected=[])
    assert json.dumps(adapted, sort_keys=True) == json.dumps(native, sort_keys=True)


def test_r0_messages_match_native_prompt_byte_for_byte() -> None:
    screenshots, history = synthetic_trajectory()
    adapted = render_agentnet_cr_messages(
        build_agentnet_cr_request(
            instruction="demo task",
            screenshots=screenshots,
            history_actions=history,
            current_step=len(screenshots),
            recent_budget=0,
            screen_size=SCREEN,
        )
    )
    native = build_gui_owl_osworld_messages(
        native_request(screenshots, history, selected=[])
    )
    assert message_digest(adapted) == message_digest(native)
    # r=0:唯一的图是当前截图
    image_parts = [
        part for message in adapted for part in message["content"]
        if part.get("type") == "image"
    ]
    assert len(image_parts) == 1


def test_r2_selects_two_most_recent_events_like_frozen_runtime() -> None:
    screenshots, history = synthetic_trajectory()
    request = build_agentnet_cr_request(
        instruction="demo task",
        screenshots=screenshots,
        history_actions=history,
        current_step=len(screenshots),
        recent_budget=2,
        screen_size=SCREEN,
    )
    current = len(screenshots)
    assert request["selected_event_step_ids"] == [current - 2, current - 1]
    native = native_request(
        screenshots, history, selected=[current - 2, current - 1]
    )
    assert json.dumps(request, sort_keys=True) == json.dumps(native, sort_keys=True)


def test_post_frame_alignment_and_screen_changed() -> None:
    screenshots, history = synthetic_trajectory(steps=4)
    screenshots[2] = screenshots[1]  # step2 的动作没改变屏幕
    events = build_agentnet_history_events(
        history_actions=history[:3], screenshots=screenshots
    )
    assert [event.step_id for event in events] == [1, 2, 3]
    for event in events:
        assert event.post_screenshot == screenshots[event.step_id]
    assert events[0].screen_changed is True
    assert events[1].screen_changed is False  # post == pre
    assert events[2].screen_changed is True


def test_extra_restored_candidate_joins_selected_set() -> None:
    screenshots, history = synthetic_trajectory()
    request = build_agentnet_cr_request(
        instruction="demo task",
        screenshots=screenshots,
        history_actions=history,
        current_step=len(screenshots),
        recent_budget=2,
        extra_restored_step_ids=[1],
        screen_size=SCREEN,
    )
    current = len(screenshots)
    assert request["selected_event_step_ids"] == [1, current - 2, current - 1]
    messages = render_agentnet_cr_messages(request)
    restored_labels = [
        part["text"]
        for message in messages
        for part in message["content"]
        if part.get("type") == "text" and part["text"].startswith("Restored screenshot")
    ]
    assert restored_labels == [
        "Restored screenshot after step 1:",
        f"Restored screenshot after step {current - 2}:",
        f"Restored screenshot after step {current - 1}:",
    ]


def test_rejects_misaligned_inputs() -> None:
    screenshots, history = synthetic_trajectory()
    with pytest.raises(ValueError):
        build_agentnet_cr_request(
            instruction="demo task",
            screenshots=screenshots,
            history_actions=history,
            current_step=len(screenshots) - 1,  # 与 history 长度不匹配
            recent_budget=0,
            screen_size=SCREEN,
        )
    with pytest.raises(ValueError):
        build_agentnet_cr_request(
            instruction="demo task",
            screenshots=screenshots[:-1],  # 缺当前帧
            history_actions=history,
            current_step=len(screenshots),
            recent_budget=0,
            screen_size=SCREEN,
        )
    with pytest.raises(ValueError):
        build_agentnet_cr_request(
            instruction="demo task",
            screenshots=screenshots,
            history_actions=history,
            current_step=len(screenshots),
            recent_budget=0,
            extra_restored_step_ids=[len(screenshots) + 5],
            screen_size=SCREEN,
        )
