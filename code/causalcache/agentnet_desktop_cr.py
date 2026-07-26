"""AgentNet 决策点到冻结桌面 GUI-Owl C_r prompt 的适配层。

# note (luojiaxuan): 桌面 C_r 不另造 prompt —— 消息一律出自冻结的
# ``build_osworld_policy_request`` + ``build_gui_owl_osworld_messages``(与
# ``GUIOwlOSWorldV2Agent.predict`` 在线路径同一份代码)。本模块只负责把离线的
# AgentNet 轨迹切片装配成与在线 runtime 逐字节同构的 request:
#   * 事件 j 的 ``post_screenshot`` = 1-based step j+1 的决策前截图(AgentNet 每步
#     ``image`` 是该步动作前的观测,故 step j 的"动作后"即下一步的"动作前");
#   * ``screen_changed`` 按在线定义算:动作后截图字节 != 动作前截图字节;
#   * ``task_id``/``domain``/``config_path`` 与在线 predict 的构造逐字段一致,
#     保证 r=0(不恢复任何历史图)时与桌面原生 prompt 逐字节相同(单测锁死);
#   * Recent-r 用冻结的 ``select_osworld_memory(arm="recent")``;候选恢复帧通过
#     ``extra_restored_step_ids`` 并入 selected 集合。
# C_r 结构:当前截图 + 完整动作文字历史(_compact_history)+ Recent-r 图,当前
# 截图不计预算 —— 这正是冻结桌面 prompt 本身的形态。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from causalcache.osworld import (
    OSWorldHistoryEvent,
    OSWorldTask,
    build_osworld_policy_request,
    select_osworld_memory,
)
from causalcache.osworld_gui_owl import build_gui_owl_osworld_messages


def build_agentnet_history_events(
    *,
    history_actions: Sequence[dict[str, Any]],
    screenshots: Sequence[bytes],
) -> list[OSWorldHistoryEvent]:
    """把决策点前的动作与截图装配成在线语义的历史事件序列。

    ``history_actions``:1-based step j(j = 1..s-1)的解析产物,元素形如
    ``{"step_id": j, "action": <DesktopAction mapping>, "osworld_action": str}``。
    ``screenshots``:``screenshots[i]`` 是 1-based step i+1 的决策前截图字节,
    至少覆盖到当前步(len >= s)。
    """
    events: list[OSWorldHistoryEvent] = []
    for record in history_actions:
        step_id = record["step_id"]
        if step_id != len(events) + 1:
            raise ValueError("history actions must be contiguous 1-based steps")
        if step_id >= len(screenshots):
            raise ValueError("screenshots must cover every history step's post frame")
        pre = screenshots[step_id - 1]
        post = screenshots[step_id]
        if not isinstance(pre, bytes) or not isinstance(post, bytes) or not post:
            raise ValueError("screenshots must be non-empty raw bytes")
        events.append(
            OSWorldHistoryEvent(
                step_id=step_id,
                action=dict(record["action"]),
                osworld_action=str(record["osworld_action"]),
                result_status="executed",
                screen_changed=post != pre,
                post_screenshot=post,
            )
        )
    return events


def build_agentnet_cr_request(
    *,
    instruction: str,
    screenshots: Sequence[bytes],
    history_actions: Sequence[dict[str, Any]],
    current_step: int,
    recent_budget: int,
    extra_restored_step_ids: Sequence[int] = (),
    screen_size: tuple[int, int],
) -> dict[str, Any]:
    """装配一个与在线 ``GUIOwlOSWorldV2Agent.predict`` 同构的 policy request。"""
    if current_step != len(history_actions) + 1:
        raise ValueError("current_step must be exactly one past the last history step")
    if len(screenshots) < current_step:
        raise ValueError("screenshots must include the current step's frame")
    events = build_agentnet_history_events(
        history_actions=history_actions, screenshots=screenshots
    )
    # note (luojiaxuan): 与在线 predict 逐字段一致的 task 构造;task_id 由指令哈希
    # 派生,保证同一条指令离线/在线产生同一 request 头。
    task = OSWorldTask(
        domain="tasks",
        task_id=hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16],
        config_path=Path("gated-task-class"),
        config={"instruction": instruction},
    )
    selected = set(
        select_osworld_memory(events, arm="recent", budget=recent_budget)
    )
    known = {event.step_id for event in events}
    extras = set(extra_restored_step_ids)
    if not extras.issubset(known):
        raise ValueError("extra restored step ids must reference existing history")
    selected |= extras
    return build_osworld_policy_request(
        task=task,
        step_id=current_step,
        screenshot=screenshots[current_step - 1],
        history=events,
        selected_event_step_ids=sorted(selected),
        screen_size=screen_size,
    )


def render_agentnet_cr_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    """冻结桌面消息构造的薄包装;存在只为让调用点导入面单一。"""
    return build_gui_owl_osworld_messages(request)


__all__ = [
    "build_agentnet_cr_request",
    "build_agentnet_history_events",
    "render_agentnet_cr_messages",
]
