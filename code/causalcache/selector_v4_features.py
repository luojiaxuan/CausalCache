"""Selector v4 stage-1 的 cheap 特征(无 GPU,全部来自 manifest + 单例表)。

# note (luojiaxuan): stage-1 起步版特征契约。设计原则:
#   * 只用部署时零成本可得的量:候选事件的年龄/新近度、该事件后续动作与当前
#     目标的动作等价信号(recurrence witness 的在线可算版)、轨迹形状统计;
#   * 不做图像前向 —— 1280-d HGKV 读出特征是后续增强(需要一遍 GPU pass),
#     cheap 版先验证"可学出排序"再谈更贵的输入;
#   * 特征名一律进 FEATURE_NAMES,顺序即向量布局,改动必须 bump SCHEMA。
# 标签:ΔU(j) = U({j}) − U(∅)(单例表,官方多轮渲染 + v4 s300 active/bypass 锚)。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scripts.build_desktop_hgkv_corpus import history_action_matches_target

FEATURE_SCHEMA = "causalcache.selector_v4_stage1_features.v1"
FEATURE_NAMES = (
    "age",
    "log_age",
    "recency_rank",          # 0 = 最近的候选(事件号最大),按候选池内排
    "relative_position",     # j / (s-1)
    "in_recent_1",
    "in_recent_2",
    "in_recent_4",
    "in_recent_8",
    "pool_size",
    "trajectory_length",
    "witness_match",         # 事件 j 后续动作与目标全动作等价(在线可算)
    "witness_count",         # 该状态全部等价事件数
    "witness_rank",          # 等价事件中按新近度的名次(非 witness = -1)
    "action_type_click",     # 事件 j+1(展示动作)的类型 one-hot(粗粒度)
    "action_type_type",
    "action_type_key",
    "action_type_scroll",
    "action_type_other",
    "coord_distance",        # witness 坐标距离(无坐标/非 witness = -1;[0,999] 系)
    "dedup_multiplicity",    # 该帧 byte 级重复次数(1 = 唯一)
)


def _action_type_flags(action: Mapping[str, Any] | None) -> tuple[float, ...]:
    kind = (action or {}).get("type", "")
    click = kind in ("click", "double_click", "right_click", "middle_click")
    typing = kind == "type_text"
    key = kind in ("press", "hotkey")
    scroll = kind == "scroll"
    other = not (click or typing or key or scroll)
    return (float(click), float(typing), float(key), float(scroll), float(other))


def candidate_features(
    record: Mapping[str, Any],
    *,
    candidate_pool: Sequence[int],
    duplicates: Mapping[str, int] | Mapping[int, int],
    event: int,
    coordinate_tolerance: int = 25,
) -> list[float]:
    """一个 (决策点, 候选事件) 的特征向量,顺序 = FEATURE_NAMES。"""
    current_step = int(record["step"])
    screen_size = tuple(record["screen_size"])
    target = record["target_tool_call"]
    history_by_step = {int(entry["step_id"]): entry for entry in record["history"]}

    age = current_step - event
    pool_sorted = sorted(candidate_pool, reverse=True)
    recency_rank = float(pool_sorted.index(event))

    witnesses: list[int] = []
    for entry in record["history"]:
        candidate_event = int(entry["step_id"]) - 1
        if candidate_event < 1:
            continue
        if history_action_matches_target(
            entry["action"], target,
            screen_size=screen_size,
            coordinate_tolerance=coordinate_tolerance,
        ):
            witnesses.append(candidate_event)
    witnesses = sorted(set(witnesses), reverse=True)
    is_witness = event in witnesses
    witness_rank = float(witnesses.index(event)) if is_witness else -1.0

    coord_distance = -1.0
    if is_witness:
        entry = history_by_step.get(event + 1)
        action = entry["action"] if entry else {}
        if (
            isinstance(action, Mapping)
            and "x" in action and "y" in action
            and isinstance(target.get("arguments"), Mapping)
            and isinstance(target["arguments"].get("coordinate"), Sequence)
        ):
            width, height = screen_size
            ax = float(action["x"]) / max(width - 1, 1) * 999
            ay = float(action["y"]) / max(height - 1, 1) * 999
            tx, ty = target["arguments"]["coordinate"]
            coord_distance = math.hypot(ax - float(tx), ay - float(ty))

    exhibited = history_by_step.get(event + 1)
    duplicate_count = 1.0
    for source, kept in dict(duplicates).items():
        if int(kept) == event:
            duplicate_count += 1.0

    values = (
        float(age),
        math.log(age),
        recency_rank,
        float(event) / max(current_step - 1, 1),
        float(age <= 2),      # in_recent_1:窗口 = {s-2},即 age<=2 且 event>=1
        float(age <= 3),
        float(age <= 5),
        float(age <= 9),
        float(len(candidate_pool)),
        float(current_step - 1),
        float(is_witness),
        float(len(witnesses)),
        witness_rank,
        *_action_type_flags(exhibited["action"] if exhibited else None),
        coord_distance,
        duplicate_count,
    )
    assert len(values) == len(FEATURE_NAMES)
    return list(values)


__all__ = ["FEATURE_NAMES", "FEATURE_SCHEMA", "candidate_features"]
