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
import os as _os
from collections.abc import Mapping, Sequence
from typing import Any

from scripts.build_desktop_hgkv_corpus import (
    _coordinate as _hist_coordinate,
    history_action_matches_target,
)

# note (luojiaxuan): 单遍救援探针 A —— CAUSALCACHE_WITNESS_PSEUDO_TARGET=last_action
# 时,witness 家族改用"上一步已执行动作"作伪目标(部署时选记忆前已知,无需
# proposal pass)。伪目标转成 target_tool_call 形制后走同一 matcher,坐标经
# _norm999 归到 [0,999] 与 _target_coordinate 口径一致。转换不了的类型给
# "__none__",不会匹配任何候选。
WITNESS_PSEUDO_TARGET = _os.environ.get("CAUSALCACHE_WITNESS_PSEUDO_TARGET", "")


def _pseudo_target_from_history(
    action: Mapping[str, Any], screen_size: tuple[int, int]
) -> Mapping[str, Any]:
    kind = action.get("type")
    coord = _hist_coordinate(action, screen_size=screen_size)
    if kind == "click" and action.get("button", "left") == "left" and coord:
        return {"arguments": {"action": "left_click", "coordinate": list(coord)}}
    if kind == "click" and action.get("button") == "middle" and coord:
        return {"arguments": {"action": "middle_click", "coordinate": list(coord)}}
    if kind == "right_click" and coord:
        return {"arguments": {"action": "right_click", "coordinate": list(coord)}}
    if kind == "double_click" and coord:
        return {"arguments": {"action": "double_click", "coordinate": list(coord)}}
    if kind == "drag" and coord:
        return {"arguments": {"action": "left_click_drag", "coordinate": list(coord)}}
    if kind == "move" and coord:
        return {"arguments": {"action": "mouse_move", "coordinate": list(coord)}}
    if kind == "type_text":
        return {"arguments": {"action": "type", "text": action.get("text")}}
    if kind == "press":
        return {"arguments": {"action": "key", "keys": [str(action.get("key", ""))]}}
    if kind == "hotkey":
        return {"arguments": {"action": "key",
                              "keys": [str(k) for k in action.get("keys", ())]}}
    if kind == "scroll":
        dy = action.get("dy")
        dx = action.get("dx", 0)
        if type(dy) is int and dy != 0 and dx == 0:
            return {"arguments": {"action": "scroll", "pixels": int(dy)}}
        if type(dx) is int and dx != 0:
            return {"arguments": {"action": "hscroll", "pixels": int(dx)}}
    return {"arguments": {"action": "__none__"}}


def _witness_target(
    record: Mapping[str, Any], screen_size: tuple[int, int]
) -> Mapping[str, Any]:
    if WITNESS_PSEUDO_TARGET == "last_action":
        history = record.get("history") or []
        if not history:
            return {"arguments": {"action": "__none__"}}
        last = max(history, key=lambda e: int(e["step_id"]))
        return _pseudo_target_from_history(last.get("action") or {}, screen_size)
    return record["target_tool_call"]

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
    target = _witness_target(record, screen_size)
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


SET_FEATURE_NAMES = (
    "set_size",
    "min_age_gap_to_set",     # |event − 最近的已选事件|;S=∅ 时 -1
    "mean_age_gap_to_set",
    "n_recent_in_set",        # 已选中 age<=5 的数量
    "n_witness_in_set",       # 已选中 witness 的数量
    "witness_redundancy",     # 候选是 witness 且 S 已含 witness
    "event_older_than_set",   # 候选比 S 中最老的还老
    "duplicate_of_set_member",  # 候选与某已选事件 byte 相同(经 duplicates 表)
)


def set_context_features(
    record: Mapping[str, Any],
    *,
    selected: Sequence[int],
    event: int,
    duplicates: Mapping[str, int] | Mapping[int, int],
    coordinate_tolerance: int = 25,
) -> list[float]:
    """候选相对已选集合 S 的上下文特征,顺序 = SET_FEATURE_NAMES;S=∅ 全零槽。

    # note (luojiaxuan): 统一集合条件边际打分器的第二段输入。witness 判定与
    # candidate_features 同一冻结等价函数;byte 重复经单例表的 duplicates 映射
    # (kept_event 的所有别名一视同仁)。
    """
    current_step = int(record["step"])
    screen_size = tuple(record["screen_size"])
    target = _witness_target(record, screen_size)
    if not selected:
        return [0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def is_witness(candidate_event: int) -> bool:
        for entry in record["history"]:
            if int(entry["step_id"]) - 1 != candidate_event:
                continue
            return history_action_matches_target(
                entry["action"], target,
                screen_size=screen_size,
                coordinate_tolerance=coordinate_tolerance,
            )
        return False

    gaps = [abs(event - s) for s in selected]
    n_recent = sum(1 for s in selected if current_step - s <= 5)
    witness_in_set = sum(1 for s in selected if is_witness(s))
    event_witness = is_witness(event)
    alias = {int(k): int(v) for k, v in dict(duplicates).items()}
    canon = lambda e: alias.get(e, e)
    duplicate_member = float(any(canon(event) == canon(s) for s in selected))
    return [
        float(len(selected)),
        float(min(gaps)),
        float(sum(gaps) / len(gaps)),
        float(n_recent),
        float(witness_in_set),
        float(event_witness and witness_in_set > 0),
        float(event < min(selected)),
        duplicate_member,
    ]


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "SET_FEATURE_NAMES",
    "candidate_features",
    "set_context_features",
]
