"""Mobile 部署侧 selector 特征(与 desktop 训练端 FEATURE_NAMES 逐位对齐)。

# note (luojiaxuan): propose-then-select 两遍推理的第一遍产出"拟议动作",
# witness 家族特征(4 维)以它为伪目标 —— 训练端用 gold target,部署端用
# proposal,失配量由离线 proposal-ablation 单独测量并披露。其余 24+8 维
# 目标无关。候选单位 = 历史步骤 s(展示其 pre-action observation,该步动作
# 即"从这张观测出发做了什么"),与 desktop"post-frame(j) ↔ 动作 j+1"同语义。
# 布局契约:向量顺序与 selector_v4_features.FEATURE_NAMES + SET_FEATURE_NAMES
# 完全一致(推理端直接吃训练 bundle 的 mean/std);任何改动必须双端同步。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.policy.gui_owl_official import parse_official_output

WITNESS_COORD_TOLERANCE = 25  # [0,999] 系,与 desktop 语料 coordinate_tolerance 一致


def parse_history_actions(full_responses: Sequence[str]) -> list[dict[str, Any] | None]:
    """每步 full_response → canonical mobile 动作 arguments(失败 → None)。"""
    parsed: list[dict[str, Any] | None] = []
    for text in full_responses:
        try:
            canonical, _dropped = parse_official_output(text)
            parsed.append({
                "action": canonical.action,
                "coordinate": canonical.coordinate,
                "coordinate2": canonical.coordinate2,
                "text": canonical.text,
                "button": canonical.button,
            })
        except Exception:  # noqa: BLE001 —— 历史响应异常时该步降级为"other/非 witness"
            parsed.append(None)
    return parsed


def actions_equivalent(
    candidate: Mapping[str, Any] | None,
    proposal: Mapping[str, Any] | None,
    *,
    tolerance: int = WITNESS_COORD_TOLERANCE,
) -> bool:
    """mobile 动作全等(类型 + 坐标容差 + 文本),对齐 desktop 的完整动作等价。"""
    if not candidate or not proposal:
        return False
    if candidate.get("action") != proposal.get("action"):
        return False
    kind = candidate.get("action")
    if kind in ("click", "long_press", "swipe"):
        c1, c2 = candidate.get("coordinate"), proposal.get("coordinate")
        if not (isinstance(c1, Sequence) and isinstance(c2, Sequence) and len(c1) == len(c2) == 2):
            return False
        if math.hypot(float(c1[0]) - float(c2[0]), float(c1[1]) - float(c2[1])) > tolerance:
            return False
        if kind == "swipe":
            d1, d2 = candidate.get("coordinate2"), proposal.get("coordinate2")
            if not (isinstance(d1, Sequence) and isinstance(d2, Sequence)):
                return False
            if math.hypot(float(d1[0]) - float(d2[0]), float(d1[1]) - float(d2[1])) > tolerance:
                return False
        return True
    if kind in ("type", "open", "answer"):
        return str(candidate.get("text", "")).casefold() == str(proposal.get("text", "")).casefold()
    if kind == "key":
        return str(candidate.get("text", "")).casefold() == str(proposal.get("text", "")).casefold()
    if kind == "system_button":
        return candidate.get("button") == proposal.get("button")
    return True  # wait/terminate 等无参数动作:同类型即等价


def _action_type_flags(action: Mapping[str, Any] | None) -> tuple[float, ...]:
    kind = (action or {}).get("action", "")
    click = kind in ("click", "long_press")
    typing = kind == "type"
    key = kind in ("key", "system_button")
    scroll = kind == "swipe"
    other = not (click or typing or key or scroll)
    return (float(click), float(typing), float(key), float(scroll), float(other))


def mobile_candidate_features(
    *,
    step: int,
    pool: Sequence[int],
    total_steps: int,
    parsed_actions: Sequence[dict[str, Any] | None],
    proposal: Mapping[str, Any] | None,
    duplicate_counts: Mapping[int, int],
) -> list[float]:
    """候选步骤 s 的 20 维 cheap 特征,布局 = desktop FEATURE_NAMES。

    desktop 语义映射:decision_step ≡ total_steps+1;事件年龄 = 决策步 − 步号。
    """
    current_step = total_steps + 1
    age = current_step - step
    pool_sorted = sorted(pool, reverse=True)
    recency_rank = float(pool_sorted.index(step))

    witnesses = [
        s for s in pool
        if actions_equivalent(parsed_actions[s - 1], proposal)
    ]
    witnesses = sorted(set(witnesses), reverse=True)
    is_witness = step in witnesses
    witness_rank = float(witnesses.index(step)) if is_witness else -1.0

    coord_distance = -1.0
    if is_witness:
        cand = parsed_actions[step - 1] or {}
        c1, c2 = cand.get("coordinate"), (proposal or {}).get("coordinate")
        if (
            isinstance(c1, Sequence) and isinstance(c2, Sequence)
            and len(c1) == 2 and len(c2) == 2
        ):
            coord_distance = math.hypot(
                float(c1[0]) - float(c2[0]), float(c1[1]) - float(c2[1])
            )

    values = (
        float(age),
        math.log(age),
        recency_rank,
        float(step) / max(current_step - 1, 1),
        float(age <= 2),
        float(age <= 3),
        float(age <= 5),
        float(age <= 9),
        float(len(pool)),
        float(current_step - 1),
        float(is_witness),
        float(len(witnesses)),
        witness_rank,
        *_action_type_flags(parsed_actions[step - 1]),
        coord_distance,
        float(duplicate_counts.get(step, 1)),
    )
    assert len(values) == 20
    return list(values)


def mobile_set_context_features(
    *,
    step: int,
    selected: Sequence[int],
    total_steps: int,
    parsed_actions: Sequence[dict[str, Any] | None],
    proposal: Mapping[str, Any] | None,
    duplicate_alias: Mapping[int, int],
) -> list[float]:
    """8 维 set-context,布局 = desktop SET_FEATURE_NAMES。"""
    if not selected:
        return [0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    current_step = total_steps + 1
    gaps = [abs(step - s) for s in selected]
    n_recent = sum(1 for s in selected if current_step - s <= 5)
    wit = lambda s: actions_equivalent(parsed_actions[s - 1], proposal)
    witness_in_set = sum(1 for s in selected if wit(s))
    canon = lambda s: duplicate_alias.get(s, s)
    return [
        float(len(selected)),
        float(min(gaps)),
        float(sum(gaps) / len(gaps)),
        float(n_recent),
        float(witness_in_set),
        float(wit(step) and witness_in_set > 0),
        float(step < min(selected)),
        float(any(canon(step) == canon(s) for s in selected)),
    ]


def dedup_pool(screenshot_hashes: Mapping[int, str]) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """byte 去重保留最近实例(与 desktop 单例池同规则)。

    返回 (pool 升序, alias 早→晚映射, kept 步的重复计数)。
    """
    seen: dict[str, int] = {}
    pool: list[int] = []
    alias: dict[int, int] = {}
    counts: dict[int, int] = {}
    for step in sorted(screenshot_hashes, reverse=True):
        digest = screenshot_hashes[step]
        if digest in seen:
            kept = seen[digest]
            alias[step] = kept
            counts[kept] = counts.get(kept, 1) + 1
            continue
        seen[digest] = step
        counts[step] = 1
        pool.append(step)
    return sorted(pool), alias, counts


def hash_screenshot(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "WITNESS_COORD_TOLERANCE",
    "actions_equivalent",
    "dedup_pool",
    "hash_screenshot",
    "mobile_candidate_features",
    "mobile_set_context_features",
    "parse_history_actions",
]
