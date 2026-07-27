"""Mobile 官方滚动 prompt 的 gap-fold 泛化(非连续历史帧恢复)。

# note (luojiaxuan): desktop 版 agentnet_desktop_official.build_desktop_official_messages
# 的 mobile 对应物,服务 selected 臂(selector 选中的旧帧非连续恢复)。语义差异:
#   * mobile 候选单位 = 步骤 s,展示其 pre-action observation(请求 history 里的
#     observation_screenshot),配对该步模型自己的完整响应(full_response 原文,
#     无需重渲染 —— 与官方保留轮逐字一致);
#   * 连续尾窗时逐消息退化为 policy/gui_owl_official.build_official_messages
#     (单测锁定);折叠文本 = 官方 "StepN: <action_text>" 全局步号;
#   * 图序 = 被选步升序 observation + 当前帧 —— HGKV mask "前 K 张历史" 契约不变。
# 每个已完成步骤恰好出现一次(折叠或保留),违反即 ValueError fail-closed。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.policy.gui_owl_official import (
    NO_PREVIOUS_ACTION,
    OFFICIAL_FIRST_USER_TEMPLATE,
    OFFICIAL_SYSTEM_PROMPT,
)

MOBILE_CONTINUED_TEMPLATE = "Previous actions (continued):\n{history}"


def _fold_lines(action_texts: Sequence[str], lo: int, hi: int) -> list[str]:
    return [f"Step{t}: {action_texts[t - 1]}" for t in range(lo, hi + 1)]


def build_mobile_official_messages_gapfold(
    *,
    goal: str,
    action_texts: Sequence[str],
    full_responses: Sequence[str],
    shown_steps: Sequence[int],
    step_images: Mapping[int, Any],
    current_image: Any,
) -> list[dict[str, Any]]:
    """官方 mobile 滚动结构的 gap-fold 泛化。

    ``action_texts``/``full_responses``:步骤 1..T 的官方形态(模型自写);
    ``shown_steps``:严格递增 ⊆ [1, T],步骤 s 的 observation 进保留轮并与
    该步完整响应配对;当前帧恒在末轮。
    """
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be non-empty text")
    total = len(action_texts)
    if len(full_responses) != total:
        raise ValueError("full_responses must align with action_texts")
    steps = [int(s) for s in shown_steps]
    if steps != sorted(set(steps)):
        raise ValueError("shown_steps must be strictly increasing and unique")
    if steps and (steps[0] < 1 or steps[-1] > total):
        raise ValueError("shown_steps must lie in [1, T]")
    for s in steps:
        if s not in step_images:
            raise ValueError(f"missing observation image for shown step {s}")
        if not isinstance(full_responses[s - 1], str) or not full_responses[s - 1].strip():
            raise ValueError(f"retained turn for step {s} has no full response")

    prefix_hi = (steps[0] - 1) if steps else total
    prefix = _fold_lines(action_texts, 1, prefix_hi)
    history_text = "\n".join(prefix) if prefix else NO_PREVIOUS_ACTION
    first_text = OFFICIAL_FIRST_USER_TEMPLATE.format(goal=goal, history=history_text)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": OFFICIAL_SYSTEM_PROMPT}]}
    ]
    if not steps:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": first_text},
                {"type": "image", "image": current_image},
            ],
        })
        return messages

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": first_text},
            {"type": "image", "image": step_images[steps[0]]},
        ],
    })
    for previous, step in zip(steps, steps[1:]):
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": full_responses[previous - 1]}],
        })
        gap = _fold_lines(action_texts, previous + 1, step - 1)
        content: list[dict[str, Any]] = []
        if gap:
            content.append({
                "type": "text",
                "text": MOBILE_CONTINUED_TEMPLATE.format(history="\n".join(gap)),
            })
        content.append({"type": "image", "image": step_images[step]})
        messages.append({"role": "user", "content": content})
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": full_responses[steps[-1] - 1]}],
    })
    tail_gap = _fold_lines(action_texts, steps[-1] + 1, total)
    tail_content: list[dict[str, Any]] = []
    if tail_gap:
        tail_content.append({
            "type": "text",
            "text": MOBILE_CONTINUED_TEMPLATE.format(history="\n".join(tail_gap)),
        })
    tail_content.append({"type": "image", "image": current_image})
    messages.append({"role": "user", "content": tail_content})
    return messages


__all__ = ["MOBILE_CONTINUED_TEMPLATE", "build_mobile_official_messages_gapfold"]
