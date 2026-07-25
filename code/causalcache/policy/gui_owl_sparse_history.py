#!/usr/bin/env python3
"""Sparse (non-contiguous) visual-history prompt for the HGKV adapter.

# note (luojiaxuan): 官方 builder 假设保留的是**连续**最近截图;CausalCache 要让
# policy 读懂"从完整历史里挑出的、带 step 标记的非连续截图"。本模块只负责渲染,
# 不改 gui_owl_official.py(官方连续 Recent-K 路径保持冻结,作为 reference 臂)。
#
# 冻结不变量(单测逐条覆盖):
#   1. 选中的历史图严格按原始 step 升序出现;
#   2. 被跳过的 step 仍以文本形式保留,不丢不重;
#   3. 每张历史图紧跟其原始 action 文本,对齐不错位;
#   4. 显式声明这些截图是 sparse / non-consecutive;
#   5. 当前截图永远在最后;
#   6. 输出协议沿用官方 ``Action:`` + ``<tool_call>``(system prompt 直接复用)。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from causalcache.policy.gui_owl_official import (
    NO_PREVIOUS_ACTION,
    OFFICIAL_SYSTEM_PROMPT,
)

SPARSE_PROTOCOL_ID = "causalcache_sparse_history_v1"
_NEXT_NOTE = "This screenshot is not necessarily adjacent to the next screenshot."
_PREV_NOTE = "This screenshot is not necessarily adjacent to the previous screenshot."


def _steps_block(action_texts: Sequence[str], start: int, end: int) -> str:
    """Render ``Step i: text`` for 1-based steps in [start, end]."""
    lines = [
        f"Step{i}: {action_texts[i - 1]}"
        for i in range(start, end + 1)
        if 1 <= i <= len(action_texts)
    ]
    return "\n".join(lines)


def build_sparse_history_messages(
    *,
    instruction: str,
    action_texts: Sequence[str],
    selected_steps: Sequence[int],
    selected_images: Sequence[Any],
    current_step: int,
    current_image: Any,
) -> list[dict[str, Any]]:
    """Assemble the sparse-history prompt.

    ``action_texts`` holds every completed step's action description in order
    (1-based: ``action_texts[i-1]`` is Step i). ``selected_steps`` are the
    1-based steps whose screenshots are restored at high fidelity; they must be
    strictly increasing and strictly older than ``current_step``.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be non-empty text")
    steps = list(selected_steps)
    if len(steps) != len(selected_images):
        raise ValueError("selected steps and images must correspond one-to-one")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("selected steps must be strictly increasing")
    if steps and (steps[0] < 1 or steps[-1] >= current_step):
        raise ValueError("selected steps must be older than the current step")
    if current_step - 1 > len(action_texts):
        raise ValueError("action_texts must cover every completed step")

    content: list[dict[str, Any]] = []
    if not steps:
        history = _steps_block(action_texts, 1, current_step - 1) or NO_PREVIOUS_ACTION
        content.append({
            "type": "text",
            "text": (
                "Please generate the next move according to the UI screenshot, "
                f"instruction and previous actions.\n\nInstruction: {instruction}\n\n"
                f"Previous actions:\n{history}"
            ),
        })
    else:
        lead = _steps_block(action_texts, 1, steps[0] - 1)
        content.append({
            "type": "text",
            "text": (
                "Please generate the next move according to the UI screenshots, "
                "instruction and previous actions. The historical screenshots below "
                "are a sparse, non-consecutive selection from the full history; each "
                "is labeled with its original step index.\n\n"
                f"Instruction: {instruction}\n\n"
                + (
                    f"Previous actions before Step{steps[0]}:\n{lead}"
                    if lead
                    else f"Previous actions before Step{steps[0]}:\n{NO_PREVIOUS_ACTION}"
                )
            ),
        })
        for position, (step, image) in enumerate(zip(steps, selected_images)):
            note = _NEXT_NOTE if position + 1 < len(steps) else _PREV_NOTE
            content.append({
                "type": "text",
                "text": f"Historical screenshot from Step{step}. {note}",
            })
            content.append({"type": "image", "image": image})
            content.append({
                "type": "text",
                "text": f"Action at Step{step}: {action_texts[step - 1]}",
            })
            nxt = steps[position + 1] if position + 1 < len(steps) else current_step
            gap = _steps_block(action_texts, step + 1, nxt - 1)
            if gap:
                content.append({"type": "text", "text": f"Intervening actions:\n{gap}"})
    content.append({"type": "text", "text": f"Current screenshot at Step{current_step}."})
    content.append({"type": "image", "image": current_image})
    return [
        {"role": "system", "content": [{"type": "text", "text": OFFICIAL_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def sparse_image_count(messages: Sequence[dict[str, Any]]) -> int:
    """Number of images in the prompt (K historical + 1 current)."""
    return sum(
        1
        for m in messages
        for part in m.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image"
    )
