#!/usr/bin/env python3
"""Official-**style** sparse multiturn prompt for the HGKV adapter.

# note (luojiaxuan): 命名先说清楚,免得下游把它当成"官方协议"引用。这个渲染器叫
# **official-style sparse multiturn**,不是 official multiturn。官方
# ``gui_owl.py`` 的历史永远是一段**连续的最近窗口**:第 i 个保留轮的截图与第 i+1 个
# 保留轮在轨迹上一定相邻,所以它从来不需要说明"这两轮之间发生了什么",也不需要给
# 截图标 step 号——位置本身就是 step 号。跳跃式的历史轮在官方代码里**不存在**。
# 本模块沿用官方的骨架(同一份 system prompt、user/assistant 交替、第一个 user turn
# 用 ``OFFICIAL_FIRST_USER_TEMPLATE``、保留轮携带完整响应),在此之上新增三样官方
# 没有的东西(见下面 "与官方布局的逐点差异")。任何声称"我们用的是官方多轮格式"的
# 说法,对本模块都是错的。
#
# 存在动机:五臂实测里 ``format_effect = R0 - N0 = -0.0217``——我们自造的
# **单轮** sparse 格式(``gui_owl_sparse_history.py``)比官方多轮格式**差** 0.0217
# nats,而冻结选点收益 ``S0 - R0 = +0.0335``。也就是渲染器本身吃掉了约 65% 的可用
# 收益。本模块把 sparse 选点搬回多轮骨架,用来回答"该不该换掉单轮格式"。
#
# 与官方布局的逐点差异(全部是**新增**,官方原有结构一律不动):
#   D1. 保留轮的截图是**非连续**的原始 step,官方只可能是连续最近窗口;
#   D2. 每张历史图前显式标注 ``Historical screenshot from Step{n}:``——官方靠位置
#       隐式编号,跳跃之后位置不再等于 step,不标就没法对齐;
#   D3. 两个相邻保留轮之间被折叠掉的步,以
#       ``Intervening actions from Step{a} through Step{b}:`` + 逐行 ``Step{i}: ...``
#       进入**下一个 user turn**(单步时用 ``Intervening action at Step{a}:``);
#       官方没有这一段,因为它的保留轮之间没有缝隙;
#   D4. 第一个 user turn 追加一句 sparse 声明(``_SPARSE_NOTICE``),告诉模型这些
#       截图是非连续抽样、且带原始 step 标注。
#   注意 D2/D3/D4 **只在 K>=1 时出现**:K=0 时本模块的输出与
#   ``build_official_messages(..., recent_images=[])`` **逐字节相同**(单测钉死),
#   因为没有历史图时既无需标号也无缝隙可填,再加声明只会平白引入格式差异。
#
# 冻结不变量(单测逐条覆盖,与 gui_owl_sparse_history.py 同一套):
#   1. 选中的历史图严格按原始 step 升序出现;
#   2. 1..current_step-1 里**每一步恰好出现一次**——选中步作为 assistant 轮的完整
#      响应出现,未选中步作为文本行出现,不丢不重;
#   3. 每张历史图后面紧跟的 assistant 轮是**它自己那一步**的完整响应,不错位;
#   4. 显式声明截图是 sparse / non-consecutive;
#   5. 当前截图永远在最后,图数 == K+1;
#   6. 输出协议沿用官方 ``Action:`` + ``<tool_call>``(system prompt 直接复用);
#   7. 保留轮缺完整响应即 fail-closed 报错,**绝不退回裸描述**(官方 builder 的
#      P0-1 教训:保留轮只放裸描述实测掉 0.133 nats,而且这种降级完全静默)。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from causalcache.policy.gui_owl_official import (
    NO_PREVIOUS_ACTION,
    OFFICIAL_FIRST_USER_TEMPLATE,
    OFFICIAL_SYSTEM_PROMPT,
)

SPARSE_MULTITURN_PROTOCOL_ID = "causalcache_official_style_sparse_multiturn_v1"
SPARSE_MULTITURN_PROMPT_FORMAT = "official_style_sparse_multiturn"

_SPARSE_NOTICE = (
    "The screenshots below are a sparse, non-consecutive selection from the full "
    "history: consecutive turns are not necessarily adjacent steps. Each historical "
    "screenshot is labeled with its original step index, and the actions taken in "
    "between are given as text."
)


def _steps_block(action_texts: Sequence[str], start: int, end: int) -> str:
    """Render ``Step i: text`` for 1-based steps in [start, end]."""
    return "\n".join(
        f"Step{i}: {action_texts[i - 1]}"
        for i in range(start, end + 1)
        if 1 <= i <= len(action_texts)
    )


def _intervening_block(action_texts: Sequence[str], start: int, end: int) -> str | None:
    """Header + step lines for the folded steps in [start, end], or None if empty.

    # note (luojiaxuan): 头部必须把区间**两端都写出来**。单轮渲染器那版只写
    # ``Intervening actions:``,于是"这几行属于哪两张图之间"要靠模型自己数,而稀疏
    # 选点恰恰把位置信息破坏掉了。区间退化成一步时用单数措辞,避免出现
    # ``from Step4 through Step4`` 这种读起来像是漏填的头部。
    """
    if start > end:
        return None
    body = _steps_block(action_texts, start, end)
    if not body:
        return None
    header = (
        f"Intervening action at Step{start}:"
        if start == end
        else f"Intervening actions from Step{start} through Step{end}:"
    )
    return f"{header}\n{body}"


def build_sparse_multiturn_messages(
    *,
    instruction: str,
    action_texts: Sequence[str],
    full_responses: Sequence[str | None] | None,
    selected_steps: Sequence[int],
    selected_images: Sequence[Any],
    current_step: int,
    current_image: Any,
) -> list[dict[str, Any]]:
    """Assemble the official-style sparse multiturn prompt.

    ``action_texts`` holds every completed step's action description in order
    (1-based: ``action_texts[i-1]`` is Step i). ``full_responses`` carries each
    step's verbatim assistant response (``Action: ...`` +
    ``<tool_call>{...}</tool_call>``) under the same 1-based indexing; entries for
    steps that are **not** selected may be ``None``, but every selected step must
    carry a non-empty response or this raises ``ValueError``. ``selected_steps``
    are the 1-based steps whose screenshots are restored at high fidelity; they
    must be strictly increasing and strictly older than ``current_step``.

    This is **not** the official protocol — see the module docstring for the
    point-by-point differences from ``build_official_messages``.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be non-empty text")
    if not isinstance(current_step, int) or isinstance(current_step, bool):
        raise ValueError("current_step must be an int")
    if current_step < 1:
        raise ValueError("current_step must be 1-based and positive")
    steps = list(selected_steps)
    images = list(selected_images)
    if len(steps) != len(images):
        raise ValueError("selected steps and images must correspond one-to-one")
    if any(not isinstance(step, int) or isinstance(step, bool) for step in steps):
        raise ValueError("selected steps must be ints")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("selected steps must be strictly increasing")
    if steps and (steps[0] < 1 or steps[-1] >= current_step):
        raise ValueError("selected steps must be older than the current step")
    if current_step - 1 > len(action_texts):
        raise ValueError("action_texts must cover every completed step")

    # note (luojiaxuan): full_responses 的契约与 build_official_messages 对齐——
    # 只有"根本没有保留轮"时才允许整个缺席,否则必须与 action_texts 等长,且每一个
    # **被选中**的 step 都要有非空完整响应。折叠掉的步允许是 None:它们只以纯描述
    # 进文本块,本来就不需要 tool_call。
    if full_responses is None:
        if steps:
            raise ValueError(
                "full_responses is required whenever screenshots are retained; "
                "retained assistant turns must carry Action + <tool_call>"
            )
        responses: list[str | None] = [None] * len(action_texts)
    else:
        responses = list(full_responses)
        if len(responses) != len(action_texts):
            raise ValueError("full_responses must align with action_texts")
    for step in steps:
        response = responses[step - 1]
        if not isinstance(response, str) or not response.strip():
            raise ValueError(
                f"retained Step{step} carries no full assistant response; the "
                "sparse multiturn renderer never falls back to a bare description"
            )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": OFFICIAL_SYSTEM_PROMPT}]}
    ]
    # 官方骨架:第一个 user turn 永远是 OFFICIAL_FIRST_USER_TEMPLATE,``Previous
    # actions`` 槽位放"第一张保留图之前"的全部步骤(K=0 时即全部已完成步骤)。
    lead_end = (steps[0] - 1) if steps else (current_step - 1)
    lead = _steps_block(action_texts, 1, lead_end) or NO_PREVIOUS_ACTION
    first_text = OFFICIAL_FIRST_USER_TEMPLATE.format(goal=instruction, history=lead)

    if not steps:
        # K=0:没有历史图,退化成官方布局本身(与 build_official_messages 逐字节相同)。
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

    first_content: list[dict[str, Any]] = [
        {"type": "text", "text": first_text},
        {"type": "text", "text": _SPARSE_NOTICE},
        {"type": "text", "text": f"Historical screenshot from Step{steps[0]}:"},
        {"type": "image", "image": images[0]},
    ]
    messages.append({"role": "user", "content": first_content})

    for position, step in enumerate(steps):
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": responses[step - 1]}],
            }
        )
        following = steps[position + 1] if position + 1 < len(steps) else current_step
        content: list[dict[str, Any]] = []
        gap = _intervening_block(action_texts, step + 1, following - 1)
        if gap:
            content.append({"type": "text", "text": gap})
        if position + 1 < len(steps):
            content.append(
                {"type": "text", "text": f"Historical screenshot from Step{following}:"}
            )
            content.append({"type": "image", "image": images[position + 1]})
        else:
            content.append(
                {"type": "text", "text": f"Current screenshot at Step{current_step}:"}
            )
            content.append({"type": "image", "image": current_image})
        messages.append({"role": "user", "content": content})
    return messages
