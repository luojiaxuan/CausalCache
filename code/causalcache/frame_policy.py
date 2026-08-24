"""可插拔的帧保留策略 —— 官方 GUI-Owl agent 的唯一改动面。

# note (luojiaxuan): 2026-08-23 路线定案(用户裁定)。此前我们自研 runner
# 复现官方协议,在同一基建同一推理栈上比官方 adapter 低 16pt
# (46.7% vs 62.7%),差距至今未定位。用户指出:直接在官方 adapter 上改即可。
# 关键在于 —— 官方 agent 里恰好有一个函数就是"保留哪几帧":
#
#     def cut_current_messages(self, messages, last_image=2):
#         indices_to_clear = non_empty_user_indices[:-last_image]   # 取最近 B 个
#
# `[:-last_image]` 就是 recent-B。我们的整个课题("哪些历史帧该留")精确地
# 就是替换这一处。于是干预面收缩为**一个函数**,executor、prompt、坐标换算、
# 判分链路全部保持官方原样、逐字节不动。
#
# 论文口径:四臂 = 同一函数的四种实现(recent / random / learned / oracle),
# 天然满足 matched-executor 因果对照 —— 这是外审明确要求的归因结构,而且
# 审稿人可以逐行核对干预面。
#
# 约定:
#   * 策略只决定保留哪些索引,不改消息结构、不改帧内容;
#   * recent 实现必须与官方逐字节等价(有回归测试守住);
#   * 预算 B 由 last_image 给出(官方语义:含当前帧,故历史帧为 B-1)。
"""

from __future__ import annotations

import random
from typing import Any, Callable, Sequence


class FramePolicy:
    """给定候选帧索引与预算,返回保留的索引(升序)。

    # note (luojiaxuan): 输入 candidates 是"当前仍带图的 user 消息索引",
    # 末位即当前帧。所有策略都必须保留末位(当前观测不可丢),这是环境约束,
    # 不是策略自由度。
    """

    name = "base"

    def select(self, candidates: Sequence[int], budget: int, **ctx: Any) -> list[int]:
        raise NotImplementedError


class RecentPolicy(FramePolicy):
    """官方原版行为:保留最近 budget 个。"""

    name = "recent"

    def select(self, candidates: Sequence[int], budget: int, **ctx: Any) -> list[int]:
        return list(candidates[-budget:]) if budget > 0 else []


class RandomPolicy(FramePolicy):
    """负对照:当前帧 + 从更早帧里随机取 budget-1 个。

    # note (luojiaxuan): 用来证明增益来自"选得对"而非"偏离最近窗口"。
    # 合成环境上 random-2 相对 recent-2 是 -0.33pp,即偏离本身无益。
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select(self, candidates: Sequence[int], budget: int, **ctx: Any) -> list[int]:
        if budget <= 0 or not candidates:
            return []
        cur = candidates[-1]
        past = list(candidates[:-1])
        k = min(budget - 1, len(past))
        keep = self._rng.sample(past, k) if k > 0 else []
        return sorted([*keep, cur])


class LearnedPolicy(FramePolicy):
    """我们的 selector:由外部打分函数给出每个候选帧的效用。

    # note (luojiaxuan): 接口先留好,等 selector+policy 联合 GRPO 收敛后再插入
    # (用户 2026-08-23 明确:方法未收敛前不跑 learned 臂)。scorer 接收
    # (past_indices, ctx) 返回等长分数;当前帧强制保留,其余取 top-(budget-1)。
    """

    name = "learned"

    def __init__(self, scorer: Callable[[Sequence[int], dict], Sequence[float]]) -> None:
        self._scorer = scorer

    def select(self, candidates: Sequence[int], budget: int, **ctx: Any) -> list[int]:
        if budget <= 0 or not candidates:
            return []
        cur = candidates[-1]
        past = list(candidates[:-1])
        k = min(budget - 1, len(past))
        if k <= 0:
            return [cur]
        scores = list(self._scorer(past, ctx))
        if len(scores) != len(past):
            raise ValueError("scorer must return one score per past frame")
        ranked = sorted(range(len(past)), key=lambda i: scores[i], reverse=True)
        keep = [past[i] for i in ranked[:k]]
        return sorted([*keep, cur])


class OraclePolicy(FramePolicy):
    """特权上界:由 ctx 给出应当保留的帧索引(来自任务标注)。

    # note (luojiaxuan): 用来测量环境里究竟有多少可利用的记忆余量
    # (oracle 减 recent = 选择侧的可得空间)。**永不参与训练**。
    # 这是三臂里最关键的一臂:若 oracle 相对 recent 无显著增益,说明该环境
    # 结构上测不出记忆差异,再好的 selector 也无从体现 —— 必须在开训前知道。
    """

    name = "oracle"

    def select(self, candidates: Sequence[int], budget: int, **ctx: Any) -> list[int]:
        if budget <= 0 or not candidates:
            return []
        cur = candidates[-1]
        want = [i for i in ctx.get("oracle_indices", ()) if i in candidates and i != cur]
        keep = want[-(budget - 1):] if budget > 1 else []
        # note (luojiaxuan): oracle 帧不足时用最近帧补齐,保证各臂帧数一致 ——
        # 否则 oracle 臂实际看到的帧更少,对比不公平。
        if budget - 1 - len(keep) > 0:
            fill = [i for i in candidates[:-1] if i not in keep]
            keep = fill[-(budget - 1 - len(keep)):] + keep
        return sorted([*keep, cur])


POLICIES: dict[str, Callable[..., FramePolicy]] = {
    "recent": RecentPolicy,
    "random": RandomPolicy,
    "learned": LearnedPolicy,
    "oracle": OraclePolicy,
}
