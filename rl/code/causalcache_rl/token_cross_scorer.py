"""token 级交叉注意力打分器(selector v2 阶梯 2)。

# note (luojiaxuan): 与被证伪的池化打分头的**唯一原则性区别**:帧的分数由
# "当前屏的每个 token 去查询这帧的每个 token"得出,而不是两个池化向量的
# 函数。动机链条:
#   * 池化特征 + 任意容量的头 → holdout 上不去(0.55-0.59,低于 recency 0.595),
#     数据规模曲线也平 —— 排除头容量、训练量、数据量;
#   * 涌现解占 31.6%(两帧单独都错、放一起才对),需要的是"帧与当前屏 /
#     帧与帧之间的空间对应",这正是池化第一个抹掉的东西;
#   * 因此嫌疑集中在特征形态。本模块保留 token 网格,由交叉注意力显式
#     建模"当前屏在这帧里能找到什么"。
#
# 结构刻意保守:单头、低维投影(dk=256)、帧分相加(仍是加性子集分)。
# 一次只换一个变量 —— 若 token 级 + 加性就见效,归因是特征形态;
# 若还要帧间交互才见效,再加(那是下一档,不混在这一档里)。
"""

from __future__ import annotations

import torch
from torch import nn


class TokenCrossScorer(nn.Module):
    """u_j = MLP( mean_q softmax(Q_ctx K_j^T / √dk) V_j ),子集分 = Σ u_j。

    use_recency:把归一化位置(1=最近)拼进 MLP 输入。池化实验里显式喂
    recency 反而更差(56.0 < 56.9),这里保留开关重测 —— 特征形态换了,
    那个结论不能想当然地搬过来。
    """

    def __init__(self, dim: int = 4096, dk: int = 256, hidden: int = 512,
                 use_recency: bool = False) -> None:
        super().__init__()
        self.dk = dk
        self.use_recency = use_recency
        self.q = nn.Linear(dim, dk, bias=False)
        self.k = nn.Linear(dim, dk, bias=False)
        self.v = nn.Linear(dim, dk, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dk + (1 if use_recency else 0), hidden),
            nn.GELU(),
            nn.Linear(hidden, 1))

    def frame_scores(self, toks: list[torch.Tensor], ctx: torch.Tensor,
                     ) -> torch.Tensor:
        """一次算好全部帧分,子集打分只做求和。toks[j]: [t_j, d],ctx: [c, d]。"""
        q = self.q(ctx)                                    # [c, dk]
        outs = []
        n = len(toks)
        for j, f in enumerate(toks):
            k = self.k(f)                                  # [t, dk]
            v = self.v(f)
            att = torch.softmax(q @ k.t() / self.dk ** 0.5, dim=-1)   # [c, t]
            z = (att @ v).mean(dim=0)                      # [dk]
            if self.use_recency:
                z = torch.cat([z, z.new_tensor([(j + 1) / n])])
            outs.append(self.mlp(z).squeeze(-1))
        return torch.stack(outs)

    @staticmethod
    def subset_score(u: torch.Tensor, cands: list[int],
                     subset: tuple[int, ...] | list[int]) -> torch.Tensor:
        return sum((u[cands.index(j)] for j in subset), start=u.new_zeros(()))
