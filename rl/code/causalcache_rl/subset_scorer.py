"""子集打分头:线性 / MLP / MLP+成对交互,可选当前屏上下文条件化。

# note (luojiaxuan): 为什么要换掉 v4 的 `Linear(4096,1)`——欠拟合有直接证据:
# 在**训练过的** 421 个 winnable 态上(按构造上界 100%)只做对 51.1%。
# 一张查表都能拿满分的地方连一半都没记住,说明不是数据量问题,是表达力问题。
# 本模块提供三处容量升级,每处对应一个**具体的、可证伪的**缺陷:
#
# 1) **当前屏上下文**(`use_context`)——这条我认为是最要命的。索引遍把所有
#    候选帧和当前屏拼在一个序列里过一次前向,而候选帧**排在当前屏前面**;
#    在因果注意力下,候选帧的 hidden state **看不到当前屏**。也就是说 v4 的
#    帧特征是"这帧长什么样",而标签问的是"这帧对**当前这一步**有没有用"——
#    打分函数的输入里根本没有"当前这一步"。当前屏的 pooled 特征本来就已经
#    算出来了(`segs[-1]`),v4 直接丢掉了。这里把它作为条件向量接回来:
#    每帧的输入变成 [f_j, f_cur, f_j ⊙ f_cur](逐元素积给出一阶交互)。
#
# 2) **成对交互**(`arch="mlp_pair"`)——v4 的子集分是成员分之和,结构上
#    表达不了"这两帧一起才有用/放一起反而互相干扰"。证据是单帧边际 68.4%
#    与集合边际 36.8% 的落差。低秩双线性项 v(a,b) = (Pa)·(Pb)/√r 补上这一层,
#    参数量只有 dim×rank。
#
# 3) **双池化**(在 index_features 里,`pooling="mean_max"`)——mean 会把
#    "画面里有一处关键区域"抹平成整图均值;max 保留极值通道。
#
# **部署侧的后果(必须一起改,否则训练与部署又脱钩)**:一旦有成对项,
# 子集分不再可分解,`top-B argmax` 不再等价于"最优子集"。选择规则要改成
# **在 C(n,B) 个子集上取 argmax**。这不贵——打分只在缓存好的特征上做
# 几次点积,B=2 且 n≤30 时最多 435 次,相对一次策略前向可以忽略。
# `score_subsets` 就是给部署侧用的那个枚举打分入口。
"""

from __future__ import annotations

import itertools

import torch
from torch import nn


class SubsetScorer(nn.Module):
    """给定候选帧特征(和可选的当前屏特征),给任意子集打分。

    arch:
      * ``linear``    —— 复现 v4:s(子集) = Σ w·f_j。保留是为了做对照,
                        换架构的收益必须相对它来读。
      * ``mlp``       —— 每帧非线性,但子集分仍是加性(隔离"非线性"的贡献)。
      * ``mlp_pair``  —— 加性项 + 低秩成对交互(隔离"交互项"的贡献)。

    三档是**消融梯度**,不是三个候选二选一:linear→mlp 的提升归因于非线性,
    mlp→mlp_pair 的提升归因于交互项。一步跳到最复杂的那档就分不清了。
    """

    def __init__(self, dim: int, *, arch: str = "linear", hidden: int = 512,
                 rank: int = 64, use_context: bool = False) -> None:
        super().__init__()
        if arch not in ("linear", "mlp", "mlp_pair"):
            raise ValueError(f"未知 arch: {arch}")
        self.arch = arch
        self.use_context = use_context
        self.dim = dim
        in_dim = dim * 3 if use_context else dim
        if arch == "linear":
            # note (luojiaxuan): linear 档不接上下文——它就是 v4 的复现基线,
            # 混进上下文就不再是同一个东西,消融也就失去了参照。
            if use_context:
                raise ValueError("linear 档不支持 use_context(它是 v4 复现基线)")
            self.unary = nn.Linear(dim, 1)
        else:
            self.unary = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.pair = nn.Linear(dim, rank, bias=False) if arch == "mlp_pair" else None
        self.rank = rank

    def _unary_input(self, feats: torch.Tensor,
                     context: torch.Tensor | None) -> torch.Tensor:
        if not self.use_context:
            return feats
        if context is None:
            raise ValueError("use_context=True 但没给当前屏特征")
        c = context.unsqueeze(0).expand_as(feats)
        return torch.cat([feats, c, feats * c], dim=-1)

    def frame_terms(self, feats: torch.Tensor,
                    context: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """一次算好所有帧的加性项与成对项,子集打分只做查表求和。

        返回 (u[n], V[n,n] 或 None)。**这是效率的关键**:训练时一个 state 要
        打几十个子集的分,若每个子集都重算一遍前向就白白多花几十倍。
        """
        u = self.unary(self._unary_input(feats, context)).squeeze(-1)
        if self.pair is None:
            return u, None
        p = self.pair(feats)
        return u, (p @ p.transpose(0, 1)) / (self.rank ** 0.5)

    def subset_score(self, u: torch.Tensor, pair: torch.Tensor | None,
                     idx: list[int]) -> torch.Tensor:
        s = u[idx].sum()
        if pair is not None and len(idx) > 1:
            for a, b in itertools.combinations(idx, 2):
                s = s + pair[a, b]
        return s

    @torch.no_grad()
    def score_subsets(self, feats: torch.Tensor, budget: int,
                      context: torch.Tensor | None = None) -> list[tuple[float, tuple[int, ...]]]:
        """部署侧入口:在 C(n,B) 个子集上枚举打分,降序返回。

        有成对项时 top-B argmax 不再等价于最优子集,必须枚举。开销可忽略:
        全部是缓存特征上的点积。
        """
        u, pair = self.frame_terms(feats, context)
        n = feats.shape[0]
        b = min(budget, n)
        out = [(float(self.subset_score(u, pair, list(c))), c)
               for c in itertools.combinations(range(n), b)]
        out.sort(reverse=True)
        return out


def load_v4_linear(bundle: dict, *, arch: str, hidden: int, rank: int,
                   use_context: bool) -> SubsetScorer:
    """从 v4 的 bundle 造新头。

    # note (luojiaxuan): 只有 arch="linear" 能真正继承 v4 权重;其余档次
    # 形状不同,**随机初始化**。这不是缺陷:v4 的头本身只到 51.1%,继承它
    # 反而会把新架构锚在同一个坏解附近。但必须显式说出来,否则读者会误以为
    # 新架构是"在 v4 基础上继续训"的。
    """
    dim = int(bundle["hidden_size"])
    m = SubsetScorer(dim, arch=arch, hidden=hidden, rank=rank,
                     use_context=use_context)
    if arch == "linear":
        m.unary.load_state_dict(bundle["model_state"])
    return m
