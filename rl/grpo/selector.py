# note (luojiaxuan): CausalCache selector —— Plackett-Luce 帧选择头(v2 recipe)。
# 输入每帧特征 + 步位置,输出 B 帧无放回采样及联合 slate logprob。
# 熵按最大可行熵归一化(外审第 6 条:候选数随步数增长,固定系数强度漂移)。
import math

import torch
import torch.nn as nn


class TinyFrameEncoder(nn.Module):
    """smoke 用的最小特征器:64x64 灰度 -> D 维。规模阶段换 executor 视觉塔 pooled 特征,接口不变。"""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 64, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:  # [T,1,64,64]
        return self.net(imgs)


class FrameSelector(nn.Module):
    def __init__(self, feat_dim: int = 256, hidden: int = 512, max_steps: int = 64):
        super().__init__()
        self.pos = nn.Embedding(max_steps, feat_dim)
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        # note (luojiaxuan): recency 偏置冷启动 —— 初始 logits 随"距当前步数"衰减,
        # 起点近似 recent-B2(executor 分布内),防开局 OOD 致 reward 全 0。
        self.recency_scale = nn.Parameter(torch.tensor(2.0))

    def logits(self, feats: torch.Tensor, positions: torch.Tensor, cur_step: int):
        h = feats + self.pos(positions.clamp(max=self.pos.num_embeddings - 1))
        base = self.scorer(h).squeeze(-1)  # [T]
        age = (cur_step - positions).float() / max(cur_step, 1)
        return base - self.recency_scale * age

    def sample(self, feats, positions, cur_step: int, budget: int):
        """返回 (indices, joint_logprob, normalized_entropy)。PL 无放回采样。"""
        lg = self.logits(feats, positions, cur_step)
        T = lg.shape[0]
        b = min(budget, T)
        idx, logp = [], torch.tensor(0.0, device=lg.device)
        mask = torch.zeros_like(lg, dtype=torch.bool)
        ent = torch.tensor(0.0, device=lg.device)
        for _ in range(b):
            masked = lg.masked_fill(mask, float("-inf"))
            probs = torch.softmax(masked, dim=0)
            dist = torch.distributions.Categorical(probs=probs)
            i = dist.sample()
            logp = logp + dist.log_prob(i)
            ent = ent + dist.entropy()
            mask[i] = True
            idx.append(int(i))
        # note (luojiaxuan): 归一化熵 = 实际熵 / log(可行序列数上界)
        max_ent = sum(math.log(max(T - k, 1)) for k in range(b)) or 1.0
        return sorted(idx), logp, ent / max_ent

    def slate_logprob(self, feats, positions, cur_step: int, chosen: list):
        """按采样顺序重算联合 logprob(训练期用;外审第 5 条:信任域要对联合概率)。"""
        lg = self.logits(feats, positions, cur_step)
        mask = torch.zeros_like(lg, dtype=torch.bool)
        logp = torch.tensor(0.0, device=lg.device)
        for i in chosen:
            masked = lg.masked_fill(mask, float("-inf"))
            logsm = torch.log_softmax(masked, dim=0)
            logp = logp + logsm[i]
            mask[i] = True
        return logp
