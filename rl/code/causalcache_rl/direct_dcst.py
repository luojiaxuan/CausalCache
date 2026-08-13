"""Direct DCST:Direct Counterfactual Subset Transformer(设计冻结版实现)。

# note (luojiaxuan): 规格见 rl/docs/selector_direct_dcst_frozen_20260812.md。
# 六条硬约束在此实现层面落实:
#   1/2 无任何加性帧分、决策前不把帧压成单标量 —— 唯一的标量出口是
#       subset 级 rescue/harm 头;
#   3   SubsetScorer 的 memory 一次装入 S∪R 全部 fine latent;
#   4   recent-B 显式进 scorer,ADD/RETAIN/REMOVE/CURRENT role embedding;
#   5   KEEP 由外部决策规则实现(G(KEEP)=0),模型不强制换子集;
#   6   接口 score_subsets(candidate_subsets, baseline_subset, ...) ——
#       子集是可变长度集合,无 pair-position 参数,B 不进结构。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def infer_grid(t: int, aspect: float = 1920 / 1080) -> tuple[int, int]:
    """token 数 → (rows, cols)。缓存未存网格形状,取最接近 16:9 的因子分解
    (1920×1080 → 38×68 = 2584 精确);无合适因子则退化为单行。"""
    best = (1, t)
    err = float("inf")
    for r in range(1, int(math.sqrt(t)) + 1):
        if t % r == 0:
            c = t // r
            e = abs(math.log((c / r) / aspect))
            if e < err:
                err, best = e, (r, c)
    return best


class _Block(nn.Module):
    """pre-LN cross-attn(可自注意)+ FFN。"""

    def __init__(self, d: int, heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                          batch_first=True)
        self.ln_f = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn, d))

    def forward(self, q, kv=None, kv_mask=None):
        kv = q if kv is None else kv
        a, _ = self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv),
                         key_padding_mask=kv_mask, need_weights=False)
        q = q + a
        return q + self.ffn(self.ln_f(q))


class AttnPool(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        q = self.q.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, self.ln(x), self.ln(x), key_padding_mask=mask,
                           need_weights=False)
        return out[:, 0]


class FrameResampler(nn.Module):
    """[T,4096] → [K,512]。96 spatial-anchor(锚在 12×8 网格)+ 32 global。"""

    def __init__(self, d: int = 512, k_spatial: int = 96, k_global: int = 32,
                 heads: int = 8, ffn: int = 2048, blocks: int = 2,
                 dropout: float = 0.1, d_in: int = 4096) -> None:
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in)
        self.w_in = nn.Linear(d_in, d)
        self.row_emb = nn.Embedding(128, d)
        self.col_emb = nn.Embedding(128, d)
        self.k_spatial, self.k_global = k_spatial, k_global
        self.queries = nn.Parameter(torch.randn(k_spatial + k_global, d) * 0.02)
        # note (luojiaxuan): spatial-anchor 的锚位加到 query 上(近 2:3 网格的
        # 归一坐标),对齐帧内真实二维位置;global 无锚。网格由 k_spatial
        # 因子分解得出(96→8×12,48→6×8)—— v3a 崩于写死 96 的教训。
        gr = max(r for r in range(1, int(math.sqrt(k_spatial)) + 1)
                 if k_spatial % r == 0)
        gc = k_spatial // gr
        anchors = [(r / max(gr - 1, 1), c / max(gc - 1, 1))
                   for r in range(gr) for c in range(gc)]
        self.register_buffer("anchor_rc", torch.tensor(anchors))  # [k_spatial,2]
        self.blocks = nn.ModuleList(
            _Block(d, heads, ffn, dropout) for _ in range(blocks))

    def _query(self, grid: tuple[int, int], device) -> torch.Tensor:
        r, c = grid
        q = self.queries
        ar = (self.anchor_rc[:, 0] * (r - 1)).round().long().clamp_max(127)
        ac = (self.anchor_rc[:, 1] * (c - 1)).round().long().clamp_max(127)
        return torch.cat([q[: self.k_spatial]
                          + self.row_emb(ar) + self.col_emb(ac),
                          q[self.k_spatial:]])

    def forward(self, x: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        return self.forward_batch([x], [grid])[0]

    def forward_batch(self, xs: list[torch.Tensor],
                      grids: list[tuple[int, int]]) -> torch.Tensor:
        """批式:帧间 pad 到同长,一次 MHA。返回 [B,K,512]。"""
        dev = xs[0].device
        tm = max(x.shape[0] for x in xs)
        rows_x, mask = [], torch.ones(len(xs), tm, dtype=torch.bool, device=dev)
        for i, (x, (r, c)) in enumerate(zip(xs, grids)):
            t = x.shape[0]
            x = self.w_in(self.ln_in(x))
            rr = (torch.arange(t, device=dev) // c).clamp_max(127)
            cc = (torch.arange(t, device=dev) % c).clamp_max(127)
            x = x + self.row_emb(rr) + self.col_emb(cc)
            if t < tm:
                x = torch.cat([x, x.new_zeros(tm - t, x.shape[1])])
            rows_x.append(x)
            mask[i, :t] = False
        x = torch.stack(rows_x)
        q = torch.stack([self._query(g, dev) for g in grids])
        for blk in self.blocks:
            q = blk(q, x, kv_mask=mask)
        return q                                          # [B,K,512]


class ContextEncoder(nn.Module):
    """冻结词嵌入(外部查好,[n,4096])→ 每段池化 → 2 blocks 自注意。"""

    def __init__(self, d: int = 512, heads: int = 8, ffn: int = 2048,
                 blocks: int = 2, dropout: float = 0.1, d_in: int = 4096,
                 max_ctx: int = 32) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.seg_pool = AttnPool(d)
        self.seg_type = nn.Embedding(2, d)                # 0=task, 1=action step
        self.pos = nn.Embedding(max_ctx, d)
        self.blocks = nn.ModuleList(
            _Block(d, heads, ffn, dropout) for _ in range(blocks))
        self.max_ctx = max_ctx

    def forward(self, segs: list[torch.Tensor]) -> torch.Tensor:
        """segs[0]=instruction 词嵌入,segs[1:]=各 step 动作行词嵌入。"""
        segs = [segs[0]] + segs[1:][-(self.max_ctx - 1):]
        toks = []
        for i, s in enumerate(segs):
            v = self.seg_pool(self.proj(s).unsqueeze(0))[0]
            v = v + self.seg_type.weight[0 if i == 0 else 1]
            toks.append(v)
        c = torch.stack(toks).unsqueeze(0)
        c = c + self.pos.weight[: c.shape[1]].unsqueeze(0)
        for blk in self.blocks:
            c = blk(c)
        return c[0]                                       # [n_ctx,512]


class CandidateSetEncoder(nn.Module):
    """z_i = AttnPool(L_i)+E_age+E_action(+E_recent) → 3 blocks(条件于 C)。"""

    AGE_BUCKETS = (1, 2, 3, 5, 8, 15, 10 ** 9)

    def __init__(self, d: int = 512, heads: int = 8, ffn: int = 2048,
                 blocks: int = 3, dropout: float = 0.1, d_in: int = 4096) -> None:
        super().__init__()
        self.pool = AttnPool(d)
        self.age_emb = nn.Embedding(len(self.AGE_BUCKETS), d)
        self.recent_emb = nn.Embedding(2, d)
        self.act_proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.blocks = nn.ModuleList(
            _Block(d, heads, ffn, dropout) for _ in range(blocks))

    def bucket(self, age: int) -> int:
        for i, b in enumerate(self.AGE_BUCKETS):
            if age <= b:
                return i
        return len(self.AGE_BUCKETS) - 1

    def forward(self, latents: torch.Tensor, ages: list[int],
                act_embs: list[torch.Tensor], recent_mask: list[bool],
                ctx: torch.Tensor) -> torch.Tensor:
        """latents [N,K,512] → H [N,512]。"""
        n = latents.shape[0]
        z = self.pool(latents)
        dev = z.device
        z = z + self.age_emb(torch.tensor([self.bucket(a) for a in ages],
                                          device=dev))
        z = z + self.recent_emb(torch.tensor([int(b) for b in recent_mask],
                                             device=dev))
        z = z + torch.stack([self.act_proj(e).mean(0) for e in act_embs])
        seq = torch.cat([ctx, z]).unsqueeze(0)
        for blk in self.blocks:
            seq = blk(seq)
        return seq[0, -n:]                                # [N,512]


class SubsetScorer(nn.Module):
    """反事实子集打分:M_S=[fine latents(带 role), current, C, H_S, H_R]
    → Q=8 subset queries(cross→self→cross)→ rescue/harm。"""

    ROLE_ADD, ROLE_RETAIN, ROLE_REMOVE, ROLE_CURRENT = 0, 1, 2, 3

    def __init__(self, d: int = 512, heads: int = 8, ffn: int = 2048,
                 n_queries: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.role_emb = nn.Embedding(4, d)
        self.h_tag = nn.Embedding(2, d)                   # H_S=0 / H_R=1
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.cross1 = _Block(d, heads, ffn, dropout)
        self.self1 = _Block(d, heads, ffn, dropout)
        self.cross2 = _Block(d, heads, ffn, dropout)
        self.pool = AttnPool(d)
        self.head_rescue = nn.Linear(d, 1)
        self.head_harm = nn.Linear(d, 1)
        # note (luojiaxuan): v5 exact policy objective 的标量策略 logit ——
        # π = softmax([z_keep, z_S...]),训练最大化 Σπ·R(全量 reward 表,
        # GUI-Owl 不在计算图);decomp 目标不用它。
        self.head_z = nn.Linear(d, 1)
        # note (luojiaxuan): 负 bias 初始化 —— 初始 rescue≈0 → G≈0 →
        # 初始行为 = always-KEEP(设计要求的保守起点)。
        nn.init.constant_(self.head_rescue.bias, -2.0)
        nn.init.constant_(self.head_harm.bias, -2.0)
        nn.init.zeros_(self.head_z.bias)

    def forward(self, fine: torch.Tensor, roles: torch.Tensor,
                mask: torch.Tensor, hs: torch.Tensor) -> tuple:
        """fine [B,Lm,512](含 current 与 pad),roles [B,Lm](pad 处任意),
        mask [B,Lm](True=pad),hs [B,4,512]=H_S(2)+H_R(2) 带 tag。
        返回 (rescue_logit[B], harm_logit[B])。"""
        m = torch.cat([fine + self.role_emb(roles), hs], dim=1)
        mask = torch.cat([mask, torch.zeros(hs.shape[:2], dtype=torch.bool,
                                            device=mask.device)], dim=1)
        q = self.queries.unsqueeze(0).expand(m.shape[0], -1, -1)
        q = self.cross1(q, m, mask)
        q = self.self1(q)
        q = self.cross2(q, m, mask)
        z = self.pool(q)
        return (self.head_rescue(z).squeeze(-1), self.head_harm(z).squeeze(-1),
                self.head_z(z).squeeze(-1))


class RecentFailureHead(nn.Module):
    """q_s = P(recent-B 错 | R, current, C)。每态一次,候选无关。
    use_draft 时追加 pass-1 分布特征 token(设计的 +pass-1 增强消融;
    E0 证明该特征单独可达 AUC 0.696,而纯视觉 q 头在 fold0 是随机)。"""

    def __init__(self, d: int = 512, heads: int = 8, ffn: int = 2048,
                 dropout: float = 0.1, draft_dim: int = 0) -> None:
        super().__init__()
        self.role_emb = nn.Embedding(2, d)                # 0=recent, 1=current
        self.queries = nn.Parameter(torch.randn(4, d) * 0.02)
        self.cross = _Block(d, heads, ffn, dropout)
        self.selfb = _Block(d, heads, ffn, dropout)
        self.pool = AttnPool(d)
        # note (luojiaxuan): draft 特征走**直连捷径**拼进头部输入 —— v3b 教训:
        # 只作为记忆 token 进注意力会被 260+ 个视觉 token 淹没(q_auc 仍 0.5,
        # 而 E0 同特征+逻辑回归可达 0.696)。
        self.draft_proj = (nn.Sequential(nn.LayerNorm(draft_dim),
                                         nn.Linear(draft_dim, 64), nn.GELU())
                           if draft_dim else None)
        self.head = nn.Linear(d + (64 if draft_dim else 0), 1)

    def forward(self, recent_fine: torch.Tensor, cur_fine: torch.Tensor,
                ctx: torch.Tensor,
                draft: torch.Tensor | None = None) -> torch.Tensor:
        m = torch.cat([recent_fine + self.role_emb.weight[0],
                       cur_fine + self.role_emb.weight[1], ctx]).unsqueeze(0)
        q = self.queries.unsqueeze(0)
        q = self.cross(q, m)
        q = self.selfb(q)
        z = self.pool(q)
        if self.draft_proj is not None and draft is not None:
            z = torch.cat([z, self.draft_proj(draft).unsqueeze(0)], dim=-1)
        return self.head(z).squeeze(-1)[0]                # 标量 logit


class DirectDCST(nn.Module):
    def __init__(self, d: int = 512, k_spatial: int = 96, k_global: int = 32,
                 dropout: float = 0.1, draft_dim: int = 0) -> None:
        super().__init__()
        self.resampler = FrameResampler(d, k_spatial, k_global, dropout=dropout)
        self.context = ContextEncoder(d, dropout=dropout)
        self.set_enc = CandidateSetEncoder(d, dropout=dropout)
        self.scorer = SubsetScorer(d, dropout=dropout)
        self.fail = RecentFailureHead(d, dropout=dropout, draft_dim=draft_dim)

    def encode_state(self, toks: list[torch.Tensor], cur: torch.Tensor,
                     ctx_segs: list[torch.Tensor], ages: list[int],
                     act_embs: list[torch.Tensor], recent_idx: list[int]):
        all_x = list(toks) + [cur]
        all_lat = self.resampler.forward_batch(
            all_x, [infer_grid(x.shape[0]) for x in all_x])
        lat, cur_lat = all_lat[:-1], all_lat[-1]          # [N,K,512], [K,512]
        ctx = self.context(ctx_segs)
        rec_mask = [i in set(recent_idx) for i in range(len(toks))]
        h = self.set_enc(lat, ages, act_embs, rec_mask, ctx)
        return {"lat": lat, "cur": cur_lat, "ctx": ctx, "h": h,
                "recent": list(recent_idx)}

    def recent_failure(self, st, draft=None) -> torch.Tensor:
        rec = st["lat"][st["recent"]].reshape(-1, st["lat"].shape[-1])
        return self.fail(rec, st["cur"], st["ctx"], draft)

    def score_subsets(self, st, subsets: list[tuple[int, ...]]):
        """subsets 为候选帧下标(局部 0..N-1)的可变长度元组;批式打分。"""
        lat, cur, ctx, h = st["lat"], st["cur"], st["ctx"], st["h"]
        rset = set(st["recent"])
        k, d = lat.shape[1], lat.shape[2]
        n_ctx = ctx.shape[0]
        rows_fine, rows_role, rows_hs = [], [], []
        max_frames = max(len(set(s) | rset) for s in subsets)
        lm = max_frames * k + k + n_ctx
        for s in subsets:
            u = sorted(set(s) | rset)
            fine = [lat[i] for i in u]
            role = []
            for i in u:
                if i in s and i in rset:
                    role.append(SubsetScorer.ROLE_RETAIN)
                elif i in s:
                    role.append(SubsetScorer.ROLE_ADD)
                else:
                    role.append(SubsetScorer.ROLE_REMOVE)
            fine = torch.cat([torch.cat(fine), cur, ctx])
            roles = torch.tensor(
                sum(([r] * k for r in role), [])
                + [SubsetScorer.ROLE_CURRENT] * k + [SubsetScorer.ROLE_CURRENT]
                * n_ctx, device=lat.device)
            pad = lm - fine.shape[0]
            if pad:
                fine = torch.cat([fine, fine.new_zeros(pad, d)])
                roles = torch.cat([roles, roles.new_zeros(pad)])
            rows_fine.append(fine)
            rows_role.append(roles)
            hs = torch.stack([h[i] for i in sorted(s)]
                             + [h[i] for i in st["recent"]])
            hs = hs + torch.cat([self.scorer.h_tag.weight[0].expand(len(s), d),
                                 self.scorer.h_tag.weight[1].expand(
                                     len(st["recent"]), d)])
            rows_hs.append(hs)
        fine = torch.stack(rows_fine)
        roles = torch.stack(rows_role)
        mask = torch.zeros(fine.shape[:2], dtype=torch.bool, device=lat.device)
        for i, s in enumerate(subsets):
            used = len(set(s) | rset) * k + k + n_ctx
            mask[i, used:] = True
        hs = torch.stack(rows_hs)
        return self.scorer(fine, roles, mask, hs)
