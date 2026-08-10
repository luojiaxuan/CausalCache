"""索引遍特征:全部候选帧低清 + 当前屏过一次冻结前向,取各图 token 段 mean-pool。

# note (luojiaxuan): 抽成共享模块,供训练(train_step_level.py)与评测
# (rl_selector_eval.py)共用。**不许各自复制一份**——分段计数、fail-closed
# 这些细节一旦两边漂移,训练与评测就不是同一个 selector 口径了,
# 而那类不一致正是本项目反复栽跟头的来源。
"""

from __future__ import annotations

from typing import Any


def index_features(rec, steps, cands, ev, cur, *, model, processor, device,
                   torch, build, tool_spec, pooling="mean", return_context=False,
                   layer=-1, layers=None):
    """索引遍:全部候选帧低清 + 当前屏过一次冻结前向,取各图 token 段 mean-pool。

    # note (luojiaxuan): 与 serve 的 hidden selector 同一机制(选择是场景级的,
    # 缩略图够用;动作遍才需要全分辨率)。此处 no_grad —— v1 边界:只训打分头,
    # 索引遍不挂 LoRA。
    """
    msgs = build(goal=rec["instruction"], steps=steps, shown_events=list(cands),
                 event_images={j: ev[j] for j in cands}, current_image=cur)
    enc = processor.apply_chat_template(
        msgs, tools=[tool_spec], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    # note (luojiaxuan): 默认取最后一层,但注意力诊断显示**选帧信号集中在前 1/3 层**
    # (前段 AUC 59.3% p=0.027,中/后段不显著)。层号因此做成参数:
    # 若浅层特征能把留出集排序推过 0.58 的天花板,说明我们一直在错误的深度取特征
    # —— 最后一层已经被压缩成"下一个动作是什么",而选帧要的是更表层的
    # 视觉/位置线索。hidden_states 有 层数+1 项(第 0 项是嵌入)。
    # note (luojiaxuan): 一次前向出多层。hidden_states 本来就全算出来了,
    # 按层各调一次本函数等于把 8B 前向做 N 遍 —— 那正好抵消缓存的意义。
    all_hs = out.hidden_states
    hs = all_hs[layer][0]
    mm = enc.get("mm_token_type_ids")
    if mm is None:
        raise ValueError("no mm_token_type_ids")
    mask = (mm[0] == 1)
    # 连续 image token 段 → 每段一图;段数须等于 候选数+1(末段是当前屏)
    segs, start = [], None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    if len(segs) != len(cands) + 1:
        raise ValueError(f"segment count {len(segs)} != {len(cands) + 1}")

    def pool(a: int, b: int):
        seg = hs[a:b]
        if pooling == "mean":
            return seg.mean(dim=0)
        if pooling == "mean_max":
            # note (luojiaxuan): mean 会把"画面里有一处关键区域"抹平成整图均值;
            # max 保留极值通道。两者拼接,维度翻倍(打分头要同步 dim*2)。
            return torch.cat([seg.mean(dim=0), seg.max(dim=0).values], dim=-1)
        raise ValueError(f"未知 pooling: {pooling}")

    if pooling == "tokens":
        # note (luojiaxuan): 阶梯 2 —— **不池化**,逐帧返回完整 token 矩阵。
        # mean-pool 是涌现解(31.6% 的可解态两帧单独都错)的头号嫌疑:
        # "两帧与当前屏的空间关系"在把 144 token 压成一个向量时就被抹掉了。
        # 返回 list 而非 stack:index processor 虽然钉死了像素预算,但不同
        # 宽高比的图 token 数可能在 144 附近浮动,不强行对齐。
        toks = [hs[a:b] for a, b in segs[:-1]]
        if not return_context:
            return toks
        return toks, hs[segs[-1][0]:segs[-1][1]]

    if layers is not None:
        got = {}
        for L in layers:
            h = all_hs[L][0]

            def pool_L(a: int, b: int, _h=h):
                seg = _h[a:b]
                if pooling == "mean":
                    return seg.mean(dim=0)
                return torch.cat([seg.mean(dim=0), seg.max(dim=0).values], dim=-1)

            f = torch.stack([pool_L(a, b) for a, b in segs[:-1]]).float()
            got[L] = (f, pool_L(*segs[-1]).float()) if return_context else f
        return got

    feats = torch.stack([pool(a, b) for a, b in segs[:-1]]).float()
    if not return_context:
        return feats
    # note (luojiaxuan): 末段是当前屏。v4 一直把它**丢掉**——而在因果注意力下
    # 候选帧排在当前屏之前、看不到当前屏,于是打分函数的输入里根本没有
    # "当前这一步长什么样",但标签问的恰恰是"这帧对当前这一步有没有用"。
    # 这个特征本来就已经算出来了,取回来零额外成本。
    return feats, pool(*segs[-1]).float()

