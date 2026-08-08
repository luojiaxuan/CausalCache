"""索引遍特征:全部候选帧低清 + 当前屏过一次冻结前向,取各图 token 段 mean-pool。

# note (luojiaxuan): 抽成共享模块,供训练(train_step_level.py)与评测
# (rl_selector_eval.py)共用。**不许各自复制一份**——分段计数、fail-closed
# 这些细节一旦两边漂移,训练与评测就不是同一个 selector 口径了,
# 而那类不一致正是本项目反复栽跟头的来源。
"""

from __future__ import annotations

from typing import Any


def index_features(rec, steps, cands, ev, cur, *, model, processor, device,
                   torch, build, tool_spec, pooling="mean", return_context=False):
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
    hs = out.hidden_states[-1][0]
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

    feats = torch.stack([pool(a, b) for a, b in segs[:-1]]).float()
    if not return_context:
        return feats
    # note (luojiaxuan): 末段是当前屏。v4 一直把它**丢掉**——而在因果注意力下
    # 候选帧排在当前屏之前、看不到当前屏,于是打分函数的输入里根本没有
    # "当前这一步长什么样",但标签问的恰恰是"这帧对当前这一步有没有用"。
    # 这个特征本来就已经算出来了,取回来零额外成本。
    return feats, pool(*segs[-1]).float()

