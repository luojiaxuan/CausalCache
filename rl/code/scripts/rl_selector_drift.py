#!/usr/bin/env python3
"""selector 学到了什么?比较两个头在同一批 rollout hidden 向量上的选择。

# note (luojiaxuan): iter-20 评测零效应,但 k_off_recent 显示学出的 selector
# 与 recent-B 截然不同(89% 两帧全非近期,随机期望 74%)。剩下的分歧是:
# 这个"偏好老帧"是训出来的,还是随机初始化的线性头本来就有的几何偏置?
# 判据 = 拿 iter-0(bootstrap,std 0.02 随机初始化)与 iter-N 的头,在**同一批**
# 审计 hidden 向量上各自取 top-B,量三件事:
#   overlap  两头选出的帧集合平均重合率(1.0 = 完全没学出新行为)
#   rank_corr 两头对候选打分的 Spearman 相关(1.0 = 排序完全一致)
#   age      各自选中帧的归一化年龄均值(看"偏好老帧"从哪来)
# 纯 CPU、零环境重建:hidden 向量已在审计里(base64 fp16)。
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import pathlib


def load_head(path: pathlib.Path, torch):
    b = torch.load(path, map_location="cpu", weights_only=False)
    if b.get("kind") != "hidden_head":
        raise SystemExit(f"{path} 不是 hidden_head bundle")
    head = torch.nn.Linear(int(b["hidden_size"]), 1)
    head.load_state_dict(b["model_state"])
    head.eval()
    return head


def spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--head-a", type=pathlib.Path, required=True, help="基准头(iter-0)")
    p.add_argument("--head-b", type=pathlib.Path, required=True, help="对比头(iter-N)")
    p.add_argument("--groups-glob", required=True, help="取 hidden 向量的 groups.jsonl")
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=3000)
    args = p.parse_args()

    import numpy as np
    import torch

    ha, hb = load_head(args.head_a, torch), load_head(args.head_b, torch)
    with torch.no_grad():
        dw = float((hb.weight - ha.weight).norm() / ha.weight.norm())
    print(f"权重相对位移 ||Δw||/||w0|| = {dw:.4f}")

    ov, rc, age_a, age_b, n = [], [], [], [], 0
    for f in sorted(glob.glob(args.groups_glob)):
        for line in open(f):
            line = line.strip()
            if not line or n >= args.max_steps:
                continue
            ep = json.loads(line)
            for step in ep.get("steps") or []:
                if n >= args.max_steps:
                    break
                hv = step.get("hidden_vectors") or {}
                rounds = step.get("rounds") or []
                si = step.get("step_index")
                if not hv or not rounds or not si or si < 3:
                    continue
                cands = rounds[0].get("candidates") or []
                if len(cands) <= args.budget:
                    continue
                try:
                    H = int(step["hidden_size"])
                    mats = [np.frombuffer(base64.b64decode(hv[str(c)]),
                                          dtype=np.float16).astype("float32")
                            for c in cands]
                except (KeyError, ValueError):
                    continue
                x = torch.tensor(np.stack(mats).reshape(len(cands), H))
                with torch.no_grad():
                    sa = ha(x).squeeze(-1).tolist()
                    sb = hb(x).squeeze(-1).tolist()
                ta = {cands[i] for i in sorted(range(len(cands)),
                                               key=lambda j: -sa[j])[: args.budget]}
                tb = {cands[i] for i in sorted(range(len(cands)),
                                               key=lambda j: -sb[j])[: args.budget]}
                ov.append(len(ta & tb) / args.budget)
                rc.append(spearman(sa, sb))
                span = max(si - 1, 1)
                age_a.append(sum(si - c for c in ta) / len(ta) / span)
                age_b.append(sum(si - c for c in tb) / len(tb) / span)
                n += 1

    if not n:
        raise SystemExit("没有可用的 hidden 向量步")
    rc = [x for x in rc if x == x]
    print(f"样本步数 {n}")
    print(f"  top-{args.budget} 选择重合率 = {sum(ov)/len(ov):.3f}"
          f"(1.0 表示两头选出完全相同的帧)")
    print(f"  打分 Spearman 相关      = {sum(rc)/len(rc):.3f}")
    print(f"  选中帧归一化年龄  A(基准)={sum(age_a)/len(age_a):.3f}  "
          f"B(对比)={sum(age_b)/len(age_b):.3f}   (1=最老,0=最新)")


if __name__ == "__main__":
    main()
