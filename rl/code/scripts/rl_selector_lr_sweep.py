#!/usr/bin/env python3
"""离线反事实:什么样的 lr/熵系数才能在同样的数据量下把 selector 训动?

# note (luojiaxuan): iter-20 复盘发现 selector 20 迭代只位移 4.4%、与随机初始化
# 头选择重合 86.7%,根因是总共只有约 75 次 optimizer.step() + lr 1e-4 + 熵正则
# 拉住初始点。本脚本把 trainer 的**通道 A(selector)原样离线重放**:同一批
# groups.jsonl、同样的组相对优势、同样的 grad_accum 与裁剪,只换 lr 与 λ_H,
# 看终态相对初始头的位移与选择重合率。
#
# 边界(必须写清,别当成性能预测):这是 **off-policy 反事实**——rollout 是
# 真实训练轨迹产生的,换了 lr 后策略会走到别的状态、采到别的数据。因此它只
# 回答"参数会不会动起来",**不回答"闭环成绩会不会变好"**。它的用途是排除
# "优化器根本没在工作"这一层,再决定要不要花两天重跑 campaign。
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import pathlib
from collections import defaultdict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init-head", type=pathlib.Path, required=True)
    p.add_argument("--groups-glob", required=True)
    p.add_argument("--lrs", default="1e-4,1e-3,1e-2,3e-2")
    p.add_argument("--entropies", default="0.01,0.0")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--adv-std-eps", type=float, default=0.1)
    p.add_argument("--w-sel", type=float, default=1.0)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--overlap-samples", type=int, default=400)
    args = p.parse_args()

    import numpy as np
    import torch

    # ---- 预解码所有步的 (特征矩阵, 候选, 选中idx, tau, 组键, reward) ----
    files = sorted(glob.glob(args.groups_glob),
                   key=lambda s: int(s.split("iter-")[1].split("/")[0]))
    iters: list[list[dict]] = []
    for f in files:
        eps = []
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            rounds_data = []
            for step in ep.get("steps") or []:
                hv = step.get("hidden_vectors") or {}
                if not hv:
                    continue
                H = int(step["hidden_size"])
                for rnd in step.get("rounds") or []:
                    cands = rnd.get("candidates") or []
                    if len(cands) < 2 or rnd.get("chosen") is None:
                        continue
                    try:
                        mats = [np.frombuffer(base64.b64decode(hv[str(c)]),
                                              dtype=np.float16).astype("float32")
                                for c in cands]
                    except (KeyError, ValueError):
                        continue
                    rounds_data.append((
                        torch.tensor(np.stack(mats).reshape(len(cands), H)),
                        cands.index(rnd["chosen"])))
            if rounds_data:
                eps.append({"task": ep["task_id"], "r": float(ep["reward"]),
                            "tau": float(ep["temperature"]), "rounds": rounds_data})
        if eps:
            iters.append(eps)
    total_rounds = sum(len(e["rounds"]) for it in iters for e in it)
    print(f"载入 {len(iters)} 个迭代,{sum(len(i) for i in iters)} episode,"
          f"{total_rounds} 轮选帧")

    base = torch.load(args.init_head, map_location="cpu", weights_only=False)
    H = int(base["hidden_size"])

    def fresh_head():
        h = torch.nn.Linear(H, 1)
        h.load_state_dict(base["model_state"])
        return h

    init = fresh_head()
    with torch.no_grad():
        w0n = float(init.weight.norm())

    # 重合率抽样集(固定,跨配置可比)
    sample = []
    for it in iters:
        for e in it:
            for feats, _ in e["rounds"]:
                if len(sample) < args.overlap_samples:
                    sample.append(feats)
    with torch.no_grad():
        init_top = [set(torch.topk(init(f).squeeze(-1),
                                   min(args.budget, f.shape[0])).indices.tolist())
                    for f in sample]

    print(f"{'lr':>8} {'λ_H':>7} {'步数':>6} {'‖Δw‖/‖w0‖':>12} {'选择重合率':>10}")
    for lr in [float(x) for x in args.lrs.split(",")]:
        for lam in [float(x) for x in args.entropies.split(",")]:
            head = fresh_head()
            opt = torch.optim.AdamW(head.parameters(), lr=lr)
            opt.zero_grad(set_to_none=True)
            n_ep = steps = 0
            for it in iters:
                groups: dict[str, list[dict]] = defaultdict(list)
                for e in it:
                    groups[e["task"]].append(e)
                for _, eps in groups.items():
                    rs = [e["r"] for e in eps]
                    if len(eps) < 2 or max(rs) == min(rs):
                        continue          # 退化组,与 trainer 一致地跳过
                    mean_r = sum(rs) / len(rs)
                    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rs) / len(rs))
                    for e in eps:
                        adv = (e["r"] - mean_r) / (std_r + args.adv_std_eps)
                        loss = torch.zeros(())
                        for feats, idx in e["rounds"]:
                            lp = torch.log_softmax(
                                head(feats).squeeze(-1) / e["tau"], dim=0)
                            ent = -(lp.exp() * lp).sum()
                            loss = loss - args.w_sel * adv * lp[idx] - lam * ent
                        (loss / args.grad_accum).backward()
                        n_ep += 1
                        if n_ep % args.grad_accum == 0:
                            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                            opt.step(); opt.zero_grad(set_to_none=True); steps += 1
            with torch.no_grad():
                drift = float((head.weight - init.weight).norm()) / w0n
                top = [set(torch.topk(head(f).squeeze(-1),
                                      min(args.budget, f.shape[0])).indices.tolist())
                       for f in sample]
            ov = sum(len(a & b) / max(len(a), 1)
                     for a, b in zip(init_top, top)) / len(top)
            print(f"{lr:>8.0e} {lam:>7.3f} {steps:>6d} {drift:>12.4f} {ov:>10.3f}")


if __name__ == "__main__":
    main()
