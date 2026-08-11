#!/usr/bin/env python3
"""selector v2 阶梯 2 训练器:token 级交叉注意力,子集级成对排序目标。

# note (luojiaxuan): 协议与池化线**逐项对齐**,否则数字不可比:
#   * 同划分:sorted(lab) → random.Random(20260809) 洗牌 → 前 20% 留出;
#   * 同目标:L = −log σ(s(正子集) − s(负子集)),8 对/态;
#   * 同评测:holdout 子集对准确率(配对采样 seed = hash(dp)%9973)、
#     帧级 hi/lo 承载率 AUC(承载率差 ≥0.2 的态);
#   * 同检查点:1 / 4 / 16 步/态,取最好 —— 池化线的 holdout 峰值出现在
#     训练量最小档,这里沿用同一读法。
# 参照线(池化线的终点,超不过就没资格谈下一步):
#   * 子集对:recency 规则 0.5952,池化最好 0.593;
#   * 帧 AUC:recency 66.1%,池化头 56.9%,未训练注意力 59.7%。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True)
    p.add_argument("--token-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260809)
    # note (luojiaxuan): 实测该指标运行间方差约 ±3pp(LoRA/头的训练轨迹对 GPU
    # 非确定性极敏感),<3pp 的差必须有重复种子。--torch-seed 只换初始化与
    # 训练随机性,数据划分(--seed)不动 —— 划分变了 holdout 就不可比了。
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--max-ram-states", type=int, default=0,
                   help=">0 时 RAM 缓存最多保留 N 个态(raw 全清缓存约 230MB/态,"
                        "全量 290GB 进不了内存,FIFO 淘汰)")
    # note (luojiaxuan): 部署口径评测(rl_selector_deploy_eval.py)需要训练完的
    # scorer 权重 —— 首轮三种子只落了 JSON 报告,argmax 选帧无从复现,被迫重训。
    # --fold 与 #26 同构:fold 0 = 原划分,折间留出互斥,便于日后扩 out-of-fold。
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--save-dir", type=Path, default=None,
                   help="给定时每个检查点保存 scorer state(部署评测用)")
    p.add_argument("--configs", default="",
                   help="逗号分隔 name@lr 白名单(如 xattn+recency@0.0001);"
                        "空 = 全跑。lr=3e-4 在三种子×两配置中从未进入最优,可省")
    args = p.parse_args()

    import torch

    from causalcache_rl.token_cross_scorer import TokenCrossScorer

    lab: dict[str, tuple] = {}
    carrier: dict[str, tuple] = {}
    for f in args.labels:
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            subs = [(tuple(a["s"]), bool(a["c"])) for a in d.get("all", [])
                    if len(a["s"]) == args.budget]
            pos = [s for s, c in subs if c]
            neg = [s for s, c in subs if not c]
            if not (pos and neg):
                continue
            if not (args.token_dir / f"{d['dp_id']}.pt").exists():
                continue
            lab[d["dp_id"]] = (pos, neg)
            cr: dict[int, tuple[int, int]] = {}
            for sset, c in subs:
                for j in sset:
                    h, t = cr.get(j, (0, 0))
                    cr[j] = (h + int(c), t + 1)
            rate = {j: h / t for j, (h, t) in cr.items() if t >= 2}
            if len(rate) >= 2:
                hi = max(rate, key=rate.get)
                lo = min(rate, key=rate.get)
                if rate[hi] - rate[lo] >= 0.2:
                    carrier[d["dp_id"]] = (hi, lo)

    ids = sorted(lab)
    random.Random(args.seed).shuffle(ids)
    nh = int(len(ids) * 0.2)
    lo, hi = args.fold * nh, (args.fold + 1) * nh
    hold, train = ids[lo:hi], ids[:lo] + ids[hi:]
    hold_c = [d for d in hold if d in carrier]
    print(json.dumps({"winnable": len(ids), "train": len(train), "fold": args.fold,
                      "holdout": len(hold), "holdout_carrier": len(hold_c)},
                     ensure_ascii=False), flush=True)

    dev = torch.device(args.device)
    ram: dict[str, dict] = {}      # token 文件懒加载进 CPU RAM(约 16GB,主机 2TB)

    def get(dp: str) -> dict:
        e = ram.get(dp)
        if e is None:
            e = torch.load(args.token_dir / f"{dp}.pt",
                           map_location="cpu", weights_only=False)
            if args.max_ram_states and len(ram) >= args.max_ram_states:
                ram.pop(next(iter(ram)))          # FIFO 淘汰,防 290GB 撑爆主机内存
            ram[dp] = e
        return e

    def to_dev(e: dict):
        return ([t.to(dev).float() for t in e["toks"]], e["ctx"].to(dev).float())

    # ---- 参照:recency 规则在同一留出集上(子集对 + 帧 AUC)----
    hit = tot = 0
    for dp in hold:
        pos, neg = lab[dp]
        r = random.Random(hash(dp) % 9973)
        cands = get(dp)["cands"]
        for _ in range(8):
            a, b = r.choice(pos), r.choice(neg)
            if not (set(a) <= set(cands) and set(b) <= set(cands)):
                continue
            tot += 1
            hit += int(sum(a) > sum(b))
    rec_pair = hit / max(tot, 1)
    fh = ft = 0
    for dp in hold_c:
        hi, lo = carrier[dp]
        ft += 1
        fh += int(hi > lo)
    print(json.dumps({"recency_ref_pair": round(rec_pair, 4),
                      "recency_ref_frame_auc": round(fh / max(ft, 1), 4)},
                     ensure_ascii=False), flush=True)

    def evaluate(m) -> tuple[float, float]:
        hit = tot = fh2 = ft2 = 0
        with torch.no_grad():
            for dp in hold:
                e = get(dp)
                toks, ctx = to_dev(e)
                u = m.frame_scores(toks, ctx)
                cands = e["cands"]
                pos, neg = lab[dp]
                r = random.Random(hash(dp) % 9973)
                for _ in range(8):
                    a, b = r.choice(pos), r.choice(neg)
                    if not (set(a) <= set(cands) and set(b) <= set(cands)):
                        continue
                    tot += 1
                    hit += int(float(m.subset_score(u, cands, a)
                                     - m.subset_score(u, cands, b)) > 0)
                if dp in carrier:
                    hi, lo = carrier[dp]
                    if hi in cands and lo in cands:
                        ft2 += 1
                        fh2 += int(float(u[cands.index(hi)]) > float(u[cands.index(lo)]))
        return hit / max(tot, 1), fh2 / max(ft2, 1)

    allowed = {c.strip() for c in args.configs.split(",") if c.strip()}
    results = []
    for name, use_rec, lr in (("xattn", False, 1e-4),
                              ("xattn", False, 3e-4),
                              ("xattn+recency", True, 1e-4)):
        if allowed and f"{name}@{lr:g}" not in allowed:
            continue
        torch.manual_seed(args.torch_seed)
        m = TokenCrossScorer(use_recency=use_rec).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        rng = random.Random(1)
        marks = {}
        step = 0
        for target in (1, 4, 16):
            while step < target * len(train):
                dp = train[step % len(train)]
                e = get(dp)
                toks, ctx = to_dev(e)
                u = m.frame_scores(toks, ctx)
                cands = e["cands"]
                pos, neg = lab[dp]
                loss = torch.zeros((), device=dev)
                k = 0
                for _ in range(8):
                    a, b = rng.choice(pos), rng.choice(neg)
                    if not (set(a) <= set(cands) and set(b) <= set(cands)):
                        continue
                    loss = loss - torch.nn.functional.logsigmoid(
                        m.subset_score(u, cands, a) - m.subset_score(u, cands, b))
                    k += 1
                if k:
                    (loss / k).backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
            pair, fauc = evaluate(m)
            marks[target] = (pair, fauc)
            print(json.dumps({"config": name, "lr": lr, "steps_per_state": target,
                              "holdout_pair": round(pair, 4),
                              "holdout_frame_auc": round(fauc, 4)},
                             ensure_ascii=False), flush=True)
            if args.save_dir is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"config": name, "lr": lr, "steps_per_state": target,
                            "use_recency": use_rec, "torch_seed": args.torch_seed,
                            "fold": args.fold, "holdout_pair": round(pair, 4),
                            "state": {k: v.cpu() for k, v in m.state_dict().items()}},
                           args.save_dir / f"{name}_lr{lr:g}_s{target}.pt")
        best = max(v[0] for v in marks.values())
        results.append({"config": name, "lr": lr,
                        "marks": {str(k): [round(a, 4), round(b, 4)]
                                  for k, (a, b) in marks.items()},
                        "best_holdout_pair": round(best, 4)})

    report = {"torch_seed": args.torch_seed, "fold": args.fold,
              "recency_ref_pair": round(rec_pair, 4),
              "pooled_best_pair": 0.593,
              "results": results,
              "note": "判据:best_holdout_pair 须明显超过 recency 0.595 与"
                      "池化线 0.593,才算特征形态是对的杠杆"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
