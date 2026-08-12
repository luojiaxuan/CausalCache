#!/usr/bin/env python3
"""E0(DCET 方案的诊断步):pass-1 draft 分布特征能否预测 recent-2 正确性。

# note (luojiaxuan): 外部方案(DCET)规定的第一步 —— 在建任何大模型之前,
# 先回答"KEEP 门的信号在不在 pass-1 分布里"。特征全部来自部署免费量
# (recent-2 前向的 token 级 top-8 logprob + 解析标记),标签 = 枚举表的
# recent_correct。5 折协议与主实验同一划分。模型:逻辑回归 + 2 层 MLP
# (torch,CPU 即可)。读数:OOF AUC 与按最优阈值的 gate 质量。
# AUC ~0.5 → draft 分布无 KEEP 信号,DCET 的门控要靠视觉/上下文;
# AUC 明显 >0.6 → 门有戏,E2 的 recent-correct 辅助头有依据。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def features(row: dict) -> list[float]:
    tk = row.get("topk") or []
    top1 = [t[0][1] for t in tk if t]
    marg = [t[0][1] - t[1][1] for t in tk if len(t) > 1]
    cov = [sum(math.exp(x[1]) for x in t) for t in tk if t]
    ent = []
    for t in tk:
        ps = [math.exp(x[1]) for x in t]
        z = sum(ps) or 1.0
        ent.append(-sum(p / z * math.log(max(p / z, 1e-9)) for p in ps))
    pred = row.get("pred") or {}
    act = pred.get("action") or "none"
    acts = ("left_click", "type", "key", "scroll", "double_click", "right_click")
    onehot = [1.0 if act == a else 0.0 for a in acts]
    def agg(v):
        return [min(v), sum(v) / len(v)] if v else [0.0, 0.0]
    return ([1.0 if pred else 0.0, float(len(top1))]
            + agg(top1) + agg(marg) + agg(cov) + agg(ent) + onehot)


def auc(y, s) -> float:
    pairs = [(a, b) for a, b in zip(s, y)]
    pos = sorted(a for a, b in pairs if b)
    neg = sorted(a for a, b in pairs if not b)
    if not pos or not neg:
        return 0.5
    import bisect
    hits = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg, p)
        hi = bisect.bisect_right(neg, p)
        hits += lo + 0.5 * (hi - lo)
    return hits / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--token-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    import torch

    lab: dict[str, bool] = {}
    for line in args.labels.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        subs = [bool(a["c"]) for a in d.get("all", []) if len(a["s"]) == 2]
        if subs and any(subs) and not all(subs) \
                and (args.token_dir / f"{d['dp_id']}.pt").exists():
            lab[d["dp_id"]] = bool(d["recent_correct"])

    feats: dict[str, list[float]] = {}
    for f in args.draft:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["dp_id"] in lab:
                    feats[r["dp_id"]] = features(r)

    ids = sorted(lab)
    random.Random(args.seed).shuffle(ids)
    nh = int(len(ids) * 0.2)
    print(json.dumps({"states": len(ids), "with_draft": len(feats),
                      "recent_correct_rate": round(
                          sum(lab[i] for i in ids) / len(ids), 4)}))

    dim = len(next(iter(feats.values())))
    oof_s, oof_y = [], []
    for fold in range(5):
        hold = ids[fold * nh:(fold + 1) * nh]
        train = [i for i in ids if i not in set(hold)]
        xt = torch.tensor([feats[i] for i in train])
        yt = torch.tensor([float(lab[i]) for i in train])
        mu, sd = xt.mean(0), xt.std(0).clamp_min(1e-6)
        xt = (xt - mu) / sd
        xh = (torch.tensor([feats[i] for i in hold]) - mu) / sd
        yh = [lab[i] for i in hold]
        best = None
        for name, mk in (("logistic", lambda: torch.nn.Linear(dim, 1)),
                          ("mlp", lambda: torch.nn.Sequential(
                              torch.nn.Linear(dim, 32), torch.nn.GELU(),
                              torch.nn.Dropout(0.2), torch.nn.Linear(32, 1)))):
            torch.manual_seed(0)
            m = mk()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.03)
            # 内层再切 15% 做早停(不碰外层留出)
            k = int(len(train) * 0.85)
            for ep in range(200):
                m.train()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    m(xt[:k]).squeeze(-1), yt[:k])
                opt.zero_grad(); loss.backward(); opt.step()
                if ep % 10 == 0:
                    m.eval()
                    with torch.no_grad():
                        va = auc([bool(v) for v in yt[k:]],
                                 m(xt[k:]).squeeze(-1).tolist())
                    if best is None or va > best[0]:
                        best = (va, name,
                                [float(v) for v in m(xh).squeeze(-1)])
        va, name, sh = best
        print(json.dumps({"fold": fold, "picked": name,
                          "inner_auc": round(va, 4),
                          "fold_auc": round(auc(yh, sh), 4)}))
        oof_s += sh
        oof_y += yh
    print(json.dumps({"OOF_auc": round(auc(oof_y, oof_s), 4),
                      "n": len(oof_y)}))


if __name__ == "__main__":
    main()
