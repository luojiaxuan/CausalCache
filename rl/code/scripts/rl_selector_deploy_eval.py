#!/usr/bin/env python3
"""selector 2.6 部署口径评测:选出的 2 帧子集查枚举表 vs recent-2,同源配对。

# note (luojiaxuan): 代理指标(holdout 子集对准确率)三种子 +1.0/+1.5/+0.9
# 三连正,但它衡量的是"随机正负对里排对序的概率",不是部署里真发生的事。
# 部署里 selector 只做一件事:在候选池上取 argmax 子集喂给策略。本脚本把
# 这件事原样做一遍:
#   * 留出态(与训练器完全同一划分:sorted → seed 洗牌 → fold 切片);
#   * scorer 帧分 argmax 取 top-2(加性子集分下 = 全 C(n,2) 的 argmax);
#   * 选出的对**查枚举表**得正确性 —— 枚举时每个 B=2 子集都真跑过策略,
#     无需再推理;
#   * 与 recent-2(部署基线,= 候选池最后两帧)**同态配对**比较,
#     McNemar 精确检验 + 态级 bootstrap CI;
#   * MultiApp 真标签分层(AgentNet 源文件 domain,经 manifest join)。
# 完整性自检:labels 里的 recent_correct 字段必须与查表结果逐态一致,
# 不一致即候选池错位,直接报错而不是继续算。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def mcnemar_p(n01: int, n10: int) -> float:
    """两侧精确 McNemar:不一致对里 sel 赢 n01 次 vs 输 n10 次。"""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_ci(diffs: list[int], reps: int = 1999, seed: int = 12345):
    """态级 bootstrap 95% CI(百分位法),diffs ∈ {-1,0,1}。"""
    r = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(r.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def summarize(rows: list[dict]) -> dict:
    """c_sel=None 的态(选择未被枚举覆盖且尚未补测)只进 n_unmeasured。"""
    meas = [r for r in rows if r.get("c_sel") is not None]
    n = len(meas)
    if not n:
        return {"n": 0, "n_unmeasured": len(rows)}
    sel = sum(r["c_sel"] for r in meas)
    rec = sum(r["c_rec"] for r in meas)
    n01 = sum(1 for r in meas if r["c_sel"] and not r["c_rec"])
    n10 = sum(1 for r in meas if r["c_rec"] and not r["c_sel"])
    diffs = [r["c_sel"] - r["c_rec"] for r in meas]
    lo, hi = boot_ci(diffs)
    out = {"n": n, "n_unmeasured": len(rows) - n,
           "sel_acc": round(sel / n, 4), "rec_acc": round(rec / n, 4),
           "diff_pp": round(100 * (sel - rec) / n, 2),
           "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
           "n01_sel_win": n01, "n10_rec_win": n10,
           "mcnemar_p": round(mcnemar_p(n01, n10), 4),
           "moved": sum(1 for r in meas if r["moved"]),
           "oracle_acc": round(sum(r["oracle"] for r in meas) / n, 4),
           "rand_acc": round(sum(r["rand"] for r in meas) / n, 4)}
    if rows and "c_pool" in rows[0]:
        np_ = len(rows)
        out["pool_acc"] = round(sum(r["c_pool"] for r in rows) / np_, 4)
        out["pool_diff_pp"] = round(
            100 * sum(r["c_pool"] - r["c_rec"] for r in rows) / np_, 2)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--domain-json", type=Path, required=True)
    p.add_argument("--token-dir", type=Path, required=True)
    p.add_argument("--scorers", nargs="+", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import torch

    from causalcache_rl.token_cross_scorer import TokenCrossScorer

    # ---- 标签表:dp → {子集: 正确性};划分协议与训练器逐字相同 ----
    table: dict[str, dict] = {}
    recent_ref: dict[str, bool] = {}
    for f in args.labels:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            subs = {frozenset(a["s"]): bool(a["c"]) for a in d.get("all", [])
                    if len(a["s"]) == args.budget}
            vals = list(subs.values())
            if not (vals and any(vals) and not all(vals)):
                continue
            if not (args.token_dir / f"{d['dp_id']}.pt").exists():
                continue
            table[d["dp_id"]] = subs
            if "recent_correct" in d:
                recent_ref[d["dp_id"]] = bool(d["recent_correct"])

    ids = sorted(table)
    random.Random(args.seed).shuffle(ids)
    nh = int(len(ids) * 0.2)
    hold = ids[args.fold * nh:(args.fold + 1) * nh]

    dom = json.loads(args.domain_json.read_text(encoding="utf-8"))
    t2d: dict[str, str] = {}
    for line in args.manifest.open(encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            t2d[d["dp_id"]] = dom.get(d.get("task_id"), "?")
    print(json.dumps({"winnable": len(ids), "fold": args.fold,
                      "holdout": len(hold),
                      "holdout_multiapp": sum(1 for x in hold
                                              if t2d.get(x) == "MultiApp")},
                     ensure_ascii=False), flush=True)

    dev = torch.device(args.device)
    scorers = []
    for sp in args.scorers:
        bundle = torch.load(sp, map_location="cpu", weights_only=False)
        m = TokenCrossScorer(use_recency=bundle["use_recency"]).to(dev)
        m.load_state_dict(bundle["state"])
        m.eval()
        scorers.append((sp, bundle, m, [], [0]))   # rows, miss(可变)

    # 外层遍历态、内层遍历 scorer:token 文件 230MB/态,只读一遍
    with torch.no_grad():
        for dp in hold:
            e = torch.load(args.token_dir / f"{dp}.pt",
                           map_location="cpu", weights_only=False)
            toks = [t.to(dev).float() for t in e["toks"]]
            ctx = e["ctx"].to(dev).float()
            cands = e["cands"]
            subs = table[dp]
            recent = frozenset(cands[-args.budget:])
            if recent not in subs:
                raise SystemExit(f"候选池错位:{dp} recent-2 {sorted(recent)} 不在枚举表")
            if dp in recent_ref and subs[recent] != recent_ref[dp]:
                raise SystemExit(f"候选池错位:{dp} recent-2 查表 "
                                 f"{subs[recent]} != 标签 {recent_ref[dp]}")
            for sp, bundle, m, rows, miss in scorers:
                u = m.frame_scores(toks, ctx)
                order = sorted(range(len(cands)),
                               key=lambda i: float(u[i]), reverse=True)
                chosen = frozenset(cands[i] for i in order[:args.budget])
                # 池内 argmax(仅诊断):限定在被枚举过的帧对里取分和最大
                pool = max(subs, key=lambda s: sum(float(u[cands.index(j)])
                                                   for j in s))
                if chosen not in subs:
                    # note (luojiaxuan): 首版在此丢态 —— 每折丢 16-44%,而被丢的
                    # 恰是 selector 选了探针池外帧的态,系统性有偏。现在把选择
                    # 落盘(c_sel=None),缺的正确性由 rl_score_chosen_subsets.py
                    # 拿冻结策略真跑补齐,归约时合并。
                    miss[0] += 1
                rows.append({"dp": dp,
                             "domain": t2d.get(dp, "?"),
                             "chosen": sorted(chosen),
                             "c_sel": (int(subs[chosen]) if chosen in subs
                                       else None),
                             "c_rec": int(subs[recent]),
                             "c_pool": int(subs[pool]),
                             "pool_moved": int(pool != recent),
                             "moved": int(chosen != recent),
                             "oracle": int(any(subs.values())),
                             "rand": sum(subs.values()) / len(subs)})

    reports = []
    for sp, bundle, m, rows, miss in scorers:
        agg = summarize(rows)
        strata = {d: summarize([r for r in rows
                                if (r["domain"] == "MultiApp") == (d == "MultiApp")])
                  for d in ("MultiApp", "单应用")}
        rep = {"scorer": sp.name, "config": bundle["config"],
               "steps_per_state": bundle["steps_per_state"],
               "torch_seed": bundle.get("torch_seed"),
               "fold": bundle.get("fold"),
               "proxy_holdout_pair": bundle.get("holdout_pair"),
               "missing_in_table": miss[0],
               "overall": agg, "strata": strata,
               "rows": rows}
        reports.append(rep)
        print(json.dumps({k: rep[k] for k in
                          ("scorer", "proxy_holdout_pair", "overall")},
                         ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"fold": args.fold, "holdout": len(hold), "reports": reports},
        ensure_ascii=False))
    print(json.dumps({"finished": True, "scorers": len(reports)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
