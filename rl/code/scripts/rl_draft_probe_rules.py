#!/usr/bin/env python3
"""pass-1 信号选帧的离线侦察:用 bcurve 落盘的原始预测做零 GPU 规则测试。

# note (luojiaxuan): 用户问"先跑一遍 forward 拿 draft action 当 selector 输入,
# 是不是就可学了"。在训练任何东西之前,先用已有数据把这条路的信号量测出来:
#   * bcurve B=1 = 每个单帧的 leave-one-in 探针(该帧+当前屏跑策略的动作与
#     正确性)—— 这是"pass-1 信号"的计算上界形态(每候选一次前向);
#   * bcurve B=2 = 帧对查表(与部署评测同一口径);
#   * b0_pred = draft 动作(B=0 前向,部署本来就要跑的那一遍)。
# 规则(全部零训练):
#   UB-b1     按 b1 正确性排序取 top-2(用了 gold,不可部署,是"探针信号
#             上界"—— 若它都不涨,这条路死);
#   disagree  按"b1 动作与 draft 不一致"排序(可部署:不一致 = 该帧改变了
#             策略主意 = 信息帧假说);
#   agree     反向规则(佐证方向);
# 全部与 recent-2 同态配对,McNemar。选出的对不在 b2 表内则计 missing
# (侦察阶段如实报缺口,不 GPU 补测)。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def mcnemar_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def disagree(a: dict | None, b: dict | None, tol: float = 25.0) -> bool:
    if not a or not b:
        return a != b
    if a.get("action") != b.get("action"):
        return True
    ca, cb = a.get("coordinate"), b.get("coordinate")
    if ca and cb:
        return math.dist(ca, cb) > tol
    # 非坐标动作:文本/键位不同即不一致
    return any(a.get(k) != b.get(k) for k in ("text", "keys", "key"))


def main() -> None:
    d = Path(sys.argv[1])
    b1, b2 = {}, {}
    for f in sorted(d.glob("bcurve_b1_sh*.jsonl")):
        for line in f.open():
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("all"):
                    b1[r["dp_id"]] = r
    for f in sorted(d.glob("bcurve_b2_sh*.jsonl")):
        for line in f.open():
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("all"):
                    b2[r["dp_id"]] = r

    both = sorted(set(b1) & set(b2))
    hard = [dp for dp in both if not b2[dp]["b0_correct"]]
    print(json.dumps({"b1": len(b1), "b2": len(b2), "join": len(both),
                      "hard(b0错)": len(hard)}, ensure_ascii=False))

    def rank_pick(dp: str, key) -> frozenset | None:
        """key(j, e1) 越大越优先;并列取更近(j 大)。取 top-2。"""
        entries = [(a["s"][0], a) for a in b1[dp]["all"]]
        entries.sort(key=lambda x: (key(x[0], x[1]), x[0]), reverse=True)
        if len(entries) < 2:
            return None
        return frozenset(x[0] for x in entries[:2])

    rules = {
        "UB-b1(不可部署)": lambda j, e: (int(e["c"]),),
        "disagree-draft": lambda j, e, : (int(disagree(e.get("p"),
                                                       b2[cur_dp]["b0_pred"])),),
        "agree-draft": lambda j, e: (int(not disagree(e.get("p"),
                                                      b2[cur_dp]["b0_pred"])),),
    }

    for stratum, ids in (("hard(b0 错,n 全量)", hard),
                         ("winnable(有对也有错的帧对)",
                          [dp for dp in hard
                           if any(a["c"] for a in b2[dp]["all"])
                           and not all(a["c"] for a in b2[dp]["all"])])):
        print(f"\n== {stratum}: n={len(ids)} ==")
        rec_acc = sum(b2[dp]["recent_correct"] for dp in ids) / max(len(ids), 1)
        orc_acc = sum(any(a["c"] for a in b2[dp]["all"])
                      for dp in ids) / max(len(ids), 1)
        print(f"recent-2 基线 {100*rec_acc:.1f}%   oracle 上界 {100*orc_acc:.1f}%")
        for name, key in rules.items():
            n01 = n10 = miss = used = 0
            hit = rechit = 0
            for dp in ids:
                global cur_dp
                cur_dp = dp
                pick = rank_pick(dp, key)
                subs = {frozenset(a["s"]): a["c"] for a in b2[dp]["all"]}
                if pick is None or pick not in subs:
                    miss += 1
                    continue
                used += 1
                c_sel, c_rec = subs[pick], b2[dp]["recent_correct"]
                hit += c_sel
                rechit += c_rec
                if c_sel and not c_rec:
                    n01 += 1
                if c_rec and not c_sel:
                    n10 += 1
            if used:
                print(f"{name}: n={used}(missing {miss}) "
                      f"sel={100*hit/used:.1f}% rec={100*rechit/used:.1f}% "
                      f"diff={100*(hit-rechit)/used:+.2f}pp "
                      f"赢/输={n01}/{n10} p={mcnemar_p(n01, n10):.4f}")


if __name__ == "__main__":
    main()
