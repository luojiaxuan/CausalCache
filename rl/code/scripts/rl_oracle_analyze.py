#!/usr/bin/env python3
"""枚举 oracle 结果归约:头寸、四臂对照、三类拆分、剪枝命中率。

# note (luojiaxuan): 契约见 rl/docs/oracle_selection_contract.md。
#
# 为什么必须做三类拆分:冒烟第一个 state 连 B=0(不给任何历史图)都答对了。
# 这类"任何选择都对"的 state 会把平均头寸稀释掉——只报总体均值会把真实
# 头寸埋掉。三类:
#   easy      B=0 已正确        → 选择无杠杆(不算头寸)
#   winnable  B=0 错,但存在正确子集 → **这才是头寸所在**
#   hopeless  所有子集都错      → 选帧救不了(信息不在历史图里,或策略读不懂)
#
# 剪枝命中率**从全枚举离线推导,零额外算力**:单帧分数 = (候选, 最近候选)
# 那一对的正确性,已含在全枚举结果里;据此取 top-k 再配对,看是否仍能命中
# 一个正确子集。命中率 ≥90% 才允许后续用便宜版铺开。
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=pathlib.Path, required=True)
    p.add_argument("--top-k", type=int, default=8)
    args = p.parse_args()

    rows = [json.loads(x) for x in
            args.results.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise SystemExit("结果为空")
    full = [r for r in rows if r.get("mode") == "full"]
    # note (luojiaxuan): --skip-easy 的 screen 行**没有 oracle**(未枚举)。
    # 绝不可假定"B=0 对则四臂都对"——加历史图完全可能干扰掉本来正确的答案,
    # oracle 也不因 B=0 正确而必然正确(空集与 B=2 子集是不同的东西)。
    # 因此:oracle 正确率只在全枚举子集上报,分母显式区分。
    screen = [r for r in rows if r.get("mode") == "screen"]
    n = len(rows)
    known = [r for r in rows if "oracle_correct" in r]
    nk = len(known)
    arms = {k: sum(1 for r in known if r[f"{k}_correct"]) for k in
            ("oracle", "recent", "random", "b0")}
    if screen:
        print(f"注意:{len(screen)}/{n} 个 state 走了 --skip-easy 筛查(B=0 已正确,"
              f"未做全枚举),其 oracle/recent/随机 状态未知,不进四臂统计。")
    print(f"state 数 {n}(全枚举 {len(full)});候选数中位 "
          f"{sorted(r['n_candidates'] for r in rows)[n // 2]}")
    print(f"\n=== 四臂步级正确率(分母 = 有 oracle 的 {nk} 态)===")
    for k, label in (("oracle", "oracle-最优2"), ("recent", "recent-2"),
                     ("random", "随机-2"), ("b0", "B=0")):
        print(f"  {label:<12} {arms[k]:3d}/{nk}  {100 * arms[k] / max(nk,1):5.1f}%")
    # note (luojiaxuan): recent-2 与 随机-2 **都是被枚举的子集之一**,故
    # oracle 按构造必然 ≥ 两者。因此 oracle−recent / oracle−随机 只能读作
    # "上界有多高",**不能**做显著性检验,也**不能**当作"策略对内容敏感"的
    # 证据(那是结构性的,不是经验性的)。真正有效的内容敏感证据见下方
    # winnable 内的子集分化率与承载帧 lift;公平的臂间对照是 recent vs 随机。
    print(f"\n上界高度 oracle − recent = "
          f"{100 * (arms['oracle'] - arms['recent']) / max(nk,1):+.1f}pp"
          f"(结构性非负,不可做检验)")
    print(f"公平对照 recent − 随机 = "
          f"{100 * (arms['recent'] - arms['random']) / max(nk,1):+.1f}pp"
          f"(两者都是任选子集,这个差才是经验性的)")

    easy = [r for r in rows if r["b0_correct"]]
    winnable = [r for r in known if not r["b0_correct"] and r["oracle_correct"]]
    hopeless = [r for r in known if not r["b0_correct"] and not r["oracle_correct"]]
    print("\n=== 三类拆分(头寸只存在于 winnable)===")
    for label, grp in (("easy(B=0 已对)", easy), ("winnable(B=0 错但有解)", winnable),
                       ("hopeless(全错)", hopeless)):
        print(f"  {label:<24} {len(grp):3d}  {100 * len(grp) / n:5.1f}%")
    if winnable:
        rec = sum(1 for r in winnable if r["recent_correct"])
        rnd = sum(1 for r in winnable if r["random_correct"])
        print(f"  winnable 内:recent-2 命中 {rec}/{len(winnable)}、"
              f"随机-2 命中 {rnd}/{len(winnable)}")
        frac = [r["n_subsets_correct"] / max(r["n_subsets_scored"], 1) for r in winnable]
        print(f"  winnable 内正确子集占比中位 "
              f"{100 * sorted(frac)[len(frac) // 2]:.1f}%(越低=越需要精准选择)")

    # ---- winnable 里"谁是承载帧":逐事件的携带率 ----
    # note (luojiaxuan): 平均数看不出 selector 能不能学。真正要问的是:正确子集
    # 是集中在少数几个"承载帧"上(→ 可学,存在可预测的信号),还是均匀散布
    # (→ 不可学,靠碰运气)。携带率 = 含该事件的子集中正确的比例;
    # 与该 state 的基准正确率相比,高出越多说明该帧越关键。
    carriers = []
    for r in winnable:
        pairs = {tuple(a["s"]): a["c"] for a in r.get("all", []) if len(a["s"]) == 2}
        if not pairs:
            continue
        base = sum(pairs.values()) / len(pairs)
        events = sorted({c for s in pairs for c in s})
        per = []
        for e in events:
            sub = [v for s, v in pairs.items() if e in s]
            if sub:
                per.append((sum(sub) / len(sub), e, len(sub)))
        per.sort(reverse=True)
        top_rate, top_e, _ = per[0]
        # 归一化年龄:1=最老,0=最新(候选区间 [1, s-2])
        span = max(max(events) - min(events), 1)
        age = (max(events) - top_e) / span
        carriers.append((r["step"], base, top_rate, top_e, age, len(events)))
    if carriers:
        print(f"\n=== winnable 承载帧分析({len(carriers)} 态)===")
        print(f"{'step':>5} {'基准':>6} {'最强帧携带率':>12} {'该帧':>5} {'归一年龄':>8} {'候选数':>6}")
        for step, base, top, e, age, ne in carriers:
            print(f"{step:>5} {100*base:>5.0f}% {100*top:>11.0f}% {e:>5} {age:>8.2f} {ne:>6}")
        lift = sum(t - b for _, b, t, _, _, _ in carriers) / len(carriers)
        ages = sorted(c[4] for c in carriers)
        print(f"  平均提升(最强帧携带率 − 基准)= {100*lift:+.1f}pp"
              f";最强帧归一年龄中位 {ages[len(ages)//2]:.2f}")
        print("  提升大 = 正确子集集中在少数承载帧上(可学);接近 0 = 均匀散布(不可学)")

    # ---- 剪枝命中率(从全枚举推导)----
    hit = tot = 0
    for r in full:
        pairs = {tuple(a["s"]): a["c"] for a in r["all"] if len(a["s"]) == 2}
        if not any(pairs.values()):
            continue                      # 无正确子集,不参与命中率统计
        cands = sorted({c for s in pairs for c in s})
        partner = cands[-1]
        singles = []
        for c in cands:
            if c == partner:
                continue
            key = (c, partner) if c < partner else (partner, c)
            if key in pairs:
                singles.append((int(pairs[key]), c))
        top = sorted(c for _, c in sorted(singles, key=lambda t: (-t[0], -t[1]))[: args.top_k])
        pruned_ok = any(pairs.get(t, False)
                        for t in itertools.combinations(top, 2))
        hit += int(pruned_ok)
        tot += 1
    if tot:
        print(f"\n=== 剪枝校验(top-{args.top_k},{tot} 个有解 state)===")
        print(f"  命中率 {hit}/{tot} = {100 * hit / tot:.1f}%"
              f"({'≥90%,可用便宜版铺开' if hit / tot >= 0.9 else '<90%,需放大 top-k 或全枚举'})")


if __name__ == "__main__":
    main()
