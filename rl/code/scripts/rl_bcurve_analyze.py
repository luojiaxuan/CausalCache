#!/usr/bin/env python3
"""B 曲线归约:在新指标下比较 B ∈ {1,2,4} 的 oracle 头寸与净收益。

# note (luojiaxuan): 这条曲线要回答的是 **"B=2 这个工作点站不站得住"**。
# 旧依据(teacher-forced log-prob DiD:B=2 头寸 +0.102 / B=4 只剩 +0.037)
# 用的是已被推翻的指标,不能再用;而 frozen_main_row_v1 显示 B2→B4 在真实
# 任务成功率上还有 +4.6pt,所以"B=2 头寸最大"必须在新指标下重新证明。
#
# 三件事必须同时报,少一件结论就有偏:
#
#   1) **配对**:三个 B 只在共同的 dp_id 上比。不同 B 若样本不同,曲线的
#      升降会混进样本差异。
#   2) **easy 态的反向效应**:easy 按 B=0 定义;B 变大后多出来的历史图完全
#      可能把本来答对的题干扰错。净收益 = winnable 修好数 − easy 弄坏数。
#      只在非 easy 上比会系统性偏向大 B。
#   3) **重加权**:easy 是抽样进来的(800/2809),非 easy 全量。直接平均等于
#      把 easy 的权重压到 1/3.5,总体口径的数字必须按真实占比加权还原。
#
# oracle 与 recent/随机 的差**依然是结构性非负**(recent-B 与 随机-B 都是被
# 枚举的子集之一),只能读作"上界有多高",不可做检验。可检验的是 recent-B
# 之间跨 B 的差 —— 那是真实可部署的两个配置在同一批态上的配对对照。
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib


def load(paths: list[pathlib.Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["dp_id"]] = r
    return out


def mcnemar(b: int, c: int) -> float:
    """配对二项检验(双侧精确版)。b/c = 两臂结果不一致的两种方向的计数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    p = argparse.ArgumentParser()
    # note (luojiaxuan): 三个 B 都设成可选,是为了在 B=4 还没跑完时就能先出
    # B=1/B=2 的部分曲线 —— 目的不是提前下结论,而是**提前把归约脚本跑通**。
    # 等最贵的那批数据落地才发现脚本有 bug,代价是重跑而不是重算。
    p.add_argument("--b1", nargs="*", type=pathlib.Path, default=[])
    p.add_argument("--b2", nargs="*", type=pathlib.Path, default=[],
                   help="已有的全枚举结果(labels_all.jsonl),不重跑")
    p.add_argument("--b4", nargs="*", type=pathlib.Path, default=[])
    p.add_argument("--easy-total", type=int, default=2809,
                   help="总体中 easy 态的真实个数(用于抽样重加权)")
    p.add_argument("--hard-total", type=int, default=2494)
    p.add_argument("--min-stratum", type=int, default=50,
                   help="每层最少态数;不足则拒绝输出重加权的总体数字")
    p.add_argument("--tolerance", type=float, default=None,
                   help="按此容差**重算**四臂正确性(需要产物里带 preds)。"
                        "不给则沿用生成时烘焙的判定。有了它,'容差取多少'"
                        "就是分析时的旋钮,换一个数不必重跑 GPU,还能报"
                        "'B 的选择随容差怎么变'这条稳健性曲线")
    args = p.parse_args()

    data = {b: load(getattr(args, f"b{b}")) for b in (1, 2, 4)}

    if args.tolerance is not None:
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from rl_oracle_enumerate import action_correct
        missing = recomputed = 0
        for d in data.values():
            for r in d.values():
                if "gold" not in r or not any("p" in a for a in r.get("all", [])):
                    missing += 1
                    continue
                gold = r["gold"]
                by = {tuple(a["s"]): a for a in r["all"]}
                for a in r["all"]:
                    a["c"] = action_correct(a.get("p"), gold, tolerance=args.tolerance)
                r["oracle_correct"] = any(a["c"] for a in r["all"] if a.get("e"))
                for arm, key in (("recent", "recent_s"), ("random", "random_s")):
                    ent = by.get(tuple(r.get(key, [])))
                    if ent is not None:
                        r[f"{arm}_correct"] = ent["c"]
                if "b0_pred" in r:
                    r["b0_correct"] = action_correct(r["b0_pred"], gold,
                                                     tolerance=args.tolerance)
                recomputed += 1
        print(f"按容差 {args.tolerance:g} 重算:{recomputed} 态;"
              f"{missing} 态缺 preds 无法重算(沿用原判定)")
        if missing:
            print("⚠️ 混用了重算与原判定的态 —— 这在同一张表里是两套口径,"
                  "要么把缺 preds 的产物重跑,要么把它们排除。")
    budgets = [b for b in (1, 2, 4) if data[b]]
    for b in (1, 2, 4):
        print(f"B={b}: 载入 {len(data[b])} 态" + ("" if data[b] else "(缺,跳过)"))
    if not budgets:
        raise SystemExit("三个 B 一个都没给")
    if len(budgets) < 3:
        print(f"⚠️ 只有 B∈{budgets} 有数据 —— 这是**部分曲线**,"
              f"不能据此对 B=2 的选型下结论。")
    # note (luojiaxuan): **两个不同的分母,不能混为一谈**。
    #   common      —— 三个 B 都测过的态,用于**可部署臂**(recent-B / 随机-B /
    #                  B=0)的跨预算对照。easy 态走 --arms-only 也在此列。
    #   common_orc  —— 其中还做了 oracle 枚举的态,用于**上界**。
    # easy 态刻意不枚举 oracle(它接近饱和,48-83 次前向换一个几乎恒为真的量
    # 不划算),所以 oracle 只在困难层定义 —— 报告时必须写明,不能让读者
    # 以为总体 oracle 也测了。
    common = set.intersection(*(set(data[b]) for b in budgets))
    common_orc = {i for i in common
                  if all("oracle_correct" in data[b][i] for b in budgets)}
    print(f"\nB∈{budgets} 共同的态:{len(common)}"
          f";其中做了 oracle 枚举的:{len(common_orc)}")
    if not common:
        raise SystemExit("交集为空:检查白名单与分片是否对齐")

    # 分层用 B=0(与预算无关);优先取自 B=2 那批,缺时退到任一可用 B ——
    # b0 与预算无关,但**必须全程用同一批的 b0**,否则 B=0 那约 2% 的
    # 数值非确定性会让分层在不同 B 之间漂移,配对就假了。
    ref = 2 if data[2] else budgets[0]
    print(f"分层基准取自 B={ref} 那批的 b0_correct")
    easy = {i for i in common if data[ref][i]["b0_correct"]}
    hard = common - easy
    w_easy = args.easy_total / max(len(easy), 1)
    w_hard = args.hard_total / max(len(hard), 1)
    # note (luojiaxuan): 分层太小就**拒绝出总体数字**,不许靠权重把 5 个态
    # 放大成 2809 个。第一次空跑就踩到了:easy 层只剩 5 态、权重 561.8×,
    # 原因是复用的 labels_all 里 easy 态走了 --skip-easy、根本没有 oracle。
    # 这种情况下"重加权到总体"是数字幻觉,必须报错而不是打印出来。
    for name, grp, w in (("easy", easy, w_easy), ("非easy", hard, w_hard)):
        if len(grp) < args.min_stratum:
            raise SystemExit(
                f"{name} 层只有 {len(grp)} 态(权重 {w:.1f}×),低于下限 "
                f"{args.min_stratum} —— 重加权会把少数几个态放大成总体,"
                f"这是数字幻觉不是估计。先把该层的枚举补齐,或用 "
                f"--min-stratum 显式下调并在报告里写明。")
    print(f"分层:easy {len(easy)}(权重 {w_easy:.2f})、非easy {len(hard)}"
          f"(权重 {w_hard:.2f});重加权还原到总体 "
          f"{args.easy_total}+{args.hard_total}={args.easy_total + args.hard_total}")

    sat = {b: sum(1 for i in common if data[b][i].get("eff_budget", b) < b) for b in budgets}

    # note (luojiaxuan): **剪枝覆盖率必须逐 B 报,而且它是对大 B 不利的偏差。**
    # 完整子集空间 C(n,B) 随 B 组合爆炸,而剪枝池只线性变大(top_k + B − 1 里
    # 取 B 个),所以覆盖率随 B 单调下降 —— 大 B 的 oracle 被低估得更狠。
    # 若最终结论是"B=2 已够、不必上 B=4",这条偏差正好指向该结论,
    # **必须主动披露**,否则就是拿方法的近似误差去支持自己想要的答案。
    print("\n=== 剪枝覆盖率(枚举池 / 完整 C(候选数,B) 空间)===")
    print(f"{'B':>3} {'覆盖率中位':>11} {'p10':>7} {'池大小中位':>11}")
    for b in budgets:
        fr, ps = [], []
        for i in common:
            r = data[b][i]
            pool = sum(1 for a in r.get("all", []) if a.get("e"))
            if not pool:
                continue                      # arms_only 行不参与
            full = math.comb(r["n_candidates"], r.get("eff_budget", b))
            fr.append(pool / max(full, 1)); ps.append(pool)
        if not fr:
            continue
        fr.sort(); ps.sort()
        print(f"{b:>3} {100*fr[len(fr)//2]:>10.1f}% {100*fr[int(.1*len(fr))]:>6.1f}% "
              f"{ps[len(ps)//2]:>11d}")
    print("  覆盖率随 B 下降 = 大 B 的 oracle 被低估得更多(对大 B 不利的偏差)。")
    print(f"预算饱和(候选数 < B,有效预算被迫降级)条数:{sat}")

    def rate(b: int, arm: str, ids: set[str]) -> float:
        sel = ids & common_orc if arm == "oracle" else ids
        if not sel:
            return float("nan")
        return sum(data[b][i][f"{arm}_correct"] for i in sel) / len(sel)

    def pop(b: int, arm: str) -> float:
        """按真实占比重加权的总体正确率。"""
        num = rate(b, arm, easy) * args.easy_total + rate(b, arm, hard) * args.hard_total
        return num / (args.easy_total + args.hard_total)

    print("\n=== 可部署臂:总体正确率(重加权到真实占比,分母 = 全部 "
          f"{len(common)} 态)===")
    print(f"{'B':>3} {'recent-B':>10} {'随机-B':>9} {'rec−随机':>10}")
    for b in budgets:
        print(f"{b:>3} {100*pop(b,'recent'):>9.1f}% {100*pop(b,'random'):>8.1f}% "
              f"{100*(pop(b,'recent')-pop(b,'random')):>+9.1f}pp")
    print("  这一张是选工作点的依据:两条臂都真实可部署,跨预算的差可检验。")

    hard_orc = hard & common_orc
    print(f"\n=== 上界:oracle 头寸(**只在困难层定义**,n={len(hard_orc)})===")
    print("  easy 态刻意未枚举 oracle —— 它接近饱和(只要存在保住正确的子集,"
          "oracle 必然找到),花 48-83 次前向测一个几乎恒为真的量不划算。")
    print(f"{'B':>3} {'oracle':>9} {'recent-B':>10} {'头寸 or−rec':>12}")
    for b in budgets:
        o, r = rate(b, "oracle", hard), rate(b, "recent", hard_orc)
        print(f"{b:>3} {100*o:>8.1f}% {100*r:>9.1f}% {100*(o-r):>+11.1f}pp")
    print("  头寸是结构性非负(recent-B 本身就是被枚举的子集之一),"
          "只能读作'上界有多高',不可做检验。")
    b0 = sum(data[ref][i]["b0_correct"] for i in common)
    print(f"\n参照 B=0(不给任何历史图):样本内 {100*b0/len(common):.1f}%"
          f"(分母 {len(common)},即 easy 占比);按真实占比加权 "
          f"{100*args.easy_total/(args.easy_total+args.hard_total):.1f}%。"
          f"\n  **把 B=0 和 recent-B 并排看**:easy 层 B=0 按定义 100%,"
          f"而 recent-B 低于 100% 的部分,就是多给历史图**弄坏**的题。")

    print("\n=== 分层拆开:大 B 修好了什么、又弄坏了什么(均为 recent-B)===")
    print(f"{'B':>3} {'easy层':>10} {'非easy层':>10}")
    for b in budgets:
        print(f"{b:>3} {100*rate(b,'recent',easy):>9.1f}% "
              f"{100*rate(b,'recent',hard):>9.1f}%")
    print("  easy 层 recent 若随 B 下降 = 多给的历史图在干扰本来答对的题;"
          "非easy 层上升 = 大 B 确实带进了缺失信息。")
    # note (luojiaxuan): 直接把净收益算出来,不要让读者自己去减两个百分比 ——
    # 两层的**权重不同**(easy 占总体 53%),分层正确率相减是错的算法。
    print(f"\n{'B':>3} {'easy层被弄坏':>13} {'非easy层被修好':>15} "
          f"{'加权净收益(相对 B=0)':>22}")
    for b in budgets:
        broke = 1 - rate(b, "recent", easy)          # easy 层 B=0 按定义 100%
        fixed = rate(b, "recent", hard)              # 非easy 层 B=0 按定义 0%
        net = (fixed * args.hard_total - broke * args.easy_total) / (
            args.easy_total + args.hard_total)
        print(f"{b:>3} {100*broke:>12.1f}% {100*fixed:>14.1f}% {100*net:>+21.1f}pp")
    print("  净收益 = (修好数 × 非easy 占比 − 弄坏数 × easy 占比) / 总数。"
          "为负说明在这个预算下,给历史图整体是**亏的**。")

    print("\n=== recent-B 跨预算配对检验(这是唯一可检验的对照)===")
    print("  两个真实可部署配置在同一批态上的配对差;oracle 与 recent 的差是"
          "结构性非负,不在此列。")
    for lo, hi in [(a, c) for a, c in ((1, 2), (2, 4), (1, 4))
                   if a in budgets and c in budgets]:
        bb = sum(1 for i in common
                 if data[hi][i]["recent_correct"] and not data[lo][i]["recent_correct"])
        cc = sum(1 for i in common
                 if data[lo][i]["recent_correct"] and not data[hi][i]["recent_correct"])
        d = 100 * (rate(hi, "recent", common) - rate(lo, "recent", common))
        print(f"  recent-{hi} − recent-{lo} = {d:+.1f}pp(未加权,n={len(common)});"
              f"翻正 {bb}、翻负 {cc},p={mcnemar(bb, cc):.3f}")

    print("\n=== 每态成本(部署侧,用于和头寸对照)===")
    print(f"{'B':>3} {'视觉 token':>11} {'相对 B=2':>9}")
    for b in budgets:
        print(f"{b:>3} {b*2560:>11d} {b/2:>8.1f}×")
    print("  头寸随 B 的增量若小于成本增量,B 就该往小取——这是选工作点的依据,"
          "比'哪个 B 头寸最大'更站得住,因为头寸本身不是可实现的收益。")


if __name__ == "__main__":
    main()
