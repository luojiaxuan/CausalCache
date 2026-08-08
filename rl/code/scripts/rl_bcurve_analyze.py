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
    p.add_argument("--b1", nargs="+", type=pathlib.Path, required=True)
    p.add_argument("--b2", nargs="+", type=pathlib.Path, required=True,
                   help="已有的全枚举结果(labels_all.jsonl),不重跑")
    p.add_argument("--b4", nargs="+", type=pathlib.Path, required=True)
    p.add_argument("--easy-total", type=int, default=2809,
                   help="总体中 easy 态的真实个数(用于抽样重加权)")
    p.add_argument("--hard-total", type=int, default=2494)
    args = p.parse_args()

    data = {1: load(args.b1), 2: load(args.b2), 4: load(args.b4)}
    for b, d in data.items():
        print(f"B={b}: 载入 {len(d)} 态")
    common = set(data[1]) & set(data[2]) & set(data[4])
    # B=2 那批里 --skip-easy 的行没有 oracle,进不了对照
    common = {i for i in common if all("oracle_correct" in data[b][i] for b in (1, 2, 4))}
    print(f"\n三个 B 共同且都有 oracle 的态:{len(common)}")
    if not common:
        raise SystemExit("交集为空:检查白名单与分片是否对齐")

    # 分层用 B=0(与预算无关),取自 B=2 那批以保证三个 B 用同一套分层
    easy = {i for i in common if data[2][i]["b0_correct"]}
    hard = common - easy
    w_easy = args.easy_total / max(len(easy), 1)
    w_hard = args.hard_total / max(len(hard), 1)
    print(f"分层:easy {len(easy)}(权重 {w_easy:.2f})、非easy {len(hard)}"
          f"(权重 {w_hard:.2f});重加权还原到总体 "
          f"{args.easy_total}+{args.hard_total}={args.easy_total + args.hard_total}")

    sat = {b: sum(1 for i in common if data[b][i].get("eff_budget", b) < b) for b in (1, 2, 4)}
    print(f"预算饱和(候选数 < B,有效预算被迫降级)条数:{sat}")

    def rate(b: int, arm: str, ids: set[str]) -> float:
        if not ids:
            return float("nan")
        return sum(data[b][i][f"{arm}_correct"] for i in ids) / len(ids)

    def pop(b: int, arm: str) -> float:
        """按真实占比重加权的总体正确率。"""
        num = rate(b, arm, easy) * args.easy_total + rate(b, arm, hard) * args.hard_total
        return num / (args.easy_total + args.hard_total)

    print("\n=== 总体正确率(重加权到真实占比)===")
    print(f"{'B':>3} {'oracle':>9} {'recent-B':>10} {'随机-B':>9} "
          f"{'头寸 or−rec':>12}")
    for b in (1, 2, 4):
        print(f"{b:>3} {100*pop(b,'oracle'):>8.1f}% {100*pop(b,'recent'):>9.1f}% "
              f"{100*pop(b,'random'):>8.1f}% {100*(pop(b,'oracle')-pop(b,'recent')):>+11.1f}pp")
    b0 = sum(data[2][i]["b0_correct"] for i in common)
    print(f"  参照 B=0(未加权 {100*b0/len(common):.1f}%;按定义 easy 层 100%、"
          f"非easy 层 0% → 加权 {100*args.easy_total/(args.easy_total+args.hard_total):.1f}%)")

    print("\n=== 分层拆开:大 B 修好了什么、又弄坏了什么 ===")
    print(f"{'B':>3} {'easy层 recent':>14} {'easy层 oracle':>14} "
          f"{'非easy层 recent':>16} {'非easy层 oracle':>16}")
    for b in (1, 2, 4):
        print(f"{b:>3} {100*rate(b,'recent',easy):>13.1f}% {100*rate(b,'oracle',easy):>13.1f}% "
              f"{100*rate(b,'recent',hard):>15.1f}% {100*rate(b,'oracle',hard):>15.1f}%")
    print("  easy 层 recent 若随 B 下降 = 多给的历史图在干扰本来答对的题;"
          "非easy 层上升 = 大 B 确实带进了缺失信息。净收益是两者相抵后的结果。")

    print("\n=== recent-B 跨预算配对检验(这是唯一可检验的对照)===")
    print("  两个真实可部署配置在同一批态上的配对差;oracle 与 recent 的差是"
          "结构性非负,不在此列。")
    for lo, hi in ((1, 2), (2, 4), (1, 4)):
        bb = sum(1 for i in common
                 if data[hi][i]["recent_correct"] and not data[lo][i]["recent_correct"])
        cc = sum(1 for i in common
                 if data[lo][i]["recent_correct"] and not data[hi][i]["recent_correct"])
        d = 100 * (rate(hi, "recent", common) - rate(lo, "recent", common))
        print(f"  recent-{hi} − recent-{lo} = {d:+.1f}pp(未加权,n={len(common)});"
              f"翻正 {bb}、翻负 {cc},p={mcnemar(bb, cc):.3f}")

    print("\n=== 每态成本(部署侧,用于和头寸对照)===")
    print(f"{'B':>3} {'视觉 token':>11} {'相对 B=2':>9}")
    for b in (1, 2, 4):
        print(f"{b:>3} {b*2560:>11d} {b/2:>8.1f}×")
    print("  头寸随 B 的增量若小于成本增量,B 就该往小取——这是选工作点的依据,"
          "比'哪个 B 头寸最大'更站得住,因为头寸本身不是可实现的收益。")


if __name__ == "__main__":
    main()
