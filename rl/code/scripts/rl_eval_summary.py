#!/usr/bin/env python3
"""iter-N held-out 双轮评测合并读数(RL vs 冻结+recent-B)。

# note (luojiaxuan): 单轮 summary.json 只给"配对数/各臂成功数/胜负",不足以
# 下判断。本脚本补三件事:
#   1. 完备性闸门——任何一臂不足 --need 条直接拒绝出数(缺口集中在长尾困难
#      任务,会系统性偏袒 RL,方向读已踩过);
#   2. 任务级配对检验——每任务两轮平均成功率 ∈ {0,0.5,1},对差值做符号检验
#      (精确二项,双侧)+ bootstrap 置信区间;
#   3. 环境噪声标定——同一臂同一任务在两轮间翻转的比例(test-retest)。
#      这是解释力最强的一项:若单臂自己就翻转 X%,那么小于该量级的臂间差异
#      不具可解释性。没有它,任何 ±N 的读数都无法判断该不该当真。
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random


def arm_results(round_dir: pathlib.Path, arm: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in (round_dir / f"out-{arm}").rglob("result.json"):
        try:
            out[f.parent.name] = int(bool(json.loads(f.read_text()).get("success")))
        except Exception:
            pass
    return out


def sign_test(wins: int, losses: int) -> float:
    """双侧精确符号检验(平局已排除)。"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", nargs="+", required=True,
                   help="各轮评测目录(如 .../eval-iter20 .../eval-iter20-r2)")
    p.add_argument("--need", type=int, default=120, help="每臂应有条数,不足即拒绝")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--output", type=pathlib.Path, default=None)
    args = p.parse_args()

    rounds = [pathlib.Path(r) for r in args.rounds]
    data: list[dict[str, dict[str, int]]] = []
    problems = []
    for rd in rounds:
        rl, rec = arm_results(rd, "rl"), arm_results(rd, "recent")
        if len(rl) < args.need or len(rec) < args.need:
            problems.append(f"{rd.name}: rl={len(rl)} recent={len(rec)} (<{args.need})")
        data.append({"rl": rl, "recent": rec})
    if problems:
        print("完备性闸门未通过,拒绝出数(缺口偏袒 RL 臂):")
        for x in problems:
            print("  " + x)
        raise SystemExit(2)

    # 三方交集:两轮 × 两臂都有结果的任务
    tasks = set(data[0]["rl"])
    for d in data:
        tasks &= set(d["rl"]) & set(d["recent"])
    tasks = sorted(tasks)
    R = len(rounds)

    report: dict[str, object] = {"rounds": [r.name for r in rounds],
                                 "tasks_common": len(tasks)}

    # ---- 每轮单独读数 ----
    per_round = []
    for i, d in enumerate(data):
        rl_s = sum(d["rl"][t] for t in tasks)
        rc_s = sum(d["recent"][t] for t in tasks)
        w = sum(1 for t in tasks if d["rl"][t] > d["recent"][t])
        l = sum(1 for t in tasks if d["rl"][t] < d["recent"][t])
        per_round.append({"round": rounds[i].name, "rl": rl_s, "recent": rc_s,
                          "delta": rl_s - rc_s, "wins": w, "losses": l,
                          "sign_p": round(sign_test(w, l), 4)})
    report["per_round"] = per_round

    # ---- 任务级(两轮平均)配对 ----
    rl_rate = {t: sum(d["rl"][t] for d in data) / R for t in tasks}
    rc_rate = {t: sum(d["recent"][t] for d in data) / R for t in tasks}
    diffs = [rl_rate[t] - rc_rate[t] for t in tasks]
    w = sum(1 for x in diffs if x > 0)
    l = sum(1 for x in diffs if x < 0)
    mean_rl = sum(rl_rate.values()) / len(tasks)
    mean_rc = sum(rc_rate.values()) / len(tasks)

    rng = random.Random(args.seed)
    boots = []
    for _ in range(args.bootstrap):
        s = sum(diffs[rng.randrange(len(diffs))] for _ in range(len(diffs)))
        boots.append(s / len(diffs))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots)) - 1]

    report["pooled"] = {
        "rl_rate": round(mean_rl, 4), "recent_rate": round(mean_rc, 4),
        "delta_rate": round(mean_rl - mean_rc, 4),
        "delta_tasks_equiv": round((mean_rl - mean_rc) * len(tasks), 2),
        "ci95": [round(lo, 4), round(hi, 4)],
        "wins": w, "losses": l, "ties": len(tasks) - w - l,
        "sign_p": round(sign_test(w, l), 4),
    }

    # ---- 环境噪声标定:同臂跨轮翻转率 ----
    if R >= 2:
        flips = {}
        for arm in ("rl", "recent"):
            f = sum(1 for t in tasks
                    if len({d[arm][t] for d in data}) > 1)
            flips[arm] = {"flipped": f, "rate": round(f / len(tasks), 4)}
        report["test_retest_flip"] = flips
        report["note"] = ("同臂跨轮翻转率 = 环境噪声下限;臂间差异若不显著大于"
                          "该量级,不具可解释性")

    print(json.dumps(report, indent=1, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
