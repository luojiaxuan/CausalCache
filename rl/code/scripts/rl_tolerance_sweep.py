#!/usr/bin/env python3
"""容差敏感性:四臂正确率随判定容差如何变化 + 预测-gold 距离分布。

# note (luojiaxuan): 25 这个容差是我从语料的 `coordinate_tolerance` 挪用的,
# 原本是筛正例帧用的,不是为评测设计的。它在归一 [0,999] 空间,折合
# 1080p 像素约 横 ±48 / 纵 ±27 —— **纵向接近一整行菜单高度**,审稿人可以
# 合理质疑"正确率被 off-by-one-row 灌水"。
#
# 用户提出:能不能干脆要求精确匹配、不给浮动?他的直觉是对的(没有魔法
# 数字最好辩护),但严格匹配可能把所有臂压到接近 0、失去分辨力。
# **这个争论应当用数据结束,不是靠辩。** 本脚本给两样东西:
#
#   1) 正确率 vs 容差曲线(0/5/10/25/50):若臂间差距在紧容差下依然存在,
#      就改用严格口径;若全塌到 0,则必须有容差,并**报整条曲线**;
#   2) 预测点到 gold 的距离分布:若呈双峰(一簇=点在同一元素上,
#      一簇=点错元素),**谷底就是有原则的阈值** —— 由数据决定而非我拍。
#
# 依赖评测产物里的 `preds` 与 `gold` 字段(2026-08-09 起落盘;此前只存
# 布尔值,容差被烘焙进生成阶段,无法事后重算)。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", type=pathlib.Path, required=True)
    p.add_argument("--tolerances", default="0,2,5,10,15,25,40,60")
    p.add_argument("--arms", default="learned,init,recent,random")
    args = p.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from rl_oracle_enumerate import _coord, action_correct

    rows = []
    seen = set()
    for f in args.results:
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("dp_id") in seen or "preds" not in d:
                continue
            seen.add(d["dp_id"])
            rows.append(d)
    if not rows:
        raise SystemExit("没有带 preds 的结果(旧版评测产物只有布尔值,需重跑)")

    arms = args.arms.split(",")
    tols = [float(x) for x in args.tolerances.split(",")]
    print(f"样本 {len(rows)} 态;带坐标的 gold 占比 "
          f"{sum(1 for r in rows if _coord(r['gold']) is not None)}/{len(rows)}")

    print("\n=== 正确率 vs 容差(归一 [0,999] 空间;1080p 下 25≈横48/纵27 px)===")
    print(f"{'容差':>6} " + " ".join(f"{a:>9}" for a in arms) + "   learned−recent")
    for t in tols:
        acc = {}
        for a in arms:
            hit = 0
            for r in rows:
                key = ",".join(map(str, r["picks"][a]))
                pred = r["preds"].get(key)
                hit += int(action_correct(pred, r["gold"], tolerance=t))
            acc[a] = hit / len(rows)
        delta = 100 * (acc.get("learned", 0) - acc.get("recent", 0))
        print(f"{t:>6.0f} " + " ".join(f"{100*acc[a]:>8.1f}%" for a in arms)
              + f"   {delta:+.1f}pp")

    # ---- 距离分布:找双峰谷底 ----
    dists = []
    for r in rows:
        g = _coord(r["gold"])
        if g is None:
            continue
        for a in arms:
            key = ",".join(map(str, r["picks"][a]))
            pred = r["preds"].get(key)
            if not pred:
                continue
            q = _coord(pred)
            if q is None:
                continue
            if str(pred.get("action", "")) != str(r["gold"].get("action", "")):
                continue        # 类型都不对,距离无意义
            dists.append(((g[0] - q[0]) ** 2 + (g[1] - q[1]) ** 2) ** 0.5)
    if dists:
        dists.sort()
        print(f"\n=== 预测-gold 距离分布(仅动作类型匹配者,n={len(dists)})===")
        edges = [0, 1, 2, 5, 10, 15, 25, 40, 60, 100, 200, 1e9]
        for lo, hi in zip(edges, edges[1:]):
            c = sum(1 for d in dists if lo <= d < hi)
            bar = "█" * int(60 * c / max(len(dists), 1))
            label = f"[{lo:g},{hi:g})" if hi < 1e9 else f"[{lo:g},∞)"
            print(f"  {label:>12} {c:5d} {100*c/len(dists):5.1f}% {bar}")
        print("  双峰则谷底可作有原则的阈值;单调衰减则说明没有天然分界,"
              "此时应报整条敏感性曲线而不是挑一个点")


if __name__ == "__main__":
    main()
