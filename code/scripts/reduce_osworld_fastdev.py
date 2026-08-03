#!/usr/bin/env python3
"""135 任务闭环快测的三臂归约(A 组方向 + B 组回归 + 逐对翻转)。

# note (luojiaxuan): 两条纪律,都是本项目踩过的坑:
#   1. **只在全 roster 后运行**。中途采样已经误判过 7 次,其中一次进了投稿摘要。
#      本脚本先核对每臂的唯一 task_id 数,不足 --require-complete 就拒绝出对比表
#      (只报完成度),要看中途数必须显式传 --allow-partial,输出会带 INTERIM 水印。
#   2. **单轮 ±4pp 不构成证据**(同硬件单轮翻转率 7.5–14.4%)。判读只分三档:
#      方向一致 / 方向相反 / 差异在噪声带内;不做四位小数的宣称。
#
# A 组(74)= 构造时 recent 与 selector 都失败的任务 → 看"新解锁"数;
# B 组(61)= 回归守卫 → 盯"原来能过现在挂"的数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True,
                   help="osworld_fast_devset_v1.json(含 group_a/group_b)")
    p.add_argument("--arm", action="append", required=True,
                   metavar="LABEL=PATH",
                   help="臂标签={task_id:success} 的 JSON 文件,可重复")
    p.add_argument("--require-complete", type=int, default=135)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def rate(wins: int, n: int) -> str:
    return f"{wins}/{n} = {100.0 * wins / n:.1f}%" if n else "—"


def main() -> None:
    args = parse_args()
    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    group_a = set(man["group_a"])
    group_b = set(man["group_b"])
    all_tasks = group_a | group_b

    arms: list[tuple[str, dict[str, bool]]] = []
    for spec in args.arm:
        label, _, path = spec.partition("=")
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        arms.append((label, {k: bool(v) for k, v in raw.items() if k in all_tasks}))

    lines = ["# 135 任务闭环快测归约", ""]
    incomplete = [l for l, m in arms if len(m) < args.require_complete]
    if incomplete and not args.allow_partial:
        print("拒绝出对比表 —— 以下臂未到全 roster:")
        for label, m in arms:
            print(f"  {label}: {len(m)}/{args.require_complete}")
        raise SystemExit(2)
    if incomplete:
        lines += ["> **INTERIM —— 有臂未到全 roster,本表不可用于任何结论。**", ""]

    lines += ["| 臂 | 总体 | A 组(74 双失败) | B 组(61 回归守卫) |",
              "|---|---|---|---|"]
    for label, m in arms:
        a_ids = [t for t in m if t in group_a]
        b_ids = [t for t in m if t in group_b]
        lines.append(
            f"| {label} | {rate(sum(m.values()), len(m))} "
            f"| {rate(sum(m[t] for t in a_ids), len(a_ids))} "
            f"| {rate(sum(m[t] for t in b_ids), len(b_ids))} |")

    lines += ["", "## 逐对翻转(同一任务,两臂结果不同)", ""]
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            la, ma = arms[i]
            lb, mb = arms[j]
            common = sorted(set(ma) & set(mb))
            only_a = [t for t in common if ma[t] and not mb[t]]
            only_b = [t for t in common if mb[t] and not ma[t]]
            lines += [
                f"- **{la} vs {lb}**(共同 {len(common)}):"
                f"仅 {la} 过 {len(only_a)} 个,仅 {lb} 过 {len(only_b)} 个,"
                f"净差 {len(only_a) - len(only_b):+d}",
            ]
            for grp, name in ((group_a, "A"), (group_b, "B")):
                oa = sum(1 for t in only_a if t in grp)
                ob = sum(1 for t in only_b if t in grp)
                lines.append(f"    - {name} 组:{la} 独赢 {oa} / {lb} 独赢 {ob}")

    lines += ["", "判读口径:单轮 ±4pp 在噪声带内(同硬件单轮翻转率 7.5–14.4%);",
              "本表只回答方向,不做点估计宣称。"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
