#!/usr/bin/env python3
"""决策步诊断的归约:配对比较各记忆臂,输出 E5 诊断表。

# note (luojiaxuan): 主判据是 **oracle − recent2 的配对差**(同状态同前缀,
# 只换记忆子集),McNemar 精确检验 + 态级 bootstrap CI + 按 regime 分层。
# 期望的证据形态(环境有效即应看到):
#   * 需老帧的三种 regime 上 oracle ≫ recent2;
#   * recent_sufficient / history_irrelevant 上两者应接近(记忆不该有帮助);
#   * none 臂普遍最低,random2 居中 —— 说明"给什么记忆"确实改变行为。
用法:python3 agentic_probe_reduce.py rows_*.jsonl [--ref recent2]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path


def mcnemar_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(diffs: list[int], reps: int = 1999, seed: int = 12345):
    if not diffs:
        return 0.0, 0.0
    r = random.Random(seed)
    n = len(diffs)
    m = sorted(sum(r.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return m[int(0.025 * reps)], m[int(0.975 * reps)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+", type=Path)
    ap.add_argument("--ref", default="recent2")
    ap.add_argument("--arms", nargs="+",
                    default=["oracle", "recent2", "random2", "none"])
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for p in args.rows:
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    arms = [a for a in args.arms if f"{a}_correct" in (rows[0] if rows else {})]
    print(f"n={len(rows)} 臂={arms} 参照={args.ref}")

    def table(sel: list[dict], tag: str) -> dict:
        if not sel:
            return {}
        out = {"n": len(sel)}
        for a in arms:
            out[a] = round(sum(r[f"{a}_correct"] for r in sel) / len(sel), 4)
        line = f"{tag:26s} n={len(sel):4d} " + " ".join(
            f"{a}={100*out[a]:5.1f}%" for a in arms)
        for a in arms:
            if a == args.ref:
                continue
            d = [r[f"{a}_correct"] - r[f"{args.ref}_correct"] for r in sel]
            n01 = sum(1 for x in d if x > 0)
            n10 = sum(1 for x in d if x < 0)
            lo, hi = boot_ci(d)
            out[f"{a}_vs_{args.ref}"] = {
                "diff_pp": round(100 * sum(d) / len(d), 2),
                "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
                "win": n01, "lose": n10, "p": round(mcnemar_p(n01, n10), 5)}
        print(line)
        for a in arms:
            k = f"{a}_vs_{args.ref}"
            if k in out:
                v = out[k]
                print(f"    {a:9s}−{args.ref}: {v['diff_pp']:+6.2f}pp "
                      f"CI[{v['ci95_pp'][0]:+.2f},{v['ci95_pp'][1]:+.2f}] "
                      f"赢/输={v['win']}/{v['lose']} p={v['p']}")
        return out

    res = {"ALL": table(rows, "总体")}
    print()
    hard = {"one_old_frame", "two_frame_complementary", "distractor_heavy"}
    res["需老帧合并"] = table([r for r in rows if r["regime"] in hard], "需老帧合并")
    res["不需老帧合并"] = table([r for r in rows if r["regime"] not in hard],
                                "不需老帧合并")
    print()
    for reg in sorted({r["regime"] for r in rows}):
        res[reg] = table([r for r in rows if r["regime"] == reg], reg)
    print()
    cnt = collections.Counter(r["template_id"] for r in rows)
    print("模板覆盖:", len(cnt), "个;每模板样本中位数",
          sorted(cnt.values())[len(cnt) // 2] if cnt else 0)
    if args.json_out:
        args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("写出", args.json_out)


if __name__ == "__main__":
    main()
