#!/usr/bin/env python3
"""在**完全相同的集合**上并排比较两个适配器的 k 分层效应。

# note (luojiaxuan): 为什么不能直接拿两次独立跑的 k 分层结果相比 ——
# 集合是 beam 搜出来的,不同适配器探索到的集合不同;加上抽样(--limit-states),
# 两次测量的 common set membership 根本不是一回事。比"水平"就会混进抽样差异。
# 这里取三方交集(frozen ∩ A ∩ B),在同一批集合上同时算 A 与 B 的
# A(S) = U_adapter(S) − U_frozen(S),n 完全一致,差异只能来自适配器本身。
#
# 判读要看**两件事**,单看一件会得出相反结论:
#   k=0 的均匀成分     —— 修法要压小的量;
#   (k=2 − k=0) 的选择性 —— 不能跟着一起塌;
# 以及派生的 **均匀占比 = k0 / k2**,它在每次测量内部计算,对抽样最不敏感。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_sets(root: Path) -> dict[tuple[str, tuple[int, ...]], float]:
    out: dict[tuple[str, tuple[int, ...]], float] = {}
    for path in sorted(root.glob("sets.shard*.jsonl")):
        with path.open(errors="ignore") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") != "set":
                    continue
                key = (row["dp_id"], tuple(sorted(int(e) for e in row["events"])))
                out.setdefault(key, float(row["score"]))
    return out


def load_pools(root: Path) -> dict[str, list[int]]:
    pools: dict[str, list[int]] = {}
    for path in sorted(root.glob("singletons.shard*.jsonl")):
        with path.open(errors="ignore") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") == "b0":
                    pools[row["dp_id"]] = [int(e) for e in row["candidate_pool"]]
    return pools


def ci95(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, m, m
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    h = 1.96 * sd / math.sqrt(n)
    return m, m - h, m + h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-sets", type=Path, required=True)
    p.add_argument("--adapter", action="append", required=True,
                   metavar="LABEL=DIR", help="标签=set 打分目录,可重复")
    p.add_argument("--singletons-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    frozen = load_sets(args.frozen_sets)
    pools = load_pools(args.singletons_root)
    arms = []
    for spec in args.adapter:
        label, _, d = spec.partition("=")
        arms.append((label, load_sets(Path(d))))

    common = set(frozen)
    for _label, m in arms:
        common &= set(m)
    common = {k for k in common if k[0] in pools}
    print(json.dumps({"frozen": len(frozen),
                      **{f"sets_{l}": len(m) for l, m in arms},
                      "三方交集": len(common)}, ensure_ascii=False))

    def k_of(key: tuple[str, tuple[int, ...]]) -> int:
        dp, events = key
        recent = sorted(pools[dp], reverse=True)[: len(events)]
        return len(set(events) - set(recent))

    report: dict[str, dict] = {"n_common": len(common), "arms": {}}
    rows = []
    for label, m in arms:
        by_k: dict[int, list[float]] = defaultdict(list)
        for key in common:
            by_k[k_of(key)].append(m[key] - frozen[key])
        stats = {}
        for k in sorted(by_k):
            mean, lo, hi = ci95(by_k[k])
            stats[k] = {"n": len(by_k[k]), "mean": mean, "ci": [lo, hi]}
        k0 = stats.get(0, {}).get("mean")
        k2 = stats.get(2, {}).get("mean")
        share = (k0 / k2) if (k0 is not None and k2) else None
        report["arms"][label] = {"by_k": stats, "uniform_share_at_k2": share}
        rows.append((label, stats, share))

    print("\n=== 同底 k 分层(n 完全一致) ===")
    for label, stats, share in rows:
        print(f"  {label}")
        for k in sorted(stats):
            s = stats[k]
            print(f"    k={k}: n={s['n']:6d}  {s['mean']:+.5f}  "
                  f"CI[{s['ci'][0]:+.5f},{s['ci'][1]:+.5f}]")
        if share is not None:
            sel = stats[2]["mean"] - stats[0]["mean"]
            print(f"    → 选择性(k2−k0) = {sel:+.5f} | 均匀占比 k0/k2 = {share:.0%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
