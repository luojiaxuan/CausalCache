#!/usr/bin/env python3
"""汇总 probe_desktop_official_k 的分片输出 → k 边际判定表。

# note (luojiaxuan): 读数(episode-cluster bootstrap,默认 2000 次):
#   * frozen_margin(k) = U_f(S_k) − U_f(recent)(bypass 口径)—— 新格式下冻结
#     模型的 k 替换边际;k=1 即 frozen_selection_effect 的官方结构版;
#   * s300_did(k) = [U_a(S_k)−U_0(S_k)] − [U_a(recent)−U_0(recent)] —— 旧格式
#     HGKV s300 的选择性在官方结构下的迁移读数;
#   * 每列同时给 all-states 与 witness_count>=k 子群(k 边际只在可组出 S_k 的
#     状态上有定义,表里 n 即该子群);
#   * availability:witness_count 分布(k≥2 语料臂的原料量)。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def cluster_ci(values_by_episode, *, iterations: int, seed: int):
    episodes = sorted(values_by_episode)
    flat = [v for ep in episodes for v in values_by_episode[ep]]
    if not flat:
        return None
    point = sum(flat) / len(flat)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [
            v
            for ep in (rng.choice(episodes) for _ in episodes)
            for v in values_by_episode[ep]
        ]
        if sample:
            means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(int(0.975 * len(means)), len(means) - 1)]
    return {"point": point, "ci_low": lo, "ci_high": hi, "n": len(flat)}


def main() -> None:
    args = parse_args()
    rows = []
    fingerprint = None
    for path in sorted(args.probe_root.glob("probe_k.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("key") == "__fingerprint__":
                fingerprint = row
                continue
            if row.get("skipped"):
                continue
            rows.append(row)

    availability = Counter(row["witness_count"] for row in rows)
    result = {
        "fingerprint": fingerprint,
        "states": len(rows),
        "witness_count_distribution": dict(sorted(availability.items())),
        "per_k": {},
        "per_k_dev_only": {},
    }
    for dev_only in (False, True):
        target = result["per_k_dev_only" if dev_only else "per_k"]
        for k in (1, 2, 3, 4):
            frozen = defaultdict(list)
            did = defaultdict(list)
            for row in rows:
                if dev_only and row["split"] != "dev":
                    continue
                scores = row["scores"]
                sel = scores.get(f"k{k}")
                if sel is None:
                    continue
                recent = scores["recent"]
                frozen[row["episode"]].append(sel["bypass"] - recent["bypass"])
                did[row["episode"]].append(
                    (sel["active"] - sel["bypass"])
                    - (recent["active"] - recent["bypass"])
                )
            target[str(k)] = {
                "frozen_margin": cluster_ci(
                    frozen, iterations=args.bootstrap, seed=args.seed + k
                ),
                "s300_did_transfer": cluster_ci(
                    did, iterations=args.bootstrap, seed=args.seed + 100 + k
                ),
            }
    # k 边际差(k vs k-1,配对在同状态上):增量收益读数
    for k in (2, 3, 4):
        paired = defaultdict(list)
        for row in rows:
            scores = row["scores"]
            if f"k{k}" in scores and f"k{k-1}" in scores:
                paired[row["episode"]].append(
                    scores[f"k{k}"]["bypass"] - scores[f"k{k-1}"]["bypass"]
                )
        result[f"frozen_marginal_k{k}_vs_k{k-1}"] = cluster_ci(
            paired, iterations=args.bootstrap, seed=args.seed + 200 + k
        )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
