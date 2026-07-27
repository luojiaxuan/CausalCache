#!/usr/bin/env python3
"""合并分片版 fill-to-B eval 输出,在全 dev 集上重算 episode-cluster CI。

# note (luojiaxuan): eval_selector_v4_fill_to_b.py --shard-count>1 时每片带 raw 明细
# (per_b/per_b_oracle 按 episode 的增量列表 + k_dist)。同一 episode 的状态可能散在
# 多片,合并时按 episode 扩展列表而不是覆盖。CI 与单进程版同一实现(直接 import),
# 判定口径不变。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.eval_selector_v4_fill_to_b import cluster_ci


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    per_b: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_b_oracle: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    k_dist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    skipped: dict[str, int] = defaultdict(int)
    fresh = 0
    seen_shards = set()
    base = None
    for path in args.shards:
        shard = json.loads(path.read_text(encoding="utf-8"))
        if "raw" not in shard:
            raise SystemExit(f"{path} 缺 raw 明细(需 --shard-count>1 的输出)")
        key = (shard["shard_index"], shard["shard_count"])
        if key in seen_shards:
            raise SystemExit(f"{path} 分片重复: {key}")
        seen_shards.add(key)
        if base is None:
            base = {k: shard[k] for k in
                    ("schema_version", "arch", "scorer",
                     "policy_checkpoint_sha256", "beam")}
        for b, eps in shard["raw"]["per_b"].items():
            for ep, vals in eps.items():
                per_b[b][ep].extend(vals)
        for b, eps in shard["raw"]["per_b_oracle"].items():
            for ep, vals in eps.items():
                per_b_oracle[b][ep].extend(vals)
        for b, dist in shard["raw"]["k_dist"].items():
            for k, n in dist.items():
                k_dist[b][k] += int(n)
        for k, n in shard.get("skipped", {}).items():
            skipped[k] += int(n)
        fresh += int(shard.get("fresh_true_scores", 0))

    counts = sorted(sc for _, sc in seen_shards)
    if len(set(counts)) != 1 or len(seen_shards) != counts[0]:
        raise SystemExit(f"分片不齐: 收到 {sorted(seen_shards)}")

    result = dict(base or {})
    result["merged_from_shards"] = len(seen_shards)
    result["skipped"] = dict(sorted(skipped.items()))
    result["fresh_true_scores"] = fresh
    for b in sorted(per_b, key=int):
        result[f"b{b}"] = {
            "selector_minus_recent": cluster_ci(
                per_b[b], iterations=args.bootstrap, seed=args.seed + int(b)),
            "oracle_gap": cluster_ci(
                per_b_oracle[b], iterations=args.bootstrap,
                seed=args.seed + 50 + int(b)),
            "k_distribution": {k: k_dist[b][k]
                               for k in sorted(k_dist[b], key=int)},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
