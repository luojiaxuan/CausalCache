#!/usr/bin/env python3
"""按状态子集过滤 selector v4 标签(教师对照实验用)。

frozen-teacher sets 只打了 shard_count=50 的 shard 0..29(60% 状态,均匀交错)。
为使 HGKV-teacher 对照在**同一状态子集**上重训(教师类型成为唯一变量),
本脚本用与打分器完全一致的枚举(split 过滤 + manifest 顺序 + index%50)把
v4 已有 singletons/sets jsonl 过滤到同一 dp 子集。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_desktop_hgkv_corpus import _split


def subset_dp_ids(manifest: Path, seed: int, splits: set[str],
                  shard_count: int, kept_shards: set[int]) -> set[str]:
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    states = [r for r in records if _split(str(r["task_id"]), seed=seed) in splits]
    return {
        str(state["dp_id"])
        for index, state in enumerate(states)
        if index % shard_count in kept_shards
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening-manifest", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--shard-count", type=int, default=50)
    ap.add_argument("--kept-shards", type=int, nargs="+",
                    default=list(range(30)))
    ap.add_argument("--input", type=Path, nargs="+", required=True,
                    help="待过滤 jsonl(singletons/sets 均可,按 dp_id 字段)")
    ap.add_argument("--output", type=Path, required=True,
                    help="输出 jsonl(合并所有输入,保留 __fingerprint__ 首行一次)")
    args = ap.parse_args()

    keep = subset_dp_ids(args.screening_manifest, args.seed, set(args.splits),
                         args.shard_count, set(args.kept_shards))
    kept_rows, dropped, fingerprinted = 0, 0, False
    with args.output.open("w", encoding="utf-8") as out:
        for path in args.input:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("key") == "__fingerprint__":
                    if not fingerprinted:
                        row["fingerprint"]["state_subset"] = {
                            "shard_count": args.shard_count,
                            "kept_shards": sorted(set(args.kept_shards)),
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fingerprinted = True
                    continue
                if str(row.get("dp_id")) in keep:
                    out.write(line + "\n")
                    kept_rows += 1
                else:
                    dropped += 1
    print(json.dumps({"kept_dp": len(keep), "kept_rows": kept_rows,
                      "dropped_rows": dropped, "output": str(args.output)}))


if __name__ == "__main__":
    main()
