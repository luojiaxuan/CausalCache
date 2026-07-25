#!/usr/bin/env python3
"""Select long-horizon GUI-Odyssey trajectories for the sparse-history V3 corpus.

# note (luojiaxuan): 现有 v1 渲染池只冻结了 1,200 条轨迹,长历史严重不足
# (>=24 事件仅 155 条、>=32 仅 29 条),且每条只取最后一个决策,导致 >=24 历史的
# 目标动作 97% 是 click。本脚本从 8,334 条原始注释里按真实步数分层挑选成功轨迹,
# 排除已渲染的,输出待渲染清单(含所在 pool 分片与行号),供后续 materialize 使用。
"""
from __future__ import annotations
import argparse, json, glob
from collections import Counter, defaultdict
from pathlib import Path


def annotation_stats(path: Path) -> tuple[int, bool, Counter] | None:
    try:
        a = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    steps = a.get("steps") or []
    if not steps:
        return None
    ok = str(steps[-1].get("action", "")).upper() == "COMPLETE"
    acts = Counter(str(s.get("action", "")).upper() for s in steps)
    return len(steps), ok, acts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pool-root", type=Path, required=True)
    p.add_argument("--existing-shards", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-long", type=int, default=600, help=">=24 步的目标条数")
    p.add_argument("--target-mid", type=int, default=600, help="16-23 步的目标条数")
    args = p.parse_args()

    from pyarrow import parquet as pq

    # 1) 已渲染的轨迹 id
    existing: set[str] = set()
    for s in sorted(args.existing_shards.glob("*.parquet")):
        existing |= set(pq.read_table(s, columns=["source_id"]).column("source_id").to_pylist())
    print(json.dumps({"already_rendered": len(existing)}), flush=True)

    # 2) 注释池的长度/成功性
    stats: dict[str, tuple[int, bool, Counter]] = {}
    for f in args.annotations.glob("*.json"):
        r = annotation_stats(f)
        if r:
            stats[f.stem] = r
    dist = Counter()
    for n, ok, _ in stats.values():
        if not ok:
            continue
        dist["16-23" if 16 <= n < 24 else ">=32" if n >= 32 else ">=24" if n >= 24 else "<16"] += 1
    print(json.dumps({"annotations": len(stats), "success_by_length": dict(dist)}), flush=True)

    # 3) pool 分片索引 (id -> shard,row)
    located: dict[str, tuple[str, int]] = {}
    for s in sorted(args.pool_root.glob("*.parquet")):
        md = pq.read_table(s, columns=["metadata"]).column("metadata").to_pylist()
        for row, m in enumerate(md):
            if isinstance(m, str):
                m = json.loads(m)
            eid = ((m or {}).get("others") or {}).get("id")
            if eid:
                # note (luojiaxuan): pool 的 id 带 ``guiodyssey_`` 前缀,而注释文件名与
                # 已渲染 source_id 都是裸数字;统一去前缀后才能三方对齐。
                located[str(eid).removeprefix("guiodyssey_")] = (s.name, row)
    print(json.dumps({"pool_indexed": len(located)}), flush=True)

    # 4) 分层挑选:未渲染 + 成功 + 长
    def pick(lo: int, hi: int, want: int) -> list[dict]:
        cands = [
            (eid, n, acts) for eid, (n, ok, acts) in stats.items()
            if ok and lo <= n < hi and eid not in existing and eid in located
        ]
        cands.sort(key=lambda x: -x[1])  # 长的优先
        out = []
        for eid, n, acts in cands[:want]:
            shard, row = located[eid]
            out.append({"source_id": eid, "steps": n, "shard": shard, "row": row,
                        "action_mix": {k: v for k, v in acts.most_common(4)}})
        return out

    sel = pick(24, 10**6, args.target_long) + pick(16, 24, args.target_mid)
    args.output.write_text(json.dumps({"selected": sel}, indent=2) + "\n", encoding="utf-8")
    lens = sorted(s["steps"] for s in sel)
    amix = Counter()
    for s in sel:
        amix.update(s["action_mix"])
    print(json.dumps({
        "selected": len(sel),
        "len_median": lens[len(lens) // 2] if lens else 0,
        "len_max": lens[-1] if lens else 0,
        "ge24": sum(1 for x in lens if x >= 24),
        "ge32": sum(1 for x in lens if x >= 32),
        "ge40": sum(1 for x in lens if x >= 40),
        "action_mix": dict(amix.most_common(6)),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
