#!/usr/bin/env python3
"""HGKV 适配器效应按 k(替换强度)分层。

假设(用户诊断):HGKV 只用 **k=1**(一张远端帧替换**最老的**recent 槽位)训练,
而部署时 selector 组合 exact-B 集合,B=4 下有 26% 的状态 k=4。若适配器学到的是
一个很窄的干预形态,它的效应应当**随 k 增大而衰减**。

测法(纯离线,不需 GPU):同一批集合有两套打分
  - `selector-v4/sets`        : 挂 HGKV(checkpoint 572092c2…)
  - `selector-v4-frozen/sets` : 冻结策略(checkpoint "frozen-bypass")
相减即适配器效应 A(S) = U_hgkv(S) − U_frozen(S),按
  k = |S \\ Recent-|S||
分层统计。k=0 表示该集合恰好就是 Recent-|S|(适配器理应中性,是内部对照)。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path


def load_sets(root: str) -> dict[str, dict]:
    """key -> {dp, events, score}。同 key 多次出现取首次(与训练侧 setdefault 一致)。"""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(root + "/sets.shard*.jsonl")):
        with open(f, errors="ignore") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("kind") != "set":
                    continue
                out.setdefault(r["key"], {
                    "dp": r["dp_id"], "events": tuple(r["events"]),
                    "score": float(r["score"])})
    return out


def mean_ci(xs: list[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0}
    m = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": round(m, 5)}
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return {"n": n, "mean": round(m, 5), "se": round(se, 5),
            "ci95": [round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hgkv-sets", required=True)
    ap.add_argument("--frozen-sets", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    hg = load_sets(args.hgkv_sets)
    fz = load_sets(args.frozen_sets)
    common = sorted(set(hg) & set(fz))
    print(json.dumps({"hgkv_sets": len(hg), "frozen_sets": len(fz),
                      "common": len(common)}, ensure_ascii=False), flush=True)
    if not common:
        raise SystemExit("两套打分没有共同集合,无法比较")

    # note (luojiaxuan): Recent-|S| = 该状态候选池里最大的 |S| 个事件号。
    # 候选池从两套打分里出现过的事件并集近似(同一 dp 的所有集合覆盖了整个池)。
    pool: dict[str, set[int]] = collections.defaultdict(set)
    for k in common:
        for e in hg[k]["events"]:
            pool[hg[k]["dp"]].add(int(e))

    by_k: dict[int, list[float]] = collections.defaultdict(list)
    by_k_size: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    for key in common:
        h, f = hg[key], fz[key]
        events = set(int(e) for e in h["events"])
        size = len(events)
        recent = set(sorted(pool[h["dp"]], reverse=True)[:size])
        k = len(events - recent)
        a = h["score"] - f["score"]
        by_k[k].append(a)
        by_k_size[(size, k)].append(a)

    out = {
        "common_sets": len(common),
        "adapter_effect_by_k": {str(k): mean_ci(v) for k, v in sorted(by_k.items())},
        "adapter_effect_by_size_and_k": {
            f"B={s},k={k}": mean_ci(v)
            for (s, k), v in sorted(by_k_size.items()) if len(v) >= 20},
    }
    Path(args.output).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("=== 适配器效应 A(S)=U_hgkv−U_frozen,按 k 分层 ===")
    for k, st in out["adapter_effect_by_k"].items():
        if st.get("n", 0) >= 20:
            ci = st.get("ci95", ["-", "-"])
            print(f"  k={k}: n={st['n']:6}  mean={st['mean']:+.5f}  CI[{ci[0]:+.5f},{ci[1]:+.5f}]")
    print("=== 按集合大小 × k ===")
    for tag, st in out["adapter_effect_by_size_and_k"].items():
        ci = st.get("ci95", ["-", "-"])
        print(f"  {tag:10} n={st['n']:6}  mean={st['mean']:+.5f}  CI[{ci[0]:+.5f},{ci[1]:+.5f}]")


if __name__ == "__main__":
    main()
