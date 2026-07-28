#!/usr/bin/env python3
"""Pooled budget-axis contrast: P − same-budget recent, pooled over B∈{1,2,4,8}.

# note (luojiaxuan): 预注册口径——预算等权(Δ̄ = mean_B Δ_B),每个预算内
# 任务级配对差;B=4 用 3v3 轮均值,B=1/2 的 recent 参照为冻结 dose 曲线
# (adapter 中性授权,论文披露);bootstrap 在每个预算内独立重采样任务,
# 复合出 Δ̄* 分布,percentile CI + 双侧 p。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_map(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "rows" in data:
        return {r["task"]: float(r["score"]) for r in data["rows"]}
    return {k: float(v) for k, v in data.items()}


def paired_diffs(p_maps: list[dict], r_maps: list[dict]) -> list[float]:
    tasks = set.intersection(*(set(m) for m in p_maps + r_maps))
    if len(tasks) != 117:
        raise SystemExit(f"expected the full 117-task roster, got {len(tasks)}")
    out = []
    for t in sorted(tasks):
        p = sum(m[t] for m in p_maps) / len(p_maps)
        r = sum(m[t] for m in r_maps) / len(r_maps)
        out.append(p - r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--p-b1", type=Path, required=True)
    ap.add_argument("--p-b2", type=Path, required=True)
    ap.add_argument("--p-b8", type=Path, required=True)
    ap.add_argument("--recent-b8", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    res = args.repo / "data/results"
    frozen = res / "mobileworld_frozen_gui_owl_official_b1_b2_b3_v2"
    b4 = res / "mobileworld_hgkv_selected_b4"

    diffs = {
        1: paired_diffs([load_map(args.p_b1)], [load_map(frozen / "b1-task-scores.json")]),
        2: paired_diffs([load_map(args.p_b2)], [load_map(frozen / "b2-task-scores.json")]),
        4: paired_diffs(
            [load_map(b4 / f"per_task_success_r{i}.json") for i in (1, 2, 3)],
            [load_map(b4 / f"recent_per_task_r{i}.json") for i in (1, 2, 3)],
        ),
        8: paired_diffs([load_map(args.p_b8)], [load_map(args.recent_b8)]),
    }

    per_b = {b: 100.0 * sum(d) / len(d) for b, d in diffs.items()}
    pooled = sum(per_b.values()) / len(per_b)

    rng = random.Random(args.seed)
    n = 117
    boots = []
    for _ in range(args.replicates):
        acc = 0.0
        for d in diffs.values():
            idx = [rng.randrange(n) for _ in range(n)]
            acc += 100.0 * sum(d[i] for i in idx) / n
        boots.append(acc / len(diffs))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots)) - 1]
    below = sum(1 for x in boots if x <= 0.0) / len(boots)
    p_two = 2.0 * min(below, 1.0 - below)

    out = {
        "estimand": "mean over B in {1,2,4,8} of task-paired (P - same-budget recent), pp",
        "per_budget_delta_pp": {str(b): round(v, 2) for b, v in per_b.items()},
        "pooled_delta_pp": round(pooled, 2),
        "ci95_pp": [round(lo, 2), round(hi, 2)],
        "two_sided_p": round(p_two, 4),
        "replicates": args.replicates,
        "seed": args.seed,
        "notes": "B=4 uses 3v3 round means vs HGKV+Recent-4; B=1/2 recent reference is the frozen-policy dose curve (adapter neutrality disclosed); B=8 single rounds, HGKV+Recent-8.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
