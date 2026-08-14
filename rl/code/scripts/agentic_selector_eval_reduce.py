#!/usr/bin/env python3
"""Phase 3 验收归约:learned selector vs recent-B(闭环、配对、按 regime 分层)。

# note (luojiaxuan): 路线文档 §5 规定的验收口径 —— 同一个 memory-aware
# executor 下**只改 memory rule**,比较任务成功率。必须报的量:
# task success、paired delta、recent-success preservation(recent 成功的任务里
# selector 保住多少)、recent-failure rescue(recent 失败的任务里救回多少)、
# regression、选 recent-B 的比例、旧帧距离分布、分 regime 结果。
# **评测必须在自然混合分布上做**(训练侧的难度分层采样不得传进结论)。
用法:python3 agentic_selector_eval_reduce.py --learned rows_learned*.jsonl \
        --baseline rows_recent*.jsonl [--extra random:rows_random*.jsonl]
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


def load(paths: list[Path]) -> dict[str, dict]:
    """task_id → 该臂在该任务上的聚合(多条 rollout 取成功率与首条元信息)。"""
    by: dict[str, dict] = {}
    for p in paths:
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            e = by.setdefault(d["task_id"], {
                "regime": d.get("regime", "?"), "template_id": d.get("template_id"),
                "n": 0, "succ": 0, "moved_steps": 0, "steps": 0, "ages": []})
            e["n"] += 1
            e["succ"] += int(bool(d.get("success")))
            for s in d.get("steps", []):
                e["steps"] += 1
                cand = s.get("candidates") or []
                chosen = s.get("chosen_subset") or []
                if cand and chosen:
                    recent = sorted(cand)[-len(chosen):]
                    if sorted(chosen) != sorted(recent):
                        e["moved_steps"] += 1
                    now = max(cand) + 1 if cand else 0
                    e["ages"] += [now - c for c in chosen]
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learned", nargs="+", type=Path, required=True)
    ap.add_argument("--baseline", nargs="+", type=Path, required=True)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="name:glob 形式的附加臂,如 random:rows_random*.jsonl")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    L = load(args.learned)
    B = load(args.baseline)
    extras: dict[str, dict] = {}
    for item in args.extra:
        name, _, pat = item.partition(":")
        extras[name] = load([Path(x) for x in Path().glob(pat)] or [Path(pat)])

    common = sorted(set(L) & set(B))
    print(f"配对任务数 {len(common)}(learned {len(L)} / baseline {len(B)})")

    def rate(e: dict) -> float:
        return e["succ"] / max(e["n"], 1)

    def block(tag: str, ids: list[str]) -> dict:
        if not ids:
            return {}
        ls = [rate(L[t]) for t in ids]
        bs = [rate(B[t]) for t in ids]
        d = [a - b for a, b in zip(ls, bs)]
        # 二值化(任务级成功率 > 0.5 视作成功)用于 McNemar 与 preservation/rescue
        lb = [1 if x > 0.5 else 0 for x in ls]
        bb = [1 if x > 0.5 else 0 for x in bs]
        n01 = sum(1 for a, b in zip(lb, bb) if a and not b)
        n10 = sum(1 for a, b in zip(lb, bb) if b and not a)
        lo, hi = boot_ci([a - b for a, b in zip(lb, bb)])
        rec_ok = [i for i, b in enumerate(bb) if b]
        rec_bad = [i for i, b in enumerate(bb) if not b]
        moved = sum(L[t]["moved_steps"] for t in ids)
        steps = sum(L[t]["steps"] for t in ids)
        ages = [a for t in ids for a in L[t]["ages"]]
        out = {
            "n": len(ids),
            "learned": round(sum(ls) / len(ls), 4),
            "baseline": round(sum(bs) / len(bs), 4),
            "diff_pp": round(100 * sum(d) / len(d), 2),
            "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "W": n01, "L": n10, "WL_ratio": round(n01 / n10, 2) if n10 else None,
            "mcnemar_p": round(mcnemar_p(n01, n10), 5),
            "preservation": round(sum(lb[i] for i in rec_ok) / len(rec_ok), 4)
            if rec_ok else None,
            "rescue": round(sum(lb[i] for i in rec_bad) / len(rec_bad), 4)
            if rec_bad else None,
            "move_rate": round(moved / max(steps, 1), 4),
            "mean_selected_age": round(sum(ages) / len(ages), 2) if ages else None,
        }
        for name, E in extras.items():
            ids2 = [t for t in ids if t in E]
            if ids2:
                out[f"{name}_acc"] = round(
                    sum(rate(E[t]) for t in ids2) / len(ids2), 4)
        print(f"{tag:22s} n={out['n']:4d} learned={100*out['learned']:5.1f}% "
              f"recent={100*out['baseline']:5.1f}% diff={out['diff_pp']:+6.2f}pp "
              f"CI[{out['ci95_pp'][0]:+.2f},{out['ci95_pp'][1]:+.2f}] "
              f"W/L={out['W']}/{out['L']} p={out['mcnemar_p']} "
              f"move={100*out['move_rate']:.1f}%")
        if out["preservation"] is not None or out["rescue"] is not None:
            print(f"{'':22s}   preservation={out['preservation']} "
                  f"rescue={out['rescue']} 平均选中帧龄={out['mean_selected_age']}")
        return out

    hard = {"one_old_frame", "two_frame_complementary", "distractor_heavy"}
    res = {"ALL": block("总体(自然分布)", common)}
    res["hard"] = block("需老帧合并", [t for t in common if L[t]["regime"] in hard])
    res["easy"] = block("不需老帧合并",
                        [t for t in common if L[t]["regime"] not in hard])
    print()
    for reg in sorted({L[t]["regime"] for t in common}):
        res[reg] = block(reg, [t for t in common if L[t]["regime"] == reg])
    if args.json_out:
        args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("\n写出", args.json_out)


if __name__ == "__main__":
    main()
