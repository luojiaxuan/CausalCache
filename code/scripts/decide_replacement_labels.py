"""Turn frozen-policy replacement utilities into training label classes.

# note (luojiaxuan): 打分与判定分开是刻意的 —— 判定只用 CPU 读 labels.jsonl,调阈值
# 不必重跑 GPU。审计已把 k 的上界定在 1(marginal_k2_over_k1 点估计为负),所以
# "最小充分 k"退化成一个二选一:U_1 够大就 k=1,否则 k=0(Recent-sufficient)。
#
# 两道闸,缺一不可:
#   1. delta_recent —— 收益要够大。只看符号会把一堆 +0.001 的噪声收成正例。
#   2. both_folds_positive —— A 折选、B 折评为正,**且**反过来也为正。单向为正
#      在 60 组审计里只有约三成能双向成立,那七成如果混进正例,训练信号会被
#      "在某一折上碰巧好看"的集合稀释。
#
# 不硬造正例:达不到就标 Recent-sufficient。审计测得 selection_over_poolmean 在
# B=4 已归零、Gate 2 测得 C-case 只覆盖 14.6% 轨迹 —— 多数决策点的 Recent 本来
# 就够用,把它们如实标成 k=0 才是对的,那正是 selector 的 STOP 要学的东西。
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path


def load_labels(patterns: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    key = record["pair_group"]
                    if key in seen:
                        raise SystemExit(
                            f"pair-group {key!r} appears in two shards; sharding is "
                            "supposed to partition, so this means a stale file is "
                            "being globbed alongside a fresh one"
                        )
                    seen.add(key)
                    rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-glob", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta-recent", type=float, default=0.01)
    parser.add_argument(
        "--require-both-folds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="demand that both cross-fit directions be positive (default: yes)",
    )
    args = parser.parse_args()

    rows = load_labels(args.labels_glob)
    if not rows:
        raise SystemExit("no label rows matched")

    decided: list[dict] = []
    stats: collections.Counter = collections.Counter()
    per_budget: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    gains: dict[str, list[float]] = collections.defaultdict(list)

    for record in rows:
        out = {
            "pair_group": record["pair_group"],
            "episode": record["episode"],
            "current_step": record["current_step"],
            "format": record["format"],
            "budgets": {},
        }
        for budget, block in sorted(record["budgets"].items()):
            k1 = block.get("k1")
            recent = block["recent_steps"]
            if not k1:
                decision = {
                    "k": 0,
                    "klass": "recent_sufficient",
                    "reason": "no old frame available",
                    "steps": recent,
                    "gain": 0.0,
                }
            else:
                gain = k1["crossfit_gain"]
                stable = k1["both_folds_positive"] or not args.require_both_folds
                big = gain > args.delta_recent
                if big and stable:
                    # note (luojiaxuan): 用 cross-fit 两折各自选出的集合可能不同。
                    # 训练要一个确定的集合,取 in-sample 最优 —— 它是"这一组到底该
                    # 恢复哪一张"的最佳单点估计,而 gain 用 cross-fit 值(无偏)。
                    decision = {
                        "k": 1,
                        "klass": "utility_positive",
                        "steps": k1["in_sample_best_steps"],
                        "gain": gain,
                        "pool_size": k1["pool_size"],
                    }
                else:
                    decision = {
                        "k": 0,
                        "klass": "recent_sufficient",
                        "reason": (
                            "gain below delta_recent" if not big else "folds disagree"
                        ),
                        "steps": recent,
                        "gain": gain,
                    }
                gains[budget].append(gain)
            out["budgets"][budget] = decision
            stats[decision["klass"]] += 1
            per_budget[budget][decision["klass"]] += 1
        decided.append(out)

    total = sum(stats.values())
    summary = {
        "schema_version": "causalcache.replacement_label_decision.v1",
        "delta_recent": args.delta_recent,
        "require_both_folds": args.require_both_folds,
        "groups": len(rows),
        "episodes": len({row["episode"] for row in rows}),
        "decisions": total,
        "class_counts": dict(stats),
        "class_fractions": {k: v / total for k, v in stats.items()},
        "per_budget": {b: dict(c) for b, c in sorted(per_budget.items())},
    }
    for budget, values in sorted(gains.items()):
        values = sorted(values)
        summary.setdefault("gain_quantiles", {})[budget] = {
            "n": len(values),
            "p10": values[int(0.10 * len(values))],
            "p50": values[len(values) // 2],
            "p90": values[int(0.90 * len(values))],
            "max": values[-1],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"summary": summary}, ensure_ascii=False, sort_keys=True) + "\n")
        for row in decided:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
