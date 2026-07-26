"""Matched single-replacement budget sweep: how does replacement value scale with B?

# note (luojiaxuan): 本脚本存在的唯一理由是**口径匹配**。此前三次比较预算给出了三种
# 互相矛盾的"趋势"(单调降 / 单调升 / 台阶),原因不是数据变了,而是每次的池定义不同:
# beam 池是精选高分集合、穷举 C(n,B) 池含大量差集合、k=1-drop-oldest 池只有一个丢弃
# 位置。selection_over_poolmean 依赖池的组成,跨池定义比大小毫无意义。
#
# 这里对每个预算用**同一个关于历史的函数**求值:
#
#     F_B = {R_B} ∪ { (R_B \\ {r_i}) ∪ {o_j} : 全部 recent 槽位 i, 全部旧帧 j }
#
# 即"保持 Recent(STOP)或恰好替换一张"。它是单替换 selector 的真实动作空间,规模
# O(B·n) 而非 C(n,B),所以 B=8 也只有约 8×20=160 个集合/组,扛得住。
#
# 不预设方向。两个相反的力同时存在:B 越大 Recent 内部越可能冗余(替换机会变多),
# 但 B 越大 Recent 覆盖也越充分(恢复旧图的必要性变小)。可能在中等 B 达峰。
"""

from __future__ import annotations

import collections
import glob
import json
import random
import sys

BOOT = 10000
SEED = 20260727
FMT = "official_style_sparse_multiturn"
BUDGETS = (1, 2, 3, 4, 8)


def fold_mean(entry, fold):
    return entry[f"sum_{fold}"] / entry[f"n_{fold}"]


def cluster_bootstrap(pairs):
    if not pairs:
        return None
    by_ep = collections.defaultdict(list)
    for ep, value in pairs:
        by_ep[ep].append(value)
    eps = sorted(by_ep)
    point = sum(v for _, v in pairs) / len(pairs)
    rng = random.Random(SEED)
    draws = []
    for _ in range(BOOT):
        vals = []
        for _ in eps:
            vals.extend(by_ep[rng.choice(eps)])
        draws.append(sum(vals) / len(vals))
    draws.sort()
    return {
        "point": point,
        "ci_low": draws[int(0.025 * BOOT)],
        "ci_high": draws[int(0.975 * BOOT) - 1],
        "n": len(pairs),
    }


def fmt(block):
    if not block:
        return "        n/a"
    star = "*" if (block["ci_low"] > 0 or block["ci_high"] < 0) else " "
    return "%+.5f [%+.5f, %+.5f] n=%d %s" % (
        block["point"], block["ci_low"], block["ci_high"], block["n"], star
    )


def main() -> None:
    report_path, cache_glob = sys.argv[1], sys.argv[2]
    report = json.load(open(report_path, encoding="utf-8"))
    cur = {g: int(g.split(":")[1]) for g in report["per_group"]}
    episode = {
        g: report["per_group"][g][FMT]["diagnostics"].get("episode", g.split(":")[0])
        for g in report["per_group"]
    }

    scores = collections.defaultdict(dict)
    for path in sorted(glob.glob(cache_glob)):
        for line in open(path, encoding="utf-8"):
            record = json.loads(line)
            key = record.get("cache_key")
            if not key:
                continue
            group, entry_format, raw = key.split("|")
            if entry_format != FMT:
                continue
            steps = () if raw in ("empty", "") else tuple(int(x) for x in raw.split("-"))
            scores[group][steps] = record["value"]

    out = {"schema_version": "causalcache.budget_replacement_sweep.v1", "budgets": {}}
    print("F_B = {R_B} ∪ {恰好替换一张};各预算同一构造规则\n")
    header = "%-4s %-6s %-38s %-38s %-38s %-38s" % (
        "B", "cover", "G_B 最佳单替换-Recent(in-sample)",
        "cross-fit gain", "selection over pool mean", "pool mean - Recent")
    print(header)
    print("-" * len(header))

    per_budget_selection = {}
    for budget in BUDGETS:
        rows = []
        evict = collections.Counter()
        covered = []
        for group, current in cur.items():
            pool_all = scores.get(group, {})
            recent = tuple(range(current - budget, current))
            anchor = pool_all.get(recent)
            olds = [j for j in range(1, current - budget)]
            if anchor is None or not olds:
                continue
            pool = {}
            slot_of = {}
            for index, dropped in enumerate(recent):
                kept = [s for s in recent if s != dropped]
                for old in olds:
                    steps = tuple(sorted(kept + [old]))
                    if steps in pool_all:
                        pool[steps] = pool_all[steps]
                        slot_of[steps] = index
            want = budget * len(olds)
            if len(pool) < 2:
                continue
            covered.append(len(pool) / want)

            in_sample = max(pool, key=lambda s: pool[s]["mean"])
            gains = []
            for select, evaluate in (("even", "odd"), ("odd", "even")):
                chosen = max(pool, key=lambda s: fold_mean(pool[s], select))
                gains.append(fold_mean(pool[chosen], evaluate) - fold_mean(anchor, evaluate))
            crossfit = sum(gains) / len(gains)
            means = [entry["mean"] for entry in pool.values()]
            gap = sum(means) / len(means) - anchor["mean"]
            # 最优淘汰槽位:0 = 最老的 recent,budget-1 = 最新的
            evict[slot_of[in_sample]] += 1
            rows.append({
                "episode": episode[group],
                "G": pool[in_sample]["mean"] - anchor["mean"],
                "crossfit": crossfit,
                "selection": crossfit - gap,
                "gap": gap,
            })

        if not rows:
            continue
        blocks = {
            name: cluster_bootstrap([(r["episode"], r[name]) for r in rows])
            for name in ("G", "crossfit", "selection", "gap")
        }
        per_budget_selection[budget] = {
            r["episode"]: r["selection"] for r in rows
        }
        cov = sum(covered) / len(covered)
        print("%-4d %-6s %-38s %-38s %-38s %-38s" % (
            budget, "%.0f%%" % (100 * cov),
            fmt(blocks["G"]), fmt(blocks["crossfit"]),
            fmt(blocks["selection"]), fmt(blocks["gap"])))
        total = sum(evict.values())
        out["budgets"][str(budget)] = {
            "pool_coverage": cov,
            "groups": len(rows),
            **{name: blocks[name] for name in blocks},
            "evicted_slot_histogram": {str(k): v for k, v in sorted(evict.items())},
            "evicted_slot_fraction": {
                str(k): v / total for k, v in sorted(evict.items())
            },
        }

    print("\n最优淘汰槽位分布(0 = 最老的 recent):")
    for budget in BUDGETS:
        block = out["budgets"].get(str(budget))
        if not block:
            continue
        frac = block["evicted_slot_fraction"]
        print("  B=%d  %s" % (budget, "  ".join(
            "slot%s=%.0f%%" % (k, 100 * v) for k, v in sorted(frac.items(), key=lambda x: int(x[0])))))

    print("\n配对差(同一 group 两预算相减;口径已匹配,可直接比较):")
    out["paired"] = {}
    keys = sorted(per_budget_selection)
    for a, b in zip(keys, keys[1:]):
        rows_a, rows_b = per_budget_selection[a], per_budget_selection[b]
        common = sorted(set(rows_a) & set(rows_b))
        pairs = [(ep, rows_a[ep] - rows_b[ep]) for ep in common]
        block = cluster_bootstrap(pairs)
        out["paired"][f"d{a}_minus_d{b}"] = block
        print("  d%-2d - d%-2d  %s" % (a, b, fmt(block)))

    with open(sys.argv[3], "w", encoding="utf-8") as handle:
        handle.write(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print("\nwrote", sys.argv[3])


if __name__ == "__main__":
    main()
