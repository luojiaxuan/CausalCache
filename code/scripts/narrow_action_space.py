"""Can the single-replacement selector drop the "which recent slot" decision?

# note (luojiaxuan): cross-fit 淘汰统计显示 oracle 最常淘汰最新那张 recent
# (B=4 时 46% vs 均匀 25%,B=8 时 61% vs 12.5%)。若把动作空间收窄成"固定淘汰
# slot B-1,只选 old",selector 的动作空间从 B*n 降到 n,标签生成成本也降 B 倍。
# 但"最常"不等于"总是" —— 收窄的代价必须直接测,不能从淘汰分布外推。
"""

import collections
import glob
import json
import random
import sys

FMT = "official_style_sparse_multiturn"
SEED = 20260727
BOOT = 10000


def fold_mean(entry, fold):
    return entry[f"sum_{fold}"] / entry[f"n_{fold}"]


def boot(pairs):
    by = collections.defaultdict(list)
    for episode, value in pairs:
        by[episode].append(value)
    keys = sorted(by)
    point = sum(v for _, v in pairs) / len(pairs)
    rng = random.Random(SEED)
    draws = []
    for _ in range(BOOT):
        vals = []
        for _ in keys:
            vals.extend(by[rng.choice(keys)])
        draws.append(sum(vals) / len(vals))
    draws.sort()
    return point, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]


def selection_over_poolmean(pool, anchor):
    if len(pool) < 2:
        return None
    gains = []
    for select, evaluate in (("even", "odd"), ("odd", "even")):
        chosen = max(pool, key=lambda s: fold_mean(pool[s], select))
        gains.append(fold_mean(pool[chosen], evaluate) - fold_mean(anchor, evaluate))
    mean = sum(e["mean"] for e in pool.values()) / len(pool)
    return sum(gains) / len(gains) - (mean - anchor["mean"])


report = json.load(open(sys.argv[1], encoding="utf-8"))
cur = {g: int(g.split(":")[1]) for g in report["per_group"]}
episode = {
    g: report["per_group"][g][FMT]["diagnostics"].get("episode", g.split(":")[0])
    for g in report["per_group"]
}
scores = collections.defaultdict(dict)
for path in sorted(glob.glob(sys.argv[2])):
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        key = rec.get("cache_key")
        if not key:
            continue
        group, entry_format, raw = key.split("|")
        if entry_format != FMT:
            continue
        steps = () if raw in ("empty", "") else tuple(int(x) for x in raw.split("-"))
        scores[group][steps] = rec["value"]

print("动作空间收窄的代价(selection over pool mean;池已按各自族重算)")
for budget in (2, 3, 4, 8):
    rows = {"full": [], "last": [], "oldest": []}
    for group, current in cur.items():
        recent = tuple(range(current - budget, current))
        anchor = scores[group].get(recent)
        olds = [j for j in range(1, current - budget)]
        if anchor is None or len(olds) < 2:
            continue
        pool = {}
        by_slot = collections.defaultdict(dict)
        for index, dropped in enumerate(recent):
            kept = [s for s in recent if s != dropped]
            for old in olds:
                steps = tuple(sorted(kept + [old]))
                if steps in scores[group]:
                    pool[steps] = scores[group][steps]
                    by_slot[index][steps] = scores[group][steps]
        values = {
            "full": selection_over_poolmean(pool, anchor),
            "last": selection_over_poolmean(by_slot.get(budget - 1, {}), anchor),
            "oldest": selection_over_poolmean(by_slot.get(0, {}), anchor),
        }
        if any(v is None for v in values.values()):
            continue
        for name, value in values.items():
            rows[name].append((episode[group], value))
    if not rows["full"]:
        continue
    label = {"full": "全部槽位", "last": "只淘汰最新", "oldest": "只淘汰最老"}
    for name in ("full", "last", "oldest"):
        point, low, high = boot(rows[name])
        star = "*" if (low > 0 or high < 0) else " "
        print("  B=%-2d %-12s %+.5f [%+.5f, %+.5f] n=%d %s"
              % (budget, label[name], point, low, high, len(rows[name]), star))
    diff = [(e, f - l) for (e, f), (_, l) in zip(rows["full"], rows["last"])]
    point, low, high = boot(diff)
    star = "*" if (low > 0 or high < 0) else " "
    print("  B=%-2d %-12s %+.5f [%+.5f, %+.5f] %s   <- 收窄若无代价则应跨零"
          % (budget, "收窄代价", point, low, high, star))
