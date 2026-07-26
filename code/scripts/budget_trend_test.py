"""Paired test of whether the selection effect really decreases with budget.

# note (luojiaxuan): "d1 > d2 > d4" 目前只是三个点估计的排序。三个各自的 CI 重叠得
# 厉害,而"两个区间重叠"既不能证明也不能否定"差值显著" —— 必须直接对**配对差**
# 做 bootstrap。同一组在不同预算下的分数是相关的(同一条轨迹、同一个目标动作),
# 配对能消掉这部分方差,比拿两个独立 CI 目测强得多。
#
# 口径限制,如实记录:d_B 定义为"全部 C(n,B) 子集池"上的 selection_over_poolmean。
# B=1 的池是全部单元素集合(Gate 3 depth-1 穷举),B=2 的池是全部 C(n,2)(本次审计
# 的 b2_all_pairs 穷举),两者口径一致,d1-d2 是干净的配对比较。
# B=4 **没有**穷举池(C(28,4)=20475,没跑),缓存里是 beam 池 + 补充族的并集,
# 池的组成与 B=1/2 不同源,所以 d2-d4 混了"预算效应"与"池定义差异"。该项标注为
# NOT_POOL_MATCHED,不能用来支撑"随预算显著下降"的措辞。
"""

from __future__ import annotations

import collections
import glob
import itertools
import json
import random
import sys

BOOT = 10000
SEED = 20260727
REPORT, CACHE_GLOB = sys.argv[1], sys.argv[2]


def fold_mean(entry, fold):
    return entry[f"sum_{fold}"] / entry[f"n_{fold}"]


def selection_over_poolmean(pool, anchor):
    """cross-fit 增益 - 池均值差,两折方向取平均。"""
    if len(pool) < 2:
        return None
    gains = []
    for select, evaluate in (("even", "odd"), ("odd", "even")):
        chosen = max(pool, key=lambda s: fold_mean(pool[s], select))
        gains.append(fold_mean(pool[chosen], evaluate) - fold_mean(anchor, evaluate))
    crossfit = sum(gains) / len(gains)
    means = [e["mean"] for e in pool.values()]
    return crossfit - (sum(means) / len(means) - anchor["mean"])


def cluster_bootstrap(pairs):
    by_ep = collections.defaultdict(list)
    for ep, v in pairs:
        by_ep[ep].append(v)
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
    return point, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1], len(pairs)


FMT = "official_style_sparse_multiturn"
report = json.load(open(REPORT, encoding="utf-8"))
cur = {g: int(g.split(":")[1]) for g in report["per_group"]}
episode = {
    g: report["per_group"][g][FMT]["diagnostics"].get("episode", g.split(":")[0])
    for g in report["per_group"]
}

scores = collections.defaultdict(dict)
for path in sorted(glob.glob(CACHE_GLOB)):
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        key = rec.get("cache_key")
        if not key:
            continue
        g, fmt, raw = key.split("|")
        if fmt != FMT:
            continue
        steps = () if raw in ("empty", "") else tuple(int(x) for x in raw.split("-"))
        scores[g][steps] = rec["value"]

per_budget = {}
coverage = {}
for budget in (1, 2, 4):
    vals = {}
    covered = []
    for g, c in cur.items():
        pool_all = scores.get(g, {})
        anchor = pool_all.get(tuple(range(c - budget, c)))
        if anchor is None:
            continue
        # note (luojiaxuan): 池必须是**同一个关于历史的函数**在各预算上求值,否则
        # 比的是池定义而不是预算。"全部 C(n,B)" 是这样一个函数,但 B=4 的穷举要
        # 15 万次前向。改用 k=1 族 —— Recent-(B-1) + 一张旧图 —— 它在每个预算下
        # 都是穷举的,而且正好是标签生成实际使用的那个族。
        kept = list(range(c - budget + 1, c))  # 保留最新的 B-1 张
        olds = [j for j in range(1, c - budget)]
        pool = {}
        for old in olds:
            steps = tuple(sorted(kept + [old]))
            if steps in pool_all:
                pool[steps] = pool_all[steps]
        if len(pool) < 2:
            continue
        covered.append(len(pool) / len(olds) if olds else 0.0)
        value = selection_over_poolmean(pool, anchor)
        if value is not None:
            vals[g] = value
    per_budget[budget] = vals
    coverage[budget] = sum(covered) / len(covered) if covered else None

print("池 = Recent-(B-1) + 一张旧图(k=1 族),各预算同一构造规则")
print("覆盖率 = 实际打分的旧图数 / 全部旧图数:")
for budget in (1, 2, 4):
    cov = coverage[budget]
    print(f"  B={budget}: n_groups={len(per_budget[budget]):3d}  pool_coverage="
          + ("%.1f%%" % (100 * cov) if cov is not None else "n/a"))

print("\n各预算的 selection_over_poolmean:")
for budget in (1, 2, 4):
    pairs = [(episode[g], v) for g, v in per_budget[budget].items()]
    pt, lo, hi, n = cluster_bootstrap(pairs)
    star = "*" if (lo > 0 or hi < 0) else " "
    print("  B=%d  %+.5f [%+.5f, %+.5f] n=%d %s" % (budget, pt, lo, hi, n, star))

print("\n配对差(同一组在两个预算下相减,消掉组间方差):")
for a, b, note in ((1, 2, "口径一致"), (2, 4, "口径一致"), (1, 4, "口径一致")):
    common = sorted(set(per_budget[a]) & set(per_budget[b]))
    pairs = [(episode[g], per_budget[a][g] - per_budget[b][g]) for g in common]
    if not pairs:
        continue
    pt, lo, hi, n = cluster_bootstrap(pairs)
    star = "*" if (lo > 0 or hi < 0) else " "
    print("  d%d - d%d  %+.5f [%+.5f, %+.5f] n=%d %s   %s"
          % (a, b, pt, lo, hi, n, star, note))
