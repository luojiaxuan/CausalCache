"""Break the prompt-format probe down by history budget K.

# note (luojiaxuan): sparse_gain_multi = +0.0027 [CI 跨零] 是**全 K 混合的均值**。
# K=1 的"稀疏"只是一张更老的图,几乎没有选择自由度;真正的选择空间在 K=3/4。
# 若收益集中在高 K 而被 K=1 稀释,"冻结选点收益消失"就下早了。
# CI 用 episode-cluster bootstrap,与主表口径一致。
"""

import collections
import glob
import json
import random
import sys

CACHE_GLOB, SAMPLES = sys.argv[1], sys.argv[2]
BOOT, SEED = 4000, 20260726

meta = {}
for line in open(SAMPLES, encoding="utf-8"):
    row = json.loads(line)
    if row.get("split") != "heldout":
        continue
    meta.setdefault(row["pair_group"], (int(row["budget"]), row["episode"]))

scores = {}
for path in sorted(glob.glob(CACHE_GLOB)):
    for line in open(path, encoding="utf-8"):
        o = json.loads(line)
        key = o.get("cache_key")
        if not key or "score" not in o:
            continue
        parts = key.split("|")          # <adapter>|<pair_group>|<arm>|<mode>
        if len(parts) < 4:
            continue
        scores[(parts[1], parts[2])] = float(o["score"])

arms = sorted({a for _, a in scores})
groups = sorted({g for g, _ in scores if g in meta})
print("arms:", arms)
print("groups:", len(groups))
kdist = collections.Counter(meta[g][0] for g in groups)
print("K distribution:", dict(sorted(kdist.items())))


def boot(pairs):
    by_ep = collections.defaultdict(list)
    for ep, d in pairs:
        by_ep[ep].append(d)
    eps = sorted(by_ep)
    point = sum(d for _, d in pairs) / len(pairs)
    rng = random.Random(SEED)
    draws = []
    for _ in range(BOOT):
        vals = []
        for _ in eps:
            vals.extend(by_ep[rng.choice(eps)])
        draws.append(sum(vals) / len(vals))
    draws.sort()
    return point, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]


for name, a, b in [
    ("format_cost_multi   (R_multi - N0)", "R_multi", "N0"),
    ("sparse_gain_multi   (S_multi - R_multi)", "S_multi", "R_multi"),
    ("sparse_gain_single  (S_single - R_single)", "S_single", "R_single"),
    ("net_effect_multi    (S_multi - N0)", "S_multi", "N0"),
]:
    if a not in arms or b not in arms:
        print(f"\n{name}: missing arm")
        continue
    print(f"\n=== {name} ===")
    for k in (0, 1, 2, 3, 4):
        sel = [(meta[g][1], scores[(g, a)] - scores[(g, b)])
               for g in groups
               if (g, a) in scores and (g, b) in scores and (k == 0 or meta[g][0] == k)]
        if len(sel) < 5:
            continue
        pt, lo, hi = boot(sel)
        star = "  *" if (lo > 0 or hi < 0) else ""
        print("  %-4s n=%-4d %+.5f  [%+.5f, %+.5f]%s"
              % ("ALL" if k == 0 else f"K={k}", len(sel), pt, lo, hi, star))
