"""Does the single-replacement selection effect grow with trajectory length?

# note (luojiaxuan): 预算维度上四个配对差全部跨零,所以"B 越大越有用"没有证据。
# 但预算不是唯一的轴 —— 更自然的假设是**轨迹已经走了多远**:历史越长,Recent-B
# 覆盖的比例越小,落在窗口外的证据越多。n(候选旧帧数)在推理时直接可见,
# 若它真的预测 selection 效应,就能立刻做成 STOP 的先验特征,不必等 selector 学。
#
# 分层用**分位数**而不是固定阈值:固定阈值会让各层组数悬殊,CI 宽度不可比。
# 三层各 20 组,bootstrap 走 episode-cluster,与仓库其余分析同口径。
"""

import collections
import glob
import json
import random
import sys

FMT = "official_style_sparse_multiturn"
SEED = 20260727
BOOT = 10000
DRAWS = 25
POOL_CAP = None  # 由 --cap 决定:统一下采样到该池大小


def fold_mean(entry, fold):
    return entry[f"sum_{fold}"] / entry[f"n_{fold}"]


def boot(pairs):
    if len(pairs) < 3:
        return None
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
    return point, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1], len(pairs)


def selection_over_poolmean(pool, anchor):
    if len(pool) < 2:
        return None
    gains = []
    for select, evaluate in (("even", "odd"), ("odd", "even")):
        chosen = max(pool, key=lambda s: fold_mean(pool[s], select))
        gains.append(fold_mean(pool[chosen], evaluate) - fold_mean(anchor, evaluate))
    mean = sum(e["mean"] for e in pool.values()) / len(pool)
    return sum(gains) / len(gains) - (mean - anchor["mean"])


POOL_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else None
print("POOL_CAP =", POOL_CAP)
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

print("按历史长度 n 分层的 selection over pool mean(F_B = 恰好替换一张)")
for budget in (1, 2, 4, 8):
    rows = []
    for group, current in cur.items():
        recent = tuple(range(current - budget, current))
        anchor = scores[group].get(recent)
        olds = [j for j in range(1, current - budget)]
        if anchor is None or len(olds) < 2:
            continue
        pool = {}
        for dropped in recent:
            kept = [s for s in recent if s != dropped]
            for old in olds:
                steps = tuple(sorted(kept + [old]))
                if steps in scores[group]:
                    pool[steps] = scores[group][steps]
        # note (luojiaxuan): 长历史组的池子天然更大(n=13,B=4 -> 36 个集合;
        # n=22 -> 72 个)。池越大能选到好集合的机会越多,不控制就会把"候选更多"
        # 误读成"长轨迹更需要历史"。统一下采样到同一池大小,25 次抽样取平均
        # (单次下采样自带方差,预算扫描里已经吃过这个亏)。
        if POOL_CAP and len(pool) > POOL_CAP:
            keys = sorted(pool)
            rng = random.Random(SEED + current)
            vals = []
            for _ in range(DRAWS):
                sub = {k: pool[k] for k in rng.sample(keys, POOL_CAP)}
                v = selection_over_poolmean(sub, anchor)
                if v is not None:
                    vals.append(v)
            value = sum(vals) / len(vals) if vals else None
        else:
            value = selection_over_poolmean(pool, anchor)
        if value is not None:
            rows.append((current - 1, episode[group], value))
    if len(rows) < 9:
        continue
    rows.sort()
    third = len(rows) // 3
    strata = (
        ("短 n<=%d" % rows[third - 1][0], rows[:third]),
        ("中", rows[third : 2 * third]),
        ("长 n>=%d" % rows[2 * third][0], rows[2 * third :]),
    )
    print(f"\n  B={budget}")
    for name, subset in strata:
        block = boot([(e, v) for _, e, v in subset])
        if not block:
            continue
        point, low, high, n = block
        star = "*" if (low > 0 or high < 0) else " "
        print("    %-12s %+.5f [%+.5f, %+.5f] n=%d %s" % (name, point, low, high, n, star))
    # note (luojiaxuan): 三层各自的 CI 重叠不能推出"层间无差异",必须直接比长短两层。
    # 这两层是**不同的组**,所以是独立两样本差,不能配对 —— bootstrap 各自重采样后相减。
    short = [(e, v) for _, e, v in strata[0][1]]
    longs = [(e, v) for _, e, v in strata[2][1]]
    rng = random.Random(SEED)

    def resample(pairs):
        by = collections.defaultdict(list)
        for e, v in pairs:
            by[e].append(v)
        keys = sorted(by)
        vals = []
        for _ in keys:
            vals.extend(by[rng.choice(keys)])
        return sum(vals) / len(vals)

    diffs = sorted(resample(longs) - resample(short) for _ in range(BOOT))
    point = (sum(v for _, v in longs) / len(longs)) - (sum(v for _, v in short) / len(short))
    low, high = diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT) - 1]
    star = "*" if (low > 0 or high < 0) else " "
    print("    %-12s %+.5f [%+.5f, %+.5f] %s  <- 长减短(独立两样本)"
          % ("长 - 短", point, low, high, star))
