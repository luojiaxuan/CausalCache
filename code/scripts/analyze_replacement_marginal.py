"""Decide the maximum useful replacement count k from the frozen-policy score cache.

# note (luojiaxuan): 本脚本回答的是"selector 要多复杂",不是"oracle 上界有多高"。
# 判据是**第二张旧图的边际**
#     marginal = U_2 - U_1
#     U_k = Q_cf(该 k 下最好的集合) - Q_cf(R_B)
# 全部走 token 奇偶两折 cross-fit:在 A 折上选集合、在 B 折上计分,再反向一次取平均。
# 这一步不能省 —— U_k 是对一个集合族取 max,in-sample 的 max 必然 >= 0,那个数只是
# "搜索规模"的函数,不是"选择有没有可迁移信息"。
#
# 同时报告 k=1 的**丢弃位置**分布:Gate 3 的 beam 只穷举了"丢最老那张",而 §12 的
# selector 公式 Delta(j<-i|S) 允许 i 取任意 recent。若"丢最新"从来不赢,selector 的
# 动作空间可以合法收窄成"只替换较老的 recent",这是个结构性结论而不是实现简化。
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import random
from pathlib import Path

BOOT = 10000
SEED = 20260727


def fold_mean(entry: dict, fold: str) -> float:
    return entry[f"sum_{fold}"] / entry[f"n_{fold}"]


def cluster_bootstrap(pairs: list[tuple[str, float]], *, confidence: float = 0.95):
    """episode-cluster bootstrap;pairs = [(episode, value)]."""
    if not pairs:
        return None
    by_episode = collections.defaultdict(list)
    for episode, value in pairs:
        by_episode[episode].append(value)
    episodes = sorted(by_episode)
    point = sum(v for _, v in pairs) / len(pairs)
    rng = random.Random(SEED)
    draws = []
    for _ in range(BOOT):
        values: list[float] = []
        for _ in episodes:
            values.extend(by_episode[rng.choice(episodes)])
        draws.append(sum(values) / len(values))
    draws.sort()
    lo = draws[int((1 - confidence) / 2 * BOOT)]
    hi = draws[int((1 + confidence) / 2 * BOOT) - 1]
    return {"point": point, "ci_low": lo, "ci_high": hi, "n": len(pairs)}


def crossfit_best(pool: dict[tuple[int, ...], dict], anchor: dict):
    """Return (mean gain over both fold directions, per-direction gains, picks).

    在 A 折选、在 B 折评;两个方向都做,取平均。两个方向的分别值也返回,因为 §3.3
    要求"两折方向都为正"才算稳定 oracle-positive。
    """
    if not pool:
        return None
    directions = {}
    for select, evaluate in (("even", "odd"), ("odd", "even")):
        chosen = max(pool, key=lambda steps: fold_mean(pool[steps], select))
        directions[f"select_{select}"] = {
            "gain": fold_mean(pool[chosen], evaluate) - fold_mean(anchor, evaluate),
            "steps": list(chosen),
        }
    gains = [d["gain"] for d in directions.values()]
    return {"gain": sum(gains) / len(gains), "directions": directions}


def load_cache(patterns: list[str], fmt: str) -> dict[str, dict[tuple[int, ...], dict]]:
    scores: dict[str, dict[tuple[int, ...], dict]] = collections.defaultdict(dict)
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    key = record.get("cache_key")
                    if not key:
                        continue
                    group, entry_format, raw = key.split("|")
                    if entry_format != fmt:
                        continue
                    steps = (
                        ()
                        if raw in ("empty", "")
                        else tuple(int(part) for part in raw.split("-"))
                    )
                    scores[group][steps] = record["value"]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-glob", action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--format", default="official_style_sparse_multiturn")
    args = parser.parse_args()

    gate3 = json.loads(args.report.read_text(encoding="utf-8"))
    episode_of = {
        group: block[args.format]["diagnostics"].get("episode", group.split(":")[0])
        for group, block in gate3["per_group"].items()
    }
    scores = load_cache(args.cache_glob, args.format)
    budget = args.budget

    rows: list[dict] = []
    missing: collections.Counter = collections.Counter()
    for group in sorted(gate3["per_group"]):
        current = int(group.split(":")[1])
        pool = scores.get(group, {})
        recent = tuple(range(current - budget, current))
        anchor = pool.get(recent)
        if anchor is None:
            missing["anchor"] += 1
            continue
        olds = [step for step in range(1, current - budget)]
        if not olds:
            missing["no_old_frames"] += 1
            continue

        families: dict[str, dict[tuple[int, ...], dict]] = {}
        # k=1,按丢弃位置分开,便于回答"动作空间能不能收窄"
        for index, dropped in enumerate(recent):
            kept = [step for step in recent if step != dropped]
            member = {}
            for old in olds:
                steps = tuple(sorted(kept + [old]))
                if steps in pool:
                    member[steps] = pool[steps]
            if member:
                families[f"k1_drop{index}"] = member
        union_k1 = {}
        for index in range(budget):
            union_k1.update(families.get(f"k1_drop{index}", {}))
        if union_k1:
            families["k1_any_drop"] = union_k1
        # k=1 的 Gate 3 原生定义:Recent-(B-1) + 一张旧图,即 drop=最老
        families["k1_drop_oldest"] = families.get("k1_drop0", {})

        kept_tail = list(recent[budget - 2 :])
        member_k2 = {}
        for first, second in itertools.combinations(olds, 2):
            steps = tuple(sorted(kept_tail + [first, second]))
            if steps in pool:
                member_k2[steps] = pool[steps]
        if member_k2:
            families["k2_recent_tail"] = member_k2

        row = {"group": group, "episode": episode_of.get(group, group.split(":")[0])}
        row["n_old"] = len(olds)
        for name, member in families.items():
            if not member:
                continue
            row[name] = crossfit_best(member, anchor)
            row[f"{name}_pool"] = len(member)

        # note (luojiaxuan): 标签生成阶段没法在 2758 组上穷举 k=2(878k 次前向 / 两台
        # 16 卡 17 小时),所以要退回"先按 singleton 效用取 top-N 再穷举 C(N,2)"。
        # 那是个近似,而近似的代价必须**测**出来而不是假设:这里直接算 top-N 预筛
        # 对穷举最优对的召回率。若召回率高,捷径可用;若低,说明最优对里确实有
        # "两张单独都不突出"的互补组合,预筛会系统性漏掉它们。
        singleton_rank = sorted(
            olds,
            key=lambda old: -(
                pool[tuple(sorted(list(recent[1:]) + [old]))]["mean"]
                if tuple(sorted(list(recent[1:]) + [old])) in pool
                else float("-inf")
            ),
        )
        for top_n in (8, 12):
            shortlist = set(singleton_rank[:top_n])
            hits = 0
            evaluated = 0
            for block in (row.get("k2_recent_tail") or {}).get("directions", {}).values():
                chosen_olds = [s for s in block["steps"] if s in set(olds)]
                if len(chosen_olds) != 2:
                    continue
                evaluated += 1
                hits += int(all(step in shortlist for step in chosen_olds))
            if evaluated:
                row[f"top{top_n}_recall_hits"] = hits
                row[f"top{top_n}_recall_evaluated"] = evaluated
        rows.append(row)

    def summarize(name: str):
        pairs = [
            (row["episode"], row[name]["gain"]) for row in rows if row.get(name)
        ]
        return cluster_bootstrap(pairs)

    summary = {}
    for name in (
        "k1_drop_oldest",
        "k1_drop0",
        "k1_drop1",
        "k1_drop2",
        "k1_drop3",
        "k1_any_drop",
        "k2_recent_tail",
    ):
        result = summarize(name)
        if result:
            summary[f"U_{name}"] = result

    # 决策量:第二张旧图相对第一张的边际
    marginal_pairs = [
        (row["episode"], row["k2_recent_tail"]["gain"] - row["k1_any_drop"]["gain"])
        for row in rows
        if row.get("k2_recent_tail") and row.get("k1_any_drop")
    ]
    summary["marginal_k2_over_k1"] = cluster_bootstrap(marginal_pairs)

    # 动作空间:丢最老 vs 丢任意
    widen_pairs = [
        (row["episode"], row["k1_any_drop"]["gain"] - row["k1_drop_oldest"]["gain"])
        for row in rows
        if row.get("k1_any_drop") and row.get("k1_drop_oldest")
    ]
    summary["widen_drop_position"] = cluster_bootstrap(widen_pairs)

    # 稳定性:两折方向是否同号(§3.3)
    stable = collections.Counter()
    for row in rows:
        for name in ("k1_any_drop", "k2_recent_tail"):
            block = row.get(name)
            if not block:
                continue
            gains = [d["gain"] for d in block["directions"].values()]
            stable[f"{name}_both_folds_positive"] += int(all(g > 0 for g in gains))
            stable[f"{name}_evaluated"] += 1

    shortcut = {}
    for top_n in (8, 12):
        hits = sum(row.get(f"top{top_n}_recall_hits", 0) for row in rows)
        evaluated = sum(row.get(f"top{top_n}_recall_evaluated", 0) for row in rows)
        if evaluated:
            shortcut[f"top{top_n}_prefilter_recall"] = {
                "hits": hits,
                "evaluated": evaluated,
                "recall": hits / evaluated,
            }

    payload = {
        "schema_version": "causalcache.replacement_marginal.v1",
        "k2_prefilter_shortcut": shortcut,
        "format": args.format,
        "budget": budget,
        "groups": len(rows),
        "missing": dict(missing),
        "bootstrap": {"replicates": BOOT, "seed": SEED, "cluster_unit": "episode"},
        "summary": summary,
        "fold_stability": dict(stable),
        "per_group": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"groups={len(rows)} missing={dict(missing)}")
    for name, block in summary.items():
        if not block:
            continue
        star = "*" if (block["ci_low"] > 0 or block["ci_high"] < 0) else " "
        print(
            "  %-28s %+.5f [%+.5f, %+.5f] n=%d %s"
            % (name, block["point"], block["ci_low"], block["ci_high"], block["n"], star)
        )
    print(" fold stability:", dict(stable))
    for name, block in shortcut.items():
        print(
            "  %-28s recall=%.1f%% (%d/%d)"
            % (name, 100 * block["recall"], block["hits"], block["evaluated"])
        )


if __name__ == "__main__":
    main()
