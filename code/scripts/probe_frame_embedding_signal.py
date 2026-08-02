#!/usr/bin/env python3
"""帧 embedding 里到底有没有"这张旧图对现在有没有用"的信号(纯 CPU 判据)。

# note (luojiaxuan): 写这个脚本是因为方案 3 的第一版训练出来 la_visual 与
# la_cheap 几乎逐位相同(dev_top1_regret 0.0977 vs 0.0986),而且**两臂都欠拟合**
# (train_probe 与 dev 的 top1-regret 只差 0.01)。欠拟合 + 加特征无变化,
# 通常意味着"特征里没有可学的东西",而不是"模型不够大"。继续调超参之前,
# 先用无需训练的相关性把这件事判掉——调参可以烧几小时,这个判据只要几分钟。
#
# 测什么:对每个决策点,候选帧 j 的真值边际 ΔU(j) = U({j}) − U(∅)(单例表),
# 与几个**零训练**的 embedding 相似度打分做**组内** Spearman:
#   cos_cq   cos(e_j, e_now)       —— "这张旧图像不像现在这一屏"
#   negdist  −‖e_j − e_now‖        —— 同上的距离版
#   cos_cmean cos(e_j, 该状态候选均值) —— "这张图有多不典型"
# 参照系:
#   age      −(current_step − j)   —— 越新越好,即 Recent 基线的排序
#   random   打乱的 ΔU             —— 组内 Spearman 的零假设带宽
#
# 判读:若 cos_cq 的组内 Spearman 在 age 基线附近、且落在 random 的噪声带内,
# 则均值池化的帧 embedding **不携带**该信号,再复杂的交互块也无从提取——
# 那时该换的是表示(patch 级 / 与指令交叉),不是架构。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--singletons-root", type=Path, required=True)
    p.add_argument("--screening-manifest", type=Path, required=True)
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-candidates", type=int, default=3,
                   help="组内 Spearman 至少要这么多候选才算,少了噪声太大")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0

    def rank(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        # 平级取平均秩,否则大量并列的 age 会被随意打散
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else 0.0


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    records = {}
    for line in args.screening_manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[str(r["dp_id"])] = r

    emb: dict[str, list[float]] = {}
    for path in sorted(args.embedding_root.glob("emb.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "frame_embedding":
                emb.setdefault(row["relpath"], row["vector"])

    b0: dict[str, dict] = {}
    singles: dict[str, dict[int, float]] = defaultdict(dict)
    for path in sorted(args.singletons_root.glob("singletons.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "b0":
                b0[row["dp_id"]] = row
            elif row.get("kind") == "singleton":
                singles[row["dp_id"]][int(row["event"])] = float(row["score"])

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def norm(a):
        return math.sqrt(dot(a, a)) or 1e-9

    scorers = ("cos_cq", "negdist", "cos_cmean", "age", "random")
    rhos: dict[str, list[float]] = {k: [] for k in scorers}
    top1_hit: dict[str, int] = {k: 0 for k in scorers}
    groups = 0
    skipped_no_emb = 0

    for dp, base in b0.items():
        rec = records.get(dp)
        if rec is None:
            continue
        have = singles.get(dp, {})
        rel = rec["image_relpaths"]
        cur_i = int(rec["step"]) - 1
        if not 0 <= cur_i < len(rel) or rel[cur_i] not in emb:
            skipped_no_emb += 1
            continue
        eq = emb[rel[cur_i]]
        nq = norm(eq)
        base_u = float(base["score"])

        cands, truths, vecs = [], [], []
        for ev in base["candidate_pool"]:
            ev = int(ev)
            if ev not in have or not 0 <= ev < len(rel):
                continue
            v = emb.get(rel[ev])
            if v is None:
                continue
            cands.append(ev)
            truths.append(have[ev] - base_u)
            vecs.append(v)
        if len(cands) < args.min_candidates:
            continue
        groups += 1

        mean_v = [sum(col) / len(vecs) for col in zip(*vecs)]
        nm = norm(mean_v)
        cur_step = int(rec["step"])
        scores = {
            "cos_cq": [dot(v, eq) / (norm(v) * nq) for v in vecs],
            "negdist": [
                -math.sqrt(sum((a - b) ** 2 for a, b in zip(v, eq))) for v in vecs
            ],
            "cos_cmean": [dot(v, mean_v) / (norm(v) * nm) for v in vecs],
            "age": [-(cur_step - ev) for ev in cands],
            "random": None,
        }
        shuffled = list(truths)
        rng.shuffle(shuffled)
        scores["random"] = shuffled

        best_truth = max(truths)
        for name, s in scores.items():
            rhos[name].append(spearman(s, truths))
            pick = max(range(len(s)), key=lambda i: s[i])
            if truths[pick] >= best_truth - 1e-12:
                top1_hit[name] += 1

    def mean(v):
        return sum(v) / len(v) if v else 0.0

    def ci95(v):
        if len(v) < 2:
            return [0.0, 0.0]
        m = mean(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
        h = 1.96 * sd / math.sqrt(len(v))
        return [round(m - h, 4), round(m + h, 4)]

    out = {
        "groups": groups,
        "skipped_no_current_embedding": skipped_no_emb,
        "min_candidates": args.min_candidates,
        "within_state_spearman": {
            k: {"mean": round(mean(v), 4), "ci95": ci95(v)} for k, v in rhos.items()
        },
        "top1_oracle_hit_pct": {
            k: round(100.0 * n / groups, 1) for k, n in top1_hit.items()
        } if groups else {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
