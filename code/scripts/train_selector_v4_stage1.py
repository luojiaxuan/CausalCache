#!/usr/bin/env python3
"""Stage-1 selector 训练(cheap features → 单例边际效用回归 + 排序评测)。

# note (luojiaxuan): 主线起步版:小 MLP 回归 ΔU(j) = U({j}) − U(∅),
# episode 级 train/dev 切分沿用轨迹 hash(与语料/标签同一 split 函数)。
# 评测按部署语义报排序质量而不只是 MSE:
#   * per-state Spearman(预测 vs 真 ΔU);
#   * top1_regret = ΔU(真最优) − ΔU(预测 argmax)(理想 0);
#   * top1_hit = 预测 argmax 恰为真最优的比例;
#   * stop_auc:状态级 "max ΔU > 0" 的判别(max 预测值作分数)——STOP 头的
#     cheap 版;阈值在 dev 上校准后另报 accuracy。
# 模型/标准化参数/特征 schema/标签指纹一并存盘,推理端 fail-closed 校验。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from causalcache.selector_v4_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    candidate_features,
)
from scripts.build_desktop_hgkv_corpus import _split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--singletons-root", type=Path, required=True)
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rank-weight", type=float, default=1.0,
                        help="state 内 pairwise hinge 的权重(0 = 纯回归)")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        for r, i in enumerate(order):
            ranks[i] = float(r)
        return ranks
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    if n < 2:
        return 0.0
    mean = (n - 1) / 2
    cov = sum((a - mean) * (b - mean) for a, b in zip(rx, ry))
    var = sum((a - mean) ** 2 for a in rx)
    return cov / var if var else 0.0


def main() -> None:
    args = parse_args()
    import torch

    records = {
        str(json.loads(line)["dp_id"]): json.loads(line)
        for line in args.screening_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    b0_rows: dict[str, dict] = {}
    singles: dict[str, dict[int, float]] = defaultdict(dict)
    fingerprint = None
    for path in sorted(args.singletons_root.glob("singletons.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("key") == "__fingerprint__":
                fingerprint = row.get("fingerprint")
            elif row.get("kind") == "b0":
                b0_rows[row["dp_id"]] = row
            elif row.get("kind") == "singleton":
                singles[row["dp_id"]][int(row["event"])] = float(row["score"])

    states = []
    for dp, base in b0_rows.items():
        record = records.get(dp)
        if record is None:
            continue
        pool = [int(e) for e in base["candidate_pool"]]
        have = singles.get(dp, {})
        if not pool or any(e not in have for e in pool):
            continue
        split = _split(str(record["task_id"]), seed=args.seed)
        if split == "test":
            continue
        states.append((dp, record, base, pool, have, split))

    feature_rows: dict[str, list[tuple[int, list[float], float]]] = {}
    for dp, record, base, pool, have, split in states:
        rows = []
        for event in pool:
            feats = candidate_features(
                record,
                candidate_pool=pool,
                duplicates=base.get("duplicates", {}),
                event=event,
            )
            rows.append((event, feats, have[event] - base["score"]))
        feature_rows[dp] = rows

    train_states = [s for s in states if s[5] == "train"]
    dev_states = [s for s in states if s[5] == "dev"]
    print(json.dumps({
        "states_total": len(states), "train_states": len(train_states),
        "dev_states": len(dev_states),
        "train_rows": sum(len(feature_rows[s[0]]) for s in train_states),
        "dev_rows": sum(len(feature_rows[s[0]]) for s in dev_states),
    }))

    x_train = [f for s in train_states for (_e, f, _y) in feature_rows[s[0]]]
    y_train = [y for s in train_states for (_e, _f, y) in feature_rows[s[0]]]
    dim = len(FEATURE_NAMES)
    mean = [sum(col) / len(col) for col in zip(*x_train)]
    std = [
        max(math.sqrt(sum((v - m) ** 2 for v in col) / len(col)), 1e-6)
        for col, m in zip(zip(*x_train), mean)
    ]

    def norm(feats):
        return [(v - m) / s for v, m, s in zip(feats, mean, std)]

    torch.manual_seed(args.seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, args.hidden), torch.nn.GELU(),
        torch.nn.Linear(args.hidden, args.hidden), torch.nn.GELU(),
        torch.nn.Linear(args.hidden, 1),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    xt = torch.tensor([norm(f) for f in x_train], dtype=torch.float32,
                      device=args.device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=args.device)
    # state 内 pairwise 索引(rank hinge 用)
    pair_left, pair_right = [], []
    offset = 0
    for s in train_states:
        rows = feature_rows[s[0]]
        for i in range(len(rows)):
            for j in range(len(rows)):
                if rows[i][2] > rows[j][2] + 1e-6:
                    pair_left.append(offset + i)
                    pair_right.append(offset + j)
        offset += len(rows)
    pl = torch.tensor(pair_left, dtype=torch.long, device=args.device)
    pr = torch.tensor(pair_right, dtype=torch.long, device=args.device)
    rng = random.Random(args.seed)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(xt.shape[0], device=args.device)
        total = 0.0
        for start in range(0, xt.shape[0], args.batch_size):
            idx = perm[start:start + args.batch_size]
            pred = model(xt[idx]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, yt[idx])
            if args.rank_weight and len(pair_left):
                take = torch.randint(0, len(pair_left), (args.batch_size,),
                                     device=args.device)
                left, right = pl[take], pr[take]
                margin = 0.005
                full = model(xt).squeeze(-1)
                rank_loss = torch.relu(
                    margin - (full[left] - full[right])
                ).mean()
                loss = loss + args.rank_weight * rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        if (epoch + 1) % 10 == 0:
            print(json.dumps({"epoch": epoch + 1, "loss": total}))

    def evaluate(subset):
        model.eval()
        spearmans, regrets, hits = [], [], []
        stop_scores, stop_labels = [], []
        with torch.no_grad():
            for s in subset:
                rows = feature_rows[s[0]]
                feats = torch.tensor([norm(f) for (_e, f, _y) in rows],
                                     dtype=torch.float32, device=args.device)
                preds = model(feats).squeeze(-1).tolist()
                if isinstance(preds, float):
                    preds = [preds]
                truths = [y for (_e, _f, y) in rows]
                if len(rows) >= 2:
                    spearmans.append(spearman(preds, truths))
                best_true = max(truths)
                chosen = truths[max(range(len(preds)), key=lambda i: preds[i])]
                regrets.append(best_true - chosen)
                hits.append(float(chosen >= best_true - 1e-9))
                stop_scores.append(max(preds))
                stop_labels.append(float(best_true > 0))
        pairs = [
            (a_pos > a_neg)
            for a_pos, l_pos in zip(stop_scores, stop_labels) if l_pos == 1.0
            for a_neg, l_neg in zip(stop_scores, stop_labels) if l_neg == 0.0
            for a_pos in [a_pos]
        ]
        auc = sum(pairs) / len(pairs) if pairs else None
        return {
            "n_states": len(subset),
            "spearman_mean": sum(spearmans) / len(spearmans) if spearmans else None,
            "top1_regret_mean": sum(regrets) / len(regrets) if regrets else None,
            "top1_hit_rate": sum(hits) / len(hits) if hits else None,
            "stop_auc": auc,
            "positive_state_share": (
                sum(stop_labels) / len(stop_labels) if stop_labels else None
            ),
        }

    report = {
        "schema_version": "causalcache.selector_v4_stage1_report.v1",
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "label_fingerprint": fingerprint,
        "train": evaluate(train_states),
        "dev": evaluate(dev_states),
        "hyper": {
            "epochs": args.epochs, "hidden": args.hidden,
            "lr": args.learning_rate, "rank_weight": args.rank_weight,
            "seed": args.seed,
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean, "std": std,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "label_fingerprint": fingerprint,
    }, args.output_root / "stage1_mlp.pt")
    (args.output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["dev"], ensure_ascii=False))


if __name__ == "__main__":
    main()
