#!/usr/bin/env python3
"""统一集合条件边际打分器 Δ̂(j|S) 的训练(2026-07-27 用户裁定的方案 2 主线)。

# note (luojiaxuan): 取代 train_selector_v4_stage1.py(独立 stage-1 不再训练)。
#   * 训练对 = 单例 edges(S=∅,边际 = U({j})−U(∅))∪ beam edges(S=父集合,
#     边际 = U(S∪{j})−U(S));特征 = 候选 cheap 特征 + set-context 特征;
#   * 损失 = MSE 回归 + 组内 pairwise hinge(组 = 同 (dp, S) 下竞争的候选);
#   * 训练中每 eval_every 个 epoch 跑一次 held-out(dev episode)验证:
#     边际回归 MSE、单例组 Spearman/top1-regret、以及 on-tree 端到端 replay:
#     从 beam teacher 已打分的树上按模型预测走贪心路径,终点集合的**真** U 与
#     Recent-B 锚 / 树上 oracle 对比(B∈{2,4})——趋势必须是 replay_gain 上行;
#   * 模型/标准化/两段特征 schema/标签指纹存盘,推理端 fail-closed 校验。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from causalcache.selector_v4_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    SET_FEATURE_NAMES,
    candidate_features,
    set_context_features,
)
from scripts.build_desktop_did_corpus_v2 import recent_window
from scripts.build_desktop_hgkv_corpus import _split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--singletons-root", type=Path, required=True)
    parser.add_argument("--sets-root", type=Path, required=True)
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=0.005)
    parser.add_argument(
        "--early-stop-patience", type=int, default=4,
        help="连续多少次 held-out 评测无提升即停;epochs 只是预算上限",
    )
    parser.add_argument(
        "--train-fraction", type=float, default=1.0,
        help="学习曲线诊断用:按 episode 哈希取 train 子集(dev 恒全量)。"
             "dev 指标随 fraction 上升 → 方差主导(加数据有效);"
             "平坦且 train 指标也平庸 → 偏差主导(上更高维特征)",
    )
    parser.add_argument(
        "--readout-root", type=Path, default=None,
        help="HGKV 反事实读出特征目录(readouts.shard*.jsonl,键 dp|event)。"
             "缺省 = 纯 cheap 特征(结构退化为单塔,与学习曲线口径逐字节一致)。"
             "提供时走 two-tower:readout 先做**候选池内 z 归一**(迁移稳健),"
             "经 dropout+强 weight decay 的残差塔并入,平台特异容量被约束在残差里",
    )
    parser.add_argument("--readout-dropout", type=float, default=0.3)
    parser.add_argument("--readout-weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--feature-cache", type=Path, default=None,
        help="cheap+set 特征缓存(.pt)。存在且 edge 数/维度匹配则直接加载;"
             "否则多进程计算后写入 —— 三臂共享一份,免去三次重复构建",
    )
    parser.add_argument("--feature-workers", type=int, default=96)
    parser.add_argument(
        "--intent-root", type=Path, default=None,
        help="intent 特征目录(intent.shard*.jsonl,8 维目标无关意图特征,"
             "extract_selector_v4_intent_features.py 产出)。给定即追加到特征尾部"
             "(dim 28+8+8),单遍部署路径的 v2 输入",
    )
    parser.add_argument(
        "--arch", choices=("two_tower", "concat"), default="two_tower",
        help="two_tower = score 可加(cheap + 正则残差塔,无跨组交互);"
             "concat = 单塔全交互,readout 维经 dropout 后并入首层,"
             "readout 首层权重吃同样的强 weight decay。n 小、训练分钟级,"
             "两种都跑 dev 上选,不靠先验拍板",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


# note (luojiaxuan): 多进程特征构建的 fork 上下文。pool.map 需要可 pickle 的
# 回调,闭包不行;worker 经 fork 继承本模块全局,主进程在建 Pool 前填充。
_FEAT_CTX: dict = {}


def _feat_worker(edge):
    dp, selected, event, _marginal = edge
    records = _FEAT_CTX["records"]
    b0 = _FEAT_CTX["b0"]
    dup_cache = _FEAT_CTX["dup"]
    record = records[dp]
    base = b0[dp]
    f = candidate_features(
        record,
        candidate_pool=[int(e) for e in base["candidate_pool"]],
        duplicates=dup_cache[dp],
        event=event,
    ) + set_context_features(
        record, selected=list(selected), event=event,
        duplicates=dup_cache[dp],
    )
    dims = _FEAT_CTX.get("intent_dims") or 0
    if dims:
        f = f + (_FEAT_CTX["intent"].get((dp, event)) or [0.0] * dims)
    return f


def load_jsonl_rows(root: Path, pattern: str):
    for path in sorted(root.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def spearman(xs, ys):
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        for r, i in enumerate(order):
            out[i] = float(r)
        return out
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
    split_of = {
        dp: _split(str(r["task_id"]), seed=args.seed) for dp, r in records.items()
    }

    b0: dict[str, dict] = {}
    singles: dict[str, dict[int, float]] = defaultdict(dict)
    for row in load_jsonl_rows(args.singletons_root, "singletons.shard*.jsonl"):
        if row.get("kind") == "b0":
            b0[row["dp_id"]] = row
        elif row.get("kind") == "singleton":
            singles[row["dp_id"]][int(row["event"])] = float(row["score"])

    set_rows: dict[str, dict[tuple[int, ...], dict]] = defaultdict(dict)
    anchors: dict[str, dict[int, float]] = defaultdict(dict)
    dedup_keys: set[str] = set()
    for row in load_jsonl_rows(args.sets_root, "sets.shard*.jsonl"):
        key = row.get("key")
        if not key or key in dedup_keys:
            continue
        dedup_keys.add(key)
        if row.get("kind") == "set":
            set_rows[row["dp_id"]][tuple(row["events"])] = row
        elif row.get("kind") == "recent_anchor":
            anchors[row["dp_id"]][int(row["size"])] = float(row["score"])

    # ---- edge 构建(特征惰性算,先收集索引)----
    edges = []  # (dp, selected_tuple, event, marginal)
    for dp, base in b0.items():
        if dp not in records or split_of[dp] == "test":
            continue
        pool = [int(e) for e in base["candidate_pool"]]
        have = singles.get(dp, {})
        for event in pool:
            if event in have:
                edges.append((dp, (), event, have[event] - base["score"]))
    for dp, by_set in set_rows.items():
        if dp not in records or split_of.get(dp) == "test":
            continue
        for events, row in by_set.items():
            parent = tuple(row["parent"])
            edges.append(
                (dp, parent, int(row["added_event"]),
                 float(row["score"]) - float(row["parent_score"]))
            )

    print(json.dumps({
        "states": len(b0), "singleton_edges": sum(1 for e in edges if not e[1]),
        "set_edges": sum(1 for e in edges if e[1]),
        "anchor_states": len(anchors),
    }))

    dup_cache = {dp: base.get("duplicates", {}) for dp, base in b0.items()}
    expected_dim = len(FEATURE_NAMES) + len(SET_FEATURE_NAMES) + intent_dims

    intent_map: dict[tuple[str, int], list[float]] = {}
    intent_dims = 0
    if args.intent_root is not None:
        for row in load_jsonl_rows(args.intent_root, "intent.shard*.jsonl"):
            if row.get("kind") == "intent":
                intent_map[(row["dp_id"], int(row["event"]))] = row["vector"]
        if not intent_map:
            raise SystemExit("intent-root 给了但没读到任何 intent 行")
        intent_dims = len(next(iter(intent_map.values())))
        print(json.dumps({"intent_rows": len(intent_map),
                           "intent_dims": intent_dims}))

    _FEAT_CTX.update({"records": records, "b0": b0, "dup": dup_cache,
                       "intent": intent_map, "intent_dims": intent_dims})

    feats = None
    if args.feature_cache is not None and args.feature_cache.exists():
        import torch as _torch
        cached = _torch.load(args.feature_cache)
        if (cached.get("edge_count") == len(edges)
                and cached.get("dim") == expected_dim):
            feats = cached["features"]
            print(json.dumps({"feature_cache": "hit",
                               "edges": len(edges)}))
        else:
            print(json.dumps({"feature_cache": "stale",
                               "cached": cached.get("edge_count"),
                               "expected": len(edges)}))
    if feats is None:
        import multiprocessing as _mp
        import time as _time
        started = _time.time()
        workers = max(1, min(args.feature_workers, len(edges) // 500 or 1))
        if workers > 1:
            with _mp.get_context("fork").Pool(workers) as pool:
                feats = pool.map(_feat_worker, edges, chunksize=512)
        else:
            feats = [_feat_worker(e) for e in edges]
        print(json.dumps({"feature_build_seconds": round(_time.time() - started, 1),
                           "workers": workers}))
        if args.feature_cache is not None:
            import torch as _torch
            _torch.save({"features": feats, "edge_count": len(edges),
                          "dim": expected_dim}, args.feature_cache)

    dim = len(FEATURE_NAMES) + len(SET_FEATURE_NAMES) + intent_dims
    import hashlib as _hashlib

    def _in_fraction(dp: str) -> bool:
        if args.train_fraction >= 1.0:
            return True
        h = int.from_bytes(
            _hashlib.sha256(f"{args.seed}:lc:{dp}".encode()).digest()[:8], "big"
        ) / 2**64
        return h < args.train_fraction

    train_idx = [
        i for i, e in enumerate(edges)
        if split_of[e[0]] == "train" and _in_fraction(e[0])
    ]
    dev_idx = [i for i, e in enumerate(edges) if split_of[e[0]] == "dev"]
    x_train = [feats[i] for i in train_idx]
    mean = [sum(col) / len(col) for col in zip(*x_train)]
    std = [
        max(math.sqrt(sum((v - m) ** 2 for v in col) / len(col)), 1e-6)
        for col, m in zip(zip(*x_train), mean)
    ]
    norm = lambda f: [(v - m) / s for v, m, s in zip(f, mean, std)]

    # ---- 可选:HGKV 读出特征(two-tower 残差塔输入) ----
    readout_dim = 0
    readout_vecs = None
    readout_mask = None
    if args.readout_root is not None:
        raw: dict[tuple[str, int], list[float]] = {}
        for row in load_jsonl_rows(args.readout_root, "readouts.shard*.jsonl"):
            if row.get("kind") != "readout":
                continue
            raw[(row["dp_id"], int(row["event"]))] = row["vector"]
        if not raw:
            raise SystemExit("readout-root 给了但没读到任何 readout 行")
        readout_dim = len(next(iter(raw.values())))
        # 候选池内 z 归一:每 dp、每维,对池内候选做 (v-μ)/σ —— scale-free,
        # 跨平台分布漂移大部分被池内归一吸收。
        by_dp: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
        for (dp, event), vec in raw.items():
            by_dp[dp].append((event, vec))
        normed: dict[tuple[str, int], list[float]] = {}
        for dp, items in by_dp.items():
            cols = list(zip(*[v for _, v in items]))
            mus = [sum(c) / len(c) for c in cols]
            sds = [
                max(math.sqrt(sum((x - m) ** 2 for x in c) / len(c)), 1e-6)
                for c, m in zip(cols, mus)
            ]
            for event, vec in items:
                normed[(dp, event)] = [
                    (x - m) / s for x, m, s in zip(vec, mus, sds)
                ]
        readout_vecs = []
        readout_mask = []
        for dp, _selected, event, _m in edges:
            vec = normed.get((dp, event))
            if vec is None:
                readout_vecs.append([0.0] * readout_dim)
                readout_mask.append(0.0)
            else:
                readout_vecs.append(vec)
                readout_mask.append(1.0)
        coverage = sum(readout_mask) / len(readout_mask)
        print(json.dumps({"readout_dim": readout_dim,
                           "readout_coverage": round(coverage, 4)}))

    torch.manual_seed(args.seed)
    device = args.device

    class TwoTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cheap = torch.nn.Sequential(
                torch.nn.Linear(dim, args.hidden), torch.nn.GELU(),
                torch.nn.Linear(args.hidden, args.hidden), torch.nn.GELU(),
                torch.nn.Linear(args.hidden, 1),
            )
            if readout_dim:
                self.readout = torch.nn.Sequential(
                    torch.nn.Linear(readout_dim, 64), torch.nn.GELU(),
                    torch.nn.Dropout(args.readout_dropout),
                    torch.nn.Linear(64, 1),
                )

        def forward(self, x, r=None, rm=None):
            score = self.cheap(x).squeeze(-1)
            if readout_dim and r is not None:
                score = score + self.readout(r).squeeze(-1) * rm
            return score

    class Concat(torch.nn.Module):
        """单塔全交互:readout 经 dropout 与 mask 门控后并入首层。"""

        def __init__(self):
            super().__init__()
            self.drop = torch.nn.Dropout(args.readout_dropout)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim + readout_dim, args.hidden),
                torch.nn.GELU(),
                torch.nn.Linear(args.hidden, args.hidden), torch.nn.GELU(),
                torch.nn.Linear(args.hidden, 1),
            )

        def forward(self, x, r=None, rm=None):
            if readout_dim and r is not None:
                r = self.drop(r) * rm.unsqueeze(-1)
                x = torch.cat([x, r], dim=-1)
            return self.net(x).squeeze(-1)

    if args.arch == "concat" and readout_dim:
        model = Concat().to(device)
        first = model.net[0]
        # readout 首层权重列吃强 weight decay(分组正则版降权,保留交互)
        decay_params = [first.weight]
        base_params = [
            p for n, p in model.named_parameters() if p is not first.weight
        ]
        param_groups = [
            {"params": base_params},
            {"params": decay_params,
             "weight_decay": args.readout_weight_decay},
        ]
    else:
        model = TwoTower().to(device)
        param_groups = [{"params": model.cheap.parameters()}]
        if readout_dim:
            param_groups.append({
                "params": model.readout.parameters(),
                "weight_decay": args.readout_weight_decay,
            })
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    x_all = torch.tensor([norm(f) for f in feats], dtype=torch.float32,
                         device=device)
    y_all = torch.tensor([e[3] for e in edges], dtype=torch.float32,
                         device=device)
    r_all = (
        torch.tensor(readout_vecs, dtype=torch.float32, device=device)
        if readout_dim else None
    )
    rm_all = (
        torch.tensor(readout_mask, dtype=torch.float32, device=device)
        if readout_dim else None
    )
    ti = torch.tensor(train_idx, dtype=torch.long, device=device)

    # 组内 pairwise(组 = 同 (dp, S)):真边际差超过 margin 的有序对。
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in train_idx:
        dp, selected, _event, _m = edges[i]
        groups[(dp, selected)].append(i)
    pair_left, pair_right = [], []
    for members in groups.values():
        for i in members:
            for j in members:
                if edges[i][3] > edges[j][3] + 1e-6:
                    pair_left.append(i)
                    pair_right.append(j)
    pl = torch.tensor(pair_left, dtype=torch.long, device=device)
    pr = torch.tensor(pair_right, dtype=torch.long, device=device)
    print(json.dumps({"rank_pairs": len(pair_left), "feature_dim": dim,
                       "train_edges": len(train_idx), "dev_edges": len(dev_idx)}))

    def _fwd(idx):
        if readout_dim:
            return model(x_all[idx], r_all[idx], rm_all[idx])
        return model(x_all[idx])

    def predict_indices(idx_list):
        idx = (
            idx_list
            if isinstance(idx_list, torch.Tensor)
            else torch.tensor(idx_list, dtype=torch.long, device=device)
        )
        with torch.no_grad():
            return _fwd(idx)

    edge_index = {(e[0], e[1], e[2]): i for i, e in enumerate(edges)}

    def replay_dev():
        """on-tree 端到端:模型在 teacher 已打分的树上贪心走,真 U 对比锚/oracle。"""
        gains = {2: [], 4: []}
        oracle_gap = {2: [], 4: []}
        k_dist = defaultdict(int)
        for dp, by_set in set_rows.items():
            if split_of.get(dp) != "dev" or dp not in b0:
                continue
            anchor = anchors.get(dp, {})
            have = singles.get(dp, {})
            base = b0[dp]
            if not have:
                continue
            # size-1 起点:模型在 (S=∅) 组里的 argmax(候选 = 单例池)
            pool = [int(e) for e in base["candidate_pool"]]
            idxs = [edge_index[(dp, (), e)] for e in pool if (dp, (), e) in edge_index]
            if not idxs:
                continue
            preds = predict_indices(idxs).tolist()
            order = [i for _, i in sorted(zip(preds, idxs), reverse=True)]
            current = (edges[order[0]][2],)
            current_u = have[current[0]]
            for target_b in (2, 3, 4):
                children = {
                    ev: row for ev_set, row in by_set.items()
                    if tuple(row["parent"]) == current
                    for ev in [int(row["added_event"])]
                }
                if not children:
                    break
                cidx = [
                    edge_index[(dp, current, ev)]
                    for ev in children if (dp, current, ev) in edge_index
                ]
                if not cidx:
                    break
                cpred = predict_indices(cidx).tolist()
                best = max(zip(cpred, cidx))[1]
                ev = edges[best][2]
                row = children[ev]
                # 模型可拒绝(预测边际<=0 → 停在当前集合)
                if cpred[cidx.index(best)] <= 0:
                    break
                current = tuple(sorted((*current, ev)))
                current_u = float(row["score"])
            for target_b in (2, 4):
                anchor_u = anchor.get(target_b)
                if anchor_u is None:
                    continue
                # 与锚同预算的模型集合真 U:若模型停得更早,补 recent 至同 B 无真分,
                # 保守用当前集合 U(等价于把剩余槽位留给 recent 的下界近似,计数披露)
                if len(current) <= target_b:
                    gains[target_b].append(current_u - anchor_u)
                scored = [
                    float(r["score"]) for s, r in by_set.items() if len(s) == target_b
                ] + [anchor_u]
                oracle_gap[target_b].append(max(scored) - max(current_u, anchor_u))
            k_dist[len(current)] += 1
        out = {}
        for b in (2, 4):
            if gains[b]:
                out[f"replay_gain_b{b}"] = round(sum(gains[b]) / len(gains[b]), 4)
                out[f"oracle_gap_b{b}"] = round(
                    sum(oracle_gap[b]) / len(oracle_gap[b]), 4)
                out[f"n_b{b}"] = len(gains[b])
        out["model_set_size_dist"] = dict(sorted(k_dist.items()))
        return out

    def _rank_metrics(idx_list, prefix):
        pred = predict_indices(idx_list).tolist()
        truth = [edges[i][3] for i in idx_list]
        mse = sum((a - b) ** 2 for a, b in zip(pred, truth)) / len(truth)
        by_state = defaultdict(list)
        for local, i in enumerate(idx_list):
            dp, selected, _e, m = edges[i]
            if not selected:
                by_state[dp].append((pred[local], m))
        spearmans, regrets = [], []
        for rows in by_state.values():
            ps = [p for p, _ in rows]
            ts = [t for _, t in rows]
            if len(rows) >= 2:
                spearmans.append(spearman(ps, ts))
            best = max(ts)
            chosen = ts[max(range(len(ps)), key=lambda k: ps[k])]
            regrets.append(best - chosen)
        return {
            f"{prefix}_edge_mse": round(mse, 5),
            f"{prefix}_singleton_spearman": round(
                sum(spearmans) / len(spearmans), 4) if spearmans else None,
            f"{prefix}_top1_regret": round(sum(regrets) / len(regrets), 4)
            if regrets else None,
        }

    # train 侧对照样本:抽**整组** dp(逐边抽样会把组抽稀成单候选,regret 恒 0、
    # Spearman 无意义 —— 学习曲线首轮的教训),取前 ~80 个 train dp 的全部边。
    _probe_dps: list[str] = []
    for i in train_idx:
        dp = edges[i][0]
        if dp not in _probe_dps:
            _probe_dps.append(dp)
        if len(_probe_dps) >= 80:
            break
    _probe_set = set(_probe_dps)
    train_probe_idx = [i for i in train_idx if edges[i][0] in _probe_set]

    def eval_dev():
        model.eval()
        return {
            **_rank_metrics(dev_idx, "dev"),
            **_rank_metrics(train_probe_idx, "train_probe"),
            **replay_dev(),
        }

    # note (luojiaxuan): epochs 是预算上限,不是训练时长的权威。选择权在
    # held-out:按 dev_top1_regret(k=1 部署关键量,样本最稳)保存最优快照,
    # 连续 patience 次评测无提升即早停;过峰值的过拟合段可见但不被选中。
    history = []
    best_metric = None
    best_state = None
    best_epoch = None
    misses = 0
    for epoch in range(args.epochs):
        model.train()
        perm = ti[torch.randperm(len(ti), device=device)]
        total = 0.0
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            pred = _fwd(idx)
            loss = torch.nn.functional.mse_loss(pred, y_all[idx])
            if args.rank_weight and len(pl):
                take = torch.randint(0, len(pl), (len(idx),), device=device)
                lp = _fwd(pl[take])
                rp = _fwd(pr[take])
                loss = loss + args.rank_weight * torch.relu(
                    args.rank_margin - (lp - rp)
                ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            snapshot = {"epoch": epoch + 1, "train_loss": round(total, 4),
                        **eval_dev()}
            history.append(snapshot)
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            metric = snapshot.get("dev_top1_regret")
            if metric is not None and (best_metric is None or metric < best_metric):
                best_metric = metric
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                misses = 0
            else:
                misses += 1
                if misses >= args.early_stop_patience:
                    print(json.dumps({
                        "early_stop_at_epoch": epoch + 1,
                        "best_epoch": best_epoch,
                        "best_dev_top1_regret": best_metric,
                    }))
                    break

    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state if best_state is not None else model.state_dict(),
        "best_epoch": best_epoch,
        "best_dev_top1_regret": best_metric,
        "mean": mean, "std": std,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "set_feature_names": list(SET_FEATURE_NAMES),
        "hidden": args.hidden,
        "readout_dim": readout_dim,
        "intent_dims": intent_dims,
    }, args.output_root / "marginal_scorer.pt")
    (args.output_root / "report.json").write_text(
        json.dumps({
            "schema_version": "causalcache.selector_v4_marginal_report.v1",
            "history": history,
            "hyper": {"epochs": args.epochs, "hidden": args.hidden,
                      "lr": args.learning_rate, "rank_weight": args.rank_weight,
                      "rank_margin": args.rank_margin, "seed": args.seed},
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"final": history[-1] if history else None},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
