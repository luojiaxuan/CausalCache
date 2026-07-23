#!/usr/bin/env python3
"""V1 main selector: state-conditioned multimodal event scorer on s75 representations.

# note (luojiaxuan): query 分支 = s75 对 b0 prompt 的末层 hidden 三段池化(策略
# 对齐表示);event 分支 = 冻结 vision tower 的 post 图池化 + pre→post delta +
# 标量特征。交互 = 投影后 [q, e, q⊙e, |q−e|, cos] + 标量残差支路。三头输出
# (gain 回归 / 正负分类 / 组内排序),损失 Huber + within-state pairwise hinge +
# 负类加权 BCE。评测:组内 Spearman、top-1、precision@B、captured/regret、STOP、
# B1/2/4/8 对 recent/random/visual-similarity。GBM 版(纯标量)同框架下作 V0 对照。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def load_all(args, torch):
    scalar_rows = [
        json.loads(line) for line in args.features.open(encoding="utf-8")
    ]
    scalar_keys = sorted(scalar_rows[0]["features"].keys())
    query_cache: dict[str, Any] = {}
    for path in args.query_emb.glob("*.pt"):
        query_cache[path.stem.replace("__", ":")] = torch.load(
            path, map_location="cpu"
        )
    event_cache: dict[tuple[str, int], Any] = {}
    for path in args.event_emb.glob("*.pt"):
        source_id, step = path.stem.rsplit("__", 1)
        event_cache[(source_id, int(step))] = torch.load(path, map_location="cpu")
    rows = []
    dropped = 0
    for r in scalar_rows:
        pg = r["pair_group"]
        source_id, decision = pg.split(":")
        cand = int(r["candidate_step"])
        q = query_cache.get(pg)
        post = event_cache.get((source_id, cand))
        if q is None or post is None:
            dropped += 1
            continue
        pre = event_cache.get((source_id, cand - 1))
        cur = event_cache.get((source_id, int(decision)))
        rows.append(
            {
                "pair_group": pg,
                "candidate_step": cand,
                "u_act": float(r["u_act"]),
                "split": r["split"],
                "scalars": [float(r["features"][k]) for k in scalar_keys],
                "q": q,
                "post": post,
                "pre": pre,
                "cur_vis": cur,
            }
        )
    return rows, scalar_keys, dropped


def build_model(q_dim, e_dim, n_scalars, torch):
    nn = torch.nn

    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            d = 256
            self.q_proj = nn.Linear(q_dim * 3, d)
            self.e_proj = nn.Linear(e_dim, d)
            self.delta_proj = nn.Linear(e_dim, d)
            self.scalar_mlp = nn.Sequential(
                nn.Linear(n_scalars, 64), nn.GELU(), nn.Linear(64, 64)
            )
            self.trunk = nn.Sequential(
                nn.Linear(d * 5 + 1 + 64, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.GELU(),
            )
            self.head_gain = nn.Linear(256, 1)
            self.head_sign = nn.Linear(256, 1)
            self.head_rank = nn.Linear(256, 1)

        def forward(self, q, e, delta, scalars):
            qp = self.q_proj(q)
            ep = self.e_proj(e)
            dp = self.delta_proj(delta)
            cos = torch.nn.functional.cosine_similarity(qp, ep, dim=-1, eps=1e-6)
            z = torch.cat(
                [qp, ep, dp, qp * ep, (qp - ep).abs(), cos.unsqueeze(-1),
                 self.scalar_mlp(scalars)],
                dim=-1,
            )
            h = self.trunk(z)
            return (
                self.head_gain(h).squeeze(-1),
                self.head_sign(h).squeeze(-1),
                self.head_rank(h).squeeze(-1),
            )

    return Scorer()


def tensors_for(rows, torch, device):
    q = torch.stack(
        [
            torch.cat([r["q"]["full"], r["q"]["text"], r["q"]["current_image"]])
            for r in rows
        ]
    ).float().to(device)
    post = torch.stack([r["post"] for r in rows]).float().to(device)
    delta = torch.stack(
        [
            (r["post"] - r["pre"]) if r["pre"] is not None else torch.zeros_like(r["post"])
            for r in rows
        ]
    ).float().to(device)
    scalars = torch.tensor([r["scalars"] for r in rows], dtype=torch.float32, device=device)
    y = torch.tensor([r["u_act"] for r in rows], dtype=torch.float32, device=device)
    return q, post, delta, scalars, y


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = float(pos)
        return rk
    ra, rb = rank(a), rank(b)
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def evaluate_groups(groups, score_fn, budgets, rng):
    """score_fn(row)->float;返回全套指标。STOP 规则:score<=0 不选。"""
    out = {}
    spearmans = []
    top1 = 0
    stop_correct = 0
    stop_total = 0
    per_budget = {b: {"captured": 0.0, "oracle": 0.0, "pos_sel": 0, "sel": 0} for b in budgets}
    n = 0
    for rows in groups.values():
        true = [r["u_act"] for r in rows]
        pred = [score_fn(r) for r in rows]
        if len(rows) >= 2:
            spearmans.append(spearman(pred, true))
        best = max(true)
        order = sorted(range(len(rows)), key=lambda i: -pred[i])
        if true[order[0]] == best:
            top1 += 1
        for i, r in enumerate(rows):
            stop_total += 1
            if (pred[i] > 0) == (true[i] > 0):
                stop_correct += 1
        for b in budgets:
            picked = [i for i in order if pred[i] > 0][:b]
            top_true = sorted(true, reverse=True)[:b]
            per_budget[b]["captured"] += sum(true[i] for i in picked)
            per_budget[b]["oracle"] += sum(v for v in top_true if v > 0)
            per_budget[b]["pos_sel"] += sum(1 for i in picked if true[i] > 0)
            per_budget[b]["sel"] += len(picked)
        n += 1
    out["n"] = n
    out["spearman_within_state"] = sum(spearmans) / max(len(spearmans), 1)
    out["top1_best_rate"] = top1 / max(n, 1)
    out["stop_accuracy"] = stop_correct / max(stop_total, 1)
    for b in budgets:
        d = per_budget[b]
        out[f"B{b}"] = {
            "captured_gain": d["captured"] / max(n, 1),
            "oracle_gain": d["oracle"] / max(n, 1),
            "regret": (d["oracle"] - d["captured"]) / max(n, 1),
            "positive_precision": d["pos_sel"] / max(d["sel"], 1),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--query-emb", type=Path, required=True)
    parser.add_argument("--event-emb", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--sign-weight", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=0.02)
    parser.add_argument("--states-per-batch", type=int, default=192)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows, scalar_keys, dropped = load_all(args, torch)
    print(json.dumps({"rows": len(rows), "dropped": dropped}), flush=True)
    train_rows = [r for r in rows if r["split"] == "train"]
    held_rows = [r for r in rows if r["split"] == "heldout"]

    groups_train: dict[str, list[dict[str, Any]]] = {}
    for r in train_rows:
        groups_train.setdefault(r["pair_group"], []).append(r)
    groups_held: dict[str, list[dict[str, Any]]] = {}
    for r in held_rows:
        groups_held.setdefault(r["pair_group"], []).append(r)

    q_dim = rows[0]["q"]["full"].shape[0]
    e_dim = rows[0]["post"].shape[0]
    model = build_model(q_dim, e_dim, len(scalar_keys), torch).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    neg_frac = sum(1 for r in train_rows if r["u_act"] <= 0) / max(len(train_rows), 1)
    pos_weight_value = neg_frac / max(1 - neg_frac, 1e-6)
    huber = torch.nn.SmoothL1Loss(beta=0.05)

    state_keys = list(groups_train.keys())
    budgets = (1, 2, 4, 8)
    best_score = -1e9
    args.output_root.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(state_keys)
        total = {"huber": 0.0, "rank": 0.0, "sign": 0.0, "batches": 0}
        for start in range(0, len(state_keys), args.states_per_batch):
            batch_states = state_keys[start : start + args.states_per_batch]
            batch_rows = []
            group_slices = []
            for pg in batch_states:
                g = groups_train[pg]
                group_slices.append((len(batch_rows), len(batch_rows) + len(g)))
                batch_rows.extend(g)
            q, post, delta, scalars, y = tensors_for(batch_rows, torch, args.device)
            gain, sign, rank_s = model(q, post, delta, scalars)
            loss_huber = huber(gain, y)
            # note (luojiaxuan): 负类只有 ~17%,BCE 用类频率反比加权,防"全选满"。
            sign_target = (y > 0).float()
            weight = torch.where(
                sign_target > 0,
                torch.full_like(y, pos_weight_value),
                torch.ones_like(y),
            )
            loss_sign = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    sign, sign_target, reduction="none"
                )
                * weight
            ).mean()
            rank_terms = []
            for lo, hi in group_slices:
                if hi - lo < 2:
                    continue
                ys = y[lo:hi]
                ss = rank_s[lo:hi]
                top = int(torch.argmax(ys))
                bot = int(torch.argmin(ys))
                pairs = [(top, bot)]
                idx = list(range(hi - lo))
                rng.shuffle(idx)
                for a, b in zip(idx[::2], idx[1::2]):
                    if ys[a] != ys[b]:
                        pairs.append((a, b) if ys[a] > ys[b] else (b, a))
                for i, j in pairs:
                    if ys[i] > ys[j]:
                        rank_terms.append(
                            torch.relu(args.rank_margin - (ss[i] - ss[j]))
                        )
            loss_rank = (
                torch.stack(rank_terms).mean()
                if rank_terms
                else torch.zeros((), device=args.device)
            )
            loss = loss_huber + args.rank_weight * loss_rank + args.sign_weight * loss_sign
            optim.zero_grad()
            loss.backward()
            optim.step()
            total["huber"] += float(loss_huber)
            total["rank"] += float(loss_rank)
            total["sign"] += float(loss_sign)
            total["batches"] += 1

        model.eval()
        pred_cache: dict[tuple[str, int], float] = {}
        rank_cache: dict[tuple[str, int], float] = {}
        with torch.no_grad():
            all_held = [r for g in groups_held.values() for r in g]
            for lo in range(0, len(all_held), 4096):
                chunk = all_held[lo : lo + 4096]
                q, post, delta, scalars, _ = tensors_for(chunk, torch, args.device)
                gain, _, rank_s = model(q, post, delta, scalars)
                for r, gv, rv in zip(chunk, gain.tolist(), rank_s.tolist()):
                    pred_cache[(r["pair_group"], r["candidate_step"])] = gv
                    rank_cache[(r["pair_group"], r["candidate_step"])] = rv

        # note (luojiaxuan): 选择用 rank 头排序、STOP 用 gain 头符号——各司其职。
        def model_score(r):
            key = (r["pair_group"], r["candidate_step"])
            return rank_cache[key] if pred_cache[key] > 0 else min(
                rank_cache[key], 0.0
            ) - 1e3 * (pred_cache[key] <= 0)

        report = evaluate_groups(groups_held, model_score, budgets, rng)
        report["epoch"] = epoch
        report["train_loss"] = {
            k: round(v / max(total["batches"], 1), 4)
            for k, v in total.items()
            if k != "batches"
        }
        print(json.dumps(report), flush=True)
        select_metric = report["spearman_within_state"]
        if select_metric > best_score:
            best_score = select_metric
            torch.save(model.state_dict(), args.output_root / "selector_v1_best.pt")
            (args.output_root / "selector_v1_best_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )

    baselines = {}
    for name in ("recent", "random", "visual_sim"):
        if name == "recent":
            fn = lambda r: float(r["candidate_step"])  # noqa: E731
        elif name == "random":
            values = {
                (r["pair_group"], r["candidate_step"]): rng.random() for r in held_rows
            }
            fn = lambda r: values[(r["pair_group"], r["candidate_step"])]  # noqa: E731
        else:
            import torch as _t

            def fn(r):  # noqa: ANN001
                if r["cur_vis"] is None:
                    return 0.0
                return float(
                    _t.nn.functional.cosine_similarity(
                        r["post"].float(), r["cur_vis"].float(), dim=0
                    )
                )

        baselines[name] = evaluate_groups(groups_held, fn, budgets, rng)
    (args.output_root / "selector_v1_baselines.json").write_text(
        json.dumps(baselines, indent=2), encoding="utf-8"
    )
    print(json.dumps({"baselines": baselines}), flush=True)


if __name__ == "__main__":
    main()
