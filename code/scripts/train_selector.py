#!/usr/bin/env python3
"""Train the Odyssey-only success-anchored selector and gate on heldout.

# note (luojiaxuan): 第一版 selector = per-event U_act 回归器(平台通用特征,
# 梯度提升;无 sklearn 时回退岭回归 numpy 实现)。at-most-B greedy 推理 =
# 按预测 gain 降序取正 gain 的前 B 个。heldout 门禁对比 Recent / Random /
# Oldest baseline 的 top-1/top-B 命中真实正 gain 事件与 oracle regret。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

FEATURE_ORDER: list[str] = []


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def vectorize(rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[float]]:
    global FEATURE_ORDER
    if not FEATURE_ORDER:
        FEATURE_ORDER = sorted(rows[0]["features"].keys())
    X = [[r["features"][k] for k in FEATURE_ORDER] for r in rows]
    y = [r["u_act"] for r in rows]
    return X, y


def group_by_decision(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["pair_group"], []).append(r)
    return groups


def evaluate(
    groups: dict[str, list[dict[str, Any]]],
    scorer,
    *,
    strategy: str,
    budget: int,
) -> dict[str, float]:
    """top-B 选择命中真实正-gain 事件的比例 + oracle regret。"""
    rng = random.Random(20240724)
    top1_hit = 0
    captured = 0.0
    oracle = 0.0
    n = 0
    for rows in groups.values():
        if not rows:
            continue
        true = {r["candidate_step"]: r["u_act"] for r in rows}
        if strategy == "model":
            order = sorted(rows, key=lambda r: scorer(r), reverse=True)
            picked = [r["candidate_step"] for r in order if scorer(r) > 0][:budget]
        elif strategy == "recent":
            picked = [
                r["candidate_step"]
                for r in sorted(rows, key=lambda r: -r["candidate_step"])
            ][:budget]
        elif strategy == "oldest":
            picked = [
                r["candidate_step"]
                for r in sorted(rows, key=lambda r: r["candidate_step"])
            ][:budget]
        elif strategy == "random":
            picked = [r["candidate_step"] for r in rng.sample(rows, len(rows))][
                :budget
            ]
        else:
            raise ValueError(strategy)
        best = max(true.values())
        top_true = sorted(true.values(), reverse=True)[:budget]
        captured += sum(true[p] for p in picked)
        oracle += sum(top_true)
        if picked and true[picked[0]] == best:
            top1_hit += 1
        n += 1
    return {
        "n": n,
        "top1_best_rate": top1_hit / max(n, 1),
        "captured_gain": captured / max(n, 1),
        "oracle_gain": oracle / max(n, 1),
        "regret": (oracle - captured) / max(n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()

    rows = load_rows(args.features)
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "heldout"]
    Xtr, ytr = vectorize(train)

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=6, random_state=271828
        )
        model.fit(Xtr, ytr)
        backend = "hist_gbm"

        def predict(row: dict[str, Any]) -> float:
            return float(model.predict([[row["features"][k] for k in FEATURE_ORDER]])[0])

    except Exception:  # noqa: BLE001
        import numpy as np

        A = np.array(Xtr, dtype=float)
        b = np.array(ytr, dtype=float)
        mu = A.mean(0)
        sd = A.std(0)
        sd[sd == 0] = 1.0
        An = (A - mu) / sd
        An = np.hstack([An, np.ones((len(An), 1))])
        w = np.linalg.solve(An.T @ An + 1e-2 * np.eye(An.shape[1]), An.T @ b)
        backend = "ridge_numpy"

        def predict(row: dict[str, Any]) -> float:
            x = np.array([row["features"][k] for k in FEATURE_ORDER], dtype=float)
            x = (x - mu) / sd
            return float(np.append(x, 1.0) @ w)

    held_groups = group_by_decision(held)
    report = {"backend": backend, "feature_order": FEATURE_ORDER, "budget": args.budget}
    for strat in ("model", "recent", "oldest", "random"):
        report[strat] = evaluate(
            held_groups, predict, strategy=strat, budget=args.budget
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selector_gate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
