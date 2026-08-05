#!/usr/bin/env python3
"""初始化 hidden-state 打分头 bundle(selector v2 的 iter-0 bootstrap)。

# note (luojiaxuan): 头 = Linear(H, 1),H 必须等于策略 text 侧 hidden_size
# (serve 启动时会 fail-closed 断言)。权重 std=0.02 小随机——起点接近均匀
# 采样,探索由 τ 提供;不做任何蒸馏(cheap 塔的排序不值得继承,那正是要废的)。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import torch

    torch.manual_seed(args.seed)
    head = torch.nn.Linear(args.hidden_size, 1)
    torch.nn.init.normal_(head.weight, std=0.02)
    torch.nn.init.zeros_(head.bias)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "hidden_head",
        "hidden_size": args.hidden_size,
        "model_state": head.state_dict(),
        "init": {"std": 0.02, "seed": args.seed},
    }, args.output)
    print(f"hidden_head 写至 {args.output}(H={args.hidden_size})")


if __name__ == "__main__":
    main()
