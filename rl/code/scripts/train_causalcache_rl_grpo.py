#!/usr/bin/env python3
"""CausalCache-RL:selector + HGKV LoRA 的联合 GRPO 训练(策略冻结)。

# note (luojiaxuan): 2026-08-04 pivot。旧线的病根是离线代理目标(teacher-forced
# log-prob 差中差 + recurrence 启发式标签)的增益不迁移到闭环成功率;RL 把训练
# 信号换成任务成败本身,relevant/wrong/age>=B+2 全部退场。
#
# 可学件与梯度来源(GUI-Owl 主干与视觉塔全冻结):
#   log p(tau) = sum_t [ log pi_sel(S_t|x_t)  +  log p_{theta0+Delta}(a_t|prompt(S_t)) ]
#     - selector:log pi_sel 从 rollout 审计里存的 Plackett-Luce 逐轮特征复算,
#       完全不需要在训练侧重建环境状态(serve --rl-audit-dir 的设计初衷);
#     - LoRA:动作 token 的 teacher-forced log-prob,prompt 由收集器用与 serve
#       同一套 builder(build_official_messages_for_request)重建。
#
# 目标:GRPO 组相对优势 + KL 信任域 + selector 熵。
#   A_i = (r_i - mean_G) / (std_G + eps_std)
#   L = -sum_i A_i [ w_sel * logpi_sel(tau_i) + w_act * logp_act(tau_i) ]
#       + beta * KL(p_{theta0+Delta} || p_theta0)   (rollout 状态上逐 token)
#       - lam_H * H(pi_sel)
# KL 项一个杠杆替代旧目标手工调不对的全部 drift cap:任何没换来回报的
# 分布偏移统一被罚;这也是防"均匀抬升"复活的原理性版本。
#
# 全 0 / 全 1 的组优势为零,自动跳过(但计数披露,不静默)。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=Path, required=True,
                   help="collect_rl_trajectories.py 产出的 groups.jsonl")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--selector-bundle", type=Path, required=True,
                   help="初始 selector(v4 two_tower bundle;warm start)")
    p.add_argument("--adapter-config", type=Path, required=True,
                   help="HGKV 结构配置(rank/层/模块;沿用 v4 形制)")
    p.add_argument("--resume-adapter", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--selector-learning-rate", type=float, default=1e-4)
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument("--entropy-lambda", type=float, default=0.01)
    p.add_argument("--w-sel", type=float, default=1.0)
    p.add_argument("--w-act", type=float, default=1.0)
    p.add_argument("--adv-std-eps", type=float, default=0.1,
                   help="优势归一的方差下限;G=8 的二值奖励 std 本身噪声大")
    p.add_argument("--max-groups-per-step", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260804)
    return p.parse_args()


# ---------------------------------------------------------------------------
# selector:与部署同构的两塔 cheap 头(readout 部署侧恒 mask,RL 不训它)
# ---------------------------------------------------------------------------
def build_selector(bundle_path: Path, torch):
    bundle = torch.load(bundle_path, map_location="cpu")
    dim = len(bundle["norm_mean"])

    class CheapHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = bundle.get("hidden", 192)
            self.cheap = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.cheap(x).squeeze(-1)

    model = CheapHead()
    # bundle 的 state 里带 readout 键时只取 cheap 塔
    state = {k: v for k, v in bundle["model_state"].items()
             if k.startswith("cheap.")}
    model.load_state_dict(state, strict=True)
    return model, bundle


def selector_logprob_and_entropy(model, episode, torch, temperature: float):
    """从审计特征复算 log pi_sel(整条轨迹)与逐轮熵(可导)。

    # note (luojiaxuan): 审计里的特征已经过 bundle 归一化(serve 侧 selector_norm),
    # 这里直接前向,千万不要再归一化一次。offset 是常数,softmax 下无影响,不加。
    """
    logp_total = None
    ent_total = None
    n_rounds = 0
    for step in episode["steps"]:
        for rnd in step.get("rounds") or []:
            feats = torch.tensor(rnd["features"], dtype=torch.float32,
                                 device=next(model.parameters()).device)
            scores = model(feats) / temperature
            logp = torch.log_softmax(scores, dim=0)
            idx = rnd["candidates"].index(rnd["chosen"])
            ent = -(logp.exp() * logp).sum()
            logp_total = logp[idx] if logp_total is None else logp_total + logp[idx]
            ent_total = ent if ent_total is None else ent_total + ent
            n_rounds += 1
    return logp_total, ent_total, n_rounds


def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    # ---- 组数据 ----
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in args.groups.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ep = json.loads(line)
        groups[ep["task_id"]].append(ep)

    usable, degenerate = [], 0
    for tid, eps in groups.items():
        rs = [float(e["reward"]) for e in eps]
        if len(eps) >= 2 and max(rs) != min(rs):
            usable.append((tid, eps))
        else:
            degenerate += 1
    print(json.dumps({"groups_total": len(groups), "groups_usable": len(usable),
                      "groups_degenerate_skipped": degenerate}), flush=True)
    if not usable:
        raise SystemExit("没有可用组(全 0/全 1)——检查任务可学带筛选")

    # ---- selector ----
    sel_model, bundle = build_selector(args.selector_bundle, torch)
    sel_model.to(args.device).train()

    # ---- 冻结策略 + LoRA:复用既有 runtime 装载(结构与 v4 同形制)----
    # note (luojiaxuan): 动作侧 teacher-forcing 走 train_success_sft_lora 的
    # encode/score 机制(同一 repo 版本,与 serve 的 prompt builder 同源)。
    # 这里只暴露两个函数:action_logprob(episode, grad) 与 kl_to_frozen(episode)。
    from causalcache_rl.action_scoring import (
        ActionScorer,
    )
    scorer = ActionScorer(
        model_dir=args.model_dir,
        snapshot_manifest=args.snapshot_manifest,
        adapter_config=json.loads(args.adapter_config.read_text()),
        resume_adapter=args.resume_adapter,
        device=args.device,
    )

    params = [
        {"params": sel_model.parameters(), "lr": args.selector_learning_rate},
        {"params": scorer.adapter_parameters(), "lr": args.learning_rate},
    ]
    optimizer = torch.optim.AdamW(params)

    random.shuffle(usable)
    batch = usable[: args.max_groups_per_step]
    stats = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)
    n_ep = 0

    for tid, eps in batch:
        rs = [float(e["reward"]) for e in eps]
        mean_r = sum(rs) / len(rs)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rs) / len(rs))
        for ep, r in zip(eps, rs):
            adv = (r - mean_r) / (std_r + args.adv_std_eps)
            tau = float(ep["temperature"])
            logp_sel, ent, n_rounds = selector_logprob_and_entropy(
                sel_model, ep, torch, tau)
            loss = torch.zeros((), device=args.device)
            if logp_sel is not None:
                loss = loss - args.w_sel * adv * logp_sel
                loss = loss - args.entropy_lambda * ent
                stats["sel_rounds"] += n_rounds
            # 动作侧:逐步 teacher-forcing(带梯度),KL 对照冻结 bypass
            logp_act = scorer.episode_action_logprob(ep, grad=True)
            if logp_act is not None:
                loss = loss - args.w_act * adv * logp_act
                kl = scorer.episode_kl_to_frozen(ep)
                if kl is not None:
                    loss = loss + args.kl_beta * kl
                    stats["kl"] += float(kl.detach())
            (loss / args.grad_accum).backward()
            n_ep += 1
            stats["loss"] += float(loss.detach())
            stats["adv_abs"] += abs(adv)
            if n_ep % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in params for p in g["params"]], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    if n_ep % args.grad_accum:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # ---- 落盘:selector bundle(部署同构)+ LoRA + 迭代统计 ----
    out_bundle = dict(bundle)
    out_bundle["model_state"] = {
        f"cheap.{k.split('cheap.', 1)[1]}" if k.startswith("cheap.") else k: v
        for k, v in sel_model.state_dict().items()
    }
    torch.save(out_bundle, args.output_root / "selector_bundle.pt")
    scorer.save_adapter(args.output_root / "adapter.pt")
    report = {
        "episodes": n_ep,
        "groups_used": len(batch),
        "mean_loss": stats["loss"] / max(n_ep, 1),
        "mean_abs_adv": stats["adv_abs"] / max(n_ep, 1),
        "mean_kl": stats["kl"] / max(n_ep, 1),
        "selector_rounds": int(stats["sel_rounds"]),
    }
    (args.output_root / "iter_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
