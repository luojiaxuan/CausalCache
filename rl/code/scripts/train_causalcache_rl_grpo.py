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
#
# DDP(2026-08-06):按 episode 分片 + 手工梯度 all-reduce,不用 DDP wrapper——
# scorer 的逐步反传是"一次 optimizer.step 前多次 backward",与 wrapper 的
# 桶式同步天生犯冲;手工 all-reduce(SUM)在累积窗口边界做一次,数学上与
# 单进程严格等价(fp 加法顺序除外)。episode 全局计数决定窗口,rank 按
# e_idx % world 认领;窗口内没活的 rank 以零梯度参与集合通信。所有 rank
# 优化器状态恒等(同种子同初值同梯度),rank0 落盘,末尾跨 rank 校验参数和。
# 单进程(WORLD_SIZE 缺省)走原路径,冒烟脚本不受影响。
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    p.add_argument("--limit-episodes-per-group", type=int, default=0,
                   help="冒烟专用:每组只取前 N 条 episode(0=关)。改变优势口径,正式训练禁用")
    return p.parse_args()


# ---------------------------------------------------------------------------
# selector:与部署同构的两塔 cheap 头(readout 部署侧恒 mask,RL 不训它)
# ---------------------------------------------------------------------------
def build_selector(bundle_path: Path, torch):
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    # selector v2:hidden_head = Linear(H,1),特征 = 策略自己的 hidden state
    # (审计里 base64-fp16 传输),无任何手写特征。
    if bundle.get("kind") == "hidden_head":
        head = torch.nn.Linear(int(bundle["hidden_size"]), 1)
        head.load_state_dict(bundle["model_state"])
        return head, bundle
    dim = len(bundle["mean"])

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


def _round_features(step, rnd, torch, device):
    """一轮的候选特征矩阵:cheap = 审计内嵌 28 维;hidden = 解码 fp16 向量。"""
    if rnd.get("features") is not None:
        return torch.tensor(rnd["features"], dtype=torch.float32, device=device)
    import base64

    import numpy as np
    hv = step.get("hidden_vectors") or {}
    H = int(step["hidden_size"])
    mats = [np.frombuffer(base64.b64decode(hv[str(c)]), dtype=np.float16)
                .astype("float32").reshape(H)
            for c in rnd["candidates"]]
    return torch.tensor(np.stack(mats), device=device)


def selector_logprob_and_entropy(model, episode, torch, temperature: float):
    """从审计特征复算 log pi_sel(整条轨迹)与逐轮熵(可导)。

    # note (luojiaxuan): cheap 特征已过 bundle 归一化,不要再归一化;
    # hidden 向量原样进头(头自己学尺度)。offset 是常数,softmax 下无影响。
    """
    logp_total = None
    ent_total = None
    n_rounds = 0
    for step in episode["steps"]:
        for rnd in step.get("rounds") or []:
            feats = _round_features(
                step, rnd, torch, next(model.parameters()).device)
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

    # ---- DDP 形态(torchrun 注入 env;单进程时 world=1 走原路径)----
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        from datetime import timedelta

        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # episode 长短不均,窗口边界的 all-reduce 可能等最慢 rank 数分钟;
        # 模型装载也在首个集合通信之前——超时给足
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        args.device = f"cuda:{local_rank}"
    is_main = rank == 0

    def rprint(payload) -> None:
        if is_main:
            print(json.dumps(payload), flush=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if is_main:
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
        if args.limit_episodes_per_group > 0:
            eps = eps[: args.limit_episodes_per_group]
        rs = [float(e["reward"]) for e in eps]
        if len(eps) >= 2 and max(rs) != min(rs):
            usable.append((tid, eps))
        else:
            degenerate += 1
    rprint({"groups_total": len(groups), "groups_usable": len(usable),
            "groups_degenerate_skipped": degenerate, "world_size": world})
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

    # note (luojiaxuan): 位移仪表(2026-08-08 加)。iter-20 复盘发现 selector 20
    # 迭代只挪了 4.4%、选择与随机初始化头重合 86.7%——总优化步数仅约 75 次。
    # 没有这个数就只能在 20 迭代后靠离线挖掘发现"根本没训动"。每迭代直接报。
    sel_w0 = [p.detach().clone() for p in sel_model.parameters()]

    random.shuffle(usable)
    batch = usable[: args.max_groups_per_step]
    stats = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)
    n_ep = 0

    trainable = [p for g in params for p in g["params"]]

    def sync_and_step() -> None:
        # 窗口边界:先跨 rank 求和梯度(无梯度参数补零参与),再 clip + step。
        # all-reduce 结果各 rank 逐位一致 → 优化器轨迹恒等,无需广播参数。
        if world > 1:
            import torch.distributed as dist
            for p in trainable:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    e_idx = 0  # 全局 episode 序号:窗口划分与 rank 认领都用它
    for tid, eps in batch:
        rs = [float(e["reward"]) for e in eps]
        mean_r = sum(rs) / len(rs)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rs) / len(rs))
        for ep, r in zip(eps, rs):
            mine = (e_idx % world) == rank
            e_idx += 1
            n_ep += 1
            if mine:
                adv = (r - mean_r) / (std_r + args.adv_std_eps)
                tau = float(ep["temperature"])
                logp_sel, ent, n_rounds = selector_logprob_and_entropy(
                    sel_model, ep, torch, tau)
                loss = torch.zeros((), device=args.device)
                if logp_sel is not None:
                    loss = loss - args.w_sel * adv * logp_sel
                    loss = loss - args.entropy_lambda * ent
                    stats["sel_rounds"] += n_rounds
                # selector 侧(小图)整条反传;动作侧 scorer 内逐步反传(防 OOM)
                if loss.requires_grad:
                    (loss / args.grad_accum).backward()
                lp, kl, _n = scorer.episode_backward(
                    ep,
                    pg_coef=args.w_act * adv / args.grad_accum,
                    kl_coef=args.kl_beta / args.grad_accum)
                stats["kl"] += kl
                stats["act_logp"] += lp
                stats["loss"] += float(loss.detach())
                stats["adv_abs"] += abs(adv)
                stats["ep_local"] += 1
            if n_ep % args.grad_accum == 0:
                sync_and_step()

    if n_ep % args.grad_accum:
        sync_and_step()

    # ---- 跨 rank 汇总统计;校验参数轨迹未分叉 ----
    param_sum = float(sum(p.detach().double().sum() for p in trainable))
    if world > 1:
        import torch.distributed as dist
        agg = torch.tensor(
            [stats["loss"], stats["adv_abs"], stats["kl"], stats["act_logp"],
             stats["sel_rounds"], stats["ep_local"]],
            dtype=torch.float64, device=args.device)
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        (stats["loss"], stats["adv_abs"], stats["kl"], stats["act_logp"],
         stats["sel_rounds"], stats["ep_local"]) = agg.tolist()
        sums = [None] * world
        dist.all_gather_object(sums, param_sum)
        if is_main and any(abs(s - param_sum) > 1e-6 * max(1.0, abs(param_sum))
                           for s in sums):
            raise SystemExit(f"DDP 参数轨迹分叉:param_sum={sums}")
    assert int(stats["ep_local"]) == n_ep, \
        f"episode 认领不完整:{int(stats['ep_local'])} != {n_ep}"

    # 本迭代 selector 相对位移 + 实际优化步数(诊断"训没训动"的第一手指标)
    with torch.no_grad():
        num = sum(float((p - q).pow(2).sum())
                  for p, q in zip(sel_model.parameters(), sel_w0)) ** 0.5
        den = sum(float(q.pow(2).sum()) for q in sel_w0) ** 0.5
    sel_drift = num / den if den else float("nan")
    opt_steps = n_ep // args.grad_accum + (1 if n_ep % args.grad_accum else 0)

    # ---- 落盘(rank0):selector bundle(部署同构)+ LoRA + 迭代统计 ----
    if is_main:
        out_bundle = dict(bundle)
        if bundle.get("kind") == "hidden_head":
            out_bundle["model_state"] = sel_model.state_dict()
        else:
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
            "world_size": world,
            "trainable_param_sum": param_sum,
            "selector_drift": round(sel_drift, 6),
            "optimizer_steps": opt_steps,
        }
        if args.limit_episodes_per_group:
            report["limit_episodes_per_group"] = args.limit_episodes_per_group
        (args.output_root / "iter_report.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False))
        print(json.dumps(report), flush=True)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
