#!/usr/bin/env python3
"""v3:离线视频步级联合训练(selector + HGKV),不开虚拟机。

# note (luojiaxuan): 2026-08-08 用户裁定——"不要傻啦吧唧的用 OSWorld 做 RL,
# 太慢,还得开模拟器,而且奖励过于稀疏,用离线视频的每一步来做监督信号"。
# OSWorld 降级为**评测环境**,不再进训练循环。
#
# 为什么不能直接端到端 SFT:选帧是**离散**的(top-B / 采样),
#   分数 = selector(...)      ← 可导
#   S    = 采样(分数)          ← ✗ 不可导
#   loss = CE(policy(prompt(S)), gold)
# ∂loss/∂分数 几乎处处为 0,天真做法只会训到 HGKV,selector 原地不动。
#
# 解法 = 两条通道分治(与 GRPO 同构,但奖励从"整条 episode 成败"换成
# "这一步的动作对数概率",组从"同任务 G 条 rollout"换成"同一 state 的 K 次采样"):
#   通道 A(selector):score-function 估计量绕过离散
#       ∇ = −(r_i − 组均值) · ∇log π_sel(S_i)
#   通道 B(HGKV):普通可导 SFT
#       ∇ = ∇(−log p(gold | prompt(S_i)))
#
# 奖励用 **teacher-forced log p(gold action)** 而不是"是否正确":
#   * 连续、稠密,不用生成(一次前向即可),比二值信号快且梯度平滑;
#   * 组内比较的是**同一个 gold action 在不同选帧下的概率**,坐标容差问题
#     在相对比较中大部分抵消(容差只影响绝对值,不影响"哪个子集更好");
#   * "是否正确"(类型匹配+25px)保留给**评测**,不进训练损失。
#
# 信号量对比:OSWorld 闭环 936 个 episode 级奖励 → 这里 25,929 个步级奖励。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--selector-bundle", type=Path, required=True,
                   help="hidden_head bundle(make_hidden_head.py 产出或上轮权重)")
    p.add_argument("--adapter-config", type=Path, required=True)
    p.add_argument("--resume-adapter", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--group-k", type=int, default=4,
                   help="每个 state 采样几个子集(组基线的组大小)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--selector-learning-rate", type=float, default=1e-3)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--adv-std-eps", type=float, default=0.1)
    p.add_argument("--w-sel", type=float, default=1.0)
    p.add_argument("--w-act", type=float, default=1.0)
    p.add_argument("--entropy-lambda", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-states", type=int, default=500)
    p.add_argument("--max-candidates", type=int, default=30)
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--index-visual-tokens", type=int, default=144)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260808)
    args = p.parse_args()

    import torch

    from causalcache.agentnet_desktop_official import (
        build_desktop_official_messages,
        official_step_forms,
    )
    from transformers import AutoProcessor

    from causalcache.osworld_gui_owl import (
        _TOOL_SPEC,
        VISION_PATCH_SIZE,
        VISION_SPATIAL_MERGE_SIZE,
        GUIOwlOSWorldRuntime,
    )

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=128,
    )
    model, processor, device = runtime.model, runtime.processor, runtime.device

    # note (luojiaxuan): 索引遍要低清缩略图,但分辨率是 processor 构造时通过
    # min/max_pixels 定死的、不是可改属性(改属性会 AttributeError)。因此
    # **单独构造一个低清 processor**,与动作遍的高清 processor 并存,互不干扰。
    _px = args.index_visual_tokens * (VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE) ** 2
    index_processor = AutoProcessor.from_pretrained(
        args.model_dir, min_pixels=_px, max_pixels=_px, local_files_only=True)

    # ---- selector 头 ----
    bundle = torch.load(args.selector_bundle, map_location="cpu", weights_only=False)
    if bundle.get("kind") != "hidden_head":
        raise SystemExit("只支持 hidden_head bundle")
    head = torch.nn.Linear(int(bundle["hidden_size"]), 1)
    head.load_state_dict(bundle["model_state"])
    head.to(device).train()

    # ---- HGKV LoRA 注入(与 v4/v7 及 RL 同形制)----
    from causalcache.policy.history_gated_lora import (
        history_gated_state_dict,
        inject_history_gated_kv,
        load_history_gated_state_dict,
    )
    cfg = json.loads(args.adapter_config.read_text())
    scope = str(cfg.get("layer_scope", "last_8"))
    layer_count = int(scope.split("_")[-1]) if scope.startswith("last_") else 8
    # note (luojiaxuan): 注入返回的是 {模块名: HistoryGatedKVLinear} 字典,
    # 不是参数列表;可学参数是每个 wrapper 的 lora_a / lora_b。
    wrapped = inject_history_gated_kv(
        model, layer_count=layer_count,
        rank=int(cfg.get("rank", 8)), alpha=int(cfg.get("alpha", 16)))
    if args.resume_adapter:
        load_history_gated_state_dict(
            wrapped, torch.load(args.resume_adapter, map_location="cpu"))
    adapter_params = [q for lora in wrapped.values()
                      for q in (lora.lora_a, lora.lora_b)]

    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": args.selector_learning_rate},
        {"params": adapter_params, "lr": args.learning_rate},
    ])

    def gold_logprob(messages: list[dict[str, Any]], target: str) -> Any:
        """teacher-forced log p(gold action);返回可导标量。"""
        enc = processor.apply_chat_template(
            messages, tools=[_TOOL_SPEC], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(device)
        tgt = processor.tokenizer(target, return_tensors="pt",
                                  add_special_tokens=False).input_ids.to(device)
        input_ids = torch.cat([enc["input_ids"], tgt], dim=1)
        kwargs = {k: v for k, v in enc.items() if k != "input_ids"}
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = torch.cat(
                [kwargs["attention_mask"], torch.ones_like(tgt)], dim=1)
        if "mm_token_type_ids" in kwargs:   # Qwen3-VL:目标段补 0
            kwargs["mm_token_type_ids"] = torch.cat(
                [kwargs["mm_token_type_ids"], torch.zeros_like(tgt)], dim=1)
        out = model(input_ids=input_ids, **kwargs)
        logits = out.logits[:, enc["input_ids"].shape[1] - 1: -1, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()

    stats: dict[str, float] = {"states": 0, "steps": 0, "reward": 0.0,
                               "ent": 0.0, "skipped": 0}
    w0 = [q.detach().clone() for q in head.parameters()]
    optimizer.zero_grad(set_to_none=True)
    n_state = 0

    for line in args.manifest.open(encoding="utf-8"):
        if n_state >= args.max_states:
            break
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        try:
            s = int(rec["step"])
            images = rec["image_relpaths"]
            if len(images) != s:
                raise ValueError("image count")
            screen = tuple(int(x) for x in rec["screen_size"])
            steps = [official_step_forms(h, screen_size=screen)
                     for h in rec["history"]]
            cands = [j for j in range(1, s - 1) if steps[j].full_response]
            if not (args.budget <= len(cands) <= args.max_candidates):
                raise ValueError("candidate count")
            paths = [args.image_root / images[j] for j in cands]
            cur = args.image_root / images[s - 1]
            if not all(q.exists() for q in paths + [cur]):
                raise ValueError("missing images")
            ev = {j: str(args.image_root / images[j]) for j in cands}
            target = str(rec["target_text"])

            # --- 索引遍:一次前向拿各候选帧的 pooled hidden(selector 特征)---
            feats = index_features(rec, steps, cands, ev, str(cur),
                                   model=model, processor=index_processor,
                                   device=device, torch=torch,
                                   build=build_desktop_official_messages,
                                   tool_spec=_TOOL_SPEC)
            scores = head(feats).squeeze(-1)

            # --- 通道 A 采样 K 个子集 + 通道 B 打分 ---
            rewards, logps, ents = [], [], []
            for _ in range(args.group_k):
                sel, lp, ent = pl_sample(scores, args.budget,
                                         args.temperature, torch, rng)
                subset = sorted(cands[i] for i in sel)
                msgs = build_desktop_official_messages(
                    goal=rec["instruction"], steps=steps,
                    shown_events=subset,
                    event_images={j: ev[j] for j in subset},
                    current_image=str(cur))
                r = gold_logprob(msgs, target)
                rewards.append(r)
                logps.append(lp)
                ents.append(ent)

            rv = torch.stack([r.detach() for r in rewards])
            adv = (rv - rv.mean()) / (rv.std(unbiased=False) + args.adv_std_eps)
            loss = torch.zeros((), device=device)
            for a, lp, ent, r in zip(adv, logps, ents, rewards):
                loss = loss - args.w_sel * a * lp - args.entropy_lambda * ent
                loss = loss - args.w_act * r / args.group_k   # HGKV 侧 SFT
            (loss / args.grad_accum).backward()

            stats["states"] += 1
            stats["reward"] += float(rv.mean())
            stats["ent"] += float(torch.stack([e.detach() for e in ents]).mean())
            n_state += 1
            if n_state % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(head.parameters()) + list(adapter_params), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                stats["steps"] += 1
            if n_state % args.report_every == 0:
                with torch.no_grad():
                    num = sum(float((q - z).pow(2).sum())
                              for q, z in zip(head.parameters(), w0)) ** 0.5
                    den = sum(float(z.pow(2).sum()) for z in w0) ** 0.5
                print(json.dumps({
                    "states": n_state, "opt_steps": int(stats["steps"]),
                    "mean_reward_logp": stats["reward"] / max(stats["states"], 1),
                    "mean_entropy": stats["ent"] / max(stats["states"], 1),
                    "selector_drift": round(num / den, 6),
                    "skipped": int(stats["skipped"]),
                }, ensure_ascii=False), flush=True)
        except (ValueError, KeyError, OSError, TypeError, RuntimeError):
            stats["skipped"] += 1
            optimizer.zero_grad(set_to_none=True)
            continue

    out_bundle = dict(bundle)
    out_bundle["model_state"] = head.state_dict()
    torch.save(out_bundle, args.output_root / "selector_bundle.pt")
    torch.save(history_gated_state_dict(wrapped), args.output_root / "adapter.pt")
    with torch.no_grad():
        num = sum(float((q - z).pow(2).sum())
                  for q, z in zip(head.parameters(), w0)) ** 0.5
        den = sum(float(z.pow(2).sum()) for z in w0) ** 0.5
    report = {"states": n_state, "opt_steps": int(stats["steps"]),
              "mean_reward_logp": stats["reward"] / max(stats["states"], 1),
              "selector_drift": round(num / den, 6),
              "skipped": int(stats["skipped"]),
              "group_k": args.group_k, "budget": args.budget}
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))


def pl_sample(scores, budget: int, temperature: float, torch, rng):
    """Plackett-Luce 无放回顺序采样;返回 (下标, log π, 熵)。"""
    avail = list(range(scores.shape[0]))
    chosen, logp, ent = [], None, None
    for _ in range(budget):
        idx = torch.tensor(avail, device=scores.device)
        lp = torch.log_softmax(scores[idx] / temperature, dim=0)
        e = -(lp.exp() * lp).sum()
        pick = int(torch.multinomial(lp.exp().detach(), 1).item())
        logp = lp[pick] if logp is None else logp + lp[pick]
        ent = e if ent is None else ent + e
        chosen.append(avail[pick])
        avail.pop(pick)
    return chosen, logp, ent


def index_features(rec, steps, cands, ev, cur, *, model, processor, device,
                   torch, build, tool_spec):
    """索引遍:全部候选帧低清 + 当前屏过一次冻结前向,取各图 token 段 mean-pool。

    # note (luojiaxuan): 与 serve 的 hidden selector 同一机制(选择是场景级的,
    # 缩略图够用;动作遍才需要全分辨率)。此处 no_grad —— v1 边界:只训打分头,
    # 索引遍不挂 LoRA。
    """
    msgs = build(goal=rec["instruction"], steps=steps, shown_events=list(cands),
                 event_images={j: ev[j] for j in cands}, current_image=cur)
    enc = processor.apply_chat_template(
        msgs, tools=[tool_spec], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states[-1][0]
    mm = enc.get("mm_token_type_ids")
    if mm is None:
        raise ValueError("no mm_token_type_ids")
    mask = (mm[0] == 1)
    # 连续 image token 段 → 每段一图;段数须等于 候选数+1(末段是当前屏)
    segs, start = [], None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    if len(segs) != len(cands) + 1:
        raise ValueError(f"segment count {len(segs)} != {len(cands) + 1}")
    return torch.stack([hs[a:b].mean(dim=0) for a, b in segs[:-1]]).float()


if __name__ == "__main__":
    main()
