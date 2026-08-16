#!/usr/bin/env python3
"""Phase 4A:joint GRPO —— selector 与 executor adapter 同时被任务奖励塑造。

# note (luojiaxuan): 路线见 agentic_memory_rl_roadmap_20260813.md §5;
# 做这一步的直接依据是 Phase 3 的诊断(审计 §4.7):闭环瓶颈已不在 selector
# (它追平了两种独立构造的 oracle),而在 executor —— 需老帧任务里 28.4% 的
# rollout **证据从未上屏**(探索失败,记忆天花板=0),证据上屏的部分也只有
# 20.6% 走完全程(而 teacher-forced 给对帧时是 98.6%)。
#
# **本阶段的设计约束(逐条对应路线 §5)**:
#   * 初始化 = policy_mem_sft(executor)+ Phase 3 最佳 selector,
#     **不从两个随机组件同时学**;
#   * 4A 只解冻 selector 全部参数 + executor 的**后 N 层 LoRA**;
#     视觉编码器与多数 backbone 保持冻结(历史帧特征缓存因此仍然有效);
#   * 一步的联合动作是 (S_t, a_t),log π = log π_sel + log π_pol,
#     **共用同一个 trajectory-level group advantage**;
#   * 奖励只有终局 0/1;
#   * 稳定手段:两组独立学习率、对 policy_mem_sft 的可选 KL、
#     交替更新(--alternate)、grad-norm/KL/entropy 分开监控。
#
# **命门:logprob 一致性自检**。selector 侧那条检查(max|Δ|=2.6e-07)救过命;
# executor 侧的分布必须与采样时**逐位同构**(同温度、同 suppress 屏蔽),
# 否则 ratio 全错而日志看上去一切正常。启动时强制自检,超阈直接退出。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def load_groups(paths: list[Path], min_group: int = 2) -> list[dict]:
    """按 (task_id, group_id) 聚合 rollout。"""
    buckets: dict[tuple[str, int], list[dict]] = {}
    for p in paths:
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # note (luojiaxuan): group_id 在 rollout schema 里是**字符串**
            # (形如 "<template>::<regime>::<seed>"),不要强转 int —— 第一次
            # 就是这么崩的。原样当 key 用即可。
            buckets.setdefault((d["task_id"], str(d.get("group_id", ""))),
                               []).append(d)
    out = []
    for (tid, gid), rolls in sorted(buckets.items()):
        if len(rolls) < min_group:
            continue
        out.append({"task_id": tid, "group_id": gid, "rollouts": rolls})
    return out


def advantages(rewards: list[float], eps: float = 1e-6):
    m = sum(rewards) / len(rewards)
    var = sum((r - m) ** 2 for r in rewards) / len(rewards)
    sd = math.sqrt(var)
    if sd < 1e-9:
        return None
    return [(r - m) / (sd + eps) for r in rewards]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", nargs="+", type=Path, required=True)
    p.add_argument("--selector-ckpt", type=Path, required=True)
    p.add_argument("--executor-adapter", type=Path, required=True,
                   help="policy_mem_sft 的 LoRA(Phase 2 产物),作为起点与 π_ref")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--out-selector", type=Path, required=True)
    p.add_argument("--out-adapter", type=Path, required=True)
    p.add_argument("--exec-last-layers", type=int, default=8,
                   help="4A:executor 只解冻后 N 层的 LoRA(0=全层,谨慎)")
    p.add_argument("--lr-selector", type=float, default=2e-4)
    p.add_argument("--lr-executor", type=float, default=2e-5)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--kl-coef", type=float, default=0.0,
                   help=">0 时对 policy_mem_sft 做 KL 正则(需额外一次前向)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--alternate", type=int, default=0,
                   help=">0 时每 N 个优化步只更新一侧(交替更新)")
    p.add_argument("--executor-temperature", type=float, default=1.0)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--limit-groups", type=int, default=0)
    p.add_argument("--consistency-checks", type=int, default=24)
    p.add_argument("--consistency-tol", type=float, default=0.05)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260816)
    args = p.parse_args()

    import sys
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for cand in (Path(__file__).resolve().parents[1],
                 Path("/data/agentic/code")):
        if (cand / "causalcache_agentic").exists():
            sys.path.insert(0, str(cand))
            break
    for cand in (Path(__file__).resolve().parents[3] / "code" / "scripts",
                 Path("/data/osworld/CausalCache/code/scripts")):
        if cand.exists():
            sys.path.insert(0, str(cand))
            break

    from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime
    from causalcache_agentic.features import build_extractor, FeatureCache
    from causalcache_agentic.policy_io import HistoryFrameBank, PolicyInputBuilder
    from causalcache_agentic.selector_model import (
        SelectorStateBuilder,
        SubsetSelectorPolicy,
    )
    from train_success_sft_lora import inject_lora, load_lora_state_dict, lora_state_dict

    groups = load_groups(list(args.rollouts))
    if args.limit_groups:
        groups = groups[: args.limit_groups]
    n_eff = sum(1 for g in groups
                if advantages([float(bool(r.get("success")))
                               for r in g["rollouts"]]) is not None)
    print(json.dumps({"groups": len(groups), "effective": n_eff,
                      "frac_effective": round(n_eff / max(len(groups), 1), 4)},
                     ensure_ascii=False), flush=True)

    dev = torch.device(args.device)
    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir, expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device, effective_visual_tokens_per_image=2560,
        max_new_tokens=128)
    model, proc = runtime.model, runtime.processor
    gt = runtime.generation_tokens
    bundle = torch.load(args.executor_adapter, map_location="cpu",
                        weights_only=False)
    wrapped_all = inject_lora(
        model, rank=int(bundle["rank"]), alpha=int(bundle["alpha"]),
        target_modules=tuple(str(bundle.get(
            "target_modules", "q_proj,k_proj,v_proj,o_proj")).split(",")),
        torch=torch,
        last_layer_count=(int(bundle["last_layers"])
                          if bundle.get("last_layers") else None))
    load_lora_state_dict(wrapped_all, bundle["state"])
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 4A:只解冻后 N 层的 LoRA;其余(含视觉塔)保持冻结
    import re as _re
    layer_re = _re.compile(r"\.layers\.(\d+)\.")
    layers = sorted({int(m.group(1)) for name in wrapped_all
                     if (m := layer_re.search(f".{name}."))})
    keep = set(layers[-args.exec_last_layers:]) if args.exec_last_layers else set(layers)
    exec_params = []
    for name, w in wrapped_all.items():
        m = layer_re.search(f".{name}.")
        train_it = (m is not None and int(m.group(1)) in keep)
        for q in (w.lora_a, w.lora_b):
            q.requires_grad_(bool(train_it))
            if train_it:
                exec_params.append(q)
    print(json.dumps({"lora_modules": len(wrapped_all),
                      "trainable_exec_params": sum(q.numel() for q in exec_params),
                      "unfrozen_layers": sorted(keep)}), flush=True)

    selector = SubsetSelectorPolicy.load(str(args.selector_ckpt), map_location=dev)
    selector.to(dev).train()
    sel_params = [q for q in selector.parameters() if q.requires_grad]
    state_builder = SelectorStateBuilder(model_dir=str(args.model_dir), device=dev)
    extractor = build_extractor(argparse.Namespace(
        dry_run=False, model_dir=args.model_dir,
        snapshot_manifest=args.snapshot_manifest, device=args.device,
        visual_tokens=2560, feature_dtype="float16"), runtime=runtime)
    cache = FeatureCache(extractor, max_items=32)

    opt = torch.optim.AdamW(
        [{"params": sel_params, "lr": args.lr_selector},
         {"params": exec_params, "lr": args.lr_executor}], weight_decay=0.0)
    opt.zero_grad(set_to_none=True)
    builder = PolicyInputBuilder(budget=args.budget)
    suppress = list(gt.standard_eos_token_ids)

    def exec_logprob(step: dict, rec: dict, task_instr: str) -> "torch.Tensor":
        """teacher-forced 重算 log π_pol(与采样时同一套 processed 分布)。"""
        toks = step.get("action_tokens")
        if not toks:
            return None
        bank = HistoryFrameBank()
        for s in rec["steps"]:
            if int(s["step"]) <= int(step["step"]):
                bank.add(int(s["step"]), s["screenshot"])
        hist = [dict(s["action"]) for s in rec["steps"]
                if int(s["step"]) < int(step["step"])]
        msgs = builder.build(task_instr, hist, step["chosen_subset"], bank,
                             int(step["step"]))
        enc = proc.apply_chat_template(
            msgs, tools=[_TOOL_SPEC], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(dev)
        tgt = torch.tensor([toks], device=dev)
        input_ids = torch.cat([enc["input_ids"], tgt], dim=1)
        kw = {k: v for k, v in enc.items() if k not in ("input_ids",
                                                        "attention_mask")}
        if "mm_token_type_ids" in kw:
            kw["mm_token_type_ids"] = torch.cat(
                [kw["mm_token_type_ids"], torch.zeros_like(tgt)], dim=1)
        out = model(input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    logits_to_keep=len(toks) + 1, **kw)
        logits = out.logits[0, :-1].float() / max(args.executor_temperature, 1e-6)
        if suppress:
            logits[:, suppress] = float("-inf")
        lp = torch.log_softmax(logits, dim=-1)
        return lp.gather(-1, tgt[0].unsqueeze(-1)).sum()

    # ---- 启动自检:重算 log π 必须与采样时记录的值一致 ----
    checked = worst = 0
    diffs = []
    with torch.no_grad():
        for g in groups:
            for rec in g["rollouts"]:
                for st in rec["steps"]:
                    if st.get("policy_logprob") is None or checked >= args.consistency_checks:
                        continue
                    v = exec_logprob(st, rec, rec.get("instruction", ""))
                    if v is None:
                        continue
                    d = abs(float(v) - float(st["policy_logprob"]))
                    diffs.append(d)
                    checked += 1
                if checked >= args.consistency_checks:
                    break
            if checked >= args.consistency_checks:
                break
    if not diffs:
        raise SystemExit("FAILED: rollout 里没有 policy_logprob —— "
                         "请用 --executor-sample 重新采数据")
    mx = max(diffs)
    print(json.dumps({"exec_logprob_check": {"n": len(diffs),
                                             "max_abs_diff": round(mx, 6),
                                             "mean_abs_diff": round(
                                                 sum(diffs) / len(diffs), 6),
                                             "tol": args.consistency_tol}},
                     ensure_ascii=False), flush=True)
    if mx > args.consistency_tol:
        raise SystemExit(
            f"FAILED: executor logprob 重算与采样值不一致(max|Δ|={mx:.4f} > "
            f"{args.consistency_tol}) —— ratio 会全错,先修分布口径再训")

    rng = random.Random(args.seed)
    stats = {"groups_used": 0, "steps": 0, "opt_steps": 0, "skipped_groups": 0}
    meters: dict[str, list[float]] = {}

    def meter(k: str, v: float) -> None:
        meters.setdefault(k, []).append(float(v))

    pending = 0
    for ep in range(args.epochs):
        order = list(range(len(groups)))
        rng.shuffle(order)
        for gi in order:
            g = groups[gi]
            rewards = [float(bool(r.get("success"))) for r in g["rollouts"]]
            adv = advantages(rewards)
            if adv is None:
                stats["skipped_groups"] += 1
                continue
            stats["groups_used"] += 1
            live = [(r, a) for r, a in zip(g["rollouts"], adv) if r.get("steps")]
            n_roll = len(live)
            if not n_roll:
                continue
            for rec, a_i in live:
                n_step = max(len(rec["steps"]), 1)
                for st in rec["steps"]:
                    if st.get("policy_logprob") is None:
                        continue
                    bank = HistoryFrameBank()
                    for s in rec["steps"]:
                        if int(s["step"]) <= int(st["step"]):
                            bank.add(int(s["step"]), s["screenshot"])
                    cands = [int(x) for x in st["candidates"]]
                    hist_lines = [s["action_line"] for s in rec["steps"]
                                  if int(s["step"]) < int(st["step"])]
                    hist_acts = [dict(s["action"]) for s in rec["steps"]
                                 if int(s["step"]) < int(st["step"])]
                    sr = state_builder.build(
                        selector=selector, task_instruction=rec.get("instruction", ""),
                        history_actions=hist_acts, history_lines=hist_lines,
                        bank=bank, current_step=int(st["step"]),
                        candidates=cands, budget=args.budget,
                        feature_provider=cache)
                    lp_sel = selector.logprob_of(
                        sr, cands, args.budget,
                        tuple(int(x) for x in st["chosen_subset"]))
                    lp_pol = exec_logprob(st, rec, rec.get("instruction", ""))
                    if lp_pol is None:
                        continue
                    old = float(st["selector_logprob"]) + float(st["policy_logprob"])
                    ratio = torch.exp(lp_sel + lp_pol - old)
                    unc = ratio * a_i
                    cl = torch.clamp(ratio, 1 - args.clip_eps,
                                     1 + args.clip_eps) * a_i
                    term = torch.min(unc, cl)
                    ent = selector.entropy(sr, cands, args.budget)
                    term = term + args.ent_coef * ent
                    scale = 1.0 / (n_roll * n_step * max(args.grad_accum, 1))
                    (-(term) * scale).backward()
                    stats["steps"] += 1
                    meter("ratio", float(ratio.detach()))
                    meter("clip_frac", 1.0 if abs(float(ratio.detach()) - 1) >
                          args.clip_eps else 0.0)
                    meter("entropy", float(ent.detach()))
                    meter("adv_abs", abs(a_i))
                    meter("reward", sum(rewards) / len(rewards))
            pending += 1
            if pending >= args.grad_accum:
                gn_s = torch.nn.utils.clip_grad_norm_(sel_params, 1.0)
                gn_e = torch.nn.utils.clip_grad_norm_(exec_params, 1.0)
                if args.alternate:
                    turn = (stats["opt_steps"] // args.alternate) % 2
                    for q in (sel_params if turn else exec_params):
                        q.grad = None
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending = 0
                stats["opt_steps"] += 1
                meter("grad_norm_selector", float(gn_s))
                meter("grad_norm_executor", float(gn_e))
                if stats["opt_steps"] % 5 == 0:
                    print(json.dumps({"opt_step": stats["opt_steps"],
                                      **{k: round(sum(v) / len(v), 4)
                                         for k, v in meters.items() if v}},
                                     ensure_ascii=False), flush=True)

    args.out_selector.parent.mkdir(parents=True, exist_ok=True)
    selector.save(str(args.out_selector))
    torch.save({**{k: v for k, v in bundle.items() if k != "state"},
                "state": lora_state_dict(wrapped_all),
                "joint_phase": "4A",
                "exec_last_layers": args.exec_last_layers},
               args.out_adapter)
    print(json.dumps({"final": True, **stats,
                      **{k: round(sum(v) / len(v), 4) for k, v in meters.items() if v},
                      "out_selector": str(args.out_selector),
                      "out_adapter": str(args.out_adapter)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
