#!/usr/bin/env python3
"""Phase 2A:sparse-history SFT —— 把 GUI-Owl 训成"会读稀疏历史"的 executor。

# note (luojiaxuan): 路线见 rl/docs/agentic_memory_rl_roadmap_20260813.md §4。
# **口径与部署逐字同源**:prompt 一律经 causalcache_agentic.policy_io
# .PolicyInputBuilder 装配(与 rollout/评测同一条代码路径),编码传
# tools=[_TOOL_SPEC](与枚举器/#26 训练器/敏感性 harness 一致),
# 目标文本用 render_official_response(与语料同源)。
#
# **决策 P2A-1(记档,可回滚)**:本阶段用 **全层 q/k/v/o LoRA r8**
# 而不是路线文档写的"后若干层":台账 §0.8/§0.9 实测过同参数量的
# 末 8 层 k/v 适配增益 ≈0,而全层 qkvo r8 拿到 +5.50pp CI[+3.00,+8.06]。
# 关键约束仍然满足:**视觉编码器全冻**(历史帧特征缓存依旧有效)、
# 非 full finetune、LoRA 的 B 零初始化 ⇒ 训练起点逐位等于原始 GUI-Owl
# (即路线文档要求的 "gate 初始≈0")。回滚:--target-modules/--last-layers。
#
# **不可解样本禁令**:只吃 solvable=True 且 trajectory_success=True 的记录
# (expert.build_sft_records 的约定),避免教模型"输入没答案也要蒙 gold"。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_records(paths: list[Path], *, only_solvable: bool = True) -> list[dict]:
    out: list[dict] = []
    stats = {"total": 0, "unsolvable": 0, "failed_traj": 0}
    for p in paths:
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            stats["total"] += 1
            if not d.get("trajectory_success", True):
                stats["failed_traj"] += 1
                continue
            if only_solvable and not d.get("solvable", True):
                stats["unsolvable"] += 1
                continue
            out.append(d)
    print(json.dumps({"loaded": len(out), **stats}, ensure_ascii=False), flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--records", nargs="+", type=Path, required=True)
    # note (luojiaxuan): SFT 记录的 action_history 只带 action_line,而官方 prompt
    # 的保留轮需要完整 tool call(render_official_response)。轨迹文件带 action dict,
    # 按 (task_id, step) join 补齐 —— 比改数据生成侧再全量重跑便宜。
    p.add_argument("--traj", nargs="+", type=Path, required=True,
                   help="expert 产出的 traj_*.jsonl,用于补齐历史动作 dict")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--last-layers", type=int, default=0,
                   help="0 = 全层(决策 P2A-1);>0 只训末 N 层")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="按 task_id 划出的内层验证集(不跨 task 泄露)")
    p.add_argument("--limit-records", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1, help="每 N 个 epoch 评一次")
    p.add_argument("--eval-limit", type=int, default=240)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--torch-seed", type=int, default=0)
    args = p.parse_args()

    import sys
    import torch

    from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime
    from causalcache.agentnet_desktop_official import render_official_response
    from causalcache_agentic.policy_io import HistoryFrameBank, PolicyInputBuilder
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rl_oracle_enumerate import action_correct, parse_tool_call
    for cand in (Path(__file__).resolve().parents[3] / "code" / "scripts",
                 Path("/data/osworld/CausalCache/code/scripts")):
        if cand.exists():
            sys.path.insert(0, str(cand))
            break
    from train_success_sft_lora import inject_lora, lora_state_dict

    recs = load_records(list(args.records))
    if not recs:
        raise SystemExit("FAILED: 没有可用记录")
    actions_by_task: dict[str, dict[int, dict]] = {}
    for tp in args.traj:
        for line in tp.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            actions_by_task.setdefault(d["task_id"], {}).update(
                {int(s["step"]): s["action"] for s in d.get("steps", [])})
    missing = [r["record_id"] for r in recs
               if any(int(h["step"]) not in actions_by_task.get(r["task_id"], {})
                      for h in r["action_history"])]
    print(json.dumps({"traj_tasks": len(actions_by_task),
                      "records_missing_actions": len(missing)}), flush=True)
    if missing:
        raise SystemExit(f"FAILED: {len(missing)} 条记录的历史动作在轨迹文件里找不到,"
                         f"例:{missing[:3]}")
    # note (luojiaxuan): 按 task_id 划分内层验证集 —— 同一条轨迹的不同决策步
    # 高度相关,按记录划分会让验证集被训练集的同轨迹样本泄露。
    task_ids = sorted({r["task_id"] for r in recs})
    random.Random(args.seed).shuffle(task_ids)
    n_val = max(1, int(len(task_ids) * args.val_frac))
    val_ids = set(task_ids[:n_val])
    train = [r for r in recs if r["task_id"] not in val_ids]
    val = [r for r in recs if r["task_id"] in val_ids]
    if args.limit_records:
        train = train[: args.limit_records]
    print(json.dumps({"train": len(train), "val": len(val),
                      "train_tasks": len(task_ids) - n_val, "val_tasks": n_val,
                      "regimes": sorted({r["regime"] for r in recs}),
                      "subset_policies": sorted({r["subset_policy"] for r in recs})},
                     ensure_ascii=False), flush=True)

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=128)
    model, device, proc = runtime.model, runtime.device, runtime.processor
    tok = proc.tokenizer
    torch.manual_seed(args.torch_seed)
    wrapped = inject_lora(
        model, rank=args.rank, alpha=args.alpha,
        target_modules=tuple(args.target_modules.split(",")),
        torch=torch, last_layer_count=args.last_layers or None)
    params = [q for w in wrapped.values() for q in (w.lora_a, w.lora_b)]
    for q in params:
        q.requires_grad_(True)
    print(json.dumps({"lora_modules": len(wrapped),
                      "lora_params": sum(q.numel() for q in params)}), flush=True)
    opt = torch.optim.AdamW(params, lr=args.learning_rate)
    opt.zero_grad(set_to_none=True)
    builder = PolicyInputBuilder(budget=int(recs[0].get("budget", 2)))

    def messages_of(r: dict) -> list[dict]:
        bank = HistoryFrameBank()
        for c in r["candidate_history"]:
            bank.add(int(c["step"]), c["screenshot"])
        bank.add(int(r["step"]), r["current_screenshot"])
        by_step = actions_by_task[r["task_id"]]
        actions = [by_step[int(h["step"])] for h in r["action_history"]]
        return builder.build(r["instruction"], actions, r["shown_subset"],
                             bank, int(r["step"]))

    def encode(r: dict):
        msgs = messages_of(r)
        enc = proc.apply_chat_template(
            msgs, tools=[_TOOL_SPEC], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        target = render_official_response(r["expert_tool_call"])
        tgt_ids = tok(target, add_special_tokens=False,
                      return_tensors="pt")["input_ids"]
        close = torch.tensor([[runtime.generation_tokens.tool_call_close_token_id]])
        tgt = torch.cat([tgt_ids, close], dim=1)
        input_ids = torch.cat([enc["input_ids"], tgt], dim=1)
        labels = torch.cat([torch.full_like(enc["input_ids"], -100), tgt], dim=1)
        out = {"input_ids": input_ids,
               "attention_mask": torch.ones_like(input_ids), "labels": labels}
        for k, v in enc.items():
            if k in ("input_ids", "attention_mask"):
                continue
            out[k] = (torch.cat([v, torch.zeros_like(tgt)], dim=1)
                      if k == "mm_token_type_ids" else v)
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in out.items()}

    @torch.no_grad()
    def evaluate(rows: list[dict]) -> dict:
        """teacher-forced 单步动作正确率(closed-loop 由 harness 负责)。"""
        model.eval()
        agg: dict[str, list[int]] = {}
        n_err = 0
        for r in rows[: args.eval_limit]:
            try:
                msgs = messages_of(r)
                enc = proc.apply_chat_template(
                    msgs, tools=[_TOOL_SPEC], tokenize=True,
                    add_generation_prompt=True, return_dict=True,
                    return_tensors="pt").to(device)
                pt_len = int(enc["input_ids"].shape[1])
                gt = runtime.generation_tokens
                o = model.generate(**enc, do_sample=False, max_new_tokens=128,
                                   eos_token_id=gt.tool_call_close_token_id,
                                   pad_token_id=gt.pad_token_id,
                                   suppress_tokens=list(gt.standard_eos_token_ids),
                                   num_beams=1, num_return_sequences=1)
                pred = parse_tool_call(proc.batch_decode(
                    o[:, pt_len:], skip_special_tokens=False)[0])
                ok = int(action_correct(pred, r["expert_action"], tolerance=25.0))
            except (ValueError, KeyError, OSError, RuntimeError):
                n_err += 1
                continue
            for key in ("ALL", f"regime={r['regime']}",
                        f"subset={r['subset_policy_effective']}"):
                agg.setdefault(key, []).append(ok)
        model.train()
        return {"errors": n_err,
                **{k: {"n": len(v), "acc": round(sum(v) / max(len(v), 1), 4)}
                   for k, v in sorted(agg.items())}}

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"eval_before": evaluate(val)}, ensure_ascii=False), flush=True)

    stats = {"seen": 0, "steps": 0, "loss": 0.0, "skipped": {}}
    for ep in range(args.epochs):
        order = list(train)
        random.Random(args.seed + ep).shuffle(order)
        for r in order:
            try:
                ex = encode(r)
                loss = model(**ex).loss
                (loss / args.grad_accum).backward()
                stats["seen"] += 1
                stats["loss"] += float(loss.detach())
                if stats["seen"] % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    stats["steps"] += 1
                if stats["seen"] % 100 == 0:
                    print(json.dumps({"epoch": ep, "seen": stats["seen"],
                                      "opt_steps": stats["steps"],
                                      "mean_loss": round(stats["loss"] / stats["seen"], 4)},
                                     ensure_ascii=False), flush=True)
            except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
                k = type(exc).__name__ if not str(exc) else str(exc)[:60]
                stats["skipped"][k] = stats["skipped"].get(k, 0) + 1
                opt.zero_grad(set_to_none=True)
                continue
        bundle = {"arm": "mem_sft_a", "rank": args.rank, "alpha": args.alpha,
                  "target_modules": args.target_modules,
                  "last_layers": args.last_layers, "epoch": ep,
                  "torch_seed": args.torch_seed,
                  "state": lora_state_dict(wrapped)}
        torch.save(bundle, args.output_root / f"adapter_ep{ep}.pt")
        ev = evaluate(val) if (ep + 1) % args.eval_every == 0 else {}
        print(json.dumps({"epoch_end": ep,
                          "mean_loss": round(stats["loss"] / max(stats["seen"], 1), 4),
                          "eval": ev, "skipped": stats["skipped"]},
                         ensure_ascii=False), flush=True)
    torch.save(bundle, args.output_root / "adapter.pt")
    n_skip = sum(stats["skipped"].values())
    print(json.dumps({"final": True, "seen": stats["seen"],
                      "opt_steps": stats["steps"], "skipped": stats["skipped"]},
                     ensure_ascii=False))
    # note (luojiaxuan): 跳过率守卫 —— 台账 §6 第 18 条前科(99.6% OOM 静默跳过
    # 却照常存 adapter,下游把没训过的当正品评)。
    if n_skip > 0.1 * max(stats["seen"] + n_skip, 1):
        raise SystemExit(f"FAILED: 跳过率 {n_skip}/{stats['seen'] + n_skip} 超过 10%")


if __name__ == "__main__":
    main()
