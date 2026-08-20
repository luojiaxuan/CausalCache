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


def load_real_rows(manifest: Path, image_root: Path, *, budget: int,
                   keep_incorrect: bool) -> tuple[list[dict], dict[str, int]]:
    """AgentNet manifest → 训练记录(kind=real)。

    # note (luojiaxuan): 过滤纪律照抄 rl_oracle_enumerate:图片数与 step 不符、
    # history 不可解析、候选为空、**缺任何一张所需图整行跳过**并分类计数 ——
    # 只丢缺图候选会让 shown 系统性偏移。shown = 候选池的 recent 尾(recent-B),
    # 与部署口径一致;target 走 render_official_response(与合成腿同一序列化,
    # close token 由 encode 统一追加,不用 manifest 的 target_text 以免双闭合)。
    """
    from causalcache.agentnet_desktop_official import official_step_forms

    rows: list[dict] = []
    skipped: dict[str, int] = {}

    def skip(key: str) -> None:
        skipped[key] = skipped.get(key, 0) + 1

    for line in manifest.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if not keep_incorrect and not rec.get("target_step_last_step_correct"):
            skip("target_marked_incorrect")
            continue
        s = int(rec["step"])
        images = rec["image_relpaths"]
        if len(images) != s:
            skip("image_count_mismatch")
            continue
        try:
            screen = tuple(int(x) for x in rec["screen_size"])
            forms = [official_step_forms(h, screen_size=screen)
                     for h in rec["history"]]
        except (ValueError, KeyError):
            skip("unparseable_history")
            continue
        if len(forms) != s - 1:
            skip("history_len_mismatch")
            continue
        cands = [j for j in range(1, s - 1) if forms[j].full_response]
        if not cands:
            skip("too_few_candidates")
            continue
        shown = cands[-budget:]
        needed = [image_root / images[j] for j in shown] + [image_root / images[s - 1]]
        if not all(q.exists() for q in needed):
            skip("missing_images")
            continue
        rows.append({
            "kind": "real",
            "task_id": f"real::{rec['task_id']}",
            "record_id": f"real::{rec['dp_id']}",
            "regime": "real_ui",
            "instruction": rec["instruction"],
            "history": rec["history"],
            "screen_size": list(rec["screen_size"]),
            "shown_subset": shown,
            "event_images": {int(j): str(image_root / images[j]) for j in shown},
            "current_screenshot": str(image_root / images[s - 1]),
            "expert_tool_call": rec["target_tool_call"],
            "step": s,
        })
    return rows, skipped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--records", nargs="+", type=Path, required=True)
    # note (luojiaxuan): bridge 混合 SFT(Phase 5)—— 真实腿。AgentNet ubuntu
    # 决策点 manifest,每行自带 history/target_tool_call/质量位;messages 构造与
    # rl_oracle_enumerate 逐字同一条路径(official_step_forms +
    # build_desktop_official_messages),与部署同源。动机:纯合成 SFT 在
    # OSWorld 135 任务上 -13.04pp 灾难遗忘(roadmap 2026-08-20)。
    p.add_argument("--real-manifest", type=Path, default=None)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--real-mix", type=float, default=1.0,
                   help="真实:合成 样本量比(每 epoch 重采真实腿到该比例)")
    p.add_argument("--real-budget", type=int, default=2,
                   help="真实腿 shown = 候选池 recent 尾(与部署 B 对齐)")
    p.add_argument("--real-keep-incorrect", action="store_true",
                   help="默认只取 target_step_last_step_correct=True 的行")
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
    # note (luojiaxuan): 中途存档 —— 共享机上长跑不能等到 epoch 末才落盘,
    # 被抢卡/OOM 就前功尽弃(台账 §6 的老教训)。
    p.add_argument("--save-every", type=int, default=200,
                   help="每 N 条样本存一次 adapter_latest.pt(0=关)")
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
    real_all: list[dict] = []
    if args.real_manifest is not None:
        real_all, real_skipped = load_real_rows(
            args.real_manifest, args.image_root, budget=args.real_budget,
            keep_incorrect=args.real_keep_incorrect)
        print(json.dumps({"real_rows": len(real_all),
                          "real_skipped": real_skipped},
                         ensure_ascii=False), flush=True)

    task_ids = sorted({r["task_id"] for r in recs})
    random.Random(args.seed).shuffle(task_ids)
    n_val = max(1, int(len(task_ids) * args.val_frac))
    val_ids = set(task_ids[:n_val])
    train = [r for r in recs if r["task_id"] not in val_ids]
    val = [r for r in recs if r["task_id"] in val_ids]
    # note (luojiaxuan): 真实腿同样按 task_id 切内层验证集(同轨迹步高度相关),
    # 训练侧每 epoch 重采样到 real_mix × 合成腿大小;验证侧固定,分开汇报
    # real/synthetic 两条精度,混合是否"兼得"直接看这两个数。
    real_train: list[dict] = []
    val_real: list[dict] = []
    if real_all:
        real_tids = sorted({r["task_id"] for r in real_all})
        random.Random(args.seed + 1).shuffle(real_tids)
        n_rv = max(1, int(len(real_tids) * args.val_frac))
        rv_ids = set(real_tids[:n_rv])
        real_train = [r for r in real_all if r["task_id"] not in rv_ids]
        val_real = [r for r in real_all if r["task_id"] in rv_ids]
        print(json.dumps({"real_train": len(real_train),
                          "real_val": len(val_real)}), flush=True)
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

    from causalcache.agentnet_desktop_official import (
        build_desktop_official_messages,
        official_step_forms,
    )

    def messages_of(r: dict) -> list[dict]:
        if r.get("kind") == "real":
            screen = tuple(int(x) for x in r["screen_size"])
            forms = [official_step_forms(h, screen_size=screen)
                     for h in r["history"]]
            return build_desktop_official_messages(
                goal=r["instruction"], steps=forms,
                shown_events=list(r["shown_subset"]),
                event_images={int(k): v for k, v in r["event_images"].items()},
                current_image=r["current_screenshot"])
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
        if real_train:
            # note (luojiaxuan): 每 epoch 从真实腿重采 real_mix×|合成腿| 条混入。
            # 有放回与否取决于池子大小:池大于需求就无放回采样,否则整池重复+
            # 截断。合成腿全保留 —— 记忆行为是本工作的目标能力,真实腿是"别忘"。
            need = int(len(train) * args.real_mix)
            rng_r = random.Random(args.seed + 1000 + ep)
            if need <= len(real_train):
                order += rng_r.sample(real_train, need)
            else:
                pool = real_train * (need // len(real_train) + 1)
                rng_r.shuffle(pool)
                order += pool[:need]
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
                if args.save_every and stats["seen"] % args.save_every == 0:
                    torch.save({"arm": "mem_sft_a", "rank": args.rank,
                                "alpha": args.alpha,
                                "target_modules": args.target_modules,
                                "last_layers": args.last_layers, "epoch": ep,
                                "seen": stats["seen"],
                                "torch_seed": args.torch_seed,
                                "state": lora_state_dict(wrapped)},
                               args.output_root / "adapter_latest.pt")
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
        if val_real and (ep % args.eval_every == 0 or ep == args.epochs - 1):
            mr = evaluate(val_real)
            print(json.dumps({"epoch": ep, "val_real": mr},
                             ensure_ascii=False), flush=True)
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
