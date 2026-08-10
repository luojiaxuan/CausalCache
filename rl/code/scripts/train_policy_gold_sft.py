#!/usr/bin/env python3
"""#26:hopeless 态上的 gold-action SFT(改策略,让历史变得可读)。

# note (luojiaxuan): selector 侧已穷尽(阶梯 1-3 + 2.5 全清特征:可见性损失
# 5pp 属实但补齐后停在 recency)——"哪帧有用"的增量信息不在冻结策略的任何
# 可及表示里。剩下的唯一活口是**改策略**:如果适配后的策略能把 recent-2 里
# 已有的信息用起来,记忆线就还活着。
#
# **目标是 gold action 监督,不是 DiD**。旧 DiD 用"历史里出现过类似动作"
# 筛正例帧 —— 用动作相似度冒充因果作用,已被用户点名并在台账判死。
# 这里的监督就是人标的下一步动作,损失只打在助手回复段上。
#
# **训练打在 hopeless 分层**(枚举证明任何 2 帧子集都救不回来的态):
# 那里选帧的贡献按构造恒为零,任何提升都只能来自策略适配,不与 selector 混淆。
#
# **两臂共用本脚本,只差注入配置**(单变量原则):
#   ungated_kv  末 8 层 k/v_proj,rank 8  —— 与 HGKV 同层位同秩、无门控
#   full_lora   全层 q/k/v/o_proj,rank 8 —— 层位与容量的上限对照
# 评测(eval_policy_gold_arms.py)必须并排 {B=0, recent-2} 两种 prompt:
# 只有"recent-2 上的提升显著大于 B=0 上的",才是"历史变得可读",
# 否则只是普通任务 SFT 的收益。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_target_text(rec: dict) -> str:
    """gold 助手回复:target_text(思考)+ 规范 tool_call 块。"""
    call = {"name": rec["target_tool_call"].get("name", "computer_use"),
            "arguments": rec["target_tool_call"].get("arguments", {})}
    thought = str(rec.get("target_text") or "").strip()
    block = "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"
    return (thought + "\n" + block) if thought else block


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True,
                   help="labels_all.jsonl,用于取 hopeless 分层")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--arm", choices=["ungated_kv", "full_lora"], required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--limit-states", type=int, default=0, help="冒烟用")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260809)
    args = p.parse_args()

    import sys
    import torch

    from causalcache.agentnet_desktop_official import (
        build_desktop_official_messages,
        official_step_forms,
    )
    from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "scripts"))
    from train_success_sft_lora import inject_lora, lora_state_dict

    # ---- hopeless 分层 + 划分(与 selector 线同一套 seed 洗牌协议)----
    hopeless: list[str] = []
    for f in args.labels:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "oracle_correct" in d and not d["b0_correct"] and not d["oracle_correct"]:
                hopeless.append(d["dp_id"])
    hopeless = sorted(set(hopeless))
    random.Random(args.seed).shuffle(hopeless)
    nh = int(len(hopeless) * args.holdout_frac)
    hold, train_ids = set(hopeless[:nh]), set(hopeless[nh:])
    print(json.dumps({"hopeless": len(hopeless), "train": len(train_ids),
                      "holdout": len(hold)}, ensure_ascii=False), flush=True)

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=128)
    model, device, proc = runtime.model, runtime.device, runtime.processor

    if args.arm == "ungated_kv":
        wrapped = inject_lora(model, rank=args.rank, alpha=args.alpha,
                              target_modules=("k_proj", "v_proj"),
                              torch=torch, last_layer_count=8)
    else:
        wrapped = inject_lora(model, rank=args.rank, alpha=args.alpha,
                              target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                              torch=torch, last_layer_count=None)
    params = [q for w in wrapped.values() for q in (w.lora_a, w.lora_b)]
    for q in params:
        q.requires_grad_(True)
    n_par = sum(q.numel() for q in params)
    print(json.dumps({"arm": args.arm, "lora_modules": len(wrapped),
                      "lora_params": n_par}, ensure_ascii=False), flush=True)
    opt = torch.optim.AdamW(params, lr=args.learning_rate)
    opt.zero_grad(set_to_none=True)
    tok = proc.tokenizer

    def encode_example(rec: dict):
        """recent-2 prompt + gold 目标;损失只打在目标段。"""
        s = int(rec["step"])
        images = rec["image_relpaths"]
        if len(images) != s:
            raise ValueError("image count")
        screen = tuple(int(x) for x in rec["screen_size"])
        steps = [official_step_forms(h, screen_size=screen) for h in rec["history"]]
        cands = [j for j in range(1, s - 1) if steps[j].full_response]
        if not cands:
            raise ValueError("no candidates")
        root = args.image_root
        recent = cands[-2:]
        needed = [root / images[j] for j in recent] + [root / images[s - 1]]
        if not all(x.exists() for x in needed):
            raise ValueError("missing images")
        msgs = build_desktop_official_messages(
            goal=rec["instruction"], steps=steps, shown_events=list(recent),
            event_images={j: str(root / images[j]) for j in recent},
            current_image=str(root / images[s - 1]))
        enc = proc.apply_chat_template(
            msgs, tools=[_TOOL_SPEC], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        tgt_ids = tok(build_target_text(rec), add_special_tokens=False,
                      return_tensors="pt")["input_ids"]
        close = torch.tensor([[runtime.generation_tokens.tool_call_close_token_id]])
        tgt = torch.cat([tgt_ids, close], dim=1)
        input_ids = torch.cat([enc["input_ids"], tgt], dim=1)
        labels = torch.cat([torch.full_like(enc["input_ids"], -100), tgt], dim=1)
        out = {"input_ids": input_ids,
               "attention_mask": torch.ones_like(input_ids),
               "labels": labels}
        # 多模态张量原样带上;mm_token_type_ids 给目标段补 0(纯文本)
        for k, v in enc.items():
            if k in ("input_ids", "attention_mask"):
                continue
            if k == "mm_token_type_ids":
                out[k] = torch.cat([v, torch.zeros_like(tgt)], dim=1)
            else:
                out[k] = v
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in out.items()}

    args.output_root.mkdir(parents=True, exist_ok=True)
    stats = {"seen": 0, "steps": 0, "loss": 0.0, "skipped": {}}
    order = [i for i in train_ids]
    order.sort()
    random.Random(args.seed + 1).shuffle(order)
    if args.limit_states:
        order = order[: args.limit_states]
    rows = {}
    for line in args.manifest.open(encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            if d["dp_id"] in train_ids:
                rows[d["dp_id"]] = d

    for ep in range(args.epochs):
        for dp in order:
            rec = rows.get(dp)
            if rec is None:
                continue
            try:
                ex = encode_example(rec)
                out = model(**ex)
                loss = out.loss
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
                k = type(exc).__name__ if not str(exc) else str(exc)[:40]
                stats["skipped"][k] = stats["skipped"].get(k, 0) + 1
                opt.zero_grad(set_to_none=True)
                continue
        torch.save({"arm": args.arm, "rank": args.rank, "alpha": args.alpha,
                    "state": lora_state_dict(wrapped)},
                   args.output_root / f"adapter_ep{ep}.pt")
        print(json.dumps({"epoch_end": ep, "seen": stats["seen"],
                          "mean_loss": round(stats["loss"] / max(stats["seen"], 1), 4),
                          "skipped": stats["skipped"]}, ensure_ascii=False), flush=True)
    torch.save({"arm": args.arm, "rank": args.rank, "alpha": args.alpha,
                "state": lora_state_dict(wrapped)},
               args.output_root / "adapter.pt")
    print(json.dumps({"final": True, "arm": args.arm, "seen": stats["seen"],
                      "opt_steps": stats["steps"],
                      "skipped": stats["skipped"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
