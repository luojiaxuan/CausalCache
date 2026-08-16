#!/usr/bin/env python3
"""诊断:executor 重算 log π 与采样值的差,是**数值噪声**还是**结构性错配**?

# note (luojiaxuan): joint trainer 的启动闸门报 max|Δ|=0.366 / mean|Δ|=0.029
# (整条动作序列求和,序列长 ~46 token)。两种病因的处方完全相反,必须分开:
#
#   (A) **数值噪声**:采样走的是带 KV cache 的增量解码,重算走整序列单次前向,
#       bf16 下两条 kernel 路径本就有 ~1e-3/token 的差,46 个 token 累加到
#       0.03-0.4 完全正常。此时逐 token 差应当**又小又随机**(正负各半、
#       与位置无关),且**重算两次完全相同**(确定性)。
#   (B) **结构性错配**:prompt 少一段、掩码不同、错位一格、tokenizer 口径不同。
#       此时逐 token 差会**集中在少数位置**且量级 ~1 nat 以上,或整体系统性
#       偏向一侧。
#
# 判据(两者相差三个数量级,不是拍脑袋的阈值):
#   per-token mean|Δ| ~ 1e-3  → 数值;~1e0 → 结构。
#
# **只读诊断,不产出任何训练信号**(路线 §9)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", type=Path, required=True)
    ap.add_argument("--executor-adapter", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--snapshot-manifest", type=Path, required=True)
    ap.add_argument("--n-steps", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for cand in (Path(__file__).resolve().parents[1], Path("/data/agentic/code")):
        if (cand / "causalcache_agentic").exists():
            sys.path.insert(0, str(cand))
            break
    for cand in (Path(__file__).resolve().parents[3] / "code" / "scripts",
                 Path("/data/osworld/CausalCache/code/scripts")):
        if cand.exists():
            sys.path.insert(0, str(cand))
            break

    from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime
    from causalcache_agentic.policy_io import HistoryFrameBank, PolicyInputBuilder
    from train_success_sft_lora import inject_lora, load_lora_state_dict

    recs = []
    for p in args.rollouts:
        for line in p.open(encoding="utf-8"):
            if line.strip():
                recs.append(json.loads(line))

    dev = torch.device(args.device)
    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir, expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device, effective_visual_tokens_per_image=2560,
        max_new_tokens=128)
    model, proc = runtime.model, runtime.processor
    gt = runtime.generation_tokens
    bundle = torch.load(args.executor_adapter, map_location="cpu",
                        weights_only=False)
    wrapped = inject_lora(
        model, rank=int(bundle["rank"]), alpha=int(bundle["alpha"]),
        target_modules=tuple(str(bundle.get(
            "target_modules", "q_proj,k_proj,v_proj,o_proj")).split(",")),
        torch=torch,
        last_layer_count=(int(bundle["last_layers"])
                          if bundle.get("last_layers") else None))
    load_lora_state_dict(wrapped, bundle["state"])
    model.eval()
    builder = PolicyInputBuilder(budget=args.budget)
    suppress = list(gt.standard_eos_token_ids)

    def per_token(step: dict, rec: dict):
        toks = step.get("action_tokens")
        if not toks:
            return None
        bank = HistoryFrameBank()
        for s in rec["steps"]:
            if int(s["step"]) <= int(step["step"]):
                bank.add(int(s["step"]), s["screenshot"])
        hist = [dict(s["action"]) for s in rec["steps"]
                if int(s["step"]) < int(step["step"])]
        msgs = builder.build(rec.get("instruction", ""), hist,
                             step["chosen_subset"], bank, int(step["step"]))
        enc = proc.apply_chat_template(
            msgs, tools=[_TOOL_SPEC], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(dev)
        tgt = torch.tensor([toks], device=dev)
        input_ids = torch.cat([enc["input_ids"], tgt], dim=1)
        kw = {k: v for k, v in enc.items()
              if k not in ("input_ids", "attention_mask")}
        if "mm_token_type_ids" in kw:
            kw["mm_token_type_ids"] = torch.cat(
                [kw["mm_token_type_ids"], torch.zeros_like(tgt)], dim=1)
        out = model(input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    logits_to_keep=len(toks) + 1, **kw)
        logits = out.logits[0, :-1].float() / max(args.temperature, 1e-6)
        if suppress:
            logits[:, suppress] = float("-inf")
        lp = torch.log_softmax(logits, dim=-1)
        return lp.gather(-1, tgt[0].unsqueeze(-1)).squeeze(-1), int(
            enc["input_ids"].shape[1])

    rows = []
    with torch.no_grad():
        for rec in recs:
            for st in rec["steps"]:
                if st.get("policy_logprob") is None or len(rows) >= args.n_steps:
                    continue
                got = per_token(st, rec)
                if got is None:
                    continue
                lp1, n_prompt = got
                lp2, _ = per_token(st, rec)      # 重算第二次:确定性检查
                total = float(lp1.sum())
                rec_lp = float(st["policy_logprob"])
                d = total - rec_lp
                n_tok = int(lp1.numel())
                rows.append({
                    "n_tok": n_tok, "n_prompt": n_prompt,
                    "recorded": round(rec_lp, 5), "recomputed": round(total, 5),
                    "delta": round(d, 5),
                    "delta_per_token": round(d / max(n_tok, 1), 6),
                    "rerun_max_abs": round(
                        float((lp1 - lp2).abs().max()), 8),
                    "worst_token_abs": None,
                })
            if len(rows) >= args.n_steps:
                break

    n = len(rows)
    if not n:
        raise SystemExit("没有可诊断的步(缺 policy_logprob)")
    tot_tok = sum(r["n_tok"] for r in rows)
    sum_abs = sum(abs(r["delta"]) for r in rows)
    per_tok = sum_abs / max(tot_tok, 1)
    signed = sum(r["delta"] for r in rows) / n
    det = max(r["rerun_max_abs"] for r in rows)
    verdict = ("数值噪声(bf16 解码 vs 单次前向)" if per_tok < 0.02
               else "结构性错配 —— 必须修 prompt/掩码口径,不能放宽阈值")
    summary = {
        "n_steps": n, "total_tokens": tot_tok,
        "mean_abs_delta_per_step": round(sum_abs / n, 5),
        "mean_abs_delta_PER_TOKEN": round(per_tok, 6),
        "mean_signed_delta_per_step": round(signed, 5),
        "recompute_determinism_max_abs": round(det, 8),
        "verdict": verdict,
    }
    print(json.dumps({"rows": rows, "summary": summary},
                     ensure_ascii=False, indent=1))
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"rows": rows, "summary": summary}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
