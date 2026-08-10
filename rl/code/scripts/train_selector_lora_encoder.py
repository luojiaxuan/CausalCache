#!/usr/bin/env python3
"""selector v2 阶梯 3:可训练索引编码器(LoRA)+ token 级交叉注意力读出。

# note (luojiaxuan): 阶梯逻辑 —— 与阶梯 2 **只差一个变量**:
#   阶梯 2 = 冻结编码器 + 交叉注意力读出 → 失败(0.561 < recency 0.613);
#   阶梯 3 = **LoRA 编码器** + 同款读出。
# 若 3 成功而 2 失败,归因是"编码器需要学着为选帧任务写表示";
# 若 3 也失败,'该信号在冻结策略的任何可及表示下都不可学'的结论成立,
# 回到改策略(HGKV)那条线。
#
# 为什么读出必须是交叉注意力而不是池化头:索引 prompt 里候选帧排在当前屏
# **前面**,因果注意力下帧 token 永远看不到当前屏 —— 无论编码器怎么训,
# 帧的池化向量都写不进"与当前这一步的关系";能搭这座桥的只有读出端
# (当前屏 token 作 query 去查帧 token)。
#
# **LoRA 只进 selector 自己的编码副本;部署的动作 policy 仍然冻结**
# (用户约束)。复用 train_success_sft_lora 的 inject_lora —— 不另写一份。
#
# 显存账:索引 prompt 约 11×144 视觉 + 文本 ≈ 2.5K token,8B bf16 权重 16G
# + 带梯度激活约 30-40G,H200(141G)单卡放得下,不开梯度检查点。
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path


def stable_rng(dp: str) -> random.Random:
    # note (luojiaxuan): 此前评测用 hash(dp) 采样配对 —— str 的 hash 每个进程
    # 加盐,导致跨运行的"同一参照"数字漂移(recency 参照 0.595 vs 0.613,
    # 同一规则、不同配对样本)。运行内比较不受影响,但跨运行必须可复现,
    # 改用 crc32。**本脚本内的所有臂共用同一套配对,比较仍然配对干净。**
    return random.Random(zlib.crc32(dp.encode()) & 0xFFFFFFFF)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-layers", type=int, default=8)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--lora-lr", type=float, default=1e-5,
                   help="比 head 低一个量级:8B 编码器上的 LoRA 一旦跑飞,"
                        "整个索引遍的表示都会漂,不像小头炸了无所谓")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--limit-states", type=int, default=0,
                   help=">0 时只用前 N 个训练态(冒烟用)")
    p.add_argument("--eval-every", type=int, default=500,
                   help="每 N 个训练态评一次留出集(顺带存 checkpoint)")
    p.add_argument("--index-visual-tokens", type=int, default=144)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260809)
    args = p.parse_args()

    import torch
    from transformers import AutoProcessor

    from causalcache.agentnet_desktop_official import (
        build_desktop_official_messages,
        official_step_forms,
    )
    from causalcache.osworld_gui_owl import (
        _TOOL_SPEC,
        VISION_PATCH_SIZE,
        VISION_SPATIAL_MERGE_SIZE,
        GUIOwlOSWorldRuntime,
    )
    from causalcache_rl.token_cross_scorer import TokenCrossScorer
    from scripts.train_success_sft_lora import inject_lora, lora_state_dict

    torch.manual_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    # ---- 标签与划分(与池化线/阶梯 2 同一划分)----
    lab: dict[str, tuple] = {}
    carrier: dict[str, tuple] = {}
    for f in args.labels:
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            subs = [(tuple(a["s"]), bool(a["c"])) for a in d.get("all", [])
                    if len(a["s"]) == args.budget]
            pos = [s for s, c in subs if c]
            neg = [s for s, c in subs if not c]
            if not (pos and neg):
                continue
            lab[d["dp_id"]] = (pos, neg)
            cr: dict[int, tuple[int, int]] = {}
            for sset, c in subs:
                for j in sset:
                    h, t = cr.get(j, (0, 0))
                    cr[j] = (h + int(c), t + 1)
            rate = {j: h / t for j, (h, t) in cr.items() if t >= 2}
            if len(rate) >= 2:
                hi = max(rate, key=rate.get)
                lo = min(rate, key=rate.get)
                if rate[hi] - rate[lo] >= 0.2:
                    carrier[d["dp_id"]] = (hi, lo)
    ids = sorted(lab)
    random.Random(args.seed).shuffle(ids)
    nh = int(len(ids) * 0.2)
    hold, train = ids[:nh], ids[nh:]
    if args.limit_states > 0:
        train = train[: args.limit_states]
        hold = hold[: max(args.limit_states // 3, 20)]
    print(json.dumps({"winnable": len(ids), "train": len(train),
                      "holdout": len(hold)}, ensure_ascii=False), flush=True)

    # ---- manifest 行缓存(训练要反复重建 prompt)----
    recs: dict[str, dict] = {}
    for line in args.manifest.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["dp_id"] in lab:
            recs[r["dp_id"]] = r

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=16)
    model, device = runtime.model, runtime.device
    _px = args.index_visual_tokens * (VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE) ** 2
    index_processor = AutoProcessor.from_pretrained(
        args.model_dir, min_pixels=_px, max_pixels=_px, local_files_only=True)

    wrapped = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                          target_modules=("q_proj", "k_proj", "v_proj"),
                          torch=torch, last_layer_count=args.lora_layers)
    lora_params = [t for w in wrapped.values() for t in (w.lora_a, w.lora_b)]
    for t in lora_params:
        t.requires_grad_(True)
    head = TokenCrossScorer().to(device)
    n_lora = sum(t.numel() for t in lora_params)
    print(json.dumps({"lora_modules": len(wrapped), "lora_params": n_lora,
                      "head_params": sum(x.numel() for x in head.parameters())},
                     ensure_ascii=False), flush=True)
    opt = torch.optim.AdamW([
        {"params": head.parameters(), "lr": args.head_lr},
        {"params": lora_params, "lr": args.lora_lr}])
    opt.zero_grad(set_to_none=True)

    def encode(dp: str):
        rec = recs[dp]
        s = int(rec["step"])
        images = rec["image_relpaths"]
        screen = tuple(int(x) for x in rec["screen_size"])
        steps = [official_step_forms(h, screen_size=screen) for h in rec["history"]]
        cands = [j for j in range(1, s - 1) if steps[j].full_response]
        ev = {j: str(args.image_root / images[j]) for j in cands}
        msgs = build_desktop_official_messages(
            goal=rec["instruction"], steps=steps, shown_events=list(cands),
            event_images=ev, current_image=str(args.image_root / images[s - 1]))
        enc = index_processor.apply_chat_template(
            msgs, tools=[_TOOL_SPEC], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(device)
        return cands, enc

    def frame_tokens(enc, n_frames: int, *, grad: bool):
        ctxmgr = torch.enable_grad() if grad else torch.no_grad()
        with ctxmgr:
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[-1][0]
        mm = enc.get("mm_token_type_ids")
        mask = (mm[0] == 1)
        segs, st = [], None
        for i, flag in enumerate(mask.tolist()):
            if flag and st is None:
                st = i
            elif not flag and st is not None:
                segs.append((st, i))
                st = None
        if st is not None:
            segs.append((st, len(mask)))
        if len(segs) != n_frames + 1:
            raise ValueError(f"segs {len(segs)} != {n_frames + 1}")
        # note (luojiaxuan): 模型是 bf16、读出头是 fp32,不上转会 matmul dtype
        # 报错(冒烟 30/30 全挂在这)。cast 对梯度透明。
        toks = [hs[a:b].float() for a, b in segs[:-1]]
        ctx = hs[segs[-1][0]:segs[-1][1]].float()
        return toks, ctx

    def evaluate() -> dict:
        head.eval()
        hit = tot = fh = ft = rhit = rtot = 0
        with torch.no_grad():
            for dp in hold:
                try:
                    cands, enc = encode(dp)
                    toks, ctx = frame_tokens(enc, len(cands), grad=False)
                    u = head.frame_scores(toks, ctx)
                except (ValueError, KeyError, OSError, RuntimeError):
                    continue
                pos, neg = lab[dp]
                r = stable_rng(dp)
                for _ in range(8):
                    a, b = r.choice(pos), r.choice(neg)
                    if not (set(a) <= set(cands) and set(b) <= set(cands)):
                        continue
                    tot += 1
                    hit += int(float(head.subset_score(u, cands, a)
                                     - head.subset_score(u, cands, b)) > 0)
                    rtot += 1
                    rhit += int(sum(a) > sum(b))
                if dp in carrier:
                    hi, lo = carrier[dp]
                    if hi in cands and lo in cands:
                        ft += 1
                        fh += int(float(u[cands.index(hi)]) > float(u[cands.index(lo)]))
        head.train()
        return {"pair": round(hit / max(tot, 1), 4),
                "frame_auc": round(fh / max(ft, 1), 4),
                "recency_ref_pair": round(rhit / max(rtot, 1), 4)}

    print(json.dumps({"init_eval": evaluate()}, ensure_ascii=False), flush=True)

    rng = random.Random(1)
    seen = 0
    loss_sum = 0.0
    skipped: dict[str, int] = {}
    for epoch in range(args.epochs):
        for dp in train:
            try:
                cands, enc = encode(dp)
                toks, ctx = frame_tokens(enc, len(cands), grad=True)
                u = head.frame_scores(toks, ctx)
                pos, neg = lab[dp]
                loss = torch.zeros((), device=device)
                k = 0
                for _ in range(8):
                    a, b = rng.choice(pos), rng.choice(neg)
                    if not (set(a) <= set(cands) and set(b) <= set(cands)):
                        continue
                    loss = loss - torch.nn.functional.logsigmoid(
                        head.subset_score(u, cands, a)
                        - head.subset_score(u, cands, b))
                    k += 1
                if not k:
                    raise ValueError("no usable pair")
                (loss / k / args.grad_accum).backward()
                loss_sum += float(loss.detach()) / k
                seen += 1
                if seen % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(head.parameters()) + lora_params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if seen % args.eval_every == 0:
                    ev = evaluate()
                    print(json.dumps({"epoch": epoch, "seen": seen,
                                      "mean_loss": round(loss_sum / args.eval_every, 4),
                                      **ev}, ensure_ascii=False), flush=True)
                    loss_sum = 0.0
                    torch.save({"lora": lora_state_dict(wrapped),
                                "head": head.state_dict(), "seen": seen},
                               args.output_root / "ckpt.pt")
            except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
                k2 = type(exc).__name__ if not str(exc) else str(exc)[:40]
                skipped[k2] = skipped.get(k2, 0) + 1
                opt.zero_grad(set_to_none=True)
                continue
        ev = evaluate()
        print(json.dumps({"epoch_end": epoch, "seen": seen, **ev,
                          "skipped": skipped}, ensure_ascii=False), flush=True)
        torch.save({"lora": lora_state_dict(wrapped), "head": head.state_dict(),
                    "seen": seen}, args.output_root / f"ckpt_epoch{epoch}.pt")

    report = {"final": evaluate(), "seen": seen, "skipped": skipped,
              "lora_params": n_lora,
              "refs": {"stage2_best_pair": 0.561, "pooled_best_pair": 0.593},
              "note": "判据:pair 明显超过本运行内的 recency_ref_pair 才算"
                      "编码器可训性是杠杆;仍低于 → 阶梯 4(冻结表示下不可学)"}
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
