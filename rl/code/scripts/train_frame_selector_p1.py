#!/usr/bin/env python3
"""P1:把 GUI-Owl-8B 自己 LoRA 微调成选帧器(Pareto 阶梯第一级:效果优先)。

# note (luojiaxuan): 依据(台账 §0.10 后记 + fa13219 侦察):
#   * 探针信号上界 UB-b1 全量 +9.7pp(MultiApp +12.3)—— 信号大且真实;
#   * 零参数 disagree 规则只吃到 +1.5pp —— 二值"不一致"分不清纠错与噪声;
#   * 2.6/2.7 的失败根因是输入里没有策略行为,不是容量。
# P1 把行为信号全部放进输入:每个候选帧 = 缩略图 + 探针动作(B=1 前向的
# 输出)+ 与草稿是否一致;外加全清当前屏 + 草稿动作(B=0 前向输出)。
# 监督 = 从该态的正确帧对里每 epoch 随机采一个,目标文本 "SELECT: i,j",
# 损失只打目标段(与 #26 同款掩码)。
# 总体口径与 2.6/2.7 完全一致:winnable 且有 token 缓存的 1263 态,
# 同一 seed 洗牌 + fold 切片 —— 部署评测(eval_frame_selector_p1.py)可比。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from pathlib import Path


def disagree(a: dict | None, b: dict | None, tol: float = 25.0) -> bool:
    if not a or not b:
        return a != b
    if a.get("action") != b.get("action"):
        return True
    ca, cb = a.get("coordinate"), b.get("coordinate")
    if ca and cb:
        return math.dist(ca, cb) > tol
    return any(a.get(k) != b.get(k) for k in ("text", "keys", "key"))


def load_winnable(labels: list[Path], token_dir: Path):
    """与 2.6/2.7 完全同一总体:winnable(有正有负帧对)且有 token 缓存。"""
    lab: dict[str, tuple[list, list]] = {}
    for f in labels:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            subs = [(tuple(a["s"]), bool(a["c"])) for a in d.get("all", [])
                    if len(a["s"]) == 2]
            pos = [s for s, c in subs if c]
            neg = [s for s, c in subs if not c]
            if not (pos and neg):
                continue
            if not (token_dir / f"{d['dp_id']}.pt").exists():
                continue
            lab[d["dp_id"]] = (pos, neg)
    return lab


def load_b1(b1_dir: Path, want: set[str]):
    """b1 探针行(probe1 新枚举优先,bcurve 兜底)→ {dp: {b0_pred, preds{j:p}}}。"""
    out: dict[str, dict] = {}
    for pat in ("bcurve_b1_sh*.jsonl", "probe1_sh*.jsonl"):
        for f in sorted(b1_dir.glob(pat)):
            for line in f.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("all") and r["dp_id"] in want:
                    out[r["dp_id"]] = {
                        "b0_pred": r.get("b0_pred"),
                        "preds": {a["s"][0]: a.get("p") for a in r["all"]}}
    return out


def build_selector_messages(*, goal: str, cands: list[int], thumbs: dict,
                            probe: dict, current_image: str) -> list[dict]:
    """选帧 prompt:候选帧(缩略图+探针动作+一致性)→ 当前屏(全清)→ 草稿。"""
    draft = probe["b0_pred"]
    lines = [("You are selecting memory frames for a GUI agent. Task goal: "
              f"{goal}\nBelow are candidate history frames (older to newer). "
              "For each: its thumbnail, the probe action (what the agent would "
              "do next if shown ONLY that frame), and whether the probe "
              "disagrees with the draft action.")]
    content: list[dict] = [{"type": "text", "text": lines[0]}]
    for j in cands:
        p = probe["preds"].get(j)
        flag = "DISAGREES with draft" if disagree(p, draft) else "agrees with draft"
        content.append({"type": "text",
                        "text": f"\n[Frame {j}] probe action: "
                                f"{json.dumps(p, ensure_ascii=False)} ({flag})"})
        content.append({"type": "image", "image": thumbs[j]})
    content.append({"type": "text", "text": "\nCurrent screen:"})
    content.append({"type": "image", "image": current_image})
    content.append({"type": "text",
                    "text": ("\nDraft action (the agent's output with no history "
                             f"frames): {json.dumps(draft, ensure_ascii=False)}\n"
                             "By default the agent is shown the two most recent "
                             f"frames [{cands[-2]}, {cands[-1]}]. If that default "
                             "is already the best choice, answer exactly: KEEP\n"
                             "Otherwise select the two frames that would most "
                             "help the agent act correctly, answering exactly: "
                             "SELECT: i,j")})
    return [{"role": "user", "content": content}]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True)
    p.add_argument("--token-dir", type=Path, required=True)
    p.add_argument("--b1-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--thumb-pixels", type=int, default=112896,
                   help="候选帧缩略图像素预算(≈144 视觉 token);P2 消融旋钮")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--limit-states", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--torch-seed", type=int, default=0)
    # note (luojiaxuan): v1 的 uniform 目标是失败根因 —— winnable 里 recent-2
    # 本就正确的态占 ~50%,均匀采正对等于教模型"对的也要挪走",学出 75%
    # 移动率、每动错赢 2:1(台账 §0.11 P1 终判)。minimal = 最小干预目标:
    # recent-2 对 → KEEP;错 → 换到最靠 recency 的正确对。
    p.add_argument("--target", choices=["minimal", "uniform"], default="minimal")
    args = p.parse_args()

    import sys
    import torch
    from PIL import Image

    from causalcache.agentnet_desktop_official import official_step_forms
    from causalcache.osworld_gui_owl import GUIOwlOSWorldRuntime
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "scripts"))
    except IndexError:
        pass  # 散件部署时靠 PYTHONPATH 提供 code/scripts
    from train_success_sft_lora import inject_lora, lora_state_dict

    lab = load_winnable(args.labels, args.token_dir)
    ids = sorted(lab)
    random.Random(args.seed).shuffle(ids)
    nh = int(len(ids) * args.holdout_frac)
    lo, hi = args.fold * nh, (args.fold + 1) * nh
    hold = set(ids[lo:hi])
    train_ids = [d for d in ids if d not in hold]
    b1 = load_b1(args.b1_dir, set(ids))
    print(json.dumps({"winnable": len(ids), "train": len(train_ids),
                      "holdout": len(hold), "fold": args.fold,
                      "with_b1": sum(1 for d in ids if d in b1)},
                     ensure_ascii=False), flush=True)

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=16)
    model, device = runtime.model, runtime.device
    # note (luojiaxuan): runtime 的 processor 把 min=max 像素钉死(策略动作遍
    # 需要),会把缩略图强行放大回 2560 token —— 首启三种子因此每样本
    # ~3.6 万视觉 token,2017/2022 全数 OOM。选帧 prompt 自建 processor:
    # min 放开让缩略图保持小,max 封顶让当前屏落在 --visual-tokens。
    from causalcache.osworld_gui_owl import (
        VISION_PATCH_SIZE,
        VISION_SPATIAL_MERGE_SIZE,
    )
    from transformers import AutoProcessor
    unit = (VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE) ** 2
    proc = AutoProcessor.from_pretrained(
        str(args.model_dir), min_pixels=4 * unit,
        max_pixels=args.visual_tokens * unit, local_files_only=True)
    torch.manual_seed(args.torch_seed)
    wrapped = inject_lora(model, rank=args.rank, alpha=args.alpha,
                          target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                          torch=torch, last_layer_count=None)
    params = [q for w in wrapped.values() for q in (w.lora_a, w.lora_b)]
    for q in params:
        q.requires_grad_(True)
    print(json.dumps({"lora_modules": len(wrapped),
                      "lora_params": sum(q.numel() for q in params)}), flush=True)
    opt = torch.optim.AdamW(params, lr=args.learning_rate)
    opt.zero_grad(set_to_none=True)
    tok = proc.tokenizer

    rows = {}
    for line in args.manifest.open(encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            if d["dp_id"] in lab:
                rows[d["dp_id"]] = d

    def make_thumb(path: Path, budget: int) -> "Image.Image":
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(1.0, (budget / (w * h)) ** 0.5)
        if scale < 1.0:
            img = img.resize((max(28, int(w * scale)), max(28, int(h * scale))))
        return img

    def encode_example(rec: dict, target_text: str):
        s = int(rec["step"])
        images = rec["image_relpaths"]
        if len(images) != s:
            raise ValueError("image count")
        screen = tuple(int(x) for x in rec["screen_size"])
        steps = [official_step_forms(h, screen_size=screen) for h in rec["history"]]
        cands = [j for j in range(1, s - 1) if steps[j].full_response]
        if len(cands) < 2:
            raise ValueError("too few candidates")
        root = args.image_root
        needed = [root / images[j] for j in cands] + [root / images[s - 1]]
        if not all(x.exists() for x in needed):
            raise ValueError("missing images")
        probe = b1.get(rec["dp_id"])
        if probe is None:
            raise ValueError("no b1 probe")
        thumbs = {j: make_thumb(root / images[j], args.thumb_pixels)
                  for j in cands}
        msgs = build_selector_messages(
            goal=rec["instruction"], cands=cands, thumbs=thumbs,
            probe=probe, current_image=str(root / images[s - 1]))
        enc = proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        tgt_ids = tok(target_text, add_special_tokens=False,
                      return_tensors="pt")["input_ids"]
        eos = torch.tensor([[tok.eos_token_id]])
        tgt = torch.cat([tgt_ids, eos], dim=1)
        input_ids = torch.cat([enc["input_ids"], tgt], dim=1)
        labels = torch.cat([torch.full_like(enc["input_ids"], -100), tgt], dim=1)
        out = {"input_ids": input_ids,
               "attention_mask": torch.ones_like(input_ids),
               "labels": labels}
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
    order = list(train_ids)
    if args.limit_states:
        order = order[: args.limit_states]
    rng = random.Random(args.seed + 7 + args.torch_seed)
    for ep in range(args.epochs):
        random.Random(args.seed + 1 + ep).shuffle(order)
        for dp in order:
            rec = rows.get(dp)
            if rec is None:
                continue
            try:
                pos = [tuple(sorted(t)) for t in lab[dp][0]]
                if args.target == "uniform":
                    a, b = rng.choice(pos)
                    tgt_text = f"SELECT: {a},{b}"
                else:
                    cand_set = sorted({j for t in lab[dp][0] + lab[dp][1]
                                       for j in t})
                    recent2 = tuple(sorted(cand_set[-2:]))
                    if recent2 in set(pos):
                        tgt_text = "KEEP"
                    else:
                        a, b = max(pos, key=lambda t: sum(t))
                        tgt_text = f"SELECT: {a},{b}"
                ex = encode_example(rec, tgt_text)
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
                k = type(exc).__name__ if not str(exc) else str(exc)[:40]
                stats["skipped"][k] = stats["skipped"].get(k, 0) + 1
                opt.zero_grad(set_to_none=True)
                continue
        torch.save({"task": "frame_selector_p1", "rank": args.rank,
                    "alpha": args.alpha, "torch_seed": args.torch_seed,
                    "fold": args.fold, "thumb_pixels": args.thumb_pixels,
                    "target": args.target,
                    "state": lora_state_dict(wrapped)},
                   args.output_root / f"adapter_ep{ep}.pt")
        print(json.dumps({"epoch_end": ep, "seen": stats["seen"],
                          "mean_loss": round(stats["loss"] / max(stats["seen"], 1), 4),
                          "skipped": stats["skipped"]}, ensure_ascii=False), flush=True)
    torch.save({"task": "frame_selector_p1", "rank": args.rank,
                "alpha": args.alpha, "torch_seed": args.torch_seed,
                "fold": args.fold, "thumb_pixels": args.thumb_pixels,
                    "target": args.target,
                "state": lora_state_dict(wrapped)},
               args.output_root / "adapter.pt")
    n_skip = sum(stats["skipped"].values())
    print(json.dumps({"final": True, "seen": stats["seen"],
                      "opt_steps": stats["steps"],
                      "skipped": stats["skipped"]}, ensure_ascii=False))
    # note (luojiaxuan): 与 #26 同款守卫 —— 跳过率超阈必须非零退出,
    # 防止"没训过的 adapter"被当正品评(§6 错误表第 18 条前科)。
    if n_skip > 0.1 * max(stats["seen"] + n_skip, 1):
        raise SystemExit(f"FAILED: 跳过率 {n_skip}/{stats['seen']+n_skip} 超过 10%")


if __name__ == "__main__":
    main()
