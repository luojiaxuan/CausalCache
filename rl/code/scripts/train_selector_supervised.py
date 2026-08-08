#!/usr/bin/env python3
"""v4:用枚举标签监督训练 selector —— 目标与评测指标同源。

# note (luojiaxuan): v3(步级 GRPO,奖励 = log p(gold action))的判据是
# **训动了但没学到**:750 优化步、位移 1.557、与初始头选帧仅 13% 重合,
# 步级正确率却与初始头完全打平(25.0% vs 25.0%,p=1.000)。
# 归因:**训的是连续的 log p,评的是离散的 argmax 正确性,两者脱钩** ——
# log p 可以大涨而 argmax 一次不翻。
#
# 本脚本换掉目标:直接学"把能做对的子集排到前面"。
#   * 标签来自枚举(rl_oracle_enumerate.py 的 `all` 字段:每个子集 correct);
#   * 每个 state 取正集合 P(correct)与负集合 N(incorrect),损失是
#     **子集级 Plackett-Luce 打分下的成对排序**:
#         L = −log σ( s(p) − s(n) ),  s(子集) = Σ_{帧∈子集} head(feat_帧)
#     子集分 = 成员帧分之和,与部署时 top-B 的 argmax 口径一致。
#   * 只用 winnable 态(既有正又有负);easy/hopeless 无梯度信息。
#
# **不需要跑策略**:标签已在枚举里算好,这里只做 hidden 特征前向 + 线性头,
# 因此比 v3 快一个数量级(v3 每 state 要 K 次带图前向)。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True,
                   help="枚举产出的 jsonl(可多个分片)")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--init-head", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--pairs-per-state", type=int, default=8,
                   help="每个 state 采样几组(正,负)对")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--holdout-frac", type=float, default=0.2,
                   help="留出比例:训练完在**未训过的 state** 上报排序准确率")
    p.add_argument("--index-visual-tokens", type=int, default=144)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--report-every", type=int, default=50)
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
    from causalcache_rl.index_features import index_features

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    # ---- 读标签:只留 winnable(既有正子集又有负子集)----
    lab: dict[str, dict[str, Any]] = {}
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
            if pos and neg:
                lab[d["dp_id"]] = {"pos": pos, "neg": neg}
    ids = sorted(lab)
    rng.shuffle(ids)
    n_hold = int(len(ids) * args.holdout_frac)
    hold, train_ids = set(ids[:n_hold]), set(ids[n_hold:])
    print(json.dumps({"winnable_states": len(ids), "train": len(train_ids),
                      "holdout": len(hold)}, ensure_ascii=False), flush=True)
    if not train_ids:
        raise SystemExit("没有 winnable 标签 —— 先把枚举跑出来")

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

    bundle = torch.load(args.init_head, map_location="cpu", weights_only=False)
    head = torch.nn.Linear(int(bundle["hidden_size"]), 1)
    head.load_state_dict(bundle["model_state"])
    head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    opt.zero_grad(set_to_none=True)

    def state_feats(rec):
        s = int(rec["step"])
        images = rec["image_relpaths"]
        if len(images) != s:
            raise ValueError("image count")
        screen = tuple(int(x) for x in rec["screen_size"])
        steps = [official_step_forms(h, screen_size=screen) for h in rec["history"]]
        cands = [j for j in range(1, s - 1) if steps[j].full_response]
        ev = {j: str(args.image_root / images[j]) for j in cands}
        cur = args.image_root / images[s - 1]
        if not all((args.image_root / images[j]).exists() for j in cands) or not cur.exists():
            raise ValueError("missing images")
        feats = index_features(rec, steps, cands, ev, str(cur),
                               model=model, processor=index_processor,
                               device=device, torch=torch,
                               build=build_desktop_official_messages,
                               tool_spec=_TOOL_SPEC)
        return cands, feats

    def subset_score(sc, cands, subset):
        """子集分 = 成员帧分之和 —— 与部署 top-B argmax 口径一致。"""
        return sum(sc[cands.index(j)] for j in subset)

    stats = {"pairs": 0, "steps": 0, "loss": 0.0, "skipped": 0}
    seen = 0
    cache: dict[str, tuple] = {}
    for _epoch in range(args.epochs):
        for lineno, line in enumerate(args.manifest.open(encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            dp = rec["dp_id"]
            if dp not in train_ids:
                continue
            try:
                cands, feats = cache.get(dp) or state_feats(rec)
                cache[dp] = (cands, feats)
                sc = head(feats).squeeze(-1)
                loss = torch.zeros((), device=device)
                k = 0
                for _ in range(args.pairs_per_state):
                    pos = rng.choice(lab[dp]["pos"])
                    neg = rng.choice(lab[dp]["neg"])
                    if not (set(pos) <= set(cands) and set(neg) <= set(cands)):
                        continue
                    margin = subset_score(sc, cands, pos) - subset_score(sc, cands, neg)
                    loss = loss - torch.nn.functional.logsigmoid(margin)
                    k += 1
                if k == 0:
                    raise ValueError("no usable pair")
                (loss / k / args.grad_accum).backward()
                stats["loss"] += float(loss.detach()) / k
                stats["pairs"] += k
                seen += 1
                if seen % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    opt.step(); opt.zero_grad(set_to_none=True)
                    stats["steps"] += 1
                if seen % args.report_every == 0:
                    print(json.dumps({
                        "states": seen, "opt_steps": stats["steps"],
                        "mean_pair_loss": round(stats["loss"] / seen, 4),
                        "skipped": stats["skipped"]}, ensure_ascii=False), flush=True)
            except (ValueError, KeyError, OSError, TypeError, RuntimeError):
                stats["skipped"] += 1
                opt.zero_grad(set_to_none=True)
                continue

    # ---- 留出集:排序准确率(正子集分 > 负子集分 的比例)----
    hits = tot = 0
    with torch.no_grad():
        for line in args.manifest.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["dp_id"] not in hold:
                continue
            try:
                cands, feats = state_feats(rec)
                sc = head(feats).squeeze(-1)
                for _ in range(args.pairs_per_state):
                    pos = rng.choice(lab[rec["dp_id"]]["pos"])
                    neg = rng.choice(lab[rec["dp_id"]]["neg"])
                    if not (set(pos) <= set(cands) and set(neg) <= set(cands)):
                        continue
                    tot += 1
                    hits += int(float(subset_score(sc, cands, pos))
                                > float(subset_score(sc, cands, neg)))
            except (ValueError, KeyError, OSError, TypeError, RuntimeError):
                continue

    out = dict(bundle)
    out["model_state"] = head.state_dict()
    torch.save(out, args.output_root / "selector_bundle.pt")
    report = {"train_states": seen, "opt_steps": stats["steps"],
              "pairs": stats["pairs"], "skipped": stats["skipped"],
              "holdout_pairs": tot,
              "holdout_rank_acc": round(hits / max(tot, 1), 4),
              "note": "排序准确率 0.5 = 与随机无异;>0.5 才说明学到了可迁移的排序"}
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
