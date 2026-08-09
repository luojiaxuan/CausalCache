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
    # ---- 容量升级(2026-08-09):v4 的 Linear(4096,1) 连训练集都拟合不上 ----
    # note (luojiaxuan): 三档是**消融梯度**,不是三选一。linear→mlp 的提升归因于
    # 非线性,mlp→mlp_pair 归因于帧间交互,开/关 --use-context 归因于"打分函数
    # 到底知不知道当前屏长什么样"。一步跳到最复杂那档就分不清是哪一项在起作用。
    p.add_argument("--head-arch", choices=["linear", "mlp", "mlp_pair"],
                   default="linear")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--pair-rank", type=int, default=64)
    p.add_argument("--use-context", action="store_true",
                   help="把当前屏 pooled 特征接进打分函数。v4 一直丢弃它 —— 而"
                        "因果注意力下候选帧看不到当前屏,等于在问'这帧有用吗'"
                        "却不告诉模型'对哪一步有用'")
    p.add_argument("--pooling", choices=["mean", "mean_max"], default="mean")
    # note (luojiaxuan): 有缓存就**完全不加载 8B 模型** —— 索引遍是 no_grad 的,
    # 输出与打分头无关,八种消融组合各过一遍前向纯属重复劳动。
    # 缓存由 cache_index_features.py 生成(mean_max,mean 取前半段切片)。
    p.add_argument("--feature-cache", type=Path, default=None,
                   help="索引遍特征缓存 .pt;给了就跳过模型加载,训练降到分钟级")
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
    from causalcache_rl.subset_scorer import SubsetScorer, load_v4_linear

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

    fcache: dict[str, Any] | None = None
    if args.feature_cache is not None:
        fcache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
        device = torch.device(args.device)
        model = index_processor = None
        print(json.dumps({"feature_cache": len(fcache), "model_loaded": False},
                         ensure_ascii=False), flush=True)
    else:
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
    # note (luojiaxuan): mean_max 池化把特征维翻倍,打分头必须同步,否则形状不符。
    feat_dim = int(bundle["hidden_size"]) * (2 if args.pooling == "mean_max" else 1)
    if fcache:
        cached_dim = next(iter(fcache.values()))["feats"].shape[-1]
        want = cached_dim // 2 if args.pooling == "mean" else cached_dim
        if want != feat_dim:
            raise SystemExit(f"缓存特征维 {cached_dim}(取 {args.pooling} 得 {want})"
                             f"与 bundle hidden_size 推出的 {feat_dim} 不符 —— "
                             f"缓存与模型不同源,拒绝继续")
    if args.head_arch == "linear" and args.pooling == "mean":
        head = load_v4_linear(bundle, arch="linear", hidden=args.hidden,
                              rank=args.pair_rank, use_context=False)
    else:
        # 形状与 v4 不同,只能随机初始化。**这不是缺陷**:v4 的头本身只到
        # 51.1%,继承它反而会把新架构锚在同一个坏解附近;但必须显式记录,
        # 否则读者会误以为新架构是"在 v4 基础上继续训"的。
        head = SubsetScorer(feat_dim, arch=args.head_arch, hidden=args.hidden,
                            rank=args.pair_rank, use_context=args.use_context)
    head.to(device).train()
    print(json.dumps({"head_arch": args.head_arch, "pooling": args.pooling,
                      "use_context": args.use_context, "feat_dim": feat_dim,
                      "params": sum(x.numel() for x in head.parameters()),
                      "init": "继承 v4" if (args.head_arch == "linear"
                                            and args.pooling == "mean") else "随机"},
                     ensure_ascii=False), flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    opt.zero_grad(set_to_none=True)

    def state_feats(rec):
        if fcache is not None:
            e = fcache.get(rec["dp_id"])
            if e is None:
                raise ValueError("not in feature cache")
            f = e["feats"].to(device).float()
            c = e["ctx"].to(device).float()
            if args.pooling == "mean":
                # 缓存按 mean_max 存;纯 mean 就是前半段
                half = f.shape[-1] // 2
                f, c = f[:, :half], c[:half]
            return e["cands"], f, (c if args.use_context else None)
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
        got = index_features(rec, steps, cands, ev, str(cur),
                             model=model, processor=index_processor,
                             device=device, torch=torch,
                             build=build_desktop_official_messages,
                             tool_spec=_TOOL_SPEC,
                             pooling=args.pooling,
                             return_context=args.use_context)
        feats, ctx = got if args.use_context else (got, None)
        return cands, feats, ctx

    def subset_score(u, pair, cands, subset):
        """子集分。加性档等价于 v4 的"成员帧分之和";mlp_pair 档另加成对项。

        # note (luojiaxuan): 有成对项后 top-B argmax **不再等价于最优子集**,
        # 部署侧的选择规则必须一并改成在 C(n,B) 上枚举打分
        # (`SubsetScorer.score_subsets`)。训练与部署口径若在这里脱钩,
        # 就是 v3 那个"训 log p、评 argmax"错误的翻版。
        """
        return head.subset_score(u, pair, [cands.index(j) for j in subset])

    stats = {"pairs": 0, "steps": 0, "loss": 0.0, "skipped": 0}
    # note (luojiaxuan): **训练集排序准确率**是这一轮最重要的仪表。v4 的教训是
    # 我们只看了留出集(0.5828),没看训练集 —— 而真正的病因是连训练集都没拟合上。
    # 前置门槛:训练集排序准确率上不去,就不必再评留出集,直接判该架构容量不足。
    fit_hits = 0
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
                cands, feats, ctx = cache.get(dp) or state_feats(rec)
                cache[dp] = (cands, feats, ctx)
                u, pair = head.frame_terms(feats, ctx)
                loss = torch.zeros((), device=device)
                k = 0
                for _ in range(args.pairs_per_state):
                    pos = rng.choice(lab[dp]["pos"])
                    neg = rng.choice(lab[dp]["neg"])
                    if not (set(pos) <= set(cands) and set(neg) <= set(cands)):
                        continue
                    margin = (subset_score(u, pair, cands, pos)
                              - subset_score(u, pair, cands, neg))
                    loss = loss - torch.nn.functional.logsigmoid(margin)
                    fit_hits += int(float(margin) > 0)
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
                        "train_rank_acc": round(fit_hits / max(stats["pairs"], 1), 4),
                        "skipped": stats["skipped"]}, ensure_ascii=False), flush=True)
            except (ValueError, KeyError, OSError, TypeError, RuntimeError):
                stats["skipped"] += 1
                opt.zero_grad(set_to_none=True)
                continue

    # ---- 训练集 / 留出集:排序准确率(正子集分 > 负子集分 的比例)----
    # note (luojiaxuan): 训练集这一遍是**训练结束后的静态复测**,不能用训练过程中
    # 累计的 fit_hits 代替 —— 那是滑动平均,早期未收敛时的错会一直压低它,
    # 用它当拟合门槛会把"其实拟合上了"误判成"容量不足"。特征已缓存,这一遍
    # 不需要重新前向,几乎不花时间。
    head.eval()

    def rank_acc(ids: set, use_cache: bool) -> tuple[int, int]:
        hits = tot = 0
        with torch.no_grad():
            for line in args.manifest.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                dp = rec["dp_id"]
                if dp not in ids:
                    continue
                try:
                    got = cache.get(dp) if use_cache else None
                    cands, feats, ctx = got or state_feats(rec)
                    u, pair = head.frame_terms(feats, ctx)
                    # 固定 seed 重放同样的配对抽样,训练集/留出集口径一致
                    r = random.Random(args.seed ^ hash(dp) & 0xFFFF)
                    for _ in range(args.pairs_per_state):
                        pos = r.choice(lab[dp]["pos"])
                        neg = r.choice(lab[dp]["neg"])
                        if not (set(pos) <= set(cands) and set(neg) <= set(cands)):
                            continue
                        tot += 1
                        hits += int(float(subset_score(u, pair, cands, pos))
                                    > float(subset_score(u, pair, cands, neg)))
                except (ValueError, KeyError, OSError, TypeError, RuntimeError):
                    continue
        return hits, tot

    fit_hits_final, fit_tot = rank_acc(train_ids, use_cache=True)

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
                cands, feats, ctx = state_feats(rec)
                u, pair = head.frame_terms(feats, ctx)
                for _ in range(args.pairs_per_state):
                    pos = rng.choice(lab[rec["dp_id"]]["pos"])
                    neg = rng.choice(lab[rec["dp_id"]]["neg"])
                    if not (set(pos) <= set(cands) and set(neg) <= set(cands)):
                        continue
                    tot += 1
                    hits += int(float(subset_score(u, pair, cands, pos))
                                > float(subset_score(u, pair, cands, neg)))
            except (ValueError, KeyError, OSError, TypeError, RuntimeError):
                continue

    out = dict(bundle)
    out["model_state"] = head.state_dict()
    # 部署侧要靠这些字段把头重建成同一个形状与同一套选择规则
    out["head_arch"] = args.head_arch
    out["pooling"] = args.pooling
    out["use_context"] = args.use_context
    out["hidden"] = args.hidden
    out["pair_rank"] = args.pair_rank
    out["feat_dim"] = feat_dim
    out["selection_rule"] = ("enumerate_subsets" if args.head_arch == "mlp_pair"
                             else "topb_argmax")
    torch.save(out, args.output_root / "selector_bundle.pt")
    train_acc = round(fit_hits_final / max(fit_tot, 1), 4)
    report = {"head_arch": args.head_arch, "pooling": args.pooling,
              "use_context": args.use_context,
              "params": sum(x.numel() for x in head.parameters()),
              "train_states": seen, "opt_steps": stats["steps"],
              "pairs": stats["pairs"], "skipped": stats["skipped"],
              "train_rank_acc": train_acc,
              "train_pairs": fit_tot,
              "train_rank_acc_running": round(fit_hits / max(stats["pairs"], 1), 4),
              "holdout_pairs": tot,
              "holdout_rank_acc": round(hits / max(tot, 1), 4),
              "fit_gate_pass": train_acc >= 0.85,
              "note": "排序准确率 0.5 = 与随机无异。**先看 train_rank_acc**:"
                      "它上不去就是容量不足,留出集数字无意义(v4 的教训:"
                      "只看留出集 0.5828,没发现训练集本身就没拟合上)"}
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
