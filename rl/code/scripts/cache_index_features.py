#!/usr/bin/env python3
"""把索引遍特征一次性缓存到磁盘,让后续架构消融不再反复过 8B 前向。

# note (luojiaxuan): 为什么值得单独做这一步 —— 打分头的容量升级要做**消融
# 梯度**(linear / mlp / mlp_pair × 上下文开关 × mean / mean_max),八种以上
# 组合。每种都重新过一遍冻结前向,等于把同一批图算八遍;而索引遍是
# `no_grad` 的,输出与打分头无关,天生该缓存。缓存后每个配置从约 1 小时
# 降到几分钟,消融才做得起。
#
# 同时这份缓存直接喂给**特征信息量探针**(TODO B2):在缓存上拟合一个强模型
# (GBDT / 深 MLP),若连它都拟合不上训练集,病因就不在打分头而在特征本身
# (144 token 缩略图 + 池化把判别信息丢了),那要动的是索引遍不是头。
# 两件事共用一份产物,不重复算。
#
# **一次性把两种池化都存下来**:mean 是 mean_max 的前半段,分开存会翻倍算力。
# 存 fp16:1263 态 × 约 10 帧 × 8192 维 ≈ 200MB,fp32 会到 400MB 而精度对
# 线性/MLP 打分头毫无意义。
#
# **当前屏特征必须一起存**。v4 一直把它丢掉 —— 因果注意力下候选帧排在当前屏
# 之前、看不到当前屏,于是打分函数的输入里根本没有"当前这一步长什么样",
# 而标签问的恰恰是"这帧对当前这一步有没有用"。它本来就在同一次前向里算好了。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True,
                   help="枚举产出的 jsonl;只缓存 winnable 态(既有正又有负子集)")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--index-visual-tokens", type=int, default=144)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--report-every", type=int, default=100)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
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

    want: set[str] = set()
    for f in args.labels:
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            subs = [bool(a["c"]) for a in d.get("all", [])
                    if len(a["s"]) == args.budget]
            if any(subs) and not all(subs):
                want.add(d["dp_id"])
    print(json.dumps({"winnable": len(want)}, ensure_ascii=False), flush=True)
    if not want:
        raise SystemExit("没有 winnable 态 —— 先把枚举跑出来")

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if args.output.exists():                      # 断点续跑
        cache = torch.load(args.output, map_location="cpu", weights_only=False)
        print(json.dumps({"resumed": len(cache)}, ensure_ascii=False), flush=True)

    skipped: dict[str, int] = {}
    done = 0
    for lineno, line in enumerate(args.manifest.open(encoding="utf-8")):
        line = line.strip()
        if not line or lineno % args.shard_count != args.shard_index:
            continue
        rec = json.loads(line)
        dp = rec["dp_id"]
        if dp not in want or dp in cache:
            continue
        try:
            s = int(rec["step"])
            images = rec["image_relpaths"]
            if len(images) != s:
                raise ValueError("image count")
            screen = tuple(int(x) for x in rec["screen_size"])
            steps = [official_step_forms(h, screen_size=screen) for h in rec["history"]]
            cands = [j for j in range(1, s - 1) if steps[j].full_response]
            cur = args.image_root / images[s - 1]
            if not cands or not cur.exists():
                raise ValueError("no candidates / missing current")
            if not all((args.image_root / images[j]).exists() for j in cands):
                raise ValueError("missing images")
            ev = {j: str(args.image_root / images[j]) for j in cands}
            feats, ctx = index_features(
                rec, steps, cands, ev, str(cur),
                model=model, processor=index_processor, device=device,
                torch=torch, build=build_desktop_official_messages,
                tool_spec=_TOOL_SPEC, pooling="mean_max", return_context=True)
            # note (luojiaxuan): mean 是 mean_max 的前半段,存一份即可,
            # 用的时候切片取 [:, :d//2] 就是纯 mean —— 别分开算两遍。
            cache[dp] = {"cands": cands,
                         "feats": feats.half().cpu(),
                         "ctx": ctx.half().cpu()}
            done += 1
            if done % args.report_every == 0:
                torch.save(cache, args.output)         # 定期落盘,断电只丢一段
                print(json.dumps({"cached": len(cache), "skipped": skipped},
                                 ensure_ascii=False), flush=True)
        except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
            k = type(exc).__name__ if not str(exc) else str(exc)[:40]
            skipped[k] = skipped.get(k, 0) + 1
            continue

    torch.save(cache, args.output)
    dim = next(iter(cache.values()))["feats"].shape[-1] if cache else 0
    print(json.dumps({"cached": len(cache), "want": len(want),
                      "feat_dim_mean_max": dim, "skipped": skipped,
                      "note": "取 [:, :dim//2] 即纯 mean 池化"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
