#!/usr/bin/env python3
"""缓存 token 级索引遍特征(selector v2 阶梯 2 的输入)。

# note (luojiaxuan): 与 cache_index_features.py 的差别只有一个:**不池化**。
# 池化缓存(mean/mean_max)已被证明学不出超过 recency 的东西
# (holdout 0.55-0.59 vs recency 规则 0.595,且不随数据量上行);
# 头号嫌疑是"两帧与当前屏的空间关系"在池化时被抹掉 —— 涌现解
# (31.6% 的可解态两帧单独都错)恰恰需要这种关系。本缓存保留每帧
# 完整的 ~144 个 token(fp16),供交叉注意力打分器用。
#
# **存储:每态一个文件**,而不是一个大 .pt —— ① 体积约 13MB/态 × 1263 ≈
# 16GB,单文件每次落盘要整体重写,断点续跑代价大;② 按文件存在性续跑,
# 多分片并发写零冲突。目录下另写 _manifest.jsonl 记录 skipped 计数。
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
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--index-visual-tokens", type=int, default=144)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--device", default="cuda:0")
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
        raise SystemExit("没有 winnable 标签")

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

    args.outdir.mkdir(parents=True, exist_ok=True)
    skipped: dict[str, int] = {}
    done = 0
    for lineno, line in enumerate(args.manifest.open(encoding="utf-8")):
        line = line.strip()
        if not line or lineno % args.shard_count != args.shard_index:
            continue
        rec = json.loads(line)
        dp = rec["dp_id"]
        out = args.outdir / f"{dp}.pt"
        if dp not in want or out.exists():
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
            toks, ctx = index_features(
                rec, steps, cands, ev, str(cur),
                model=model, processor=index_processor, device=device,
                torch=torch, build=build_desktop_official_messages,
                tool_spec=_TOOL_SPEC, pooling="tokens", return_context=True)
            # note (luojiaxuan): 先写临时名再原子改名 —— 多分片并发下,
            # 半写文件被别的进程当成品读走比重算一次贵得多。
            tmp = out.with_suffix(".tmp")
            torch.save({"cands": cands,
                        "toks": [t.half().cpu() for t in toks],
                        "ctx": ctx.half().cpu()}, tmp)
            tmp.rename(out)
            done += 1
            if done % 50 == 0:
                print(json.dumps({"shard": args.shard_index, "done": done,
                                  "skipped": skipped}, ensure_ascii=False), flush=True)
        except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
            k = type(exc).__name__ if not str(exc) else str(exc)[:40]
            skipped[k] = skipped.get(k, 0) + 1
            continue
    print(json.dumps({"shard": args.shard_index, "done": done,
                      "skipped": skipped, "finished": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
