#!/usr/bin/env python3
"""为 selector 抽取候选帧的视觉 embedding(每张唯一图片只算一次,离线缓存)。

动机(见 data/results/indomain_gap_v1/ROOT_CAUSE.md):
- 现有 selector 只有 28 维手工特征,学习曲线判定**偏差主导**——加标签、训更久都无用;
- 唯一带"当前意图"的特征是 witness(拟议动作),而它需要 pass-2 多跑一遍策略;
- 用图像 embedding 替代 witness,可以同时突破特征天花板**并免掉 pass-2**。

为什么可以离线缓存:embedding 只依赖图片本身,不依赖已选集合 S,也不依赖 beam 层级。
所以每张唯一图片算一次即可,部署时增量成本是"每步一次当前截图编码",而不是
"候选数 × beam 层数 × 视觉编码"——这是整个方案在 <40ms 预算下可行的前提。

直接调 `get_image_features`(Qwen3-VL 的视觉塔入口),不走完整 LM 前向。
输出:{relpath: [float]*out_hidden},分片 jsonl,断点续跑。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--screening-manifest", type=Path, required=True)
    p.add_argument("--singletons-root", type=Path, required=True,
                   help="从 b0 行读 candidate_pool,只抽真正会被打分的帧")
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--batch-images", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # ---- 收集本分片需要的唯一图片 ----
    records: dict[str, dict] = {}
    with open(args.screening_manifest, errors="ignore") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "dp_id" in r:
                records[r["dp_id"]] = r

    wanted: set[str] = set()
    for f in sorted(args.singletons_root.glob("singletons.shard*.jsonl")):
        with open(f, errors="ignore") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("kind") != "b0":
                    continue
                rec = records.get(r["dp_id"])
                if rec is None:
                    continue
                rel = rec["image_relpaths"]
                # note (luojiaxuan): 候选帧 + 当前帧都要。当前帧是 selector 的
                # "现在在看什么"侧输入,没有它就退化回与意图无关的特征。
                for e in r.get("candidate_pool", []):
                    if 0 <= int(e) < len(rel):
                        wanted.add(rel[int(e)])
                cur = int(rec["step"]) - 1
                if 0 <= cur < len(rel):
                    wanted.add(rel[cur])

    todo = sorted(wanted)
    todo = [p for i, p in enumerate(todo) if i % args.shard_count == args.shard_index]

    done: set[str] = set()
    if args.output.exists():
        with open(args.output, errors="ignore") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["relpath"])
                except Exception:
                    continue
    todo = [p for p in todo if p not in done]
    print(json.dumps({"unique_images_total": len(wanted), "shard_todo": len(todo),
                      "already_done": len(done)}), flush=True)
    if not todo:
        return

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to(args.device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(args.output, "a") as out:
        for start in range(0, len(todo), args.batch_images):
            chunk = todo[start:start + args.batch_images]
            images = []
            ok = []
            for rel in chunk:
                path = args.image_root / rel
                try:
                    from PIL import Image
                    images.append(Image.open(path).convert("RGB"))
                    ok.append(rel)
                except Exception as err:  # noqa: BLE001
                    print(json.dumps({"skip": rel, "error": str(err)[:120]}), flush=True)
            if not images:
                continue
            # note (luojiaxuan): Qwen3-VL 的 processor 必须同时给 text(它要在文本里
            # 找 image_token 做展开),只传 images 会 NoneType 崩。这里只需要像素张量,
            # 所以直接用 image_processor,绕过文本分支。
            inputs = processor.image_processor(images=images, return_tensors="pt")
            pv = inputs["pixel_values"].to(args.device, dtype=torch.bfloat16)
            grid = inputs["image_grid_thw"].to(args.device)
            with torch.no_grad():
                feats = model.get_image_features(pixel_values=pv, image_grid_thw=grid)
            seq = feats[0] if isinstance(feats, tuple) else getattr(feats, "last_hidden_state", feats)
            # note (luojiaxuan): get_image_features 返回的是**拼接后的**所有图片
            # patch 序列,必须按 grid 切回每张图,不能整体池化。
            counts = [int(g[0] * g[1] * g[2]) for g in grid.tolist()]
            merge = int(getattr(processor.image_processor, "merge_size", 2)) ** 2
            counts = [c // merge for c in counts]
            offset = 0
            for rel, n in zip(ok, counts):
                vec = seq[offset:offset + n].float().mean(dim=0)
                offset += n
                out.write(json.dumps({
                    "relpath": rel, "kind": "frame_embedding",
                    "dim": int(vec.numel()),
                    "vector": [round(float(x), 5) for x in vec.tolist()],
                }) + "\n")
            out.flush()
            if (start // args.batch_images) % 20 == 0:
                el = time.time() - started
                rate = (start + len(chunk)) / max(el, 1e-6)
                print(json.dumps({"done": start + len(chunk), "of": len(todo),
                                  "img_per_s": round(rate, 2),
                                  "eta_min": round((len(todo) - start) / max(rate, 1e-6) / 60, 1)}),
                      flush=True)


if __name__ == "__main__":
    main()
