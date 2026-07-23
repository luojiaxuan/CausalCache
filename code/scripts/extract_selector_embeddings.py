#!/usr/bin/env python3
"""Extract policy-aligned query states and event visual embeddings for the selector.

# note (luojiaxuan): 主 selector 的表示空间对齐 s75(标签由它定义)。query 分支:
# 对每个决策的 b0 prompt(instruction+低保真 summaries+当前图)做一次 s75 前向,
# 取末层 hidden states 三段池化(全序列/文本段/当前图段);event 分支:候选 post
# 图过冻结 vision tower 池化(delta = post−pre 在特征拼接时算)。全部 fp16 落盘,
# 提取一次,V0 基线与 V1 主模型共用。--role query|events 分开跑,支持分片续跑。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from scripts.run_exploratory_closed_loop_episode import (
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
)
from scripts.train_success_sft_lora import inject_lora, load_lora_state_dict
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime


def build_runtime(args, torch):
    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    runtime.model.eval()
    if args.lora_checkpoint is not None:
        wrapped = inject_lora(
            runtime.model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            torch=torch,
        )
        load_lora_state_dict(wrapped, torch.load(args.lora_checkpoint, map_location="cpu"))
    return runtime


def encode_prompt(runtime, sample, dataset_root):
    messages = []
    opened = []
    try:
        for message in sample["messages"]:
            content = []
            for part in message["content"]:
                if part["type"] == "image":
                    image = Image.open(dataset_root / part["path"]).convert("RGB")
                    opened.append(image)
                    content.append({"type": "image", "image": image})
                else:
                    content.append(part)
            messages.append({"role": message["role"], "content": content})
        model_inputs, _ = runtime._encode_exact_batch((messages,))
    finally:
        for image in opened:
            image.close()
    return model_inputs


def run_query(args, torch) -> None:
    runtime = build_runtime(args, torch)
    samples = [
        json.loads(line)
        for line in (args.dataset_root / "samples.jsonl").open(encoding="utf-8")
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in args.output.glob("*.pt")}
    core = runtime.model.model  # backbone without LM head; 末层 hidden 即所需
    for index, sample in enumerate(samples):
        if index % args.shard_count != args.shard_index:
            continue
        key = sample["pair_group"].replace(":", "__")
        if key in done:
            continue
        encoded = encode_prompt(runtime, sample, args.dataset_root)
        with torch.inference_mode():
            hidden = core(**encoded).last_hidden_state[0].float()
        mm = encoded.get("mm_token_type_ids")
        if mm is None:
            raise RuntimeError("mm_token_type_ids missing; segment pooling impossible")
        mask_img = mm[0].bool()
        pooled = {
            "full": hidden.mean(0),
            "text": hidden[~mask_img].mean(0),
            "current_image": hidden[mask_img].mean(0),
        }
        torch.save(
            {k: v.half().cpu() for k, v in pooled.items()},
            args.output / f"{key}.pt",
        )
    print(json.dumps({"role": "query", "shard": args.shard_index, "status": "done"}))


def run_events(args, torch) -> None:
    runtime = build_runtime(args, torch)
    wanted: set[tuple[str, int]] = set()
    for line in args.labels.open(encoding="utf-8"):
        row = json.loads(line)
        source_id = row["pair_group"].split(":")[0]
        for cand in row["candidates"]:
            step = int(cand)
            wanted.add((source_id, step))
            if step - 1 >= 0:
                wanted.add((source_id, step - 1))  # pre 图,delta 用
    items = sorted(wanted)
    args.output.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in args.output.glob("*.pt")}
    visual = runtime.model.model.visual
    processor = runtime.processor.image_processor
    for index, (source_id, step) in enumerate(items):
        if index % args.shard_count != args.shard_index:
            continue
        key = f"{source_id}__{step:03d}"
        if key in done:
            continue
        path = args.images_root / source_id / f"observation-{step:03d}.png"
        if not path.exists():
            continue
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        features = processor(images=[image], return_tensors="pt")
        pixel_values = features["pixel_values"].to(
            device=args.device, dtype=runtime.model.dtype
        )
        grid = features["image_grid_thw"].to(device=args.device)
        with torch.inference_mode():
            tokens = visual(pixel_values, grid_thw=grid).float()
        torch.save(tokens.mean(0).half().cpu(), args.output / f"{key}.pt")
    print(json.dumps({"role": "events", "shard": args.shard_index, "status": "done"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("query", "events"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    import torch

    if args.role == "query":
        if args.dataset_root is None:
            raise ValueError("--dataset-root required for query role")
        run_query(args, torch)
    else:
        if args.labels is None or args.images_root is None:
            raise ValueError("--labels and --images-root required for events role")
        run_events(args, torch)


if __name__ == "__main__":
    main()
