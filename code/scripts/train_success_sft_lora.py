#!/usr/bin/env python3
"""LoRA SFT of the GUI-Owl policy on re-rendered success trajectories.

# note (luojiaxuan): 编码路径复用冻结 runtime 的 _encode_exact_batch,保证训练
# prompt 与闭环推理逐 token 一致;损失只作用于目标输出段;vision encoder 与全部
# 基座权重冻结,仅训练全 LM 层 q/k/v/o 的 LoRA。多卡用 torchrun DDP,样本按
# rank 切分,断点按 epoch 恢复。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.run_exploratory_closed_loop_episode import (
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
)
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["schema_version"] != "causalcache.success_sft_lora_config.v1":
        raise ValueError("unexpected SFT config schema")
    if (
        config["target_effective_visual_tokens_per_image"]
        != EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
    ):
        raise ValueError("SFT visual token budget drifted from the frozen runtime")
    return config


class LoRALinear:
    """In-place LoRA wrapper for a frozen nn.Linear via forward hook."""

    def __init__(self, module: Any, *, rank: int, alpha: int, torch: Any) -> None:
        self.module = module
        self.scaling = alpha / rank
        device = module.weight.device
        self.lora_a = torch.nn.Parameter(
            torch.zeros(rank, module.in_features, dtype=torch.float32, device=device)
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros(module.out_features, rank, dtype=torch.float32, device=device)
        )
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        x = inputs[0]
        delta = (
            x.to(self.lora_a.dtype) @ self.lora_a.T @ self.lora_b.T
        ) * self.scaling
        return output + delta.to(output.dtype)


def inject_lora(
    model: Any, *, rank: int, alpha: int, target_modules: tuple[str, ...], torch: Any
) -> dict[str, LoRALinear]:
    wrapped: dict[str, LoRALinear] = {}
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and name.split(".")[-1] in target_modules
            and ".visual." not in f".{name}."
            and ("language_model" in name or ".model.layers." in name)
        ):
            wrapped[name] = LoRALinear(module, rank=rank, alpha=alpha, torch=torch)
    if not wrapped:
        raise RuntimeError("LoRA injection matched no language-model modules")
    return wrapped


def lora_state_dict(wrapped: dict[str, LoRALinear]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, lora in wrapped.items():
        state[f"{name}.lora_a"] = lora.lora_a.detach().to(device="cpu")
        state[f"{name}.lora_b"] = lora.lora_b.detach().to(device="cpu")
    return state


def load_lora_state_dict(
    wrapped: dict[str, LoRALinear], state: dict[str, Any]
) -> None:
    expected = {
        f"{name}.{part}" for name in wrapped for part in ("lora_a", "lora_b")
    }
    if expected != set(state):
        raise ValueError("LoRA checkpoint key inventory drifted")
    import torch

    with torch.no_grad():
        for name, lora in wrapped.items():
            lora.lora_a.copy_(state[f"{name}.lora_a"].to(lora.lora_a.device))
            lora.lora_b.copy_(state[f"{name}.lora_b"].to(lora.lora_b.device))


def encode_sample(
    runtime: GUIOwlV21OfficialToolsRuntime,
    sample: dict[str, Any],
    *,
    dataset_root: Path,
    torch: Any,
) -> dict[str, Any] | None:
    messages = []
    opened: list[Any] = []
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
    tokenizer = runtime.processor.tokenizer
    target_ids = tokenizer(sample["target_text"], add_special_tokens=False)[
        "input_ids"
    ]
    if not target_ids:
        return None
    prompt_ids = model_inputs["input_ids"]
    prompt_length = int(prompt_ids.shape[1])
    device = prompt_ids.device
    target_tensor = torch.tensor([target_ids], dtype=prompt_ids.dtype, device=device)
    input_ids = torch.cat([prompt_ids, target_tensor], dim=1)
    labels = torch.cat(
        [
            torch.full(
                (1, prompt_length), -100, dtype=prompt_ids.dtype, device=device
            ),
            target_tensor,
        ],
        dim=1,
    )
    attention_mask = torch.ones_like(input_ids)
    encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    for key in ("pixel_values", "image_grid_thw"):
        if key in model_inputs:
            encoded[key] = model_inputs[key]
    if "mm_token_type_ids" in model_inputs:
        mm = model_inputs["mm_token_type_ids"]
        encoded["mm_token_type_ids"] = torch.cat(
            [
                mm,
                torch.zeros(
                    (1, int(target_tensor.shape[1])),
                    dtype=mm.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
    return encoded


def mean_target_logprob(model: Any, encoded: dict[str, Any], *, torch: Any) -> Any:
    # note (luojiaxuan): 目标段固定在序列末尾;logits_to_keep 只物化末端 logits,
    # 全长 float32 log_softmax 的 ~15GB 峰值降到 MB 级,H100 80GB 才放得下
    # margin 的双前向图。
    labels = encoded.pop("labels")
    targets_full = labels[:, 1:]
    token_count = int((targets_full != -100).sum())
    try:
        outputs = model(**encoded, logits_to_keep=token_count + 1)
    except TypeError:
        outputs = model(**encoded)
    logits = outputs.logits[:, -(token_count + 1) : -1].float()
    targets = targets_full[:, -token_count:]
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return gathered.sum() / token_count


def build_training_units(
    samples: list[dict[str, Any]], *, training: dict[str, Any]
) -> tuple[list[tuple[str, int, int | None]], set[str]]:
    """Return (units, heldout_episodes); units are (kind, idx, negative_idx)."""
    import hashlib as _hashlib

    fraction = float(training.get("heldout_episode_fraction", 0.0))
    salt = training.get("heldout_hash_salt", "")
    episodes = sorted({sample["episode"] for sample in samples})
    heldout = {
        episode
        for episode in episodes
        if fraction > 0.0
        and int.from_bytes(
            _hashlib.sha256(f"{salt}:{episode}".encode()).digest()[:4], "big"
        )
        / 2**32
        < fraction
    }
    negatives = tuple(training.get("margin_negatives", ()))
    by_group: dict[str, dict[str, int]] = {}
    for index, sample in enumerate(samples):
        if sample["episode"] in heldout:
            continue
        by_group.setdefault(sample["pair_group"], {})[
            sample.get("variant", "correct")
        ] = index
    units: list[tuple[str, int, int | None]] = []
    for group in by_group.values():
        if "correct" not in group:
            continue
        units.append(("ce", group["correct"], None))
        if "b0" in group and training.get("b0_ce_weight", 0.0) > 0.0:
            units.append(("ce_b0", group["b0"], None))
        if training.get("margin_lambda", 0.0) > 0.0:
            for negative in negatives:
                if negative in group:
                    units.append(("margin", group["correct"], group[negative]))
    return units, heldout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume-lora", type=Path, default=None)
    parser.add_argument("--start-epoch", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    config = load_config(args.config)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    manifest_path = args.dataset_root / "manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    samples = [
        json.loads(line)
        for line in (args.dataset_root / "samples.jsonl").open(encoding="utf-8")
    ]
    if not samples:
        raise ValueError("SFT dataset is empty")

    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / config["policy_snapshot_manifest"]
        ),
        device=device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    model = runtime.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    wrapped = inject_lora(
        model,
        rank=config["lora"]["rank"],
        alpha=config["lora"]["alpha"],
        target_modules=tuple(config["lora"]["target_modules"]),
        torch=torch,
    )
    parameters = [
        tensor for lora in wrapped.values() for tensor in (lora.lora_a, lora.lora_b)
    ]
    if args.resume_lora is not None:
        load_lora_state_dict(
            wrapped, torch.load(args.resume_lora, map_location="cpu")
        )
        if rank == 0:
            print(json.dumps({"resumed_from": str(args.resume_lora)}), flush=True)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    if rank == 0:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "causalcache.success_sft_lora_run.v1",
                    "config": config,
                    "dataset_manifest_sha256": manifest_sha,
                    "sample_count": len(samples),
                    "world_size": world_size,
                    "lora_module_count": len(wrapped),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    accumulation = config["training"]["gradient_accumulation_steps"]
    training_config = config["training"]
    margin_lambda = float(training_config.get("margin_lambda", 0.0))
    margin_value = float(training_config.get("margin_per_token", 0.0))
    b0_weight = float(training_config.get("b0_ce_weight", 1.0))
    units, heldout_episodes = build_training_units(
        samples, training=training_config
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "training_units": len(units),
                    "heldout_episodes": len(heldout_episodes),
                }
            ),
            flush=True,
        )
    ordering = random.Random(training_config["seed"])
    global_step = 0
    for epoch in range(args.start_epoch, training_config["epochs"]):
        order = list(range(len(units)))
        ordering.shuffle(order)
        shard = order[rank::world_size]
        model.train()
        running_loss = 0.0
        contributing = 0
        for position, unit_index in enumerate(shard):
            kind, sample_index, negative_index = units[unit_index]
            encoded = encode_sample(
                runtime,
                samples[sample_index],
                dataset_root=args.dataset_root,
                torch=torch,
            )
            if encoded is None:
                continue
            if kind == "margin":
                negative_encoded = encode_sample(
                    runtime,
                    samples[negative_index],
                    dataset_root=args.dataset_root,
                    torch=torch,
                )
                if negative_encoded is None:
                    continue
                positive_lp = mean_target_logprob(model, encoded, torch=torch)
                negative_lp = mean_target_logprob(
                    model, negative_encoded, torch=torch
                )
                unit_loss = margin_lambda * torch.relu(
                    margin_value - (positive_lp - negative_lp)
                )
            else:
                unit_loss = -mean_target_logprob(model, encoded, torch=torch)
                if kind == "ce_b0":
                    unit_loss = unit_loss * b0_weight
            loss = unit_loss / accumulation
            loss.backward()
            running_loss += float(unit_loss.detach())
            contributing += 1
            if (position + 1) % accumulation == 0:
                if world_size > 1:
                    for parameter in parameters:
                        if parameter.grad is not None:
                            dist.all_reduce(parameter.grad)
                            parameter.grad /= world_size
                torch.nn.utils.clip_grad_norm_(
                    parameters, config["training"]["max_grad_norm"]
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and global_step % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "mean_loss": running_loss / max(contributing, 1),
                            }
                        ),
                        flush=True,
                    )
                    running_loss = 0.0
                    contributing = 0
                if args.max_steps and global_step >= args.max_steps:
                    break
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            state = lora_state_dict(wrapped)
            checkpoint_path = args.output_root / f"lora-epoch{epoch + 1}.pt"
            torch.save(state, checkpoint_path)
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            print(
                json.dumps(
                    {
                        "epoch_complete": epoch + 1,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": digest,
                    }
                ),
                flush=True,
            )
        if args.max_steps and global_step >= args.max_steps:
            break
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({"training_complete": True, "global_steps": global_step}))


if __name__ == "__main__":
    main()
