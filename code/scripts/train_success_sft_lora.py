#!/usr/bin/env python3
"""LoRA SFT of the GUI-Owl policy on re-rendered success trajectories.

# note (luojiaxuan): 编码路径复用冻结 runtime 的 _encode_exact_batch,保证训练
# prompt 与闭环推理逐 token 一致;损失只作用于目标输出段;vision encoder 与全部
# 基座权重冻结,仅训练全 LM 层 q/k/v/o 的 LoRA。多卡用 torchrun DDP,样本按
# rank 切分,断点按 epoch 恢复。config.adapter.adapter_type 缺省 full_policy_lora
# 走上述老路不变;history_gated_kv 走冻结契约 docs/history_gated_mainline_v1.md:
# 仅注入最后 N 层 k/v_proj 的 mask 门控 LoRA,训练单元为 pair-group,每组
# correct/b0/shuffled/irrelevant 四次前向(b0 在 ctx=None 下产生 detach 的冻结
# 参考 ℓ0),损失为 gate/contrast/anchor 三组 hinge+smooth_l1 加 LoRA L2,CE 权重 0。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
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


def adapter_settings(config: dict[str, Any]) -> tuple[str, dict[str, int] | None]:
    """Return (adapter_type, history-gated options); default keeps the old path."""
    adapter = config.get("adapter") or {}
    adapter_type = adapter.get("adapter_type", "full_policy_lora")
    if adapter_type == "full_policy_lora":
        return adapter_type, None
    if adapter_type != "history_gated_kv":
        raise ValueError(f"unsupported adapter_type {adapter_type!r}")
    match = re.fullmatch(r"last_([1-9]\d*)", str(adapter["layer_scope"]))
    if match is None:
        raise ValueError("history_gated_kv layer_scope must look like last_<n>")
    declared_targets = adapter.get("target_modules")
    if declared_targets is not None and tuple(declared_targets) != (
        "k_proj",
        "v_proj",
    ):
        raise ValueError("history_gated_kv target_modules are frozen to k/v_proj")
    return adapter_type, {
        "layer_count": int(match.group(1)),
        "rank": int(adapter["rank"]),
        "alpha": int(adapter["alpha"]),
    }


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


def heldout_episode_set(
    samples: list[dict[str, Any]], *, training: dict[str, Any]
) -> set[str]:
    fraction = float(training.get("heldout_episode_fraction", 0.0))
    salt = training.get("heldout_hash_salt", "")
    episodes = sorted({sample["episode"] for sample in samples})
    return {
        episode
        for episode in episodes
        if fraction > 0.0
        and int.from_bytes(
            hashlib.sha256(f"{salt}:{episode}".encode()).digest()[:4], "big"
        )
        / 2**32
        < fraction
    }


def build_training_units(
    samples: list[dict[str, Any]], *, training: dict[str, Any]
) -> tuple[list[tuple[str, int, int | None]], set[str]]:
    """Return (units, heldout_episodes); units are (kind, idx, negative_idx)."""
    heldout = heldout_episode_set(samples, training=training)
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


HISTORY_GATED_VARIANTS = ("correct", "b0", "shuffled", "irrelevant")


def build_history_gated_units(
    samples: list[dict[str, Any]], *, training: dict[str, Any]
) -> tuple[list[tuple[str, dict[str, int], None]], set[str]]:
    """Return pair-group units for history_gated_kv training.

    # note (luojiaxuan): 训练单元是整个 pair-group(variant -> sample index);
    # 组内必须有 correct,且至少一个 b0/shuffled/irrelevant 对照,否则该组
    # (如早期共享决策的 correct-only 组)没有任何损失项,直接不进训练单元。
    """
    heldout = heldout_episode_set(samples, training=training)
    by_group: dict[str, dict[str, int]] = {}
    for index, sample in enumerate(samples):
        if sample["episode"] in heldout:
            continue
        variant = sample.get("variant", "correct")
        if variant not in HISTORY_GATED_VARIANTS:
            continue
        by_group.setdefault(sample["pair_group"], {})[variant] = index
    units: list[tuple[str, dict[str, int], None]] = []
    for group in by_group.values():
        if "correct" not in group:
            continue
        if not any(variant in group for variant in HISTORY_GATED_VARIANTS[1:]):
            continue
        units.append(("history_group", group, None))
    return units, heldout


def history_sample_context(
    encoded: dict[str, Any], sample: dict[str, Any], *, merge_size: int
) -> Any:
    """Build the fail-closed adapter context for one encoded sample, or None.

    # note (luojiaxuan): mask 覆盖 prompt 里恢复历史图 1..K 的 image token,
    # K 只取自样本 memory_config.restored_event_step_ids(b0 的 K=0 天然得到
    # ctx=None = 完全 bypass);构造后再断言 mask 与末尾目标段不相交,任何
    # 几何不一致由 build_history_token_mask 直接抛错拒绝,绝不带病前向。
    """
    from causalcache.policy.history_adapter_context import HistoryAdapterContext
    from causalcache.policy.history_token_roles import (
        assert_mask_disjoint,
        build_history_token_mask,
    )

    if "mm_token_type_ids" not in encoded or "image_grid_thw" not in encoded:
        raise ValueError(
            "history_gated_kv requires mm_token_type_ids and image_grid_thw"
        )
    history_count = len(sample["memory_config"]["restored_event_step_ids"])
    mask = build_history_token_mask(
        encoded["input_ids"],
        encoded["mm_token_type_ids"],
        encoded["image_grid_thw"],
        history_count,
        merge_size,
    )
    target_length = int((encoded["labels"] != -100).sum())
    assert_mask_disjoint(mask, int(encoded["input_ids"].shape[1]) - target_length)
    if history_count == 0:
        return None
    return HistoryAdapterContext(
        history_token_mask=mask,
        history_present=True,
        image_roles=("history",) * history_count + ("current",),
    )


def history_group_unit_loss(
    runtime: GUIOwlV21OfficialToolsRuntime,
    samples: list[dict[str, Any]],
    group: dict[str, int],
    *,
    dataset_root: Path,
    training: dict[str, Any],
    adapter_parameters: list[Any],
    merge_size: int,
    accumulation: int,
    torch: Any,
) -> Any:
    """One pair-group forward set and its history-gated loss, or None to skip.

    # note (luojiaxuan): 冻结契约的组损失:ℓc 在自身 mask scope 下带梯度前向;
    # ℓ0 是 b0 prompt 在 ctx=None 下的冻结参考,no_grad+detach 不回传;ℓs/ℓi
    # 在各自 mask scope 下带梯度前向。损失 = gate*relu(m-(ℓc-ℓ0))
    # + contrast*[relu(m-(ℓc-ℓs))+relu(m-(ℓc-ℓi))]
    # + anchor*[smooth_l1(ℓs,ℓ0)+smooth_l1(ℓi,ℓ0)] + l2*Σ(‖A‖²+‖B‖²),
    # CE 权重缺省 0;组内缺哪个变体就跳过含它的项。
    """
    from causalcache.policy.history_adapter_context import history_adapter_scope

    def forward(variant: str, *, grad: bool) -> Any:
        index = group.get(variant)
        if index is None:
            return None
        sample = samples[index]
        encoded = encode_sample(
            runtime, sample, dataset_root=dataset_root, torch=torch
        )
        if encoded is None:
            return None
        context = history_sample_context(encoded, sample, merge_size=merge_size)
        if variant == "b0" and context is not None:
            raise ValueError("b0 variant carries restored history images")
        with history_adapter_scope(context):
            if grad:
                return mean_target_logprob(runtime.model, encoded, torch=torch)
            with torch.no_grad():
                return mean_target_logprob(
                    runtime.model, encoded, torch=torch
                ).detach()

    correct_lp = forward("correct", grad=False)
    if correct_lp is None:
        return None
    b0_lp = forward("b0", grad=False)
    shuffled_lp = forward("shuffled", grad=False)
    irrelevant_lp = forward("irrelevant", grad=False)

    margin = float(training.get("history_gate_margin", 0.01))
    gate_weight = float(training.get("history_gate_weight", 1.0))
    contrast_weight = float(training.get("history_contrast_weight", 1.0))
    anchor_weight = float(training.get("history_anchor_weight", 1.0))
    l2_weight = float(training.get("history_lora_l2_weight", 1e-4))
    ce_weight = float(training.get("history_ce_weight", 0.0))

    # note (luojiaxuan): 两遍法。第一遍 no-grad 取各变体 ℓ 值并按 hinge/huber
    # 求每个前向的次梯度权重;第二遍逐变体在各自 scope 内带梯度前向并立即
    # backward(权重×ℓ/accum)。同时只活一张计算图(显存),且梯度检查点的
    # 重算发生在 scope 内(修复 ContextVar 与 checkpoint 的张量数不一致崩溃)。
    lc = float(correct_lp)
    l0 = float(b0_lp) if b0_lp is not None else None
    ls = float(shuffled_lp) if shuffled_lp is not None else None
    li = float(irrelevant_lp) if irrelevant_lp is not None else None

    weight_c = 0.0
    weight_s = 0.0
    weight_i = 0.0
    total_value = 0.0
    if ce_weight > 0.0:
        weight_c += -ce_weight
        total_value += ce_weight * (-lc)
    if l0 is not None and (margin - (lc - l0)) > 0.0:
        weight_c += -gate_weight
        total_value += gate_weight * (margin - (lc - l0))
    for value, tag in ((ls, "s"), (li, "i")):
        if value is None:
            continue
        if (margin - (lc - value)) > 0.0:
            weight_c += -contrast_weight
            if tag == "s":
                weight_s += contrast_weight
            else:
                weight_i += contrast_weight
            total_value += contrast_weight * (margin - (lc - value))
        if l0 is not None:
            diff = value - l0
            huber_grad = max(-1.0, min(1.0, diff))
            huber_value = 0.5 * diff * diff if abs(diff) < 1.0 else abs(diff) - 0.5
            if tag == "s":
                weight_s += anchor_weight * huber_grad
            else:
                weight_i += anchor_weight * huber_grad
            total_value += anchor_weight * huber_value
    if total_value == 0.0 and l2_weight <= 0.0:
        return None
    for variant, weight in (
        ("correct", weight_c),
        ("shuffled", weight_s),
        ("irrelevant", weight_i),
    ):
        if weight == 0.0:
            continue
        lp = forward(variant, grad=True)
        if lp is None:
            continue
        ((weight / accumulation) * lp).backward()
    if l2_weight > 0.0:
        l2_term = l2_weight * sum(
            parameter.pow(2).sum() for parameter in adapter_parameters
        )
        total_value += float(l2_term.detach())
        (l2_term / accumulation).backward()
    return total_value


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
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
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
    adapter_type, adapter_options = adapter_settings(config)
    if adapter_type == "history_gated_kv":
        from causalcache.policy.history_gated_lora import (
            history_gated_state_dict,
            inject_history_gated_kv,
            load_history_gated_state_dict,
        )

        wrapped = inject_history_gated_kv(
            model,
            layer_count=adapter_options["layer_count"],
            rank=adapter_options["rank"],
            alpha=adapter_options["alpha"],
        )
        adapter_state_dict = history_gated_state_dict
        adapter_load_state_dict = load_history_gated_state_dict
        merge_size = int(runtime.processor.image_processor.merge_size)
    else:
        wrapped = inject_lora(
            model,
            rank=config["lora"]["rank"],
            alpha=config["lora"]["alpha"],
            target_modules=tuple(config["lora"]["target_modules"]),
            torch=torch,
        )
        adapter_state_dict = lora_state_dict
        adapter_load_state_dict = load_lora_state_dict
        merge_size = None
    parameters = [
        tensor for lora in wrapped.values() for tensor in (lora.lora_a, lora.lora_b)
    ]
    if args.resume_lora is not None:
        adapter_load_state_dict(
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
    if adapter_type == "history_gated_kv":
        units, heldout_episodes = build_history_gated_units(
            samples, training=training_config
        )
    else:
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
            kind, payload, negative_index = units[unit_index]
            if kind == "history_group":
                unit_value = history_group_unit_loss(
                    runtime,
                    samples,
                    payload,
                    dataset_root=args.dataset_root,
                    training=training_config,
                    adapter_parameters=parameters,
                    merge_size=merge_size,
                    accumulation=accumulation,
                    torch=torch,
                )
                if unit_value is None:
                    continue
                running_loss += unit_value
                contributing += 1
                unit_loss = None
            else:
                encoded = encode_sample(
                    runtime,
                    samples[payload],
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
            if unit_loss is not None:
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
                if (
                    rank == 0
                    and args.checkpoint_every_steps
                    and global_step % args.checkpoint_every_steps == 0
                ):
                    step_path = (
                        args.output_root / f"lora-step{global_step}.pt"
                    )
                    torch.save(adapter_state_dict(wrapped), step_path)
                    print(
                        json.dumps(
                            {"step_checkpoint": global_step, "path": str(step_path)}
                        ),
                        flush=True,
                    )
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
            state = adapter_state_dict(wrapped)
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
