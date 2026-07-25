#!/usr/bin/env python3
"""LoRA SFT of the GUI-Owl policy on re-rendered success trajectories.

# note (luojiaxuan): 编码路径复用冻结 runtime:v2.1 私有样本走 _encode_exact_batch,
# official_multiturn / sparse_single_turn 走 apply_chat_template,但两条路径都跑
# prompt_aligned_input_keys 与 assert_pinned_assistant_prefix,"训练 prompt 与闭环
# 推理逐 token 一致"因此始终有机器校验;损失只作用于目标输出段;vision encoder 与全部
# 基座权重冻结,仅训练全 LM 层 q/k/v/o 的 LoRA。多卡用 torchrun DDP,样本按
# rank 切分,断点按 epoch 恢复。config.adapter.adapter_type 缺省 full_policy_lora
# 走上述老路不变;history_gated_kv 走冻结契约 docs/history_gated_mainline_v1.md:
# 仅注入最后 N 层 k/v_proj 的 mask 门控 LoRA,训练单元为 pair-group,每组
# correct/b0/shuffled/irrelevant 四次前向(b0 在 ctx=None 下产生 detach 的冻结
# 参考 ℓ0),损失为 gate/contrast/anchor 三组 hinge+smooth_l1 加 LoRA L2,CE 权重 0。
#
# training.sparse_history=true 时走独立的 sparse-history 五臂分支
# (schema causalcache.sparse_history_sample.v2),它与上面两条旧路径完全不共享
# 分组、损失与超参解析代码,旧冻结契约因此逐字节不受影响。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
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


_CHAT_TEMPLATE_PROMPT_FORMATS = ("official_multiturn", "sparse_single_turn")
_EXACT_BATCH_PROMPT_FORMATS = ("v2_1_private",)


def _prompt_encoding_path(sample: dict[str, Any]) -> str:
    """Resolve the encoder from the sample's declared prompt_format.

    # note (luojiaxuan): 只有 v2.1 老样本允许缺 prompt_format(它们那时还没有这个
    # 字段),缺省即私有 exact-batch 路径,旧行为逐字节不变。sparse-history v1 样本
    # 带 selected_steps 却没有 prompt_format,这里 fail-closed 报错要求重建为 v2,
    # 而不是像旧代码那样按字段存在性猜一个编码器。
    """
    prompt_format = sample.get("prompt_format")
    if prompt_format is None:
        if "selected_steps" in sample or "sparse" in sample:
            raise ValueError(
                "sparse-history sample lacks prompt_format; rebuild the corpus as "
                f"{SPARSE_SAMPLE_SCHEMA}"
            )
        return "exact_batch"
    if prompt_format in _CHAT_TEMPLATE_PROMPT_FORMATS:
        return "chat_template"
    if prompt_format in _EXACT_BATCH_PROMPT_FORMATS:
        return "exact_batch"
    raise ValueError(f"unknown prompt_format {prompt_format!r}")


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
        # note (luojiaxuan): 编码路径由样本显式声明的 prompt_format 决定,不再嗅探
        # 字段存在性(审计 P0-4:"sample.get('sparse') is not None or 'selected_steps'
        # in sample" 这种推断会让一条字段半缺的样本悄悄走错编码器)。
        # official_multiturn / sparse_single_turn 都用**官方** system prompt,而
        # _encode_exact_batch 会按 v2.1 私有契约校验(system message drifted),
        # 故这两类走 processor.apply_chat_template,编码参数与冻结路径一致,只跳过
        # 那条针对 v2.1 的结构校验;无 prompt_format 的 v2.1 老样本仍走原路。
        if _prompt_encoding_path(sample) == "chat_template":
            # note (luojiaxuan): 这里**只**跳过 _encode_exact_batch 里那条 v2.1 私有
            # 结构校验和 tools=_official_tools_argument()(官方 system prompt 已内嵌
            # <tools>,再传一次会改 prompt);另外两项与 prompt 格式无关的保真检查
            # 必须照跑,否则模块开头"训练 prompt 与闭环推理逐 token 一致"在五臂上
            # 就退化成一句无人校验的注释。helper 一律复用冻结 runtime 的实现:
            # prompt_aligned_input_keys 是 runtime 已导出的模块级函数,收尾断言用
            # runtime.assert_pinned_assistant_prefix(与 exact-batch 路径同一份代码)。
            # 惰性 import 是为了让本模块的顶层导入面保持不变。
            from causalcache.policy.gui_owl_v2_1_runtime import (
                prompt_aligned_input_keys,
            )

            encoded_batch = runtime.processor.apply_chat_template(
                [messages],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=False,
            )
            model_inputs = dict(
                encoded_batch.to(runtime.device)
                if hasattr(encoded_batch, "to")
                else {k: v.to(runtime.device) for k, v in encoded_batch.items()}
            )
            prompt_aligned_input_keys(model_inputs)
            runtime.assert_pinned_assistant_prefix(model_inputs["input_ids"], 1)
        else:
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


# ---------------------------------------------------------------------------
# sparse-history 五臂契约(schema causalcache.sparse_history_sample.v2)
# ---------------------------------------------------------------------------
# note (luojiaxuan): 审计 P0-4 的修复。"是否开 adapter / 谁是 reference / 谁是负
# 样本 / 属于哪个 split" 全部只读样本的显式字段;下表**只用于校验**样本自称的
# 字段是否与 arm_slot 主键自洽,任何控制流都不得解析 variant 或名字前缀。
# 列含义:arm_slot -> (arm_id, role, prompt_format, selection_mode, adapter_mode)
SPARSE_SAMPLE_SCHEMA = "causalcache.sparse_history_sample.v2"
SPARSE_ARM_CONTRACT: dict[str, tuple[str, str, str, str, str]] = {
    "N0": ("N0", "deployment_baseline", "official_multiturn", "recent", "bypass"),
    "R0": ("R0", "reference", "sparse_single_turn", "recent", "bypass"),
    "S0": ("S0", "measurement", "sparse_single_turn", "sparse", "bypass"),
    "RA": ("RA", "measurement", "sparse_single_turn", "recent", "active"),
    "SA": ("SA", "positive", "sparse_single_turn", "sparse", "active"),
    "SA_neg_step_shuffled": (
        "SA", "negative", "sparse_single_turn", "sparse", "active",
    ),
    "SA_neg_irrelevant": (
        "SA", "negative", "sparse_single_turn", "sparse", "active",
    ),
    "SA_neg_duplicate": (
        "SA", "negative", "sparse_single_turn", "sparse", "active",
    ),
}
SPARSE_POSITIVE_SLOT = "SA"
SPARSE_NEGATIVE_KINDS = ("step_shuffled", "irrelevant", "duplicate")
SPARSE_REQUIRED_KEYS = (
    "schema_version", "sample_id", "pair_group", "episode", "decision_step",
    "arm_slot", "arm_id", "role", "prompt_format", "selection_mode",
    "adapter_mode", "budget", "selected_steps", "selected_images",
    "current_image", "target_text", "messages", "split", "reference_arm_id",
    "deployment_baseline_arm_id",
)
# 旧字段一旦出现说明样本没迁完:"sparse" 混淆了 prompt_format 与 selection_mode,
# "reference_variant" 指向 native_recent{K}(与 trainer 实际参考臂矛盾)。
SPARSE_FORBIDDEN_KEYS = ("sparse", "reference_variant", "_needs_donor")
# note (luojiaxuan): 每 25 组打印一次趋势用的滚动窗口,**不是**诊断的对外通道——
# 调用方要拿本组诊断请给 _sparse_history_group_loss 传 diagnostics_out(审计第 10 条)。
_SPARSE_DIAG: list[dict[str, float]] = []


def _sparse_image_part_count(sample: dict[str, Any]) -> int:
    return sum(
        1
        for message in sample["messages"]
        for part in message["content"]
        if part.get("type") == "image"
    )


def validate_sparse_sample(sample: dict[str, Any], *, index: int) -> None:
    """Fail-closed field validation for one sparse-history v2 sample."""
    where = f"samples[{index}]"
    if sample.get("schema_version") != SPARSE_SAMPLE_SCHEMA:
        raise ValueError(
            f"{where} schema_version {sample.get('schema_version')!r} is not "
            f"{SPARSE_SAMPLE_SCHEMA}"
        )
    missing = [key for key in SPARSE_REQUIRED_KEYS if key not in sample]
    if missing:
        raise ValueError(f"{where} misses required fields {missing}")
    stale = [key for key in SPARSE_FORBIDDEN_KEYS if key in sample]
    if stale:
        raise ValueError(f"{where} still carries retired fields {stale}")
    slot = sample["arm_slot"]
    contract = SPARSE_ARM_CONTRACT.get(slot)
    if contract is None:
        raise ValueError(f"{where} unknown arm_slot {slot!r}")
    declared = (
        sample["arm_id"], sample["role"], sample["prompt_format"],
        sample["selection_mode"], sample["adapter_mode"],
    )
    if declared != contract:
        raise ValueError(
            f"{where} arm_slot {slot!r} declares {declared} but the frozen "
            f"five-arm contract requires {contract}"
        )
    if sample["sample_id"] != f"{sample['pair_group']}|{slot}":
        raise ValueError(f"{where} sample_id must be '<pair_group>|<arm_slot>'")
    if sample["split"] not in ("train", "heldout"):
        raise ValueError(f"{where} unknown split {sample['split']!r}")
    if sample["reference_arm_id"] != "R0":
        raise ValueError(f"{where} reference_arm_id must be 'R0'")
    if sample["deployment_baseline_arm_id"] != "N0":
        raise ValueError(f"{where} deployment_baseline_arm_id must be 'N0'")
    budget = sample["budget"]
    if type(budget) is not int or budget < 1:
        raise ValueError(f"{where} budget must be a positive int")
    steps = sample["selected_steps"]
    if len(steps) != budget or len(sample["selected_images"]) != budget:
        raise ValueError(
            f"{where} budget {budget} differs from len(selected_steps)="
            f"{len(steps)} / len(selected_images)={len(sample['selected_images'])}"
        )
    if any(type(step) is not int for step in steps):
        raise ValueError(f"{where} selected_steps must be ints")
    if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
        raise ValueError(f"{where} selected_steps must be strictly increasing")
    decision_step = sample["decision_step"]
    if type(decision_step) is not int or decision_step < 1:
        raise ValueError(f"{where} decision_step must be a positive int")
    if steps[0] < 1 or steps[-1] >= decision_step:
        raise ValueError(
            f"{where} selected_steps must be 1-based and older than decision_step"
        )
    images = _sparse_image_part_count(sample)
    if images != budget + 1:
        raise ValueError(
            f"{where} carries {images} image parts, expected budget+1={budget + 1}"
        )
    if sample["role"] != "negative":
        return
    kind = sample.get("negative_kind")
    if kind not in SPARSE_NEGATIVE_KINDS:
        raise ValueError(f"{where} unknown negative_kind {kind!r}")
    if slot != f"SA_neg_{kind}":
        raise ValueError(f"{where} arm_slot {slot!r} disagrees with negative_kind")
    scale = sample.get("negative_scale")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
        raise ValueError(f"{where} negative_scale must be a positive number")
    if kind != "irrelevant":
        return
    donor = sample.get("donor_episode")
    if not isinstance(donor, str) or not donor or donor == sample["episode"]:
        raise ValueError(f"{where} irrelevant negative needs a foreign donor_episode")


def sparse_reference_slot(
    samples: list[dict[str, Any]], group: dict[str, int]
) -> str:
    """Reference arm slot read from reference_arm_id — never from a name prefix."""
    declared = {samples[index]["reference_arm_id"] for index in group.values()}
    if len(declared) != 1:
        raise ValueError(
            f"pair-group disagrees on reference_arm_id: {sorted(declared)}"
        )
    return declared.pop()


def validate_sparse_group(
    samples: list[dict[str, Any]],
    *,
    pair_group: str,
    group: dict[str, int],
    label: str = "train",
) -> str:
    """Check one pair-group and return its reference arm slot.

    # note (luojiaxuan): ``label`` 只进错误消息(train / heldout),让同一份组内一致性
    # 校验既服务训练单元也服务留出集打分,而不是让打分脚本抄一份弱化版校验。
    """
    if SPARSE_POSITIVE_SLOT not in group:
        raise ValueError(
            f"{label} pair-group {pair_group!r} lacks the SA arm; "
            f"slots={sorted(group)}"
        )
    positive = samples[group[SPARSE_POSITIVE_SLOT]]
    ref_slot = sparse_reference_slot(samples, group)
    if ref_slot not in group:
        raise ValueError(
            f"{label} pair-group {pair_group!r} declares reference_arm_id "
            f"{ref_slot!r} but that arm is absent; slots={sorted(group)}"
        )
    reference = samples[group[ref_slot]]
    if (
        reference["adapter_mode"] != "bypass"
        or reference["role"] != "reference"
        or reference["prompt_format"] != "sparse_single_turn"
        or reference["selection_mode"] != "recent"
    ):
        raise ValueError(
            f"pair-group {pair_group!r} reference arm {ref_slot!r} is not a "
            "budget-matched, same-format, adapter-bypassed recent-K arm"
        )
    for slot, index in sorted(group.items()):
        sample = samples[index]
        if sample["target_text"] != positive["target_text"]:
            raise ValueError(
                f"pair-group {pair_group!r} arm {slot!r} target_text differs from SA"
            )
        if int(sample["budget"]) != int(positive["budget"]):
            raise ValueError(
                f"pair-group {pair_group!r} arm {slot!r} budget "
                f"{sample['budget']} differs from SA budget {positive['budget']}"
            )
        if (
            sample["episode"] != positive["episode"]
            or sample["decision_step"] != positive["decision_step"]
        ):
            raise ValueError(
                f"pair-group {pair_group!r} arm {slot!r} points at another decision"
            )
    return ref_slot


SPARSE_HELDOUT_REQUIRED_SLOTS = ("N0", "R0", "S0", "RA", "SA")


def _sparse_groups_by_split(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, int]]], set[str]]:
    """Validate every sample once and bucket its pair-group by the split field.

    # note (luojiaxuan): 训练单元与留出集打分共用这一遍校验/去重/分桶,免得两处对
    # "什么算一个组、谁属于哪个 split" 给出不同答案(split 的唯一权威是样本的
    # ``split`` 字段,trainer 与 scorer 都不再重算 hash)。
    """
    seen: set[str] = set()
    buckets: dict[str, dict[str, dict[str, int]]] = {"train": {}, "heldout": {}}
    heldout_episodes: set[str] = set()
    for index, sample in enumerate(samples):
        validate_sparse_sample(sample, index=index)
        sample_id = sample["sample_id"]
        if sample_id in seen:
            raise ValueError(f"duplicate (pair_group, arm_slot) {sample_id!r}")
        seen.add(sample_id)
        split = sample["split"]
        if split == "heldout":
            heldout_episodes.add(sample["episode"])
        buckets[split].setdefault(sample["pair_group"], {})[
            sample["arm_slot"]
        ] = index
    return buckets, heldout_episodes


def build_sparse_history_units(
    samples: list[dict[str, Any]],
) -> tuple[list[tuple[str, dict[str, int], None]], set[str]]:
    """Return (units, heldout_episodes) for the sparse-history five-arm corpus.

    # note (luojiaxuan): split 的唯一权威是样本的 ``split`` 字段(审计 P0-4);
    # trainer 不再调用 heldout_episode_set 重算 hash——构建期与训练期各算一次
    # hash 是留出集悄悄漂移的经典成因。索引主键是 arm_slot,重复即报错。
    """
    buckets, heldout_episodes = _sparse_groups_by_split(samples)
    units: list[tuple[str, dict[str, int], None]] = []
    for pair_group, group in sorted(buckets["train"].items()):
        validate_sparse_group(samples, pair_group=pair_group, group=group)
        units.append(("sparse_group", group, None))
    return units, heldout_episodes


def build_sparse_history_heldout_units(
    samples: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Return the heldout pair-groups the gate scorer measures, keyed by pair_group.

    # note (luojiaxuan): 审计第 9 条——gate 要从"只被 schema 校验"变成"真的被执行",
    # 就得有一处按同一份 split 权威(样本的 ``split`` 字段)取出留出组,而不是让打分
    # 脚本自己再筛一遍。留出组的要求比训练组更严:五臂 N0/R0/S0/RA/SA 必须齐全,
    # 缺 RA 主 claim(SA-RA)无定义、缺 N0 deployment_delta 无定义。这里直接报错而不是
    # 静默少算——"一部分组没进主 claim 的分母"是最难被发现的评测偏差。
    """
    buckets, _ = _sparse_groups_by_split(samples)
    groups: dict[str, dict[str, int]] = {}
    for pair_group, group in sorted(buckets["heldout"].items()):
        validate_sparse_group(
            samples, pair_group=pair_group, group=group, label="heldout"
        )
        missing = [
            slot for slot in SPARSE_HELDOUT_REQUIRED_SLOTS if slot not in group
        ]
        if missing:
            raise ValueError(
                f"heldout pair-group {pair_group!r} lacks arms {missing}; every "
                "scored group needs all five arms or its derived quantities are "
                "undefined"
            )
        groups[pair_group] = group
    return groups


def sparse_diagnostic_keys(negative_kind: str) -> tuple[str, str]:
    """Return (rank-gap key, drift key) for one negative kind — the gate spelling.

    # note (luojiaxuan): 审计第 10 条。损失里原来写 ``SA_minus_step_shuffled`` /
    # ``step_shuffled_drift``,而 gates 词表与 config.gates.must_pass 写的是
    # ``SA_minus_SA_neg_step_shuffled`` / ``step_shuffled_drift_abs``,于是"诊断可
    # 直接与 config.gates 的量名对齐"那句注释是假的:两套名字谁也对不上谁。现在
    # **名字只有这一处定义**,损失的诊断键与下面的 SPARSE_GATE_VOCABULARY /
    # SPARSE_REQUIRED_GATES 都由它生成,对齐是结构性的而不是靠人肉同步。
    # 记法统一到 arm_slot:负样本臂的 arm_slot 就是 ``SA_neg_<kind>``,所以差值量
    # 名 = ``SA_minus_<arm_slot>``。
    """
    return f"SA_minus_SA_neg_{negative_kind}", f"{negative_kind}_drift_abs"


def _sparse_history_group_loss(
    *,
    samples: list[dict[str, Any]],
    group: dict[str, int],
    forward: Any,
    training: dict[str, Any],
    adapter_parameters: list[Any],
    accumulation: int,
    torch: Any,
    diagnostics_out: dict[str, float] | None = None,
) -> float | None:
    """Sparse-history objective against the budget-matched R0 reference.

    # note (luojiaxuan): 记号 ℓc=SA(adapter active,带梯度)、ℓr=R0(同格式、
    # 同预算、最近 K 张、adapter bypass,冻结无梯度)、ℓn=各负样本 adapter active、
    # ℓn⁰=同一负样本 adapter bypass 的冻结分数。w_n = scale_n / Σ_m scale_m。
    #   L_gain  = gain_w  * [m_g - (ℓc - ℓr)]+
    #   L_rank  = rank_w  * Σ_n w_n * [m_r - (ℓc - ℓn)]+
    #   L_drift = drift_w * Σ_n w_n * SmoothL1(ℓn - ℓn⁰, 0)
    #   L       = L_gain + L_rank + L_drift + l2 * Σ(‖A‖²+‖B‖²),CE 权重恒为 0。
    #
    # 审计 P1-4:旧式 anchor 是 SmoothL1(ℓn_active, ℓr_frozen),等价于假设
    # shuffled/irrelevant/duplicate 在**冻结模型**上本就该等于 recent reference。
    # 这条假设不成立(冻结模型对这些条件本来就有各自的偏好),于是 anchor 会把
    # adapter 往一个错误的常数上拽,并与 rank 项直接对冲。现在每个负样本锚到**它
    # 自己**的冻结分数:只惩罚 adapter 造成的漂移,不再规定负样本该落在哪。
    # 代价是每个负样本多一次 bypass 前向(no_grad),K=3 组由 9 次前向变成 12 次。
    #
    # 审计:负样本按**实际存在**项归一化。K=1 组的 shuffled/duplicate 与 SA 逐字节
    # 相同因而不入库,若仍按固定 scale 求和,这些组的负样本项质量只有别组的 40%,
    # 等于按预算给梯度加权。归一化后每组的负样本总质量恒为 1。
    #
    # 两遍法保持不变:先 no-grad 取全部 ℓ 值、解析求各前向的次梯度权重,再逐臂在
    # 各自 history_adapter_scope 内带梯度前向并立即 backward,同一时刻只活一张
    # 计算图(这是为规避梯度检查点重算跑在 autograd 线程时 ContextVar 读空的崩溃)。
    """
    ref_slot = sparse_reference_slot(samples, group)
    # note (luojiaxuan): 审计第 8 条。缺 R0 时旧代码走 forward("R0") → None →
    # ``return None``,一组组静默跳过:主 claim SA−RA 的分母悄悄变小,而 stdout 上
    # 只会看到组数变少。上游 validate_sparse_group 确实也查这一条,但损失函数是被
    # 测试与将来其它调用方直接调用的入口,不能把 fail-closed 外包给调用方。
    if ref_slot not in group:
        raise ValueError(
            f"pair-group declares reference_arm_id {ref_slot!r} but that arm is "
            f"absent; slots={sorted(group)}"
        )
    if SPARSE_POSITIVE_SLOT not in group:
        raise ValueError(
            f"pair-group lacks the {SPARSE_POSITIVE_SLOT!r} positive arm; "
            f"slots={sorted(group)}"
        )
    positive_value = forward(SPARSE_POSITIVE_SLOT, grad=False)
    reference_value = forward(ref_slot, grad=False)
    if positive_value is None or reference_value is None:
        return None
    lc = float(positive_value)
    lr = float(reference_value)

    gain_margin = float(training.get("sparse_gain_margin", 0.01))
    rank_margin = float(training.get("sparse_rank_margin", 0.02))
    gain_weight = float(training.get("sparse_gain_weight", 1.0))
    rank_weight = float(training.get("sparse_rank_weight", 1.0))
    drift_weight = float(training.get("sparse_drift_weight", 2.0))
    l2_weight = float(training.get("history_lora_l2_weight", 1e-4))

    negatives = {
        slot: index
        for slot, index in group.items()
        if samples[index]["role"] == "negative"
    }
    # note (luojiaxuan): 归一化的分母必须与**真正进入损失**的负样本集合一致。旧写法
    # 先用全部负样本算 normalized、再在循环里 continue 掉取不到值的项,剩下份额之和
    # 就 < 1(三项里掉一项只剩 0.6),等于按"哪些前向恰好失败"给该组梯度打折扣。
    # 现在先跑完两次 no-grad 前向、收齐有效负样本,再用它们的 scale 归一化。
    scored: list[tuple[str, str, float, float, float]] = []
    for slot in sorted(negatives):
        sample = samples[negatives[slot]]
        active_value = forward(slot, grad=False)
        # bypass 前向是同一条负样本在冻结 policy 上的分数,drift 的锚点
        frozen_value = forward(slot, grad=False, adapter_mode="bypass")
        if active_value is None or frozen_value is None:
            continue
        scored.append(
            (
                slot,
                sample["negative_kind"],
                float(sample["negative_scale"]),
                float(active_value),
                float(frozen_value),
            )
        )
    scale_mass = sum(scale for _slot, _kind, scale, _ln, _lnf in scored)
    normalized = (
        {slot: scale / scale_mass for slot, _kind, scale, _ln, _lnf in scored}
        if scale_mass > 0.0
        else {}
    )

    weight_c = 0.0
    weights: dict[str, float] = {}
    total = 0.0
    diagnostics: dict[str, float] = {
        "SA_minus_R0": lc - lr,
        # 只数真正参与损失的负样本,和归一化分母同源
        "negatives": float(len(scored)),
    }

    if (gain_margin - (lc - lr)) > 0.0:
        weight_c += -gain_weight
        total += gain_weight * (gain_margin - (lc - lr))

    for slot, kind, _scale, ln, ln_frozen in scored:
        share = normalized[slot]
        gap_key, drift_key = sparse_diagnostic_keys(kind)
        diagnostics[gap_key] = lc - ln
        diagnostics[drift_key] = abs(ln - ln_frozen)
        if (rank_margin - (lc - ln)) > 0.0:
            weight_c += -rank_weight * share
            weights[slot] = weights.get(slot, 0.0) + rank_weight * share
            total += rank_weight * share * (rank_margin - (lc - ln))
        drift = ln - ln_frozen
        huber_grad = max(-1.0, min(1.0, drift))
        huber_value = 0.5 * drift * drift if abs(drift) < 1.0 else abs(drift) - 0.5
        weights[slot] = weights.get(slot, 0.0) + drift_weight * share * huber_grad
        total += drift_weight * share * huber_value

    for slot, weight in [(SPARSE_POSITIVE_SLOT, weight_c), *weights.items()]:
        if weight == 0.0:
            continue
        forward(slot, grad=True, backward_weight=weight / accumulation)

    if l2_weight > 0.0 and adapter_parameters:
        l2_term = l2_weight * sum(
            parameter.pow(2).sum() for parameter in adapter_parameters
        )
        total += float(l2_term.detach())
        (l2_term / accumulation).backward()

    diagnostics["loss"] = total
    # note (luojiaxuan): 审计第 10 条。诊断的**唯一对外通道**是 diagnostics_out
    # (原地填充),返回值恒为 float | None——旧版把 dict 当返回值,契约与测试都按
    # "标量损失或 None"写,调用方多写一层 ["loss"] 才能拿到数,任何一处忘了就是
    # 静默类型错。``_SPARSE_DIAG`` 只是本进程的 25 组滚动打印窗口,不是返回通道。
    # 键名:除 loss / negatives 两个运行指标外,其余键都来自 sparse_diagnostic_keys
    # 与 SPARSE_DERIVED_QUANTITIES,与 config.gates 的量名逐字相同(见 F3-d)。
    # 唯一要注意的是聚合次序:这里的 <kind>_drift_abs 是**本组** |ℓn-ℓn⁰|,滚动窗口
    # 打印的是 mean|drift|;gate 判定按 config.gates.drift_definition 在留出集上算
    # |mean drift|。两者同名、同符号约定(非负、越小越好),前者是后者的上界。
    # 损失为 0 的组也照常记录,否则"全部达标"的样本会从趋势里消失。
    if diagnostics_out is not None:
        diagnostics_out.update(diagnostics)
    _SPARSE_DIAG.append(diagnostics)
    if len(_SPARSE_DIAG) % 25 == 0:
        import statistics as _st
        window = _SPARSE_DIAG[-25:]
        keys = sorted({key for entry in window for key in entry})
        average = {
            key: round(
                _st.mean([entry[key] for entry in window if key in entry]), 5
            )
            for key in keys
        }
        print(
            json.dumps(
                {
                    "sparse_history_diag_last25": average,
                    "reference_arm_id": ref_slot,
                    "groups": len(_SPARSE_DIAG),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    # note (luojiaxuan): total == 0.0 意味着 gain/rank 的 hinge 全未触发、drift 全为
    # 0、L2 也没有质量(权重为 0 或没有 adapter 参数),此时上面一个带梯度前向都没跑,
    # 返回 None 让调用方把这组算作"跳过",而不是把 0 计进 running_loss 的分母。
    return None if total == 0.0 else total


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
    # note (luojiaxuan): v2.1 老样本用 memory_config.restored_event_step_ids 记录
    # 恢复了哪几步;sparse-history 样本用 selected_steps。两者语义相同(K 张历史图),
    # 掩码只需要数量;缺两者则视为无历史(K=0),由下面的 return None 走 bypass。
    if "memory_config" in sample:
        history_count = len(sample["memory_config"]["restored_event_step_ids"])
    elif sample.get("schema_version") == SPARSE_SAMPLE_SCHEMA:
        history_count = len(sample["selected_steps"])
        if history_count != int(sample["budget"]):
            raise ValueError(
                f"sample {sample['sample_id']!r} budget {sample['budget']} differs "
                f"from len(selected_steps)={history_count}"
            )
    else:
        # note (luojiaxuan): 审计第 8 条(fail-closed 缺口)。这里原本兜底成
        # ``len(sample.get("selected_steps") or ())``:既没有 memory_config、又不是
        # sparse v2 schema 的样本会被当成 K=0,于是下面 return None → 整条样本在
        # adapter bypass 下前向。后果不是报错而是"以为在训练 adapter,实际全程在给
        # 冻结模型打分",且日志里看不出任何异常。K 只能有显式来源,缺则拒绝。
        raise ValueError(
            f"sample {sample.get('sample_id')!r} carries neither the sparse-history "
            f"schema {SPARSE_SAMPLE_SCHEMA} (selected_steps + budget) nor "
            "memory_config.restored_event_step_ids; the restored-history count K has "
            "no authoritative source and the adapter would silently run in bypass"
        )
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


def adapter_context_for_sample(
    encoded: dict[str, Any],
    sample: dict[str, Any],
    *,
    merge_size: int,
    adapter_mode: str | None = None,
    slot: str | None = None,
) -> Any:
    """Dispatch one sample's declared ``adapter_mode`` to a context, or None.

    ``bypass`` -> ``None`` (the frozen policy sees no adapter at all); ``active``
    -> the geometric context, asserted non-empty; anything else raises.

    # note (luojiaxuan): 审计第 10 条的根因。这段"读 adapter_mode → 决定开不开
    # adapter"的分派原来内嵌在 sparse_history_group_unit_loss 的 forward 闭包里,
    # 而它恰恰是审计 P0-4 点名的语义(adapter 开关只准来自显式字段,永不猜名字
    # 前缀)——闭包既 import 不出来也 monkeypatch 不到,复核者只能读代码,等于这条
    # 最关键的契约没有任何自动化保护。提到模块级后,bypass/active/未知值三条路径
    # 都能在没有 GPU、没有 runtime 的情况下直接被断言。
    #
    # 职责分界(别再合并这两个函数):
    #   * history_sample_context 是**几何**函数——从 encoded 与 K 算 token mask,
    #     它不读也不该读 adapter_mode;
    #   * 本函数是**策略**函数——只做 adapter_mode 的分派与 fail-closed 断言,
    #     几何一律委托给上面那个。
    # ``adapter_mode`` 形参只用于显式覆盖(P1-4 的 per-negative 冻结锚点要拿同一条
    # 负样本在 bypass 下的分数);缺省 None 表示"以样本字段为准"。样本没有
    # adapter_mode 字段时故意直接 KeyError,不给任何默认值。``slot`` 只进报错文本。
    #
    # 训练侧的 forward 闭包与留出集打分脚本共用本函数:打分侧若自己再写一份 mode
    # 解析,五臂契约就会在"训练"与"验收"两处分叉,而这种分叉在日志里看不出来。
    """
    label = slot if slot is not None else sample.get("arm_slot")
    mode = sample["adapter_mode"] if adapter_mode is None else adapter_mode
    if mode == "bypass":
        return None
    if mode != "active":
        raise ValueError(f"unknown adapter_mode {mode!r} on arm {label!r}")
    context = history_sample_context(encoded, sample, merge_size=merge_size)
    if context is None or int(context.history_token_mask.sum()) == 0:
        raise ValueError(
            f"arm {label!r} declares adapter_mode=active but produced an "
            "empty history token mask"
        )
    return context


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

    def forward(
        variant: str, *, grad: bool, backward_weight: float | None = None
    ) -> Any:
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
        # note (luojiaxuan): 带梯度路径必须在 scope 内完成 backward——梯度检查点
        # 的重算发生在 backward 期间,scope 提前退出会让 hook 读到 ctx=None,
        # 造成"保存张量数不一致"崩溃。
        with history_adapter_scope(context):
            if grad:
                lp = mean_target_logprob(runtime.model, encoded, torch=torch)
                if backward_weight is not None:
                    (backward_weight * lp).backward()
                    return float(lp.detach())
                return lp
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
        forward(variant, grad=True, backward_weight=weight / accumulation)
    if l2_weight > 0.0:
        l2_term = l2_weight * sum(
            parameter.pow(2).sum() for parameter in adapter_parameters
        )
        total_value += float(l2_term.detach())
        (l2_term / accumulation).backward()
    return total_value


# note (luojiaxuan): adapter_mode 分派**只有一份实现**,就是上面的
# adapter_context_for_sample(审计 P0-4 的单一入口)。训练侧的 forward 闭包与留出集
# 打分脚本 scripts/score_sparse_history_arms.py 都直接调用它——两份各自解析 mode 的
# 拷贝一旦出现,五臂契约就会在"训练"与"验收"两处悄悄分叉,而这种分叉在日志里
# 完全看不出来。


def sparse_history_group_unit_loss(
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
    diagnostics_out: dict[str, float] | None = None,
) -> float | None:
    """One sparse-history pair-group forward set and its loss, or None to skip.

    # note (luojiaxuan): 与旧 history_group_unit_loss 完全分开,因为两者的臂命名、
    # adapter 开关来源和损失结构都不同;共用一个 forward 闭包正是审计 P0-4 里
    # "靠 variant 前缀猜 adapter" 的成因。这里 adapter 开关只读 adapter_mode——
    # 且分派本身已提到模块级的 adapter_context_for_sample,闭包里不再有任何
    # "只能靠读代码复核"的语义(审计第 10 条)。
    # 返回值与 _sparse_history_group_loss 一致:float(本组损失)或 None(跳过);
    # 诊断走 diagnostics_out 原地填充,调用方要拿就传一个空 dict 进来。
    """
    from causalcache.policy.history_adapter_context import history_adapter_scope

    def forward(
        slot: str,
        *,
        grad: bool,
        backward_weight: float | None = None,
        adapter_mode: str | None = None,
    ) -> Any:
        index = group.get(slot)
        if index is None:
            return None
        sample = samples[index]
        # note (luojiaxuan): adapter_mode 覆盖只为 P1-4 的 per-negative drift 锚点
        # 服务——拿同一条负样本在冻结 policy 上的分数,故只允许 no-grad + bypass。
        if adapter_mode is not None and (grad or adapter_mode != "bypass"):
            raise ValueError(
                "adapter_mode override is reserved for the no-grad bypass anchor"
            )
        encoded = encode_sample(
            runtime, sample, dataset_root=dataset_root, torch=torch
        )
        if encoded is None:
            return None
        context = adapter_context_for_sample(
            encoded,
            sample,
            merge_size=merge_size,
            adapter_mode=adapter_mode,
            slot=slot,
        )
        # note (luojiaxuan): 带梯度路径必须在 scope 内完成 backward——梯度检查点
        # 的重算发生在 backward 期间,scope 提前退出会让 hook 读到 ctx=None。
        with history_adapter_scope(context):
            if grad:
                lp = mean_target_logprob(runtime.model, encoded, torch=torch)
                if backward_weight is not None:
                    (backward_weight * lp).backward()
                    return float(lp.detach())
                return lp
            with torch.no_grad():
                return float(
                    mean_target_logprob(runtime.model, encoded, torch=torch).detach()
                )

    return _sparse_history_group_loss(
        samples=samples,
        group=group,
        forward=forward,
        training=training,
        adapter_parameters=adapter_parameters,
        accumulation=accumulation,
        torch=torch,
        diagnostics_out=diagnostics_out,
    )


# ---------------------------------------------------------------------------
# sparse-history 超参与 gate 的权威解析(审计 P1-6)
# ---------------------------------------------------------------------------
# note (luojiaxuan): 旧行为是 config 声明 max_optimizer_steps / checkpoint_every_steps
# 却被 CLI 的 --max-steps / --checkpoint-every-steps 覆盖,gates 段则从没被读过——
# 外部拿到 config 也复现不出这次 run。现在 sparse 分支的训练超参**只从 config 读**,
# CLI 若同时给出覆盖值直接报错;config.training 出现任何未被消费的 key 也报错退出,
# 免得"写了但没生效"的参数继续伪装成实验设定。旧两条路径的 CLI 语义不变。
SPARSE_TRAINING_KEYS = (
    "sparse_history", "epochs", "max_optimizer_steps", "learning_rate",
    "lr_schedule", "weight_decay", "micro_batch_size",
    "gradient_accumulation_steps", "max_grad_norm", "seed",
    "gradient_checkpointing", "checkpoint_every_steps", "sparse_gain_margin",
    "sparse_rank_margin", "sparse_gain_weight", "sparse_rank_weight",
    "sparse_drift_weight", "history_lora_l2_weight", "history_ce_weight",
)
SPARSE_BUILD_TIME_ONLY_KEYS = ("heldout_episode_fraction", "heldout_hash_salt")
SPARSE_GATE_SCHEMA = "causalcache.sparse_history_gates.v2"
SPARSE_DERIVED_QUANTITIES = {
    "format_effect": "R0 - N0",
    "frozen_selection_effect": "S0 - R0",
    "adapter_on_recent": "RA - R0",
    "adapter_on_sparse": "SA - S0",
    "SA_minus_RA": "SA - RA",
    "SA_minus_R0": "SA - R0",
    "deployment_delta": "SA - N0",
}
SPARSE_MAIN_CLAIM_QUANTITY = "SA_minus_RA"
# note (luojiaxuan): 负样本相关的量名由 sparse_diagnostic_keys 生成,而训练损失写
# 诊断时调用的是**同一个函数**(审计第 10 条)。原来这里各写一遍 f-string,损失那边
# 写的是 SA_minus_<kind> / <kind>_drift,词表这边写的是 SA_minus_SA_neg_<kind> /
# <kind>_drift_abs,两套名字对不上,"诊断可直接与 config.gates 对齐"只是句愿望。
SPARSE_NEGATIVE_GATE_KEYS = frozenset(
    key for kind in SPARSE_NEGATIVE_KINDS for key in sparse_diagnostic_keys(kind)
)
SPARSE_GATE_VOCABULARY = frozenset(
    set(SPARSE_DERIVED_QUANTITIES) | SPARSE_NEGATIVE_GATE_KEYS
)
SPARSE_REQUIRED_GATES = frozenset(
    {"SA_minus_RA", "SA_minus_R0"} | SPARSE_NEGATIVE_GATE_KEYS
)

# ---------------------------------------------------------------------------
# gate 表达式:可执行的比较式与合成分表达式(审计第 9 条)
# ---------------------------------------------------------------------------
# note (luojiaxuan): 旧实现里 must_pass 的 "> 0" / "< 0.02" 是**从不被解析的字符串**,
# composite_score / selection_rule 只被断言"非空字符串",于是整套 gate 只能靠人肉执行
# ——config 里写什么都不会让任何一次 run 失败,这等于没有验收标准。现在三者都有真正的
# 解析器:trainer 启动时用它们把语法错误挡在训练之前(写错的 gate 必须当场炸,而不是
# 训练 150 步之后才发现没人能执行它),scripts/score_sparse_history_arms.py 在留出集上
# 用**同一份**解析器执行。两侧共用一份语法是关键——各写一套正是"config 声明的验收标准"
# 与"脚本实际执行的验收标准"悄悄分叉的成因。
GATE_COMPARISON_OPERATORS: dict[str, Any] = {
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<": lambda value, threshold: value < threshold,
    "<=": lambda value, threshold: value <= threshold,
}
# point = 点估计;ci_low / ci_high = episode-cluster bootstrap 的置信区间端点。
GATE_STATISTICS = ("point", "ci_low", "ci_high")
_GATE_COMPARISON_PATTERN = re.compile(
    r"^\s*(?P<operator>>=|<=|>|<)\s*"
    r"(?P<threshold>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"(?:@\s*(?P<statistic>[A-Za-z_][A-Za-z_0-9]*)\s*)?$"
)


class GateComparison:
    """One parsed must_pass expression: ``<op> <threshold> [@ <statistic>]``."""

    __slots__ = ("expression", "operator", "threshold", "statistic")

    def __init__(
        self, *, expression: str, operator: str, threshold: float, statistic: str
    ) -> None:
        self.expression = expression
        self.operator = operator
        self.threshold = threshold
        self.statistic = statistic

    def evaluate(self, value: float | None) -> bool:
        # note (luojiaxuan): 量算不出来(留出集上没有支撑组)一律判 FAIL,不是 skip。
        # "这条 gate 没数据所以不参与判定" 会让一个缺臂/缺负样本的留出集自动全过。
        if value is None:
            return False
        return bool(GATE_COMPARISON_OPERATORS[self.operator](float(value), self.threshold))

    def conservative_statistic(self) -> str:
        """Which CI end makes this comparison strictly harder to pass."""
        return "ci_low" if self.operator in (">", ">=") else "ci_high"

    def as_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "operator": self.operator,
            "threshold": self.threshold,
            "statistic": self.statistic,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GateComparison({self.expression!r})"


def parse_gate_comparison(
    expression: Any, *, where: str = "gates.must_pass"
) -> GateComparison:
    """Parse ``"> 0"`` / ``"< 0.02 @ci_high"`` into an executable comparison."""
    if not isinstance(expression, str):
        raise ValueError(
            f"{where} comparison must be a string, got {type(expression).__name__}"
        )
    match = _GATE_COMPARISON_PATTERN.match(expression)
    if match is None:
        raise ValueError(
            f"{where} comparison {expression!r} is not parseable; expected "
            "'<op> <number>' with op in >, >=, <, <= plus an optional "
            "'@point' / '@ci_low' / '@ci_high' suffix"
        )
    statistic = match.group("statistic") or "point"
    if statistic not in GATE_STATISTICS:
        raise ValueError(
            f"{where} comparison {expression!r} names unknown statistic "
            f"{statistic!r}; expected one of {GATE_STATISTICS}"
        )
    return GateComparison(
        expression=expression,
        operator=match.group("operator"),
        threshold=float(match.group("threshold")),
        statistic=statistic,
    )


def parse_gate_comparisons(
    must_pass: dict[str, Any], *, where: str = "gates.must_pass"
) -> dict[str, GateComparison]:
    """Parse every must_pass entry; one unparseable expression fails the block."""
    return {
        quantity: parse_gate_comparison(expression, where=f"{where}[{quantity!r}]")
        for quantity, expression in sorted(must_pass.items())
    }


class GateQuantityUnavailable(ValueError):
    """A gate expression needs a quantity that has no supporting heldout groups."""


GATE_EXPRESSION_FUNCTIONS = ("abs", "mean", "min", "max")
_GATE_TOKEN_PATTERN = re.compile(
    r"\s*(?:(?P<number>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+)"
    r"|(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<symbol>[()+\-*/,]))"
)


def _tokenize_gate_expression(expression: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(expression):
        match = _GATE_TOKEN_PATTERN.match(expression, position)
        if match is None:
            if not expression[position:].strip():
                break
            raise ValueError(
                f"gate expression {expression!r} has an unparseable character at "
                f"offset {position}: {expression[position:position + 12]!r}"
            )
        position = match.end()
        for kind in ("number", "name", "symbol"):
            text = match.group(kind)
            if text is not None:
                tokens.append((kind, text))
                break
    return tokens


class _GateExpressionParser:
    """Recursive-descent evaluator for the gate arithmetic mini-language.

    # note (luojiaxuan): 手写而不是 eval()——config 是要被外部 reviewer 编辑的文件,
    # 用 eval 等于让 config 拿到任意代码执行,而且 eval 会悄悄接受 `__import__`、属性
    # 访问、比较运算这些本不该出现在打分表达式里的东西。这里只认:数字、词表里的量名、
    # + - * / 一元正负、括号,以及 abs/mean/min/max 四个函数。
    """

    def __init__(self, expression: str, values: dict[str, float | None]) -> None:
        self.expression = expression
        self.values = values
        self.tokens = _tokenize_gate_expression(expression)
        self.position = 0

    def evaluate(self) -> float:
        if not self.tokens:
            raise ValueError(f"gate expression {self.expression!r} is empty")
        value = self._sum()
        if self.position != len(self.tokens):
            raise ValueError(
                f"gate expression {self.expression!r} has trailing tokens from "
                f"{self.tokens[self.position][1]!r}"
            )
        return value

    def _peek(self) -> tuple[str, str] | None:
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def _accept(self, symbol: str) -> bool:
        if self._peek() == ("symbol", symbol):
            self.position += 1
            return True
        return False

    def _expect(self, symbol: str) -> None:
        if not self._accept(symbol):
            raise ValueError(
                f"gate expression {self.expression!r} expects {symbol!r} at token "
                f"{self.position}"
            )

    def _sum(self) -> float:
        value = self._product()
        while True:
            if self._accept("+"):
                value += self._product()
            elif self._accept("-"):
                value -= self._product()
            else:
                return value

    def _product(self) -> float:
        value = self._unary()
        while True:
            if self._accept("*"):
                value *= self._unary()
            elif self._accept("/"):
                divisor = self._unary()
                if divisor == 0.0:
                    raise ValueError(
                        f"gate expression {self.expression!r} divides by zero"
                    )
                value /= divisor
            else:
                return value

    def _unary(self) -> float:
        if self._accept("-"):
            return -self._unary()
        if self._accept("+"):
            return self._unary()
        return self._primary()

    def _primary(self) -> float:
        token = self._peek()
        if token is None:
            raise ValueError(
                f"gate expression {self.expression!r} ends mid-expression"
            )
        kind, text = token
        self.position += 1
        if kind == "number":
            return float(text)
        if kind == "symbol":
            if text != "(":
                raise ValueError(
                    f"gate expression {self.expression!r} has an unexpected {text!r}"
                )
            value = self._sum()
            self._expect(")")
            return value
        if self._accept("("):
            return self._call(text)
        if text not in self.values:
            raise ValueError(
                f"gate expression {self.expression!r} references unknown quantity "
                f"{text!r}; the vocabulary is {sorted(self.values)}"
            )
        value = self.values[text]
        if value is None:
            raise GateQuantityUnavailable(
                f"gate expression {self.expression!r} needs quantity {text!r}, "
                "which has no supporting heldout groups"
            )
        return float(value)

    def _call(self, name: str) -> float:
        if name not in GATE_EXPRESSION_FUNCTIONS:
            raise ValueError(
                f"gate expression {self.expression!r} calls unknown function "
                f"{name!r}; allowed: {GATE_EXPRESSION_FUNCTIONS}"
            )
        arguments: list[float] = []
        if not self._accept(")"):
            arguments.append(self._sum())
            while self._accept(","):
                arguments.append(self._sum())
            self._expect(")")
        if not arguments:
            raise ValueError(
                f"gate expression {self.expression!r} calls {name}() with no arguments"
            )
        if name == "abs":
            if len(arguments) != 1:
                raise ValueError(
                    f"gate expression {self.expression!r} calls abs() with "
                    f"{len(arguments)} arguments"
                )
            return abs(arguments[0])
        if name == "mean":
            return sum(arguments) / len(arguments)
        if name == "min":
            return min(arguments)
        return max(arguments)


def evaluate_gate_expression(
    expression: str, values: dict[str, float | None]
) -> float:
    """Evaluate one gate arithmetic expression over a quantity -> value mapping."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("gate expression must be a non-empty string")
    return _GateExpressionParser(expression, values).evaluate()


def validate_gate_expression(
    expression: Any,
    *,
    allowed: Any = SPARSE_GATE_VOCABULARY,
    where: str = "gates.composite_score",
) -> str:
    """Parse-check an expression against a vocabulary without needing real numbers."""
    # note (luojiaxuan): 哨兵值取 1.0 而不是 0.0——用 0.0 会让含除法的合法表达式在
    # 纯语法检查阶段误报 divide-by-zero。
    try:
        evaluate_gate_expression(expression, {name: 1.0 for name in allowed})
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error
    return expression


SPARSE_SELECTION_FILTERS = ("all_must_pass",)
SPARSE_SELECTION_OBJECTIVES = ("maximize", "minimize")
SPARSE_SELECTION_TIE_BREAKS = ("earliest_checkpoint", "latest_checkpoint")
SPARSE_SELECTION_QUANTITY = "composite_score"
_SPARSE_SELECTION_KEYS = ("filter", "objective", "quantity", "tie_break")


def parse_selection_rule(
    rule: Any, *, where: str = "gates.selection_rule"
) -> dict[str, str]:
    """Parse the structured checkpoint-selection rule into a validated dict."""
    if not isinstance(rule, dict):
        raise ValueError(
            f"{where} must be an object; the old free-text rule was never executed "
            "by anything, so in practice a checkpoint could be picked by any rule"
        )
    missing = [key for key in _SPARSE_SELECTION_KEYS if key not in rule]
    if missing:
        raise ValueError(f"{where} misses keys {missing}")
    unknown = sorted(set(rule) - set(_SPARSE_SELECTION_KEYS) - {"note"})
    if unknown:
        raise ValueError(f"{where} carries unknown keys {unknown}")
    if rule["filter"] not in SPARSE_SELECTION_FILTERS:
        raise ValueError(
            f"{where}.filter must be one of {SPARSE_SELECTION_FILTERS}"
        )
    if rule["objective"] not in SPARSE_SELECTION_OBJECTIVES:
        raise ValueError(
            f"{where}.objective must be one of {SPARSE_SELECTION_OBJECTIVES}"
        )
    if rule["tie_break"] not in SPARSE_SELECTION_TIE_BREAKS:
        raise ValueError(
            f"{where}.tie_break must be one of {SPARSE_SELECTION_TIE_BREAKS}"
        )
    if rule["quantity"] != SPARSE_SELECTION_QUANTITY:
        # note (luojiaxuan): 把 config 里那条中文警告变成可执行约束。按单个 derived
        # quantity(尤其 SA_minus_R0)选点会挑出"见图就整体放大"的 checkpoint,
        # composite_score 存在的全部意义就是不让人这么选,所以这里只接受它。
        raise ValueError(
            f"{where}.quantity must be {SPARSE_SELECTION_QUANTITY!r}; selecting on a "
            "single derived quantity picks the checkpoint that merely amplifies any "
            "history, which is exactly what the composite exists to prevent"
        )
    return {key: str(rule[key]) for key in _SPARSE_SELECTION_KEYS}


def compile_sparse_gates(gates: dict[str, Any]) -> dict[str, Any]:
    """Return the executable form of an already-validated gate block."""
    return {
        "comparisons": parse_gate_comparisons(gates["must_pass"]),
        "composite_score": gates["composite_score"],
        "selection_rule": parse_selection_rule(gates["selection_rule"]),
    }


def resolve_sparse_training(
    training: dict[str, Any], *, args: argparse.Namespace
) -> dict[str, Any]:
    """Return the run controls the sparse branch takes from config alone."""
    stale = [key for key in SPARSE_BUILD_TIME_ONLY_KEYS if key in training]
    if stale:
        raise ValueError(
            f"training block carries build-time-only keys {stale}; move them under "
            "data.split — the trainer reads each sample's split field"
        )
    unknown = sorted(set(training) - set(SPARSE_TRAINING_KEYS))
    if unknown:
        raise ValueError(f"training block has unconsumed keys {unknown}")
    missing = sorted(set(SPARSE_TRAINING_KEYS) - set(training))
    if missing:
        raise ValueError(f"training block misses required keys {missing}")
    if str(training["lr_schedule"]) != "constant":
        raise ValueError("sparse_history only implements a constant learning rate")
    if int(training["micro_batch_size"]) != 1:
        raise ValueError("sparse_history forwards one sample at a time")
    if float(training["history_ce_weight"]) != 0.0:
        raise ValueError("sparse_history keeps the CE weight at exactly 0")
    if bool(training["gradient_checkpointing"]):
        raise ValueError(
            "sparse_history disables gradient checkpointing: the recompute runs on "
            "the autograd thread where the adapter ContextVar reads empty"
        )
    if args.max_steps or args.checkpoint_every_steps:
        raise ValueError(
            "--max-steps/--checkpoint-every-steps are rejected under sparse_history; "
            "max_optimizer_steps and checkpoint_every_steps come from the config"
        )
    return {
        "max_steps": int(training["max_optimizer_steps"]),
        "checkpoint_every_steps": int(training["checkpoint_every_steps"]),
    }


def validate_sparse_gates(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the checkpoint-selection gate block and return it verbatim."""
    gates = config.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("sparse_history config must carry a gates block")
    if gates.get("gate_schema") != SPARSE_GATE_SCHEMA:
        raise ValueError(f"gates.gate_schema must be {SPARSE_GATE_SCHEMA}")
    if gates.get("reference_arm_id") != "R0":
        raise ValueError("gates.reference_arm_id must match the training reference R0")
    if gates.get("main_claim_quantity") != SPARSE_MAIN_CLAIM_QUANTITY:
        raise ValueError(
            f"gates.main_claim_quantity must be {SPARSE_MAIN_CLAIM_QUANTITY} "
            "(SA - RA isolates the adapter effect on sparse selection)"
        )
    declared = gates.get("derived_quantities")
    if declared != SPARSE_DERIVED_QUANTITIES:
        raise ValueError(
            "gates.derived_quantities drifted from the frozen arm algebra "
            f"{SPARSE_DERIVED_QUANTITIES}"
        )
    must_pass = gates.get("must_pass")
    if not isinstance(must_pass, dict) or not must_pass:
        raise ValueError("gates.must_pass must be a non-empty object")
    unknown = sorted(set(must_pass) - SPARSE_GATE_VOCABULARY)
    if unknown:
        raise ValueError(f"gates.must_pass references unknown quantities {unknown}")
    absent = sorted(SPARSE_REQUIRED_GATES - set(must_pass))
    if absent:
        raise ValueError(f"gates.must_pass misses required gates {absent}")
    if not isinstance(gates.get("drift_definition"), str) or not gates[
        "drift_definition"
    ].strip():
        raise ValueError("gates.drift_definition must be a non-empty string")
    # note (luojiaxuan): 审计第 9 条。这三行是"gate 可执行"的入口:每条 must_pass
    # 比较式必须解析成 (op, threshold, statistic);composite_score 必须在 gate 词表上
    # 解析通过;selection_rule 必须是结构化且被支持的规则。语法错误在训练开始前就炸,
    # 而不是训练 150 步后才发现留出集打分脚本根本执行不了这份 config。
    parse_gate_comparisons(must_pass)
    validate_gate_expression(
        gates.get("composite_score"), where="gates.composite_score"
    )
    parse_selection_rule(gates.get("selection_rule"))
    for quantity, algebra in SPARSE_DERIVED_QUANTITIES.items():
        # 派生量的代数式也必须可执行:打分脚本按这些字符串在五臂分数上求值。
        validate_gate_expression(
            algebra,
            allowed=frozenset(SPARSE_HELDOUT_REQUIRED_SLOTS),
            where=f"gates.derived_quantities[{quantity!r}]",
        )
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="legacy paths only; rejected under training.sparse_history",
    )
    parser.add_argument("--resume-lora", type=Path, default=None)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=0,
        help="legacy paths only; rejected under training.sparse_history",
    )
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    config = load_config(args.config)
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    sparse_mode = bool(config["training"].get("sparse_history", False))
    sparse_gates: dict[str, Any] | None = None
    if sparse_mode:
        if adapter_settings(config)[0] != "history_gated_kv":
            raise ValueError("sparse_history requires adapter_type history_gated_kv")
        sparse_gates = validate_sparse_gates(config)
        sparse_controls = resolve_sparse_training(config["training"], args=args)
        max_steps = sparse_controls["max_steps"]
        checkpoint_every_steps = sparse_controls["checkpoint_every_steps"]
    else:
        max_steps = args.max_steps
        checkpoint_every_steps = args.checkpoint_every_steps
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
    # note (luojiaxuan): history_gated_kv 禁用梯度检查点——重算跑在 autograd
    # 线程上,ContextVar(线程局部)读不到 mask 上下文,hook 旁路导致保存张量数
    # 不一致;两遍法已保证同时只活一张图,直接全激活反而更快。
    adapter_type_early, _ = adapter_settings(config)
    if config["training"]["gradient_checkpointing"] and adapter_type_early != "history_gated_kv":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    elif adapter_type_early == "history_gated_kv":
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
    resume_sha: str | None = None
    if args.resume_lora is not None:
        # note (luojiaxuan): resume checkpoint 的 sha256 只在 sparse 路径算并打印。
        # 旧 full_policy_lora / history_gated_kv 的 stdout 是冻结契约的一部分,
        # 多打一个键会让既有的解析脚本与逐字节比对失效 —— 复审 N1 点名的就是这个漂移。
        if sparse_mode:
            resume_sha = hashlib.sha256(args.resume_lora.read_bytes()).hexdigest()
        adapter_load_state_dict(
            wrapped, torch.load(args.resume_lora, map_location="cpu")
        )
        if rank == 0:
            resumed: dict[str, Any] = {"resumed_from": str(args.resume_lora)}
            if sparse_mode:
                resumed["resume_checkpoint_sha256"] = resume_sha
            print(json.dumps(resumed), flush=True)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    if rank == 0:
        args.output_root.mkdir(parents=True, exist_ok=True)
        run_manifest: dict[str, Any] = {
            "schema_version": "causalcache.success_sft_lora_run.v1",
            "config": config,
            "dataset_manifest_sha256": manifest_sha,
            "sample_count": len(samples),
            "world_size": world_size,
            "lora_module_count": len(wrapped),
        }
        if sparse_mode:
            # note (luojiaxuan): 审计 P1-6——外部要能只凭 manifest 复现这次 run,
            # 所以完整记录 argv、config 的 sha256、生效的超参(它们只来自 config)、
            # resume checkpoint 的 sha256,以及被校验过的 checkpoint 选择 gate。
            run_manifest["schema_version"] = "causalcache.success_sft_lora_run.v2"
            run_manifest["sample_schema_version"] = SPARSE_SAMPLE_SCHEMA
            run_manifest["cli_argv"] = list(sys.argv)
            run_manifest["cli_args"] = {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in sorted(vars(args).items())
            }
            run_manifest["config_path"] = str(args.config)
            run_manifest["config_sha256"] = config_sha
            run_manifest["hyperparameter_authority"] = "config.training only"
            run_manifest["effective_training"] = {
                key: config["training"][key] for key in sorted(SPARSE_TRAINING_KEYS)
            }
            run_manifest["effective_max_optimizer_steps"] = max_steps
            run_manifest["effective_checkpoint_every_steps"] = checkpoint_every_steps
            run_manifest["resume_lora"] = (
                None if args.resume_lora is None else str(args.resume_lora)
            )
            run_manifest["resume_checkpoint_sha256"] = resume_sha
            run_manifest["checkpoint_selection_gates"] = sparse_gates
        (args.output_root / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    accumulation = config["training"]["gradient_accumulation_steps"]
    training_config = config["training"]
    margin_lambda = float(training_config.get("margin_lambda", 0.0))
    margin_value = float(training_config.get("margin_per_token", 0.0))
    b0_weight = float(training_config.get("b0_ce_weight", 1.0))
    if sparse_mode:
        units, heldout_episodes = build_sparse_history_units(samples)
    elif adapter_type == "history_gated_kv":
        units, heldout_episodes = build_history_gated_units(
            samples, training=training_config
        )
    else:
        units, heldout_episodes = build_training_units(
            samples, training=training_config
        )
    if rank == 0:
        startup = {
            "training_units": len(units),
            "heldout_episodes": len(heldout_episodes),
        }
        if sparse_mode:
            startup["sample_schema_version"] = SPARSE_SAMPLE_SCHEMA
            startup["reference_arm_id"] = "R0"
            startup["main_claim_quantity"] = SPARSE_MAIN_CLAIM_QUANTITY
            startup["config_sha256"] = config_sha
            startup["max_optimizer_steps"] = max_steps
            startup["checkpoint_every_steps"] = checkpoint_every_steps
        # note (luojiaxuan): 键序保持插入序,旧两条路径的这行 stdout 逐字节不变。
        print(json.dumps(startup), flush=True)
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
            if kind == "sparse_group":
                # note (luojiaxuan): 组损失返回 float | None(与另外两条路径同型);
                # 诊断从 diagnostics_out 原地取,组被跳过时它也已填好,便于排查。
                diagnostics: dict[str, float] = {}
                unit_value = sparse_history_group_unit_loss(
                    runtime,
                    samples,
                    payload,
                    dataset_root=args.dataset_root,
                    training=training_config,
                    adapter_parameters=parameters,
                    merge_size=merge_size,
                    accumulation=accumulation,
                    torch=torch,
                    diagnostics_out=diagnostics,
                )
                if unit_value is None:
                    continue
                running_loss += unit_value
                contributing += 1
                unit_loss = None
            elif kind == "history_group":
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
                    and checkpoint_every_steps
                    and global_step % checkpoint_every_steps == 0
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
                if max_steps and global_step >= max_steps:
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
        if max_steps and global_step >= max_steps:
            break
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({"training_complete": True, "global_steps": global_step}))


if __name__ == "__main__":
    main()
