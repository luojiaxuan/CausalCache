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
from collections.abc import Sequence
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


_CHAT_TEMPLATE_PROMPT_FORMATS = (
    "official_multiturn",
    "sparse_single_turn",
    # note (luojiaxuan): official-style sparse multiturn(gui_owl_sparse_multiturn.py)。
    # 它与前两者一样用官方 system prompt + processor.apply_chat_template 编码,唯一
    # 区别是 user/assistant 轮的排布,所以走同一条 chat_template 路径。这里只是把名字
    # 加进白名单;编码参数与两项保真校验一律不变。目前只有 probe_prompt_format.py 在
    # **冻结模型**上用它做格式对照,尚未进入任何冻结的五臂契约。
    "official_style_sparse_multiturn",
)
_EXACT_BATCH_PROMPT_FORMATS = ("v2_1_private",)
# note (luojiaxuan): 桌面语料(交接 §8.2)。它与上面 chat_template 家族的唯一差别
# 是编码时必须带 tools=[_TOOL_SPEC] —— 官方桌面 prompt 的 <tools> 段由 chat template
# 从 tools 实参注入,漏传等于换 prompt。与 GUIOwlOSWorldRuntime.generate_raw 的
# token-by-token parity 由 scripts/audit_osworld_official_trainer_parity.py 在真模型
# 上执行,单测只锁"编码实参与在线路径逐项相同"。
_OSWORLD_CHAT_TEMPLATE_PROMPT_FORMATS = ("osworld_official",)


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
    if prompt_format in _OSWORLD_CHAT_TEMPLATE_PROMPT_FORMATS:
        return "osworld_chat_template"
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
        encoding_path = _prompt_encoding_path(sample)
        if encoding_path == "osworld_chat_template":
            # note (luojiaxuan): 与 GUIOwlOSWorldRuntime.generate_raw 逐实参相同:
            # 单会话实参(不包 batch 列表)、tools=[_TOOL_SPEC]、add_generation_prompt。
            # 不跑 v2.1 的 assert_pinned_assistant_prefix —— 那是 v2.1 私有契约,
            # 桌面 prompt 的前缀不变量由 parity audit 脚本对着在线 runtime 逐 token 锁。
            from causalcache.osworld_gui_owl import _TOOL_SPEC
            from causalcache.policy.gui_owl_v2_1_runtime import (
                prompt_aligned_input_keys,
            )

            encoded_batch = runtime.processor.apply_chat_template(
                messages,
                tools=[_TOOL_SPEC],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = dict(
                encoded_batch.to(runtime.device)
                if hasattr(encoded_batch, "to")
                else {k: v.to(runtime.device) for k, v in encoded_batch.items()}
            )
            prompt_aligned_input_keys(model_inputs)
        elif encoding_path == "chat_template":
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
# note (luojiaxuan): v6 替换语料(causalcache.replacement_corpus_sample.v6)的契约。
# 与 v2 **并存**,v2 的每一条行为逐字不变——旧 checkpoint 与 v5 语料必须继续可复现。
# 差异只有三处,全部是 v6 的构造决定,不是这里的口味:
#   1. renderer 冻结成 official_style_sparse_multiturn(docs/renderer_freeze_v1.md),
#      所以整表的 prompt_format 换掉;
#   2. S0/SA 的历史是**冻结 policy 实测的 k=1 替换集**而不是随机稀疏集,故
#      selection_mode 叫 replacement;负样本是同决策点、同被替换位置、age 匹配的
#      **内容无关**旧帧,故叫 wrong;
#   3. 负样本只有 active 一行:A_n 由**同一行**跑 active + bypass 两次前向得到
#      (见 _sparse_history_group_loss_did),bypass 孪生臂是多余的。
# N0 与 v5 逐字段相同(原生 build_official_messages + recent + bypass),format_effect
# = R0 - N0 与 deployment_delta = SA - N0 因此在 v6 上照样可算。
SPARSE_REPLACEMENT_SAMPLE_SCHEMA = "causalcache.replacement_corpus_sample.v6"
SPARSE_REPLACEMENT_PROMPT_FORMAT = "official_style_sparse_multiturn"
SPARSE_REPLACEMENT_NEGATIVE_SLOT = "SA_neg_age_matched"
SPARSE_REPLACEMENT_ARM_CONTRACT: dict[str, tuple[str, str, str, str, str]] = {
    "N0": ("N0", "deployment_baseline", "official_multiturn", "recent", "bypass"),
    "R0": ("R0", "reference", SPARSE_REPLACEMENT_PROMPT_FORMAT, "recent", "bypass"),
    "RA": ("RA", "measurement", SPARSE_REPLACEMENT_PROMPT_FORMAT, "recent", "active"),
    "S0": (
        "S0", "measurement", SPARSE_REPLACEMENT_PROMPT_FORMAT, "replacement", "bypass",
    ),
    "SA": (
        "SA", "positive", SPARSE_REPLACEMENT_PROMPT_FORMAT, "replacement", "active",
    ),
    SPARSE_REPLACEMENT_NEGATIVE_SLOT: (
        "SA", "negative", SPARSE_REPLACEMENT_PROMPT_FORMAT, "wrong", "active",
    ),
}
# note (luojiaxuan): Desktop DiD 语料(causalcache.desktop_did_sample.v1,交接 §6)。
# 构建器为 scripts/build_desktop_hgkv_corpus.py,契约表与其 DESKTOP_ARM_CONTRACT
# 逐字段一致。与 v2/v6 的三点结构差异,均为语料构造决定:
#   1. renderer 是官方桌面 osworld_official(processor.apply_chat_template +
#      tools=[_TOOL_SPEC]),R0 与部署 prompt 同一 renderer 同一 recent 选择器,
#      因此**没有 N0 臂** —— deployment_baseline_arm_id 显式指向 R0,
#      format_effect 结构性为 0 不作为量报告;
#   2. 负样本只有一条 WA(同轨迹 age-matched wrong,kind="wrong"),W0 由
#      _sparse_history_group_loss_did 对同一行临时 bypass 重算;
#   3. B0 parity 行在语料侧写进独立文件/独立 schema,不进本契约。
# split:语料按轨迹发 train/dev/test,进 trainer 前必须先过
# normalize_desktop_splits(dev→heldout、test 整组剥离),validate_sparse_sample
# 只认 train/heldout。
SPARSE_DESKTOP_SAMPLE_SCHEMA = "causalcache.desktop_did_sample.v1"
SPARSE_DESKTOP_PROMPT_FORMAT = "osworld_official"
SPARSE_DESKTOP_NEGATIVE_SLOT = "WA"
SPARSE_DESKTOP_ARM_CONTRACT: dict[str, tuple[str, str, str, str, str]] = {
    "R0": ("R0", "reference", SPARSE_DESKTOP_PROMPT_FORMAT, "recent", "bypass"),
    "RA": ("RA", "measurement", SPARSE_DESKTOP_PROMPT_FORMAT, "recent", "active"),
    "S0": (
        "S0", "measurement", SPARSE_DESKTOP_PROMPT_FORMAT, "recurrence", "bypass",
    ),
    "SA": (
        "SA", "positive", SPARSE_DESKTOP_PROMPT_FORMAT, "recurrence", "active",
    ),
    SPARSE_DESKTOP_NEGATIVE_SLOT: (
        "WA", "negative", SPARSE_DESKTOP_PROMPT_FORMAT, "wrong", "active",
    ),
}
SPARSE_SAMPLE_SCHEMAS = (
    SPARSE_SAMPLE_SCHEMA,
    SPARSE_REPLACEMENT_SAMPLE_SCHEMA,
    SPARSE_DESKTOP_SAMPLE_SCHEMA,
)
SPARSE_ARM_CONTRACTS: dict[str, dict[str, tuple[str, str, str, str, str]]] = {
    SPARSE_SAMPLE_SCHEMA: SPARSE_ARM_CONTRACT,
    SPARSE_REPLACEMENT_SAMPLE_SCHEMA: SPARSE_REPLACEMENT_ARM_CONTRACT,
    SPARSE_DESKTOP_SAMPLE_SCHEMA: SPARSE_DESKTOP_ARM_CONTRACT,
}
# 部署基线臂按 schema 索引:v2/v6 有独立的官方 renderer N0;桌面的部署 prompt 与
# R0 同 renderer 同选择器,基线就是 R0 本身(见上面的桌面契约注释)。
SPARSE_DEPLOYMENT_BASELINE_BY_SCHEMA: dict[str, str] = {
    SPARSE_SAMPLE_SCHEMA: "N0",
    SPARSE_REPLACEMENT_SAMPLE_SCHEMA: "N0",
    SPARSE_DESKTOP_SAMPLE_SCHEMA: "R0",
}
# note (luojiaxuan): 两个集合,**别合并**,它们回答的是不同的问题:
#   * ``SPARSE_NEGATIVE_KINDS`` —— v5 五臂语料里真实存在的三个 kind。它同时是
#     ``SPARSE_REQUIRED_GATES``(config 必须声明哪些 gate)的来源,所以往里加东西
#     等于给**每一份既有 v5 config** 强加一个新的必需 gate,而那个 kind 在 v5 语料里
#     根本不存在 —— 旧 run 会在 validate_sparse_gates 当场报错,不再可复现。
#   * ``SPARSE_ALL_NEGATIVE_KINDS`` —— 校验器/gate 词表/运行诊断**认识**的全部 kind。
#     放宽这三处是安全的(词表只决定"允许引用哪些名字"),而且必须放宽:否则 v6 的
#     ``age_matched_drift_abs`` 不进词表,score_sparse_history_arms 的留出集报告里
#     这一项会**静默消失**,而不是报错。
# 反过来:v6 语料里不存在 shuffled/duplicate/irrelevant,所以"必需 gate 集合"本来就
# 是**随语料 schema 变**的,而 gates schema v2 已冻结。v6 run 该要求哪些 gate 是一个
# 尚未做出的科学判断,留给写 v6 训练 config 时显式决定,不在这里顺手替人定。
SPARSE_NEGATIVE_KINDS = ("step_shuffled", "irrelevant", "duplicate")
SPARSE_REPLACEMENT_NEGATIVE_KINDS = ("age_matched",)
SPARSE_DESKTOP_NEGATIVE_KINDS = ("wrong",)
SPARSE_ALL_NEGATIVE_KINDS = (
    SPARSE_NEGATIVE_KINDS
    + SPARSE_REPLACEMENT_NEGATIVE_KINDS
    + SPARSE_DESKTOP_NEGATIVE_KINDS
)
# 哪个 schema 的语料里**真实存在**哪些负样本 kind。必需 gate 与"外来 gate"检查都由它
# 生成:v5 语料里没有 age_matched,v6 语料里没有另外三个,两边都不该被要求声明对方的量。
SPARSE_NEGATIVE_KINDS_BY_SCHEMA: dict[str, tuple[str, ...]] = {
    SPARSE_SAMPLE_SCHEMA: SPARSE_NEGATIVE_KINDS,
    SPARSE_REPLACEMENT_SAMPLE_SCHEMA: SPARSE_REPLACEMENT_NEGATIVE_KINDS,
    SPARSE_DESKTOP_SAMPLE_SCHEMA: SPARSE_DESKTOP_NEGATIVE_KINDS,
}
# note (luojiaxuan): 负样本 kind → arm_slot,按 schema 索引。v2/v6 的命名约定是
# ``SA_neg_<kind>``,桌面语料沿用交接 §6 的臂名 WA。差值量名的记法是
# ``SA_minus_<arm_slot>``(见 sparse_diagnostic_keys),所以 gate 词表生成与样本
# 校验都必须从这张表拿 slot,不得再从 kind 字符串拼前缀。
SPARSE_NEGATIVE_SLOTS_BY_SCHEMA: dict[str, dict[str, str]] = {
    SPARSE_SAMPLE_SCHEMA: {
        kind: f"SA_neg_{kind}" for kind in SPARSE_NEGATIVE_KINDS
    },
    SPARSE_REPLACEMENT_SAMPLE_SCHEMA: {
        "age_matched": SPARSE_REPLACEMENT_NEGATIVE_SLOT,
    },
    SPARSE_DESKTOP_SAMPLE_SCHEMA: {"wrong": SPARSE_DESKTOP_NEGATIVE_SLOT},
}
# note (luojiaxuan): v6 语料里 k=0(recent_sufficient)组只有 N0/R0/RA 三臂 —— 它们
# 教的是"当前 Recent 已够用",是 **selector 的 STOP** 训练材料,不是 adapter 的。
# adapter 在 recent-sufficient 状态上按部署语义**根本不运行**(k=0 即 bypass),所以
# "教 adapter 在这些状态别乱动"是个不存在的需求;而 |A_r| 的约束在每个 positive 组里
# 都有(R0/RA 俱全),剔除三臂组不会让它失去管束。
# 但**默认不剔除**:缺臂静默跳过正是 fail-closed 该拦的事(审计第 8 条)。要排除必须在
# config 里显式声明 ``data.train_on_label_classes``,排除结果逐类计数打印。
SPARSE_LABEL_CLASSES = ("utility_positive", "recent_sufficient")
SPARSE_LABEL_CLASS_CONFIG_KEY = "train_on_label_classes"
SPARSE_REQUIRED_KEYS = (
    "schema_version", "sample_id", "pair_group", "episode", "decision_step",
    "arm_slot", "arm_id", "role", "prompt_format", "selection_mode",
    "adapter_mode", "budget", "selected_steps", "selected_images",
    "current_image", "target_text", "messages", "split", "reference_arm_id",
    "deployment_baseline_arm_id",
)
# v6 在 v5 必填键之上再加三个显式字段:有没有内容对照臂、该臂保留了几张最近帧、
# 标签类别。三者都不准靠数臂数或猜名字推断 —— 单条样本根本数不出臂数。
SPARSE_REPLACEMENT_REQUIRED_KEYS = SPARSE_REQUIRED_KEYS + (
    "label_class",
    "has_content_control",
    "recent_frames_kept",
)
# 桌面语料在 v5 必填键之上加两个:memory_config 是 history mask 的 K 的唯一权威
# 来源(history_sample_context 首选读它),recent_frames_kept 记录该臂选中步落在
# Recent-B 窗口内的数量(k=1 时 R 臂恒 1、S/W 臂恒 0,分层报告直接读)。
SPARSE_DESKTOP_REQUIRED_KEYS = SPARSE_REQUIRED_KEYS + (
    "memory_config",
    "recent_frames_kept",
)
SPARSE_REQUIRED_KEYS_BY_SCHEMA: dict[str, tuple[str, ...]] = {
    SPARSE_SAMPLE_SCHEMA: SPARSE_REQUIRED_KEYS,
    SPARSE_REPLACEMENT_SAMPLE_SCHEMA: SPARSE_REPLACEMENT_REQUIRED_KEYS,
    SPARSE_DESKTOP_SAMPLE_SCHEMA: SPARSE_DESKTOP_REQUIRED_KEYS,
}
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
    """Fail-closed field validation for one sparse-history v2 / replacement v6 sample."""
    where = f"samples[{index}]"
    schema = sample.get("schema_version")
    if schema not in SPARSE_SAMPLE_SCHEMAS:
        raise ValueError(
            f"{where} schema_version {schema!r} is not one of "
            f"{list(SPARSE_SAMPLE_SCHEMAS)}"
        )
    missing = [
        key for key in SPARSE_REQUIRED_KEYS_BY_SCHEMA[schema] if key not in sample
    ]
    if missing:
        raise ValueError(f"{where} misses required fields {missing}")
    stale = [key for key in SPARSE_FORBIDDEN_KEYS if key in sample]
    if stale:
        raise ValueError(f"{where} still carries retired fields {stale}")
    slot = sample["arm_slot"]
    contract = SPARSE_ARM_CONTRACTS[schema].get(slot)
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
        hint = (
            "; desktop corpora ship train/dev/test — run normalize_desktop_splits "
            "before grouping"
            if schema == SPARSE_DESKTOP_SAMPLE_SCHEMA
            else ""
        )
        raise ValueError(f"{where} unknown split {sample['split']!r}{hint}")
    if sample["reference_arm_id"] != "R0":
        raise ValueError(f"{where} reference_arm_id must be 'R0'")
    expected_baseline = SPARSE_DEPLOYMENT_BASELINE_BY_SCHEMA[schema]
    if sample["deployment_baseline_arm_id"] != expected_baseline:
        raise ValueError(
            f"{where} deployment_baseline_arm_id must be {expected_baseline!r} "
            f"for schema {schema}"
        )
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
    if kind not in SPARSE_ALL_NEGATIVE_KINDS:
        raise ValueError(f"{where} unknown negative_kind {kind!r}")
    expected_slot = SPARSE_NEGATIVE_SLOTS_BY_SCHEMA[schema].get(kind)
    if expected_slot is None or slot != expected_slot:
        raise ValueError(f"{where} arm_slot {slot!r} disagrees with negative_kind")
    scale = sample.get("negative_scale")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
        raise ValueError(f"{where} negative_scale must be a positive number")
    if kind == "irrelevant":
        donor = sample.get("donor_episode")
        if not isinstance(donor, str) or not donor or donor == sample["episode"]:
            raise ValueError(
                f"{where} irrelevant negative needs a foreign donor_episode"
            )
        return
    if kind == "age_matched":
        _validate_age_matched_negative(sample, where=where)
        return
    if kind == "wrong":
        # 桌面 WA 的硬约束与 v6 age_matched 完全同构:同 episode、distractor/oracle
        # 两个来源步不同、都在 Recent-B 窗口外、selected_steps 里有 distractor 无
        # oracle —— 直接复用同一份校验,不抄第二份。
        _validate_age_matched_negative(sample, where=where)
        return


def _validate_age_matched_negative(sample: dict[str, Any], *, where: str) -> None:
    """The v6 content control: same episode, age-matched, content-irrelevant frame.

    # note (luojiaxuan): 与 ``irrelevant`` **正好相反**——那个负样本强制跨 episode 供体,
    # 而跨 episode 按构造就是跨 app 的,于是它同时改变了内容、app、分辨率与 age 分布,
    # 无法充当"排除模型只是偏爱旧图 / 跨 app 图"的对照,而那正是内容对照臂存在的理由。
    # 本 kind 反过来把除"恢复了哪一张旧帧"以外的一切都钉死:同一条 episode、同一个决策
    # 点、同一个被替换的 recent 位置、age 尽可能接近,只有内容相关性不同。
    #
    # 四条硬检查(任何一条不满足,这一行就不是"内容对照",而是别的东西):
    #   1. 供体 episode 必须**等于**本样本的 episode;
    #   2. 两个来源步必须不同——相同就意味着负样本与 SA 是同一个集合;
    #   3. 两者都必须落在 Recent-B 窗口**之外**(它们都是"被恢复的旧帧",在窗口里就说明
    #      替换根本没发生);
    #   4. distractor 帧必须真的出现在这一行的 selected_steps 里,而 oracle 帧必须**不**
    #      出现——否则字段说的是一回事,prompt 里放的是另一回事。
    """
    episode = sample["episode"]
    donor = sample.get("donor_episode")
    if donor != episode:
        raise ValueError(
            f"{where} age_matched negative must come from its own episode "
            f"(donor_episode={donor!r}, episode={episode!r}); a foreign donor is the "
            "'irrelevant' kind and cannot control for content relevance"
        )
    distractor = sample.get("distractor_source_step")
    oracle = sample.get("oracle_source_step")
    for name, value in (
        ("distractor_source_step", distractor),
        ("oracle_source_step", oracle),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{where} {name} must be a positive int, got {value!r}")
    if distractor == oracle:
        raise ValueError(
            f"{where} age_matched negative restores the oracle frame itself "
            f"(step {oracle}); it would be identical to the positive arm"
        )
    recent_floor = int(sample["decision_step"]) - int(sample["budget"])
    inside = [
        name
        for name, value in (
            ("distractor_source_step", distractor),
            ("oracle_source_step", oracle),
        )
        if int(value) >= recent_floor
    ]
    if inside:
        raise ValueError(
            f"{where} {inside} lie inside the Recent-{sample['budget']} window "
            f"(steps >= {recent_floor}); both source frames must be older history"
        )
    steps = [int(step) for step in sample["selected_steps"]]
    if int(distractor) not in steps:
        raise ValueError(
            f"{where} distractor_source_step {distractor} is absent from "
            f"selected_steps {steps}"
        )
    if int(oracle) in steps:
        raise ValueError(
            f"{where} oracle_source_step {oracle} is still present in selected_steps "
            f"{steps}; the content control must not carry the oracle frame"
        )


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
    # note (luojiaxuan): "同格式"这一条以前写死成 sparse_single_turn。v6 语料的 renderer
    # 已冻结为 official_style_sparse_multiturn,所以期望值改从**该 schema 自己的契约表**
    # 里查(仍然是一个写死的常量,只是按 schema 索引),而不是放宽成"随便什么格式"。
    expected_format = SPARSE_ARM_CONTRACTS[reference["schema_version"]][ref_slot][2]
    if (
        reference["adapter_mode"] != "bypass"
        or reference["role"] != "reference"
        or reference["prompt_format"] != expected_format
        or reference["selection_mode"] != "recent"
    ):
        raise ValueError(
            f"pair-group {pair_group!r} reference arm {ref_slot!r} is not a "
            "budget-matched, same-format, adapter-bypassed recent-K arm"
        )
    schemas = {samples[index]["schema_version"] for index in group.values()}
    if len(schemas) != 1:
        raise ValueError(
            f"pair-group {pair_group!r} mixes corpus schemas {sorted(schemas)}; "
            "one group's arms must all come from the same corpus"
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
# 桌面 schema 没有 N0(部署基线 = R0,见契约注释),留出组四臂齐即可;缺 RA 主
# claim(SA-RA)无定义、缺 S0 则 did_select 无定义,这两条与 v2/v6 相同。
SPARSE_HELDOUT_REQUIRED_SLOTS_BY_SCHEMA: dict[str, tuple[str, ...]] = {
    SPARSE_SAMPLE_SCHEMA: SPARSE_HELDOUT_REQUIRED_SLOTS,
    SPARSE_REPLACEMENT_SAMPLE_SCHEMA: SPARSE_HELDOUT_REQUIRED_SLOTS,
    SPARSE_DESKTOP_SAMPLE_SCHEMA: ("R0", "S0", "RA", "SA"),
}

# ---------------------------------------------------------------------------
# 训练目标的两个版本(唯一权威是 config.objective.kind)
# ---------------------------------------------------------------------------
# note (luojiaxuan): v5 留出集判定旧目标 ``SA − R0`` 是**假阳性发生器**。同一次训练
# 里主 claim SA−RA 从 identity 的 +0.0335 单调降到 −0.0006,而 SA−R0 却从 +0.034
# 涨到 +0.118 —— adapter 学到的是"见历史就整体放大",且对 recent 窗口的放大**强于**
# 对 sparse 选点的放大。根因在目标本身:SA−R0 = (S0−R0) + (SA−S0) 里混着两样东西,
#   (a) 冻结模型**本来就有**的选点优势 S0−R0(identity 上即 +0.0335),
#   (b) adapter 对 sparse 臂的整体抬升 SA−S0(它对 recent 臂同样成立),
# 两者任何一个变大都能把它刷高,唯独"adapter 让 sparse 选点比 recent 窗口更有用"
# 这件事**不是**刷高它的必要条件。
#
# 新目标做 RA-aware 的差中差:先对每条臂取 adapter 增量(active 分数减它**自己**的
# bypass 分数),再在 sparse 与 recent 之间做差。identity 上每个增量恒等于 0,差中差
# 于是**严格等于 0** —— 冻结模型那 +0.0335 一分也进不来,"统一放大"也被两端同时抵消。
SPARSE_OBJECTIVE_DID_RA_AWARE = "did_ra_aware"
SPARSE_OBJECTIVE_LEGACY = "legacy_sa_minus_r0"
SPARSE_OBJECTIVE_KINDS = (SPARSE_OBJECTIVE_DID_RA_AWARE, SPARSE_OBJECTIVE_LEGACY)
# 每个目标真正读进训练损失的**测量臂**(负样本臂不在此列,它们由 role 字段决定)。
# config 的 objective.excluded_arms 由这张表取补集算出并逐项比对,不再是一句散文——
# 散文写着"RA 不进损失"而代码已经在打 RA 的分,这种分叉在日志里完全看不出来。
SPARSE_OBJECTIVE_ARMS: dict[str, tuple[str, ...]] = {
    SPARSE_OBJECTIVE_DID_RA_AWARE: ("R0", "S0", "RA", "SA"),
    SPARSE_OBJECTIVE_LEGACY: ("R0", "SA"),
}
# 差中差用到的两条新臂;参考臂仍只从每条样本的 reference_arm_id 字段读,不写死。
SPARSE_RECENT_ACTIVE_SLOT = "RA"
SPARSE_SPARSE_BYPASS_SLOT = "S0"


def sparse_objective_arms(objective_kind: str) -> tuple[str, ...]:
    """Measurement arms one objective scores during training — fail-closed lookup."""
    arms = SPARSE_OBJECTIVE_ARMS.get(objective_kind)
    if arms is None:
        raise ValueError(
            f"unknown objective kind {objective_kind!r}; expected one of "
            f"{SPARSE_OBJECTIVE_KINDS}"
        )
    return arms


def sparse_objective_excluded_arms(
    objective_kind: str, *, sample_schema: str = SPARSE_SAMPLE_SCHEMA
) -> list[str]:
    """Measurement arms this objective never scores — the complement, not prose.

    # note (luojiaxuan): 补集的全集随语料 schema 变(桌面 schema 没有 N0),缺省值
    # 保持 v2 —— 既有 config 与调用点的行为逐字节不变。
    """
    consumed = set(sparse_objective_arms(objective_kind))
    return sorted(
        set(SPARSE_HELDOUT_REQUIRED_SLOTS_BY_SCHEMA[sample_schema]) - consumed
    )


DESKTOP_SPLIT_TO_TRAINER = {"train": "train", "dev": "heldout"}


def normalize_desktop_splits(
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Map the desktop corpus train/dev/test split onto trainer vocabulary.

    # note (luojiaxuan): 桌面语料按 Stage A.4 发 train/dev/test;trainer 的 split
    # 词表只有 train/heldout。映射规则是交接 §9 的语义:checkpoint 只在 Desktop dev
    # 选择(dev 即 trainer 的 heldout),test 在 policy 阶段一行都不许进来(整行剥离
    # 并计数)。只动桌面 schema 的样本,旧语料逐字节原样通过;未知 split fail-closed。
    # 返回 (kept_samples, counters);调用方必须把 counters 打印出来 —— 剥离要有据可查。
    """
    kept: list[dict[str, Any]] = []
    counters = {"desktop_dev_to_heldout": 0, "desktop_test_dropped": 0}
    for sample in samples:
        if sample.get("schema_version") != SPARSE_DESKTOP_SAMPLE_SCHEMA:
            kept.append(sample)
            continue
        split = sample.get("split")
        if split == "test":
            counters["desktop_test_dropped"] += 1
            continue
        if split not in DESKTOP_SPLIT_TO_TRAINER:
            raise ValueError(
                f"desktop sample {sample.get('sample_id')!r} carries unknown split "
                f"{split!r}; expected one of "
                f"{sorted(DESKTOP_SPLIT_TO_TRAINER) + ['test']}"
            )
        mapped = DESKTOP_SPLIT_TO_TRAINER[split]
        if mapped != split:
            counters["desktop_dev_to_heldout"] += 1
            sample = {**sample, "split": mapped}
        kept.append(sample)
    return kept, counters


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


def resolve_train_on_label_classes(config: dict[str, Any]) -> tuple[str, ...] | None:
    """``data.train_on_label_classes``, or ``None`` when the key is absent.

    # note (luojiaxuan): 缺键返回 None,含义是"**不过滤**" —— 也就是今天的行为:每个
    # 训练组都必须臂齐,缺臂当场报错。既有 config 一个字都不用改。
    # 声明了才过滤,而且必须是已知类名的非空列表:拼错一个类名会静默把整批组过滤掉,
    # 训练照跑、组数变少、日志上看不出来,这正是要防的。
    """
    data = config.get("data")
    if not isinstance(data, dict) or SPARSE_LABEL_CLASS_CONFIG_KEY not in data:
        return None
    declared = data[SPARSE_LABEL_CLASS_CONFIG_KEY]
    if not isinstance(declared, list) or not declared:
        raise ValueError(
            f"data.{SPARSE_LABEL_CLASS_CONFIG_KEY} must be a non-empty list of "
            f"label classes from {list(SPARSE_LABEL_CLASSES)}"
        )
    unknown = sorted(str(name) for name in declared if name not in SPARSE_LABEL_CLASSES)
    if unknown:
        raise ValueError(
            f"data.{SPARSE_LABEL_CLASS_CONFIG_KEY} references unknown label classes "
            f"{unknown}; known classes are {list(SPARSE_LABEL_CLASSES)}"
        )
    if len(set(declared)) != len(declared):
        raise ValueError(f"data.{SPARSE_LABEL_CLASS_CONFIG_KEY} repeats a label class")
    return tuple(str(name) for name in declared)


def filter_samples_by_label_class(
    samples: list[dict[str, Any]],
    *,
    allowed: Sequence[str],
    selection_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Drop whole pair-groups whose ``label_class`` is not in ``allowed``.

    整组进出,不会把一个组切成半个;样本顺序原样保留。计数写进 ``selection_out``,
    由调用方打印 —— 排除必须是**有记录**的,而不是安静地少几组。

    # note (luojiaxuan): 语料没有 label_class 字段却声明了这个键,是 config 与语料
    # 不匹配,直接报错:这种情况下"过滤"要么全丢要么全留,两个结果都是错的。
    """
    allowed_set = set(allowed)
    order: list[str] = []
    members: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        pair_group = sample["pair_group"]
        if pair_group not in members:
            members[pair_group] = []
            order.append(pair_group)
        members[pair_group].append(index)

    kept_groups: dict[str, dict[str, int]] = {}
    skipped_groups: dict[str, dict[str, int]] = {}
    keep: set[str] = set()
    kept_samples = skipped_samples = 0
    kept_train_groups = 0
    for pair_group in order:
        indices = members[pair_group]
        classes = {samples[index].get("label_class") for index in indices}
        if classes == {None}:
            raise ValueError(
                f"data.{SPARSE_LABEL_CLASS_CONFIG_KEY} is declared but pair-group "
                f"{pair_group!r} carries no label_class field; that key only applies "
                f"to a {SPARSE_REPLACEMENT_SAMPLE_SCHEMA} corpus"
            )
        if len(classes) != 1:
            raise ValueError(
                f"pair-group {pair_group!r} disagrees on label_class {sorted(classes)}"
            )
        label_class = classes.pop()
        if label_class not in SPARSE_LABEL_CLASSES:
            raise ValueError(
                f"pair-group {pair_group!r} declares unknown label_class "
                f"{label_class!r}"
            )
        splits = {samples[index]["split"] for index in indices}
        if len(splits) != 1:
            raise ValueError(
                f"pair-group {pair_group!r} spans several splits {sorted(splits)}"
            )
        split = splits.pop()
        bucket = kept_groups if label_class in allowed_set else skipped_groups
        bucket.setdefault(split, {})[label_class] = (
            bucket.setdefault(split, {}).get(label_class, 0) + 1
        )
        if label_class in allowed_set:
            keep.add(pair_group)
            kept_samples += len(indices)
            kept_train_groups += int(split == "train")
        else:
            skipped_samples += len(indices)

    if kept_train_groups == 0:
        raise ValueError(
            f"data.{SPARSE_LABEL_CLASS_CONFIG_KEY}={list(allowed)} excluded every "
            "train pair-group; there is nothing left to train on"
        )
    if selection_out is not None:
        selection_out.clear()
        selection_out.update(
            {
                "filter": list(allowed),
                "kept_groups": {
                    split: dict(sorted(counts.items()))
                    for split, counts in sorted(kept_groups.items())
                },
                "skipped_groups": {
                    split: dict(sorted(counts.items()))
                    for split, counts in sorted(skipped_groups.items())
                },
                "kept_samples": kept_samples,
                "skipped_samples": skipped_samples,
            }
        )
    return [sample for sample in samples if sample["pair_group"] in keep]


def build_sparse_history_units(
    samples: list[dict[str, Any]], *, objective_kind: str
) -> tuple[list[tuple[str, dict[str, int], None]], set[str]]:
    """Return (units, heldout_episodes) for the sparse-history five-arm corpus.

    # note (luojiaxuan): split 的唯一权威是样本的 ``split`` 字段(审计 P0-4);
    # trainer 不再调用 heldout_episode_set 重算 hash——构建期与训练期各算一次
    # hash 是留出集悄悄漂移的经典成因。索引主键是 arm_slot,重复即报错。
    #
    # ``objective_kind`` 只决定**这个目标需要哪些臂齐全**,不碰 split、不碰分组。
    # did_ra_aware 要求 R0/S0/RA/SA 四臂俱全:缺 S0 则 A_c 无定义,缺 RA 则 A_r 无
    # 定义,而这两者正是"差中差在 identity 上恒为 0"的构造前提。这里当场报错而不是
    # 留到损失里逐组返回 None —— 后者会让主 claim 的分母悄悄变小,stdout 上只表现为
    # 组数变少(审计第 8 条点名的失败模式)。
    """
    required = tuple(
        slot for slot in sparse_objective_arms(objective_kind) if slot != "R0"
    )
    buckets, heldout_episodes = _sparse_groups_by_split(samples)
    units: list[tuple[str, dict[str, int], None]] = []
    for pair_group, group in sorted(buckets["train"].items()):
        # 参考臂由 validate_sparse_group 按 reference_arm_id 查在不在组里,所以
        # required 里把 R0 摘掉——这里不得再写死"R0"。
        validate_sparse_group(samples, pair_group=pair_group, group=group)
        missing = [slot for slot in required if slot not in group]
        if missing:
            raise ValueError(
                f"train pair-group {pair_group!r} lacks arms {missing} required by "
                f"objective {objective_kind!r}; slots={sorted(group)}"
            )
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
        # 必需臂集合随语料 schema 变(validate_sparse_group 已保证组内 schema 唯一)。
        schema = samples[next(iter(group.values()))]["schema_version"]
        missing = [
            slot
            for slot in SPARSE_HELDOUT_REQUIRED_SLOTS_BY_SCHEMA[schema]
            if slot not in group
        ]
        if missing:
            raise ValueError(
                f"heldout pair-group {pair_group!r} lacks arms {missing}; every "
                "scored group needs its schema's full measurement arms or its "
                "derived quantities are undefined"
            )
        groups[pair_group] = group
    return groups


# ---------------------------------------------------------------------------
# rank 调度:按 K 分桶拉平 accumulation window 内的 straggler
# ---------------------------------------------------------------------------
# note (luojiaxuan): 这一段**只决定"哪个 rank 在哪个 accumulation window 里跑哪一
# 组"**,不碰任何一处损失算术。能这么做的前提是现有切分方式的一条性质:旧代码
# ``shard = order[rank::world_size]`` 配合 ``(position + 1) % accumulation == 0``
# 处的 all_reduce,意味着第 w 个优化步恰好消费 ``order`` 上连续的一段
# ``order[w*B : (w+1)*B]``(B = accumulation * world_size),各 rank 只是按 stride
# 从这 B 组里各取 accumulation 组;梯度先在 rank 内累加、再 all_reduce 求和除以
# world_size,所以**一个优化步的总梯度只取决于这 B 组的集合,与谁跑哪一组无关**。
# 于是"在 block 内重新分配"是逐步等价的:同一个 seed 下每一个优化步看到的组集合
# 与旧代码逐组相同,只是各 rank 到达梯度同步屏障的时间被拉平了(K=1 组 6 次前向、
# K≥2 组 12 次前向,随机切分下同一 window 内快慢差可到 2 倍,快的 rank 空等)。
#
# 三条不能做的事,每一条都会**静默**改变训练而不是报错:
#   * 跨 block 搬运组——会改变某个优化步的组集合,即改变那一步的梯度;
#   * 全局按 K 排序——K 的分布会随训练进程漂移,后期梯度分布突变;
#   * 改变任一 rank 的 shard 长度——``(position + 1) % accumulation`` 的屏障位置
#     会整体错位,连"哪些组进同一个优化步"都不再对得上。
# 因此:只在**整块**内做平衡,末尾不满一个 block 的尾巴照旧走 stride(full*block 是
# world_size 的整数倍,尾部 offset % world_size 与旧代码的全局 position % world_size
# 逐个相同),各 rank 的 shard 长度与旧代码完全一致。
def sparse_group_schedule_cost(
    samples: list[dict[str, Any]], group: dict[str, int], *, objective_kind: str
) -> tuple[int, int]:
    """Return ``(forward_count, budget)`` — one pair-group's scheduling cost proxy.

    # note (luojiaxuan): 纯启发式,只喂给调度器,**永远不进损失**,估偏了最多是没把
    # 屏障拉平,不会动到任何一个梯度。前向次数按两遍法数,n = 该组实际存在的负样本数:
    #   * legacy_sa_minus_r0:SA + R0 两次 no-grad,每个负样本 active/bypass 各一次
    #     no-grad,再加最多 1+n 次带梯度前向 —— 合计 3*(1+n),即 n=1 → 6、n=3 → 12;
    #   * did_ra_aware:第一遍多了 S0 与 RA 两条臂(S0 冻结可缓存、RA 是 active 的
    #     no-grad 取值),第二遍多了 RA 的带梯度前向(L_select 与 L_cap 都要对 A_r
    #     的 active 端求导),即 (4 + 2n) + (2 + n) = 6 + 3n —— n=1 → 9、n=3 → 15。
    # budget 作次要项:同为 n=3 的组里 K=4 的序列比 K=2 长(每臂 K+1 张图),前向次数
    # 拉平之后再拉平图数,屏障对得更紧。
    """
    negatives = sum(
        1 for index in group.values() if samples[index]["role"] == "negative"
    )
    budget = int(samples[group[SPARSE_POSITIVE_SLOT]]["budget"])
    if objective_kind == SPARSE_OBJECTIVE_DID_RA_AWARE:
        return 6 + 3 * negatives, budget
    sparse_objective_arms(objective_kind)  # fail-closed:未知目标不给默认代价
    return 3 * (1 + negatives), budget


def _assert_shard_partition_preserved(
    order: list[int],
    shards: list[list[int]],
    stride: list[list[int]],
    *,
    world_size: int,
    accumulation: int,
    full_blocks: int,
) -> None:
    """Fail-closed proof that bucketing only permuted ranks **inside** one window.

    # note (luojiaxuan): 三条不变量,任何一条破了训练都会悄悄变成另一个实验:
    #   1. 组的**多重集**逐组相同——不丢组、不重复组(集合相等 + 每组计数相等);
    #   2. 每个 rank 的 shard 长度与旧 stride 切分逐个相同——屏障位置一个不动;
    #   3. 每个 accumulation window 的组集合与旧 stride 切分相同——这一条才是
    #      "每个优化步的总梯度不变"的充要条件,前两条只是它的必要条件。
    # 断言而不是单元测试:分桶是每个 epoch 现算的,真正要挡的是"某次改了成本函数或
    # 块大小之后,某个 epoch 的某个 window 悄悄少了一组",那只有在线检查才拦得住。
    """
    produced: dict[int, int] = {}
    for shard in shards:
        for unit_index in shard:
            produced[unit_index] = produced.get(unit_index, 0) + 1
    expected: dict[int, int] = {}
    for unit_index in order:
        expected[unit_index] = expected.get(unit_index, 0) + 1
    if produced != expected:
        raise AssertionError(
            "K-bucketing changed the multiset of training units "
            f"({len(produced)} distinct vs {len(expected)} expected)"
        )
    lengths = [len(shard) for shard in shards]
    stride_lengths = [len(part) for part in stride]
    if lengths != stride_lengths:
        raise AssertionError(
            f"K-bucketing changed shard lengths {lengths} != {stride_lengths}; "
            "the gradient-sync barrier would move"
        )
    block = accumulation * world_size
    for window in range(full_blocks):
        low = window * accumulation
        seen: dict[int, int] = {}
        for shard in shards:
            for unit_index in shard[low : low + accumulation]:
                seen[unit_index] = seen.get(unit_index, 0) + 1
        want: dict[int, int] = {}
        for unit_index in order[window * block : (window + 1) * block]:
            want[unit_index] = want.get(unit_index, 0) + 1
        if seen != want:
            raise AssertionError(
                f"accumulation window {window} no longer consumes the same "
                "pair-groups; that optimizer step's total gradient would change"
            )


def balanced_sparse_shards(
    order: list[int],
    costs: dict[int, tuple[int, int]],
    *,
    world_size: int,
    accumulation: int,
) -> list[list[int]]:
    """Split one epoch's shuffled unit order into cost-balanced per-rank shards.

    ``order`` is the already-seed-shuffled unit ordering; ``costs`` maps a unit
    index to :func:`sparse_group_schedule_cost`. Returns one shard per rank.

    # note (luojiaxuan): 策略 = 「先按 seed 全局打散 → 切成 B = accumulation *
    # world_size 的整块 → 块内 LPT(longest-processing-time)装箱,每个 rank 恰好
    # accumulation 组」。选 LPT 而不是"按 K 分层再 round-robin"是因为后者在 K 分布
    # 不整除 world_size 时(实际分布 0.15/0.25/0.25/0.35)仍会留下整组的偏斜,而
    # LPT 直接对着代价装箱,块内各 rank 的前向次数差最多一组的代价。
    # 随机性从两处进来,所以不会出现"所有 K=4 排到最后"的梯度分布漂移:
    #   * 块的划分来自 seed 打散后的 order,块与块之间的 K 组成仍是随机的;
    #   * 块内并列代价的 tie-break 用**打散后的次序**(而不是单元编号),分配本身
    #     也就继承了 seed 的随机性;
    #   * 装箱完成后,每个 rank 块内的遍历次序退回打散后的 order 次序,不留"每个
    #     window 都先重后轻"这种人造结构(优化步只在 window 末尾发生,块内次序对
    #     梯度无影响,这一步纯粹是不给后续分析引入假信号)。
    # 确定性:全过程是 (order, costs, world_size, accumulation) 的纯函数,不再抽随机
    # 数,所以同 seed 逐组可复现;也因此**每个 rank 各自算出的是同一份分配**,不需要
    # 任何通信——若这里引入了 rank 相关的随机性,各 rank 的 window 就会分叉。
    """
    stride = [order[rank::world_size] for rank in range(world_size)]
    if world_size < 2 or accumulation < 1:
        return stride
    block = accumulation * world_size
    full_blocks = len(order) // block
    shards: list[list[int]] = [[] for _ in range(world_size)]
    for window_index in range(full_blocks):
        start = window_index * block
        window = order[start : start + block]
        heavy_first = sorted(
            range(len(window)),
            key=lambda position: (
                -costs[window[position]][0],
                -costs[window[position]][1],
                position,
            ),
        )
        # loads[rank] = [前向次数累计, 预算累计, rank];容量硬上限是 accumulation。
        loads = [[0, 0, rank] for rank in range(world_size)]
        picked: list[list[int]] = [[] for _ in range(world_size)]
        for position in heavy_first:
            forwards, budget = costs[window[position]]
            # note (luojiaxuan): 并列时的 rank 名次按 window 轮转。不轮转的话所有
            # load 都从 0 起步、并列一律给最小 rank,于是**每个 window 最重的那一组
            # 恒定落在 rank 0**;梯度是全 rank 求和因而不受影响,但 rank 0 打印的
            # mean_loss 是它本地那 8 组的均值,会被系统性地偏向大 K 组,让日志里的
            # 损失曲线偏离全局均值。轮转后这个"接最重一组"的角色在各 rank 间平摊,
            # 且仍是 (window_index, rank) 的纯函数,确定性不受影响。
            target = min(
                (load for load in loads if len(picked[load[2]]) < accumulation),
                key=lambda load: (
                    load[0],
                    load[1],
                    (load[2] - window_index) % world_size,
                ),
            )
            target[0] += forwards
            target[1] += budget
            picked[target[2]].append(position)
        for rank in range(world_size):
            shards[rank].extend(window[position] for position in sorted(picked[rank]))
    for offset, unit_index in enumerate(order[full_blocks * block :]):
        shards[offset % world_size].append(unit_index)
    _assert_shard_partition_preserved(
        order,
        shards,
        stride,
        world_size=world_size,
        accumulation=accumulation,
        full_blocks=full_blocks,
    )
    return shards


def sparse_diagnostic_keys(
    negative_kind: str, *, arm_slot: str | None = None
) -> tuple[str, str]:
    """Return (rank-gap key, drift key) for one negative kind — the gate spelling.

    # note (luojiaxuan): 审计第 10 条。损失里原来写 ``SA_minus_step_shuffled`` /
    # ``step_shuffled_drift``,而 gates 词表与 config.gates.must_pass 写的是
    # ``SA_minus_SA_neg_step_shuffled`` / ``step_shuffled_drift_abs``,于是"诊断可
    # 直接与 config.gates 的量名对齐"那句注释是假的:两套名字谁也对不上谁。现在
    # **名字只有这一处定义**,损失的诊断键与下面的 SPARSE_GATE_VOCABULARY /
    # SPARSE_REQUIRED_GATES 都由它生成,对齐是结构性的而不是靠人肉同步。
    # 记法统一到 arm_slot:差值量名 = ``SA_minus_<arm_slot>``。v2/v6 的负样本臂
    # slot 就是 ``SA_neg_<kind>``(缺省分支,取值逐字节不变);桌面语料的负臂叫
    # WA(交接 §6 臂名),调用方给 arm_slot 时以它为准 —— 权威映射是
    # SPARSE_NEGATIVE_SLOTS_BY_SCHEMA,不许再从 kind 拼前缀。
    """
    if arm_slot is not None:
        return f"SA_minus_{arm_slot}", f"{negative_kind}_drift_abs"
    return f"SA_minus_SA_neg_{negative_kind}", f"{negative_kind}_drift_abs"


def sparse_content_diagnostic_key(negative_kind: str) -> str:
    """Runtime-only diagnostic name for one negative's difference-in-differences.

    # note (luojiaxuan): 与 sparse_diagnostic_keys 分开,因为**语义不同**:后者的
    # ``SA_minus_SA_neg_<kind>`` 是原始差 ℓSA−ℓn(留出集打分脚本按 gates 词表算的
    # 就是它,gate 阈值也挂在它上面),这里是差中差 A_c−A_n。差中差**不进 gate 词表**
    # ——本次只换训练目标,不动已冻结的验收标准(gates schema v2)。
    """
    return f"did_content_{negative_kind}"


# 运行指标而非 gate 量:诊断里除这些键之外的每一个都必须在 SPARSE_GATE_VOCABULARY 里
# (审计第 10 条)。集中定义一处,测试直接读它,免得"损失新加了一个分量诊断"与"测试
# 里硬编码的白名单"两处各自漂移。
SPARSE_RUNTIME_DIAGNOSTIC_KEYS = frozenset(
    {
        "loss", "negatives",
        "loss_select", "loss_gain", "loss_content", "loss_cap", "loss_l2",
    }
    | {sparse_content_diagnostic_key(kind) for kind in SPARSE_ALL_NEGATIVE_KINDS}
)


def _sparse_history_group_loss(
    *,
    samples: list[dict[str, Any]],
    group: dict[str, int],
    forward: Any,
    training: dict[str, Any],
    adapter_parameters: list[Any],
    accumulation: int,
    torch: Any,
    objective_kind: str,
    diagnostics_out: dict[str, float] | None = None,
) -> float | None:
    """Dispatch one pair-group to the configured sparse-history objective.

    Returns ``float`` (this group's loss) or ``None`` (skipped); diagnostics are
    delivered **only** by filling ``diagnostics_out`` in place.

    # note (luojiaxuan): ``objective_kind`` 是必填关键字,没有缺省值。给默认值等于
    # 让"忘了传"静默落到某一个目标上,而两个目标训出来的 adapter 语义完全不同——
    # 这正是本次要修的那类"config 说一套、代码跑另一套"的分叉。
    #
    # 两个 body 各自返回 ``(total | None, diagnostics, ref_slot)``:total 为 None 表示
    # 某条臂编码失败、本组整组跳过(与旧行为逐字一致,此时不写滚动窗口);诊断的落地、
    # 25 组滚动打印、以及 "total == 0.0 视作跳过" 的规则集中在这里,两个目标共用。
    """
    if objective_kind == SPARSE_OBJECTIVE_DID_RA_AWARE:
        body = _sparse_history_group_loss_did
    elif objective_kind == SPARSE_OBJECTIVE_LEGACY:
        body = _sparse_history_group_loss_legacy
    else:
        raise ValueError(
            f"unknown objective kind {objective_kind!r}; expected one of "
            f"{SPARSE_OBJECTIVE_KINDS}"
        )
    total, diagnostics, ref_slot = body(
        samples=samples,
        group=group,
        forward=forward,
        training=training,
        adapter_parameters=adapter_parameters,
        accumulation=accumulation,
        torch=torch,
    )
    if total is None:
        return None
    diagnostics["loss"] = total
    # note (luojiaxuan): 审计第 10 条。诊断的**唯一对外通道**是 diagnostics_out
    # (原地填充),返回值恒为 float | None——旧版把 dict 当返回值,契约与测试都按
    # "标量损失或 None"写,调用方多写一层 ["loss"] 才能拿到数,任何一处忘了就是
    # 静默类型错。``_SPARSE_DIAG`` 只是本进程的 25 组滚动打印窗口,不是返回通道。
    # 键名:除 SPARSE_RUNTIME_DIAGNOSTIC_KEYS 之外的每一个都来自 sparse_diagnostic_keys
    # 与 SPARSE_DERIVED_QUANTITIES,与 config.gates 的量名逐字相同(见 F3-d)。
    # 唯一要注意的是聚合次序:这里的 <kind>_drift_abs 是**本组** |A_n|,滚动窗口打印的
    # 是 mean|drift|;gate 判定按 config.gates.drift_definition 在留出集上算 |mean drift|。
    # 两者同名、同符号约定(非负、越小越好),前者是后者的上界。
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
                    "objective_kind": objective_kind,
                    "groups": len(_SPARSE_DIAG),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    # note (luojiaxuan): total == 0.0 意味着所有 hinge / dead-zone 都未触发、L2 也没有
    # 质量(权重为 0 或没有 adapter 参数),此时上面一个带梯度前向都没跑,返回 None 让
    # 调用方把这组算作"跳过",而不是把 0 计进 running_loss 的分母。
    return None if total == 0.0 else total


def _sparse_history_group_loss_did(
    *,
    samples: list[dict[str, Any]],
    group: dict[str, int],
    forward: Any,
    training: dict[str, Any],
    adapter_parameters: list[Any],
    accumulation: int,
    torch: Any,
) -> tuple[float | None, dict[str, float], str]:
    """RA-aware difference-in-differences objective (``objective.kind`` 权威).

    # note (luojiaxuan): 记号 —— 每条臂先取它**自己**的 adapter 增量,
    #   A_c = ℓ_SA − ℓ_S0                 (adapter 对"正确稀疏历史"的增量)
    #   A_r = ℓ_RA − ℓ_R0                 (adapter 对"最近窗口"的增量)
    #   A_n = ℓ_n_active − ℓ_n_bypass     (adapter 对第 n 条错误历史的增量)
    # 其中 ℓ_S0 / ℓ_R0 / ℓ_n_bypass 是**冻结量**(样本字段或显式覆盖写着 bypass,跑在
    # 冻结 policy 上、与 adapter 参数无关),因此三者都走冻结分数缓存;ℓ_SA / ℓ_RA /
    # ℓ_n_active 是 adapter active,需要带梯度前向。w_n = scale_n / Σ_m scale_m,只对
    # 该组**实际存在**的负样本归一化。损失:
    #   L_select  = w_s * [m_s − (A_c − A_r)]+
    #   L_gain    = w_g * [m_g − A_c]+
    #   L_content = w_c * Σ_n w_n * [m_c − (A_c − A_n)]+
    #   L_cap     = λ * [|A_r| − ε]+ + λ * Σ_n w_n * [|A_n| − ε]+
    #   L         = L_select + L_gain + L_content + L_cap + l2 * Σ(‖A‖²+‖B‖²),CE 恒 0
    #
    # 为什么是 A_c − A_r 而不是 ℓ_SA − ℓ_R0:后者 = (ℓ_S0 − ℓ_R0) + A_c,第一项是**冻结
    # 模型自带**的选点优势(identity 上就有 +0.0335),第二项对 recent 臂同样成立,于是
    # "见历史就统一放大"能把它一路刷到 +0.118 而主 claim SA−RA 反而掉到 −0.0006。
    # A_c − A_r 在 identity 上**严格等于 0**(每个增量各自为 0),把冻结模型的既有优势
    # 与"统一放大"两条捷径同时封死:只有 adapter 对 sparse 的抬升**超过**它对 recent
    # 的抬升,这一项才会下降。
    #
    # L_gain 不是可有可无的补丁:单看 L_select,压低 ℓ_RA 与抬高 ℓ_SA 同样能减损失,
    # 而"把 recent 臂打坏"不是我们要的能力。L_gain = [m_g − A_c]+ 要求 sparse 臂的绝对
    # 增量自己也得为正,于是刷 select 必须真的抬 SA。
    #
    # L_cap 用 **dead-zone hinge**([|A| − ε]+)而不是旧的 SmoothL1,理由是尺度而不是
    # 训练偶然性:drift 落在 0.06 量级时,权重 2 的 SmoothL1 梯度只有 2*0.06 ≈ 0.12,
    # 而 select/gain/content 三个 rank hinge 一旦激活梯度量级就是 1,anchor 压根压不住,
    # 于是"统一放大"是**损失尺度决定**的最优解。dead-zone hinge 在 |A| > ε 之后梯度恒为
    # λ = 2,与 rank 项同量级;在 |A| ≤ ε 之内梯度恒为 0,不去规定 adapter 该落在哪。
    #
    # 两遍法保持不变:先 no-grad 取全部 ℓ 值、解析求各前向的次梯度权重,再逐臂在各自
    # history_adapter_scope 内带梯度前向并立即 backward,同一时刻只活一张计算图(这是
    # 为规避梯度检查点重算跑在 autograd 线程时 ContextVar 读空的崩溃)。带梯度的臂只有
    # SA / RA / 各负样本 —— A_r 的 bypass 端 R0 是冻结的,只有 active 端需要梯度。
    """
    ref_slot = sparse_reference_slot(samples, group)
    # note (luojiaxuan): 审计第 8 条。缺臂时旧代码走 forward(...) → None → 静默跳过
    # 一组:主 claim 的分母悄悄变小,而 stdout 上只会看到组数变少。上游
    # build_sparse_history_units 确实也查这一条,但损失函数是被测试与将来其它调用方
    # 直接调用的入口,不能把 fail-closed 外包给调用方。
    if ref_slot not in group:
        raise ValueError(
            f"pair-group declares reference_arm_id {ref_slot!r} but that arm is "
            f"absent; slots={sorted(group)}"
        )
    missing = [
        slot
        for slot in (
            SPARSE_SPARSE_BYPASS_SLOT,
            SPARSE_RECENT_ACTIVE_SLOT,
            SPARSE_POSITIVE_SLOT,
        )
        if slot not in group
    ]
    if missing:
        raise ValueError(
            f"objective {SPARSE_OBJECTIVE_DID_RA_AWARE!r} needs the full "
            f"{ref_slot}/S0/RA/SA quartet; pair-group lacks {missing}; "
            f"slots={sorted(group)}"
        )

    # 第一遍:四条测量臂全部 no-grad 取值。S0 与 R0 的样本字段写着 bypass,因此这两次
    # 前向天然走冻结分数缓存;SA 与 RA 是 active,取值这一次不带梯度。
    sparse_active = forward(SPARSE_POSITIVE_SLOT, grad=False)
    sparse_frozen = forward(SPARSE_SPARSE_BYPASS_SLOT, grad=False)
    recent_active = forward(SPARSE_RECENT_ACTIVE_SLOT, grad=False)
    recent_frozen = forward(ref_slot, grad=False)
    quartet = (sparse_active, sparse_frozen, recent_active, recent_frozen)
    if any(value is None for value in quartet):
        return None, {}, ref_slot
    l_sa = float(sparse_active)
    l_s0 = float(sparse_frozen)
    l_ra = float(recent_active)
    l_r0 = float(recent_frozen)
    a_c = l_sa - l_s0
    a_r = l_ra - l_r0

    select_margin = float(training.get("sparse_select_margin", 0.01))
    gain_margin = float(training.get("sparse_gain_margin", 0.01))
    content_margin = float(training.get("sparse_content_margin", 0.01))
    select_weight = float(training.get("sparse_select_weight", 1.0))
    gain_weight = float(training.get("sparse_gain_weight", 1.0))
    content_weight = float(training.get("sparse_content_weight", 1.0))
    cap_epsilon = float(training.get("sparse_drift_cap_eps", 0.02))
    cap_weight = float(training.get("sparse_drift_cap_weight", 2.0))
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
        # bypass 前向是同一条负样本在冻结 policy 上的分数,A_n 的基准点
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

    # 次梯度权重。可导量只有三类 active 分数,而 dA_c/dℓ_SA = dA_r/dℓ_RA =
    # dA_n/dℓ_n_active = 1,所以每一项的权重就是它对该 A 的偏导。
    weight_sparse = 0.0
    weight_recent = 0.0
    weights: dict[str, float] = {}
    loss_select = 0.0
    loss_gain = 0.0
    loss_content = 0.0
    loss_cap = 0.0
    diagnostics: dict[str, float] = {
        "SA_minus_R0": l_sa - l_r0,
        "SA_minus_RA": l_sa - l_ra,
        "frozen_selection_effect": l_s0 - l_r0,
        "adapter_on_sparse": a_c,
        "adapter_on_recent": a_r,
        "did_select": a_c - a_r,
        # 只数真正参与损失的负样本,和归一化分母同源
        "negatives": float(len(scored)),
    }

    select_slack = select_margin - (a_c - a_r)
    if select_slack > 0.0:
        loss_select = select_weight * select_slack
        weight_sparse -= select_weight
        weight_recent += select_weight

    gain_slack = gain_margin - a_c
    if gain_slack > 0.0:
        loss_gain = gain_weight * gain_slack
        weight_sparse -= gain_weight

    recent_excess = abs(a_r) - cap_epsilon
    if recent_excess > 0.0:
        loss_cap += cap_weight * recent_excess
        weight_recent += cap_weight * (1.0 if a_r > 0.0 else -1.0)

    for slot, kind, _scale, ln, ln_frozen in scored:
        share = normalized[slot]
        a_n = ln - ln_frozen
        # 量名记法 SA_minus_<arm_slot>:v2/v6 的 slot 本就是 SA_neg_<kind>,传 slot
        # 后取值逐字节不变;桌面语料由此得到 SA_minus_WA。
        gap_key, drift_key = sparse_diagnostic_keys(kind, arm_slot=slot)
        # gap_key 仍是原始差 ℓSA−ℓn(gate 词表与留出集打分脚本认的就是它),
        # drift_key 仍是 |A_n|(与 P1-4 之后的定义逐字相同),差中差另开一个运行键。
        diagnostics[gap_key] = l_sa - ln
        diagnostics[drift_key] = abs(a_n)
        diagnostics[sparse_content_diagnostic_key(kind)] = a_c - a_n
        content_slack = content_margin - (a_c - a_n)
        if content_slack > 0.0:
            loss_content += content_weight * share * content_slack
            weight_sparse -= content_weight * share
            weights[slot] = weights.get(slot, 0.0) + content_weight * share
        negative_excess = abs(a_n) - cap_epsilon
        if negative_excess > 0.0:
            loss_cap += cap_weight * share * negative_excess
            weights[slot] = weights.get(slot, 0.0) + cap_weight * share * (
                1.0 if a_n > 0.0 else -1.0
            )

    total = loss_select + loss_gain + loss_content + loss_cap
    # 第二遍:逐臂带梯度前向并立即 backward。RA 在这里出现是 did 目标的新增开销 ——
    # L_select 与 L_cap 都要对 A_r 的 active 端求导,而 R0 那端是冻结的。
    for slot, weight in (
        (SPARSE_POSITIVE_SLOT, weight_sparse),
        (SPARSE_RECENT_ACTIVE_SLOT, weight_recent),
        *weights.items(),
    ):
        if weight == 0.0:
            continue
        forward(slot, grad=True, backward_weight=weight / accumulation)

    loss_l2 = 0.0
    if l2_weight > 0.0 and adapter_parameters:
        l2_term = l2_weight * sum(
            parameter.pow(2).sum() for parameter in adapter_parameters
        )
        loss_l2 = float(l2_term.detach())
        total += loss_l2
        (l2_term / accumulation).backward()

    diagnostics["loss_select"] = loss_select
    diagnostics["loss_gain"] = loss_gain
    diagnostics["loss_content"] = loss_content
    diagnostics["loss_cap"] = loss_cap
    diagnostics["loss_l2"] = loss_l2
    return total, diagnostics, ref_slot


def _sparse_history_group_loss_legacy(
    *,
    samples: list[dict[str, Any]],
    group: dict[str, int],
    forward: Any,
    training: dict[str, Any],
    adapter_parameters: list[Any],
    accumulation: int,
    torch: Any,
) -> tuple[float | None, dict[str, float], str]:
    """DEPRECATED — sparse-history objective against the budget-matched R0 reference.

    # note (luojiaxuan): **已弃用,保留只为可复现旧 run。** v5 留出集证明这个目标是
    # 假阳性发生器:``ℓc − ℓr = (ℓ_S0 − ℓ_R0) + (ℓ_SA − ℓ_S0)`` 里混着冻结模型自带的
    # 选点优势与 adapter 的整体放大,"见历史就放大"能把它从 +0.034 刷到 +0.118,同时
    # 主 claim SA−RA 从 +0.0335 掉到 −0.0006。新 run 一律用 did_ra_aware;这段算术保持
    # 逐字不动(改它等于让旧 checkpoint 不再可复现),只是把诊断落地与滚动打印上提到
    # 分发器,因此返回 (total, diagnostics, ref_slot) 而不再自己写 _SPARSE_DIAG。
    #
    # 记号 ℓc=SA(adapter active,带梯度)、ℓr=R0(同格式、
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
        return None, {}, ref_slot
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
        gap_key, drift_key = sparse_diagnostic_keys(kind, arm_slot=slot)
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

    return total, diagnostics, ref_slot


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


def effective_adapter_mode(
    sample: dict[str, Any], adapter_mode: str | None = None
) -> str:
    """The adapter mode one forward actually runs under — the only definition.

    # note (luojiaxuan): 这行分派("显式覆盖优先,否则以样本字段为准")原来只存在于
    # adapter_context_for_sample 里。冻结分数缓存必须问同一个问题——"这次前向到底跑
    # 在不在 adapter 上"——而它在 encode 之前就得知道答案(命中就不 encode 了),
    # 拿不到 context。抄一份判断正是审计 P0-4 点名的分叉成因(两处对"谁 bypass"给出
    # 不同答案时,日志里看不出任何异常),所以提成一个函数,两个调用方共用。
    # 样本缺 adapter_mode 字段时故意 KeyError,不给任何默认值。
    """
    return sample["adapter_mode"] if adapter_mode is None else adapter_mode


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
    mode = effective_adapter_mode(sample, adapter_mode)
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


# ---------------------------------------------------------------------------
# 冻结 bypass 分数的磁盘缓存(默认关闭)
# ---------------------------------------------------------------------------
# note (luojiaxuan): 可缓存量的定义:一次 **no-grad 且有效 adapter 模式为 bypass**
# 的 teacher-forced 打分。这种前向跑在冻结策略上,与 adapter 参数完全无关,因此对
# 同一条样本在整个训练过程中恒定 —— 缓存命中返回的是与重算**逐位相同**的 float,
# 损失与各前向的次梯度权重一个都不变。当前每组命中 4 次:P1-4 的 3 个 per-negative
# bypass 锚点 ℓn⁰,以及 R0 参考臂本身(它的样本字段就写着 adapter_mode=bypass)。
#
# 收益边界,别误读:``epochs=1`` 的正常训练里每组只见一次,**这一轮一次都不会命中**,
# 提速为 0(只多写一份缓存)。真正吃到 12→8 次前向(约 33%)的是**第二遍及以后**:
# 多 epoch、断点重跑同一份语料、以及固定语料只扫超参的搜索。所以这个开关默认关闭,
# 由调用方在确实会重复扫同一份语料时显式打开。
#
# 正确性由指纹保证,而不是由"目录名不同"这种约定保证:分数由 (冻结模型快照, 语料,
# 编码路径, 视觉 token 预算, 打分函数) 共同决定,其中任何一项变了,旧数就不再是这条
# 样本的分数。指纹逐条存在记录里,不匹配即当未命中重算 —— 绝不静默复用。
FROZEN_SCORE_CACHE_SCHEMA = "causalcache.frozen_bypass_score_cache.v1"
_FROZEN_SCORE_SHARD_GLOB = "scores-rank*.jsonl"


def frozen_score_cache_base_digest(
    *,
    policy_snapshot_sha256: str,
    model_dir: Path,
    dataset_manifest_sha256: str,
    visual_tokens_per_image: int,
) -> str:
    """Digest of everything **global** a frozen bypass score depends on.

    # note (luojiaxuan): 逐项的理由(少任何一项都会造成静默复用旧分数):
    #   * policy_snapshot_sha256 —— 冻结策略快照的 revision。换模型即换分数,这是
    #     最要命的一项,单靠 model_dir 路径挡不住"同一路径换了权重";
    #   * model_dir —— 同一份快照 manifest 也可能指向不同的本地落盘副本;
    #   * dataset_manifest_sha256 —— 分数是 messages/target_text 的函数,而 key 用的
    #     sample_id 只是 "<pair_group>|<arm_slot>",重建语料后同名样本内容会变;
    #   * visual_tokens_per_image —— 改视觉 token 预算等于改 prompt 的 token 序列;
    #   * schema —— 缓存记录格式本身的版本。
    # 逐样本那部分(编码路径 / prompt_format)在 frozen_score_fingerprint 里补上。
    """
    return hashlib.sha256(
        "\x1f".join(
            (
                FROZEN_SCORE_CACHE_SCHEMA,
                policy_snapshot_sha256,
                str(Path(model_dir)),
                dataset_manifest_sha256,
                str(int(visual_tokens_per_image)),
            )
        ).encode("utf-8")
    ).hexdigest()


def frozen_score_fingerprint(sample: dict[str, Any], *, base_digest: str) -> str:
    """Per-sample fingerprint: exactly what this cached number is a function of."""
    return hashlib.sha256(
        "\x1f".join(
            (
                FROZEN_SCORE_CACHE_SCHEMA,
                base_digest,
                str(sample.get("schema_version")),
                # 编码路径(chat_template / exact_batch)决定 prompt 的 token 序列
                _prompt_encoding_path(sample),
                str(sample.get("prompt_format")),
                # 缓存的**永远**是 bypass 下的分数:负样本自称 active,能进缓存靠的是
                # P1-4 的显式覆盖,所以这里写死有效模式,不读样本的 adapter_mode。
                "bypass",
                "mean_target_logprob",
            )
        ).encode("utf-8")
    ).hexdigest()


class FrozenBypassScoreCache:
    """Disk cache of adapter-independent (bypass) teacher-forced scores.

    # note (luojiaxuan): 并发模型 —— **每个 rank 只写自己的分片**
    # ``scores-rank<NNN>.jsonl``,读取时把所有分片合并成一张只读视图。于是多 rank
    # 并发写不可能互相覆盖(各写各的文件名),又能互相命中彼此上一轮算过的分数。
    # 写入是原子的:整份分片先写 ``<name>.tmp-<pid>`` 再 ``os.replace`` 换名,
    # 读者永远看到的要么是旧的完整文件、要么是新的完整文件,不会读到半行。
    # (临时文件后缀不以 .jsonl 结尾,因此不会被合并时的 glob 扫进来。)
    # 攒够 ``flush_every`` 条才落盘一次:一次 bypass 前向是秒级的,而分片文件是
    # KB 级,整份重写的代价可以忽略;崩溃最多丢最后几条,缓存是纯优化,丢了只是慢。
    """

    def __init__(
        self,
        root: Path,
        *,
        base_digest: str,
        rank: int,
        flush_every: int = 32,
    ) -> None:
        self.root = Path(root)
        self.base_digest = base_digest
        self.rank = int(rank)
        self.flush_every = int(flush_every)
        self.hits = 0
        self.misses = 0
        self.fingerprint_rejects = 0
        # _merged 是查询用的全分片视图;_own 只含本 rank 负责持久化的记录。
        self._merged: dict[str, tuple[str, float]] = {}
        self._own: dict[str, tuple[str, float]] = {}
        self._pending = 0
        self.root.mkdir(parents=True, exist_ok=True)
        self._shard = self.root / f"scores-rank{self.rank:03d}.jsonl"
        self._load()

    def _load(self) -> None:
        for path in sorted(self.root.glob(_FROZEN_SCORE_SHARD_GLOB)):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            mine = path == self._shard
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = str(record["sample_id"])
                    fingerprint = str(record["fingerprint"])
                    value = float(record["score"])
                except (ValueError, TypeError, KeyError):
                    # note (luojiaxuan): 损坏行只丢这一条。缓存是纯优化,不该让一份
                    # 写坏的分片把训练拖垮 —— 丢了就是重算一次。
                    continue
                self._merged[key] = (fingerprint, value)
                if mine:
                    self._own[key] = (fingerprint, value)

    def lookup(self, sample: dict[str, Any]) -> float | None:
        """Return the cached bypass score, or None when it must be recomputed."""
        entry = self._merged.get(sample["sample_id"])
        if entry is None:
            self.misses += 1
            return None
        fingerprint, value = entry
        if fingerprint != frozen_score_fingerprint(
            sample, base_digest=self.base_digest
        ):
            # 指纹不匹配 = 换了模型快照 / 编码路径 / 语料,旧数已经不是这条样本的
            # 分数。当未命中处理并重算,**绝不**静默返回它。
            self.fingerprint_rejects += 1
            self.misses += 1
            return None
        self.hits += 1
        return value

    def store(self, sample: dict[str, Any], value: float) -> None:
        entry = (
            frozen_score_fingerprint(sample, base_digest=self.base_digest),
            float(value),
        )
        key = str(sample["sample_id"])
        self._merged[key] = entry
        self._own[key] = entry
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Atomically rewrite this rank's shard (temp file + rename)."""
        if not self._pending:
            return
        # note (luojiaxuan): json.dumps 的 float 用 repr 打印,是能逐位还原的最短
        # 表示,所以 store→flush→_load 的往返是**精确**的,命中值与重算值逐位相同。
        payload = "".join(
            json.dumps(
                {"sample_id": key, "fingerprint": fingerprint, "score": score},
                sort_keys=True,
            )
            + "\n"
            for key, (fingerprint, score) in sorted(self._own.items())
        )
        temporary = self._shard.parent / f"{self._shard.name}.tmp-{os.getpid()}"
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, self._shard)
        self._pending = 0

    def stats(self) -> dict[str, Any]:
        return {
            "frozen_score_cache": str(self.root),
            "rank": self.rank,
            "hits": self.hits,
            "misses": self.misses,
            "fingerprint_rejects": self.fingerprint_rejects,
            "entries": len(self._own),
        }


# ---------------------------------------------------------------------------
# 组内编码记忆化(默认关闭)
# ---------------------------------------------------------------------------
# note (luojiaxuan): 两遍法里同一条样本会被前向多次,但 forward 闭包每次都重新
# encode_sample 一遍:K≥2 组 12 次前向只对应 **5 条不同样本**(SA×2、每个负样本×3、
# R0×1),即 7 次编码是纯重复。而 encode_sample 不便宜——它对 K+1 张图逐张
# ``Image.open().convert("RGB")``,再跑一整趟 apply_chat_template(resize/patchify/
# tokenize),全部在 CPU 上同步执行,GPU 在这段时间是空的。这比 straggler 更能解释
# 实测的 46% 利用率。
#
# 复用**必须**处理一处陷阱:``mean_target_logprob`` 的第一行是
# ``labels = encoded.pop("labels")``——它就地**改掉了传进去的 dict**(删掉 labels
# 这个键),因为后面要 ``model(**encoded)``,labels 不能混进模型 kwargs。于是把同一
# 个 dict 对象交给第二次 forward,``adapter_context_for_sample`` 读 encoded["labels"]
# 会直接 KeyError。所以记忆化交出去的永远是**浅拷贝** ``dict(entry)``:pop 只作用在
# 这一次的副本上,底层张量仍是同一批(共享引用,不额外占显存)。
#
# 张量层面的复用安全性是**查过的**,不是假设的(证据见 docs/交付说明):
#   * mean_target_logprob 只做 labels[:, 1:] 切片、!= 比较、model(**encoded)、
#     log_softmax/gather,没有任何对入参张量的 in-place 写;
#   * history_sample_context / build_history_token_mask 只读 input_ids /
#     mm_token_type_ids / image_grid_thw / labels,mask 是 torch.zeros_like 新建的
#     张量,``mask[0, s:e] = True`` 写的是那个新张量,不是 encoded 里的任何一个;
#   * assert_mask_disjoint 只读 mask;
#   * HistoryGatedKVLinear._hook 作用在 k/v 投影的**激活**上,返回 output + delta*gate
#     这个新张量,从不碰输入 dict。
#   * 全仓 grep 尾部下划线 in-place 算子,除 LoRA 参数 copy_ 外无命中。
# 剩下唯一无法静态证明的是 HF 模型 forward 内部是否改写入参张量(本地没有 torch,
# 读不到那份实现)。标准 HF 前向不这么做——否则梯度累积、eval 循环、beam search
# 复用同一批输入全都会坏掉——但这属于"有充分理由相信"而非"已证明",所以这个开关
# **默认 off**,启用前建议先在真卡上做一次 A/B(同 seed 跑二十组比损失曲线)。
#
# 显存代价见 --encode-cache-scope 的帮助文本:K=4 时同时持有 5 份 encoding 的
# pixel_values,峰值约 +1 GB(80 GB 卡的 ~1.2%)。因为本路径**故意关掉了梯度检查点**
# (重算跑在 autograd 线程上会让 adapter 的 ContextVar 读空),带梯度前向本身的激活
# 峰值已经很高,这 1 GB 不是白捡的,所以默认关闭由调用方显式权衡。
ENCODE_CACHE_SCOPES = ("off", "group")


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
    objective_kind: str,
    diagnostics_out: dict[str, float] | None = None,
    frozen_cache: FrozenBypassScoreCache | None = None,
    encode_cache_scope: str = "off",
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

    if encode_cache_scope not in ENCODE_CACHE_SCOPES:
        raise ValueError(
            f"unknown encode_cache_scope {encode_cache_scope!r}; "
            f"expected one of {ENCODE_CACHE_SCOPES}"
        )
    # note (luojiaxuan): 记忆化的生命周期**就是这一组**。本函数每个 pair-group 被调用
    # 一次,这个 dict 是闭包的局部状态,函数返回即失去最后一个引用被回收,因此不存在
    # 跨组复用(组间 arm_slot 会重名,跨组复用等于拿另一条决策的 encoding 去打分),
    # 也不会把 5 份 pixel_values 一直压在显存里。key 用 arm_slot:它在组内唯一,而
    # sample_id = f"{pair_group}|{arm_slot}",组内 pair_group 恒定,两者等价。
    encode_cache: dict[str, dict[str, Any]] = {}

    def encode_for_slot(slot: str, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Encode one arm, reusing this group's encoding when memoization is on."""
        if encode_cache_scope != "group":
            return encode_sample(
                runtime, sample, dataset_root=dataset_root, torch=torch
            )
        entry = encode_cache.get(slot)
        if entry is None:
            entry = encode_sample(
                runtime, sample, dataset_root=dataset_root, torch=torch
            )
            if entry is None:
                # 编码失败不进缓存:该臂每次都照旧返回 None,"哪些负样本因取不到值
                # 而被剔出归一化分母"的判定与关缓存时逐个相同。
                return None
            encode_cache[slot] = entry
        # note (luojiaxuan): **必须**是浅拷贝。mean_target_logprob 的第一行
        # ``encoded.pop("labels")`` 会就地删掉这个键(labels 不能进 model(**encoded)),
        # 交出同一个 dict 对象的话,同一条样本的第二次 forward 在
        # adapter_context_for_sample 里读 encoded["labels"] 就是 KeyError。
        # 浅拷贝让 pop 只作用于本次副本,张量本身共享引用,不产生额外显存。
        return dict(entry)

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
        # note (luojiaxuan): 只有"不带梯度 **且** 有效 adapter 模式是 bypass"的前向
        # 才可缓存 —— 那是跑在冻结策略上的分数,与 adapter 参数无关因而全程恒定。
        # 覆盖已被上面限死为 no-grad + bypass,所以这里命中的正好是 3 个 per-negative
        # 锚点 ℓn_bypass,外加参考臂 R0(样本字段本就写着 bypass);did_ra_aware 下还多
        # 一条 S0 —— A_c 的基准端同样是冻结量,四类都吃同一份缓存。
        # 命中直接返回缓存值(与重算逐位相同),连 encode 都省掉。
        cacheable = (
            frozen_cache is not None
            and not grad
            and effective_adapter_mode(sample, adapter_mode) == "bypass"
        )
        if cacheable:
            cached = frozen_cache.lookup(sample)
            if cached is not None:
                return cached
        # 两项缓存正交:冻结分数缓存命中就连编码都不用做;没命中才走编码记忆化。
        encoded = encode_for_slot(slot, sample)
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
                value = float(
                    mean_target_logprob(runtime.model, encoded, torch=torch).detach()
                )
            # note (luojiaxuan): 只写非 None 的值。encode 失败的样本这里根本走不到,
            # 于是它永远进不了缓存、每次都照旧返回 None —— 缓存开与不开,"哪些负样本
            # 因取不到值而被剔出归一化分母"的判定完全一致。
            if cacheable:
                frozen_cache.store(sample, value)
            return value

    return _sparse_history_group_loss(
        samples=samples,
        group=group,
        forward=forward,
        training=training,
        adapter_parameters=adapter_parameters,
        accumulation=accumulation,
        torch=torch,
        objective_kind=objective_kind,
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
SPARSE_TRAINING_KEYS_COMMON = (
    "sparse_history", "epochs", "max_optimizer_steps", "learning_rate",
    "lr_schedule", "weight_decay", "micro_batch_size",
    "gradient_accumulation_steps", "max_grad_norm", "seed",
    "gradient_checkpointing", "checkpoint_every_steps",
    "history_lora_l2_weight", "history_ce_weight",
)
# note (luojiaxuan): 目标相关的超参按 objective.kind 分组,**不取并集**。取并集的话
# did_ra_aware 的 config 得照抄一份从不被读的 sparse_rank_margin / sparse_drift_weight,
# 而"写了但没生效的参数"正是 P1-6 要挡的东西;分组之后,给 did 目标写 rank 超参会以
# "unconsumed keys"报错,给 legacy 目标漏写 rank 超参会以"misses required keys"报错。
SPARSE_TRAINING_KEYS_BY_OBJECTIVE: dict[str, tuple[str, ...]] = {
    SPARSE_OBJECTIVE_DID_RA_AWARE: (
        "sparse_select_margin", "sparse_gain_margin", "sparse_content_margin",
        "sparse_select_weight", "sparse_gain_weight", "sparse_content_weight",
        "sparse_drift_cap_eps", "sparse_drift_cap_weight",
    ),
    SPARSE_OBJECTIVE_LEGACY: (
        "sparse_gain_margin", "sparse_rank_margin", "sparse_gain_weight",
        "sparse_rank_weight", "sparse_drift_weight",
    ),
}


def sparse_training_keys(objective_kind: str) -> tuple[str, ...]:
    """Exactly the ``config.training`` keys one objective consumes."""
    per_objective = SPARSE_TRAINING_KEYS_BY_OBJECTIVE.get(objective_kind)
    if per_objective is None:
        raise ValueError(
            f"unknown objective kind {objective_kind!r}; expected one of "
            f"{SPARSE_OBJECTIVE_KINDS}"
        )
    return SPARSE_TRAINING_KEYS_COMMON + per_objective


SPARSE_BUILD_TIME_ONLY_KEYS = ("heldout_episode_fraction", "heldout_hash_salt")
SPARSE_GATE_SCHEMA = "causalcache.sparse_history_gates.v2"
SPARSE_DERIVED_QUANTITIES = {
    "format_effect": "R0 - N0",
    "frozen_selection_effect": "S0 - R0",
    "adapter_on_recent": "RA - R0",
    "adapter_on_recent_abs": "abs(RA - R0)",
    "adapter_on_sparse": "SA - S0",
    # note (luojiaxuan): did_select 就是 did_ra_aware 直接优化的那个量 —— adapter 对
    # sparse 的增量减去它对 recent 的增量。它与主 claim SA−RA 的差别是把冻结模型自带的
    # 选点优势 S0−R0 也扣掉,因此 identity 上恒为 0(SA−RA 在 identity 上是 +0.0335)。
    # 只加进派生量词表供训练诊断与留出集报告使用,**没有**进 gates.must_pass:本次只换
    # 训练目标,已冻结的验收标准(gates schema v2)一字不动。
    "did_select": "(SA - S0) - (RA - R0)",
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
    key
    for schema, kinds in SPARSE_NEGATIVE_KINDS_BY_SCHEMA.items()
    for kind in kinds
    for key in sparse_diagnostic_keys(
        kind, arm_slot=SPARSE_NEGATIVE_SLOTS_BY_SCHEMA[schema][kind]
    )
)
# note (luojiaxuan): 桌面 schema 的派生量表。没有 N0:format_effect 结构性为 0 不
# 报告;deployment_delta 的基线就是 R0(与 SA_minus_R0 重合,保留两个名字是让
# 部署语义的读数不用换名字找)。其余代数式与 v2 逐字相同。
SPARSE_DESKTOP_DERIVED_QUANTITIES = {
    "frozen_selection_effect": "S0 - R0",
    "adapter_on_recent": "RA - R0",
    "adapter_on_recent_abs": "abs(RA - R0)",
    "adapter_on_sparse": "SA - S0",
    "did_select": "(SA - S0) - (RA - R0)",
    "SA_minus_RA": "SA - RA",
    "SA_minus_R0": "SA - R0",
    "deployment_delta": "SA - R0",
}
SPARSE_DERIVED_QUANTITIES_BY_SCHEMA: dict[str, dict[str, str]] = {
    SPARSE_SAMPLE_SCHEMA: SPARSE_DERIVED_QUANTITIES,
    SPARSE_REPLACEMENT_SAMPLE_SCHEMA: SPARSE_DERIVED_QUANTITIES,
    SPARSE_DESKTOP_SAMPLE_SCHEMA: SPARSE_DESKTOP_DERIVED_QUANTITIES,
}
# 词表只决定"允许引用哪些名字",放宽是安全的(见上方 v6 注释);按 schema 的
# 必需/外来 gate 检查才是挡错的那一层。
SPARSE_GATE_VOCABULARY = frozenset(
    {
        name
        for quantities in SPARSE_DERIVED_QUANTITIES_BY_SCHEMA.values()
        for name in quantities
    }
    | SPARSE_NEGATIVE_GATE_KEYS
)
# note (luojiaxuan): 2026-07-25 随目标一并换成 RA-aware 判据。旧集合要求
# ``SA_minus_R0``,而新目标**根本不优化它** —— 它恰是会被"见历史就放大"刷高的量
# (v5 实测它从 +0.034 涨到 +0.118,而真实的 SA_minus_RA 归零)。把它留在必需 gate 里
# 会让新 checkpoint 因一个它没在优化的指标判 FAIL,更糟的是 composite 会把选点拉向
# 放大最严重的那个。
#
# ``SA_minus_RA`` 也不单列:预注册判据里的 SA-RA >= S0-R0 展开就是 did_select >= 0,
# 与主判据重复,did_select > 0 @ci_low 是它的严格版本。
#
# 三个负样本 margin 退出必需集但**仍在 derived_quantities 里报告** —— 用户预注册的
# PASS 条件是 did_select / A_c / |A_r| / 三个 drift,不含它们;内容敏感性由
# L_content 在训练中优化,验收看 drift 是否被封住即可。
#
# note (luojiaxuan): 必需集**按语料 schema 索引**,不是一个全局常量。理由:每个 schema
# 的负样本 kind 不同(v5 三个、v6 一个 age_matched),而"必须声明的 drift gate"正是由
# 语料里真实存在的 kind 决定的。做成单一常量会二选一地犯错——要么逼 v5 config 声明
# 语料里不存在的 age_matched_drift_abs(旧 run 全部不可复现),要么让 v6 run 的唯一
# 负样本没有必需 gate(内容对照臂被推动多少无人验收)。
# ``SPARSE_REQUIRED_GATES`` 保留为 v2 那一份的别名:它的**取值逐字不变**,既有测试与
# 外部引用者不受影响。
_SPARSE_BASE_REQUIRED_GATES = frozenset(
    {"did_select", "adapter_on_sparse", "adapter_on_recent_abs"}
)
SPARSE_REQUIRED_GATES_BY_SCHEMA: dict[str, frozenset[str]] = {
    schema: frozenset(
        _SPARSE_BASE_REQUIRED_GATES
        | {
            sparse_diagnostic_keys(
                kind, arm_slot=SPARSE_NEGATIVE_SLOTS_BY_SCHEMA[schema][kind]
            )[1]
            for kind in kinds
        }
    )
    for schema, kinds in SPARSE_NEGATIVE_KINDS_BY_SCHEMA.items()
}
SPARSE_REQUIRED_GATES = SPARSE_REQUIRED_GATES_BY_SCHEMA[SPARSE_SAMPLE_SCHEMA]
# 某个 schema 的语料里**不可能产生**的负样本量名 —— 声明了它们的 config 会在打分阶段
# 拿到 None(GateQuantityUnavailable),这里提前到训练启动前拦下。
SPARSE_NEGATIVE_GATE_KEYS_BY_SCHEMA: dict[str, frozenset[str]] = {
    schema: frozenset(
        key
        for kind in kinds
        for key in sparse_diagnostic_keys(
            kind, arm_slot=SPARSE_NEGATIVE_SLOTS_BY_SCHEMA[schema][kind]
        )
    )
    for schema, kinds in SPARSE_NEGATIVE_KINDS_BY_SCHEMA.items()
}


def sparse_config_sample_schema(config: dict[str, Any]) -> str:
    """Which corpus schema this config's gates and arms are written against.

    # note (luojiaxuan): 缺 ``data.sample_schema_version`` 时按 v2 处理 —— 既有 config
    # 一律没有这个键,默认值必须让它们的行为逐字不变。写了就必须是已知 schema,
    # 拼错不给默认值(拼错会静默套用 v2 的必需 gate,而那正是这次要消灭的分叉)。
    """
    data = config.get("data")
    schema = data.get("sample_schema_version") if isinstance(data, dict) else None
    if schema is None:
        return SPARSE_SAMPLE_SCHEMA
    if schema not in SPARSE_SAMPLE_SCHEMAS:
        raise ValueError(
            f"data.sample_schema_version {schema!r} is not one of "
            f"{list(SPARSE_SAMPLE_SCHEMAS)}"
        )
    return str(schema)

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
    training: dict[str, Any], *, args: argparse.Namespace, objective_kind: str
) -> dict[str, Any]:
    """Return the run controls the sparse branch takes from config alone."""
    consumed = sparse_training_keys(objective_kind)
    stale = [key for key in SPARSE_BUILD_TIME_ONLY_KEYS if key in training]
    if stale:
        raise ValueError(
            f"training block carries build-time-only keys {stale}; move them under "
            "data.split — the trainer reads each sample's split field"
        )
    unknown = sorted(set(training) - set(consumed))
    if unknown:
        raise ValueError(
            f"training block has unconsumed keys {unknown} under objective "
            f"{objective_kind!r}"
        )
    missing = sorted(set(consumed) - set(training))
    if missing:
        raise ValueError(
            f"training block misses required keys {missing} under objective "
            f"{objective_kind!r}"
        )
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


# note (luojiaxuan): objective 段过去纯粹是散文,一个字都没被读过 —— 于是"config 写着
# RA 不进损失"与"代码到底读不读 RA"可以无声分叉,而这次的假阳性正是这类分叉的后果。
# 现在这一段与 training 段同规格:key 集合按 kind 精确匹配(多一个报 unconsumed、少一个
# 报 misses),excluded_arms 必须是**列表**且与 SPARSE_OBJECTIVE_ARMS 的补集逐项相等。
# 后者是结构性的:did_ra_aware 一旦真的开始给 RA 打分,excluded_arms 里还留着 "RA" 就
# 会当场报错,不需要谁记得去改那句散文。
SPARSE_OBJECTIVE_DOC_KEYS: dict[str, frozenset[str]] = {
    SPARSE_OBJECTIVE_DID_RA_AWARE: frozenset(
        {
            "kind", "notation", "A_c", "A_r", "A_n",
            "L_select", "L_gain", "L_content", "L_cap", "L_l2", "L",
            "did_rationale", "gain_rationale", "drift_cap_rationale",
            "normalization_rationale", "excluded_arms", "excluded_arms_note",
        }
    ),
    SPARSE_OBJECTIVE_LEGACY: frozenset(
        {
            "kind", "deprecated", "notation",
            "L_gain", "L_rank", "L_drift", "L_l2", "L",
            "drift_rationale", "normalization_rationale",
            "excluded_arms", "excluded_arms_note",
        }
    ),
}


def validate_sparse_objective(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Validate the objective block and return ``(kind, block)``."""
    objective = config.get("objective")
    if not isinstance(objective, dict):
        raise ValueError("sparse_history config must carry an objective block")
    kind = objective.get("kind")
    if kind not in SPARSE_OBJECTIVE_KINDS:
        raise ValueError(
            f"objective.kind must be one of {SPARSE_OBJECTIVE_KINDS}, got {kind!r}"
        )
    expected_keys = SPARSE_OBJECTIVE_DOC_KEYS[kind]
    unknown = sorted(set(objective) - expected_keys)
    if unknown:
        raise ValueError(
            f"objective block has unconsumed keys {unknown} under kind {kind!r}"
        )
    absent = sorted(expected_keys - set(objective))
    if absent:
        raise ValueError(
            f"objective block misses required keys {absent} under kind {kind!r}"
        )
    excluded = objective["excluded_arms"]
    expected_excluded = sparse_objective_excluded_arms(
        kind, sample_schema=sparse_config_sample_schema(config)
    )
    if excluded != expected_excluded:
        raise ValueError(
            f"objective.excluded_arms {excluded!r} disagrees with the arms "
            f"{kind!r} actually scores; expected {expected_excluded!r}"
        )
    return kind, objective


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
    # 派生量代数与必需 gate 都随语料 schema 变(桌面 schema 没有 N0/format_effect)。
    schema = sparse_config_sample_schema(config)
    declared = gates.get("derived_quantities")
    if declared != SPARSE_DERIVED_QUANTITIES_BY_SCHEMA[schema]:
        raise ValueError(
            "gates.derived_quantities drifted from the frozen arm algebra "
            f"{SPARSE_DERIVED_QUANTITIES_BY_SCHEMA[schema]} for schema {schema}"
        )
    must_pass = gates.get("must_pass")
    if not isinstance(must_pass, dict) or not must_pass:
        raise ValueError("gates.must_pass must be a non-empty object")
    unknown = sorted(set(must_pass) - SPARSE_GATE_VOCABULARY)
    if unknown:
        raise ValueError(f"gates.must_pass references unknown quantities {unknown}")
    absent = sorted(SPARSE_REQUIRED_GATES_BY_SCHEMA[schema] - set(must_pass))
    if absent:
        raise ValueError(
            f"gates.must_pass misses required gates {absent} for corpus schema "
            f"{schema!r}"
        )
    foreign = sorted(
        set(must_pass)
        & (
            frozenset().union(*SPARSE_NEGATIVE_GATE_KEYS_BY_SCHEMA.values())
            - SPARSE_NEGATIVE_GATE_KEYS_BY_SCHEMA[schema]
        )
    )
    if foreign:
        raise ValueError(
            f"gates.must_pass declares {foreign}, but corpus schema {schema!r} never "
            f"produces those negatives (its kinds are "
            f"{list(SPARSE_NEGATIVE_KINDS_BY_SCHEMA[schema])}); the scorer would hand "
            "the gate a None and fail only after the run"
        )
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
    for quantity, algebra in SPARSE_DERIVED_QUANTITIES_BY_SCHEMA[schema].items():
        # 派生量的代数式也必须可执行:打分脚本按这些字符串在该 schema 的测量臂上求值。
        validate_gate_expression(
            algebra,
            allowed=frozenset(SPARSE_HELDOUT_REQUIRED_SLOTS_BY_SCHEMA[schema]),
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
        "--frozen-score-cache",
        type=Path,
        default=None,
        help=(
            "sparse_history only; directory caching adapter-independent bypass "
            "scores across runs. Default off — one epoch sees each group once, so "
            "the payoff is multi-epoch / reruns / hyper-parameter sweeps."
        ),
    )
    parser.add_argument(
        "--encode-cache-scope",
        choices=ENCODE_CACHE_SCOPES,
        default="off",
        help=(
            "sparse_history only; 'group' reuses one encoding per arm inside a "
            "pair-group, cutting encode_sample calls from 12 to 5 at K>=2 "
            "(6 to 4 at K=1). Default off: it raises peak GPU memory by roughly "
            "1 GB at K=4 (five live pixel_values sets, ~48 MB per image at 2560 "
            "effective visual tokens), and this path deliberately runs without "
            "gradient checkpointing, so activation memory is already high."
        ),
    )
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
    sparse_objective: dict[str, Any] | None = None
    sparse_objective_kind = ""
    if sparse_mode:
        if adapter_settings(config)[0] != "history_gated_kv":
            raise ValueError("sparse_history requires adapter_type history_gated_kv")
        sparse_gates = validate_sparse_gates(config)
        sparse_objective_kind, sparse_objective = validate_sparse_objective(config)
        sparse_controls = resolve_sparse_training(
            config["training"], args=args, objective_kind=sparse_objective_kind
        )
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

    # 桌面语料的 dev→heldout / test 剥离必须发生在任何分组与校验之前;旧语料原样通过。
    samples, desktop_split_counters = normalize_desktop_splits(samples)
    if not samples:
        raise ValueError("normalize_desktop_splits dropped every sample")
    if rank == 0 and any(desktop_split_counters.values()):
        print(
            json.dumps(
                {"desktop_split_normalization": desktop_split_counters},
                sort_keys=True,
            ),
            flush=True,
        )

    # note (luojiaxuan): 标签类过滤放在**加载模型之前**。config 与语料不匹配(声明了
    # train_on_label_classes 而语料没有 label_class 字段)应该几秒内报错,而不是等 8B
    # 权重载完、几分钟之后才炸。排除是有记录的:这一行 JSON 就是"哪些组没进训练"的
    # 唯一凭据,缺了它就退化成"组数怎么变少了"那种查不出来的静默过滤。
    label_classes = resolve_train_on_label_classes(config)
    if label_classes is not None:
        label_selection: dict[str, Any] = {}
        samples = filter_samples_by_label_class(
            samples, allowed=label_classes, selection_out=label_selection
        )
        if rank == 0:
            print(
                json.dumps({"label_class_selection": label_selection}, sort_keys=True),
                flush=True,
            )

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
    # note (luojiaxuan): 冻结分数缓存默认关闭 —— 不给 --frozen-score-cache 时
    # frozen_cache 恒为 None,forward 闭包里那段 cacheable 判定短路,行为与旧版逐字节
    # 相同。只在 sparse 分支支持:旧两条路径的臂语义里没有"有效 bypass"这个概念。
    # 与 --frozen-score-cache 正交:一个省"冻结分数的前向",一个省"重复的编码",
    # 两者可各自独立开关,同开时先查分数缓存(命中就连编码都不用做)。
    if args.encode_cache_scope != "off" and not sparse_mode:
        raise ValueError(
            "--encode-cache-scope is only defined under training.sparse_history"
        )
    frozen_cache: FrozenBypassScoreCache | None = None
    if args.frozen_score_cache is not None:
        if not sparse_mode:
            raise ValueError(
                "--frozen-score-cache is only defined under training.sparse_history"
            )
        snapshot_path = args.repository_root / config["policy_snapshot_manifest"]
        frozen_cache = FrozenBypassScoreCache(
            args.frozen_score_cache,
            base_digest=frozen_score_cache_base_digest(
                policy_snapshot_sha256=hashlib.sha256(
                    snapshot_path.read_bytes()
                ).hexdigest(),
                model_dir=args.model_dir,
                dataset_manifest_sha256=manifest_sha,
                visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
            ),
            rank=rank,
        )
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
            # note (luojiaxuan): v2 → v3,因为 manifest 新增了 objective_kind /
            # objective,而 effective_training 消费的 key 集合从此**取决于**目标。
            # 不升版本的话,一份 v2 reader 读到没有 objective_kind 的记录只能默认它是
            # 旧目标 —— 这正是本次要修的那类静默假设。目前没有任何脚本消费这份
            # manifest(全仓 grep 只有 trainer 自己在写),升版本无下游影响。
            run_manifest["schema_version"] = "causalcache.success_sft_lora_run.v3"
            run_manifest["sample_schema_version"] = SPARSE_SAMPLE_SCHEMA
            run_manifest["cli_argv"] = list(sys.argv)
            run_manifest["cli_args"] = {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in sorted(vars(args).items())
            }
            run_manifest["config_path"] = str(args.config)
            run_manifest["config_sha256"] = config_sha
            run_manifest["hyperparameter_authority"] = "config.training only"
            run_manifest["objective_kind"] = sparse_objective_kind
            run_manifest["objective"] = sparse_objective
            run_manifest["effective_training"] = {
                key: config["training"][key]
                for key in sorted(sparse_training_keys(sparse_objective_kind))
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
        units, heldout_episodes = build_sparse_history_units(
            samples, objective_kind=sparse_objective_kind
        )
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
            # note (luojiaxuan): 这里原本写死 SPARSE_SAMPLE_SCHEMA,于是 v6 run 的启动
            # 行会自称跑在 v2 语料上 —— 一个"声明了但与实际不符"的字段比没有更糟。
            startup["sample_schema_version"] = sparse_config_sample_schema(config)
            if label_classes is not None:
                startup["train_on_label_classes"] = list(label_classes)
            startup["reference_arm_id"] = "R0"
            startup["main_claim_quantity"] = SPARSE_MAIN_CLAIM_QUANTITY
            startup["objective_kind"] = sparse_objective_kind
            startup["config_sha256"] = config_sha
            startup["max_optimizer_steps"] = max_steps
            startup["checkpoint_every_steps"] = checkpoint_every_steps
        # note (luojiaxuan): 键序保持插入序,旧两条路径的这行 stdout 逐字节不变。
        print(json.dumps(startup), flush=True)
    # note (luojiaxuan): 调度代价只对 sparse 分支有定义(它的组才有负样本臂与 budget),
    # 预先算好一份 unit_index -> (前向次数, 预算),每个 epoch 复用同一份。
    schedule_costs: dict[int, tuple[int, int]] = (
        {
            unit_index: sparse_group_schedule_cost(
                samples, payload, objective_kind=sparse_objective_kind
            )
            for unit_index, (_kind, payload, _negative) in enumerate(units)
        }
        if sparse_mode
        else {}
    )
    ordering = random.Random(training_config["seed"])
    global_step = 0
    for epoch in range(args.start_epoch, training_config["epochs"]):
        order = list(range(len(units)))
        ordering.shuffle(order)
        if sparse_mode:
            # note (luojiaxuan): 按 K 分桶消除 straggler。只在 accumulation window
            # **内部**重排"哪个 rank 跑哪一组",每个优化步消费的组集合与旧 stride
            # 切分逐组相同(见 balanced_sparse_shards 的不变量断言),因此损失与梯度
            # 的数学定义一字未动。旧两条路径继续走原来的 stride,逐字节不受影响。
            shard = balanced_sparse_shards(
                order,
                schedule_costs,
                world_size=world_size,
                accumulation=accumulation,
            )[rank]
        else:
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
                    objective_kind=sparse_objective_kind,
                    diagnostics_out=diagnostics,
                    frozen_cache=frozen_cache,
                    encode_cache_scope=args.encode_cache_scope,
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
        if frozen_cache is not None:
            # 每个 epoch 收尾落一次盘,别把整轮的命中攒到进程退出才写。
            frozen_cache.flush()
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
    if frozen_cache is not None:
        # note (luojiaxuan): 命中率按 rank 打印(缓存分片也是按 rank 的),这行只在
        # 显式给了 --frozen-score-cache 时才出现,默认关闭的 run 的 stdout 不变。
        # 第一遍扫语料时 hits 本来就该是 0 —— 收益在第二遍及以后,别当成没生效。
        frozen_cache.flush()
        print(json.dumps(frozen_cache.stats(), sort_keys=True), flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({"training_complete": True, "global_steps": global_step}))


if __name__ == "__main__":
    main()
