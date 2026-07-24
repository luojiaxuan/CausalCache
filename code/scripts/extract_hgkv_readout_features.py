#!/usr/bin/env python3
"""HGKV counterfactual-readout feature extractor for singleton selector samples.

# note (luojiaxuan): selector V2 协议(docs/selector_v1_protocol.md V2 节)Stage-1/
# Stage-2 的特征骨干。对 singleton 渲染集(1 张恢复历史图 + 当前观测)的每个样本,
# 在 hg-s100 适配的末 8 个 LM 层上计算逐层反事实 readout
#   Δr_l = Attn(Q_l, K0_l+ΔK_l)(V0_l+ΔV_l) − Attn(Q_l, K0_l)V0_l,
# 其中 Q_l 取当前决策 readout 位置(prompt 最后一个 token,即官方 assistant 生成
# 前缀 "<|im_start|>assistant\n" 的末 token——它的 hidden state 预测第一个动作
# token,是决策读出态;目标动作跨多 token 的后续 query 位置不入特征,属已记录的
# 简化)、K0/V0 取候选历史图 token 的冻结 k/v_proj 输出、ΔK/ΔV 由 hg-s100 wrapper
# 的 lora 权重手工算出 (alpha/rank)·B·A·h。
#
# 关键不变量:整个前向绝不安装 history adapter context(ContextVar 缺省 None ⇒
# HistoryGatedKVLinear hook 完全 bypass),K0/V0 保持冻结干净;Δ 只离线由 wrapper
# 的 lora_a/lora_b 计算,不改任何模型状态。所有量取自投影输出层(pre-RoPE、
# pre-q/k-norm)——正是 adapter 注入的位置;不复现 M-RoPE/qk-norm 后的真实注意力
# 几何,两个对照项共享同一简化,差分只隔离 adapter 效应(已记录的简化)。
#
# 特征维度:逐层 [mean_heads(Δr_l) (head_dim) ‖ per-head ‖Δr_l‖₂ (n_q_heads)],
# 层按 index 升序拼接 ⇒ feature_dim = L × (head_dim + n_q_heads);GUI-Owl-1.5-8B
# (Qwen3-VL 系,head_dim=128、32 query heads、末 8 层)= 8 × 160 = 1280。标量
# 摘要每层:‖Δr‖_F、cos(r0, rHG)、全 prompt softmax 下历史 token 注意力质量
# (base 与 adapted 两侧)。断点续跑幂等键 = (pair_group, restored 事件 id)。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SINGLETON_VARIANT_PREFIX = "single"

_LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")


@dataclass(frozen=True)
class LayerHandles:
    """One adapted LM layer: frozen q_proj plus wrapped k/v projections."""

    layer_index: int
    q_proj: torch.nn.Module
    k_wrap: Any
    v_wrap: Any


def build_layer_handles(model: Any, wrapped: dict[str, Any]) -> list[LayerHandles]:
    """Group HistoryGatedKVLinear wrappers by layer and attach each q_proj."""
    per_layer: dict[int, dict[str, Any]] = {}
    for name, wrap in wrapped.items():
        leaf = name.split(".")[-1]
        match = _LAYER_INDEX_PATTERN.search(f".{name}.")
        if match is None:
            raise RuntimeError(f"wrapped module {name} lacks a parseable layer index")
        entry = per_layer.setdefault(int(match.group(1)), {})
        if leaf in entry:
            raise RuntimeError(f"duplicate {leaf} wrapper for layer {match.group(1)}")
        entry[leaf] = wrap
        if leaf == "k_proj":
            entry["q_proj"] = model.get_submodule(name[: -len("k_proj")] + "q_proj")
    handles: list[LayerHandles] = []
    for index in sorted(per_layer):
        entry = per_layer[index]
        if set(entry) != {"q_proj", "k_proj", "v_proj"}:
            raise RuntimeError(
                f"layer {index} exposes {sorted(entry)}, expected q/k/v projections"
            )
        handles.append(
            LayerHandles(
                layer_index=index,
                q_proj=entry["q_proj"],
                k_wrap=entry["k_proj"],
                v_wrap=entry["v_proj"],
            )
        )
    if not handles:
        raise RuntimeError("no adapted layers found for readout extraction")
    return handles


def lora_delta(wrap: Any, hidden: torch.Tensor) -> torch.Tensor:
    """(alpha/rank)·B·A·h — bit-identical math to HistoryGatedKVLinear._hook."""
    return (
        hidden.to(wrap.lora_a.dtype) @ wrap.lora_a.T @ wrap.lora_b.T
    ) * wrap.scaling


def capture_layer_states(
    model: Any,
    encoded: dict[str, Any],
    *,
    layers: list[LayerHandles],
    hist_idx: torch.Tensor,
    query_pos: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """One clean forward; per adapted layer capture the attention-input hidden
    states (query position + history positions) and frozen K/V slices.

    # note (luojiaxuan): hook 挂在 k_proj/v_proj 上——三个投影吃同一个
    # post-layernorm hidden,所以 k_proj 的 inputs[0] 即 q_proj 的输入;K 另存
    # 0..query_pos 全前缀(k_prompt)供全 prompt 注意力质量;所有捕获立即切片
    # 并转 fp32,避免整层驻留。前向前断言 adapter context 未安装,保证 K0/V0
    # 是冻结输出(wrapper hook bypass)。
    """
    from causalcache.policy.history_adapter_context import (
        get_history_adapter_context,
    )

    if get_history_adapter_context() is not None:
        raise RuntimeError(
            "history adapter context must not be installed during the clean forward"
        )
    captures: dict[int, dict[str, torch.Tensor]] = {
        layer.layer_index: {} for layer in layers
    }

    def _flat(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            if int(tensor.shape[0]) != 1:
                raise ValueError("readout capture requires batch size 1")
            return tensor[0]
        if tensor.ndim == 2:
            return tensor
        raise ValueError(f"unexpected projection tensor rank {tensor.ndim}")

    def make_k_hook(layer_index: int):
        def hook(module: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            hidden = _flat(inputs[0])
            out = _flat(output)
            slot = captures[layer_index]
            slot["hidden_query"] = hidden[query_pos].detach().float()
            slot["hidden_hist"] = hidden[hist_idx].detach().float()
            slot["k_hist"] = out[hist_idx].detach().float()
            slot["k_prompt"] = out[: query_pos + 1].detach().float()

        return hook

    def make_v_hook(layer_index: int):
        def hook(module: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            captures[layer_index]["v_hist"] = _flat(output)[hist_idx].detach().float()

        return hook

    handles = []
    try:
        for layer in layers:
            handles.append(
                layer.k_wrap.module.register_forward_hook(
                    make_k_hook(layer.layer_index)
                )
            )
            handles.append(
                layer.v_wrap.module.register_forward_hook(
                    make_v_hook(layer.layer_index)
                )
            )
        with torch.inference_mode():
            try:
                model(**encoded, logits_to_keep=1)
            except TypeError:
                model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
    expected = {"hidden_query", "hidden_hist", "k_hist", "k_prompt", "v_hist"}
    for layer in layers:
        if set(captures[layer.layer_index]) != expected:
            raise RuntimeError(
                f"layer {layer.layer_index} capture incomplete: "
                f"{sorted(captures[layer.layer_index])}"
            )
    return captures


def layer_readout_feature(
    *,
    q: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    k_prompt: torch.Tensor,
    hist_idx: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Counterfactual readout for one layer.

    # note (luojiaxuan): q [q_dim];k0/dk/v0/dv [T, kv_dim];k_prompt
    # [query_pos+1, kv_dim](0..query_pos 全前缀,含历史列);GQA 用
    # repeat_interleave 展开 kv 头对齐 HF repeat_kv 语义。readout 的 softmax
    # 只在历史 token 上(协议定义);hist_mass_* 则在全 prompt 前缀 softmax
    # 下衡量历史列质量,adapted 侧仅替换历史列 logits(ΔK 只作用历史 token,
    # 与 wrapper 门控一致)。返回 (feature [head_dim+n_q_heads], scalars)。
    """
    if q.ndim != 1:
        raise ValueError("q must be a flat [q_dim] tensor")
    n_q, rem = divmod(int(q.numel()), head_dim)
    if rem or n_q <= 0:
        raise ValueError("q_proj width is not a multiple of head_dim")
    n_kv, rem = divmod(int(k0.shape[-1]), head_dim)
    if rem or n_kv <= 0 or n_q % n_kv:
        raise ValueError("k/v width incompatible with head_dim / GQA grouping")
    group = n_q // n_kv
    scale = 1.0 / math.sqrt(head_dim)

    def heads(flat: torch.Tensor) -> torch.Tensor:
        return flat.view(flat.shape[0], n_kv, head_dim).repeat_interleave(
            group, dim=1
        )

    qh = q.view(n_q, head_dim)
    logits0 = torch.einsum("hd,thd->ht", qh, heads(k0)) * scale
    logits1 = torch.einsum("hd,thd->ht", qh, heads(k0 + dk)) * scale
    r0 = torch.einsum("ht,thd->hd", torch.softmax(logits0, dim=-1), heads(v0))
    r1 = torch.einsum(
        "ht,thd->hd", torch.softmax(logits1, dim=-1), heads(v0 + dv)
    )
    dr = r1 - r0
    feature = torch.cat([dr.mean(dim=0), dr.norm(dim=1)])

    base = torch.einsum("hd,phd->hp", qh, heads(k_prompt)) * scale
    adapted = base.clone()
    adapted[:, hist_idx] = logits1
    mass0 = torch.softmax(base, dim=-1)[:, hist_idx].sum(dim=-1).mean()
    mass1 = torch.softmax(adapted, dim=-1)[:, hist_idx].sum(dim=-1).mean()
    cos = torch.nn.functional.cosine_similarity(
        r0.flatten(), r1.flatten(), dim=0, eps=1e-8
    )
    scalars = {
        "dr_norm": float(dr.norm()),
        "cos_r0_rhg": float(cos),
        "hist_mass_base": float(mass0),
        "hist_mass_adapted": float(mass1),
    }
    return feature, scalars


def extract_sample_features(
    model: Any,
    encoded: dict[str, Any],
    *,
    history_mask: torch.Tensor,
    query_pos: int,
    layers: list[LayerHandles],
    head_dim: int,
) -> tuple[list[float], dict[str, dict[str, float]]]:
    """Full per-sample extraction: one clean forward, then per-layer Δr features."""
    if history_mask.ndim != 2 or int(history_mask.shape[0]) != 1:
        raise ValueError("history_mask must have shape [1, seq]")
    hist_idx = history_mask[0].nonzero(as_tuple=False).squeeze(-1)
    if int(hist_idx.numel()) == 0:
        raise ValueError("singleton sample produced an empty history mask")
    if int(hist_idx.max()) >= query_pos:
        raise ValueError("history tokens must precede the decision query position")
    captures = capture_layer_states(
        model, encoded, layers=layers, hist_idx=hist_idx, query_pos=query_pos
    )
    features: list[torch.Tensor] = []
    scalars: dict[str, dict[str, float]] = {}
    for layer in layers:
        slot = captures[layer.layer_index]
        weight_dtype = layer.q_proj.weight.dtype
        with torch.inference_mode():
            q = layer.q_proj(slot["hidden_query"].to(weight_dtype)).float()
            dk = lora_delta(layer.k_wrap, slot["hidden_hist"])
            dv = lora_delta(layer.v_wrap, slot["hidden_hist"])
            feature, layer_scalars = layer_readout_feature(
                q=q,
                k0=slot["k_hist"],
                v0=slot["v_hist"],
                dk=dk,
                dv=dv,
                k_prompt=slot["k_prompt"],
                hist_idx=hist_idx,
                head_dim=head_dim,
            )
        features.append(feature)
        scalars[str(layer.layer_index)] = layer_scalars
    flat = torch.cat(features).cpu()
    return [float(f"{value:.6g}") for value in flat.tolist()], scalars


def is_singleton_row(sample: dict[str, Any]) -> bool:
    return str(sample.get("variant", "")).startswith(SINGLETON_VARIANT_PREFIX)


def singleton_restored_id(sample: dict[str, Any]) -> int:
    """Resolve the restored event id; fail closed on non-singleton geometry."""
    restored = sample["memory_config"]["restored_event_step_ids"]
    if len(restored) != 1:
        raise ValueError(
            f"singleton row {sample.get('pair_group')} restored "
            f"{len(restored)} events, expected exactly 1"
        )
    singleton_id = sample.get("singleton_event_step_id")
    if singleton_id is not None and int(singleton_id) != int(restored[0]):
        raise ValueError(
            f"singleton row {sample.get('pair_group')} id mismatch: "
            f"{singleton_id} vs restored {restored[0]}"
        )
    return int(restored[0])


def sample_resume_key(sample: dict[str, Any]) -> tuple[str, int]:
    return (str(sample["pair_group"]), singleton_restored_id(sample))


def load_done_keys(output: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not output.exists():
        return done
    for line in output.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        pair_group = row.get("pair_group")
        restored_id = row.get("singleton_event_step_id")
        if pair_group is None or restored_id is None:
            continue
        done.add((str(pair_group), int(restored_id)))
    return done


def resolve_head_dim(model: Any) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        head_dim = int(text_config.hidden_size) // int(
            text_config.num_attention_heads
        )
    return int(head_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--adapter-layer-count", type=int, default=8)
    parser.add_argument("--episodes-filter", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    from scripts.run_exploratory_closed_loop_episode import (
        EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from scripts.train_success_sft_lora import (
        encode_sample,
        history_sample_context,
    )
    from causalcache.policy.gui_owl_v2_1_runtime import (
        GUIOwlV21OfficialToolsRuntime,
    )
    from causalcache.policy.history_gated_lora import (
        inject_history_gated_kv,
        load_history_gated_state_dict,
    )

    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    runtime.model.eval()
    merge_size = int(runtime.processor.image_processor.merge_size)
    wrapped = inject_history_gated_kv(
        runtime.model,
        layer_count=args.adapter_layer_count,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    load_history_gated_state_dict(
        wrapped, torch.load(args.lora_checkpoint, map_location="cpu")
    )
    layers = build_layer_handles(runtime.model, wrapped)
    head_dim = resolve_head_dim(runtime.model)
    n_q_heads = int(layers[0].q_proj.weight.shape[0]) // head_dim
    n_kv_heads = int(layers[0].k_wrap.module.weight.shape[0]) // head_dim
    feature_dim = len(layers) * (head_dim + n_q_heads)
    print(
        json.dumps(
            {
                "adapter_type": "history_gated_kv",
                "lora_modules": len(wrapped),
                "layer_indices": [layer.layer_index for layer in layers],
                "head_dim": head_dim,
                "n_q_heads": n_q_heads,
                "n_kv_heads": n_kv_heads,
                "feature_dim": feature_dim,
            }
        ),
        flush=True,
    )

    allowed_episodes = None
    if args.episodes_filter is not None:
        allowed_episodes = set(
            args.episodes_filter.read_text(encoding="utf-8").split()
        )
    samples = [
        json.loads(line)
        for line in (args.dataset_root / "samples.jsonl").open(encoding="utf-8")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_keys = load_done_keys(args.output)
    written = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            if index % args.shard_count != args.shard_index:
                continue
            if (
                allowed_episodes is not None
                and sample["episode"] not in allowed_episodes
            ):
                continue
            if not is_singleton_row(sample):
                continue
            key = sample_resume_key(sample)
            if key in done_keys:
                continue
            encoded = encode_sample(
                runtime, sample, dataset_root=args.dataset_root, torch=torch
            )
            if encoded is None:
                continue
            # note (luojiaxuan): mask 构造要用 labels 定目标段,必须先建
            # context 再 pop labels;context 只取 mask,绝不安装进
            # ContextVar——前向保持冻结 K0/V0。
            context = history_sample_context(encoded, sample, merge_size=merge_size)
            if context is None:
                raise RuntimeError(
                    f"singleton row {key} yielded no history context"
                )
            labels = encoded.pop("labels")
            target_length = int((labels != -100).sum())
            query_pos = int(encoded["input_ids"].shape[1]) - target_length - 1
            feature, scalars = extract_sample_features(
                runtime.model,
                encoded,
                history_mask=context.history_token_mask,
                query_pos=query_pos,
                layers=layers,
                head_dim=head_dim,
            )
            if len(feature) != feature_dim:
                raise RuntimeError(
                    f"feature length {len(feature)} != expected {feature_dim}"
                )
            handle.write(
                json.dumps(
                    {
                        "episode": sample["episode"],
                        "step_index": sample["step_index"],
                        "pair_group": sample["pair_group"],
                        "variant": sample.get("variant"),
                        "singleton_event_step_id": key[1],
                        "restored_event_step_ids": sample["memory_config"][
                            "restored_event_step_ids"
                        ],
                        "layer_indices": [layer.layer_index for layer in layers],
                        "feature_dim": feature_dim,
                        "feature": feature,
                        "scalars": scalars,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            done_keys.add(key)
            written += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "extracted_shard": args.shard_index,
                "of": args.shard_count,
                "written": written,
            }
        )
    )


if __name__ == "__main__":
    main()
