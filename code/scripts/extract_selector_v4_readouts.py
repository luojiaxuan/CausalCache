#!/usr/bin/env python3
"""HGKV 反事实读出特征提取(selector two-tower 残差塔输入)。

# note (luojiaxuan): 学习曲线判定 cheap 特征封顶(剂量反应平坦 + train/dev MSE
# 无缺口)后的处方。对每个 (决策点, 候选事件) 跑一次**单例 active 前向**(与
# 单例标签同一渲染/编码路径),在 8 个 HGKV 层的 k/v hook 里捕获候选历史 token
# 位置上的 adapted 输出与 LoRA 残差(base = output − delta·gate,解析恢复,
# 无需第二次 bypass 前向),q_proj 捕获末位置 query。每层每 kv-head 7 个
# 标量,4 kv-head × 8 层 = 224 维:
#   [mean‖Δk‖/‖k‖, max‖Δk‖/‖k‖, mean‖Δv‖/‖v‖, max‖Δv‖/‖v‖,
#    cos(Δv̄, v̄_base), ‖Δv̄‖, cos(q̄_group, Δk̄)]
# 输出原始值;**候选池内 z 归一在训练端做**(train_selector_v4_marginal
# --readout-root),跨平台分布漂移由池内归一吸收。候选池逐字复用单例表 b0 行
# (保证与标签同池);checkpoint SHA fail-closed。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from causalcache.agentnet_desktop_official import (
    build_desktop_official_messages,
    build_official_forms_for_record,
    render_official_target_text,
)

SCHEMA_VERSION = "causalcache.selector_v4_readouts.v1"
PROMPT_FORMAT = "desktop_official_multiturn"
FEATURES_PER_HEAD = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--singletons-root", type=Path, required=True)
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-states", type=int, default=0)
    return parser.parse_args()


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> None:
    args = parse_args()
    import torch
    from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
    from causalcache.policy.history_adapter_context import (
        get_history_adapter_context,
    )
    from causalcache.policy.history_gated_lora import (
        inject_history_gated_kv,
        load_history_gated_state_dict,
    )
    from scripts.build_desktop_hgkv_corpus import _split
    from scripts.run_exploratory_closed_loop_episode import (
        EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from scripts.train_success_sft_lora import (
        adapter_scope_for_sample,
        encode_sample,
    )

    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != args.checkpoint_sha256:
        raise SystemExit(
            f"checkpoint SHA drifted: {digest} != {args.checkpoint_sha256}"
        )

    records = {
        str(json.loads(line)["dp_id"]): json.loads(line)
        for line in args.screening_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    b0_rows: dict[str, dict] = {}
    for path in sorted(args.singletons_root.glob("singletons.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "b0":
                b0_rows[row["dp_id"]] = row

    ordered = [
        (dp, base) for dp, base in b0_rows.items()
        if dp in records
        and _split(str(records[dp]["task_id"]), seed=args.seed) in set(args.splits)
    ]
    states = [
        item for index, item in enumerate(ordered)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit_states:
        states = states[: args.limit_states]

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / (
        f"readouts.shard{args.shard_index:03d}-of-{args.shard_count:03d}.jsonl"
    )
    done = load_done(out_path)
    handle = out_path.open("a", encoding="utf-8")
    if not done:
        handle.write(json.dumps({
            "key": "__fingerprint__",
            "fingerprint": {
                "schema_version": SCHEMA_VERSION,
                "prompt_format": PROMPT_FORMAT,
                "checkpoint_sha256": digest,
                "features_per_head": FEATURES_PER_HEAD,
                "seed": args.seed,
                "splits": sorted(args.splits),
                "visual_tokens": EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
            }
        }, ensure_ascii=False) + "\n")
        handle.flush()
        done.add("__fingerprint__")

    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    model = runtime.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    merge_size = int(runtime.processor.image_processor.merge_size)
    wrapped = inject_history_gated_kv(model, layer_count=8, rank=8, alpha=16)
    load_history_gated_state_dict(
        wrapped, torch.load(args.checkpoint, map_location="cpu")
    )
    for lora in wrapped.values():
        lora.lora_a.requires_grad_(False)
        lora.lora_b.requires_grad_(False)

    text_config = getattr(model.config, "text_config", model.config)
    n_kv = int(text_config.num_key_value_heads)
    n_q = int(text_config.num_attention_heads)
    group = n_q // n_kv

    # ---- 捕获 hook(注册在 HGKV hook 之后 → output 已是 adapted)----
    captured: dict[str, dict] = {}

    def make_kv_hook(name: str, lora) -> object:
        def hook(module, inputs, output):
            context = get_history_adapter_context()
            if context is None or not context.history_present:
                return None
            mask = context.history_token_mask
            if mask is None or mask.shape != output.shape[:-1] or not bool(mask.any()):
                return None
            x = inputs[0]
            delta = (
                x.to(lora.lora_a.dtype) @ lora.lora_a.T @ lora.lora_b.T
            ) * lora.scaling
            sel = mask[0].bool()
            captured[name] = {
                "delta": delta[0, sel, :].detach().float(),
                "base": (output[0, sel, :].float() - delta[0, sel, :]).detach(),
            }
            return None
        return hook

    def q_hook(module, inputs, output):
        captured["__q__" + str(id(module))] = output[0, -1, :].detach().float()
        return None

    handles = []
    layer_of: dict[str, int] = {}
    q_of_layer: dict[int, str] = {}
    import re as _re
    for name, lora in wrapped.items():
        handles.append(lora.module.register_forward_hook(make_kv_hook(name, lora)))
        layer_of[name] = int(_re.search(r"\.layers\.(\d+)\.", f".{name}.").group(1))
    seen_layers = sorted(set(layer_of.values()))
    for mod_name, module in model.named_modules():
        match = _re.search(r"\.layers\.(\d+)\.self_attn\.q_proj$", f".{mod_name}")
        if match and int(match.group(1)) in seen_layers:
            handles.append(module.register_forward_hook(q_hook))
            q_of_layer[int(match.group(1))] = "__q__" + str(id(module))

    eps = 1e-6

    def readout_vector() -> list[float]:
        vec: list[float] = []
        for layer in seen_layers:
            k_name = next(n for n, l in layer_of.items()
                          if l == layer and n.endswith("k_proj"))
            v_name = next(n for n, l in layer_of.items()
                          if l == layer and n.endswith("v_proj"))
            k = captured[k_name]
            v = captured[v_name]
            q_last = captured[q_of_layer[layer]]
            head_dim = k["base"].shape[-1] // n_kv
            for h in range(n_kv):
                sl = slice(h * head_dim, (h + 1) * head_dim)
                dk, kb = k["delta"][:, sl], k["base"][:, sl]
                dv, vb = v["delta"][:, sl], v["base"][:, sl]
                rel_dk = dk.norm(dim=-1) / (kb.norm(dim=-1) + eps)
                rel_dv = dv.norm(dim=-1) / (vb.norm(dim=-1) + eps)
                dv_mean = dv.mean(dim=0)
                vb_mean = vb.mean(dim=0)
                dk_mean = dk.mean(dim=0)
                q_group = q_last.view(n_q, -1)[h * group:(h + 1) * group].mean(dim=0)
                cos_v = torch.nn.functional.cosine_similarity(
                    dv_mean, vb_mean, dim=0).item()
                cos_qk = torch.nn.functional.cosine_similarity(
                    q_group, dk_mean, dim=0).item()
                vec.extend([
                    rel_dk.mean().item(), rel_dk.max().item(),
                    rel_dv.mean().item(), rel_dv.max().item(),
                    cos_v, dv_mean.norm().item(), cos_qk,
                ])
        return vec

    import time
    for position, (dp, base) in enumerate(states, start=1):
        record = records[dp]
        relpaths = record["image_relpaths"]
        current_step = int(record["step"])
        forms = build_official_forms_for_record(record)
        pool = [int(e) for e in base["candidate_pool"]]
        for event in pool:
            key = f"{dp}|{event}"
            if key in done:
                continue
            messages = build_desktop_official_messages(
                goal=str(record["instruction"]),
                steps=forms,
                shown_events=[event],
                event_images={event: relpaths[event]},
                current_image=relpaths[current_step - 1],
            )
            serialized = []
            for message in messages:
                content = []
                for part in message["content"]:
                    if part.get("type") == "image":
                        content.append({"type": "image", "path": part["image"]})
                    else:
                        content.append(dict(part))
                serialized.append({"role": message["role"], "content": content})
            sample = {
                "sample_id": key,
                "prompt_format": PROMPT_FORMAT,
                "messages": serialized,
                "target_text": render_official_target_text(
                    record["target_tool_call"]),
                "memory_config": {"restored_event_step_ids": [event]},
                "adapter_mode": "active",
            }
            encoded = encode_sample(
                runtime, sample, dataset_root=args.image_root, torch=torch
            )
            if encoded is None:
                raise RuntimeError(f"empty encoding for {key}")
            captured.clear()
            scope = adapter_scope_for_sample(
                "history_gated_kv", encoded, sample, merge_size=merge_size
            )
            with scope, torch.no_grad():
                model(**encoded, logits_to_keep=1)
            missing = [n for n in layer_of if n not in captured]
            if missing:
                raise RuntimeError(f"hooks missed modules: {missing[:2]}")
            handle.write(json.dumps({
                "key": key, "kind": "readout", "dp_id": dp, "event": event,
                "vector": [round(x, 6) for x in readout_vector()],
            }, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(key)
        if position % 5 == 0:
            (args.output_root / f"heartbeat-shard{args.shard_index:03d}.json"
             ).write_text(json.dumps({
                 "states_done": position, "states_total": len(states),
                 "time": time.time(),
             }) + "\n", encoding="utf-8")
    handle.close()
    for h in handles:
        h.remove()
    print(json.dumps({"shard": args.shard_index, "states": len(states)}))


if __name__ == "__main__":
    main()
