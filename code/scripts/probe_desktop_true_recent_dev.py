#!/usr/bin/env python3
"""dev 探针:把 R 臂换成『真·前一帧』重算 HGKV 的选择性。

# note (luojiaxuan): v1 语料的 R0/RA 沿用部署 recent 选择器,k=1 时选中事件 s-1,
# 其 post 帧与当前截图逐字节相同(在线语义如此)——它是『纯冗余图』控制,不是
# 『有信息的 recent』控制。本探针对 dev 94 组构造 R'0/R'A:恢复事件 s-2 的 post
# 帧(= 当前截图的前一帧,决策步 s-1 动作执行前的观测),budget/渲染/target 与
# v1 臂完全同构。只算这两臂(~188 前向);S0(冻结)与 SA@s150(active)从已有
# dev 分数缓存 join,由归约方计算:
#   true_selection_effect = S0 - R'0            (冻结模型 vs 真前一帧)
#   did_select_true       = (SA - S0) - (R'A - R'0)
# 同时记录事件 s-1 的 screen_changed(前一帧与当前帧是否逐字节相同)以便分层。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.agentnet_desktop_cr import (
    build_agentnet_cr_request,
    render_agentnet_cr_messages,
)
from scripts.build_desktop_hgkv_corpus import (
    SAMPLE_SCHEMA,
    _serialize_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="HGKV 选点 checkpoint(如 s150)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    import torch
    from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
    from causalcache.policy.history_gated_lora import (
        inject_history_gated_kv,
        load_history_gated_state_dict,
    )
    from scripts.run_exploratory_closed_loop_episode import (
        EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from scripts.train_success_sft_lora import (
        adapter_scope_for_sample,
        encode_sample,
        mean_target_logprob,
    )

    records = {
        record["dp_id"]: record
        for record in load_jsonl(args.screening_manifest)
    }
    rows = load_jsonl(args.corpus_root / "samples.jsonl")
    dev_positives = [
        row for row in rows
        if row["split"] == "dev" and row["arm_slot"] == "SA"
    ]
    if not dev_positives:
        raise SystemExit("corpus has no dev groups")

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

    results = []
    for position, positive in enumerate(dev_positives, start=1):
        record = records[positive["source"]["dp_id"]]
        current_step = int(record["step"])
        true_recent = current_step - 2
        if true_recent < 1:
            raise SystemExit(f"{positive['pair_group']}: no previous frame")
        relpaths = record["image_relpaths"]
        screenshots = [
            (args.corpus_root / relpath).read_bytes() for relpath in relpaths
        ]
        request = build_agentnet_cr_request(
            instruction=record["instruction"],
            screenshots=screenshots,
            history_actions=record["history"],
            current_step=current_step,
            recent_budget=0,
            extra_restored_step_ids=(true_recent,),
            screen_size=tuple(record["screen_size"]),
        )
        messages = _serialize_messages(
            render_agentnet_cr_messages(request),
            image_paths=[relpaths[true_recent], relpaths[current_step - 1]],
        )
        sample = {
            "schema_version": SAMPLE_SCHEMA,
            "sample_id": f"{positive['pair_group']}|Rtrue",
            "prompt_format": "osworld_official",
            "messages": messages,
            "target_text": positive["target_text"],
            "budget": 1,
            "selected_steps": [true_recent],
            "memory_config": {"restored_event_step_ids": [true_recent]},
        }

        def score(mode: str) -> float:
            encoded = encode_sample(
                runtime, sample, dataset_root=args.corpus_root, torch=torch
            )
            scope = adapter_scope_for_sample(
                "history_gated_kv",
                encoded,
                {**sample, "adapter_mode": mode},
                merge_size=merge_size,
            )
            with scope, torch.no_grad():
                return float(
                    mean_target_logprob(model, encoded, torch=torch).detach()
                )

        prev_bytes = screenshots[true_recent]
        current_bytes = screenshots[current_step - 1]
        results.append(
            {
                "pair_group": positive["pair_group"],
                "episode": positive["episode"],
                "true_recent_event": true_recent,
                "prev_equals_current": prev_bytes == current_bytes,
                "r_true_bypass": score("bypass"),
                "r_true_active": score("active"),
            }
        )
        if position % 10 == 0:
            print(f"{position}/{len(dev_positives)} groups", flush=True)

    args.output.write_text(
        json.dumps(
            {
                "schema_version": "causalcache.desktop_true_recent_probe.v1",
                "checkpoint": str(args.checkpoint),
                "groups": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"groups": len(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
