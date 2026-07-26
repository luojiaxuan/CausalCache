#!/usr/bin/env python3
"""Selector v3 单例标签:全候选池逐图真实效用(固定预算重分配框架)。

# note (luojiaxuan): §10.1 按 fixed-budget 重冻结后的第一阶段标签。对每个
# screening 决策点(train/dev;test 整体不碰):
#   * base = summaries + 当前图(零恢复图);anchor = frozen B0 分数
#     (HGKV B0 逐位 parity,锚用冻结模型即锚用 s300);
#   * 候选池 = 事件 1..s-2 的 post 帧(s-1 与当前图同帧,不作候选),逐 byte
#     去重保留**最近**一份(重复帧的更早实例记 duplicate_of);
#   * 每个候选 j:U({j}) = logp_s300(a*|base+{j}) − anchor,teacher-forced,
#     osworld_official 渲染与训练/打分同一条编码路径;
#   * Recent 帧就是普通候选 —— k 是 selector 行为的结果统计量。
# 输出 jsonl cache(dp|cand 键,断点续跑);集合标签(B=2/4 beam+STOP)由后续
# 阶段基于本表 shortlist 构造,每个 edge 都完整重打分,禁止 singleton 求和。
# 风险登记:s300 的 §9-6/7 生成式评测并行进行中;若该闸失败则本表按新选点重跑
# (用户 2026-07-27 裁定以速度优先并接受该风险)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from causalcache.agentnet_desktop_cr import (
    build_agentnet_cr_request,
    render_agentnet_cr_messages,
)
from scripts.build_desktop_hgkv_corpus import _serialize_messages, _split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True,
                        help="fail-closed 校验;标签与 checkpoint 身份绑定")
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

    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != args.checkpoint_sha256:
        raise SystemExit(
            f"checkpoint SHA drifted: {digest} != {args.checkpoint_sha256}"
        )

    records = [
        json.loads(line)
        for line in args.screening_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    states = [
        record for record in records
        if _split(str(record["task_id"]), seed=args.seed) in set(args.splits)
    ]
    states = [
        state for index, state in enumerate(states)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit_states:
        states = states[: args.limit_states]

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / (
        f"singletons.shard{args.shard_index:03d}-of-{args.shard_count:03d}.jsonl"
    )
    done = load_done(out_path)
    handle = out_path.open("a", encoding="utf-8")
    if not done:
        handle.write(json.dumps({
            "fingerprint": {
                "schema_version": "causalcache.selector_v3_singletons.v1",
                "checkpoint_sha256": digest,
                "manifest_sha256": hashlib.sha256(
                    args.screening_manifest.read_bytes()).hexdigest(),
                "visual_tokens": EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
                "seed": args.seed,
                "splits": sorted(args.splits),
            }
        }, ensure_ascii=False) + "\n")
        handle.flush()

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

    heartbeat = args.output_root / f"heartbeat-shard{args.shard_index:03d}.json"

    def score_prompt(record, steps, *, adapter_mode: str) -> float:
        relpaths = record["image_relpaths"]
        current_step = int(record["step"])
        screenshots = [
            (args.image_root / relpath).read_bytes() for relpath in relpaths
        ]
        request = build_agentnet_cr_request(
            instruction=record["instruction"],
            screenshots=screenshots,
            history_actions=record["history"],
            current_step=current_step,
            recent_budget=0,
            extra_restored_step_ids=tuple(steps),
            screen_size=tuple(record["screen_size"]),
        )
        messages = _serialize_messages(
            render_agentnet_cr_messages(request),
            image_paths=[*(relpaths[step] for step in steps),
                          relpaths[current_step - 1]],
        )
        sample = {
            "sample_id": f"{record['dp_id']}|{','.join(map(str, steps)) or 'b0'}",
            "prompt_format": "osworld_official",
            "messages": messages,
            "target_text": record["target_text"],
            "memory_config": {"restored_event_step_ids": list(steps)},
            "adapter_mode": adapter_mode,
        }
        encoded = encode_sample(
            runtime, sample, dataset_root=args.image_root, torch=torch
        )
        if encoded is None:
            raise RuntimeError(f"empty encoding for {sample['sample_id']}")
        scope = adapter_scope_for_sample(
            "history_gated_kv", encoded, sample, merge_size=merge_size
        )
        with scope, torch.no_grad():
            return float(mean_target_logprob(model, encoded, torch=torch).detach())

    import time
    scored = 0
    for position, record in enumerate(states, start=1):
        dp = record["dp_id"]
        current_step = int(record["step"])
        relpaths = record["image_relpaths"]
        # 候选池:事件 1..s-2 的 post 帧,逐 byte 去重保留最近实例
        seen_hash: dict[str, int] = {}
        candidates: list[int] = []
        duplicates: dict[int, int] = {}
        for event in range(current_step - 2, 0, -1):
            frame_hash = hashlib.sha256(
                (args.image_root / relpaths[event]).read_bytes()
            ).hexdigest()
            if frame_hash in seen_hash:
                duplicates[event] = seen_hash[frame_hash]
                continue
            seen_hash[frame_hash] = event
            candidates.append(event)
        candidates.sort()

        base_key = f"{dp}|b0"
        if base_key not in done:
            base = score_prompt(record, (), adapter_mode="bypass")
            handle.write(json.dumps({
                "key": base_key, "dp_id": dp, "kind": "b0",
                "split": _split(str(record["task_id"]), seed=args.seed),
                "episode": str(record["task_id"]),
                "decision_step": current_step,
                "candidate_pool": candidates,
                "duplicates": {str(k): v for k, v in duplicates.items()},
                "score": base,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(base_key)
        for event in candidates:
            key = f"{dp}|{event}"
            if key in done:
                continue
            value = score_prompt(record, (event,), adapter_mode="active")
            handle.write(json.dumps({
                "key": key, "dp_id": dp, "kind": "singleton",
                "event": event, "age": current_step - event,
                "score": value,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(key)
            scored += 1
        if position % 5 == 0:
            heartbeat.write_text(json.dumps({
                "states_done": position, "states_total": len(states),
                "scored": scored, "time": time.time(),
            }) + "\n", encoding="utf-8")
    handle.close()
    print(json.dumps({"shard": args.shard_index, "states": len(states),
                       "scored": scored, "output": str(out_path)}))


if __name__ == "__main__":
    main()
