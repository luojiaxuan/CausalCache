#!/usr/bin/env python3
"""Selector v4 单例标签:官方多轮渲染 + v4 policy 的逐候选真实效用。

# note (luojiaxuan): v3 版(score_selector_v3_singletons.py)绑定单轮 renderer 与
# 旧格式 s300;probe 已证明 adapter 选择性不跨 renderer 迁移,故 selector 标签
# 必须用 v4 官方多轮 renderer + v4 checkpoint 重打。与 v3 的差异仅三处:
#   1. 渲染:agentnet_desktop_official.build_desktop_official_messages
#      (B0 = 官方 kept=0 全折叠;候选 {j} = gap-fold 单保留轮);
#   2. prompt_format=desktop_official_multiturn(chat_template 分支);
#      target = 官方完整响应(Action 行 + tool_call);
#   3. 候选合格性多一条:保留轮需要该步官方响应可渲染(tripleClick 类跳过并计数)。
# 其余口径逐字继承 v3:候选池 = 事件 1..s-2 post 帧逐 byte 去重保留最近实例、
# base 锚 = 冻结 bypass、候选 = active、jsonl 断点续跑、checkpoint SHA fail-closed。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from causalcache.agentnet_desktop_official import (
    build_desktop_official_messages,
    build_official_forms_for_record,
    render_official_target_text,
)
from scripts.build_desktop_hgkv_corpus import _split

SCHEMA_VERSION = "causalcache.selector_v4_singletons.v1"
PROMPT_FORMAT = "desktop_official_multiturn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-sha256", default=None,
                        help="fail-closed 校验;标签与 checkpoint 身份绑定")
    parser.add_argument(
        "--teacher", choices=("hgkv", "frozen"), default="hgkv",
        help="frozen = 锚与候选全部用冻结策略 bypass 打分(不注入 adapter),"
             "此时不接受 --checkpoint;hgkv = v4 原口径(锚 bypass、候选 active)",
    )
    parser.add_argument("--teacher-layer-count", type=int, default=8,
                        help="HGKV 教师的 layer_scope 层数,须与训练配置一致")
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

    if args.teacher == "frozen":
        if args.checkpoint is not None or args.checkpoint_sha256 is not None:
            raise SystemExit("--teacher frozen 不接受 --checkpoint(教师即冻结策略)")
        digest = "frozen-bypass"
    else:
        if args.checkpoint is None or args.checkpoint_sha256 is None:
            raise SystemExit("--teacher hgkv 需要 --checkpoint 与 --checkpoint-sha256")
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
            "key": "__fingerprint__",
            "fingerprint": {
                "schema_version": SCHEMA_VERSION,
                "prompt_format": PROMPT_FORMAT,
                "checkpoint_sha256": digest,
                "manifest_sha256": hashlib.sha256(
                    args.screening_manifest.read_bytes()).hexdigest(),
                "visual_tokens": EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
                "seed": args.seed,
                "splits": sorted(args.splits),
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
    if args.teacher == "hgkv":
        # note (luojiaxuan): 同 sets 脚本,层数须与教师训练配置一致,不能写死 8。
        wrapped = inject_history_gated_kv(
            model, layer_count=args.teacher_layer_count, rank=8, alpha=16)
        load_history_gated_state_dict(
            wrapped, torch.load(args.checkpoint, map_location="cpu")
        )
        for lora in wrapped.values():
            lora.lora_a.requires_grad_(False)
            lora.lora_b.requires_grad_(False)

    candidate_adapter_mode = "active" if args.teacher == "hgkv" else "bypass"

    heartbeat = args.output_root / f"heartbeat-shard{args.shard_index:03d}.json"

    def score_prompt(record, forms, steps, *, adapter_mode: str) -> float:
        relpaths = record["image_relpaths"]
        current_step = int(record["step"])
        messages = build_desktop_official_messages(
            goal=str(record["instruction"]),
            steps=forms,
            shown_events=list(steps),
            event_images={step: relpaths[step] for step in steps},
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
            "sample_id": f"{record['dp_id']}|{','.join(map(str, steps)) or 'b0'}",
            "prompt_format": PROMPT_FORMAT,
            "messages": serialized,
            "target_text": render_official_target_text(record["target_tool_call"]),
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
        dp = str(record["dp_id"])
        current_step = int(record["step"])
        relpaths = record["image_relpaths"]
        forms = build_official_forms_for_record(record)
        # 候选池:事件 1..s-2 的 post 帧,逐 byte 去重保留最近实例;
        # v4 追加:保留轮官方响应可渲染(steps[event] 即步骤 event+1)。
        seen_hash: dict[str, int] = {}
        candidates: list[int] = []
        duplicates: dict[int, int] = {}
        unrenderable: list[int] = []
        for event in range(current_step - 2, 0, -1):
            if not forms[event].full_response:
                unrenderable.append(event)
                continue
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
            base = score_prompt(record, forms, (), adapter_mode="bypass")
            handle.write(json.dumps({
                "key": base_key, "dp_id": dp, "kind": "b0",
                "split": _split(str(record["task_id"]), seed=args.seed),
                "episode": str(record["task_id"]),
                "decision_step": current_step,
                "candidate_pool": candidates,
                "duplicates": {str(k): v for k, v in duplicates.items()},
                "unrenderable_events": sorted(unrenderable),
                "score": base,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(base_key)
        for event in candidates:
            key = f"{dp}|{event}"
            if key in done:
                continue
            value = score_prompt(
                record, forms, (event,), adapter_mode=candidate_adapter_mode
            )
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
