#!/usr/bin/env python3
"""Probe:官方多轮结构下 k=1..4 替换的边际效用 + 旧格式 HGKV 的格式迁移。

# note (luojiaxuan): 2026-07-27 用户指令"probe看看要不要构造新一版本训练数据,
# 比如看看要不要增加k=2/3/4"。决策量(全部 teacher-forced,官方多轮渲染):
#   * 每个决策点固定 B=4 预算:R = Recent-4;S_k = Recent-(4-k) + k 个最近的
#     distinct 正例旧帧(替换最老的 k 个槽位,k=1..4,受可用正例数限制);
#   * frozen 口径:U_f(S_k) − U_f(R) —— 新格式下冻结模型的 k 边际(k=1 与
#     v3 的 frozen_selection_effect 同型,直接检验格式迁移 ef19e1a 风险);
#   * s300 口径(旧格式训出的 HGKV):[U_a(S_k)−U_0(S_k)] − [U_a(R)−U_0(R)]
#     —— did_select 的 k 版本,检验旧结构学的选择性换结构后是否存活;
#   * 可用性:每状态合格正例事件数分布(k≥2 语料臂的原料够不够)。
# 输出 jsonl(断点续跑)+ 终端汇总。正例枚举复用冻结的
# history_action_matches_target;年龄门槛 age ≥ B+2 = 6 与 v3 B=4 臂一致。
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
from scripts.build_desktop_did_corpus_v2 import recent_window
from scripts.build_desktop_hgkv_corpus import (
    _split,
    history_action_matches_target,
)

PROBE_B = 4
MIN_AGE = PROBE_B + 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--coordinate-tolerance", type=int, default=25)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-states", type=int, default=0)
    parser.add_argument(
        "--min-witnesses", type=int, default=1,
        help="只打分正例事件数 >= 此值的状态(k 边际要 >=2 才有信息;=1 时含 k=1 基线)",
    )
    return parser.parse_args()


def witness_events(record, *, coordinate_tolerance: int) -> list[int]:
    """全部合格正例事件(age >= MIN_AGE),按事件号降序(最近优先)。"""
    current_step = int(record["step"])
    screen_size = tuple(record["screen_size"])
    target = record["target_tool_call"]
    events: list[int] = []
    for entry in record["history"]:
        event = int(entry["step_id"]) - 1
        if event < 1 or current_step - event < MIN_AGE:
            continue
        if history_action_matches_target(
            entry["action"], target,
            screen_size=screen_size,
            coordinate_tolerance=coordinate_tolerance,
        ):
            events.append(event)
    return sorted(set(events), reverse=True)


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
    states = []
    for record in records:
        if _split(str(record["task_id"]), seed=args.seed) not in set(args.splits):
            continue
        # 预算窗口本身要放得下:与 v3 B=4 组的合格条件一致。
        if int(record["step"]) - 1 - PROBE_B < 1:
            continue
        witnesses = witness_events(
            record, coordinate_tolerance=args.coordinate_tolerance
        )
        if len(witnesses) < args.min_witnesses:
            continue
        states.append((record, witnesses))
    states = [
        item for index, item in enumerate(states)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit_states:
        states = states[: args.limit_states]

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / (
        f"probe_k.shard{args.shard_index:03d}-of-{args.shard_count:03d}.jsonl"
    )
    done = load_done(out_path)
    handle = out_path.open("a", encoding="utf-8")
    if not done:
        handle.write(json.dumps({
            "key": "__fingerprint__",
            "schema_version": "causalcache.desktop_official_k_probe.v1",
            "checkpoint_sha256": digest,
            "manifest_sha256": hashlib.sha256(
                args.screening_manifest.read_bytes()).hexdigest(),
            "probe_b": PROBE_B,
            "min_age": MIN_AGE,
            "visual_tokens": EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
            "seed": args.seed,
            "splits": sorted(args.splits),
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

    heartbeat = args.output_root / f"heartbeat-shard{args.shard_index:03d}.json"

    def score(record, forms, steps, *, adapter_mode: str) -> float:
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
            "sample_id": f"{record['dp_id']}|{','.join(map(str, steps)) or 'r'}",
            "prompt_format": "desktop_official_multiturn",
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
    for position, (record, witnesses) in enumerate(states, start=1):
        dp = str(record["dp_id"])
        if f"{dp}|done" in done:
            continue
        current_step = int(record["step"])
        forms = build_official_forms_for_record(record)
        window = recent_window(current_step, PROBE_B)

        selections: dict[str, list[int]] = {"recent": list(window)}
        max_k = min(PROBE_B, len(witnesses))
        for k in range(1, max_k + 1):
            chosen = sorted(witnesses[:k])
            kept = window[k:]  # 替换最老的 k 个槽位
            selections[f"k{k}"] = sorted([*chosen, *kept])
        try:
            row: dict[str, object] = {
                "key": f"{dp}|done", "dp_id": dp,
                "split": _split(str(record["task_id"]), seed=args.seed),
                "episode": str(record["task_id"]),
                "decision_step": current_step,
                "witness_count": len(witnesses),
                "witness_events": witnesses,
                "selections": {name: steps for name, steps in selections.items()},
                "scores": {},
            }
            for name, steps in selections.items():
                bypass = score(record, forms, steps, adapter_mode="bypass")
                active = score(record, forms, steps, adapter_mode="active")
                row["scores"][name] = {"bypass": bypass, "active": active}
        except ValueError as error:
            handle.write(json.dumps({
                "key": f"{dp}|done", "dp_id": dp, "skipped": str(error),
            }, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(f"{dp}|done")
            continue
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        done.add(f"{dp}|done")
        if position % 5 == 0:
            heartbeat.write_text(json.dumps({
                "states_done": position, "states_total": len(states),
                "time": time.time(),
            }) + "\n", encoding="utf-8")
    handle.close()
    print(json.dumps({"shard": args.shard_index, "states": len(states)}))


if __name__ == "__main__":
    main()
