#!/usr/bin/env python3
"""Selector v4 集合标签:beam+STOP teacher,每条 edge 整集重打分。

# note (luojiaxuan): stage-2(set-conditioned)训练数据。冻结口径:
#   * 输入 = v4 单例标签表(b0 锚 + 逐候选 U({j}),官方多轮渲染,s300 active);
#     只处理单例覆盖完整的状态(b0 行 + candidate_pool 每个事件都有行),
#     不完整的跳过计数,靠断点续跑在上游补齐后收尾;
#   * shortlist = 按 U({j}) 降序取前 M(默认 6);
#   * beam(默认宽 3)从 size-1 种子逐级扩到 max_b(默认 4),每个新集合
#     **完整重打分** U(S)(shown_events = 升序集合,官方 gap-fold 渲染,active)
#     —— 禁止 singleton 求和近似;size-1 的 U 直接取单例表,不重算;
#   * STOP 不单独打分:每条 edge 的边际 U(S∪{j})−U(S) 已在表里,训练侧自行
#     构造 STOP 标签(最优边际 ≤ 0 即停);
#   * 输出 jsonl(dp|set 键,断点续跑),绑定同一 checkpoint SHA。
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
from scripts.build_desktop_did_corpus_v2 import recent_window
from scripts.build_desktop_hgkv_corpus import _split

SCHEMA_VERSION = "causalcache.selector_v4_sets.v1"
PROMPT_FORMAT = "desktop_official_multiturn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--singletons-root", type=Path, required=True,
                        help="singleton shard jsonl 所在目录(本地快照)")
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument(
        "--teacher", choices=("hgkv", "frozen"), default="hgkv",
        help="frozen = 全部集合用冻结策略 bypass 打分(不注入 adapter),"
             "不接受 --checkpoint;hgkv = v4 原口径(集合 active)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--shortlist", type=int, default=6)
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--max-b", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-states", type=int, default=0)
    return parser.parse_args()


def load_singletons(root: Path):
    """singleton 分片 → (b0_by_dp, scores_by_dp[event] = U)。"""
    b0: dict[str, dict] = {}
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    for path in sorted(root.glob("singletons.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # 快照可能截断在半行,断点续跑补齐
            if row.get("kind") == "b0":
                b0[row["dp_id"]] = row
            elif row.get("kind") == "singleton":
                scores[row["dp_id"]][int(row["event"])] = float(row["score"])
    return b0, scores


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

    b0_rows, singleton_scores = load_singletons(args.singletons_root)
    records = {
        str(json.loads(line)["dp_id"]): json.loads(line)
        for line in args.screening_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    states = []
    skipped_incomplete = 0
    ordered = [
        record for record in records.values()
        if _split(str(record["task_id"]), seed=args.seed) in set(args.splits)
    ]
    for index, record in enumerate(ordered):
        if index % args.shard_count != args.shard_index:
            continue
        dp = str(record["dp_id"])
        base = b0_rows.get(dp)
        if base is None:
            skipped_incomplete += 1
            continue
        pool = [int(e) for e in base["candidate_pool"]]
        have = singleton_scores.get(dp, {})
        if any(event not in have for event in pool):
            skipped_incomplete += 1
            continue
        if len(pool) < 2:
            continue  # 组不出 size>=2 的集合
        states.append((record, base, have))
    if args.limit_states:
        states = states[: args.limit_states]

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / (
        f"sets.shard{args.shard_index:03d}-of-{args.shard_count:03d}.jsonl"
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
                "shortlist": args.shortlist,
                "beam": args.beam,
                "max_b": args.max_b,
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
    if args.teacher == "hgkv":
        wrapped = inject_history_gated_kv(model, layer_count=8, rank=8, alpha=16)
        load_history_gated_state_dict(
            wrapped, torch.load(args.checkpoint, map_location="cpu")
        )
        for lora in wrapped.values():
            lora.lora_a.requires_grad_(False)
            lora.lora_b.requires_grad_(False)

    set_adapter_mode = "active" if args.teacher == "hgkv" else "bypass"

    heartbeat = args.output_root / f"heartbeat-shard{args.shard_index:03d}.json"

    def score_set(record, forms, events) -> float:
        relpaths = record["image_relpaths"]
        current_step = int(record["step"])
        messages = build_desktop_official_messages(
            goal=str(record["instruction"]),
            steps=forms,
            shown_events=sorted(events),
            event_images={e: relpaths[e] for e in events},
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
            "sample_id": f"{record['dp_id']}|set:{','.join(map(str, sorted(events)))}",
            "prompt_format": PROMPT_FORMAT,
            "messages": serialized,
            "target_text": render_official_target_text(record["target_tool_call"]),
            "memory_config": {"restored_event_step_ids": sorted(events)},
            "adapter_mode": set_adapter_mode,
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
    for position, (record, base, have) in enumerate(states, start=1):
        dp = str(record["dp_id"])
        state_key = f"{dp}|sets-done"
        if state_key in done:
            continue
        forms = build_official_forms_for_record(record)
        # note (luojiaxuan): 部署接受规则(2026-07-27 用户裁定"恒用满 B,recent
        # 兜底")的参照系是 U(Recent-B) 本身,beam shortlist 不保证覆盖它——
        # 每状态强制打 Recent-2 / Recent-4 锚集合(窗口步须全部可渲染,否则跳过
        # 计数;Recent-1 = 单例表里的事件 s-2,不重算)。
        current_step = int(record["step"])
        for anchor_b in (2, 4):
            window = recent_window(current_step, anchor_b)
            if not window or window[0] < 1:
                continue
            key = f"{dp}|set:{','.join(map(str, window))}"
            if key in done:
                continue
            if any(not forms[e].full_response for e in window):
                handle.write(json.dumps({
                    "key": key, "dp_id": dp, "kind": "recent_anchor_skipped",
                    "events": list(window), "size": anchor_b,
                }, ensure_ascii=False) + "\n")
                handle.flush()
                done.add(key)
                continue
            value = score_set(record, forms, window)
            handle.write(json.dumps({
                "key": key, "dp_id": dp, "kind": "recent_anchor",
                "events": list(window), "size": anchor_b,
                "score": value,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(key)
            scored += 1
        shortlist = sorted(have, key=lambda e: have[e], reverse=True)[: args.shortlist]
        # size-1 种子直接取单例表(不重算)。
        level = [((event,), have[event]) for event in
                 sorted(shortlist, key=lambda e: have[e], reverse=True)[: args.beam]]
        for size in range(2, args.max_b + 1):
            candidates = []
            seen: set[tuple[int, ...]] = set()
            for parent_set, parent_u in level:
                for event in shortlist:
                    if event in parent_set:
                        continue
                    child = tuple(sorted((*parent_set, event)))
                    if child in seen:
                        continue
                    seen.add(child)
                    candidates.append((child, parent_set, parent_u, event))
            scored_children = []
            for child, parent_set, parent_u, event in candidates:
                key = f"{dp}|set:{','.join(map(str, child))}"
                if key in done:
                    continue
                value = score_set(record, forms, child)
                handle.write(json.dumps({
                    "key": key, "dp_id": dp, "kind": "set",
                    "events": list(child), "size": size,
                    "score": value,
                    "parent": list(parent_set), "parent_score": parent_u,
                    "added_event": event,
                }, ensure_ascii=False) + "\n")
                handle.flush()
                done.add(key)
                scored += 1
                scored_children.append((child, value))
            if not scored_children:
                break
            scored_children.sort(key=lambda item: item[1], reverse=True)
            level = scored_children[: args.beam]
        handle.write(json.dumps({
            "key": state_key, "dp_id": dp, "kind": "state_done",
            "split": _split(str(record["task_id"]), seed=args.seed),
            "episode": str(record["task_id"]),
            "b0_score": base["score"],
            "shortlist": shortlist,
        }, ensure_ascii=False) + "\n")
        handle.flush()
        done.add(state_key)
        if position % 5 == 0:
            heartbeat.write_text(json.dumps({
                "states_done": position, "states_total": len(states),
                "scored": scored, "skipped_incomplete": skipped_incomplete,
                "time": time.time(),
            }) + "\n", encoding="utf-8")
    handle.close()
    print(json.dumps({
        "shard": args.shard_index, "states": len(states), "scored": scored,
        "skipped_incomplete": skipped_incomplete, "output": str(out_path),
    }))


if __name__ == "__main__":
    main()
