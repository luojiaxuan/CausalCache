#!/usr/bin/env python3
"""Build the fixed-budget replacement corpus v4 (official multi-turn renderer).

# note (luojiaxuan): v4 = v3 的组代数逐字不变 + renderer 整体换成官方多轮结构
# (2026-07-27 用户裁定;Codex 报告确认单轮结构/执行器 JSON 历史/像素-[0,999]
# 量纲混用是协议缺陷,MobileWorld 前例 ef19e1a 表明格式差异可以吞掉稀疏选点收益)。
# 冻结口径不变:
#   * B ∈ {1,2,4} 训练、B=8 只建不训、B=0 仅 parity;R = Recent-B 不同帧窗口,
#     S/W = 替换最老 recent 槽位(k=1),正/错旧帧 age ≥ B+2;split 同 seed 轨迹哈希。
# v4 专属:
#   * 消息 = agentnet_desktop_official.build_desktop_official_messages 的 gap-fold
#     官方滚动结构;R 臂(连续窗口)逐消息退化为官方结构;
#   * 历史 = 官方响应格式(Action 行 + <tool_call>,坐标全 [0,999]),由冻结
#     parse_agentnet_step 重解析 raw_code 渲染;
#   * target_text 换为官方完整响应(Action 行 + tool_call),与保留轮示范同构;
#   * 保留轮必须有官方完整响应 —— 任一所需保留步 target 形态缺失(如 tripleClick)
#     则该 (dp, B) 组整组弃用并计数(fail-closed,不许静默降级为纯文本轮)。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.agentnet_desktop_official import (
    DESKTOP_OFFICIAL_PROTOCOL_ID,
    build_desktop_official_messages,
    build_official_forms_for_record,
    render_official_target_text,
)
from scripts.build_desktop_did_corpus_v2 import recent_window
from scripts.build_desktop_did_corpus_v3 import (
    TRAIN_B_VALUES,
    eligible_events_b,
    replaced_slot_event,
)
from scripts.build_desktop_hgkv_corpus import (
    DESKTOP_ARM_CONTRACT,
    NEGATIVE_ARM_SLOT,
    NEGATIVE_KIND,
    NEGATIVE_SCALE,
    _link_image_roots,
    _percentile_summary,
    _split,
    messages_sha256,
    sha256_file,
)

CORPUS_SCHEMA = "causalcache.desktop_did_corpus_b.v2"
SAMPLE_SCHEMA_V2 = "causalcache.desktop_did_sample.v2"
B0_SAMPLE_SCHEMA_V2 = "causalcache.desktop_did_b0_sample.v2"
PROMPT_FORMAT_V2 = "desktop_official_multiturn"
B_VALUES = (1, 2, 4, 8)

# 臂契约 = v1 表换 prompt_format;与 trainer 的 SPARSE_DESKTOP_ARM_CONTRACT_V2
# 逐字段一致(单测锁)。
DESKTOP_ARM_CONTRACT_V2: dict[str, tuple[str, str, str, str, str]] = {
    slot: (arm_id, role, PROMPT_FORMAT_V2, mode, adapter)
    for slot, (arm_id, role, _format, mode, adapter) in DESKTOP_ARM_CONTRACT.items()
}
B0_ARM_CONTRACT_V2 = ("B0", "parity", PROMPT_FORMAT_V2, "none", "bypass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--b-values", type=int, nargs="+", default=list(B_VALUES))
    parser.add_argument("--coordinate-tolerance", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-groups-per-b", type=int, default=0)
    parser.add_argument(
        "--link-images", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def _pathify(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """image payload(此处即 relpath 字符串)→ 语料的 {"type":"image","path":...}。"""
    serialized: list[dict[str, Any]] = []
    for message in messages:
        content: list[dict[str, Any]] = []
        for part in message["content"]:
            if part.get("type") == "image":
                path = part["image"]
                if not isinstance(path, str) or not path:
                    raise ValueError("image payload must be a relpath string")
                content.append({"type": "image", "path": path})
            else:
                content.append(dict(part))
        serialized.append({"role": message["role"], "content": content})
    return serialized


def render_official_selection(
    record: Mapping[str, Any],
    forms: Sequence[Any],
    *,
    steps: Sequence[int],
) -> list[dict[str, Any]]:
    """一个选中集的官方多轮消息(路径化)。保留步缺官方响应时向上抛 ValueError。"""
    relpaths = record["image_relpaths"]
    current_step = int(record["step"])
    messages = build_desktop_official_messages(
        goal=str(record["instruction"]),
        steps=forms,
        shown_events=list(steps),
        event_images={step: relpaths[step] for step in steps},
        current_image=relpaths[current_step - 1],
    )
    return _pathify(messages)


def build_b_groups(
    records: Sequence[dict[str, Any]],
    *,
    b: int,
    coordinate_tolerance: int,
    seed: int,
    max_groups: int = 0,
    emit_parity: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    referenced: set[str] = set()
    positive_ages: list[int] = []
    forms_cache: dict[str, Any] = {}
    for record in records:
        if max_groups and counters["groups"] >= max_groups:
            break
        selected = eligible_events_b(
            record, b=b, coordinate_tolerance=coordinate_tolerance, seed=seed
        )
        if selected is None:
            counters["rejected"] += 1
            continue
        positive_event, positive_action_step, wrong_event = selected
        current_step = int(record["step"])
        full_window = recent_window(current_step, b)
        kept_window = recent_window(current_step, b - 1)
        replaced_slot = replaced_slot_event(current_step, b)
        relpaths = record["image_relpaths"]
        current_path = relpaths[current_step - 1]
        dp = str(record["dp_id"])
        forms = forms_cache.get(dp)
        if forms is None:
            forms = build_official_forms_for_record(record)
            forms_cache[dp] = forms

        selection_specs = {
            "recent": list(full_window),
            "recurrence": sorted([*kept_window, positive_event]),
            "wrong": sorted([*kept_window, wrong_event]),
        }
        if emit_parity:
            selection_specs["none"] = []
        rendered: dict[str, dict[str, Any]] = {}
        try:
            for mode, steps in selection_specs.items():
                messages = render_official_selection(record, forms, steps=steps)
                rendered[mode] = {
                    "selected_steps": list(steps),
                    "selected_images": [relpaths[step] for step in steps],
                    "messages": messages,
                    "messages_sha256": messages_sha256(messages),
                }
        except ValueError:
            # 所需保留步没有官方完整响应(如 tripleClick)→ 整组 fail-closed。
            counters["dropped_retained_unrenderable"] += 1
            continue

        target_text = render_official_target_text(record["target_tool_call"])
        pair_group = f"agentnet:{dp}:b{b}"
        episode = str(record["task_id"])
        split = _split(episode, seed=seed)
        common = {
            "pair_group": pair_group,
            "episode": episode,
            "decision_step": current_step,
            "budget": b,
            "kept_recent_steps": list(kept_window),
            "replaced_slot_event": replaced_slot,
            "k_replaced": 1,
            "split": split,
            "instruction": record["instruction"],
            "current_image": current_path,
            "target_text": target_text,
            "reference_arm_id": "R0",
            "deployment_baseline_arm_id": "R0",
            "source": {
                "dataset": "AgentNet/OpenCUA",
                "os": record["os"],
                "dp_id": dp,
                "positive_rule": "full_action_recurrence_pre_state",
                "positive_action_step": positive_action_step,
                "positive_event_step": positive_event,
                "wrong_event_step": wrong_event,
                "positive_age": current_step - positive_event,
                "wrong_age": current_step - wrong_event,
                "recent_definition": "distinct_frames_v2",
                "budget_semantics": "fixed_budget_replacement_v3",
                "renderer": DESKTOP_OFFICIAL_PROTOCOL_ID,
            },
        }
        for slot, (arm_id, role, prompt_format, mode, adapter_mode) in (
            DESKTOP_ARM_CONTRACT_V2.items()
        ):
            row = {
                "schema_version": SAMPLE_SCHEMA_V2,
                "sample_id": f"{pair_group}|{slot}",
                **common,
                "arm_slot": slot,
                "arm_id": arm_id,
                "role": role,
                "prompt_format": prompt_format,
                "selection_mode": mode,
                "adapter_mode": adapter_mode,
                "variant": f"{slot}_{mode}{b}",
                "recent_frames_kept": sum(
                    1
                    for step in rendered[mode]["selected_steps"]
                    if step in full_window
                ),
                "memory_config": {
                    "restored_event_step_ids": rendered[mode]["selected_steps"]
                },
                **{
                    key: rendered[mode][key]
                    for key in (
                        "selected_steps", "selected_images",
                        "messages", "messages_sha256",
                    )
                },
            }
            if slot == NEGATIVE_ARM_SLOT:
                row["negative_kind"] = NEGATIVE_KIND
                row["negative_scale"] = NEGATIVE_SCALE
                row["donor_episode"] = episode
                row["distractor_source_step"] = wrong_event
                row["oracle_source_step"] = positive_event
            rows.append(row)
        if emit_parity:
            b0_arm_id, b0_role, b0_format, b0_mode, b0_adapter = B0_ARM_CONTRACT_V2
            parity_rows.append(
                {
                    "schema_version": B0_SAMPLE_SCHEMA_V2,
                    "sample_id": f"agentnet:{dp}|B0",
                    **common,
                    "pair_group": f"agentnet:{dp}:b0",
                    "budget": 0,
                    "arm_slot": "B0",
                    "arm_id": b0_arm_id,
                    "role": b0_role,
                    "prompt_format": b0_format,
                    "selection_mode": b0_mode,
                    "adapter_mode": b0_adapter,
                    "variant": "B0_none0",
                    "recent_frames_kept": 0,
                    "memory_config": {"restored_event_step_ids": []},
                    **{
                        key: rendered["none"][key]
                        for key in (
                            "selected_steps", "selected_images",
                            "messages", "messages_sha256",
                        )
                    },
                }
            )
        for mode in selection_specs:
            referenced.update(rendered[mode]["selected_images"])
        referenced.add(current_path)
        positive_ages.append(current_step - positive_event)
        counters["groups"] += 1
        counters[f"groups_{split}"] += 1
        counters[
            "groups_" + record["target_tool_call"]["arguments"]["action"]
        ] += 1
    stats = {
        "counters": dict(sorted(counters.items())),
        "positive_age": _percentile_summary(positive_ages) if positive_ages else {},
        "referenced_paths": sorted(referenced),
    }
    return rows, parity_rows, stats


def merge_outputs(output_root: Path) -> None:
    partial_paths = sorted(output_root.glob("manifest-b*.json"))
    if not partial_paths:
        raise SystemExit("no manifest-b*.json found; run per-B builds first")
    partials = {
        int(path.stem.split("-b")[1]): json.loads(path.read_text(encoding="utf-8"))
        for path in partial_paths
    }
    b_values = sorted(partials)
    train_bs = [b for b in b_values if b in TRAIN_B_VALUES]
    if not train_bs:
        raise SystemExit("no training-eligible B levels present")
    combined = output_root / "samples.jsonl"
    with combined.open("w", encoding="utf-8") as handle:
        for b in train_bs:
            handle.write(
                (output_root / f"samples-b{b}.jsonl").read_text(encoding="utf-8")
            )
    manifest = {
        "schema_version": CORPUS_SCHEMA,
        "sample_schema": SAMPLE_SCHEMA_V2,
        "b0_sample_schema": B0_SAMPLE_SCHEMA_V2,
        "prompt_format": PROMPT_FORMAT_V2,
        "renderer": DESKTOP_OFFICIAL_PROTOCOL_ID,
        "recent_definition": "distinct_frames_v2",
        "budget_semantics": "fixed_budget_replacement_v3",
        "seed": partials[b_values[0]]["seed"],
        "coordinate_tolerance": partials[b_values[0]]["coordinate_tolerance"],
        "b_values_built": b_values,
        "b_values_trained": train_bs,
        "per_b": {str(b): partials[b]["per_b"][str(b)] for b in b_values},
        "inputs": partials[b_values[0]]["inputs"],
        "samples_sha256": sha256_file(combined),
        "sample_count": sum(
            partials[b]["per_b"][str(b)]["sample_count"] for b in train_bs
        ),
        "group_count": sum(
            partials[b]["per_b"][str(b)]["counters"].get("groups", 0)
            for b in train_bs
        ),
    }
    parity = output_root / "parity_b0.jsonl"
    if parity.exists():
        manifest["parity_b0_sha256"] = sha256_file(parity)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {"merged_train_b": train_bs, "groups": manifest["group_count"],
         "samples": manifest["sample_count"],
         "samples_sha256": manifest["samples_sha256"]},
        ensure_ascii=False, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.merge:
        merge_outputs(args.output_root)
        return
    unknown = sorted(set(args.b_values) - set(B_VALUES))
    if unknown:
        raise SystemExit(f"unsupported B values {unknown}; frozen set is {B_VALUES}")
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("input manifest is empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    referenced: set[str] = set()
    for b in args.b_values:
        emit_parity = b == 1
        rows, parity_rows, stats = build_b_groups(
            records,
            b=b,
            coordinate_tolerance=args.coordinate_tolerance,
            seed=args.seed,
            max_groups=args.max_groups_per_b,
            emit_parity=emit_parity,
        )
        samples_path = args.output_root / f"samples-b{b}.jsonl"
        with samples_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        if emit_parity:
            with (args.output_root / "parity_b0.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for row in parity_rows:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
        referenced.update(stats.pop("referenced_paths"))
        partial = {
            "schema_version": CORPUS_SCHEMA,
            "sample_schema": SAMPLE_SCHEMA_V2,
            "prompt_format": PROMPT_FORMAT_V2,
            "renderer": DESKTOP_OFFICIAL_PROTOCOL_ID,
            "recent_definition": "distinct_frames_v2",
            "budget_semantics": "fixed_budget_replacement_v3",
            "seed": args.seed,
            "coordinate_tolerance": args.coordinate_tolerance,
            "per_b": {
                str(b): {
                    **stats,
                    "sample_count": len(rows),
                    "parity_count": len(parity_rows),
                    "samples_sha256": sha256_file(samples_path),
                }
            },
            "referenced_image_count": len(referenced),
            "inputs": {
                "manifest": str(args.manifest),
                "manifest_sha256": sha256_file(args.manifest),
                "image_root": str(args.image_root),
            },
        }
        (args.output_root / f"manifest-b{b}.json").write_text(
            json.dumps(partial, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(
            {"b": b, "groups": stats["counters"].get("groups", 0),
             "dropped_retained_unrenderable": stats["counters"].get(
                 "dropped_retained_unrenderable", 0)},
            sort_keys=True,
        ), flush=True)
    if args.link_images:
        _link_image_roots(
            sorted(referenced),
            image_root=args.image_root,
            output_root=args.output_root,
        )
    print(json.dumps({"built_b": list(args.b_values)}, sort_keys=True))


if __name__ == "__main__":
    main()
