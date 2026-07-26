#!/usr/bin/env python3
"""Build the fixed-budget replacement corpus v3 (frozen claim experiment).

# note (luojiaxuan): v3 冻结口径(2026-07-27,用户确认):
#   * 固定高保真历史图预算 B(不含当前图),问题 = 预算内重分配,不是加图:
#       R0/RA = Recent-B(最近 B 张**不同帧**,事件 s-2 .. s-1-B;部署选择器
#               会选中与当前截图同帧的 s-1,该重复帧现象单独披露,不进语料);
#       S0/SA = Recent-(B-1) + target-recurrence 正例旧帧
#               (**替换最老的 recent 槽位** s-1-B,冻结规则);
#       WA    = Recent-(B-1) + age-matched wrong 旧帧(同一被替换槽位,
#               W0 由 trainer bypass 重算);
#   * 正/错旧帧 age ≥ B+2:严格老于整个 Recent-B 窗口;B=1 退化为
#     真前一帧探针口径的 v1(R={s-2},S={age≥3 正例});
#   * B ∈ {1,2,4} 进训练合并(samples.jsonl);B=8 只建不训(评测外推),
#     单独留在 samples-b8.jsonl;B=0 仅 parity 行(零恢复图);
#   * k(选中集中来自窗口外的图数)是**结果统计量**:训练只教 k=1 替换接口,
#     k>1 组合由 selector(候选=完整历史含 recent 帧,at-most-B + STOP)产生;
#   * 行结构保持 trainer 契约 schema 不变;pair_group 带 :b 后缀;split 沿用
#     轨迹 hash(同 seed,跨 B 一致)。
# 主因果比较:每 B 行内 matched budget 的 Selected-vs-Recent;跨 B 只是分析。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.build_desktop_did_corpus_v2 import recent_window
from scripts.build_desktop_hgkv_corpus import (
    B0_ARM_CONTRACT,
    B0_SAMPLE_SCHEMA,
    DESKTOP_ARM_CONTRACT,
    NEGATIVE_ARM_SLOT,
    NEGATIVE_KIND,
    NEGATIVE_SCALE,
    SAMPLE_SCHEMA,
    _link_image_roots,
    _load_screenshots,
    _messages_for_selection,
    _percentile_summary,
    _split,
    messages_sha256,
    select_contrast_events,
    sha256_file,
)

CORPUS_SCHEMA = "causalcache.desktop_did_corpus_b.v1"
B_VALUES = (1, 2, 4, 8)
TRAIN_B_VALUES = (1, 2, 4)


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
    parser.add_argument(
        "--merge", action="store_true",
        help="合并模式:samples.jsonl = concat B∈{1,2,4};B=8 留在独立文件;"
             "B=1 分片同时产 parity_b0.jsonl(零图行)并写总 manifest.json",
    )
    return parser.parse_args()


def replaced_slot_event(current_step: int, b: int) -> int:
    """被替换的最老 recent 槽位(冻结规则)。"""
    return current_step - 1 - b


def eligible_events_b(
    record: Mapping[str, Any], *, b: int, coordinate_tolerance: int, seed: int
) -> tuple[int, int, int] | None:
    if replaced_slot_event(int(record["step"]), b) < 1:
        return None
    return select_contrast_events(
        record,
        coordinate_tolerance=coordinate_tolerance,
        min_age=b + 2,
        seed=seed,
    )


def build_b_groups(
    records: Sequence[dict[str, Any]],
    *,
    image_root: Path,
    b: int,
    coordinate_tolerance: int,
    seed: int,
    max_groups: int = 0,
    emit_parity: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """One budget level: (six-arm rows, B=0 parity rows, stats)."""
    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    referenced: set[str] = set()
    positive_ages: list[int] = []
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
        screenshots = _load_screenshots(record, image_root=image_root)
        current_path = relpaths[current_step - 1]
        pair_group = f"agentnet:{record['dp_id']}:b{b}"
        episode = str(record["task_id"])
        split = _split(episode, seed=seed)

        selection_specs = {
            "recent": list(full_window),
            "recurrence": sorted([*kept_window, positive_event]),
            "wrong": sorted([*kept_window, wrong_event]),
        }
        if emit_parity:
            selection_specs["none"] = []
        rendered: dict[str, dict[str, Any]] = {}
        for mode, steps in selection_specs.items():
            selected_images = [relpaths[step] for step in steps]
            messages = _messages_for_selection(
                record,
                screenshots=screenshots,
                recent_budget=0,
                extra_restored=steps,
                expected_steps=steps,
                image_paths=[*selected_images, current_path],
            )
            rendered[mode] = {
                "selected_steps": steps,
                "selected_images": selected_images,
                "messages": messages,
                "messages_sha256": messages_sha256(messages),
            }

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
            "target_text": record["target_text"],
            "reference_arm_id": "R0",
            "deployment_baseline_arm_id": "R0",
            "source": {
                "dataset": "AgentNet/OpenCUA",
                "os": record["os"],
                "dp_id": record["dp_id"],
                "positive_rule": "full_action_recurrence_pre_state",
                "positive_action_step": positive_action_step,
                "positive_event_step": positive_event,
                "wrong_event_step": wrong_event,
                "positive_age": current_step - positive_event,
                "wrong_age": current_step - wrong_event,
                "recent_definition": "distinct_frames_v2",
                "budget_semantics": "fixed_budget_replacement_v3",
            },
        }
        for slot, (arm_id, role, prompt_format, mode, adapter_mode) in (
            DESKTOP_ARM_CONTRACT.items()
        ):
            row = {
                "schema_version": SAMPLE_SCHEMA,
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
            b0_arm_id, b0_role, b0_format, b0_mode, b0_adapter = B0_ARM_CONTRACT
            parity_rows.append(
                {
                    "schema_version": B0_SAMPLE_SCHEMA,
                    "sample_id": f"agentnet:{record['dp_id']}|B0",
                    **common,
                    "pair_group": f"agentnet:{record['dp_id']}:b0",
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
    """确定性合并:训练集只并 B∈{1,2,4};B=8 留独立文件;写总 manifest。"""
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
        "sample_schema": SAMPLE_SCHEMA,
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
        emit_parity = b == 1  # B=1 分片顺带产零图 parity 行(合格 dp 的超集)
        rows, parity_rows, stats = build_b_groups(
            records,
            image_root=args.image_root,
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
            "sample_schema": SAMPLE_SCHEMA,
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
            {"b": b, "groups": stats["counters"].get("groups", 0)}, sort_keys=True
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
