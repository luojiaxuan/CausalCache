#!/usr/bin/env python3
"""Build the r-conditioned Desktop DiD corpus v2 (frozen claim experiment).

# note (luojiaxuan): v2 冻结口径(2026-07-27,README『主线』段):
#   * `C_r` = instruction + 完整动作 summaries + 当前截图 + Recent-r,其中
#     Recent-r = 最近 r 张**不同帧**(事件 s-2 .. s-1-r 的 post;部署选择器
#     k=1 会选中与当前截图同帧的事件 s-1,该重复帧现象单独披露,不进 v2 语料);
#   * 每组只分配一个新增高保真槽位,四臂共享 C_r:
#       R0/RA  = C_r + next-recent(事件 s-2-r 的 post),bypass/active 孪生;
#       S0/SA  = C_r + target-recurrence 正例旧帧;
#       WA     = C_r + age-matched wrong 旧帧(W0 由 trainer bypass 重算);
#   * 正例与 wrong 都要求 age ≥ r+3:严格老于 recent 窗口与 next-recent 候选,
#     r=0 时退化为 v1 的 min_age=3;
#   * 每组另出一行 Base(C_r 本身,budget=r)进 base_cr.jsonl —— 它是将来
#     selector Δ 标签与 G_restore(r) 的基准,不进 DiD 训练;r=0 的 Base(零恢复
#     图)同时写 parity_b0.jsonl 供 HGKV bitwise parity;
#   * 行结构与 trainer 契约(causalcache.desktop_did_sample.v1)不变,新增
#     `recent_r`/`base_steps`/`next_recent_step` 字段;pair_group 带 r 后缀;
#     split 沿用 v1 的轨迹 hash(同 seed → 同轨迹归属,跨 r 一致)。
# 跨 r 的绝对分数只是分析;主因果比较是每行内 matched-budget 的
# selected-vs-recent(见 paper Table rdesign)。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.build_desktop_hgkv_corpus import (
    B0_ARM_CONTRACT,
    B0_SAMPLE_SCHEMA,
    DESKTOP_ARM_CONTRACT,
    NEGATIVE_ARM_SLOT,
    NEGATIVE_KIND,
    NEGATIVE_SCALE,
    PROMPT_FORMAT,
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

CORPUS_SCHEMA = "causalcache.desktop_did_corpus_r.v1"
BASE_SAMPLE_SCHEMA = "causalcache.desktop_did_base.v1"
R_VALUES = (0, 1, 2, 4, 8)
BASE_ARM_CONTRACT = ("BASE", "cr_baseline", PROMPT_FORMAT, "cr_base", "bypass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--r-values", type=int, nargs="+", default=list(R_VALUES))
    parser.add_argument("--coordinate-tolerance", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-groups-per-r", type=int, default=0)
    parser.add_argument(
        "--link-images", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="合并模式:读取 output-root 下已有的 manifest-r*.json,拼 combined "
             "samples.jsonl / base_cr.jsonl / parity_b0.jsonl 并写总 manifest.json",
    )
    return parser.parse_args()


def merge_outputs(output_root: Path) -> None:
    """并行 per-r 构建后的确定性合并(升序 r 拼接,CPU 秒级)。"""
    partial_paths = sorted(output_root.glob("manifest-r*.json"))
    if not partial_paths:
        raise SystemExit("no manifest-r*.json found; run per-r builds first")
    partials = {
        int(path.stem.split("-r")[1]): json.loads(path.read_text(encoding="utf-8"))
        for path in partial_paths
    }
    r_values = sorted(partials)
    combined = output_root / "samples.jsonl"
    base_combined = output_root / "base_cr.jsonl"
    with combined.open("w", encoding="utf-8") as samples_handle, \
            base_combined.open("w", encoding="utf-8") as base_handle:
        for r in r_values:
            samples_handle.write(
                (output_root / f"samples-r{r}.jsonl").read_text(encoding="utf-8")
            )
            base_handle.write(
                (output_root / f"base-r{r}.jsonl").read_text(encoding="utf-8")
            )
    if 0 in r_values:
        (output_root / "parity_b0.jsonl").write_text(
            (output_root / "base-r0.jsonl").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    first = partials[r_values[0]]
    manifest = {
        "schema_version": CORPUS_SCHEMA,
        "sample_schema": SAMPLE_SCHEMA,
        "recent_definition": "distinct_frames_v2",
        "seed": first["seed"],
        "coordinate_tolerance": first["coordinate_tolerance"],
        "r_values": r_values,
        "per_r": {str(r): partials[r]["per_r"][str(r)] for r in r_values},
        "inputs": first["inputs"],
        "referenced_image_count": None,
        "samples_sha256": sha256_file(combined),
        "base_cr_sha256": sha256_file(base_combined),
        "sample_count": sum(
            partials[r]["per_r"][str(r)]["sample_count"] for r in r_values
        ),
        "group_count": sum(
            partials[r]["per_r"][str(r)]["counters"].get("groups", 0)
            for r in r_values
        ),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {"merged": r_values, "groups": manifest["group_count"],
         "samples": manifest["sample_count"],
         "samples_sha256": manifest["samples_sha256"]},
        ensure_ascii=False, sort_keys=True))


def recent_window(current_step: int, r: int) -> list[int]:
    """事件 s-1-r .. s-2 的升序列表 —— 最近 r 张不同帧(不含同帧的 s-1)。"""
    return [current_step - 1 - i for i in range(r, 0, -1)]


def next_recent_event(current_step: int, r: int) -> int:
    return current_step - 2 - r


def eligible_events(
    record: Mapping[str, Any], *, r: int, coordinate_tolerance: int, seed: int
) -> tuple[int, int, int] | None:
    """Return (positive_event, positive_action_step, wrong_event) for one r, or None.

    # note (luojiaxuan): 直接复用 v1 的 select_contrast_events,把 min_age 提到
    # r+3 —— 正/错旧帧都严格老于 recent 窗口与 next-recent 候选;r=0 退化为 v1。
    """
    if next_recent_event(int(record["step"]), r) < 1:
        return None
    return select_contrast_events(
        record,
        coordinate_tolerance=coordinate_tolerance,
        min_age=r + 3,
        seed=seed,
    )


def build_r_groups(
    records: Sequence[dict[str, Any]],
    *,
    image_root: Path,
    r: int,
    coordinate_tolerance: int,
    seed: int,
    max_groups: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """One r level: (six-arm rows, base-C_r rows, stats)."""
    rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    referenced: set[str] = set()
    positive_ages: list[int] = []
    budget = r + 1
    for record in records:
        if max_groups and counters["groups"] >= max_groups:
            break
        selected = eligible_events(
            record, r=r, coordinate_tolerance=coordinate_tolerance, seed=seed
        )
        if selected is None:
            counters["rejected"] += 1
            continue
        positive_event, positive_action_step, wrong_event = selected
        current_step = int(record["step"])
        base_steps = recent_window(current_step, r)
        next_recent = next_recent_event(current_step, r)
        relpaths = record["image_relpaths"]
        screenshots = _load_screenshots(record, image_root=image_root)
        current_path = relpaths[current_step - 1]
        pair_group = f"agentnet:{record['dp_id']}:r{r}"
        episode = str(record["task_id"])
        split = _split(episode, seed=seed)

        selection_specs = {
            "recent": sorted([*base_steps, next_recent]),
            "recurrence": sorted([*base_steps, positive_event]),
            "wrong": sorted([*base_steps, wrong_event]),
            "cr_base": list(base_steps),
        }
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
            "budget": budget,
            "recent_r": r,
            "base_steps": list(base_steps),
            "next_recent_step": next_recent,
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
                "variant": f"{slot}_{mode}{budget}",
                "recent_frames_kept": sum(
                    1
                    for step in rendered[mode]["selected_steps"]
                    if step >= current_step - budget
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
        base_arm_id, base_role, base_format, base_mode, base_adapter = (
            B0_ARM_CONTRACT if r == 0 else BASE_ARM_CONTRACT
        )
        base_rows.append(
            {
                "schema_version": B0_SAMPLE_SCHEMA if r == 0 else BASE_SAMPLE_SCHEMA,
                "sample_id": f"{pair_group}|{base_arm_id}",
                **common,
                "budget": r,
                "arm_slot": base_arm_id,
                "arm_id": base_arm_id,
                "role": base_role,
                "prompt_format": base_format,
                "selection_mode": base_mode,
                "adapter_mode": base_adapter,
                "variant": f"{base_arm_id}_{base_mode}{r}",
                "recent_frames_kept": len(base_steps),
                "memory_config": {"restored_event_step_ids": list(base_steps)},
                **{
                    key: rendered["cr_base"][key]
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
    return rows, base_rows, stats


def main() -> None:
    args = parse_args()
    if args.merge:
        merge_outputs(args.output_root)
        return
    unknown = sorted(set(args.r_values) - set(R_VALUES))
    if unknown:
        raise SystemExit(f"unsupported r values {unknown}; frozen set is {R_VALUES}")
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("input manifest is empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    referenced: set[str] = set()
    per_r: dict[str, Any] = {}
    for r in args.r_values:
        rows, base_rows, stats = build_r_groups(
            records,
            image_root=args.image_root,
            r=r,
            coordinate_tolerance=args.coordinate_tolerance,
            seed=args.seed,
            max_groups=args.max_groups_per_r,
        )
        samples_path = args.output_root / f"samples-r{r}.jsonl"
        base_path = args.output_root / f"base-r{r}.jsonl"
        for path, payload in ((samples_path, rows), (base_path, base_rows)):
            with path.open("w", encoding="utf-8") as handle:
                for row in payload:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
        referenced.update(stats.pop("referenced_paths"))
        per_r[str(r)] = {
            **stats,
            "sample_count": len(rows),
            "base_count": len(base_rows),
            "samples_sha256": sha256_file(samples_path),
            "base_sha256": sha256_file(base_path),
        }
        print(
            json.dumps(
                {"r": r, "groups": stats["counters"].get("groups", 0)},
                sort_keys=True,
            ),
            flush=True,
        )
    linked: list[str] = []
    if args.link_images:
        linked = _link_image_roots(
            sorted(referenced),
            image_root=args.image_root,
            output_root=args.output_root,
        )
    # 每个 build 进程写自己的 manifest-r*.json;combined manifest 由 --merge 产生,
    # 并行 per-r 构建因此不会互相覆盖共享文件。
    for r in args.r_values:
        partial = {
            "schema_version": CORPUS_SCHEMA,
            "sample_schema": SAMPLE_SCHEMA,
            "recent_definition": "distinct_frames_v2",
            "seed": args.seed,
            "coordinate_tolerance": args.coordinate_tolerance,
            "per_r": {str(r): per_r[str(r)]},
            "referenced_image_count": len(referenced),
            "inputs": {
                "manifest": str(args.manifest),
                "manifest_sha256": sha256_file(args.manifest),
                "image_root": str(args.image_root),
                "linked_image_roots": linked,
            },
        }
        (args.output_root / f"manifest-r{r}.json").write_text(
            json.dumps(partial, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"built_r": list(args.r_values)}, sort_keys=True))


if __name__ == "__main__":
    main()
