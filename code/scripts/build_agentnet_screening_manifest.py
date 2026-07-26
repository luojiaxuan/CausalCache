#!/usr/bin/env python3
"""构建 AgentNet 桌面决策点 screening manifest(第一批:ubuntu,20K 决策点)。

# note (luojiaxuan): 过滤与抽样契约(与协调人 2026-07-26 指令逐条对应):
#   * 轨迹级:``task_completed is True`` 且步数 >= --min-traj-steps(默认 10);
#   * 决策点级:1-based step s >= --min-step(默认 6,历史候选池至少 5 帧)、
#     target 步 ``value.last_step_correct is True``、target 可解析(parser 双形态
#     齐备)、且 1..s-1 每一步 history 形态可解析(否则文字历史不忠实,整点弃用);
#   * ``last_step_redundant`` 只记录不剔;同时记录 s+1 步的 last_step_correct
#     (OpenCUA 的 reflector 语义下,step i 的该字段描述的是 step i-1 的动作,
#     故 target 动作本身的"正确性"其实写在 s+1 步 —— 两种口径都入库,报告里对账);
#   * 每轨迹最多 --max-per-traj(默认 3)个决策点,按轨迹内三等分段各取一个
#     (seeded RNG),防单轨迹主导;
#   * 分层:OS x 轨迹长度三档(tertile 由过滤后分布现算);超出 --target-count 时
#     按层比例 seeded 下采样。
# 产出:manifest JSONL(每行一个决策点,含 r 值列表、图片相对路径、结构化动作
# 历史、target tool_call 与 teacher-forced 文本)+ summary JSON(规模、分层、
# 解析覆盖率统计)。不做图片存在性校验 —— 那是下载完成后的抽验脚本参数
# ``--check-images`` 的事。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from causalcache.agentnet_actions import parse_agentnet_step, render_target_text

MANIFEST_SCHEMA = "causalcache.agentnet_screening_manifest.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-jsonl", type=Path, required=True, action="append",
                        help="AgentNet 轨迹 JSONL(可重复;第一批只传 ubuntu)")
    parser.add_argument("--meta-jsonl", type=Path, required=True,
                        help="meta_data_merged.jsonl,提供 task 级 screen_width/height")
    parser.add_argument("--image-subdir", action="append", required=True,
                        help="与 --trajectory-jsonl 一一对应的图片相对目录,如 "
                             "ubuntu_images/extracted/images")
    parser.add_argument("--os-label", action="append", required=True,
                        help="与 --trajectory-jsonl 一一对应的 OS 分层标签,如 ubuntu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=20000)
    parser.add_argument("--r-values", default="0,2")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-per-traj", type=int, default=3)
    parser.add_argument("--min-step", type=int, default=6)
    parser.add_argument("--min-traj-steps", type=int, default=10)
    parser.add_argument("--image-root", type=Path, default=None,
                        help="传入时逐决策点校验 png 存在性(下载完成后用)")
    parser.add_argument("--check-images", type=int, default=0,
                        help="抽验的决策点数量;0 表示全量(仅当 --image-root 给出)")
    return parser.parse_args()


def load_meta(path: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            meta[record["task_id"]] = record
    return meta


def iter_trajectories(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def spread_pick(eligible: list[int], *, cap: int, rng: random.Random) -> list[int]:
    """轨迹内三等分段各取一个决策点,防同轨迹决策点扎堆。"""
    if len(eligible) <= cap:
        return list(eligible)
    picked: list[int] = []
    bins = [eligible[(i * len(eligible)) // cap:((i + 1) * len(eligible)) // cap]
            for i in range(cap)]
    for chunk in bins:
        if chunk:
            picked.append(chunk[rng.randrange(len(chunk))])
    return sorted(set(picked))


def length_bin(length: int, tertiles: tuple[int, int]) -> str:
    if length <= tertiles[0]:
        return "short"
    if length <= tertiles[1]:
        return "medium"
    return "long"


def main() -> None:
    args = parse_args()
    if not (len(args.trajectory_jsonl) == len(args.image_subdir) == len(args.os_label)):
        raise SystemExit("--trajectory-jsonl/--image-subdir/--os-label 数量必须一致")
    r_values = sorted({int(part) for part in args.r_values.split(",")})
    if any(r < 0 for r in r_values):
        raise SystemExit("--r-values 必须非负")
    meta = load_meta(args.meta_jsonl)
    rng = random.Random(args.seed)

    counters: Counter[str] = Counter()
    step_reason_counter: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    filtered_lengths: list[int] = []

    for source, subdir, os_label in zip(
        args.trajectory_jsonl, args.image_subdir, args.os_label
    ):
        for record in iter_trajectories(source):
            counters["trajectories_seen"] += 1
            traj = record.get("traj") or []
            task_id = record.get("task_id")
            if record.get("task_completed") is not True:
                counters["traj_dropped_not_completed"] += 1
                continue
            if len(traj) < args.min_traj_steps:
                counters["traj_dropped_too_short"] += 1
                continue
            meta_record = meta.get(task_id)
            if meta_record is None:
                counters["traj_dropped_missing_meta"] += 1
                continue
            width = meta_record.get("screen_width")
            height = meta_record.get("screen_height")
            if type(width) is not int or type(height) is not int or min(width, height) <= 0:
                # note (luojiaxuan): 格式意外 —— ubuntu 5K 的 meta 里
                # screen_width/height 全是 null(win_mac 才有值)。回退到直接读
                # 该轨迹第一帧 PNG 的头部拿真实分辨率;读不到才丢弃。target 坐标
                # 是 [0,999] 归一化,不依赖分辨率;分辨率只进 history 像素文本,
                # PNG 实测值反而比 meta 更可信。
                if args.image_root is None:
                    counters["traj_dropped_bad_screen_size"] += 1
                    continue
                first_image = args.image_root / subdir / traj[0]["image"]
                try:
                    from PIL import Image

                    with Image.open(first_image) as probe:
                        width, height = probe.size
                    counters["traj_screen_size_from_png"] += 1
                except (OSError, ValueError):
                    counters["traj_dropped_bad_screen_size"] += 1
                    continue
            screen_size = (width, height)
            instruction = record.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                counters["traj_dropped_missing_instruction"] += 1
                continue
            counters["trajectories_kept"] += 1
            filtered_lengths.append(len(traj))

            parsed = []
            for step in traj:
                value = step.get("value") or {}
                parsed_step = parse_agentnet_step(
                    value.get("code", ""), screen_size=screen_size
                )
                parsed.append((step, value, parsed_step))
                counters["steps_seen"] += 1
                if not parsed_step.parseable:
                    counters["steps_unparseable"] += 1
                    step_reason_counter[parsed_step.unparseable_reason] += 1
                elif not parsed_step.target_available:
                    counters["steps_history_only"] += 1
                    step_reason_counter[parsed_step.target_unavailable_reason] += 1

            # note (luojiaxuan): prefix_ok[j] == 前 j 步(1-based 1..j)history 形态
            # 全部可解析;决策点 s 要求 prefix_ok[s-1],否则文字历史会缺步。
            prefix_ok = [True]
            for _, _, parsed_step in parsed:
                prefix_ok.append(prefix_ok[-1] and parsed_step.parseable)

            eligible: list[int] = []
            for s in range(args.min_step, len(traj) + 1):
                step, value, parsed_step = parsed[s - 1]
                if not parsed_step.target_available:
                    counters["dp_dropped_target_unparseable"] += 1
                    continue
                if value.get("last_step_correct") is not True:
                    counters["dp_dropped_last_step_correct"] += 1
                    continue
                if not prefix_ok[s - 1]:
                    counters["dp_dropped_history_unparseable"] += 1
                    continue
                eligible.append(s)
            counters["dp_eligible_before_cap"] += len(eligible)

            for s in spread_pick(eligible, cap=args.max_per_traj, rng=rng):
                step, value, parsed_step = parsed[s - 1]
                next_value = parsed[s][1] if s < len(traj) else None
                history = [
                    {
                        "step_id": j,
                        "action": parsed[j - 1][2].history_action,
                        "osworld_action": parsed[j - 1][2].osworld_action,
                        "raw_code": parsed[j - 1][2].raw_code,
                    }
                    for j in range(1, s)
                ]
                image_relpaths = [
                    f"{subdir}/{parsed[i][0]['image']}" for i in range(s)
                ]
                dp_id = hashlib.sha256(
                    f"{task_id}|{s}".encode("utf-8")
                ).hexdigest()[:20]
                candidates.append(
                    {
                        "schema_version": MANIFEST_SCHEMA,
                        "dp_id": dp_id,
                        "source_file": source.name,
                        "os": os_label,
                        "task_id": task_id,
                        "instruction": instruction,
                        "step": s,
                        "traj_len": len(traj),
                        "screen_size": [width, height],
                        "r_values": r_values,
                        "image_relpaths": image_relpaths,
                        "history": history,
                        "target_tool_call": parsed_step.target_tool_call,
                        "target_text": render_target_text(parsed_step.target_tool_call),
                        "target_raw_code": parsed_step.raw_code,
                        "target_is_terminal": (
                            parsed_step.target_tool_call["arguments"]["action"]
                            == "terminate"
                        ),
                        "target_step_last_step_correct": value.get("last_step_correct"),
                        "target_step_last_step_redundant": value.get(
                            "last_step_redundant"
                        ),
                        "next_step_last_step_correct": (
                            None if next_value is None
                            else next_value.get("last_step_correct")
                        ),
                        "next_step_last_step_redundant": (
                            None if next_value is None
                            else next_value.get("last_step_redundant")
                        ),
                    }
                )

    if not candidates:
        raise SystemExit("过滤后没有任何决策点;检查输入与过滤参数")

    lengths = sorted(filtered_lengths)
    tertiles = (
        lengths[len(lengths) // 3],
        lengths[(2 * len(lengths)) // 3],
    )
    for candidate in candidates:
        candidate["length_bin"] = length_bin(candidate["traj_len"], tertiles)
        candidate["stratum"] = f"{candidate['os']}|{candidate['length_bin']}"

    strata: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        strata.setdefault(candidate["stratum"], []).append(candidate)

    total = len(candidates)
    if total > args.target_count:
        # note (luojiaxuan): 按层比例配额下采样(最大余数法凑满 target),层内
        # seeded shuffle 后截断,保证同 seed 复现。
        quotas: dict[str, int] = {}
        remainders: list[tuple[float, str]] = []
        assigned = 0
        for name, members in sorted(strata.items()):
            exact = args.target_count * len(members) / total
            quotas[name] = int(exact)
            assigned += int(exact)
            remainders.append((exact - int(exact), name))
        for _, name in sorted(remainders, reverse=True)[: args.target_count - assigned]:
            quotas[name] += 1
        selected: list[dict[str, Any]] = []
        for name, members in sorted(strata.items()):
            members = sorted(members, key=lambda c: c["dp_id"])
            rng.shuffle(members)
            selected.extend(members[: quotas[name]])
    else:
        selected = candidates

    selected.sort(key=lambda c: c["dp_id"])

    image_check = None
    if args.image_root is not None:
        pool = selected
        if args.check_images and args.check_images < len(selected):
            pool = rng.sample(selected, args.check_images)
        missing: list[str] = []
        checked_files = 0
        for candidate in pool:
            for relpath in candidate["image_relpaths"]:
                checked_files += 1
                if not (args.image_root / relpath).is_file():
                    missing.append(relpath)
        image_check = {
            "decision_points_checked": len(pool),
            "files_checked": checked_files,
            "files_missing": len(missing),
            "missing_examples": missing[:20],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for candidate in selected:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    per_stratum = Counter(c["stratum"] for c in selected)
    per_step_bucket = Counter(
        "terminal" if c["target_is_terminal"] else c["target_tool_call"]["arguments"]["action"]
        for c in selected
    )
    summary = {
        "schema_version": MANIFEST_SCHEMA + ".summary",
        "seed": args.seed,
        "r_values": r_values,
        "tertiles": list(tertiles),
        "counters": dict(counters),
        "step_reasons": dict(step_reason_counter),
        "candidates_before_downsample": total,
        "decision_points": len(selected),
        "forwards_expected": len(selected) * len(r_values),
        "per_stratum": dict(per_stratum),
        "target_action_distribution": dict(per_step_bucket),
        "redundant_recorded_not_dropped": sum(
            1 for c in selected if c["target_step_last_step_redundant"] is True
        ),
        "next_step_annotation_would_drop": sum(
            1 for c in selected if c["next_step_last_step_correct"] is False
        ),
        "image_check": image_check,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
