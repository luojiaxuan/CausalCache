#!/usr/bin/env python3
"""Build Desktop DiD six-arm history-contrast groups from AgentNet successes.

# note (luojiaxuan): 交接 §6 冻结的 DiD 六臂目标。每个训练组物化五行样本:
#   R0/RA — Recent-k(冻结部署选择器 select_osworld_memory(arm="recent")),
#           bypass/active 孪生,消息逐字节相同;
#   S0/SA — target-specific selected-k(同轨迹完整等价旧动作的执行前画面,
#           即前一事件的 post screenshot),bypass/active 孪生;
#   WA    — 同轨迹、age 尽量匹配、下一动作与 a* 不等价的 wrong-k,只发 active 行;
#           W0 由 trainer 对同一行临时 bypass 重算(train_success_sft_lora 的
#           did_ra_aware 负样本路径),不在语料里重复一行。
# B0(无恢复图)另写 parity_b0.jsonl:它只用于 HGKV bitwise parity 与总体能力
# 报告,不得混入内容选择主差值,schema 独立以便误拼接时 fail-closed。
# irrelevant 跨轨迹 donor 臂不再进训练组:§6 的 L_content/L_cap 只消费 wrong,
# harm 对照如 §11.1 需要,可在评测阶段以同一 seed 另行物化(v1 四臂构造见 git 历史)。
# split 按轨迹 hash 发 train/dev/test(80/10/10);trainer 侧的 dev→heldout 映射与
# test 剥离属于 osworld_official 接入(交接 §8.2-4),在语料里不做。
# 正例不以 frozen policy 做错为前置(information addition 与 evidence
# amplification 两种机制都允许,见交接 §2)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.agentnet_desktop_cr import (
    build_agentnet_cr_request,
    render_agentnet_cr_messages,
)

CORPUS_SCHEMA = "causalcache.desktop_hgkv_corpus.v2"
SAMPLE_SCHEMA = "causalcache.desktop_did_sample.v1"
B0_SAMPLE_SCHEMA = "causalcache.desktop_did_b0_sample.v1"
PROMPT_FORMAT = "osworld_official"
BUDGET = 1
REFERENCE_ARM_ID = "R0"
# note (luojiaxuan): 桌面的部署 prompt 与 R0 出自同一冻结 renderer + 同一 recent
# 选择器,不存在 GUI-Odyssey v6 那种 official/sparse 双格式,所以不发 N0 臂——
# 一条与 R0 逐字节相同的"部署基线"不携带任何信息,format_effect 恒等于 0 只会
# 伪装成一次测量。deployment_baseline_arm_id 显式指向 R0,trainer 契约按此校验。
DEPLOYMENT_BASELINE_ARM_ID = "R0"
NEGATIVE_ARM_SLOT = "WA"
NEGATIVE_KIND = "wrong"
NEGATIVE_SCALE = 1.0
# arm_slot -> (arm_id, role, prompt_format, selection_mode, adapter_mode)
# 与 trainer 的 SPARSE_ARM_CONTRACTS 新增项(交接 §8.2-4)保持逐字段一致。
DESKTOP_ARM_CONTRACT: dict[str, tuple[str, str, str, str, str]] = {
    "R0": ("R0", "reference", PROMPT_FORMAT, "recent", "bypass"),
    "RA": ("RA", "measurement", PROMPT_FORMAT, "recent", "active"),
    "S0": ("S0", "measurement", PROMPT_FORMAT, "recurrence", "bypass"),
    "SA": ("SA", "positive", PROMPT_FORMAT, "recurrence", "active"),
    NEGATIVE_ARM_SLOT: ("WA", "negative", PROMPT_FORMAT, "wrong", "active"),
}
B0_ARM_CONTRACT = ("B0", "parity_baseline", PROMPT_FORMAT, "none", "bypass")
ADAPTER_TWINS = (("R0", "RA"), ("S0", "SA"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coordinate-tolerance", type=int, default=25)
    parser.add_argument("--min-age", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument(
        "--link-images", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def messages_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _norm999(pixel: int | float, size: int) -> int:
    if size <= 1:
        raise ValueError("screen dimension must exceed one pixel")
    return max(0, min(999, round(float(pixel) * 999 / (size - 1))))


def _coordinate(
    action: Mapping[str, Any], *, screen_size: tuple[int, int]
) -> tuple[int, int] | None:
    if "x" not in action or "y" not in action:
        return None
    width, height = screen_size
    return _norm999(action["x"], width), _norm999(action["y"], height)


def _target_coordinate(arguments: Mapping[str, Any]) -> tuple[int, int] | None:
    raw = arguments.get("coordinate")
    if raw is None:
        raw = arguments.get("coordinate2")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes, bytearray))
        or len(raw) != 2
        or any(type(value) not in (int, float) for value in raw)
    ):
        return None
    return round(float(raw[0])), round(float(raw[1]))


def _near(
    left: tuple[int, int] | None,
    right: tuple[int, int] | None,
    *,
    tolerance: int,
) -> bool:
    return (
        left is not None
        and right is not None
        and max(abs(left[0] - right[0]), abs(left[1] - right[1])) <= tolerance
    )


def history_action_matches_target(
    history_action: Mapping[str, Any],
    target_tool_call: Mapping[str, Any],
    *,
    screen_size: tuple[int, int],
    coordinate_tolerance: int,
) -> bool:
    """Full action equivalence used to label an old visual state as target-specific."""
    arguments = target_tool_call.get("arguments")
    if not isinstance(arguments, Mapping):
        return False
    target_type = arguments.get("action")
    history_type = history_action.get("type")
    if target_type in ("left_click", "click"):
        return history_type == "click" and history_action.get("button", "left") == "left" and _near(
            _coordinate(history_action, screen_size=screen_size),
            _target_coordinate(arguments),
            tolerance=coordinate_tolerance,
        )
    if target_type == "right_click":
        return history_type == "right_click" and _near(
            _coordinate(history_action, screen_size=screen_size),
            _target_coordinate(arguments),
            tolerance=coordinate_tolerance,
        )
    if target_type == "middle_click":
        return (
            history_type == "click"
            and history_action.get("button") == "middle"
            and _near(
                _coordinate(history_action, screen_size=screen_size),
                _target_coordinate(arguments),
                tolerance=coordinate_tolerance,
            )
        )
    if target_type == "double_click":
        return history_type == "double_click" and _near(
            _coordinate(history_action, screen_size=screen_size),
            _target_coordinate(arguments),
            tolerance=coordinate_tolerance,
        )
    if target_type in ("left_click_drag", "drag"):
        return history_type == "drag" and _near(
            _coordinate(history_action, screen_size=screen_size),
            _target_coordinate(arguments),
            tolerance=coordinate_tolerance,
        )
    if target_type == "mouse_move":
        return history_type == "move" and _near(
            _coordinate(history_action, screen_size=screen_size),
            _target_coordinate(arguments),
            tolerance=coordinate_tolerance,
        )
    if target_type == "type":
        return (
            history_type == "type_text"
            and history_action.get("text") == arguments.get("text")
        )
    if target_type == "key":
        keys = arguments.get("keys")
        if history_type == "press":
            observed = [str(history_action.get("key", "")).casefold()]
        elif history_type == "hotkey":
            observed = [
                str(key).casefold() for key in history_action.get("keys", ())
            ]
        else:
            return False
        return (
            isinstance(keys, list)
            and [str(key).casefold() for key in keys] == observed
        )
    if target_type == "scroll":
        pixels = arguments.get("pixels")
        return (
            history_type == "scroll"
            and type(pixels) is int
            and type(history_action.get("dy")) is int
            and history_action.get("dx", 0) == 0
            and (pixels > 0) == (history_action["dy"] > 0)
        )
    if target_type == "hscroll":
        pixels = arguments.get("pixels")
        return (
            history_type == "scroll"
            and type(pixels) is int
            and type(history_action.get("dx")) is int
            and (pixels > 0) == (history_action["dx"] > 0)
        )
    return False


def _stable_choice(values: Sequence[int], *, key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def select_contrast_events(
    record: Mapping[str, Any],
    *,
    coordinate_tolerance: int,
    min_age: int,
    seed: int,
) -> tuple[int, int, int] | None:
    """Return (positive event, positive action step, wrong event)."""
    current_step = int(record["step"])
    screen_size = tuple(record["screen_size"])
    target = record["target_tool_call"]
    history = record["history"]
    positive_pairs: list[tuple[int, int]] = []
    for entry in history:
        action_step = int(entry["step_id"])
        # The screenshot before action j is the post screenshot of event j-1.
        event_step = action_step - 1
        if event_step < 1 or current_step - event_step < min_age:
            continue
        if history_action_matches_target(
            entry["action"],
            target,
            screen_size=screen_size,
            coordinate_tolerance=coordinate_tolerance,
        ):
            positive_pairs.append((event_step, action_step))
    if not positive_pairs:
        return None
    positive_events = [event for event, _ in positive_pairs]
    positive_event = _stable_choice(
        positive_events, key=str(record["dp_id"]), seed=seed
    )
    positive_action_step = next(
        action_step
        for event, action_step in positive_pairs
        if event == positive_event
    )

    negative_events: list[int] = []
    # A selected event e restores the state immediately before action e+1.
    history_by_step = {int(entry["step_id"]): entry for entry in history}
    for event_step in range(1, current_step - min_age + 1):
        if event_step == positive_event:
            continue
        following = history_by_step.get(event_step + 1)
        if following is None:
            continue
        if history_action_matches_target(
            following["action"],
            target,
            screen_size=screen_size,
            coordinate_tolerance=coordinate_tolerance,
        ):
            continue
        negative_events.append(event_step)
    if not negative_events:
        return None
    negative_events.sort(
        key=lambda event: (abs(event - positive_event), event)
    )
    closest_distance = abs(negative_events[0] - positive_event)
    closest = [
        event
        for event in negative_events
        if abs(event - positive_event) == closest_distance
    ]
    wrong_event = _stable_choice(
        closest, key=f"{record['dp_id']}:negative", seed=seed
    )
    return positive_event, positive_action_step, wrong_event


def _serialize_messages(
    messages: Sequence[Mapping[str, Any]], *, image_paths: Sequence[str]
) -> list[dict[str, Any]]:
    slots = iter(image_paths)
    serialized: list[dict[str, Any]] = []
    for message in messages:
        content: list[dict[str, Any]] = []
        for part in message["content"]:
            if part.get("type") == "image":
                try:
                    path = next(slots)
                except StopIteration as error:
                    raise ValueError("fewer image paths than rendered image slots") from error
                content.append({"type": "image", "path": path})
            else:
                content.append(dict(part))
        serialized.append({"role": message["role"], "content": content})
    try:
        next(slots)
    except StopIteration:
        return serialized
    raise ValueError("more image paths than rendered image slots")


def _load_screenshots(
    record: Mapping[str, Any], *, image_root: Path
) -> list[bytes]:
    screenshots = []
    for relpath in record["image_relpaths"]:
        data = (image_root / relpath).read_bytes()
        if not data:
            raise ValueError(f"empty screenshot {relpath!r}")
        screenshots.append(data)
    return screenshots


def _messages_for_selection(
    record: Mapping[str, Any],
    *,
    screenshots: Sequence[bytes],
    recent_budget: int,
    extra_restored: Sequence[int],
    expected_steps: Sequence[int],
    image_paths: Sequence[str],
) -> list[dict[str, Any]]:
    request = build_agentnet_cr_request(
        instruction=record["instruction"],
        screenshots=screenshots,
        history_actions=record["history"],
        current_step=int(record["step"]),
        recent_budget=recent_budget,
        extra_restored_step_ids=tuple(extra_restored),
        screen_size=tuple(record["screen_size"]),
    )
    if request["selected_event_step_ids"] != sorted(expected_steps):
        raise ValueError(
            "frozen memory selection disagrees with the corpus bookkeeping: "
            f"{request['selected_event_step_ids']} != {sorted(expected_steps)}"
        )
    return _serialize_messages(
        render_agentnet_cr_messages(request), image_paths=image_paths
    )


def _split(episode: str, *, seed: int) -> str:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}:split:{episode}".encode("utf-8")).digest()[:8],
        "big",
    ) / 2**64
    if value < 0.10:
        return "test"
    if value < 0.20:
        return "dev"
    return "train"


def _link_image_roots(
    paths: Iterable[str], *, image_root: Path, output_root: Path
) -> list[str]:
    linked: list[str] = []
    for name in sorted({Path(path).parts[0] for path in paths}):
        source = image_root / name
        target = output_root / name
        if target.exists() or target.is_symlink():
            continue
        if not source.exists():
            raise ValueError(f"image root has no top-level member {name!r}")
        os.symlink(os.path.realpath(source), target)
        linked.append(name)
    return linked


def _count_images(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for message in messages
        for part in message["content"]
        if part.get("type") == "image"
    )


def _validate_group(
    rows: Sequence[Mapping[str, Any]],
    b0_row: Mapping[str, Any],
    *,
    budget: int,
) -> None:
    """Fail-closed in-group checks — the hard §6/§8.6 constraints, at build time."""
    by_slot = {row["arm_slot"]: row for row in rows}
    if sorted(by_slot) != sorted(DESKTOP_ARM_CONTRACT) or len(by_slot) != len(rows):
        raise ValueError(
            f"group must carry exactly the arms {sorted(DESKTOP_ARM_CONTRACT)}; "
            f"got {sorted(row['arm_slot'] for row in rows)}"
        )
    positive = by_slot["SA"]
    for row in (*rows, b0_row):
        contract = (
            B0_ARM_CONTRACT
            if row["arm_slot"] == "B0"
            else DESKTOP_ARM_CONTRACT[row["arm_slot"]]
        )
        declared = (
            row["arm_id"], row["role"], row["prompt_format"],
            row["selection_mode"], row["adapter_mode"],
        )
        if declared != contract:
            raise ValueError(
                f"arm {row['arm_slot']!r} declares {declared}, contract says {contract}"
            )
        row_budget = 0 if row["arm_slot"] == "B0" else budget
        if (
            len(row["selected_steps"]) != row_budget
            or len(row["selected_images"]) != row_budget
            or _count_images(row["messages"]) != row_budget + 1
            or row["memory_config"]["restored_event_step_ids"] != row["selected_steps"]
            or row["messages_sha256"] != messages_sha256(row["messages"])
        ):
            raise ValueError(f"arm {row['arm_slot']!r} budget/image bookkeeping broken")
        if (
            row["target_text"] != positive["target_text"]
            or row["current_image"] != positive["current_image"]
            or row["episode"] != positive["episode"]
            or row["decision_step"] != positive["decision_step"]
        ):
            raise ValueError(f"arm {row['arm_slot']!r} disagrees with SA on shared fields")
    # note (luojiaxuan): §8.6 —— bypass/active 孪生臂除 adapter_mode 外必须逐字节
    # 同 prompt;负臂的 W0 由 trainer 重算,语料侧只须锁 WA 自身的账目字段。
    for bypass_slot, active_slot in ADAPTER_TWINS:
        if by_slot[bypass_slot]["messages_sha256"] != by_slot[active_slot]["messages_sha256"]:
            raise ValueError(f"{bypass_slot}/{active_slot} prompts diverged")
    negative = by_slot[NEGATIVE_ARM_SLOT]
    if int(negative["distractor_source_step"]) not in negative["selected_steps"]:
        raise ValueError("WA distractor_source_step is absent from its selected_steps")
    if int(negative["oracle_source_step"]) in negative["selected_steps"]:
        raise ValueError("WA still carries the oracle frame")


def _percentile_summary(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    def rank(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return {
        "min": ordered[0],
        "p50": rank(0.50),
        "p90": rank(0.90),
        "max": ordered[-1],
    }


def build_corpus(
    records: Sequence[dict[str, Any]],
    *,
    image_root: Path,
    coordinate_tolerance: int,
    min_age: int,
    seed: int,
    max_groups: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    b0_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    referenced_paths: set[str] = set()
    positive_ages: list[int] = []
    wrong_ages: list[int] = []
    positive_wrong_distances: list[int] = []
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    for index in order:
        if max_groups and counters["groups"] >= max_groups:
            break
        record = records[index]
        selected = select_contrast_events(
            record,
            coordinate_tolerance=coordinate_tolerance,
            min_age=min_age,
            seed=seed,
        )
        if selected is None:
            counters["rejected_no_recurrence_or_negative"] += 1
            continue
        positive_event, positive_action_step, wrong_event = selected
        current_step = int(record["step"])
        # 冻结 recent 选择器在 k=1 下取最后一个事件;其 post 帧与当前截图同帧,
        # 这正是部署语义(在线 predict 的 Recent-1 同样如此),不做"改良"。
        recent_events = list(range(current_step - BUDGET, current_step))
        relpaths = record["image_relpaths"]
        screenshots = _load_screenshots(record, image_root=image_root)
        current_path = relpaths[current_step - 1]
        pair_group = f"agentnet:{record['dp_id']}"
        episode = str(record["task_id"])
        split = _split(episode, seed=seed)

        selection_specs = {
            "recent": (recent_events, BUDGET, ()),
            "recurrence": ([positive_event], 0, (positive_event,)),
            "wrong": ([wrong_event], 0, (wrong_event,)),
            "none": ([], 0, ()),
        }
        rendered: dict[str, dict[str, Any]] = {}
        for mode, (steps, recent_budget, extras) in selection_specs.items():
            selected_images = [relpaths[step] for step in steps]
            messages = _messages_for_selection(
                record,
                screenshots=screenshots,
                recent_budget=recent_budget,
                extra_restored=extras,
                expected_steps=steps,
                image_paths=[*selected_images, current_path],
            )
            rendered[mode] = {
                "selected_steps": list(steps),
                "selected_images": selected_images,
                "messages": messages,
                "messages_sha256": messages_sha256(messages),
            }

        common = {
            "pair_group": pair_group,
            "episode": episode,
            "decision_step": current_step,
            "budget": BUDGET,
            "split": split,
            "instruction": record["instruction"],
            "current_image": current_path,
            "target_text": record["target_text"],
            "reference_arm_id": REFERENCE_ARM_ID,
            "deployment_baseline_arm_id": DEPLOYMENT_BASELINE_ARM_ID,
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
            },
        }
        group_rows = []
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
                "variant": f"{slot}_{mode}{BUDGET}",
                "recent_frames_kept": sum(
                    1
                    for step in rendered[mode]["selected_steps"]
                    if step in recent_events
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
            group_rows.append(row)
        b0_arm_id, b0_role, b0_format, b0_mode, b0_adapter = B0_ARM_CONTRACT
        b0_row = {
            "schema_version": B0_SAMPLE_SCHEMA,
            "sample_id": f"{pair_group}|B0",
            **common,
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
        _validate_group(group_rows, b0_row, budget=BUDGET)
        rows.extend(group_rows)
        b0_rows.append(b0_row)
        for mode in selection_specs:
            referenced_paths.update(rendered[mode]["selected_images"])
        referenced_paths.add(current_path)
        positive_ages.append(current_step - positive_event)
        wrong_ages.append(current_step - wrong_event)
        positive_wrong_distances.append(abs(positive_event - wrong_event))
        counters["groups"] += 1
        counters[f"groups_{split}"] += 1
        counters[
            "groups_" + record["target_tool_call"]["arguments"]["action"]
        ] += 1
    manifest = {
        "schema_version": CORPUS_SCHEMA,
        "sample_schema": SAMPLE_SCHEMA,
        "b0_sample_schema": B0_SAMPLE_SCHEMA,
        "seed": seed,
        "coordinate_tolerance": coordinate_tolerance,
        "min_age": min_age,
        "budget": BUDGET,
        "arm_contract": {
            slot: list(spec) for slot, spec in DESKTOP_ARM_CONTRACT.items()
        },
        "b0_arm_contract": list(B0_ARM_CONTRACT),
        "prompt_format": PROMPT_FORMAT,
        "counters": dict(sorted(counters.items())),
        "sample_count": len(rows),
        "b0_sample_count": len(b0_rows),
        "group_count": counters["groups"],
        "sources": {"AgentNet/OpenCUA": counters["groups"]},
        "stats": (
            {
                "positive_age": _percentile_summary(positive_ages),
                "wrong_age": _percentile_summary(wrong_ages),
                "positive_wrong_distance": _percentile_summary(
                    positive_wrong_distances
                ),
            }
            if positive_ages
            else {}
        ),
        "referenced_image_count": len(referenced_paths),
        "referenced_paths": sorted(referenced_paths),
    }
    return rows, b0_rows, manifest


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.coordinate_tolerance < 0:
        raise SystemExit("--coordinate-tolerance must be non-negative")
    if args.min_age < 2:
        raise SystemExit("--min-age must be at least two")
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("input manifest is empty")
    rows, b0_rows, manifest = build_corpus(
        records,
        image_root=args.image_root,
        coordinate_tolerance=args.coordinate_tolerance,
        min_age=args.min_age,
        seed=args.seed,
        max_groups=args.max_groups,
    )
    if not rows:
        raise SystemExit("no target-specific recurrence groups were found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_root / "samples.jsonl"
    b0_path = args.output_root / "parity_b0.jsonl"
    _write_jsonl(samples_path, rows)
    _write_jsonl(b0_path, b0_rows)
    linked: list[str] = []
    if args.link_images:
        linked = _link_image_roots(
            manifest.pop("referenced_paths"),
            image_root=args.image_root,
            output_root=args.output_root,
        )
    else:
        manifest.pop("referenced_paths")
    manifest["inputs"] = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "image_root": str(args.image_root),
        "linked_image_roots": linked,
    }
    manifest["samples_sha256"] = sha256_file(samples_path)
    manifest["parity_b0_sha256"] = sha256_file(b0_path)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
