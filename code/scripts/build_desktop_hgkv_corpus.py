#!/usr/bin/env python3
"""Build Desktop history-use contrast groups from AgentNet success trajectories.

# note (luojiaxuan): 这条管线直接产 policy 训练数据,不以 frozen policy 是否做错
# 为前置筛选。正例是同一成功轨迹里与当前 gold 动作完整等价的旧动作状态:恢复该
# 旧动作执行前的截图(即前一事件的 post screenshot),让 adapter 学习
# summary-action ↔ high-fidelity visual state 的绑定。两个内容对照分别是同轨迹、
# 同 age 的错误状态和跨轨迹同分辨率截图;四臂共享完全相同的当前截图、文字历史与
# target tool call。输出沿用 success_sft_sample.v1 + memory_config,可直接交给
# train_success_sft_lora.py 的 history_group 路径。
"""

from __future__ import annotations

import argparse
import base64
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

CORPUS_SCHEMA = "causalcache.desktop_hgkv_corpus.v1"
SAMPLE_SCHEMA = "causalcache.success_sft_sample.v1"
PROMPT_FORMAT = "osworld_official"
VARIANTS = ("correct", "b0", "shuffled", "irrelevant")


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
    """Return (positive event, positive action step, shuffled event)."""
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
    shuffled_event = _stable_choice(
        closest, key=f"{record['dp_id']}:negative", seed=seed
    )
    return positive_event, positive_action_step, shuffled_event


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


def _replace_selected_image(
    request: dict[str, Any], *, event_step: int, replacement: bytes
) -> None:
    encoded = base64.b64encode(replacement).decode("ascii")
    changed = 0
    for event in request["history"]:
        if event["step_id"] == event_step:
            if event["restored_post_screenshot_png_base64"] is None:
                raise ValueError("selected event has no restored image")
            event["restored_post_screenshot_png_base64"] = encoded
            changed += 1
    if changed != 1:
        raise ValueError("selected event replacement did not match exactly one event")


def _messages_for_variant(
    record: Mapping[str, Any],
    *,
    screenshots: Sequence[bytes],
    selected_event: int | None,
    image_paths: Sequence[str],
    irrelevant_bytes: bytes | None = None,
) -> list[dict[str, Any]]:
    request = build_agentnet_cr_request(
        instruction=record["instruction"],
        screenshots=screenshots,
        history_actions=record["history"],
        current_step=int(record["step"]),
        recent_budget=0,
        extra_restored_step_ids=(
            () if selected_event is None else (selected_event,)
        ),
        screen_size=tuple(record["screen_size"]),
    )
    if irrelevant_bytes is not None:
        if selected_event is None:
            raise ValueError("irrelevant replacement requires a selected event")
        _replace_selected_image(
            request, event_step=selected_event, replacement=irrelevant_bytes
        )
    return _serialize_messages(
        render_agentnet_cr_messages(request), image_paths=image_paths
    )


def _donor_index(
    records: Sequence[Mapping[str, Any]], index: int
) -> tuple[int, str] | None:
    record = records[index]
    size = tuple(record["screen_size"])
    for offset in range(1, len(records)):
        donor_index = (index + offset) % len(records)
        donor = records[donor_index]
        if (
            donor["task_id"] != record["task_id"]
            and tuple(donor["screen_size"]) == size
        ):
            return donor_index, donor["image_relpaths"][-1]
    return None


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


def build_corpus(
    records: Sequence[dict[str, Any]],
    *,
    image_root: Path,
    coordinate_tolerance: int,
    min_age: int,
    seed: int,
    max_groups: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    referenced_paths: set[str] = set()
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
        donor = _donor_index(records, index)
        if donor is None:
            counters["rejected_no_donor"] += 1
            continue
        positive_event, positive_action_step, shuffled_event = selected
        donor_index, donor_path = donor
        screenshots = _load_screenshots(record, image_root=image_root)
        donor_bytes = (image_root / donor_path).read_bytes()
        current_path = record["image_relpaths"][int(record["step"]) - 1]
        correct_path = record["image_relpaths"][positive_event]
        shuffled_path = record["image_relpaths"][shuffled_event]
        pair_group = f"agentnet:{record['dp_id']}"
        episode = str(record["task_id"])
        split = _split(episode, seed=seed)
        variant_specs = (
            ("correct", positive_event, [correct_path, current_path], None),
            ("b0", None, [current_path], None),
            ("shuffled", shuffled_event, [shuffled_path, current_path], None),
            (
                "irrelevant",
                positive_event,
                [donor_path, current_path],
                donor_bytes,
            ),
        )
        for variant, event_step, image_paths, irrelevant_bytes in variant_specs:
            restored = [] if event_step is None else [event_step]
            sample = {
                "schema_version": SAMPLE_SCHEMA,
                "sample_id": f"{pair_group}|{variant}",
                "episode": episode,
                "task_type": "agentnet_desktop",
                "task_index": record["dp_id"],
                "decision_step_id": int(record["step"]),
                "step_index": int(record["step"]),
                "variant": variant,
                "pair_group": pair_group,
                "prompt_format": PROMPT_FORMAT,
                "memory_config": {
                    "mode": "target_action_recurrence",
                    "budget": len(restored),
                    "restored_event_step_ids": restored,
                },
                "messages": _messages_for_variant(
                    record,
                    screenshots=screenshots,
                    selected_event=event_step,
                    image_paths=image_paths,
                    irrelevant_bytes=irrelevant_bytes,
                ),
                "target_text": record["target_text"],
                "official_terminal_success": 1.0,
                "split": split,
                "source": {
                    "dataset": "AgentNet/OpenCUA",
                    "os": record["os"],
                    "dp_id": record["dp_id"],
                    "positive_rule": "full_action_recurrence_pre_state",
                    "positive_action_step": positive_action_step,
                    "positive_event_step": positive_event,
                    "shuffled_event_step": shuffled_event,
                    "donor_dp_id": records[donor_index]["dp_id"],
                },
            }
            rows.append(sample)
            referenced_paths.update(image_paths)
        counters["groups"] += 1
        counters[f"groups_{split}"] += 1
        counters[
            "groups_" + record["target_tool_call"]["arguments"]["action"]
        ] += 1
    manifest = {
        "schema_version": CORPUS_SCHEMA,
        "seed": seed,
        "coordinate_tolerance": coordinate_tolerance,
        "min_age": min_age,
        "variants": list(VARIANTS),
        "prompt_format": PROMPT_FORMAT,
        "counters": dict(sorted(counters.items())),
        "sample_count": len(rows),
        "group_count": counters["groups"],
        "referenced_image_count": len(referenced_paths),
        "referenced_paths": sorted(referenced_paths),
    }
    return rows, manifest


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
    rows, manifest = build_corpus(
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
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
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
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
