"""Run the FindingDory Object Attributes exact-B restoration experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from causalcache.findingdory_oracle_gap import (
    exact_budget_selections,
    frame_is_valid,
    logical_episode_shard,
    normalize_content_summary,
    parse_answer_groups,
    parse_predicted_frame,
    reduce_paired_results,
)
from causalcache.policy.qwen_runtime import QwenPolicyRuntime
from causalcache.remote_zip import download_remote_zip_member, remote_zip_catalog


SUMMARY_SYSTEM = """You build a task-independent episodic memory for an embodied household agent. Convert the supplied chronological observations into compact structured JSON. The pixels contain legacy overlaid frame numbers and times from a longer source trajectory: ignore and never transcribe those overlays. Do not output any frame id, index, timestamp, time, or number. Record only visible objects and receptacles, rooms, interactions or state changes, and fine visual attributes supported by pixels. Never invent a future task or a navigation answer."""

POLICY_SYSTEM = """You are a frozen high-level embodied navigation policy. Use the compressed episode memory and any restored high-fidelity frames to choose an original frame id that is a viable navigation goal for the task. You may choose any original frame from 0 through 95, including one described only by text. Return exactly one compact JSON object: {\"frame_indices\":[integer]}."""


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
    partial.replace(path)


def _episode_number(episode_id: str) -> int:
    return int(episode_id.rsplit("_", 1)[1])


def _task_rows(parquet_path: Path, *, category: str, episode_limit: int) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    selected = frame[frame["low_level_category"] == category]
    rows = [
        {key: (int(value) if key == "num_interactions" else str(value)) for key, value in row.items()}
        for row in selected.to_dict(orient="records")
        if _episode_number(str(row["ep_id"])) <= episode_limit
    ]
    return sorted(rows, key=lambda row: (_episode_number(row["ep_id"]), row["task_id"]))


def prepare_data(args: argparse.Namespace, config: dict[str, Any]) -> None:
    benchmark = config["benchmark"]
    base = f"https://huggingface.co/datasets/{benchmark['repo']}/resolve/{benchmark['revision']}"
    parquet_path = args.data_dir / benchmark["validation_parquet"]
    _download(f"{base}/{benchmark['validation_parquet']}?download=true", parquet_path)
    tasks = _task_rows(
        parquet_path,
        category=benchmark["low_level_category"],
        episode_limit=args.episode_limit,
    )
    episodes = {row["ep_id"]: row["video"] for row in tasks}
    zip_url = f"{base}/{benchmark['videos_zip']}?download=true"
    catalog = remote_zip_catalog(zip_url)
    videos = []
    for episode_id, member_name in sorted(episodes.items(), key=lambda item: _episode_number(item[0])):
        if member_name not in catalog:
            raise ValueError(f"video missing from public ZIP: {member_name}")
        destination = args.data_dir / member_name
        download_remote_zip_member(zip_url, catalog[member_name], destination)
        videos.append(
            {
                "episode_id": episode_id,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output_dir / "tasks.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in tasks),
        encoding="utf-8",
    )
    manifest = {
        "benchmark": benchmark,
        "episode_limit": args.episode_limit,
        "task_count": len(tasks),
        "episode_count": len(episodes),
        "validation_parquet": {
            "path": str(parquet_path),
            "sha256": _sha256(parquet_path),
        },
        "videos": videos,
    }
    (args.output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"tasks": len(tasks), "episodes": len(episodes)}, sort_keys=True))


def _read_video(path: Path, *, expected_frames: int) -> list[Any]:
    import cv2
    from PIL import Image, ImageDraw

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # note (luojiaxuan): Public videos burn the full-trajectory Frame/Time
        # namespace at (10,50)/(10,70); answers instead use subsampled 0--95.
        ImageDraw.Draw(image).rectangle((0, 30, 280, 84), fill=(0, 0, 0))
        frames.append(image)
    capture.release()
    if len(frames) != expected_frames:
        raise ValueError(f"{path} has {len(frames)} frames, expected {expected_frames}")
    return frames


def _summary_messages(frames: list[Any], *, start: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Chronological observation chunk. Canonical subsampled range {start}-{start + len(frames) - 1} is attached by the memory system outside your JSON."}
    ]
    for offset, frame in enumerate(frames):
        content.extend(
            [
                {"type": "text", "text": f"Canonical subsampled observation slot {offset}; ignore any frame/time text inside the pixels:"},
                {"type": "image", "image": frame},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": "Return one compact JSON object using only these keys when applicable: objects, receptacles, rooms, interactions, fine_attributes, state_changes. Values are short strings or lists of strings. Output no numbers or frame/time fields.",
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SUMMARY_SYSTEM}]},
        {"role": "user", "content": content},
    ]


def _policy_messages(
    *,
    task: str,
    summaries: list[dict[str, Any]],
    selected: tuple[int, ...],
    frames: list[Any],
    current_frame: int,
    budget: int,
) -> list[dict[str, Any]]:
    compressed_memory = [
        {
            "start_frame": int(chunk["start_frame"]),
            "end_frame": int(chunk["end_frame"]),
            "content_summary": chunk["content_summary"],
        }
        for chunk in summaries
    ]
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Task: {task}"},
        {
            "type": "text",
            "text": "Task-independent compressed episode memory:\n" + json.dumps(compressed_memory, ensure_ascii=False, separators=(",", ":")),
        },
        {"type": "text", "text": f"Restoration budget B={budget}; exactly {budget} historical frames follow."},
    ]
    for frame_id in selected:
        content.extend(
            [
                {"type": "text", "text": f"Restored high-fidelity original frame {frame_id}:"},
                {"type": "image", "image": frames[frame_id]},
            ]
        )
    content.extend(
        [
            {"type": "text", "text": f"Fixed current observation, original frame {current_frame} (not charged to B):"},
            {"type": "image", "image": frames[current_frame]},
            {"type": "text", "text": "Choose one viable original frame id from 0 through 95."},
        ]
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": POLICY_SYSTEM}]},
        {"role": "user", "content": content},
    ]


def _runtime(args: argparse.Namespace, config: dict[str, Any], *, summary: bool) -> QwenPolicyRuntime:
    key = "visual_tokens_per_summary_image" if summary else "visual_tokens_per_restored_image"
    return QwenPolicyRuntime(
        model_dir=args.model_dir,
        device=args.device,
        visual_tokens_per_image=int(config["model"][key]),
    )


def _task_file(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(args: argparse.Namespace, config: dict[str, Any]) -> None:
    tasks = _task_file(args.input_dir / "tasks.jsonl")
    episode_paths = {row["ep_id"]: args.data_dir / row["video"] for row in tasks}
    assigned = logical_episode_shard(
        episode_paths,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    runtime = _runtime(args, config, summary=True)
    chunk_size = int(config["memory"]["summary_chunk_frames"])
    expected_frames = int(config["benchmark"]["video_frames"])
    summaries_dir = args.output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    for episode_id in assigned:
        output_path = summaries_dir / f"{episode_id}.json"
        if output_path.exists():
            continue
        frames = _read_video(episode_paths[episode_id], expected_frames=expected_frames)
        chunks = []
        for start in range(0, len(frames), chunk_size):
            result = runtime.generate_text(
                _summary_messages(frames[start : start + chunk_size], start=start),
                max_new_tokens=int(config["model"]["summary_max_new_tokens"]),
            )
            chunks.append(
                {
                    "start_frame": start,
                    "end_frame": min(start + chunk_size, len(frames)) - 1,
                    "content_summary": normalize_content_summary(result["output_text"]),
                    **result,
                }
            )
        output_path.write_text(
            json.dumps({"episode_id": episode_id, "model": runtime.metadata, "chunks": chunks}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"episode": episode_id, "chunks": len(chunks)}, sort_keys=True), flush=True)


def evaluate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    tasks = _task_file(args.input_dir / "tasks.jsonl")
    assigned_episode_ids = frozenset(
        logical_episode_shard(
            [task["ep_id"] for task in tasks],
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    )
    assigned = [task for task in tasks if task["ep_id"] in assigned_episode_ids]
    runtime = _runtime(args, config, summary=False)
    current_frame = int(config["benchmark"]["current_frame"])
    expected_frames = int(config["benchmark"]["video_frames"])
    budgets = [int(value) for value in config["memory"]["budgets"]]
    output_path = args.output_dir / f"results-shard-{args.shard_index:02d}.jsonl"
    completed: set[tuple[str, str, int, str]] = set()
    if output_path.exists():
        for row in _task_file(output_path):
            completed.add((row["episode_id"], row["task_id"], int(row["budget"]), row["arm"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    loaded_episode = None
    frames = None
    summaries = None
    with output_path.open("a", encoding="utf-8") as stream:
        for task in assigned:
            episode_id = task["ep_id"]
            if episode_id != loaded_episode:
                frames = _read_video(args.data_dir / task["video"], expected_frames=expected_frames)
                summaries = _json(args.input_dir / "summaries" / f"{episode_id}.json")["chunks"]
                loaded_episode = episode_id
            groups = parse_answer_groups(task["answer"])
            assert frames is not None and summaries is not None
            for budget in budgets:
                selections = exact_budget_selections(groups, budget=budget, current_frame=current_frame)
                for arm in ("recent", "oracle"):
                    key = (episode_id, task["task_id"], budget, arm)
                    if key in completed:
                        continue
                    selected = selections[arm]
                    generated = runtime.generate_text(
                        _policy_messages(
                            task=task["question"],
                            summaries=summaries,
                            selected=selected,
                            frames=frames,
                            current_frame=current_frame,
                            budget=budget,
                        ),
                        max_new_tokens=int(config["model"]["policy_max_new_tokens"]),
                    )
                    prediction = parse_predicted_frame(generated["output_text"])
                    row = {
                        "episode_id": episode_id,
                        "task_id": task["task_id"],
                        "question": task["question"],
                        "answer_groups": groups,
                        "budget": budget,
                        "arm": arm,
                        "selected_frames": selected,
                        "predicted_frame": prediction,
                        "success": frame_is_valid(prediction, groups),
                        "generation": generated,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    print(json.dumps({"key": key, "prediction": prediction, "success": row["success"]}), flush=True)


def reduce(args: argparse.Namespace, config: dict[str, Any]) -> None:
    rows = []
    for path in sorted(args.input_dir.glob("results-shard-*.jsonl")):
        rows.extend(_task_file(path))
    statistics = config["statistics"]
    report = reduce_paired_results(
        rows,
        bootstrap_samples=int(statistics["bootstrap_samples"]),
        seed=int(config["schedule"]["seed"]),
        minimum_oracle_success=float(statistics["minimum_oracle_hl_sr"]),
        minimum_delta=float(statistics["minimum_oracle_minus_recent"]),
        minimum_coverage=float(statistics["minimum_selection_difference_coverage"]),
    )
    tasks = _task_file(args.input_dir / "tasks.jsonl")
    expected_rows = len(tasks) * len(config["memory"]["budgets"]) * 2
    if len(rows) != expected_rows:
        raise ValueError(f"incomplete result rows: {len(rows)} != {expected_rows}")
    scope = "full" if args.episode_limit == int(config["schedule"]["full_episode_limit"]) else "pilot"
    decision = "GO" if scope == "full" and report["any_budget_gate_pass"] else (
        "PILOT_PASS_EXPAND" if scope == "pilot" and report["any_budget_gate_pass"] else "NO_GO"
    )
    result = {
        "schema_version": "1.0.0",
        "scope": scope,
        "decision": decision,
        "episode_limit": args.episode_limit,
        "task_count": len(tasks),
        "result_rows": len(rows),
        "config_sha256": _sha256(args.config),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip(),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        **report,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare-data", "summarize", "evaluate", "reduce"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-limit", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.episode_limit <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid episode limit or shard specification")
    return args


def main() -> None:
    args = parse_args()
    config = _json(args.config)
    if args.phase == "prepare-data":
        if args.data_dir is None:
            raise ValueError("prepare-data requires --data-dir")
        prepare_data(args, config)
    elif args.phase == "summarize":
        if args.data_dir is None or args.model_dir is None or args.input_dir is None:
            raise ValueError("summarize requires data, model, and input directories")
        summarize(args, config)
    elif args.phase == "evaluate":
        if args.data_dir is None or args.model_dir is None or args.input_dir is None:
            raise ValueError("evaluate requires data, model, and input directories")
        evaluate(args, config)
    else:
        if args.input_dir is None:
            raise ValueError("reduce requires --input-dir")
        reduce(args, config)


if __name__ == "__main__":
    main()
