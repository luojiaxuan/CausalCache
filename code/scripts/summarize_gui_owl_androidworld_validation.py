"""Summarize a mathematically decisive partial AndroidWorld validation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_gui_owl_androidworld_validation import (
    aggregate_early_stopped_validation,
    write_json_atomic,
)


def load_episodes(episodes_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = sorted(episodes_dir.glob("*.json"))
    episodes = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    episodes.sort(key=lambda episode: int(episode["plan_index"]))
    return episodes, paths


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(args.validation_plan.read_text(encoding="utf-8"))
    episodes, paths = load_episodes(args.episodes_dir)
    summary = aggregate_early_stopped_validation(
        plan=plan,
        episodes=episodes,
        minimum_parse_coverage=args.minimum_parse_coverage,
        minimum_official_success=args.minimum_official_success,
    )
    started_at = sorted(
        episode["started_at"] for episode in episodes if episode.get("started_at")
    )
    base_urls = sorted(
        {
            episode["environment_runtime"]["base_url"]
            for episode in episodes
            if episode.get("environment_runtime")
        }
    )
    runtime_records = [
        episode["environment_runtime"]
        for episode in episodes
        if episode.get("environment_runtime")
    ]
    policy_records = [episode["policy"] for episode in episodes if episode.get("policy")]
    generations = [
        step["generation"]
        for episode in episodes
        for step in episode.get("steps", [])
        if step.get("generation")
    ]
    image_grids = sorted(
        {
            tuple(grid)
            for generation in generations
            for grid in generation.get("image_grid_thw", [])
        }
    )
    per_image_visual_tokens = sorted(
        {
            int(generation["effective_visual_tokens"] / generation["image_count"])
            for generation in generations
            if generation.get("image_count")
        }
    )
    summary.update(
        {
            "first_started_at": started_at[0] if started_at else None,
            "last_started_at": started_at[-1] if started_at else None,
            "base_urls": base_urls,
            "server_image": runtime_records[0]["server_image"],
            "server_image_sha256": runtime_records[0]["server_image_sha256"],
            "policy": {
                "snapshot": policy_records[0]["snapshot"],
                "model_class": policy_records[0]["model_class"],
                "processor_class": policy_records[0]["processor_class"],
                "maximum_visible_images": policy_records[0][
                    "maximum_visible_images"
                ],
                "max_new_tokens": policy_records[0]["max_new_tokens"],
                "do_sample": policy_records[0]["do_sample"],
                "visual_preprocessing": policy_records[0]["visual_preprocessing"],
                "observed_image_grid_thw": [list(grid) for grid in image_grids],
                "observed_effective_visual_tokens_per_image": per_image_visual_tokens,
            },
            "episode_files": [
                str(Path("episodes") / path.name) for path in paths
            ],
        }
    )
    write_json_atomic(args.output, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-parse-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-official-success", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    summary = summarize(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
