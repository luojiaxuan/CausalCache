#!/usr/bin/env python3
"""Build the frozen 30-prompt OSWorld LoRA leakage manifest from live episodes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from causalcache.osworld_benchmark import (
    load_osworld_benchmark_config,
    validate_osworld_benchmark_roster,
)
from causalcache.osworld_lora_leakage import (
    MANIFEST_SCHEMA_VERSION,
    load_config,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    config = load_config(args.config.resolve())
    benchmark_config = load_osworld_benchmark_config(
        repository_root / config["osworld_benchmark_config"]
    )
    roster = validate_osworld_benchmark_roster(
        args.osworld_root.resolve(), benchmark_config
    ).selected
    count = int(config["prompt_count"])
    indices = [round(index * (len(roster) - 1) / (count - 1)) for index in range(count)]
    selected = [roster[index] for index in indices]
    prompts = []
    for prompt_index, (domain, task_id) in enumerate(selected):
        task_root = args.episodes_root.resolve() / domain / task_id
        result_path = task_root / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "COMPLETE_OSWORLD_EPISODE":
            raise ValueError(f"incomplete OSWorld fixture episode: {result_path}")
        attempt_root = task_root / "attempts" / result["attempt_id"]
        initial = attempt_root / "initial.png"
        step = attempt_root / "step-001.png"
        if not initial.is_file() or not step.is_file():
            raise FileNotFoundError(f"fixture screenshots missing: {task_root}")
        recent_b4 = prompt_index % 2 == 1
        history = []
        if recent_b4:
            for step_id in range(1, 5):
                source = step if step_id % 2 == 0 else initial
                history.append(
                    {
                        "step_id": step_id,
                        "action": {"type": "wait"},
                        "result_status": "executed",
                        "screen_changed": sha256_file(initial) != sha256_file(step),
                        "restored_screenshot_path": str(source),
                        "restored_screenshot_sha256": sha256_file(source),
                    }
                )
        current = step if recent_b4 else initial
        prompts.append(
            {
                "prompt_id": f"{prompt_index:03d}-{domain}-{task_id}",
                "roster_index": indices[prompt_index],
                "domain": domain,
                "task_id": task_id,
                "instruction": result["task"]["instruction"],
                "screen_size": config["screen_size"],
                "input_mode": "recent_b4" if recent_b4 else "single_image",
                "current_screenshot_path": str(current),
                "current_screenshot_sha256": sha256_file(current),
                "history": history,
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "prompt_count": len(prompts),
        "selection": "evenly_spaced_official_no_gdrive_roster",
        "domain_counts": dict(sorted(Counter(p["domain"] for p in prompts).items())),
        "input_mode_counts": dict(
            sorted(Counter(p["input_mode"] for p in prompts).items())
        ),
        "synthetic_history_for_leakage_probe": True,
        "synthetic_history_note": (
            "recent-B4 prompts alternate the live initial and post-WAIT screenshots; "
            "the probe tests grammar leakage, not memory quality"
        ),
        "config_sha256": sha256_file(args.config.resolve()),
        "episodes_root": str(args.episodes_root.resolve()),
        "prompts": prompts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
