#!/usr/bin/env python3
"""Reduce paired frozen/LoRA OSWorld episode results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.osworld_transfer import (
    load_arm_results,
    load_transfer_config,
    reduce_transfer_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_transfer_config(args.config.resolve())
    expected_count = int(config["selection"]["task_count"])
    results = {
        profile["arm"]: load_arm_results(
            args.raw_root.resolve() / profile["arm"],
            expected_profile_id=profile["policy_profile_id"],
            expected_task_count=expected_count,
        )
        for profile in config["profiles"]
    }
    summary = reduce_transfer_results(config=config, arm_results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
