#!/usr/bin/env python3
"""Reduce OSWorld LoRA leakage profiles under the preregistered gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.osworld_lora_leakage import (
    SUMMARY_SCHEMA_VERSION,
    classify_profile,
    load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profiles-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    summaries = {}
    for profile in config["profiles"]:
        profile_id = profile["profile_id"]
        path = args.profiles_root.resolve() / profile_id / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE_OSWORLD_LORA_LEAKAGE_PROFILE":
            raise ValueError(f"incomplete leakage profile: {profile_id}")
        summaries[profile_id] = summary
    frozen = summaries[config["frozen_profile_id"]]
    comparisons = {
        profile_id: classify_profile(
            frozen=frozen,
            adapted=summary,
            thresholds=config["thresholds"],
        )
        for profile_id, summary in summaries.items()
        if profile_id != config["frozen_profile_id"]
    }
    output = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "COMPLETE_OSWORLD_LORA_LEAKAGE_TEST",
        "frozen_profile_id": config["frozen_profile_id"],
        "profile_summaries": summaries,
        "comparisons_to_frozen": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
