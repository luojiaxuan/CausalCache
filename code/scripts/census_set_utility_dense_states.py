#!/usr/bin/env python3
"""Count all eligible per-step states and legacy image coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from causalcache.set_utility_dense import dense_legacy_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assignments = manifest["assignments"]
    eligible = Counter()
    covered = Counter()
    missing = Counter()
    missing_trajectories = Counter()
    for assignment in assignments:
        role = assignment["role"]
        total, available, missing_steps = dense_legacy_coverage(
            decision_count=assignment["decision_count"]
        )
        eligible[role] += total
        covered[role] += available
        missing[role] += len(missing_steps)
        if missing_steps:
            missing_trajectories[role] += 1
    payload = {
        "candidate_count": 4,
        "eligible_state_count": sum(eligible.values()),
        "eligible_state_count_by_role": dict(sorted(eligible.items())),
        "legacy_artifact_covered_state_count": sum(covered.values()),
        "legacy_artifact_covered_state_count_by_role": dict(sorted(covered.items())),
        "legacy_artifact_missing_state_count": sum(missing.values()),
        "legacy_artifact_missing_state_count_by_role": dict(sorted(missing.items())),
        "legacy_artifact_missing_trajectory_count": sum(missing_trajectories.values()),
        "legacy_artifact_missing_trajectory_count_by_role": dict(
            sorted(missing_trajectories.items())
        ),
        "minimum_decision_step": 6,
        "source_trajectory_count": len(assignments),
        "status": "COMPLETED_DENSE_PER_STEP_CENSUS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
