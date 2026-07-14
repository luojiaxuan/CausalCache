"""Validate an experiment contract and one serialized decision record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.contracts import ExperimentContract
from causalcache.schema import DecisionRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = ExperimentContract.load(args.config)
    with args.decision.open("r", encoding="utf-8") as handle:
        decision = DecisionRecord.from_dict(json.load(handle))
    if not decision.reference_is_valid(contract.teacher.validation_mode):
        raise ValueError("decision does not satisfy the validated teacher contract")
    print(
        json.dumps(
            {
                "contract_version": contract.version,
                "trajectory_id": decision.trajectory_id,
                "decision_step_id": decision.decision_step_id,
                "history_events": len(decision.history),
                "teacher_validation_mode": contract.teacher.validation_mode,
                "teacher_valid": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
