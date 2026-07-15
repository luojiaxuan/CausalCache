"""Validate the frozen restoration-v2 scientific contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.restoration_v2_contract import RestorationV2Contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    contract = RestorationV2Contract.load(parse_args().config)
    reference = contract.data["reference_definition"]
    budget = contract.data["memory_budget"]
    confirm = contract.data["data"]["roles"]["v2_confirm_primary"]
    print(
        json.dumps(
            {
                "candidate_events": budget["candidate_event_count_at_confirm_state"],
                "confirm_states": confirm["state_count"],
                "decision_step_id": reference["confirm_decision_step_id"],
                "primary_event_capacity": budget["primary_selected_event_capacity"],
                "protocol_id": contract.protocol_id,
                "reference_images_including_current": reference["total_reference_images"],
                "source_sha256": contract.source_sha256,
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
