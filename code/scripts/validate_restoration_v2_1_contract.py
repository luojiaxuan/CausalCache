"""Validate the immutable restoration-v2.1 pilot contract and CPU preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causalcache.restoration_v2_1_contract import (
    RestorationV21PilotContract,
    _load_json_object,
    validate_restoration_v2_1_processor_preflight,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processor-preflight", type=Path)
    return parser


def validate_from_args(args: argparse.Namespace) -> dict[str, object]:
    root = args.repository_root.resolve()
    contract = RestorationV21PilotContract.load(
        args.config,
        repository_root=root,
    )
    data = contract.data
    preflight_result = None
    if args.processor_preflight is not None:
        audit = _load_json_object(args.processor_preflight.resolve())
        preflight_result = validate_restoration_v2_1_processor_preflight(
            audit,
            contract_sha256=contract.source_sha256,
            selection_manifest_sha256=data["data"]["selection_manifest"]["sha256"],
            policy_interface_source_sha256=data["policy_interface"]["source"]["sha256"],
        )
    return {
        "contract_sha256": contract.source_sha256,
        "contract_valid": True,
        "fixed_pilot_states": contract.validation["pilot_state_count"],
        "planned_generation_calls": contract.validation[
            "planned_generation_call_count"
        ],
        "policy_generation_authorized_by_this_validator": False,
        "processor_preflight": (
            preflight_result
            if preflight_result is not None
            else {
                "required_prompt_count": contract.validation[
                    "processor_preflight_prompt_count"
                ],
                "status": "PENDING_SEPARATE_POLICY_OUTPUT_FREE_AUDIT",
                "valid": False,
            }
        ),
        "protocol_id": contract.protocol_id,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = validate_from_args(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
