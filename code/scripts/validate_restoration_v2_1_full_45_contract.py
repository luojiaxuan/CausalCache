"""Validate the source-only restoration-v2.1 full-45 child contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causalcache.restoration_v2_1_full_45_contract import (
    RestorationV21Full45Contract,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def validate_from_args(args: argparse.Namespace) -> dict[str, object]:
    contract = RestorationV21Full45Contract.load(
        args.config,
        repository_root=args.repository_root,
    )
    validation = contract.validation
    parent = contract.data["parent_authorization"]
    return {
        "contract_sha256": contract.source_sha256,
        "contract_valid": True,
        "external_runtime_evidence": {
            "fixed_15_fresh_immutable_archive_required": parent[
                "fixed_15_pass_manifest"
            ]["fresh_immutable_archive_validation_required_before_runtime_import"],
            "processor_fresh_immutable_evidence_required": parent[
                "processor_pass_manifest"
            ]["fresh_immutable_evidence_validation_required_before_runtime_import"],
            "status": "NOT_CONSUMED_BY_SOURCE_ONLY_VALIDATOR",
        },
        "fixed_state_denominator": validation["fixed_state_denominator"],
        "maximum_schedule": {
            "generation_call_count": validation[
                "maximum_generation_call_count"
            ],
            "teacher_forward_count": validation[
                "maximum_teacher_forward_count"
            ],
            "kl_measurement_count": validation[
                "maximum_kl_measurement_count"
            ],
        },
        "policy_execution_authorized_by_this_validator": False,
        "gpu_execution_authorized_by_this_validator": False,
        "protocol_id": contract.protocol_id,
        "state_projection_sha256": validation[
            "state_projection_sha256"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = validate_from_args(_build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
