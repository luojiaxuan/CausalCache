#!/usr/bin/env python3
"""Source-only validator for the restoration-only confirm continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from causalcache.independent_confirm_continuation_contract import (
    FAILURE_EVIDENCE_COMMIT,
    IndependentConfirmContinuationContract,
    source_inventory,
)


def _git_bytes(root: Path, revision: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def validate_source_only(
    *, repository_root: str | Path, contract_path: str | Path
) -> dict[str, object]:
    root = Path(repository_root).resolve()
    contract = IndependentConfirmContinuationContract.load(
        contract_path, repository_root=root, require_runner_absent=True
    )
    failed = contract.data["failed_attempt"]
    for path_key, sha_key in (
        ("failure_path", "failure_sha256"),
        ("summary_path", "summary_sha256"),
    ):
        historical = _git_bytes(root, FAILURE_EVIDENCE_COMMIT, failed[path_key])
        live = (root / failed[path_key]).read_bytes()
        if historical != live or hashlib.sha256(historical).hexdigest() != failed[sha_key]:
            raise ValueError("retained v1 evidence differs from its committed tombstone")
    inventory = source_inventory(contract)
    return {
        "schema_version": "1.0.0",
        "protocol_id": contract.data["protocol_id"],
        "status": "VALID_SOURCE_ONLY_INDEPENDENT_CONFIRM_CONTINUATION_V1",
        "contract_sha256": contract.sha256,
        "source_path_count": len(inventory),
        "source_inventory_sha256": hashlib.sha256(
            json.dumps(
                list(inventory),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "parent_failure_replayed_from_git": True,
        "execution_b_runner_absent": True,
        "payload_download_count": 0,
        "confirm_semantic_decode_count": 0,
        "reference_policy_output_count": 0,
        "restoration_teacher_forward_count": 0,
        "model_load_count": 0,
        "gpu_operation_count": 0,
        "huggingface_mutation_count": 0,
        "closed_loop_episode_count": 0,
        "sealed_androidworld_test_access_count": 0,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    result = validate_source_only(
        repository_root=args.repository_root, contract_path=args.contract
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
