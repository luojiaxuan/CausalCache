"""Run the CPU-only policy-vision v3 validation repair."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from causalcache.restoration_v2_2_policy_vision_v3_validation_repair import (
    replay_exact_producer_artifact,
    read_completion_seal,
    read_repair_attempt_ledger,
    run_formal_validation_repair,
    validate_existing_audit_result,
    validate_no_nvidia_devices,
    validate_producer_attempt_ledger,
)
from causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract import (
    EXPECTED_OUTPUT_FILES,
    VALIDATE_STATUS,
    PolicyVisionV3ValidationRepairContract,
    sha256_bytes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "validate"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--labels-archive", type=Path, required=True)
    parser.add_argument("--producer-attempt-ledger", type=Path, required=True)
    parser.add_argument("--validation-source-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--completion-seal", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(sys.argv) if argv is None else [__name__, *argv]
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    contract = PolicyVisionV3ValidationRepairContract.load(
        args.contract.resolve(),
        repository_root=root,
    )
    output = args.output_dir.resolve()
    labels = args.labels_archive.resolve()
    producer_ledger = args.producer_attempt_ledger.resolve()
    if args.mode == "run":
        files = run_formal_validation_repair(
            contract=contract,
            labels_archive=labels,
            producer_attempt_ledger=producer_ledger,
            attempt_ledger=args.attempt_ledger.resolve(),
            completion_seal=args.completion_seal.resolve(),
            output_dir=output,
            validation_source_git_commit=args.validation_source_git_commit,
            argv=raw_argv,
        )
        status = contract.data["output_contract"]["run_status"]
    else:
        no_nvidia = validate_no_nvidia_devices()
        validation_source = contract.validate_clean_pushed_descendant(
            args.validation_source_git_commit,
        )
        producer_ledger_before = validate_producer_attempt_ledger(
            producer_ledger, contract
        )
        repair_ledger_before = read_repair_attempt_ledger(
            args.attempt_ledger.resolve()
        )
        completion_seal_before = read_completion_seal(
            args.completion_seal.resolve()
        )
        replay = replay_exact_producer_artifact(
            contract=contract,
            labels_archive=labels,
        )
        if not all(
            replay.rebuilt_files[name] == replay.producer_files_before[name]
            for name in replay.rebuilt_files
        ):
            raise ValueError("read-only replay differs from producer artifact")
        files = validate_existing_audit_result(
            output,
            contract=contract,
            replay=replay,
            validation_source=validation_source,
            producer_ledger_payload=producer_ledger_before,
            repair_ledger=repair_ledger_before,
            completion_seal=completion_seal_before,
            actual_no_nvidia_runtime=no_nvidia,
        )
        if (
            validate_producer_attempt_ledger(producer_ledger, contract)
            != producer_ledger_before
        ):
            raise ValueError("producer GPU ledger changed during read-only replay")
        if (
            read_repair_attempt_ledger(args.attempt_ledger.resolve())
            != repair_ledger_before
        ):
            raise ValueError("validation-repair ledger changed during read-only replay")
        if (
            read_completion_seal(args.completion_seal.resolve())
            != completion_seal_before
        ):
            raise ValueError(
                "validation-repair completion seal changed during read-only replay"
            )
        contract.validate_clean_pushed_descendant(
            args.validation_source_git_commit,
        )
        validate_no_nvidia_devices()
        status = VALIDATE_STATUS
    print(
        json.dumps(
            {
                "status": status,
                "producer_source_git_commit": contract.data["producer"][
                    "producer_source_git_commit"
                ],
                "validation_source_git_commit": args.validation_source_git_commit,
                "producer_artifact_directory": str(contract.artifact_directory),
                "audit_result_directory": str(output),
                "device": "cpu",
                "gpu_or_model_operation_count": 0,
                "files": {
                    name: {
                        "size_bytes": len(files[name]),
                        "sha256": sha256_bytes(files[name]),
                    }
                    for name in EXPECTED_OUTPUT_FILES
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
