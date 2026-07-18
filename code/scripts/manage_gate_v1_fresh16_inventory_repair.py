#!/usr/bin/env python3
"""Validate, freeze, execute, or replay the fresh-16 inventory repair."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.gate_v1_fresh16_inventory_repair_contract import (
    CANONICAL_CONFIG_PATH,
    load_frozen_fresh16_inventory_repair_contract,
    validate_fresh16_inventory_repair_source_only_contract,
)
from causalcache.gate_v1_fresh16_inventory_repair_runner import (
    capture_docker_inspect_receipt,
    execute_fresh16_inventory_repair,
    materialize_runner_freeze,
    validate_retained_v1_failure,
    validate_execution_b_source,
    validate_source_a,
)
from scripts import manage_gate_v1_fresh16_evaluation as base_manager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def source(name: str) -> argparse.ArgumentParser:
        child = commands.add_parser(name)
        child.add_argument("--repository-root", type=Path, required=True)
        child.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
        child.add_argument("--source-a-git-commit")
        return child

    source("validate-source")
    source("materialize-runner-freeze")

    capture = commands.add_parser("capture-runtime")
    capture.add_argument("--repository-root", type=Path, required=True)
    capture.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    capture.add_argument("--execution-b-git-commit", required=True)
    capture.add_argument("--container-name", required=True)
    capture.add_argument("--host-data-root", type=Path, required=True)

    for name in ("run", "validate"):
        child = commands.add_parser(name)
        child.add_argument("--repository-root", type=Path, required=True)
        child.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
        child.add_argument("--execution-b-git-commit", required=True)
        child.add_argument("--hf-token-file", type=Path, required=True)
        child.add_argument("--data-root", type=Path, required=True)
        child.add_argument("--fresh-download-parent", type=Path, required=True)
        child.add_argument("--model-dir", type=Path)
        child.add_argument("--snapshot-manifest", type=Path)
        child.add_argument("--docker-inspect-receipt", type=Path)
        child.add_argument("--device", action="append", default=[])
        child.add_argument("--gpu-uuid", action="append", default=[])
    return parser


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    contract = load_frozen_fresh16_inventory_repair_contract(
        args.contract,
        repository_root=args.repository_root,
    )
    if args.command == "validate-source":
        source_only = validate_fresh16_inventory_repair_source_only_contract(
            args.contract,
            repository_root=args.repository_root,
        )
        return {
            **source_only,
            "runner": validate_source_a(
                contract,
                expected_source_a_git_commit=args.source_a_git_commit,
            ),
        }
    if args.command == "materialize-runner-freeze":
        validate_fresh16_inventory_repair_source_only_contract(
            args.contract,
            repository_root=args.repository_root,
        )
        return materialize_runner_freeze(
            contract,
            expected_source_a_git_commit=args.source_a_git_commit,
        )
    if args.command == "capture-runtime":
        source = validate_execution_b_source(
            contract,
            expected_execution_b_git_commit=args.execution_b_git_commit,
        )
        return capture_docker_inspect_receipt(
            contract,
            source=source,
            container_name=args.container_name,
            host_data_root=args.host_data_root,
        )

    if args.docker_inspect_receipt is None:
        raise ValueError("run and validate require the frozen Docker inspect receipt")
    if args.command == "run" and (
        args.model_dir is None
        or args.snapshot_manifest is None
        or len(args.device) != 2
        or len(args.gpu_uuid) != 2
    ):
        raise ValueError(
            "run requires model/snapshot/runtime receipt and exactly two devices/UUIDs"
        )

    validate_execution_b_source(
        contract,
        expected_execution_b_git_commit=args.execution_b_git_commit,
    )
    retained_evidence = validate_retained_v1_failure(contract, args.data_root)
    token = base_manager._secure_token(args.hf_token_file)
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)

    def authenticated_download(**kwargs: Any) -> str:
        return hf_hub_download(**kwargs, token=token)

    def cached_retained_evidence(
        checked_contract: Any, checked_root: Path
    ) -> Mapping[str, Any]:
        if checked_contract is not contract or Path(checked_root) != args.data_root:
            raise ValueError("cached retained-evidence invocation drifted")
        return retained_evidence

    return execute_fresh16_inventory_repair(
        mode=args.command,
        contract=contract,
        api=api,
        download_fn=authenticated_download,
        operation_factory=CommitOperationAdd,
        expected_execution_b_git_commit=args.execution_b_git_commit,
        data_root=args.data_root,
        fresh_download_parent=args.fresh_download_parent,
        model_dir=args.model_dir or Path("."),
        snapshot_manifest=args.snapshot_manifest or Path("."),
        docker_inspect_receipt=args.docker_inspect_receipt,
        devices=tuple(args.device),
        gpu_uuids=tuple(args.gpu_uuid),
        evidence_validator=cached_retained_evidence,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
