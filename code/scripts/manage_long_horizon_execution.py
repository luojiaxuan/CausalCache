#!/usr/bin/env python3
"""Freeze and validate the A -> selection-B -> runner-C Git lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from causalcache.long_horizon_contract import (
    CANONICAL_CONFIG_PATH,
    RUNNER_FREEZE_B_PATH,
    SELECTION_MANIFEST_FREEZE_PATH,
    LongHorizonContract,
)
from causalcache.long_horizon_execution import (
    PROTOCOL_ID,
    RUNNER_STATUS,
    SCHEMA_VERSION,
    SELECTOR_PREPARATION_MANIFEST_PATH,
    inventory_sha256,
    load_json_object,
    pretty_json_bytes,
    sha256_bytes,
    validate_runner_config,
)


VALIDATION_STATUS = "VALIDATED_CAUSALCACHE_LONG_HORIZON_EXECUTION_B_V1"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True
    ).stdout


def _remote_branch_head(root: Path, branch: str) -> str:
    ref = f"refs/heads/{branch}"
    rows = [
        line.split()
        for line in _git(root, "ls-remote", "--exit-code", "origin", ref).splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or rows[0][1] != ref:
        raise ValueError("canonical remote branch could not be resolved uniquely")
    return rows[0][0]


def _validate_clean_pushed_branch(root: Path, branch: str) -> str:
    head = _git(root, "rev-parse", "HEAD")
    remote_ref = f"origin/{branch}"
    if (
        _git(root, "branch", "--show-current") != branch
        or _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        or _git(root, "rev-parse", remote_ref) != head
        or _remote_branch_head(root, branch) != head
    ):
        raise ValueError("formal lifecycle requires a clean pushed frozen branch")
    return head


def _single_parent(root: Path, commit: str) -> str:
    row = _git(root, "rev-list", "--parents", "-n", "1", commit).split()
    if len(row) != 2 or row[0] != commit:
        raise ValueError("formal freeze commits must have exactly one parent")
    return row[1]


def _git_path_exists(root: Path, commit: str, relative: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{relative}"],
        cwd=root,
        capture_output=True,
    )
    if result.returncode not in {0, 1, 128}:
        raise RuntimeError("git cat-file returned an unexpected status")
    return result.returncode == 0


def _exact_added_diff(root: Path, parent: str, child: str, path: str) -> None:
    rows = _git(root, "diff", "--name-status", parent, child, "--").splitlines()
    if rows != [f"A\t{path}"]:
        raise ValueError(f"formal freeze must add only {path}")


def _inventory_at_commit(
    root: Path, commit: str, paths: tuple[str, ...]
) -> tuple[dict[str, Any], ...]:
    records = []
    for relative in sorted(paths):
        payload = _git_bytes(root, "show", f"{commit}:{relative}")
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return tuple(records)


def _load_contract(root: Path, contract_path: str | Path) -> LongHorizonContract:
    return LongHorizonContract.load(
        contract_path, repository_root=root, require_runner_absent=False
    )


def _selection_freeze_chain(
    root: Path,
    *,
    selection_commit: str,
    contract: LongHorizonContract,
) -> tuple[str, bytes]:
    source_a = _single_parent(root, selection_commit)
    _exact_added_diff(
        root,
        source_a,
        selection_commit,
        SELECTION_MANIFEST_FREEZE_PATH,
    )
    if _git_path_exists(root, source_a, SELECTION_MANIFEST_FREEZE_PATH):
        raise ValueError("selection manifest already existed in Source-A")
    if _git_path_exists(root, source_a, RUNNER_FREEZE_B_PATH) or _git_path_exists(
        root, selection_commit, RUNNER_FREEZE_B_PATH
    ):
        raise ValueError("runner config existed before Execution-B freeze")
    selection_payload = _git_bytes(
        root, "show", f"{selection_commit}:{SELECTION_MANIFEST_FREEZE_PATH}"
    )
    return source_a, selection_payload


def _artifact_binding(
    *,
    repo: str,
    repo_type: str,
    revision: str,
    relative_path: str,
    local_path: str | Path,
) -> dict[str, Any]:
    payload = Path(local_path).read_bytes()
    return {
        "repo": repo,
        "repo_type": repo_type,
        "revision": revision,
        "path": relative_path,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _preparation_binding(
    *, repo_id: str, revision: str, local_path: str | Path
) -> dict[str, Any]:
    payload = Path(local_path).read_bytes()
    return {
        "repo_id": repo_id,
        "revision": revision,
        "path": SELECTOR_PREPARATION_MANIFEST_PATH,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def build_runner_freeze(
    *,
    repository_root: str | Path,
    contract_path: str | Path,
    substrate_repo: str,
    substrate_revision: str,
    substrate_manifest_path: str | Path,
    substrate_manifest_relative_path: str,
    selector_seal_repo: str,
    selector_seal_revision: str,
    selector_preparation_manifest_path: str | Path,
    selector_seal_path: str | Path,
    selector_seal_relative_path: str,
    gpu_ids: tuple[str, ...],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    runner_path = root / RUNNER_FREEZE_B_PATH
    if os.path.lexists(runner_path):
        raise FileExistsError("Execution-B runner config already exists")
    contract = _load_contract(root, contract_path)
    branch = str(contract.data["source_freeze"]["branch"])
    selection_commit = _validate_clean_pushed_branch(root, branch)
    source_a, committed_selection_payload = _selection_freeze_chain(
        root, selection_commit=selection_commit, contract=contract
    )
    working_selection_path = root / SELECTION_MANIFEST_FREEZE_PATH
    if working_selection_path.read_bytes() != committed_selection_payload:
        raise ValueError("working selection manifest differs from selection freeze")
    required_paths = tuple(contract.data["source_freeze"]["required_source_a_paths"])
    inventory = _inventory_at_commit(root, source_a, required_paths)
    contract_at_source_a = _git_bytes(root, "show", f"{source_a}:{CANONICAL_CONFIG_PATH}")
    if sha256_bytes(contract_at_source_a) != contract.sha256:
        raise ValueError("Source-A contract blob differs from the live frozen contract")
    selection_manifest = load_json_object(
        working_selection_path, label="selection freeze manifest"
    )[1]
    if selection_manifest.get("generator", {}).get("git_revision") != source_a:
        raise ValueError("selection freeze generator does not bind Source-A")
    substrate_binding = _artifact_binding(
        repo=substrate_repo,
        repo_type="dataset",
        revision=substrate_revision,
        relative_path=substrate_manifest_relative_path,
        local_path=substrate_manifest_path,
    )
    seal_binding = _artifact_binding(
        repo=selector_seal_repo,
        repo_type="dataset",
        revision=selector_seal_revision,
        relative_path=selector_seal_relative_path,
        local_path=selector_seal_path,
    )
    preparation_binding = _preparation_binding(
        repo_id=selector_seal_repo,
        revision=selector_seal_revision,
        local_path=selector_preparation_manifest_path,
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUNNER_STATUS,
        "source_a": {
            "branch": branch,
            "commit": source_a,
            "remote_ref_commit": source_a,
            "contract_path": CANONICAL_CONFIG_PATH,
            "contract_sha256": contract.sha256,
            "required_inventory": list(inventory),
            "required_inventory_sha256": inventory_sha256(inventory),
        },
        "selection_manifest": {
            "path": SELECTION_MANIFEST_FREEZE_PATH,
            "sha256": sha256_bytes(committed_selection_payload),
            "size_bytes": len(committed_selection_payload),
            "freeze_commit": selection_commit,
            "freeze_parent_source_a_commit": source_a,
        },
        "substrate": substrate_binding,
        "selector_preparation_manifest": preparation_binding,
        "selector_seal": seal_binding,
        "worker_topology": {
            "host_alias": "hyper00",
            "worker_count": 4,
            "gpu_assignments": [
                {
                    "worker_index": index,
                    "host_gpu_id": host_gpu_id,
                    "container_cuda_ordinal": index,
                }
                for index, host_gpu_id in enumerate(gpu_ids)
            ],
            "selection_rank_ranges": [[0, 5], [6, 11], [12, 17], [18, 23]],
            "trajectories_per_worker": 6,
            "one_explicit_gpu_per_worker": True,
        },
        "access_firewall": {
            "reserve_access_count": 0,
            "top_up_count": 0,
            "replacement_count": 0,
            "post_selection_filter_count": 0,
            "selector_reseal_count": 0,
            "threshold_update_count": 0,
            "old_confirm_access_count": 0,
            "closed_loop_access_count": 0,
            "sealed_test_access_count": 0,
        },
        "operation_accounting": dict(contract.data["operation_accounting"]),
        "authorization": {
            "development_policy_access": True,
            "development_restoration_access": True,
            "selector_sets_already_label_blind_sealed": True,
            "reserve_access": False,
            "old_confirm_access": False,
            "closed_loop_access": False,
            "sealed_test_access": False,
        },
    }
    validate_runner_config(
        config,
        contract=contract,
        selection_manifest_payload=committed_selection_payload,
        substrate_manifest_payload=Path(substrate_manifest_path).read_bytes(),
        selector_preparation_manifest_payload=Path(
            selector_preparation_manifest_path
        ).read_bytes(),
        selector_seal_payload=Path(selector_seal_path).read_bytes(),
    )
    return config


def write_runner_freeze(
    *,
    output_path: str | Path,
    value: dict[str, Any],
    repository_root: str | Path,
) -> None:
    root = Path(repository_root).resolve()
    expected = (root / RUNNER_FREEZE_B_PATH).resolve()
    output = Path(output_path).resolve()
    if output != expected:
        raise ValueError("runner freeze must use the canonical repository path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(pretty_json_bytes(value))


def validate_existing(
    *,
    repository_root: str | Path,
    contract_path: str | Path,
    expected_execution_b_commit: str,
    substrate_manifest_path: str | Path | None = None,
    selector_preparation_manifest_path: str | Path | None = None,
    selector_seal_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    contract = _load_contract(root, contract_path)
    branch = str(contract.data["source_freeze"]["branch"])
    execution_b = _validate_clean_pushed_branch(root, branch)
    if execution_b != expected_execution_b_commit:
        raise ValueError("Execution-B HEAD differs from the explicitly expected commit")
    selection_commit = _single_parent(root, execution_b)
    _exact_added_diff(root, selection_commit, execution_b, RUNNER_FREEZE_B_PATH)
    source_a, committed_selection_payload = _selection_freeze_chain(
        root, selection_commit=selection_commit, contract=contract
    )
    runner_payload, runner = load_json_object(
        root / RUNNER_FREEZE_B_PATH, label="Execution-B runner config"
    )
    if runner_payload != pretty_json_bytes(runner):
        raise ValueError("Execution-B runner config is not canonical pretty JSON")
    substrate_payload = (
        None
        if substrate_manifest_path is None
        else Path(substrate_manifest_path).read_bytes()
    )
    seal_payload = (
        None if selector_seal_path is None else Path(selector_seal_path).read_bytes()
    )
    preparation_payload = (
        None
        if selector_preparation_manifest_path is None
        else Path(selector_preparation_manifest_path).read_bytes()
    )
    validated = validate_runner_config(
        runner,
        contract=contract,
        selection_manifest_payload=committed_selection_payload,
        substrate_manifest_payload=substrate_payload,
        selector_preparation_manifest_payload=preparation_payload,
        selector_seal_payload=seal_payload,
    )
    if (
        validated["source_a"]["commit"] != source_a
        or validated["selection_manifest"]["freeze_commit"] != selection_commit
    ):
        raise ValueError("runner config does not bind the exact A -> B ancestry")
    inventory = _inventory_at_commit(
        root,
        source_a,
        tuple(contract.data["source_freeze"]["required_source_a_paths"]),
    )
    if list(inventory) != validated["source_a"]["required_inventory"]:
        raise ValueError("runner Source-A inventory differs from committed blobs")
    if _git_path_exists(root, source_a, SELECTION_MANIFEST_FREEZE_PATH) or _git_path_exists(
        root, selection_commit, RUNNER_FREEZE_B_PATH
    ):
        raise ValueError("A -> B -> C freeze boundaries are not isolated")
    return {
        "status": VALIDATION_STATUS,
        "source_a_commit": source_a,
        "selection_freeze_commit": selection_commit,
        "execution_b_commit": execution_b,
        "selection_manifest_sha256": validated["selection_manifest"]["sha256"],
        "runner_config_sha256": sha256_bytes(runner_payload),
        "substrate_revision": validated["substrate"]["revision"],
        "selector_seal_revision": validated["selector_seal"]["revision"],
        "only_execution_b_changed_path": RUNNER_FREEZE_B_PATH,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repository-root", type=Path, required=True)
    common.add_argument("--contract", type=Path, default=Path(CANONICAL_CONFIG_PATH))
    build = subparsers.add_parser("build", parents=[common])
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--substrate-repo", required=True)
    build.add_argument("--substrate-revision", required=True)
    build.add_argument("--substrate-manifest", type=Path, required=True)
    build.add_argument("--substrate-manifest-relative-path", required=True)
    build.add_argument("--selector-seal-repo", required=True)
    build.add_argument("--selector-seal-revision", required=True)
    build.add_argument("--selector-preparation-manifest", type=Path, required=True)
    build.add_argument("--selector-seal", type=Path, required=True)
    build.add_argument("--selector-seal-relative-path", required=True)
    build.add_argument("--gpu-id", action="append", required=True)
    validate = subparsers.add_parser("validate", parents=[common])
    validate.add_argument("--expected-execution-b-commit", required=True)
    validate.add_argument("--substrate-manifest", type=Path)
    validate.add_argument("--selector-preparation-manifest", type=Path)
    validate.add_argument("--selector-seal", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        value = build_runner_freeze(
            repository_root=args.repository_root,
            contract_path=args.contract,
            substrate_repo=args.substrate_repo,
            substrate_revision=args.substrate_revision,
            substrate_manifest_path=args.substrate_manifest,
            substrate_manifest_relative_path=args.substrate_manifest_relative_path,
            selector_seal_repo=args.selector_seal_repo,
            selector_seal_revision=args.selector_seal_revision,
            selector_preparation_manifest_path=args.selector_preparation_manifest,
            selector_seal_path=args.selector_seal,
            selector_seal_relative_path=args.selector_seal_relative_path,
            gpu_ids=tuple(args.gpu_id),
        )
        write_runner_freeze(
            output_path=args.output,
            value=value,
            repository_root=args.repository_root,
        )
        result = {
            "status": RUNNER_STATUS,
            "output": str(args.output),
            "runner_config_sha256": sha256_bytes(pretty_json_bytes(value)),
            "source_a_commit": value["source_a"]["commit"],
            "selection_freeze_commit": value["selection_manifest"]["freeze_commit"],
        }
    else:
        result = validate_existing(
            repository_root=args.repository_root,
            contract_path=args.contract,
            expected_execution_b_commit=args.expected_execution_b_commit,
            substrate_manifest_path=args.substrate_manifest,
            selector_preparation_manifest_path=args.selector_preparation_manifest,
            selector_seal_path=args.selector_seal,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
