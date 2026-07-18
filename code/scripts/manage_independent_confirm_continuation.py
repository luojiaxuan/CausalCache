#!/usr/bin/env python3
"""Build or validate the one-file continuation Execution-B runner freeze."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from causalcache.independent_confirm_continuation_contract import (
    PROTOCOL_ID,
    RUNNER_FREEZE_PATH,
    SCHEMA_VERSION,
    IndependentConfirmContinuationContract,
    canonical_json_bytes,
    continuation_topology_nonce,
    load_runner_freeze_file,
    source_inventory,
    validate_runner_freeze,
)


RUNNER_STATUS = (
    "FROZEN_CAUSALCACHE_INDEPENDENT_CONFIRM_CONTINUATION_EXECUTION_B_V1"
)


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


def _live_remote_main(root: Path) -> str:
    output = _git(root, "ls-remote", "--exit-code", "origin", "refs/heads/main")
    rows = [line.split() for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][1] != "refs/heads/main":
        raise ValueError("canonical remote main could not be resolved uniquely")
    return rows[0][0]


def _validate_clean_pushed_main(
    root: Path, contract: IndependentConfirmContinuationContract
) -> str:
    head = _git(root, "rev-parse", "HEAD")
    if (
        _git(root, "branch", "--show-current") != "main"
        or _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        or _git(root, "remote", "get-url", "origin") != contract.source["origin_url"]
        or _git(root, "rev-parse", "origin/main") != head
        or _live_remote_main(root) != head
    ):
        raise ValueError("continuation lifecycle requires clean pushed canonical main")
    return head


def _required_execution_modules(
    contract: IndependentConfirmContinuationContract,
) -> tuple[str, ...]:
    raw = contract.source.get("required_execution_modules")
    if (
        not isinstance(raw, list)
        or not raw
        or len(raw) != len(set(raw))
        or any(not isinstance(name, str) or not name for name in raw)
    ):
        raise ValueError("continuation execution-module inventory is malformed")
    modules = tuple(raw)
    for name in modules:
        importlib.import_module(name)
    return modules


def _module_inventory(
    root: Path, modules: tuple[str, ...]
) -> tuple[dict[str, Any], ...]:
    records = {}
    for name in modules:
        module = sys.modules.get(name)
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str):
            raise ValueError(f"continuation execution module has no source: {name}")
        path = Path(raw).resolve()
        if path.suffix == ".pyc" and Path(str(path)[:-1]).is_file():
            path = Path(str(path)[:-1]).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"continuation execution module is outside repo: {name}") from error
        if relative == RUNNER_FREEZE_PATH or not path.is_file():
            raise ValueError(f"continuation execution module path is unsafe: {name}")
        payload = path.read_bytes()
        records[relative] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    expected = {f"code/{name.replace('.', '/')}.py" for name in modules}
    if set(records) != expected or len(records) != len(modules):
        raise ValueError("continuation loaded-module path inventory drifted")
    return tuple(records[path] for path in sorted(records))


def _inventory_at_commit(
    root: Path, commit: str, paths: tuple[str, ...]
) -> tuple[dict[str, Any], ...]:
    result = []
    for relative in paths:
        payload = _git_bytes(root, "show", f"{commit}:{relative}")
        result.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return tuple(result)


def _git_path_exists(root: Path, commit: str, relative: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{relative}"],
        cwd=root,
        capture_output=True,
    )
    if result.returncode not in {0, 1, 128}:
        raise RuntimeError("git cat-file returned an unexpected status")
    return result.returncode == 0


def build_runner_freeze(
    *, repository_root: str | Path, contract_path: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if (root / RUNNER_FREEZE_PATH).exists():
        raise FileExistsError("continuation runner freeze already exists")
    contract = IndependentConfirmContinuationContract.load(
        contract_path, repository_root=root, require_runner_absent=True
    )
    modules = _required_execution_modules(contract)
    head = _validate_clean_pushed_main(root, contract)
    parents = _git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(parents) != 2:
        raise ValueError("continuation Source-A must be a pushed single-parent commit")
    inventory = source_inventory(contract)
    committed = _inventory_at_commit(
        root, head, tuple(contract.source["required_source_a_paths"])
    )
    if inventory != committed:
        raise ValueError("continuation Source-A working bytes differ from committed blobs")
    module_inventory = _module_inventory(root, modules)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUNNER_STATUS,
        "source_a_commit": head,
        "source_a_parent_count": 1,
        "source_a_remote_main": head,
        "contract_sha256": contract.sha256,
        "source_a_inventory": list(inventory),
        "source_a_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(list(inventory))
        ).hexdigest(),
        "loaded_module_inventory": list(module_inventory),
        "loaded_module_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(list(module_inventory))
        ).hexdigest(),
        "fresh_topology_receipt_required": True,
        "topology_receipt_nonce": continuation_topology_nonce(
            source_a_commit=head,
            contract_sha256=contract.sha256,
        ),
        "adopt_existing_payload_authorized": True,
        "restoration_access_requires_adopted_payload_receipt": True,
        "selector_reexecution_authorized": False,
        "closed_loop_requires_confirm_go": True,
    }


def validate_existing(
    *,
    repository_root: str | Path,
    contract_path: str | Path,
    expected_execution_b_commit: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    contract = IndependentConfirmContinuationContract.load(
        contract_path, repository_root=root, require_runner_absent=False
    )
    head = _validate_clean_pushed_main(root, contract)
    if head != expected_execution_b_commit:
        raise ValueError("continuation Execution-B differs from explicit expected commit")
    modules = _required_execution_modules(contract)
    value = load_runner_freeze_file(root / RUNNER_FREEZE_PATH)
    source_a = value.get("source_a_commit")
    validate_runner_freeze(contract, value, source_a_commit=str(source_a))
    parents = _git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(parents) != 2 or parents[1] != source_a:
        raise ValueError("continuation Execution-B is not direct child of Source-A")
    if _git_path_exists(root, str(source_a), RUNNER_FREEZE_PATH):
        raise ValueError("continuation runner freeze already existed in Source-A")
    diff = _git(root, "diff", "--name-status", str(source_a), head, "--").splitlines()
    if diff != [f"A\t{RUNNER_FREEZE_PATH}"]:
        raise ValueError("continuation Execution-B must add only runner freeze")
    source_paths = tuple(contract.source["required_source_a_paths"])
    source_at_commit = _inventory_at_commit(root, str(source_a), source_paths)
    if list(source_at_commit) != value["source_a_inventory"]:
        raise ValueError("continuation runner Source-A inventory drifted")
    live_modules = _module_inventory(root, modules)
    committed_modules = _inventory_at_commit(
        root, str(source_a), tuple(row["path"] for row in live_modules)
    )
    if live_modules != committed_modules or list(live_modules) != value["loaded_module_inventory"]:
        raise ValueError("continuation runner module inventory drifted")
    return {
        "status": "VALIDATED_CAUSALCACHE_INDEPENDENT_CONFIRM_CONTINUATION_EXECUTION_B_V1",
        "source_a_commit": source_a,
        "execution_b_commit": head,
        "only_changed_path": RUNNER_FREEZE_PATH,
        "runner_freeze": value,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--expected-execution-b-commit")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        value = build_runner_freeze(
            repository_root=args.repository_root, contract_path=args.contract
        )
    else:
        if not args.expected_execution_b_commit:
            raise ValueError("continuation validate requires Execution-B commit")
        value = validate_existing(
            repository_root=args.repository_root,
            contract_path=args.contract,
            expected_execution_b_commit=args.expected_execution_b_commit,
        )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
