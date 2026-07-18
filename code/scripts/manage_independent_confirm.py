#!/usr/bin/env python3
"""Build or validate the one-file Execution-B runner freeze."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from causalcache.independent_confirm_contract import (
    PROTOCOL_ID,
    RUNNER_FREEZE_PATH,
    SCHEMA_VERSION,
    IndependentConfirmContract,
    canonical_json_bytes,
    load_runner_freeze_file,
    source_inventory,
    validate_runner_freeze,
)


RUNNER_STATUS = "FROZEN_CAUSALCACHE_INDEPENDENT_CONFIRM_EXECUTION_B_V1"


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
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _required_execution_modules(
    contract: IndependentConfirmContract,
) -> tuple[str, ...]:
    raw = contract.source.get("required_execution_modules")
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(name, str) or not name for name in raw)
        or len(raw) != len(set(raw))
    ):
        raise ValueError("required execution-module inventory is malformed")
    modules = tuple(raw)
    for name in modules:
        importlib.import_module(name)
    return modules


def _module_inventory(
    root: Path, module_names: tuple[str, ...]
) -> tuple[dict[str, Any], ...]:
    records: dict[str, dict[str, Any]] = {}
    for name in module_names:
        module = sys.modules.get(name)
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str):
            raise ValueError(f"required execution module has no source file: {name}")
        path = Path(raw).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"execution module is outside the repository: {name}") from error
        if path.suffix == ".pyc":
            source = Path(str(path)[:-1])
            if source.is_file():
                path = source
                relative = path.relative_to(root).as_posix()
        if not path.is_file() or relative == RUNNER_FREEZE_PATH:
            raise ValueError(f"required execution module source is unsafe: {name}")
        payload = path.read_bytes()
        records[relative] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    expected_paths = {
        f"code/{name.replace('.', '/')}.py" for name in module_names
    }
    if len(records) != len(module_names) or set(records) != expected_paths:
        raise ValueError("required execution module path inventory drifted")
    return tuple(records[path] for path in sorted(records))


def _live_remote_main(root: Path) -> str:
    output = _git(root, "ls-remote", "--exit-code", "origin", "refs/heads/main")
    rows = [line.split() for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != "refs/heads/main":
        raise ValueError("canonical remote main could not be resolved uniquely")
    return rows[0][0]


def _validate_clean_pushed_main(root: Path, contract: IndependentConfirmContract) -> str:
    head = _git(root, "rev-parse", "HEAD")
    if (
        _git(root, "branch", "--show-current") != "main"
        or _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        or _git(root, "remote", "get-url", "origin") != contract.source["origin_url"]
        or _git(root, "rev-parse", "origin/main") != head
        or _live_remote_main(root) != head
    ):
        raise ValueError("formal lifecycle requires clean pushed canonical main")
    return head


def _inventory_at_commit(
    root: Path, commit: str, paths: tuple[str, ...]
) -> tuple[dict[str, Any], ...]:
    records = []
    for relative in paths:
        payload = _git_bytes(root, "show", f"{commit}:{relative}")
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return tuple(records)


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
    *, repository_root: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if (root / RUNNER_FREEZE_PATH).exists():
        raise FileExistsError("runner freeze already exists")
    contract = IndependentConfirmContract.load(
        contract_path,
        repository_root=root,
        require_runner_absent=True,
    )
    modules = _required_execution_modules(contract)
    head = _validate_clean_pushed_main(root, contract)
    parents = _git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(parents) != 2:
        raise ValueError("runner freeze requires a clean pushed single-parent Source-A")
    inventory = source_inventory(contract)
    committed_inventory = _inventory_at_commit(
        root, head, tuple(contract.source["required_source_a_paths"])
    )
    if inventory != committed_inventory:
        raise ValueError("Source-A working bytes differ from the committed blobs")
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
        "confirm_access_authorized": True,
        "restoration_access_requires_payload_commit": True,
        "closed_loop_train_requires_confirm_go": True,
        "sealed_test_requires_development_go": True,
    }


def validate_existing(
    *,
    repository_root: str | Path,
    contract_path: str | Path,
    expected_execution_b_commit: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    contract = IndependentConfirmContract.load(
        contract_path,
        repository_root=root,
        require_runner_absent=False,
    )
    head = _validate_clean_pushed_main(root, contract)
    if head != expected_execution_b_commit:
        raise ValueError("Execution-B HEAD differs from the explicitly expected commit")
    modules = _required_execution_modules(contract)
    path = root / RUNNER_FREEZE_PATH
    value = load_runner_freeze_file(path)
    source_a = value.get("source_a_commit")
    validate_runner_freeze(contract, value, source_a_commit=source_a)
    parents = _git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(parents) != 2 or parents[1] != source_a:
        raise ValueError("Execution-B is not the direct child of Source-A")
    source_parents = _git(
        root, "rev-list", "--parents", "-n", "1", str(source_a)
    ).split()
    if len(source_parents) != 2:
        raise ValueError("Source-A is not a single-parent commit")
    if _git_path_exists(root, str(source_a), RUNNER_FREEZE_PATH):
        raise ValueError("runner freeze already existed in Source-A")
    diff = _git(
        root, "diff", "--name-status", str(source_a), head, "--"
    ).splitlines()
    if diff != [f"A\t{RUNNER_FREEZE_PATH}"]:
        raise ValueError("Execution-B must add only the runner-freeze file")

    live_source_inventory = source_inventory(contract)
    source_commit_inventory = _inventory_at_commit(
        root, str(source_a), tuple(contract.source["required_source_a_paths"])
    )
    frozen_source_inventory = tuple(value["source_a_inventory"])
    if (
        live_source_inventory != source_commit_inventory
        or list(source_commit_inventory) != list(frozen_source_inventory)
    ):
        raise ValueError("Execution-B source inventory differs from Source-A")
    live_module_inventory = _module_inventory(root, modules)
    source_module_paths = tuple(item["path"] for item in live_module_inventory)
    source_module_inventory = _inventory_at_commit(root, str(source_a), source_module_paths)
    if (
        live_module_inventory != source_module_inventory
        or list(live_module_inventory) != list(value["loaded_module_inventory"])
    ):
        raise ValueError("Execution-B loaded-module inventory differs from Source-A")
    return {
        "status": "VALIDATED_CAUSALCACHE_INDEPENDENT_CONFIRM_EXECUTION_B_V1",
        "source_a_commit": source_a,
        "execution_b_commit": _git(root, "rev-parse", "HEAD"),
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
            repository_root=args.repository_root,
            contract_path=args.contract,
        )
    else:
        if not args.expected_execution_b_commit:
            raise ValueError("validate requires --expected-execution-b-commit")
        value = validate_existing(
            repository_root=args.repository_root,
            contract_path=args.contract,
            expected_execution_b_commit=args.expected_execution_b_commit,
        )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
