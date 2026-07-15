"""Materialize the frozen restoration-v2 selection and exposure ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from causalcache.data.restoration_v2_selection import (
    attach_state_content_witnesses,
    build_exposure_ledger,
    build_selection_manifest,
    pretty_json_bytes,
    reconstruct_eligible_pool,
    sha256_bytes,
)
from causalcache.restoration_v2_contract import RestorationV2Contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-contract", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--v1-summary", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output-selection", type=Path, required=True)
    parser.add_argument("--output-exposure", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> tuple[bytes, dict]:
    payload = path.read_bytes()
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload, parsed


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _verify_git_checkout(repo_root: Path, expected_revision: str) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    if actual != expected_revision:
        raise ValueError(
            f"--git-revision does not match checkout HEAD: {expected_revision} != {actual}"
        )
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        text=True,
    )
    if status:
        raise ValueError("materialization requires a clean Git checkout")
    pushed_main = subprocess.check_output(
        ["git", "rev-parse", "origin/main"],
        cwd=repo_root,
        text=True,
    ).strip()
    if pushed_main != expected_revision:
        raise ValueError(
            "materialization revision must equal the fetched canonical origin/main"
        )


def main() -> None:
    args = parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.git_revision) is None:
        raise ValueError("--git-revision must be a full 40-character Git commit")
    if args.output_selection.resolve() == args.output_exposure.resolve():
        raise ValueError("selection and exposure outputs must be different files")
    if args.output_selection.exists() or args.output_exposure.exists():
        raise FileExistsError("selection and exposure outputs must not already exist")

    repo_root = Path(__file__).resolve().parents[2]
    _verify_git_checkout(repo_root, args.git_revision)

    contract = RestorationV2Contract.load(args.v2_contract)
    v1_config_bytes, v1_config = _load_json(args.v1_config)
    source_manifest_bytes, source_manifest = _load_json(args.source_file_manifest)
    parent_manifest_bytes, parent_manifest = _load_json(args.parent_manifest)
    v1_summary_bytes, v1_summary = _load_json(args.v1_summary)

    reconstruction = reconstruct_eligible_pool(
        source_root=args.source_root,
        v1_config=v1_config,
        source_file_manifest=source_manifest,
    )
    module_path = Path(__file__).resolve().parents[1] / "causalcache" / "data" / (
        "restoration_v2_selection.py"
    )
    validator_path = Path(__file__).with_name("validate_restoration_v2_selection.py")
    selection = build_selection_manifest(
        reconstruction=reconstruction,
        v1_config=v1_config,
        v1_config_sha256=sha256_bytes(v1_config_bytes),
        v2_contract=contract.data,
        v2_contract_sha256=contract.source_sha256,
        parent_manifest=parent_manifest,
        parent_manifest_sha256=sha256_bytes(parent_manifest_bytes),
        source_file_manifest=source_manifest,
        source_file_manifest_sha256=sha256_bytes(source_manifest_bytes),
        generator={
            "git_revision": args.git_revision,
            "module_path": "code/causalcache/data/restoration_v2_selection.py",
            "module_sha256": _source_sha256(module_path),
            "cli_path": "code/scripts/materialize_restoration_v2_selection.py",
            "cli_sha256": _source_sha256(Path(__file__)),
            "validator_path": "code/scripts/validate_restoration_v2_selection.py",
            "validator_sha256": _source_sha256(validator_path),
        },
    )
    selection = attach_state_content_witnesses(
        selection,
        source_root=args.source_root,
        source_file_manifest=source_manifest,
        v1_config=v1_config,
        v2_contract=contract.data,
    )
    selection_payload = pretty_json_bytes(selection)
    selection_sha256 = sha256_bytes(selection_payload)
    exposure = build_exposure_ledger(
        selection,
        selection_manifest_sha256=selection_sha256,
        v1_summary_sha256=sha256_bytes(v1_summary_bytes),
        v1_summary=v1_summary,
    )
    exposure_payload = pretty_json_bytes(exposure)

    _exclusive_write(args.output_selection, selection_payload)
    _exclusive_write(args.output_exposure, exposure_payload)
    print(
        json.dumps(
            {
                "outcome": "PASSED_PREOUTPUT_SELECTION",
                "selection_path": str(args.output_selection),
                "selection_sha256": selection_sha256,
                "exposure_path": str(args.output_exposure),
                "exposure_sha256": sha256_bytes(exposure_payload),
                "eligible_pool_count": selection["reconstruction"][
                    "eligible_pool_count"
                ],
                "confirm_source_ids": selection["confirm_selection_proof"][
                    "selected_source_ids"
                ],
                "state_counts": {
                    role: len(record["states"])
                    for role, record in selection["roles"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
