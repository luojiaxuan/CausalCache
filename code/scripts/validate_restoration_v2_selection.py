"""Validate committed restoration-v2 selection and exposure artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from causalcache.data.restoration_v2_selection import (
    sha256_bytes,
    validate_exposure_ledger,
    validate_selection_manifest,
    validate_state_content_witnesses,
)
from causalcache.restoration_v2_contract import RestorationV2Contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-contract", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = RestorationV2Contract.load(args.v2_contract)
    selection_payload = args.selection.read_bytes()
    selection = json.loads(selection_payload)
    exposure = json.loads(args.exposure.read_bytes())
    repo_root = Path(__file__).resolve().parents[2]
    generator = selection.get("generator", {})
    expected_sources = {
        "module": "code/causalcache/data/restoration_v2_selection.py",
        "cli": "code/scripts/materialize_restoration_v2_selection.py",
        "validator": "code/scripts/validate_restoration_v2_selection.py",
    }
    git_revision = generator.get("git_revision")
    if not isinstance(git_revision, str) or re.fullmatch(
        r"[0-9a-f]{40}", git_revision
    ) is None:
        raise ValueError("selection generator Git revision is invalid")
    subprocess.run(
        ["git", "cat-file", "-e", f"{git_revision}^{{commit}}"],
        cwd=repo_root,
        check=True,
    )
    for prefix, relative_path in expected_sources.items():
        if generator.get(f"{prefix}_path") != relative_path:
            raise ValueError(f"selection generator {prefix} path mismatch")
        actual_sha = hashlib.sha256((repo_root / relative_path).read_bytes()).hexdigest()
        if generator.get(f"{prefix}_sha256") != actual_sha:
            raise ValueError(f"selection generator {prefix} SHA256 mismatch")
        committed_source = subprocess.check_output(
            ["git", "show", f"{git_revision}:{relative_path}"],
            cwd=repo_root,
        )
        if hashlib.sha256(committed_source).hexdigest() != actual_sha:
            raise ValueError(f"selection generator {prefix} differs from its Git revision")
    for input_name in ("v1_selection_config", "source_file_manifest"):
        record = selection["inputs"][input_name]
        payload = (repo_root / record["path"]).read_bytes()
        expected_sha = record.get("current_sha256", record.get("sha256"))
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError(f"selection input {input_name} SHA256 mismatch")
    source_manifest = json.loads(
        (repo_root / selection["inputs"]["source_file_manifest"]["path"]).read_bytes()
    )
    if selection["inputs"]["source_file_manifest"]["files"] != source_manifest["files"]:
        raise ValueError("selection embedded source-file inventory mismatch")
    validate_selection_manifest(selection, v2_contract=contract.data)
    validate_state_content_witnesses(selection)
    if exposure.get("selection_manifest_sha256") != sha256_bytes(selection_payload):
        raise ValueError("exposure ledger selection-manifest SHA256 mismatch")
    validate_exposure_ledger(exposure, selection_manifest=selection)
    print(
        json.dumps(
            {
                "outcome": "PASSED_PREOUTPUT_SELECTION_VALIDATION",
                "selection_sha256": sha256_bytes(selection_payload),
                "exposure_sha256": sha256_bytes(args.exposure.read_bytes()),
                "confirm_trajectories": len(
                    selection["roles"]["v2_confirm_primary"]["trajectories"]
                ),
                "screening_states": len(
                    selection["roles"]["v2_label_train"]["states"]
                )
                + len(selection["roles"]["v2_development"]["states"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
