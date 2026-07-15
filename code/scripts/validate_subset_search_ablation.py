"""Validate a committed subset-search ablation result by deterministic replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from causalcache.subset_search_ablation import build_scientific_payload, sha256_file
from scripts.run_subset_search_ablation import SOURCE_FILES


GIT_SHA = re.compile(r"[0-9a-f]{40}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def _git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout.strip() if text else result.stdout


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_clean_pushed_descendant(root: Path, source_commit: str) -> dict[str, str]:
    if GIT_SHA.fullmatch(source_commit) is None:
        raise ValueError("recorded source commit is invalid")
    if Path(str(_git(root, "rev-parse", "--show-toplevel"))).resolve() != root:
        raise ValueError("repository root differs from Git top level")
    head = str(_git(root, "rev-parse", "HEAD"))
    origin_main = str(_git(root, "rev-parse", "refs/remotes/origin/main"))
    branch = str(_git(root, "branch", "--show-current"))
    status = str(_git(root, "status", "--porcelain=v1", "--untracked-files=all"))
    advertised = str(_git(root, "ls-remote", "--heads", "origin", "refs/heads/main"))
    ancestor = subprocess.run(
        ("git", "-C", str(root), "merge-base", "--is-ancestor", source_commit, head),
        check=False,
    ).returncode
    if head != origin_main or advertised != f"{head}\trefs/heads/main" or branch != "main" or status:
        raise ValueError("validation requires clean pushed main")
    if ancestor != 0:
        raise ValueError("current main is not a descendant of the recorded source commit")
    return {"head": head, "origin_main": origin_main, "source_commit": source_commit}


def main() -> None:
    args = parse_args()
    if not args.repository_root.is_absolute() or not args.config.is_absolute() or not args.summary.is_absolute():
        raise ValueError("all paths must be absolute")
    root = args.repository_root.resolve()
    config = args.config.resolve()
    summary_path = args.summary.resolve()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("schema_version") != "1.0.0" or summary.get("experiment_id") != "subset-search-ablation-v1":
        raise ValueError("result schema or experiment id drifted")
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("git"), dict):
        raise ValueError("result provenance is incomplete")
    source_commit = str(provenance["git"].get("commit"))
    expected_runtime_declarations = {
        "device": "cpu",
        "gpu_operations_executed": 0,
        "policy_operations_executed": 0,
        "existing_git_inputs_read": True,
        "new_external_or_untouched_data_accessed": False,
        "seed": "not_applicable_deterministic",
    }
    for key, expected in expected_runtime_declarations.items():
        if provenance.get(key) != expected:
            raise ValueError(f"runtime declaration drifted: {key}")
    git_status = _require_clean_pushed_descendant(root, source_commit)
    inventory = provenance.get("source_files")
    if not isinstance(inventory, list) or [row.get("path") for row in inventory] != list(SOURCE_FILES):
        raise ValueError("source inventory paths drifted")
    for row in inventory:
        relative = str(row["path"])
        expected_hash = str(row["sha256"])
        if sha256_file(root / relative) != expected_hash:
            raise ValueError(f"current source hash drifted: {relative}")
        blob = _git(root, "show", f"{source_commit}:{relative}", text=False)
        if hashlib.sha256(blob).hexdigest() != expected_hash:
            raise ValueError(f"recorded source commit hash drifted: {relative}")
    recomputed = build_scientific_payload(root, config)
    if recomputed != summary.get("scientific_payload"):
        raise ValueError("recomputed scientific payload differs from the committed result")
    payload_hash = _canonical_sha(recomputed)
    if payload_hash != summary.get("scientific_payload_sha256"):
        raise ValueError("scientific payload SHA256 mismatch")
    relative_summary = summary_path.relative_to(root).as_posix()
    current_bytes = summary_path.read_bytes()
    committed_bytes = _git(root, "show", f"HEAD:{relative_summary}", text=False)
    if current_bytes != committed_bytes:
        raise ValueError("summary is not identical to the committed HEAD blob")
    print(
        json.dumps(
            {
                "status": "VALID_SUBSET_SEARCH_ABLATION_V1",
                "scientific_payload_sha256": payload_hash,
                "source_commit": source_commit,
                "validated_commit": git_status["head"],
                "v2_1_outcome_unchanged": recomputed["v2_1_outcome_unchanged"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
