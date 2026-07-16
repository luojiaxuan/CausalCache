"""Run or byte-validate the frozen restoration-v2.2 OCR/RGB baseline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.restoration_v2_2_ocr_rgb import (
    STATUS,
    build_state_score_records,
    summarize_state_scores,
)
from causalcache.restoration_v2_2_ocr_rgb_contract import (
    CANONICAL_CONFIG_PATH,
    EXPECTED_OUTPUT_FILES,
    PROTOCOL_ID,
    RestorationV22OcrRgbContract,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


FORMAL_SOURCE_PATHS = (
    CANONICAL_CONFIG_PATH,
    "code/causalcache/restoration_v2_2_ocr_rgb_contract.py",
    "code/causalcache/restoration_v2_2_ocr_rgb.py",
    "code/scripts/validate_restoration_v2_2_ocr_rgb_contract.py",
    "code/scripts/run_restoration_v2_2_ocr_rgb_baseline.py",
    "code/causalcache/restoration_v2_baselines.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/restoration_v2_2_geometry_stats.py",
    "code/causalcache/restoration_v2_2_selector_geometry.py",
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/configs/restoration_v2_ocr_backend.json",
    "code/requirements/restoration_v2_ocr_lock.txt",
)
FORMAL_PYTHON_SOURCE_ROOTS = ("code/causalcache", "code/scripts")
RUN_STATUS = STATUS
VALID_STATUS = "VALID_RESTORATION_V2_2_OCR_RGB_BASELINE_V1"


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(record)) + b"\n" for record in records)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _validate_run_git(root: Path, source_commit: str) -> None:
    if _git(root, "rev-parse", "HEAD") != source_commit:
        raise ValueError("OCR/RGB source commit must equal HEAD")
    if _git(root, "symbolic-ref", "--short", "HEAD") != "main":
        raise ValueError("formal OCR/RGB run must use main")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("formal OCR/RGB run requires a clean worktree")
    if _git(root, "rev-parse", "origin/main") != source_commit:
        raise ValueError("formal OCR/RGB source must already be pushed")


def _formal_python_source_closure(
    root: Path,
    source_commit: str,
) -> dict[str, Any]:
    paths = tuple(
        path
        for path in _git(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            source_commit,
            "--",
            *FORMAL_PYTHON_SOURCE_ROOTS,
        ).splitlines()
        if path.endswith(".py")
    )
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("formal Python source closure is empty or duplicated")
    records = [
        {
            "path": path,
            "git_blob_sha": _git(root, "rev-parse", f"{source_commit}:{path}"),
        }
        for path in paths
    ]
    return {
        "rule": "all_tracked_python_under_code_causalcache_and_code_scripts_at_source_commit",
        "path_count": len(paths),
        "inventory_sha256": sha256_bytes(canonical_json_bytes(records)),
        "paths": paths,
    }


def _validate_source_unchanged(root: Path, source_commit: str) -> None:
    if _git(root, "cat-file", "-t", source_commit) != "commit":
        raise ValueError("OCR/RGB source_git_commit is not a local commit")
    closure = _formal_python_source_closure(root, source_commit)
    protected_paths = tuple(
        sorted(set(FORMAL_SOURCE_PATHS) | set(closure["paths"]))
    )
    if _git(root, "diff", "--name-only", source_commit, "--", *protected_paths):
        raise ValueError("formal OCR/RGB source changed after execution")


def _input_identity(contract: RestorationV22OcrRgbContract) -> dict[str, Any]:
    inputs = contract.data["immutable_inputs"]
    geometry = inputs["selector_geometry_result"]
    labels = inputs["restoration_labels"]
    derived = inputs["derived_dataset"]
    baseline = inputs["baseline_implementation"]
    return {
        "selector_geometry": {
            "git_commit": geometry["git_commit"],
            "execution_source_git_commit": geometry["execution_source_git_commit"],
            "artifact_git_commit": geometry["artifact_git_commit"],
            "validation_git_commit": geometry["validation_git_commit"],
            "repair_contract_sha256": geometry["repair_contract_sha256"],
            "parent_contract_sha256": geometry["parent_contract_sha256"],
            "directory": geometry["directory"],
            "scientific_payload_sha256": geometry["scientific_payload_sha256"],
            "files": geometry["files"],
        },
        "restoration_labels": {
            key: labels[key]
            for key in (
                "hf_repo",
                "hf_revision",
                "hf_tag",
                "hf_path",
                "raw_archive_sha256",
                "raw_archive_size_bytes",
                "raw_tree_inventory_sha256",
            )
        },
        "derived_dataset": {
            key: derived[key]
            for key in (
                "hf_repo",
                "hf_revision",
                "hf_tag",
                "artifact_tree_sha256",
                "exact_files",
            )
        },
        "baseline_implementation": {
            "manifest_sha256": baseline["manifest_sha256"],
            "source_files": baseline["source_files"],
            "source_closure": baseline["source_closure"],
            "runtime": baseline["runtime"],
        },
    }


def _readme(summary: Mapping[str, Any]) -> bytes:
    aggregate = summary["aggregate"]["by_role"]
    lines = []
    for role, label in (
        ("v2_label_train", "Train"),
        ("v2_development", "Development"),
        ("overall_stratified", "Overall"),
    ):
        row = aggregate[role]
        lines.append(
            f"| {label} | {row['state_count']} | "
            f"{row['mean_normalized_recovery']:.6f} | "
            f"{row['exact_coalition_match_rate']:.6f} | "
            f"{row['exact_cardinality_coalition_match_rate']:.6f} |"
        )
    text = f"""# Restoration v2.2 OCR/RGB baseline v1

本结果只覆盖 frozen primary `n=4,B=2` 的 15 条 train/development states。它使用 archived
`full_spatial_tokens` 和 deterministic 256×256 RGB histogram；没有 OCR inference、模型加载、GPU 或 policy
forward。完整 derived files 只做 byte hash 与非选中 identity 的 opaque scan；confirm prompt/image/OCR 内容不做
semantic parse、不进入 feature scorer，confirm/test feature score 为 0。

| Split | States | Mean recovery | At-most-B exact match | Exact-B match |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(lines)}

- source commit: `{summary['source_execution']['source_git_commit']}`
- contract SHA256: `{summary['source_execution']['contract_sha256']}`
- scientific payload SHA256: `{summary['scientific_payload_sha256']}`
- state score rows: `{summary['state_scores']['record_count']}` (`{summary['state_scores']['sha256']}`)
"""
    return text.encode("utf-8")


def _expected_files(
    *,
    contract: RestorationV22OcrRgbContract,
    labels_archive: Path,
    derived_root: Path,
    source_commit: str,
) -> dict[str, bytes]:
    records = build_state_score_records(
        contract=contract,
        labels_archive=labels_archive,
        derived_root=derived_root,
    )
    aggregate = summarize_state_scores(records)
    record_bytes = _jsonl_bytes(records)
    inputs = _input_identity(contract)
    source_closure = _formal_python_source_closure(
        contract.repository_root,
        source_commit,
    )
    full_payload = {
        "protocol_id": PROTOCOL_ID,
        "source_git_commit": source_commit,
        "contract_sha256": contract.sha256,
        "input_identity": inputs,
        "aggregate": aggregate,
        "state_scores": list(records),
    }
    summary = {
        "schema_version": "1.0.0",
        "status": RUN_STATUS,
        "protocol_id": PROTOCOL_ID,
        "source_execution": {
            "source_git_commit": source_commit,
            "contract_sha256": contract.sha256,
            "formal_source_paths": list(FORMAL_SOURCE_PATHS),
            "formal_python_source_closure": {
                key: source_closure[key]
                for key in ("rule", "path_count", "inventory_sha256")
            },
        },
        "input_identity": inputs,
        "aggregate": aggregate,
        "scientific_payload_sha256": sha256_bytes(
            canonical_json_bytes(full_payload)
        ),
        "state_scores": {
            "path": "state_scores.jsonl",
            "record_count": len(records),
            "size_bytes": len(record_bytes),
            "sha256": sha256_bytes(record_bytes),
        },
    }
    return {
        "README.md": _readme(summary),
        "state_scores.jsonl": record_bytes,
        "summary.json": _pretty_json_bytes(summary),
    }


def _write_new_result(output_dir: Path, files: Mapping[str, bytes]) -> None:
    if output_dir.exists():
        raise FileExistsError("canonical OCR/RGB output already exists")
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        raise FileExistsError("OCR/RGB staging directory already exists")
    staging.mkdir(parents=False)
    try:
        for name in EXPECTED_OUTPUT_FILES:
            (staging / name).write_bytes(files[name])
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_existing_result(output_dir: Path, files: Mapping[str, bytes]) -> None:
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("canonical OCR/RGB output is missing or symlinked")
    if sorted(path.name for path in output_dir.iterdir()) != sorted(
        EXPECTED_OUTPUT_FILES
    ):
        raise ValueError("OCR/RGB output exact-three inventory drifted")
    for name in EXPECTED_OUTPUT_FILES:
        if (output_dir / name).read_bytes() != files[name]:
            raise ValueError(f"OCR/RGB output bytes drifted: {name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "validate"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--labels-archive", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    contract = RestorationV22OcrRgbContract.load(
        args.contract.resolve(),
        repository_root=root,
        validate_bound_sources=True,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (root / contract.data["output_contract"]["canonical_result_directory"])
        .resolve()
    )
    canonical_output = (
        root / contract.data["output_contract"]["canonical_result_directory"]
    ).resolve()
    if output_dir != canonical_output or root not in output_dir.parents:
        raise ValueError("OCR/RGB output must use the canonical Git path")
    labels_archive = args.labels_archive.resolve()
    labels = contract.data["immutable_inputs"]["restoration_labels"]
    if (
        not labels_archive.is_file()
        or labels_archive.is_symlink()
        or labels_archive.stat().st_size != labels["raw_archive_size_bytes"]
        or sha256_file(labels_archive) != labels["raw_archive_sha256"]
    ):
        raise ValueError("OCR/RGB labels archive identity drifted")
    if args.mode == "run":
        _validate_run_git(root, args.source_git_commit)
        if output_dir.exists() or output_dir.with_name(
            output_dir.name + ".tmp"
        ).exists():
            raise FileExistsError("canonical OCR/RGB output or staging already exists")
    else:
        _validate_source_unchanged(root, args.source_git_commit)
    files = _expected_files(
        contract=contract,
        labels_archive=labels_archive,
        derived_root=args.derived_root.resolve(),
        source_commit=args.source_git_commit,
    )
    if args.mode == "run":
        _write_new_result(output_dir, files)
        status = RUN_STATUS
    else:
        _validate_existing_result(output_dir, files)
        status = VALID_STATUS
    print(
        json.dumps(
            {
                "status": status,
                "source_git_commit": args.source_git_commit,
                "contract_sha256": contract.sha256,
                "output_dir": str(output_dir),
                "formal_state_count": 15,
                "candidate_comparison_count": 60,
                "unique_image_count": 75,
                "gpu_count": 0,
                "ocr_inference_count": 0,
                "policy_forward_count": 0,
                "confirm_state_access_count": 0,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
