"""Run or byte-validate the OCR/RGB v2 identity-scanner repair."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.restoration_v2_2_ocr_rgb import (
    build_state_score_records,
    summarize_state_scores,
)
from causalcache.restoration_v2_2_ocr_rgb_contract import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from causalcache.restoration_v2_2_ocr_rgb_contract_v2 import (
    CANONICAL_CONFIG_PATH,
    FAILURE_DIRECTORY,
    FAILURE_GIT_COMMIT,
    OCR_IDENTITY_OCCURRENCES_PER_LINE,
    PARENT_CONFIG_PATH,
    PARENT_CONFIG_SHA256,
    PARENT_SOURCE_GIT_COMMIT,
    PARENT_TO_REPAIR_SOURCE_CHANGED_PATHS,
    PROTOCOL_ID,
    TRAJECTORY_IDENTITY_OCCURRENCES_PER_LINE,
    V1_CANONICAL_RESULT_DIRECTORY,
    RestorationV22OcrRgbIdentityRepairContract,
)
from scripts.run_restoration_v2_2_ocr_rgb_baseline import (
    _formal_python_source_closure,
    _git,
    _input_identity,
    _jsonl_bytes,
    _pretty_json_bytes,
    _validate_existing_result,
    _validate_run_git,
    _write_new_result,
)


FORMAL_SOURCE_PATHS = (
    PARENT_CONFIG_PATH,
    CANONICAL_CONFIG_PATH,
    "code/causalcache/restoration_v2_2_ocr_rgb_contract.py",
    "code/causalcache/restoration_v2_2_ocr_rgb_contract_v2.py",
    "code/causalcache/restoration_v2_2_ocr_rgb.py",
    "code/scripts/validate_restoration_v2_2_ocr_rgb_contract.py",
    "code/scripts/validate_restoration_v2_2_ocr_rgb_contract_v2.py",
    "code/scripts/run_restoration_v2_2_ocr_rgb_baseline.py",
    "code/scripts/run_restoration_v2_2_ocr_rgb_baseline_v2.py",
    "code/causalcache/restoration_v2_baselines.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/restoration_v2_2_geometry_stats.py",
    "code/causalcache/restoration_v2_2_selector_geometry.py",
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/configs/restoration_v2_ocr_backend.json",
    "code/requirements/restoration_v2_ocr_lock.txt",
    f"{FAILURE_DIRECTORY}/README.md",
    f"{FAILURE_DIRECTORY}/summary.json",
)
RUN_STATUS = (
    "COMPLETED_RESTORATION_V2_2_OCR_RGB_BASELINE_V2_IDENTITY_REPAIR"
)
VALID_STATUS = "VALID_RESTORATION_V2_2_OCR_RGB_BASELINE_V2_IDENTITY_REPAIR"


def _validate_source_unchanged(root: Path, source_commit: str) -> None:
    if _git(root, "cat-file", "-t", source_commit) != "commit":
        raise ValueError("OCR/RGB v2 source_git_commit is not a local commit")
    closure = _formal_python_source_closure(root, source_commit)
    protected_paths = tuple(sorted(set(FORMAL_SOURCE_PATHS) | set(closure["paths"])))
    if _git(root, "diff", "--name-only", source_commit, "--", *protected_paths):
        raise ValueError("formal OCR/RGB v2 source changed after execution")


def _validate_repair_source_lineage(root: Path, source_commit: str) -> None:
    _git(root, "merge-base", "--is-ancestor", PARENT_SOURCE_GIT_COMMIT, source_commit)
    _git(root, "merge-base", "--is-ancestor", FAILURE_GIT_COMMIT, source_commit)
    changed = tuple(
        _git(
            root,
            "diff",
            "--name-only",
            PARENT_SOURCE_GIT_COMMIT,
            source_commit,
            "--",
        ).splitlines()
    )
    if changed != PARENT_TO_REPAIR_SOURCE_CHANGED_PATHS:
        raise ValueError(
            "parent-to-repair source diff escaped the frozen path allowlist"
        )


def _validate_invalid_v1_absent(root: Path) -> None:
    path = root / V1_CANONICAL_RESULT_DIRECTORY
    if path.exists() or path.with_name(path.name + ".tmp").exists():
        raise ValueError("invalid OCR/RGB v1 canonical output must remain absent")


def _repair_identity(
    contract: RestorationV22OcrRgbIdentityRepairContract,
) -> dict[str, Any]:
    failed = contract.data["failed_attempt"]
    return {
        "identity_scanner_only": True,
        "parent_protocol_id": contract.parent.data["protocol_id"],
        "parent_contract_sha256": contract.parent.sha256,
        "parent_source_git_commit": PARENT_SOURCE_GIT_COMMIT,
        "failed_attempt_git_commit": FAILURE_GIT_COMMIT,
        "failed_attempt_directory": FAILURE_DIRECTORY,
        "failed_attempt_files": failed["exact_files"],
        "v1_formal_state_score_row_count": 0,
        "v1_scientific_payload_generated": False,
        "v1_scientific_comparison_available": False,
        "parent_scientific_contract_changed": False,
        "immutable_inputs_changed": False,
        "feature_contract_changed": False,
        "selection_contract_changed": False,
        "statistics_contract_changed": False,
        "operation_ceiling_changed": False,
        "trajectory_identity_occurrences_per_line": (
            TRAJECTORY_IDENTITY_OCCURRENCES_PER_LINE
        ),
        "ocr_identity_occurrences_per_line": OCR_IDENTITY_OCCURRENCES_PER_LINE,
        "all_identity_occurrences_must_match": True,
        "invalid_v1_canonical_output_absent": True,
    }


def _readme(summary: Mapping[str, Any]) -> bytes:
    aggregate = summary["aggregate"]["by_role"]
    rows = []
    for role, label in (
        ("v2_label_train", "Train"),
        ("v2_development", "Development"),
        ("overall_stratified", "Overall"),
    ):
        row = aggregate[role]
        rows.append(
            f"| {label} | {row['state_count']} | "
            f"{row['mean_normalized_recovery']:.6f} | "
            f"{row['exact_coalition_match_rate']:.6f} | "
            f"{row['exact_cardinality_coalition_match_rate']:.6f} |"
        )
    text = f"""# Restoration v2.2 OCR/RGB baseline v2 identity repair

本结果是 v1 zero-score identity-scanner failure 的 versioned repair。唯一 execution-contract 变化是：derived
trajectory 每行要求两个相同 `source_id` occurrence，OCR 每行要求一个 `image_member_path`；数量漂移或同一行
值不一致均 fail closed。parent scientific contract、immutable inputs、feature、selection、statistics 与 operation
ceiling 不变。v1 没有产生 scientific payload，因此这里不报告 v1/v2 scientific-value comparison。

本结果只覆盖 frozen primary `n=4,B=2` 的 15 条 train/development states。它使用 archived
`full_spatial_tokens` 和 deterministic 256×256 RGB histogram；没有 OCR inference、模型加载、GPU 或 policy
forward。confirm/test semantic access 与 feature score 均为 0。

| Split | States | Mean recovery | At-most-B exact match | Exact-B match |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

- source commit: `{summary['source_execution']['source_git_commit']}`
- repair contract SHA256: `{summary['source_execution']['contract_sha256']}`
- parent contract SHA256: `{summary['source_execution']['parent_contract_sha256']}`
- failed-attempt commit: `{summary['repair_identity']['failed_attempt_git_commit']}`
- scientific payload SHA256: `{summary['scientific_payload_sha256']}`
- state score rows: `{summary['state_scores']['record_count']}` (`{summary['state_scores']['sha256']}`)
"""
    return text.encode("utf-8")


def _expected_files(
    *,
    contract: RestorationV22OcrRgbIdentityRepairContract,
    labels_archive: Path,
    derived_root: Path,
    source_commit: str,
) -> dict[str, bytes]:
    records = build_state_score_records(
        contract=contract.parent,
        labels_archive=labels_archive,
        derived_root=derived_root,
        trajectory_identity_occurrences_per_line=(
            TRAJECTORY_IDENTITY_OCCURRENCES_PER_LINE
        ),
        ocr_identity_occurrences_per_line=OCR_IDENTITY_OCCURRENCES_PER_LINE,
        output_protocol_id=PROTOCOL_ID,
    )
    if any(record.get("protocol_id") != PROTOCOL_ID for record in records):
        raise RuntimeError("OCR/RGB v2 state-row protocol identity drifted")
    aggregate = summarize_state_scores(records)
    record_bytes = _jsonl_bytes(records)
    inputs = _input_identity(contract.parent)
    repair = _repair_identity(contract)
    source_closure = _formal_python_source_closure(
        contract.repository_root,
        source_commit,
    )
    full_payload = {
        "protocol_id": PROTOCOL_ID,
        "source_git_commit": source_commit,
        "contract_sha256": contract.sha256,
        "parent_contract_sha256": PARENT_CONFIG_SHA256,
        "repair_identity": repair,
        "input_identity": inputs,
        "aggregate": aggregate,
        "state_scores": list(records),
    }
    summary = {
        "schema_version": "1.1.0",
        "status": RUN_STATUS,
        "protocol_id": PROTOCOL_ID,
        "source_execution": {
            "source_git_commit": source_commit,
            "contract_sha256": contract.sha256,
            "parent_contract_sha256": contract.parent.sha256,
            "formal_source_paths": list(FORMAL_SOURCE_PATHS),
            "formal_python_source_closure": {
                key: source_closure[key]
                for key in ("rule", "path_count", "inventory_sha256")
            },
        },
        "repair_identity": repair,
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
    contract = RestorationV22OcrRgbIdentityRepairContract.load(
        args.contract.resolve(),
        repository_root=root,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            root / contract.data["output_contract"]["canonical_result_directory"]
        ).resolve()
    )
    canonical_output = (
        root / contract.data["output_contract"]["canonical_result_directory"]
    ).resolve()
    if output_dir != canonical_output or root not in output_dir.parents:
        raise ValueError("OCR/RGB v2 output must use the canonical Git path")
    labels_archive = args.labels_archive.resolve()
    labels = contract.parent.data["immutable_inputs"]["restoration_labels"]
    if (
        not labels_archive.is_file()
        or labels_archive.is_symlink()
        or labels_archive.stat().st_size != labels["raw_archive_size_bytes"]
        or sha256_file(labels_archive) != labels["raw_archive_sha256"]
    ):
        raise ValueError("OCR/RGB v2 labels archive identity drifted")
    _validate_invalid_v1_absent(root)
    if args.mode == "run":
        _validate_run_git(root, args.source_git_commit)
        if output_dir.exists() or output_dir.with_name(
            output_dir.name + ".tmp"
        ).exists():
            raise FileExistsError("canonical OCR/RGB v2 output already exists")
    else:
        _validate_source_unchanged(root, args.source_git_commit)
    _validate_repair_source_lineage(root, args.source_git_commit)
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
    _validate_invalid_v1_absent(root)
    print(
        json.dumps(
            {
                "status": status,
                "source_git_commit": args.source_git_commit,
                "contract_sha256": contract.sha256,
                "parent_contract_sha256": contract.parent.sha256,
                "failed_attempt_git_commit": FAILURE_GIT_COMMIT,
                "output_dir": str(output_dir),
                "formal_state_count": 15,
                "candidate_comparison_count": 60,
                "unique_image_count": 75,
                "trajectory_identity_occurrences_per_line": (
                    TRAJECTORY_IDENTITY_OCCURRENCES_PER_LINE
                ),
                "ocr_identity_occurrences_per_line": (
                    OCR_IDENTITY_OCCURRENCES_PER_LINE
                ),
                "gpu_count": 0,
                "ocr_inference_count": 0,
                "policy_forward_count": 0,
                "confirm_state_access_count": 0,
                "invalid_v1_canonical_output_absent": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
