"""Run or validate the deterministic restoration-v2.2 selector geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.restoration_v2_2_selector_geometry import (
    load_selector_geometry_states,
)
from causalcache.restoration_v2_2_selector_geometry_contract import (
    CANONICAL_CONFIG_PATH,
    PROTOCOL_ID,
    RestorationV22SelectorGeometryContract,
    sha256_file,
)
from causalcache.restoration_v2_2_selector_geometry_result import (
    build_selector_geometry_scientific_payload,
)


RESULT_FILES = ("README.md", "state_budget_records.jsonl", "summary.json")
FORMAL_SOURCE_PATHS = (
    CANONICAL_CONFIG_PATH,
    "code/causalcache/restoration_v2_2_geometry_stats.py",
    "code/causalcache/restoration_v2_2_selector_geometry.py",
    "code/causalcache/restoration_v2_2_selector_geometry_contract.py",
    "code/causalcache/restoration_v2_2_selector_geometry_result.py",
    "code/scripts/run_restoration_v2_2_selector_geometry.py",
)
RUN_STATUS = "COMPLETED_RESTORATION_V2_2_SELECTOR_GEOMETRY_V1"
VALID_STATUS = "VALID_RESTORATION_V2_2_SELECTOR_GEOMETRY_V1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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
    return b"".join(_canonical_json_bytes(dict(record)) + b"\n" for record in records)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _validate_run_git(root: Path, source_commit: str) -> None:
    head = _git(root, "rev-parse", "HEAD")
    if head != source_commit:
        raise ValueError("run source commit must equal current HEAD")
    if _git(root, "symbolic-ref", "--short", "HEAD") != "main":
        raise ValueError("formal selector geometry must run from main")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("formal selector geometry requires a clean worktree")
    if _git(root, "rev-parse", "origin/main") != source_commit:
        raise ValueError("formal selector geometry source must already be pushed")


def _validate_source_unchanged(root: Path, source_commit: str) -> None:
    if _git(root, "cat-file", "-t", source_commit) != "commit":
        raise ValueError("source_git_commit is not a local commit")
    changed = _git(
        root,
        "diff",
        "--name-only",
        source_commit,
        "--",
        *FORMAL_SOURCE_PATHS,
    )
    if changed:
        raise ValueError("formal selector-geometry source changed after execution")


def _validate_label_archive(path: Path, contract: Mapping[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError("label archive must be a non-symlink regular file")
    labels = contract["immutable_inputs"]["restoration_labels"]
    if path.stat().st_size != labels["raw_archive_size_bytes"]:
        raise ValueError("label archive size differs from the immutable contract")
    if sha256_file(path) != labels["raw_archive_sha256"]:
        raise ValueError("label archive SHA256 differs from the immutable contract")


def _result_readme(summary: Mapping[str, Any]) -> bytes:
    science = summary["science"]
    primary = science["slices"]["primary_n4_b2"]
    method_rows = []
    labels = (
        ("exact_subset", "Exact subset"),
        ("true_conditional_greedy", "True conditional greedy"),
        ("budget_conditioned_independent", "Budget-conditioned independent"),
        ("full_shapley_independent", "Full-path Shapley independent"),
        ("dynamic_recent", "Recent"),
        ("analytic_exact_cardinality_random", "Random exact expectation"),
    )
    for method, label in labels:
        values = primary["methods"][method]
        method_rows.append(
            "| "
            + label
            + " | "
            + f"{values['v2_label_train']['mean_normalized_recovery']:.6f}"
            + " | "
            + f"{values['v2_development']['mean_normalized_recovery']:.6f}"
            + " | "
            + f"{values['overall_stratified']['mean_normalized_recovery']:.6f}"
            + " |"
        )
    projection = primary["paired_bootstrap"]["objective_projection_gap"][
        "development"
    ]
    search = primary["paired_bootstrap"]["search_gap"]["development"]
    shaping = science["internal_method_shaping"]
    curve_rows = []
    for event_count in (2, 3, 4):
        for budget in range(event_count + 1):
            values = science["budget_curve"][f"n{event_count}_b{budget}"][
                "methods"
            ]
            curve_rows.append(
                f"| {event_count} | {budget} | "
                f"{values['exact_subset']['overall_stratified']['mean_normalized_recovery']:.6f} | "
                f"{values['true_conditional_greedy']['overall_stratified']['mean_normalized_recovery']:.6f} | "
                f"{values['budget_conditioned_independent']['overall_stratified']['mean_normalized_recovery']:.6f} | "
                f"{values['dynamic_recent']['overall_stratified']['mean_normalized_recovery']:.6f} |"
            )
    text = f"""# Restoration v2.2 selector geometry

## 结论边界

本结果是 45 个 label-train/development states 上的 policy-free $D(S)$ table reduction；新 policy forward、
generation、teacher KL、gate training、matched-NLL、closed-loop、confirm/test access 均为 0。primary slice 是
`n=4,B=2`；`n=2,B=2` 只作恢复全部 candidate 的 ceiling sanity。development 只有 5 条 trajectories，
bootstrap 只作 method-shaping screen，不表述为统计显著性。

## Primary `n=4,B=2`

| Selector | Train recovery | Development recovery | Overall recovery |
| --- | ---: | ---: | ---: |
{chr(10).join(method_rows)}

- development search gap（exact - true greedy）：`{search['mean_difference']:.6f}`，90% interval
  `[{search['mean_difference_interval']['lower']:.6f}, {search['mean_difference_interval']['upper']:.6f}]`；
- development objective-projection gap（true greedy - budget-conditioned independent）：
  `{projection['mean_difference']:.6f}`，90% interval
  `[{projection['mean_difference_interval']['lower']:.6f}, {projection['mean_difference_interval']['upper']:.6f}]`，
  trajectory win/tie/loss = `{projection['win_tie_loss']['wins']}/{projection['win_tie_loss']['ties']}/{projection['win_tie_loss']['losses']}`；
- conditioning decision：`{shaping['conditioning_decision']}`；
- search decision：`{shaping['search_decision']}`。

上述 decision 只决定下一版 student/search contract，不是 paper success gate，也不打开 confirm。

## Budget curve（overall trajectory-weighted）

| n | B | Exact | True greedy | Independent | Recent |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(curve_rows)}

## Artifact identity

- source commit：`{summary['source_execution']['source_git_commit']}`；
- contract SHA256：`{summary['source_execution']['contract_sha256']}`；
- immutable label revision：`{summary['immutable_label_input']['hf_revision']}`；
- raw label SHA256：`{summary['immutable_label_input']['raw_archive_sha256']}`；
- complete scientific payload SHA256：`{summary['scientific_payload_sha256']}`；
- state-budget rows：`{summary['state_budget_records']['record_count']}`，SHA256
  `{summary['state_budget_records']['sha256']}`。

OCR/RGB 仍是 `pending_feature_stage`；policy-vision 仍是
`pending_separate_feature_only_source_freeze`。两者补齐前不冻结 gate training contract。
"""
    return text.encode("utf-8")


def _expected_files(
    *,
    contract: RestorationV22SelectorGeometryContract,
    label_archive: Path,
    source_commit: str,
) -> dict[str, bytes]:
    states = load_selector_geometry_states(label_archive)
    science = dict(build_selector_geometry_scientific_payload(states))
    records = science.pop("state_budget_records")
    record_bytes = _jsonl_bytes(records)
    full_payload = {
        "protocol_id": PROTOCOL_ID,
        "source_git_commit": source_commit,
        "contract_sha256": contract.sha256,
        "immutable_label_archive_sha256": sha256_file(label_archive),
        "science": science,
        "state_budget_records": records,
    }
    labels = contract.data["immutable_inputs"]["restoration_labels"]
    summary = {
        "schema_version": "1.0.0",
        "status": RUN_STATUS,
        "protocol_id": PROTOCOL_ID,
        "source_execution": {
            "source_git_commit": source_commit,
            "contract_sha256": contract.sha256,
            "formal_source_paths": list(FORMAL_SOURCE_PATHS),
        },
        "immutable_label_input": {
            "hf_repo": labels["hf_repo"],
            "hf_revision": labels["hf_revision"],
            "hf_tag": labels["hf_tag"],
            "hf_path": labels["hf_path"],
            "raw_archive_sha256": labels["raw_archive_sha256"],
            "raw_archive_size_bytes": labels["raw_archive_size_bytes"],
            "raw_tree_inventory_sha256": labels["raw_tree_inventory_sha256"],
            "validated_by": "read_label_evidence_archive",
        },
        "scientific_payload_sha256": _sha256_bytes(
            _canonical_json_bytes(full_payload)
        ),
        "state_budget_records": {
            "path": "state_budget_records.jsonl",
            "record_count": len(records),
            "sha256": _sha256_bytes(record_bytes),
            "size_bytes": len(record_bytes),
        },
        "science": science,
    }
    return {
        "README.md": _result_readme(summary),
        "state_budget_records.jsonl": record_bytes,
        "summary.json": _pretty_json_bytes(summary),
    }


def _write_new_result(output_dir: Path, files: Mapping[str, bytes]) -> None:
    if output_dir.exists():
        raise FileExistsError("canonical selector-geometry output already exists")
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        raise FileExistsError("selector-geometry staging directory already exists")
    staging.mkdir(parents=False)
    try:
        for name in RESULT_FILES:
            (staging / name).write_bytes(files[name])
        if sorted(path.name for path in staging.iterdir()) != sorted(RESULT_FILES):
            raise RuntimeError("selector-geometry staging inventory drifted")
        for name in RESULT_FILES:
            if (staging / name).read_bytes() != files[name]:
                raise RuntimeError("selector-geometry staging bytes drifted")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_existing_result(output_dir: Path, files: Mapping[str, bytes]) -> None:
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("selector-geometry result directory is missing or symlinked")
    observed = sorted(path.name for path in output_dir.iterdir())
    if observed != sorted(RESULT_FILES):
        raise ValueError("selector-geometry result file inventory drifted")
    for name in RESULT_FILES:
        if (output_dir / name).read_bytes() != files[name]:
            raise ValueError(f"selector-geometry result bytes drifted: {name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "validate"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--labels-archive", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    contract = RestorationV22SelectorGeometryContract.load(
        args.contract.resolve(),
        repository_root=root,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (root / contract.data["output_contract"]["canonical_result_directory"]).resolve()
    )
    canonical_output = (
        root / contract.data["output_contract"]["canonical_result_directory"]
    ).resolve()
    if output_dir != canonical_output or root not in output_dir.parents:
        raise ValueError("selector-geometry output must use the canonical Git path")
    label_archive = args.labels_archive.resolve()
    _validate_label_archive(label_archive, contract.data)
    if args.mode == "run":
        _validate_run_git(root, args.source_git_commit)
    else:
        _validate_source_unchanged(root, args.source_git_commit)
    files = _expected_files(
        contract=contract,
        label_archive=label_archive,
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
                "label_archive_sha256": sha256_file(label_archive),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
