"""Run or validate the versioned restoration-v2.2 geometry reporting repair."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from causalcache.restoration_v2_2_selector_geometry import (
    load_selector_geometry_states,
)
from causalcache.restoration_v2_2_selector_geometry_contract_v2 import (
    CANONICAL_CONFIG_PATH,
    PROTOCOL_ID,
    RestorationV22SelectorGeometryRepairContract,
    sha256_file,
    validate_repair_preserves_legacy_payload,
    validate_repaired_scientific_payload,
)
from causalcache.restoration_v2_2_selector_geometry_result import (
    build_selector_geometry_scientific_payload as build_legacy_payload,
)
from causalcache.restoration_v2_2_selector_geometry_result_v2 import (
    build_selector_geometry_scientific_payload,
)
from scripts.run_restoration_v2_2_selector_geometry import (
    _canonical_json_bytes,
    _jsonl_bytes,
    _pretty_json_bytes,
    _sha256_bytes,
    _validate_existing_result,
    _validate_label_archive,
    _write_new_result,
)


FORMAL_SOURCE_PATHS = (
    "code/configs/causalcache_restoration_v2_2_selector_geometry.json",
    CANONICAL_CONFIG_PATH,
    "code/causalcache/restoration_v2_2_geometry_stats.py",
    "code/causalcache/restoration_v2_2_selector_geometry.py",
    "code/causalcache/restoration_v2_2_selector_geometry_contract.py",
    "code/causalcache/restoration_v2_2_selector_geometry_result.py",
    "code/scripts/run_restoration_v2_2_selector_geometry.py",
    "code/causalcache/restoration_v2_2_selector_geometry_contract_v2.py",
    "code/causalcache/restoration_v2_2_selector_geometry_result_v2.py",
    "code/scripts/validate_restoration_v2_2_selector_geometry_v2_contract.py",
    "code/scripts/run_restoration_v2_2_selector_geometry_v2.py",
)
RUN_STATUS = "COMPLETED_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR"
VALID_STATUS = "VALID_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR"


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
    if _git(root, "rev-parse", "HEAD") != source_commit:
        raise ValueError("repair source commit must equal current HEAD")
    if _git(root, "symbolic-ref", "--short", "HEAD") != "main":
        raise ValueError("formal repair must run from main")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("formal repair requires a clean worktree")
    if _git(root, "rev-parse", "origin/main") != source_commit:
        raise ValueError("formal repair source must already be pushed")


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
        raise ValueError("formal selector-geometry repair source changed after execution")


def _result_tree_fingerprint(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    if not path.is_dir() or path.is_symlink():
        raise ValueError("superseded result path must be a non-symlink directory")
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink() or not item.is_file():
            raise ValueError("superseded result tree may contain only regular files")
        result[item.relative_to(path).as_posix()] = sha256_file(item)
    return result


def _result_readme(summary: Mapping[str, Any]) -> bytes:
    science = summary["science"]
    primary = science["slices"]["primary_n4_b2"]
    rows = []
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
        rows.append(
            f"| {label} | "
            f"{values['v2_label_train']['mean_normalized_recovery']:.6f} | "
            f"{values['v2_development']['mean_normalized_recovery']:.6f} | "
            f"{values['overall_stratified']['mean_normalized_recovery']:.6f} |"
        )
    random = primary["methods"]["analytic_exact_cardinality_random"][
        "overall_stratified"
    ]
    reports = science["interaction_selector_reports"]
    text = f"""# Restoration v2.2 selector geometry v2 repair

## 结论边界

这是 v1 table-only selector geometry 的 reporting-only versioned repair。它没有改变 selector algorithm、
utility/recovery、bootstrap 或 method-shaping 值，也没有新增 policy forward、KL、gate training、matched-NLL、
closed-loop、confirm/test access。v1 目录未被覆盖。

## Primary `n=4,B=2`

| Selector | Train recovery | Development recovery | Overall recovery |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

## 修复内容

- interaction report materialize 完整 `role × n × B × strength × negative-marginal` joint Cartesian：
  `{reports['observed_cell_count']}` cells，其中 nonempty `{reports['nonempty_cell_count']}`、empty
  `{reports['empty_cell_count']}`；每个 cell 均含 10 个 selectors，空 cell 使用统一 null schema；
- analytic exact-cardinality random 显式报告 `k=B`、expected exact-match 与 expected Jaccard。primary overall
  cardinality/match/Jaccard 分别为 `{random['mean_selected_cardinality']:.6f}`、
  `{random['exact_coalition_match_rate']:.6f}`、`{random['mean_jaccard_to_exact_coalition']:.6f}`；
- train-only tertile 在每个 `n` 内独立计算，并原样应用到 matching-`n` development states；独立 validator
  从 state records 重算 cutpoint、assignment、random expectation 与完整 cell coverage。

## Artifact identity

- source commit：`{summary['source_execution']['source_git_commit']}`；
- v2 repair contract SHA256：`{summary['source_execution']['contract_sha256']}`；
- parent v1 contract SHA256：`{summary['source_execution']['parent_contract_sha256']}`；
- immutable label revision：`{summary['immutable_label_input']['hf_revision']}`；
- complete scientific payload SHA256：`{summary['scientific_payload_sha256']}`；
- state-budget rows：`{summary['state_budget_records']['record_count']}`，SHA256
  `{summary['state_budget_records']['sha256']}`。
"""
    return text.encode("utf-8")


def _expected_files(
    *,
    contract: RestorationV22SelectorGeometryRepairContract,
    label_archive: Path,
    source_commit: str,
    superseded_fingerprint: Mapping[str, str] | None,
) -> dict[str, bytes]:
    states = load_selector_geometry_states(label_archive)
    science = dict(build_selector_geometry_scientific_payload(states))
    legacy_science = build_legacy_payload(states)
    validate_repaired_scientific_payload(science)
    validate_repair_preserves_legacy_payload(science, legacy_science)
    records = science.pop("state_budget_records")
    record_bytes = _jsonl_bytes(records)
    full_payload = {
        "protocol_id": PROTOCOL_ID,
        "source_git_commit": source_commit,
        "contract_sha256": contract.sha256,
        "parent_contract_sha256": contract.parent.sha256,
        "immutable_label_archive_sha256": sha256_file(label_archive),
        "science": science,
        "state_budget_records": records,
    }
    labels = contract.parent.data["immutable_inputs"]["restoration_labels"]
    summary = {
        "schema_version": "1.1.0",
        "status": RUN_STATUS,
        "protocol_id": PROTOCOL_ID,
        "source_execution": {
            "source_git_commit": source_commit,
            "contract_sha256": contract.sha256,
            "parent_contract_sha256": contract.parent.sha256,
            "formal_source_paths": list(FORMAL_SOURCE_PATHS),
        },
        "repair_identity": {
            "reporting_only": True,
            "selector_values_changed": False,
            "bootstrap_values_changed": False,
            "superseded_result_directory": contract.data["output_contract"][
                "superseded_result_directory"
            ],
            "superseded_result_fingerprint_before_repair": (
                None
                if superseded_fingerprint is None
                else dict(superseded_fingerprint)
            ),
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
    contract = RestorationV22SelectorGeometryRepairContract.load(
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
        raise ValueError("repair output must use the canonical Git path")
    superseded_dir = (
        root / contract.data["output_contract"]["superseded_result_directory"]
    ).resolve()
    if superseded_dir == output_dir or root not in superseded_dir.parents:
        raise ValueError("repair output must not alias the superseded v1 result")
    superseded_before = _result_tree_fingerprint(superseded_dir)
    label_archive = args.labels_archive.resolve()
    _validate_label_archive(label_archive, contract.parent.data)
    if args.mode == "run":
        _validate_run_git(root, args.source_git_commit)
    else:
        _validate_source_unchanged(root, args.source_git_commit)
    files = _expected_files(
        contract=contract,
        label_archive=label_archive,
        source_commit=args.source_git_commit,
        superseded_fingerprint=superseded_before,
    )
    if args.mode == "run":
        _write_new_result(output_dir, files)
        status = RUN_STATUS
    else:
        _validate_existing_result(output_dir, files)
        status = VALID_STATUS
    if _result_tree_fingerprint(superseded_dir) != superseded_before:
        raise RuntimeError("superseded v1 result changed during v2 repair")
    print(
        json.dumps(
            {
                "status": status,
                "source_git_commit": args.source_git_commit,
                "contract_sha256": contract.sha256,
                "parent_contract_sha256": contract.parent.sha256,
                "label_archive_sha256": sha256_file(label_archive),
                "output_dir": str(output_dir),
                "superseded_v1_preserved": True,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
