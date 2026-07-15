"""Run the frozen CPU-only subset-search ablation from clean pushed main."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.subset_search_ablation import build_scientific_payload, sha256_file


GIT_SHA = re.compile(r"[0-9a-f]{40}")
SOURCE_FILES = (
    "code/causalcache/subset_search.py",
    "code/causalcache/subset_search_ablation.py",
    "code/scripts/run_subset_search_ablation.py",
    "code/scripts/validate_subset_search_ablation.py",
    "code/configs/subset_search_ablation_v1.json",
    "data/fixtures/subset_search_scenarios_v1.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_clean_pushed_main(root: Path, expected_commit: str) -> dict[str, str]:
    if GIT_SHA.fullmatch(expected_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("repository root differs from the Git top level")
    head = _git(root, "rev-parse", "HEAD")
    origin_main = _git(root, "rev-parse", "refs/remotes/origin/main")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    remote_url = _git(root, "remote", "get-url", "origin")
    advertised = _git(root, "ls-remote", "--heads", "origin", "refs/heads/main")
    if head != expected_commit or origin_main != expected_commit or branch != "main" or status:
        raise ValueError("formal ablation requires clean local main synchronized with origin/main")
    if advertised != f"{expected_commit}\trefs/heads/main":
        raise ValueError("formal ablation source commit is not the advertised remote main")
    return {
        "branch": branch,
        "commit": head,
        "origin_main": origin_main,
        "remote_main": expected_commit,
        "remote_url": remote_url,
        "worktree": "clean_including_untracked",
    }


def _require_absolute(path: Path, name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path.resolve()


def _source_inventory(root: Path, source_commit: str) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"source file is missing: {relative}")
        working_hash = sha256_file(path)
        blob = subprocess.run(
            ("git", "-C", str(root), "show", f"{source_commit}:{relative}"),
            check=True,
            capture_output=True,
        ).stdout
        blob_hash = hashlib.sha256(blob).hexdigest()
        if blob_hash != working_hash:
            raise ValueError(f"working source differs from source commit: {relative}")
        inventory.append({"path": relative, "sha256": working_hash})
    return inventory


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _method(scenario: dict[str, Any], name: str) -> dict[str, Any]:
    return next(method for method in scenario["methods"] if method["method"] == name)


def render_report(summary: dict[str, Any]) -> str:
    payload = summary["scientific_payload"]
    scenarios = {scenario["scenario_id"]: scenario for scenario in payload["scenarios"]}
    phase0 = scenarios["existing_synthetic_phase0_mixed"]
    complement = scenarios["controlled_complementary_trap"]
    real_rows = [
        scenario
        for scenario in payload["scenarios"]
        if scenario["source_type"] == "selection_biased_v1_cached_policy_table"
    ]
    lines = [
        "# Subset-search ablation v1",
        "",
        "> 这是 CPU-only optimizer diagnostic 与旧 v1 development table 的 post-hoc replay；不是新的 policy evidence，",
        "> 不评估 learned gate/closed-loop success，也不改变 v2.1 的 frozen NO-GO。",
        "",
        "## 核心结果",
        "",
        f"- Phase-0 average-marginal knapsack / exact subset：`{_method(phase0, 'exact_average_marginal_positive_value_knapsack')['utility_ratio_to_exact_subset']:.6f}`；这是 objective-projection gap，不是 greedy search gap。",
        f"- Phase-0 true conditional greedy / exact subset：`{_method(phase0, 'true_conditional_greedy_raw_gain')['utility_ratio_to_exact_subset']:.6f}`。",
        f"- Complementary trap：raw greedy 选择 `{_method(complement, 'true_conditional_greedy_raw_gain')['selected_coalition']}`，utility ratio `{_method(complement, 'true_conditional_greedy_raw_gain')['utility_ratio_to_exact_subset']:.3f}`；2x2 exchange 选择 `{_method(complement, 'true_conditional_greedy_raw_gain_exchange_2x2')['selected_coalition']}`。",
        f"- Beam-2 / Beam-4 ratio：`{_method(complement, 'true_utility_beam_2')['utility_ratio_to_exact_subset']:.3f}` / `{_method(complement, 'true_utility_beam_4')['utility_ratio_to_exact_subset']:.3f}`。",
        "",
        "## 旧真实 policy coalition-table replay",
        "",
        "| State | Budget | Exact subset | True greedy | Greedy / exact | Exact queries | Greedy queries |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for scenario in real_rows:
        exact = _method(scenario, "exact_subset")
        greedy = _method(scenario, "true_conditional_greedy_raw_gain")
        lines.append(
            f"| {scenario['metadata']['decision_step_id']} | {scenario['budget']} | "
            f"`{exact['selected_coalition']}` | `{greedy['selected_coalition']}` | "
            f"{greedy['utility_ratio_to_exact_subset']:.3f} | "
            f"{exact['unique_set_utility_evaluations']} | {greedy['unique_set_utility_evaluations']} |"
        )
    lines.extend(
        [
            "",
            "四个缓存 state/budget 上 true greedy 都等于 exact。样本只有两个 selection-biased states、成本相等且最多选两个 event，",
            "所以这只能验证 replay/search implementation 一致性，不能证明 stronger search 在真实任务上优于 greedy。",
            "",
            "## 成本与声明边界",
            "",
            "- `unique_set_utility_evaluations` 计入所有被评估但未采用的 coalition；exchange 还包含 seed greedy 成本。",
            "- true-U greedy/exchange/beam 的一次 query 在真实部署中意味着一次 policy rerun；本地 CSV lookup wall time 不是线上 latency。",
            "- equal-cost scenario 省略 density greedy，因为它与 raw greedy 完全相同。",
            "- learned conditional-marginal head 不能直接无定义地用于 beam/removal；需要独立的 direct set-utility 或 path-score contract。",
            "- Exact 只保证冻结 single-step restoration utility 最优，不保证 terminal task success 最优。",
            "",
            "## Provenance",
            "",
            f"- Source Git commit：`{summary['provenance']['git']['commit']}`",
            f"- Scientific payload SHA256：`{summary['scientific_payload_sha256']}`",
            f"- Device：CPU；GPU/policy operations：`0`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = _require_absolute(args.repository_root, "--repository-root")
    config = _require_absolute(args.config, "--config")
    output_dir = _require_absolute(args.output_dir, "--output-dir")
    if output_dir.exists():
        raise ValueError("formal output directory must not already exist")
    git_identity = validate_clean_pushed_main(root, args.source_git_commit)
    source_inventory = _source_inventory(root, args.source_git_commit)
    started_at = _utc_now()
    started_clock = time.monotonic()
    scientific_payload = build_scientific_payload(root, config)
    duration = time.monotonic() - started_clock
    completed_at = _utc_now()
    summary = {
        "schema_version": "1.0.0",
        "experiment_id": "subset-search-ablation-v1",
        "scientific_payload_sha256": _canonical_sha(scientific_payload),
        "scientific_payload": scientific_payload,
        "provenance": {
            "git": git_identity,
            "source_files": source_inventory,
            "argv": list(sys.argv),
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_time_seconds": round(duration, 6),
            "host": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pid": os.getpid(),
            "device": "cpu",
            "gpu_operations_executed": 0,
            "policy_operations_executed": 0,
            "existing_git_inputs_read": True,
            "new_external_or_untouched_data_accessed": False,
            "seed": "not_applicable_deterministic",
        },
    }
    report = render_report(summary)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "README.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "scientific_payload_sha256": summary["scientific_payload_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
