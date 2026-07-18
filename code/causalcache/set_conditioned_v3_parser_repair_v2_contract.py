"""Overlay contract for exact state-id joining in v3 parser repair v2."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.set_conditioned_v3_parser_repair_contract import (
    _clean_pushed_identity,
    _git,
    _git_blob,
    _live_remote_branch,
    _mapping,
    _regular_file_bytes,
    _safe_relative,
    _sequence,
    _strict_json,
    canonical_json_bytes,
    exact_input_bytes,
    load_frozen_parser_repair_contract,
    sha256_bytes,
    validate_parent_sealed_artifacts,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_conditioned_v3_historical_protocol_parser_repair_v2"
SOURCE_STATUS = "source_a_frozen_before_set_conditioned_v3_parser_repair_v2"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_conditioned_v3_"
    "historical_protocol_parser_repair_v2.json"
)
RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_set_conditioned_v3_"
    "historical_protocol_parser_repair_v2_runner.json"
)
FROZEN_CONFIG_SHA256 = "1da7073c55437afdc8e4ca85dc59a414c6ed331619828210af54347f80b63ce4"
REQUIRED_BRANCH = "luojiaxuan/set-conditioned-v3-pair-residual"
BASE_ANCESTOR = "1c57799bd4ce4ebb98563ab92608ae27b1f69cc3"
V1_REPAIR_EXECUTION_B = "f53954f9e2c142738c77ffa150f8ddc8c7088ff2"
REPAIR_REPORT_NAME = "fresh16-development-report-parser-repair-v2.json"
EXECUTION_FREEZE_STATUS = "frozen_execution_b_before_v3_parser_repair_v2"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ParserRepairV2Contract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path


def validate_contract_data(value: Mapping[str, Any]) -> None:
    data = _mapping(value, "parser-repair v2 contract")
    if set(data) != {
        "schema_version",
        "protocol_id",
        "status",
        "base_repair",
        "source_freeze",
        "state_join_repair",
        "access_accounting",
        "runtime",
        "output",
    }:
        raise ValueError("parser-repair v2 contract keys drifted")
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("protocol_id") != PROTOCOL_ID
        or data.get("status") != SOURCE_STATUS
    ):
        raise ValueError("parser-repair v2 identity drifted")
    base = _mapping(data["base_repair"], "base repair")
    attempt = _mapping(base.get("attempt_record"), "v1 repair attempt record")
    if (
        base.get("contract_sha256")
        != "da28f5851598703d255b5de7010002d00cb9cb7d43541d981910a67165bbbe0a"
        or base.get("execution_b_git_commit") != V1_REPAIR_EXECUTION_B
        or attempt
        != {
            "path": "data/results/set_conditioned_v3_parser_repair_v1_attempt/summary.json",
            "sha256": "071fbe19f1cc5c02b1ed2b7891614a77e7886399e821eaba616620e10497ab51",
            "size_bytes": 1733,
            "git_commit": BASE_ANCESTOR,
            "status": "INVALID_PRE_LABEL_STATE_ORDER_PARSER_REPAIR_V1",
        }
    ):
        raise ValueError("parser-repair v2 base identity drifted")
    source = _mapping(data["source_freeze"], "v2 source freeze")
    if (
        source.get("branch") != REQUIRED_BRANCH
        or source.get("required_base_ancestor") != BASE_ANCESTOR
        or source.get("runner_freeze_path") != RUNNER_FREEZE_PATH
        or source.get("runner_must_be_absent_at_source_a") is not True
        or source.get("execution_b_direct_single_parent") is not True
        or source.get("execution_b_unique_added_runner_diff") is not True
    ):
        raise ValueError("parser-repair v2 source policy drifted")
    paths = tuple(
        _safe_relative(path, "v2 required source path")
        for path in _sequence(source.get("required_source_a_paths"), "v2 source paths")
    )
    if len(paths) != 21 or len(set(paths)) != len(paths):
        raise ValueError("parser-repair v2 source inventory drifted")
    join = _mapping(data["state_join_repair"], "state-id join repair")
    if (
        join.get("v2_exact_rule")
        != "require_unique_state_ids_and_equal_state_id_sets_then_join_by_state_id"
        or join.get("observed_feature_state_count") != 48
        or join.get("observed_historical_state_count") != 48
        or join.get("observed_unique_count_each") != 48
        or join.get("observed_state_id_set_equal") is not True
        or join.get("observed_positional_mismatch_count") != 45
        or join.get("protocol_allowlist_forbidden") is not True
        or join.get(
            "checkpoint_prediction_selector_threshold_bootstrap_change_allowed"
        )
        is not False
    ):
        raise ValueError("parser-repair v2 exact join drifted")
    for field in (
        "record_schema_unchanged",
        "seed_feasibility_unchanged",
        "selection_digest_check_unchanged",
    ):
        if join.get(field) is not True:
            raise ValueError(f"parser-repair v2 invariant drifted: {field}")
    accounting = _mapping(data["access_accounting"], "v2 access accounting")
    if (
        accounting.get("fresh_label_access_claim_count_total") != 1
        or accounting.get("parent_fresh_label_semantic_decode_attempt_count") != 1
        or accounting.get(
            "parser_repair_v1_fresh_label_semantic_decode_attempt_count"
        )
        != 0
        or accounting.get(
            "parser_repair_v2_fresh_label_semantic_decode_attempt_count"
        )
        != 1
        or accounting.get("fresh_label_semantic_decode_attempt_count_total") != 2
        or accounting.get("parent_historical_parse_attempt_count") != 1
        or accounting.get("parser_repair_v1_historical_parse_attempt_count") != 1
        or accounting.get("parser_repair_v2_historical_parse_attempt_count") != 1
        or accounting.get("historical_parse_attempt_count_total") != 3
        or accounting.get("parser_repair_v2_real_artifact_dry_run_count") != 0
        or accounting.get("training_run_count_total") != 1
        or accounting.get("prediction_generation_count_total") != 1
        or accounting.get("development_report_completion_count_total") != 1
    ):
        raise ValueError("parser-repair v2 access accounting drifted")
    for field in (
        "confirm20_access_count",
        "legacy_dev5_access_count",
        "matched_nll_evaluation_count",
        "closed_loop_episode_count",
        "policy_forward_count",
        "gpu_operation_count",
    ):
        if accounting.get(field) != 0:
            raise ValueError(f"parser-repair v2 forbidden operation drifted: {field}")
    runtime = _mapping(data["runtime"], "v2 runtime")
    if runtime != {
        "host": "hyper00",
        "device": "cpu",
        "gpu_count": 0,
        "thread_count": 1,
        "training_or_prediction_entrypoint_allowed": False,
    }:
        raise ValueError("parser-repair v2 runtime drifted")
    output = _mapping(data["output"], "v2 output")
    if (
        output.get("parent_report_path") != "fresh16-development-report.json"
        or output.get("v1_repair_report_path")
        != "fresh16-development-report-parser-repair-v1.json"
        or output.get("v2_repair_report_path") != REPAIR_REPORT_NAME
        or output.get("parent_and_v1_reports_must_be_absent_before_v2") is not True
        or output.get("validate_requires_expected_report_sha256") is not True
        or output.get("validate_binds_execution_lineage_and_authority_boundary")
        is not True
        or output.get("may_authorize_confirm") is not False
        or output.get("may_change_v1_verdict") is not False
    ):
        raise ValueError("parser-repair v2 output boundary drifted")


def load_frozen_parser_repair_v2_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
) -> ParserRepairV2Contract:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    payload = _regular_file_bytes(source, label="parser-repair v2 contract")
    digest = sha256_bytes(payload)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError("parser-repair v2 contract SHA-256 drifted")
    data = _strict_json(payload, label="parser-repair v2 contract")
    validate_contract_data(data)
    return ParserRepairV2Contract(data, digest, root, source.resolve())


def _source_inventory(
    contract: ParserRepairV2Contract,
) -> tuple[tuple[dict[str, Any], ...], str]:
    paths = _sequence(
        _mapping(contract.data["source_freeze"], "v2 source freeze")[
            "required_source_a_paths"
        ],
        "v2 source paths",
    )
    inventory = []
    for relative in paths:
        canonical = _safe_relative(relative, "v2 source path")
        payload = _regular_file_bytes(contract.repository_root / canonical, label=canonical)
        inventory.append(
            {"path": canonical, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        )
    frozen = tuple(inventory)
    return frozen, sha256_bytes(canonical_json_bytes(frozen))


def validate_parser_repair_v2_source_a(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    contract = load_frozen_parser_repair_v2_contract(
        path,
        repository_root=repository_root,
    )
    if os.path.lexists(contract.repository_root / RUNNER_FREEZE_PATH):
        raise ValueError("parser-repair v2 runner freeze must be absent at Source-A")
    branch, head = _clean_pushed_identity(contract.repository_root)
    _git(contract.repository_root, "merge-base", "--is-ancestor", BASE_ANCESTOR, head)
    inventory, digest = _source_inventory(contract)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VALID_SET_CONDITIONED_V3_PARSER_REPAIR_V2_SOURCE_A",
        "contract_sha256": contract.sha256,
        "source_a": {
            "branch": branch,
            "head": head,
            "source_inventory": list(inventory),
            "source_inventory_sha256": digest,
            "required_path_count": len(inventory),
        },
        "semantic_label_read_count": 0,
        "execution_authorized": False,
    }


def parser_repair_v2_execution_b_freeze_payload(
    contract: ParserRepairV2Contract,
    *,
    source_a_git_commit: str,
    source_inventory_sha256: str,
) -> dict[str, Any]:
    if (
        _COMMIT.fullmatch(source_a_git_commit) is None
        or _SHA256.fullmatch(source_inventory_sha256) is None
    ):
        raise ValueError("parser-repair v2 Source-A identity is malformed")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": EXECUTION_FREEZE_STATUS,
        "source_a_git_commit": source_a_git_commit,
        "source_inventory_sha256": source_inventory_sha256,
        "contract_sha256": contract.sha256,
        "runner_freeze_path": RUNNER_FREEZE_PATH,
        "execution_b_unique_diff": [RUNNER_FREEZE_PATH],
        "base_repair_execution_b_git_commit": V1_REPAIR_EXECUTION_B,
        "base_repair_attempt_record_sha256": (
            "071fbe19f1cc5c02b1ed2b7891614a77e7886399e821eaba616620e10497ab51"
        ),
        "exact_join_rule": (
            "require_unique_state_ids_and_equal_state_id_sets_then_join_by_state_id"
        ),
        "fresh_label_semantic_decode_attempt_count_total": 2,
        "historical_parse_attempt_count_total": 3,
        "parser_repair_v2_real_artifact_dry_run_count": 0,
        "confirm20_access_count": 0,
        "gpu_count": 0,
    }


def _load_runner_freeze(
    contract: ParserRepairV2Contract,
    *,
    source_a_git_commit: str,
    source_inventory_sha256: str,
) -> tuple[Mapping[str, Any], bytes]:
    payload = _regular_file_bytes(
        contract.repository_root / RUNNER_FREEZE_PATH,
        label="parser-repair v2 runner freeze",
    )
    value = _strict_json(payload, label="parser-repair v2 runner freeze")
    expected = parser_repair_v2_execution_b_freeze_payload(
        contract,
        source_a_git_commit=source_a_git_commit,
        source_inventory_sha256=source_inventory_sha256,
    )
    if value != expected or payload != canonical_json_bytes(expected) + b"\n":
        raise ValueError("parser-repair v2 runner freeze bytes drifted")
    return value, payload


def validate_parser_repair_v2_execution_b(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
    source_a_git_commit: str,
    execution_b_git_commit: str,
) -> dict[str, Any]:
    contract = load_frozen_parser_repair_v2_contract(
        path,
        repository_root=repository_root,
    )
    branch, head = _clean_pushed_identity(contract.repository_root)
    if head != execution_b_git_commit or _live_remote_branch(contract.repository_root) != head:
        raise ValueError("parser-repair v2 Execution-B is not clean live-pushed HEAD")
    parents = _git(
        contract.repository_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        head,
    ).split()
    if parents != [head, source_a_git_commit]:
        raise ValueError("parser-repair v2 Execution-B is not direct Source-A child")
    changed = tuple(
        line
        for line in _git(
            contract.repository_root,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            source_a_git_commit,
            head,
        ).splitlines()
        if line
    )
    if changed != (f"A\t{RUNNER_FREEZE_PATH}",):
        raise ValueError("parser-repair v2 Execution-B has extra source diffs")
    inventory, digest = _source_inventory(contract)
    freeze, payload = _load_runner_freeze(
        contract,
        source_a_git_commit=source_a_git_commit,
        source_inventory_sha256=digest,
    )
    if _git_blob(contract.repository_root, f"{head}:{RUNNER_FREEZE_PATH}") != payload:
        raise ValueError("parser-repair v2 runner differs from committed blob")
    tree = _git(contract.repository_root, "ls-tree", head, "--", RUNNER_FREEZE_PATH)
    if not tree.startswith("100644 blob ") or not tree.endswith(f"\t{RUNNER_FREEZE_PATH}"):
        raise ValueError("parser-repair v2 runner must be mode 100644")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VALID_SET_CONDITIONED_V3_PARSER_REPAIR_V2_EXECUTION_B",
        "contract_sha256": contract.sha256,
        "source_a_git_commit": source_a_git_commit,
        "execution_b_git_commit": head,
        "branch": branch,
        "source_inventory_sha256": digest,
        "required_path_count": len(inventory),
        "runner_freeze_sha256": sha256_bytes(payload),
        "runner_freeze": dict(freeze),
        "confirm20_access_count": 0,
        "gpu_operation_count": 0,
    }


def validate_v2_parent_artifacts(
    contract: ParserRepairV2Contract,
    *,
    output_dir: Path,
    require_v2_report_absent: bool,
) -> Mapping[str, Any]:
    base = load_frozen_parser_repair_contract(repository_root=contract.repository_root)
    parent = validate_parent_sealed_artifacts(
        base,
        output_dir=output_dir,
        require_repair_report_absent=True,
    )
    v2_path = output_dir / REPAIR_REPORT_NAME
    if require_v2_report_absent and os.path.lexists(v2_path):
        raise ValueError("parser-repair v2 report already exists")
    return parent


def exact_v2_input_bytes(
    contract: ParserRepairV2Contract,
    *,
    input_dir: Path,
    name: str,
) -> bytes:
    base = load_frozen_parser_repair_contract(repository_root=contract.repository_root)
    return exact_input_bytes(base, input_dir=input_dir, name=name)


__all__ = [
    "CANONICAL_CONFIG_PATH",
    "FROZEN_CONFIG_SHA256",
    "PROTOCOL_ID",
    "REPAIR_REPORT_NAME",
    "RUNNER_FREEZE_PATH",
    "ParserRepairV2Contract",
    "exact_v2_input_bytes",
    "load_frozen_parser_repair_v2_contract",
    "parser_repair_v2_execution_b_freeze_payload",
    "validate_contract_data",
    "validate_parser_repair_v2_execution_b",
    "validate_parser_repair_v2_source_a",
    "validate_v2_parent_artifacts",
]
