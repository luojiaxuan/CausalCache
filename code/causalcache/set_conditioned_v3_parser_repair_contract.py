"""Fail-closed contract for the v3 historical-protocol parser-only repair."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_conditioned_v3_historical_protocol_parser_repair_v1"
SOURCE_STATUS = "source_a_frozen_before_set_conditioned_v3_parser_repair"
SOURCE_VALIDATION_STATUS = "VALID_SET_CONDITIONED_V3_PARSER_REPAIR_SOURCE_A_V1"
EXECUTION_VALIDATION_STATUS = "VALID_SET_CONDITIONED_V3_PARSER_REPAIR_EXECUTION_B_V1"
EXECUTION_FREEZE_STATUS = "frozen_execution_b_before_v3_parser_repair"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_conditioned_v3_"
    "historical_protocol_parser_repair_v1.json"
)
RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_set_conditioned_v3_"
    "historical_protocol_parser_repair_runner_v1.json"
)
FROZEN_CONFIG_SHA256 = "da28f5851598703d255b5de7010002d00cb9cb7d43541d981910a67165bbbe0a"
REQUIRED_BRANCH = "luojiaxuan/set-conditioned-v3-pair-residual"
ORIGIN_URL = "https://github.com/luojiaxuan/CausalCache.git"
FAILURE_RECORD_COMMIT = "e0f70cf17fc8d0bbc00f1ceac51224149c41cc87"
PARENT_SOURCE_A_COMMIT = "d4e7b0065916513d18621925c212798018960c6b"
PARENT_EXECUTION_B_COMMIT = "2bfd8bc39c760e6590f923f88c2097d164b94bb4"
EXACT_HISTORICAL_PROTOCOL_ID = "causalcache_gate_v1_fresh16_evaluation_v1"
REPAIR_REPORT_NAME = "fresh16-development-report-parser-repair-v1.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ParserRepairContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be canonical relative POSIX")
    return value


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    fingerprint = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} has duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(value, label)


def validate_contract_data(value: Mapping[str, Any]) -> None:
    data = _mapping(value, "parser-repair contract")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "lineage",
        "source_freeze",
        "parser_repair",
        "input_contract",
        "parent_sealed_artifacts",
        "access_accounting",
        "runtime_contract",
        "output_contract",
    }
    if set(data) != expected_keys:
        raise ValueError("parser-repair contract keys drifted")
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("protocol_id") != PROTOCOL_ID
        or data.get("status") != SOURCE_STATUS
    ):
        raise ValueError("parser-repair contract identity drifted")

    lineage = _mapping(data["lineage"], "repair lineage")
    if (
        lineage.get("branch") != REQUIRED_BRANCH
        or lineage.get("origin_url") != ORIGIN_URL
        or lineage.get("parent_source_a_git_commit") != PARENT_SOURCE_A_COMMIT
        or lineage.get("parent_execution_b_git_commit")
        != PARENT_EXECUTION_B_COMMIT
        or lineage.get("failure_record_git_commit") != FAILURE_RECORD_COMMIT
    ):
        raise ValueError("parser-repair lineage drifted")
    failure = _mapping(lineage.get("failure_record"), "failure record")
    if failure != {
        "path": "data/results/set_conditioned_v3_pair_residual_attempt_v1/summary.json",
        "sha256": "81d5a36c22f2e36ac2c48e8a7b446b389c3e3ac30ebeb499c90fe14f03a52ce9",
        "size_bytes": 3203,
        "status": "INVALID_SET_CONDITIONED_V3_HISTORICAL_PROTOCOL_PARSER_ATTEMPT_V1",
    }:
        raise ValueError("failure-record identity drifted")

    source = _mapping(data["source_freeze"], "source freeze")
    if (
        source.get("required_branch") != REQUIRED_BRANCH
        or source.get("required_base_ancestor") != FAILURE_RECORD_COMMIT
        or source.get("execution_b_runner_freeze_path") != RUNNER_FREEZE_PATH
        or source.get("source_only_network_call_count") != 0
        or source.get("source_only_semantic_data_read_count") != 0
    ):
        raise ValueError("parser-repair source policy drifted")
    for field in (
        "execution_b_must_be_direct_single_parent_of_source_a",
        "execution_b_runner_freeze_must_be_unique_added_diff",
        "execution_b_runner_freeze_must_be_absent_during_source_a_validation",
        "execution_b_must_be_clean_and_pushed",
    ):
        if source.get(field) is not True:
            raise ValueError(f"parser-repair source field drifted: {field}")
    paths = tuple(
        _safe_relative(path, "required parser-repair source path")
        for path in _sequence(source.get("required_source_a_paths"), "source paths")
    )
    if len(paths) != 21 or len(set(paths)) != len(paths):
        raise ValueError("parser-repair source inventory drifted")

    repair = _mapping(data["parser_repair"], "parser repair")
    if (
        repair.get("failed_expected_protocol_id")
        != "causalcache_gate_v1_fresh16_primary_v1"
        or repair.get("exact_required_protocol_id")
        != EXACT_HISTORICAL_PROTOCOL_ID
        or repair.get("only_semantic_change")
        != "bind_historical_decision_artifact_to_its_exact_producer_protocol_id"
        or repair.get("protocol_allowlist_forbidden") is not True
    ):
        raise ValueError("historical parser repair drifted")
    for field in (
        "checkpoint_change_allowed",
        "training_report_change_allowed",
        "prediction_change_allowed",
        "selector_change_allowed",
        "threshold_change_allowed",
        "bootstrap_change_allowed",
    ):
        if repair.get(field) is not False:
            raise ValueError(f"forbidden repair mutation enabled: {field}")

    inputs = _mapping(data["input_contract"], "repair inputs")
    if set(inputs) != {"fresh_feature", "fresh_label", "historical_independent"}:
        raise ValueError("repair input inventory drifted")
    for name, raw in inputs.items():
        item = _mapping(raw, f"repair input {name}")
        _safe_relative(item.get("local_name"), f"repair input name {name}")
        if (
            _SHA256.fullmatch(str(item.get("sha256"))) is None
            or type(item.get("size_bytes")) is not int
            or item["size_bytes"] <= 0
        ):
            raise ValueError(f"repair input identity is malformed: {name}")

    artifacts = _sequence(data["parent_sealed_artifacts"], "parent artifacts")
    if len(artifacts) != 10:
        raise ValueError("parent sealed artifact denominator drifted")
    artifact_paths: list[str] = []
    for index, raw in enumerate(artifacts):
        item = _mapping(raw, f"parent artifact {index}")
        if set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("parent artifact schema drifted")
        artifact_paths.append(_safe_relative(item["path"], "parent artifact path"))
        if (
            _SHA256.fullmatch(str(item["sha256"])) is None
            or type(item["size_bytes"]) is not int
            or item["size_bytes"] <= 0
        ):
            raise ValueError("parent artifact identity is malformed")
    if len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError("parent artifact path is duplicated")

    accounting = _mapping(data["access_accounting"], "repair access accounting")
    expected_accounting = {
        "fresh_label_access_claim_count_total": 1,
        "parent_fresh_label_semantic_decode_attempt_count": 1,
        "repair_fresh_label_semantic_decode_attempt_count": 1,
        "fresh_label_semantic_decode_attempt_count_total": 2,
        "parent_historical_parse_attempt_count": 1,
        "repair_historical_parse_attempt_count": 1,
        "historical_parse_attempt_count_total": 2,
        "training_run_count_total": 1,
        "prediction_generation_count_total": 1,
        "development_report_completion_count_total": 1,
        "confirm20_access_count": 0,
        "legacy_dev5_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
    }
    if accounting != expected_accounting:
        raise ValueError("repair access accounting drifted")
    runtime = _mapping(data["runtime_contract"], "repair runtime")
    if runtime != {
        "host": "hyper00",
        "device": "cpu",
        "gpu_count": 0,
        "thread_count": 1,
        "training_entrypoint_allowed": False,
        "prediction_generation_allowed": False,
    }:
        raise ValueError("repair runtime drifted")
    output = _mapping(data["output_contract"], "repair output")
    if (
        output.get("parent_development_report_path")
        != "fresh16-development-report.json"
        or output.get("parent_development_report_must_remain_absent") is not True
        or output.get("repair_report_path") != REPAIR_REPORT_NAME
        or output.get("repair_report_exclusive_create") is not True
        or output.get("repair_report_byte_replay_required") is not True
        or output.get("may_authorize_confirm") is not False
        or output.get("may_change_v1_verdict") is not False
    ):
        raise ValueError("repair output boundary drifted")


def load_frozen_parser_repair_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
) -> ParserRepairContract:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = root / source_path
    payload = _regular_file_bytes(source_path, label="parser-repair contract")
    digest = sha256_bytes(payload)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            f"parser-repair contract hash drifted: expected {FROZEN_CONFIG_SHA256}, got {digest}"
        )
    data = _strict_json(payload, label="parser-repair contract")
    validate_contract_data(data)
    return ParserRepairContract(data, digest, root, source_path.resolve())


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()
        raise ValueError(
            f"Git parser-repair identity check failed: {' '.join(arguments)}"
            + (f": {detail}" if detail else "")
        )
    return result.stdout.strip()


def _git_blob(root: Path, specification: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), "show", specification],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError("parser-repair committed blob is unavailable")
    return result.stdout


def _source_inventory(
    contract: ParserRepairContract,
) -> tuple[tuple[dict[str, Any], ...], str]:
    paths = _sequence(
        _mapping(contract.data["source_freeze"], "source freeze")[
            "required_source_a_paths"
        ],
        "required source paths",
    )
    inventory = []
    for relative in paths:
        canonical = _safe_relative(relative, "required source path")
        payload = _regular_file_bytes(
            contract.repository_root / canonical,
            label=canonical,
        )
        inventory.append(
            {"path": canonical, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        )
    frozen = tuple(inventory)
    return frozen, sha256_bytes(canonical_json_bytes(frozen))


def _clean_pushed_identity(root: Path) -> tuple[str, str]:
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("parser-repair repository root differs from Git top level")
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    if branch != REQUIRED_BRANCH or _COMMIT.fullmatch(head) is None:
        raise ValueError("parser-repair branch or HEAD drifted")
    if _git(root, "remote", "get-url", "origin") != ORIGIN_URL:
        raise ValueError("parser-repair origin URL drifted")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("parser-repair worktree must be clean")
    if _git(root, "rev-parse", f"origin/{REQUIRED_BRANCH}") != head:
        raise ValueError("parser-repair HEAD differs from pushed tracking branch")
    return branch, head


def _live_remote_branch(root: Path) -> str:
    reference = f"refs/heads/{REQUIRED_BRANCH}"
    output = _git(root, "ls-remote", "--exit-code", "origin", reference)
    lines = output.splitlines()
    suffix = f"\t{reference}"
    if len(lines) != 1 or not lines[0].endswith(suffix):
        raise ValueError("parser-repair live origin response drifted")
    commit = lines[0][: -len(suffix)]
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("parser-repair live origin commit is malformed")
    return commit


def validate_parser_repair_source_a(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
    expected_source_a_git_commit: str | None = None,
) -> dict[str, Any]:
    contract = load_frozen_parser_repair_contract(
        path,
        repository_root=repository_root,
    )
    root = contract.repository_root
    if os.path.lexists(root / RUNNER_FREEZE_PATH):
        raise ValueError("parser-repair Execution-B runner freeze must be absent")
    branch, head = _clean_pushed_identity(root)
    if expected_source_a_git_commit is not None and head != expected_source_a_git_commit:
        raise ValueError("requested parser-repair Source-A differs from HEAD")
    _git(root, "merge-base", "--is-ancestor", FAILURE_RECORD_COMMIT, head)
    inventory, inventory_sha256 = _source_inventory(contract)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": SOURCE_VALIDATION_STATUS,
        "contract_sha256": contract.sha256,
        "source_a": {
            "branch": branch,
            "head": head,
            "source_inventory": list(inventory),
            "source_inventory_sha256": inventory_sha256,
            "required_path_count": len(inventory),
        },
        "network_call_count": 0,
        "semantic_data_read_count": 0,
        "repair_execution_authorized": False,
    }


def parser_repair_execution_b_freeze_payload(
    contract: ParserRepairContract,
    *,
    source_a_git_commit: str,
    source_inventory_sha256: str,
) -> dict[str, Any]:
    if _COMMIT.fullmatch(source_a_git_commit) is None:
        raise ValueError("parser-repair Source-A commit is malformed")
    if _SHA256.fullmatch(source_inventory_sha256) is None:
        raise ValueError("parser-repair source inventory SHA-256 is malformed")
    accounting = _mapping(contract.data["access_accounting"], "access accounting")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": EXECUTION_FREEZE_STATUS,
        "source_a_git_commit": source_a_git_commit,
        "source_inventory_sha256": source_inventory_sha256,
        "contract_sha256": contract.sha256,
        "runner_freeze_path": RUNNER_FREEZE_PATH,
        "execution_b_unique_diff": [RUNNER_FREEZE_PATH],
        "parent_execution_b_git_commit": PARENT_EXECUTION_B_COMMIT,
        "parent_label_blind_seal_sha256": "9029f0dce9f7c76d159c3c3860aa95e43fdacf6daaf8b2ee2c65d34248775470",
        "parent_label_access_claim_sha256": "7bb9a95b326b7a618875fbf975837e397aa726d193f8cfd4f27d91b7ef5077dd",
        "parent_prediction_sha256": "23235c59403abc4eef45754bbf63981c935ffacf72a51a0bd925884eedd34ec3",
        "access_accounting_sha256": sha256_bytes(canonical_json_bytes(accounting)),
        "runtime_device": "cpu",
        "gpu_count": 0,
        "confirm20_access_count": 0,
    }


def _load_runner_freeze(
    contract: ParserRepairContract,
    *,
    source_a_git_commit: str,
    source_inventory_sha256: str,
) -> tuple[Mapping[str, Any], bytes]:
    payload = _regular_file_bytes(
        contract.repository_root / RUNNER_FREEZE_PATH,
        label="parser-repair Execution-B runner freeze",
    )
    value = _strict_json(payload, label="parser-repair Execution-B runner freeze")
    expected = parser_repair_execution_b_freeze_payload(
        contract,
        source_a_git_commit=source_a_git_commit,
        source_inventory_sha256=source_inventory_sha256,
    )
    if value != expected or payload != canonical_json_bytes(expected) + b"\n":
        raise ValueError("parser-repair runner freeze differs from canonical bytes")
    return value, payload


def validate_parser_repair_execution_b(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path | None = None,
    source_a_git_commit: str,
    execution_b_git_commit: str,
) -> dict[str, Any]:
    if (
        _COMMIT.fullmatch(source_a_git_commit) is None
        or _COMMIT.fullmatch(execution_b_git_commit) is None
    ):
        raise ValueError("parser-repair requested Git commit is malformed")
    contract = load_frozen_parser_repair_contract(
        path,
        repository_root=repository_root,
    )
    root = contract.repository_root
    branch, head = _clean_pushed_identity(root)
    if head != execution_b_git_commit or _live_remote_branch(root) != head:
        raise ValueError("parser-repair Execution-B is not clean live-pushed HEAD")
    _git(root, "merge-base", "--is-ancestor", FAILURE_RECORD_COMMIT, source_a_git_commit)
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    if parents != [head, source_a_git_commit]:
        raise ValueError("parser-repair Execution-B must directly descend Source-A")
    changed = tuple(
        line
        for line in _git(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            source_a_git_commit,
            head,
        ).splitlines()
        if line
    )
    expected_change = f"A\t{RUNNER_FREEZE_PATH}"
    if changed != (expected_change,):
        raise ValueError("parser-repair Execution-B has a non-runner source diff")
    inventory, inventory_sha256 = _source_inventory(contract)
    freeze, freeze_payload = _load_runner_freeze(
        contract,
        source_a_git_commit=source_a_git_commit,
        source_inventory_sha256=inventory_sha256,
    )
    if _git_blob(root, f"{head}:{RUNNER_FREEZE_PATH}") != freeze_payload:
        raise ValueError("parser-repair runner differs from committed blob")
    tree = _git(root, "ls-tree", head, "--", RUNNER_FREEZE_PATH)
    if not tree.startswith("100644 blob ") or not tree.endswith(
        f"\t{RUNNER_FREEZE_PATH}"
    ):
        raise ValueError("parser-repair runner must be a mode-100644 blob")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": EXECUTION_VALIDATION_STATUS,
        "contract_sha256": contract.sha256,
        "source_a_git_commit": source_a_git_commit,
        "execution_b_git_commit": head,
        "branch": branch,
        "source_inventory_sha256": inventory_sha256,
        "required_path_count": len(inventory),
        "runner_freeze_sha256": sha256_bytes(freeze_payload),
        "runner_freeze": dict(freeze),
        "confirm20_access_count": 0,
        "gpu_operation_count": 0,
    }


def validate_parent_sealed_artifacts(
    contract: ParserRepairContract,
    *,
    output_dir: Path,
    require_repair_report_absent: bool,
) -> dict[str, Any]:
    inventory = []
    for raw in _sequence(
        contract.data["parent_sealed_artifacts"],
        "parent sealed artifacts",
    ):
        item = _mapping(raw, "parent sealed artifact")
        relative = _safe_relative(item["path"], "parent sealed artifact path")
        payload = _regular_file_bytes(output_dir / relative, label=relative)
        observed = {
            "path": relative,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }
        if observed != item:
            raise ValueError(f"parent sealed artifact drifted: {relative}")
        inventory.append(observed)
    if os.path.lexists(output_dir / "fresh16-development-report.json"):
        raise ValueError("invalid parent attempt unexpectedly has a development report")
    repair_path = output_dir / REPAIR_REPORT_NAME
    if require_repair_report_absent and os.path.lexists(repair_path):
        raise ValueError("parser-repair report already exists")
    return {
        "artifact_count": len(inventory),
        "artifact_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "label_blind_seal_sha256": inventory[-2]["sha256"],
        "label_access_claim_sha256": inventory[-1]["sha256"],
        "parent_development_report_absent": True,
    }


def exact_input_bytes(
    contract: ParserRepairContract,
    *,
    input_dir: Path,
    name: str,
) -> bytes:
    inputs = _mapping(contract.data["input_contract"], "repair inputs")
    item = _mapping(inputs.get(name), f"repair input {name}")
    payload = _regular_file_bytes(
        input_dir / str(item["local_name"]),
        label=f"repair input {name}",
    )
    if sha256_bytes(payload) != item["sha256"] or len(payload) != item["size_bytes"]:
        raise ValueError(f"repair input identity drifted: {name}")
    return payload


__all__ = [
    "CANONICAL_CONFIG_PATH",
    "EXACT_HISTORICAL_PROTOCOL_ID",
    "EXECUTION_FREEZE_STATUS",
    "FROZEN_CONFIG_SHA256",
    "PROTOCOL_ID",
    "ParserRepairContract",
    "REPAIR_REPORT_NAME",
    "RUNNER_FREEZE_PATH",
    "canonical_json_bytes",
    "exact_input_bytes",
    "load_frozen_parser_repair_contract",
    "parser_repair_execution_b_freeze_payload",
    "sha256_bytes",
    "validate_contract_data",
    "validate_parent_sealed_artifacts",
    "validate_parser_repair_execution_b",
    "validate_parser_repair_source_a",
]
