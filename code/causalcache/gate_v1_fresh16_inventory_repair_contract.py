"""Fail-closed overlay contract for the fresh-16 input-inventory repair."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.gate_v1_fresh16_evaluation_contract import (
    CANONICAL_CONFIG_PATH as PARENT_CONFIG_PATH,
    FROZEN_CONFIG_SHA256 as PARENT_CONFIG_SHA256,
    Fresh16EvaluationContract,
    canonical_json_bytes,
    load_frozen_fresh16_evaluation_contract,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_fresh16_inventory_repair_v1"
SOURCE_STATUS = (
    "source_only_frozen_before_fresh16_inventory_repair_semantic_access"
)
VALIDATION_STATUS = "VALID_SOURCE_ONLY_GATE_V1_FRESH16_INVENTORY_REPAIR_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json"
)
RUNNER_FREEZE_B_PATH = (
    "code/configs/causalcache_gate_v1_fresh16_inventory_repair_runner_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "5ba1b2d433c01defa0faae92a163dc9a9a916d967218abdc7607bd3724829a44"
)

PARENT_SOURCE_A_GIT_COMMIT = "97694eff052ecbdc5f12f58b6f9ee10f4dd616ab"
PARENT_EXECUTION_B_GIT_COMMIT = "a8bb27ccf9b8820c1d8487f63883f813e6845645"
FAILURE_EVIDENCE_GIT_COMMIT = "2d85d088591fa1c7196f2a8e220439120f64f600"
PARENT_RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_gate_v1_fresh16_evaluation_runner_v1.json"
)
PARENT_RUNNER_FREEZE_SHA256 = (
    "539db6112184d169f57d6f912399cc1b159d8590a53b3f76378b455c460e211b"
)
FAILURE_SUMMARY_PATH = (
    "data/results/gate_v1_fresh16_evaluation_v1_attempt/summary.json"
)
FAILURE_SUMMARY_SHA256 = (
    "43f005ef6ac80f71283b4346d515a47c9ed16e552e1657b6947163f2695d2354"
)
FAILURE_README_PATH = (
    "data/results/gate_v1_fresh16_evaluation_v1_attempt/README.md"
)
FAILURE_README_SHA256 = (
    "5d4f216c509811dbee4e9d7f1f6f65af8a377a5d7d8bd20492201fbf767ee363"
)

DERIVED_REPO = "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile"
DERIVED_REVISION = "630363a6adb692d72774f16dd0653a50216313ff"
DERIVED_FULL_INVENTORY_SHA256 = (
    "f881b67d028dfe7ce147df0611213a2b9147b2a1432563180e68ebfaee47a202"
)

NEW_EXECUTION_NAMESPACE = "gate-v1-fresh16-evaluation-inventory-repair-v1"
NEW_ARTIFACT_DIRECTORY = (
    "/data/artifacts/causalcache/gate-v1-fresh16-evaluation-inventory-repair-v1"
)
NEW_RUNTIME_RECEIPT_PATH = (
    "/data/experiments/causalcache/"
    ".gate-v1-fresh16-evaluation-inventory-repair-v1.docker-inspect.json"
)
NEW_DESTINATION_REPO = (
    "gavinlaw/causalcache-gate-v1-fresh16-inventory-repair-mobile"
)
NEW_DESTINATION_TAG = "gate-v1-fresh16-inventory-repair-v1"

REPAIR_SOURCE_A_PATHS = (
    CANONICAL_CONFIG_PATH,
    "code/causalcache/gate_v1_fresh16_inventory_repair_contract.py",
    "code/scripts/validate_gate_v1_fresh16_inventory_repair_contract.py",
    "code/tests/test_gate_v1_fresh16_inventory_repair_contract.py",
    "code/causalcache/gate_v1_fresh16_inventory_repair_runner.py",
    "code/scripts/manage_gate_v1_fresh16_inventory_repair.py",
    "code/tests/test_gate_v1_fresh16_inventory_repair_runner.py",
)

DERIVED_BASE_PATHS = (".gitattributes", "README.md")
DERIVED_HISTORICAL_AUXILIARY_PATHS = (
    "derived/restoration-v2-v1/images-00000-of-00001.tar",
    "derived/restoration-v2-v1/manifest.json",
    "derived/restoration-v2-v1/ocr-records-00000-of-00001.jsonl",
    "derived/restoration-v2-v1/trajectories-00000-of-00001.jsonl",
    "golden/real-screen-v1/images-00000-of-00001.tar",
    "golden/real-screen-v1/manifest.json",
    "golden/real-screen-v1/ocr-records-00000-of-00001.jsonl",
    "runs/restoration-v2-substrate-screening-v1/artifact_manifest.json",
    (
        "runs/restoration-v2-substrate-screening-v1/"
        "restoration-v2-substrate-screening-20260715T182823Z.tar.gz"
    ),
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "parent_contract",
    "source_lineage",
    "inventory_repair",
    "effective_overrides",
    "source_only_contract",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Fresh16InventoryRepairContract:
    """Validated overlay exposed through the parent runner's contract interface."""

    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path
    overlay: Mapping[str, Any]
    parent: Fresh16EvaluationContract

    @property
    def source(self) -> Mapping[str, Any]:
        return _mapping(self.data["source_freeze"], "source freeze")

    @property
    def model(self) -> Mapping[str, Any]:
        return _mapping(self.data["formal_model_input"], "formal model input")

    @property
    def derived(self) -> Mapping[str, Any]:
        return _mapping(self.data["derived_input"], "derived input")

    @property
    def labels(self) -> Mapping[str, Any]:
        return _mapping(self.data["label_input"], "label input")

    @property
    def geometry(self) -> Mapping[str, Any]:
        return _mapping(self.data["fresh_geometry"], "fresh geometry")

    @property
    def output(self) -> Mapping[str, Any]:
        return _mapping(self.data["output_contract"], "output contract")

    @property
    def destination(self) -> Mapping[str, Any]:
        return _mapping(self.data["destination"], "destination")

    @property
    def repair(self) -> Mapping[str, Any]:
        return _mapping(self.overlay["inventory_repair"], "inventory repair")


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


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}")


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
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _safe_data_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    path = PurePosixPath(value)
    if (
        not value
        or not path.is_absolute()
        or path.parts[:2] != ("/", "data")
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(f"{label} must be a canonical absolute path under /data")
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
            raise ValueError(f"{label} must be a regular file")
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
    if fingerprint(before) != fingerprint(after) or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(value, label)


def _binding(
    value: Any,
    *,
    expected_path: str,
    expected_sha256: str,
    expected_size: int,
    label: str,
) -> None:
    record = _mapping(value, label)
    _exact_keys(record, {"path", "sha256", "size_bytes"}, label)
    _safe_relative(record["path"], f"{label} path")
    _equal(
        dict(record),
        {
            "path": expected_path,
            "sha256": expected_sha256,
            "size_bytes": expected_size,
        },
        label,
    )


def _read_bound_repository_file(
    root: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
) -> bytes:
    relative = _safe_relative(binding.get("path"), f"{label} path")
    payload = _regular_file_bytes(root / relative, label=label)
    if (
        len(payload) != binding.get("size_bytes")
        or sha256_bytes(payload) != binding.get("sha256")
    ):
        raise ValueError(f"{label} byte binding drifted")
    return payload


def _validate_source_lineage(config: Mapping[str, Any]) -> None:
    parent = _mapping(config["parent_contract"], "parent contract")
    _equal(
        dict(parent),
        {
            "path": PARENT_CONFIG_PATH,
            "protocol_id": "causalcache_gate_v1_fresh16_evaluation_v1",
            "sha256": PARENT_CONFIG_SHA256,
            "size_bytes": 25334,
        },
        "parent contract binding",
    )
    lineage = _mapping(config["source_lineage"], "source lineage")
    _exact_keys(
        lineage,
        {
            "parent_source_a_git_commit",
            "parent_execution_b_git_commit",
            "failure_evidence_git_commit",
            "parent_execution_b_runner_freeze",
            "failure_summary",
            "failure_readme",
            "repair_source_a_paths",
            "execution_b_runner_freeze",
        },
        "source lineage",
    )
    for key, expected in (
        ("parent_source_a_git_commit", PARENT_SOURCE_A_GIT_COMMIT),
        ("parent_execution_b_git_commit", PARENT_EXECUTION_B_GIT_COMMIT),
        ("failure_evidence_git_commit", FAILURE_EVIDENCE_GIT_COMMIT),
    ):
        value = lineage.get(key)
        if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
            raise ValueError(f"source lineage {key} must be a full Git commit")
        _equal(value, expected, f"source lineage {key}")
    _binding(
        lineage["parent_execution_b_runner_freeze"],
        expected_path=PARENT_RUNNER_FREEZE_PATH,
        expected_sha256=PARENT_RUNNER_FREEZE_SHA256,
        expected_size=20702,
        label="parent runner freeze",
    )
    _binding(
        lineage["failure_summary"],
        expected_path=FAILURE_SUMMARY_PATH,
        expected_sha256=FAILURE_SUMMARY_SHA256,
        expected_size=3430,
        label="failure summary",
    )
    _binding(
        lineage["failure_readme"],
        expected_path=FAILURE_README_PATH,
        expected_sha256=FAILURE_README_SHA256,
        expected_size=1889,
        label="failure README",
    )
    paths = tuple(
        _safe_relative(path, "repair Source-A path")
        for path in _sequence(
            lineage["repair_source_a_paths"], "repair Source-A paths"
        )
    )
    _equal(paths, REPAIR_SOURCE_A_PATHS, "repair Source-A path inventory")
    runner = _mapping(lineage["execution_b_runner_freeze"], "runner freeze")
    _equal(
        dict(runner),
        {
            "path": RUNNER_FREEZE_B_PATH,
            "must_be_absent_during_source_only_validation": True,
            "only_allowed_execution_b_source_tree_diff": True,
            "direct_single_parent_child_of_source_a": True,
            "bind_required_source_a_paths": True,
            "bind_git_prerequisites": True,
            "bind_source_a_inventory_sha256": True,
            "bind_loaded_module_inventory_sha256": True,
        },
        "repair runner-freeze contract",
    )


def _validate_parent_failure(config: Mapping[str, Any]) -> None:
    repair = _mapping(config["inventory_repair"], "inventory repair")
    _exact_keys(
        repair, {"scope", "parent_failure", "derived_remote_tree"}, "inventory repair"
    )
    _equal(
        dict(_mapping(repair["scope"], "repair scope")),
        {
            "failure_stage": "derived_input_inventory_validation_before_download",
            "metadata_only_inventory_repair": True,
            "consumed_input_bytes_changed": False,
            "scientific_protocol_changed": False,
            "output_target_paths_changed": False,
            "fresh_semantic_access_before_execution_b_allowed": False,
        },
        "inventory-repair scope",
    )
    failure = _mapping(repair["parent_failure"], "parent failure")
    _exact_keys(
        failure,
        {
            "status",
            "source_a_git_commit",
            "execution_b_git_commit",
            "parent_contract_sha256",
            "old_state_namespace",
            "ordered_receipts",
            "expected_absent_successor_state_names",
            "old_artifact_directory",
            "old_artifact_directory_expected_empty",
            "run_evidence",
            "old_runtime_receipt",
            "old_destination",
        },
        "parent failure",
    )
    for key, expected in (
        ("status", "INVALID_PRE_SEMANTIC_INPUT_INVENTORY_FAILURE"),
        ("source_a_git_commit", PARENT_SOURCE_A_GIT_COMMIT),
        ("execution_b_git_commit", PARENT_EXECUTION_B_GIT_COMMIT),
        ("parent_contract_sha256", PARENT_CONFIG_SHA256),
        (
            "old_state_namespace",
            "/data/experiments/causalcache/gate-v1-fresh16-evaluation-v1",
        ),
    ):
        _equal(failure.get(key), expected, f"parent failure {key}")
    _safe_data_path(failure["old_state_namespace"], "old state namespace")
    expected_receipts = [
        {
            "state_name": "runtime_receipts",
            "path": (
                "/data/experiments/causalcache/gate-v1-fresh16-evaluation-v1/"
                "ordered-state-receipts/00-runtime_receipts.json"
            ),
            "mode": 0o600,
            "size_bytes": 2535,
            "sha256": "b27fbf6f80e46fdc7d50d862508783d0bec8e073d662db52ba2b3b979d037c9e",
        },
        {
            "state_name": "global_claim",
            "path": (
                "/data/experiments/causalcache/gate-v1-fresh16-evaluation-v1/"
                "ordered-state-receipts/01-global_claim.json"
            ),
            "mode": 0o600,
            "size_bytes": 858,
            "sha256": "23955e7cd7d81306888a11d3bbecf318339ce77d8943fc243ff239419498d6c5",
        },
    ]
    receipts = [dict(_mapping(item, "old ordered receipt")) for item in _sequence(
        failure["ordered_receipts"], "old ordered receipts"
    )]
    _equal(receipts, expected_receipts, "old ordered receipt set")
    for item in receipts:
        _safe_data_path(item["path"], "old ordered receipt path")
    expected_successors = [
        "transport_verification",
        "label_blind_cpu_completion",
        "checkpoint_replay_completion",
        "policy_worker_even_completion",
        "policy_worker_odd_completion",
        "heuristic_local_seal",
        "label_access_claim",
        "label_cache_completion",
        "remote_base_receipt",
        "payload_commit_receipt",
        "primary_report_completion",
        "report_commit_receipt",
        "completion_staging",
        "final_completion",
    ]
    _equal(
        list(failure["expected_absent_successor_state_names"]),
        expected_successors,
        "old absent successor states",
    )
    _equal(
        failure["old_artifact_directory"],
        "/data/artifacts/causalcache/gate-v1-fresh16-evaluation-v1",
        "old artifact directory",
    )
    _safe_data_path(failure["old_artifact_directory"], "old artifact directory")
    _equal(
        failure["old_artifact_directory_expected_empty"],
        True,
        "old artifact expected-empty claim",
    )
    evidence = _mapping(failure["run_evidence"], "old run evidence")
    _equal(
        dict(evidence),
        {
            "log": {
                "path": "/data/logs/gate-v1-fresh16-evaluation-v1.run.log",
                "mode": 0o600,
                "size_bytes": 1386,
                "sha256": "348d821024c3ab49f6c8d7185f18d10727b6d909aa4eb57c0061d0aa08a0a1dd",
            },
            "started": {
                "path": "/data/logs/gate-v1-fresh16-evaluation-v1.run.started",
                "mode": 0o600,
                "size_bytes": 21,
                "sha256": "f01a0bce707e6ec8c90d708a65fb668e1c6baf8e81567e93f4882f3e415ad7d1",
            },
            "exit": {
                "path": "/data/logs/gate-v1-fresh16-evaluation-v1.run.exit",
                "mode": 0o600,
                "size_bytes": 2,
                "sha256": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
            },
        },
        "old run evidence",
    )
    for record in evidence.values():
        _safe_data_path(_mapping(record, "run evidence record")["path"], "run evidence path")
    receipt = _mapping(failure["old_runtime_receipt"], "old runtime receipt")
    _equal(
        dict(receipt),
        {
            "path": (
                "/data/experiments/causalcache/"
                ".gate-v1-fresh16-evaluation-v1.docker-inspect.json"
            ),
            "mode": 0o600,
            "size_bytes": 18117,
            "sha256": "7eccdc0cf09ec6e8e2d771b6de50793d53011daba28fc24e21d627335a1d3043",
        },
        "old runtime receipt",
    )
    _safe_data_path(receipt["path"], "old runtime receipt path")
    _equal(
        dict(_mapping(failure["old_destination"], "old destination")),
        {
            "repo": "gavinlaw/causalcache-gate-v1-fresh16-evaluation-mobile",
            "repo_type": "dataset",
            "tag": "gate-v1-fresh16-evaluation-v1",
            "expected_absent": True,
            "remote_mutation_count": 0,
        },
        "old destination",
    )


def _validate_derived_inventory(
    config: Mapping[str, Any], parent: Fresh16EvaluationContract
) -> None:
    tree = _mapping(
        _mapping(config["inventory_repair"], "inventory repair")[
            "derived_remote_tree"
        ],
        "derived remote tree",
    )
    _exact_keys(
        tree,
        {
            "repo",
            "repo_type",
            "immutable_revision",
            "exact_remote_path_count",
            "full_inventory_sha256",
            "require_exact_no_extra_paths",
            "download_only_consumed_paths",
            "consumed_files",
            "base_paths",
            "historical_auxiliary_paths",
            "full_inventory_paths",
        },
        "derived remote tree",
    )
    for key, expected in (
        ("repo", DERIVED_REPO),
        ("repo_type", "dataset"),
        ("immutable_revision", DERIVED_REVISION),
        ("exact_remote_path_count", 15),
        ("full_inventory_sha256", DERIVED_FULL_INVENTORY_SHA256),
        ("require_exact_no_extra_paths", True),
        ("download_only_consumed_paths", True),
    ):
        _equal(tree.get(key), expected, f"derived remote tree {key}")
    consumed = tuple(
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in _sequence(tree["consumed_files"], "consumed derived files")
    )
    parent_consumed = tuple(
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in parent.derived["files"]
    )
    _equal(consumed, parent_consumed, "derived consumed byte bindings")
    base = tuple(
        _safe_relative(path, "derived base path")
        for path in _sequence(tree["base_paths"], "derived base paths")
    )
    auxiliary = tuple(
        _safe_relative(path, "derived historical auxiliary path")
        for path in _sequence(
            tree["historical_auxiliary_paths"],
            "derived historical auxiliary paths",
        )
    )
    full = tuple(
        _safe_relative(path, "derived full-inventory path")
        for path in _sequence(tree["full_inventory_paths"], "derived full inventory")
    )
    _equal(base, DERIVED_BASE_PATHS, "derived base path inventory")
    _equal(
        auxiliary,
        DERIVED_HISTORICAL_AUXILIARY_PATHS,
        "derived historical auxiliary inventory",
    )
    consumed_paths = tuple(record["path"] for record in consumed)
    if set(consumed_paths) & set(base) or set(consumed_paths) & set(auxiliary) or set(base) & set(auxiliary):
        raise ValueError("derived consumed/base/auxiliary inventories overlap")
    expected_full = tuple(sorted((*consumed_paths, *base, *auxiliary)))
    if (
        full != expected_full
        or len(full) != 15
        or len(set(full)) != 15
        or sha256_bytes(canonical_json_bytes(list(full)))
        != DERIVED_FULL_INVENTORY_SHA256
    ):
        raise ValueError("derived exact full-15 path inventory drifted")


def _validate_effective_overrides(config: Mapping[str, Any]) -> None:
    overrides = _mapping(config["effective_overrides"], "effective overrides")
    _equal(
        dict(overrides),
        {
            "execution_namespace": NEW_EXECUTION_NAMESPACE,
            "artifact_directory": NEW_ARTIFACT_DIRECTORY,
            "docker_inspect_receipt": {
                "status": (
                    "CAPTURED_GATE_V1_FRESH16_INVENTORY_REPAIR_"
                    "DOCKER_INSPECT_V1"
                ),
                "path": NEW_RUNTIME_RECEIPT_PATH,
            },
            "destination": {
                "repo": NEW_DESTINATION_REPO,
                "tag": NEW_DESTINATION_TAG,
                "tag_message": (
                    "Freeze CausalCache gate v1 fresh16 inventory-repair evaluation"
                ),
            },
        },
        "effective overrides",
    )
    _safe_data_path(overrides["artifact_directory"], "new artifact directory")
    _safe_data_path(
        _mapping(overrides["docker_inspect_receipt"], "receipt override")["path"],
        "new runtime receipt",
    )
    _equal(
        dict(_mapping(config["source_only_contract"], "source-only contract")),
        {
            "network_call_count": 0,
            "file_write_count": 0,
            "torch_import_count": 0,
            "fresh16_semantic_access_count": 0,
            "model_load_count": 0,
            "model_forward_count": 0,
            "primary_report_count": 0,
            "hf_mutation_count": 0,
            "evaluation_executed": False,
            "execution_authorized": False,
        },
        "repair source-only contract",
    )


def validate_fresh16_inventory_repair_contract_data(
    config: Mapping[str, Any],
    *,
    parent: Fresh16EvaluationContract,
) -> Mapping[str, Any]:
    _exact_keys(config, _TOP_LEVEL_KEYS, "fresh16 inventory-repair config")
    _equal(config.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(config.get("protocol_id"), PROTOCOL_ID, "overlay protocol ID")
    _equal(config.get("status"), SOURCE_STATUS, "overlay source status")
    _validate_source_lineage(config)
    _validate_parent_failure(config)
    _validate_derived_inventory(config, parent)
    _validate_effective_overrides(config)
    return config


def _effective_contract_data(
    parent: Fresh16EvaluationContract,
    overlay: Mapping[str, Any],
) -> Mapping[str, Any]:
    effective = copy.deepcopy(dict(parent.data))
    lineage = _mapping(overlay["source_lineage"], "source lineage")
    source = _mapping(effective["source_freeze"], "effective source freeze")
    additional_paths = (
        PARENT_RUNNER_FREEZE_PATH,
        FAILURE_SUMMARY_PATH,
        FAILURE_README_PATH,
        *REPAIR_SOURCE_A_PATHS,
    )
    parent_paths = tuple(source["required_source_a_paths"])
    if set(parent_paths) & set(additional_paths):
        raise ValueError("repair Source-A additions overlap parent Source-A paths")
    source["required_source_a_paths"] = [*parent_paths, *additional_paths]
    parent_prerequisites = list(source["git_prerequisites"])
    source["git_prerequisites"] = [
        *parent_prerequisites,
        dict(lineage["parent_execution_b_runner_freeze"]),
        dict(lineage["failure_summary"]),
        dict(lineage["failure_readme"]),
    ]
    source["execution_b_runner_freeze"] = dict(
        lineage["execution_b_runner_freeze"]
    )

    tree = _mapping(
        _mapping(overlay["inventory_repair"], "inventory repair")[
            "derived_remote_tree"
        ],
        "derived remote tree",
    )
    derived = _mapping(effective["derived_input"], "effective derived input")
    derived["remote_tree_base_paths"] = list(tree["base_paths"])
    derived["remote_tree_auxiliary_paths"] = list(
        tree["historical_auxiliary_paths"]
    )
    derived["remote_tree_full_inventory_paths"] = list(
        tree["full_inventory_paths"]
    )
    derived["remote_tree_full_inventory_sha256"] = tree[
        "full_inventory_sha256"
    ]

    overrides = _mapping(overlay["effective_overrides"], "effective overrides")
    local = _mapping(
        effective["local_first_state_machine"], "effective local state machine"
    )
    local["execution_namespace"] = overrides["execution_namespace"]
    local["artifact_directory"] = overrides["artifact_directory"]
    receipt = _mapping(
        _mapping(effective["runtime_contract"], "effective runtime contract")[
            "docker_inspect_receipt"
        ],
        "effective runtime receipt",
    )
    receipt.update(dict(overrides["docker_inspect_receipt"]))
    destination = _mapping(effective["destination"], "effective destination")
    destination.update(dict(overrides["destination"]))
    return effective


def _prove_effective_delta(
    parent: Fresh16EvaluationContract,
    effective: Mapping[str, Any],
    overlay: Mapping[str, Any],
) -> None:
    expected = _effective_contract_data(parent, overlay)
    if canonical_json_bytes(effective) != canonical_json_bytes(expected):
        raise ValueError("effective inventory-repair contract contains an undeclared delta")
    changed_top = {
        key
        for key in parent.data
        if canonical_json_bytes(parent.data[key])
        != canonical_json_bytes(effective[key])
    }
    _equal(
        changed_top,
        {
            "source_freeze",
            "derived_input",
            "local_first_state_machine",
            "runtime_contract",
            "destination",
        },
        "effective top-level delta",
    )
    _equal(
        effective["output_contract"],
        parent.data["output_contract"],
        "unchanged output target contract",
    )
    _equal(
        effective["evaluation_contract"],
        parent.data["evaluation_contract"],
        "unchanged scientific evaluation contract",
    )
    _equal(
        effective["formal_model_input"],
        parent.data["formal_model_input"],
        "unchanged formal model input",
    )
    _equal(
        effective["label_input"],
        parent.data["label_input"],
        "unchanged label input",
    )


def _validate_bound_failure_summary(
    root: Path,
    overlay: Mapping[str, Any],
) -> None:
    lineage = _mapping(overlay["source_lineage"], "source lineage")
    _read_bound_repository_file(
        root,
        _mapping(lineage["parent_execution_b_runner_freeze"], "old runner binding"),
        label="parent Execution-B runner freeze",
    )
    summary_payload = _read_bound_repository_file(
        root,
        _mapping(lineage["failure_summary"], "failure summary binding"),
        label="fresh16 v1 failure summary",
    )
    _read_bound_repository_file(
        root,
        _mapping(lineage["failure_readme"], "failure README binding"),
        label="fresh16 v1 failure README",
    )
    summary = _strict_json(summary_payload, label="fresh16 v1 failure summary")
    parent_failure = _mapping(
        _mapping(overlay["inventory_repair"], "inventory repair")[
            "parent_failure"
        ],
        "parent failure",
    )
    for key, expected in (
        ("status", parent_failure["status"]),
        ("source_a_git_commit", parent_failure["source_a_git_commit"]),
        ("execution_b_git_commit", parent_failure["execution_b_git_commit"]),
        ("contract_sha256", parent_failure["parent_contract_sha256"]),
    ):
        _equal(summary.get(key), expected, f"failure summary {key}")
    failure = _mapping(summary.get("failure"), "failure summary failure")
    tree = _mapping(
        _mapping(overlay["inventory_repair"], "inventory repair")[
            "derived_remote_tree"
        ],
        "derived remote tree",
    )
    for key, expected in (
        ("stage", "derived_input_inventory_validation_before_download"),
        ("derived_repo", tree["repo"]),
        ("derived_revision", tree["immutable_revision"]),
        ("configured_consumed_path_count", 4),
        ("actual_remote_path_count", 15),
        ("allowed_base_path_count", 2),
        ("unbound_historical_artifact_path_count", 9),
        (
            "unbound_historical_artifact_paths",
            tree["historical_auxiliary_paths"],
        ),
        (
            "run_log_sha256",
            parent_failure["run_evidence"]["log"]["sha256"],
        ),
    ):
        _equal(failure.get(key), expected, f"failure summary {key}")
    retained = _mapping(
        summary.get("retained_local_evidence"), "retained local evidence"
    )
    _equal(
        retained.get("state_namespace"),
        parent_failure["old_state_namespace"],
        "retained old state namespace",
    )
    _equal(
        retained.get("ordered_state_names"),
        [item["state_name"] for item in parent_failure["ordered_receipts"]],
        "retained ordered states",
    )
    _equal(
        retained.get("runtime_receipt_sha256"),
        parent_failure["ordered_receipts"][0]["sha256"],
        "retained runtime receipt SHA",
    )
    _equal(
        retained.get("global_claim_sha256"),
        parent_failure["ordered_receipts"][1]["sha256"],
        "retained global claim SHA",
    )
    remote = _mapping(summary.get("remote"), "failure summary remote")
    _equal(remote.get("planned_repo"), parent_failure["old_destination"]["repo"], "old destination repo")
    _equal(remote.get("planned_tag"), parent_failure["old_destination"]["tag"], "old destination tag")
    _equal(remote.get("destination_exists_after_failure"), False, "old destination absence")
    _equal(remote.get("remote_mutation_count"), 0, "old remote mutation count")


def _canonical_config(root: Path, supplied: str | Path) -> Path:
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = root / candidate
    canonical = (root / CANONICAL_CONFIG_PATH).resolve()
    if candidate.resolve() != canonical:
        raise ValueError("fresh16 inventory-repair config must use its canonical path")
    return canonical


def load_frozen_fresh16_inventory_repair_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> Fresh16InventoryRepairContract:
    root = Path(repository_root).resolve()
    source_path = _canonical_config(root, path)
    payload = _regular_file_bytes(source_path, label="fresh16 inventory-repair config")
    _equal(sha256_bytes(payload), FROZEN_CONFIG_SHA256, "frozen repair config SHA256")
    overlay = _strict_json(payload, label="fresh16 inventory-repair config")
    parent = load_frozen_fresh16_evaluation_contract(
        PARENT_CONFIG_PATH,
        repository_root=root,
    )
    validate_fresh16_inventory_repair_contract_data(overlay, parent=parent)
    _validate_bound_failure_summary(root, overlay)
    effective = _effective_contract_data(parent, overlay)
    _prove_effective_delta(parent, effective, overlay)
    return Fresh16InventoryRepairContract(
        data=effective,
        sha256=FROZEN_CONFIG_SHA256,
        repository_root=root,
        source_path=source_path,
        overlay=overlay,
        parent=parent,
    )


def _source_inventory(
    contract: Fresh16InventoryRepairContract,
) -> tuple[Mapping[str, Any], ...]:
    result = []
    for raw in _sequence(
        contract.source["required_source_a_paths"], "repair Source-A paths"
    ):
        relative = _safe_relative(raw, "repair Source-A path")
        payload = _regular_file_bytes(
            contract.repository_root / relative,
            label=f"repair Source-A path {relative}",
        )
        result.append(
            {
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    return tuple(result)


def validate_fresh16_inventory_repair_source_only_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> Mapping[str, Any]:
    before_modules = frozenset(sys.modules)
    contract = load_frozen_fresh16_inventory_repair_contract(
        path,
        repository_root=repository_root,
    )
    if os.path.lexists(contract.repository_root / RUNNER_FREEZE_B_PATH):
        raise ValueError("fresh16 inventory-repair runner-freeze B must be absent in Source-A")
    inventory = _source_inventory(contract)
    newly_loaded = frozenset(sys.modules) - before_modules
    prohibited = sorted(
        name
        for name in newly_loaded
        if name == "torch"
        or name.startswith("torch.")
        or name == "huggingface_hub"
        or name.startswith("huggingface_hub.")
    )
    if prohibited:
        raise ValueError("repair Source-A validation imported torch or Hugging Face")
    source_only = dict(
        _mapping(contract.overlay["source_only_contract"], "source-only contract")
    )
    parent_zero = dict(
        _mapping(
            contract.data["source_only_operation_contract"],
            "parent source-only operation contract",
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": VALIDATION_STATUS,
        "config_path": CANONICAL_CONFIG_PATH,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "source_a_path_total": len(inventory),
        "source_a_inventory_sha256": sha256_bytes(
            canonical_json_bytes(inventory)
        ),
        "runner_freeze_b_path": RUNNER_FREEZE_B_PATH,
        "runner_freeze_b_present": False,
        "derived_remote_path_count": 15,
        "derived_full_inventory_sha256": DERIVED_FULL_INVENTORY_SHA256,
        **parent_zero,
        **source_only,
        "gate_trained": True,
        "fresh16_access_authorized": False,
        "label_access_authorized": False,
        "legacy_dev5_access_authorized": False,
        "confirm20_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }
