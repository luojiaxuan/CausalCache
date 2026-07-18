"""Fail-closed Source-A overlay for the fresh-16 claim serialization repair."""

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

from causalcache.gate_v1_fresh16_evaluation_contract import canonical_json_bytes
from causalcache.gate_v1_fresh16_inventory_repair_contract import (
    CANONICAL_CONFIG_PATH as PARENT_CONFIG_PATH,
    FROZEN_CONFIG_SHA256 as PARENT_CONFIG_SHA256,
    Fresh16InventoryRepairContract,
    load_frozen_fresh16_inventory_repair_contract,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_fresh16_claim_serialization_repair_v1"
SOURCE_STATUS = (
    "source_only_frozen_before_fresh16_claim_serialization_repair_semantic_access"
)
VALIDATION_STATUS = (
    "VALID_SOURCE_ONLY_GATE_V1_FRESH16_CLAIM_SERIALIZATION_REPAIR_V1"
)
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_v1.json"
)
RUNNER_FREEZE_B_PATH = (
    "code/configs/causalcache_gate_v1_fresh16_claim_serialization_repair_runner_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "3979573be235d630ee2f46dc23be8747a843190c9b57ee81e1b3a17b4416d8c7"
)

PARENT_SOURCE_A_GIT_COMMIT = "6fb3e868e293bce191ce30a5c6a15ecc544c4591"
PARENT_EXECUTION_B_GIT_COMMIT = "c22734ffc85935882f57ddb081c9194d6dae92d0"
FAILURE_EVIDENCE_GIT_COMMIT = "7fbfe1b8314ea61d7d646a47be902fdf1c6d4af9"
PARENT_RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_gate_v1_fresh16_inventory_repair_runner_v1.json"
)
PARENT_RUNNER_FREEZE_SHA256 = (
    "10856ba73e2fe2a6ccfe313718f4ddde7a28a5ca24be3747fed25205e60b32c5"
)
FAILURE_SUMMARY_PATH = (
    "data/results/gate_v1_fresh16_inventory_repair_v1_attempt/summary.json"
)
FAILURE_SUMMARY_SHA256 = (
    "eebbb9be9d88b98607651246165f61c5521bc30f8a360d6713e7d9ec431f0864"
)
FAILURE_README_PATH = (
    "data/results/gate_v1_fresh16_inventory_repair_v1_attempt/README.md"
)
FAILURE_README_SHA256 = (
    "1d2f434143c6d1b775fc9585e2e64de4882328e2306b5d187cf134f695ba3aaa"
)

NEW_EXECUTION_NAMESPACE = (
    "gate-v1-fresh16-evaluation-claim-serialization-repair-v1"
)
NEW_ARTIFACT_DIRECTORY = (
    "/data/artifacts/causalcache/"
    "gate-v1-fresh16-evaluation-claim-serialization-repair-v1"
)
NEW_RUNTIME_RECEIPT_PATH = (
    "/data/experiments/causalcache/"
    ".gate-v1-fresh16-evaluation-claim-serialization-repair-v1.docker-inspect.json"
)
NEW_DESTINATION_REPO = (
    "gavinlaw/causalcache-gate-v1-fresh16-claim-serialization-repair-mobile"
)
NEW_DESTINATION_TAG = "gate-v1-fresh16-claim-serialization-repair-v1"

CLAIM_SERIALIZATION_FIX_PATHS = (
    "code/causalcache/gate_v1_fresh16_evaluation_runner.py",
    "code/tests/test_gate_v1_fresh16_evaluation_runner.py",
)
REPAIR_SOURCE_A_PATHS = (
    CANONICAL_CONFIG_PATH,
    "code/causalcache/gate_v1_fresh16_claim_serialization_repair_contract.py",
    "code/scripts/validate_gate_v1_fresh16_claim_serialization_repair_contract.py",
    "code/tests/test_gate_v1_fresh16_claim_serialization_repair_contract.py",
    "code/causalcache/gate_v1_fresh16_claim_serialization_repair_runner.py",
    "code/scripts/manage_gate_v1_fresh16_claim_serialization_repair.py",
    "code/tests/test_gate_v1_fresh16_claim_serialization_repair_runner.py",
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "parent_contract",
    "source_lineage",
    "claim_serialization_repair",
    "effective_overrides",
    "source_only_contract",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Fresh16ClaimSerializationRepairContract:
    """Validated claim-repair overlay with the inventory repair as parent."""

    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path
    overlay: Mapping[str, Any]
    parent: Fresh16InventoryRepairContract

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
        return _mapping(
            self.overlay["claim_serialization_repair"],
            "claim serialization repair",
        )


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
    root: Path, binding: Mapping[str, Any], *, label: str
) -> bytes:
    relative = _safe_relative(binding.get("path"), f"{label} path")
    payload = _regular_file_bytes(root / relative, label=label)
    if (
        len(payload) != binding.get("size_bytes")
        or sha256_bytes(payload) != binding.get("sha256")
    ):
        raise ValueError(f"{label} byte binding drifted")
    return payload


def _validate_parent_and_lineage(config: Mapping[str, Any]) -> None:
    parent = _mapping(config["parent_contract"], "parent contract")
    _exact_keys(
        parent,
        {
            "path",
            "protocol_id",
            "sha256",
            "size_bytes",
            "source_a_git_commit",
            "execution_b_git_commit",
            "execution_b_runner_freeze",
        },
        "parent contract",
    )
    _equal(parent["path"], PARENT_CONFIG_PATH, "parent contract path")
    _equal(
        parent["protocol_id"],
        "causalcache_gate_v1_fresh16_inventory_repair_v1",
        "parent protocol",
    )
    _equal(parent["sha256"], PARENT_CONFIG_SHA256, "parent config SHA")
    _equal(parent["size_bytes"], 10064, "parent config size")
    _equal(
        parent["source_a_git_commit"],
        PARENT_SOURCE_A_GIT_COMMIT,
        "parent Source-A commit",
    )
    _equal(
        parent["execution_b_git_commit"],
        PARENT_EXECUTION_B_GIT_COMMIT,
        "parent Execution-B commit",
    )
    _binding(
        parent["execution_b_runner_freeze"],
        expected_path=PARENT_RUNNER_FREEZE_PATH,
        expected_sha256=PARENT_RUNNER_FREEZE_SHA256,
        expected_size=24825,
        label="parent Execution-B runner freeze",
    )

    lineage = _mapping(config["source_lineage"], "source lineage")
    _exact_keys(
        lineage,
        {
            "failure_evidence_git_commit",
            "failure_summary",
            "failure_readme",
            "claim_serialization_fix_paths",
            "repair_source_a_paths",
            "execution_b_runner_freeze",
        },
        "source lineage",
    )
    commit = lineage["failure_evidence_git_commit"]
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("failure evidence commit must be a full Git commit")
    _equal(commit, FAILURE_EVIDENCE_GIT_COMMIT, "failure evidence commit")
    _binding(
        lineage["failure_summary"],
        expected_path=FAILURE_SUMMARY_PATH,
        expected_sha256=FAILURE_SUMMARY_SHA256,
        expected_size=5666,
        label="failure summary",
    )
    _binding(
        lineage["failure_readme"],
        expected_path=FAILURE_README_PATH,
        expected_sha256=FAILURE_README_SHA256,
        expected_size=2539,
        label="failure README",
    )
    fix_paths = tuple(
        _safe_relative(item, "claim serialization fix path")
        for item in _sequence(
            lineage["claim_serialization_fix_paths"],
            "claim serialization fix paths",
        )
    )
    source_paths = tuple(
        _safe_relative(item, "repair Source-A path")
        for item in _sequence(lineage["repair_source_a_paths"], "repair Source-A paths")
    )
    _equal(fix_paths, CLAIM_SERIALIZATION_FIX_PATHS, "claim fix path inventory")
    _equal(source_paths, REPAIR_SOURCE_A_PATHS, "repair Source-A path inventory")
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
        "claim-repair runner-freeze contract",
    )


def _validate_scope(config: Mapping[str, Any]) -> None:
    repair = _mapping(
        config["claim_serialization_repair"], "claim serialization repair"
    )
    _exact_keys(repair, {"scope", "parent_failure"}, "claim serialization repair")
    _equal(
        dict(_mapping(repair["scope"], "claim serialization repair scope")),
        {
            "failure_stage": "after_heuristic_local_seal_before_label_access_claim_write",
            "failure_type": "TypeError",
            "failure_message": "Object of type mappingproxy is not JSON serializable",
            "failing_expression": "pretty_json_bytes(claim.claim)",
            "root_cause": (
                "LabelAccessClaim.claim returned a non-JSON-serializable "
                "MappingProxyType"
            ),
            "fix_contract": (
                "return a canonical-JSON deep snapshot from LabelAccessClaim.claim"
            ),
            "canonical_claim_bytes_changed": False,
            "consumed_input_bytes_changed": False,
            "scientific_protocol_changed": False,
            "derived_inventory_changed": False,
            "model_input_changed": False,
            "label_input_changed": False,
            "geometry_changed": False,
            "evaluation_contract_changed": False,
            "output_target_paths_changed": False,
            "thresholds_changed": False,
            "old_run_continuation_allowed": False,
            "fresh_semantic_access_before_execution_b_allowed": False,
        },
        "claim serialization repair scope",
    )


def _identity_record(record: Any, *, label: str) -> None:
    value = _mapping(record, label)
    if set(value) - {"path", "mode", "size_bytes", "sha256", "state_name"}:
        raise ValueError(f"{label} contains unsupported keys")
    _safe_data_path(value.get("path"), f"{label} path")
    if (
        value.get("mode") != 0o600
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] < 0
        or not isinstance(value.get("sha256"), str)
        or _SHA256.fullmatch(value["sha256"]) is None
    ):
        raise ValueError(f"{label} identity is malformed")


def _validate_parent_failure(config: Mapping[str, Any]) -> None:
    failure = _mapping(
        _mapping(
            config["claim_serialization_repair"], "claim serialization repair"
        )["parent_failure"],
        "parent failure",
    )
    expected_keys = {
        "status",
        "source_a_git_commit",
        "execution_b_git_commit",
        "parent_contract_sha256",
        "old_state_namespace",
        "expected_state_root_entries",
        "ordered_receipts",
        "expected_absent_successor_state_names",
        "heuristic_local_seal",
        "expected_absent_standalone_files",
        "artifact_tree",
        "run_evidence",
        "old_runtime_receipt",
        "old_destinations",
        "fresh_label_semantic_decode_count",
        "label_access_claim_count",
        "primary_report_count",
        "hf_mutation_count",
    }
    _exact_keys(failure, expected_keys, "parent failure")
    for key, expected in (
        ("status", "INVALID_PRE_LABEL_CLAIM_SERIALIZATION_FAILURE"),
        ("source_a_git_commit", PARENT_SOURCE_A_GIT_COMMIT),
        ("execution_b_git_commit", PARENT_EXECUTION_B_GIT_COMMIT),
        ("parent_contract_sha256", PARENT_CONFIG_SHA256),
        (
            "old_state_namespace",
            "/data/experiments/causalcache/"
            "gate-v1-fresh16-evaluation-inventory-repair-v1",
        ),
        (
            "expected_state_root_entries",
            ["heuristic-local-seal.json", "ordered-state-receipts"],
        ),
        ("fresh_label_semantic_decode_count", 0),
        ("label_access_claim_count", 0),
        ("primary_report_count", 0),
        ("hf_mutation_count", 0),
    ):
        _equal(failure.get(key), expected, f"parent failure {key}")
    _safe_data_path(failure["old_state_namespace"], "old state namespace")

    state_names = (
        "runtime_receipts",
        "global_claim",
        "transport_verification",
        "label_blind_cpu_completion",
        "checkpoint_replay_completion",
        "policy_worker_even_completion",
        "policy_worker_odd_completion",
        "heuristic_local_seal",
    )
    expected_sizes = (6790, 858, 1581, 821, 1035, 715, 713, 645)
    expected_hashes = (
        "1febc80296005bcf107e3916c654d2915f1333f4e97cc7085291fede2f91bb03",
        "d5d7045383b2579a2d13a52ce8290f60207919c98ecf639d9ce22c13bc95f587",
        "bdd24f769bbcc399a0abec7b9ac8617e77ca66a0a6e04712480be12c072db607",
        "8038aa16e0e1f0c909a70c69ac47d5f953e7e499e8d397efc04fed97cfb3e52c",
        "3814f9961c498cf724bac7041c5ad400716029da223c5cf5bf717544d4f5ff3c",
        "0368cc17ea78eb0e3ddadc7c1b44bed17c0671a80668993e80f1ac5d3486788e",
        "7bb9f5f402a90fe601c0bef1eb4461e7ddc58de84cbd77819c843e9b604ae759",
        "f8202b8f0dd0945368c1c77bbf75c910d9244923493ea79813978796718cee13",
    )
    receipts = tuple(
        _mapping(item, "old ordered receipt")
        for item in _sequence(failure["ordered_receipts"], "old ordered receipts")
    )
    if len(receipts) != 8:
        raise ValueError("parent failure must bind exactly eight ordered receipts")
    namespace = failure["old_state_namespace"]
    for ordinal, (record, name, size, digest) in enumerate(
        zip(receipts, state_names, expected_sizes, expected_hashes, strict=True)
    ):
        _identity_record(record, label=f"old ordered receipt {ordinal}")
        _equal(
            dict(record),
            {
                "state_name": name,
                "path": (
                    f"{namespace}/ordered-state-receipts/"
                    f"{ordinal:02d}-{name}.json"
                ),
                "mode": 0o600,
                "size_bytes": size,
                "sha256": digest,
            },
            f"old ordered receipt {ordinal}",
        )
    _equal(
        list(failure["expected_absent_successor_state_names"]),
        [
            "label_access_claim",
            "label_cache_completion",
            "remote_base_receipt",
            "payload_commit_receipt",
            "primary_report_completion",
            "report_commit_receipt",
            "completion_staging",
            "final_completion",
        ],
        "old absent successor states",
    )
    seal = _mapping(failure["heuristic_local_seal"], "heuristic local seal")
    _identity_record(seal, label="heuristic local seal")
    _equal(
        dict(seal),
        {
            "path": f"{namespace}/heuristic-local-seal.json",
            "mode": 0o600,
            "size_bytes": 1783,
            "sha256": "bd11c4a8220df2095023bd34d0ef7769fec4c4ee78c38df702099cec82142f0e",
        },
        "heuristic local seal",
    )
    absent = list(failure["expected_absent_standalone_files"])
    expected_absent = [
        f"{namespace}/label-access-claim.json",
        f"{namespace}/remote-base-pre-mutation.json",
        f"{namespace}/remote-base-receipt.json",
        f"{namespace}/final-completion.json",
    ]
    _equal(absent, expected_absent, "old absent standalone files")
    for path in absent:
        _safe_data_path(path, "old absent standalone path")

    artifact = _mapping(failure["artifact_tree"], "old artifact tree")
    _equal(
        dict(artifact),
        {
            "root": (
                "/data/artifacts/causalcache/"
                "gate-v1-fresh16-evaluation-inventory-repair-v1"
            ),
            "top_level_entries": ["fresh16-eval", "selected-images"],
            "file_count": 88,
            "total_bytes": 51508707,
            "canonical_inventory_json_size_bytes": 15551,
            "canonical_inventory_sha256": (
                "5443db6df16234c29b32501bc9f766670d428141fb02c4a665be936e1ca2587c"
            ),
        },
        "old artifact tree",
    )
    _safe_data_path(artifact["root"], "old artifact root")

    run = _mapping(failure["run_evidence"], "old run evidence")
    _exact_keys(run, {"log", "started", "exit"}, "old run evidence")
    expected_run = {
        "log": (
            "/data/logs/gate-v1-fresh16-evaluation-inventory-repair-v1.run.log",
            2214,
            "a246ab63e3946c37cbc2f9e2b90c3b418acecf24bad93e0fdfc253d51357afda",
        ),
        "started": (
            "/data/logs/gate-v1-fresh16-evaluation-inventory-repair-v1.run.started",
            21,
            "3fe1edd98f9fe4f79324c7ea2fbb95f5df6c0eea96e9dd40be69b47bf774ad58",
        ),
        "exit": (
            "/data/logs/gate-v1-fresh16-evaluation-inventory-repair-v1.run.exit",
            2,
            "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
        ),
    }
    for name, (path, size, digest) in expected_run.items():
        record = _mapping(run[name], f"old run {name}")
        _identity_record(record, label=f"old run {name}")
        _equal(record["path"], path, f"old run {name} path")
        _equal(record["size_bytes"], size, f"old run {name} size")
        _equal(record["sha256"], digest, f"old run {name} SHA")
    runtime = _mapping(failure["old_runtime_receipt"], "old runtime receipt")
    _identity_record(runtime, label="old runtime receipt")
    _equal(
        runtime["path"],
        "/data/experiments/causalcache/"
        ".gate-v1-fresh16-evaluation-inventory-repair-v1.docker-inspect.json",
        "old runtime receipt path",
    )
    _equal(runtime["size_bytes"], 21084, "old runtime receipt size")
    _equal(
        runtime["sha256"],
        "07df8a6d5714e770089ca30285d4bc3a6526eb9ad023a93d8b9f181f1d12a6f4",
        "old runtime receipt SHA",
    )
    destinations = [
        dict(_mapping(item, "old destination"))
        for item in _sequence(failure["old_destinations"], "old destinations")
    ]
    _equal(
        destinations,
        [
            {
                "repo": "gavinlaw/causalcache-gate-v1-fresh16-evaluation-mobile",
                "repo_type": "dataset",
                "tag": "gate-v1-fresh16-evaluation-v1",
                "expected_absent": True,
                "remote_mutation_count": 0,
            },
            {
                "repo": "gavinlaw/causalcache-gate-v1-fresh16-inventory-repair-mobile",
                "repo_type": "dataset",
                "tag": "gate-v1-fresh16-inventory-repair-v1",
                "expected_absent": True,
                "remote_mutation_count": 0,
            },
        ],
        "old destination inventory",
    )


def _validate_effective_overrides(config: Mapping[str, Any]) -> None:
    overrides = _mapping(config["effective_overrides"], "effective overrides")
    _equal(
        dict(overrides),
        {
            "execution_namespace": NEW_EXECUTION_NAMESPACE,
            "artifact_directory": NEW_ARTIFACT_DIRECTORY,
            "docker_inspect_receipt": {
                "status": (
                    "CAPTURED_GATE_V1_FRESH16_CLAIM_SERIALIZATION_REPAIR_"
                    "DOCKER_INSPECT_V1"
                ),
                "path": NEW_RUNTIME_RECEIPT_PATH,
            },
            "destination": {
                "repo": NEW_DESTINATION_REPO,
                "tag": NEW_DESTINATION_TAG,
                "tag_message": (
                    "Freeze CausalCache gate v1 fresh16 "
                    "claim-serialization-repair evaluation"
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
            "label_semantic_decode_count": 0,
            "model_load_count": 0,
            "model_forward_count": 0,
            "primary_report_count": 0,
            "hf_mutation_count": 0,
            "evaluation_executed": False,
            "execution_authorized": False,
        },
        "claim-repair source-only contract",
    )


def validate_fresh16_claim_serialization_repair_contract_data(
    config: Mapping[str, Any],
    *,
    parent: Fresh16InventoryRepairContract,
) -> Mapping[str, Any]:
    _exact_keys(config, _TOP_LEVEL_KEYS, "fresh16 claim-repair config")
    _equal(config.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(config.get("protocol_id"), PROTOCOL_ID, "overlay protocol ID")
    _equal(config.get("status"), SOURCE_STATUS, "overlay source status")
    _validate_parent_and_lineage(config)
    _validate_scope(config)
    _validate_parent_failure(config)
    _validate_effective_overrides(config)
    if parent.sha256 != PARENT_CONFIG_SHA256:
        raise ValueError("loaded inventory-repair parent identity drifted")
    return config


def _effective_contract_data(
    parent: Fresh16InventoryRepairContract,
    overlay: Mapping[str, Any],
) -> Mapping[str, Any]:
    effective = copy.deepcopy(dict(parent.data))
    lineage = _mapping(overlay["source_lineage"], "source lineage")
    parent_binding = _mapping(overlay["parent_contract"], "parent contract")
    source = _mapping(effective["source_freeze"], "effective source freeze")
    additions = (
        PARENT_RUNNER_FREEZE_PATH,
        FAILURE_SUMMARY_PATH,
        FAILURE_README_PATH,
        *REPAIR_SOURCE_A_PATHS,
    )
    parent_paths = tuple(source["required_source_a_paths"])
    if set(parent_paths) & set(additions):
        raise ValueError("claim-repair Source-A additions overlap parent paths")
    if not set(CLAIM_SERIALIZATION_FIX_PATHS).issubset(parent_paths):
        raise ValueError("claim serialization fix paths escaped parent source inventory")
    source["required_source_a_paths"] = [*parent_paths, *additions]
    source["git_prerequisites"] = [
        *list(source["git_prerequisites"]),
        dict(parent_binding["execution_b_runner_freeze"]),
        dict(lineage["failure_summary"]),
        dict(lineage["failure_readme"]),
    ]
    source["execution_b_runner_freeze"] = dict(
        lineage["execution_b_runner_freeze"]
    )

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
    parent: Fresh16InventoryRepairContract,
    effective: Mapping[str, Any],
    overlay: Mapping[str, Any],
) -> None:
    expected = _effective_contract_data(parent, overlay)
    if canonical_json_bytes(effective) != canonical_json_bytes(expected):
        raise ValueError("effective claim-repair contract has an undeclared delta")
    changed = {
        key
        for key in parent.data
        if canonical_json_bytes(parent.data[key])
        != canonical_json_bytes(effective[key])
    }
    _equal(
        changed,
        {
            "source_freeze",
            "local_first_state_machine",
            "runtime_contract",
            "destination",
        },
        "effective top-level delta",
    )
    for key, label in (
        ("derived_input", "derived full-15 input"),
        ("formal_model_input", "formal model input"),
        ("label_input", "label input"),
        ("fresh_geometry", "fresh geometry"),
        ("evaluation_contract", "evaluation contract"),
        ("output_contract", "output contract"),
    ):
        if canonical_json_bytes(effective[key]) != canonical_json_bytes(parent.data[key]):
            raise ValueError(f"{label} changed in claim serialization repair")


def _validate_bound_failure_summary(root: Path, overlay: Mapping[str, Any]) -> None:
    parent = _mapping(overlay["parent_contract"], "parent contract")
    lineage = _mapping(overlay["source_lineage"], "source lineage")
    _read_bound_repository_file(
        root,
        _mapping(parent["execution_b_runner_freeze"], "parent runner binding"),
        label="parent inventory-repair Execution-B runner freeze",
    )
    summary_payload = _read_bound_repository_file(
        root,
        _mapping(lineage["failure_summary"], "failure summary binding"),
        label="inventory-repair failure summary",
    )
    _read_bound_repository_file(
        root,
        _mapping(lineage["failure_readme"], "failure README binding"),
        label="inventory-repair failure README",
    )
    summary = _strict_json(summary_payload, label="inventory-repair failure summary")
    failure = _mapping(
        _mapping(
            overlay["claim_serialization_repair"], "claim serialization repair"
        )["parent_failure"],
        "parent failure",
    )
    for key, expected in (
        ("status", failure["status"]),
        ("source_a_git_commit", failure["source_a_git_commit"]),
        ("execution_b_git_commit", failure["execution_b_git_commit"]),
        ("contract_sha256", failure["parent_contract_sha256"]),
    ):
        _equal(summary.get(key), expected, f"failure summary {key}")
    failure_record = _mapping(summary.get("failure"), "failure summary failure")
    for key, expected in (
        ("exception", "TypeError"),
        ("message", "Object of type mappingproxy is not JSON serializable"),
        ("failing_expression", "pretty_json_bytes(claim.claim)"),
        (
            "stage",
            "after_heuristic_local_seal_before_label_access_claim_write",
        ),
        ("label_access_claim_path_exists", False),
        ("continuation_allowed", False),
    ):
        _equal(failure_record.get(key), expected, f"failure summary failure {key}")
    retained = _mapping(
        summary.get("retained_local_evidence"), "failure retained evidence"
    )
    _equal(
        retained.get("state_namespace"),
        failure["old_state_namespace"],
        "failure retained namespace",
    )
    summary_receipts = retained.get("ordered_receipts")
    if not isinstance(summary_receipts, list) or len(summary_receipts) != 8:
        raise ValueError("failure summary ordered receipt denominator drifted")
    expected_receipts = failure["ordered_receipts"]
    for ordinal, (summary_item, bound) in enumerate(
        zip(summary_receipts, expected_receipts, strict=True)
    ):
        _equal(
            summary_item,
            {
                "ordinal": ordinal,
                "name": bound["state_name"],
                "mode": bound["mode"],
                "size_bytes": bound["size_bytes"],
                "sha256": bound["sha256"],
            },
            f"failure summary ordered receipt {ordinal}",
        )
    seal = _mapping(retained.get("heuristic_local_seal"), "summary heuristic seal")
    for key in ("mode", "size_bytes", "sha256"):
        _equal(seal.get(key), failure["heuristic_local_seal"][key], f"summary seal {key}")
    artifact = _mapping(retained.get("artifact_tree"), "summary artifact tree")
    bound_artifact = failure["artifact_tree"]
    for key, summary_key in (
        ("root", "root"),
        ("top_level_entries", "top_level_entries"),
        ("file_count", "file_count"),
        ("total_bytes", "total_bytes"),
        ("canonical_inventory_json_size_bytes", "canonical_inventory_json_size_bytes"),
        ("canonical_inventory_sha256", "failure_executor_canonical_inventory_sha256"),
    ):
        _equal(artifact.get(summary_key), bound_artifact[key], f"summary artifact {key}")
    for summary_key, failure_key in (
        ("run_log", "log"),
        ("run_started", "started"),
        ("run_exit", "exit"),
    ):
        summary_item = _mapping(retained.get(summary_key), f"summary {summary_key}")
        bound = failure["run_evidence"][failure_key]
        for key in ("mode", "size_bytes", "sha256"):
            _equal(summary_item.get(key), bound[key], f"summary {summary_key} {key}")
    runtime = _mapping(
        retained.get("docker_inspect_receipt"), "summary runtime receipt"
    )
    for key in ("mode", "size_bytes", "sha256"):
        _equal(runtime.get(key), failure["old_runtime_receipt"][key], f"summary runtime {key}")
    counts = _mapping(
        summary.get("forbidden_or_unreached_operation_counts"),
        "failure forbidden counts",
    )
    for key in (
        "fresh_label_semantic_decode",
        "label_access_claim",
        "primary_report",
        "hf_mutation",
    ):
        _equal(counts.get(key), 0, f"failure forbidden count {key}")
    remote = _mapping(summary.get("remote"), "failure remote")
    _equal(
        remote.get("both_planned_repositories_absent_after_failure"),
        True,
        "failure old destination absence",
    )
    _equal(remote.get("remote_mutation_count"), 0, "failure remote mutation count")


def _canonical_config(root: Path, supplied: str | Path) -> Path:
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = root / candidate
    canonical = (root / CANONICAL_CONFIG_PATH).resolve()
    if candidate.resolve() != canonical:
        raise ValueError("fresh16 claim-repair config must use its canonical path")
    return canonical


def load_frozen_fresh16_claim_serialization_repair_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> Fresh16ClaimSerializationRepairContract:
    root = Path(repository_root).resolve()
    source_path = _canonical_config(root, path)
    payload = _regular_file_bytes(source_path, label="fresh16 claim-repair config")
    _equal(sha256_bytes(payload), FROZEN_CONFIG_SHA256, "frozen claim-repair SHA256")
    overlay = _strict_json(payload, label="fresh16 claim-repair config")
    parent = load_frozen_fresh16_inventory_repair_contract(
        PARENT_CONFIG_PATH,
        repository_root=root,
    )
    validate_fresh16_claim_serialization_repair_contract_data(
        overlay,
        parent=parent,
    )
    _validate_bound_failure_summary(root, overlay)
    effective = _effective_contract_data(parent, overlay)
    _prove_effective_delta(parent, effective, overlay)
    return Fresh16ClaimSerializationRepairContract(
        data=effective,
        sha256=FROZEN_CONFIG_SHA256,
        repository_root=root,
        source_path=source_path,
        overlay=overlay,
        parent=parent,
    )


def _source_inventory(
    contract: Fresh16ClaimSerializationRepairContract,
) -> tuple[Mapping[str, Any], ...]:
    result = []
    for raw in _sequence(contract.source["required_source_a_paths"], "Source-A paths"):
        relative = _safe_relative(raw, "Source-A path")
        payload = _regular_file_bytes(
            contract.repository_root / relative,
            label=f"Source-A path {relative}",
        )
        result.append(
            {
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    return tuple(result)


def validate_fresh16_claim_serialization_repair_source_only_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> Mapping[str, Any]:
    before_modules = frozenset(sys.modules)
    contract = load_frozen_fresh16_claim_serialization_repair_contract(
        path,
        repository_root=repository_root,
    )
    if os.path.lexists(contract.repository_root / RUNNER_FREEZE_B_PATH):
        raise ValueError("fresh16 claim-repair runner-freeze B must be absent in Source-A")
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
        raise ValueError("claim-repair Source-A imported torch or Hugging Face")
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
        "source_a_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "runner_freeze_b_path": RUNNER_FREEZE_B_PATH,
        "runner_freeze_b_present": False,
        "derived_remote_path_count": 15,
        "derived_full_inventory_sha256": contract.derived[
            "remote_tree_full_inventory_sha256"
        ],
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


__all__ = [
    "CANONICAL_CONFIG_PATH",
    "FROZEN_CONFIG_SHA256",
    "Fresh16ClaimSerializationRepairContract",
    "RUNNER_FREEZE_B_PATH",
    "load_frozen_fresh16_claim_serialization_repair_contract",
    "validate_fresh16_claim_serialization_repair_contract_data",
    "validate_fresh16_claim_serialization_repair_source_only_contract",
]
