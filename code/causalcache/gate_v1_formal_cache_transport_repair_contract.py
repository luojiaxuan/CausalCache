"""Fail-closed overlay contract for the formal-58 transport-only repair."""

from __future__ import annotations

import copy
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

from causalcache.gate_v1_contract import canonical_json_bytes
from causalcache.gate_v1_formal_cache_contract import (
    CANONICAL_CONFIG_PATH as PARENT_CONFIG_PATH,
    FROZEN_CONFIG_SHA256 as PARENT_CONFIG_SHA256,
    load_frozen_formal_cache_contract,
)


SCHEMA_VERSION = "1.0.0"
OVERLAY_PROTOCOL_ID = "causalcache_gate_v1_formal_cache_transport_repair_v1"
PARENT_PROTOCOL_ID = "causalcache_gate_v1_formal_cache_v1"
PARENT_SOURCE_STATUS = "source_only_frozen_before_formal_cache_execution"
OVERLAY_SOURCE_STATUS = "source_only_frozen_before_formal58_transport_repair_execution"
SOURCE_STATUS = PARENT_SOURCE_STATUS
VALIDATION_STATUS = "VALID_SOURCE_ONLY_GATE_V1_FORMAL58_TRANSPORT_REPAIR_CONTRACT_V1"
FAILED_ATTEMPT_VALIDATION_STATUS = (
    "VALID_FAILED_GATE_V1_FORMAL58_CACHE_ATTEMPT_PRECLAIM_REPAIR_V1"
)
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_gate_v1_formal_cache_transport_repair_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "aaf82fd5e994588bc22f0f745139e349219863eee5fa8cb4cc93c4a9e48e123a"
)
FAILURE_SUMMARY_SHA256 = (
    "a769252c29495c1cfab979ca3c93ea47b0979b5636027b3ff313cb7656be26c4"
)
OLD_EXECUTION_B_GIT_COMMIT = "079c0952017a9e2936f5741bbe15345255b0e481"
OLD_SOURCE_A_GIT_COMMIT = "990f015b02e0dc89dedaab3853fab9f07fb0884d"
OLD_RUNNER_FREEZE_PATH = "code/configs/causalcache_gate_v1_formal_cache_runner_v1.json"
OLD_RUNNER_FREEZE_SHA256 = (
    "9e941eaff5cc44a44784d22a7568a692d460aea9b07cfd580bf362940158f106"
)
RUNNER_FREEZE_B_PATH = (
    "code/configs/causalcache_gate_v1_formal_cache_transport_repair_runner_v1.json"
)
CORRECTED_TRANSPORT_SHA256 = (
    "fe93e9deeeb9a3c018227fb781b29f6efefefbb728db987584b650d9df353a6d"
)
WRONG_TRANSPORT_SHA256 = (
    "00fe93e9deeeb9a3c018227fb781b29f6efefbb728db987584b650d9df353a6d"
)
TRANSPORT_PATH = (
    "derived/restoration-v2-label-expansion-v1/trajectories-00000-of-00001.jsonl"
)
TRANSPORT_SIZE_BYTES = 1245673
PRODUCER_ARTIFACT_PATH = (
    "data/results/restoration_v2_2_label_expansion_derived/artifact.json"
)
PRODUCER_ARTIFACT_SHA256 = (
    "db518ce830fc233d6f27905fe9b341933995fc272262ccb64d8ca436e3a8e04e"
)
PRODUCER_ARTIFACT_SIZE_BYTES = 20707
FAILURE_SUMMARY_PATH = "data/results/gate_v1_formal58_cache_v1_attempt/summary.json"

EXPECTED_SECTION_SHA256 = {
    "parent_gate_contract": "48132ed7035f59e2f7fa9784b13455c86d51fcaf684eeb05c1940c7a5fedafaf",
    "source_freeze": "7cf3f97a4c379b3a8bba871749889d82508cf263223abf29b8b30a242c9dfb3d",
    "input_artifacts": "a1cd24ee35e393ba9eb0c13b9363a81c47fd13a791311d20246184c0c100da34",
    "publication_completion_binding": "0778b510b9131b81736d6b6fff9c2d8d5f363a551e3f956db26d5ec2c852c841",
    "formal_geometry": "5da623863743d3daf7db3a61ad707a685cf13d6435bb5ffbb937bebe9c2b1c66",
    "access_firewall": "c3e157a3079e55f66fa919daabf4cfeb41d5a866ae63728ced30ed6e2bab0841",
    "cache_formats": "3f588a4a491e81fc26cc3f5155a317bf9f3da3e5504c30e894b01f4deb30e2b6",
    "local_first_state_machine": "40a70c9642fd4fd2e27348374f6d6794a44ea7454d77121493451a29c87d3984",
    "destination": "5d7b56196e2c44932a74ebfc97cd6f3515f1a09101170549a26cc10dc50fcbb6",
    "runtime_contract": "af0362daab6ccbfd86fbd812f0eec53241708c0620b89cbe7ddc2b7cfe5d940b",
    "source_only_operation_contract": "29827efa954416710ae51a7671baaff5bccc4533944c4df94c5f706e0d0354b6",
    "execution_expected_operation_contract": "6f40aa1a2faf1dd41f064903d3f1c985a307113c17c81520fc46f227c0d9fadd",
    "authorization": "d36508bcddc47b6d86dace9de8f90be39b9b8fcdb8bd7bef82e1a44d6f6dbd36",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    *EXPECTED_SECTION_SHA256,
}

REPAIR_SOURCE_PATHS = (
    "code/configs/causalcache_gate_v1_formal_cache_v1.json",
    "code/causalcache/gate_v1_formal_cache_contract.py",
    "code/causalcache/gate_v1_formal_cache.py",
    "code/scripts/validate_gate_v1_formal_cache_contract.py",
    "code/tests/test_gate_v1_formal_cache_contract.py",
    "code/tests/test_gate_v1_formal_cache.py",
    "code/causalcache/gate_v1_data.py",
    "code/tests/test_gate_v1_pipeline.py",
    "code/causalcache/gate_v1_formal_cache_runner.py",
    "code/scripts/manage_gate_v1_formal_cache.py",
    "code/tests/test_gate_v1_formal_cache_runner.py",
    CANONICAL_CONFIG_PATH,
    "code/causalcache/gate_v1_formal_cache_transport_repair_contract.py",
    "code/scripts/validate_gate_v1_formal_cache_transport_repair_contract.py",
    "code/tests/test_gate_v1_formal_cache_transport_repair_contract.py",
    "code/scripts/manage_gate_v1_formal_cache_transport_repair.py",
    "code/tests/test_gate_v1_formal_cache_transport_repair_runner.py",
)
ADDED_GIT_PREREQUISITES = (
    {
        "path": PARENT_CONFIG_PATH,
        "sha256": PARENT_CONFIG_SHA256,
        "size_bytes": 21044,
    },
    {
        "path": OLD_RUNNER_FREEZE_PATH,
        "sha256": OLD_RUNNER_FREEZE_SHA256,
        "size_bytes": 8533,
    },
    {
        "path": FAILURE_SUMMARY_PATH,
        "sha256": FAILURE_SUMMARY_SHA256,
        "size_bytes": 3198,
    },
)
REPAIR_MARKER = {
    "protocol_id": OVERLAY_PROTOCOL_ID,
    "parent_config_sha256": PARENT_CONFIG_SHA256,
    "input_key": "expansion_feature_trajectories",
    "artifact_name": "expansion_derived_features",
    "file_path": TRANSPORT_PATH,
    "wrong_sha256": WRONG_TRANSPORT_SHA256,
    "corrected_sha256": CORRECTED_TRANSPORT_SHA256,
    "size_bytes": TRANSPORT_SIZE_BYTES,
    "data_bytes_changed": False,
    "semantic_contract_changed": False,
}
REPAIR_EVIDENCE = {
    "parent_original_config": {
        "path": PARENT_CONFIG_PATH,
        "sha256": PARENT_CONFIG_SHA256,
        "size_bytes": 21044,
        "protocol_id": PARENT_PROTOCOL_ID,
    },
    "failure_summary": {
        "path": FAILURE_SUMMARY_PATH,
        "sha256": FAILURE_SUMMARY_SHA256,
        "size_bytes": 3198,
        "status": "INVALID_PRE_SEMANTIC_GATE_V1_FORMAL58_CACHE_ATTEMPT_V1",
    },
    "old_execution_b": {
        "git_commit": OLD_EXECUTION_B_GIT_COMMIT,
        "source_a_git_commit": OLD_SOURCE_A_GIT_COMMIT,
        "runner_freeze": {
            "path": OLD_RUNNER_FREEZE_PATH,
            "sha256": OLD_RUNNER_FREEZE_SHA256,
            "size_bytes": 8533,
        },
    },
    "old_global_claim": {
        "path": "/data/experiments/causalcache/.gate-v1-formal58-cache-v1.claim.json",
        "mode": 384,
        "sha256": "a9372c7a4b606323b60e27b8984c5c6d456d62ad16c297fb4737c3d4abd54b1b",
        "size_bytes": 7062,
        "retained_unchanged": True,
    },
    "old_absence": {
        "successor_state_names": [
            "feature_completion",
            "label_access_claim",
            "label_completion",
            "remote_base_receipt",
            "completion_staging",
            "final_completion",
        ],
        "artifact_paths": [
            "/data/artifacts/causalcache/gate-v1-formal58-feature-cache-v1.tar",
            "/data/artifacts/causalcache/gate-v1-formal58-label-cache-v1.tar",
        ],
    },
    "producer_artifact": {
        "path": PRODUCER_ARTIFACT_PATH,
        "sha256": PRODUCER_ARTIFACT_SHA256,
        "size_bytes": PRODUCER_ARTIFACT_SIZE_BYTES,
        "protocol_id": "causalcache_restoration_v2_2_label_expansion_derived_completion_v1",
        "status": "VERIFIED_LABEL_EXPANSION_DERIVED_ARTIFACT",
        "exact_file_record_locations": [
            "artifact.fresh_download.files",
            "artifact.preupload.files",
            "payload_manifest_snapshot.payload_files",
        ],
    },
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class TransportRepairContract:
    """Immutable repair data exposed through the base cache contract interface."""

    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def inputs(self) -> Mapping[str, Any]:
        return _mapping(self.data["input_artifacts"], "input artifacts")

    @property
    def firewall(self) -> Mapping[str, Any]:
        return _mapping(self.data["access_firewall"], "access firewall")

    @property
    def formats(self) -> Mapping[str, Any]:
        return _mapping(self.data["cache_formats"], "cache formats")

    @property
    def local_state(self) -> Mapping[str, Any]:
        return _mapping(self.data["local_first_state_machine"], "local state")

    @property
    def destination(self) -> Mapping[str, Any]:
        return _mapping(self.data["destination"], "destination")

    @property
    def runtime(self) -> Mapping[str, Any]:
        return _mapping(self.data["runtime_contract"], "runtime")

    @property
    def operations(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["source_only_operation_contract"], "source-only operations"
        )

    @property
    def execution_operations(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["execution_expected_operation_contract"],
            "execution operations",
        )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(value, label)


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


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} is not a canonical relative POSIX path")
    return value


def _regular_file_bytes(
    path: Path,
    *,
    label: str,
    expected_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
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
        if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
            raise ValueError(f"{label} mode drifted")
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
    return payload, after


def _canonical_contract_path(root: Path, supplied: str | Path) -> Path:
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = root / candidate
    canonical = (root / CANONICAL_CONFIG_PATH).resolve()
    if candidate.resolve() != canonical:
        raise ValueError("transport-repair config must use its canonical path")
    return canonical


def _base_prerequisites(parent: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    source = _mapping(parent["source_freeze"], "parent source freeze")
    return [
        dict(_mapping(item, "parent Git prerequisite"))
        for item in _sequence(source["git_prerequisites"], "parent Git prerequisites")
    ]


def _expected_repair_data(parent: Mapping[str, Any]) -> Mapping[str, Any]:
    data = copy.deepcopy(dict(parent))
    source = _mapping(data["source_freeze"], "repair source freeze")
    source["required_source_a_paths"] = list(REPAIR_SOURCE_PATHS)
    source["git_prerequisites"] = [
        *_base_prerequisites(parent),
        *copy.deepcopy(ADDED_GIT_PREREQUISITES),
    ]
    runner = dict(_mapping(source["execution_b_runner_freeze"], "runner freeze"))
    runner["path"] = RUNNER_FREEZE_B_PATH
    source["execution_b_runner_freeze"] = runner
    source["transport_repair"] = copy.deepcopy(REPAIR_MARKER)
    source["transport_repair_evidence"] = copy.deepcopy(REPAIR_EVIDENCE)

    inputs = _mapping(data["input_artifacts"], "repair inputs")
    expansion = _mapping(inputs["expansion_derived_features"], "expansion features")
    files = _sequence(expansion["files"], "expansion feature files")
    matches = [
        _mapping(item, "expansion feature file")
        for item in files
        if _mapping(item, "expansion feature file").get("path") == TRANSPORT_PATH
    ]
    if len(matches) != 1:
        raise ValueError("parent expansion transport record is not unique")
    matches[0]["sha256"] = CORRECTED_TRANSPORT_SHA256
    matches[0]["size_bytes"] = TRANSPORT_SIZE_BYTES

    formats = _mapping(data["cache_formats"], "repair formats")
    feature = _mapping(formats["feature"], "repair feature format")
    label = _mapping(formats["label"], "repair label format")
    feature["archive_path"] = (
        "/data/artifacts/causalcache/"
        "gate-v1-formal58-transport-repair-feature-cache-v1.tar"
    )
    label["archive_path"] = (
        "/data/artifacts/causalcache/"
        "gate-v1-formal58-transport-repair-label-cache-v1.tar"
    )

    local = _mapping(data["local_first_state_machine"], "repair local state")
    local["artifact_paths"] = [feature["archive_path"], label["archive_path"]]
    repair_paths = {
        "global_claim": (
            "/data/experiments/causalcache/"
            ".gate-v1-formal58-cache-transport-repair-v1.claim.json"
        ),
        "feature_completion": (
            "/data/experiments/causalcache/"
            ".gate-v1-formal58-cache-transport-repair-v1.feature.completion.json"
        ),
        "label_access_claim": (
            "/data/experiments/causalcache/"
            ".gate-v1-formal58-cache-transport-repair-v1.label-access.claim.json"
        ),
        "label_completion": (
            "/data/experiments/causalcache/"
            ".gate-v1-formal58-cache-transport-repair-v1.label.completion.json"
        ),
        "remote_base_receipt": (
            "/data/experiments/causalcache/"
            ".gate-v1-formal58-cache-transport-repair-v1.remote-base.json"
        ),
        "completion_staging": (
            "/data/experiments/causalcache/"
            ".gate-v1-formal58-cache-transport-repair-v1.completion.staged.json"
        ),
        "final_completion": (
            "/data/experiments/causalcache/"
            ".gate-v1-formal58-cache-transport-repair-v1.completion.json"
        ),
    }
    states = _sequence(local["ordered_states"], "repair ordered states")
    for state in states:
        item = _mapping(state, "repair ordered state")
        item["path"] = repair_paths[item["name"]]

    destination = _mapping(data["destination"], "repair destination")
    destination["repo"] = (
        "gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile"
    )
    destination["tag"] = "gate-v1-formal58-cache-transport-repair-v1"
    destination["commit_title"] = (
        "Publish CausalCache gate v1 formal-58 transport-repair caches"
    )
    destination["exact_three_targets"] = [
        "formal58-transport-repair/v1/feature-cache-v1.tar",
        "formal58-transport-repair/v1/label-cache-v1.tar",
        "formal58-transport-repair/v1/cache-bundle-manifest-v1.json",
    ]
    return data


def _validate_git_prerequisites(config: Mapping[str, Any], root: Path) -> None:
    source = _mapping(config["source_freeze"], "source freeze")
    seen: set[str] = set()
    for raw in _sequence(source["git_prerequisites"], "Git prerequisites"):
        record = _mapping(raw, "Git prerequisite")
        _exact_keys(record, {"path", "sha256", "size_bytes"}, "Git prerequisite")
        relative = _safe_relative_path(record["path"], "Git prerequisite path")
        if relative in seen:
            raise ValueError("Git prerequisite paths must be unique")
        seen.add(relative)
        payload, _ = _regular_file_bytes(root / relative, label=relative)
        if (
            type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
            or not isinstance(record["sha256"], str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or len(payload) != record["size_bytes"]
            or sha256_bytes(payload) != record["sha256"]
        ):
            raise ValueError(f"Git prerequisite identity drifted: {relative}")


def _validate_section_hashes(config: Mapping[str, Any]) -> None:
    for name, expected in EXPECTED_SECTION_SHA256.items():
        section = _mapping(config.get(name), name)
        _equal(
            sha256_bytes(canonical_json_bytes(section)),
            expected,
            f"{name} canonical SHA256",
        )


def _validate_repair_marker(config: Mapping[str, Any]) -> None:
    source = _mapping(config["source_freeze"], "source freeze")
    marker = _mapping(source.get("transport_repair"), "transport-repair marker")
    _exact_keys(marker, set(REPAIR_MARKER), "transport-repair marker")
    _equal(marker, REPAIR_MARKER, "transport-repair marker")
    evidence = _mapping(
        source.get("transport_repair_evidence"), "transport-repair evidence"
    )
    _exact_keys(evidence, set(REPAIR_EVIDENCE), "transport-repair evidence")
    _equal(evidence, REPAIR_EVIDENCE, "transport-repair evidence")
    runner = _mapping(source["execution_b_runner_freeze"], "runner freeze")
    _equal(runner.get("path"), RUNNER_FREEZE_B_PATH, "repair runner-freeze path")
    for key in (
        "must_be_absent_during_source_only_validation",
        "only_allowed_execution_b_source_tree_diff",
        "bind_required_source_a_paths",
        "bind_git_prerequisites",
        "bind_source_a_inventory_sha256",
    ):
        _equal(runner.get(key), True, f"repair runner freeze {key}")


def validate_transport_repair_config(
    config: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    _exact_keys(config, TOP_LEVEL_KEYS, "transport-repair config")
    _equal(config["schema_version"], SCHEMA_VERSION, "schema version")
    # note (luojiaxuan): The overlay deliberately retains the parent public
    # data shape so the repair core can enforce a transport-only delta.
    _equal(config["protocol_id"], PARENT_PROTOCOL_ID, "parent protocol ID")
    _equal(config["status"], PARENT_SOURCE_STATUS, "parent source status")
    parent = load_frozen_formal_cache_contract(
        PARENT_CONFIG_PATH, repository_root=root
    )
    expected = _expected_repair_data(parent.data)
    if dict(config) != dict(expected):
        raise ValueError("repair config differs from the frozen transport-only overlay")
    _validate_section_hashes(config)
    _validate_repair_marker(config)
    _validate_git_prerequisites(config, root)
    return {
        "status": VALIDATION_STATUS,
        "overlay_protocol_id": OVERLAY_PROTOCOL_ID,
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "corrected_transport_sha256": CORRECTED_TRANSPORT_SHA256,
        "transport_size_bytes": TRANSPORT_SIZE_BYTES,
        "execution_authorized": False,
        "formal_label_access_authorized": False,
    }


def load_frozen_transport_repair_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> TransportRepairContract:
    root = Path(repository_root).resolve()
    source = _canonical_contract_path(root, path)
    payload, _ = _regular_file_bytes(source, label="transport-repair config")
    _equal(sha256_bytes(payload), FROZEN_CONFIG_SHA256, "frozen repair config SHA256")
    config = _strict_json_bytes(payload, label="transport-repair config")
    validate_transport_repair_config(config, repository_root=root)
    return TransportRepairContract(
        data=config,
        sha256=FROZEN_CONFIG_SHA256,
        repository_root=root,
        source_path=source,
    )


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"offline git {' '.join(arguments)} failed")
    return result.stdout


def _record_for_path(records: Any, path: str, *, label: str) -> Mapping[str, Any]:
    matches = [
        _mapping(item, label)
        for item in _sequence(records, label)
        if _mapping(item, label).get("path") == path
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} does not bind one exact transport record")
    return matches[0]


def _validate_source_failure_summary(contract: TransportRepairContract) -> None:
    root = contract.repository_root
    evidence = _mapping(
        contract.data["source_freeze"]["transport_repair_evidence"],
        "transport-repair evidence",
    )
    failure = _mapping(evidence["failure_summary"], "failure summary binding")
    payload, _ = _regular_file_bytes(root / failure["path"], label="failure summary")
    if len(payload) != failure["size_bytes"] or sha256_bytes(payload) != failure["sha256"]:
        raise ValueError("failure summary transport identity drifted")
    _equal(
        _git_bytes(root, "show", f"HEAD:{failure['path']}"),
        payload,
        "failure summary HEAD blob",
    )
    summary = _strict_json_bytes(payload, label="failure summary")
    _equal(summary.get("status"), failure["status"], "failure summary status")
    execution = _mapping(summary.get("execution"), "failure execution")
    old_b = _mapping(evidence["old_execution_b"], "old Execution-B")
    _equal(execution.get("execution_b_git_commit"), old_b["git_commit"], "old B commit")
    _equal(execution.get("source_a_git_commit"), old_b["source_a_git_commit"], "old A commit")
    observed_failure = _mapping(summary.get("failure"), "attempt failure")
    marker = _mapping(contract.data["source_freeze"]["transport_repair"], "marker")
    failure_keys = {
        "input_key": "input_key",
        "file_path": "input_path",
        "wrong_sha256": "wrong_frozen_sha256",
        "corrected_sha256": "observed_sha256",
        "size_bytes": "observed_size_bytes",
    }
    for marker_key, failure_key in failure_keys.items():
        _equal(
            observed_failure.get(failure_key),
            marker[marker_key],
            f"failure {marker_key}",
        )
    _equal(
        observed_failure.get("wrong_frozen_size_bytes"),
        marker["size_bytes"],
        "failure wrong size",
    )
    _equal(
        observed_failure.get("exception_message"),
        "downloaded expansion_feature_trajectories transport identity drifted",
        "failure message",
    )
    local = _mapping(summary.get("local_state"), "failure local state")
    old_claim = _mapping(evidence["old_global_claim"], "old global claim")
    for key in ("path", "mode", "sha256", "size_bytes"):
        summary_key = f"claim_{key}" if key != "path" else "claim_path"
        _equal(local.get(summary_key), old_claim[key], f"old claim {key}")
    _equal(local.get("claim_retained_unchanged"), True, "old claim retention")
    absent = _mapping(evidence["old_absence"], "old absence")
    _equal(
        local.get("absent_successor_state_names"),
        absent["successor_state_names"],
        "old successor absence",
    )
    _equal(local.get("absent_artifact_paths"), absent["artifact_paths"], "old cache absence")
    provenance = _mapping(summary.get("provenance"), "failure provenance")
    _equal(provenance.get("contract_sha256"), PARENT_CONFIG_SHA256, "old config SHA")
    runner = _mapping(old_b["runner_freeze"], "old runner freeze")
    _equal(provenance.get("runner_freeze_sha256"), runner["sha256"], "old runner SHA")
    _equal(summary.get("authorization", {}).get("formal_label_semantic_decode_count"), 0, "old label decode count")
    _equal(summary.get("destination_observation", {}).get("hf_api_mutation_count"), 0, "old HF mutation count")

    runner_payload, _ = _regular_file_bytes(root / runner["path"], label="old runner")
    if len(runner_payload) != runner["size_bytes"] or sha256_bytes(runner_payload) != runner["sha256"]:
        raise ValueError("old runner-freeze transport identity drifted")
    _equal(
        _git_bytes(root, "show", f"{old_b['git_commit']}:{runner['path']}"),
        runner_payload,
        "old runner Git blob",
    )
    parents = _git_bytes(root, "rev-list", "--parents", "-n", "1", old_b["git_commit"])
    _equal(
        parents.decode("ascii").strip().split(),
        [old_b["git_commit"], old_b["source_a_git_commit"]],
        "old Execution-B direct parent",
    )


def _validate_source_producer_artifact(contract: TransportRepairContract) -> None:
    root = contract.repository_root
    source = _mapping(contract.data["source_freeze"], "source freeze")
    evidence = _mapping(source["transport_repair_evidence"], "repair evidence")
    producer = _mapping(evidence["producer_artifact"], "producer artifact")
    payload, _ = _regular_file_bytes(root / producer["path"], label="producer artifact")
    if len(payload) != producer["size_bytes"] or sha256_bytes(payload) != producer["sha256"]:
        raise ValueError("producer artifact identity drifted")
    _equal(
        _git_bytes(root, "show", f"HEAD:{producer['path']}"),
        payload,
        "producer artifact HEAD blob",
    )
    artifact = _strict_json_bytes(payload, label="producer artifact")
    _equal(artifact.get("protocol_id"), producer["protocol_id"], "producer protocol")
    _equal(artifact.get("status"), producer["status"], "producer status")
    marker = _mapping(source["transport_repair"], "repair marker")
    input_artifact = _mapping(
        contract.inputs[marker["artifact_name"]], "repair input artifact"
    )
    source_artifact = _mapping(artifact.get("artifact"), "producer artifact section")
    for key in ("repo", "immutable_revision", "tag"):
        _equal(source_artifact.get(key), input_artifact.get(key), f"producer {key}")
    payload_snapshot = _mapping(
        artifact.get("payload_manifest_snapshot"), "producer payload snapshot"
    )
    _equal(
        payload_snapshot.get("dataset_repo"), input_artifact.get("repo"), "payload repo"
    )
    locations = producer["exact_file_record_locations"]
    records = (
        _record_for_path(
            _mapping(source_artifact.get("fresh_download"), "fresh download").get("files"),
            marker["file_path"],
            label="fresh-download files",
        ),
        _record_for_path(
            _mapping(source_artifact.get("preupload"), "preupload").get("files"),
            marker["file_path"],
            label="preupload files",
        ),
        _record_for_path(
            payload_snapshot.get("payload_files"),
            marker["file_path"],
            label="payload snapshot files",
        ),
    )
    _equal(
        tuple(locations),
        (
            "artifact.fresh_download.files",
            "artifact.preupload.files",
            "payload_manifest_snapshot.payload_files",
        ),
        "producer record locations",
    )
    input_record = _record_for_path(
        input_artifact.get("files"), marker["file_path"], label="repair input files"
    )
    for record in (*records, input_record):
        _equal(record.get("sha256"), marker["corrected_sha256"], "corrected transport SHA")
        _equal(record.get("size_bytes"), marker["size_bytes"], "corrected transport size")


def validate_transport_repair_source_only_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    contract = load_frozen_transport_repair_contract(path, repository_root=repository_root)
    source = _mapping(contract.data["source_freeze"], "source freeze")
    inventory: list[Mapping[str, Any]] = []
    for raw in _sequence(source["required_source_a_paths"], "repair source paths"):
        relative = _safe_relative_path(raw, "repair source path")
        payload, _ = _regular_file_bytes(
            contract.repository_root / relative, label=f"repair source path {relative}"
        )
        inventory.append(
            {"path": relative, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        )
    if len(inventory) != len(REPAIR_SOURCE_PATHS):
        raise ValueError("repair source path inventory count drifted")
    runner = _mapping(source["execution_b_runner_freeze"], "repair runner freeze")
    pending_path = contract.repository_root / runner["path"]
    if os.path.lexists(pending_path):
        raise ValueError("transport-repair runner freeze B must remain absent in Source-A")
    _validate_source_failure_summary(contract)
    _validate_source_producer_artifact(contract)
    return {
        "status": VALIDATION_STATUS,
        "config_path": CANONICAL_CONFIG_PATH,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "corrected_transport_sha256": CORRECTED_TRANSPORT_SHA256,
        "source_a_path_count": len(inventory),
        "source_a_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "runner_freeze_b_path": RUNNER_FREEZE_B_PATH,
        "runner_freeze_b_present": False,
        "failed_v1_attempt_local_validation_deferred": True,
        "execution_authorized": False,
    }


def _data_root_path(data_root: Path, canonical_path: str, *, label: str) -> Path:
    supplied_root = Path(data_root)
    supplied_metadata = supplied_root.lstat()
    if supplied_root.is_symlink() or not stat.S_ISDIR(supplied_metadata.st_mode):
        raise ValueError("data root must be a real directory")
    root = supplied_root.resolve()
    path = PurePosixPath(canonical_path)
    if not path.is_absolute() or len(path.parts) < 3 or path.parts[1] != "data":
        raise ValueError(f"{label} is not rooted in canonical /data")
    relative = Path(*path.parts[2:])
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if os.path.lexists(current):
            metadata = current.lstat()
            if current.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"{label} parent is unsafe")
    return root / relative


def _assert_absent(data_root: Path, canonical_path: str, *, label: str) -> None:
    path = _data_root_path(data_root, canonical_path, label=label)
    if os.path.lexists(path):
        raise ValueError(f"{label} must remain absent before repair claim")


def validate_failed_v1_attempt(
    contract: TransportRepairContract,
    data_root: str | Path,
) -> dict[str, Any]:
    """Securely prove that the retained v1 claim cannot be resumed or overwritten."""

    source = _mapping(contract.data["source_freeze"], "source freeze")
    evidence = _mapping(source["transport_repair_evidence"], "repair evidence")
    old_claim = _mapping(evidence["old_global_claim"], "old global claim")
    root = Path(data_root)
    claim_path = _data_root_path(root, old_claim["path"], label="old global claim")
    payload, metadata = _regular_file_bytes(
        claim_path,
        label="old global claim",
        expected_mode=old_claim["mode"],
    )
    if (
        len(payload) != old_claim["size_bytes"]
        or sha256_bytes(payload) != old_claim["sha256"]
        or stat.S_IMODE(metadata.st_mode) != old_claim["mode"]
    ):
        raise ValueError("old global claim identity drifted")
    parent = load_frozen_formal_cache_contract(
        PARENT_CONFIG_PATH, repository_root=contract.repository_root
    )
    old_states = _sequence(parent.local_state["ordered_states"], "old ordered states")
    by_name = {
        _mapping(item, "old ordered state")["name"]: _mapping(item, "old ordered state")
        for item in old_states
    }
    absent = _mapping(evidence["old_absence"], "old absence")
    for name in _sequence(absent["successor_state_names"], "old successor names"):
        state = by_name.get(name)
        if state is None:
            raise ValueError("old successor name escaped parent state machine")
        _assert_absent(root, state["path"], label=f"old successor {name}")
    for path in _sequence(absent["artifact_paths"], "old cache paths"):
        _assert_absent(root, path, label="old cache artifact")
    return {
        "status": FAILED_ATTEMPT_VALIDATION_STATUS,
        "old_claim_path": old_claim["path"],
        "old_claim_sha256": old_claim["sha256"],
        "old_claim_size_bytes": old_claim["size_bytes"],
        "old_claim_mode": old_claim["mode"],
        "absent_successor_state_count": len(absent["successor_state_names"]),
        "absent_artifact_count": len(absent["artifact_paths"]),
        "new_claim_permitted": True,
    }


__all__ = [
    "CANONICAL_CONFIG_PATH",
    "CORRECTED_TRANSPORT_SHA256",
    "FAILED_ATTEMPT_VALIDATION_STATUS",
    "FAILURE_SUMMARY_SHA256",
    "FROZEN_CONFIG_SHA256",
    "OVERLAY_PROTOCOL_ID",
    "PARENT_CONFIG_SHA256",
    "RUNNER_FREEZE_B_PATH",
    "TransportRepairContract",
    "VALIDATION_STATUS",
    "load_frozen_transport_repair_contract",
    "validate_failed_v1_attempt",
    "validate_transport_repair_config",
    "validate_transport_repair_source_only_contract",
]
