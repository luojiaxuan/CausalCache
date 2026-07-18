"""Fail-closed Source-A contract for formal-58 gate training."""

from __future__ import annotations

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


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_formal_train_v1"
SOURCE_STATUS = "source_only_frozen_before_formal58_train_execution"
VALIDATION_STATUS = "VALID_SOURCE_ONLY_GATE_V1_FORMAL58_TRAIN_CONTRACT_V1"
CANONICAL_CONFIG_PATH = "code/configs/causalcache_gate_v1_formal_train_v1.json"
RUNNER_FREEZE_B_PATH = (
    "code/configs/causalcache_gate_v1_formal_train_runner_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "bff920266b3f691618005b239c8a0aaa99369b45c0612ae628eec1a8f7eebf2f"
)
GATE_PREREGISTRATION_SHA256 = (
    "37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b"
)
CACHE_RESULT_SUMMARY_SHA256 = (
    "67d0bb81737b4755925ae2744d4af59e2638f69fe0bf9ba1c19c3ba3b89c38b2"
)
FORMAL_SOURCE_IDS_SHA256 = (
    "ccec55afb0882e602f5d2c83ec420682488e2663fe26a7c574a851825c43f225"
)
FORMAL_CACHE_REVISION = "a61b31bf2e69be00f94469f4a2f2d6b336fcc386"
FORMAL_CACHE_TAG_OBJECT = "c603397b9b1b1c3ad472f49125f827b8a096997d"
FORMAL_CACHE_JOIN_AUDIT_SHA256 = (
    "551e70b7e99f7a761f9c50d2adae3be72b0933044982a015a65e93372e41d77b"
)

REQUIRED_SOURCE_A_PATHS = (
    CANONICAL_CONFIG_PATH,
    "code/causalcache/gate_v1_formal_train_contract.py",
    "code/causalcache/gate_v1_formal_train.py",
    "code/causalcache/gate_v1_formal_train_runner.py",
    "code/scripts/validate_gate_v1_formal_train_contract.py",
    "code/scripts/manage_gate_v1_formal_train.py",
    "code/tests/test_gate_v1_formal_train_contract.py",
    "code/tests/test_gate_v1_formal_train.py",
    "code/tests/test_gate_v1_formal_train_runner.py",
    "code/configs/causalcache_gate_v1_preregistration.json",
    "data/results/gate_v1_formal58_cache_transport_repair_v1/summary.json",
    "code/causalcache/gate_v1_contract.py",
    "code/causalcache/gate_v1_data.py",
    "code/causalcache/gate_v1_training.py",
    "code/causalcache/gate_v1_provenance.py",
    "code/causalcache/gate_v1_formal_cache.py",
    "code/causalcache/restoration_v2_2_label_table.py",
)

EXPECTED_SECTION_SHA256 = {
    "lineage": "8883e1321db30b26393df75034274c17067d2392d4214a9f9b26c3e915e558fd",
    "source_freeze": "7dd598a0f4980ab6bb7ad0dd9a0fb5459f7db58df5a47ccc8d7b824da79f0ea4",
    "formal_cache_input": "8c6d2c36bb7dc6a9ce41b6edf88afb951e7183e71361ab4f5d562682022a051e",
    "formal_geometry": "3b1762bccdb37e09ab7316739d61d66f6030785bfa2983159c34b1492a3f3d2d",
    "access_firewall": "36986c32c03a1ff3693e65c4520702c6743c8eb26c42e5312792d0142b730dae",
    "training_contract": "a3dbe76123330137dd874b24aeb67882c80ed9c779233ceabe6a700e23bbd2e5",
    "output_contract": "e58f8224d154f78c7322671b3fd72f7a902d36a9fc7940b394e37c7bde6e194f",
    "destination": "a2f2d088f6364eedff62b3cf0d7e916ef4f785cd2af1269e7fde4e3f9aa4c44d",
    "local_first_state_machine": "53d6e96cfa6540f16cd072f1c38820b443d0a6c8b0e01c9daf85e01e68f60ffe",
    "runtime_contract": "c294847544370db7cee84a0f201b8d527f82cf4b6c7142e4a5543b5e5d845795",
    "source_only_operation_contract": "642e72202395f8cac4ce4130e42c0d0ecf2518bdcb05c5d1d20442a90e6d2d2f",
    "execution_fixed_operation_contract": "816b669660e47e0b290273434d8ce7fc91a356faa2d6b801c22ca7a7602fb5de",
    "authorization": "4a8042a619e2d96fba684110f9c0fb01c7c1c6f570a5f8b863cebcc0671364e1",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    *EXPECTED_SECTION_SHA256,
}

EXPECTED_CACHE_FILES = (
    {
        "kind": "feature_cache",
        "path": "formal58-transport-repair/v1/feature-cache-v1.tar",
        "sha256": "81fded50c4450700220742d3e0be9a5585d1bc51086150515b463bbdf4b4df8e",
        "size_bytes": 993280,
    },
    {
        "kind": "label_cache",
        "path": "formal58-transport-repair/v1/label-cache-v1.tar",
        "sha256": "4f9ef172aaa94c3ea8ce53aa43336c9fcb7800c24e181e1462d8239e31053cee",
        "size_bytes": 163840,
    },
    {
        "kind": "bundle_manifest",
        "path": "formal58-transport-repair/v1/cache-bundle-manifest-v1.json",
        "sha256": "15c8bf56ddad4f6f278599c32aaadcd813e0e016db0d523b4892eb47f8d9d144",
        "size_bytes": 13856,
    },
)
PAYLOAD_TARGETS = (
    *(f"formal58-train/v1/checkpoints/conditional-seed-{seed}.safetensors" for seed in range(5)),
    *(f"formal58-train/v1/checkpoints/independent-seed-{seed}.safetensors" for seed in range(5)),
    "formal58-train/v1/reports/conditional-full-oof-report.json",
    "formal58-train/v1/reports/independent-full-oof-report.json",
)
MANIFEST_TARGETS = (
    "formal58-train/v1/manifests/conditional-ensemble-manifest.json",
    "formal58-train/v1/manifests/independent-ensemble-manifest.json",
    "formal58-train/v1/manifests/run-manifest.json",
    "formal58-train/v1/manifests/bundle-manifest.json",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class FormalTrainContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def cache_input(self) -> Mapping[str, Any]:
        return _mapping(self.data["formal_cache_input"], "formal cache input")

    @property
    def geometry(self) -> Mapping[str, Any]:
        return _mapping(self.data["formal_geometry"], "formal geometry")

    @property
    def firewall(self) -> Mapping[str, Any]:
        return _mapping(self.data["access_firewall"], "access firewall")

    @property
    def training(self) -> Mapping[str, Any]:
        return _mapping(self.data["training_contract"], "training contract")

    @property
    def output(self) -> Mapping[str, Any]:
        return _mapping(self.data["output_contract"], "output contract")

    @property
    def destination(self) -> Mapping[str, Any]:
        return _mapping(self.data["destination"], "destination")

    @property
    def local_state(self) -> Mapping[str, Any]:
        return _mapping(self.data["local_first_state_machine"], "local state")

    @property
    def runtime(self) -> Mapping[str, Any]:
        return _mapping(self.data["runtime_contract"], "runtime contract")

    @property
    def source_operations(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["source_only_operation_contract"], "source-only operations"
        )

    @property
    def execution_operations(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["execution_fixed_operation_contract"], "execution operations"
        )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
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


def _canonical_config_path(root: Path, supplied: str | Path) -> Path:
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = root / candidate
    canonical = (root / CANONICAL_CONFIG_PATH).resolve()
    if candidate.resolve() != canonical:
        raise ValueError("formal-train config must use its canonical path")
    return canonical


def _validate_section_hashes(config: Mapping[str, Any]) -> None:
    for name, expected in EXPECTED_SECTION_SHA256.items():
        section = _mapping(config.get(name), name)
        _equal(
            sha256_bytes(canonical_json_bytes(section)),
            expected,
            f"{name} canonical SHA256",
        )


def _validate_source_freeze(config: Mapping[str, Any]) -> None:
    source = _mapping(config["source_freeze"], "source freeze")
    _equal(source.get("branch"), "main", "source branch")
    _equal(source.get("origin_name"), "origin", "source remote")
    _equal(
        source.get("origin_url"),
        "https://github.com/luojiaxuan/CausalCache.git",
        "source origin URL",
    )
    paths = tuple(
        _safe_relative_path(item, "required Source-A path")
        for item in _sequence(
            source.get("required_source_a_paths"), "required Source-A paths"
        )
    )
    _equal(paths, REQUIRED_SOURCE_A_PATHS, "required Source-A paths")
    if len(paths) != len(set(paths)):
        raise ValueError("required Source-A paths contain duplicates")
    runner = _mapping(source.get("execution_b_runner_freeze"), "runner freeze")
    _equal(runner.get("path"), RUNNER_FREEZE_B_PATH, "runner-freeze B path")
    for key in (
        "must_be_absent_during_source_only_validation",
        "only_allowed_execution_b_source_tree_diff",
        "bind_required_source_a_paths",
        "bind_git_prerequisites",
        "bind_source_a_inventory_sha256",
        "bind_loaded_module_inventory_sha256",
    ):
        _equal(runner.get(key), True, f"runner freeze {key}")
    prerequisites = _sequence(source.get("git_prerequisites"), "Git prerequisites")
    if len(prerequisites) != 8:
        raise ValueError("Git prerequisite count drifted")
    seen: set[str] = set()
    for raw in prerequisites:
        record = _mapping(raw, "Git prerequisite")
        _exact_keys(record, {"path", "sha256", "size_bytes"}, "Git prerequisite")
        path = _safe_relative_path(record["path"], "Git prerequisite path")
        if path in seen:
            raise ValueError("Git prerequisite paths must be unique")
        seen.add(path)
        if (
            not isinstance(record["sha256"], str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError("Git prerequisite identity is malformed")


def _validate_cache_binding(config: Mapping[str, Any]) -> None:
    cache = _mapping(config["formal_cache_input"], "formal cache input")
    _equal(
        cache.get("repo"),
        "gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile",
        "formal cache repo",
    )
    _equal(cache.get("repo_type"), "dataset", "formal cache repo type")
    _equal(cache.get("private"), True, "formal cache privacy")
    _equal(
        cache.get("tag"),
        "gate-v1-formal58-cache-transport-repair-v1",
        "formal cache tag",
    )
    _equal(cache.get("immutable_revision"), FORMAL_CACHE_REVISION, "cache revision")
    _equal(
        cache.get("annotated_tag_object"),
        FORMAL_CACHE_TAG_OBJECT,
        "cache tag object",
    )
    files = tuple(
        dict(_mapping(item, "formal cache file"))
        for item in _sequence(cache.get("exact_three_files"), "formal cache files")
    )
    _equal(files, EXPECTED_CACHE_FILES, "formal cache exact-three files")
    for record in files:
        _safe_relative_path(record["path"], "formal cache file path")
    _equal(
        cache.get("join_only_audit_sha256"),
        FORMAL_CACHE_JOIN_AUDIT_SHA256,
        "formal cache join audit",
    )


def _validate_geometry_training_and_outputs(config: Mapping[str, Any]) -> None:
    geometry = _mapping(config["formal_geometry"], "formal geometry")
    required_geometry = {
        "trajectory_count": 58,
        "state_count": 174,
        "candidate_feature_count": 522,
        "distance_value_count": 1624,
        "conditional_edge_count": 1682,
        "independent_target_count": 522,
        "source_ids_sha256": FORMAL_SOURCE_IDS_SHA256,
        "oof_fold_sizes": [12, 12, 12, 11, 11],
    }
    for key, expected in required_geometry.items():
        _equal(geometry.get(key), expected, f"formal geometry {key}")
    training = _mapping(config["training_contract"], "training contract")
    required_training = {
        "device": "cpu",
        "dtype": "float32",
        "families_in_execution_order": ["conditional", "independent"],
        "learning_rates_in_grid_order": [0.0003, 0.001],
        "seeds_in_grid_order": [0, 1, 2, 3, 4],
        "fold_count": 5,
        "oof_trial_count_per_family": 10,
        "oof_trial_count_total": 20,
        "fold_training_track_count_total": 100,
        "patience_epochs": 50,
        "maximum_epochs": 500,
        "final_fit_track_count_total": 10,
        "model_initialization_count_total": 110,
        "persistent_checkpoint_count": 10,
        "checkpoint_format": "safetensors",
    }
    for key, expected in required_training.items():
        _equal(training.get(key), expected, f"training contract {key}")
    output = _mapping(config["output_contract"], "output contract")
    payload = _mapping(output.get("payload_commit"), "payload commit")
    manifest = _mapping(output.get("manifest_commit"), "manifest commit")
    _equal(
        tuple(payload.get("exact_twelve_targets", ())),
        PAYLOAD_TARGETS,
        "payload targets",
    )
    _equal(
        tuple(manifest.get("exact_four_new_targets", ())),
        MANIFEST_TARGETS,
        "manifest targets",
    )
    if len(set((*PAYLOAD_TARGETS, *MANIFEST_TARGETS))) != 16:
        raise ValueError("formal-train output paths collide")
    for path in (*PAYLOAD_TARGETS, *MANIFEST_TARGETS):
        _safe_relative_path(path, "formal-train output path")
    _equal(output.get("final_tree_target_count"), 16, "final target count")
    _equal(
        manifest.get("ensemble_checkpoint_bindings_must_use_payload_commit"),
        True,
        "ensemble payload binding",
    )
    _equal(
        manifest.get("manifest_files_must_not_embed_manifest_commit"),
        True,
        "manifest non-self-reference rule",
    )
    destination = _mapping(config["destination"], "destination")
    required_destination = {
        "repo": "gavinlaw/causalcache-gate-v1-formal58-selector-mobile",
        "repo_type": "model",
        "private": True,
        "tag": "gate-v1-formal58-train-v1",
        "annotated_tag_required": True,
        "tag_must_resolve_to_manifest_commit": True,
        "completed_replay_remote_mutation_count": 0,
    }
    for key, expected in required_destination.items():
        _equal(destination.get(key), expected, f"destination {key}")


def _validate_firewall_runtime_and_authorization(config: Mapping[str, Any]) -> None:
    firewall = _mapping(config["access_firewall"], "access firewall")
    zero_fields = (
        "source_a_formal58_semantic_decode_count",
        "source_a_fresh16_semantic_decode_count",
        "source_a_legacy_dev5_semantic_decode_count",
        "source_a_confirm20_access_count",
        "source_a_matched_nll_evaluation_count",
        "source_a_closed_loop_episode_count",
        "execution_b_fresh16_semantic_decode_count",
        "execution_b_legacy_dev5_semantic_decode_count",
        "execution_b_confirm20_access_count",
        "execution_b_matched_nll_evaluation_count",
        "execution_b_closed_loop_episode_count",
    )
    for key in zero_fields:
        _equal(firewall.get(key), 0, f"access firewall {key}")
    _equal(
        firewall.get("execution_b_allowed_semantic_roster"),
        "formal_train",
        "execution semantic roster",
    )
    runtime = _mapping(config["runtime_contract"], "runtime contract")
    required_runtime = {
        "device": "cpu",
        "dtype": "float32",
        "gpu_required": False,
        "normalized_device_requests": [],
        "container_privileged": False,
        "container_runtime": "runc",
        "nvidia_visible_devices": "void",
        "cuda_visible_devices": "",
        "nvidia_device_nodes": [],
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "python_implementation": "CPython",
        "python_version": "3.12.3",
        "machine": "x86_64",
        "torch_version": "2.11.0+cu130",
        "safetensors_version": "0.7.0",
        "huggingface_hub_version": "1.16.1",
        "torch_deterministic_algorithms": True,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    for key, expected in required_runtime.items():
        _equal(runtime.get(key), expected, f"runtime {key}")
    receipt = _mapping(
        runtime.get("docker_inspect_receipt"),
        "Docker inspect receipt",
    )
    expected_receipt = {
        "path": (
            "/data/experiments/causalcache/"
            ".gate-v1-formal58-train-v1.docker-inspect.json"
        ),
        "mode": 0o600,
        "schema_version": SCHEMA_VERSION,
        "status": "CAPTURED_GATE_V1_FORMAL58_TRAIN_DOCKER_INSPECT_V1",
        "container_name_prefix": "sglang-omni-jaxan-",
        "normalized_device_requests": [],
        "privileged": False,
        "runtime": "runc",
        "data_mount_destination": "/data",
        "data_mount_rw": True,
    }
    _exact_keys(receipt, set(expected_receipt), "Docker inspect receipt")
    for key, expected in expected_receipt.items():
        _equal(receipt.get(key), expected, f"Docker inspect receipt {key}")
    thread_environment = _mapping(runtime.get("thread_environment"), "thread environment")
    if set(thread_environment.values()) != {"0", "1", "UTC", "C.UTF-8"}:
        raise ValueError("runtime thread environment drifted")
    source_operations = _mapping(
        config["source_only_operation_contract"], "source-only operations"
    )
    if not source_operations or any(value != 0 for value in source_operations.values()):
        raise ValueError("source-only operation contract must be all zero")
    authorization = _mapping(config["authorization"], "authorization")
    if not authorization or any(value is not False for value in authorization.values()):
        raise ValueError("Source-A authorization must remain entirely false")


def validate_formal_train_contract_data(config: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact_keys(config, TOP_LEVEL_KEYS, "formal-train config")
    _equal(config.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(config.get("protocol_id"), PROTOCOL_ID, "protocol ID")
    _equal(config.get("status"), SOURCE_STATUS, "source status")
    _validate_section_hashes(config)
    _validate_source_freeze(config)
    _validate_cache_binding(config)
    _validate_geometry_training_and_outputs(config)
    _validate_firewall_runtime_and_authorization(config)
    return config


def _validate_git_prerequisite_bytes(contract: FormalTrainContract) -> None:
    source = _mapping(contract.data["source_freeze"], "source freeze")
    for raw in _sequence(source["git_prerequisites"], "Git prerequisites"):
        record = _mapping(raw, "Git prerequisite")
        path = _safe_relative_path(record["path"], "Git prerequisite path")
        payload = _regular_file_bytes(
            contract.repository_root / path, label=f"Git prerequisite {path}"
        )
        if len(payload) != record["size_bytes"] or sha256_bytes(payload) != record["sha256"]:
            raise ValueError(f"Git prerequisite identity drifted: {path}")


def _validate_lightweight_lineage(contract: FormalTrainContract) -> None:
    lineage = _mapping(contract.data["lineage"], "lineage")
    prereg_binding = _mapping(
        lineage["gate_preregistration"], "gate preregistration binding"
    )
    prereg_payload = _regular_file_bytes(
        contract.repository_root / prereg_binding["path"],
        label="gate preregistration",
    )
    prereg = _strict_json_bytes(prereg_payload, label="gate preregistration")
    _equal(sha256_bytes(prereg_payload), GATE_PREREGISTRATION_SHA256, "prereg SHA")
    _equal(prereg.get("protocol_id"), prereg_binding["protocol_id"], "prereg protocol")
    formal_roster = _mapping(prereg.get("rosters"), "preregistered rosters")
    _equal(
        _mapping(formal_roster.get("formal_train"), "formal train roster").get(
            "source_ids_sha256"
        ),
        FORMAL_SOURCE_IDS_SHA256,
        "preregistered formal roster",
    )
    result_binding = _mapping(
        lineage["formal_cache_completion_result"], "cache result binding"
    )
    result_payload = _regular_file_bytes(
        contract.repository_root / result_binding["path"],
        label="formal cache completion result",
    )
    result = _strict_json_bytes(result_payload, label="formal cache completion result")
    _equal(sha256_bytes(result_payload), CACHE_RESULT_SUMMARY_SHA256, "cache result SHA")
    _equal(result.get("status"), result_binding["status"], "cache result status")
    artifact = _mapping(result.get("artifact"), "cache result artifact")
    canonical = _mapping(artifact.get("canonical_hf"), "canonical cache artifact")
    _equal(canonical.get("repo"), contract.cache_input["repo"], "cache result repo")
    _equal(canonical.get("repo_type"), "dataset", "cache result repo type")
    _equal(canonical.get("private"), True, "cache result privacy")
    _equal(canonical.get("tag"), contract.cache_input["tag"], "cache result tag")
    _equal(
        canonical.get("tag_resolved_immutable_commit"),
        FORMAL_CACHE_REVISION,
        "cache result revision",
    )
    _equal(
        canonical.get("annotated_tag_object"),
        FORMAL_CACHE_TAG_OBJECT,
        "cache result tag object",
    )
    exact_files = _mapping(canonical.get("exact_three_files"), "cache result files")
    _equal(
        exact_files,
        {record["path"]: record["sha256"] for record in EXPECTED_CACHE_FILES},
        "cache result exact-three",
    )
    formal_cache = _mapping(result.get("formal_cache"), "formal cache result")
    _equal(
        formal_cache.get("join_only_audit_sha256"),
        FORMAL_CACHE_JOIN_AUDIT_SHA256,
        "cache result join audit",
    )
    _equal(
        formal_cache.get("formal58_training_input_eligible"),
        True,
        "cache training eligibility",
    )
    _equal(formal_cache.get("gate_training_authorized"), True, "cache gate authorization")
    _equal(formal_cache.get("gate_trained"), False, "cache result gate state")


def load_frozen_formal_train_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> FormalTrainContract:
    root = Path(repository_root).resolve()
    source = _canonical_config_path(root, path)
    payload = _regular_file_bytes(source, label="formal-train config")
    _equal(sha256_bytes(payload), FROZEN_CONFIG_SHA256, "frozen config SHA256")
    config = _strict_json_bytes(payload, label="formal-train config")
    validate_formal_train_contract_data(config)
    contract = FormalTrainContract(
        data=config,
        sha256=FROZEN_CONFIG_SHA256,
        repository_root=root,
        source_path=source,
    )
    _validate_git_prerequisite_bytes(contract)
    _validate_lightweight_lineage(contract)
    return contract


def _source_inventory(contract: FormalTrainContract) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    for relative in REQUIRED_SOURCE_A_PATHS:
        payload = _regular_file_bytes(
            contract.repository_root / relative,
            label=f"required Source-A path {relative}",
        )
        records.append(
            {
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    return tuple(records)


def validate_formal_train_source_only_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> Mapping[str, Any]:
    before_modules = frozenset(sys.modules)
    contract = load_frozen_formal_train_contract(path, repository_root=repository_root)
    runner_path = contract.repository_root / RUNNER_FREEZE_B_PATH
    if os.path.lexists(runner_path):
        raise ValueError("formal-train runner-freeze B must remain absent in Source-A")
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
        raise ValueError("source-only validation imported torch or Hugging Face")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": VALIDATION_STATUS,
        "config_path": CANONICAL_CONFIG_PATH,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "source_a_path_count": len(inventory),
        "source_a_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "runner_freeze_b_path": RUNNER_FREEZE_B_PATH,
        "runner_freeze_b_present": False,
        "network_call_count": 0,
        "hf_api_call_count": 0,
        "file_write_count": 0,
        "torch_import_count": 0,
        "formal58_semantic_decode_count": 0,
        "fresh16_semantic_decode_count": 0,
        "legacy_dev5_semantic_decode_count": 0,
        "confirm20_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "training_executed": False,
        "execution_authorized": False,
        "formal58_cache_access_authorized": False,
        "formal58_training_authorized": False,
        "checkpoint_publication_authorized": False,
        "gate_trained": False,
        "fresh16_access_authorized": False,
        "legacy_dev5_access_authorized": False,
        "confirm20_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }


__all__ = [
    "CACHE_RESULT_SUMMARY_SHA256",
    "CANONICAL_CONFIG_PATH",
    "EXPECTED_CACHE_FILES",
    "EXPECTED_SECTION_SHA256",
    "FORMAL_CACHE_JOIN_AUDIT_SHA256",
    "FORMAL_CACHE_REVISION",
    "FORMAL_SOURCE_IDS_SHA256",
    "FROZEN_CONFIG_SHA256",
    "FormalTrainContract",
    "GATE_PREREGISTRATION_SHA256",
    "MANIFEST_TARGETS",
    "PAYLOAD_TARGETS",
    "PROTOCOL_ID",
    "REQUIRED_SOURCE_A_PATHS",
    "RUNNER_FREEZE_B_PATH",
    "SOURCE_STATUS",
    "VALIDATION_STATUS",
    "canonical_json_bytes",
    "load_frozen_formal_train_contract",
    "pretty_json_bytes",
    "sha256_bytes",
    "validate_formal_train_contract_data",
    "validate_formal_train_source_only_contract",
]
