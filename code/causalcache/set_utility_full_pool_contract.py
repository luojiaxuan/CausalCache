"""Two-phase authorization contract for the policy-blind full-pool census."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.set_utility_consumed_ledger import (
    CANONICAL_LEDGER_MANIFEST_PATH,
    EXPECTED_PARTITION_COUNTS,
    EXPECTED_UNION_COUNT,
    STATUS as CONSUMED_LEDGER_STATUS,
)
from causalcache.set_utility_consumed_ledger_contract import (
    PROTOCOL_ID as CONSUMED_LEDGER_PROTOCOL_ID,
    SCHEMA_VERSION as CONSUMED_LEDGER_SCHEMA_VERSION,
)
from causalcache.set_utility_full_pool_inventory_v1 import (
    CANONICAL_CONFIG_PATH as P1_CONTRACT_PATH,
    COMPLETE_STATUS as P1_COMPLETE_STATUS,
    FROZEN_CONFIG_SHA256 as P1_CONTRACT_SHA256,
    PROTOCOL_ID as P1_PROTOCOL_ID,
    SCHEMA_VERSION as P1_SCHEMA_VERSION,
    sha256_bytes,
)


SCHEMA_VERSION = "1.0.0"
SOURCE_PROTOCOL_ID = "causalcache_set_utility_full_pool_census_v2_source"
EXECUTION_PROTOCOL_ID = "causalcache_set_utility_full_pool_census_v2_execution"
SOURCE_STATUS = "SOURCE_ONLY_NOT_EXECUTION_AUTHORIZED"
EXECUTION_STATUS = "EXECUTION_AUTHORIZED_AFTER_P1_AND_LEDGER_COMMIT"
CANONICAL_SOURCE_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_full_pool_census_v2_source.json"
)
FROZEN_SOURCE_CONFIG_SHA256 = (
    "7e65227e710009d3626bd0063d425c831dfc59e9a6bbe871b9d3e4d15e085e8b"
)
BASE_CONFIG_PATH = "code/configs/independent_reference_gate_v1.json"
BASE_CONFIG_SHA256 = (
    "7b62aa31c80536f28bc4e8a3d684ff535bc2315d44a31cb5f002ed6dcb493fd8"
)
P1_MANIFEST_PATH = "data/manifests/set_utility_full_pool_inventory_v1.json"
RUNNER_PATH = "code/scripts/materialize_set_utility_full_pool.py"
RUNNER_SHA256 = (
    "ec43e7da837835a1208580e236b3b268cf48b5b69af2d2921f97d24ccbfb01e4"
)
OUTPUT_PATH = "data/manifests/set_utility_full_pool_census_v2.json"
CONSUMED_LEDGER_SHA256 = (
    "b6f44c603b99d2f954b981e01818cf0afa028ce3a410ed935203532a097bb4ad"
)


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
        raise ValueError(f"{label} keys drifted")


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}")
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
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    return _mapping(value, label)


def _safe_repo_path(root: Path, relative_path: str, *, must_exist: bool) -> Path:
    relative = Path(relative_path)
    if (
        not relative_path
        or relative.is_absolute()
        or ".." in relative.parts
        or relative_path != relative.as_posix()
    ):
        raise ValueError("bound path must be normalized and repository-relative")
    path = root / relative
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"bound path is symlinked: {relative_path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"bound path escaped repository: {relative_path}") from error
    if must_exist and not path.is_file():
        raise ValueError(f"bound file is missing: {relative_path}")
    return path


def _read_bound_bytes(
    root: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
) -> bytes:
    _exact_keys(binding, {"path", "sha256"}, label)
    path = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise ValueError(f"{label} path/SHA256 is malformed")
    payload = _safe_repo_path(root, path, must_exist=True).read_bytes()
    if sha256_bytes(payload) != digest:
        raise ValueError(f"{label} SHA256 drifted")
    return payload


def _expected_science() -> dict[str, Any]:
    return {
        "minimum_decision_count": 6,
        "scientific_maximum_decision_count": None,
        "format_safety_maximum_decisions": None,
        "candidate_capacity_strata": [
            {"name": "decisions_6_9", "minimum": 6, "maximum": 9},
            {"name": "decisions_10_17", "minimum": 10, "maximum": 17},
            {"name": "decisions_18_plus", "minimum": 18, "maximum": None},
        ],
        "candidate_scope": "UNCONSUMED_ELIGIBLE_TRAJECTORIES_ONLY",
        "consumed_group_audit_required": True,
        "train_tune_evaluation_assignment_allowed": False,
        "query_state_selection_allowed": False,
    }


def validate_source_config(config: Mapping[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "status",
            "immutable_inputs",
            "science",
            "output",
            "authorization",
        },
        "source config",
    )
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("protocol_id") != SOURCE_PROTOCOL_ID
        or config.get("status") != SOURCE_STATUS
    ):
        raise ValueError("source config identity drifted")
    inputs = _mapping(config.get("immutable_inputs"), "immutable_inputs")
    _exact_keys(
        inputs,
        {
            "base_parser_config",
            "p1_inventory_contract",
            "p1_inventory_manifest_requirement",
            "consumed_ledger_requirement",
            "execution_runner",
        },
        "immutable_inputs",
    )
    if dict(_mapping(inputs["base_parser_config"], "base_parser_config")) != {
        "path": BASE_CONFIG_PATH,
        "sha256": BASE_CONFIG_SHA256,
    }:
        raise ValueError("base parser binding drifted")
    if dict(_mapping(inputs["p1_inventory_contract"], "p1 contract")) != {
        "path": P1_CONTRACT_PATH,
        "sha256": P1_CONTRACT_SHA256,
    }:
        raise ValueError("P-1 source contract binding drifted")
    if dict(_mapping(inputs["p1_inventory_manifest_requirement"], "P-1 requirement")) != {
        "path": P1_MANIFEST_PATH,
        "schema_version": P1_SCHEMA_VERSION,
        "protocol_id": P1_PROTOCOL_ID,
        "status": P1_COMPLETE_STATUS,
        "sha256": "UNBOUND_REQUIRES_P1_COMMIT",
    }:
        raise ValueError("P-1 future manifest requirement drifted")
    if dict(_mapping(inputs["consumed_ledger_requirement"], "ledger requirement")) != {
        "path": CANONICAL_LEDGER_MANIFEST_PATH,
        "schema_version": CONSUMED_LEDGER_SCHEMA_VERSION,
        "protocol_id": CONSUMED_LEDGER_PROTOCOL_ID,
        "status": CONSUMED_LEDGER_STATUS,
        "sha256": CONSUMED_LEDGER_SHA256,
        "partition_counts": EXPECTED_PARTITION_COUNTS,
        "union_count": EXPECTED_UNION_COUNT,
    }:
        raise ValueError("consumed ledger future binding requirement drifted")
    if dict(_mapping(inputs["execution_runner"], "execution runner")) != {
        "path": RUNNER_PATH,
        "sha256": RUNNER_SHA256,
    }:
        raise ValueError("execution runner binding drifted")
    if dict(_mapping(config.get("science"), "science")) != _expected_science():
        raise ValueError("source science contract drifted")
    if dict(_mapping(config.get("output"), "output")) != {
        "future_manifest_path": OUTPUT_PATH,
        "canonical_json": True,
        "overwrite_allowed": False,
    }:
        raise ValueError("source output contract drifted")
    authorization = _mapping(config.get("authorization"), "authorization")
    expected_authorization = {
        "execution_config_present": False,
        "source_manifest_hash_bound": False,
        "consumed_ledger_hash_bound": True,
        "remote_file_download_allowed": False,
        "row_decode_allowed": False,
        "semantic_census_allowed": False,
        "output_write_allowed": False,
        "trajectory_role_assignment_allowed": False,
        "query_state_selection_allowed": False,
        "ocr_allowed": False,
        "policy_or_model_load_allowed": False,
        "restoration_label_generation_allowed": False,
        "gate_training_allowed": False,
        "gpu_allowed": False,
        "closed_loop_allowed": False,
        "sealed_androidworld_test_access_allowed": False,
    }
    if dict(authorization) != expected_authorization:
        raise ValueError("source-only authorization drifted")


def load_frozen_source_contract(*, repository_root: str | Path) -> Mapping[str, Any]:
    root = Path(repository_root).resolve()
    path = _safe_repo_path(root, CANONICAL_SOURCE_CONFIG_PATH, must_exist=True)
    payload = path.read_bytes()
    if sha256_bytes(payload) != FROZEN_SOURCE_CONFIG_SHA256:
        raise ValueError("full-pool source config SHA256 drifted")
    config = _strict_json(payload, label="full-pool source config")
    validate_source_config(config)
    inputs = _mapping(config["immutable_inputs"], "immutable_inputs")
    for key in ("base_parser_config", "p1_inventory_contract", "execution_runner"):
        _read_bound_bytes(root, _mapping(inputs[key], key), label=key)
    ledger_requirement = _mapping(
        inputs["consumed_ledger_requirement"], "consumed ledger requirement"
    )
    ledger_path = ledger_requirement.get("path")
    if not isinstance(ledger_path, str):
        raise ValueError("consumed ledger requirement path is malformed")
    ledger_payload = _safe_repo_path(root, ledger_path, must_exist=True).read_bytes()
    if sha256_bytes(ledger_payload) != CONSUMED_LEDGER_SHA256:
        raise ValueError("bound consumed ledger SHA256 drifted")
    ledger = _strict_json(ledger_payload, label="bound consumed ledger")
    if (
        ledger.get("schema_version") != CONSUMED_LEDGER_SCHEMA_VERSION
        or ledger.get("protocol_id") != CONSUMED_LEDGER_PROTOCOL_ID
        or ledger.get("status") != CONSUMED_LEDGER_STATUS
        or ledger.get("assignment_count") != EXPECTED_UNION_COUNT
    ):
        raise ValueError("bound consumed ledger identity drifted")
    return config


def validate_source_only(*, repository_root: str | Path) -> dict[str, Any]:
    load_frozen_source_contract(repository_root=repository_root)
    return {
        "status": "VALID_SET_UTILITY_FULL_POOL_CENSUS_V2_SOURCE_ONLY",
        "protocol_id": SOURCE_PROTOCOL_ID,
        "config_sha256": FROZEN_SOURCE_CONFIG_SHA256,
        "p1_manifest_hash_binding_count": 0,
        "consumed_ledger_hash_binding_count": 1,
        "row_decode_count": 0,
        "semantic_census_count": 0,
        "output_write_count": 0,
        "role_assignment_count": 0,
        "query_state_selection_count": 0,
        "ocr_count": 0,
        "model_load_count": 0,
        "restoration_label_count": 0,
        "gate_training_count": 0,
        "gpu_count": 0,
        "closed_loop_count": 0,
    }


@dataclass(frozen=True)
class ExecutionContract:
    data: Mapping[str, Any]
    repository_root: Path
    bound_json: Mapping[str, Mapping[str, Any]]

    def binding_sha256(self, name: str) -> str:
        return str(_mapping(self.data["bindings"], "bindings")[name]["sha256"])

    @property
    def output_path(self) -> Path:
        return _safe_repo_path(
            self.repository_root,
            str(self.data["output"]["manifest_path"]),
            must_exist=False,
        )


def load_execution_contract(
    *,
    repository_root: str | Path,
    execution_config_path: str | Path,
) -> ExecutionContract:
    root = Path(repository_root).resolve()
    supplied = Path(execution_config_path)
    path = supplied if supplied.is_absolute() else root / supplied
    if path.is_symlink() or not path.is_file():
        raise ValueError("a regular future execution config is required")
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("execution config must be committed inside the repository") from error
    config = _strict_json(path.read_bytes(), label="execution config")
    _exact_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "status",
            "bindings",
            "science",
            "infrastructure",
            "output",
            "authorization",
        },
        "execution config",
    )
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("protocol_id") != EXECUTION_PROTOCOL_ID
        or config.get("status") != EXECUTION_STATUS
    ):
        raise ValueError("execution config identity drifted")
    bindings = _mapping(config.get("bindings"), "execution bindings")
    _exact_keys(
        bindings,
        {
            "source_contract",
            "base_parser_config",
            "p1_inventory_manifest",
            "consumed_ledger",
            "execution_runner",
        },
        "execution bindings",
    )
    expected_fixed = {
        "source_contract": (CANONICAL_SOURCE_CONFIG_PATH, FROZEN_SOURCE_CONFIG_SHA256),
        "base_parser_config": (BASE_CONFIG_PATH, BASE_CONFIG_SHA256),
        "execution_runner": (RUNNER_PATH, RUNNER_SHA256),
    }
    for name, (expected_path, expected_hash) in expected_fixed.items():
        binding = _mapping(bindings[name], name)
        if dict(binding) != {"path": expected_path, "sha256": expected_hash}:
            raise ValueError(f"execution binding {name} drifted")
        _read_bound_bytes(root, binding, label=name)
    variable_paths = {
        "p1_inventory_manifest": P1_MANIFEST_PATH,
        "consumed_ledger": CANONICAL_LEDGER_MANIFEST_PATH,
    }
    bound_json: dict[str, Mapping[str, Any]] = {}
    for name, expected_path in variable_paths.items():
        binding = _mapping(bindings[name], name)
        if binding.get("path") != expected_path:
            raise ValueError(f"execution binding {name} path drifted")
        payload = _read_bound_bytes(root, binding, label=name)
        bound_json[name] = _strict_json(payload, label=name)
    bound_json["base_parser_config"] = _strict_json(
        _read_bound_bytes(root, _mapping(bindings["base_parser_config"], "base"), label="base"),
        label="base_parser_config",
    )
    if (
        bound_json["p1_inventory_manifest"].get("protocol_id") != P1_PROTOCOL_ID
        or bound_json["p1_inventory_manifest"].get("status") != P1_COMPLETE_STATUS
    ):
        raise ValueError("bound P-1 manifest is not complete")
    if (
        bound_json["consumed_ledger"].get("protocol_id")
        != CONSUMED_LEDGER_PROTOCOL_ID
        or bound_json["consumed_ledger"].get("status") != CONSUMED_LEDGER_STATUS
    ):
        raise ValueError("bound consumed ledger identity drifted")
    if dict(_mapping(config.get("science"), "execution science")) != _expected_science():
        raise ValueError("execution science drifted from source contract")
    if dict(_mapping(config.get("infrastructure"), "infrastructure")) != {
        "parquet_batch_size": 32,
        "source_bytes_location": "EXTERNAL_LOCAL_PINNED_SHARDS",
    }:
        raise ValueError("execution infrastructure drifted")
    if dict(_mapping(config.get("output"), "execution output")) != {
        "manifest_path": OUTPUT_PATH,
        "canonical_json": True,
        "overwrite_allowed": False,
    }:
        raise ValueError("execution output drifted")
    expected_authorization = {
        "p1_manifest_hash_bound": True,
        "consumed_ledger_hash_bound": True,
        "local_pinned_source_file_access_allowed": True,
        "row_decode_allowed": True,
        "semantic_census_allowed": True,
        "output_write_allowed": True,
        "trajectory_role_assignment_allowed": False,
        "query_state_selection_allowed": False,
        "ocr_allowed": False,
        "policy_or_model_load_allowed": False,
        "restoration_label_generation_allowed": False,
        "gate_training_allowed": False,
        "gpu_allowed": False,
        "closed_loop_allowed": False,
        "sealed_androidworld_test_access_allowed": False,
    }
    if dict(_mapping(config.get("authorization"), "authorization")) != expected_authorization:
        raise ValueError("execution authorization drifted")
    load_frozen_source_contract(repository_root=root)
    return ExecutionContract(data=config, repository_root=root, bound_json=bound_json)


__all__ = [
    "CANONICAL_SOURCE_CONFIG_PATH",
    "EXECUTION_PROTOCOL_ID",
    "EXECUTION_STATUS",
    "FROZEN_SOURCE_CONFIG_SHA256",
    "SOURCE_PROTOCOL_ID",
    "ExecutionContract",
    "load_execution_contract",
    "load_frozen_source_contract",
    "validate_source_config",
    "validate_source_only",
]
