"""Fail-closed source contract for the versioned parallel v2 postflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.set_utility_processor_artifacts import canonical_pretty_json_bytes
from causalcache.set_utility_processor_freeze_contract_v2 import (
    CANONICAL_EXECUTION_CONFIG_PATH,
    ProcessorFreezeExecutionContractV2,
    load_execution_contract,
)
from causalcache.set_utility_processor_freeze import WORKER_COUNT


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_processor_postflight_parallel_contract_v1"
VALIDATION_STATUS = "VALID_SET_UTILITY_PROCESSOR_POSTFLIGHT_PARALLEL_CONTRACT_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_processor_postflight_parallel_v1.json"
)
CONTRACT_PATH = (
    "code/causalcache/set_utility_processor_postflight_parallel_contract_v1.py"
)
PARALLEL_POSTFLIGHT_PATH = (
    "code/causalcache/set_utility_processor_postflight_parallel_v1.py"
)
CLI_PATH = (
    "code/scripts/validate_set_utility_processor_freeze_v2_output_parallel_v1.py"
)
HISTORICAL_POSTFLIGHT_PATH = (
    "code/causalcache/set_utility_processor_postflight_v2.py"
)
HISTORICAL_POSTFLIGHT_SHA256 = (
    "de6eb1a18ea896890efc8361e287733300c1e704887582c5f82342e69382f7cc"
)
HISTORICAL_POSTFLIGHT_BYTE_COUNT = 37150
HISTORICAL_EXECUTION_CONFIG_SHA256 = (
    "e2c271e00749ca7643899c86fd216a635d007337630ba9a1a19b2314ff4afb70"
)
REQUIRED_SOURCE_PATHS = (
    CONTRACT_PATH,
    PARALLEL_POSTFLIGHT_PATH,
    CLI_PATH,
)


@dataclass(frozen=True)
class ParallelPostflightContractV1:
    data: Mapping[str, Any]
    repository_root: Path
    config_sha256: str
    historical_execution_contract: ProcessorFreezeExecutionContractV2


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("parallel postflight contract has duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("parallel postflight contract is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("parallel postflight contract must be one JSON object")
    return value


def _repository_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError("parallel postflight source path is unsafe")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"parallel postflight source is not a regular file: {relative}")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("parallel postflight source escaped repository root") from exc
    return path


def _validate_binding(root: Path, value: Any, *, expected_path: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "byte_count",
        "path",
        "sha256",
    }:
        raise ValueError("parallel postflight source binding schema drifted")
    if value.get("path") != expected_path:
        raise ValueError("parallel postflight source binding path drifted")
    payload = _repository_file(root, expected_path).read_bytes()
    if value.get("byte_count") != len(payload) or value.get("sha256") != _sha256(
        payload
    ):
        raise ValueError(f"parallel postflight source binding drifted: {expected_path}")


def load_parallel_postflight_contract_v1(
    *,
    repository_root: str | Path,
    contract_path: str | Path = CANONICAL_CONFIG_PATH,
) -> ParallelPostflightContractV1:
    root = Path(repository_root).resolve()
    supplied = Path(contract_path)
    path = supplied if supplied.is_absolute() else root / supplied
    expected = (root / CANONICAL_CONFIG_PATH).resolve()
    if path.resolve() != expected:
        raise ValueError(f"parallel postflight contract must be canonical: {expected}")
    if path.is_symlink() or not path.is_file():
        raise ValueError("parallel postflight contract must be a regular file")
    payload = path.read_bytes()
    config = _strict_json_object(payload)
    if payload != canonical_pretty_json_bytes(config):
        raise ValueError("parallel postflight contract must be canonical pretty JSON")
    if set(config) != {
        "execution_contract",
        "historical_postflight",
        "parallel_postflight",
        "protocol_id",
        "schema_version",
        "sources",
    }:
        raise ValueError("parallel postflight contract top-level schema drifted")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("parallel postflight contract schema version drifted")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("parallel postflight contract protocol drifted")

    execution = config.get("execution_contract")
    if execution != {
        "path": CANONICAL_EXECUTION_CONFIG_PATH,
        "sha256": HISTORICAL_EXECUTION_CONFIG_SHA256,
    }:
        raise ValueError("parallel postflight execution binding drifted")
    historical_config = _repository_file(
        root,
        CANONICAL_EXECUTION_CONFIG_PATH,
    ).read_bytes()
    if _sha256(historical_config) != HISTORICAL_EXECUTION_CONFIG_SHA256:
        raise ValueError("historical execution config bytes drifted")
    historical_contract = load_execution_contract(
        repository_root=root,
        execution_config_path=root / CANONICAL_EXECUTION_CONFIG_PATH,
    )

    historical = config.get("historical_postflight")
    if historical != {
        "byte_count": HISTORICAL_POSTFLIGHT_BYTE_COUNT,
        "path": HISTORICAL_POSTFLIGHT_PATH,
        "sha256": HISTORICAL_POSTFLIGHT_SHA256,
    }:
        raise ValueError("parallel postflight historical source binding drifted")
    _validate_binding(root, historical, expected_path=HISTORICAL_POSTFLIGHT_PATH)

    parallel = config.get("parallel_postflight")
    if parallel != {
        "aggregation_order": list(range(WORKER_COUNT)),
        "artifact_semantics": "historical_v2_unchanged",
        "executor": "ThreadPoolExecutor",
        "read_only": True,
        "worker_count": WORKER_COUNT,
    }:
        raise ValueError("parallel postflight execution semantics drifted")
    sources = config.get("sources")
    if not isinstance(sources, list) or len(sources) != len(REQUIRED_SOURCE_PATHS):
        raise ValueError("parallel postflight source inventory drifted")
    for binding, expected_path in zip(sources, REQUIRED_SOURCE_PATHS, strict=True):
        _validate_binding(root, binding, expected_path=expected_path)

    return ParallelPostflightContractV1(
        data=config,
        repository_root=root,
        config_sha256=_sha256(payload),
        historical_execution_contract=historical_contract,
    )


__all__ = [
    "CANONICAL_CONFIG_PATH",
    "CLI_PATH",
    "CONTRACT_PATH",
    "HISTORICAL_EXECUTION_CONFIG_SHA256",
    "HISTORICAL_POSTFLIGHT_BYTE_COUNT",
    "HISTORICAL_POSTFLIGHT_PATH",
    "HISTORICAL_POSTFLIGHT_SHA256",
    "PARALLEL_POSTFLIGHT_PATH",
    "PROTOCOL_ID",
    "ParallelPostflightContractV1",
    "REQUIRED_SOURCE_PATHS",
    "SCHEMA_VERSION",
    "VALIDATION_STATUS",
    "load_parallel_postflight_contract_v1",
]
