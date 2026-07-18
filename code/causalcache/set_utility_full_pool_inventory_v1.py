"""Metadata-only inventory for the full untouched GUIOdyssey shard universe."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_full_pool_inventory_v1"
SOURCE_STATUS = "source_only_authorizes_remote_metadata_inventory_only"
COMPLETE_STATUS = "COMPLETE_METADATA_ONLY_FULL_POOL_INVENTORY_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_full_pool_inventory_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "1b2b4374d1653bcf22444d8e708c71bc41ac87fa956243ddcb9bd963eeca7e96"
)
EXPECTED_REPO_ID = "cua-lite/GUIOdyssey"
EXPECTED_REVISION = "ea08072b30e523fb4492e4f4597505879ffcd63b"
EXPECTED_DIRECTORY = "mobile/use/train"
EXPECTED_FILE_COUNT = 610
EXPECTED_PATH_REGEX = (
    r"^mobile/use/train/shard-(?P<index>[0-9]{5})-of-00610[.]parquet$"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FullPoolInventoryContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def source(self) -> Mapping[str, Any]:
        return _mapping(self.data["source"], "source")

    @property
    def remote_operation(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["remote_metadata_operation"], "remote metadata operation"
        )

    @property
    def output(self) -> Mapping[str, Any]:
        return _mapping(self.data["output"], "output")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
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


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys drifted")


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
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
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(value, label)


def validate_inventory_config(config: Mapping[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "status",
            "source",
            "remote_metadata_operation",
            "output",
            "authorization",
        },
        "full-pool inventory config",
    )
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["protocol_id"] != PROTOCOL_ID
        or config["status"] != SOURCE_STATUS
    ):
        raise ValueError("full-pool inventory config identity drifted")

    source = _mapping(config["source"], "source")
    _exact_keys(
        source,
        {
            "repo_id",
            "repo_type",
            "revision",
            "target_directory",
            "target_path_regex",
            "expected_file_count",
            "expected_first_index",
            "expected_last_index",
            "index_width",
        },
        "source",
    )
    expected_source = {
        "repo_id": EXPECTED_REPO_ID,
        "repo_type": "dataset",
        "revision": EXPECTED_REVISION,
        "target_directory": EXPECTED_DIRECTORY,
        "target_path_regex": EXPECTED_PATH_REGEX,
        "expected_file_count": EXPECTED_FILE_COUNT,
        "expected_first_index": 0,
        "expected_last_index": EXPECTED_FILE_COUNT - 1,
        "index_width": 5,
    }
    if dict(source) != expected_source:
        raise ValueError("full-pool inventory source binding drifted")

    operation = _mapping(
        config["remote_metadata_operation"], "remote metadata operation"
    )
    _exact_keys(
        operation,
        {
            "api_method",
            "path_in_repo",
            "recursive",
            "expand",
            "required_entry_kind",
            "manifest_fields",
        },
        "remote metadata operation",
    )
    if (
        operation["api_method"] != "huggingface_hub.HfApi.list_repo_tree"
        or operation["path_in_repo"] != EXPECTED_DIRECTORY
        or operation["recursive"] is not True
        or operation["expand"] is not True
        or operation["required_entry_kind"] != "LFS_file"
        or list(_sequence(operation["manifest_fields"], "manifest fields"))
        != ["path", "size_bytes", "lfs_sha256"]
    ):
        raise ValueError("full-pool remote metadata operation drifted")

    output = _mapping(config["output"], "output")
    expected_output = {
        "manifest_path": "data/manifests/set_utility_full_pool_inventory_v1.json",
        "canonical_json": True,
        "overwrite_allowed": False,
        "contains_file_bytes": False,
        "contains_row_or_semantic_content": False,
        "contains_access_token": False,
    }
    if dict(output) != expected_output:
        raise ValueError("full-pool inventory output contract drifted")

    authorization = _mapping(config["authorization"], "authorization")
    expected_authorization = {
        "remote_metadata_inventory_allowed": True,
        "remote_file_download_allowed": False,
        "row_decode_allowed": False,
        "semantic_census_allowed": False,
        "trajectory_role_assignment_allowed": False,
        "query_state_selection_allowed": False,
        "restoration_label_generation_allowed": False,
        "training_allowed": False,
        "hugging_face_mutation_allowed": False,
        "gpu_allowed": False,
        "model_load_allowed": False,
        "policy_forward_allowed": False,
        "closed_loop_allowed": False,
        "sealed_androidworld_test_access_allowed": False,
    }
    if dict(authorization) != expected_authorization:
        raise ValueError("full-pool inventory authorization drifted")


def load_frozen_inventory_contract(
    *,
    repository_root: str | Path,
    contract_path: str | Path = CANONICAL_CONFIG_PATH,
) -> FullPoolInventoryContract:
    root = Path(repository_root).resolve()
    supplied = Path(contract_path)
    supplied = supplied if supplied.is_absolute() else root / supplied
    canonical = root / CANONICAL_CONFIG_PATH
    if supplied.resolve() != canonical.resolve() or supplied.is_symlink() or canonical.is_symlink():
        raise ValueError("full-pool inventory contract path is not canonical")
    try:
        payload = canonical.read_bytes()
    except OSError as error:
        raise ValueError("full-pool inventory contract is missing or unreadable") from error
    observed = sha256_bytes(payload)
    if observed != FROZEN_CONFIG_SHA256:
        raise ValueError("full-pool inventory config SHA256 drifted")
    config = _strict_json_object(payload, label="full-pool inventory contract")
    validate_inventory_config(config)
    return FullPoolInventoryContract(
        data=config,
        sha256=observed,
        repository_root=root,
        source_path=canonical,
    )


def validate_inventory_source_only(
    *,
    repository_root: str | Path,
    contract_path: str | Path = CANONICAL_CONFIG_PATH,
) -> dict[str, Any]:
    contract = load_frozen_inventory_contract(
        repository_root=repository_root,
        contract_path=contract_path,
    )
    return {
        "status": "VALID_SOURCE_ONLY_SET_UTILITY_FULL_POOL_INVENTORY_V1",
        "protocol_id": PROTOCOL_ID,
        "config_sha256": contract.sha256,
        "config_read_count": 1,
        "remote_metadata_api_call_count": 0,
        "remote_file_download_count": 0,
        "row_decode_count": 0,
        "semantic_census_count": 0,
        "restoration_label_count": 0,
        "gpu_count": 0,
        "model_load_count": 0,
    }


def _entry_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)


def _project_lfs_file(entry: Any, *, path_pattern: re.Pattern[str]) -> tuple[int, dict[str, Any]]:
    path = _entry_value(entry, "path")
    if not isinstance(path, str):
        raise ValueError("remote tree entry path is malformed")
    match = path_pattern.fullmatch(path)
    if match is None:
        raise ValueError("remote tree path regex drifted")
    blob_id = _entry_value(entry, "blob_id")
    size = _entry_value(entry, "size")
    lfs = _entry_value(entry, "lfs")
    if not isinstance(blob_id, str) or not blob_id:
        raise ValueError("remote target entry is not a file")
    if type(size) is not int or size <= 0:
        raise ValueError("remote target file size is malformed")
    if lfs is None:
        raise ValueError("remote target file is not LFS-backed")
    lfs_size = _entry_value(lfs, "size")
    lfs_sha256 = _entry_value(lfs, "sha256")
    if (
        type(lfs_size) is not int
        or lfs_size != size
        or not isinstance(lfs_sha256, str)
        or SHA256_PATTERN.fullmatch(lfs_sha256) is None
    ):
        raise ValueError("remote target LFS identity is malformed")
    return int(match.group("index")), {
        "path": path,
        "size_bytes": size,
        "lfs_sha256": lfs_sha256,
    }


def build_remote_inventory_manifest(
    *,
    api: Any,
    contract: FullPoolInventoryContract,
) -> dict[str, Any]:
    source = contract.source
    operation = contract.remote_operation
    entries = api.list_repo_tree(
        source["repo_id"],
        path_in_repo=operation["path_in_repo"],
        recursive=operation["recursive"],
        expand=operation["expand"],
        revision=source["revision"],
        repo_type=source["repo_type"],
    )
    pattern = re.compile(str(source["target_path_regex"]))
    seen_paths: set[str] = set()
    seen_indices: set[int] = set()
    files: list[dict[str, Any]] = []
    for entry in entries:
        index, record = _project_lfs_file(entry, path_pattern=pattern)
        if record["path"] in seen_paths:
            raise ValueError("remote target tree contains a duplicate path")
        if index in seen_indices:
            raise ValueError("remote target tree contains a duplicate shard index")
        seen_paths.add(record["path"])
        seen_indices.add(index)
        files.append(record)

    expected_indices = set(
        range(
            int(source["expected_first_index"]),
            int(source["expected_last_index"]) + 1,
        )
    )
    if len(files) != source["expected_file_count"]:
        raise ValueError("remote target shard universe is missing or has extra files")
    if seen_indices != expected_indices:
        raise ValueError("remote target shard index universe drifted")
    files.sort(key=lambda record: record["path"])
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": COMPLETE_STATUS,
        "source": {
            "repo_id": source["repo_id"],
            "repo_type": source["repo_type"],
            "revision": source["revision"],
            "target_directory": source["target_directory"],
            "config_sha256": contract.sha256,
        },
        "inventory": {
            "file_count": len(files),
            "total_size_bytes": sum(record["size_bytes"] for record in files),
            "files_sha256": sha256_bytes(canonical_json_bytes(files)),
            "files": files,
        },
        "authorization_scope": {
            "remote_metadata_inventory_only": True,
            "file_download_or_row_decode": False,
            "semantic_census_or_role_assignment": False,
            "restoration_labels_or_training": False,
            "model_or_gpu_operations": False,
            "hugging_face_mutation": False,
        },
    }


def read_token_file(path: str | Path) -> str:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            Path(path),
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16 * 1024:
            raise ValueError("token file is not a small regular file")
        payload = os.read(descriptor, 16 * 1024 + 1)
    except OSError as error:
        raise ValueError("token file is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        token = payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("token file is not UTF-8 text") from error
    if not token or any(character.isspace() for character in token):
        raise ValueError("token file does not contain one non-empty token")
    return token


def write_manifest_exclusive(manifest: Mapping[str, Any], output_path: str | Path) -> str:
    payload = canonical_json_bytes(manifest) + b"\n"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise FileExistsError("full-pool inventory output already exists") from error
    return sha256_bytes(payload)


def materialize_remote_inventory(
    *,
    api: Any,
    contract: FullPoolInventoryContract,
    output_path: str | Path,
) -> tuple[dict[str, Any], str]:
    manifest = build_remote_inventory_manifest(api=api, contract=contract)
    digest = write_manifest_exclusive(manifest, output_path)
    return manifest, digest
