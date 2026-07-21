"""Exact byte-level verification for formal structured-training inputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_REQUIREMENT_STATUS,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes


CONTEXTUAL_CACHE_STATUS = "COMPLETED_SET_UTILITY_CONTEXTUAL_HIDDEN_CACHE"
FORMAL_CACHE_RECEIPT_STATUS = "VERIFIED_FORMAL_CONTEXTUAL_CACHE_ON_HOST"
FORMAL_CACHE_RECEIPT_SCHEMA = "causalcache.formal_cache_verification.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _bound_file(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is invalid")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{label} path escapes or is missing")
    return path


def formal_cache_verification_receipt_path(
    root: Path, *, expected_content_sha256: str
) -> Path:
    expected = _sha256(expected_content_sha256, label="formal cache content SHA256")
    return root.resolve() / ".causalcache-verification" / f"cache-{expected}.json"


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("receipt_sha256", None)
    result["receipt_sha256"] = _sha256_bytes(canonical_json_bytes(result))
    return result


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value, pretty=True) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_signed_receipt(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != _sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("formal cache verification receipt signature drifted")
    return value


def _validated_cache_manifest(
    root: Path, *, expected_content_sha256: str
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    expected = _sha256(expected_content_sha256, label="formal cache content SHA256")
    manifest = _read_json(root / "manifest.json")
    shards = manifest.get("shards")
    inventory = manifest.get("tensor_inventory")
    unsigned = dict(manifest)
    unsigned["content_sha256"] = ""
    observed_content = _sha256_bytes(canonical_json_bytes(unsigned))
    if (
        manifest.get("schema_version") != "3.0.0"
        or manifest.get("status") != CONTEXTUAL_CACHE_STATUS
        or manifest.get("evaluation_labels_included") is not False
        or manifest.get("content_sha256") != expected
        or observed_content != expected
        or not isinstance(shards, list)
        or not shards
        or not isinstance(inventory, Mapping)
        or not inventory
    ):
        raise ValueError("formal contextual cache manifest or content drifted")

    normalized_shards = []
    shard_paths: set[str] = set()
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise ValueError("formal contextual cache shard receipt is invalid")
        path = _bound_file(
            root, shard.get("path"), label=f"formal contextual cache shard {index}"
        )
        normalized = path.relative_to(root).as_posix()
        if normalized in shard_paths:
            raise ValueError("formal contextual cache shard path is duplicated")
        shard_paths.add(normalized)
        byte_count = shard.get("byte_count")
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("formal contextual cache shard size is invalid")
        normalized_shards.append(
            {
                "byte_count": byte_count,
                "path": normalized,
                "sha256": _sha256(
                    shard.get("sha256"), label="formal cache shard SHA256"
                ),
            }
        )

    inventory_paths = set()
    for record in inventory.values():
        if not isinstance(record, Mapping):
            raise ValueError("formal contextual cache tensor inventory is invalid")
        partition = record.get("partition")
        shard = record.get("shard")
        if not isinstance(partition, str) or not isinstance(shard, str):
            raise ValueError("formal contextual cache tensor path is invalid")
        relative = (Path(partition) / shard).as_posix()
        if relative not in shard_paths:
            raise ValueError("formal contextual cache tensor escapes sealed shards")
        inventory_paths.add(relative)
    if inventory_paths != shard_paths:
        raise ValueError("formal contextual cache shard/tensor inventory drifted")
    return manifest, tuple(normalized_shards)


def _stat_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "byte_count": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def _receipt_matches(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    expected_content_sha256: str,
    manifest_sha256: str,
    shards: tuple[dict[str, Any], ...],
) -> bool:
    if (
        receipt.get("schema_version") != FORMAL_CACHE_RECEIPT_SCHEMA
        or receipt.get("status") != FORMAL_CACHE_RECEIPT_STATUS
        or receipt.get("cache_content_sha256") != expected_content_sha256
        or receipt.get("manifest_file_sha256") != manifest_sha256
        or receipt.get("cache_root") != str(root)
        or not isinstance(receipt.get("shards"), list)
        or len(receipt["shards"]) != len(shards)
    ):
        return False
    for declared, sealed in zip(shards, receipt["shards"], strict=True):
        if not isinstance(sealed, Mapping) or any(
            sealed.get(key) != declared[key]
            for key in ("byte_count", "path", "sha256")
        ):
            return False
        path = root / declared["path"]
        try:
            observed = _stat_fingerprint(path)
        except FileNotFoundError:
            return False
        if any(sealed.get(key) != value for key, value in observed.items()):
            return False
    return True


def verify_formal_training_input(
    root: Path, *, expected_content_sha256: str
) -> dict[str, Any]:
    """Verify the contextual input content formula and every requirement shard."""
    root = root.resolve()
    expected = _sha256(expected_content_sha256, label="formal input content SHA256")
    manifest = _read_json(root / "manifest.json")
    receipts = manifest.get("requirement_shards")
    if (
        manifest.get("status") != CONTEXTUAL_INPUT_STATUS
        or manifest.get("evaluation_labels_included") is not False
        or not isinstance(receipts, list)
        or not receipts
    ):
        raise ValueError("formal contextual input manifest or firewall drifted")
    states_path = _bound_file(
        root, manifest.get("states_jsonl"), label="formal contextual states"
    )
    states_payload = states_path.read_bytes()
    if _sha256_bytes(states_payload) != _sha256(
        manifest.get("states_sha256"), label="formal contextual states SHA256"
    ):
        raise ValueError("formal contextual state bytes drifted")
    state_count = sum(bool(line.strip()) for line in states_payload.splitlines())
    if type(manifest.get("state_count")) is not int or state_count != manifest[
        "state_count"
    ]:
        raise ValueError("formal contextual state count drifted")

    receipt_hashes = []
    seen_paths: set[Path] = set()
    seen_shards: set[int] = set()
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping):
            raise ValueError("formal contextual requirement receipt is invalid")
        logical_shard = receipt.get("logical_shard")
        if (
            receipt.get("status") != CONTEXTUAL_REQUIREMENT_STATUS
            or type(logical_shard) is not int
            or not 0 <= logical_shard < 256
            or logical_shard in seen_shards
        ):
            raise ValueError("formal contextual requirement identity drifted")
        seen_shards.add(logical_shard)
        path = _bound_file(
            root,
            receipt.get("path"),
            label=f"formal contextual requirement {index}",
        )
        if path in seen_paths:
            raise ValueError("formal contextual requirement path is duplicated")
        seen_paths.add(path)
        payload = path.read_bytes()
        claimed = _sha256(
            receipt.get("sha256"), label="formal contextual requirement SHA256"
        )
        if (
            _sha256_bytes(payload) != claimed
            or type(receipt.get("byte_count")) is not int
            or len(payload) != receipt["byte_count"]
        ):
            raise ValueError("formal contextual requirement bytes drifted")
        receipt_hashes.append(claimed)
    if seen_shards != set(range(256)):
        raise ValueError("formal contextual requirement shard inventory is incomplete")
    observed_content = _sha256_bytes(
        states_payload + "".join(receipt_hashes).encode("ascii")
    )
    if manifest.get("content_sha256") != expected or observed_content != expected:
        raise ValueError("formal contextual input content binding drifted")
    return manifest


def verify_formal_contextual_cache(
    root: Path,
    *,
    expected_content_sha256: str,
    receipt_mode: str = "auto",
) -> dict[str, Any]:
    """Verify once by bytes, then reuse a signed host-local stat receipt."""
    root = root.resolve()
    expected = _sha256(expected_content_sha256, label="formal cache content SHA256")
    if receipt_mode not in {"auto", "require"}:
        raise ValueError("formal cache receipt mode must be auto or require")
    manifest, shards = _validated_cache_manifest(
        root, expected_content_sha256=expected
    )
    manifest_sha256 = _sha256_file(root / "manifest.json")
    receipt_path = formal_cache_verification_receipt_path(
        root, expected_content_sha256=expected
    )
    receipt = None
    if receipt_path.is_file():
        receipt = _read_signed_receipt(receipt_path)
        if _receipt_matches(
            receipt,
            root=root,
            expected_content_sha256=expected,
            manifest_sha256=manifest_sha256,
            shards=shards,
        ):
            return manifest
    if receipt_mode == "require":
        raise ValueError("valid host-local formal cache verification receipt is missing")

    sealed_shards = []
    for shard in shards:
        path = root / shard["path"]
        observed = _stat_fingerprint(path)
        if (
            observed["byte_count"] != shard["byte_count"]
            or _sha256_file(path) != shard["sha256"]
        ):
            raise ValueError("formal contextual cache shard bytes drifted")
        sealed_shards.append({**shard, **observed})
    receipt = _signed(
        {
            "cache_content_sha256": expected,
            "cache_root": str(root),
            "manifest_file_sha256": manifest_sha256,
            "schema_version": FORMAL_CACHE_RECEIPT_SCHEMA,
            "shards": sealed_shards,
            "status": FORMAL_CACHE_RECEIPT_STATUS,
        }
    )
    _write_atomic(receipt_path, receipt)
    return manifest


def coordinate_formal_cache_verification(
    root: Path,
    *,
    expected_content_sha256: str,
    rank: int,
    distributed: bool,
    torch: Any,
) -> str:
    """Let rank zero create/refresh the receipt before other ranks consume it."""
    if not distributed:
        return "auto"
    outcome: dict[str, Any] | None = None
    if rank == 0:
        try:
            verify_formal_contextual_cache(
                root,
                expected_content_sha256=expected_content_sha256,
                receipt_mode="auto",
            )
            outcome = {"status": "ok"}
        except Exception as error:  # noqa: BLE001 - every rank needs the failure.
            outcome = {
                "error": str(error),
                "error_type": type(error).__name__,
                "status": "error",
            }
    payload = [outcome]
    torch.distributed.broadcast_object_list(payload, src=0)
    outcome = payload[0]
    if not outcome or outcome.get("status") != "ok":
        detail = "unknown" if not outcome else outcome.get("error", "unknown")
        raise RuntimeError(f"formal cache verification preflight failed: {detail}")
    return "require"


__all__ = [
    "CONTEXTUAL_CACHE_STATUS",
    "FORMAL_CACHE_RECEIPT_SCHEMA",
    "FORMAL_CACHE_RECEIPT_STATUS",
    "coordinate_formal_cache_verification",
    "formal_cache_verification_receipt_path",
    "verify_formal_contextual_cache",
    "verify_formal_training_input",
]
