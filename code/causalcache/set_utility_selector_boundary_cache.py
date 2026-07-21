"""Persistent full-sequence cache for a selector-only GUI-Owl branch."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from causalcache.set_utility_contextual_hidden import (
    CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS,
    CONTEXTUAL_SOURCE_HIDDEN_SIZE,
    CONTEXTUAL_TEXT_TOKEN_LIMIT,
)
from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_PROFILE_ID,
    CONTEXTUAL_REQUIREMENT_STATUS,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file


SELECTOR_BOUNDARY_CACHE_STATUS = "COMPLETED_SET_UTILITY_SELECTOR_BOUNDARY_CACHE"
SELECTOR_BOUNDARY_CHUNK_STATUS = "COMPLETED_SET_UTILITY_SELECTOR_BOUNDARY_CHUNK"
SELECTOR_BOUNDARY_WORKER_STATUS = "COMPLETED_SET_UTILITY_SELECTOR_BOUNDARY_WORKER"
SELECTOR_BOUNDARY_SCHEMA_VERSION = "1.0.0"
SELECTOR_BOUNDARY_TENSOR_ROLES = (
    "boundary_hidden",
    "input_ids",
    "attention_mask",
    "position_ids",
    "visual_indices",
    "text_indices",
)
_CONTEXT_KEY = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SelectorBoundaryTensorBundle:
    tensors: Mapping[str, Any]
    sequence_length: int
    visual_token_count: int
    text_token_count: int
    image_token_id: int


def selector_boundary_profile_id(*, trainable_layer_count: int) -> str:
    if type(trainable_layer_count) is not int or not 1 <= trainable_layer_count < 36:
        raise ValueError("selector boundary layer count is outside the valid range")
    return f"gui_owl_selector_top{trainable_layer_count}_full_sequence_boundary_v1"


def load_selector_context_allowlist(
    path: Path | None,
) -> tuple[frozenset[str] | None, str | None]:
    if path is None:
        return None, None
    payload = path.read_bytes()
    if not payload:
        raise ValueError("selector boundary context allowlist is empty")
    values: Any
    try:
        document = json.loads(payload)
    except json.JSONDecodeError:
        values = [
            line.strip()
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
    else:
        if isinstance(document, list):
            values = document
        elif isinstance(document, dict):
            values = document.get("context_keys")
            if values is None:
                values = document.get("required_context_keys")
            if values is None:
                values = document.get("contexts")
        else:
            values = None
    if not isinstance(values, list):
        raise ValueError("selector boundary allowlist JSON lacks a context-key list")
    keys = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("context_key")
        if not isinstance(value, str) or _CONTEXT_KEY.fullmatch(value) is None:
            raise ValueError(
                "selector boundary allowlist contains an invalid context key"
            )
        keys.append(value)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("selector boundary allowlist is empty or duplicated")
    return frozenset(keys), hashlib.sha256(payload).hexdigest()


def _contextual_indices(
    *, input_ids: Any, image_token_id: int
) -> tuple[Any, Any]:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("selector boundary cache requires PyTorch") from error
    if type(image_token_id) is not int or image_token_id < 0:
        raise ValueError("image token id must be a non-negative integer")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("selector boundary input ids must have batch size one")
    visual = torch.nonzero(input_ids[0] == image_token_id, as_tuple=False).flatten()
    if int(visual.numel()) not in CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS:
        raise ValueError("selector boundary visual token count drifted")
    first_visual = int(visual[0].item())
    text = torch.arange(
        max(0, first_visual - CONTEXTUAL_TEXT_TOKEN_LIMIT),
        first_visual,
        dtype=torch.long,
        device=input_ids.device,
    )
    text = text[input_ids[0].index_select(0, text) != image_token_id]
    if not 1 <= int(text.numel()) <= CONTEXTUAL_TEXT_TOKEN_LIMIT:
        raise ValueError("selector boundary text-token selection is empty or oversized")
    return visual, text


def selector_boundary_tensor_bundle(
    *, forward: Any, image_token_id: int
) -> SelectorBoundaryTensorBundle:
    """Convert a captured branch boundary into lossless CPU cache tensors."""
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("selector boundary cache requires PyTorch") from error
    hidden = forward.boundary_hidden_state
    input_ids = forward.input_ids
    attention_mask = forward.attention_mask
    position_ids = forward.position_ids
    if (
        hidden.ndim != 3
        or hidden.shape[0] != 1
        or hidden.shape[2] != CONTEXTUAL_SOURCE_HIDDEN_SIZE
    ):
        raise ValueError("selector boundary hidden-state geometry drifted")
    sequence_length = int(hidden.shape[1])
    if tuple(input_ids.shape) != (1, sequence_length):
        raise ValueError("selector boundary input-id geometry drifted")
    if tuple(attention_mask.shape) != (1, sequence_length):
        raise ValueError("selector boundary attention-mask geometry drifted")
    if tuple(position_ids.shape) != (3, 1, sequence_length):
        raise ValueError("selector boundary position-id geometry drifted")
    if hidden.dtype != torch.bfloat16:
        raise ValueError("selector boundary hidden state must be bfloat16")
    visual, text = _contextual_indices(
        input_ids=input_ids, image_token_id=image_token_id
    )
    tensors = {
        "boundary_hidden": hidden[0].detach().to(device="cpu").contiguous(),
        "input_ids": input_ids[0]
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous(),
        "attention_mask": attention_mask[0]
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous(),
        "position_ids": position_ids[:, 0]
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous(),
        "visual_indices": visual.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous(),
        "text_indices": text.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous(),
    }
    return SelectorBoundaryTensorBundle(
        tensors=tensors,
        sequence_length=sequence_length,
        visual_token_count=int(visual.numel()),
        text_token_count=int(text.numel()),
        image_token_id=image_token_id,
    )


def selector_boundary_chunk_paths(
    output_root: Path, *, logical_shard: int, chunk_index: int
) -> tuple[Path, Path]:
    if not 0 <= logical_shard < 256 or chunk_index < 0:
        raise ValueError("selector boundary chunk identity is invalid")
    stem = f"chunk-{chunk_index:05d}"
    relative = Path(f"shard-{logical_shard:03d}-of-256")
    return (
        output_root / "boundary-shards" / relative / f"{stem}.safetensors",
        output_root / "receipts" / relative / f"{stem}.json",
    )


def existing_selector_boundary_chunk(
    output_root: Path,
    *,
    logical_shard: int,
    chunk_index: int,
    identity_sha256: str,
) -> dict[str, Any] | None:
    tensor_path, receipt_path = selector_boundary_chunk_paths(
        output_root, logical_shard=logical_shard, chunk_index=chunk_index
    )
    if not tensor_path.exists() and not receipt_path.exists():
        return None
    if not tensor_path.is_file() or not receipt_path.is_file():
        raise ValueError("incomplete selector boundary chunk cannot resume")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != SELECTOR_BOUNDARY_CHUNK_STATUS
        or receipt.get("identity_sha256") != identity_sha256
        or receipt.get("sha256") != sha256_file(tensor_path)
        or receipt.get("byte_count") != tensor_path.stat().st_size
    ):
        raise ValueError("existing selector boundary chunk drifted")
    return receipt


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("safetensors file is shorter than its header length")
        header = json.loads(handle.read(struct.unpack("<Q", raw_length)[0]))
    return {key: value for key, value in header.items() if key != "__metadata__"}


def _requirements(
    input_root: Path, input_manifest: Mapping[str, Any]
) -> dict[str, int]:
    expected: dict[str, int] = {}
    for receipt in input_manifest["requirement_shards"]:
        if receipt.get("status") != CONTEXTUAL_REQUIREMENT_STATUS:
            raise ValueError("selector boundary requirement status drifted")
        path = input_root / receipt["path"]
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError("selector boundary requirement shard drifted")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(rows) != receipt["context_count"]:
            raise ValueError("selector boundary requirement count drifted")
        for row in rows:
            key = row["context_key"]
            if row.get("logical_shard") != receipt["logical_shard"]:
                raise ValueError("selector boundary requirement crossed logical shards")
            if key in expected:
                raise ValueError("selector boundary requirement key is duplicated")
            expected[key] = int(receipt["logical_shard"])
    if len(expected) != input_manifest["context_count"]:
        raise ValueError("selector boundary requirement inventory drifted")
    return expected


def _validate_header_entry(
    *, entry: Mapping[str, Any], dtype: str, shape: list[int], message: str
) -> None:
    if entry.get("dtype") != dtype or entry.get("shape") != shape:
        raise ValueError(message)


def finalize_selector_boundary_cache(
    *,
    input_root: Path,
    source_root: Path,
    cache_root: Path,
    config_path: Path,
    source_revision: str,
    trainable_layer_count: int,
    context_allowlist_path: Path | None = None,
) -> dict[str, Any]:
    """Validate every chunk and atomically publish the boundary-cache manifest."""
    destination = cache_root / "manifest.json"
    if destination.exists():
        raise FileExistsError("selector boundary cache manifest already exists")
    input_manifest_path = input_root / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if (
        input_manifest.get("status") != CONTEXTUAL_INPUT_STATUS
        or input_manifest.get("evaluation_labels_included") is not False
        or input_manifest.get("contextual_profile_id") != CONTEXTUAL_PROFILE_ID
    ):
        raise ValueError("selector boundary cache requires frozen contextual inputs")
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETED_VARIABLE_HISTORY_SOURCE":
        raise ValueError("selector boundary source manifest is not complete")
    all_expected = _requirements(input_root, input_manifest)
    allowlist, allowlist_sha256 = load_selector_context_allowlist(
        context_allowlist_path
    )
    if allowlist is not None and not allowlist.issubset(all_expected):
        raise ValueError("selector boundary allowlist is outside contextual inputs")
    expected = {
        key: value
        for key, value in all_expected.items()
        if allowlist is None or key in allowlist
    }
    profile_id = selector_boundary_profile_id(
        trainable_layer_count=trainable_layer_count
    )
    bindings = {
        "config_sha256": sha256_file(config_path),
        "context_allowlist_sha256": allowlist_sha256,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "selector_boundary_profile_id": profile_id,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_revision": source_revision,
        "trainable_layer_count": trainable_layer_count,
    }
    observed: set[str] = set()
    inventory: dict[str, dict[str, Any]] = {}
    shard_records = []
    image_token_ids: set[int] = set()
    contexts_per_chunks: set[int] = set()
    receipt_paths = sorted((cache_root / "receipts").glob("**/chunk-*.json"))
    try:
        import torch
        from safetensors import safe_open
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError(
            "selector boundary finalization requires torch/safetensors"
        ) from error
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != SELECTOR_BOUNDARY_CHUNK_STATUS
            or receipt.get("schema_version") != SELECTOR_BOUNDARY_SCHEMA_VERSION
            or receipt.get("source_hidden_size")
            != CONTEXTUAL_SOURCE_HIDDEN_SIZE
            or receipt.get("text_token_limit") != CONTEXTUAL_TEXT_TOKEN_LIMIT
            or receipt.get("allowed_visual_token_counts")
            != sorted(CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS)
            or receipt.get("tensor_roles") != list(SELECTOR_BOUNDARY_TENSOR_ROLES)
            or any(receipt.get(key) != value for key, value in bindings.items())
        ):
            raise ValueError("selector boundary chunk binding drifted")
        logical_shard = receipt.get("logical_shard")
        chunk_index = receipt.get("chunk_index")
        expected_tensor_path, expected_receipt_path = selector_boundary_chunk_paths(
            cache_root,
            logical_shard=logical_shard,
            chunk_index=chunk_index,
        )
        if receipt_path != expected_receipt_path:
            raise ValueError("selector boundary receipt path drifted")
        tensor_path = expected_tensor_path
        if (
            not tensor_path.is_file()
            or sha256_file(tensor_path) != receipt.get("sha256")
            or tensor_path.stat().st_size != receipt.get("byte_count")
        ):
            raise ValueError("selector boundary chunk bytes drifted")
        keys = receipt.get("context_keys")
        if (
            not isinstance(keys, list)
            or len(keys) != receipt.get("context_count")
            or len(keys) != len(set(keys))
            or observed.intersection(keys)
        ):
            raise ValueError("selector boundary chunk inventory overlaps or drifts")
        if any(expected.get(key) != logical_shard for key in keys):
            raise ValueError("selector boundary chunk crossed logical shards")
        header = safetensors_header(tensor_path)
        tensor_names = {
            f"{role}__{key}" for key in keys for role in SELECTOR_BOUNDARY_TENSOR_ROLES
        }
        if set(header) != tensor_names:
            raise ValueError("selector boundary tensor inventory differs from receipt")
        image_token_id = receipt.get("image_token_id")
        if type(image_token_id) is not int or image_token_id < 0:
            raise ValueError("selector boundary image token id drifted")
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            for key in keys:
                length = receipt["sequence_lengths"].get(key)
                visual_count = receipt["visual_token_counts"].get(key)
                text_count = receipt["text_token_counts"].get(key)
                if (
                    type(length) is not int
                    or length <= 0
                    or visual_count not in CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS
                    or type(text_count) is not int
                    or not 1 <= text_count <= CONTEXTUAL_TEXT_TOKEN_LIMIT
                ):
                    raise ValueError("selector boundary receipt geometry drifted")
                shapes = {
                    "boundary_hidden": (
                        "BF16",
                        [length, CONTEXTUAL_SOURCE_HIDDEN_SIZE],
                    ),
                    "input_ids": ("I64", [length]),
                    "attention_mask": ("I64", [length]),
                    "position_ids": ("I64", [3, length]),
                    "visual_indices": ("I64", [visual_count]),
                    "text_indices": ("I64", [text_count]),
                }
                for role, (dtype, shape) in shapes.items():
                    _validate_header_entry(
                        entry=header[f"{role}__{key}"],
                        dtype=dtype,
                        shape=shape,
                        message=f"selector boundary {role} tensor geometry drifted",
                    )
                input_ids = handle.get_tensor(f"input_ids__{key}")
                visual = handle.get_tensor(f"visual_indices__{key}")
                text = handle.get_tensor(f"text_indices__{key}")
                if (
                    visual.ndim != 1
                    or text.ndim != 1
                    or not bool(torch.all(visual[1:] > visual[:-1]))
                    or not bool(torch.all(text[1:] > text[:-1]))
                    or int(visual[-1]) >= length
                    or int(text[-1]) >= length
                    or not bool(
                        torch.all(
                            input_ids.index_select(0, visual) == image_token_id
                        )
                    )
                ):
                    raise ValueError("selector boundary contextual indices drifted")
                expected_visual, expected_text = _contextual_indices(
                    input_ids=input_ids[None], image_token_id=image_token_id
                )
                if not torch.equal(visual, expected_visual) or not torch.equal(
                    text, expected_text
                ):
                    raise ValueError("selector boundary contextual selection drifted")
                relative = tensor_path.relative_to(cache_root)
                for role, (_, shape) in shapes.items():
                    inventory[f"{role}:{key}"] = {
                        "dtype": (
                            "torch.bfloat16"
                            if role == "boundary_hidden"
                            else "torch.int64"
                        ),
                        "partition": str(relative.parent),
                        "shape": shape,
                        "shard": relative.name,
                        "tensor": f"{role}__{key}",
                    }
        observed.update(keys)
        image_token_ids.add(image_token_id)
        contexts_per_chunks.add(receipt["contexts_per_chunk"])
        relative = tensor_path.relative_to(cache_root)
        shard_records.append(
            {
                "byte_count": tensor_path.stat().st_size,
                "context_count": len(keys),
                "logical_shard": logical_shard,
                "path": str(relative),
                "sha256": receipt["sha256"],
            }
        )
    if observed != set(expected):
        raise ValueError("selector boundary cache is incomplete")
    if len(image_token_ids) != 1 or len(contexts_per_chunks) != 1:
        raise ValueError("selector boundary chunks mix extraction contracts")
    manifest = {
        **bindings,
        "content_sha256": "",
        "context_count": len(observed),
        "context_scope": "allowlist" if allowlist is not None else "all_contexts",
        "contexts_per_chunk": next(iter(contexts_per_chunks)),
        "contextual_input_content_sha256": input_manifest["content_sha256"],
        "contextual_profile_id": CONTEXTUAL_PROFILE_ID,
        "evaluation_labels_included": False,
        "image_token_id": next(iter(image_token_ids)),
        "schema_version": SELECTOR_BOUNDARY_SCHEMA_VERSION,
        "shards": shard_records,
        "source_hidden_size": CONTEXTUAL_SOURCE_HIDDEN_SIZE,
        "status": SELECTOR_BOUNDARY_CACHE_STATUS,
        "tensor_inventory": inventory,
        "tensor_roles": list(SELECTOR_BOUNDARY_TENSOR_ROLES),
        "unfiltered_context_count": len(all_expected),
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    write_atomic(destination, canonical_json_bytes(manifest, pretty=True))
    return manifest


__all__ = [
    "SELECTOR_BOUNDARY_CACHE_STATUS",
    "SELECTOR_BOUNDARY_CHUNK_STATUS",
    "SELECTOR_BOUNDARY_SCHEMA_VERSION",
    "SELECTOR_BOUNDARY_TENSOR_ROLES",
    "SELECTOR_BOUNDARY_WORKER_STATUS",
    "SelectorBoundaryTensorBundle",
    "existing_selector_boundary_chunk",
    "finalize_selector_boundary_cache",
    "load_selector_context_allowlist",
    "safetensors_header",
    "selector_boundary_chunk_paths",
    "selector_boundary_profile_id",
    "selector_boundary_tensor_bundle",
    "write_atomic",
]
