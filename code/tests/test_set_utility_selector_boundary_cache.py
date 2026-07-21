from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_PROFILE_ID,
    CONTEXTUAL_REQUIREMENT_STATUS,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file
from causalcache import set_utility_selector_boundary_cache as boundary_cache


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value, pretty=True))


def test_context_allowlist_accepts_lines_and_binds_raw_bytes(tmp_path: Path) -> None:
    keys = ("a" * 64, "b" * 64)
    path = tmp_path / "contexts.txt"
    path.write_text("\n".join(keys) + "\n", encoding="utf-8")
    observed, digest = boundary_cache.load_selector_context_allowlist(path)
    assert observed == frozenset(keys)
    assert digest == sha256_file(path)


def test_boundary_bundle_preserves_full_sequence_and_contextual_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        boundary_cache, "CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS", frozenset({2})
    )
    forward = SimpleNamespace(
        boundary_hidden_state=torch.zeros((1, 6, 4096), dtype=torch.bfloat16),
        input_ids=torch.tensor([[10, 11, 99, 99, 12, 13]], dtype=torch.int64),
        attention_mask=torch.ones((1, 6), dtype=torch.int64),
        position_ids=torch.arange(18, dtype=torch.int64).reshape(3, 1, 6),
    )
    bundle = boundary_cache.selector_boundary_tensor_bundle(
        forward=forward, image_token_id=99
    )
    assert bundle.sequence_length == 6
    assert bundle.visual_token_count == 2
    assert bundle.text_token_count == 2
    assert set(bundle.tensors) == set(boundary_cache.SELECTOR_BOUNDARY_TENSOR_ROLES)
    assert tuple(bundle.tensors["boundary_hidden"].shape) == (6, 4096)
    assert torch.equal(bundle.tensors["visual_indices"], torch.tensor([2, 3]))
    assert torch.equal(bundle.tensors["text_indices"], torch.tensor([0, 1]))
    assert torch.equal(bundle.tensors["position_ids"], forward.position_ids[:, 0])


def test_existing_boundary_chunk_is_resumable_only_under_same_identity(
    tmp_path: Path,
) -> None:
    tensor_path, receipt_path = boundary_cache.selector_boundary_chunk_paths(
        tmp_path, logical_shard=3, chunk_index=1
    )
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(b"stable")
    receipt = {
        "byte_count": tensor_path.stat().st_size,
        "allowed_visual_token_counts": [2],
        "identity_sha256": "a" * 64,
        "sha256": sha256_file(tensor_path),
        "status": boundary_cache.SELECTOR_BOUNDARY_CHUNK_STATUS,
    }
    _write_json(receipt_path, receipt)
    assert (
        boundary_cache.existing_selector_boundary_chunk(
            tmp_path,
            logical_shard=3,
            chunk_index=1,
            identity_sha256="a" * 64,
        )
        == receipt
    )
    with pytest.raises(ValueError, match="drifted"):
        boundary_cache.existing_selector_boundary_chunk(
            tmp_path,
            logical_shard=3,
            chunk_index=1,
            identity_sha256="b" * 64,
        )


def test_finalizer_validates_complete_allowlisted_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    monkeypatch.setattr(
        boundary_cache, "CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS", frozenset({2})
    )
    input_root = tmp_path / "inputs"
    source_root = tmp_path / "source"
    cache_root = tmp_path / "cache"
    config_path = tmp_path / "config.json"
    allowlist_path = tmp_path / "allowlist.json"
    key = "a" * 64
    requirement = {
        "context_key": key,
        "logical_shard": 0,
    }
    requirement_path = input_root / "requirement-shards/shard-000-of-256.jsonl"
    requirement_path.parent.mkdir(parents=True)
    requirement_payload = canonical_json_bytes(requirement) + b"\n"
    requirement_path.write_bytes(requirement_payload)
    input_manifest = {
        "content_sha256": "c" * 64,
        "context_count": 1,
        "contextual_profile_id": CONTEXTUAL_PROFILE_ID,
        "evaluation_labels_included": False,
        "requirement_shards": [
            {
                "context_count": 1,
                "logical_shard": 0,
                "path": str(requirement_path.relative_to(input_root)),
                "sha256": hashlib.sha256(requirement_payload).hexdigest(),
                "status": CONTEXTUAL_REQUIREMENT_STATUS,
            }
        ],
        "status": CONTEXTUAL_INPUT_STATUS,
    }
    _write_json(input_root / "manifest.json", input_manifest)
    _write_json(
        source_root / "manifest.json", {"status": "COMPLETED_VARIABLE_HISTORY_SOURCE"}
    )
    config_path.write_text("{}\n", encoding="utf-8")
    allowlist_path.write_text(json.dumps({"context_keys": [key]}), encoding="utf-8")
    allowlist_sha = sha256_file(allowlist_path)
    tensor_path, receipt_path = boundary_cache.selector_boundary_chunk_paths(
        cache_root, logical_shard=0, chunk_index=0
    )
    tensor_path.parent.mkdir(parents=True)
    input_ids = torch.tensor([10, 99, 99, 11, 12], dtype=torch.int64)
    tensors = {
        f"boundary_hidden__{key}": torch.zeros((5, 4096), dtype=torch.bfloat16),
        f"input_ids__{key}": input_ids,
        f"attention_mask__{key}": torch.ones(5, dtype=torch.int64),
        f"position_ids__{key}": torch.zeros((3, 5), dtype=torch.int64),
        f"visual_indices__{key}": torch.tensor([1, 2], dtype=torch.int64),
        f"text_indices__{key}": torch.tensor([0], dtype=torch.int64),
    }
    safetensors.save_file(tensors, str(tensor_path))
    source_revision = "d" * 40
    bindings = {
        "config_sha256": sha256_file(config_path),
        "context_allowlist_sha256": allowlist_sha,
        "input_manifest_sha256": sha256_file(input_root / "manifest.json"),
        "selector_boundary_profile_id": (
            boundary_cache.selector_boundary_profile_id(trainable_layer_count=4)
        ),
        "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "source_revision": source_revision,
        "trainable_layer_count": 4,
    }
    receipt = {
        **bindings,
        "allowed_visual_token_counts": [2],
        "byte_count": tensor_path.stat().st_size,
        "chunk_index": 0,
        "context_count": 1,
        "context_keys": [key],
        "contexts_per_chunk": 8,
        "identity_sha256": "e" * 64,
        "image_token_id": 99,
        "logical_shard": 0,
        "schema_version": boundary_cache.SELECTOR_BOUNDARY_SCHEMA_VERSION,
        "sequence_lengths": {key: 5},
        "sha256": sha256_file(tensor_path),
        "status": boundary_cache.SELECTOR_BOUNDARY_CHUNK_STATUS,
        "source_hidden_size": 4096,
        "tensor_roles": list(boundary_cache.SELECTOR_BOUNDARY_TENSOR_ROLES),
        "text_token_counts": {key: 1},
        "text_token_limit": 64,
        "visual_token_counts": {key: 2},
    }
    _write_json(receipt_path, receipt)
    manifest = boundary_cache.finalize_selector_boundary_cache(
        input_root=input_root,
        source_root=source_root,
        cache_root=cache_root,
        config_path=config_path,
        source_revision=source_revision,
        trainable_layer_count=4,
        context_allowlist_path=allowlist_path,
    )
    assert manifest["context_count"] == 1
    assert manifest["context_scope"] == "allowlist"
    assert manifest["context_allowlist_sha256"] == allowlist_sha
    assert manifest["status"] == boundary_cache.SELECTOR_BOUNDARY_CACHE_STATUS
    assert set(manifest["tensor_inventory"]) == {
        f"{role}:{key}" for role in boundary_cache.SELECTOR_BOUNDARY_TENSOR_ROLES
    }
