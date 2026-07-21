from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_REQUIREMENT_STATUS,
)
from causalcache.set_utility_formal_input_verification import (
    CONTEXTUAL_CACHE_STATUS,
    formal_cache_verification_receipt_path,
    verify_formal_contextual_cache,
    verify_formal_training_input,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value, pretty=True) + b"\n")


def _input_fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "input"
    root.mkdir()
    state_payload = canonical_json_bytes(
        {
            "candidate_event_step_ids": [1, 2, 3, 4, 5],
            "role": "train",
            "state_id": "t0:decision:006",
            "trajectory_id": "t0",
        }
    ) + b"\n"
    (root / "states.jsonl").write_bytes(state_payload)
    receipts = []
    for logical_shard in range(256):
        relative = f"requirement-shards/shard-{logical_shard:03d}-of-256.jsonl"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"requirement\n" if logical_shard == 0 else b""
        path.write_bytes(payload)
        receipts.append(
            {
                "byte_count": len(payload),
                "context_count": int(bool(payload)),
                "logical_shard": logical_shard,
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "status": CONTEXTUAL_REQUIREMENT_STATUS,
            }
        )
    content = hashlib.sha256(
        state_payload
        + "".join(receipt["sha256"] for receipt in receipts).encode("ascii")
    ).hexdigest()
    _write_json(
        root / "manifest.json",
        {
            "content_sha256": content,
            "evaluation_labels_included": False,
            "requirement_shards": receipts,
            "state_count": 1,
            "states_jsonl": "states.jsonl",
            "states_sha256": hashlib.sha256(state_payload).hexdigest(),
            "status": CONTEXTUAL_INPUT_STATUS,
        },
    )
    return root, content


def _cache_fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "cache"
    relative = "context-shards/part-000/shard.safetensors"
    payload = b"sealed contextual tensors"
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    manifest = {
        "content_sha256": "",
        "evaluation_labels_included": False,
        "schema_version": "3.0.0",
        "shards": [
            {
                "byte_count": len(payload),
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
        "status": CONTEXTUAL_CACHE_STATUS,
        "tensor_inventory": {
            "visual:key": {
                "dtype": "torch.bfloat16",
                "partition": "context-shards/part-000",
                "shape": [4, 4096],
                "shard": "shard.safetensors",
                "tensor": "visual__key",
            }
        },
    }
    content = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    manifest["content_sha256"] = content
    _write_json(root / "manifest.json", manifest)
    return root, content


def test_formal_input_recomputes_content_and_requirement_bytes(tmp_path: Path) -> None:
    root, content = _input_fixture(tmp_path)
    assert verify_formal_training_input(
        root, expected_content_sha256=content
    )["content_sha256"] == content

    states = root / "states.jsonl"
    states.write_bytes(states.read_bytes() + b"{}\n")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["states_sha256"] = hashlib.sha256(states.read_bytes()).hexdigest()
    manifest["state_count"] = 2
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="content binding"):
        verify_formal_training_input(root, expected_content_sha256=content)


def test_formal_input_rejects_requirement_shard_tampering(tmp_path: Path) -> None:
    root, content = _input_fixture(tmp_path)
    shard = root / "requirement-shards/shard-000-of-256.jsonl"
    shard.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="requirement bytes"):
        verify_formal_training_input(root, expected_content_sha256=content)


def test_formal_cache_recomputes_manifest_and_shard_bytes(tmp_path: Path) -> None:
    root, content = _cache_fixture(tmp_path)
    assert verify_formal_contextual_cache(
        root, expected_content_sha256=content
    )["content_sha256"] == content

    shard = root / "context-shards/part-000/shard.safetensors"
    shard.write_bytes(b"different contextual tensors")
    with pytest.raises(ValueError, match="shard bytes"):
        verify_formal_contextual_cache(root, expected_content_sha256=content)


def test_formal_cache_rejects_rehashed_shard_under_stale_content(tmp_path: Path) -> None:
    root, content = _cache_fixture(tmp_path)
    shard = root / "context-shards/part-000/shard.safetensors"
    shard.write_bytes(b"different contextual tensors")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["byte_count"] = shard.stat().st_size
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest or content"):
        verify_formal_contextual_cache(root, expected_content_sha256=content)


def test_formal_cache_receipt_reuses_stats_without_rehashing_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, content = _cache_fixture(tmp_path)
    verify_formal_contextual_cache(root, expected_content_sha256=content)
    assert formal_cache_verification_receipt_path(
        root, expected_content_sha256=content
    ).is_file()

    from causalcache import set_utility_formal_input_verification as verification

    original = verification._sha256_file
    hashed = []

    def observed(path: Path) -> str:
        hashed.append(path)
        return original(path)

    monkeypatch.setattr(verification, "_sha256_file", observed)
    verify_formal_contextual_cache(
        root, expected_content_sha256=content, receipt_mode="require"
    )
    assert hashed == [root / "manifest.json"]


def test_formal_cache_require_mode_never_falls_back_to_full_hash(tmp_path: Path) -> None:
    root, content = _cache_fixture(tmp_path)
    with pytest.raises(ValueError, match="host-local"):
        verify_formal_contextual_cache(
            root, expected_content_sha256=content, receipt_mode="require"
        )
