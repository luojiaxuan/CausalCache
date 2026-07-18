from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from causalcache.set_utility_full_pool_inventory_v1 import (
    COMPLETE_STATUS,
    EXPECTED_FILE_COUNT,
    EXPECTED_REVISION,
    FROZEN_CONFIG_SHA256,
    PROTOCOL_ID,
    build_remote_inventory_manifest,
    canonical_json_bytes,
    load_frozen_inventory_contract,
    read_token_file,
    sha256_bytes,
    validate_inventory_config,
    validate_inventory_source_only,
    write_manifest_exclusive,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "code/configs/causalcache_set_utility_full_pool_inventory_v1.json"
)


def _entry(index: int, *, size: int = 1000) -> SimpleNamespace:
    return SimpleNamespace(
        path=f"mobile/use/train/shard-{index:05d}-of-00610.parquet",
        blob_id=f"blob-{index}",
        size=size,
        lfs=SimpleNamespace(size=size, sha256=f"{index % 16:x}" * 64),
    )


def _entries() -> list[SimpleNamespace]:
    return [_entry(index, size=1000 + index) for index in range(EXPECTED_FILE_COUNT)]


class FakeApi:
    def __init__(self, entries: list[object]) -> None:
        self.entries = entries
        self.calls: list[dict[str, object]] = []

    def list_repo_tree(
        self,
        repo_id: str,
        *,
        path_in_repo: str,
        recursive: bool,
        expand: bool,
        revision: str,
        repo_type: str,
    ) -> list[object]:
        self.calls.append(
            {
                "repo_id": repo_id,
                "path_in_repo": path_in_repo,
                "recursive": recursive,
                "expand": expand,
                "revision": revision,
                "repo_type": repo_type,
            }
        )
        return self.entries


def test_frozen_source_contract_is_metadata_only() -> None:
    payload = CONFIG_PATH.read_bytes()
    assert sha256_bytes(payload) == FROZEN_CONFIG_SHA256
    config = json.loads(payload)
    validate_inventory_config(config)
    authorization = config["authorization"]
    assert authorization["remote_metadata_inventory_allowed"] is True
    assert all(
        value is False
        for key, value in authorization.items()
        if key != "remote_metadata_inventory_allowed"
    )
    result = validate_inventory_source_only(repository_root=ROOT)
    assert result["config_sha256"] == FROZEN_CONFIG_SHA256
    assert result["remote_metadata_api_call_count"] == 0
    assert result["remote_file_download_count"] == 0
    assert result["row_decode_count"] == 0
    assert result["restoration_label_count"] == 0
    assert result["gpu_count"] == 0


def test_config_rejects_download_or_label_authorization() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in (
        "remote_file_download_allowed",
        "row_decode_allowed",
        "semantic_census_allowed",
        "restoration_label_generation_allowed",
        "hugging_face_mutation_allowed",
        "gpu_allowed",
    ):
        changed = copy.deepcopy(config)
        changed["authorization"][key] = True
        with pytest.raises(ValueError, match="authorization"):
            validate_inventory_config(changed)


def test_exact_610_lfs_tree_produces_canonical_manifest() -> None:
    contract = load_frozen_inventory_contract(repository_root=ROOT)
    api = FakeApi(list(reversed(_entries())))
    manifest = build_remote_inventory_manifest(api=api, contract=contract)
    assert api.calls == [
        {
            "repo_id": "cua-lite/GUIOdyssey",
            "path_in_repo": "mobile/use/train",
            "recursive": True,
            "expand": True,
            "revision": EXPECTED_REVISION,
            "repo_type": "dataset",
        }
    ]
    assert manifest["status"] == COMPLETE_STATUS
    assert manifest["protocol_id"] == PROTOCOL_ID
    assert manifest["inventory"]["file_count"] == EXPECTED_FILE_COUNT
    assert manifest["inventory"]["total_size_bytes"] == sum(
        1000 + index for index in range(EXPECTED_FILE_COUNT)
    )
    files = manifest["inventory"]["files"]
    assert files[0]["path"].endswith("shard-00000-of-00610.parquet")
    assert files[-1]["path"].endswith("shard-00609-of-00610.parquet")
    assert set(files[0]) == {"path", "size_bytes", "lfs_sha256"}
    assert manifest["inventory"]["files_sha256"] == sha256_bytes(
        canonical_json_bytes(files)
    )
    assert manifest["authorization_scope"]["remote_metadata_inventory_only"] is True
    assert manifest["authorization_scope"]["file_download_or_row_decode"] is False


def test_mapping_entries_are_supported() -> None:
    entries = [
        {
            "path": entry.path,
            "blob_id": entry.blob_id,
            "size": entry.size,
            "lfs": {
                "size": entry.lfs.size,
                "sha256": entry.lfs.sha256,
            },
        }
        for entry in _entries()
    ]
    manifest = build_remote_inventory_manifest(
        api=FakeApi(entries),
        contract=load_frozen_inventory_contract(repository_root=ROOT),
    )
    assert manifest["inventory"]["file_count"] == 610


def test_missing_shard_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing or has extra"):
        build_remote_inventory_manifest(
            api=FakeApi(_entries()[:-1]),
            contract=load_frozen_inventory_contract(repository_root=ROOT),
        )


def test_duplicate_shard_path_is_rejected() -> None:
    entries = _entries()
    entries[-1] = entries[0]
    with pytest.raises(ValueError, match="duplicate path"):
        build_remote_inventory_manifest(
            api=FakeApi(entries),
            contract=load_frozen_inventory_contract(repository_root=ROOT),
        )


def test_regex_drift_is_rejected() -> None:
    entries = _entries()
    entries[-1].path = "mobile/use/train/README.md"
    with pytest.raises(ValueError, match="path regex drifted"):
        build_remote_inventory_manifest(
            api=FakeApi(entries),
            contract=load_frozen_inventory_contract(repository_root=ROOT),
        )


def test_non_lfs_target_is_rejected() -> None:
    entries = _entries()
    entries[-1].lfs = None
    with pytest.raises(ValueError, match="not LFS-backed"):
        build_remote_inventory_manifest(
            api=FakeApi(entries),
            contract=load_frozen_inventory_contract(repository_root=ROOT),
        )


@pytest.mark.parametrize(
    "lfs",
    (
        SimpleNamespace(size=99, sha256="a" * 64),
        SimpleNamespace(size=1000, sha256="A" * 64),
        SimpleNamespace(size=1000, sha256="a" * 63),
    ),
)
def test_malformed_lfs_identity_is_rejected(lfs: object) -> None:
    entries = _entries()
    entries[0].lfs = lfs
    with pytest.raises(ValueError, match="LFS identity is malformed"):
        build_remote_inventory_manifest(
            api=FakeApi(entries),
            contract=load_frozen_inventory_contract(repository_root=ROOT),
        )


def test_token_file_accepts_one_token_and_never_returns_padding(tmp_path: Path) -> None:
    token_path = tmp_path / "token.txt"
    token_path.write_text("hf_fixture_secret\n", encoding="utf-8")
    assert read_token_file(token_path) == "hf_fixture_secret"
    token_path.write_text("hf_fixture secret\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one non-empty token") as captured:
        read_token_file(token_path)
    assert "hf_fixture" not in str(captured.value)


def test_manifest_write_is_canonical_and_no_replace(tmp_path: Path) -> None:
    manifest = {"status": COMPLETE_STATUS, "items": [2, 1]}
    output = tmp_path / "nested/inventory.json"
    digest = write_manifest_exclusive(manifest, output)
    assert output.read_bytes() == canonical_json_bytes(manifest) + b"\n"
    assert digest == sha256_bytes(output.read_bytes())
    assert digest != sha256_bytes(canonical_json_bytes(manifest))
    with pytest.raises(FileExistsError, match="already exists"):
        write_manifest_exclusive(manifest, output)
