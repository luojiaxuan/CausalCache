from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from causalcache.set_utility_full_pool_contract import (
    FROZEN_SOURCE_CONFIG_SHA256,
    load_execution_contract,
    validate_source_config,
    validate_source_only,
)
from causalcache.set_utility_full_pool_inventory_v1 import sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = (
    ROOT / "code/configs/causalcache_set_utility_full_pool_census_v2_source.json"
)
EXECUTION_CONFIG = (
    ROOT / "code/configs/causalcache_set_utility_full_pool_census_v2_execution.json"
)


def test_source_contract_is_byte_frozen_and_authorizes_no_execution() -> None:
    assert sha256_bytes(SOURCE_CONFIG.read_bytes()) == FROZEN_SOURCE_CONFIG_SHA256
    config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    validate_source_config(config)
    assert config["authorization"]["consumed_ledger_hash_bound"] is True
    assert all(
        value is False
        for key, value in config["authorization"].items()
        if key != "consumed_ledger_hash_bound"
    )
    assert config["immutable_inputs"]["p1_inventory_manifest_requirement"][
        "sha256"
    ] == "UNBOUND_REQUIRES_P1_COMMIT"
    result = validate_source_only(repository_root=ROOT)
    assert result["p1_manifest_hash_binding_count"] == 0
    assert result["consumed_ledger_hash_binding_count"] == 1
    assert result["row_decode_count"] == 0
    assert result["semantic_census_count"] == 0
    assert result["output_write_count"] == 0
    assert result["gpu_count"] == 0


def test_source_contract_rejects_ad_hoc_execution_authorization() -> None:
    config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    changed["authorization"]["row_decode_allowed"] = True
    with pytest.raises(ValueError, match="authorization drifted"):
        validate_source_config(changed)


def test_materialization_requires_a_separate_committed_execution_config() -> None:
    contract = load_execution_contract(
        repository_root=ROOT,
        execution_config_path=EXECUTION_CONFIG,
    )
    assert contract.binding_sha256("p1_inventory_manifest") == (
        "e892e7e8f226e9500d978147a9698ad206a70ad9c303ebd918350f9e10ae6c5e"
    )
    assert contract.binding_sha256("consumed_ledger") == (
        "b6f44c603b99d2f954b981e01818cf0afa028ce3a410ed935203532a097bb4ad"
    )
    assert contract.data["authorization"]["semantic_census_allowed"] is True
    assert contract.data["authorization"]["trajectory_role_assignment_allowed"] is False
    assert contract.data["authorization"]["restoration_label_generation_allowed"] is False
    assert contract.data["authorization"]["gate_training_allowed"] is False

    missing = ROOT / "code/configs/missing_set_utility_execution.json"
    with pytest.raises(ValueError, match="execution config is required"):
        load_execution_contract(
            repository_root=ROOT,
            execution_config_path=missing,
        )
