from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from causalcache.independent_confirm_contract import (
    CANONICAL_CONFIG_PATH,
    FROZEN_CONFIG_SHA256,
    PROTOCOL_ID,
    RUNNER_FREEZE_PATH,
    IndependentConfirmContract,
    canonical_json_bytes,
    load_runner_freeze_file,
    pretty_json_bytes,
    validate_runner_freeze,
)


ROOT = Path(__file__).resolve().parents[2]


def _contract() -> IndependentConfirmContract:
    return IndependentConfirmContract.load(
        ROOT / CANONICAL_CONFIG_PATH,
        repository_root=ROOT,
    )


def _runner_freeze_value(
    contract: IndependentConfirmContract | None = None,
) -> dict:
    contract = _contract() if contract is None else contract
    source_a = "1" * 40
    source_inventory = [
        {"path": "README.md", "sha256": "2" * 64, "size_bytes": 1}
    ]
    module_inventory = [
        {
            "path": "code/causalcache/independent_confirm_runner.py",
            "sha256": "3" * 64,
            "size_bytes": 1,
        }
    ]
    return {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_CAUSALCACHE_INDEPENDENT_CONFIRM_EXECUTION_B_V1",
        "source_a_commit": source_a,
        "source_a_parent_count": 1,
        "source_a_remote_main": source_a,
        "contract_sha256": contract.sha256,
        "source_a_inventory": source_inventory,
        "source_a_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(source_inventory)
        ).hexdigest(),
        "loaded_module_inventory": module_inventory,
        "loaded_module_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(module_inventory)
        ).hexdigest(),
        "confirm_access_authorized": True,
        "restoration_access_requires_payload_commit": True,
        "closed_loop_train_requires_confirm_go": True,
        "sealed_test_requires_development_go": True,
    }


def _rebind_inventory_digest(value: dict, name: str) -> None:
    inventory = value[f"{name}_inventory"]
    value[f"{name}_inventory_sha256"] = hashlib.sha256(
        canonical_json_bytes(inventory)
    ).hexdigest()


def test_frozen_config_hash_and_source_contract() -> None:
    path = ROOT / CANONICAL_CONFIG_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_CONFIG_SHA256
    contract = IndependentConfirmContract.load(path, repository_root=ROOT)
    assert contract.data["protocol_id"] == PROTOCOL_ID
    assert contract.data["paper_decision"]["primary_method"] == (
        "restoration_guided_independent_gate"
    )
    assert contract.data["paper_decision"]["frozen_negative_result"] == (
        "NO_V2_CONDITIONAL_RESCUE"
    )
    assert contract.data["confirm_geometry"]["trajectory_count"] == 20
    assert contract.data["confirm_geometry"]["candidate_event_step_ids"] == [1, 2, 3, 4]
    assert contract.data["closed_loop_contract"]["validated_teacher_claim_allowed"] is False
    assert contract.data["closed_loop_contract"]["development"][
        "primary_arm_episode_count"
    ] == 120
    assert contract.data["closed_loop_contract"]["sealed_test"][
        "task_indices_per_template"
    ] == [0, 1, 2]
    assert contract.data["closed_loop_contract"]["sealed_test"][
        "primary_arm_episode_count"
    ] == 150
    assert contract.data["matched_nll_contract"]["primary_caliper_nats_per_token"] == 0.05
    assert contract.data["matched_nll_contract"]["primary_claim_partition"] == (
        "sealed_test75_only"
    )
    assert contract.data["matched_nll_contract"][
        "minimum_matched_template_clusters_for_claim"
    ] == 15


def test_confirm_roster_digest_is_canonical() -> None:
    value = json.loads((ROOT / CANONICAL_CONFIG_PATH).read_text())
    geometry = value["confirm_geometry"]
    assert hashlib.sha256(
        canonical_json_bytes(geometry["source_ids"])
    ).hexdigest() == geometry["source_ids_sha256"]


def test_source_load_rejects_runner_freeze_presence(tmp_path: Path) -> None:
    canonical = tmp_path / CANONICAL_CONFIG_PATH
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes((ROOT / CANONICAL_CONFIG_PATH).read_bytes())
    freeze = tmp_path / RUNNER_FREEZE_PATH
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text("{}\n")
    with pytest.raises(PermissionError, match="Execution-B runner freeze"):
        IndependentConfirmContract.load(canonical, repository_root=tmp_path)


def test_source_load_rejects_broken_runner_freeze_symlink(tmp_path: Path) -> None:
    canonical = tmp_path / CANONICAL_CONFIG_PATH
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes((ROOT / CANONICAL_CONFIG_PATH).read_bytes())
    freeze = tmp_path / RUNNER_FREEZE_PATH
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.symlink_to(tmp_path / "missing-runner-freeze.json")
    with pytest.raises(PermissionError, match="Execution-B runner freeze"):
        IndependentConfirmContract.load(canonical, repository_root=tmp_path)


def test_runner_freeze_schema_is_fail_closed() -> None:
    contract = _contract()
    source_a = "1" * 40
    value = _runner_freeze_value(contract)
    validate_runner_freeze(contract, value, source_a_commit=source_a)
    value["sealed_test_requires_development_go"] = False
    with pytest.raises(ValueError, match="test barrier"):
        validate_runner_freeze(contract, value, source_a_commit=source_a)


def test_runner_freeze_rejects_unbound_inventory_hash() -> None:
    contract = _contract()
    source_a = "1" * 40
    value = _runner_freeze_value(contract)
    value["source_a_inventory_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="source_a inventory SHA256"):
        validate_runner_freeze(contract, value, source_a_commit=source_a)


@pytest.mark.parametrize("name", ("source_a", "loaded_module"))
def test_runner_freeze_rejects_inventory_bytes_with_stale_digest(name: str) -> None:
    contract = _contract()
    value = _runner_freeze_value(contract)
    value[f"{name}_inventory"][0]["size_bytes"] += 1
    with pytest.raises(ValueError, match=rf"{name} inventory SHA256"):
        validate_runner_freeze(contract, value, source_a_commit="1" * 40)


@pytest.mark.parametrize("name", ("source_a", "loaded_module"))
@pytest.mark.parametrize(
    ("inventory", "message"),
    (
        ({"path": "README.md"}, "must be a sequence"),
        ([], "inventory is empty"),
        (["not-an-object"], "must be a mapping"),
        (
            [{"path": "README.md", "sha256": "2" * 64}],
            "inventory schema drifted",
        ),
        (
            [
                {
                    "path": "README.md",
                    "sha256": "2" * 64,
                    "size_bytes": 1,
                    "extra": False,
                }
            ],
            "inventory schema drifted",
        ),
        (
            [{"path": "../README.md", "sha256": "2" * 64, "size_bytes": 1}],
            "canonical relative POSIX path",
        ),
        (
            [{"path": "README.md", "sha256": "X" * 64, "size_bytes": 1}],
            "item SHA256",
        ),
        (
            [{"path": "README.md", "sha256": "2" * 64, "size_bytes": True}],
            "item size is invalid",
        ),
        (
            [{"path": "README.md", "sha256": "2" * 64, "size_bytes": 0}],
            "item size is invalid",
        ),
    ),
)
def test_runner_freeze_inventory_arrays_are_fail_closed(
    name: str, inventory: object, message: str
) -> None:
    contract = _contract()
    value = _runner_freeze_value(contract)
    value[f"{name}_inventory"] = copy.deepcopy(inventory)
    _rebind_inventory_digest(value, name)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_runner_freeze(contract, value, source_a_commit="1" * 40)


@pytest.mark.parametrize("name", ("source_a", "loaded_module"))
def test_runner_freeze_rejects_duplicate_inventory_paths(name: str) -> None:
    contract = _contract()
    value = _runner_freeze_value(contract)
    value[f"{name}_inventory"] *= 2
    _rebind_inventory_digest(value, name)
    with pytest.raises(ValueError, match="duplicate paths"):
        validate_runner_freeze(contract, value, source_a_commit="1" * 40)


def test_runner_freeze_rejects_unknown_top_level_key() -> None:
    contract = _contract()
    value = _runner_freeze_value(contract)
    value["unbound_authorization"] = True
    with pytest.raises(ValueError, match="key inventory drifted"):
        validate_runner_freeze(contract, value, source_a_commit="1" * 40)


def test_runner_freeze_file_requires_strict_canonical_pretty_json(
    tmp_path: Path,
) -> None:
    value = _runner_freeze_value()
    path = tmp_path / "runner.json"
    path.write_bytes(pretty_json_bytes(value))
    assert load_runner_freeze_file(path) == value

    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="canonical pretty JSON"):
        load_runner_freeze_file(path)

    path.write_bytes(pretty_json_bytes(value) + b" \n")
    with pytest.raises(ValueError, match="canonical pretty JSON"):
        load_runner_freeze_file(path)


def test_runner_freeze_file_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    value = _runner_freeze_value()
    payload = pretty_json_bytes(value)
    needle = b'{\n  "closed_loop_train_requires_confirm_go"'
    replacement = (
        b'{\n  "schema_version": "forged",\n'
        b'  "closed_loop_train_requires_confirm_go"'
    )
    assert needle in payload
    path = tmp_path / "runner.json"
    path.write_bytes(payload.replace(needle, replacement, 1))
    with pytest.raises(ValueError, match="duplicate JSON key.*schema_version"):
        load_runner_freeze_file(path)


def test_runner_freeze_file_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(pretty_json_bytes(_runner_freeze_value()))
    path = tmp_path / "runner.json"
    path.symlink_to(target.name)
    with pytest.raises(ValueError, match="missing or unsafe"):
        load_runner_freeze_file(path)
