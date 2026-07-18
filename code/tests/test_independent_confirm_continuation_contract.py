from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from causalcache.independent_confirm_continuation_contract import (
    CONFIG_PATH,
    EXPECTED_PAYLOAD_COMMIT,
    IndependentConfirmContinuationContract,
    canonical_json_bytes,
    continuation_topology_nonce,
    validate_contract,
    validate_runner_freeze,
)


ROOT = Path(__file__).resolve().parents[2]


def test_loads_frozen_continuation_source_a_without_authorization() -> None:
    contract = IndependentConfirmContinuationContract.load(
        ROOT / CONFIG_PATH,
        repository_root=ROOT,
        require_runner_absent=True,
    )

    assert contract.payload["payload_commit"] == EXPECTED_PAYLOAD_COMMIT
    assert contract.data["failed_attempt"]["reference_policy_output_count"] == 0
    assert contract.data["failed_attempt"]["restoration_output_count"] == 0
    assert contract.continuation["runtime_continuation_count"] == 1
    assert contract.continuation["scientific_retry_count"] == 0
    assert set(contract.data["authorization"].values()) == {False}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["scientific_contract"].update(
            {"threshold_data_model_seed_comparator_or_go_change_allowed": True}
        ),
        lambda value: value["sealed_payload"].update(
            {"payload_commit": "0" * 40}
        ),
        lambda value: value["continuation_contract"].update(
            {"scientific_retry_count": 1}
        ),
        lambda value: value["authorization"].update(
            {"restoration_access_authorized": True}
        ),
        lambda value: value["scientific_contract"].update(
            {"threshold_data_model_seed_comparator_or_go_change_allowed": 0}
        ),
        lambda value: value["failed_attempt"].update(
            {"reference_policy_output_count": False}
        ),
        lambda value: value["sealed_payload"].update({"private": 1}),
        lambda value: value["sealed_payload"].update(
            {"remote_file_count_before_continuation": 9.0}
        ),
        lambda value: value["continuation_contract"].update(
            {"runtime_continuation_count": True}
        ),
        lambda value: value["source_freeze"].update({"unexpected": False}),
    ],
)
def test_rejects_science_payload_retry_or_authorization_drift(mutate) -> None:
    value = json.loads((ROOT / CONFIG_PATH).read_text())
    mutate(value)

    with pytest.raises(ValueError):
        validate_contract(copy.deepcopy(value), repository_root=ROOT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["runtime_contract"].update({"gpu_count": 4.0}),
        lambda value: value["runtime_contract"].update({"host_aliases": ["hyper00"]}),
        lambda value: value["runtime_contract"].update({"unexpected": False}),
        lambda value: value["runtime_contract"].pop("python_version"),
        lambda value: value["output_contract"].update(
            {"report_targets_and_tag_identical_to_parent": False}
        ),
        lambda value: value["output_contract"].update({"unexpected": False}),
        lambda value: value["output_contract"].pop("failure_status"),
        lambda value: value["authorization"].update({"unexpected": False}),
        lambda value: value["authorization"].pop("matched_nll_authorized"),
    ],
)
def test_rejects_runtime_output_or_authorization_exact_contract_drift(mutate) -> None:
    value = json.loads((ROOT / CONFIG_PATH).read_text())
    mutate(value)

    with pytest.raises(ValueError):
        validate_contract(copy.deepcopy(value), repository_root=ROOT)


def test_runner_freeze_requires_source_bound_fresh_topology_nonce() -> None:
    contract = IndependentConfirmContinuationContract.load(
        ROOT / CONFIG_PATH,
        repository_root=ROOT,
        require_runner_absent=True,
    )
    source_a = "a" * 40
    source_inventory = [
        {"path": path, "sha256": "b" * 64, "size_bytes": 1}
        for path in contract.source["required_source_a_paths"]
    ]
    module_inventory = [
        {"path": path, "sha256": "c" * 64, "size_bytes": 1}
        for path in sorted(
            f"code/{name.replace('.', '/')}.py"
            for name in contract.source["required_execution_modules"]
        )
    ]
    source_inventory_sha256 = hashlib.sha256(
        canonical_json_bytes(source_inventory)
    ).hexdigest()
    module_inventory_sha256 = hashlib.sha256(
        canonical_json_bytes(module_inventory)
    ).hexdigest()
    value = {
        "schema_version": "1.0.0",
        "protocol_id": contract.data["protocol_id"],
        "status": "FROZEN_CAUSALCACHE_INDEPENDENT_CONFIRM_CONTINUATION_EXECUTION_B_V1",
        "source_a_commit": source_a,
        "source_a_parent_count": 1,
        "source_a_remote_main": source_a,
        "contract_sha256": contract.sha256,
        "source_a_inventory": source_inventory,
        "source_a_inventory_sha256": source_inventory_sha256,
        "loaded_module_inventory": module_inventory,
        "loaded_module_inventory_sha256": module_inventory_sha256,
        "fresh_topology_receipt_required": True,
        "topology_receipt_nonce": continuation_topology_nonce(
            source_a_commit=source_a,
            contract_sha256=contract.sha256,
        ),
        "adopt_existing_payload_authorized": True,
        "restoration_access_requires_adopted_payload_receipt": True,
        "selector_reexecution_authorized": False,
        "closed_loop_requires_confirm_go": True,
    }
    validate_runner_freeze(contract, value, source_a_commit=source_a)

    tampered = copy.deepcopy(value)
    tampered["topology_receipt_nonce"] = "0" * 64
    with pytest.raises(ValueError, match="runner freeze"):
        validate_runner_freeze(contract, tampered, source_a_commit=source_a)

    for field, replacement in (
        ("source_a_parent_count", True),
        (
            "source_a_inventory",
            [
                {
                    **item,
                    "size_bytes": 1.0 if index == 0 else item["size_bytes"],
                }
                for index, item in enumerate(source_inventory)
            ],
        ),
    ):
        tampered = copy.deepcopy(value)
        tampered[field] = replacement
        if field == "source_a_inventory":
            tampered["source_a_inventory_sha256"] = hashlib.sha256(
                canonical_json_bytes(replacement)
            ).hexdigest()
        with pytest.raises(ValueError, match="runner freeze|inventory"):
            validate_runner_freeze(contract, tampered, source_a_commit=source_a)
