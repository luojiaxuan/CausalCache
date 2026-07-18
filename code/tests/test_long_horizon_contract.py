from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from causalcache.long_horizon_contract import (
    CANONICAL_CONFIG_PATH,
    FROZEN_CONFIG_SHA256,
    LongHorizonContract,
    REQUIRED_SOURCE_A_PATHS,
    RUNNER_FREEZE_B_PATH,
    SOURCE_A_INVENTORY_PLACEHOLDER,
    historical_role_inventory,
    load_strict_json_bytes,
    validate_contract,
    validate_source_a,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_value() -> dict:
    return json.loads((ROOT / CANONICAL_CONFIG_PATH).read_text(encoding="utf-8"))


def test_canonical_source_a_contract_is_frozen_and_unauthorized() -> None:
    contract = LongHorizonContract.load(repository_root=ROOT)
    assert contract.sha256 == FROZEN_CONFIG_SHA256
    assert contract.selection["trajectory_salt"] == (
        "causalcache-long-horizon-development-v1"
    )
    assert set(contract.data["authorization"].values()) == {False}
    assert (
        contract.data["source_freeze"]["source_a_inventory_sha256"]
        == SOURCE_A_INVENTORY_PLACEHOLDER
    )
    result = validate_source_a(repository_root=ROOT)
    assert result["runner_freeze_b_present"] is False
    assert result["historical_role_union_count"] == 107
    assert result["policy_load_or_forward_count"] == 0
    assert len(result["source_a_inventory"]) == len(REQUIRED_SOURCE_A_PATHS)


def test_historical_ids_are_derived_from_bound_role_manifests() -> None:
    contract = LongHorizonContract.load(repository_root=ROOT)
    inventory = historical_role_inventory(contract.data, repository_root=ROOT)
    assert inventory["union_source_id_count"] == 107
    assert len(inventory["manifests"]) == 3
    assert [record["source_id_count"] for record in inventory["manifests"]] == [
        23,
        43,
        64,
    ]
    config_text = (ROOT / CANONICAL_CONFIG_PATH).read_text(encoding="utf-8")
    for source_id in inventory["union_source_ids"]:
        assert source_id not in config_text


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["source_pool"]["structural_eligibility"].update(
            {"minimum_decisions_per_trajectory": 17}
        ),
        lambda value: value["selection"].update({"padding_allowed": True}),
        lambda value: value["selection"].update(
            {"development_trajectory_count": 25}
        ),
        lambda value: value["state_geometry"]["states"][1].update(
            {"candidate_event_count": 15}
        ),
        lambda value: value["reference_contract"]["n16"].update(
            {"global_oracle_or_search_claim_allowed": True}
        ),
        lambda value: value["reference_contract"]["n16"].update(
            {"enumeration_or_shortlist_allowed": True}
        ),
        lambda value: value["selector_matrix"].update(
            {"budget_4": ["restoration_independent_gate", "v1_conditional"]}
        ),
        lambda value: value["authorization"].update(
            {"development_restoration_access": True}
        ),
        lambda value: value["source_freeze"].update(
            {"source_a_inventory_sha256": "1" * 64}
        ),
        lambda value: value["policy_context_profile"].update(
            {"n16_full_reference_allowed": True}
        ),
        lambda value: value["execution_b_plan"].update({"worker_count": 3}),
        lambda value: value["artifact_plan"].update({"visibility": "public"}),
        lambda value: value["operation_accounting"].update(
            {"n8_budget_4_total_requested_unique_subset_rows": 3911}
        ),
        lambda value: value["learned_model_artifacts"][
            "formal58_base_and_conditional"
        ].update({"revision": "0" * 40}),
        lambda value: value["learned_model_artifacts"][
            "v4_safe_frozen_base_residual"
        ]["residual_checkpoints"][0].update({"sha256": "0" * 64}),
    ],
)
def test_contract_rejects_geometry_selector_search_or_access_drift(mutate) -> None:
    value = _load_value()
    mutate(value)
    with pytest.raises(ValueError):
        validate_contract(value, repository_root=ROOT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["validity_and_go_contract"][
            "canonical_action_validity"
        ].update({"independent_parse_repetitions": 1}),
        lambda value: value["validity_and_go_contract"][
            "canonical_action_validity"
        ].update({"invalid_trajectory_top_up_allowed": True}),
        lambda value: value["validity_and_go_contract"][
            "policy_coverage_thresholds"
        ].update({"minimum_valid_n8_trajectories": 17}),
        lambda value: value["validity_and_go_contract"][
            "policy_coverage_thresholds"
        ].update({"overall_go_claim_allowed_below_any_threshold": True}),
        lambda value: value["validity_and_go_contract"][
            "fixed_denominator_reduction"
        ].update({"invalid_state_contribution": 1.0}),
        lambda value: value["validity_and_go_contract"][
            "fixed_denominator_reduction"
        ].update({"bootstrap_uses_all_selected_trajectories": False}),
        lambda value: value["validity_and_go_contract"]["normalization"].update(
            {"zero_threshold": 1e-9}
        ),
        lambda value: value["validity_and_go_contract"][
            "set_aware_vs_independent_development_go"
        ].update({"n8_trajectory_equal_mean_normalized_delta_minimum": 0.0}),
        lambda value: value["validity_and_go_contract"][
            "set_aware_vs_independent_development_go"
        ].update({"n8_minimum_positive_trajectories": 13}),
        lambda value: value["validity_and_go_contract"][
            "set_aware_vs_independent_development_go"
        ].update(
            {
                "corresponding_n16_pair_mean_normalized_delta_must_be_strictly_positive": False
            }
        ),
        lambda value: value["validity_and_go_contract"][
            "independent_long_horizon_development_go"
        ].update({"n8_trajectory_equal_mean_raw_utility_ratio_to_exact_minimum": 0.84}),
        lambda value: value["validity_and_go_contract"][
            "independent_long_horizon_development_go"
        ].update({"budget_2_is_secondary_report_only": False}),
        lambda value: value["validity_and_go_contract"][
            "independent_long_horizon_development_go"
        ].update({"old_closed_loop_or_test_direct_unlock_allowed": True}),
    ],
)
def test_contract_rejects_policy_validity_or_go_threshold_drift(mutate) -> None:
    value = _load_value()
    mutate(value)
    with pytest.raises(ValueError):
        validate_contract(value, repository_root=ROOT)


def test_source_a_load_rejects_present_execution_b_runner(tmp_path: Path) -> None:
    paths = [
        CANONICAL_CONFIG_PATH,
        "code/configs/gui_owl_1_5_8b_snapshot.json",
        "code/configs/independent_reference_gate_v1.json",
        "code/configs/restoration_v2_ocr_backend.json",
        "data/manifests/independent_reference_gate_v1_source_files.json",
        "data/manifests/independent_reference_gate_v1_artifact.json",
        "data/manifests/restoration_v2_ocr_backend.json",
        "data/manifests/restoration_v2_selection.json",
        "data/manifests/restoration_v2_2_label_expansion_selection.json",
    ]
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_FREEZE_B_PATH
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be absent"):
        LongHorizonContract.load(repository_root=tmp_path)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_strict_json_bytes(b'{"x":1,"x":2}', label="fixture")
    with pytest.raises(ValueError, match="non-finite"):
        load_strict_json_bytes(b'{"x":NaN}', label="fixture")


def test_contract_mutation_does_not_modify_canonical_fixture() -> None:
    before = (ROOT / CANONICAL_CONFIG_PATH).read_bytes()
    value = copy.deepcopy(_load_value())
    value["selection"]["trajectory_salt"] = "changed"
    with pytest.raises(ValueError):
        validate_contract(value, repository_root=ROOT)
    assert (ROOT / CANONICAL_CONFIG_PATH).read_bytes() == before
