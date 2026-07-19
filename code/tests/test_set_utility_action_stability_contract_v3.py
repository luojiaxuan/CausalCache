from __future__ import annotations

import copy
from pathlib import Path

import pytest

from causalcache.set_utility_action_stability_contract_v3 import (
    build_action_stability_source_v3_config_skeleton,
    load_action_stability_source_v3_contract,
    validate_action_stability_source_v3_config,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_d2_source_config_equals_live_skeleton() -> None:
    contract = load_action_stability_source_v3_contract(repository_root=ROOT)
    receipt = validate_action_stability_source_v3_config(
        contract.data, repository_root=ROOT
    )
    assert receipt["status"] == "VALID_STRICT_DETERMINISM_ACTION_STABILITY_D2_SOURCE"
    assert receipt["state_count"] == 3
    assert receipt["encode_call_ceiling"] == 3
    assert receipt["generation_call_ceiling"] == 6


def test_source_contract_rejects_post_hoc_determinism_or_gate_changes() -> None:
    config = build_action_stability_source_v3_config_skeleton(repository_root=ROOT)
    mutated = copy.deepcopy(config)
    mutated["diagnostic"]["condition"]["deterministic_algorithms"] = False
    with pytest.raises(ValueError, match="differs from the live skeleton"):
        validate_action_stability_source_v3_config(mutated, repository_root=ROOT)
    mutated = copy.deepcopy(config)
    mutated["preregistered_interpretation"][
        "pass_does_not_unlock_labels_training_matched_nll_or_closed_loop"
    ] = False
    with pytest.raises(ValueError, match="differs from the live skeleton"):
        validate_action_stability_source_v3_config(mutated, repository_root=ROOT)
