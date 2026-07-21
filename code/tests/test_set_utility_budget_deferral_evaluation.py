from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from causalcache.set_utility_budget_deferral_evaluation import (
    SEAL_STATUS,
    SELECTION_STATUS,
    budget_deferral_methods,
    build_selection_payload,
    evaluation_slice_memberships,
    project_label_blind_evaluation_state,
    verify_selection_and_seal,
    write_selection_and_seal,
)


class AccessGuard(Mapping[str, Any]):
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values
        self.accessed: list[str] = []

    def __getitem__(self, key: str) -> Any:
        if key in {"distance_rows", "restoration_truth"}:
            raise AssertionError("selector accessed evaluation truth")
        self.accessed.append(key)
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def _config() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "status": "FROZEN_BEFORE_CANDIDATE_EVALUATION",
        "predictor": {
            "model_family": "deepsets_structured_marginal",
            "variant": "structured",
            "selected_epoch": 1,
            "checkpoint_sha256": "1" * 64,
            "stop_semantics": "learned_threshold",
        },
        "deployment": {
            "budgets": [1, 2, 3, 4],
            "selection_by_budget": {
                "1": "recent",
                "2": "recent",
                "3": "deepsets_direct_conditional_marginal",
                "4": "deepsets_direct_conditional_marginal",
            },
            "selection_cardinality": "at_most_B",
            "cross_budget_nestedness_required": False,
            "within_budget_iterative_selection": True,
            "post_evaluation_route_or_threshold_changes_allowed": False,
        },
        "frozen_candidate_evaluation": {
            "role": "evaluation",
            "state_count": 4,
            "state_inventory_sha256": "2" * 64,
            "historical_access_boundary": {
                "previously_consumed_state_count": 2,
                "previously_consumed_trajectory_count": 1,
                "state_new_count": 2,
                "trajectory_new_state_count": 1,
                "trajectory_new_trajectory_count": 1,
            },
            "label_blind_selection_must_be_sealed_before_truth_access": True,
        },
        "firewall": {
            "evaluation_features_allowed_after_this_freeze": True,
            "evaluation_truth_allowed_only_after_signed_selection_seal": True,
            "evaluation_truth_for_training_or_calibration": False,
        },
    }


def _feature_state(state_id: str = "t1:decision:006") -> dict[str, Any]:
    return {
        "candidate_event_step_ids": [1, 2, 3, 4, 5],
        "current_image_key": f"{state_id}:current",
        "event_image_keys": [f"{state_id}:image:{index}" for index in range(5)],
        "event_numeric_features": [[float(index), 1.0] for index in range(5)],
        "event_text_keys": [f"{state_id}:text:{index}" for index in range(5)],
        "instruction_text_key": f"{state_id}:instruction",
        "logical_shard": 3,
        "role": "evaluation",
        "state_id": state_id,
        "trajectory_id": state_id.split(":", 1)[0],
    }


def _states() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "state_id": state_id,
            "trajectory_id": trajectory_id,
        }
        for state_id, trajectory_id in (
            ("t1:decision:006", "t1"),
            ("t1:decision:007", "t1"),
            ("t1:decision:008", "t1"),
            ("t2:decision:006", "t2"),
        )
    )


def test_projection_is_label_blind_and_rejects_embedded_truth() -> None:
    guarded = AccessGuard(_feature_state())
    projected = project_label_blind_evaluation_state(guarded)
    assert projected["history_bin"] == "short"
    assert projected["logical_shard"] == 3
    assert "distance_rows" not in guarded.accessed
    tainted = _feature_state()
    tainted["distance_rows"] = [{"distance": 0.1}]
    with pytest.raises(ValueError, match="forbidden truth fields"):
        project_label_blind_evaluation_state(tainted)


def test_slice_membership_uses_only_frozen_historical_state_ids() -> None:
    inventory = {
        "status": "FROZEN_VARIABLE_HISTORY_STATE_INVENTORY",
        "state_identity_sha256": "2" * 64,
        "evaluation_tracks": {
            "exact_state_ids": ["t1:decision:006"],
            "large_history_state_ids": ["t1:decision:007"],
        },
    }
    membership, counts = evaluation_slice_memberships(
        _states(), inventory=inventory, config=_config()
    )
    assert membership == {
        "t1:decision:006": ("all",),
        "t1:decision:007": ("all",),
        "t1:decision:008": ("all", "state_new"),
        "t2:decision:006": ("all", "state_new", "trajectory_new"),
    }
    assert counts == {
        "all": {"state_count": 4, "trajectory_count": 2},
        "state_new": {"state_count": 2, "trajectory_count": 2},
        "trajectory_new": {"state_count": 1, "trajectory_count": 1},
    }


def test_budget_deferral_is_recent_for_b1_b2_and_learned_for_b3_b4() -> None:
    methods = budget_deferral_methods(
        (1, 2, 3, 4, 5),
        learned_selections={
            "1": [1],
            "2": [1, 2],
            "3": [1, 3],
            "4": [2, 4, 5],
        },
    )
    assert methods["recent"] == {
        "1": [5],
        "2": [4, 5],
        "3": [3, 4, 5],
        "4": [2, 3, 4, 5],
    }
    assert methods["structured_deepsets_budget_deferral"] == {
        "1": [5],
        "2": [4, 5],
        "3": [1, 3],
        "4": [2, 4, 5],
    }


def test_selection_and_seal_bind_file_content_and_slice_denominator(
    tmp_path: Path,
) -> None:
    methods = budget_deferral_methods(
        (1, 2, 3, 4, 5),
        learned_selections={
            "1": [1],
            "2": [1, 2],
            "3": [1, 2, 3],
            "4": [1, 2, 3, 4],
        },
    )
    records = []
    membership = {
        "t1:decision:006": ["all"],
        "t1:decision:007": ["all"],
        "t1:decision:008": ["all", "state_new"],
        "t2:decision:006": ["all", "state_new", "trajectory_new"],
    }
    for state_id, slices in membership.items():
        records.append(
            {
                "candidate_event_ids": [1, 2, 3, 4, 5],
                "history_bin": "short",
                "latency_ms": {"learned_search": 1.25},
                "logical_shard": 3,
                "methods": methods,
                "slices": slices,
                "state_id": state_id,
                "trajectory_id": state_id.split(":", 1)[0],
            }
        )
    bindings = {
        "cache_content_sha256": "0" * 64,
        "cache_manifest_file_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "config_content_sha256": "3" * 64,
        "config_file_sha256": "4" * 64,
        "input_content_sha256": "5" * 64,
        "input_manifest_file_sha256": "6" * 64,
        "input_states_sha256": "7" * 64,
        "predictor_config_file_sha256": "8" * 64,
        "state_inventory_content_sha256": "9" * 64,
        "state_inventory_file_sha256": "a" * 64,
    }
    slices = {
        "all": {"state_count": 4, "trajectory_count": 2},
        "state_new": {"state_count": 2, "trajectory_count": 2},
        "trajectory_new": {"state_count": 1, "trajectory_count": 1},
    }
    selection = build_selection_payload(
        records, bindings=bindings, slice_counts=slices
    )
    selection_path = tmp_path / "selections.json"
    seal_path = tmp_path / "selection-seal.json"
    seal = write_selection_and_seal(
        selection, selection_path=selection_path, seal_path=seal_path
    )
    loaded, loaded_seal = verify_selection_and_seal(selection_path, seal_path)
    assert loaded["status"] == SELECTION_STATUS
    assert loaded["truth_accessed"] is False
    assert loaded["denominator"]["slice_counts"] == slices
    assert loaded_seal["status"] == SEAL_STATUS
    assert loaded_seal["bindings"] == loaded["bindings"]
    assert loaded_seal["denominator"] == loaded["denominator"]
    assert seal["selection_content_sha256"] == loaded["content_sha256"]

    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    payload["records"][0]["methods"]["recent"]["1"] = [1]
    selection_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="drifted"):
        verify_selection_and_seal(selection_path, seal_path)
