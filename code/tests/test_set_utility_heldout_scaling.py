from __future__ import annotations

import hashlib

import pytest

from causalcache.set_utility_heldout_inference import canonical_json_bytes
from causalcache.set_utility_heldout_scaling import (
    COMPLETED_SCALING_DIAGNOSTIC,
    COMPLETED_SCALING_DIAGNOSTIC_WITH_SKIPS,
    evaluate_scaling_exact_track,
)


def _config() -> dict[str, object]:
    return {
        "state_inventory": {"exact_state_count": 2, "union_state_count": 2},
        "representation": {"normalization_floor": 0.01},
        "frozen_candidates": {
            "models": {
                "lc10_deepsets": {
                    "best_tune_total": 0.3,
                    "family": "deepsets",
                    "fraction": 0.1,
                    "sha256": "a",
                    "train_state_count": 10,
                    "train_trajectory_count": 1,
                },
                "lc100_deepsets": {
                    "best_tune_total": 0.2,
                    "family": "deepsets",
                    "fraction": 1.0,
                    "sha256": "b",
                    "train_state_count": 100,
                    "train_trajectory_count": 10,
                },
            }
        },
        "diagnostic_evaluation": {
            "budgets": [1, 2],
            "bootstrap": {"interval": 0.95, "resamples": 100, "seed": 7},
        },
        "claim_boundary": {"may_authorize_policy_replay_or_closed_loop": False},
    }


def _selection(name: str, checkpoint: str, chosen: int) -> dict[str, object]:
    records = []
    for trajectory in ("t1", "t2"):
        methods = {
            name: {"1": [chosen], "2": [1, 2], "3": [1, 2], "4": [1, 2]},
            "ocr_rgb": {str(budget): [2] for budget in range(1, 5)},
            "random": {str(budget): [1] for budget in range(1, 5)},
            "recent": {str(budget): [2] for budget in range(1, 5)},
        }
        records.append(
            {
                "candidate_event_ids": [1, 2],
                "methods": methods,
                "state_id": f"{trajectory}:decision:003",
                "tracks": ["exact_oracle"],
                "trajectory_id": trajectory,
            }
        )
    payload: dict[str, object] = {
        "checkpoint_sha256": checkpoint,
        "config_sha256": "config",
        "model_name": name,
        "records": records,
        "status": "COMPLETED_SET_UTILITY_HELDOUT_MODEL_SELECTIONS",
    }
    payload["content_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _terminal(trajectory: str) -> dict[str, object]:
    return {
        "candidate_event_step_ids": [1, 2],
        "distance_rows": [
            {"coalition_event_step_ids": [], "distance": 10.0},
            {"coalition_event_step_ids": [1], "distance": 2.0},
            {"coalition_event_step_ids": [2], "distance": 8.0},
            {"coalition_event_step_ids": [1, 2], "distance": 0.0},
        ],
        "state_id": f"{trajectory}:decision:003",
        "status": "COMPLETED_VARIABLE_HISTORY_LABEL_STATE",
        "trajectory_id": trajectory,
    }


def test_scaling_diagnostic_uses_exact_truth_and_trajectory_equal_metrics() -> None:
    result = evaluate_scaling_exact_track(
        config=_config(),
        config_sha256="config",
        selections={
            "lc10_deepsets": _selection("lc10_deepsets", "a", 2),
            "lc100_deepsets": _selection("lc100_deepsets", "b", 1),
        },
        terminals={
            "t1:decision:003": _terminal("t1"),
            "t2:decision:003": _terminal("t2"),
        },
    )
    assert result["status"] == COMPLETED_SCALING_DIAGNOSTIC
    assert result["coverage"]["completed_exact_state_count"] == 2
    assert result["scaling"]["deepsets"]["smallest_to_largest_delta"] == pytest.approx(
        0.3
    )
    assert result["scaling"]["deepsets"]["monotonic_nondecreasing"] is True
    assert result["scaling"]["deepsets"]["heldout_best_candidate"] == "lc100_deepsets"


def test_scaling_diagnostic_retains_preexisting_exact_track_skip() -> None:
    terminals = {
        "t1:decision:003": _terminal("t1"),
        "t2:decision:003": {
            "failure_class": "ParseFailure",
            "state_id": "t2:decision:003",
            "status": "SKIPPED_VARIABLE_HISTORY_LABEL_STATE",
        },
    }
    result = evaluate_scaling_exact_track(
        config=_config(),
        config_sha256="config",
        selections={
            "lc10_deepsets": _selection("lc10_deepsets", "a", 2),
            "lc100_deepsets": _selection("lc100_deepsets", "b", 1),
        },
        terminals=terminals,
    )
    assert result["status"] == COMPLETED_SCALING_DIAGNOSTIC_WITH_SKIPS
    assert result["coverage"]["skipped_exact_state_count"] == 1


def test_scaling_diagnostic_rejects_unbound_checkpoint() -> None:
    selections = {
        "lc10_deepsets": _selection("lc10_deepsets", "wrong", 2),
        "lc100_deepsets": _selection("lc100_deepsets", "b", 1),
    }
    with pytest.raises(ValueError, match="invalid scaling selection payload"):
        evaluate_scaling_exact_track(
            config=_config(),
            config_sha256="config",
            selections=selections,
            terminals={
                "t1:decision:003": _terminal("t1"),
                "t2:decision:003": _terminal("t2"),
            },
        )
