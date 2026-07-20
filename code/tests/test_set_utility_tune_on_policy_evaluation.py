from __future__ import annotations

import hashlib

from causalcache.set_utility_heldout_inference import canonical_json_bytes
from causalcache.set_utility_tune_on_policy_evaluation import evaluate_tune_on_policy


def _selection(name: str, learned: dict[str, list[int]]) -> dict:
    payload = {
        "cache_content_sha256": "cache",
        "checkpoint_sha256": f"checkpoint-{name}",
        "config_sha256": "config",
        "input_content_sha256": "input",
        "records": [
            {
                "candidate_event_ids": [1, 2, 3, 4, 5],
                "latency_ms": {
                    "conditioning": 1.0,
                    "event_source_encoding": 2.0,
                    "query_source_encoding": 1.0,
                    "search": 2.0,
                },
                "learned": learned,
                "logical_shard": 0,
                "predicted_utilities": {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0},
                "recent": {"1": [5], "2": [4, 5], "3": [3, 4, 5], "4": [2, 3, 4, 5]},
                "state_id": "trajectory:decision:006",
                "subset_score_count": 15,
                "trajectory_id": "trajectory",
            }
        ],
        "schema_version": "1.0.0",
        "status": "COMPLETED_SET_UTILITY_TUNE_SELECTIONS",
        "variant": f"{name}_contextual_frozen_variant",
    }
    payload["content_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def test_tune_on_policy_evaluation_selects_true_winner() -> None:
    set_transformer = {"1": [1], "2": [1, 2], "3": [1, 2], "4": [1, 2]}
    deepsets = {"1": [2], "2": [2, 3], "3": [2, 3], "4": [2, 3]}
    terminal = {
        "candidate_event_step_ids": [1, 2, 3, 4, 5],
        "distance_rows": [
            {"coalition_event_step_ids": [], "distance": 1.0},
            {"coalition_event_step_ids": [1], "distance": 0.4},
            {"coalition_event_step_ids": [1, 2], "distance": 0.1},
            {"coalition_event_step_ids": [2], "distance": 0.7},
            {"coalition_event_step_ids": [2, 3], "distance": 0.5},
            {"coalition_event_step_ids": [5], "distance": 0.8},
            {"coalition_event_step_ids": [4, 5], "distance": 0.7},
            {"coalition_event_step_ids": [3, 4, 5], "distance": 0.6},
            {"coalition_event_step_ids": [2, 3, 4, 5], "distance": 0.4},
            {"coalition_event_step_ids": [1, 2, 3, 4, 5], "distance": 0.0},
        ],
        "role": "tune",
        "state_id": "trajectory:decision:006",
        "status": "COMPLETED_VARIABLE_HISTORY_LABEL_STATE",
        "trajectory_id": "trajectory",
    }
    result = evaluate_tune_on_policy(
        selections={
            "deepsets": _selection("deepsets", deepsets),
            "set_transformer": _selection("set_transformer", set_transformer),
        },
        terminals={terminal["state_id"]: terminal},
        normalization_floor=0.01,
        bootstrap_resamples=100,
        bootstrap_seed=7,
        bootstrap_interval=0.95,
    )
    assert result["coverage"]["complete"] is True
    assert result["winner"]["model"] == "set_transformer"
    assert (
        result["method_summaries"]["set_transformer"]
        ["primary_macro_B1_B4_trajectory_equal_normalized_recovery"]["mean"]
        == 0.825
    )
