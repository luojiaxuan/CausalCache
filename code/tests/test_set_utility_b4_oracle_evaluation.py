from __future__ import annotations

import hashlib
import itertools

from causalcache.set_utility_b4_oracle_evaluation import (
    COMPLETED_B4_EVALUATION,
    evaluate_b4_oracle,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes


def _fixture() -> tuple[dict, dict, dict]:
    candidates = (1, 2, 3, 4, 5)
    methods = {
        "deepsets": {"1": [1], "2": [1, 2], "3": [1, 2], "4": [1, 2, 3, 4]},
        "set_transformer": {"1": [2], "2": [2, 3], "3": [2, 3], "4": [1, 2, 3, 4]},
        "recent": {"1": [5], "2": [4, 5], "3": [3, 4, 5], "4": [2, 3, 4, 5]},
        "ocr_rgb": {"1": [1], "2": [1, 3], "3": [1, 3, 5], "4": [1, 2, 3, 4]},
        "random": {"1": [3], "2": [3, 4], "3": [2, 3, 4], "4": [1, 2, 3, 4]},
    }
    selections: dict = {
        "records": [
            {
                "candidate_event_ids": list(candidates),
                "methods": methods,
                "state_id": "t1:decision:006",
                "tracks": ["exact_oracle"],
                "trajectory_id": "t1",
            }
        ]
    }
    selections["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(selections)
    ).hexdigest()
    distances = {
        subset: float(10 - len(subset))
        for cardinality in range(5)
        for subset in itertools.combinations(candidates, cardinality)
    }
    distances[(1,)] = 6.0
    distances[(2,)] = 7.0
    distances[(3,)] = 7.0
    distances[(1, 2)] = 5.0
    distances[(1, 3)] = 5.0
    distances[(2, 3)] = 1.0
    distances[(1, 2, 3, 4)] = 0.5
    distances[candidates] = 0.0
    terminal = {
        "candidate_event_step_ids": list(candidates),
        "distance_rows": [
            {"coalition_event_step_ids": list(subset), "distance": distance}
            for subset, distance in sorted(distances.items(), key=lambda item: (len(item[0]), item[0]))
        ],
        "execution_config_sha256": "e" * 64,
        "role": "evaluation",
        "scientific_config_sha256": "s" * 64,
        "source_revision": "r" * 40,
        "state_id": "t1:decision:006",
        "status": "COMPLETED_VARIABLE_HISTORY_LABEL_STATE",
        "trajectory_id": "t1",
    }
    config = {
        "evaluation": {"budgets": [1, 2, 3, 4], "normalization_floor": 0.01},
        "inputs": {"sealed_selections": {"content_sha256": selections["content_sha256"]}},
        "reference": {"scientific_config_sha256": "s" * 64},
        "state_selection": {
            "expected_state_count": 1,
            "expected_trajectory_count": 1,
            "maximum_candidate_count": 5,
            "minimum_candidate_count": 5,
            "required_track": "exact_oracle",
        },
    }
    return config, selections, {terminal["state_id"]: terminal}


def test_b4_evaluation_separates_exact_greedy_and_distillation_gaps() -> None:
    config, selections, terminals = _fixture()
    result = evaluate_b4_oracle(
        config=config,
        config_sha256="c" * 64,
        execution_config_sha256="e" * 64,
        selections=selections,
        terminals=terminals,
        source_revision="r" * 40,
    )
    assert result["status"] == COMPLETED_B4_EVALUATION
    assert result["coverage"]["completed_state_count"] == 1
    record = result["state_records"][0]
    assert record["exact"]["2"]["selected_event_ids"] == [2, 3]
    assert record["exact"]["2"]["normalized_recovery"] == 0.9
    assert record["true_conditional_greedy"]["2"]["selected_event_ids"] == [1, 2]
    assert record["true_conditional_greedy"]["2"]["normalized_recovery"] == 0.5
    assert result["diagnostics"]["exact_minus_true_greedy_normalized_recovery"]["2"]["mean"] == 0.4
    assert result["diagnostics"]["distillation_gap_exact_minus_method_normalized_recovery"]["set_transformer"]["2"]["mean"] == 0.0
