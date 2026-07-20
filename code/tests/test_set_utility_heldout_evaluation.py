from __future__ import annotations

import hashlib
import math

import pytest

from causalcache.set_utility_heldout_evaluation import (
    COMPLETED_EVALUATION,
    INCOMPLETE_EVALUATION,
    canonical_json_bytes,
    evaluate_heldout,
    percentile_type7,
)


def _selection() -> dict[str, object]:
    records = []
    for ordinal, (trajectory, step, tracks) in enumerate(
        (
            ("t1", 6, ["exact_oracle", "large_history"]),
            ("t2", 10, ["exact_oracle", "large_history"]),
            ("t1", 18, ["large_history"]),
            ("t2", 34, ["large_history"]),
        )
    ):
        events = list(range(1, step))
        choices = {
            "deepsets": {"1": [2], "2": [2, 3], "3": [2, 3], "4": [2, 3]},
            "set_transformer": {
                "1": [1],
                "2": [1, 2],
                "3": [1, 2, 3],
                "4": [1, 2, 3, 4],
            },
            "ocr_rgb": {str(b): [4] for b in range(1, 5)},
            "random": {str(b): [3] for b in range(1, 5)},
            "recent": {str(b): [events[-1]] for b in range(1, 5)},
        }
        records.append(
            {
                "candidate_event_ids": events,
                "logical_shard": ordinal,
                "methods": choices,
                "model_metrics": {
                    model: {
                        "latency_ms": {
                            "selector_total_from_cached_source_tokens": latency
                        },
                        "predicted_utilities": {str(b): float(b) for b in range(1, 5)},
                        "subset_score_count": 5,
                    }
                    for model, latency in (("deepsets", 2.0), ("set_transformer", 3.0))
                },
                "state_id": f"{trajectory}:decision:{step:03d}",
                "tracks": tracks,
                "trajectory_id": trajectory,
            }
        )
    result: dict[str, object] = {
        "cache_content_sha256": "c",
        "config_sha256": "d",
        "input_content_sha256": "i",
        "inventory_sha256": "v",
        "model_artifacts": {
            "deepsets": {"checkpoint_sha256": "a"},
            "set_transformer": {"checkpoint_sha256": "b"},
        },
        "records": records,
        "schema_version": "1.0.0",
        "status": "SEALED_SET_UTILITY_HELDOUT_SELECTIONS",
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _terminals(selection: dict[str, object]) -> dict[str, dict[str, object]]:
    terminals = {}
    for record in selection["records"]:
        candidates = tuple(record["candidate_event_ids"])
        subsets = {(), candidates}
        if "exact_oracle" in record["tracks"]:
            subsets.update((event,) for event in candidates)
            subsets.update(
                (left, right)
                for left in candidates
                for right in candidates
                if left < right
            )
        for method in record["methods"].values():
            subsets.update(tuple(value) for value in method.values())

        def distance(subset: tuple[int, ...]) -> float:
            if subset == candidates:
                return 0.0
            if subset == (1, 2):
                return 1.0
            if subset == (1, 2, 3):
                return 0.8
            if subset == (1, 2, 3, 4):
                return 0.6
            if subset == (1,):
                return 3.0
            if subset == (2, 3):
                return 5.0
            if subset == (2,):
                return 6.0
            if subset == (4,):
                return 7.0
            if subset and subset[-1] == candidates[-1]:
                return 8.0
            return 9.0 if subset else 10.0

        state_id = record["state_id"]
        terminals[state_id] = {
            "candidate_event_step_ids": list(candidates),
            "distance_rows": [
                {
                    "coalition_event_step_ids": list(subset),
                    "distance": distance(subset),
                }
                for subset in sorted(subsets, key=lambda item: (len(item), item))
            ],
            "role": "evaluation",
            "scientific_config_sha256": "f" * 64,
            "state_id": state_id,
            "status": "COMPLETED_VARIABLE_HISTORY_LABEL_STATE",
            "trajectory_id": record["trajectory_id"],
        }
    return terminals


def _evaluate(selection: dict[str, object], terminals: dict[str, dict[str, object]]):
    return evaluate_heldout(
        selections=selection,
        terminals=terminals,
        budgets=(1, 2, 3, 4),
        normalization_floor=0.01,
        bootstrap_resamples=200,
        bootstrap_seed=7,
        bootstrap_interval=0.95,
        reference_config_sha256="f" * 64,
        checkpoint_sizes={"deepsets": 20, "set_transformer": 10},
    )


def test_reducer_computes_oracle_bootstrap_bins_and_winner() -> None:
    selection = _selection()
    result = _evaluate(selection, _terminals(selection))
    assert result["status"] == COMPLETED_EVALUATION
    assert result["coverage"]["completed_state_count"] == 4
    assert result["state_records"][0]["oracle"]["2"]["selected_event_ids"] == [1, 2]
    assert (
        result["method_summaries"]["set_transformer"]["exact_oracle_regret"]["2"]["mean"]
        == 0.0
    )
    assert result["comparisons"]["set_transformer"]["minus_recent"]["lower"] > 0.0
    assert result["comparisons"]["set_transformer"]["verdict"] == "GO"
    assert result["winner"]["model"] == "set_transformer"
    assert result["method_summaries"]["set_transformer"]["history_bins"]["very_long"] is not None


def test_incomplete_terminal_coverage_is_a_no_go_report() -> None:
    selection = _selection()
    terminals = _terminals(selection)
    terminals.pop(next(iter(terminals)))
    result = _evaluate(selection, terminals)
    assert result["status"] == INCOMPLETE_EVALUATION
    assert result["coverage"]["complete"] is False
    assert result["winner"]["verdict"] == "NO_GO"


def test_nonfinite_prediction_fails_the_finite_requirement() -> None:
    selection = _selection()
    selection["records"][0]["model_metrics"]["deepsets"]["predicted_utilities"]["1"] = math.nan
    with pytest.raises(ValueError, match="JSON compliant"):
        _evaluate(selection, _terminals(selection))


def test_percentile_type7_matches_linear_definition() -> None:
    assert percentile_type7((0.0, 10.0), 0.25) == pytest.approx(2.5)
