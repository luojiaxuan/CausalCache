from __future__ import annotations

import pytest

from causalcache.set_utility_train_on_policy import (
    candidate_complete_enrichment_coalitions,
    enrichment_coalitions,
    select_targeted_states,
    validate_train_selections,
)


def _record(index: int, count: int, *, model: str) -> dict:
    events = list(range(1, count + 1))
    if model == "set_transformer":
        path = {str(budget): events[:budget] for budget in range(1, 5)}
    else:
        path = {
            str(budget): sorted(events[max(0, count - budget) :])
            for budget in range(1, 5)
        }
    return {
        "candidate_event_ids": events,
        "conditional_steps": [
            {
                "base_subset": [],
                "ranked_candidates": [
                    {"predicted_utility": 0.5, "subset": [1]},
                    {
                        "predicted_utility": 0.49,
                        "subset": [3] if model == "set_transformer" else [2],
                    },
                ],
            }
        ],
        "beam_steps": [
            {
                "base_subsets": [list(range(1, budget))],
                "budget": budget,
                "candidate_count": count - budget + 1,
                "frontier_subsets": [list(range(1, budget + 1))],
            }
            for budget in range(1, 5)
        ],
        "learned": path,
        "recent": {
            str(budget): events[count - budget :]
            for budget in range(1, 5)
        },
        "state_id": f"state-{index:03d}",
        "trajectory_id": f"trajectory-{index:03d}",
    }


def _payloads() -> dict:
    counts = [5] * 5 + [9] * 5 + [17] * 5 + [33] * 5
    result = {}
    for model in ("deepsets", "set_transformer"):
        result[model] = {
            "cache_content_sha256": "cache",
            "checkpoint_sha256": model,
            "config_sha256": "config",
            "input_content_sha256": "input",
            "records": [
                _record(index, count, model=model)
                for index, count in enumerate(counts)
            ],
            "role": "train",
            "search_config_sha256": "search-config",
            "status": "COMPLETED_SET_UTILITY_TRAIN_SELECTIONS",
            "variant": model,
        }
    return result


def test_targeted_selection_is_variable_history_and_deterministic() -> None:
    selections = _payloads()
    selected, summary = select_targeted_states(
        selections,
        target_fraction=0.2,
        history_bin_weights={
            "short": 0.1,
            "medium": 0.25,
            "long": 0.6,
            "very_long": 0.05,
        },
        maximum_states_per_trajectory=1,
    )
    repeated, repeated_summary = select_targeted_states(
        selections,
        target_fraction=0.2,
        history_bin_weights={
            "short": 0.1,
            "medium": 0.25,
            "long": 0.6,
            "very_long": 0.05,
        },
        maximum_states_per_trajectory=1,
    )
    assert selected == repeated
    assert summary == repeated_summary
    assert len(selected) == 4
    assert summary["history_bin_selected_counts"] == {
        "long": 3,
        "medium": 1,
    }
    assert summary["maximum_realized_states_per_trajectory"] == 1


def test_enrichment_contains_paths_alternatives_and_anchors() -> None:
    records = validate_train_selections(_payloads())
    rows = enrichment_coalitions("state-000", records)
    subsets = [row["event_ids"] for row in rows]
    sources = {row["source"] for row in rows}
    assert subsets[0] == []
    assert subsets[-1] == [1, 2, 3, 4, 5]
    assert [2] in subsets and [3] in subsets
    assert "conditional_set_transformer_step1_rank2" in sources
    assert "anchor_full" in sources


def test_candidate_complete_enrichment_covers_every_beam_expansion() -> None:
    records = validate_train_selections(_payloads())
    rows = candidate_complete_enrichment_coalitions("state-000", records)
    subsets = {tuple(row["event_ids"]) for row in rows}
    for budget in range(1, 5):
        base = tuple(range(1, budget))
        assert base in subsets
        for event_id in range(1, 6):
            if event_id not in base:
                assert tuple(sorted((*base, event_id))) in subsets


def test_train_selection_rejects_mismatched_inventory() -> None:
    selections = _payloads()
    selections["deepsets"]["records"].pop()
    with pytest.raises(ValueError, match="inventories differ"):
        validate_train_selections(selections)


def test_train_selection_allows_distinct_training_configs_only() -> None:
    selections = _payloads()
    selections["deepsets"]["config_sha256"] = "deepsets-training-config"
    selections["set_transformer"]["config_sha256"] = "sett-training-config"
    validate_train_selections(selections)
    selections["set_transformer"]["search_config_sha256"] = "other-search-config"
    with pytest.raises(ValueError, match="bindings differ"):
        validate_train_selections(selections)
