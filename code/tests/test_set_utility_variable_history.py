from __future__ import annotations

import itertools

from causalcache.set_utility_variable_history import (
    age_quantile_bins,
    build_variable_history_states,
    combined_pair_similarity,
    logical_shard_for_trajectory,
    sample_broad_subsets,
    select_evaluation_tracks,
)


def _similarities(events: tuple[int, ...]) -> dict[tuple[int, int], float]:
    return {
        pair: (pair[0] + pair[1]) / (2 * len(events))
        for pair in itertools.combinations(events, 2)
    }


def test_state_candidate_universe_grows_with_decision_step() -> None:
    states = build_variable_history_states(
        trajectory_id="trajectory",
        role="train",
        decision_count=30,
    )
    by_step = {state.decision_step_id: state for state in states}
    assert by_step[6].candidate_event_ids == tuple(range(1, 6))
    assert by_step[15].candidate_event_ids == tuple(range(1, 15))
    assert by_step[30].candidate_event_ids == tuple(range(1, 30))
    assert by_step[6].history_bin == "short"
    assert by_step[15].history_bin == "medium"
    assert by_step[30].history_bin == "long"


def test_age_bins_are_ordered_complete_and_nearly_balanced() -> None:
    bins = age_quantile_bins(tuple(range(1, 30)))
    assert tuple(event for age_bin in bins for event in age_bin) == tuple(range(1, 30))
    assert max(map(len, bins)) - min(map(len, bins)) == 1
    assert all(left[-1] < right[0] for left, right in zip(bins, bins[1:]))


def test_sampler_is_exact_for_n5_and_unique_40_for_long_histories() -> None:
    events5 = tuple(range(1, 6))
    exact = sample_broad_subsets(
        state_id="trajectory:decision:006",
        candidate_event_ids=events5,
        pair_similarity=_similarities(events5),
    )
    assert exact.exact is True
    assert len(exact.coalitions) == 32
    assert set(exact.coalitions) == {
        coalition
        for cardinality in range(6)
        for coalition in itertools.combinations(events5, cardinality)
    }

    for count in (6, 14, 29, 45):
        events = tuple(range(1, count + 1))
        first = sample_broad_subsets(
            state_id=f"trajectory:decision:{count + 1:03d}",
            candidate_event_ids=events,
            pair_similarity=_similarities(events),
        )
        second = sample_broad_subsets(
            state_id=f"trajectory:decision:{count + 1:03d}",
            candidate_event_ids=events,
            pair_similarity=_similarities(events),
        )
        assert first == second
        assert first.exact is False
        assert len(first.coalitions) == len(set(first.coalitions)) == 40
        assert () in first.coalitions
        assert events in first.coalitions
        assert all(set(coalition).issubset(events) for coalition in first.coalitions)
        if count <= 8:
            assert all((event,) in first.coalitions for event in events)


def test_similarity_combines_visual_cosine_and_ocr_jaccard() -> None:
    scores = combined_pair_similarity(
        (1, 2, 3),
        visual_embeddings={1: (1.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0)},
        ocr_token_sets={1: {"a", "b"}, 2: {"a", "b"}, 3: {"z"}},
    )
    assert scores[(1, 2)] == 1.0
    assert scores[(1, 3)] == 0.0
    assert scores[(2, 3)] == 0.0


def test_logical_shard_is_stable_and_in_range() -> None:
    first = logical_shard_for_trajectory("trajectory", shard_count=256)
    second = logical_shard_for_trajectory("trajectory", shard_count=256)
    assert first == second
    assert 0 <= first < 256


def test_evaluation_tracks_have_frozen_counts_and_all_very_long_states() -> None:
    states = tuple(
        state
        for trajectory_index in range(100)
        for state in build_variable_history_states(
            trajectory_id=f"evaluation-{trajectory_index:03d}",
            role="evaluation",
            decision_count=45 if trajectory_index < 20 else 32,
        )
    )
    tracks = select_evaluation_tracks(states)
    assert len(tracks.exact_state_ids) == 320
    assert len(tracks.large_history_state_ids) == 720
    very_long = {
        state.state_id for state in states if state.history_bin == "very_long"
    }
    assert very_long.issubset(tracks.exact_state_ids)
    assert very_long.issubset(tracks.large_history_state_ids)
