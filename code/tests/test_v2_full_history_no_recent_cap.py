"""Full-history singleton rendering and temporal-feature invariants."""

from __future__ import annotations

from causalcache.hgkv_selector_v2 import temporal_features
from scripts.render_hgkv_selector_v2_singletons import (
    annotate_v2_singleton_samples,
    load_cached_singletons,
)


def test_full_history_singletons_are_not_recent8_capped():
    candidates = list(range(1, 11))
    state = {
        "pair_group": "episode:12",
        "decision_step": 12,
        "history_length": 11,
        "candidate_event_step_ids": candidates,
    }
    samples = [
        {
            "pair_group": "episode:12",
            "decision_step_id": 12,
            "singleton_event_step_id": None,
            "variant": "b0",
        }
    ] + [
        {
            "pair_group": "episode:12",
            "decision_step_id": 12,
            "singleton_event_step_id": candidate,
            "variant": "singleton",
        }
        for candidate in candidates
    ]
    annotated = annotate_v2_singleton_samples(samples, state)
    singleton_rows = [
        row for row in annotated if row["singleton_event_step_id"] is not None
    ]
    assert len(singleton_rows) == 10
    assert [row["singleton_event_step_id"] for row in singleton_rows] == candidates
    assert singleton_rows[0]["temporal_features"] == list(
        temporal_features(
            event_step_id=1,
            decision_step=12,
            history_length=11,
            candidate_event_step_ids=candidates,
        )
    )
    assert singleton_rows[-1]["temporal_features"][3] == 1.0


def test_existing_exact_singletons_are_loaded_for_incremental_render(tmp_path):
    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        "\n".join(
            [
                '{"pair_group":"episode:12","restored_event_step_ids":[]}',
                '{"pair_group":"episode:12","restored_event_step_ids":[3]}',
                '{"pair_group":"episode:12","restored_event_step_ids":[3,4]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_cached_singletons(str(cache)) == {("episode:12", 3)}
