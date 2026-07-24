"""Full-history singleton rendering and temporal-feature invariants."""

from __future__ import annotations

import json

from causalcache.hgkv_selector_v2 import temporal_features
from scripts.build_hgkv_readout_cache_v2 import convert_rows
from scripts.render_hgkv_selector_v2_singletons import (
    annotate_v2_singleton_samples,
    load_cached_singletons,
)
from scripts.run_hgkv_readout_extraction_v2_shards import input_manifest
from scripts.validate_hgkv_selector_v2_singletons import validate


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


def test_missing_singleton_validator_requires_exact_complement(tmp_path):
    state_path = tmp_path / "states.jsonl"
    state_path.write_text(
        '{"episode":"episode","pair_group":"episode:12","decision_step":12,'
        '"history_length":11,"candidate_event_step_ids":[3,4],'
        '"recent_candidate_cap":null}\n',
        encoding="utf-8",
    )
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(
        '{"pair_group":"episode:12","restored_event_step_ids":[3]}\n',
        encoding="utf-8",
    )
    image_path = tmp_path / "candidate.png"
    image_path.write_bytes(b"png")
    sample_path = tmp_path / "samples.jsonl"
    temporal = temporal_features(
        event_step_id=4,
        decision_step=12,
        history_length=11,
        candidate_event_step_ids=[3, 4],
    )
    sample_path.write_text(
        json.dumps(
            {
                "variant": "singleton",
                "pair_group": "episode:12",
                "singleton_event_step_id": 4,
                "candidate_event_step_ids": [3, 4],
                "memory_config": {"restored_event_step_ids": [4]},
                "temporal_features": list(temporal),
                "messages": [
                    {
                        "content": [
                            {"type": "image", "path": image_path.name},
                        ]
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = validate(
        state_paths=[state_path],
        sample_paths=[sample_path],
        coalition_cache_pattern=str(cache_path),
    )
    assert result["required_singletons"] == 2
    assert result["cached_singletons"] == 1
    assert result["rendered_missing_singletons"] == 1
    assert result["complete_singletons"] == 2
    assert result["referenced_images"] == 1
    assert result["status"] == "DONE"


def test_v1_readout_reuse_appends_v2_temporal_features(tmp_path):
    state = {
        "episode": "episode",
        "pair_group": "episode:12",
        "decision_step": 12,
        "history_length": 11,
        "candidate_event_step_ids": [3, 4],
    }
    v1_path = tmp_path / "v1.jsonl"
    v1_path.write_text(
        json.dumps(
            {
                "pair_group": "episode:12",
                "singleton_event_step_id": 3,
                "feature": [0.0] * 1280,
                "layer_indices": list(range(28, 36)),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows, missing = convert_rows(
        expected={
            ("episode:12", 3): state,
            ("episode:12", 4): state,
        },
        v1_paths=[v1_path],
    )
    assert len(rows) == 1
    assert len(rows[0]["feature"]) == 1285
    assert rows[0]["feature"][-5:] == list(
        temporal_features(
            event_step_id=3,
            decision_step=12,
            history_length=11,
            candidate_event_step_ids=[3, 4],
        )
    )
    assert missing == {("episode:12", 4)}


def test_v2_readout_input_manifest_freezes_exact_singleton_keys(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "variant": "singleton",
                    "pair_group": "episode:12",
                    "singleton_event_step_id": candidate,
                }
            )
            for candidate in (3, 4)
        )
        + "\n",
        encoding="utf-8",
    )
    hashes, keys = input_manifest([path])
    assert set(hashes) == {str(path)}
    assert keys == {("episode:12", 3), ("episode:12", 4)}
