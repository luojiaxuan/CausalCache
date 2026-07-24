"""Exact scientific-key join and conflict rejection for the V2 score cache."""

from __future__ import annotations

import json

import pytest

from scripts.build_hgkv_coalition_score_cache_v2 import (
    build_cache_rows,
    inventory_coverage,
    load_b0_scores,
    load_target_actions,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_exact_key_cache_joins_b0_and_deduplicates(tmp_path):
    targets_path = tmp_path / "samples.jsonl"
    b0_path = tmp_path / "b0.jsonl"
    score_a = tmp_path / "score-a.jsonl"
    score_b = tmp_path / "score-b.jsonl"
    _write(
        targets_path,
        [{"pair_group": "episode:7", "target_text": "click(1,2)"}],
    )
    _write(
        b0_path,
        [
            {
                "pair_group": "episode:7",
                "variant": "b0",
                "memory_config": {"restored_event_step_ids": []},
                "target_logprob_mean": -0.5,
            }
        ],
    )
    singleton = {
        "pair_group": "episode:7",
        "variant": "singleton",
        "memory_config": {"restored_event_step_ids": [3]},
        "target_logprob_mean": -0.3,
    }
    _write(score_a, [singleton])
    _write(score_b, [singleton])
    rows = build_cache_rows(
        targets=load_target_actions([targets_path]),
        b0_scores=load_b0_scores([b0_path]),
        score_sources=[("v1-a", [score_a]), ("v1-b", [score_b])],
        prompt_revision="gui-owl-v2.1",
        hgkv_checkpoint_sha256=SHA_A,
        b0_policy_sha256=SHA_B,
    )
    assert len(rows) == 2
    assert rows[0]["restored_set_key"] == ""
    assert rows[0]["u_act"] == 0.0
    singleton = next(row for row in rows if row["restored_set_key"] == "3")
    assert singleton["u_act"] == pytest.approx(0.2)
    assert singleton["source_versions"] == ["v1-a", "v1-b"]
    inventory = tmp_path / "states.jsonl"
    _write(
        inventory,
        [
            {
                "pair_group": "episode:7",
                "candidate_event_step_ids": [1, 2, 3],
            }
        ],
    )
    coverage = inventory_coverage(rows, [inventory])
    assert coverage["b0_cached_states"] == 1
    assert coverage["required_singletons"] == 3
    assert coverage["cached_singletons"] == 1
    assert coverage["missing_singletons"] == 2


def test_exact_key_conflict_fails_closed(tmp_path):
    targets_path = tmp_path / "samples.jsonl"
    score_a = tmp_path / "score-a.jsonl"
    score_b = tmp_path / "score-b.jsonl"
    _write(
        targets_path,
        [{"pair_group": "episode:7", "target_text": "click(1,2)"}],
    )
    base = {
        "pair_group": "episode:7",
        "memory_config": {"restored_event_step_ids": [3]},
    }
    _write(score_a, [{**base, "target_logprob_mean": -0.3}])
    _write(score_b, [{**base, "target_logprob_mean": -0.2}])
    with pytest.raises(ValueError, match="conflicting exact-key"):
        build_cache_rows(
            targets=load_target_actions([targets_path]),
            b0_scores={"episode:7": -0.5},
            score_sources=[("a", [score_a]), ("b", [score_b])],
            prompt_revision="gui-owl-v2.1",
            hgkv_checkpoint_sha256=SHA_A,
            b0_policy_sha256=SHA_B,
        )
