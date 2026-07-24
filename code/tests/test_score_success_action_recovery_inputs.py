from __future__ import annotations

import json
from pathlib import Path

from scripts.run_success_action_scoring_shards import (
    score_key,
    validate_score_outputs,
)
from scripts.score_success_action_recovery import (
    iter_sharded_samples,
    resolve_sample_paths,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_file_sharding_reads_disjoint_sorted_inputs(tmp_path: Path) -> None:
    for index in range(4):
        _write(
            tmp_path / f"samples-shard{index:03d}.jsonl",
            [{"pair_group": f"state-{index}"}],
        )
    paths = resolve_sample_paths(tmp_path, "samples-shard*.jsonl")

    shard0 = list(
        iter_sharded_samples(
            paths, shard_index=0, shard_count=2, shard_by_file=True
        )
    )
    shard1 = list(
        iter_sharded_samples(
            paths, shard_index=1, shard_count=2, shard_by_file=True
        )
    )

    assert [row["pair_group"] for row in shard0] == ["state-0", "state-2"]
    assert [row["pair_group"] for row in shard1] == ["state-1", "state-3"]


def test_sample_sharding_preserves_global_line_order(tmp_path: Path) -> None:
    _write(
        tmp_path / "part-a.jsonl",
        [{"pair_group": "state-0"}, {"pair_group": "state-1"}],
    )
    _write(
        tmp_path / "part-b.jsonl",
        [{"pair_group": "state-2"}, {"pair_group": "state-3"}],
    )
    paths = resolve_sample_paths(tmp_path, "part-*.jsonl")

    shard = list(
        iter_sharded_samples(
            paths, shard_index=1, shard_count=2, shard_by_file=False
        )
    )

    assert [row["pair_group"] for row in shard] == ["state-1", "state-3"]


def test_score_output_validation_uses_resume_identity(tmp_path: Path) -> None:
    row0 = {
        "pair_group": "state-0",
        "restored_set_key": "1",
        "singleton_event_step_id": None,
        "target_logprob_mean": -0.2,
        "variant": "cond_base",
    }
    row1 = {
        "pair_group": "state-0",
        "restored_set_key": "1-2",
        "singleton_event_step_id": None,
        "target_logprob_mean": -0.1,
        "variant": "cond_edge1",
    }
    shard0 = tmp_path / "scores-shard0.jsonl"
    shard1 = tmp_path / "scores-shard1.jsonl"
    _write(shard0, [row0])
    _write(shard1, [row1])

    result = validate_score_outputs(
        [shard0, shard1], expected_keys={score_key(row0), score_key(row1)}
    )

    assert result["row_count"] == 2
    assert result["unique_key_count"] == 2
