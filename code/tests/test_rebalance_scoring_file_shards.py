from __future__ import annotations

import json

import pytest

from scripts.rebalance_scoring_file_shards import (
    assignment_for,
    rebalance,
)


def _row(pair_group: str, restored_set_key: str) -> dict:
    return {
        "pair_group": pair_group,
        "restored_set_key": restored_set_key,
        "singleton_event_step_id": None,
        "target_logprob_mean": -0.5,
        "variant": "cond_edge1",
    }


def test_rebalance_preserves_rows_and_keeps_duplicate_identity_together(
    tmp_path,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    rows = [
        _row("episode-a:4", "1-2"),
        _row("episode-a:4", "1-2"),
        _row("episode-a:4", "1-3"),
        _row("episode-b:5", "2-4"),
    ]
    for index, selected in enumerate((rows[:3], rows[3:])):
        (input_root / f"samples-shard{index:03d}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in selected),
            encoding="utf-8",
        )
    images = input_root / "images"
    images.mkdir()
    output_root = tmp_path / "balanced"

    manifest = rebalance(
        input_root=input_root,
        output_root=output_root,
        input_glob="samples-shard*.jsonl",
        output_count=3,
        images_target=images,
        source_commit="abc123",
    )

    assert manifest["input_raw_rows"] == 4
    assert manifest["output_raw_rows"] == 4
    assert manifest["input_unique_score_identities"] == 3
    assert manifest["output_unique_score_identities"] == 3
    assert (output_root / "images").resolve() == images.resolve()
    emitted = []
    owners = {}
    for path in sorted(output_root.glob("samples-balanced-shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            emitted.append(row)
            key = (row["pair_group"], row["restored_set_key"])
            assert key not in owners or owners[key] == path.name
            owners[key] = path.name
    assert sorted(json.dumps(row, sort_keys=True) for row in emitted) == sorted(
        json.dumps(row, sort_keys=True) for row in rows
    )
    assert assignment_for(
        ("episode-a:4", "cond_edge1", None, "1-2"), output_count=3
    ) in range(3)


def test_rebalance_rejects_cross_file_identity(tmp_path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    duplicate = _row("episode-a:4", "1-2")
    for index in range(2):
        (input_root / f"samples-shard{index:03d}.jsonl").write_text(
            json.dumps(duplicate) + "\n", encoding="utf-8"
        )
    images = input_root / "images"
    images.mkdir()

    with pytest.raises(ValueError, match="crosses physical input files"):
        rebalance(
            input_root=input_root,
            output_root=tmp_path / "balanced",
            input_glob="samples-shard*.jsonl",
            output_count=2,
            images_target=images,
            source_commit="abc123",
        )
    assert not (tmp_path / "balanced.partial").exists()
