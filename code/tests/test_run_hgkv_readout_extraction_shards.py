from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.run_hgkv_readout_extraction_shards import (
    parse_gpu_indices,
    parse_shard_indices,
    resolve_shard_indices,
    validate_feature_outputs,
)


def _row(pair_group: str, event_id: int, feature: list[float]) -> dict:
    return {
        "feature": feature,
        "feature_dim": len(feature),
        "pair_group": pair_group,
        "singleton_event_step_id": event_id,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_parse_gpu_indices_requires_unique_non_negative_values() -> None:
    assert parse_gpu_indices("0,2,7") == [0, 2, 7]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gpu_indices("0,0")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gpu_indices("-1")


def test_resolve_shard_indices_supports_sparse_resume() -> None:
    assert parse_shard_indices("1,5,6") == [1, 5, 6]
    assert resolve_shard_indices(8, None) == list(range(8))
    assert resolve_shard_indices(8, [1, 5, 6]) == [1, 5, 6]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shard_indices("1,1")
    with pytest.raises(ValueError, match="exceed shard-count"):
        resolve_shard_indices(8, [8])


def test_validate_feature_outputs_checks_rows_keys_and_dimensions(
    tmp_path: Path,
) -> None:
    shard0 = tmp_path / "features-shard0.jsonl"
    shard1 = tmp_path / "features-shard1.jsonl"
    _write_rows(shard0, [_row("episode-a:1", 0, [0.1, 0.2])])
    _write_rows(shard1, [_row("episode-b:2", 1, [0.3, 0.4])])

    result = validate_feature_outputs(
        [shard0, shard1], expected_rows=2
    )

    assert result == {
        "feature_dim": 2,
        "row_count": 2,
        "rows_per_shard": {
            "features-shard0.jsonl": 1,
            "features-shard1.jsonl": 1,
        },
        "unique_key_count": 2,
    }


def test_validate_feature_outputs_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    shard0 = tmp_path / "features-shard0.jsonl"
    shard1 = tmp_path / "features-shard1.jsonl"
    duplicate = _row("episode-a:1", 0, [0.1])
    _write_rows(shard0, [duplicate])
    _write_rows(shard1, [duplicate])

    with pytest.raises(ValueError, match="duplicate feature key"):
        validate_feature_outputs([shard0, shard1], expected_rows=2)


def test_validate_feature_outputs_rejects_non_finite_values(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "features-shard0.jsonl"
    _write_rows(shard, [_row("episode-a:1", 0, [float("nan")])])

    with pytest.raises(ValueError, match="non-finite features"):
        validate_feature_outputs([shard], expected_rows=1)
