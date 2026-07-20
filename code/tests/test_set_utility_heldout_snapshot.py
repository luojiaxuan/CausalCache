from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causalcache.restoration_v2_baselines import RESIZED_RGB_BYTE_COUNT
from causalcache.set_utility_heldout_snapshot import (
    OUTPUT_STATUS,
    materialize_heldout_feature_snapshot,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history(trajectory_id: str) -> list[dict[str, object]]:
    return [
        {
            "event_step_id": step,
            "high_fidelity_observation_ref": (
                f"images/{trajectory_id}/observation-{step:03d}.png"
            ),
            "low_fidelity_summary": {
                "action_argument": "wait",
                "action_type": "wait",
                "executor_result": "unknown",
                "foreground_app": "unknown",
                "screen_change": "none",
                "screen_text_added": [],
                "screen_text_removed": [],
                "step_id": step,
            },
        }
        for step in range(1, 6)
    ]


def _source_row(trajectory_id: str) -> dict[str, object]:
    ocr = {
        f"images/{trajectory_id}/observation-{step:03d}.png": {
            "full_spatial_tokens": [f"screen-{step}", "shared"]
        }
        for step in range(6)
    }
    return {
        "decision_count": 5,
        "history_events_json": json.dumps(_history(trajectory_id)),
        "images": [
            {"bytes": bytes([step * 40]), "path": None} for step in range(6)
        ],
        "ocr_records_json": json.dumps(ocr),
        "role": "evaluation",
        "source_id": trajectory_id,
        "task_instruction": f"complete {trajectory_id}",
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    trajectories = ("trajectory-0", "trajectory-1", "trajectory-2")
    state_ids = tuple(f"{value}:decision:006" for value in trajectories)
    inventory = {
        "assignment_manifest_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "evaluation_tracks": {
            "exact_state_ids": [state_ids[0], state_ids[1]],
            "large_history_state_ids": [state_ids[1], state_ids[2]],
        },
        "schema_version": "1.0.0",
        "state_identity_sha256": "c" * 64,
        "status": "FROZEN_VARIABLE_HISTORY_STATE_INVENTORY",
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    source_root = tmp_path / "source"
    source_path = source_root / "trajectory-shards/shard-000-of-256.parquet"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"source-shard")
    source_manifest = {
        "assignment_manifest_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "logical_shard_count": 256,
        "shards": [
            {
                "byte_count": source_path.stat().st_size,
                "logical_shard": 0,
                "sha256": _sha256(source_path),
                "trajectory_ids": list(trajectories),
            }
        ],
        "status": "COMPLETED_VARIABLE_HISTORY_SOURCE",
    }
    (source_root / "manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )

    visual_root = tmp_path / "visual"
    token_path = visual_root / "token-shards/shard-000-of-256.safetensors"
    receipt_path = visual_root / "receipts/shard-000-of-256.json"
    token_path.parent.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True)
    token_path.write_bytes(b"visual-token-shard")
    receipt_path.write_text(
        json.dumps(
            {
                "byte_count": token_path.stat().st_size,
                "logical_shard": 0,
                "sha256": _sha256(token_path),
                "status": "COMPLETED_VARIABLE_HISTORY_TOKEN_SHARD",
            }
        ),
        encoding="utf-8",
    )
    return source_root, visual_root, inventory_path, [
        _source_row(trajectory_id) for trajectory_id in trajectories
    ]


def test_materializer_emits_label_blind_track_union(tmp_path: Path) -> None:
    source_root, visual_root, inventory_path, source_rows = _write_inputs(tmp_path)
    matrices = {
        f"means__{row['source_id']}": [[1.0, 0.0] for _ in range(6)]
        for row in source_rows
    }
    output_root = tmp_path / "output"
    manifest = materialize_heldout_feature_snapshot(
        source_root=source_root,
        full_visual_token_root=visual_root,
        frozen_state_inventory_path=inventory_path,
        output_root=output_root,
        read_source_rows=lambda _: source_rows,
        load_visual_tensors=lambda _: matrices,
        prepare_resized_rgb=lambda payload: payload * RESIZED_RGB_BYTE_COUNT,
        expected_exact_state_count=2,
        expected_large_history_state_count=2,
        expected_union_state_count=3,
    )
    assert manifest["status"] == OUTPUT_STATUS
    assert manifest["state_count"] == 3
    assert manifest["track_counts"] == {
        "exact_oracle": 2,
        "large_history": 2,
        "overlap": 1,
        "union": 3,
    }
    assert manifest["evaluation_labels_loaded"] is False
    assert manifest["evaluation_labels_included"] is False
    assert manifest["label_file_read_count"] == 0
    assert manifest["numeric_feature_names"][-1] == (
        "current_vlm_mean_cosine_rescaled"
    )
    assert manifest["frozen_state_inventory_sha256"] == _sha256(inventory_path)

    states = [
        json.loads(line)
        for line in (output_root / "states.jsonl").read_text().splitlines()
    ]
    assert len(states) == 3
    assert all(row["role"] == "evaluation" for row in states)
    assert all(len(row["candidate_event_step_ids"]) == 5 for row in states)
    assert all(len(row["event_numeric_features"]) == 5 for row in states)
    assert all(len(row["ocr_rgb_scores"]) == 5 for row in states)
    assert all(
        len(features) == 11
        for row in states
        for features in row["event_numeric_features"]
    )
    assert not any("distance_rows" in row for row in states)
    memberships = {row["state_id"]: row["track_membership"] for row in states}
    assert memberships["trajectory-1:decision:006"] == [
        "exact_oracle",
        "large_history",
    ]
    assert all(row["current_image_key"].endswith("observation:005") for row in states)
    assert all(len(row["event_image_keys"]) == 5 for row in states)
    assert all(row["ocr_rgb_scores"][-1] == 1.0 for row in states)
    assert all(row["ocr_rgb_scores"][0] == pytest.approx(1.0 / 6.0) for row in states)
    assert all(row["event_numeric_features"][0][-1] == 1.0 for row in states)


def test_materializer_rejects_non_evaluation_source(tmp_path: Path) -> None:
    source_root, visual_root, inventory_path, source_rows = _write_inputs(tmp_path)
    source_rows[0] = {**source_rows[0], "role": "train"}
    matrices = {
        f"means__{row['source_id']}": [[1.0, 0.0] for _ in range(6)]
        for row in source_rows
    }
    with pytest.raises(ValueError, match="not in the evaluation split"):
        materialize_heldout_feature_snapshot(
            source_root=source_root,
            full_visual_token_root=visual_root,
            frozen_state_inventory_path=inventory_path,
            output_root=tmp_path / "output",
            read_source_rows=lambda _: source_rows,
            load_visual_tensors=lambda _: matrices,
            prepare_resized_rgb=lambda payload: payload * RESIZED_RGB_BYTE_COUNT,
            expected_exact_state_count=2,
            expected_large_history_state_count=2,
            expected_union_state_count=3,
        )
