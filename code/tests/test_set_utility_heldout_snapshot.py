from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causalcache.restoration_v2_baselines import RESIZED_RGB_BYTE_COUNT
from causalcache.set_utility_heldout_snapshot import (
    FULL_EVALUATION_OUTPUT_STATUS,
    OUTPUT_STATUS,
    full_evaluation_state_membership,
    materialize_full_evaluation_feature_snapshot,
    materialize_heldout_feature_snapshot,
    merge_heldout_feature_snapshots,
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


def test_full_evaluation_membership_covers_1046_and_canonical_slices() -> None:
    assignments = []
    for index in range(100):
        trajectory_id = f"{index:016d}"
        if index < 93:
            state_count = 10
        elif index == 93:
            state_count = 100
        else:
            state_count = (2, 2, 3, 3, 3, 3)[index - 94]
        assignments.append(
            {
                "decision_count": state_count + 4,
                "role": "evaluation",
                "source_id": trajectory_id,
                "trajectory_id": trajectory_id,
            }
        )
    all_state_ids = sorted(
        f"{row['trajectory_id']}:decision:{decision:03d}"
        for row in assignments
        for decision in range(6, row["decision_count"] + 2)
    )
    states_by_trajectory = {
        row["trajectory_id"]: [
            state_id
            for state_id in all_state_ids
            if state_id.startswith(f"{row['trajectory_id']}:decision:")
        ]
        for row in assignments[:94]
    }
    previous_union = [values[0] for values in states_by_trajectory.values()]
    previous_union.extend(
        state_id
        for values in states_by_trajectory.values()
        for state_id in values[1:]
    )
    previous_union = previous_union[:805]
    exact = previous_union[:320]
    large = [*previous_union[:235], *previous_union[320:805]]
    previous_trajectories = {
        state_id.rpartition(":decision:")[0] for state_id in set(exact) | set(large)
    }
    trajectory_new_ids = {
        row["trajectory_id"] for row in assignments
    } - previous_trajectories
    trajectory_new_states = {
        state_id
        for state_id in all_state_ids
        if state_id.rpartition(":decision:")[0] in trajectory_new_ids
    }
    inventory = {
        "evaluation_tracks": {
            "exact_state_ids": exact,
            "large_history_state_ids": large,
        },
        "status": "FROZEN_VARIABLE_HISTORY_STATE_INVENTORY",
        "summary": {"state_count_by_role": {"evaluation": 1046}},
    }
    membership, exposure = full_evaluation_state_membership(
        {"assignments": assignments},
        inventory,
    )
    assert len(membership) == 1046
    assert exposure["counts"] == {
        "all": 1046,
        "exact_oracle": 320,
        "large_history": 720,
        "previous_heldout_union": 805,
        "state_new": 241,
        "trajectory_new": 16,
    }
    assert set(exposure["trajectory_new_state_ids"]) == trajectory_new_states
    assert set(exposure["trajectory_new_state_ids"]).issubset(
        exposure["state_new_state_ids"]
    )


def test_full_evaluation_materializer_is_label_blind_and_marks_slices(
    tmp_path: Path,
) -> None:
    source_root, visual_root, inventory_path, source_rows = _write_inputs(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    state_ids = [f"{row['source_id']}:decision:006" for row in source_rows]
    inventory["evaluation_tracks"] = {
        "exact_state_ids": [state_ids[0]],
        "large_history_state_ids": [state_ids[1]],
    }
    inventory["summary"] = {"state_count_by_role": {"evaluation": 3}}
    assignments = {
        "assignments": [
            {
                "decision_count": 5,
                "role": "evaluation",
                "source_id": row["source_id"],
                "trajectory_id": row["source_id"],
            }
            for row in source_rows
        ]
    }
    assignment_path = tmp_path / "assignments.json"
    assignment_path.write_text(json.dumps(assignments), encoding="utf-8")
    assignment_sha256 = _sha256(assignment_path)
    inventory["assignment_manifest_sha256"] = assignment_sha256
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["assignment_manifest_sha256"] = assignment_sha256
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    matrices = {
        f"means__{row['source_id']}": [[1.0, 0.0] for _ in range(6)]
        for row in source_rows
    }
    output_root = tmp_path / "full-output"
    manifest = materialize_full_evaluation_feature_snapshot(
        source_root=source_root,
        full_visual_token_root=visual_root,
        frozen_state_inventory_path=inventory_path,
        assignment_manifest_path=assignment_path,
        output_root=output_root,
        read_source_rows=lambda _: source_rows,
        load_visual_tensors=lambda _: matrices,
        prepare_resized_rgb=lambda payload: payload * RESIZED_RGB_BYTE_COUNT,
        expected_full_evaluation_state_count=3,
        expected_evaluation_trajectory_count=3,
        expected_exact_state_count=1,
        expected_large_history_state_count=1,
        expected_union_state_count=2,
        expected_previous_heldout_trajectory_count=2,
        expected_trajectory_new_count=1,
        expected_trajectory_new_state_count=1,
    )
    assert manifest["status"] == FULL_EVALUATION_OUTPUT_STATUS
    assert manifest["state_count"] == 3
    assert manifest["historical_exposure_counts"] == {
        "all": 3,
        "exact_oracle": 1,
        "large_history": 1,
        "previous_heldout_union": 2,
        "state_new": 1,
        "trajectory_new": 1,
    }
    assert manifest["label_file_read_count"] == 0
    exposure = json.loads((output_root / "exposure-slices.json").read_text())
    assert exposure["state_new_state_ids"] == [state_ids[2]]
    assert exposure["trajectory_new_state_ids"] == [state_ids[2]]
    states = [
        json.loads(line)
        for line in (output_root / "states.jsonl").read_text().splitlines()
    ]
    assert states[-1]["historical_exposure_slices"] == [
        "all",
        "state_new",
        "trajectory_new",
    ]
    assert not any("distance_rows" in row for row in states)


def test_full_evaluation_partial_merge_preserves_canonical_slices(
    tmp_path: Path,
) -> None:
    state_ids = [f"trajectory-{index}:decision:006" for index in range(3)]
    partitions = (
        (
            0,
            (
                {
                    "historical_exposure_slices": ["all"],
                    "state_id": state_ids[0],
                    "track_membership": ["exact_oracle"],
                    "trajectory_id": "trajectory-0",
                },
                {
                    "historical_exposure_slices": [
                        "all",
                        "state_new",
                        "trajectory_new",
                    ],
                    "state_id": state_ids[2],
                    "track_membership": [],
                    "trajectory_id": "trajectory-2",
                },
            ),
        ),
        (
            1,
            (
                {
                    "historical_exposure_slices": ["all"],
                    "state_id": state_ids[1],
                    "track_membership": ["large_history"],
                    "trajectory_id": "trajectory-1",
                },
            ),
        ),
    )
    roots = []
    for logical_shard, rows in partitions:
        root = tmp_path / f"partition-{logical_shard}"
        root.mkdir()
        state_payload = b"".join(
            json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows
        )
        text_payload = b""
        (root / "states.jsonl").write_bytes(state_payload)
        (root / "texts.jsonl").write_bytes(text_payload)
        row_ids = {row["state_id"] for row in rows}
        exposure = {
            "all_state_ids": sorted(row_ids),
            "counts": {},
            "exact_oracle_state_ids": sorted(row_ids & {state_ids[0]}),
            "large_history_state_ids": sorted(row_ids & {state_ids[1]}),
            "previous_heldout_union_state_ids": sorted(
                row_ids & {state_ids[0], state_ids[1]}
            ),
            "schema_version": "1.0.0",
            "state_new_state_ids": sorted(row_ids & {state_ids[2]}),
            "status": "FROZEN_SET_UTILITY_EVALUATION_EXPOSURE_SLICES_V1",
            "trajectory_counts": {},
            "trajectory_new_state_ids": sorted(row_ids & {state_ids[2]}),
            "trajectory_new_trajectory_ids": (
                ["trajectory-2"] if state_ids[2] in row_ids else []
            ),
        }
        exposure_payload = json.dumps(exposure, sort_keys=True).encode()
        (root / "exposure-slices.json").write_bytes(exposure_payload)
        manifest = {
            "assignment_manifest_sha256": "a" * 64,
            "content_sha256": hashlib.sha256(state_payload).hexdigest(),
            "evaluation_labels_loaded": False,
            "exposure_slices_json": "exposure-slices.json",
            "exposure_slices_sha256": hashlib.sha256(exposure_payload).hexdigest(),
            "frozen_state_inventory_sha256": "b" * 64,
            "input_bindings": {
                "assignment_manifest_sha256": "a" * 64,
                "config_sha256": "c" * 64,
                "source_manifest_sha256": "d" * 64,
                "state_identity_sha256": "e" * 64,
            },
            "label_file_read_count": 0,
            "numeric_feature_names": ["age"],
            "ocr_rgb_profile": {"formula": "fixture"},
            "source_shards": [
                {
                    "byte_count": 1,
                    "logical_shard": logical_shard,
                    "sha256": f"{logical_shard + 1}" * 64,
                }
            ],
            "states_jsonl": "states.jsonl",
            "states_sha256": hashlib.sha256(state_payload).hexdigest(),
            "status": "COMPLETED_SET_UTILITY_FULL_EVALUATION_FEATURE_PARTITION_V1",
            "texts_jsonl": "texts.jsonl",
            "texts_sha256": hashlib.sha256(text_payload).hexdigest(),
            "visual_shards": [
                {
                    "logical_shard": logical_shard,
                    "sha256": f"{logical_shard + 3}" * 64,
                }
            ],
            "visual_token_profile": "fixture",
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        roots.append(root)

    output_root = tmp_path / "merged"
    manifest = merge_heldout_feature_snapshots(
        input_roots=roots,
        output_root=output_root,
        expected_exact_state_count=1,
        expected_large_history_state_count=1,
        expected_union_state_count=2,
        expected_full_evaluation_state_count=3,
        expected_evaluation_trajectory_count=3,
        expected_trajectory_new_count=1,
        expected_trajectory_new_state_count=1,
    )
    assert manifest["status"] == FULL_EVALUATION_OUTPUT_STATUS
    assert manifest["state_count"] == 3
    assert manifest["historical_exposure_counts"] == {
        "all": 3,
        "exact_oracle": 1,
        "large_history": 1,
        "previous_heldout_union": 2,
        "state_new": 1,
        "trajectory_new": 1,
    }
    exposure = json.loads((output_root / "exposure-slices.json").read_text())
    assert exposure["trajectory_new_trajectory_ids"] == ["trajectory-2"]
