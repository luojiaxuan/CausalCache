from __future__ import annotations

import hashlib
import json

import pytest

from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_PROFILE_ID,
    EVALUATION_ROLE_SCOPE,
    materialize_contextual_inputs,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes
from causalcache.set_utility_heldout_snapshot import FULL_EVALUATION_OUTPUT_STATUS


def test_contextual_inputs_deduplicate_events_and_keep_query_roles_distinct(tmp_path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    rows = []
    for decision in (6, 7):
        event_count = decision - 1
        rows.append(
            {
                "candidate_event_step_ids": list(range(1, decision)),
                "current_image_key": f"123:observation:{event_count:03d}",
                "event_image_keys": [
                    f"123:observation:{step:03d}" for step in range(1, decision)
                ],
                "event_text_keys": [f"old-{step}" for step in range(1, decision)],
                "event_texts": [f"summary {step}" for step in range(1, decision)],
                "instruction": "Complete the task",
                "instruction_text_key": "old-query",
                "logical_shard": 9,
                "role": "train" if decision == 6 else "tune",
                "state_id": f"123:decision:{decision:03d}",
                "trajectory_id": "123",
            }
        )
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    (input_root / "states.jsonl").write_bytes(payload)
    manifest = {
        "content_sha256": "i" * 64,
        "evaluation_labels_included": False,
        "state_count": 2,
        "states_jsonl": "states.jsonl",
        "states_sha256": hashlib.sha256(payload).hexdigest(),
        "status": "COMPLETED_MERGED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT",
    }
    (input_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output_root = tmp_path / "output"
    result = materialize_contextual_inputs(
        input_root=input_root, output_root=output_root
    )
    assert result["status"] == CONTEXTUAL_INPUT_STATUS
    assert result["contextual_profile_id"] == CONTEXTUAL_PROFILE_ID
    assert result["state_count"] == 2
    assert result["context_count"] == 8
    assert result["entity_role_counts"] == {"event": 6, "query": 2}
    transformed = [
        json.loads(line)
        for line in (output_root / "states.jsonl").read_text().splitlines()
    ]
    assert transformed[0]["event_image_keys"][:5] == transformed[1][
        "event_image_keys"
    ][:5]
    assert transformed[0]["current_image_key"] != transformed[0]["event_image_keys"][-1]
    requirement_path = output_root / "requirement-shards/shard-009-of-256.jsonl"
    assert len(requirement_path.read_text().splitlines()) == 8


def test_contextual_inputs_accept_full_evaluation_features_without_truth(
    tmp_path,
) -> None:
    input_root = tmp_path / "evaluation-input"
    input_root.mkdir()
    row = {
        "candidate_event_step_ids": [1, 2, 3, 4, 5],
        "current_image_key": "123:observation:005",
        "event_image_keys": [
            f"123:observation:{step:03d}" for step in range(1, 6)
        ],
        "event_text_keys": [f"old-{step}" for step in range(1, 6)],
        "event_texts": [f"summary {step}" for step in range(1, 6)],
        "historical_exposure_slices": ["all", "state_new", "trajectory_new"],
        "instruction": "Complete the task",
        "instruction_text_key": "old-query",
        "logical_shard": 9,
        "role": "evaluation",
        "state_id": "123:decision:006",
        "track_membership": [],
        "trajectory_id": "123",
    }
    payload = canonical_json_bytes(row) + b"\n"
    (input_root / "states.jsonl").write_bytes(payload)
    exposure_payload = canonical_json_bytes(
        {
            "all_state_ids": [row["state_id"]],
            "state_new_state_ids": [row["state_id"]],
            "trajectory_new_state_ids": [row["state_id"]],
        }
    )
    (input_root / "exposure-slices.json").write_bytes(exposure_payload)
    manifest = {
        "content_sha256": "i" * 64,
        "distance_rows_included": False,
        "evaluation_labels_included": False,
        "evaluation_labels_loaded": False,
        "evaluation_scope": "all_assignment_evaluation_states",
        "exposure_slices_json": "exposure-slices.json",
        "exposure_slices_sha256": hashlib.sha256(exposure_payload).hexdigest(),
        "historical_exposure_counts": {
            "all": 1,
            "state_new": 1,
            "trajectory_new": 1,
        },
        "label_file_read_count": 0,
        "state_count": 1,
        "states_jsonl": "states.jsonl",
        "states_sha256": hashlib.sha256(payload).hexdigest(),
        "status": FULL_EVALUATION_OUTPUT_STATUS,
    }
    manifest_path = input_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_root = tmp_path / "evaluation-output"
    result = materialize_contextual_inputs(
        input_root=input_root,
        output_root=output_root,
        role_scope=EVALUATION_ROLE_SCOPE,
    )
    assert result["role_counts"] == {"evaluation": 1}
    assert result["role_scope"] == EVALUATION_ROLE_SCOPE
    assert result["evaluation_labels_loaded"] is False
    assert result["distance_rows_included"] is False
    assert result["label_file_read_count"] == 0
    assert result["exposure_slices_sha256"] == hashlib.sha256(
        exposure_payload
    ).hexdigest()
    transformed = json.loads(
        (output_root / "states.jsonl").read_text().splitlines()[0]
    )
    assert transformed["historical_exposure_slices"] == [
        "all",
        "state_new",
        "trajectory_new",
    ]

    manifest["evaluation_labels_loaded"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="truth firewall"):
        materialize_contextual_inputs(
            input_root=input_root,
            output_root=tmp_path / "forbidden-output",
            role_scope=EVALUATION_ROLE_SCOPE,
        )
