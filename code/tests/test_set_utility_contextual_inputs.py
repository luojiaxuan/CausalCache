from __future__ import annotations

import hashlib
import json

from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_PROFILE_ID,
    materialize_contextual_inputs,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes


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
