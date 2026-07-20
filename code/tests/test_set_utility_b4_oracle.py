from __future__ import annotations

import hashlib
import json

from causalcache.set_utility_b4_oracle import (
    B4_SCHEDULE_STATUS,
    materialize_b4_oracle_schedules,
)
from causalcache.set_utility_heldout_evaluation import sha256_file
from causalcache.set_utility_heldout_inference import canonical_json_bytes


def test_b4_schedule_selects_small_exact_states_and_uses_configured_cardinalities(
    tmp_path,
) -> None:
    records = []
    for trajectory, logical_shard, candidate_count in (
        ("t1", 0, 5),
        ("t2", 1, 6),
        ("excluded", 1, 9),
    ):
        records.append(
            {
                "candidate_event_ids": list(range(1, candidate_count + 1)),
                "logical_shard": logical_shard,
                "state_id": f"{trajectory}:decision:{candidate_count + 1:03d}",
                "tracks": ["exact_oracle"],
                "trajectory_id": trajectory,
            }
        )
    selections: dict[str, object] = {"records": records}
    selections["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(selections)
    ).hexdigest()
    selections_path = tmp_path / "selections.json"
    selections_path.write_bytes(canonical_json_bytes(selections))
    source = {
        "shards": [
            {"logical_shard": 0, "sha256": "a" * 64, "trajectory_ids": ["t1"]},
            {
                "logical_shard": 1,
                "sha256": "b" * 64,
                "trajectory_ids": ["t2", "excluded"],
            },
        ],
        "status": "COMPLETED_VARIABLE_HISTORY_SOURCE",
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    config = {
        "inputs": {
            "sealed_selections": {
                "content_sha256": selections["content_sha256"],
                "file_sha256": sha256_file(selections_path),
            }
        },
        "new_truth_schedule": {
            "coalition_cardinalities": [0, 1, 2, 3, 4],
            "expected_forward_coalition_count": 88,
            "expected_total_schedule_coalition_count": 90,
            "logical_shard_count": 2,
        },
        "state_selection": {
            "expected_state_count": 2,
            "expected_trajectory_count": 2,
            "maximum_candidate_count": 8,
            "minimum_candidate_count": 5,
            "required_track": "exact_oracle",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_root = tmp_path / "schedules"
    result = materialize_b4_oracle_schedules(
        config_path=config_path,
        selections_path=selections_path,
        source_manifest_path=source_path,
        output_root=output_root,
        workers=2,
    )
    assert result["status"] == B4_SCHEDULE_STATUS
    assert result["state_count"] == 2
    assert result["coalition_count"] == 90
    assert result["forward_coalition_count"] == 88
    rows = [
        json.loads(line)
        for path in sorted((output_root / "schedule-shards").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert {row["state_id"] for row in rows} == {
        "t1:decision:006",
        "t2:decision:007",
    }
    for row in rows:
        candidates = row["candidate_event_ids"]
        cardinalities = {len(item["event_ids"]) for item in row["coalitions"]}
        assert cardinalities == {0, 1, 2, 3, 4, len(candidates)}
    assert materialize_b4_oracle_schedules(
        config_path=config_path,
        selections_path=selections_path,
        source_manifest_path=source_path,
        output_root=output_root,
        workers=1,
    ) == result
