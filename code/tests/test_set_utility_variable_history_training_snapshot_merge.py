import hashlib
import json
import sys
from pathlib import Path

from scripts.merge_set_utility_variable_history_training_snapshots import main


def _write_snapshot(root: Path, *, snapshot_id: str, state_id: str, role: str) -> None:
    root.mkdir()
    trajectory_id = state_id.split(":", 1)[0]
    state_payload = (
        json.dumps(
            {
                "logical_shard": 1 if role == "train" else 2,
                "role": role,
                "state_id": state_id,
                "trajectory_id": trajectory_id,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    text_payload = (json.dumps({"key": "key", "text": "text"}, sort_keys=True) + "\n").encode()
    (root / "states.jsonl").write_bytes(state_payload)
    (root / "texts.jsonl").write_bytes(text_payload)
    manifest = {
        "content_sha256": hashlib.sha256(state_payload + text_payload).hexdigest(),
        "evaluation_labels_included": False,
        "label_source_revisions": ["a" * 40],
        "numeric_feature_names": ["age"],
        "snapshot_id": snapshot_id,
        "source_manifest_sha256": "b" * 64,
        "state_count": 1,
        "states_jsonl": "states.jsonl",
        "states_sha256": hashlib.sha256(state_payload).hexdigest(),
        "status": "COMPLETED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT",
        "text_count": 1,
        "texts_jsonl": "texts.jsonl",
        "texts_sha256": hashlib.sha256(text_payload).hexdigest(),
        "visual_token_profile": "full",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_merge_training_snapshots_preserves_disjoint_states(
    tmp_path: Path, monkeypatch
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "merged"
    _write_snapshot(left, snapshot_id="left", state_id="train:decision:006", role="train")
    _write_snapshot(right, snapshot_id="right", state_id="tune:decision:006", role="tune")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge",
            "--input-roots",
            str(left),
            str(right),
            "--output-root",
            str(output),
            "--snapshot-id",
            "merged",
        ],
    )
    main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["state_count"] == 2
    assert manifest["text_count"] == 1
    assert manifest["role_counts"] == {"train": 1, "tune": 1}
    assert manifest["visual_logical_shards"] == [1, 2]
