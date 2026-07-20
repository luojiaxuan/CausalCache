import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts.export_set_utility_selected_visual_tokens import main as export_main
from scripts.filter_set_utility_training_snapshot_by_trajectory import (
    main as filter_main,
)
from scripts.merge_set_utility_selected_visual_tokens import main as merge_main


def _write_input_snapshot(root: Path) -> None:
    root.mkdir()
    states = []
    texts = {}
    for role, count, shard_offset in (("train", 8, 0), ("tune", 2, 8)):
        for index in range(count):
            trajectory_id = f"{role}-{index}"
            instruction_key = hashlib.sha256(trajectory_id.encode()).hexdigest()
            event_key = hashlib.sha256(f"event-{trajectory_id}".encode()).hexdigest()
            texts[instruction_key] = trajectory_id
            texts[event_key] = f"event-{trajectory_id}"
            states.append(
                {
                    "current_image_key": f"{trajectory_id}:observation:001",
                    "event_image_keys": [f"{trajectory_id}:observation:000"],
                    "event_text_keys": [event_key],
                    "instruction_text_key": instruction_key,
                    "logical_shard": shard_offset + index,
                    "role": role,
                    "state_id": f"{trajectory_id}:decision:002",
                    "trajectory_id": trajectory_id,
                }
            )
    state_payload = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode()
        for row in sorted(states, key=lambda row: row["state_id"])
    )
    text_payload = b"".join(
        (json.dumps({"key": key, "text": value}, sort_keys=True) + "\n").encode()
        for key, value in sorted(texts.items())
    )
    (root / "states.jsonl").write_bytes(state_payload)
    (root / "texts.jsonl").write_bytes(text_payload)
    manifest = {
        "content_sha256": hashlib.sha256(state_payload + text_payload).hexdigest(),
        "evaluation_labels_included": False,
        "label_source_revisions": ["a" * 40],
        "numeric_feature_names": ["age"],
        "snapshot_id": "parent",
        "source_manifest_sha256": "b" * 64,
        "states_jsonl": "states.jsonl",
        "status": "COMPLETED_MERGED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT",
        "texts_jsonl": "texts.jsonl",
        "visual_token_profile": "full",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_filter_snapshot_selects_trajectory_fraction_and_keeps_tune(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_input_snapshot(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter",
            "--input-root",
            str(source),
            "--output-root",
            str(output),
            "--snapshot-id",
            "tuning25",
            "--selection-seed",
            "seed",
            "--train-fraction",
            "0.25",
        ],
    )
    filter_main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["selection"]["contract"] == "TUNING_ONLY_FROZEN_TRAJECTORY_FRACTION"
    assert len(manifest["selection"]["selected_trajectory_ids"]["train"]) == 2
    assert len(manifest["selection"]["selected_trajectory_ids"]["tune"]) == 2
    assert manifest["role_counts"] == {"train": 2, "tune": 2}
    assert manifest["text_count"] == 8


def test_partial_visual_exports_merge_without_overlap(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    input_root = tmp_path / "input"
    _write_input_snapshot(input_root)
    states = [
        json.loads(line)
        for line in (input_root / "states.jsonl").read_text().splitlines()
    ]
    left_source = tmp_path / "left-source" / "token-shards"
    right_source = tmp_path / "right-source" / "token-shards"
    left_source.mkdir(parents=True)
    right_source.mkdir(parents=True)
    for index, state in enumerate(states):
        source = left_source if index % 2 == 0 else right_source
        trajectory_id = state["trajectory_id"]
        safetensors.save_file(
            {
                f"tokens__{trajectory_id}__000": torch.zeros(
                    (2, 4), dtype=torch.bfloat16
                ),
                f"tokens__{trajectory_id}__001": torch.ones(
                    (2, 4), dtype=torch.bfloat16
                ),
            },
            str(source / f"shard-{state['logical_shard']:03d}-of-256.safetensors"),
        )

    exports = []
    for alias, source in (("left", left_source.parent), ("right", right_source.parent)):
        output = tmp_path / f"{alias}-export"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "export",
                "--input-root",
                str(input_root),
                "--source-token-root",
                str(source),
                "--output-root",
                str(output),
                "--host-alias",
                alias,
                "--allow-partial",
            ],
        )
        export_main()
        exports.append(output)
    merged = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge",
            "--input-roots",
            *(str(root) for root in exports),
            "--output-root",
            str(merged),
        ],
    )
    merge_main()
    manifest = json.loads((merged / "visual_manifest.json").read_text())
    assert manifest["visual_count"] == 20
    assert len(manifest["shards"]) == 10
