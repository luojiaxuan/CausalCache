from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.snapshot_running_set_utility_checkpoint import (
    snapshot_running_checkpoint,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_running_checkpoint_snapshot_is_selector_compatible(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    cache_root = tmp_path / "cache"
    training_root = tmp_path / "training"
    output_root = tmp_path / "snapshot"
    config_path = tmp_path / "config.json"
    _write_json(
        input_root / "manifest.json",
        {"content_sha256": "input", "evaluation_labels_included": False},
    )
    _write_json(
        cache_root / "manifest.json",
        {"content_sha256": "cache", "evaluation_labels_included": False},
    )
    _write_json(
        config_path,
        {
            "input": {
                "training_input_content_sha256": "input",
                "contextual_cache_content_sha256": "cache",
            },
            "training": {"seed": 17},
            "variants": {"model": {}},
        },
    )
    training_root.mkdir()
    checkpoint = b"stable-checkpoint"
    (training_root / "best.safetensors").write_bytes(checkpoint)
    summary = snapshot_running_checkpoint(
        input_root=input_root,
        cache_root=cache_root,
        config_path=config_path,
        variant="model",
        training_root=training_root,
        output_root=output_root,
        best_epoch=1,
        best_tune_total=0.25,
    )
    expected = hashlib.sha256(checkpoint).hexdigest()
    assert summary["best_checkpoint"]["sha256"] == expected
    assert summary["evaluation_records_loaded"] is False
    assert (output_root / "best.safetensors").read_bytes() == checkpoint
    assert json.loads((output_root / "summary.json").read_text()) == summary
