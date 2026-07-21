from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from causalcache.set_utility_heldout_inference import canonical_json_bytes


def _shard(state_id: str) -> dict:
    payload = {
        "cache_content_sha256": "c" * 64,
        "checkpoint_sha256": "p" * 64,
        "config_sha256": "g" * 64,
        "input_content_sha256": "i" * 64,
        "records": [{"state_id": state_id}],
        "role": "tune",
        "schema_version": "test.v1",
        "status": "COMPLETED_SET_UTILITY_TUNE_SELECTIONS",
        "variant": "direct",
    }
    payload["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def test_merge_rejects_overlap_and_preserves_binding(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    inputs = []
    for index, state_id in enumerate(("b", "a")):
        path = tmp_path / f"shard-{index}.json"
        path.write_bytes(canonical_json_bytes(_shard(state_id)) + b"\n")
        inputs.append(path)
    output = tmp_path / "merged.json"
    command = [
        sys.executable,
        str(repository / "code/scripts/merge_set_utility_tune_selection_shards.py"),
        "--input",
        str(inputs[0]),
        "--input",
        str(inputs[1]),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        env={**os.environ, "PYTHONPATH": str(repository / "code")},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    merged = json.loads(output.read_text(encoding="utf-8"))
    assert [record["state_id"] for record in merged["records"]] == ["a", "b"]
    assert merged["checkpoint_sha256"] == "p" * 64

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(canonical_json_bytes(_shard("a")) + b"\n")
    failed = subprocess.run(
        [*command[:-1], str(tmp_path / "unused.json"), "--input", str(duplicate)],
        env={**os.environ, "PYTHONPATH": str(repository / "code")},
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "duplicate tune state" in failed.stderr
