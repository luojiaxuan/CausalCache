from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalcache.set_utility_action_stability_diagnostic_v2 import (
    STATE_IDS,
    STATE_WAVES,
)
from causalcache.set_utility_action_stability_envelope_v2 import (
    HISTORICAL_D1_AGGREGATE_SHA256,
    VALIDATION_STATUS,
)
from scripts.run_set_utility_action_stability_state_v2 import main


GPU_UUID = "GPU-1111111111111111-2222-3333-4444-555555555555"


def _projection(tmp_path: Path, envelope_path: Path) -> dict[str, Any]:
    state_id = STATE_WAVES[1][0]
    root = tmp_path / "repo"
    return {
        "envelope_path": str(envelope_path),
        "historical_d1_aggregate_path": str(tmp_path / "d1.json"),
        "model_dir": str(tmp_path / "model"),
        "parent_envelope_path": str(tmp_path / "parent.json"),
        "processor_root": str(tmp_path / "processor"),
        "repository_root": str(root),
        "run_root": str(tmp_path / "run"),
        "source_config_path": str(
            root
            / "code/configs/causalcache_set_utility_action_stability_diagnostic_v2.json"
        ),
        "state_processes": [
            {
                "cuda_visible_devices": GPU_UUID,
                "device": "cuda:0",
                "gpu_uuid": GPU_UUID,
                "process_scope": "exactly_one_state_fresh_os_process",
                "state_id": state_id,
                "state_index": STATE_IDS.index(state_id),
                "wave_index": 1,
                "wave_slot": 0,
            }
        ],
        "validation": {
            "current_environment_validated": True,
            "envelope_sha256": "a" * 64,
            "freshness_validated": True,
            "historical_d1_aggregate_sha256": HISTORICAL_D1_AGGREGATE_SHA256,
            "source_config_sha256": "b" * 64,
            "source_inventory_sha256": "c" * 64,
            "state_process_count": 6,
            "status": VALIDATION_STATUS,
            "this_validated_envelope_authorizes_gpu_execution": True,
            "wave_state_ids": [list(wave) for wave in STATE_WAVES],
        },
    }


def test_cli_routes_exactly_one_state_from_fresh_current_environment_projection(
    tmp_path: Path, monkeypatch
) -> None:
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text("{}\n", encoding="utf-8")
    projection = _projection(tmp_path, envelope_path)
    launches = []

    def runner(launch: Any) -> Mapping[str, Any]:
        launches.append(launch)
        return {"state_id": launch.state_id, "status": "ok"}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    result = main(
        [
            "--execution-envelope",
            str(envelope_path),
            "--state-id",
            STATE_WAVES[1][0],
        ],
        envelope_loader=lambda path: projection,
        state_runner=runner,
    )
    assert result == {"state_id": STATE_WAVES[1][0], "status": "ok"}
    assert len(launches) == 1
    assert launches[0].state_id == STATE_WAVES[1][0]
    assert launches[0].wave_index == 1
    assert launches[0].wave_slot == 0
    assert launches[0].gpu_uuid == GPU_UUID
