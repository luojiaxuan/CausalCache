from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from causalcache.set_utility_action_stability_execution_v1 import (
    CANONICAL_D1_CONFIG_PATH,
    FRESH_ENVELOPE_VALIDATION_STATUS,
    PARENT_EXECUTION_ENVELOPE_SHA256,
)
from scripts.run_set_utility_action_stability_worker_v1 import main


GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _projection(tmp_path: Path, envelope: Path) -> dict[str, Any]:
    repository = tmp_path / "repository"
    processor = tmp_path / "processor"
    model = tmp_path / "model"
    for path in (repository, processor, model):
        path.mkdir()
    return {
        "repository_root": str(repository),
        "source_config_path": str(repository / CANONICAL_D1_CONFIG_PATH),
        "envelope_path": str(envelope),
        "run_root": str(tmp_path / "fresh-run"),
        "parent_envelope_path": str(tmp_path / "parent-envelope.json"),
        "processor_root": str(processor),
        "model_dir": str(model),
        "device_by_worker": {str(index): "cuda:0" for index in range(4)},
        "worker_gpu_uuid": {str(index): GPU_UUID for index in range(4)},
        "profiles": ["auto", "eager"],
        "validation": {
            "status": FRESH_ENVELOPE_VALIDATION_STATUS,
            "worker_count": 4,
            "profiles": ["auto", "eager"],
            "freshness_validated": True,
            "this_validated_envelope_authorizes_gpu_execution": True,
            "envelope_sha256": "a" * 64,
            "source_config_sha256": "b" * 64,
            "source_inventory_sha256": "c" * 64,
            "parent_envelope_sha256": PARENT_EXECUTION_ENVELOPE_SHA256,
        },
    }


def test_cli_consumes_fresh_projection_and_routes_one_profile_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    envelope = tmp_path / "execution-envelope.json"
    projection = _projection(tmp_path, envelope)
    loader_paths: list[Path] = []
    launches: list[Any] = []

    def loader(path: str | Path) -> Mapping[str, Any]:
        loader_paths.append(Path(path))
        return projection

    def runner(launch: Any) -> Mapping[str, Any]:
        launches.append(launch)
        return {
            "metric_safe": True,
            "profile": launch.profile,
            "worker_index": launch.worker_index,
        }

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    result = main(
        [
            "--execution-envelope",
            str(envelope),
            "--profile",
            "eager",
            "--worker-index",
            "2",
        ],
        envelope_loader=loader,
        worker_runner=runner,
    )

    assert loader_paths == [envelope]
    assert len(launches) == 1
    launch = launches[0]
    assert launch.profile == "eager"
    assert launch.worker_index == 2
    assert launch.device == "cuda:0"
    assert launch.worker_gpu_uuid == GPU_UUID
    assert launch.output_root == tmp_path / "fresh-run"
    assert result == {"metric_safe": True, "profile": "eager", "worker_index": 2}
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("drift", ("uuid", "freshness", "parent"))
def test_cli_rejects_non_authorizing_or_wrong_gpu_projection_before_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    envelope = tmp_path / "execution-envelope.json"
    projection = _projection(tmp_path, envelope)
    if drift == "uuid":
        projection["worker_gpu_uuid"]["0"] = (
            "GPU-ffffffff-bbbb-cccc-dddd-eeeeeeeeeeee"
        )
    elif drift == "freshness":
        projection["validation"]["freshness_validated"] = False
    else:
        projection["validation"]["parent_envelope_sha256"] = "0" * 64
    runner_calls = 0

    def runner(_: Any) -> Mapping[str, Any]:
        nonlocal runner_calls
        runner_calls += 1
        return {}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    with pytest.raises(PermissionError):
        main(
            [
                "--execution-envelope",
                str(envelope),
                "--profile",
                "auto",
                "--worker-index",
                "0",
            ],
            envelope_loader=lambda _: projection,
            worker_runner=runner,
        )
    assert runner_calls == 0


def test_cli_rejects_source_config_and_profile_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = tmp_path / "execution-envelope.json"
    projection = _projection(tmp_path, envelope)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    projection["source_config_path"] = str(tmp_path / "other.json")
    with pytest.raises(PermissionError):
        main(
            [
                "--execution-envelope",
                str(envelope),
                "--profile",
                "auto",
                "--worker-index",
                "0",
            ],
            envelope_loader=lambda _: projection,
        )
