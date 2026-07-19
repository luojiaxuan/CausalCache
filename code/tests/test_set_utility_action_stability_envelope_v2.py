from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from causalcache.set_utility_action_stability_contract_v1 import (
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_action_stability_diagnostic_v2 import (
    STATE_IDS,
    STATE_WAVES,
)
from causalcache import set_utility_action_stability_envelope_v2 as envelope


@dataclass(frozen=True)
class _Contract:
    repository_root: Path
    config_sha256: str
    data: Mapping[str, Any]


def _source_data(*, execution_authorized: bool = False) -> dict[str, Any]:
    return {
        "authorization": {"execution_authorized": execution_authorized},
        "diagnostic": {
            "condition_id": envelope.CONDITION_ID,
            "encode_call_ceiling": 6,
            "generation_call_ceiling": 12,
            "repeat_count": 2,
            "state_ids": list(STATE_IDS),
        },
        "execution_boundary": {
            "max_concurrent_state_processes": 4,
            "no_retry": True,
            "no_top_up": True,
            "process_per_state": True,
            "state_process_count": 6,
            "waves": [list(wave) for wave in STATE_WAVES],
        },
        "immutable_inputs": {
            "parent_d1_aggregate": {
                "path": envelope.HISTORICAL_D1_AGGREGATE_PATH,
                "sha256": envelope.HISTORICAL_D1_AGGREGATE_SHA256,
                "status": envelope.HISTORICAL_D1_STATUS,
                "verdict": envelope.HISTORICAL_D1_VERDICT,
            }
        },
        "source": {"inventory_sha256": "b" * 64},
    }


def _gpu(index: int) -> dict[str, Any]:
    return {
        "host_index": index,
        "name": "NVIDIA H200",
        "sample_window_seconds": 10,
        "sampled_utilization_percent": [0, 0, 0],
        "total_memory_bytes": 150_000_000_000,
        "uuid": f"GPU-{index:016x}-1111-2222-3333-444444444444",
        "visible_index": index,
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "repo"
    persistent = tmp_path / "data"
    run_root = persistent / "run"
    config_path = root / envelope.CANONICAL_CONFIG_PATH
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(b"{}\n")
    for relative in (envelope.STATE_ENTRYPOINT, envelope.AGGREGATE_ENTRYPOINT):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# placeholder\n", encoding="utf-8")
    python = root / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    cleanup = run_root / "preflight-raw.log"
    cleanup.parent.mkdir(parents=True)
    cleanup.write_text(
        "\n".join(
            (
                "PREFLIGHT_STARTED_AT_UTC=2026-07-19T12:00:00Z",
                "=== Sampling GPU Utilization (10s continuous-zero window, 1s interval) ===",
                "killed_container_csv=stale-container",
                "selected_gpu_csv=0,1,2,3",
                "Reusing the 10s preflight sample",
                "PREFLIGHT_COMPLETED_AT_UTC=2026-07-19T12:00:10Z",
                "",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(envelope, "PERSISTENT_DATA_ROOT", persistent)
    historical = {
        "aggregate": {"path": "/bound/d1.json", "sha256": "c" * 64, "size_bytes": 1},
        "container": {
            "id": "d" * 64,
            "image_digest": "sha256:" + "e" * 64,
            "image_reference": "hongccc/sglang-omni:dev",
        },
        "execution_envelope": {
            "path": "/bound/d1-envelope.json",
            "sha256": envelope.HISTORICAL_D1_ENVELOPE_SHA256,
            "size_bytes": 1,
        },
        "host": {"alias": "hyper00", "hostname": "node-radixark-16-0000"},
        "model_dir": "/bound/model",
        "parent_execution_envelope": {"path": "/bound/parent.json", "sha256": "f" * 64},
        "processor_root": "/bound/processor",
        "runtime": {
            "python_executable": str(python),
            "software_versions": {
                "cuda_runtime_version": "13.0",
                "python_version": "3.12.3",
                "torch_version": "2.11.0+cu130",
                "transformers_version": "5.6.0",
            },
        },
    }
    monkeypatch.setattr(envelope, "_historical_inputs", lambda _root: historical)
    contract = _Contract(
        repository_root=root.resolve(),
        config_sha256=sha256_bytes(config_path.read_bytes()),
        data=_source_data(),
    )
    evidence = envelope.build_action_stability_preflight_evidence_v2(
        raw_cleanup_log_path=cleanup,
        started_at_utc="2026-07-19T12:00:00Z",
        completed_at_utc="2026-07-19T12:00:10Z",
        host_alias="hyper00",
        hostname="node-radixark-16-0000",
        container_id="d" * 64,
        driver_version="570.172.08",
        selected_gpus=[_gpu(index) for index in range(4)],
        killed_containers=["stale-container"],
    )
    evidence_path = Path(envelope.canonical_run_layout(run_root)["preflight_evidence_path"])
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(canonical_pretty_json_bytes(evidence))
    kwargs = {
        "repository_root": root,
        "source_git_revision": "a" * 40,
        "run_root": run_root,
        "python_executable": str(python),
        "preflight_evidence_path": evidence_path,
        "host_alias": "hyper00",
        "hostname": "node-radixark-16-0000",
        "container_id": "d" * 64,
        "container_image_reference": "hongccc/sglang-omni:dev",
        "container_image_digest": "sha256:" + "e" * 64,
        "driver_version": "570.172.08",
        "software_versions": historical["runtime"]["software_versions"],
        "materialized_at_utc": "2026-07-19T12:00:11Z",
        "source_contract_loader": lambda **_: contract,
    }
    return root, run_root, contract, evidence_path, kwargs


def test_envelope_locks_fresh_preflight_and_noncontiguous_four_plus_two_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run_root, contract, _, kwargs = _fixture(tmp_path, monkeypatch)
    built = envelope.build_action_stability_envelope_v2(**kwargs)
    processes = built["execution"]["state_processes"]
    assert [item["state_id"] for item in processes] == [
        *STATE_WAVES[0],
        *STATE_WAVES[1],
    ]
    assert [(item["wave_index"], item["wave_slot"]) for item in processes] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
    ]
    assert all(item["process_scope"] == "exactly_one_state_fresh_os_process" for item in processes)
    assert built["preflight"]["freshness_maximum_seconds"] == 900
    assert built["authorization"]["source_a_alone_authorizes_execution"] is False
    assert built["artifacts"]["container"] == built["container"]

    validation = envelope.validate_action_stability_envelope_v2(
        built,
        repository_root=root,
        envelope_path=envelope.canonical_run_layout(run_root)["envelope_path"],
        verify_repository=False,
        require_fresh_preflight=True,
        verify_current_environment=False,
        now_utc="2026-07-19T12:14:59Z",
        source_contract_loader=lambda **_: contract,
    )
    assert validation["freshness_validated"] is True
    assert validation["current_environment_validated"] is False


def test_preflight_rejects_busy_gpu_staleness_and_unrecorded_source_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run_root, contract, evidence_path, kwargs = _fixture(tmp_path, monkeypatch)
    built = envelope.build_action_stability_envelope_v2(**kwargs)
    with pytest.raises(ValueError, match="no longer fresh"):
        envelope.validate_action_stability_envelope_v2(
            built,
            repository_root=root,
            envelope_path=envelope.canonical_run_layout(run_root)["envelope_path"],
            verify_repository=False,
            require_fresh_preflight=True,
            now_utc="2026-07-19T12:15:11Z",
            source_contract_loader=lambda **_: contract,
        )

    cleanup_path = Path(json_load(evidence_path)["cleanup"]["path"])
    original_cleanup = cleanup_path.read_text(encoding="utf-8")
    cleanup_path.write_text(
        original_cleanup.replace(
            "selected_gpu_csv=0,1,2,3", "selected_gpu_csv=0,1,2,3,4,5,6,7"
        ),
        encoding="utf-8",
    )
    extra = envelope.build_action_stability_preflight_evidence_v2(
        raw_cleanup_log_path=cleanup_path,
        started_at_utc="2026-07-19T12:00:00Z",
        completed_at_utc="2026-07-19T12:00:10Z",
        host_alias="hyper00",
        hostname="node-radixark-16-0000",
        container_id="d" * 64,
        driver_version="570.172.08",
        selected_gpus=[_gpu(index) for index in range(4)],
        killed_containers=["stale-container"],
    )
    assert [gpu["host_index"] for gpu in extra["gpus"]] == [0, 1, 2, 3]
    cleanup_path.write_text(
        original_cleanup.replace("selected_gpu_csv=0,1,2,3", "selected_gpu_csv=0,1,2"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="omitted a bound"):
        envelope.build_action_stability_preflight_evidence_v2(
            raw_cleanup_log_path=cleanup_path,
            started_at_utc="2026-07-19T12:00:00Z",
            completed_at_utc="2026-07-19T12:00:10Z",
            host_alias="hyper00",
            hostname="node-radixark-16-0000",
            container_id="d" * 64,
            driver_version="570.172.08",
            selected_gpus=[_gpu(index) for index in range(4)],
            killed_containers=["stale-container"],
        )
    cleanup_path.write_text(original_cleanup, encoding="utf-8")

    raw = json_load(evidence_path)
    raw["gpus"][2]["sampled_utilization_percent"] = [0, 1]
    evidence_path.write_bytes(canonical_pretty_json_bytes(raw))
    with pytest.raises(ValueError, match="idle sample"):
        envelope.build_action_stability_envelope_v2(**kwargs)

    forbidden = _Contract(
        repository_root=contract.repository_root,
        config_sha256=contract.config_sha256,
        data=_source_data(execution_authorized=True),
    )
    with pytest.raises(PermissionError, match="Source-A"):
        envelope._source_fields(forbidden)


def json_load(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_committed_envelope_lifecycle_requires_direct_child_and_only_reserved_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    source = "a" * 40
    head = "b" * 40
    expected = {"protocol_id": "test"}
    payload = canonical_pretty_json_bytes(expected)
    git_path = root / envelope.CANONICAL_GIT_ENVELOPE_PATH
    data_path = tmp_path / "execution-envelope.json"
    git_path.parent.mkdir(parents=True)
    git_path.write_bytes(payload)
    data_path.write_bytes(payload)

    def fake_git(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return head
        if args == ("rev-list", "--parents", "-n", "1", head):
            return f"{head} {source}"
        if args[:3] == ("diff", "--name-only", "--no-renames"):
            return envelope.CANONICAL_GIT_ENVELOPE_PATH
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if args == ("rev-parse", "refs/remotes/origin/main"):
            return head
        raise AssertionError(args)

    monkeypatch.setattr(envelope, "_git", fake_git)
    monkeypatch.setattr(
        envelope.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=payload),
    )
    receipt = envelope.validate_committed_envelope_lifecycle_v2(
        repository_root=root,
        source_git_revision=source,
        data_envelope_path=data_path,
        expected_envelope=expected,
    )
    assert receipt == {
        "envelope_git_revision": head,
        "source_git_revision": source,
    }

    def extra_path_git(_root: Path, *args: str) -> str:
        value = fake_git(_root, *args)
        if args[:3] == ("diff", "--name-only", "--no-renames"):
            return value + "\nREADME.md"
        return value

    monkeypatch.setattr(envelope, "_git", extra_path_git)
    with pytest.raises(ValueError, match="direct-child"):
        envelope.validate_committed_envelope_lifecycle_v2(
            repository_root=root,
            source_git_revision=source,
            data_envelope_path=data_path,
            expected_envelope=expected,
        )
