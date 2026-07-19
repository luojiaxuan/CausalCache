from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from causalcache.set_utility_action_stability_diagnostic_v2 import (
    SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
    SDPA_NUMERICAL_CONTROL_PROFILE,
    STATE_IDS,
    STATE_WAVES,
)
from causalcache.set_utility_action_stability_envelope_v2 import (
    HISTORICAL_D1_AGGREGATE_SHA256,
    VALIDATION_STATUS,
)
from causalcache import set_utility_action_stability_envelope_v2 as envelope
from causalcache import set_utility_action_stability_execution_v2 as execution


@dataclass(frozen=True)
class _Contract:
    repository_root: Path
    config_sha256: str = "a" * 64
    data: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", {"source": {"inventory_sha256": "b" * 64}})


def _launch(tmp_path: Path, *, state_id: str) -> execution.ActionStabilityStateLaunchV2:
    wave_index = 0 if state_id in STATE_WAVES[0] else 1
    return execution.ActionStabilityStateLaunchV2(
        repository_root=(tmp_path / "repo").resolve(),
        source_config_path=envelope.CANONICAL_CONFIG_PATH,
        execution_envelope_path=(tmp_path / "envelope.json").resolve(),
        parent_envelope_path=(tmp_path / "parent.json").resolve(),
        historical_d1_aggregate_path=(tmp_path / "d1.json").resolve(),
        processor_root=(tmp_path / "processor").resolve(),
        model_dir=(tmp_path / "model").resolve(),
        output_root=(tmp_path / "run").resolve(),
        state_id=state_id,
        state_index=STATE_IDS.index(state_id),
        wave_index=wave_index,
        wave_slot=STATE_WAVES[wave_index].index(state_id),
        device="cuda:0",
        gpu_uuid="GPU-1111111111111111-2222-3333-4444-555555555555",
        execution_envelope_sha256="c" * 64,
        source_config_sha256="a" * 64,
        source_inventory_sha256="b" * 64,
        historical_d1_aggregate_sha256=HISTORICAL_D1_AGGREGATE_SHA256,
        validation_status=VALIDATION_STATUS,
        gpu_execution_authorized=True,
    )


def _condition(*, failure_class: str | None = None) -> dict[str, Any]:
    completed = failure_class is None
    return {
        "canonical_action_equal": True if completed else None,
        "condition_id": SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
        "decoded_output_equal": True if completed else None,
        "encode_call_count": 1,
        "encoded_input_unchanged_after": True,
        "encoded_input_unchanged_before": True,
        "encoded_input_unchanged_between": True if completed else None,
        "exact_generated_sequence_equal": True if completed else None,
        "failure_class": failure_class,
        "generation_call_count": 2 if completed else 1,
        "generation_completed_count": 2 if completed else 0,
        "metric_safe": True,
        "repeat_count": 2,
    }


def _partial(state_id: str, *, failure_class: str | None = None) -> dict[str, Any]:
    return {
        "conditions": [_condition(failure_class=failure_class)],
        "metric_safe": True,
        "profile": SDPA_NUMERICAL_CONTROL_PROFILE,
        "state_id": state_id,
    }


def _dependencies(launch: execution.ActionStabilityStateLaunchV2):
    validated = SimpleNamespace(
        candidate_schedule_bytes=b"schedule",
        model_inventory_sha256=execution.PARENT_MODEL_INVENTORY_SHA256,
        processor_inventory_sha256=execution.PARENT_PROCESSOR_INVENTORY_SHA256,
    )
    contract = _Contract(launch.repository_root)
    state = SimpleNamespace(
        state_id=launch.state_id,
        query=object(),
        joined_input=object(),
    )
    bundle = execution.ActionStabilityRuntimeBundleV2(
        runtime=object(),
        reference_input_builder_for=lambda joined: lambda plan: (joined, plan),
        runtime_identity_sha256="f" * 64,
    )
    return {
        "source_contract_loader": lambda **_: contract,
        "parent_binding_loader": lambda current: (object(), validated),
        "semantic_loader": lambda _validated, _contract: [state],
        "runtime_factory": lambda current, _validated, _contract: bundle,
        "condition_runner": lambda query, **kwargs: _partial(launch.state_id),
        "stage_gate": lambda current: None,
    }


def test_state_runner_claims_one_state_once_and_preserves_class_only_condition_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envelope, "PERSISTENT_DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        execution,
        "PARENT_CANDIDATE_SCHEDULE_SHA256",
        hashlib.sha256(b"schedule").hexdigest(),
    )
    launch = _launch(tmp_path, state_id=STATE_WAVES[0][0])
    dependencies = _dependencies(launch)
    dependencies["condition_runner"] = lambda query, **kwargs: _partial(
        launch.state_id, failure_class="OutOfMemoryError"
    )
    terminal = execution.run_action_stability_state_v2(launch, **dependencies)
    assert terminal["status"] == execution.COMPLETED_STATUS
    assert terminal["failure_class"] is None
    assert terminal["partial_result"]["conditions"][0]["failure_class"] == "OutOfMemoryError"
    assert terminal["counts"]["generation_call_count"] == 1
    assert terminal["counts"]["retry_count"] == 0
    layout = envelope.canonical_run_layout(launch.output_root)
    attempt, _ = execution._strict_canonical(
        Path(layout["states"][launch.state_index]["attempt_path"]), label="attempt"
    )
    execution.validate_attempt_payload_v2(
        attempt,
        state_id=launch.state_id,
        envelope_sha256=launch.execution_envelope_sha256,
        source_config_sha256=launch.source_config_sha256,
        source_inventory_sha256=launch.source_inventory_sha256,
    )
    attempt["profile"] = "mutated"
    with pytest.raises(ValueError, match="execution boundary"):
        execution.validate_attempt_payload_v2(
            attempt,
            state_id=launch.state_id,
            envelope_sha256=launch.execution_envelope_sha256,
            source_config_sha256=launch.source_config_sha256,
            source_inventory_sha256=launch.source_inventory_sha256,
        )
    with pytest.raises(FileExistsError):
        execution.run_action_stability_state_v2(launch, **dependencies)


def test_state_runner_records_pre_condition_process_failure_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envelope, "PERSISTENT_DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        execution,
        "PARENT_CANDIDATE_SCHEDULE_SHA256",
        hashlib.sha256(b"schedule").hexdigest(),
    )
    launch = _launch(tmp_path, state_id=STATE_WAVES[0][0])
    dependencies = _dependencies(launch)
    dependencies["runtime_factory"] = lambda *args: (_ for _ in ()).throw(
        RuntimeError("load failed")
    )
    terminal = execution.run_action_stability_state_v2(launch, **dependencies)
    assert terminal["status"] == execution.FAILED_STATUS
    assert terminal["failure_class"] == "RuntimeError"
    assert terminal["partial_result"] is None
    assert terminal["runtime_identity_sha256"] is None
    assert terminal["counts"]["encode_call_count"] == 0
    assert terminal["counts"]["generation_call_count"] == 0
    validated, partial = execution.validate_terminal_payload_v2(
        terminal,
        state_id=launch.state_id,
        envelope_sha256=launch.execution_envelope_sha256,
        source_config_sha256=launch.source_config_sha256,
        source_inventory_sha256=launch.source_inventory_sha256,
    )
    assert validated["failure_class"] == "RuntimeError"
    assert partial is None


def test_launch_requires_fresh_current_environment_receipt_and_exact_gpu(
    tmp_path: Path,
) -> None:
    state_id = STATE_WAVES[1][0]
    launch = _launch(tmp_path, state_id=state_id)
    process = {
        "cuda_visible_devices": launch.gpu_uuid,
        "device": "cuda:0",
        "gpu_uuid": launch.gpu_uuid,
        "process_scope": "exactly_one_state_fresh_os_process",
        "state_id": state_id,
        "state_index": launch.state_index,
        "wave_index": launch.wave_index,
        "wave_slot": launch.wave_slot,
    }
    projection = {
        "envelope_path": str(launch.execution_envelope_path),
        "historical_d1_aggregate_path": str(launch.historical_d1_aggregate_path),
        "model_dir": str(launch.model_dir),
        "parent_envelope_path": str(launch.parent_envelope_path),
        "processor_root": str(launch.processor_root),
        "repository_root": str(launch.repository_root),
        "run_root": str(launch.output_root),
        "source_config_path": str(
            launch.repository_root / envelope.CANONICAL_CONFIG_PATH
        ),
        "state_processes": [process],
        "validation": {
            "current_environment_validated": True,
            "envelope_sha256": launch.execution_envelope_sha256,
            "freshness_validated": True,
            "historical_d1_aggregate_sha256": HISTORICAL_D1_AGGREGATE_SHA256,
            "source_config_sha256": launch.source_config_sha256,
            "source_inventory_sha256": launch.source_inventory_sha256,
            "state_process_count": 6,
            "status": VALIDATION_STATUS,
            "this_validated_envelope_authorizes_gpu_execution": True,
            "wave_state_ids": [list(wave) for wave in STATE_WAVES],
        },
    }
    built = execution.launch_from_fresh_projection_v2(
        projection,
        execution_envelope_path=launch.execution_envelope_path,
        state_id=state_id,
        visible_gpu_uuid=launch.gpu_uuid,
    )
    assert built.state_id == state_id
    projection["validation"]["freshness_validated"] = False
    with pytest.raises(PermissionError, match="not authorizing"):
        execution.launch_from_fresh_projection_v2(
            projection,
            execution_envelope_path=launch.execution_envelope_path,
            state_id=state_id,
            visible_gpu_uuid=launch.gpu_uuid,
        )


def test_wave_two_gate_requires_exact_four_wave_one_terminals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envelope, "PERSISTENT_DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        execution,
        "PARENT_CANDIDATE_SCHEDULE_SHA256",
        hashlib.sha256(b"schedule").hexdigest(),
    )
    for state_id in STATE_WAVES[0]:
        launch = _launch(tmp_path, state_id=state_id)
        execution.run_action_stability_state_v2(launch, **_dependencies(launch))
    wave_two = _launch(tmp_path, state_id=STATE_WAVES[1][0])
    execution.require_prior_wave_complete_v2(wave_two)
    layout = envelope.canonical_run_layout(wave_two.output_root)
    Path(layout["states"][STATE_IDS.index(STATE_WAVES[0][0])]["terminal_path"]).unlink()
    with pytest.raises(ValueError):
        execution.require_prior_wave_complete_v2(wave_two)
