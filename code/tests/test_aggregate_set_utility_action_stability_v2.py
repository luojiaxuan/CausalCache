from __future__ import annotations

import hashlib
import json
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
from scripts.aggregate_set_utility_action_stability_v2 import (
    aggregate_action_stability_terminals_v2,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Contract:
    repository_root: Path
    config_sha256: str = "a" * 64
    data: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", {"source": {"inventory_sha256": "b" * 64}})


def _launch(run_root: Path, *, state_id: str) -> execution.ActionStabilityStateLaunchV2:
    base = run_root.parent
    wave_index = 0 if state_id in STATE_WAVES[0] else 1
    return execution.ActionStabilityStateLaunchV2(
        repository_root=(base / "repo").resolve(),
        source_config_path=envelope.CANONICAL_CONFIG_PATH,
        execution_envelope_path=(base / "envelope.json").resolve(),
        parent_envelope_path=(base / "parent.json").resolve(),
        historical_d1_aggregate_path=(ROOT / envelope.HISTORICAL_D1_AGGREGATE_PATH).resolve(),
        processor_root=(base / "processor").resolve(),
        model_dir=(base / "model").resolve(),
        output_root=run_root.resolve(),
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


def _write_run(
    run_root: Path,
    *,
    condition_failure_state: str | None = None,
    outer_failure_state: str | None = None,
) -> None:
    validated = SimpleNamespace(
        candidate_schedule_bytes=b"schedule",
        model_inventory_sha256=execution.PARENT_MODEL_INVENTORY_SHA256,
        processor_inventory_sha256=execution.PARENT_PROCESSOR_INVENTORY_SHA256,
    )
    for wave_index, wave in enumerate(STATE_WAVES):
        for state_id in wave:
            launch = _launch(run_root, state_id=state_id)
            contract = _Contract(launch.repository_root)
            state = SimpleNamespace(
                state_id=state_id, query=object(), joined_input=object()
            )
            bundle = execution.ActionStabilityRuntimeBundleV2(
                runtime=object(),
                reference_input_builder_for=lambda joined: lambda plan: (joined, plan),
                runtime_identity_sha256="f" * 64,
            )
            kwargs: dict[str, Any] = {
                "source_contract_loader": lambda **_: contract,
                "parent_binding_loader": lambda current: (object(), validated),
                "semantic_loader": lambda _validated, _contract, state=state: [state],
                "runtime_factory": lambda current, _validated, _contract, bundle=bundle: bundle,
                "condition_runner": (
                    lambda query, state_id=state_id, **_: {
                        "conditions": [
                            _condition(
                                failure_class=(
                                    "OutOfMemoryError"
                                    if state_id == condition_failure_state
                                    else None
                                )
                            )
                        ],
                        "metric_safe": True,
                        "profile": SDPA_NUMERICAL_CONTROL_PROFILE,
                        "state_id": state_id,
                    }
                ),
                "stage_gate": (
                    (lambda current: None)
                    if wave_index == 0
                    else execution.require_prior_wave_complete_v2
                ),
            }
            if state_id == outer_failure_state:
                kwargs["runtime_factory"] = lambda *args: (_ for _ in ()).throw(
                    RuntimeError("model load failed")
                )
            execution.run_action_stability_state_v2(launch, **kwargs)


def _projection(run_root: Path) -> dict[str, Any]:
    return {
        "historical_d1_aggregate_path": str(
            (ROOT / envelope.HISTORICAL_D1_AGGREGATE_PATH).resolve()
        ),
        "run_root": str(run_root.resolve()),
        "validation": {
            "envelope_sha256": "c" * 64,
            "historical_d1_aggregate_sha256": HISTORICAL_D1_AGGREGATE_SHA256,
            "source_config_sha256": "a" * 64,
            "source_inventory_sha256": "b" * 64,
            "state_process_count": 6,
            "status": VALIDATION_STATUS,
            "this_validated_envelope_authorizes_gpu_execution": True,
            "wave_state_ids": [list(wave) for wave in STATE_WAVES],
        },
    }


@pytest.fixture(autouse=True)
def _parent_candidate_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        execution,
        "PARENT_CANDIDATE_SCHEDULE_SHA256",
        hashlib.sha256(b"schedule").hexdigest(),
    )


def test_exact_six_process_aggregate_passes_and_excludes_historical_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envelope, "PERSISTENT_DATA_ROOT", tmp_path)
    run_root = tmp_path / "run"
    _write_run(run_root)
    result = aggregate_action_stability_terminals_v2(_projection(run_root))
    assert result["verdict"] == "PASS_MEMORY_SAFE_SDPA_NUMERICAL_CONTROL_REPEAT_STABILITY"
    assert result["counts"]["generation_call_count"] == 12
    assert result["counts"]["encode_call_count"] == 6
    assert result["counts"]["historical_context_calls_counted"] is False
    assert result["historical_d1"]["eager_rows_consumed"] is False
    assert len(result["state_terminal_hashes"]) == 6


@pytest.mark.parametrize("outer", [False, True])
def test_condition_or_outer_process_failure_aggregates_to_invalid_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outer: bool
) -> None:
    monkeypatch.setattr(envelope, "PERSISTENT_DATA_ROOT", tmp_path)
    run_root = tmp_path / "run"
    failed = STATE_WAVES[0][0]
    _write_run(
        run_root,
        condition_failure_state=None if outer else failed,
        outer_failure_state=failed if outer else None,
    )
    result = aggregate_action_stability_terminals_v2(_projection(run_root))
    assert result["verdict"] == "INVALID_RUNTIME_FAILURE"
    expected_generation = 10 if outer else 11
    expected_encode = 5 if outer else 6
    assert result["counts"]["generation_call_count"] == expected_generation
    assert result["counts"]["encode_call_count"] == expected_encode


def test_aggregate_rejects_duplicate_fresh_process_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envelope, "PERSISTENT_DATA_ROOT", tmp_path)
    run_root = tmp_path / "run"
    _write_run(run_root)
    layout = envelope.canonical_run_layout(run_root)
    first = layout["states"][STATE_IDS.index(STATE_WAVES[0][0])]
    second = layout["states"][STATE_IDS.index(STATE_WAVES[1][0])]
    first_terminal = json.loads(Path(first["terminal_path"]).read_text(encoding="utf-8"))
    attempt_path = Path(second["attempt_path"])
    terminal_path = Path(second["terminal_path"])
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    process_sha = first_terminal["process_identity_sha256"]
    attempt["process_identity_sha256"] = process_sha
    attempt_bytes = canonical_pretty_json_bytes(attempt)
    attempt_path.write_bytes(attempt_bytes)
    terminal["attempt_sha256"] = sha256_bytes(attempt_bytes)
    terminal["process_identity_sha256"] = process_sha
    terminal_path.write_bytes(canonical_pretty_json_bytes(terminal))
    with pytest.raises(ValueError, match="distinct fresh process"):
        aggregate_action_stability_terminals_v2(_projection(run_root))
