from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import causalcache.set_utility_action_stability_envelope_v1 as envelope_module
from causalcache.set_utility_action_stability_contract_v1 import (
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    STATE_IDS,
    state_ids_for_worker_v1,
)
from causalcache.set_utility_action_stability_envelope_v1 import (
    PARENT_EXECUTION_ENVELOPE_SHA256,
    canonical_run_layout,
)
from causalcache.set_utility_action_stability_execution_v1 import (
    TERMINAL_PROTOCOL_ID,
)


ROOT = Path(__file__).resolve().parents[2]


def _module() -> ModuleType:
    path = ROOT / "code/scripts/aggregate_set_utility_action_stability_v1.py"
    spec = importlib.util.spec_from_file_location("d1_aggregate_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _condition(condition_id: str) -> dict[str, Any]:
    fresh = condition_id == "auto_fresh_encode"
    return {
        "canonical_action_equal": True,
        "condition_id": condition_id,
        "decoded_output_equal": True,
        "encode_call_count": 2 if fresh else 1,
        "encoded_input_unchanged_after": None if fresh else True,
        "encoded_input_unchanged_before": None if fresh else True,
        "encoded_input_unchanged_between": None if fresh else True,
        "exact_generated_sequence_equal": True,
        "failure_class": None,
        "generation_call_count": 2,
        "generation_completed_count": 2,
        "metric_safe": True,
        "repeat_count": 2,
    }


def _partial(state_id: str, profile: str) -> dict[str, Any]:
    return {
        "conditions": (
            [
                _condition("auto_fresh_encode"),
                _condition("auto_frozen_encoded"),
            ]
            if profile == "auto"
            else [_condition("eager_frozen_encoded_control")]
        ),
        "metric_safe": True,
        "profile": (
            "auto_default"
            if profile == "auto"
            else "eager_numerical_control_not_strict_cuda_determinism"
        ),
        "state_id": state_id,
    }


def _write_terminals(run_root: Path) -> dict[str, Any]:
    layout = canonical_run_layout(run_root)
    envelope_sha = "a" * 64
    config_sha = "b" * 64
    inventory_sha = "c" * 64
    for profile_index, profile in enumerate(("auto", "eager")):
        for worker_index in range(4):
            state_ids = list(state_ids_for_worker_v1(worker_index))
            paths = layout["workers"][worker_index]["profiles"][profile]
            attempt_path = Path(paths["attempt_path"])
            terminal_path = Path(paths["terminal_path"])
            attempt_path.parent.mkdir(parents=True, exist_ok=True)
            attempt = {"profile": profile, "worker_index": worker_index}
            attempt_payload = canonical_pretty_json_bytes(attempt)
            attempt_path.write_bytes(attempt_payload)
            count = len(state_ids)
            terminal = {
                "attempt_sha256": sha256_bytes(attempt_payload),
                "counts": {
                    "condition_completed_count": (2 if profile == "auto" else 1) * count,
                    "encode_call_count": (3 if profile == "auto" else 1) * count,
                    "generation_call_count": (4 if profile == "auto" else 2) * count,
                    "label_count": 0,
                    "partial_state_completed_count": count,
                    "partial_state_expected_count": count,
                    "restoration_distance_count": 0,
                    "retry_count": 0,
                    "teacher_forward_call_count": 0,
                    "training_example_count": 0,
                },
                "failure_class": None,
                "fresh_execution_envelope_sha256": envelope_sha,
                "metric_safe": True,
                "parent_execution_envelope_sha256": PARENT_EXECUTION_ENVELOPE_SHA256,
                "partial_results": [_partial(state_id, profile) for state_id in state_ids],
                "profile": profile,
                "protocol_id": TERMINAL_PROTOCOL_ID,
                "runtime_identity_sha256": f"{worker_index + profile_index * 4:064x}",
                "schema_version": "1.0.0",
                "source_config_sha256": config_sha,
                "source_inventory_sha256": inventory_sha,
                "state_ids": state_ids,
                "status": "COMPLETED_ACTION_STABILITY_PROFILE_WORKER_V1",
                "worker_index": worker_index,
            }
            terminal_path.write_bytes(canonical_pretty_json_bytes(terminal))
            timestamp = 1_700_000_000 + profile_index * 10
            os.utime(attempt_path, (timestamp, timestamp))
            os.utime(terminal_path, (timestamp + 1, timestamp + 1))
    return {
        "envelope_path": str(run_root / "execution-envelope.json"),
        "run_root": str(run_root),
        "validation": {
            "envelope_sha256": envelope_sha,
            "source_config_sha256": config_sha,
            "source_inventory_sha256": inventory_sha,
            "status": "VALID_SET_UTILITY_ACTION_STABILITY_EXECUTION_ENVELOPE_V1",
        },
    }


def test_exact_eight_terminal_aggregate_calls_core_reducer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envelope_module, "PERSISTENT_DATA_ROOT", tmp_path)
    run_root = tmp_path / "runs" / "d1"
    projection = _write_terminals(run_root)
    aggregate = _module().aggregate_action_stability_terminals_v1(projection)

    assert aggregate["counts"] == {
        "encode_call_ceiling": 24,
        "encode_call_count": 24,
        "generation_call_ceiling": 36,
        "generation_call_count": 36,
        "profile_worker_terminal_count": 8,
        "retry_count": 0,
        "state_count": 6,
    }
    assert aggregate["diagnostic"]["verdict"] == "PARENT_MISMATCH_NOT_REPRODUCED"
    assert [state["state_id"] for state in aggregate["diagnostic"]["states"]] == list(STATE_IDS)


def test_aggregate_rejects_eager_before_auto_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envelope_module, "PERSISTENT_DATA_ROOT", tmp_path)
    run_root = tmp_path / "runs" / "d1"
    projection = _write_terminals(run_root)
    layout = canonical_run_layout(run_root)
    eager_attempt = Path(layout["workers"][0]["profiles"]["eager"]["attempt_path"])
    os.utime(eager_attempt, (1, 1))

    with pytest.raises(ValueError, match="before all auto"):
        _module().aggregate_action_stability_terminals_v1(projection)


def test_condition_failure_reaches_invalid_runtime_aggregate_below_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envelope_module, "PERSISTENT_DATA_ROOT", tmp_path)
    run_root = tmp_path / "runs" / "d1"
    projection = _write_terminals(run_root)
    layout = canonical_run_layout(run_root)
    terminal_path = Path(
        layout["workers"][0]["profiles"]["auto"]["terminal_path"]
    )
    terminal = json.loads(terminal_path.read_text())
    condition = terminal["partial_results"][0]["conditions"][0]
    for key in (
        "canonical_action_equal",
        "decoded_output_equal",
        "exact_generated_sequence_equal",
    ):
        condition[key] = None
    condition["encode_call_count"] = 1
    condition["failure_class"] = "RuntimeError"
    condition["generation_call_count"] = 1
    condition["generation_completed_count"] = 0
    terminal["counts"]["encode_call_count"] -= 1
    terminal["counts"]["generation_call_count"] -= 1
    terminal_path.write_bytes(canonical_pretty_json_bytes(terminal))
    os.utime(terminal_path, (1_700_000_001, 1_700_000_001))

    aggregate = _module().aggregate_action_stability_terminals_v1(projection)

    assert aggregate["diagnostic"]["verdict"] == "INVALID_RUNTIME_FAILURE"
    assert aggregate["counts"]["encode_call_count"] == 23
    assert aggregate["counts"]["generation_call_count"] == 35
