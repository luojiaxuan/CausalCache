from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from causalcache.set_utility_action_stability_contract_v2 import (
    AGGREGATE_VERDICT_PRECEDENCE,
    CANONICAL_CONFIG_PATH,
    CANONICAL_EXECUTION_CONFIG_PATH,
    CLI_PATH,
    D1_AGGREGATE_PATH,
    D1_AGGREGATE_SHA256,
    D1_RESULT_COMMIT,
    D1_SUMMARY_PATH,
    D1_SUMMARY_SHA256,
    ENCODE_CALL_CEILING,
    GENERATION_CALL_CEILING,
    GPU_SLOT_COUNT,
    SOURCE_ENTRYPOINT_CANDIDATES,
    STATE_DIAGNOSIS_PRECEDENCE,
    STATE_IDS,
    STATE_PROCESS_COUNT,
    VALIDATION_STATUS,
    WAVE_STATE_INDICES,
    _parent_state_projection,
    build_action_stability_source_v2_config_skeleton,
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    sha256_bytes,
    validate_action_stability_source_v2_config,
)


ROOT = Path(__file__).resolve().parents[2]


def _skeleton() -> dict[str, object]:
    return build_action_stability_source_v2_config_skeleton(repository_root=ROOT)


def test_frozen_d1_context_is_commit_and_hash_bound_without_eager_backfill() -> None:
    data = _skeleton()
    inputs = data["immutable_inputs"]
    context = data["historical_d1_auto_context"]

    assert inputs["d1_result_commit"] == D1_RESULT_COMMIT
    assert inputs["d1_aggregate"] == {
        "byte_count": (ROOT / D1_AGGREGATE_PATH).stat().st_size,
        "git_commit": D1_RESULT_COMMIT,
        "path": D1_AGGREGATE_PATH,
        "sha256": D1_AGGREGATE_SHA256,
    }
    assert inputs["d1_summary"] == {
        "byte_count": (ROOT / D1_SUMMARY_PATH).stat().st_size,
        "git_commit": D1_RESULT_COMMIT,
        "path": D1_SUMMARY_PATH,
        "sha256": D1_SUMMARY_SHA256,
    }
    assert inputs["parent_d1_aggregate"] == {
        "path": D1_AGGREGATE_PATH,
        "sha256": D1_AGGREGATE_SHA256,
        "status": "COMPLETED_EXACT_SIX_STATE_ACTION_STABILITY_AGGREGATE_V1",
        "verdict": "INVALID_RUNTIME_FAILURE",
    }
    assert context["condition_ids"] == [
        "auto_fresh_encode",
        "auto_frozen_encoded",
    ]
    assert context["historical_generation_call_count"] == 24
    assert context["historical_encode_call_count"] == 18
    assert context["historical_calls_counted_against_d1b_execution"] is False
    assert context["d1_eager_condition_consumed"] is False
    assert context["d1_eager_oom_backfilled"] is False


def test_parent_auto_projection_does_not_read_eager_row_or_old_diagnosis() -> None:
    aggregate = json.loads((ROOT / D1_AGGREGATE_PATH).read_text(encoding="utf-8"))
    parent = copy.deepcopy(aggregate["diagnostic"]["states"][0])
    expected = _parent_state_projection(parent)
    parent["conditions"][2] = {"forbidden_eager_row": object()}
    parent["diagnosis"] = object()
    assert _parent_state_projection(parent) == expected


def test_exact_noncontiguous_four_plus_two_fresh_process_schedule_is_frozen() -> None:
    diagnostic = _skeleton()["diagnostic"]
    expected_waves = [
        [STATE_IDS[index] for index in WAVE_STATE_INDICES[0]],
        [STATE_IDS[index] for index in WAVE_STATE_INDICES[1]],
    ]
    assert WAVE_STATE_INDICES == ((0, 3, 4, 5), (1, 2))
    assert diagnostic["state_waves"] == expected_waves
    assert diagnostic["wave_sizes"] == [4, 2]
    assert diagnostic["process_count"] == STATE_PROCESS_COUNT == 6
    assert diagnostic["gpu_slot_count"] == GPU_SLOT_COUNT == 4
    assert diagnostic["process_schedule"] == [
        {
            "gpu_slot": gpu_slot,
            "state_id": STATE_IDS[state_index],
            "state_index": state_index,
            "wave_index": wave_index,
        }
        for wave_index, state_indices in enumerate(WAVE_STATE_INDICES)
        for gpu_slot, state_index in enumerate(state_indices)
    ]
    assert diagnostic["aggregation_state_order"] == list(STATE_IDS)
    assert diagnostic["state_ids"] == list(STATE_IDS)
    assert diagnostic["condition_id"] == "sdpa_numerical_control_frozen_encoded"
    assert diagnostic["repeat_count"] == 2
    assert diagnostic["generation_call_ceiling"] == GENERATION_CALL_CEILING == 12
    assert diagnostic["encode_call_ceiling"] == ENCODE_CALL_CEILING == 6
    assert diagnostic["no_retry"] is True
    assert diagnostic["no_top_up"] is True
    assert all(
        row["state_id"] == STATE_IDS[row["state_index"]]
        for row in diagnostic["roster"]
    )


def test_only_new_condition_profile_and_pre_cuda_controls_are_authorized() -> None:
    data = _skeleton()
    condition = data["diagnostic"]["condition"]
    assert condition == {
        "condition_id": "sdpa_numerical_control_frozen_encoded",
        "encode_calls_per_state": 1,
        "encoding_mode": "one_exact_gpu_tensor_mapping_reused_for_both_generations",
        "fresh_os_process_per_state": True,
        "generation_calls_per_state": 2,
        "memory_efficient_sdpa_required": True,
        "numerical_controls_configured_before_cuda_initialization": True,
        "observed_attention_implementation": {
            "text": "sdpa",
            "top": "sdpa",
            "vision": "sdpa",
        },
        "profile": "sdpa_numerical_control_not_strict_cuda_determinism",
        "repeat_count": 2,
        "requested_attention_implementation": "auto_parent_default",
        "strict_cuda_determinism_claimed": False,
    }
    assert data["authorization"]["run_gpu_or_cuda"] is False
    assert data["authorization"]["load_policy_or_vision_model"] is False
    assert data["authorization"]["execution_authorized"] is False
    assert data["execution_boundary"]["execution_requires_separate_exact_envelope"] is True
    assert data["execution_boundary"]["execution_b_must_be_direct_child_of_source_a"] is True
    assert (
        data["execution_boundary"]["execution_b_only_changed_path"]
        == CANONICAL_EXECUTION_CONFIG_PATH
    )
    assert data["execution_boundary"]["wave_0_terminal_barrier_before_wave_1_attempt"] is True
    assert data["execution_boundary"]["process_per_state"] is True
    assert data["execution_boundary"]["state_process_count"] == 6
    assert data["execution_boundary"]["max_concurrent_processes"] == 4
    assert data["execution_boundary"]["max_concurrent_state_processes"] == 4
    assert data["execution_boundary"]["waves"] == data["diagnostic"]["state_waves"]
    assert data["execution_boundary"]["no_retry"] is True
    assert data["execution_boundary"]["no_top_up"] is True


def test_precise_state_matrix_and_global_verdict_precedence_reject_persistent_name() -> None:
    interpretation = _skeleton()["preregistered_interpretation"]
    assert interpretation["state_diagnosis_precedence"] == [
        dict(rule) for rule in STATE_DIAGNOSIS_PRECEDENCE
    ]
    assert interpretation["aggregate_verdict_precedence"] == list(
        AGGREGATE_VERDICT_PRECEDENCE
    )
    assert AGGREGATE_VERDICT_PRECEDENCE == (
        "INVALID_RUNTIME_FAILURE",
        "INVALID_STABLE_CONTROL_INSTABILITY",
        "NO_GO_SDPA_CONTROL_REPEAT_INSTABILITY",
        "PASS_MEMORY_SAFE_SDPA_NUMERICAL_CONTROL_REPEAT_STABILITY",
    )
    diagnoses = {
        rule["diagnosis"] for rule in STATE_DIAGNOSIS_PRECEDENCE
    }
    assert {
        "D1_AUTO_FROZEN_INSTABILITY_UNRESOLVED_BY_SDPA_CONTROL",
        "MIXED_D1_FRESH_PATH_AND_SDPA_CONTROL_INSTABILITY",
        "SDPA_CONTROL_ONLY_INSTABILITY",
        "AUTO_VS_CONTROLLED_SDPA_PROFILE_ASSOCIATION",
        "FRESH_VS_FROZEN_PATH_ASSOCIATION_WITH_SDPA_CONTROL_STABLE",
    }.issubset(diagnoses)
    assert interpretation["forbidden_ambiguous_verdict_names"] == [
        "PERSISTENT_GENERATION_INSTABILITY"
    ]
    assert "PERSISTENT_GENERATION_INSTABILITY" not in diagnoses
    assert interpretation["operational_gate"] == {
        "go_verdict": "PASS_MEMORY_SAFE_SDPA_NUMERICAL_CONTROL_REPEAT_STABILITY",
        "go_authorizes_only_new_v3_throughput_source_freeze": True,
        "labels_or_training_unlocked": False,
    }


def test_metric_firewall_zero_source_operations_and_downstream_locks_are_frozen() -> None:
    data = _skeleton()
    firewall = data["metric_only_result_contract"]
    assert firewall["metric_safe_required"] is True
    assert firewall["retry_count"] == 0
    assert {
        "action",
        "coordinate",
        "decoded_output",
        "token_ids",
        "logits",
        "utility",
    }.issubset(firewall["forbidden_serialized_fields"])
    assert all(value == 0 for value in data["negative_operations"].values())
    assert all(data["scientific_locks"].values())


def test_source_inventory_reserves_exact_v2_surface_and_is_canonical() -> None:
    source = _skeleton()["source"]
    paths = [record["path"] for record in source["inventory"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths)) == source["file_count"]
    assert source["reserved_entrypoint_paths"] == list(SOURCE_ENTRYPOINT_CANDIDATES)
    assert source["inventory_sha256"] == sha256_bytes(
        canonical_json_bytes(source["inventory"])
    )
    assert "code/causalcache/set_utility_action_stability_contract_v2.py" in paths
    assert CLI_PATH in paths
    assert "code/causalcache/set_utility_action_stability_diagnostic_v2.py" in paths


def test_validator_is_exact_once_reserved_entrypoints_exist_and_mutations_fail_closed() -> None:
    data = _skeleton()
    missing = data["source"]["missing_reserved_entrypoint_paths"]
    if missing:
        with pytest.raises(ValueError, match="missing entrypoints"):
            validate_action_stability_source_v2_config(data, repository_root=ROOT)
    else:
        validation = validate_action_stability_source_v2_config(
            data,
            repository_root=ROOT,
        )
        assert validation == {
            "config_sha256": sha256_bytes(
                canonical_pretty_json_bytes(data)
            ),
            "encode_call_ceiling": 6,
            "generation_call_ceiling": 12,
            "process_count": 6,
            "source_file_count": data["source"]["file_count"],
            "source_inventory_sha256": data["source"]["inventory_sha256"],
            "state_count": 6,
            "status": VALIDATION_STATUS,
            "wave_sizes": [4, 2],
        }

    mutations = (
        lambda value: value.__setitem__("status", "drift"),
        lambda value: value["diagnostic"].__setitem__("generation_call_ceiling", 13),
        lambda value: value["diagnostic"]["process_schedule"][0].__setitem__(
            "state_index", 1
        ),
        lambda value: value["historical_d1_auto_context"].__setitem__(
            "d1_eager_condition_consumed", True
        ),
        lambda value: value["preregistered_interpretation"].__setitem__(
            "aggregate_verdict_precedence", []
        ),
        lambda value: value["scientific_locks"].__setitem__(
            "formal_labels_locked", False
        ),
    )
    for mutate in mutations:
        changed = copy.deepcopy(data)
        mutate(changed)
        if missing:
            with pytest.raises(ValueError):
                validate_action_stability_source_v2_config(
                    changed,
                    repository_root=ROOT,
                )
        else:
            with pytest.raises(ValueError, match="live skeleton"):
                validate_action_stability_source_v2_config(
                    changed,
                    repository_root=ROOT,
                )


def test_cli_emits_source_only_skeleton_without_gpu_work() -> None:
    assert CANONICAL_CONFIG_PATH == (
        "code/configs/causalcache_set_utility_action_stability_diagnostic_v2.json"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "code")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / CLI_PATH),
            "--repository-root",
            str(ROOT),
            "--emit-skeleton",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == (
        "SOURCE_ONLY_SET_UTILITY_ACTION_STABILITY_DIAGNOSTIC_V2_FROZEN"
    )
    assert payload["diagnostic"]["generation_call_ceiling"] == 12
    assert payload["diagnostic"]["encode_call_ceiling"] == 6
    assert payload["negative_operations"]["gpu_or_cuda_call_count"] == 0
