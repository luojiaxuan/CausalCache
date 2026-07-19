from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from causalcache.set_utility_action_stability_contract_v1 import (
    CANONICAL_CONFIG_PATH,
    CLI_PATH,
    ENCODE_CALL_CEILING,
    GENERATION_CALL_CEILING,
    PARENT_AGGREGATE_PATH,
    PARENT_AGGREGATE_SHA256,
    PARENT_RESULT_COMMIT,
    PARENT_SUMMARY_PATH,
    PARENT_SUMMARY_SHA256,
    STATE_IDS,
    VALIDATION_STATUS,
    WORKER_COUNT,
    build_action_stability_source_v1_config_skeleton,
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    load_action_stability_source_v1_contract,
    sha256_bytes,
    validate_action_stability_source_v1_config,
)


ROOT = Path(__file__).resolve().parents[2]


def _skeleton() -> dict[str, object]:
    return build_action_stability_source_v1_config_skeleton(repository_root=ROOT)


def test_parent_result_and_source_inventory_are_exactly_bound() -> None:
    data = _skeleton()
    inputs = data["immutable_inputs"]

    assert inputs["parent_result_commit"] == PARENT_RESULT_COMMIT
    assert inputs["parent_summary"] == {
        "byte_count": (ROOT / PARENT_SUMMARY_PATH).stat().st_size,
        "path": PARENT_SUMMARY_PATH,
        "sha256": PARENT_SUMMARY_SHA256,
    }
    assert inputs["parent_aggregate"] == {
        "byte_count": (ROOT / PARENT_AGGREGATE_PATH).stat().st_size,
        "path": PARENT_AGGREGATE_PATH,
        "sha256": PARENT_AGGREGATE_SHA256,
    }
    source = data["source"]
    paths = [record["path"] for record in source["inventory"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths)) == source["file_count"]
    assert "code/causalcache/set_utility_action_stability_contract_v1.py" in paths
    assert CLI_PATH in paths
    assert "code/causalcache/set_utility_action_stability_diagnostic_v1.py" in paths
    assert source["inventory_sha256"] == sha256_bytes(
        canonical_json_bytes(source["inventory"])
    )


def test_six_state_roster_workers_and_operation_ceilings_are_frozen() -> None:
    diagnostic = _skeleton()["diagnostic"]

    assert diagnostic["state_count"] == 6
    assert [record["state_id"] for record in diagnostic["roster"]] == list(STATE_IDS)
    assert diagnostic["state_ids_sha256"] == sha256_bytes(
        canonical_json_bytes(list(STATE_IDS))
    )
    assert diagnostic["worker_count"] == WORKER_COUNT == 4
    assert diagnostic["worker_mapping"] == [
        {"state_ids": [STATE_IDS[0]], "worker_index": 0},
        {"state_ids": [STATE_IDS[1], STATE_IDS[5]], "worker_index": 1},
        {"state_ids": [STATE_IDS[2], STATE_IDS[3]], "worker_index": 2},
        {"state_ids": [STATE_IDS[4]], "worker_index": 3},
    ]
    assert diagnostic["generation_call_ceiling"] == GENERATION_CALL_CEILING == 36
    assert diagnostic["encode_call_ceiling"] == ENCODE_CALL_CEILING == 24
    assert diagnostic["no_retry"] is True
    assert diagnostic["no_top_up"] is True


def test_three_conditions_metric_firewall_and_scientific_locks_are_frozen() -> None:
    data = _skeleton()
    conditions = data["diagnostic"]["conditions"]

    assert [condition["condition_id"] for condition in conditions] == [
        "auto_fresh_encode",
        "auto_frozen_encoded",
        "eager_frozen_encoded_control",
    ]
    assert [condition["repeat_count"] for condition in conditions] == [2, 2, 2]
    assert [condition["process_stage"] for condition in conditions] == [
        "auto",
        "auto",
        "eager",
    ]
    assert conditions[0]["observed_attention_implementation"] == {
        "text": "sdpa",
        "top": "sdpa",
        "vision": "sdpa",
    }
    assert conditions[2]["observed_attention_implementation"] == {
        "text": "eager",
        "top": "eager",
        "vision": "eager",
    }
    assert [condition["encode_calls_per_state"] for condition in conditions] == [
        2,
        1,
        1,
    ]
    assert all(condition["generation_calls_per_state"] == 2 for condition in conditions)
    assert data["authorization"]["run_gpu_or_cuda"] is False
    assert data["authorization"]["load_policy_or_vision_model"] is False
    assert data["execution_boundary"]["execution_requires_separate_exact_envelope"] is True
    assert data["execution_boundary"]["parent_v2_retry_allowed"] is False
    assert data["metric_only_result_contract"]["metric_safe_required"] is True
    assert {"action", "token_ids", "logits", "utility"}.issubset(
        data["metric_only_result_contract"]["forbidden_serialized_fields"]
    )
    assert all(value == 0 for value in data["negative_operations"].values())
    assert data["scientific_locks"]["formal_labels_locked"] is True
    assert data["scientific_locks"]["predictor_training_locked"] is True
    assert data["scientific_locks"]["twelve_state_v3_throughput_required_before_labels"] is True


def test_validator_is_exact_and_mutations_fail_closed() -> None:
    data = _skeleton()
    validation = validate_action_stability_source_v1_config(
        data,
        repository_root=ROOT,
    )
    assert validation["status"] == VALIDATION_STATUS
    assert validation["state_count"] == 6
    assert validation["generation_call_ceiling"] == 36
    assert validation["encode_call_ceiling"] == 24

    mutations = (
        lambda value: value.__setitem__("status", "drift"),
        lambda value: value["diagnostic"].__setitem__("generation_call_ceiling", 35),
        lambda value: value["diagnostic"]["roster"][0].__setitem__("state_id", "drift"),
        lambda value: value["authorization"].__setitem__("run_gpu_or_cuda", True),
        lambda value: value["scientific_locks"].__setitem__("formal_labels_locked", False),
        lambda value: value["source"].__setitem__("inventory_sha256", "0" * 64),
    )
    for mutate in mutations:
        changed = copy.deepcopy(data)
        mutate(changed)
        with pytest.raises(ValueError, match="live skeleton"):
            validate_action_stability_source_v1_config(
                changed,
                repository_root=ROOT,
            )


def test_cli_can_emit_the_unfrozen_live_skeleton_without_gpu_work() -> None:
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
    assert payload["status"] == "SOURCE_ONLY_SET_UTILITY_ACTION_STABILITY_DIAGNOSTIC_V1_FROZEN"
    assert payload["diagnostic"]["generation_call_ceiling"] == 36
    assert payload["negative_operations"]["gpu_or_cuda_call_count"] == 0


def test_canonical_config_is_exactly_the_live_source_skeleton() -> None:
    assert CANONICAL_CONFIG_PATH == (
        "code/configs/causalcache_set_utility_action_stability_diagnostic_v1.json"
    )
    path = ROOT / CANONICAL_CONFIG_PATH
    assert path.read_bytes() == canonical_pretty_json_bytes(_skeleton())
    contract = load_action_stability_source_v1_contract(repository_root=ROOT)
    assert contract.config_sha256 == sha256_bytes(path.read_bytes())
