from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from causalcache.set_utility_throughput_pilot_contract_v1 import (
    ALLOWED_MICROBATCH_ORDER,
    CANONICAL_CONFIG_PATH,
    CLI_PATH,
    EXPECTED_CANDIDATE_SCHEDULE_SHA256,
    EXPECTED_HF_REVISION,
    EXPECTED_STATE_IDS,
    MODEL_SNAPSHOT_MANIFEST_PATH,
    MODEL_SNAPSHOT_MANIFEST_SHA256,
    NATIVE_CALL_CEILING,
    VALIDATION_STATUS,
    WORKER_COUNT,
    build_train_only_throughput_pilot_v1_config_skeleton,
    canonical_json_bytes,
    load_train_only_throughput_pilot_v1_contract,
    sha256_bytes,
    validate_train_only_throughput_pilot_v1_config,
)


ROOT = Path(__file__).resolve().parents[2]


def _contract():
    return load_train_only_throughput_pilot_v1_contract(repository_root=ROOT)


def _copy_contract_tree(destination: Path) -> None:
    data = _contract().data
    paths = {CANONICAL_CONFIG_PATH}
    paths.update(record["path"] for record in data["source"]["inventory"])
    paths.update(
        {
            data["inputs"]["freeze_b_v2_manifest"]["path"],
            data["inputs"]["model_snapshot_manifest"]["path"],
            data["inputs"]["processor_formal_result_summary"]["path"],
            data["inputs"]["processor_publication"]["path"],
        }
    )
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def test_canonical_config_is_exact_live_skeleton() -> None:
    contract = _contract()
    expected = build_train_only_throughput_pilot_v1_config_skeleton(
        repository_root=ROOT
    )

    assert contract.data == expected
    validation = validate_train_only_throughput_pilot_v1_config(
        contract.data,
        repository_root=ROOT,
    )
    assert validation == {
        "candidate_schedule_sha256": EXPECTED_CANDIDATE_SCHEDULE_SHA256,
        "config_sha256": contract.config_sha256,
        "hf_revision": EXPECTED_HF_REVISION,
        "native_call_ceiling": NATIVE_CALL_CEILING,
        "source_file_count": contract.data["source"]["file_count"],
        "source_inventory_sha256": contract.data["source"]["inventory_sha256"],
        "state_count": 12,
        "status": VALIDATION_STATUS,
        "worker_count": WORKER_COUNT,
    }


def test_roster_strata_and_worker_mapping_are_exact() -> None:
    pilot = _contract().data["pilot"]

    assert [record["state_id"] for record in pilot["roster"]] == list(
        EXPECTED_STATE_IDS
    )
    assert pilot["state_ids_sha256"] == sha256_bytes(
        canonical_json_bytes(list(EXPECTED_STATE_IDS))
    )
    assert [record["candidate_capacity_stratum"] for record in pilot["strata"]] == [
        "decisions_6_9",
        "decisions_10_17",
        "decisions_18_plus",
    ]
    assert [record["state_count"] for record in pilot["strata"]] == [4, 4, 4]
    assert pilot["worker_count"] == 4
    assert pilot["aggregation_order"] == [0, 1, 2, 3]
    assert pilot["worker_mapping"] == [
        {
            "worker_index": worker,
            "state_ids": [
                EXPECTED_STATE_IDS[worker],
                EXPECTED_STATE_IDS[4 + worker],
                EXPECTED_STATE_IDS[8 + worker],
            ],
        }
        for worker in range(4)
    ]


def test_84_call_budget_and_metric_only_firewall_are_frozen() -> None:
    data = _contract().data
    pilot = data["pilot"]

    assert pilot["allowed_microbatch_order"] == list(ALLOWED_MICROBATCH_ORDER)
    assert pilot["operations_by_microbatch"] == [
        {
            "generation_calls_per_state": 2,
            "microbatch_size": 1,
            "native_calls_per_state": 4,
            "teacher_calls_per_state": 2,
            "teacher_examples_per_state": 2,
        },
        {
            "generation_calls_per_state": 2,
            "microbatch_size": 2,
            "native_calls_per_state": 3,
            "teacher_calls_per_state": 1,
            "teacher_examples_per_state": 2,
        },
    ]
    assert pilot["native_call_ceiling"] == 84
    assert pilot["expected_success_counts"]["reference_generation_call_count"] == 48
    assert pilot["expected_success_counts"]["reference_teacher_forward_call_count"] == 36
    assert pilot["expected_success_counts"]["native_call_count"] == 84
    assert data["metric_only_result_contract"]["pair_metric_only_required"] is True
    assert data["metric_only_result_contract"]["retry_count"] == 0
    assert {"action", "tokens", "logits", "kl", "utility"}.issubset(
        data["metric_only_result_contract"]["forbidden_serialized_fields"]
    )
    assert all(value == 0 for value in data["negative_operations"].values())
    assert data["authorization"]["run_gpu_or_cuda"] is False
    assert data["authorization"]["read_remote_processor_artifact"] is False
    assert data["authorization"]["write_pilot_result"] is False


def test_deployment_decision_rule_is_preregistered() -> None:
    rule = _contract().data["deployment_decision_rule"]

    assert rule["device_reserved_memory_fraction_maximum"] == 0.80
    assert rule["memory_headroom_fraction_minimum"] == 0.20
    assert rule["memory_requirement_scope"] == (
        "every_state_variant_full_call_peak_must_satisfy_the_limit"
    )
    assert rule["mb2_teacher_wall_ratio_maximum_for_selection"] == 0.95
    assert rule["mb1_any_failure_or_memory_violation_outcome"] == "NO_GO"
    assert rule["mb2_failure_fallback_to_mb1"] == {
        "all_failures_must_be_mb2_teacher_stage": True,
        "cross_variant_action_equality_must_complete_for_every_state": True,
        "otherwise_outcome": "NO_GO",
    }
    assert rule["no_retry"] is True
    assert rule["no_top_up"] is True
    assert rule["cross_host_or_heterogeneous_metric_merge_allowed"] is False
    assert rule["execution_requires_separate_exact_envelope"] is True


def test_publication_schedule_inventory_and_model_are_bound() -> None:
    inputs = _contract().data["inputs"]
    publication = inputs["processor_publication"]

    assert publication["publication_status"] == (
        "FINALIZED_PROCESSOR_V2_IMMUTABLE_HF_PUBLICATION"
    )
    assert publication["hf_revision"] == EXPECTED_HF_REVISION
    assert publication["hf_tag"] == (
        "phase1-b2-processor-freeze-v2-image-contract-repair"
    )
    assert publication["hf_prefix"] == (
        "artifacts/processor-freeze-v2-image-contract-repair"
    )
    assert publication["formal_file_count"] == 23
    assert publication["formal_total_byte_count"] == 18_730_620_511
    assert publication["candidate_schedule"] == {
        "byte_count": 18_718_642,
        "candidate_inventory_sha256": (
            "c0b0fa26c6d64705d5d533200d96c624eb58b75cd7f9b4b9564507d7442b04ad"
        ),
        "path": "processor-candidate-freeze-schedule.json",
        "sha256": EXPECTED_CANDIDATE_SCHEDULE_SHA256,
    }
    assert len(publication["formal_file_inventory"]) == 23
    assert inputs["model_snapshot_manifest"]["path"] == MODEL_SNAPSHOT_MANIFEST_PATH
    assert inputs["model_snapshot_manifest"]["sha256"] == (
        MODEL_SNAPSHOT_MANIFEST_SHA256
    )


def test_transitive_source_inventory_is_exact_and_self_bound() -> None:
    source = _contract().data["source"]
    paths = [record["path"] for record in source["inventory"]]

    assert paths == sorted(paths)
    assert len(paths) == len(set(paths)) == source["file_count"] == 44
    assert "code/causalcache/set_utility_throughput_pilot.py" in paths
    assert "code/causalcache/set_utility_throughput_pilot_pair_v1.py" in paths
    assert "code/causalcache/set_utility_throughput_pilot_envelope_v1.py" in paths
    assert "code/causalcache/set_utility_throughput_pilot_execution_v1.py" in paths
    assert "code/causalcache/set_utility_gui_owl_v2_1_throughput_adapter.py" in paths
    assert "code/causalcache/policy/gui_owl_v2_1_throughput_runtime.py" in paths
    assert "code/scripts/run_set_utility_throughput_pilot_worker_v1.py" in paths
    assert "code/scripts/aggregate_set_utility_throughput_pilot_v1.py" in paths
    assert "code/scripts/materialize_set_utility_throughput_pilot_envelope_v1.py" in paths
    assert "code/scripts/validate_set_utility_throughput_pilot_envelope_v1.py" in paths
    assert "code/causalcache/set_utility_throughput_pilot_contract_v1.py" in paths
    assert CLI_PATH in paths
    assert source["inventory_sha256"] == sha256_bytes(
        canonical_json_bytes(source["inventory"])
    )


def test_config_mutations_fail_closed() -> None:
    original = _contract().data
    mutations = (
        lambda value: value.__setitem__("status", "drift"),
        lambda value: value["pilot"].__setitem__("native_call_ceiling", 83),
        lambda value: value["pilot"]["roster"][0].__setitem__("state_id", "drift"),
        lambda value: value["pilot"].__setitem__("allowed_microbatch_order", [2, 1]),
        lambda value: value["pilot"]["worker_mapping"][0].__setitem__(
            "worker_index", 3
        ),
        lambda value: value["inputs"]["processor_publication"].__setitem__(
            "hf_revision", "0" * 40
        ),
        lambda value: value["inputs"]["processor_publication"][
            "candidate_schedule"
        ].__setitem__("sha256", "0" * 64),
        lambda value: value["inputs"]["model_snapshot_manifest"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda value: value["source"].__setitem__("inventory_sha256", "0" * 64),
        lambda value: value["deployment_decision_rule"].__setitem__(
            "mb2_teacher_wall_ratio_maximum_for_selection", 1.0
        ),
        lambda value: value["authorization"].__setitem__("run_gpu_or_cuda", True),
    )
    for mutate in mutations:
        changed = copy.deepcopy(original)
        mutate(changed)
        with pytest.raises(ValueError, match="live skeleton"):
            validate_train_only_throughput_pilot_v1_config(
                changed,
                repository_root=ROOT,
            )


def test_transitive_source_byte_drift_fails_closed(tmp_path: Path) -> None:
    _copy_contract_tree(tmp_path)
    path = tmp_path / "code/causalcache/set_utility_throughput_pilot_pair_v1.py"
    path.write_bytes(path.read_bytes() + b"\n# drift\n")

    with pytest.raises(ValueError, match="live skeleton"):
        load_train_only_throughput_pilot_v1_contract(repository_root=tmp_path)


def test_bound_model_manifest_drift_fails_before_config_comparison(
    tmp_path: Path,
) -> None:
    _copy_contract_tree(tmp_path)
    path = tmp_path / MODEL_SNAPSHOT_MANIFEST_PATH
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="snapshot manifest SHA256 drifted"):
        load_train_only_throughput_pilot_v1_contract(repository_root=tmp_path)


def test_bound_publication_sibling_drift_fails_before_config_comparison(
    tmp_path: Path,
) -> None:
    _copy_contract_tree(tmp_path)
    relative = _contract().data["inputs"]["processor_publication"]["path"]
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="publication sibling summary SHA256 drifted"):
        load_train_only_throughput_pilot_v1_contract(repository_root=tmp_path)


def test_loader_rejects_noncanonical_json_and_noncanonical_path(
    tmp_path: Path,
) -> None:
    _copy_contract_tree(tmp_path)
    path = tmp_path / CANONICAL_CONFIG_PATH
    path.write_text(json.dumps(_contract().data), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical pretty JSON"):
        load_train_only_throughput_pilot_v1_contract(repository_root=tmp_path)

    with pytest.raises(ValueError, match="must be canonical"):
        load_train_only_throughput_pilot_v1_contract(
            repository_root=ROOT,
            config_path=MODEL_SNAPSHOT_MANIFEST_PATH,
        )


@pytest.mark.parametrize(
    "payload,pattern",
    [
        (b'{"schema_version":"1","schema_version":"2"}\n', "duplicate key"),
        (b'{"schema_version":NaN}\n', "non-finite"),
    ],
)
def test_loader_rejects_non_strict_json(
    tmp_path: Path,
    payload: bytes,
    pattern: str,
) -> None:
    _copy_contract_tree(tmp_path)
    (tmp_path / CANONICAL_CONFIG_PATH).write_bytes(payload)

    with pytest.raises(ValueError, match=pattern):
        load_train_only_throughput_pilot_v1_contract(repository_root=tmp_path)


def test_cli_prints_metric_only_validation_summary() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "code")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / CLI_PATH),
            "--repository-root",
            str(ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == VALIDATION_STATUS
    assert payload["state_count"] == 12
    assert payload["native_call_ceiling"] == 84
    assert set(payload) == {
        "candidate_schedule_sha256",
        "config_sha256",
        "hf_revision",
        "native_call_ceiling",
        "source_file_count",
        "source_inventory_sha256",
        "state_count",
        "status",
        "worker_count",
    }
