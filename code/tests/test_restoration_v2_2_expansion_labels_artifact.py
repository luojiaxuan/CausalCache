from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import itertools
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import causalcache.restoration_v2_2_expansion_labels_artifact as artifact_module
from causalcache.restoration_v2_2_expansion_labels_artifact import (
    AGGREGATE_FILENAME,
    ALGEBRA_RESIDUAL_TOLERANCE,
    CANONICAL_ARCHIVE_PATH,
    CANONICAL_CONFIG_PATH,
    CANONICAL_DERIVED_ARTIFACT,
    CANONICAL_HF_PATH,
    CANONICAL_HF_REPO,
    CANONICAL_HF_TAG,
    CANONICAL_LEDGER_PATH,
    CANONICAL_OUTPUT_DIR,
    CANONICAL_PARENT_SUBSTRATE,
    CANONICAL_RUNNER_FREEZE_PATH,
    CANONICAL_RUNTIME,
    EXECUTION_EVIDENCE_FILENAME,
    EXPECTED_TOTALS,
    FROZEN_SUBSTRATE_STATE_PROJECTION_SHA256,
    GLOBAL_LEDGER_ARCHIVE_NAME,
    LOG_PATHS,
    MONITOR_SUMMARY_FILENAME,
    PASS_OUTCOME,
    PROHIBITED_OPERATION_COUNTS,
    PROTOCOL_ID,
    RUN_MANIFEST_FILENAME,
    RUNNER_FREEZE_PROTOCOL_ID,
    RUNNER_FREEZE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SOURCE_PATHS_REQUIRED_IN_RUNNER_FREEZE,
    STATE_STATUS,
    WORKER_STATUS,
    aggregate_from_reduction,
    atomic_exclusive_publish_bytes,
    canonical_json_bytes,
    deterministic_ustar_bytes,
    expected_file_names,
    expected_worker_specs,
    pretty_json_bytes,
    read_expansion_label_archive,
    reduce_raw_distance_states,
    runner_interface_contract,
    sha256_bytes,
    sha256_file,
    validate_execution_run_contract,
    validate_external_input_replay,
    validate_expansion_label_evidence_files,
    validate_raw_state_record,
    validate_runner_freeze_data,
    validate_state_projections,
)
from scripts import manage_restoration_v2_2_expansion_labels_artifact as manager


SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_CONFIG = "c" * 64
HF_REVISION = "f" * 40
GPU_UUIDS = (
    "GPU-11111111-1111-1111-1111-111111111111",
    "GPU-22222222-2222-2222-2222-222222222222",
)
START = "2026-07-17T00:00:00Z"
SAMPLE = "2026-07-17T00:00:01Z"
CLAIM = "2026-07-17T00:00:02Z"
STOP = "2026-07-17T00:10:00Z"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _source_ids() -> list[str]:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causalcache_restoration_v2_2_label_expansion_v1.json"
    )
    selection = json.loads(path.read_text())["selection"]
    return [
        *selection["gate_train_expansion"]["source_ids"],
        *selection["gate_development_expansion"]["source_ids"],
    ]


def _coalitions(event_ids: list[int]) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(len(event_ids) + 1)
        for subset in itertools.combinations(event_ids, size)
    ]


def _states() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index in range(192):
        source_id = _source_ids()[index // 3]
        step = 4 + index % 3
        event_ids = list(range(1, step - 1))
        result.append(
            {
                "state_index": index,
                "role": (
                    "gate_train_expansion"
                    if index < 144
                    else "gate_development_expansion"
                ),
                "source_id": source_id,
                "decision_step_id": step,
                "state_id": f"{source_id}:decision_step:{step:03d}",
                "history_event_step_ids": list(range(1, step)),
                "candidate_event_step_ids": event_ids,
                "current_equivalent_event_step_id": step - 1,
                "parent_member_name": (
                    f"workers/{'even' if index % 2 == 0 else 'odd'}"
                    f"/states/{index:03d}.json"
                ),
                "parent_member_sha256": _sha(f"parent-member-{index}"),
                "canonical_action_sha256": _sha(f"action-{index}"),
                "teacher_target_sha256": _sha(f"target-{index}"),
                "request_manifest_sha256": _sha(f"request-manifest-{index}"),
                "slice_witness_sha256": _sha(f"slice-witness-{index}"),
                "immutable_artifact_tree_sha256": CANONICAL_DERIVED_ARTIFACT[
                    "artifact_tree_sha256"
                ],
                "coalition_inputs": [
                    {
                        "coalition": list(coalition),
                        "input_sha256": _sha(f"input-{index}-{coalition}"),
                    }
                    for coalition in _coalitions(event_ids)
                ],
            }
        )
    return result


def _state_record(
    state: dict[str, object], run_hash: str
) -> dict[str, object]:
    index = int(state["state_index"])
    event_ids = list(state["candidate_event_step_ids"])
    coalition_inputs = {
        tuple(item["coalition"]): item["input_sha256"]
        for item in state["coalition_inputs"]
    }
    rows = []
    baseline = 1.0 + index / 10_000
    for coalition in _coalitions(event_ids):
        is_reference = coalition == tuple(event_ids)
        if is_reference:
            distance = 0.0
        elif coalition == (1,) and index % 2 == 0:
            distance = baseline + 0.2
        else:
            distance = max(
                0.0,
                baseline - 0.15 * len(coalition) - 0.01 * sum(coalition),
            )
        rows.append(
            {
                "coalition": list(coalition),
                "distance": distance,
                "candidate_input_sha256": coalition_inputs[coalition],
                "teacher_forward_count": 0 if is_reference else 1,
                "kl_measurement_count": 0 if is_reference else 1,
                "scalar_host_transfer_count": 0 if is_reference else 1,
                "is_full_history_reference": is_reference,
                "full_logit_tensor_host_transfer_count": 0,
            }
        )
    event_count = len(event_ids)
    power = 1 << event_count
    spec = expected_worker_specs()[index % 2]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATE_STATUS,
        "run_contract_sha256": run_hash,
        "worker": spec.envelope_identity(),
        "state": state,
        "reference_teacher": {
            "canonical_action_sha256": state["canonical_action_sha256"],
            "teacher_target_sha256": state["teacher_target_sha256"],
            "reference_input_sha256": coalition_inputs[tuple(event_ids)],
            "reference_repeat_kl": 0.0,
            "repeat_scalar_host_transfer_count": 1,
            "action_token_count": 7,
            "vocabulary_size": 151_936,
            "action_log_probs_shape": [7, 151_936],
            "action_log_probs_dtype": "torch.float32",
            "action_log_probs_device_type": "cuda",
            "full_logit_tensor_host_transfer_count": 0,
        },
        "distance_rows": rows,
        "operation_counts": {
            "reference_teacher_forward_count": 1,
            "reference_repeat_teacher_forward_count": 1,
            "non_reference_coalition_teacher_forward_count": power - 1,
            "teacher_forward_count": power + 1,
            "kl_measurement_count": power,
            "scalar_host_transfer_count": power,
            "raw_distance_row_count": power,
            "full_logit_tensor_host_transfer_count": 0,
            "retry_count": 0,
            "top_up_count": 0,
        },
    }


def _run_contract(states: list[dict[str, object]]) -> dict[str, object]:
    options = runner_interface_contract()["required_cli_options"]
    argv = ["python", "-m", "scripts.run_restoration_v2_2_expansion_labels"]
    for option in options:
        argv.extend([option, f"value-for-{option[2:]}"])
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "AUTHORIZED_EXPANSION_EXACT_LABEL_EXECUTION",
        "runner_source_git_commit": SHA_A,
        "execution_git_commit": SHA_B,
        "config": {
            "path": CANONICAL_CONFIG_PATH,
            "sha256": SHA_CONFIG,
            "size_bytes": 123,
        },
        "runner_freeze": {
            "path": CANONICAL_RUNNER_FREEZE_PATH,
            "sha256": _sha("runner-freeze"),
            "size_bytes": 456,
            "runner_source_git_commit": SHA_A,
        },
        "parent_substrate": CANONICAL_PARENT_SUBSTRATE,
        "derived_artifact": CANONICAL_DERIVED_ARTIFACT,
        "states": states,
        "worker_topology": [spec.to_dict() for spec in expected_worker_specs()],
        "planned_counts": EXPECTED_TOTALS,
        "prohibited_operation_counts": PROHIBITED_OPERATION_COUNTS,
        "attempt_identity": {
            "attempt_id": "restoration-v2-2-expansion-exact-labels-v1",
            "output_dir": str(CANONICAL_OUTPUT_DIR),
            "global_ledger_path": str(CANONICAL_LEDGER_PATH),
            "archive_path": str(CANONICAL_ARCHIVE_PATH),
        },
        "execution_argv": argv,
    }


def _worker_totals(records: list[dict[str, object]]) -> dict[str, int]:
    keys = list(records[0]["operation_counts"])
    return {
        key: sum(int(record["operation_counts"][key]) for record in records)
        for key in keys
    }


def _bindings_metadata(spec) -> dict[str, object]:
    return {
        **CANONICAL_RUNTIME,
        "device": spec.device,
        "logical_device_index": spec.index_parity,
        "nvidia_smi_index": 2 + spec.index_parity,
        "gpu_uuid": GPU_UUIDS[spec.index_parity],
        "gpu_pci_bus_id": f"00000000:{32 + spec.index_parity:02x}:00.0",
        "model_identity_sha256": _sha("shared-model-identity"),
    }


def _runtime_record(spec, run_hash: str) -> dict[str, object]:
    bindings = _bindings_metadata(spec)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VALIDATED_EXPANSION_EXACT_LABEL_WORKER_RUNTIME",
        "run_contract_sha256": run_hash,
        "worker": spec.envelope_identity(),
        "runtime_metadata": {
            "container_id": "d" * 64,
            "host_alias": "hyper00",
            "hostname": "node-radixark-16-0000",
            "visible_gpu_count": 2,
            "visible_gpu_uuids": list(GPU_UUIDS),
            "worker_gpu_uuid": GPU_UUIDS[spec.index_parity],
            "worker_cuda_device": spec.device,
            "bindings_metadata": bindings,
            "bindings_metadata_sha256": sha256_bytes(
                canonical_json_bytes(bindings)
            ),
        },
    }


def _monitor_sample(
    sample_index: int = 0,
    elapsed_seconds: int = 1,
) -> dict[str, object]:
    sampled_at = datetime(2026, 7, 17, tzinfo=timezone.utc) + timedelta(
        seconds=elapsed_seconds
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "GPU_UTILIZATION_SAMPLE",
        "sample_index": sample_index,
        "sampled_at_utc": sampled_at.isoformat().replace("+00:00", "Z"),
        "visible_gpu_count": 2,
        "gpus": [
            {
                "nvidia_smi_index": 2,
                "gpu_uuid": GPU_UUIDS[0],
                "utilization_percent": 95,
            },
            {
                "nvidia_smi_index": 3,
                "gpu_uuid": GPU_UUIDS[1],
                "utilization_percent": 96,
            },
        ],
    }


def _evidence_files() -> dict[str, bytes]:
    states = _states()
    contract = _run_contract(states)
    run_hash = sha256_bytes(canonical_json_bytes(contract))
    records = [_state_record(state, run_hash) for state in states]
    files: dict[str, bytes] = {}
    files[RUN_MANIFEST_FILENAME] = pretty_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "IMMUTABLE_EXPANSION_EXACT_LABEL_RUN_MANIFEST",
            "run_contract": contract,
            "run_contract_sha256": run_hash,
        }
    )
    files[GLOBAL_LEDGER_ARCHIVE_NAME] = pretty_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETED_EXPANSION_EXACT_LABEL_ATTEMPT",
            "attempt_id": "restoration-v2-2-expansion-exact-labels-v1",
            "run_contract_sha256": run_hash,
            "output_dir": str(CANONICAL_OUTPUT_DIR),
            "worker_sibling_ledgers": {
                spec.worker_id: str(
                    CANONICAL_OUTPUT_DIR.parent
                    / (
                        ".restoration-v2-2-expansion-exact-labels-v1."
                        f"{spec.worker_id}.attempt.json"
                    )
                )
                for spec in expected_worker_specs()
            },
            "attempted_state_indices": list(range(192)),
            "completed_state_indices": list(range(192)),
            "retry_count": 0,
            "top_up_count": 0,
            "started_at_utc": START,
            "ended_at_utc": STOP,
        }
    )
    for spec in expected_worker_specs():
        base = f"workers/{spec.worker_id}"
        worker_records = [records[index] for index in spec.state_indices]
        ledger = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": WORKER_STATUS,
            "run_contract_sha256": run_hash,
            "worker": spec.to_dict(),
            "attempted_state_indices": list(spec.state_indices),
            "completed_state_indices": list(spec.state_indices),
            "retry_count": 0,
            "top_up_count": 0,
            "started_at_utc": START,
            "ended_at_utc": STOP,
        }
        files[f"{base}/worker_attempt_ledger.json"] = pretty_json_bytes(ledger)
        files[f"worker_sibling_ledgers/{spec.worker_id}.json"] = pretty_json_bytes(
            ledger
        )
        files[f"{base}/runtime_identity.json"] = pretty_json_bytes(
            _runtime_record(spec, run_hash)
        )
        for index in spec.state_indices:
            files[f"{base}/attempts/{index:03d}.json"] = pretty_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "status": "CLAIMED_EXPANSION_EXACT_LABEL_STATE_NO_RETRY",
                    "run_contract_sha256": run_hash,
                    "worker": spec.envelope_identity(),
                    "state_index": index,
                    "retry_count": 0,
                    "top_up_count": 0,
                    "claimed_at_utc": CLAIM,
                }
            )
            files[f"{base}/states/{index:03d}.json"] = pretty_json_bytes(
                records[index]
            )
        files[f"{base}/terminal.json"] = pretty_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": "TERMINAL_EXPANSION_EXACT_LABEL_WORKER",
                "outcome": WORKER_STATUS,
                "run_contract_sha256": run_hash,
                "worker": spec.to_dict(),
                "attempted_state_indices": list(spec.state_indices),
                "completed_state_indices": list(spec.state_indices),
                "operation_counts": _worker_totals(worker_records),
                "failure": None,
                "ended_at_utc": STOP,
            }
        )
    files[LOG_PATHS["preflight"]] = b"preflight passed\n"
    files[LOG_PATHS["execution"]] = b"execution completed\n"
    monitor_samples = [
        _monitor_sample(index, elapsed)
        for index, elapsed in enumerate(range(1, 600, 2))
    ]
    files[LOG_PATHS["utilization_monitor"]] = b"".join(
        canonical_json_bytes(sample) + b"\n" for sample in monitor_samples
    )
    ready = {
        "schema_version": SCHEMA_VERSION,
        "status": "EXPANSION_EXACT_LABEL_MONITOR_READY",
        "run_contract_sha256": run_hash,
        "monitor_pid": 123,
        "visible_gpu_count": 2,
        "visible_gpu_uuids": list(GPU_UUIDS),
        "started_at_utc": START,
    }
    stop = {
        "schema_version": SCHEMA_VERSION,
        "status": "EXPANSION_EXACT_LABEL_MONITOR_STOP_REQUEST",
        "run_contract_sha256": run_hash,
        "requested_at_utc": STOP,
    }
    files[LOG_PATHS["utilization_monitor_ready"]] = pretty_json_bytes(ready)
    files[LOG_PATHS["utilization_monitor_stop_request"]] = pretty_json_bytes(stop)
    files[MONITOR_SUMMARY_FILENAME] = pretty_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "GPU_UTILIZATION_MONITOR_SUMMARY",
            "run_contract_sha256": run_hash,
            "container_name_prefix": "sglang-omni-jaxan",
            "minimum_gpu_utilization_percent": 90,
            "started_before_first_teacher_forward": True,
            "started_at_utc": START,
            "stopped_at_utc": STOP,
            "sample_count": len(monitor_samples),
            "low_utilization_incident_count": 0,
            "visible_gpu_count": 2,
            "visible_gpu_uuids": list(GPU_UUIDS),
            "stop_request_sha256": sha256_bytes(
                files[LOG_PATHS["utilization_monitor_stop_request"]]
            ),
        }
    )
    files[EXECUTION_EVIDENCE_FILENAME] = pretty_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "VALIDATED_EXPANSION_EXACT_LABEL_EXECUTION_EVIDENCE",
            "run_contract_sha256": run_hash,
            "preflight_log_sha256": sha256_bytes(files[LOG_PATHS["preflight"]]),
            "execution_log_sha256": sha256_bytes(files[LOG_PATHS["execution"]]),
            "utilization_monitor_log_sha256": sha256_bytes(
                files[LOG_PATHS["utilization_monitor"]]
            ),
            "monitor_ready_sha256": sha256_bytes(
                files[LOG_PATHS["utilization_monitor_ready"]]
            ),
            "monitor_stop_request_sha256": sha256_bytes(
                files[LOG_PATHS["utilization_monitor_stop_request"]]
            ),
            "monitor_summary_sha256": sha256_bytes(
                files[MONITOR_SUMMARY_FILENAME]
            ),
        }
    )
    reduction = reduce_raw_distance_states(
        records,
        expected_states=states,
        run_contract_sha256=run_hash,
    )
    files[AGGREGATE_FILENAME] = pretty_json_bytes(
        aggregate_from_reduction(
            reduction,
            run_contract_sha256=run_hash,
            execution_evidence_sha256=sha256_bytes(
                files[EXECUTION_EVIDENCE_FILENAME]
            ),
            started_at_utc=START,
            ended_at_utc=STOP,
        )
    )
    assert set(files) == expected_file_names()
    return files


def _mutate(files: dict[str, bytes], path: str, fn) -> dict[str, bytes]:
    changed = dict(files)
    value = json.loads(changed[path])
    fn(value)
    changed[path] = pretty_json_bytes(value)
    return changed


class ExpansionLabelArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.files = _evidence_files()

    def test_full_raw_evidence_reduces_exact_frozen_totals(self) -> None:
        evidence = validate_expansion_label_evidence_files(
            self.files,
            expected_source_git_commit=SHA_A,
            expected_config_sha256=SHA_CONFIG,
            require_canonical_attempt_identity=False,
        )
        self.assertEqual(evidence.outcome, PASS_OUTCOME)
        self.assertEqual(evidence.reduction["counts"], EXPECTED_TOTALS)
        self.assertEqual(len(evidence.reduction["states"]), 192)
        gate = evidence.reduction["summary"]["algebra_residual_gate"]
        self.assertTrue(gate["passed"])
        self.assertLessEqual(
            gate["maximum_shapley_efficiency_abs_residual"],
            ALGEBRA_RESIDUAL_TOLERANCE,
        )
        self.assertGreater(
            evidence.reduction["summary"]["marginal_sign_counts"]["negative"], 0
        )

    def test_dynamic_projection_matches_frozen_substrate(self) -> None:
        states = validate_state_projections(_states())
        base = [
            {
                key: state[key]
                for key in (
                    "state_index",
                    "role",
                    "source_id",
                    "state_id",
                    "decision_step_id",
                    "history_event_step_ids",
                    "candidate_event_step_ids",
                    "current_equivalent_event_step_id",
                )
            }
            for state in states
        ]
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(base)),
            FROZEN_SUBSTRATE_STATE_PROJECTION_SHA256,
        )

    def test_external_replay_requires_byte_identical_dynamic_inputs(self) -> None:
        states = _states()
        metadata = {
            "external_input_replay_verified": True,
            "policy_or_model_forward_executed": False,
            "state_count": 192,
            "coalition_input_witness_count": 1_792,
            "external_state_projections_sha256": sha256_bytes(
                canonical_json_bytes(states)
            ),
        }
        result = validate_external_input_replay(
            _run_contract(states), states, metadata
        )
        self.assertTrue(result["byte_identical_to_embedded_run_contract"])
        drifted = copy.deepcopy(states)
        drifted[0]["coalition_inputs"][0]["input_sha256"] = _sha("drifted")
        drifted_metadata = {
            **metadata,
            "external_state_projections_sha256": sha256_bytes(
                canonical_json_bytes(drifted)
            ),
        }
        with self.assertRaisesRegex(ValueError, "byte-for-byte"):
            validate_external_input_replay(
                _run_contract(states), drifted, drifted_metadata
            )

    def test_missing_or_duplicate_coalition_is_rejected(self) -> None:
        path = "workers/even/states/000.json"
        changed = _mutate(
            self.files,
            path,
            lambda value: value["distance_rows"].__setitem__(
                1, copy.deepcopy(value["distance_rows"][0])
            ),
        )
        with self.assertRaisesRegex(ValueError, "coalition"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_raw_input_sha_must_equal_coalition_witness(self) -> None:
        states = _states()
        contract = _run_contract(states)
        run_hash = sha256_bytes(canonical_json_bytes(contract))
        record = _state_record(states[0], run_hash)
        record["distance_rows"][0]["candidate_input_sha256"] = _sha("tampered")
        with self.assertRaisesRegex(ValueError, "run-contract witness"):
            validate_raw_state_record(
                record,
                expected_state=states[0],
                expected_worker=expected_worker_specs()[0],
                run_contract_sha256=run_hash,
            )

    def test_reference_action_must_equal_parent_witness(self) -> None:
        states = _states()
        contract = _run_contract(states)
        run_hash = sha256_bytes(canonical_json_bytes(contract))
        record = _state_record(states[0], run_hash)
        record["reference_teacher"]["canonical_action_sha256"] = _sha("tampered")
        with self.assertRaisesRegex(ValueError, "parent witness"):
            validate_raw_state_record(
                record,
                expected_state=states[0],
                expected_worker=expected_worker_specs()[0],
                run_contract_sha256=run_hash,
            )

    def test_derived_artifact_identity_is_exact(self) -> None:
        contract = _run_contract(_states())
        contract["derived_artifact"] = {
            **CANONICAL_DERIVED_ARTIFACT,
            "artifact_tree_sha256": _sha("wrong tree"),
        }
        with self.assertRaisesRegex(ValueError, "derived artifact"):
            validate_execution_run_contract(contract)

    def test_negative_marginal_is_preserved_not_clamped(self) -> None:
        states = _states()
        contract = _run_contract(states)
        run_hash = sha256_bytes(canonical_json_bytes(contract))
        record = _state_record(states[0], run_hash)
        _normalized, derived = validate_raw_state_record(
            record,
            expected_state=states[0],
            expected_worker=expected_worker_specs()[0],
            run_contract_sha256=run_hash,
        )
        edge = next(
            item
            for item in derived["full_hypercube_edges"]
            if item["base_coalition"] == [] and item["event_id"] == 1
        )
        self.assertAlmostEqual(edge["marginal_gain"], -0.2)

    def test_scalar_host_transfer_accounting_is_exact(self) -> None:
        changed = _mutate(
            self.files,
            "workers/even/states/000.json",
            lambda value: value["distance_rows"][0].__setitem__(
                "scalar_host_transfer_count", 0
            ),
        )
        with self.assertRaisesRegex(ValueError, "operation accounting"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_retry_or_top_up_schedule_is_rejected(self) -> None:
        changed = _mutate(
            self.files,
            "workers/even/attempts/000.json",
            lambda value: value.__setitem__("retry_count", 1),
        )
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_any_prohibited_operation_in_run_contract_is_rejected(self) -> None:
        contract = _run_contract(_states())
        for key in PROHIBITED_OPERATION_COUNTS:
            changed = copy.deepcopy(contract)
            changed["prohibited_operation_counts"][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "prohibited-operation"
            ):
                validate_execution_run_contract(changed)

    def test_monitor_gpu_uuid_drift_is_rejected(self) -> None:
        changed = _mutate(
            self.files,
            MONITOR_SUMMARY_FILENAME,
            lambda value: value["visible_gpu_uuids"].reverse(),
        )
        with self.assertRaisesRegex(ValueError, "GPU UUID"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_monitor_summary_is_reconstructed_from_jsonl(self) -> None:
        changed = dict(self.files)
        sample = _monitor_sample()
        sample["gpus"][0]["utilization_percent"] = 1
        changed[LOG_PATHS["utilization_monitor"]] = (
            canonical_json_bytes(sample) + b"\n"
        )
        with self.assertRaisesRegex(ValueError, "archived JSONL"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_single_sample_cannot_claim_ten_minute_coverage(self) -> None:
        changed = dict(self.files)
        changed[LOG_PATHS["utilization_monitor"]] = (
            canonical_json_bytes(_monitor_sample()) + b"\n"
        )
        summary = json.loads(changed[MONITOR_SUMMARY_FILENAME])
        summary["sample_count"] = 1
        changed[MONITOR_SUMMARY_FILENAME] = pretty_json_bytes(summary)
        with self.assertRaisesRegex(ValueError, "complete worker execution window"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_monitor_cannot_stop_before_worker_terminal(self) -> None:
        changed = _mutate(
            self.files,
            MONITOR_SUMMARY_FILENAME,
            lambda value: value.__setitem__("stopped_at_utc", SAMPLE),
        )
        with self.assertRaisesRegex(ValueError, "complete worker execution window"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_monitor_physical_indices_bind_runtime(self) -> None:
        changed = dict(self.files)
        sample = _monitor_sample()
        sample["gpus"][0]["nvidia_smi_index"] = 0
        changed[LOG_PATHS["utilization_monitor"]] = (
            canonical_json_bytes(sample) + b"\n"
        )
        with self.assertRaisesRegex(ValueError, "physical GPU inventory"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_full_runtime_bindings_hash_is_verified(self) -> None:
        changed = _mutate(
            self.files,
            "workers/even/runtime_identity.json",
            lambda value: value["runtime_metadata"]["bindings_metadata"].__setitem__(
                "torch_version", "0.0"
            ),
        )
        with self.assertRaisesRegex(ValueError, "bindings metadata SHA256"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_runtime_dtype_and_eager_attention_are_exact(self) -> None:
        drifts = {
            "dtype": "torch.float16",
            "requested_attention_implementation": "sdpa",
            "observed_attention_implementation": {
                "top": "sdpa",
                "text": "sdpa",
                "vision": "sdpa",
            },
        }
        for key, drifted_value in drifts.items():
            def mutate(value, *, field=key, replacement=drifted_value):
                runtime = value["runtime_metadata"]
                runtime["bindings_metadata"][field] = replacement
                runtime["bindings_metadata_sha256"] = sha256_bytes(
                    canonical_json_bytes(runtime["bindings_metadata"])
                )

            changed = _mutate(
                self.files,
                "workers/even/runtime_identity.json",
                mutate,
            )
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "runtime"
            ):
                validate_expansion_label_evidence_files(
                    changed, require_canonical_attempt_identity=False
                )

    def test_full_tensor_host_transfer_is_rejected(self) -> None:
        changed = _mutate(
            self.files,
            "workers/even/states/000.json",
            lambda value: value["operation_counts"].__setitem__(
                "full_logit_tensor_host_transfer_count", 1
            ),
        )
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_stored_derived_hash_cannot_override_raw_truth(self) -> None:
        changed = _mutate(
            self.files,
            AGGREGATE_FILENAME,
            lambda value: value["reduction"].__setitem__(
                "derived_payload_sha256", "f" * 64
            ),
        )
        with self.assertRaisesRegex(ValueError, "independent raw"):
            validate_expansion_label_evidence_files(
                changed, require_canonical_attempt_identity=False
            )

    def test_deterministic_ustar_round_trip(self) -> None:
        payload = deterministic_ustar_bytes(self.files)
        self.assertEqual(payload, deterministic_ustar_bytes(self.files))
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "labels.tar"
            archive.write_bytes(payload)
            evidence = read_expansion_label_archive(
                archive, require_canonical_attempt_identity=False
            )
        self.assertEqual(evidence.reduction["counts"], EXPECTED_TOTALS)

    def test_sha256_file_is_standard_digest(self) -> None:
        payload = b"known digest payload"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(payload)
            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_atomic_publish_never_leaves_partial_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "artifact.json"

            def fail_publish(_temporary: Path, _final: Path) -> None:
                raise OSError("injected publication failure")

            with self.assertRaisesRegex(OSError, "injected"):
                atomic_exclusive_publish_bytes(
                    final, b"complete", publish_fn=fail_publish
                )
            self.assertFalse(final.exists())
            self.assertEqual(list(root.iterdir()), [])
            atomic_exclusive_publish_bytes(final, b"complete")
            self.assertEqual(final.read_bytes(), b"complete")
            with self.assertRaises(FileExistsError):
                atomic_exclusive_publish_bytes(final, b"replacement")

    def test_runner_freeze_materialization_uses_atomic_publisher(self) -> None:
        freeze = {"frozen": "runner"}

        def fake_git(_root, *args):
            if args[:2] == ("status", "--porcelain"):
                return ""
            return SHA_A

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / CANONICAL_RUNNER_FREEZE_PATH
            with (
                mock.patch.object(artifact_module, "_git_output", side_effect=fake_git),
                mock.patch.object(
                    artifact_module,
                    "build_runner_freeze",
                    return_value=freeze,
                ),
                mock.patch.object(
                    artifact_module,
                    "atomic_exclusive_publish_bytes",
                    wraps=atomic_exclusive_publish_bytes,
                ) as publish,
            ):
                result = artifact_module.materialize_runner_freeze(
                    repository_root=root,
                    runner_source_git_commit=SHA_A,
                )
            publish.assert_called_once_with(output, pretty_json_bytes(freeze))
            self.assertEqual(output.read_bytes(), pretty_json_bytes(freeze))
            self.assertEqual(result["sha256"], sha256_file(output))

    def test_forced_hf_download_resolves_tag_and_revision(self) -> None:
        api_calls: list[dict[str, object]] = []
        download_calls: list[dict[str, object]] = []

        class FakeApi:
            def dataset_info(self, **kwargs):
                api_calls.append(kwargs)
                return SimpleNamespace(
                    sha=HF_REVISION,
                    id=CANONICAL_HF_REPO,
                    private=True,
                )

        def fake_download(**kwargs):
            download_calls.append(kwargs)
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fresh immutable archive")
            return str(path)

        with tempfile.TemporaryDirectory() as directory:
            downloaded, attestation = manager.download_fresh_hf_archive(
                hf_repo=CANONICAL_HF_REPO,
                hf_tag=CANONICAL_HF_TAG,
                hf_path=CANONICAL_HF_PATH,
                hf_immutable_revision=HF_REVISION,
                download_dir=Path(directory).resolve() / "fresh",
                api_factory=FakeApi,
                download_fn=fake_download,
            )
        self.assertEqual(len(api_calls), 4)
        self.assertEqual(download_calls[0]["revision"], HF_REVISION)
        self.assertTrue(download_calls[0]["force_download"])
        self.assertTrue(attestation["tag_resolution_verified"])
        self.assertEqual(attestation["tag_query_revision"], CANONICAL_HF_TAG)
        self.assertEqual(
            attestation["download_requested_revision"], HF_REVISION
        )
        self.assertEqual(attestation["downloaded_path"], str(downloaded))

    def test_fresh_download_rejects_symlink_target(self) -> None:
        class FakeApi:
            def dataset_info(self, **_kwargs):
                return SimpleNamespace(
                    sha=HF_REVISION,
                    id=CANONICAL_HF_REPO,
                    private=True,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root / "outside.tar"
            outside.write_bytes(b"outside")

            def fake_download(**kwargs):
                target = Path(kwargs["local_dir"]) / kwargs["filename"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(outside)
                return str(target)

            with self.assertRaisesRegex(ValueError, "symlink"):
                manager.download_fresh_hf_archive(
                    hf_repo=CANONICAL_HF_REPO,
                    hf_tag=CANONICAL_HF_TAG,
                    hf_path=CANONICAL_HF_PATH,
                    hf_immutable_revision=HF_REVISION,
                    download_dir=root / "fresh",
                    api_factory=FakeApi,
                    download_fn=fake_download,
                )

    def test_fresh_download_rejects_tag_toctou(self) -> None:
        class FakeApi:
            def __init__(self):
                self.calls = 0

            def dataset_info(self, **_kwargs):
                self.calls += 1
                revision = HF_REVISION if self.calls <= 2 else "e" * 40
                return SimpleNamespace(
                    sha=revision,
                    id=CANONICAL_HF_REPO,
                    private=True,
                )

        def fake_download(**kwargs):
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"download")
            return str(target)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "resolved revision"):
                manager.download_fresh_hf_archive(
                    hf_repo=CANONICAL_HF_REPO,
                    hf_tag=CANONICAL_HF_TAG,
                    hf_path=CANONICAL_HF_PATH,
                    hf_immutable_revision=HF_REVISION,
                    download_dir=Path(directory).resolve() / "fresh",
                    api_factory=FakeApi,
                    download_fn=fake_download,
                )

    def test_upload_and_tag_is_private_fresh_and_no_overwrite(self) -> None:
        parent_commit = "1" * 40
        immutable_commit = "2" * 40

        class FakeApi:
            def __init__(self):
                self.calls: list[tuple[str, dict[str, object]]] = []

            def create_repo(self, **kwargs):
                self.calls.append(("create_repo", kwargs))

            def dataset_info(self, **kwargs):
                self.calls.append(("dataset_info", kwargs))
                revision = kwargs.get("revision")
                sha = parent_commit if revision is None else immutable_commit
                return SimpleNamespace(
                    sha=sha,
                    id=CANONICAL_HF_REPO,
                    private=True,
                )

            def list_repo_refs(self, **kwargs):
                self.calls.append(("list_repo_refs", kwargs))
                return SimpleNamespace(tags=[])

            def list_repo_files(self, **kwargs):
                self.calls.append(("list_repo_files", kwargs))
                if kwargs["revision"] == parent_commit:
                    return []
                return [CANONICAL_HF_PATH]

            def upload_file(self, **kwargs):
                self.calls.append(("upload_file", kwargs))
                return SimpleNamespace(oid=immutable_commit)

            def create_tag(self, **kwargs):
                self.calls.append(("create_tag", kwargs))

        api = FakeApi()
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory).resolve() / "labels.tar"
            archive.write_bytes(deterministic_ustar_bytes(self.files))
            attestation = manager.upload_and_tag_archive(
                source_archive=archive,
                source_git_commit=SHA_A,
                config_sha256=SHA_CONFIG,
                hf_repo=CANONICAL_HF_REPO,
                hf_tag=CANONICAL_HF_TAG,
                hf_path=CANONICAL_HF_PATH,
                api_factory=lambda: api,
            )
        upload_call = next(payload for name, payload in api.calls if name == "upload_file")
        tag_call = next(payload for name, payload in api.calls if name == "create_tag")
        repo_call = next(payload for name, payload in api.calls if name == "create_repo")
        self.assertEqual(upload_call["parent_commit"], parent_commit)
        self.assertFalse(tag_call["exist_ok"])
        self.assertEqual(tag_call["revision"], immutable_commit)
        self.assertTrue(repo_call["private"])
        self.assertTrue(repo_call["exist_ok"])
        self.assertEqual(attestation["immutable_revision"], immutable_commit)

    def test_upload_refuses_existing_tag_or_path_before_upload(self) -> None:
        class CollisionApi:
            def __init__(self, *, tag_exists: bool, path_exists: bool):
                self.tag_exists = tag_exists
                self.path_exists = path_exists
                self.upload_called = False

            def create_repo(self, **_kwargs):
                return None

            def dataset_info(self, **_kwargs):
                return SimpleNamespace(
                    sha="1" * 40,
                    id=CANONICAL_HF_REPO,
                    private=True,
                )

            def list_repo_refs(self, **_kwargs):
                tags = (
                    [SimpleNamespace(name=CANONICAL_HF_TAG)]
                    if self.tag_exists
                    else []
                )
                return SimpleNamespace(tags=tags)

            def list_repo_files(self, **_kwargs):
                return [CANONICAL_HF_PATH] if self.path_exists else []

            def upload_file(self, **_kwargs):
                self.upload_called = True
                raise AssertionError("collision guard failed")

        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory).resolve() / "labels.tar"
            archive.write_bytes(deterministic_ustar_bytes(self.files))
            for tag_exists, path_exists in ((True, False), (False, True)):
                api = CollisionApi(
                    tag_exists=tag_exists,
                    path_exists=path_exists,
                )
                with self.subTest(
                    tag_exists=tag_exists,
                    path_exists=path_exists,
                ), self.assertRaises(FileExistsError):
                    manager.upload_and_tag_archive(
                        source_archive=archive,
                        source_git_commit=SHA_A,
                        config_sha256=SHA_CONFIG,
                        hf_repo=CANONICAL_HF_REPO,
                        hf_tag=CANONICAL_HF_TAG,
                        hf_path=CANONICAL_HF_PATH,
                        api_factory=lambda current=api: current,
                    )
                self.assertFalse(api.upload_called)

    def test_create_manifest_cli_uses_atomic_output(self) -> None:
        fake_manifest = {
            "result": {"outcome": PASS_OUTCOME},
            "raw_archive": {"sha256": _sha("archive")},
            "hf_artifact": {"immutable_revision": HF_REVISION},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "manifest.json"
            downloaded = root / "downloaded.tar"
            downloaded.write_bytes(b"download")
            argv = [
                "create-manifest",
                "--source-archive",
                str(root / "source.tar"),
                "--download-dir",
                str(root / "fresh"),
                "--source-git-commit",
                SHA_A,
                "--config-sha256",
                SHA_CONFIG,
                "--hf-immutable-revision",
                HF_REVISION,
                "--parent-substrate-archive",
                str(root / "parent.tar"),
                "--derived-artifact-root",
                str(root / "derived"),
                "--ocr-backend-config",
                str(root / "code/configs/restoration_v2_ocr_backend.json"),
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(
                    manager,
                    "_download",
                    return_value=(downloaded, {"download": "attested"}),
                ),
                mock.patch.object(
                    manager,
                    "build_expansion_label_artifact_manifest",
                    return_value=fake_manifest,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                manager.main(argv)
            self.assertEqual(json.loads(output.read_bytes()), fake_manifest)
            self.assertFalse(any(path.name.endswith(".tmp") for path in root.iterdir()))

    def test_runner_freeze_schema_binds_locked_base_and_reserved(self) -> None:
        locked = [
            {"path": "code/locked.py", "sha256": _sha("locked"), "size_bytes": 1}
        ]
        base = {
            "scientific_source_lock": {
                "source_files": locked,
                "reserved_execution_source_paths": sorted(
                    SOURCE_PATHS_REQUIRED_IN_RUNNER_FREEZE
                ),
            }
        }
        base_payload = pretty_json_bytes(base)
        records = locked + [
            {
                "path": CANONICAL_CONFIG_PATH,
                "sha256": sha256_bytes(base_payload),
                "size_bytes": len(base_payload),
            }
        ]
        records.extend(
            {"path": path, "sha256": _sha(path), "size_bytes": 1}
            for path in SOURCE_PATHS_REQUIRED_IN_RUNNER_FREEZE
        )
        records.sort(key=lambda item: item["path"])
        freeze = {
            "schema_version": RUNNER_FREEZE_SCHEMA_VERSION,
            "protocol_id": RUNNER_FREEZE_PROTOCOL_ID,
            "status": "FROZEN_COMMITTED_PUSHED_EXPANSION_EXACT_LABEL_RUNNER_SOURCE",
            "runner_source_git_commit": SHA_A,
            "base_config": next(
                item for item in records if item["path"] == CANONICAL_CONFIG_PATH
            ),
            "source_inventory": records,
            "source_inventory_sha256": sha256_bytes(canonical_json_bytes(records)),
            "interfaces": runner_interface_contract(),
            "authorization": {
                "runner_source_commit_must_be_committed": True,
                "runner_source_commit_must_be_ancestor_of_execution_head": True,
                "execution_head_and_origin_main_must_match": True,
                "runner_freeze_file_must_be_committed_at_execution_head": True,
                "source_blobs_must_equal_runner_source_commit": True,
                "source_only_freeze_authorizes_gpu_execution": False,
                "formal_execution_requires_runtime_authorization_validation": True,
            },
        }
        validation = validate_runner_freeze_data(freeze, base_config=base)
        self.assertEqual(validation["runner_source_git_commit"], SHA_A)
        broken = copy.deepcopy(freeze)
        broken["source_inventory"] = broken["source_inventory"][:-1]
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_runner_freeze_data(broken, base_config=base)


if __name__ == "__main__":
    unittest.main()
