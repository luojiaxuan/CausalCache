from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest import mock

import causalcache.restoration_v2_2_expansion_labels_artifact as frozen_v1
import causalcache.restoration_v2_2_expansion_labels_invalid_forensic as forensic
import causalcache.restoration_v2_2_expansion_labels_scientific_repair as repair
from causalcache.restoration_v2_2_expansion_labels_invalid_forensic import (
    ARTIFACT_CLASS,
    ATTEMPT_ROOT_NAMESPACE,
    FORENSIC_MANIFEST_PATH,
    PROTOCOL_ID as FORENSIC_PROTOCOL_ID,
    PRODUCER_COMPLETED_STATUS,
    PRODUCER_INVALID_STATUS,
    SOURCE_STATUS,
    TERMINAL_LEDGER_NAMESPACE,
    InvalidForensicContract,
    InvalidForensicEvidence,
    canonical_json_bytes as forensic_canonical_json_bytes,
    forensic_contract_from_data,
    pretty_json_bytes as forensic_pretty_json_bytes,
    sha256_bytes as forensic_sha256_bytes,
    validate_invalid_forensic_files,
)
from causalcache.restoration_v2_2_expansion_labels_scientific_repair import (
    FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION,
    FROZEN_OCR_BACKEND_CONFIG,
    FROZEN_P0_CONFIG_SHA256,
    FROZEN_P0_FORENSIC_TREE_SHA256,
    LEGACY_CADENCE_FAILURE,
    PASS_STATUS,
    TEST_EXTERNAL_REPLAY_STATUS,
    TEST_ONLY_STATUS,
    FrozenExternalReplayAttestation,
    FrozenExternalReplayExecutor,
    TestOnlyExternalReplayAttestation,
    validate_ledger_neutral_scientific_payload,
)
from tests.test_restoration_v2_2_expansion_labels_artifact import _evidence_files


TEST_FIXTURE_ID = "synthetic-frozen-v1-embedded-state-echo"


def _inventory(files: dict[str, bytes]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "path": name,
            "sha256": forensic_sha256_bytes(files[name]),
            "size_bytes": len(files[name]),
        }
        for name in sorted(files)
    )


def _strict_object(payload: bytes) -> dict[str, object]:
    value = json.loads(payload)
    assert isinstance(value, dict)
    return value


def _ordered_state_records(files: dict[str, bytes]) -> list[dict[str, object]]:
    return [
        _strict_object(
            files[
                f"workers/{'even' if index % 2 == 0 else 'odd'}/states/{index:03d}.json"
            ]
        )
        for index in range(frozen_v1.EXPECTED_STATE_COUNT)
    ]


def _rebind_monitor_and_aggregate(files: dict[str, bytes]) -> None:
    execution = _strict_object(files[frozen_v1.EXECUTION_EVIDENCE_FILENAME])
    execution["utilization_monitor_log_sha256"] = frozen_v1.sha256_bytes(
        files[frozen_v1.LOG_PATHS["utilization_monitor"]]
    )
    execution["monitor_summary_sha256"] = frozen_v1.sha256_bytes(
        files[frozen_v1.MONITOR_SUMMARY_FILENAME]
    )
    execution["monitor_stop_request_sha256"] = frozen_v1.sha256_bytes(
        files[frozen_v1.LOG_PATHS["utilization_monitor_stop_request"]]
    )
    files[frozen_v1.EXECUTION_EVIDENCE_FILENAME] = frozen_v1.pretty_json_bytes(
        execution
    )
    run_manifest = _strict_object(files[frozen_v1.RUN_MANIFEST_FILENAME])
    run_contract = run_manifest["run_contract"]
    assert isinstance(run_contract, dict)
    run_sha = run_manifest["run_contract_sha256"]
    assert isinstance(run_sha, str)
    reduction = frozen_v1.reduce_raw_distance_states(
        _ordered_state_records(files),
        expected_states=run_contract["states"],
        run_contract_sha256=run_sha,
    )
    aggregate = _strict_object(files[frozen_v1.AGGREGATE_FILENAME])
    files[frozen_v1.AGGREGATE_FILENAME] = frozen_v1.pretty_json_bytes(
        frozen_v1.aggregate_from_reduction(
            reduction,
            run_contract_sha256=run_sha,
            execution_evidence_sha256=frozen_v1.sha256_bytes(
                files[frozen_v1.EXECUTION_EVIDENCE_FILENAME]
            ),
            started_at_utc=str(aggregate["started_at_utc"]),
            ended_at_utc=str(aggregate["ended_at_utc"]),
        )
    )


def _forensic_fixture(
    *,
    cadence_failure: bool = True,
    duplicate_sample_timestamp: bool = False,
    failure_message: str = LEGACY_CADENCE_FAILURE,
    suffix_timestamp: str = "2026-07-17T00:10:04Z",
) -> tuple[InvalidForensicContract, InvalidForensicEvidence]:
    root = _evidence_files()
    samples = [
        json.loads(line)
        for line in root[frozen_v1.LOG_PATHS["utilization_monitor"]].splitlines()
    ]
    if cadence_failure:
        samples[100]["sampled_at_utc"] = "2026-07-17T00:03:22.500000Z"
    samples[-1]["sampled_at_utc"] = "2026-07-17T00:10:00Z"
    if duplicate_sample_timestamp:
        samples[50]["sampled_at_utc"] = samples[49]["sampled_at_utc"]
    root[frozen_v1.LOG_PATHS["utilization_monitor"]] = b"".join(
        frozen_v1.canonical_json_bytes(sample) + b"\n" for sample in samples
    )
    stop = _strict_object(
        root[frozen_v1.LOG_PATHS["utilization_monitor_stop_request"]]
    )
    stop["requested_at_utc"] = "2026-07-17T00:10:01.500000Z"
    root[frozen_v1.LOG_PATHS["utilization_monitor_stop_request"]] = (
        frozen_v1.pretty_json_bytes(stop)
    )
    summary = _strict_object(root[frozen_v1.MONITOR_SUMMARY_FILENAME])
    summary["stopped_at_utc"] = "2026-07-17T00:10:02Z"
    summary["stop_request_sha256"] = frozen_v1.sha256_bytes(
        root[frozen_v1.LOG_PATHS["utilization_monitor_stop_request"]]
    )
    root[frozen_v1.MONITOR_SUMMARY_FILENAME] = frozen_v1.pretty_json_bytes(summary)
    _rebind_monitor_and_aggregate(root)

    prefix = root[frozen_v1.LOG_PATHS["execution"]]
    suffix = (
        suffix_timestamp
        + " attempt invalidated: "
        + failure_message
        + "\n"
    ).encode("utf-8")
    root[frozen_v1.LOG_PATHS["execution"]] = prefix + suffix
    internal = _strict_object(root[frozen_v1.GLOBAL_LEDGER_ARCHIVE_NAME])
    external = {
        **internal,
        "status": PRODUCER_INVALID_STATUS,
        "ended_at_utc": "2026-07-17T00:10:03Z",
        "failure": {
            "exception_type": "ValueError",
            "message": failure_message,
        },
        "claimed_ledger_sha256": forensic_sha256_bytes(
            root[frozen_v1.GLOBAL_LEDGER_ARCHIVE_NAME]
        ),
    }
    payload_files = {
        f"{ATTEMPT_ROOT_NAMESPACE}/{name}": payload for name, payload in root.items()
    }
    payload_files[
        f"{TERMINAL_LEDGER_NAMESPACE}/global_attempt_ledger.json"
    ] = forensic_pretty_json_bytes(external)
    for worker in ("even", "odd"):
        payload_files[
            f"{TERMINAL_LEDGER_NAMESPACE}/workers/{worker}.json"
        ] = root[f"{frozen_v1.WORKER_SIBLING_LEDGER_DIRECTORY}/{worker}.json"]
    internal_global = _strict_object(root[frozen_v1.GLOBAL_LEDGER_ARCHIVE_NAME])
    worker_paths = internal_global["worker_sibling_ledgers"]
    assert isinstance(worker_paths, dict)
    run_manifest = _strict_object(root[frozen_v1.RUN_MANIFEST_FILENAME])
    run_contract = run_manifest["run_contract"]
    assert isinstance(run_contract, dict)
    attempt_identity = run_contract["attempt_identity"]
    assert isinstance(attempt_identity, dict)
    root_inventory = _inventory(root)
    critical_names = {
        "aggregate.json",
        "execution_evidence.json",
        "global_attempt_ledger.json",
        "logs/execution.log",
        "logs/gpu_utilization_monitor.log",
        "monitor_summary.json",
        "run_manifest.json",
    }
    execution_evidence = root[frozen_v1.EXECUTION_EVIDENCE_FILENAME]
    contract_data = {
        "schema_version": "1.0.0",
        "protocol_id": FORENSIC_PROTOCOL_ID,
        "status": SOURCE_STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "source_attempt": {
            "producer_protocol_id": frozen_v1.PROTOCOL_ID,
            "attempt_id": frozen_v1.ATTEMPT_ID,
            "run_contract_sha256": internal["run_contract_sha256"],
            "expected_state_count": frozen_v1.EXPECTED_STATE_COUNT,
            "output_root": internal_global["output_dir"],
            "root_file_count": len(root),
            "root_total_file_bytes": sum(len(payload) for payload in root.values()),
            "root_inventory_sha256": forensic_sha256_bytes(
                forensic_canonical_json_bytes(root_inventory)
            ),
            "critical_root_file_sha256": {
                name: forensic_sha256_bytes(root[name]) for name in critical_names
            },
            "external_global_ledger": {
                "path": attempt_identity["global_ledger_path"],
                "sha256": forensic_sha256_bytes(
                    forensic_pretty_json_bytes(external)
                ),
                "size_bytes": len(forensic_pretty_json_bytes(external)),
            },
            "external_worker_ledgers": {
                worker: {
                    "path": worker_paths[worker],
                    "sha256": forensic_sha256_bytes(
                        root[
                            f"{frozen_v1.WORKER_SIBLING_LEDGER_DIRECTORY}/"
                            f"{worker}.json"
                        ]
                    ),
                    "size_bytes": len(
                        root[
                            f"{frozen_v1.WORKER_SIBLING_LEDGER_DIRECTORY}/"
                            f"{worker}.json"
                        ]
                    ),
                }
                for worker in ("even", "odd")
            },
            "append_only_execution_log": {
                "execution_evidence_path": frozen_v1.EXECUTION_EVIDENCE_FILENAME,
                "execution_evidence_sha256": forensic_sha256_bytes(
                    execution_evidence
                ),
                "final_log_path": frozen_v1.LOG_PATHS["execution"],
                "final_log_sha256": forensic_sha256_bytes(prefix + suffix),
                "final_log_size_bytes": len(prefix + suffix),
                "pre_failure_prefix_sha256": forensic_sha256_bytes(prefix),
                "pre_failure_prefix_size_bytes": len(prefix),
                "terminal_suffix_sha256": forensic_sha256_bytes(suffix),
                "terminal_suffix_size_bytes": len(suffix),
            },
        },
        "failure_contract": {
            "exception_type": "ValueError",
            "message": failure_message,
            "original_attempt_status": PRODUCER_INVALID_STATUS,
            "root_embedded_snapshot_status": PRODUCER_COMPLETED_STATUS,
        },
        "archive_contract": {
            "archive_format": "ustar",
            "archive_member_prefix": "synthetic-scientific-repair-v1",
            "canonical_archive_path": (
                str(attempt_identity["archive_path"]) + ".invalid-forensic.tar"
            ),
            "attempt_root_namespace": ATTEMPT_ROOT_NAMESPACE,
            "terminal_external_ledger_namespace": TERMINAL_LEDGER_NAMESPACE,
            "forensic_manifest_path": FORENSIC_MANIFEST_PATH,
            "expected_payload_file_count_before_manifest": len(payload_files),
            "expected_member_count": len(payload_files) + 1,
            "deterministic_regular_file_metadata": {
                "mode": 0o644,
                "uid": 0,
                "gid": 0,
                "uname": "",
                "gname": "",
                "mtime": 0,
                "pax_headers": {},
            },
            "overwrite_allowed": False,
        },
        "formal_consumption": {
            "formal_label_loader_eligible": False,
            "gate_training_unlocked": False,
            "matched_nll_unlocked": False,
            "closed_loop_unlocked": False,
            "reason": "synthetic scientific-repair fixture is never formal labels",
        },
        "network_contract": {
            "hf_publish_authorized": False,
            "network_access_required": False,
            "upload_command_present": False,
        },
    }
    contract = forensic_contract_from_data(contract_data)
    manifest = forensic._manifest_payload(payload_files, contract)
    files = dict(payload_files)
    files[FORENSIC_MANIFEST_PATH] = forensic_pretty_json_bytes(manifest)
    evidence = validate_invalid_forensic_files(files, contract=contract)
    return contract, evidence


def _external_replay(run_contract: dict[str, object]):
    states = frozen_v1.validate_state_projections(run_contract["states"])
    state_bytes = frozen_v1.canonical_json_bytes(states)
    state_sha256 = frozen_v1.sha256_bytes(state_bytes)
    return states, {
        "external_input_replay_verified": True,
        "policy_or_model_forward_executed": False,
        "state_count": frozen_v1.EXPECTED_STATE_COUNT,
        "coalition_input_witness_count": frozen_v1.EXPECTED_DISTANCE_ROWS,
        "external_state_projections_sha256": state_sha256,
        "test_only_fixture_replay": True,
        "fixture_id": TEST_FIXTURE_ID,
        "fixture_state_projections_sha256": state_sha256,
    }


def _test_replay_attestation() -> TestOnlyExternalReplayAttestation:
    root = _evidence_files()
    manifest = _strict_object(root[frozen_v1.RUN_MANIFEST_FILENAME])
    run_contract = manifest["run_contract"]
    assert isinstance(run_contract, dict)
    states = frozen_v1.validate_state_projections(run_contract["states"])
    return TestOnlyExternalReplayAttestation(
        fixture_id=TEST_FIXTURE_ID,
        fixture_state_projections_sha256=frozen_v1.sha256_bytes(
            frozen_v1.canonical_json_bytes(states)
        ),
    )


def _formal_replay_attestation() -> FrozenExternalReplayAttestation:
    return FrozenExternalReplayAttestation(
        implementation=dict(FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION),
        parent_substrate=dict(repair.FROZEN_PARENT_SUBSTRATE),
        derived_artifact=dict(repair.FROZEN_DERIVED_ARTIFACT),
        ocr_backend_config=dict(FROZEN_OCR_BACKEND_CONFIG),
        parent_substrate_archive_path="/tmp/parent-substrate.tar",
        derived_artifact_root="/tmp/derived-artifact",
        ocr_backend_config_path=(
            "/tmp/repository/code/configs/restoration_v2_ocr_backend.json"
        ),
    )


class ExpansionLabelScientificRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract, cls.evidence = _forensic_fixture()
        cls.replay_attestation = _test_replay_attestation()
        cls.before = copy.deepcopy(cls.evidence.files)
        cls.result = validate_ledger_neutral_scientific_payload(
            cls.evidence,
            forensic_contract=cls.contract,
            external_input_replay_callback=_external_replay,
            external_replay_attestation=cls.replay_attestation,
        )

    def test_full_ledger_neutral_repair_replays_science(self) -> None:
        report = self.result.report
        self.assertEqual(report["status"], TEST_ONLY_STATUS)
        self.assertNotEqual(report["status"], PASS_STATUS)
        self.assertFalse(report["formal_scientific_repair_pass"])
        self.assertTrue(
            report["original_attempt"]["formal_v1_outcome_remains_invalid"]
        )
        self.assertFalse(report["original_attempt"]["formal_v1_pass_claimed"])
        self.assertEqual(
            report["legacy_validator_negative_control"],
            {
                "exception_type": "ValueError",
                "message": LEGACY_CADENCE_FAILURE,
                "exactly_reproduced": True,
            },
        )
        self.assertEqual(
            report["raw_distance_reduction"]["counts"], frozen_v1.EXPECTED_TOTALS
        )
        self.assertTrue(
            report["raw_distance_reduction"]["stored_aggregate_exactly_rebuilt"]
        )
        self.assertEqual(
            report["independent_stdlib_math_audit"]["status"],
            "PASS_INDEPENDENT_EXPANSION_MATH_AUDIT",
        )
        self.assertEqual(
            report["independent_stdlib_math_audit"]["compared_state_count"], 192
        )
        self.assertTrue(
            report["external_input_replay"]["byte_identical_to_embedded_run_contract"]
        )
        self.assertEqual(
            report["external_input_replay"]["status"],
            TEST_EXTERNAL_REPLAY_STATUS,
        )
        self.assertFalse(
            report["external_input_replay"][
                "formal_external_input_replay_verified"
            ]
        )
        self.assertTrue(
            report["external_input_replay"][
                "embedded_echo_is_not_external_replay_evidence"
            ]
        )
        self.assertTrue(
            report["forensic_evidence"]["strict_p0_revalidation_performed"]
        )
        self.assertEqual(
            report["forensic_evidence"][
                "formal_required_forensic_contract_sha256"
            ],
            FROZEN_P0_CONFIG_SHA256,
        )
        self.assertEqual(
            report["forensic_evidence"][
                "formal_required_tree_inventory_sha256"
            ],
            FROZEN_P0_FORENSIC_TREE_SHA256,
        )
        self.assertFalse(
            report["forensic_evidence"][
                "exact_frozen_p0_source_verified_for_formal_pass"
            ]
        )

    def test_core_has_no_publication_or_hf_dependency(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "causalcache/restoration_v2_2_expansion_labels_scientific_repair.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("invalid_forensic_publication", source)
        self.assertNotIn("huggingface_hub", source)

    def test_structural_monitor_accepts_gap_without_changing_legacy_fact(self) -> None:
        monitor = self.result.report["monitor_validation"]
        self.assertEqual(
            monitor["status"], "PASS_STRUCTURAL_MONITOR_LIFECYCLE_VALIDATION"
        )
        self.assertFalse(monitor["acceptance_uses_numeric_gap_threshold"])
        self.assertTrue(monitor["legacy_cadence_formula_exactly_reproduced"])
        self.assertFalse(monitor["legacy_cadence_gate_passed"])
        self.assertEqual(
            monitor["sample_gap_statistics_seconds"]["maximum"], 3.5
        )
        self.assertEqual(
            monitor["sample_gap_statistics_seconds"]["count_above_3_seconds"], 1
        )
        self.assertEqual(
            len(monitor["sample_gap_microseconds"]), monitor["sample_count"] - 1
        )
        self.assertEqual(
            len(monitor["coverage_gap_microseconds"]), monitor["sample_count"] + 1
        )
        self.assertEqual(
            monitor["coverage_gap_microseconds"][0],
            monitor["boundary_gap_microseconds"]["ready_to_first_sample"],
        )
        self.assertEqual(
            monitor["coverage_gap_microseconds"][-1],
            monitor["boundary_gap_microseconds"]["last_sample_to_summary"],
        )

    def test_execution_log_prefix_is_validated_without_mutating_suffix(self) -> None:
        log = self.result.report["append_only_execution_log"]
        self.assertTrue(log["execution_evidence_binds_pre_invalidation_prefix"])
        self.assertFalse(log["forensic_source_log_bytes_modified"])
        self.assertTrue(log["in_memory_pre_invalidation_view_reconstructed"])
        self.assertTrue(
            log["invalidation_suffix_not_before_terminal_invalid_ledger"]
        )
        self.assertEqual(
            log["invalidation_suffix_after_terminal_ledger_microseconds"],
            1_000_000,
        )
        self.assertEqual(self.evidence.files, self.before)
        self.assertEqual(
            self.result.report["forensic_evidence"]["ledger_mutation_count"], 0
        )

    def test_nonfrozen_failure_is_rejected_after_strict_p0_validation(self) -> None:
        contract, evidence = _forensic_fixture(
            failure_message=LEGACY_CADENCE_FAILURE + " extra"
        )
        with self.assertRaisesRegex(ValueError, "global ledger chain drifted"):
            validate_ledger_neutral_scientific_payload(
                evidence,
                forensic_contract=contract,
                external_input_replay_callback=_external_replay,
                external_replay_attestation=self.replay_attestation,
            )

    def test_unvalidated_file_mapping_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "P0-validated"):
            validate_ledger_neutral_scientific_payload(
                self.evidence.files,
                forensic_contract=self.contract,
                external_input_replay_callback=_external_replay,
                external_replay_attestation=self.replay_attestation,
            )

    def test_forged_evidence_dataclass_is_strictly_reconstructed(self) -> None:
        variants = {
            "manifest": {"manifest": {}},
            "inventory": {"inventory": tuple(reversed(self.evidence.inventory))},
            "tree": {"tree_inventory_sha256": "f" * 64},
        }
        for label, replacement in variants.items():
            values = {
                "files": self.evidence.files,
                "manifest": self.evidence.manifest,
                "inventory": self.evidence.inventory,
                "tree_inventory_sha256": self.evidence.tree_inventory_sha256,
                **replacement,
            }
            forged = InvalidForensicEvidence(**values)
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "strict reconstruction"
            ):
                validate_ledger_neutral_scientific_payload(
                    forged,
                    forensic_contract=self.contract,
                    external_input_replay_callback=_external_replay,
                    external_replay_attestation=self.replay_attestation,
                )

    def test_forged_forensic_contract_sha_is_rejected(self) -> None:
        forged_contract = InvalidForensicContract(
            data=self.contract.data,
            sha256="f" * 64,
        )
        with self.assertRaisesRegex(ValueError, "identity|contract differs"):
            validate_ledger_neutral_scientific_payload(
                self.evidence,
                forensic_contract=forged_contract,
                external_input_replay_callback=_external_replay,
                external_replay_attestation=self.replay_attestation,
            )

    def test_invalidation_suffix_cannot_predate_terminal_invalid_ledger(self) -> None:
        with self.assertRaisesRegex(ValueError, "predates invalid ledger"):
            _forensic_fixture(suffix_timestamp="2026-07-17T00:10:02Z")

    def test_non_strict_monitor_timestamps_fail_even_with_cadence_failure(self) -> None:
        contract, evidence = _forensic_fixture(duplicate_sample_timestamp=True)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_ledger_neutral_scientific_payload(
                evidence,
                forensic_contract=contract,
                external_input_replay_callback=_external_replay,
                external_replay_attestation=self.replay_attestation,
            )

    def test_exact_legacy_failure_is_required(self) -> None:
        contract, evidence = _forensic_fixture(cadence_failure=False)
        with self.assertRaisesRegex(ValueError, "unexpectedly accepted"):
            validate_ledger_neutral_scientific_payload(
                evidence,
                forensic_contract=contract,
                external_input_replay_callback=_external_replay,
                external_replay_attestation=self.replay_attestation,
            )

    def test_external_replay_callback_is_fail_closed(self) -> None:
        def bad_callback(run_contract):
            states, metadata = _external_replay(run_contract)
            metadata["coalition_input_witness_count"] -= 1
            return states, metadata

        with self.assertRaisesRegex(ValueError, "metadata drifted"):
            validate_ledger_neutral_scientific_payload(
                self.evidence,
                forensic_contract=self.contract,
                external_input_replay_callback=bad_callback,
                external_replay_attestation=self.replay_attestation,
            )

    def test_external_replay_callback_cannot_mutate_contract_copy(self) -> None:
        def mutating_callback(run_contract):
            states, metadata = _external_replay(run_contract)
            run_contract["states"][0]["state_id"] = "mutated-after-replay"
            return states, metadata

        with self.assertRaisesRegex(ValueError, "mutated its run-contract copy"):
            validate_ledger_neutral_scientific_payload(
                self.evidence,
                forensic_contract=self.contract,
                external_input_replay_callback=mutating_callback,
                external_replay_attestation=self.replay_attestation,
            )
        self.assertEqual(self.evidence.files, self.before)

    def test_embedded_echo_callback_cannot_claim_formal_replay(self) -> None:
        formal_attestation = _formal_replay_attestation()
        with self.assertRaisesRegex(TypeError, "bound frozen replay executor"):
            repair._validate_external_replay(
                self.result.run_contract,
                callback=_external_replay,
                attestation=formal_attestation,
            )

    def test_formal_stub_cannot_pass_with_synthetic_p0_source(self) -> None:
        callback_count = 0

        def formal_stub_callback(_run_contract):
            nonlocal callback_count
            callback_count += 1
            raise AssertionError("formal callback must not run for synthetic P0")

        with self.assertRaisesRegex(ValueError, "exact frozen P0 config and tree"):
            validate_ledger_neutral_scientific_payload(
                self.evidence,
                forensic_contract=self.contract,
                external_input_replay_callback=formal_stub_callback,
                external_replay_attestation=_formal_replay_attestation(),
            )
        self.assertEqual(callback_count, 0)

    def test_same_named_monkeypatched_replay_callable_is_rejected(self) -> None:
        def fake_replay_external_input_projections(**_kwargs):
            raise AssertionError("fake replay must never execute")

        fake_replay_external_input_projections.__module__ = (
            FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION["module"]
        )
        fake_replay_external_input_projections.__qualname__ = (
            FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION["qualname"]
        )
        attestation = _formal_replay_attestation()
        executor = FrozenExternalReplayExecutor(attestation)
        with mock.patch.object(
            frozen_v1,
            "replay_external_input_projections",
            fake_replay_external_input_projections,
        ), self.assertRaisesRegex(ValueError, "implementation bytes drifted"):
            repair._validate_external_replay(
                self.result.run_contract,
                callback=executor,
                attestation=attestation,
            )

    def test_formal_path_ignores_monkeypatched_executor_call(self) -> None:
        callback_count = 0
        attestation = _formal_replay_attestation()
        executor = FrozenExternalReplayExecutor(attestation)
        self.assertIs(
            executor._replay_callable,
            repair._CAPTURED_FROZEN_EXTERNAL_REPLAY,
        )

        def fake_executor_call(_executor, _run_contract):
            nonlocal callback_count
            callback_count += 1
            raise AssertionError("monkeypatched executor call must not run")

        with mock.patch.object(
            FrozenExternalReplayExecutor,
            "__call__",
            autospec=True,
            side_effect=fake_executor_call,
        ), self.assertRaisesRegex(
            ValueError, "parent substrate archive bytes drifted"
        ):
            repair._validate_external_replay(
                self.result.run_contract,
                callback=executor,
                attestation=attestation,
            )
        self.assertEqual(callback_count, 0)

    def test_mutated_provenance_global_is_rejected_before_callback(self) -> None:
        callback_count = 0

        def counting_callback(_run_contract):
            nonlocal callback_count
            callback_count += 1
            raise AssertionError("callback must not execute after provenance drift")

        mutated_parent = dict(repair.FROZEN_PARENT_SUBSTRATE)
        mutated_parent["archive_sha256"] = "f" * 64
        mutated_attestation = FrozenExternalReplayAttestation(
            implementation=dict(FROZEN_EXTERNAL_REPLAY_IMPLEMENTATION),
            parent_substrate=mutated_parent,
            derived_artifact=dict(repair.FROZEN_DERIVED_ARTIFACT),
            ocr_backend_config=dict(FROZEN_OCR_BACKEND_CONFIG),
            parent_substrate_archive_path="/tmp/parent-substrate.tar",
            derived_artifact_root="/tmp/derived-artifact",
            ocr_backend_config_path=(
                "/tmp/repository/code/configs/restoration_v2_ocr_backend.json"
            ),
        )
        with mock.patch.object(
            frozen_v1,
            "CANONICAL_PARENT_SUBSTRATE",
            mutated_parent,
        ), self.assertRaisesRegex(ValueError, "provenance globals drifted"):
            repair._validate_external_replay(
                self.result.run_contract,
                callback=counting_callback,
                attestation=mutated_attestation,
            )
        self.assertEqual(callback_count, 0)

    def test_core_does_not_claim_global_zero_operation_counts(self) -> None:
        report = self.result.report
        self.assertNotIn("validation_added_operation_counts", report)
        self.assertTrue(
            report["execution_scope"][
                "core_does_not_claim_global_gpu_or_model_operation_counts"
            ]
        )
        self.assertTrue(
            report["execution_scope"][
                "formal_zero_operation_attestation_deferred_to_runner"
            ]
        )


if __name__ == "__main__":
    unittest.main()
