from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from causalcache.restoration_v2_2_policy_vision import (
    load_identity_witness,
    validate_feature_worker_provenance,
)
from causalcache.restoration_v2_2_policy_vision_v3_validation_repair import (
    EVALUATED_STATE_KEYS,
    FEATURE_STATE_KEYS,
    CompletionSealSnapshot,
    RepairLedgerSnapshot,
    ReplayResult,
    ZERO_CPU_REPAIR_OPERATIONS,
    _audit_readme,
    _artifact_snapshot,
    build_audit_files,
    build_completion_seal_record,
    build_runtime_identity,
    build_runtime_record,
    claim_attempt_ledger,
    claim_completion_seal,
    expected_no_nvidia_runtime,
    feature_records_from_evaluated_rows,
    project_evaluated_state_to_feature_state,
    replay_exact_producer_artifact,
    validate_audit_against_replay,
    validate_no_nvidia_devices,
    validate_repair_attempt_ledger,
    write_atomic_result,
)
from causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract import (
    ARTIFACT_GIT_COMMIT,
    CANONICAL_CONFIG_PATH,
    FORMAL_ATTEMPT_LEDGER_PATH,
    FORMAL_COMPLETION_SEAL_PATH,
    PRODUCER_SOURCE_GIT_COMMIT,
    PROTOCOL_ID,
    PolicyVisionV3ValidationRepairContract,
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_bytes,
)
from causalcache.restoration_v2_2_policy_vision_v3_contract import (
    RestorationV22PolicyVisionV3Contract,
)
from scripts.run_restoration_v2_2_policy_vision_baseline import (
    _feature_record_from_evaluated_row,
)


ROOT = Path(__file__).resolve().parents[2]


class PolicyVisionV3ValidationRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = PolicyVisionV3ValidationRepairContract.load(
            ROOT / CANONICAL_CONFIG_PATH,
            repository_root=ROOT,
        )
        cls.rows = tuple(
            json.loads(line)
            for line in (
                cls.contract.artifact_directory / "state_scores.jsonl"
            ).read_text().splitlines()
        )

    def test_legacy_negative_control_and_exact_projection(self) -> None:
        producer = RestorationV22PolicyVisionV3Contract.load(
            ROOT / self.contract.data["producer"]["v3_contract"]["path"],
            repository_root=ROOT,
            validate_bound_sources=False,
        )
        work_items, _ = load_identity_witness(producer)
        legacy = tuple(_feature_record_from_evaluated_row(row) for row in self.rows)
        with self.assertRaisesRegex(ValueError, "row-to-worker provenance drifted"):
            validate_feature_worker_provenance(legacy, work_items)
        repaired = feature_records_from_evaluated_rows(
            self.rows,
            projection_contract=self.contract.data["projection_contract"],
        )
        validate_feature_worker_provenance(repaired, work_items)
        self.assertEqual(len(repaired), 15)
        self.assertEqual(
            {tuple(record["state"]) for record in repaired},
            {FEATURE_STATE_KEYS},
        )
        self.assertEqual(
            {frozenset(row["state"]) for row in self.rows},
            {EVALUATED_STATE_KEYS},
        )

    def test_projection_rejects_missing_extra_and_value_drift(self) -> None:
        state = dict(self.rows[0]["state"])
        expected = project_evaluated_state_to_feature_state(state)
        self.assertEqual(tuple(expected), FEATURE_STATE_KEYS)
        for mutation in (
            lambda value: value.pop("budget_event_capacity"),
            lambda value: value.__setitem__("extra", 1),
            lambda value: value.__setitem__("decision_step_id", 5),
            lambda value: value.__setitem__("candidate_event_step_ids", [1, 2]),
            lambda value: value.__setitem__("index", True),
            lambda value: value.__setitem__("state_id", "wrong"),
        ):
            changed = dict(state)
            mutation(changed)
            with self.assertRaisesRegex(ValueError, "evaluated-state"):
                project_evaluated_state_to_feature_state(changed)

    def test_mock_label_replay_matches_all_exact_three_bytes_without_gpu_path(self) -> None:
        expected = _artifact_snapshot(self.contract)
        with tempfile.TemporaryDirectory() as temporary:
            labels = Path(temporary) / "labels.tar"
            labels.write_bytes(b"mock labels")
            with (
                mock.patch(
                    "causalcache.restoration_v2_2_policy_vision_v3_validation_repair._validate_labels_archive"
                ),
                mock.patch(
                    "causalcache.restoration_v2_2_policy_vision_v3_validation_repair._expected_files_from_features",
                    return_value=expected,
                ) as rebuild,
                mock.patch(
                    "scripts.run_restoration_v2_2_policy_vision_baseline._run_feature_workers",
                    side_effect=AssertionError("GPU feature workers are forbidden"),
                ),
            ):
                replay = replay_exact_producer_artifact(
                    contract=self.contract,
                    labels_archive=labels,
                )
        self.assertEqual(replay.rebuilt_files, expected)
        self.assertEqual(replay.producer_files_before, replay.producer_files_after)
        self.assertEqual(replay.feature_record_count, 15)
        self.assertEqual(replay.candidate_score_count, 60)
        self.assertEqual(rebuild.call_count, 1)

    def test_attempt_ledger_is_exclusive_fsynced_and_mode_0600(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempt.json"
            record = {"status": "CLAIMED", "value": 1}
            payload, digest = claim_attempt_ledger(path, record)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(len(digest), 64)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                claim_attempt_ledger(path, record)
            symlink = Path(temporary) / "symlink.json"
            symlink.symlink_to(path.name)
            with self.assertRaises(FileExistsError):
                claim_attempt_ledger(symlink, record)

    def test_completion_seal_is_exclusive_fsynced_and_mode_0600(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "completion.json"
            record = {"status": "SEALED", "value": 1}
            payload, digest = claim_completion_seal(path, record)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(len(digest), 64)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                claim_completion_seal(path, record)

    def test_existing_repair_ledger_is_bound_to_summary_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempt.json"
            record = {"status": "CLAIMED", "value": 1}
            payload, digest = claim_attempt_ledger(path, record)
            summary = {
                "formal_attempt_claim": {"record": record, "sha256": digest}
            }
            with mock.patch(
                "causalcache.restoration_v2_2_policy_vision_v3_validation_repair.FORMAL_ATTEMPT_LEDGER_PATH",
                str(path),
            ):
                self.assertEqual(
                    validate_repair_attempt_ledger(path, summary=summary),
                    payload,
                )
                path.write_bytes(payload + b"x")
                with self.assertRaisesRegex(ValueError, "attempt ledger"):
                    validate_repair_attempt_ledger(path, summary=summary)

    def test_atomic_sibling_output_rejects_existing_staging_and_bad_inventory(self) -> None:
        files = {"README.md": b"readme", "summary.json": b"{}\n"}
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output = parent / "result"
            staging = parent / ".result.staging"
            prepublish_observations = []

            def pre_publish_check() -> None:
                prepublish_observations.append(
                    (
                        staging.is_dir(),
                        output.exists(),
                        sorted(path.name for path in staging.iterdir()),
                    )
                )

            write_atomic_result(
                output,
                files,
                pre_publish_check=pre_publish_check,
            )
            self.assertEqual(
                prepublish_observations,
                [(True, False, ["README.md", "summary.json"])],
            )
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["README.md", "summary.json"],
            )
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_atomic_result(output, files, pre_publish_check=lambda: None)

            other = parent / "other"
            staging = parent / ".other.staging"
            staging.mkdir()
            with self.assertRaisesRegex(FileExistsError, "staging"):
                write_atomic_result(other, files, pre_publish_check=lambda: None)
            shutil.rmtree(staging)
            with self.assertRaisesRegex(ValueError, "inventory"):
                write_atomic_result(
                    other,
                    {**files, "extra": b"x"},
                    pre_publish_check=lambda: None,
                )

    def test_no_nvidia_boundary_and_zero_operation_contract(self) -> None:
        with mock.patch.object(Path, "glob", return_value=[]):
            runtime = validate_no_nvidia_devices()
        self.assertEqual(runtime, expected_no_nvidia_runtime())
        self.assertTrue(all(value == 0 for value in ZERO_CPU_REPAIR_OPERATIONS.values()))
        fake_device = mock.Mock()
        fake_device.__str__ = mock.Mock(return_value="/dev/nvidia0")
        fake_device.exists.return_value = True
        fake_device.is_symlink.return_value = False
        with mock.patch.object(Path, "glob", return_value=[fake_device]):
            with self.assertRaisesRegex(ValueError, "NVIDIA"):
                validate_no_nvidia_devices()
        with mock.patch.dict(sys.modules, {"torch": mock.Mock()}):
            with self.assertRaisesRegex(ValueError, "forbidden runtime module"):
                validate_no_nvidia_devices()

    def test_audit_validation_rebuilds_from_fresh_replay_source_and_ledgers(
        self,
    ) -> None:
        artifact = _artifact_snapshot(self.contract)
        replay = ReplayResult(
            rebuilt_files=artifact,
            producer_files_before=artifact,
            producer_files_after=artifact,
            feature_record_count=15,
            candidate_score_count=60,
        )
        validation_commit = "b" * 40
        validation_source = {
            "producer_source_git_commit": PRODUCER_SOURCE_GIT_COMMIT,
            "validation_source_git_commit": validation_commit,
            "source_diff_baseline_git_commit": ARTIFACT_GIT_COMMIT,
            "formal_source_changed_paths_exact": self.contract.data[
                "validation_source"
            ]["formal_source_changed_paths_exact"],
            "producer_python_source_closure": self.contract.data["producer"][
                "formal_python_source_closure"
            ],
            "validation_python_source_closure": {
                "rule": (
                    "all_tracked_python_under_code_causalcache_and_code_scripts_"
                    "at_source_commit"
                ),
                "path_count": 159,
                "inventory_sha256": "c" * 64,
            },
        }
        producer_ledger = b"producer-ledger"
        started = "2026-07-16T00:00:00+00:00"
        no_nvidia = expected_no_nvidia_runtime()
        runtime_identity = build_runtime_identity(no_nvidia)
        claim = {
            "schema_version": "1.0.0",
            "status": "CLAIMED_POLICY_VISION_V3_VALIDATION_REPAIR_V1_ATTEMPT",
            "protocol_id": PROTOCOL_ID,
            "producer_source_git_commit": PRODUCER_SOURCE_GIT_COMMIT,
            "artifact_git_commit": ARTIFACT_GIT_COMMIT,
            "validation_source_git_commit": validation_commit,
            "contract_sha256": self.contract.sha256,
            "canonical_result_directory": self.contract.data["output_contract"][
                "canonical_result_directory"
            ],
            "formal_attempt_ledger_path": FORMAL_ATTEMPT_LEDGER_PATH,
            "formal_completion_seal_path": FORMAL_COMPLETION_SEAL_PATH,
            "producer_gpu_attempt_ledger_sha256": sha256_bytes(producer_ledger),
            "argv": ["repair-runner", "validate"],
            "started_at_utc": started,
            "runtime_identity": runtime_identity,
        }
        repair_payload = pretty_json_bytes(claim)
        repair_ledger = RepairLedgerSnapshot(
            payload=repair_payload,
            record=claim,
            sha256=sha256_bytes(repair_payload),
        )
        finished = "2026-07-16T00:00:01+00:00"
        runtime = build_runtime_record(
            runtime_identity=runtime_identity,
            argv=claim["argv"],
            started_at_utc=started,
            finished_at_utc=finished,
        )
        files = build_audit_files(
            contract=self.contract,
            replay=replay,
            validation_source=validation_source,
            attempt_claim=claim,
            attempt_claim_sha256=repair_ledger.sha256,
            producer_ledger_sha256=sha256_bytes(producer_ledger),
            runtime_record=runtime,
        )
        completion_record = build_completion_seal_record(
            files=files,
            repair_ledger=repair_ledger,
            finished_at_utc=finished,
        )
        completion_payload = pretty_json_bytes(completion_record)
        completion_seal = CompletionSealSnapshot(
            payload=completion_payload,
            record=completion_record,
            sha256=sha256_bytes(completion_payload),
        )
        common = {
            "contract": self.contract,
            "replay": replay,
            "validation_source": validation_source,
            "producer_ledger_payload": producer_ledger,
            "repair_ledger": repair_ledger,
            "completion_seal": completion_seal,
            "actual_no_nvidia_runtime": no_nvidia,
        }
        self.assertEqual(
            validate_audit_against_replay(files, **common),
            files,
        )

        def source_drift(summary: dict) -> None:
            summary["source_execution"]["validation_source_git_commit"] = "d" * 40

        def replay_count_drift(summary: dict) -> None:
            summary["exact_byte_replay"]["feature_record_count"] = 14

        def producer_file_drift(summary: dict) -> None:
            summary["exact_byte_replay"]["producer_files"][0]["sha256"] = "0" * 64

        def runtime_drift(summary: dict) -> None:
            summary["runtime"]["device"] = "cuda"

        def hostname_drift(summary: dict) -> None:
            summary["runtime"]["hostname"] = "forged-host"

        def working_directory_drift(summary: dict) -> None:
            summary["runtime"]["working_directory"] = "/forged/cwd"

        def finished_at_drift(summary: dict) -> None:
            summary["runtime"]["finished_at_utc"] = (
                "2026-07-16T00:00:02+00:00"
            )

        def repair_drift(summary: dict) -> None:
            summary["repair"]["only_semantic_change"] = "forged_repair"

        def producer_ledger_drift(summary: dict) -> None:
            summary["producer_identity"][
                "producer_gpu_attempt_ledger_sha256"
            ] = "1" * 64

        for name, mutate in (
            ("source", source_drift),
            ("replay_count", replay_count_drift),
            ("producer_file", producer_file_drift),
            ("runtime", runtime_drift),
            ("hostname", hostname_drift),
            ("working_directory", working_directory_drift),
            ("finished_at", finished_at_drift),
            ("repair", repair_drift),
            ("producer_ledger", producer_ledger_drift),
        ):
            with self.subTest(name=name):
                summary = json.loads(files["summary.json"])
                mutate(summary)
                payload = dict(summary)
                payload.pop("validation_payload_sha256")
                summary["validation_payload_sha256"] = sha256_bytes(
                    canonical_json_bytes(payload)
                )
                tampered = {
                    "README.md": _audit_readme(summary),
                    "summary.json": pretty_json_bytes(summary),
                }
                with self.assertRaises(ValueError):
                    validate_audit_against_replay(tampered, **common)

        output_tamper = dict(files)
        output_tamper["README.md"] += b"tampered"
        with self.assertRaisesRegex(ValueError, "completion seal"):
            validate_audit_against_replay(output_tamper, **common)


if __name__ == "__main__":
    unittest.main()
