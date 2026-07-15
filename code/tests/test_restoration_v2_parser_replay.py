from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from causalcache.restoration_v2_parser_replay import (
    EXPECTED_STATE_COUNT,
    SOURCE_PROTOCOL_ID,
    _expected_archive_members,
    audit_restoration_v2_parser_compatibility_archive,
)
from scripts.replay_restoration_v2_parser_compat import (
    CANONICAL_REMOTE,
    _canonical_sha256,
    _source_identity,
    _validate_golden_result,
    _validate_clean_pushed_main,
    _write_json_exclusive,
)


RUN_PREFIX = "synthetic-restoration-v2"
SOURCE_COMMIT = "a" * 40


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _recorded_output(index: int, *, all_compatible: bool = False) -> str:
    prefix = (
        "Action: Execute one action.\n<tool_call>\n"
        '{"action":"click","coordinate":[100,200]}'
    )
    if all_compatible or index < 40:
        return prefix
    if index == 40:
        return prefix + "\nAction: Again.\n<tool_call>\n{\"action\":\"wait\"}"
    if index == 41:
        return "Action: Click.\n<tool_call>\n{\"action\":\"click\",\"coordinate\":[1,2]"
    if index == 42:
        return prefix + "\n{\"action\":\"wait\"}"
    return prefix + "\n<observation>unexpected</observation>"


def _archive_payloads(
    *,
    official_outcome: str = "NO_GO_V2_SUBSTRATE",
    all_compatible: bool = False,
) -> tuple[dict[str, bytes], str]:
    states = [
        {
            "candidate_event_step_ids": [1, 2],
            "decision_step_id": 4 + (index % 3),
            "index": index,
            "role": "v2_label_train" if index < 30 else "v2_development",
            "state_id": f"trajectory-{index // 3:03d}:decision_step:{4 + (index % 3):03d}",
            "trajectory_id": f"trajectory-{index // 3:03d}",
        }
        for index in range(EXPECTED_STATE_COUNT)
    ]
    contract = {
        "git_commit": SOURCE_COMMIT,
        "authorization": {
            "confirm_state": "CONFIRM_LOCKED",
            "confirm_locked": True,
        },
        "states": states,
    }
    contract_sha256 = _canonical_sha256(contract)
    payloads: dict[str, bytes] = {
        f"{RUN_PREFIX}/run_manifest.json": _json_bytes(
            {
                "created_at_utc": "2026-07-15T00:00:00Z",
                "protocol_id": SOURCE_PROTOCOL_ID,
                "run_contract": contract,
                "run_contract_sha256": contract_sha256,
                "schema_version": "1.0.0",
                "status": "RESTORATION_V2_SUBSTRATE_SCREENING",
            }
        ),
        f"{RUN_PREFIX}/aggregate.json": _json_bytes(
            {
                "checks": {
                    "minimum_finite_logit_coverage": False,
                    "minimum_memory_sensitive_states": False,
                    "minimum_parse_coverage": False,
                    "minimum_repeat_canonical_action_agreement": False,
                    "minimum_screening_states": True,
                },
                "confirm_role_used": False,
                "failure_category_counts": {"PARSE_FAILURE": 45},
                "gate_contract": {
                    "fail_outcome": "NO_GO_V2_SUBSTRATE",
                    "minimum_finite_logit_coverage": 1.0,
                    "minimum_memory_sensitive_states": 8,
                    "minimum_parse_coverage": 0.99,
                    "minimum_repeat_canonical_action_agreement": 1.0,
                    "minimum_screening_states": 20,
                    "pass_outcome": "RUN_UNTOUCHED_RESTORATION_CONFIRM",
                },
                "gate_passed": False,
                "metrics": {
                    "fixed_state_denominator": 45,
                    "parse_success_count": 0,
                    "parse_coverage": 0.0,
                },
                "outcome": official_outcome,
                "protocol_id": SOURCE_PROTOCOL_ID,
                "run_contract_sha256": contract_sha256,
                "sample_mutation_performed": False,
                "schema_version": "1.0.0",
                "status": "COMPLETED_FIXED_45_STATE_SUBSTRATE_SCREENING",
                "top_up_performed": False,
            }
        ),
        f"{RUN_PREFIX}.log": b"synthetic immutable log\n",
    }
    for index, state_identity in enumerate(states):
        payloads[f"{RUN_PREFIX}/attempts/{index:03d}.json"] = _json_bytes(
            {
                "created_at_utc": "2026-07-15T00:00:00Z",
                "protocol_id": SOURCE_PROTOCOL_ID,
                "run_contract_sha256": contract_sha256,
                "schema_version": "1.0.0",
                "state": state_identity,
                "status": "ATTEMPT_STARTED_NO_RETRY",
            }
        )
        payloads[f"{RUN_PREFIX}/states/{index:03d}.json"] = _json_bytes(
            {
                "canonical_action": None,
                "distance_audits": {},
                "distances": {
                    "repeat_reference_kl": None,
                    "summary_reference_kl": None,
                },
                "duration_seconds": 1.0,
                "ended_at_utc": "2026-07-15T00:00:01Z",
                "failure": {
                    "category": "PARSE_FAILURE",
                    "exception_type": "GUIOwlV2GenerationParseError",
                    "message": "strict parse rejected output",
                    "stage": "reference_generation_1",
                },
                "finite_logit_distances": False,
                "message_shapes": {},
                "native_generations": [
                    {
                        "canonical_action": None,
                        "metadata": {
                            "do_sample": False,
                            "effective_visual_tokens": 100,
                            "full_logit_tensor_host_transfers": 0,
                            "generated_tokens": 64,
                            "image_count": 3,
                            "image_grid_thw": [[1, 10, 10]] * 3,
                            "latency_seconds": 1.0,
                            "max_new_tokens": 256,
                            "peak_gpu_memory_allocated_bytes": 1,
                            "peak_gpu_memory_reserved_bytes": 1,
                            "policy_visible_text_tokens": 10,
                            "prompt_input_tokens": 110,
                        },
                        "output_text": _recorded_output(
                            index,
                            all_compatible=all_compatible,
                        ),
                        "parse_error_message": "strict parse rejected output",
                        "parse_error_type": "ValueError",
                        "repeat_index": 1,
                    }
                ],
                "outcome": "FAILED_SUBSTRATE_STATE",
                "parse_success": False,
                "parse_success_count": 0,
                "protocol_id": SOURCE_PROTOCOL_ID,
                "repeat_canonical_action_agreement": False,
                "run_contract_sha256": contract_sha256,
                "schema_version": "1.0.0",
                "started_at_utc": "2026-07-15T00:00:00Z",
                "state": state_identity,
                "teacher_forwards": {},
            }
        )
    return payloads, contract_sha256


def _write_archive(
    path: Path,
    *,
    official_outcome: str = "NO_GO_V2_SUBSTRATE",
    reverse_first_two_files: bool = False,
    all_compatible: bool = False,
) -> tuple[str, str]:
    payloads, contract_sha256 = _archive_payloads(
        official_outcome=official_outcome,
        all_compatible=all_compatible,
    )
    members = list(_expected_archive_members(RUN_PREFIX))
    if reverse_first_two_files:
        members[1], members[2] = members[2], members[1]
    with tarfile.open(path, mode="w:gz") as handle:
        for name, kind in members:
            info = tarfile.TarInfo(name)
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            if kind == "directory":
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                handle.addfile(info)
                continue
            payload = payloads[name]
            info.size = len(payload)
            info.mode = 0o644
            handle.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest(), contract_sha256


class RestorationV2ParserReplayTest(unittest.TestCase):
    def test_committed_formal_result_matches_golden_and_audit_commit(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        result_path = (
            repository_root
            / "data/results/restoration_v2_parser_compatibility/formal_result.json"
        )
        golden_path = (
            repository_root
            / "data/manifests/restoration_v2_parser_compatibility_golden.json"
        )
        result_payload = result_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(result_payload).hexdigest(),
            "78272ee54c1234244239eff080f03d7d97b5973fa5d5cf26fddd77b95794a6e0",
        )
        result = json.loads(result_payload)
        golden = json.loads(golden_path.read_bytes())
        _validate_golden_result(result, golden)
        self.assertEqual(result["immutable_source"], golden["source_artifact"])
        self.assertEqual(
            result["audit_run"]["git_pre_and_post_identity"]["git_commit"],
            "fc3adf13d48bb016014f7efa62bd27c8a4d12f49",
        )
        self.assertTrue(result["audit_run"]["pre_and_post_validation_matched"])
        self.assertNotIn("output_text", result_payload.decode("utf-8"))

    def test_immutable_replay_reports_the_expected_adapter_only_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "trace.tar.gz"
            archive_sha256, contract_sha256 = _write_archive(archive)
            with mock.patch(
                "causalcache.restoration_v2_parser_replay.tarfile.open",
                wraps=tarfile.open,
            ) as tar_open:
                result = audit_restoration_v2_parser_compatibility_archive(
                    archive_path=archive,
                    expected_archive_sha256=archive_sha256,
                    run_prefix=RUN_PREFIX,
                    expected_run_contract_sha256=contract_sha256,
                    expected_source_run_git_commit=SOURCE_COMMIT,
                )
            self.assertIsInstance(tar_open.call_args.kwargs["fileobj"], io.BytesIO)
            self.assertNotIn("name", tar_open.call_args.kwargs)

        self.assertEqual(result["outcome"], "NO_GO_ADAPTER_ONLY")
        self.assertEqual(
            result["metrics"],
            {
                "fixed_state_denominator": 45,
                "strict_parse_success_count": 0,
                "single_action_outputs_with_first_balanced_json_count": 43,
                "single_action_outputs_with_canonical_first_payload_count": 43,
                "conservative_compatibility_parse_success_count": 40,
                "conservative_compatibility_parse_coverage": 40 / 45,
                "minimum_parse_coverage": 0.99,
                "required_parse_success_count": 45,
                "clean_eof_compatibility_count": 40,
                "model_emitted_canonical_closer_count": 0,
                "compatibility_acceptance_requiring_suffix_recovery_count": 0,
                "adapter_gate_passed": False,
            },
        )
        self.assertEqual(
            result["inventory"]["rejection_code_counts"],
            {
                "EXTRA_OBSERVATION": 2,
                "MULTIPLE_ACTIONS": 1,
                "SECOND_JSON": 1,
                "TRUNCATED_OR_INVALID_JSON": 1,
            },
        )
        self.assertNotIn("output_text", json.dumps(result))
        self.assertTrue(all(len(record["raw_output_sha256"]) == 64 for record in result["records"]))
        self.assertFalse(result["negative_declarations"]["policy_generation_executed"])

    def test_archive_sha_inventory_and_official_result_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "trace.tar.gz"
            archive_sha256, contract_sha256 = _write_archive(archive)
            common = {
                "archive_path": archive,
                "run_prefix": RUN_PREFIX,
                "expected_run_contract_sha256": contract_sha256,
                "expected_source_run_git_commit": SOURCE_COMMIT,
            }
            with self.assertRaisesRegex(ValueError, "archive SHA256"):
                audit_restoration_v2_parser_compatibility_archive(
                    expected_archive_sha256="0" * 64,
                    **common,
                )

            reordered = root / "reordered.tar.gz"
            reordered_sha256, _ = _write_archive(
                reordered,
                reverse_first_two_files=True,
            )
            with self.assertRaisesRegex(ValueError, "member order"):
                audit_restoration_v2_parser_compatibility_archive(
                    archive_path=reordered,
                    expected_archive_sha256=reordered_sha256,
                    run_prefix=RUN_PREFIX,
                    expected_run_contract_sha256=contract_sha256,
                    expected_source_run_git_commit=SOURCE_COMMIT,
                )

            drifted = root / "drifted.tar.gz"
            drifted_sha256, _ = _write_archive(drifted, official_outcome="GO")
            with self.assertRaisesRegex(ValueError, "official aggregate"):
                audit_restoration_v2_parser_compatibility_archive(
                    archive_path=drifted,
                    expected_archive_sha256=drifted_sha256,
                    run_prefix=RUN_PREFIX,
                    expected_run_contract_sha256=contract_sha256,
                    expected_source_run_git_commit=SOURCE_COMMIT,
                )

            self.assertEqual(len(archive_sha256), 64)

    @mock.patch("scripts.replay_restoration_v2_parser_compat._git")
    def test_cli_requires_clean_pushed_canonical_main(self, git: mock.Mock) -> None:
        remote_record = f"{SOURCE_COMMIT}\trefs/heads/main"
        git.side_effect = [
            SOURCE_COMMIT,
            SOURCE_COMMIT,
            "main",
            CANONICAL_REMOTE,
            "",
            remote_record,
        ]
        identity = _validate_clean_pushed_main(
            Path("/repo"),
            expected_git_commit=SOURCE_COMMIT,
        )
        self.assertEqual(identity["git_commit"], SOURCE_COMMIT)

        git.side_effect = [
            SOURCE_COMMIT,
            SOURCE_COMMIT,
            "main",
            CANONICAL_REMOTE,
            "?? x",
            remote_record,
        ]
        with self.assertRaisesRegex(ValueError, "clean pushed canonical remote main"):
            _validate_clean_pushed_main(
                Path("/repo"),
                expected_git_commit=SOURCE_COMMIT,
            )

        git.side_effect = [
            SOURCE_COMMIT,
            SOURCE_COMMIT,
            "main",
            CANONICAL_REMOTE,
            "",
            f"{'b' * 40}\trefs/heads/main",
        ]
        with self.assertRaisesRegex(ValueError, "clean pushed canonical remote main"):
            _validate_clean_pushed_main(
                Path("/repo"),
                expected_git_commit=SOURCE_COMMIT,
            )

    def test_outcome_is_derived_when_adapter_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "trace.tar.gz"
            archive_sha256, contract_sha256 = _write_archive(
                archive,
                all_compatible=True,
            )
            result = audit_restoration_v2_parser_compatibility_archive(
                archive_path=archive,
                expected_archive_sha256=archive_sha256,
                run_prefix=RUN_PREFIX,
                expected_run_contract_sha256=contract_sha256,
                expected_source_run_git_commit=SOURCE_COMMIT,
            )
        self.assertEqual(result["outcome"], "PASSED_ADAPTER_GATE")
        self.assertTrue(result["metrics"]["adapter_gate_passed"])
        self.assertEqual(
            result["metrics"]["conservative_compatibility_parse_success_count"],
            45,
        )

    def test_golden_validator_binds_aggregate_records_and_rejections(self) -> None:
        records = [
            {
                "state_index": 7,
                "state_id": "state-007",
                "raw_output_sha256": "c" * 64,
                "rejection_code": "SECOND_JSON",
            }
        ]
        result = {
            "outcome": "NO_GO_ADAPTER_ONLY",
            "metrics": {"accepted": 0},
            "inventory": {"SECOND_JSON": 1},
            "records": records,
        }
        golden = {
            "expected_outcome": result["outcome"],
            "expected_metrics": result["metrics"],
            "expected_inventory": result["inventory"],
            "expected_records_canonical_sha256": _canonical_sha256(records),
            "expected_rejections": [records[0]],
        }
        _validate_golden_result(result, golden)
        mutated = dict(result)
        mutated["records"] = [dict(records[0], rejection_code="EXTRA_OBSERVATION")]
        with self.assertRaisesRegex(ValueError, "classifications drifted"):
            _validate_golden_result(mutated, golden)

    @mock.patch("scripts.replay_restoration_v2_parser_compat._git_blob")
    def test_source_identity_requires_the_exact_committed_blob(
        self,
        git_blob: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "code" / "source.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"committed source\n")
            git_blob.return_value = b"different source\n"
            with self.assertRaisesRegex(ValueError, "differs from committed Git blob"):
                _source_identity(
                    root,
                    "code/source.py",
                    git_commit=SOURCE_COMMIT,
                )
            git_blob.return_value = source.read_bytes()
            identity = _source_identity(
                root,
                "code/source.py",
                git_commit=SOURCE_COMMIT,
            )
        self.assertEqual(identity["sha256"], hashlib.sha256(b"committed source\n").hexdigest())

    def test_result_writer_is_exclusive_and_rejects_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "summary.json"
            _write_json_exclusive(output, {"ok": True})
            self.assertEqual(json.loads(output.read_text()), {"ok": True})
            with self.assertRaises(FileExistsError):
                _write_json_exclusive(output, {"ok": False})
            with self.assertRaises(ValueError):
                _write_json_exclusive(
                    Path(temporary_directory) / "nan.json",
                    {"value": float("nan")},
                )


if __name__ == "__main__":
    unittest.main()
