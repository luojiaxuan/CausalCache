from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from causalcache.restoration_v2_2_policy_vision_v3_validation_repair import (
    run_formal_validation_repair,
)
from causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract import (
    CANONICAL_CONFIG_PATH,
    PolicyVisionV3ValidationRepairContract,
)
from scripts import (
    run_restoration_v2_2_policy_vision_v3_validation_repair_v1 as runner,
)
from scripts import (
    validate_restoration_v2_2_policy_vision_v3_validation_repair_v1_contract
    as source_validator,
)


ROOT = Path(__file__).resolve().parents[2]


class _FakeContract:
    def __init__(self, root: Path) -> None:
        self.data = {
            "output_contract": {"run_status": "RUN_OK"},
            "producer": {"producer_source_git_commit": "a" * 40},
        }
        self.artifact_directory = root / "artifact"

    def validate_formal_source(self, *args, **kwargs):
        return {"validation_source_git_commit": args[0]}

    def validate_clean_pushed_descendant(self, *args, **kwargs):
        return {"validation_source_git_commit": args[0]}


class PolicyVisionV3ValidationRepairRunnerTest(unittest.TestCase):
    def test_parser_has_no_gpu_model_or_feature_inputs(self) -> None:
        destinations = {action.dest for action in runner._parser()._actions}
        self.assertTrue(
            {
                "mode",
                "repository_root",
                "contract",
                "labels_archive",
                "producer_attempt_ledger",
                "validation_source_git_commit",
                "output_dir",
                "attempt_ledger",
                "completion_seal",
            }.issubset(destinations)
        )
        self.assertTrue(
            destinations.isdisjoint(
                {
                    "model_dir",
                    "derived_root",
                    "expected_gpu_uuid",
                    "device",
                    "host_evidence",
                }
            )
        )

    def test_run_mode_delegates_with_full_argv_and_reports_zero_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = _FakeContract(root)
            argv = [
                "run",
                "--repository-root",
                str(root),
                "--contract",
                str(root / "contract.json"),
                "--labels-archive",
                str(root / "labels.tar"),
                "--producer-attempt-ledger",
                str(root / "producer-ledger.json"),
                "--validation-source-git-commit",
                "b" * 40,
                "--output-dir",
                str(root / "output"),
                "--attempt-ledger",
                str(root / "attempt.json"),
                "--completion-seal",
                str(root / "completion.json"),
            ]
            files = {"README.md": b"r", "summary.json": b"{}\n"}
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    runner.PolicyVisionV3ValidationRepairContract,
                    "load",
                    return_value=fake,
                ),
                mock.patch.object(
                    runner,
                    "run_formal_validation_repair",
                    return_value=files,
                ) as formal,
                mock.patch("sys.stdout", stdout),
            ):
                runner.main(argv)
        call = formal.call_args.kwargs
        self.assertEqual(call["validation_source_git_commit"], "b" * 40)
        self.assertEqual(
            call["completion_seal"],
            (root / "completion.json").resolve(),
        )
        self.assertEqual(call["argv"][1:], argv)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["gpu_or_model_operation_count"], 0)

    def test_validate_mode_requires_and_verifies_existing_attempt_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = _FakeContract(root)
            argv = [
                "validate",
                "--repository-root",
                str(root),
                "--contract",
                str(root / "contract.json"),
                "--labels-archive",
                str(root / "labels.tar"),
                "--producer-attempt-ledger",
                str(root / "producer-ledger.json"),
                "--validation-source-git-commit",
                "b" * 40,
                "--output-dir",
                str(root / "output"),
                "--attempt-ledger",
                str(root / "attempt.json"),
                "--completion-seal",
                str(root / "completion.json"),
            ]
            files = {
                "README.md": b"r",
                "summary.json": b"{}\n",
            }
            replay = SimpleNamespace(
                rebuilt_files={"README.md": b"x"},
                producer_files_before={"README.md": b"x"},
            )
            repair_ledger = SimpleNamespace(
                payload=b"repair",
                record={"status": "CLAIMED"},
                sha256="f" * 64,
            )
            completion_seal = SimpleNamespace(
                payload=b"completion",
                record={"status": "SEALED"},
                sha256="e" * 64,
            )
            no_nvidia = {"nvidia_device_node_count": 0}
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    runner.PolicyVisionV3ValidationRepairContract,
                    "load",
                    return_value=fake,
                ),
                mock.patch.object(
                    runner,
                    "validate_no_nvidia_devices",
                    return_value=no_nvidia,
                ) as validate_no_nvidia,
                mock.patch.object(
                    runner,
                    "validate_producer_attempt_ledger",
                    return_value=b"producer",
                ),
                mock.patch.object(
                    runner,
                    "read_repair_attempt_ledger",
                    return_value=repair_ledger,
                ) as read_ledger,
                mock.patch.object(
                    runner,
                    "read_completion_seal",
                    return_value=completion_seal,
                ) as read_seal,
                mock.patch.object(
                    runner,
                    "replay_exact_producer_artifact",
                    return_value=replay,
                ),
                mock.patch.object(
                    runner,
                    "validate_existing_audit_result",
                    return_value=files,
                ) as validate_audit,
                mock.patch("sys.stdout", stdout),
            ):
                runner.main(argv)
        self.assertEqual(read_ledger.call_count, 2)
        self.assertEqual(read_seal.call_count, 2)
        self.assertEqual(
            read_ledger.call_args.args[0],
            (root / "attempt.json").resolve(),
        )
        self.assertEqual(
            read_seal.call_args.args[0],
            (root / "completion.json").resolve(),
        )
        self.assertEqual(validate_no_nvidia.call_count, 2)
        audit_call = validate_audit.call_args.kwargs
        self.assertIs(audit_call["replay"], replay)
        self.assertEqual(
            audit_call["validation_source"],
            {"validation_source_git_commit": "b" * 40},
        )
        self.assertEqual(audit_call["producer_ledger_payload"], b"producer")
        self.assertIs(audit_call["repair_ledger"], repair_ledger)
        self.assertIs(audit_call["completion_seal"], completion_seal)
        self.assertIs(audit_call["actual_no_nvidia_runtime"], no_nvidia)
        self.assertEqual(
            json.loads(stdout.getvalue())["gpu_or_model_operation_count"], 0
        )

    def test_formal_core_rejects_noncanonical_output_and_ledger(self) -> None:
        contract = PolicyVisionV3ValidationRepairContract.load(
            ROOT / CANONICAL_CONFIG_PATH,
            repository_root=ROOT,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "contract": contract,
                "labels_archive": root / "labels.tar",
                "producer_attempt_ledger": root / "producer.json",
                "validation_source_git_commit": "b" * 40,
                "argv": ["runner"],
            }
            with self.assertRaisesRegex(ValueError, "canonical sibling"):
                run_formal_validation_repair(
                    **common,
                    attempt_ledger=root / "attempt.json",
                    completion_seal=root / "completion.json",
                    output_dir=root / "wrong-output",
                )
            with self.assertRaisesRegex(ValueError, "ledger path"):
                run_formal_validation_repair(
                    **common,
                    attempt_ledger=root / "attempt.json",
                    completion_seal=root / "completion.json",
                    output_dir=contract.output_directory,
                )

    def test_source_validator_requires_clean_pushed_source_contract(self) -> None:
        result = {"status": "PASS"}
        stdout = io.StringIO()
        with (
            mock.patch.object(
                source_validator,
                "validate_contract",
                return_value=result,
            ) as validate,
            mock.patch("sys.stdout", stdout),
        ):
            source_validator.main(
                [
                    "--repository-root",
                    str(ROOT),
                    "--contract",
                    str(ROOT / CANONICAL_CONFIG_PATH),
                    "--validation-source-git-commit",
                    "b" * 40,
                ]
            )
        self.assertTrue(validate.call_args.kwargs["require_clean_pushed_main"])
        self.assertTrue(validate.call_args.kwargs["require_outputs_absent"])
        self.assertEqual(json.loads(stdout.getvalue()), result)


if __name__ == "__main__":
    unittest.main()
