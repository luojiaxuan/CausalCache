"""Tests for the formal-58 transport-only repair overlay contract."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import subprocess
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from causalcache import gate_v1_formal_cache_transport_repair_contract as repair
from scripts.validate_gate_v1_formal_cache_transport_repair_contract import _parser


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / repair.CANONICAL_CONFIG_PATH
SOURCE_A_GIT_COMMIT = "4f8c01b026167d6e9429716a082f3abd7c0c1bc9"


class TransportRepairContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    @contextmanager
    def _source_a_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            source_a_root = Path(directory) / "source-a"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "worktree",
                    "add",
                    "--detach",
                    "--quiet",
                    str(source_a_root),
                    SOURCE_A_GIT_COMMIT,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            try:
                yield source_a_root
            finally:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(ROOT),
                        "worktree",
                        "remove",
                        "--force",
                        str(source_a_root),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

    def test_loads_frozen_full_parent_shaped_overlay(self) -> None:
        contract = repair.load_frozen_transport_repair_contract(
            CONFIG, repository_root=ROOT
        )
        self.assertEqual(contract.sha256, repair.FROZEN_CONFIG_SHA256)
        self.assertEqual(
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(), repair.FROZEN_CONFIG_SHA256
        )
        self.assertEqual(contract.data["protocol_id"], repair.PARENT_PROTOCOL_ID)
        self.assertEqual(contract.data["status"], repair.PARENT_SOURCE_STATUS)
        self.assertEqual(contract.inputs, contract.data["input_artifacts"])
        self.assertEqual(contract.firewall, contract.data["access_firewall"])
        self.assertEqual(contract.formats, contract.data["cache_formats"])
        self.assertEqual(contract.local_state, contract.data["local_first_state_machine"])
        self.assertEqual(contract.destination, contract.data["destination"])
        self.assertEqual(contract.runtime, contract.data["runtime_contract"])
        self.assertEqual(contract.operations, contract.data["source_only_operation_contract"])
        self.assertEqual(
            contract.execution_operations,
            contract.data["execution_expected_operation_contract"],
        )

    def test_marker_is_exact_and_only_input_leaf_is_corrected(self) -> None:
        marker = self.config["source_freeze"]["transport_repair"]
        self.assertEqual(marker, repair.REPAIR_MARKER)
        self.assertEqual(
            self.config["input_artifacts"]["expansion_derived_features"]["files"][2][
                "sha256"
            ],
            repair.CORRECTED_TRANSPORT_SHA256,
        )
        parent = repair.load_frozen_formal_cache_contract(
            repair.PARENT_CONFIG_PATH, repository_root=ROOT
        )
        expected = repair._expected_repair_data(parent.data)
        self.assertEqual(self.config, expected)

    def test_unapproved_delta_fails_closed(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["formal_geometry"]["trajectory_count"] = 57
        with self.assertRaisesRegex(ValueError, "transport-only overlay"):
            repair.validate_transport_repair_config(changed, repository_root=ROOT)

    def test_source_only_validator_cross_checks_failure_and_producer(self) -> None:
        with self._source_a_snapshot() as source_a_root:
            self.assertFalse((source_a_root / repair.RUNNER_FREEZE_B_PATH).exists())
            result = repair.validate_transport_repair_source_only_contract(
                repair.CANONICAL_CONFIG_PATH, repository_root=source_a_root
            )
        self.assertEqual(result["status"], repair.VALIDATION_STATUS)
        self.assertEqual(result["config_sha256"], repair.FROZEN_CONFIG_SHA256)
        self.assertEqual(
            result["corrected_transport_sha256"], repair.CORRECTED_TRANSPORT_SHA256
        )
        self.assertFalse(result["runner_freeze_b_present"])
        self.assertTrue(result["failed_v1_attempt_local_validation_deferred"])

    def test_source_validator_rejects_repair_runner_present(self) -> None:
        runner = ROOT / repair.RUNNER_FREEZE_B_PATH
        self.assertTrue(runner.is_file())
        with self.assertRaisesRegex(ValueError, "runner freeze B must remain absent"):
            repair.validate_transport_repair_source_only_contract(
                CONFIG, repository_root=ROOT
            )

    def test_failed_attempt_validator_securely_checks_claim_and_absence(self) -> None:
        contract = repair.load_frozen_transport_repair_contract(
            CONFIG, repository_root=ROOT
        )
        old_claim = contract.data["source_freeze"]["transport_repair_evidence"][
            "old_global_claim"
        ]
        original_reader = repair._regular_file_bytes
        original_hash = repair.sha256_bytes

        def fake_reader(path: Path, *, label: str, expected_mode: int | None = None):
            if label == "old global claim":
                self.assertEqual(expected_mode, old_claim["mode"])
                return b"x" * old_claim["size_bytes"], SimpleNamespace(
                    st_mode=stat.S_IFREG | old_claim["mode"]
                )
            return original_reader(path, label=label, expected_mode=expected_mode)

        def fake_hash(payload: bytes) -> str:
            if payload == b"x" * old_claim["size_bytes"]:
                return old_claim["sha256"]
            return original_hash(payload)

        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            (data_root / "experiments" / "causalcache").mkdir(parents=True)
            with (
                mock.patch.object(repair, "_regular_file_bytes", side_effect=fake_reader),
                mock.patch.object(repair, "sha256_bytes", side_effect=fake_hash),
            ):
                result = repair.validate_failed_v1_attempt(contract, data_root)
            self.assertEqual(result["status"], repair.FAILED_ATTEMPT_VALIDATION_STATUS)
            self.assertTrue(result["new_claim_permitted"])

    def test_failed_attempt_validator_rejects_successor(self) -> None:
        contract = repair.load_frozen_transport_repair_contract(
            CONFIG, repository_root=ROOT
        )
        old_claim = contract.data["source_freeze"]["transport_repair_evidence"][
            "old_global_claim"
        ]
        original_reader = repair._regular_file_bytes
        original_hash = repair.sha256_bytes

        def fake_reader(path: Path, *, label: str, expected_mode: int | None = None):
            if label == "old global claim":
                return b"x" * old_claim["size_bytes"], SimpleNamespace(
                    st_mode=stat.S_IFREG | old_claim["mode"]
                )
            return original_reader(path, label=label, expected_mode=expected_mode)

        def fake_hash(payload: bytes) -> str:
            if payload == b"x" * old_claim["size_bytes"]:
                return old_claim["sha256"]
            return original_hash(payload)

        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            old_directory = data_root / "experiments" / "causalcache"
            old_directory.mkdir(parents=True)
            (old_directory / ".gate-v1-formal58-cache-v1.feature.completion.json").write_text(
                "unexpected", encoding="utf-8"
            )
            with (
                mock.patch.object(repair, "_regular_file_bytes", side_effect=fake_reader),
                mock.patch.object(repair, "sha256_bytes", side_effect=fake_hash),
            ):
                with self.assertRaisesRegex(ValueError, "old successor feature_completion"):
                    repair.validate_failed_v1_attempt(contract, data_root)

    def test_cli_defaults_to_canonical_contract(self) -> None:
        args = _parser().parse_args([])
        self.assertEqual(args.contract, Path(repair.CANONICAL_CONFIG_PATH))


if __name__ == "__main__":
    unittest.main()
