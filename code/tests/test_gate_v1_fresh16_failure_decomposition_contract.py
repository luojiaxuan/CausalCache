from __future__ import annotations

import copy
import json
import os
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from causalcache import gate_v1_fresh16_failure_decomposition_contract as contract_module
from causalcache.gate_v1_fresh16_failure_decomposition_contract import (
    ALLOWED_PARENT_HF_API_METHODS,
    CANONICAL_CONFIG_PATH,
    CHILD_REPO,
    CHILD_TAG,
    CHILD_TARGETS,
    FROZEN_CONFIG_SHA256,
    ORDERED_LOCAL_STATES,
    PARENT_DOWNLOAD_PATHS,
    PARENT_RESULT_COMMIT,
    PARENT_REPORT_COMMIT,
    PROTOCOL_ID,
    REQUIRED_SOURCE_A_PATHS,
    RUNNER_FREEZE_B_PATH,
    RUNNER_FREEZE_STATUS,
    SOURCE_VALIDATION_STATUS,
    canonical_json_bytes,
    load_frozen_failure_decomposition_contract,
    load_runner_freeze,
    materialize_runner_freeze,
    runner_freeze_bytes,
    sha256_bytes,
    validate_contract_data,
    validate_execution_b_source,
    validate_source_a,
)
from scripts import manage_gate_v1_fresh16_failure_decomposition as manager


ROOT = Path(__file__).resolve().parents[2]
SOURCE_A = "1" * 40
EXECUTION_B = "2" * 40


class FailureDecompositionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_frozen_failure_decomposition_contract(
            repository_root=ROOT
        )

    def _source_validation(self) -> dict:
        source_inventory = [
            {
                "path": path,
                "sha256": f"{index + 1:064x}",
                "size_bytes": index + 1,
            }
            for index, path in enumerate(REQUIRED_SOURCE_A_PATHS)
        ]
        loaded = [
            {
                "module": "causalcache.gate_v1_fresh16_failure_decomposition_contract",
                "path": REQUIRED_SOURCE_A_PATHS[1],
                "sha256": "f" * 64,
                "size_bytes": 17,
            }
        ]
        return {
            "schema_version": "1.0.0",
            "protocol_id": PROTOCOL_ID,
            "status": SOURCE_VALIDATION_STATUS,
            "contract_sha256": self.contract.sha256,
            "source_a": {
                "git_commit": SOURCE_A,
                "origin_main_git_commit": SOURCE_A,
                "branch": "main",
                "origin_url": "https://github.com/luojiaxuan/CausalCache.git",
                "source_inventory": source_inventory,
                "source_inventory_sha256": sha256_bytes(
                    canonical_json_bytes(source_inventory)
                ),
                "loaded_module_inventory": loaded,
                "loaded_module_inventory_sha256": sha256_bytes(
                    canonical_json_bytes(loaded)
                ),
            },
        }

    def test_canonical_contract_identity_and_parent_lineage(self) -> None:
        self.assertEqual(self.contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(self.contract.data["protocol_id"], PROTOCOL_ID)
        parent = self.contract.parent
        self.assertEqual(parent["result_git_commit"], PARENT_RESULT_COMMIT)
        self.assertEqual(
            parent["source_a_git_commit"],
            "f0dd53b9a0249f259a833b4d8ad3ff26096a0ed1",
        )
        self.assertEqual(
            parent["execution_b_git_commit"],
            "ce523ff54ebfdc19a9c2bd49ad21548f0934a634",
        )
        self.assertEqual(parent["hf"]["report_commit"], PARENT_REPORT_COMMIT)
        self.assertFalse(parent["verdict"]["go_selector"])
        self.assertFalse(parent["verdict"]["go_set_conditioning_primary"])
        self.assertFalse(parent["verdict"]["reclassification_allowed"])

    def test_selective_three_file_parent_reader_is_exact(self) -> None:
        inputs = self.contract.data["input_contract"]
        self.assertEqual(
            tuple(item["path"] for item in inputs["exact_force_download_targets"]),
            PARENT_DOWNLOAD_PATHS,
        )
        self.assertEqual(
            tuple(inputs["allowed_parent_hf_api_methods"]),
            ALLOWED_PARENT_HF_API_METHODS,
        )
        self.assertEqual(inputs["parent_remote_mutation_call_count"], 0)
        self.assertTrue(inputs["selective_reader_only"])

    def test_child_destination_and_local_state_are_minimal(self) -> None:
        destination = self.contract.destination
        self.assertEqual(destination["repo"], CHILD_REPO)
        self.assertEqual(destination["tag"], CHILD_TAG)
        self.assertTrue(destination["single_exact_report_commit"])
        self.assertEqual(
            tuple(self.contract.data["output_contract"]["exact_targets"]),
            CHILD_TARGETS,
        )
        self.assertEqual(
            tuple(
                self.contract.data["local_first_state_machine"]["ordered_states"]
            ),
            ORDERED_LOCAL_STATES,
        )

    def test_source_only_contract_is_all_zero_and_all_locked(self) -> None:
        operations = self.contract.data["source_only_operation_contract"]
        authorizations = self.contract.data["authorization"]
        self.assertTrue(operations)
        self.assertTrue(authorizations)
        self.assertTrue(all(value == 0 for value in operations.values()))
        self.assertTrue(all(value is False for value in authorizations.values()))
        self.assertEqual(self.contract.data["runtime_contract"]["gpu_count"], 0)
        self.assertFalse(
            self.contract.data["runtime_contract"]["gpu_preflight_required"]
        )

    def test_analysis_thresholds_and_gap_identity_are_frozen(self) -> None:
        analysis = self.contract.data["analysis_contract"]
        self.assertEqual(analysis["budget"], 2)
        self.assertEqual(
            analysis["decomposition_identity"],
            "exact_minus_student_equals_search_plus_distillation",
        )
        self.assertEqual(
            analysis["normalization"]["eligible_only_when_strictly_greater_than"],
            1e-12,
        )
        self.assertEqual(
            analysis["interaction_mass"][
                "historical_train_only_tertile_cutpoints_by_n"
            ]["4"],
            [0.19346783303918597, 0.21532511632530205],
        )
        self.assertTrue(analysis["diagnostic_only"])

    def test_routing_thresholds_and_one_rescue_boundary_are_frozen(self) -> None:
        routing = self.contract.data["routing_contract"]
        self.assertEqual(
            routing["method_aliases"]["J"],
            {
                "budget": 2,
                "definition": "oracle_budget_conditioned_independent",
                "event_score": (
                    "0.5_times_empty_marginal_plus_mean_of_other_singleton_base_"
                    "conditional_marginals"
                ),
                "strictly_positive_score_required": True,
                "tie_break": "lower_event_step_id",
                "top_b": True,
            },
        )
        self.assertEqual(
            routing["search_pass"]["normalized_recovery_G_over_E_minimum"],
            0.95,
        )
        self.assertEqual(
            routing["oracle_set_headroom_pass"][
                "normalized_mean_delta_G_minus_J_minimum"
            ],
            0.02,
        )
        self.assertEqual(
            routing["material_student_gap"][
                "raw_utility_C_over_G_maximum_exclusive"
            ],
            0.95,
        )
        self.assertEqual(routing["bootstrap"]["seed"], 271828)
        final = routing["final_routing"]
        self.assertEqual(final["pass_status"], "ONE_V2_CONDITIONAL_RESCUE")
        self.assertEqual(
            final["all_required"],
            ["search_pass", "oracle_set_headroom_pass", "material_student_gap"],
        )
        self.assertTrue(final["fresh16_already_consumed"])
        self.assertFalse(final["fresh16_may_be_used_as_v2_holdout"])
        self.assertFalse(final["parent_v1_verdict_may_change"])

    def test_routing_rejects_threshold_or_holdout_drift(self) -> None:
        changed = copy.deepcopy(self.contract.data)
        changed["routing_contract"]["search_pass"][
            "normalized_recovery_G_over_E_minimum"
        ] = 0.94
        with self.assertRaisesRegex(ValueError, "search-pass"):
            validate_contract_data(changed)
        changed = copy.deepcopy(self.contract.data)
        changed["routing_contract"]["final_routing"][
            "fresh16_may_be_used_as_v2_holdout"
        ] = True
        with self.assertRaisesRegex(ValueError, "one-rescue"):
            validate_contract_data(changed)

    def test_contract_rejects_verdict_or_input_expansion(self) -> None:
        changed = copy.deepcopy(self.contract.data)
        changed["parent_primary"]["verdict"]["go_selector"] = True
        with self.assertRaisesRegex(ValueError, "verdict"):
            validate_contract_data(changed)
        changed = copy.deepcopy(self.contract.data)
        changed["input_contract"]["exact_force_download_targets"].append(
            {"path": "fresh16-eval/v1/learned/conditional-v1.json"}
        )
        with self.assertRaisesRegex(ValueError, "input"):
            validate_contract_data(changed)

    def test_contract_rejects_source_side_effect_or_runner_path_drift(self) -> None:
        changed = copy.deepcopy(self.contract.data)
        changed["source_only_operation_contract"]["network_call_count"] = 1
        with self.assertRaisesRegex(ValueError, "zero"):
            validate_contract_data(changed)
        changed = copy.deepcopy(self.contract.data)
        changed["source_freeze"]["execution_b_runner_freeze"]["path"] += ".other"
        with self.assertRaisesRegex(ValueError, "runner-freeze"):
            validate_contract_data(changed)

    def test_source_a_validator_is_local_read_only_and_requires_b_absent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = replace(self.contract, repository_root=root)
            calls: list[tuple[str, ...]] = []

            def fake_git(_root: Path, *arguments: str) -> bytes:
                calls.append(tuple(arguments))
                table = {
                    ("rev-parse", "HEAD"): f"{SOURCE_A}\n".encode(),
                    ("rev-parse", "origin/main"): f"{SOURCE_A}\n".encode(),
                    ("branch", "--show-current"): b"main\n",
                    (
                        "remote",
                        "get-url",
                        "origin",
                    ): b"https://github.com/luojiaxuan/CausalCache.git\n",
                    (
                        "status",
                        "--porcelain=v1",
                        "--untracked-files=all",
                    ): b"",
                }
                if arguments[:2] == ("merge-base", "--is-ancestor"):
                    return b""
                return table[tuple(arguments)]

            inventory = tuple(self._source_validation()["source_a"]["source_inventory"])
            loaded = tuple(
                self._source_validation()["source_a"]["loaded_module_inventory"]
            )
            before = tuple(root.rglob("*"))
            with (
                mock.patch.object(contract_module, "_git", side_effect=fake_git),
                mock.patch.object(contract_module, "_validate_prerequisites"),
                mock.patch.object(
                    contract_module, "_source_inventory", return_value=inventory
                ),
                mock.patch.object(
                    contract_module, "_loaded_module_inventory", return_value=loaded
                ),
            ):
                result = validate_source_a(
                    contract,
                    expected_source_a_git_commit=SOURCE_A,
                )
            self.assertEqual(result["status"], SOURCE_VALIDATION_STATUS)
            self.assertFalse(result["runner_freeze_b_present"])
            self.assertEqual(before, tuple(root.rglob("*")))
            self.assertNotIn("ls-remote", {argument for call in calls for argument in call})

            runner = root / RUNNER_FREEZE_B_PATH
            runner.parent.mkdir(parents=True)
            runner.symlink_to(root / "missing-runner.json")
            with self.assertRaisesRegex(ValueError, "absent"):
                validate_source_a(contract)

    def test_source_a_validator_rejects_expected_commit_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract = replace(self.contract, repository_root=Path(raw).resolve())

            def fake_git(_root: Path, *arguments: str) -> bytes:
                table = {
                    ("rev-parse", "HEAD"): f"{SOURCE_A}\n".encode(),
                    ("rev-parse", "origin/main"): f"{SOURCE_A}\n".encode(),
                    ("branch", "--show-current"): b"main\n",
                    (
                        "remote",
                        "get-url",
                        "origin",
                    ): b"https://github.com/luojiaxuan/CausalCache.git\n",
                    (
                        "status",
                        "--porcelain=v1",
                        "--untracked-files=all",
                    ): b"",
                }
                return table[tuple(arguments)]

            with mock.patch.object(contract_module, "_git", side_effect=fake_git):
                with self.assertRaisesRegex(ValueError, "clean pushed"):
                    validate_source_a(
                        contract,
                        expected_source_a_git_commit="2" * 40,
                    )

    def test_runner_freeze_bytes_are_deterministic_and_bind_source_a(self) -> None:
        validation = self._source_validation()
        first = runner_freeze_bytes(self.contract, validation)
        second = runner_freeze_bytes(self.contract, validation)
        self.assertEqual(first, second)
        value = json.loads(first)
        self.assertEqual(value["status"], RUNNER_FREEZE_STATUS)
        self.assertEqual(value["source_a_git_commit"], SOURCE_A)
        self.assertEqual(value["execution_b_direct_parent_required"], SOURCE_A)
        self.assertEqual(
            value["execution_b_required_unique_diff"], [RUNNER_FREEZE_B_PATH]
        )
        self.assertEqual(value["required_source_a_paths"], list(REQUIRED_SOURCE_A_PATHS))
        self.assertEqual(value["parent_result_git_commit"], PARENT_RESULT_COMMIT)
        self.assertEqual(value["parent_exact_download_paths"], list(PARENT_DOWNLOAD_PATHS))
        self.assertFalse(value["v1_verdict_reclassification_allowed"])

    def test_materializer_creates_only_canonical_b(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            config_dir = root / "code/configs"
            config_dir.mkdir(parents=True)
            contract = replace(self.contract, repository_root=root)
            validation = self._source_validation()

            def fake_git(_root: Path, *arguments: str) -> bytes:
                self.assertEqual(
                    arguments,
                    ("status", "--porcelain=v1", "--untracked-files=all"),
                )
                return f"?? {RUNNER_FREEZE_B_PATH}\n".encode()

            with (
                mock.patch.object(
                    contract_module, "validate_source_a", return_value=validation
                ),
                mock.patch.object(contract_module, "_git", side_effect=fake_git),
            ):
                value = materialize_runner_freeze(
                    contract,
                    expected_source_a_git_commit=SOURCE_A,
                )
            path = root / RUNNER_FREEZE_B_PATH
            self.assertEqual(value["status"], RUNNER_FREEZE_STATUS)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
            self.assertEqual(path.read_bytes(), runner_freeze_bytes(contract, validation))
            files = [item for item in root.rglob("*") if item.is_file()]
            self.assertEqual(files, [path])

    def test_runner_freeze_loader_and_execution_b_source_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            path = root / RUNNER_FREEZE_B_PATH
            path.parent.mkdir(parents=True)
            contract = replace(self.contract, repository_root=root)
            validation = self._source_validation()
            payload = runner_freeze_bytes(contract, validation)
            path.write_bytes(payload)
            path.chmod(0o644)
            freeze = load_runner_freeze(contract)
            self.assertEqual(freeze["source_a_git_commit"], SOURCE_A)

            calls: list[tuple[str, ...]] = []

            def fake_git(_root: Path, *arguments: str) -> bytes:
                calls.append(tuple(arguments))
                table = {
                    ("rev-parse", "HEAD"): f"{EXECUTION_B}\n".encode(),
                    ("rev-parse", "origin/main"): f"{EXECUTION_B}\n".encode(),
                    ("branch", "--show-current"): b"main\n",
                    (
                        "remote",
                        "get-url",
                        "origin",
                    ): b"https://github.com/luojiaxuan/CausalCache.git\n",
                    (
                        "status",
                        "--porcelain=v1",
                        "--untracked-files=all",
                    ): b"",
                    (
                        "ls-remote",
                        "--exit-code",
                        "origin",
                        "refs/heads/main",
                    ): f"{EXECUTION_B}\trefs/heads/main\n".encode(),
                    (
                        "rev-list",
                        "--parents",
                        "-n",
                        "1",
                        EXECUTION_B,
                    ): f"{EXECUTION_B} {SOURCE_A}\n".encode(),
                    (
                        "diff",
                        "--name-only",
                        SOURCE_A,
                        EXECUTION_B,
                    ): f"{RUNNER_FREEZE_B_PATH}\n".encode(),
                    ("show", f"{EXECUTION_B}:{RUNNER_FREEZE_B_PATH}"): payload,
                }
                return table[tuple(arguments)]

            source_inventory = tuple(validation["source_a"]["source_inventory"])
            loaded = tuple(validation["source_a"]["loaded_module_inventory"])
            with (
                mock.patch.object(contract_module, "_git", side_effect=fake_git),
                mock.patch.object(contract_module, "_validate_prerequisites"),
                mock.patch.object(
                    contract_module,
                    "_source_inventory",
                    return_value=source_inventory,
                ),
                mock.patch.object(
                    contract_module,
                    "_loaded_module_inventory",
                    return_value=loaded,
                ),
            ):
                result = validate_execution_b_source(
                    contract,
                    expected_execution_b_git_commit=EXECUTION_B,
                )
            self.assertTrue(result["direct_single_parent"])
            self.assertEqual(result["source_a_git_commit"], SOURCE_A)
            self.assertEqual(result["execution_b_git_commit"], EXECUTION_B)
            self.assertIn(
                (
                    "ls-remote",
                    "--exit-code",
                    "origin",
                    "refs/heads/main",
                ),
                calls,
            )

    def test_execution_b_rejects_non_direct_parent_before_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            path = root / RUNNER_FREEZE_B_PATH
            path.parent.mkdir(parents=True)
            contract = replace(self.contract, repository_root=root)
            path.write_bytes(runner_freeze_bytes(contract, self._source_validation()))
            path.chmod(0o644)

            def fake_git(_root: Path, *arguments: str) -> bytes:
                if arguments == ("rev-parse", "HEAD") or arguments == (
                    "rev-parse",
                    "origin/main",
                ):
                    return f"{EXECUTION_B}\n".encode()
                if arguments == ("branch", "--show-current"):
                    return b"main\n"
                if arguments == ("remote", "get-url", "origin"):
                    return b"https://github.com/luojiaxuan/CausalCache.git\n"
                if arguments == (
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ):
                    return b""
                if arguments[0] == "ls-remote":
                    return f"{EXECUTION_B}\trefs/heads/main\n".encode()
                if arguments[0] == "rev-list":
                    return f"{EXECUTION_B} {'3' * 40}\n".encode()
                raise AssertionError(arguments)

            with mock.patch.object(contract_module, "_git", side_effect=fake_git):
                with self.assertRaisesRegex(ValueError, "direct single-parent"):
                    validate_execution_b_source(
                        contract,
                        expected_execution_b_git_commit=EXECUTION_B,
                    )

    def test_manager_reads_only_safe_stable_token_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            token = root / "hf-token.txt"
            token.write_text("hf_example\n", encoding="ascii")
            token.chmod(0o600)
            self.assertEqual(manager._secure_token(token), "hf_example")
            token.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "0400/0600"):
                manager._secure_token(token)
            link = root / "token-link"
            link.symlink_to(token)
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                manager._secure_token(link)

    def test_manager_validates_b_before_token_or_hf_transport(self) -> None:
        args = manager._parser().parse_args(
            [
                "run",
                "--repository-root",
                str(ROOT),
                "--execution-b-git-commit",
                EXECUTION_B,
                "--data-root",
                "/data",
                "--fresh-download-parent",
                "/data/tmp",
                "--hf-token-file",
                "/data/.secrets/hf_key.txt",
            ]
        )
        with (
            mock.patch.object(
                manager,
                "validate_execution_b_source",
                side_effect=ValueError("Execution-B identity drifted"),
            ),
            mock.patch.object(manager, "_secure_token") as token,
        ):
            with self.assertRaisesRegex(ValueError, "Execution-B identity"):
                manager._delegate_execution(args, self.contract)
        token.assert_not_called()

    def test_strict_config_parser_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            contract_module._strict_json(
                b'{"schema_version":"1.0.0","schema_version":"2.0.0"}',
                label="duplicate contract",
            )

    def test_modified_config_bytes_fail_frozen_sha(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            path = root / CANONICAL_CONFIG_PATH
            path.parent.mkdir(parents=True)
            payload = (ROOT / CANONICAL_CONFIG_PATH).read_bytes()
            path.write_bytes(payload + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_frozen_failure_decomposition_contract(
                    path,
                    repository_root=root,
                )


if __name__ == "__main__":
    unittest.main()
