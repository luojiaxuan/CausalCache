from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from causalcache import gate_v1_formal_train_runner as runner
from causalcache.gate_v1_formal_train_contract import (
    MANIFEST_TARGETS,
    PAYLOAD_TARGETS,
    RUNNER_FREEZE_B_PATH,
    FormalTrainContract,
    canonical_json_bytes,
)
from scripts import manage_gate_v1_formal_train as manager


COMMIT_A = "1" * 40
COMMIT_B = "2" * 40
PAYLOAD_COMMIT = "3" * 40
MANIFEST_COMMIT = "4" * 40
TAG_OBJECT = "5" * 40
REMOTE_BASE_COMMIT = "6" * 40
CONTRACT_SHA256 = "a" * 64
GATE_CONFIG_SHA256 = (
    "37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b"
)
FORMAL_SOURCE_IDS_SHA256 = (
    "ccec55afb0882e602f5d2c83ec420682488e2663fe26a7c574a851825c43f225"
)


def _contract(root: Path, *, origin_url: str = "https://example.invalid/repo.git"):
    cache_files = [
        {
            "kind": "feature_cache",
            "path": "formal58-transport-repair/v1/feature-cache-v1.tar",
            "sha256": "1" * 64,
            "size_bytes": 1,
        },
        {
            "kind": "label_cache",
            "path": "formal58-transport-repair/v1/label-cache-v1.tar",
            "sha256": "2" * 64,
            "size_bytes": 1,
        },
        {
            "kind": "bundle_manifest",
            "path": "formal58-transport-repair/v1/cache-bundle-manifest-v1.json",
            "sha256": "3" * 64,
            "size_bytes": 1,
        },
    ]
    runtime_environment = {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "TZ": "UTC",
        "LC_ALL": "C.UTF-8",
    }
    data = {
        "lineage": {
            "gate_preregistration": {"sha256": GATE_CONFIG_SHA256},
        },
        "source_freeze": {
            "branch": "main",
            "origin_name": "origin",
            "origin_url": origin_url,
            "required_source_a_paths": ["source.py"],
            "git_prerequisites": [],
        },
        "formal_cache_input": {
            "repo": "owner/formal-cache",
            "repo_type": "dataset",
            "private": True,
            "tag": "cache-v1",
            "immutable_revision": "6" * 40,
            "annotated_tag_object": "7" * 40,
            "exact_three_files": cache_files,
            "join_only_audit_sha256": "8" * 64,
        },
        "formal_geometry": {
            "source_ids_sha256": FORMAL_SOURCE_IDS_SHA256,
            "trajectory_count": 58,
            "state_count": 174,
            "oof_fold_sizes": [12, 12, 12, 11, 11],
        },
        "destination": {
            "repo": "owner/formal-model",
            "repo_type": "model",
            "private": True,
            "tag": "train-v1",
            "tag_message": "Freeze formal train",
        },
        "output_contract": {
            "payload_commit": {"commit_title": "payload"},
            "manifest_commit": {"commit_title": "manifest"},
        },
        "local_first_state_machine": {
            "state_file_mode": 0o600,
            "artifact_file_mode": 0o444,
            "artifact_directory": "/data/artifacts/formal",
            "ordered_states": [],
        },
        "runtime_contract": {
            "python_implementation": "CPython",
            "python_version": "3.12.3",
            "machine": "x86_64",
            "torch_version": "2.11.0+cu130",
            "safetensors_version": "0.7.0",
            "huggingface_hub_version": "1.16.1",
            "docker_inspect_receipt": {
                "path": "/data/runtime-inspect.json",
                "mode": 0o600,
                "schema_version": "1.0.0",
                "status": runner.DOCKER_INSPECT_STATUS,
                "container_name_prefix": "sglang-omni-jaxan-",
                "normalized_device_requests": [],
                "privileged": False,
                "runtime": "runc",
                "data_mount_destination": "/data",
                "data_mount_rw": True,
            },
            "thread_environment": runtime_environment,
        },
        "execution_fixed_operation_contract": {
            "training_family_count": 2,
            "oof_trial_count": 20,
            "fold_training_track_count": 100,
            "final_fit_track_count": 10,
            "model_initialization_count": 110,
            "checkpoint_count": 10,
            "full_oof_report_count": 2,
            "ensemble_manifest_count": 2,
            "run_manifest_count": 1,
            "bundle_manifest_count": 1,
            "fresh16_semantic_decode_count": 0,
            "legacy_dev5_semantic_decode_count": 0,
            "confirm20_access_count": 0,
            "matched_nll_evaluation_count": 0,
            "closed_loop_episode_count": 0,
        },
    }
    return FormalTrainContract(
        data=data,
        sha256=CONTRACT_SHA256,
        repository_root=root,
        source_path=root / "contract.json",
    )


def _source(commit: str = COMMIT_B) -> runner.SourceIdentity:
    return runner.SourceIdentity(
        head=commit,
        remote_main=commit,
        branch="main",
        origin_url="https://example.invalid/repo.git",
        source_inventory=(
            {"path": "source.py", "sha256": "9" * 64, "size_bytes": 1},
        ),
        loaded_module_inventory=(
            {
                "module": "causalcache.synthetic",
                "path": "source.py",
                "sha256": "9" * 64,
                "size_bytes": 1,
            },
        ),
    )


def _runtime() -> runner.RuntimeIdentity:
    return runner.RuntimeIdentity(
        python_implementation="CPython",
        python_version="3.12.3",
        machine="x86_64",
        torch_version="2.11.0+cu130",
        safetensors_version="0.7.0",
        huggingface_hub_version="1.16.1",
        torch_num_threads=1,
        torch_num_interop_threads=1,
        thread_environment={"PYTHONHASHSEED": "0"},
        docker_inspect_receipt_sha256="f" * 64,
        container_id="a" * 64,
        container_name="sglang-omni-jaxan-07172332",
        container_hostname="a" * 12,
        image_id="sha256:" + "b" * 64,
    )


def _family_report(family: str):
    selection = {"family": family, "learning_rate": 0.0003}
    selection_sha = hashlib.sha256(canonical_json_bytes(selection)).hexdigest()
    checkpoints = [
        {
            "family": family,
            "seed": seed,
            "selected_epoch": seed + 1,
            "format": "safetensors",
            "model_state_sha256": hashlib.sha256(
                f"{family}:state:{seed}".encode()
            ).hexdigest(),
            "checkpoint_sha256": "0" * 64,
            "size_bytes": 0,
        }
        for seed in range(5)
    ]
    return {
        "family": family,
        "oof": {
            "complete_grid": selection,
            "selection_sha256": selection_sha,
        },
        "final_fit": {
            "learning_rate": 0.0003,
            "selected_epochs": [seed + 1 for seed in range(5)],
            "checkpoints": checkpoints,
        },
    }


def _payload_files():
    reports = {
        family: _family_report(family)
        for family in ("conditional", "independent")
    }
    checkpoint_items = []
    for family in ("conditional", "independent"):
        for seed in range(5):
            payload = f"{family}:{seed}:checkpoint".encode()
            digest = hashlib.sha256(payload).hexdigest()
            reports[family]["final_fit"]["checkpoints"][seed][
                "checkpoint_sha256"
            ] = digest
            reports[family]["final_fit"]["checkpoints"][seed]["size_bytes"] = len(
                payload
            )
            checkpoint_items.append(
                SimpleNamespace(family=family, seed=seed, payload=payload)
            )
    result = SimpleNamespace(checkpoints=checkpoint_items, family_reports=reports)
    return runner._payload_files(result), reports


def _strict_payload_files(contract):
    reports = {}
    checkpoint_items = []
    for family in ("conditional", "independent"):
        grid = []
        for learning_rate, metric in ((0.0003, 0.8), (0.001, 0.7)):
            for seed in range(5):
                grid.append(
                    runner.OOFTrial(
                        family=family,
                        learning_rate=learning_rate,
                        seed=seed,
                        selected_epoch=1,
                        best_raw_utility_ratio=metric,
                        epochs_run=51,
                        metric_by_epoch=(metric,) * 51,
                    )
                )
        selection = runner.FamilySelection(
            family=family,
            learning_rate=0.0003,
            five_seed_mean_oof_ratio=0.8,
            trials=tuple(grid[:5]),
            grid_trials=tuple(grid),
        )
        selection_payload = runner.family_selection_payload(selection)
        checkpoints = []
        for seed in range(5):
            payload = f"strict:{family}:{seed}".encode()
            checkpoint_items.append(
                SimpleNamespace(family=family, seed=seed, payload=payload)
            )
            checkpoints.append(
                {
                    "family": family,
                    "seed": seed,
                    "selected_epoch": 1,
                    "format": "safetensors",
                    "model_state_sha256": hashlib.sha256(
                        f"strict-state:{family}:{seed}".encode()
                    ).hexdigest(),
                    "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
        reports[family] = {
            "schema_version": runner.SCHEMA_VERSION,
            "protocol_id": runner.PROTOCOL_ID,
            "status": "COMPLETED_GATE_V1_FORMAL_TRAIN_FAMILY_V1",
            "family": family,
            "input_binding": runner._expected_input_binding(contract),
            "oof": {
                "complete_grid": selection_payload,
                "selection_sha256": hashlib.sha256(
                    canonical_json_bytes(selection_payload)
                ).hexdigest(),
            },
            "final_fit": {
                "learning_rate": 0.0003,
                "seeds": list(range(5)),
                "selected_epochs": [1] * 5,
                "checkpoints": checkpoints,
            },
            "operation_counts": runner._family_operation_counts(selection),
            "access_counts": {
                "formal58_trajectory_semantic_decode_count": 58,
                "formal58_feature_state_semantic_decode_count": 174,
                "formal58_label_state_semantic_decode_count": 174,
                "fresh16_semantic_decode_count": 0,
                "legacy_dev5_semantic_decode_count": 0,
                "confirm20_access_count": 0,
                "matched_nll_evaluation_count": 0,
                "closed_loop_episode_count": 0,
            },
        }
    result = SimpleNamespace(
        checkpoints=checkpoint_items,
        family_reports=reports,
    )
    return runner._payload_files(result), reports


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class FormalTrainCliAndTokenTest(unittest.TestCase):
    def test_cli_exposes_only_protocol_and_infrastructure_arguments(self) -> None:
        parser = manager._parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, __import__("argparse")._SubParsersAction)
        )
        forbidden = {
            "--family",
            "--learning-rate",
            "--seed",
            "--epochs",
            "--patience",
            "--budget",
            "--trainer",
            "--feature-cache",
            "--label-cache",
            "--checkpoint-format",
        }
        for child in subparsers.choices.values():
            exposed = {
                option
                for action in child._actions
                for option in action.option_strings
            }
            self.assertFalse(exposed & forbidden)
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parser.parse_args(
                [
                    "run",
                    "--repository-root",
                    "/repo",
                    "--execution-b-git-commit",
                    COMMIT_B,
                    "--hf-token-file",
                    "/token",
                    "--data-root",
                    "/data",
                    "--fresh-download-parent",
                    "/fresh",
                    "--seed",
                    "9",
                ]
            )

    def test_secure_token_mode_symlink_and_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            token = root / "token"
            token.write_bytes(b"hf_secret\n")
            token.chmod(0o600)
            self.assertEqual(manager._secure_token(token), "hf_secret")
            token.chmod(0o400)
            self.assertEqual(manager._secure_token(token), "hf_secret")
            token.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "0400/0600"):
                manager._secure_token(token)
            token.chmod(0o600)
            symlink = root / "token-link"
            symlink.symlink_to(token)
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                manager._secure_token(symlink)
            token.write_bytes(b"hf secret")
            with self.assertRaisesRegex(ValueError, "malformed"):
                manager._secure_token(token)
            token.write_bytes(b" \n\t")
            with self.assertRaisesRegex(ValueError, "empty or malformed"):
                manager._secure_token(token)


class FormalTrainSourceBoundaryTest(unittest.TestCase):
    def test_source_a_freeze_and_direct_execution_b_use_temporary_git(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            remote = root / "remote.git"
            work = root / "work"
            subprocess.run(
                ["git", "init", "--bare", str(remote)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(
                ["git", "init", "-b", "main", str(work)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _git(work, "config", "user.name", "CausalCache Test")
            _git(work, "config", "user.email", "test@example.invalid")
            (work / "source.py").write_text("SOURCE = 1\n")
            _git(work, "add", "source.py")
            _git(work, "commit", "-m", "source-a")
            _git(work, "remote", "add", "origin", str(remote))
            _git(work, "push", "-u", "origin", "main")
            commit_a = _git(work, "rev-parse", "HEAD")
            contract = _contract(work, origin_url=str(remote))
            module_inventory = (
                {
                    "module": "causalcache.synthetic",
                    "path": "source.py",
                    "sha256": hashlib.sha256(b"SOURCE = 1\n").hexdigest(),
                    "size_bytes": len(b"SOURCE = 1\n"),
                },
            )
            with mock.patch.object(
                runner,
                "_loaded_module_inventory",
                return_value=module_inventory,
            ):
                source_validation = runner.validate_source_a(
                    contract,
                    expected_source_a_git_commit=commit_a,
                )
                self.assertFalse(source_validation["execution_authorized"])
                with self.assertRaises(ValueError):
                    runner.load_runner_freeze(contract)
                freeze = runner.materialize_runner_freeze(
                    contract,
                    expected_source_a_git_commit=commit_a,
                )
                self.assertEqual(freeze["source_a_git_commit"], commit_a)
                self.assertEqual(
                    freeze["execution_b_required_unique_diff"],
                    [RUNNER_FREEZE_B_PATH],
                )
                self.assertEqual(runner.load_runner_freeze(contract), freeze)
                freeze_path = work / RUNNER_FREEZE_B_PATH
                freeze_bytes = freeze_path.read_bytes()
                forged_freeze = dict(freeze)
                forged_freeze["unexpected"] = True
                freeze_path.write_bytes(runner.pretty_json_bytes(forged_freeze))
                with self.assertRaisesRegex(ValueError, "keys drifted"):
                    runner.load_runner_freeze(contract)
                freeze_path.write_bytes(freeze_bytes)
                with self.assertRaisesRegex(ValueError, "must be absent"):
                    runner.validate_source_a(contract)
                _git(work, "add", RUNNER_FREEZE_B_PATH)
                _git(work, "commit", "-m", "execution-b")
                _git(work, "push", "origin", "main")
                commit_b = _git(work, "rev-parse", "HEAD")
                source_b = runner.validate_execution_b_source(
                    contract,
                    expected_execution_b_git_commit=commit_b,
                )
            self.assertEqual(source_b.head, commit_b)
            self.assertEqual(
                _git(work, "diff", "--name-only", commit_a, commit_b),
                RUNNER_FREEZE_B_PATH,
            )


class FormalTrainRuntimeTest(unittest.TestCase):
    def _runtime_modules(self, *, cuda_available: bool = False):
        deterministic = {"enabled": False}
        threads = {"intra": 1, "interop": 1}
        torch = types.ModuleType("torch")
        torch.__version__ = "2.11.0+cu130"
        torch.set_num_threads = lambda value: threads.__setitem__("intra", value)
        torch.get_num_threads = lambda: threads["intra"]
        torch.set_num_interop_threads = lambda value: threads.__setitem__(
            "interop", value
        )
        torch.get_num_interop_threads = lambda: threads["interop"]
        torch.use_deterministic_algorithms = lambda value: deterministic.__setitem__(
            "enabled", value
        )
        torch.are_deterministic_algorithms_enabled = lambda: deterministic["enabled"]
        torch.cuda = SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: int(cuda_available),
        )
        safetensors = types.ModuleType("safetensors")
        safetensors.__version__ = "0.7.0"
        hub = types.ModuleType("huggingface_hub")
        hub.__version__ = "1.16.1"
        return {"torch": torch, "safetensors": safetensors, "huggingface_hub": hub}

    def test_capture_runtime_binds_docker_inspect_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = _contract(root)
            container_id = "a" * 64
            inspect_record = [
                {
                    "Id": container_id,
                    "Name": "/sglang-omni-jaxan-07172332",
                    "Image": "sha256:" + "b" * 64,
                    "Config": {"Hostname": container_id[:12]},
                    "State": {"Running": True},
                    "HostConfig": {
                        "DeviceRequests": None,
                        "Privileged": False,
                        "Runtime": "runc",
                    },
                    "Mounts": [
                        {
                            "Type": "bind",
                            "Source": str(root),
                            "Destination": "/data",
                            "RW": True,
                        }
                    ],
                }
            ]
            completed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(inspect_record).encode(),
                stderr=b"",
            )
            with mock.patch.object(runner.subprocess, "run", return_value=completed):
                receipt = runner.capture_docker_inspect_receipt(
                    contract,
                    source=_source(),
                    container_name="sglang-omni-jaxan-07172332",
                    host_data_root=root,
                )
            self.assertEqual(receipt["runtime"]["normalized_device_requests"], [])
            self.assertFalse(receipt["runtime"]["privileged"])
            self.assertEqual(receipt["runtime"]["runtime"], "runc")
            self.assertTrue((root / "runtime-inspect.json").is_file())

            bad = copy.deepcopy(inspect_record)
            bad[0]["HostConfig"]["DeviceRequests"] = [{"Driver": "nvidia"}]
            failed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(bad).encode(),
                stderr=b"",
            )
            (root / "runtime-inspect.json").unlink()
            with (
                mock.patch.object(runner.subprocess, "run", return_value=failed),
                self.assertRaisesRegex(ValueError, "runtime boundary"),
            ):
                runner.capture_docker_inspect_receipt(
                    contract,
                    source=_source(),
                    container_name="sglang-omni-jaxan-07172332",
                    host_data_root=root,
                )

    def test_runtime_contract_passes_exact_mock_and_rejects_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = _contract(root)
            source = _source()
            container_id = "a" * 64
            receipt = {
                "schema_version": "1.0.0",
                "protocol_id": runner.PROTOCOL_ID,
                "status": runner.DOCKER_INSPECT_STATUS,
                "contract_sha256": contract.sha256,
                "source": runner._source_record(source),
                "container": {
                    "id": container_id,
                    "name": "sglang-omni-jaxan-07172332",
                    "hostname": container_id[:12],
                    "image_id": "sha256:" + "b" * 64,
                    "running": True,
                },
                "runtime": {
                    "normalized_device_requests": [],
                    "privileged": False,
                    "runtime": "runc",
                },
                "data_mount": {
                    "type": "bind",
                    "source": "/host/data",
                    "destination": "/data",
                    "rw": True,
                },
                "training_executed": False,
                "execution_authorized": True,
            }
            runner._exclusive_or_identical(
                root / "runtime-inspect.json",
                runner.pretty_json_bytes(receipt),
                mode=0o600,
            )
            environment = {
                **contract.runtime["thread_environment"],
                "NVIDIA_VISIBLE_DEVICES": "void",
                "CUDA_VISIBLE_DEVICES": "",
            }
            common = (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(runner.Path, "glob", return_value=[]),
                mock.patch.object(
                    runner.platform, "python_implementation", return_value="CPython"
                ),
                mock.patch.object(
                    runner.platform, "python_version", return_value="3.12.3"
                ),
                mock.patch.object(runner.platform, "machine", return_value="x86_64"),
                mock.patch.object(
                    runner.socket,
                    "gethostname",
                    return_value=container_id[:12],
                ),
            )
            with contextlib.ExitStack() as stack:
                for item in common:
                    stack.enter_context(item)
                stack.enter_context(
                    mock.patch.dict(
                        sys.modules,
                        self._runtime_modules(cuda_available=False),
                        clear=False,
                    )
                )
                identity = runner.validate_execution_runtime(
                    contract,
                    source=source,
                    data_root=root,
                )
            self.assertEqual(identity.torch_num_threads, 1)
            self.assertEqual(identity.torch_num_interop_threads, 1)
            receipt_path = root / "runtime-inspect.json"
            receipt_bytes = receipt_path.read_bytes()
            forged_receipt = dict(receipt)
            forged_receipt["unexpected"] = True
            receipt_path.write_bytes(runner.pretty_json_bytes(forged_receipt))
            with self.assertRaisesRegex(ValueError, "keys drifted"):
                runner.validate_execution_runtime(
                    contract,
                    source=source,
                    data_root=root,
                )
            receipt_path.write_bytes(receipt_bytes)
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, environment, clear=False))
                stack.enter_context(mock.patch.object(runner.Path, "glob", return_value=[]))
                stack.enter_context(
                    mock.patch.object(
                        runner.platform, "python_implementation", return_value="CPython"
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runner.platform, "python_version", return_value="3.12.3"
                    )
                )
                stack.enter_context(
                    mock.patch.object(runner.platform, "machine", return_value="x86_64")
                )
                stack.enter_context(
                    mock.patch.object(
                        runner.socket,
                        "gethostname",
                        return_value=container_id[:12],
                    )
                )
                stack.enter_context(
                    mock.patch.dict(
                        sys.modules,
                        self._runtime_modules(cuda_available=True),
                        clear=False,
                    )
                )
                with self.assertRaisesRegex(ValueError, "framework/runtime"):
                    runner.validate_execution_runtime(
                        contract,
                        source=source,
                        data_root=root,
                    )


class FormalTrainLocalArtifactTest(unittest.TestCase):
    def test_complete_oof_reports_replay_dynamic_counts_and_reject_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract = _contract(Path(raw).resolve())
            payload_files, reports = _strict_payload_files(contract)
            counts = runner._expected_operation_counts(contract, payload_files)
            self.assertEqual(counts["oof_trial_count"], 20)
            self.assertEqual(counts["fold_training_track_count"], 100)
            self.assertEqual(counts["oof_epoch_metric_count"], 1020)
            self.assertEqual(counts["optimizer_step_count"], 5110)
            runner._validate_operation_counts(contract, counts, payload_files)
            with self.assertRaisesRegex(ValueError, "operation counts"):
                runner._validate_operation_counts(
                    contract,
                    {**counts, "unknown_count": 1},
                    payload_files,
                )
            changed_reports = copy.deepcopy(reports)
            changed_reports["conditional"]["oof"]["complete_grid"][
                "grid_trials"
            ][0]["selected_epoch"] = 2
            changed = dict(payload_files)
            changed[runner._report_path("conditional")] = (
                canonical_json_bytes(changed_reports["conditional"]) + b"\n"
            )
            with self.assertRaises(ValueError):
                runner._expected_operation_counts(contract, changed)

    def test_exclusive_or_identical_and_symlink_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            path = root / "state.json"
            self.assertTrue(
                runner._exclusive_or_identical(path, b"payload", mode=0o600)
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertFalse(
                runner._exclusive_or_identical(path, b"payload", mode=0o600)
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                runner._exclusive_or_identical(path, b"other", mode=0o600)
            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                runner._exclusive_or_identical(link, b"payload", mode=0o600)

    def test_payload_manifest_checkpoint_provenance_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract = _contract(Path(raw))
            payload_files, reports = _payload_files()
            self.assertEqual(set(payload_files), set(PAYLOAD_TARGETS))
            for family in ("conditional", "independent"):
                report_path = runner._report_path(family)
                self.assertEqual(
                    payload_files[report_path],
                    canonical_json_bytes(reports[family]) + b"\n",
                )
            operation_counts = {
                "oof_trial_count": 20,
                "fold_training_track_count": 100,
                "final_fit_track_count": 10,
                "model_initialization_count": 110,
                "checkpoint_count": 10,
                "optimizer_step_count": 100,
            }
            training_completion = {
                "selection_sha256": {
                    family: reports[family]["oof"]["selection_sha256"]
                    for family in ("conditional", "independent")
                },
                "operation_counts": operation_counts,
            }
            def selection(family, _report, _contract):
                return SimpleNamespace(
                    family=family,
                    learning_rate=0.0003,
                    trials=tuple(
                        SimpleNamespace(selected_epoch=seed + 1)
                        for seed in range(5)
                    ),
                )

            with mock.patch.object(
                runner,
                "_validate_family_report",
                side_effect=selection,
            ):
                manifest_files = runner._manifest_files(
                    contract=contract,
                    source=_source(),
                    runtime=_runtime(),
                    payload_commit=PAYLOAD_COMMIT,
                    payload_files=payload_files,
                    training_completion=training_completion,
                )
            self.assertEqual(set(manifest_files), set(MANIFEST_TARGETS))
            for payload in manifest_files.values():
                self.assertNotIn(MANIFEST_COMMIT.encode(), payload)
            with mock.patch.object(
                runner, "load_safetensors_checkpoint"
            ) as checkpoint_load:
                replay = runner._validate_checkpoints_from_manifests(
                    payload_files,
                    manifest_files,
                )
            self.assertEqual(checkpoint_load.call_count, 10)
            self.assertEqual(set(replay), {"conditional", "independent"})

    def test_completion_replay_requires_retained_stage_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = _contract(root)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            payload_files = {"payload.bin": b"payload"}
            manifest_files = {"manifest.json": b"manifest"}
            runner._publish_local_files(artifacts, payload_files, mode=0o444)
            runner._publish_local_files(artifacts, manifest_files, mode=0o444)
            completion = {
                "remote_base_commit": REMOTE_BASE_COMMIT,
                "remote_base_files": [],
                "payload_commit": PAYLOAD_COMMIT,
                "manifest_commit": MANIFEST_COMMIT,
                "annotated_tag_object": TAG_OBJECT,
                "payload_files": [runner._file_record("payload.bin", b"payload")],
                "manifest_files": [
                    runner._file_record("manifest.json", b"manifest")
                ],
                "selection_sha256": {
                    "conditional": "1" * 64,
                    "independent": "2" * 64,
                },
                "operation_counts": {
                    **contract.execution_operations,
                    "optimizer_step_count": 100,
                },
            }
            source = _source()
            state_names = {
                "global_claim": (runner.GLOBAL_CLAIM_STATUS, "claim.json"),
                "cache_input_completion": (
                    runner.CACHE_INPUT_COMPLETION_STATUS,
                    "cache.json",
                ),
                "training_artifact_completion": (
                    runner.TRAINING_ARTIFACT_COMPLETION_STATUS,
                    "training.json",
                ),
                "remote_base_receipt": (runner.REMOTE_BASE_STATUS, "remote.json"),
                "payload_commit_receipt": (
                    runner.PAYLOAD_COMMIT_STATUS,
                    "payload.json",
                ),
                "completion_staging": (
                    runner.FINAL_COMPLETION_STATUS,
                    "completion.staged.json",
                ),
                "final_completion": (
                    runner.FINAL_COMPLETION_STATUS,
                    "completion.json",
                ),
            }
            paths = {name: root / filename for name, (_, filename) in state_names.items()}
            claim_bytes = runner._state_bytes(
                contract,
                status=runner.GLOBAL_CLAIM_STATUS,
                source=source,
                payload={},
            )
            runner._exclusive_or_identical(
                paths["global_claim"], claim_bytes, mode=0o600
            )
            claim_sha256 = hashlib.sha256(claim_bytes).hexdigest()
            chained_payloads = {
                "cache_input_completion": {"claim_sha256": claim_sha256},
                "training_artifact_completion": {
                    "claim_sha256": claim_sha256,
                    "payload_files": completion["payload_files"],
                    "selection_sha256": completion["selection_sha256"],
                    "operation_counts": completion["operation_counts"],
                },
                "remote_base_receipt": {
                    "claim_sha256": claim_sha256,
                    "repo_initially_absent": True,
                },
                "payload_commit_receipt": {
                    "claim_sha256": claim_sha256,
                    "remote_base_commit": REMOTE_BASE_COMMIT,
                    "remote_base_files": [],
                    "payload_commit": PAYLOAD_COMMIT,
                    "payload_files": completion["payload_files"],
                },
            }
            for name, payload in chained_payloads.items():
                state_bytes = runner._state_bytes(
                    contract,
                    status=state_names[name][0],
                    source=source,
                    payload=payload,
                )
                runner._exclusive_or_identical(paths[name], state_bytes, mode=0o600)
            completion_bytes = runner._state_bytes(
                contract,
                status=runner.FINAL_COMPLETION_STATUS,
                source=source,
                payload={"claim_sha256": claim_sha256, **completion},
            )
            staged = paths["completion_staging"]
            final = paths["final_completion"]
            runner._exclusive_or_identical(staged, completion_bytes, mode=0o600)
            os.link(staged, final)
            with (
                mock.patch.object(
                    runner,
                    "_tag_snapshot",
                    return_value=(TAG_OBJECT, MANIFEST_COMMIT),
                ),
                mock.patch.object(
                    runner,
                    "_download_and_compare_revision",
                    return_value={},
                ) as download,
                mock.patch.object(runner, "_validate_two_commit_chain"),
                mock.patch.object(runner, "_validate_operation_counts"),
                mock.patch.object(
                    runner,
                    "_report_selection_sha256",
                    return_value=completion["selection_sha256"],
                ),
                mock.patch.object(
                    runner,
                    "_manifest_files",
                    return_value=manifest_files,
                ),
                mock.patch.object(
                    runner,
                    "_validate_checkpoints_from_manifests",
                    return_value={"replayed": True},
                ),
            ):
                replay = runner._completion_replay(
                    api=object(),
                    download_fn=lambda **_kwargs: "unused",
                    contract=contract,
                    source=source,
                    runtime=_runtime(),
                    paths=paths,
                    artifact_root=artifacts,
                    fresh_parent=root,
                    state_mode=0o600,
                    artifact_mode=0o444,
                )
            self.assertEqual(download.call_count, 3)
            self.assertFalse(replay["training_executed"])
            self.assertEqual(replay["remote_mutation_call_count"], 0)
            final.unlink()
            runner._exclusive_or_identical(final, completion_bytes, mode=0o600)
            with self.assertRaisesRegex(ValueError, "hard link"):
                runner._completion_replay(
                    api=object(),
                    download_fn=lambda **_kwargs: "unused",
                    contract=contract,
                    source=source,
                    runtime=_runtime(),
                    paths=paths,
                    artifact_root=artifacts,
                    fresh_parent=root,
                    state_mode=0o600,
                    artifact_mode=0o444,
                )


class FormalTrainOrchestrationTest(unittest.TestCase):
    def test_fresh_run_and_read_only_replay_cover_remote_state_machine(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = _contract(root)
            source = _source()
            runtime = _runtime()
            fresh = root / "fresh"
            fresh.mkdir()
            artifacts = root / "artifacts"
            state_names = (
                "global_claim",
                "cache_input_completion",
                "training_artifact_completion",
                "remote_base_receipt",
                "payload_commit_receipt",
                "completion_staging",
                "final_completion",
            )
            paths = {name: root / "states" / f"{name}.json" for name in state_names}
            payload_files = {
                path: f"payload:{path}".encode("utf-8")
                for path in PAYLOAD_TARGETS
            }
            manifest_files = {
                path: f"manifest:{path}".encode("utf-8")
                for path in MANIFEST_TARGETS
            }
            selection_sha256 = {
                "conditional": "a" * 64,
                "independent": "b" * 64,
            }
            operation_counts = {"optimizer_step_count": 100}
            training_payload = {
                "training_executed": True,
                "payload_files": [
                    runner._file_record(path, payload_files[path])
                    for path in sorted(payload_files)
                ],
                "selection_sha256": selection_sha256,
                "operation_counts": operation_counts,
            }
            cache_bytes = {
                record["path"]: b"cache"
                for record in contract.cache_input["exact_three_files"]
            }
            formal_inputs = SimpleNamespace(
                source_ids=tuple(f"trajectory-{index}" for index in range(58)),
                states=tuple(range(174)),
                folds=(
                    tuple(range(12)),
                    tuple(range(12)),
                    tuple(range(12)),
                    tuple(range(11)),
                    tuple(range(11)),
                ),
            )
            checkpoint_replay = {
                "conditional": {"checkpoint_count": 5},
                "independent": {"checkpoint_count": 5},
            }
            tag_identity = (TAG_OBJECT, MANIFEST_COMMIT)
            tag_side_effect = [None, tag_identity, tag_identity, tag_identity]
            patches = (
                mock.patch.object(
                    runner, "validate_execution_b_source", return_value=source
                ),
                mock.patch.object(
                    runner, "validate_execution_runtime", return_value=runtime
                ),
                mock.patch.object(runner, "_state_paths", return_value=paths),
                mock.patch.object(
                    runner, "_artifact_root", return_value=artifacts
                ),
                mock.patch.object(
                    runner,
                    "download_formal_cache_inputs",
                    return_value=cache_bytes,
                ),
                mock.patch.object(
                    runner,
                    "load_repaired_formal_training_inputs",
                    return_value=formal_inputs,
                ),
                mock.patch.object(
                    runner,
                    "_formal_source_ids",
                    return_value=formal_inputs.source_ids,
                ),
                mock.patch.object(
                    runner, "run_formal_training", return_value=SimpleNamespace()
                ),
                mock.patch.object(
                    runner, "_payload_files", return_value=payload_files
                ),
                mock.patch.object(
                    runner,
                    "_training_completion_payload",
                    return_value=training_payload,
                ),
                mock.patch.object(runner, "_repo_exists_snapshot", return_value=None),
                mock.patch.object(
                    runner,
                    "_ensure_model_repo",
                    return_value=(REMOTE_BASE_COMMIT, (".gitattributes",), 1),
                ),
                mock.patch.object(runner, "_validate_initial_model_base"),
                mock.patch.object(
                    runner,
                    "_commit_files",
                    side_effect=[(PAYLOAD_COMMIT, 1), (MANIFEST_COMMIT, 1)],
                ),
                mock.patch.object(
                    runner, "_manifest_files", return_value=manifest_files
                ),
                mock.patch.object(
                    runner,
                    "_repo_snapshot",
                    return_value=(
                        PAYLOAD_COMMIT,
                        (".gitattributes", *PAYLOAD_TARGETS),
                    ),
                ),
                mock.patch.object(runner, "_validate_two_commit_chain"),
                mock.patch.object(
                    runner,
                    "_create_or_validate_tag",
                    return_value=(TAG_OBJECT, 1),
                ),
                mock.patch.object(
                    runner,
                    "_validate_checkpoints_from_manifests",
                    return_value=checkpoint_replay,
                ),
                mock.patch.object(runner, "_validate_operation_counts"),
                mock.patch.object(
                    runner,
                    "_report_selection_sha256",
                    return_value=selection_sha256,
                ),
            )
            with contextlib.ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                download_revision = stack.enter_context(
                    mock.patch.object(
                        runner,
                        "_download_and_compare_revision",
                        return_value={},
                    )
                )
                tag_snapshot = stack.enter_context(
                    mock.patch.object(
                    runner,
                    "_tag_snapshot",
                    autospec=True,
                    side_effect=tag_side_effect,
                    )
                )
                first = runner.execute_formal_train(
                    mode="run",
                    contract=contract,
                    api=object(),
                    download_fn=lambda **_kwargs: "unused",
                    operation_factory=object(),
                    expected_execution_b_git_commit=COMMIT_B,
                    data_root=root,
                    fresh_download_parent=fresh,
                )
                replay = runner.execute_formal_train(
                    mode="validate",
                    contract=contract,
                    api=object(),
                    download_fn=lambda **_kwargs: "unused",
                    operation_factory=object(),
                    expected_execution_b_git_commit=COMMIT_B,
                    data_root=root,
                    fresh_download_parent=fresh,
                )
            self.assertEqual(first["status"], runner.RUN_STATUS)
            self.assertEqual(first["remote_mutation_call_count"], 4)
            self.assertTrue(first["training_executed"])
            self.assertEqual(replay["status"], runner.VALIDATE_STATUS)
            self.assertEqual(replay["remote_mutation_call_count"], 0)
            self.assertFalse(replay["training_executed"])
            self.assertEqual(download_revision.call_count, 6)
            self.assertEqual(tag_snapshot.call_count, 4)
            self.assertEqual(
                paths["completion_staging"].stat().st_ino,
                paths["final_completion"].stat().st_ino,
            )


class FormalTrainHubErrorTest(unittest.TestCase):
    def test_remote_target_partition_accepts_only_complete_phases(self) -> None:
        self.assertEqual(runner._remote_target_partition([".gitattributes"]), (set(), set()))
        payload = set(PAYLOAD_TARGETS)
        manifest = set(MANIFEST_TARGETS)
        self.assertEqual(
            runner._remote_target_partition(
                [".gitattributes", *PAYLOAD_TARGETS, *MANIFEST_TARGETS]
            ),
            (payload, manifest),
        )
        with self.assertRaisesRegex(ValueError, "partial or conflicting"):
            runner._remote_target_partition([PAYLOAD_TARGETS[0]])
        with self.assertRaisesRegex(ValueError, "partial or conflicting"):
            runner._remote_target_partition([*PAYLOAD_TARGETS, MANIFEST_TARGETS[0]])

    def test_exact_two_commit_history_and_manifest_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract = _contract(Path(raw).resolve())

            class Api:
                def __init__(self, manifest_title="manifest"):
                    self.manifest_title = manifest_title

                def list_repo_commits(self, *_args, **_kwargs):
                    return [
                        SimpleNamespace(
                            commit_id=MANIFEST_COMMIT,
                            title=self.manifest_title,
                        ),
                        SimpleNamespace(commit_id=PAYLOAD_COMMIT, title="payload"),
                        SimpleNamespace(commit_id=REMOTE_BASE_COMMIT, title="initial"),
                    ]

                def repo_info(self, *_args, **_kwargs):
                    return SimpleNamespace(private=True, sha=MANIFEST_COMMIT)

                def list_repo_files(self, *_args, **_kwargs):
                    return [".gitattributes", *PAYLOAD_TARGETS, *MANIFEST_TARGETS]

            api = Api()
            self.assertEqual(
                runner._recover_manifest_chain(
                    api,
                    contract,
                    manifest_commit=MANIFEST_COMMIT,
                ),
                (REMOTE_BASE_COMMIT, PAYLOAD_COMMIT),
            )
            runner._validate_two_commit_chain(
                api,
                contract,
                remote_base_commit=REMOTE_BASE_COMMIT,
                payload_commit=PAYLOAD_COMMIT,
                manifest_commit=MANIFEST_COMMIT,
            )
            with self.assertRaisesRegex(ValueError, "history"):
                runner._recover_manifest_chain(
                    Api(manifest_title="unrelated"),
                    contract,
                    manifest_commit=MANIFEST_COMMIT,
                )

    def test_generic_hf_http_error_is_not_repo_absence(self) -> None:
        class HfHubHTTPError(RuntimeError):
            pass

        class Api:
            create_repo_calls = 0

            def repo_info(self, *_args, **_kwargs):
                raise HfHubHTTPError("503 service unavailable")

            def create_repo(self, *_args, **_kwargs):
                self.create_repo_calls += 1

        with tempfile.TemporaryDirectory() as raw:
            contract = _contract(Path(raw))
            api = Api()
            with self.assertRaisesRegex(HfHubHTTPError, "503"):
                runner._repo_exists_snapshot(api, contract)
            with self.assertRaisesRegex(HfHubHTTPError, "503"):
                runner._ensure_model_repo(api, contract)
            self.assertEqual(api.create_repo_calls, 0)


if __name__ == "__main__":
    unittest.main()
