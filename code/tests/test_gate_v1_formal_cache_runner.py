from __future__ import annotations

import os
import json
import hashlib
from contextlib import ExitStack
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from causalcache.gate_v1_formal_cache_runner import (
    CacheArtifact,
    FINAL_COMPLETION_STATUS,
    FormalCacheHooks,
    REMOTE_TAGGED_BYTE_IDENTICAL,
    RUNNER_FREEZE_PATH,
    SourceIdentity,
    TRANSPORT_REPAIR_RUNNER_FREEZE_PATH,
    _runner_freeze_path,
    execute_formal_cache,
    materialize_runner_freeze,
    sha256_bytes,
    validate_execution_b_source,
    validate_no_gpu_or_model_runtime,
    validate_source_a,
)
from causalcache import gate_v1_formal_cache as cache_core
from causalcache.gate_v1_formal_cache_contract import load_frozen_formal_cache_contract
from scripts.manage_gate_v1_formal_cache import (
    _input_specs,
    _observed_execution_counts,
    _parser,
)


COMMIT_A = "1" * 40
COMMIT_B = "2" * 40
TAG_OBJECT = "3" * 40
INTERPOSED_COMMIT = "6" * 40
CONTRACT_SHA = "a" * 64


EXPECTED_OPERATIONS = {
    "trajectory_semantic_decode_count": 58,
    "feature_state_count": 174,
    "ocr_semantic_decode_count": 290,
    "label_state_semantic_decode_count": 174,
    "distance_value_decode_count": 1624,
    "join_validation_count": 174,
    "candidate_feature_count": 522,
    "conditional_edge_count": 1682,
    "independent_target_count": 522,
    "development_semantic_decode_count": 0,
    "confirm_semantic_decode_count": 0,
    "training_example_count": 0,
    "optimizer_step_count": 0,
    "model_load_count": 0,
    "model_forward_count": 0,
    "oracle_metric_count": 0,
    "oof_metric_count": 0,
    "checkpoint_count": 0,
    "matched_nll_evaluation_count": 0,
    "closed_loop_episode_count": 0,
}


@dataclass(frozen=True)
class _Contract:
    data: dict
    sha256: str
    repository_root: Path

    @property
    def local_state(self):
        return self.data["local_first_state_machine"]

    @property
    def formats(self):
        return self.data["cache_formats"]

    @property
    def destination(self):
        return self.data["destination"]

    @property
    def execution_operations(self):
        return self.data["execution_expected_operation_contract"]


def _contract(root: Path) -> _Contract:
    artifacts = root / "artifacts"
    state = root / "state"
    artifacts.mkdir()
    state.mkdir()
    ordered = []
    for name, filename in (
        ("global_claim", "claim.json"),
        ("feature_completion", "feature.json"),
        ("label_access_claim", "label-access.json"),
        ("label_completion", "label.json"),
        ("remote_base_receipt", "remote.json"),
        ("completion_staging", "completion.staged.json"),
        ("final_completion", "completion.json"),
    ):
        ordered.append({"name": name, "path": str(state / filename)})
    data = {
        "source_freeze": {
            "branch": "main",
            "origin_name": "origin",
            "origin_url": "https://github.com/luojiaxuan/CausalCache.git",
            "required_source_a_paths": [
                "code/causalcache/gate_v1_formal_cache_runner.py",
                "code/scripts/manage_gate_v1_formal_cache.py",
                "code/tests/test_gate_v1_formal_cache_runner.py",
            ],
            "git_prerequisites": [
                {"path": "README.md", "sha256": "b" * 64, "size_bytes": 1}
            ],
            "execution_b_runner_freeze": {
                "path": RUNNER_FREEZE_PATH,
                "only_allowed_execution_b_source_tree_diff": True,
                "bind_required_source_a_paths": True,
                "bind_git_prerequisites": True,
                "bind_source_a_inventory_sha256": True,
            },
        },
        "local_first_state_machine": {
            "state_file_mode": 0o600,
            "ordered_states": ordered,
        },
        "cache_formats": {
            "feature": {"archive_path": str(artifacts / "feature.tar")},
            "label": {"archive_path": str(artifacts / "label.tar")},
        },
        "destination": {
            "repo": "owner/private-cache",
            "repo_type": "dataset",
            "private": True,
            "tag": "formal-v1",
            "annotated_tag_required": True,
            "commit_title": "Publish formal caches",
            "sidecar_status": "IMMUTABLE_GATE_V1_FORMAL58_CACHE_BUNDLE_MANIFEST_V1",
            "archive_and_sidecar_same_commit_required": True,
            "exact_three_targets": [
                "formal/feature.tar",
                "formal/label.tar",
                "formal/manifest.json",
            ],
        },
        "execution_expected_operation_contract": dict(EXPECTED_OPERATIONS),
        "runtime_contract": {
            "device": "cpu",
            "gpu_required": False,
            "normalized_device_requests": [],
            "nvidia_visible_devices": "void",
            "cuda_visible_devices": "",
            "nvidia_device_nodes": [],
            "model_framework_import_allowed": False,
            "python_implementation": "CPython",
            "python_version": "3.12.3",
            "machine": "x86_64",
            "thread_environment": {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
                "TZ": "UTC",
                "LC_ALL": "C.UTF-8",
            },
        },
    }
    return _Contract(data, CONTRACT_SHA, root)


def _source() -> SourceIdentity:
    return SourceIdentity(
        head=COMMIT_B,
        remote_main=COMMIT_B,
        branch="main",
        origin_url="https://github.com/luojiaxuan/CausalCache.git",
        source_inventory=({"path": "runner.py", "sha256": "4" * 64, "size_bytes": 1},),
        loaded_module_inventory=(
            {"module": "causalcache.runner", "path": "runner.py", "sha256": "4" * 64, "size_bytes": 1},
        ),
    )


class _Operation:
    def __init__(self, *, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class _FakeHF:
    def __init__(
        self,
        *,
        lose_commit=False,
        lose_tag=False,
        interpose_on_commit_loss=False,
    ):
        self.commits = {COMMIT_A: {".gitattributes": b"lfs\n"}}
        self.titles = {COMMIT_A: "Initial commit"}
        self.parents = {COMMIT_A: None}
        self.main = COMMIT_A
        self.tag_commit = None
        self.tag_object = None
        self.lose_commit = lose_commit
        self.lose_tag = lose_tag
        self.interpose_on_commit_loss = interpose_on_commit_loss
        self.create_commit_call_count = 0
        self.create_tag_call_count = 0

    def create_repo(self, *_args, **_kwargs):
        return None

    def dataset_info(self, _repo, *, revision):
        commit = self.tag_commit if revision == "formal-v1" else revision
        if revision == "main":
            commit = self.main
        if commit not in self.commits:
            raise ValueError("unknown revision")
        return SimpleNamespace(sha=commit, private=True)

    def list_repo_files(self, _repo, *, repo_type, revision):
        self._dataset(repo_type)
        return sorted(self.commits[revision])

    def list_repo_refs(self, _repo, *, repo_type):
        self._dataset(repo_type)
        tags = []
        if self.tag_commit is not None:
            tags.append(SimpleNamespace(name="formal-v1", target_commit=self.tag_object))
        return SimpleNamespace(tags=tags)

    def list_repo_commits(self, _repo, *, repo_type, revision):
        self._dataset(repo_type)
        identifiers = []
        current = revision
        while current is not None:
            identifiers.append(current)
            current = self.parents[current]
        return [
            SimpleNamespace(commit_id=item, title=self.titles[item])
            for item in identifiers
        ]

    def list_repo_tree(
        self,
        _repo,
        *,
        path_in_repo,
        recursive,
        expand,
        revision,
        repo_type,
    ):
        self._dataset(repo_type)
        if path_in_repo is not None or not recursive or not expand:
            raise ValueError("tree request drifted")
        return [
            SimpleNamespace(
                path=path,
                blob_id=hashlib.sha1(payload).hexdigest(),
                size=len(payload),
                lfs=None,
                xet_hash=None,
            )
            for path, payload in sorted(self.commits[revision].items())
        ]

    def create_commit(
        self,
        _repo,
        *,
        operations,
        commit_message,
        repo_type,
        revision,
        parent_commit,
    ):
        self._dataset(repo_type)
        if revision != "main" or parent_commit != self.main or not commit_message:
            raise ValueError("commit precondition drifted")
        self.create_commit_call_count += 1
        files = dict(self.commits[self.main])
        for operation in operations:
            files[operation.path_in_repo] = operation.path_or_fileobj.read()
        self.commits[COMMIT_B] = files
        self.titles[COMMIT_B] = commit_message
        self.parents[COMMIT_B] = parent_commit
        self.main = COMMIT_B
        if self.lose_commit:
            self.lose_commit = False
            if self.interpose_on_commit_loss:
                self.commits[INTERPOSED_COMMIT] = dict(files)
                self.titles[INTERPOSED_COMMIT] = commit_message
                self.parents[INTERPOSED_COMMIT] = COMMIT_B
                self.main = INTERPOSED_COMMIT
            raise ConnectionError("synthetic commit response loss")
        return SimpleNamespace(oid=COMMIT_B)

    def create_tag(
        self,
        _repo,
        *,
        tag,
        tag_message,
        revision,
        repo_type,
        exist_ok,
    ):
        self._dataset(repo_type)
        if tag != "formal-v1" or not tag_message or exist_ok or revision != COMMIT_B:
            raise ValueError("tag precondition drifted")
        self.create_tag_call_count += 1
        self.tag_commit = revision
        self.tag_object = TAG_OBJECT
        if self.lose_tag:
            self.lose_tag = False
            raise ConnectionError("synthetic tag response loss")

    def download(self, *, filename, revision, local_dir, **_kwargs):
        commit = self.tag_commit if revision == "formal-v1" else revision
        payload = self.commits[commit][filename]
        target = Path(local_dir).resolve() / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)

    @staticmethod
    def _dataset(repo_type):
        if repo_type != "dataset":
            raise ValueError("wrong repo type")


def _hooks(contract: _Contract, events: list[str], *, bad_label=False, fail_feature=False):
    feature_audit = {
        "development_semantic_decode_count": 0,
        "confirm_semantic_decode_count": 0,
    }
    label_audit = dict(feature_audit)
    if bad_label:
        label_audit["development_semantic_decode_count"] = 1

    def build_feature(_contract, _data):
        events.append("build_feature")
        return CacheArtifact(b"FEATURE", feature_audit)

    def read_feature(_contract, _payload):
        events.append("read_feature")
        if fail_feature:
            raise ValueError("synthetic feature readback failure")
        return feature_audit

    def build_label(_contract, _data):
        events.append("build_label")
        state = contract.local_state["ordered_states"]
        paths = {item["name"]: Path(item["path"]) for item in state}
        if not paths["feature_completion"].exists() or not paths["label_access_claim"].exists():
            raise AssertionError("label callback preceded durable feature/claim state")
        return CacheArtifact(b"LABEL", label_audit)

    def read_label(_contract, _payload):
        events.append("read_label")
        return label_audit

    def join(_contract, _feature, _label):
        events.append("join")
        return {"execution_operation_counts": dict(EXPECTED_OPERATIONS)}

    return FormalCacheHooks(build_feature, read_feature, build_label, read_label, join)


class FormalCacheRunnerTest(unittest.TestCase):
    def _execute(self, root, contract, hooks, api, *, mode="run"):
        data = root / "data"
        fresh = root / "fresh"
        data.mkdir(exist_ok=True)
        fresh.mkdir(exist_ok=True)
        with (
            mock.patch(
                "causalcache.gate_v1_formal_cache_runner.validate_execution_b_source",
                return_value=_source(),
            ),
            mock.patch(
                "causalcache.gate_v1_formal_cache_runner.validate_no_gpu_or_model_runtime",
                return_value={"device": "cpu"},
            ),
        ):
            return execute_formal_cache(
                mode=mode,
                contract=contract,
                hooks=hooks,
                api=api,
                download_fn=api.download,
                operation_factory=_Operation,
                expected_execution_b_git_commit=COMMIT_B,
                data_root=data,
                fresh_download_parent=fresh,
            )

    def test_phase_order_exact_three_and_completed_replay(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            api = _FakeHF()
            events = []
            first = self._execute(root, contract, _hooks(contract, events), api)
            self.assertEqual(
                events,
                ["build_feature", "read_feature", "build_label", "read_label", "join"],
            )
            self.assertEqual(first["remote_mutation_call_count"], 2)
            self.assertEqual(api.create_commit_call_count, 1)
            self.assertEqual(api.create_tag_call_count, 1)
            self.assertEqual(api.tag_commit, COMMIT_B)
            self.assertNotEqual(api.tag_commit, api.tag_object)
            targets = set(contract.destination["exact_three_targets"])
            self.assertEqual(set(api.commits[COMMIT_B]) - {".gitattributes"}, targets)
            sidecar = json.loads(api.commits[COMMIT_B]["formal/manifest.json"])
            sidecar_consumption = sidecar["formal_consumption"]
            self.assertFalse(
                sidecar_consumption[
                    "formal58_training_input_eligible_by_sidecar_alone"
                ]
            )
            self.assertFalse(
                sidecar_consumption["gate_training_authorized_by_sidecar_alone"]
            )
            self.assertTrue(sidecar_consumption["final_completion_required"])
            final = Path(contract.local_state["ordered_states"][-1]["path"])
            staged = Path(contract.local_state["ordered_states"][-2]["path"])
            before = (final.read_bytes(), final.stat().st_ino, staged.stat().st_ino)
            completion = json.loads(final.read_bytes())
            self.assertEqual(completion["status"], FINAL_COMPLETION_STATUS)
            self.assertTrue(completion["formal58_cache_loader_eligible"])
            self.assertTrue(completion["formal58_training_input_eligible"])
            self.assertTrue(completion["gate_training_authorized"])
            self.assertFalse(completion["gate_trained"])
            replay = self._execute(root, contract, _hooks(contract, []), api)
            after = (final.read_bytes(), final.stat().st_ino, staged.stat().st_ino)
            self.assertEqual(replay["remote_mutation_call_count"], 0)
            self.assertEqual(before, after)

    def test_completed_run_never_repairs_missing_tag(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            api = _FakeHF()
            self._execute(root, contract, _hooks(contract, []), api)
            api.tag_commit = None
            api.tag_object = None
            before = (api.create_commit_call_count, api.create_tag_call_count)
            with self.assertRaisesRegex(ValueError, "completed/read-only"):
                self._execute(root, contract, _hooks(contract, []), api)
            self.assertEqual(
                before,
                (api.create_commit_call_count, api.create_tag_call_count),
            )

    def test_completed_run_never_recreates_missing_local_cache(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            api = _FakeHF()
            self._execute(root, contract, _hooks(contract, []), api)
            feature = Path(contract.formats["feature"]["archive_path"])
            feature.unlink()
            with self.assertRaisesRegex(ValueError, "out of phase order"):
                self._execute(root, contract, _hooks(contract, []), api)
            self.assertFalse(feature.exists())

    def test_response_loss_recovers_without_duplicate_mutation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            api = _FakeHF(lose_commit=True, lose_tag=True)
            result = self._execute(root, contract, _hooks(contract, []), api)
            self.assertEqual(result["remote_mutation_call_count"], 2)
            self.assertEqual(api.create_commit_call_count, 1)
            self.assertEqual(api.create_tag_call_count, 1)

    def test_interposed_identical_response_loss_never_creates_tag(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            api = _FakeHF(
                lose_commit=True,
                interpose_on_commit_loss=True,
            )
            with self.assertRaisesRegex(ValueError, "direct child"):
                self._execute(root, contract, _hooks(contract, []), api)
            self.assertEqual(api.create_commit_call_count, 1)
            self.assertEqual(api.create_tag_call_count, 0)

    def test_feature_failure_never_calls_label_callback(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            events = []
            with self.assertRaisesRegex(ValueError, "feature readback"):
                self._execute(
                    root,
                    contract,
                    _hooks(contract, events, fail_feature=True),
                    _FakeHF(),
                )
            self.assertEqual(events, ["build_feature", "read_feature"])
            state = {item["name"]: Path(item["path"]) for item in contract.local_state["ordered_states"]}
            self.assertFalse(state["label_access_claim"].exists())

    def test_development_sentinel_fails_before_label_publication(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            with self.assertRaisesRegex(ValueError, "prohibited operation"):
                self._execute(root, contract, _hooks(contract, [], bad_label=True), _FakeHF())
            label_path = Path(contract.formats["label"]["archive_path"])
            self.assertFalse(label_path.exists())

    def test_partial_remote_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            api = _FakeHF()
            api.commits[COMMIT_A][contract.destination["exact_three_targets"][0]] = b"bad"
            with self.assertRaisesRegex(ValueError, "partial exact-three"):
                self._execute(root, contract, _hooks(contract, []), api)
            self.assertEqual(api.create_commit_call_count, 0)
            self.assertEqual(api.create_tag_call_count, 0)

    def test_validate_is_read_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            api = _FakeHF()
            self._execute(root, contract, _hooks(contract, []), api)
            before = (api.create_commit_call_count, api.create_tag_call_count)
            result = self._execute(root, contract, _hooks(contract, []), api, mode="validate")
            self.assertEqual(result["remote_mutation_call_count"], 0)
            self.assertEqual(before, (api.create_commit_call_count, api.create_tag_call_count))

    def test_runner_freeze_binds_source_paths_prerequisites_and_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            source = _source()
            validation = {
                "source_a": {
                    "git_commit": source.head,
                    "remote_main_git_commit": source.remote_main,
                    "branch": source.branch,
                    "origin_url": source.origin_url,
                    "source_inventory": list(source.source_inventory),
                    "loaded_module_inventory": list(source.loaded_module_inventory),
                }
            }
            expected_status = f"?? {RUNNER_FREEZE_PATH}\n".encode()
            with (
                mock.patch(
                    "causalcache.gate_v1_formal_cache_runner.validate_source_a",
                    return_value=validation,
                ),
                mock.patch(
                    "causalcache.gate_v1_formal_cache_runner._git",
                    return_value=expected_status,
                ),
            ):
                freeze = materialize_runner_freeze(contract)
            self.assertEqual(
                freeze["required_source_a_paths"],
                contract.data["source_freeze"]["required_source_a_paths"],
            )
            self.assertEqual(
                freeze["git_prerequisites"],
                contract.data["source_freeze"]["git_prerequisites"],
            )
            self.assertEqual(
                freeze["source_a_inventory_sha256"],
                sha256_bytes(
                    __import__("json").dumps(
                        list(source.source_inventory),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                ),
            )

    def test_transport_repair_runner_freeze_path_is_canonical_and_selected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = _contract(root)
            contract.data["source_freeze"]["execution_b_runner_freeze"][
                "path"
            ] = TRANSPORT_REPAIR_RUNNER_FREEZE_PATH
            formal_path = root / RUNNER_FREEZE_PATH
            formal_path.parent.mkdir(parents=True, exist_ok=True)
            formal_path.write_text("{}\n", encoding="utf-8")
            source = _source()
            with (
                mock.patch(
                    "causalcache.gate_v1_formal_cache_runner.validate_clean_pushed_source",
                    return_value=source,
                ),
                mock.patch(
                    "causalcache.gate_v1_formal_cache_runner.validate_no_gpu_or_model_runtime",
                    return_value={"device": "cpu"},
                ),
            ):
                validation = validate_source_a(contract)
            self.assertEqual(
                validation["source_a"]["git_commit"], source.head
            )

            runner_freeze = root / TRANSPORT_REPAIR_RUNNER_FREEZE_PATH
            runner_freeze.parent.mkdir(parents=True, exist_ok=True)
            runner_freeze.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be absent"):
                validate_source_a(contract)

            runner_freeze.unlink()
            source_validation = {
                "source_a": {
                    "git_commit": source.head,
                    "remote_main_git_commit": source.remote_main,
                    "branch": source.branch,
                    "origin_url": source.origin_url,
                    "source_inventory": list(source.source_inventory),
                    "loaded_module_inventory": list(source.loaded_module_inventory),
                }
            }
            with (
                mock.patch(
                    "causalcache.gate_v1_formal_cache_runner.validate_source_a",
                    return_value=source_validation,
                ),
                mock.patch(
                    "causalcache.gate_v1_formal_cache_runner._git",
                    return_value=(
                        f"?? {TRANSPORT_REPAIR_RUNNER_FREEZE_PATH}\n".encode()
                    ),
                ),
            ):
                freeze = materialize_runner_freeze(contract)
            self.assertEqual(
                freeze["execution_b_required_unique_diff"],
                [TRANSPORT_REPAIR_RUNNER_FREEZE_PATH],
            )
            self.assertTrue(runner_freeze.is_file())

    def test_runner_freeze_path_rejects_noncanonical_paths(self):
        for path in (
            RUNNER_FREEZE_PATH,
            TRANSPORT_REPAIR_RUNNER_FREEZE_PATH,
        ):
            contract = SimpleNamespace(
                data={
                    "source_freeze": {
                        "execution_b_runner_freeze": {"path": path}
                    }
                }
            )
            self.assertEqual(_runner_freeze_path(contract), path)
        invalid = SimpleNamespace(
            data={
                "source_freeze": {
                    "execution_b_runner_freeze": {
                        "path": "code/configs/arbitrary-runner-freeze.json"
                    }
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "runner-freeze path drifted"):
            _runner_freeze_path(invalid)

    def test_source_only_cli_rejects_execution_arguments(self):
        with self.assertRaises(SystemExit):
            _parser().parse_args(
                [
                    "validate-source",
                    "--repository-root",
                    "/tmp/repo",
                    "--hf-token-file",
                    "/tmp/token",
                ]
            )

    def test_cli_does_not_expose_cache_api_override(self):
        parser = _parser()
        choices = next(
            action.choices
            for action in parser._actions
            if getattr(action, "choices", None)
        )
        self.assertNotIn("--cache-api", choices["run"].format_help())

    def test_cli_real_input_inventory_includes_repaired_sidecar_once(self):
        root = Path(__file__).resolve().parents[2]
        contract = load_frozen_formal_cache_contract(repository_root=root)
        specs = _input_specs(contract)
        self.assertEqual(set(specs), set(cache_core.DOWNLOAD_KEYS))
        artifact, file_record = specs[cache_core.EXPANSION_LABEL_SIDECAR]
        self.assertTrue(file_record["path"].endswith(".json"))
        self.assertEqual(
            artifact["immutable_revision"],
            contract.inputs["repaired_expansion_restoration_labels"][
                "immutable_revision"
            ],
        )
        self.assertEqual(
            _input_specs(contract),
            _input_specs(contract, cache_api=cache_core),
        )

    def test_observed_operation_counts_are_derived_and_tamper_fails(self):
        zero = {
            key: 0
            for key in EXPECTED_OPERATIONS
            if EXPECTED_OPERATIONS[key] == 0
        }
        feature = {
            "counts": {
                "trajectory_semantic_decode_count": 58,
                "feature_state_count": 174,
                "candidate_feature_count": 522,
                "ocr_semantic_decode_count": 290,
            },
            "access_audit": zero,
            "operation_audit": zero,
        }
        label = {
            "counts": {
                "label_state_semantic_decode_count": 174,
                "distance_value_decode_count": 1624,
                "conditional_edge_count": 1682,
                "independent_target_count": 522,
            },
            "access_audit": zero,
            "operation_audit": zero,
        }
        join = {
            "joined_state_count": 174,
            "counts": {
                "trajectory_count": 58,
                "state_count": 174,
                "candidate_feature_count": 522,
                "distance_value_count": 1624,
                "conditional_edge_count": 1682,
                "independent_target_count": 522,
            },
            "access_audit": zero,
            "operation_audit": zero,
        }
        self.assertEqual(
            _observed_execution_counts(feature, label, join), EXPECTED_OPERATIONS
        )
        tampered = {**join, "joined_state_count": 173}
        with self.assertRaisesRegex(ValueError, "count audits disagree"):
            _observed_execution_counts(feature, label, tampered)

    def test_execution_runtime_fails_closed_on_env_python_and_machine_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            contract = _contract(Path(raw))
            runtime = contract.data["runtime_contract"]
            environment = {
                **runtime["thread_environment"],
                "NVIDIA_VISIBLE_DEVICES": "void",
                "CUDA_VISIBLE_DEVICES": "",
            }
            def enter_runtime(stack: ExitStack, *, version: str, env: Mapping[str, str]):
                stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
                stack.enter_context(
                    mock.patch(
                        "causalcache.gate_v1_formal_cache_runner.Path.glob",
                        return_value=[],
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "causalcache.gate_v1_formal_cache_runner.platform.python_implementation",
                        return_value="CPython",
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "causalcache.gate_v1_formal_cache_runner.platform.python_version",
                        return_value=version,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "causalcache.gate_v1_formal_cache_runner.platform.machine",
                        return_value="x86_64",
                    )
                )

            with ExitStack() as stack:
                enter_runtime(stack, version="3.12.3", env=environment)
                observed = validate_no_gpu_or_model_runtime(
                    require_environment=True, contract=contract
                )
                self.assertEqual(observed["python_version"], "3.12.3")
            with ExitStack() as stack:
                enter_runtime(
                    stack,
                    version="3.12.3",
                    env={**environment, "OMP_NUM_THREADS": "2"},
                )
                with self.assertRaisesRegex(ValueError, "runtime differs"):
                    validate_no_gpu_or_model_runtime(
                        require_environment=True, contract=contract
                    )
            with ExitStack() as stack:
                enter_runtime(stack, version="3.12.4", env=environment)
                with self.assertRaisesRegex(ValueError, "runtime differs"):
                    validate_no_gpu_or_model_runtime(
                        require_environment=True, contract=contract
                    )

    def test_execution_b_must_be_direct_single_parent_of_source_a(self):
        source = _source()
        freeze = {
            "source_a_git_commit": COMMIT_A,
            "source_blob_inventory": list(source.source_inventory),
            "loaded_module_inventory": list(source.loaded_module_inventory),
        }
        for runner_path in (
            RUNNER_FREEZE_PATH,
            TRANSPORT_REPAIR_RUNNER_FREEZE_PATH,
        ):
            contract = SimpleNamespace(
                repository_root=Path("/tmp"),
                data={
                    "source_freeze": {
                        "execution_b_runner_freeze": {"path": runner_path}
                    }
                },
            )

            def git_direct(_root, *arguments):
                if arguments[:3] == ("rev-list", "--parents", "-n"):
                    return f"{COMMIT_B} {COMMIT_A}\n".encode()
                if arguments[:2] == ("diff", "--name-only"):
                    return f"{runner_path}\n".encode()
                raise AssertionError(arguments)

            with (
                mock.patch(
                    "causalcache.gate_v1_formal_cache_runner.load_runner_freeze",
                    return_value=freeze,
                ),
                mock.patch(
                    "causalcache.gate_v1_formal_cache_runner.validate_clean_pushed_source",
                    return_value=source,
                ),
                mock.patch(
                    "causalcache.gate_v1_formal_cache_runner._git",
                    side_effect=git_direct,
                ),
            ):
                self.assertEqual(
                    validate_execution_b_source(
                        contract, expected_execution_b_git_commit=COMMIT_B
                    ),
                    source,
                )

        intervening = "5" * 40
        contract = SimpleNamespace(
            repository_root=Path("/tmp"),
            data={
                "source_freeze": {
                    "execution_b_runner_freeze": {"path": RUNNER_FREEZE_PATH}
                }
            },
        )

        def git_direct(_root, *arguments):
            if arguments[:3] == ("rev-list", "--parents", "-n"):
                return f"{COMMIT_B} {COMMIT_A}\n".encode()
            if arguments[:2] == ("diff", "--name-only"):
                return f"{RUNNER_FREEZE_PATH}\n".encode()
            raise AssertionError(arguments)

        def git_intervening(_root, *arguments):
            if arguments[:3] == ("rev-list", "--parents", "-n"):
                return f"{COMMIT_B} {intervening}\n".encode()
            return git_direct(_root, *arguments)

        with (
            mock.patch(
                "causalcache.gate_v1_formal_cache_runner.load_runner_freeze",
                return_value=freeze,
            ),
            mock.patch(
                "causalcache.gate_v1_formal_cache_runner.validate_clean_pushed_source",
                return_value=source,
            ),
            mock.patch(
                "causalcache.gate_v1_formal_cache_runner._git",
                side_effect=git_intervening,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "direct child"):
                validate_execution_b_source(
                    contract, expected_execution_b_git_commit=COMMIT_B
                )


if __name__ == "__main__":
    unittest.main()
