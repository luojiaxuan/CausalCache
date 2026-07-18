from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from causalcache import independent_confirm_failure_decomposition_runner as runner
from causalcache.independent_confirm_artifact import (
    fixed_report_bytes,
    state_records_jsonl_bytes,
)
from causalcache.independent_confirm_data import (
    CONFIRM_DECISION_STEP_ID,
    CONFIRM_SOURCE_IDS,
)


CHILD_BASE = "7" * 40
CHILD_COMMIT = "8" * 40
CHILD_TAG_OBJECT = "9" * 40


class _Contract:
    def __init__(self, root: Path):
        self.data = {
            "protocol_id": runner.PROTOCOL_ID,
            "parent_confirm": {
                "hf": {
                    "repo": runner.PARENT_REPO,
                    "repo_type": "dataset",
                    "private": True,
                    "tag": runner.PARENT_TAG,
                    "annotated_tag_object": runner.PARENT_TAG_OBJECT,
                    "base_commit": runner.PARENT_BASE_COMMIT,
                    "payload_commit": runner.PARENT_PAYLOAD_COMMIT,
                    "report_commit": runner.PARENT_REVISION,
                    "tag_resolved_commit": runner.PARENT_REVISION,
                }
            },
            "input_contract": {
                "exact_force_download_targets": [
                    {"path": path} for path in runner.PARENT_TARGETS
                ]
            },
            "destination": {
                "repo": runner.CHILD_REPO,
                "repo_type": "dataset",
                "private": True,
                "tag": runner.CHILD_TAG,
                "commit_title": "Publish synthetic confirm decomposition",
                "tag_message": "Freeze synthetic confirm decomposition",
            },
            "output_contract": {"exact_targets": list(runner.CHILD_TARGETS)},
            "runtime_contract": {
                "cpu_only": True,
                "gpu_count": 0,
                "model_runtime_allowed": False,
            },
            "local_first_state_machine": {
                "artifact_directory": str(root / "artifact")
            },
        }
        self.sha256 = "a" * 64


class _Operation:
    def __init__(self, *, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class _NotFound(Exception):
    pass


class _FakeHF:
    def __init__(self, parent_files):
        self.parent_files = dict(parent_files)
        self.child_exists = False
        self.child_main = CHILD_BASE
        self.child_files = {".gitattributes": b"lfs\n"}
        self.child_tag_commit = None
        self.child_tag_object = None
        self.create_repo_count = 0
        self.create_commit_count = 0
        self.create_tag_count = 0

    @staticmethod
    def whoami():
        return {
            "name": "gavinlaw",
            "auth": {"accessToken": {"role": "write"}},
        }

    def dataset_info(self, repo, *, revision):
        if repo == runner.PARENT_REPO:
            if revision not in {runner.PARENT_REVISION, runner.PARENT_TAG}:
                raise _NotFound(revision)
            return SimpleNamespace(sha=runner.PARENT_REVISION, private=True)
        if repo != runner.CHILD_REPO or not self.child_exists:
            raise _NotFound(repo)
        if revision == "main":
            resolved = self.child_main
        elif revision == runner.CHILD_TAG and self.child_tag_commit is not None:
            resolved = self.child_tag_commit
        elif revision in {CHILD_BASE, CHILD_COMMIT}:
            resolved = revision
        else:
            raise _NotFound(revision)
        return SimpleNamespace(sha=resolved, private=True)

    def list_repo_refs(self, repo, *, repo_type):
        assert repo_type == "dataset"
        if repo == runner.PARENT_REPO:
            return SimpleNamespace(
                tags=[
                    SimpleNamespace(
                        name=runner.PARENT_TAG,
                        target_commit=runner.PARENT_TAG_OBJECT,
                    )
                ]
            )
        if not self.child_exists:
            raise _NotFound(repo)
        tags = []
        if self.child_tag_commit is not None:
            tags.append(
                SimpleNamespace(
                    name=runner.CHILD_TAG,
                    target_commit=self.child_tag_object,
                )
            )
        return SimpleNamespace(tags=tags)

    def list_repo_files(self, repo, *, repo_type, revision):
        assert repo_type == "dataset"
        if repo == runner.PARENT_REPO:
            return sorted(self.parent_files)
        if not self.child_exists:
            raise _NotFound(repo)
        return sorted(self.child_files)

    def list_repo_commits(self, repo, *, repo_type, revision):
        assert repo_type == "dataset"
        if repo == runner.PARENT_REPO:
            values = (
                runner.PARENT_REVISION,
                runner.PARENT_PAYLOAD_COMMIT,
                runner.PARENT_BASE_COMMIT,
            )
        elif self.child_exists:
            values = (self.child_main, CHILD_BASE)
        else:
            raise _NotFound(repo)
        return [SimpleNamespace(commit_id=value) for value in values]

    def create_repo(self, repo, *, repo_type, private, exist_ok):
        assert repo == runner.CHILD_REPO
        assert repo_type == "dataset" and private and not exist_ok
        self.child_exists = True
        self.create_repo_count += 1
        return SimpleNamespace(repo_id=repo)

    def create_commit(
        self,
        repo,
        *,
        repo_type,
        revision,
        parent_commit,
        commit_message,
        operations,
    ):
        assert repo == runner.CHILD_REPO and repo_type == "dataset"
        assert revision == "main" and parent_commit == CHILD_BASE and commit_message
        for operation in operations:
            self.child_files[operation.path_in_repo] = operation.path_or_fileobj.read()
        self.child_main = CHILD_COMMIT
        self.create_commit_count += 1
        return SimpleNamespace(oid=CHILD_COMMIT)

    def create_tag(
        self,
        repo,
        *,
        repo_type,
        tag,
        tag_message,
        revision,
        exist_ok,
    ):
        assert repo == runner.CHILD_REPO and repo_type == "dataset"
        assert tag == runner.CHILD_TAG and tag_message and not exist_ok
        self.child_tag_commit = revision
        self.child_tag_object = CHILD_TAG_OBJECT
        self.create_tag_count += 1

    def download(
        self,
        *,
        repo_id,
        repo_type,
        filename,
        revision,
        local_dir,
        force_download,
    ):
        assert repo_type == "dataset" and force_download
        source = self.parent_files if repo_id == runner.PARENT_REPO else self.child_files
        path = Path(local_dir).resolve() / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source[filename])
        return str(path)


def _parent_fixture():
    raw_records = []
    report_records = []
    for ordinal, source_id in enumerate(CONFIRM_SOURCE_IDS):
        state_id = f"{source_id}:decision_step:{CONFIRM_DECISION_STEP_ID:03d}"
        raw_records.append(
            {
                "state": {
                    "ordinal": ordinal,
                    "source_id": source_id,
                    "state_id": state_id,
                }
            }
        )
        report_records.append({"source_id": source_id, "state_id": state_id})
    raw = state_records_jsonl_bytes(raw_records)
    fixed = fixed_report_bytes(
        {
            "status": "NO_GO_INDEPENDENT_CONFIRM",
            "evaluation_performed": True,
            "fixed_state_denominator": 20,
            "reference": {},
            "bootstrap": {},
            "metrics": {},
            "records": report_records,
            "gate_checks": {},
            "go": False,
        }
    )
    raw_binding = {
        "path": runner.PARENT_RAW_STATE_PATH,
        "size_bytes": len(raw),
        "sha256": runner.sha256_bytes(raw),
    }
    fixed_binding = {
        "path": runner.PARENT_FIXED_REPORT_PATH,
        "size_bytes": len(fixed),
        "sha256": runner.sha256_bytes(fixed),
    }
    bundle = runner.pretty_json_bytes(
        {
            "schema_version": "1.0.0",
            "protocol_id": "causalcache_independent_confirm_closed_loop_v1",
            "status": "BUNDLED_INDEPENDENT_CONFIRM20_REPORT_V1",
            "payload_commit": runner.PARENT_PAYLOAD_COMMIT,
            "payload_inventory": [],
            "payload_inventory_sha256": "0" * 64,
            "report_inventory_without_bundle": [fixed_binding, raw_binding],
            "report_inventory_without_bundle_sha256": "1" * 64,
        }
    )
    files = {
        runner.PARENT_BUNDLE_PATH: bundle,
        runner.PARENT_RAW_STATE_PATH: raw,
        runner.PARENT_FIXED_REPORT_PATH: fixed,
    }
    bindings = {
        path: {
            "size_bytes": len(payload),
            "sha256": runner.sha256_bytes(payload),
        }
        for path, payload in files.items()
    }
    return files, bindings


def _reducer(events):
    def reduce(raw, fixed):
        events.append((raw, fixed))
        return {
            "schema_version": "1.0.0",
            "protocol_id": runner.PROTOCOL_ID,
            "status": runner.DECOMPOSITION_STATUS,
            "inputs": {
                "raw_state_records_sha256": runner.sha256_bytes(raw),
                "fixed_report_sha256": runner.sha256_bytes(fixed),
                "state_count": 20,
            },
            "aggregate": {"methods": {}},
            "comparisons": {},
            "decision": {
                "status": "INCONCLUSIVE_ORACLE_INDEPENDENT_CONFIRM_DECOMPOSITION"
            },
            "state_rows": [
                {
                    "ordinal": ordinal,
                    "source_id": source_id,
                    "state_id": f"{source_id}:decision_step:006",
                }
                for ordinal, source_id in enumerate(CONFIRM_SOURCE_IDS)
            ],
            "operation_counts": {
                "file_read_count": 0,
                "network_call_count": 0,
                "gpu_call_count": 0,
            },
        }

    return reduce


def _execute(root, contract, api, reducer, *, mode):
    fresh = root / "fresh"
    fresh.mkdir(exist_ok=True)
    return runner.execute_failure_decomposition(
        mode=mode,
        contract=contract,
        api=api,
        download_fn=api.download,
        operation_factory=_Operation,
        fresh_download_parent=fresh,
        reducer=reducer,
    )


def test_run_then_validate_replays_parent_and_child_exact_three():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        events = []
        reducer = _reducer(events)
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            result = _execute(root, contract, api, reducer, mode="run")
            replay = _execute(root, contract, api, reducer, mode="validate")
        assert result["status"] == runner.RUN_STATUS
        assert replay["status"] == runner.VALIDATE_STATUS
        assert result["parent_selective_download_count"] == 3
        assert result["child_immutable_download_count"] == 6
        assert replay["remote_mutation_count"] == 0
        assert api.create_repo_count == 1
        assert api.create_commit_count == 1
        assert api.create_tag_count == 1
        assert len(events) == 2
        assert set(api.child_files) - {".gitattributes"} == set(runner.CHILD_TARGETS)
        assert result["gpu_model_checkpoint_policy_restoration_training_count"] == 0


def test_parent_hash_mismatch_fails_before_reducer_or_child_mutation():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        files[runner.PARENT_RAW_STATE_PATH] += b"tamper"
        api = _FakeHF(files)
        events = []
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            with pytest.raises(ValueError, match="frozen bindings"):
                _execute(root, contract, api, _reducer(events), mode="run")
        assert events == []
        assert api.create_repo_count == 0


def test_child_manifest_binds_parent_without_copying_parent_raw_bytes():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            result = _execute(root, contract, api, _reducer([]), mode="run")
        manifest = json.loads(api.child_files[runner.CHILD_BUNDLE_PATH])
        assert manifest["parent"]["revision"] == runner.PARENT_REVISION
        assert manifest["parent"]["payload_commit"] == runner.PARENT_PAYLOAD_COMMIT
        assert manifest["parent_verdict_unchanged"] is True
        assert manifest["operation_counts"]["gpu_operation_count"] == 0
        assert runner.PARENT_RAW_STATE_PATH not in api.child_files
        assert result["closed_loop_matched_nll_sealed_test_authorized"] is False


def test_validate_contract_rejects_non_cpu_runtime():
    with tempfile.TemporaryDirectory() as raw:
        contract = _Contract(Path(raw))
        contract.data["runtime_contract"]["gpu_count"] = 1
        with pytest.raises(ValueError, match="CPU-only"):
            runner.validate_contract_identity(contract)
