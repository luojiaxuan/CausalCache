from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from causalcache import gate_v1_fresh16_failure_decomposition_runner as runner


PARENT_TAG_OBJECT = runner.PARENT_TAG_OBJECT
DESTINATION_BASE = "9" * 40
DESTINATION_COMMIT = "a" * 40
DESTINATION_TAG_OBJECT = "b" * 40


class _Contract:
    def __init__(self, root: Path):
        self.data = {
            "parent_input": {
                "repo": runner.PARENT_REPO,
                "repo_type": "dataset",
                "immutable_revision": runner.PARENT_REVISION,
                "tag": runner.PARENT_TAG,
                "exact_three_targets": list(runner.PARENT_TARGETS),
            },
            "destination": {
                "repo": "gavinlaw/causalcache-fresh16-decomposition-test",
                "repo_type": "dataset",
                "private": True,
                "tag": "fresh16-decomposition-v1",
                "tag_message": "Freeze synthetic failure decomposition",
                "exact_three_targets": list(runner.CHILD_TARGETS),
                "commit_message": "Publish synthetic failure decomposition",
            },
            "local_first_state_machine": {
                "artifact_directory": str(root / "artifacts"),
                "state_directory": str(root / "state"),
                "state_file_mode": 0o600,
                "ordered_states": list(runner.ORDERED_LOCAL_STATES),
            },
            "routing_contract": {
                "final_routing": {
                    "pass_status": "ONE_V2_CONDITIONAL_RESCUE",
                    "fail_status": "NO_V2_CONDITIONAL_RESCUE",
                }
            },
        }
        self.sha256 = "c" * 64


class _Operation:
    def __init__(self, *, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class _NotFound(Exception):
    pass


class _FakeHF:
    def __init__(self, parent_files, *, destination_exists=False):
        self.parent_files = dict(parent_files)
        self.destination_exists = destination_exists
        self.destination_files = {".gitattributes": b"lfs\n"}
        self.destination_main = DESTINATION_BASE
        self.destination_tag_commit = None
        self.destination_tag_object = None
        self.download_calls = []
        self.create_repo_calls = 0
        self.create_commit_calls = 0
        self.create_tag_calls = 0
        self.commit_response_loss = False
        self.tag_response_loss = False

    @staticmethod
    def whoami():
        return {
            "name": "gavinlaw",
            "auth": {"type": "access_token", "accessToken": {"role": "write"}},
        }

    @staticmethod
    def _is_parent(repo):
        return repo == runner.PARENT_REPO

    def dataset_info(self, repo, *, revision):
        if self._is_parent(repo):
            resolved = runner.PARENT_REVISION
            if revision not in {runner.PARENT_REVISION, runner.PARENT_TAG}:
                raise _NotFound(revision)
            return SimpleNamespace(sha=resolved, private=True)
        if not self.destination_exists:
            raise _NotFound(repo)
        if revision == "main":
            resolved = self.destination_main
        elif revision == "fresh16-decomposition-v1":
            if self.destination_tag_commit is None:
                raise _NotFound(revision)
            resolved = self.destination_tag_commit
        elif revision in {DESTINATION_BASE, DESTINATION_COMMIT}:
            resolved = revision
        else:
            raise _NotFound(revision)
        return SimpleNamespace(sha=resolved, private=True)

    def list_repo_refs(self, repo, *, repo_type):
        assert repo_type == "dataset"
        if self._is_parent(repo):
            return SimpleNamespace(
                tags=[
                    SimpleNamespace(
                        name=runner.PARENT_TAG,
                        target_commit=PARENT_TAG_OBJECT,
                    )
                ]
            )
        if not self.destination_exists:
            raise _NotFound(repo)
        tags = []
        if self.destination_tag_commit is not None:
            tags.append(
                SimpleNamespace(
                    name="fresh16-decomposition-v1",
                    target_commit=self.destination_tag_object,
                )
            )
        return SimpleNamespace(tags=tags)

    def list_repo_files(self, repo, *, repo_type, revision):
        assert repo_type == "dataset"
        if self._is_parent(repo):
            return sorted(self.parent_files)
        if not self.destination_exists:
            raise _NotFound(repo)
        return sorted(self.destination_files)

    def list_repo_commits(self, repo, *, repo_type, revision):
        assert repo_type == "dataset"
        if self._is_parent(repo):
            return [
                SimpleNamespace(commit_id=runner.PARENT_REVISION),
                SimpleNamespace(
                    commit_id="9f0c61b9437773ca5d3f7e0cabd6e2a987e3c908"
                ),
            ]
        if not self.destination_exists:
            raise _NotFound(repo)
        values = [self.destination_main]
        if self.destination_main == DESTINATION_COMMIT:
            values.append(DESTINATION_BASE)
        return [SimpleNamespace(commit_id=item) for item in values]

    def create_repo(self, repo, *, repo_type, private, exist_ok):
        assert repo_type == "dataset" and private and not exist_ok
        if self.destination_exists:
            raise ValueError("repo exists")
        self.create_repo_calls += 1
        self.destination_exists = True
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
        assert repo_type == "dataset"
        assert revision == "main"
        assert parent_commit == DESTINATION_BASE
        assert commit_message
        self.create_commit_calls += 1
        for operation in operations:
            self.destination_files[operation.path_in_repo] = (
                operation.path_or_fileobj.read()
            )
        self.destination_main = DESTINATION_COMMIT
        if self.commit_response_loss:
            self.commit_response_loss = False
            raise ConnectionError("synthetic commit response loss")
        return SimpleNamespace(oid=DESTINATION_COMMIT)

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
        assert repo_type == "dataset"
        assert tag == "fresh16-decomposition-v1"
        assert tag_message and revision == DESTINATION_COMMIT and not exist_ok
        self.create_tag_calls += 1
        self.destination_tag_commit = revision
        self.destination_tag_object = DESTINATION_TAG_OBJECT
        if self.tag_response_loss:
            self.tag_response_loss = False
            raise ConnectionError("synthetic tag response loss")

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
        self.download_calls.append((repo_id, revision, filename))
        if self._is_parent(repo_id):
            payload = self.parent_files[filename]
        else:
            if not self.destination_exists:
                raise _NotFound(repo_id)
            payload = self.destination_files[filename]
        path = Path(local_dir).resolve() / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _parent_fixture():
    label_rows = []
    state_rows = []
    for index in range(48):
        source = f"source-{index // 3:02d}"
        state_id = f"{source}:decision:{index % 3 + 4}"
        label_rows.append(
            _canonical(
                {
                    "ordinal": index,
                    "source_id": source,
                    "state_id": state_id,
                }
            )
            + b"\n"
        )
        state_rows.append(
            _canonical(
                {
                    "ordinal": index,
                    "source_id": source,
                    "state_id": state_id,
                    "evaluation": {},
                }
            )
            + b"\n"
        )
    labels = b"".join(label_rows)
    states = b"".join(state_rows)
    label_binding = {
        "path": runner.PARENT_LABEL_PATH,
        "size_bytes": len(labels),
        "sha256": runner.sha256_bytes(labels),
    }
    state_binding = {
        "path": runner.PARENT_STATE_RECORDS_PATH,
        "size_bytes": len(states),
        "sha256": runner.sha256_bytes(states),
    }
    bundle = runner.pretty_json_bytes(
        {
            "schema_version": "1.0.0",
            "protocol_id": "causalcache_gate_v1_fresh16_evaluation_v1",
            "status": "FROZEN_GATE_V1_FRESH16_BUNDLE_MANIFEST_V1",
            "final_target_count": 13,
            "report_commit_embedded": False,
            "payload_files": [label_binding],
            "report_files_excluding_bundle": [state_binding],
        }
    )
    files = {
        runner.PARENT_BUNDLE_PATH: bundle,
        runner.PARENT_LABEL_PATH: labels,
        runner.PARENT_STATE_RECORDS_PATH: states,
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
    state_rows = [
        {
            "schema_version": "1.0.0",
            "protocol_id": runner.PROTOCOL_ID,
            "status": "FROZEN_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_V1",
            "ordinal": index,
            "source_id": f"source-{index // 3:02d}",
            "state_id": f"state-{index:02d}",
            "search_gap": 0.0,
        }
        for index in range(48)
    ]
    state = b"".join(_canonical(row) + b"\n" for row in state_rows)
    trajectory_rows = [
        {"source_id": f"source-{index:02d}", "state_count": 3}
        for index in range(16)
    ]
    trajectory_payload = b"".join(
        _canonical(row) + b"\n" for row in trajectory_rows
    )

    def reduce(labels, records, *, contract):
        events.append((labels, records, contract))
        return {
            "schema_version": "1.0.0",
            "protocol_id": runner.PROTOCOL_ID,
            "status": "FROZEN_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_V1",
            "contract_sha256": contract.sha256,
            "source_count": 16,
            "state_count": 48,
            "state_rows": state_rows,
            "state_rows_sha256": runner.sha256_bytes(state),
            "trajectory_rows": trajectory_rows,
            "trajectory_rows_sha256": runner.sha256_bytes(trajectory_payload),
            "trajectory_rows_size_bytes": len(trajectory_payload),
            "decision_tree": {
                "outcome": "NO_V2_CONDITIONAL_RESCUE",
                "v1_verdict_unchanged": True,
                "fresh16_is_consumed_diagnostic_only": True,
                "confirm_matched_nll_closed_loop_authorized": False,
            },
            "operation_counts": {
                "scope": "pure_reducer_after_two_sealed_48_record_decodes",
                "sealed_fresh16_label_state_decode_count": 48,
                "sealed_parent_state_record_decode_count": 48,
                **{
                    name: 0
                    for name in (
                        "file_read",
                        "file_write",
                        "network",
                        "torch_import",
                        "gpu",
                        "model_load",
                        "model_forward",
                        "policy_forward",
                        "fresh16_raw_source_access_count",
                        "upstream_raw_label_read_count",
                        "legacy_dev5_semantic_decode_count",
                        "confirm20_access_count",
                        "matched_nll_evaluation_count",
                        "closed_loop_episode_count",
                        "gate_training_step_count",
                        "hf_mutation",
                    )
                },
            },
        }

    return reduce


def _execute(root, contract, api, reducer, *, mode="run", output_directory=None):
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
        output_directory=output_directory,
    )


def test_selective_parent_read_and_child_exact_three_publication():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        events = []
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            result = _execute(root, contract, api, _reducer(events))
        assert result["status"] == runner.RUN_STATUS
        assert result["parent_selective_download_count"] == 3
        assert result["child_immutable_download_count"] == 6
        assert len(events) == 1
        assert events[0][0] == files[runner.PARENT_LABEL_PATH]
        assert events[0][1] == files[runner.PARENT_STATE_RECORDS_PATH]
        assert api.create_repo_calls == 1
        assert api.create_commit_calls == 1
        assert api.create_tag_calls == 1
        assert set(api.destination_files) - {".gitattributes"} == set(
            runner.CHILD_TARGETS
        )
        assert api.destination_tag_commit == DESTINATION_COMMIT
        assert api.destination_tag_object != api.destination_tag_commit
        parent_downloads = [item for item in api.download_calls if item[0] == runner.PARENT_REPO]
        assert [item[2] for item in parent_downloads] == list(runner.PARENT_TARGETS)
        assert all(item[1] == runner.PARENT_REVISION for item in parent_downloads)
        child_downloads = [item for item in api.download_calls if item[0] != runner.PARENT_REPO]
        assert len(child_downloads) == 6
        assert {item[1] for item in child_downloads} == {
            DESTINATION_COMMIT,
            "fresh16-decomposition-v1",
        }


def test_completed_validate_replays_without_remote_or_local_mutation():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            _execute(root, contract, api, _reducer([]))
            before = (
                api.create_repo_calls,
                api.create_commit_calls,
                api.create_tag_calls,
                {
                    path: (payload, (root / "artifacts" / path).stat().st_ino)
                    for path, payload in api.destination_files.items()
                    if path in runner.CHILD_TARGETS
                },
            )
            result = _execute(root, contract, api, _reducer([]), mode="validate")
            after = (
                api.create_repo_calls,
                api.create_commit_calls,
                api.create_tag_calls,
                {
                    path: (payload, (root / "artifacts" / path).stat().st_ino)
                    for path, payload in api.destination_files.items()
                    if path in runner.CHILD_TARGETS
                },
            )
        assert result["status"] == runner.VALIDATE_STATUS
        assert result["remote_mutation_call_count"] == 0
        assert result["local"]["local_write_count"] == 0
        assert before == after


@pytest.mark.parametrize("target", runner.PARENT_TARGETS)
def test_parent_byte_drift_fails_before_reducer_or_destination_mutation(target):
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        files[target] += b"tamper"
        api = _FakeHF(files)
        events = []
        with (
            mock.patch.object(runner, "PARENT_BINDINGS", bindings),
            pytest.raises(ValueError, match="immutable binding"),
        ):
            _execute(root, contract, api, _reducer(events))
        assert not events
        assert api.create_repo_calls == api.create_commit_calls == api.create_tag_calls == 0


def test_parent_tag_mismatch_fails_before_any_download():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        original = api.dataset_info

        def mismatched(repo, *, revision):
            if repo == runner.PARENT_REPO and revision == runner.PARENT_TAG:
                return SimpleNamespace(sha="d" * 40, private=True)
            return original(repo, revision=revision)

        api.dataset_info = mismatched
        with (
            mock.patch.object(runner, "PARENT_BINDINGS", bindings),
            pytest.raises(ValueError, match="revision/tag identity"),
        ):
            _execute(root, contract, api, _reducer([]))
        assert not api.download_calls
        assert api.create_repo_calls == 0


def test_reducer_failure_never_creates_local_or_remote_child():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)

        def failure(*_args, **_kwargs):
            raise RuntimeError("synthetic reducer failure")

        with (
            mock.patch.object(runner, "PARENT_BINDINGS", bindings),
            pytest.raises(RuntimeError, match="synthetic reducer"),
        ):
            _execute(root, contract, api, failure)
        assert not (root / "artifacts").exists()
        assert api.create_repo_calls == api.create_commit_calls == api.create_tag_calls == 0


def test_partial_child_remote_fails_closed_without_repair():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files, destination_exists=True)
        api.destination_files[runner.CHILD_STATE_PATH] = b"partial\n"
        with (
            mock.patch.object(runner, "PARENT_BINDINGS", bindings),
            pytest.raises(ValueError, match="partial exact-three"),
        ):
            _execute(root, contract, api, _reducer([]))
        assert api.create_commit_calls == api.create_tag_calls == 0


def test_completed_remote_tamper_fails_without_second_commit():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            _execute(root, contract, api, _reducer([]))
            api.destination_files[runner.CHILD_REPORT_PATH] += b"tamper"
            with pytest.raises(ValueError, match="readback differs"):
                _execute(root, contract, api, _reducer([]), mode="validate")
        assert api.create_commit_calls == api.create_tag_calls == 1


def test_validate_never_creates_missing_tag():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            _execute(root, contract, api, _reducer([]))
            api.destination_tag_commit = None
            api.destination_tag_object = None
            with pytest.raises(ValueError, match="tag is absent"):
                _execute(root, contract, api, _reducer([]), mode="validate")
        assert api.create_tag_calls == 1


def test_response_loss_is_recovered_without_duplicate_commit_or_tag():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        api.commit_response_loss = True
        api.tag_response_loss = True
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            result = _execute(root, contract, api, _reducer([]))
        assert result["status"] == runner.RUN_STATUS
        assert api.create_commit_calls == 1
        assert api.create_tag_calls == 1


def test_local_exact_three_rejects_extra_file_and_symlink():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            _execute(root, contract, api, _reducer([]))
            extra = root / "artifacts" / "extra"
            extra.write_bytes(b"x")
            with pytest.raises(ValueError, match="not exact-three"):
                _execute(root, contract, api, _reducer([]), mode="validate")
            extra.unlink()
            target = root / "artifacts" / runner.CHILD_REPORT_PATH
            payload = target.read_bytes()
            target.unlink()
            backing = root / "backing"
            backing.write_bytes(payload)
            target.symlink_to(backing)
            with pytest.raises(ValueError, match="missing or unsafe"):
                _execute(root, contract, api, _reducer([]), mode="validate")


def test_child_manifest_binds_parent_and_forbidden_zero_counts():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            _execute(root, contract, api, _reducer([]))
        manifest = json.loads(api.destination_files[runner.CHILD_BUNDLE_PATH])
        assert manifest["status"] == runner.BUNDLE_STATUS
        assert manifest["exact_target_count"] == 3
        assert manifest["report_commit_embedded"] is False
        assert manifest["source_execution"] == {}
        assert manifest["parent"]["selective_target_count"] == 3
        assert [item["path"] for item in manifest["parent"]["selective_files"]] == sorted(
            runner.PARENT_TARGETS
        )
        assert set(manifest["prohibited_access"].values()) == {0}


def test_local_state_machine_is_exact_nine_and_final_is_hard_link():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            _execute(root, contract, api, _reducer([]))
        state = root / "state"
        observed = sorted(path.name for path in state.iterdir())
        expected = [
            f"{index:02d}-{name.replace('_', '-')}.json"
            for index, name in enumerate(runner.ORDERED_LOCAL_STATES)
        ]
        assert observed == expected
        staging = state / expected[-2]
        final = state / expected[-1]
        assert staging.stat().st_ino == final.stat().st_ino
        assert staging.stat().st_dev == final.stat().st_dev
        assert staging.read_bytes() == final.read_bytes()


def test_write_identity_failure_precedes_child_repo_mutation():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        api.whoami = lambda: {
            "name": "gavinlaw",
            "auth": {"type": "access_token", "accessToken": {"role": "read"}},
        }
        with (
            mock.patch.object(runner, "PARENT_BINDINGS", bindings),
            pytest.raises(ValueError, match="write access"),
        ):
            _execute(root, contract, api, _reducer([]))
        assert api.create_repo_calls == api.create_commit_calls == api.create_tag_calls == 0


def test_real_contract_section_layout_is_supported():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        parent = contract.data.pop("parent_input")
        contract.data["parent_primary"] = {
            "hf": {
                **parent,
                "report_commit": runner.PARENT_REVISION,
                "tag_resolved_commit": runner.PARENT_REVISION,
                "annotated_tag_object": runner.PARENT_TAG_OBJECT,
            }
        }
        contract.data["input_contract"] = {
            "exact_force_download_targets": [
                {"path": path} for path in runner.PARENT_TARGETS
            ]
        }
        contract.data["output_contract"] = {
            "exact_targets": list(runner.CHILD_TARGETS)
        }
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            result = _execute(root, contract, api, _reducer([]))
        assert result["status"] == runner.RUN_STATUS


def test_full_reducer_payload_is_split_without_duplicate_state_rows():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        with mock.patch.object(runner, "PARENT_BINDINGS", bindings):
            _execute(root, contract, api, _reducer([]))
        state_payload = api.destination_files[runner.CHILD_STATE_PATH]
        report = json.loads(api.destination_files[runner.CHILD_REPORT_PATH])
        assert "state_rows" not in report
        assert report["state_rows_sha256"] == runner.sha256_bytes(state_payload)
        assert report["state_rows_size_bytes"] == len(state_payload)
        assert report["state_rows_record_count"] == 48


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract.data["parent_input"].__setitem__(
            "immutable_revision", "0" * 40
        ),
        lambda contract: contract.data["parent_input"]["exact_three_targets"].append(
            "fresh16-eval/v1/caches/feature-states-v1.jsonl"
        ),
        lambda contract: contract.data["destination"].__setitem__("private", False),
        lambda contract: contract.data["destination"]["exact_three_targets"].reverse(),
    ],
)
def test_contract_identity_and_exact_target_drift_fail_closed(mutation):
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        mutation(contract)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        with (
            mock.patch.object(runner, "PARENT_BINDINGS", bindings),
            pytest.raises(ValueError),
        ):
            _execute(root, contract, api, _reducer([]))
        assert api.create_repo_calls == api.create_commit_calls == api.create_tag_calls == 0


def test_state_output_requires_exactly_48_strict_json_rows():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        malformed = b"{}\n" * 47

        def bad(*_args, **_kwargs):
            return {"state_decomposition": malformed, "report": {"ok": True}}

        with (
            mock.patch.object(runner, "PARENT_BINDINGS", bindings),
            pytest.raises(ValueError, match="exactly 48"),
        ):
            _execute(root, contract, api, bad)
        assert api.create_repo_calls == 0


def test_download_path_escape_is_rejected():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        contract = _Contract(root)
        files, bindings = _parent_fixture()
        api = _FakeHF(files)
        original = api.download
        (root / "fresh").mkdir()

        def escape(**kwargs):
            original(**kwargs)
            return str(root / "outside")

        with (
            mock.patch.object(runner, "PARENT_BINDINGS", bindings),
            pytest.raises(ValueError, match="noncanonical"),
        ):
            runner.execute_failure_decomposition(
                mode="run",
                contract=contract,
                api=api,
                download_fn=escape,
                operation_factory=_Operation,
                fresh_download_parent=root / "fresh",
                reducer=_reducer([]),
            )
