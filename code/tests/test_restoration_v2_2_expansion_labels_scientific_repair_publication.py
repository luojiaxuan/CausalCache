from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from causalcache.restoration_v2_2_expansion_labels_scientific_repair_publication import (
    CANONICAL_CONFIG_PATH,
    CLAIM_STATUS,
    COMPLETION_STATUS,
    DESTINATION,
    FROZEN_CONFIG_SHA256,
    PAIR_COMMIT_TITLE,
    REMOTE_BYTE_IDENTICAL_UNTAGGED,
    REMOTE_EMPTY,
    REMOTE_TAGGED_BYTE_IDENTICAL,
    ScientificRepairPublicationContract,
    canonical_json_bytes,
    load_frozen_publication_contract,
    prepare_scientific_repair_publication,
    pretty_json_bytes,
    publish_scientific_repair_archive,
    sha256_bytes,
)
from scripts import (
    manage_restoration_v2_2_expansion_labels_scientific_repair_publication as manager,
)
from causalcache import (
    restoration_v2_2_expansion_labels_scientific_repair_publication as publication,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "1" * 40
PAIR_COMMIT = "2" * 40
TAG_OBJECT = "3" * 40
MOVED_COMMIT = "4" * 40
SOURCE_COMMIT = "a" * 40
RESULT_COMMIT = "b" * 40
PUBLICATION_COMMIT = "d" * 40
RUNNER_SHA = "c" * 64


def _operation_factory(*, path_in_repo: str, path_or_fileobj: io.BytesIO):
    return SimpleNamespace(path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj)


def _ustar(files: dict[str, bytes], prefix: str) -> bytes:
    destination = io.BytesIO()
    with tarfile.open(fileobj=destination, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for relative in sorted(files):
            payload = files[relative]
            info = tarfile.TarInfo(f"{prefix}/{relative}")
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return destination.getvalue()


def _source_identity(summary_sha: str) -> dict[str, object]:
    return {
        "publication_source_git_commit": PUBLICATION_COMMIT,
        "origin_main_git_commit": PUBLICATION_COMMIT,
        "externally_resolved_remote_main_git_commit": PUBLICATION_COMMIT,
        "branch": "main",
        "origin_url": "https://github.com/luojiaxuan/CausalCache.git",
        "source_path_count": 6,
        "source_inventory_sha256": "e" * 64,
        "repair_source_git_commit": SOURCE_COMMIT,
        "repair_result_git_commit": RESULT_COMMIT,
        "repair_result_summary_sha256": summary_sha,
    }


def _fixture(root: Path):
    prefix = "synthetic-repaired-labels"
    manifest = {
        "status": "IMMUTABLE_REPAIRED_EXPANSION_LABEL_PAYLOAD_MANIFEST_V1",
        "runner_config_sha256": RUNNER_SHA,
        "source_git_commit": SOURCE_COMMIT,
        "original_attempt_remains_invalid": True,
        "producer_reclassified": False,
    }
    audit = {
        "status": "VALIDATED_EXPANSION_LABEL_SCIENTIFIC_REPAIR_AUDIT_V1",
        "formal_scientific_repair_pass": True,
        "runner_config_sha256": RUNNER_SHA,
        "source": {"git_commit": SOURCE_COMMIT},
        "original_invalid_attempt": {"producer_reclassified": False},
        "formal_consumption": {"gate_training_unlocked": False},
    }
    files = {
        "audit.json": pretty_json_bytes(audit),
        "derived_labels.jsonl": b'{"state":0}\n',
        "manifest.json": pretty_json_bytes(manifest),
        "raw_states.jsonl": b'{"state":0}\n',
    }
    inventory = [
        {
            "path": name,
            "sha256": sha256_bytes(files[name]),
            "size_bytes": len(files[name]),
        }
        for name in sorted(files)
    ]
    archive = _ustar(files, prefix)
    archive_path = root / "repaired.tar"
    archive_path.write_bytes(archive)
    claim = pretty_json_bytes(
        {
            "protocol_id": "synthetic-runner",
            "runner_config_sha256": RUNNER_SHA,
            "status": "CLAIMED_LEDGER_NEUTRAL_EXPANSION_LABEL_SCIENTIFIC_REPAIR_V1",
        }
    )
    claim_path = root / "producer.claim.json"
    claim_path.write_bytes(claim)
    os.chmod(claim_path, 0o600)
    completion = pretty_json_bytes(
        {
            "archive_sha256": sha256_bytes(archive),
            "archive_size_bytes": len(archive),
            "claim_sha256": sha256_bytes(claim),
            "completion_created_after_strict_output_readback": True,
            "formal_consumption": {"gate_training_unlocked": False},
            "member_count": len(inventory),
            "member_inventory": inventory,
            "original_attempt_reclassified": False,
            "runner_config_sha256": RUNNER_SHA,
            "source_git_commit": SOURCE_COMMIT,
            "status": "COMPLETED_LEDGER_NEUTRAL_EXPANSION_LABEL_SCIENTIFIC_REPAIR_V1",
            "tree_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        }
    )
    completion_path = root / "producer.completion.json"
    completion_path.write_bytes(completion)
    os.chmod(completion_path, 0o600)
    summary = {
        "archive": {
            "sha256": sha256_bytes(archive),
            "size_bytes": len(archive),
            "tree_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        },
        "formal_consumption": {"gate_training_unlocked": False},
        "formal_scientific_repair_pass": True,
        "original_invalid_attempt": {"producer_reclassified": False},
        "source": {
            "git_commit": SOURCE_COMMIT,
            "runner_config_sha256": RUNNER_SHA,
        },
        "state_files": {
            "claim": {"sha256": sha256_bytes(claim)},
            "completion": {"sha256": sha256_bytes(completion)},
        },
    }
    summary_bytes = pretty_json_bytes(summary)
    summary_path = root / "result-summary.json"
    summary_path.write_bytes(summary_bytes)
    state_root = root / "publication-state"
    state_root.mkdir()
    data = {
        "schema_version": "1.0.0",
        "protocol_id": (
            "causalcache_restoration_v2_2_expansion_exact_labels_"
            "scientific_repair_publication_v1"
        ),
        "status": "source_only_frozen_before_repaired_private_hf_publication",
        "destination": dict(DESTINATION),
        "scientific_repair_source": {
            "protocol_id": (
                "causalcache_restoration_v2_2_expansion_exact_labels_"
                "scientific_repair_runner_v1"
            ),
            "source_git_commit": SOURCE_COMMIT,
            "result_commit": RESULT_COMMIT,
            "runner_config": {"path": "runner.json", "sha256": RUNNER_SHA},
            "result_summary": {
                "path": summary_path.name,
                "sha256": sha256_bytes(summary_bytes),
                "size_bytes": len(summary_bytes),
            },
            "archive": {
                "path": str(archive_path),
                "sha256": sha256_bytes(archive),
                "size_bytes": len(archive),
                "member_count": len(inventory),
                "member_inventory": inventory,
                "tree_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
                "archive_member_prefix": prefix,
            },
            "claim": {
                "path": str(claim_path),
                "sha256": sha256_bytes(claim),
                "size_bytes": len(claim),
                "mode": 0o600,
                "status": (
                    "CLAIMED_LEDGER_NEUTRAL_EXPANSION_LABEL_SCIENTIFIC_REPAIR_V1"
                ),
            },
            "completion": {
                "path": str(completion_path),
                "sha256": sha256_bytes(completion),
                "size_bytes": len(completion),
                "mode": 0o600,
                "status": (
                    "COMPLETED_LEDGER_NEUTRAL_EXPANSION_LABEL_SCIENTIFIC_REPAIR_V1"
                ),
            },
            "formal_audit_status": (
                "VALIDATED_EXPANSION_LABEL_SCIENTIFIC_REPAIR_AUDIT_V1"
            ),
            "manifest_status": (
                "IMMUTABLE_REPAIRED_EXPANSION_LABEL_PAYLOAD_MANIFEST_V1"
            ),
        },
        "local_state": {
            "claim_path": str(state_root / "publication.claim.json"),
            "remote_base_receipt_path": str(state_root / "publication.remote-base.json"),
            "completion_staging_path": str(
                state_root / "publication.completion.staged.json"
            ),
            "completion_seal_path": str(state_root / "publication.completion.json"),
            "file_mode": 0o600,
            "o_excl_first_write": True,
            "retained_completion_staging_required": True,
            "same_bytes_reusable": True,
        },
        "formal_consumption": {
            "gate_training_unlocked_before_publication_completion": False,
            "gate_training_unlocked_by_sidecar_alone": False,
            "original_invalid_forensic_formal_label_loader_eligible": False,
            "producer_attempt_reclassified": False,
            "repaired_payload_formal_label_loader_eligible_after_publication_completion": True,
        },
    }
    contract = ScientificRepairPublicationContract(
        data=data,
        sha256="f" * 64,
        repository_root=root,
    )
    prepared = prepare_scientific_repair_publication(
        contract=contract,
        source_identity=_source_identity(sha256_bytes(summary_bytes)),
    )
    downloads = root / "downloads"
    downloads.mkdir()
    return contract, prepared, downloads


class _FakeHF:
    def __init__(self):
        self.main = BASE_COMMIT
        self.commits = {
            BASE_COMMIT: {
                ".gitattributes": b"base",
                "metadata/base.txt": b"base metadata",
            }
        }
        self.parents: dict[str, str] = {}
        self.titles = {BASE_COMMIT: "Initial commit"}
        self.tags: dict[str, tuple[str, str]] = {}
        self.create_repo_calls = 0
        self.create_commit_calls = 0
        self.create_tag_calls = 0
        self.download_calls = 0
        self.fail_repo_response_once = False
        self.lose_commit_response_once = False
        self.lose_tag_response_once = False
        self.fail_tag_before_write_once = False
        self.deleted = False
        self.move_tag_on_download_call: int | None = None
        self.malicious_symlink = False

    def create_repo(self, repo_id: str, **kwargs: object) -> None:
        self.create_repo_calls += 1
        if repo_id != DESTINATION["repo"] or kwargs.get("private") is not True:
            raise AssertionError("private repo contract drifted")
        if self.fail_repo_response_once:
            self.fail_repo_response_once = False
            raise ConnectionError("synthetic create_repo response loss")

    def dataset_info(self, repo_id: str, revision: str):
        if self.deleted:
            raise ValueError("remote repository is deleted")
        if repo_id != DESTINATION["repo"]:
            raise AssertionError("wrong repo")
        if revision == "main":
            resolved = self.main
        elif revision in self.tags:
            resolved = self.tags[revision][1]
        elif revision in self.commits:
            resolved = revision
        else:
            raise ValueError("unknown revision")
        return SimpleNamespace(sha=resolved, private=True)

    def list_repo_refs(self, repo_id: str, *, repo_type: str):
        if self.deleted:
            raise ValueError("remote repository is deleted")
        return SimpleNamespace(
            tags=[
                SimpleNamespace(name=name, target_commit=identity)
                for name, (identity, _resolved) in sorted(self.tags.items())
            ]
        )

    def list_repo_files(self, repo_id: str, *, repo_type: str, revision: str):
        return sorted(self.commits[revision])

    def list_repo_commits(self, repo_id: str, *, repo_type: str, revision: str):
        reachable: list[str] = []
        current: str | None = revision
        while current is not None:
            if current in reachable:
                raise AssertionError("synthetic history cycle")
            reachable.append(current)
            current = self.parents.get(current)
        # note (luojiaxuan): Deliberately return commit-id order rather than
        # parent order so production code cannot depend on list position.
        return [
            SimpleNamespace(commit_id=identifier, title=self.titles[identifier])
            for identifier in sorted(reachable)
        ]

    def list_repo_tree(
        self,
        repo_id: str,
        path_in_repo: str | None,
        *,
        recursive: bool,
        expand: bool,
        revision: str,
        repo_type: str,
    ):
        if path_in_repo is not None or not recursive or not expand:
            raise AssertionError("recursive expanded tree contract drifted")
        files = self.commits[revision]
        file_entries = [
            SimpleNamespace(
                path=path,
                blob_id=sha256_bytes(payload)[:40],
                size=len(payload),
                lfs=None,
                xet_hash=None,
            )
            for path, payload in sorted(files.items())
        ]
        folder_paths = {
            "/".join(Path(path).parts[:depth])
            for path in files
            for depth in range(1, len(Path(path).parts))
        }
        folder_entries = []
        for folder in sorted(folder_paths):
            descendants = {
                path: sha256_bytes(payload)
                for path, payload in sorted(files.items())
                if path.startswith(folder + "/")
            }
            folder_entries.append(
                SimpleNamespace(
                    path=folder,
                    tree_id=sha256_bytes(canonical_json_bytes(descendants))[:40],
                )
            )
        return sorted(
            [*file_entries, *folder_entries],
            key=lambda entry: entry.path,
        )

    def create_commit(
        self,
        repo_id: str,
        *,
        operations: list[object],
        commit_message: str,
        repo_type: str,
        revision: str,
        parent_commit: str,
    ):
        self.create_commit_calls += 1
        if parent_commit != self.main or revision != "main":
            raise AssertionError("create_commit omitted observed-main CAS")
        paths = [operation.path_in_repo for operation in operations]
        if paths != [DESTINATION["archive_path"], DESTINATION["sidecar_path"]]:
            raise AssertionError("create_commit did not contain exact pair")
        if any(path in self.commits[parent_commit] for path in paths):
            raise AssertionError("create_commit would overwrite a path")
        files = dict(self.commits[parent_commit])
        for operation in operations:
            files[operation.path_in_repo] = operation.path_or_fileobj.getvalue()
        self.commits[PAIR_COMMIT] = files
        self.parents[PAIR_COMMIT] = parent_commit
        self.titles[PAIR_COMMIT] = commit_message
        self.main = PAIR_COMMIT
        if self.lose_commit_response_once:
            self.lose_commit_response_once = False
            raise ConnectionError("synthetic create_commit response loss")
        return SimpleNamespace(oid=PAIR_COMMIT)

    def create_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        tag_message: str,
        revision: str,
        repo_type: str,
        exist_ok: bool,
    ) -> None:
        self.create_tag_calls += 1
        if exist_ok or not tag_message:
            raise AssertionError("publication tag must be annotated and no-replace")
        if self.fail_tag_before_write_once:
            self.fail_tag_before_write_once = False
            raise ConnectionError("synthetic pre-write tag failure")
        if tag in self.tags:
            raise FileExistsError("tag already exists")
        self.tags[tag] = (TAG_OBJECT, revision)
        if self.lose_tag_response_once:
            self.lose_tag_response_once = False
            raise ConnectionError("synthetic create_tag response loss")

    def download(self, **kwargs: object) -> str:
        self.download_calls += 1
        revision = str(kwargs["revision"])
        filename = str(kwargs["filename"])
        local_dir = Path(kwargs["local_dir"])
        if kwargs.get("force_download") is not True:
            raise AssertionError("download was not forced")
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.malicious_symlink and filename == DESTINATION["archive_path"]:
            target.parent.rmdir()
            target.parent.symlink_to(local_dir)
            target = local_dir / Path(filename).name
        target.write_bytes(self.commits[revision][filename])
        if self.move_tag_on_download_call == self.download_calls:
            self.tags[DESTINATION["tag"]] = ("9" * 40, BASE_COMMIT)
        return str(target.resolve() if self.malicious_symlink else target)

    def seed_pair(
        self,
        prepared,
        *,
        tagged: bool,
        extra_same_commit: bool = False,
    ) -> None:
        files = dict(self.commits[BASE_COMMIT])
        files[DESTINATION["archive_path"]] = prepared.archive_bytes
        files[DESTINATION["sidecar_path"]] = prepared.sidecar_bytes
        if extra_same_commit:
            files["unexpected.bin"] = b"unexpected"
        self.commits[PAIR_COMMIT] = files
        self.parents[PAIR_COMMIT] = BASE_COMMIT
        self.titles[PAIR_COMMIT] = PAIR_COMMIT_TITLE
        self.main = PAIR_COMMIT
        if tagged:
            self.tags[DESTINATION["tag"]] = (TAG_OBJECT, PAIR_COMMIT)

    def seed_split_pair(self, prepared) -> None:
        middle = "5" * 40
        middle_files = dict(self.commits[BASE_COMMIT])
        middle_files[DESTINATION["archive_path"]] = prepared.archive_bytes
        self.commits[middle] = middle_files
        self.parents[middle] = BASE_COMMIT
        self.titles[middle] = "archive only"
        pair_files = dict(middle_files)
        pair_files[DESTINATION["sidecar_path"]] = prepared.sidecar_bytes
        self.commits[PAIR_COMMIT] = pair_files
        self.parents[PAIR_COMMIT] = middle
        self.titles[PAIR_COMMIT] = PAIR_COMMIT_TITLE
        self.main = PAIR_COMMIT


def _publish(api: _FakeHF, fixture):
    contract, prepared, downloads = fixture
    return publish_scientific_repair_archive(
        api=api,
        download_fn=api.download,
        operation_factory=_operation_factory,
        contract=contract,
        prepared=prepared,
        fresh_download_parent=downloads,
    )


def _seal_remote_base(api: _FakeHF, fixture) -> None:
    contract, prepared, _downloads = fixture
    claim = publication._claim_payload(contract, prepared)
    claim_bytes = pretty_json_bytes(claim)
    claim_sha = sha256_bytes(claim_bytes)
    receipt = publication._capture_remote_base_receipt(
        api=api,
        contract=contract,
        claim_sha256=claim_sha,
    )
    mode = contract.local_state["file_mode"]
    publication._exclusive_or_identical_state(
        Path(contract.local_state["claim_path"]), claim_bytes, mode
    )
    publication._exclusive_or_identical_state(
        Path(contract.local_state["remote_base_receipt_path"]),
        pretty_json_bytes(receipt),
        mode,
    )


class ScientificRepairPublicationTest(unittest.TestCase):
    def test_frozen_contract_and_validate_cli_are_offline_and_locked(self) -> None:
        contract = load_frozen_publication_contract(
            CANONICAL_CONFIG_PATH, repository_root=ROOT
        )
        self.assertEqual(contract.sha256, FROZEN_CONFIG_SHA256)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            manager.main(["validate-contract", "--repository-root", str(ROOT)])
        result = json.loads(output.getvalue())
        self.assertFalse(result["network_access_performed"])
        self.assertFalse(result["git_remote_verification_performed"])
        self.assertFalse(result["hf_network_access_performed"])
        self.assertFalse(result["formal_label_loader_eligible"])
        self.assertFalse(result["gate_training_unlocked"])

    def test_prepare_cli_reports_git_network_but_no_hf_network(self) -> None:
        prepared = SimpleNamespace(
            archive_path=Path("/synthetic/repaired.tar"),
            archive_bytes=b"archive",
            archive_sha256="a" * 64,
            sidecar_sha256="b" * 64,
            producer_completion_sha256="c" * 64,
        )
        output = io.StringIO()
        with (
            mock.patch.object(manager, "_load", return_value=object()),
            mock.patch.object(manager, "_prepare", return_value=prepared),
            contextlib.redirect_stdout(output),
        ):
            manager.main(
                [
                    "prepare",
                    "--repository-root",
                    str(ROOT),
                ]
            )
        result = json.loads(output.getvalue())
        self.assertTrue(result["network_access_performed"])
        self.assertTrue(result["git_remote_verification_performed"])
        self.assertFalse(result["hf_network_access_performed"])
        self.assertFalse(result["formal_label_loader_eligible"])
        self.assertFalse(result["gate_training_unlocked"])

    def test_prepare_binds_completed_local_source_and_keeps_gate_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract, prepared, _downloads = _fixture(Path(directory).resolve())
            self.assertEqual(
                prepared.archive_sha256,
                contract.source["archive"]["sha256"],
            )
            self.assertFalse(
                prepared.sidecar["formal_consumption"][
                    "gate_training_unlocked_by_sidecar_alone"
                ]
            )
            Path(contract.source["completion"]["path"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                prepare_scientific_repair_publication(
                    contract=contract,
                    source_identity=prepared.source_identity,
                )

    def test_remote_main_resolution_uses_real_git_ls_remote(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout=(PUBLICATION_COMMIT + "\trefs/heads/main\n").encode("ascii"),
            stderr=b"",
        )
        with mock.patch.object(publication.subprocess, "run", return_value=completed) as run:
            resolved = publication._remote_main_head(ROOT, "origin")
        self.assertEqual(resolved, PUBLICATION_COMMIT)
        self.assertEqual(
            run.call_args.args[0],
            [
                "git",
                "ls-remote",
                "--exit-code",
                "origin",
                "refs/heads/main",
            ],
        )

    def test_empty_remote_publishes_exact_pair_and_annotated_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            completion = _publish(api, fixture)
            self.assertEqual(completion["remote_state_before_publish"], REMOTE_EMPTY)
            self.assertTrue(completion["formal_label_loader_eligible"])
            self.assertTrue(completion["gate_training_unlocked"])
            self.assertEqual(completion["immutable_revision"], PAIR_COMMIT)
            self.assertEqual(completion["tag_resolved_commit"], PAIR_COMMIT)
            self.assertEqual(completion["annotated_tag_object_identity"], TAG_OBJECT)
            self.assertNotEqual(TAG_OBJECT, PAIR_COMMIT)
            claim = Path(fixture[0].local_state["claim_path"])
            receipt = Path(fixture[0].local_state["remote_base_receipt_path"])
            stage = Path(fixture[0].local_state["completion_staging_path"])
            seal = Path(fixture[0].local_state["completion_seal_path"])
            self.assertEqual(os.stat(claim).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(receipt).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(stage).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(seal).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(stage).st_ino, os.stat(seal).st_ino)
            receipt_json = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt_json["base_reachable_history_commit_ids"], [BASE_COMMIT]
            )
            self.assertTrue(receipt_json["base_recursive_tree_inventory"])
            self.assertEqual(
                {
                    record["type"]
                    for record in receipt_json["base_recursive_tree_inventory"]
                },
                {"directory", "file"},
            )
            self.assertEqual(api.create_commit_calls, 1)
            self.assertEqual(api.create_tag_calls, 1)

    def test_byte_identical_untagged_remote_only_creates_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            _seal_remote_base(api, fixture)
            api.seed_pair(fixture[1], tagged=False)
            completion = _publish(api, fixture)
            self.assertEqual(
                completion["remote_state_before_publish"],
                REMOTE_EMPTY,
            )
            self.assertEqual(api.create_commit_calls, 0)
            self.assertEqual(api.create_tag_calls, 1)
            self.assertEqual(api.create_repo_calls, 0)

    def test_annotated_tagged_remote_is_idempotent_and_nonmutating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            _seal_remote_base(api, fixture)
            api.seed_pair(fixture[1], tagged=True)
            first = _publish(api, fixture)
            second = _publish(api, fixture)
            self.assertEqual(first, second)
            self.assertEqual(first["remote_state_before_publish"], REMOTE_EMPTY)
            self.assertEqual(api.create_commit_calls, 0)
            self.assertEqual(api.create_tag_calls, 0)

    def test_tag_response_loss_recovers_by_resolved_commit_not_tag_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            _seal_remote_base(api, fixture)
            api.seed_pair(fixture[1], tagged=False)
            api.lose_tag_response_once = True
            completion = _publish(api, fixture)
            self.assertEqual(completion["tag_resolved_commit"], PAIR_COMMIT)
            self.assertEqual(completion["annotated_tag_object_identity"], TAG_OBJECT)
            self.assertEqual(api.create_tag_calls, 1)

    def test_commit_response_loss_requeries_and_does_not_write_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            api.lose_commit_response_once = True
            completion = _publish(api, fixture)
            self.assertEqual(completion["immutable_revision"], PAIR_COMMIT)
            self.assertEqual(api.create_commit_calls, 1)
            self.assertEqual(api.create_tag_calls, 1)

    def test_create_repo_response_loss_is_requeried_after_durable_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            api.fail_repo_response_once = True
            completion = _publish(api, fixture)
            self.assertTrue(completion["gate_training_unlocked"])
            self.assertEqual(api.create_repo_calls, 1)
            self.assertTrue(Path(fixture[0].local_state["claim_path"]).is_file())

    def test_prewrite_tag_failure_keeps_gate_locked_then_retry_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            api.fail_tag_before_write_once = True
            with self.assertRaisesRegex(ConnectionError, "pre-write"):
                _publish(api, fixture)
            self.assertFalse(
                Path(fixture[0].local_state["completion_seal_path"]).exists()
            )
            completion = _publish(api, fixture)
            self.assertTrue(completion["gate_training_unlocked"])
            self.assertEqual(api.create_commit_calls, 1)
            self.assertEqual(api.create_tag_calls, 2)

    def test_partial_mismatch_and_exact_pair_extra_file_fail_closed(self) -> None:
        scenarios = ("partial", "mismatch", "extra")
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                fixture = _fixture(Path(directory).resolve())
                api = _FakeHF()
                _seal_remote_base(api, fixture)
                api.seed_pair(
                    fixture[1],
                    tagged=False,
                    extra_same_commit=scenario == "extra",
                )
                if scenario == "partial":
                    del api.commits[PAIR_COMMIT][DESTINATION["sidecar_path"]]
                elif scenario == "mismatch":
                    api.commits[PAIR_COMMIT][DESTINATION["archive_path"]] = b"wrong"
                with self.assertRaises(ValueError):
                    _publish(api, fixture)
                self.assertFalse(
                    Path(fixture[0].local_state["completion_seal_path"]).exists()
                )
                self.assertEqual(api.create_commit_calls, 0)

    def test_split_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            _seal_remote_base(api, fixture)
            api.seed_split_pair(fixture[1])
            with self.assertRaisesRegex(ValueError, "reachable history"):
                _publish(api, fixture)

    def test_pair_cannot_modify_an_existing_base_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            _seal_remote_base(api, fixture)
            api.seed_pair(fixture[1], tagged=False)
            api.commits[PAIR_COMMIT][".gitattributes"] = b"modified base blob"
            with self.assertRaisesRegex(ValueError, "recursive blobs"):
                _publish(api, fixture)
            self.assertFalse(
                Path(fixture[0].local_state["completion_seal_path"]).exists()
            )

    def test_history_interposition_is_rejected_setwise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            _seal_remote_base(api, fixture)
            api.seed_pair(fixture[1], tagged=False)
            interposed = "6" * 40
            api.commits[interposed] = dict(api.commits[BASE_COMMIT])
            api.parents[interposed] = BASE_COMMIT
            api.titles[interposed] = "interposed commit"
            api.parents[PAIR_COMMIT] = interposed
            with self.assertRaisesRegex(ValueError, "base plus exactly pair"):
                _publish(api, fixture)

    def test_preexisting_pair_without_remote_base_receipt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            api.seed_pair(fixture[1], tagged=False)
            with self.assertRaisesRegex(ValueError, "already exist"):
                _publish(api, fixture)
            self.assertFalse(
                Path(fixture[0].local_state["remote_base_receipt_path"]).exists()
            )
            self.assertFalse(
                Path(fixture[0].local_state["completion_seal_path"]).exists()
            )
            self.assertEqual(api.create_commit_calls, 0)

    def test_crash_after_retained_completion_stage_recovers_by_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            with mock.patch.object(
                publication,
                "_publish_completion_from_retained_stage_last",
                side_effect=ConnectionError("synthetic crash after retained stage"),
            ):
                with self.assertRaisesRegex(ConnectionError, "retained stage"):
                    _publish(api, fixture)
            stage = Path(fixture[0].local_state["completion_staging_path"])
            seal = Path(fixture[0].local_state["completion_seal_path"])
            self.assertTrue(stage.is_file())
            self.assertFalse(seal.exists())
            staged_bytes = stage.read_bytes()
            completion = _publish(api, fixture)
            self.assertTrue(completion["gate_training_unlocked"])
            self.assertEqual(stage.read_bytes(), staged_bytes)
            self.assertEqual(stage.read_bytes(), seal.read_bytes())
            self.assertEqual(os.stat(stage).st_ino, os.stat(seal).st_ino)
            self.assertEqual(api.create_commit_calls, 1)
            self.assertEqual(api.create_tag_calls, 1)

    def test_completion_hard_link_is_the_last_publication_state_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            events: list[str] = []
            original_state = publication._exclusive_or_identical_state
            original_link = publication._publish_completion_from_retained_stage_last

            def record_state(path, payload, mode):
                events.append(f"state:{Path(path).name}")
                return original_state(path, payload, mode)

            def record_final(**kwargs):
                original_link(**kwargs)
                events.append("final-completion-hard-link")

            with (
                mock.patch.object(
                    publication,
                    "_exclusive_or_identical_state",
                    side_effect=record_state,
                ),
                mock.patch.object(
                    publication,
                    "_publish_completion_from_retained_stage_last",
                    side_effect=record_final,
                ),
            ):
                _publish(api, fixture)
            self.assertEqual(events[-1], "final-completion-hard-link")
            self.assertEqual(events.count("final-completion-hard-link"), 1)

    def test_lightweight_or_wrong_resolved_tag_is_rejected(self) -> None:
        for tag in ((PAIR_COMMIT, PAIR_COMMIT), (TAG_OBJECT, BASE_COMMIT)):
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as directory:
                fixture = _fixture(Path(directory).resolve())
                api = _FakeHF()
                _seal_remote_base(api, fixture)
                api.seed_pair(fixture[1], tagged=False)
                api.tags[DESTINATION["tag"]] = tag
                with self.assertRaises(ValueError):
                    _publish(api, fixture)

    def test_tag_move_during_final_fresh_download_prevents_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            # note (luojiaxuan): Calls 1-2 verify the pair before tagging; call
            # 3 is the archive fetch in the final completion attestation.
            api.move_tag_on_download_call = 3
            with self.assertRaises(ValueError):
                _publish(api, fixture)
            self.assertFalse(
                Path(fixture[0].local_state["completion_seal_path"]).exists()
            )

    def test_fresh_download_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            _seal_remote_base(api, fixture)
            api.seed_pair(fixture[1], tagged=True)
            api.malicious_symlink = True
            with self.assertRaises(ValueError):
                _publish(api, fixture)
            self.assertFalse(
                Path(fixture[0].local_state["completion_seal_path"]).exists()
            )

    def test_orphan_completion_and_nonidentical_claim_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            completion = Path(fixture[0].local_state["completion_seal_path"])
            completion.write_bytes(b"orphan")
            os.chmod(completion, 0o600)
            with self.assertRaisesRegex(ValueError, "orphan"):
                _publish(_FakeHF(), fixture)
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            claim = Path(fixture[0].local_state["claim_path"])
            claim.write_bytes(b"wrong")
            os.chmod(claim, 0o600)
            with self.assertRaisesRegex(ValueError, "not byte-identical"):
                _publish(_FakeHF(), fixture)

    def test_completed_replay_never_recreates_deleted_remote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory).resolve())
            api = _FakeHF()
            _publish(api, fixture)
            create_repo_calls = api.create_repo_calls
            api.deleted = True
            with self.assertRaisesRegex(ValueError, "deleted"):
                _publish(api, fixture)
            self.assertEqual(api.create_repo_calls, create_repo_calls)

    def test_token_reader_rejects_symlink_and_open_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory).resolve() / "token"
            token.write_text("synthetic-token\n", encoding="utf-8")
            os.chmod(token, 0o600)
            self.assertEqual(manager._read_hf_token(token), "synthetic-token")
            link = token.with_name("link")
            link.symlink_to(token)
            with self.assertRaises(ValueError):
                manager._read_hf_token(link)
            os.chmod(token, 0o644)
            with self.assertRaisesRegex(ValueError, "0400 or 0600"):
                manager._read_hf_token(token)

    def test_publish_cli_preserves_lexical_paths_for_symlink_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            token = root / "token"
            token.write_text("synthetic-token\n", encoding="utf-8")
            os.chmod(token, 0o600)
            token_link = root / "token-link"
            token_link.symlink_to(token)
            fresh = root / "fresh"
            fresh.mkdir()
            fresh_link = root / "fresh-link"
            fresh_link.symlink_to(fresh, target_is_directory=True)
            fake_hub = SimpleNamespace(
                CommitOperationAdd=object(),
                HfApi=lambda **_kwargs: object(),
                hf_hub_download=lambda **_kwargs: "unused",
            )
            with (
                mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
                mock.patch.object(manager, "_load", return_value=object()),
                mock.patch.object(manager, "_prepare", return_value=object()),
                mock.patch.object(
                    manager,
                    "_read_hf_token",
                    return_value="synthetic-token",
                ) as token_reader,
                mock.patch.object(
                    manager,
                    "publish_scientific_repair_archive",
                    return_value={"status": "synthetic"},
                ) as publisher,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                manager.main(
                    [
                        "publish",
                        "--repository-root",
                        str(root),
                        "--hf-token-file",
                        str(token_link),
                        "--fresh-download-parent",
                        str(fresh_link),
                    ]
                )
            token_reader.assert_called_once_with(token_link)
            self.assertEqual(
                publisher.call_args.kwargs["fresh_download_parent"], fresh_link
            )


if __name__ == "__main__":
    unittest.main()
