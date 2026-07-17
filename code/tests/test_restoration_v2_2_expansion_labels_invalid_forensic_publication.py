from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from causalcache.restoration_v2_2_expansion_labels_invalid_forensic import (
    package_invalid_forensic_archive,
)
from causalcache.restoration_v2_2_expansion_labels_invalid_forensic_publication import (
    CANONICAL_CONFIG_PATH,
    DESTINATION,
    FROZEN_CONFIG_SHA256,
    PAIR_COMMIT_TITLE,
    REMOTE_BYTE_IDENTICAL_UNTAGGED,
    REMOTE_EMPTY,
    REMOTE_TAGGED_BYTE_IDENTICAL,
    load_frozen_publication_contract,
    prepare_invalid_forensic_publication,
    publication_contract_from_data,
    publish_invalid_forensic_archive,
)
from tests.test_restoration_v2_2_expansion_labels_invalid_forensic import _fixture
from scripts import (
    manage_restoration_v2_2_expansion_labels_invalid_forensic_publication as manager,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "1" * 40
PAIR_COMMIT = "2" * 40
SPLIT_ARCHIVE_COMMIT = "3" * 40


@dataclass
class _Operation:
    path_in_repo: str
    path_or_fileobj: object


class _FakeHF:
    def __init__(self) -> None:
        self.private = True
        self.commits: dict[str, dict[str, bytes]] = {
            BASE_COMMIT: {
                ".gitattributes": b"*.tar filter=lfs diff=lfs merge=lfs -text\n"
            }
        }
        self.main = BASE_COMMIT
        self.parents: dict[str, str | None] = {BASE_COMMIT: None}
        self.titles = {BASE_COMMIT: "initial commit"}
        self.tags: dict[str, str] = {}
        self.create_commit_calls: list[dict[str, object]] = []
        self.create_tag_calls = 0
        self.create_repo_calls = 0
        self.download_calls: list[dict[str, object]] = []
        self.lose_commit_response_once = False
        self.lose_tag_response_once = False
        self.fail_tag_before_write_once = False
        self.fail_create_repo_response_once = False
        self.malicious_intermediate_symlink = False
        self.move_tag_on_download_call: int | None = None
        self.deleted = False

    def _resolve(self, revision: str) -> str:
        if revision == "main":
            return self.main
        if revision in self.tags:
            return self.tags[revision]
        if revision in self.commits:
            return revision
        raise ValueError(f"unknown fake revision: {revision}")

    def create_repo(self, repo_id: str, **kwargs: object) -> None:
        self.create_repo_calls += 1
        if repo_id != DESTINATION["repo"]:
            raise AssertionError("wrong fake repository")
        if kwargs != {"repo_type": "dataset", "private": True, "exist_ok": True}:
            raise AssertionError("repository must be private and idempotent")
        self.deleted = False
        if self.fail_create_repo_response_once:
            self.fail_create_repo_response_once = False
            raise ConnectionError("synthetic create_repo response loss")

    def dataset_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
        if repo_id != DESTINATION["repo"]:
            raise AssertionError("wrong fake repository")
        if self.deleted:
            raise ValueError("synthetic remote repository is deleted")
        return SimpleNamespace(sha=self._resolve(revision), private=self.private)

    def list_repo_refs(self, repo_id: str, *, repo_type: str) -> SimpleNamespace:
        if repo_id != DESTINATION["repo"] or repo_type != "dataset":
            raise AssertionError("wrong fake repository")
        refs = [
            SimpleNamespace(name=name, target_commit=commit)
            for name, commit in sorted(self.tags.items())
        ]
        return SimpleNamespace(tags=refs)

    def list_repo_commits(
        self, repo_id: str, *, repo_type: str, revision: str
    ) -> list[SimpleNamespace]:
        if repo_id != DESTINATION["repo"] or repo_type != "dataset":
            raise AssertionError("wrong fake repository")
        commit = self._resolve(revision)
        result: list[SimpleNamespace] = []
        while commit is not None:
            result.append(
                SimpleNamespace(
                    commit_id=commit,
                    title=self.titles[commit],
                )
            )
            commit = self.parents[commit]
        return result

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str
    ) -> list[str]:
        if repo_id != DESTINATION["repo"] or repo_type != "dataset":
            raise AssertionError("wrong fake repository")
        return sorted(self.commits[self._resolve(revision)])

    def create_commit(
        self,
        repo_id: str,
        *,
        operations: list[_Operation],
        commit_message: str,
        repo_type: str,
        revision: str,
        parent_commit: str,
    ) -> SimpleNamespace:
        if (
            repo_id != DESTINATION["repo"]
            or repo_type != "dataset"
            or revision != "main"
            or parent_commit != self.main
            or commit_message != PAIR_COMMIT_TITLE
        ):
            raise AssertionError("create_commit did not use the observed main CAS")
        paths = [operation.path_in_repo for operation in operations]
        if paths != [DESTINATION["archive_path"], DESTINATION["sidecar_path"]]:
            raise AssertionError("archive and sidecar must be one ordered commit")
        files = dict(self.commits[self.main])
        for operation in operations:
            payload = operation.path_or_fileobj
            payload.seek(0)
            files[operation.path_in_repo] = payload.read()
        self.commits[PAIR_COMMIT] = files
        self.parents[PAIR_COMMIT] = parent_commit
        self.titles[PAIR_COMMIT] = commit_message
        self.main = PAIR_COMMIT
        self.create_commit_calls.append(
            {"parent": parent_commit, "paths": paths, "message": commit_message}
        )
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
        if (
            repo_id != DESTINATION["repo"]
            or repo_type != "dataset"
            or exist_ok
            or tag != DESTINATION["tag"]
            or not tag_message
        ):
            raise AssertionError("tag publication contract drifted")
        if self.fail_tag_before_write_once:
            self.fail_tag_before_write_once = False
            raise ConnectionError("synthetic pre-write tag failure")
        if tag in self.tags:
            raise ValueError("tag already exists")
        self.tags[tag] = self._resolve(revision)
        if self.lose_tag_response_once:
            self.lose_tag_response_once = False
            raise ConnectionError("synthetic create_tag response loss")

    def download(self, **kwargs: object) -> str:
        required = {
            "repo_id",
            "filename",
            "repo_type",
            "revision",
            "local_dir",
            "force_download",
        }
        if set(kwargs) != required:
            raise AssertionError("unexpected fake download arguments")
        if (
            kwargs["repo_id"] != DESTINATION["repo"]
            or kwargs["repo_type"] != "dataset"
            or kwargs["force_download"] is not True
        ):
            raise AssertionError("fresh force-download contract drifted")
        local_dir = Path(kwargs["local_dir"])
        first_for_dir = not any(
            call["local_dir"] == str(local_dir) for call in self.download_calls
        )
        started_empty = not any(local_dir.iterdir()) if first_for_dir else None
        revision = self._resolve(str(kwargs["revision"]))
        filename = str(kwargs["filename"])
        payload = self.commits[revision][filename]
        if self.malicious_intermediate_symlink and first_for_dir:
            escape = local_dir.parent / f"{local_dir.name}-escape"
            escape.mkdir()
            (local_dir / "attempts").symlink_to(
                escape, target_is_directory=True
            )
        output = local_dir / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        self.download_calls.append(
            {
                "revision": revision,
                "filename": filename,
                "local_dir": str(local_dir),
                "force_download": True,
                "started_empty": started_empty,
            }
        )
        if self.move_tag_on_download_call == len(self.download_calls):
            self.tags[DESTINATION["tag"]] = BASE_COMMIT
        return str(output)

    def seed_pair(self, archive: bytes, sidecar: bytes, *, tagged: bool) -> None:
        self.commits[PAIR_COMMIT] = {
            **self.commits[BASE_COMMIT],
            DESTINATION["archive_path"]: archive,
            DESTINATION["sidecar_path"]: sidecar,
        }
        self.parents[PAIR_COMMIT] = BASE_COMMIT
        self.titles[PAIR_COMMIT] = PAIR_COMMIT_TITLE
        self.main = PAIR_COMMIT
        if tagged:
            self.tags[DESTINATION["tag"]] = PAIR_COMMIT

    def seed_split_pair(self, archive: bytes, sidecar: bytes) -> None:
        self.commits[SPLIT_ARCHIVE_COMMIT] = {
            **self.commits[BASE_COMMIT],
            DESTINATION["archive_path"]: archive,
        }
        self.parents[SPLIT_ARCHIVE_COMMIT] = BASE_COMMIT
        self.titles[SPLIT_ARCHIVE_COMMIT] = "Publish archive alone"
        self.commits[PAIR_COMMIT] = {
            **self.commits[SPLIT_ARCHIVE_COMMIT],
            DESTINATION["sidecar_path"]: sidecar,
        }
        self.parents[PAIR_COMMIT] = SPLIT_ARCHIVE_COMMIT
        self.titles[PAIR_COMMIT] = PAIR_COMMIT_TITLE
        self.main = PAIR_COMMIT


def _operation_factory(**kwargs: object) -> _Operation:
    return _Operation(**kwargs)


def _publication_fixture(directory: Path):
    p0_contract, _data, root, global_ledger, workers, archive = _fixture(directory)
    package_invalid_forensic_archive(
        contract=p0_contract,
        raw_output_dir=root,
        external_global_ledger=global_ledger,
        external_worker_ledgers=workers,
        output_archive=archive,
    )
    publication_data = json.loads((ROOT / CANONICAL_CONFIG_PATH).read_bytes())
    publication_data["p0_forensic_source"]["config_sha256"] = p0_contract.sha256
    state = directory / "publication-state"
    state.mkdir()
    publication_data["local_state"]["claim_path"] = str(
        (state / "claim.json").resolve()
    )
    publication_data["local_state"]["completion_seal_path"] = str(
        (state / "completion.json").resolve()
    )
    contract = publication_contract_from_data(publication_data)
    source_identity = {
        "publication_source_git_commit": "a" * 40,
        "origin_main_git_commit": "a" * 40,
        "branch": "main",
        "origin_url": "https://github.com/luojiaxuan/CausalCache.git",
        "source_path_count": 5,
        "source_inventory_sha256": "b" * 64,
    }
    prepared = prepare_invalid_forensic_publication(
        archive_path=archive,
        p0_contract=p0_contract,
        contract=contract,
        source_identity=source_identity,
    )
    downloads = directory / "fresh-downloads"
    downloads.mkdir()
    return p0_contract, contract, prepared, downloads


def _publish(api: _FakeHF, fixture: tuple[object, ...]):
    p0_contract, contract, prepared, downloads = fixture
    return publish_invalid_forensic_archive(
        api=api,
        download_fn=api.download,
        operation_factory=_operation_factory,
        contract=contract,
        p0_contract=p0_contract,
        prepared=prepared,
        fresh_download_parent=downloads,
    )


class ExpansionLabelsInvalidForensicPublicationTest(unittest.TestCase):
    def test_cli_keeps_network_behind_explicit_publish_command(self) -> None:
        choices = manager._parser()._subparsers._group_actions[0].choices
        self.assertEqual(set(choices), {"validate-contract", "prepare", "publish"})
        token_action = next(
            action
            for action in choices["publish"]._actions
            if action.dest == "hf_token_file"
        )
        self.assertTrue(token_action.required)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            manager.main(
                [
                    "validate-contract",
                    "--repository-root",
                    str(ROOT),
                ]
            )
        result = json.loads(output.getvalue())
        self.assertEqual(
            result["status"],
            "VALID_SOURCE_ONLY_INVALID_FORENSIC_PUBLICATION_CONTRACT",
        )
        self.assertFalse(result["network_access_performed"])
        self.assertFalse(result["formal_label_loader_eligible"])

    def test_cli_token_reader_is_explicit_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / "token.txt"
            token.write_text("synthetic-private-token\n", encoding="utf-8")
            os.chmod(token, 0o600)
            self.assertEqual(
                manager._read_hf_token(token), "synthetic-private-token"
            )
            link = Path(directory) / "token-link.txt"
            link.symlink_to(token)
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                manager._read_hf_token(link)

    def test_cli_token_reader_rejects_group_or_other_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory) / "token.txt"
            token.write_text("synthetic-private-token\n", encoding="utf-8")
            os.chmod(token, 0o644)
            with self.assertRaisesRegex(ValueError, "0400 or 0600"):
                manager._read_hf_token(token)
            os.chmod(token, 0o400)
            self.assertEqual(
                manager._read_hf_token(token), "synthetic-private-token"
            )

    def test_cli_prepare_returns_bound_prepared_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p0, contract, prepared, _downloads = _publication_fixture(
                Path(directory)
            )
            args = SimpleNamespace(
                repository_root=ROOT,
                archive=prepared.archive_path,
            )
            with mock.patch.object(
                manager,
                "validate_clean_pushed_source",
                return_value=prepared.source_identity,
            ):
                result = manager._prepare(args, contract, p0)
            self.assertIsNotNone(result)
            self.assertEqual(result.archive_sha256, prepared.archive_sha256)
            self.assertEqual(result.sidecar_bytes, prepared.sidecar_bytes)

    def test_cli_publish_injects_explicit_token_and_prepared_payload(self) -> None:
        import huggingface_hub

        with tempfile.TemporaryDirectory() as directory:
            p0, contract, prepared, downloads = _publication_fixture(
                Path(directory)
            )
            token_file = Path(directory) / "hf-token.txt"
            token_file.write_text("synthetic-private-token\n", encoding="utf-8")
            os.chmod(token_file, 0o600)
            api_instance = object()
            output = io.StringIO()
            with (
                mock.patch.object(manager, "_load", return_value=(contract, p0)),
                mock.patch.object(manager, "_prepare", return_value=prepared),
                mock.patch.object(
                    manager,
                    "publish_invalid_forensic_archive",
                    return_value={"status": "MOCK_PUBLISHED"},
                ) as publish,
                mock.patch.object(
                    huggingface_hub, "HfApi", return_value=api_instance
                ) as hf_api,
                mock.patch.object(huggingface_hub, "hf_hub_download") as download,
                contextlib.redirect_stdout(output),
            ):
                manager.main(
                    [
                        "publish",
                        "--repository-root",
                        str(ROOT),
                        "--archive",
                        str(prepared.archive_path),
                        "--fresh-download-parent",
                        str(downloads),
                        "--hf-token-file",
                        str(token_file),
                    ]
                )
            self.assertEqual(json.loads(output.getvalue())["status"], "MOCK_PUBLISHED")
            hf_api.assert_called_once_with(token="synthetic-private-token")
            arguments = publish.call_args.kwargs
            self.assertIs(arguments["prepared"], prepared)
            self.assertIs(arguments["api"], api_instance)
            self.assertIs(arguments["download_fn"].func, download)
            self.assertEqual(
                arguments["download_fn"].keywords,
                {"token": "synthetic-private-token"},
            )

    def test_frozen_publication_config_and_sidecar_bindings_are_exact(self) -> None:
        contract = load_frozen_publication_contract(
            CANONICAL_CONFIG_PATH, repository_root=ROOT
        )
        self.assertEqual(contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(contract.destination, DESTINATION)
        self.assertFalse(
            contract.data["formal_consumption"]["formal_label_loader_eligible"]
        )
        with tempfile.TemporaryDirectory() as directory:
            _p0, synthetic, prepared, _downloads = _publication_fixture(
                Path(directory)
            )
            self.assertEqual(
                prepared.sidecar["publication_source"][
                    "publication_config_sha256"
                ],
                synthetic.sha256,
            )
            self.assertEqual(
                prepared.sidecar["source_archive"]["archive_sha256"],
                prepared.archive_sha256,
            )
            self.assertFalse(
                prepared.sidecar["formal_consumption"][
                    "formal_label_loader_eligible"
                ]
            )

    def test_prepare_rejects_a_symlinked_p0_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p0, contract, prepared, _downloads = _publication_fixture(
                Path(directory)
            )
            link = Path(directory) / "archive-link.tar"
            link.symlink_to(prepared.archive_path)
            with self.assertRaisesRegex(ValueError, "not safe to read"):
                prepare_invalid_forensic_publication(
                    archive_path=link,
                    p0_contract=p0,
                    contract=contract,
                    source_identity=prepared.source_identity,
                )

    def test_empty_remote_uses_one_two_file_commit_then_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            api = _FakeHF()
            completion = _publish(api, fixture)
            self.assertEqual(completion["remote_state_before_publish"], REMOTE_EMPTY)
            self.assertEqual(completion["immutable_revision"], PAIR_COMMIT)
            self.assertEqual(len(api.create_commit_calls), 1)
            self.assertEqual(api.create_tag_calls, 1)
            self.assertEqual(api.tags[DESTINATION["tag"]], PAIR_COMMIT)
            self.assertTrue(all(call["force_download"] for call in api.download_calls))
            first_calls = [
                call for call in api.download_calls if call["started_empty"] is not None
            ]
            self.assertTrue(all(call["started_empty"] for call in first_calls))

    def test_byte_identical_untagged_remote_only_creates_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            prepared = fixture[2]
            api = _FakeHF()
            api.seed_pair(prepared.archive_bytes, prepared.sidecar_bytes, tagged=False)
            completion = _publish(api, fixture)
            self.assertEqual(
                completion["remote_state_before_publish"],
                REMOTE_BYTE_IDENTICAL_UNTAGGED,
            )
            self.assertEqual(api.create_commit_calls, [])
            self.assertEqual(api.create_tag_calls, 1)

    def test_tagged_byte_identical_remote_is_nonmutating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            prepared = fixture[2]
            api = _FakeHF()
            api.seed_pair(prepared.archive_bytes, prepared.sidecar_bytes, tagged=True)
            completion = _publish(api, fixture)
            self.assertEqual(
                completion["remote_state_before_publish"],
                REMOTE_TAGGED_BYTE_IDENTICAL,
            )
            self.assertEqual(api.create_commit_calls, [])
            self.assertEqual(api.create_tag_calls, 0)

    def test_partial_mismatch_and_tag_conflicts_fail_closed(self) -> None:
        scenarios = ("partial", "mismatch", "tagged-mismatch")
        for scenario in scenarios:
            with (
                self.subTest(scenario=scenario),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = _publication_fixture(Path(directory))
                prepared = fixture[2]
                api = _FakeHF()
                api.seed_pair(
                    prepared.archive_bytes,
                    prepared.sidecar_bytes,
                    tagged=scenario == "tagged-mismatch",
                )
                if scenario == "partial":
                    del api.commits[PAIR_COMMIT][DESTINATION["sidecar_path"]]
                else:
                    api.commits[PAIR_COMMIT][DESTINATION["archive_path"]] = (
                        b"wrong archive"
                    )
                with self.assertRaises(ValueError):
                    _publish(api, fixture)
                self.assertEqual(api.create_commit_calls, [])
                self.assertEqual(api.create_tag_calls, 0)
                self.assertFalse(
                    Path(fixture[1].local_state["completion_seal_path"]).exists()
                )

    def test_same_final_bytes_added_in_two_commits_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            prepared = fixture[2]
            api = _FakeHF()
            api.seed_split_pair(prepared.archive_bytes, prepared.sidecar_bytes)
            with self.assertRaisesRegex(ValueError, "first appear together"):
                _publish(api, fixture)
            self.assertEqual(api.create_commit_calls, [])
            self.assertEqual(api.create_tag_calls, 0)

    def test_tag_response_loss_is_requeried_and_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            prepared = fixture[2]
            api = _FakeHF()
            api.seed_pair(prepared.archive_bytes, prepared.sidecar_bytes, tagged=False)
            api.lose_tag_response_once = True
            completion = _publish(api, fixture)
            self.assertEqual(completion["immutable_revision"], PAIR_COMMIT)
            self.assertEqual(api.tags[DESTINATION["tag"]], PAIR_COMMIT)
            self.assertEqual(api.create_tag_calls, 1)

    def test_commit_response_loss_recovers_without_second_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            api = _FakeHF()
            api.lose_commit_response_once = True
            with self.assertRaisesRegex(ConnectionError, "create_commit response loss"):
                _publish(api, fixture)
            self.assertEqual(len(api.create_commit_calls), 1)
            self.assertFalse(
                Path(fixture[1].local_state["completion_seal_path"]).exists()
            )
            completion = _publish(api, fixture)
            self.assertEqual(
                completion["remote_state_before_publish"],
                REMOTE_BYTE_IDENTICAL_UNTAGGED,
            )
            self.assertEqual(len(api.create_commit_calls), 1)
            self.assertEqual(api.create_tag_calls, 1)

    def test_prewrite_tag_failure_recovers_from_identical_untagged_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            api = _FakeHF()
            api.fail_tag_before_write_once = True
            with self.assertRaisesRegex(ConnectionError, "pre-write tag failure"):
                _publish(api, fixture)
            self.assertEqual(len(api.create_commit_calls), 1)
            self.assertNotIn(DESTINATION["tag"], api.tags)
            completion = _publish(api, fixture)
            self.assertEqual(
                completion["remote_state_before_publish"],
                REMOTE_BYTE_IDENTICAL_UNTAGGED,
            )
            self.assertEqual(len(api.create_commit_calls), 1)
            self.assertEqual(api.create_tag_calls, 2)

    def test_claim_and_completion_are_mode_600_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            api = _FakeHF()
            first = _publish(api, fixture)
            contract = fixture[1]
            claim = Path(contract.local_state["claim_path"])
            completion = Path(contract.local_state["completion_seal_path"])
            claim_bytes = claim.read_bytes()
            completion_bytes = completion.read_bytes()
            self.assertEqual(
                os.stat(claim, follow_symlinks=False).st_mode & 0o777, 0o600
            )
            self.assertEqual(
                os.stat(completion, follow_symlinks=False).st_mode & 0o777, 0o600
            )
            second = _publish(api, fixture)
            self.assertEqual(first, second)
            self.assertEqual(claim.read_bytes(), claim_bytes)
            self.assertEqual(completion.read_bytes(), completion_bytes)
            self.assertEqual(len(api.create_commit_calls), 1)
            self.assertEqual(api.create_tag_calls, 1)

    def test_existing_nonidentical_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            api = _FakeHF()
            claim = Path(fixture[1].local_state["claim_path"])
            claim.write_bytes(b"wrong\n")
            os.chmod(claim, 0o600)
            with self.assertRaisesRegex(ValueError, "not byte-identical"):
                _publish(api, fixture)
            self.assertEqual(api.create_commit_calls, [])
            self.assertEqual(api.create_tag_calls, 0)

    def test_create_repo_response_loss_occurs_after_durable_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            api = _FakeHF()
            api.fail_create_repo_response_once = True
            claim = Path(fixture[1].local_state["claim_path"])
            completion_path = Path(
                fixture[1].local_state["completion_seal_path"]
            )
            with self.assertRaisesRegex(ConnectionError, "create_repo response loss"):
                _publish(api, fixture)
            self.assertTrue(claim.is_file())
            self.assertFalse(completion_path.exists())
            completion = _publish(api, fixture)
            self.assertEqual(completion["remote_state_before_publish"], REMOTE_EMPTY)
            self.assertEqual(len(api.create_commit_calls), 1)

    def test_intermediate_download_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            prepared = fixture[2]
            api = _FakeHF()
            api.seed_pair(prepared.archive_bytes, prepared.sidecar_bytes, tagged=True)
            api.malicious_intermediate_symlink = True
            with self.assertRaisesRegex(
                ValueError, "symlink or non-directory component"
            ):
                _publish(api, fixture)
            self.assertEqual(api.create_commit_calls, [])
            self.assertEqual(api.create_tag_calls, 0)

    def test_completed_replay_never_recreates_a_deleted_remote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            api = _FakeHF()
            _publish(api, fixture)
            create_repo_calls = api.create_repo_calls
            api.deleted = True
            with self.assertRaisesRegex(ValueError, "remote repository is deleted"):
                _publish(api, fixture)
            self.assertEqual(api.create_repo_calls, create_repo_calls)
            self.assertEqual(len(api.create_commit_calls), 1)
            self.assertEqual(api.create_tag_calls, 1)

    def test_tag_move_during_fresh_download_prevents_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _publication_fixture(Path(directory))
            api = _FakeHF()
            # note (luojiaxuan): Calls 1-2 verify the new pair commit; call 3
            # is the archive fetch inside the final fresh-attestation window.
            api.move_tag_on_download_call = 3
            completion_path = Path(
                fixture[1].local_state["completion_seal_path"]
            )
            with self.assertRaisesRegex(ValueError, "drifted after fresh download"):
                _publish(api, fixture)
            self.assertEqual(api.tags[DESTINATION["tag"]], BASE_COMMIT)
            self.assertFalse(completion_path.exists())

    def test_relative_fresh_download_parent_fails_before_claim_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p0, contract, prepared, _downloads = _publication_fixture(
                Path(directory)
            )
            api = _FakeHF()
            with self.assertRaisesRegex(ValueError, "must be an absolute"):
                publish_invalid_forensic_archive(
                    api=api,
                    download_fn=api.download,
                    operation_factory=_operation_factory,
                    contract=contract,
                    p0_contract=p0,
                    prepared=prepared,
                    fresh_download_parent=Path("."),
                )
            self.assertFalse(Path(contract.local_state["claim_path"]).exists())
            self.assertEqual(api.create_repo_calls, 0)


if __name__ == "__main__":
    unittest.main()
