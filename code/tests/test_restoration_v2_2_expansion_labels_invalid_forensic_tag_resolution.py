from __future__ import annotations

import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from causalcache.restoration_v2_2_expansion_labels_invalid_forensic import (
    ARCHIVE_STATUS,
    ARTIFACT_CLASS,
    package_invalid_forensic_archive,
    pretty_json_bytes,
    sha256_bytes,
)
from causalcache.restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution import (
    CANONICAL_CONFIG_PATH,
    COMPLETION_STATUS,
    FROZEN_CONFIG_SHA256,
    PAIR_COMMIT_TITLE,
    load_frozen_tag_resolution_contract,
    tag_resolution_contract_from_data,
    validate_clean_pushed_source,
    validate_remote_tag_resolution,
)
from scripts import (
    manage_restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution as manager,
)
from tests.test_restoration_v2_2_expansion_labels_invalid_forensic import _fixture


ROOT = Path(__file__).resolve().parents[2]
PAIR = "5efe1ae861d16e2ee144ed5f4c7b5ad25a28b416"
PREDECESSOR = "1e1bb828d196779e4fc855890922c4865fcd6458"
TAG_OBJECT = "ca652858c3d59eab066eb7b31399690a65ea5f44"
DRIFT = "d" * 40


class _ReadOnlyFakeHF:
    def __init__(
        self,
        *,
        remote: dict[str, object],
        archive: bytes,
        sidecar: bytes,
    ) -> None:
        self.remote = remote
        self.archive = archive
        self.sidecar = sidecar
        self.private = True
        self.main_calls = 0
        self.tag_calls = 0
        self.immutable_calls = 0
        self.ref_calls = 0
        self.read_calls = 0
        self.download_calls: list[dict[str, object]] = []
        self.tag_object_drift = False
        self.tag_resolved_drift = False
        self.main_drift = False
        self.immutable_drift = False
        self.split_pair = False
        self.bytes_mismatch = False
        self.symlink_download = False
        self.read_error_at: int | None = None
        self.parent_completion_on_download: Path | None = None
        self.title_drift = False
        self.extra_pair_file = False

    def _read(self) -> None:
        self.read_calls += 1
        if self.read_error_at == self.read_calls:
            raise ConnectionError("synthetic read response failure")

    def dataset_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
        self._read()
        if repo_id != self.remote["repo"]:
            raise AssertionError("wrong read-only repository")
        if revision == "main":
            self.main_calls += 1
            sha = DRIFT if self.main_drift and self.main_calls > 1 else PAIR
        elif revision == self.remote["tag"]:
            self.tag_calls += 1
            sha = DRIFT if self.tag_resolved_drift and self.tag_calls > 1 else PAIR
        elif revision == PAIR:
            self.immutable_calls += 1
            sha = DRIFT if self.immutable_drift and self.immutable_calls > 1 else PAIR
        else:
            raise AssertionError(f"unexpected revision: {revision}")
        return SimpleNamespace(sha=sha, private=self.private)

    def list_repo_refs(self, repo_id: str, *, repo_type: str) -> SimpleNamespace:
        self._read()
        if repo_id != self.remote["repo"] or repo_type != "dataset":
            raise AssertionError("wrong read-only repository")
        self.ref_calls += 1
        target = DRIFT if self.tag_object_drift and self.ref_calls > 1 else TAG_OBJECT
        return SimpleNamespace(
            tags=[SimpleNamespace(name=self.remote["tag"], target_commit=target)]
        )

    def list_repo_commits(
        self, repo_id: str, *, repo_type: str, revision: str
    ) -> list[SimpleNamespace]:
        self._read()
        if (
            repo_id != self.remote["repo"]
            or repo_type != "dataset"
            or revision != PAIR
        ):
            raise AssertionError("wrong pair-history read")
        return [
            SimpleNamespace(
                commit_id=PAIR,
                title="drifted title" if self.title_drift else PAIR_COMMIT_TITLE,
            ),
            SimpleNamespace(commit_id=PREDECESSOR, title="System commit"),
        ]

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str
    ) -> list[str]:
        self._read()
        if repo_id != self.remote["repo"] or repo_type != "dataset":
            raise AssertionError("wrong file-inventory read")
        if revision == PAIR:
            files = list(self.remote["pair_commit"]["files"])
            if self.extra_pair_file:
                files.append("unexpected.txt")
            return sorted(files)
        if revision == PREDECESSOR:
            files = list(self.remote["pair_commit"]["predecessor_files"])
            if self.split_pair:
                files.append(str(self.remote["archive_path"]))
            return sorted(files)
        raise AssertionError("unexpected file revision")

    def download(self, **kwargs: object) -> str:
        self._read()
        expected = {
            "repo_id",
            "filename",
            "repo_type",
            "revision",
            "local_dir",
            "force_download",
        }
        if set(kwargs) != expected:
            raise AssertionError("download arguments drifted")
        if (
            kwargs["repo_id"] != self.remote["repo"]
            or kwargs["repo_type"] != "dataset"
            or kwargs["revision"] != PAIR
            or kwargs["force_download"] is not True
        ):
            raise AssertionError("download must be immutable and forced")
        directory = Path(kwargs["local_dir"])
        started_empty = not any(directory.iterdir()) if not self.download_calls else None
        filename = str(kwargs["filename"])
        if self.symlink_download and not self.download_calls:
            escape = directory.parent / f"{directory.name}-escape"
            escape.mkdir()
            (directory / "attempts").symlink_to(escape, target_is_directory=True)
        output = directory / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        if filename == self.remote["archive_path"]:
            payload = self.archive
        elif filename == self.remote["sidecar_path"]:
            payload = self.sidecar
        else:
            raise AssertionError("unexpected download path")
        if self.bytes_mismatch and filename == self.remote["sidecar_path"]:
            payload += b"x"
        output.write_bytes(payload)
        self.download_calls.append(
            {
                "filename": filename,
                "force_download": True,
                "started_empty": started_empty,
            }
        )
        if (
            self.parent_completion_on_download is not None
            and len(self.download_calls) == 2
        ):
            self.parent_completion_on_download.write_bytes(b"unexpected-mid-replay")
            os.chmod(self.parent_completion_on_download, 0o600)
        return str(output.resolve() if self.symlink_download else output)

    def create_repo(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only repair called create_repo")

    def create_commit(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only repair called create_commit")

    def create_tag(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only repair called create_tag")

    def delete_repo(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only repair called delete_repo")

    def move_repo(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only repair called move_repo")


def _source_identity() -> dict[str, object]:
    return {
        "tag_resolution_source_git_commit": "a" * 40,
        "origin_main_git_commit": "a" * 40,
        "branch": "main",
        "origin_url": "https://github.com/luojiaxuan/CausalCache.git",
        "source_path_count": 7,
        "source_inventory_sha256": "b" * 64,
        "parent_publication_source_git_commit": (
            "3941d82a8ad83daaec55726a799d4d200735428d"
        ),
        "parent_source_inventory_sha256": (
            "d4ec445884bc2296c824e5f90718a1ede0be13023d5d969f77adf379a1c29cfa"
        ),
        "failure_evidence_commit": "dcc37e5ce62b844a1814a9f0fe29ec4827f2547a",
        "failure_evidence_sha256": (
            "7dd71de55e7adc2080154caa2679d3cdd23fae96808d07a51df7db1a6a6f5c8b"
        ),
    }


def _publication_source() -> dict[str, object]:
    return {
        "publication_config_sha256": (
            "affb54cf44a656394c983cebe5d004d3e1ff216537c43da942dc031fa6baaccb"
        ),
        "publication_source_git_commit": (
            "3941d82a8ad83daaec55726a799d4d200735428d"
        ),
        "origin_main_git_commit": "3941d82a8ad83daaec55726a799d4d200735428d",
        "branch": "main",
        "origin_url": "https://github.com/luojiaxuan/CausalCache.git",
        "source_path_count": 5,
        "source_inventory_sha256": (
            "d4ec445884bc2296c824e5f90718a1ede0be13023d5d969f77adf379a1c29cfa"
        ),
    }


def _remote_fixture(directory: Path):
    p0, _data, root, global_ledger, workers, archive_path = _fixture(directory)
    packaged = package_invalid_forensic_archive(
        contract=p0,
        raw_output_dir=root,
        external_global_ledger=global_ledger,
        external_worker_ledgers=workers,
        output_archive=archive_path,
    )
    config = json.loads((ROOT / CANONICAL_CONFIG_PATH).read_bytes())
    state = directory / "state"
    state.mkdir()
    parent_claim_path = state / "parent-claim.json"
    parent_completion_path = state / "parent-completion.json"
    child_claim_path = state / "child-claim.json"
    child_completion_path = state / "child-completion.json"
    config["child_local_state"]["claim_path"] = str(child_claim_path.resolve())
    config["child_local_state"]["completion_seal_path"] = str(
        child_completion_path.resolve()
    )
    config["p0_forensic_source"].update(
        {
            "config_sha256": p0.sha256,
            "archive_sha256": packaged["archive_sha256"],
            "archive_size_bytes": packaged["archive_size_bytes"],
            "archive_member_count": packaged["member_count"],
            "tree_inventory_sha256": packaged["tree_inventory_sha256"],
        }
    )
    config["parent_publication_attempt"]["claim"]["path"] = str(
        parent_claim_path.resolve()
    )
    config["parent_publication_attempt"]["completion_seal"]["path"] = str(
        parent_completion_path.resolve()
    )
    remote = config["remote_evidence"]
    remote["archive_sha256"] = packaged["archive_sha256"]
    source_archive = {
        "p0_protocol_id": p0.data["protocol_id"],
        "p0_archive_status": ARCHIVE_STATUS,
        "p0_config_path": config["p0_forensic_source"]["config_path"],
        "p0_config_sha256": p0.sha256,
        "archive_sha256": packaged["archive_sha256"],
        "archive_size_bytes": packaged["archive_size_bytes"],
        "tree_inventory_sha256": packaged["tree_inventory_sha256"],
        "member_count": packaged["member_count"],
    }
    destination = {
        "archive_path": remote["archive_path"],
        "archive_sidecar_single_commit_required": True,
        "private": True,
        "repo": remote["repo"],
        "repo_type": "dataset",
        "sidecar_path": remote["sidecar_path"],
        "tag": remote["tag"],
    }
    sidecar = {
        "schema_version": "1.0.0",
        "protocol_id": config["parent_publication_attempt"]["protocol_id"],
        "status": "IMMUTABLE_INVALID_FORENSIC_ARCHIVE_PUBLICATION_MANIFEST_V1",
        "artifact_class": ARTIFACT_CLASS,
        "destination": destination,
        "source_archive": source_archive,
        "publication_source": _publication_source(),
        "formal_consumption": {
            "formal_label_loader_eligible": False,
            "gate_training_unlocked": False,
            "producer_attempt_reclassified": False,
        },
    }
    sidecar_bytes = pretty_json_bytes(sidecar)
    remote["sidecar_sha256"] = sha256_bytes(sidecar_bytes)
    parent_claim = {
        "schema_version": "1.0.0",
        "protocol_id": config["parent_publication_attempt"]["protocol_id"],
        "status": "CLAIMED_INVALID_FORENSIC_ARCHIVE_PUBLICATION_V1",
        "destination": destination,
        "archive_sha256": packaged["archive_sha256"],
        "sidecar_sha256": sha256_bytes(sidecar_bytes),
        "publication_source": _publication_source(),
    }
    parent_claim_bytes = pretty_json_bytes(parent_claim)
    parent_claim_path.write_bytes(parent_claim_bytes)
    os.chmod(parent_claim_path, 0o600)
    config["parent_publication_attempt"]["claim"].update(
        {
            "sha256": sha256_bytes(parent_claim_bytes),
            "size_bytes": len(parent_claim_bytes),
        }
    )
    contract = tag_resolution_contract_from_data(config)
    downloads = directory / "downloads"
    downloads.mkdir()
    fake = _ReadOnlyFakeHF(
        remote=remote,
        archive=archive_path.read_bytes(),
        sidecar=sidecar_bytes,
    )
    return (
        p0,
        contract,
        fake,
        downloads,
        parent_claim_path,
        parent_completion_path,
        child_claim_path,
        child_completion_path,
    )


def _run(fixture):
    p0, contract, fake, downloads, *_ = fixture
    return validate_remote_tag_resolution(
        api=fake,
        download_fn=fake.download,
        contract=contract,
        p0_contract=p0,
        source_identity=_source_identity(),
        fresh_download_parent=downloads,
    )


class ExpansionLabelsInvalidForensicTagResolutionTest(unittest.TestCase):
    def test_frozen_contract_separates_tag_object_and_resolved_commit(self) -> None:
        contract = load_frozen_tag_resolution_contract(
            CANONICAL_CONFIG_PATH, repository_root=ROOT
        )
        self.assertEqual(contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(contract.remote["tag_object_identity"], TAG_OBJECT)
        self.assertEqual(contract.remote["tag_resolved_commit"], PAIR)
        self.assertNotEqual(
            contract.remote["tag_object_identity"],
            contract.remote["tag_resolved_commit"],
        )
        self.assertFalse(
            contract.data["formal_consumption"]["formal_label_loader_eligible"]
        )

    def test_cli_exposes_only_explicit_read_only_remote_command(self) -> None:
        choices = manager._parser()._subparsers._group_actions[0].choices
        self.assertEqual(
            set(choices),
            {"validate-contract", "validate-local-state", "validate-remote"},
        )
        token_actions = [
            action
            for command in choices.values()
            for action in command._actions
            if action.dest == "hf_token_file"
        ]
        self.assertEqual(len(token_actions), 1)
        self.assertTrue(token_actions[0].required)

    def test_token_reader_requires_absolute_0400_or_0600_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token = Path(directory).resolve() / "token"
            token.write_text("synthetic-token\n", encoding="utf-8")
            os.chmod(token, 0o600)
            self.assertEqual(manager._read_hf_token(token), "synthetic-token")
            os.chmod(token, 0o644)
            with self.assertRaisesRegex(ValueError, "0400 or 0600"):
                manager._read_hf_token(token)
            os.chmod(token, 0o400)
            link = token.with_name("token-link")
            link.symlink_to(token)
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                manager._read_hf_token(link)
            with self.assertRaisesRegex(ValueError, "absolute"):
                manager._read_hf_token(Path("relative-token"))

    def test_source_root_rejects_relative_and_symlink_paths(self) -> None:
        contract = load_frozen_tag_resolution_contract(
            CANONICAL_CONFIG_PATH, repository_root=ROOT
        )
        with self.assertRaisesRegex(ValueError, "absolute and non-symlink"):
            validate_clean_pushed_source(
                repository_root=Path("."), contract=contract
            )
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "repo-link"
            link.symlink_to(ROOT, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "absolute and non-symlink"):
                validate_clean_pushed_source(repository_root=link, contract=contract)

    def test_happy_path_is_read_only_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            result = _run(fixture)
            *_, parent_claim, parent_completion, child_claim, child_completion = fixture
            self.assertEqual(result["status"], COMPLETION_STATUS)
            self.assertEqual(result["remote_mutation_call_count"], 0)
            self.assertTrue(
                result["remote_attestation"]["fresh_download"][
                    "p0_strict_archive_readback_passed"
                ]
            )
            self.assertEqual(
                result["remote_attestation"]["fresh_download"]["archive_sha256"],
                fixture[1].remote["archive_sha256"],
            )
            self.assertEqual(
                result["remote_attestation"]["fresh_download"]["sidecar_sha256"],
                fixture[1].remote["sidecar_sha256"],
            )
            self.assertFalse(parent_completion.exists())
            self.assertEqual(oct(parent_claim.stat().st_mode & 0o777), "0o600")
            self.assertEqual(oct(child_claim.stat().st_mode & 0o777), "0o600")
            self.assertEqual(oct(child_completion.stat().st_mode & 0o777), "0o600")
            first = child_completion.read_bytes()
            repeated = _run(fixture)
            self.assertEqual(repeated, result)
            self.assertEqual(child_completion.read_bytes(), first)

    def test_tag_object_drift_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].tag_object_drift = True
            with self.assertRaisesRegex(ValueError, "identity drifted"):
                _run(fixture)
            self.assertFalse(fixture[-1].exists())

    def test_tag_resolved_drift_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].tag_resolved_drift = True
            with self.assertRaisesRegex(ValueError, "identity drifted"):
                _run(fixture)
            self.assertFalse(fixture[-1].exists())

    def test_main_drift_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].main_drift = True
            with self.assertRaisesRegex(ValueError, "identity drifted"):
                _run(fixture)
            self.assertFalse(fixture[-1].exists())

    def test_immutable_revision_drift_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].immutable_drift = True
            with self.assertRaisesRegex(ValueError, "identity drifted"):
                _run(fixture)
            self.assertFalse(fixture[-1].exists())

    def test_non_private_repository_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].private = False
            with self.assertRaisesRegex(ValueError, "remain private"):
                _run(fixture)
            self.assertFalse(fixture[-1].exists())

    def test_pair_split_commit_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].split_pair = True
            with self.assertRaisesRegex(ValueError, "inventory drifted"):
                _run(fixture)
            self.assertFalse(fixture[-1].exists())

    def test_pair_title_or_extra_file_drift_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].title_drift = True
            with self.assertRaisesRegex(ValueError, "title or direct history"):
                _run(fixture)
            self.assertFalse(fixture[-1].exists())
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].extra_pair_file = True
            with self.assertRaisesRegex(ValueError, "inventory drifted"):
                _run(fixture)
            self.assertFalse(fixture[-1].exists())

    def test_downloaded_bytes_mismatch_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].bytes_mismatch = True
            with self.assertRaisesRegex(ValueError, "bytes drifted"):
                _run(fixture)
            self.assertFalse(fixture[-1].exists())

    def test_sidecar_source_mismatch_fails_after_exact_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            bad_sidecar = json.loads(fixture[2].sidecar)
            bad_sidecar["publication_source"]["source_inventory_sha256"] = "f" * 64
            bad_bytes = pretty_json_bytes(bad_sidecar)
            config = deepcopy(fixture[1].data)
            config["remote_evidence"]["sidecar_sha256"] = sha256_bytes(bad_bytes)
            parent_claim = json.loads(fixture[4].read_bytes())
            parent_claim["sidecar_sha256"] = sha256_bytes(bad_bytes)
            parent_claim_bytes = pretty_json_bytes(parent_claim)
            fixture[4].write_bytes(parent_claim_bytes)
            os.chmod(fixture[4], 0o600)
            config["parent_publication_attempt"]["claim"]["sha256"] = (
                sha256_bytes(parent_claim_bytes)
            )
            config["parent_publication_attempt"]["claim"]["size_bytes"] = len(
                parent_claim_bytes
            )
            contract = tag_resolution_contract_from_data(config)
            fixture[2].remote = contract.remote
            fixture[2].sidecar = bad_bytes
            with self.assertRaisesRegex(ValueError, "semantic binding drifted"):
                validate_remote_tag_resolution(
                    api=fixture[2],
                    download_fn=fixture[2].download,
                    contract=contract,
                    p0_contract=fixture[0],
                    source_identity=_source_identity(),
                    fresh_download_parent=fixture[3],
                )
            self.assertGreater(fixture[2].read_calls, 0)
            self.assertTrue(fixture[-2].exists())
            self.assertFalse(fixture[-1].exists())

    def test_parent_p1_completion_unexpectedly_present_blocks_all_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[5].write_bytes(b"unexpected")
            os.chmod(fixture[5], 0o600)
            with self.assertRaisesRegex(ValueError, "must remain absent"):
                _run(fixture)
            self.assertEqual(fixture[2].read_calls, 0)
            self.assertFalse(fixture[-1].exists())

    def test_parent_p1_completion_appearing_mid_replay_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].parent_completion_on_download = fixture[5]
            with self.assertRaisesRegex(ValueError, "must remain absent"):
                _run(fixture)
            self.assertTrue(fixture[5].exists())
            self.assertTrue(fixture[-2].exists())
            self.assertFalse(fixture[-1].exists())

    def test_parent_claim_byte_or_mode_drift_blocks_all_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[4].write_bytes(fixture[4].read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "bytes drifted"):
                _run(fixture)
            self.assertEqual(fixture[2].read_calls, 0)
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            os.chmod(fixture[4], 0o644)
            with self.assertRaisesRegex(ValueError, "mode drifted"):
                _run(fixture)
            self.assertEqual(fixture[2].read_calls, 0)

    def test_injected_source_lineage_fails_before_claim_or_remote_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            source = _source_identity()
            source["failure_evidence_commit"] = "e" * 40
            with self.assertRaisesRegex(ValueError, "frozen clean-main lineage"):
                validate_remote_tag_resolution(
                    api=fixture[2],
                    download_fn=fixture[2].download,
                    contract=fixture[1],
                    p0_contract=fixture[0],
                    source_identity=source,
                    fresh_download_parent=fixture[3],
                )
            self.assertEqual(fixture[2].read_calls, 0)
            self.assertFalse(fixture[-2].exists())
            self.assertFalse(fixture[-1].exists())

    def test_read_response_error_leaves_claim_without_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].read_error_at = 5
            with self.assertRaisesRegex(ConnectionError, "response failure"):
                _run(fixture)
            self.assertTrue(fixture[-2].exists())
            self.assertFalse(fixture[-1].exists())

    def test_orphan_child_completion_fails_before_remote_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[-1].write_bytes(b"orphan")
            os.chmod(fixture[-1], 0o600)
            with self.assertRaisesRegex(ValueError, "orphan"):
                _run(fixture)
            self.assertEqual(fixture[2].read_calls, 0)

    def test_symlink_and_relative_fresh_paths_are_rejected_without_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            link = Path(directory) / "download-link"
            link.symlink_to(fixture[3], target_is_directory=True)
            p0, contract, fake, _downloads, *_ = fixture
            with self.assertRaisesRegex(ValueError, "absolute non-symlink"):
                validate_remote_tag_resolution(
                    api=fake,
                    download_fn=fake.download,
                    contract=contract,
                    p0_contract=p0,
                    source_identity=_source_identity(),
                    fresh_download_parent=link,
                )
            with self.assertRaisesRegex(ValueError, "absolute non-symlink"):
                validate_remote_tag_resolution(
                    api=fake,
                    download_fn=fake.download,
                    contract=contract,
                    p0_contract=p0,
                    source_identity=_source_identity(),
                    fresh_download_parent=Path("relative-downloads"),
                )
            self.assertFalse(fixture[-1].exists())

    def test_download_symlink_escape_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _remote_fixture(Path(directory))
            fixture[2].symlink_download = True
            with self.assertRaisesRegex(ValueError, "lexically unexpected|symlink"):
                _run(fixture)
            self.assertFalse(fixture[-1].exists())


if __name__ == "__main__":
    unittest.main()
