from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from causalcache.restoration_v2_2_eager_artifact import (
    ARCHIVE_LEDGER_NAME,
    WORKER_DIRECTORY,
    build_v22_artifact_manifest,
    collect_raw_v22_evidence,
    deterministic_tar_bytes,
    expected_complete_file_names,
    pretty_json_bytes,
    read_v22_evidence_archive,
    validate_v22_evidence_files,
)
from causalcache.restoration_v2_2_eager_contract import (
    CANONICAL_HF_PATH,
    CANONICAL_HF_REPO,
)
import causalcache.restoration_v2_2_eager_artifact as artifact_module
from tests.test_run_restoration_v2_2_eager_substrate import (
    SOURCE_COMMIT,
    deleted_root_ledger_attempt,
    fake_run_contract,
    successful_attempt,
)
from scripts import run_restoration_v2_2_eager_substrate as runner


def _files(root: Path, ledger: Path) -> dict[str, bytes]:
    import json

    result = {ARCHIVE_LEDGER_NAME: ledger.read_bytes()}
    global_record = json.loads(ledger.read_text())
    for worker_id, path in global_record["worker_sibling_ledgers"].items():
        result[f"worker_sibling_ledgers/{worker_id}.json"] = Path(path).read_bytes()
    for path in root.rglob("*"):
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


def _json_bytes_mutation(payload: bytes, mutate: object) -> bytes:
    import json

    value = json.loads(payload)
    mutate(value)
    return pretty_json_bytes(value)


class V22ArtifactTest(unittest.TestCase):
    def test_prebound_only_invalid_evidence_is_packageable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            result = runner.execute_two_worker_attempt(
                run_contract=fake_run_contract(root, ledger),
                output_dir=root,
                global_ledger=ledger,
                worker_launcher=lambda layout, specs: None,
            )
            evidence = collect_raw_v22_evidence(
                root,
                ledger,
                expected_source_git_commit=SOURCE_COMMIT,
                require_canonical_location=False,
            )
        self.assertEqual(result["outcome"], runner.INVALID_OUTCOME)
        self.assertEqual(evidence.outcome, runner.INVALID_OUTCOME)
        self.assertEqual(evidence.attempted_state_count, 0)

    def test_deleted_root_ledger_preserves_sibling_truth_in_invalid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, root, ledger = deleted_root_ledger_attempt(Path(directory))
            evidence = collect_raw_v22_evidence(
                root,
                ledger,
                expected_source_git_commit=SOURCE_COMMIT,
                require_canonical_location=False,
            )
        self.assertEqual(result["outcome"], runner.INVALID_OUTCOME)
        self.assertEqual(evidence.attempted_state_count, 45)
        self.assertEqual(evidence.completed_state_count, 45)

    def test_stale_root_prefix_is_preserved_in_invalid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            layout = runner.claim_global_attempt(
                run_contract=fake_run_contract(root, ledger),
                output_dir=root,
                global_ledger=ledger,
            )
            even = runner.expected_worker_specs()[0]
            stale = runner._worker_ledger_record(
                layout=layout,
                spec=even,
                status=runner.WORKER_CLAIM_STATUS,
                claimed_at_utc=layout.started_at_utc,
                attempted=(),
                completed=(),
            )
            root_ledger = (
                root / "workers" / "even" / "worker_attempt_ledger.json"
            )
            runner._write_json_exclusive(root_ledger, stale)
            advanced = runner._worker_ledger_record(
                layout=layout,
                spec=even,
                status="WORKER_SHARD_RUNNING_NO_RETRY",
                claimed_at_utc=layout.started_at_utc,
                attempted=(0,),
                completed=(),
            )
            runner._replace_json_durable(
                layout.worker_sibling_ledgers["even"], advanced
            )
            result = runner.seal_interrupted_attempt(
                output_dir=root,
                global_ledger=ledger,
            )
            evidence = collect_raw_v22_evidence(
                root,
                ledger,
                expected_source_git_commit=SOURCE_COMMIT,
                require_canonical_location=False,
            )
        self.assertEqual(evidence.outcome, runner.INVALID_OUTCOME)
        self.assertEqual(evidence.attempted_state_count, 1)
        forensic = result["forensic_inventory"]["even"]
        self.assertEqual(forensic["root_ledger_status"], "missing_or_stale")
        self.assertEqual(forensic["root_ledger_attempted_state_indices"], [])
        self.assertEqual(forensic["sibling_attempted_state_indices"], [0])

    def test_invalid_root_ledger_ahead_of_sibling_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            layout = runner.claim_global_attempt(
                run_contract=fake_run_contract(root, ledger),
                output_dir=root,
                global_ledger=ledger,
            )
            even = runner.expected_worker_specs()[0]
            ahead = runner._worker_ledger_record(
                layout=layout,
                spec=even,
                status="WORKER_SHARD_RUNNING_NO_RETRY",
                claimed_at_utc=layout.started_at_utc,
                attempted=(0,),
                completed=(),
            )
            root_ledger = (
                root / "workers" / "even" / "worker_attempt_ledger.json"
            )
            runner._write_json_exclusive(root_ledger, ahead)
            runner.seal_interrupted_attempt(output_dir=root, global_ledger=ledger)
            files = _files(root, ledger)
            with self.assertRaisesRegex(ValueError, "not a sibling prefix"):
                validate_v22_evidence_files(
                    files, require_canonical_attempt_identity=False
                )

    def test_complete_raw_tree_reduces_and_packages_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, root, ledger = successful_attempt(Path(directory))
            evidence = collect_raw_v22_evidence(
                root,
                ledger,
                expected_source_git_commit=SOURCE_COMMIT,
                require_canonical_location=False,
            )
            first = deterministic_tar_bytes(evidence.files)
            second = deterministic_tar_bytes(evidence.files)
            archive = Path(directory) / "evidence.tar"
            archive.write_bytes(first)
            reread = read_v22_evidence_archive(
                archive,
                expected_source_git_commit=SOURCE_COMMIT,
                require_canonical_attempt_identity=False,
            )
        self.assertEqual(result["outcome"], evidence.outcome)
        self.assertEqual(first, second)
        self.assertEqual(reread.inventory, evidence.inventory)
        self.assertEqual(set(evidence.files), expected_complete_file_names())

    def test_manifest_requires_independent_byte_identical_fresh_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, root, ledger = successful_attempt(base)
            evidence = collect_raw_v22_evidence(
                root,
                ledger,
                expected_source_git_commit=SOURCE_COMMIT,
                require_canonical_location=False,
            )
            payload = deterministic_tar_bytes(evidence.files)
            source = base / "source.tar"
            fresh = base / "fresh-download.tar"
            source.write_bytes(payload)
            fresh.write_bytes(payload)
            with mock.patch.object(
                artifact_module,
                "read_v22_evidence_archive",
                return_value=evidence,
            ):
                manifest = build_v22_artifact_manifest(
                    evidence=evidence,
                    source_archive_path=source,
                    fresh_immutable_archive=fresh,
                    hf_repo=CANONICAL_HF_REPO,
                    hf_immutable_revision="b" * 40,
                    hf_path=CANONICAL_HF_PATH,
                )
                with self.assertRaisesRegex(ValueError, "independent"):
                    build_v22_artifact_manifest(
                        evidence=evidence,
                        source_archive_path=source,
                        fresh_immutable_archive=source,
                        hf_repo=CANONICAL_HF_REPO,
                        hf_immutable_revision="b" * 40,
                        hf_path=CANONICAL_HF_PATH,
                    )
                fresh.write_bytes(payload + b"tampered")
                with self.assertRaisesRegex(ValueError, "differs"):
                    build_v22_artifact_manifest(
                        evidence=evidence,
                        source_archive_path=source,
                        fresh_immutable_archive=fresh,
                        hf_repo=CANONICAL_HF_REPO,
                        hf_immutable_revision="b" * 40,
                        hf_path=CANONICAL_HF_PATH,
                    )
        self.assertIs(
            manifest["hf_artifact"]["fresh_immutable_download_verified"], True
        )
        self.assertEqual(manifest["hf_artifact"]["immutable_revision"], "b" * 40)

    def test_manifest_rejects_unfrozen_hf_provenance(self) -> None:
        evidence = mock.Mock(source_git_commit=SOURCE_COMMIT)
        with self.assertRaisesRegex(ValueError, "full commit SHA"):
            build_v22_artifact_manifest(
                evidence=evidence,
                source_archive_path="/tmp/source.tar",
                fresh_immutable_archive="/tmp/fresh.tar",
                hf_repo=CANONICAL_HF_REPO,
                hf_immutable_revision="short",
                hf_path=CANONICAL_HF_PATH,
            )
        for repo, path in (
            ("someone/other", CANONICAL_HF_PATH),
            (CANONICAL_HF_REPO, "raw/other.tar"),
        ):
            with self.subTest(repo=repo, path=path), self.assertRaisesRegex(
                ValueError, "repo or path"
            ):
                build_v22_artifact_manifest(
                    evidence=evidence,
                    source_archive_path="/tmp/source.tar",
                    fresh_immutable_archive="/tmp/fresh.tar",
                    hf_repo=repo,
                    hf_immutable_revision="b" * 40,
                    hf_path=path,
                )

    def test_tampered_canonical_action_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, root, ledger = successful_attempt(Path(directory))
            files = _files(root, ledger)
            name = f"{WORKER_DIRECTORY}/even/states/000.json"

            def mutate(value: dict[str, object]) -> None:
                kernel = value["measurement_kernel"]
                kernel["record"]["canonical_action"]["action"] = "press_back"

            files[name] = _json_bytes_mutation(files[name], mutate)
            with self.assertRaisesRegex(ValueError, "teacher evidence|canonical"):
                validate_v22_evidence_files(
                    files, require_canonical_attempt_identity=False
                )

    def test_tampered_teacher_distance_audit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, root, ledger = successful_attempt(Path(directory))
            files = _files(root, ledger)
            name = f"{WORKER_DIRECTORY}/odd/states/001.json"

            def mutate(value: dict[str, object]) -> None:
                kernel = value["measurement_kernel"]
                audit = kernel["record"]["distance_audits"]["summary_reference_kl"]
                audit["full_tensor_host_transfers"] = 1

            files[name] = _json_bytes_mutation(files[name], mutate)
            with self.assertRaisesRegex(ValueError, "distance audit"):
                validate_v22_evidence_files(
                    files, require_canonical_attempt_identity=False
                )

    def test_tampered_generation_closer_token_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, root, ledger = successful_attempt(Path(directory))
            files = _files(root, ledger)
            name = f"{WORKER_DIRECTORY}/even/states/000.json"

            def mutate(value: dict[str, object]) -> None:
                generation = value["measurement_kernel"]["record"][
                    "native_generations"
                ][0]
                generation["metadata"]["final_generated_token_id"] = 7

            files[name] = _json_bytes_mutation(files[name], mutate)
            with self.assertRaisesRegex(ValueError, "generation evidence"):
                validate_v22_evidence_files(
                    files, require_canonical_attempt_identity=False
                )

    def test_wrong_worker_device_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, root, ledger = successful_attempt(Path(directory))
            files = _files(root, ledger)
            name = f"{WORKER_DIRECTORY}/odd/states/001.json"

            def mutate(value: dict[str, object]) -> None:
                value["worker"]["device"] = "cuda:0"

            files[name] = _json_bytes_mutation(files[name], mutate)
            with self.assertRaisesRegex(ValueError, "state or measurement-kernel"):
                validate_v22_evidence_files(
                    files, require_canonical_attempt_identity=False
                )

    def test_wrong_runtime_stack_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, root, ledger = successful_attempt(Path(directory))
            files = _files(root, ledger)
            name = f"{WORKER_DIRECTORY}/even/runtime_identity.json"

            def mutate(value: dict[str, object]) -> None:
                value["runtime_metadata"]["torch_version"] = "2.10.0"

            files[name] = _json_bytes_mutation(files[name], mutate)
            with self.assertRaisesRegex(ValueError, "runtime metadata"):
                validate_v22_evidence_files(
                    files, require_canonical_attempt_identity=False
                )

    def test_partial_or_unknown_file_inventory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, root, ledger = successful_attempt(Path(directory))
            files = _files(root, ledger)
            partial = copy.deepcopy(files)
            del partial[f"{WORKER_DIRECTORY}/odd/states/043.json"]
            with self.assertRaises(ValueError):
                validate_v22_evidence_files(
                    partial, require_canonical_attempt_identity=False
                )
            files["unexpected.json"] = pretty_json_bytes({"unexpected": True})
            with self.assertRaisesRegex(ValueError, "inventory"):
                validate_v22_evidence_files(
                    files, require_canonical_attempt_identity=False
                )

    def test_rolled_back_worker_ledger_is_rejected_by_global_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, root, ledger = successful_attempt(Path(directory))
            files = _files(root, ledger)
            name = f"{WORKER_DIRECTORY}/even/worker_attempt_ledger.json"

            def mutate(value: dict[str, object]) -> None:
                value["completed_state_indices"] = value["completed_state_indices"][:-1]

            files[name] = _json_bytes_mutation(files[name], mutate)
            with self.assertRaisesRegex(ValueError, "root and sibling ledgers"):
                validate_v22_evidence_files(
                    files, require_canonical_attempt_identity=False
                )

    def test_tampered_parent_conclusion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            contract = fake_run_contract(base / "raw", base / "attempt.json")
            contract["parent_authorization"]["v2_1_full_45_evidence"][
                "validation_outcome"
            ] = "PASS_V2_1_FULL_45_SUBSTRATE"
            with self.assertRaisesRegex(ValueError, "parent authorization"):
                artifact_module._validate_run_contract_identity(
                    contract, require_canonical_attempt_identity=False
                )


if __name__ == "__main__":
    unittest.main()
