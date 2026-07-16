from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from causalcache.data.guiodyssey_restoration_v2 import sha256_file
from causalcache.spatial_reference_audit_v1_artifact import (
    ARCHIVE_LEDGER_MEMBER,
    ARCHIVE_MEMBER_PREFIX,
    _deterministic_ustar_bytes,
    _read_canonical_archive_bytes,
    collect_spatial_reference_audit_files,
    package_spatial_reference_audit_evidence,
    read_spatial_reference_audit_archive,
)
from scripts.package_spatial_reference_audit_v1 import _build_parser, main


SOURCE_COMMIT = "a" * 40


class SpatialReferenceAuditArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository_root = Path(__file__).resolve().parents[2]
        self.config_path = (
            self.repository_root / "code/configs/spatial_reference_audit_v1.json"
        )

    @staticmethod
    def _write_source_tree(root: Path, config_path: Path) -> tuple[Path, Path]:
        audit_root = root / "audit"
        (audit_root / "profiles/000-bf16_auto").mkdir(parents=True)
        (audit_root / "attempt_start.json").write_text(
            '{"status":"started"}\n',
            encoding="utf-8",
        )
        (audit_root / "profiles/000-bf16_auto/terminal.json").write_text(
            '{"status":"completed"}\n',
            encoding="utf-8",
        )
        ledger = root / ".attempt.json"
        ledger.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "protocol_id": "spatial_reference_audit_v1",
                    "attempt_id": "spatial-reference-audit-v1",
                    "status": "COMPLETED_SPATIAL_REFERENCE_AUDIT_ATTEMPT",
                    "source_git_commit": SOURCE_COMMIT,
                    "config_sha256": sha256_file(config_path),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return audit_root, ledger

    def test_deterministic_sorted_ustar_and_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_root, ledger = self._write_source_tree(root, self.config_path)
            files = collect_spatial_reference_audit_files(audit_root, ledger)
            first = _deterministic_ustar_bytes(files)
            second = _deterministic_ustar_bytes(dict(reversed(list(files.items()))))
            self.assertEqual(first, second)
            self.assertEqual(_read_canonical_archive_bytes(first), files)
            with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
                members = archive.getmembers()
            self.assertEqual([item.name for item in members], sorted(item.name for item in members))
            self.assertTrue(all(item.mode == 0o644 for item in members))
            self.assertTrue(all(item.uid == item.gid == item.mtime == 0 for item in members))
            self.assertIn(ARCHIVE_LEDGER_MEMBER, files)

    def test_collection_rejects_symlink_and_nonregular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit_root, ledger = self._write_source_tree(root, self.config_path)
            link = audit_root / "link.json"
            link.symlink_to(audit_root / "attempt_start.json")
            with self.assertRaisesRegex(ValueError, "symlink or non-regular"):
                collect_spatial_reference_audit_files(audit_root, ledger)
            link.unlink()
            fifo = audit_root / "pipe"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "symlink or non-regular"):
                collect_spatial_reference_audit_files(audit_root, ledger)

    def test_reader_rejects_noncanonical_metadata_and_trailing_bytes(self) -> None:
        payload = b"evidence\n"
        destination = io.BytesIO()
        with tarfile.open(
            fileobj=destination,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as archive:
            info = tarfile.TarInfo(
                f"{ARCHIVE_MEMBER_PREFIX}/{ARCHIVE_LEDGER_MEMBER}"
            )
            info.type = tarfile.REGTYPE
            info.size = len(payload)
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
        with self.assertRaisesRegex(ValueError, "non-canonical member"):
            _read_canonical_archive_bytes(destination.getvalue())

        canonical = _deterministic_ustar_bytes(
            {ARCHIVE_LEDGER_MEMBER: b'{"status":"terminal"}\n'}
        )
        with self.assertRaisesRegex(ValueError, "canonical deterministic USTAR"):
            _read_canonical_archive_bytes(canonical + b"trailing-junk")

    def test_package_is_exclusive_and_revalidates_written_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            config = repository / "code/configs/spatial_reference_audit_v1.json"
            config.parent.mkdir(parents=True)
            config.write_bytes(self.config_path.read_bytes())
            audit_root, ledger = self._write_source_tree(Path(temporary), config)
            output = Path(temporary) / "audit.tar"
            with mock.patch(
                "causalcache.spatial_reference_audit_v1_artifact.validate_repository_inputs"
            ):
                result = package_spatial_reference_audit_evidence(
                    repository_root=repository,
                    config_path=config,
                    source_git_commit=SOURCE_COMMIT,
                    audit_root=audit_root,
                    global_attempt_ledger=ledger,
                    output_archive=output,
                    require_canonical_location=False,
                )
                with self.assertRaises(FileExistsError):
                    package_spatial_reference_audit_evidence(
                        repository_root=repository,
                        config_path=config,
                        source_git_commit=SOURCE_COMMIT,
                        audit_root=audit_root,
                        global_attempt_ledger=ledger,
                        output_archive=output,
                        require_canonical_location=False,
                    )
            self.assertEqual(
                result["status"],
                "PACKAGED_SPATIAL_REFERENCE_AUDIT_V1_RAW_EVIDENCE",
            )
            self.assertEqual(result["archive_sha256"], sha256_file(output))
            self.assertEqual(
                read_spatial_reference_audit_archive(output),
                collect_spatial_reference_audit_files(audit_root, ledger),
            )

    def test_package_rejects_active_or_wrong_source_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            config = repository / "code/configs/spatial_reference_audit_v1.json"
            config.parent.mkdir(parents=True)
            config.write_bytes(self.config_path.read_bytes())
            audit_root, ledger = self._write_source_tree(Path(temporary), config)
            value = json.loads(ledger.read_text())
            value["status"] = "ACTIVE_SPATIAL_REFERENCE_AUDIT_ATTEMPT"
            ledger.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch(
                "causalcache.spatial_reference_audit_v1_artifact.validate_repository_inputs"
            ):
                with self.assertRaisesRegex(ValueError, "not terminal or frozen"):
                    package_spatial_reference_audit_evidence(
                        repository_root=repository,
                        config_path=config,
                        source_git_commit=SOURCE_COMMIT,
                        audit_root=audit_root,
                        global_attempt_ledger=ledger,
                        output_archive=Path(temporary) / "audit.tar",
                        require_canonical_location=False,
                    )

    def test_cli_requires_every_path_and_passes_explicit_values(self) -> None:
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        result = {
            "status": "PACKAGED_SPATIAL_REFERENCE_AUDIT_V1_RAW_EVIDENCE"
        }
        argv = [
            "--repository-root",
            "/repo",
            "--config",
            "/repo/code/configs/spatial_reference_audit_v1.json",
            "--source-git-commit",
            SOURCE_COMMIT,
            "--audit-root",
            "/audit",
            "--global-attempt-ledger",
            "/ledger",
            "--output",
            "/archive.tar",
        ]
        output = io.StringIO()
        with mock.patch(
            "scripts.package_spatial_reference_audit_v1.package_spatial_reference_audit_evidence",
            return_value=result,
        ) as package, redirect_stdout(output):
            self.assertEqual(main(argv), 0)
        package.assert_called_once_with(
            repository_root=Path("/repo"),
            config_path=Path("/repo/code/configs/spatial_reference_audit_v1.json"),
            source_git_commit=SOURCE_COMMIT,
            audit_root=Path("/audit"),
            global_attempt_ledger=Path("/ledger"),
            output_archive=Path("/archive.tar"),
        )
        self.assertEqual(json.loads(output.getvalue()), result)


if __name__ == "__main__":
    unittest.main()
