from __future__ import annotations

import copy
import contextlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import causalcache.restoration_v2_2_expansion_labels_invalid_forensic as forensic_module
from causalcache.restoration_v2_2_expansion_labels_invalid_forensic import (
    ARCHIVE_STATUS,
    ARTIFACT_CLASS,
    ATTEMPT_ROOT_NAMESPACE,
    CANONICAL_CONFIG_PATH,
    FORENSIC_MANIFEST_PATH,
    FROZEN_CONFIG_SHA256,
    InvalidForensicEvidence,
    PROTOCOL_ID,
    PRODUCER_COMPLETED_STATUS,
    PRODUCER_EXECUTION_EVIDENCE_STATUS,
    PRODUCER_INVALID_STATUS,
    PRODUCER_PROTOCOL_ID,
    PRODUCER_WORKER_COMPLETED_STATUS,
    SOURCE_STATUS,
    TERMINAL_LEDGER_NAMESPACE,
    canonical_json_bytes,
    collect_invalid_forensic_source,
    deterministic_invalid_forensic_ustar_bytes,
    forensic_contract_from_data,
    load_frozen_forensic_contract,
    package_invalid_forensic_archive,
    pretty_json_bytes,
    read_invalid_forensic_archive,
    require_formal_label_loader_eligible,
    sha256_bytes,
)
from scripts import manage_restoration_v2_2_expansion_labels_invalid_forensic as manager


ROOT = Path(__file__).resolve().parents[2]
FAILURE_MESSAGE = (
    "monitor did not cover the complete worker execution window at the frozen cadence"
)
RUN_SHA = "a" * 64


def _inventory(files: dict[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "path": name,
            "sha256": sha256_bytes(files[name]),
            "size_bytes": len(files[name]),
        }
        for name in sorted(files)
    ]


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _worker_ledger(worker: str, state_count: int) -> dict[str, object]:
    parity = 0 if worker == "even" else 1
    indices = list(range(parity, state_count, 2))
    return {
        "schema_version": "1.0.0",
        "protocol_id": PRODUCER_PROTOCOL_ID,
        "status": PRODUCER_WORKER_COMPLETED_STATUS,
        "run_contract_sha256": RUN_SHA,
        "worker": {
            "worker_id": worker,
            "device": f"cuda:{parity}",
            "index_parity": parity,
            "state_indices": indices,
        },
        "attempted_state_indices": indices,
        "completed_state_indices": indices,
        "retry_count": 0,
        "top_up_count": 0,
        "started_at_utc": "2026-07-17T00:00:00Z",
        "ended_at_utc": "2026-07-17T00:09:00Z",
    }


def _source_tree_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _refresh_source_bindings(
    data: dict[str, object],
    *,
    root: Path,
    global_ledger: Path,
    worker_ledgers: dict[str, Path],
) -> None:
    source = data["source_attempt"]
    assert isinstance(source, dict)
    root_files = _source_tree_files(root)
    source["root_file_count"] = len(root_files)
    source["root_total_file_bytes"] = sum(len(value) for value in root_files.values())
    source["root_inventory_sha256"] = sha256_bytes(
        canonical_json_bytes(_inventory(root_files))
    )
    critical = source["critical_root_file_sha256"]
    assert isinstance(critical, dict)
    for name in critical:
        critical[name] = sha256_bytes(root_files[name])
    append = source["append_only_execution_log"]
    assert isinstance(append, dict)
    final_log = root_files[append["final_log_path"]]
    prefix_size = append["pre_failure_prefix_size_bytes"]
    assert isinstance(prefix_size, int)
    append["execution_evidence_sha256"] = critical[append["execution_evidence_path"]]
    append["final_log_sha256"] = critical[append["final_log_path"]]
    append["final_log_size_bytes"] = len(final_log)
    append["pre_failure_prefix_sha256"] = sha256_bytes(final_log[:prefix_size])
    append["terminal_suffix_sha256"] = sha256_bytes(final_log[prefix_size:])
    append["terminal_suffix_size_bytes"] = len(final_log[prefix_size:])
    external = source["external_global_ledger"]
    assert isinstance(external, dict)
    external_payload = global_ledger.read_bytes()
    external["size_bytes"] = len(external_payload)
    external["sha256"] = sha256_bytes(external_payload)
    workers = source["external_worker_ledgers"]
    assert isinstance(workers, dict)
    for worker, path in worker_ledgers.items():
        payload = path.read_bytes()
        workers[worker]["size_bytes"] = len(payload)
        workers[worker]["sha256"] = sha256_bytes(payload)


def _fixture(directory: Path):
    state_count = 4
    root = directory / "producer-root"
    global_ledger = directory / ".global.attempt.json"
    worker_ledgers = {
        "even": directory / ".even.attempt.json",
        "odd": directory / ".odd.attempt.json",
    }
    output = directory / "invalid-forensic.tar"
    root.mkdir()
    prefix = b"2026-07-17T00:00:00Z global attempt claimed\n"
    suffix = (
        "2026-07-17T00:12:00Z attempt invalidated: "
        + FAILURE_MESSAGE
        + "\n"
    ).encode("utf-8")
    execution_log = prefix + suffix
    execution_evidence = {
        "schema_version": "1.0.0",
        "protocol_id": PRODUCER_PROTOCOL_ID,
        "status": PRODUCER_EXECUTION_EVIDENCE_STATUS,
        "run_contract_sha256": RUN_SHA,
        "preflight_log_sha256": "1" * 64,
        "execution_log_sha256": sha256_bytes(prefix),
        "utilization_monitor_log_sha256": "2" * 64,
        "monitor_ready_sha256": "3" * 64,
        "monitor_stop_request_sha256": "4" * 64,
        "monitor_summary_sha256": "5" * 64,
    }
    sibling_paths = {
        worker: str(path.resolve()) for worker, path in worker_ledgers.items()
    }
    full_indices = list(range(state_count))
    internal_global = {
        "schema_version": "1.0.0",
        "protocol_id": PRODUCER_PROTOCOL_ID,
        "status": PRODUCER_COMPLETED_STATUS,
        "attempt_id": "synthetic-invalid-attempt",
        "run_contract_sha256": RUN_SHA,
        "output_dir": str(root.resolve()),
        "worker_sibling_ledgers": sibling_paths,
        "attempted_state_indices": full_indices,
        "completed_state_indices": full_indices,
        "retry_count": 0,
        "top_up_count": 0,
        "started_at_utc": "2026-07-17T00:00:00Z",
        "ended_at_utc": "2026-07-17T00:10:00Z",
    }
    internal_payload = pretty_json_bytes(internal_global)
    external_global = {
        **internal_global,
        "status": PRODUCER_INVALID_STATUS,
        "ended_at_utc": "2026-07-17T00:11:00Z",
        "failure": {
            "exception_type": "ValueError",
            "message": FAILURE_MESSAGE,
        },
        "claimed_ledger_sha256": sha256_bytes(internal_payload),
    }
    _write(root / "aggregate.json", pretty_json_bytes({"outcome": "inner-pass"}))
    _write(root / "execution_evidence.json", pretty_json_bytes(execution_evidence))
    _write(root / "global_attempt_ledger.json", internal_payload)
    _write(root / "logs/execution.log", execution_log)
    _write(root / "logs/gpu_utilization_monitor.log", b'{"sample_index":0}\n')
    _write(root / "monitor_summary.json", pretty_json_bytes({"samples": 1}))
    _write(root / "run_manifest.json", pretty_json_bytes({"run": "synthetic"}))
    for worker in ("even", "odd"):
        payload = pretty_json_bytes(_worker_ledger(worker, state_count))
        _write(root / f"workers/{worker}/worker_attempt_ledger.json", payload)
        _write(root / f"worker_sibling_ledgers/{worker}.json", payload)
        _write(worker_ledgers[worker], payload)
    _write(global_ledger, pretty_json_bytes(external_global))

    root_files = _source_tree_files(root)
    critical_names = {
        "aggregate.json",
        "execution_evidence.json",
        "global_attempt_ledger.json",
        "logs/execution.log",
        "logs/gpu_utilization_monitor.log",
        "monitor_summary.json",
        "run_manifest.json",
    }
    data: dict[str, object] = {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "status": SOURCE_STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "source_attempt": {
            "producer_protocol_id": PRODUCER_PROTOCOL_ID,
            "attempt_id": "synthetic-invalid-attempt",
            "run_contract_sha256": RUN_SHA,
            "expected_state_count": state_count,
            "output_root": str(root.resolve()),
            "root_file_count": len(root_files),
            "root_total_file_bytes": sum(len(value) for value in root_files.values()),
            "root_inventory_sha256": sha256_bytes(
                canonical_json_bytes(_inventory(root_files))
            ),
            "critical_root_file_sha256": {
                name: sha256_bytes(root_files[name]) for name in critical_names
            },
            "external_global_ledger": {
                "path": str(global_ledger.resolve()),
                "sha256": sha256_bytes(global_ledger.read_bytes()),
                "size_bytes": len(global_ledger.read_bytes()),
            },
            "external_worker_ledgers": {
                worker: {
                    "path": str(path.resolve()),
                    "sha256": sha256_bytes(path.read_bytes()),
                    "size_bytes": len(path.read_bytes()),
                }
                for worker, path in worker_ledgers.items()
            },
            "append_only_execution_log": {
                "execution_evidence_path": "execution_evidence.json",
                "execution_evidence_sha256": sha256_bytes(
                    pretty_json_bytes(execution_evidence)
                ),
                "final_log_path": "logs/execution.log",
                "final_log_sha256": sha256_bytes(execution_log),
                "final_log_size_bytes": len(execution_log),
                "pre_failure_prefix_sha256": sha256_bytes(prefix),
                "pre_failure_prefix_size_bytes": len(prefix),
                "terminal_suffix_sha256": sha256_bytes(suffix),
                "terminal_suffix_size_bytes": len(suffix),
            },
        },
        "failure_contract": {
            "exception_type": "ValueError",
            "message": FAILURE_MESSAGE,
            "original_attempt_status": PRODUCER_INVALID_STATUS,
            "root_embedded_snapshot_status": PRODUCER_COMPLETED_STATUS,
        },
        "archive_contract": {
            "archive_format": "ustar",
            "archive_member_prefix": "synthetic-invalid-forensic-v1",
            "canonical_archive_path": str(output.resolve()),
            "attempt_root_namespace": ATTEMPT_ROOT_NAMESPACE,
            "terminal_external_ledger_namespace": TERMINAL_LEDGER_NAMESPACE,
            "forensic_manifest_path": FORENSIC_MANIFEST_PATH,
            "expected_payload_file_count_before_manifest": len(root_files) + 3,
            "expected_member_count": len(root_files) + 4,
            "deterministic_regular_file_metadata": {
                "mode": 0o644,
                "uid": 0,
                "gid": 0,
                "uname": "",
                "gname": "",
                "mtime": 0,
                "pax_headers": {},
            },
            "overwrite_allowed": False,
        },
        "formal_consumption": {
            "formal_label_loader_eligible": False,
            "gate_training_unlocked": False,
            "matched_nll_unlocked": False,
            "closed_loop_unlocked": False,
            "reason": "synthetic invalid-forensic fixture is never formal labels",
        },
        "network_contract": {
            "hf_publish_authorized": False,
            "network_access_required": False,
            "upload_command_present": False,
        },
    }
    return (
        forensic_contract_from_data(data),
        data,
        root,
        global_ledger,
        worker_ledgers,
        output,
    )


class ExpansionLabelsInvalidForensicTest(unittest.TestCase):
    def test_cli_exposes_source_validation_without_upload_command(self) -> None:
        choices = manager._parser()._subparsers._group_actions[0].choices
        self.assertEqual(set(choices), {"validate-contract", "package", "validate"})
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
            result["status"], "VALID_SOURCE_ONLY_INVALID_FORENSIC_CONTRACT"
        )
        self.assertFalse(result["formal_label_loader_eligible"])
        self.assertFalse(result["hf_publish_authorized"])

    def test_frozen_source_config_is_exact_and_formal_ineligible(self) -> None:
        contract = load_frozen_forensic_contract(
            CANONICAL_CONFIG_PATH,
            repository_root=ROOT,
        )
        self.assertEqual(contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(
            contract.archive_contract["expected_member_count"],
            406,
        )
        self.assertFalse(
            contract.data["formal_consumption"]["formal_label_loader_eligible"]
        )
        self.assertFalse(contract.data["network_contract"]["hf_publish_authorized"])

    def test_double_ledger_namespaces_and_append_only_chain_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract, _data, root, global_ledger, workers, _output = _fixture(
                Path(directory)
            )
            evidence = collect_invalid_forensic_source(
                contract=contract,
                raw_output_dir=root,
                external_global_ledger=global_ledger,
                external_worker_ledgers=workers,
            )
        internal = evidence.files[
            f"{ATTEMPT_ROOT_NAMESPACE}/global_attempt_ledger.json"
        ]
        external = evidence.files[
            f"{TERMINAL_LEDGER_NAMESPACE}/global_attempt_ledger.json"
        ]
        self.assertNotEqual(internal, external)
        self.assertEqual(json.loads(internal)["status"], PRODUCER_COMPLETED_STATUS)
        self.assertEqual(json.loads(external)["status"], PRODUCER_INVALID_STATUS)
        self.assertEqual(
            json.loads(external)["claimed_ledger_sha256"], sha256_bytes(internal)
        )
        manifest = evidence.manifest
        self.assertEqual(manifest["status"], ARCHIVE_STATUS)
        self.assertFalse(
            manifest["formal_consumption"]["formal_label_loader_eligible"]
        )
        self.assertTrue(
            manifest["append_only_execution_log"][
                "execution_evidence_binds_pre_failure_prefix"
            ]
        )

    def test_deterministic_strict_ustar_round_trip_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract, _data, root, global_ledger, workers, output = _fixture(
                Path(directory)
            )
            source = collect_invalid_forensic_source(
                contract=contract,
                raw_output_dir=root,
                external_global_ledger=global_ledger,
                external_worker_ledgers=workers,
            )
            first = deterministic_invalid_forensic_ustar_bytes(
                source.files,
                member_prefix=contract.archive_contract["archive_member_prefix"],
            )
            second = deterministic_invalid_forensic_ustar_bytes(
                source.files,
                member_prefix=contract.archive_contract["archive_member_prefix"],
            )
            self.assertEqual(first, second)
            result = package_invalid_forensic_archive(
                contract=contract,
                raw_output_dir=root,
                external_global_ledger=global_ledger,
                external_worker_ledgers=workers,
                output_archive=output,
            )
            reread = read_invalid_forensic_archive(output, contract=contract)
            self.assertEqual(reread.files, source.files)
            self.assertEqual(result["archive_sha256"], sha256_bytes(first))
            self.assertFalse(result["formal_label_loader_eligible"])
            with self.assertRaises(FileExistsError):
                package_invalid_forensic_archive(
                    contract=contract,
                    raw_output_dir=root,
                    external_global_ledger=global_ledger,
                    external_worker_ledgers=workers,
                    output_archive=output,
                )

    def test_noncanonical_ustar_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract, _data, root, global_ledger, workers, _output = _fixture(
                Path(directory)
            )
            evidence = collect_invalid_forensic_source(
                contract=contract,
                raw_output_dir=root,
                external_global_ledger=global_ledger,
                external_worker_ledgers=workers,
            )
            bad_path = Path(directory) / "bad.tar"
            stream = io.BytesIO()
            with tarfile.open(
                fileobj=stream,
                mode="w",
                format=tarfile.USTAR_FORMAT,
            ) as tar:
                for index, name in enumerate(sorted(evidence.files)):
                    payload = evidence.files[name]
                    info = tarfile.TarInfo(
                        f"{contract.archive_contract['archive_member_prefix']}/{name}"
                    )
                    info.size = len(payload)
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 1 if index == 0 else 0
                    tar.addfile(info, io.BytesIO(payload))
            bad_path.write_bytes(stream.getvalue())
            with self.assertRaisesRegex(ValueError, "USTAR metadata"):
                read_invalid_forensic_archive(bad_path, contract=contract)

    def test_source_is_recollected_after_archive_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract, _data, root, global_ledger, workers, output = _fixture(
                Path(directory)
            )
            source = collect_invalid_forensic_source(
                contract=contract,
                raw_output_dir=root,
                external_global_ledger=global_ledger,
                external_worker_ledgers=workers,
            )
            changed_files = dict(source.files)
            changed_files[f"{ATTEMPT_ROOT_NAMESPACE}/aggregate.json"] += b"drift"
            changed = InvalidForensicEvidence(
                files=changed_files,
                manifest=source.manifest,
                inventory=source.inventory,
                tree_inventory_sha256=source.tree_inventory_sha256,
            )
            with mock.patch.object(
                forensic_module,
                "collect_invalid_forensic_source",
                side_effect=[source, source, changed],
            ) as collect, self.assertRaisesRegex(
                ValueError, "changed after archive publication"
            ):
                package_invalid_forensic_archive(
                    contract=contract,
                    raw_output_dir=root,
                    external_global_ledger=global_ledger,
                    external_worker_ledgers=workers,
                    output_archive=output,
                )
            self.assertEqual(collect.call_count, 3)
            self.assertTrue(output.is_file())

    def test_claimed_completed_ledger_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract, data, root, global_ledger, workers, _output = _fixture(
                Path(directory)
            )
            external = json.loads(global_ledger.read_bytes())
            external["claimed_ledger_sha256"] = "f" * 64
            global_ledger.write_bytes(pretty_json_bytes(external))
            changed = copy.deepcopy(data)
            _refresh_source_bindings(
                changed,
                root=root,
                global_ledger=global_ledger,
                worker_ledgers=workers,
            )
            changed_contract = forensic_contract_from_data(changed)
            with self.assertRaisesRegex(ValueError, "ledger chain"):
                collect_invalid_forensic_source(
                    contract=changed_contract,
                    raw_output_dir=root,
                    external_global_ledger=global_ledger,
                    external_worker_ledgers=workers,
                )
            self.assertNotEqual(contract.sha256, changed_contract.sha256)

    def test_execution_evidence_must_bind_pre_failure_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _contract, data, root, global_ledger, workers, _output = _fixture(
                Path(directory)
            )
            evidence_path = root / "execution_evidence.json"
            execution = json.loads(evidence_path.read_bytes())
            execution["execution_log_sha256"] = "e" * 64
            evidence_path.write_bytes(pretty_json_bytes(execution))
            changed = copy.deepcopy(data)
            _refresh_source_bindings(
                changed,
                root=root,
                global_ledger=global_ledger,
                worker_ledgers=workers,
            )
            changed_contract = forensic_contract_from_data(changed)
            with self.assertRaisesRegex(ValueError, "pre-failure prefix"):
                collect_invalid_forensic_source(
                    contract=changed_contract,
                    raw_output_dir=root,
                    external_global_ledger=global_ledger,
                    external_worker_ledgers=workers,
                )

    def test_external_symlink_is_rejected_by_nofollow_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract, _data, root, global_ledger, workers, _output = _fixture(
                Path(directory)
            )
            real_ledger = global_ledger.with_name("external-global-real.json")
            global_ledger.rename(real_ledger)
            global_ledger.symlink_to(real_ledger)
            with self.assertRaisesRegex(ValueError, "external global ledger"):
                collect_invalid_forensic_source(
                    contract=contract,
                    raw_output_dir=root,
                    external_global_ledger=global_ledger,
                    external_worker_ledgers=workers,
                )

    def test_manifest_is_rejected_before_formal_label_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract, _data, root, global_ledger, workers, _output = _fixture(
                Path(directory)
            )
            evidence = collect_invalid_forensic_source(
                contract=contract,
                raw_output_dir=root,
                external_global_ledger=global_ledger,
                external_worker_ledgers=workers,
            )
        with self.assertRaisesRegex(ValueError, "ineligible for formal labels"):
            require_formal_label_loader_eligible(evidence.manifest)


if __name__ == "__main__":
    unittest.main()
