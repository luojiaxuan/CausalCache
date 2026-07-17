from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import causalcache.restoration_v2_2_expansion_substrate_artifact as artifact_contract
from causalcache.data.guiodyssey_restoration_v2 import canonical_json_bytes
import scripts.run_restoration_v2_2_expansion_substrate as runner


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str) -> dict:
    return json.loads((REPOSITORY_ROOT / relative).read_text(encoding="utf-8"))


def _identity(relative: str) -> dict[str, object]:
    path = REPOSITORY_ROOT / relative
    return {
        "path": relative,
        "sha256": runner.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _states() -> tuple[runner.ExpansionState, ...]:
    selection = _load(
        "data/manifests/restoration_v2_2_label_expansion_selection.json"
    )
    result: list[runner.ExpansionState] = []
    for role in ("gate_train_expansion", "gate_development_expansion"):
        for decision in selection["splits"][role]["states"]:
            result.append(
                runner.ExpansionState(
                    index=len(result),
                    role=role,
                    source_id=decision["source_id"],
                    decision=decision,
                )
            )
    return tuple(result)


def _runtime_metadata(spec: artifact_contract.WorkerSpec) -> dict[str, object]:
    return {
        "device": spec.device,
        "gpu_name": "NVIDIA H200",
        "gpu_uuid": f"GPU-{spec.index_parity}",
        "gpu_pci_bus_id": f"00000000:0{spec.index_parity}:00.0",
        "logical_device_index": spec.index_parity,
        "nvidia_smi_index": spec.index_parity,
        "container_image_digest": runner.CANONICAL_IMAGE_DIGEST,
        "python_version": "3.12.3",
        "torch_version": "2.11.0+cu130",
        "torch_cuda_version": "13.0",
        "cudnn_version": 91900,
        "transformers_version": "5.6.0",
        "nvidia_driver_version": "570.172.08",
        "dtype": "torch.bfloat16",
        "requested_attention_implementation": "eager",
        "observed_attention_implementation": {
            "top": "eager",
            "text": "eager",
            "vision": "eager",
        },
        "seed": 0,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "strict_cuda_determinism_claimed": False,
    }


class ExpansionSubstrateRunnerTest(unittest.TestCase):
    def test_main_routes_monitor_sidecar_without_changing_formal_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = str(root / "monitor.ready.json")
            log = str(root / "monitor.jsonl")
            summary = str(root / "monitor.summary.json")
            produced = {
                "schema_version": runner.SCHEMA_VERSION,
                "status": runner.MONITOR_SUMMARY_STATUS,
            }
            with (
                patch.object(runner, "run_monitor_sidecar", return_value=produced) as call,
                patch("builtins.print") as printed,
            ):
                return_code = runner.main(
                    [
                        runner.MONITOR_SIDECAR_SUBCOMMAND,
                        "--ready-file",
                        ready,
                        "--log-file",
                        log,
                        "--summary-file",
                        summary,
                    ]
                )
            self.assertEqual(return_code, 0)
            call.assert_called_once_with(
                ready_file=ready,
                log_file=log,
                summary_file=summary,
            )
            printed.assert_called_once()

        formal_flags = {
            option
            for action in runner._build_parser()._actions
            for option in action.option_strings
            if option != "--help" and option != "-h"
        }
        frozen_flags = set(
            artifact_contract._runner_interface_contract()["required_cli_options"]
        )
        self.assertEqual(formal_flags, frozen_flags)
        self.assertTrue(
            set(runner.MONITOR_SIDECAR_REQUIRED_OPTIONS).isdisjoint(formal_flags)
        )

    def test_monitor_sidecar_producer_round_trip_has_exact_schemas(self) -> None:
        run_contract_sha256 = "4" * 64
        gpus = (
            {
                "nvidia_smi_index": 0,
                "gpu_uuid": "GPU-monitor-even",
                "utilization_percent": 95,
            },
            {
                "nvidia_smi_index": 1,
                "gpu_uuid": "GPU-monitor-odd",
                "utilization_percent": 89,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "monitor.ready.json"
            log = root / "monitor.jsonl"
            summary = root / "monitor.summary.json"
            stop = runner.monitor_stop_request_path(summary)
            sleep_calls: list[float] = []

            def request_stop(delay_seconds: float) -> None:
                sleep_calls.append(delay_seconds)
                runner.publish_monitor_stop_request(
                    summary_file=summary,
                    run_contract_sha256=run_contract_sha256,
                )

            result = runner.run_monitor_sidecar(
                ready_file=ready,
                log_file=log,
                summary_file=summary,
                sampler=lambda: gpus,
                sleep_fn=request_stop,
            )

            ready_record = json.loads(ready.read_text(encoding="utf-8"))
            stop_record = json.loads(stop.read_text(encoding="utf-8"))
            summary_record = json.loads(summary.read_text(encoding="utf-8"))
            samples = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(sleep_calls, [runner.MONITOR_SAMPLE_INTERVAL_SECONDS])
            self.assertEqual(
                set(ready_record),
                {
                    "schema_version",
                    "status",
                    "container_name_prefix",
                    "minimum_gpu_utilization_percent",
                    "monitor_process_alive",
                    "monitor_pid",
                    "started_at_utc",
                },
            )
            self.assertEqual(
                set(stop_record),
                {
                    "schema_version",
                    "status",
                    "run_contract_sha256",
                    "requested_at_utc",
                },
            )
            expected_summary_keys = {
                "schema_version",
                "status",
                "run_contract_sha256",
                "container_name_prefix",
                "minimum_gpu_utilization_percent",
                "started_before_first_generation",
                "started_at_utc",
                "stopped_at_utc",
                "sample_count",
                "low_utilization_incident_count",
                "visible_gpu_count",
                "visible_gpu_uuids",
                "stop_request_sha256",
            }
            self.assertEqual(set(result), expected_summary_keys)
            self.assertEqual(set(summary_record), expected_summary_keys)
            self.assertEqual(summary_record, result)
            self.assertEqual(summary_record["run_contract_sha256"], run_contract_sha256)
            self.assertEqual(summary_record["sample_count"], 1)
            self.assertEqual(summary_record["low_utilization_incident_count"], 1)
            self.assertEqual(summary_record["stop_request_sha256"], runner.sha256_file(stop))
            self.assertEqual(len(samples), 1)
            self.assertEqual(
                set(samples[0]),
                {
                    "schema_version",
                    "status",
                    "sample_index",
                    "sampled_at_utc",
                    "visible_gpu_count",
                    "gpus",
                },
            )
            self.assertEqual(samples[0]["sample_index"], 0)
            self.assertEqual(samples[0]["gpus"], list(gpus))
            self.assertTrue(
                all(
                    set(gpu)
                    == {"nvidia_smi_index", "gpu_uuid", "utilization_percent"}
                    for gpu in samples[0]["gpus"]
                )
            )

    def test_monitor_stop_request_validates_and_binds_run_contract_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = Path(temporary) / "monitor.summary.json"
            with self.assertRaisesRegex(ValueError, "run-contract SHA256"):
                runner.publish_monitor_stop_request(
                    summary_file=summary,
                    run_contract_sha256="not-a-sha256",
                )
            self.assertFalse(runner.monitor_stop_request_path(summary).exists())

            expected = "5" * 64
            record = runner.publish_monitor_stop_request(
                summary_file=summary,
                run_contract_sha256=expected,
            )
            persisted = json.loads(
                runner.monitor_stop_request_path(summary).read_text(encoding="utf-8")
            )
            self.assertEqual(record, persisted)
            self.assertEqual(record["run_contract_sha256"], expected)

    def test_terminal_monitor_evidence_rejects_wrong_stop_run_contract_hash(self) -> None:
        expected_run_sha = "8" * 64
        wrong_run_sha = "9" * 64
        started_at = "2026-07-16T20:00:00.000000Z"
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "attempt"
            staging = parent / "publish-staging"
            root.mkdir()
            staging.mkdir()
            for relative in (
                runner.LOG_PATHS["preflight"],
                runner.LOG_PATHS["execution"],
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("evidence\n", encoding="utf-8")
            ready = {
                "schema_version": runner.SCHEMA_VERSION,
                "status": runner.MONITOR_READY_STATUS,
                "container_name_prefix": "sglang-omni-jaxan",
                "minimum_gpu_utilization_percent": 90,
                "monitor_process_alive": True,
                "monitor_pid": 12345,
                "started_at_utc": started_at,
            }
            ready_path = root / runner.LOG_PATHS["utilization_monitor_ready"]
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_bytes(runner.pretty_json_bytes(ready))

            external_log = parent / "monitor.jsonl"
            external_log.write_text("sample\n", encoding="utf-8")
            external_summary = parent / "monitor.summary.json"
            stop = runner.publish_monitor_stop_request(
                summary_file=external_summary,
                run_contract_sha256=wrong_run_sha,
            )
            summary = {
                "schema_version": runner.SCHEMA_VERSION,
                "status": runner.MONITOR_SUMMARY_STATUS,
                "run_contract_sha256": wrong_run_sha,
                "container_name_prefix": "sglang-omni-jaxan",
                "minimum_gpu_utilization_percent": 90,
                "started_before_first_generation": True,
                "started_at_utc": started_at,
                "stopped_at_utc": "2026-07-16T20:01:00.000000Z",
                "sample_count": 1,
                "low_utilization_incident_count": 0,
                "visible_gpu_count": 2,
                "visible_gpu_uuids": ["GPU-monitor-even", "GPU-monitor-odd"],
                "stop_request_sha256": runner.sha256_file(
                    runner.monitor_stop_request_path(external_summary)
                ),
            }
            external_summary.write_bytes(runner.pretty_json_bytes(summary))
            layout = SimpleNamespace(
                root=root,
                publish_staging=staging,
                run_contract_sha256=expected_run_sha,
                execution_evidence=root / runner.EXECUTION_EVIDENCE_FILENAME,
                run_contract={
                    "source": {
                        "runner_source_git_commit": "a" * 40,
                        "execution_git_commit": "b" * 40,
                    },
                    "frozen_config": {"sha256": "c" * 64},
                    "derived_completion": {"sha256": "d" * 64},
                    "derived_artifact": {"immutable_revision": "e" * 40},
                },
            )
            runtime_values = {
                spec.worker_id: _runtime_metadata(spec)
                for spec in artifact_contract.expected_worker_specs()
            }
            evidence = runner._materialize_terminal_execution_evidence(
                layout=layout,
                runtime_values=runtime_values,
                utilization_monitor_log=external_log,
                monitor_summary=external_summary,
            )
            self.assertEqual(evidence["status"], "FAILED_TERMINAL_EXECUTION_EVIDENCE")
            self.assertIsNone(evidence["monitor_summary"])
            self.assertEqual(
                evidence["terminal_failure"]["category"],
                "MONITOR_SUMMARY_FINALIZATION_FAILURE",
            )
            self.assertIn("summary drifted", evidence["terminal_failure"]["message"])
            self.assertEqual(stop["run_contract_sha256"], wrong_run_sha)
            archived_stop = root / runner.LOG_PATHS[
                "utilization_monitor_stop_request"
            ]
            self.assertEqual(
                archived_stop.read_bytes(),
                runner.monitor_stop_request_path(external_summary).read_bytes(),
            )

    def test_monitor_sidecar_ready_log_summary_and_stop_are_exclusive(self) -> None:
        gpus = (
            {
                "nvidia_smi_index": 0,
                "gpu_uuid": "GPU-monitor-even",
                "utilization_percent": 95,
            },
            {
                "nvidia_smi_index": 1,
                "gpu_uuid": "GPU-monitor-odd",
                "utilization_percent": 96,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for occupied_name in ("ready", "log", "summary", "stop"):
                case_root = root / occupied_name
                ready = case_root / "monitor.ready.json"
                log = case_root / "monitor.jsonl"
                summary = case_root / "monitor.summary.json"
                stop = runner.monitor_stop_request_path(summary)
                selected = {
                    "ready": ready,
                    "log": log,
                    "summary": summary,
                    "stop": stop,
                }[occupied_name]
                selected.parent.mkdir(parents=True, exist_ok=True)
                selected.write_bytes(b"do-not-overwrite\n")
                with self.subTest(occupied_name=occupied_name):
                    with self.assertRaisesRegex(FileExistsError, "four fresh O_EXCL"):
                        runner.run_monitor_sidecar(
                            ready_file=ready,
                            log_file=log,
                            summary_file=summary,
                            sampler=lambda: gpus,
                            sleep_fn=lambda _: None,
                        )
                    self.assertEqual(selected.read_bytes(), b"do-not-overwrite\n")

            summary = root / "stop-republish" / "monitor.summary.json"
            runner.publish_monitor_stop_request(
                summary_file=summary,
                run_contract_sha256="6" * 64,
            )
            stop = runner.monitor_stop_request_path(summary)
            original = stop.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.publish_monitor_stop_request(
                    summary_file=summary,
                    run_contract_sha256="7" * 64,
                )
            self.assertEqual(stop.read_bytes(), original)

    def test_parity_topology_and_planned_counts_are_exact(self) -> None:
        first, second = artifact_contract.expected_worker_specs()
        self.assertEqual(len(first.state_indices), 96)
        self.assertEqual(len(second.state_indices), 96)
        self.assertFalse(set(first.state_indices) & set(second.state_indices))
        self.assertEqual(
            sorted((*first.state_indices, *second.state_indices)), list(range(192))
        )
        self.assertEqual(
            runner.PLANNED_OPERATION_COUNTS,
            {
                "generation_call_count": 384,
                "teacher_forward_count": 576,
                "kl_measurement_count": 384,
            },
        )

    def test_runtime_envelopes_bind_run_and_worker_before_release(self) -> None:
        run_sha = "1" * 64
        envelopes = {}
        for spec in artifact_contract.expected_worker_specs():
            envelopes[spec.worker_id] = {
                "schema_version": runner.SCHEMA_VERSION,
                "protocol_id": runner.RUNNER_PROTOCOL_ID,
                "run_contract_sha256": run_sha,
                "worker": spec.to_dict(),
                "runtime_metadata": _runtime_metadata(spec),
                "created_at_utc": "2026-07-16T20:00:00.000000Z",
            }
        values = runner.validate_worker_runtime_envelopes(
            envelopes, run_contract_sha256=run_sha
        )
        self.assertEqual(set(values), {"even", "odd"})
        corrupted = {key: dict(value) for key, value in envelopes.items()}
        corrupted["odd"]["run_contract_sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "runtime envelope drifted"):
            runner.validate_worker_runtime_envelopes(
                corrupted, run_contract_sha256=run_sha
            )

    def test_worker_rejects_incomplete_runtime_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempt"
            root.mkdir()
            for spec in artifact_contract.expected_worker_specs():
                worker = root / artifact_contract.WORKER_DIRECTORY / spec.worker_id
                worker.mkdir(parents=True)
                runner._publish_json_exclusive_atomic(
                    worker / artifact_contract.WORKER_RUNTIME_FILENAME,
                    {"worker": spec.worker_id},
                    staging_directory=Path(temporary),
                )
            layout = SimpleNamespace(root=root, run_contract_sha256="3" * 64)
            hashes = runner._worker_evidence_hashes(
                layout, filename=artifact_contract.WORKER_RUNTIME_FILENAME
            )
            release = {
                "schema_version": runner.SCHEMA_VERSION,
                "protocol_id": runner.RUNNER_PROTOCOL_ID,
                "status": "COORDINATOR_RELEASED_BOTH_WORKER_RUNTIMES",
                "run_contract_sha256": "3" * 64,
                "worker_runtime_sha256": hashes,
                "released_at_utc": "2026-07-16T20:00:01.000000Z",
            }
            runner._validate_worker_barrier_release(
                layout,
                release_filename=artifact_contract.RUNTIME_BARRIER_RELEASE_FILENAME,
                record=release,
            )
            release["worker_runtime_sha256"] = {"even": hashes["even"]}
            with self.assertRaisesRegex(ValueError, "runtime barrier release"):
                runner._validate_worker_barrier_release(
                    layout,
                    release_filename=artifact_contract.RUNTIME_BARRIER_RELEASE_FILENAME,
                    record=release,
                )

    def test_atomic_publication_stages_outside_attempt_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "attempt"
            root.mkdir()
            staging = parent / ".publish-staging"
            staging.mkdir()
            final = root / "state.json"
            runner._write_json_exclusive(
                final, {"ok": True}, staging_directory=staging
            )
            self.assertEqual(json.loads(final.read_text()), {"ok": True})
            self.assertEqual(list(staging.iterdir()), [])
            failed = root / "failed.json"
            with patch.object(runner.os, "link", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    runner._write_json_exclusive(
                        failed, {"ok": False}, staging_directory=staging
                    )
            self.assertFalse(failed.exists())
            self.assertEqual(list(staging.iterdir()), [])

    def test_run_contract_builder_round_trips_formal_validator(self) -> None:
        base = _load(runner.CANONICAL_CONFIG_PATH)
        states = _states()
        projections = [runner._state_projection(state) for state in states]
        self.assertEqual(
            runner.sha256_bytes(canonical_json_bytes(projections)),
            base["data_projection"]["state_projection_sha256"],
        )
        inventory = [dict(value) for value in base["scientific_source_lock"]["source_files"]]
        inventory.append(_identity(runner.CANONICAL_CONFIG_PATH))
        for relative in base["scientific_source_lock"]["reserved_execution_source_paths"]:
            inventory.append(_identity(relative))
        inventory.sort(key=lambda value: value["path"])
        completion = _identity(runner.CANONICAL_COMPLETION_PATH)
        runner_commit = "a" * 40
        execution_commit = "b" * 40
        freeze = {
            "schema_version": artifact_contract.RUNNER_FREEZE_SCHEMA_VERSION,
            "protocol_id": artifact_contract.RUNNER_FREEZE_PROTOCOL_ID,
            "status": "FROZEN_COMMITTED_PUSHED_EXPANSION_SUBSTRATE_RUNNER_SOURCE",
            "runner_source_git_commit": runner_commit,
            "base_config": _identity(runner.CANONICAL_CONFIG_PATH),
            "derived_completion": completion,
            "source_inventory": inventory,
            "source_inventory_sha256": runner.sha256_bytes(
                canonical_json_bytes(inventory)
            ),
            "interfaces": artifact_contract._runner_interface_contract(),
            "authorization": {
                "runner_source_commit_must_be_committed": True,
                "runner_source_commit_must_be_ancestor_of_execution_head": True,
                "execution_head_and_origin_main_must_match": True,
                "runner_freeze_file_must_be_committed_at_execution_head": True,
                "source_blobs_must_equal_runner_source_commit": True,
                "source_only_freeze_authorizes_gpu_execution": False,
                "formal_execution_requires_runtime_authorization_validation": True,
            },
        }
        derived = {
            key: base["immutable_inputs"]["derived_artifact"][key]
            for key in (
                "repo",
                "tag",
                "immutable_revision",
                "payload_prefix",
                "artifact_tree_sha256",
                "artifact_file_count",
                "artifact_total_bytes",
            )
        }
        snapshot = _load(runner.CANONICAL_SNAPSHOT_MANIFEST_PATH)
        snapshot_witness = _identity(runner.CANONICAL_SNAPSHOT_MANIFEST_PATH)
        canonical_inputs = {
            "derived_artifact_root": "/data/derived-exact-six",
            "derived_artifact": derived,
            "completion_manifest": completion,
            "ocr_backend_config": next(
                value for value in inventory if value["path"] == runner.CANONICAL_OCR_BACKEND_CONFIG_PATH
            ),
            "ocr_backend_manifest": next(
                value for value in inventory if value["path"] == runner.CANONICAL_OCR_BACKEND_MANIFEST_PATH
            ),
            "model_snapshot_manifest": {**snapshot_witness, "data": snapshot},
            "model_snapshot": {
                "local_path": str(runner.CANONICAL_MODEL_DIR),
                "repo": snapshot["repo"],
                "revision": snapshot["revision"],
                "files": snapshot["files"],
                "file_inventory_sha256": runner.sha256_bytes(
                    canonical_json_bytes(snapshot["files"])
                ),
                "all_files_verified": True,
            },
            "fresh_immutable_derived_validation": {
                "repo": derived["repo"],
                "tag": derived["tag"],
                "immutable_revision": derived["immutable_revision"],
                "artifact_tree_sha256": derived["artifact_tree_sha256"],
                "validation_passed": True,
                "validated_before_runtime_import": True,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            freeze_path = Path(temporary) / "runner-freeze.json"
            freeze_path.write_bytes(runner.pretty_json_bytes(freeze))
            option_values = {
                "--repository-root": str(REPOSITORY_ROOT),
                "--contract": str(REPOSITORY_ROOT / runner.CANONICAL_CONFIG_PATH),
                "--runner-freeze": str(freeze_path),
                "--derived-artifact-root": canonical_inputs["derived_artifact_root"],
                "--derived-hf-repo": derived["repo"],
                "--derived-hf-tag": derived["tag"],
                "--derived-hf-revision": derived["immutable_revision"],
                "--ocr-backend-config": str(REPOSITORY_ROOT / runner.CANONICAL_OCR_BACKEND_CONFIG_PATH),
                "--snapshot-manifest": str(REPOSITORY_ROOT / runner.CANONICAL_SNAPSHOT_MANIFEST_PATH),
                "--model-dir": str(runner.CANONICAL_MODEL_DIR),
                "--host-alias": "hyper00",
                "--host-hostname": "node-radixark-16-0000",
                "--container-id": "c" * 64,
                "--container-image-digest": runner.CANONICAL_IMAGE_DIGEST,
                "--output-dir": str(artifact_contract.CANONICAL_OUTPUT_DIR),
                "--global-ledger": str(artifact_contract.CANONICAL_LEDGER_PATH),
                "--preflight-log": "/data/preflight.log",
                "--utilization-monitor-log": "/data/monitor.log",
                "--monitor-ready-file": "/data/monitor.ready.json",
                "--monitor-summary": "/data/monitor.summary.json",
            }
            argv = ["/usr/bin/python3", str(REPOSITORY_ROOT / runner.RUNNER_SOURCE_PATH)]
            for option in artifact_contract._runner_interface_contract()["required_cli_options"]:
                argv.extend((option, option_values[option]))
            args = argparse.Namespace(
                output_dir=option_values["--output-dir"],
                global_ledger=option_values["--global-ledger"],
                host_alias=option_values["--host-alias"],
                host_hostname=option_values["--host-hostname"],
                container_id=option_values["--container-id"],
                container_image_digest=option_values["--container-image-digest"],
                execution_argv=argv,
            )
            authorized = runner.AuthorizedExpansion(
                repository_root=REPOSITORY_ROOT,
                frozen_config=base,
                contract_validation={},
                git_identity={
                    "runner_source_commit": runner_commit,
                    "commit": execution_commit,
                    "branch": "main",
                    "remote_url": artifact_contract.CANONICAL_GIT_ORIGIN_URL,
                    "runner_freeze_validation": {"runner_git_commit": runner_commit},
                    "runner_freeze": freeze,
                },
                source_inventory=tuple(inventory),
                completion_manifest={},
                artifact=SimpleNamespace(states=states),
                snapshot_manifest_path=REPOSITORY_ROOT / runner.CANONICAL_SNAPSHOT_MANIFEST_PATH,
                model_identity=canonical_inputs["model_snapshot"],
                canonical_inputs=canonical_inputs,
            )
            with (
                patch.object(runner, "CANONICAL_RUNNER_FREEZE_PATH", str(freeze_path)),
                patch.object(artifact_contract, "CANONICAL_RUNNER_FREEZE_PATH", str(freeze_path)),
            ):
                contract = runner.build_execution_run_contract(args, authorized)
                validated = artifact_contract.validate_execution_run_contract(
                    contract, require_canonical_attempt_identity=True
                )
            self.assertEqual(len(validated), 192)
            self.assertEqual(contract["source"]["runner_source_git_commit"], runner_commit)
            self.assertEqual(contract["source"]["execution_git_commit"], execution_commit)


if __name__ == "__main__":
    unittest.main()
