from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from causalcache.data.restoration_v2_2_expansion_label_parent import (
    extract_expansion_label_parent_states,
)
from causalcache.restoration_v2_2_expansion_labels_artifact import (
    EXPECTED_KL_MEASUREMENTS,
    EXPECTED_STATE_COUNT,
    EXPECTED_TEACHER_FORWARDS,
    EXPECTED_TOTALS,
    PROHIBITED_OPERATION_COUNTS,
    expected_worker_specs,
    reduce_raw_distance_states,
)
import scripts.run_restoration_v2_2_expansion_labels as runner
from tests.test_restoration_v2_2_expansion_label_inputs import (
    _artifact,
    _decode,
    _evidence,
)


RUN_SHA = "a" * 64


class _Device:
    def __init__(self, device_type: str = "cuda") -> None:
        self.type = device_type

    def __str__(self) -> str:
        return f"{self.type}:0"


class _Tensor:
    def __init__(self, dtype: str, shape: tuple[int, int, int] = (1, 3, 101)) -> None:
        self.ndim = 3
        self.shape = shape
        self.dtype = dtype
        self.device = _Device()
        self.requires_grad = False


class _Runtime:
    def __init__(self, *, bad_transfer_call: int | None = None) -> None:
        self.teacher_forward_count = 0
        self.bad_transfer_call = bad_transfer_call

    def teacher_forced_distance_logits(self, messages, actions):
        self.teacher_forward_count += 1
        self.assert_single(messages, actions)
        transfer = int(self.teacher_forward_count == self.bad_transfer_call)
        return _Tensor("torch.bfloat16"), {
            "vocabulary_size": 101,
            "full_logit_tensor_host_transfers": transfer,
            "samples": [{"distance_action_tokens": 3}],
        }

    @staticmethod
    def assert_single(messages, actions) -> None:
        if len(messages) != 1 or len(actions) != 1:
            raise AssertionError("runtime expected one exact state example")


class _Backend:
    def __init__(
        self,
        *,
        values: tuple[float, ...] = (),
        transfer_count: int = 0,
        validation_scalar_reads: int = 0,
    ) -> None:
        self.torch = object()
        self.values = list(values)
        self.transfer_count = transfer_count
        self.validation_scalar_reads = validation_scalar_reads
        self.measurement_count = 0

    def prepare_reference(self, logits):
        if logits.dtype != "torch.bfloat16":
            raise AssertionError("reference logits did not remain BF16")
        return _Tensor("torch.float32", shape=logits.shape)

    def measure(self, reference, candidate):
        if reference.dtype != "torch.float32" or candidate.dtype != "torch.bfloat16":
            raise AssertionError("KL input dtype drifted")
        self.measurement_count += 1
        value = self.values.pop(0) if self.values else 0.0
        return SimpleNamespace(
            value=value,
            audit={
                "full_tensor_host_transfers": self.transfer_count,
                "validation_scalar_host_reads": self.validation_scalar_reads,
            },
        )


def _runtime_metadata(worker_index: int) -> dict[str, object]:
    visible = ["GPU-00000000", "GPU-11111111"]
    bindings = {
        "protocol_id": "causalcache_restoration_v2_1_official_tool_interface",
        "runtime_profile_id": "causalcache_restoration_v2_2_eager_runtime",
        "device": f"cuda:{worker_index}",
        "gpu_name": "NVIDIA H200",
        "gpu_uuid": visible[worker_index],
        "gpu_pci_bus_id": f"00000000:0{worker_index}:00.0",
        "logical_device_index": worker_index,
        "nvidia_smi_index": worker_index,
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
        "deterministic_algorithms_requested": False,
        "deterministic_algorithms_enabled": False,
        "deterministic_warn_only_enabled": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "strict_cuda_determinism_claimed": False,
    }
    return {
        "container_id": "b" * 64,
        "host_alias": "hyper00",
        "hostname": "node-radixark-16-0000",
        "visible_gpu_count": 2,
        "visible_gpu_uuids": visible,
        "worker_gpu_uuid": visible[worker_index],
        "worker_cuda_device": f"cuda:{worker_index}",
        "bindings_metadata": bindings,
        "bindings_metadata_sha256": runner.sha256_bytes(
            runner.canonical_json_bytes(bindings)
        ),
    }


class ExpansionExactLabelRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = _artifact()
        cls.parents = extract_expansion_label_parent_states(
            _evidence(cls.artifact),
            cls.artifact,
        )
        cls.state_projections = tuple(
            runner._state_projection(parent, artifact=cls.artifact, image_decoder=_decode)
            for parent in cls.parents
        )

    def test_full_192_state_kernel_has_exact_totals_and_only_raw_distance_rows(self) -> None:
        runtime = _Runtime()
        backend = _Backend()
        workers = expected_worker_specs()
        records = []
        for parent in self.parents:
            worker = workers[parent.index % 2]
            record = runner.run_expansion_label_state_once(
                artifact=self.artifact,
                parent=parent,
                worker=worker,
                runtime=runtime,
                distance_backend=backend,
                run_contract_sha256=RUN_SHA,
                expected_state=self.state_projections[parent.index],
                image_decoder=_decode,
            )
            records.append(record)
            self.assertEqual(record["worker"], worker.envelope_identity())
            self.assertEqual(
                set(record),
                {
                    "schema_version",
                    "protocol_id",
                    "status",
                    "run_contract_sha256",
                    "worker",
                    "state",
                    "reference_teacher",
                    "distance_rows",
                    "operation_counts",
                },
            )
            self.assertNotIn("deployment_conditional_edges", record)
            self.assertNotIn("pair_interactions", record)
            self.assertNotIn("primary_exact_subset_oracle", record)
            self.assertEqual(
                record["operation_counts"]["scalar_host_transfer_count"],
                1 << len(parent.candidate_event_step_ids),
            )

        reduction = reduce_raw_distance_states(
            records,
            expected_states=self.state_projections,
            run_contract_sha256=RUN_SHA,
        )
        self.assertEqual(len(records), EXPECTED_STATE_COUNT)
        self.assertEqual(runtime.teacher_forward_count, EXPECTED_TEACHER_FORWARDS)
        self.assertEqual(backend.measurement_count, EXPECTED_KL_MEASUREMENTS)
        self.assertEqual(
            sum(len(record["distance_rows"]) for record in records),
            1792,
        )
        self.assertEqual(reduction["counts"], EXPECTED_TOTALS)
        self.assertEqual(
            [len(spec.state_indices) for spec in workers],
            [96, 96],
        )
        self.assertEqual(
            [record["state"]["state_index"] for record in records[::2]],
            list(workers[0].state_indices),
        )
        self.assertEqual(
            [record["state"]["state_index"] for record in records[1::2]],
            list(workers[1].state_indices),
        )

    def test_state_schema_has_exact_scalar_accounting_and_no_fingerprint(self) -> None:
        runtime = _Runtime()
        backend = _Backend()
        record = runner.run_expansion_label_state_once(
            artifact=self.artifact,
            parent=self.parents[0],
            worker=expected_worker_specs()[0],
            runtime=runtime,
            distance_backend=backend,
            run_contract_sha256=RUN_SHA,
            expected_state=self.state_projections[0],
            image_decoder=_decode,
        )
        reference = record["reference_teacher"]
        self.assertEqual(
            set(reference),
            {
                "canonical_action_sha256",
                "teacher_target_sha256",
                "reference_input_sha256",
                "reference_repeat_kl",
                "repeat_scalar_host_transfer_count",
                "action_token_count",
                "vocabulary_size",
                "action_log_probs_shape",
                "action_log_probs_dtype",
                "action_log_probs_device_type",
                "full_logit_tensor_host_transfer_count",
            },
        )
        self.assertEqual(reference["action_log_probs_shape"], [3, 101])
        self.assertEqual(reference["action_log_probs_dtype"], "torch.float32")
        self.assertEqual(reference["action_log_probs_device_type"], "cuda")
        self.assertEqual(reference["repeat_scalar_host_transfer_count"], 1)
        self.assertNotIn("fingerprint", json.dumps(record))
        row_scalar_transfers = 0
        for row in record["distance_rows"]:
            self.assertEqual(
                set(row),
                {
                    "coalition",
                    "distance",
                    "candidate_input_sha256",
                    "teacher_forward_count",
                    "kl_measurement_count",
                    "scalar_host_transfer_count",
                    "is_full_history_reference",
                    "full_logit_tensor_host_transfer_count",
                },
            )
            row_scalar_transfers += row["scalar_host_transfer_count"]
        self.assertEqual(
            row_scalar_transfers + reference["repeat_scalar_host_transfer_count"],
            record["operation_counts"]["scalar_host_transfer_count"],
        )
        self.assertEqual(
            [row["coalition"] for row in record["distance_rows"]],
            [
                [],
                [1],
                [2],
                [1, 2],
            ],
        )
        self.assertEqual(
            [row["candidate_input_sha256"] for row in record["distance_rows"]],
            [
                witness["input_sha256"]
                for witness in self.state_projections[0]["coalition_inputs"]
            ],
        )

    def test_repeat_and_all_teacher_transfer_failures_are_fail_closed(self) -> None:
        parent = self.parents[0]
        worker = expected_worker_specs()[0]
        with self.assertRaisesRegex(RuntimeError, "1e-4 ceiling"):
            runner.run_expansion_label_state_once(
                artifact=self.artifact,
                parent=parent,
                worker=worker,
                runtime=_Runtime(),
                distance_backend=_Backend(values=(1e-3,)),
                run_contract_sha256=RUN_SHA,
                expected_state=self.state_projections[0],
                image_decoder=_decode,
            )
        for bad_call in (1, 2, 3):
            with self.subTest(bad_call=bad_call), self.assertRaisesRegex(
                RuntimeError, "metadata"
            ):
                runner.run_expansion_label_state_once(
                    artifact=self.artifact,
                    parent=parent,
                    worker=worker,
                    runtime=_Runtime(bad_transfer_call=bad_call),
                    distance_backend=_Backend(),
                    run_contract_sha256=RUN_SHA,
                    expected_state=self.state_projections[0],
                    image_decoder=_decode,
                )
        with self.assertRaisesRegex(RuntimeError, "full-tensor host transfers"):
            runner.run_expansion_label_state_once(
                artifact=self.artifact,
                parent=parent,
                worker=worker,
                runtime=_Runtime(),
                distance_backend=_Backend(transfer_count=1),
                run_contract_sha256=RUN_SHA,
                expected_state=self.state_projections[0],
                image_decoder=_decode,
            )
        with self.assertRaisesRegex(RuntimeError, "validation-scalar"):
            runner.run_expansion_label_state_once(
                artifact=self.artifact,
                parent=parent,
                worker=worker,
                runtime=_Runtime(),
                distance_backend=_Backend(validation_scalar_reads=1),
                run_contract_sha256=RUN_SHA,
                expected_state=self.state_projections[0],
                image_decoder=_decode,
            )

    def test_preclaim_prompt_witness_drift_fails_before_teacher_forward(self) -> None:
        drifted = json.loads(json.dumps(self.state_projections[0]))
        drifted["coalition_inputs"][0]["input_sha256"] = "f" * 64
        runtime = _Runtime()
        with self.assertRaisesRegex(ValueError, "preclaim witness"):
            runner.run_expansion_label_state_once(
                artifact=self.artifact,
                parent=self.parents[0],
                worker=expected_worker_specs()[0],
                runtime=runtime,
                distance_backend=_Backend(),
                run_contract_sha256=RUN_SHA,
                expected_state=drifted,
                image_decoder=_decode,
            )
        self.assertEqual(runtime.teacher_forward_count, 0)

    def test_wrong_worker_fails_before_any_teacher_forward(self) -> None:
        runtime = _Runtime()
        with self.assertRaisesRegex(ValueError, "wrong fixed-parity"):
            runner.run_expansion_label_state_once(
                artifact=self.artifact,
                parent=self.parents[0],
                worker=expected_worker_specs()[1],
                runtime=runtime,
                distance_backend=_Backend(),
                run_contract_sha256=RUN_SHA,
                image_decoder=_decode,
            )
        self.assertEqual(runtime.teacher_forward_count, 0)

    def test_runtime_barrier_binds_even_odd_to_same_two_gpu_container(self) -> None:
        envelopes = {
            spec.worker_id: {
                "schema_version": runner.SCHEMA_VERSION,
                "protocol_id": runner.PROTOCOL_ID,
                "status": "VALIDATED_EXPANSION_EXACT_LABEL_WORKER_RUNTIME",
                "run_contract_sha256": RUN_SHA,
                "worker": spec.envelope_identity(),
                "runtime_metadata": _runtime_metadata(spec.index_parity),
            }
            for spec in expected_worker_specs()
        }
        self.assertEqual(
            runner.validate_runtime_pair_for_release(
                envelopes,
                run_contract_sha256=RUN_SHA,
            ),
            ("GPU-00000000", "GPU-11111111"),
        )
        drifted = json.loads(json.dumps(envelopes))
        drifted["odd"]["runtime_metadata"]["worker_gpu_uuid"] = "GPU-00000000"
        with self.assertRaisesRegex(ValueError, "GPU assignment"):
            runner.validate_runtime_pair_for_release(
                drifted,
                run_contract_sha256=RUN_SHA,
            )
        binding_drift = json.loads(json.dumps(envelopes))
        binding_drift["odd"]["runtime_metadata"]["bindings_metadata"]["dtype"] = (
            "torch.float16"
        )
        bindings = binding_drift["odd"]["runtime_metadata"]["bindings_metadata"]
        binding_drift["odd"]["runtime_metadata"]["bindings_metadata_sha256"] = (
            runner.sha256_bytes(runner.canonical_json_bytes(bindings))
        )
        with self.assertRaisesRegex(ValueError, "runtime metadata"):
            runner.validate_runtime_pair_for_release(
                binding_drift,
                run_contract_sha256=RUN_SHA,
            )

    def test_worker_failure_writes_one_shot_invalid_tombstone_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempt"
            staging = Path(temporary) / "staging"
            root.mkdir()
            staging.mkdir()
            layout = runner.ExpansionLabelLayout(
                root=root,
                global_ledger=Path(temporary) / "global.json",
                publish_staging=staging,
                run_contract={},
                run_contract_sha256=RUN_SHA,
                worker_sibling_ledgers={
                    spec.worker_id: Path(temporary) / f"{spec.worker_id}.json"
                    for spec in expected_worker_specs()
                },
                started_at_utc="2026-07-17T00:00:00.000000Z",
            )
            authorized = SimpleNamespace(
                artifact=self.artifact,
                parents=self.parents,
                state_projections=self.state_projections,
            )
            args = argparse.Namespace()

            def runtime_loader(spec, _authorized, _args):
                return runner.WorkerRuntime(
                    runtime=_Runtime(),
                    distance_backend=_Backend(),
                    runtime_metadata=_runtime_metadata(spec.index_parity),
                )

            calls = []

            def fail_state(**kwargs):
                calls.append(kwargs["parent"].index)
                raise RuntimeError("synthetic first-state failure")

            terminal = runner.run_worker(
                layout=layout,
                spec=expected_worker_specs()[0],
                authorized=authorized,
                args=args,
                runtime_loader=runtime_loader,
                state_runner=fail_state,
            )
            self.assertEqual(calls, [0])
            self.assertEqual(terminal["attempted_state_indices"], [0])
            self.assertEqual(terminal["completed_state_indices"], [])
            self.assertEqual(terminal["outcome"], runner.WORKER_INVALID_STATUS)
            self.assertEqual(terminal["operation_counts"]["retry_count"], 0)
            self.assertEqual(terminal["operation_counts"]["top_up_count"], 0)
            marker = root / "workers/even/attempts/000.json"
            self.assertTrue(marker.is_file())
            self.assertFalse((root / "workers/even/attempts/002.json").exists())

    def test_sibling_abort_stops_before_the_next_teacher_forward(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempt"
            staging = Path(temporary) / "staging"
            root.mkdir()
            staging.mkdir()
            layout = runner.ExpansionLabelLayout(
                root=root,
                global_ledger=Path(temporary) / "global.json",
                publish_staging=staging,
                run_contract={},
                run_contract_sha256=RUN_SHA,
                worker_sibling_ledgers={
                    spec.worker_id: Path(temporary) / f"{spec.worker_id}.json"
                    for spec in expected_worker_specs()
                },
                started_at_utc="2026-07-17T00:00:00.000000Z",
            )
            authorized = SimpleNamespace(
                artifact=self.artifact,
                parents=self.parents,
                state_projections=self.state_projections,
            )
            abort = threading.Event()
            calls: list[int] = []

            def runtime_loader(spec, _authorized, _args):
                return runner.WorkerRuntime(
                    runtime=_Runtime(),
                    distance_backend=_Backend(),
                    runtime_metadata=_runtime_metadata(spec.index_parity),
                )

            def abort_after_first(**kwargs):
                calls.append(kwargs["parent"].index)
                abort.set()
                return {
                    "operation_counts": runner.state_operation_counts(
                        len(kwargs["parent"].candidate_event_step_ids)
                    )
                }

            terminal = runner.run_worker(
                layout=layout,
                spec=expected_worker_specs()[0],
                authorized=authorized,
                args=argparse.Namespace(),
                abort_event=abort,
                runtime_loader=runtime_loader,
                state_runner=abort_after_first,
            )
            self.assertEqual(calls, [0])
            self.assertEqual(terminal["completed_state_indices"], [0])
            self.assertEqual(terminal["outcome"], runner.WORKER_INVALID_STATUS)

    def test_worker_rebuilt_authorization_drift_fails_closed(self) -> None:
        layout = runner.ExpansionLabelLayout(
            root=Path("/tmp/not-used"),
            global_ledger=Path("/tmp/not-used-ledger"),
            publish_staging=Path("/tmp/not-used-staging"),
            run_contract={},
            run_contract_sha256=RUN_SHA,
            worker_sibling_ledgers={},
            started_at_utc="2026-07-17T00:00:00.000000Z",
        )
        authorized = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "differs from global claim"):
            runner.rebuild_worker_authorization(
                layout,
                argparse.Namespace(),
                authorization_loader=lambda _args: authorized,
                contract_builder=lambda _args, _authorized: {"drifted": True},
            )

    def test_parent_supervisor_propagates_abnormal_worker_exit(self) -> None:
        joined: list[str] = []

        class DoneProcess:
            def __init__(self, name: str, exitcode: int) -> None:
                self.name = name
                self.exitcode = exitcode

            def join(self) -> None:
                joined.append(self.name)

        abort = threading.Event()
        runner._supervise_released_workers(
            {
                "even": DoneProcess("even", 1),
                "odd": DoneProcess("odd", 0),
            },
            abort,
        )
        self.assertTrue(abort.is_set())
        self.assertEqual(joined, ["even", "odd"])

    def test_claim_preflight_copy_failure_seals_durable_global_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "attempt"
            ledger = base / "global.json"
            preflight = base / "preflight.log"
            preflight.write_text("ready\n", encoding="utf-8")
            args = argparse.Namespace(
                output_dir=str(root),
                global_ledger=str(ledger),
                preflight_log=str(preflight),
            )
            with (
                patch.object(
                    runner,
                    "_copy_exclusive",
                    side_effect=RuntimeError("synthetic preflight copy failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "preflight copy failure"),
            ):
                runner.claim_attempt(args, {"contract": "fixed"}, require_canonical=False)
            tombstone = json.loads(ledger.read_text(encoding="utf-8"))
            archived = json.loads(
                (root / runner.GLOBAL_LEDGER_ARCHIVE_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(tombstone, archived)
            self.assertEqual(tombstone["status"], runner.GLOBAL_INVALID_STATUS)
            self.assertEqual(tombstone["retry_count"], 0)
            self.assertEqual(tombstone["top_up_count"], 0)
            self.assertEqual(tombstone["failure"]["stage"], "claim_attempt")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                runner.claim_attempt(args, {"contract": "fixed"}, require_canonical=False)

    def test_claim_permanent_staging_failure_creates_explainable_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = (base / "attempt").resolve()
            ledger = (base / "global.json").resolve()
            staging = root.parent / f".{runner.ATTEMPT_ID}.publication-staging"
            preflight = base / "preflight.log"
            preflight.write_text("ready\n", encoding="utf-8")
            args = argparse.Namespace(
                output_dir=str(root),
                global_ledger=str(ledger),
                preflight_log=str(preflight),
            )
            original_mkdir = Path.mkdir

            def fail_staging_mkdir(path, *call_args, **call_kwargs):
                if Path(path) == staging:
                    raise OSError("synthetic staging mkdir failure")
                return original_mkdir(path, *call_args, **call_kwargs)

            with (
                patch.object(Path, "mkdir", new=fail_staging_mkdir),
                self.assertRaisesRegex(OSError, "staging mkdir failure"),
            ):
                runner.claim_attempt(args, {"contract": "fixed"}, require_canonical=False)
            self.assertTrue(root.is_dir())
            tombstone = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(tombstone["status"], runner.GLOBAL_INVALID_STATUS)
            self.assertFalse(tombstone["claim_ledger_was_present"])
            self.assertIsNone(tombstone["claimed_ledger_sha256"])
            self.assertEqual(
                json.loads(
                    (root / runner.GLOBAL_LEDGER_ARCHIVE_NAME).read_text(
                        encoding="utf-8"
                    )
                ),
                tombstone,
            )
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                runner.claim_attempt(args, {"contract": "fixed"}, require_canonical=False)

    def test_spawned_worker_reauthorization_failure_writes_invalid_tombstones(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "attempt"
            staging = base / "staging"
            root.mkdir()
            staging.mkdir()
            layout = runner.ExpansionLabelLayout(
                root=root,
                global_ledger=base / "global.json",
                publish_staging=staging,
                run_contract={},
                run_contract_sha256=RUN_SHA,
                worker_sibling_ledgers={
                    spec.worker_id: base / f"{spec.worker_id}.json"
                    for spec in expected_worker_specs()
                },
                started_at_utc="2026-07-17T00:00:00.000000Z",
            )
            runtime_loads: list[int] = []
            teacher_forwards: list[int] = []

            def fail_authorization(_layout, _args):
                raise RuntimeError("synthetic worker authorization failure")

            def forbidden_runtime(*_args):
                runtime_loads.append(1)
                raise AssertionError("runtime must not load")

            def forbidden_teacher(**_kwargs):
                teacher_forwards.append(1)
                raise AssertionError("teacher must not run")

            terminal = runner.run_worker(
                layout=layout,
                spec=expected_worker_specs()[0],
                authorized=None,
                args=argparse.Namespace(),
                authorization_loader=fail_authorization,
                runtime_loader=forbidden_runtime,
                state_runner=forbidden_teacher,
            )
            self.assertEqual(terminal["outcome"], runner.WORKER_INVALID_STATUS)
            self.assertEqual(terminal["attempted_state_indices"], [])
            self.assertEqual(terminal["completed_state_indices"], [])
            self.assertEqual(runtime_loads, [])
            self.assertEqual(teacher_forwards, [])
            ledgers = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in runner._worker_ledger_paths(
                    layout,
                    expected_worker_specs()[0],
                )
            ]
            self.assertEqual(ledgers[0], ledgers[1])
            self.assertEqual(ledgers[1], ledgers[2])
            self.assertEqual(ledgers[0]["status"], runner.WORKER_INVALID_STATUS)
            self.assertFalse(
                (root / "workers/even" / runner.WORKER_RUNTIME_FILENAME).exists()
            )
            self.assertTrue(
                (root / "workers/even" / runner.WORKER_TERMINAL_FILENAME).is_file()
            )

    def test_preclaim_authorization_failure_stops_monitor_without_contract_hash(self) -> None:
        samples = (
            {
                "nvidia_smi_index": 2,
                "gpu_uuid": "GPU-00000000",
                "utilization_percent": 95,
            },
            {
                "nvidia_smi_index": 3,
                "gpu_uuid": "GPU-11111111",
                "utilization_percent": 96,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            ready = base / "ready.json"
            log = base / "monitor.jsonl"
            summary = base / "summary.json"
            failures: list[BaseException] = []

            def monitor() -> None:
                try:
                    runner.run_monitor_sidecar(
                        ready_file=ready,
                        log_file=log,
                        summary_file=summary,
                        sampler=lambda: samples,
                        sleep_fn=lambda _seconds: time.sleep(0.01),
                    )
                except BaseException as error:
                    failures.append(error)

            thread = threading.Thread(target=monitor)
            thread.start()
            deadline = time.monotonic() + 2.0
            while not ready.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.is_file())
            args = argparse.Namespace(
                monitor_ready_file=str(ready),
                utilization_monitor_log=str(log),
                monitor_summary=str(summary),
            )
            with (
                patch.object(
                    runner,
                    "authorize_expansion_label_run",
                    side_effect=RuntimeError("synthetic preclaim authorization failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "preclaim authorization failure"),
            ):
                runner.execute_attempt(args)
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            stop = json.loads(
                runner.monitor_stop_request_path(summary).read_text(encoding="utf-8")
            )
            aborted = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(stop["status"], runner.MONITOR_PRECLAIM_ABORT_STOP_STATUS)
            self.assertNotIn("run_contract_sha256", stop)
            self.assertEqual(
                aborted["status"],
                runner.MONITOR_PRECLAIM_ABORT_SUMMARY_STATUS,
            )
            self.assertEqual(
                aborted["preclaim_abort_identity_sha256"],
                stop["preclaim_abort_identity_sha256"],
            )

    def test_all_preclaim_failures_request_monitor_termination(self) -> None:
        args = argparse.Namespace(monitor_ready_file="ready.json")
        for failed_stage in ("authorization", "build", "claim"):
            with self.subTest(failed_stage=failed_stage):
                authorization_error = (
                    RuntimeError("authorization failed")
                    if failed_stage == "authorization"
                    else None
                )
                build_error = (
                    RuntimeError("build failed") if failed_stage == "build" else None
                )
                claim_error = (
                    RuntimeError("claim failed") if failed_stage == "claim" else None
                )
                with (
                    patch.object(runner, "_validate_external_ready"),
                    patch.object(
                        runner,
                        "authorize_expansion_label_run",
                        return_value=SimpleNamespace(),
                        side_effect=authorization_error,
                    ),
                    patch.object(
                        runner,
                        "build_run_contract",
                        return_value={"contract": "fixed"},
                        side_effect=build_error,
                    ),
                    patch.object(
                        runner,
                        "claim_attempt",
                        return_value=SimpleNamespace(),
                        side_effect=claim_error,
                    ),
                    patch.object(runner, "terminate_preclaim_monitor") as terminate,
                    self.assertRaisesRegex(RuntimeError, f"{failed_stage} failed"),
                ):
                    runner.execute_attempt(args)
                terminate.assert_called_once_with(args)

    def test_external_json_hardlink_fsyncs_destination_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            with (
                patch.object(runner.os, "open", return_value=123) as opened,
                patch.object(runner.os, "fsync") as fsync,
                patch.object(runner.os, "close") as closed,
            ):
                runner._write_external_json_exclusive(path, {"value": 1})
            opened.assert_called_once_with(path.parent, os.O_RDONLY)
            fsync.assert_any_call(123)
            closed.assert_called_once_with(123)

    def test_monitor_sidecar_produces_bound_fixed_schema(self) -> None:
        samples = (
            {
                "nvidia_smi_index": 2,
                "gpu_uuid": "GPU-00000000",
                "utilization_percent": 95,
            },
            {
                "nvidia_smi_index": 3,
                "gpu_uuid": "GPU-11111111",
                "utilization_percent": 96,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "ready.json"
            log = root / "monitor.jsonl"
            summary = root / "summary.json"

            def stop(_seconds: float) -> None:
                runner.publish_monitor_stop_request(
                    summary_file=summary,
                    run_contract_sha256=RUN_SHA,
                )

            result = runner.run_monitor_sidecar(
                ready_file=ready,
                log_file=log,
                summary_file=summary,
                sampler=lambda: samples,
                sleep_fn=stop,
            )
            self.assertEqual(result["run_contract_sha256"], RUN_SHA)
            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(result["visible_gpu_uuids"], ["GPU-00000000", "GPU-11111111"])
            self.assertTrue(ready.is_file())
            self.assertTrue(log.read_bytes())
            self.assertTrue(runner.monitor_stop_request_path(summary).is_file())
            sample = json.loads(log.read_text(encoding="utf-8").strip())
            self.assertEqual(
                [gpu["nvidia_smi_index"] for gpu in sample["gpus"]],
                [2, 3],
            )

    def test_monitor_sample_validation_rejects_bad_or_changed_inventory(self) -> None:
        valid = (
            {
                "nvidia_smi_index": 2,
                "gpu_uuid": "GPU-00000000",
                "utilization_percent": 95,
            },
            {
                "nvidia_smi_index": 3,
                "gpu_uuid": "GPU-11111111",
                "utilization_percent": 96,
            },
        )
        normalized = runner._validated_monitor_sample(valid)
        inventory = tuple(
            (row["nvidia_smi_index"], row["gpu_uuid"]) for row in normalized
        )
        with self.assertRaisesRegex(ValueError, "utilization"):
            runner._validated_monitor_sample(
                ({**valid[0], "utilization_percent": 101}, valid[1])
            )
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            runner._validated_monitor_sample(
                ({**valid[0], "nvidia_smi_index": 4}, valid[1]),
                expected_inventory=inventory,
            )

    def test_prohibited_operations_and_formal_cli_are_exactly_frozen(self) -> None:
        self.assertEqual(len(PROHIBITED_OPERATION_COUNTS), 15)
        self.assertEqual(set(PROHIBITED_OPERATION_COUNTS.values()), {0})
        parser_flags = [
            option
            for action in runner._build_parser()._actions
            for option in action.option_strings
            if option not in {"-h", "--help"}
        ]
        self.assertEqual(
            parser_flags,
            runner.runner_interface_contract()["required_cli_options"],
        )
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("generate_native_action(", source)
        self.assertNotIn("fingerprint", source)
        self.assertNotIn('.to(device="cpu")', source)
        self.assertNotIn(".item()", source)
        self.assertIn('multiprocessing.get_context("spawn")', source)
        self.assertNotIn('multiprocessing.get_context("fork")', source)


if __name__ == "__main__":
    unittest.main()
