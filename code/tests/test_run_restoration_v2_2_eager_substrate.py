from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from causalcache.data.restoration_v2_screening import ScreeningState
from causalcache.policy.gui_owl_v2 import gui_owl_v2_action_to_androidworld
from causalcache.policy.gui_owl_v2_1 import parse_gui_owl_v2_1_output
from causalcache.restoration_v2_2_eager_artifact import (
    EXPECTED_SOURCE_INVENTORY_PATHS,
    PASS_OUTCOME,
    WorkerSpec,
    expected_worker_specs,
    pretty_json_bytes,
)
from scripts import run_restoration_v2_1_full_45_substrate as old_runner
from scripts import run_restoration_v2_2_eager_substrate as runner


VALID_OUTPUT = (
    '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n'
    "</tool_call>"
)
SOURCE_COMMIT = "a" * 40


def fake_states() -> tuple[ScreeningState, ...]:
    return tuple(
        ScreeningState(
            index=index,
            role="v2_label_train" if index < 30 else "v2_development",
            trajectory_id=f"trajectory-{index // 3:02d}",
            decision_step_id=4 + index % 3,
            candidate_event_step_ids=tuple(range(1, 3 + index % 3)),
        )
        for index in range(45)
    )


def fake_generation_metadata(device: str) -> dict[str, Any]:
    return {
        "protocol_id": "causalcache_restoration_v2_1_official_tool_interface",
        "generation_interface": "processor_apply_chat_template_official_tools_kwarg",
        "official_tools_argument_count": 1,
        "official_tool_schema_sha256": "1" * 64,
        "chat_template_file_sha256": "2" * 64,
        "chat_template_text_sha256": "3" * 64,
        "assistant_prefix_token_ids": [151644, 77091, 198],
        "tool_call_open_token_id": 151657,
        "tool_call_close_token_id": 151658,
        "generation_eos_token_id": 151658,
        "suppressed_standard_eos_token_ids": [151645, 151643],
        "generation_pad_token_id": 151643,
        "generation_num_beams": 1,
        "generation_num_return_sequences": 1,
        "generation_standard_eos_suppression": (
            "negative_infinity_via_transformers_suppress_tokens"
        ),
        "host_injected_tool_call_closer": False,
        "output_recovery_or_normalization": False,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "max_new_tokens": 256,
        "decoded_output_utf8_sha256": hashlib.sha256(
            VALID_OUTPUT.encode()
        ).hexdigest(),
        "generated_tool_call_close_token_count": 1,
        "final_generated_token_id": 151658,
        "termination_reason": "model_emitted_tool_call_close",
        "model_emitted_tool_call_close": True,
        "device": device,
    }


def fake_teacher_metadata(device: str) -> dict[str, Any]:
    generation = fake_generation_metadata(device)
    interface_keys = old_runner.RUNTIME_GENERATION_BINDING_KEYS
    return {
        **{key: generation[key] for key in interface_keys},
        "batch_size": 1,
        "device": device,
        "dtype": "torch.bfloat16",
        "vocabulary_size": 151669,
        "distance_span": "official_tool_call_open_through_close_inclusive",
        "teacher_context": "official_tools_prompt_plus_assistant_prefix_direct",
        "teacher_carrier": None,
        "teacher_target_json_separators": [", ", ": "],
        "teacher_target_ends_with_model_generation_eos": True,
        "teacher_target_disjoint_from_suppressed_standard_eos": True,
        "teacher_standard_eos_suppressed_token_ids": [151645, 151643],
        "teacher_standard_eos_suppression_value": -3.3895313892515355e38,
        "teacher_standard_eos_suppression_semantics": (
            "torch_finfo_bfloat16_min_finite_generation_alignment"
        ),
        "teacher_standard_eos_mask_application": (
            "same_mask_on_every_reference_and_candidate_action_path_position_"
            "before_float32_log_softmax"
        ),
        "teacher_raw_logits_mutated": False,
        "finite_logits_validation": (
            "deferred_to_gpu_kl_invalid_to_nan_final_distance"
        ),
        "extended_prompt_aligned_inputs": ["attention_mask"],
        "logits_to_keep": 5,
        "latency_seconds": 0.1,
        "peak_gpu_memory_allocated_bytes": 1,
        "peak_gpu_memory_reserved_bytes": 1,
        "full_logit_tensor_host_transfers": 0,
        "samples": [{"distance_action_tokens": 5}],
    }


def fake_distance_audit(device: str) -> dict[str, Any]:
    return {
        "operation": "teacher_forced_full_vocabulary_mean_kl_on_distance_token_span",
        "candidate_representation": "logits",
        "batch_size": 1,
        "distance_tokens": 5,
        "vocabulary_size": 151669,
        "device": device,
        "reference_input_dtype": "torch.float32",
        "candidate_input_dtype": "torch.bfloat16",
        "compute_dtype": "torch.float32",
        "output_dtype": "torch.float32",
        "reference_batch_stride": 758345,
        "reference_zero_copy_batch_expansion": False,
        "reference_compute_batch_size": 1,
        "reduction": "full_vocabulary_sum_then_distance_token_mean_per_example",
        "log_normalization_atol": 5e-4,
        "negative_kl_atol": 1e-5,
        "numeric_validation": "gpu_resident_per_example_predicates",
        "invalid_numeric_output": "nan_final_distance",
        "device_validation_category_count": 4,
        "validation_scalar_host_reads": 0,
        "full_tensor_host_transfers": 0,
    }


def fake_inner_record(
    *, projection: dict[str, Any], contract_sha256: str, device: str
) -> dict[str, Any]:
    record = old_runner._base_state_record(
        projection=projection,
        run_contract_sha256=contract_sha256,
    )
    parsed = parse_gui_owl_v2_1_output(VALID_OUTPUT)
    action = parsed.canonical_action.arguments()
    bridge = gui_owl_v2_action_to_androidworld(
        parsed.canonical_action,
        screen_width=1080,
        screen_height=1920,
    )
    generation = fake_generation_metadata(device)
    teacher = fake_teacher_metadata(device)
    record.update(
        {
            "outcome": old_runner.STATE_OUTCOME_VALID,
            "failure": None,
            "parse_success": True,
            "parse_success_count": 2,
            "model_emitted_closer_count": 2,
            "androidworld_bridge_count": 2,
            "repeat_canonical_action_agreement": True,
            "finite_logit_distances": True,
            "canonical_action": action,
            "androidworld_bridge": bridge,
            "screen_dimensions": {"width": 1080, "height": 1920},
            "native_generations": [
                {
                    "repeat_index": repeat,
                    "output_text": VALID_OUTPUT,
                    "metadata": copy.deepcopy(generation),
                    "canonical_action": action,
                    "androidworld_bridge": bridge,
                    "parse_error_type": None,
                    "parse_error_message": None,
                }
                for repeat in (1, 2)
            ],
            "teacher_forwards": {
                "reference_1": copy.deepcopy(teacher),
                "reference_2": copy.deepcopy(teacher),
                "summary_only": copy.deepcopy(teacher),
            },
            "distances": {
                "repeat_reference_kl": 1e-6,
                "summary_reference_kl": 0.01,
            },
            "distance_audits": {
                "repeat_reference_kl": fake_distance_audit(device),
                "summary_reference_kl": fake_distance_audit(device),
            },
            "operation_counts": {
                "generation_call_count": 2,
                "teacher_forward_count": 3,
                "kl_measurement_count": 2,
                **old_runner.FORBIDDEN_OPERATION_COUNTS,
            },
            "started_at_utc": "2026-07-15T00:00:00Z",
            "ended_at_utc": "2026-07-15T00:00:01Z",
            "duration_seconds": 1.0,
        }
    )
    return record


def fake_runtime_metadata(spec: WorkerSpec) -> dict[str, Any]:
    generation = fake_generation_metadata(spec.device)
    metadata = {
        key: generation[key] for key in old_runner.RUNTIME_GENERATION_BINDING_KEYS
    }
    metadata.update(
        {
            "runtime_profile_id": "causalcache_restoration_v2_2_eager_runtime",
            "device": spec.device,
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
            "deterministic_algorithms_requested": False,
            "deterministic_algorithms_enabled": False,
            "strict_cuda_determinism_claimed": False,
            "gpu_name": "NVIDIA H200",
            "gpu_uuid": f"GPU-{spec.worker_id}",
            "gpu_pci_bus_id": f"00000000:0{8 + spec.index_parity}:00.0",
            "logical_device_index": spec.index_parity,
            "nvidia_smi_index": 4 + 2 * spec.index_parity,
            "container_image_digest": runner.CANONICAL_IMAGE_DIGEST,
            "python_version": "3.12.3",
            "torch_version": "2.11.0+cu130",
            "torch_cuda_version": "13.0",
            "cudnn_version": 91900,
            "transformers_version": "5.6.0",
            "nvidia_driver_version": "570.172.08",
        }
    )
    return metadata


def fake_run_contract(root: Path, ledger: Path) -> dict[str, Any]:
    states = fake_states()
    parent_values = {
        "spatial_reference_evidence": (
            "d62ad05f6fdef06a3551f2ebe9f83f28327068f0020ff3e61891da4f46ce5ecc",
            1873920,
            "VALIDATED_CANONICAL_SPATIAL_REFERENCE_RAW_ARCHIVE",
            "EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY",
        ),
        "v2_1_full_45_evidence": (
            "8cd53d6e56d5ad509da2af91d73d9e83b4db989ffc26aa18e1bca84e4c4f4fa4",
            962560,
            "VALID_RESTORATION_V2_1_FULL_45_ARTIFACT",
            "NO_GO_V2_1_FULL_45_SUBSTRATE",
        ),
        "pilot_evidence": (
            "f71d5fd575dde48ae8b3e50a19dd2fbecfa02d7d5ae6087f909a47dfd7032064",
            133120,
            "VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT",
            "PASS_V2_1_INTERFACE_PILOT",
        ),
        "processor_evidence": (
            "5349ffc6b91bf93ed26d25104fe6c907f31ccc497007a5c0ae6ed5ddf5c84991",
            7609803,
            "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT",
            "PASSED_90_PROMPT_PROCESSOR_PREFLIGHT",
        ),
    }
    parent = {
        name: {
            "path": str(root.parent / f"{name}.raw"),
            "sha256": values[0],
            "size_bytes": values[1],
            "validation_status": values[2],
            "validation_outcome": values[3],
        }
        for name, values in parent_values.items()
    }
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "protocol_id": runner.PROTOCOL_ID,
        "contract_source": {
            "path": runner.CANONICAL_CONFIG_PATH,
            "sha256": runner.FROZEN_RESTORATION_V2_2_EAGER_SHA256,
        },
        "git_identity": {
            "branch": "main",
            "commit": SOURCE_COMMIT,
            "origin_main": SOURCE_COMMIT,
            "remote_main": SOURCE_COMMIT,
            "remote_url": "https://github.com/luojiaxuan/CausalCache.git",
            "worktree": "clean_including_untracked",
        },
        "source_inventory": [
            {"path": path, "sha256": "2" * 64, "git_commit": SOURCE_COMMIT}
            for path in EXPECTED_SOURCE_INVENTORY_PATHS
        ],
        "parent_authorization": parent,
        "canonical_inputs": {
            "scientific_config": {
                "path": runner.CANONICAL_SCIENTIFIC_CONFIG_PATH,
                "sha256": "3" * 64,
            },
            "selection_manifest": {
                "path": runner.CANONICAL_SELECTION_MANIFEST_PATH,
                "sha256": "4" * 64,
            },
            "ocr_backend_config": {
                "path": runner.CANONICAL_OCR_BACKEND_CONFIG_PATH,
                "sha256": "5" * 64,
            },
            "snapshot_manifest": {
                "path": runner.CANONICAL_SNAPSHOT_MANIFEST_PATH,
                "sha256": "6" * 64,
            },
            "derived_artifact": {
                "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
                "immutable_revision": "89f136abaff797e14fe758a198996e51032a10a6",
                "artifact_tree_sha256": (
                    "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e"
                ),
            },
        },
        "runtime_requirements": {
            "gpu_name": "NVIDIA H200",
            "container_image_digest": runner.CANONICAL_IMAGE_DIGEST,
            "python_version": "3.12.3",
            "torch_version": "2.11.0+cu130",
            "torch_cuda_version": "13.0",
            "cudnn_version": 91900,
            "transformers_version": "5.6.0",
            "nvidia_driver_version": "570.172.08",
            "runtime_profile_id": "causalcache_restoration_v2_2_eager_runtime",
            "attention_implementation": "eager",
        },
        "worker_topology": {
            "worker_count": 2,
            "gpu_model": "NVIDIA H200",
            "one_process_per_device": True,
            "workers": [spec.to_dict() for spec in expected_worker_specs()],
            "cross_worker_state_stealing_allowed": False,
            "worker_failure_invalidates_entire_attempt": True,
            "worker_outputs_must_be_disjoint": True,
            "worker_union_must_equal_fixed_denominator": True,
        },
        "states": [
            old_runner._full45_projection(state, index)
            for index, state in enumerate(states)
        ],
        "gate": {
            "minimum_screening_states": 20,
            "minimum_parse_coverage": 0.99,
            "minimum_finite_logit_coverage": 1.0,
            "minimum_repeat_canonical_action_agreement": 1.0,
            "minimum_memory_sensitive_states": 8,
        },
        "operation_policy": {
            "retry_count": 0,
            "top_up_count": 0,
            "v2_1_raw_state_reuse_count": 0,
            "same_state_repeats_must_remain_on_one_worker_device": True,
            "merge_only_after_both_worker_terminals": True,
            "cpu_recompute_gate_after_index_order_merge": True,
            "resume_allowed": False,
            "interrupted_attempt_is_terminal_invalid": True,
            "resume_or_top_up_count": 0,
        },
        "attempt_identity": {
            "attempt_id": "restoration-v2-2-eager-full-45-substrate-v1",
            "output_dir": str(root.resolve()),
            "global_ledger": str(ledger.resolve()),
            "host_alias": "hyper01",
            "host_hostname": "node-radixark-16-0001",
            "container_id": "7" * 64,
            "container_image_digest": runner.CANONICAL_IMAGE_DIGEST,
        },
        "execution_argv": ["/usr/bin/python3", "/data/CausalCache/code/scripts/run.py"],
    }


def successful_attempt(directory: Path) -> tuple[dict[str, Any], Path, Path]:
    root = directory / "raw"
    ledger = directory / "attempt.json"
    states = fake_states()
    contract = fake_run_contract(root, ledger)

    def launch(layout: runner.V22Layout, specs: Any) -> None:
        runner._write_json_exclusive(
            layout.root / runner.RUNTIME_BARRIER_RELEASE_FILENAME,
            {
                "schema_version": runner.SCHEMA_VERSION,
                "protocol_id": runner.PROTOCOL_ID,
                "status": "COORDINATOR_RELEASED_BOTH_WORKERS",
                "run_contract_sha256": layout.run_contract_sha256,
            },
        )
        for spec in specs:
            runner.run_worker_shard(
                layout=layout,
                spec=spec,
                artifact=SimpleNamespace(),
                states=states,
                runtime_loader=lambda selected: runner.RuntimeBindings(
                    runtime=object(),
                    parse_error_class=ValueError,
                    distance_backend=object(),
                    metadata=fake_runtime_metadata(selected),
                ),
                state_kernel=lambda **kwargs: fake_inner_record(
                    projection=old_runner._full45_projection(
                        kwargs["state"], kwargs["full45_index"]
                    ),
                    contract_sha256=kwargs["run_contract_sha256"],
                    device=spec.device,
                ),
            )

    result = runner.execute_two_worker_attempt(
        run_contract=contract,
        output_dir=root,
        global_ledger=ledger,
        worker_launcher=launch,
    )
    return result, root, ledger


def deleted_root_ledger_attempt(
    directory: Path,
) -> tuple[dict[str, Any], Path, Path]:
    root = directory / "raw"
    ledger = directory / "attempt.json"
    states = fake_states()
    contract = fake_run_contract(root, ledger)

    def launch(layout: runner.V22Layout, specs: Any) -> None:
        runner._write_json_exclusive(
            layout.root / runner.RUNTIME_BARRIER_RELEASE_FILENAME,
            {
                "schema_version": runner.SCHEMA_VERSION,
                "protocol_id": runner.PROTOCOL_ID,
                "status": "COORDINATOR_RELEASED_BOTH_WORKERS",
                "run_contract_sha256": layout.run_contract_sha256,
            },
        )
        for spec in specs:
            runner.run_worker_shard(
                layout=layout,
                spec=spec,
                artifact=SimpleNamespace(),
                states=states,
                runtime_loader=lambda selected: runner.RuntimeBindings(
                    runtime=object(),
                    parse_error_class=ValueError,
                    distance_backend=object(),
                    metadata=fake_runtime_metadata(selected),
                ),
                state_kernel=lambda **kwargs: fake_inner_record(
                    projection=old_runner._full45_projection(
                        kwargs["state"], kwargs["full45_index"]
                    ),
                    contract_sha256=kwargs["run_contract_sha256"],
                    device=spec.device,
                ),
            )
        (
            root
            / runner.WORKER_DIRECTORY
            / "even"
            / runner.WORKER_LEDGER_FILENAME
        ).unlink()

    result = runner.execute_two_worker_attempt(
        run_contract=contract,
        output_dir=root,
        global_ledger=ledger,
        worker_launcher=launch,
    )
    return result, root, ledger


class TwoWorkerRunnerTest(unittest.TestCase):
    def test_parity_shards_are_exact_and_disjoint(self) -> None:
        even, odd = expected_worker_specs()
        self.assertEqual(len(even.state_indices), 23)
        self.assertEqual(len(odd.state_indices), 22)
        self.assertEqual(set(even.state_indices) | set(odd.state_indices), set(range(45)))
        self.assertFalse(set(even.state_indices) & set(odd.state_indices))
        self.assertNotEqual(even.device, odd.device)

    def test_runtime_barrier_rejects_same_physical_gpu(self) -> None:
        even, odd = expected_worker_specs()
        runtimes = {
            "even": fake_runtime_metadata(even),
            "odd": fake_runtime_metadata(odd),
        }
        for field in ("gpu_uuid", "gpu_pci_bus_id", "nvidia_smi_index"):
            mutated = copy.deepcopy(runtimes)
            mutated["odd"][field] = mutated["even"][field]
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "distinct physical"
            ):
                runner.validate_worker_runtime_pair(mutated)

    def test_device_identity_maps_logical_device_by_uuid(self) -> None:
        properties = SimpleNamespace(uuid="selected-uuid")
        torch_module = SimpleNamespace(
            device=lambda value: value,
            cuda=SimpleNamespace(get_device_properties=lambda device: properties),
        )
        nvidia_smi = SimpleNamespace(
            stdout=(
                "4, NVIDIA H200, GPU-other, 00000000:08:00.0, 570.172.08\n"
                "6, NVIDIA H200, GPU-selected-uuid, 00000000:0a:00.0, 570.172.08\n"
            )
        )
        with mock.patch.object(runner.subprocess, "run", return_value=nvidia_smi):
            identity = runner._device_identity("cuda:1", torch_module=torch_module)
        self.assertEqual(identity["logical_device_index"], 1)
        self.assertEqual(identity["nvidia_smi_index"], 6)
        self.assertEqual(identity["gpu_uuid"], "GPU-selected-uuid")

    def test_runtime_barrier_rejects_shared_wrong_stack(self) -> None:
        even, odd = expected_worker_specs()
        runtimes = {
            "even": fake_runtime_metadata(even),
            "odd": fake_runtime_metadata(odd),
        }
        for metadata in runtimes.values():
            metadata["nvidia_driver_version"] = "999.0"
        with self.assertRaisesRegex(ValueError, "runtime metadata"):
            runner.validate_worker_runtime_pair(runtimes)

    def test_global_claim_precedes_worker_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            observed = []

            def launch(layout: runner.V22Layout, specs: Any) -> None:
                del specs
                observed.append(layout.global_ledger.is_file())
                observed.append(layout.manifest.is_file())

            result = runner.execute_two_worker_attempt(
                run_contract=fake_run_contract(root, ledger),
                output_dir=root,
                global_ledger=ledger,
                worker_launcher=launch,
            )
            self.assertEqual(observed, [True, True])
            self.assertEqual(result["outcome"], runner.INVALID_OUTCOME)

    def test_complete_workers_merge_in_global_index_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = successful_attempt(Path(directory))
        self.assertEqual(result["outcome"], PASS_OUTCOME)
        self.assertEqual(result["merge_order"], list(range(45)))
        self.assertEqual(result["metrics"]["generation_call_count"], 90)
        self.assertEqual(result["metrics"]["gate_model_forward_count"], 0)

    def test_partial_worker_invalidates_entire_attempt_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            states = fake_states()

            def launch(layout: runner.V22Layout, specs: Any) -> None:
                spec = specs[0]
                runner.run_worker_shard(
                    layout=layout,
                    spec=spec,
                    artifact=SimpleNamespace(),
                    states=states,
                    runtime_loader=lambda selected: runner.RuntimeBindings(
                        runtime=object(),
                        parse_error_class=ValueError,
                        distance_backend=object(),
                        metadata=fake_runtime_metadata(selected),
                    ),
                    state_kernel=lambda **kwargs: (_ for _ in ()).throw(
                        RuntimeError("synthetic worker failure")
                    ),
                )

            result = runner.execute_two_worker_attempt(
                run_contract=fake_run_contract(root, ledger),
                output_dir=root,
                global_ledger=ledger,
                worker_launcher=launch,
            )
            self.assertEqual(result["outcome"], runner.INVALID_OUTCOME)
            self.assertFalse(result["retry_performed"])
            self.assertFalse(result["top_up_performed"])
            self.assertEqual(result["attempted_state_count"], 1)

    def test_existing_global_claim_forbids_second_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            ledger.write_text("claimed")
            with self.assertRaises(FileExistsError):
                runner.claim_global_attempt(
                    run_contract=fake_run_contract(root, ledger),
                    output_dir=root,
                    global_ledger=ledger,
                )

    def test_existing_worker_sibling_ledger_forbids_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            sibling = ledger.with_name("attempt.even.json")
            sibling.write_text("claimed")
            with self.assertRaisesRegex(FileExistsError, "sibling"):
                runner.claim_global_attempt(
                    run_contract=fake_run_contract(root, ledger),
                    output_dir=root,
                    global_ledger=ledger,
                )

    def test_hard_interruption_seals_without_retry_or_policy_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            ledger = base / "attempt.json"
            layout = runner.claim_global_attempt(
                run_contract=fake_run_contract(root, ledger),
                output_dir=root,
                global_ledger=ledger,
            )
            even = expected_worker_specs()[0]
            sibling = runner._worker_ledger_record(
                layout=layout,
                spec=even,
                status="WORKER_SHARD_RUNNING_NO_RETRY",
                claimed_at_utc=layout.started_at_utc,
                attempted=[0],
                completed=[],
            )
            runner._replace_json_durable(
                layout.worker_sibling_ledgers["even"], sibling
            )
            result = runner.seal_interrupted_attempt(
                output_dir=root,
                global_ledger=ledger,
            )
            self.assertEqual(result["outcome"], runner.INVALID_OUTCOME)
            self.assertEqual(
                result["forensic_inventory"]["even"][
                    "missing_attempt_marker_indices"
                ],
                [0],
            )
            self.assertFalse(result["retry_performed"])

    def test_authorization_rejects_alternate_path_host_image_and_container(self) -> None:
        canonical_output = (
            "/data/experiments/causalcache/"
            "restoration-v2-2-eager-full-45-substrate-v1"
        )
        canonical_ledger = (
            "/data/experiments/causalcache/"
            ".restoration-v2-2-eager-full-45-substrate-v1.attempt.json"
        )
        base = {
            "repository_root": "/tmp/repository",
            "contract": "/tmp/repository/code/configs/contract.json",
            "output_dir": canonical_output,
            "global_ledger": canonical_ledger,
            "container_image_digest": runner.CANONICAL_IMAGE_DIGEST,
            "host_alias": "hyper01",
            "host_hostname": "node-radixark-16-0001",
            "container_id": "7" * 64,
            "model_dir": "/data/model",
        }
        contract = SimpleNamespace(
            data={
                "execution": {
                    "canonical_persistent_output_dir": canonical_output,
                    "canonical_global_attempt_ledger": canonical_ledger,
                    "required_host_class": "Hyper_H200",
                }
            }
        )
        mutations = (
            {"output_dir": "/tmp/alternate"},
            {"host_alias": "aries", "host_hostname": "aries"},
            {"container_image_digest": "sha256:" + "0" * 64},
            {"container_id": "short"},
        )
        for mutation in mutations:
            values = {**base, **mutation}
            with self.subTest(mutation=mutation), mock.patch.object(
                runner.RestorationV22EagerContract,
                "load",
                return_value=contract,
            ):
                with self.assertRaisesRegex(ValueError, "canonical output"):
                    runner.authorize_production_v22(SimpleNamespace(**values))


if __name__ == "__main__":
    unittest.main()
