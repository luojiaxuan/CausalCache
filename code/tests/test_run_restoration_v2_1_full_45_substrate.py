from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from causalcache.data.restoration_v2_screening import ScreeningState
from causalcache.policy.gui_owl_v2_1 import parse_gui_owl_v2_1_output
import scripts.run_restoration_v2_1_full_45_substrate as runner


VALID_OUTPUT = (
    '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n'
    "</tool_call>"
)
POLICY_PROTOCOL = "causalcache_restoration_v2_1_official_tool_interface"


class _Image:
    size = (1080, 1920)


class _ParseError(ValueError):
    def __init__(self, output_text: str, metadata: dict[str, Any]) -> None:
        super().__init__("strict parse failed")
        self.output_text = output_text
        self.metadata = metadata
        self.parse_error_type = "ValueError"
        self.parse_error_message = "whole output mismatch"


def _generation_metadata(output: str) -> dict[str, Any]:
    closed = output.endswith("</tool_call>")
    return {
        "protocol_id": POLICY_PROTOCOL,
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
            output.encode("utf-8")
        ).hexdigest(),
        "generated_tool_call_close_token_count": int(closed),
        "final_generated_token_id": 151658 if closed else None,
        "termination_reason": (
            "model_emitted_tool_call_close"
            if closed
            else "max_new_tokens_without_tool_call_close"
        ),
        "model_emitted_tool_call_close": closed,
    }


def _runtime_metadata(model_dir: str) -> dict[str, Any]:
    interface = _generation_metadata(VALID_OUTPUT)
    metadata = {
        key: interface[key]
        for key in runner.RUNTIME_METADATA_KEYS
        if key in interface
    }
    metadata.update(
        {
            "model_dir": str(Path(model_dir).resolve()),
            "model_repo": "repo",
            "model_revision": "revision",
            "snapshot_manifest_sha256": "4" * 64,
            "verified_model_file_count": 10,
            "verified_model_total_bytes": 1024,
            "transformers_version": "test",
            "transformers_source_sha256": {"modeling": "5" * 64},
            "dtype": "bfloat16",
            "device": runner.CANONICAL_PILOT_DEVICE,
            "frozen": True,
            "single_device": True,
            "processor_class": "TestProcessor",
            "model_class": "TestModel",
            "torch_version": "test",
            "target_effective_visual_tokens_per_image": 4096,
            "min_pixels": 4096 * 28 * 28,
            "max_pixels": 4096 * 28 * 28,
        }
    )
    if set(metadata) != runner.RUNTIME_METADATA_KEYS:
        raise AssertionError("runtime metadata fixture drifted")
    return metadata


def _teacher_metadata() -> dict[str, Any]:
    return {
        **_generation_metadata(VALID_OUTPUT),
        "batch_size": 1,
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "vocabulary_size": 151700,
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
        "logits_to_keep": 8,
        "latency_seconds": 0.1,
        "peak_gpu_memory_allocated_bytes": 1,
        "peak_gpu_memory_reserved_bytes": 1,
        "full_logit_tensor_host_transfers": 0,
        "samples": [
            {
                "distance_action_tokens": 8,
                "image_count": 1,
            }
        ],
    }


def _states() -> tuple[ScreeningState, ...]:
    result = []
    for index in range(45):
        step = 4 + index % 3
        result.append(
            ScreeningState(
                index=index,
                role="v2_label_train" if index < 30 else "v2_development",
                trajectory_id=f"trajectory-{index // 3:02d}",
                decision_step_id=step,
                candidate_event_step_ids=tuple(range(1, step - 1)),
            )
        )
    return tuple(result)


def _attempt_identity(root: Path, ledger: Path) -> dict[str, Any]:
    return {
        "attempt_id": runner.CANONICAL_ATTEMPT_ID,
        "canonical_persistent_output_dir": str(root.resolve()),
        "canonical_global_attempt_ledger": str(ledger.resolve()),
        "canonical_host_alias": runner.CANONICAL_PILOT_HOST_ALIAS,
        "canonical_host_hostname": runner.CANONICAL_PILOT_HOST_HOSTNAME,
        "canonical_container_id": runner.CANONICAL_PILOT_CONTAINER_ID,
        "canonical_container_image_digest": (
            runner.CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST
        ),
        "canonical_device": runner.CANONICAL_PILOT_DEVICE,
        "cross_host_attempt_allowed": False,
        "alternate_output_or_ledger_allowed": False,
        "output_or_ledger_deletion_after_first_attempt_allowed": False,
    }


def _gate_mapping() -> dict[str, Any]:
    return {
        "fixed_state_denominator": 45,
        "minimum_screening_states": 20,
        "minimum_parse_coverage": 0.99,
        "derived_required_parse_success_count": 45,
        "minimum_finite_logit_coverage": 1.0,
        "derived_required_finite_logit_state_count": 45,
        "minimum_repeat_canonical_action_agreement": 1.0,
        "derived_required_repeat_agreement_count": 45,
        "minimum_memory_sensitive_states": 8,
        "memory_sensitive_definition": (
            "summary_reference_kl_greater_than_repeat_noise_epsilon"
        ),
        "repeat_noise_epsilon": "max(1e-4,10*mean_repeat_kl)",
        "failed_states_remain_in_fixed_denominator": True,
        "pass_outcome": runner.PASS_OUTCOME,
        "fail_outcome": runner.NO_GO_OUTCOME,
        "invalid_outcome": runner.INVALID_OUTCOME,
    }


def _run_contract(
    states: tuple[ScreeningState, ...],
    root: Path,
    ledger: Path,
    model_dir: Path,
) -> dict[str, Any]:
    return {
        "states": [
            runner._full45_projection(state, index)
            for index, state in enumerate(states)
        ],
        "prohibited_operation_counts": dict(runner.FORBIDDEN_OPERATION_COUNTS),
        "attempt_identity": _attempt_identity(root, ledger),
        "policy": {
            "runtime_metadata_requirement": {
                "required_policy_protocol_id": POLICY_PROTOCOL,
                "required_metadata_keys": sorted(runner.RUNTIME_METADATA_KEYS),
                "model_dir": str(model_dir.resolve()),
                "model_repo": "repo",
                "model_revision": "revision",
                "snapshot_manifest_sha256": "4" * 64,
                "device": "cuda:0",
                "target_effective_visual_tokens_per_image": 4096,
            }
        },
        "test_identity": "full-45-fixed-run",
    }


def _message_builder(
    artifact: Any,
    state: ScreeningState,
    fidelity: str,
    decoder: Any,
) -> list[dict[str, Any]]:
    del artifact, state, fidelity, decoder
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _Image()},
                {"type": "text", "text": "fixed"},
            ],
        }
    ]


class _Runtime:
    def __init__(self, *, parse_failure_call: int | None = None) -> None:
        self.generation_calls = 0
        self.teacher_calls = 0
        self.parse_failure_call = parse_failure_call

    def generate_native_action(self, messages: Any) -> Any:
        del messages
        index = self.generation_calls
        self.generation_calls += 1
        if index == self.parse_failure_call:
            raw = "incomplete"
            raise _ParseError(raw, _generation_metadata(raw))
        return SimpleNamespace(
            output_text=VALID_OUTPUT,
            parsed_output=parse_gui_owl_v2_1_output(VALID_OUTPUT),
            metadata=_generation_metadata(VALID_OUTPUT),
        )

    def teacher_forced_distance_logits(self, messages: Any, actions: Any) -> Any:
        del messages, actions
        self.teacher_calls += 1
        return object(), _teacher_metadata()


class _DistanceBackend:
    def __init__(self, values: list[float] | None = None) -> None:
        self.values = list(values or [])
        self.measure_calls = 0

    def prepare_reference(self, logits: Any) -> object:
        del logits
        return object()

    def measure(self, reference: Any, candidate: Any) -> Any:
        del reference, candidate
        index = self.measure_calls
        self.measure_calls += 1
        value = self.values[index] if index < len(self.values) else (1e-6 if index % 2 == 0 else 0.5)
        return SimpleNamespace(
            value=value,
            audit={"full_tensor_host_transfers": 0, "test_index": index},
        )


class Full45RunnerTest(unittest.TestCase):
    def _patched_paths(self, root: Path, ledger: Path) -> Any:
        return patch.multiple(
            runner,
            CANONICAL_OUTPUT_DIR=root.resolve(),
            CANONICAL_LEDGER_PATH=ledger.resolve(),
        )

    def test_state_uses_two_generations_then_three_teachers_and_two_kl(self) -> None:
        runtime = _Runtime()
        backend = _DistanceBackend([1e-6, 0.5])
        state = _states()[0]
        record = runner.run_full45_state_once(
            artifact=SimpleNamespace(),
            state=state,
            full45_index=0,
            runtime=runtime,
            parse_error_class=_ParseError,
            distance_backend=backend,
            image_decoder=lambda value: value,
            run_contract_sha256="a" * 64,
            message_builder=_message_builder,
        )
        self.assertEqual(record["outcome"], runner.STATE_OUTCOME_VALID)
        self.assertEqual(runtime.generation_calls, 2)
        self.assertEqual(runtime.teacher_calls, 3)
        self.assertEqual(backend.measure_calls, 2)
        self.assertTrue(record["repeat_canonical_action_agreement"])

    def test_parse_failure_still_runs_two_scheduled_generations_without_teacher(self) -> None:
        runtime = _Runtime(parse_failure_call=0)
        backend = _DistanceBackend()
        record = runner.run_full45_state_once(
            artifact=SimpleNamespace(),
            state=_states()[0],
            full45_index=0,
            runtime=runtime,
            parse_error_class=_ParseError,
            distance_backend=backend,
            image_decoder=lambda value: value,
            run_contract_sha256="a" * 64,
            message_builder=_message_builder,
        )
        self.assertEqual(runtime.generation_calls, 2)
        self.assertEqual(runtime.teacher_calls, 0)
        self.assertEqual(backend.measure_calls, 0)
        self.assertEqual(record["failure"]["category"], "PARSE_FAILURE")
        self.assertEqual(record["operation_counts"]["generation_call_count"], 2)

    def test_two_parseable_but_different_actions_fail_before_teacher(self) -> None:
        alternate = (
            '<tool_call>\n{"name":"mobile_use","arguments":'
            '{"action":"system_button","button":"Back"}}\n</tool_call>'
        )

        class MismatchRuntime(_Runtime):
            def generate_native_action(self, messages: Any) -> Any:
                del messages
                output = VALID_OUTPUT if self.generation_calls == 0 else alternate
                self.generation_calls += 1
                return SimpleNamespace(
                    output_text=output,
                    parsed_output=parse_gui_owl_v2_1_output(output),
                    metadata=_generation_metadata(output),
                )

        runtime = MismatchRuntime()
        record = runner.run_full45_state_once(
            artifact=SimpleNamespace(),
            state=_states()[0],
            full45_index=0,
            runtime=runtime,
            parse_error_class=_ParseError,
            distance_backend=_DistanceBackend(),
            image_decoder=lambda value: value,
            run_contract_sha256="a" * 64,
            message_builder=_message_builder,
        )
        self.assertEqual(record["failure"]["category"], "CANONICAL_ACTION_MISMATCH")
        self.assertEqual(runtime.generation_calls, 2)
        self.assertEqual(runtime.teacher_calls, 0)

    def test_nonfinite_distance_is_scientific_failure(self) -> None:
        runtime = _Runtime()
        record = runner.run_full45_state_once(
            artifact=SimpleNamespace(),
            state=_states()[0],
            full45_index=0,
            runtime=runtime,
            parse_error_class=_ParseError,
            distance_backend=_DistanceBackend([1e-6, math.nan]),
            image_decoder=lambda value: value,
            run_contract_sha256="a" * 64,
            message_builder=_message_builder,
        )
        self.assertEqual(record["outcome"], runner.STATE_OUTCOME_FAILED)
        self.assertEqual(record["failure"]["category"], "NONFINITE_DISTANCE")
        self.assertEqual(record["operation_counts"]["teacher_forward_count"], 3)
        self.assertEqual(record["operation_counts"]["kl_measurement_count"], 2)
        self.assertIsNone(record["distances"]["summary_reference_kl"])

    def _records(self, *, sensitive_count: int) -> tuple[list[dict[str, Any]], str]:
        states = _states()
        records = []
        for index, state in enumerate(states):
            record = runner.run_full45_state_once(
                artifact=SimpleNamespace(),
                state=state,
                full45_index=index,
                runtime=_Runtime(),
                parse_error_class=_ParseError,
                distance_backend=_DistanceBackend(
                    [1e-6, 0.5 if index < sensitive_count else 0.0]
                ),
                image_decoder=lambda value: value,
                run_contract_sha256="b" * 64,
                message_builder=_message_builder,
            )
            records.append(record)
        return records, "b" * 64

    def test_aggregate_applies_eight_state_memory_sensitivity_threshold(self) -> None:
        gate = runner.Full45Gate.from_mapping(_gate_mapping())
        for sensitive_count, expected in ((8, runner.PASS_OUTCOME), (7, runner.NO_GO_OUTCOME)):
            records, run_sha = self._records(sensitive_count=sensitive_count)
            aggregate = runner.aggregate_full45_gate(
                records,
                expected_states=_states(),
                gate=gate,
                run_contract_sha256=run_sha,
                started_at_utc="2026-01-01T00:00:00Z",
                ended_at_utc="2026-01-01T00:01:00Z",
                runtime_metadata=_runtime_metadata("/model"),
            )
            self.assertEqual(aggregate["outcome"], expected)
            self.assertEqual(aggregate["metrics"]["fixed_state_denominator"], 45)
            self.assertEqual(aggregate["metrics"]["generation_call_count"], 90)
            self.assertEqual(aggregate["metrics"]["teacher_forward_count"], 135)
            self.assertEqual(aggregate["metrics"]["kl_measurement_count"], 90)

    def _claim_and_runtime(
        self,
        *,
        root: Path,
        ledger: Path,
        model_dir: Path,
    ) -> tuple[tuple[ScreeningState, ...], dict[str, Any], runner.Full45RunLayout]:
        states = _states()
        contract = _run_contract(states, root, ledger, model_dir)
        runner.claim_full45_attempt(
            states=states,
            gate=runner.Full45Gate.from_mapping(_gate_mapping()),
            run_contract=contract,
            output_dir=root,
            resume=False,
        )
        layout = runner._prepare_layout(
            states=states,
            run_contract=contract,
            output_dir=root,
        )
        runtime_record = runner._runtime_identity_record(
            layout=layout,
            runtime_metadata=_runtime_metadata(str(model_dir)),
        )
        runner._write_json_exclusive(layout.runtime_identity_path, runtime_record)
        return states, contract, layout

    def _persist_first_terminal(
        self,
        *,
        states: tuple[ScreeningState, ...],
        layout: runner.Full45RunLayout,
    ) -> None:
        first = runner.run_full45_state_once(
            artifact=SimpleNamespace(),
            state=states[0],
            full45_index=0,
            runtime=_Runtime(),
            parse_error_class=_ParseError,
            distance_backend=_DistanceBackend([1e-6, 0.5]),
            image_decoder=lambda value: value,
            run_contract_sha256=layout.contract_sha256,
            message_builder=_message_builder,
        )
        runner._advance_ledger_high_water(
            layout,
            event="STATE_ATTEMPT_CLAIMED",
            state_index=0,
        )
        runner._write_json_exclusive(
            layout.attempt_root / "000.json",
            runner._attempt_marker(
                projection=layout.projections[0],
                run_contract_sha256=layout.contract_sha256,
            ),
        )
        runner._write_json_exclusive(layout.state_root / "000.json", first)
        runner._advance_ledger_high_water(
            layout,
            event="STATE_TERMINAL_PERSISTED",
            state_index=0,
        )

    def test_resume_skips_terminal_prefix_and_runs_only_never_attempted_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "run"
            ledger = parent / ".attempt.json"
            model_dir = parent / "model"
            with self._patched_paths(root, ledger):
                states, contract, layout = self._claim_and_runtime(
                    root=root,
                    ledger=ledger,
                    model_dir=model_dir,
                )
                first = runner.run_full45_state_once(
                    artifact=SimpleNamespace(),
                    state=states[0],
                    full45_index=0,
                    runtime=_Runtime(),
                    parse_error_class=_ParseError,
                    distance_backend=_DistanceBackend([1e-6, 0.5]),
                    image_decoder=lambda value: value,
                    run_contract_sha256=layout.contract_sha256,
                    message_builder=_message_builder,
                )
                runner._advance_ledger_high_water(
                    layout,
                    event="STATE_ATTEMPT_CLAIMED",
                    state_index=0,
                )
                runner._write_json_exclusive(
                    layout.attempt_root / "000.json",
                    runner._attempt_marker(
                        projection=layout.projections[0],
                        run_contract_sha256=layout.contract_sha256,
                    ),
                )
                runner._write_json_exclusive(layout.state_root / "000.json", first)
                runner._advance_ledger_high_water(
                    layout,
                    event="STATE_TERMINAL_PERSISTED",
                    state_index=0,
                )
                runtime = _Runtime()
                aggregate = runner.run_full45_substrate(
                    artifact=SimpleNamespace(),
                    states=states,
                    runtime=runtime,
                    parse_error_class=_ParseError,
                    distance_backend=_DistanceBackend(),
                    image_decoder=lambda value: value,
                    gate=runner.Full45Gate.from_mapping(_gate_mapping()),
                    run_contract=contract,
                    output_dir=root,
                    resume=True,
                    integrity_guard=lambda: None,
                    message_builder=_message_builder,
                )
                self.assertEqual(aggregate["outcome"], runner.PASS_OUTCOME)
                self.assertEqual(runtime.generation_calls, 88)
                self.assertEqual(runtime.teacher_calls, 132)

    def test_marker_without_terminal_state_makes_resume_invalid_without_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "run"
            ledger = parent / ".attempt.json"
            model_dir = parent / "model"
            with self._patched_paths(root, ledger):
                states, contract, layout = self._claim_and_runtime(
                    root=root,
                    ledger=ledger,
                    model_dir=model_dir,
                )
                runner._advance_ledger_high_water(
                    layout,
                    event="STATE_ATTEMPT_CLAIMED",
                    state_index=0,
                )
                runner._write_json_exclusive(
                    layout.attempt_root / "000.json",
                    runner._attempt_marker(
                        projection=layout.projections[0],
                        run_contract_sha256=layout.contract_sha256,
                    ),
                )
                runtime = _Runtime()
                result = runner.run_full45_substrate(
                    artifact=SimpleNamespace(),
                    states=states,
                    runtime=runtime,
                    parse_error_class=_ParseError,
                    distance_backend=_DistanceBackend(),
                    image_decoder=lambda value: value,
                    gate=runner.Full45Gate.from_mapping(_gate_mapping()),
                    run_contract=contract,
                    output_dir=root,
                    resume=True,
                    integrity_guard=lambda: None,
                    message_builder=_message_builder,
                )
                self.assertEqual(result["outcome"], runner.INVALID_OUTCOME)
                self.assertEqual(result["attempted_state_count"], 1)
                self.assertEqual(result["completed_state_count"], 0)
                self.assertEqual(runtime.generation_calls, 0)

    def test_deleted_terminal_pair_is_detected_by_durable_high_water_without_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "run"
            ledger = parent / ".attempt.json"
            with self._patched_paths(root, ledger):
                states, contract, layout = self._claim_and_runtime(
                    root=root,
                    ledger=ledger,
                    model_dir=parent / "model",
                )
                self._persist_first_terminal(states=states, layout=layout)
                (layout.state_root / "000.json").unlink()
                (layout.attempt_root / "000.json").unlink()
                runtime = _Runtime()
                result = runner.run_full45_substrate(
                    artifact=SimpleNamespace(),
                    states=states,
                    runtime=runtime,
                    parse_error_class=_ParseError,
                    distance_backend=_DistanceBackend(),
                    image_decoder=lambda value: value,
                    gate=runner.Full45Gate.from_mapping(_gate_mapping()),
                    run_contract=contract,
                    output_dir=root,
                    resume=True,
                    integrity_guard=lambda: None,
                    message_builder=_message_builder,
                )
                self.assertEqual(result["outcome"], runner.INVALID_OUTCOME)
                self.assertEqual(result["completed_state_count"], 1)
                self.assertEqual(result["attempted_state_count"], 1)
                self.assertEqual(result["observed_inventory"]["state_file_count"], 0)
                self.assertEqual(
                    result["observed_inventory"]["attempt_marker_file_count"],
                    0,
                )
                self.assertEqual(runtime.generation_calls, 0)

    def test_deleted_state_and_attempt_directories_are_canonical_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "run"
            ledger = parent / ".attempt.json"
            with self._patched_paths(root, ledger):
                states, contract, layout = self._claim_and_runtime(
                    root=root,
                    ledger=ledger,
                    model_dir=parent / "model",
                )
                self._persist_first_terminal(states=states, layout=layout)
                shutil.rmtree(layout.state_root)
                shutil.rmtree(layout.attempt_root)
                runtime = _Runtime()
                result = runner.run_full45_substrate(
                    artifact=SimpleNamespace(),
                    states=states,
                    runtime=runtime,
                    parse_error_class=_ParseError,
                    distance_backend=_DistanceBackend(),
                    image_decoder=lambda value: value,
                    gate=runner.Full45Gate.from_mapping(_gate_mapping()),
                    run_contract=contract,
                    output_dir=root,
                    resume=True,
                    integrity_guard=lambda: None,
                    message_builder=_message_builder,
                )
                self.assertEqual(result["outcome"], runner.INVALID_OUTCOME)
                self.assertEqual(result["completed_state_count"], 1)
                self.assertEqual(result["attempted_state_count"], 1)
                self.assertEqual(runtime.generation_calls, 0)

    def test_deleted_terminal_aggregate_is_invalid_without_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "run"
            ledger = parent / ".attempt.json"
            with self._patched_paths(root, ledger):
                states, contract, layout = self._claim_and_runtime(
                    root=root,
                    ledger=ledger,
                    model_dir=parent / "model",
                )
                initial_runtime = _Runtime()
                initial = runner.run_full45_substrate(
                    artifact=SimpleNamespace(),
                    states=states,
                    runtime=initial_runtime,
                    parse_error_class=_ParseError,
                    distance_backend=_DistanceBackend(),
                    image_decoder=lambda value: value,
                    gate=runner.Full45Gate.from_mapping(_gate_mapping()),
                    run_contract=contract,
                    output_dir=root,
                    resume=False,
                    preclaimed=True,
                    integrity_guard=lambda: None,
                    message_builder=_message_builder,
                )
                self.assertEqual(initial["outcome"], runner.PASS_OUTCOME)
                self.assertEqual(initial_runtime.generation_calls, 90)
                layout.aggregate_path.unlink()

                resumed_runtime = _Runtime()
                result = runner.run_full45_substrate(
                    artifact=SimpleNamespace(),
                    states=states,
                    runtime=resumed_runtime,
                    parse_error_class=_ParseError,
                    distance_backend=_DistanceBackend(),
                    image_decoder=lambda value: value,
                    gate=runner.Full45Gate.from_mapping(_gate_mapping()),
                    run_contract=contract,
                    output_dir=root,
                    resume=True,
                    integrity_guard=lambda: None,
                    message_builder=_message_builder,
                )
                self.assertEqual(result["outcome"], runner.INVALID_OUTCOME)
                self.assertEqual(
                    result["invalid_failure"]["stage"],
                    "resume_detected_deleted_terminal_aggregate",
                )
                self.assertEqual(result["completed_state_count"], 45)
                self.assertEqual(result["attempted_state_count"], 45)
                self.assertEqual(result["observed_inventory"]["state_file_count"], 45)
                self.assertEqual(
                    result["observed_inventory"]["attempt_marker_file_count"],
                    45,
                )
                self.assertEqual(resumed_runtime.generation_calls, 0)
                self.assertEqual(resumed_runtime.teacher_calls, 0)

    def test_teacher_suppression_requires_exact_bfloat16_min(self) -> None:
        metadata = _teacher_metadata()
        runner._validate_teacher_metadata(
            metadata,
            runtime_metadata=_runtime_metadata("/model"),
        )
        metadata["teacher_standard_eos_suppression_value"] = -1.0
        with self.assertRaisesRegex(ValueError, "exact BF16"):
            runner._validate_teacher_metadata(
                metadata,
                runtime_metadata=_runtime_metadata("/model"),
            )

    def test_alternate_output_or_ledger_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "run"
            ledger = parent / ".attempt.json"
            states = _states()
            with self._patched_paths(root, ledger):
                contract = _run_contract(states, root, ledger, parent / "model")
                contract["attempt_identity"]["canonical_global_attempt_ledger"] = str(
                    parent / "other-ledger"
                )
                with self.assertRaisesRegex(ValueError, "attempt identity"):
                    runner._prepare_layout(
                        states=states,
                        run_contract=contract,
                        output_dir=root,
                    )
                contract = _run_contract(states, root, ledger, parent / "model")
                with self.assertRaisesRegex(ValueError, "alternate"):
                    runner._prepare_layout(
                        states=states,
                        run_contract=contract,
                        output_dir=parent / "other-root",
                    )

    def test_durable_claim_precedes_runtime_import_and_constructor_oom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "run"
            ledger = parent / ".attempt.json"
            model_dir = parent / "model"
            states = _states()
            contract_data = {
                "substrate_gate": _gate_mapping(),
                "data": {"derived_artifact": {"repo": "dataset"}},
                "computation_schedule": {"fixed": True},
                "promotion": {"pass_authorizes_only": "freeze"},
            }
            contract = SimpleNamespace(
                source_sha256="a" * 64,
                data=contract_data,
            )
            artifact = SimpleNamespace(
                artifact_tree_sha256="b" * 64,
                artifact_manifest_sha256="c" * 64,
                screening_manifest_sha256="d" * 64,
            )
            authorized = runner.AuthorizedFull45(
                repository_root=parent,
                contract=contract,
                processor_audit={"status": "processor"},
                pilot_authorization={"status": "pilot"},
                git_identity={"commit": "e" * 40},
                source_inventory=(),
                artifact=artifact,
                states=states,
                snapshot_manifest_path=parent / "snapshot.json",
                target_effective_visual_tokens_per_image=4096,
                policy_identity={
                    "repo": "repo",
                    "revision": "revision",
                    "snapshot_manifest_path": "snapshot.json",
                    "snapshot_manifest_sha256": "4" * 64,
                    "policy_protocol_id": POLICY_PROTOCOL,
                },
                runtime_identity={"selected_device": "cuda:0"},
                canonical_inputs={},
                attempt_identity=_attempt_identity(root, ledger),
                external_integrity={},
            )
            events: list[str] = []

            def authorize(args: Any) -> runner.AuthorizedFull45:
                del args
                events.extend(["pilot_raw_validated", "processor_raw_validated"])
                return authorized

            class OOMRuntime:
                def __init__(self, **kwargs: Any) -> None:
                    del kwargs
                    self.assert_claimed()
                    events.append("constructor")
                    raise RuntimeError("CUDA out of memory")

                @staticmethod
                def assert_claimed() -> None:
                    if not root.is_dir() or not ledger.is_file():
                        raise AssertionError("runtime constructor ran before durable claim")

            def load_runtime() -> runner.RuntimeBindings:
                self.assertEqual(
                    events[:2],
                    ["pilot_raw_validated", "processor_raw_validated"],
                )
                self.assertTrue((root / runner.RUN_MANIFEST_FILENAME).is_file())
                events.append("runtime_import")
                return runner.RuntimeBindings(
                    runtime_class=OOMRuntime,
                    parse_error_class=_ParseError,
                    kl_kernel=lambda *args: None,
                )

            args = SimpleNamespace(
                model_dir=str(model_dir),
                device="cuda:0",
                output_dir=str(root),
                resume=False,
                execution_argv=["python", "runner", "--fixed"],
            )
            with self._patched_paths(root, ledger):
                result = runner.execute_production_full45(
                    args,
                    authorization_loader=authorize,
                    integrity_revalidator=lambda value: None,
                    runtime_bindings_loader=load_runtime,
                )
            self.assertEqual(result["outcome"], runner.INVALID_OUTCOME)
            self.assertEqual(result["invalid_failure"]["category"], "OUT_OF_MEMORY")
            self.assertEqual(
                events,
                [
                    "pilot_raw_validated",
                    "processor_raw_validated",
                    "runtime_import",
                    "constructor",
                ],
            )

    def test_run_contract_schema_binds_parent_sources_policy_and_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "run"
            ledger = parent / ".attempt.json"
            states = _states()
            contract = SimpleNamespace(
                source_sha256="a" * 64,
                data={
                    "data": {"derived_artifact": {"repo": "dataset"}},
                    "computation_schedule": {"maximum": 90},
                    "promotion": {"pass_authorizes_only": "freeze"},
                },
            )
            authorized = runner.AuthorizedFull45(
                repository_root=parent,
                contract=contract,
                processor_audit={"status": "processor"},
                pilot_authorization={"status": "pilot"},
                git_identity={"commit": "e" * 40},
                source_inventory=({"path": "runner.py", "sha256": "f" * 64},),
                artifact=SimpleNamespace(
                    artifact_tree_sha256="b" * 64,
                    artifact_manifest_sha256="c" * 64,
                    screening_manifest_sha256="d" * 64,
                ),
                states=states,
                snapshot_manifest_path=parent / "snapshot.json",
                target_effective_visual_tokens_per_image=4096,
                policy_identity={
                    "repo": "repo",
                    "revision": "revision",
                    "snapshot_manifest_path": "snapshot.json",
                    "snapshot_manifest_sha256": "4" * 64,
                    "policy_protocol_id": POLICY_PROTOCOL,
                },
                runtime_identity={"selected_device": "cuda:0"},
                canonical_inputs={"pilot_evidence": {"sha256": "1" * 64}},
                attempt_identity=_attempt_identity(root, ledger),
                external_integrity={},
            )
            args = SimpleNamespace(
                model_dir=str(parent / "model"),
                device="cuda:0",
                output_dir=str(root),
                execution_argv=["python", "runner", "--resume"],
            )
            result = runner.build_production_run_contract(args, authorized)
            self.assertEqual(
                set(result),
                {
                    "schema_version",
                    "protocol_id",
                    "contract_sha256",
                    "git_identity",
                    "source_inventory",
                    "parent_authorization",
                    "canonical_inputs",
                    "artifact",
                    "policy",
                    "runtime_identity",
                    "execution_argv",
                    "operational_argv_policy",
                    "seed_policy",
                    "output_dir",
                    "attempt_identity",
                    "states",
                    "computation_schedule",
                    "prohibited_operation_counts",
                    "confirm_state",
                    "promotion",
                },
            )
            self.assertNotIn("--resume", result["execution_argv"])
            self.assertEqual(
                result["parent_authorization"],
                {
                    "pilot_artifact_validation": {"status": "pilot"},
                    "processor_audit": {"status": "processor"},
                },
            )


if __name__ == "__main__":
    unittest.main()
