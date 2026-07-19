from __future__ import annotations

import dataclasses
import json
import unittest

from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_label_producer import (
    MixedFidelityPromptPlan,
    UtilityHistoryEvent,
    UtilityQuerySpec,
    freeze_recent_candidates,
)
from causalcache.set_utility_throughput_pilot import (
    PilotReferenceGeneration,
    PilotReferenceTeacherForward,
    PilotRuntimeFailure,
    PilotRuntimePerformance,
    run_train_only_set_utility_throughput_pilot,
)


def _history_event(step_id: int) -> UtilityHistoryEvent:
    return UtilityHistoryEvent(
        event_step_id=step_id,
        low_fidelity_summary=LowFidelityEventV2(
            step_id=step_id,
            action_type="wait",
            action_argument="wait",
            foreground_app="fixture.app",
            screen_text_added=(f"screen-{step_id}",),
            screen_text_removed=(),
            screen_change="low",
            executor_result="accepted",
        ),
        high_fidelity_observation_ref=f"fixture://post-{step_id:03d}.png",
    )


def _query(*, split: str = "train") -> UtilityQuerySpec:
    history_ids = (1, 2, 3, 4, 5)
    context = freeze_recent_candidates(
        history_ids[:-1],
        processor_only_length=lambda candidates: 20 + len(candidates),
        reserved_action_tokens=8,
        context_limit=32,
    )
    return UtilityQuerySpec(
        split=split,
        trajectory_id="trajectory-pilot",
        source_id="source-pilot",
        state_id="state-pilot",
        instruction_app_group_sha256="a" * 64,
        task_instruction="Open the fixture and wait.",
        decision_step_id=6,
        history_events=tuple(_history_event(step_id) for step_id in history_ids),
        current_equivalent_event_step_id=5,
        candidate_context=context,
        maximum_labeled_cardinality=2,
        current_observation_ref="fixture://post-005.png",
        source_artifact_sha256="b" * 64,
        request_manifest_sha256="c" * 64,
        slice_witness_sha256="d" * 64,
    )


def _performance(
    latency: float,
    allocated: int,
    reserved: int,
) -> PilotRuntimePerformance:
    return PilotRuntimePerformance(
        end_to_end_wall_seconds=latency,
        full_call_cuda_peak_allocated_bytes=allocated,
        full_call_cuda_peak_reserved_bytes=reserved,
    )


class _FakeRuntime:
    def __init__(
        self,
        *,
        action_handles: tuple[object, object] = (
            "opaque-secret-action",
            "opaque-secret-action",
        ),
        fail_generation_call: int | None = None,
        projected_generation_failure_call: int | None = None,
        projected_teacher_failure_call: int | None = None,
        unexpected_generation_error: Exception | None = None,
        teacher_example_count_offset: int = 0,
    ) -> None:
        self.action_handles = action_handles
        self.fail_generation_call = fail_generation_call
        self.projected_generation_failure_call = projected_generation_failure_call
        self.projected_teacher_failure_call = projected_teacher_failure_call
        self.unexpected_generation_error = unexpected_generation_error
        self.teacher_example_count_offset = teacher_example_count_offset
        self.generation_inputs: list[object] = []
        self.teacher_calls: list[tuple[tuple[object, ...], tuple[object, ...]]] = []

    def generate_reference_action(
        self,
        reference_input: object,
    ) -> PilotReferenceGeneration | PilotRuntimeFailure:
        self.generation_inputs.append(reference_input)
        call_index = len(self.generation_inputs)
        if call_index == self.projected_generation_failure_call:
            return PilotRuntimeFailure(
                error_class_identifier="OutOfMemoryError",
                performance=_performance(9.0, 900, 1000),
            )
        if call_index == self.fail_generation_call:
            raise self.unexpected_generation_error or SensitiveRuntimeError(
                "action=opaque-secret-action tokens=[4,5] logits=[6] "
                "KL=7 utility=8"
            )
        performance = (
            _performance(0.1, 100, 200)
            if call_index == 1
            else _performance(0.2, 300, 400)
        )
        return PilotReferenceGeneration(
            action_handle=self.action_handles[call_index - 1],
            performance=performance,
        )

    def reference_actions_equal(self, left: object, right: object) -> bool:
        return left == right

    def teacher_force_reference(
        self,
        reference_inputs: tuple[object, ...],
        action_handles: tuple[object, ...],
    ) -> PilotReferenceTeacherForward | PilotRuntimeFailure:
        self.teacher_calls.append((reference_inputs, action_handles))
        if len(self.teacher_calls) == self.projected_teacher_failure_call:
            return PilotRuntimeFailure(
                error_class_identifier="CudaOutOfMemoryError",
                performance=_performance(2.5, 850, 950),
            )
        if len(reference_inputs) == 2:
            performance = _performance(0.5, 800, 900)
        elif len(self.teacher_calls) == 1:
            performance = _performance(0.3, 500, 600)
        else:
            performance = _performance(0.4, 450, 700)
        return PilotReferenceTeacherForward(
            example_count=(
                len(reference_inputs) + self.teacher_example_count_offset
            ),
            performance=performance,
        )


class SensitiveRuntimeError(RuntimeError):
    pass


class SetUtilityThroughputPilotTest(unittest.TestCase):
    def _run(
        self,
        *,
        microbatch_size: int,
        runtime: _FakeRuntime,
    ) -> tuple[dict[str, object], list[MixedFidelityPromptPlan], object]:
        plans: list[MixedFidelityPromptPlan] = []
        reference_input = object()

        def build(plan: MixedFidelityPromptPlan) -> object:
            plans.append(plan)
            return reference_input

        result = run_train_only_set_utility_throughput_pilot(
            _query(),
            reference_input_builder=build,
            runtime=runtime,
            reference_teacher_microbatch_size=microbatch_size,
        )
        return result, plans, reference_input

    def test_microbatch_one_runs_two_generation_and_two_teacher_calls(self) -> None:
        runtime = _FakeRuntime()
        result, plans, reference_input = self._run(
            microbatch_size=1,
            runtime=runtime,
        )

        self.assertEqual(len(plans), 1)
        self.assertEqual(
            plans[0].restored_event_step_ids,
            _query().candidate_event_step_ids,
        )
        self.assertEqual(runtime.generation_inputs, [reference_input, reference_input])
        self.assertEqual(len(runtime.teacher_calls), 2)
        self.assertTrue(
            all(len(inputs) == len(actions) == 1 for inputs, actions in runtime.teacher_calls)
        )
        self.assertTrue(
            all(actions == ("opaque-secret-action",) for _, actions in runtime.teacher_calls)
        )
        self.assertIsNone(result["failure_class"])
        self.assertEqual(
            result["counts"],
            {
                "reference_teacher_microbatch_size": 1,
                "reference_plan_build_count": 1,
                "reference_input_build_call_count": 1,
                "reference_input_build_completed_count": 1,
                "reference_generation_call_count": 2,
                "reference_generation_completed_count": 2,
                "reference_teacher_forward_call_count": 2,
                "reference_teacher_forward_completed_call_count": 2,
                "reference_teacher_forward_example_count": 2,
                "reference_teacher_forward_completed_example_count": 2,
            },
        )
        latency = result["latency_seconds"]
        self.assertAlmostEqual(
            latency["reference_generation_end_to_end_wall_total"], 0.3
        )
        self.assertAlmostEqual(
            latency["reference_teacher_forward_end_to_end_wall_total"], 0.7
        )
        self.assertAlmostEqual(latency["native_call_end_to_end_wall_total"], 1.0)
        self.assertEqual(
            result["peak_memory_bytes"],
            {
                "full_call_cuda_allocated_max": 500,
                "full_call_cuda_reserved_max": 700,
            },
        )
        self._assert_metric_only_payload(result)

    def test_microbatch_two_keeps_two_examples_but_uses_one_teacher_call(self) -> None:
        runtime = _FakeRuntime()
        result, _, reference_input = self._run(
            microbatch_size=2,
            runtime=runtime,
        )

        self.assertEqual(len(runtime.teacher_calls), 1)
        inputs, actions = runtime.teacher_calls[0]
        self.assertEqual(inputs, (reference_input, reference_input))
        self.assertEqual(actions, ("opaque-secret-action", "opaque-secret-action"))
        counts = result["counts"]
        self.assertEqual(counts["reference_teacher_forward_call_count"], 1)
        self.assertEqual(counts["reference_teacher_forward_completed_call_count"], 1)
        self.assertEqual(counts["reference_teacher_forward_example_count"], 2)
        self.assertEqual(counts["reference_teacher_forward_completed_example_count"], 2)
        self.assertEqual(
            result["peak_memory_bytes"],
            {
                "full_call_cuda_allocated_max": 800,
                "full_call_cuda_reserved_max": 900,
            },
        )
        self._assert_metric_only_payload(result)

    def test_action_mismatch_stops_before_teacher_without_leaking_handles(self) -> None:
        runtime = _FakeRuntime(action_handles=("secret-left", "secret-right"))
        result, _, _ = self._run(microbatch_size=2, runtime=runtime)

        self.assertEqual(result["failure_class"], "REFERENCE_ACTION_MISMATCH")
        self.assertEqual(runtime.teacher_calls, [])
        counts = result["counts"]
        self.assertEqual(counts["reference_generation_completed_count"], 2)
        self.assertEqual(counts["reference_teacher_forward_call_count"], 0)
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("secret-left", serialized)
        self.assertNotIn("secret-right", serialized)

    def test_metric_only_generation_failure_includes_failed_call_metrics(self) -> None:
        runtime = _FakeRuntime(projected_generation_failure_call=2)
        result, _, _ = self._run(microbatch_size=1, runtime=runtime)

        self.assertEqual(
            result["failure_class"],
            "REFERENCE_GENERATION:OutOfMemoryError",
        )
        counts = result["counts"]
        self.assertEqual(counts["reference_generation_call_count"], 2)
        self.assertEqual(counts["reference_generation_completed_count"], 1)
        latency = result["latency_seconds"]
        self.assertAlmostEqual(
            latency["reference_generation_end_to_end_wall_total"],
            9.1,
        )
        self.assertEqual(
            result["peak_memory_bytes"],
            {
                "full_call_cuda_allocated_max": 900,
                "full_call_cuda_reserved_max": 1000,
            },
        )
        self._assert_metric_only_payload(result)

    def test_metric_only_teacher_failure_includes_failed_call_metrics(self) -> None:
        runtime = _FakeRuntime(projected_teacher_failure_call=1)
        result, _, _ = self._run(microbatch_size=2, runtime=runtime)

        self.assertEqual(
            result["failure_class"],
            "REFERENCE_TEACHER_FORWARD:CudaOutOfMemoryError",
        )
        counts = result["counts"]
        self.assertEqual(counts["reference_teacher_forward_call_count"], 1)
        self.assertEqual(counts["reference_teacher_forward_example_count"], 2)
        self.assertEqual(counts["reference_teacher_forward_completed_call_count"], 0)
        self.assertEqual(counts["reference_teacher_forward_completed_example_count"], 0)
        latency = result["latency_seconds"]
        self.assertAlmostEqual(
            latency["reference_generation_end_to_end_wall_total"],
            0.3,
        )
        self.assertAlmostEqual(
            latency["reference_teacher_forward_end_to_end_wall_total"],
            2.5,
        )
        self.assertAlmostEqual(
            latency["native_call_end_to_end_wall_total"],
            2.8,
        )
        self.assertEqual(
            result["peak_memory_bytes"],
            {
                "full_call_cuda_allocated_max": 850,
                "full_call_cuda_reserved_max": 950,
            },
        )
        self._assert_metric_only_payload(result)

    def test_runtime_error_serializes_only_stage_and_exception_class(self) -> None:
        runtime = _FakeRuntime(fail_generation_call=2)
        result, _, _ = self._run(microbatch_size=1, runtime=runtime)

        self.assertEqual(
            result["failure_class"],
            "REFERENCE_GENERATION:SensitiveRuntimeError",
        )
        counts = result["counts"]
        self.assertEqual(counts["reference_generation_call_count"], 2)
        self.assertEqual(counts["reference_generation_completed_count"], 1)
        self.assertEqual(counts["reference_teacher_forward_call_count"], 0)
        serialized = json.dumps(result, sort_keys=True).lower()
        for forbidden in ("opaque-secret-action", "tokens", "logits", "kl=", "utility="):
            self.assertNotIn(forbidden, serialized)

    def test_unexpected_exception_with_unsafe_class_name_is_sanitized(self) -> None:
        unsafe_error_type = type(
            "Bad:action=secret tokens=4",
            (RuntimeError,),
            {},
        )
        runtime = _FakeRuntime(
            fail_generation_call=1,
            unexpected_generation_error=unsafe_error_type(
                "output_text=secret logits=secret"
            ),
        )
        result, _, _ = self._run(microbatch_size=1, runtime=runtime)

        self.assertEqual(
            result["failure_class"],
            "REFERENCE_GENERATION:UnexpectedException",
        )
        serialized = json.dumps(result, sort_keys=True).lower()
        for forbidden in ("action=secret", "tokens=4", "output_text", "logits"):
            self.assertNotIn(forbidden, serialized)

    def test_invalid_teacher_projection_is_a_metric_only_failure(self) -> None:
        runtime = _FakeRuntime(teacher_example_count_offset=1)
        result, _, _ = self._run(microbatch_size=2, runtime=runtime)

        self.assertEqual(
            result["failure_class"],
            "REFERENCE_TEACHER_FORWARD:ValueError",
        )
        counts = result["counts"]
        self.assertEqual(counts["reference_teacher_forward_call_count"], 1)
        self.assertEqual(counts["reference_teacher_forward_example_count"], 2)
        self.assertEqual(counts["reference_teacher_forward_completed_call_count"], 0)
        self.assertEqual(counts["reference_teacher_forward_completed_example_count"], 0)
        self._assert_metric_only_payload(result)

    def test_non_train_and_invalid_microbatch_fail_before_any_runtime_call(self) -> None:
        for query, microbatch_size in (
            (dataclasses.replace(_query(), split="tune"), 1),
            (_query(), 3),
        ):
            with self.subTest(split=query.split, microbatch_size=microbatch_size):
                runtime = _FakeRuntime()
                builder_calls = 0

                def build(_: MixedFidelityPromptPlan) -> object:
                    nonlocal builder_calls
                    builder_calls += 1
                    return object()

                with self.assertRaises(ValueError):
                    run_train_only_set_utility_throughput_pilot(
                        query,
                        reference_input_builder=build,
                        runtime=runtime,
                        reference_teacher_microbatch_size=microbatch_size,
                    )
                self.assertEqual(builder_calls, 0)
                self.assertEqual(runtime.generation_inputs, [])
                self.assertEqual(runtime.teacher_calls, [])

    def test_performance_projection_rejects_invalid_or_inconsistent_metrics(self) -> None:
        with self.assertRaises(ValueError):
            _performance(float("nan"), 0, 0)
        with self.assertRaises(ValueError):
            _performance(0.1, 2, 1)
        with self.assertRaises(ValueError):
            PilotRuntimePerformance(
                end_to_end_wall_seconds=True,
                full_call_cuda_peak_allocated_bytes=0,
                full_call_cuda_peak_reserved_bytes=0,
            )

    def test_runtime_failure_accepts_no_message_output_or_unsafe_identifier(self) -> None:
        performance = _performance(0.1, 2, 3)
        with self.assertRaises(ValueError):
            PilotRuntimeFailure(
                error_class_identifier="OutOfMemoryError:action=secret",
                performance=performance,
            )
        with self.assertRaises(TypeError):
            PilotRuntimeFailure(
                error_class_identifier="OutOfMemoryError",
                performance=performance,
                message="action=secret",
            )

    def _assert_metric_only_payload(self, result: dict[str, object]) -> None:
        self.assertEqual(
            set(result),
            {"latency_seconds", "peak_memory_bytes", "failure_class", "counts"},
        )
        self.assertEqual(
            set(result["latency_seconds"]),
            {
                "reference_generation_end_to_end_wall_total",
                "reference_teacher_forward_end_to_end_wall_total",
                "native_call_end_to_end_wall_total",
            },
        )
        self.assertEqual(
            set(result["peak_memory_bytes"]),
            {
                "full_call_cuda_allocated_max",
                "full_call_cuda_reserved_max",
            },
        )
        serialized = json.dumps(result, sort_keys=True).lower()
        for forbidden in (
            "opaque-secret-action",
            "output_text",
            "action_text",
            "token_ids",
            "logits",
            "kl",
            "utility",
            "message",
        ):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
