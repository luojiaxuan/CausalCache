from __future__ import annotations

import ast
import dataclasses
import inspect
import unittest
from pathlib import Path

from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import serialize_gui_owl_v2_1_teacher_target
from causalcache.set_utility_label_producer import (
    CandidateContextDoesNotFit,
    TeacherTargetRecord,
    UtilityHistoryEvent,
    UtilityQuerySpec,
    build_failed_capped_rows,
    build_mixed_fidelity_prompt_plan,
    build_successful_capped_rows,
    freeze_recent_candidates,
    prompt_plan_sha256,
    validate_capped_restoration_batch,
)
from causalcache.set_utility_label_schedule import (
    enumerate_cardinality_capped_subsets,
)


ROOT = Path(__file__).resolve().parents[2]


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


def _query(
    state_id: str,
    *,
    history_count: int,
    candidate_count: int,
    maximum_cardinality: int,
) -> UtilityQuerySpec:
    history_ids = tuple(range(1, history_count + 1))
    context = freeze_recent_candidates(
        history_ids[:-1],
        processor_only_length=lambda candidates: 50 + 10 * len(candidates),
        reserved_action_tokens=10,
        context_limit=60 + 10 * candidate_count,
    )
    if len(context.candidate_event_step_ids) != candidate_count:
        raise AssertionError("fixture candidate count drifted")
    return UtilityQuerySpec(
        split="train",
        trajectory_id=f"trajectory-{state_id}",
        source_id=f"source-{state_id}",
        state_id=state_id,
        instruction_app_group_sha256="a" * 64,
        task_instruction="Open the fixture and wait.",
        decision_step_id=history_count + 1,
        history_events=tuple(_history_event(step_id) for step_id in history_ids),
        current_equivalent_event_step_id=history_count,
        candidate_context=context,
        maximum_labeled_cardinality=maximum_cardinality,
        current_observation_ref=f"fixture://post-{history_count:03d}.png",
        source_artifact_sha256="b" * 64,
        request_manifest_sha256="c" * 64,
        slice_witness_sha256="d" * 64,
    )


def _distances(query: UtilityQuerySpec) -> dict[tuple[int, ...], float]:
    coalitions = enumerate_cardinality_capped_subsets(
        query.candidate_event_step_ids,
        maximum_cardinality=query.maximum_labeled_cardinality,
    )
    identity = (
        query.candidate_event_step_ids
        if len(query.candidate_event_step_ids)
        <= query.maximum_labeled_cardinality
        else None
    )
    return {
        coalition: float(len(coalition) + 1) / 10.0
        for coalition in coalitions
        if coalition != identity
    }


def _target(query: UtilityQuerySpec) -> TeacherTargetRecord:
    action = GUIOwlV2Action(action="wait")
    reference = build_mixed_fidelity_prompt_plan(
        query,
        query.candidate_event_step_ids,
    )
    return TeacherTargetRecord(
        state_id=query.state_id,
        reference_restored_event_step_ids=query.candidate_event_step_ids,
        first_generated_action=action,
        repeated_generated_action=action,
        serialized_teacher_target=serialize_gui_owl_v2_1_teacher_target(action),
        reference_prompt_sha256=prompt_plan_sha256(reference),
        reference_input_token_count=(
            query.candidate_context.processor_input_token_count
        ),
        action_token_count=3,
        vocabulary_size=1024,
        reference_repeat_kl=1e-7,
    )


class CandidateFreezeTest(unittest.TestCase):
    def test_uses_newest_sixteen_then_removes_only_the_oldest(self) -> None:
        calls: list[tuple[int, ...]] = []

        def processor_length(candidates: tuple[int, ...]) -> int:
            calls.append(candidates)
            return 80 + 10 * len(candidates)

        result = freeze_recent_candidates(
            tuple(range(1, 21)),
            processor_only_length=processor_length,
            reserved_action_tokens=10,
            context_limit=120,
        )
        self.assertEqual(result.initial_candidate_event_step_ids, tuple(range(5, 21)))
        self.assertEqual(result.candidate_event_step_ids, (18, 19, 20))
        self.assertEqual(result.dropped_event_step_ids, tuple(range(5, 18)))
        self.assertEqual(result.processor_input_token_count, 110)
        self.assertEqual(calls[0], tuple(range(5, 21)))
        self.assertEqual(calls[-1], (18, 19, 20))
        for previous, current in zip(calls, calls[1:], strict=False):
            self.assertEqual(current, previous[1:])

    def test_no_nonempty_suffix_fit_is_an_explicit_failure(self) -> None:
        with self.assertRaises(CandidateContextDoesNotFit) as captured:
            freeze_recent_candidates(
                (1, 2, 3),
                processor_only_length=lambda candidates: 1000 + len(candidates),
                reserved_action_tokens=10,
                context_limit=100,
            )
        self.assertEqual(
            tuple(attempt.candidate_event_step_ids for attempt in captured.exception.attempts),
            ((1, 2, 3), (2, 3), (3,)),
        )

    def test_length_contract_has_no_label_or_outcome_parameter(self) -> None:
        parameters = set(inspect.signature(freeze_recent_candidates).parameters)
        self.assertTrue(
            parameters.isdisjoint(
                {"distance", "distances", "utility", "label", "outcome", "success"}
            )
        )
        with self.assertRaises(ValueError):
            freeze_recent_candidates(
                (1,),
                processor_only_length=lambda candidates: True,
                reserved_action_tokens=1,
                context_limit=10,
            )


class MixedFidelityPlanTest(unittest.TestCase):
    def test_all_summaries_survive_but_only_coalition_candidates_get_images(self) -> None:
        query = _query(
            "state-plan",
            history_count=5,
            candidate_count=3,
            maximum_cardinality=2,
        )
        plan = build_mixed_fidelity_prompt_plan(query, (2, 4))
        self.assertEqual(plan.summary_event_step_ids, (1, 2, 3, 4, 5))
        self.assertEqual(plan.high_fidelity_history_event_step_ids, (2, 4))
        self.assertTrue(plan.current_observation_high_fidelity)
        self.assertEqual(
            plan.current_observation_ref,
            "fixture://post-005.png",
        )
        non_candidates = (plan.history_events[0], plan.history_events[-1])
        self.assertTrue(
            all(not event.is_frozen_candidate for event in non_candidates)
        )
        self.assertTrue(
            all(
                event.high_fidelity_observation_ref is None
                for event in non_candidates
            )
        )
        self.assertEqual(
            tuple(event.low_fidelity_summary.step_id for event in plan.history_events),
            (1, 2, 3, 4, 5),
        )
        self.assertFalse(plan.history_events[-1].is_frozen_candidate)
        self.assertIsNone(plan.history_events[-1].high_fidelity_observation_ref)

    def test_non_candidate_history_cannot_be_upgraded(self) -> None:
        query = _query(
            "state-plan",
            history_count=5,
            candidate_count=3,
            maximum_cardinality=2,
        )
        with self.assertRaisesRegex(ValueError, "outside the frozen candidates"):
            build_mixed_fidelity_prompt_plan(query, (1,))


class CappedRestorationValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n4 = _query(
            "state-n4",
            history_count=6,
            candidate_count=4,
            maximum_cardinality=2,
        )
        self.n2 = _query(
            "state-n2",
            history_count=3,
            candidate_count=2,
            maximum_cardinality=2,
        )
        self.n4_rows = build_successful_capped_rows(
            self.n4,
            _distances(self.n4),
        )
        self.n2_rows = build_successful_capped_rows(
            self.n2,
            _distances(self.n2),
        )

    def test_inventory_and_r_minus_identity_operation_arithmetic(self) -> None:
        result = validate_capped_restoration_batch(
            (self.n4, self.n2),
            (_target(self.n4), _target(self.n2)),
            (*self.n4_rows, *self.n2_rows),
            maximum_reference_repeat_kl=1e-6,
        )
        self.assertEqual(result.maximum_reference_repeat_kl, 1e-6)
        self.assertEqual(result.state_denominator, 2)
        self.assertEqual(result.raw_row_denominator, 15)
        self.assertEqual(result.successful_label_row_count, 15)
        self.assertEqual(result.failed_row_count, 0)
        operations = result.planned_operations
        self.assertEqual(operations.canonical_action_generations, 4)
        self.assertEqual(operations.reference_teacher_forwards, 2)
        self.assertEqual(operations.identical_reference_repeat_teacher_forwards, 2)
        self.assertEqual(operations.capped_candidate_teacher_forwards, 14)
        self.assertEqual(operations.total_teacher_forwards, 18)
        self.assertEqual(operations.kl_measurements, 16)
        self.assertEqual(operations.raw_label_rows, 15)
        self.assertEqual(operations.total_model_operations, 22)
        self.assertEqual(result.committed_success_operations, operations)

        identity = self.n2_rows[-1]
        self.assertEqual(identity.coalition_event_step_ids, (1, 2))
        self.assertEqual(identity.status, "reference_identity_zero")
        self.assertEqual(identity.distance, 0.0)
        self.assertEqual(identity.candidate_teacher_forward_count, 0)
        self.assertEqual(identity.candidate_kl_measurement_count, 0)
        self.assertTrue(all(row.status == "measured_kl" for row in self.n4_rows))

    def test_failed_state_retains_denominator_and_all_capped_row_identities(self) -> None:
        failed = build_failed_capped_rows(
            self.n4,
            failure_stage="reference_generation_repeat",
            failure_class="canonical_action_mismatch",
        )
        result = validate_capped_restoration_batch(
            (self.n4, self.n2),
            (_target(self.n2),),
            (*failed, *self.n2_rows),
            maximum_reference_repeat_kl=1e-6,
        )
        self.assertEqual(result.state_denominator, 2)
        self.assertEqual(result.successful_state_count, 1)
        self.assertEqual(result.failed_state_count, 1)
        self.assertEqual(result.failed_state_ids, ("state-n4",))
        self.assertEqual(result.raw_row_denominator, 15)
        self.assertEqual(result.failed_row_count, 11)
        self.assertEqual(result.successful_label_row_count, 4)
        committed = result.committed_success_operations
        self.assertEqual(committed.canonical_action_generations, 2)
        self.assertEqual(committed.reference_teacher_forwards, 1)
        self.assertEqual(committed.identical_reference_repeat_teacher_forwards, 1)
        self.assertEqual(committed.capped_candidate_teacher_forwards, 3)
        self.assertEqual(committed.kl_measurements, 4)
        self.assertEqual(committed.raw_label_rows, 4)

    def test_partial_state_missing_row_and_prompt_drift_fail_closed(self) -> None:
        failed = build_failed_capped_rows(
            self.n4,
            failure_stage="candidate_forward",
            failure_class="runtime_error",
        )
        partial = (*failed[:-1], self.n4_rows[-1], *self.n2_rows)
        with self.assertRaisesRegex(ValueError, "atomically"):
            validate_capped_restoration_batch(
                (self.n4, self.n2),
                (_target(self.n4), _target(self.n2)),
                partial,
                maximum_reference_repeat_kl=1e-6,
            )
        with self.assertRaisesRegex(ValueError, "exact capped row inventory"):
            validate_capped_restoration_batch(
                (self.n4, self.n2),
                (_target(self.n2),),
                (*failed[:-1], *self.n2_rows),
                maximum_reference_repeat_kl=1e-6,
            )
        corrupted = dataclasses.replace(
            self.n4_rows[0],
            prompt_plan_sha256="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "prompt differs"):
            validate_capped_restoration_batch(
                (self.n4, self.n2),
                (_target(self.n4), _target(self.n2)),
                (corrupted, *self.n4_rows[1:], *self.n2_rows),
                maximum_reference_repeat_kl=1e-6,
            )

    def test_reference_repeat_kl_uses_explicit_frozen_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen stability threshold"):
            validate_capped_restoration_batch(
                (self.n2,),
                (_target(self.n2),),
                self.n2_rows,
                maximum_reference_repeat_kl=1e-8,
            )
        with self.assertRaisesRegex(ValueError, "must be finite and non-negative"):
            validate_capped_restoration_batch(
                (self.n2,),
                (_target(self.n2),),
                self.n2_rows,
                maximum_reference_repeat_kl=float("nan"),
            )

    def test_full_candidate_identity_cannot_be_supplied_as_a_measurement(self) -> None:
        distances = _distances(self.n2)
        distances[self.n2.candidate_event_step_ids] = 0.0
        with self.assertRaisesRegex(ValueError, "must be derived"):
            build_successful_capped_rows(self.n2, distances)

    def test_teacher_target_requires_two_identical_canonical_generations(self) -> None:
        action = GUIOwlV2Action(action="wait")
        with self.assertRaisesRegex(ValueError, "two reference generations"):
            TeacherTargetRecord(
                state_id=self.n2.state_id,
                reference_restored_event_step_ids=self.n2.candidate_event_step_ids,
                first_generated_action=action,
                repeated_generated_action=GUIOwlV2Action(
                    action="system_button",
                    button="Back",
                ),
                serialized_teacher_target=serialize_gui_owl_v2_1_teacher_target(action),
                reference_prompt_sha256="1" * 64,
                reference_input_token_count=70,
                action_token_count=3,
                vocabulary_size=1024,
                reference_repeat_kl=0.0,
            )


class SourcePurityTest(unittest.TestCase):
    def test_producer_core_has_no_model_gpu_network_or_subprocess_import(self) -> None:
        path = ROOT / "code/causalcache/set_utility_label_producer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertTrue(
            imported_roots.isdisjoint(
                {
                    "torch",
                    "transformers",
                    "socket",
                    "urllib",
                    "http",
                    "subprocess",
                    "docker",
                    "cuda",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
