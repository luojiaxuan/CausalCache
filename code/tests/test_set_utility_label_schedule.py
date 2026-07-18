from __future__ import annotations

import ast
import dataclasses
import unittest
from pathlib import Path

from causalcache.set_utility_label_schedule import (
    CompletedSubsetIdentity,
    LabelStateRequest,
    build_state_label_schedule,
    build_variable_n_label_schedule,
    enumerate_cardinality_capped_subsets,
    expected_subset_forward_count,
    validate_completed_subset_inventory,
    validate_variable_n_label_schedule,
)


ROOT = Path(__file__).resolve().parents[2]


def _requests() -> tuple[LabelStateRequest, ...]:
    return (
        LabelStateRequest("state-c", tuple(range(1, 7)), 3),
        LabelStateRequest("state-a", (1, 2, 3, 4), 2),
        LabelStateRequest("state-b", tuple(range(1, 9)), 4),
    )


class CardinalityCappedInventoryTest(unittest.TestCase):
    def test_subsets_use_cardinality_then_lexicographic_order(self) -> None:
        subsets = enumerate_cardinality_capped_subsets(
            (2, 4, 7, 9), maximum_cardinality=2
        )
        self.assertEqual(
            subsets,
            (
                (),
                (2,),
                (4,),
                (7,),
                (9,),
                (2, 4),
                (2, 7),
                (2, 9),
                (4, 7),
                (4, 9),
                (7, 9),
            ),
        )

    def test_expected_counts_cover_primary_and_transfer_tracks(self) -> None:
        self.assertEqual(expected_subset_forward_count(16, 2), 137)
        self.assertEqual(expected_subset_forward_count(6, 3), 42)
        self.assertEqual(expected_subset_forward_count(8, 4), 163)
        self.assertEqual(expected_subset_forward_count(1, 4), 2)

    def test_each_state_budgets_generation_reference_repeat_and_capped_rows(self) -> None:
        schedule = build_state_label_schedule(
            LabelStateRequest("state-a", (1, 2, 3, 4), 2)
        )
        self.assertEqual(len(schedule.subsets), 11)
        self.assertEqual(schedule.subsets[0].coalition_event_ids, ())
        self.assertEqual(schedule.subsets[-1].coalition_event_ids, (3, 4))
        self.assertEqual(schedule.operations.canonical_action_generations, 2)
        self.assertEqual(schedule.operations.reference_teacher_forwards, 1)
        self.assertEqual(
            schedule.operations.identical_reference_repeat_teacher_forwards, 1
        )
        self.assertEqual(schedule.operations.capped_candidate_teacher_forwards, 11)
        self.assertEqual(schedule.operations.total_teacher_forwards, 13)
        self.assertEqual(schedule.operations.kl_measurements, 12)
        self.assertEqual(schedule.operations.raw_label_rows, 11)
        self.assertEqual(schedule.operations.total_model_operations, 15)

    def test_full_candidate_inside_cap_reuses_the_reference_forward(self) -> None:
        schedule = build_state_label_schedule(
            LabelStateRequest("state-a", (1, 2), 2)
        )
        self.assertEqual(schedule.operations.raw_label_rows, 4)
        self.assertEqual(schedule.operations.capped_candidate_teacher_forwards, 3)
        self.assertEqual(schedule.operations.total_teacher_forwards, 5)
        self.assertEqual(schedule.operations.kl_measurements, 4)
        self.assertEqual(schedule.operations.total_model_operations, 7)

    def test_request_validation_rejects_identity_candidate_and_k_drift(self) -> None:
        with self.assertRaises(ValueError):
            LabelStateRequest(" state", (1, 2), 2)
        with self.assertRaises(ValueError):
            LabelStateRequest("state", (2, 1), 2)
        with self.assertRaises(ValueError):
            LabelStateRequest("state", tuple(range(1, 18)), 2)
        with self.assertRaises(ValueError):
            LabelStateRequest("state", (1, 2), 1)
        with self.assertRaises(ValueError):
            expected_subset_forward_count(0, 2)


class VariableNLabelScheduleTest(unittest.TestCase):
    def test_aggregate_counts_and_hashes_cover_the_complete_inventory(self) -> None:
        schedule = build_variable_n_label_schedule(_requests(), worker_count=2)
        self.assertEqual(
            tuple(state.request.state_id for state in schedule.states),
            ("state-a", "state-b", "state-c"),
        )
        self.assertEqual(schedule.operations.canonical_action_generations, 6)
        self.assertEqual(schedule.operations.reference_teacher_forwards, 3)
        self.assertEqual(
            schedule.operations.identical_reference_repeat_teacher_forwards, 3
        )
        self.assertEqual(schedule.operations.capped_candidate_teacher_forwards, 216)
        self.assertEqual(schedule.operations.total_teacher_forwards, 222)
        self.assertEqual(schedule.operations.kl_measurements, 219)
        self.assertEqual(schedule.operations.raw_label_rows, 216)
        self.assertEqual(schedule.operations.total_model_operations, 228)
        self.assertEqual(
            schedule.subset_count_by_cardinality,
            ((0, 3), (1, 18), (2, 49), (3, 76), (4, 70)),
        )
        self.assertEqual(
            schedule.state_count_by_candidate_count,
            ((4, 1), (6, 1), (8, 1)),
        )
        self.assertEqual(
            schedule.state_count_by_maximum_cardinality,
            ((2, 1), (3, 1), (4, 1)),
        )
        for digest in (
            schedule.subset_identity_sha256,
            schedule.inventory_sha256,
            schedule.execution_sha256,
        ):
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_deterministic_lpt_sharding_never_splits_a_state(self) -> None:
        schedule = build_variable_n_label_schedule(_requests(), worker_count=2)
        self.assertEqual(schedule.workers[0].state_ids, ("state-b",))
        self.assertEqual(schedule.workers[1].state_ids, ("state-a", "state-c"))
        assigned = [
            state_id for worker in schedule.workers for state_id in worker.state_ids
        ]
        self.assertEqual(sorted(assigned), ["state-a", "state-b", "state-c"])
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(schedule.workers[0].operations.total_model_operations, 167)
        self.assertEqual(schedule.workers[1].operations.total_model_operations, 61)
        for worker in schedule.workers:
            for state in worker.states:
                self.assertTrue(
                    all(
                        subset.state_id == state.request.state_id
                        for subset in state.subsets
                    )
                )

    def test_input_order_does_not_change_inventory_or_execution(self) -> None:
        requests = _requests()
        forward = build_variable_n_label_schedule(requests, worker_count=3)
        reverse = build_variable_n_label_schedule(
            tuple(reversed(requests)), worker_count=3
        )
        self.assertEqual(forward, reverse)

    def test_worker_count_only_changes_execution_hash(self) -> None:
        two = build_variable_n_label_schedule(_requests(), worker_count=2)
        four = build_variable_n_label_schedule(_requests(), worker_count=4)
        self.assertEqual(two.subset_identity_sha256, four.subset_identity_sha256)
        self.assertEqual(two.inventory_sha256, four.inventory_sha256)
        self.assertNotEqual(two.execution_sha256, four.execution_sha256)
        self.assertEqual(len(four.workers), 4)
        self.assertEqual(four.workers[3].state_ids, ())
        with self.assertRaises(ValueError):
            build_variable_n_label_schedule(_requests(), worker_count=5)

    def test_duplicate_state_id_and_internal_schedule_drift_fail_closed(self) -> None:
        duplicate = (
            LabelStateRequest("state-a", (1, 2), 2),
            LabelStateRequest("state-a", (1, 2, 3), 2),
        )
        with self.assertRaisesRegex(ValueError, "duplicate state"):
            build_variable_n_label_schedule(duplicate, worker_count=1)

        schedule = build_variable_n_label_schedule(_requests(), worker_count=2)
        first = schedule.states[0]
        truncated = dataclasses.replace(first, subsets=first.subsets[:-1])
        corrupted = dataclasses.replace(
            schedule, states=(truncated, *schedule.states[1:])
        )
        with self.assertRaisesRegex(ValueError, "incomplete or drifted"):
            validate_variable_n_label_schedule(corrupted)

    def test_validator_accepts_only_the_exact_rebuilt_schedule(self) -> None:
        schedule = build_variable_n_label_schedule(_requests(), worker_count=3)
        self.assertIs(validate_variable_n_label_schedule(schedule), schedule)


class CompletionInventoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = build_variable_n_label_schedule(_requests(), worker_count=3)
        self.rows = tuple(
            CompletedSubsetIdentity(
                state.request.state_id,
                subset.coalition_event_ids,
            )
            for state in self.schedule.states
            for subset in state.subsets
        )

    def test_completion_may_arrive_out_of_order_but_hashes_canonically(self) -> None:
        result = validate_completed_subset_inventory(
            self.schedule, tuple(reversed(self.rows))
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.row_count, 216)
        self.assertEqual(
            result.subset_identity_sha256,
            self.schedule.subset_identity_sha256,
        )

    def test_duplicate_missing_and_unexpected_rows_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_completed_subset_inventory(
                self.schedule, (*self.rows, self.rows[0])
            )
        with self.assertRaisesRegex(ValueError, "missing=1"):
            validate_completed_subset_inventory(self.schedule, self.rows[:-1])
        unexpected = CompletedSubsetIdentity("unknown-state", ())
        with self.assertRaisesRegex(ValueError, "unexpected=1"):
            validate_completed_subset_inventory(
                self.schedule, (*self.rows, unexpected)
            )


class SourcePurityTest(unittest.TestCase):
    def test_schedule_module_has_no_model_network_or_gpu_dependency(self) -> None:
        path = ROOT / "code/causalcache/set_utility_label_schedule.py"
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
