from __future__ import annotations

import inspect
import json
import unittest

from causalcache.restoration_v2_batching import (
    FROZEN_COALITION_MICROBATCH_SIZE,
    CoalitionMicrobatchRecord,
    plan_coalition_microbatches,
)


def _record(
    coalition_id: str,
    image_count: int,
    sequence_length: int,
    input_index: int,
) -> CoalitionMicrobatchRecord:
    return CoalitionMicrobatchRecord(
        coalition_id=coalition_id,
        image_count=image_count,
        sequence_length=sequence_length,
        input_index=input_index,
    )


class RestorationV2MicrobatchPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = (
            _record("state-a:0", 2, 100, 0),
            _record("state-a:1", 1, 120, 1),
            _record("state-a:2", 2, 100, 2),
            _record("state-a:3", 1, 120, 3),
            _record("state-a:4", 1, 100, 4),
            _record("state-a:5", 2, 100, 5),
        )

    def test_groups_exact_shapes_in_sorted_group_order(self) -> None:
        plan = plan_coalition_microbatches(
            self.records,
            microbatch_size=FROZEN_COALITION_MICROBATCH_SIZE,
        )
        observed = [
            (
                batch.microbatch_index,
                batch.group_index,
                batch.image_count,
                batch.sequence_length,
                batch.coalition_ids,
                batch.input_indices,
            )
            for batch in plan.microbatches
        ]
        self.assertEqual(
            observed,
            [
                (0, 0, 1, 100, ("state-a:4",), (4,)),
                (1, 1, 1, 120, ("state-a:1", "state-a:3"), (1, 3)),
                (2, 2, 2, 100, ("state-a:0", "state-a:2"), (0, 2)),
                (3, 2, 2, 100, ("state-a:5",), (5,)),
            ],
        )
        for batch in plan.microbatches:
            self.assertLessEqual(len(batch.records), 2)
            self.assertTrue(
                all(
                    (record.image_count, record.sequence_length)
                    == (batch.image_count, batch.sequence_length)
                    for record in batch.records
                )
            )

    def test_input_index_makes_iterable_traversal_irrelevant(self) -> None:
        forward = plan_coalition_microbatches(self.records, microbatch_size=2)
        shuffled = plan_coalition_microbatches(
            (self.records[index] for index in (5, 2, 0, 4, 3, 1)),
            microbatch_size=2,
        )
        self.assertEqual(shuffled, forward)

    def test_audit_metadata_is_json_ready_and_records_no_oom_fallback(self) -> None:
        plan = plan_coalition_microbatches(self.records, microbatch_size=2)
        audit = plan.audit.as_dict()
        self.assertEqual(audit["schema_version"], "0.1.0")
        self.assertEqual(audit["microbatch_size"], 2)
        self.assertEqual(audit["microbatch_size_source"], "explicit_argument")
        self.assertIs(audit["automatic_oom_fallback"], False)
        self.assertEqual(audit["grouping_fields"], ["image_count", "sequence_length"])
        self.assertEqual(audit["input_record_count"], 6)
        self.assertEqual(audit["group_count"], 3)
        self.assertEqual(audit["microbatch_count"], 4)
        self.assertEqual(audit["groups"][2]["microbatch_sizes"], [2, 1])
        self.assertEqual(audit["groups"][2]["microbatch_indices"], [2, 3])
        json.dumps(audit, sort_keys=True)

    def test_microbatch_size_is_explicit_keyword_only_and_frozen_to_two(self) -> None:
        parameter = inspect.signature(plan_coalition_microbatches).parameters[
            "microbatch_size"
        ]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            plan_coalition_microbatches(self.records)
        with self.assertRaisesRegex(ValueError, "frozen restoration-v2 value 2"):
            plan_coalition_microbatches(self.records, microbatch_size=1)
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            plan_coalition_microbatches(self.records, microbatch_size=True)

    def test_invalid_record_fields_fail_closed(self) -> None:
        invalid = (
            ({"coalition_id": 1}, TypeError, "coalition_id"),
            ({"coalition_id": "  "}, ValueError, "not be empty"),
            ({"image_count": 0}, ValueError, "image_count"),
            ({"image_count": True}, TypeError, "image_count"),
            ({"sequence_length": 0}, ValueError, "sequence_length"),
            ({"input_index": -1}, ValueError, "input_index"),
            ({"input_index": False}, TypeError, "input_index"),
        )
        base = {
            "coalition_id": "state-a:0",
            "image_count": 1,
            "sequence_length": 10,
            "input_index": 0,
        }
        for override, error_type, pattern in invalid:
            with self.subTest(override=override):
                values = {**base, **override}
                with self.assertRaisesRegex(error_type, pattern):
                    CoalitionMicrobatchRecord(**values)

    def test_invalid_collections_and_duplicate_identity_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            plan_coalition_microbatches((), microbatch_size=2)
        with self.assertRaisesRegex(TypeError, "contain only"):
            plan_coalition_microbatches([{"coalition_id": "x"}], microbatch_size=2)
        with self.assertRaisesRegex(ValueError, "coalition_id values must be unique"):
            plan_coalition_microbatches(
                (_record("same", 1, 10, 0), _record("same", 1, 10, 1)),
                microbatch_size=2,
            )
        with self.assertRaisesRegex(ValueError, "input_index values must be unique"):
            plan_coalition_microbatches(
                (_record("a", 1, 10, 0), _record("b", 1, 10, 0)),
                microbatch_size=2,
            )


if __name__ == "__main__":
    unittest.main()
