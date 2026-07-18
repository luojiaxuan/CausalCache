from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from causalcache.long_horizon_runtime import (
    DeterministicReceiptStore,
    DistanceMeasurement,
    N8_COALITIONS,
    SealedComparatorPair,
    build_long_horizon_messages,
    build_long_horizon_runtime,
    run_n8_state_once,
    run_n16_pair_once,
    run_n16_pair_with_resume,
)
from causalcache.low_fidelity_v2 import LowFidelityEventV2, serialize_low_fidelity_v2
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import (
    parse_gui_owl_v2_1_output,
    serialize_gui_owl_v2_1_teacher_target,
)


def _trajectory(*, role: str = "development") -> tuple[dict[str, Any], dict[str, bytes]]:
    images: dict[str, bytes] = {}
    events = []
    for step_id in range(1, 18):
        path = f"images/source/observation-{step_id:03d}.png"
        payload = f"image-{step_id}".encode("utf-8")
        images[path] = payload
        summary = LowFidelityEventV2(
            step_id=step_id,
            action_type="wait",
            action_argument="wait",
            foreground_app="test",
            screen_text_added=(),
            screen_text_removed=(),
            screen_change="none",
            executor_result="accepted",
        )
        serialized = serialize_low_fidelity_v2(summary)
        events.append(
            {
                "step_id": step_id,
                "observation_before_path": f"unused-before-{step_id}",
                "observation_after_path": path,
                "observation_after_sha256": hashlib.sha256(payload).hexdigest(),
                "low_fidelity_v2": summary.to_ordered_dict(),
                "low_fidelity_v2_serialized": serialized.decode("utf-8"),
                "low_fidelity_v2_sha256": hashlib.sha256(serialized).hexdigest(),
            }
        )
    decisions = []
    for step_id in (10, 18):
        current = events[step_id - 2]
        decisions.append(
            {
                "state_id": f"source:decision_step:{step_id:03d}",
                "decision_step_id": step_id,
                "history_event_step_ids": list(range(1, step_id)),
                "candidate_event_step_ids": list(range(1, step_id - 1)),
                "current_equivalent_event_step_id": step_id - 1,
                "current_observation_path": current["observation_after_path"],
                "current_observation_sha256": current["observation_after_sha256"],
                "current_expert_action_payload_included": False,
            }
        )
    return (
        {
            "source_id": "source",
            "role": role,
            "instruction": "Finish the task.",
            "content_witness_sha256": "a" * 64,
            "events": events,
            "decisions": decisions,
        },
        images,
    )


def _image_count(messages: Any) -> int:
    return sum(
        block.get("type") == "image"
        for message in messages
        for block in message["content"]
    )


class _FakeRuntime:
    def __init__(
        self,
        *,
        actions: tuple[GUIOwlV2Action, GUIOwlV2Action] | None = None,
        prompt_tokens: int | None = None,
    ) -> None:
        self.actions = actions or (
            GUIOwlV2Action(action="wait"),
            GUIOwlV2Action(action="wait"),
        )
        self.prompt_tokens = prompt_tokens
        self.prepare_calls: list[int] = []
        self.generate_calls: list[int] = []
        self.teacher_calls: list[int] = []

    def prepare_native_message_shape(self, messages: Any) -> dict[str, Any]:
        image_count = _image_count(messages)
        self.prepare_calls.append(image_count)
        tokens = self.prompt_tokens if self.prompt_tokens is not None else 100 + image_count
        return {
            "image_count": image_count,
            "prompt_input_tokens": tokens,
            "sequence_length": tokens,
            "policy_forward_executed": False,
        }

    def generate_native_action(self, messages: Any) -> Any:
        index = len(self.generate_calls)
        self.generate_calls.append(_image_count(messages))
        action = self.actions[index]
        output = serialize_gui_owl_v2_1_teacher_target(action)
        return SimpleNamespace(
            output_text=output,
            parsed_output=parse_gui_owl_v2_1_output(output),
            metadata={"fake_generation": index},
        )

    def teacher_forced_distance_logits(self, messages_batch: Any, actions: Any) -> Any:
        self.assert_single(messages_batch, actions)
        image_count = _image_count(messages_batch[0])
        self.teacher_calls.append(image_count)
        return SimpleNamespace(image_count=image_count), {"image_count": image_count}

    @staticmethod
    def assert_single(messages_batch: Any, actions: Any) -> None:
        if len(messages_batch) != 1 or len(actions) != 1:
            raise AssertionError("fake runtime requires singleton teacher forwards")


class _FakeDistanceBackend:
    def __init__(self) -> None:
        self.measure_calls = 0

    def prepare_reference(self, logits: Any) -> Any:
        return SimpleNamespace(reference_image_count=logits.image_count)

    def measure(self, reference_log_probs: Any, candidate_logits: Any) -> Any:
        self.measure_calls += 1
        return DistanceMeasurement(
            value=float(reference_log_probs.reference_image_count - candidate_logits.image_count),
            audit={"full_tensor_host_transfers": 0},
        )


class _BrokenContractDistanceBackend(_FakeDistanceBackend):
    def measure(self, reference_log_probs: Any, candidate_logits: Any) -> Any:
        del reference_log_probs, candidate_logits
        raise ValueError("injected backend contract failure")


def _pair(
    *,
    left: tuple[int, ...] = (1, 2),
    right: tuple[int, ...] = (2, 3),
    budget: int = 2,
) -> SealedComparatorPair:
    return SealedComparatorPair(
        source_id="source",
        state_id="source:decision_step:018",
        budget_event_capacity=budget,
        left_selector_name="restoration_independent_gate",
        right_selector_name="recent",
        left_selected_event_step_ids=left,
        right_selected_event_step_ids=right,
        selection_seal_sha256="b" * 64,
    )


def _assert_accounting(test: unittest.TestCase, record: dict[str, Any]) -> None:
    for operation in record["operation_counts"].values():
        test.assertEqual(
            operation["completed"] + operation["missing"],
            operation["requested"],
        )


class LongHorizonRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectory, self.images = _trajectory()
        self.loader = self.images.__getitem__
        self.decoder = lambda payload: {"payload": payload}

    def test_builder_slices_n8_history_and_uses_explicit_extended_allowlist(self) -> None:
        messages = build_long_horizon_messages(
            self.trajectory,
            decision_step_id=10,
            restored_event_step_ids=(1, 8),
            image_bytes_loader=self.loader,
            image_decoder=self.decoder,
        )
        text = "\n".join(
            block["text"]
            for message in messages
            for block in message["content"]
            if block["type"] == "text"
        )
        self.assertIn('"step_id":9', text)
        self.assertNotIn('"step_id":10', text)
        self.assertEqual(_image_count(messages), 3)

    def test_runtime_factory_binds_explicit_model_snapshot_and_device(self) -> None:
        captured = {}

        class FakeConstructedRuntime(_FakeRuntime):
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)
                super().__init__()

        runtime = build_long_horizon_runtime(
            model_dir="/model",
            expected_snapshot_manifest="/snapshot.json",
            device="cuda:2",
            runtime_class=FakeConstructedRuntime,
        )
        self.assertIsInstance(runtime, FakeConstructedRuntime)
        self.assertEqual(
            captured,
            {
                "model_dir": "/model",
                "expected_snapshot_manifest": "/snapshot.json",
                "device": "cuda:2",
            },
        )

    def test_n8_runs_exact_163_rows_and_never_forwards_full_reference_as_candidate(self) -> None:
        runtime = _FakeRuntime()
        backend = _FakeDistanceBackend()
        record = run_n8_state_once(
            trajectory=self.trajectory,
            runtime=runtime,
            distance_backend=backend,
            image_bytes_loader=self.loader,
            image_decoder=self.decoder,
        )
        self.assertTrue(record["valid"])
        self.assertEqual(record["outcome"], "VALID_LONG_HORIZON_N8_STATE")
        self.assertEqual(len(N8_COALITIONS), 163)
        self.assertEqual(len(record["distance_rows"]), 163)
        self.assertEqual(record["full_reference_distance"], 0.0)
        self.assertFalse(record["full_reference_distance_measured"])
        self.assertEqual(record["full_reference_candidate_forward_count"], 0)
        self.assertEqual(runtime.generate_calls, [9, 9])
        self.assertEqual(len(runtime.teacher_calls), 164)
        self.assertEqual(runtime.teacher_calls.count(9), 1)
        self.assertEqual(backend.measure_calls, 163)
        self.assertEqual(
            record["operation_counts"]["distance_rows"],
            {"requested": 163, "completed": 163, "missing": 0},
        )
        _assert_accounting(self, record)

    def test_context_overflow_fails_before_any_policy_forward_and_is_not_topped_up(self) -> None:
        runtime = _FakeRuntime(prompt_tokens=32_513)
        record = run_n8_state_once(
            trajectory=self.trajectory,
            runtime=runtime,
            distance_backend=_FakeDistanceBackend(),
            image_bytes_loader=self.loader,
            image_decoder=self.decoder,
        )
        self.assertFalse(record["valid"])
        self.assertEqual(record["failure"]["category"], "CONTEXT_OVERFLOW")
        self.assertEqual(runtime.generate_calls, [])
        self.assertEqual(runtime.teacher_calls, [])
        self.assertEqual(record["top_up_count"], 0)
        _assert_accounting(self, record)

    def test_context_preflight_reserves_all_256_generation_tokens(self) -> None:
        runtime = _FakeRuntime(prompt_tokens=32_512)
        record = run_n8_state_once(
            trajectory=self.trajectory,
            runtime=runtime,
            distance_backend=_FakeDistanceBackend(),
            image_bytes_loader=self.loader,
            image_decoder=self.decoder,
        )
        self.assertTrue(record["valid"])
        self.assertTrue(
            all(
                row["metadata"]["completion_token_reserve"] == 256
                for row in record["processor_preflights"]
            )
        )

    def test_canonical_action_mismatch_is_retained_invalid_without_labels(self) -> None:
        runtime = _FakeRuntime(
            actions=(
                GUIOwlV2Action(action="wait"),
                GUIOwlV2Action(action="terminate", status="success"),
            )
        )
        record = run_n8_state_once(
            trajectory=self.trajectory,
            runtime=runtime,
            distance_backend=_FakeDistanceBackend(),
            image_bytes_loader=self.loader,
            image_decoder=self.decoder,
        )
        self.assertFalse(record["valid"])
        self.assertEqual(record["failure"]["category"], "CANONICAL_ACTION_MISMATCH")
        self.assertEqual(len(record["native_generations"]), 2)
        self.assertEqual(runtime.teacher_calls, [])
        _assert_accounting(self, record)

    def test_backend_contract_value_error_is_not_mislabeled_as_scientific_invalid(self) -> None:
        runtime = _FakeRuntime()
        with self.assertRaisesRegex(ValueError, "backend contract failure"):
            run_n8_state_once(
                trajectory=self.trajectory,
                runtime=runtime,
                distance_backend=_BrokenContractDistanceBackend(),
                image_bytes_loader=self.loader,
                image_decoder=self.decoder,
            )

    def test_n16_measures_only_unique_empty_left_right_under_pair_union(self) -> None:
        runtime = _FakeRuntime()
        backend = _FakeDistanceBackend()
        record = run_n16_pair_once(
            trajectory=self.trajectory,
            pair=_pair(),
            runtime=runtime,
            distance_backend=backend,
            image_bytes_loader=self.loader,
            image_decoder=self.decoder,
        )
        self.assertTrue(record["valid"])
        self.assertEqual(record["pair_union_reference_event_step_ids"], [1, 2, 3])
        self.assertEqual(runtime.generate_calls, [4, 4])
        self.assertEqual(runtime.teacher_calls, [4, 1, 3, 3])
        self.assertEqual(backend.measure_calls, 3)
        self.assertEqual(
            [row["restored_event_step_ids"] for row in record["distance_rows"]],
            [[], [1, 2], [2, 3], [1, 2, 3]],
        )
        self.assertTrue(record["distance_rows"][-1]["analytic_self_distance"])
        self.assertEqual(record["all_16_reference_forward_count"], 0)
        self.assertEqual(record["cross_pair_table_construction_count"], 0)
        self.assertEqual(record["enumerated_subset_count"], 0)
        _assert_accounting(self, record)

    def test_n16_deduplicates_equal_arms_and_never_remeasures_reference(self) -> None:
        runtime = _FakeRuntime()
        record = run_n16_pair_once(
            trajectory=self.trajectory,
            pair=_pair(left=(1, 2), right=(1, 2)),
            runtime=runtime,
            distance_backend=_FakeDistanceBackend(),
            image_bytes_loader=self.loader,
            image_decoder=self.decoder,
        )
        self.assertTrue(record["valid"])
        self.assertEqual(runtime.teacher_calls, [3, 1])
        self.assertEqual(
            record["operation_counts"]["distance_rows"],
            {"requested": 1, "completed": 1, "missing": 0},
        )
        self.assertEqual(
            [row["restored_event_step_ids"] for row in record["distance_rows"]],
            [[], [1, 2]],
        )

    def test_n16_rejects_full_history_or_more_than_one_pair_contract(self) -> None:
        full = _pair(
            left=tuple(range(1, 9)),
            right=tuple(range(9, 17)),
            budget=4,
        )
        with self.assertRaisesRegex(PermissionError, "full-history"):
            full.validate(self.trajectory)
        duplicate_names = SealedComparatorPair(
            **{
                **_pair().__dict__,
                "right_selector_name": "restoration_independent_gate",
            }
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            duplicate_names.validate(self.trajectory)

    def test_receipt_resume_uses_deterministic_work_item_without_second_forward(self) -> None:
        runtime = _FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            store = DeterministicReceiptStore(
                Path(temporary), run_identity_sha256="c" * 64
            )
            first, resumed_first = run_n16_pair_with_resume(
                receipt_store=store,
                trajectory=self.trajectory,
                pair=_pair(),
                runtime=runtime,
                distance_backend=_FakeDistanceBackend(),
                image_bytes_loader=self.loader,
                image_decoder=self.decoder,
            )
            calls = (len(runtime.generate_calls), len(runtime.teacher_calls))
            second, resumed_second = run_n16_pair_with_resume(
                receipt_store=store,
                trajectory=self.trajectory,
                pair=_pair(),
                runtime=runtime,
                distance_backend=_FakeDistanceBackend(),
                image_bytes_loader=self.loader,
                image_decoder=self.decoder,
            )
            self.assertFalse(resumed_first)
            self.assertTrue(resumed_second)
            self.assertEqual(second, first)
            self.assertEqual((len(runtime.generate_calls), len(runtime.teacher_calls)), calls)
            self.assertEqual(len(list(Path(temporary).glob("*.json"))), 1)

    def test_reserve_trajectory_is_rejected_before_prompt_or_policy_access(self) -> None:
        reserve, images = _trajectory(role="unopened_reserve")
        runtime = _FakeRuntime()
        with self.assertRaisesRegex(PermissionError, "development"):
            run_n8_state_once(
                trajectory=reserve,
                runtime=runtime,
                distance_backend=_FakeDistanceBackend(),
                image_bytes_loader=images.__getitem__,
                image_decoder=self.decoder,
            )
        self.assertEqual(runtime.prepare_calls, [])
        self.assertEqual(runtime.generate_calls, [])


if __name__ == "__main__":
    unittest.main()
