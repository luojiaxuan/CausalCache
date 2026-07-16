from __future__ import annotations

import copy
import itertools
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import causalcache.policy.gui_owl_v2_2_vision_runtime as vision_runtime_module
import causalcache.restoration_v2_2_policy_vision as policy_vision_module
from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GPU_UUID_TYPE_PROFILE_V1,
    GPU_UUID_TYPE_PROFILE_V2,
    GUI_OWL_V2_2_VISION_RUNTIME_ID,
    GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID,
    GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID,
    IMAGE_PROCESSOR_SIZE_PROFILE_V1,
    IMAGE_PROCESSOR_SIZE_PROFILE_V3,
)
from causalcache.restoration_v2_2_label_table import (
    validate_complete_distance_table,
)
from causalcache.restoration_v2_2_ocr_rgb import COMPARATOR_METHODS
from causalcache.restoration_v2_2_policy_vision import (
    DEFAULT_IMAGE_PROCESSOR_SIZE_PROFILE,
    DEFAULT_GPU_UUID_TYPE_PROFILE,
    EXPECTED_CANDIDATE_SCORE_COUNT,
    EXPECTED_DEVELOPMENT_COUNT,
    EXPECTED_STATE_COUNT,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_UNIQUE_IMAGE_COUNT,
    OCR_RGB_METHOD,
    OUTLIER_TRAJECTORY_ID,
    PolicyVisionWorkItem,
    evaluate_feature_record,
    feature_worker_shards,
    load_identity_witness,
    merge_policy_vision_workers,
    run_policy_vision_worker,
    summarize_policy_vision_records,
    validate_feature_worker_provenance,
)
from causalcache.restoration_v2_2_policy_vision_contract import (
    EXPECTED_OPERATION_CEILING,
    PROTOCOL_ID,
)
from causalcache.restoration_v2_2_selector_geometry import (
    SelectorGeometryState,
)
from causalcache.restoration_v2_baselines import select_top_two


ROOT = Path(__file__).resolve().parents[2]
OCR_RESULT = (
    ROOT
    / "data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair"
)
OCR_PROTOCOL_ID = (
    "causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair"
)


def _identity_contract(directory: Path = OCR_RESULT) -> SimpleNamespace:
    return SimpleNamespace(
        repository_root=ROOT,
        data={
            "immutable_inputs": {
                "ocr_rgb_identity_witness": {
                    "directory": str(directory),
                    "protocol_id": OCR_PROTOCOL_ID,
                }
            }
        },
    )


def _feature_record(
    item: PolicyVisionWorkItem,
    *,
    worker_id: str,
    scores: dict[int, float] | None = None,
) -> dict[str, Any]:
    if scores is None:
        scores = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
    selection = select_top_two(scores)
    event_images = item.event_image_map()
    return {
        "state": {
            "index": item.state_index,
            "role": item.role,
            "trajectory_id": item.trajectory_id,
            "state_id": item.state_id,
        },
        "worker_id": worker_id,
        "device": "cuda:0" if worker_id == "even" else "cuda:1",
        "gpu_uuid": "GPU-even" if worker_id == "even" else "GPU-odd",
        "candidate_scores": [
            {
                "event_step_id": step,
                "post_image_member_path": event_images[step].image_member_path,
                "post_image_sha256": event_images[step].image_sha256,
                "policy_vision_cosine": scores[step],
                "image_grid_thw": [1, 2, 2],
                "merged_token_count": 1,
            }
            for step in (1, 2, 3, 4)
        ],
        "current_image_member_path": item.current_image.image_member_path,
        "current_image_sha256": item.current_image.image_sha256,
        "current_image_grid_thw": [1, 2, 2],
        "current_merged_token_count": 1,
        "ranked_event_step_ids": list(selection.ranked_event_step_ids),
        "selected_coalition": list(selection.selected_event_step_ids),
        "score_replay": {
            "absolute_tolerance": 1e-6,
            "ranking_equal": True,
            "selection_equal": True,
            "max_abs_score_difference": 0.0,
            "repeat_results": [{"repeat_index": 0}, {"repeat_index": 1}],
        },
    }


def _worker_outputs(
    work_items: tuple[PolicyVisionWorkItem, ...],
) -> list[dict[str, Any]]:
    even, odd = feature_worker_shards(work_items)
    even_rows = [_feature_record(item, worker_id="even") for item in even]
    odd_rows = [_feature_record(item, worker_id="odd") for item in odd]
    source = even_rows[0]
    sentinel = {
        "state": {
            "index": even[0].state_index,
            "state_id": even[0].state_id,
        },
        "worker_id": "odd",
        "device": "cuda:1",
        "gpu_uuid": "GPU-odd",
        "scores_by_event_step": {
            str(row["event_step_id"]): row["policy_vision_cosine"]
            for row in source["candidate_scores"]
        },
        "ranked_event_step_ids": source["ranked_event_step_ids"],
        "selected_coalition": source["selected_coalition"],
        "image_grid_thw": [[1, 2, 2]] * 5,
        "merged_token_counts": [1] * 5,
        "normalized_embedding_norm_range": {
            "minimum": 1.0,
            "maximum": 1.0,
            "maximum_absolute_deviation_from_one": 0.0,
        },
    }
    forbidden = {
        "top_model_forward_count": 0,
        "language_model_forward_count": 0,
        "lm_head_forward_count": 0,
        "generation_count": 0,
    }
    return [
        {
            "worker_id": "even",
            "device": "cuda:0",
            "gpu_uuid": "GPU-even",
            "canonical_state_records": even_rows,
            "cross_device_sentinel_record": None,
            "runtime_metadata": {"feature_only": True, "device": "cuda:0"},
            "runtime_operation_counts": {
                "image_processor_batch_count": 8,
                "policy_vision_feature_forward_count": 16,
                **forbidden,
            },
            "worker_input_counts": {
                "canonical_state_count": 8,
                "sentinel_state_count": 0,
                "image_payload_count": 40,
            },
            "peak_cuda_memory": {
                "max_memory_allocated_bytes": 1,
                "max_memory_reserved_bytes": 2,
            },
            "elapsed_seconds": 1.0,
        },
        {
            "worker_id": "odd",
            "device": "cuda:1",
            "gpu_uuid": "GPU-odd",
            "canonical_state_records": odd_rows,
            "cross_device_sentinel_record": sentinel,
            "runtime_metadata": {"feature_only": True, "device": "cuda:1"},
            "runtime_operation_counts": {
                "image_processor_batch_count": 8,
                "policy_vision_feature_forward_count": 15,
                **forbidden,
            },
            "worker_input_counts": {
                "canonical_state_count": 7,
                "sentinel_state_count": 1,
                "image_payload_count": 40,
            },
            "peak_cuda_memory": {
                "max_memory_allocated_bytes": 1,
                "max_memory_reserved_bytes": 2,
            },
            "elapsed_seconds": 1.0,
        },
    ]


def _label_state(
    item: PolicyVisionWorkItem,
    distances: dict[tuple[int, ...], float],
) -> SelectorGeometryState:
    table = validate_complete_distance_table((1, 2, 3, 4), distances)
    return SelectorGeometryState(
        member_name="workers/even/states/000.json",
        index=item.state_index,
        role=item.role,
        trajectory_id=item.trajectory_id,
        state_id=item.state_id,
        decision_step_id=6,
        candidate_event_step_ids=(1, 2, 3, 4),
        table=table,
    )


def _complete_distances() -> dict[tuple[int, ...], float]:
    distances = {
        coalition: 9.0
        for size in range(5)
        for coalition in itertools.combinations((1, 2, 3, 4), size)
    }
    distances[()] = 10.0
    distances[(1, 2)] = 12.0
    distances[(3, 4)] = 1.0
    return distances


def _synthetic_summary_record(
    item: PolicyVisionWorkItem,
    *,
    recovery: float,
) -> dict[str, Any]:
    scores = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
    baseline = 10.0
    exact_recovery = 0.8
    return {
        "state": {
            "role": item.role,
            "trajectory_id": item.trajectory_id,
            "state_id": item.state_id,
        },
        "normalized_recovery": recovery,
        "actual_utility": baseline * recovery,
        "baseline_summary_only_distance": baseline,
        "selected_distance": baseline * (1.0 - recovery),
        "exact_subset_normalized_recovery": exact_recovery,
        "exact_subset_coalition": [3, 4],
        "exact_subset_distance": baseline * (1.0 - exact_recovery),
        "exact_coalition_match": False,
        "jaccard_to_exact_coalition": 0.0,
        "exact_cardinality_coalition_match": False,
        "jaccard_to_exact_cardinality_coalition": 0.0,
        "candidate_scores": [
            {
                "event_step_id": step,
                "policy_vision_cosine": scores[step],
                "image_grid_thw": [1, 2, 2],
                "merged_token_count": 1,
            }
            for step in (1, 2, 3, 4)
        ],
        "current_image": {
            "image_grid_thw": [1, 2, 2],
            "merged_token_count": 1,
        },
        "ranked_event_step_ids": [1, 2, 3, 4],
        "selected_coalition": [1, 2],
        "ocr_rgb_comparator": {
            "selected_coalition": [1, 2],
            "normalized_recovery": 0.1,
        },
        "geometry_comparators": {
            method: {
                "selected_coalition": (
                    None
                    if method == "analytic_exact_cardinality_random"
                    else [3, 4]
                ),
                "normalized_recovery": 0.2,
            }
            for method in COMPARATOR_METHODS
        },
    }


class RestorationV22PolicyVisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.work_items, cls.witness_by_state = load_identity_witness(
            _identity_contract()
        )

    def test_real_ocr_witness_freezes_15_states_75_images_and_8_7_shards(self) -> None:
        self.assertEqual(len(self.work_items), EXPECTED_STATE_COUNT)
        self.assertEqual(
            Counter(item.role for item in self.work_items),
            {
                "v2_label_train": EXPECTED_TRAIN_COUNT,
                "v2_development": EXPECTED_DEVELOPMENT_COUNT,
            },
        )
        self.assertEqual(
            [item.state_index for item in self.work_items],
            list(range(2, 45, 3)),
        )
        identities = {
            (identity.image_member_path, identity.image_sha256)
            for item in self.work_items
            for identity in (*item.event_image_map().values(), item.current_image)
        }
        self.assertEqual(len(identities), EXPECTED_UNIQUE_IMAGE_COUNT)
        even, odd = feature_worker_shards(self.work_items)
        self.assertEqual((len(even), len(odd)), (8, 7))
        self.assertTrue(all(item.state_index % 2 == 0 for item in even))
        self.assertTrue(all(item.state_index % 2 == 1 for item in odd))

    def test_identity_witness_bad_count_identity_and_hash_fail_closed(self) -> None:
        rows = [
            json.loads(line)
            for line in (OCR_RESULT / "state_scores.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

        def assert_rejected(mutated: list[dict[str, Any]]) -> None:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                payload = b"".join(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                    for row in mutated
                )
                (directory / "state_scores.jsonl").write_bytes(payload)
                with self.assertRaises(ValueError):
                    load_identity_witness(_identity_contract(directory))

        assert_rejected(copy.deepcopy(rows[:-1]))
        duplicate = copy.deepcopy(rows)
        duplicate[1]["state"]["state_id"] = duplicate[0]["state"]["state_id"]
        assert_rejected(duplicate)
        bad_hash = copy.deepcopy(rows)
        bad_hash[0]["candidate_scores"][0]["post_image_sha256"] = "x" * 64
        assert_rejected(bad_hash)

    def test_full_distance_table_evaluation_occurs_after_top2_and_keeps_negative(self) -> None:
        item = self.work_items[0]
        witness = copy.deepcopy(self.witness_by_state[item.state_id])
        witness["baseline_summary_only_distance"] = 10.0
        witness["exact_subset_coalition"] = [3, 4]
        witness["exact_cardinality_oracle_coalition"] = [3, 4]
        feature = _feature_record(item, worker_id="even")
        record = evaluate_feature_record(
            feature_record=feature,
            work_item=item,
            witness_record=witness,
            label_state=_label_state(item, _complete_distances()),
            protocol_id=PROTOCOL_ID,
        )
        self.assertEqual(record["selected_coalition"], [1, 2])
        self.assertEqual(record["selected_distance"], 12.0)
        self.assertEqual(record["actual_utility"], -2.0)
        self.assertEqual(record["normalized_recovery"], -0.2)
        self.assertEqual(record["exact_subset_coalition"], [3, 4])
        self.assertAlmostEqual(
            record["normalized_recovery_regret_to_exact_subset"],
            1.1,
        )

    def test_frozen_near_tie_prefers_lower_event_id(self) -> None:
        item = self.work_items[0]
        scores = {1: 0.9, 2: 0.9000000001, 3: 0.8, 4: 0.7}
        feature = _feature_record(item, worker_id="even", scores=scores)
        self.assertEqual(feature["ranked_event_step_ids"], [1, 2, 3, 4])
        self.assertEqual(feature["selected_coalition"], [1, 2])

    def test_worker_explicitly_propagates_uuid_profile_without_data_leakage(
        self,
    ) -> None:
        expected_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        constructor_calls: list[dict[str, Any]] = []

        class FakeRuntime:
            def __init__(self, **kwargs: Any) -> None:
                constructor_calls.append(kwargs)
                profile = (
                    kwargs["gpu_uuid_type_profile"],
                    kwargs["image_processor_size_profile"],
                )
                runtime_profile_id = {
                    (
                        GPU_UUID_TYPE_PROFILE_V1,
                        IMAGE_PROCESSOR_SIZE_PROFILE_V1,
                    ): GUI_OWL_V2_2_VISION_RUNTIME_ID,
                    (
                        GPU_UUID_TYPE_PROFILE_V2,
                        IMAGE_PROCESSOR_SIZE_PROFILE_V1,
                    ): (
                        GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID
                    ),
                    (
                        GPU_UUID_TYPE_PROFILE_V2,
                        IMAGE_PROCESSOR_SIZE_PROFILE_V3,
                    ): GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID,
                }[profile]
                self.metadata = {
                    "gpu_uuid": expected_uuid,
                    "runtime_profile_id": runtime_profile_id,
                }
                self.operation_counts = {
                    "image_processor_batch_count": 0,
                    "policy_vision_feature_forward_count": 0,
                }
                self.device = kwargs["device"]
                self.torch = SimpleNamespace(
                    cuda=SimpleNamespace(
                        max_memory_allocated=lambda _: 1,
                        max_memory_reserved=lambda _: 2,
                    )
                )

            def score_five_images(
                self,
                images: Any,
                *,
                feature_repeats: int,
            ) -> dict[str, Any]:
                self.assert_five_images(images)
                return {
                    "scores_by_event_step": {
                        "1": 0.9,
                        "2": 0.8,
                        "3": 0.7,
                        "4": 0.6,
                    },
                    "ranked_event_step_ids": [1, 2, 3, 4],
                    "selected_event_step_ids": [1, 2],
                    "image_grid_thw": [[1, 2, 2]] * 5,
                    "merged_token_counts": [1] * 5,
                    "normalized_embedding_norm_range": {
                        "minimum": 1.0,
                        "maximum": 1.0,
                    },
                    "feature_repeats": feature_repeats,
                    "same_device_replay": {
                        "performed": feature_repeats == 2,
                        "ranking_equal": True,
                        "selection_equal": True,
                        "max_abs_score_difference": 0.0,
                    },
                    "repeat_results": [{"repeat_index": 0}],
                }

            @staticmethod
            def assert_five_images(images: Any) -> None:
                if len(images) != 5:
                    raise AssertionError("fake worker expected exactly five images")

        item = self.work_items[0]
        common = {
            "worker_id": "even",
            "device": "cuda:0",
            "expected_gpu_uuid": expected_uuid.upper(),
            "canonical_items": (item,),
            "sentinel_item": None,
            "derived_root": ROOT,
            "derived_payload_prefix": "unused",
            "expected_tar_member_count": 1,
            "model_dir": ROOT,
            "snapshot_manifest": ROOT / "unused.json",
        }
        payloads = {"one": b"payload"}
        with (
            mock.patch.object(
                vision_runtime_module,
                "GUIOwlV22VisionFeatureRuntime",
                FakeRuntime,
            ),
            mock.patch.object(
                policy_vision_module,
                "extract_worker_image_payloads",
                return_value=payloads,
            ) as extract,
            mock.patch.object(
                policy_vision_module,
                "decode_rgb_images",
                return_value=(object(),) * 5,
            ),
            mock.patch.object(policy_vision_module, "close_images"),
        ):
            default_output = run_policy_vision_worker(**common)
            v2_output = run_policy_vision_worker(
                **common,
                gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
            )
            v3_output = run_policy_vision_worker(
                **common,
                gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
                image_processor_size_profile=IMAGE_PROCESSOR_SIZE_PROFILE_V3,
            )

        self.assertEqual(DEFAULT_GPU_UUID_TYPE_PROFILE, GPU_UUID_TYPE_PROFILE_V1)
        self.assertEqual(
            DEFAULT_IMAGE_PROCESSOR_SIZE_PROFILE,
            IMAGE_PROCESSOR_SIZE_PROFILE_V1,
        )
        self.assertEqual(
            [
                (
                    call["gpu_uuid_type_profile"],
                    call["image_processor_size_profile"],
                )
                for call in constructor_calls
            ],
            [
                (GPU_UUID_TYPE_PROFILE_V1, IMAGE_PROCESSOR_SIZE_PROFILE_V1),
                (GPU_UUID_TYPE_PROFILE_V2, IMAGE_PROCESSOR_SIZE_PROFILE_V1),
                (GPU_UUID_TYPE_PROFILE_V2, IMAGE_PROCESSOR_SIZE_PROFILE_V3),
            ],
        )
        self.assertEqual(
            default_output["runtime_metadata"]["runtime_profile_id"],
            GUI_OWL_V2_2_VISION_RUNTIME_ID,
        )
        self.assertEqual(
            v2_output["runtime_metadata"]["runtime_profile_id"],
            GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID,
        )
        self.assertEqual(
            v3_output["runtime_metadata"]["runtime_profile_id"],
            GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID,
        )
        self.assertEqual(extract.call_count, 3)
        for call in extract.call_args_list:
            self.assertEqual(call.kwargs["work_items"], (item,))
            self.assertNotIn("gpu_uuid_type_profile", call.kwargs)

    def test_worker_rejects_unfrozen_profile_tuple_before_reading_payloads(
        self,
    ) -> None:
        item = self.work_items[0]
        with mock.patch.object(
            policy_vision_module,
            "extract_worker_image_payloads",
        ) as extract:
            with self.assertRaisesRegex(ValueError, "profile tuple is not frozen"):
                run_policy_vision_worker(
                    worker_id="even",
                    device="cuda:0",
                    expected_gpu_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    canonical_items=(item,),
                    sentinel_item=None,
                    derived_root=ROOT,
                    derived_payload_prefix="unused",
                    expected_tar_member_count=1,
                    model_dir=ROOT,
                    snapshot_manifest=ROOT / "unused.json",
                    gpu_uuid_type_profile="unregistered-profile",
                )
            with self.assertRaisesRegex(ValueError, "profile tuple is not frozen"):
                run_policy_vision_worker(
                    worker_id="even",
                    device="cuda:0",
                    expected_gpu_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    canonical_items=(item,),
                    sentinel_item=None,
                    derived_root=ROOT,
                    derived_payload_prefix="unused",
                    expected_tar_member_count=1,
                    model_dir=ROOT,
                    snapshot_manifest=ROOT / "unused.json",
                    gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V1,
                    image_processor_size_profile=IMAGE_PROCESSOR_SIZE_PROFILE_V3,
                )
        extract.assert_not_called()

    def test_worker_merger_accepts_exact_ledger_and_both_replay_levels(self) -> None:
        outputs = _worker_outputs(self.work_items)
        records, replay = merge_policy_vision_workers(
            worker_outputs=outputs,
            work_items=self.work_items,
            operation_ceiling=EXPECTED_OPERATION_CEILING,
        )
        self.assertEqual(len(records), EXPECTED_STATE_COUNT)
        self.assertEqual(
            [record["state"]["state_id"] for record in records],
            [item.state_id for item in self.work_items],
        )
        self.assertEqual(
            replay["same_device"],
            {
                "state_count": 15,
                "all_rankings_equal": True,
                "all_selections_equal": True,
                "maximum_absolute_score_difference": 0.0,
                "absolute_tolerance": 1e-6,
            },
        )
        self.assertEqual(
            replay["cross_device_sentinel"][
                "maximum_absolute_score_difference"
            ],
            0.0,
        )
        self.assertEqual(
            replay["runtime_operation_counts"],
            {
                "image_processor_batch_count": 16,
                "policy_vision_feature_forward_count": 31,
                "top_model_forward_count": 0,
                "language_model_forward_count": 0,
                "lm_head_forward_count": 0,
                "generation_count": 0,
            },
        )

    def test_worker_merger_bad_identity_count_and_ledger_fail_closed(self) -> None:
        cases = []
        bad_identity = _worker_outputs(self.work_items)
        bad_identity[0]["canonical_state_records"][0]["state"]["state_id"] = (
            "wrong:decision_step:006"
        )
        cases.append(bad_identity)
        bad_count = _worker_outputs(self.work_items)
        bad_count[1]["canonical_state_records"].pop()
        cases.append(bad_count)
        bad_ledger = _worker_outputs(self.work_items)
        bad_ledger[0]["runtime_operation_counts"][
            "policy_vision_feature_forward_count"
        ] -= 1
        cases.append(bad_ledger)
        for outputs in cases:
            with self.subTest():
                with self.assertRaises(ValueError):
                    merge_policy_vision_workers(
                        worker_outputs=outputs,
                        work_items=self.work_items,
                        operation_ceiling=EXPECTED_OPERATION_CEILING,
                    )

    def test_worker_merger_binds_every_row_to_top_level_worker(self) -> None:
        cases = []
        bad_state = _worker_outputs(self.work_items)
        bad_state[0]["canonical_state_records"][0]["state"]["role"] = (
            "v2_development"
        )
        cases.append(bad_state)
        bad_worker = _worker_outputs(self.work_items)
        bad_worker[0]["canonical_state_records"][0]["worker_id"] = "odd"
        cases.append(bad_worker)
        bad_device = _worker_outputs(self.work_items)
        bad_device[0]["canonical_state_records"][0]["device"] = "cuda:1"
        cases.append(bad_device)
        bad_uuid = _worker_outputs(self.work_items)
        bad_uuid[0]["canonical_state_records"][0]["gpu_uuid"] = "GPU-odd"
        cases.append(bad_uuid)
        duplicate_top_uuid = _worker_outputs(self.work_items)
        duplicate_top_uuid[1]["gpu_uuid"] = "GPU-even"
        cases.append(duplicate_top_uuid)
        for outputs in cases:
            with self.subTest():
                with self.assertRaises(ValueError):
                    merge_policy_vision_workers(
                        worker_outputs=outputs,
                        work_items=self.work_items,
                        operation_ceiling=EXPECTED_OPERATION_CEILING,
                    )

    def test_worker_merger_binds_sentinel_to_odd_top_level_worker(self) -> None:
        cases = []
        for key, value in (
            ("worker_id", "even"),
            ("device", "cuda:0"),
            ("gpu_uuid", "GPU-even"),
        ):
            outputs = _worker_outputs(self.work_items)
            outputs[1]["cross_device_sentinel_record"][key] = value
            cases.append(outputs)
        for outputs in cases:
            with self.subTest():
                with self.assertRaises(ValueError):
                    merge_policy_vision_workers(
                        worker_outputs=outputs,
                        work_items=self.work_items,
                        operation_ceiling=EXPECTED_OPERATION_CEILING,
                    )

    def test_reconstructed_rows_remain_bound_to_frozen_parity(self) -> None:
        outputs = _worker_outputs(self.work_items)
        records, _ = merge_policy_vision_workers(
            worker_outputs=outputs,
            work_items=self.work_items,
            operation_ceiling=EXPECTED_OPERATION_CEILING,
        )
        validate_feature_worker_provenance(records, self.work_items)
        for field, value in (
            ("worker_id", "odd"),
            ("device", "cuda:1"),
            ("gpu_uuid", "GPU-odd"),
        ):
            tampered = copy.deepcopy(records)
            tampered[0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_feature_worker_provenance(tampered, self.work_items)

    def test_worker_merger_bad_hash_and_same_device_replay_fail_closed(self) -> None:
        bad_hash = _worker_outputs(self.work_items)
        bad_hash[0]["canonical_state_records"][0]["candidate_scores"][0][
            "post_image_sha256"
        ] = "0" * 64
        bad_replay = _worker_outputs(self.work_items)
        bad_replay[0]["canonical_state_records"][0]["score_replay"][
            "ranking_equal"
        ] = False
        bad_tolerance = _worker_outputs(self.work_items)
        bad_tolerance[0]["canonical_state_records"][0]["score_replay"][
            "max_abs_score_difference"
        ] = 1.000001e-6
        for outputs in (bad_hash, bad_replay, bad_tolerance):
            with self.subTest():
                with self.assertRaises(ValueError):
                    merge_policy_vision_workers(
                        worker_outputs=outputs,
                        work_items=self.work_items,
                        operation_ceiling=EXPECTED_OPERATION_CEILING,
                    )

    def test_worker_merger_cross_device_identity_replay_and_tolerance_fail_closed(self) -> None:
        bad_identity = _worker_outputs(self.work_items)
        bad_identity[1]["cross_device_sentinel_record"]["state"]["index"] += 1
        bad_replay = _worker_outputs(self.work_items)
        bad_replay[1]["cross_device_sentinel_record"][
            "ranked_event_step_ids"
        ] = [2, 1, 3, 4]
        bad_tolerance = _worker_outputs(self.work_items)
        bad_tolerance[1]["cross_device_sentinel_record"][
            "scores_by_event_step"
        ]["1"] += 1.000001e-6
        for outputs in (bad_identity, bad_replay, bad_tolerance):
            with self.subTest():
                with self.assertRaises(ValueError):
                    merge_policy_vision_workers(
                        worker_outputs=outputs,
                        work_items=self.work_items,
                        operation_ceiling=EXPECTED_OPERATION_CEILING,
                    )

    def test_summary_train_dev_overall_ocr_bootstrap_and_outlier(self) -> None:
        records = []
        for item in self.work_items:
            recovery = 0.6 if item.role == "v2_label_train" else 0.4
            if item.trajectory_id == OUTLIER_TRAJECTORY_ID:
                recovery = -0.2
            records.append(_synthetic_summary_record(item, recovery=recovery))
        summary = summarize_policy_vision_records(
            records,
            operation_counts={"policy_vision_feature_forward_count": 31},
        )
        self.assertEqual(summary["state_count"], EXPECTED_STATE_COUNT)
        self.assertEqual(
            summary["candidate_comparison_count"],
            EXPECTED_CANDIDATE_SCORE_COUNT,
        )
        self.assertEqual(summary["unique_image_count"], EXPECTED_UNIQUE_IMAGE_COUNT)
        self.assertEqual(
            summary["by_role"]["v2_label_train"]["mean_normalized_recovery"],
            0.6,
        )
        self.assertAlmostEqual(
            summary["by_role"]["v2_development"]["mean_normalized_recovery"],
            0.28,
        )
        self.assertAlmostEqual(
            summary["by_role"]["overall_stratified"][
                "mean_normalized_recovery"
            ],
            7.4 / 15.0,
        )
        ocr = summary["paired_trajectory_bootstrap"][OCR_RGB_METHOD]
        self.assertAlmostEqual(ocr["train"]["mean_difference"], 0.5)
        self.assertAlmostEqual(ocr["development"]["mean_difference"], 0.18)
        self.assertEqual(ocr["train"]["win_tie_loss"]["count"], 10)
        self.assertEqual(ocr["development"]["win_tie_loss"]["count"], 5)
        self.assertEqual(
            len(summary["development_paired_deltas"][OCR_RGB_METHOD]),
            EXPECTED_DEVELOPMENT_COUNT,
        )
        outlier = summary["pre_registered_outlier"]
        self.assertEqual(outlier["trajectory_id"], OUTLIER_TRAJECTORY_ID)
        self.assertEqual(outlier["normalized_recovery"], -0.2)
        self.assertTrue(outlier["negative_recovery_preserved_without_clamp"])
        self.assertEqual(
            summary["operation_counts"],
            {"policy_vision_feature_forward_count": 31},
        )
        self.assertTrue(math.isfinite(ocr["overall_stratified"]["mean_difference"]))


if __name__ == "__main__":
    unittest.main()
