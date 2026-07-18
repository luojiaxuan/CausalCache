from __future__ import annotations

import copy
import json
import math
import unittest
from dataclasses import replace

from causalcache.gate_v1_fresh16 import (
    FEATURE_STATUS,
    FRESH_CANDIDATE_OCCURRENCE_COUNT,
    FRESH_IMAGE_COUNT,
    FRESH_STATE_COUNT,
    HEURISTIC_STATUS,
    POLICY_SCORE_STATUS,
    Fresh16ImageFeature,
    build_fresh16_image_plan,
    build_fresh16_label_blind_bundle,
    combine_policy_score_records,
    feature_states_jsonl_bytes,
    join_fresh16_evaluation_states,
    jsonl_score_records_bytes,
    label_states_jsonl_bytes,
    policy_selection_record,
    read_feature_states_jsonl,
    read_label_states_jsonl,
    read_selection_artifact,
    selection_artifact_bytes,
)
from causalcache.gate_v1_fresh16_data import (
    RAW_STATE_TRANSPORT,
    TRAJECTORY_TRANSPORT,
    Fresh16TrajectorySlice,
    decode_fresh16_labels,
    decode_fresh16_trajectories,
)
from causalcache.restoration_v2_baselines import RESIZED_RGB_BYTE_COUNT
from tests.test_gate_v1_fresh16_data import (
    SOURCE_IDS,
    SOURCE_IDS_SHA256,
    _raw_state_payload,
    _sha,
    _trajectory,
    _trajectory_payload,
    _verified,
)


def _selected_trajectories() -> list[dict[str, object]]:
    trajectories = [_trajectory(source_id) for source_id in SOURCE_IDS]
    for trajectory in trajectories:
        events = trajectory["events"]
        for event in events:
            step = int(event["step_id"])
            path = str(event["observation_after_path"])
            image_sha256 = str(event["observation_after_sha256"])
            event["high_fidelity_v2"] = {
                "image_member_path": path,
                "image_sha256": image_sha256,
            }
            event["ocr_record_refs"] = {
                "after_canonical_ocr_record_sha256": _sha(f"ocr-{path}")
            }
            event["low_fidelity_v2"] = {
                "step_id": step,
                "action_type": "wait",
                "action_argument": "",
                "foreground_app": "fixture",
                "screen_text_added": [f"token-{step}"],
                "screen_text_removed": [],
                "screen_change": "low",
                "executor_result": "accepted",
            }
        event_by_step = {int(event["step_id"]): event for event in events}
        for decision in trajectory["decisions"]:
            candidates = tuple(int(value) for value in decision["candidate_event_step_ids"])
            current_step = int(decision["current_equivalent_event_step_id"])
            decision["candidate_event_post_states"] = [
                {
                    "event_step_id": step,
                    "post_state_member_path": event_by_step[step][
                        "observation_after_path"
                    ],
                    "post_state_sha256": event_by_step[step][
                        "observation_after_sha256"
                    ],
                }
                for step in candidates
            ]
            decision["current_equivalence_witness"] = {
                "event_step_id": current_step,
                "post_state_member_path": event_by_step[current_step][
                    "observation_after_path"
                ],
                "post_state_sha256": event_by_step[current_step][
                    "observation_after_sha256"
                ],
            }
    return trajectories


def _trajectory_slice(
    selected: list[dict[str, object]] | None = None,
) -> Fresh16TrajectorySlice:
    payload = _trajectory_payload(selected or _selected_trajectories())
    return decode_fresh16_trajectories(
        _verified(payload, TRAJECTORY_TRANSPORT),
        expected_source_ids=SOURCE_IDS,
        expected_source_ids_sha256=SOURCE_IDS_SHA256,
    )


def _image_features(
    trajectories: Fresh16TrajectorySlice,
) -> dict[str, Fresh16ImageFeature]:
    _, requirements = build_fresh16_image_plan(trajectories)
    result: dict[str, Fresh16ImageFeature] = {}
    for ordinal, requirement in enumerate(requirements):
        pixel = (ordinal % 251, (ordinal * 3) % 251, (ordinal * 7) % 251)
        rgb = bytes(pixel) * (RESIZED_RGB_BYTE_COUNT // 3)
        result[requirement.image_member_path] = Fresh16ImageFeature(
            requirement=requirement,
            full_spatial_tokens=(
                "synthetic",
                requirement.image_member_path,
                str(ordinal % 5),
            ),
            resized_rgb_bytes=rgb,
        )
    return result


class GateV1Fresh16PreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectories = _trajectory_slice()
        cls.image_features = _image_features(cls.trajectories)
        cls.bundle = build_fresh16_label_blind_bundle(
            cls.trajectories,
            cls.image_features,
        )
        label_payload = _raw_state_payload()
        cls.labels = decode_fresh16_labels(
            _verified(label_payload, RAW_STATE_TRANSPORT),
            expected_source_ids=SOURCE_IDS,
            expected_source_ids_sha256=SOURCE_IDS_SHA256,
        )

    def test_image_plan_has_exact_48_80_144_geometry_and_order(self) -> None:
        work_items, requirements = build_fresh16_image_plan(self.trajectories)
        self.assertEqual(len(work_items), FRESH_STATE_COUNT)
        self.assertEqual(len(requirements), FRESH_IMAGE_COUNT)
        self.assertEqual(
            sum(len(item.event_images) for item in work_items),
            FRESH_CANDIDATE_OCCURRENCE_COUNT,
        )
        self.assertEqual(tuple(item.ordinal for item in work_items), tuple(range(48)))
        self.assertEqual(
            tuple(len(item.event_images) for item in work_items[:3]),
            (2, 3, 4),
        )
        self.assertEqual(
            tuple(item.candidate_event_step_ids for item in work_items[:3]),
            ((1, 2), (1, 2, 3), (1, 2, 3, 4)),
        )
        self.assertEqual(
            tuple(item.image_member_path for item in requirements),
            tuple(sorted(item.image_member_path for item in requirements)),
        )

    def test_label_blind_features_and_variable_n_heuristics_are_complete(self) -> None:
        bundle = self.bundle
        self.assertEqual(len(bundle.feature_states), 48)
        self.assertEqual(len(bundle.work_items), 48)
        self.assertEqual(len(bundle.image_requirements), 80)
        self.assertEqual(len(bundle.dynamic_recent), 48)
        self.assertEqual(len(bundle.ocr_rgb_v2), 48)
        self.assertEqual(len(bundle.ocr_rgb_score_records), 48)
        self.assertEqual(
            tuple(len(state.candidates) for state in bundle.feature_states[:3]),
            (2, 3, 4),
        )
        state_ids = tuple(state.state_id for state in bundle.feature_states[:3])
        self.assertEqual(
            tuple(bundle.dynamic_recent[state_id] for state_id in state_ids),
            ((1, 2), (2, 3), (3, 4)),
        )
        for state, record in zip(
            bundle.feature_states,
            bundle.ocr_rgb_score_records,
            strict=True,
        ):
            self.assertEqual(record["state_id"], state.state_id)
            self.assertEqual(
                len(record["scores_by_event_step"]),
                len(state.candidate_event_step_ids),
            )
            self.assertEqual(
                len(bundle.ocr_rgb_v2[state.state_id]),
                min(2, len(state.candidate_event_step_ids)),
            )

    def test_feature_and_selection_artifacts_round_trip_canonically(self) -> None:
        feature_payload = feature_states_jsonl_bytes(self.bundle.feature_states)
        replay = read_feature_states_jsonl(feature_payload)
        self.assertEqual(replay, self.bundle.feature_states)
        self.assertEqual(feature_states_jsonl_bytes(replay), feature_payload)
        self.assertEqual(len(feature_payload.rstrip(b"\n").split(b"\n")), 48)
        first_feature = json.loads(feature_payload.splitlines()[0])
        self.assertEqual(first_feature["status"], FEATURE_STATUS)

        label_payload = label_states_jsonl_bytes(self.labels.labels)
        replay_labels = read_label_states_jsonl(label_payload)
        self.assertEqual(replay_labels, self.labels.labels)
        self.assertEqual(label_states_jsonl_bytes(replay_labels), label_payload)

        for name, selections in (
            ("dynamic_recent", self.bundle.dynamic_recent),
            ("ocr_rgb_v2", self.bundle.ocr_rgb_v2),
        ):
            with self.subTest(name=name):
                payload = selection_artifact_bytes(name, selections)
                self.assertEqual(
                    read_selection_artifact(payload, expected_name=name), selections
                )
                self.assertEqual(json.loads(payload)["status"], HEURISTIC_STATUS)

        score_payload = jsonl_score_records_bytes(
            self.bundle.ocr_rgb_score_records
        )
        self.assertEqual(len(score_payload.rstrip(b"\n").split(b"\n")), 48)

    def test_policy_score_records_combine_by_ordinal_and_preserve_identity(self) -> None:
        records = [
            policy_selection_record(
                item,
                {
                    step: float(step)
                    for step in item.candidate_event_step_ids
                },
                worker_id=f"worker-{item.ordinal % 2}",
                device=f"cuda:{item.ordinal % 2}",
                gpu_uuid=f"GPU-SYNTHETIC-{item.ordinal % 2}",
            )
            for item in self.bundle.work_items
        ]
        ordered, selections = combine_policy_score_records(
            tuple(reversed(records)),
            self.bundle.work_items,
        )
        self.assertEqual(tuple(record["ordinal"] for record in ordered), tuple(range(48)))
        self.assertTrue(all(record["status"] == POLICY_SCORE_STATUS for record in ordered))
        first_state_ids = tuple(item.state_id for item in self.bundle.work_items[:3])
        self.assertEqual(
            tuple(selections[state_id] for state_id in first_state_ids),
            ((1, 2), (2, 3), (3, 4)),
        )
        payload = selection_artifact_bytes("policy_vision_v3", selections)
        self.assertEqual(
            read_selection_artifact(payload, expected_name="policy_vision_v3"),
            selections,
        )

    def test_image_witness_identity_and_inventory_fail_closed(self) -> None:
        selected = copy.deepcopy(_selected_trajectories())
        selected[0]["decisions"][0]["current_equivalence_witness"][
            "post_state_sha256"
        ] = "0" * 64
        drifted = _trajectory_slice(selected)
        with self.assertRaisesRegex(ValueError, "equivalence witness"):
            build_fresh16_image_plan(drifted)

        missing = dict(self.image_features)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "inventory"):
            build_fresh16_label_blind_bundle(self.trajectories, missing)

        trajectory_values = list(self.trajectories.trajectories)
        duplicated = copy.deepcopy(trajectory_values)
        duplicated[1]["events"][0] = copy.deepcopy(duplicated[0]["events"][0])
        shared_event = duplicated[1]["events"][0]
        for decision in duplicated[1]["decisions"]:
            decision["candidate_event_post_states"][0] = {
                "event_step_id": 1,
                "post_state_member_path": shared_event["observation_after_path"],
                "post_state_sha256": shared_event["observation_after_sha256"],
            }
        unsafe_slice = replace(self.trajectories, trajectories=tuple(duplicated))
        with self.assertRaisesRegex(ValueError, "denominator"):
            build_fresh16_image_plan(unsafe_slice)

    def test_policy_records_and_artifact_identity_fail_closed(self) -> None:
        records = [
            policy_selection_record(
                item,
                {step: float(step) for step in item.candidate_event_step_ids},
                worker_id="worker",
                device="cuda:0",
                gpu_uuid="GPU-SYNTHETIC",
            )
            for item in self.bundle.work_items
        ]
        inconsistent = copy.deepcopy(records)
        inconsistent[1]["scores_by_event_step"][0]["score"] = 100.0
        with self.assertRaisesRegex(ValueError, "differs from its work item"):
            combine_policy_score_records(inconsistent, self.bundle.work_items)

        records[0] = {**records[0], "state_id": "drifted"}
        with self.assertRaisesRegex(ValueError, "differs from its work item"):
            combine_policy_score_records(records, self.bundle.work_items)

        duplicate = list(records)
        duplicate[0] = {**duplicate[0], "ordinal": 1}
        with self.assertRaisesRegex(ValueError, "duplicated"):
            combine_policy_score_records(duplicate, self.bundle.work_items)

        payload = selection_artifact_bytes(
            "dynamic_recent", self.bundle.dynamic_recent
        )
        value = json.loads(payload)
        value["name"] = "ocr_rgb_v2"
        tampered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        with self.assertRaisesRegex(ValueError, "identity|replay"):
            read_selection_artifact(tampered, expected_name="dynamic_recent")

    def test_label_join_is_exact_and_rejects_identity_or_geometry_drift(self) -> None:
        joined = join_fresh16_evaluation_states(
            self.bundle.feature_states,
            self.labels,
        )
        self.assertEqual(len(joined), 48)
        self.assertEqual(
            tuple((state.source_id, state.state_id) for state in joined),
            tuple((label.source_id, label.state_id) for label in self.labels.labels),
        )

        drifted = list(self.bundle.feature_states)
        drifted[0] = replace(drifted[0], state_id="drifted-state")
        with self.assertRaisesRegex(ValueError, "join-key inventories"):
            join_fresh16_evaluation_states(drifted, self.labels)

        geometry = list(self.bundle.feature_states)
        geometry[0] = replace(
            geometry[0],
            candidate_event_step_ids=(1, 2, 3),
        )
        with self.assertRaisesRegex(ValueError, "geometry"):
            join_fresh16_evaluation_states(geometry, self.labels)

    def test_nonfinite_or_wrong_count_score_artifact_is_rejected(self) -> None:
        records = list(self.bundle.ocr_rgb_score_records)
        with self.assertRaisesRegex(ValueError, "exactly 48"):
            jsonl_score_records_bytes(records[:-1])
        changed = copy.deepcopy(records)
        changed[0]["scores_by_event_step"][0]["score"] = math.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            jsonl_score_records_bytes(changed)


if __name__ == "__main__":
    unittest.main()
