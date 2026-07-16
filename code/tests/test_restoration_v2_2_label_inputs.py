from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import MappingProxyType

from causalcache.data.restoration_v2_2_label_inputs import (
    build_v2_2_label_messages,
    iter_v2_2_label_prompt_specs,
    label_prompt_image_inventory,
    make_v2_2_label_prompt_spec,
)
from causalcache.data.restoration_v2_screening import (
    ScreeningState,
    ValidatedScreeningArtifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROMPT_FIXTURE = (
    REPOSITORY_ROOT / "data/fixtures/restoration_v2_prompt_low_fidelity.json"
)


def _artifact() -> ValidatedScreeningArtifact:
    manifest = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))
    states = tuple(
        ScreeningState(
            index=index,
            role="v2_label_train" if index < 30 else "v2_development",
            trajectory_id="fixture-step-6" if index < 3 else f"unused-{index // 3:02d}",
            decision_step_id=4 + index % 3,
            candidate_event_step_ids=tuple(range(1, 3 + index % 3)),
        )
        for index in range(45)
    )
    image_paths = {
        event["observation_after_path"]
        for trajectory in manifest["trajectories"]
        for event in trajectory["events"]
    } | {
        decision["current_observation_path"]
        for trajectory in manifest["trajectories"]
        for decision in trajectory["decisions"]
    }
    return ValidatedScreeningArtifact(
        artifact_root=Path("/screening"),
        artifact_tree_sha256="1" * 64,
        artifact_manifest_sha256="2" * 64,
        screening_manifest_sha256="3" * 64,
        states=states,
        validation=MappingProxyType({}),
        _screening_manifest_json=json.dumps(manifest).encode("utf-8"),
        _screening_image_payloads=MappingProxyType(
            {path: path.encode("utf-8") for path in image_paths}
        ),
    )


def _decode(payload: bytes) -> str:
    return payload.decode("utf-8")


def _images(messages: list[dict[str, object]]) -> list[object]:
    return [
        block["image"]
        for block in messages[1]["content"]
        if block["type"] == "image"
    ]


class RestorationV22LabelInputsTest(unittest.TestCase):
    def test_summary_full_and_arbitrary_coalitions(self) -> None:
        artifact = _artifact()
        state = artifact.states[2]
        cases = (
            ((), "summary_only", ["fixture://current.png"]),
            (
                (1, 3),
                "mixed_fidelity",
                [
                    "fixture://post-001.png",
                    "fixture://post-003.png",
                    "fixture://current.png",
                ],
            ),
            (
                (1, 2, 3, 4),
                "full_reference",
                [
                    "fixture://post-001.png",
                    "fixture://post-002.png",
                    "fixture://post-003.png",
                    "fixture://post-004.png",
                    "fixture://current.png",
                ],
            ),
        )
        for restored, fidelity, expected_images in cases:
            with self.subTest(restored=restored):
                spec = make_v2_2_label_prompt_spec(
                    artifact,
                    state,
                    restored_event_step_ids=restored,
                )
                messages = build_v2_2_label_messages(
                    artifact,
                    spec,
                    image_decoder=_decode,
                )
                inventory = label_prompt_image_inventory(spec, messages)
                self.assertEqual(spec.fidelity, fidelity)
                self.assertEqual(_images(messages), expected_images)
                self.assertEqual(inventory["fidelity"], fidelity)
                self.assertEqual(
                    inventory["high_fidelity_restored_event_step_ids"],
                    list(restored),
                )
                self.assertEqual(inventory["image_count"], 1 + len(restored))

    def test_full_power_set_is_deterministic_and_unfiltered_by_budget(self) -> None:
        artifact = _artifact()
        specs = list(
            iter_v2_2_label_prompt_specs(
                artifact,
                budget_event_capacity=1,
            )
        )
        self.assertEqual(len(specs), 420)
        state_specs = [spec for spec in specs if spec.state is artifact.states[2]]
        self.assertEqual(len(state_specs), 16)
        self.assertEqual([spec.coalition_mask for spec in state_specs], list(range(16)))
        self.assertEqual(state_specs[0].restored_event_step_ids, ())
        self.assertEqual(state_specs[-1].restored_event_step_ids, (1, 2, 3, 4))
        self.assertEqual(state_specs[-1].budget_event_capacity, 1)

    def test_unknown_duplicate_unsorted_and_confirm_like_inputs_are_rejected(self) -> None:
        artifact = _artifact()
        state = artifact.states[2]
        invalid_coalitions = (
            ((1, 1), "duplicate"),
            ((2, 1), "canonical sorted"),
            ((1, 5), "unknown candidate"),
        )
        for coalition, message in invalid_coalitions:
            with self.subTest(coalition=coalition), self.assertRaisesRegex(
                ValueError, message
            ):
                make_v2_2_label_prompt_spec(
                    artifact,
                    state,
                    restored_event_step_ids=coalition,
                )

        unknown = ScreeningState(
            index=2,
            role="v2_label_train",
            trajectory_id="unknown",
            decision_step_id=6,
            candidate_event_step_ids=(1, 2, 3, 4),
        )
        with self.assertRaisesRegex(ValueError, "not a member"):
            make_v2_2_label_prompt_spec(
                artifact,
                unknown,
                restored_event_step_ids=(),
            )

        confirm_like = ScreeningState(
            index=0,
            role="v2_confirm_primary",
            trajectory_id="confirm",
            decision_step_id=6,
            candidate_event_step_ids=(1, 2, 3, 4),
        )
        with self.assertRaisesRegex(ValueError, "confirm-like"):
            make_v2_2_label_prompt_spec(
                artifact,
                confirm_like,
                restored_event_step_ids=(),
            )

    def test_same_coalition_is_deterministic_and_budget_never_enters_prompt(self) -> None:
        artifact = _artifact()
        state = artifact.states[2]
        first_spec = make_v2_2_label_prompt_spec(
            artifact,
            state,
            restored_event_step_ids=(1, 4),
            budget_event_capacity=1,
        )
        second_spec = make_v2_2_label_prompt_spec(
            artifact,
            state,
            restored_event_step_ids=(1, 4),
            budget_event_capacity=3,
        )
        first = build_v2_2_label_messages(
            artifact,
            first_spec,
            image_decoder=_decode,
        )
        repeated = build_v2_2_label_messages(
            artifact,
            first_spec,
            image_decoder=_decode,
        )
        second = build_v2_2_label_messages(
            artifact,
            second_spec,
            image_decoder=_decode,
        )
        self.assertEqual(first, repeated)
        self.assertEqual(first, second)
        self.assertEqual(
            label_prompt_image_inventory(first_spec, first),
            label_prompt_image_inventory(second_spec, second),
        )


if __name__ == "__main__":
    unittest.main()
