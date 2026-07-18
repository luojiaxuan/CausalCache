from __future__ import annotations

import copy
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from causalcache.gate_v1_data import (
    INDEPENDENT_INPUT_DIMENSION,
    feature_state_from_derived,
)
from causalcache.independent_closed_loop_features import (
    FROZEN_GATE_V1_OCR_BACKEND,
    feature_state_from_live_history,
    independent_input_from_feature_state,
    live_history_event_from_transition,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.restoration_v2_text_backend import (
    build_ocr_record,
    decode_fixture_image,
    load_backend_config,
)


ROOT = Path(__file__).resolve().parents[2]
OCR_CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"
OCR_FIXTURE_PATH = ROOT / "data/fixtures/restoration_v2_ocr_golden.json"


def _low(step_id: int) -> dict[str, object]:
    return {
        "step_id": step_id,
        "action_type": "click",
        "action_argument": f"coordinate_bin:x{step_id % 10}_y{(step_id + 1) % 10}",
        "foreground_app": "com.example.app",
        "screen_text_added": [f"added-{step_id}"],
        "screen_text_removed": [],
        "screen_change": "medium",
        "executor_result": "accepted",
    }


def _live_event(step_id: int) -> dict[str, object]:
    return {
        "low_fidelity_v2": _low(step_id),
        "post_ocr_spatial_tokens": ["Post", str(step_id)],
    }


def _raw_observations(count: int):
    fixture = json.loads(OCR_FIXTURE_PATH.read_text(encoding="utf-8"))
    payloads = tuple(
        decode_fixture_image(fixture["cases"][index % 2]) for index in range(count)
    )
    backend = load_backend_config(OCR_CONFIG_PATH)
    records = []
    for index, payload in enumerate(payloads):
        records.append(
            build_ocr_record(
                image_member_path=f"images/live/observation-{index:03d}.png",
                image_bytes=payload,
                backend_config_sha256=FROZEN_GATE_V1_OCR_BACKEND.config_sha256,
                boxes=[[[0, 0], [1, 0], [1, 1], [0, 1]]],
                texts=[f"Screen {index} repeated repeated"],
                scores=[0.75],
            )
        )
    return payloads, backend, tuple(records)


def test_raw_transition_builder_replays_guiodyssey_low_fidelity_contract() -> None:
    payloads, backend, records = _raw_observations(2)
    event = live_history_event_from_transition(
        step_id=17,
        action=GUIOwlV2Action(action="click", coordinate=(123, 456)),
        before_image_bytes=payloads[0],
        after_image_bytes=payloads[1],
        backend_config=backend,
        backend_config_sha256=FROZEN_GATE_V1_OCR_BACKEND.config_sha256,
        before_ocr_record=records[0],
        after_ocr_record=records[1],
        ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
    )

    assert set(event) == {"low_fidelity_v2", "post_ocr_spatial_tokens"}
    assert event["low_fidelity_v2"] == {
        "step_id": 17,
        "action_type": "click",
        "action_argument": "coordinate_bin:x1_y4",
        "foreground_app": "unknown",
        "screen_text_added": ["1"],
        "screen_text_removed": ["0"],
        "screen_change": "high",
        "executor_result": "unknown",
    }
    assert event["post_ocr_spatial_tokens"] == [
        "Screen",
        "1",
        "repeated",
        "repeated",
    ]


def test_raw_ordered_history_exactly_matches_derived_prefix_features() -> None:
    payloads, backend, records = _raw_observations(6)
    live_events = []
    derived_events = []
    ocr_by_path = {}
    for index, record in enumerate(records):
        ocr_by_path[f"observation-{index}.png"] = {
            "full_spatial_tokens": list(record["full_spatial_tokens"])
        }
    for step in range(1, 6):
        live_event = live_history_event_from_transition(
            step_id=step,
            action=GUIOwlV2Action(
                action="click",
                coordinate=(step * 100, step * 100),
            ),
            before_image_bytes=payloads[step - 1],
            after_image_bytes=payloads[step],
            backend_config=backend,
            backend_config_sha256=FROZEN_GATE_V1_OCR_BACKEND.config_sha256,
            before_ocr_record=records[step - 1],
            after_ocr_record=records[step],
            ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
        )
        live_events.append(live_event)
        derived_events.append(
            {
                "step_id": step,
                "observation_after_path": f"observation-{step}.png",
                "low_fidelity_v2": live_event["low_fidelity_v2"],
            }
        )
    live = feature_state_from_live_history(
        source_id="source",
        state_id="source:decision_step:006",
        decision_step_id=6,
        instruction="Open the saved item",
        current_ocr_spatial_tokens=records[5]["full_spatial_tokens"],
        history_events=live_events,
        ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
    )
    derived = feature_state_from_derived(
        {
            "source_id": "source",
            "instruction": "Open the saved item",
            "events": derived_events,
        },
        {
            "state_id": "source:decision_step:006",
            "decision_step_id": 6,
            "candidate_event_step_ids": [1, 2, 3, 4],
            "history_event_step_ids": [1, 2, 3, 4, 5],
            "current_observation_path": "observation-5.png",
        },
        ocr_records_by_path=ocr_by_path,
    )

    assert live == derived


def test_raw_transition_builder_rejects_noncanonical_ocr_record() -> None:
    payloads, backend, records = _raw_observations(2)
    changed = copy.deepcopy(records[1])
    changed["full_spatial_tokens"].append("tampered")
    with pytest.raises(ValueError, match="exact canonical rebuild"):
        live_history_event_from_transition(
            step_id=1,
            action=GUIOwlV2Action(action="wait"),
            before_image_bytes=payloads[0],
            after_image_bytes=payloads[1],
            backend_config=backend,
            backend_config_sha256=FROZEN_GATE_V1_OCR_BACKEND.config_sha256,
            before_ocr_record=records[0],
            after_ocr_record=changed,
            ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
        )


def test_raw_transition_builder_accepts_androidworld_rgb_png() -> None:
    image_module = pytest.importorskip("PIL.Image")
    payloads, backend, _ = _raw_observations(2)
    rgb_payloads = []
    for payload in payloads:
        with image_module.open(io.BytesIO(payload)) as image:
            destination = io.BytesIO()
            image.convert("RGB").save(destination, format="PNG")
        rgb_payloads.append(destination.getvalue())
    records = [
        build_ocr_record(
            image_member_path=f"images/live/rgb-{index}.png",
            image_bytes=payload,
            backend_config_sha256=FROZEN_GATE_V1_OCR_BACKEND.config_sha256,
            boxes=[[[0, 0], [1, 0], [1, 1], [0, 1]]],
            texts=[f"RGB {index}"],
            scores=[0.75],
        )
        for index, payload in enumerate(rgb_payloads)
    ]
    event = live_history_event_from_transition(
        step_id=1,
        action=GUIOwlV2Action(action="wait"),
        before_image_bytes=rgb_payloads[0],
        after_image_bytes=rgb_payloads[1],
        backend_config=backend,
        backend_config_sha256=FROZEN_GATE_V1_OCR_BACKEND.config_sha256,
        before_ocr_record=records[0],
        after_ocr_record=records[1],
        ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
    )

    assert event["low_fidelity_v2"]["screen_text_added"] == ["1"]
    assert event["low_fidelity_v2"]["screen_text_removed"] == ["0"]


def test_live_projection_exactly_replays_frozen_prefix_features() -> None:
    events = []
    ocr = {"current.png": {"full_spatial_tokens": ["Post", "5"]}}
    for step in range(1, 6):
        path = f"post-{step}.png"
        events.append(
            {
                "step_id": step,
                "observation_after_path": path,
                "low_fidelity_v2": _low(step),
            }
        )
        ocr[path] = {"full_spatial_tokens": ["Post", str(step)]}
    trajectory = {
        "source_id": "source",
        "instruction": "Open the saved item",
        "events": events,
    }
    decision = {
        "state_id": "source:decision_step:006",
        "decision_step_id": 6,
        "candidate_event_step_ids": [1, 2, 3, 4],
        "history_event_step_ids": [1, 2, 3, 4, 5],
        "current_observation_path": "current.png",
    }
    derived = feature_state_from_derived(
        trajectory,
        decision,
        ocr_records_by_path=ocr,
    )
    live = feature_state_from_live_history(
        source_id="source",
        state_id="source:decision_step:006",
        decision_step_id=6,
        instruction="Open the saved item",
        current_ocr_spatial_tokens=["Post", "5"],
        history_events=[_live_event(step) for step in range(1, 6)],
        ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
    )

    assert live == derived
    assert all(
        len(independent_input_from_feature_state(live, event_id))
        == INDEPENDENT_INPUT_DIMENSION
        for event_id in live.candidate_event_step_ids
    )


def test_live_projection_supports_long_history_and_arbitrary_ids() -> None:
    count = 137
    newest_id = 10_000 + (count - 1) * 7
    records = [
        _live_event(10_000 + chronological_index * 7)
        for chronological_index in range(count)
    ]
    state = feature_state_from_live_history(
        source_id="androidworld:task",
        state_id="androidworld:task:decision:9001",
        decision_step_id=9001,
        instruction="Create the event and save it",
        current_ocr_spatial_tokens=("Post", str(newest_id)),
        history_events=records,
        ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
    )

    assert len(state.candidates) == count - 1
    assert newest_id not in state.candidate_event_step_ids
    assert state.candidate_event_step_ids == tuple(sorted(state.candidate_event_step_ids))
    oldest = next(
        candidate for candidate in state.candidates if candidate.event_step_id == 10_000
    )
    assert oldest.g8[:2] == ((count - 1) / count, 1 / count)
    assert len(independent_input_from_feature_state(state, 10_000)) == 200


def test_live_projection_empty_and_current_equivalent_only_have_no_candidates() -> None:
    empty = feature_state_from_live_history(
        source_id="source",
        state_id="state-1",
        decision_step_id=1,
        instruction="Wait",
        current_ocr_spatial_tokens=(),
        history_events=(),
        ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
    )
    current_only = feature_state_from_live_history(
        source_id="source",
        state_id="state-2",
        decision_step_id=2,
        instruction="Wait",
        current_ocr_spatial_tokens=("Post", "987"),
        history_events=(_live_event(987),),
        ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
    )

    assert empty.candidates == ()
    assert current_only.candidates == ()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].update({"age": 0}), "field inventory"),
        (
            lambda rows: rows[0]["low_fidelity_v2"].update({"step_id": 22}),
            "step ids must be unique",
        ),
        (
            lambda rows: rows[0]["low_fidelity_v2"].update({"screen_change": "huge"}),
            "not canonical",
        ),
        (
            lambda rows: rows[0].update({"post_ocr_spatial_tokens": ["two words"]}),
            "exact canonical OCR",
        ),
    ],
)
def test_live_projection_rejects_contract_drift(mutate, message: str) -> None:
    rows = [_live_event(11), _live_event(22)]
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        feature_state_from_live_history(
            source_id="source",
            state_id="state",
            decision_step_id=99,
            instruction="Do the task",
            current_ocr_spatial_tokens=["Current"],
            history_events=copy.deepcopy(rows),
            ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
        )


def test_live_projection_requires_exact_frozen_ocr_backend_identity() -> None:
    with pytest.raises(ValueError, match="OCR backend identity drifted"):
        feature_state_from_live_history(
            source_id="source",
            state_id="state",
            decision_step_id=2,
            instruction="Do the task",
            current_ocr_spatial_tokens=["Current"],
            history_events=[_live_event(1)],
            ocr_backend_binding=replace(
                FROZEN_GATE_V1_OCR_BACKEND,
                model_revision="f" * 40,
            ),
        )


def test_live_projection_preserves_uncapped_ocr_token_multiplicity() -> None:
    once = feature_state_from_live_history(
        source_id="source",
        state_id="state",
        decision_step_id=2,
        instruction="Do the task",
        current_ocr_spatial_tokens=["Alpha", "Beta"],
        history_events=[],
        ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
    )
    repeated = feature_state_from_live_history(
        source_id="source",
        state_id="state",
        decision_step_id=2,
        instruction="Do the task",
        current_ocr_spatial_tokens=["Alpha", "Beta", "Beta"],
        history_events=[],
        ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
    )

    assert once.q64 != repeated.q64
