from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from causalcache.data.guiodyssey_restoration_v2 import (
    build_image_tar_bytes,
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from causalcache.independent_confirm_data import (
    CONFIRM_CANDIDATE_EVENT_STEP_IDS,
    CONFIRM_IMAGE_COUNT,
    CONFIRM_SOURCE_IDS,
    CONFIRM_SOURCE_IDS_SHA256,
    CONFIRM_STATE_COUNT,
    Confirm20TransportIdentity,
    build_confirm20_label_blind_bundle,
    validate_confirm20_payloads,
)
from causalcache.restoration_v2_text_backend import (
    build_ocr_record,
    decode_fixture_image,
    load_backend_config,
)


ROOT = Path(__file__).resolve().parents[2]
BACKEND_CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"
OCR_FIXTURE_PATH = ROOT / "data/fixtures/restoration_v2_ocr_golden.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decision(source_id: str, decision_step: int) -> dict:
    current_step = decision_step - 1
    current_path = f"images/{source_id}/observation-{current_step:03d}.png"
    validated_action = {
        "action_type": "tap",
        "target": "coordinate_bin:x1_y1",
        "text_argument": None,
        "text_case_sensitive": True,
    }
    return {
        "state_id": f"{source_id}:decision_step:{decision_step:03d}",
        "decision_step_id": decision_step,
        "history_event_step_ids": list(range(1, decision_step)),
        "candidate_event_step_ids": list(range(1, current_step)),
        "current_equivalent_event_step_id": current_step,
        "current_observation_path": current_path,
        "current_observation_sha256": "",
        "validated_action": validated_action,
        "validated_action_sha256": _sha256(canonical_json_bytes(validated_action)),
        "validation_source": "synthetic_fixture",
    }


def _fixture_payloads() -> dict:
    backend = load_backend_config(BACKEND_CONFIG_PATH)
    backend_payload = BACKEND_CONFIG_PATH.read_bytes()
    backend_sha256 = _sha256(backend_payload)
    golden = json.loads(OCR_FIXTURE_PATH.read_text(encoding="utf-8"))
    image_payload = decode_fixture_image(golden["cases"][0])
    alternate_image_payload = decode_fixture_image(golden["cases"][1])
    train_ids = tuple(f"train-{index:02d}" for index in range(10))
    development_ids = tuple(f"development-{index:02d}" for index in range(5))
    roster = (
        *((source_id, "v2_label_train") for source_id in train_ids),
        *((source_id, "v2_development") for source_id in development_ids),
        *((source_id, "v2_confirm_primary") for source_id in CONFIRM_SOURCE_IDS),
    )
    image_payloads: dict[str, bytes] = {}
    ocr_records: dict[str, dict] = {}
    for source_id, _ in roster:
        for observation in range(6):
            path = f"images/{source_id}/observation-{observation:03d}.png"
            image_payloads[path] = image_payload
            ocr_records[path] = build_ocr_record(
                image_member_path=path,
                image_bytes=image_payload,
                backend_config_sha256=backend_sha256,
                boxes=[[[0, 0], [1, 0], [1, 1], [0, 1]]],
                texts=["same fixture token"],
                scores=[0.75],
            )
    trajectories = []
    for source_id, role in roster:
        events = []
        for step in range(1, 6):
            before_path = f"images/{source_id}/observation-{step - 1:03d}.png"
            after_path = f"images/{source_id}/observation-{step:03d}.png"
            low_fidelity = {
                "step_id": step,
                "action_type": "click",
                "action_argument": f"coordinate_bin:x{step}_y{step}",
                "foreground_app": "fixture",
                "screen_text_added": [],
                "screen_text_removed": [],
                "screen_change": "none",
                "executor_result": "accepted",
            }
            events.append(
                {
                    "step_id": step,
                    "observation_before_path": before_path,
                    "observation_before_sha256": _sha256(image_payload),
                    "observation_after_path": after_path,
                    "observation_after_sha256": _sha256(image_payload),
                    "executed_action": {"action_type": "tap"},
                    "source_tool_call": {"function": {"name": "tap"}},
                    "low_fidelity_v2": low_fidelity,
                    "low_fidelity_v2_serialized": "fixture\n",
                    "low_fidelity_v2_sha256": _sha256(b"fixture\n"),
                    "low_fidelity_v2_metadata": {},
                    "high_fidelity_v2": {
                        "content_type": "image",
                        "image_role": "post_action_state",
                        "image_member_path": after_path,
                        "image_sha256": _sha256(image_payload),
                        "include_before_image": False,
                        "include_additional_action_text": False,
                    },
                    "ocr_record_refs": {
                        "before_canonical_ocr_record_sha256": ocr_records[before_path][
                            "canonical_ocr_record_sha256"
                        ],
                        "after_canonical_ocr_record_sha256": ocr_records[after_path][
                            "canonical_ocr_record_sha256"
                        ],
                    },
                }
            )
        decision_steps = (4, 5, 6) if role != "v2_confirm_primary" else (6,)
        decisions = [_decision(source_id, step) for step in decision_steps]
        for decision in decisions:
            decision["current_observation_sha256"] = _sha256(image_payload)
        instruction = f"Fixture instruction for {source_id}"
        trajectories.append(
            {
                "source_id": source_id,
                "role": role,
                "instruction": instruction,
                "instruction_sha256": _sha256(instruction.encode("utf-8")),
                "platform": "android",
                "apps": ["fixture"],
                "device_name": "fixture-device",
                "resolution": [16, 16],
                "terminal_status": "success",
                "source": {"fixture": True},
                "selection": {"source_id": source_id},
                "events": events,
                "decisions": decisions,
            }
        )
    trajectory_payload = canonical_jsonl_bytes(trajectories)
    ocr_payload = canonical_jsonl_bytes(
        [ocr_records[path] for path in sorted(ocr_records)]
    )
    image_tar_payload, members = build_image_tar_bytes(image_payloads)
    assert len(members) == 210
    return {
        "backend": backend,
        "backend_sha256": backend_sha256,
        "trajectories": trajectories,
        "ocr_records": ocr_records,
        "image_payloads": image_payloads,
        "trajectory_payload": trajectory_payload,
        "ocr_payload": ocr_payload,
        "image_tar_payload": image_tar_payload,
        "alternate_image_payload": alternate_image_payload,
    }


@pytest.fixture(scope="module")
def confirm_fixture() -> dict:
    return _fixture_payloads()


def _identity(
    fixture: dict,
    *,
    trajectory_payload: bytes | None = None,
    ocr_payload: bytes | None = None,
    image_tar_payload: bytes | None = None,
) -> Confirm20TransportIdentity:
    trajectory = trajectory_payload or fixture["trajectory_payload"]
    ocr = ocr_payload or fixture["ocr_payload"]
    images = image_tar_payload or fixture["image_tar_payload"]
    return Confirm20TransportIdentity(
        artifact_tree_sha256="f" * 64,
        trajectory_sha256=_sha256(trajectory),
        trajectory_size_bytes=len(trajectory),
        trajectory_record_count=35,
        ocr_sha256=_sha256(ocr),
        ocr_size_bytes=len(ocr),
        ocr_record_count=210,
        image_tar_sha256=_sha256(images),
        image_tar_size_bytes=len(images),
        image_member_count=210,
    )


def _validate(fixture: dict, **overrides):
    trajectory_payload = overrides.get(
        "trajectory_payload", fixture["trajectory_payload"]
    )
    ocr_payload = overrides.get("ocr_payload", fixture["ocr_payload"])
    image_tar_payload = overrides.get(
        "image_tar_payload", fixture["image_tar_payload"]
    )
    return validate_confirm20_payloads(
        artifact_tree_sha256="f" * 64,
        trajectory_payload=trajectory_payload,
        ocr_payload=ocr_payload,
        image_tar_payload=image_tar_payload,
        backend_config=fixture["backend"],
        backend_config_sha256=fixture["backend_sha256"],
        transport_identity=overrides.get(
            "transport_identity",
            _identity(
                fixture,
                trajectory_payload=trajectory_payload,
                ocr_payload=ocr_payload,
                image_tar_payload=image_tar_payload,
            ),
        ),
    )


def test_confirm20_projection_builds_only_frozen_label_blind_inputs(
    confirm_fixture: dict,
) -> None:
    assert _sha256(canonical_json_bytes(list(CONFIRM_SOURCE_IDS))) == (
        CONFIRM_SOURCE_IDS_SHA256
    )
    validated = _validate(confirm_fixture)
    bundle = build_confirm20_label_blind_bundle(validated)

    assert validated.source_ids == CONFIRM_SOURCE_IDS
    assert len(validated.trajectories) == CONFIRM_STATE_COUNT
    assert len(validated.image_requirements) == CONFIRM_IMAGE_COUNT
    assert len(validated.ocr_records_by_path) == CONFIRM_IMAGE_COUNT
    assert len(validated.image_payloads_by_path) == CONFIRM_IMAGE_COUNT
    assert all(
        any(path.startswith(f"images/{source_id}/") for source_id in CONFIRM_SOURCE_IDS)
        for path in validated.image_payloads_by_path
    )
    assert len(bundle.feature_states) == CONFIRM_STATE_COUNT
    assert len(bundle.work_items) == CONFIRM_STATE_COUNT
    assert tuple(state.source_id for state in bundle.feature_states) == CONFIRM_SOURCE_IDS
    assert all(
        state.candidate_event_step_ids == CONFIRM_CANDIDATE_EVENT_STEP_IDS
        and len(state.q64) == 64
        and len(state.candidates) == 4
        for state in bundle.feature_states
    )
    assert all(
        item.ordinal == ordinal
        and item.candidate_event_step_ids == CONFIRM_CANDIDATE_EVENT_STEP_IDS
        and len(item.event_images) == 4
        for ordinal, item in enumerate(bundle.work_items)
    )
    assert set(bundle.dynamic_recent.values()) == {(3, 4)}
    assert set(bundle.ocr_rgb_v2.values()) == {(1, 2)}
    assert len(bundle.ocr_rgb_score_records) == CONFIRM_STATE_COUNT


def test_confirm20_rejects_transport_tampering(confirm_fixture: dict) -> None:
    tampered = confirm_fixture["trajectory_payload"] + b"\n"
    with pytest.raises(ValueError, match="trajectory transport byte identity drifted"):
        _validate(
            confirm_fixture,
            trajectory_payload=tampered,
            transport_identity=_identity(confirm_fixture),
        )


def test_confirm20_rejects_roster_and_geometry_drift_after_rehash(
    confirm_fixture: dict,
) -> None:
    roster_drift = copy.deepcopy(confirm_fixture["trajectories"])
    roster_drift[15]["source_id"] = "different-confirm-id"
    roster_payload = canonical_jsonl_bytes(roster_drift)
    with pytest.raises(ValueError, match="confirm source roster"):
        _validate(confirm_fixture, trajectory_payload=roster_payload)

    geometry_drift = copy.deepcopy(confirm_fixture["trajectories"])
    geometry_drift[15]["decisions"][0]["candidate_event_step_ids"] = [1, 2, 3]
    geometry_payload = canonical_jsonl_bytes(geometry_drift)
    with pytest.raises(ValueError, match="frozen n=4 geometry"):
        _validate(confirm_fixture, trajectory_payload=geometry_payload)

    event_order_drift = copy.deepcopy(confirm_fixture["trajectories"])
    event_order_drift[15]["events"][0]["step_id"] = 2
    event_order_payload = canonical_jsonl_bytes(event_order_drift)
    with pytest.raises(ValueError, match="event/state geometry"):
        _validate(confirm_fixture, trajectory_payload=event_order_payload)


def test_confirm20_extracts_and_hashes_only_required_tar_members(
    confirm_fixture: dict,
) -> None:
    changed_images = dict(confirm_fixture["image_payloads"])
    selected_path = f"images/{CONFIRM_SOURCE_IDS[0]}/observation-001.png"
    changed_images[selected_path] = confirm_fixture["alternate_image_payload"]
    changed_tar, members = build_image_tar_bytes(changed_images)
    assert len(members) == 210
    with pytest.raises(ValueError, match="selected image member identity drifted"):
        _validate(confirm_fixture, image_tar_payload=changed_tar)
