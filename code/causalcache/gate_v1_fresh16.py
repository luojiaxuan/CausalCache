"""Label-blind fresh-16 preparation and immutable evaluation artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import validate_ocr_record
from causalcache.gate_v1_contract import canonical_json_bytes
from causalcache.gate_v1_data import (
    CandidateFeatures,
    FeatureState,
    GateState,
    LabelState,
    feature_state_from_derived,
    join_feature_and_label_states,
)
from causalcache.gate_v1_fresh16_data import Fresh16LabelSlice, Fresh16TrajectorySlice
from causalcache.gate_v1_fresh16_heuristics import (
    ocr_rgb_similarity_selection,
    policy_vision_similarity_selection,
    recent_selection,
)
from causalcache.gate_v1_provenance import canonical_selection_sha256
from causalcache.restoration_v2_text_backend import PreparedImage
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_fresh16_primary_v1"
FEATURE_STATUS = "FROZEN_GATE_V1_FRESH16_FEATURE_STATES_V1"
LABEL_STATUS = "FROZEN_GATE_V1_FRESH16_LABEL_STATES_V1"
HEURISTIC_STATUS = "FROZEN_GATE_V1_FRESH16_HEURISTIC_SELECTIONS_V1"
LEARNED_STATUS = "FROZEN_GATE_V1_FRESH16_LEARNED_SELECTIONS_V1"
POLICY_SCORE_STATUS = "FROZEN_GATE_V1_FRESH16_POLICY_VISION_SCORES_V1"
FRESH_STATE_COUNT = 48
FRESH_TRAJECTORY_COUNT = 16
FRESH_IMAGE_COUNT = 80
FRESH_CANDIDATE_OCCURRENCE_COUNT = 144


@dataclass(frozen=True)
class Fresh16ImageRequirement:
    image_member_path: str
    image_sha256: str
    canonical_ocr_record_sha256: str


@dataclass(frozen=True)
class Fresh16ImageFeature:
    requirement: Fresh16ImageRequirement
    full_spatial_tokens: tuple[str, ...]
    resized_rgb_bytes: bytes


@dataclass(frozen=True)
class Fresh16PolicyWorkItem:
    ordinal: int
    source_id: str
    state_id: str
    decision_step_id: int
    event_images: tuple[tuple[int, Fresh16ImageRequirement], ...]
    current_image: Fresh16ImageRequirement

    @property
    def candidate_event_step_ids(self) -> tuple[int, ...]:
        return tuple(step for step, _ in self.event_images)


@dataclass(frozen=True)
class Fresh16LabelBlindBundle:
    feature_states: tuple[FeatureState, ...]
    work_items: tuple[Fresh16PolicyWorkItem, ...]
    image_requirements: tuple[Fresh16ImageRequirement, ...]
    dynamic_recent: Mapping[str, tuple[int, ...]]
    ocr_rgb_v2: Mapping[str, tuple[int, ...]]
    ocr_rgb_score_records: tuple[Mapping[str, Any], ...]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _safe_member(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _event_requirement(event: Mapping[str, Any]) -> Fresh16ImageRequirement:
    path = _safe_member(event.get("observation_after_path"), "event post image")
    image_sha256 = _require_sha256(
        event.get("observation_after_sha256"), "event post image SHA256"
    )
    high = event.get("high_fidelity_v2")
    refs = event.get("ocr_record_refs")
    if not isinstance(high, Mapping) or not isinstance(refs, Mapping):
        raise ValueError("event lacks high-fidelity or OCR identity")
    if (
        high.get("image_member_path") != path
        or high.get("image_sha256") != image_sha256
    ):
        raise ValueError("event high-fidelity image identity drifted")
    return Fresh16ImageRequirement(
        image_member_path=path,
        image_sha256=image_sha256,
        canonical_ocr_record_sha256=_require_sha256(
            refs.get("after_canonical_ocr_record_sha256"),
            "event OCR record SHA256",
        ),
    )


def build_fresh16_image_plan(
    trajectories: Fresh16TrajectorySlice,
) -> tuple[tuple[Fresh16PolicyWorkItem, ...], tuple[Fresh16ImageRequirement, ...]]:
    """Bind all 48 variable-history states to exactly 80 post-state screenshots."""
    if not isinstance(trajectories, Fresh16TrajectorySlice):
        raise TypeError("fresh-16 image planning requires a validated trajectory slice")
    work_items: list[Fresh16PolicyWorkItem] = []
    by_path: dict[str, Fresh16ImageRequirement] = {}
    ordinal = 0
    for trajectory in trajectories.trajectories:
        source_id = trajectory["source_id"]
        events = trajectory["events"]
        event_by_step = {event["step_id"]: event for event in events}
        requirements = {
            step: _event_requirement(event_by_step[step]) for step in range(1, 6)
        }
        for requirement in requirements.values():
            previous = by_path.setdefault(requirement.image_member_path, requirement)
            if previous != requirement:
                raise ValueError("one fresh image path has inconsistent identities")
        for decision in trajectory["decisions"]:
            candidate_ids = tuple(decision["candidate_event_step_ids"])
            current_step = decision["current_equivalent_event_step_id"]
            current = requirements[current_step]
            witnesses = decision.get("candidate_event_post_states")
            current_witness = decision.get("current_equivalence_witness")
            if not isinstance(witnesses, list) or not isinstance(current_witness, Mapping):
                raise ValueError("fresh decision image witnesses are missing")
            expected_witnesses = [
                {
                    "event_step_id": step,
                    "post_state_member_path": requirements[step].image_member_path,
                    "post_state_sha256": requirements[step].image_sha256,
                }
                for step in candidate_ids
            ]
            expected_current = {
                "event_step_id": current_step,
                "post_state_member_path": current.image_member_path,
                "post_state_sha256": current.image_sha256,
            }
            if (
                witnesses != expected_witnesses
                or dict(current_witness) != expected_current
                or decision.get("current_observation_path") != current.image_member_path
                or decision.get("current_observation_sha256") != current.image_sha256
            ):
                raise ValueError("fresh decision image equivalence witness drifted")
            work_items.append(
                Fresh16PolicyWorkItem(
                    ordinal=ordinal,
                    source_id=source_id,
                    state_id=decision["state_id"],
                    decision_step_id=decision["decision_step_id"],
                    event_images=tuple((step, requirements[step]) for step in candidate_ids),
                    current_image=current,
                )
            )
            ordinal += 1
    if (
        len(work_items) != FRESH_STATE_COUNT
        or len(by_path) != FRESH_IMAGE_COUNT
        or sum(len(item.event_images) for item in work_items)
        != FRESH_CANDIDATE_OCCURRENCE_COUNT
    ):
        raise ValueError("fresh-16 state, image, or candidate denominator drifted")
    if [item.ordinal for item in work_items] != list(range(FRESH_STATE_COUNT)):
        raise ValueError("fresh-16 policy work-item order drifted")
    return tuple(work_items), tuple(by_path[path] for path in sorted(by_path))


def materialize_fresh16_image_features(
    requirements: Sequence[Fresh16ImageRequirement],
    *,
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
    image_payloads_by_path: Mapping[str, bytes],
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
) -> Mapping[str, Fresh16ImageFeature]:
    """Validate selected OCR/image identities and reuse the frozen 256x256 RGB path."""
    expected = {item.image_member_path: item for item in requirements}
    if (
        len(expected) != FRESH_IMAGE_COUNT
        or set(ocr_records_by_path) != set(expected)
        or set(image_payloads_by_path) != set(expected)
    ):
        raise ValueError("fresh-16 selected OCR/image coverage drifted")
    result: dict[str, Fresh16ImageFeature] = {}
    for path in sorted(expected):
        requirement = expected[path]
        record = ocr_records_by_path[path]
        payload = image_payloads_by_path[path]
        if _sha256(payload) != requirement.image_sha256:
            raise ValueError(f"fresh image payload SHA256 drifted: {path}")
        prepared: PreparedImage = validate_ocr_record(
            record,
            image_bytes=payload,
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
        )
        tokens = record.get("full_spatial_tokens")
        if (
            record.get("image_member_path") != path
            or record.get("image_sha256") != requirement.image_sha256
            or record.get("canonical_ocr_record_sha256")
            != requirement.canonical_ocr_record_sha256
            or not isinstance(tokens, list)
            or any(not isinstance(token, str) for token in tokens)
        ):
            raise ValueError("fresh OCR/trajectory identity or token schema drifted")
        result[path] = Fresh16ImageFeature(
            requirement=requirement,
            full_spatial_tokens=tuple(tokens),
            resized_rgb_bytes=prepared.resized_rgb_bytes,
        )
    return result


def build_fresh16_label_blind_bundle(
    trajectories: Fresh16TrajectorySlice,
    image_features: Mapping[str, Fresh16ImageFeature],
) -> Fresh16LabelBlindBundle:
    """Build gate features and CPU heuristics without accepting any D(S) object."""
    work_items, requirements = build_fresh16_image_plan(trajectories)
    if set(image_features) != {item.image_member_path for item in requirements}:
        raise ValueError("fresh image-feature inventory differs from the frozen plan")
    ocr_records = {
        path: {"full_spatial_tokens": list(feature.full_spatial_tokens)}
        for path, feature in image_features.items()
    }
    features: list[FeatureState] = []
    dynamic: dict[str, tuple[int, ...]] = {}
    ocr_rgb: dict[str, tuple[int, ...]] = {}
    score_records: list[Mapping[str, Any]] = []
    item_by_state = {item.state_id: item for item in work_items}
    for trajectory in trajectories.trajectories:
        for decision in trajectory["decisions"]:
            state = feature_state_from_derived(
                trajectory,
                decision,
                ocr_records_by_path=ocr_records,
            )
            item = item_by_state[state.state_id]
            event_features = {
                step: image_features[identity.image_member_path]
                for step, identity in item.event_images
            }
            current = image_features[item.current_image.image_member_path]
            selection = ocr_rgb_similarity_selection(
                event_step_ids=state.candidate_event_step_ids,
                event_ocr_tokens={
                    step: feature.full_spatial_tokens
                    for step, feature in event_features.items()
                },
                current_ocr_tokens=current.full_spatial_tokens,
                event_resized_rgb_bytes={
                    step: feature.resized_rgb_bytes
                    for step, feature in event_features.items()
                },
                current_resized_rgb_bytes=current.resized_rgb_bytes,
            )
            features.append(state)
            dynamic[state.state_id] = recent_selection(state.candidate_event_step_ids)
            ocr_rgb[state.state_id] = selection.selected_event_step_ids
            score_records.append(
                {
                    "source_id": state.source_id,
                    "state_id": state.state_id,
                    "candidate_event_step_ids": list(state.candidate_event_step_ids),
                    "scores_by_event_step": [
                        {"event_step_id": step, "score": score}
                        for step, score in selection.scores_by_event_step
                    ],
                    "ranked_event_step_ids": list(selection.ranked_event_step_ids),
                    "selected_event_step_ids": list(selection.selected_event_step_ids),
                }
            )
    if len(features) != FRESH_STATE_COUNT or len({item.state_id for item in features}) != FRESH_STATE_COUNT:
        raise ValueError("fresh feature-state inventory drifted")
    return Fresh16LabelBlindBundle(
        feature_states=tuple(features),
        work_items=work_items,
        image_requirements=requirements,
        dynamic_recent=dynamic,
        ocr_rgb_v2=ocr_rgb,
        ocr_rgb_score_records=tuple(score_records),
    )


def decode_policy_work_item_images(
    item: Fresh16PolicyWorkItem,
    image_payloads_by_path: Mapping[str, bytes],
) -> tuple[Any, ...]:
    """Decode candidate/current screenshots in the variable-n frozen order."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("fresh policy-vision decoding requires Pillow") from error
    identities = (*(identity for _, identity in item.event_images), item.current_image)
    images = []
    for identity in identities:
        try:
            payload = image_payloads_by_path[identity.image_member_path]
        except KeyError as error:
            raise ValueError("fresh policy work item lacks an image payload") from error
        if _sha256(payload) != identity.image_sha256:
            raise ValueError("fresh policy image payload identity drifted")
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            images.append(opened.convert("RGB"))
    if len(images) != len(item.event_images) + 1:
        raise RuntimeError("fresh policy image batch geometry drifted")
    return tuple(images)


def policy_selection_record(
    item: Fresh16PolicyWorkItem,
    scores_by_event_step: Mapping[int, float],
    *,
    worker_id: str,
    device: str,
    gpu_uuid: str,
) -> Mapping[str, Any]:
    selection = policy_vision_similarity_selection(
        scores_by_event_step,
        event_step_ids=item.candidate_event_step_ids,
    )
    if not worker_id or not device or not gpu_uuid:
        raise ValueError("policy-vision worker provenance must be non-empty")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": POLICY_SCORE_STATUS,
        "ordinal": item.ordinal,
        "source_id": item.source_id,
        "state_id": item.state_id,
        "decision_step_id": item.decision_step_id,
        "candidate_event_step_ids": list(item.candidate_event_step_ids),
        "scores_by_event_step": [
            {"event_step_id": step, "score": score}
            for step, score in selection.scores_by_event_step
        ],
        "ranked_event_step_ids": list(selection.ranked_event_step_ids),
        "selected_event_step_ids": list(selection.selected_event_step_ids),
        "worker_id": worker_id,
        "device": device,
        "gpu_uuid": gpu_uuid,
    }


def combine_policy_score_records(
    records: Sequence[Mapping[str, Any]],
    work_items: Sequence[Fresh16PolicyWorkItem],
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, tuple[int, ...]]]:
    by_ordinal: dict[int, Mapping[str, Any]] = {}
    for record in records:
        ordinal = record.get("ordinal")
        if type(ordinal) is not int or ordinal in by_ordinal:
            raise ValueError("policy-vision score ordinal is invalid or duplicated")
        by_ordinal[ordinal] = record
    if set(by_ordinal) != set(range(FRESH_STATE_COUNT)) or len(work_items) != FRESH_STATE_COUNT:
        raise ValueError("policy-vision score state denominator drifted")
    ordered = tuple(by_ordinal[index] for index in range(FRESH_STATE_COUNT))
    selections: dict[str, tuple[int, ...]] = {}
    for item, record in zip(work_items, ordered, strict=True):
        score_rows = record.get("scores_by_event_step")
        if not isinstance(score_rows, list) or len(score_rows) != len(
            item.candidate_event_step_ids
        ):
            raise ValueError("policy-vision score record differs from its work item")
        scores: dict[int, float] = {}
        for expected_step, raw in zip(
            item.candidate_event_step_ids, score_rows, strict=True
        ):
            if (
                not isinstance(raw, Mapping)
                or set(raw) != {"event_step_id", "score"}
                or raw.get("event_step_id") != expected_step
                or isinstance(raw.get("score"), bool)
                or not isinstance(raw.get("score"), (int, float))
                or not math.isfinite(float(raw["score"]))
            ):
                raise ValueError(
                    "policy-vision score record differs from its work item"
                )
            scores[expected_step] = float(raw["score"])
        replay = policy_vision_similarity_selection(
            scores,
            event_step_ids=item.candidate_event_step_ids,
        )
        selected = replay.selected_event_step_ids
        if (
            set(record)
            != {
                "schema_version",
                "protocol_id",
                "status",
                "ordinal",
                "source_id",
                "state_id",
                "decision_step_id",
                "candidate_event_step_ids",
                "scores_by_event_step",
                "ranked_event_step_ids",
                "selected_event_step_ids",
                "worker_id",
                "device",
                "gpu_uuid",
            }
            or record.get("schema_version") != SCHEMA_VERSION
            or record.get("protocol_id") != PROTOCOL_ID
            or record.get("status") != POLICY_SCORE_STATUS
            or record.get("ordinal") != item.ordinal
            or record.get("source_id") != item.source_id
            or record.get("state_id") != item.state_id
            or record.get("decision_step_id") != item.decision_step_id
            or record.get("candidate_event_step_ids")
            != list(item.candidate_event_step_ids)
            or record.get("scores_by_event_step")
            != [
                {"event_step_id": step, "score": score}
                for step, score in replay.scores_by_event_step
            ]
            or record.get("ranked_event_step_ids")
            != list(replay.ranked_event_step_ids)
            or record.get("selected_event_step_ids") != list(selected)
            or not isinstance(record.get("worker_id"), str)
            or not record["worker_id"]
            or not isinstance(record.get("device"), str)
            or not record["device"]
            or not isinstance(record.get("gpu_uuid"), str)
            or not record["gpu_uuid"]
        ):
            raise ValueError("policy-vision score record differs from its work item")
        selections[item.state_id] = selected
    return ordered, selections


def feature_states_jsonl_bytes(states: Sequence[FeatureState]) -> bytes:
    if len(states) != FRESH_STATE_COUNT:
        raise ValueError("fresh feature serialization requires exactly 48 states")
    records = []
    for ordinal, state in enumerate(states):
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": FEATURE_STATUS,
                "ordinal": ordinal,
                "source_id": state.source_id,
                "state_id": state.state_id,
                "decision_step_id": state.decision_step_id,
                "candidate_event_step_ids": list(state.candidate_event_step_ids),
                "q64": list(state.q64),
                "candidates": [
                    {
                        "event_step_id": item.event_step_id,
                        "h64": list(item.h64),
                        "g8": list(item.g8),
                    }
                    for item in state.candidates
                ],
            }
        )
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def read_feature_states_jsonl(payload: bytes) -> tuple[FeatureState, ...]:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise ValueError("fresh feature artifact must be LF-terminated bytes")
    lines = payload[:-1].split(b"\n")
    if len(lines) != FRESH_STATE_COUNT or any(not line for line in lines):
        raise ValueError("fresh feature artifact row denominator drifted")
    result = []
    for ordinal, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("fresh feature artifact is not JSONL") from error
        if not isinstance(record, Mapping) or canonical_json_bytes(record) != line:
            raise ValueError("fresh feature artifact is not canonical JSONL")
        candidates = record.get("candidates")
        if (
            record.get("status") != FEATURE_STATUS
            or record.get("ordinal") != ordinal
            or not isinstance(candidates, list)
        ):
            raise ValueError("fresh feature artifact identity drifted")
        result.append(
            FeatureState(
                source_id=record["source_id"],
                state_id=record["state_id"],
                decision_step_id=record["decision_step_id"],
                candidate_event_step_ids=tuple(record["candidate_event_step_ids"]),
                q64=tuple(float(value) for value in record["q64"]),
                candidates=tuple(
                    CandidateFeatures(
                        event_step_id=item["event_step_id"],
                        h64=tuple(float(value) for value in item["h64"]),
                        g8=tuple(float(value) for value in item["g8"]),
                    )
                    for item in candidates
                ),
            )
        )
    if feature_states_jsonl_bytes(result) != payload:
        raise ValueError("fresh feature artifact canonical replay differs")
    return tuple(result)


def _float64_hex(value: Any, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or (
        scalar == 0.0 and math.copysign(1.0, scalar) < 0.0
    ):
        raise ValueError(f"{label} must be finite and cannot be negative zero")
    return struct.pack(">d", scalar).hex()


def _hex_float64(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, str)
        or len(value) != 16
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 16 lowercase f64 hex digits")
    scalar = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(scalar) or (
        scalar == 0.0 and math.copysign(1.0, scalar) < 0.0
    ):
        raise ValueError(f"{label} decodes to non-finite or negative zero")
    return scalar


def label_states_jsonl_bytes(states: Sequence[LabelState]) -> bytes:
    if len(states) != FRESH_STATE_COUNT:
        raise ValueError("fresh label serialization requires exactly 48 states")
    records = []
    distance_count = 0
    for ordinal, state in enumerate(states):
        rows = [
            {
                "coalition_event_step_ids": list(row.coalition),
                "distance_kl_f64_hex": _float64_hex(
                    row.distance,
                    label="fresh label distance",
                ),
            }
            for row in state.table.rows
        ]
        distance_count += len(rows)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": LABEL_STATUS,
                "ordinal": ordinal,
                "source_id": state.source_id,
                "state_id": state.state_id,
                "decision_step_id": state.decision_step_id,
                "candidate_event_step_ids": list(state.table.event_ids),
                "distance_rows": rows,
            }
        )
    if distance_count != 448:
        raise ValueError("fresh label distance denominator drifted")
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def read_label_states_jsonl(payload: bytes) -> tuple[LabelState, ...]:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise ValueError("fresh label artifact must be LF-terminated bytes")
    lines = payload[:-1].split(b"\n")
    if len(lines) != FRESH_STATE_COUNT or any(not line for line in lines):
        raise ValueError("fresh label artifact row denominator drifted")
    result: list[LabelState] = []
    for ordinal, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("fresh label artifact is not JSONL") from error
        if (
            not isinstance(record, Mapping)
            or canonical_json_bytes(record) != line
            or record.get("status") != LABEL_STATUS
            or record.get("ordinal") != ordinal
            or not isinstance(record.get("candidate_event_step_ids"), list)
            or not isinstance(record.get("distance_rows"), list)
        ):
            raise ValueError("fresh label artifact identity or encoding drifted")
        distances: dict[tuple[int, ...], float] = {}
        for row in record["distance_rows"]:
            if not isinstance(row, Mapping) or set(row) != {
                "coalition_event_step_ids",
                "distance_kl_f64_hex",
            }:
                raise ValueError("fresh label distance row schema drifted")
            coalition = row["coalition_event_step_ids"]
            if not isinstance(coalition, list):
                raise ValueError("fresh label coalition is malformed")
            key = tuple(coalition)
            if key in distances:
                raise ValueError("fresh label artifact has duplicate coalitions")
            distances[key] = _hex_float64(
                row["distance_kl_f64_hex"],
                label="fresh label distance",
            )
        table = validate_complete_distance_table(
            tuple(record["candidate_event_step_ids"]),
            distances,
        )
        result.append(
            LabelState(
                source_id=record["source_id"],
                state_id=record["state_id"],
                decision_step_id=record["decision_step_id"],
                table=table,
            )
        )
    if (
        sum(len(state.table.rows) for state in result) != 448
        or label_states_jsonl_bytes(result) != payload
    ):
        raise ValueError("fresh label artifact canonical replay differs")
    return tuple(result)


def selection_artifact_bytes(
    name: str,
    selections: Mapping[str, Sequence[int]],
    *,
    status: str = HEURISTIC_STATUS,
) -> bytes:
    if name not in {
        "dynamic_recent",
        "ocr_rgb_v2",
        "policy_vision_v3",
        "conditional",
        "independent",
    }:
        raise ValueError("fresh selection artifact name is not allowlisted")
    normalized = {
        state_id: tuple(int(event) for event in selected)
        for state_id, selected in selections.items()
    }
    if len(normalized) != FRESH_STATE_COUNT:
        raise ValueError("fresh selection artifact must cover exactly 48 states")
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "name": name,
        "selection_sha256": canonical_selection_sha256(normalized),
        "records": [
            {"state_id": state_id, "selected_event_step_ids": list(normalized[state_id])}
            for state_id in sorted(normalized)
        ],
    }
    return canonical_json_bytes(value) + b"\n"


def read_selection_artifact(
    payload: bytes,
    *,
    expected_name: str,
    expected_status: str = HEURISTIC_STATUS,
) -> Mapping[str, tuple[int, ...]]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fresh selection artifact is not JSON") from error
    if (
        not isinstance(value, Mapping)
        or canonical_json_bytes(value) + b"\n" != payload
        or value.get("name") != expected_name
        or value.get("status") != expected_status
        or not isinstance(value.get("records"), list)
    ):
        raise ValueError("fresh selection artifact identity or encoding drifted")
    result: dict[str, tuple[int, ...]] = {}
    for record in value["records"]:
        if not isinstance(record, Mapping):
            raise ValueError("fresh selection record is malformed")
        state_id = record.get("state_id")
        selected = record.get("selected_event_step_ids")
        if (
            not isinstance(state_id, str)
            or state_id in result
            or not isinstance(selected, list)
            or any(type(event) is not int for event in selected)
        ):
            raise ValueError("fresh selection record identity or events are malformed")
        result[state_id] = tuple(selected)
    if (
        len(result) != FRESH_STATE_COUNT
        or canonical_selection_sha256(result) != value.get("selection_sha256")
        or selection_artifact_bytes(expected_name, result, status=expected_status) != payload
    ):
        raise ValueError("fresh selection artifact replay differs")
    return result


def jsonl_score_records_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    if len(records) != FRESH_STATE_COUNT:
        raise ValueError("fresh score artifact must contain exactly 48 records")
    payload = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    if any(not math.isfinite(float(row["score"])) for record in records for row in record["scores_by_event_step"]):
        raise ValueError("fresh score artifact contains a non-finite score")
    return payload


def join_fresh16_evaluation_states(
    features: Sequence[FeatureState],
    labels: Fresh16LabelSlice,
) -> tuple[GateState, ...]:
    if not isinstance(labels, Fresh16LabelSlice):
        raise TypeError("fresh evaluation join requires a validated label slice")
    states = join_feature_and_label_states(
        features,
        labels.labels,
        expected_source_ids=labels.source_ids,
    )
    if len(states) != FRESH_STATE_COUNT:
        raise ValueError("fresh evaluation join state denominator drifted")
    return states


__all__ = [
    "FEATURE_STATUS",
    "FRESH_CANDIDATE_OCCURRENCE_COUNT",
    "FRESH_IMAGE_COUNT",
    "FRESH_STATE_COUNT",
    "FRESH_TRAJECTORY_COUNT",
    "HEURISTIC_STATUS",
    "LABEL_STATUS",
    "LEARNED_STATUS",
    "POLICY_SCORE_STATUS",
    "PROTOCOL_ID",
    "Fresh16ImageFeature",
    "Fresh16ImageRequirement",
    "Fresh16LabelBlindBundle",
    "Fresh16PolicyWorkItem",
    "build_fresh16_image_plan",
    "build_fresh16_label_blind_bundle",
    "combine_policy_score_records",
    "decode_policy_work_item_images",
    "feature_states_jsonl_bytes",
    "join_fresh16_evaluation_states",
    "jsonl_score_records_bytes",
    "label_states_jsonl_bytes",
    "materialize_fresh16_image_features",
    "policy_selection_record",
    "read_feature_states_jsonl",
    "read_label_states_jsonl",
    "read_selection_artifact",
    "selection_artifact_bytes",
]
