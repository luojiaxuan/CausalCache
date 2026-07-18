"""Frozen, label-blind data path for the independent-gate confirm-20 slice."""

from __future__ import annotations

import hashlib
import io
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import (
    IMAGE_TAR_RELATIVE_PATH,
    OCR_JSONL_RELATIVE_PATH,
    TRAJECTORY_JSONL_RELATIVE_PATH,
    parse_canonical_jsonl,
    sha256_bytes,
    validate_artifact,
    validate_ocr_record,
)
from causalcache.gate_v1_data import FeatureState, feature_state_from_derived
from causalcache.gate_v1_fresh16_heuristics import (
    ocr_rgb_similarity_selection,
    recent_selection,
)
from causalcache.restoration_v2_text_backend import PreparedImage


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_independent_confirm20_data_v1"
CONFIRM_ROLE = "v2_confirm_primary"
CONFIRM_DECISION_STEP_ID = 6
CONFIRM_CANDIDATE_EVENT_STEP_IDS = (1, 2, 3, 4)
CONFIRM_CURRENT_EVENT_STEP_ID = 5
CONFIRM_BUDGET_EVENT_CAPACITY = 2
CONFIRM_TRAJECTORY_COUNT = 20
CONFIRM_STATE_COUNT = 20
CONFIRM_IMAGE_COUNT = 100
DERIVED_TRAJECTORY_COUNT = 35
DERIVED_EVENT_COUNT = 175
DERIVED_STATE_COUNT = 65
DERIVED_IMAGE_COUNT = 210

CONFIRM_SOURCE_IDS = (
    "0138580662882674",
    "0141017089161442",
    "0088916287721604",
    "0182869798349621",
    "0062595953452329",
    "0206356317737773",
    "0128873485175141",
    "0097146299278532",
    "0126322136688861",
    "0003269005797789",
    "0107883386619316",
    "0015844124232805",
    "0007216161758334",
    "0208932442851349",
    "0140656067727410",
    "0205621648359625",
    "0047234313779636",
    "0125033580932478",
    "0163270032450668",
    "0061317255667027",
)
CONFIRM_SOURCE_IDS_SHA256 = (
    "c84ba8b9b705abc7ba7d6d7230b868fd4600e3a835b1c763a8aaf1f93feb052e"
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_PAYLOAD_SEAL = object()
_TRAJECTORY_KEYS = {
    "source_id",
    "role",
    "instruction",
    "instruction_sha256",
    "platform",
    "apps",
    "device_name",
    "resolution",
    "terminal_status",
    "source",
    "selection",
    "events",
    "decisions",
}
_EVENT_KEYS = {
    "step_id",
    "observation_before_path",
    "observation_before_sha256",
    "observation_after_path",
    "observation_after_sha256",
    "executed_action",
    "source_tool_call",
    "low_fidelity_v2",
    "low_fidelity_v2_serialized",
    "low_fidelity_v2_sha256",
    "low_fidelity_v2_metadata",
    "high_fidelity_v2",
    "ocr_record_refs",
}
_DECISION_KEYS = {
    "state_id",
    "decision_step_id",
    "history_event_step_ids",
    "candidate_event_step_ids",
    "current_equivalent_event_step_id",
    "current_observation_path",
    "current_observation_sha256",
    "validated_action",
    "validated_action_sha256",
    "validation_source",
}


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _safe_member(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return value


@dataclass(frozen=True)
class Confirm20TransportIdentity:
    artifact_tree_sha256: str
    trajectory_sha256: str
    trajectory_size_bytes: int
    trajectory_record_count: int
    ocr_sha256: str
    ocr_size_bytes: int
    ocr_record_count: int
    image_tar_sha256: str
    image_tar_size_bytes: int
    image_member_count: int

    def __post_init__(self) -> None:
        for field in (
            "artifact_tree_sha256",
            "trajectory_sha256",
            "ocr_sha256",
            "image_tar_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        expected_counts = {
            "trajectory_record_count": DERIVED_TRAJECTORY_COUNT,
            "ocr_record_count": DERIVED_IMAGE_COUNT,
            "image_member_count": DERIVED_IMAGE_COUNT,
        }
        for field, expected in expected_counts.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} must equal the frozen denominator {expected}")
        for field in (
            "trajectory_size_bytes",
            "ocr_size_bytes",
            "image_tar_size_bytes",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")


FROZEN_TRANSPORT_IDENTITY = Confirm20TransportIdentity(
    artifact_tree_sha256=(
        "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e"
    ),
    trajectory_sha256=(
        "c08ae40d0fe27783c03c00459066db43dd8610cc8b77f74babdcc97dc18b747e"
    ),
    trajectory_size_bytes=595_807,
    trajectory_record_count=35,
    ocr_sha256=(
        "b8c96b9951d8166839dde71b4c8ca60f099b92e892bbeafde819beb96a9fc97d"
    ),
    ocr_size_bytes=1_897_573,
    ocr_record_count=210,
    image_tar_sha256=(
        "1a383c116117d26d7a123343816187695c8312a5923359a6518d7f0051d17326"
    ),
    image_tar_size_bytes=155_883_520,
    image_member_count=210,
)


@dataclass(frozen=True)
class Confirm20ImageRequirement:
    image_member_path: str
    image_sha256: str
    canonical_ocr_record_sha256: str


@dataclass(frozen=True)
class Confirm20ImageFeature:
    requirement: Confirm20ImageRequirement
    full_spatial_tokens: tuple[str, ...]
    resized_rgb_bytes: bytes


@dataclass(frozen=True)
class Confirm20PolicyWorkItem:
    ordinal: int
    source_id: str
    state_id: str
    decision_step_id: int
    event_images: tuple[tuple[int, Confirm20ImageRequirement], ...]
    current_image: Confirm20ImageRequirement

    @property
    def candidate_event_step_ids(self) -> tuple[int, ...]:
        return tuple(step for step, _ in self.event_images)


class ValidatedConfirm20Payloads:
    """Opaque confirm-only projection produced by strict byte/semantic validation."""

    __slots__ = (
        "artifact_tree_sha256",
        "image_features_by_path",
        "image_payloads_by_path",
        "image_requirements",
        "ocr_records_by_path",
        "source_ids",
        "trajectories",
        "transport_identity",
        "_seal",
    )

    def __init__(
        self,
        *,
        artifact_tree_sha256: str,
        trajectories: tuple[Mapping[str, Any], ...],
        source_ids: tuple[str, ...],
        ocr_records_by_path: Mapping[str, Mapping[str, Any]],
        image_payloads_by_path: Mapping[str, bytes],
        image_requirements: tuple[Confirm20ImageRequirement, ...],
        image_features_by_path: Mapping[str, Confirm20ImageFeature],
        transport_identity: Confirm20TransportIdentity,
        _seal: object,
    ) -> None:
        if _seal is not _PAYLOAD_SEAL:
            raise TypeError("confirm-20 payloads must come from strict validation")
        self.artifact_tree_sha256 = artifact_tree_sha256
        self.trajectories = trajectories
        self.source_ids = source_ids
        self.ocr_records_by_path = MappingProxyType(dict(ocr_records_by_path))
        self.image_payloads_by_path = MappingProxyType(dict(image_payloads_by_path))
        self.image_requirements = image_requirements
        self.image_features_by_path = MappingProxyType(dict(image_features_by_path))
        self.transport_identity = transport_identity
        self._seal = _seal


@dataclass(frozen=True)
class Confirm20LabelBlindBundle:
    feature_states: tuple[FeatureState, ...]
    work_items: tuple[Confirm20PolicyWorkItem, ...]
    image_requirements: tuple[Confirm20ImageRequirement, ...]
    image_payloads_by_path: Mapping[str, bytes]
    dynamic_recent: Mapping[str, tuple[int, ...]]
    ocr_rgb_v2: Mapping[str, tuple[int, ...]]
    ocr_rgb_score_records: tuple[Mapping[str, Any], ...]


def _verify_transport(payload: bytes, *, sha256: str, size_bytes: int, label: str) -> None:
    if not isinstance(payload, bytes):
        raise TypeError(f"{label} transport must be bytes")
    if len(payload) != size_bytes or sha256_bytes(payload) != sha256:
        raise ValueError(f"{label} transport byte identity drifted")


def _event_requirement(event: Mapping[str, Any]) -> Confirm20ImageRequirement:
    if set(event) != _EVENT_KEYS:
        raise ValueError("confirm event field inventory drifted")
    path = _safe_member(event.get("observation_after_path"), "event post image")
    image_sha256 = _require_sha256(
        event.get("observation_after_sha256"), "event post image SHA256"
    )
    high = event.get("high_fidelity_v2")
    refs = event.get("ocr_record_refs")
    if not isinstance(high, Mapping) or not isinstance(refs, Mapping):
        raise ValueError("confirm event lacks high-fidelity or OCR identity")
    if high.get("image_member_path") != path or high.get("image_sha256") != image_sha256:
        raise ValueError("confirm event high-fidelity image identity drifted")
    return Confirm20ImageRequirement(
        image_member_path=path,
        image_sha256=image_sha256,
        canonical_ocr_record_sha256=_require_sha256(
            refs.get("after_canonical_ocr_record_sha256"),
            "event OCR record SHA256",
        ),
    )


def _select_confirm_trajectories(
    trajectory_payload: bytes,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Confirm20ImageRequirement, ...]]:
    records = parse_canonical_jsonl(trajectory_payload, label="confirm parent trajectories")
    if len(records) != DERIVED_TRAJECTORY_COUNT:
        raise ValueError("confirm parent trajectory denominator drifted")
    if any(set(record) != _TRAJECTORY_KEYS for record in records):
        raise ValueError("confirm parent trajectory field inventory drifted")
    source_ids = [record.get("source_id") for record in records]
    if (
        any(not isinstance(source_id, str) or not source_id for source_id in source_ids)
        or len(set(source_ids)) != DERIVED_TRAJECTORY_COUNT
    ):
        raise ValueError("confirm parent source identity inventory drifted")
    roles = [record.get("role") for record in records]
    expected_roles = (
        ["v2_label_train"] * 10
        + ["v2_development"] * 5
        + [CONFIRM_ROLE] * CONFIRM_TRAJECTORY_COUNT
    )
    if roles != expected_roles:
        raise ValueError("confirm parent frozen role order drifted")
    if (
        sum(len(record.get("events", ())) for record in records) != DERIVED_EVENT_COUNT
        or sum(len(record.get("decisions", ())) for record in records) != DERIVED_STATE_COUNT
    ):
        raise ValueError("confirm parent event/state denominator drifted")
    confirm = tuple(record for record in records if record["role"] == CONFIRM_ROLE)
    observed_roster = tuple(str(record["source_id"]) for record in confirm)
    if (
        observed_roster != CONFIRM_SOURCE_IDS
        or hashlib.sha256(
            ("[" + ",".join(f'"{value}"' for value in observed_roster) + "]").encode(
                "utf-8"
            )
        ).hexdigest()
        != CONFIRM_SOURCE_IDS_SHA256
    ):
        raise ValueError("confirm source roster, order, or digest drifted")

    requirements_by_path: dict[str, Confirm20ImageRequirement] = {}
    for trajectory in confirm:
        source_id = str(trajectory["source_id"])
        events = trajectory.get("events")
        decisions = trajectory.get("decisions")
        if (
            not isinstance(events, list)
            or len(events) != CONFIRM_CURRENT_EVENT_STEP_ID
            or [event.get("step_id") for event in events if isinstance(event, Mapping)]
            != list(range(1, CONFIRM_CURRENT_EVENT_STEP_ID + 1))
            or not isinstance(decisions, list)
            or len(decisions) != 1
        ):
            raise ValueError("confirm trajectory event/state geometry drifted")
        decision = decisions[0]
        if not isinstance(decision, Mapping) or set(decision) != _DECISION_KEYS:
            raise ValueError("confirm decision field inventory drifted")
        expected_state_id = f"{source_id}:decision_step:{CONFIRM_DECISION_STEP_ID:03d}"
        if (
            decision.get("state_id") != expected_state_id
            or decision.get("decision_step_id") != CONFIRM_DECISION_STEP_ID
            or decision.get("history_event_step_ids") != [1, 2, 3, 4, 5]
            or decision.get("candidate_event_step_ids")
            != list(CONFIRM_CANDIDATE_EVENT_STEP_IDS)
            or decision.get("current_equivalent_event_step_id")
            != CONFIRM_CURRENT_EVENT_STEP_ID
        ):
            raise ValueError("confirm decision frozen n=4 geometry drifted")
        by_step: dict[int, Confirm20ImageRequirement] = {}
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError("confirm event must be an object")
            requirement = _event_requirement(event)
            step = int(event["step_id"])
            expected_path = f"images/{source_id}/observation-{step:03d}.png"
            if requirement.image_member_path != expected_path:
                raise ValueError("confirm event post-image path drifted")
            by_step[step] = requirement
            prior = requirements_by_path.setdefault(requirement.image_member_path, requirement)
            if prior != requirement:
                raise ValueError("one confirm image path has inconsistent identities")
        current = by_step[CONFIRM_CURRENT_EVENT_STEP_ID]
        if (
            decision.get("current_observation_path") != current.image_member_path
            or decision.get("current_observation_sha256") != current.image_sha256
        ):
            raise ValueError("confirm current-observation identity drifted")
    if len(requirements_by_path) != CONFIRM_IMAGE_COUNT:
        raise ValueError("confirm required-image denominator drifted")
    return confirm, tuple(requirements_by_path[path] for path in sorted(requirements_by_path))


def _select_ocr_records(
    ocr_payload: bytes,
    requirements: Sequence[Confirm20ImageRequirement],
) -> Mapping[str, Mapping[str, Any]]:
    records = parse_canonical_jsonl(ocr_payload, label="confirm parent OCR records")
    if len(records) != DERIVED_IMAGE_COUNT:
        raise ValueError("confirm parent OCR denominator drifted")
    by_path: dict[str, Mapping[str, Any]] = {}
    for record in records:
        path = _safe_member(record.get("image_member_path"), "OCR image path")
        if not path.startswith("images/") or path in by_path:
            raise ValueError("confirm parent OCR image path inventory drifted")
        by_path[path] = record
    if list(by_path) != sorted(by_path):
        raise ValueError("confirm parent OCR records are not path-sorted")
    expected = {item.image_member_path: item for item in requirements}
    if len(expected) != CONFIRM_IMAGE_COUNT or not set(expected).issubset(by_path):
        raise ValueError("confirm selected OCR coverage drifted")
    selected: dict[str, Mapping[str, Any]] = {}
    for path in sorted(expected):
        record = by_path[path]
        requirement = expected[path]
        if (
            record.get("image_sha256") != requirement.image_sha256
            or record.get("canonical_ocr_record_sha256")
            != requirement.canonical_ocr_record_sha256
        ):
            raise ValueError("confirm selected OCR identity drifted")
        selected[path] = record
    return selected


def _extract_selected_images(
    image_tar_payload: bytes,
    requirements: Sequence[Confirm20ImageRequirement],
    *,
    expected_member_count: int,
) -> Mapping[str, bytes]:
    expected = {item.image_member_path: item for item in requirements}
    if len(expected) != CONFIRM_IMAGE_COUNT:
        raise ValueError("confirm image extraction requires exactly 100 unique paths")
    selected: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(image_tar_payload), mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if (
            len(members) != expected_member_count
            or names != sorted(names)
            or len(names) != len(set(names))
        ):
            raise ValueError("confirm parent image archive inventory drifted")
        for member in members:
            name = _safe_member(member.name, "image tar member")
            if (
                not name.startswith("images/")
                or not member.isfile()
                or member.type != tarfile.REGTYPE
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mode != 0o644
                or member.pax_headers
            ):
                raise ValueError("confirm parent image archive metadata drifted")
            if name not in expected:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("confirm selected image member is unreadable")
            payload = stream.read()
            if (
                not payload
                or len(payload) != member.size
                or sha256_bytes(payload) != expected[name].image_sha256
            ):
                raise ValueError(f"confirm selected image member identity drifted: {name}")
            selected[name] = payload
    if set(selected) != set(expected):
        raise ValueError("confirm selected image archive coverage drifted")
    return selected


def validate_confirm20_payloads(
    *,
    artifact_tree_sha256: str,
    trajectory_payload: bytes,
    ocr_payload: bytes,
    image_tar_payload: bytes,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    transport_identity: Confirm20TransportIdentity = FROZEN_TRANSPORT_IDENTITY,
) -> ValidatedConfirm20Payloads:
    """Validate immutable parent transports and expose only the frozen confirm slice."""
    if not isinstance(transport_identity, Confirm20TransportIdentity):
        raise TypeError("confirm transport identity must be versioned")
    if artifact_tree_sha256 != transport_identity.artifact_tree_sha256:
        raise ValueError("confirm parent artifact-tree identity drifted")
    _verify_transport(
        trajectory_payload,
        sha256=transport_identity.trajectory_sha256,
        size_bytes=transport_identity.trajectory_size_bytes,
        label="trajectory",
    )
    _verify_transport(
        ocr_payload,
        sha256=transport_identity.ocr_sha256,
        size_bytes=transport_identity.ocr_size_bytes,
        label="OCR",
    )
    _verify_transport(
        image_tar_payload,
        sha256=transport_identity.image_tar_sha256,
        size_bytes=transport_identity.image_tar_size_bytes,
        label="image tar",
    )
    trajectories, requirements = _select_confirm_trajectories(trajectory_payload)
    ocr_records = _select_ocr_records(ocr_payload, requirements)
    image_payloads = _extract_selected_images(
        image_tar_payload,
        requirements,
        expected_member_count=transport_identity.image_member_count,
    )
    image_features: dict[str, Confirm20ImageFeature] = {}
    for requirement in requirements:
        path = requirement.image_member_path
        record = ocr_records[path]
        prepared: PreparedImage = validate_ocr_record(
            record,
            image_bytes=image_payloads[path],
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
        )
        tokens = record.get("full_spatial_tokens")
        if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
            raise ValueError("confirm selected OCR spatial-token schema drifted")
        image_features[path] = Confirm20ImageFeature(
            requirement=requirement,
            full_spatial_tokens=tuple(tokens),
            resized_rgb_bytes=prepared.resized_rgb_bytes,
        )
    if len(image_features) != CONFIRM_IMAGE_COUNT:
        raise ValueError("confirm selected image-feature denominator drifted")
    return ValidatedConfirm20Payloads(
        artifact_tree_sha256=artifact_tree_sha256,
        trajectories=trajectories,
        source_ids=CONFIRM_SOURCE_IDS,
        ocr_records_by_path=ocr_records,
        image_payloads_by_path=image_payloads,
        image_requirements=requirements,
        image_features_by_path=image_features,
        transport_identity=transport_identity,
        _seal=_PAYLOAD_SEAL,
    )


def load_frozen_confirm20_artifact(
    *,
    output_dir: str | Path,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    validation_bindings: Mapping[str, Any],
) -> ValidatedConfirm20Payloads:
    """Run the full formal parent validator before the confirm-only projection."""
    bindings = dict(validation_bindings)
    forbidden = {
        "output_dir",
        "backend_config",
        "backend_config_sha256",
        "require_formal",
        "require_ocr_replay",
    }
    overlap = forbidden.intersection(bindings)
    if overlap:
        raise ValueError(f"artifact validation bindings contain controlled keys: {sorted(overlap)}")
    validation = validate_artifact(
        output_dir=output_dir,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
        require_formal=True,
        require_ocr_replay=True,
        **bindings,
    )
    if (
        validation.get("outcome")
        != "PASSED_GUIODYSSEY_RESTORATION_V2_ARTIFACT_VALIDATION"
        or validation.get("artifact_tree_sha256")
        != FROZEN_TRANSPORT_IDENTITY.artifact_tree_sha256
        or validation.get("counts")
        != {
            "trajectory_count": DERIVED_TRAJECTORY_COUNT,
            "event_count": DERIVED_EVENT_COUNT,
            "state_count": DERIVED_STATE_COUNT,
            "image_member_count": DERIVED_IMAGE_COUNT,
            "ocr_record_count": DERIVED_IMAGE_COUNT,
        }
        or validation.get("formal_counts_enforced") is not True
        or validation.get("ocr_replay_performed") is not True
        or validation.get("ocr_replay_record_count") != DERIVED_IMAGE_COUNT
        or validation.get("policy_output_generated") is not False
        or validation.get("restoration_output_generated") is not False
    ):
        raise ValueError("formal confirm parent artifact validation outcome drifted")
    root = Path(output_dir)
    return validate_confirm20_payloads(
        artifact_tree_sha256=str(validation["artifact_tree_sha256"]),
        trajectory_payload=(root / TRAJECTORY_JSONL_RELATIVE_PATH).read_bytes(),
        ocr_payload=(root / OCR_JSONL_RELATIVE_PATH).read_bytes(),
        image_tar_payload=(root / IMAGE_TAR_RELATIVE_PATH).read_bytes(),
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
    )


def build_confirm20_label_blind_bundle(
    payloads: ValidatedConfirm20Payloads,
) -> Confirm20LabelBlindBundle:
    """Build features and fixed-budget baselines without accepting any D(S) input."""
    if not isinstance(payloads, ValidatedConfirm20Payloads) or payloads._seal is not _PAYLOAD_SEAL:
        raise TypeError("confirm bundle requires strictly validated payloads")
    features: list[FeatureState] = []
    work_items: list[Confirm20PolicyWorkItem] = []
    dynamic_recent: dict[str, tuple[int, ...]] = {}
    ocr_rgb: dict[str, tuple[int, ...]] = {}
    score_records: list[Mapping[str, Any]] = []
    for ordinal, trajectory in enumerate(payloads.trajectories):
        decision = trajectory["decisions"][0]
        state = feature_state_from_derived(
            trajectory,
            decision,
            ocr_records_by_path=payloads.ocr_records_by_path,
        )
        event_by_step = {event["step_id"]: event for event in trajectory["events"]}
        requirements = {
            step: payloads.image_features_by_path[
                event_by_step[step]["observation_after_path"]
            ]
            for step in range(1, CONFIRM_CURRENT_EVENT_STEP_ID + 1)
        }
        current = requirements[CONFIRM_CURRENT_EVENT_STEP_ID]
        work_item = Confirm20PolicyWorkItem(
            ordinal=ordinal,
            source_id=state.source_id,
            state_id=state.state_id,
            decision_step_id=state.decision_step_id,
            event_images=tuple(
                (step, requirements[step].requirement)
                for step in CONFIRM_CANDIDATE_EVENT_STEP_IDS
            ),
            current_image=current.requirement,
        )
        selection = ocr_rgb_similarity_selection(
            event_step_ids=CONFIRM_CANDIDATE_EVENT_STEP_IDS,
            event_ocr_tokens={
                step: requirements[step].full_spatial_tokens
                for step in CONFIRM_CANDIDATE_EVENT_STEP_IDS
            },
            current_ocr_tokens=current.full_spatial_tokens,
            event_resized_rgb_bytes={
                step: requirements[step].resized_rgb_bytes
                for step in CONFIRM_CANDIDATE_EVENT_STEP_IDS
            },
            current_resized_rgb_bytes=current.resized_rgb_bytes,
            budget_event_capacity=CONFIRM_BUDGET_EVENT_CAPACITY,
        )
        features.append(state)
        work_items.append(work_item)
        dynamic_recent[state.state_id] = recent_selection(
            CONFIRM_CANDIDATE_EVENT_STEP_IDS,
            budget_event_capacity=CONFIRM_BUDGET_EVENT_CAPACITY,
        )
        ocr_rgb[state.state_id] = selection.selected_event_step_ids
        score_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "ordinal": ordinal,
                "source_id": state.source_id,
                "state_id": state.state_id,
                "decision_step_id": state.decision_step_id,
                "candidate_event_step_ids": list(CONFIRM_CANDIDATE_EVENT_STEP_IDS),
                "scores_by_event_step": [
                    {"event_step_id": step, "score": score}
                    for step, score in selection.scores_by_event_step
                ],
                "ranked_event_step_ids": list(selection.ranked_event_step_ids),
                "selected_event_step_ids": list(selection.selected_event_step_ids),
            }
        )
    if (
        tuple(state.source_id for state in features) != CONFIRM_SOURCE_IDS
        or len(features) != CONFIRM_STATE_COUNT
        or len(work_items) != CONFIRM_STATE_COUNT
        or any(
            item.ordinal != ordinal
            or item.candidate_event_step_ids != CONFIRM_CANDIDATE_EVENT_STEP_IDS
            for ordinal, item in enumerate(work_items)
        )
    ):
        raise ValueError("confirm feature/work-item denominator or order drifted")
    return Confirm20LabelBlindBundle(
        feature_states=tuple(features),
        work_items=tuple(work_items),
        image_requirements=payloads.image_requirements,
        image_payloads_by_path=payloads.image_payloads_by_path,
        dynamic_recent=MappingProxyType(dynamic_recent),
        ocr_rgb_v2=MappingProxyType(ocr_rgb),
        ocr_rgb_score_records=tuple(score_records),
    )


__all__ = [
    "CONFIRM_BUDGET_EVENT_CAPACITY",
    "CONFIRM_CANDIDATE_EVENT_STEP_IDS",
    "CONFIRM_DECISION_STEP_ID",
    "CONFIRM_IMAGE_COUNT",
    "CONFIRM_SOURCE_IDS",
    "CONFIRM_SOURCE_IDS_SHA256",
    "CONFIRM_STATE_COUNT",
    "Confirm20ImageFeature",
    "Confirm20ImageRequirement",
    "Confirm20LabelBlindBundle",
    "Confirm20PolicyWorkItem",
    "Confirm20TransportIdentity",
    "FROZEN_TRANSPORT_IDENTITY",
    "ValidatedConfirm20Payloads",
    "build_confirm20_label_blind_bundle",
    "load_frozen_confirm20_artifact",
    "validate_confirm20_payloads",
]
