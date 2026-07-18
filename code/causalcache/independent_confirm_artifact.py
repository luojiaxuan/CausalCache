"""Canonical artifacts and two-stage publication for independent confirm-20.

The payload stage is intentionally label blind.  The report stage is a direct
child of that payload commit and may contain reference-policy/restoration output.
All remote effects are injected so the state machine can be tested without HF.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.gate_v1_provenance import canonical_selection_sha256
from causalcache.independent_confirm_data import (
    CONFIRM_BUDGET_EVENT_CAPACITY,
    CONFIRM_CANDIDATE_EVENT_STEP_IDS,
    CONFIRM_DECISION_STEP_ID,
    CONFIRM_SOURCE_IDS,
    CONFIRM_STATE_COUNT,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_independent_confirm_closed_loop_v1"
DESTINATION_REPO = "gavinlaw/causalcache-independent-confirm20-mobile"
DESTINATION_REPO_TYPE = "dataset"
DESTINATION_PRIVATE = True
DESTINATION_TAG = "independent-confirm20-v1"
DESTINATION_TAG_MESSAGE = "CausalCache independent confirm-20 v1"

PREFIX = "independent-confirm20/v1"
FEATURE_STATE_PATH = f"{PREFIX}/payload/feature-states-v1.jsonl"
SELECTION_PATHS = MappingProxyType(
    {
        "dynamic_recent": f"{PREFIX}/payload/dynamic-recent-v1.json",
        "ocr_rgb_v2": f"{PREFIX}/payload/ocr-rgb-v2-v1.json",
        "policy_vision_v3": f"{PREFIX}/payload/policy-vision-v3-v1.json",
    }
)
SCORE_PATHS = MappingProxyType(
    {
        "ocr_rgb_v2": f"{PREFIX}/payload/ocr-rgb-v2-scores-v1.jsonl",
        "policy_vision_v3": f"{PREFIX}/payload/policy-vision-v3-scores-v1.jsonl",
    }
)
INDEPENDENT_DECISION_PATH = f"{PREFIX}/payload/independent-decisions-v1.json"
PAYLOAD_MANIFEST_PATH = f"{PREFIX}/payload/payload-manifest-v1.json"
PAYLOAD_TARGETS = (
    FEATURE_STATE_PATH,
    *SELECTION_PATHS.values(),
    *SCORE_PATHS.values(),
    INDEPENDENT_DECISION_PATH,
    PAYLOAD_MANIFEST_PATH,
)

STATE_RECORDS_PATH = f"{PREFIX}/report/raw-state-records-v1.jsonl"
FIXED_REPORT_PATH = f"{PREFIX}/report/fixed-report-v1.json"
RUN_MANIFEST_PATH = f"{PREFIX}/report/run-manifest-v1.json"
BUNDLE_MANIFEST_PATH = f"{PREFIX}/report/bundle-manifest-v1.json"
REPORT_TARGETS = (
    STATE_RECORDS_PATH,
    FIXED_REPORT_PATH,
    RUN_MANIFEST_PATH,
    BUNDLE_MANIFEST_PATH,
)

FEATURE_STATUS = "SEALED_INDEPENDENT_CONFIRM20_FEATURE_STATES_V1"
SELECTION_STATUS = "SEALED_INDEPENDENT_CONFIRM20_SELECTIONS_V1"
SCORE_STATUS = "SEALED_INDEPENDENT_CONFIRM20_SCORES_V1"
DECISION_STATUS = "SEALED_INDEPENDENT_CONFIRM20_INDEPENDENT_DECISIONS_V1"
PAYLOAD_STATUS = "SEALED_INDEPENDENT_CONFIRM20_LABEL_BLIND_PAYLOAD_V1"
STATE_STATUS = "RECORDED_INDEPENDENT_CONFIRM20_RAW_STATE_V1"
RUN_STATUS = "COMPLETED_INDEPENDENT_CONFIRM20_RUN_V1"
BUNDLE_STATUS = "BUNDLED_INDEPENDENT_CONFIRM20_REPORT_V1"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ALLOWED_BASE_FILES = frozenset({".gitattributes", "README.md"})
_SEAL_TOKEN = object()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be LF-terminated bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _strict_jsonl(payload: bytes, *, count: int, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be LF-terminated bytes")
    lines = payload[:-1].split(b"\n")
    if len(lines) != count or any(not line for line in lines):
        raise ValueError(f"{label} row denominator drifted")
    return tuple(_strict_json(line + b"\n", label=f"{label} row") for line in lines)


def _safe_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be one finite scalar")
    result = float(value)
    if not math.isfinite(result) or (result == 0.0 and math.copysign(1.0, result) < 0.0):
        raise ValueError(f"{label} must be finite and cannot be negative zero")
    return result


def _inventory(files: Mapping[str, bytes]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "path": _safe_path(path, label="artifact path"),
            "sha256": sha256_bytes(files[path]),
            "size_bytes": len(files[path]),
        }
        for path in sorted(files)
    )


def _inventory_sha256(inventory: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes([dict(item) for item in inventory]))


def _state_identity(ordinal: int) -> tuple[str, str]:
    source_id = CONFIRM_SOURCE_IDS[ordinal]
    return source_id, f"{source_id}:decision_step:{CONFIRM_DECISION_STEP_ID:03d}"


def feature_states_jsonl_bytes(states: Sequence[FeatureState]) -> bytes:
    values = tuple(states)
    if len(values) != CONFIRM_STATE_COUNT:
        raise ValueError("confirm feature serialization requires exactly 20 states")
    records = []
    for ordinal, state in enumerate(values):
        source_id, state_id = _state_identity(ordinal)
        if (
            not isinstance(state, FeatureState)
            or state.source_id != source_id
            or state.state_id != state_id
            or state.decision_step_id != CONFIRM_DECISION_STEP_ID
            or state.candidate_event_step_ids != CONFIRM_CANDIDATE_EVENT_STEP_IDS
            or tuple(item.event_step_id for item in state.candidates)
            != CONFIRM_CANDIDATE_EVENT_STEP_IDS
        ):
            raise ValueError("confirm feature identity, order, or geometry drifted")
        q64 = [_finite(value, label="q64 value") for value in state.q64]
        if len(q64) != 64:
            raise ValueError("confirm q64 dimension drifted")
        candidates = []
        for candidate in state.candidates:
            h64 = [_finite(value, label="h64 value") for value in candidate.h64]
            g8 = [_finite(value, label="g8 value") for value in candidate.g8]
            if len(h64) != 64 or len(g8) != 8:
                raise ValueError("confirm candidate feature dimension drifted")
            candidates.append(
                {"event_step_id": candidate.event_step_id, "h64": h64, "g8": g8}
            )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": FEATURE_STATUS,
                "ordinal": ordinal,
                "source_id": source_id,
                "state_id": state_id,
                "decision_step_id": CONFIRM_DECISION_STEP_ID,
                "candidate_event_step_ids": list(CONFIRM_CANDIDATE_EVENT_STEP_IDS),
                "q64": q64,
                "candidates": candidates,
            }
        )
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def read_feature_states_jsonl(payload: bytes) -> tuple[FeatureState, ...]:
    records = _strict_jsonl(payload, count=CONFIRM_STATE_COUNT, label="feature states")
    states = []
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "ordinal",
        "source_id",
        "state_id",
        "decision_step_id",
        "candidate_event_step_ids",
        "q64",
        "candidates",
    }
    for ordinal, record in enumerate(records):
        source_id, state_id = _state_identity(ordinal)
        candidates = record.get("candidates")
        if (
            set(record) != expected_keys
            or record.get("schema_version") != SCHEMA_VERSION
            or record.get("protocol_id") != PROTOCOL_ID
            or record.get("status") != FEATURE_STATUS
            or record.get("ordinal") != ordinal
            or record.get("source_id") != source_id
            or record.get("state_id") != state_id
            or record.get("decision_step_id") != CONFIRM_DECISION_STEP_ID
            or record.get("candidate_event_step_ids")
            != list(CONFIRM_CANDIDATE_EVENT_STEP_IDS)
            or not isinstance(record.get("q64"), list)
            or not isinstance(candidates, list)
            or len(candidates) != 4
        ):
            raise ValueError("confirm feature state schema or identity drifted")
        parsed_candidates = []
        for event_step_id, candidate in zip(
            CONFIRM_CANDIDATE_EVENT_STEP_IDS, candidates, strict=True
        ):
            if (
                not isinstance(candidate, Mapping)
                or set(candidate) != {"event_step_id", "h64", "g8"}
                or candidate.get("event_step_id") != event_step_id
                or not isinstance(candidate.get("h64"), list)
                or not isinstance(candidate.get("g8"), list)
            ):
                raise ValueError("confirm candidate feature schema drifted")
            parsed_candidates.append(
                CandidateFeatures(
                    event_step_id=event_step_id,
                    h64=tuple(_finite(value, label="h64 value") for value in candidate["h64"]),
                    g8=tuple(_finite(value, label="g8 value") for value in candidate["g8"]),
                )
            )
        states.append(
            FeatureState(
                source_id=source_id,
                state_id=state_id,
                decision_step_id=CONFIRM_DECISION_STEP_ID,
                candidate_event_step_ids=CONFIRM_CANDIDATE_EVENT_STEP_IDS,
                q64=tuple(_finite(value, label="q64 value") for value in record["q64"]),
                candidates=tuple(parsed_candidates),
            )
        )
    result = tuple(states)
    if feature_states_jsonl_bytes(result) != payload:
        raise ValueError("confirm feature state canonical replay differs")
    return result


def _validate_selected(raw: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or any(type(item) is not int for item in raw):
        raise ValueError(f"{label} selection is malformed")
    selected = tuple(raw)
    if (
        selected != tuple(sorted(selected))
        or len(set(selected)) != len(selected)
        or len(selected) > CONFIRM_BUDGET_EVENT_CAPACITY
        or not set(selected).issubset(CONFIRM_CANDIDATE_EVENT_STEP_IDS)
    ):
        raise ValueError(f"{label} selection is infeasible")
    return selected


def selection_artifact_bytes(name: str, selections: Mapping[str, Sequence[int]]) -> bytes:
    if name not in SELECTION_PATHS or not isinstance(selections, Mapping):
        raise ValueError("confirm selection artifact name or mapping is invalid")
    records = []
    normalized: dict[str, tuple[int, ...]] = {}
    for ordinal in range(CONFIRM_STATE_COUNT):
        _, state_id = _state_identity(ordinal)
        if state_id not in selections:
            raise ValueError("confirm selection state coverage drifted")
        selected = _validate_selected(
            list(selections[state_id]), label=f"{name} state {ordinal}"
        )
        normalized[state_id] = selected
        records.append({"state_id": state_id, "selected_event_step_ids": list(selected)})
    if set(selections) != set(normalized):
        raise ValueError("confirm selection state inventory drifted")
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": SELECTION_STATUS,
        "name": name,
        "selection_sha256": canonical_selection_sha256(normalized),
        "records": records,
    }
    return canonical_json_bytes(value) + b"\n"


def read_selection_artifact(payload: bytes, *, name: str) -> Mapping[str, tuple[int, ...]]:
    value = _strict_json(payload, label=f"{name} selections")
    if (
        set(value)
        != {"schema_version", "protocol_id", "status", "name", "selection_sha256", "records"}
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != SELECTION_STATUS
        or value.get("name") != name
        or name not in SELECTION_PATHS
        or not isinstance(value.get("records"), list)
        or len(value["records"]) != CONFIRM_STATE_COUNT
    ):
        raise ValueError("confirm selection artifact schema drifted")
    result: dict[str, tuple[int, ...]] = {}
    for ordinal, record in enumerate(value["records"]):
        _, state_id = _state_identity(ordinal)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"state_id", "selected_event_step_ids"}
            or record.get("state_id") != state_id
        ):
            raise ValueError("confirm selection record identity drifted")
        result[state_id] = _validate_selected(
            record.get("selected_event_step_ids"), label=name
        )
    if (
        value.get("selection_sha256") != canonical_selection_sha256(result)
        or selection_artifact_bytes(name, result) != payload
    ):
        raise ValueError("confirm selection artifact canonical replay differs")
    return MappingProxyType(result)


def _rank_scores(rows: Sequence[Mapping[str, Any]], *, positive_only: bool) -> tuple[tuple[int, ...], tuple[Mapping[str, Any], ...]]:
    if len(rows) != 4:
        raise ValueError("confirm score vector requires exactly four candidates")
    scores: dict[int, float] = {}
    normalized = []
    for event_step_id, row in zip(CONFIRM_CANDIDATE_EVENT_STEP_IDS, rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"event_step_id", "score"}
            or row.get("event_step_id") != event_step_id
        ):
            raise ValueError("confirm score row identity drifted")
        score = _finite(row.get("score"), label="confirm selector score")
        scores[event_step_id] = score
        normalized.append({"event_step_id": event_step_id, "score": score})
    eligible = [
        event for event in CONFIRM_CANDIDATE_EVENT_STEP_IDS if not positive_only or scores[event] > 0.0
    ]
    ranked = tuple(sorted(eligible, key=lambda event: (-scores[event], event)))
    return ranked, tuple(normalized)


def score_records_jsonl_bytes(
    name: str,
    records: Sequence[Mapping[str, Any]],
) -> bytes:
    if name not in SCORE_PATHS or len(records) != CONFIRM_STATE_COUNT:
        raise ValueError("confirm score artifact name or denominator drifted")
    output = []
    for ordinal, raw in enumerate(records):
        source_id, state_id = _state_identity(ordinal)
        if not isinstance(raw, Mapping):
            raise ValueError("confirm score source record is malformed")
        rows = raw.get("scores_by_event_step")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            raise ValueError("confirm score vector is missing")
        ranked, normalized = _rank_scores(rows, positive_only=False)
        selected = tuple(sorted(ranked[:CONFIRM_BUDGET_EVENT_CAPACITY]))
        if (
            raw.get("source_id") != source_id
            or raw.get("state_id") != state_id
            or raw.get("ordinal") != ordinal
            or raw.get("decision_step_id") != CONFIRM_DECISION_STEP_ID
            or raw.get("candidate_event_step_ids")
            != list(CONFIRM_CANDIDATE_EVENT_STEP_IDS)
            or raw.get("ranked_event_step_ids") != list(ranked)
            or raw.get("selected_event_step_ids") != list(selected)
        ):
            raise ValueError("confirm score record contradicts frozen ranking semantics")
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": SCORE_STATUS,
                "name": name,
                "ordinal": ordinal,
                "source_id": source_id,
                "state_id": state_id,
                "decision_step_id": CONFIRM_DECISION_STEP_ID,
                "candidate_event_step_ids": list(CONFIRM_CANDIDATE_EVENT_STEP_IDS),
                "scores_by_event_step": list(normalized),
                "ranked_event_step_ids": list(ranked),
                "selected_event_step_ids": list(selected),
            }
        )
    return b"".join(canonical_json_bytes(record) + b"\n" for record in output)


def read_score_records_jsonl(payload: bytes, *, name: str) -> tuple[Mapping[str, Any], ...]:
    records = _strict_jsonl(payload, count=CONFIRM_STATE_COUNT, label=f"{name} scores")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "name",
        "ordinal",
        "source_id",
        "state_id",
        "decision_step_id",
        "candidate_event_step_ids",
        "scores_by_event_step",
        "ranked_event_step_ids",
        "selected_event_step_ids",
    }
    for ordinal, record in enumerate(records):
        source_id, state_id = _state_identity(ordinal)
        if (
            set(record) != expected_keys
            or record.get("schema_version") != SCHEMA_VERSION
            or record.get("protocol_id") != PROTOCOL_ID
            or record.get("status") != SCORE_STATUS
            or record.get("name") != name
            or record.get("ordinal") != ordinal
            or record.get("source_id") != source_id
            or record.get("state_id") != state_id
            or record.get("decision_step_id") != CONFIRM_DECISION_STEP_ID
            or record.get("candidate_event_step_ids")
            != list(CONFIRM_CANDIDATE_EVENT_STEP_IDS)
            or not isinstance(record.get("scores_by_event_step"), list)
        ):
            raise ValueError("confirm score artifact schema or identity drifted")
        ranked, _ = _rank_scores(record["scores_by_event_step"], positive_only=False)
        if (
            record.get("ranked_event_step_ids") != list(ranked)
            or _validate_selected(record.get("selected_event_step_ids"), label=name)
            != tuple(sorted(ranked[:CONFIRM_BUDGET_EVENT_CAPACITY]))
        ):
            raise ValueError("confirm score artifact ranking replay differs")
    if score_records_jsonl_bytes(name, records) != payload:
        raise ValueError("confirm score artifact canonical replay differs")
    return records


def independent_decisions_bytes(
    *,
    ensemble_scores: Mapping[str, Mapping[int, float]],
    seed_scores: Sequence[Mapping[str, Mapping[int, float]]],
) -> bytes:
    if not isinstance(ensemble_scores, Mapping) or len(seed_scores) != 5:
        raise ValueError("confirm independent decisions require one ensemble and five seeds")

    def decision(scores_by_event: Mapping[int, float]) -> dict[str, Any]:
        if not isinstance(scores_by_event, Mapping) or set(scores_by_event) != set(
            CONFIRM_CANDIDATE_EVENT_STEP_IDS
        ):
            raise ValueError("confirm independent score vector inventory drifted")
        rows = [
            {"event_step_id": event, "score": scores_by_event[event]}
            for event in CONFIRM_CANDIDATE_EVENT_STEP_IDS
        ]
        ranked, normalized = _rank_scores(rows, positive_only=True)
        selected = tuple(sorted(ranked[:CONFIRM_BUDGET_EVENT_CAPACITY]))
        return {
            "scores_by_event_step": list(normalized),
            "positive_ranked_event_step_ids": list(ranked),
            "selected_event_step_ids": list(selected),
        }

    records = []
    expected_states = set()
    for ordinal in range(CONFIRM_STATE_COUNT):
        _, state_id = _state_identity(ordinal)
        expected_states.add(state_id)
        if state_id not in ensemble_scores or any(state_id not in seed for seed in seed_scores):
            raise ValueError("confirm independent decision state coverage drifted")
        records.append(
            {
                "state_id": state_id,
                "ensemble": decision(ensemble_scores[state_id]),
                "seeds": [
                    {"seed": seed, **decision(seed_scores[seed][state_id])}
                    for seed in range(5)
                ],
            }
        )
    if set(ensemble_scores) != expected_states or any(set(seed) != expected_states for seed in seed_scores):
        raise ValueError("confirm independent decision state inventory drifted")
    selections = {
        record["state_id"]: tuple(record["ensemble"]["selected_event_step_ids"])
        for record in records
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": DECISION_STATUS,
        "family": "independent",
        "seed_ids": list(range(5)),
        "selection_sha256": canonical_selection_sha256(selections),
        "records": records,
    }
    return canonical_json_bytes(value) + b"\n"


def read_independent_decisions(payload: bytes) -> Mapping[str, Any]:
    value = _strict_json(payload, label="independent decisions")
    if (
        set(value)
        != {"schema_version", "protocol_id", "status", "family", "seed_ids", "selection_sha256", "records"}
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != DECISION_STATUS
        or value.get("family") != "independent"
        or value.get("seed_ids") != list(range(5))
        or not isinstance(value.get("records"), list)
        or len(value["records"]) != CONFIRM_STATE_COUNT
    ):
        raise ValueError("independent decision artifact schema drifted")
    ensemble_scores: dict[str, Mapping[int, float]] = {}
    seed_scores: list[dict[str, Mapping[int, float]]] = [dict() for _ in range(5)]

    def parse_decision(raw: Any) -> Mapping[int, float]:
        if (
            not isinstance(raw, Mapping)
            or set(raw)
            != {"scores_by_event_step", "positive_ranked_event_step_ids", "selected_event_step_ids"}
            or not isinstance(raw.get("scores_by_event_step"), list)
        ):
            raise ValueError("independent decision row schema drifted")
        ranked, normalized = _rank_scores(raw["scores_by_event_step"], positive_only=True)
        if (
            raw.get("positive_ranked_event_step_ids") != list(ranked)
            or _validate_selected(raw.get("selected_event_step_ids"), label="independent")
            != tuple(sorted(ranked[:CONFIRM_BUDGET_EVENT_CAPACITY]))
        ):
            raise ValueError("independent decision ranking replay differs")
        return MappingProxyType({row["event_step_id"]: row["score"] for row in normalized})

    selections = {}
    for ordinal, record in enumerate(value["records"]):
        _, state_id = _state_identity(ordinal)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"state_id", "ensemble", "seeds"}
            or record.get("state_id") != state_id
            or not isinstance(record.get("seeds"), list)
            or len(record["seeds"]) != 5
        ):
            raise ValueError("independent decision state identity drifted")
        ensemble_scores[state_id] = parse_decision(record["ensemble"])
        selections[state_id] = tuple(record["ensemble"]["selected_event_step_ids"])
        for seed, seed_record in enumerate(record["seeds"]):
            if (
                not isinstance(seed_record, Mapping)
                or set(seed_record)
                != {"seed", "scores_by_event_step", "positive_ranked_event_step_ids", "selected_event_step_ids"}
                or seed_record.get("seed") != seed
            ):
                raise ValueError("independent seed decision identity drifted")
            seed_scores[seed][state_id] = parse_decision(
                {key: value for key, value in seed_record.items() if key != "seed"}
            )
    if value.get("selection_sha256") != canonical_selection_sha256(selections):
        raise ValueError("independent decision selection digest drifted")
    if independent_decisions_bytes(ensemble_scores=ensemble_scores, seed_scores=seed_scores) != payload:
        raise ValueError("independent decision canonical replay differs")
    return MappingProxyType(
        {"ensemble_scores": MappingProxyType(ensemble_scores), "seed_scores": tuple(MappingProxyType(item) for item in seed_scores)}
    )


def _payload_manifest_bytes(files: Mapping[str, bytes]) -> bytes:
    expected = set(PAYLOAD_TARGETS) - {PAYLOAD_MANIFEST_PATH}
    if set(files) != expected:
        raise ValueError("payload manifest requires all and only seven data files")
    inventory = _inventory(files)
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": PAYLOAD_STATUS,
        "destination": {
            "repo": DESTINATION_REPO,
            "repo_type": DESTINATION_REPO_TYPE,
            "private": DESTINATION_PRIVATE,
            "tag": DESTINATION_TAG,
        },
        "state_count": CONFIRM_STATE_COUNT,
        "candidate_event_count_per_state": len(CONFIRM_CANDIDATE_EVENT_STEP_IDS),
        "seed_count": 5,
        "label_blind": True,
        "reference_policy_output_count": 0,
        "restoration_output_count": 0,
        "inventory": list(inventory),
        "inventory_sha256": _inventory_sha256(inventory),
    }
    return canonical_json_bytes(value) + b"\n"


class LabelBlindPayloadSeal:
    __slots__ = ("_files", "_inventory", "inventory_sha256", "_token")

    def __init__(self, files: Mapping[str, bytes], *, _token: object) -> None:
        if _token is not _SEAL_TOKEN:
            raise TypeError("label-blind seals must be constructed by strict replay")
        copied = {path: bytes(payload) for path, payload in files.items()}
        self._files = MappingProxyType(copied)
        self._inventory = tuple(MappingProxyType(dict(item)) for item in _inventory(copied))
        self.inventory_sha256 = _inventory_sha256(self._inventory)
        self._token = _token

    @property
    def files(self) -> Mapping[str, bytes]:
        return self._files

    @property
    def inventory(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self._inventory)


def seal_label_blind_payload(files: Mapping[str, bytes]) -> LabelBlindPayloadSeal:
    if set(files) != set(PAYLOAD_TARGETS):
        raise ValueError("label-blind payload inventory drifted")
    copied = {path: bytes(payload) for path, payload in files.items()}
    read_feature_states_jsonl(copied[FEATURE_STATE_PATH])
    selections = {
        name: read_selection_artifact(copied[path], name=name)
        for name, path in SELECTION_PATHS.items()
    }
    if any(selected != (3, 4) for selected in selections["dynamic_recent"].values()):
        raise ValueError("dynamic-recent selections contradict frozen n=4/B=2 semantics")
    for name, path in SCORE_PATHS.items():
        records = read_score_records_jsonl(copied[path], name=name)
        replayed = {
            record["state_id"]: tuple(record["selected_event_step_ids"])
            for record in records
        }
        if replayed != dict(selections[name]):
            raise ValueError(f"{name} score/selection artifacts contradict each other")
    read_independent_decisions(copied[INDEPENDENT_DECISION_PATH])
    seven = {path: copied[path] for path in PAYLOAD_TARGETS if path != PAYLOAD_MANIFEST_PATH}
    if copied[PAYLOAD_MANIFEST_PATH] != _payload_manifest_bytes(seven):
        raise ValueError("label-blind payload manifest replay differs")
    manifest = _strict_json(copied[PAYLOAD_MANIFEST_PATH], label="payload manifest")
    if (
        manifest.get("label_blind") is not True
        or manifest.get("reference_policy_output_count") != 0
        or manifest.get("restoration_output_count") != 0
    ):
        raise ValueError("payload manifest does not prove the label-blind barrier")
    return LabelBlindPayloadSeal(copied, _token=_SEAL_TOKEN)


def build_label_blind_payload(
    *,
    feature_states: Sequence[FeatureState],
    selections: Mapping[str, Mapping[str, Sequence[int]]],
    score_records: Mapping[str, Sequence[Mapping[str, Any]]],
    ensemble_scores: Mapping[str, Mapping[int, float]],
    seed_scores: Sequence[Mapping[str, Mapping[int, float]]],
) -> LabelBlindPayloadSeal:
    if set(selections) != set(SELECTION_PATHS) or set(score_records) != set(SCORE_PATHS):
        raise ValueError("label-blind selector family inventory drifted")
    files: dict[str, bytes] = {
        FEATURE_STATE_PATH: feature_states_jsonl_bytes(feature_states),
        **{
            SELECTION_PATHS[name]: selection_artifact_bytes(name, selections[name])
            for name in SELECTION_PATHS
        },
        **{
            SCORE_PATHS[name]: score_records_jsonl_bytes(name, score_records[name])
            for name in SCORE_PATHS
        },
        INDEPENDENT_DECISION_PATH: independent_decisions_bytes(
            ensemble_scores=ensemble_scores, seed_scores=seed_scores
        ),
    }
    files[PAYLOAD_MANIFEST_PATH] = _payload_manifest_bytes(files)
    return seal_label_blind_payload(files)


def state_records_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    if len(records) != CONFIRM_STATE_COUNT:
        raise ValueError("confirm raw state artifact requires exactly 20 records")
    output = []
    for ordinal, record in enumerate(records):
        source_id, state_id = _state_identity(ordinal)
        state = record.get("state") if isinstance(record, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or not isinstance(state, Mapping)
            or state.get("ordinal") != ordinal
            or state.get("source_id") != source_id
            or state.get("state_id") != state_id
        ):
            raise ValueError("confirm raw state identity or order drifted")
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": STATE_STATUS,
                "ordinal": ordinal,
                "source_id": source_id,
                "state_id": state_id,
                "record": dict(record),
            }
        )
    return b"".join(canonical_json_bytes(record) + b"\n" for record in output)


def read_state_records_jsonl(payload: bytes) -> tuple[Mapping[str, Any], ...]:
    envelopes = _strict_jsonl(payload, count=CONFIRM_STATE_COUNT, label="raw state records")
    result = []
    for ordinal, envelope in enumerate(envelopes):
        source_id, state_id = _state_identity(ordinal)
        if (
            set(envelope)
            != {"schema_version", "protocol_id", "status", "ordinal", "source_id", "state_id", "record"}
            or envelope.get("schema_version") != SCHEMA_VERSION
            or envelope.get("protocol_id") != PROTOCOL_ID
            or envelope.get("status") != STATE_STATUS
            or envelope.get("ordinal") != ordinal
            or envelope.get("source_id") != source_id
            or envelope.get("state_id") != state_id
            or not isinstance(envelope.get("record"), Mapping)
            or not isinstance(envelope["record"].get("state"), Mapping)
            or envelope["record"]["state"].get("ordinal") != ordinal
            or envelope["record"]["state"].get("source_id") != source_id
            or envelope["record"]["state"].get("state_id") != state_id
        ):
            raise ValueError("confirm raw state envelope drifted")
        result.append(envelope["record"])
    if state_records_jsonl_bytes(result) != payload:
        raise ValueError("confirm raw state canonical replay differs")
    return tuple(result)


def fixed_report_bytes(report: Mapping[str, Any]) -> bytes:
    if not isinstance(report, Mapping):
        raise ValueError("confirm fixed report must be a mapping")
    expected = {
        "status",
        "evaluation_performed",
        "fixed_state_denominator",
        "reference",
        "bootstrap",
        "metrics",
        "records",
        "gate_checks",
        "go",
    }
    if (
        set(report) != expected
        or report.get("status")
        not in {"GO_TO_PAIRED_CLOSED_LOOP", "NO_GO_REFERENCE_COVERAGE", "NO_GO_INDEPENDENT_CONFIRM", "INVALID_CONFIRM_EXECUTION"}
        or report.get("fixed_state_denominator") != CONFIRM_STATE_COUNT
        or type(report.get("evaluation_performed")) is not bool
        or type(report.get("go")) is not bool
        or not isinstance(report.get("records"), list)
        or len(report["records"]) not in {0, CONFIRM_STATE_COUNT}
    ):
        raise ValueError("confirm fixed report schema or denominator drifted")
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "FIXED_INDEPENDENT_CONFIRM20_REPORT_V1",
        "report": dict(report),
    }
    return canonical_json_bytes(value) + b"\n"


def read_fixed_report(payload: bytes) -> Mapping[str, Any]:
    value = _strict_json(payload, label="fixed report")
    if (
        set(value) != {"schema_version", "protocol_id", "status", "report"}
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "FIXED_INDEPENDENT_CONFIRM20_REPORT_V1"
        or not isinstance(value.get("report"), Mapping)
    ):
        raise ValueError("confirm fixed report envelope drifted")
    if fixed_report_bytes(value["report"]) != payload:
        raise ValueError("confirm fixed report canonical replay differs")
    return value["report"]


def run_manifest_bytes(manifest: Mapping[str, Any], *, payload_commit: str) -> bytes:
    if _COMMIT.fullmatch(payload_commit) is None or not isinstance(manifest, Mapping):
        raise ValueError("run manifest requires one immutable payload commit")
    expected = {
        "source_git_commit",
        "contract_sha256",
        "runtime_metadata",
        "operation_counts",
    }
    if (
        set(manifest) != expected
        or _COMMIT.fullmatch(str(manifest.get("source_git_commit"))) is None
        or _SHA256.fullmatch(str(manifest.get("contract_sha256"))) is None
        or not isinstance(manifest.get("runtime_metadata"), Mapping)
        or not isinstance(manifest.get("operation_counts"), Mapping)
    ):
        raise ValueError("confirm run manifest schema drifted")
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUN_STATUS,
        "payload_commit": payload_commit,
        **dict(manifest),
    }
    return canonical_json_bytes(value) + b"\n"


def read_run_manifest(payload: bytes, *, payload_commit: str) -> Mapping[str, Any]:
    value = _strict_json(payload, label="run manifest")
    if (
        set(value)
        != {"schema_version", "protocol_id", "status", "payload_commit", "source_git_commit", "contract_sha256", "runtime_metadata", "operation_counts"}
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != RUN_STATUS
        or value.get("payload_commit") != payload_commit
    ):
        raise ValueError("confirm run manifest identity drifted")
    body = {key: value[key] for key in ("source_git_commit", "contract_sha256", "runtime_metadata", "operation_counts")}
    if run_manifest_bytes(body, payload_commit=payload_commit) != payload:
        raise ValueError("confirm run manifest canonical replay differs")
    return value


def _bundle_manifest_bytes(
    *,
    payload_seal: LabelBlindPayloadSeal,
    payload_commit: str,
    report_files_without_bundle: Mapping[str, bytes],
) -> bytes:
    if (
        not isinstance(payload_seal, LabelBlindPayloadSeal)
        or payload_seal._token is not _SEAL_TOKEN
        or _COMMIT.fullmatch(payload_commit) is None
        or set(report_files_without_bundle)
        != {STATE_RECORDS_PATH, FIXED_REPORT_PATH, RUN_MANIFEST_PATH}
    ):
        raise ValueError("bundle manifest inputs differ from the frozen report stage")
    report_inventory = _inventory(report_files_without_bundle)
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": BUNDLE_STATUS,
        "payload_commit": payload_commit,
        "payload_inventory": list(payload_seal.inventory),
        "payload_inventory_sha256": payload_seal.inventory_sha256,
        "report_inventory_without_bundle": list(report_inventory),
        "report_inventory_without_bundle_sha256": _inventory_sha256(report_inventory),
    }
    return canonical_json_bytes(value) + b"\n"


def build_report_files(
    *,
    state_records: Sequence[Mapping[str, Any]],
    fixed_report: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    payload_seal: LabelBlindPayloadSeal,
    payload_commit: str,
) -> Mapping[str, bytes]:
    files = {
        STATE_RECORDS_PATH: state_records_jsonl_bytes(state_records),
        FIXED_REPORT_PATH: fixed_report_bytes(fixed_report),
        RUN_MANIFEST_PATH: run_manifest_bytes(run_manifest, payload_commit=payload_commit),
    }
    files[BUNDLE_MANIFEST_PATH] = _bundle_manifest_bytes(
        payload_seal=payload_seal,
        payload_commit=payload_commit,
        report_files_without_bundle=files,
    )
    validate_report_files(files, payload_seal=payload_seal, payload_commit=payload_commit)
    return MappingProxyType(files)


def validate_report_files(
    files: Mapping[str, bytes],
    *,
    payload_seal: LabelBlindPayloadSeal,
    payload_commit: str,
) -> None:
    if set(files) != set(REPORT_TARGETS):
        raise ValueError("confirm report file inventory drifted")
    state_records = read_state_records_jsonl(files[STATE_RECORDS_PATH])
    read_fixed_report(files[FIXED_REPORT_PATH])
    run_manifest = read_run_manifest(
        files[RUN_MANIFEST_PATH], payload_commit=payload_commit
    )
    runtime_metadata = run_manifest.get("runtime_metadata")
    run_contract_sha256 = (
        runtime_metadata.get("run_contract_sha256")
        if isinstance(runtime_metadata, Mapping)
        else None
    )
    if (
        _SHA256.fullmatch(str(run_contract_sha256)) is None
        or any(
            record.get("payload_commit") != payload_commit
            or record.get("run_contract_sha256") != run_contract_sha256
            for record in state_records
        )
    ):
        raise ValueError(
            "confirm raw records differ from payload/run-manifest bindings"
        )
    first_three = {path: files[path] for path in REPORT_TARGETS if path != BUNDLE_MANIFEST_PATH}
    if files[BUNDLE_MANIFEST_PATH] != _bundle_manifest_bytes(
        payload_seal=payload_seal,
        payload_commit=payload_commit,
        report_files_without_bundle=first_three,
    ):
        raise ValueError("confirm bundle manifest replay differs")


@dataclass(frozen=True)
class PayloadPublicationReceipt:
    base_commit: str
    payload_commit: str
    payload_inventory_sha256: str


@dataclass(frozen=True)
class FinalPublicationReceipt:
    base_commit: str
    payload_commit: str
    report_commit: str
    annotated_tag_object: str
    created_tag_count: int


def _response_commit(response: Any) -> str:
    value = getattr(response, "oid", None) or getattr(response, "commit_id", None)
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError("HF commit response lacks an immutable commit id")
    return value


def _operation(factory: Callable[..., Any], path: str, payload: bytes) -> Any:
    try:
        return factory(path_in_repo=path, path_or_fileobj=io.BytesIO(payload))
    except TypeError:
        return factory(path, payload)


def _snapshot(api: Any, *, revision: str | None = None) -> tuple[str, frozenset[str]]:
    info = api.repo_info(
        DESTINATION_REPO,
        repo_type=DESTINATION_REPO_TYPE,
        revision=revision or "main",
    )
    commit = getattr(info, "sha", None)
    if (
        getattr(info, "private", None) is not DESTINATION_PRIVATE
        or not isinstance(commit, str)
        or _COMMIT.fullmatch(commit) is None
        or (revision is not None and commit != revision)
    ):
        raise ValueError("confirm destination privacy or revision drifted")
    files = frozenset(
        api.list_repo_files(
            DESTINATION_REPO,
            repo_type=DESTINATION_REPO_TYPE,
            revision=revision or "main",
        )
    )
    if any(_safe_path(path, label="remote file") != path for path in files):
        raise ValueError("confirm destination file inventory is unsafe")
    return commit, files


def _tag_snapshot(api: Any) -> tuple[str, str] | None:
    refs = api.list_repo_refs(DESTINATION_REPO, repo_type=DESTINATION_REPO_TYPE)
    matches = [item for item in refs.tags if item.name == DESTINATION_TAG]
    if len(matches) > 1:
        raise ValueError("confirm destination tag is duplicated")
    if not matches:
        return None
    tag_object = getattr(matches[0], "target_commit", None)
    resolved = getattr(
        api.repo_info(
            DESTINATION_REPO,
            repo_type=DESTINATION_REPO_TYPE,
            revision=DESTINATION_TAG,
        ),
        "sha",
        None,
    )
    if (
        not isinstance(tag_object, str)
        or _COMMIT.fullmatch(tag_object) is None
        or not isinstance(resolved, str)
        or _COMMIT.fullmatch(resolved) is None
        or tag_object == resolved
    ):
        raise ValueError("confirm destination tag must be annotated")
    return tag_object, resolved


def publish_payload_commit(
    *,
    api: Any,
    operation_factory: Callable[..., Any],
    payload_seal: LabelBlindPayloadSeal,
    expected_base_commit: str | None = None,
) -> PayloadPublicationReceipt:
    if not isinstance(payload_seal, LabelBlindPayloadSeal) or payload_seal._token is not _SEAL_TOKEN:
        raise TypeError("payload publication requires a strict label-blind seal")
    base_commit, initial_files = _snapshot(api)
    if (
        (expected_base_commit is not None and base_commit != expected_base_commit)
        or initial_files - _ALLOWED_BASE_FILES
        or set(initial_files).intersection(PAYLOAD_TARGETS + REPORT_TARGETS)
    ):
        raise ValueError("confirm destination is not an empty non-conflicting base")
    if _tag_snapshot(api) is not None:
        raise ValueError("confirm destination tag already exists before payload publication")
    response = api.create_commit(
        DESTINATION_REPO,
        repo_type=DESTINATION_REPO_TYPE,
        revision="main",
        parent_commit=base_commit,
        commit_message="independent confirm-20 label-blind payload",
        operations=[
            _operation(operation_factory, path, payload_seal.files[path])
            for path in sorted(PAYLOAD_TARGETS)
        ],
    )
    payload_commit = _response_commit(response)
    observed, files = _snapshot(api, revision=payload_commit)
    if observed != payload_commit or files - _ALLOWED_BASE_FILES != set(PAYLOAD_TARGETS):
        raise ValueError("confirm payload commit tree differs from the eight-file stage")
    main, _ = _snapshot(api)
    if main != payload_commit:
        raise ValueError("confirm destination main moved during payload publication")
    return PayloadPublicationReceipt(
        base_commit=base_commit,
        payload_commit=payload_commit,
        payload_inventory_sha256=payload_seal.inventory_sha256,
    )


def _history(api: Any, *, revision: str) -> tuple[tuple[str, str], ...]:
    result = []
    for item in api.list_repo_commits(
        DESTINATION_REPO,
        repo_type=DESTINATION_REPO_TYPE,
        revision=revision,
        formatted=False,
    ):
        commit = getattr(item, "commit_id", None)
        title = getattr(item, "title", None)
        if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None or not isinstance(title, str):
            raise ValueError("confirm destination commit history is malformed")
        result.append((commit, title))
    return tuple(result)


def _fresh_download_replay(
    *,
    download_fn: Callable[..., str],
    revision: str,
    expected: Mapping[str, bytes],
    fresh_parent: Path,
) -> Mapping[str, bytes]:
    if (
        not fresh_parent.is_absolute()
        or not fresh_parent.exists()
        or not fresh_parent.is_dir()
        or fresh_parent.is_symlink()
        or fresh_parent.resolve(strict=True) != fresh_parent
    ):
        raise ValueError("fresh-download parent must be an absolute real directory")
    downloaded = {}
    with tempfile.TemporaryDirectory(prefix="independent-confirm20-", dir=fresh_parent) as raw:
        root = Path(raw).resolve()
        for path in sorted(expected):
            returned = Path(
                download_fn(
                    repo_id=DESTINATION_REPO,
                    repo_type=DESTINATION_REPO_TYPE,
                    filename=path,
                    revision=revision,
                    local_dir=root,
                    force_download=True,
                )
            )
            if not returned.is_absolute() or returned != root / path:
                raise ValueError("fresh download returned a noncanonical path")
            payload = returned.read_bytes()
            if payload != expected[path]:
                raise ValueError(f"fresh-downloaded artifact bytes drifted: {path}")
            downloaded[path] = payload
    return MappingProxyType(downloaded)


def publish_report_commit(
    *,
    api: Any,
    operation_factory: Callable[..., Any],
    download_fn: Callable[..., str],
    payload_seal: LabelBlindPayloadSeal,
    payload_receipt: PayloadPublicationReceipt,
    report_files: Mapping[str, bytes],
    fresh_parent: Path,
) -> FinalPublicationReceipt:
    if payload_receipt.payload_inventory_sha256 != payload_seal.inventory_sha256:
        raise ValueError("payload publication receipt differs from the sealed bytes")
    validate_report_files(
        report_files,
        payload_seal=payload_seal,
        payload_commit=payload_receipt.payload_commit,
    )
    main, files = _snapshot(api)
    if (
        main != payload_receipt.payload_commit
        or files - _ALLOWED_BASE_FILES != set(PAYLOAD_TARGETS)
        or _tag_snapshot(api) is not None
    ):
        raise ValueError("confirm payload stage drifted before report publication")
    response = api.create_commit(
        DESTINATION_REPO,
        repo_type=DESTINATION_REPO_TYPE,
        revision="main",
        parent_commit=payload_receipt.payload_commit,
        commit_message="independent confirm-20 fixed report",
        operations=[
            _operation(operation_factory, path, report_files[path])
            for path in sorted(REPORT_TARGETS)
        ],
    )
    report_commit = _response_commit(response)
    history = _history(api, revision=report_commit)
    expected_history = (
        (report_commit, "independent confirm-20 fixed report"),
        (payload_receipt.payload_commit, "independent confirm-20 label-blind payload"),
    )
    if (
        history[:2] != expected_history
        or len(history) < 3
        or history[2][0] != payload_receipt.base_commit
    ):
        raise ValueError("confirm payload/report commits are not one direct two-commit chain")
    main, final_files = _snapshot(api)
    if (
        main != report_commit
        or final_files - _ALLOWED_BASE_FILES != set(PAYLOAD_TARGETS + REPORT_TARGETS)
    ):
        raise ValueError("confirm final report tree inventory drifted")
    api.create_tag(
        DESTINATION_REPO,
        repo_type=DESTINATION_REPO_TYPE,
        tag=DESTINATION_TAG,
        tag_message=DESTINATION_TAG_MESSAGE,
        revision=report_commit,
        exist_ok=False,
    )
    tag = _tag_snapshot(api)
    if tag is None or tag[1] != report_commit:
        raise ValueError("confirm annotated tag does not dereference to the report commit")
    expected = {**dict(payload_seal.files), **dict(report_files)}
    downloaded = _fresh_download_replay(
        download_fn=download_fn,
        revision=DESTINATION_TAG,
        expected=expected,
        fresh_parent=fresh_parent,
    )
    replayed_seal = seal_label_blind_payload(
        {path: downloaded[path] for path in PAYLOAD_TARGETS}
    )
    validate_report_files(
        {path: downloaded[path] for path in REPORT_TARGETS},
        payload_seal=replayed_seal,
        payload_commit=payload_receipt.payload_commit,
    )
    if _tag_snapshot(api) != tag:
        raise ValueError("confirm annotated tag drifted after fresh download")
    return FinalPublicationReceipt(
        base_commit=payload_receipt.base_commit,
        payload_commit=payload_receipt.payload_commit,
        report_commit=report_commit,
        annotated_tag_object=tag[0],
        created_tag_count=1,
    )


__all__ = [
    "BUNDLE_MANIFEST_PATH",
    "DESTINATION_REPO",
    "DESTINATION_TAG",
    "FEATURE_STATE_PATH",
    "FIXED_REPORT_PATH",
    "FinalPublicationReceipt",
    "INDEPENDENT_DECISION_PATH",
    "LabelBlindPayloadSeal",
    "PAYLOAD_MANIFEST_PATH",
    "PAYLOAD_TARGETS",
    "PayloadPublicationReceipt",
    "REPORT_TARGETS",
    "RUN_MANIFEST_PATH",
    "SCORE_PATHS",
    "SELECTION_PATHS",
    "STATE_RECORDS_PATH",
    "build_label_blind_payload",
    "build_report_files",
    "canonical_json_bytes",
    "feature_states_jsonl_bytes",
    "fixed_report_bytes",
    "independent_decisions_bytes",
    "publish_payload_commit",
    "publish_report_commit",
    "read_feature_states_jsonl",
    "read_fixed_report",
    "read_independent_decisions",
    "read_score_records_jsonl",
    "read_selection_artifact",
    "read_state_records_jsonl",
    "score_records_jsonl_bytes",
    "seal_label_blind_payload",
    "selection_artifact_bytes",
    "sha256_bytes",
    "validate_report_files",
]
