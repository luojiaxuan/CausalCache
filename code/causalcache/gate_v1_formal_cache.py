"""Selective, deterministic materialization for the formal-58 gate caches."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
import struct
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from causalcache.gate_v1_contract import canonical_json_bytes, sha256_bytes
from causalcache.gate_v1_formal_cache_contract import EXPECTED_SECTION_SHA256
from causalcache.gate_v1_data import (
    CandidateFeatures,
    FeatureState,
    LabelState,
    feature_state_from_derived,
    join_feature_and_label_states,
    label_state_from_restoration_record,
    validate_canonical_gate_state_roster,
)
from causalcache.restoration_v2_2_label_table import (
    deployment_conditional_edges,
    validate_complete_distance_table,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_formal_cache_v1"
SOURCE_STATUS = "source_only_frozen_before_formal_cache_execution"
TRANSPORT_REPAIR_PROTOCOL_ID = "causalcache_gate_v1_formal_cache_transport_repair_v1"
FORMAL_CACHE_SOURCE_CONFIG_SHA256 = (
    "1d7527e8a7bede8aaab8a21f7757f786674530238ae99fe3ce5b196cbce67261"
)
TRANSPORT_REPAIR_INPUT_KEY = "expansion_feature_trajectories"
TRANSPORT_REPAIR_ARTIFACT_NAME = "expansion_derived_features"
TRANSPORT_REPAIR_FILE_PATH = (
    "derived/restoration-v2-label-expansion-v1/trajectories-00000-of-00001.jsonl"
)
TRANSPORT_REPAIR_WRONG_SHA256 = (
    "00fe93e9deeeb9a3c018227fb781b29f6efefbb728db987584b650d9df353a6d"
)
TRANSPORT_REPAIR_CORRECTED_SHA256 = (
    "fe93e9deeeb9a3c018227fb781b29f6efefefbb728db987584b650d9df353a6d"
)
TRANSPORT_REPAIR_SIZE_BYTES = 1245673
FEATURE_STATUS = "VALID_GATE_V1_FORMAL58_FEATURE_CACHE_V1"
LABEL_STATUS = "VALID_GATE_V1_FORMAL58_LABEL_CACHE_V1"
JOIN_AUDIT_STATUS = "VALID_GATE_V1_FORMAL58_JOIN_AUDIT_V1"
GATE_CONFIG_SHA256 = (
    "37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b"
)
FORMAL_SOURCE_IDS_SHA256 = (
    "ccec55afb0882e602f5d2c83ec420682488e2663fe26a7c574a851825c43f225"
)
FEATURE_CACHE_PREFIX = "gate-v1-formal58-feature-cache-v1"
LABEL_CACHE_PREFIX = "gate-v1-formal58-label-cache-v1"

LEGACY_FEATURE_MANIFEST = "legacy_feature_manifest"
LEGACY_FEATURE_TRAJECTORIES = "legacy_feature_trajectories"
LEGACY_FEATURE_OCR = "legacy_feature_ocr"
EXPANSION_FEATURE_MANIFEST = "expansion_feature_manifest"
EXPANSION_FEATURE_TRAJECTORIES = "expansion_feature_trajectories"
EXPANSION_FEATURE_OCR = "expansion_feature_ocr"
LEGACY_LABEL_ARCHIVE = "legacy_label_archive"
EXPANSION_LABEL_ARCHIVE = "expansion_label_archive"
EXPANSION_LABEL_SIDECAR = "expansion_label_sidecar"

FEATURE_SOURCE_KEYS = (
    LEGACY_FEATURE_MANIFEST,
    LEGACY_FEATURE_TRAJECTORIES,
    LEGACY_FEATURE_OCR,
    EXPANSION_FEATURE_MANIFEST,
    EXPANSION_FEATURE_TRAJECTORIES,
    EXPANSION_FEATURE_OCR,
)
LABEL_SOURCE_KEYS = (
    LEGACY_LABEL_ARCHIVE,
    EXPANSION_LABEL_ARCHIVE,
    EXPANSION_LABEL_SIDECAR,
)
DOWNLOAD_KEYS = frozenset((*FEATURE_SOURCE_KEYS, *LABEL_SOURCE_KEYS))

LEGACY_FEATURE_PREFIX = "derived/restoration-v2-v1"
EXPANSION_FEATURE_PREFIX = "derived/restoration-v2-label-expansion-v1"
LEGACY_LABEL_PREFIX = "restoration-v2-2-eager-labels-v2"
EXPANSION_LABEL_PREFIX = (
    "restoration-v2-2-expansion-exact-labels-scientific-repair-v1"
)

EXPECTED_TRAJECTORY_COUNT = 58
EXPECTED_STATE_COUNT = 174
EXPECTED_CANDIDATE_COUNT = 522
EXPECTED_SELECTED_OCR_COUNT = 290
EXPECTED_RAW_DISTANCE_ROW_COUNT = 1624
EXPECTED_CONDITIONAL_TARGET_COUNT = 1682
EXPECTED_INDEPENDENT_TARGET_COUNT = 522

_SHA256 = re.compile(r"[0-9a-f]{64}")
_OCR_PATH = re.compile(rb'"image_member_path":"([A-Za-z0-9._/:+-]+)"')
_F64_HEX = re.compile(r"[0-9a-f]{16}")

_ACCESS_AUDIT = {
    "development_semantic_decode_count": 0,
    "confirm_semantic_decode_count": 0,
    "test_semantic_decode_count": 0,
}
_OPERATION_AUDIT = {
    "training_example_count": 0,
    "optimizer_step_count": 0,
    "model_load_count": 0,
    "model_forward_count": 0,
    "policy_operation_count": 0,
    "oracle_metric_count": 0,
    "oof_metric_count": 0,
    "checkpoint_count": 0,
    "matched_nll_evaluation_count": 0,
    "closed_loop_episode_count": 0,
}

_LEGACY_TRAJECTORY_KEYS = {
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
_EXPANSION_TRAJECTORY_KEYS = _LEGACY_TRAJECTORY_KEYS | {"content_witness_sha256"}
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
_LEGACY_DECISION_KEYS = {
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
_EXPANSION_DECISION_KEYS = {
    "state_id",
    "decision_step_id",
    "history_event_step_ids",
    "candidate_event_step_ids",
    "current_equivalent_event_step_id",
    "current_observation_path",
    "current_observation_sha256",
    "candidate_event_post_states",
    "current_equivalence_witness",
    "content_witness_sha256",
    "current_expert_action_payload_included",
}
_OCR_RECORD_KEYS = {
    "schema_version",
    "backend_id",
    "backend_config_sha256",
    "image_member_path",
    "image_sha256",
    "source_format",
    "source_mode",
    "width",
    "height",
    "exif_present",
    "alpha_extrema",
    "rgb_bytes_sha256",
    "resized_rgb_256x256_sha256",
    "nodes",
    "full_spatial_tokens",
    "full_spatial_tokens_sha256",
    "canonical_ocr_record_sha256",
}

_CACHE_MANIFEST_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "cache_kind",
    "gate_config_sha256",
    "source_ids",
    "source_ids_sha256",
    "counts",
    "source_bindings",
    "access_audit",
    "operation_audit",
    "payload",
}
_FEATURE_STATE_KEYS = {
    "source_id",
    "state_id",
    "decision_step_id",
    "candidate_event_step_ids",
    "q64_f64_hex",
    "candidates",
}
_FEATURE_CANDIDATE_KEYS = {"event_step_id", "h64_f64_hex", "g8_f64_hex"}
_LABEL_STATE_KEYS = {
    "source_id",
    "state_id",
    "decision_step_id",
    "candidate_event_step_ids",
    "distance_rows",
}
_LABEL_ROW_KEYS = {"coalition_event_step_ids", "distance_kl_f64_hex"}
_TRANSPORT_REPAIR_MARKER_KEYS = {
    "protocol_id",
    "parent_config_sha256",
    "input_key",
    "artifact_name",
    "file_path",
    "wrong_sha256",
    "corrected_sha256",
    "size_bytes",
    "data_bytes_changed",
    "semantic_contract_changed",
}


@dataclass(frozen=True)
class FormalCacheArtifact:
    kind: Literal["feature", "label"]
    archive: bytes
    sha256: str
    counts: Mapping[str, int]
    access_audit: Mapping[str, int]
    operation_audit: Mapping[str, int]


@dataclass(frozen=True)
class FormalJoinAudit:
    status: str
    feature_sha256: str
    label_sha256: str
    source_ids_sha256: str
    joined_state_count: int
    counts: Mapping[str, int]
    access_audit: Mapping[str, int]
    operation_audit: Mapping[str, int]


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json_loads(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not strict UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=float,
            parse_int=int,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"{label} is not valid strict JSON") from error


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_pretty_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    value = _json_loads(payload, label=label)
    if not isinstance(value, Mapping) or _pretty_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical pretty JSON")
    return value


def _strict_canonical_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    value = _json_loads(payload, label=label)
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} keys drifted: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _canonical_jsonl_lines(payload: bytes, *, expected_count: int, label: str) -> list[bytes]:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be LF-terminated JSONL")
    lines = payload[:-1].split(b"\n")
    if len(lines) != expected_count or any(not line for line in lines):
        raise ValueError(f"{label} record count or blank-line inventory drifted")
    return lines


def _binding_payload(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    _exact_keys(value, {"sha256", "size_bytes"}, label)
    digest = _require_sha256(value.get("sha256"), f"{label} sha256")
    size = value.get("size_bytes")
    if type(size) is not int or size <= 0:
        raise ValueError(f"{label} size must be a positive integer")
    return {"sha256": digest, "size_bytes": size}


def _verify_downloads(
    downloaded: Mapping[str, bytes],
    bindings: Mapping[str, Mapping[str, Any]],
    *,
    expected_keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    expected = set(expected_keys)
    if set(downloaded) != expected or set(bindings) != expected:
        raise ValueError("formal cache downloaded blob inventory drifted")
    normalized: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        payload = downloaded[name]
        if not isinstance(payload, bytes) or not payload:
            raise ValueError(f"downloaded {name} must be non-empty bytes")
        binding = _binding_payload(bindings[name], label=f"binding {name}")
        if len(payload) != binding["size_bytes"] or sha256_bytes(payload) != binding["sha256"]:
            raise ValueError(f"downloaded {name} transport identity drifted")
        normalized[name] = binding
    return normalized


def _expected_phase_bindings(
    config: Mapping[str, Any], *, kind: Literal["feature", "label"]
) -> dict[str, dict[str, Any]]:
    inputs = config.get("input_artifacts")
    if not isinstance(inputs, Mapping):
        raise ValueError("formal cache input artifact contract is missing")

    def one(artifact: str, path: str, label: str) -> dict[str, Any]:
        value = inputs.get(artifact)
        if not isinstance(value, Mapping) or not isinstance(value.get("files"), list):
            raise ValueError(f"{label} artifact file inventory is missing")
        matches = [
            record
            for record in value["files"]
            if isinstance(record, Mapping) and record.get("path") == path
        ]
        if len(matches) != 1:
            raise ValueError(f"{label} frozen file binding is missing or duplicated")
        record = matches[0]
        return _binding_payload(
            {"sha256": record.get("sha256"), "size_bytes": record.get("size_bytes")},
            label=f"{label} frozen binding",
        )

    if kind == "feature":
        return {
            LEGACY_FEATURE_MANIFEST: one(
                "legacy_derived_features",
                f"{LEGACY_FEATURE_PREFIX}/manifest.json",
                LEGACY_FEATURE_MANIFEST,
            ),
            LEGACY_FEATURE_OCR: one(
                "legacy_derived_features",
                f"{LEGACY_FEATURE_PREFIX}/ocr-records-00000-of-00001.jsonl",
                LEGACY_FEATURE_OCR,
            ),
            LEGACY_FEATURE_TRAJECTORIES: one(
                "legacy_derived_features",
                f"{LEGACY_FEATURE_PREFIX}/trajectories-00000-of-00001.jsonl",
                LEGACY_FEATURE_TRAJECTORIES,
            ),
            EXPANSION_FEATURE_MANIFEST: one(
                "expansion_derived_features",
                f"{EXPANSION_FEATURE_PREFIX}/manifest.json",
                EXPANSION_FEATURE_MANIFEST,
            ),
            EXPANSION_FEATURE_OCR: one(
                "expansion_derived_features",
                f"{EXPANSION_FEATURE_PREFIX}/ocr-records-00000-of-00001.jsonl",
                EXPANSION_FEATURE_OCR,
            ),
            EXPANSION_FEATURE_TRAJECTORIES: one(
                "expansion_derived_features",
                f"{EXPANSION_FEATURE_PREFIX}/trajectories-00000-of-00001.jsonl",
                EXPANSION_FEATURE_TRAJECTORIES,
            ),
        }
    return {
        LEGACY_LABEL_ARCHIVE: one(
            "legacy_restoration_labels",
            "raw/v2.2-eager-train-dev-exact-v2.tar",
            LEGACY_LABEL_ARCHIVE,
        ),
        EXPANSION_LABEL_ARCHIVE: one(
            "repaired_expansion_restoration_labels",
            "repaired/v2.2-expansion-exact-labels-scientific-repair-v1/repaired-labels-v1.tar",
            EXPANSION_LABEL_ARCHIVE,
        ),
        EXPANSION_LABEL_SIDECAR: one(
            "repaired_expansion_restoration_labels",
            "repaired/v2.2-expansion-exact-labels-scientific-repair-v1/artifact-manifest-v1.json",
            EXPANSION_LABEL_SIDECAR,
        ),
    }


def _validate_base_source_contract(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("protocol_id") != PROTOCOL_ID
        or config.get("status") != SOURCE_STATUS
    ):
        raise ValueError("formal cache source contract identity/status drifted")
    expected_top = {"schema_version", "protocol_id", "status", *EXPECTED_SECTION_SHA256}
    _exact_keys(config, expected_top, "formal cache source contract")
    for name, digest in EXPECTED_SECTION_SHA256.items():
        if sha256_bytes(canonical_json_bytes(config[name])) != digest:
            raise ValueError(f"formal cache source contract section {name} drifted")


def _transport_repair_marker(config: Mapping[str, Any]) -> Mapping[str, Any]:
    source_freeze = config.get("source_freeze")
    if not isinstance(source_freeze, Mapping):
        raise ValueError("transport repair source freeze is missing")
    marker = source_freeze.get("transport_repair")
    if not isinstance(marker, Mapping):
        raise ValueError("transport repair marker is missing")
    _exact_keys(marker, _TRANSPORT_REPAIR_MARKER_KEYS, "transport repair marker")
    expected = {
        "protocol_id": TRANSPORT_REPAIR_PROTOCOL_ID,
        "parent_config_sha256": FORMAL_CACHE_SOURCE_CONFIG_SHA256,
        "input_key": TRANSPORT_REPAIR_INPUT_KEY,
        "artifact_name": TRANSPORT_REPAIR_ARTIFACT_NAME,
        "file_path": TRANSPORT_REPAIR_FILE_PATH,
        "wrong_sha256": TRANSPORT_REPAIR_WRONG_SHA256,
        "corrected_sha256": TRANSPORT_REPAIR_CORRECTED_SHA256,
        "size_bytes": TRANSPORT_REPAIR_SIZE_BYTES,
        "data_bytes_changed": False,
        "semantic_contract_changed": False,
    }
    for name, value in expected.items():
        if marker.get(name) != value:
            raise ValueError(f"transport repair marker {name} drifted")
    if type(marker["size_bytes"]) is not int:
        raise ValueError("transport repair marker size type drifted")
    if (
        marker["data_bytes_changed"] is not False
        or marker["semantic_contract_changed"] is not False
    ):
        raise ValueError("transport repair marker boolean flags drifted")
    return marker


def _validate_transport_repair_input_leaf(config: Mapping[str, Any]) -> None:
    inputs = config.get("input_artifacts")
    if not isinstance(inputs, Mapping):
        raise ValueError("transport repair input artifacts are missing")
    normalized = copy.deepcopy(inputs)
    if not isinstance(normalized, dict):
        normalized = dict(normalized)
    artifact = normalized.get(TRANSPORT_REPAIR_ARTIFACT_NAME)
    if not isinstance(artifact, dict):
        raise ValueError("transport repair feature artifact is missing")
    files = artifact.get("files")
    if not isinstance(files, list):
        raise ValueError("transport repair feature file inventory is missing")
    matches = [
        record
        for record in files
        if isinstance(record, dict)
        and record.get("path") == TRANSPORT_REPAIR_FILE_PATH
    ]
    if len(matches) != 1:
        raise ValueError("transport repair feature file binding is missing or duplicated")
    leaf = matches[0]
    if (
        leaf.get("sha256") != TRANSPORT_REPAIR_CORRECTED_SHA256
        or leaf.get("size_bytes") != TRANSPORT_REPAIR_SIZE_BYTES
        or type(leaf.get("size_bytes")) is not int
    ):
        raise ValueError("transport repair corrected feature binding drifted")
    leaf["sha256"] = TRANSPORT_REPAIR_WRONG_SHA256
    if (
        sha256_bytes(canonical_json_bytes(normalized))
        != EXPECTED_SECTION_SHA256["input_artifacts"]
    ):
        raise ValueError("transport repair changed more than the allowed input leaf")


def _validate_transport_repair_source_contract(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("protocol_id") != PROTOCOL_ID
        or config.get("status") != SOURCE_STATUS
    ):
        raise ValueError("transport repair source contract identity/status drifted")
    expected_top = {"schema_version", "protocol_id", "status", *EXPECTED_SECTION_SHA256}
    _exact_keys(config, expected_top, "transport repair source contract")
    _transport_repair_marker(config)
    _validate_transport_repair_input_leaf(config)
    changed_sections = {
        "source_freeze",
        "input_artifacts",
        "cache_formats",
        "local_first_state_machine",
        "destination",
    }
    for name, digest in EXPECTED_SECTION_SHA256.items():
        if name in changed_sections:
            continue
        if sha256_bytes(canonical_json_bytes(config[name])) != digest:
            raise ValueError(
                f"transport repair source contract section {name} drifted"
            )


def _validate_rosters_common(
    config: Mapping[str, Any],
    source_rosters: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    parent = config.get("parent_gate_contract")
    geometry = config.get("formal_geometry")
    formats = config.get("cache_formats")
    if not isinstance(parent, Mapping) or not isinstance(geometry, Mapping) or not isinstance(formats, Mapping):
        raise ValueError("formal cache parent/geometry/format contract is missing")
    if (
        parent.get("protocol_id") != "causalcache_gate_v1_preregistration"
        or parent.get("sha256") != GATE_CONFIG_SHA256
        or parent.get("status")
        != "frozen_before_any_label_expansion_policy_output"
    ):
        raise ValueError("formal cache parent gate binding drifted")
    if not isinstance(source_rosters, Mapping):
        raise ValueError("formal source rosters must be a mapping")
    _exact_keys(
        source_rosters,
        {"legacy_train", "fresh_train_expansion", "formal_train"},
        "formal source rosters",
    )
    values: list[tuple[str, ...]] = []
    for name, count in (
        ("legacy_train", 10),
        ("fresh_train_expansion", 48),
        ("formal_train", EXPECTED_TRAJECTORY_COUNT),
    ):
        raw = source_rosters[name]
        if isinstance(raw, (str, bytes, bytearray, Mapping)) or not isinstance(raw, Sequence):
            raise ValueError(f"{name} roster must be a sequence")
        roster = tuple(raw)
        if (
            len(roster) != count
            or len(set(roster)) != count
            or any(not isinstance(item, str) or not item for item in roster)
        ):
            raise ValueError(f"{name} roster count or identities drifted")
        values.append(roster)
    legacy, expansion, formal = values
    if formal != legacy + expansion or len(set(formal)) != EXPECTED_TRAJECTORY_COUNT:
        raise ValueError("formal roster composition or disjointness drifted")
    observed_digest = sha256_bytes(canonical_json_bytes(list(formal)))
    if (
        geometry.get("roster") != "formal_train"
        or geometry.get("composition")
        != ["legacy_train", "fresh_train_expansion"]
        or geometry.get("trajectory_count") != EXPECTED_TRAJECTORY_COUNT
        or geometry.get("state_count") != EXPECTED_STATE_COUNT
        or geometry.get("candidate_feature_count") != EXPECTED_CANDIDATE_COUNT
        or geometry.get("distance_value_count")
        != EXPECTED_RAW_DISTANCE_ROW_COUNT
        or geometry.get("conditional_edge_count")
        != EXPECTED_CONDITIONAL_TARGET_COUNT
        or geometry.get("source_ids_sha256") != observed_digest
        or observed_digest != FORMAL_SOURCE_IDS_SHA256
    ):
        raise ValueError("formal roster digest drifted")
    common = formats.get("common_ustar")
    feature_format = formats.get("feature")
    label_format = formats.get("label")
    if not all(isinstance(value, Mapping) for value in (common, feature_format, label_format)):
        raise ValueError("formal cache format sections are missing")
    if (
        common.get("format") != "ustar"
        or common.get("regular_file_mode") != 0o644
        or common.get("uid") != 0
        or common.get("gid") != 0
        or common.get("mtime") != 0
        or common.get("uname") != ""
        or common.get("gname") != ""
        or common.get("pax_headers_allowed") is not False
        or common.get("float_hex_pattern") != "[0-9a-f]{16}"
        or feature_format.get("archive_member_prefix") != FEATURE_CACHE_PREFIX
        or feature_format.get("exact_members")
        != ["feature_states.jsonl", "manifest.json"]
        or feature_format.get("manifest_status") != FEATURE_STATUS
        or set(feature_format.get("state_exact_keys", ())) != _FEATURE_STATE_KEYS
        or len(feature_format.get("state_exact_keys", ())) != len(_FEATURE_STATE_KEYS)
        or set(feature_format.get("candidate_exact_keys", ()))
        != _FEATURE_CANDIDATE_KEYS
        or len(feature_format.get("candidate_exact_keys", ()))
        != len(_FEATURE_CANDIDATE_KEYS)
        or label_format.get("archive_member_prefix") != LABEL_CACHE_PREFIX
        or label_format.get("exact_members")
        != ["label_states.jsonl", "manifest.json"]
        or label_format.get("manifest_status") != LABEL_STATUS
        or set(label_format.get("state_exact_keys", ())) != _LABEL_STATE_KEYS
        or len(label_format.get("state_exact_keys", ())) != len(_LABEL_STATE_KEYS)
        or set(label_format.get("distance_exact_keys", ())) != _LABEL_ROW_KEYS
        or len(label_format.get("distance_exact_keys", ())) != len(_LABEL_ROW_KEYS)
    ):
        raise ValueError("formal cache serialization contract drifted")
    return legacy, expansion, formal


def _validate_rosters(
    config: Mapping[str, Any], source_rosters: Mapping[str, Sequence[str]]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    _validate_base_source_contract(config)
    return _validate_rosters_common(config, source_rosters)


def _validate_transport_repair_rosters(
    config: Mapping[str, Any], source_rosters: Mapping[str, Sequence[str]]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    _validate_transport_repair_source_contract(config)
    return _validate_rosters_common(config, source_rosters)


def _validate_selected_trajectory(
    record: Mapping[str, Any],
    *,
    lineage: Literal["legacy", "expansion"],
    expected_source_id: str,
) -> tuple[Mapping[str, Any], set[str]]:
    expected_keys = _LEGACY_TRAJECTORY_KEYS if lineage == "legacy" else _EXPANSION_TRAJECTORY_KEYS
    _exact_keys(record, expected_keys, f"{lineage} selected trajectory")
    role = "v2_label_train" if lineage == "legacy" else "gate_train_expansion"
    if record.get("source_id") != expected_source_id or record.get("role") != role:
        raise ValueError(f"{lineage} selected trajectory identity/order drifted")
    instruction = record.get("instruction")
    if (
        not isinstance(instruction, str)
        or record.get("instruction_sha256") != sha256_bytes(instruction.encode("utf-8"))
    ):
        raise ValueError(f"{lineage} selected instruction witness drifted")
    events = record.get("events")
    decisions = record.get("decisions")
    if not isinstance(events, list) or not isinstance(decisions, list):
        raise ValueError(f"{lineage} selected events/decisions are missing")
    if [event.get("step_id") for event in events if isinstance(event, Mapping)] != [1, 2, 3, 4, 5]:
        raise ValueError(f"{lineage} selected event geometry drifted")
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError(f"{lineage} selected event is not an object")
        _exact_keys(event, _EVENT_KEYS, f"{lineage} selected event")
    expected_decision_keys = (
        _LEGACY_DECISION_KEYS if lineage == "legacy" else _EXPANSION_DECISION_KEYS
    )
    if [decision.get("decision_step_id") for decision in decisions if isinstance(decision, Mapping)] != [4, 5, 6]:
        raise ValueError(f"{lineage} selected decision order drifted")
    required_paths: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError(f"{lineage} selected decision is not an object")
        _exact_keys(decision, expected_decision_keys, f"{lineage} selected decision")
        step = decision["decision_step_id"]
        history = list(range(1, step))
        candidates = history[:-1]
        if (
            decision.get("state_id") != f"{expected_source_id}:decision_step:{step:03d}"
            or decision.get("history_event_step_ids") != history
            or decision.get("candidate_event_step_ids") != candidates
            or decision.get("current_equivalent_event_step_id") != step - 1
            or decision.get("current_observation_path")
            != events[step - 2].get("observation_after_path")
        ):
            raise ValueError(f"{lineage} selected decision geometry drifted")
        if lineage == "expansion" and decision.get("current_expert_action_payload_included") is not False:
            raise ValueError("expansion current expert action payload leaked")
        required_paths.add(str(decision["current_observation_path"]))
        required_paths.update(str(events[event_id - 1]["observation_after_path"]) for event_id in candidates)
    if len(required_paths) != 5:
        raise ValueError(f"{lineage} selected trajectory must require exactly five OCR paths")
    return record, required_paths


def _selected_trajectories(
    payload: bytes,
    *,
    lineage: Literal["legacy", "expansion"],
    selected_ids: Sequence[str],
    full_count: int,
) -> tuple[tuple[Mapping[str, Any], ...], set[str]]:
    lines = _canonical_jsonl_lines(payload, expected_count=full_count, label=f"{lineage} trajectories")
    trajectories: list[Mapping[str, Any]] = []
    paths: set[str] = set()
    for index, source_id in enumerate(selected_ids):
        record = _strict_canonical_object(lines[index], label=f"{lineage} trajectory {index}")
        trajectory, required = _validate_selected_trajectory(
            record, lineage=lineage, expected_source_id=source_id
        )
        trajectories.append(trajectory)
        if paths.intersection(required):
            raise ValueError(f"{lineage} selected trajectories share OCR paths")
        paths.update(required)
    return tuple(trajectories), paths


def _ocr_path_without_json_decode(line: bytes, *, label: str) -> str:
    matches = _OCR_PATH.findall(line)
    if len(matches) != 1:
        raise ValueError(f"{label} lacks one unescaped canonical image_member_path")
    try:
        path = matches[0].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} image path is not ASCII") from error
    if path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError(f"{label} image path is not canonical relative POSIX")
    return path


def _selected_ocr_records(
    payload: bytes,
    *,
    required_paths: set[str],
    full_count: int,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    lines = _canonical_jsonl_lines(payload, expected_count=full_count, label=label)
    observed_paths: list[str] = []
    selected: dict[str, Mapping[str, Any]] = {}
    for index, line in enumerate(lines):
        path = _ocr_path_without_json_decode(line, label=f"{label} row {index}")
        observed_paths.append(path)
        if path not in required_paths:
            continue
        record = _strict_canonical_object(line, label=f"selected {label} row {index}")
        _exact_keys(record, _OCR_RECORD_KEYS, f"selected {label} record")
        if record.get("image_member_path") != path or not isinstance(
            record.get("full_spatial_tokens"), list
        ):
            raise ValueError(f"selected {label} OCR schema drifted")
        if path in selected:
            raise ValueError(f"selected {label} contains duplicate OCR paths")
        selected[path] = record
    if observed_paths != sorted(observed_paths) or len(set(observed_paths)) != len(observed_paths):
        raise ValueError(f"{label} path order or uniqueness drifted")
    if set(selected) != required_paths:
        raise ValueError(f"{label} selected OCR path coverage drifted")
    return selected


def _feature_states_for_lineage(
    trajectories_payload: bytes,
    ocr_payload: bytes,
    *,
    lineage: Literal["legacy", "expansion"],
    source_ids: Sequence[str],
) -> tuple[FeatureState, ...]:
    full_count = 35 if lineage == "legacy" else 64
    full_ocr_count = 210 if lineage == "legacy" else 384
    trajectories, paths = _selected_trajectories(
        trajectories_payload,
        lineage=lineage,
        selected_ids=source_ids,
        full_count=full_count,
    )
    if len(paths) != 5 * len(source_ids):
        raise ValueError(f"{lineage} selected OCR denominator drifted")
    ocr = _selected_ocr_records(
        ocr_payload,
        required_paths=paths,
        full_count=full_ocr_count,
        label=f"{lineage} OCR",
    )
    states: list[FeatureState] = []
    for trajectory in trajectories:
        decisions = trajectory["decisions"]
        for decision in decisions:
            states.append(
                feature_state_from_derived(
                    trajectory,
                    decision,
                    ocr_records_by_path=ocr,
                )
            )
    return tuple(states)


def _expected_legacy_label_members() -> frozenset[str]:
    names = {
        "global_attempt_ledger.json",
        "worker_sibling_ledgers/even.json",
        "worker_sibling_ledgers/odd.json",
        "run_manifest.json",
        "aggregate.json",
    }
    for worker, parity in (("even", 0), ("odd", 1)):
        base = f"workers/{worker}"
        names.update(
            {
                f"{base}/worker_attempt_ledger.json",
                f"{base}/runtime_identity.json",
                f"{base}/terminal.json",
            }
        )
        for index in range(parity, 45, 2):
            names.add(f"{base}/attempts/{index:03d}.json")
            names.add(f"{base}/states/{index:03d}.json")
    return frozenset(names)


def _deterministic_ustar(files: Mapping[str, bytes]) -> bytes:
    destination = io.BytesIO()
    with tarfile.open(fileobj=destination, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(files):
            payload = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return destination.getvalue()


def _read_source_ustar(
    payload: bytes,
    *,
    prefix: str,
    expected_relative_members: frozenset[str],
    label: str,
) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    names: list[str] = []
    prefix_slash = f"{prefix}/"
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            for member in archive.getmembers():
                names.append(member.name)
                if (
                    not member.isfile()
                    or not member.name.startswith(prefix_slash)
                    or member.mode != 0o644
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.pax_headers
                ):
                    raise ValueError(f"{label} USTAR metadata drifted")
                relative = member.name[len(prefix_slash) :]
                if (
                    not relative
                    or relative.startswith("/")
                    or any(part in {"", ".", ".."} for part in relative.split("/"))
                    or relative in files
                ):
                    raise ValueError(f"{label} USTAR member path drifted")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"{label} USTAR member is unreadable")
                value = handle.read()
                if len(value) != member.size:
                    raise ValueError(f"{label} USTAR member size drifted")
                files[relative] = value
    except tarfile.TarError as error:
        raise ValueError(f"{label} is not readable USTAR") from error
    expected_names = [f"{prefix}/{name}" for name in sorted(expected_relative_members)]
    if names != expected_names or set(files) != expected_relative_members:
        raise ValueError(f"{label} USTAR member inventory/order drifted")
    rebuilt = _deterministic_ustar({f"{prefix}/{name}": value for name, value in files.items()})
    if rebuilt != payload:
        raise ValueError(f"{label} is not canonical deterministic USTAR")
    return files


def _full_history_zero(label: LabelState) -> None:
    value = label.table.distance(label.table.event_ids)
    if value != 0.0 or math.copysign(1.0, value) < 0.0:
        raise ValueError("formal label full-history D(S) must be canonical +0")


def _legacy_label_states(payload: bytes, source_ids: Sequence[str]) -> tuple[LabelState, ...]:
    files = _read_source_ustar(
        payload,
        prefix=LEGACY_LABEL_PREFIX,
        expected_relative_members=_expected_legacy_label_members(),
        label="legacy label archive",
    )
    labels: list[LabelState] = []
    for index in range(30):
        source_id = source_ids[index // 3]
        step = 4 + index % 3
        expected_projection = {
            "index": index,
            "role": "v2_label_train",
            "trajectory_id": source_id,
            "decision_step_id": step,
            "state_id": f"{source_id}:decision_step:{step:03d}",
            "candidate_event_step_ids": list(range(1, step - 1)),
        }
        worker = "even" if index % 2 == 0 else "odd"
        record = _strict_pretty_object(
            files[f"workers/{worker}/states/{index:03d}.json"],
            label=f"legacy selected label state {index}",
        )
        if record.get("state") != expected_projection:
            raise ValueError("legacy selected label state projection drifted")
        rows = record.get("distance_rows")
        if not isinstance(rows, list):
            raise ValueError("legacy selected distance rows are missing")
        for row in rows:
            if not isinstance(row, Mapping) or set(row).intersection({"coalition", "distance"}):
                raise ValueError("legacy selected distance row schema drifted")
            if "coalition_event_step_ids" not in row or "distance_kl" not in row:
                raise ValueError("legacy selected distance row fields are missing")
        label = label_state_from_restoration_record(record, record_schema="legacy")
        _full_history_zero(label)
        labels.append(label)
    return tuple(labels)


def _expansion_label_manifest(files: Mapping[str, bytes]) -> Mapping[str, Any]:
    manifest = _strict_pretty_object(files["manifest.json"], label="expansion label manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "runner_config_sha256",
            "source_git_commit",
            "raw_state_count",
            "derived_state_count",
            "payload_inventory",
            "scientific_payload_sha256",
            "original_attempt_remains_invalid",
            "producer_reclassified",
            "formal_consumption",
        },
        "expansion label manifest",
    )
    inventory = [
        {"path": name, "sha256": sha256_bytes(files[name]), "size_bytes": len(files[name])}
        for name in ("audit.json", "derived_labels.jsonl", "raw_states.jsonl")
    ]
    if (
        manifest.get("raw_state_count") != 192
        or manifest.get("derived_state_count") != 192
        or manifest.get("payload_inventory") != inventory
        or manifest.get("original_attempt_remains_invalid") is not True
        or manifest.get("producer_reclassified") is not False
    ):
        raise ValueError("expansion label manifest binding/count drifted")
    return manifest


def _expansion_label_states(payload: bytes, source_ids: Sequence[str]) -> tuple[LabelState, ...]:
    expected_members = frozenset(
        {"audit.json", "derived_labels.jsonl", "manifest.json", "raw_states.jsonl"}
    )
    files = _read_source_ustar(
        payload,
        prefix=EXPANSION_LABEL_PREFIX,
        expected_relative_members=expected_members,
        label="expansion label archive",
    )
    _expansion_label_manifest(files)
    lines = _canonical_jsonl_lines(
        files["raw_states.jsonl"], expected_count=192, label="expansion raw states"
    )
    labels: list[LabelState] = []
    expected_top_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "run_contract_sha256",
        "worker",
        "state",
        "reference_teacher",
        "distance_rows",
        "operation_counts",
    }
    for index in range(144):
        record = _strict_canonical_object(lines[index], label=f"expansion selected label {index}")
        _exact_keys(record, expected_top_keys, "expansion selected label")
        source_id = source_ids[index // 3]
        step = 4 + index % 3
        state = record.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("expansion selected state projection is missing")
        if (
            state.get("state_index") != index
            or state.get("role") != "gate_train_expansion"
            or state.get("source_id") != source_id
            or state.get("decision_step_id") != step
            or state.get("state_id") != f"{source_id}:decision_step:{step:03d}"
            or state.get("candidate_event_step_ids") != list(range(1, step - 1))
        ):
            raise ValueError("expansion selected state roster/order drifted")
        rows = record.get("distance_rows")
        if not isinstance(rows, list):
            raise ValueError("expansion selected distance rows are missing")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("expansion selected distance row is malformed")
            _exact_keys(
                row,
                {
                    "coalition",
                    "distance",
                    "candidate_input_sha256",
                    "teacher_forward_count",
                    "kl_measurement_count",
                    "scalar_host_transfer_count",
                    "is_full_history_reference",
                    "full_logit_tensor_host_transfer_count",
                },
                "expansion selected distance row",
            )
        label = label_state_from_restoration_record(record, record_schema="expansion")
        _full_history_zero(label)
        labels.append(label)
    return tuple(labels)


def _float_to_hex(value: Any, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or (scalar == 0.0 and math.copysign(1.0, scalar) < 0.0):
        raise ValueError(f"{label} must be finite and cannot be negative zero")
    return struct.pack(">d", scalar).hex()


def _hex_to_float(value: Any, *, label: str) -> float:
    if not isinstance(value, str) or _F64_HEX.fullmatch(value) is None:
        raise ValueError(f"{label} must be 16 lowercase big-endian f64 hex digits")
    scalar = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(scalar) or (scalar == 0.0 and math.copysign(1.0, scalar) < 0.0):
        raise ValueError(f"{label} decodes to non-finite or negative zero")
    return scalar


def _feature_record(state: FeatureState) -> dict[str, Any]:
    return {
        "source_id": state.source_id,
        "state_id": state.state_id,
        "decision_step_id": state.decision_step_id,
        "candidate_event_step_ids": list(state.candidate_event_step_ids),
        "q64_f64_hex": [_float_to_hex(value, label="q64") for value in state.q64],
        "candidates": [
            {
                "event_step_id": candidate.event_step_id,
                "h64_f64_hex": [
                    _float_to_hex(value, label="h64") for value in candidate.h64
                ],
                "g8_f64_hex": [
                    _float_to_hex(value, label="g8") for value in candidate.g8
                ],
            }
            for candidate in state.candidates
        ],
    }


def _label_record(state: LabelState) -> dict[str, Any]:
    return {
        "source_id": state.source_id,
        "state_id": state.state_id,
        "decision_step_id": state.decision_step_id,
        "candidate_event_step_ids": list(state.table.event_ids),
        "distance_rows": [
            {
                "coalition_event_step_ids": list(row.coalition),
                "distance_kl_f64_hex": _float_to_hex(row.distance, label="distance"),
            }
            for row in state.table.rows
        ],
    }


def _cache_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _cache_manifest(
    *,
    kind: Literal["feature", "label"],
    source_ids: Sequence[str],
    gate_config_sha256: str,
    counts: Mapping[str, int],
    source_bindings: Mapping[str, Mapping[str, Any]],
    payload_name: str,
    payload: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": FEATURE_STATUS if kind == "feature" else LABEL_STATUS,
        "cache_kind": kind,
        "gate_config_sha256": gate_config_sha256,
        "source_ids": list(source_ids),
        "source_ids_sha256": sha256_bytes(canonical_json_bytes(list(source_ids))),
        "counts": dict(counts),
        "source_bindings": {name: dict(source_bindings[name]) for name in sorted(source_bindings)},
        "access_audit": dict(_ACCESS_AUDIT),
        "operation_audit": dict(_OPERATION_AUDIT),
        "payload": {
            "path": payload_name,
            "record_count": EXPECTED_STATE_COUNT,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        },
    }


def _cache_archive(manifest: Mapping[str, Any], payload_name: str, payload: bytes) -> bytes:
    prefix = (
        FEATURE_CACHE_PREFIX
        if payload_name == "feature_states.jsonl"
        else LABEL_CACHE_PREFIX
    )
    files = {
        f"{prefix}/manifest.json": canonical_json_bytes(manifest) + b"\n",
        f"{prefix}/{payload_name}": payload,
    }
    return _deterministic_ustar(files)


def _read_cache_members(payload: bytes, *, kind: Literal["feature", "label"]) -> dict[str, bytes]:
    payload_name = "feature_states.jsonl" if kind == "feature" else "label_states.jsonl"
    expected = frozenset({"manifest.json", payload_name})
    prefix = FEATURE_CACHE_PREFIX if kind == "feature" else LABEL_CACHE_PREFIX
    files: dict[str, bytes] = {}
    names: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            for member in archive.getmembers():
                names.append(member.name)
                if (
                    not member.isfile()
                    or member.mode != 0o644
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.pax_headers
                    or not member.name.startswith(f"{prefix}/")
                ):
                    raise ValueError(f"{kind} cache USTAR metadata/member drifted")
                relative = member.name[len(prefix) + 1 :]
                if relative not in expected or relative in files:
                    raise ValueError(f"{kind} cache USTAR metadata/member drifted")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"{kind} cache member is unreadable")
                member_payload = handle.read()
                if len(member_payload) != member.size:
                    raise ValueError(f"{kind} cache member size drifted")
                files[relative] = member_payload
    except tarfile.TarError as error:
        raise ValueError(f"{kind} cache is not readable USTAR") from error
    expected_names = sorted(f"{prefix}/{name}" for name in expected)
    rebuilt = _deterministic_ustar(
        {f"{prefix}/{name}": value for name, value in files.items()}
    )
    if names != expected_names or set(files) != expected or rebuilt != payload:
        raise ValueError(f"{kind} cache is not exact deterministic USTAR")
    return files


def _validate_cache_manifest(
    manifest: Mapping[str, Any],
    *,
    kind: Literal["feature", "label"],
    payload_name: str,
    payload: bytes,
) -> tuple[str, ...]:
    _exact_keys(manifest, _CACHE_MANIFEST_KEYS, f"{kind} cache manifest")
    expected_status = FEATURE_STATUS if kind == "feature" else LABEL_STATUS
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != expected_status
        or manifest.get("cache_kind") != kind
        or manifest.get("gate_config_sha256") != GATE_CONFIG_SHA256
    ):
        raise ValueError(f"{kind} cache manifest identity drifted")
    source_ids = manifest.get("source_ids")
    if (
        not isinstance(source_ids, list)
        or len(source_ids) != EXPECTED_TRAJECTORY_COUNT
        or len(set(source_ids)) != EXPECTED_TRAJECTORY_COUNT
        or any(not isinstance(item, str) or not item for item in source_ids)
        or manifest.get("source_ids_sha256")
        != sha256_bytes(canonical_json_bytes(source_ids))
        or manifest.get("source_ids_sha256") != FORMAL_SOURCE_IDS_SHA256
    ):
        raise ValueError(f"{kind} cache source roster/digest drifted")
    if manifest.get("access_audit") != _ACCESS_AUDIT or manifest.get("operation_audit") != _OPERATION_AUDIT:
        raise ValueError(f"{kind} cache access/operation audit drifted")
    source_bindings = manifest.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise ValueError(f"{kind} cache source bindings are missing")
    expected_binding_names = set(FEATURE_SOURCE_KEYS if kind == "feature" else LABEL_SOURCE_KEYS)
    if set(source_bindings) != expected_binding_names:
        raise ValueError(f"{kind} cache physical source separation drifted")
    for name, binding in source_bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"{kind} cache source binding {name} is malformed")
        _binding_payload(binding, label=f"{kind} cache source binding {name}")
    payload_record = manifest.get("payload")
    if not isinstance(payload_record, Mapping):
        raise ValueError(f"{kind} cache payload binding is missing")
    _exact_keys(payload_record, {"path", "record_count", "sha256", "size_bytes"}, f"{kind} payload binding")
    if (
        payload_record.get("path") != payload_name
        or payload_record.get("record_count") != EXPECTED_STATE_COUNT
        or payload_record.get("size_bytes") != len(payload)
        or payload_record.get("sha256") != sha256_bytes(payload)
    ):
        raise ValueError(f"{kind} cache payload identity drifted")
    return tuple(source_ids)


def read_feature_cache(payload: bytes) -> tuple[FeatureState, ...]:
    files = _read_cache_members(payload, kind="feature")
    manifest_bytes = files["manifest.json"]
    if not manifest_bytes.endswith(b"\n"):
        raise ValueError("feature cache manifest lacks LF termination")
    manifest = _strict_canonical_object(manifest_bytes[:-1], label="feature cache manifest")
    state_payload = files["feature_states.jsonl"]
    source_ids = _validate_cache_manifest(
        manifest,
        kind="feature",
        payload_name="feature_states.jsonl",
        payload=state_payload,
    )
    lines = _canonical_jsonl_lines(
        state_payload, expected_count=EXPECTED_STATE_COUNT, label="feature cache states"
    )
    states: list[FeatureState] = []
    for index, line in enumerate(lines):
        record = _strict_canonical_object(line, label=f"feature cache state {index}")
        _exact_keys(record, _FEATURE_STATE_KEYS, "feature cache state")
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("feature cache candidates must be an array")
        candidate_values: list[CandidateFeatures] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("feature cache candidate is malformed")
            _exact_keys(candidate, _FEATURE_CANDIDATE_KEYS, "feature cache candidate")
            h64 = candidate.get("h64_f64_hex")
            g8 = candidate.get("g8_f64_hex")
            if not isinstance(h64, list) or len(h64) != 64 or not isinstance(g8, list) or len(g8) != 8:
                raise ValueError("feature cache candidate dimensions drifted")
            candidate_values.append(
                CandidateFeatures(
                    event_step_id=candidate["event_step_id"],
                    h64=tuple(_hex_to_float(value, label="h64") for value in h64),
                    g8=tuple(_hex_to_float(value, label="g8") for value in g8),
                )
            )
        q64 = record.get("q64_f64_hex")
        candidate_ids = record.get("candidate_event_step_ids")
        if not isinstance(q64, list) or len(q64) != 64 or not isinstance(candidate_ids, list):
            raise ValueError("feature cache state dimensions/geometry drifted")
        states.append(
            FeatureState(
                source_id=record["source_id"],
                state_id=record["state_id"],
                decision_step_id=record["decision_step_id"],
                candidate_event_step_ids=tuple(candidate_ids),
                q64=tuple(_hex_to_float(value, label="q64") for value in q64),
                candidates=tuple(candidate_values),
            )
        )
    if len({state.source_id for state in states}) != EXPECTED_TRAJECTORY_COUNT:
        raise ValueError("feature cache trajectory denominator drifted")
    if sum(len(state.candidates) for state in states) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("feature cache candidate denominator drifted")
    observed_source_order = tuple(dict.fromkeys(state.source_id for state in states))
    if observed_source_order != source_ids:
        raise ValueError("feature cache state/source order drifted")
    expected_counts = {
        "trajectory_semantic_decode_count": EXPECTED_TRAJECTORY_COUNT,
        "feature_state_count": EXPECTED_STATE_COUNT,
        "candidate_feature_count": EXPECTED_CANDIDATE_COUNT,
        "ocr_semantic_decode_count": EXPECTED_SELECTED_OCR_COUNT,
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("feature cache manifest counts drifted")
    _validate_projected_roster(states, source_ids)
    return tuple(states)


def read_label_cache(payload: bytes) -> tuple[LabelState, ...]:
    files = _read_cache_members(payload, kind="label")
    manifest_bytes = files["manifest.json"]
    if not manifest_bytes.endswith(b"\n"):
        raise ValueError("label cache manifest lacks LF termination")
    manifest = _strict_canonical_object(manifest_bytes[:-1], label="label cache manifest")
    state_payload = files["label_states.jsonl"]
    source_ids = _validate_cache_manifest(
        manifest,
        kind="label",
        payload_name="label_states.jsonl",
        payload=state_payload,
    )
    lines = _canonical_jsonl_lines(
        state_payload, expected_count=EXPECTED_STATE_COUNT, label="label cache states"
    )
    labels: list[LabelState] = []
    for index, line in enumerate(lines):
        record = _strict_canonical_object(line, label=f"label cache state {index}")
        _exact_keys(record, _LABEL_STATE_KEYS, "label cache state")
        candidate_ids = record.get("candidate_event_step_ids")
        rows = record.get("distance_rows")
        if not isinstance(candidate_ids, list) or not isinstance(rows, list):
            raise ValueError("label cache state geometry is malformed")
        distances: dict[tuple[int, ...], float] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("label cache distance row is malformed")
            _exact_keys(row, _LABEL_ROW_KEYS, "label cache distance row")
            coalition = row.get("coalition_event_step_ids")
            if not isinstance(coalition, list):
                raise ValueError("label cache coalition is malformed")
            key = tuple(coalition)
            if key in distances:
                raise ValueError("label cache has duplicate coalition rows")
            distances[key] = _hex_to_float(
                row.get("distance_kl_f64_hex"), label="distance"
            )
        table = validate_complete_distance_table(tuple(candidate_ids), distances)
        label = LabelState(
            source_id=record["source_id"],
            state_id=record["state_id"],
            decision_step_id=record["decision_step_id"],
            table=table,
        )
        _full_history_zero(label)
        labels.append(label)
    observed_source_order = tuple(dict.fromkeys(state.source_id for state in labels))
    if observed_source_order != source_ids:
        raise ValueError("label cache state/source order drifted")
    counts = {
        "label_state_semantic_decode_count": len(labels),
        "distance_value_decode_count": sum(len(state.table.rows) for state in labels),
        "conditional_edge_count": sum(
            len(deployment_conditional_edges(state.table)) for state in labels
        ),
        "independent_target_count": sum(len(state.table.event_ids) for state in labels),
    }
    expected_counts = {
        "label_state_semantic_decode_count": EXPECTED_STATE_COUNT,
        "distance_value_decode_count": EXPECTED_RAW_DISTANCE_ROW_COUNT,
        "conditional_edge_count": EXPECTED_CONDITIONAL_TARGET_COUNT,
        "independent_target_count": EXPECTED_INDEPENDENT_TARGET_COUNT,
    }
    if counts != expected_counts or manifest.get("counts") != expected_counts:
        raise ValueError("label cache scientific counts drifted")
    _validate_projected_roster(labels, source_ids)
    return tuple(labels)


def _validate_projected_roster(
    states: Sequence[FeatureState | LabelState], source_ids: Sequence[str]
) -> None:
    expected = tuple(
        (
            source_id,
            f"{source_id}:decision_step:{step:03d}",
            step,
            tuple(range(1, step - 1)),
        )
        for source_id in source_ids
        for step in (4, 5, 6)
    )
    if len(states) != len(expected):
        raise ValueError("formal projected state count drifted")
    for state, identity in zip(states, expected, strict=True):
        source_id, state_id, step, candidate_ids = identity
        observed_candidates = (
            state.candidate_event_step_ids
            if isinstance(state, FeatureState)
            else state.table.event_ids
        )
        observed_feature_candidates = (
            tuple(candidate.event_step_id for candidate in state.candidates)
            if isinstance(state, FeatureState)
            else candidate_ids
        )
        if (
            state.source_id != source_id
            or state.state_id != state_id
            or state.decision_step_id != step
            or observed_candidates != candidate_ids
            or observed_feature_candidates != candidate_ids
        ):
            raise ValueError("formal projected state roster/order drifted")


def _materialize_formal_feature_cache(
    downloaded_features: Mapping[str, bytes],
    *,
    normalized_bindings: Mapping[str, Mapping[str, Any]],
    legacy_ids: Sequence[str],
    expansion_ids: Sequence[str],
    formal_ids: Sequence[str],
) -> FormalCacheArtifact:
    # note (luojiaxuan): Feature manifests share development source IDs. Their
    # frozen transport identities are verified above, but their JSON is never
    # decoded in the train-only process; selected rows prove train order again.
    features = (
        *_feature_states_for_lineage(
            downloaded_features[LEGACY_FEATURE_TRAJECTORIES],
            downloaded_features[LEGACY_FEATURE_OCR],
            lineage="legacy",
            source_ids=legacy_ids,
        ),
        *_feature_states_for_lineage(
            downloaded_features[EXPANSION_FEATURE_TRAJECTORIES],
            downloaded_features[EXPANSION_FEATURE_OCR],
            lineage="expansion",
            source_ids=expansion_ids,
        ),
    )
    _validate_projected_roster(features, formal_ids)
    feature_counts = {
        "trajectory_semantic_decode_count": EXPECTED_TRAJECTORY_COUNT,
        "feature_state_count": EXPECTED_STATE_COUNT,
        "candidate_feature_count": EXPECTED_CANDIDATE_COUNT,
        "ocr_semantic_decode_count": EXPECTED_SELECTED_OCR_COUNT,
    }
    if len(features) != EXPECTED_STATE_COUNT or feature_counts["candidate_feature_count"] != sum(
        len(state.candidates) for state in features
    ):
        raise ValueError("formal feature denominator drifted")
    feature_payload = _cache_jsonl([_feature_record(state) for state in features])
    feature_manifest = _cache_manifest(
        kind="feature",
        source_ids=formal_ids,
        gate_config_sha256=GATE_CONFIG_SHA256,
        counts=feature_counts,
        source_bindings={name: normalized_bindings[name] for name in FEATURE_SOURCE_KEYS},
        payload_name="feature_states.jsonl",
        payload=feature_payload,
    )
    feature_archive = _cache_archive(
        feature_manifest, "feature_states.jsonl", feature_payload
    )
    read_features = read_feature_cache(feature_archive)
    _validate_projected_roster(read_features, formal_ids)
    return FormalCacheArtifact(
        kind="feature",
        archive=feature_archive,
        sha256=sha256_bytes(feature_archive),
        counts=feature_counts,
        access_audit=dict(_ACCESS_AUDIT),
        operation_audit=dict(_OPERATION_AUDIT),
    )


def build_formal_feature_cache(
    downloaded_features: Mapping[str, bytes],
    *,
    transport_bindings: Mapping[str, Mapping[str, Any]],
    frozen_rosters: Mapping[str, Sequence[str]],
    frozen_config: Mapping[str, Any],
) -> FormalCacheArtifact:
    """Build the v1 feature cache without accepting any label bytes."""
    normalized_bindings = _verify_downloads(
        downloaded_features,
        transport_bindings,
        expected_keys=FEATURE_SOURCE_KEYS,
    )
    legacy_ids, expansion_ids, formal_ids = _validate_rosters(
        frozen_config, frozen_rosters
    )
    if normalized_bindings != _expected_phase_bindings(
        frozen_config, kind="feature"
    ):
        raise ValueError("feature transport bindings differ from frozen source contract")
    return _materialize_formal_feature_cache(
        downloaded_features,
        normalized_bindings=normalized_bindings,
        legacy_ids=legacy_ids,
        expansion_ids=expansion_ids,
        formal_ids=formal_ids,
    )


def build_formal_feature_cache_transport_repair_v1(
    downloaded_features: Mapping[str, bytes],
    *,
    transport_bindings: Mapping[str, Mapping[str, Any]],
    frozen_rosters: Mapping[str, Sequence[str]],
    frozen_config: Mapping[str, Any],
) -> FormalCacheArtifact:
    """Build the isolated transport-repair feature cache under its exact overlay."""
    legacy_ids, expansion_ids, formal_ids = _validate_transport_repair_rosters(
        frozen_config, frozen_rosters
    )
    normalized_bindings = _verify_downloads(
        downloaded_features,
        transport_bindings,
        expected_keys=FEATURE_SOURCE_KEYS,
    )
    if normalized_bindings != _expected_phase_bindings(
        frozen_config, kind="feature"
    ):
        raise ValueError(
            "transport repair feature bindings differ from corrected source contract"
        )
    return _materialize_formal_feature_cache(
        downloaded_features,
        normalized_bindings=normalized_bindings,
        legacy_ids=legacy_ids,
        expansion_ids=expansion_ids,
        formal_ids=formal_ids,
    )


def _materialize_formal_label_cache(
    downloaded_labels: Mapping[str, bytes],
    *,
    normalized_bindings: Mapping[str, Mapping[str, Any]],
    legacy_ids: Sequence[str],
    expansion_ids: Sequence[str],
    formal_ids: Sequence[str],
) -> FormalCacheArtifact:
    labels = (
        *_legacy_label_states(downloaded_labels[LEGACY_LABEL_ARCHIVE], legacy_ids),
        *_expansion_label_states(
            downloaded_labels[EXPANSION_LABEL_ARCHIVE], expansion_ids
        ),
    )
    _validate_projected_roster(labels, formal_ids)
    label_counts = {
        "label_state_semantic_decode_count": len(labels),
        "distance_value_decode_count": sum(len(state.table.rows) for state in labels),
        "conditional_edge_count": sum(
            len(deployment_conditional_edges(state.table)) for state in labels
        ),
        "independent_target_count": sum(
            len(state.table.event_ids) for state in labels
        ),
    }
    expected_label_counts = {
        "label_state_semantic_decode_count": EXPECTED_STATE_COUNT,
        "distance_value_decode_count": EXPECTED_RAW_DISTANCE_ROW_COUNT,
        "conditional_edge_count": EXPECTED_CONDITIONAL_TARGET_COUNT,
        "independent_target_count": EXPECTED_INDEPENDENT_TARGET_COUNT,
    }
    if label_counts != expected_label_counts:
        raise ValueError("formal label denominator drifted")
    label_payload = _cache_jsonl([_label_record(state) for state in labels])
    label_manifest = _cache_manifest(
        kind="label",
        source_ids=formal_ids,
        gate_config_sha256=GATE_CONFIG_SHA256,
        counts=label_counts,
        source_bindings={name: normalized_bindings[name] for name in LABEL_SOURCE_KEYS},
        payload_name="label_states.jsonl",
        payload=label_payload,
    )
    label_archive = _cache_archive(
        label_manifest, "label_states.jsonl", label_payload
    )
    read_labels = read_label_cache(label_archive)
    _validate_projected_roster(read_labels, formal_ids)
    return FormalCacheArtifact(
        kind="label",
        archive=label_archive,
        sha256=sha256_bytes(label_archive),
        counts=label_counts,
        access_audit=dict(_ACCESS_AUDIT),
        operation_audit=dict(_OPERATION_AUDIT),
    )


def build_formal_label_cache(
    downloaded_labels: Mapping[str, bytes],
    *,
    transport_bindings: Mapping[str, Mapping[str, Any]],
    frozen_rosters: Mapping[str, Sequence[str]],
    frozen_config: Mapping[str, Any],
) -> FormalCacheArtifact:
    """Build the v1 label cache after the runner has sealed feature completion."""
    normalized_bindings = _verify_downloads(
        downloaded_labels,
        transport_bindings,
        expected_keys=LABEL_SOURCE_KEYS,
    )
    legacy_ids, expansion_ids, formal_ids = _validate_rosters(
        frozen_config, frozen_rosters
    )
    if normalized_bindings != _expected_phase_bindings(
        frozen_config, kind="label"
    ):
        raise ValueError("label transport bindings differ from frozen source contract")
    return _materialize_formal_label_cache(
        downloaded_labels,
        normalized_bindings=normalized_bindings,
        legacy_ids=legacy_ids,
        expansion_ids=expansion_ids,
        formal_ids=formal_ids,
    )


def build_formal_label_cache_transport_repair_v1(
    downloaded_labels: Mapping[str, bytes],
    *,
    transport_bindings: Mapping[str, Mapping[str, Any]],
    frozen_rosters: Mapping[str, Sequence[str]],
    frozen_config: Mapping[str, Any],
) -> FormalCacheArtifact:
    """Build the isolated transport-repair label cache under its exact overlay."""
    legacy_ids, expansion_ids, formal_ids = _validate_transport_repair_rosters(
        frozen_config, frozen_rosters
    )
    normalized_bindings = _verify_downloads(
        downloaded_labels,
        transport_bindings,
        expected_keys=LABEL_SOURCE_KEYS,
    )
    if normalized_bindings != _expected_phase_bindings(
        frozen_config, kind="label"
    ):
        raise ValueError(
            "transport repair label bindings differ from corrected source contract"
        )
    return _materialize_formal_label_cache(
        downloaded_labels,
        normalized_bindings=normalized_bindings,
        legacy_ids=legacy_ids,
        expansion_ids=expansion_ids,
        formal_ids=formal_ids,
    )


def _audit_formal_cache_join(
    feature_archive: bytes,
    label_archive: bytes,
    *,
    formal_ids: Sequence[str],
) -> FormalJoinAudit:
    if feature_archive == label_archive or sha256_bytes(feature_archive) == sha256_bytes(
        label_archive
    ):
        raise ValueError("formal feature and label caches are not physically distinct")
    features = read_feature_cache(feature_archive)
    labels = read_label_cache(label_archive)
    joined = join_feature_and_label_states(
        features, labels, expected_source_ids=formal_ids
    )
    validate_canonical_gate_state_roster(
        joined, formal_ids, expected_source_count=EXPECTED_TRAJECTORY_COUNT
    )
    return FormalJoinAudit(
        status=JOIN_AUDIT_STATUS,
        feature_sha256=sha256_bytes(feature_archive),
        label_sha256=sha256_bytes(label_archive),
        source_ids_sha256=FORMAL_SOURCE_IDS_SHA256,
        joined_state_count=len(joined),
        counts={
            "trajectory_count": EXPECTED_TRAJECTORY_COUNT,
            "state_count": EXPECTED_STATE_COUNT,
            "candidate_feature_count": EXPECTED_CANDIDATE_COUNT,
            "distance_value_count": EXPECTED_RAW_DISTANCE_ROW_COUNT,
            "conditional_edge_count": EXPECTED_CONDITIONAL_TARGET_COUNT,
            "independent_target_count": EXPECTED_INDEPENDENT_TARGET_COUNT,
        },
        access_audit=dict(_ACCESS_AUDIT),
        operation_audit=dict(_OPERATION_AUDIT),
    )


def audit_formal_cache_join(
    feature_archive: bytes,
    label_archive: bytes,
    *,
    frozen_rosters: Mapping[str, Sequence[str]],
    frozen_config: Mapping[str, Any],
) -> FormalJoinAudit:
    """Join only the two v1 caches after strict read-back."""
    _, _, formal_ids = _validate_rosters(frozen_config, frozen_rosters)
    return _audit_formal_cache_join(
        feature_archive,
        label_archive,
        formal_ids=formal_ids,
    )


def audit_formal_cache_join_transport_repair_v1(
    feature_archive: bytes,
    label_archive: bytes,
    *,
    frozen_rosters: Mapping[str, Sequence[str]],
    frozen_config: Mapping[str, Any],
) -> FormalJoinAudit:
    """Join only the two transport-repair caches after strict read-back."""
    _, _, formal_ids = _validate_transport_repair_rosters(
        frozen_config, frozen_rosters
    )
    return _audit_formal_cache_join(
        feature_archive,
        label_archive,
        formal_ids=formal_ids,
    )


__all__ = [
    "DOWNLOAD_KEYS",
    "EXPANSION_FEATURE_MANIFEST",
    "EXPANSION_FEATURE_OCR",
    "EXPANSION_FEATURE_TRAJECTORIES",
    "EXPANSION_LABEL_ARCHIVE",
    "EXPANSION_LABEL_SIDECAR",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_CONDITIONAL_TARGET_COUNT",
    "EXPECTED_INDEPENDENT_TARGET_COUNT",
    "EXPECTED_RAW_DISTANCE_ROW_COUNT",
    "EXPECTED_STATE_COUNT",
    "EXPECTED_TRAJECTORY_COUNT",
    "FEATURE_SOURCE_KEYS",
    "FormalCacheArtifact",
    "FormalJoinAudit",
    "FEATURE_CACHE_PREFIX",
    "LEGACY_FEATURE_MANIFEST",
    "LEGACY_FEATURE_OCR",
    "LEGACY_FEATURE_TRAJECTORIES",
    "LEGACY_LABEL_ARCHIVE",
    "LABEL_CACHE_PREFIX",
    "LABEL_SOURCE_KEYS",
    "audit_formal_cache_join",
    "audit_formal_cache_join_transport_repair_v1",
    "build_formal_feature_cache",
    "build_formal_feature_cache_transport_repair_v1",
    "build_formal_label_cache",
    "build_formal_label_cache_transport_repair_v1",
    "read_feature_cache",
    "read_label_cache",
]
