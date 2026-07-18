"""Policy-blind long-horizon GUIOdyssey substrate and exact replay validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.guiodyssey import build_pilot_manifest
from causalcache.data.guiodyssey_independent import (
    source_file_specs,
    verify_local_source_files,
)
from causalcache.data.guiodyssey_restoration_v2 import (
    EVENT_KEYS,
    PreparedImage,
    _build_derived_event,
    _safe_member_path,
    _validate_tar,
    artifact_gitattributes_bytes,
    build_image_tar_bytes,
    build_ocr_backend_provenance,
    generate_ocr_records,
    ocr_record_aggregate_sha256,
    parse_canonical_jsonl,
    sha256_file,
    source_dataset_identity_from_v1_config,
    validate_ocr_record,
    validate_ocr_runtime_identity,
)
from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.independent_closed_loop_features import (
    FROZEN_GATE_V1_OCR_BACKEND,
    feature_state_from_live_history,
)
from causalcache.restoration_v2_text_backend import (
    run_rapidocr_record,
    validate_backend_config,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_long_horizon_development_substrate_v1"
ARTIFACT_ID = "causalcache-long-horizon-development-mobile-v1"
STATUS = "POLICY_BLIND_LONG_HORIZON_DEVELOPMENT_SUBSTRATE_MATERIALIZED"
DATASET_REPO = "gavinlaw/causalcache-long-horizon-development-mobile"
PAYLOAD_PREFIX = "derived/long-horizon-development-v1"
IMAGE_TAR_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/images-00000-of-00001.tar"
OCR_JSONL_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/ocr-records-00000-of-00001.jsonl"
TRAJECTORY_JSONL_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/trajectories-00000-of-00001.jsonl"
FEATURE_JSONL_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/feature-states-00000-of-00001.jsonl"
MANIFEST_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/manifest.json"
GITATTRIBUTES_RELATIVE_PATH = ".gitattributes"
README_RELATIVE_PATH = "README.md"
ARTIFACT_RELATIVE_PATHS = (
    GITATTRIBUTES_RELATIVE_PATH,
    README_RELATIVE_PATH,
    IMAGE_TAR_RELATIVE_PATH,
    OCR_JSONL_RELATIVE_PATH,
    TRAJECTORY_JSONL_RELATIVE_PATH,
    FEATURE_JSONL_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
)
DEVELOPMENT_SPLIT = "development"
RESERVE_SPLIT = "unopened_reserve"
SELECTION_SALT = "causalcache-long-horizon-development-v1"
DEVELOPMENT_TRAJECTORY_COUNT = 24
RESERVE_TRAJECTORY_COUNT = 18
OBSERVATION_COUNT_PER_TRAJECTORY = 18
EVENT_COUNT_PER_TRAJECTORY = 17
STATE_STEPS = (10, 18)
CANDIDATES_BY_STEP = {
    10: tuple(range(1, 9)),
    18: tuple(range(1, 17)),
}
EXPECTED_COUNTS = {
    "trajectory_count": 24,
    "event_count": 408,
    "state_count": 48,
    "candidate_count": 576,
    "image_member_count": 432,
    "ocr_record_count": 432,
}
INPUT_KEYS = {
    "selection_manifest",
    "v1_config",
    "source_file_manifest",
    "ocr_backend_config",
    "ocr_backend_manifest",
}
GENERATOR_KEYS = {
    "git_revision",
    "module_path",
    "module_sha256",
    "build_cli_path",
    "build_cli_sha256",
    "validator_path",
    "validator_sha256",
}
SELECTION_REQUIRED_KEYS = {
    "trajectory_id",
    "shard_path",
    "row_index",
    "decision_count",
    "selection_rank",
    "role",
}
TRAJECTORY_KEYS = {
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
    "content_witness_sha256",
    "events",
    "decisions",
}
DECISION_KEYS = {
    "state_id",
    "decision_step_id",
    "history_event_step_ids",
    "candidate_event_step_ids",
    "current_equivalent_event_step_id",
    "current_observation_path",
    "current_observation_sha256",
    "candidate_event_post_states",
    "current_equivalence_witness",
    "current_expert_action_payload_included",
    "feature_state_sha256",
    "content_witness_sha256",
}
FEATURE_KEYS = {
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
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    if not records:
        raise ValueError("canonical JSONL requires at least one record")
    return b"".join(canonical_json_bytes(dict(record)) + b"\n" for record in records)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def load_json_object(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    payload = Path(path).read_bytes()
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload, value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result) or (
        result == 0.0 and math.copysign(1.0, result) < 0.0
    ):
        raise ValueError(f"{label} must be finite and cannot be negative zero")
    return result


def _split_records(
    selection_manifest: Mapping[str, Any], split_name: str
) -> Sequence[Any]:
    splits = selection_manifest.get("splits")
    if isinstance(splits, Mapping):
        split = splits.get(split_name)
        if isinstance(split, Mapping):
            records = split.get("trajectories")
        else:
            records = split
    else:
        # note (luojiaxuan): Legacy aliases are read-only compatibility for early
        # local fixtures; formal manifests are canonical only under splits.*.
        alias = "reserve" if split_name == RESERVE_SPLIT else split_name
        split = selection_manifest.get(alias)
        records = split.get("trajectories") if isinstance(split, Mapping) else split
    if isinstance(records, (str, bytes, bytearray, Mapping)) or not isinstance(
        records, Sequence
    ):
        raise ValueError(f"selection split is missing trajectories: {split_name}")
    return records


def selection_records(
    selection_manifest: Mapping[str, Any],
    *,
    require_formal: bool = True,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if not isinstance(selection_manifest, Mapping):
        raise TypeError("selection manifest must be a mapping")
    parsed: dict[str, tuple[dict[str, Any], ...]] = {}
    all_ids: list[str] = []
    all_rows: list[tuple[str, int]] = []
    for split_name in (DEVELOPMENT_SPLIT, RESERVE_SPLIT):
        values: list[dict[str, Any]] = []
        split_ranks: list[int] = []
        for raw in _split_records(selection_manifest, split_name):
            if not isinstance(raw, Mapping) or not SELECTION_REQUIRED_KEYS.issubset(
                raw
            ):
                raise ValueError("selection trajectory record schema drifted")
            record = dict(raw)
            source_id = record["trajectory_id"]
            shard_path = record["shard_path"]
            row_index = record["row_index"]
            decision_count = record["decision_count"]
            selection_rank = record["selection_rank"]
            if (
                not isinstance(source_id, str)
                or SOURCE_ID_PATTERN.fullmatch(source_id) is None
                or _safe_member_path(shard_path) != shard_path
                or type(row_index) is not int
                or row_index < 0
                or type(decision_count) is not int
                or not 18 <= decision_count <= 60
                or type(selection_rank) is not int
                or selection_rank < 0
                or record["role"] != split_name
            ):
                raise ValueError("selection trajectory identity or geometry drifted")
            values.append(record)
            split_ranks.append(selection_rank)
            all_ids.append(source_id)
            all_rows.append((shard_path, row_index))
        if split_ranks != sorted(split_ranks) or len(set(split_ranks)) != len(
            split_ranks
        ):
            raise ValueError("selection ranks must be unique and sorted within split")
        parsed[split_name] = tuple(values)
    if len(set(all_ids)) != len(all_ids) or len(set(all_rows)) != len(all_rows):
        raise ValueError("development and unopened reserve selections must be disjoint")
    if require_formal and (
        len(parsed[DEVELOPMENT_SPLIT]) != DEVELOPMENT_TRAJECTORY_COUNT
        or len(parsed[RESERVE_SPLIT]) != RESERVE_TRAJECTORY_COUNT
    ):
        raise ValueError("formal long-horizon selection denominator drifted")
    if require_formal:
        expected_record_keys = {
            DEVELOPMENT_SPLIT: SELECTION_REQUIRED_KEYS
            | {
                "selection_sha256",
                "action_type_counts",
                "normalized_app_labels",
            },
            RESERVE_SPLIT: SELECTION_REQUIRED_KEYS | {"selection_sha256"},
        }
        for split_name in (DEVELOPMENT_SPLIT, RESERVE_SPLIT):
            for record in parsed[split_name]:
                source_id = str(record["trajectory_id"])
                expected_selection_sha = sha256_bytes(
                    f"{SELECTION_SALT}\0{source_id}".encode("utf-8")
                )
                if (
                    set(record) != expected_record_keys[split_name]
                    or record["selection_sha256"] != expected_selection_sha
                ):
                    raise ValueError(
                        f"formal {split_name} selection record schema drifted"
                    )
                if split_name == DEVELOPMENT_SPLIT:
                    action_counts = record["action_type_counts"]
                    app_labels = record["normalized_app_labels"]
                    if (
                        not isinstance(action_counts, Mapping)
                        or not action_counts
                        or any(
                            type(value) is not int or value <= 0
                            for value in action_counts.values()
                        )
                        or sum(action_counts.values()) != record["decision_count"]
                        or not isinstance(app_labels, list)
                        or not app_labels
                        or app_labels != sorted(set(app_labels))
                    ):
                        raise ValueError(
                            "formal development structural summary drifted"
                        )
        ranks = [
            int(record["selection_rank"])
            for split_name in (DEVELOPMENT_SPLIT, RESERVE_SPLIT)
            for record in parsed[split_name]
        ]
        if sorted(ranks) != list(
            range(DEVELOPMENT_TRAJECTORY_COUNT + RESERVE_TRAJECTORY_COUNT)
        ):
            raise ValueError("formal long-horizon selection ranks must cover 0..41")
        splits = selection_manifest.get("splits")
        if not isinstance(splits, Mapping):
            raise ValueError("formal selection requires canonical splits.* schema")
        development_split = splits.get(DEVELOPMENT_SPLIT)
        reserve_split = splits.get(RESERVE_SPLIT)
        if not isinstance(development_split, Mapping) or not isinstance(
            reserve_split, Mapping
        ):
            raise ValueError("formal selection split records are missing")
        states = development_split.get("states")
        reserve_states = reserve_split.get("states")
        if not isinstance(states, list) or reserve_states != []:
            raise ValueError("formal selection state inventory drifted")
        expected_states = []
        for record in parsed[DEVELOPMENT_SPLIT]:
            source_id = str(record["trajectory_id"])
            for step in STATE_STEPS:
                candidates = list(CANDIDATES_BY_STEP[step])
                expected_states.append(
                    {
                        "state_id": f"{source_id}:decision_step:{step:03d}",
                        "trajectory_id": source_id,
                        "decision_step_id": step,
                        "history_event_step_ids": list(range(1, step)),
                        "current_equivalent_event_step_id": step - 1,
                        "candidate_event_count": len(candidates),
                        "candidate_event_step_ids": candidates,
                    }
                )
        if states != expected_states:
            raise ValueError("formal selection state geometry or order drifted")
    return parsed[DEVELOPMENT_SPLIT], parsed[RESERVE_SPLIT]


def _read_selected_parquet_rows(
    source_root: Path,
    wanted_by_file: Mapping[str, Mapping[int, str]],
) -> dict[str, Mapping[str, Any]]:
    import pyarrow.parquet as pq

    rows: dict[str, Mapping[str, Any]] = {}
    for shard_path, wanted in wanted_by_file.items():
        parquet = pq.ParquetFile(source_root.joinpath(*PurePosixPath(shard_path).parts))
        row_group_starts: list[tuple[int, int]] = []
        start = 0
        for group_index in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group_index).num_rows
            row_group_starts.append((start, start + count))
            start += count
        for row_index, source_id in sorted(wanted.items()):
            group = next(
                (
                    index
                    for index, (left, right) in enumerate(row_group_starts)
                    if left <= row_index < right
                ),
                None,
            )
            if group is None:
                raise ValueError("selected row index is outside the Parquet shard")
            local_index = row_index - row_group_starts[group][0]
            table = parquet.read_row_group(group).slice(local_index, 1)
            decoded = table.to_pylist()
            if len(decoded) != 1 or not isinstance(decoded[0], Mapping):
                raise ValueError("selected GUIOdyssey row cannot be decoded")
            rows[source_id] = decoded[0]
    return rows


def load_development_source_pilots(
    *,
    source_root: str | Path,
    source_file_manifest: Mapping[str, Any],
    v1_config: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    derived_repo: str = DATASET_REPO,
    require_formal: bool = True,
    row_reader: Callable[
        [Path, Mapping[str, Mapping[int, str]]],
        dict[str, Mapping[str, Any]],
    ] = _read_selected_parquet_rows,
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    """Reload only development rows; unopened reserve row contents stay untouched."""
    development, _ = selection_records(
        selection_manifest, require_formal=require_formal
    )
    specs = source_file_specs(v1_config, source_file_manifest)
    verify_local_source_files(Path(source_root), specs)
    spec_by_path = {spec.transport_file: spec for spec in specs}
    wanted_by_file: dict[str, dict[int, str]] = defaultdict(dict)
    for selected in development:
        shard_path = str(selected["shard_path"])
        if shard_path not in spec_by_path:
            raise ValueError("selected shard is absent from the pinned source files")
        row_index = int(selected["row_index"])
        if row_index in wanted_by_file[shard_path]:
            raise ValueError("selected development source row is duplicated")
        wanted_by_file[shard_path][row_index] = str(selected["trajectory_id"])
    source_rows = row_reader(Path(source_root), wanted_by_file)
    expected_ids = {str(record["trajectory_id"]) for record in development}
    if set(source_rows) != expected_ids:
        raise ValueError("failed to reload every selected development row")

    source_pool = v1_config["source_pool"]
    grid_size = int(v1_config["policy"]["coordinate_grid_size"])
    pilots: dict[str, dict[str, Any]] = {}
    image_payloads: dict[str, bytes] = {}
    for selected in development:
        source_id = str(selected["trajectory_id"])
        shard_path = str(selected["shard_path"])
        pilot, images = build_pilot_manifest(
            source_rows[source_id],
            row_index=int(selected["row_index"]),
            upstream_repo=str(source_pool["upstream_repo"]),
            upstream_revision=str(source_pool["upstream_revision"]),
            transport_repo=str(source_pool["transport_repo"]),
            transport_revision=str(source_pool["transport_revision"]),
            transport_file=shard_path,
            transport_file_sha256=spec_by_path[shard_path].sha256,
            hf_destination=derived_repo,
            grid_size=grid_size,
        )
        trajectory = pilot["trajectory"]
        if (
            trajectory["source_id"] != source_id
            or len(trajectory["decisions"]) != int(selected["decision_count"])
            or len(trajectory["events"]) < EVENT_COUNT_PER_TRAJECTORY
        ):
            raise ValueError("reloaded development trajectory identity drifted")
        prefix_events = list(trajectory["events"][:EVENT_COUNT_PER_TRAJECTORY])
        prefix_paths = {
            str(event[field])
            for event in prefix_events
            for field in ("observation_before_path", "observation_after_path")
        }
        state_decisions = [
            decision
            for decision in trajectory["decisions"]
            if decision["decision_step_id"] in STATE_STEPS
        ]
        if len(prefix_paths) != OBSERVATION_COUNT_PER_TRAJECTORY or [
            decision["decision_step_id"] for decision in state_decisions
        ] != list(STATE_STEPS):
            raise ValueError("development long-horizon prefix cannot be projected")
        projected_trajectory = {
            key: value
            for key, value in trajectory.items()
            if key not in {"steps", "events", "decisions"}
        }
        projected_trajectory["events"] = prefix_events
        projected_trajectory["decisions"] = state_decisions
        pilots[source_id] = {
            **{key: value for key, value in pilot.items() if key != "trajectory"},
            "trajectory": projected_trajectory,
        }
        for path in sorted(prefix_paths):
            payload = images[path]
            if path in image_payloads:
                raise ValueError("development image paths must be globally unique")
            image_payloads[path] = payload
    return pilots, image_payloads


def required_image_member_paths(
    *,
    pilots_by_source: Mapping[str, Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
    require_formal: bool = True,
) -> tuple[str, ...]:
    development, _ = selection_records(
        selection_manifest, require_formal=require_formal
    )
    expected_ids = [str(record["trajectory_id"]) for record in development]
    if list(pilots_by_source) != expected_ids:
        raise ValueError("development pilots must follow exact selection order")
    paths: set[str] = set()
    for source_id in expected_ids:
        trajectory = pilots_by_source[source_id].get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise ValueError("development pilot trajectory is missing")
        events = trajectory.get("events")
        if not isinstance(events, list) or len(events) < EVENT_COUNT_PER_TRAJECTORY:
            raise ValueError("development pilot lacks the 17-event prefix")
        prefix = events[:EVENT_COUNT_PER_TRAJECTORY]
        if [event.get("step_id") for event in prefix] != list(
            range(1, EVENT_COUNT_PER_TRAJECTORY + 1)
        ):
            raise ValueError("development event prefix drifted")
        source_paths = {
            _safe_member_path(event[field])
            for event in prefix
            for field in ("observation_before_path", "observation_after_path")
        }
        prefix_text = f"images/{source_id}/observation-"
        indices = []
        for path in source_paths:
            parsed = PurePosixPath(path)
            index_text = parsed.stem.removeprefix("observation-")
            if (
                not path.startswith(prefix_text)
                or re.fullmatch(r"[0-9]{3}", index_text) is None
            ):
                raise ValueError("development image naming drifted")
            indices.append(int(index_text))
        if sorted(indices) != list(range(OBSERVATION_COUNT_PER_TRAJECTORY)):
            raise ValueError("development image prefix is not observation 0..17")
        paths.update(source_paths)
    result = tuple(sorted(paths))
    expected_count = len(expected_ids) * OBSERVATION_COUNT_PER_TRAJECTORY
    if len(result) != expected_count:
        raise ValueError("development image inventory denominator drifted")
    return result


def _feature_record(state: FeatureState, *, ordinal: int) -> dict[str, Any]:
    if not isinstance(state, FeatureState):
        raise TypeError("feature record requires FeatureState")
    q64 = [_finite(value, label="q64") for value in state.q64]
    if len(q64) != 64:
        raise ValueError("feature q64 dimension drifted")
    candidates = []
    for candidate in state.candidates:
        h64 = [_finite(value, label="h64") for value in candidate.h64]
        g8 = [_finite(value, label="g8") for value in candidate.g8]
        if len(h64) != 64 or len(g8) != 8:
            raise ValueError("candidate feature dimension drifted")
        candidates.append(
            {"event_step_id": candidate.event_step_id, "h64": h64, "g8": g8}
        )
    if tuple(item["event_step_id"] for item in candidates) != (
        state.candidate_event_step_ids
    ):
        raise ValueError("feature candidate identity drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "LABEL_BLIND_FROZEN_GATE_V1_FEATURE_STATE",
        "ordinal": ordinal,
        "source_id": state.source_id,
        "state_id": state.state_id,
        "decision_step_id": state.decision_step_id,
        "candidate_event_step_ids": list(state.candidate_event_step_ids),
        "q64": q64,
        "candidates": candidates,
    }


def feature_states_jsonl_bytes(states: Sequence[FeatureState]) -> bytes:
    return canonical_jsonl_bytes(
        [
            _feature_record(state, ordinal=ordinal)
            for ordinal, state in enumerate(states)
        ]
    )


def read_feature_states_jsonl(
    payload: bytes, *, require_formal: bool = True
) -> tuple[FeatureState, ...]:
    records = parse_canonical_jsonl(payload, label="long-horizon feature states")
    if require_formal and len(records) != EXPECTED_COUNTS["state_count"]:
        raise ValueError("formal feature-state denominator drifted")
    result: list[FeatureState] = []
    for ordinal, record in enumerate(records):
        candidates = record.get("candidates")
        if (
            set(record) != FEATURE_KEYS
            or record["schema_version"] != SCHEMA_VERSION
            or record["protocol_id"] != PROTOCOL_ID
            or record["status"] != "LABEL_BLIND_FROZEN_GATE_V1_FEATURE_STATE"
            or record["ordinal"] != ordinal
            or not isinstance(candidates, list)
            or not isinstance(record["q64"], list)
        ):
            raise ValueError("feature-state schema or ordering drifted")
        parsed_candidates: list[CandidateFeatures] = []
        for candidate in candidates:
            if (
                not isinstance(candidate, Mapping)
                or set(candidate) != {"event_step_id", "h64", "g8"}
                or not isinstance(candidate["h64"], list)
                or not isinstance(candidate["g8"], list)
            ):
                raise ValueError("candidate feature schema drifted")
            h64 = tuple(_finite(value, label="h64") for value in candidate["h64"])
            g8 = tuple(_finite(value, label="g8") for value in candidate["g8"])
            if len(h64) != 64 or len(g8) != 8:
                raise ValueError("candidate feature dimension drifted")
            parsed_candidates.append(
                CandidateFeatures(
                    event_step_id=int(candidate["event_step_id"]),
                    h64=h64,
                    g8=g8,
                )
            )
        state = FeatureState(
            source_id=str(record["source_id"]),
            state_id=str(record["state_id"]),
            decision_step_id=int(record["decision_step_id"]),
            candidate_event_step_ids=tuple(record["candidate_event_step_ids"]),
            q64=tuple(_finite(value, label="q64") for value in record["q64"]),
            candidates=tuple(parsed_candidates),
        )
        if _feature_record(state, ordinal=ordinal) != record:
            raise ValueError("feature-state canonical object replay drifted")
        result.append(state)
    if feature_states_jsonl_bytes(result) != payload:
        raise ValueError("feature-state canonical JSONL replay drifted")
    return tuple(result)


def _decision_content_witness(decision: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                key: decision[key]
                for key in DECISION_KEYS
                if key != "content_witness_sha256"
            }
        )
    )


def _trajectory_content_witness(record: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "source_id": record["source_id"],
                "instruction_sha256": record["instruction_sha256"],
                "selection": record["selection"],
                "event_sha256": [
                    sha256_bytes(canonical_json_bytes(event))
                    for event in record["events"]
                ],
                "decision_content_witness_sha256": [
                    decision["content_witness_sha256"]
                    for decision in record["decisions"]
                ],
            }
        )
    )


def build_long_horizon_records(
    *,
    pilots_by_source: Mapping[str, Mapping[str, Any]],
    source_image_payloads: Mapping[str, bytes],
    selection_manifest: Mapping[str, Any],
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    require_formal: bool = True,
    ocr_record_validator: Callable[..., PreparedImage] = validate_ocr_record,
    event_builder: Callable[..., Mapping[str, Any]] = _build_derived_event,
) -> tuple[list[dict[str, Any]], tuple[FeatureState, ...], dict[str, bytes]]:
    development, _ = selection_records(
        selection_manifest, require_formal=require_formal
    )
    required_paths = required_image_member_paths(
        pilots_by_source=pilots_by_source,
        selection_manifest=selection_manifest,
        require_formal=require_formal,
    )
    if set(ocr_records_by_path) != set(required_paths):
        raise ValueError("long-horizon OCR inventory drifted")
    try:
        images = {path: source_image_payloads[path] for path in required_paths}
    except KeyError as error:
        raise ValueError("source images are missing a long-horizon member") from error
    prepared: dict[str, PreparedImage] = {}
    for path in required_paths:
        record = ocr_records_by_path[path]
        if record.get("image_member_path") != path:
            raise ValueError("OCR mapping key differs from its member path")
        prepared[path] = ocr_record_validator(
            record,
            image_bytes=images[path],
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
        )

    trajectories: list[dict[str, Any]] = []
    feature_states: list[FeatureState] = []
    for selected in development:
        source_id = str(selected["trajectory_id"])
        pilot = pilots_by_source[source_id]
        trajectory = pilot["trajectory"]
        source_events = {int(event["step_id"]): event for event in trajectory["events"]}
        events = [
            dict(
                event_builder(
                    source_events[step],
                    image_payloads=images,
                    ocr_records_by_path=ocr_records_by_path,
                    prepared_by_path=prepared,
                )
            )
            for step in range(1, EVENT_COUNT_PER_TRAJECTORY + 1)
        ]
        source_decisions = {
            int(decision["decision_step_id"]): decision
            for decision in trajectory["decisions"]
        }
        decisions = []
        for step in STATE_STEPS:
            current_equivalent = step - 1
            candidate_ids = CANDIDATES_BY_STEP[step]
            history_ids = list(range(1, step))
            source_decision = source_decisions.get(step)
            if source_decision is None:
                raise ValueError("long-horizon source decision is missing")
            current_path = _safe_member_path(
                source_decision["current_observation_path"]
            )
            current_sha = sha256_bytes(images[current_path])
            equivalent_event = events[current_equivalent - 1]
            if (
                equivalent_event["observation_after_path"] != current_path
                or equivalent_event["observation_after_sha256"] != current_sha
            ):
                raise ValueError("long-horizon current observation equivalence drifted")
            live_events = [
                {
                    "low_fidelity_v2": events[event_step - 1]["low_fidelity_v2"],
                    "post_ocr_spatial_tokens": ocr_records_by_path[
                        events[event_step - 1]["observation_after_path"]
                    ]["full_spatial_tokens"],
                }
                for event_step in history_ids
            ]
            state_id = f"{source_id}:decision_step:{step:03d}"
            state = feature_state_from_live_history(
                source_id=source_id,
                state_id=state_id,
                decision_step_id=step,
                instruction=str(trajectory["instruction"]),
                current_ocr_spatial_tokens=ocr_records_by_path[current_path][
                    "full_spatial_tokens"
                ],
                history_events=live_events,
                ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
            )
            if state.candidate_event_step_ids != candidate_ids:
                raise ValueError("frozen live-history feature geometry drifted")
            feature_ordinal = len(feature_states)
            feature_record = _feature_record(state, ordinal=feature_ordinal)
            feature_sha = sha256_bytes(canonical_json_bytes(feature_record))
            candidate_witnesses = [
                {
                    "event_step_id": event_step,
                    "post_state_member_path": events[event_step - 1][
                        "observation_after_path"
                    ],
                    "post_state_sha256": events[event_step - 1][
                        "observation_after_sha256"
                    ],
                }
                for event_step in candidate_ids
            ]
            current_witness = {
                "event_step_id": current_equivalent,
                "post_state_member_path": current_path,
                "post_state_sha256": current_sha,
            }
            decision = {
                "state_id": state_id,
                "decision_step_id": step,
                "history_event_step_ids": history_ids,
                "candidate_event_step_ids": list(candidate_ids),
                "current_equivalent_event_step_id": current_equivalent,
                "current_observation_path": current_path,
                "current_observation_sha256": current_sha,
                "candidate_event_post_states": candidate_witnesses,
                "current_equivalence_witness": current_witness,
                "current_expert_action_payload_included": False,
                "feature_state_sha256": feature_sha,
                "content_witness_sha256": "",
            }
            decision["content_witness_sha256"] = _decision_content_witness(decision)
            decisions.append(decision)
            feature_states.append(state)
        instruction = str(trajectory["instruction"])
        record = {
            "source_id": source_id,
            "role": DEVELOPMENT_SPLIT,
            "instruction": instruction,
            "instruction_sha256": sha256_bytes(instruction.encode("utf-8")),
            "platform": trajectory["platform"],
            "apps": list(trajectory["apps"]),
            "device_name": trajectory["device_name"],
            "resolution": trajectory["resolution"],
            "terminal_status": trajectory["terminal_status"],
            "source": dict(pilot["source"]),
            "selection": dict(selected),
            "content_witness_sha256": "",
            "events": events,
            "decisions": decisions,
        }
        record["content_witness_sha256"] = _trajectory_content_witness(record)
        trajectories.append(record)
    return trajectories, tuple(feature_states), images


def artifact_readme_bytes(*, dataset_repo: str = DATASET_REPO) -> bytes:
    return (
        "---\nlicense: cc-by-4.0\ntask_categories:\n"
        "- image-to-text\n- visual-question-answering\n---\n\n"
        "# CausalCache long-horizon development substrate\n\n"
        "Policy-blind deterministic GUIOdyssey substrate for development-only "
        "long-horizon states. It contains no policy or restoration output. The "
        "24 development trajectories contribute observation 0..17, events 1..17, "
        "and decisions 10 and 18 (candidate counts 8 and 16). Unopened-reserve "
        "rows are never decoded or materialized.\n\n"
        "The current expert action payload and its reversible low-entropy hash are "
        "omitted from every decision. Shared event storage must be sliced by each "
        "decision's `history_event_step_ids`. Feature states are exact projections "
        "through the frozen live-history independent-gate adapter.\n\n"
        f"Hugging Face dataset repo: `{dataset_repo}`. Payload prefix: "
        f"`{PAYLOAD_PREFIX}`.\n"
    ).encode("utf-8")


def _file_record(path: str, payload: bytes, **extra: Any) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        **extra,
    }


def _identity_record(record: Mapping[str, Any], *, label: str) -> None:
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} input identity drifted")
    _safe_member_path(record["path"])
    _require_sha256(record["sha256"], f"{label}.sha256")


def _validate_generator(generator: Mapping[str, Any]) -> None:
    if set(generator) != GENERATOR_KEYS:
        raise ValueError("long-horizon generator fields drifted")
    if GIT_SHA_PATTERN.fullmatch(str(generator["git_revision"])) is None:
        raise ValueError("generator Git revision must be immutable")
    for prefix in ("module", "build_cli", "validator"):
        _safe_member_path(generator[f"{prefix}_path"])
        _require_sha256(generator[f"{prefix}_sha256"], f"generator.{prefix}")


def build_payload_manifest(
    *,
    dataset_repo: str,
    trajectories: Sequence[Mapping[str, Any]],
    feature_states: Sequence[FeatureState],
    ocr_records: Sequence[Mapping[str, Any]],
    image_members: Sequence[Mapping[str, Any]],
    payload_files: Sequence[Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    generator: Mapping[str, Any],
    ocr_backend_provenance: Mapping[str, Any],
    ocr_runtime_identity: Mapping[str, Any],
    formal_counts_enforced: bool,
) -> dict[str, Any]:
    if set(inputs) != INPUT_KEYS:
        raise ValueError("long-horizon input inventory drifted")
    for key, value in inputs.items():
        _identity_record(value, label=key)
    _validate_generator(generator)
    development, reserve = selection_records(
        selection_manifest, require_formal=formal_counts_enforced
    )
    counts = {
        "trajectory_count": len(trajectories),
        "event_count": sum(len(record["events"]) for record in trajectories),
        "state_count": len(feature_states),
        "candidate_count": sum(
            len(state.candidate_event_step_ids) for state in feature_states
        ),
        "image_member_count": len(image_members),
        "ocr_record_count": len(ocr_records),
    }
    if formal_counts_enforced and counts != EXPECTED_COUNTS:
        raise ValueError("formal long-horizon payload count mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_id": ARTIFACT_ID,
        "status": STATUS,
        "dataset_repo": dataset_repo,
        "license": "cc-by-4.0",
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "current_expert_action_payload_included": False,
        "unopened_reserve_semantic_content_access_count": 0,
        "source_transport_row_group_overread_possible": True,
        "consumer_must_slice_events_by_history_event_step_ids": True,
        "formal_counts_enforced": formal_counts_enforced,
        "inputs": {key: dict(value) for key, value in inputs.items()},
        "generator": dict(generator),
        "ocr_backend": dict(ocr_backend_provenance),
        "ocr_runtime": dict(ocr_runtime_identity),
        "source_dataset": source_dataset_identity_from_v1_config(
            # note (luojiaxuan): The caller-bound v1 identity is repeated below
            # through trajectory source records and independently checked.
            {
                "source_pool": {
                    key: trajectories[0]["source"][key]
                    for key in (
                        "upstream_repo",
                        "upstream_revision",
                        "transport_repo",
                        "transport_revision",
                        "license",
                    )
                }
            }
        ),
        "development_source_ids": [
            str(record["trajectory_id"]) for record in development
        ],
        "unopened_reserve": {
            "trajectory_count": len(reserve),
            "source_ids_sha256": sha256_bytes(
                canonical_json_bytes(
                    [str(record["trajectory_id"]) for record in reserve]
                )
            ),
            "python_row_materialization_count": 0,
            "semantic_content_access_count": 0,
            "transport_row_group_overread_possible": True,
        },
        "state_geometry": {
            "decision_step_ids": list(STATE_STEPS),
            "candidate_event_step_ids_by_decision": {
                str(step): list(CANDIDATES_BY_STEP[step]) for step in STATE_STEPS
            },
        },
        "counts": counts,
        "inventories": {
            "image_members_sha256": sha256_bytes(
                canonical_json_bytes(list(image_members))
            ),
            "ocr_records_sha256": ocr_record_aggregate_sha256(
                {record["image_member_path"]: record for record in ocr_records}
            ),
            "trajectory_index_sha256": sha256_bytes(
                canonical_json_bytes(
                    [
                        {
                            "source_id": record["source_id"],
                            "content_witness_sha256": record["content_witness_sha256"],
                        }
                        for record in trajectories
                    ]
                )
            ),
            "feature_states_sha256": sha256_bytes(
                feature_states_jsonl_bytes(feature_states)
            ),
        },
        "payload_files": [dict(record) for record in payload_files],
    }


def write_artifact(
    *,
    output_dir: str | Path,
    dataset_repo: str,
    trajectories: Sequence[Mapping[str, Any]],
    feature_states: Sequence[FeatureState],
    image_payloads: Mapping[str, bytes],
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    generator: Mapping[str, Any],
    ocr_backend_provenance: Mapping[str, Any],
    ocr_runtime_identity: Mapping[str, Any],
    formal_counts_enforced: bool = True,
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists() or root.is_symlink() or os.path.lexists(root):
        raise FileExistsError(f"long-horizon output already exists: {root}")
    ocr_records = [ocr_records_by_path[path] for path in sorted(ocr_records_by_path)]
    image_tar, image_members = build_image_tar_bytes(image_payloads)
    ocr_payload = canonical_jsonl_bytes(ocr_records)
    trajectory_payload = canonical_jsonl_bytes(trajectories)
    feature_payload = feature_states_jsonl_bytes(feature_states)
    payload_files = [
        _file_record(
            IMAGE_TAR_RELATIVE_PATH, image_tar, member_count=len(image_members)
        ),
        _file_record(
            OCR_JSONL_RELATIVE_PATH, ocr_payload, record_count=len(ocr_records)
        ),
        _file_record(
            TRAJECTORY_JSONL_RELATIVE_PATH,
            trajectory_payload,
            record_count=len(trajectories),
        ),
        _file_record(
            FEATURE_JSONL_RELATIVE_PATH,
            feature_payload,
            record_count=len(feature_states),
        ),
    ]
    manifest = build_payload_manifest(
        dataset_repo=dataset_repo,
        trajectories=trajectories,
        feature_states=feature_states,
        ocr_records=ocr_records,
        image_members=image_members,
        payload_files=payload_files,
        selection_manifest=selection_manifest,
        inputs=inputs,
        generator=generator,
        ocr_backend_provenance=ocr_backend_provenance,
        ocr_runtime_identity=ocr_runtime_identity,
        formal_counts_enforced=formal_counts_enforced,
    )
    payloads = {
        GITATTRIBUTES_RELATIVE_PATH: artifact_gitattributes_bytes(),
        README_RELATIVE_PATH: artifact_readme_bytes(dataset_repo=dataset_repo),
        IMAGE_TAR_RELATIVE_PATH: image_tar,
        OCR_JSONL_RELATIVE_PATH: ocr_payload,
        TRAJECTORY_JSONL_RELATIVE_PATH: trajectory_payload,
        FEATURE_JSONL_RELATIVE_PATH: feature_payload,
        MANIFEST_RELATIVE_PATH: pretty_json_bytes(manifest),
    }
    root.mkdir(parents=True)
    for relative in ARTIFACT_RELATIVE_PATHS:
        path = root.joinpath(*PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payloads[relative])
    return artifact_tree_identity(root)


def artifact_tree_identity(root: str | Path) -> dict[str, Any]:
    artifact_root = Path(root)
    observed = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    expected = set(ARTIFACT_RELATIVE_PATHS)
    if observed != expected:
        raise ValueError(
            "long-horizon exact-seven inventory drifted: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    files = [
        {
            "path": relative,
            "size_bytes": artifact_root.joinpath(relative).stat().st_size,
            "sha256": sha256_file(artifact_root.joinpath(relative)),
        }
        for relative in ARTIFACT_RELATIVE_PATHS
    ]
    return {
        "files": files,
        "artifact_tree_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "protocol_id",
        "artifact_id",
        "status",
        "dataset_repo",
        "license",
        "policy_output_generated",
        "restoration_output_generated",
        "current_expert_action_payload_included",
        "unopened_reserve_semantic_content_access_count",
        "source_transport_row_group_overread_possible",
        "consumer_must_slice_events_by_history_event_step_ids",
        "formal_counts_enforced",
        "inputs",
        "generator",
        "ocr_backend",
        "ocr_runtime",
        "source_dataset",
        "development_source_ids",
        "unopened_reserve",
        "state_geometry",
        "counts",
        "inventories",
        "payload_files",
    }
    if set(manifest) != expected:
        raise ValueError("long-horizon manifest field inventory drifted")
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["protocol_id"] != PROTOCOL_ID
        or manifest["artifact_id"] != ARTIFACT_ID
        or manifest["status"] != STATUS
        or manifest["license"] != "cc-by-4.0"
        or manifest["policy_output_generated"] is not False
        or manifest["restoration_output_generated"] is not False
        or manifest["current_expert_action_payload_included"] is not False
        or manifest["unopened_reserve_semantic_content_access_count"] != 0
        or manifest["source_transport_row_group_overread_possible"] is not True
        or manifest["consumer_must_slice_events_by_history_event_step_ids"] is not True
    ):
        raise ValueError("long-horizon manifest identity or firewall drifted")
    if set(manifest["inputs"]) != INPUT_KEYS:
        raise ValueError("long-horizon manifest input inventory drifted")
    for key, record in manifest["inputs"].items():
        _identity_record(record, label=key)
    _validate_generator(manifest["generator"])


def _validate_payload_files(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    expected_paths = [
        IMAGE_TAR_RELATIVE_PATH,
        OCR_JSONL_RELATIVE_PATH,
        TRAJECTORY_JSONL_RELATIVE_PATH,
        FEATURE_JSONL_RELATIVE_PATH,
    ]
    if [record.get("path") for record in records] != expected_paths:
        raise ValueError("long-horizon payload file order drifted")
    for record in records:
        count_key = (
            "member_count"
            if record["path"] == IMAGE_TAR_RELATIVE_PATH
            else "record_count"
        )
        if set(record) != {"path", "size_bytes", "sha256", count_key}:
            raise ValueError("long-horizon payload file schema drifted")
        path = root.joinpath(*PurePosixPath(record["path"]).parts)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError("long-horizon payload file identity drifted")


def validate_payload_file_identities(
    root: str | Path, manifest: Mapping[str, Any]
) -> None:
    """Verify every locally consumed payload against the bound substrate manifest."""
    _validate_manifest(manifest)
    _validate_payload_files(Path(root), manifest["payload_files"])


def _validate_internal_records(
    trajectories: Sequence[Mapping[str, Any]],
    feature_states: Sequence[FeatureState],
    *,
    image_payloads: Mapping[str, bytes],
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
    prepared_by_path: Mapping[str, PreparedImage],
) -> None:
    features_by_state = {state.state_id: state for state in feature_states}
    if len(features_by_state) != len(feature_states):
        raise ValueError("feature state IDs must be unique")
    for trajectory in trajectories:
        if set(trajectory) != TRAJECTORY_KEYS:
            raise ValueError("long-horizon trajectory schema drifted")
        if trajectory["instruction_sha256"] != sha256_bytes(
            str(trajectory["instruction"]).encode("utf-8")
        ):
            raise ValueError("long-horizon instruction witness drifted")
        events = trajectory["events"]
        if [event.get("step_id") for event in events] != list(
            range(1, EVENT_COUNT_PER_TRAJECTORY + 1)
        ):
            raise ValueError("long-horizon event prefix drifted")
        for event in events:
            if not isinstance(event, Mapping) or set(event) != EVENT_KEYS:
                raise ValueError("long-horizon event schema drifted")
            rebuilt = _build_derived_event(
                event,
                image_payloads=image_payloads,
                ocr_records_by_path=ocr_records_by_path,
                prepared_by_path=prepared_by_path,
            )
            if rebuilt != event:
                raise ValueError("long-horizon derived event replay drifted")
        decisions = trajectory["decisions"]
        if len(decisions) != len(STATE_STEPS):
            raise ValueError("long-horizon state denominator drifted")
        for decision, step in zip(decisions, STATE_STEPS, strict=True):
            if not isinstance(decision, Mapping) or set(decision) != DECISION_KEYS:
                raise ValueError("long-horizon decision schema drifted")
            history = list(range(1, step))
            candidates = list(CANDIDATES_BY_STEP[step])
            current_equivalent = step - 1
            if (
                decision["decision_step_id"] != step
                or decision["history_event_step_ids"] != history
                or decision["candidate_event_step_ids"] != candidates
                or decision["current_equivalent_event_step_id"] != current_equivalent
                or decision["current_expert_action_payload_included"] is not False
            ):
                raise ValueError("long-horizon decision geometry drifted")
            for forbidden in (
                "validated_action",
                "canonical_action",
                "expert_action",
                "current_expert_action",
            ):
                if forbidden in decision:
                    raise ValueError(
                        "current expert action payload leaked into decision"
                    )
            current_path = decision["current_observation_path"]
            current_sha = decision["current_observation_sha256"]
            equivalent = events[current_equivalent - 1]
            if (
                sha256_bytes(image_payloads[current_path]) != current_sha
                or equivalent["observation_after_path"] != current_path
                or equivalent["observation_after_sha256"] != current_sha
            ):
                raise ValueError("long-horizon current equivalence witness drifted")
            expected_candidates = [
                {
                    "event_step_id": candidate,
                    "post_state_member_path": events[candidate - 1][
                        "observation_after_path"
                    ],
                    "post_state_sha256": events[candidate - 1][
                        "observation_after_sha256"
                    ],
                }
                for candidate in candidates
            ]
            if decision["candidate_event_post_states"] != expected_candidates:
                raise ValueError("long-horizon candidate witness drifted")
            expected_current = {
                "event_step_id": current_equivalent,
                "post_state_member_path": current_path,
                "post_state_sha256": current_sha,
            }
            if decision["current_equivalence_witness"] != expected_current:
                raise ValueError("long-horizon current-equivalence record drifted")
            state = features_by_state.get(decision["state_id"])
            if state is None:
                raise ValueError("long-horizon feature-state join is missing")
            live_events = [
                {
                    "low_fidelity_v2": events[index - 1]["low_fidelity_v2"],
                    "post_ocr_spatial_tokens": ocr_records_by_path[
                        events[index - 1]["observation_after_path"]
                    ]["full_spatial_tokens"],
                }
                for index in history
            ]
            replay = feature_state_from_live_history(
                source_id=trajectory["source_id"],
                state_id=decision["state_id"],
                decision_step_id=step,
                instruction=trajectory["instruction"],
                current_ocr_spatial_tokens=ocr_records_by_path[current_path][
                    "full_spatial_tokens"
                ],
                history_events=live_events,
                ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
            )
            if replay != state:
                raise ValueError("long-horizon frozen feature replay drifted")
            ordinal = feature_states.index(state)
            if decision["feature_state_sha256"] != sha256_bytes(
                canonical_json_bytes(_feature_record(state, ordinal=ordinal))
            ):
                raise ValueError("long-horizon feature-state witness drifted")
            if decision["content_witness_sha256"] != _decision_content_witness(
                decision
            ):
                raise ValueError("long-horizon decision content witness drifted")
        if trajectory["content_witness_sha256"] != _trajectory_content_witness(
            trajectory
        ):
            raise ValueError("long-horizon trajectory content witness drifted")


def validate_artifact(
    *,
    output_dir: str | Path,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    selection_manifest: Mapping[str, Any],
    expected_input_sha256: Mapping[str, str] | None = None,
    expected_dataset_repo: str = DATASET_REPO,
    expected_ocr_backend_provenance: Mapping[str, Any] | None = None,
    expected_generator_git_revision: str | None = None,
    repository_root: str | Path | None = None,
    pilots_by_source: Mapping[str, Mapping[str, Any]] | None = None,
    source_image_payloads: Mapping[str, bytes] | None = None,
    ocr_engine: Any | None = None,
    require_ocr_replay: bool = False,
    require_formal: bool = True,
    ocr_record_runner: Callable[..., Mapping[str, Any]] = run_rapidocr_record,
) -> dict[str, Any]:
    validate_backend_config(backend_config)
    selection_records(selection_manifest, require_formal=require_formal)
    root = Path(output_dir)
    tree = artifact_tree_identity(root)
    if (
        root / GITATTRIBUTES_RELATIVE_PATH
    ).read_bytes() != artifact_gitattributes_bytes():
        raise ValueError("long-horizon .gitattributes drifted")
    manifest_payload, manifest = load_json_object(root / MANIFEST_RELATIVE_PATH)
    if manifest_payload != pretty_json_bytes(manifest):
        raise ValueError("long-horizon manifest is not canonical pretty JSON")
    _validate_manifest(manifest)
    if manifest["dataset_repo"] != expected_dataset_repo:
        raise ValueError("long-horizon dataset repo drifted")
    if require_formal and manifest["formal_counts_enforced"] is not True:
        raise ValueError("formal long-horizon artifact flag is false")
    if (root / README_RELATIVE_PATH).read_bytes() != artifact_readme_bytes(
        dataset_repo=manifest["dataset_repo"]
    ):
        raise ValueError("long-horizon README drifted")
    if manifest["ocr_backend"]["backend_config_sha256"] != backend_config_sha256:
        raise ValueError("long-horizon OCR backend config drifted")
    if expected_ocr_backend_provenance is not None and manifest["ocr_backend"] != dict(
        expected_ocr_backend_provenance
    ):
        raise ValueError("long-horizon OCR backend provenance drifted")
    validate_ocr_runtime_identity(
        manifest["ocr_runtime"], backend_config=backend_config
    )
    if (
        expected_generator_git_revision is not None
        and manifest["generator"]["git_revision"] != expected_generator_git_revision
    ):
        raise ValueError("long-horizon generator revision drifted")
    if expected_input_sha256 is not None:
        if set(expected_input_sha256) != INPUT_KEYS:
            raise ValueError("expected long-horizon input inventory drifted")
        for key, digest in expected_input_sha256.items():
            if manifest["inputs"][key]["sha256"] != digest:
                raise ValueError(f"long-horizon input digest drifted: {key}")
    if repository_root is not None:
        checkout = Path(repository_root)
        for prefix in ("module", "build_cli", "validator"):
            path = checkout / manifest["generator"][f"{prefix}_path"]
            if sha256_file(path) != manifest["generator"][f"{prefix}_sha256"]:
                raise ValueError(f"long-horizon generator source drifted: {prefix}")
    _validate_payload_files(root, manifest["payload_files"])
    image_payloads, image_members = _validate_tar(root / IMAGE_TAR_RELATIVE_PATH)
    ocr_records = parse_canonical_jsonl(
        (root / OCR_JSONL_RELATIVE_PATH).read_bytes(), label="long-horizon OCR"
    )
    ocr_by_path = {record["image_member_path"]: record for record in ocr_records}
    if list(ocr_by_path) != sorted(ocr_by_path) or set(ocr_by_path) != set(
        image_payloads
    ):
        raise ValueError("long-horizon OCR/image inventory drifted")
    prepared = {
        path: validate_ocr_record(
            ocr_by_path[path],
            image_bytes=image_payloads[path],
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
        )
        for path in sorted(image_payloads)
    }
    trajectories = parse_canonical_jsonl(
        (root / TRAJECTORY_JSONL_RELATIVE_PATH).read_bytes(),
        label="long-horizon trajectories",
    )
    feature_states = read_feature_states_jsonl(
        (root / FEATURE_JSONL_RELATIVE_PATH).read_bytes(),
        require_formal=require_formal,
    )
    _validate_internal_records(
        trajectories,
        feature_states,
        image_payloads=image_payloads,
        ocr_records_by_path=ocr_by_path,
        prepared_by_path=prepared,
    )
    development, reserve = selection_records(
        selection_manifest, require_formal=require_formal
    )
    expected_ids = [str(record["trajectory_id"]) for record in development]
    if [record["source_id"] for record in trajectories] != expected_ids:
        raise ValueError("long-horizon trajectory order differs from selection")
    if manifest["development_source_ids"] != expected_ids:
        raise ValueError("long-horizon manifest development roster drifted")
    expected_selection_by_source = {
        str(record["trajectory_id"]): dict(record) for record in development
    }
    source_fields = (
        "upstream_repo",
        "upstream_revision",
        "transport_repo",
        "transport_revision",
        "license",
    )
    for trajectory in trajectories:
        source_id = str(trajectory["source_id"])
        source = trajectory["source"]
        selected = expected_selection_by_source[source_id]
        if trajectory["selection"] != selected:
            raise ValueError("long-horizon embedded selection witness drifted")
        if (
            source["transport_file"] != selected["shard_path"]
            or source["transport_row_index"] != selected["row_index"]
            or any(
                source.get(field) != manifest["source_dataset"].get(field)
                for field in source_fields
            )
        ):
            raise ValueError("long-horizon source provenance witness drifted")
    reserve_ids = [str(record["trajectory_id"]) for record in reserve]
    if manifest["unopened_reserve"] != {
        "trajectory_count": len(reserve),
        "source_ids_sha256": sha256_bytes(canonical_json_bytes(reserve_ids)),
        "python_row_materialization_count": 0,
        "semantic_content_access_count": 0,
        "transport_row_group_overread_possible": True,
    }:
        raise ValueError("long-horizon reserve structural witness drifted")
    expected_geometry = {
        "decision_step_ids": list(STATE_STEPS),
        "candidate_event_step_ids_by_decision": {
            str(step): list(CANDIDATES_BY_STEP[step]) for step in STATE_STEPS
        },
    }
    if manifest["state_geometry"] != expected_geometry:
        raise ValueError("long-horizon manifest state geometry drifted")
    expected_inventories = {
        "image_members_sha256": sha256_bytes(canonical_json_bytes(list(image_members))),
        "ocr_records_sha256": ocr_record_aggregate_sha256(ocr_by_path),
        "trajectory_index_sha256": sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        "source_id": record["source_id"],
                        "content_witness_sha256": record["content_witness_sha256"],
                    }
                    for record in trajectories
                ]
            )
        ),
        "feature_states_sha256": sha256_bytes(
            feature_states_jsonl_bytes(feature_states)
        ),
    }
    if manifest["inventories"] != expected_inventories:
        raise ValueError("long-horizon manifest content inventory drifted")
    if pilots_by_source is not None or source_image_payloads is not None:
        if pilots_by_source is None or source_image_payloads is None:
            raise ValueError("raw replay requires pilots and source images together")
        rebuilt_trajectories, rebuilt_features, rebuilt_images = (
            build_long_horizon_records(
                pilots_by_source=pilots_by_source,
                source_image_payloads=source_image_payloads,
                selection_manifest=selection_manifest,
                ocr_records_by_path=ocr_by_path,
                backend_config=backend_config,
                backend_config_sha256=backend_config_sha256,
                require_formal=require_formal,
            )
        )
        if (
            rebuilt_trajectories != trajectories
            or rebuilt_features != feature_states
            or rebuilt_images != image_payloads
        ):
            raise ValueError("long-horizon raw source replay drifted")
    if require_ocr_replay:
        if ocr_engine is None:
            raise ValueError("OCR replay requested without an OCR engine")
        for path in sorted(image_payloads):
            replay = dict(
                ocr_record_runner(
                    engine=ocr_engine,
                    backend_config=backend_config,
                    image_member_path=path,
                    image_bytes=image_payloads[path],
                    backend_config_sha256=backend_config_sha256,
                )
            )
            if replay != ocr_by_path[path]:
                raise ValueError("long-horizon OCR replay drifted")
    counts = {
        "trajectory_count": len(trajectories),
        "event_count": sum(len(record["events"]) for record in trajectories),
        "state_count": len(feature_states),
        "candidate_count": sum(
            len(state.candidate_event_step_ids) for state in feature_states
        ),
        "image_member_count": len(image_members),
        "ocr_record_count": len(ocr_records),
    }
    if manifest["counts"] != counts or (require_formal and counts != EXPECTED_COUNTS):
        raise ValueError("long-horizon count accounting drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "outcome": "PASSED_LONG_HORIZON_SUBSTRATE_VALIDATION",
        "artifact_tree_sha256": tree["artifact_tree_sha256"],
        "manifest_sha256": sha256_file(root / MANIFEST_RELATIVE_PATH),
        "counts": counts,
        "ocr_replay_record_count": len(ocr_records) if require_ocr_replay else 0,
        "raw_development_replay": pilots_by_source is not None,
        "unopened_reserve_semantic_content_access_count": 0,
        "source_transport_row_group_overread_possible": True,
        "policy_output_used": False,
        "restoration_output_used": False,
    }


__all__ = [
    "ARTIFACT_RELATIVE_PATHS",
    "DATASET_REPO",
    "EXPECTED_COUNTS",
    "FEATURE_JSONL_RELATIVE_PATH",
    "IMAGE_TAR_RELATIVE_PATH",
    "MANIFEST_RELATIVE_PATH",
    "OCR_JSONL_RELATIVE_PATH",
    "TRAJECTORY_JSONL_RELATIVE_PATH",
    "artifact_tree_identity",
    "build_long_horizon_records",
    "build_ocr_backend_provenance",
    "build_payload_manifest",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "feature_states_jsonl_bytes",
    "generate_ocr_records",
    "load_development_source_pilots",
    "load_json_object",
    "pretty_json_bytes",
    "read_feature_states_jsonl",
    "required_image_member_paths",
    "selection_records",
    "sha256_bytes",
    "validate_artifact",
    "validate_payload_file_identities",
    "write_artifact",
]
