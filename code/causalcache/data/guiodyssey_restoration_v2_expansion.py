"""Deterministic policy-blind GUIOdyssey label-expansion artifact."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.guiodyssey import build_pilot_manifest
from causalcache.data.guiodyssey_independent import (
    source_file_specs,
    trajectory_selection_sha256,
    verify_local_source_files,
)
from causalcache.data.guiodyssey_restoration_v2 import (
    EVENT_KEYS,
    FROZEN_OCR_BACKEND_MANIFEST_SHA256,
    OCR_RUNTIME_IDENTITY_KEYS,
    PreparedImage,
    _build_derived_event,
    _iter_parquet_rows,
    _safe_member_path,
    _validate_tar,
    artifact_gitattributes_bytes,
    build_image_tar_bytes,
    build_ocr_backend_provenance,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    generate_ocr_records,
    ocr_record_aggregate_sha256,
    parse_canonical_jsonl,
    pretty_json_bytes,
    sha256_bytes,
    sha256_file,
    source_dataset_identity_from_v1_config,
    validate_ocr_record,
    validate_ocr_runtime_identity,
)
from causalcache.restoration_v2_2_label_expansion import (
    SPLIT_NAMES,
    validate_expansion_manifest,
)
from causalcache.restoration_v2_text_backend import (
    BACKEND_ID,
    run_rapidocr_record,
    validate_backend_config,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_restoration_v2_label_expansion_derived_v1"
ARTIFACT_ID = "causalcache-guiodyssey-restoration-v2-label-expansion-mobile-v1"
STATUS = "POLICY_BLIND_LABEL_EXPANSION_DATASET_MATERIALIZED"
PAYLOAD_PREFIX = "derived/restoration-v2-label-expansion-v1"
IMAGE_TAR_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/images-00000-of-00001.tar"
OCR_JSONL_RELATIVE_PATH = (
    f"{PAYLOAD_PREFIX}/ocr-records-00000-of-00001.jsonl"
)
TRAJECTORY_JSONL_RELATIVE_PATH = (
    f"{PAYLOAD_PREFIX}/trajectories-00000-of-00001.jsonl"
)
MANIFEST_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/manifest.json"
GITATTRIBUTES_RELATIVE_PATH = ".gitattributes"
README_RELATIVE_PATH = "README.md"
ARTIFACT_RELATIVE_PATHS = (
    GITATTRIBUTES_RELATIVE_PATH,
    README_RELATIVE_PATH,
    IMAGE_TAR_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    OCR_JSONL_RELATIVE_PATH,
    TRAJECTORY_JSONL_RELATIVE_PATH,
)
DERIVED_ROLE_ORDER = (
    "gate_train_expansion",
    "gate_development_expansion",
)
EXPECTED_FORMAL_COUNTS = {
    "trajectory_count": 64,
    "event_count": 320,
    "state_count": 192,
    "image_member_count": 384,
    "ocr_record_count": 384,
}
EXPECTED_ROLE_COUNTS = {
    "gate_train_expansion": 48,
    "gate_development_expansion": 16,
}
INPUT_KEYS = {
    "label_expansion_config",
    "parent_selection_manifest",
    "expansion_selection_manifest",
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
    "content_witness_sha256",
    "current_expert_action_payload_included",
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
FORBIDDEN_DECISION_KEYS = {
    "validated_action",
    "validated_action_sha256",
    "canonical_action",
    "canonical_action_sha256",
    "expert_action",
    "expert_action_sha256",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".webp"}


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _load_json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = child
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload, value


def _identity_record(record: Mapping[str, Any], *, field: str) -> None:
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{field} identity must contain exactly path and sha256")
    _safe_member_path(record.get("path"))
    _require_sha256(record.get("sha256"), f"{field}.sha256")


def _bind_supplied_manifest_sha256(
    manifest: Mapping[str, Any], *, expected_sha256: str, field: str
) -> None:
    _require_sha256(expected_sha256, field)
    if sha256_bytes(pretty_json_bytes(manifest)) != expected_sha256:
        raise ValueError(f"supplied {field} does not bind canonical pretty bytes")


def _file_record(path: str, payload: bytes, **extra: Any) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        **extra,
    }


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _selection_index(
    selection_manifest: Mapping[str, Any],
) -> tuple[
    dict[str, str],
    dict[str, Mapping[str, Any]],
    dict[str, list[Mapping[str, Any]]],
]:
    if selection_manifest.get("policy_output_generated") is not False:
        raise ValueError("expansion selection must precede policy output")
    if selection_manifest.get("restoration_output_generated") is not False:
        raise ValueError("expansion selection must precede restoration output")
    if selection_manifest.get("structural_manifest_only") is not True:
        raise ValueError("expansion selection must be structural only")
    splits = selection_manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise ValueError("expansion selection split inventory drifted")

    role_by_source: dict[str, str] = {}
    trajectory_by_source: dict[str, Mapping[str, Any]] = {}
    states_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for role in DERIVED_ROLE_ORDER:
        split = splits.get(role)
        if not isinstance(split, Mapping) or set(split) != {"trajectories", "states"}:
            raise ValueError(f"expansion selection split schema drifted: {role}")
        trajectories = split["trajectories"]
        states = split["states"]
        if not isinstance(trajectories, list) or not isinstance(states, list):
            raise ValueError("expansion selection records must be arrays")
        for record in trajectories:
            if not isinstance(record, Mapping):
                raise ValueError("expansion trajectory selections must be objects")
            source_id = str(record.get("source_id"))
            if not source_id or source_id in role_by_source:
                raise ValueError("expansion source IDs must be non-empty and unique")
            _require_sha256(record.get("selection_sha256"), "selection_sha256")
            role_by_source[source_id] = role
            trajectory_by_source[source_id] = record
        for state in states:
            if not isinstance(state, Mapping):
                raise ValueError("expansion states must be objects")
            source_id = str(state.get("source_id"))
            if role_by_source.get(source_id) != role:
                raise ValueError("expansion state belongs to the wrong split")
            states_by_source[source_id].append(state)

    if set(states_by_source) != set(role_by_source):
        raise ValueError("every expansion trajectory must have states")
    for source_id, states in states_by_source.items():
        if [state.get("decision_step_id") for state in states] != [4, 5, 6]:
            raise ValueError(f"expansion state steps drifted: {source_id}")
        for state in states:
            step = int(state["decision_step_id"])
            if (
                state.get("history_event_step_ids") != list(range(1, step))
                or state.get("candidate_event_step_ids") != list(range(1, step - 1))
                or state.get("current_equivalent_event_step_id") != step - 1
            ):
                raise ValueError("expansion state history geometry drifted")
    return role_by_source, trajectory_by_source, states_by_source


def _forbidden_prior_source_ids(parent_selection: Mapping[str, Any]) -> set[str]:
    roles = parent_selection.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("parent selection roles are missing")
    source_ids: set[str] = set()
    for role in (
        "v1_reference_contract_audit_only",
        "v2_label_train",
        "v2_development",
        "v2_confirm_primary",
    ):
        role_record = roles.get(role)
        if not isinstance(role_record, Mapping):
            raise ValueError(f"parent selection role is missing: {role}")
        records = role_record.get("trajectories")
        if not isinstance(records, list):
            raise ValueError("parent selection trajectories must be an array")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("parent selection trajectory must be an object")
            source_id = str(record.get("source_id"))
            if source_id in source_ids:
                raise ValueError("parent role source IDs overlap")
            source_ids.add(source_id)
    return source_ids


def validate_frozen_inputs(
    *,
    expansion_manifest: Mapping[str, Any],
    expansion_manifest_sha256: str,
    expansion_config: Mapping[str, Any],
    expansion_config_sha256: str,
    parent_selection: Mapping[str, Any],
    parent_selection_sha256: str,
    v1_config: Mapping[str, Any],
    v1_config_sha256: str,
    source_file_manifest_sha256: str,
    backend_config: Mapping[str, Any],
) -> None:
    for value, field in (
        (expansion_manifest_sha256, "expansion_manifest_sha256"),
        (expansion_config_sha256, "expansion_config_sha256"),
        (parent_selection_sha256, "parent_selection_sha256"),
        (v1_config_sha256, "v1_config_sha256"),
        (source_file_manifest_sha256, "source_file_manifest_sha256"),
    ):
        _require_sha256(value, field)
    if sha256_bytes(pretty_json_bytes(expansion_manifest)) != (
        expansion_manifest_sha256
    ):
        raise ValueError(
            "expansion manifest SHA256 does not bind canonical pretty bytes"
        )
    validate_expansion_manifest(
        expansion_manifest,
        config=expansion_config,
        config_sha256=expansion_config_sha256,
        parent=parent_selection,
        parent_sha256=parent_selection_sha256,
    )
    role_by_source, _, _ = _selection_index(expansion_manifest)
    overlap = set(role_by_source).intersection(
        _forbidden_prior_source_ids(parent_selection)
    )
    if overlap:
        raise ValueError("expansion selection overlaps legacy or confirm source IDs")
    parent_inputs = parent_selection["inputs"]
    if parent_inputs["v1_selection_config"]["current_sha256"] != v1_config_sha256:
        raise ValueError("v1 config bytes differ from the frozen parent input")
    if parent_inputs["source_file_manifest"]["sha256"] != (
        source_file_manifest_sha256
    ):
        raise ValueError("source-file manifest differs from the frozen parent input")
    if len(v1_config["source_pool"]["transport_files"]) != 16:
        raise ValueError("label expansion requires the exact original 16 Parquets")
    validate_backend_config(backend_config)


def load_expansion_source_pilots(
    *,
    source_root: str | Path,
    source_file_manifest: Mapping[str, Any],
    v1_config: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    derived_repo: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    if not isinstance(derived_repo, str) or not derived_repo:
        raise ValueError("derived_repo must be a non-empty string")
    role_by_source, selected_by_source, _ = _selection_index(selection_manifest)
    specs = source_file_specs(v1_config, source_file_manifest)
    if len(specs) != 16:
        raise ValueError("label expansion requires exactly 16 source files")
    verify_local_source_files(Path(source_root), specs)
    spec_by_path = {spec.transport_file: spec for spec in specs}
    wanted_by_file: dict[str, dict[int, str]] = defaultdict(dict)
    salt = str(v1_config["selection"]["trajectory_salt"])
    for source_id, selected in selected_by_source.items():
        if trajectory_selection_sha256(source_id, salt=salt) != selected.get(
            "selection_sha256"
        ):
            raise ValueError("selected trajectory salted hash drifted")
        transport_file = str(selected.get("transport_file"))
        row_index = selected.get("transport_row_index")
        if transport_file not in spec_by_path or type(row_index) is not int:
            raise ValueError("selected source transport identity drifted")
        if row_index in wanted_by_file[transport_file]:
            raise ValueError("selected source row is duplicated")
        wanted_by_file[transport_file][row_index] = source_id

    import pyarrow.parquet as pq

    source_rows: dict[str, Mapping[str, Any]] = {}
    root = Path(source_root)
    for transport_file, wanted in wanted_by_file.items():
        parquet = pq.ParquetFile(
            root.joinpath(*PurePosixPath(transport_file).parts)
        )
        for row_index, row in _iter_parquet_rows(parquet):
            source_id = wanted.get(row_index)
            if source_id is not None:
                source_rows[source_id] = row
    if set(source_rows) != set(role_by_source):
        raise ValueError("failed to reload every expansion source row")

    pilots: dict[str, dict[str, Any]] = {}
    image_payloads: dict[str, bytes] = {}
    source_pool = v1_config["source_pool"]
    grid_size = int(v1_config["policy"]["coordinate_grid_size"])
    for role in DERIVED_ROLE_ORDER:
        for selected in selection_manifest["splits"][role]["trajectories"]:
            source_id = str(selected["source_id"])
            transport_file = str(selected["transport_file"])
            spec = spec_by_path[transport_file]
            pilot, images = build_pilot_manifest(
                source_rows[source_id],
                row_index=int(selected["transport_row_index"]),
                upstream_repo=str(source_pool["upstream_repo"]),
                upstream_revision=str(source_pool["upstream_revision"]),
                transport_repo=str(source_pool["transport_repo"]),
                transport_revision=str(source_pool["transport_revision"]),
                transport_file=transport_file,
                transport_file_sha256=spec.sha256,
                hf_destination=derived_repo,
                grid_size=grid_size,
            )
            trajectory = pilot["trajectory"]
            if trajectory.get("source_id") != source_id:
                raise ValueError("reloaded expansion source ID drifted")
            if len(trajectory.get("decisions", [])) != int(
                selected["decision_count"]
            ):
                raise ValueError("reloaded expansion decision count drifted")
            pilots[source_id] = pilot
            for path, payload in images.items():
                if path in image_payloads:
                    raise ValueError("GUIOdyssey image member paths must be unique")
                image_payloads[path] = payload
    return pilots, image_payloads


def required_image_member_paths(
    *,
    pilots_by_source: Mapping[str, Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    role_by_source, _, _ = _selection_index(selection_manifest)
    if set(pilots_by_source) != set(role_by_source):
        raise ValueError("source pilots must cover exactly the expansion selection")
    observed: set[str] = set()
    for source_id in role_by_source:
        trajectory = pilots_by_source[source_id].get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise ValueError("source pilot trajectory is missing")
        events = trajectory.get("events")
        if not isinstance(events, list) or len(events) < 5:
            raise ValueError("expansion source pilot must contain five history events")
        prefix = events[:5]
        if [event.get("step_id") for event in prefix] != [1, 2, 3, 4, 5]:
            raise ValueError("expansion source event prefix drifted")
        source_paths: set[str] = set()
        for event in prefix:
            source_paths.add(_safe_member_path(event["observation_before_path"]))
            source_paths.add(_safe_member_path(event["observation_after_path"]))
        if _observation_indices(source_paths, source_id=source_id) != set(range(6)):
            raise ValueError("expansion source image naming or continuity drifted")
        observed.update(source_paths)
    return tuple(sorted(observed))


def _observation_indices(paths: Sequence[str], *, source_id: str) -> set[int]:
    indices: set[int] = set()
    prefix = f"images/{source_id}/observation-"
    for value in paths:
        path = _safe_member_path(value)
        if not path.startswith(prefix):
            raise ValueError("expansion image path belongs to the wrong source")
        suffix = PurePosixPath(path).suffix
        stem = PurePosixPath(path).stem
        index_text = stem.removeprefix("observation-")
        if suffix not in ALLOWED_IMAGE_SUFFIXES or not re.fullmatch(
            r"[0-9]{3}", index_text
        ):
            raise ValueError("expansion image path format drifted")
        index = int(index_text)
        if index in indices:
            raise ValueError("expansion source has duplicate observation indices")
        indices.add(index)
    return indices


def validate_formal_image_inventory(
    image_member_paths: Sequence[str],
    *,
    selection_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    observed = tuple(image_member_paths)
    if observed != tuple(sorted(observed)) or len(observed) != len(set(observed)):
        raise ValueError("formal expansion images must be sorted and unique")
    role_by_source, _, _ = _selection_index(selection_manifest)
    if len(role_by_source) != EXPECTED_FORMAL_COUNTS["trajectory_count"]:
        raise ValueError("formal expansion must contain exactly 64 trajectories")
    observed_set = set(observed)
    for source_id in role_by_source:
        source_paths = [
            path
            for path in observed
            if path.startswith(f"images/{source_id}/")
        ]
        if _observation_indices(source_paths, source_id=source_id) != set(range(6)):
            raise ValueError("formal expansion image inventory mismatch")
        observed_set.difference_update(source_paths)
    if observed_set or len(observed) != EXPECTED_FORMAL_COUNTS["image_member_count"]:
        raise ValueError("formal expansion image inventory mismatch")
    return observed


def _decision_content_witness(
    *,
    source_id: str,
    state_id: str,
    current_observation_path: str,
    current_observation_sha256: str,
    candidate_event_post_states: Sequence[Mapping[str, Any]],
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "source_id": source_id,
                "state_id": state_id,
                "current_observation_path": current_observation_path,
                "current_observation_sha256": current_observation_sha256,
                "candidate_event_post_states": list(candidate_event_post_states),
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
                "event_observation_sha256": [
                    {
                        "step_id": event["step_id"],
                        "before": event["observation_before_sha256"],
                        "after": event["observation_after_sha256"],
                    }
                    for event in record["events"]
                ],
                "decision_content_witness_sha256": [
                    decision["content_witness_sha256"]
                    for decision in record["decisions"]
                ],
            }
        )
    )


def decision_view_events(
    events: Sequence[Mapping[str, Any]], decision: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    step = decision.get("decision_step_id")
    history = decision.get("history_event_step_ids")
    if type(step) is not int or not isinstance(history, list):
        raise ValueError("decision view geometry is malformed")
    if history != list(range(1, step)):
        raise ValueError("decision view must contain exactly the strict history prefix")
    by_step = {event.get("step_id"): event for event in events}
    if len(by_step) != len(events):
        raise ValueError("decision view source events contain duplicate steps")
    try:
        view = tuple(by_step[event_step] for event_step in history)
    except KeyError as error:
        raise ValueError("decision view is missing a required history event") from error
    if any(int(event["step_id"]) >= step for event in view):
        raise ValueError("decision view exposed a current or future event action")
    return view


def _validate_image_suffix(path: str, payload: bytes) -> None:
    suffix = PurePosixPath(path).suffix
    valid = {
        ".png": payload.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": payload.startswith(b"\xff\xd8\xff"),
        ".webp": (
            payload.startswith(b"RIFF")
            and len(payload) >= 12
            and payload[8:12] == b"WEBP"
        ),
    }
    if suffix not in valid or not valid[suffix]:
        raise ValueError("expansion image suffix differs from encoded image format")


def build_trajectory_records(
    *,
    pilots_by_source: Mapping[str, Mapping[str, Any]],
    source_image_payloads: Mapping[str, bytes],
    selection_manifest: Mapping[str, Any],
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    ocr_record_validator: Callable[..., PreparedImage] = validate_ocr_record,
    event_builder: Callable[..., Mapping[str, Any]] = _build_derived_event,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    role_by_source, selected_by_source, states_by_source = _selection_index(
        selection_manifest
    )
    required_paths = required_image_member_paths(
        pilots_by_source=pilots_by_source,
        selection_manifest=selection_manifest,
    )
    if set(ocr_records_by_path) != set(required_paths):
        raise ValueError("expansion OCR image inventory mismatch")
    try:
        image_payloads = {path: source_image_payloads[path] for path in required_paths}
    except KeyError as error:
        raise ValueError("source images are missing an expansion member") from error

    prepared_by_path: dict[str, PreparedImage] = {}
    for path in required_paths:
        prepared_by_path[path] = ocr_record_validator(
            ocr_records_by_path[path],
            image_bytes=image_payloads[path],
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
        )

    trajectory_records: list[dict[str, Any]] = []
    for role in DERIVED_ROLE_ORDER:
        for selected in selection_manifest["splits"][role]["trajectories"]:
            source_id = str(selected["source_id"])
            pilot = pilots_by_source[source_id]
            trajectory = pilot["trajectory"]
            if role_by_source[source_id] != role or trajectory["source_id"] != source_id:
                raise ValueError("expansion role/source identity drifted")
            if (
                pilot["source"]["transport_file"] != selected["transport_file"]
                or pilot["source"]["transport_row_index"]
                != selected["transport_row_index"]
            ):
                raise ValueError("expansion source transport witness drifted")
            instruction = str(trajectory["instruction"])
            instruction_sha = sha256_bytes(instruction.encode("utf-8"))
            source_events = {
                int(event["step_id"]): event for event in trajectory["events"]
            }
            events = [
                dict(
                    event_builder(
                        source_events[step_id],
                        image_payloads=image_payloads,
                        ocr_records_by_path=ocr_records_by_path,
                        prepared_by_path=prepared_by_path,
                    )
                )
                for step_id in range(1, 6)
            ]
            source_decisions = {
                int(decision["decision_step_id"]): decision
                for decision in trajectory["decisions"]
            }
            decisions: list[dict[str, Any]] = []
            for state in states_by_source[source_id]:
                step = int(state["decision_step_id"])
                current_equivalent = int(state["current_equivalent_event_step_id"])
                source_decision = source_decisions.get(step)
                if source_decision is None:
                    raise ValueError("selected source decision is missing")
                current_path = _safe_member_path(
                    source_decision["current_observation_path"]
                )
                current_sha = sha256_bytes(image_payloads[current_path])
                equivalent_event = events[current_equivalent - 1]
                if (
                    equivalent_event["observation_after_path"] != current_path
                    or equivalent_event["observation_after_sha256"] != current_sha
                ):
                    raise ValueError("expansion current observation equivalence drifted")
                candidate_witnesses = [
                    {
                        "event_step_id": candidate_step,
                        "post_state_member_path": events[candidate_step - 1][
                            "observation_after_path"
                        ],
                        "post_state_sha256": events[candidate_step - 1][
                            "observation_after_sha256"
                        ],
                    }
                    for candidate_step in state["candidate_event_step_ids"]
                ]
                current_witness = {
                    "event_step_id": current_equivalent,
                    "post_state_member_path": current_path,
                    "post_state_sha256": current_sha,
                }
                witness_sha = _decision_content_witness(
                    source_id=source_id,
                    state_id=str(state["state_id"]),
                    current_observation_path=current_path,
                    current_observation_sha256=current_sha,
                    candidate_event_post_states=candidate_witnesses,
                )
                decisions.append(
                    {
                        "state_id": state["state_id"],
                        "decision_step_id": step,
                        "history_event_step_ids": list(
                            state["history_event_step_ids"]
                        ),
                        "candidate_event_step_ids": list(
                            state["candidate_event_step_ids"]
                        ),
                        "current_equivalent_event_step_id": current_equivalent,
                        "current_observation_path": current_path,
                        "current_observation_sha256": current_sha,
                        "candidate_event_post_states": candidate_witnesses,
                        "current_equivalence_witness": current_witness,
                        "content_witness_sha256": witness_sha,
                        "current_expert_action_payload_included": False,
                    }
                )
            record = {
                "source_id": source_id,
                "role": role,
                "instruction": instruction,
                "instruction_sha256": instruction_sha,
                "platform": trajectory["platform"],
                "apps": list(trajectory["apps"]),
                "device_name": trajectory["device_name"],
                "resolution": trajectory["resolution"],
                "terminal_status": trajectory["terminal_status"],
                "source": dict(pilot["source"]),
                "selection": dict(selected_by_source[source_id]),
                "content_witness_sha256": "",
                "events": events,
                "decisions": decisions,
            }
            record["content_witness_sha256"] = _trajectory_content_witness(record)
            trajectory_records.append(record)
    return trajectory_records, image_payloads


def artifact_readme_bytes(*, dataset_repo: str) -> bytes:
    if not isinstance(dataset_repo, str) or not dataset_repo:
        raise ValueError("dataset_repo must be a non-empty string")
    return (
        "---\n"
        "license: cc-by-4.0\n"
        "task_categories:\n"
        "- image-to-text\n"
        "- visual-question-answering\n"
        "---\n\n"
        "# CausalCache restoration-v2 label expansion\n\n"
        "Policy-blind deterministic source artifact for the frozen 48/16 gate "
        "expansion. It contains no policy or restoration output. Each per-decision "
        "view omits its current expert action; consumers must slice the shared "
        "event storage by that decision's `history_event_step_ids`, because later "
        "stored events are valid history only for later decisions.\n\n"
        "Formal inventory: 64 trajectories, 320 history events, 192 decision "
        "states, and 384 archived images with full OCR records.\n\n"
        f"Payload prefix: `{PAYLOAD_PREFIX}`. Hugging Face dataset repo: "
        f"`{dataset_repo}`.\n"
    ).encode("utf-8")


def _validate_generator(generator: Mapping[str, Any]) -> None:
    if set(generator) != GENERATOR_KEYS:
        raise ValueError("expansion artifact generator fields drifted")
    if GIT_SHA_PATTERN.fullmatch(str(generator["git_revision"])) is None:
        raise ValueError("expansion generator Git revision must be immutable")
    for prefix in ("module", "build_cli", "validator"):
        _safe_member_path(generator[f"{prefix}_path"])
        _require_sha256(generator[f"{prefix}_sha256"], f"generator.{prefix}")


def build_payload_manifest(
    *,
    dataset_repo: str,
    trajectories: Sequence[Mapping[str, Any]],
    ocr_records: Sequence[Mapping[str, Any]],
    image_members: Sequence[Mapping[str, Any]],
    payload_files: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Mapping[str, Any]],
    generator: Mapping[str, Any],
    ocr_backend_provenance: Mapping[str, Any],
    ocr_runtime_identity: Mapping[str, Any],
    formal_counts_enforced: bool,
) -> dict[str, Any]:
    if set(inputs) != INPUT_KEYS:
        raise ValueError("expansion artifact input inventory drifted")
    for key, record in inputs.items():
        _identity_record(record, field=key)
    _validate_generator(generator)
    if ocr_backend_provenance.get("backend_id") != BACKEND_ID:
        raise ValueError("expansion OCR backend identity drifted")
    if ocr_backend_provenance.get("backend_config_sha256") != inputs[
        "ocr_backend_config"
    ]["sha256"]:
        raise ValueError("expansion OCR backend/input binding drifted")
    counts = {
        "trajectory_count": len(trajectories),
        "event_count": sum(len(record["events"]) for record in trajectories),
        "state_count": sum(len(record["decisions"]) for record in trajectories),
        "image_member_count": len(image_members),
        "ocr_record_count": len(ocr_records),
    }
    if formal_counts_enforced and counts != EXPECTED_FORMAL_COUNTS:
        raise ValueError("formal expansion artifact count mismatch")
    role_source_ids = {
        role: [
            str(record["source_id"])
            for record in trajectories
            if record["role"] == role
        ]
        for role in DERIVED_ROLE_ORDER
    }
    if formal_counts_enforced and {
        role: len(values) for role, values in role_source_ids.items()
    } != EXPECTED_ROLE_COUNTS:
        raise ValueError("formal expansion role count mismatch")
    first_source = trajectories[0]["source"]
    source_dataset = {
        field: first_source[field]
        for field in (
            "upstream_repo",
            "upstream_revision",
            "transport_repo",
            "transport_revision",
            "license",
        )
    }
    for trajectory in trajectories:
        if any(
            trajectory["source"].get(field) != value
            for field, value in source_dataset.items()
        ):
            raise ValueError("expansion trajectories have mixed source identity")
    trajectory_index = [
        {
            "source_id": record["source_id"],
            "role": record["role"],
            "event_count": len(record["events"]),
            "state_count": len(record["decisions"]),
            "content_witness_sha256": record["content_witness_sha256"],
        }
        for record in trajectories
    ]
    ocr_index = [
        {
            "image_member_path": record["image_member_path"],
            "image_sha256": record["image_sha256"],
            "canonical_ocr_record_sha256": record[
                "canonical_ocr_record_sha256"
            ],
        }
        for record in ocr_records
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_id": ARTIFACT_ID,
        "status": STATUS,
        "dataset_repo": dataset_repo,
        "license": "cc-by-4.0",
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "per_decision_view_current_expert_action_payload_included": False,
        "consumer_must_slice_events_by_history_event_step_ids": True,
        "formal_counts_enforced": formal_counts_enforced,
        "inputs": {key: dict(value) for key, value in inputs.items()},
        "generator": dict(generator),
        "ocr_backend": dict(ocr_backend_provenance),
        "ocr_runtime": dict(ocr_runtime_identity),
        "source_dataset": source_dataset,
        "role_source_ids": role_source_ids,
        "counts": counts,
        "inventories": {
            "image_members_sha256": sha256_bytes(
                canonical_json_bytes(list(image_members))
            ),
            "ocr_records_sha256": sha256_bytes(canonical_json_bytes(ocr_index)),
            "ocr_record_aggregate_sha256": ocr_record_aggregate_sha256(
                {
                    str(record["image_member_path"]): record
                    for record in ocr_records
                }
            ),
            "trajectory_index_sha256": sha256_bytes(
                canonical_json_bytes(trajectory_index)
            ),
        },
        "payload_files": [dict(record) for record in payload_files],
    }


def write_artifact(
    *,
    output_dir: str | Path,
    dataset_repo: str,
    trajectories: Sequence[Mapping[str, Any]],
    image_payloads: Mapping[str, bytes],
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
    inputs: Mapping[str, Mapping[str, Any]],
    generator: Mapping[str, Any],
    ocr_backend_provenance: Mapping[str, Any],
    ocr_runtime_identity: Mapping[str, Any],
    formal_counts_enforced: bool,
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"expansion artifact output already exists: {root}")
    expected_order = [
        (role, record["source_id"])
        for role in DERIVED_ROLE_ORDER
        for record in trajectories
        if record["role"] == role
    ]
    if [(record["role"], record["source_id"]) for record in trajectories] != (
        expected_order
    ):
        raise ValueError("expansion trajectories must follow frozen role order")
    ocr_records = [ocr_records_by_path[path] for path in sorted(ocr_records_by_path)]
    trajectory_bytes = canonical_jsonl_bytes(trajectories)
    ocr_bytes = canonical_jsonl_bytes(ocr_records)
    image_tar_bytes, image_members = build_image_tar_bytes(image_payloads)
    payload_files = [
        _file_record(
            IMAGE_TAR_RELATIVE_PATH,
            image_tar_bytes,
            member_count=len(image_members),
        ),
        _file_record(
            OCR_JSONL_RELATIVE_PATH,
            ocr_bytes,
            record_count=len(ocr_records),
        ),
        _file_record(
            TRAJECTORY_JSONL_RELATIVE_PATH,
            trajectory_bytes,
            record_count=len(trajectories),
        ),
    ]
    manifest = build_payload_manifest(
        dataset_repo=dataset_repo,
        trajectories=trajectories,
        ocr_records=ocr_records,
        image_members=image_members,
        payload_files=payload_files,
        inputs=inputs,
        generator=generator,
        ocr_backend_provenance=ocr_backend_provenance,
        ocr_runtime_identity=ocr_runtime_identity,
        formal_counts_enforced=formal_counts_enforced,
    )
    payloads = {
        GITATTRIBUTES_RELATIVE_PATH: artifact_gitattributes_bytes(),
        README_RELATIVE_PATH: artifact_readme_bytes(dataset_repo=dataset_repo),
        IMAGE_TAR_RELATIVE_PATH: image_tar_bytes,
        MANIFEST_RELATIVE_PATH: pretty_json_bytes(manifest),
        OCR_JSONL_RELATIVE_PATH: ocr_bytes,
        TRAJECTORY_JSONL_RELATIVE_PATH: trajectory_bytes,
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
        raise ValueError("expansion artifact exact-six inventory drifted")
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
    expected_keys = {
        "schema_version",
        "protocol_id",
        "artifact_id",
        "status",
        "dataset_repo",
        "license",
        "policy_output_generated",
        "restoration_output_generated",
        "per_decision_view_current_expert_action_payload_included",
        "consumer_must_slice_events_by_history_event_step_ids",
        "formal_counts_enforced",
        "inputs",
        "generator",
        "ocr_backend",
        "ocr_runtime",
        "source_dataset",
        "role_source_ids",
        "counts",
        "inventories",
        "payload_files",
    }
    if set(manifest) != expected_keys:
        raise ValueError("expansion artifact manifest fields drifted")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("artifact_id") != ARTIFACT_ID
        or manifest.get("status") != STATUS
    ):
        raise ValueError("expansion artifact identity drifted")
    if manifest.get("license") != "cc-by-4.0":
        raise ValueError("expansion artifact license drifted")
    if manifest.get("policy_output_generated") is not False:
        raise ValueError("expansion artifact cannot contain policy output")
    if manifest.get("restoration_output_generated") is not False:
        raise ValueError("expansion artifact cannot contain restoration output")
    if manifest.get(
        "per_decision_view_current_expert_action_payload_included"
    ) is not False:
        raise ValueError("expansion decision views cannot contain current actions")
    if manifest.get("consumer_must_slice_events_by_history_event_step_ids") is not True:
        raise ValueError("expansion consumers must enforce decision history slicing")
    if type(manifest.get("formal_counts_enforced")) is not bool:
        raise ValueError("formal_counts_enforced must be bool")
    if not isinstance(manifest.get("dataset_repo"), str) or not manifest[
        "dataset_repo"
    ]:
        raise ValueError("expansion dataset_repo must be non-empty")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != INPUT_KEYS:
        raise ValueError("expansion artifact input inventory drifted")
    for key, record in inputs.items():
        if not isinstance(record, Mapping):
            raise ValueError("expansion artifact input identities must be objects")
        _identity_record(record, field=str(key))
    generator = manifest.get("generator")
    if not isinstance(generator, Mapping):
        raise ValueError("expansion artifact generator is missing")
    _validate_generator(generator)
    roles = manifest.get("role_source_ids")
    if not isinstance(roles, Mapping) or set(roles) != set(DERIVED_ROLE_ORDER):
        raise ValueError("expansion role inventory drifted")
    source_ids = [source_id for role in DERIVED_ROLE_ORDER for source_id in roles[role]]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("expansion role source IDs overlap")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(EXPECTED_FORMAL_COUNTS):
        raise ValueError("expansion artifact count fields drifted")
    if manifest["formal_counts_enforced"] and dict(counts) != EXPECTED_FORMAL_COUNTS:
        raise ValueError("formal expansion artifact counts drifted")
    if manifest["formal_counts_enforced"] and {
        role: len(roles[role]) for role in DERIVED_ROLE_ORDER
    } != EXPECTED_ROLE_COUNTS:
        raise ValueError("formal expansion artifact role counts drifted")
    inventories = manifest.get("inventories")
    if not isinstance(inventories, Mapping) or set(inventories) != {
        "image_members_sha256",
        "ocr_records_sha256",
        "ocr_record_aggregate_sha256",
        "trajectory_index_sha256",
    }:
        raise ValueError("expansion artifact inventory digests drifted")
    for key, value in inventories.items():
        _require_sha256(value, f"inventories.{key}")
    if not isinstance(manifest.get("ocr_backend"), Mapping):
        raise ValueError("expansion OCR backend provenance is missing")
    if manifest["ocr_backend"].get("backend_id") != BACKEND_ID:
        raise ValueError("expansion OCR backend ID drifted")
    if manifest["ocr_backend"].get("backend_config_sha256") != inputs[
        "ocr_backend_config"
    ]["sha256"]:
        raise ValueError("expansion OCR backend input binding drifted")
    if not isinstance(manifest.get("ocr_runtime"), Mapping) or set(
        manifest["ocr_runtime"]
    ) != OCR_RUNTIME_IDENTITY_KEYS:
        raise ValueError("expansion OCR runtime identity drifted")
    source = manifest.get("source_dataset")
    if not isinstance(source, Mapping) or set(source) != {
        "upstream_repo",
        "upstream_revision",
        "transport_repo",
        "transport_revision",
        "license",
    }:
        raise ValueError("expansion source dataset identity drifted")
    if source.get("license") != "cc-by-4.0":
        raise ValueError("expansion source license drifted")
    payload_files = manifest.get("payload_files")
    if not isinstance(payload_files, list) or len(payload_files) != 3:
        raise ValueError("expansion payload file inventory drifted")


def _validate_payload_files(
    root: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    expected_paths = [
        IMAGE_TAR_RELATIVE_PATH,
        OCR_JSONL_RELATIVE_PATH,
        TRAJECTORY_JSONL_RELATIVE_PATH,
    ]
    if [record.get("path") for record in records] != expected_paths:
        raise ValueError("expansion payload file order drifted")
    counts: dict[str, int] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("expansion payload records must be objects")
        count_key = (
            "member_count"
            if record.get("path") == IMAGE_TAR_RELATIVE_PATH
            else "record_count"
        )
        if set(record) != {"path", "size_bytes", "sha256", count_key}:
            raise ValueError("expansion payload record schema drifted")
        if (
            type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
            or type(record[count_key]) is not int
            or record[count_key] <= 0
        ):
            raise ValueError("expansion payload record counts or size are invalid")
        _require_sha256(record["sha256"], "payload record sha256")
        path = root.joinpath(*PurePosixPath(str(record["path"])).parts)
        if (
            path.stat().st_size != record.get("size_bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError("expansion payload file identity drifted")
        counts[str(record["path"])] = int(record[count_key])
    return counts


def _validate_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    image_payloads: Mapping[str, bytes],
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
    prepared_by_path: Mapping[str, PreparedImage],
    selection_manifest: Mapping[str, Any] | None,
    event_builder: Callable[..., Mapping[str, Any]],
) -> set[str]:
    expected_order = [
        (role, source_id)
        for role in DERIVED_ROLE_ORDER
        for source_id in manifest["role_source_ids"][role]
    ]
    if [(record.get("role"), record.get("source_id")) for record in trajectories] != (
        expected_order
    ):
        raise ValueError("expansion trajectory order drifted")
    selected_by_source: dict[str, Mapping[str, Any]] = {}
    states_by_source: dict[str, list[Mapping[str, Any]]] = {}
    if selection_manifest is not None:
        _, selected_by_source, states_by_source = _selection_index(selection_manifest)
        expected_role_source_ids = {
            role: [
                str(record["source_id"])
                for record in selection_manifest["splits"][role]["trajectories"]
            ]
            for role in DERIVED_ROLE_ORDER
        }
        if manifest["role_source_ids"] != expected_role_source_ids:
            raise ValueError(
                "expansion role/source order differs from frozen selection"
            )

    required_paths: set[str] = set()
    for trajectory in trajectories:
        if set(trajectory) != TRAJECTORY_KEYS:
            raise ValueError("expansion trajectory fields drifted")
        source_id = str(trajectory["source_id"])
        if trajectory["instruction_sha256"] != sha256_bytes(
            str(trajectory["instruction"]).encode("utf-8")
        ):
            raise ValueError("expansion instruction witness drifted")
        if selection_manifest is not None and trajectory["selection"] != (
            selected_by_source[source_id]
        ):
            raise ValueError("expansion selection witness drifted")
        source = trajectory["source"]
        selection = trajectory["selection"]
        if (
            source["transport_file"] != selection["transport_file"]
            or source["transport_row_index"] != selection["transport_row_index"]
        ):
            raise ValueError("expansion transport-row witness drifted")
        if any(
            source.get(field) != value
            for field, value in manifest["source_dataset"].items()
        ):
            raise ValueError("expansion source dataset witness drifted")
        events = trajectory["events"]
        if not isinstance(events, list) or [event.get("step_id") for event in events] != (
            list(range(1, len(events) + 1))
        ):
            raise ValueError("expansion event prefix drifted")
        for event in events:
            if not isinstance(event, Mapping) or set(event) != EVENT_KEYS:
                raise ValueError("expansion event fields drifted")
            required_paths.update(
                (event["observation_before_path"], event["observation_after_path"])
            )
            if dict(
                event_builder(
                    event,
                    image_payloads=image_payloads,
                    ocr_records_by_path=ocr_records_by_path,
                    prepared_by_path=prepared_by_path,
                )
            ) != event:
                raise ValueError("expansion event cannot be replayed")
        decisions = trajectory["decisions"]
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("expansion decisions must be non-empty")
        if FORBIDDEN_DECISION_KEYS.intersection(_walk_keys(decisions)):
            raise ValueError("current expert action leaked into expansion decisions")
        if selection_manifest is not None and len(decisions) != len(
            states_by_source[source_id]
        ):
            raise ValueError("expansion state count drifted from selection")
        for index, decision in enumerate(decisions):
            if not isinstance(decision, Mapping) or set(decision) != DECISION_KEYS:
                raise ValueError("expansion decision fields drifted")
            if decision["current_expert_action_payload_included"] is not False:
                raise ValueError("current expert action payload must be omitted")
            step = int(decision["decision_step_id"])
            history = list(decision["history_event_step_ids"])
            candidates = list(decision["candidate_event_step_ids"])
            current_equivalent = int(decision["current_equivalent_event_step_id"])
            if (
                history != list(range(1, step))
                or candidates != history[:-1]
                or current_equivalent != history[-1]
            ):
                raise ValueError("expansion decision geometry drifted")
            view = decision_view_events(events, decision)
            if [event["step_id"] for event in view] != history:
                raise ValueError("expansion decision view history slice drifted")
            if selection_manifest is not None:
                state = states_by_source[source_id][index]
                for key in (
                    "state_id",
                    "decision_step_id",
                    "history_event_step_ids",
                    "candidate_event_step_ids",
                    "current_equivalent_event_step_id",
                ):
                    if decision[key] != state[key]:
                        raise ValueError("expansion decision differs from selection")
            current_path = str(decision["current_observation_path"])
            current_sha = decision["current_observation_sha256"]
            required_paths.add(current_path)
            equivalent_event = events[current_equivalent - 1]
            if (
                sha256_bytes(image_payloads[current_path]) != current_sha
                or equivalent_event["observation_after_path"] != current_path
                or equivalent_event["observation_after_sha256"] != current_sha
            ):
                raise ValueError("expansion current observation witness drifted")
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
                raise ValueError("expansion candidate content witnesses drifted")
            expected_current = {
                "event_step_id": current_equivalent,
                "post_state_member_path": current_path,
                "post_state_sha256": current_sha,
            }
            if decision["current_equivalence_witness"] != expected_current:
                raise ValueError("expansion current-equivalence witness drifted")
            if decision["content_witness_sha256"] != _decision_content_witness(
                source_id=source_id,
                state_id=str(decision["state_id"]),
                current_observation_path=current_path,
                current_observation_sha256=current_sha,
                candidate_event_post_states=expected_candidates,
            ):
                raise ValueError("expansion decision content digest drifted")
        if trajectory["content_witness_sha256"] != _trajectory_content_witness(
            trajectory
        ):
            raise ValueError("expansion trajectory content digest drifted")
    return required_paths


def validate_artifact(
    *,
    output_dir: str | Path,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    selection_manifest: Mapping[str, Any] | None = None,
    parent_selection: Mapping[str, Any] | None = None,
    selection_manifest_sha256: str | None = None,
    parent_selection_sha256: str | None = None,
    expected_input_sha256: Mapping[str, str] | None = None,
    expected_ocr_backend_provenance: Mapping[str, Any] | None = None,
    expected_ocr_runtime_identity: Mapping[str, Any] | None = None,
    expected_generator_git_revision: str | None = None,
    expected_dataset_repo: str | None = None,
    expected_source_dataset: Mapping[str, str] | None = None,
    repository_root: str | Path | None = None,
    require_formal: bool = False,
    ocr_engine: Any | None = None,
    require_ocr_replay: bool = False,
    ocr_record_runner: Callable[..., Mapping[str, Any]] = run_rapidocr_record,
    ocr_record_validator: Callable[..., PreparedImage] = validate_ocr_record,
    event_builder: Callable[..., Mapping[str, Any]] = _build_derived_event,
) -> dict[str, Any]:
    validate_backend_config(backend_config)
    if type(require_formal) is not bool or type(require_ocr_replay) is not bool:
        raise TypeError("formal and OCR replay requirements must be bool")
    if require_formal:
        required = {
            "selection_manifest": selection_manifest,
            "parent_selection": parent_selection,
            "selection_manifest_sha256": selection_manifest_sha256,
            "parent_selection_sha256": parent_selection_sha256,
            "expected_input_sha256": expected_input_sha256,
            "expected_ocr_backend_provenance": expected_ocr_backend_provenance,
            "expected_ocr_runtime_identity": expected_ocr_runtime_identity,
            "expected_generator_git_revision": expected_generator_git_revision,
            "expected_dataset_repo": expected_dataset_repo,
            "expected_source_dataset": expected_source_dataset,
            "repository_root": repository_root,
        }
        missing = sorted(key for key, value in required.items() if value is None)
        if missing:
            raise ValueError(f"formal expansion validation bindings missing: {missing}")
        if not require_ocr_replay or ocr_engine is None:
            raise ValueError("formal expansion validation requires full OCR replay")

    root = Path(output_dir)
    tree = artifact_tree_identity(root)
    if (root / GITATTRIBUTES_RELATIVE_PATH).read_bytes() != (
        artifact_gitattributes_bytes()
    ):
        raise ValueError("expansion .gitattributes drifted")
    manifest_payload, manifest = _load_json_object(root / MANIFEST_RELATIVE_PATH)
    if manifest_payload != pretty_json_bytes(manifest):
        raise ValueError("expansion manifest is not canonical pretty JSON")
    _validate_manifest(manifest)
    if require_formal and manifest["formal_counts_enforced"] is not True:
        raise ValueError("formal expansion validation requires formal counts")
    if (root / README_RELATIVE_PATH).read_bytes() != artifact_readme_bytes(
        dataset_repo=manifest["dataset_repo"]
    ):
        raise ValueError("expansion artifact README drifted")
    if manifest["ocr_backend"]["backend_config_sha256"] != backend_config_sha256:
        raise ValueError("expansion OCR backend config digest drifted")
    if expected_ocr_backend_provenance is not None and manifest[
        "ocr_backend"
    ] != expected_ocr_backend_provenance:
        raise ValueError("expansion OCR backend provenance drifted")
    validate_ocr_runtime_identity(manifest["ocr_runtime"], backend_config=backend_config)
    if expected_ocr_runtime_identity is not None and manifest[
        "ocr_runtime"
    ] != expected_ocr_runtime_identity:
        raise ValueError("expansion OCR runtime identity drifted")
    if expected_generator_git_revision is not None and manifest["generator"][
        "git_revision"
    ] != expected_generator_git_revision:
        raise ValueError("expansion generator revision drifted")
    if expected_dataset_repo is not None and manifest["dataset_repo"] != (
        expected_dataset_repo
    ):
        raise ValueError("expansion dataset repo drifted")
    if expected_source_dataset is not None and manifest["source_dataset"] != dict(
        expected_source_dataset
    ):
        raise ValueError("expansion source dataset drifted")
    if expected_input_sha256 is not None:
        if set(expected_input_sha256) != INPUT_KEYS:
            raise ValueError("expected expansion input inventory drifted")
        for key, digest in expected_input_sha256.items():
            _require_sha256(digest, f"expected_input_sha256.{key}")
            if manifest["inputs"][key]["sha256"] != digest:
                raise ValueError(f"expansion input digest drifted: {key}")
        if selection_manifest is not None:
            expected_selection_sha = (
                selection_manifest_sha256
                if selection_manifest_sha256 is not None
                else expected_input_sha256["expansion_selection_manifest"]
            )
            _bind_supplied_manifest_sha256(
                selection_manifest,
                expected_sha256=expected_selection_sha,
                field="expansion_selection_manifest",
            )
            if expected_selection_sha != expected_input_sha256[
                "expansion_selection_manifest"
            ]:
                raise ValueError("expansion selection external bindings disagree")
        if parent_selection is not None:
            expected_parent_sha = (
                parent_selection_sha256
                if parent_selection_sha256 is not None
                else expected_input_sha256["parent_selection_manifest"]
            )
            _bind_supplied_manifest_sha256(
                parent_selection,
                expected_sha256=expected_parent_sha,
                field="parent_selection_manifest",
            )
            if expected_parent_sha != expected_input_sha256[
                "parent_selection_manifest"
            ]:
                raise ValueError("parent selection external bindings disagree")
    if repository_root is not None:
        checkout = Path(repository_root)
        for prefix in ("module", "build_cli", "validator"):
            path = checkout.joinpath(manifest["generator"][f"{prefix}_path"])
            if not path.is_file() or sha256_file(path) != manifest["generator"][
                f"{prefix}_sha256"
            ]:
                raise ValueError(f"expansion generator source drifted: {prefix}")
    payload_counts = _validate_payload_files(root, manifest["payload_files"])

    image_payloads, image_members = _validate_tar(root / IMAGE_TAR_RELATIVE_PATH)
    if require_formal:
        assert selection_manifest is not None
        assert parent_selection is not None
        validate_formal_image_inventory(
            tuple(image_payloads), selection_manifest=selection_manifest
        )
        for path, payload in image_payloads.items():
            _validate_image_suffix(path, payload)
        expansion_ids = {
            source_id
            for role in DERIVED_ROLE_ORDER
            for source_id in manifest["role_source_ids"][role]
        }
        if expansion_ids.intersection(_forbidden_prior_source_ids(parent_selection)):
            raise ValueError("formal artifact includes legacy or confirm source IDs")

    ocr_records = parse_canonical_jsonl(
        (root / OCR_JSONL_RELATIVE_PATH).read_bytes(), label="expansion OCR"
    )
    ocr_records_by_path: dict[str, Mapping[str, Any]] = {}
    prepared_by_path: dict[str, PreparedImage] = {}
    for record in ocr_records:
        path = _safe_member_path(record.get("image_member_path"))
        if path in ocr_records_by_path or path not in image_payloads:
            raise ValueError("expansion OCR image identity is duplicate or absent")
        prepared_by_path[path] = ocr_record_validator(
            record,
            image_bytes=image_payloads[path],
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
        )
        ocr_records_by_path[path] = record
    if list(ocr_records_by_path) != sorted(ocr_records_by_path):
        raise ValueError("expansion OCR records are not sorted")
    if set(ocr_records_by_path) != set(image_payloads):
        raise ValueError("expansion OCR and image inventories differ")

    replay_count = 0
    replay_sha256: str | None = None
    if require_ocr_replay:
        if ocr_engine is None:
            raise ValueError("OCR replay requires an engine")
        replayed, replay_sha256 = generate_ocr_records(
            engine=ocr_engine,
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
            image_payloads=image_payloads,
            record_runner=ocr_record_runner,
        )
        replay_count = len(replayed)
        if replayed != ocr_records_by_path:
            raise ValueError("full expansion OCR replay differs from archived records")

    trajectories = parse_canonical_jsonl(
        (root / TRAJECTORY_JSONL_RELATIVE_PATH).read_bytes(),
        label="expansion trajectories",
    )
    required_paths = _validate_trajectories(
        trajectories,
        manifest=manifest,
        image_payloads=image_payloads,
        ocr_records_by_path=ocr_records_by_path,
        prepared_by_path=prepared_by_path,
        selection_manifest=selection_manifest,
        event_builder=event_builder,
    )
    if required_paths != set(image_payloads):
        raise ValueError("expansion images contain missing or unused members")
    counts = {
        "trajectory_count": len(trajectories),
        "event_count": sum(len(record["events"]) for record in trajectories),
        "state_count": sum(len(record["decisions"]) for record in trajectories),
        "image_member_count": len(image_members),
        "ocr_record_count": len(ocr_records),
    }
    if counts != manifest["counts"]:
        raise ValueError("expansion artifact counts are inconsistent")
    if payload_counts != {
        IMAGE_TAR_RELATIVE_PATH: len(image_members),
        OCR_JSONL_RELATIVE_PATH: len(ocr_records),
        TRAJECTORY_JSONL_RELATIVE_PATH: len(trajectories),
    }:
        raise ValueError("expansion payload member/record counts drifted")
    if require_formal and counts != EXPECTED_FORMAL_COUNTS:
        raise ValueError("formal expansion artifact counts drifted")
    trajectory_index = [
        {
            "source_id": record["source_id"],
            "role": record["role"],
            "event_count": len(record["events"]),
            "state_count": len(record["decisions"]),
            "content_witness_sha256": record["content_witness_sha256"],
        }
        for record in trajectories
    ]
    ocr_index = [
        {
            "image_member_path": record["image_member_path"],
            "image_sha256": record["image_sha256"],
            "canonical_ocr_record_sha256": record[
                "canonical_ocr_record_sha256"
            ],
        }
        for record in ocr_records
    ]
    inventories = {
        "image_members_sha256": sha256_bytes(canonical_json_bytes(image_members)),
        "ocr_records_sha256": sha256_bytes(canonical_json_bytes(ocr_index)),
        "ocr_record_aggregate_sha256": ocr_record_aggregate_sha256(
            ocr_records_by_path
        ),
        "trajectory_index_sha256": sha256_bytes(
            canonical_json_bytes(trajectory_index)
        ),
    }
    if inventories != manifest["inventories"]:
        raise ValueError("expansion artifact inventory digests drifted")
    return {
        "outcome": "PASSED_GUIODYSSEY_RESTORATION_V2_EXPANSION_VALIDATION",
        "artifact_tree_sha256": tree["artifact_tree_sha256"],
        "counts": counts,
        "formal_counts_enforced": manifest["formal_counts_enforced"],
        "ocr_record_aggregate_sha256": inventories[
            "ocr_record_aggregate_sha256"
        ],
        "ocr_replay_performed": require_ocr_replay,
        "ocr_replay_record_count": replay_count,
        "ocr_replay_aggregate_sha256": replay_sha256,
        "policy_loaded": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "per_decision_view_current_expert_action_payload_included": False,
        "consumer_must_slice_events_by_history_event_step_ids": True,
    }


__all__ = [
    "ARTIFACT_ID",
    "ARTIFACT_RELATIVE_PATHS",
    "DERIVED_ROLE_ORDER",
    "EXPECTED_FORMAL_COUNTS",
    "FROZEN_OCR_BACKEND_MANIFEST_SHA256",
    "PAYLOAD_PREFIX",
    "artifact_tree_identity",
    "build_ocr_backend_provenance",
    "build_trajectory_records",
    "decision_view_events",
    "generate_ocr_records",
    "load_expansion_source_pilots",
    "required_image_member_paths",
    "source_dataset_identity_from_v1_config",
    "validate_artifact",
    "validate_formal_image_inventory",
    "validate_frozen_inputs",
    "write_artifact",
]
