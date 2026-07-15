"""Deterministic policy-blind GUIOdyssey restoration-v2 derived dataset."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.guiodyssey import build_pilot_manifest, canonicalize_tool_call
from causalcache.data.guiodyssey_independent import (
    source_file_specs,
    verify_local_source_files,
)
from causalcache.data.restoration_v2_selection import (
    validate_exposure_ledger,
    validate_selection_manifest,
    validate_state_content_witnesses,
)
from causalcache.low_fidelity_v2 import (
    LowFidelityEventV2,
    ScreenTextNode,
    action_argument,
    normalize_screen_text,
    screen_change_from_mean_absolute_rgb_difference,
    screen_text_delta,
    serialize_low_fidelity_v2,
)
from causalcache.restoration_v2_text_backend import (
    BACKEND_ID,
    OCR_RECORD_SCHEMA_VERSION,
    PreparedImage,
    canonicalize_rapidocr_nodes,
    mean_absolute_rgb_difference_from_prepared,
    prepare_selected_guiodyssey_image,
    run_rapidocr_record,
    validate_backend_config,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_restoration_v2"
ARTIFACT_ID = "causalcache-guiodyssey-restoration-v2-mobile-v1"
PAYLOAD_PREFIX = "derived/restoration-v2-v1"
IMAGE_TAR_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/images-00000-of-00001.tar"
OCR_JSONL_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/ocr-records-00000-of-00001.jsonl"
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
    "v2_label_train",
    "v2_development",
    "v2_confirm_primary",
)
EXPECTED_FORMAL_COUNTS = {
    "trajectory_count": 35,
    "event_count": 175,
    "state_count": 65,
    "image_member_count": 210,
    "ocr_record_count": 210,
}
FROZEN_OCR_BACKEND_MANIFEST_SHA256 = (
    "107478672438b52e9c2ccc9ef8de5d13d4e16329ab1afbe95772e2d3df5bd2a9"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
OCR_RECORD_KEYS = {
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
OCR_RUNTIME_IDENTITY_KEYS = {
    "runtime_packages",
    "rapidocr_package_file_sha256",
    "wheel_sha256",
    "model_sha256",
    "recognizer_character_inventory",
}
EVENT_KEYS = {
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
DECISION_KEYS = {
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


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _safe_member_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact member path must be a safe relative POSIX path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or str(parsed) != value
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise ValueError("artifact member path must be a safe relative POSIX path")
    return value


def _strict_json_loads(payload: str | bytes) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(payload, object_pairs_hook=pairs_hook)


def load_json_object(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    payload = Path(path).read_bytes()
    value = _strict_json_loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload, value


def canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    if not records:
        raise ValueError("canonical JSONL requires at least one record")
    return b"\n".join(canonical_json_bytes(dict(record)) for record in records) + b"\n"


def parse_canonical_jsonl(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be non-empty canonical newline-terminated JSONL")
    raw_lines = payload.splitlines()
    if not raw_lines or any(not line for line in raw_lines):
        raise ValueError(f"{label} cannot contain blank JSONL records")
    records: list[dict[str, Any]] = []
    for line in raw_lines:
        value = _strict_json_loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label} JSONL records must be objects")
        if line != canonical_json_bytes(value):
            raise ValueError(f"{label} JSONL is not canonical compact JSON")
        records.append(value)
    return records


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
        "# CausalCache GUIOdyssey restoration-v2 derived dataset\n\n"
        "Policy-blind, deterministic restoration-v2 source artifact. It contains "
        "no policy or restoration output.\n\n"
        "## Schema and formal counts\n\n"
        f"- Schema: `{SCHEMA_VERSION}` (`{ARTIFACT_ID}`)\n"
        "- 35 trajectories, 175 events, 65 decision states\n"
        "- 210 archived GUI observations and 210 full uncapped OCR records\n\n"
        "## Exact six-file layout\n\n"
        "- `.gitattributes`\n"
        "- `README.md`\n"
        "- `derived/restoration-v2-v1/images-00000-of-00001.tar`\n"
        "- `derived/restoration-v2-v1/manifest.json`\n"
        "- `derived/restoration-v2-v1/ocr-records-00000-of-00001.jsonl`\n"
        "- `derived/restoration-v2-v1/trajectories-00000-of-00001.jsonl`\n\n"
        "The shared HF repo may also contain the existing `golden/real-screen-v1` "
        "prefix. Download formal derived data with an exact `allow_patterns` "
        "projection of the six paths above into a clean root; validation applies "
        "an exact six-file allowlist to that projected root, not to every file in "
        "the full HF revision.\n\n"
        "## Provenance and reproducibility\n\n"
        "GUIOdyssey upstream/transport revisions, frozen selection and exposure "
        "identities, OCR backend/model/wheel hashes, installed package source "
        "hashes, runtime packages, and recognizer inventory are recorded in "
        "`manifest.json`. Formal validation replays all 210 OCR records exactly.\n\n"
        "Build with `code/scripts/build_guiodyssey_restoration_v2.py` using "
        "`--v2-contract`, `--v1-config`, `--source-file-manifest`, `--source-root`, "
        "`--selection-manifest`, `--exposure-manifest`, `--ocr-backend-config`, "
        "`--ocr-backend-manifest`, `--model-dir`, `--wheel-dir`, `--git-revision`, "
        "and `--output-dir`. Validate a downloaded artifact with "
        "`code/scripts/validate_guiodyssey_restoration_v2.py` using the same pinned "
        "inputs except `--source-root`, plus `--output-dir`.\n\n"
        f"Hugging Face dataset repo: `{dataset_repo}`. Planned stable tag: "
        "`restoration-v2-derived-v1.0.0`. The immutable upload revision is written "
        "back through the Git completion manifest after upload to avoid "
        "self-reference.\n"
    ).encode("utf-8")


def artifact_gitattributes_bytes() -> bytes:
    return b"*.tar filter=lfs diff=lfs merge=lfs -text\n"


def _screen_text_nodes(record: Mapping[str, Any]) -> tuple[ScreenTextNode, ...]:
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("OCR nodes must be a JSON array")
    result = []
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("OCR nodes must be JSON objects")
        bbox = node.get("bbox_top_left_bottom_right")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("OCR node bbox must contain top,left,bottom,right")
        result.append(
            ScreenTextNode(
                text=str(node["normalized_text"]),
                top=bbox[0],
                left=bbox[1],
                bottom=bbox[2],
                right=bbox[3],
            )
        )
    return tuple(result)


def validate_ocr_record(
    record: Mapping[str, Any],
    *,
    image_bytes: bytes,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
) -> PreparedImage:
    validate_backend_config(backend_config)
    _require_sha256(backend_config_sha256, "backend_config_sha256")
    if set(record) != OCR_RECORD_KEYS:
        raise ValueError("OCR record fields drifted from the pinned full schema")
    if record.get("schema_version") != OCR_RECORD_SCHEMA_VERSION:
        raise ValueError("OCR record schema_version drifted")
    if record.get("backend_id") != BACKEND_ID:
        raise ValueError("OCR record backend_id drifted")
    if record.get("backend_config_sha256") != backend_config_sha256:
        raise ValueError("OCR record backend config identity drifted")
    member_path = _safe_member_path(record.get("image_member_path"))
    if not member_path.startswith("images/"):
        raise ValueError("OCR image_member_path must belong to images/")
    if record.get("image_sha256") != sha256_bytes(image_bytes):
        raise ValueError("OCR record image SHA256 drifted")

    prepared = prepare_selected_guiodyssey_image(image_bytes, backend_config)
    expected_prepared = {
        "source_format": prepared.source_format,
        "source_mode": prepared.source_mode,
        "width": prepared.width,
        "height": prepared.height,
        "exif_present": prepared.exif_present,
        "alpha_extrema": (
            list(prepared.alpha_extrema)
            if prepared.alpha_extrema is not None
            else None
        ),
        "rgb_bytes_sha256": prepared.rgb_bytes_sha256,
        "resized_rgb_256x256_sha256": prepared.resized_rgb_bytes_sha256,
    }
    for field, value in expected_prepared.items():
        if record.get(field) != value:
            raise ValueError(f"OCR record prepared-image field drifted: {field}")

    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("OCR record nodes must be a JSON array")
    try:
        canonical_nodes = canonicalize_rapidocr_nodes(
            boxes=[node["polygon_xy"] for node in nodes],
            texts=[node["raw_text"] for node in nodes],
            scores=[float(node["confidence_decimal_string"]) for node in nodes],
            width=prepared.width,
            height=prepared.height,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("OCR record node schema is invalid") from error
    if list(canonical_nodes) != nodes:
        raise ValueError("OCR record nodes are not pinned canonical nodes")
    full_tokens = [
        token
        for node in nodes
        for token in str(node["normalized_text"]).split(" ")
    ]
    if record.get("full_spatial_tokens") != full_tokens:
        raise ValueError("OCR record full spatial tokens are not uncapped canonical tokens")
    if record.get("full_spatial_tokens_sha256") != sha256_bytes(
        canonical_json_bytes(full_tokens)
    ):
        raise ValueError("OCR record full spatial token SHA256 drifted")
    unhashed = dict(record)
    observed_record_sha = unhashed.pop("canonical_ocr_record_sha256")
    if observed_record_sha != sha256_bytes(canonical_json_bytes(unhashed)):
        raise ValueError("OCR record canonical SHA256 drifted")
    return prepared


def source_tool_call_to_v2_arguments(
    source_tool_call: Mapping[str, Any],
    *,
    executed_action: Mapping[str, Any],
) -> dict[str, Any]:
    function = source_tool_call.get("function")
    if not isinstance(function, Mapping) or set(function) != {"name", "arguments"}:
        raise ValueError("GUIOdyssey source tool call must contain name and arguments")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("GUIOdyssey source function is malformed")
    expected_executed_keys = {
        "action_type",
        "target",
        "text_argument",
        "text_case_sensitive",
    }
    canonical_action, _ = canonicalize_tool_call(source_tool_call, grid_size=10)
    expected_executed_action = {
        "action_type": canonical_action.action_type.value,
        "target": canonical_action.target,
        "text_argument": canonical_action.text_argument,
        "text_case_sensitive": canonical_action.text_case_sensitive,
    }
    if set(executed_action) != expected_executed_keys or dict(
        executed_action
    ) != expected_executed_action:
        raise ValueError(
            "GUIOdyssey source tool call differs from the full canonical "
            "executed action"
        )
    observed_action_type = executed_action.get("action_type")

    if name == "tap":
        if set(arguments) not in ({"coordinate"}, {"coordinate", "clicks"}):
            raise ValueError("GUIOdyssey tap arguments drifted")
        if "clicks" in arguments and (
            type(arguments["clicks"]) is not int or arguments["clicks"] != 1
        ):
            raise ValueError("GUIOdyssey tap clicks must equal one")
        if observed_action_type != "tap":
            raise ValueError("GUIOdyssey tap source/executed action mismatch")
        return {"action": "click", "coordinate": arguments["coordinate"]}
    if name == "long_press":
        if set(arguments) != {"coordinate"} or observed_action_type != "long_press":
            raise ValueError("GUIOdyssey long_press source/executed action mismatch")
        return {"action": "long_press", "coordinate": arguments["coordinate"]}
    if name == "swipe":
        if set(arguments) != {"start_coordinate", "coordinate"}:
            raise ValueError("GUIOdyssey swipe arguments drifted")
        if observed_action_type != "swipe":
            raise ValueError("GUIOdyssey swipe source/executed action mismatch")
        return {
            "action": "swipe",
            "coordinate": arguments["start_coordinate"],
            "coordinate2": arguments["coordinate"],
        }
    if name == "type":
        if set(arguments) != {"text"} or observed_action_type != "type_text":
            raise ValueError("GUIOdyssey type source/executed action mismatch")
        return {"action": "type", "text": arguments["text"]}
    if name == "system_button":
        if set(arguments) != {"button"} or not isinstance(arguments["button"], str):
            raise ValueError("GUIOdyssey system_button arguments drifted")
        button_by_source = {"back": "Back", "home": "Home", "enter": "Enter"}
        button = button_by_source.get(arguments["button"].casefold())
        expected_action = {
            "Back": "back",
            "Home": "home",
            "Enter": "enter",
        }.get(button)
        if button is None or observed_action_type != expected_action:
            raise ValueError("GUIOdyssey system_button is outside the v2 action contract")
        return {"action": "system_button", "button": button}
    if name in {"open", "answer"}:
        expected_action = name
        if set(arguments) != {"text"} or observed_action_type != expected_action:
            raise ValueError(f"GUIOdyssey {name} source/executed action mismatch")
        return {"action": name, "text": arguments["text"]}
    if name == "wait":
        if arguments or observed_action_type != "wait":
            raise ValueError("GUIOdyssey wait source/executed action mismatch")
        return {"action": "wait"}
    if name == "stop":
        if arguments or observed_action_type != "stop":
            raise ValueError("GUIOdyssey stop source/executed action mismatch")
        return {"action": "terminate", "status": "success"}
    raise ValueError(f"GUIOdyssey source action is outside restoration v2: {name}")


def _build_derived_event(
    event: Mapping[str, Any],
    *,
    image_payloads: Mapping[str, bytes],
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
    prepared_by_path: Mapping[str, PreparedImage],
) -> dict[str, Any]:
    step_id = event.get("step_id")
    if type(step_id) is not int or step_id < 1:
        raise ValueError("GUIOdyssey event step_id must be a positive integer")
    before_path = _safe_member_path(event.get("observation_before_path"))
    after_path = _safe_member_path(event.get("observation_after_path"))
    if before_path == after_path:
        raise ValueError("event before/after member paths must differ")
    try:
        before_bytes = image_payloads[before_path]
        after_bytes = image_payloads[after_path]
        before_ocr = ocr_records_by_path[before_path]
        after_ocr = ocr_records_by_path[after_path]
        before_prepared = prepared_by_path[before_path]
        after_prepared = prepared_by_path[after_path]
    except KeyError as error:
        raise ValueError("event is missing image, OCR, or prepared-image input") from error

    executed_action = event.get("executed_action")
    source_tool_call = event.get("source_tool_call")
    if not isinstance(executed_action, Mapping) or not isinstance(
        source_tool_call, Mapping
    ):
        raise ValueError("event source action fields must be objects")
    v2_arguments = source_tool_call_to_v2_arguments(
        source_tool_call,
        executed_action=executed_action,
    )
    text_delta = screen_text_delta(
        _screen_text_nodes(before_ocr),
        _screen_text_nodes(after_ocr),
    )
    mean_difference = mean_absolute_rgb_difference_from_prepared(
        before_prepared,
        after_prepared,
    )
    low_fidelity = LowFidelityEventV2(
        step_id=step_id,
        action_type=str(v2_arguments["action"]),
        action_argument=action_argument(v2_arguments),
        foreground_app="unknown",
        screen_text_added=text_delta.added,
        screen_text_removed=text_delta.removed,
        screen_change=screen_change_from_mean_absolute_rgb_difference(
            mean_difference
        ),
        executor_result="unknown",
    )
    serialized = serialize_low_fidelity_v2(low_fidelity)
    before_sha = sha256_bytes(before_bytes)
    after_sha = sha256_bytes(after_bytes)
    return {
        "step_id": step_id,
        "observation_before_path": before_path,
        "observation_before_sha256": before_sha,
        "observation_after_path": after_path,
        "observation_after_sha256": after_sha,
        "executed_action": dict(executed_action),
        "source_tool_call": dict(source_tool_call),
        "low_fidelity_v2": low_fidelity.to_ordered_dict(),
        "low_fidelity_v2_serialized": serialized.decode("utf-8"),
        "low_fidelity_v2_sha256": sha256_bytes(serialized),
        "low_fidelity_v2_metadata": {
            "before_full_spatial_tokens_sha256": before_ocr[
                "full_spatial_tokens_sha256"
            ],
            "after_full_spatial_tokens_sha256": after_ocr[
                "full_spatial_tokens_sha256"
            ],
            "screen_text_added_discarded_count": (
                text_delta.added_discarded_count
            ),
            "screen_text_removed_discarded_count": (
                text_delta.removed_discarded_count
            ),
            "mean_absolute_rgb_difference_decimal_string": format(
                mean_difference,
                ".17g",
            ),
        },
        "high_fidelity_v2": {
            "content_type": "image",
            "image_role": "post_action_state",
            "image_member_path": after_path,
            "image_sha256": after_sha,
            "include_before_image": False,
            "include_additional_action_text": False,
        },
        "ocr_record_refs": {
            "before_canonical_ocr_record_sha256": before_ocr[
                "canonical_ocr_record_sha256"
            ],
            "after_canonical_ocr_record_sha256": after_ocr[
                "canonical_ocr_record_sha256"
            ],
        },
    }


def _states_by_source(
    selection_manifest: Mapping[str, Any],
) -> tuple[
    dict[str, str],
    dict[str, list[Mapping[str, Any]]],
    dict[str, Mapping[str, Any]],
]:
    role_by_source: dict[str, str] = {}
    states_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    selection_record_by_source: dict[str, Mapping[str, Any]] = {}
    roles = selection_manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("selection manifest roles must be an object")
    for role in DERIVED_ROLE_ORDER:
        role_record = roles.get(role)
        if not isinstance(role_record, Mapping):
            raise ValueError(f"selection manifest is missing role {role}")
        trajectories = role_record.get("trajectories")
        states = role_record.get("states")
        if not isinstance(trajectories, list) or not isinstance(states, list):
            raise ValueError("selection role trajectories/states must be arrays")
        for trajectory_record in trajectories:
            if not isinstance(trajectory_record, Mapping):
                raise ValueError("selection trajectory records must be objects")
            source_id = str(trajectory_record.get("source_id"))
            if source_id in role_by_source:
                raise ValueError("selected source IDs must be unique across derived roles")
            role_by_source[source_id] = role
            selection_record_by_source[source_id] = trajectory_record
        for state in states:
            if not isinstance(state, Mapping):
                raise ValueError("selection state records must be objects")
            source_id = str(state.get("source_id"))
            if role_by_source.get(source_id) != role:
                raise ValueError("selection state source ID does not belong to its role")
            states_by_source[source_id].append(state)
    if set(states_by_source) != set(role_by_source):
        raise ValueError("every selected derived trajectory must contain states")
    for source_id, states in states_by_source.items():
        steps = [int(state["decision_step_id"]) for state in states]
        if steps != sorted(steps) or len(steps) != len(set(steps)):
            raise ValueError(f"selected state order drifted: {source_id}")
    return role_by_source, states_by_source, selection_record_by_source


def required_image_member_paths(
    *,
    pilots_by_source: Mapping[str, Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    role_by_source, states_by_source, _ = _states_by_source(selection_manifest)
    if set(pilots_by_source) != set(role_by_source):
        raise ValueError("source pilots must cover exactly the 35 derived trajectories")
    required: set[str] = set()
    for source_id in role_by_source:
        pilot = pilots_by_source[source_id]
        trajectory = pilot.get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise ValueError("pilot manifest trajectory must be an object")
        events = {
            int(event["step_id"]): event
            for event in trajectory.get("events", [])
            if isinstance(event, Mapping)
        }
        max_history_step = max(
            max(int(value) for value in state["history_event_step_ids"])
            for state in states_by_source[source_id]
        )
        if tuple(events)[:max_history_step] != tuple(range(1, max_history_step + 1)):
            raise ValueError("pilot event prefix is incomplete")
        for step_id in range(1, max_history_step + 1):
            event = events[step_id]
            required.add(_safe_member_path(event["observation_before_path"]))
            required.add(_safe_member_path(event["observation_after_path"]))
        for state in states_by_source[source_id]:
            required.add(_safe_member_path(state["current_observation"]["member_path"]))
    return tuple(sorted(required))


def validate_formal_image_inventory(
    image_member_paths: Sequence[str],
    *,
    selection_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    observed = tuple(image_member_paths)
    if observed != tuple(sorted(observed)) or len(observed) != len(set(observed)):
        raise ValueError("formal image member paths must be sorted and unique")
    role_by_source, _, _ = _states_by_source(selection_manifest)
    expected = tuple(
        sorted(
            f"images/{source_id}/observation-{index:03d}.png"
            for source_id in role_by_source
            for index in range(6)
        )
    )
    if len(role_by_source) != EXPECTED_FORMAL_COUNTS["trajectory_count"]:
        raise ValueError("formal OCR inventory must cover exactly 35 trajectories")
    if len(expected) != EXPECTED_FORMAL_COUNTS["image_member_count"]:
        raise RuntimeError("formal OCR expected image inventory is inconsistent")
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(
            f"formal OCR image inventory mismatch: missing={missing}, extra={extra}"
        )
    return observed


def ocr_record_aggregate_sha256(
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
) -> str:
    if not ocr_records_by_path:
        raise ValueError("OCR record aggregate requires at least one record")
    paths = sorted(ocr_records_by_path)
    aggregate = []
    for path in paths:
        record = ocr_records_by_path[path]
        if record.get("image_member_path") != path:
            raise ValueError("OCR record aggregate path identity drifted")
        aggregate.append(
            {
                "image_member_path": path,
                "image_sha256": _require_sha256(
                    record.get("image_sha256"),
                    "OCR aggregate image_sha256",
                ),
                "canonical_ocr_record_sha256": _require_sha256(
                    record.get("canonical_ocr_record_sha256"),
                    "OCR aggregate canonical record SHA256",
                ),
            }
        )
    return sha256_bytes(canonical_json_bytes(aggregate))


def generate_ocr_records(
    *,
    engine: Any,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    image_payloads: Mapping[str, bytes],
    record_runner: Callable[..., Mapping[str, Any]] = run_rapidocr_record,
) -> tuple[dict[str, dict[str, Any]], str]:
    validate_backend_config(backend_config)
    _require_sha256(backend_config_sha256, "backend_config_sha256")
    if not callable(record_runner):
        raise TypeError("record_runner must be callable")
    if not image_payloads:
        raise ValueError("OCR generation requires at least one image")
    paths = sorted(image_payloads)
    if any(_safe_member_path(path) != path for path in paths):
        raise ValueError("OCR generation image paths must be canonical")
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        record = record_runner(
            engine=engine,
            backend_config=backend_config,
            image_member_path=path,
            image_bytes=image_payloads[path],
            backend_config_sha256=backend_config_sha256,
        )
        if not isinstance(record, Mapping):
            raise TypeError("OCR record runner must return a mapping")
        copied = dict(record)
        if copied.get("image_member_path") != path:
            raise ValueError("OCR record runner changed the sorted image member path")
        _require_sha256(copied.get("image_sha256"), "generated OCR image_sha256")
        _require_sha256(
            copied.get("canonical_ocr_record_sha256"),
            "generated OCR canonical record SHA256",
        )
        records[path] = copied
    return records, ocr_record_aggregate_sha256(records)


def build_trajectory_records(
    *,
    pilots_by_source: Mapping[str, Mapping[str, Any]],
    source_image_payloads: Mapping[str, bytes],
    selection_manifest: Mapping[str, Any],
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    role_by_source, states_by_source, selection_by_source = _states_by_source(
        selection_manifest
    )
    required_paths = required_image_member_paths(
        pilots_by_source=pilots_by_source,
        selection_manifest=selection_manifest,
    )
    if set(ocr_records_by_path) != set(required_paths):
        missing = sorted(set(required_paths) - set(ocr_records_by_path))
        extra = sorted(set(ocr_records_by_path) - set(required_paths))
        raise ValueError(f"OCR image inventory mismatch: missing={missing}, extra={extra}")
    try:
        image_payloads = {path: source_image_payloads[path] for path in required_paths}
    except KeyError as error:
        raise ValueError("source image inventory is missing a required member") from error

    prepared_by_path: dict[str, PreparedImage] = {}
    for path in required_paths:
        record = ocr_records_by_path[path]
        if record.get("image_member_path") != path:
            raise ValueError("OCR record mapping key differs from image_member_path")
        prepared_by_path[path] = validate_ocr_record(
            record,
            image_bytes=image_payloads[path],
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
        )

    trajectory_records: list[dict[str, Any]] = []
    for role in DERIVED_ROLE_ORDER:
        role_selection = selection_manifest["roles"][role]["trajectories"]
        for selection_record in role_selection:
            source_id = str(selection_record["source_id"])
            pilot = pilots_by_source[source_id]
            trajectory = pilot["trajectory"]
            if trajectory.get("source_id") != source_id:
                raise ValueError("pilot source ID drifted")
            instruction = str(trajectory["instruction"])
            instruction_sha = sha256_bytes(instruction.encode("utf-8"))
            if selection_record.get("instruction_sha256") != instruction_sha:
                raise ValueError("pilot instruction differs from selection witness")

            pilot_events = {
                int(event["step_id"]): event for event in trajectory["events"]
            }
            pilot_decisions = {
                int(decision["decision_step_id"]): decision
                for decision in trajectory["decisions"]
            }
            selected_states = states_by_source[source_id]
            max_history_step = max(
                max(int(value) for value in state["history_event_step_ids"])
                for state in selected_states
            )
            events = [
                _build_derived_event(
                    pilot_events[step_id],
                    image_payloads=image_payloads,
                    ocr_records_by_path=ocr_records_by_path,
                    prepared_by_path=prepared_by_path,
                )
                for step_id in range(1, max_history_step + 1)
            ]

            decisions = []
            for state in selected_states:
                decision_step = int(state["decision_step_id"])
                decision = pilot_decisions[decision_step]
                action_sha = sha256_bytes(
                    canonical_json_bytes(decision["validated_action"])
                )
                current = state["current_observation"]
                current_path = str(decision["current_observation_path"])
                current_sha = sha256_bytes(image_payloads[current_path])
                if state["validated_action_sha256"] != action_sha:
                    raise ValueError("validated action differs from selection witness")
                if current != {"member_path": current_path, "sha256": current_sha}:
                    raise ValueError("current observation differs from selection witness")
                expected_candidate_records = [
                    {
                        "event_step_id": step_id,
                        "post_state_member_path": events[step_id - 1][
                            "observation_after_path"
                        ],
                        "post_state_sha256": events[step_id - 1][
                            "observation_after_sha256"
                        ],
                    }
                    for step_id in state["candidate_event_step_ids"]
                ]
                if state["candidate_event_post_states"] != expected_candidate_records:
                    raise ValueError("candidate post-state witnesses drifted")
                current_equivalent_step = int(
                    state["current_equivalent_event_step_id"]
                )
                current_equivalent = {
                    "event_step_id": current_equivalent_step,
                    "post_state_member_path": events[current_equivalent_step - 1][
                        "observation_after_path"
                    ],
                    "post_state_sha256": events[current_equivalent_step - 1][
                        "observation_after_sha256"
                    ],
                }
                if state["current_equivalence_witness"] != current_equivalent:
                    raise ValueError("current-equivalence witness drifted")
                decisions.append(
                    {
                        "state_id": state["state_id"],
                        "decision_step_id": decision_step,
                        "history_event_step_ids": list(
                            state["history_event_step_ids"]
                        ),
                        "candidate_event_step_ids": list(
                            state["candidate_event_step_ids"]
                        ),
                        "current_equivalent_event_step_id": (
                            current_equivalent_step
                        ),
                        "current_observation_path": current_path,
                        "current_observation_sha256": current_sha,
                        "validated_action": dict(decision["validated_action"]),
                        "validated_action_sha256": action_sha,
                        "validation_source": decision["validation_source"],
                    }
                )

            trajectory_records.append(
                {
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
                    "selection": dict(selection_by_source[source_id]),
                    "events": events,
                    "decisions": decisions,
                }
            )
    return trajectory_records, image_payloads


def validate_frozen_inputs(
    *,
    v2_contract: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    selection_manifest_sha256: str,
    v1_config_sha256: str,
    source_file_manifest_sha256: str,
    exposure_manifest: Mapping[str, Any],
    backend_config: Mapping[str, Any],
) -> None:
    _require_sha256(selection_manifest_sha256, "selection_manifest_sha256")
    _require_sha256(v1_config_sha256, "v1_config_sha256")
    _require_sha256(
        source_file_manifest_sha256,
        "source_file_manifest_sha256",
    )
    validate_selection_manifest(selection_manifest, v2_contract=v2_contract)
    validate_state_content_witnesses(selection_manifest)
    validate_exposure_ledger(
        exposure_manifest,
        selection_manifest=selection_manifest,
    )
    if exposure_manifest.get("selection_manifest_sha256") != (
        selection_manifest_sha256
    ):
        raise ValueError("exposure ledger is not bound to the selection input bytes")
    selection_inputs = selection_manifest["inputs"]
    if selection_inputs["v1_selection_config"]["current_sha256"] != (
        v1_config_sha256
    ):
        raise ValueError("actual v1 config bytes differ from the frozen selection input")
    if selection_inputs["source_file_manifest"]["sha256"] != (
        source_file_manifest_sha256
    ):
        raise ValueError(
            "actual source-file manifest bytes differ from the frozen selection input"
        )
    validate_backend_config(backend_config)


def source_dataset_identity_from_v1_config(
    v1_config: Mapping[str, Any],
) -> dict[str, str]:
    source_pool = v1_config.get("source_pool")
    if not isinstance(source_pool, Mapping):
        raise ValueError("v1 config source_pool must be an object")
    fields = (
        "upstream_repo",
        "upstream_revision",
        "transport_repo",
        "transport_revision",
        "license",
    )
    identity = {field: source_pool.get(field) for field in fields}
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise ValueError("v1 source dataset identity values must be non-empty strings")
    for field in ("upstream_revision", "transport_revision"):
        if GIT_SHA_PATTERN.fullmatch(identity[field]) is None:
            raise ValueError(f"v1 source dataset {field} must be immutable")
    if identity["license"] != "cc-by-4.0":
        raise ValueError("v1 source dataset license drifted")
    return identity


def build_ocr_backend_provenance(
    *,
    backend_config_sha256: str,
    backend_manifest_sha256: str,
    backend_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _require_sha256(backend_config_sha256, "backend_config_sha256")
    _require_sha256(backend_manifest_sha256, "backend_manifest_sha256")
    if backend_manifest_sha256 != FROZEN_OCR_BACKEND_MANIFEST_SHA256:
        raise ValueError("OCR backend manifest differs from the frozen baseline milestone")
    model = backend_manifest.get("hf_model_artifact")
    real_screen = backend_manifest.get("real_screen_golden")
    if not isinstance(model, Mapping) or not isinstance(real_screen, Mapping):
        raise ValueError("OCR backend manifest is missing immutable HF provenance")
    provenance = {
        "backend_id": BACKEND_ID,
        "backend_config_sha256": backend_config_sha256,
        "hf_model_repo": model.get("repo"),
        "hf_model_revision": model.get("immutable_revision"),
        "hf_model_tag": model.get("tag"),
        "real_screen_golden_dataset_revision": real_screen.get(
            "immutable_revision"
        ),
    }
    if GIT_SHA_PATTERN.fullmatch(str(provenance["hf_model_revision"])) is None:
        raise ValueError("OCR HF model revision must be a full immutable revision")
    if GIT_SHA_PATTERN.fullmatch(
        str(provenance["real_screen_golden_dataset_revision"])
    ) is None:
        raise ValueError("OCR real-screen revision must be a full immutable revision")
    if not isinstance(provenance["hf_model_repo"], str) or not provenance[
        "hf_model_repo"
    ]:
        raise ValueError("OCR HF model repo must be a non-empty string")
    if not isinstance(provenance["hf_model_tag"], str) or not provenance[
        "hf_model_tag"
    ]:
        raise ValueError("OCR HF model tag must be a non-empty string")
    return provenance


def validate_ocr_runtime_identity(
    identity: Mapping[str, Any],
    *,
    backend_config: Mapping[str, Any],
) -> None:
    validate_backend_config(backend_config)
    if not isinstance(identity, Mapping) or set(identity) != OCR_RUNTIME_IDENTITY_KEYS:
        raise ValueError("OCR runtime identity fields drifted")
    if identity["runtime_packages"] != backend_config["runtime_packages"]:
        raise ValueError("OCR runtime package versions drifted")
    expected_package_files = {
        "config.yaml": backend_config["upstream_package_files"][
            "config_yaml_sha256"
        ],
        "default_models.yaml": backend_config["upstream_package_files"][
            "default_models_yaml_sha256"
        ],
        "utils/load_image.py": backend_config["upstream_package_files"][
            "load_image_py_sha256"
        ],
        "inference_engine/onnxruntime/main.py": backend_config[
            "upstream_package_files"
        ]["onnxruntime_main_py_sha256"],
        "ch_ppocr_rec/main.py": backend_config["upstream_package_files"][
            "recognizer_main_py_sha256"
        ],
    }
    if identity["rapidocr_package_file_sha256"] != expected_package_files:
        raise ValueError("OCR installed package source hashes drifted")
    expected_wheels = {
        name: record["sha256"] for name, record in backend_config["wheels"].items()
    }
    if identity["wheel_sha256"] != expected_wheels:
        raise ValueError("OCR wheel hashes drifted")
    expected_models = {
        role: record["sha256"] for role, record in backend_config["models"].items()
    }
    if identity["model_sha256"] != expected_models:
        raise ValueError("OCR model hashes drifted")
    recognizer = backend_config["models"]["recognizer"]
    expected_characters = {
        "metadata_key": recognizer["embedded_character_metadata_key"],
        "entry_count": recognizer["embedded_character_entry_count"],
        "utf8_sha256": recognizer["embedded_character_utf8_sha256"],
        "canonical_json_sha256": recognizer[
            "embedded_character_canonical_json_sha256"
        ],
    }
    if identity["recognizer_character_inventory"] != expected_characters:
        raise ValueError("OCR recognizer character inventory drifted")


def _iter_parquet_rows(parquet_file: Any) -> Iterator[tuple[int, Mapping[str, Any]]]:
    row_index = 0
    for batch in parquet_file.iter_batches():
        for row in batch.to_pylist():
            if not isinstance(row, Mapping):
                raise ValueError("GUIOdyssey Parquet rows must be objects")
            yield row_index, row
            row_index += 1


def load_selected_source_pilots(
    *,
    source_root: str | Path,
    source_file_manifest: Mapping[str, Any],
    v1_config: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    derived_repo: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    if not isinstance(derived_repo, str) or not derived_repo:
        raise ValueError("derived_repo must be a non-empty string")
    specs = source_file_specs(v1_config, source_file_manifest)
    root = Path(source_root)
    verify_local_source_files(root, specs)
    spec_by_path = {spec.transport_file: spec for spec in specs}
    role_by_source, _, selection_by_source = _states_by_source(selection_manifest)
    wanted_by_file: dict[str, dict[int, str]] = defaultdict(dict)
    for source_id, record in selection_by_source.items():
        transport_file = str(record["transport_file"])
        row_index = int(record["transport_row_index"])
        if transport_file not in spec_by_path:
            raise ValueError("selected transport file is absent from source manifest")
        if row_index in wanted_by_file[transport_file]:
            raise ValueError("selected Parquet row is duplicated")
        wanted_by_file[transport_file][row_index] = source_id

    import pyarrow.parquet as pq

    source_rows: dict[str, Mapping[str, Any]] = {}
    for transport_file, wanted in wanted_by_file.items():
        source_path = root.joinpath(*PurePosixPath(transport_file).parts)
        parquet = pq.ParquetFile(source_path)
        for row_index, row in _iter_parquet_rows(parquet):
            source_id = wanted.get(row_index)
            if source_id is not None:
                source_rows[source_id] = row
    if set(source_rows) != set(role_by_source):
        raise ValueError("failed to reload every selected GUIOdyssey source row")

    pilots: dict[str, dict[str, Any]] = {}
    image_payloads: dict[str, bytes] = {}
    source_pool = v1_config["source_pool"]
    grid_size = int(v1_config["policy"]["coordinate_grid_size"])
    for role in DERIVED_ROLE_ORDER:
        for selected in selection_manifest["roles"][role]["trajectories"]:
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
            if pilot["trajectory"]["source_id"] != source_id:
                raise ValueError("reloaded GUIOdyssey source ID drifted")
            if len(pilot["trajectory"]["decisions"]) != int(
                selected["decision_count"]
            ):
                raise ValueError("reloaded GUIOdyssey decision count drifted")
            pilots[source_id] = pilot
            for path, payload in images.items():
                if path in image_payloads:
                    raise ValueError("GUIOdyssey image member paths must be unique")
                image_payloads[path] = payload
    return pilots, image_payloads


def load_ocr_records_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    records = parse_canonical_jsonl(Path(path).read_bytes(), label="OCR records")
    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        member_path = _safe_member_path(record.get("image_member_path"))
        if member_path in by_path:
            raise ValueError("OCR JSONL image_member_path values must be unique")
        by_path[member_path] = record
    if list(by_path) != sorted(by_path):
        raise ValueError("OCR JSONL records must be sorted by image_member_path")
    return by_path


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    info.type = tarfile.REGTYPE
    return info


def build_image_tar_bytes(
    image_payloads: Mapping[str, bytes],
) -> tuple[bytes, list[dict[str, Any]]]:
    if not image_payloads:
        raise ValueError("image tar requires at least one payload")
    paths = sorted(image_payloads)
    if any(_safe_member_path(path) != path or not path.startswith("images/") for path in paths):
        raise ValueError("image tar paths must be canonical images/ members")
    buffer = io.BytesIO()
    members = []
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for path in paths:
            payload = image_payloads[path]
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("image tar payloads must be non-empty bytes")
            archive.addfile(_tar_info(path, len(payload)), io.BytesIO(payload))
            members.append(
                {
                    "path": path,
                    "size_bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )
    return buffer.getvalue(), members


def _file_record(path: str, payload: bytes, **extra: Any) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        **extra,
    }


def _validate_identity_record(record: Mapping[str, Any], *, field: str) -> None:
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"{field} identity must contain exactly path and sha256")
    _safe_member_path(record.get("path"))
    _require_sha256(record.get("sha256"), f"{field}.sha256")


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
    expected_input_keys = {
        "scientific_contract",
        "v1_config",
        "source_file_manifest",
        "selection_manifest",
        "exposure_manifest",
        "ocr_backend_config",
        "ocr_backend_manifest",
    }
    if set(inputs) != expected_input_keys:
        raise ValueError("derived artifact input provenance inventory drifted")
    for key, record in inputs.items():
        _validate_identity_record(record, field=key)
    expected_generator_keys = {
        "git_revision",
        "module_path",
        "module_sha256",
        "build_cli_path",
        "build_cli_sha256",
        "validator_path",
        "validator_sha256",
    }
    if set(generator) != expected_generator_keys:
        raise ValueError("derived artifact generator provenance fields drifted")
    if GIT_SHA_PATTERN.fullmatch(str(generator["git_revision"])) is None:
        raise ValueError("derived artifact generator git_revision must be a full SHA")
    for prefix in ("module", "build_cli", "validator"):
        _safe_member_path(generator[f"{prefix}_path"])
        _require_sha256(generator[f"{prefix}_sha256"], f"generator.{prefix}")
    expected_ocr_backend_keys = {
        "backend_id",
        "backend_config_sha256",
        "hf_model_repo",
        "hf_model_revision",
        "hf_model_tag",
        "real_screen_golden_dataset_revision",
    }
    if set(ocr_backend_provenance) != expected_ocr_backend_keys:
        raise ValueError("derived artifact OCR backend provenance fields drifted")
    if ocr_backend_provenance["backend_id"] != BACKEND_ID:
        raise ValueError("derived artifact OCR backend_id drifted")
    _require_sha256(
        ocr_backend_provenance["backend_config_sha256"],
        "ocr_backend.backend_config_sha256",
    )
    if ocr_backend_provenance["backend_config_sha256"] != inputs[
        "ocr_backend_config"
    ]["sha256"]:
        raise ValueError("OCR backend provenance is not bound to its input config")
    for field in ("hf_model_revision", "real_screen_golden_dataset_revision"):
        if GIT_SHA_PATTERN.fullmatch(str(ocr_backend_provenance[field])) is None:
            raise ValueError(f"ocr_backend.{field} must be a full immutable revision")
    for field in ("hf_model_repo", "hf_model_tag"):
        if not isinstance(ocr_backend_provenance[field], str) or not (
            ocr_backend_provenance[field]
        ):
            raise ValueError(f"ocr_backend.{field} must be a non-empty string")
    if type(formal_counts_enforced) is not bool:
        raise TypeError("formal_counts_enforced must be bool")

    role_source_ids = {
        role: [
            str(record["source_id"])
            for record in trajectories
            if record["role"] == role
        ]
        for role in DERIVED_ROLE_ORDER
    }
    counts = {
        "trajectory_count": len(trajectories),
        "event_count": sum(len(record["events"]) for record in trajectories),
        "state_count": sum(len(record["decisions"]) for record in trajectories),
        "image_member_count": len(image_members),
        "ocr_record_count": len(ocr_records),
    }
    if formal_counts_enforced and counts != EXPECTED_FORMAL_COUNTS:
        raise ValueError(
            f"formal derived artifact count mismatch: {counts} != {EXPECTED_FORMAL_COUNTS}"
        )
    trajectory_index = [
        {
            "source_id": record["source_id"],
            "role": record["role"],
            "event_count": len(record["events"]),
            "state_count": len(record["decisions"]),
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
    first_source = trajectories[0]["source"]
    source_dataset = {
        "upstream_repo": first_source["upstream_repo"],
        "upstream_revision": first_source["upstream_revision"],
        "transport_repo": first_source["transport_repo"],
        "transport_revision": first_source["transport_revision"],
        "license": first_source["license"],
    }
    for trajectory in trajectories:
        source = trajectory["source"]
        if any(source.get(field) != value for field, value in source_dataset.items()):
            raise ValueError("derived trajectories do not share one pinned source identity")
    if source_dataset["license"] != "cc-by-4.0":
        raise ValueError("derived GUIOdyssey source license drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_id": ARTIFACT_ID,
        "status": "POLICY_BLIND_DERIVED_DATASET_MATERIALIZED",
        "dataset_repo": dataset_repo,
        "license": "cc-by-4.0",
        "policy_output_generated": False,
        "restoration_output_generated": False,
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
        raise FileExistsError(f"derived artifact output already exists: {root}")
    if [record["source_id"] for record in trajectories] != [
        source_id
        for role in DERIVED_ROLE_ORDER
        for source_id in [
            record["source_id"] for record in trajectories if record["role"] == role
        ]
    ]:
        raise ValueError("trajectory records must follow the frozen role order")
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
        raise ValueError(
            "derived artifact file inventory drifted: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    files = []
    for relative in ARTIFACT_RELATIVE_PATHS:
        path = artifact_root.joinpath(*PurePosixPath(relative).parts)
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "files": files,
        "artifact_tree_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def _validate_tar(
    path: Path,
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("derived image tar inventory must be sorted and unique")
        for member in members:
            name = _safe_member_path(member.name)
            if not name.startswith("images/"):
                raise ValueError("derived image tar members must belong to images/")
            if (
                not member.isfile()
                or member.type != tarfile.REGTYPE
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mode != 0o644
                or member.pax_headers
            ):
                raise ValueError("derived image tar metadata drifted")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("derived image tar member is unreadable")
            payload = handle.read()
            if not payload or len(payload) != member.size:
                raise ValueError("derived image tar member size drifted")
            payloads[name] = payload
    expected_bytes, member_records = build_image_tar_bytes(payloads)
    if path.read_bytes() != expected_bytes:
        raise ValueError("derived image tar is not canonical USTAR bytes")
    return payloads, member_records


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "artifact_id",
        "status",
        "dataset_repo",
        "license",
        "policy_output_generated",
        "restoration_output_generated",
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
        raise ValueError("derived artifact manifest top-level fields drifted")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("derived artifact schema_version drifted")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("derived artifact protocol_id drifted")
    if manifest.get("artifact_id") != ARTIFACT_ID:
        raise ValueError("derived artifact_id drifted")
    if manifest.get("status") != "POLICY_BLIND_DERIVED_DATASET_MATERIALIZED":
        raise ValueError("derived artifact status drifted")
    if manifest.get("license") != "cc-by-4.0":
        raise ValueError("derived artifact license drifted")
    if not isinstance(manifest.get("dataset_repo"), str) or not manifest["dataset_repo"]:
        raise ValueError("derived artifact dataset_repo must be non-empty")
    if manifest.get("policy_output_generated") is not False:
        raise ValueError("derived artifact cannot contain policy output")
    if manifest.get("restoration_output_generated") is not False:
        raise ValueError("derived artifact cannot contain restoration output")
    if type(manifest.get("formal_counts_enforced")) is not bool:
        raise ValueError("derived artifact formal_counts_enforced must be bool")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("derived artifact inputs must be an object")
    expected_input_keys = {
        "scientific_contract",
        "v1_config",
        "source_file_manifest",
        "selection_manifest",
        "exposure_manifest",
        "ocr_backend_config",
        "ocr_backend_manifest",
    }
    if set(inputs) != expected_input_keys:
        raise ValueError("derived artifact input identity inventory drifted")
    for key, record in inputs.items():
        if not isinstance(record, Mapping):
            raise ValueError("derived artifact input identities must be objects")
        _validate_identity_record(record, field=str(key))
    generator = manifest.get("generator")
    if not isinstance(generator, Mapping):
        raise ValueError("derived artifact generator must be an object")
    expected_generator_keys = {
        "git_revision",
        "module_path",
        "module_sha256",
        "build_cli_path",
        "build_cli_sha256",
        "validator_path",
        "validator_sha256",
    }
    if set(generator) != expected_generator_keys:
        raise ValueError("derived artifact generator fields drifted")
    if GIT_SHA_PATTERN.fullmatch(str(generator["git_revision"])) is None:
        raise ValueError("derived artifact generator Git revision drifted")
    for prefix in ("module", "build_cli", "validator"):
        _safe_member_path(generator[f"{prefix}_path"])
        _require_sha256(generator[f"{prefix}_sha256"], f"generator.{prefix}")
    ocr_backend = manifest.get("ocr_backend")
    if not isinstance(ocr_backend, Mapping) or set(ocr_backend) != {
        "backend_id",
        "backend_config_sha256",
        "hf_model_repo",
        "hf_model_revision",
        "hf_model_tag",
        "real_screen_golden_dataset_revision",
    }:
        raise ValueError("derived artifact OCR backend provenance fields drifted")
    if ocr_backend.get("backend_id") != BACKEND_ID:
        raise ValueError("derived artifact OCR backend_id drifted")
    _require_sha256(
        ocr_backend.get("backend_config_sha256"),
        "ocr_backend.backend_config_sha256",
    )
    if ocr_backend["backend_config_sha256"] != inputs["ocr_backend_config"][
        "sha256"
    ]:
        raise ValueError("derived OCR backend provenance/input binding drifted")
    for field in ("hf_model_revision", "real_screen_golden_dataset_revision"):
        if GIT_SHA_PATTERN.fullmatch(str(ocr_backend.get(field))) is None:
            raise ValueError(f"ocr_backend.{field} must be a full immutable revision")
    for field in ("hf_model_repo", "hf_model_tag"):
        if not isinstance(ocr_backend.get(field), str) or not ocr_backend[field]:
            raise ValueError(f"ocr_backend.{field} must be a non-empty string")
    ocr_runtime = manifest.get("ocr_runtime")
    if not isinstance(ocr_runtime, Mapping) or set(ocr_runtime) != (
        OCR_RUNTIME_IDENTITY_KEYS
    ):
        raise ValueError("derived artifact OCR runtime identity fields drifted")
    source_dataset = manifest.get("source_dataset")
    if not isinstance(source_dataset, Mapping) or set(source_dataset) != {
        "upstream_repo",
        "upstream_revision",
        "transport_repo",
        "transport_revision",
        "license",
    }:
        raise ValueError("derived artifact source_dataset fields drifted")
    if any(not isinstance(value, str) or not value for value in source_dataset.values()):
        raise ValueError("derived artifact source_dataset values must be non-empty")
    if source_dataset["license"] != "cc-by-4.0":
        raise ValueError("derived artifact source dataset license drifted")
    roles = manifest.get("role_source_ids")
    if not isinstance(roles, Mapping) or set(roles) != set(DERIVED_ROLE_ORDER):
        raise ValueError("derived artifact role inventory drifted")
    if any(
        not isinstance(values, list)
        or any(not isinstance(value, str) or not value for value in values)
        for values in roles.values()
    ):
        raise ValueError("derived artifact role source IDs are invalid")
    if len({value for values in roles.values() for value in values}) != sum(
        len(values) for values in roles.values()
    ):
        raise ValueError("derived artifact role source IDs overlap")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(EXPECTED_FORMAL_COUNTS):
        raise ValueError("derived artifact count fields drifted")
    if any(type(value) is not int or value <= 0 for value in counts.values()):
        raise ValueError("derived artifact counts must be positive integers")
    if manifest["formal_counts_enforced"] and dict(counts) != EXPECTED_FORMAL_COUNTS:
        raise ValueError("formal derived artifact counts drifted")
    inventories = manifest.get("inventories")
    if not isinstance(inventories, Mapping) or set(inventories) != {
        "image_members_sha256",
        "ocr_records_sha256",
        "ocr_record_aggregate_sha256",
        "trajectory_index_sha256",
    }:
        raise ValueError("derived artifact inventory digest fields drifted")
    for key, value in inventories.items():
        _require_sha256(value, f"inventories.{key}")


def _validate_payload_file_records(
    root: Path,
    records: Any,
) -> None:
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("derived artifact must list exactly three payload shards")
    expected_paths = [
        IMAGE_TAR_RELATIVE_PATH,
        OCR_JSONL_RELATIVE_PATH,
        TRAJECTORY_JSONL_RELATIVE_PATH,
    ]
    if [record.get("path") for record in records if isinstance(record, Mapping)] != expected_paths:
        raise ValueError("derived artifact payload shard inventory or order drifted")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("derived artifact payload file records must be objects")
        path = root.joinpath(*PurePosixPath(record["path"]).parts)
        if record["path"] == IMAGE_TAR_RELATIVE_PATH:
            expected_keys = {"path", "size_bytes", "sha256", "member_count"}
        else:
            expected_keys = {"path", "size_bytes", "sha256", "record_count"}
        if set(record) != expected_keys:
            raise ValueError("derived artifact payload file fields drifted")
        if (
            type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
            or type(record[next(iter(expected_keys - {"path", "size_bytes", "sha256"}))])
            is not int
        ):
            raise ValueError("derived artifact payload file counts/sizes are invalid")
        _require_sha256(record["sha256"], "payload file sha256")
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record[
            "sha256"
        ]:
            raise ValueError("derived artifact payload file identity drifted")


def _validate_trajectory_records(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    image_payloads: Mapping[str, bytes],
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
    prepared_by_path: Mapping[str, PreparedImage],
    selection_manifest: Mapping[str, Any] | None,
) -> set[str]:
    expected_top_keys = {
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
    expected_order = [
        (role, source_id)
        for role in DERIVED_ROLE_ORDER
        for source_id in manifest["role_source_ids"][role]
    ]
    observed_order = [(record.get("role"), record.get("source_id")) for record in trajectories]
    if observed_order != expected_order:
        raise ValueError("derived trajectory role/source order drifted")
    selected_trajectories: dict[str, Mapping[str, Any]] = {}
    selected_states: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    if selection_manifest is not None:
        for role in DERIVED_ROLE_ORDER:
            for record in selection_manifest["roles"][role]["trajectories"]:
                selected_trajectories[str(record["source_id"])] = record
            for state in selection_manifest["roles"][role]["states"]:
                selected_states[str(state["source_id"])].append(state)

    required_paths: set[str] = set()
    for trajectory in trajectories:
        if set(trajectory) != expected_top_keys:
            raise ValueError("derived trajectory fields drifted")
        source_id = str(trajectory["source_id"])
        if trajectory["instruction_sha256"] != sha256_bytes(
            str(trajectory["instruction"]).encode("utf-8")
        ):
            raise ValueError("derived trajectory instruction SHA256 drifted")
        if selection_manifest is not None and trajectory["selection"] != selected_trajectories[
            source_id
        ]:
            raise ValueError("derived trajectory selection provenance drifted")
        source = trajectory["source"]
        if not isinstance(source, Mapping) or source.get("transport_file") != trajectory[
            "selection"
        ]["transport_file"] or source.get("transport_row_index") != trajectory[
            "selection"
        ]["transport_row_index"]:
            raise ValueError("derived trajectory source provenance drifted")
        if any(
            source.get(field) != value
            for field, value in manifest["source_dataset"].items()
        ):
            raise ValueError("derived trajectory pinned source identity drifted")
        events = trajectory["events"]
        if not isinstance(events, list) or [event.get("step_id") for event in events] != list(
            range(1, len(events) + 1)
        ):
            raise ValueError("derived trajectory event prefix drifted")
        for event in events:
            if not isinstance(event, Mapping) or set(event) != EVENT_KEYS:
                raise ValueError("derived event fields drifted")
            before = str(event["observation_before_path"])
            after = str(event["observation_after_path"])
            required_paths.update((before, after))
            rebuilt = _build_derived_event(
                event,
                image_payloads=image_payloads,
                ocr_records_by_path=ocr_records_by_path,
                prepared_by_path=prepared_by_path,
            )
            if rebuilt != event:
                raise ValueError("derived event is not reproducible from raw/OCR inputs")
        decisions = trajectory["decisions"]
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("derived trajectory decisions must be a non-empty array")
        if selection_manifest is not None and len(decisions) != len(
            selected_states[source_id]
        ):
            raise ValueError("derived trajectory state count drifted from selection")
        for index, decision in enumerate(decisions):
            if not isinstance(decision, Mapping) or set(decision) != DECISION_KEYS:
                raise ValueError("derived decision fields drifted")
            step = decision["decision_step_id"]
            history = decision["history_event_step_ids"]
            candidates = decision["candidate_event_step_ids"]
            current_equivalent = decision["current_equivalent_event_step_id"]
            if (
                type(step) is not int
                or history != list(range(1, step))
                or current_equivalent != history[-1]
                or candidates != history[:-1]
            ):
                raise ValueError("derived decision history/candidate identity drifted")
            current_path = str(decision["current_observation_path"])
            current_sha = decision["current_observation_sha256"]
            required_paths.add(current_path)
            if (
                current_path not in image_payloads
                or sha256_bytes(image_payloads[current_path]) != current_sha
                or events[current_equivalent - 1]["observation_after_path"]
                != current_path
                or events[current_equivalent - 1]["observation_after_sha256"]
                != current_sha
            ):
                raise ValueError("derived decision current-equivalence identity drifted")
            if decision["validated_action_sha256"] != sha256_bytes(
                canonical_json_bytes(decision["validated_action"])
            ):
                raise ValueError("derived decision validated action SHA256 drifted")
            if selection_manifest is not None:
                state = selected_states[source_id][index]
                expected_selection_fields = {
                    "state_id": state["state_id"],
                    "decision_step_id": state["decision_step_id"],
                    "history_event_step_ids": state["history_event_step_ids"],
                    "candidate_event_step_ids": state["candidate_event_step_ids"],
                    "current_equivalent_event_step_id": state[
                        "current_equivalent_event_step_id"
                    ],
                    "current_observation_path": state["current_observation"][
                        "member_path"
                    ],
                    "current_observation_sha256": state["current_observation"][
                        "sha256"
                    ],
                    "validated_action_sha256": state["validated_action_sha256"],
                }
                if any(decision[key] != value for key, value in expected_selection_fields.items()):
                    raise ValueError("derived decision differs from selection witness")
    return required_paths


def validate_artifact(
    *,
    output_dir: str | Path,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    selection_manifest: Mapping[str, Any] | None = None,
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
) -> dict[str, Any]:
    validate_backend_config(backend_config)
    if type(require_formal) is not bool or type(require_ocr_replay) is not bool:
        raise TypeError("formal and OCR replay requirements must be bool")
    root = Path(output_dir)
    tree = artifact_tree_identity(root)
    if (root / GITATTRIBUTES_RELATIVE_PATH).read_bytes() != artifact_gitattributes_bytes():
        raise ValueError("derived artifact .gitattributes drifted")
    manifest_payload, manifest = load_json_object(root / MANIFEST_RELATIVE_PATH)
    if manifest_payload != pretty_json_bytes(manifest):
        raise ValueError("derived artifact manifest is not canonical pretty JSON")
    _validate_manifest_structure(manifest)
    if require_formal:
        if manifest["formal_counts_enforced"] is not True:
            raise ValueError(
                "formal artifact validation requires formal_counts_enforced=true"
            )
        required_external_bindings = {
            "selection_manifest": selection_manifest,
            "expected_input_sha256": expected_input_sha256,
            "expected_ocr_backend_provenance": expected_ocr_backend_provenance,
            "expected_ocr_runtime_identity": expected_ocr_runtime_identity,
            "expected_generator_git_revision": expected_generator_git_revision,
            "expected_dataset_repo": expected_dataset_repo,
            "expected_source_dataset": expected_source_dataset,
            "repository_root": repository_root,
        }
        missing_bindings = sorted(
            key for key, value in required_external_bindings.items() if value is None
        )
        if missing_bindings:
            raise ValueError(
                "formal artifact validation requires external bindings: "
                f"{missing_bindings}"
            )
        if not require_ocr_replay or ocr_engine is None:
            raise ValueError("formal artifact validation requires exact OCR replay")
    if (root / README_RELATIVE_PATH).read_bytes() != artifact_readme_bytes(
        dataset_repo=manifest["dataset_repo"]
    ):
        raise ValueError("derived artifact README drifted")
    if manifest["ocr_backend"]["backend_config_sha256"] != backend_config_sha256:
        raise ValueError("derived artifact OCR backend config SHA256 drifted")
    if expected_ocr_backend_provenance is not None and manifest[
        "ocr_backend"
    ] != expected_ocr_backend_provenance:
        raise ValueError("derived artifact OCR backend provenance drifted")
    validate_ocr_runtime_identity(
        manifest["ocr_runtime"],
        backend_config=backend_config,
    )
    if expected_ocr_runtime_identity is not None and manifest[
        "ocr_runtime"
    ] != expected_ocr_runtime_identity:
        raise ValueError("derived artifact live OCR runtime identity drifted")
    if expected_generator_git_revision is not None:
        if GIT_SHA_PATTERN.fullmatch(expected_generator_git_revision) is None:
            raise ValueError("expected generator Git revision must be a full SHA")
        if manifest["generator"]["git_revision"] != (
            expected_generator_git_revision
        ):
            raise ValueError("derived artifact generator Git revision drifted")
    if expected_dataset_repo is not None and manifest["dataset_repo"] != (
        expected_dataset_repo
    ):
        raise ValueError("derived artifact dataset repo drifted from the v2 contract")
    if expected_source_dataset is not None and manifest["source_dataset"] != dict(
        expected_source_dataset
    ):
        raise ValueError("derived artifact source dataset drifted from the v1 config")
    _validate_payload_file_records(root, manifest["payload_files"])
    if expected_input_sha256 is not None:
        if set(expected_input_sha256) != set(manifest["inputs"]):
            raise ValueError("expected derived input identity inventory drifted")
        for key, digest in expected_input_sha256.items():
            _require_sha256(digest, f"expected_input_sha256.{key}")
            if manifest["inputs"][key]["sha256"] != digest:
                raise ValueError(f"derived artifact input SHA256 drifted: {key}")
    if repository_root is not None:
        checkout = Path(repository_root)
        for prefix in ("module", "build_cli", "validator"):
            path = checkout.joinpath(
                *PurePosixPath(manifest["generator"][f"{prefix}_path"]).parts
            )
            if not path.is_file() or sha256_file(path) != manifest["generator"][
                f"{prefix}_sha256"
            ]:
                raise ValueError(f"derived artifact generator source drifted: {prefix}")

    image_payloads, image_members = _validate_tar(root / IMAGE_TAR_RELATIVE_PATH)
    if require_formal:
        if selection_manifest is None:
            raise RuntimeError("formal selection binding disappeared")
        validate_formal_image_inventory(
            tuple(image_payloads),
            selection_manifest=selection_manifest,
        )
    ocr_records = parse_canonical_jsonl(
        (root / OCR_JSONL_RELATIVE_PATH).read_bytes(),
        label="derived OCR records",
    )
    ocr_records_by_path: dict[str, Mapping[str, Any]] = {}
    prepared_by_path: dict[str, PreparedImage] = {}
    for record in ocr_records:
        path = _safe_member_path(record.get("image_member_path"))
        if path in ocr_records_by_path:
            raise ValueError("derived OCR records contain duplicate image paths")
        if path not in image_payloads:
            raise ValueError("derived OCR record has no archived image")
        prepared_by_path[path] = validate_ocr_record(
            record,
            image_bytes=image_payloads[path],
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
        )
        ocr_records_by_path[path] = record
    if list(ocr_records_by_path) != sorted(ocr_records_by_path):
        raise ValueError("derived OCR records are not sorted by image_member_path")
    if set(ocr_records_by_path) != set(image_payloads):
        raise ValueError("derived OCR and image inventories differ")
    replay_record_count = 0
    replay_aggregate_sha256: str | None = None
    if require_ocr_replay:
        if ocr_engine is None:
            raise ValueError("exact OCR replay requires an OCR engine")
        replayed_records, replay_aggregate_sha256 = generate_ocr_records(
            engine=ocr_engine,
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
            image_payloads=image_payloads,
            record_runner=ocr_record_runner,
        )
        replay_record_count = len(replayed_records)
        for path in sorted(ocr_records_by_path):
            if replayed_records[path] != ocr_records_by_path[path]:
                raise ValueError(
                    f"replayed OCR record differs from archived record: {path}"
                )

    trajectories = parse_canonical_jsonl(
        (root / TRAJECTORY_JSONL_RELATIVE_PATH).read_bytes(),
        label="derived trajectories",
    )
    required_paths = _validate_trajectory_records(
        trajectories,
        manifest=manifest,
        image_payloads=image_payloads,
        ocr_records_by_path=ocr_records_by_path,
        prepared_by_path=prepared_by_path,
        selection_manifest=selection_manifest,
    )
    if required_paths != set(image_payloads):
        raise ValueError("derived image archive contains missing or unused members")

    counts = {
        "trajectory_count": len(trajectories),
        "event_count": sum(len(record["events"]) for record in trajectories),
        "state_count": sum(len(record["decisions"]) for record in trajectories),
        "image_member_count": len(image_members),
        "ocr_record_count": len(ocr_records),
    }
    if counts != manifest["counts"]:
        raise ValueError("derived artifact counts are not self-consistent")
    if require_formal:
        if counts != EXPECTED_FORMAL_COUNTS:
            raise ValueError(
                f"formal artifact exact counts drifted: {counts} != "
                f"{EXPECTED_FORMAL_COUNTS}"
            )
    trajectory_index = [
        {
            "source_id": record["source_id"],
            "role": record["role"],
            "event_count": len(record["events"]),
            "state_count": len(record["decisions"]),
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
    expected_inventories = {
        "image_members_sha256": sha256_bytes(canonical_json_bytes(image_members)),
        "ocr_records_sha256": sha256_bytes(canonical_json_bytes(ocr_index)),
        "ocr_record_aggregate_sha256": ocr_record_aggregate_sha256(
            ocr_records_by_path
        ),
        "trajectory_index_sha256": sha256_bytes(
            canonical_json_bytes(trajectory_index)
        ),
    }
    if manifest["inventories"] != expected_inventories:
        raise ValueError("derived artifact inventory digests drifted")
    payload_counts = {
        record["path"]: record.get("member_count", record.get("record_count"))
        for record in manifest["payload_files"]
    }
    if payload_counts != {
        IMAGE_TAR_RELATIVE_PATH: len(image_members),
        OCR_JSONL_RELATIVE_PATH: len(ocr_records),
        TRAJECTORY_JSONL_RELATIVE_PATH: len(trajectories),
    }:
        raise ValueError("derived artifact payload shard counts drifted")
    return {
        "outcome": "PASSED_GUIODYSSEY_RESTORATION_V2_ARTIFACT_VALIDATION",
        "artifact_tree_sha256": tree["artifact_tree_sha256"],
        "counts": counts,
        "formal_counts_enforced": manifest["formal_counts_enforced"],
        "ocr_record_aggregate_sha256": expected_inventories[
            "ocr_record_aggregate_sha256"
        ],
        "ocr_replay_performed": require_ocr_replay,
        "ocr_replay_record_count": replay_record_count,
        "ocr_replay_aggregate_sha256": replay_aggregate_sha256,
        "policy_loaded": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
    }
