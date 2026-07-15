"""Deterministic multi-trajectory GUIOdyssey dataset construction."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from causalcache.data.guiodyssey import build_pilot_manifest, canonicalize_tool_call


class ExclusionReason(str, Enum):
    EXCLUDED_SOURCE_ID = "excluded_source_id"
    PLATFORM_MISMATCH = "platform_mismatch"
    UNSAFE_SOURCE_ID = "unsafe_source_id"
    INVALID_ROW_SCHEMA = "invalid_row_schema"
    OBSERVATION_ACTION_COUNT_MISMATCH = "observation_action_count_mismatch"
    EXECUTABLE_ACTION_COUNT_MISMATCH = "executable_action_count_mismatch"
    TERMINAL_SIGNAL_COUNT_MISMATCH = "terminal_signal_count_mismatch"
    TERMINAL_STATUS_MISMATCH = "terminal_status_mismatch"
    TERMINAL_SIGNAL_NOT_LAST = "terminal_signal_not_last"
    DECISION_COUNT_BELOW_MINIMUM = "decision_count_below_minimum"
    DECISION_COUNT_ABOVE_MAXIMUM = "decision_count_above_maximum"
    PARSER_INCOMPATIBLE_ACTION_TYPE = "parser_incompatible_action_type"
    INVALID_EXECUTABLE_ACTION = "invalid_executable_action"
    EMBEDDED_IMAGE_MISSING = "embedded_image_missing"
    EMBEDDED_IMAGE_UNSUPPORTED = "embedded_image_unsupported"


@dataclass(frozen=True)
class SourceFileSpec:
    transport_file: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Candidate:
    source_id: str
    transport_file: str
    transport_row_index: int
    selection_sha256: str
    decision_count: int
    normalized_app_labels: tuple[str, ...]
    action_type_counts: tuple[tuple[str, int], ...]

    def action_counts(self) -> dict[str, int]:
        return dict(self.action_type_counts)

    def pool_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "transport_file": self.transport_file,
            "transport_row_index": self.transport_row_index,
            "selection_sha256": self.selection_sha256,
            "decision_count": self.decision_count,
            "normalized_app_labels": list(self.normalized_app_labels),
            "action_type_counts": self.action_counts(),
        }


@dataclass(frozen=True)
class InspectionResult:
    source_id: str | None
    candidate: Candidate | None
    exclusion_reason: ExclusionReason | None

    def __post_init__(self) -> None:
        included = self.candidate is not None
        if included == (self.exclusion_reason is not None):
            raise ValueError("inspection must contain exactly one candidate or exclusion reason")


@dataclass(frozen=True)
class SelectedSplits:
    eligible_candidates: tuple[Candidate, ...]
    reference_gate: tuple[Candidate, ...]
    oracle_pilot: tuple[Candidate, ...]
    eligible_pool_sha256: str


class InsufficientDataError(RuntimeError):
    def __init__(self, split: str, available: Sequence[Candidate]) -> None:
        self.split = split
        self.available = tuple(available)
        super().__init__(f"INSUFFICIENT_DATA: {split} quotas are not satisfied")


class _RowSchemaError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_app_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def trajectory_selection_sha256(source_id: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{source_id}".encode("utf-8")).hexdigest()


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def validate_protocol_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_id") != "independent_reference_gate_v1":
        raise ValueError("unexpected independent gate protocol_id")
    source_pool = config["source_pool"]
    eligibility = config["eligibility"]
    selection = config["selection"]
    if selection["trajectory_hash_input"] != "UTF8(salt + NUL + source_id)":
        raise ValueError("unsupported trajectory hash input contract")
    if selection["trajectory_order"] != [
        "sha256_hex",
        "source_id",
        "transport_file",
        "transport_row_index",
    ]:
        raise ValueError("unsupported trajectory ordering contract")
    if not source_pool["transport_files"]:
        raise ValueError("source_pool.transport_files must not be empty")
    if len(set(source_pool["transport_files"])) != len(source_pool["transport_files"]):
        raise ValueError("source_pool.transport_files must be unique")
    if any(not _safe_relative_path(str(path)) for path in source_pool["transport_files"]):
        raise ValueError("transport files must be safe relative paths")
    re.compile(str(eligibility["source_id_regex"]))
    minimum = int(eligibility["minimum_decisions_per_trajectory"])
    maximum = int(eligibility["maximum_decisions_per_trajectory"])
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid decision-count eligibility range")
    compatible = {str(value) for value in eligibility["parser_compatible_action_types"]}
    required = {str(value) for value in selection["required_action_types"]}
    if not required or not required.issubset(compatible):
        raise ValueError("required action types must be parser compatible")
    for split_name in ("reference_gate", "oracle_pilot"):
        quota = selection[split_name]
        if any(
            int(quota[key]) <= 0
            for key in (
                "minimum_trajectories",
                "minimum_decisions",
                "minimum_distinct_app_labels",
                "minimum_candidate_count_per_required_action_type",
            )
        ):
            raise ValueError(f"{split_name} quotas must be positive")
    shard = str(config["artifact"]["shard"])
    if not _safe_relative_path(shard) or not shard.startswith("data/") or not shard.endswith(".tar"):
        raise ValueError("artifact shard must be a safe data/*.tar path")


def source_file_specs(
    config: Mapping[str, Any],
    source_file_manifest: Mapping[str, Any],
) -> tuple[SourceFileSpec, ...]:
    validate_protocol_config(config)
    source_pool = config["source_pool"]
    if source_file_manifest.get("schema_version") != "0.1.0":
        raise ValueError("unexpected source file manifest schema_version")
    if source_file_manifest.get("protocol_id") != config["protocol_id"]:
        raise ValueError("source file manifest protocol_id mismatch")
    if source_file_manifest.get("repo") != source_pool["transport_repo"]:
        raise ValueError("source file manifest repo mismatch")
    if source_file_manifest.get("revision") != source_pool["transport_revision"]:
        raise ValueError("source file manifest revision mismatch")
    records = source_file_manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("source file manifest files must be a list")
    expected_paths = [str(value) for value in source_pool["transport_files"]]
    actual_paths = [str(record.get("path")) for record in records if isinstance(record, Mapping)]
    if len(actual_paths) != len(records) or actual_paths != expected_paths:
        raise ValueError("source file manifest paths must exactly match the frozen order")
    specs = []
    for record in records:
        size = int(record["size"])
        digest = str(record["sha256"])
        if size <= 0 or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("source file manifest contains an invalid size or SHA256")
        specs.append(SourceFileSpec(str(record["path"]), size, digest))
    if sum(spec.size_bytes for spec in specs) != int(source_file_manifest["total_bytes"]):
        raise ValueError("source file manifest total_bytes mismatch")
    return tuple(specs)


def verify_local_source_files(
    source_root: Path,
    specs: Sequence[SourceFileSpec],
) -> None:
    for spec in specs:
        path = source_root.joinpath(*PurePosixPath(spec.transport_file).parts)
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned source file: {spec.transport_file}")
        if path.stat().st_size != spec.size_bytes:
            raise ValueError(f"source file size mismatch: {spec.transport_file}")
        if sha256_file(path) != spec.sha256:
            raise ValueError(f"source file SHA256 mismatch: {spec.transport_file}")


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        raise _RowSchemaError("serialized row field must be a string")
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise _RowSchemaError("serialized row field is not valid JSON") from error


def _image_is_supported(value: bytes) -> bool:
    return (
        value.startswith(b"\x89PNG\r\n\x1a\n")
        or value.startswith(b"\xff\xd8\xff")
        or (value.startswith(b"RIFF") and len(value) >= 12 and value[8:12] == b"WEBP")
    )


def _excluded(
    reason: ExclusionReason,
    *,
    source_id: str | None,
) -> InspectionResult:
    return InspectionResult(source_id=source_id, candidate=None, exclusion_reason=reason)


def inspect_candidate(
    row: Mapping[str, Any],
    *,
    transport_file: str,
    transport_row_index: int,
    config: Mapping[str, Any],
) -> InspectionResult:
    validate_protocol_config(config)
    source_id: str | None = None
    try:
        messages = _json_value(row["messages"])
        metadata = _json_value(row["metadata"])
    except (KeyError, _RowSchemaError):
        return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=None)
    if not isinstance(messages, list) or not isinstance(metadata, Mapping):
        return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=None)
    try:
        others = metadata["others"]
        if not isinstance(others, Mapping):
            raise _RowSchemaError("metadata.others must be an object")
        source_id_value = others["source_id"]
        if not isinstance(source_id_value, str):
            raise _RowSchemaError("source_id must be a string")
        source_id = source_id_value
    except (KeyError, _RowSchemaError):
        return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=None)

    source_pool = config["source_pool"]
    eligibility = config["eligibility"]
    if source_id in {str(value) for value in source_pool["excluded_source_ids"]}:
        return _excluded(ExclusionReason.EXCLUDED_SOURCE_ID, source_id=source_id)
    if re.fullmatch(str(eligibility["source_id_regex"]), source_id) is None:
        return _excluded(ExclusionReason.UNSAFE_SOURCE_ID, source_id=source_id)
    if metadata.get("platform") != eligibility["platform"]:
        return _excluded(ExclusionReason.PLATFORM_MISMATCH, source_id=source_id)

    try:
        images = row["images"]
    except KeyError:
        return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=source_id)
    if not isinstance(images, list):
        return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=source_id)
    user_turns = [turn for turn in messages if isinstance(turn, Mapping) and turn.get("role") == "user"]
    assistant_turns = [
        turn for turn in messages if isinstance(turn, Mapping) and turn.get("role") == "assistant"
    ]
    if len(user_turns) != len(assistant_turns) or len(images) != len(assistant_turns):
        return _excluded(
            ExclusionReason.OBSERVATION_ACTION_COUNT_MISMATCH,
            source_id=source_id,
        )
    if not assistant_turns:
        return _excluded(
            ExclusionReason.OBSERVATION_ACTION_COUNT_MISMATCH,
            source_id=source_id,
        )
    first_content = user_turns[0].get("content")
    if not isinstance(first_content, list) or not any(
        isinstance(item, Mapping)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
        and bool(item["text"])
        for item in first_content
    ):
        return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=source_id)

    actions = []
    terminal_positions = []
    terminal_statuses = []
    for position, turn in enumerate(assistant_turns):
        tool_calls = turn.get("tool_calls")
        if not isinstance(tool_calls, list):
            return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=source_id)
        executable_calls = []
        terminal_calls = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping) or not isinstance(tool_call.get("function"), Mapping):
                return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=source_id)
            function = tool_call["function"]
            if function.get("name") == "terminate":
                terminal_calls.append(function)
            else:
                executable_calls.append(tool_call)
        if len(executable_calls) != int(eligibility["executable_actions_per_observation"]):
            return _excluded(
                ExclusionReason.EXECUTABLE_ACTION_COUNT_MISMATCH,
                source_id=source_id,
            )
        if len(terminal_calls) > 1:
            return _excluded(
                ExclusionReason.TERMINAL_SIGNAL_COUNT_MISMATCH,
                source_id=source_id,
            )
        if terminal_calls:
            arguments = terminal_calls[0].get("arguments")
            if not isinstance(arguments, Mapping):
                return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=source_id)
            terminal_positions.append(position)
            terminal_statuses.append(str(arguments.get("status", "unknown")))
        try:
            action, _ = canonicalize_tool_call(
                executable_calls[0],
                grid_size=int(config["policy"]["coordinate_grid_size"]),
            )
        except (KeyError, TypeError, ValueError):
            return _excluded(
                ExclusionReason.INVALID_EXECUTABLE_ACTION,
                source_id=source_id,
            )
        actions.append(action)

    if len(terminal_positions) != 1:
        return _excluded(
            ExclusionReason.TERMINAL_SIGNAL_COUNT_MISMATCH,
            source_id=source_id,
        )
    if terminal_statuses[0] != eligibility["terminal_status"]:
        return _excluded(ExclusionReason.TERMINAL_STATUS_MISMATCH, source_id=source_id)
    if bool(eligibility["terminal_signal_must_be_last"]) and terminal_positions[0] != len(actions) - 1:
        return _excluded(ExclusionReason.TERMINAL_SIGNAL_NOT_LAST, source_id=source_id)

    decision_count = len(actions) - 1
    if decision_count < int(eligibility["minimum_decisions_per_trajectory"]):
        return _excluded(ExclusionReason.DECISION_COUNT_BELOW_MINIMUM, source_id=source_id)
    if decision_count > int(eligibility["maximum_decisions_per_trajectory"]):
        return _excluded(ExclusionReason.DECISION_COUNT_ABOVE_MAXIMUM, source_id=source_id)
    compatible = {str(value) for value in eligibility["parser_compatible_action_types"]}
    action_counts = Counter(action.action_type.value for action in actions[1:])
    if not set(action_counts).issubset(compatible):
        return _excluded(
            ExclusionReason.PARSER_INCOMPATIBLE_ACTION_TYPE,
            source_id=source_id,
        )

    for image in images:
        if not isinstance(image, Mapping) or not isinstance(image.get("bytes"), bytes):
            return _excluded(ExclusionReason.EMBEDDED_IMAGE_MISSING, source_id=source_id)
        if not _image_is_supported(image["bytes"]):
            return _excluded(ExclusionReason.EMBEDDED_IMAGE_UNSUPPORTED, source_id=source_id)

    apps = others.get("apps")
    if not isinstance(apps, list) or not apps or any(not isinstance(value, str) for value in apps):
        return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=source_id)
    normalized_apps = tuple(
        sorted(
            {
                normalized
                for value in apps
                if (normalized := normalize_app_label(value))
            }
        )
    )
    if not normalized_apps:
        return _excluded(ExclusionReason.INVALID_ROW_SCHEMA, source_id=source_id)
    candidate = Candidate(
        source_id=source_id,
        transport_file=transport_file,
        transport_row_index=transport_row_index,
        selection_sha256=trajectory_selection_sha256(
            source_id,
            salt=str(config["selection"]["trajectory_salt"]),
        ),
        decision_count=decision_count,
        normalized_app_labels=normalized_apps,
        action_type_counts=tuple(sorted(action_counts.items())),
    )
    return InspectionResult(source_id=source_id, candidate=candidate, exclusion_reason=None)


def candidate_sort_key(candidate: Candidate) -> tuple[str, str, str, int]:
    return (
        candidate.selection_sha256,
        candidate.source_id,
        candidate.transport_file,
        candidate.transport_row_index,
    )


def _split_summary(candidates: Sequence[Candidate]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    app_labels = set()
    for candidate in candidates:
        action_counts.update(candidate.action_counts())
        app_labels.update(candidate.normalized_app_labels)
    return {
        "source_ids": [candidate.source_id for candidate in candidates],
        "trajectory_count": len(candidates),
        "decision_count": sum(candidate.decision_count for candidate in candidates),
        "distinct_app_labels": sorted(app_labels),
        "action_type_counts": dict(sorted(action_counts.items())),
    }


def _quota_satisfied(
    candidates: Sequence[Candidate],
    *,
    quota: Mapping[str, Any],
    required_action_types: Sequence[str],
) -> bool:
    summary = _split_summary(candidates)
    minimum_per_action = int(quota["minimum_candidate_count_per_required_action_type"])
    return (
        summary["trajectory_count"] >= int(quota["minimum_trajectories"])
        and summary["decision_count"] >= int(quota["minimum_decisions"])
        and len(summary["distinct_app_labels"]) >= int(quota["minimum_distinct_app_labels"])
        and all(
            summary["action_type_counts"].get(action_type, 0) >= minimum_per_action
            for action_type in required_action_types
        )
    )


def _shortest_prefix(
    candidates: Sequence[Candidate],
    *,
    split_name: str,
    quota: Mapping[str, Any],
    required_action_types: Sequence[str],
) -> tuple[tuple[Candidate, ...], tuple[Candidate, ...]]:
    for length in range(1, len(candidates) + 1):
        prefix = candidates[:length]
        if _quota_satisfied(
            prefix,
            quota=quota,
            required_action_types=required_action_types,
        ):
            return tuple(prefix), tuple(candidates[length:])
    raise InsufficientDataError(split_name, candidates)


def select_splits(
    candidates: Iterable[Candidate],
    *,
    config: Mapping[str, Any],
) -> SelectedSplits:
    validate_protocol_config(config)
    ordered = tuple(sorted(candidates, key=candidate_sort_key))
    source_ids = [candidate.source_id for candidate in ordered]
    if bool(config["eligibility"]["require_unique_source_id"]) and len(source_ids) != len(
        set(source_ids)
    ):
        raise ValueError("eligible source_id values must be unique")
    pool_digest = hashlib.sha256(
        canonical_json_bytes([candidate.pool_record() for candidate in ordered])
    ).hexdigest()
    required = [str(value) for value in config["selection"]["required_action_types"]]
    reference, remainder = _shortest_prefix(
        ordered,
        split_name="reference_gate",
        quota=config["selection"]["reference_gate"],
        required_action_types=required,
    )
    oracle, _ = _shortest_prefix(
        remainder,
        split_name="oracle_pilot",
        quota=config["selection"]["oracle_pilot"],
        required_action_types=required,
    )
    return SelectedSplits(
        eligible_candidates=ordered,
        reference_gate=reference,
        oracle_pilot=oracle,
        eligible_pool_sha256=pool_digest,
    )


def build_independent_manifest(
    *,
    selected: SelectedSplits,
    selected_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    config: Mapping[str, Any],
    source_specs: Sequence[SourceFileSpec],
    source_row_counts: Mapping[str, int],
    exclusion_counts: Mapping[str, int],
    total_source_rows: int,
    protocol_config_sha256: str,
    source_file_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    spec_by_path = {spec.transport_file: spec for spec in source_specs}
    if len(spec_by_path) != len(source_specs):
        raise ValueError("source file specs must be unique")
    if len(selected.eligible_candidates) + sum(exclusion_counts.values()) != total_source_rows:
        raise ValueError("eligible and excluded rows must reconstruct the source pool")
    split_candidates = {
        "reference_gate": selected.reference_gate,
        "oracle_pilot": selected.oracle_pilot,
    }
    trajectories = []
    image_payloads: dict[str, bytes] = {}
    for split_name, candidates in split_candidates.items():
        for candidate in candidates:
            key = (candidate.transport_file, candidate.transport_row_index)
            if key not in selected_rows:
                raise ValueError(f"missing selected source row: {key}")
            spec = spec_by_path[candidate.transport_file]
            pilot, images = build_pilot_manifest(
                selected_rows[key],
                row_index=candidate.transport_row_index,
                upstream_repo=str(config["source_pool"]["upstream_repo"]),
                upstream_revision=str(config["source_pool"]["upstream_revision"]),
                transport_repo=str(config["source_pool"]["transport_repo"]),
                transport_revision=str(config["source_pool"]["transport_revision"]),
                transport_file=candidate.transport_file,
                transport_file_sha256=spec.sha256,
                hf_destination=str(config["artifact"]["repo"]),
                grid_size=int(config["policy"]["coordinate_grid_size"]),
            )
            trajectory = dict(pilot["trajectory"])
            actual_action_counts = Counter(
                decision["validated_action"]["action_type"]
                for decision in trajectory["decisions"]
            )
            actual_normalized_apps = tuple(
                sorted(
                    {
                        normalized
                        for value in trajectory["apps"]
                        if (normalized := normalize_app_label(value))
                    }
                )
            )
            if (
                trajectory["source_id"] != candidate.source_id
                or len(trajectory["decisions"]) != candidate.decision_count
                or dict(sorted(actual_action_counts.items())) != candidate.action_counts()
                or actual_normalized_apps != candidate.normalized_app_labels
            ):
                raise RuntimeError("selected row changed between inspection and packaging")
            trajectory.update(
                {
                    "split": split_name,
                    "transport_file": candidate.transport_file,
                    "transport_row_index": candidate.transport_row_index,
                    "selection_sha256": candidate.selection_sha256,
                    "normalized_app_labels": list(candidate.normalized_app_labels),
                }
            )
            trajectories.append(trajectory)
            for name, payload in images.items():
                if name in image_payloads:
                    raise ValueError(f"duplicate image member path: {name}")
                image_payloads[name] = payload

    source_pool = config["source_pool"]
    manifest = {
        "schema_version": "0.4.0",
        "protocol_id": config["protocol_id"],
        "dataset_repo": config["artifact"]["repo"],
        "source": {
            "upstream_repo": source_pool["upstream_repo"],
            "upstream_revision": source_pool["upstream_revision"],
            "transport_repo": source_pool["transport_repo"],
            "transport_revision": source_pool["transport_revision"],
            "transport_files": [
                {
                    "transport_file": spec.transport_file,
                    "sha256": spec.sha256,
                    "size_bytes": spec.size_bytes,
                    "row_count": int(source_row_counts[spec.transport_file]),
                }
                for spec in source_specs
            ],
            "license": source_pool["license"],
            "protocol_config_sha256": protocol_config_sha256,
            "source_file_manifest_sha256": source_file_manifest_sha256,
        },
        "selection": {
            "algorithm": config["selection"]["algorithm"],
            "trajectory_salt": config["selection"]["trajectory_salt"],
            "trajectory_hash_input": config["selection"]["trajectory_hash_input"],
            "trajectory_order": config["selection"]["trajectory_order"],
            "eligibility": config["eligibility"],
            "excluded_source_ids": source_pool["excluded_source_ids"],
            "total_source_rows": total_source_rows,
            "eligible_pool_count": len(selected.eligible_candidates),
            "eligible_pool_sha256": selected.eligible_pool_sha256,
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
        },
        "splits": {
            name: _split_summary(candidates)
            for name, candidates in split_candidates.items()
        },
        "trajectories": trajectories,
    }
    return manifest, image_payloads


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    return info


def write_independent_dataset(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    image_payloads: Mapping[str, bytes],
    shard_relative_path: str,
) -> Path:
    if not _safe_relative_path(shard_relative_path) or not shard_relative_path.startswith("data/"):
        raise ValueError("shard path must be a safe relative data path")
    for name in image_payloads:
        if not _safe_relative_path(name) or name == "manifest.json":
            raise ValueError(f"unsafe tar member path: {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    tar_path = output_dir.joinpath(*PurePosixPath(shard_relative_path).parts)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as archive:
        archive.addfile(_tar_info("manifest.json", len(manifest_bytes)), io.BytesIO(manifest_bytes))
        for name in sorted(image_payloads):
            payload = image_payloads[name]
            archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
    return tar_path
