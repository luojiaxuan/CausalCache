"""Fail-closed independent multi-trajectory UI-TARS reference gate."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
import subprocess
import tarfile
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.policy.qwen_runtime import QwenPolicyRuntime, action_dict
from causalcache.policy.ui_tars import build_ui_tars_messages, parse_ui_tars_action
from causalcache.schema import ExecutableAction, LowFidelityEvent


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_sha256(value: Any, *, name: str) -> str:
    result = _required_string(value, name=name)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return result


def _normalized_app_label(value: Any) -> str:
    label = unicodedata.normalize("NFKC", _required_string(value, name="app label"))
    normalized = re.sub(r"\s+", " ", label).strip().casefold()
    if not normalized:
        raise ValueError("normalized app label must not be empty")
    return normalized


def _git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest_payload(dataset_tar: Path) -> tuple[dict[str, Any], bytes]:
    with tarfile.open(dataset_tar) as archive:
        member = archive.extractfile("manifest.json")
        if member is None:
            raise ValueError("dataset tar does not contain manifest.json")
        payload = member.read()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be a JSON object")
    return manifest, payload


def _trajectory_source_id(trajectory: Mapping[str, Any]) -> str:
    return _required_string(trajectory.get("source_id"), name="trajectory source_id")


def _split_source_ids(split: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(split, Mapping):
        raise ValueError(f"split {name} must be an object")
    values = split.get("source_ids")
    if not isinstance(values, list) or not values:
        raise ValueError(f"split {name}.source_ids must be a non-empty list")
    source_ids = tuple(_required_string(value, name=f"split {name} source_id") for value in values)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"split {name} contains duplicate source ids")
    return source_ids


def _selection_key(
    trajectory: Mapping[str, Any],
    *,
    salt: str,
) -> tuple[str, str, str, int]:
    source_id = _trajectory_source_id(trajectory)
    expected_hash = hashlib.sha256(f"{salt}\0{source_id}".encode("utf-8")).hexdigest()
    selection_hash = trajectory.get("selection_sha256")
    if selection_hash != expected_hash:
        raise ValueError(f"trajectory {source_id} has an invalid selection SHA256")
    transport_file = _required_string(
        trajectory.get("transport_file"),
        name=f"trajectory {source_id} transport_file",
    )
    row_index = trajectory.get("transport_row_index")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise ValueError(f"trajectory {source_id} transport_row_index must be non-negative")
    return expected_hash, source_id, transport_file, row_index


def _trajectory_decisions(trajectory: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    source_id = _trajectory_source_id(trajectory)
    decisions = trajectory.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError(f"trajectory {source_id} must contain decisions")
    result: list[Mapping[str, Any]] = []
    step_ids: list[int] = []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError(f"trajectory {source_id} decision must be an object")
        step_id = decision.get("decision_step_id")
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id <= 1:
            raise ValueError(f"trajectory {source_id} has an invalid decision step id")
        history = decision.get("history_event_step_ids")
        if history != list(range(1, step_id)):
            raise ValueError(
                f"trajectory {source_id} decision {step_id} is not full-prefix aligned"
            )
        if not isinstance(decision.get("validated_action"), Mapping):
            raise ValueError(
                f"trajectory {source_id} decision {step_id} lacks a validated action"
            )
        ExecutableAction.from_dict(decision["validated_action"])
        _required_string(
            decision.get("current_observation_path"),
            name=f"trajectory {source_id} current observation path",
        )
        step_ids.append(step_id)
        result.append(decision)
    if step_ids != sorted(set(step_ids)):
        raise ValueError(f"trajectory {source_id} decision ids must be unique and sorted")
    return tuple(result)


def _validate_trajectory(
    trajectory: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source_id = _trajectory_source_id(trajectory)
    eligibility = config["eligibility"]
    if re.fullmatch(str(eligibility["source_id_regex"]), source_id) is None:
        raise ValueError(f"trajectory {source_id} violates the frozen source-id grammar")
    if trajectory.get("platform") != eligibility["platform"]:
        raise ValueError(f"trajectory {source_id} has an invalid platform")
    if trajectory.get("terminal_status") != eligibility["terminal_status"]:
        raise ValueError(f"trajectory {source_id} is not a successful frozen trajectory")
    _required_string(trajectory.get("instruction"), name=f"trajectory {source_id} instruction")
    apps = trajectory.get("apps")
    if not isinstance(apps, list) or not apps:
        raise ValueError(f"trajectory {source_id} must include at least one app label")
    normalized_apps = tuple(sorted({_normalized_app_label(value) for value in apps}))
    decisions = _trajectory_decisions(trajectory)
    minimum = int(eligibility["minimum_decisions_per_trajectory"])
    maximum = int(eligibility["maximum_decisions_per_trajectory"])
    if not minimum <= len(decisions) <= maximum:
        raise ValueError(f"trajectory {source_id} is outside the frozen decision range")
    compatible_types = set(str(value) for value in eligibility["parser_compatible_action_types"])
    action_types = tuple(
        str(decision["validated_action"]["action_type"]) for decision in decisions
    )
    unsupported = sorted(set(action_types) - compatible_types)
    if unsupported:
        raise ValueError(
            f"trajectory {source_id} contains parser-incompatible actions: {unsupported}"
        )
    events = trajectory.get("events")
    if not isinstance(events, list) or not all(isinstance(event, Mapping) for event in events):
        raise ValueError(f"trajectory {source_id} events must be a list")
    event_ids = [event.get("step_id") for event in events]
    maximum_history = max(len(decision["history_event_step_ids"]) for decision in decisions)
    if event_ids != list(range(1, maximum_history + 1)):
        raise ValueError(f"trajectory {source_id} events do not cover every decision prefix")
    required_event_fields = (
        "observation_before_path",
        "observation_after_path",
        "executed_action",
        "source_tool_call",
        "low_fidelity",
    )
    for event in events:
        if any(field not in event for field in required_event_fields):
            raise ValueError(f"trajectory {source_id} event is incomplete")
        _required_string(
            event["observation_before_path"],
            name=f"trajectory {source_id} event before image",
        )
        _required_string(
            event["observation_after_path"],
            name=f"trajectory {source_id} event after image",
        )
        if not isinstance(event["executed_action"], Mapping):
            raise ValueError(f"trajectory {source_id} event executed action is invalid")
        ExecutableAction.from_dict(event["executed_action"])
        if not isinstance(event["source_tool_call"], Mapping):
            raise ValueError(f"trajectory {source_id} event source tool call is invalid")
        if not isinstance(event["low_fidelity"], Mapping):
            raise ValueError(f"trajectory {source_id} event low-fidelity record is invalid")
        low_fidelity = LowFidelityEvent.from_dict(event["low_fidelity"])
        if low_fidelity.step_id != event["step_id"]:
            raise ValueError(f"trajectory {source_id} event low-fidelity step is inconsistent")
    return {
        "source_id": source_id,
        "decisions": decisions,
        "decision_count": len(decisions),
        "action_types": action_types,
        "normalized_apps": normalized_apps,
    }


def validate_v04_manifest(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema_version") != "0.4.0":
        raise ValueError("independent dataset manifest must use schema version 0.4.0")
    if manifest.get("protocol_id") != config["protocol_id"]:
        raise ValueError("dataset manifest protocol does not match the frozen config")
    artifact = config["artifact"]
    if manifest.get("dataset_repo") != artifact["repo"]:
        raise ValueError("dataset manifest repo does not match the frozen config")

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("dataset manifest must include source provenance")
    source_pool = config["source_pool"]
    for key in ("upstream_repo", "upstream_revision", "transport_repo", "transport_revision"):
        if source.get(key) != source_pool[key]:
            raise ValueError(f"dataset source {key} does not match the frozen config")
    manifest_transport_files = source.get("transport_files")
    if not isinstance(manifest_transport_files, list):
        raise ValueError("dataset source transport_files must be a list")
    observed_transport_files = [
        value["transport_file"] if isinstance(value, Mapping) else value
        for value in manifest_transport_files
    ]
    if observed_transport_files != source_pool["transport_files"]:
        raise ValueError("dataset source files do not match the frozen source pool")
    transport_file_records: dict[str, Mapping[str, Any]] = {}
    for value in manifest_transport_files:
        if not isinstance(value, Mapping):
            raise ValueError("every transport-file record must contain immutable provenance")
        transport_file = _required_string(
            value.get("transport_file"),
            name="transport file",
        )
        _required_sha256(value.get("sha256"), name=f"transport file {transport_file} SHA256")
        for field in ("size_bytes", "row_count"):
            observed = value.get(field)
            if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
                raise ValueError(f"transport file {transport_file} {field} must be positive")
        transport_file_records[transport_file] = value

    trajectories = manifest.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("dataset manifest trajectories must be a non-empty list")
    if not all(isinstance(value, Mapping) for value in trajectories):
        raise ValueError("every trajectory manifest entry must be an object")
    source_ids = [_trajectory_source_id(value) for value in trajectories]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("dataset manifest contains duplicate source ids")
    excluded = set(str(value) for value in source_pool["excluded_source_ids"])
    leaked = sorted(set(source_ids) & excluded)
    if leaked:
        raise ValueError(f"dataset contains excluded source ids: {leaked}")
    transport_rows: set[tuple[str, int]] = set()
    for trajectory in trajectories:
        transport_file = _required_string(
            trajectory.get("transport_file"),
            name="trajectory transport file",
        )
        if transport_file not in transport_file_records:
            raise ValueError("trajectory references a source file outside the frozen pool")
        row_index = trajectory.get("transport_row_index")
        if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
            raise ValueError("trajectory source row must be a non-negative integer")
        if row_index >= int(transport_file_records[transport_file]["row_count"]):
            raise ValueError("trajectory source row exceeds the recorded source file")
        source_row = transport_file, row_index
        if source_row in transport_rows:
            raise ValueError("dataset contains a duplicate source row")
        transport_rows.add(source_row)

    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("dataset manifest must contain frozen splits")
    reference_ids = _split_source_ids(splits.get("reference_gate"), name="reference_gate")
    oracle_ids = _split_source_ids(splits.get("oracle_pilot"), name="oracle_pilot")
    overlap = sorted(set(reference_ids) & set(oracle_ids))
    if overlap:
        raise ValueError(f"reference and oracle splits overlap: {overlap}")
    if set(source_ids) != set(reference_ids) | set(oracle_ids):
        raise ValueError("every artifact trajectory must belong to exactly one frozen split")

    by_source = {_trajectory_source_id(value): value for value in trajectories}
    salt = str(config["selection"]["trajectory_salt"])
    expected_global_order = sorted(trajectories, key=lambda value: _selection_key(value, salt=salt))
    expected_reference_order = [
        _trajectory_source_id(value)
        for value in expected_global_order
        if _trajectory_source_id(value) in set(reference_ids)
    ]
    expected_oracle_order = [
        _trajectory_source_id(value)
        for value in expected_global_order
        if _trajectory_source_id(value) in set(oracle_ids)
    ]
    if list(reference_ids) != expected_reference_order:
        raise ValueError("reference split is not in frozen selection order")
    if list(oracle_ids) != expected_oracle_order:
        raise ValueError("oracle split is not in frozen selection order")

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("dataset manifest must include frozen selection metadata")
    for key in ("algorithm", "trajectory_salt", "trajectory_hash_input", "trajectory_order"):
        if selection.get(key) != config["selection"][key]:
            raise ValueError(f"dataset selection {key} differs from the frozen config")
    if selection.get("eligibility") != config["eligibility"]:
        raise ValueError("dataset selection eligibility differs from the frozen config")
    if selection.get("excluded_source_ids") != source_pool["excluded_source_ids"]:
        raise ValueError("dataset selection exclusion list differs from the frozen config")
    _required_sha256(
        selection.get("eligible_pool_sha256"),
        name="eligible pool SHA256",
    )
    eligible_pool_count = selection.get("eligible_pool_count")
    total_source_rows = selection.get("total_source_rows")
    if (
        isinstance(eligible_pool_count, bool)
        or not isinstance(eligible_pool_count, int)
        or eligible_pool_count < len(trajectories)
    ):
        raise ValueError("eligible pool count is inconsistent")
    if (
        isinstance(total_source_rows, bool)
        or not isinstance(total_source_rows, int)
        or total_source_rows < eligible_pool_count
    ):
        raise ValueError("total source-row count is inconsistent")
    exclusion_counts = selection.get("exclusion_counts")
    if not isinstance(exclusion_counts, Mapping) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in exclusion_counts.values()
    ):
        raise ValueError("dataset exclusion counts are invalid")
    if eligible_pool_count + sum(exclusion_counts.values()) != total_source_rows:
        raise ValueError("eligible and excluded rows do not reconstruct the frozen source pool")
    for field in ("protocol_config_sha256", "source_file_manifest_sha256"):
        _required_sha256(source.get(field), name=f"dataset source {field}")

    validated = {
        source_id: _validate_trajectory(by_source[source_id], config=config)
        for source_id in source_ids
    }
    required_types = tuple(str(value) for value in config["selection"]["required_action_types"])
    split_summaries: dict[str, Any] = {}
    for split_name, split_ids in (
        ("reference_gate", reference_ids),
        ("oracle_pilot", oracle_ids),
    ):
        quota = config["selection"][split_name]
        decision_count = sum(validated[source_id]["decision_count"] for source_id in split_ids)
        app_labels = sorted(
            {
                app
                for source_id in split_ids
                for app in validated[source_id]["normalized_apps"]
            }
        )
        observed_action_types = sorted(
            {
                action
                for source_id in split_ids
                for action in validated[source_id]["action_types"]
            }
        )
        action_counts = {
            action_type: sum(
                action == action_type
                for source_id in split_ids
                for action in validated[source_id]["action_types"]
            )
            for action_type in observed_action_types
        }
        if len(split_ids) < int(quota["minimum_trajectories"]):
            raise ValueError(f"split {split_name} does not meet its trajectory quota")
        if decision_count < int(quota["minimum_decisions"]):
            raise ValueError(f"split {split_name} does not meet its decision quota")
        if len(app_labels) < int(quota["minimum_distinct_app_labels"]):
            raise ValueError(f"split {split_name} does not meet its app-label quota")
        minimum_actions = int(quota["minimum_candidate_count_per_required_action_type"])
        if any(action_counts.get(action_type, 0) < minimum_actions for action_type in required_types):
            raise ValueError(f"split {split_name} lacks a frozen required action type")
        split_manifest = splits[split_name]
        if int(split_manifest.get("trajectory_count", -1)) != len(split_ids):
            raise ValueError(f"split {split_name} trajectory count is inconsistent")
        if int(split_manifest.get("decision_count", -1)) != decision_count:
            raise ValueError(f"split {split_name} decision count is inconsistent")
        if split_manifest.get("distinct_app_labels") != app_labels:
            raise ValueError(f"split {split_name} app labels are inconsistent")
        if split_manifest.get("action_type_counts") != action_counts:
            raise ValueError(f"split {split_name} action counts are inconsistent")
        for source_id in split_ids:
            trajectory = by_source[source_id]
            if trajectory.get("split") != split_name:
                raise ValueError(f"trajectory {source_id} has an inconsistent split label")
            if trajectory.get("normalized_app_labels") != list(
                validated[source_id]["normalized_apps"]
            ):
                raise ValueError(f"trajectory {source_id} has inconsistent normalized apps")
        split_summaries[split_name] = {
            "source_ids": list(split_ids),
            "trajectories": len(split_ids),
            "decisions": decision_count,
            "normalized_app_labels": app_labels,
            "action_type_counts": action_counts,
            "required_action_candidate_counts": {
                action_type: action_counts[action_type] for action_type in required_types
            },
        }
    return {
        "by_source": by_source,
        "validated": validated,
        "splits": split_summaries,
    }


def _coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matches = sum(record["executable_match"] is True for record in records)
    parsed = sum(record["parse_error"] is None for record in records)
    return {
        "decisions": len(records),
        "parsed": parsed,
        "matches": matches,
        "coverage": matches / len(records) if records else 0.0,
    }


def reduce_reference_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not records:
        raise ValueError("reference gate cannot reduce an empty denominator")
    identities: set[tuple[str, int]] = set()
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        source_id = _required_string(record.get("trajectory_source_id"), name="record source id")
        step_id = record.get("decision_step_id")
        if isinstance(step_id, bool) or not isinstance(step_id, int):
            raise ValueError("record decision step id must be an integer")
        identity = source_id, step_id
        if identity in identities:
            raise ValueError(f"duplicate reference decision record: {identity}")
        identities.add(identity)
        executable_match = record.get("executable_match")
        if not isinstance(executable_match, bool):
            raise ValueError("record executable_match must be boolean")
        if record.get("parse_error") is not None and executable_match:
            raise ValueError("a parse failure cannot executable-match")
        validated_action = record.get("validated_action")
        if not isinstance(validated_action, Mapping):
            raise ValueError("record validated_action must be an object")
        action_type = _required_string(
            validated_action.get("action_type"),
            name="record validated action type",
        )
        grouped[action_type].append(record)

    gate = config["reference_gate"]
    required_types = tuple(str(value) for value in gate["required_action_types"])
    by_action_type = {
        action_type: _coverage(grouped.get(action_type, []))
        for action_type in sorted(set(grouped) | set(required_types))
    }
    if any(by_action_type[action_type]["decisions"] == 0 for action_type in required_types):
        raise ValueError("the reference denominator is missing a required action type")
    overall = _coverage(records)
    threshold = float(gate["minimum_overall_coverage"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("reference coverage threshold must be a probability")
    overall_pass = overall["coverage"] >= threshold
    action_type_pass = all(
        by_action_type[action_type]["matches"] >= 1 for action_type in required_types
    )
    passed = overall_pass and action_type_pass
    return {
        "outcome": gate["pass_outcome"] if passed else gate["fail_outcome"],
        "overall": overall,
        "by_action_type": by_action_type,
        "gate": {
            "minimum_overall_coverage": threshold,
            "required_action_types": list(required_types),
            "overall_coverage_pass": overall_pass,
            "action_type_coverage_pass": action_type_pass,
            "passed": passed,
        },
    }


def validate_hardware_anchor(
    anchor_path: Path,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    if not isinstance(anchor, Mapping):
        raise ValueError("hardware anchor summary must be a JSON object")
    if anchor.get("outcome") != "HARDWARE_ANCHOR_PASSED":
        raise ValueError("hardware anchor does not record a passing outcome")
    expected = config["hardware_anchor"]
    dataset = anchor.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("hardware anchor lacks dataset provenance")
    if dataset.get("repo") != expected["dataset_repo"]:
        raise ValueError("hardware anchor dataset repo mismatch")
    if dataset.get("revision") != expected["dataset_revision"]:
        raise ValueError("hardware anchor dataset revision mismatch")
    observed_shard_sha = dataset.get("shard_sha256")
    if observed_shard_sha is None and isinstance(anchor.get("run"), Mapping):
        observed_shard_sha = anchor["run"].get("dataset_sha256")
    if observed_shard_sha != expected["dataset_shard_sha256"]:
        raise ValueError("hardware anchor dataset shard SHA256 mismatch")
    _required_sha256(observed_shard_sha, name="hardware anchor dataset shard SHA256")
    model = anchor.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("snapshot"), Mapping):
        raise ValueError("hardware anchor lacks model snapshot provenance")
    if model["snapshot"].get("repo") != config["policy"]["repo"]:
        raise ValueError("hardware anchor model repo mismatch")
    if model["snapshot"].get("revision") != config["policy"]["revision"]:
        raise ValueError("hardware anchor model revision mismatch")
    if model.get("dtype") != config["policy"]["dtype"]:
        raise ValueError("hardware anchor policy dtype mismatch")
    overall = anchor.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError("hardware anchor lacks overall metrics")
    for field in ("decisions", "parsed", "matches"):
        value = overall.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"hardware anchor {field} must be a non-negative integer")
    if overall["decisions"] != len(expected["expected_decision_match_vector"]):
        raise ValueError("hardware anchor denominator mismatch")
    if overall["parsed"] != int(expected["expected_parsed"]):
        raise ValueError("hardware anchor parse count mismatch")
    if overall["matches"] != int(expected["expected_matches"]):
        raise ValueError("hardware anchor match count mismatch")
    decisions = anchor.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("hardware anchor lacks decision records")
    observed_vector = []
    for record in decisions:
        if not isinstance(record, Mapping):
            raise ValueError("hardware anchor decision must be an object")
        step_id = record.get("decision_step_id")
        executable_match = record.get("executable_match")
        if isinstance(step_id, bool) or not isinstance(step_id, int):
            raise ValueError("hardware anchor decision step must be an integer")
        if not isinstance(executable_match, bool):
            raise ValueError("hardware anchor executable match must be boolean")
        observed_vector.append([step_id, executable_match])
    if observed_vector != expected["expected_decision_match_vector"]:
        raise ValueError("hardware anchor decision-match vector mismatch")
    if int(anchor.get("visual_tokens_per_image", -1)) != int(
        config["policy"]["visual_tokens_per_image"]
    ):
        raise ValueError("hardware anchor visual preprocessing mismatch")
    if int(anchor.get("max_new_tokens", -1)) != int(config["policy"]["max_new_tokens"]):
        raise ValueError("hardware anchor generation limit mismatch")
    anchor_runtime = anchor.get("runtime")
    if not isinstance(anchor_runtime, Mapping):
        raise ValueError("hardware anchor lacks runtime provenance")
    for field in ("raw_summary_sha256", "gpu_monitor_sha256"):
        _required_sha256(anchor_runtime.get(field), name=f"hardware anchor {field}")
    return {
        "path": str(anchor_path),
        "sha256": sha256_file(anchor_path),
        "dataset": dict(dataset),
        "overall": dict(overall),
        "decision_match_vector": observed_vector,
        "runtime": dict(anchor_runtime),
    }


def _validate_dataset_members(
    dataset_tar: Path,
    *,
    manifest_validation: Mapping[str, Any],
) -> dict[str, Any]:
    required_paths: set[str] = set()
    for trajectory in manifest_validation["by_source"].values():
        for event in trajectory["events"]:
            required_paths.add(str(event["observation_before_path"]))
            required_paths.add(str(event["observation_after_path"]))
        for decision in trajectory["decisions"]:
            required_paths.add(str(decision["current_observation_path"]))
    for path in required_paths:
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != path:
            raise ValueError(f"dataset image path is unsafe or non-canonical: {path}")
    with tarfile.open(dataset_tar) as archive:
        members = archive.getmembers()
        member_names = [member.name for member in members]
        if len(member_names) != len(set(member_names)):
            raise ValueError("dataset tar contains duplicate member names")
        by_name = {member.name: member for member in members}
        missing = sorted(required_paths - set(by_name))
        if missing:
            raise ValueError(f"dataset tar is missing {len(missing)} referenced images")
        for path in required_paths:
            if not by_name[path].isfile() or by_name[path].size <= 0:
                raise ValueError(f"dataset image member is invalid: {path}")
    return {
        "referenced_images": len(required_paths),
        "tar_members": len(member_names),
    }


def _validate_base_gate(config: Mapping[str, Any], *, repo_root: Path) -> None:
    gate_path = repo_root / "code/configs/policy_coverage_gate.json"
    base_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate = config["reference_gate"]
    if base_gate["validation_mode"] != gate["validation_mode"]:
        raise ValueError("reference validation mode differs from the original gate")
    if float(base_gate["minimum_overall_coverage"]) != float(
        gate["minimum_overall_coverage"]
    ):
        raise ValueError("reference coverage threshold differs from the original gate")
    if base_gate["present_action_types"] != gate["required_action_types"]:
        raise ValueError("reference action-type gate differs from the original gate")
    if bool(base_gate["require_at_least_one_match_per_present_action_type"]) is not True:
        raise ValueError("original action-type gate is no longer enabled")


def _validate_model_snapshot(
    *,
    config: Mapping[str, Any],
    model_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    policy = config["policy"]
    specification_path = repo_root / str(policy["snapshot_manifest"])
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    snapshot = json.loads((model_dir / ".snapshot.json").read_text(encoding="utf-8"))
    for name, value in (("repo", policy["repo"]), ("revision", policy["revision"])):
        if specification.get(name) != value or snapshot.get(name) != value:
            raise ValueError(f"model snapshot {name} does not match the frozen policy")
    expected_files = {str(value["path"]): value for value in specification["files"]}
    observed_files = {str(value["path"]): value for value in snapshot["files"]}
    if set(expected_files) != set(observed_files):
        raise ValueError("model snapshot file set differs from the pinned manifest")
    for name, expected_file in expected_files.items():
        observed_file = observed_files[name]
        if int(observed_file["size"]) != int(expected_file["size"]):
            raise ValueError(f"model snapshot size mismatch for {name}")
        expected_sha = expected_file.get("sha256")
        if expected_sha is not None and observed_file.get("sha256") != expected_sha:
            raise ValueError(f"model snapshot SHA256 mismatch for {name}")
        local_path = model_dir / name
        if not local_path.is_file() or local_path.stat().st_size != int(expected_file["size"]):
            raise ValueError(f"model snapshot file is missing or truncated: {name}")
        observed_sha = _required_sha256(
            observed_file.get("sha256"),
            name=f"model snapshot recorded SHA256 for {name}",
        )
        if sha256_file(local_path) != observed_sha:
            raise ValueError(f"model snapshot content SHA256 mismatch for {name}")
    return {
        "snapshot_manifest_path": str(specification_path),
        "snapshot_manifest_sha256": sha256_file(specification_path),
        "snapshot": snapshot,
    }


def validate_reference_run_contract(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    dataset_tar: Path,
    hardware_anchor_summary: Path,
    model_dir: Path,
    device: str,
    run_git_commit: str,
    container_image_digest: str,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("protocol_id") != "independent_reference_gate_v1":
        raise ValueError("unexpected independent reference-gate protocol")
    if config.get("preregistration_status") != "frozen_before_source_row_decoding_or_policy_inference":
        raise ValueError("reference-gate config is not preregistered")
    if device != config["policy"]["device"]:
        raise ValueError("runtime device differs from the frozen policy config")
    if config["policy"].get("do_sample") is not False:
        raise ValueError("reference policy decoding must remain deterministic")
    if config["reference_gate"].get("parse_failure_counts_as_non_match") is not True:
        raise ValueError("parse failures must remain non-matches")
    if config["reference_gate"].get("run_entire_fixed_denominator") is not True:
        raise ValueError("reference gate must run the complete frozen denominator")
    if config["policy"].get("dtype") != "bfloat16":
        raise ValueError("reference policy dtype must remain bfloat16")
    if int(config["policy"].get("coordinate_grid_size", -1)) != 10:
        raise ValueError("reference executable equivalence must remain on the 10x10 grid")
    if int(config["policy"].get("visual_tokens_per_image", 0)) <= 0:
        raise ValueError("reference visual-token target must be positive")
    if int(config["policy"].get("max_new_tokens", 0)) <= 0:
        raise ValueError("reference generation limit must be positive")
    if config["selection"]["required_action_types"] != config["reference_gate"][
        "required_action_types"
    ]:
        raise ValueError("selection and reference gate action types differ")
    if not container_image_digest.startswith("sha256:") or _SHA256_PATTERN.fullmatch(
        container_image_digest.removeprefix("sha256:")
    ) is None:
        raise ValueError("container image digest must be an explicit sha256 digest")

    actual_commit = _git_output(repo_root, "rev-parse", "HEAD")
    if actual_commit != run_git_commit:
        raise ValueError("run Git commit does not match the checked-out commit")
    if _git_output(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("formal reference gate requires a clean Git worktree")

    interface_hashes = config["policy"]["frozen_interface_sha256"]
    for relative_path, expected_sha in interface_hashes.items():
        observed_sha = sha256_file(repo_root / relative_path)
        if observed_sha != expected_sha:
            raise ValueError(f"frozen interface changed: {relative_path}")
    _validate_base_gate(config, repo_root=repo_root)

    artifact = config["artifact"]
    artifact_revision = _required_string(artifact.get("revision"), name="artifact revision")
    if _REVISION_PATTERN.fullmatch(artifact_revision) is None:
        raise ValueError("artifact revision must be an immutable 40-character commit")
    expected_tar_sha = _required_sha256(artifact.get("shard_sha256"), name="artifact shard SHA256")
    expected_manifest_sha = _required_sha256(
        artifact.get("manifest_sha256"),
        name="artifact manifest SHA256",
    )
    observed_tar_sha = sha256_file(dataset_tar)
    if observed_tar_sha != expected_tar_sha:
        raise ValueError("independent dataset shard SHA256 mismatch")
    manifest, manifest_payload = _manifest_payload(dataset_tar)
    if sha256_bytes(manifest_payload) != expected_manifest_sha:
        raise ValueError("independent dataset manifest SHA256 mismatch")
    manifest_validation = validate_v04_manifest(manifest, config=config)
    dataset_members = _validate_dataset_members(
        dataset_tar,
        manifest_validation=manifest_validation,
    )
    anchor = validate_hardware_anchor(hardware_anchor_summary, config=config)
    model = _validate_model_snapshot(config=config, model_dir=model_dir, repo_root=repo_root)
    provenance = {
        "git_commit": actual_commit,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_path": str(dataset_tar),
        "dataset_sha256": observed_tar_sha,
        "manifest_sha256": expected_manifest_sha,
        "dataset_members": dataset_members,
        "hardware_anchor": anchor,
        "model": model,
        "container_image_digest": container_image_digest,
    }
    return provenance, manifest_validation


def _load_image(archive: tarfile.TarFile, relative_path: str) -> Any:
    from PIL import Image

    member = archive.extractfile(relative_path)
    if member is None:
        raise ValueError(f"dataset image is missing: {relative_path}")
    return Image.open(io.BytesIO(member.read())).convert("RGB")


def _gpu_metadata(runtime: QwenPolicyRuntime) -> dict[str, Any]:
    device_index = runtime.torch.device(runtime.device).index
    if device_index is None:
        raise ValueError("runtime device must contain an explicit CUDA index")
    properties = runtime.torch.cuda.get_device_properties(device_index)
    return {
        "logical_device": runtime.device,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
    }


def run_reference_gate(
    *,
    config_path: Path,
    dataset_tar: Path,
    hardware_anchor_summary: Path,
    model_dir: Path,
    device: str,
    run_git_commit: str,
    container_image_digest: str,
    repo_root: Path,
    argv: Sequence[str],
    runtime_factory: Callable[..., Any] = QwenPolicyRuntime,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance, manifest_validation = validate_reference_run_contract(
        config=config,
        config_path=config_path,
        dataset_tar=dataset_tar,
        hardware_anchor_summary=hardware_anchor_summary,
        model_dir=model_dir,
        device=device,
        run_git_commit=run_git_commit,
        container_image_digest=container_image_digest,
        repo_root=repo_root,
    )

    runtime = runtime_factory(
        model_dir=model_dir,
        device=device,
        visual_tokens_per_image=int(config["policy"]["visual_tokens_per_image"]),
    )
    records: list[dict[str, Any]] = []
    with tarfile.open(dataset_tar) as archive:
        reference_ids = manifest_validation["splits"]["reference_gate"]["source_ids"]
        by_source = manifest_validation["by_source"]

        def image_loader(path: str) -> Any:
            return _load_image(archive, path)

        first_trajectory = by_source[reference_ids[0]]
        first_decision = _trajectory_decisions(first_trajectory)[0]
        warmup_manifest = {
            "dataset_repo": config["artifact"]["repo"],
            "trajectory": first_trajectory,
        }
        runtime.warmup(
            build_ui_tars_messages(
                warmup_manifest,
                decision_step_id=int(first_decision["decision_step_id"]),
                restored_event_step_ids=[],
                image_loader=image_loader,
            )
        )

        for source_id in reference_ids:
            trajectory = by_source[source_id]
            trajectory_manifest = {
                "dataset_repo": config["artifact"]["repo"],
                "trajectory": trajectory,
            }
            for decision in _trajectory_decisions(trajectory):
                step_id = int(decision["decision_step_id"])
                history_ids = [int(value) for value in decision["history_event_step_ids"]]
                validated_action = ExecutableAction.from_dict(decision["validated_action"])
                result = runtime.generate(
                    build_ui_tars_messages(
                        trajectory_manifest,
                        decision_step_id=step_id,
                        restored_event_step_ids=history_ids,
                        image_loader=image_loader,
                    ),
                    max_new_tokens=int(config["policy"]["max_new_tokens"]),
                    validated_action=validated_action,
                    action_parser=parse_ui_tars_action,
                )
                records.append(
                    result
                    | {
                        "trajectory_source_id": source_id,
                        "trajectory_apps": list(trajectory["apps"]),
                        "decision_step_id": step_id,
                        "history_events": len(history_ids),
                        "validated_action": action_dict(validated_action),
                    }
                )

    expected_decisions = manifest_validation["splits"]["reference_gate"]["decisions"]
    if len(records) != expected_decisions:
        raise RuntimeError("reference runner did not evaluate the complete frozen denominator")
    expected_identities = {
        (source_id, int(decision["decision_step_id"]))
        for source_id in manifest_validation["splits"]["reference_gate"]["source_ids"]
        for decision in _trajectory_decisions(manifest_validation["by_source"][source_id])
    }
    observed_identities = {
        (str(record["trajectory_source_id"]), int(record["decision_step_id"]))
        for record in records
    }
    if observed_identities != expected_identities:
        raise RuntimeError("reference runner changed the frozen decision denominator")
    reduction = reduce_reference_gate(records, config=config)
    by_trajectory = {
        source_id: _coverage(
            [record for record in records if record["trajectory_source_id"] == source_id]
        )
        for source_id in manifest_validation["splits"]["reference_gate"]["source_ids"]
    }
    completed_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "0.1.0",
        "protocol_id": config["protocol_id"],
        "outcome": reduction["outcome"],
        "run": provenance
        | {
            "argv": list(argv),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "started_at": started_at,
            "completed_at": completed_at,
            "wall_time_seconds": time.perf_counter() - wall_start,
        },
        "runtime": runtime.metadata,
        "gpu": _gpu_metadata(runtime),
        "dataset": {
            "repo": config["artifact"]["repo"],
            "revision": config["artifact"]["revision"],
            "shard": config["artifact"]["shard"],
            "shard_sha256": config["artifact"]["shard_sha256"],
            "manifest_sha256": config["artifact"]["manifest_sha256"],
            "reference_split": manifest_validation["splits"]["reference_gate"],
        },
        "validation_mode": config["reference_gate"]["validation_mode"],
        "overall": reduction["overall"],
        "by_action_type": reduction["by_action_type"],
        "by_trajectory": by_trajectory,
        "gate": reduction["gate"],
        "decisions": records,
    }
