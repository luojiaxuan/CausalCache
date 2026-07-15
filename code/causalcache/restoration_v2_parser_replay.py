"""Offline replay audit for the immutable restoration-v2 screening trace."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2 import (
    gui_owl_v2_action_to_androidworld,
    parse_gui_owl_v2_output,
    serialize_gui_owl_v2_teacher_target,
)
from causalcache.policy.gui_owl_v2_compat import (
    GUI_OWL_V2_COMPATIBILITY_PARSER_CONTRACT_ID,
    inspect_gui_owl_v2_output_compatibility,
    parse_gui_owl_v2_output_compat,
)


PARSER_REPLAY_PROTOCOL_ID = "causalcache_restoration_v2_parser_replay_v1"
SOURCE_PROTOCOL_ID = "causalcache_restoration_v2"
EXPECTED_STATE_COUNT = 45
RUN_PREFIX_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_RUN_MANIFEST_KEYS = {
    "created_at_utc",
    "protocol_id",
    "run_contract",
    "run_contract_sha256",
    "schema_version",
    "status",
}
_AGGREGATE_KEYS = {
    "checks",
    "confirm_role_used",
    "failure_category_counts",
    "gate_contract",
    "gate_passed",
    "metrics",
    "outcome",
    "protocol_id",
    "run_contract_sha256",
    "sample_mutation_performed",
    "schema_version",
    "status",
    "top_up_performed",
}
_ATTEMPT_KEYS = {
    "created_at_utc",
    "protocol_id",
    "run_contract_sha256",
    "schema_version",
    "state",
    "status",
}
_STATE_KEYS = {
    "canonical_action",
    "distance_audits",
    "distances",
    "duration_seconds",
    "ended_at_utc",
    "failure",
    "finite_logit_distances",
    "message_shapes",
    "native_generations",
    "outcome",
    "parse_success",
    "parse_success_count",
    "protocol_id",
    "repeat_canonical_action_agreement",
    "run_contract_sha256",
    "schema_version",
    "started_at_utc",
    "state",
    "teacher_forwards",
}
_GENERATION_KEYS = {
    "canonical_action",
    "metadata",
    "output_text",
    "parse_error_message",
    "parse_error_type",
    "repeat_index",
}
_FAILURE_KEYS = {"category", "exception_type", "message", "stage"}
_GENERATION_METADATA_KEYS = {
    "do_sample",
    "effective_visual_tokens",
    "full_logit_tensor_host_transfers",
    "generated_tokens",
    "image_count",
    "image_grid_thw",
    "latency_seconds",
    "max_new_tokens",
    "peak_gpu_memory_allocated_bytes",
    "peak_gpu_memory_reserved_bytes",
    "policy_visible_text_tokens",
    "prompt_input_tokens",
}
_STATE_IDENTITY_KEYS = {
    "candidate_event_step_ids",
    "decision_step_id",
    "index",
    "role",
    "state_id",
    "trajectory_id",
}
_EXPECTED_CHECKS = {
    "minimum_finite_logit_coverage": False,
    "minimum_memory_sensitive_states": False,
    "minimum_parse_coverage": False,
    "minimum_repeat_canonical_action_agreement": False,
    "minimum_screening_states": True,
}
_EXPECTED_GATE_CONTRACT = {
    "fail_outcome": "NO_GO_V2_SUBSTRATE",
    "minimum_finite_logit_coverage": 1.0,
    "minimum_memory_sensitive_states": 8,
    "minimum_parse_coverage": 0.99,
    "minimum_repeat_canonical_action_agreement": 1.0,
    "minimum_screening_states": 20,
    "pass_outcome": "RUN_UNTOUCHED_RESTORATION_CONFIRM",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _strict_json_object(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expected_archive_members(run_prefix: str) -> tuple[tuple[str, str], ...]:
    if RUN_PREFIX_PATTERN.fullmatch(run_prefix) is None:
        raise ValueError("run prefix must be one safe path component")
    members: list[tuple[str, str]] = [
        (run_prefix, "directory"),
        (f"{run_prefix}/aggregate.json", "file"),
        (f"{run_prefix}/attempts", "directory"),
    ]
    members.extend(
        (f"{run_prefix}/attempts/{index:03d}.json", "file")
        for index in range(EXPECTED_STATE_COUNT)
    )
    members.extend(
        (
            (f"{run_prefix}/run_manifest.json", "file"),
            (f"{run_prefix}/states", "directory"),
        )
    )
    members.extend(
        (f"{run_prefix}/states/{index:03d}.json", "file")
        for index in range(EXPECTED_STATE_COUNT)
    )
    members.append((f"{run_prefix}.log", "file"))
    return tuple(members)


def _load_archive_files(
    archive_path: str | Path,
    *,
    expected_archive_sha256: str,
    run_prefix: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive}")
    archive_bytes = archive.read_bytes()
    actual_archive_sha256 = _sha256_bytes(archive_bytes)
    if actual_archive_sha256 != expected_archive_sha256:
        raise ValueError("archive SHA256 differs from the explicit expectation")

    expected_members = _expected_archive_members(run_prefix)
    expected_names = [name for name, _ in expected_members]
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as handle:
        members = handle.getmembers()
        names = [member.name for member in members]
        if names != expected_names or len(names) != len(set(names)):
            raise ValueError("archive member order or exact inventory drifted")
        for member, (_, expected_kind) in zip(members, expected_members, strict=True):
            if member.uid != 0 or member.gid != 0 or member.mtime != 0:
                raise ValueError("archive ownership or normalized timestamp drifted")
            if expected_kind == "directory":
                if not member.isdir():
                    raise ValueError(f"archive member must be a directory: {member.name}")
                continue
            if not member.isfile():
                raise ValueError(f"archive member must be a regular file: {member.name}")
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"archive file cannot be read: {member.name}")
            payload = source.read()
            if len(payload) != member.size:
                raise ValueError(f"archive member size drifted: {member.name}")
            files[member.name] = payload
    if len(files) != 93:
        raise ValueError("archive must contain exactly 93 regular files")
    return files, {
        "archive_filename": archive.name,
        "archive_sha256": actual_archive_sha256,
        "archive_size_bytes": len(archive_bytes),
        "member_count": len(expected_members),
        "regular_file_count": len(files),
    }


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def _counter_dict(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def audit_restoration_v2_parser_compatibility_archive(
    *,
    archive_path: str | Path,
    expected_archive_sha256: str,
    run_prefix: str,
    expected_run_contract_sha256: str,
    expected_source_run_git_commit: str,
) -> dict[str, Any]:
    files, archive_identity = _load_archive_files(
        archive_path,
        expected_archive_sha256=expected_archive_sha256,
        run_prefix=run_prefix,
    )
    run_manifest_name = f"{run_prefix}/run_manifest.json"
    aggregate_name = f"{run_prefix}/aggregate.json"
    log_name = f"{run_prefix}.log"
    run_manifest = _strict_json_object(files[run_manifest_name], name=run_manifest_name)
    aggregate = _strict_json_object(files[aggregate_name], name=aggregate_name)
    _require_exact_keys(run_manifest, _RUN_MANIFEST_KEYS, "run manifest")
    _require_exact_keys(aggregate, _AGGREGATE_KEYS, "aggregate")

    if (
        run_manifest.get("schema_version") != "1.0.0"
        or run_manifest.get("protocol_id") != SOURCE_PROTOCOL_ID
        or run_manifest.get("status") != "RESTORATION_V2_SUBSTRATE_SCREENING"
        or run_manifest.get("run_contract_sha256")
        != expected_run_contract_sha256
    ):
        raise ValueError("run manifest identity drifted")
    run_contract = run_manifest.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("run manifest lacks a run contract object")
    recomputed_contract_sha256 = _sha256_bytes(_canonical_json_bytes(run_contract))
    if recomputed_contract_sha256 != expected_run_contract_sha256:
        raise ValueError("run contract canonical SHA256 drifted")
    if run_contract.get("git_commit") != expected_source_run_git_commit:
        raise ValueError("source run Git commit drifted")
    authorization = run_contract.get("authorization")
    if not isinstance(authorization, Mapping) or (
        authorization.get("confirm_state") != "CONFIRM_LOCKED"
        or authorization.get("confirm_locked") is not True
    ):
        raise ValueError("source run did not preserve the confirm lock")
    contract_states = run_contract.get("states")
    if not isinstance(contract_states, list) or len(contract_states) != EXPECTED_STATE_COUNT:
        raise ValueError("run contract state denominator drifted")
    if [state.get("index") for state in contract_states if isinstance(state, Mapping)] != list(
        range(EXPECTED_STATE_COUNT)
    ):
        raise ValueError("run contract state ordering drifted")
    state_ids: list[str] = []
    roles: list[str] = []
    for index, state in enumerate(contract_states):
        if not isinstance(state, Mapping) or set(state) != _STATE_IDENTITY_KEYS:
            raise ValueError("run contract state identity schema drifted")
        state_id = state.get("state_id")
        trajectory_id = state.get("trajectory_id")
        role = state.get("role")
        if (
            type(state.get("index")) is not int
            or state.get("index") != index
            or not isinstance(state_id, str)
            or not state_id
            or not isinstance(trajectory_id, str)
            or not trajectory_id
            or role not in {"v2_label_train", "v2_development"}
            or state.get("decision_step_id") not in {4, 5, 6}
            or not isinstance(state.get("candidate_event_step_ids"), list)
        ):
            raise ValueError("run contract state identity value drifted")
        state_ids.append(state_id)
        roles.append(role)
    if len(set(state_ids)) != EXPECTED_STATE_COUNT:
        raise ValueError("run contract state IDs are not unique")
    if Counter(roles) != Counter({"v2_label_train": 30, "v2_development": 15}):
        raise ValueError("run contract role denominator drifted")

    if (
        aggregate.get("schema_version") != "1.0.0"
        or aggregate.get("protocol_id") != SOURCE_PROTOCOL_ID
        or aggregate.get("status") != "COMPLETED_FIXED_45_STATE_SUBSTRATE_SCREENING"
        or aggregate.get("outcome") != "NO_GO_V2_SUBSTRATE"
        or aggregate.get("gate_passed") is not False
        or aggregate.get("run_contract_sha256") != expected_run_contract_sha256
        or aggregate.get("failure_category_counts") != {"PARSE_FAILURE": 45}
        or aggregate.get("confirm_role_used") is not False
        or aggregate.get("sample_mutation_performed") is not False
        or aggregate.get("top_up_performed") is not False
        or aggregate.get("checks") != _EXPECTED_CHECKS
        or aggregate.get("gate_contract") != _EXPECTED_GATE_CONTRACT
    ):
        raise ValueError("official aggregate is not the fixed parser-failure result")
    metrics = aggregate.get("metrics")
    if not isinstance(metrics, Mapping) or (
        metrics.get("fixed_state_denominator") != EXPECTED_STATE_COUNT
        or metrics.get("parse_success_count") != 0
        or metrics.get("parse_coverage") != 0.0
    ):
        raise ValueError("official aggregate parse denominator drifted")

    records: list[dict[str, Any]] = []
    inspections = []
    for index, contract_state in enumerate(contract_states):
        if not isinstance(contract_state, Mapping):
            raise ValueError("run contract contains a non-object state")
        attempt_name = f"{run_prefix}/attempts/{index:03d}.json"
        state_name = f"{run_prefix}/states/{index:03d}.json"
        attempt = _strict_json_object(files[attempt_name], name=attempt_name)
        state = _strict_json_object(files[state_name], name=state_name)
        _require_exact_keys(attempt, _ATTEMPT_KEYS, f"attempt {index:03d}")
        _require_exact_keys(state, _STATE_KEYS, f"state {index:03d}")
        if (
            attempt.get("protocol_id") != SOURCE_PROTOCOL_ID
            or attempt.get("schema_version") != "1.0.0"
            or attempt.get("run_contract_sha256") != expected_run_contract_sha256
            or attempt.get("status") != "ATTEMPT_STARTED_NO_RETRY"
            or attempt.get("state") != contract_state
        ):
            raise ValueError(f"attempt marker drifted: {index:03d}")
        failure = state.get("failure")
        generations = state.get("native_generations")
        if (
            state.get("protocol_id") != SOURCE_PROTOCOL_ID
            or state.get("schema_version") != "1.0.0"
            or state.get("run_contract_sha256") != expected_run_contract_sha256
            or state.get("state") != contract_state
            or state.get("outcome") != "FAILED_SUBSTRATE_STATE"
            or state.get("parse_success") is not False
            or state.get("parse_success_count") != 0
            or state.get("canonical_action") is not None
            or state.get("repeat_canonical_action_agreement") is not False
            or state.get("finite_logit_distances") is not False
            or state.get("teacher_forwards") != {}
            or state.get("distance_audits") != {}
            or state.get("distances")
            != {"repeat_reference_kl": None, "summary_reference_kl": None}
            or not isinstance(failure, Mapping)
            or set(failure) != _FAILURE_KEYS
            or failure.get("category") != "PARSE_FAILURE"
            or failure.get("exception_type") != "GUIOwlV2GenerationParseError"
            or failure.get("stage") != "reference_generation_1"
            or not isinstance(generations, list)
            or len(generations) != 1
        ):
            raise ValueError(f"terminal parser-failure state drifted: {index:03d}")
        generation = generations[0]
        if not isinstance(generation, Mapping):
            raise ValueError(f"generation is not an object: {index:03d}")
        _require_exact_keys(generation, _GENERATION_KEYS, f"generation {index:03d}")
        output_text = generation.get("output_text")
        metadata = generation.get("metadata")
        if (
            not isinstance(output_text, str)
            or generation.get("repeat_index") != 1
            or generation.get("canonical_action") is not None
            or generation.get("parse_error_type") != "ValueError"
            or not isinstance(generation.get("parse_error_message"), str)
            or not isinstance(metadata, Mapping)
            or set(metadata) != _GENERATION_METADATA_KEYS
            or type(metadata.get("generated_tokens")) is not int
            or metadata.get("generated_tokens") <= 0
            or metadata.get("do_sample") is not False
            or metadata.get("max_new_tokens") != 256
            or metadata.get("full_logit_tensor_host_transfers") != 0
        ):
            raise ValueError(f"generation failure payload drifted: {index:03d}")
        try:
            parse_gui_owl_v2_output(output_text)
        except ValueError:
            pass
        else:
            raise ValueError(f"recorded official parse failure now parses strictly: {index:03d}")

        inspection = inspect_gui_owl_v2_output_compatibility(output_text)
        inspections.append(inspection)
        canonical_action = None
        canonical_tool_call_sha256 = None
        if inspection.compatibility_parse_success:
            compatible = parse_gui_owl_v2_output_compat(output_text)
            canonical_action = compatible.parsed_output.canonical_action.action
            canonical_tool_call_sha256 = compatible.canonical_tool_call_sha256
            strict_round_trip = parse_gui_owl_v2_output(
                serialize_gui_owl_v2_teacher_target(
                    compatible.parsed_output.canonical_action
                )
            )
            if strict_round_trip.canonical_action != compatible.parsed_output.canonical_action:
                raise ValueError(f"canonical strict round trip drifted: {index:03d}")
            gui_owl_v2_action_to_androidworld(
                compatible.parsed_output.canonical_action,
                screen_width=1000,
                screen_height=1000,
            )
        records.append(
            {
                "state_index": index,
                "state_id": contract_state.get("state_id"),
                "raw_output_sha256": _sha256_bytes(output_text.encode("utf-8")),
                "generated_tokens": metadata["generated_tokens"],
                "strict_parse_success": inspection.strict_parse_success,
                "single_action_first_balanced_json": inspection.first_balanced_json,
                "canonical_first_payload": inspection.canonical_first_payload,
                "compatibility_parse_success": inspection.compatibility_parse_success,
                "wrapper_variant": inspection.wrapper_variant,
                "suffix_variant": inspection.suffix_variant,
                "rejection_code": inspection.rejection_code,
                "canonical_action": canonical_action,
                "canonical_tool_call_sha256": canonical_tool_call_sha256,
            }
        )

    strict_count = sum(item.strict_parse_success for item in inspections)
    first_balanced_count = sum(item.first_balanced_json for item in inspections)
    canonical_first_count = sum(item.canonical_first_payload for item in inspections)
    compatibility_count = sum(item.compatibility_parse_success for item in inspections)
    gate = aggregate["gate_contract"]
    if not isinstance(gate, Mapping):
        raise ValueError("official aggregate lacks gate contract")
    minimum_parse_coverage = gate.get("minimum_parse_coverage")
    if not isinstance(minimum_parse_coverage, (int, float)) or isinstance(
        minimum_parse_coverage, bool
    ):
        raise ValueError("minimum parse coverage is not numeric")
    required_parse_count = math.ceil(
        float(minimum_parse_coverage) * EXPECTED_STATE_COUNT
    )
    if required_parse_count != 45:
        raise ValueError("fixed parse threshold no longer requires 45/45")

    wrapper_values = [
        record["wrapper_variant"]
        for record in records
        if record["canonical_first_payload"]
        and isinstance(record["wrapper_variant"], str)
    ]
    suffix_values = [
        record["suffix_variant"]
        for record in records
        if record["compatibility_parse_success"]
        and isinstance(record["suffix_variant"], str)
    ]
    action_values = [
        record["canonical_action"]
        for record in records
        if isinstance(record["canonical_action"], str)
    ]
    rejection_values = [
        record["rejection_code"]
        for record in records
        if isinstance(record["rejection_code"], str)
    ]
    suffix_counts = _counter_dict(suffix_values)
    clean_eof_count = suffix_counts.get("eof", 0)
    canonical_closer_count = suffix_counts.get("canonical_closer", 0)
    suffix_recovery_count = compatibility_count - clean_eof_count - canonical_closer_count
    adapter_gate_passed = compatibility_count >= required_parse_count
    return {
        "schema_version": "1.0.0",
        "protocol_id": PARSER_REPLAY_PROTOCOL_ID,
        "status": "COMPLETED_IMMUTABLE_PARSER_COMPATIBILITY_REPLAY",
        "outcome": (
            "PASSED_ADAPTER_GATE" if adapter_gate_passed else "NO_GO_ADAPTER_ONLY"
        ),
        "source_run": {
            "protocol_id": SOURCE_PROTOCOL_ID,
            "git_commit": expected_source_run_git_commit,
            "run_contract_sha256": expected_run_contract_sha256,
            "run_manifest_sha256": _sha256_bytes(files[run_manifest_name]),
            "aggregate_sha256": _sha256_bytes(files[aggregate_name]),
            "log_sha256": _sha256_bytes(files[log_name]),
            "official_outcome": aggregate["outcome"],
            "official_strict_parse_success_count": metrics["parse_success_count"],
            "official_result_modified": False,
        },
        "archive": archive_identity,
        "parser_contract": {
            "compatibility_parser_contract_id": (
                GUI_OWL_V2_COMPATIBILITY_PARSER_CONTRACT_ID
            ),
            "strict_parser_runs_first": True,
            "syntax_completion_allowed": False,
            "multiple_action_collapse_allowed": False,
            "second_json_allowed": False,
            "observation_suffix_allowed": False,
            "suffix_recovery_allowed": True,
            "allowed_suffix_variants": [
                "eof",
                "canonical_closer",
                "bare_tool_call_echo",
                "extra_closing_brace",
                "extra_brace_then_tool_call_echo",
            ],
            "compatibility_acceptance_is_native_well_formed_parse": False,
        },
        "metrics": {
            "fixed_state_denominator": EXPECTED_STATE_COUNT,
            "strict_parse_success_count": strict_count,
            "single_action_outputs_with_first_balanced_json_count": (
                first_balanced_count
            ),
            "single_action_outputs_with_canonical_first_payload_count": (
                canonical_first_count
            ),
            "conservative_compatibility_parse_success_count": compatibility_count,
            "conservative_compatibility_parse_coverage": (
                compatibility_count / EXPECTED_STATE_COUNT
            ),
            "minimum_parse_coverage": float(minimum_parse_coverage),
            "required_parse_success_count": required_parse_count,
            "clean_eof_compatibility_count": clean_eof_count,
            "model_emitted_canonical_closer_count": canonical_closer_count,
            "compatibility_acceptance_requiring_suffix_recovery_count": (
                suffix_recovery_count
            ),
            "adapter_gate_passed": adapter_gate_passed,
        },
        "inventory": {
            "wrapper_counts_for_canonical_first_payloads": _counter_dict(
                wrapper_values
            ),
            "safe_suffix_counts_for_accepted_outputs": suffix_counts,
            "accepted_canonical_action_counts": _counter_dict(action_values),
            "rejection_code_counts": _counter_dict(rejection_values),
        },
        "records": records,
        "negative_declarations": {
            "archive_extracted_to_filesystem": False,
            "confirm_role_used": False,
            "input_records_modified": False,
            "model_imported": False,
            "model_loaded": False,
            "policy_forward_executed": False,
            "policy_generation_executed": False,
            "state_retry_performed": False,
            "teacher_forward_executed": False,
            "top_up_performed": False,
        },
    }
