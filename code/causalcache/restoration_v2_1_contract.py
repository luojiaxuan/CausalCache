"""Fail-closed validation for the restoration-v2.1 official-tool pilot."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FROZEN_RESTORATION_V2_1_PILOT_SHA256 = (
    "5b4c1e176e25ba30d84965f7c32c09594bb5a47dc3cd4f6be47d61e94cfeba03"
)
CANONICAL_CONFIG_PATH = "code/configs/causalcache_restoration_v2_1_pilot.json"
PROTOCOL_ID = "causalcache_restoration_v2_1_official_tool_interface"
PREFLIGHT_STATUS = "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

PILOT_SOURCE_IDS = (
    "0119685762769531",
    "0126163828013448",
    "0106443124024468",
    "0141544666483837",
    "0035782072567389",
)
PILOT_DECISION_STEPS = (4, 5, 6)
CANONICAL_ACTIONS = (
    "click",
    "long_press",
    "swipe",
    "type",
    "system_button",
    "open",
    "wait",
    "answer",
    "terminate",
)
PILOT_PROJECTION_SCHEMA = (
    "pilot_index",
    "source_state_index",
    "role",
    "trajectory_id",
    "decision_step_id",
    "state_id",
    "candidate_event_step_ids",
)
PILOT_PROJECTION_SHA256 = (
    "f29b787f3d0b1e9436f20d2a3c782c8e27d3617c063d2326ac9ced31c5d45956"
)
FULL_45_PROJECTION_SHA256 = (
    "65e7085f01bc26e425bab7f5148c6729bc8c6d0a62802ebc5e8dfeaed6439249"
)
RUNTIME_SOURCE_SHA256 = (
    "a34b4417c23c085fb6a01e29dcc6398a00804c07eff4945812f9ba4688de6ba5"
)
CANONICAL_TOOL_SCHEMA_SHA256 = (
    "8f92f2fe5eeda45af1e41852f364d2fa4738fbb502eb4fb80f6ef1025691f6c7"
)
CHAT_TEMPLATE_FILE_SHA256 = (
    "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce"
)
CHAT_TEMPLATE_TEXT_SHA256 = (
    "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
)
ASSISTANT_PREFIX_TOKEN_IDS = (151644, 77091, 198)
TOOL_CALL_OPEN_TOKEN_ID = 151657
TOOL_CALL_CLOSE_TOKEN_ID = 151658
STANDARD_EOS_TOKEN_IDS = (151645, 151643)
PAD_TOKEN_ID = 151643
OFFICIAL_TOOLS_INJECTED_PROMPT_COUNT = 90
IMAGE_COUNT_DISTRIBUTION = {"1": 45, "3": 15, "4": 15, "5": 15}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def _equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must equal {expected!r}; got {actual!r}")


def _false(value: Any, name: str) -> None:
    if value is not False:
        raise ValueError(f"{name} must be false")


def _true(value: Any, name: str) -> None:
    if value is not True:
        raise ValueError(f"{name} must be true")


def _resolve_repository_file(
    repository_root: Path,
    record: Any,
    *,
    name: str,
) -> Path:
    value = _object(record, name)
    _exact_keys(value, {"path", "sha256"}, name)
    relative = value["path"]
    digest = value["sha256"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{name}.path must be repository-relative")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name}.sha256 is invalid")
    root = repository_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{name}.path is missing or escapes the repository")
    if _sha256_file(path) != digest:
        raise ValueError(f"{name} source SHA256 drifted")
    return path


def _selection_pilot_projection(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    roles = _object(selection.get("roles"), "selection.roles")
    development = _object(roles.get("v2_development"), "selection.v2_development")
    trajectories = development.get("trajectories")
    states = development.get("states")
    if not isinstance(trajectories, list) or not isinstance(states, list):
        raise ValueError("selection v2_development trajectories/states are invalid")
    source_ids = tuple(
        record.get("source_id")
        for record in trajectories
        if isinstance(record, Mapping)
    )
    _equal(source_ids, PILOT_SOURCE_IDS, "selection v2_development source IDs")
    if len(states) != 15 or any(not isinstance(state, Mapping) for state in states):
        raise ValueError("selection v2_development must contain exactly 15 states")

    projection: list[dict[str, Any]] = []
    for pilot_index, state in enumerate(states):
        source_id = state.get("source_id")
        decision_step_id = state.get("decision_step_id")
        expected_source = PILOT_SOURCE_IDS[pilot_index // 3]
        expected_step = PILOT_DECISION_STEPS[pilot_index % 3]
        _equal(source_id, expected_source, "pilot state source order")
        _equal(decision_step_id, expected_step, "pilot decision-step order")
        candidate_ids = state.get("candidate_event_step_ids")
        if candidate_ids != list(range(1, expected_step - 1)):
            raise ValueError("pilot candidate event IDs drifted")
        expected_state_id = f"{source_id}:decision_step:{expected_step:03d}"
        _equal(state.get("state_id"), expected_state_id, "pilot state ID")
        projection.append(
            {
                "pilot_index": pilot_index,
                "source_state_index": pilot_index + 30,
                "role": "v2_development",
                "trajectory_id": source_id,
                "decision_step_id": decision_step_id,
                "state_id": expected_state_id,
                "candidate_event_step_ids": candidate_ids,
            }
        )
    return projection


def _selection_full_45_projection(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    roles = _object(selection.get("roles"), "selection.roles")
    result: list[dict[str, Any]] = []
    for role in ("v2_label_train", "v2_development"):
        role_record = _object(roles.get(role), f"selection.{role}")
        states = role_record.get("states")
        if not isinstance(states, list) or any(
            not isinstance(state, Mapping) for state in states
        ):
            raise ValueError(f"selection {role} states are invalid")
        for state in states:
            result.append(
                {
                    "index": len(result),
                    "role": role,
                    "trajectory_id": state.get("source_id"),
                    "decision_step_id": state.get("decision_step_id"),
                    "state_id": state.get("state_id"),
                    "candidate_event_step_ids": state.get(
                        "candidate_event_step_ids"
                    ),
                }
            )
    if len(result) != 45:
        raise ValueError("selection screening denominator must contain exactly 45 states")
    return result


def _validate_parent_results(data: Mapping[str, Any], repository_root: Path) -> None:
    change = _object(data["change_control"], "change_control")
    parent_contract_path = _resolve_repository_file(
        repository_root,
        change["parent_scientific_contract"],
        name="change_control.parent_scientific_contract",
    )
    _equal(
        parent_contract_path.relative_to(repository_root.resolve()).as_posix(),
        "code/configs/causalcache_restoration_v2.json",
        "parent contract path",
    )

    substrate_record = _object(
        change["parent_substrate_result"], "change_control.parent_substrate_result"
    )
    substrate_path = _resolve_repository_file(
        repository_root,
        {key: substrate_record[key] for key in ("path", "sha256")},
        name="change_control.parent_substrate_result.source",
    )
    substrate = _load_json_object(substrate_path)
    _equal(substrate.get("outcome"), "NO_GO_V2_SUBSTRATE", "parent outcome")
    metrics = _object(substrate.get("metrics"), "parent substrate metrics")
    _equal(metrics.get("parse_success_count"), 0, "parent strict parse count")
    _equal(
        substrate_record.get("outcome_must_remain"),
        "NO_GO_V2_SUBSTRATE",
        "parent outcome declaration",
    )
    _equal(
        substrate_record.get("strict_parse_success_count_must_remain"),
        0,
        "parent strict parse declaration",
    )

    replay_record = _object(
        change["parent_parser_replay"], "change_control.parent_parser_replay"
    )
    replay_path = _resolve_repository_file(
        repository_root,
        {key: replay_record[key] for key in ("path", "sha256")},
        name="change_control.parent_parser_replay.source",
    )
    replay = _load_json_object(replay_path)
    _equal(replay.get("outcome"), "NO_GO_ADAPTER_ONLY", "parser replay outcome")
    _equal(
        replay_record.get("outcome_must_remain"),
        "NO_GO_ADAPTER_ONLY",
        "parser replay declaration",
    )


def validate_restoration_v2_1_pilot_contract(
    data: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    expected_top_level = {
        "schema_version",
        "protocol_id",
        "preregistration_status",
        "change_control",
        "primary_policy",
        "policy_interface",
        "data",
        "processor_preflight",
        "pilot_execution",
        "pilot_gate",
        "promotion",
    }
    _exact_keys(data, expected_top_level, "contract")
    _equal(data["schema_version"], "0.1.0", "schema_version")
    _equal(data["protocol_id"], PROTOCOL_ID, "protocol_id")
    _equal(
        data["preregistration_status"],
        "policy_interface_reset_frozen_before_any_v2_1_policy_output",
        "preregistration_status",
    )

    change = _object(data["change_control"], "change_control")
    _equal(
        change.get("change_class"),
        "new_policy_interface_protocol_not_format_only_adapter",
        "change_control.change_class",
    )
    _equal(change.get("parent_protocol_id"), "causalcache_restoration_v2", "parent protocol")
    for field in (
        "v2_results_may_be_relabelled",
        "v2_compatibility_outputs_may_count_as_v2_1_outputs",
        "scientific_fields_may_change_after_first_v2_1_policy_output",
    ):
        _false(change.get(field), f"change_control.{field}")
    for field in (
        "v2_1_was_designed_after_v2_development_output",
        "pilot_is_interface_engineering_not_confirmatory_evidence",
    ):
        _true(change.get(field), f"change_control.{field}")
    _validate_parent_results(data, repository_root)

    policy = _object(data["primary_policy"], "primary_policy")
    _equal(policy.get("repo"), "mPLUG/GUI-Owl-1.5-8B-Instruct", "policy repo")
    _equal(
        policy.get("revision"),
        "06d5faecff74840bab2be2425e9c42667a5d04fc",
        "policy revision",
    )
    _true(policy.get("frozen"), "primary_policy.frozen")
    _equal(policy.get("dtype"), "torch.bfloat16", "policy dtype")
    _equal(policy.get("device_count"), 1, "policy device count")
    snapshot_manifest_path = _resolve_repository_file(
        repository_root, policy.get("snapshot_manifest"), name="policy snapshot manifest"
    )
    generation = _object(policy.get("generation"), "primary_policy.generation")
    expected_generation = {
        "batch_size": 1,
        "planned_generations_per_state": 1,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "max_new_tokens": 256,
        "eos_token_id": 151658,
        "eos_token_text": "</tool_call>",
        "pad_token_id": 151643,
        "pad_token_text": "<|endoftext|>",
        "suppress_token_ids": [151643, 151645],
        "suppress_token_texts": ["<|endoftext|>", "<|im_end|>"],
        "automatic_retry": False,
        "automatic_oom_fallback": False,
        "fallback_generation_interface": None,
    }
    _equal(dict(generation), expected_generation, "primary_policy.generation")

    interface = _object(data["policy_interface"], "policy_interface")
    _resolve_repository_file(
        repository_root, interface.get("source"), name="policy_interface.source"
    )
    expected_interface_scalars = {
        "contract_id": "gui_owl_v2_1_official_tool_interface",
        "chat_template_mode": "official_tools_argument",
        "tools_argument_count": 1,
        "tool_name": "mobile_use",
        "manual_tools_xml_in_system_prompt": False,
        "action_line_carrier_present": False,
        "generation_and_teacher_prompt_identical": True,
        "output_envelope": "exact_single_complete_official_tool_call_only",
        "whole_output_pattern": "<tool_call>\\n{strict_json}\\n</tool_call>",
        "leading_or_trailing_whitespace_allowed": False,
        "surrounding_prose_allowed": False,
        "action_line_allowed": False,
        "observation_allowed": False,
        "second_json_allowed": False,
        "second_tool_call_allowed": False,
        "model_emitted_closer_required": True,
        "host_appended_closer_allowed": False,
        "syntax_completion_allowed": False,
        "compatibility_parser_fallback_allowed": False,
        "parent_v2_parser_fallback_allowed": False,
        "model_action_aliases_allowed": False,
        "androidworld_bridge_required": True,
    }
    for field, expected in expected_interface_scalars.items():
        _equal(interface.get(field), expected, f"policy_interface.{field}")
    _equal(tuple(interface.get("actions", ())), CANONICAL_ACTIONS, "interface actions")
    wrapper = _object(interface.get("wrapper"), "policy_interface.wrapper")
    _equal(
        dict(wrapper),
        {
            "required_top_level_keys": ["name", "arguments"],
            "name": "mobile_use",
            "arguments_must_be_object": True,
            "duplicate_json_keys_allowed": False,
            "nonfinite_json_numbers_allowed": False,
            "text_arguments_must_already_be_unicode_nfkc": True,
            "non_nfkc_text_arguments_rejected_by_parser": True,
        },
        "policy_interface.wrapper",
    )
    teacher = _object(interface.get("teacher_target"), "policy_interface.teacher_target")
    _equal(
        dict(teacher),
        {
            "serialization": "official_tool_call_json_with_comma_space_and_colon_space",
            "action_carrier": None,
            "distance_span": "first_token_of_<tool_call>_through_model_emitted_</tool_call>_inclusive",
            "suppressed_standard_eos_token_ids": [151645, 151643],
            "generation_suppression_semantics": (
                "negative_infinity_via_transformers_suppress_tokens"
            ),
            "teacher_suppression_semantics": "torch_finfo_bfloat16_min_finite",
            "teacher_mask_application": (
                "same_mask_on_every_reference_and_candidate_action_path_position_"
                "before_float32_log_softmax"
            ),
            "target_disjoint_from_suppressed_standard_eos_required": True,
        },
        "policy_interface.teacher_target",
    )
    identity = _object(
        interface.get("immutable_identity"), "policy_interface.immutable_identity"
    )
    _exact_keys(
        identity,
        {
            "official_chat_template",
            "system_prompt_utf8_sha256",
            "final_user_instruction_utf8_sha256",
            "canonical_tool_schema_sha256",
            "parser_source",
            "teacher_serializer_source",
            "runtime_source",
        },
        "policy_interface.immutable_identity",
    )
    chat_template = _object(
        identity["official_chat_template"],
        "policy_interface.immutable_identity.official_chat_template",
    )
    _equal(
        dict(chat_template),
        {
            "snapshot_path": "chat_template.json",
            "sha256": "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
        },
        "official chat-template identity",
    )
    snapshot_manifest = _load_json_object(snapshot_manifest_path)
    snapshot_files = snapshot_manifest.get("files")
    if not isinstance(snapshot_files, list):
        raise ValueError("policy snapshot manifest files are invalid")
    chat_records = [
        record
        for record in snapshot_files
        if isinstance(record, Mapping) and record.get("path") == "chat_template.json"
    ]
    if len(chat_records) != 1 or chat_records[0].get("sha256") != chat_template["sha256"]:
        raise ValueError("official chat-template differs from the frozen snapshot")
    parser_path = _resolve_repository_file(
        repository_root,
        identity["parser_source"],
        name="policy_interface.immutable_identity.parser_source",
    )
    teacher_path = _resolve_repository_file(
        repository_root,
        identity["teacher_serializer_source"],
        name="policy_interface.immutable_identity.teacher_serializer_source",
    )
    _resolve_repository_file(
        repository_root,
        identity["runtime_source"],
        name="policy_interface.immutable_identity.runtime_source",
    )
    if parser_path != teacher_path:
        raise ValueError("parser and teacher serializer source identity diverged")
    from causalcache.policy.gui_owl_v2_1 import (
        GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
        GUI_OWL_V2_1_MOBILE_USE_TOOL,
        GUI_OWL_V2_1_SYSTEM_PROMPT,
    )

    _equal(
        identity["system_prompt_utf8_sha256"],
        _sha256_bytes(GUI_OWL_V2_1_SYSTEM_PROMPT.encode("utf-8")),
        "system prompt SHA256",
    )
    _equal(
        identity["final_user_instruction_utf8_sha256"],
        _sha256_bytes(GUI_OWL_V2_1_FINAL_USER_INSTRUCTION.encode("utf-8")),
        "final user instruction SHA256",
    )
    _equal(
        identity["canonical_tool_schema_sha256"],
        _canonical_json_sha256(GUI_OWL_V2_1_MOBILE_USE_TOOL),
        "canonical tool schema SHA256",
    )

    data_section = _object(data["data"], "data")
    selection_path = _resolve_repository_file(
        repository_root,
        data_section.get("selection_manifest"),
        name="data.selection_manifest",
    )
    selection = _load_json_object(selection_path)
    projection = _selection_pilot_projection(selection)
    _equal(data_section.get("pilot_role"), "v2_development", "pilot role")
    _equal(tuple(data_section.get("pilot_source_ids", ())), PILOT_SOURCE_IDS, "pilot source IDs")
    _equal(tuple(data_section.get("pilot_decision_step_ids", ())), PILOT_DECISION_STEPS, "pilot decision steps")
    _equal(data_section.get("pilot_state_count"), 15, "pilot state count")
    _equal(data_section.get("source_state_indices"), list(range(30, 45)), "pilot source indexes")
    _equal(
        data_section.get("pilot_state_ids"),
        [record["state_id"] for record in projection],
        "pilot state IDs",
    )
    _equal(
        tuple(data_section.get("pilot_state_projection_schema", ())),
        PILOT_PROJECTION_SCHEMA,
        "pilot projection schema",
    )
    _equal(_canonical_json_sha256(projection), PILOT_PROJECTION_SHA256, "derived pilot projection SHA256")
    _equal(data_section.get("pilot_state_projection_sha256"), PILOT_PROJECTION_SHA256, "pilot projection SHA256")
    for field in ("full_history_only",):
        _true(data_section.get(field), f"data.{field}")
    for field in (
        "confirm_policy_output_allowed",
        "label_train_policy_output_allowed",
        "top_up_allowed",
        "state_filtering_after_output_allowed",
    ):
        _false(data_section.get(field), f"data.{field}")
    _equal(data_section.get("confirm_role"), "v2_confirm_primary", "confirm role")

    preflight = _object(data["processor_preflight"], "processor_preflight")
    expected_preflight = {
        "required_before_any_v2_1_policy_generation": True,
        "required_success_status": PREFLIGHT_STATUS,
        "runtime_source_sha256": RUNTIME_SOURCE_SHA256,
        "canonical_tool_schema_sha256": CANONICAL_TOOL_SCHEMA_SHA256,
        "chat_template_file_sha256": CHAT_TEMPLATE_FILE_SHA256,
        "chat_template_text_sha256": CHAT_TEMPLATE_TEXT_SHA256,
        "assistant_prefix_token_ids": list(ASSISTANT_PREFIX_TOKEN_IDS),
        "tool_call_open_token_id": TOOL_CALL_OPEN_TOKEN_ID,
        "tool_call_close_token_id": TOOL_CALL_CLOSE_TOKEN_ID,
        "standard_eos_token_ids": list(STANDARD_EOS_TOKEN_IDS),
        "pad_token_id": PAD_TOKEN_ID,
        "roles": ["v2_label_train", "v2_development"],
        "state_count": 45,
        "fidelities": ["reference", "summary_only"],
        "prompt_count": 90,
        "official_tools_injected_prompt_count": OFFICIAL_TOOLS_INJECTED_PROMPT_COUNT,
        "reference_prompt_count": 45,
        "summary_only_prompt_count": 45,
        "image_count_distribution": IMAGE_COUNT_DISTRIBUTION,
        "teacher_boundary_validated": True,
        "full_45_state_projection_sha256": FULL_45_PROJECTION_SHA256,
        "reserved_generation_tokens": 256,
        "verified_context_limit_required": True,
        "context_overflow_count": 0,
        "policy_model_loaded": False,
        "policy_forward_executed": False,
        "policy_generation_executed": False,
        "confirm_accessed": False,
        "restoration_output_generated": False,
        "must_bind_contract_sha256": True,
        "must_bind_selection_manifest_sha256": True,
        "must_bind_policy_interface_source_sha256": True,
        "failure_outcome": "INVALID_BEFORE_V2_1_POLICY_GENERATION",
    }
    _equal(dict(preflight), expected_preflight, "processor_preflight")

    execution = _object(data["pilot_execution"], "pilot_execution")
    expected_execution = {
        "fixed_state_denominator": 15,
        "planned_generation_call_count": 15,
        "full_history_reference_generation_only": True,
        "state_attempt_marker_written_before_generation": True,
        "resume_may_only_reuse_terminal_state_records": True,
        "incomplete_attempt_may_be_retried": False,
        "completed_state_may_be_regenerated": False,
        "state_retry_count": 0,
        "top_up_count": 0,
        "teacher_forward_count": 0,
        "kl_measurement_count": 0,
        "restoration_label_count": 0,
        "expert_action_read_count": 0,
        "confirm_state_generation_count": 0,
        "label_train_state_generation_count": 0,
        "sample_mutation_allowed": False,
    }
    _equal(dict(execution), expected_execution, "pilot_execution")

    gate = _object(data["pilot_gate"], "pilot_gate")
    expected_gate = {
        "fixed_state_denominator": 15,
        "required_exact_whole_output_parse_count": 15,
        "required_model_emitted_closer_count": 15,
        "required_androidworld_bridge_count": 15,
        "maximum_max_token_truncation_count": 0,
        "maximum_surrounding_prose_count": 0,
        "maximum_action_line_count": 0,
        "maximum_observation_count": 0,
        "maximum_second_json_count": 0,
        "maximum_second_tool_call_count": 0,
        "maximum_retry_count": 0,
        "maximum_top_up_count": 0,
        "pass_outcome": "PASS_V2_1_INTERFACE_PILOT",
        "fail_outcome": "NO_GO_V2_1_INTERFACE_PILOT",
        "invalid_outcome": "INVALID_V2_1_PILOT",
    }
    _equal(dict(gate), expected_gate, "pilot_gate")

    promotion = _object(data["promotion"], "promotion")
    _true(promotion.get("pilot_pass_does_not_authorize_confirm"), "promotion confirm lock")
    _equal(
        promotion.get("pilot_pass_authorizes_only"),
        "unchanged_interface_fixed_45_state_substrate_screening",
        "promotion authorization",
    )
    _equal(
        _canonical_json_sha256(_selection_full_45_projection(selection)),
        FULL_45_PROJECTION_SHA256,
        "derived full-45 projection SHA256",
    )
    _equal(
        promotion.get("full_45_state_projection_sha256"),
        FULL_45_PROJECTION_SHA256,
        "full-45 projection SHA256",
    )
    _false(
        promotion.get(
            "interface_prompt_generation_parser_teacher_and_gate_sources_may_change_before_full_45"
        ),
        "promotion source changes",
    )
    _true(
        promotion.get("confirm_remains_locked_until_full_45_substrate_gate_passes"),
        "promotion confirm lock until full-45",
    )
    _true(
        promotion.get("same_parent_v2_restoration_gate_thresholds_required_after_substrate_pass"),
        "promotion restoration thresholds",
    )
    return {
        "pilot_state_count": 15,
        "pilot_state_projection_sha256": PILOT_PROJECTION_SHA256,
        "processor_preflight_prompt_count": 90,
        "planned_generation_call_count": 15,
    }


def validate_restoration_v2_1_processor_preflight(
    value: Mapping[str, Any],
    *,
    contract_sha256: str,
    selection_manifest_sha256: str,
    policy_interface_source_sha256: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "contract_sha256",
        "selection_manifest_sha256",
        "policy_interface_source_sha256",
        "runtime_source_sha256",
        "canonical_tool_schema_sha256",
        "chat_template_file_sha256",
        "chat_template_text_sha256",
        "assistant_prefix_token_ids",
        "tool_call_open_token_id",
        "tool_call_close_token_id",
        "standard_eos_token_ids",
        "pad_token_id",
        "state_count",
        "prompt_count",
        "official_tools_injected_prompt_count",
        "fidelity_counts",
        "role_state_counts",
        "image_count_distribution",
        "teacher_boundary_validated",
        "full_45_state_projection_sha256",
        "reserved_generation_tokens",
        "maximum_context_tokens",
        "context_overflow_count",
        "shape_records_sha256",
        "policy_model_loaded",
        "policy_forward_executed",
        "policy_generation_executed",
        "confirm_accessed",
        "restoration_output_generated",
    }
    _exact_keys(value, expected_keys, "processor preflight")
    _equal(value["schema_version"], "0.1.0", "processor preflight schema")
    _equal(value["protocol_id"], PROTOCOL_ID, "processor preflight protocol")
    _equal(value["status"], PREFLIGHT_STATUS, "processor preflight status")
    _equal(value["contract_sha256"], contract_sha256, "processor preflight contract SHA256")
    _equal(
        value["selection_manifest_sha256"],
        selection_manifest_sha256,
        "processor preflight selection SHA256",
    )
    _equal(
        value["policy_interface_source_sha256"],
        policy_interface_source_sha256,
        "processor preflight interface SHA256",
    )
    _equal(
        value["runtime_source_sha256"],
        RUNTIME_SOURCE_SHA256,
        "processor preflight runtime SHA256",
    )
    _equal(
        value["canonical_tool_schema_sha256"],
        CANONICAL_TOOL_SCHEMA_SHA256,
        "processor preflight tool-schema SHA256",
    )
    _equal(
        value["chat_template_file_sha256"],
        CHAT_TEMPLATE_FILE_SHA256,
        "processor preflight chat-template file SHA256",
    )
    _equal(
        value["chat_template_text_sha256"],
        CHAT_TEMPLATE_TEXT_SHA256,
        "processor preflight chat-template text SHA256",
    )
    _equal(
        value["assistant_prefix_token_ids"],
        list(ASSISTANT_PREFIX_TOKEN_IDS),
        "processor preflight assistant-prefix token IDs",
    )
    _equal(
        value["tool_call_open_token_id"],
        TOOL_CALL_OPEN_TOKEN_ID,
        "processor preflight tool-call opener token ID",
    )
    _equal(
        value["tool_call_close_token_id"],
        TOOL_CALL_CLOSE_TOKEN_ID,
        "processor preflight tool-call closer token ID",
    )
    _equal(
        value["standard_eos_token_ids"],
        list(STANDARD_EOS_TOKEN_IDS),
        "processor preflight standard EOS token IDs",
    )
    _equal(
        value["pad_token_id"],
        PAD_TOKEN_ID,
        "processor preflight pad token ID",
    )
    _equal(value["state_count"], 45, "processor preflight state count")
    _equal(value["prompt_count"], 90, "processor preflight prompt count")
    _equal(
        value["official_tools_injected_prompt_count"],
        OFFICIAL_TOOLS_INJECTED_PROMPT_COUNT,
        "processor preflight official-tools prompt count",
    )
    _equal(
        value["fidelity_counts"],
        {"reference": 45, "summary_only": 45},
        "processor preflight fidelity counts",
    )
    _equal(
        value["role_state_counts"],
        {"v2_label_train": 30, "v2_development": 15},
        "processor preflight role counts",
    )
    _equal(
        value["image_count_distribution"],
        IMAGE_COUNT_DISTRIBUTION,
        "processor preflight image-count distribution",
    )
    _true(
        value["teacher_boundary_validated"],
        "processor preflight teacher boundary",
    )
    _equal(
        value["full_45_state_projection_sha256"],
        FULL_45_PROJECTION_SHA256,
        "processor preflight full-45 projection SHA256",
    )
    _equal(value["reserved_generation_tokens"], 256, "preflight reservation")
    if type(value["maximum_context_tokens"]) is not int or value["maximum_context_tokens"] <= 256:
        raise ValueError("processor preflight context limit is invalid")
    _equal(value["context_overflow_count"], 0, "processor context overflow count")
    shape_hash = value["shape_records_sha256"]
    if not isinstance(shape_hash, str) or SHA256_PATTERN.fullmatch(shape_hash) is None:
        raise ValueError("processor preflight shape-record SHA256 is invalid")
    for field in (
        "policy_model_loaded",
        "policy_forward_executed",
        "policy_generation_executed",
        "confirm_accessed",
        "restoration_output_generated",
    ):
        _false(value[field], f"processor preflight {field}")
    return {
        "prompt_count": 90,
        "shape_records_sha256": shape_hash,
        "valid": True,
    }


@dataclass(frozen=True)
class RestorationV21PilotContract:
    data: Mapping[str, Any]
    source_sha256: str
    validation: Mapping[str, Any]

    @property
    def protocol_id(self) -> str:
        return str(self.data["protocol_id"])

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "RestorationV21PilotContract":
        root = Path(repository_root).resolve()
        supplied = Path(path)
        resolved = (
            supplied.resolve()
            if supplied.is_absolute()
            else (root / supplied).resolve()
        )
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file():
            raise ValueError(f"v2.1 pilot contract must be {CANONICAL_CONFIG_PATH}")
        source_sha256 = _sha256_file(resolved)
        if source_sha256 != FROZEN_RESTORATION_V2_1_PILOT_SHA256:
            raise ValueError("restoration-v2.1 pilot contract SHA256 mismatch")
        data = _load_json_object(resolved)
        validation = validate_restoration_v2_1_pilot_contract(
            data,
            repository_root=root,
        )
        return cls(data=data, source_sha256=source_sha256, validation=validation)
