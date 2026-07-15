"""Independent reduction and validation for the v2.1 processor-only audit."""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import re
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_MOBILE_USE_TOOL,
    GUI_OWL_V2_1_PROTOCOL_ID,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    canonical_json_sha256,
    parse_gui_owl_v2_1_output,
    serialize_gui_owl_v2_1_teacher_target,
)
from causalcache.restoration_v2_1_contract import (
    ASSISTANT_PREFIX_TOKEN_IDS,
    CANONICAL_TOOL_SCHEMA_SHA256,
    CHAT_TEMPLATE_FILE_SHA256,
    CHAT_TEMPLATE_TEXT_SHA256,
    FULL_45_PROJECTION_SHA256,
    IMAGE_COUNT_DISTRIBUTION,
    PAD_TOKEN_ID,
    PREFLIGHT_STATUS,
    RUNTIME_SOURCE_SHA256,
    STANDARD_EOS_TOKEN_IDS,
    TOOL_CALL_CLOSE_TOKEN_ID,
    TOOL_CALL_OPEN_TOKEN_ID,
    validate_restoration_v2_1_processor_preflight,
)


SCHEMA_VERSION = "0.1.0"
EVIDENCE_TYPE = "gui_owl_v2_1_processor_only_preflight"
EXPECTED_STATE_COUNT = 45
EXPECTED_PROMPT_COUNT = 90
EXPECTED_MAXIMUM_CONTEXT_TOKENS = 32_768
RESERVED_GENERATION_TOKENS = 256
EXPECTED_PATCH_VECTOR_SIZE = 3 * 2 * 16 * 16
CANONICAL_REMOTE_URL = "https://github.com/luojiaxuan/CausalCache.git"
CANONICAL_EVIDENCE_ARTIFACT_PATH = (
    "data/results/restoration_v2_1_processor_preflight/artifact.json"
)
CANONICAL_EVIDENCE_HF_REPO = (
    "gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile"
)
CANONICAL_EVIDENCE_HF_TAG = "v2.1-processor-preflight-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
EXPECTED_INPUT_PATHS = {
    "v2_contract": "code/configs/causalcache_restoration_v2.json",
    "v2_1_contract": "code/configs/causalcache_restoration_v2_1_pilot.json",
    "selection_manifest": "data/manifests/restoration_v2_selection.json",
    "snapshot_manifest": "code/configs/gui_owl_1_5_8b_snapshot.json",
    "ocr_backend_config": "code/configs/restoration_v2_ocr_backend.json",
    "artifact_binding_manifest": "data/manifests/restoration_v2_derived_artifact.json",
}
AUDITED_SOURCE_PATHS = (
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/causalcache/data/guiodyssey.py",
    "code/causalcache/data/guiodyssey_independent.py",
    "code/causalcache/data/restoration_v2_screening.py",
    "code/causalcache/data/restoration_v2_selection.py",
    "code/causalcache/data/restoration_v2_1_processor_inputs.py",
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_1.py",
    "code/causalcache/policy/gui_owl_v2_1_runtime.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/restoration_v2_contract.py",
    "code/causalcache/restoration_v2_1_contract.py",
    "code/causalcache/restoration_v2_1_processor_audit.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/schema.py",
    "code/scripts/audit_gui_owl_v2_1_processor.py",
    "code/scripts/package_restoration_v2_1_processor_evidence.py",
)
EXPECTED_TENSOR_DTYPES = {
    "attention_mask": "torch.int64",
    "image_grid_thw": "torch.int64",
    "input_ids": "torch.int64",
    "mm_token_type_ids": "torch.int64",
    "pixel_values": "torch.float32",
}
REQUIRED_TENSORS = frozenset(
    {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
)
ALLOWED_TENSORS = REQUIRED_TENSORS.union({"mm_token_type_ids"})
OFFICIAL_TOOL_JSON = json.dumps(
    GUI_OWL_V2_1_MOBILE_USE_TOOL,
    ensure_ascii=False,
    sort_keys=False,
    separators=(", ", ": "),
    allow_nan=False,
)
OFFICIAL_TOOL_BLOCK = f"<tools>\n{OFFICIAL_TOOL_JSON}\n</tools>"


def _golden_actions() -> tuple[GUIOwlV2Action, ...]:
    return (
        GUIOwlV2Action(action="click", coordinate=(0, 999)),
        GUIOwlV2Action(action="long_press", coordinate=(500, 500)),
        GUIOwlV2Action(
            action="swipe",
            coordinate=(1, 2),
            coordinate2=(998, 997),
        ),
        GUIOwlV2Action(action="type", text="Café"),
        GUIOwlV2Action(action="system_button", button="Back"),
        GUIOwlV2Action(action="open", text="设置"),
        GUIOwlV2Action(action="wait"),
        GUIOwlV2Action(action="answer", text="完成 ✅"),
        GUIOwlV2Action(action="terminate", status="success"),
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    return strict_json_object_bytes(path.read_bytes(), label=label)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value,
        Sequence,
    ):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def screening_state_projection(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Derive the exact 45-state order directly from the selection manifest."""
    roles = _mapping(selection.get("roles"), "selection.roles")
    projection: list[dict[str, Any]] = []
    for role, expected_count in (("v2_label_train", 30), ("v2_development", 15)):
        role_record = _mapping(roles.get(role), f"selection.{role}")
        states = _sequence(role_record.get("states"), f"selection.{role}.states")
        if len(states) != expected_count:
            raise ValueError(f"selection {role} state count drifted")
        for state in states:
            record = _mapping(state, f"selection.{role}.state")
            source_id = record.get("source_id")
            decision_step_id = record.get("decision_step_id")
            candidate_ids = record.get("candidate_event_step_ids")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("selection source ID is invalid")
            if decision_step_id not in (4, 5, 6):
                raise ValueError("selection decision step drifted")
            expected_candidates = list(range(1, decision_step_id - 1))
            if candidate_ids != expected_candidates:
                raise ValueError("selection candidate-event IDs drifted")
            state_id = f"{source_id}:decision_step:{decision_step_id:03d}"
            if record.get("state_id") != state_id:
                raise ValueError("selection state ID drifted")
            projection.append(
                {
                    "index": len(projection),
                    "role": role,
                    "trajectory_id": source_id,
                    "decision_step_id": decision_step_id,
                    "state_id": state_id,
                    "candidate_event_step_ids": expected_candidates,
                }
            )
    if len(projection) != EXPECTED_STATE_COUNT:
        raise ValueError("selection screening denominator drifted")
    if canonical_json_sha256(projection) != FULL_45_PROJECTION_SHA256:
        raise ValueError("selection full-45 projection SHA256 drifted")
    return projection


def _validate_tensor_inventory(
    record: Mapping[str, Any],
    *,
    sequence_tokens: int,
    image_count: int,
) -> tuple[list[list[int]], int]:
    inventory = _mapping(record.get("tensor_inventory"), "prompt.tensor_inventory")
    names = frozenset(inventory)
    if REQUIRED_TENSORS.difference(names) or names.difference(ALLOWED_TENSORS):
        raise ValueError("processor tensor inventory drifted")
    shapes: dict[str, list[int]] = {}
    for name in sorted(inventory):
        tensor = _mapping(inventory[name], f"prompt.tensor_inventory.{name}")
        _exact_keys(
            tensor,
            {"shape", "dtype", "device", "requires_grad"},
            f"prompt.tensor_inventory.{name}",
        )
        shape = list(_sequence(tensor["shape"], f"prompt.{name}.shape"))
        if not shape or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError(f"prompt {name} shape drifted")
        if tensor["dtype"] != EXPECTED_TENSOR_DTYPES[name]:
            raise ValueError(f"prompt {name} dtype drifted")
        if tensor["device"] != "cpu" or tensor["requires_grad"] is not False:
            raise ValueError(f"prompt {name} must be a no-grad CPU tensor")
        shapes[name] = shape
    if shapes["input_ids"] != [1, sequence_tokens]:
        raise ValueError("prompt input_ids shape drifted")
    if shapes["attention_mask"] != [1, sequence_tokens]:
        raise ValueError("prompt attention_mask shape drifted")
    if "mm_token_type_ids" in shapes and shapes["mm_token_type_ids"] != [
        1,
        sequence_tokens,
    ]:
        raise ValueError("prompt mm_token_type_ids shape drifted")

    grids_value = _sequence(record.get("image_grid_thw"), "prompt.image_grid_thw")
    grids: list[list[int]] = []
    for row in grids_value:
        values = list(_sequence(row, "prompt.image_grid_thw row"))
        if len(values) != 3 or any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("prompt image grid row drifted")
        if values[1] % 2 or values[2] % 2:
            raise ValueError("prompt image grid is incompatible with merge size 2")
        grids.append(values)
    if len(grids) != image_count or shapes["image_grid_thw"] != [image_count, 3]:
        raise ValueError("prompt image grid count drifted")
    raw_patch_count = sum(t * h * w for t, h, w in grids)
    if shapes["pixel_values"] != [raw_patch_count, EXPECTED_PATCH_VECTOR_SIZE]:
        raise ValueError("prompt pixel tensor geometry drifted")
    merged_visual_tokens = sum(t * h * w // 4 for t, h, w in grids)
    if merged_visual_tokens >= sequence_tokens:
        raise ValueError("prompt has no positive text-token contribution")
    return grids, merged_visual_tokens


def _validate_prompt_record(
    value: Any,
    *,
    expected_prompt_index: int,
    expected_state: Mapping[str, Any],
) -> dict[str, Any]:
    record = _mapping(value, f"prompt_records[{expected_prompt_index}]")
    _exact_keys(
        record,
        {
            "prompt_index",
            "state_index",
            "state_id",
            "role",
            "trajectory_id",
            "decision_step_id",
            "candidate_event_step_ids",
            "fidelity",
            "restored_event_step_ids",
            "native_system_text",
            "native_user_text_blocks",
            "rendered_prompt",
            "rendered_without_tools",
            "input_ids",
            "attention_mask",
            "image_grid_thw",
            "tensor_inventory",
        },
        f"prompt_records[{expected_prompt_index}]",
    )
    if record["prompt_index"] != expected_prompt_index:
        raise ValueError("prompt index/order drifted")
    state_index = expected_prompt_index // 2
    fidelity = ("reference", "summary_only")[expected_prompt_index % 2]
    if record["state_index"] != state_index or record["fidelity"] != fidelity:
        raise ValueError("prompt state/fidelity order drifted")
    for key in (
        "state_id",
        "role",
        "trajectory_id",
        "decision_step_id",
        "candidate_event_step_ids",
    ):
        if record[key] != expected_state[key]:
            raise ValueError(f"prompt {key} differs from selection")
    expected_restored = (
        expected_state["candidate_event_step_ids"] if fidelity == "reference" else []
    )
    if record["restored_event_step_ids"] != expected_restored:
        raise ValueError("prompt restored-event set differs from fidelity")
    if record["native_system_text"] != GUI_OWL_V2_1_SYSTEM_PROMPT:
        raise ValueError("prompt native system text drifted")
    user_text_blocks = list(
        _sequence(record["native_user_text_blocks"], "prompt.native_user_text_blocks")
    )
    if not user_text_blocks or any(not isinstance(text, str) for text in user_text_blocks):
        raise ValueError("prompt native user text blocks are invalid")
    if user_text_blocks[-1] != GUI_OWL_V2_1_FINAL_USER_INSTRUCTION:
        raise ValueError("prompt final user instruction drifted")
    native_text = "\n".join([record["native_system_text"], *user_text_blocks])
    for forbidden in ("# Tools", "<tools>", "</tools>", OFFICIAL_TOOL_JSON):
        if forbidden in native_text:
            raise ValueError("prompt contains a manual tools duplicate")

    rendered = record["rendered_prompt"]
    rendered_without = record["rendered_without_tools"]
    if not isinstance(rendered, str) or not isinstance(rendered_without, str):
        raise ValueError("prompt rendered branches must be strings")
    if rendered == rendered_without:
        raise ValueError("official tools branch did not change the rendered prompt")
    if rendered.count(OFFICIAL_TOOL_BLOCK) != 1:
        raise ValueError("rendered prompt lacks exactly one canonical tools block")
    if rendered.count("# Tools\n\n") != 1:
        raise ValueError("rendered prompt did not use the official tools branch")
    if rendered.count("<tools></tools>") != 1:
        raise ValueError("rendered prompt official tools instruction drifted")
    if rendered.count("<tools>") != 2 or rendered.count("</tools>") != 2:
        raise ValueError("rendered prompt tools envelope count drifted")
    if (
        OFFICIAL_TOOL_BLOCK in rendered_without
        or "<tools>" in rendered_without
        or "# Tools\n\n" in rendered_without
    ):
        raise ValueError("no-tools render unexpectedly contains a tools block")
    if not rendered.endswith("<|im_start|>assistant\n"):
        raise ValueError("rendered prompt lacks the official assistant prefix")

    input_ids = list(_sequence(record["input_ids"], "prompt.input_ids"))
    attention_mask = list(
        _sequence(record["attention_mask"], "prompt.attention_mask")
    )
    if not input_ids or any(type(token) is not int or token < 0 for token in input_ids):
        raise ValueError("prompt input IDs are invalid")
    if attention_mask != [1] * len(input_ids):
        raise ValueError("prompt attention mask contains padding or drift")
    prefix = list(ASSISTANT_PREFIX_TOKEN_IDS)
    if input_ids[-len(prefix) :] != prefix:
        raise ValueError("prompt input IDs do not end at the assistant boundary")
    sequence_tokens = len(input_ids)
    decision_step = int(expected_state["decision_step_id"])
    image_count = 1 if fidelity == "summary_only" else decision_step - 1
    grids, merged_visual_tokens = _validate_tensor_inventory(
        record,
        sequence_tokens=sequence_tokens,
        image_count=image_count,
    )
    overflow = int(
        sequence_tokens + RESERVED_GENERATION_TOKENS
        > EXPECTED_MAXIMUM_CONTEXT_TOKENS
    )
    return {
        "prompt_index": expected_prompt_index,
        "state_index": state_index,
        "state_id": expected_state["state_id"],
        "role": expected_state["role"],
        "decision_step_id": decision_step,
        "fidelity": fidelity,
        "restored_event_step_ids": list(expected_restored),
        "image_count": image_count,
        "input_ids_shape": [1, sequence_tokens],
        "input_ids_sha256": sha256_bytes(canonical_json_bytes(input_ids)),
        "sequence_tokens": sequence_tokens,
        "merged_visual_tokens": merged_visual_tokens,
        "image_grid_thw": grids,
        "context_overflow": overflow,
    }


def _validate_golden_record(value: Any, expected_action: GUIOwlV2Action) -> dict[str, Any]:
    record = _mapping(value, f"teacher_golden.{expected_action.action}")
    _exact_keys(
        record,
        {
            "action",
            "assistant_content",
            "assistant_tool_calls",
            "target_text",
            "rendered_conversation",
            "assistant_prefix_token_ids",
            "target_token_ids",
            "joint_token_ids",
        },
        f"teacher_golden.{expected_action.action}",
    )
    if record["action"] != expected_action.action or record["assistant_content"] != "":
        raise ValueError("teacher golden action/content drifted")
    expected_tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "mobile_use",
                "arguments": expected_action.arguments(),
            },
        }
    ]
    if record["assistant_tool_calls"] != expected_tool_calls:
        raise ValueError("teacher golden is not the canonical assistant tool_calls form")
    target = serialize_gui_owl_v2_1_teacher_target(expected_action)
    if record["target_text"] != target:
        raise ValueError("teacher golden target text drifted")
    if parse_gui_owl_v2_1_output(target).canonical_action != expected_action:
        raise ValueError("teacher golden target failed strict parser round trip")
    rendered = record["rendered_conversation"]
    expected_suffix = f"<|im_start|>assistant\n{target}<|im_end|>\n"
    if not isinstance(rendered, str) or not rendered.endswith(expected_suffix):
        raise ValueError("official assistant tool_calls render drifted")

    prefix_ids = list(
        _sequence(record["assistant_prefix_token_ids"], "teacher prefix IDs")
    )
    target_ids = list(_sequence(record["target_token_ids"], "teacher target IDs"))
    joint_ids = list(_sequence(record["joint_token_ids"], "teacher joint IDs"))
    for label, values in (
        ("prefix", prefix_ids),
        ("target", target_ids),
        ("joint", joint_ids),
    ):
        if not values or any(type(token) is not int or token < 0 for token in values):
            raise ValueError(f"teacher {label} token IDs are invalid")
    if prefix_ids != list(ASSISTANT_PREFIX_TOKEN_IDS):
        raise ValueError("teacher assistant prefix token IDs drifted")
    if joint_ids != [*prefix_ids, *target_ids]:
        raise ValueError("tokenizer merges across assistant/tool-call boundary")
    if target_ids[0] != TOOL_CALL_OPEN_TOKEN_ID:
        raise ValueError("teacher target lacks the singleton opener token")
    if target_ids[-1] != TOOL_CALL_CLOSE_TOKEN_ID:
        raise ValueError("teacher target lacks the singleton closer token")
    if set(target_ids).intersection(STANDARD_EOS_TOKEN_IDS):
        raise ValueError("teacher target overlaps suppressed standard EOS tokens")
    return {
        "action": expected_action.action,
        "target_text_sha256": sha256_bytes(target.encode("utf-8")),
        "target_token_count": len(target_ids),
        "target_token_ids_sha256": sha256_bytes(canonical_json_bytes(target_ids)),
    }


def reduce_restoration_v2_1_processor_records(
    *,
    prompt_records: Sequence[Any],
    teacher_golden_records: Sequence[Any],
    expected_state_projection: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every aggregate from primitive prompt and teacher records."""
    expected_bindings = {
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
        "maximum_context_tokens",
    }
    _exact_keys(bindings, expected_bindings, "processor bindings")
    for field in (
        "contract_sha256",
        "selection_manifest_sha256",
        "policy_interface_source_sha256",
        "runtime_source_sha256",
        "canonical_tool_schema_sha256",
        "chat_template_file_sha256",
        "chat_template_text_sha256",
    ):
        _sha256(bindings[field], f"processor bindings {field}")
    fixed_values = {
        "runtime_source_sha256": RUNTIME_SOURCE_SHA256,
        "canonical_tool_schema_sha256": CANONICAL_TOOL_SCHEMA_SHA256,
        "chat_template_file_sha256": CHAT_TEMPLATE_FILE_SHA256,
        "chat_template_text_sha256": CHAT_TEMPLATE_TEXT_SHA256,
        "assistant_prefix_token_ids": list(ASSISTANT_PREFIX_TOKEN_IDS),
        "tool_call_open_token_id": TOOL_CALL_OPEN_TOKEN_ID,
        "tool_call_close_token_id": TOOL_CALL_CLOSE_TOKEN_ID,
        "standard_eos_token_ids": list(STANDARD_EOS_TOKEN_IDS),
        "pad_token_id": PAD_TOKEN_ID,
        "maximum_context_tokens": EXPECTED_MAXIMUM_CONTEXT_TOKENS,
    }
    for field, expected in fixed_values.items():
        if bindings[field] != expected:
            raise ValueError(f"processor binding {field} drifted")
    states = list(expected_state_projection)
    if len(states) != EXPECTED_STATE_COUNT:
        raise ValueError("expected state projection must contain exactly 45 states")
    if canonical_json_sha256(states) != FULL_45_PROJECTION_SHA256:
        raise ValueError("expected state projection SHA256 drifted")
    prompts = list(prompt_records)
    if len(prompts) != EXPECTED_PROMPT_COUNT:
        raise ValueError("processor audit must contain exactly 90 prompt records")
    shape_records = [
        _validate_prompt_record(
            prompt,
            expected_prompt_index=index,
            expected_state=states[index // 2],
        )
        for index, prompt in enumerate(prompts)
    ]
    goldens = list(teacher_golden_records)
    expected_actions = _golden_actions()
    if len(goldens) != len(expected_actions):
        raise ValueError("teacher golden denominator must contain all nine actions")
    teacher_projection = [
        _validate_golden_record(record, action)
        for record, action in zip(goldens, expected_actions, strict=True)
    ]

    fidelity_counts = Counter(record["fidelity"] for record in shape_records)
    image_counts = Counter(str(record["image_count"]) for record in shape_records)
    role_state_counts = Counter(state["role"] for state in states)
    overflow_count = sum(record["context_overflow"] for record in shape_records)
    official_count = sum(
        _mapping(prompt, "prompt")["rendered_prompt"].count(OFFICIAL_TOOL_BLOCK) == 1
        for prompt in prompts
    )
    preflight = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": GUI_OWL_V2_1_PROTOCOL_ID,
        "status": PREFLIGHT_STATUS,
        "contract_sha256": bindings["contract_sha256"],
        "selection_manifest_sha256": bindings["selection_manifest_sha256"],
        "policy_interface_source_sha256": bindings[
            "policy_interface_source_sha256"
        ],
        "runtime_source_sha256": bindings["runtime_source_sha256"],
        "canonical_tool_schema_sha256": bindings[
            "canonical_tool_schema_sha256"
        ],
        "chat_template_file_sha256": bindings["chat_template_file_sha256"],
        "chat_template_text_sha256": bindings["chat_template_text_sha256"],
        "assistant_prefix_token_ids": bindings["assistant_prefix_token_ids"],
        "tool_call_open_token_id": bindings["tool_call_open_token_id"],
        "tool_call_close_token_id": bindings["tool_call_close_token_id"],
        "standard_eos_token_ids": bindings["standard_eos_token_ids"],
        "pad_token_id": bindings["pad_token_id"],
        "state_count": len(states),
        "prompt_count": len(shape_records),
        "official_tools_injected_prompt_count": official_count,
        "fidelity_counts": dict(sorted(fidelity_counts.items())),
        "role_state_counts": dict(sorted(role_state_counts.items())),
        "image_count_distribution": dict(sorted(image_counts.items())),
        "teacher_boundary_validated": len(teacher_projection) == 9,
        "full_45_state_projection_sha256": canonical_json_sha256(states),
        "reserved_generation_tokens": RESERVED_GENERATION_TOKENS,
        "maximum_context_tokens": bindings["maximum_context_tokens"],
        "context_overflow_count": overflow_count,
        "shape_records_sha256": sha256_bytes(canonical_json_bytes(shape_records)),
        "policy_model_loaded": False,
        "policy_forward_executed": False,
        "policy_generation_executed": False,
        "full_artifact_including_confirm_bytes_validated_by_loader": True,
        "confirm_state_prompt_or_image_exposed_to_decoder_or_processor": False,
        "confirm_processor_prompt_count": 0,
        "restoration_output_generated": False,
    }
    if preflight["image_count_distribution"] != IMAGE_COUNT_DISTRIBUTION:
        raise ValueError("processor image-count distribution drifted")
    validate_restoration_v2_1_processor_preflight(
        preflight,
        contract_sha256=str(bindings["contract_sha256"]),
        selection_manifest_sha256=str(bindings["selection_manifest_sha256"]),
        policy_interface_source_sha256=str(
            bindings["policy_interface_source_sha256"]
        ),
    )
    return {
        "preflight_summary": preflight,
        "prompt_records_sha256": sha256_bytes(canonical_json_bytes(prompts)),
        "teacher_golden_records_sha256": sha256_bytes(canonical_json_bytes(goldens)),
        "shape_records": shape_records,
        "teacher_projection": teacher_projection,
    }


def validate_stored_processor_reduction(
    stored_reduction: Mapping[str, Any],
    *,
    prompt_records: Sequence[Any],
    teacher_golden_records: Sequence[Any],
    expected_state_projection: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject any stored aggregate that differs from a fresh record reduction."""
    recomputed = reduce_restoration_v2_1_processor_records(
        prompt_records=prompt_records,
        teacher_golden_records=teacher_golden_records,
        expected_state_projection=expected_state_projection,
        bindings=bindings,
    )
    if dict(stored_reduction) != recomputed:
        raise ValueError("stored processor reduction differs from record recomputation")
    return recomputed


def _safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.as_posix() != value or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise ValueError(f"{label} must be a safe relative POSIX path")
    return value


def verify_processor_only_model_snapshot(
    *,
    model_dir: str | Path,
    expected_snapshot_manifest: str | Path,
) -> dict[str, Any]:
    """Hash the pinned snapshot and parse config JSON without importing a model."""
    root = Path(model_dir).resolve()
    manifest_path = Path(expected_snapshot_manifest).resolve()
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("model directory or snapshot manifest is missing")
    expected_bytes = manifest_path.read_bytes()
    expected = strict_json_object_bytes(expected_bytes, label="snapshot manifest")
    if set(expected) != {"repo", "revision", "files"}:
        raise ValueError("snapshot manifest keys drifted")
    if expected["repo"] != "mPLUG/GUI-Owl-1.5-8B-Instruct":
        raise ValueError("snapshot model repo drifted")
    if expected["revision"] != "06d5faecff74840bab2be2425e9c42667a5d04fc":
        raise ValueError("snapshot model revision drifted")
    local_snapshot_path = root / ".snapshot.json"
    if not local_snapshot_path.is_file():
        raise ValueError("model directory lacks .snapshot.json")
    local = strict_json_object(local_snapshot_path, label="local .snapshot.json")
    if local != expected:
        raise ValueError("local snapshot identity differs from the Git manifest")
    files = _sequence(expected["files"], "snapshot files")
    if len(files) != 14:
        raise ValueError("snapshot file count drifted")
    observed_paths: set[str] = set()
    total_bytes = 0
    inventory: list[dict[str, Any]] = []
    for index, raw_record in enumerate(files):
        record = _mapping(raw_record, f"snapshot.files[{index}]")
        _exact_keys(record, {"path", "size", "sha256"}, "snapshot file")
        relative = _safe_relative_path(record["path"], label="snapshot file path")
        if relative in observed_paths:
            raise ValueError("snapshot file paths are not unique")
        size = _nonnegative_int(record["size"], "snapshot file size")
        digest = _sha256(record["sha256"], "snapshot file SHA256")
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"snapshot file size drifted: {relative}")
        if sha256_file(path) != digest:
            raise ValueError(f"snapshot file SHA256 drifted: {relative}")
        observed_paths.add(relative)
        total_bytes += size
        inventory.append({"path": relative, "size": size, "sha256": digest})
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != observed_paths.union({".snapshot.json"}):
        raise ValueError("model directory contains an unexpected file inventory")

    config = strict_json_object(root / "config.json", label="model config.json")
    text_config = _mapping(config.get("text_config"), "model config.text_config")
    maximum_context = text_config.get("max_position_embeddings")
    if maximum_context != EXPECTED_MAXIMUM_CONTEXT_TOKENS:
        raise ValueError("model maximum context length drifted")
    generation = strict_json_object(
        root / "generation_config.json",
        label="model generation_config.json",
    )
    configured_eos = generation.get("eos_token_id")
    if type(configured_eos) is int:
        standard_eos = [configured_eos]
    else:
        standard_eos = list(
            _sequence(configured_eos, "generation config eos_token_id")
        )
    if standard_eos != list(STANDARD_EOS_TOKEN_IDS):
        raise ValueError("snapshot standard EOS token IDs drifted")
    if generation.get("pad_token_id") != PAD_TOKEN_ID:
        raise ValueError("snapshot pad token ID drifted")
    return {
        "model_dir": str(root),
        "model_repo": expected["repo"],
        "model_revision": expected["revision"],
        "snapshot_manifest_sha256": sha256_bytes(expected_bytes),
        "verified_file_count": len(inventory),
        "verified_total_bytes": total_bytes,
        "file_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "config_sha256": sha256_file(root / "config.json"),
        "generation_config_sha256": sha256_file(root / "generation_config.json"),
        "maximum_context_tokens": maximum_context,
        "standard_eos_token_ids": standard_eos,
        "pad_token_id": generation["pad_token_id"],
    }


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git command failed: {stderr or arguments}") from error


def _git_text(repository_root: Path, *arguments: str) -> str:
    return _git_bytes(repository_root, *arguments).decode("utf-8").strip()


def collect_clean_pushed_main_identity(
    *,
    repository_root: str | Path,
    run_git_commit: str,
) -> dict[str, Any]:
    """Require clean canonical main at the same local, tracking, and remote SHA."""
    root = Path(repository_root).resolve()
    if not root.is_dir() or GIT_SHA_PATTERN.fullmatch(run_git_commit) is None:
        raise ValueError("repository root or run Git commit is invalid")
    if Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("repository root must be the exact Git worktree root")
    head = _git_text(root, "rev-parse", "HEAD")
    if head != run_git_commit:
        raise ValueError("run Git commit must exactly equal HEAD")
    if _git_text(root, "rev-parse", "--verify", "HEAD^{commit}") != head:
        raise ValueError("Git HEAD is not a verified commit")
    branch = _git_text(root, "branch", "--show-current")
    if branch != "main":
        raise ValueError("formal processor audit must run on main")
    status = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise ValueError("formal processor audit requires a clean worktree")
    remote_url = _git_text(root, "remote", "get-url", "origin")
    if remote_url != CANONICAL_REMOTE_URL:
        raise ValueError("origin URL differs from the canonical repository")
    tracking = _git_text(root, "rev-parse", "refs/remotes/origin/main")
    if tracking != head:
        raise ValueError("origin/main differs from HEAD")
    remote_line = _git_text(
        root,
        "ls-remote",
        "--heads",
        "origin",
        "refs/heads/main",
    )
    fields = remote_line.split()
    if fields != [head, "refs/heads/main"]:
        raise ValueError("remote main differs from HEAD")
    return {
        "repository_root": str(root),
        "run_git_commit": head,
        "branch": branch,
        "origin_url": remote_url,
        "origin_main_commit": tracking,
        "remote_main_commit": fields[0],
        "worktree_status_porcelain": "",
    }


def require_evidence_commit_ancestor(
    *,
    repository_root: str | Path,
    evidence_git_commit: str,
    current_git_commit: str,
) -> None:
    """Require reusable evidence source X to be an ancestor of current main Y."""
    root = Path(repository_root).resolve()
    if GIT_SHA_PATTERN.fullmatch(evidence_git_commit) is None or GIT_SHA_PATTERN.fullmatch(
        current_git_commit
    ) is None:
        raise ValueError("evidence/current Git commit is invalid")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                evidence_git_commit,
                current_git_commit,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "processor evidence commit is not an ancestor of current clean main"
        ) from error


def _compact_evidence_reduction(value: Mapping[str, Any]) -> dict[str, Any]:
    reduction = _mapping(value.get("reduction"), "processor evidence reduction")
    preflight = _mapping(
        reduction.get("preflight_summary"),
        "processor evidence preflight summary",
    )
    processor_identity = _mapping(
        value.get("processor_identity"),
        "processor identity",
    )
    classes = _mapping(processor_identity.get("classes"), "processor classes")
    expected_class_labels = {
        "auto_processor",
        "processor",
        "tokenizer",
        "image_processor",
    }
    _exact_keys(classes, expected_class_labels, "processor classes")
    class_projection: list[dict[str, str]] = []
    for label in sorted(expected_class_labels):
        record = _mapping(classes[label], f"processor class {label}")
        _exact_keys(
            record,
            {"name", "module", "source_path", "source_sha256"},
            f"processor class {label}",
        )
        if any(
            not isinstance(record.get(field), str) or not record[field]
            for field in ("name", "module", "source_path")
        ):
            raise ValueError(f"processor class {label} identity is invalid")
        _sha256(record.get("source_sha256"), f"processor class {label} SHA256")
        class_projection.append(
            {
                "label": label,
                "name": record["name"],
                "module": record["module"],
                "source_sha256": record["source_sha256"],
            }
        )
    return {
        "status": preflight.get("status"),
        "contract_sha256": preflight.get("contract_sha256"),
        "selection_manifest_sha256": preflight.get("selection_manifest_sha256"),
        "policy_interface_source_sha256": preflight.get(
            "policy_interface_source_sha256"
        ),
        "runtime_source_sha256": preflight.get("runtime_source_sha256"),
        "state_count": preflight.get("state_count"),
        "prompt_count": preflight.get("prompt_count"),
        "official_tools_injected_prompt_count": preflight.get(
            "official_tools_injected_prompt_count"
        ),
        "image_count_distribution": preflight.get("image_count_distribution"),
        "context_overflow_count": preflight.get("context_overflow_count"),
        "shape_records_sha256": preflight.get("shape_records_sha256"),
        "prompt_records_sha256": reduction.get("prompt_records_sha256"),
        "teacher_golden_records_sha256": reduction.get(
            "teacher_golden_records_sha256"
        ),
        "processor_classes_sha256": sha256_bytes(
            canonical_json_bytes(class_projection)
        ),
    }


def build_processor_evidence_artifact_manifest(
    value: Mapping[str, Any],
    *,
    raw_evidence_path: str | Path,
    hf_repo: str,
    hf_immutable_revision: str,
    hf_path: str,
) -> dict[str, Any]:
    """Build the lightweight Git manifest after immutable private-HF upload."""
    raw_path = Path(raw_evidence_path).resolve()
    if not raw_path.is_file():
        raise ValueError("raw processor evidence file is missing")
    raw = raw_path.read_bytes()
    if strict_json_object_bytes(raw, label="raw processor evidence") != dict(value):
        raise ValueError("raw processor evidence file differs from supplied value")
    if hf_repo != CANONICAL_EVIDENCE_HF_REPO:
        raise ValueError("processor evidence HF repo differs from the canonical repo")
    if GIT_SHA_PATTERN.fullmatch(hf_immutable_revision) is None:
        raise ValueError("processor evidence HF revision must be an immutable 40-hex commit")
    safe_hf_path = _safe_relative_path(hf_path, label="processor evidence HF path")
    if not safe_hf_path.endswith(".json"):
        raise ValueError("processor evidence HF path must name a JSON artifact")
    source_commit = value.get("run_git_commit")
    if not isinstance(source_commit, str) or GIT_SHA_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("processor evidence source commit is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": GUI_OWL_V2_1_PROTOCOL_ID,
        "artifact_type": "private_hf_processor_preflight_evidence",
        "hf_artifact": {
            "repo": hf_repo,
            "repo_type": "dataset",
            "visibility": "private",
            "tag": CANONICAL_EVIDENCE_HF_TAG,
            "immutable_revision": hf_immutable_revision,
            "path": safe_hf_path,
        },
        "raw_evidence": {
            "sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
            "source_git_commit": source_commit,
        },
        "compact_reduction": _compact_evidence_reduction(value),
    }


def validate_committed_processor_evidence_artifact(
    value: Mapping[str, Any],
    *,
    repository_root: str | Path,
    current_git_commit: str,
    raw_evidence_path: str | Path,
) -> dict[str, Any]:
    """Bind external raw evidence to its lightweight current Git/HF manifest."""
    root = Path(repository_root).resolve()
    canonical = (root / CANONICAL_EVIDENCE_ARTIFACT_PATH).resolve()
    if not canonical.is_file():
        raise ValueError("canonical processor evidence artifact manifest is missing")
    live_manifest = canonical.read_bytes()
    committed_manifest = _git_bytes(
        root,
        "show",
        f"{current_git_commit}:{CANONICAL_EVIDENCE_ARTIFACT_PATH}",
    )
    if live_manifest != committed_manifest:
        raise ValueError("processor evidence artifact manifest differs from current Git blob")
    manifest = strict_json_object_bytes(
        committed_manifest,
        label="processor evidence artifact manifest",
    )
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "artifact_type",
            "hf_artifact",
            "raw_evidence",
            "compact_reduction",
        },
        "processor evidence artifact manifest",
    )
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["protocol_id"] != GUI_OWL_V2_1_PROTOCOL_ID
        or manifest["artifact_type"] != "private_hf_processor_preflight_evidence"
    ):
        raise ValueError("processor evidence artifact manifest identity drifted")
    hf = _mapping(manifest["hf_artifact"], "processor evidence hf_artifact")
    if dict(hf) != {
        "repo": CANONICAL_EVIDENCE_HF_REPO,
        "repo_type": "dataset",
        "visibility": "private",
        "tag": CANONICAL_EVIDENCE_HF_TAG,
        "immutable_revision": hf.get("immutable_revision"),
        "path": hf.get("path"),
    }:
        raise ValueError("processor evidence HF artifact metadata drifted")
    if GIT_SHA_PATTERN.fullmatch(str(hf["immutable_revision"])) is None:
        raise ValueError("processor evidence HF revision is not immutable")
    safe_hf_path = _safe_relative_path(
        hf["path"],
        label="processor evidence HF path",
    )
    if not safe_hf_path.endswith(".json"):
        raise ValueError("processor evidence HF path must name a JSON artifact")
    raw_path = Path(raw_evidence_path).resolve()
    if not raw_path.is_file():
        raise ValueError("external processor evidence file is missing")
    raw = raw_path.read_bytes()
    if strict_json_object_bytes(raw, label="external processor evidence") != dict(value):
        raise ValueError("external processor evidence differs from supplied value")
    expected_raw = {
        "sha256": sha256_bytes(raw),
        "size_bytes": len(raw),
        "source_git_commit": value.get("run_git_commit"),
    }
    if manifest["raw_evidence"] != expected_raw:
        raise ValueError("external processor evidence hash/size/source binding drifted")
    expected_compact = _compact_evidence_reduction(value)
    if manifest["compact_reduction"] != expected_compact:
        raise ValueError("processor evidence compact reduction drifted")
    return manifest


def committed_source_inventory(
    *,
    repository_root: str | Path,
    run_git_commit: str,
) -> list[dict[str, Any]]:
    """Bind every audit-critical live source to its exact committed Git blob."""
    root = Path(repository_root).resolve()
    inventory: list[dict[str, Any]] = []
    for relative in AUDITED_SOURCE_PATHS:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"audit source is missing or unsafe: {relative}")
        committed = _git_bytes(root, "show", f"{run_git_commit}:{relative}")
        live = path.read_bytes()
        if live != committed:
            raise ValueError(f"audit source differs from committed blob: {relative}")
        inventory.append(
            {
                "path": relative,
                "size_bytes": len(committed),
                "sha256": sha256_bytes(committed),
            }
        )
    return inventory


def _validate_processor_only_source(source: bytes) -> None:
    try:
        tree = ast.parse(source.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("processor audit script is not valid UTF-8 Python") from error
    forbidden_runtime_modules = {
        "causalcache.policy.gui_owl_v2_runtime",
        "causalcache.policy.gui_owl_v2_1_runtime",
        "causalcache.policy.gui_owl_v2_vision",
    }
    pretrained_loaders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                "modeling" in alias.name or alias.name in forbidden_runtime_modules
                for alias in node.names
            ):
                raise ValueError("processor audit imports a modeling/runtime module")
        elif isinstance(node, ast.ImportFrom):
            if (
                "modeling" in (node.module or "")
                or (node.module or "") in forbidden_runtime_modules
                or any(
                    alias.name.startswith("AutoModel")
                    or alias.name in {"GUIOwlV2Runtime", "GUIOwlV21Runtime"}
                    for alias in node.names
                )
            ):
                raise ValueError("processor audit imports a model/runtime class")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"forward", "generate"}:
                raise ValueError("processor audit calls a policy execution method")
            if node.func.attr == "from_pretrained":
                owner = (
                    node.func.value.id
                    if isinstance(node.func.value, ast.Name)
                    else ""
                )
                pretrained_loaders.append(owner)
        elif isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name.startswith("AutoModel"):
                raise ValueError("processor audit references an automatic model class")
    if pretrained_loaders != ["AutoProcessor"]:
        raise ValueError("processor audit must use only AutoProcessor.from_pretrained")


def _committed_json(
    root: Path,
    run_git_commit: str,
    relative: str,
) -> tuple[bytes, dict[str, Any]]:
    payload = _git_bytes(root, "show", f"{run_git_commit}:{relative}")
    return payload, strict_json_object_bytes(payload, label=relative)


def committed_input_bindings(
    *,
    repository_root: str | Path,
    run_git_commit: str,
) -> dict[str, dict[str, str]]:
    root = Path(repository_root).resolve()
    result: dict[str, dict[str, str]] = {}
    for label, relative in EXPECTED_INPUT_PATHS.items():
        payload = _git_bytes(root, "show", f"{run_git_commit}:{relative}")
        path = (root / relative).resolve()
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"input binding differs from committed blob: {relative}")
        result[label] = {
            "path": relative,
            "sha256": sha256_bytes(payload),
        }
    return result


def artifact_binding_identity(binding_manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _mapping(
        binding_manifest.get("hf_dataset_artifact"),
        "artifact binding hf_dataset_artifact",
    )
    result = {
        "repo": artifact.get("repo"),
        "immutable_revision": artifact.get("immutable_revision"),
        "payload_prefix": artifact.get("payload_prefix"),
        "artifact_tree_sha256": artifact.get("artifact_tree_sha256"),
    }
    expected = {
        "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
        "immutable_revision": "89f136abaff797e14fe758a198996e51032a10a6",
        "payload_prefix": "derived/restoration-v2-v1",
        "artifact_tree_sha256": (
            "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e"
        ),
    }
    if result != expected:
        raise ValueError("derived artifact immutable binding drifted")
    return result


def validate_processor_identity(value: Any) -> dict[str, Any]:
    identity = _mapping(value, "processor_identity")
    _exact_keys(
        identity,
        {
            "packages",
            "classes",
            "pixel_target",
            "chat_template",
            "tokenizer",
            "maximum_context_tokens",
            "loaded_qwen_modeling_modules",
        },
        "processor_identity",
    )
    packages = _mapping(identity["packages"], "processor_identity.packages")
    _exact_keys(packages, {"transformers", "torch", "Pillow"}, "packages")
    if packages["transformers"] != "5.6.0" or any(
        not isinstance(packages[name], str) or not packages[name]
        for name in ("torch", "Pillow")
    ):
        raise ValueError("processor package identity drifted")

    classes = _mapping(identity["classes"], "processor_identity.classes")
    expected_class_labels = {
        "auto_processor",
        "processor",
        "tokenizer",
        "image_processor",
    }
    _exact_keys(classes, expected_class_labels, "processor classes")
    class_projection: list[dict[str, str]] = []
    for label in sorted(expected_class_labels):
        record = _mapping(classes[label], f"processor class {label}")
        _exact_keys(
            record,
            {"name", "module", "source_path", "source_sha256"},
            f"processor class {label}",
        )
        if any(
            not isinstance(record[field], str) or not record[field]
            for field in ("name", "module", "source_path")
        ):
            raise ValueError(f"processor class {label} identity is invalid")
        _sha256(record["source_sha256"], f"processor class {label} source SHA256")
        source_path = Path(record["source_path"])
        if not source_path.is_absolute() or not source_path.is_file():
            raise ValueError(f"processor class {label} source path is invalid")
        if sha256_file(source_path) != record["source_sha256"]:
            raise ValueError(f"processor class {label} source SHA256 drifted")
        class_projection.append(
            {
                "label": label,
                "name": record["name"],
                "module": record["module"],
                "source_sha256": record["source_sha256"],
            }
        )

    target = _mapping(identity["pixel_target"], "processor_identity.pixel_target")
    expected_target = {
        "representation": "image_processor.size.shortest_edge_longest_edge",
        "target_pixels": 2_621_440,
        "actual_min_pixels": 2_621_440,
        "actual_max_pixels": 2_621_440,
        "spatial_merge_size": 2,
        "direct_min_pixels_attribute_present": False,
        "direct_max_pixels_attribute_present": False,
        "size_non_edge_fields": {
            "height": None,
            "width": None,
            "max_height": None,
            "max_width": None,
        },
    }
    if dict(target) != expected_target:
        raise ValueError("processor fixed visual pixel target drifted")
    template = _mapping(
        identity["chat_template"],
        "processor_identity.chat_template",
    )
    if dict(template) != {
        "file_sha256": CHAT_TEMPLATE_FILE_SHA256,
        "text_sha256": CHAT_TEMPLATE_TEXT_SHA256,
    }:
        raise ValueError("processor chat-template identity drifted")
    tokenizer = _mapping(identity["tokenizer"], "processor_identity.tokenizer")
    expected_tokenizer = {
        "assistant_prefix_token_ids": list(ASSISTANT_PREFIX_TOKEN_IDS),
        "tool_call_open_token_ids": [TOOL_CALL_OPEN_TOKEN_ID],
        "tool_call_close_token_ids": [TOOL_CALL_CLOSE_TOKEN_ID],
        "standard_eos_token_ids": list(STANDARD_EOS_TOKEN_IDS),
        "pad_token_id": PAD_TOKEN_ID,
        "tokenizer_eos_token_id": STANDARD_EOS_TOKEN_IDS[0],
        "tokenizer_pad_token_id": PAD_TOKEN_ID,
        "tool_call_tokens_in_all_special_ids": False,
        "decoded_tool_call_closer": "</tool_call>",
    }
    if dict(tokenizer) != expected_tokenizer:
        raise ValueError("processor tokenizer identity drifted")
    if identity["maximum_context_tokens"] != EXPECTED_MAXIMUM_CONTEXT_TOKENS:
        raise ValueError("processor maximum context length drifted")
    if identity["loaded_qwen_modeling_modules"] != []:
        raise ValueError("processor audit loaded a Qwen modeling module")
    return {
        "packages": dict(packages),
        "classes_sha256": sha256_bytes(canonical_json_bytes(class_projection)),
        "maximum_context_tokens": EXPECTED_MAXIMUM_CONTEXT_TOKENS,
    }


def _validate_artifact_identity(
    value: Any,
    *,
    expected_binding: Mapping[str, Any],
    loaded_artifact: Any,
    expected_state_projection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identity = _mapping(value, "artifact_identity")
    _exact_keys(
        identity,
        {
            "root",
            "repo",
            "immutable_revision",
            "payload_prefix",
            "artifact_tree_sha256",
            "artifact_manifest_sha256",
            "screening_manifest_sha256",
            "state_projection_sha256",
        },
        "artifact_identity",
    )
    if not isinstance(identity["root"], str) or Path(identity["root"]).resolve() != Path(
        loaded_artifact.artifact_root
    ).resolve():
        raise ValueError("artifact root identity drifted")
    for field in ("repo", "immutable_revision", "payload_prefix", "artifact_tree_sha256"):
        if identity[field] != expected_binding[field]:
            raise ValueError(f"artifact {field} identity drifted")
    if identity["artifact_manifest_sha256"] != loaded_artifact.artifact_manifest_sha256:
        raise ValueError("artifact manifest SHA256 drifted")
    if identity["screening_manifest_sha256"] != loaded_artifact.screening_manifest_sha256:
        raise ValueError("screening manifest SHA256 drifted")
    if identity["state_projection_sha256"] != canonical_json_sha256(
        list(expected_state_projection)
    ):
        raise ValueError("artifact state projection SHA256 drifted")
    loaded_projection = [
        {
            "index": state.index,
            "role": state.role,
            "trajectory_id": state.trajectory_id,
            "decision_step_id": state.decision_step_id,
            "state_id": state.state_id,
            "candidate_event_step_ids": list(state.candidate_event_step_ids),
        }
        for state in loaded_artifact.states
    ]
    if loaded_projection != list(expected_state_projection):
        raise ValueError("confirm-safe loader states differ from committed selection")
    return dict(identity)


def validate_restoration_v2_1_processor_audit(
    value: Mapping[str, Any],
    *,
    repository_root: str | Path,
    current_git_commit: str,
    mode: str,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Independently replay Git, data, snapshot, records, and summary validation."""
    audit = _mapping(value, "processor audit")
    _exact_keys(
        audit,
        {
            "schema_version",
            "protocol_id",
            "evidence_type",
            "run_git_commit",
            "execution",
            "repository_identity",
            "source_inventory",
            "input_bindings",
            "artifact_identity",
            "model_snapshot_identity",
            "processor_identity",
            "prompt_records",
            "teacher_golden_records",
            "negative_evidence",
            "reduction",
        },
        "processor audit",
    )
    if audit["schema_version"] != SCHEMA_VERSION:
        raise ValueError("processor audit schema version drifted")
    if audit["protocol_id"] != GUI_OWL_V2_1_PROTOCOL_ID:
        raise ValueError("processor audit protocol drifted")
    if audit["evidence_type"] != EVIDENCE_TYPE:
        raise ValueError("processor audit evidence type drifted")
    root = Path(repository_root).resolve()
    if mode not in {"generation", "reuse"}:
        raise ValueError("processor audit validation mode must be generation or reuse")
    evidence_commit = audit["run_git_commit"]
    if not isinstance(evidence_commit, str) or GIT_SHA_PATTERN.fullmatch(
        evidence_commit
    ) is None:
        raise ValueError("processor audit evidence Git commit is invalid")
    execution = _mapping(audit["execution"], "processor audit execution")
    _exact_keys(
        execution,
        {"argv", "output_path", "runtime"},
        "processor audit execution",
    )
    runtime = _mapping(execution["runtime"], "processor audit execution runtime")
    _exact_keys(
        runtime,
        {
            "started_at_utc",
            "ended_at_utc",
            "duration_seconds",
            "host_alias",
            "host_hostname",
            "container_id",
            "container_hostname",
            "container_image_digest",
            "python_version",
            "python_executable",
            "platform_system",
            "platform_machine",
            "processor_device",
            "gpu_operations_executed",
            "policy_dtype",
            "dtype_not_applicable",
            "random_seed",
            "seed_not_applicable",
        },
        "processor audit execution runtime",
    )
    for field in (
        "started_at_utc",
        "ended_at_utc",
        "host_alias",
        "host_hostname",
        "container_id",
        "container_hostname",
        "container_image_digest",
        "python_version",
        "python_executable",
        "platform_system",
        "platform_machine",
        "processor_device",
    ):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise ValueError(f"processor audit runtime {field} is invalid")
    try:
        started_at = datetime.fromisoformat(runtime["started_at_utc"].replace("Z", "+00:00"))
        ended_at = datetime.fromisoformat(runtime["ended_at_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("processor audit runtime UTC timestamp is invalid") from error
    if not runtime["started_at_utc"].endswith("Z") or not runtime[
        "ended_at_utc"
    ].endswith("Z"):
        raise ValueError("processor audit runtime timestamps must use UTC Z")
    duration = runtime["duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError("processor audit runtime duration is invalid")
    wall_duration = (ended_at - started_at).total_seconds()
    if wall_duration < 0 or abs(wall_duration - float(duration)) > 5.0:
        raise ValueError("processor audit runtime duration differs from UTC timestamps")
    if re.fullmatch(r"[0-9a-f]{64}", runtime["container_id"]) is None:
        raise ValueError("processor audit container ID is invalid")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["container_image_digest"]) is None:
        raise ValueError("processor audit container image digest is invalid")
    if not runtime["container_id"].startswith(runtime["container_hostname"]):
        raise ValueError("processor audit container hostname/ID binding drifted")
    if (
        runtime["processor_device"] != "cpu"
        or runtime["gpu_operations_executed"] is not False
        or runtime["policy_dtype"] is not None
        or runtime["dtype_not_applicable"] is not True
        or runtime["random_seed"] is not None
        or runtime["seed_not_applicable"] is not True
    ):
        raise ValueError("processor audit CPU/dtype/seed provenance drifted")
    argv = list(_sequence(execution["argv"], "processor audit argv"))
    if len(argv) != 32 or any(
        not isinstance(argument, str) or not argument for argument in argv
    ):
        raise ValueError("processor audit argv is invalid")
    if Path(argv[1]).name != "audit_gui_owl_v2_1_processor.py":
        raise ValueError("processor audit argv lacks the canonical CLI source")
    ordered_flags = (
        "--repository-root",
        "--derived-artifact-root",
        "--model-dir",
        "--snapshot-manifest",
        "--v2-config",
        "--v2-1-config",
        "--selection-manifest",
        "--ocr-backend-config",
        "--artifact-binding-manifest",
        "--host-alias",
        "--host-hostname",
        "--container-id",
        "--container-image-digest",
        "--run-git-commit",
        "--output",
    )
    required_flags = set(ordered_flags)
    if tuple(argv[2::2]) != ordered_flags:
        raise ValueError("processor audit argv flag order drifted")
    flag_values: dict[str, str] = {}
    for flag in required_flags:
        positions = [index for index, argument in enumerate(argv) if argument == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            raise ValueError(f"processor audit argv {flag} occurrence drifted")
        flag_values[flag] = argv[positions[0] + 1]
    if set(argument for argument in argv if argument.startswith("--")) != required_flags:
        raise ValueError("processor audit argv flag inventory drifted")
    if flag_values["--run-git-commit"] != evidence_commit:
        raise ValueError("processor audit argv Git commit drifted")
    current_identity = collect_clean_pushed_main_identity(
        repository_root=root,
        run_git_commit=current_git_commit,
    )
    stored_repository_value = _mapping(
        audit["repository_identity"],
        "stored generation-time repository identity",
    )
    recorded_root_value = stored_repository_value.get("repository_root")
    if not isinstance(recorded_root_value, str) or not Path(
        recorded_root_value
    ).is_absolute():
        raise ValueError("stored generation-time repository root is invalid")
    recorded_root = Path(recorded_root_value).resolve()
    stored_repository = {
        "repository_root": str(recorded_root),
        "run_git_commit": evidence_commit,
        "branch": "main",
        "origin_url": CANONICAL_REMOTE_URL,
        "origin_main_commit": evidence_commit,
        "remote_main_commit": evidence_commit,
        "worktree_status_porcelain": "",
    }
    if dict(stored_repository_value) != stored_repository:
        raise ValueError("stored generation-time clean pushed identity drifted")
    if Path(flag_values["--repository-root"]).resolve() != recorded_root:
        raise ValueError("processor audit argv repository root drifted")
    expected_argv_paths = {
        "--snapshot-manifest": EXPECTED_INPUT_PATHS["snapshot_manifest"],
        "--v2-config": EXPECTED_INPUT_PATHS["v2_contract"],
        "--v2-1-config": EXPECTED_INPUT_PATHS["v2_1_contract"],
        "--selection-manifest": EXPECTED_INPUT_PATHS["selection_manifest"],
        "--ocr-backend-config": EXPECTED_INPUT_PATHS["ocr_backend_config"],
        "--artifact-binding-manifest": EXPECTED_INPUT_PATHS[
            "artifact_binding_manifest"
        ],
    }
    for flag, relative in expected_argv_paths.items():
        if Path(flag_values[flag]).resolve() != (recorded_root / relative).resolve():
            raise ValueError(f"processor audit argv {flag} path drifted")
    if not isinstance(execution["output_path"], str) or Path(
        flag_values["--output"]
    ).resolve() != Path(execution["output_path"]).resolve():
        raise ValueError("processor audit argv output path drifted")
    output_path = Path(execution["output_path"]).resolve()
    if output_path == root or root in output_path.parents:
        raise ValueError("processor audit raw output path must remain outside Git")
    if mode == "generation":
        if evidence_path is not None:
            raise ValueError("generation-time validation does not accept an evidence path")
        if current_git_commit != evidence_commit or current_identity != stored_repository:
            raise ValueError("generation-time audit must run at its evidence commit")
        actual_runtime = {
            "container_hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
        }
        for field, expected in actual_runtime.items():
            if runtime[field] != expected:
                raise ValueError(f"processor audit live runtime {field} drifted")
    else:
        if evidence_path is None:
            raise ValueError("reuse requires an external raw processor evidence path")
        validate_committed_processor_evidence_artifact(
            audit,
            repository_root=root,
            current_git_commit=current_git_commit,
            raw_evidence_path=evidence_path,
        )
        require_evidence_commit_ancestor(
            repository_root=root,
            evidence_git_commit=evidence_commit,
            current_git_commit=current_git_commit,
        )
    source_inventory = committed_source_inventory(
        repository_root=root,
        run_git_commit=evidence_commit,
    )
    if audit["source_inventory"] != source_inventory:
        raise ValueError("stored source inventory differs from committed blobs")
    script_source = _git_bytes(
        root,
        "show",
        f"{evidence_commit}:code/scripts/audit_gui_owl_v2_1_processor.py",
    )
    _validate_processor_only_source(script_source)
    input_bindings = committed_input_bindings(
        repository_root=root,
        run_git_commit=evidence_commit,
    )
    if audit["input_bindings"] != input_bindings:
        raise ValueError("stored input bindings differ from committed blobs")

    _, v2_contract_json = _committed_json(
        root,
        evidence_commit,
        EXPECTED_INPUT_PATHS["v2_contract"],
    )
    _, v21_contract_json = _committed_json(
        root,
        evidence_commit,
        EXPECTED_INPUT_PATHS["v2_1_contract"],
    )
    _, selection = _committed_json(
        root,
        evidence_commit,
        EXPECTED_INPUT_PATHS["selection_manifest"],
    )
    _, artifact_binding_json = _committed_json(
        root,
        evidence_commit,
        EXPECTED_INPUT_PATHS["artifact_binding_manifest"],
    )
    from causalcache.restoration_v2_contract import validate_restoration_v2_contract

    from causalcache.restoration_v2_1_contract import (
        validate_restoration_v2_1_pilot_contract,
    )

    validate_restoration_v2_contract(v2_contract_json)
    validate_restoration_v2_1_pilot_contract(v21_contract_json, repository_root=root)
    contract_execution = _mapping(
        v21_contract_json.get("pilot_execution"),
        "v2.1 contract pilot_execution",
    )
    expected_runtime_binding = {
        "host_alias": contract_execution.get("canonical_host_alias"),
        "host_hostname": contract_execution.get("canonical_host_hostname"),
        "container_id": contract_execution.get("canonical_container_id"),
        "container_image_digest": contract_execution.get(
            "canonical_container_image_digest"
        ),
    }
    for field, expected in expected_runtime_binding.items():
        if runtime[field] != expected or flag_values[f"--{field.replace('_', '-')}"] != expected:
            raise ValueError(f"processor audit frozen runtime {field} drifted")
    states = screening_state_projection(selection)
    expected_artifact_binding = artifact_binding_identity(artifact_binding_json)

    from causalcache.data.restoration_v2_screening import (
        load_validated_screening_artifact,
    )

    artifact_identity = _mapping(audit["artifact_identity"], "artifact_identity")
    if Path(flag_values["--derived-artifact-root"]).resolve() != Path(
        str(artifact_identity.get("root"))
    ).resolve():
        raise ValueError("processor audit argv derived artifact root drifted")
    loaded_artifact = load_validated_screening_artifact(
        artifact_root=str(artifact_identity.get("root")),
        backend_config_path=root / EXPECTED_INPUT_PATHS["ocr_backend_config"],
        scientific_config_path=root / EXPECTED_INPUT_PATHS["v2_contract"],
        selection_manifest_path=root / EXPECTED_INPUT_PATHS["selection_manifest"],
        expected_artifact_tree_sha256=str(
            expected_artifact_binding["artifact_tree_sha256"]
        ),
    )
    _validate_artifact_identity(
        artifact_identity,
        expected_binding=expected_artifact_binding,
        loaded_artifact=loaded_artifact,
        expected_state_projection=states,
    )
    snapshot_identity = verify_processor_only_model_snapshot(
        model_dir=str(
            _mapping(audit["model_snapshot_identity"], "model snapshot").get(
                "model_dir"
            )
        ),
        expected_snapshot_manifest=root / EXPECTED_INPUT_PATHS["snapshot_manifest"],
    )
    if Path(flag_values["--model-dir"]).resolve() != Path(
        str(snapshot_identity["model_dir"])
    ).resolve():
        raise ValueError("processor audit argv model directory drifted")
    if audit["model_snapshot_identity"] != snapshot_identity:
        raise ValueError("stored model snapshot identity differs from verified files")
    processor_projection = validate_processor_identity(audit["processor_identity"])
    expected_negative = {
        "only_pretrained_loader": "AutoProcessor.from_pretrained",
        "model_weights_materialized_as_tensors": False,
        "policy_model_loaded": False,
        "policy_forward_executed": False,
        "policy_generation_executed": False,
        "restoration_output_generated": False,
        "full_artifact_including_confirm_bytes_validated_by_loader": True,
        "confirm_state_prompt_or_image_exposed_to_decoder_or_processor": False,
        "confirm_processor_prompt_count": 0,
    }
    if audit["negative_evidence"] != expected_negative:
        raise ValueError("processor-only negative evidence drifted")

    bindings = {
        "contract_sha256": input_bindings["v2_1_contract"]["sha256"],
        "selection_manifest_sha256": input_bindings["selection_manifest"][
            "sha256"
        ],
        "policy_interface_source_sha256": next(
            record["sha256"]
            for record in source_inventory
            if record["path"] == "code/causalcache/policy/gui_owl_v2_1.py"
        ),
        "runtime_source_sha256": next(
            record["sha256"]
            for record in source_inventory
            if record["path"] == "code/causalcache/policy/gui_owl_v2_1_runtime.py"
        ),
        "canonical_tool_schema_sha256": canonical_json_sha256(
            GUI_OWL_V2_1_MOBILE_USE_TOOL
        ),
        "chat_template_file_sha256": audit["processor_identity"]["chat_template"][
            "file_sha256"
        ],
        "chat_template_text_sha256": audit["processor_identity"]["chat_template"][
            "text_sha256"
        ],
        "assistant_prefix_token_ids": audit["processor_identity"]["tokenizer"][
            "assistant_prefix_token_ids"
        ],
        "tool_call_open_token_id": audit["processor_identity"]["tokenizer"][
            "tool_call_open_token_ids"
        ][0],
        "tool_call_close_token_id": audit["processor_identity"]["tokenizer"][
            "tool_call_close_token_ids"
        ][0],
        "standard_eos_token_ids": snapshot_identity["standard_eos_token_ids"],
        "pad_token_id": snapshot_identity["pad_token_id"],
        "maximum_context_tokens": snapshot_identity["maximum_context_tokens"],
    }
    reduction = validate_stored_processor_reduction(
        _mapping(audit["reduction"], "stored processor reduction"),
        prompt_records=list(
            _sequence(audit["prompt_records"], "processor prompt records")
        ),
        teacher_golden_records=list(
            _sequence(audit["teacher_golden_records"], "teacher golden records")
        ),
        expected_state_projection=states,
        bindings=bindings,
    )
    return {
        "status": PREFLIGHT_STATUS,
        "prompt_count": reduction["preflight_summary"]["prompt_count"],
        "prompt_records_sha256": reduction["prompt_records_sha256"],
        "shape_records_sha256": reduction["preflight_summary"][
            "shape_records_sha256"
        ],
        "teacher_golden_records_sha256": reduction[
            "teacher_golden_records_sha256"
        ],
        "processor_classes_sha256": processor_projection["classes_sha256"],
        "evidence_git_commit": evidence_commit,
        "current_git_commit": current_git_commit,
        "validation_mode": mode,
    }
