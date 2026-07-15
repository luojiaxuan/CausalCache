"""Run the frozen 90-prompt GUI-Owl v2.1 processor-only preflight."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import io
import json
import platform
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.data.restoration_v2_1_processor_inputs import (
    build_v2_1_processor_messages,
    iter_v2_1_processor_prompt_specs,
)
from causalcache.data.restoration_v2_screening import (
    load_validated_screening_artifact,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_MOBILE_USE_TOOL,
    GUI_OWL_V2_1_PROTOCOL_ID,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    canonical_json_sha256,
    serialize_gui_owl_v2_1_teacher_target,
    validate_gui_owl_v2_1_native_messages,
)
from causalcache.restoration_v2_1_contract import RestorationV21PilotContract
from causalcache.restoration_v2_1_processor_audit import (
    EVIDENCE_TYPE,
    EXPECTED_INPUT_PATHS,
    EXPECTED_MAXIMUM_CONTEXT_TOKENS,
    EXPECTED_TENSOR_DTYPES,
    SCHEMA_VERSION,
    artifact_binding_identity,
    canonical_json_bytes,
    collect_clean_pushed_main_identity,
    committed_input_bindings,
    committed_source_inventory,
    reduce_restoration_v2_1_processor_records,
    screening_state_projection,
    sha256_file,
    strict_json_object,
    validate_processor_identity,
    validate_restoration_v2_1_processor_audit,
    verify_processor_only_model_snapshot,
)
from causalcache.restoration_v2_contract import RestorationV2Contract


TARGET_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE = 2_560
TARGET_PIXELS = TARGET_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE * (16 * 2) ** 2
ASSISTANT_PREFIX = "<|im_start|>assistant\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit all 90 v2.1 prompts with AutoProcessor only."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--derived-artifact-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--v2-config", type=Path, required=True)
    parser.add_argument("--v2-1-config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--ocr-backend-config", type=Path, required=True)
    parser.add_argument("--artifact-binding-manifest", type=Path, required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--run-git-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args.audit_argv = [sys.executable, str(Path(__file__).resolve()), *raw_argv]
    return args


def _require_canonical_input_paths(args: argparse.Namespace, root: Path) -> None:
    supplied = {
        "v2_contract": args.v2_config,
        "v2_1_contract": args.v2_1_config,
        "selection_manifest": args.selection_manifest,
        "snapshot_manifest": args.snapshot_manifest,
        "ocr_backend_config": args.ocr_backend_config,
        "artifact_binding_manifest": args.artifact_binding_manifest,
    }
    for label, path in supplied.items():
        expected = (root / EXPECTED_INPUT_PATHS[label]).resolve()
        if Path(path).resolve() != expected or not expected.is_file():
            raise ValueError(f"--{label.replace('_', '-')} must be {expected}")


def _require_absolute_paths(args: argparse.Namespace) -> None:
    fields = (
        "repository_root",
        "derived_artifact_root",
        "model_dir",
        "snapshot_manifest",
        "v2_config",
        "v2_1_config",
        "selection_manifest",
        "ocr_backend_config",
        "artifact_binding_manifest",
        "output",
    )
    for field in fields:
        value = getattr(args, field)
        if not Path(value).is_absolute():
            raise ValueError(
                f"processor audit --{field.replace('_', '-')} must be absolute"
            )


def _require_external_new_output(path: Path, repository_root: Path) -> Path:
    output = path.expanduser().resolve()
    if output == repository_root or repository_root in output.parents:
        raise ValueError("formal processor audit output must be outside the Git worktree")
    if not output.parent.is_dir():
        raise ValueError("processor audit output parent directory must already exist")
    if output.exists():
        raise FileExistsError(f"processor audit output already exists: {output}")
    return output


def _recorded_argv(args: argparse.Namespace) -> list[str]:
    observed = getattr(args, "audit_argv", None)
    if isinstance(observed, list) and observed and all(
        isinstance(value, str) and value for value in observed
    ):
        return list(observed)
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--repository-root",
        str(args.repository_root),
        "--derived-artifact-root",
        str(args.derived_artifact_root),
        "--model-dir",
        str(args.model_dir),
        "--snapshot-manifest",
        str(args.snapshot_manifest),
        "--v2-config",
        str(args.v2_config),
        "--v2-1-config",
        str(args.v2_1_config),
        "--selection-manifest",
        str(args.selection_manifest),
        "--ocr-backend-config",
        str(args.ocr_backend_config),
        "--artifact-binding-manifest",
        str(args.artifact_binding_manifest),
        "--host-alias",
        str(args.host_alias),
        "--host-hostname",
        str(args.host_hostname),
        "--container-id",
        str(args.container_id),
        "--container-image-digest",
        str(args.container_image_digest),
        "--run-git-commit",
        str(args.run_git_commit),
        "--output",
        str(args.output),
    ]


def _decode_rgb_image(payload: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return image.convert("RGB")


def _tensor_to_list(value: Any, *, name: str) -> Any:
    if not hasattr(value, "detach") or not hasattr(value, "tolist"):
        raise TypeError(f"processor output {name} is not a tensor")
    detached = value.detach()
    if hasattr(detached, "to"):
        detached = detached.to(device="cpu")
    return detached.tolist()


def _tensor_shape(value: Any, *, name: str) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"processor output {name} lacks a shape")
    try:
        result = [int(dimension) for dimension in shape]
    except (TypeError, ValueError) as error:
        raise TypeError(f"processor output {name} shape is invalid") from error
    if not result or any(dimension <= 0 for dimension in result):
        raise ValueError(f"processor output {name} shape is empty or invalid")
    return result


def _tensor_inventory(encoded: Mapping[str, Any]) -> dict[str, Any]:
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    allowed = required.union({"mm_token_type_ids"})
    if required.difference(encoded) or set(encoded).difference(allowed):
        raise ValueError("processor tensor inventory drifted")
    result: dict[str, Any] = {}
    for name in sorted(encoded):
        value = encoded[name]
        record = {
            "shape": _tensor_shape(value, name=name),
            "dtype": str(getattr(value, "dtype", "")),
            "device": str(getattr(value, "device", "")),
            "requires_grad": bool(getattr(value, "requires_grad", False)),
        }
        if record["dtype"] != EXPECTED_TENSOR_DTYPES[name]:
            raise ValueError(f"processor tensor {name} dtype drifted")
        if record["device"] != "cpu" or record["requires_grad"]:
            raise ValueError(f"processor tensor {name} must remain a no-grad CPU tensor")
        result[name] = record
    return result


def _single_render(value: Any, *, label: str) -> str:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str):
        raise ValueError(f"{label} must return one rendered string for batch size one")
    return value[0]


def _image_objects(messages: Sequence[Mapping[str, Any]]) -> list[Any]:
    content = messages[1].get("content")
    if not isinstance(content, list):
        raise ValueError("native user content drifted")
    return [block["image"] for block in content if block.get("type") == "image"]


def _native_user_text_blocks(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    content = messages[1].get("content")
    if not isinstance(content, list):
        raise ValueError("native user content drifted")
    result = [block["text"] for block in content if block.get("type") == "text"]
    if any(not isinstance(text, str) for text in result):
        raise ValueError("native user text block drifted")
    return result


def _prompt_record(
    processor: Any,
    *,
    spec: Any,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_gui_owl_v2_1_native_messages(messages)
    rendered = _single_render(
        processor.apply_chat_template(
            [messages],
            tools=[copy.deepcopy(GUI_OWL_V2_1_MOBILE_USE_TOOL)],
            tokenize=False,
            add_generation_prompt=True,
        ),
        label="official-tools render",
    )
    rendered_without_tools = _single_render(
        processor.apply_chat_template(
            [messages],
            tokenize=False,
            add_generation_prompt=True,
        ),
        label="no-tools render",
    )
    encoded_value = processor.apply_chat_template(
        [messages],
        tools=[copy.deepcopy(GUI_OWL_V2_1_MOBILE_USE_TOOL)],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=False,
    )
    if not isinstance(encoded_value, Mapping):
        raise TypeError("tokenized chat template must return a mapping")
    encoded = dict(encoded_value)
    inventory = _tensor_inventory(encoded)
    input_rows = _tensor_to_list(encoded["input_ids"], name="input_ids")
    mask_rows = _tensor_to_list(encoded["attention_mask"], name="attention_mask")
    grids = _tensor_to_list(encoded["image_grid_thw"], name="image_grid_thw")
    if not isinstance(input_rows, list) or len(input_rows) != 1:
        raise ValueError("processor input_ids batch dimension drifted")
    if not isinstance(mask_rows, list) or len(mask_rows) != 1:
        raise ValueError("processor attention_mask batch dimension drifted")
    state = spec.state
    return {
        "prompt_index": spec.prompt_index,
        "state_index": state.index,
        "state_id": state.state_id,
        "role": state.role,
        "trajectory_id": state.trajectory_id,
        "decision_step_id": state.decision_step_id,
        "candidate_event_step_ids": list(state.candidate_event_step_ids),
        "fidelity": spec.fidelity,
        "restored_event_step_ids": list(spec.restored_event_step_ids),
        "native_system_text": messages[0]["content"][0]["text"],
        "native_user_text_blocks": _native_user_text_blocks(messages),
        "rendered_prompt": rendered,
        "rendered_without_tools": rendered_without_tools,
        "input_ids": input_rows[0],
        "attention_mask": mask_rows[0],
        "image_grid_thw": grids,
        "tensor_inventory": inventory,
    }


def _golden_actions() -> tuple[GUIOwlV2Action, ...]:
    return (
        GUIOwlV2Action(action="click", coordinate=(0, 999)),
        GUIOwlV2Action(action="long_press", coordinate=(500, 500)),
        GUIOwlV2Action(
            action="swipe", coordinate=(1, 2), coordinate2=(998, 997)
        ),
        GUIOwlV2Action(action="type", text="Café"),
        GUIOwlV2Action(action="system_button", button="Back"),
        GUIOwlV2Action(action="open", text="设置"),
        GUIOwlV2Action(action="wait"),
        GUIOwlV2Action(action="answer", text="完成 ✅"),
        GUIOwlV2Action(action="terminate", status="success"),
    )


def _teacher_golden_records(processor: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    tokenizer = processor.tokenizer
    for action in _golden_actions():
        tool_calls = [
            {
                "type": "function",
                "function": {
                    "name": "mobile_use",
                    "arguments": action.arguments(),
                },
            }
        ]
        prefix_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION}
                ],
            },
        ]
        conversation = [
            *prefix_messages,
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]
        rendered = _single_render(
            processor.apply_chat_template(
                [conversation],
                tools=[copy.deepcopy(GUI_OWL_V2_1_MOBILE_USE_TOOL)],
                tokenize=False,
                add_generation_prompt=False,
            ),
            label=f"assistant tool_calls golden {action.action}",
        )
        target = serialize_gui_owl_v2_1_teacher_target(action)
        prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        joint_ids = tokenizer.encode(
            ASSISTANT_PREFIX + target,
            add_special_tokens=False,
        )
        records.append(
            {
                "action": action.action,
                "assistant_content": "",
                "assistant_tool_calls": tool_calls,
                "target_text": target,
                "rendered_conversation": rendered,
                "assistant_prefix_token_ids": list(prefix_ids),
                "target_token_ids": list(target_ids),
                "joint_token_ids": list(joint_ids),
            }
        )
    return records


def _class_identity(value: type[Any]) -> dict[str, str]:
    source = inspect.getsourcefile(value)
    if source is None:
        source = inspect.getfile(value)
    path = Path(source).resolve()
    if not path.is_file():
        raise ValueError(f"class source is not a file: {value}")
    return {
        "name": value.__name__,
        "module": value.__module__,
        "source_path": str(path),
        "source_sha256": sha256_file(path),
    }


def _processor_identity(
    processor: Any,
    *,
    auto_processor_class: type[Any],
    model_dir: Path,
    model_snapshot_identity: Mapping[str, Any],
) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    chat_template_file = model_dir / "chat_template.json"
    chat_template_record = strict_json_object(
        chat_template_file,
        label="chat_template.json",
    )
    if set(chat_template_record) != {"chat_template"}:
        raise ValueError("chat_template.json keys drifted")
    template = chat_template_record["chat_template"]
    if not isinstance(template, str) or template != getattr(
        tokenizer,
        "chat_template",
        None,
    ):
        raise ValueError("in-memory chat template differs from snapshot")
    size = getattr(image_processor, "size", None)
    pixel_target = {
        "representation": "image_processor.size.shortest_edge_longest_edge",
        "target_pixels": TARGET_PIXELS,
        "actual_min_pixels": getattr(size, "shortest_edge", None),
        "actual_max_pixels": getattr(size, "longest_edge", None),
        "spatial_merge_size": getattr(image_processor, "merge_size", None),
        "direct_min_pixels_attribute_present": hasattr(image_processor, "min_pixels"),
        "direct_max_pixels_attribute_present": hasattr(image_processor, "max_pixels"),
        "size_non_edge_fields": {
            name: getattr(size, name, None)
            for name in ("height", "width", "max_height", "max_width")
        },
    }
    prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    open_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    close_ids = tokenizer.encode("</tool_call>", add_special_tokens=False)
    all_special_ids = set(int(token) for token in tokenizer.all_special_ids)
    identity = {
        "packages": {
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "Pillow": importlib.metadata.version("Pillow"),
        },
        "classes": {
            "auto_processor": _class_identity(auto_processor_class),
            "processor": _class_identity(processor.__class__),
            "tokenizer": _class_identity(tokenizer.__class__),
            "image_processor": _class_identity(image_processor.__class__),
        },
        "pixel_target": pixel_target,
        "chat_template": {
            "file_sha256": sha256_file(chat_template_file),
            "text_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        },
        "tokenizer": {
            "assistant_prefix_token_ids": list(prefix_ids),
            "tool_call_open_token_ids": list(open_ids),
            "tool_call_close_token_ids": list(close_ids),
            "standard_eos_token_ids": model_snapshot_identity[
                "standard_eos_token_ids"
            ],
            "pad_token_id": model_snapshot_identity["pad_token_id"],
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
            "tool_call_tokens_in_all_special_ids": bool(
                set(open_ids).union(close_ids).intersection(all_special_ids)
            ),
            "decoded_tool_call_closer": tokenizer.decode(
                close_ids,
                skip_special_tokens=False,
            ),
        },
        "maximum_context_tokens": model_snapshot_identity[
            "maximum_context_tokens"
        ],
        "loaded_qwen_modeling_modules": sorted(
            name
            for name in sys.modules
            if name.startswith("transformers.models.qwen") and ".modeling_" in name
        ),
    }
    validate_processor_identity(identity)
    return identity


def _load_auto_processor(model_dir: Path) -> tuple[Any, type[Any]]:
    before = {
        name
        for name in sys.modules
        if name.startswith("transformers.models.qwen") and ".modeling_" in name
    }
    if before:
        raise ValueError("a Qwen modeling module was loaded before AutoProcessor")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        trust_remote_code=False,
        local_files_only=True,
        min_pixels=TARGET_PIXELS,
        max_pixels=TARGET_PIXELS,
    )
    return processor, AutoProcessor


def _validate_explicit_runtime_args(
    args: argparse.Namespace,
    *,
    contract: RestorationV21PilotContract,
) -> None:
    execution = contract.data["pilot_execution"]
    expected = {
        "host_alias": execution["canonical_host_alias"],
        "host_hostname": execution["canonical_host_hostname"],
        "container_id": execution["canonical_container_id"],
        "container_image_digest": execution["canonical_container_image_digest"],
    }
    for field, value in expected.items():
        if getattr(args, field) != value:
            raise ValueError(f"--{field.replace('_', '-')} differs from frozen contract")
    if not args.container_id.startswith(socket.gethostname()):
        raise ValueError("actual container hostname is not a prefix of --container-id")


def _execution_runtime_identity(
    args: argparse.Namespace,
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": duration_seconds,
        "host_alias": args.host_alias,
        "host_hostname": args.host_hostname,
        "container_id": args.container_id,
        "container_hostname": socket.gethostname(),
        "container_image_digest": args.container_image_digest,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "processor_device": "cpu",
        "gpu_operations_executed": False,
        "policy_dtype": None,
        "dtype_not_applicable": True,
        "random_seed": None,
        "seed_not_applicable": True,
    }


def run_processor_audit(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at_utc = _utc_now()
    started_monotonic = time.monotonic()
    _require_absolute_paths(args)
    root = Path(args.repository_root).expanduser().resolve()
    output = _require_external_new_output(Path(args.output), root)
    _require_canonical_input_paths(args, root)
    repository_identity = collect_clean_pushed_main_identity(
        repository_root=root,
        run_git_commit=args.run_git_commit,
    )
    source_inventory = committed_source_inventory(
        repository_root=root,
        run_git_commit=args.run_git_commit,
    )
    input_bindings = committed_input_bindings(
        repository_root=root,
        run_git_commit=args.run_git_commit,
    )
    RestorationV2Contract.load(args.v2_config)
    contract = RestorationV21PilotContract.load(
        args.v2_1_config,
        repository_root=root,
    )
    _validate_explicit_runtime_args(args, contract=contract)
    selection = strict_json_object(args.selection_manifest, label="selection manifest")
    states = screening_state_projection(selection)
    artifact_binding = artifact_binding_identity(
        strict_json_object(
            args.artifact_binding_manifest,
            label="artifact binding manifest",
        )
    )
    artifact = load_validated_screening_artifact(
        artifact_root=args.derived_artifact_root,
        backend_config_path=args.ocr_backend_config,
        scientific_config_path=args.v2_config,
        selection_manifest_path=args.selection_manifest,
        expected_artifact_tree_sha256=artifact_binding["artifact_tree_sha256"],
    )
    loaded_projection = [
        {
            "index": state.index,
            "role": state.role,
            "trajectory_id": state.trajectory_id,
            "decision_step_id": state.decision_step_id,
            "state_id": state.state_id,
            "candidate_event_step_ids": list(state.candidate_event_step_ids),
        }
        for state in artifact.states
    ]
    if loaded_projection != states:
        raise ValueError("confirm-safe loader states differ from committed selection")
    snapshot_identity = verify_processor_only_model_snapshot(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
    )
    processor, auto_processor_class = _load_auto_processor(Path(args.model_dir).resolve())
    processor_identity = _processor_identity(
        processor,
        auto_processor_class=auto_processor_class,
        model_dir=Path(args.model_dir).resolve(),
        model_snapshot_identity=snapshot_identity,
    )

    prompt_records: list[dict[str, Any]] = []
    for spec in iter_v2_1_processor_prompt_specs(artifact):
        messages = build_v2_1_processor_messages(
            artifact,
            spec,
            image_decoder=_decode_rgb_image,
        )
        images = _image_objects(messages)
        try:
            prompt_records.append(
                _prompt_record(processor, spec=spec, messages=messages)
            )
        finally:
            for image in images:
                if hasattr(image, "close"):
                    image.close()
    golden_records = _teacher_golden_records(processor)
    if processor_identity != _processor_identity(
        processor,
        auto_processor_class=auto_processor_class,
        model_dir=Path(args.model_dir).resolve(),
        model_snapshot_identity=snapshot_identity,
    ):
        raise ValueError("processor identity drifted during the prompt sweep")
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
        "chat_template_file_sha256": processor_identity["chat_template"][
            "file_sha256"
        ],
        "chat_template_text_sha256": processor_identity["chat_template"][
            "text_sha256"
        ],
        "assistant_prefix_token_ids": processor_identity["tokenizer"][
            "assistant_prefix_token_ids"
        ],
        "tool_call_open_token_id": processor_identity["tokenizer"][
            "tool_call_open_token_ids"
        ][0],
        "tool_call_close_token_id": processor_identity["tokenizer"][
            "tool_call_close_token_ids"
        ][0],
        "standard_eos_token_ids": snapshot_identity["standard_eos_token_ids"],
        "pad_token_id": snapshot_identity["pad_token_id"],
        "maximum_context_tokens": EXPECTED_MAXIMUM_CONTEXT_TOKENS,
    }
    reduction = reduce_restoration_v2_1_processor_records(
        prompt_records=prompt_records,
        teacher_golden_records=golden_records,
        expected_state_projection=states,
        bindings=bindings,
    )
    ended_at_utc = _utc_now()
    duration_seconds = round(time.monotonic() - started_monotonic, 6)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": GUI_OWL_V2_1_PROTOCOL_ID,
        "evidence_type": EVIDENCE_TYPE,
        "run_git_commit": args.run_git_commit,
        "execution": {
            "argv": _recorded_argv(args),
            "output_path": str(output),
            "runtime": _execution_runtime_identity(
                args,
                started_at_utc=started_at_utc,
                ended_at_utc=ended_at_utc,
                duration_seconds=duration_seconds,
            ),
        },
        "repository_identity": repository_identity,
        "source_inventory": source_inventory,
        "input_bindings": input_bindings,
        "artifact_identity": {
            "root": str(Path(args.derived_artifact_root).resolve()),
            **artifact_binding,
            "artifact_manifest_sha256": artifact.artifact_manifest_sha256,
            "screening_manifest_sha256": artifact.screening_manifest_sha256,
            "state_projection_sha256": canonical_json_sha256(states),
        },
        "model_snapshot_identity": snapshot_identity,
        "processor_identity": processor_identity,
        "prompt_records": prompt_records,
        "teacher_golden_records": golden_records,
        "negative_evidence": {
            "only_pretrained_loader": "AutoProcessor.from_pretrained",
            "model_weights_materialized_as_tensors": False,
            "policy_model_loaded": False,
            "policy_forward_executed": False,
            "policy_generation_executed": False,
            "restoration_output_generated": False,
            "full_artifact_including_confirm_bytes_validated_by_loader": True,
            "confirm_state_prompt_or_image_exposed_to_decoder_or_processor": False,
            "confirm_processor_prompt_count": 0,
        },
        "reduction": reduction,
    }
    validation = validate_restoration_v2_1_processor_audit(
        audit,
        repository_root=root,
        current_git_commit=args.run_git_commit,
        mode="generation",
    )
    if collect_clean_pushed_main_identity(
        repository_root=root,
        run_git_commit=args.run_git_commit,
    ) != repository_identity:
        raise ValueError("clean pushed Git identity drifted during processor audit")
    if committed_source_inventory(
        repository_root=root,
        run_git_commit=args.run_git_commit,
    ) != source_inventory:
        raise ValueError("committed source identity drifted during processor audit")
    with output.open("xb") as destination:
        destination.write(canonical_json_bytes(audit) + b"\n")
    return audit, {**validation, "output": str(output)}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _, validation = run_processor_audit(args)
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
