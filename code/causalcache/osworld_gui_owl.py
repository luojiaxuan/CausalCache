"""Frozen GUI-Owl prompt, action, and inference adapter for OSWorld."""

from __future__ import annotations

import base64
import io
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.osworld import DesktopAction
from causalcache.policy.gui_owl_v2_vision import (
    VISION_PATCH_SIZE,
    VISION_SPATIAL_MERGE_SIZE,
    _validate_model_identity,
    verify_frozen_vision_runtime,
)


GUI_OWL_OSWORLD_RUNTIME_PROFILE_ID = "gui_owl_1_5_osworld_recent_b4_v1"
GUI_OWL_OSWORLD_DEFAULT_VISUAL_TOKENS = 480
GUI_OWL_OSWORLD_DEFAULT_MAX_NEW_TOKENS = 256

_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": "Use the visible desktop GUI and return one executable action.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "key",
                        "type",
                        "mouse_move",
                        "left_click",
                        "left_click_drag",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "scroll",
                        "hscroll",
                        "wait",
                        "terminate",
                        "answer",
                    ],
                },
                "keys": {"type": "array", "items": {"type": "string"}},
                "text": {"type": "string"},
                "coordinate": {"type": "array", "items": {"type": "integer"}},
                "coordinate2": {"type": "array", "items": {"type": "integer"}},
                "pixels": {"type": "integer"},
                "time": {"type": "number"},
                "status": {"type": "string", "enum": ["success", "failure"]},
            },
            "required": ["action"],
        },
    },
}

GUI_OWL_OSWORLD_SYSTEM_PROMPT = """# Tools

You are provided with one function signature within <tools></tools> XML tags:
<tools>
{tool_spec}
</tools>

Return exactly two parts: one short line beginning with `Action:`, followed by one
<tool_call> block containing JSON with name `computer_use` and its arguments. Coordinates
are integer values in [0, 999], normalized across the current screenshot. Use only the
visible UI. Return no thinking block or other prose.
""".format(tool_spec=json.dumps(_TOOL_SPEC, separators=(",", ":")))


def _decode_png(encoded: str) -> Any:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("GUI-Owl screenshot must be non-empty base64 text")
    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
    image.load()
    return image.convert("RGB")


def _compact_history(history: Sequence[Mapping[str, Any]]) -> str:
    records = []
    for event in history:
        records.append(
            {
                "step_id": event["step_id"],
                "action": event["action"],
                "result_status": event["result_status"],
                "screen_changed": event["screen_changed"],
            }
        )
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def build_gui_owl_osworld_messages(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build a stateless mixed-fidelity GUI-Owl conversation from one policy request."""
    task = request.get("task")
    history = request.get("history")
    selected = request.get("selected_event_step_ids")
    if not isinstance(task, Mapping) or not isinstance(task.get("instruction"), str):
        raise ValueError("OSWorld policy request is missing the task instruction")
    if not isinstance(history, list) or not isinstance(selected, list):
        raise ValueError("OSWorld policy request is missing history selection")
    selected_ids = set(selected)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Generate the next desktop action for this task.\n"
                f"Instruction: {task['instruction']}\n"
                f"Previous event summaries: {_compact_history(history)}"
            ),
        }
    ]
    observed_selected: set[int] = set()
    for event in history:
        step_id = event.get("step_id")
        if step_id not in selected_ids:
            continue
        encoded = event.get("restored_post_screenshot_png_base64")
        if encoded is None:
            raise ValueError("selected history event lacks its restored screenshot")
        content.extend(
            [
                {"type": "text", "text": f"Restored screenshot after step {step_id}:"},
                {"type": "image", "image": _decode_png(encoded)},
            ]
        )
        observed_selected.add(step_id)
    if observed_selected != selected_ids:
        raise ValueError("selected history identities do not match restored screenshots")
    content.extend(
        [
            {"type": "text", "text": "Current screenshot:"},
            {"type": "image", "image": _decode_png(request["current_screenshot_png_base64"])},
        ]
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_OSWORLD_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]


def _normalized_coordinate(
    value: Any, *, screen_size: tuple[int, int]
) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) not in (int, float) for item in value)
    ):
        raise ValueError("coordinate must contain two numbers")
    if any(float(item) < 0 or float(item) > 999 for item in value):
        raise ValueError("normalized coordinate must fall within [0, 999]")
    width, height = screen_size
    return (
        round(float(value[0]) / 999 * (width - 1)),
        round(float(value[1]) / 999 * (height - 1)),
    )


def parse_gui_owl_osworld_action(
    output_text: str, *, screen_size: tuple[int, int]
) -> DesktopAction:
    """Parse one GUI-Owl computer-use tool call into the typed OSWorld contract."""
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output_text, re.DOTALL)
    if len(matches) != 1:
        raise ValueError("GUI-Owl output must contain exactly one tool_call block")
    payload = json.loads(matches[0])
    if not isinstance(payload, Mapping) or payload.get("name") != "computer_use":
        raise ValueError("GUI-Owl tool call must target computer_use")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping) or not isinstance(arguments.get("action"), str):
        raise ValueError("GUI-Owl tool call lacks action arguments")
    action = arguments["action"]
    if action in {"left_click", "double_click", "right_click", "middle_click"}:
        x, y = _normalized_coordinate(arguments.get("coordinate"), screen_size=screen_size)
        mapping: dict[str, Any] = {"type": "click", "x": x, "y": y}
        if action == "double_click":
            mapping["type"] = "double_click"
        elif action == "right_click":
            mapping["type"] = "right_click"
        elif action == "middle_click":
            mapping["button"] = "middle"
        return DesktopAction.from_mapping(mapping, screen_size=screen_size)
    if action == "mouse_move":
        x, y = _normalized_coordinate(arguments.get("coordinate"), screen_size=screen_size)
        return DesktopAction.from_mapping(
            {"type": "move", "x": x, "y": y}, screen_size=screen_size
        )
    if action == "left_click_drag":
        destination = arguments.get("coordinate2", arguments.get("coordinate"))
        x, y = _normalized_coordinate(destination, screen_size=screen_size)
        return DesktopAction.from_mapping(
            {"type": "drag", "x": x, "y": y}, screen_size=screen_size
        )
    if action == "type":
        return DesktopAction.from_mapping(
            {"type": "type_text", "text": arguments.get("text")},
            screen_size=screen_size,
        )
    if action == "key":
        keys = arguments.get("keys")
        if not isinstance(keys, list) or not keys:
            raise ValueError("key action requires a non-empty keys list")
        mapping = (
            {"type": "press", "key": keys[0]}
            if len(keys) == 1
            else {"type": "hotkey", "keys": keys}
        )
        return DesktopAction.from_mapping(mapping, screen_size=screen_size)
    if action in {"scroll", "hscroll"}:
        pixels = arguments.get("pixels")
        if type(pixels) is not int or pixels == 0:
            raise ValueError("scroll action requires a non-zero integer pixels value")
        mapping = (
            {"type": "scroll", "dy": pixels}
            if action == "scroll"
            else {"type": "scroll", "dy": 0, "dx": pixels}
        )
        return DesktopAction.from_mapping(mapping, screen_size=screen_size)
    if action == "wait":
        return DesktopAction.from_mapping({"type": "wait"}, screen_size=screen_size)
    if action == "terminate":
        terminal = "done" if arguments.get("status") == "success" else "fail"
        return DesktopAction.from_mapping({"type": terminal}, screen_size=screen_size)
    if action == "answer":
        return DesktopAction.from_mapping({"type": "done"}, screen_size=screen_size)
    raise ValueError(f"unsupported GUI-Owl OSWorld action: {action!r}")


class GUIOwlOSWorldRuntime:
    """One frozen GUI-Owl replica for stateless OSWorld action generation."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        expected_snapshot_manifest: str | Path,
        device: str,
        effective_visual_tokens_per_image: int,
        max_new_tokens: int,
    ) -> None:
        if effective_visual_tokens_per_image <= 0 or max_new_tokens <= 0:
            raise ValueError("visual and generation token limits must be positive")
        identity = verify_frozen_vision_runtime(
            model_dir=model_dir,
            expected_snapshot_manifest=expected_snapshot_manifest,
        )
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        selected_device = torch.device(device)
        pixels = effective_visual_tokens_per_image * (
            VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE
        ) ** 2
        processor = AutoProcessor.from_pretrained(
            identity.model_dir,
            min_pixels=pixels,
            max_pixels=pixels,
            local_files_only=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            identity.model_dir,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(selected_device)
        model.eval().requires_grad_(False)
        _validate_model_identity(model, identity)
        self.torch = torch
        self.processor = processor
        self.model = model
        self.device = selected_device
        self.max_new_tokens = max_new_tokens
        self.metadata = {
            "runtime_profile_id": GUI_OWL_OSWORLD_RUNTIME_PROFILE_ID,
            "model_repo": identity.model_repo,
            "model_revision": identity.model_revision,
            "device": str(selected_device),
            "dtype": "torch.bfloat16",
            "frozen": True,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "effective_visual_tokens_per_image": effective_visual_tokens_per_image,
            "max_new_tokens": max_new_tokens,
        }

    def generate(self, request: Mapping[str, Any]) -> tuple[DesktopAction, dict[str, Any]]:
        messages = build_gui_owl_osworld_messages(request)
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        self.torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - started
        new_tokens = generated[:, prompt_tokens:]
        output_text = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        action = parse_gui_owl_osworld_action(
            output_text, screen_size=tuple(request["screen_size"])
        )
        return action, {
            **self.metadata,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(new_tokens.shape[1]),
            "generation_seconds": generation_seconds,
            "image_count": 1 + len(request["selected_event_step_ids"]),
            "peak_gpu_memory_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated(self.device)
            ),
            "output_text": output_text,
        }


__all__ = [
    "GUIOwlOSWorldRuntime",
    "GUI_OWL_OSWORLD_DEFAULT_MAX_NEW_TOKENS",
    "GUI_OWL_OSWORLD_DEFAULT_VISUAL_TOKENS",
    "GUI_OWL_OSWORLD_RUNTIME_PROFILE_ID",
    "build_gui_owl_osworld_messages",
    "parse_gui_owl_osworld_action",
]
