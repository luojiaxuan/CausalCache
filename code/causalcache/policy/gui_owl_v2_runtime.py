"""Frozen GPU runtime for restoration-v2 GUI-Owl policy behavior."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2 import (
    GUI_OWL_V2_SYSTEM_PROMPT,
    GUI_OWL_V2_TEACHER_CARRIER,
    GUIOwlV2Action,
    ParsedGUIOwlV2Output,
    parse_gui_owl_v2_output,
    serialize_gui_owl_v2_tool_call,
)
from causalcache.policy.gui_owl_v2_vision import (
    VISION_PATCH_SIZE,
    VISION_SPATIAL_MERGE_SIZE,
    VerifiedVisionRuntimeIdentity,
    _validate_model_identity,
    canonical_image_grid_thw,
    verify_frozen_vision_runtime,
    visual_token_geometry,
)


FROZEN_GUI_OWL_V2_DTYPE = "torch.bfloat16"
FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS = 256
FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE = 2560
FROZEN_GUI_OWL_V2_MAX_MICROBATCH_SIZE = 2
GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX = "<|im_start|>assistant\n"
PROMPT_ALIGNED_FILL_VALUES = {
    "attention_mask": 1,
    "mm_token_type_ids": 0,
}


@dataclass(frozen=True)
class GUIOwlV2TeacherTokens:
    """Exact carrier context and tool-call-only distance tokens."""

    assistant_prefix_text: str
    carrier_text: str
    distance_text: str
    assistant_prefix_token_ids: tuple[int, ...]
    carrier_token_ids: tuple[int, ...]
    distance_token_ids: tuple[int, ...]

    @property
    def forced_suffix_token_ids(self) -> tuple[int, ...]:
        return (*self.carrier_token_ids, *self.distance_token_ids[:-1])


@dataclass(frozen=True)
class GUIOwlV2TeacherLayout:
    """Causal positions for one exact-shape teacher-forced input."""

    prompt_input_tokens: int
    carrier_tokens: int
    distance_action_tokens: int
    model_input_tokens: int
    distance_logit_positions: tuple[int, ...]
    forced_suffix_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class GUIOwlV2GenerationResult:
    """One deterministic native generation and its strict parsed action."""

    output_text: str
    parsed_output: ParsedGUIOwlV2Output
    metadata: Mapping[str, Any]


class GUIOwlV2GenerationParseError(ValueError):
    """Strict parse failure that preserves the generated text and run metadata."""

    def __init__(
        self,
        *,
        output_text: str,
        metadata: Mapping[str, Any],
        parse_error: ValueError,
    ) -> None:
        super().__init__(f"strict GUI-Owl v2 action parse failed: {parse_error}")
        self.output_text = output_text
        self.metadata = dict(metadata)
        self.parse_error_type = parse_error.__class__.__name__
        self.parse_error_message = str(parse_error)


def _validated_token_ids(values: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be a sequence of token ids")
    try:
        materialized = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of token ids") from error
    if not materialized:
        raise ValueError(f"{name} must not be empty")
    if any(type(value) is not int or value < 0 for value in materialized):
        raise ValueError(f"{name} must contain non-negative integer token ids")
    return materialized


def _tokenize_without_added_special_tokens(
    tokenizer: Any,
    text: str,
    *,
    name: str,
) -> tuple[int, ...]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return _validated_token_ids(token_ids, name=name)


def build_gui_owl_v2_teacher_tokens(
    tokenizer: Any,
    action: GUIOwlV2Action,
) -> GUIOwlV2TeacherTokens:
    """Tokenize the fixed carrier and canonical tool call as disjoint spans."""
    if not isinstance(action, GUIOwlV2Action):
        raise TypeError("action must be a GUIOwlV2Action")
    distance_text = serialize_gui_owl_v2_tool_call(action)
    assistant_prefix_ids = _tokenize_without_added_special_tokens(
        tokenizer,
        GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
        name="assistant_prefix_token_ids",
    )
    carrier_ids = _tokenize_without_added_special_tokens(
        tokenizer,
        GUI_OWL_V2_TEACHER_CARRIER,
        name="carrier_token_ids",
    )
    distance_ids = _tokenize_without_added_special_tokens(
        tokenizer,
        distance_text,
        name="distance_token_ids",
    )
    joint_ids = _tokenize_without_added_special_tokens(
        tokenizer,
        GUI_OWL_V2_TEACHER_CARRIER + distance_text,
        name="joint_teacher_token_ids",
    )
    if joint_ids != (*carrier_ids, *distance_ids):
        raise ValueError(
            "tokenizer merges across the frozen carrier/tool-call boundary; "
            "the distance span would be ambiguous"
        )
    assistant_and_carrier_ids = _tokenize_without_added_special_tokens(
        tokenizer,
        GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX + GUI_OWL_V2_TEACHER_CARRIER,
        name="assistant_and_carrier_token_ids",
    )
    if assistant_and_carrier_ids != (*assistant_prefix_ids, *carrier_ids):
        raise ValueError(
            "tokenizer merges across the processor assistant-prefix/carrier boundary; "
            "the teacher context would be ambiguous"
        )
    return GUIOwlV2TeacherTokens(
        assistant_prefix_text=GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
        carrier_text=GUI_OWL_V2_TEACHER_CARRIER,
        distance_text=distance_text,
        assistant_prefix_token_ids=assistant_prefix_ids,
        carrier_token_ids=carrier_ids,
        distance_token_ids=distance_ids,
    )


def gui_owl_v2_teacher_layout(
    *,
    prompt_input_tokens: int,
    teacher_tokens: GUIOwlV2TeacherTokens,
) -> GUIOwlV2TeacherLayout:
    """Place distance logits after the assistant prefix and fixed carrier."""
    if type(prompt_input_tokens) is not int or prompt_input_tokens <= 0:
        raise ValueError("prompt_input_tokens must be a positive integer")
    if not isinstance(teacher_tokens, GUIOwlV2TeacherTokens):
        raise TypeError("teacher_tokens must be GUIOwlV2TeacherTokens")
    carrier_tokens = len(teacher_tokens.carrier_token_ids)
    distance_tokens = len(teacher_tokens.distance_token_ids)
    forced_suffix = teacher_tokens.forced_suffix_token_ids
    first_distance_logit = prompt_input_tokens + carrier_tokens - 1
    positions = tuple(
        range(first_distance_logit, first_distance_logit + distance_tokens)
    )
    return GUIOwlV2TeacherLayout(
        prompt_input_tokens=prompt_input_tokens,
        carrier_tokens=carrier_tokens,
        distance_action_tokens=distance_tokens,
        model_input_tokens=prompt_input_tokens + len(forced_suffix),
        distance_logit_positions=positions,
        forced_suffix_token_ids=forced_suffix,
    )


def validate_gui_owl_v2_native_messages(messages: Any) -> int:
    """Validate the native two-message envelope and return its image count."""
    if isinstance(messages, (str, bytes, bytearray, Mapping)):
        raise TypeError("GUI-Owl v2 messages must be a two-message sequence")
    try:
        records = tuple(messages)
    except TypeError as error:
        raise TypeError("GUI-Owl v2 messages must be a two-message sequence") from error
    if len(records) != 2 or any(not isinstance(record, Mapping) for record in records):
        raise ValueError("GUI-Owl v2 requires exactly one system and one user message")
    system, user = records
    if set(system) != {"role", "content"} or system["role"] != "system":
        raise ValueError("GUI-Owl v2 system message envelope drifted")
    expected_system_content = [{"type": "text", "text": GUI_OWL_V2_SYSTEM_PROMPT}]
    if system["content"] != expected_system_content:
        raise ValueError("GUI-Owl v2 system prompt drifted")
    if set(user) != {"role", "content"} or user["role"] != "user":
        raise ValueError("GUI-Owl v2 user message envelope drifted")
    content = user["content"]
    if isinstance(content, (str, bytes, bytearray, Mapping)):
        raise TypeError("GUI-Owl v2 user content must be a non-empty block sequence")
    try:
        blocks = tuple(content)
    except TypeError as error:
        raise TypeError("GUI-Owl v2 user content must be a non-empty block sequence") from error
    if not blocks or any(not isinstance(block, Mapping) for block in blocks):
        raise ValueError("GUI-Owl v2 user content must contain mapping blocks")
    image_count = 0
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            if set(block) != {"type", "text"} or not isinstance(block["text"], str):
                raise ValueError("GUI-Owl v2 text block schema drifted")
        elif block_type == "image":
            if set(block) != {"type", "image"}:
                raise ValueError("GUI-Owl v2 image block schema drifted")
            image_count += 1
        else:
            raise ValueError("GUI-Owl v2 messages allow only text and image blocks")
    if image_count <= 0:
        raise ValueError("GUI-Owl v2 messages require the current observation image")
    return image_count


def prompt_aligned_input_keys(model_inputs: Mapping[str, Any]) -> tuple[str, ...]:
    """Identify only the frozen prompt-aligned processor tensor inventory."""
    if "input_ids" not in model_inputs:
        raise ValueError("processor output is missing input_ids")
    input_ids = model_inputs["input_ids"]
    if getattr(input_ids, "ndim", None) != 2:
        raise ValueError("processor input_ids must have rank 2 [batch, sequence]")
    input_shape = tuple(input_ids.shape)
    aligned: list[str] = []
    for key, value in model_inputs.items():
        shape = getattr(value, "shape", None)
        if shape is None:
            continue
        value_shape = tuple(shape)
        is_prompt_aligned = len(value_shape) >= 2 and value_shape[:2] == input_shape
        if not is_prompt_aligned:
            continue
        if key == "input_ids":
            continue
        if key not in PROMPT_ALIGNED_FILL_VALUES:
            raise ValueError(f"unknown prompt-aligned processor tensor: {key}")
        if value_shape != input_shape:
            raise ValueError(f"{key} must have the same rank-2 shape as input_ids")
        aligned.append(key)
    if "attention_mask" not in aligned:
        raise ValueError("processor output is missing prompt-aligned attention_mask")
    return tuple(sorted(aligned))


def extend_gui_owl_v2_teacher_inputs(
    model_inputs: Mapping[str, Any],
    forced_suffix_token_ids: Sequence[Sequence[int]],
    *,
    tensor_module: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Append exact-shape teacher suffixes without padding or CPU materialization."""
    aligned_keys = prompt_aligned_input_keys(model_inputs)
    input_ids = model_inputs["input_ids"]
    batch_size = int(input_ids.shape[0])
    if isinstance(forced_suffix_token_ids, (str, bytes, bytearray, Mapping)):
        raise TypeError("forced suffixes must be a batch of token-id sequences")
    rows = tuple(
        _validated_token_ids(row, name="forced_suffix_token_ids")
        for row in forced_suffix_token_ids
    )
    if len(rows) != batch_size:
        raise ValueError("forced suffix batch size differs from processor input batch")
    suffix_length = len(rows[0])
    if any(len(row) != suffix_length for row in rows):
        raise ValueError("teacher-forced batch requires equal suffix token lengths")
    suffix = tensor_module.tensor(
        rows,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    extended = dict(model_inputs)
    extended["input_ids"] = tensor_module.cat((input_ids, suffix), dim=1)
    for key in aligned_keys:
        value = model_inputs[key]
        fill = tensor_module.full(
            (batch_size, suffix_length),
            PROMPT_ALIGNED_FILL_VALUES[key],
            dtype=value.dtype,
            device=value.device,
        )
        extended[key] = tensor_module.cat((value, fill), dim=1)
    return extended, aligned_keys


def _runtime_identity_metadata(identity: VerifiedVisionRuntimeIdentity) -> dict[str, Any]:
    return {
        "model_dir": identity.model_dir,
        "model_repo": identity.model_repo,
        "model_revision": identity.model_revision,
        "snapshot_manifest_sha256": identity.snapshot_manifest_sha256,
        "verified_model_file_count": identity.verified_model_file_count,
        "verified_model_total_bytes": identity.verified_model_total_bytes,
        "transformers_version": identity.transformers_version,
        "transformers_source_sha256": dict(identity.transformers_source_sha256),
    }


class GUIOwlV2Runtime:
    """Pinned GUI-Owl Instruct runtime for generation and GPU teacher logits."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        expected_snapshot_manifest: str | Path,
        device: str,
        target_effective_visual_tokens_per_image: int,
    ) -> None:
        if re.fullmatch(r"cuda:[0-9]+", device) is None:
            raise ValueError("GUI-Owl v2 requires one explicit CUDA device such as cuda:0")
        if (
            target_effective_visual_tokens_per_image
            != FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ):
            raise ValueError("GUI-Owl v2 effective visual token target is frozen to 2560")
        identity = verify_frozen_vision_runtime(
            model_dir=model_dir,
            expected_snapshot_manifest=expected_snapshot_manifest,
        )

        try:
            import torch
            import transformers
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as error:
            raise RuntimeError("GUI-Owl v2 runtime requires PyTorch and Transformers") from error

        selected_device = torch.device(device)
        if not torch.cuda.is_available() or selected_device.index is None:
            raise RuntimeError("GUI-Owl v2 requires an available explicit CUDA device")
        if selected_device.index >= torch.cuda.device_count():
            raise ValueError("selected GUI-Owl v2 CUDA device does not exist")
        pixels_per_image = target_effective_visual_tokens_per_image * (
            VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE
        ) ** 2
        processor = AutoProcessor.from_pretrained(
            identity.model_dir,
            min_pixels=pixels_per_image,
            max_pixels=pixels_per_image,
            local_files_only=True,
        )
        if int(processor.image_processor.merge_size) != VISION_SPATIAL_MERGE_SIZE:
            raise ValueError("GUI-Owl v2 processor spatial merge size drifted")
        model = AutoModelForImageTextToText.from_pretrained(
            identity.model_dir,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(selected_device)
        model.eval().requires_grad_(False)
        _validate_model_identity(model, identity)
        for parameter in model.parameters():
            if parameter.device != selected_device:
                raise RuntimeError("GUI-Owl v2 model spans more than the selected CUDA device")
            if parameter.is_floating_point() and parameter.dtype is not torch.bfloat16:
                raise RuntimeError("GUI-Owl v2 floating parameters must remain bfloat16")
            if parameter.requires_grad:
                raise RuntimeError("GUI-Owl v2 model parameters must remain frozen")

        self.torch = torch
        self.device = selected_device
        self.processor = processor
        self.model = model
        self.runtime_identity = identity
        self.metadata = {
            **_runtime_identity_metadata(identity),
            "dtype": FROZEN_GUI_OWL_V2_DTYPE,
            "device": str(selected_device),
            "frozen": True,
            "single_device": True,
            "processor_class": processor.__class__.__name__,
            "model_class": model.__class__.__name__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "target_effective_visual_tokens_per_image": (
                target_effective_visual_tokens_per_image
            ),
            "min_pixels": pixels_per_image,
            "max_pixels": pixels_per_image,
        }

    def maximum_context_tokens(self) -> int:
        """Return the verified model text context limit without a policy forward."""
        text_config = getattr(self.model.config, "text_config", None)
        value = getattr(text_config, "max_position_embeddings", None)
        if type(value) is not int or value <= 0:
            raise ValueError(
                "verified GUI-Owl text config lacks a positive max_position_embeddings"
            )
        return value

    def _encode_exact_batch(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, Any]]],
    ) -> tuple[dict[str, Any], tuple[int, ...]]:
        if isinstance(messages_batch, (str, bytes, bytearray, Mapping)):
            raise TypeError("messages_batch must contain native GUI-Owl v2 conversations")
        conversations = tuple(messages_batch)
        if not conversations or len(conversations) > FROZEN_GUI_OWL_V2_MAX_MICROBATCH_SIZE:
            raise ValueError("GUI-Owl v2 microbatch size must be one or two")
        image_counts = tuple(
            validate_gui_owl_v2_native_messages(messages) for messages in conversations
        )
        if len(set(image_counts)) != 1:
            raise ValueError("GUI-Owl v2 batch requires equal image counts")
        encoded = self.processor.apply_chat_template(
            list(conversations),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=False,
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(self.device)
        else:
            encoded = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
        model_inputs = dict(encoded)
        prompt_aligned_input_keys(model_inputs)
        input_ids = model_inputs["input_ids"]
        if int(input_ids.shape[0]) != len(conversations):
            raise ValueError("processor batch dimension differs from conversation count")
        for key, value in model_inputs.items():
            value_device = getattr(value, "device", None)
            if value_device is not None and value_device != self.device:
                raise RuntimeError(f"processor tensor {key} left the selected CUDA device")
        assistant_prefix_ids = _tokenize_without_added_special_tokens(
            self.processor.tokenizer,
            GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
            name="assistant_prefix_token_ids",
        )
        if int(input_ids.shape[1]) < len(assistant_prefix_ids):
            raise ValueError("processor prompt is shorter than the native assistant prefix")
        prefix_tail = (
            input_ids[:, -len(assistant_prefix_ids) :]
            .detach()
            .to(device="cpu")
            .tolist()
        )
        expected_tail = [list(assistant_prefix_ids) for _ in conversations]
        if prefix_tail != expected_tail:
            raise ValueError("processor prompt does not end with the frozen assistant prefix")
        return model_inputs, image_counts

    def prepare_native_message_shape(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Prepare one processor input and expose its exact pre-forward shape."""
        model_inputs, image_counts = self._encode_exact_batch((messages,))
        image_metadata = self._image_metadata(model_inputs, image_counts)[0]
        return {
            **image_metadata,
            "sequence_length": int(model_inputs["input_ids"].shape[1]),
            "assistant_generation_prefix_tokens": len(
                _tokenize_without_added_special_tokens(
                    self.processor.tokenizer,
                    GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
                    name="assistant_prefix_token_ids",
                )
            ),
            "policy_forward_executed": False,
        }

    def _image_metadata(
        self,
        model_inputs: Mapping[str, Any],
        image_counts: Sequence[int],
    ) -> tuple[dict[str, Any], ...]:
        if "image_grid_thw" not in model_inputs:
            raise ValueError("processor output is missing image_grid_thw")
        grids = canonical_image_grid_thw(model_inputs["image_grid_thw"])
        if sum(image_counts) != len(grids):
            raise ValueError("image_grid_thw count differs from message image inventory")
        result: list[dict[str, Any]] = []
        offset = 0
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        for image_count in image_counts:
            sample_grids = grids[offset : offset + image_count]
            offset += image_count
            geometry = visual_token_geometry(sample_grids)
            merged = geometry["merged_token_counts"]
            assert isinstance(merged, tuple)
            effective_tokens = sum(merged)
            text_tokens = prompt_tokens - effective_tokens
            if text_tokens <= 0:
                raise ValueError("policy-visible text token count must be positive")
            result.append(
                {
                    "image_count": image_count,
                    "image_grid_thw": [list(grid) for grid in sample_grids],
                    "effective_visual_tokens": effective_tokens,
                    "policy_visible_text_tokens": text_tokens,
                    "prompt_input_tokens": prompt_tokens,
                }
            )
        return tuple(result)

    def teacher_forced_distance_logits(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, Any]]],
        actions: Sequence[GUIOwlV2Action],
    ) -> tuple[Any, dict[str, Any]]:
        """Return GPU BF16 logits only for the canonical tool-call distance span."""
        model_inputs, image_counts = self._encode_exact_batch(messages_batch)
        if isinstance(actions, (str, bytes, bytearray, Mapping)):
            raise TypeError("actions must be a sequence of GUIOwlV2Action values")
        canonical_actions = tuple(actions)
        batch_size = int(model_inputs["input_ids"].shape[0])
        if len(canonical_actions) != batch_size:
            raise ValueError("action count differs from the exact-shape message batch")
        teacher_tokens = tuple(
            build_gui_owl_v2_teacher_tokens(self.processor.tokenizer, action)
            for action in canonical_actions
        )
        canonical_reference_actions = {
            tokens.distance_text for tokens in teacher_tokens
        }
        if len(canonical_reference_actions) != 1:
            raise ValueError(
                "one decision-state microbatch must share one canonical reference action"
            )
        carrier_ids = {tokens.carrier_token_ids for tokens in teacher_tokens}
        distance_lengths = {len(tokens.distance_token_ids) for tokens in teacher_tokens}
        if len(carrier_ids) != 1 or len(distance_lengths) != 1:
            raise ValueError(
                "teacher-forced batch requires equal carrier and action token lengths"
            )
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        layouts = tuple(
            gui_owl_v2_teacher_layout(
                prompt_input_tokens=prompt_tokens,
                teacher_tokens=tokens,
            )
            for tokens in teacher_tokens
        )
        extended, aligned_keys = extend_gui_owl_v2_teacher_inputs(
            model_inputs,
            tuple(layout.forced_suffix_token_ids for layout in layouts),
            tensor_module=self.torch,
        )
        action_tokens = layouts[0].distance_action_tokens
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(
                **extended,
                use_cache=False,
                return_dict=True,
                logits_to_keep=action_tokens,
            )
        self.torch.cuda.synchronize(self.device)
        latency_seconds = time.perf_counter() - started
        logits = outputs.logits
        expected_shape = (batch_size, action_tokens)
        if getattr(logits, "ndim", None) != 3 or tuple(logits.shape[:2]) != expected_shape:
            raise RuntimeError(
                "GUI-Owl v2 teacher logits must have shape [batch, distance_tokens, vocab]"
            )
        if logits.device != self.device or str(logits.dtype) != FROZEN_GUI_OWL_V2_DTYPE:
            raise RuntimeError("GUI-Owl v2 teacher logits must remain BF16 on the selected GPU")
        if getattr(logits, "requires_grad", False):
            raise RuntimeError("GUI-Owl v2 teacher logits must not require gradients")

        samples = []
        for image_record, layout in zip(
            self._image_metadata(model_inputs, image_counts),
            layouts,
            strict=True,
        ):
            samples.append(
                {
                    **image_record,
                    "teacher_carrier_tokens": layout.carrier_tokens,
                    "distance_action_tokens": layout.distance_action_tokens,
                    "model_input_tokens": layout.model_input_tokens,
                    "distance_logit_positions": list(layout.distance_logit_positions),
                }
            )
        metadata = {
            "batch_size": batch_size,
            "device": str(logits.device),
            "dtype": str(logits.dtype),
            "vocabulary_size": int(logits.shape[2]),
            "distance_span": "tool_call_open_through_tool_call_close_inclusive",
            "teacher_context": "processor_assistant_prefix_plus_exact_fixed_carrier",
            "teacher_carrier": GUI_OWL_V2_TEACHER_CARRIER,
            "finite_logits_validation": (
                "deferred_to_gpu_kl_invalid_to_nan_final_distance"
            ),
            "extended_prompt_aligned_inputs": list(aligned_keys),
            "logits_to_keep": action_tokens,
            "latency_seconds": latency_seconds,
            "peak_gpu_memory_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated(self.device)
            ),
            "peak_gpu_memory_reserved_bytes": int(
                self.torch.cuda.max_memory_reserved(self.device)
            ),
            "full_logit_tensor_host_transfers": 0,
            "samples": samples,
        }
        return logits, metadata

    def generate_native_action(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> GUIOwlV2GenerationResult:
        """Generate once with the frozen deterministic settings and parse strictly."""
        model_inputs, image_counts = self._encode_exact_batch((messages,))
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
            )
        self.torch.cuda.synchronize(self.device)
        latency_seconds = time.perf_counter() - started
        if generated.device != self.device or int(generated.shape[0]) != 1:
            raise RuntimeError("GUI-Owl v2 generated tokens left the single-item CUDA batch")
        new_tokens = generated[:, prompt_tokens:]
        decoded = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
            raise TypeError("GUI-Owl v2 processor batch_decode must return a sequence")
        if len(decoded) != 1 or not isinstance(decoded[0], str):
            raise ValueError("GUI-Owl v2 generation must decode to exactly one string")
        output_text = decoded[0]
        metadata = {
            **self._image_metadata(model_inputs, image_counts)[0],
            "generated_tokens": int(new_tokens.shape[1]),
            "do_sample": False,
            "max_new_tokens": FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
            "latency_seconds": latency_seconds,
            "peak_gpu_memory_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated(self.device)
            ),
            "peak_gpu_memory_reserved_bytes": int(
                self.torch.cuda.max_memory_reserved(self.device)
            ),
            "full_logit_tensor_host_transfers": 0,
        }
        try:
            parsed = parse_gui_owl_v2_output(output_text)
        except ValueError as error:
            raise GUIOwlV2GenerationParseError(
                output_text=output_text,
                metadata=metadata,
                parse_error=error,
            ) from error
        return GUIOwlV2GenerationResult(
            output_text=output_text,
            parsed_output=parsed,
            metadata=metadata,
        )
