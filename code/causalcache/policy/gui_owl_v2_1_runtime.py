"""Pinned official-tool GPU runtime for the restoration v2.1 rescue."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_MOBILE_USE_TOOL,
    GUI_OWL_V2_1_PROTOCOL_ID,
    ParsedGUIOwlV21Output,
    canonical_json_sha256,
    parse_gui_owl_v2_1_output,
    serialize_gui_owl_v2_1_teacher_target,
    validate_gui_owl_v2_1_native_messages,
    validate_gui_owl_v2_1_tool_schema,
)
from causalcache.policy.gui_owl_v2_runtime import (
    FROZEN_GUI_OWL_V2_DTYPE,
    FROZEN_GUI_OWL_V2_MAX_MICROBATCH_SIZE,
    FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
    GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
    GUIOwlV2Runtime,
    extend_gui_owl_v2_teacher_inputs,
    prompt_aligned_input_keys,
)


GUI_OWL_V2_1_EXPECTED_ASSISTANT_PREFIX_TOKEN_IDS = (151644, 77091, 198)
GUI_OWL_V2_1_EXPECTED_TOOL_CALL_OPEN_TOKEN_ID = 151657
GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID = 151658
GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS = (151645, 151643)
GUI_OWL_V2_1_EXPECTED_PAD_TOKEN_ID = 151643
GUI_OWL_V2_1_EXPECTED_CHAT_TEMPLATE_FILE_SHA256 = (
    "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce"
)
GUI_OWL_V2_1_EXPECTED_CHAT_TEMPLATE_TEXT_SHA256 = (
    "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
)


@dataclass(frozen=True)
class GUIOwlV21ChatTemplateIdentity:
    file_sha256: str
    text_sha256: str


@dataclass(frozen=True)
class GUIOwlV21GenerationTokens:
    assistant_prefix_token_ids: tuple[int, ...]
    tool_call_open_token_id: int
    tool_call_close_token_id: int
    standard_eos_token_ids: tuple[int, ...]
    pad_token_id: int


@dataclass(frozen=True)
class GUIOwlV21TeacherTokens:
    assistant_prefix_text: str
    distance_text: str
    assistant_prefix_token_ids: tuple[int, ...]
    distance_token_ids: tuple[int, ...]
    tool_call_open_token_id: int
    tool_call_close_token_id: int

    @property
    def forced_suffix_token_ids(self) -> tuple[int, ...]:
        return self.distance_token_ids[:-1]


@dataclass(frozen=True)
class GUIOwlV21TeacherLayout:
    prompt_input_tokens: int
    distance_action_tokens: int
    model_input_tokens: int
    distance_logit_positions: tuple[int, ...]
    forced_suffix_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class GUIOwlV21GenerationResult:
    output_text: str
    parsed_output: ParsedGUIOwlV21Output
    metadata: Mapping[str, Any]


class GUIOwlV21GenerationParseError(ValueError):
    """Strict parse failure preserving the native output and generation audit."""

    def __init__(
        self,
        *,
        output_text: str,
        metadata: Mapping[str, Any],
        parse_error: ValueError,
    ) -> None:
        super().__init__(f"strict GUI-Owl v2.1 official-tool parse failed: {parse_error}")
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


def _tokenize(tokenizer: Any, text: str, *, name: str) -> tuple[int, ...]:
    return _validated_token_ids(
        tokenizer.encode(text, add_special_tokens=False),
        name=name,
    )


def validate_gui_owl_v2_1_generation_tokens(
    tokenizer: Any,
    generation_config: Any,
) -> GUIOwlV21GenerationTokens:
    """Bind official tags and standard EOS handling to the pinned tokenizer."""
    assistant_prefix_ids = _tokenize(
        tokenizer,
        GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
        name="assistant_prefix_token_ids",
    )
    open_ids = _tokenize(tokenizer, "<tool_call>", name="tool_call_open_token_ids")
    close_ids = _tokenize(
        tokenizer,
        "</tool_call>",
        name="tool_call_close_token_ids",
    )
    if assistant_prefix_ids != GUI_OWL_V2_1_EXPECTED_ASSISTANT_PREFIX_TOKEN_IDS:
        raise ValueError("GUI-Owl v2.1 assistant prefix token ids drifted")
    if open_ids != (GUI_OWL_V2_1_EXPECTED_TOOL_CALL_OPEN_TOKEN_ID,):
        raise ValueError("GUI-Owl v2.1 tool-call opener must remain one pinned token")
    if close_ids != (GUI_OWL_V2_1_EXPECTED_TOOL_CALL_CLOSE_TOKEN_ID,):
        raise ValueError("GUI-Owl v2.1 tool-call closer must remain one pinned token")

    configured_eos = getattr(generation_config, "eos_token_id", None)
    if type(configured_eos) is int:
        standard_eos_ids = (configured_eos,)
    else:
        standard_eos_ids = _validated_token_ids(
            configured_eos,
            name="generation_config_eos_token_ids",
        )
    pad_token_id = getattr(generation_config, "pad_token_id", None)
    if standard_eos_ids != GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS:
        raise ValueError("GUI-Owl v2.1 standard EOS token ids drifted")
    if pad_token_id != GUI_OWL_V2_1_EXPECTED_PAD_TOKEN_ID:
        raise ValueError("GUI-Owl v2.1 pad token id drifted")
    if getattr(generation_config, "forced_eos_token_id", None) is not None:
        raise ValueError("GUI-Owl v2.1 forbids a configured forced EOS token")
    if getattr(generation_config, "stop_strings", None) is not None:
        raise ValueError("GUI-Owl v2.1 forbids configured stop strings")
    if getattr(tokenizer, "eos_token_id", None) != standard_eos_ids[0]:
        raise ValueError("GUI-Owl v2.1 tokenizer EOS differs from generation config")
    if getattr(tokenizer, "pad_token_id", None) != pad_token_id:
        raise ValueError("GUI-Owl v2.1 tokenizer pad differs from generation config")
    all_special_ids = tuple(int(value) for value in tokenizer.all_special_ids)
    if close_ids[0] in all_special_ids or open_ids[0] in all_special_ids:
        raise ValueError("GUI-Owl v2.1 tool tags must survive special-token decoding")
    if set(standard_eos_ids).intersection({open_ids[0], close_ids[0]}):
        raise ValueError("GUI-Owl v2.1 tool tags overlap the standard EOS inventory")
    if tokenizer.decode(close_ids, skip_special_tokens=False) != "</tool_call>":
        raise ValueError("GUI-Owl v2.1 decoder does not preserve the tool-call closer")
    return GUIOwlV21GenerationTokens(
        assistant_prefix_token_ids=assistant_prefix_ids,
        tool_call_open_token_id=open_ids[0],
        tool_call_close_token_id=close_ids[0],
        standard_eos_token_ids=standard_eos_ids,
        pad_token_id=pad_token_id,
    )


def validate_gui_owl_v2_1_chat_template(
    *,
    model_dir: str | Path,
    tokenizer: Any,
) -> GUIOwlV21ChatTemplateIdentity:
    """Bind the in-memory official template to the pinned snapshot file."""
    path = Path(model_dir).resolve() / "chat_template.json"
    payload = path.read_bytes()
    file_sha256 = hashlib.sha256(payload).hexdigest()
    if file_sha256 != GUI_OWL_V2_1_EXPECTED_CHAT_TEMPLATE_FILE_SHA256:
        raise ValueError("GUI-Owl v2.1 chat_template.json SHA256 drifted")
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("GUI-Owl v2.1 chat_template.json is invalid") from error
    if not isinstance(record, Mapping) or set(record) != {"chat_template"}:
        raise ValueError("GUI-Owl v2.1 chat template file schema drifted")
    template = record["chat_template"]
    if not isinstance(template, str) or template != getattr(
        tokenizer,
        "chat_template",
        None,
    ):
        raise ValueError("GUI-Owl v2.1 in-memory chat template differs from snapshot")
    text_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if text_sha256 != GUI_OWL_V2_1_EXPECTED_CHAT_TEMPLATE_TEXT_SHA256:
        raise ValueError("GUI-Owl v2.1 in-memory chat template SHA256 drifted")
    return GUIOwlV21ChatTemplateIdentity(
        file_sha256=file_sha256,
        text_sha256=text_sha256,
    )


def build_gui_owl_v2_1_teacher_tokens(
    tokenizer: Any,
    action: GUIOwlV2Action,
) -> GUIOwlV21TeacherTokens:
    """Tokenize the direct official tool-call target as one distance span."""
    if not isinstance(action, GUIOwlV2Action):
        raise TypeError("action must be a GUIOwlV2Action")
    distance_text = serialize_gui_owl_v2_1_teacher_target(action)
    assistant_prefix_ids = _tokenize(
        tokenizer,
        GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
        name="assistant_prefix_token_ids",
    )
    distance_ids = _tokenize(
        tokenizer,
        distance_text,
        name="distance_token_ids",
    )
    joint_ids = _tokenize(
        tokenizer,
        GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX + distance_text,
        name="assistant_and_distance_token_ids",
    )
    if joint_ids != (*assistant_prefix_ids, *distance_ids):
        raise ValueError(
            "tokenizer merges across the official assistant/tool-call boundary; "
            "the distance span would be ambiguous"
        )
    open_ids = _tokenize(tokenizer, "<tool_call>", name="tool_call_open_token_ids")
    close_ids = _tokenize(
        tokenizer,
        "</tool_call>",
        name="tool_call_close_token_ids",
    )
    if len(open_ids) != 1 or distance_ids[0] != open_ids[0]:
        raise ValueError("official teacher target must begin with one tool-call opener token")
    if len(close_ids) != 1 or distance_ids[-1] != close_ids[0]:
        raise ValueError("official teacher target must end with one tool-call closer token")
    if set(distance_ids).intersection(GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS):
        raise ValueError(
            "official teacher target contains a suppressed standard EOS token"
        )
    return GUIOwlV21TeacherTokens(
        assistant_prefix_text=GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
        distance_text=distance_text,
        assistant_prefix_token_ids=assistant_prefix_ids,
        distance_token_ids=distance_ids,
        tool_call_open_token_id=open_ids[0],
        tool_call_close_token_id=close_ids[0],
    )


def gui_owl_v2_1_teacher_layout(
    *,
    prompt_input_tokens: int,
    teacher_tokens: GUIOwlV21TeacherTokens,
) -> GUIOwlV21TeacherLayout:
    """Place the first official tool token directly after the assistant prefix."""
    if type(prompt_input_tokens) is not int or prompt_input_tokens <= 0:
        raise ValueError("prompt_input_tokens must be a positive integer")
    if not isinstance(teacher_tokens, GUIOwlV21TeacherTokens):
        raise TypeError("teacher_tokens must be GUIOwlV21TeacherTokens")
    distance_tokens = len(teacher_tokens.distance_token_ids)
    positions = tuple(
        range(prompt_input_tokens - 1, prompt_input_tokens - 1 + distance_tokens)
    )
    return GUIOwlV21TeacherLayout(
        prompt_input_tokens=prompt_input_tokens,
        distance_action_tokens=distance_tokens,
        model_input_tokens=prompt_input_tokens + distance_tokens - 1,
        distance_logit_positions=positions,
        forced_suffix_token_ids=teacher_tokens.forced_suffix_token_ids,
    )


def _official_tools_argument() -> list[dict[str, Any]]:
    validate_gui_owl_v2_1_tool_schema()
    return [copy.deepcopy(GUI_OWL_V2_1_MOBILE_USE_TOOL)]


def mask_gui_owl_v2_1_teacher_standard_eos_logits(
    logits: Any,
    *,
    standard_eos_token_ids: Sequence[int],
    tensor_module: Any,
) -> tuple[Any, float]:
    """Clone and finitely mask generation-forbidden EOS columns on the GPU."""
    token_ids = _validated_token_ids(
        standard_eos_token_ids,
        name="teacher_standard_eos_token_ids",
    )
    if getattr(logits, "ndim", None) != 3:
        raise ValueError("teacher logits must have rank three before EOS masking")
    vocabulary_size = int(logits.shape[2])
    if any(token_id >= vocabulary_size for token_id in token_ids):
        raise ValueError("teacher EOS mask token id exceeds the vocabulary")
    masked = logits.clone()
    mask_value = float(tensor_module.finfo(masked.dtype).min)
    if not math.isfinite(mask_value):
        raise RuntimeError("teacher EOS suppression value must remain finite")
    masked[..., list(token_ids)] = mask_value
    if masked.device != logits.device or masked.dtype != logits.dtype:
        raise RuntimeError("teacher EOS masking moved or cast the logits")
    if getattr(masked, "requires_grad", False):
        raise RuntimeError("teacher EOS masking unexpectedly enabled gradients")
    return masked, mask_value


class GUIOwlV21OfficialToolsRuntime(GUIOwlV2Runtime):
    """GUI-Owl runtime using only the pinned official ``tools=`` interface."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.chat_template_identity = validate_gui_owl_v2_1_chat_template(
            model_dir=self.runtime_identity.model_dir,
            tokenizer=self.processor.tokenizer,
        )
        self.generation_tokens = validate_gui_owl_v2_1_generation_tokens(
            self.processor.tokenizer,
            self.model.generation_config,
        )
        self.metadata = {
            **self.metadata,
            **self._interface_metadata(),
        }

    def _interface_metadata(self) -> dict[str, Any]:
        tokens = self.generation_tokens
        return {
            "protocol_id": GUI_OWL_V2_1_PROTOCOL_ID,
            "generation_interface": "processor_apply_chat_template_official_tools_kwarg",
            "official_tools_argument_count": 1,
            "official_tool_schema_sha256": canonical_json_sha256(
                GUI_OWL_V2_1_MOBILE_USE_TOOL
            ),
            "chat_template_file_sha256": self.chat_template_identity.file_sha256,
            "chat_template_text_sha256": self.chat_template_identity.text_sha256,
            "assistant_prefix_token_ids": list(tokens.assistant_prefix_token_ids),
            "tool_call_open_token_id": tokens.tool_call_open_token_id,
            "tool_call_close_token_id": tokens.tool_call_close_token_id,
            "generation_eos_token_id": tokens.tool_call_close_token_id,
            "suppressed_standard_eos_token_ids": list(tokens.standard_eos_token_ids),
            "generation_pad_token_id": tokens.pad_token_id,
            "generation_num_beams": 1,
            "generation_num_return_sequences": 1,
            "generation_standard_eos_suppression": (
                "negative_infinity_via_transformers_suppress_tokens"
            ),
            "host_injected_tool_call_closer": False,
            "output_recovery_or_normalization": False,
        }

    def _encode_exact_batch(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, Any]]],
    ) -> tuple[dict[str, Any], tuple[int, ...]]:
        if isinstance(messages_batch, (str, bytes, bytearray, Mapping)):
            raise TypeError("messages_batch must contain native GUI-Owl v2.1 conversations")
        conversations = tuple(messages_batch)
        if not conversations or len(conversations) > FROZEN_GUI_OWL_V2_MAX_MICROBATCH_SIZE:
            raise ValueError("GUI-Owl v2.1 microbatch size must be one or two")
        image_counts = tuple(
            validate_gui_owl_v2_1_native_messages(messages)
            for messages in conversations
        )
        if len(set(image_counts)) != 1:
            raise ValueError("GUI-Owl v2.1 batch requires equal image counts")
        encoded = self.processor.apply_chat_template(
            list(conversations),
            tools=_official_tools_argument(),
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
        assistant_ids = self.generation_tokens.assistant_prefix_token_ids
        if int(input_ids.shape[1]) < len(assistant_ids):
            raise ValueError("processor prompt is shorter than the official assistant prefix")
        observed_tail = (
            input_ids[:, -len(assistant_ids) :]
            .detach()
            .to(device="cpu")
            .tolist()
        )
        if observed_tail != [list(assistant_ids) for _ in conversations]:
            raise ValueError("official-tools prompt does not end with the pinned assistant prefix")
        return model_inputs, image_counts

    def prepare_native_message_shape(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        model_inputs, image_counts = self._encode_exact_batch((messages,))
        return {
            **self._image_metadata(model_inputs, image_counts)[0],
            **self._interface_metadata(),
            "sequence_length": int(model_inputs["input_ids"].shape[1]),
            "assistant_generation_prefix_tokens": len(
                self.generation_tokens.assistant_prefix_token_ids
            ),
            "policy_forward_executed": False,
        }

    def teacher_forced_distance_logits(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, Any]]],
        actions: Sequence[GUIOwlV2Action],
    ) -> tuple[Any, dict[str, Any]]:
        """Return BF16 logits for the complete official tool-call token span."""
        model_inputs, image_counts = self._encode_exact_batch(messages_batch)
        if isinstance(actions, (str, bytes, bytearray, Mapping)):
            raise TypeError("actions must be a sequence of GUIOwlV2Action values")
        canonical_actions = tuple(actions)
        batch_size = int(model_inputs["input_ids"].shape[0])
        if len(canonical_actions) != batch_size:
            raise ValueError("action count differs from the exact-shape message batch")
        teacher_tokens = tuple(
            build_gui_owl_v2_1_teacher_tokens(self.processor.tokenizer, action)
            for action in canonical_actions
        )
        targets = {tokens.distance_text for tokens in teacher_tokens}
        if len(targets) != 1:
            raise ValueError(
                "one decision-state microbatch must share one canonical reference action"
            )
        distance_lengths = {len(tokens.distance_token_ids) for tokens in teacher_tokens}
        if len(distance_lengths) != 1:
            raise ValueError("teacher-forced batch requires equal action token lengths")
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        layouts = tuple(
            gui_owl_v2_1_teacher_layout(
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
        raw_logits = outputs.logits
        if (
            getattr(raw_logits, "ndim", None) != 3
            or tuple(raw_logits.shape[:2]) != (batch_size, action_tokens)
        ):
            raise RuntimeError(
                "GUI-Owl v2.1 teacher logits must have shape "
                "[batch, distance_tokens, vocab]"
            )
        if (
            raw_logits.device != self.device
            or str(raw_logits.dtype) != FROZEN_GUI_OWL_V2_DTYPE
        ):
            raise RuntimeError(
                "GUI-Owl v2.1 teacher logits must remain BF16 on the selected GPU"
            )
        if getattr(raw_logits, "requires_grad", False):
            raise RuntimeError("GUI-Owl v2.1 teacher logits must not require gradients")
        logits, suppression_value = mask_gui_owl_v2_1_teacher_standard_eos_logits(
            raw_logits,
            standard_eos_token_ids=self.generation_tokens.standard_eos_token_ids,
            tensor_module=self.torch,
        )

        samples = []
        for image_record, layout in zip(
            self._image_metadata(model_inputs, image_counts),
            layouts,
            strict=True,
        ):
            samples.append(
                {
                    **image_record,
                    "teacher_carrier_tokens": 0,
                    "distance_action_tokens": layout.distance_action_tokens,
                    "model_input_tokens": layout.model_input_tokens,
                    "distance_logit_positions": list(layout.distance_logit_positions),
                }
            )
        return logits, {
            **self._interface_metadata(),
            "batch_size": batch_size,
            "device": str(logits.device),
            "dtype": str(logits.dtype),
            "vocabulary_size": int(logits.shape[2]),
            "distance_span": "official_tool_call_open_through_close_inclusive",
            "teacher_context": "official_tools_prompt_plus_assistant_prefix_direct",
            "teacher_carrier": None,
            "teacher_target_json_separators": [", ", ": "],
            "teacher_target_ends_with_model_generation_eos": True,
            "teacher_target_disjoint_from_suppressed_standard_eos": True,
            "teacher_standard_eos_suppressed_token_ids": list(
                self.generation_tokens.standard_eos_token_ids
            ),
            "teacher_standard_eos_suppression_value": suppression_value,
            "teacher_standard_eos_suppression_semantics": (
                "torch_finfo_bfloat16_min_finite_generation_alignment"
            ),
            "teacher_standard_eos_mask_application": (
                "same_mask_on_every_reference_and_candidate_action_path_position_"
                "before_float32_log_softmax"
            ),
            "teacher_raw_logits_mutated": False,
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

    def generate_native_action(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> GUIOwlV21GenerationResult:
        """Generate one strict official tool call without host-side completion."""
        model_inputs, image_counts = self._encode_exact_batch((messages,))
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        tokens = self.generation_tokens
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
                eos_token_id=tokens.tool_call_close_token_id,
                pad_token_id=tokens.pad_token_id,
                suppress_tokens=list(tokens.standard_eos_token_ids),
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=False,
            )
        self.torch.cuda.synchronize(self.device)
        latency_seconds = time.perf_counter() - started
        if generated.device != self.device or int(generated.shape[0]) != 1:
            raise RuntimeError("GUI-Owl v2.1 generated tokens left the single-item CUDA batch")
        new_tokens = generated[:, prompt_tokens:]
        generated_ids = new_tokens[0].detach().to(device="cpu").tolist()
        if not isinstance(generated_ids, list) or any(
            type(token_id) is not int for token_id in generated_ids
        ):
            raise TypeError("GUI-Owl v2.1 generated token ids must be an integer list")
        if any(token_id in tokens.standard_eos_token_ids for token_id in generated_ids):
            raise RuntimeError("GUI-Owl v2.1 generated a suppressed standard EOS token")
        closer_count = generated_ids.count(tokens.tool_call_close_token_id)
        if generated_ids and generated_ids[-1] == tokens.tool_call_close_token_id:
            if closer_count != 1:
                raise RuntimeError("GUI-Owl v2.1 generated more than one tool-call closer")
            termination_reason = "model_emitted_tool_call_close"
        elif len(generated_ids) == FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS:
            if closer_count != 0:
                raise RuntimeError("GUI-Owl v2.1 generated a non-terminal tool-call closer")
            termination_reason = "max_new_tokens_without_tool_call_close"
        else:
            raise RuntimeError(
                "GUI-Owl v2.1 generation terminated before close or max_new_tokens"
            )
        decoded = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
            raise TypeError("GUI-Owl v2.1 processor batch_decode must return a sequence")
        if len(decoded) != 1 or not isinstance(decoded[0], str):
            raise ValueError("GUI-Owl v2.1 generation must decode to exactly one string")
        output_text = decoded[0]
        if closer_count == 1 and not output_text.endswith("</tool_call>"):
            raise RuntimeError(
                "GUI-Owl v2.1 decoded output hid or moved the generated closer"
            )
        metadata = {
            **self._image_metadata(model_inputs, image_counts)[0],
            **self._interface_metadata(),
            "generated_tokens": len(generated_ids),
            "generated_token_ids_sha256": canonical_json_sha256(generated_ids),
            "decoded_output_utf8_sha256": hashlib.sha256(
                output_text.encode("utf-8")
            ).hexdigest(),
            "final_generated_token_id": generated_ids[-1] if generated_ids else None,
            "generated_tool_call_close_token_count": closer_count,
            "termination_reason": termination_reason,
            "model_emitted_tool_call_close": closer_count == 1,
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "return_dict_in_generate": False,
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
            parsed = parse_gui_owl_v2_1_output(output_text)
        except ValueError as error:
            raise GUIOwlV21GenerationParseError(
                output_text=output_text,
                metadata=metadata,
                parse_error=error,
            ) from error
        return GUIOwlV21GenerationResult(
            output_text=output_text,
            parsed_output=parsed,
            metadata=metadata,
        )
