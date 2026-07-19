"""Versioned GUI-Owl v2.1 runtime for caller-owned throughput measurement."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import (
    canonical_json_sha256,
    parse_gui_owl_v2_1_output,
)
from causalcache.policy.gui_owl_v2_1_runtime import (
    GUIOwlV21GenerationParseError,
    GUIOwlV21GenerationResult,
    GUIOwlV21OfficialToolsRuntime,
    build_gui_owl_v2_1_teacher_tokens,
    gui_owl_v2_1_teacher_layout,
    mask_gui_owl_v2_1_teacher_standard_eos_logits,
)
from causalcache.policy.gui_owl_v2_runtime import (
    FROZEN_GUI_OWL_V2_DTYPE,
    FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
    extend_gui_owl_v2_teacher_inputs,
)


GUI_OWL_V2_1_THROUGHPUT_CUDA_PEAK_MEASUREMENT_OWNERS = ("runtime", "caller")


def _cuda_peak_measurement_owner(value: object) -> str:
    if (
        type(value) is not str
        or value not in GUI_OWL_V2_1_THROUGHPUT_CUDA_PEAK_MEASUREMENT_OWNERS
    ):
        raise ValueError("cuda_peak_measurement_owner must be 'runtime' or 'caller'")
    return value


def _reset_cuda_peak_memory_stats(
    runtime: GUIOwlV21OfficialToolsRuntime,
    *,
    owner: str,
) -> None:
    if owner == "runtime":
        runtime.torch.cuda.reset_peak_memory_stats(runtime.device)


class GUIOwlV21ThroughputRuntime(GUIOwlV21OfficialToolsRuntime):
    """Official-tools runtime with an explicit CUDA peak-measurement owner."""

    def teacher_forced_distance_logits(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, Any]]],
        actions: Sequence[GUIOwlV2Action],
        *,
        cuda_peak_measurement_owner: str = "runtime",
    ) -> tuple[Any, dict[str, Any]]:
        """Return BF16 logits for the complete official tool-call token span."""
        cuda_peak_measurement_owner = _cuda_peak_measurement_owner(
            cuda_peak_measurement_owner
        )
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
        _reset_cuda_peak_memory_stats(
            self,
            owner=cuda_peak_measurement_owner,
        )
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
        *,
        cuda_peak_measurement_owner: str = "runtime",
    ) -> GUIOwlV21GenerationResult:
        """Generate one strict official tool call without host-side completion."""
        cuda_peak_measurement_owner = _cuda_peak_measurement_owner(
            cuda_peak_measurement_owner
        )
        model_inputs, image_counts = self._encode_exact_batch((messages,))
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        tokens = self.generation_tokens
        _reset_cuda_peak_memory_stats(
            self,
            owner=cuda_peak_measurement_owner,
        )
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
