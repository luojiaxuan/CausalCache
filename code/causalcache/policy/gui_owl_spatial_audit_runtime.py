"""Isolated runtime profiles for the spatial-reference numerical audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import parse_gui_owl_v2_1_output
from causalcache.policy.gui_owl_v2_1_runtime import (
    GUIOwlV21GenerationParseError,
    GUIOwlV21GenerationResult,
    GUIOwlV21OfficialToolsRuntime,
    build_gui_owl_v2_1_teacher_tokens,
    gui_owl_v2_1_teacher_layout,
    mask_gui_owl_v2_1_teacher_standard_eos_logits,
    validate_gui_owl_v2_1_chat_template,
    validate_gui_owl_v2_1_generation_tokens,
)
from causalcache.policy.gui_owl_v2_runtime import (
    FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
    extend_gui_owl_v2_teacher_inputs,
)
from causalcache.policy.gui_owl_v2_vision import (
    VISION_PATCH_SIZE,
    VISION_SPATIAL_MERGE_SIZE,
    _validate_model_identity,
    verify_frozen_vision_runtime,
)
from causalcache.spatial_reference_audit_v1 import (
    AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES,
    ProfileSpec,
    first_divergent_index,
)


def _runtime_identity_metadata(identity: Any) -> dict[str, Any]:
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


def _attention_metadata(model: Any) -> dict[str, Any]:
    config = model.config
    result = {"model": getattr(config, "_attn_implementation", None)}
    for name in ("text_config", "vision_config"):
        nested = getattr(config, name, None)
        result[name] = getattr(nested, "_attn_implementation", None)
    return result


def audit_absent_scientific_environment() -> dict[str, Any]:
    present = [
        name for name in AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES if name in os.environ
    ]
    if present:
        raise RuntimeError(
            "behavior-changing scientific environment variables must be absent: "
            + ", ".join(present)
        )
    return {
        "audited_names": list(AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES),
        "present_names": [],
        "all_absent": True,
    }


def _validate_observed_attention(
    profile: ProfileSpec,
    attention: Mapping[str, Any],
) -> None:
    observed = [value for value in attention.values() if value is not None]
    if not observed or any(not isinstance(value, str) for value in observed):
        raise RuntimeError("attention implementation was not observably resolved")
    if profile.attention_implementation == "eager":
        if any(value != "eager" for value in observed):
            raise RuntimeError("requested eager attention was not applied everywhere")
    elif any(value == "eager" for value in observed):
        raise RuntimeError("default attention unexpectedly resolved to eager")


class GUIOwlSpatialAuditRuntime(GUIOwlV21OfficialToolsRuntime):
    """GUI-Owl v2.1 interface under one explicitly recorded numerical profile."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        expected_snapshot_manifest: str | Path,
        device: str,
        profile: ProfileSpec,
    ) -> None:
        if not isinstance(profile, ProfileSpec):
            raise TypeError("profile must be a validated ProfileSpec")
        if not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("spatial audit requires an explicit CUDA device")
        scientific_environment_audit = audit_absent_scientific_environment()
        try:
            import torch
            import transformers
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as error:
            raise RuntimeError("spatial audit runtime requires PyTorch and Transformers") from error

        selected_device = torch.device(device)
        if (
            not torch.cuda.is_available()
            or selected_device.index is None
            or selected_device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError("spatial audit requires the selected CUDA device")
        torch.use_deterministic_algorithms(False, warn_only=False)
        torch.manual_seed(profile.seed)
        torch.cuda.manual_seed_all(profile.seed)
        if profile.eager_numerical_controls:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")

        identity = verify_frozen_vision_runtime(
            model_dir=model_dir,
            expected_snapshot_manifest=expected_snapshot_manifest,
        )
        pixels_per_image = FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE * (
            VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE
        ) ** 2
        processor = AutoProcessor.from_pretrained(
            identity.model_dir,
            min_pixels=pixels_per_image,
            max_pixels=pixels_per_image,
            local_files_only=True,
        )
        if int(processor.image_processor.merge_size) != VISION_SPATIAL_MERGE_SIZE:
            raise ValueError("spatial audit processor merge size drifted")
        dtype = torch.bfloat16 if profile.dtype == "bfloat16" else torch.float32
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if profile.attention_implementation != "default":
            load_kwargs["attn_implementation"] = profile.attention_implementation
        model = AutoModelForImageTextToText.from_pretrained(
            identity.model_dir,
            **load_kwargs,
        ).to(selected_device)
        model.eval().requires_grad_(False)
        _validate_model_identity(model, identity)
        for parameter in model.parameters():
            if parameter.device != selected_device:
                raise RuntimeError("spatial audit model spans multiple devices")
            if parameter.is_floating_point() and parameter.dtype is not dtype:
                raise RuntimeError("spatial audit parameter dtype differs from profile")
            if parameter.requires_grad:
                raise RuntimeError("spatial audit model parameters are not frozen")
        attention = _attention_metadata(model)
        _validate_observed_attention(profile, attention)

        self.torch = torch
        self.device = selected_device
        self.processor = processor
        self.model = model
        self.runtime_identity = identity
        self.profile = profile
        self.chat_template_identity = validate_gui_owl_v2_1_chat_template(
            model_dir=identity.model_dir,
            tokenizer=processor.tokenizer,
        )
        self.generation_tokens = validate_gui_owl_v2_1_generation_tokens(
            processor.tokenizer,
            model.generation_config,
        )
        self.metadata = {
            **_runtime_identity_metadata(identity),
            **self._interface_metadata(),
            "audit_profile_id": profile.profile_id,
            "dtype": str(dtype),
            "device": str(selected_device),
            "frozen": True,
            "single_device": True,
            "processor_class": processor.__class__.__name__,
            "model_class": model.__class__.__name__,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "transformers_version": transformers.__version__,
            "target_effective_visual_tokens_per_image": (
                FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
            ),
            "min_pixels": pixels_per_image,
            "max_pixels": pixels_per_image,
            "requested_attention_implementation": profile.attention_implementation,
            "observed_attention_implementation": attention,
            "eager_numerical_controls_requested": profile.eager_numerical_controls,
            "strict_cuda_determinism_claimed": False,
            "scientific_environment_variables_set_by_runtime": False,
            "scientific_environment_audit": scientific_environment_audit,
            "deterministic_algorithms_requested": False,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only_enabled": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "seed": profile.seed,
            "numerical_control_claim": (
                "eager_fixed_seed_tf32_disabled_numerical_control_not_strict_cuda_determinism"
                if profile.eager_numerical_controls
                else "legacy_auto_backend_control"
            ),
            "instrumented_generation_output_scores": False,
        }

    def generate_native_action(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> GUIOwlV21GenerationResult:
        """Run the frozen generation path while preserving actual generated IDs."""
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
            raise RuntimeError("spatial audit generation left the single-item CUDA batch")
        new_tokens = generated[:, prompt_tokens:]
        generated_ids = new_tokens[0].detach().to(device="cpu").tolist()
        if not isinstance(generated_ids, list) or any(
            type(token_id) is not int or token_id < 0 for token_id in generated_ids
        ):
            raise TypeError("spatial audit generated token ids must be non-negative integers")
        if any(token_id in tokens.standard_eos_token_ids for token_id in generated_ids):
            raise RuntimeError("spatial audit generated a suppressed standard EOS token")
        closer_count = generated_ids.count(tokens.tool_call_close_token_id)
        if generated_ids and generated_ids[-1] == tokens.tool_call_close_token_id:
            if closer_count != 1:
                raise RuntimeError("spatial audit generated more than one tool-call closer")
            termination_reason = "model_emitted_tool_call_close"
        elif len(generated_ids) == FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS:
            if closer_count != 0:
                raise RuntimeError("spatial audit generated a non-terminal tool-call closer")
            termination_reason = "max_new_tokens_without_tool_call_close"
        else:
            raise RuntimeError("spatial audit generation stopped before a valid boundary")
        decoded = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if (
            not isinstance(decoded, Sequence)
            or isinstance(decoded, (str, bytes))
            or len(decoded) != 1
            or not isinstance(decoded[0], str)
        ):
            raise TypeError("spatial audit generation must decode to one string")
        output_text = decoded[0]
        if closer_count == 1 and not output_text.endswith("</tool_call>"):
            raise RuntimeError("spatial audit decode hid or moved the generated closer")
        reconstructed = self.processor.tokenizer.encode(
            output_text,
            add_special_tokens=False,
        )
        if not isinstance(reconstructed, list) or any(
            type(token_id) is not int or token_id < 0 for token_id in reconstructed
        ):
            raise TypeError("spatial audit decoded-output reconstruction is invalid")
        metadata = {
            **self._image_metadata(model_inputs, image_counts)[0],
            **self._interface_metadata(),
            "generated_tokens": len(generated_ids),
            "generated_token_ids": generated_ids,
            "generated_token_ids_sha256": hashlib.sha256(
                json.dumps(
                    generated_ids,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "decoded_output_utf8_sha256": hashlib.sha256(
                output_text.encode("utf-8")
            ).hexdigest(),
            "decoded_output_retokenized_ids": reconstructed,
            "decoded_output_retokenization_matches_actual_ids": (
                reconstructed == generated_ids
            ),
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

    def _vector_summary(self, vector: Any, *, target_token_id: int) -> dict[str, Any]:
        if getattr(vector, "ndim", None) != 1:
            raise ValueError("audit score vector must be rank one")
        vocabulary_size = int(vector.shape[0])
        if target_token_id < 0 or target_token_id >= vocabulary_size:
            raise ValueError("audit target token id exceeds vocabulary")
        with self.torch.inference_mode():
            fp32 = vector.to(dtype=self.torch.float32)
            values, indices = self.torch.topk(fp32, k=2, dim=-1)
            target_score = fp32[target_token_id]
            target_log_probability = self.torch.log_softmax(fp32, dim=-1)[target_token_id]
            strictly_greater = self.torch.sum(fp32 > target_score)
        top_values = values.detach().to(device="cpu").tolist()
        top_indices = indices.detach().to(device="cpu").tolist()
        target_value = float(target_score.detach().to(device="cpu").item())
        target_log_prob = float(
            target_log_probability.detach().to(device="cpu").item()
        )
        rank = int(strictly_greater.detach().to(device="cpu").item()) + 1
        if (
            len(top_values) != 2
            or len(top_indices) != 2
            or not all(math.isfinite(float(value)) for value in top_values)
            or not math.isfinite(target_value)
            or not math.isfinite(target_log_prob)
        ):
            raise RuntimeError("audit divergence score summary is non-finite")
        return {
            "target_token_id": target_token_id,
            "target_logit": target_value,
            "target_log_probability": target_log_prob,
            "target_rank_strict": rank,
            "top1_token_id": int(top_indices[0]),
            "top2_token_id": int(top_indices[1]),
            "top1_logit": float(top_values[0]),
            "top2_logit": float(top_values[1]),
            "top1_minus_top2_margin": float(top_values[0] - top_values[1]),
        }

    def _teacher_branch_vector(
        self,
        messages: Sequence[Mapping[str, Any]],
        action: GUIOwlV2Action,
        *,
        divergence_index: int,
    ) -> tuple[Any, Any, dict[str, Any]]:
        model_inputs, image_counts = self._encode_exact_batch((messages,))
        teacher_tokens = build_gui_owl_v2_1_teacher_tokens(
            self.processor.tokenizer,
            action,
        )
        if divergence_index < 0 or divergence_index >= len(teacher_tokens.distance_token_ids):
            raise ValueError("teacher divergence index exceeds action token span")
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        layout = gui_owl_v2_1_teacher_layout(
            prompt_input_tokens=prompt_tokens,
            teacher_tokens=teacher_tokens,
        )
        extended, aligned_keys = extend_gui_owl_v2_teacher_inputs(
            model_inputs,
            (layout.forced_suffix_token_ids,),
            tensor_module=self.torch,
        )
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(
                **extended,
                use_cache=False,
                return_dict=True,
                logits_to_keep=layout.distance_action_tokens,
            )
        self.torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - started
        logits = outputs.logits
        expected_dtype = "torch.bfloat16" if self.profile.dtype == "bfloat16" else "torch.float32"
        if (
            getattr(logits, "ndim", None) != 3
            or tuple(logits.shape[:2]) != (1, layout.distance_action_tokens)
            or logits.device != self.device
            or str(logits.dtype) != expected_dtype
            or getattr(logits, "requires_grad", False)
        ):
            raise RuntimeError("audit teacher logits violate profile shape/device/dtype")
        masked, suppression_value = mask_gui_owl_v2_1_teacher_standard_eos_logits(
            logits,
            standard_eos_token_ids=self.generation_tokens.standard_eos_token_ids,
            tensor_module=self.torch,
        )
        raw_vector = logits[0, divergence_index, :]
        masked_vector = masked[0, divergence_index, :]
        target_id = teacher_tokens.distance_token_ids[divergence_index]
        metadata = {
            **self._image_metadata(model_inputs, image_counts)[0],
            "distance_action_tokens": layout.distance_action_tokens,
            "divergence_index": divergence_index,
            "target_token_id": target_id,
            "raw": self._vector_summary(raw_vector, target_token_id=target_id),
            "generation_aligned_suppressed": self._vector_summary(
                masked_vector,
                target_token_id=target_id,
            ),
            "suppressed_standard_eos_token_ids": list(
                self.generation_tokens.standard_eos_token_ids
            ),
            "suppression_value": suppression_value,
            "extended_prompt_aligned_inputs": list(aligned_keys),
            "latency_seconds": latency,
            "peak_gpu_memory_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated(self.device)
            ),
            "peak_gpu_memory_reserved_bytes": int(
                self.torch.cuda.max_memory_reserved(self.device)
            ),
            "full_vocabulary_logits_host_transfers": 0,
        }
        return raw_vector, masked_vector, metadata

    def shared_prefix_parent_pair_diagnostics(
        self,
        messages: Sequence[Mapping[str, Any]],
        actions: Sequence[GUIOwlV2Action],
    ) -> dict[str, Any]:
        if len(actions) != 2 or any(not isinstance(action, GUIOwlV2Action) for action in actions):
            raise ValueError("teacher audit requires exactly two parent actions")
        token_rows = [
            list(
                build_gui_owl_v2_1_teacher_tokens(
                    self.processor.tokenizer,
                    action,
                ).distance_token_ids
            )
            for action in actions
        ]
        divergence = first_divergent_index(token_rows[0], token_rows[1])
        if divergence is None or divergence <= 0:
            raise ValueError("parent actions do not diverge in the teacher token path")
        if token_rows[0][:divergence] != token_rows[1][:divergence]:
            raise ValueError("parent teacher tokens lack an exact common prefix")
        model_inputs, image_counts = self._encode_exact_batch((messages,))
        extended, aligned_keys = extend_gui_owl_v2_teacher_inputs(
            model_inputs,
            (tuple(token_rows[0][:divergence]),),
            tensor_module=self.torch,
        )
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(
                **extended,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
        self.torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - started
        logits = outputs.logits
        expected_dtype = (
            "torch.bfloat16" if self.profile.dtype == "bfloat16" else "torch.float32"
        )
        if (
            getattr(logits, "ndim", None) != 3
            or tuple(logits.shape[:2]) != (1, 1)
            or logits.device != self.device
            or str(logits.dtype) != expected_dtype
            or getattr(logits, "requires_grad", False)
        ):
            raise RuntimeError("shared-prefix audit logits violate shape/device/dtype")
        masked, suppression_value = mask_gui_owl_v2_1_teacher_standard_eos_logits(
            logits,
            standard_eos_token_ids=self.generation_tokens.standard_eos_token_ids,
            tensor_module=self.torch,
        )
        raw_vector = logits[0, 0, :]
        masked_vector = masked[0, 0, :]
        candidate_ids = [token_rows[0][divergence], token_rows[1][divergence]]
        with self.torch.inference_mode():
            raw_pair = raw_vector.to(dtype=self.torch.float32)[candidate_ids]
            masked_pair = masked_vector.to(dtype=self.torch.float32)[candidate_ids]
        raw_values = raw_pair.detach().to(device="cpu").tolist()
        masked_values = masked_pair.detach().to(device="cpu").tolist()
        if (
            len(raw_values) != 2
            or len(masked_values) != 2
            or not all(math.isfinite(float(value)) for value in (*raw_values, *masked_values))
        ):
            raise RuntimeError("shared-prefix competing-token logits are non-finite")
        return {
            **self._image_metadata(model_inputs, image_counts)[0],
            "teacher_token_ids": token_rows,
            "first_divergent_token_index": divergence,
            "first_divergent_token_ids": candidate_ids,
            "shared_prefix_token_ids": token_rows[0][:divergence],
            "raw_candidate_summaries": [
                self._vector_summary(raw_vector, target_token_id=token_id)
                for token_id in candidate_ids
            ],
            "generation_aligned_candidate_summaries": [
                self._vector_summary(masked_vector, target_token_id=token_id)
                for token_id in candidate_ids
            ],
            "raw_first_minus_second_candidate_logit": float(
                raw_values[0] - raw_values[1]
            ),
            "generation_aligned_first_minus_second_candidate_logit": float(
                masked_values[0] - masked_values[1]
            ),
            "suppressed_standard_eos_token_ids": list(
                self.generation_tokens.standard_eos_token_ids
            ),
            "suppression_value": suppression_value,
            "extended_prompt_aligned_inputs": list(aligned_keys),
            "latency_seconds": latency,
            "peak_gpu_memory_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated(self.device)
            ),
            "peak_gpu_memory_reserved_bytes": int(
                self.torch.cuda.max_memory_reserved(self.device)
            ),
            "interpretation": (
                "single_shared_prefix_post_hoc_teacher_forced_competing_token_"
                "margin_not_original_generation_time_margin"
            ),
            "full_vocabulary_logits_host_transfers": 0,
        }

    def full_parent_action_diagnostic(
        self,
        messages: Sequence[Mapping[str, Any]],
        action: GUIOwlV2Action,
        *,
        divergence_index: int,
    ) -> dict[str, Any]:
        if not isinstance(action, GUIOwlV2Action):
            raise TypeError("full parent diagnostic requires one GUIOwlV2Action")
        _, _, metadata = self._teacher_branch_vector(
            messages,
            action,
            divergence_index=divergence_index,
        )
        return {
            **metadata,
            "interpretation": (
                "full_parent_action_branch_post_hoc_teacher_forced_diagnostic_"
                "with_shape_specific_suffix"
            )
        }
