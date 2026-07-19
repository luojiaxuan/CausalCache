"""Versioned prepared-input runtimes for the D1 action-stability diagnostic."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.policy.gui_owl_v2_1 import (
    canonical_json_sha256,
    parse_gui_owl_v2_1_output,
)
from causalcache.policy.gui_owl_v2_1_runtime import (
    GUIOwlV21GenerationParseError,
    GUIOwlV21GenerationResult,
)
from causalcache.policy.gui_owl_v2_1_throughput_runtime import (
    GUIOwlV21ThroughputRuntime,
)
from causalcache.policy.gui_owl_v2_2_eager_runtime import (
    GUIOwlV22EagerRuntime,
    gui_owl_v2_2_observed_attention,
)
from causalcache.policy.gui_owl_v2_runtime import FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS


AUTO_FRESH_ENCODE_CONDITION = "auto_fresh_encode"
AUTO_FROZEN_ENCODED_CONDITION = "auto_frozen_encoded"
EAGER_FROZEN_ENCODED_CONTROL = "eager_frozen_encoded_control"
EXPECTED_AUTO_ATTENTION = {"text": "sdpa", "top": "sdpa", "vision": "sdpa"}


def validate_auto_attention_v1(attention: Mapping[str, Any]) -> dict[str, str]:
    observed = dict(attention)
    if observed != EXPECTED_AUTO_ATTENTION:
        raise RuntimeError(
            "D1 parent-compatible automatic attention did not resolve to SDPA"
        )
    return dict(EXPECTED_AUTO_ATTENTION)


@dataclass(frozen=True, slots=True)
class PreparedExactInputV1:
    """Opaque same-process processor output and its immutable tensor snapshot."""

    owner_identity: int
    model_inputs: Mapping[str, Any]
    image_counts: tuple[int, ...]
    snapshots: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.owner_identity) is not int or self.owner_identity <= 0:
            raise ValueError("prepared input owner identity is invalid")
        if not isinstance(self.model_inputs, Mapping) or not self.model_inputs:
            raise ValueError("prepared input requires a non-empty tensor mapping")
        if set(self.model_inputs) != set(self.snapshots):
            raise ValueError("prepared input snapshot keys drifted")
        if not isinstance(self.image_counts, tuple) or not self.image_counts:
            raise ValueError("prepared input image counts are invalid")


def _clone_tensor_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in values.items():
        clone = getattr(value, "clone", None)
        if not callable(clone):
            raise TypeError(f"prepared processor value {key} is not a tensor")
        cloned[key] = clone()
    return cloned


def _tensor_mapping_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key in sorted(left):
        left_value = left[key]
        right_value = right[key]
        if (
            getattr(left_value, "shape", None) != getattr(right_value, "shape", None)
            or getattr(left_value, "dtype", None) != getattr(right_value, "dtype", None)
            or getattr(left_value, "device", None)
            != getattr(right_value, "device", None)
        ):
            return False
        equal = getattr(left_value, "equal", None)
        if not callable(equal) or type(equal(right_value)) is not bool:
            return False
        if not equal(right_value):
            return False
    return True


class _PreparedInputGenerationMixin:
    """Add an opaque prepared-input generation path without changing parent bytes."""

    def prepare_exact_input_v1(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> PreparedExactInputV1:
        model_inputs, image_counts = self._encode_exact_batch((messages,))
        return PreparedExactInputV1(
            owner_identity=id(self),
            model_inputs=model_inputs,
            image_counts=tuple(image_counts),
            snapshots=_clone_tensor_mapping(model_inputs),
        )

    def prepared_input_unchanged_v1(self, prepared: PreparedExactInputV1) -> bool:
        if not isinstance(prepared, PreparedExactInputV1):
            raise TypeError("prepared input has the wrong diagnostic type")
        if prepared.owner_identity != id(self):
            raise ValueError("prepared input belongs to a different runtime")
        return _tensor_mapping_equal(prepared.model_inputs, prepared.snapshots)

    def generate_from_prepared_v1(
        self,
        prepared: PreparedExactInputV1,
    ) -> GUIOwlV21GenerationResult:
        if not self.prepared_input_unchanged_v1(prepared):
            raise RuntimeError("prepared input changed before generation")
        result = self._generate_from_model_inputs_v1(
            dict(prepared.model_inputs),
            prepared.image_counts,
        )
        if not self.prepared_input_unchanged_v1(prepared):
            raise RuntimeError("prepared input changed during generation")
        return result

    def _generate_from_model_inputs_v1(
        self,
        model_inputs: Mapping[str, Any],
        image_counts: tuple[int, ...],
    ) -> GUIOwlV21GenerationResult:
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        tokens = self.generation_tokens
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
            raise RuntimeError("prepared generation left the single-item CUDA batch")
        new_tokens = generated[:, prompt_tokens:]
        generated_ids = new_tokens[0].detach().to(device="cpu").tolist()
        if not isinstance(generated_ids, list) or any(
            type(token_id) is not int for token_id in generated_ids
        ):
            raise TypeError("prepared generation token ids are invalid")
        if any(token_id in tokens.standard_eos_token_ids for token_id in generated_ids):
            raise RuntimeError("prepared generation emitted a suppressed standard EOS")
        closer_count = generated_ids.count(tokens.tool_call_close_token_id)
        if generated_ids and generated_ids[-1] == tokens.tool_call_close_token_id:
            if closer_count != 1:
                raise RuntimeError("prepared generation emitted multiple closers")
            termination_reason = "model_emitted_tool_call_close"
        elif len(generated_ids) == FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS:
            if closer_count != 0:
                raise RuntimeError("prepared generation emitted a non-terminal closer")
            termination_reason = "max_new_tokens_without_tool_call_close"
        else:
            raise RuntimeError("prepared generation terminated before its frozen boundary")
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
            raise ValueError("prepared generation must decode to exactly one string")
        output_text = decoded[0]
        if closer_count == 1 and not output_text.endswith("</tool_call>"):
            raise RuntimeError("prepared generation decoded closer drifted")
        metadata = {
            **self._image_metadata(model_inputs, image_counts)[0],
            **self._interface_metadata(),
            "decoded_output_utf8_sha256": hashlib.sha256(
                output_text.encode("utf-8")
            ).hexdigest(),
            "do_sample": False,
            "generated_token_ids_sha256": canonical_json_sha256(generated_ids),
            "generated_tokens": len(generated_ids),
            "generated_tool_call_close_token_count": closer_count,
            "host_injected_tool_call_closer": False,
            "latency_seconds": latency_seconds,
            "max_new_tokens": FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
            "model_emitted_tool_call_close": closer_count == 1,
            "num_beams": 1,
            "num_return_sequences": 1,
            "output_recovery_or_normalization": False,
            "prepared_exact_input_reused": True,
            "return_dict_in_generate": False,
            "termination_reason": termination_reason,
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


class GUIOwlV21AutoActionStabilityRuntimeV1(
    _PreparedInputGenerationMixin,
    GUIOwlV21ThroughputRuntime,
):
    """Parent-v2 automatic-attention runtime plus prepared-input generation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        attention = validate_auto_attention_v1(
            gui_owl_v2_2_observed_attention(self.model)
        )
        self.metadata = {
            **self.metadata,
            "observed_attention_implementation": attention,
            "requested_attention_implementation": "auto_parent_default",
            "runtime_profile_id": "causalcache_action_stability_auto_sdpa_v1",
        }


class GUIOwlV21EagerActionStabilityRuntimeV1(
    _PreparedInputGenerationMixin,
    GUIOwlV22EagerRuntime,
):
    """Recovered eager numerical-control runtime plus prepared-input generation."""


__all__ = [
    "AUTO_FRESH_ENCODE_CONDITION",
    "AUTO_FROZEN_ENCODED_CONDITION",
    "EAGER_FROZEN_ENCODED_CONTROL",
    "EXPECTED_AUTO_ATTENTION",
    "GUIOwlV21AutoActionStabilityRuntimeV1",
    "GUIOwlV21EagerActionStabilityRuntimeV1",
    "PreparedExactInputV1",
    "validate_auto_attention_v1",
]
