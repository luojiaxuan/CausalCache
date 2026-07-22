"""Temperature-sampled GUI-Owl v2.1 runtime for success-trajectory collection."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.policy.gui_owl_v2_1 import (
    canonical_json_sha256,
    parse_gui_owl_v2_1_output,
)
from causalcache.policy.gui_owl_v2_1_runtime import (
    GUIOwlV21GenerationParseError,
    GUIOwlV21GenerationResult,
    GUIOwlV21OfficialToolsRuntime,
)
from causalcache.policy.gui_owl_v2_runtime import FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS


class GUIOwlV21SampledToolsRuntime(GUIOwlV21OfficialToolsRuntime):
    """Collection-only sampling variant; the frozen greedy path is untouched.

    # note (luojiaxuan): generate_native_action 的校验逻辑必须与父类冻结实现
    # 逐条一致(EOS 抑制、closer 唯一性、解码后缀检查),仅解码策略从贪心换成
    # 温度采样。此类只用于成功轨迹采集,禁止进入任何评估或裁决路径。
    """

    def __init__(
        self,
        *,
        temperature: float,
        top_p: float,
        **kwargs: Any,
    ) -> None:
        if not 0.0 < temperature <= 2.0:
            raise ValueError("sampling temperature must be within (0, 2]")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("sampling top_p must be within (0, 1]")
        super().__init__(**kwargs)
        self.sampling_temperature = float(temperature)
        self.sampling_top_p = float(top_p)

    def generate_native_action(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> GUIOwlV21GenerationResult:
        """Generate one strict official tool call via temperature sampling."""
        model_inputs, image_counts = self._encode_exact_batch((messages,))
        prompt_tokens = int(model_inputs["input_ids"].shape[1])
        tokens = self.generation_tokens
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                do_sample=True,
                temperature=self.sampling_temperature,
                top_p=self.sampling_top_p,
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
            raise RuntimeError(
                "GUI-Owl v2.1 sampled tokens left the single-item CUDA batch"
            )
        new_tokens = generated[:, prompt_tokens:]
        generated_ids = new_tokens[0].detach().to(device="cpu").tolist()
        if not isinstance(generated_ids, list) or any(
            type(token_id) is not int for token_id in generated_ids
        ):
            raise TypeError("GUI-Owl v2.1 sampled token ids must be an integer list")
        if any(token_id in tokens.standard_eos_token_ids for token_id in generated_ids):
            raise RuntimeError("GUI-Owl v2.1 sampled a suppressed standard EOS token")
        closer_count = generated_ids.count(tokens.tool_call_close_token_id)
        if generated_ids and generated_ids[-1] == tokens.tool_call_close_token_id:
            if closer_count != 1:
                raise RuntimeError(
                    "GUI-Owl v2.1 sampled more than one tool-call closer"
                )
            termination_reason = "model_emitted_tool_call_close"
        elif len(generated_ids) == FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS:
            if closer_count != 0:
                raise RuntimeError(
                    "GUI-Owl v2.1 sampled a non-terminal tool-call closer"
                )
            termination_reason = "max_new_tokens_without_tool_call_close"
        else:
            raise RuntimeError(
                "GUI-Owl v2.1 sampling terminated before close or max_new_tokens"
            )
        decoded = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
            raise TypeError(
                "GUI-Owl v2.1 processor batch_decode must return a sequence"
            )
        if len(decoded) != 1 or not isinstance(decoded[0], str):
            raise ValueError(
                "GUI-Owl v2.1 sampling must decode to exactly one string"
            )
        output_text = decoded[0]
        if closer_count == 1 and not output_text.endswith("</tool_call>"):
            raise RuntimeError(
                "GUI-Owl v2.1 decoded output hid or moved the sampled closer"
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
            "do_sample": True,
            "temperature": self.sampling_temperature,
            "top_p": self.sampling_top_p,
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
