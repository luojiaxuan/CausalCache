"""Generation runtime for the official-faithful GUI-Owl mobile protocol."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.policy.gui_owl_official import (
    OFFICIAL_PROTOCOL_ID,
    OfficialParseError,
    extract_action_line,
    parse_official_output,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from causalcache.policy.gui_owl_v2_runtime import (
    FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
)


@dataclass(frozen=True)
class GUIOwlOfficialGenerationResult:
    output_text: str
    canonical_action: GUIOwlV2Action | None
    action_text: str
    full_response: str
    dropped_arguments: Mapping[str, Any]
    parse_error: str | None
    metadata: Mapping[str, Any]

    @property
    def policy_parsed(self) -> bool:
        return self.canonical_action is not None


def _official_response(
    action_text: str,
    action: GUIOwlV2Action,
) -> str:
    arguments = json.dumps(
        action.arguments(),
        ensure_ascii=False,
        separators=(", ", ": "),
    )
    return (
        f"Action: {action_text}\n"
        f'<tool_call>\n{{"name": "mobile_use", "arguments": {arguments}}}\n'
        "</tool_call>"
    )


def interpret_official_output(
    output_text: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> GUIOwlOfficialGenerationResult:
    """Interpret one generation without turning parser drift into task failure."""
    action: GUIOwlV2Action | None
    dropped: Mapping[str, Any]
    parse_error: str | None
    try:
        action, dropped = parse_official_output(output_text)
        parse_error = None
    except OfficialParseError as error:
        action = None
        dropped = {}
        parse_error = str(error)
    action_text = extract_action_line(output_text).strip()
    if not action_text:
        action_text = action.action if action is not None else "unknown action"
    if isinstance(output_text, str) and output_text.strip():
        full_response = output_text
    elif action is not None:
        full_response = _official_response(action_text, action)
    else:
        full_response = f"Action: {action_text}"
    return GUIOwlOfficialGenerationResult(
        output_text=output_text,
        canonical_action=action,
        action_text=action_text,
        full_response=full_response,
        dropped_arguments=dict(dropped),
        parse_error=parse_error,
        metadata=dict(metadata or {}),
    )


class GUIOwlOfficialRuntime:
    """Reuse the frozen model while preserving the official prompt encoding."""

    def __init__(self, base: GUIOwlV21OfficialToolsRuntime) -> None:
        if not isinstance(base, GUIOwlV21OfficialToolsRuntime):
            raise TypeError("base must be a GUIOwlV21OfficialToolsRuntime")
        self.base = base

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **dict(self.base.metadata),
            "prompt_protocol": OFFICIAL_PROTOCOL_ID,
            "output_parser": "official_permissive_unknown_step",
            "chat_template_tools_kwarg": False,
        }

    def encode(self, messages: Sequence[Mapping[str, Any]]) -> Any:
        """Tokenize + preprocess one prompt. 与 generate 内联版逐字节等价。

        # note (luojiaxuan): 单独暴露出来是因为调用方(HGKV mask 计算)本来就要
        # 编码一次;不复用的话每个请求要把 5 张图的 resize/patchify 做两遍。
        """
        return self.base.processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=False,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        encoded: Any = None,
    ) -> GUIOwlOfficialGenerationResult:
        base = self.base
        encode_started = time.perf_counter()
        reused_encoding = encoded is not None
        if encoded is None:
            encoded = self.encode(messages)
        encode_seconds = time.perf_counter() - encode_started
        encoded = (
            encoded.to(base.device)
            if hasattr(encoded, "to")
            else {key: value.to(base.device) for key, value in encoded.items()}
        )
        prompt_tokens = int(encoded["input_ids"].shape[1])
        tokens = base.generation_tokens
        started = time.perf_counter()
        with base.torch.inference_mode():
            generated = base.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
                eos_token_id=tokens.tool_call_close_token_id,
                pad_token_id=tokens.pad_token_id,
                num_beams=1,
                num_return_sequences=1,
            )
        latency_seconds = time.perf_counter() - started
        new_tokens = generated[:, prompt_tokens:]
        output_text = base.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0]
        metadata = {
            "prompt_protocol": OFFICIAL_PROTOCOL_ID,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(new_tokens.shape[1]),
            "generated_token_ids_sha256": hashlib.sha256(
                json.dumps(
                    new_tokens[0].detach().to(device="cpu").tolist(),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "do_sample": False,
            "max_new_tokens": FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS,
            "num_beams": 1,
            "latency_seconds": latency_seconds,
            "encode_seconds": encode_seconds,
            "reused_encoding": reused_encoding,
        }
        return interpret_official_output(output_text, metadata=metadata)


__all__ = [
    "GUIOwlOfficialGenerationResult",
    "GUIOwlOfficialRuntime",
    "interpret_official_output",
]
