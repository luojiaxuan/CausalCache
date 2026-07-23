"""History-gated KV LoRA residual over the frozen GUI-Owl language model.

# note (luojiaxuan): 冻结契约 docs/history_gated_mainline_v1.md 的 V1 架构:仅 LM
# 最后 layer_count 层的 k_proj/v_proj 挂 mask 门控 LoRA residual
# (K_l = W_K h + M_hist·ΔW_K h,V 同理),rank 8 / alpha 16 / dropout 0。
# B0 不变量:ctx 为 None、history_present 为 False、mask 全 False,或 mask 与
# 当前序列长度对不上(decode 增量单 token 步)时,hook 直接原样返回冻结输出,
# 一条乘法都不执行,保证 bitwise parity;lora_b 零初始化使注入后立即恒等。
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch

from causalcache.policy.history_adapter_context import get_history_adapter_context

HISTORY_GATED_TARGET_MODULES = ("k_proj", "v_proj")

_LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")


class HistoryGatedKVLinear:
    """Mask-gated LoRA residual on one frozen k/v projection via forward hook."""

    def __init__(self, module: torch.nn.Linear, *, rank: int, alpha: int) -> None:
        self.module = module
        self.scaling = alpha / rank
        device = module.weight.device
        self.lora_a = torch.nn.Parameter(
            torch.zeros(rank, module.in_features, dtype=torch.float32, device=device)
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros(module.out_features, rank, dtype=torch.float32, device=device)
        )
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.handle = module.register_forward_hook(self._hook)

    def _hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        context = get_history_adapter_context()
        if context is None or not context.history_present:
            return output
        mask = context.history_token_mask
        # note (luojiaxuan): mask 固定为 [1, seq] 且必须与当前前向的序列长度
        # 精确相等才施加 residual;prefill use_cache=True 与 teacher-forced
        # use_cache=False 都是全长前向,天然满足。decode 增量步只送 1 个新
        # token,新生成 token 永远不是历史 token,连同任何其它长度/批次不一致
        # 一律按非历史 bypass,原样返回冻结输出。
        if mask is None or mask.shape != output.shape[:-1]:
            return output
        if not bool(mask.any()):
            return output
        x = inputs[0]
        delta = (
            x.to(self.lora_a.dtype) @ self.lora_a.T @ self.lora_b.T
        ) * self.scaling
        gate = mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
        return output + (delta * gate).to(output.dtype)


def inject_history_gated_kv(
    model: Any, *, layer_count: int = 8, rank: int = 8, alpha: int = 16
) -> dict[str, HistoryGatedKVLinear]:
    """Wrap k_proj/v_proj of the last ``layer_count`` language-model layers."""
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    candidates: list[tuple[int, str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and name.split(".")[-1] in HISTORY_GATED_TARGET_MODULES
            and ".visual." not in f".{name}."
            and ("language_model" in name or ".model.layers." in f".{name}.")
        ):
            match = _LAYER_INDEX_PATTERN.search(f".{name}.")
            if match is None:
                raise RuntimeError(
                    f"language-model projection {name} lacks a parseable layer index"
                )
            candidates.append((int(match.group(1)), name, module))
    if not candidates:
        raise RuntimeError(
            "history-gated injection matched no language-model k/v projections"
        )
    layer_indices = sorted({index for index, _, _ in candidates})
    if len(layer_indices) < layer_count:
        raise RuntimeError(
            f"model exposes {len(layer_indices)} language-model layers, fewer than "
            f"layer_count={layer_count}"
        )
    selected = set(layer_indices[-layer_count:])
    wrapped: dict[str, HistoryGatedKVLinear] = {}
    for index, name, module in candidates:
        if index in selected:
            wrapped[name] = HistoryGatedKVLinear(module, rank=rank, alpha=alpha)
    expected_count = layer_count * len(HISTORY_GATED_TARGET_MODULES)
    if len(wrapped) != expected_count:
        raise RuntimeError(
            f"history-gated injection wrapped {len(wrapped)} modules, expected "
            f"{expected_count} (layer_count x k/v)"
        )
    return wrapped


def history_gated_state_dict(
    wrapped: dict[str, HistoryGatedKVLinear],
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, lora in wrapped.items():
        state[f"{name}.lora_a"] = lora.lora_a.detach().to(device="cpu")
        state[f"{name}.lora_b"] = lora.lora_b.detach().to(device="cpu")
    return state


def load_history_gated_state_dict(
    wrapped: dict[str, HistoryGatedKVLinear], state: dict[str, Any]
) -> None:
    expected = {
        f"{name}.{part}" for name in wrapped for part in ("lora_a", "lora_b")
    }
    if expected != set(state):
        raise ValueError("history-gated LoRA checkpoint key inventory drifted")
    with torch.no_grad():
        for name, lora in wrapped.items():
            lora.lora_a.copy_(state[f"{name}.lora_a"].to(lora.lora_a.device))
            lora.lora_b.copy_(state[f"{name}.lora_b"].to(lora.lora_b.device))
