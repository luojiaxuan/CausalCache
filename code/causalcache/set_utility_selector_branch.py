"""Selector-only representation adapters for frozen GUI-Owl evidence."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight local installs.
    torch = None


QWEN3_VL_LANGUAGE_LAYER_COUNT = 36


@dataclass(frozen=True)
class SelectorBoundaryForward:
    """Full-sequence state needed to replay a selector-only top-layer branch."""

    boundary_hidden_state: Any
    final_hidden_state: Any | None
    input_ids: Any
    attention_mask: Any
    position_ids: Any


def qwen3vl_language_model(policy_model: Any) -> Any:
    model = getattr(policy_model, "model", None)
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if language_model is None or layers is None:
        raise ValueError("policy model does not expose Qwen3-VL language layers")
    if len(layers) != QWEN3_VL_LANGUAGE_LAYER_COUNT:
        raise ValueError("Qwen3-VL language-layer count drifted")
    return language_model


def selector_first_layer(*, trainable_layer_count: int) -> int:
    if type(trainable_layer_count) is not int or not (
        1 <= trainable_layer_count < QWEN3_VL_LANGUAGE_LAYER_COUNT
    ):
        raise ValueError("selector trainable-layer count is outside the valid range")
    return QWEN3_VL_LANGUAGE_LAYER_COUNT - trainable_layer_count


def _position_ids(policy_model: Any, model_inputs: Mapping[str, Any]) -> Any:
    base = getattr(policy_model, "model", None)
    get_rope_index = getattr(base, "get_rope_index", None)
    if not callable(get_rope_index):
        raise ValueError("Qwen3-VL base model does not expose get_rope_index")
    attention_mask = model_inputs.get("attention_mask")
    if getattr(attention_mask, "ndim", None) != 2:
        raise ValueError("selector boundary requires a rank-two attention mask")
    position_ids, _ = get_rope_index(
        input_ids=model_inputs["input_ids"],
        mm_token_type_ids=model_inputs.get("mm_token_type_ids"),
        image_grid_thw=model_inputs.get("image_grid_thw"),
        video_grid_thw=model_inputs.get("video_grid_thw"),
        attention_mask=attention_mask,
    )
    if getattr(position_ids, "ndim", None) != 3 or position_ids.shape[0] != 3:
        raise ValueError("Qwen3-VL multimodal position-id geometry drifted")
    return position_ids


def capture_selector_boundary_forward(
    *,
    runtime: Any,
    messages: Sequence[Mapping[str, Any]],
    trainable_layer_count: int,
    include_final_hidden_state: bool = True,
) -> SelectorBoundaryForward:
    """Run frozen GUI-Owl once and capture the input to its selector branch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("selector boundary extraction requires PyTorch")
    if type(include_final_hidden_state) is not bool:
        raise TypeError("include_final_hidden_state must be boolean")
    model_inputs, image_counts = runtime._encode_exact_batch((messages,))
    if image_counts != (1,):
        raise ValueError("selector boundary forward requires exactly one image")
    language_model = qwen3vl_language_model(runtime.model)
    first_layer = selector_first_layer(trainable_layer_count=trainable_layer_count)
    captured: list[Any] = []

    def capture(_module: Any, args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        if not args or getattr(args[0], "ndim", None) != 3:
            raise ValueError("selector boundary hook did not receive hidden states")
        captured.append(args[0].detach().clone())

    position_ids = _position_ids(runtime.model, model_inputs)
    handle = language_model.layers[first_layer].register_forward_pre_hook(
        capture, with_kwargs=True
    )
    try:
        with torch.inference_mode():
            outputs = runtime.model(
                **model_inputs,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                output_hidden_states=include_final_hidden_state,
                logits_to_keep=1,
            )
    finally:
        handle.remove()
    hidden_states = getattr(outputs, "hidden_states", None)
    if len(captured) != 1:
        raise RuntimeError("selector boundary hook did not fire exactly once")
    boundary = captured[0]
    final_hidden_state = None
    if include_final_hidden_state:
        if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
            raise RuntimeError("GUI-Owl did not expose its final language hidden state")
        final_hidden_state = hidden_states[-1].detach().clone()
        if boundary.shape != final_hidden_state.shape:
            raise ValueError("selector boundary/final hidden geometry differs")
    return SelectorBoundaryForward(
        boundary_hidden_state=boundary,
        final_hidden_state=final_hidden_state,
        input_ids=model_inputs["input_ids"].detach().clone(),
        attention_mask=model_inputs["attention_mask"].detach().clone(),
        position_ids=position_ids.detach().clone(),
    )


@contextmanager
def _temporary_top_layers(
    language_model: Any, *, trainable_layer_count: int
) -> Iterator[None]:
    first_layer = selector_first_layer(trainable_layer_count=trainable_layer_count)
    original = language_model.layers
    language_model.layers = torch.nn.ModuleList(tuple(original[first_layer:]))
    try:
        yield
    finally:
        language_model.layers = original


def replay_selector_top_layers(
    *,
    policy_model: Any,
    boundary: SelectorBoundaryForward,
    trainable_layer_count: int,
) -> Any:
    """Replay only the selected frozen top layers from a cached boundary."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("selector branch replay requires PyTorch")
    language_model = qwen3vl_language_model(policy_model)
    with _temporary_top_layers(
        language_model, trainable_layer_count=trainable_layer_count
    ):
        with torch.inference_mode():
            output = language_model(
                input_ids=None,
                inputs_embeds=boundary.boundary_hidden_state,
                attention_mask=boundary.attention_mask,
                position_ids=boundary.position_ids,
                use_cache=False,
            )
    return output.last_hidden_state


if torch is not None:

    class ZeroInitResidualTokenAdapter(torch.nn.Module):
        """Low-rank post-final-hidden control initialized as exact identity."""

        def __init__(self, hidden_size: int, rank: int) -> None:
            super().__init__()
            if type(hidden_size) is not int or hidden_size <= 0:
                raise ValueError("adapter hidden size must be positive")
            if type(rank) is not int or not 1 <= rank <= hidden_size:
                raise ValueError("adapter rank is outside the valid range")
            self.norm = torch.nn.LayerNorm(hidden_size)
            self.down = torch.nn.Linear(hidden_size, rank, bias=False)
            self.up = torch.nn.Linear(rank, hidden_size, bias=False)
            torch.nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
            torch.nn.init.zeros_(self.up.weight)

        def forward(self, values: Any) -> Any:
            update = self.up(torch.nn.functional.silu(self.down(self.norm(values))))
            return values + update

    class LoRALinear(torch.nn.Module):
        """Frozen linear layer with a zero-initialized trainable LoRA update."""

        def __init__(self, base: Any, *, rank: int, alpha: float) -> None:
            super().__init__()
            if not isinstance(base, torch.nn.Linear):
                raise TypeError("LoRA base must be torch.nn.Linear")
            if type(rank) is not int or rank <= 0:
                raise ValueError("LoRA rank must be positive")
            if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
                raise TypeError("LoRA alpha must be numeric")
            if float(alpha) <= 0.0:
                raise ValueError("LoRA alpha must be positive")
            self.base = base
            self.base.requires_grad_(False)
            # note (luojiaxuan): Keep trainable LoRA master weights in FP32,
            # like the downstream selector head, while the frozen Qwen branch
            # remains BF16. The update is cast back to the base output dtype.
            self.lora_a = torch.nn.Linear(base.in_features, rank, bias=False).to(
                device=base.weight.device, dtype=torch.float32
            )
            self.lora_b = torch.nn.Linear(rank, base.out_features, bias=False).to(
                device=base.weight.device, dtype=torch.float32
            )
            self.scale = float(alpha) / rank
            torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
            torch.nn.init.zeros_(self.lora_b.weight)

        def forward(self, values: Any) -> Any:
            base = self.base(values)
            update = self.lora_b(self.lora_a(values.to(dtype=self.lora_a.weight.dtype)))
            return base + update.to(dtype=base.dtype) * self.scale

    def inject_qwen_attention_lora(
        language_model: Any, *, rank: int, alpha: float
    ) -> tuple[str, ...]:
        """Attach LoRA to q/k/v/o in every layer of a pruned selector branch."""
        language_model.requires_grad_(False)
        targets = []
        for layer_index, layer in enumerate(language_model.layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                raise ValueError("selector branch layer lacks self attention")
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                base = getattr(attention, name, None)
                if not isinstance(base, torch.nn.Linear):
                    raise ValueError(f"selector LoRA target drifted: {name}")
                setattr(
                    attention,
                    name,
                    LoRALinear(base, rank=rank, alpha=alpha),
                )
                targets.append(f"layers.{layer_index}.self_attn.{name}")
        return tuple(targets)

else:  # pragma: no cover

    class ZeroInitResidualTokenAdapter:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("token adapter requires PyTorch")

    class LoRALinear:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("LoRA requires PyTorch")

    def inject_qwen_attention_lora(*_: Any, **__: Any) -> tuple[str, ...]:
        raise RuntimeError("LoRA requires PyTorch")


__all__ = [
    "LoRALinear",
    "QWEN3_VL_LANGUAGE_LAYER_COUNT",
    "SelectorBoundaryForward",
    "ZeroInitResidualTokenAdapter",
    "capture_selector_boundary_forward",
    "inject_qwen_attention_lora",
    "qwen3vl_language_model",
    "replay_selector_top_layers",
    "selector_first_layer",
]
