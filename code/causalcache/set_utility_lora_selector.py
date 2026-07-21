"""Selector-side Qwen3-VL top-layer LoRA and restoration marginal head."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight local installs.
    torch = None

from causalcache.set_utility_selector_branch import (
    LoRALinear,
    inject_qwen_attention_lora,
)
from causalcache.set_utility_token_models import TokenConditionalMarginalPredictor

SELECTOR_TOP_LAYER_COUNT = 4


@dataclass(frozen=True)
class FullSequenceBoundaryEntity:
    """One unpadded entity at the input to the selector's Qwen top layers."""

    boundary_hidden_state: Any
    input_ids: Any
    attention_mask: Any
    position_ids: Any
    visual_positions: Any
    text_positions: Any


@dataclass(frozen=True)
class SelectorPredictorTokens:
    """Top-layer outputs reshaped to the existing token-predictor interface."""

    query_visual_tokens: Any
    query_visual_mask: Any
    query_text_tokens: Any
    query_text_mask: Any
    event_visual_tokens: Any
    event_visual_mask: Any
    event_text_tokens: Any
    event_text_mask: Any

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "query_visual_tokens": self.query_visual_tokens,
            "query_visual_mask": self.query_visual_mask,
            "query_text_tokens": self.query_text_tokens,
            "query_text_mask": self.query_text_mask,
            "event_visual_tokens": self.event_visual_tokens,
            "event_visual_mask": self.event_visual_mask,
            "event_text_tokens": self.event_text_tokens,
            "event_text_mask": self.event_text_mask,
        }


if torch is not None:

    def _squeeze_cached_batch_axis(value: Any, *, rank: int, name: str) -> Any:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.ndim == rank + 1 and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.ndim != rank:
            raise ValueError(f"{name} geometry drifted")
        return value

    def _normalized_entity(
        entity: FullSequenceBoundaryEntity,
    ) -> FullSequenceBoundaryEntity:
        if not isinstance(entity, FullSequenceBoundaryEntity):
            raise TypeError("selector entity must be FullSequenceBoundaryEntity")
        hidden = _squeeze_cached_batch_axis(
            entity.boundary_hidden_state,
            rank=2,
            name="boundary hidden state",
        )
        input_ids = _squeeze_cached_batch_axis(
            entity.input_ids, rank=1, name="boundary input ids"
        )
        attention_mask = _squeeze_cached_batch_axis(
            entity.attention_mask, rank=1, name="boundary attention mask"
        )
        position_ids = entity.position_ids
        if not isinstance(position_ids, torch.Tensor):
            raise TypeError("boundary position ids must be a tensor")
        if position_ids.ndim == 3 and position_ids.shape[1] == 1:
            position_ids = position_ids.squeeze(1)
        if position_ids.ndim != 2 or position_ids.shape[0] != 3:
            raise ValueError("boundary position-id geometry drifted")
        visual_positions = _squeeze_cached_batch_axis(
            entity.visual_positions,
            rank=1,
            name="visual positions",
        )
        text_positions = _squeeze_cached_batch_axis(
            entity.text_positions,
            rank=1,
            name="text positions",
        )
        sequence_length = hidden.shape[0]
        if hidden.shape[1] <= 0 or any(
            value.shape[0] != sequence_length for value in (input_ids, attention_mask)
        ):
            raise ValueError("boundary sequence lengths differ")
        if position_ids.shape[1] != sequence_length:
            raise ValueError("boundary position-id length differs")
        if input_ids.dtype != torch.long:
            raise TypeError("boundary input ids must use torch.long")
        if position_ids.dtype != torch.long:
            raise TypeError("boundary position ids must use torch.long")
        if attention_mask.dtype not in (torch.bool, torch.long, torch.int64):
            raise TypeError("boundary attention mask must be boolean or int64")
        for name, positions in (
            ("visual", visual_positions),
            ("text", text_positions),
        ):
            if positions.dtype != torch.long:
                raise TypeError(f"{name} positions must use torch.long")
            if positions.numel() == 0:
                raise ValueError(f"{name} token selection is empty")
            if bool(((positions < 0) | (positions >= sequence_length)).any()):
                raise ValueError(f"{name} token selection is out of range")
            if positions.numel() != torch.unique(positions).numel():
                raise ValueError(f"{name} token positions are duplicated")
        if bool(torch.isin(visual_positions, text_positions).any()):
            raise ValueError("visual and text token selections overlap")
        return FullSequenceBoundaryEntity(
            boundary_hidden_state=hidden,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_positions=visual_positions,
            text_positions=text_positions,
        )

    def _last_hidden_state(output: Any) -> Any:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(
                "selector top-layer branch returned invalid hidden states"
            )
        return hidden

    def _padded_tokens(values: Sequence[Any], *, hidden_size: int) -> tuple[Any, Any]:
        if not values:
            raise ValueError("cannot pad an empty token collection")
        if any(value.ndim != 2 or value.shape[1] != hidden_size for value in values):
            raise ValueError("selected token geometry drifted")
        lengths = torch.tensor(
            [value.shape[0] for value in values],
            dtype=torch.long,
            device=values[0].device,
        )
        padded = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
        positions = torch.arange(padded.shape[1], device=padded.device)
        return padded, positions.unsqueeze(0) < lengths.unsqueeze(1)

    class SelectorSideLoRAMarginalPredictor(torch.nn.Module):
        """Replay a frozen Qwen top-4 branch with LoRA before marginal scoring."""

        def __init__(
            self,
            *,
            top_language_model: Any,
            predictor: TokenConditionalMarginalPredictor,
            lora_rank: int,
            lora_alpha: float,
            entity_microbatch_size: int = 8,
        ) -> None:
            super().__init__()
            layers = getattr(top_language_model, "layers", None)
            if not isinstance(layers, torch.nn.ModuleList):
                raise TypeError(
                    "selector language model must expose a ModuleList of layers"
                )
            if len(layers) != SELECTOR_TOP_LAYER_COUNT:
                raise ValueError(
                    "selector language model must be pruned to the top four layers"
                )
            if not isinstance(predictor, TokenConditionalMarginalPredictor):
                raise TypeError("predictor must be TokenConditionalMarginalPredictor")
            if type(entity_microbatch_size) is not int or entity_microbatch_size <= 0:
                raise ValueError("entity microbatch size must be positive")
            self.top_language_model = top_language_model
            self.predictor = predictor
            self.entity_microbatch_size = entity_microbatch_size
            self.lora_targets = inject_qwen_attention_lora(
                self.top_language_model,
                rank=lora_rank,
                alpha=lora_alpha,
            )
            expected_targets = SELECTOR_TOP_LAYER_COUNT * 4
            if len(self.lora_targets) != expected_targets:
                raise RuntimeError("selector LoRA target count drifted")

        def _branch_device_and_dtype(self) -> tuple[Any, Any]:
            parameter = next(self.top_language_model.parameters(), None)
            if parameter is None:
                raise ValueError("selector language model contains no parameters")
            return parameter.device, parameter.dtype

        def _encode_entities(
            self,
            entities: Sequence[FullSequenceBoundaryEntity],
            *,
            microbatch_size: int,
        ) -> list[tuple[Any, Any]]:
            if not entities:
                raise ValueError("selector entity batch is empty")
            normalized = tuple(_normalized_entity(entity) for entity in entities)
            hidden_size = self.predictor.config.source_hidden_size
            if any(
                entity.boundary_hidden_state.shape[1] != hidden_size
                for entity in normalized
            ):
                raise ValueError(
                    "boundary hidden size differs from predictor source size"
                )
            # note (luojiaxuan): Exact-length buckets avoid changing top-layer
            # numerics through padding while still batching repeated GUI lengths.
            buckets: dict[int, list[int]] = defaultdict(list)
            for index, entity in enumerate(normalized):
                buckets[int(entity.boundary_hidden_state.shape[0])].append(index)
            selected: list[tuple[Any, Any] | None] = [None] * len(normalized)
            device, dtype = self._branch_device_and_dtype()
            for sequence_length in sorted(buckets):
                indices = buckets[sequence_length]
                for start in range(0, len(indices), microbatch_size):
                    chunk = indices[start : start + microbatch_size]
                    hidden = torch.stack(
                        [normalized[index].boundary_hidden_state for index in chunk]
                    ).to(device=device, dtype=dtype)
                    attention_mask = torch.stack(
                        [normalized[index].attention_mask for index in chunk]
                    ).to(device=device)
                    position_ids = torch.stack(
                        [normalized[index].position_ids for index in chunk], dim=1
                    ).to(device=device)
                    output = self.top_language_model(
                        input_ids=None,
                        inputs_embeds=hidden,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    final_hidden = _last_hidden_state(output)
                    if final_hidden.shape != hidden.shape:
                        raise RuntimeError("selector top-layer output geometry drifted")
                    for local_index, entity_index in enumerate(chunk):
                        entity = normalized[entity_index]
                        visual_positions = entity.visual_positions.to(device=device)
                        text_positions = entity.text_positions.to(device=device)
                        selected[entity_index] = (
                            final_hidden[local_index].index_select(0, visual_positions),
                            final_hidden[local_index].index_select(0, text_positions),
                        )
            if any(value is None for value in selected):
                raise RuntimeError("selector entity encoding is incomplete")
            return [value for value in selected if value is not None]

        def encode_predictor_tokens(
            self,
            *,
            query_entities: Sequence[FullSequenceBoundaryEntity],
            event_entities: Sequence[Sequence[FullSequenceBoundaryEntity | None]],
            event_mask: Any,
            entity_microbatch_size: int | None = None,
        ) -> SelectorPredictorTokens:
            """Run valid query/events and return padded predictor source tensors."""
            if not isinstance(event_mask, torch.Tensor) or event_mask.ndim != 2:
                raise ValueError("event mask must be a rank-two tensor")
            if event_mask.dtype != torch.bool:
                raise TypeError("event mask must be boolean")
            batch_size, event_count = event_mask.shape
            if len(query_entities) != batch_size or len(event_entities) != batch_size:
                raise ValueError("selector state batch geometry drifted")
            if any(len(events) != event_count for events in event_entities):
                raise ValueError("selector event inventory geometry drifted")
            if bool((~event_mask).all(dim=1).any()):
                raise ValueError("every selector state requires a candidate event")
            flat_event_entities: list[FullSequenceBoundaryEntity] = []
            flat_valid_indices: list[int] = []
            event_mask_cpu = event_mask.detach().to(device="cpu")
            for batch_index, events in enumerate(event_entities):
                for event_index, entity in enumerate(events):
                    valid = bool(event_mask_cpu[batch_index, event_index].item())
                    if valid and entity is None:
                        raise ValueError("valid event is missing its boundary entity")
                    if not valid and entity is not None:
                        raise ValueError(
                            "padded event unexpectedly has a boundary entity"
                        )
                    if valid:
                        assert entity is not None
                        flat_event_entities.append(entity)
                        flat_valid_indices.append(
                            batch_index * event_count + event_index
                        )
            microbatch_size = (
                self.entity_microbatch_size
                if entity_microbatch_size is None
                else entity_microbatch_size
            )
            if type(microbatch_size) is not int or microbatch_size <= 0:
                raise ValueError("entity microbatch size must be positive")
            entity_outputs = self._encode_entities(
                tuple(query_entities) + tuple(flat_event_entities),
                microbatch_size=microbatch_size,
            )
            query_outputs = entity_outputs[:batch_size]
            event_outputs = entity_outputs[batch_size:]
            hidden_size = self.predictor.config.source_hidden_size
            query_visual, query_visual_mask = _padded_tokens(
                [value[0] for value in query_outputs], hidden_size=hidden_size
            )
            query_text, query_text_mask = _padded_tokens(
                [value[1] for value in query_outputs], hidden_size=hidden_size
            )
            valid_event_visual, valid_event_visual_mask = _padded_tokens(
                [value[0] for value in event_outputs], hidden_size=hidden_size
            )
            valid_event_text, valid_event_text_mask = _padded_tokens(
                [value[1] for value in event_outputs], hidden_size=hidden_size
            )
            flat_count = batch_size * event_count
            valid_indices = torch.tensor(
                flat_valid_indices, dtype=torch.long, device=query_visual.device
            )
            event_visual = valid_event_visual.new_zeros(
                (flat_count, *valid_event_visual.shape[1:])
            )
            event_visual_masks = torch.zeros(
                (flat_count, valid_event_visual_mask.shape[1]),
                dtype=torch.bool,
                device=query_visual.device,
            )
            event_text = valid_event_text.new_zeros(
                (flat_count, *valid_event_text.shape[1:])
            )
            event_text_masks = torch.zeros(
                (flat_count, valid_event_text_mask.shape[1]),
                dtype=torch.bool,
                device=query_visual.device,
            )
            event_visual.index_copy_(0, valid_indices, valid_event_visual)
            event_visual_masks.index_copy_(0, valid_indices, valid_event_visual_mask)
            event_text.index_copy_(0, valid_indices, valid_event_text)
            event_text_masks.index_copy_(0, valid_indices, valid_event_text_mask)
            return SelectorPredictorTokens(
                query_visual_tokens=query_visual,
                query_visual_mask=query_visual_mask,
                query_text_tokens=query_text,
                query_text_mask=query_text_mask,
                event_visual_tokens=event_visual.reshape(
                    batch_size, event_count, *event_visual.shape[1:]
                ),
                event_visual_mask=event_visual_masks.reshape(
                    batch_size, event_count, event_visual_masks.shape[1]
                ),
                event_text_tokens=event_text.reshape(
                    batch_size, event_count, *event_text.shape[1:]
                ),
                event_text_mask=event_text_masks.reshape(
                    batch_size, event_count, event_text_masks.shape[1]
                ),
            )

        def named_lora_parameters(self) -> tuple[tuple[str, Any], ...]:
            values = []
            for module_name, module in self.top_language_model.named_modules():
                if isinstance(module, LoRALinear):
                    values.extend(
                        (
                            (
                                f"top_language_model.{module_name}.lora_a.weight",
                                module.lora_a.weight,
                            ),
                            (
                                f"top_language_model.{module_name}.lora_b.weight",
                                module.lora_b.weight,
                            ),
                        )
                    )
            return tuple(values)

        def load_head_state_dict(
            self, state_dict: Mapping[str, Any], *, strict: bool = True
        ) -> Any:
            """Warm-start the complete resampler/set/marginal head checkpoint."""
            return self.predictor.load_state_dict(state_dict, strict=strict)

        def named_head_parameters(self) -> tuple[tuple[str, Any], ...]:
            return tuple(
                (f"predictor.{name}", parameter)
                for name, parameter in self.predictor.named_parameters()
                if parameter.requires_grad
            )

        def optimizer_parameter_groups(
            self,
            *,
            lora_learning_rate: float,
            head_learning_rate: float,
            weight_decay: float,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            for name, value in (
                ("LoRA learning rate", lora_learning_rate),
                ("head learning rate", head_learning_rate),
            ):
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"{name} must be numeric")
                if float(value) <= 0.0:
                    raise ValueError(f"{name} must be positive")
            if not isinstance(weight_decay, (int, float)) or isinstance(
                weight_decay, bool
            ):
                raise TypeError("weight decay must be numeric")
            if float(weight_decay) < 0.0:
                raise ValueError("weight decay must be non-negative")
            lora = [parameter for _, parameter in self.named_lora_parameters()]
            head = [parameter for _, parameter in self.named_head_parameters()]
            if not lora or not head:
                raise RuntimeError("selector optimizer parameter group is empty")
            if set(map(id, lora)) & set(map(id, head)):
                raise RuntimeError("selector optimizer parameter groups overlap")
            trainable = {
                id(parameter)
                for parameter in self.parameters()
                if parameter.requires_grad
            }
            if trainable != set(map(id, (*lora, *head))):
                raise RuntimeError(
                    "selector trainable parameters escaped optimizer groups"
                )
            return (
                {
                    "name": "selector_lora",
                    "params": lora,
                    "lr": float(lora_learning_rate),
                    "weight_decay": float(weight_decay),
                },
                {
                    "name": "selector_head",
                    "params": head,
                    "lr": float(head_learning_rate),
                    "weight_decay": float(weight_decay),
                },
            )

        def forward(
            self,
            *,
            query_entities: Sequence[FullSequenceBoundaryEntity],
            event_entities: Sequence[Sequence[FullSequenceBoundaryEntity | None]],
            event_numeric_features: Any,
            event_mask: Any,
            subset_masks: Any | None = None,
            selected_masks: Any | None = None,
            entity_microbatch_size: int | None = None,
        ) -> Any:
            tokens = self.encode_predictor_tokens(
                query_entities=query_entities,
                event_entities=event_entities,
                event_mask=event_mask,
                entity_microbatch_size=entity_microbatch_size,
            )
            device = tokens.query_visual_tokens.device
            return self.predictor(
                **tokens.as_kwargs(),
                event_numeric_features=event_numeric_features.to(device=device),
                event_mask=event_mask.to(device=device),
                subset_masks=(
                    None if subset_masks is None else subset_masks.to(device=device)
                ),
                selected_masks=(
                    None if selected_masks is None else selected_masks.to(device=device)
                ),
            )

else:  # pragma: no cover

    class SelectorSideLoRAMarginalPredictor:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("selector-side LoRA requires PyTorch")


__all__ = [
    "FullSequenceBoundaryEntity",
    "SELECTOR_TOP_LAYER_COUNT",
    "SelectorPredictorTokens",
    "SelectorSideLoRAMarginalPredictor",
]
