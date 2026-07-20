"""Token-level multimodal predictors for restoration subset utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local lightweight installs.
    torch = None


MODEL_FAMILIES = ("deepsets", "set_transformer")


@dataclass(frozen=True)
class TokenUtilityModelConfig:
    """Architecture shared by the paper model and its aggregation baseline."""

    family: str
    source_hidden_size: int = 4096
    numeric_feature_size: int = 11
    hidden_size: int = 256
    latent_count: int = 16
    resampler_layers: int = 2
    set_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.family not in MODEL_FAMILIES:
            raise ValueError(f"unknown token utility family: {self.family}")
        for name in (
            "source_hidden_size",
            "numeric_feature_size",
            "hidden_size",
            "latent_count",
            "resampler_layers",
            "set_layers",
            "num_heads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise TypeError("dropout must be numeric")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class EncodedTokenUtilityState:
    """Reusable query/event encodings for scoring multiple subset batches."""

    query: Any
    events: Any
    event_mask: Any


if torch is not None:

    class _CrossAttentionBlock(torch.nn.Module):
        def __init__(self, config: TokenUtilityModelConfig) -> None:
            super().__init__()
            hidden = config.hidden_size
            self.query_norm = torch.nn.LayerNorm(hidden)
            self.source_norm = torch.nn.LayerNorm(hidden)
            self.attention = torch.nn.MultiheadAttention(
                hidden,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.output_norm = torch.nn.LayerNorm(hidden)
            self.feed_forward = torch.nn.Sequential(
                torch.nn.Linear(hidden, 4 * hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(4 * hidden, hidden),
            )

        def forward(
            self,
            latents: Any,
            source: Any,
            source_padding_mask: Any,
        ) -> Any:
            query = self.query_norm(latents)
            normalized_source = self.source_norm(source)
            attended, _ = self.attention(
                query,
                normalized_source,
                normalized_source,
                key_padding_mask=source_padding_mask,
                need_weights=False,
            )
            latents = latents + attended
            return latents + self.feed_forward(self.output_norm(latents))


    class MultimodalLatentResampler(torch.nn.Module):
        """Compress full GUI-Owl image/text sequences with learned queries."""

        def __init__(self, config: TokenUtilityModelConfig) -> None:
            super().__init__()
            self.config = config
            self.visual_projection = torch.nn.Linear(
                config.source_hidden_size,
                config.hidden_size,
                bias=False,
            )
            self.text_projection = torch.nn.Linear(
                config.source_hidden_size,
                config.hidden_size,
                bias=False,
            )
            self.modality_embedding = torch.nn.Embedding(2, config.hidden_size)
            self.role_embedding = torch.nn.Embedding(2, config.hidden_size)
            self.latents = torch.nn.Parameter(
                torch.empty(config.latent_count, config.hidden_size)
            )
            torch.nn.init.normal_(self.latents, std=config.hidden_size**-0.5)
            self.blocks = torch.nn.ModuleList(
                _CrossAttentionBlock(config) for _ in range(config.resampler_layers)
            )
            self.output_norm = torch.nn.LayerNorm(config.hidden_size)

        def forward(
            self,
            visual_tokens: Any,
            visual_mask: Any,
            text_tokens: Any,
            text_mask: Any,
            role_ids: Any,
        ) -> Any:
            if visual_tokens.ndim != 3 or text_tokens.ndim != 3:
                raise ValueError("entity token tensors must be rank three")
            if visual_tokens.shape[0] != text_tokens.shape[0]:
                raise ValueError("visual and text entity batches differ")
            if visual_tokens.shape[-1] != self.config.source_hidden_size:
                raise ValueError("visual source hidden size drifted")
            if text_tokens.shape[-1] != self.config.source_hidden_size:
                raise ValueError("text source hidden size drifted")
            if visual_mask.shape != visual_tokens.shape[:2]:
                raise ValueError("visual token mask shape drifted")
            if text_mask.shape != text_tokens.shape[:2]:
                raise ValueError("text token mask shape drifted")
            if visual_mask.dtype != torch.bool or text_mask.dtype != torch.bool:
                raise TypeError("entity token masks must be boolean")
            if role_ids.shape != (visual_tokens.shape[0],):
                raise ValueError("entity role ids shape drifted")
            visual = self.visual_projection(visual_tokens)
            text = self.text_projection(text_tokens)
            visual = visual + self.modality_embedding.weight[0]
            text = text + self.modality_embedding.weight[1]
            source = torch.cat((visual, text), dim=1)
            source_padding_mask = ~torch.cat((visual_mask, text_mask), dim=1)
            if bool(source_padding_mask.all(dim=1).any()):
                raise ValueError("an entity has no visible image or text token")
            latents = self.latents.unsqueeze(0).expand(source.shape[0], -1, -1)
            latents = latents + self.role_embedding(role_ids).unsqueeze(1)
            for block in self.blocks:
                latents = block(latents, source, source_padding_mask)
            return self.output_norm(latents).mean(dim=1)


    class TokenSetUtilityPredictor(torch.nn.Module):
        """Predict U(S) from unpooled frozen VLM tokens and a selected mask."""

        def __init__(self, config: TokenUtilityModelConfig) -> None:
            super().__init__()
            if not isinstance(config, TokenUtilityModelConfig):
                raise TypeError("config must be TokenUtilityModelConfig")
            self.config = config
            hidden = config.hidden_size
            self.entity_encoder = MultimodalLatentResampler(config)
            self.numeric_encoder = torch.nn.Sequential(
                torch.nn.Linear(config.numeric_feature_size, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
            )
            self.event_conditioner = torch.nn.Sequential(
                torch.nn.Linear(5 * hidden, 2 * hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(2 * hidden, hidden),
            )
            self.selection_embedding = torch.nn.Embedding(2, hidden)
            if config.family == "set_transformer":
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=config.num_heads,
                    dim_feedforward=4 * hidden,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.set_encoder = torch.nn.TransformerEncoder(
                    layer,
                    num_layers=config.set_layers,
                    norm=torch.nn.LayerNorm(hidden),
                    enable_nested_tensor=False,
                )
                utility_input_size = hidden + 1
            else:
                self.set_encoder = None
                utility_input_size = 3 * hidden + 1
            self.utility_head = torch.nn.Sequential(
                torch.nn.Linear(utility_input_size, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, 1),
            )

        def encode_query_source(
            self,
            *,
            query_visual_tokens: Any,
            query_visual_mask: Any,
            query_text_tokens: Any,
            query_text_mask: Any,
        ) -> Any:
            """Resample current-query source tokens independently of event ingest."""
            if query_visual_tokens.ndim != 3 or query_text_tokens.ndim != 3:
                raise ValueError("query token tensors must be rank three")
            batch_size = query_visual_tokens.shape[0]
            if query_text_tokens.shape[0] != batch_size:
                raise ValueError("query visual/text batches differ")
            query_roles = torch.zeros(
                (batch_size,), dtype=torch.long, device=query_visual_tokens.device
            )
            return self.entity_encoder(
                query_visual_tokens,
                query_visual_mask,
                query_text_tokens,
                query_text_mask,
                query_roles,
            )

        def encode_event_sources(
            self,
            *,
            event_visual_tokens: Any,
            event_visual_mask: Any,
            event_text_tokens: Any,
            event_text_mask: Any,
            event_mask: Any,
        ) -> Any:
            """Resample valid event sources once and preserve padded geometry."""
            if event_visual_tokens.ndim != 4 or event_text_tokens.ndim != 4:
                raise ValueError("event token tensors must be rank four")
            batch_size, event_count = event_visual_tokens.shape[:2]
            if event_text_tokens.shape[:2] != (batch_size, event_count):
                raise ValueError("event visual/text batches differ")
            if event_mask.shape != (batch_size, event_count):
                raise ValueError("event mask shape drifted")
            if event_mask.dtype != torch.bool:
                raise TypeError("event mask must be boolean")
            if bool((~event_mask).all(dim=1).any()):
                raise ValueError("every state must contain a candidate event")
            flat_count = batch_size * event_count
            flat_event_mask = event_mask.reshape(flat_count)
            valid_indices = flat_event_mask.nonzero(as_tuple=False).squeeze(1)
            event_roles = torch.ones(
                (valid_indices.shape[0],),
                dtype=torch.long,
                device=event_visual_tokens.device,
            )
            flat_visual = event_visual_tokens.reshape(
                flat_count,
                event_visual_tokens.shape[2],
                event_visual_tokens.shape[3],
            )
            flat_visual_mask = event_visual_mask.reshape(
                flat_count, event_visual_mask.shape[2]
            )
            flat_text = event_text_tokens.reshape(
                flat_count,
                event_text_tokens.shape[2],
                event_text_tokens.shape[3],
            )
            flat_text_mask = event_text_mask.reshape(
                flat_count, event_text_mask.shape[2]
            )
            encoded_valid_events = self.entity_encoder(
                flat_visual.index_select(0, valid_indices),
                flat_visual_mask.index_select(0, valid_indices),
                flat_text.index_select(0, valid_indices),
                flat_text_mask.index_select(0, valid_indices),
                event_roles,
            )
            flat_events = encoded_valid_events.new_zeros(
                (flat_count, encoded_valid_events.shape[-1])
            )
            flat_events.index_copy_(0, valid_indices, encoded_valid_events)
            return flat_events.reshape(batch_size, event_count, -1)

        def condition_encoded_state(
            self,
            *,
            query: Any,
            event_sources: Any,
            event_numeric_features: Any,
            event_mask: Any,
        ) -> EncodedTokenUtilityState:
            """Apply query-time numeric and query/event conditioning."""
            if query.ndim != 2 or event_sources.ndim != 3:
                raise ValueError("encoded query/event tensors have invalid rank")
            batch_size, event_count, hidden_size = event_sources.shape
            if query.shape != (batch_size, hidden_size):
                raise ValueError("encoded query/event geometry drifted")
            if event_numeric_features.shape != (
                batch_size,
                event_count,
                self.config.numeric_feature_size,
            ):
                raise ValueError("event numeric feature shape drifted")
            if event_mask.shape != (batch_size, event_count):
                raise ValueError("event mask shape drifted")
            if event_mask.dtype != torch.bool:
                raise TypeError("event mask must be boolean")
            expanded_query = query.unsqueeze(1).expand(-1, event_count, -1)
            numeric = self.numeric_encoder(event_numeric_features)
            events = self.event_conditioner(
                torch.cat(
                    (
                        event_sources,
                        expanded_query,
                        event_sources * expanded_query,
                        torch.abs(event_sources - expanded_query),
                        numeric,
                    ),
                    dim=-1,
                )
            )
            return EncodedTokenUtilityState(
                query=query,
                events=events.masked_fill(~event_mask.unsqueeze(-1), 0.0),
                event_mask=event_mask,
            )

        def _encode_state(
            self,
            query_visual_tokens: Any,
            query_visual_mask: Any,
            query_text_tokens: Any,
            query_text_mask: Any,
            event_visual_tokens: Any,
            event_visual_mask: Any,
            event_text_tokens: Any,
            event_text_mask: Any,
            event_numeric_features: Any,
            event_mask: Any,
        ) -> tuple[Any, Any]:
            encoded = self.encode_state_once(
                query_visual_tokens=query_visual_tokens,
                query_visual_mask=query_visual_mask,
                query_text_tokens=query_text_tokens,
                query_text_mask=query_text_mask,
                event_visual_tokens=event_visual_tokens,
                event_visual_mask=event_visual_mask,
                event_text_tokens=event_text_tokens,
                event_text_mask=event_text_mask,
                event_numeric_features=event_numeric_features,
                event_mask=event_mask,
            )
            return encoded.query, encoded.events

        def encode_state_once(
            self,
            *,
            query_visual_tokens: Any,
            query_visual_mask: Any,
            query_text_tokens: Any,
            query_text_mask: Any,
            event_visual_tokens: Any,
            event_visual_mask: Any,
            event_text_tokens: Any,
            event_text_mask: Any,
            event_numeric_features: Any,
            event_mask: Any,
        ) -> EncodedTokenUtilityState:
            """Run source-token resampling once before batched subset scoring."""
            query = self.encode_query_source(
                query_visual_tokens=query_visual_tokens,
                query_visual_mask=query_visual_mask,
                query_text_tokens=query_text_tokens,
                query_text_mask=query_text_mask,
            )
            event_sources = self.encode_event_sources(
                event_visual_tokens=event_visual_tokens,
                event_visual_mask=event_visual_mask,
                event_text_tokens=event_text_tokens,
                event_text_mask=event_text_mask,
                event_mask=event_mask,
            )
            return self.condition_encoded_state(
                query=query,
                event_sources=event_sources,
                event_numeric_features=event_numeric_features,
                event_mask=event_mask,
            )

        def _raw_scores(
            self,
            query: Any,
            events: Any,
            subset_masks: Any,
            event_mask: Any,
        ) -> Any:
            batch_size, subset_count, event_count = subset_masks.shape
            hidden = events.shape[-1]
            memberships = subset_masks.to(dtype=events.dtype)
            cardinality = torch.log1p(
                memberships.sum(dim=-1, keepdim=True)
            )
            if self.config.family == "deepsets":
                selected = torch.einsum("bkn,bnh->bkh", memberships, events)
                universe = events.sum(dim=1, keepdim=True)
                unselected = universe - selected
                expanded_query = query.unsqueeze(1).expand(-1, subset_count, -1)
                return self.utility_head(
                    torch.cat(
                        (expanded_query, selected, unselected, cardinality), dim=-1
                    )
                ).squeeze(-1)

            flat_membership = subset_masks.reshape(
                batch_size * subset_count, event_count
            )
            expanded_events = events.unsqueeze(1).expand(
                -1, subset_count, -1, -1
            ).reshape(batch_size * subset_count, event_count, hidden)
            expanded_events = expanded_events + self.selection_embedding(
                flat_membership.to(dtype=torch.long)
            )
            seed = query.unsqueeze(1).expand(-1, subset_count, -1).reshape(
                batch_size * subset_count, 1, hidden
            )
            tokens = torch.cat((seed, expanded_events), dim=1)
            expanded_event_mask = event_mask.unsqueeze(1).expand(
                -1, subset_count, -1
            ).reshape(batch_size * subset_count, event_count)
            padding_mask = torch.cat(
                (
                    torch.zeros(
                        (batch_size * subset_count, 1),
                        dtype=torch.bool,
                        device=event_mask.device,
                    ),
                    ~expanded_event_mask,
                ),
                dim=1,
            )
            assert self.set_encoder is not None
            pooled = self.set_encoder(
                tokens, src_key_padding_mask=padding_mask
            )[:, 0].reshape(batch_size, subset_count, hidden)
            return self.utility_head(
                torch.cat((pooled, cardinality), dim=-1)
            ).squeeze(-1)

        def score_encoded_subsets(
            self,
            encoded_state: EncodedTokenUtilityState,
            subset_masks: Any,
        ) -> Any:
            """Score one subset batch without rerunning query/event resampling."""
            if not isinstance(encoded_state, EncodedTokenUtilityState):
                raise TypeError("encoded_state must be EncodedTokenUtilityState")
            query = encoded_state.query
            events = encoded_state.events
            event_mask = encoded_state.event_mask
            if (
                query.ndim != 2
                or events.ndim != 3
                or event_mask.ndim != 2
                or query.shape[0] != events.shape[0]
                or event_mask.shape != events.shape[:2]
                or query.shape[1] != events.shape[2]
            ):
                raise ValueError("encoded token-utility state geometry drifted")
            if event_mask.dtype != torch.bool:
                raise TypeError("encoded event mask must be boolean")
            if subset_masks.ndim == 2:
                subset_masks = subset_masks.unsqueeze(1)
                squeeze = True
            else:
                squeeze = False
            if subset_masks.ndim != 3 or subset_masks.shape[:1] != event_mask.shape[:1]:
                raise ValueError("subset mask shape drifted")
            if subset_masks.shape[2] != event_mask.shape[1]:
                raise ValueError("subset event dimension drifted")
            if subset_masks.dtype != torch.bool:
                raise TypeError("subset mask must be boolean")
            if bool((subset_masks & ~event_mask.unsqueeze(1)).any()):
                raise ValueError("a subset selects a padded event")
            raw = self._raw_scores(query, events, subset_masks, event_mask)
            empty_masks = torch.zeros_like(subset_masks[:, :1])
            empty = self._raw_scores(query, events, empty_masks, event_mask)
            utilities = (raw - empty).masked_fill(~subset_masks.any(dim=-1), 0.0)
            return utilities.squeeze(1) if squeeze else utilities

        def forward(
            self,
            *,
            query_visual_tokens: Any,
            query_visual_mask: Any,
            query_text_tokens: Any,
            query_text_mask: Any,
            event_visual_tokens: Any,
            event_visual_mask: Any,
            event_text_tokens: Any,
            event_text_mask: Any,
            event_numeric_features: Any,
            event_mask: Any,
            subset_masks: Any,
        ) -> Any:
            encoded_state = self.encode_state_once(
                query_visual_tokens=query_visual_tokens,
                query_visual_mask=query_visual_mask,
                query_text_tokens=query_text_tokens,
                query_text_mask=query_text_mask,
                event_visual_tokens=event_visual_tokens,
                event_visual_mask=event_visual_mask,
                event_text_tokens=event_text_tokens,
                event_text_mask=event_text_mask,
                event_numeric_features=event_numeric_features,
                event_mask=event_mask,
            )
            return self.score_encoded_subsets(encoded_state, subset_masks)


else:

    class MultimodalLatentResampler:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("token utility models require PyTorch")


    class TokenSetUtilityPredictor:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("token utility models require PyTorch")


__all__ = [
    "EncodedTokenUtilityState",
    "MODEL_FAMILIES",
    "MultimodalLatentResampler",
    "TokenSetUtilityPredictor",
    "TokenUtilityModelConfig",
]
