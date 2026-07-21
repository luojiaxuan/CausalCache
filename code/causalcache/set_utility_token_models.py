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
    preserve_entity_latents: bool = False

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
        if type(self.preserve_entity_latents) is not bool:
            raise TypeError("preserve_entity_latents must be boolean")


@dataclass(frozen=True)
class EncodedTokenUtilityState:
    """Reusable query/event encodings for scoring multiple subset batches."""

    query: Any
    events: Any
    event_mask: Any


@dataclass(frozen=True)
class EncodedConditionalMarginalState:
    """Subset-independent state used by the direct conditional marginal head."""

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
            latents = self.output_norm(latents)
            return (
                latents if self.config.preserve_entity_latents else latents.mean(dim=1)
            )

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
                (flat_count, *encoded_valid_events.shape[1:])
            )
            flat_events.index_copy_(0, valid_indices, encoded_valid_events)
            return flat_events.reshape(
                batch_size, event_count, *encoded_valid_events.shape[1:]
            )

        def condition_encoded_state(
            self,
            *,
            query: Any,
            event_sources: Any,
            event_numeric_features: Any,
            event_mask: Any,
        ) -> EncodedTokenUtilityState:
            """Apply query-time numeric and query/event conditioning."""
            if self.config.preserve_entity_latents:
                if query.ndim != 3 or event_sources.ndim != 4:
                    raise ValueError(
                        "multi-latent query/event tensors have invalid rank"
                    )
                batch_size, event_count, latent_count, hidden_size = event_sources.shape
                if query.shape != (batch_size, latent_count, hidden_size):
                    raise ValueError("multi-latent query/event geometry drifted")
                query_summary = query.mean(dim=1)
                expanded_query = query_summary[:, None, None, :].expand(
                    -1, event_count, latent_count, -1
                )
            else:
                if query.ndim != 2 or event_sources.ndim != 3:
                    raise ValueError("encoded query/event tensors have invalid rank")
                batch_size, event_count, hidden_size = event_sources.shape
                if query.shape != (batch_size, hidden_size):
                    raise ValueError("encoded query/event geometry drifted")
                expanded_query = query.unsqueeze(1).expand(-1, event_count, -1)
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
            numeric = self.numeric_encoder(event_numeric_features)
            if self.config.preserve_entity_latents:
                numeric = numeric.unsqueeze(2).expand(-1, -1, latent_count, -1)
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
                events=events.masked_fill(
                    ~event_mask.reshape(
                        batch_size,
                        event_count,
                        *([1, 1] if self.config.preserve_entity_latents else [1]),
                    ),
                    0.0,
                ),
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
            cardinality = torch.log1p(memberships.sum(dim=-1, keepdim=True))
            if self.config.family == "deepsets":
                if self.config.preserve_entity_latents:
                    selected = (
                        torch.einsum("bkn,bnlh->bkh", memberships, events)
                        / events.shape[2]
                    )
                    universe = (
                        events.sum(dim=(1, 2), keepdim=False) / events.shape[2]
                    ).unsqueeze(1)
                    query_for_head = query.mean(dim=1)
                else:
                    selected = torch.einsum("bkn,bnh->bkh", memberships, events)
                    universe = events.sum(dim=1, keepdim=True)
                    query_for_head = query
                unselected = universe - selected
                expanded_query = query_for_head.unsqueeze(1).expand(
                    -1, subset_count, -1
                )
                return self.utility_head(
                    torch.cat(
                        (expanded_query, selected, unselected, cardinality), dim=-1
                    )
                ).squeeze(-1)

            flat_membership = subset_masks.reshape(
                batch_size * subset_count, event_count
            )
            if self.config.preserve_entity_latents:
                latent_count = events.shape[2]
                expanded_events = (
                    events.unsqueeze(1)
                    .expand(-1, subset_count, -1, -1, -1)
                    .reshape(
                        batch_size * subset_count,
                        event_count * latent_count,
                        hidden,
                    )
                )
                latent_membership = (
                    flat_membership.unsqueeze(-1)
                    .expand(-1, -1, latent_count)
                    .reshape(batch_size * subset_count, event_count * latent_count)
                )
                expanded_events = expanded_events + self.selection_embedding(
                    latent_membership.to(dtype=torch.long)
                )
                seed = (
                    query.unsqueeze(1)
                    .expand(-1, subset_count, -1, -1)
                    .reshape(batch_size * subset_count, latent_count, hidden)
                )
                expanded_event_mask = (
                    event_mask.unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(-1, subset_count, -1, latent_count)
                    .reshape(batch_size * subset_count, event_count * latent_count)
                )
                seed_count = latent_count
            else:
                expanded_events = (
                    events.unsqueeze(1)
                    .expand(-1, subset_count, -1, -1)
                    .reshape(batch_size * subset_count, event_count, hidden)
                )
                expanded_events = expanded_events + self.selection_embedding(
                    flat_membership.to(dtype=torch.long)
                )
                seed = (
                    query.unsqueeze(1)
                    .expand(-1, subset_count, -1)
                    .reshape(batch_size * subset_count, 1, hidden)
                )
                expanded_event_mask = (
                    event_mask.unsqueeze(1)
                    .expand(-1, subset_count, -1)
                    .reshape(batch_size * subset_count, event_count)
                )
                seed_count = 1
            tokens = torch.cat((seed, expanded_events), dim=1)
            padding_mask = torch.cat(
                (
                    torch.zeros(
                        (batch_size * subset_count, seed_count),
                        dtype=torch.bool,
                        device=event_mask.device,
                    ),
                    ~expanded_event_mask,
                ),
                dim=1,
            )
            assert self.set_encoder is not None
            pooled = (
                self.set_encoder(tokens, src_key_padding_mask=padding_mask)[
                    :, :seed_count
                ]
                .mean(dim=1)
                .reshape(batch_size, subset_count, hidden)
            )
            return self.utility_head(torch.cat((pooled, cardinality), dim=-1)).squeeze(
                -1
            )

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
                query.ndim != (3 if self.config.preserve_entity_latents else 2)
                or events.ndim != (4 if self.config.preserve_entity_latents else 3)
                or event_mask.ndim != 2
                or query.shape[0] != events.shape[0]
                or event_mask.shape != events.shape[:2]
                or query.shape[-1] != events.shape[-1]
            ):
                raise ValueError("encoded token-utility state geometry drifted")
            if (
                self.config.preserve_entity_latents
                and query.shape[1] != events.shape[2]
            ):
                raise ValueError("encoded token-utility latent count drifted")
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

    class TokenConditionalMarginalPredictor(torch.nn.Module):
        """Score every event's gain conditional on a selected coalition and STOP."""

        def __init__(self, config: TokenUtilityModelConfig) -> None:
            super().__init__()
            if config.family != "set_transformer":
                raise ValueError(
                    "conditional marginal predictor requires set_transformer"
                )
            if not config.preserve_entity_latents:
                raise ValueError(
                    "conditional marginal predictor requires entity latents"
                )
            self.config = config
            self.encoder = TokenSetUtilityPredictor(config)
            hidden = config.hidden_size
            self.event_pool = torch.nn.MultiheadAttention(
                hidden,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.event_pool_norm = torch.nn.LayerNorm(hidden)
            self.selected_attention = torch.nn.MultiheadAttention(
                hidden,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.selected_norm = torch.nn.LayerNorm(hidden)
            self.empty_selected = torch.nn.Parameter(torch.empty(hidden))
            torch.nn.init.normal_(self.empty_selected, std=hidden**-0.5)
            self.marginal_head = torch.nn.Sequential(
                torch.nn.Linear(7 * hidden, 2 * hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(2 * hidden, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, 1),
            )
            # note (luojiaxuan): Scalar-only parameters stay outside this optimizer;
            # the candidate set encoder remains trainable and runs once per state.
            for module in (
                self.encoder.selection_embedding,
                self.encoder.utility_head,
            ):
                if module is not None:
                    module.requires_grad_(False)

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
        ) -> EncodedConditionalMarginalState:
            encoded = self.encoder.encode_state_once(
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
            batch_size, event_count, latent_count, hidden = encoded.events.shape
            query = encoded.query.mean(dim=1)
            flat_events = encoded.events.reshape(
                batch_size * event_count, latent_count, hidden
            )
            pool_queries = (
                query[:, None, :]
                .expand(-1, event_count, -1)
                .reshape(batch_size * event_count, 1, hidden)
            )
            pooled, _ = self.event_pool(pool_queries, flat_events, flat_events)
            events = self.event_pool_norm(
                pooled.squeeze(1) + pool_queries.squeeze(1)
            ).reshape(batch_size, event_count, hidden)
            events = events.masked_fill(~event_mask.unsqueeze(-1), 0.0)

            tokens = torch.cat((query.unsqueeze(1), events), dim=1)
            padding_mask = torch.cat(
                (
                    torch.zeros(
                        (batch_size, 1),
                        dtype=torch.bool,
                        device=event_mask.device,
                    ),
                    ~event_mask,
                ),
                dim=1,
            )
            assert self.encoder.set_encoder is not None
            contextual = self.encoder.set_encoder(
                tokens, src_key_padding_mask=padding_mask
            )
            return EncodedConditionalMarginalState(
                query=contextual[:, 0],
                events=contextual[:, 1:].masked_fill(~event_mask.unsqueeze(-1), 0.0),
                event_mask=event_mask,
            )

        def score_encoded_candidates(
            self,
            encoded_state: EncodedConditionalMarginalState,
            selected_masks: Any,
        ) -> Any:
            """Return `[STOP, event_1, ..., event_n]` gains without re-encoding."""
            if not isinstance(encoded_state, EncodedConditionalMarginalState):
                raise TypeError("encoded_state must be EncodedConditionalMarginalState")
            if selected_masks.ndim == 2:
                selected_masks = selected_masks.unsqueeze(1)
                squeeze = True
            else:
                squeeze = False
            if (
                selected_masks.ndim != 3
                or selected_masks.dtype != torch.bool
                or selected_masks.shape[0] != encoded_state.event_mask.shape[0]
                or selected_masks.shape[2] != encoded_state.event_mask.shape[1]
            ):
                raise ValueError("selected coalition mask geometry drifted")
            if bool((selected_masks & ~encoded_state.event_mask.unsqueeze(1)).any()):
                raise ValueError("a selected coalition contains a padded event")

            events = encoded_state.events
            query = encoded_state.query
            event_mask = encoded_state.event_mask
            batch_size, group_count, event_count = selected_masks.shape
            hidden = events.shape[-1]
            flat_count = batch_size * group_count
            candidates = (
                events.unsqueeze(1)
                .expand(-1, group_count, -1, -1)
                .reshape(flat_count, event_count, hidden)
            )
            selected_events = candidates
            empty = self.empty_selected.reshape(1, 1, hidden).expand(flat_count, -1, -1)
            keys = torch.cat((empty, selected_events), dim=1)
            flat_selected = selected_masks.reshape(flat_count, event_count)
            has_selected = flat_selected.any(dim=1, keepdim=True)
            selected_padding_mask = torch.cat((has_selected, ~flat_selected), dim=1)
            selected_context, _ = self.selected_attention(
                candidates,
                keys,
                keys,
                key_padding_mask=selected_padding_mask,
            )
            selected_context = self.selected_norm(
                candidates + selected_context
            ).reshape(batch_size, group_count, event_count, hidden)

            memberships = event_mask.to(dtype=events.dtype)
            universe = torch.einsum("bn,bnh->bh", memberships, events)
            universe = universe / memberships.sum(dim=1, keepdim=True).clamp_min(1.0)
            expanded_events = events.unsqueeze(1).expand(-1, group_count, -1, -1)
            expanded_query = query[:, None, None, :].expand_as(expanded_events)
            expanded_universe = universe[:, None, None, :].expand_as(expanded_events)
            scores = self.marginal_head(
                torch.cat(
                    (
                        expanded_events,
                        expanded_query,
                        selected_context,
                        expanded_events * expanded_query,
                        expanded_events * selected_context,
                        torch.abs(expanded_events - selected_context),
                        expanded_universe,
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
            scores = scores.masked_fill(~event_mask[:, None, :].expand_as(scores), 0.0)
            stop = torch.zeros(
                (batch_size, group_count, 1),
                dtype=scores.dtype,
                device=scores.device,
            )
            result = torch.cat((stop, scores), dim=2)
            return result.squeeze(1) if squeeze else result

        def score_singleton_utilities(
            self,
            encoded_state: EncodedConditionalMarginalState,
            subset_masks: Any,
        ) -> Any:
            """Compatibility path for the train-only empty-coalition fit probe."""
            if bool((subset_masks.sum(dim=-1) > 1).any()):
                raise ValueError("singleton fit probe received a larger subset")
            empty_selected = torch.zeros_like(subset_masks[:, :1])
            event_scores = self.score_encoded_candidates(encoded_state, empty_selected)[
                :, 0, 1:
            ]
            utilities = torch.einsum(
                "bkn,bn->bk",
                subset_masks.to(dtype=event_scores.dtype),
                event_scores,
            )
            return utilities.masked_fill(~subset_masks.any(dim=-1), 0.0)

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
            subset_masks: Any | None = None,
            selected_masks: Any | None = None,
        ) -> Any:
            if (subset_masks is None) == (selected_masks is None):
                raise ValueError(
                    "exactly one of subset_masks or selected_masks is required"
                )
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
            if selected_masks is not None:
                return self.score_encoded_candidates(encoded, selected_masks)
            return self.score_singleton_utilities(encoded, subset_masks)

    class TokenSingletonMarginalPredictor(TokenConditionalMarginalPredictor):
        """Backward-compatible name for the empty-coalition v3 fit probe."""

else:

    class MultimodalLatentResampler:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("token utility models require PyTorch")

    class TokenSetUtilityPredictor:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("token utility models require PyTorch")

    class TokenSingletonMarginalPredictor:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("token utility models require PyTorch")

    class TokenConditionalMarginalPredictor:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("token utility models require PyTorch")


__all__ = [
    "EncodedConditionalMarginalState",
    "EncodedTokenUtilityState",
    "MODEL_FAMILIES",
    "MultimodalLatentResampler",
    "TokenConditionalMarginalPredictor",
    "TokenSingletonMarginalPredictor",
    "TokenSetUtilityPredictor",
    "TokenUtilityModelConfig",
]
