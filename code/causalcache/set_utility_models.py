"""Budget-agnostic neural predictors for restoration set utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised by lightweight installs.
    torch = None


@dataclass(frozen=True)
class SetUtilityDimensions:
    """Feature dimensions shared by the set-utility model families."""

    query: int
    context: int
    event: int
    hidden: int = 128

    def __post_init__(self) -> None:
        for name in ("query", "context", "event", "hidden"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"set-utility {name} dimension must be positive")


def _validate_pair_feature_dimension(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("pair feature dimension must be non-negative")
    return value


def _validate_set_transformer_hyperparameters(
    dimensions: SetUtilityDimensions,
    *,
    num_heads: int,
    num_layers: int,
    dropout: float,
) -> tuple[int, int, float]:
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError("set-transformer num_heads must be a positive integer")
    if dimensions.hidden % num_heads:
        raise ValueError("set-transformer hidden dimension must be divisible by num_heads")
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError("set-transformer num_layers must be a positive integer")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
        raise TypeError("set-transformer dropout must be numeric")
    dropout = float(dropout)
    if not 0.0 <= dropout < 1.0:
        raise ValueError("set-transformer dropout must be in [0, 1)")
    return num_heads, num_layers, dropout


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("set-utility models require PyTorch")
    return torch


if torch is not None:

    class _SetUtilityPredictor(torch.nn.Module):
        """Shared fail-closed input contract for batched subset scoring."""

        def __init__(self, dimensions: SetUtilityDimensions) -> None:
            super().__init__()
            if not isinstance(dimensions, SetUtilityDimensions):
                raise TypeError("dimensions must be SetUtilityDimensions")
            self.dimensions = dimensions

        def _validated_inputs(
            self,
            query_features: Any,
            context_features: Any,
            event_features: Any,
            subset_masks: Any,
            event_mask: Any | None,
        ) -> tuple[Any, Any, Any, Any, Any, bool]:
            tensors = (query_features, context_features, event_features, subset_masks)
            if any(not isinstance(value, torch.Tensor) for value in tensors):
                raise TypeError("set-utility inputs must be PyTorch tensors")
            if query_features.ndim != 2 or query_features.shape[-1] != self.dimensions.query:
                raise ValueError("query feature shape drifted")
            if (
                context_features.ndim != 2
                or context_features.shape[-1] != self.dimensions.context
            ):
                raise ValueError("context feature shape drifted")
            if event_features.ndim != 3 or event_features.shape[-1] != self.dimensions.event:
                raise ValueError("event feature shape drifted")
            batch_size = query_features.shape[0]
            event_count = event_features.shape[1]
            if context_features.shape[0] != batch_size or event_features.shape[0] != batch_size:
                raise ValueError("set-utility batch dimensions differ")
            feature_tensors = (query_features, context_features, event_features)
            if any(not value.is_floating_point() for value in feature_tensors):
                raise TypeError("set-utility features must use floating-point tensors")
            if any(value.device != query_features.device for value in feature_tensors):
                raise ValueError("set-utility features must share one device")
            if any(value.dtype != query_features.dtype for value in feature_tensors):
                raise ValueError("set-utility features must share one dtype")
            model_reference = next(self.parameters())
            if (
                query_features.device != model_reference.device
                or query_features.dtype != model_reference.dtype
            ):
                raise ValueError("set-utility features must match the model device and dtype")
            if any(not bool(torch.isfinite(value).all()) for value in feature_tensors):
                raise ValueError("set-utility features must be finite")

            single_subset = subset_masks.ndim == 2
            if single_subset:
                subset_masks = subset_masks.unsqueeze(1)
            if (
                subset_masks.ndim != 3
                or subset_masks.shape[0] != batch_size
                or subset_masks.shape[2] != event_count
                or subset_masks.shape[1] == 0
            ):
                raise ValueError("subset mask shape drifted")
            if subset_masks.dtype != torch.bool or subset_masks.device != query_features.device:
                raise TypeError("subset masks must be boolean tensors on the feature device")
            if event_mask is None:
                event_mask = torch.ones(
                    (batch_size, event_count),
                    dtype=torch.bool,
                    device=query_features.device,
                )
            elif not isinstance(event_mask, torch.Tensor):
                raise TypeError("event mask must be a PyTorch tensor")
            if event_mask.shape != (batch_size, event_count):
                raise ValueError("event mask shape drifted")
            if event_mask.dtype != torch.bool or event_mask.device != query_features.device:
                raise TypeError("event mask must be boolean and on the feature device")
            if bool((subset_masks & ~event_mask.unsqueeze(1)).any()):
                raise ValueError("a subset selects a padded event")
            return (
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask,
                single_subset,
            )

        @staticmethod
        def _restore_subset_shape(values: Any, single_subset: bool) -> Any:
            return values.squeeze(1) if single_subset else values


    class PairwiseAdditiveUtilityPredictor(_SetUtilityPredictor):
        """Unary-plus-pairwise utility baseline for arbitrary-size subsets."""

        def __init__(
            self,
            dimensions: SetUtilityDimensions,
            *,
            pair_feature_dimension: int = 0,
        ) -> None:
            super().__init__(dimensions)
            self.pair_feature_dimension = _validate_pair_feature_dimension(
                pair_feature_dimension
            )
            condition_dimension = dimensions.query + dimensions.context
            hidden = dimensions.hidden
            self.condition_encoder = torch.nn.Sequential(
                torch.nn.Linear(condition_dimension, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(approximate="none"),
            )
            self.universe_encoder = torch.nn.Sequential(
                torch.nn.Linear(dimensions.event + 1, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
            )
            self.event_encoder = torch.nn.Sequential(
                torch.nn.Linear(dimensions.event + hidden, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(approximate="none"),
            )
            self.unary_head = torch.nn.Linear(hidden, 1)
            self.pair_head = torch.nn.Sequential(
                torch.nn.Linear(
                    3 * hidden + self.pair_feature_dimension,
                    hidden,
                ),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, 1),
            )

        def _validated_pair_features(
            self,
            pair_features: Any | None,
            *,
            batch_size: int,
            event_count: int,
            reference: Any,
        ) -> Any:
            expected_shape = (
                batch_size,
                event_count,
                event_count,
                self.pair_feature_dimension,
            )
            if pair_features is None:
                if self.pair_feature_dimension:
                    raise ValueError("configured pair features are missing")
                return torch.empty(
                    expected_shape,
                    dtype=reference.dtype,
                    device=reference.device,
                )
            if not isinstance(pair_features, torch.Tensor):
                raise TypeError("pair features must be a PyTorch tensor")
            if pair_features.shape != expected_shape:
                raise ValueError("pair feature shape drifted")
            if (
                not pair_features.is_floating_point()
                or pair_features.dtype != reference.dtype
                or pair_features.device != reference.device
            ):
                raise TypeError("pair features must match the event feature dtype and device")
            if not bool(torch.isfinite(pair_features).all()):
                raise ValueError("pair features must be finite")
            return pair_features

        def score_subsets(
            self,
            query_features: Any,
            context_features: Any,
            event_features: Any,
            subset_masks: Any,
            event_mask: Any | None = None,
            *,
            pair_features: Any | None = None,
        ) -> Any:
            """Score one or many subsets per state without observing a budget."""
            (
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask,
                single_subset,
            ) = self._validated_inputs(
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask,
            )
            event_count = event_features.shape[1]
            pair_features = self._validated_pair_features(
                pair_features,
                batch_size=event_features.shape[0],
                event_count=event_count,
                reference=event_features,
            )
            condition = self.condition_encoder(
                torch.cat((query_features, context_features), dim=-1)
            )
            valid_memberships = event_mask.to(dtype=event_features.dtype)
            candidate_count = valid_memberships.sum(dim=1, keepdim=True)
            universe_mean = torch.einsum(
                "bn,bne->be", valid_memberships, event_features
            ) / candidate_count.clamp_min(1.0)
            universe_condition = self.universe_encoder(
                torch.cat((universe_mean, torch.log1p(candidate_count)), dim=-1)
            )
            condition = condition + universe_condition
            expanded_condition = condition.unsqueeze(1).expand(-1, event_count, -1)
            encoded_events = self.event_encoder(
                torch.cat((event_features, expanded_condition), dim=-1)
            )
            unary_scores = self.unary_head(encoded_events).squeeze(-1)
            unary_scores = unary_scores.masked_fill(~event_mask, 0.0)

            left = encoded_events.unsqueeze(2)
            right = encoded_events.unsqueeze(1)
            pair_features = torch.cat(
                (
                    left + right,
                    torch.abs(left - right),
                    left * right,
                    0.5 * (pair_features + pair_features.transpose(1, 2)),
                ),
                dim=-1,
            )
            pair_scores = self.pair_head(pair_features).squeeze(-1)
            valid_pairs = event_mask.unsqueeze(2) & event_mask.unsqueeze(1)
            diagonal = torch.eye(
                event_count, dtype=torch.bool, device=event_features.device
            ).unsqueeze(0)
            pair_scores = pair_scores.masked_fill(~valid_pairs | diagonal, 0.0)

            memberships = subset_masks.to(dtype=event_features.dtype)
            unary_utility = torch.einsum("bkn,bn->bk", memberships, unary_scores)
            pair_utility = 0.5 * torch.einsum(
                "bki,bij,bkj->bk", memberships, pair_scores, memberships
            )
            utility = (unary_utility + pair_utility).masked_fill(
                ~subset_masks.any(dim=-1), 0.0
            )
            return self._restore_subset_shape(utility, single_subset)

        def forward(
            self,
            query_features: Any,
            context_features: Any,
            event_features: Any,
            subset_masks: Any,
            event_mask: Any | None = None,
            *,
            pair_features: Any | None = None,
        ) -> Any:
            return self.score_subsets(
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask,
                pair_features=pair_features,
            )


    class DeepSetsUtilityPredictor(_SetUtilityPredictor):
        """Query-conditioned DeepSets predictor for arbitrary-size subsets."""

        def __init__(self, dimensions: SetUtilityDimensions) -> None:
            super().__init__(dimensions)
            condition_dimension = dimensions.query + dimensions.context
            hidden = dimensions.hidden
            self.condition_encoder = torch.nn.Sequential(
                torch.nn.Linear(condition_dimension, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(approximate="none"),
            )
            self.element_encoder = torch.nn.Sequential(
                torch.nn.Linear(dimensions.event + hidden, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(approximate="none"),
            )
            self.utility_head = torch.nn.Sequential(
                torch.nn.Linear(3 * hidden + 1, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, 1),
            )

        def score_subsets(
            self,
            query_features: Any,
            context_features: Any,
            event_features: Any,
            subset_masks: Any,
            event_mask: Any | None = None,
        ) -> Any:
            """Score one or many subsets per state without observing a budget."""
            (
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask,
                single_subset,
            ) = self._validated_inputs(
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask,
            )
            event_count = event_features.shape[1]
            condition = self.condition_encoder(
                torch.cat((query_features, context_features), dim=-1)
            )
            expanded_condition = condition.unsqueeze(1).expand(-1, event_count, -1)
            elements = self.element_encoder(
                torch.cat((event_features, expanded_condition), dim=-1)
            )
            elements = elements.masked_fill(~event_mask.unsqueeze(-1), 0.0)
            memberships = subset_masks.to(dtype=event_features.dtype)
            selected_pool = torch.einsum("bkn,bnh->bkh", memberships, elements)
            universe_pool = elements.sum(dim=1, keepdim=True)
            unselected_pool = universe_pool - selected_pool
            cardinality = torch.log1p(memberships.sum(dim=-1, keepdim=True))
            expanded_for_subsets = condition.unsqueeze(1).expand(
                -1, subset_masks.shape[1], -1
            )
            raw_utility = self.utility_head(
                torch.cat(
                    (
                        expanded_for_subsets,
                        selected_pool,
                        unselected_pool,
                        cardinality,
                    ),
                    dim=-1,
                )
            ).squeeze(-1)

            empty_selected_pool = torch.zeros_like(selected_pool[:, :1, :])
            empty_cardinality = torch.zeros_like(cardinality[:, :1, :])
            empty_baseline = self.utility_head(
                torch.cat(
                    (
                        condition.unsqueeze(1),
                        empty_selected_pool,
                        universe_pool,
                        empty_cardinality,
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
            utility = (raw_utility - empty_baseline).masked_fill(
                ~subset_masks.any(dim=-1), 0.0
            )
            return self._restore_subset_shape(utility, single_subset)

        def forward(
            self,
            query_features: Any,
            context_features: Any,
            event_features: Any,
            subset_masks: Any,
            event_mask: Any | None = None,
        ) -> Any:
            return self.score_subsets(
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask,
            )


    class SetTransformerUtilityPredictor(_SetUtilityPredictor):
        """Query-token Set Transformer for budget-agnostic subset utility."""

        def __init__(
            self,
            dimensions: SetUtilityDimensions,
            *,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.0,
        ) -> None:
            super().__init__(dimensions)
            self.num_heads, self.num_layers, self.dropout = (
                _validate_set_transformer_hyperparameters(
                    dimensions,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            )
            condition_dimension = dimensions.query + dimensions.context
            hidden = dimensions.hidden
            self.condition_encoder = torch.nn.Sequential(
                torch.nn.Linear(condition_dimension, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
            )
            self.element_encoder = torch.nn.Sequential(
                torch.nn.Linear(dimensions.event + hidden, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
            )
            self.selection_embedding = torch.nn.Embedding(2, hidden)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=self.num_heads,
                dim_feedforward=4 * hidden,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.set_encoder = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.num_layers,
                norm=torch.nn.LayerNorm(hidden),
                enable_nested_tensor=False,
            )
            self.utility_head = torch.nn.Sequential(
                torch.nn.Linear(hidden + 1, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, 1),
            )

        def _raw_subset_scores(
            self,
            condition: Any,
            elements: Any,
            subset_masks: Any,
            event_mask: Any,
        ) -> Any:
            batch_size, subset_count, event_count = subset_masks.shape
            hidden = condition.shape[-1]
            expanded_elements = elements.unsqueeze(1).expand(
                -1, subset_count, -1, -1
            )
            expanded_elements = expanded_elements.reshape(
                batch_size * subset_count, event_count, hidden
            )
            flat_membership = subset_masks.reshape(
                batch_size * subset_count, event_count
            )
            expanded_elements = expanded_elements + self.selection_embedding(
                flat_membership.to(dtype=torch.long)
            )
            seed = condition.unsqueeze(1).expand(-1, subset_count, -1)
            seed = seed.reshape(batch_size * subset_count, 1, hidden)
            tokens = torch.cat((seed, expanded_elements), dim=1)
            seed_is_visible = torch.zeros(
                (batch_size * subset_count, 1),
                dtype=torch.bool,
                device=subset_masks.device,
            )
            expanded_event_mask = event_mask.unsqueeze(1).expand(
                -1, subset_count, -1
            )
            expanded_event_mask = expanded_event_mask.reshape(
                batch_size * subset_count, event_count
            )
            padding_mask = torch.cat(
                (seed_is_visible, ~expanded_event_mask), dim=1
            )
            encoded = self.set_encoder(tokens, src_key_padding_mask=padding_mask)
            pooled = encoded[:, 0].reshape(batch_size, subset_count, hidden)
            cardinality = torch.log1p(
                subset_masks.sum(dim=-1, keepdim=True).to(dtype=elements.dtype)
            )
            return self.utility_head(torch.cat((pooled, cardinality), dim=-1)).squeeze(-1)

        def score_subsets(
            self,
            query_features: Any,
            context_features: Any,
            event_features: Any,
            subset_masks: Any,
            event_mask: Any | None = None,
        ) -> Any:
            """Score complete subsets jointly without positional or budget inputs."""
            (
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask,
                single_subset,
            ) = self._validated_inputs(
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask,
            )
            event_count = event_features.shape[1]
            condition = self.condition_encoder(
                torch.cat((query_features, context_features), dim=-1)
            )
            expanded_condition = condition.unsqueeze(1).expand(-1, event_count, -1)
            elements = self.element_encoder(
                torch.cat((event_features, expanded_condition), dim=-1)
            )
            elements = elements.masked_fill(~event_mask.unsqueeze(-1), 0.0)
            raw_utility = self._raw_subset_scores(
                condition,
                elements,
                subset_masks,
                event_mask,
            )
            empty_masks = torch.zeros(
                (subset_masks.shape[0], 1, event_count),
                dtype=torch.bool,
                device=subset_masks.device,
            )
            empty_baseline = self._raw_subset_scores(
                condition,
                elements,
                empty_masks,
                event_mask,
            )
            utility = (raw_utility - empty_baseline).masked_fill(
                ~subset_masks.any(dim=-1), 0.0
            )
            return self._restore_subset_shape(utility, single_subset)

        def forward(
            self,
            query_features: Any,
            context_features: Any,
            event_features: Any,
            subset_masks: Any,
            event_mask: Any | None = None,
        ) -> Any:
            return self.score_subsets(
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask,
            )


else:

    class PairwiseAdditiveUtilityPredictor:
        """Placeholder that keeps lightweight package imports dependency-free."""

        def __init__(
            self,
            dimensions: SetUtilityDimensions,
            *,
            pair_feature_dimension: int = 0,
        ) -> None:
            _validate_pair_feature_dimension(pair_feature_dimension)
            del dimensions
            _require_torch()


    class DeepSetsUtilityPredictor:
        """Placeholder that keeps lightweight package imports dependency-free."""

        def __init__(self, dimensions: SetUtilityDimensions) -> None:
            del dimensions
            _require_torch()


    class SetTransformerUtilityPredictor:
        """Placeholder that keeps lightweight package imports dependency-free."""

        def __init__(
            self,
            dimensions: SetUtilityDimensions,
            *,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.0,
        ) -> None:
            _validate_set_transformer_hyperparameters(
                dimensions,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
            _require_torch()


def build_pairwise_additive_utility_predictor(
    dimensions: SetUtilityDimensions,
    *,
    pair_feature_dimension: int = 0,
) -> Any:
    """Build the unary-plus-pairwise baseline."""
    _require_torch()
    return PairwiseAdditiveUtilityPredictor(
        dimensions,
        pair_feature_dimension=pair_feature_dimension,
    )


def build_deepsets_utility_predictor(dimensions: SetUtilityDimensions) -> Any:
    """Build the higher-order permutation-invariant baseline."""
    _require_torch()
    return DeepSetsUtilityPredictor(dimensions)


def build_set_transformer_utility_predictor(
    dimensions: SetUtilityDimensions,
    *,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.0,
) -> Any:
    """Build the higher-capacity permutation-invariant main candidate."""
    _require_torch()
    return SetTransformerUtilityPredictor(
        dimensions,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
