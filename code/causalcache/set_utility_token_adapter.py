"""Post-final-hidden token adapter control for conditional memory selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight local installs.
    torch = None

from causalcache.set_utility_selector_branch import ZeroInitResidualTokenAdapter
from causalcache.set_utility_token_models import (
    EncodedConditionalMarginalState,
    TokenConditionalMarginalPredictor,
    TokenUtilityModelConfig,
)


@dataclass(frozen=True)
class TokenAdapterConfig:
    """Configuration for the selector-only post-final-hidden control."""

    source_hidden_size: int = 4096
    bottleneck_size: int = 64

    def __post_init__(self) -> None:
        if type(self.source_hidden_size) is not int or self.source_hidden_size <= 0:
            raise ValueError("adapter source hidden size must be positive")
        if type(self.bottleneck_size) is not int or not (
            1 <= self.bottleneck_size <= self.source_hidden_size
        ):
            raise ValueError("adapter bottleneck size is outside the valid range")


if torch is not None:

    class TokenAdapterConditionalMarginalPredictor(torch.nn.Module):
        """Wrap a direct marginal predictor with a shared residual token adapter.

        The adapter transforms the frozen GUI-Owl final hidden tokens before the
        existing resampler. Its zero initialization makes this model exactly
        reproduce the wrapped predictor at construction time.
        """

        _TOKEN_ARGUMENTS = (
            "query_visual_tokens",
            "query_text_tokens",
            "event_visual_tokens",
            "event_text_tokens",
        )

        def __init__(
            self,
            predictor_config: TokenUtilityModelConfig,
            adapter_config: TokenAdapterConfig,
        ) -> None:
            super().__init__()
            if not isinstance(predictor_config, TokenUtilityModelConfig):
                raise TypeError("predictor_config must be TokenUtilityModelConfig")
            if not isinstance(adapter_config, TokenAdapterConfig):
                raise TypeError("adapter_config must be TokenAdapterConfig")
            if predictor_config.source_hidden_size != adapter_config.source_hidden_size:
                raise ValueError("predictor and adapter source hidden sizes differ")
            self.predictor_config = predictor_config
            self.adapter_config = adapter_config
            self.predictor = TokenConditionalMarginalPredictor(predictor_config)
            self.adapter = ZeroInitResidualTokenAdapter(
                adapter_config.source_hidden_size,
                adapter_config.bottleneck_size,
            )
            # note (luojiaxuan): The direct predictor intentionally freezes its
            # legacy scalar-utility modules. Preserve that mask when a staged
            # adapter-only phase later unfreezes the downstream predictor.
            self._default_head_trainable = frozenset(
                name
                for name, parameter in self.predictor.named_parameters()
                if parameter.requires_grad
            )

        @property
        def config(self) -> TokenUtilityModelConfig:
            """Expose the wrapped predictor config for existing trainer code."""
            return self.predictor_config

        @property
        def encoder(self) -> Any:
            """Expose the wrapped encoder for compatibility with diagnostics."""
            return self.predictor.encoder

        def load_predictor_state_dict(self, state_dict: Mapping[str, Any]) -> None:
            """Strictly load a legacy predictor checkpoint, excluding the adapter."""
            if not isinstance(state_dict, Mapping):
                raise TypeError("predictor state_dict must be a mapping")
            self.predictor.load_state_dict(state_dict, strict=True)

        def set_adapter_trainable(self, enabled: bool) -> None:
            if type(enabled) is not bool:
                raise TypeError("adapter trainable flag must be boolean")
            self.adapter.requires_grad_(enabled)

        def set_head_trainable(self, enabled: bool) -> None:
            """Toggle the downstream predictor while preserving its frozen mask."""
            if type(enabled) is not bool:
                raise TypeError("head trainable flag must be boolean")
            for name, parameter in self.predictor.named_parameters():
                parameter.requires_grad_(enabled and name in self._default_head_trainable)

        def freeze_adapter(self) -> None:
            self.set_adapter_trainable(False)

        def unfreeze_adapter(self) -> None:
            self.set_adapter_trainable(True)

        def freeze_head(self) -> None:
            self.set_head_trainable(False)

        def unfreeze_head(self) -> None:
            self.set_head_trainable(True)

        def optimizer_parameter_groups(
            self,
            *,
            adapter_lr: float,
            head_lr: float,
        ) -> list[dict[str, Any]]:
            """Build disjoint optimizer groups for the adapter and predictor."""
            for name, value in (("adapter_lr", adapter_lr), ("head_lr", head_lr)):
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"{name} must be numeric")
                if float(value) <= 0.0:
                    raise ValueError(f"{name} must be positive")
            groups = []
            adapter_parameters = [
                parameter
                for parameter in self.adapter.parameters()
                if parameter.requires_grad
            ]
            head_parameters = [
                parameter
                for parameter in self.predictor.parameters()
                if parameter.requires_grad
            ]
            if adapter_parameters:
                groups.append(
                    {
                        "name": "adapter",
                        "params": adapter_parameters,
                        "lr": float(adapter_lr),
                    }
                )
            if head_parameters:
                groups.append(
                    {
                        "name": "head",
                        "params": head_parameters,
                        "lr": float(head_lr),
                    }
                )
            return groups

        def adapt_source_tokens(self, values: Any) -> Any:
            if getattr(values, "ndim", 0) < 2:
                raise ValueError("source token tensor must have at least two dimensions")
            if values.shape[-1] != self.adapter_config.source_hidden_size:
                raise ValueError("source token hidden size drifted")
            return self.adapter(values)

        def _adapt_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
            adapted = dict(inputs)
            missing = [name for name in self._TOKEN_ARGUMENTS if name not in adapted]
            if missing:
                raise ValueError(f"token adapter inputs are missing: {missing}")
            for name in self._TOKEN_ARGUMENTS:
                adapted[name] = self.adapt_source_tokens(adapted[name])
            return adapted

        def encode_state_once(self, **inputs: Any) -> EncodedConditionalMarginalState:
            return self.predictor.encode_state_once(**self._adapt_inputs(inputs))

        def score_encoded_candidates(
            self,
            encoded_state: EncodedConditionalMarginalState,
            selected_masks: Any,
        ) -> Any:
            return self.predictor.score_encoded_candidates(
                encoded_state, selected_masks
            )

        def score_singleton_utilities(
            self,
            encoded_state: EncodedConditionalMarginalState,
            subset_masks: Any,
        ) -> Any:
            return self.predictor.score_singleton_utilities(
                encoded_state, subset_masks
            )

        def forward(
            self,
            *,
            subset_masks: Any | None = None,
            selected_masks: Any | None = None,
            **inputs: Any,
        ) -> Any:
            if (subset_masks is None) == (selected_masks is None):
                raise ValueError(
                    "exactly one of subset_masks or selected_masks is required"
                )
            encoded = self.encode_state_once(**inputs)
            if selected_masks is not None:
                return self.score_encoded_candidates(encoded, selected_masks)
            return self.score_singleton_utilities(encoded, subset_masks)


else:  # pragma: no cover - exercised only without PyTorch.

    class TokenAdapterConditionalMarginalPredictor:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("token adapter model requires PyTorch")


__all__ = [
    "TokenAdapterConditionalMarginalPredictor",
    "TokenAdapterConfig",
]
