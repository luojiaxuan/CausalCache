"""Loss and optimization primitives for budget-agnostic set-utility models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from causalcache.set_utility_data import TensorSetUtilityBatch


@dataclass(frozen=True)
class UtilityLossWeights:
    """Explicit, pre-freeze coefficients for the three required objectives."""

    raw_regression: float
    normalized_regression: float
    within_state_ranking: float
    smooth_l1_beta: float

    def __post_init__(self) -> None:
        coefficients = (
            self.raw_regression,
            self.normalized_regression,
            self.within_state_ranking,
        )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in coefficients
        ):
            raise ValueError("utility loss coefficients must be finite and non-negative")
        if not any(float(value) > 0.0 for value in coefficients):
            raise ValueError("at least one utility objective must have positive weight")
        if (
            not math.isfinite(float(self.smooth_l1_beta))
            or float(self.smooth_l1_beta) <= 0.0
        ):
            raise ValueError("smooth-L1 beta must be finite and positive")


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("set-utility training requires PyTorch") from error
    return torch


def _score_batch(model: Any, batch: TensorSetUtilityBatch) -> Any:
    keyword: dict[str, Any] = {}
    if batch.pair_features is not None:
        try:
            import inspect

            parameters = inspect.signature(model.score_subsets).parameters
        except (AttributeError, TypeError, ValueError):
            parameters = {}
        if "pair_features" in parameters:
            keyword["pair_features"] = batch.pair_features
    predictions = model(
        batch.query_features,
        batch.context_features,
        batch.event_features,
        batch.subset_masks,
        batch.event_mask,
        **keyword,
    )
    if predictions.shape != batch.raw_targets.shape:
        raise ValueError("set-utility prediction shape differs from the padded targets")
    return predictions


def utility_predictor_loss(
    model: Any,
    batch: TensorSetUtilityBatch,
    *,
    weights: UtilityLossWeights,
) -> tuple[Any, dict[str, float]]:
    """Combine balanced raw/normalized regression and within-state ranking."""
    if not isinstance(batch, TensorSetUtilityBatch):
        raise TypeError("batch must be TensorSetUtilityBatch")
    if not isinstance(weights, UtilityLossWeights):
        raise TypeError("weights must be UtilityLossWeights")
    torch = _torch()
    if weights.normalized_regression > 0.0 and not bool(
        batch.normalized_state_mask.any()
    ):
        raise ValueError("normalized regression is enabled but the batch has no scale")
    if weights.within_state_ranking > 0.0 and not batch.ranking_weights.numel():
        raise ValueError("ranking is enabled but the batch has no untied target pair")
    predictions = _score_batch(model, batch)
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError("set-utility predictions are non-finite")

    raw_rows = torch.nn.functional.smooth_l1_loss(
        predictions,
        batch.raw_targets,
        reduction="none",
        beta=weights.smooth_l1_beta,
    )
    raw = torch.sum(raw_rows * batch.raw_regression_weights)

    safe_scales = torch.where(
        batch.normalized_state_mask,
        batch.normalization_scales,
        torch.ones_like(batch.normalization_scales),
    )
    normalized_predictions = predictions / safe_scales.unsqueeze(1)
    normalized_rows = torch.nn.functional.smooth_l1_loss(
        normalized_predictions,
        batch.normalized_targets,
        reduction="none",
        beta=weights.smooth_l1_beta,
    )
    normalized = torch.sum(
        normalized_rows * batch.normalized_regression_weights
    )

    if batch.ranking_weights.numel():
        left = predictions[
            batch.ranking_batch_indices,
            batch.ranking_left_rows,
        ]
        right = predictions[
            batch.ranking_batch_indices,
            batch.ranking_right_rows,
        ]
        ranking_rows = torch.nn.functional.softplus(
            -(left - right) * batch.ranking_signs
        )
        ranking = torch.sum(ranking_rows * batch.ranking_weights)
    else:
        ranking = predictions.sum() * 0.0

    total = (
        weights.raw_regression * raw
        + weights.normalized_regression * normalized
        + weights.within_state_ranking * ranking
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("set-utility loss is non-finite")
    return total, {
        "raw_regression": float(raw.detach()),
        "normalized_regression": float(normalized.detach()),
        "within_state_ranking": float(ranking.detach()),
        "total": float(total.detach()),
    }


def utility_optimization_step(
    model: Any,
    optimizer: Any,
    batch: TensorSetUtilityBatch,
    *,
    weights: UtilityLossWeights,
    maximum_gradient_norm: float,
) -> dict[str, float]:
    """Run one explicit optimizer step without binding an optimizer or schedule."""
    if (
        not math.isfinite(float(maximum_gradient_norm))
        or float(maximum_gradient_norm) <= 0.0
    ):
        raise ValueError("maximum gradient norm must be finite and positive")
    torch = _torch()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = utility_predictor_loss(model, batch, weights=weights)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(maximum_gradient_norm)
    )
    if not bool(torch.isfinite(norm)):
        raise ValueError("set-utility gradient norm is non-finite")
    optimizer.step()
    if any(
        not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    ):
        raise ValueError("set-utility parameter became non-finite")
    return {**metrics, "preclip_gradient_norm": float(norm)}


__all__ = [
    "UtilityLossWeights",
    "utility_optimization_step",
    "utility_predictor_loss",
]
