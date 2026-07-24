"""Lightweight HGKV-readout selector models and losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class HGKVReadoutEncoder(nn.Module):
    """Aggregate per-layer HGKV counterfactual readout features."""

    def __init__(
        self,
        *,
        layer_count: int,
        layer_feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.layer_count = layer_count
        self.layer_feature_dim = layer_feature_dim
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(layer_feature_dim)
        self.layer_projection = nn.Linear(layer_feature_dim, hidden_dim)
        self.layer_attention = nn.Linear(hidden_dim, 1)
        self.layer_bias = nn.Parameter(torch.zeros(layer_count))

    @property
    def feature_dim(self) -> int:
        return self.layer_count * self.layer_feature_dim

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feature dim {features.shape[-1]} != {self.feature_dim}"
            )
        shaped = features.reshape(
            *features.shape[:-1],
            self.layer_count,
            self.layer_feature_dim,
        )
        hidden = F.gelu(self.layer_projection(self.layer_norm(shaped)))
        attention_logits = self.layer_attention(hidden).squeeze(-1)
        attention_logits = attention_logits + self.layer_bias
        attention = torch.softmax(attention_logits, dim=-1)
        encoded = torch.sum(hidden * attention.unsqueeze(-1), dim=-2)
        return encoded, attention


class HGKVSingletonSelector(nn.Module):
    """Stage-1 singleton scorer with gain, rank, positive, and STOP outputs."""

    def __init__(
        self,
        *,
        layer_count: int,
        layer_feature_dim: int,
        hidden_dim: int,
        head_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = HGKVReadoutEncoder(
            layer_count=layer_count,
            layer_feature_dim=layer_feature_dim,
            hidden_dim=hidden_dim,
        )
        self.gain_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )
        self.positive_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.rank_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self, features: torch.Tensor, candidate_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        encoded, layer_attention = self.encoder(features)
        gains = self.gain_head(encoded).squeeze(-1)
        positive_logits = self.positive_head(encoded).squeeze(-1)
        rank_scores = self.rank_head(encoded).squeeze(-1)
        mask = candidate_mask.bool()
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (encoded * mask.unsqueeze(-1)).sum(dim=1) / count
        masked = encoded.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        maximum = masked.max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        stop_logits = self.stop_head(torch.cat([mean, maximum], dim=-1)).squeeze(-1)
        return {
            "encoded": encoded,
            "gain": gains,
            "layer_attention": layer_attention,
            "positive_logit": positive_logits,
            "rank_score": rank_scores,
            "stop_logit": stop_logits,
        }


class HGKVSetConditionedSelector(nn.Module):
    """Stage-2 marginal selector with two lightweight attention blocks."""

    def __init__(
        self,
        *,
        layer_count: int,
        layer_feature_dim: int,
        hidden_dim: int,
        head_hidden_dim: int,
        attention_heads: int,
        max_budget: int,
    ) -> None:
        super().__init__()
        self.encoder = HGKVReadoutEncoder(
            layer_count=layer_count,
            layer_feature_dim=layer_feature_dim,
            hidden_dim=hidden_dim,
        )
        self.selected_set_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            batch_first=True,
        )
        self.candidate_query_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            batch_first=True,
        )
        self.empty_selected = nn.Parameter(torch.zeros(hidden_dim))
        self.budget_embedding = nn.Embedding(max_budget + 1, hidden_dim)
        candidate_input_dim = 3 * hidden_dim
        self.marginal_head = nn.Sequential(
            nn.LayerNorm(candidate_input_dim),
            nn.Linear(candidate_input_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )
        self.rank_head = nn.Sequential(
            nn.LayerNorm(candidate_input_dim),
            nn.Linear(candidate_input_dim, 1),
        )
        self.positive_head = nn.Sequential(
            nn.LayerNorm(candidate_input_dim),
            nn.Linear(candidate_input_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        selected_features: torch.Tensor,
        selected_mask: torch.Tensor,
        remaining_budget: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        candidates, candidate_layer_attention = self.encoder(candidate_features)
        selected, selected_layer_attention = self.encoder(selected_features)
        effective_mask = selected_mask.bool().clone()
        effective_selected = selected.clone()
        empty_rows = ~effective_mask.any(dim=1)
        if empty_rows.any():
            effective_mask[empty_rows, 0] = True
            effective_selected[empty_rows, 0] = self.empty_selected
        selected_context, _ = self.selected_set_attention(
            effective_selected,
            effective_selected,
            effective_selected,
            key_padding_mask=~effective_mask,
            need_weights=False,
        )
        candidate_context, _ = self.candidate_query_attention(
            candidates,
            selected_context,
            selected_context,
            key_padding_mask=~effective_mask,
            need_weights=False,
        )
        budget = self.budget_embedding(remaining_budget)
        expanded_budget = budget.unsqueeze(1).expand_as(candidates)
        candidate_joint = torch.cat(
            [candidates, candidate_context, expanded_budget], dim=-1
        )
        marginal = self.marginal_head(candidate_joint).squeeze(-1)
        rank_scores = self.rank_head(candidate_joint).squeeze(-1)
        positive_logits = self.positive_head(candidate_joint).squeeze(-1)

        selected_count = effective_mask.sum(dim=1, keepdim=True).clamp_min(1)
        selected_summary = (
            selected_context * effective_mask.unsqueeze(-1)
        ).sum(dim=1) / selected_count
        mask = candidate_mask.bool()
        candidate_count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        candidate_mean = (candidates * mask.unsqueeze(-1)).sum(
            dim=1
        ) / candidate_count
        candidate_max = candidates.masked_fill(
            ~mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values
        candidate_max = torch.where(
            torch.isfinite(candidate_max),
            candidate_max,
            torch.zeros_like(candidate_max),
        )
        stop_logits = self.stop_head(
            torch.cat(
                [selected_summary, candidate_mean, candidate_max, budget],
                dim=-1,
            )
        ).squeeze(-1)
        return {
            "candidate_layer_attention": candidate_layer_attention,
            "encoded_candidates": candidates,
            "marginal": marginal,
            "positive_logit": positive_logits,
            "rank_score": rank_scores,
            "selected_layer_attention": selected_layer_attention,
            "stop_logit": stop_logits,
        }


def singleton_multitask_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    gain_weight: float,
    rank_weight: float,
    positive_weight: float,
    stop_weight: float,
    huber_beta: float,
    rank_min_delta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = candidate_mask.bool()
    gains = outputs["gain"]
    positive_logits = outputs["positive_logit"]
    gain_loss = F.smooth_l1_loss(
        gains[mask], targets[mask], beta=huber_beta
    )
    positive_loss = F.binary_cross_entropy_with_logits(
        positive_logits[mask], (targets[mask] > 0).to(gains.dtype)
    )

    target_delta = targets.unsqueeze(2) - targets.unsqueeze(1)
    rank_scores = outputs["rank_score"]
    rank_delta = rank_scores.unsqueeze(2) - rank_scores.unsqueeze(1)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    upper = torch.triu(
        torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
    )
    pair_mask = pair_mask & upper & (target_delta.abs() > rank_min_delta)
    rank_terms = []
    if pair_mask.any():
        signs = target_delta[pair_mask].sign()
        rank_terms.append(F.softplus(-signs * rank_delta[pair_mask]))
    stop_pair_mask = mask & (targets.abs() > rank_min_delta)
    if stop_pair_mask.any():
        stop_delta = (
            rank_scores - outputs["stop_logit"].unsqueeze(1)
        )[stop_pair_mask]
        stop_signs = targets[stop_pair_mask].sign()
        rank_terms.append(F.softplus(-stop_signs * stop_delta))
    rank_loss = (
        torch.cat(rank_terms).mean()
        if rank_terms
        else gains.sum() * 0
    )

    masked_targets = targets.masked_fill(~mask, float("-inf"))
    stop_targets = (masked_targets.max(dim=1).values <= 0).to(gains.dtype)
    stop_loss = F.binary_cross_entropy_with_logits(
        outputs["stop_logit"], stop_targets
    )
    total = (
        gain_weight * gain_loss
        + rank_weight * rank_loss
        + positive_weight * positive_loss
        + stop_weight * stop_loss
    )
    return total, {
        "gain": gain_loss.detach(),
        "positive": positive_loss.detach(),
        "rank": rank_loss.detach(),
        "stop": stop_loss.detach(),
        "total": total.detach(),
    }


def set_conditioned_multitask_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    marginal_weight: float,
    rank_weight: float,
    positive_weight: float,
    stop_weight: float,
    huber_beta: float,
    rank_min_delta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = candidate_mask.bool()
    marginal = outputs["marginal"]
    marginal_loss = F.smooth_l1_loss(
        marginal[mask], targets[mask], beta=huber_beta
    )
    positive_loss = F.binary_cross_entropy_with_logits(
        outputs["positive_logit"][mask],
        (targets[mask] > 0).to(marginal.dtype),
    )
    target_delta = targets.unsqueeze(2) - targets.unsqueeze(1)
    rank_delta = (
        outputs["rank_score"].unsqueeze(2)
        - outputs["rank_score"].unsqueeze(1)
    )
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    pair_mask = pair_mask & torch.triu(
        torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
    )
    pair_mask = pair_mask & (target_delta.abs() > rank_min_delta)
    rank_terms = []
    if pair_mask.any():
        signs = target_delta[pair_mask].sign()
        rank_terms.append(F.softplus(-signs * rank_delta[pair_mask]))
    stop_pair_mask = mask & (targets.abs() > rank_min_delta)
    if stop_pair_mask.any():
        stop_delta = (
            outputs["rank_score"] - outputs["stop_logit"].unsqueeze(1)
        )[stop_pair_mask]
        stop_signs = targets[stop_pair_mask].sign()
        rank_terms.append(F.softplus(-stop_signs * stop_delta))
    rank_loss = (
        torch.cat(rank_terms).mean()
        if rank_terms
        else marginal.sum() * 0
    )
    stop_targets = (
        targets.masked_fill(~mask, float("-inf")).max(dim=1).values <= 0
    ).to(marginal.dtype)
    stop_loss = F.binary_cross_entropy_with_logits(
        outputs["stop_logit"], stop_targets
    )
    total = (
        marginal_weight * marginal_loss
        + rank_weight * rank_loss
        + positive_weight * positive_loss
        + stop_weight * stop_loss
    )
    return total, {
        "marginal": marginal_loss.detach(),
        "positive": positive_loss.detach(),
        "rank": rank_loss.detach(),
        "stop": stop_loss.detach(),
        "total": total.detach(),
    }


def model_config_from_dict(config: dict[str, Any]) -> dict[str, int]:
    architecture = config["architecture"]
    return {
        "head_hidden_dim": int(architecture["head_hidden_dim"]),
        "hidden_dim": int(architecture["hidden_dim"]),
        "layer_count": int(architecture["layer_count"]),
        "layer_feature_dim": int(architecture["layer_feature_dim"]),
    }


def set_model_config_from_dict(config: dict[str, Any]) -> dict[str, int]:
    architecture = config["architecture"]
    return {
        "attention_heads": int(architecture["attention_heads"]),
        "head_hidden_dim": int(architecture["head_hidden_dim"]),
        "hidden_dim": int(architecture["hidden_dim"]),
        "layer_count": int(architecture["layer_count"]),
        "layer_feature_dim": int(architecture["layer_feature_dim"]),
        "max_budget": int(architecture["max_budget"]),
    }
