"""Full-history HGKV selector V2 primitives, model, losses, and beam search."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from causalcache.hgkv_selector import HGKVReadoutEncoder


HGKV_READOUT_DIM = 1280
TEMPORAL_FEATURE_DIM = 5
V2_FEATURE_DIM = HGKV_READOUT_DIM + TEMPORAL_FEATURE_DIM


def canonical_event_set(values: Iterable[int]) -> tuple[int, ...]:
    """Return a sorted, unique, non-negative event set."""
    result = tuple(sorted(int(value) for value in values))
    if any(value <= 0 for value in result):
        raise ValueError("event step ids must be positive")
    if len(result) != len(set(result)):
        raise ValueError("event set contains duplicate ids")
    return result


def restored_set_key(values: Iterable[int]) -> str:
    return "-".join(str(value) for value in canonical_event_set(values))


def target_action_sha256(target_text: str) -> str:
    if not isinstance(target_text, str) or not target_text:
        raise ValueError("target action text must be non-empty")
    return hashlib.sha256(target_text.encode("utf-8")).hexdigest()


def temporal_features(
    *,
    event_step_id: int,
    decision_step: int,
    history_length: int,
    candidate_event_step_ids: Sequence[int],
) -> tuple[float, float, float, float, float]:
    candidates = canonical_event_set(candidate_event_step_ids)
    event_step_id = int(event_step_id)
    decision_step = int(decision_step)
    history_length = int(history_length)
    if event_step_id not in candidates:
        raise ValueError("event is outside the full-history candidate inventory")
    if decision_step <= event_step_id:
        raise ValueError("decision step must follow candidate event")
    if history_length <= 0:
        raise ValueError("history length must be positive")
    relative_age = decision_step - event_step_id
    return (
        float(relative_age),
        float(relative_age / history_length),
        float(event_step_id / decision_step),
        float(event_step_id == max(candidates)),
        float(math.log1p(history_length)),
    )


@dataclass(frozen=True, order=True)
class CoalitionCacheKey:
    pair_group: str
    restored_set_key: str
    target_action_sha256: str
    prompt_revision: str
    hgkv_checkpoint_sha256: str
    b0_policy_sha256: str

    def __post_init__(self) -> None:
        if not self.pair_group:
            raise ValueError("pair_group must be non-empty")
        for label, value in (
            ("target_action_sha256", self.target_action_sha256),
            ("hgkv_checkpoint_sha256", self.hgkv_checkpoint_sha256),
            ("b0_policy_sha256", self.b0_policy_sha256),
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{label} must be lowercase SHA256")
        if not self.prompt_revision:
            raise ValueError("prompt revision must be non-empty")

    @classmethod
    def build(
        cls,
        *,
        pair_group: str,
        restored_event_step_ids: Iterable[int],
        target_text: str,
        prompt_revision: str,
        hgkv_checkpoint_sha256: str,
        b0_policy_sha256: str,
    ) -> "CoalitionCacheKey":
        return cls(
            pair_group=str(pair_group),
            restored_set_key=restored_set_key(restored_event_step_ids),
            target_action_sha256=target_action_sha256(target_text),
            prompt_revision=str(prompt_revision),
            hgkv_checkpoint_sha256=str(hgkv_checkpoint_sha256),
            b0_policy_sha256=str(b0_policy_sha256),
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "pair_group": self.pair_group,
            "restored_set_key": self.restored_set_key,
            "target_action_sha256": self.target_action_sha256,
            "prompt_revision": self.prompt_revision,
            "hgkv_checkpoint_sha256": self.hgkv_checkpoint_sha256,
            "b0_policy_sha256": self.b0_policy_sha256,
        }


def teacher_depth_expansions(
    candidate_event_step_ids: Sequence[int],
    prefixes: Sequence[Sequence[int]],
    *,
    depth: int,
) -> list[dict[str, Any]]:
    candidates = canonical_event_set(candidate_event_step_ids)
    if depth not in (0, 1, 2, 3):
        raise ValueError("teacher depth must be 0, 1, 2, or 3")
    canonical_prefixes = [canonical_event_set(prefix) for prefix in prefixes]
    if depth == 0 and canonical_prefixes != [()]:
        raise ValueError("depth 0 must start from exactly one empty prefix")
    if any(len(prefix) != depth for prefix in canonical_prefixes):
        raise ValueError("teacher prefix size does not match depth")
    if len(set(canonical_prefixes)) != len(canonical_prefixes):
        raise ValueError("teacher prefixes must be unique")
    rows: list[dict[str, Any]] = []
    for prefix in canonical_prefixes:
        if not set(prefix).issubset(candidates):
            raise ValueError("teacher prefix is outside candidate inventory")
        for candidate in candidates:
            if candidate in prefix:
                continue
            child = canonical_event_set((*prefix, candidate))
            rows.append(
                {
                    "depth": depth,
                    "edge": f"edge{depth}",
                    "selected_event_step_ids": list(prefix),
                    "candidate_event_step_id": candidate,
                    "restored_event_step_ids": list(child),
                    "restored_set_key": restored_set_key(child),
                }
            )
    return rows


def select_true_u_beam(
    expansions: Sequence[Mapping[str, Any]],
    utility_by_set_key: Mapping[str, float],
    *,
    beam_width: int,
) -> tuple[tuple[int, ...], ...]:
    if beam_width <= 0:
        raise ValueError("beam width must be positive")
    utility_by_child: dict[tuple[int, ...], float] = {}
    for row in expansions:
        child = canonical_event_set(row["restored_event_step_ids"])
        key = restored_set_key(child)
        if key not in utility_by_set_key:
            raise KeyError(f"missing true utility for coalition {key}")
        utility = float(utility_by_set_key[key])
        if not math.isfinite(utility):
            raise ValueError(f"non-finite true utility for coalition {key}")
        previous = utility_by_child.get(child)
        if previous is not None and previous != utility:
            raise ValueError(f"conflicting true utility for coalition {key}")
        utility_by_child[child] = utility
    ranked = sorted(
        utility_by_child,
        key=lambda child: (-utility_by_child[child], child),
    )
    return tuple(ranked[:beam_width])


def exact_best_sets(
    candidate_event_step_ids: Sequence[int],
    utility_by_set_key: Mapping[str, float],
    *,
    budgets: Sequence[int] = (1, 2, 4),
) -> dict[int, tuple[int, ...]]:
    from itertools import combinations

    candidates = canonical_event_set(candidate_event_step_ids)
    result: dict[int, tuple[int, ...]] = {}
    for budget in budgets:
        if budget <= 0 or budget > len(candidates):
            continue
        subsets = list(combinations(candidates, budget))
        missing = [
            subset
            for subset in subsets
            if restored_set_key(subset) not in utility_by_set_key
        ]
        if missing:
            raise KeyError(
                f"exact search missing {len(missing)} size-{budget} coalitions"
            )
        result[int(budget)] = min(
            subsets,
            key=lambda subset: (
                -float(utility_by_set_key[restored_set_key(subset)]),
                subset,
            ),
        )
    return result


class HGKVV2CandidateEncoder(nn.Module):
    """Fresh encoder for 1280-d HGKV readout plus five temporal features."""

    def __init__(
        self,
        *,
        layer_count: int,
        layer_feature_dim: int,
        hidden_dim: int,
        temporal_hidden_dim: int,
    ) -> None:
        super().__init__()
        if layer_count * layer_feature_dim != HGKV_READOUT_DIM:
            raise ValueError("V2 readout layout must equal 1280 dimensions")
        self.readout_encoder = HGKVReadoutEncoder(
            layer_count=layer_count,
            layer_feature_dim=layer_feature_dim,
            hidden_dim=hidden_dim,
        )
        self.temporal_encoder = nn.Sequential(
            nn.LayerNorm(TEMPORAL_FEATURE_DIM),
            nn.Linear(TEMPORAL_FEATURE_DIM, temporal_hidden_dim),
            nn.GELU(),
        )
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim + temporal_hidden_dim),
            nn.Linear(hidden_dim + temporal_hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != V2_FEATURE_DIM:
            raise ValueError(
                f"V2 feature dim {features.shape[-1]} != {V2_FEATURE_DIM}"
            )
        readout, layer_attention = self.readout_encoder(
            features[..., :HGKV_READOUT_DIM]
        )
        temporal = self.temporal_encoder(features[..., HGKV_READOUT_DIM:])
        return self.joint_projection(torch.cat([readout, temporal], dim=-1)), (
            layer_attention
        )


class HGKVSetSelectorV2(nn.Module):
    """Unified empty-start set-conditioned selector used at every depth."""

    def __init__(
        self,
        *,
        layer_count: int = 8,
        layer_feature_dim: int = 160,
        hidden_dim: int = 64,
        temporal_hidden_dim: int = 16,
        head_hidden_dim: int = 32,
        attention_heads: int = 4,
        max_budget: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = HGKVV2CandidateEncoder(
            layer_count=layer_count,
            layer_feature_dim=layer_feature_dim,
            hidden_dim=hidden_dim,
            temporal_hidden_dim=temporal_hidden_dim,
        )
        self.selected_set_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            batch_first=True,
        )
        self.candidate_to_selected_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            batch_first=True,
        )
        self.empty_selected = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.empty_selected, std=hidden_dim**-0.5)
        self.budget_embedding = nn.Embedding(max_budget + 1, hidden_dim)
        candidate_joint_dim = 3 * hidden_dim
        self.marginal_head = nn.Sequential(
            nn.LayerNorm(candidate_joint_dim),
            nn.Linear(candidate_joint_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )
        self.rank_head = nn.Sequential(
            nn.LayerNorm(candidate_joint_dim),
            nn.Linear(candidate_joint_dim, 1),
        )
        self.positive_head = nn.Sequential(
            nn.LayerNorm(candidate_joint_dim),
            nn.Linear(candidate_joint_dim, 1),
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
        batch_size = int(candidates.shape[0])
        if selected_features.shape[1] == 0:
            selected = candidates.new_zeros((batch_size, 1, candidates.shape[-1]))
            selected_layer_attention = candidates.new_zeros(
                (batch_size, 1, self.encoder.readout_encoder.layer_count)
            )
            effective_mask = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=candidates.device
            )
            selected[:, 0] = self.empty_selected
        else:
            selected, selected_layer_attention = self.encoder(selected_features)
            effective_mask = selected_mask.bool().clone()
            empty_rows = ~effective_mask.any(dim=1)
            if empty_rows.any():
                effective_mask[empty_rows, 0] = True
                selected = selected.clone()
                selected[empty_rows, 0] = self.empty_selected
        selected_context, _ = self.selected_set_attention(
            selected,
            selected,
            selected,
            key_padding_mask=~effective_mask,
            need_weights=False,
        )
        candidate_context, _ = self.candidate_to_selected_attention(
            candidates,
            selected_context,
            selected_context,
            key_padding_mask=~effective_mask,
            need_weights=False,
        )
        budget = self.budget_embedding(remaining_budget)
        candidate_joint = torch.cat(
            [
                candidates,
                candidate_context,
                budget.unsqueeze(1).expand_as(candidates),
            ],
            dim=-1,
        )
        marginal = self.marginal_head(candidate_joint).squeeze(-1)
        rank_score = self.rank_head(candidate_joint).squeeze(-1)
        positive_logit = self.positive_head(candidate_joint).squeeze(-1)

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
        stop_logit = self.stop_head(
            torch.cat(
                [selected_summary, candidate_mean, candidate_max, budget],
                dim=-1,
            )
        ).squeeze(-1)
        return {
            "candidate_layer_attention": candidate_layer_attention,
            "encoded_candidates": candidates,
            "marginal": marginal,
            "positive_logit": positive_logit,
            "rank_score": rank_score,
            "selected_layer_attention": selected_layer_attention,
            "stop_logit": stop_logit,
        }


def _mean_group_loss(
    terms: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask_float = mask.to(terms.dtype)
    counts = mask_float.sum(dim=1)
    group = (terms * mask_float).sum(dim=1) / counts.clamp_min(1)
    valid = counts > 0
    if valid.any():
        return group[valid].mean()
    return terms.sum() * 0.0


def hgkv_selector_v2_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    marginal_weight: float = 1.0,
    rank_weight: float = 0.5,
    positive_weight: float = 0.25,
    stop_weight: float = 0.25,
    huber_beta: float = 0.02,
    rank_min_delta: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = candidate_mask.bool()
    marginal_terms = F.smooth_l1_loss(
        outputs["marginal"],
        targets,
        beta=huber_beta,
        reduction="none",
    )
    marginal_loss = _mean_group_loss(marginal_terms, mask)
    positive_terms = F.binary_cross_entropy_with_logits(
        outputs["positive_logit"],
        (targets > 0).to(targets.dtype),
        reduction="none",
    )
    positive_loss = _mean_group_loss(positive_terms, mask)

    target_delta = targets.unsqueeze(2) - targets.unsqueeze(1)
    score_delta = outputs["rank_score"].unsqueeze(2) - outputs[
        "rank_score"
    ].unsqueeze(1)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    pair_mask &= torch.triu(
        torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
    )
    pair_mask &= target_delta.abs() > rank_min_delta
    pair_terms = F.softplus(-target_delta.sign() * score_delta)
    pair_terms = pair_terms.reshape(pair_terms.shape[0], -1)
    pair_mask_flat = pair_mask.reshape(pair_mask.shape[0], -1)

    stop_pair_mask = mask & (targets.abs() > rank_min_delta)
    stop_delta = outputs["rank_score"] - outputs["stop_logit"].unsqueeze(1)
    stop_pair_terms = F.softplus(-targets.sign() * stop_delta)
    combined_terms = torch.cat([pair_terms, stop_pair_terms], dim=1)
    combined_mask = torch.cat([pair_mask_flat, stop_pair_mask], dim=1)
    rank_loss = _mean_group_loss(combined_terms, combined_mask)

    masked_targets = targets.masked_fill(~mask, float("-inf"))
    max_target = masked_targets.max(dim=1).values
    if (~mask.any(dim=1)).any():
        raise ValueError("every prefix group must contain a candidate")
    stop_target = (max_target <= 0).to(targets.dtype)
    stop_loss = F.binary_cross_entropy_with_logits(
        outputs["stop_logit"], stop_target
    )
    total = (
        marginal_weight * marginal_loss
        + rank_weight * rank_loss
        + positive_weight * positive_loss
        + stop_weight * stop_loss
    )
    return total, {
        "marginal": marginal_loss,
        "positive": positive_loss,
        "rank": rank_loss,
        "stop": stop_loss,
        "total": total,
    }


@dataclass(frozen=True)
class BeamPath:
    selected_event_step_ids: tuple[int, ...]
    cumulative_predicted_u: float
    terminal: bool
    trace: tuple[dict[str, Any], ...]


MarginalScorer = Callable[
    [tuple[int, ...], tuple[int, ...], int],
    Mapping[int, float],
]


def learned_beam_search(
    candidate_event_step_ids: Sequence[int],
    *,
    budget: int,
    scorer: MarginalScorer,
    beam_width: int = 4,
) -> BeamPath:
    candidates = canonical_event_set(candidate_event_step_ids)
    if budget <= 0 or budget > 4:
        raise ValueError("budget must be in 1..4")
    if beam_width <= 0:
        raise ValueError("beam width must be positive")
    beam = [BeamPath((), 0.0, False, ())]
    for _depth in range(budget):
        children: list[BeamPath] = []
        for path in beam:
            if path.terminal:
                children.append(path)
                continue
            remaining = tuple(
                candidate
                for candidate in candidates
                if candidate not in path.selected_event_step_ids
            )
            children.append(
                BeamPath(
                    path.selected_event_step_ids,
                    path.cumulative_predicted_u,
                    True,
                    path.trace
                    + (
                        {
                            "prefix": list(path.selected_event_step_ids),
                            "candidate": None,
                            "predicted_marginal": 0.0,
                            "cumulative_predicted_u": path.cumulative_predicted_u,
                            "stop": True,
                        },
                    ),
                )
            )
            if not remaining:
                continue
            marginals = scorer(
                path.selected_event_step_ids,
                remaining,
                budget - len(path.selected_event_step_ids),
            )
            if set(marginals) != set(remaining):
                raise ValueError("scorer must return every remaining candidate exactly")
            for candidate in remaining:
                marginal = float(marginals[candidate])
                if not math.isfinite(marginal):
                    raise ValueError("predicted marginal must be finite")
                selected = canonical_event_set(
                    (*path.selected_event_step_ids, candidate)
                )
                cumulative = path.cumulative_predicted_u + marginal
                children.append(
                    BeamPath(
                        selected,
                        cumulative,
                        len(selected) >= budget,
                        path.trace
                        + (
                            {
                                "prefix": list(path.selected_event_step_ids),
                                "candidate": candidate,
                                "predicted_marginal": marginal,
                                "cumulative_predicted_u": cumulative,
                                "stop": False,
                            },
                        ),
                    )
                )
        deduplicated: dict[tuple[tuple[int, ...], bool], BeamPath] = {}
        for child in children:
            key = (child.selected_event_step_ids, child.terminal)
            previous = deduplicated.get(key)
            if previous is None or (
                child.cumulative_predicted_u,
                tuple(json.dumps(step, sort_keys=True) for step in child.trace),
            ) > (
                previous.cumulative_predicted_u,
                tuple(json.dumps(step, sort_keys=True) for step in previous.trace),
            ):
                deduplicated[key] = child
        beam = sorted(
            deduplicated.values(),
            key=lambda path: (
                -path.cumulative_predicted_u,
                path.selected_event_step_ids,
                not path.terminal,
            ),
        )[:beam_width]
        if all(path.terminal for path in beam):
            break
    return min(
        beam,
        key=lambda path: (
            -path.cumulative_predicted_u,
            path.selected_event_step_ids,
            not path.terminal,
        ),
    )


__all__ = [
    "BeamPath",
    "CoalitionCacheKey",
    "HGKVSetSelectorV2",
    "HGKVV2CandidateEncoder",
    "HGKV_READOUT_DIM",
    "TEMPORAL_FEATURE_DIM",
    "V2_FEATURE_DIM",
    "canonical_event_set",
    "exact_best_sets",
    "hgkv_selector_v2_loss",
    "learned_beam_search",
    "restored_set_key",
    "select_true_u_beam",
    "target_action_sha256",
    "teacher_depth_expansions",
    "temporal_features",
]
