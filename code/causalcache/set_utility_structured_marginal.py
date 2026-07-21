"""Small structured predictor for conditional restoration marginals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local lightweight installs.
    torch = None

from causalcache.set_utility_token_models import (
    EncodedConditionalMarginalState,
    TokenSetUtilityPredictor,
    TokenUtilityModelConfig,
)


@dataclass(frozen=True)
class StructuredMarginalHeadConfig:
    """Capacity and calibration controls for the structured DeepSets head."""

    input_hidden_size: int = 256
    hidden_size: int = 128
    pair_rank: int = 32
    dropout: float = 0.1
    stop_temperature: float = 0.1

    def __post_init__(self) -> None:
        for name in ("input_hidden_size", "hidden_size", "pair_rank"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.dropout, (int, float)) or isinstance(
            self.dropout, bool
        ):
            raise TypeError("dropout must be numeric")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.stop_temperature, (int, float)) or isinstance(
            self.stop_temperature, bool
        ):
            raise TypeError("stop_temperature must be numeric")
        if not math.isfinite(float(self.stop_temperature)) or float(
            self.stop_temperature
        ) <= 0.0:
            raise ValueError("stop_temperature must be finite and positive")


@dataclass(frozen=True)
class StructuredMarginalLossConfig:
    """Loss weights for gain, interaction, ranking, and STOP supervision."""

    marginal_regression: float = 1.0
    decision_ranking: float = 1.0
    stop_calibration: float = 1.0
    interaction_regression: float = 0.5
    interaction_sign: float = 0.25
    smooth_l1_beta: float = 0.1
    interaction_epsilon: float = 1e-4
    decision_temperature: float = 0.1

    def __post_init__(self) -> None:
        nonnegative = (
            "marginal_regression",
            "decision_ranking",
            "stop_calibration",
            "interaction_regression",
            "interaction_sign",
            "interaction_epsilon",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("smooth_l1_beta", "decision_temperature"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


if torch is not None:

    class StructuredConditionalMarginalHead(torch.nn.Module):
        """Predict signed ``Delta(j | S)`` with explicit pair interactions."""

        def __init__(self, config: StructuredMarginalHeadConfig) -> None:
            super().__init__()
            if not isinstance(config, StructuredMarginalHeadConfig):
                raise TypeError("config must be StructuredMarginalHeadConfig")
            self.config = config
            hidden = config.hidden_size
            self.entity_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(config.input_hidden_size),
                torch.nn.Linear(config.input_hidden_size, hidden),
                torch.nn.GELU(approximate="none"),
            )
            self.selected_phi = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
            )
            self.selected_rho = torch.nn.Sequential(
                torch.nn.Linear(hidden + 1, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Linear(hidden, hidden),
            )
            self.base_head = torch.nn.Sequential(
                torch.nn.Linear(5 * hidden, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(hidden, 1),
            )
            self.pair_left = torch.nn.Linear(hidden, config.pair_rank, bias=False)
            self.pair_right = torch.nn.Linear(hidden, config.pair_rank, bias=False)
            self.context_head = torch.nn.Sequential(
                torch.nn.Linear(6 * hidden + 1, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(hidden, 1),
            )
            self.stop_head = torch.nn.Sequential(
                torch.nn.Linear(3 * hidden + 1, hidden),
                torch.nn.GELU(approximate="none"),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(hidden, 1),
            )
            torch.nn.init.zeros_(self.stop_head[-1].weight)
            torch.nn.init.zeros_(self.stop_head[-1].bias)

        def decompose(
            self,
            *,
            query: Any,
            events: Any,
            event_mask: Any,
            selected_masks: Any,
        ) -> dict[str, Any]:
            """Return auditable base, pair, context, marginal, and STOP terms."""
            if query.ndim != 2 or events.ndim != 3:
                raise ValueError("structured query/events must have rank two/three")
            batch_size, event_count, input_hidden = events.shape
            if query.shape != (batch_size, input_hidden):
                raise ValueError("structured query/event geometry drifted")
            if input_hidden != self.config.input_hidden_size:
                raise ValueError("structured input hidden size drifted")
            if event_mask.shape != (batch_size, event_count):
                raise ValueError("structured event mask geometry drifted")
            if event_mask.dtype != torch.bool:
                raise TypeError("structured event mask must be boolean")
            squeeze = selected_masks.ndim == 2
            if squeeze:
                selected_masks = selected_masks.unsqueeze(1)
            if (
                selected_masks.ndim != 3
                or selected_masks.dtype != torch.bool
                or selected_masks.shape[0] != batch_size
                or selected_masks.shape[2] != event_count
            ):
                raise ValueError("structured selected mask geometry drifted")
            if bool((selected_masks & ~event_mask.unsqueeze(1)).any()):
                raise ValueError("a structured coalition selects a padded event")

            group_count = selected_masks.shape[1]
            query_hidden = self.entity_projection(query)
            event_hidden = self.entity_projection(events).masked_fill(
                ~event_mask.unsqueeze(-1), 0.0
            )
            universe_weights = event_mask.to(dtype=event_hidden.dtype)
            universe = torch.einsum("bn,bnh->bh", universe_weights, event_hidden)
            universe = universe / universe_weights.sum(dim=1, keepdim=True).clamp_min(
                1.0
            )
            selected_weights = selected_masks.to(dtype=event_hidden.dtype)
            selected_count = selected_weights.sum(dim=2, keepdim=True)
            selected_mean = torch.einsum(
                "bgn,bnh->bgh", selected_weights, self.selected_phi(event_hidden)
            ) / selected_count.clamp_min(1.0)
            log_cardinality = torch.log1p(selected_count)
            selected_summary = self.selected_rho(
                torch.cat((selected_mean, log_cardinality), dim=-1)
            )
            has_selected = (selected_count > 0).to(dtype=event_hidden.dtype)
            selected_summary = selected_summary * has_selected

            expanded_events = event_hidden[:, None, :, :].expand(
                -1, group_count, -1, -1
            )
            expanded_query = query_hidden[:, None, None, :].expand_as(expanded_events)
            expanded_universe = universe[:, None, None, :].expand_as(expanded_events)
            expanded_selected = selected_summary[:, :, None, :].expand_as(
                expanded_events
            )
            base = self.base_head(
                torch.cat(
                    (
                        expanded_events,
                        expanded_query,
                        expanded_universe,
                        expanded_events * expanded_query,
                        torch.abs(expanded_events - expanded_query),
                    ),
                    dim=-1,
                )
            ).squeeze(-1)

            left = self.pair_left(event_hidden)
            right = self.pair_right(event_hidden)
            pair_matrix = torch.einsum("bnr,bmr->bnm", left, right)
            pair_matrix = 0.5 * (pair_matrix + pair_matrix.transpose(1, 2))
            pair_matrix = pair_matrix / math.sqrt(self.config.pair_rank)
            pair = torch.einsum("bnm,bgm->bgn", pair_matrix, selected_weights)
            pair = pair / selected_count.sqrt().clamp_min(1.0)
            pair = pair * has_selected

            context = self.context_head(
                torch.cat(
                    (
                        expanded_events,
                        expanded_query,
                        expanded_universe,
                        expanded_selected,
                        expanded_events * expanded_selected,
                        torch.abs(expanded_events - expanded_selected),
                        log_cardinality[:, :, None, :].expand(
                            -1, -1, event_count, -1
                        ),
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
            context = context * has_selected
            marginal = (base + pair + context).masked_fill(
                ~event_mask[:, None, :], 0.0
            )
            stop = self.stop_head(
                torch.cat(
                    (query_hidden[:, None, :].expand(-1, group_count, -1),
                     universe[:, None, :].expand(-1, group_count, -1),
                     selected_summary,
                     log_cardinality),
                    dim=-1,
                )
            ).squeeze(-1)
            result = {
                "base": base,
                "candidate_mask": event_mask[:, None, :] & ~selected_masks,
                "context": context,
                "marginal": marginal,
                "pair": pair,
                "stop": stop,
            }
            if squeeze:
                return {key: value.squeeze(1) for key, value in result.items()}
            return result

        def forward(
            self,
            *,
            query: Any,
            events: Any,
            event_mask: Any,
            selected_masks: Any,
        ) -> Any:
            terms = self.decompose(
                query=query,
                events=events,
                event_mask=event_mask,
                selected_masks=selected_masks,
            )
            stop = terms["stop"]
            marginal = terms["marginal"]
            if marginal.ndim == 2:
                return torch.cat((stop.unsqueeze(1), marginal), dim=1)
            return torch.cat((stop.unsqueeze(2), marginal), dim=2)


    class StructuredTokenConditionalMarginalPredictor(torch.nn.Module):
        """Reuse the rich token encoder with a compact structured DeepSets head."""

        def __init__(
            self,
            encoder_config: TokenUtilityModelConfig,
            head_config: StructuredMarginalHeadConfig,
        ) -> None:
            super().__init__()
            if encoder_config.family != "deepsets":
                raise ValueError("structured marginal encoder must use deepsets")
            if encoder_config.preserve_entity_latents:
                raise ValueError("structured marginal encoder must pool entity latents")
            if encoder_config.hidden_size != head_config.input_hidden_size:
                raise ValueError("encoder/head hidden sizes differ")
            self.encoder_config = encoder_config
            self.head_config = head_config
            self.source_encoder = TokenSetUtilityPredictor(encoder_config)
            self.head = StructuredConditionalMarginalHead(head_config)
            self.source_encoder.selection_embedding.requires_grad_(False)
            self.source_encoder.utility_head.requires_grad_(False)

        def load_shared_encoder_from_direct_state(
            self, state_dict: Mapping[str, Any]
        ) -> None:
            """Load only source/numeric/event encoders from a v3 direct checkpoint."""
            prefixes = ("entity_encoder.", "numeric_encoder.", "event_conditioner.")
            shared = {
                key.removeprefix("encoder."): value
                for key, value in state_dict.items()
                if key.startswith("encoder.")
                and key.removeprefix("encoder.").startswith(prefixes)
            }
            expected = {
                key: value
                for key, value in self.source_encoder.state_dict().items()
                if key.startswith(prefixes)
            }
            if set(shared) != set(expected):
                missing = sorted(set(expected) - set(shared))
                extra = sorted(set(shared) - set(expected))
                raise ValueError(
                    f"shared encoder checkpoint mismatch: missing={missing}, extra={extra}"
                )
            for key, value in shared.items():
                if value.shape != expected[key].shape:
                    raise ValueError(f"shared encoder tensor shape drifted: {key}")
            current = self.source_encoder.state_dict()
            current.update(shared)
            self.source_encoder.load_state_dict(current, strict=True)

        def encode_state_once(self, **inputs: Any) -> EncodedConditionalMarginalState:
            encoded = self.source_encoder.encode_state_once(**inputs)
            return EncodedConditionalMarginalState(
                query=encoded.query,
                events=encoded.events,
                event_mask=encoded.event_mask,
            )

        def score_encoded_candidates(
            self,
            encoded_state: EncodedConditionalMarginalState,
            selected_masks: Any,
        ) -> Any:
            if not isinstance(encoded_state, EncodedConditionalMarginalState):
                raise TypeError("encoded_state must be EncodedConditionalMarginalState")
            return self.head(
                query=encoded_state.query,
                events=encoded_state.events,
                event_mask=encoded_state.event_mask,
                selected_masks=selected_masks,
            )

        def forward(self, *, selected_masks: Any, **inputs: Any) -> Any:
            return self.score_encoded_candidates(
                self.encode_state_once(**inputs), selected_masks
            )


else:  # pragma: no cover - exercised only without PyTorch.

    class StructuredConditionalMarginalHead:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("structured marginal model requires PyTorch")

    class StructuredTokenConditionalMarginalPredictor:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("structured marginal model requires PyTorch")


def structured_conditional_marginal_loss(
    action_scores: Any,
    *,
    normalized_targets: Any,
    action_mask: Any,
    group_mask: Any,
    selected_masks: Any,
    config: StructuredMarginalLossConfig,
) -> tuple[Any, dict[str, float]]:
    """Train gains, pair interactions, rankings, and calibrated STOP jointly."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("structured marginal loss requires PyTorch")
    if not isinstance(config, StructuredMarginalLossConfig):
        raise TypeError("config must be StructuredMarginalLossConfig")
    if action_scores.shape != normalized_targets.shape or action_scores.shape != (
        *selected_masks.shape[:2],
        selected_masks.shape[2] + 1,
    ):
        raise ValueError("structured loss prediction/target geometry drifted")
    if action_mask.shape != action_scores.shape or group_mask.shape != selected_masks.shape[:2]:
        raise ValueError("structured loss masks have invalid geometry")
    if action_mask.dtype != torch.bool or group_mask.dtype != torch.bool:
        raise TypeError("structured action/group masks must be boolean")
    if selected_masks.dtype != torch.bool:
        raise TypeError("structured selected masks must be boolean")
    valid = action_mask & group_mask.unsqueeze(-1)
    candidates = valid.clone()
    candidates[:, :, 0] = False
    if not bool(candidates.any()):
        raise ValueError("structured loss has no candidate marginal targets")

    marginal_terms = torch.nn.functional.smooth_l1_loss(
        action_scores,
        normalized_targets,
        reduction="none",
        beta=config.smooth_l1_beta,
    )
    marginal = (marginal_terms * candidates).sum() / candidates.sum()
    logits = (action_scores / config.decision_temperature).masked_fill(~valid, -1e9)
    teacher_action = normalized_targets.masked_fill(~valid, -1e9).argmax(dim=2)
    ranking_rows = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        teacher_action.reshape(-1),
        reduction="none",
    ).reshape_as(group_mask)
    ranking = (ranking_rows * group_mask).sum() / group_mask.sum().clamp_min(1)

    candidate_predictions = action_scores[:, :, 1:].masked_fill(
        ~candidates[:, :, 1:], -1e9
    )
    candidate_targets = normalized_targets[:, :, 1:].masked_fill(
        ~candidates[:, :, 1:], -1e9
    )
    best_prediction = candidate_predictions.max(dim=2).values
    best_target = candidate_targets.max(dim=2).values
    teacher_stop = (best_target <= 0.0).to(dtype=action_scores.dtype)
    stop_logit = (action_scores[:, :, 0] - best_prediction) / config.decision_temperature
    stop_rows = torch.nn.functional.binary_cross_entropy_with_logits(
        stop_logit, teacher_stop, reduction="none"
    )
    stop_calibration = (stop_rows * group_mask).sum() / group_mask.sum().clamp_min(1)
    stop_probability = torch.sigmoid(stop_logit)
    stop_brier = (
        ((stop_probability - teacher_stop).square() * group_mask).sum()
        / group_mask.sum().clamp_min(1)
    )

    interaction_predictions = []
    interaction_targets = []
    for batch_index in range(action_scores.shape[0]):
        active = torch.nonzero(group_mask[batch_index], as_tuple=False).flatten()
        empty = active[selected_masks[batch_index, active].sum(dim=1) == 0]
        if empty.numel() != 1:
            raise ValueError("each structured state must contain one empty coalition")
        empty_index = int(empty.item())
        singletons = active[selected_masks[batch_index, active].sum(dim=1) == 1]
        for group_index_tensor in singletons:
            group_index = int(group_index_tensor.item())
            shared = (
                action_mask[batch_index, empty_index, 1:]
                & action_mask[batch_index, group_index, 1:]
            )
            if bool(shared.any()):
                interaction_predictions.append(
                    action_scores[batch_index, group_index, 1:][shared]
                    - action_scores[batch_index, empty_index, 1:][shared]
                )
                interaction_targets.append(
                    normalized_targets[batch_index, group_index, 1:][shared]
                    - normalized_targets[batch_index, empty_index, 1:][shared]
                )
    if interaction_targets:
        predicted_interactions = torch.cat(interaction_predictions)
        target_interactions = torch.cat(interaction_targets)
        interaction_regression = torch.nn.functional.smooth_l1_loss(
            predicted_interactions,
            target_interactions,
            beta=config.smooth_l1_beta,
        )
        signed = target_interactions.abs() > config.interaction_epsilon
        if bool(signed.any()):
            interaction_sign = torch.nn.functional.softplus(
                -predicted_interactions[signed]
                * torch.sign(target_interactions[signed])
            ).mean()
            interaction_sign_accuracy = float(
                (
                    (predicted_interactions[signed] > 0)
                    == (target_interactions[signed] > 0)
                )
                .to(dtype=torch.float32)
                .mean()
                .detach()
            )
        else:
            interaction_sign = action_scores.sum() * 0.0
            interaction_sign_accuracy = 0.0
        interaction_count = int(target_interactions.numel())
    else:
        interaction_regression = action_scores.sum() * 0.0
        interaction_sign = action_scores.sum() * 0.0
        interaction_sign_accuracy = 0.0
        interaction_count = 0

    total = (
        config.marginal_regression * marginal
        + config.decision_ranking * ranking
        + config.stop_calibration * stop_calibration
        + config.interaction_regression * interaction_regression
        + config.interaction_sign * interaction_sign
    )
    predicted_action = logits.argmax(dim=2)
    action_accuracy = float(
        (((predicted_action == teacher_action) & group_mask).sum()
         / group_mask.sum().clamp_min(1)).detach()
    )
    teacher_stop_mask = (teacher_action == 0) & group_mask
    stop_accuracy = float(
        (((predicted_action == 0) & teacher_stop_mask).sum()
         / teacher_stop_mask.sum().clamp_min(1)).detach()
    )
    return total, {
        "action_top1_accuracy": action_accuracy,
        "interaction_count": float(interaction_count),
        "interaction_regression": float(interaction_regression.detach()),
        "interaction_sign": float(interaction_sign.detach()),
        "interaction_sign_accuracy": interaction_sign_accuracy,
        "marginal_regression": float(marginal.detach()),
        "decision_ranking": float(ranking.detach()),
        "stop_action_accuracy": stop_accuracy,
        "stop_brier": float(stop_brier.detach()),
        "stop_calibration": float(stop_calibration.detach()),
        "total": float(total.detach()),
    }


def structured_greedy_budget_path(
    score: Callable[[tuple[int, ...]], Sequence[float]],
    event_ids: Sequence[int],
    *,
    budgets: Sequence[int] = (1, 2, 3, 4),
) -> dict[str, Any]:
    """Run at-most-B selection using a calibrated STOP threshold."""
    candidates = tuple(event_ids)
    normalized_budgets = tuple(budgets)
    if (
        not candidates
        or len(candidates) != len(set(candidates))
        or not normalized_budgets
        or tuple(sorted(set(normalized_budgets))) != normalized_budgets
        or normalized_budgets[0] <= 0
    ):
        raise ValueError("structured greedy inputs are invalid")
    selected: list[int] = []
    stopped = False
    cumulative = 0.0
    selections: dict[str, list[int]] = {}
    predicted_utilities: dict[str, float] = {}
    trace = []
    for budget in normalized_budgets:
        while not stopped and len(selected) < min(budget, len(candidates)):
            values = tuple(float(value) for value in score(tuple(selected)))
            if len(values) != len(candidates) + 1 or any(
                not math.isfinite(value) for value in values
            ):
                raise ValueError("structured score callback returned invalid actions")
            stop_threshold = values[0]
            remaining = [
                (index, event_id, values[index + 1])
                for index, event_id in enumerate(candidates)
                if event_id not in selected
            ]
            best_index, best_event, best_marginal = min(
                remaining, key=lambda row: (-row[2], row[1])
            )
            trace.append(
                {
                    "base_subset": list(selected),
                    "best_event": best_event,
                    "best_marginal": best_marginal,
                    "stop_threshold": stop_threshold,
                }
            )
            if best_marginal <= stop_threshold:
                stopped = True
            else:
                selected.append(candidates[best_index])
                selected.sort()
                cumulative += best_marginal
        selections[str(budget)] = list(selected)
        predicted_utilities[str(budget)] = cumulative
    return {
        "predicted_utilities": predicted_utilities,
        "selections": selections,
        "trace": trace,
    }


__all__ = [
    "StructuredConditionalMarginalHead",
    "StructuredMarginalHeadConfig",
    "StructuredMarginalLossConfig",
    "StructuredTokenConditionalMarginalPredictor",
    "structured_conditional_marginal_loss",
    "structured_greedy_budget_path",
]
