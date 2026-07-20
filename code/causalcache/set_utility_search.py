"""Joint exhaustive at-most-budget search for predicted set utility."""

from __future__ import annotations

import inspect
import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.set_utility_data import SetUtilityState


@dataclass(frozen=True)
class JointSearchResult:
    selected_subset: tuple[int, ...]
    selected_predicted_utility: float
    scored_subsets: tuple[tuple[tuple[int, ...], float], ...]


def _event_id_tuple(event_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(event_ids, (str, bytes, bytearray, Mapping)):
        raise TypeError("event ids must be a sequence")
    result = tuple(event_ids)
    if (
        any(type(event_id) is not int for event_id in result)
        or result != tuple(sorted(result))
        or len(set(result)) != len(result)
    ):
        raise ValueError("event ids must be sorted unique integers")
    return result


def enumerate_at_most_budget_subsets(
    event_ids: Sequence[int],
    *,
    budget: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate empty plus every feasible subset in cardinality/lexical order."""
    events = _event_id_tuple(event_ids)
    if type(budget) is not int or budget < 0:
        raise ValueError("search budget must be a non-negative integer")
    maximum = min(budget, len(events))
    return tuple(
        subset
        for cardinality in range(maximum + 1)
        for subset in itertools.combinations(events, cardinality)
    )


def joint_at_most_budget_search(
    event_ids: Sequence[int],
    *,
    budget: int,
    score: Callable[[tuple[int, ...]], float],
) -> JointSearchResult:
    """Maximize a black-box predicted utility with the frozen deterministic ties."""
    if not callable(score):
        raise TypeError("joint-search score must be callable")
    subsets = enumerate_at_most_budget_subsets(event_ids, budget=budget)
    scored: list[tuple[tuple[int, ...], float]] = []
    for subset in subsets:
        value = float(score(subset))
        if not math.isfinite(value):
            raise ValueError("predicted set utility is non-finite")
        scored.append((subset, value))
    selected_subset, selected_utility = min(
        scored,
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )
    return JointSearchResult(
        selected_subset=selected_subset,
        selected_predicted_utility=selected_utility,
        scored_subsets=tuple(scored),
    )


def _batched_scores(
    subsets: tuple[tuple[int, ...], ...],
    *,
    score_batch: Callable[[tuple[tuple[int, ...], ...]], Iterable[float]],
) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in score_batch(subsets))
    except TypeError as error:
        raise TypeError("batched subset scorer must return an iterable") from error
    if len(values) != len(subsets):
        raise ValueError("batched subset score count drifted")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("predicted set utility is non-finite")
    return values


def conditional_greedy_at_most_budget_search(
    event_ids: Sequence[int],
    *,
    budget: int,
    score_batch: Callable[[tuple[tuple[int, ...], ...]], Iterable[float]],
) -> JointSearchResult:
    """Greedily add one event while predicted joint utility strictly improves."""
    events = _event_id_tuple(event_ids)
    if type(budget) is not int or budget not in (1, 2, 3, 4):
        raise ValueError("conditional greedy budget must be one of 1, 2, 3, 4")
    if not callable(score_batch):
        raise TypeError("conditional greedy score_batch must be callable")

    current: tuple[int, ...] = ()
    current_utility = _batched_scores((current,), score_batch=score_batch)[0]
    scored: list[tuple[tuple[int, ...], float]] = [(current, current_utility)]
    for _ in range(min(budget, len(events))):
        candidates = tuple(
            tuple(sorted((*current, event_id)))
            for event_id in events
            if event_id not in current
        )
        if not candidates:
            break
        values = _batched_scores(candidates, score_batch=score_batch)
        scored.extend(zip(candidates, values, strict=True))
        best_subset, best_utility = min(
            zip(candidates, values, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        if best_utility <= current_utility:
            break
        current = best_subset
        current_utility = best_utility
    return JointSearchResult(
        selected_subset=current,
        selected_predicted_utility=current_utility,
        scored_subsets=tuple(scored),
    )


def exact_utility_oracle(
    state: SetUtilityState,
    *,
    budget: int,
) -> JointSearchResult:
    """Use the same at-most-budget contract with true utility as an offline oracle."""
    if not isinstance(state, SetUtilityState):
        raise TypeError("state must be SetUtilityState")
    if budget > state.maximum_labeled_cardinality:
        raise ValueError("exact oracle budget exceeds the state's label-cardinality cap")
    return joint_at_most_budget_search(
        state.event_ids,
        budget=budget,
        score=state.utility,
    )


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("learned joint subset search requires PyTorch") from error
    return torch


def learned_joint_at_most_budget_search(
    model: Any,
    state: SetUtilityState,
    *,
    budget: int,
) -> JointSearchResult:
    """Score every feasible subset in one model call; B never enters the model."""
    if not isinstance(state, SetUtilityState):
        raise TypeError("state must be SetUtilityState")
    subsets = enumerate_at_most_budget_subsets(state.event_ids, budget=budget)
    torch = _torch()
    try:
        reference = next(model.parameters())
    except (AttributeError, StopIteration, TypeError) as error:
        raise TypeError("learned search requires a parameterized PyTorch model") from error
    event_index = {
        event_id: index for index, event_id in enumerate(state.event_ids)
    }
    subset_masks = [
        [event_id in subset for event_id in state.event_ids]
        for subset in subsets
    ]
    query = torch.tensor(
        [state.query_features], dtype=reference.dtype, device=reference.device
    )
    context = torch.tensor(
        [state.context_features], dtype=reference.dtype, device=reference.device
    )
    events = torch.tensor(
        [state.event_features], dtype=reference.dtype, device=reference.device
    )
    masks = torch.tensor(
        [subset_masks], dtype=torch.bool, device=reference.device
    )
    event_mask = torch.ones(
        (1, len(event_index)), dtype=torch.bool, device=reference.device
    )
    keyword: dict[str, Any] = {}
    try:
        parameters = inspect.signature(model.score_subsets).parameters
    except (AttributeError, TypeError, ValueError):
        parameters = {}
    if state.pair_features is not None and "pair_features" in parameters:
        keyword["pair_features"] = torch.tensor(
            [state.pair_features],
            dtype=reference.dtype,
            device=reference.device,
        )
    model.eval()
    with torch.no_grad():
        predictions = model(
            query,
            context,
            events,
            masks,
            event_mask,
            **keyword,
        )
    if predictions.shape != (1, len(subsets)):
        raise ValueError("learned joint-search prediction shape drifted")
    values = tuple(float(value) for value in predictions[0].tolist())
    if any(not math.isfinite(value) for value in values):
        raise ValueError("learned joint-search prediction is non-finite")
    scored = tuple(zip(subsets, values, strict=True))
    selected_subset, selected_utility = min(
        scored,
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )
    return JointSearchResult(selected_subset, selected_utility, scored)


__all__ = [
    "JointSearchResult",
    "conditional_greedy_at_most_budget_search",
    "enumerate_at_most_budget_subsets",
    "exact_utility_oracle",
    "joint_at_most_budget_search",
    "learned_joint_at_most_budget_search",
]
