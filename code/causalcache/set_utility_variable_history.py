"""Variable-history states and label-blind stratified coalition sampling."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


HISTORY_BINS = (
    ("short", 5, 8),
    ("medium", 9, 16),
    ("long", 17, 32),
    ("very_long", 33, None),
)
SPLIT_ROLES = ("train", "tune", "evaluation")


def _canonical_event_ids(values: Sequence[int], *, allow_empty: bool) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError("event ids must be an ordered sequence")
    result = tuple(values)
    if (
        (not allow_empty and not result)
        or any(type(value) is not int or value <= 0 for value in result)
        or result != tuple(sorted(result))
        or len(result) != len(set(result))
    ):
        raise ValueError("event ids must be sorted unique positive integers")
    return result


def history_bin(candidate_count: int) -> str:
    if type(candidate_count) is not int or candidate_count < 5:
        raise ValueError("candidate count must be an integer at least five")
    for name, lower, upper in HISTORY_BINS:
        if candidate_count >= lower and (upper is None or candidate_count <= upper):
            return name
    raise RuntimeError("history-bin routing is incomplete")


@dataclass(frozen=True)
class VariableHistoryState:
    state_id: str
    trajectory_id: str
    role: str
    decision_step_id: int
    candidate_event_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.state_id or not self.trajectory_id:
            raise ValueError("state and trajectory identities must be non-empty")
        if self.role not in SPLIT_ROLES:
            raise ValueError("state role is invalid")
        if type(self.decision_step_id) is not int or self.decision_step_id < 6:
            raise ValueError("decision step must be at least six")
        candidates = _canonical_event_ids(
            self.candidate_event_ids,
            allow_empty=False,
        )
        expected = tuple(range(1, self.decision_step_id))
        if candidates != expected:
            raise ValueError("candidate universe is not the complete prior history")
        expected_state_id = f"{self.trajectory_id}:decision:{self.decision_step_id:03d}"
        if self.state_id != expected_state_id:
            raise ValueError("state identity differs from trajectory/decision identity")

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_event_ids)

    @property
    def history_bin(self) -> str:
        return history_bin(self.candidate_count)

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_event_ids": list(self.candidate_event_ids),
            "decision_step_id": self.decision_step_id,
            "history_bin": self.history_bin,
            "role": self.role,
            "state_id": self.state_id,
            "trajectory_id": self.trajectory_id,
        }


def build_variable_history_states(
    *,
    trajectory_id: str,
    role: str,
    decision_count: int,
) -> tuple[VariableHistoryState, ...]:
    if type(decision_count) is not int or decision_count < 6:
        raise ValueError("decision count must be at least six")
    return tuple(
        VariableHistoryState(
            state_id=f"{trajectory_id}:decision:{step:03d}",
            trajectory_id=trajectory_id,
            role=role,
            decision_step_id=step,
            candidate_event_ids=tuple(range(1, step)),
        )
        for step in range(6, decision_count + 2)
    )


def states_from_assignments(
    assignments: Sequence[Mapping[str, Any]],
) -> tuple[VariableHistoryState, ...]:
    if isinstance(assignments, (str, bytes, bytearray, Mapping)):
        raise TypeError("assignments must be an ordered sequence")
    states = tuple(
        state
        for assignment in assignments
        for state in build_variable_history_states(
            trajectory_id=str(assignment["trajectory_id"]),
            role=str(assignment["role"]),
            decision_count=int(assignment["decision_count"]),
        )
    )
    identities = tuple(state.state_id for state in states)
    if len(identities) != len(set(identities)):
        raise ValueError("variable-history state identities are duplicated")
    return states


def age_quantile_bins(
    candidate_event_ids: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    events = _canonical_event_ids(candidate_event_ids, allow_empty=False)
    if len(events) < 4:
        raise ValueError("four age bins require at least four events")
    base, remainder = divmod(len(events), 4)
    sizes = tuple(base + int(index < remainder) for index in range(4))
    result = []
    offset = 0
    for size in sizes:
        result.append(events[offset : offset + size])
        offset += size
    if offset != len(events) or any(not values for values in result):
        raise RuntimeError("age-bin partition is incomplete")
    return tuple(result)


def combined_pair_similarity(
    candidate_event_ids: Sequence[int],
    *,
    visual_embeddings: Mapping[int, Sequence[float]],
    ocr_token_sets: Mapping[int, Sequence[str] | set[str] | frozenset[str]],
) -> dict[tuple[int, int], float]:
    events = _canonical_event_ids(candidate_event_ids, allow_empty=False)
    if set(visual_embeddings) != set(events) or set(ocr_token_sets) != set(events):
        raise ValueError("similarity inputs do not exactly cover the candidate universe")
    vectors = {event: tuple(float(value) for value in visual_embeddings[event]) for event in events}
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) <= 0:
        raise ValueError("visual embeddings must share one positive dimension")
    norms = {
        event: math.sqrt(sum(value * value for value in vector))
        for event, vector in vectors.items()
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in norms.values()):
        raise ValueError("visual embedding norms must be finite and positive")
    tokens = {event: frozenset(ocr_token_sets[event]) for event in events}
    result: dict[tuple[int, int], float] = {}
    for left, right in itertools.combinations(events, 2):
        cosine = sum(
            a * b for a, b in zip(vectors[left], vectors[right], strict=True)
        ) / (norms[left] * norms[right])
        union = tokens[left] | tokens[right]
        jaccard = len(tokens[left] & tokens[right]) / len(union) if union else 1.0
        score = 0.5 * cosine + 0.5 * jaccard
        if not math.isfinite(score):
            raise ValueError("combined similarity is not finite")
        result[(left, right)] = score
    return result


@dataclass(frozen=True)
class BroadSubsetSchedule:
    state_id: str
    candidate_event_ids: tuple[int, ...]
    coalitions: tuple[tuple[int, ...], ...]
    sources: tuple[str, ...]
    exact: bool

    def __post_init__(self) -> None:
        candidates = _canonical_event_ids(self.candidate_event_ids, allow_empty=False)
        if len(self.coalitions) != len(self.sources) or not self.coalitions:
            raise ValueError("coalitions and sources must be non-empty and aligned")
        normalized = tuple(
            _canonical_event_ids(coalition, allow_empty=True)
            for coalition in self.coalitions
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("sampled coalitions are duplicated")
        if any(not set(coalition).issubset(candidates) for coalition in normalized):
            raise ValueError("sampled coalition escapes the candidate universe")
        if normalized[0] != () or candidates not in normalized:
            raise ValueError("sampled coalitions must contain empty and full anchors")

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_event_ids": list(self.candidate_event_ids),
            "coalitions": [
                {"event_ids": list(coalition), "source": source}
                for coalition, source in zip(self.coalitions, self.sources, strict=True)
            ],
            "exact": self.exact,
            "state_id": self.state_id,
        }


def _state_rng(state_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{state_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big"))


def logical_shard_for_trajectory(
    trajectory_id: str,
    *,
    shard_count: int,
) -> int:
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory identity must be non-empty")
    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("logical shard count must be positive")
    digest = hashlib.sha256(trajectory_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % shard_count


def _pair_inventory(
    events: tuple[int, ...],
    pair_similarity: Mapping[tuple[int, int], float],
) -> tuple[tuple[int, int], ...]:
    expected = tuple(itertools.combinations(events, 2))
    if set(pair_similarity) != set(expected):
        raise ValueError("pair similarity must exactly cover every candidate pair")
    if any(not math.isfinite(float(pair_similarity[pair])) for pair in expected):
        raise ValueError("pair similarity contains a non-finite value")
    return expected


def sample_broad_subsets(
    *,
    state_id: str,
    candidate_event_ids: Sequence[int],
    pair_similarity: Mapping[tuple[int, int], float],
    target_unique_count: int = 40,
    small_history_exact_maximum_n: int = 5,
    seed: int = 20260720,
) -> BroadSubsetSchedule:
    events = _canonical_event_ids(candidate_event_ids, allow_empty=False)
    if len(events) < 5:
        raise ValueError("formal variable-history sampling starts at n=5")
    pairs = _pair_inventory(events, pair_similarity)
    if len(events) <= small_history_exact_maximum_n:
        coalitions = tuple(
            coalition
            for cardinality in range(len(events) + 1)
            for coalition in itertools.combinations(events, cardinality)
        )
        return BroadSubsetSchedule(
            state_id=state_id,
            candidate_event_ids=events,
            coalitions=coalitions,
            sources=tuple("small_history_exact" for _ in coalitions),
            exact=True,
        )
    if type(target_unique_count) is not int or target_unique_count <= 0:
        raise ValueError("target unique count must be positive")
    rng = _state_rng(state_id, seed)
    bins = age_quantile_bins(events)
    sampled: dict[tuple[int, ...], str] = {}

    def add(values: Sequence[int], source: str) -> bool:
        coalition = tuple(sorted(values))
        if not set(coalition).issubset(events):
            raise RuntimeError("sampler emitted an event outside the universe")
        if coalition in sampled:
            return False
        sampled[coalition] = source
        return True

    add((), "anchor_empty")
    add(events, "anchor_full")

    if len(events) <= 8:
        singleton_events = events
    else:
        singleton_events = tuple(
            event
            for age_bin in bins
            for event in sorted(rng.sample(age_bin, 2))
        )
    for event in singleton_events:
        add((event,), "singleton_age_stratified")

    for chain_index in range(4):
        chosen = [age_bin[chain_index % len(age_bin)] for age_bin in bins]
        rng.shuffle(chosen)
        for prefix_length in range(1, 5):
            add(chosen[:prefix_length], f"conditional_chain_{chain_index}")

    def add_pairs(inventory: Sequence[tuple[int, int]], count: int, source: str) -> None:
        added = 0
        values = list(inventory)
        rng.shuffle(values)
        for pair in values:
            if add(pair, source):
                added += 1
                if added == count:
                    break

    add_pairs(tuple(itertools.product(bins[0], bins[3])), 2, "pair_very_old_recent")
    add_pairs(tuple(itertools.combinations(bins[0], 2)), 1, "pair_very_old")
    add_pairs(tuple(itertools.combinations(bins[3], 2)), 1, "pair_very_recent")
    similarity_descending = sorted(
        pairs,
        key=lambda pair: (-float(pair_similarity[pair]), pair),
    )
    similarity_cross_age = [
        pair
        for pair in sorted(pairs, key=lambda pair: (float(pair_similarity[pair]), pair))
        if (
            (pair[0] in bins[0] and pair[1] in bins[2] + bins[3])
            or (pair[0] in bins[1] and pair[1] in bins[3])
        )
    ]
    add_pairs(similarity_descending, 2, "pair_high_similarity")
    add_pairs(similarity_cross_age, 2, "pair_low_similarity_cross_age")

    for cardinality in (2, 3, 4):
        for repeat in range(2):
            selected_bins = rng.sample(range(4), cardinality)
            coalition = tuple(
                rng.choice(bins[index]) for index in selected_bins
            )
            add(coalition, f"higher_cardinality_{cardinality}")

    attempts = 0
    cardinalities = (1, 2, 3, 4)
    while len(sampled) < target_unique_count and attempts < 10000:
        cardinality = cardinalities[attempts % len(cardinalities)]
        chosen_bins = rng.sample(range(4), min(cardinality, 4))
        coalition = [rng.choice(bins[index]) for index in chosen_bins]
        while len(set(coalition)) < cardinality:
            coalition.append(rng.choice(events))
        add(tuple(set(coalition)), "dedup_backfill_age_stratified")
        attempts += 1
    if len(sampled) < target_unique_count:
        for cardinality in range(5):
            for coalition in itertools.combinations(events, cardinality):
                add(coalition, "dedup_backfill_global")
                if len(sampled) == target_unique_count:
                    break
            if len(sampled) == target_unique_count:
                break
    if len(sampled) != target_unique_count:
        raise RuntimeError("sampler could not produce the requested unique coalitions")
    coalitions = tuple(sampled)
    return BroadSubsetSchedule(
        state_id=state_id,
        candidate_event_ids=events,
        coalitions=coalitions,
        sources=tuple(sampled[coalition] for coalition in coalitions),
        exact=False,
    )


def _stable_state_order(states: Sequence[VariableHistoryState], *, salt: str) -> tuple[VariableHistoryState, ...]:
    return tuple(
        sorted(
            states,
            key=lambda state: (
                hashlib.sha256(f"{salt}:{state.state_id}".encode("utf-8")).hexdigest(),
                state.state_id,
            ),
        )
    )


@dataclass(frozen=True)
class EvaluationTracks:
    exact_state_ids: tuple[str, ...]
    large_history_state_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.exact_state_ids) != len(set(self.exact_state_ids)):
            raise ValueError("exact evaluation state IDs are duplicated")
        if len(self.large_history_state_ids) != len(set(self.large_history_state_ids)):
            raise ValueError("large-history evaluation state IDs are duplicated")


def select_evaluation_tracks(
    states: Sequence[VariableHistoryState],
    *,
    exact_state_count: int = 320,
    large_history_state_count: int = 720,
    seed: int = 20260720,
) -> EvaluationTracks:
    evaluation = tuple(state for state in states if state.role == "evaluation")
    by_bin = {
        name: _stable_state_order(
            tuple(state for state in evaluation if state.history_bin == name),
            salt=f"{seed}:exact:{name}",
        )
        for name, _, _ in HISTORY_BINS
    }
    very_long = by_bin["very_long"]
    remaining = exact_state_count - len(very_long)
    if remaining < 0:
        raise ValueError("exact count is smaller than the very-long inventory")
    base, remainder = divmod(remaining, 3)
    exact = list(very_long)
    for index, name in enumerate(("short", "medium", "long")):
        count = base + int(index < remainder)
        if len(by_bin[name]) < count:
            raise ValueError("an exact evaluation history bin is undersized")
        exact.extend(by_bin[name][:count])
    exact = list(_stable_state_order(exact, salt=f"{seed}:exact:final"))

    large: list[VariableHistoryState] = []
    for name in ("very_long", "long", "medium", "short"):
        for state in _stable_state_order(by_bin[name], salt=f"{seed}:large:{name}"):
            if len(large) == large_history_state_count:
                break
            large.append(state)
        if len(large) == large_history_state_count:
            break
    if len(exact) != exact_state_count or len(large) != large_history_state_count:
        raise RuntimeError("evaluation track selection did not reach its frozen counts")
    return EvaluationTracks(
        exact_state_ids=tuple(state.state_id for state in exact),
        large_history_state_ids=tuple(state.state_id for state in large),
    )


def state_count_summary(states: Sequence[VariableHistoryState]) -> dict[str, Any]:
    role_counts = Counter(state.role for state in states)
    role_bin_counts = Counter((state.role, state.history_bin) for state in states)
    return {
        "state_count": len(states),
        "state_count_by_role": dict(sorted(role_counts.items())),
        "state_count_by_role_and_history_bin": {
            role: {
                name: role_bin_counts[(role, name)]
                for name, _, _ in HISTORY_BINS
            }
            for role in SPLIT_ROLES
        },
    }


__all__ = [
    "BroadSubsetSchedule",
    "EvaluationTracks",
    "HISTORY_BINS",
    "VariableHistoryState",
    "age_quantile_bins",
    "build_variable_history_states",
    "combined_pair_similarity",
    "history_bin",
    "logical_shard_for_trajectory",
    "sample_broad_subsets",
    "select_evaluation_tracks",
    "state_count_summary",
    "states_from_assignments",
]
