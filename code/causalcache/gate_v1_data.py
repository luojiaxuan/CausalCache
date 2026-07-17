"""Label-blind features and training rows for the frozen CausalCache gate v1."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.low_fidelity_v2 import LOW_FIDELITY_V2_KEYS
from causalcache.restoration_v2_2_label_table import (
    ValidatedDistanceTable,
    deployment_conditional_edges,
    validate_complete_distance_table,
)


HASH_DIMENSION = 64
CONDITIONAL_INPUT_DIMENSION = 330
INDEPENDENT_INPUT_DIMENSION = 200
DEPLOYMENT_BUDGET = 2
NORMALIZATION_EPSILON = 1e-12
LABEL_TIE_EPSILON = 1e-12
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class CandidateFeatures:
    event_step_id: int
    h64: tuple[float, ...]
    g8: tuple[float, ...]


@dataclass(frozen=True)
class FeatureState:
    """A feature-only state; identity fields are join keys, never model inputs."""

    source_id: str
    state_id: str
    decision_step_id: int
    candidate_event_step_ids: tuple[int, ...]
    q64: tuple[float, ...]
    candidates: tuple[CandidateFeatures, ...]


@dataclass(frozen=True)
class LabelState:
    """A label-only state; it intentionally contains no semantic feature fields."""

    source_id: str
    state_id: str
    decision_step_id: int
    table: ValidatedDistanceTable


@dataclass(frozen=True)
class GateState:
    source_id: str
    state_id: str
    decision_step_id: int
    candidate_event_step_ids: tuple[int, ...]
    q64: tuple[float, ...]
    candidates: tuple[CandidateFeatures, ...]
    table: ValidatedDistanceTable

    def candidate(self, event_step_id: int) -> CandidateFeatures:
        for candidate in self.candidates:
            if candidate.event_step_id == event_step_id:
                return candidate
        raise KeyError(event_step_id)


@dataclass(frozen=True)
class GateExample:
    source_id: str
    state_id: str
    decision_step_id: int
    event_step_id: int
    coalition: tuple[int, ...]
    input_vector: tuple[float, ...]
    raw_target: float
    normalized_target: float | None
    regression_weight: float


@dataclass(frozen=True)
class RankingPair:
    left_index: int
    right_index: int
    target_sign: int
    weight: float


@dataclass(frozen=True)
class TrainingBatch:
    family: str
    examples: tuple[GateExample, ...]
    ranking_pairs: tuple[RankingPair, ...]

    @property
    def input_dimension(self) -> int:
        return (
            CONDITIONAL_INPUT_DIMENSION
            if self.family == "conditional"
            else INDEPENDENT_INPUT_DIMENSION
        )

    @property
    def regression_weight_sum(self) -> float:
        return math.fsum(example.regression_weight for example in self.examples)


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("gate text feature must be a string")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _field_tokens(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        normalized = normalize_text(value)
        return tuple(normalized.split(" ")) if normalized else ()
    if isinstance(value, Mapping) or isinstance(value, (bytes, bytearray)):
        raise TypeError("gate feature field cannot be a mapping or bytes")
    if not isinstance(value, Sequence):
        normalized = normalize_text(str(value))
        return (normalized,) if normalized else ()
    tokens: list[str] = []
    for item in value:
        normalized = normalize_text(str(item))
        if normalized:
            tokens.extend(normalized.split(" "))
    return tuple(tokens)


def signed_hash64(fields: Sequence[tuple[str, str | Sequence[Any]]]) -> tuple[float, ...]:
    """Replay the frozen SHA256 signed-hashing contract."""
    values = [0.0] * HASH_DIMENSION
    for namespace, raw_tokens in fields:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("feature namespace must be non-empty text")
        for token in _field_tokens(raw_tokens):
            digest = hashlib.sha256(f"{namespace}\0{token}".encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % HASH_DIMENSION
            values[index] += 1.0 if digest[8] % 2 == 0 else -1.0
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm > 0.0:
        values = [value / norm for value in values]
    result = tuple(values)
    _finite_vector(result, HASH_DIMENSION, "signed hash")
    return result


def _finite_vector(values: Sequence[float], dimension: int, label: str) -> None:
    if len(values) != dimension or any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} must contain {dimension} finite values")


def _candidate_h64(
    low_fidelity: Mapping[str, Any], post_ocr_tokens: Sequence[str]
) -> tuple[float, ...]:
    if set(low_fidelity) != set(LOW_FIDELITY_V2_KEYS):
        raise ValueError("low-fidelity field inventory drifted from gate v1")
    fields: list[tuple[str, str | Sequence[Any]]] = []
    for name in LOW_FIDELITY_V2_KEYS:
        fields.append((name, low_fidelity[name]))
    fields.append(("candidate_post_ocr_spatial_token", post_ocr_tokens))
    return signed_hash64(fields)


def _candidate_g8(
    low_fidelity: Mapping[str, Any],
    *,
    history_length: int,
    post_ocr_tokens: Sequence[str],
) -> tuple[float, ...]:
    step = low_fidelity.get("step_id")
    if type(history_length) is not int or history_length <= 0:
        raise ValueError("history length must be positive")
    if type(step) is not int or step <= 0 or step > history_length:
        raise ValueError("candidate step is outside the strict history")
    argument = unicodedata.normalize("NFKC", str(low_fidelity["action_argument"]))
    added = low_fidelity["screen_text_added"]
    removed = low_fidelity["screen_text_removed"]
    if not isinstance(added, list) or not isinstance(removed, list):
        raise ValueError("screen-text deltas must be arrays")
    change_score = {"none": 0.0, "low": 1.0 / 3.0, "medium": 2.0 / 3.0, "high": 1.0}
    result_score = {"failed": 0.0, "unknown": 0.5, "accepted": 1.0}
    try:
        values = (
            (history_length - step) / history_length,
            step / history_length,
            min(len(argument), 64) / 64.0,
            min(len(added), 32) / 32.0,
            min(len(removed), 32) / 32.0,
            min(len(post_ocr_tokens), 128) / 128.0,
            change_score[str(low_fidelity["screen_change"])],
            result_score[str(low_fidelity["executor_result"])],
        )
    except KeyError as error:
        raise ValueError("gate g8 categorical value drifted") from error
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("gate g8 values must be finite in [0, 1]")
    return values


def feature_state_from_derived(
    trajectory: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
) -> FeatureState:
    """Build one label-blind feature state from a validated derived trajectory."""
    source_id = trajectory.get("source_id")
    state_id = decision.get("state_id")
    decision_step = decision.get("decision_step_id")
    candidate_ids = decision.get("candidate_event_step_ids")
    history_ids = decision.get("history_event_step_ids")
    if not isinstance(source_id, str) or not source_id or not isinstance(state_id, str) or not state_id:
        raise ValueError("feature join identities must be non-empty strings")
    if type(decision_step) is not int or decision_step <= 0:
        raise ValueError("decision step must be positive")
    if not isinstance(candidate_ids, list) or not isinstance(history_ids, list):
        raise ValueError("decision history geometry is malformed")
    event_ids = tuple(candidate_ids)
    if event_ids not in ((1, 2), (1, 2, 3), (1, 2, 3, 4)):
        raise ValueError("gate v1 candidates must be the frozen 2/3/4 prefixes")
    if history_ids != list(range(1, decision_step)) or event_ids != tuple(history_ids[:-1]):
        raise ValueError("decision view is not a strict history prefix")
    current_path = decision.get("current_observation_path")
    try:
        current_tokens = ocr_records_by_path[str(current_path)]["full_spatial_tokens"]
    except (KeyError, TypeError) as error:
        raise ValueError("current OCR feature record is missing") from error
    if not isinstance(current_tokens, list):
        raise ValueError("current OCR spatial tokens must be an array")
    q64 = signed_hash64(
        (
            ("instruction", str(trajectory.get("instruction", ""))),
            ("current_ocr_spatial_token", current_tokens),
        )
    )
    events = trajectory.get("events")
    if not isinstance(events, list):
        raise ValueError("derived trajectory events are missing")
    by_step = {event.get("step_id"): event for event in events if isinstance(event, Mapping)}
    if len(by_step) != len(events):
        raise ValueError("derived events contain duplicate or malformed steps")
    candidates: list[CandidateFeatures] = []
    for event_id in event_ids:
        try:
            event = by_step[event_id]
            low_fidelity = event["low_fidelity_v2"]
            post_path = event["observation_after_path"]
            post_tokens = ocr_records_by_path[post_path]["full_spatial_tokens"]
        except (KeyError, TypeError) as error:
            raise ValueError("candidate semantic feature input is missing") from error
        if not isinstance(low_fidelity, Mapping) or not isinstance(post_tokens, list):
            raise ValueError("candidate low-fidelity or OCR feature is malformed")
        candidates.append(
            CandidateFeatures(
                event_step_id=event_id,
                h64=_candidate_h64(low_fidelity, post_tokens),
                g8=_candidate_g8(
                    low_fidelity,
                    history_length=len(history_ids),
                    post_ocr_tokens=post_tokens,
                ),
            )
        )
    return FeatureState(
        source_id=source_id,
        state_id=state_id,
        decision_step_id=decision_step,
        candidate_event_step_ids=event_ids,
        q64=q64,
        candidates=tuple(candidates),
    )


def label_state_from_restoration_record(
    record: Mapping[str, Any],
    *,
    record_schema: str = "legacy",
) -> LabelState:
    """Project a validated restoration state using one explicit identity schema."""
    schemas = {
        "legacy": {
            "identity": "trajectory_id",
            "other_identity": "source_id",
            "coalition": "coalition_event_step_ids",
            "distance": "distance_kl",
            "other_row_fields": {"coalition", "distance"},
        },
        "expansion": {
            "identity": "source_id",
            "other_identity": "trajectory_id",
            "coalition": "coalition",
            "distance": "distance",
            "other_row_fields": {
                "coalition_event_step_ids",
                "distance_kl",
            },
        },
    }
    if not isinstance(record_schema, str) or record_schema not in schemas:
        raise ValueError("restoration label record schema is unsupported")
    schema = schemas[record_schema]
    state = record.get("state")
    rows = record.get("distance_rows")
    if not isinstance(state, Mapping) or not isinstance(rows, list):
        raise ValueError("restoration label state or distance rows are missing")
    identity_field = schema["identity"]
    other_identity_field = schema["other_identity"]
    source_id = state.get(identity_field)
    if other_identity_field in state:
        raise ValueError("restoration label state mixes identity schemas")
    state_id = state.get("state_id")
    decision_step_id = state.get("decision_step_id")
    event_ids = state.get("candidate_event_step_ids")
    if (
        not isinstance(source_id, str)
        or not source_id
        or not isinstance(state_id, str)
        or not state_id
        or type(decision_step_id) is not int
        or decision_step_id <= 0
        or not isinstance(event_ids, list)
    ):
        raise ValueError("restoration label state identity is malformed")
    distances: dict[tuple[int, ...], Any] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("restoration distance row is malformed")
        if set(row) & schema["other_row_fields"]:
            raise ValueError("restoration distance row mixes record schemas")
        coalition_field = schema["coalition"]
        distance_field = schema["distance"]
        if not isinstance(row.get(coalition_field), list) or distance_field not in row:
            raise ValueError("restoration distance row is malformed")
        coalition = tuple(row[coalition_field])
        if coalition in distances:
            raise ValueError("restoration distance rows contain duplicate coalitions")
        distances[coalition] = row[distance_field]
    table = validate_complete_distance_table(tuple(event_ids), distances)
    return LabelState(
        source_id=source_id,
        state_id=state_id,
        decision_step_id=decision_step_id,
        table=table,
    )


def join_feature_and_label_states(
    features: Sequence[FeatureState],
    labels: Sequence[LabelState],
    *,
    expected_source_ids: Sequence[str] | None = None,
) -> tuple[GateState, ...]:
    """Join physically separate caches only after both sides were validated."""
    feature_by_key = {(state.source_id, state.state_id): state for state in features}
    label_by_key = {(state.source_id, state.state_id): state for state in labels}
    if len(feature_by_key) != len(features) or len(label_by_key) != len(labels):
        raise ValueError("feature or label cache contains duplicate join keys")
    if set(feature_by_key) != set(label_by_key):
        raise ValueError("feature and label cache join-key inventories differ")
    source_order = tuple(dict.fromkeys(state.source_id for state in features))
    if expected_source_ids is not None and source_order != tuple(expected_source_ids):
        raise ValueError("gate cache source roster or order drifted")
    result: list[GateState] = []
    for feature in features:
        label = label_by_key[(feature.source_id, feature.state_id)]
        candidate_ids = feature.candidate_event_step_ids
        if (
            label.decision_step_id != feature.decision_step_id
            or label.table.event_ids != candidate_ids
            or tuple(candidate.event_step_id for candidate in feature.candidates) != candidate_ids
        ):
            raise ValueError("feature-label state geometry differs")
        _finite_vector(feature.q64, HASH_DIMENSION, "q64")
        for candidate in feature.candidates:
            _finite_vector(candidate.h64, HASH_DIMENSION, "h64")
            _finite_vector(candidate.g8, 8, "g8")
        result.append(
            GateState(
                source_id=feature.source_id,
                state_id=feature.state_id,
                decision_step_id=feature.decision_step_id,
                candidate_event_step_ids=candidate_ids,
                q64=feature.q64,
                candidates=feature.candidates,
                table=label.table,
            )
        )
    return tuple(result)


def validate_canonical_gate_state_roster(
    states: Sequence[GateState],
    expected_source_ids: Sequence[str],
    *,
    expected_source_count: int,
) -> None:
    """Require the frozen source-major step-4/5/6 state and edge order."""
    source_ids = tuple(expected_source_ids)
    if (
        len(source_ids) != expected_source_count
        or len(set(source_ids)) != expected_source_count
        or any(not isinstance(source_id, str) or not source_id for source_id in source_ids)
    ):
        raise ValueError("expected gate source roster is malformed")
    expected = tuple(
        (
            source_id,
            f"{source_id}:decision_step:{decision_step:03d}",
            decision_step,
            tuple(range(1, decision_step - 1)),
        )
        for source_id in source_ids
        for decision_step in (4, 5, 6)
    )
    if len(states) != len(expected):
        raise ValueError("gate state roster count differs from the frozen geometry")
    observed_state_ids: set[str] = set()
    for state, (source_id, state_id, decision_step, candidate_ids) in zip(
        states, expected, strict=True
    ):
        if state.state_id in observed_state_ids:
            raise ValueError("gate state roster contains a duplicate state id")
        observed_state_ids.add(state.state_id)
        if (
            state.source_id != source_id
            or state.state_id != state_id
            or state.decision_step_id != decision_step
            or state.candidate_event_step_ids != candidate_ids
            or state.table.event_ids != candidate_ids
            or tuple(candidate.event_step_id for candidate in state.candidates)
            != candidate_ids
        ):
            raise ValueError("gate state roster order or canonical geometry drifted")


def conditional_input(
    state: GateState,
    event_step_id: int,
    coalition: Sequence[int],
    *,
    budget: int = DEPLOYMENT_BUDGET,
) -> tuple[float, ...]:
    selected = tuple(coalition)
    if tuple(sorted(selected)) != selected or len(set(selected)) != len(selected):
        raise ValueError("selected coalition must be sorted and unique")
    if event_step_id in selected or not set(selected).issubset(state.candidate_event_step_ids):
        raise ValueError("conditional candidate/coalition geometry is invalid")
    if type(budget) is not int or budget <= 0 or len(selected) >= budget:
        raise ValueError("conditional coalition must leave deployment budget")
    candidate = state.candidate(event_step_id)
    selected_sum = tuple(
        math.fsum(state.candidate(chosen).h64[index] for chosen in selected)
        for index in range(HASH_DIMENSION)
    )
    vector = (
        *state.q64,
        *candidate.h64,
        *(left * right for left, right in zip(state.q64, candidate.h64, strict=True)),
        *candidate.g8,
        *selected_sum,
        *(left * right for left, right in zip(state.q64, selected_sum, strict=True)),
        len(selected) / budget,
        (budget - len(selected)) / budget,
    )
    _finite_vector(vector, CONDITIONAL_INPUT_DIMENSION, "conditional input")
    return vector


def independent_input(state: GateState, event_step_id: int) -> tuple[float, ...]:
    candidate = state.candidate(event_step_id)
    vector = (
        *state.q64,
        *candidate.h64,
        *(left * right for left, right in zip(state.q64, candidate.h64, strict=True)),
        *candidate.g8,
    )
    _finite_vector(vector, INDEPENDENT_INPUT_DIMENSION, "independent input")
    return vector


def _eligible_state_counts(states: Sequence[GateState]) -> tuple[set[str], dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for state in states:
        if state.table.distance(()) > NORMALIZATION_EPSILON:
            counts[state.source_id] += 1
    return set(counts), dict(counts)


def build_training_batch(states: Sequence[GateState], *, family: str) -> TrainingBatch:
    if family not in {"conditional", "independent"}:
        raise ValueError("gate family must be conditional or independent")
    if not states:
        raise ValueError("gate training batch cannot be empty")
    source_order = tuple(dict.fromkeys(state.source_id for state in states))
    state_counts: dict[str, int] = defaultdict(int)
    for state in states:
        state_counts[state.source_id] += 1
    eligible_sources, eligible_counts = _eligible_state_counts(states)
    if not eligible_sources:
        raise ValueError("gate batch has no state eligible for normalized regression")
    source_regression_weight = 1.0 / len(eligible_sources)
    source_ranking_weight = 1.0 / len(source_order)
    examples: list[GateExample] = []
    ranking_groups: list[tuple[list[int], float]] = []

    for state in states:
        event_ids = state.candidate_event_step_ids
        baseline = state.table.distance(())
        eligible = baseline > NORMALIZATION_EPSILON
        if family == "conditional":
            edges = deployment_conditional_edges(state.table)
            by_group: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
            for edge in edges:
                coalition = edge.base_coalition
                k = len(coalition)
                coalition_count = math.comb(len(event_ids), k)
                candidate_count = len(event_ids) - k
                regression_weight = (
                    source_regression_weight
                    / eligible_counts[state.source_id]
                    / 2
                    / coalition_count
                    / candidate_count
                    if eligible
                    else 0.0
                )
                index = len(examples)
                examples.append(
                    GateExample(
                        source_id=state.source_id,
                        state_id=state.state_id,
                        decision_step_id=state.decision_step_id,
                        event_step_id=edge.event_id,
                        coalition=coalition,
                        input_vector=conditional_input(state, edge.event_id, coalition),
                        raw_target=edge.marginal_gain,
                        normalized_target=(edge.marginal_gain / baseline if eligible else None),
                        regression_weight=regression_weight,
                    )
                )
                by_group[(k, coalition)].append(index)
            for (k, _), indices in by_group.items():
                group_weight = (
                    source_ranking_weight
                    / state_counts[state.source_id]
                    / 2
                    / math.comb(len(event_ids), k)
                )
                ranking_groups.append((indices, group_weight))
        else:
            indices: list[int] = []
            for event_id in event_ids:
                empty_gain = state.table.distance(()) - state.table.distance((event_id,))
                singleton_gains = [
                    state.table.distance((other,))
                    - state.table.distance(tuple(sorted((other, event_id))))
                    for other in event_ids
                    if other != event_id
                ]
                raw_target = 0.5 * (
                    empty_gain + math.fsum(singleton_gains) / len(singleton_gains)
                )
                regression_weight = (
                    source_regression_weight
                    / eligible_counts[state.source_id]
                    / len(event_ids)
                    if eligible
                    else 0.0
                )
                indices.append(len(examples))
                examples.append(
                    GateExample(
                        source_id=state.source_id,
                        state_id=state.state_id,
                        decision_step_id=state.decision_step_id,
                        event_step_id=event_id,
                        coalition=(),
                        input_vector=independent_input(state, event_id),
                        raw_target=raw_target,
                        normalized_target=(raw_target / baseline if eligible else None),
                        regression_weight=regression_weight,
                    )
                )
            ranking_groups.append(
                (
                    indices,
                    source_ranking_weight / state_counts[state.source_id],
                )
            )

    pairs: list[RankingPair] = []
    for indices, group_weight in ranking_groups:
        untied: list[tuple[int, int, int]] = []
        for left, right in itertools.combinations(indices, 2):
            delta = examples[left].raw_target - examples[right].raw_target
            if abs(delta) <= LABEL_TIE_EPSILON:
                continue
            untied.append((left, right, 1 if delta > 0.0 else -1))
        if untied:
            pair_weight = group_weight / len(untied)
            pairs.extend(
                RankingPair(left, right, sign, pair_weight)
                for left, right, sign in untied
            )
    batch = TrainingBatch(family=family, examples=tuple(examples), ranking_pairs=tuple(pairs))
    if not math.isclose(batch.regression_weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("gate normalized-regression weights do not sum to one")
    return batch


def select_conditional(
    state: GateState,
    score,
    *,
    budget: int = DEPLOYMENT_BUDGET,
) -> tuple[tuple[int, ...], tuple[tuple[int, float], ...]]:
    selected: tuple[int, ...] = ()
    trace: list[tuple[int, float]] = []
    for _ in range(budget):
        candidates = [event for event in state.candidate_event_step_ids if event not in selected]
        scores = [(event, float(score(state, event, selected))) for event in candidates]
        if any(not math.isfinite(value) for _, value in scores):
            raise ValueError("gate prediction is non-finite")
        event, value = min(scores, key=lambda item: (-item[1], item[0]))
        if value <= 0.0:
            break
        selected = tuple(sorted((*selected, event)))
        trace.append((event, value))
    return selected, tuple(trace)


def select_independent(
    state: GateState,
    score,
    *,
    budget: int = DEPLOYMENT_BUDGET,
) -> tuple[int, ...]:
    scores = {
        event: float(score(state, event, ()))
        for event in state.candidate_event_step_ids
    }
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("gate prediction is non-finite")
    ranked = sorted(
        (event for event, value in scores.items() if value > 0.0),
        key=lambda event: (-scores[event], event),
    )
    return tuple(sorted(ranked[:budget]))


__all__ = [
    "CONDITIONAL_INPUT_DIMENSION",
    "CandidateFeatures",
    "DEPLOYMENT_BUDGET",
    "FeatureState",
    "GateExample",
    "GateState",
    "INDEPENDENT_INPUT_DIMENSION",
    "LabelState",
    "RankingPair",
    "TrainingBatch",
    "build_training_batch",
    "conditional_input",
    "feature_state_from_derived",
    "independent_input",
    "join_feature_and_label_states",
    "label_state_from_restoration_record",
    "normalize_text",
    "select_conditional",
    "select_independent",
    "signed_hash64",
    "validate_canonical_gate_state_roster",
]
