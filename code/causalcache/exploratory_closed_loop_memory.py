"""Pure online-memory primitives for exploratory AndroidWorld closed loop."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cmp_to_key
from numbers import Real
from typing import Any

from causalcache.gate_v1_data import (
    CONDITIONAL_INPUT_DIMENSION,
    DEPLOYMENT_BUDGET,
    INDEPENDENT_INPUT_DIMENSION,
    CandidateFeatures,
    FeatureState,
    conditional_input,
    independent_input,
)
from causalcache.low_fidelity_v2 import (
    LowFidelityEventV2,
    serialize_low_fidelity_v2,
    sha256_bytes,
)
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    validate_gui_owl_v2_1_native_messages,
)
from causalcache.restoration_v2_baselines import (
    OCR_JACCARD_WEIGHT,
    RGB_HISTOGRAM_WEIGHT,
    SCORE_TIE_ABS_TOL,
    SCORE_TIE_REL_TOL,
    BaselineSelection,
    joint_rgb_histogram_cosine,
    ocr_token_set_jaccard,
)


EXPLORATORY_MEMORY_BUDGET = DEPLOYMENT_BUDGET
INDEPENDENT_ARM = "independent_B2"
CONDITIONAL_ARM = "conditional_B2"
RECENT_ARM = "recent_B2"
OCR_RGB_ARM = "ocr_rgb_B2"
SUMMARY_ARM = "summary_B0"
EXPLORATORY_FIVE_ARMS = (
    INDEPENDENT_ARM,
    RECENT_ARM,
    OCR_RGB_ARM,
    CONDITIONAL_ARM,
    SUMMARY_ARM,
)


VectorBatchScorer = Callable[
    [tuple[tuple[float, ...], ...]],
    Iterable[Real],
]
ModelBatchPredictor = Callable[
    [Any, tuple[tuple[float, ...], ...]],
    Iterable[Real],
]


@dataclass(frozen=True)
class GateMemorySelection:
    """Selected event ids plus all score rounds needed for online auditing."""

    selected_event_step_ids: tuple[int, ...]
    selection_trace: tuple[tuple[int, float], ...]
    score_rounds: tuple[tuple[tuple[int, float], ...], ...]


class _FeatureStateView:
    """Give a label-free ``FeatureState`` the frozen gate input interface."""

    def __init__(self, state: FeatureState) -> None:
        self.source_id = state.source_id
        self.state_id = state.state_id
        self.decision_step_id = state.decision_step_id
        self.candidate_event_step_ids = state.candidate_event_step_ids
        self.q64 = state.q64
        self.candidates = state.candidates

    def candidate(self, event_step_id: int) -> CandidateFeatures:
        for candidate in self.candidates:
            if candidate.event_step_id == event_step_id:
                return candidate
        raise KeyError(event_step_id)


def _ordered_event_step_ids(event_step_ids: Iterable[int]) -> tuple[int, ...]:
    if isinstance(event_step_ids, (str, bytes, bytearray, Mapping)):
        raise TypeError("event step ids must be an ordered iterable")
    try:
        result = tuple(event_step_ids)
    except TypeError as error:
        raise TypeError("event step ids must be an ordered iterable") from error
    if any(type(step_id) is not int for step_id in result):
        raise TypeError("event step ids must be integers")
    if any(step_id <= 0 for step_id in result):
        raise ValueError("event step ids must be positive")
    if len(set(result)) != len(result) or tuple(sorted(result)) != result:
        raise ValueError("event step ids must be strictly increasing and unique")
    return result


def candidate_event_step_ids_from_history(
    history_events: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    """Return every history id except the newest current-equivalent event."""
    parsed = _canonical_history_summaries(history_events)
    return tuple(summary.step_id for summary, _ in parsed[:-1])


def _canonical_history_summaries(
    history_events: Sequence[Mapping[str, Any]],
) -> tuple[tuple[LowFidelityEventV2, str], ...]:
    if (
        isinstance(history_events, (str, bytes, bytearray, Mapping))
        or not isinstance(history_events, Sequence)
    ):
        raise TypeError("live history must be an ordered event sequence")
    parsed: list[tuple[LowFidelityEventV2, str]] = []
    for index, event in enumerate(history_events):
        if not isinstance(event, Mapping):
            raise TypeError(f"live history event {index} must be a mapping")
        try:
            summary = LowFidelityEventV2.from_mapping(event["low_fidelity_v2"])
        except KeyError as error:
            raise ValueError("live history event is missing low_fidelity_v2") from error
        except (TypeError, ValueError) as error:
            raise ValueError("live history low_fidelity_v2 is not canonical") from error
        serialized_bytes = serialize_low_fidelity_v2(summary)
        serialized = serialized_bytes.decode("utf-8")
        if (
            "low_fidelity_v2_serialized" in event
            and event["low_fidelity_v2_serialized"] != serialized
        ):
            raise ValueError("stored live summary serialization drifted")
        if (
            "low_fidelity_v2_sha256" in event
            and event["low_fidelity_v2_sha256"] != sha256_bytes(serialized_bytes)
        ):
            raise ValueError("stored live summary SHA256 drifted")
        parsed.append((summary, serialized))
    _ordered_event_step_ids(summary.step_id for summary, _ in parsed)
    return tuple(parsed)


def _exact_candidate_mapping(
    value: Mapping[int, Any],
    *,
    candidate_ids: tuple[int, ...],
    label: str,
) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(type(step_id) is not int for step_id in value):
        raise TypeError(f"{label} keys must be integer event step ids")
    expected = set(candidate_ids)
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} must cover exactly {candidate_ids}; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return {step_id: value[step_id] for step_id in candidate_ids}


def build_live_gui_owl_v2_1_mixed_fidelity_messages(
    *,
    instruction: str,
    history_events: Sequence[Mapping[str, Any]],
    restored_event_step_ids: Iterable[int],
    selected_post_images_by_event_step: Mapping[int, Any],
    current_image: Any,
) -> list[dict[str, Any]]:
    """Build arbitrary-history v2.1 input without performing image or model I/O."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("live instruction must be non-empty text")
    parsed = _canonical_history_summaries(history_events)
    history_ids = tuple(summary.step_id for summary, _ in parsed)
    candidate_ids = history_ids[:-1]
    restored_ids = _ordered_event_step_ids(restored_event_step_ids)
    if not set(restored_ids).issubset(candidate_ids):
        raise ValueError(
            "restored events must be older than the current-equivalent latest event"
        )
    post_images = _exact_candidate_mapping(
        selected_post_images_by_event_step,
        candidate_ids=restored_ids,
        label="selected post images",
    )
    if current_image is None or any(image is None for image in post_images.values()):
        raise ValueError("current and selected post images cannot be None")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Please generate the next move from the task, event summaries, restored "
                f"post-action states, and current observation.\n\nInstruction: {instruction}"
            ),
        }
    ]
    restored = set(restored_ids)
    for summary, serialized in parsed:
        content.append({"type": "text", "text": f"Event summary:\n{serialized}"})
        if summary.step_id in restored:
            content.append(
                {
                    "type": "image",
                    "image": post_images[summary.step_id],
                }
            )
    content.extend(
        (
            {"type": "text", "text": "Current observation:"},
            {"type": "image", "image": current_image},
            {
                "type": "text",
                "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
            },
        )
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]
    validate_gui_owl_v2_1_native_messages(messages)
    return messages


def select_summary_memory(event_step_ids: Iterable[int]) -> tuple[()]:
    """Keep all summaries and restore no post-state image."""
    _ordered_event_step_ids(event_step_ids)
    return ()


def select_recent_memory(event_step_ids: Iterable[int]) -> tuple[int, ...]:
    """Restore the latest ``min(B, n)`` eligible post-state images."""
    candidates = _ordered_event_step_ids(event_step_ids)
    return candidates[-min(EXPLORATORY_MEMORY_BUDGET, len(candidates)) :]


def _similarity_selection(
    scores_by_event_step: Mapping[int, Real],
    *,
    candidate_ids: tuple[int, ...],
) -> BaselineSelection:
    raw = _exact_candidate_mapping(
        scores_by_event_step,
        candidate_ids=candidate_ids,
        label="scores_by_event_step",
    )
    scores: dict[int, float] = {}
    for step_id in candidate_ids:
        value = raw[step_id]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"score for event {step_id} must be a real number")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"score for event {step_id} must be finite")
        scores[step_id] = score

    def compare(left: int, right: int) -> int:
        if math.isclose(
            scores[left],
            scores[right],
            rel_tol=SCORE_TIE_REL_TOL,
            abs_tol=SCORE_TIE_ABS_TOL,
        ):
            return -1 if left < right else 1
        return -1 if scores[left] > scores[right] else 1

    ranking = tuple(sorted(candidate_ids, key=cmp_to_key(compare)))
    selected_count = min(EXPLORATORY_MEMORY_BUDGET, len(candidate_ids))
    return BaselineSelection(
        scores_by_event_step=tuple((step_id, scores[step_id]) for step_id in candidate_ids),
        ranked_event_step_ids=ranking,
        selected_event_step_ids=tuple(sorted(ranking[:selected_count])),
    )


def select_ocr_rgb_memory(
    *,
    event_step_ids: Iterable[int],
    event_ocr_tokens: Mapping[int, Iterable[str]],
    current_ocr_tokens: Iterable[str],
    event_resized_rgb_bytes: Mapping[int, bytes | bytearray | memoryview],
    current_resized_rgb_bytes: bytes | bytearray | memoryview,
) -> BaselineSelection:
    """Apply the frozen equal-weight OCR/RGB heuristic to arbitrary history."""
    candidates = _ordered_event_step_ids(event_step_ids)
    if not candidates:
        return BaselineSelection((), (), ())
    ocr_by_event = _exact_candidate_mapping(
        event_ocr_tokens,
        candidate_ids=candidates,
        label="event OCR tokens",
    )
    rgb_by_event = _exact_candidate_mapping(
        event_resized_rgb_bytes,
        candidate_ids=candidates,
        label="event resized RGB bytes",
    )
    scores = {
        step_id: (
            OCR_JACCARD_WEIGHT
            * ocr_token_set_jaccard(ocr_by_event[step_id], current_ocr_tokens)
            + RGB_HISTOGRAM_WEIGHT
            * joint_rgb_histogram_cosine(
                rgb_by_event[step_id],
                current_resized_rgb_bytes,
            )
        )
        for step_id in candidates
    }
    return _similarity_selection(scores, candidate_ids=candidates)


def _validated_feature_view(state: FeatureState) -> _FeatureStateView:
    if not isinstance(state, FeatureState):
        raise TypeError("gate memory selection requires one FeatureState")
    candidate_ids = _ordered_event_step_ids(state.candidate_event_step_ids)
    observed = tuple(candidate.event_step_id for candidate in state.candidates)
    if observed != candidate_ids:
        raise ValueError("FeatureState candidate inventory or order drifted")
    return _FeatureStateView(state)


def _score_vector_batch(
    score_vectors: VectorBatchScorer,
    vectors: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    if not callable(score_vectors):
        raise TypeError("score_vectors must be callable")
    if not vectors:
        return ()
    raw = score_vectors(vectors)
    if isinstance(raw, (str, bytes, bytearray, Mapping)):
        raise TypeError("vector scorer must return an ordered score iterable")
    try:
        values = tuple(raw)
    except TypeError as error:
        raise TypeError("vector scorer must return an ordered score iterable") from error
    if len(values) != len(vectors):
        raise ValueError("vector scorer returned the wrong score count")
    scores: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("vector scores must be real numbers")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("vector scores must be finite")
        scores.append(score)
    return tuple(scores)


def mean_ensemble_vector_scores(
    models: Sequence[Any],
    vectors: Sequence[Sequence[float]],
    *,
    model_batch_predictor: ModelBatchPredictor,
) -> tuple[float, ...]:
    """Average injected per-model predictions without importing a model runtime."""
    if isinstance(models, (str, bytes, bytearray, Mapping)) or not isinstance(
        models, Sequence
    ):
        raise TypeError("ensemble models must be an ordered sequence")
    ensemble = tuple(models)
    if not ensemble:
        raise ValueError("ensemble must contain at least one model")
    if not callable(model_batch_predictor):
        raise TypeError("model_batch_predictor must be callable")
    vector_batch = tuple(tuple(float(value) for value in vector) for vector in vectors)
    if not vector_batch:
        return ()
    predictions = tuple(
        _score_vector_batch(
            lambda batch, model=model: model_batch_predictor(model, batch),
            vector_batch,
        )
        for model in ensemble
    )
    return tuple(
        math.fsum(model_scores[index] for model_scores in predictions)
        / len(predictions)
        for index in range(len(vector_batch))
    )


def make_mean_ensemble_vector_scorer(
    models: Sequence[Any],
    *,
    model_batch_predictor: ModelBatchPredictor,
) -> VectorBatchScorer:
    """Bind loaded models to the pure mean-ensemble vector interface."""
    ensemble = tuple(models)
    if not ensemble:
        raise ValueError("ensemble must contain at least one model")

    def score(vectors: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
        return mean_ensemble_vector_scores(
            ensemble,
            vectors,
            model_batch_predictor=model_batch_predictor,
        )

    return score


def select_independent_gate_memory(
    state: FeatureState,
    *,
    score_vectors: VectorBatchScorer,
) -> GateMemorySelection:
    """Run frozen independent tau=0, B=2 selection on one live feature state."""
    view = _validated_feature_view(state)
    candidates = view.candidate_event_step_ids
    if not candidates:
        return GateMemorySelection((), (), ())
    vectors = tuple(independent_input(view, step_id) for step_id in candidates)
    if any(len(vector) != INDEPENDENT_INPUT_DIMENSION for vector in vectors):
        raise RuntimeError("independent gate input dimension drifted")
    scores = _score_vector_batch(score_vectors, vectors)
    score_by_event = dict(zip(candidates, scores, strict=True))
    ranking = tuple(
        sorted(
            (
                step_id
                for step_id in candidates
                if score_by_event[step_id] > 0.0
            ),
            key=lambda step_id: (-score_by_event[step_id], step_id),
        )
    )
    selected_ranked = ranking[:EXPLORATORY_MEMORY_BUDGET]
    return GateMemorySelection(
        selected_event_step_ids=tuple(sorted(selected_ranked)),
        selection_trace=tuple(
            (step_id, score_by_event[step_id]) for step_id in selected_ranked
        ),
        score_rounds=(tuple(zip(candidates, scores, strict=True)),),
    )


def select_conditional_gate_memory(
    state: FeatureState,
    *,
    score_vectors: VectorBatchScorer,
) -> GateMemorySelection:
    """Run frozen iterative conditional tau=0, B=2 selection and rescore."""
    view = _validated_feature_view(state)
    if not view.candidate_event_step_ids:
        return GateMemorySelection((), (), ())
    selected: tuple[int, ...] = ()
    trace: list[tuple[int, float]] = []
    rounds: list[tuple[tuple[int, float], ...]] = []
    for _ in range(EXPLORATORY_MEMORY_BUDGET):
        candidates = tuple(
            step_id
            for step_id in view.candidate_event_step_ids
            if step_id not in selected
        )
        if not candidates:
            break
        vectors = tuple(
            conditional_input(
                view,
                step_id,
                selected,
                budget=EXPLORATORY_MEMORY_BUDGET,
            )
            for step_id in candidates
        )
        if any(len(vector) != CONDITIONAL_INPUT_DIMENSION for vector in vectors):
            raise RuntimeError("conditional gate input dimension drifted")
        scores = _score_vector_batch(score_vectors, vectors)
        round_scores = tuple(zip(candidates, scores, strict=True))
        rounds.append(round_scores)
        chosen, value = min(round_scores, key=lambda item: (-item[1], item[0]))
        if value <= 0.0:
            break
        selected = tuple(sorted((*selected, chosen)))
        trace.append((chosen, value))
    return GateMemorySelection(
        selected_event_step_ids=selected,
        selection_trace=tuple(trace),
        score_rounds=tuple(rounds),
    )


def deterministic_five_arm_order(
    *,
    protocol_id: str,
    split: str,
    suite_seed: int,
    task_type: str,
    task_index: int,
) -> tuple[str, ...]:
    """Rotate the frozen five-arm order by a deterministic instance hash."""
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("protocol_id must be non-empty text")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be non-empty text")
    if not isinstance(task_type, str) or not task_type:
        raise ValueError("task_type must be non-empty text")
    if type(suite_seed) is not int or suite_seed < 0:
        raise ValueError("suite_seed must be a non-negative integer")
    if type(task_index) is not int or task_index < 0:
        raise ValueError("task_index must be a non-negative integer")
    key = "\0".join(
        (protocol_id, split, str(suite_seed), task_type, str(task_index))
    ).encode("utf-8")
    offset = hashlib.sha256(key).digest()[0] % len(EXPLORATORY_FIVE_ARMS)
    return EXPLORATORY_FIVE_ARMS[offset:] + EXPLORATORY_FIVE_ARMS[:offset]


__all__ = [
    "CONDITIONAL_ARM",
    "EXPLORATORY_FIVE_ARMS",
    "EXPLORATORY_MEMORY_BUDGET",
    "GateMemorySelection",
    "INDEPENDENT_ARM",
    "OCR_RGB_ARM",
    "RECENT_ARM",
    "SUMMARY_ARM",
    "build_live_gui_owl_v2_1_mixed_fidelity_messages",
    "candidate_event_step_ids_from_history",
    "deterministic_five_arm_order",
    "make_mean_ensemble_vector_scorer",
    "mean_ensemble_vector_scores",
    "select_conditional_gate_memory",
    "select_independent_gate_memory",
    "select_ocr_rgb_memory",
    "select_recent_memory",
    "select_summary_memory",
]
