from __future__ import annotations

import hashlib
import math

import pytest

from causalcache.exploratory_closed_loop_memory import (
    EXPLORATORY_FIVE_ARMS,
    EXPLORATORY_MEMORY_BUDGET,
    build_live_gui_owl_v2_1_mixed_fidelity_messages,
    candidate_event_step_ids_from_history,
    deterministic_five_arm_order,
    make_mean_ensemble_vector_scorer,
    mean_ensemble_vector_scores,
    select_conditional_gate_memory,
    select_independent_gate_memory,
    select_ocr_rgb_memory,
    select_recent_memory,
    select_summary_memory,
)
from causalcache.gate_v1_data import CandidateFeatures, FeatureState
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
from causalcache.restoration_v2_baselines import RESIZED_RGB_BYTE_COUNT


def _low(step_id: int) -> dict[str, object]:
    return {
        "step_id": step_id,
        "action_type": "click",
        "action_argument": f"coordinate_bin:x{step_id % 10}_y0",
        "foreground_app": "unknown",
        "screen_text_added": [f"added-{step_id}"],
        "screen_text_removed": [],
        "screen_change": "medium",
        "executor_result": "unknown",
    }


def _history_event(step_id: int, *, with_binding: bool = False) -> dict[str, object]:
    event: dict[str, object] = {
        "low_fidelity_v2": _low(step_id),
        "post_ocr_spatial_tokens": ["screen", str(step_id)],
    }
    if with_binding:
        low = LowFidelityEventV2.from_mapping(event["low_fidelity_v2"])
        serialized = serialize_low_fidelity_v2(low)
        event["low_fidelity_v2_serialized"] = serialized.decode("utf-8")
        event["low_fidelity_v2_sha256"] = sha256_bytes(serialized)
    return event


def _feature_state(candidate_ids: tuple[int, ...]) -> FeatureState:
    candidates = []
    for index, step_id in enumerate(candidate_ids):
        h64 = [0.0] * 64
        h64[index % 64] = 1.0
        candidates.append(
            CandidateFeatures(
                event_step_id=step_id,
                h64=tuple(h64),
                g8=(0.0,) * 8,
            )
        )
    return FeatureState(
        source_id="source",
        state_id="source:live",
        decision_step_id=len(candidate_ids) + 2,
        candidate_event_step_ids=candidate_ids,
        q64=(0.0,) * 64,
        candidates=tuple(candidates),
    )


def _solid_rgb(red: int, green: int, blue: int) -> bytes:
    return bytes((red, green, blue)) * (RESIZED_RGB_BYTE_COUNT // 3)


def test_live_prompt_supports_empty_and_arbitrary_history() -> None:
    current = object()
    empty = build_live_gui_owl_v2_1_mixed_fidelity_messages(
        instruction="Open Settings",
        history_events=(),
        restored_event_step_ids=(),
        selected_post_images_by_event_step={},
        current_image=current,
    )
    assert empty[0] == {
        "role": "system",
        "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
    }
    assert validate_gui_owl_v2_1_native_messages(empty) == 1
    assert empty[1]["content"][-2] == {"type": "image", "image": current}

    history = tuple(_history_event(step, with_binding=True) for step in range(1, 9))
    post_2 = object()
    post_7 = object()
    messages = build_live_gui_owl_v2_1_mixed_fidelity_messages(
        instruction="Complete a long task",
        history_events=history,
        restored_event_step_ids=(2, 7),
        selected_post_images_by_event_step={2: post_2, 7: post_7},
        current_image=current,
    )
    content = messages[1]["content"]
    summaries = [block for block in content if block.get("type") == "text" and block["text"].startswith("Event summary:")]
    images = [block["image"] for block in content if block.get("type") == "image"]
    assert len(summaries) == 8
    assert images == [post_2, post_7, current]
    assert summaries[-1]["text"].endswith(serialize_low_fidelity_v2(LowFidelityEventV2.from_mapping(_low(8))).decode("utf-8"))
    assert content[-1] == {
        "type": "text",
        "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    }
    assert validate_gui_owl_v2_1_native_messages(messages) == 3
    assert candidate_event_step_ids_from_history(history) == tuple(range(1, 8))


def test_live_prompt_fails_closed_on_current_equivalent_and_summary_drift() -> None:
    history = [_history_event(1), _history_event(2, with_binding=True)]
    with pytest.raises(ValueError, match="current-equivalent"):
        build_live_gui_owl_v2_1_mixed_fidelity_messages(
            instruction="Task",
            history_events=history,
            restored_event_step_ids=(2,),
            selected_post_images_by_event_step={2: object()},
            current_image=object(),
        )
    history[1]["low_fidelity_v2_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 drifted"):
        build_live_gui_owl_v2_1_mixed_fidelity_messages(
            instruction="Task",
            history_events=history,
            restored_event_step_ids=(),
            selected_post_images_by_event_step={},
            current_image=object(),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        candidate_event_step_ids_from_history((_history_event(2), _history_event(1)))


def test_dynamic_summary_recent_and_ocr_rgb_cover_empty_and_long_history() -> None:
    assert EXPLORATORY_MEMORY_BUDGET == 2
    assert select_summary_memory(()) == ()
    assert select_recent_memory(()) == ()
    assert select_recent_memory((2,)) == (2,)
    assert select_recent_memory((2, 5, 9, 14, 20)) == (14, 20)
    assert select_ocr_rgb_memory(
        event_step_ids=(),
        event_ocr_tokens={},
        current_ocr_tokens=object(),
        event_resized_rgb_bytes={},
        current_resized_rgb_bytes=object(),
    ).selected_event_step_ids == ()

    red = _solid_rgb(255, 0, 0)
    candidates = (2, 5, 9, 14, 20)
    result = select_ocr_rgb_memory(
        event_step_ids=candidates,
        event_ocr_tokens={step: ["Settings"] for step in candidates},
        current_ocr_tokens=["Settings"],
        event_resized_rgb_bytes={step: red for step in candidates},
        current_resized_rgb_bytes=red,
    )
    assert result.ranked_event_step_ids == candidates
    assert result.selected_event_step_ids == (2, 5)
    assert len(result.scores_by_event_step) == len(candidates)


def test_ocr_rgb_dynamic_mapping_and_numeric_contract_fail_closed() -> None:
    black = _solid_rgb(0, 0, 0)
    with pytest.raises(ValueError, match=r"missing=\[3\]"):
        select_ocr_rgb_memory(
            event_step_ids=(1, 3),
            event_ocr_tokens={1: []},
            current_ocr_tokens=[],
            event_resized_rgb_bytes={1: black, 3: black},
            current_resized_rgb_bytes=black,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        select_recent_memory((3, 1))


def test_independent_gate_is_empty_safe_positive_only_and_step_tied() -> None:
    calls = []

    def must_not_run(_vectors):
        raise AssertionError("empty history must not invoke the scorer")

    assert select_independent_gate_memory(
        _feature_state(()),
        score_vectors=must_not_run,
    ).selected_event_step_ids == ()

    def score(vectors):
        calls.append(tuple(len(vector) for vector in vectors))
        return (0.0, 0.7, 0.7, -1.0)

    result = select_independent_gate_memory(
        _feature_state((2, 5, 9, 14)),
        score_vectors=score,
    )
    assert calls == [(200, 200, 200, 200)]
    assert result.selected_event_step_ids == (5, 9)
    assert result.selection_trace == ((5, 0.7), (9, 0.7))
    assert result.score_rounds == (((2, 0.0), (5, 0.7), (9, 0.7), (14, -1.0)),)


def test_conditional_gate_rescores_and_is_empty_safe_positive_only() -> None:
    assert select_conditional_gate_memory(
        _feature_state(()),
        score_vectors=lambda _: (_ for _ in ()).throw(AssertionError()),
    ).score_rounds == ()
    calls = []

    def score(vectors):
        calls.append(tuple(len(vector) for vector in vectors))
        if len(calls) == 1:
            return (0.3, 0.3, -1.0)
        return (0.0, 0.2)

    result = select_conditional_gate_memory(
        _feature_state((2, 5, 9)),
        score_vectors=score,
    )
    assert calls == [(330, 330, 330), (330, 330)]
    assert result.selected_event_step_ids == (2, 9)
    assert result.selection_trace == ((2, 0.3), (9, 0.2))
    assert result.score_rounds == (
        ((2, 0.3), (5, 0.3), (9, -1.0)),
        ((5, 0.0), (9, 0.2)),
    )

    stopped = select_conditional_gate_memory(
        _feature_state((1, 2)),
        score_vectors=lambda vectors: (0.0,) * len(vectors),
    )
    assert stopped.selected_event_step_ids == ()
    assert len(stopped.score_rounds) == 1


def test_gate_scorers_reject_nonfinite_or_wrong_length_outputs() -> None:
    state = _feature_state((1, 2))
    with pytest.raises(ValueError, match="wrong score count"):
        select_independent_gate_memory(state, score_vectors=lambda _: (1.0,))
    with pytest.raises(ValueError, match="finite"):
        select_conditional_gate_memory(
            state,
            score_vectors=lambda vectors: (math.nan,) * len(vectors),
        )


def test_injected_mean_ensemble_vector_scorer_averages_models_and_skips_empty() -> None:
    calls = []

    def predict(model, vectors):
        calls.append((model, len(vectors)))
        return tuple(model * sum(vector) for vector in vectors)

    models = (1.0, 2.0, 3.0, 4.0, 5.0)
    vectors = ((1.0, 2.0), (4.0,))
    assert mean_ensemble_vector_scores(
        models,
        vectors,
        model_batch_predictor=predict,
    ) == (9.0, 12.0)
    assert calls == [(model, 2) for model in models]
    scorer = make_mean_ensemble_vector_scorer(
        models,
        model_batch_predictor=predict,
    )
    calls.clear()
    assert tuple(scorer(vectors)) == (9.0, 12.0)
    assert len(calls) == 5
    calls.clear()
    assert mean_ensemble_vector_scores(
        models,
        (),
        model_batch_predictor=predict,
    ) == ()
    assert calls == []


def test_five_arm_order_is_exact_deterministic_hash_rotation() -> None:
    assert EXPLORATORY_FIVE_ARMS == (
        "independent_B2",
        "recent_B2",
        "ocr_rgb_B2",
        "conditional_B2",
        "summary_B0",
    )
    arguments = {
        "protocol_id": "causalcache_exploratory_closed_loop_validation12_v1",
        "split": "validation",
        "suite_seed": 271828,
        "task_type": "RetroSavePlaylist",
        "task_index": 0,
    }
    key = "\0".join(
        (
            arguments["protocol_id"],
            arguments["split"],
            str(arguments["suite_seed"]),
            arguments["task_type"],
            str(arguments["task_index"]),
        )
    ).encode("utf-8")
    offset = hashlib.sha256(key).digest()[0] % 5
    expected = EXPLORATORY_FIVE_ARMS[offset:] + EXPLORATORY_FIVE_ARMS[:offset]
    assert deterministic_five_arm_order(**arguments) == expected
    assert deterministic_five_arm_order(**arguments) == expected
    assert len(set(expected)) == 5
    assert set(expected) == set(EXPLORATORY_FIVE_ARMS)

    observed = {
        deterministic_five_arm_order(
            protocol_id=arguments["protocol_id"],
            split="validation",
            suite_seed=271828,
            task_type=f"Task{index}",
            task_index=0,
        )
        for index in range(100)
    }
    assert len(observed) == 5
    with pytest.raises(ValueError, match="task_index"):
        deterministic_five_arm_order(**{**arguments, "task_index": -1})
