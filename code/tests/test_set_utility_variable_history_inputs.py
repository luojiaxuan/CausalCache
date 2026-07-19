from __future__ import annotations

import json

from causalcache.set_utility_variable_history_inputs import (
    build_variable_history_prompt_plan,
    build_variable_history_queries,
    build_variable_history_queries_from_source_row,
    trajectory_from_source_row,
)


def _row(decision_count: int = 8) -> dict[str, object]:
    trajectory_id = "trajectory"
    history = [
        {
            "event_step_id": step,
            "high_fidelity_observation_ref": (
                f"images/{trajectory_id}/observation-{step:03d}.png"
            ),
            "low_fidelity_summary": {
                "action_argument": "wait",
                "action_type": "wait",
                "executor_result": "unknown",
                "foreground_app": "unknown",
                "screen_change": "none",
                "screen_text_added": [],
                "screen_text_removed": [],
                "step_id": step,
            },
        }
        for step in range(1, decision_count + 1)
    ]
    ocr = {
        f"images/{trajectory_id}/observation-{step:03d}.png": {}
        for step in range(decision_count + 1)
    }
    return {
        "history_events_json": json.dumps(history),
        "images": [{"bytes": b"png", "path": None} for _ in range(decision_count + 1)],
        "ocr_records_json": json.dumps(ocr),
        "role": "train",
        "source_id": trajectory_id,
        "task_instruction": "do the task",
    }


def test_queries_use_every_prior_event_and_variable_candidate_count() -> None:
    trajectory = trajectory_from_source_row(_row())
    queries = build_variable_history_queries(trajectory)
    assert [query.decision_step_id for query in queries] == [6, 7, 8, 9]
    assert [len(query.candidate_event_step_ids) for query in queries] == [5, 6, 7, 8]
    assert queries[-1].candidate_event_step_ids == tuple(range(1, 9))
    metadata_queries = build_variable_history_queries_from_source_row(
        {**_row(), "decision_count": 8}
    )
    assert metadata_queries == queries


def test_prompt_restores_only_selected_events_but_keeps_full_summary_history() -> None:
    trajectory = trajectory_from_source_row(_row())
    query = build_variable_history_queries(trajectory)[-1]
    plan = build_variable_history_prompt_plan(query, (1, 4, 8))
    assert len(plan.history_events) == 8
    assert plan.summary_event_step_ids == tuple(range(1, 9))
    assert plan.high_fidelity_history_event_step_ids == (1, 4, 8)
    assert plan.current_observation_ref.endswith("observation-008.png")
