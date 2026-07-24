"""Longest-real-state inventory contract for HGKV selector V2."""

from __future__ import annotations

import json

from scripts.build_hgkv_selector_v2_state_inventory import (
    inventory_audit,
    select_longest_real_state,
)


def _event(step_id: int) -> dict[str, object]:
    return {
        "event_step_id": step_id,
        "low_fidelity_summary": {
            "step_id": step_id,
            "action_type": "click",
            "action_argument": "coordinate_bin:x5_y5",
            "foreground_app": "unknown",
            "screen_text_added": [],
            "screen_text_removed": [],
            "screen_change": "low",
            "executor_result": "unknown",
        },
    }


def _trajectory(event_count: int = 12):
    source_id = "trajectory-a"
    row = {
        "source_id": source_id,
        "history_events_json": json.dumps(
            [_event(step_id) for step_id in range(1, event_count + 1)]
        ),
        "raw_metadata": json.dumps({}),
    }
    steps = [
        {"step": index, "action": "CLICK", "info": [[500.0, 600.0]]}
        for index in range(event_count)
    ]
    steps.append({"step": event_count, "action": "COMPLETE", "info": None})
    annotation = {
        "device_info": {"device_resolution": [1080, 2400]},
        "steps": steps,
    }
    return row, annotation


def test_selects_longest_real_decision_and_excludes_terminal():
    row, annotation = _trajectory()
    selected, counters = select_longest_real_state(
        row,
        annotation,
        [4, 8, 12, 13],
        split="train",
    )
    assert selected is not None
    assert selected["decision_step"] == 12
    assert selected["candidate_event_step_ids"] == list(range(1, 11))
    assert selected["current_equivalent_event_step_id"] == 11
    assert selected["candidate_count"] == 10
    assert selected["recent_candidate_cap"] is None
    assert selected["synthetic_terminal"] is False
    assert counters["synthetic_terminal"] == 1


def test_inventory_audit_reports_full_history_tail():
    rows = [
        {
            "pair_group": "a:4",
            "candidate_count": 2,
            "history_length": 3,
            "action_type": "click",
            "app": "unknown",
            "synthetic_terminal": False,
            "recent_candidate_cap": None,
        },
        {
            "pair_group": "b:12",
            "candidate_count": 10,
            "history_length": 11,
            "action_type": "text",
            "app": "unknown",
            "synthetic_terminal": False,
            "recent_candidate_cap": None,
        },
    ]
    audit = inventory_audit(
        rows,
        trajectories_seen=2,
        trajectories_successful=2,
        synthetic_terminal_excluded=0,
        ineligible_decisions=0,
    )
    assert audit["states"] == 2
    assert audit["candidate_count"]["max"] == 10
    assert audit["states_with_candidate_count_gt_8"] == 1
    assert audit["recent_8_truncation_count"] == 0
    assert audit["synthetic_terminal_count"] == 0
