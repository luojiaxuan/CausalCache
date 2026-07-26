from __future__ import annotations

from scripts.build_desktop_hgkv_corpus import (
    history_action_matches_target,
    select_contrast_events,
)


def tool(action: str, **kwargs):
    return {"name": "computer_use", "arguments": {"action": action, **kwargs}}


def test_full_click_equivalence_uses_coordinates() -> None:
    screen = (1920, 1080)
    assert history_action_matches_target(
        {"type": "click", "x": 960, "y": 540},
        tool("left_click", coordinate=[500, 500]),
        screen_size=screen,
        coordinate_tolerance=2,
    )
    assert not history_action_matches_target(
        {"type": "click", "x": 1300, "y": 540},
        tool("left_click", coordinate=[500, 500]),
        screen_size=screen,
        coordinate_tolerance=25,
    )


def test_text_and_key_equivalence_include_parameters() -> None:
    assert history_action_matches_target(
        {"type": "type_text", "text": "hello"},
        tool("type", text="hello"),
        screen_size=(100, 100),
        coordinate_tolerance=0,
    )
    assert not history_action_matches_target(
        {"type": "type_text", "text": "bye"},
        tool("type", text="hello"),
        screen_size=(100, 100),
        coordinate_tolerance=0,
    )
    assert history_action_matches_target(
        {"type": "hotkey", "keys": ["CTRL", "S"]},
        tool("key", keys=["ctrl", "s"]),
        screen_size=(100, 100),
        coordinate_tolerance=0,
    )


def test_selects_pre_state_and_age_matched_wrong_state() -> None:
    record = {
        "dp_id": "dp-1",
        "step": 8,
        "screen_size": [1000, 1000],
        "target_tool_call": tool("left_click", coordinate=[500, 500]),
        "history": [
            {"step_id": 1, "action": {"type": "click", "x": 10, "y": 10}},
            {"step_id": 2, "action": {"type": "click", "x": 500, "y": 500}},
            {"step_id": 3, "action": {"type": "press", "key": "tab"}},
            {"step_id": 4, "action": {"type": "click", "x": 50, "y": 50}},
            {"step_id": 5, "action": {"type": "click", "x": 500, "y": 500}},
            {"step_id": 6, "action": {"type": "press", "key": "enter"}},
            {"step_id": 7, "action": {"type": "click", "x": 80, "y": 80}},
        ],
    }
    selected = select_contrast_events(
        record, coordinate_tolerance=2, min_age=3, seed=7
    )
    assert selected is not None
    positive_event, positive_action_step, shuffled_event = selected
    assert (positive_event, positive_action_step) in {(1, 2), (4, 5)}
    assert shuffled_event != positive_event
    assert 8 - shuffled_event >= 3
