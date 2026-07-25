from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from causalcache.mobileworld import (
    MobileWorldHistoryEvent,
    build_mobileworld_policy_request,
    mobileworld_action_from_gui_owl,
    select_mobileworld_memory,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "data/manifests/mobileworld_memory_split_v1.json"
CONFIG_PATH = ROOT / "code/configs/causalcache_mobileworld_memory_v1.json"


def _history(count: int) -> tuple[MobileWorldHistoryEvent, ...]:
    return tuple(
        MobileWorldHistoryEvent(
            step_id=index,
            action={"action_type": "wait"},
            screen_changed=index % 2 == 0,
            post_screenshot=f"png-{index}".encode(),
        )
        for index in range(1, count + 1)
    )


def test_mobileworld_memory_selection_is_at_most_budget() -> None:
    history = _history(5)
    assert select_mobileworld_memory(history, arm="summary", budget=4) == ()
    assert select_mobileworld_memory(history, arm="recent", budget=4) == (
        2,
        3,
        4,
        5,
    )
    assert select_mobileworld_memory(history, arm="full", budget=1) == (
        1,
        2,
        3,
        4,
        5,
    )


def test_mobileworld_policy_request_restores_only_selected_images() -> None:
    request = build_mobileworld_policy_request(
        instruction="Do the task",
        step_id=4,
        screenshot=b"current",
        history=_history(3),
        selected_event_step_ids=(2, 3),
        screen_size=(432, 768),
    )
    assert request["history"][0]["restored_post_screenshot_png_base64"] is None
    assert base64.b64decode(
        request["history"][1]["restored_post_screenshot_png_base64"]
    ) == b"png-2"
    with pytest.raises(ValueError, match="unknown"):
        build_mobileworld_policy_request(
            instruction="Do the task",
            step_id=4,
            screenshot=b"current",
            history=_history(3),
            selected_event_step_ids=(4,),
            screen_size=(432, 768),
        )


def test_gui_owl_actions_map_to_mobileworld_schema() -> None:
    assert mobileworld_action_from_gui_owl(
        GUIOwlV2Action(action="click", coordinate=(999, 0)),
        screen_size=(100, 200),
    ) == {"action_type": "click", "x": 99, "y": 0}
    assert mobileworld_action_from_gui_owl(
        GUIOwlV2Action(
            action="swipe",
            coordinate=(0, 0),
            coordinate2=(999, 999),
        ),
        screen_size=(100, 200),
    ) == {
        "action_type": "drag",
        "start_x": 0,
        "start_y": 0,
        "end_x": 99,
        "end_y": 199,
    }
    assert mobileworld_action_from_gui_owl(
        GUIOwlV2Action(action="terminate", status="success"),
        screen_size=(100, 200),
    ) == {"action_type": "finished", "text": "success"}


def test_mobileworld_manifest_and_config_lock_gui_only_denominator() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert plan["upstream"]["revision"] == config["upstream"]["revision"]
    assert plan["counts"] == {
        "all_cross_app_tasks": 125,
        "all_tasks": 201,
        "gui_only_cross_app_memory_candidates": 62,
        "gui_only_single_app_controls": 55,
        "gui_only_tasks": 117,
        "interfaces": {
            "gui_only": 117,
            "mcp": 38,
            "mcp_and_user_interaction": 2,
            "user_interaction": 44,
        },
    }
    assert len(plan["benchmark_profiles"]["frozen_gui_owl_gui_only"]) == 117
    assert config["benchmark"]["expected_denominator"] == 117
    assert config["capacity"]["auto_retry"] == 2
    assert config["environment"]["task_source"] == (
        "pinned_upstream_src_read_only_mount"
    )
    assert config["environment"]["image"].endswith(
        "@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240"
    )
