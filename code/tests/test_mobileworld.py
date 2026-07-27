from __future__ import annotations

import base64
import io
import json
import urllib.error
from pathlib import Path

import pytest

from scripts.run_mobileworld_gui_owl import (
    _load_task_subset,
    _task_names_sha256,
)
from scripts.serve_mobileworld_gui_owl_policy import _history_image_count
from causalcache.mobileworld import (
    HTTPMobileWorldPolicy,
    MobileWorldHistoryEvent,
    build_mobileworld_gui_owl_messages,
    build_mobileworld_policy_request,
    mobileworld_action_from_gui_owl,
    select_mobileworld_memory,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_official import (
    OFFICIAL_FIRST_USER_TEMPLATE,
    OFFICIAL_SYSTEM_PROMPT,
)
from causalcache.policy.gui_owl_official_runtime import interpret_official_output


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "data/manifests/mobileworld_memory_split_v1.json"
CONFIG_PATH = ROOT / "code/configs/causalcache_mobileworld_memory_v1.json"
OFFICIAL_B0_CONFIG_PATH = (
    ROOT / "code/configs/causalcache_mobileworld_official_b0_v2.json"
)
OFFICIAL_B4_CONFIG_PATH = (
    ROOT / "code/configs/causalcache_mobileworld_official_b4_v2.json"
)


def _history(count: int) -> tuple[MobileWorldHistoryEvent, ...]:
    return tuple(
        MobileWorldHistoryEvent(
            step_id=index,
            action={"action_type": "wait"},
            action_text=f"Wait at step {index}",
            full_response=(
                f"Action: Wait at step {index}\n"
                '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n'
                "</tool_call>"
            ),
            policy_parsed=True,
            screen_changed=index % 2 == 0,
            observation_screenshot=f"observation-{index}".encode(),
            post_screenshot=f"png-{index}".encode(),
        )
        for index in range(1, count + 1)
    )


def test_mobileworld_memory_selection_is_at_most_budget() -> None:
    history = _history(5)
    assert select_mobileworld_memory(history, arm="summary", budget=4) == ()
    assert select_mobileworld_memory(history, arm="recent", budget=0) == ()
    assert select_mobileworld_memory(history, arm="full", budget=0) == ()
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


def test_mobileworld_policy_prompt_audits_history_images_separately() -> None:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "restored"},
                {"type": "image", "image": "current"},
            ],
        },
    ]
    assert _history_image_count(messages) == 1
    assert _history_image_count(
        [{"role": "user", "content": [{"type": "image", "image": "current"}]}]
    ) == 0


def test_mobileworld_policy_request_restores_only_selected_images() -> None:
    request = build_mobileworld_policy_request(
        instruction="Do the task",
        step_id=4,
        screenshot=b"current",
        history=_history(3),
        selected_event_step_ids=(2, 3),
        screen_size=(432, 768),
    )
    assert (
        request["history"][0]["restored_observation_screenshot_png_base64"]
        is None
    )
    assert base64.b64decode(
        request["history"][1]["restored_observation_screenshot_png_base64"]
    ) == b"observation-2"
    with pytest.raises(ValueError, match="unknown"):
        build_mobileworld_policy_request(
            instruction="Do the task",
            step_id=4,
            screenshot=b"current",
            history=_history(3),
            selected_event_step_ids=(4,),
            screen_size=(432, 768),
        )


def _png_bytes(color: str) -> bytes:
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _official_history(count: int) -> tuple[MobileWorldHistoryEvent, ...]:
    return tuple(
        MobileWorldHistoryEvent(
            step_id=index,
            action={"action_type": "click", "x": 120, "y": 1600},
            action_text=f"Tap target {index}",
            full_response=(
                f"Action: Tap target {index}\n"
                '<tool_call>\n{"name":"mobile_use","arguments":'
                f'{{"action":"click","coordinate":[120,{200 + index}]}}'
                "}\n</tool_call>"
            ),
            policy_parsed=True,
            screen_changed=True,
            observation_screenshot=_png_bytes("red"),
            post_screenshot=_png_bytes("blue"),
        )
        for index in range(1, count + 1)
    )


def test_mobileworld_b0_uses_official_text_history_without_executor_json() -> None:
    request = build_mobileworld_policy_request(
        instruction="Do the task",
        step_id=3,
        screenshot=_png_bytes("white"),
        history=_official_history(2),
        selected_event_step_ids=(),
        screen_size=(1080, 2400),
    )
    messages = build_mobileworld_gui_owl_messages(request)
    assert len(messages) == 2
    assert messages[0]["content"][0]["text"] == OFFICIAL_SYSTEM_PROMPT
    expected_text = OFFICIAL_FIRST_USER_TEMPLATE.format(
        goal="Do the task",
        history="Step1: Tap target 1\nStep2: Tap target 2",
    )
    assert messages[1]["content"][0]["text"] == expected_text
    assert "action_type" not in expected_text
    assert "1600" not in expected_text
    assert [block["type"] for block in messages[1]["content"]] == [
        "text",
        "image",
    ]


def test_mobileworld_b2_uses_official_alternating_full_response_history() -> None:
    history = _official_history(3)
    request = build_mobileworld_policy_request(
        instruction="Do the task",
        step_id=4,
        screenshot=_png_bytes("white"),
        history=history,
        selected_event_step_ids=(2, 3),
        screen_size=(1080, 2400),
    )
    messages = build_mobileworld_gui_owl_messages(request)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"][0]["text"] == history[1].full_response
    assert messages[4]["content"][0]["text"] == history[2].full_response
    assert [block["type"] for block in messages[-1]["content"]] == ["image"]


def test_mobileworld_official_prompt_rejects_non_recent_selection() -> None:
    request = build_mobileworld_policy_request(
        instruction="Do the task",
        step_id=4,
        screenshot=_png_bytes("white"),
        history=_official_history(3),
        selected_event_step_ids=(1, 3),
        screen_size=(1080, 2400),
    )
    with pytest.raises(ValueError, match="contiguous recent suffix"):
        build_mobileworld_gui_owl_messages(request)


def test_official_parse_failure_becomes_a_history_step_not_an_exception() -> None:
    result = interpret_official_output("Action: try again")
    assert result.policy_parsed is False
    assert result.canonical_action is None
    assert result.action_text == "try again"
    assert result.full_response == "Action: try again"
    assert result.parse_error == "no tool call present"


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


def test_mobileworld_policy_http_error_preserves_server_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = io.BytesIO(
        json.dumps(
            {
                "error_type": "ValueError",
                "error": "invalid generated action",
            }
        ).encode("utf-8")
    )

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise urllib.error.HTTPError(
            "http://policy/act",
            500,
            "Internal Server Error",
            hdrs=None,
            fp=body,
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)
    policy = HTTPMobileWorldPolicy("http://policy/act")
    with pytest.raises(
        RuntimeError,
        match='MobileWorld policy HTTP 500:.*"invalid generated action"',
    ):
        policy.act({"request": "fixture"})


def test_mobileworld_task_subset_is_hash_locked(
    tmp_path: Path,
) -> None:
    roster = ["TaskA", "TaskB", "TaskC"]
    subset_path = tmp_path / "subset.json"
    subset_path.write_text(
        json.dumps(
            {
                "schema_version": "causalcache.mobileworld.task_subset.v1",
                "full_roster_sha256": _task_names_sha256(roster),
                "task_names_sha256": _task_names_sha256(["TaskA", "TaskC"]),
                "tasks": ["TaskA", "TaskC"],
            }
        ),
        encoding="utf-8",
    )
    assert _load_task_subset(subset_path, roster) == ["TaskA", "TaskC"]
    value = json.loads(subset_path.read_text(encoding="utf-8"))
    value["tasks"] = ["TaskA", "Unknown"]
    subset_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown tasks"):
        _load_task_subset(subset_path, roster)


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


def test_official_mobileworld_configs_differ_only_in_memory_budget() -> None:
    b0 = json.loads(OFFICIAL_B0_CONFIG_PATH.read_text(encoding="utf-8"))
    b4 = json.loads(OFFICIAL_B4_CONFIG_PATH.read_text(encoding="utf-8"))
    assert b0["schema_version"] == b4["schema_version"] == (
        "causalcache.mobileworld.benchmark_config.v2"
    )
    assert b0["policy"]["prompt_protocol"] == b4["policy"]["prompt_protocol"] == (
        "mobile_agent_v3_5_gui_owl_official_faithful"
    )
    assert (b0["policy"]["memory_budget"], b0["policy"]["last_image"]) == (0, 1)
    assert (b4["policy"]["memory_budget"], b4["policy"]["last_image"]) == (4, 5)
    b0_comparable = json.loads(json.dumps(b0))
    b4_comparable = json.loads(json.dumps(b4))
    for value in (b0_comparable, b4_comparable):
        value["policy"].pop("memory_budget")
        value["policy"].pop("last_image")
        value["outputs"] = {}
    assert b0_comparable == b4_comparable
