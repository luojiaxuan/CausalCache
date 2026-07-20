from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import serialize_gui_owl_v2_1_teacher_target
from causalcache.set_utility_evaluation_schedule import SelectorState
from causalcache.set_utility_native_replay import (
    build_native_replay_state,
    compare_action_components,
)


def _action(action: str, **kwargs):
    return GUIOwlV2Action(action=action, **kwargs)


def test_action_component_comparison_uses_applicable_denominators() -> None:
    reference = _action("click", coordinate=(12, 34))
    exact = compare_action_components(
        reference, _action("click", coordinate=(12, 34))
    )
    assert exact == {
        "canonical_action_exact": True,
        "action_type_exact": True,
        "target_applicable": True,
        "target_exact": True,
        "text_applicable": False,
        "text_nfkc_exact": None,
    }

    different_type_same_point = compare_action_components(
        reference, _action("long_press", coordinate=(12, 34))
    )
    assert different_type_same_point["canonical_action_exact"] is False
    assert different_type_same_point["action_type_exact"] is False
    assert different_type_same_point["target_exact"] is True


def test_action_component_comparison_text_is_nfkc_exact() -> None:
    reference = _action("type", text="ABC")
    exact = compare_action_components(reference, _action("answer", text="ABC"))
    assert exact["action_type_exact"] is False
    assert exact["target_applicable"] is False
    assert exact["target_exact"] is None
    assert exact["text_applicable"] is True
    assert exact["text_nfkc_exact"] is True

    missing = compare_action_components(reference, _action("wait"))
    assert missing["text_applicable"] is True
    assert missing["text_nfkc_exact"] is False


def test_native_replay_state_deduplicates_method_budget_coalitions() -> None:
    state = SelectorState(
        state_id="trajectory:decision:006",
        trajectory_id="trajectory",
        logical_shard=3,
        candidate_event_ids=(1, 2, 3, 4, 5),
        tracks=("exact_oracle",),
        methods={
            "set_transformer": {
                1: (1,),
                2: (1,),
                3: (1, 2),
                4: (1, 2),
            },
            "recent": {
                1: (5,),
                2: (4, 5),
                3: (3, 4, 5),
                4: (2, 3, 4, 5),
            },
            "ocr_rgb": {
                1: (1,),
                2: (1, 2),
                3: (1, 2),
                4: (1, 2),
            },
        },
    )
    action = _action("click", coordinate=(100, 200))
    terminal = {
        "candidate_event_step_ids": [1, 2, 3, 4, 5],
        "reference": {
            "serialized_action": serialize_gui_owl_v2_1_teacher_target(action)
        },
        "state_id": state.state_id,
        "status": "COMPLETED_VARIABLE_HISTORY_LABEL_STATE",
        "trajectory_id": state.trajectory_id,
    }
    replay = build_native_replay_state(
        state,
        winner_model="set_transformer",
        budgets=(1, 2, 3, 4),
        reference_terminal=terminal,
        reference_terminal_sha256="a" * 64,
    )
    coalitions = {
        tuple(row["event_ids"]): tuple(row["sources"])
        for row in replay["coalitions"]
    }
    assert len(coalitions) == 6
    assert coalitions[(1,)] == ("ocr_rgb:B1", "winner:B1", "winner:B2")
    assert coalitions[(1, 2)] == (
        "ocr_rgb:B2",
        "ocr_rgb:B3",
        "ocr_rgb:B4",
        "winner:B3",
        "winner:B4",
    )
    assert replay["reference"]["canonical_action"] == {
        "action": "click",
        "coordinate": [100, 200],
    }

