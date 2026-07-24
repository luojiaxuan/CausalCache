from __future__ import annotations

from causalcache.hgkv_selector_v2 import restored_set_key
from scripts.build_selector_gate_plan_v2 import build_plan
from scripts.reduce_selector_selected_set_gate_v2 import reduce_gate


def test_v2_gate_uses_true_complete_set_scores_and_frozen_conjunction():
    methods = [
        "hgkv_v2_beam4",
        "hgkv_v2_greedy",
        "random",
        "recent",
        "similarity",
    ]
    plan = []
    scores = {}
    b0 = {}
    for episode_index in range(3):
        episode = f"episode-{episode_index}"
        pair_group = f"{episode}:20"
        b0[pair_group] = -1.0
        for budget in (1, 2, 4):
            for method_index, method in enumerate(methods):
                if budget == 1 and method in (
                    "hgkv_v2_beam4",
                    "hgkv_v2_greedy",
                ):
                    selected = [1]
                else:
                    start = 1 + method_index * 5
                    selected = list(range(start, start + budget))
                plan.append(
                    {
                        "episode": episode,
                        "pair_group": pair_group,
                        "method": method,
                        "budget": budget,
                        "selected_event_step_ids": selected,
                    }
                )
                if method == "hgkv_v2_beam4":
                    utility = 0.5
                elif method == "hgkv_v2_greedy":
                    utility = 0.5 if budget == 1 else 0.4
                else:
                    utility = 0.1
                scores[(pair_group, restored_set_key(selected))] = (
                    -1.0 + utility
                )

    per_state, summary = reduce_gate(
        plan,
        selected_scores=scores,
        b0=b0,
        seed=7,
        bootstrap_replicates=100,
    )
    assert len(per_state) == 45
    assert summary["status"] == "PASS_SELECTED_SET_GATE_V2"
    assert summary["b1_parity_mismatches"] == 0
    assert summary["comparisons"][
        "B2:hgkv_v2_beam4-minus-recent"
    ]["ci95"][0] > 0
    assert summary["comparisons"][
        "AvgB1B2B4:hgkv_v2_beam4-minus-recent"
    ]["ci95"][0] > 0


def test_v2_gate_plan_keeps_full_history_baselines_and_at_most_b():
    pair_group = "episode:20"
    state = {
        "episode": "episode",
        "pair_group": pair_group,
        "candidate_event_step_ids": list(range(1, 13)),
    }
    v2 = {}
    v1 = {}
    teacher = {}
    for budget in (1, 2, 4):
        for method in ("hgkv_v2_greedy", "hgkv_v2_beam4"):
            v2[(pair_group, method, budget)] = {
                "selected_event_step_ids": list(range(1, budget + 1))
            }
        v1[(pair_group, "hgkv_v1_singleton", budget)] = {
            "selected_event_step_ids": list(range(2, 2 + budget))
        }
        teacher[(pair_group, "teacher_beam4", budget)] = {
            "selected_event_step_ids": []
        }
    rows = build_plan(
        states={pair_group: state},
        v2=v2,
        v1=v1,
        teacher=teacher,
        exact={},
        similarity={
            pair_group: (
                list(range(12, 0, -1)),
                {candidate: candidate / 12 for candidate in range(1, 13)},
            )
        },
        random_seed=7,
    )
    assert len(rows) == 21
    recent_b4 = next(
        row
        for row in rows
        if row["method"] == "recent" and row["budget"] == 4
    )
    assert recent_b4["selected_event_step_ids"] == [9, 10, 11, 12]
    teacher_rows = [row for row in rows if row["method"] == "teacher_beam4"]
    assert all(row["selected_event_step_ids"] == [] for row in teacher_rows)
