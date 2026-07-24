from __future__ import annotations

from causalcache.hgkv_selector_v2 import restored_set_key
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
