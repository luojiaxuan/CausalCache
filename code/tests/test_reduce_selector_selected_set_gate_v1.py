from __future__ import annotations

from scripts.build_odyssey_sft_dataset import restored_set_key
from scripts.reduce_selector_selected_set_gate_v1 import reduce_gate


def test_reduce_gate_uses_true_set_scores_and_clustered_pairs() -> None:
    plan = []
    scores = {}
    b0 = {}
    methods = [
        "hgkv_set_conditioned",
        "hgkv_singleton",
        "random",
        "recent",
        "similarity",
    ]
    for episode_index in range(2):
        episode = f"episode-{episode_index}"
        pair_group = f"{episode}:6"
        b0[pair_group] = -1.0
        for method_index, method in enumerate(methods):
            selected = [method_index + 1]
            plan.append(
                {
                    "budget": 1,
                    "episode": episode,
                    "method": method,
                    "pair_group": pair_group,
                    "selected_event_step_ids": selected,
                }
            )
            utility = 0.5 if method == "hgkv_set_conditioned" else 0.1
            scores[(pair_group, restored_set_key(selected))] = -1.0 + utility

    per_state, summary = reduce_gate(
        plan,
        selected_scores=scores,
        b0=b0,
        seed=7,
        bootstrap_replicates=100,
    )

    assert len(per_state) == 10
    assert summary["status"] == "PASS_SELECTED_SET_GATE_V1"
    assert (
        summary["comparisons"][
            "B1:hgkv_set_conditioned-minus-recent"
        ]["episode_macro_mean"]
        == 0.4
    )
