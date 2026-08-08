from causalcache.findingdory_oracle_gap import (
    exact_budget_selections,
    frame_is_valid,
    logical_episode_shard,
    normalize_content_summary,
    oracle_evidence_frame,
    parse_answer_groups,
    parse_predicted_frame,
    reduce_paired_results,
)


def test_answer_and_exact_budget_selection() -> None:
    groups = parse_answer_groups("[[20, 21, 22, 23], [40, 41]]")
    assert oracle_evidence_frame(groups, maximum_history_frame=94) == 21
    assert exact_budget_selections(groups, budget=1, current_frame=95) == {
        "recent": (94,),
        "oracle": (21,),
    }
    assert exact_budget_selections(groups, budget=4, current_frame=95) == {
        "recent": (91, 92, 93, 94),
        "oracle": (21, 92, 93, 94),
    }


def test_prediction_parser_and_validity() -> None:
    groups = ((20, 21),)
    assert parse_predicted_frame('text {"frame_indices":[21]}') == 21
    assert parse_predicted_frame('{"frame_id":"20"}') == 20
    assert parse_predicted_frame('{"frame_indices":[]}') is None
    assert frame_is_valid(21, groups)
    assert not frame_is_valid(None, groups)


def test_content_summary_removes_legacy_frame_and_time_namespace() -> None:
    raw = '''```json
    {"frame_id": 106, "time_of_day": "06:25", "objects": ["2 purple rolls"],
     "interactions": ["picked object at frame 112"], "fine_attributes": ["striped"]}
    ```'''
    assert normalize_content_summary(raw) == {
        "objects": ["purple rolls"],
        "interactions": ["picked object at frame"],
        "fine_attributes": ["striped"],
    }


def test_logical_episode_shards_are_disjoint_and_complete() -> None:
    episodes = ["ep_10", "ep_2", "ep_1", "ep_3"]
    left = logical_episode_shard(episodes, shard_index=0, num_shards=2)
    right = logical_episode_shard(episodes, shard_index=1, num_shards=2)
    assert left == ("ep_1", "ep_3")
    assert right == ("ep_2", "ep_10")
    assert set(left).isdisjoint(right)
    assert set(left) | set(right) == set(episodes)


def test_paired_reduction() -> None:
    rows = []
    for episode_id in ("ep_1", "ep_2"):
        for task_id in ("a", "b"):
            common = {"episode_id": episode_id, "task_id": task_id, "budget": 1}
            rows.append(common | {"arm": "recent", "success": False, "selected_frames": [94]})
            rows.append(common | {"arm": "oracle", "success": True, "selected_frames": [20]})
    result = reduce_paired_results(
        rows,
        bootstrap_samples=100,
        seed=7,
        minimum_oracle_success=0.3,
        minimum_delta=0.1,
        minimum_coverage=0.5,
    )
    budget = result["budgets"]["1"]
    assert budget["recent_hl_sr"] == 0.0
    assert budget["oracle_hl_sr"] == 1.0
    assert budget["oracle_minus_recent"] == 1.0
    assert budget["gate_pass"] is True
