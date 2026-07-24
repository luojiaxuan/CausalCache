"""True-utility teacher beam planning and reduction invariants."""

from __future__ import annotations

import json
from argparse import Namespace

import torch

from causalcache.hgkv_selector_v2 import (
    HGKVSetSelectorV2,
    learned_beam_search,
)
from scripts.plan_hgkv_teacher_beam_v2 import build_plan_rows
from scripts.build_hgkv_selector_v2_training_data import build_training_rows
from scripts.reduce_hgkv_teacher_beam_v2 import reduce_depth
from scripts.render_hgkv_teacher_beam_v2 import (
    annotate_samples,
    load_missing_coalitions,
    selected_set_plan,
)
from scripts.run_hgkv_teacher_beam_v2 import run
from scripts.select_hgkv_sets_v2 import select_state
from scripts.train_hgkv_selector_v2 import (
    collate_groups,
    trajectory_group_folds,
)
from scripts.validate_hgkv_teacher_render_v2 import validate_render
from scripts.validate_selector_inference_v2 import validate_selections


def _state():
    return {
        "episode": "episode",
        "pair_group": "episode:9",
        "decision_step": 9,
        "history_length": 8,
        "candidate_event_step_ids": [1, 2, 3, 4, 5],
        "recent_candidate_cap": None,
    }


def _cache_row(pair_group: str, key: str, utility: float):
    return {
        "pair_group": pair_group,
        "restored_set_key": key,
        "target_action_sha256": "a" * 64,
        "u_act": utility,
    }


def test_depth0_beam_uses_true_singleton_utility():
    pair_group = "episode:9"
    cache = {(pair_group, ""): _cache_row(pair_group, "", 0.0)}
    for candidate, utility in {1: 0.1, 2: 0.5, 3: -0.2, 4: 0.4, 5: 0.3}.items():
        cache[(pair_group, str(candidate))] = _cache_row(
            pair_group, str(candidate), utility
        )
    plan = build_plan_rows(
        states={pair_group: _state()},
        cache=cache,
        prefixes_by_state={},
        depth=0,
    )
    labels, beam = reduce_depth(
        plan_rows=plan,
        cache=cache,
        depth=0,
        beam_width=4,
    )
    assert len(plan) == 5
    assert len(labels) == 1
    assert labels[0]["marginal_targets"] == [0.1, 0.5, -0.2, 0.4, 0.3]
    assert labels[0]["stop_is_optimal"] is False
    assert beam[0]["prefixes"] == [[2], [4], [5], [1]]


def test_depth1_renders_unique_pairs_but_keeps_all_prefix_edges():
    pair_group = "episode:9"
    singleton_prefixes = [[1], [2], [3], [4]]
    plan = build_plan_rows(
        states={pair_group: _state()},
        cache={},
        prefixes_by_state={pair_group: singleton_prefixes},
        depth=1,
    )
    assert len(plan) == 16
    assert len({row["restored_set_key"] for row in plan}) == 10

    cache = {
        (pair_group, str(candidate)): _cache_row(
            pair_group, str(candidate), candidate / 100
        )
        for candidate in range(1, 5)
    }
    for row in plan:
        key = row["restored_set_key"]
        values = [int(value) for value in key.split("-")]
        cache[(pair_group, key)] = _cache_row(
            pair_group, key, sum(values) / 10
        )
    labels, beam = reduce_depth(
        plan_rows=plan,
        cache=cache,
        depth=1,
        beam_width=4,
    )
    assert len(labels) == 4
    assert all(len(row["marginal_targets"]) == 4 for row in labels)
    assert beam[0]["prefixes"] == [[4, 5], [3, 5], [2, 5], [3, 4]]


def test_teacher_renderer_deduplicates_child_set_without_losing_plan_edges(
    tmp_path,
):
    pair_group = "episode:9"
    rows = [
        {
            "episode": "episode",
            "pair_group": pair_group,
            "decision_step": 9,
            "candidate_event_step_ids": [1, 2, 3],
            "selected_event_step_ids": [prefix],
            "candidate_event_step_id": candidate,
            "restored_event_step_ids": [1, 2],
            "restored_set_key": "1-2",
            "depth": 1,
            "cache_hit": False,
        }
        for prefix, candidate in ((1, 2), (2, 1))
    ]
    path = tmp_path / "plan.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    coalitions = load_missing_coalitions([path])
    assert len(coalitions) == 1
    plan = selected_set_plan(coalitions)
    assert plan[pair_group][0]["selected_event_step_ids"] == [1, 2]
    sample = {
        "pair_group": pair_group,
        "variant": "selected_set",
        "restored_set_key": "1-2",
    }
    annotated = annotate_samples(
        [sample],
        expected_by_key=coalitions,
    )
    assert annotated[0]["teacher_depth"] == 1

    image = tmp_path / "candidate.png"
    image.write_bytes(b"png")
    annotated[0]["memory_config"] = {
        "restored_event_step_ids": [1, 2]
    }
    annotated[0]["decision_step_id"] = 9
    annotated[0]["messages"] = [
        {"content": [{"type": "image", "path": image.name}]}
    ]
    sample_path = tmp_path / "samples.jsonl"
    sample_path.write_text(
        json.dumps(annotated[0]) + "\n",
        encoding="utf-8",
    )
    result = validate_render(
        plan_paths=[path],
        sample_paths=[sample_path],
    )
    assert result["observed_coalitions"] == 1
    assert result["referenced_images"] == 1


def test_teacher_runner_reduces_covered_depth_then_waits_for_exact_cache(
    tmp_path,
):
    state_path = tmp_path / "states.jsonl"
    state_path.write_text(json.dumps(_state()) + "\n", encoding="utf-8")
    pair_group = "episode:9"
    cache_rows = [_cache_row(pair_group, "", 0.0)]
    cache_rows.extend(
        _cache_row(pair_group, str(candidate), candidate / 10)
        for candidate in range(1, 6)
    )
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(
        "".join(json.dumps(row) + "\n" for row in cache_rows),
        encoding="utf-8",
    )
    args = Namespace(
        states=[str(state_path)],
        coalition_cache=[str(cache_path)],
        run_root=tmp_path / "run",
        beam_width=4,
        source_commit="deadbeef",
    )
    status = run(args)
    assert status["status"] == "AWAITING_RENDER_SCORE_CACHE_REFRESH"
    assert status["depth"] == 1
    assert status["missing_child_coalitions"] == 10
    assert (tmp_path / "run/depth0/labels.jsonl").is_file()
    assert (tmp_path / "run/depth0/beam.jsonl").is_file()
    assert run(args) == status


def test_training_groups_replicate_only_frozen_remaining_budgets():
    state = _state() | {"split": "train"}
    label = {
        "pair_group": "episode:9",
        "candidate_event_step_ids": [1, 2, 3, 4, 5],
        "selected_event_step_ids": [],
        "remaining_candidate_event_step_ids": [1, 2, 3, 4, 5],
        "marginal_targets": [0.1, 0.2, 0.3, 0.4, 0.5],
        "prefix_u_act": 0.0,
        "depth": 0,
        "stop_is_optimal": False,
    }
    rows = build_training_rows(
        states={"episode:9": state},
        feature_index={
            ("episode:9", candidate): candidate - 1
            for candidate in range(1, 6)
        },
        labels=[label],
    )
    assert [row["remaining_budget"] for row in rows] == [1, 2, 4]
    assert all(row["candidate_feature_indices"] == [0, 1, 2, 3, 4] for row in rows)
    assert all(row["selected_feature_indices"] == [] for row in rows)


def test_dynamic_collation_supports_empty_set_and_full_history_width():
    features = torch.arange(12 * 1285, dtype=torch.float32).reshape(12, 1285)
    rows = [
        {
            "candidate_feature_indices": list(range(10)),
            "selected_feature_indices": [],
            "marginal_targets": [0.1] * 10,
            "remaining_budget": 4,
        },
        {
            "candidate_feature_indices": [10, 11],
            "selected_feature_indices": [0, 1, 2],
            "marginal_targets": [0.2, -0.1],
            "remaining_budget": 1,
        },
    ]
    batch = collate_groups(rows, features, device=torch.device("cpu"))
    assert batch["candidate_features"].shape == (2, 10, 1285)
    assert batch["selected_features"].shape == (2, 3, 1285)
    assert batch["candidate_mask"].sum(dim=1).tolist() == [10, 2]
    assert batch["selected_mask"].sum(dim=1).tolist() == [0, 3]


def test_group_folds_never_split_one_trajectory():
    rows = [
        {"episode": episode}
        for episode, count in (("a", 5), ("b", 4), ("c", 3), ("d", 2), ("e", 1))
        for _ in range(count)
    ]
    folds = trajectory_group_folds(rows, folds=3)
    assert set.union(*folds) == {"a", "b", "c", "d", "e"}
    assert sum(len(fold) for fold in folds) == 5
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(folds)
        for right in folds[index + 1 :]
    )


def test_unified_model_accepts_an_explicit_empty_selected_width():
    model = HGKVSetSelectorV2()
    outputs = model(
        torch.zeros(2, 10, 1285),
        torch.ones(2, 10, dtype=torch.bool),
        torch.zeros(2, 0, 1285),
        torch.zeros(2, 0, dtype=torch.bool),
        torch.tensor([1, 4]),
    )
    assert outputs["marginal"].shape == (2, 10)
    assert outputs["rank_score"].shape == (2, 10)
    assert outputs["stop_logit"].shape == (2,)


def test_learned_beam_prunes_by_cumulative_calibrated_marginal():
    def scorer(prefix, remaining, _remaining_budget):
        table = {
            (): {1: 0.4, 2: 0.3, 3: 0.0},
            (1,): {2: -0.1, 3: -0.2},
            (2,): {1: -0.1, 3: 0.5},
            (3,): {1: -0.2, 2: 0.1},
        }
        return {candidate: table[prefix][candidate] for candidate in remaining}

    path = learned_beam_search(
        [1, 2, 3],
        budget=2,
        scorer=scorer,
        beam_width=4,
    )
    assert path.selected_event_step_ids == (2, 3)
    assert path.cumulative_predicted_u == 0.8


def test_v2_greedy_and_beam4_have_exact_b1_parity(tmp_path):
    torch.manual_seed(0)
    model = HGKVSetSelectorV2().eval()
    state = _state() | {"split": "dev"}
    rows = select_state(
        state=state,
        model=model,
        feature_by_key={
            ("episode:9", candidate): torch.zeros(1285)
            for candidate in range(1, 6)
        },
        budgets=(1, 2, 4),
        device=torch.device("cpu"),
    )
    b1 = [row for row in rows if row["budget"] == 1]
    assert len(b1) == 2
    assert b1[0]["selected_event_step_ids"] == b1[1][
        "selected_event_step_ids"
    ]
    assert b1[0]["cumulative_predicted_u"] == b1[1][
        "cumulative_predicted_u"
    ]
    selection_path = tmp_path / "selections.jsonl"
    selection_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = validate_selections(
        states={"episode:9": state},
        selection_paths=[selection_path],
    )
    assert result["rows"] == 6
    assert result["b1_parity_mismatches"] == 0
