from __future__ import annotations

from causalcache.set_utility_dense import (
    build_dense_prompt_plan,
    dense_decision_steps,
    dense_legacy_coverage,
    dense_required_event_step_ids,
    dense_trajectory_partition,
    derive_dense_states_from_query_pair,
    legacy_available_event_step_ids,
)
from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_processor_artifacts import ProcessorQueryArtifactRecord


def _query_pair(decision_count: int) -> tuple[ProcessorQueryArtifactRecord, ...]:
    trajectory_id = "dense-fixture"
    history = tuple(
        {
            "event_step_id": step,
            "high_fidelity_observation_ref": f"images/{trajectory_id}/observation-{step:03d}.png",
            "low_fidelity_summary": LowFidelityEventV2(
                step_id=step,
                action_type="wait",
                action_argument="wait",
                foreground_app="fixture",
                screen_text_added=(f"screen-{step}",),
                screen_text_removed=(),
                screen_change="low",
                executor_result="unknown",
            ).to_ordered_dict(),
            "observation_after_ref": f"images/{trajectory_id}/observation-{step:03d}.png",
            "observation_before_ref": f"images/{trajectory_id}/observation-{step - 1:03d}.png",
        }
        for step in range(1, decision_count + 1)
    )
    all_ocr = {
        f"images/{trajectory_id}/observation-{step:03d}.png": {
            "image_member_path": f"images/{trajectory_id}/observation-{step:03d}.png",
            "full_spatial_tokens": [f"screen-{step}"],
        }
        for step in range(decision_count + 1)
    }
    anchor_step = 6 if decision_count <= 9 else 10 if decision_count <= 17 else 18
    result = []
    for kind, decision_step in (
        ("stratum_anchor", anchor_step),
        ("terminal", decision_count + 1),
    ):
        current = decision_step - 1
        candidates = tuple(range(1, current))[-16:]
        refs = {
            f"images/{trajectory_id}/observation-{current:03d}.png",
            *(f"images/{trajectory_id}/observation-{step:03d}.png" for step in candidates),
        }
        prefix = history[:current]
        ocr_refs = {
            event[field]
            for event in prefix
            for field in ("observation_before_ref", "observation_after_ref")
        }
        result.append(
            ProcessorQueryArtifactRecord(
                state_id=f"{trajectory_id}:decision:{decision_step:03d}",
                trajectory_id=trajectory_id,
                role="train",
                query_kind=kind,
                decision_step_id=decision_step,
                current_equivalent_event_step_id=current,
                maximum_labeled_cardinality=2,
                initial_candidate_event_step_ids=candidates,
                task_instruction="Do the dense fixture",
                history_events=prefix,
                current_observation_ref=f"images/{trajectory_id}/observation-{current:03d}.png",
                ocr_records_by_path={ref: all_ocr[ref] for ref in sorted(ocr_refs)},
                image_payloads={ref: ref.encode() for ref in sorted(refs)},
            )
        )
    return tuple(result)


def test_dense_state_count_uses_every_eligible_step() -> None:
    assert dense_decision_steps(decision_count=6) == (6, 7)
    assert dense_decision_steps(decision_count=17) == tuple(range(6, 19))
    assert dense_required_event_step_ids(6) == frozenset({1, 2, 3, 4, 5})
    assert dense_required_event_step_ids(18) == frozenset({13, 14, 15, 16, 17})


def test_legacy_anchor_terminal_images_cover_short_trajectories() -> None:
    total, covered, missing = dense_legacy_coverage(decision_count=29)
    assert (total, covered, missing) == (25, 25, ())
    assert legacy_available_event_step_ids(decision_count=29) == frozenset(range(1, 30))


def test_legacy_anchor_terminal_images_expose_long_trajectory_gap() -> None:
    total, covered, missing = dense_legacy_coverage(decision_count=38)
    assert total == 34
    assert covered == 26
    assert missing == tuple(range(19, 27))


def test_query_pair_expands_every_short_trajectory_step() -> None:
    states = derive_dense_states_from_query_pair(_query_pair(10))
    assert len(states) == 6
    assert [state.query.decision_step_id for state in states] == list(range(6, 12))
    assert states[0].query.candidate_event_step_ids == (1, 2, 3, 4)
    assert states[-1].query.candidate_event_step_ids == (6, 7, 8, 9)
    plan = build_dense_prompt_plan(states[-1].query, (6, 9))
    assert plan.restored_event_step_ids == (6, 9)


def test_supplemental_images_close_long_trajectory_gap() -> None:
    pair = _query_pair(38)
    partial = derive_dense_states_from_query_pair(pair)
    supplemental = {
        f"images/dense-fixture/observation-{step:03d}.png": (
            f"images/dense-fixture/observation-{step:03d}.png".encode()
        )
        for step in range(18, 22)
    }
    complete = derive_dense_states_from_query_pair(
        pair, supplemental_image_payloads=supplemental
    )
    assert len(partial) == 26
    assert len(complete) == 34


def test_dense_trajectory_partition_is_stable_and_disjoint() -> None:
    ids = tuple(f"trajectory-{index}" for index in range(100))
    first = {item: dense_trajectory_partition(item, partition_count=2) for item in ids}
    second = {item: dense_trajectory_partition(item, partition_count=2) for item in reversed(ids)}
    assert first == second
    assert set(first.values()) == {0, 1}
