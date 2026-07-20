from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalcache.set_utility_evaluation_schedule import (
    EvaluationScheduleConfig,
    EvaluationTracks,
    SelectorState,
    build_evaluation_schedule,
    deterministic_random_subset,
    materialize_evaluation_schedules,
    selector_states_from_payload,
)


def _config(
    *,
    exact_state_count: int = 1,
    large_history_state_count: int = 2,
    union_state_count: int = 2,
    logical_shard_count: int = 3,
    seed: int = 19,
) -> EvaluationScheduleConfig:
    return EvaluationScheduleConfig.from_mapping(
        {
            "evaluation_schedule": {
                "budgets": [1, 2, 3, 4],
                "exact_state_count": exact_state_count,
                "large_history_state_count": large_history_state_count,
                "union_state_count": union_state_count,
                "logical_shard_count": logical_shard_count,
                "random_control": {
                    "rule": "sha256_seed_state_event_nested_prefix",
                    "seed": seed,
                },
            }
        }
    )


def _methods() -> dict[str, dict[int, tuple[int, ...]]]:
    return {
        "deepsets": {
            1: (1,),
            2: (1, 2),
            3: (1, 2, 3),
            4: (1, 2, 3, 4),
        },
        "set_transformer": {
            1: (5,),
            2: (4, 5),
            3: (3, 4, 5),
            4: (2, 3, 4, 5),
        },
    }


def _state(
    trajectory_id: str,
    *,
    logical_shard: int,
    tracks: tuple[str, ...],
    candidate_count: int = 5,
) -> SelectorState:
    return SelectorState(
        state_id=f"{trajectory_id}:decision:{candidate_count + 1:03d}",
        trajectory_id=trajectory_id,
        logical_shard=logical_shard,
        candidate_event_ids=tuple(range(1, candidate_count + 1)),
        tracks=tracks,
        methods=_methods(),
    )


def _selection_record(state: SelectorState) -> dict[str, object]:
    return {
        "candidate_event_ids": list(state.candidate_event_ids),
        "logical_shard": state.logical_shard,
        "methods": {
            method: {
                str(budget): list(subset)
                for budget, subset in sorted(by_budget.items())
            }
            for method, by_budget in sorted(state.methods.items())
        },
        "state_id": state.state_id,
        "tracks": list(state.tracks),
        "trajectory_id": state.trajectory_id,
    }


def test_exact_oracle_schedule_is_exhaustive_through_pairs() -> None:
    config = _config()
    state = _state(
        "exact-trajectory",
        logical_shard=0,
        tracks=("exact_oracle", "large_history"),
    )
    schedule = build_evaluation_schedule(state, config=config)
    rows = {
        tuple(row["event_ids"]): tuple(row["sources"])
        for row in schedule["coalitions"]
    }

    assert schedule["role"] == "evaluation"
    assert schedule["tracks"] == ["exact_oracle", "large_history"]
    assert () in rows
    assert state.candidate_event_ids in rows
    assert all((event_id,) in rows for event_id in state.candidate_event_ids)
    assert all(
        (left, right) in rows
        for left in state.candidate_event_ids
        for right in state.candidate_event_ids
        if left < right
    )
    assert "anchor:empty" in rows[()]
    assert "exact:cardinality:0" in rows[()]
    assert "anchor:full" in rows[state.candidate_event_ids]
    assert "exact:full" in rows[state.candidate_event_ids]
    assert all(
        row["source"] == min(row["sources"])
        and row["sources"] == sorted(set(row["sources"]))
        for row in schedule["coalitions"]
    )


def test_large_history_schedule_only_adds_selected_and_control_subsets() -> None:
    config = _config()
    state = _state(
        "large-trajectory",
        logical_shard=1,
        tracks=("large_history",),
    )
    schedule = build_evaluation_schedule(state, config=config)

    assert not any(
        source.startswith("exact:")
        for row in schedule["coalitions"]
        for source in row["sources"]
    )
    allowed = {(), state.candidate_event_ids}
    allowed.update(
        subset for by_budget in state.methods.values() for subset in by_budget.values()
    )
    allowed.update(
        deterministic_random_subset(
            state,
            budget=budget,
            seed=config.random_control_seed,
        )
        for budget in config.budgets
    )
    assert {
        tuple(row["event_ids"]) for row in schedule["coalitions"]
    } == allowed


def test_random_controls_are_deterministic_nested_prefixes() -> None:
    state = _state(
        "random-trajectory",
        logical_shard=0,
        tracks=("large_history",),
        candidate_count=20,
    )
    subsets = [
        deterministic_random_subset(state, budget=budget, seed=73)
        for budget in range(1, 5)
    ]

    assert subsets == [
        deterministic_random_subset(state, budget=budget, seed=73)
        for budget in range(1, 5)
    ]
    assert all(set(left) < set(right) for left, right in zip(subsets, subsets[1:]))
    assert subsets[1] != deterministic_random_subset(state, budget=2, seed=74)


def test_selector_payload_requires_snapshot_track_names_and_complete_budgets() -> None:
    config = _config()
    exact = _state(
        "exact-trajectory",
        logical_shard=0,
        tracks=("exact_oracle", "large_history"),
    )
    large = _state(
        "large-trajectory",
        logical_shard=1,
        tracks=("large_history",),
    )
    tracks = EvaluationTracks(
        exact_state_ids=frozenset({exact.state_id}),
        large_history_state_ids=frozenset({exact.state_id, large.state_id}),
    )
    payload = {"records": [_selection_record(exact), _selection_record(large)]}

    parsed = selector_states_from_payload(payload, config=config, tracks=tracks)
    assert [state.state_id for state in parsed] == sorted(tracks.union_state_ids)

    wrong_track = json.loads(json.dumps(payload))
    wrong_track["records"][0]["tracks"][0] = "exact"
    with pytest.raises(ValueError, match="track tags"):
        selector_states_from_payload(wrong_track, config=config, tracks=tracks)

    missing_budget = json.loads(json.dumps(payload))
    del missing_budget["records"][0]["methods"]["deepsets"]["4"]
    with pytest.raises(ValueError, match="budget inventory"):
        selector_states_from_payload(missing_budget, config=config, tracks=tracks)


def _write_materializer_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "evaluation_schedule": {
                    "budgets": [1, 2, 3, 4],
                    "exact_state_count": 1,
                    "large_history_state_count": 2,
                    "logical_shard_count": 3,
                    "random_control": {
                        "rule": "sha256_seed_state_event_nested_prefix",
                        "seed": 19,
                    },
                    "union_state_count": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    exact = _state(
        "exact-trajectory",
        logical_shard=0,
        tracks=("exact_oracle", "large_history"),
    )
    large = _state(
        "large-trajectory",
        logical_shard=1,
        tracks=("large_history",),
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "evaluation_tracks": {
                    "exact_state_ids": [exact.state_id],
                    "large_history_state_ids": [exact.state_id, large.state_id],
                },
                "status": "FROZEN_VARIABLE_HISTORY_STATE_INVENTORY",
            }
        ),
        encoding="utf-8",
    )
    selections_path = tmp_path / "selections.json"
    selections_path.write_text(
        json.dumps(
            {"records": [_selection_record(exact), _selection_record(large)]}
        ),
        encoding="utf-8",
    )
    source_manifest_path = tmp_path / "source-manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "logical_shard": 0,
                        "sha256": "a" * 64,
                        "trajectory_ids": [exact.trajectory_id],
                    },
                    {
                        "logical_shard": 1,
                        "sha256": "b" * 64,
                        "trajectory_ids": [large.trajectory_id],
                    },
                    {
                        "logical_shard": 2,
                        "sha256": "c" * 64,
                        "trajectory_ids": [],
                    },
                ],
                "status": "COMPLETED_VARIABLE_HISTORY_SOURCE",
            }
        ),
        encoding="utf-8",
    )
    return config_path, inventory_path, selections_path, source_manifest_path


def test_materializer_emits_resumable_bound_runner_schedules(tmp_path: Path) -> None:
    config_path, inventory_path, selections_path, source_path = (
        _write_materializer_inputs(tmp_path)
    )
    output_root = tmp_path / "output"
    summaries = [
        materialize_evaluation_schedules(
            config_path=config_path,
            inventory_path=inventory_path,
            selections_path=selections_path,
            source_manifest_path=source_path,
            output_root=output_root,
            partition_index=partition_index,
            partition_count=2,
            workers=2,
        )
        for partition_index in range(2)
    ]

    assert sum(summary["state_count"] for summary in summaries) == 2
    assert all(
        summary["method_names"] == ["deepsets", "set_transformer"]
        for summary in summaries
    )
    assert len(tuple((output_root / "schedule-shards").glob("*.jsonl"))) == 3
    assert len(tuple((output_root / "receipts").glob("*.json"))) == 3
    assert len(tuple((output_root / "workers").glob("*.json"))) == 2
    assert (output_root / "schedule-shards/shard-002-of-3.jsonl").read_bytes() == b""
    exact_row = json.loads(
        (output_root / "schedule-shards/shard-000-of-3.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert exact_row["role"] == "evaluation"
    assert any(not row["event_ids"] for row in exact_row["coalitions"])
    assert any(
        row["event_ids"] == exact_row["candidate_event_ids"]
        for row in exact_row["coalitions"]
    )
    receipt = json.loads(
        (output_root / "receipts/shard-000-of-3.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["config_sha256"]
    assert receipt["inventory_sha256"]
    assert receipt["selections_sha256"]
    assert receipt["source_manifest_sha256"]
    assert receipt["source_shard_sha256"] == "a" * 64

    repeated = materialize_evaluation_schedules(
        config_path=config_path,
        inventory_path=inventory_path,
        selections_path=selections_path,
        source_manifest_path=source_path,
        output_root=output_root,
        partition_index=0,
        partition_count=2,
        workers=2,
    )
    assert repeated == summaries[0]

    schedule_path = output_root / "schedule-shards/shard-000-of-3.jsonl"
    schedule_path.write_bytes(schedule_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="receipt drifted"):
        materialize_evaluation_schedules(
            config_path=config_path,
            inventory_path=inventory_path,
            selections_path=selections_path,
            source_manifest_path=source_path,
            output_root=output_root,
            partition_index=0,
            partition_count=2,
            workers=2,
        )
