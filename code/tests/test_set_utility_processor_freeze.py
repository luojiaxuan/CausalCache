from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_processor_freeze import (
    CandidateMinimumViolation,
    FrozenQueryCandidateRecord,
    aggregate_freeze_summary,
    assert_processor_only_import_state,
    build_final_label_schedule,
    build_full_reference_prompt_plan,
    build_shard_worker_schedule,
    canonical_json_bytes,
    freeze_query_candidates,
    label_schedule_payload,
    sha256_bytes,
    verify_processor_only_snapshot,
    validate_freeze_query_topology,
    validate_processor_only_source,
    worker_read_plan,
    write_once_or_verify,
)
from causalcache.set_utility_processor_substrate import (
    SelectedRowReadPlan,
    SelectedShardRead,
    SelectedTrajectoryAssignment,
)


ROOT = Path(__file__).resolve().parents[2]


def _assignment(index: int, *, path: str, decision_count: int) -> SelectedTrajectoryAssignment:
    source_id = f"source-{index:03d}"
    return SelectedTrajectoryAssignment(
        trajectory_id=source_id,
        source_id=source_id,
        instruction_app_group_sha256=f"{index + 1:064x}",
        role="train",
        candidate_capacity_stratum=(
            "decisions_6_9"
            if decision_count <= 9
            else "decisions_10_17"
            if decision_count <= 17
            else "decisions_18_plus"
        ),
        decision_count=decision_count,
        terminal_decision_step_id=decision_count + 1,
        transport_file=path,
        transport_row_index=index,
        p0_selection_sha256=f"{index + 101:064x}",
    )


def _plan() -> SelectedRowReadPlan:
    shards = []
    assignment_index = 0
    for shard_index, counts in enumerate(((30,), (20, 10), (14,), (8,), (6,), (6,))):
        path = f"mobile/use/train/shard-{shard_index:05d}.parquet"
        assignments = tuple(
            _assignment(
                assignment_index + offset,
                path=path,
                decision_count=count,
            )
            for offset, count in enumerate(counts)
        )
        assignment_index += len(assignments)
        shards.append(
            SelectedShardRead(
                source_file=SourceFileSpec(path, 100 + shard_index, f"{shard_index + 1:064x}"),
                source_row_count=100,
                assignments=assignments,
            )
        )
    return SelectedRowReadPlan(shards=tuple(shards), assignment_count=assignment_index)


def _summary(step_id: int) -> dict:
    return LowFidelityEventV2(
        step_id=step_id,
        action_type="wait",
        action_argument="wait",
        foreground_app="fixture",
        screen_text_added=(f"step-{step_id}",),
        screen_text_removed=(),
        screen_change="low",
        executor_result="accepted",
    ).to_ordered_dict()


def _query_payload(*, candidate_count: int = 6) -> tuple[dict, dict]:
    current_equivalent = candidate_count + 1
    trajectory = {
        "trajectory_id": "trajectory-1",
        "source_id": "trajectory-1",
        "role": "train",
        "task_instruction": "Complete the fixture task.",
        "history_events": [
            {
                "event_step_id": step,
                "low_fidelity_summary": _summary(step),
                "high_fidelity_observation_ref": f"images/{step}.png",
            }
            for step in range(1, current_equivalent + 1)
        ],
    }
    query = {
        "state_id": f"trajectory-1:decision:{current_equivalent + 1:03d}",
        "query_kind": "terminal",
        "decision_step_id": current_equivalent + 1,
        "current_equivalent_event_step_id": current_equivalent,
        "initial_candidate_event_step_ids": list(range(1, candidate_count + 1)),
        "maximum_labeled_cardinality": 2,
        "current_observation_ref": f"images/{current_equivalent}.png",
    }
    return trajectory, query


def test_shard_lpt_is_deterministic_and_never_splits_a_selected_shard() -> None:
    plan = _plan()
    first = build_shard_worker_schedule(plan)
    second = build_shard_worker_schedule(plan)

    assert first == second
    assert first.selected_shard_count == 6
    assert first.selected_trajectory_count == 7
    paths = [
        shard.source_file.transport_file
        for worker in first.workers
        for shard in worker.shards
    ]
    assert len(paths) == len(set(paths)) == 6
    assert max(worker.observation_load for worker in first.workers) <= 33
    assert sum(worker_read_plan(first, index).assignment_count for index in range(4)) == 7


def test_committed_v2_selected_shards_have_one_deterministic_owner() -> None:
    manifest = json.loads(
        (ROOT / "data/manifests/set_utility_freeze_b_v2_terminal_index_repair.json").read_text()
    )
    inventory = json.loads(
        (ROOT / "data/manifests/set_utility_full_pool_inventory_v1.json").read_text()
    )
    census = json.loads(
        (ROOT / "data/manifests/set_utility_full_pool_census_v2.json").read_text()
    )
    from causalcache.set_utility_processor_substrate import build_selected_row_read_plan

    source_files = tuple(
        SourceFileSpec(record["path"], record["size_bytes"], record["lfs_sha256"])
        for record in inventory["inventory"]["files"]
    )
    plan = build_selected_row_read_plan(
        manifest["assignments"],
        source_files=source_files,
        source_row_counts=census["source"]["row_counts_by_file"],
    )
    schedule = build_shard_worker_schedule(plan)

    assert schedule.selected_shard_count == 527
    assert schedule.selected_trajectory_count == 1200
    assert schedule.selected_observation_count == 18_792
    assert max(worker.observation_load for worker in schedule.workers) - min(
        worker.observation_load for worker in schedule.workers
    ) <= 10


def test_committed_v2_has_exact_stratum_anchor_and_terminal_topology() -> None:
    manifest = json.loads(
        (ROOT / "data/manifests/set_utility_freeze_b_v2_terminal_index_repair.json").read_text()
    )
    assignments, queries = validate_freeze_query_topology(
        manifest["assignments"],
        manifest["query_states"],
        anchor_step_by_stratum=manifest["repair"]["stratum_anchor_decision_step"],
    )
    assert len(assignments) == 1200
    assert len(queries) == 1200
    assert {tuple(query["query_kind"] for query in values) for values in queries.values()} == {
        ("stratum_anchor", "terminal")
    }


def test_query_prefix_is_exact_and_future_history_is_rejected() -> None:
    trajectory, query = _query_payload()
    plan = build_full_reference_prompt_plan(trajectory, query, (1, 2, 3, 4, 5, 6))
    assert plan.summary_event_step_ids == tuple(range(1, 8))
    assert plan.high_fidelity_history_event_step_ids == (1, 2, 3, 4, 5, 6)

    leaked = dict(trajectory)
    leaked["history_events"] = [*trajectory["history_events"], {
        "event_step_id": 8,
        "low_fidelity_summary": _summary(8),
        "high_fidelity_observation_ref": "images/8.png",
    }]
    with pytest.raises(ValueError, match="exactly its visible history prefix"):
        build_full_reference_prompt_plan(leaked, query, (1, 2, 3, 4, 5, 6))


def test_processor_only_freeze_drops_one_oldest_until_full_reference_fits() -> None:
    trajectory, query = _query_payload()
    observed: list[tuple[int, ...]] = []

    def length(plan: object) -> int:
        candidate_ids = plan.restored_event_step_ids
        observed.append(candidate_ids)
        return 33_000 if len(candidate_ids) == 6 else 32_000

    record = freeze_query_candidates(
        trajectory,
        query,
        processor_only_length=length,
    )

    assert observed == [(1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6)]
    assert record.candidate_context.candidate_event_step_ids == (2, 3, 4, 5, 6)
    assert record.candidate_context.dropped_event_step_ids == (1,)
    assert record.candidate_context.processor_input_token_count + 256 <= 32_768
    assert FrozenQueryCandidateRecord.from_payload(record.to_payload()) == record


def test_processor_only_freeze_fails_closed_below_four_candidates() -> None:
    trajectory, query = _query_payload(candidate_count=4)

    def length(plan: object) -> int:
        return 33_000 if len(plan.restored_event_step_ids) >= 4 else 32_000

    with pytest.raises(CandidateMinimumViolation) as caught:
        freeze_query_candidates(trajectory, query, processor_only_length=length)
    assert [attempt.candidate_event_step_ids for attempt in caught.value.attempts] == [
        (1, 2, 3, 4)
    ]


def test_post_freeze_schedule_materializes_exact_subsets_and_four_workers() -> None:
    records = []
    for index in range(4):
        trajectory, query = _query_payload(candidate_count=4)
        trajectory["trajectory_id"] = f"trajectory-{index}"
        trajectory["source_id"] = f"trajectory-{index}"
        query["state_id"] = f"state-{index}"
        records.append(
            freeze_query_candidates(
                trajectory,
                query,
                processor_only_length=lambda _: 1000,
            )
        )
    schedule = build_final_label_schedule(records)
    payload = label_schedule_payload(schedule)

    assert payload["worker_count"] == 4
    assert payload["operations"]["raw_label_rows"] == 44
    assert all(len(state["subsets"]) == 11 for state in payload["states"])
    assert sum(len(worker["state_ids"]) for worker in payload["workers"]) == 4
    assert aggregate_freeze_summary(records)["state_count"] == 4


def test_write_once_is_atomic_resumable_and_refuses_drift(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = canonical_json_bytes({"status": "ok"})
    assert write_once_or_verify(path, payload) == sha256_bytes(payload)
    assert write_once_or_verify(path, payload) == sha256_bytes(payload)
    with pytest.raises(FileExistsError):
        write_once_or_verify(path, b"different")


def test_processor_only_guard_rejects_modeling_module(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "transformers.models.qwen3_vl.modeling_qwen3_vl"
    monkeypatch.setitem(sys.modules, key, object())
    with pytest.raises(RuntimeError, match="forbidden modeling modules"):
        assert_processor_only_import_state()


def test_formal_runner_ast_allows_only_auto_processor_pretrained_loader() -> None:
    audit = validate_processor_only_source(
        ROOT / "code/scripts/run_set_utility_processor_freeze.py"
    )
    assert audit["from_pretrained_loaders"] == ["AutoProcessor"]
    assert audit["forbidden_model_symbol_count"] == 0


def test_processor_snapshot_verifier_hashes_files_without_loading_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    names = [
        "config.json",
        "chat_template.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ]
    records = []
    for index, name in enumerate(names):
        payload = (
            canonical_json_bytes({"text_config": {"max_position_embeddings": 32768}})
            if name == "config.json"
            else f"fixture-{index}".encode()
        )
        (model_dir / name).write_bytes(payload)
        records.append(
            {"path": name, "size": len(payload), "sha256": sha256_bytes(payload)}
        )
    manifest = {
        "repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
        "revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
        "files": records,
    }
    manifest_path = tmp_path / "snapshot.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest, pretty=True))
    (model_dir / ".snapshot.json").write_bytes(canonical_json_bytes(manifest, pretty=True))

    identity = verify_processor_only_snapshot(
        model_dir=model_dir,
        snapshot_manifest=manifest_path,
    )
    assert identity["verified_file_count"] == 14
    assert identity["maximum_context_tokens"] == 32768
    assert identity["model_or_policy_loaded"] is False
