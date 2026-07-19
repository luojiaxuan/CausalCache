from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_label_producer import (
    CandidateLengthAttempt,
    FrozenCandidateContext,
    UtilityHistoryEvent,
)
from causalcache.set_utility_processor_artifacts import (
    CandidateScheduleDescriptor,
    ProcessorArtifactShardDescriptor,
    build_final_candidate_schedule,
    build_git_safe_processor_freeze_summary,
    build_prefix_safe_processor_artifact_record,
    build_processor_worker_schedule,
    inspect_processor_artifact_shard,
    iter_processor_query_artifact_records,
    materialize_final_candidate_schedule,
    materialize_processor_artifact_shard,
)
from causalcache.set_utility_processor_freeze import FrozenQueryCandidateRecord
from causalcache.set_utility_processor_substrate import (
    QuerySubstrateRef,
    SelectedRowReadPlan,
    SelectedShardRead,
    SelectedTrajectoryAssignment,
    TrajectoryProcessorSubstrate,
)


def _assignment(
    trajectory_id: str,
    *,
    transport_file: str,
    transport_row_index: int,
    decision_count: int = 10,
) -> SelectedTrajectoryAssignment:
    return SelectedTrajectoryAssignment(
        trajectory_id=trajectory_id,
        source_id=trajectory_id,
        instruction_app_group_sha256="a" * 64,
        role="train",
        candidate_capacity_stratum="decisions_10_17",
        decision_count=decision_count,
        terminal_decision_step_id=decision_count + 1,
        transport_file=transport_file,
        transport_row_index=transport_row_index,
        p0_selection_sha256="b" * 64,
    )


def _read_plan() -> SelectedRowReadPlan:
    shard_specs = tuple(
        SourceFileSpec(f"mobile/use/train/{name}.parquet", 100, name * 64)
        for name in ("a", "b", "c", "d", "e")
    )
    shards = []
    for index, spec in enumerate(shard_specs):
        count = 2 if index == 0 else 1
        assignments = tuple(
            _assignment(
                f"trajectory-{index}-{row}",
                transport_file=spec.transport_file,
                transport_row_index=row,
            )
            for row in range(count)
        )
        shards.append(
            SelectedShardRead(
                source_file=spec,
                source_row_count=count,
                assignments=assignments,
            )
        )
    return SelectedRowReadPlan(shards=tuple(shards), assignment_count=6)


def _low(step: int) -> LowFidelityEventV2:
    return LowFidelityEventV2(
        step_id=step,
        action_type="wait",
        action_argument="wait",
        foreground_app="fixture",
        screen_text_added=(
            "FUTURE_HISTORY_SECRET" if step == 10 else f"screen-{step}",
        ),
        screen_text_removed=(),
        screen_change="low",
        executor_result="unknown",
    )


def _substrate(assignment: SelectedTrajectoryAssignment) -> tuple[
    TrajectoryProcessorSubstrate,
    dict[str, bytes],
    dict[str, dict],
]:
    history = tuple(
        UtilityHistoryEvent(
            event_step_id=step,
            low_fidelity_summary=_low(step),
            high_fidelity_observation_ref=f"images/{assignment.trajectory_id}/obs-{step}.png",
        )
        for step in range(1, assignment.decision_count + 1)
    )
    derived = tuple(
        {
            "step_id": step,
            "observation_before_path": f"images/{assignment.trajectory_id}/obs-{step - 1}.png",
            "observation_after_path": f"images/{assignment.trajectory_id}/obs-{step}.png",
            "executed_action": {
                "target": "RAW_TARGET_SECRET" if step == 10 else "discarded"
            },
            "source_tool_call": {"terminal_outcome": "RAW_OUTCOME_SECRET"},
        }
        for step in range(1, assignment.decision_count + 1)
    )
    anchor_current = 9
    terminal_current = assignment.decision_count
    queries = tuple(
        QuerySubstrateRef(
            state_id=f"{assignment.trajectory_id}:decision:{step + 1:03d}",
            query_kind=kind,
            decision_step_id=step + 1,
            current_equivalent_event_step_id=step,
            initial_candidate_event_step_ids=tuple(range(1, step))[-16:],
            maximum_labeled_cardinality=2,
            current_observation_ref=history[step - 1].high_fidelity_observation_ref,
        )
        for kind, step in (
            ("stratum_anchor", anchor_current),
            ("terminal", terminal_current),
        )
    )
    substrate = TrajectoryProcessorSubstrate(
        assignment=assignment,
        transport_file_sha256="c" * 64,
        task_instruction="Do the fixture task",
        history_events=history,
        derived_event_records=derived,
        query_states=queries,
    )
    images = {
        f"images/{assignment.trajectory_id}/obs-{step}.png": (
            b"PNG-FUTURE-IMAGE-SECRET" if step == 10 else f"PNG-{step}".encode()
        )
        for step in range(assignment.decision_count + 1)
    }
    ocr = {
        path: {
            "image_member_path": path,
            "full_spatial_tokens": [
                "FUTURE_OCR_SECRET" if path.endswith("obs-10.png") else path
            ],
            "nodes": [{"score": 0.95}],
        }
        for path in sorted(images)
    }
    return substrate, images, ocr


def test_whole_shard_lpt_preserves_stream_order_and_never_splits_a_file() -> None:
    plan = _read_plan()
    left = build_processor_worker_schedule(plan)
    right = build_processor_worker_schedule(plan)

    assert left == right
    assert left.trajectory_count == 6
    assert left.observation_count == 66
    files = [path for worker in left.workers for path in worker.transport_files]
    assert len(files) == len(set(files)) == 5
    first_file = plan.shards[0].source_file.transport_file
    owner = next(worker for worker in left.workers if first_file in worker.transport_files)
    assert [
        item.transport_row_index
        for item in owner.trajectories
        if item.transport_file == first_file
    ] == [0, 1]
    for worker in left.workers:
        assert tuple(
            (item.transport_file, item.transport_row_index)
            for item in worker.trajectories
        ) == tuple(
            sorted(
                (item.transport_file, item.transport_row_index)
                for item in worker.trajectories
            )
        )


def test_prefix_projection_excludes_anchor_future_but_keeps_exact_ocr_scores() -> None:
    assignment = _assignment(
        "prefix-fixture",
        transport_file="mobile/use/train/a.parquet",
        transport_row_index=0,
    )
    substrate, images, ocr = _substrate(assignment)
    record = build_prefix_safe_processor_artifact_record(
        substrate,
        image_payloads=images,
        ocr_records_by_path=ocr,
    )
    anchor, terminal = record.queries

    assert anchor.query_kind == "stratum_anchor"
    assert [event["event_step_id"] for event in anchor.history_events] == list(
        range(1, 10)
    )
    assert set(anchor.ocr_records_by_path) == {
        f"images/{assignment.trajectory_id}/obs-{step}.png" for step in range(10)
    }
    assert all(
        node["score"] == 0.95
        for ocr_record in anchor.ocr_records_by_path.values()
        for node in ocr_record["nodes"]
    )
    anchor_bytes = json.dumps(
        anchor.to_payload(include_image_inventory=False), ensure_ascii=False
    ).encode()
    assert b"FUTURE_HISTORY_SECRET" not in anchor_bytes
    assert b"FUTURE_OCR_SECRET" not in anchor_bytes
    assert b"FUTURE-IMAGE-SECRET" not in b"".join(anchor.image_payloads.values())
    terminal_bytes = json.dumps(
        terminal.to_payload(include_image_inventory=False), ensure_ascii=False
    ).encode()
    assert b"FUTURE_HISTORY_SECRET" in terminal_bytes
    assert b"FUTURE_OCR_SECRET" in terminal_bytes
    assert b"RAW_TARGET_SECRET" not in terminal_bytes
    assert b"RAW_OUTCOME_SECRET" not in terminal_bytes


def _records_for_worker(worker) -> tuple:
    result = []
    for item in worker.trajectories:
        assignment = _assignment(
            item.trajectory_id,
            transport_file=item.transport_file,
            transport_row_index=item.transport_row_index,
        )
        substrate, images, ocr = _substrate(assignment)
        result.append(
            build_prefix_safe_processor_artifact_record(
                substrate,
                image_payloads=images,
                ocr_records_by_path=ocr,
            )
        )
    return tuple(result)


def test_tar_is_deterministic_atomic_resumable_and_reader_is_prefix_safe(
    tmp_path: Path,
) -> None:
    worker = build_processor_worker_schedule(_read_plan()).workers[0]
    records = _records_for_worker(worker)
    first = materialize_processor_artifact_shard(
        tmp_path / "first",
        worker=worker,
        records=records,
    )
    second = materialize_processor_artifact_shard(
        tmp_path / "second",
        worker=worker,
        records=_records_for_worker(worker),
    )

    assert first.sha256 == second.sha256
    assert first.query_count == 2 * first.trajectory_count
    assert inspect_processor_artifact_shard(
        tmp_path / "first" / first.filename,
        expected_worker=worker,
    ) == first
    with pytest.raises(FileExistsError, match="destination exists"):
        materialize_processor_artifact_shard(
            tmp_path / "first",
            worker=worker,
            records=(),
        )
    assert materialize_processor_artifact_shard(
        tmp_path / "first",
        worker=worker,
        records=(),
        resume=True,
    ) == first
    streamed = tuple(
        iter_processor_query_artifact_records(
            tmp_path / "first" / first.filename,
            expected_worker=worker,
        )
    )
    assert len(streamed) == first.query_count
    anchor = next(record for record in streamed if record.query_kind == "stratum_anchor")
    assert max(event["event_step_id"] for event in anchor.history_events) == 9
    raw_tar = (tmp_path / "first" / first.filename).read_bytes()
    assert b"RAW_TARGET_SECRET" not in raw_tar
    assert b"RAW_OUTCOME_SECRET" not in raw_tar
    assert sorted(path.name for path in (tmp_path / "first").iterdir()) == [
        first.filename
    ]


def _frozen_record(index: int, *, dropped: bool) -> FrozenQueryCandidateRecord:
    initial = (1, 2, 3, 4, 5) if dropped else (1, 2, 3, 4)
    attempts = (
        (
            CandidateLengthAttempt(
                candidate_event_step_ids=initial,
                processor_input_token_count=32_600,
            ),
            CandidateLengthAttempt(
                candidate_event_step_ids=(2, 3, 4, 5),
                processor_input_token_count=32_000,
            ),
        )
        if dropped
        else (
            CandidateLengthAttempt(
                candidate_event_step_ids=initial,
                processor_input_token_count=32_000,
            ),
        )
    )
    final = attempts[-1].candidate_event_step_ids
    return FrozenQueryCandidateRecord(
        state_id=f"state-{index}",
        trajectory_id=f"trajectory-{index}",
        source_id=f"trajectory-{index}",
        role="train",
        query_kind="terminal",
        decision_step_id=11,
        maximum_labeled_cardinality=2,
        candidate_context=FrozenCandidateContext(
            initial_candidate_event_step_ids=initial,
            candidate_event_step_ids=final,
            dropped_event_step_ids=initial[: len(initial) - len(final)],
            processor_input_token_count=32_000,
            reserved_action_tokens=256,
            context_limit=32_768,
            attempts=attempts,
        ),
    )


def test_candidate_schedule_records_attempts_and_exact_R_ops_with_git_safe_summary(
    tmp_path: Path,
) -> None:
    records = tuple(_frozen_record(index, dropped=index % 2 == 0) for index in range(4))
    schedule = build_final_candidate_schedule(records)
    descriptor = materialize_final_candidate_schedule(
        tmp_path,
        schedule=schedule,
    )

    assert schedule.label_schedule.operations.raw_label_rows == 44
    assert schedule.label_schedule.operations.total_model_operations == 60
    payload = json.loads((tmp_path / descriptor.filename).read_text())
    first = payload["candidate_inventory"]["records"][0]
    assert first["dropped_event_step_ids"] == [1]
    assert [attempt["processor_input_token_count"] for attempt in first["attempts"]] == [
        32_600,
        32_000,
    ]
    assert materialize_final_candidate_schedule(
        tmp_path,
        schedule=schedule,
        resume=True,
    ) == descriptor

    worker_schedule = build_processor_worker_schedule(_read_plan())
    shard_descriptors = tuple(
        ProcessorArtifactShardDescriptor(
            worker_index=worker.worker_index,
            filename=f"processor-substrate-worker-{worker.worker_index:02d}.tar",
            sha256=str(worker.worker_index + 1) * 64,
            byte_count=100,
            trajectory_count=len(worker.trajectories),
            query_count=2 * len(worker.trajectories),
            observation_count=worker.observation_count,
            image_member_count=10,
            image_byte_count=100,
            content_inventory_sha256=str(worker.worker_index + 5) * 64,
        )
        for worker in worker_schedule.workers
    )
    summary = build_git_safe_processor_freeze_summary(
        worker_schedule=worker_schedule,
        artifact_shards=shard_descriptors,
        candidate_schedule=schedule,
        candidate_artifact=CandidateScheduleDescriptor(
            filename=descriptor.filename,
            sha256=descriptor.sha256,
            byte_count=descriptor.byte_count,
            state_count=descriptor.state_count,
            subset_forward_count=descriptor.subset_forward_count,
            total_model_operations=descriptor.total_model_operations,
            candidate_inventory_sha256=descriptor.candidate_inventory_sha256,
        ),
    )
    serialized = json.dumps(summary, sort_keys=True)
    assert summary["candidate_freeze"]["operation_budget"]["raw_label_rows"] == 44
    assert "Do the fixture task" not in serialized
    assert "FUTURE_OCR_SECRET" not in serialized
    assert "state-0" not in serialized
