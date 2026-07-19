from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest

from causalcache import set_utility_label_inputs as label_inputs
from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_label_inputs import (
    LabelExecutionPartition,
    build_joined_utility_query_input,
    read_allowlisted_processor_queries,
    read_allowlisted_train_processor_queries,
)
from causalcache.set_utility_label_producer import (
    CandidateLengthAttempt,
    FrozenCandidateContext,
    UtilityHistoryEvent,
)
from causalcache.set_utility_processor_artifacts import (
    ProcessorWorkerShard,
    TrajectoryWorkItem,
    build_prefix_safe_processor_artifact_record,
    materialize_processor_artifact_shard,
)
from causalcache.set_utility_processor_freeze import FrozenQueryCandidateRecord
from causalcache.set_utility_processor_substrate import (
    QuerySubstrateRef,
    SelectedTrajectoryAssignment,
    TrajectoryProcessorSubstrate,
)


def _assignment(
    trajectory_id: str,
    *,
    role: str,
    transport_file: str,
    transport_row_index: int,
) -> SelectedTrajectoryAssignment:
    return SelectedTrajectoryAssignment(
        trajectory_id=trajectory_id,
        source_id=trajectory_id,
        instruction_app_group_sha256=(
            {"train": "a", "tune": "b", "evaluation": "c"}[role] * 64
        ),
        role=role,
        candidate_capacity_stratum="decisions_10_17",
        decision_count=10,
        terminal_decision_step_id=11,
        transport_file=transport_file,
        transport_row_index=transport_row_index,
        p0_selection_sha256="d" * 64,
    )


def _low(step: int) -> LowFidelityEventV2:
    return LowFidelityEventV2(
        step_id=step,
        action_type="wait",
        action_argument="wait",
        foreground_app="fixture",
        screen_text_added=(f"screen-{step}",),
        screen_text_removed=(),
        screen_change="low",
        executor_result="accepted",
    )


def _processor_record(assignment: SelectedTrajectoryAssignment):
    history = tuple(
        UtilityHistoryEvent(
            event_step_id=step,
            low_fidelity_summary=_low(step),
            high_fidelity_observation_ref=(
                f"images/{assignment.trajectory_id}/obs-{step}.png"
            ),
        )
        for step in range(1, 11)
    )
    derived = tuple(
        {
            "step_id": step,
            "observation_before_path": (
                f"images/{assignment.trajectory_id}/obs-{step - 1}.png"
            ),
            "observation_after_path": (
                f"images/{assignment.trajectory_id}/obs-{step}.png"
            ),
        }
        for step in range(1, 11)
    )
    queries = tuple(
        QuerySubstrateRef(
            state_id=f"{assignment.trajectory_id}:decision:{decision:03d}",
            query_kind=kind,
            decision_step_id=decision,
            current_equivalent_event_step_id=decision - 1,
            initial_candidate_event_step_ids=(
                tuple(range(1, decision - 1))[-16:]
            ),
            maximum_labeled_cardinality=2,
            current_observation_ref=(
                f"images/{assignment.trajectory_id}/obs-{decision - 1}.png"
            ),
        )
        for kind, decision in (("stratum_anchor", 10), ("terminal", 11))
    )
    substrate = TrajectoryProcessorSubstrate(
        assignment=assignment,
        transport_file_sha256="e" * 64,
        task_instruction=f"Complete {assignment.trajectory_id}",
        history_events=history,
        derived_event_records=derived,
        query_states=queries,
    )
    images = {
        f"images/{assignment.trajectory_id}/obs-{step}.png": (
            f"image-{assignment.trajectory_id}-{step}".encode()
        )
        for step in range(11)
    }
    ocr = {
        path: {
            "image_member_path": path,
            "full_spatial_tokens": [path],
            "nodes": [],
        }
        for path in sorted(images)
    }
    return build_prefix_safe_processor_artifact_record(
        substrate,
        image_payloads=images,
        ocr_records_by_path=ocr,
    )


def _mixed_worker():
    assignments = (
        _assignment(
            "tune-first",
            role="tune",
            transport_file="mobile/use/train/a.parquet",
            transport_row_index=0,
        ),
        _assignment(
            "train-middle",
            role="train",
            transport_file="mobile/use/train/b.parquet",
            transport_row_index=0,
        ),
        _assignment(
            "evaluation-last",
            role="evaluation",
            transport_file="mobile/use/train/c.parquet",
            transport_row_index=0,
        ),
    )
    items = tuple(TrajectoryWorkItem.from_assignment(item) for item in assignments)
    worker = ProcessorWorkerShard(
        worker_index=2,
        trajectories=items,
        transport_files=tuple(item.transport_file for item in assignments),
        observation_count=sum(item.observation_count for item in items),
    )
    return assignments, worker


def _materialize_mixed_shard(tmp_path: Path):
    assignments, worker = _mixed_worker()
    descriptor = materialize_processor_artifact_shard(
        tmp_path / "valid",
        worker=worker,
        records=tuple(_processor_record(item) for item in assignments),
    )
    return assignments, worker, tmp_path / "valid" / descriptor.filename


def _poison_non_train_query_json(source: Path, destination: Path) -> None:
    members: list[tuple[str, bytes, int]] = []
    with tarfile.open(source, mode="r:") as archive:
        for member in archive:
            handle = archive.extractfile(member)
            if handle is None:
                raise AssertionError("fixture tar contains a non-file member")
            payload = handle.read()
            if (
                member.name.endswith("/record.json")
                and (
                    member.name.startswith("trajectories/0000/")
                    or member.name.startswith("trajectories/0002/")
                )
            ):
                payload = b"\xffNON_TRAIN_JSON_MUST_NOT_BE_DECODED"
            members.append((member.name, payload, member.mode))
    with tarfile.open(destination, mode="w:", format=tarfile.PAX_FORMAT) as archive:
        for name, payload, mode in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = mode
            archive.addfile(info, io.BytesIO(payload))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_anchor_candidate(
    assignment: SelectedTrajectoryAssignment,
) -> FrozenQueryCandidateRecord:
    initial = tuple(range(1, 9))
    attempts = tuple(
        CandidateLengthAttempt(
            candidate_event_step_ids=initial[dropped:],
            processor_input_token_count=(32_000 if dropped == 4 else 32_600),
        )
        for dropped in range(5)
    )
    return FrozenQueryCandidateRecord(
        state_id=f"{assignment.trajectory_id}:decision:010",
        trajectory_id=assignment.trajectory_id,
        source_id=assignment.source_id,
        role=assignment.role,
        query_kind="stratum_anchor",
        decision_step_id=10,
        maximum_labeled_cardinality=2,
        candidate_context=FrozenCandidateContext(
            initial_candidate_event_step_ids=initial,
            candidate_event_step_ids=(5, 6, 7, 8),
            dropped_event_step_ids=(1, 2, 3, 4),
            processor_input_token_count=32_000,
            reserved_action_tokens=256,
            context_limit=32_768,
            attempts=attempts,
        ),
    )


def test_train_allowlist_skips_malformed_tune_and_evaluation_records(
    tmp_path: Path,
) -> None:
    _, worker, valid = _materialize_mixed_shard(tmp_path)
    poisoned = tmp_path / "poisoned.tar"
    _poison_non_train_query_json(valid, poisoned)
    state_id = "train-middle:decision:010"

    selected = read_allowlisted_train_processor_queries(
        poisoned,
        expected_worker=worker,
        expected_artifact_sha256=_sha256(poisoned),
        state_ids=(state_id,),
    )

    assert len(selected) == 1
    assert selected[0].query.state_id == state_id
    assert selected[0].query.role == "train"
    assert selected[0].processor_worker_index == 2
    assert selected[0].role_partition == "train"


def test_train_allowlist_rejects_non_train_identity_before_record_decode(
    tmp_path: Path,
) -> None:
    _, worker, valid = _materialize_mixed_shard(tmp_path)
    poisoned = tmp_path / "poisoned.tar"
    _poison_non_train_query_json(valid, poisoned)

    with pytest.raises(ValueError, match="allowed role partitions"):
        read_allowlisted_train_processor_queries(
            poisoned,
            expected_worker=worker,
            expected_artifact_sha256=_sha256(poisoned),
            state_ids=("tune-first:decision:010",),
        )


def test_role_aware_allowlist_reads_tune_and_evaluation_records(
    tmp_path: Path,
) -> None:
    _, worker, artifact = _materialize_mixed_shard(tmp_path)
    selected = read_allowlisted_processor_queries(
        artifact,
        expected_worker=worker,
        expected_artifact_sha256=_sha256(artifact),
        state_ids=(
            "tune-first:decision:010",
            "evaluation-last:decision:010",
        ),
        allowed_roles=("tune", "evaluation"),
    )

    assert tuple(item.query.role for item in selected) == ("evaluation", "tune")


def test_train_allowlist_rejects_symlink_artifact(tmp_path: Path) -> None:
    _, worker, artifact = _materialize_mixed_shard(tmp_path)
    link = tmp_path / "artifact-link.tar"
    link.symlink_to(artifact)

    with pytest.raises(ValueError, match="opened no-follow"):
        read_allowlisted_train_processor_queries(
            link,
            expected_worker=worker,
            expected_artifact_sha256=_sha256(artifact),
            state_ids=("train-middle:decision:010",),
        )


def test_train_allowlist_rejects_path_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, worker, artifact = _materialize_mixed_shard(tmp_path)
    replacement = tmp_path / "same-bytes-new-inode.tar"
    replacement.write_bytes(artifact.read_bytes())
    expected_sha = _sha256(artifact)
    original = label_inputs._artifacts._strict_json
    swapped = False

    def replace_after_hash(payload: bytes, *, label: str):
        nonlocal swapped
        if not swapped and label == "artifact manifest":
            os.replace(replacement, artifact)
            swapped = True
        return original(payload, label=label)

    monkeypatch.setattr(label_inputs._artifacts, "_strict_json", replace_after_hash)
    with pytest.raises(ValueError, match="identity changed"):
        read_allowlisted_train_processor_queries(
            artifact,
            expected_worker=worker,
            expected_artifact_sha256=expected_sha,
            state_ids=("train-middle:decision:010",),
        )
    assert swapped is True


def test_processor_query_strict_join_preserves_execution_and_role_partitions(
    tmp_path: Path,
) -> None:
    assignments, worker, artifact = _materialize_mixed_shard(tmp_path)
    train = assignments[1]
    state_id = "train-middle:decision:010"
    selected = read_allowlisted_train_processor_queries(
        artifact,
        expected_worker=worker,
        expected_artifact_sha256=_sha256(artifact),
        state_ids=(state_id,),
    )[0]
    candidate = _frozen_anchor_candidate(train)

    joined = build_joined_utility_query_input(
        selected,
        candidate=candidate,
        assignment=train,
        execution=LabelExecutionPartition(
            state_id=state_id,
            worker_index=3,
            role_partition="train",
        ),
        request_manifest_sha256="f" * 64,
    )

    assert joined.query.state_id == state_id
    assert joined.query.source_id == train.source_id
    assert joined.query.candidate_event_step_ids == (5, 6, 7, 8)
    assert joined.query.source_artifact_sha256 == _sha256(artifact)
    assert joined.query.request_manifest_sha256 == "f" * 64
    assert joined.query.slice_witness_sha256 == selected.query_record_sha256
    assert joined.processor_worker_index == 2
    assert joined.execution_worker_index == 3
    assert joined.role_partition == "train"
    assert selected.image_payloads is selected.query.image_payloads
    assert joined.image_payloads is selected.image_payloads
    assert joined.image_payloads == selected.query.image_payloads


def test_processor_query_join_rejects_role_partition_drift(tmp_path: Path) -> None:
    assignments, worker, artifact = _materialize_mixed_shard(tmp_path)
    train = assignments[1]
    state_id = "train-middle:decision:010"
    selected = read_allowlisted_train_processor_queries(
        artifact,
        expected_worker=worker,
        expected_artifact_sha256=_sha256(artifact),
        state_ids=(state_id,),
    )[0]

    with pytest.raises(ValueError, match="execution partition"):
        build_joined_utility_query_input(
            selected,
            candidate=_frozen_anchor_candidate(train),
            assignment=train,
            execution=LabelExecutionPartition(
                state_id=state_id,
                worker_index=3,
                role_partition="tune",
            ),
            request_manifest_sha256="f" * 64,
        )
