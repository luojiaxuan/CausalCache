from __future__ import annotations

import io
import json
import os
import tarfile
from collections import Counter
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
    build_final_candidate_schedule,
    build_git_safe_processor_freeze_summary,
    build_prefix_safe_processor_artifact_record,
    build_processor_worker_schedule,
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    inspect_processor_artifact_shard,
    materialize_final_candidate_schedule,
    materialize_processor_artifact_shard,
    sha256_bytes,
)
from causalcache.set_utility_processor_freeze import FrozenQueryCandidateRecord
from causalcache.set_utility_processor_freeze_contract import (
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_PATH_ARGUMENTS,
    REQUIRED_VERSION_ARGUMENTS_BY_PHASE,
    SNAPSHOT_MANIFEST_PATH,
)
from causalcache.set_utility_processor_postflight import (
    COMPLETED_STATUS,
    ExpectedFrozenQuery,
    ProcessorFreezePostflightContext,
    VALIDATION_STATUS,
    validate_completed_processor_freeze_root,
)
from causalcache.set_utility_processor_substrate import (
    QuerySubstrateRef,
    SelectedRowReadPlan,
    SelectedShardRead,
    SelectedTrajectoryAssignment,
    TrajectoryProcessorSubstrate,
)


GIT_REVISION = "d" * 40


def _assignment(index: int) -> SelectedTrajectoryAssignment:
    path = f"mobile/use/train/shard-{index:05d}.parquet"
    return SelectedTrajectoryAssignment(
        trajectory_id=f"trajectory-{index}",
        source_id=f"trajectory-{index}",
        instruction_app_group_sha256="a" * 64,
        role="train",
        candidate_capacity_stratum="decisions_10_17",
        decision_count=10,
        terminal_decision_step_id=11,
        transport_file=path,
        transport_row_index=0,
        p0_selection_sha256="b" * 64,
    )


def _worker_schedule():
    shards = []
    for index, digest_character in enumerate("1234"):
        assignment = _assignment(index)
        source = SourceFileSpec(
            assignment.transport_file,
            100,
            digest_character * 64,
        )
        shards.append(
            SelectedShardRead(
                source_file=source,
                source_row_count=1,
                assignments=(assignment,),
            )
        )
    return build_processor_worker_schedule(
        SelectedRowReadPlan(shards=tuple(shards), assignment_count=4)
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
        executor_result="unknown",
    )


def _trajectory_record(assignment: SelectedTrajectoryAssignment):
    history = tuple(
        UtilityHistoryEvent(
            event_step_id=step,
            low_fidelity_summary=_low(step),
            high_fidelity_observation_ref=(
                f"images/{assignment.trajectory_id}/obs-{step}.png"
            ),
        )
        for step in range(1, assignment.decision_count + 1)
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
        for step in range(1, assignment.decision_count + 1)
    )
    queries = tuple(
        QuerySubstrateRef(
            state_id=f"{assignment.trajectory_id}:decision:{decision_step:03d}",
            query_kind=query_kind,
            decision_step_id=decision_step,
            current_equivalent_event_step_id=decision_step - 1,
            initial_candidate_event_step_ids=tuple(range(1, decision_step - 1))[
                -16:
            ],
            maximum_labeled_cardinality=2,
            current_observation_ref=(
                f"images/{assignment.trajectory_id}/obs-{decision_step - 1}.png"
            ),
        )
        for query_kind, decision_step in (("stratum_anchor", 10), ("terminal", 11))
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
        f"images/{assignment.trajectory_id}/obs-{step}.png": f"image-{step}".encode()
        for step in range(assignment.decision_count + 1)
    }
    ocr = {}
    for step, path in enumerate(sorted(images)):
        ocr[path] = {
            "image_member_path": path,
            "source_format": "PNG",
            "source_mode": "RGB" if step % 2 == 0 else "RGBA",
        }
    return build_prefix_safe_processor_artifact_record(
        substrate,
        image_payloads=images,
        ocr_records_by_path=ocr,
    )


def _candidate(query: ExpectedFrozenQuery) -> FrozenQueryCandidateRecord:
    initial = query.initial_candidate_event_step_ids
    attempt = CandidateLengthAttempt(
        candidate_event_step_ids=initial,
        processor_input_token_count=32_000,
    )
    return FrozenQueryCandidateRecord(
        state_id=query.state_id,
        trajectory_id=query.trajectory_id,
        source_id=query.source_id,
        role=query.role,
        query_kind=query.query_kind,
        decision_step_id=query.decision_step_id,
        maximum_labeled_cardinality=query.maximum_labeled_cardinality,
        candidate_context=FrozenCandidateContext(
            initial_candidate_event_step_ids=initial,
            candidate_event_step_ids=initial,
            dropped_event_step_ids=(),
            processor_input_token_count=32_000,
            reserved_action_tokens=256,
            context_limit=32_768,
            attempts=(attempt,),
        ),
    )


def _expected_queries(schedule) -> dict[str, ExpectedFrozenQuery]:
    result = {}
    for worker in schedule.workers:
        for item in worker.trajectories:
            for query_kind, decision_step in (
                ("stratum_anchor", 10),
                ("terminal", 11),
            ):
                state_id = f"{item.trajectory_id}:decision:{decision_step:03d}"
                result[state_id] = ExpectedFrozenQuery(
                    state_id=state_id,
                    trajectory_id=item.trajectory_id,
                    source_id=item.trajectory_id,
                    role=item.role,
                    query_kind=query_kind,
                    decision_step_id=decision_step,
                    maximum_labeled_cardinality=2,
                    initial_candidate_event_step_ids=tuple(
                        range(1, decision_step - 1)
                    )[-16:],
                )
    return result


def _runtime_cli(context, output_root: Path) -> dict[str, str]:
    keys = {
        item.removeprefix("--").replace("-", "_")
        for item in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }
    keys.update(
        item.removeprefix("--").replace("-", "_")
        for values in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.values()
        for item in values
    )
    runtime = {key: "fixture" for key in keys}
    for key in {
        item.removeprefix("--").replace("-", "_")
        for item in REQUIRED_PATH_ARGUMENTS
    }:
        runtime[key] = str((output_root.parent / key).resolve())
    runtime.update(
        {
            "repository_root": str(context.repository_root.resolve()),
            "execution_config": str(context.execution_config_path.resolve()),
            "snapshot_manifest": str(
                (context.repository_root / SNAPSHOT_MANIFEST_PATH).resolve()
            ),
            "output_root": str(output_root.resolve()),
            "git_revision": GIT_REVISION,
            "worker_count": "4",
            "container_image_digest": f"sha256:{'e' * 64}",
        }
    )
    return runtime


def _terminal_format_tally(record) -> dict[str, int]:
    terminal = next(query for query in record.queries if query.query_kind == "terminal")
    tally = Counter(
        f"{ocr['source_format']}:{ocr['source_mode']}"
        for ocr in terminal.ocr_records_by_path.values()
    )
    return dict(sorted(tally.items()))


def _write_root(tmp_path: Path):
    output_root = tmp_path / "completed"
    for name in ("candidate-parts", "logs", "receipts", "substrate"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    schedule = _worker_schedule()
    expected_queries = _expected_queries(schedule)
    context = ProcessorFreezePostflightContext(
        repository_root=(tmp_path / "repo").resolve(),
        execution_config_path=(tmp_path / "repo" / "execution.json").resolve(),
        execution_config_sha256="f" * 64,
        expected_git_revision=GIT_REVISION,
        worker_schedule=schedule,
        expected_queries=expected_queries,
        source_audit={"status": "fixture-source-audit"},
        snapshot_identity_without_model_dir={
            "file_inventory_sha256": "8" * 64,
            "maximum_context_tokens": 32_768,
            "model_or_policy_loaded": False,
            "model_repo": "fixture/model",
            "model_revision": "fixture-revision",
            "snapshot_manifest_sha256": "9" * 64,
            "verified_file_count": 1,
            "verified_total_bytes": 1,
        },
        ocr_backend_config_sha256="7" * 64,
        ocr_runtime_identity={"runtime": "fixture"},
    )
    records_by_worker = []
    descriptors = []
    tallies = []
    candidates = []
    for worker in schedule.workers:
        records = tuple(
            _trajectory_record(_assignment(int(item.trajectory_id.rsplit("-", 1)[1])))
            for item in worker.trajectories
        )
        records_by_worker.append(records)
        descriptor = materialize_processor_artifact_shard(
            output_root / "substrate",
            worker=worker,
            records=records,
        )
        descriptors.append(descriptor)
        tally: Counter[str] = Counter()
        for record in records:
            tally.update(_terminal_format_tally(record))
        tallies.append(dict(sorted(tally.items())))
        worker_candidates = sorted(
            (
                _candidate(query)
                for query in expected_queries.values()
                if query.trajectory_id
                in {item.trajectory_id for item in worker.trajectories}
            ),
            key=lambda item: item.state_id,
        )
        candidates.extend(worker_candidates)
        candidate_part = (
            output_root
            / "candidate-parts"
            / f"worker-{worker.worker_index:02d}.jsonl"
        )
        candidate_part.write_bytes(
            b"".join(
                canonical_json_bytes(record.to_payload()) + b"\n"
                for record in worker_candidates
            )
        )
        for phase in ("ocr", "processor"):
            log = (
                output_root
                / "logs"
                / f"{phase}-worker-{worker.worker_index:02d}.log"
            )
            log.write_bytes(b"")
    candidate_schedule = build_final_candidate_schedule(candidates)
    candidate_descriptor = materialize_final_candidate_schedule(
        output_root,
        schedule=candidate_schedule,
    )
    for worker_index, (descriptor, tally) in enumerate(
        zip(descriptors, tallies, strict=True)
    ):
        receipt = {
            "backend_config_sha256": context.ocr_backend_config_sha256,
            "format_tally": tally,
            "ocr_runtime_identity": context.ocr_runtime_identity,
            "shard": descriptor.to_payload(),
            "worker_index": worker_index,
        }
        (output_root / "receipts" / f"ocr-worker-{worker_index:02d}.json").write_bytes(
            canonical_json_bytes(receipt)
        )
    runtime = _runtime_cli(context, output_root)
    run_identity = {
        "config_sha256": context.execution_config_sha256,
        "git_revision": GIT_REVISION,
        "runtime_cli": runtime,
        "source_audit": context.source_audit,
        "snapshot": {
            "model_dir": str(Path(runtime["model_dir"]).resolve()),
            **context.snapshot_identity_without_model_dir,
        },
        "worker_execution_sha256": schedule.execution_sha256,
    }
    (output_root / "run-identity.json").write_bytes(
        canonical_json_bytes(run_identity)
    )
    summary = build_git_safe_processor_freeze_summary(
        worker_schedule=schedule,
        artifact_shards=descriptors,
        candidate_schedule=candidate_schedule,
        candidate_artifact=candidate_descriptor,
    )
    manifest = {
        "contains_model_or_policy_output": False,
        "contains_restoration_labels": False,
        "execution_config_sha256": context.execution_config_sha256,
        "git_revision": GIT_REVISION,
        "processor_freeze_summary": summary,
        "run_identity_sha256": sha256_bytes(canonical_json_bytes(run_identity)),
        "status": COMPLETED_STATUS,
    }
    (output_root / "manifest.json").write_bytes(
        canonical_pretty_json_bytes(manifest)
    )
    return output_root, context, candidate_schedule, tallies


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, bytes | None], ...]:
    result = []
    for path in sorted(root.rglob("*")):
        result.append(
            (
                path.relative_to(root).as_posix(),
                path.lstat().st_mode,
                path.read_bytes() if path.is_file() and not path.is_symlink() else None,
            )
        )
    return tuple(result)


def test_postflight_accepts_mixed_ocr_formats_and_is_read_only(tmp_path: Path) -> None:
    root, context, candidate_schedule, tallies = _write_root(tmp_path)
    before = _tree_snapshot(root)

    result = validate_completed_processor_freeze_root(root, context=context)

    assert result["status"] == VALIDATION_STATUS
    assert result["operation_budget"]["raw_label_rows"] == (
        candidate_schedule.label_schedule.operations.raw_label_rows
    )
    assert all(set(tally) == {"PNG:RGB", "PNG:RGBA"} for tally in tallies)
    assert _tree_snapshot(root) == before


@pytest.mark.parametrize("partial_path", [".manifest.json.partial", "logs/x.partial"])
def test_postflight_rejects_partial_or_extra_files(
    tmp_path: Path,
    partial_path: str,
) -> None:
    root, context, _, _ = _write_root(tmp_path)
    path = root / partial_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"partial")

    with pytest.raises(ValueError, match="inventory drifted"):
        validate_completed_processor_freeze_root(root, context=context)


def test_postflight_rejects_symlinked_expected_file(tmp_path: Path) -> None:
    root, context, _, _ = _write_root(tmp_path)
    victim = root / "logs" / "ocr-worker-00.log"
    victim.unlink()
    os.symlink(root / "logs" / "ocr-worker-01.log", victim)

    with pytest.raises(ValueError, match="real regular file"):
        validate_completed_processor_freeze_root(root, context=context)


def test_postflight_rejects_candidate_schedule_tamper(tmp_path: Path) -> None:
    root, context, _, _ = _write_root(tmp_path)
    path = root / "processor-candidate-freeze-schedule.json"
    payload = json.loads(path.read_text())
    payload["contains_generated_labels"] = True
    path.chmod(0o644)
    path.write_bytes(canonical_pretty_json_bytes(payload))

    with pytest.raises(ValueError, match="canonical reconstruction"):
        validate_completed_processor_freeze_root(root, context=context)


def test_postflight_rejects_noncanonical_candidate_part(tmp_path: Path) -> None:
    root, context, _, _ = _write_root(tmp_path)
    path = root / "candidate-parts" / "worker-00.jsonl"
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    path.write_bytes(
        b"".join(json.dumps(record).encode() + b"\n" for record in records)
    )

    with pytest.raises(ValueError, match="not canonical"):
        validate_completed_processor_freeze_root(root, context=context)


def test_postflight_rejects_receipt_format_tally_not_backed_by_ocr(
    tmp_path: Path,
) -> None:
    root, context, _, _ = _write_root(tmp_path)
    path = root / "receipts" / "ocr-worker-00.json"
    receipt = json.loads(path.read_text())
    receipt["format_tally"] = {
        "PNG:RGBA": receipt["shard"]["observation_count"]
    }
    path.write_bytes(canonical_json_bytes(receipt))

    with pytest.raises(ValueError, match="identity or count drifted"):
        validate_completed_processor_freeze_root(root, context=context)


def test_postflight_rejects_run_git_identity_tamper(tmp_path: Path) -> None:
    root, context, _, _ = _write_root(tmp_path)
    identity_path = root / "run-identity.json"
    identity = json.loads(identity_path.read_text())
    identity["git_revision"] = "a" * 40
    identity["runtime_cli"]["git_revision"] = "a" * 40
    identity_path.write_bytes(canonical_json_bytes(identity))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["run_identity_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    manifest_path.write_bytes(canonical_pretty_json_bytes(manifest))

    with pytest.raises(ValueError, match="fixed runtime values drifted"):
        validate_completed_processor_freeze_root(root, context=context)


def _rewrite_shard_manifest_noncanonically(path: Path) -> None:
    members = []
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            source = archive.extractfile(member)
            if source is None:
                raise AssertionError("fixture tar member is not readable")
            payload = source.read()
            if member.name == "shard-manifest.json":
                payload = json.dumps(json.loads(payload), indent=1).encode()
            members.append((member.name, payload))
    replacement = path.with_suffix(".replacement")
    with tarfile.open(replacement, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    replacement.replace(path)


def test_postflight_rejects_noncanonical_tar_json_even_when_semantics_match(
    tmp_path: Path,
) -> None:
    root, context, candidate_schedule, tallies = _write_root(tmp_path)
    shard = root / "substrate" / "processor-substrate-worker-00.tar"
    _rewrite_shard_manifest_noncanonically(shard)
    descriptors = tuple(
        inspect_processor_artifact_shard(
            root / "substrate" / worker.filename,
            expected_worker=worker,
        )
        for worker in context.worker_schedule.workers
    )
    for worker_index, (descriptor, tally) in enumerate(
        zip(descriptors, tallies, strict=True)
    ):
        receipt = {
            "backend_config_sha256": context.ocr_backend_config_sha256,
            "format_tally": tally,
            "ocr_runtime_identity": context.ocr_runtime_identity,
            "shard": descriptor.to_payload(),
            "worker_index": worker_index,
        }
        (root / "receipts" / f"ocr-worker-{worker_index:02d}.json").write_bytes(
            canonical_json_bytes(receipt)
        )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["processor_freeze_summary"] = build_git_safe_processor_freeze_summary(
        worker_schedule=context.worker_schedule,
        artifact_shards=descriptors,
        candidate_schedule=candidate_schedule,
        candidate_artifact=(
            CandidateScheduleDescriptor(
                **manifest["processor_freeze_summary"]["candidate_artifact"]
            )
        ),
    )
    manifest_path.write_bytes(canonical_pretty_json_bytes(manifest))

    with pytest.raises(ValueError, match="not canonical compact JSON"):
        validate_completed_processor_freeze_root(root, context=context)
