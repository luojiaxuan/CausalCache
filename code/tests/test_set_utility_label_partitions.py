from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from causalcache.set_utility_label_partitions import (
    LABEL_ROLES,
    LabelPartitionLayout,
    LabelShardDescriptor,
    PyArrowLabelShardCodec,
    TrainerLabelInventory,
    WorkerLabelRow,
    read_evaluation_label_dataset,
    read_trainer_label_dataset,
    release_evaluation_label_inventory,
    validate_role_partitioned_label_shards,
    write_role_partitioned_label_shards,
)
from causalcache.set_utility_label_table import SetUtilityDistanceInputRow


class JsonFixtureCodec:
    """Small deterministic codec exercising the same aggregate-shard boundary."""

    def __init__(self) -> None:
        self.decoded_roles: list[str] = []

    def encode(self, rows: Sequence[SetUtilityDistanceInputRow]) -> bytes:
        payload = [
            {
                "split": row.split,
                "state_id": row.state_id,
                "candidate_event_step_ids": list(row.candidate_event_step_ids),
                "maximum_labeled_cardinality": row.maximum_labeled_cardinality,
                "coalition_event_step_ids": list(row.coalition_event_step_ids),
                "distance": row.distance,
            }
            for row in rows
        ]
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def decode(self, payload: bytes) -> tuple[SetUtilityDistanceInputRow, ...]:
        values = json.loads(payload)
        rows = tuple(
            SetUtilityDistanceInputRow(
                split=value["split"],
                state_id=value["state_id"],
                candidate_event_step_ids=tuple(value["candidate_event_step_ids"]),
                maximum_labeled_cardinality=value[
                    "maximum_labeled_cardinality"
                ],
                coalition_event_step_ids=tuple(value["coalition_event_step_ids"]),
                distance=value["distance"],
            )
            for value in values
        )
        self.decoded_roles.extend(sorted({row.split for row in rows}))
        return rows


def _state_rows(
    *,
    role: str,
    state_id: str,
    worker_index: int,
    event_ids: tuple[int, ...] = (2, 5),
    cap: int = 2,
) -> list[WorkerLabelRow]:
    return [
        WorkerLabelRow(
            execution_worker_index=worker_index,
            label_row=SetUtilityDistanceInputRow(
                split=role,
                state_id=state_id,
                candidate_event_step_ids=event_ids,
                maximum_labeled_cardinality=cap,
                coalition_event_step_ids=coalition,
                distance=float(10 - len(coalition)),
            ),
        )
        for cardinality in range(min(len(event_ids), cap) + 1)
        for coalition in itertools.combinations(event_ids, cardinality)
    ]


def _mixed_execution_rows() -> list[WorkerLabelRow]:
    rows: list[WorkerLabelRow] = []
    rows.extend(_state_rows(role="train", state_id="train-0", worker_index=0))
    rows.extend(_state_rows(role="tune", state_id="tune-0", worker_index=0))
    rows.extend(
        _state_rows(role="evaluation", state_id="evaluation-0", worker_index=0)
    )
    rows.extend(_state_rows(role="train", state_id="train-1", worker_index=1))
    return rows


def test_mixed_execution_workers_publish_only_role_isolated_shards(
    tmp_path: Path,
) -> None:
    codec = JsonFixtureCodec()
    layout = LabelPartitionLayout(worker_count=2)
    root = tmp_path / "formal-labels"
    result = write_role_partitioned_label_shards(
        root,
        _mixed_execution_rows(),
        layout=layout,
        codec=codec,
    )

    assert tuple(item.relative_path for item in result.shards) == (
        "labels/train/part-worker-00.parquet",
        "labels/train/part-worker-01.parquet",
        "labels/tune/part-worker-00.parquet",
        "labels/tune/part-worker-01.parquet",
        "labels/evaluation/part-worker-00.parquet",
        "labels/evaluation/part-worker-01.parquet",
    )
    for descriptor in result.shards:
        rows = codec.decode((root / descriptor.relative_path).read_bytes())
        assert {row.split for row in rows} <= {descriptor.role}
    assert {table.split for table in result.dataset.tables} == set(LABEL_ROLES)
    assert len(result.dataset.tables) == 4


def test_postflight_rejects_cross_role_row_even_when_worker_is_shared(
    tmp_path: Path,
) -> None:
    codec = JsonFixtureCodec()
    layout = LabelPartitionLayout(worker_count=2)
    root = tmp_path / "formal-labels"
    write_role_partitioned_label_shards(
        root,
        _mixed_execution_rows(),
        layout=layout,
        codec=codec,
    )
    path = root / "labels/train/part-worker-00.parquet"
    rows = list(codec.decode(path.read_bytes()))
    leaked = rows[0]
    rows[0] = SetUtilityDistanceInputRow(
        split="evaluation",
        state_id=leaked.state_id,
        candidate_event_step_ids=leaked.candidate_event_step_ids,
        maximum_labeled_cardinality=leaked.maximum_labeled_cardinality,
        coalition_event_step_ids=leaked.coalition_event_step_ids,
        distance=leaked.distance,
    )
    path.write_bytes(codec.encode(rows))

    with pytest.raises(ValueError, match="another physical role"):
        validate_role_partitioned_label_shards(
            root,
            layout=layout,
            codec=codec,
        )


def test_trainer_inventory_has_no_evaluation_identity_and_never_reads_it(
    tmp_path: Path,
) -> None:
    publication_codec = JsonFixtureCodec()
    root = tmp_path / "formal-labels"
    result = write_role_partitioned_label_shards(
        root,
        _mixed_execution_rows(),
        layout=LabelPartitionLayout(worker_count=2),
        codec=publication_codec,
    )
    inventory_payload = json.dumps(result.trainer_inventory.to_payload())
    assert "evaluation" not in inventory_payload
    assert tuple(item.role for item in result.trainer_inventory.shards) == (
        "train",
        "train",
        "tune",
        "tune",
    )

    for path in (root / "labels/evaluation").iterdir():
        path.write_bytes(b"this is deliberately not a readable shard")
    trainer_codec = JsonFixtureCodec()
    dataset = read_trainer_label_dataset(
        root,
        result.trainer_inventory,
        codec=trainer_codec,
    )
    assert {table.split for table in dataset.tables} == {"train", "tune"}
    assert "evaluation" not in trainer_codec.decoded_roles


def test_evaluation_inventory_requires_model_seal_before_identity_release(
    tmp_path: Path,
) -> None:
    codec = JsonFixtureCodec()
    root = tmp_path / "formal-labels"
    result = write_role_partitioned_label_shards(
        root,
        _mixed_execution_rows(),
        layout=LabelPartitionLayout(worker_count=2),
        codec=codec,
    )

    with pytest.raises(ValueError, match="model seal"):
        release_evaluation_label_inventory(result, model_seal_sha256="")
    sealed = release_evaluation_label_inventory(
        result,
        model_seal_sha256="a" * 64,
    )
    assert sealed.model_seal_sha256 == "a" * 64
    assert tuple(item.role for item in sealed.shards) == (
        "evaluation",
        "evaluation",
    )
    dataset = read_evaluation_label_dataset(root, sealed, codec=codec)
    assert {table.split for table in dataset.tables} == {"evaluation"}


def test_state_cannot_be_split_across_execution_workers(tmp_path: Path) -> None:
    rows = _mixed_execution_rows()
    first = rows[0]
    rows[0] = WorkerLabelRow(
        execution_worker_index=1,
        label_row=first.label_row,
    )
    with pytest.raises(ValueError, match="split across execution workers"):
        write_role_partitioned_label_shards(
            tmp_path / "formal-labels",
            rows,
            layout=LabelPartitionLayout(worker_count=2),
            codec=JsonFixtureCodec(),
        )


def test_trainer_inventory_constructor_rejects_evaluation_descriptor() -> None:
    descriptor = LabelShardDescriptor(
        role="evaluation",
        execution_worker_index=0,
        relative_path="labels/evaluation/part-worker-00.parquet",
        row_count=1,
        state_ids=("evaluation-0",),
        byte_count=1,
        sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="train/tune"):
        TrainerLabelInventory(worker_count=1, shards=(descriptor,))


def test_postflight_rejects_symlinked_role_shard(tmp_path: Path) -> None:
    codec = JsonFixtureCodec()
    root = tmp_path / "formal-labels"
    write_role_partitioned_label_shards(
        root,
        _mixed_execution_rows(),
        layout=LabelPartitionLayout(worker_count=2),
        codec=codec,
    )
    path = root / "labels/tune/part-worker-01.parquet"
    target = tmp_path / "replacement.parquet"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="non-regular"):
        validate_role_partitioned_label_shards(
            root,
            layout=LabelPartitionLayout(worker_count=2),
            codec=codec,
        )


def test_writer_rejects_relative_output_root() -> None:
    with pytest.raises(ValueError, match="absolute canonical path"):
        write_role_partitioned_label_shards(
            Path("relative-formal-labels"),
            _mixed_execution_rows(),
            layout=LabelPartitionLayout(worker_count=2),
            codec=JsonFixtureCodec(),
        )


def test_writer_rejects_symlink_in_output_root_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    redirected_parent = tmp_path / "redirected-parent"
    redirected_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        write_role_partitioned_label_shards(
            redirected_parent / "formal-labels",
            _mixed_execution_rows(),
            layout=LabelPartitionLayout(worker_count=2),
            codec=JsonFixtureCodec(),
        )
    assert not (real_parent / "formal-labels").exists()


def test_postflight_rejects_symlinked_role_directory(tmp_path: Path) -> None:
    codec = JsonFixtureCodec()
    root = tmp_path / "formal-labels"
    write_role_partitioned_label_shards(
        root,
        _mixed_execution_rows(),
        layout=LabelPartitionLayout(worker_count=2),
        codec=codec,
    )
    role_dir = root / "labels/tune"
    relocated = tmp_path / "relocated-tune"
    role_dir.rename(relocated)
    role_dir.symlink_to(relocated, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_role_partitioned_label_shards(
            root,
            layout=LabelPartitionLayout(worker_count=2),
            codec=codec,
        )


def test_real_parquet_codec_round_trip_when_optional_dependency_exists() -> None:
    pytest.importorskip("pyarrow")
    codec = PyArrowLabelShardCodec()
    source = tuple(item.label_row for item in _state_rows(
        role="train",
        state_id="train-0",
        worker_index=0,
    ))
    assert codec.decode(codec.encode(source)) == source


def test_real_parquet_codec_rejects_metadata_when_dependency_exists() -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    codec = PyArrowLabelShardCodec()
    source = tuple(
        item.label_row
        for item in _state_rows(
            role="train",
            state_id="train-0",
            worker_index=0,
        )
    )
    schema = codec._schema().with_metadata({b"forbidden": b"payload"})
    table = pyarrow.Table.from_pylist(
        [
            {
                "split": row.split,
                "state_id": row.state_id,
                "candidate_event_step_ids": list(row.candidate_event_step_ids),
                "maximum_labeled_cardinality": row.maximum_labeled_cardinality,
                "coalition_event_step_ids": list(row.coalition_event_step_ids),
                "distance": row.distance,
            }
            for row in source
        ],
        schema=schema,
    )
    sink = pyarrow.BufferOutputStream()
    parquet.write_table(table, sink)
    with pytest.raises(ValueError, match="metadata must be empty"):
        codec.decode(sink.getvalue().to_pybytes())
