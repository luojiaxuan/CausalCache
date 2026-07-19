"""Role-partitioned publication firewall for formal set-utility labels."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from causalcache.set_utility_label_table import (
    SetUtilityDistanceInputRow,
    ValidatedSetUtilityDataset,
    validate_cardinality_capped_label_rows,
)


SCHEMA_VERSION = "1.0.0"
LABEL_ROLES = ("train", "tune", "evaluation")
TRAINER_ROLES = ("train", "tune")
LABEL_ROW_COLUMNS = (
    "split",
    "state_id",
    "candidate_event_step_ids",
    "maximum_labeled_cardinality",
    "coalition_event_step_ids",
    "distance",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class LabelShardCodec(Protocol):
    """Encode and decode one schema-bound aggregate shard."""

    def encode(self, rows: Sequence[SetUtilityDistanceInputRow]) -> bytes: ...

    def decode(self, payload: bytes) -> tuple[SetUtilityDistanceInputRow, ...]: ...


@dataclass(frozen=True)
class LabelPartitionLayout:
    worker_count: int

    def __post_init__(self) -> None:
        if type(self.worker_count) is not int or not 1 <= self.worker_count <= 100:
            raise ValueError("label worker count must be an integer between 1 and 100")

    def relative_path(self, role: str, worker_index: int) -> str:
        _role(role)
        _worker_index(worker_index, worker_count=self.worker_count)
        return f"labels/{role}/part-worker-{worker_index:02d}.parquet"

    @property
    def expected_relative_paths(self) -> tuple[str, ...]:
        return tuple(
            self.relative_path(role, worker_index)
            for role in LABEL_ROLES
            for worker_index in range(self.worker_count)
        )


@dataclass(frozen=True)
class WorkerLabelRow:
    execution_worker_index: int
    label_row: SetUtilityDistanceInputRow

    def __post_init__(self) -> None:
        if type(self.execution_worker_index) is not int:
            raise TypeError("execution worker index must be an integer")
        if not isinstance(self.label_row, SetUtilityDistanceInputRow):
            raise TypeError("worker label row must wrap SetUtilityDistanceInputRow")


@dataclass(frozen=True)
class LabelShardDescriptor:
    role: str
    execution_worker_index: int
    relative_path: str
    row_count: int
    state_ids: tuple[str, ...]
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _role(self.role)
        if (
            type(self.execution_worker_index) is not int
            or self.execution_worker_index < 0
        ):
            raise ValueError("descriptor worker index must be non-negative")
        if type(self.row_count) is not int or self.row_count < 0:
            raise ValueError("descriptor row count must be non-negative")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("descriptor byte count must be positive")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("descriptor SHA256 must be one lowercase digest")
        if (
            any(not isinstance(item, str) or not item for item in self.state_ids)
            or self.state_ids != tuple(sorted(self.state_ids))
            or len(self.state_ids) != len(set(self.state_ids))
        ):
            raise ValueError("descriptor state ids must be sorted unique text")
        expected = (
            f"labels/{self.role}/"
            f"part-worker-{self.execution_worker_index:02d}.parquet"
        )
        if self.relative_path != expected:
            raise ValueError("descriptor path is not the canonical role/worker path")

    def to_payload(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "execution_worker_index": self.execution_worker_index,
            "relative_path": self.relative_path,
            "role": self.role,
            "row_count": self.row_count,
            "sha256": self.sha256,
            "state_ids": list(self.state_ids),
        }


@dataclass(frozen=True)
class TrainerLabelInventory:
    worker_count: int
    shards: tuple[LabelShardDescriptor, ...]

    def __post_init__(self) -> None:
        layout = LabelPartitionLayout(self.worker_count)
        expected = tuple(
            layout.relative_path(role, worker_index)
            for role in TRAINER_ROLES
            for worker_index in range(layout.worker_count)
        )
        observed = tuple(item.relative_path for item in self.shards)
        if observed != expected:
            raise ValueError("trainer inventory must contain exactly train/tune shards")
        if any(item.role not in TRAINER_ROLES for item in self.shards):
            raise ValueError("trainer inventory cannot expose evaluation shards")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "roles": list(TRAINER_ROLES),
            "worker_count": self.worker_count,
            "shards": [item.to_payload() for item in self.shards],
        }


@dataclass(frozen=True)
class SealedEvaluationLabelInventory:
    model_seal_sha256: str
    worker_count: int
    shards: tuple[LabelShardDescriptor, ...]

    def __post_init__(self) -> None:
        _sha256(self.model_seal_sha256, label="model seal SHA256")
        layout = LabelPartitionLayout(self.worker_count)
        expected = tuple(
            layout.relative_path("evaluation", worker_index)
            for worker_index in range(layout.worker_count)
        )
        if tuple(item.relative_path for item in self.shards) != expected:
            raise ValueError("sealed evaluation inventory is incomplete or reordered")
        if any(item.role != "evaluation" for item in self.shards):
            raise ValueError("sealed evaluation inventory contains a non-evaluation shard")

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_seal_sha256": self.model_seal_sha256,
            "role": "evaluation",
            "schema_version": SCHEMA_VERSION,
            "worker_count": self.worker_count,
            "shards": [item.to_payload() for item in self.shards],
        }


@dataclass(frozen=True)
class LabelPartitionPostflight:
    """Privileged producer-side audit result; do not pass it to a trainer."""

    layout: LabelPartitionLayout
    shards: tuple[LabelShardDescriptor, ...]
    dataset: ValidatedSetUtilityDataset
    trainer_inventory: TrainerLabelInventory

    def __post_init__(self) -> None:
        if not isinstance(self.layout, LabelPartitionLayout):
            raise TypeError("label postflight layout is invalid")
        expected = self.layout.expected_relative_paths
        if tuple(item.relative_path for item in self.shards) != expected:
            raise ValueError("label postflight shard inventory drifted")
        if not isinstance(self.dataset, ValidatedSetUtilityDataset):
            raise TypeError("label postflight dataset is invalid")
        if not isinstance(self.trainer_inventory, TrainerLabelInventory):
            raise TypeError("label postflight trainer inventory is invalid")
        expected_trainer = tuple(
            item for item in self.shards if item.role in TRAINER_ROLES
        )
        if self.trainer_inventory.shards != expected_trainer:
            raise ValueError("label postflight trainer inventory drifted")


class PyArrowLabelShardCodec:
    """Production Parquet codec with an explicit, metadata-free row schema."""

    @staticmethod
    def _modules() -> tuple[Any, Any]:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "formal label Parquet I/O requires the project pyarrow extra"
            ) from error
        return pa, pq

    @classmethod
    def _schema(cls) -> Any:
        pa, _ = cls._modules()
        return pa.schema(
            [
                pa.field("split", pa.string(), nullable=False),
                pa.field("state_id", pa.string(), nullable=False),
                pa.field(
                    "candidate_event_step_ids",
                    pa.list_(pa.field("element", pa.int64(), nullable=True)),
                    nullable=False,
                ),
                pa.field(
                    "maximum_labeled_cardinality",
                    pa.int32(),
                    nullable=False,
                ),
                pa.field(
                    "coalition_event_step_ids",
                    pa.list_(pa.field("element", pa.int64(), nullable=True)),
                    nullable=False,
                ),
                pa.field("distance", pa.float64(), nullable=False),
            ]
        )

    def encode(self, rows: Sequence[SetUtilityDistanceInputRow]) -> bytes:
        pa, pq = self._modules()
        schema = self._schema()
        table = pa.Table.from_pylist(
            [_row_payload(row) for row in rows],
            schema=schema,
        )
        sink = pa.BufferOutputStream()
        pq.write_table(
            table,
            sink,
            compression="zstd",
            data_page_version="2.0",
            use_dictionary=False,
            version="2.6",
            write_statistics=True,
        )
        return sink.getvalue().to_pybytes()

    def decode(self, payload: bytes) -> tuple[SetUtilityDistanceInputRow, ...]:
        pa, pq = self._modules()
        source = pa.BufferReader(payload)
        table = pq.read_table(source)
        if table.schema.metadata is not None:
            raise ValueError("formal label Parquet metadata must be empty")
        if not table.schema.equals(self._schema(), check_metadata=True):
            raise ValueError("formal label Parquet schema drifted")
        return tuple(_row_from_payload(item) for item in table.to_pylist())


def write_role_partitioned_label_shards(
    output_root: str | Path,
    execution_rows: Sequence[WorkerLabelRow],
    *,
    layout: LabelPartitionLayout,
    codec: LabelShardCodec | None = None,
) -> LabelPartitionPostflight:
    """Publish mixed-worker results into physically role-isolated shards."""
    root = Path(output_root)
    _require_absolute_path(root, label="label output root")
    canonical = _validate_execution_rows(execution_rows, layout=layout)
    grouped: dict[tuple[str, int], list[SetUtilityDistanceInputRow]] = defaultdict(list)
    for item in canonical:
        grouped[(item.label_row.split, item.execution_worker_index)].append(
            item.label_row
        )
    selected_codec = codec or PyArrowLabelShardCodec()
    _create_label_directories(root)
    for role in LABEL_ROLES:
        for worker_index in range(layout.worker_count):
            rows = _canonical_rows(grouped[(role, worker_index)])
            payload = selected_codec.encode(rows)
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("label shard codec must emit non-empty bytes")
            _write_no_clobber(root / layout.relative_path(role, worker_index), payload)
    return validate_role_partitioned_label_shards(
        root,
        layout=layout,
        codec=selected_codec,
    )


def validate_role_partitioned_label_shards(
    output_root: str | Path,
    *,
    layout: LabelPartitionLayout,
    codec: LabelShardCodec | None = None,
    expected_shards: Sequence[LabelShardDescriptor] | None = None,
) -> LabelPartitionPostflight:
    """Read every producer shard and fail closed on role or state leakage."""
    root = Path(output_root)
    _validate_exact_tree(root, layout=layout)
    selected_codec = codec or PyArrowLabelShardCodec()
    descriptors: list[LabelShardDescriptor] = []
    all_rows: list[SetUtilityDistanceInputRow] = []
    state_worker: dict[str, int] = {}
    role_state_counts = {role: 0 for role in LABEL_ROLES}
    for role in LABEL_ROLES:
        for worker_index in range(layout.worker_count):
            relative_path = layout.relative_path(role, worker_index)
            payload = _read_regular_file(root / relative_path)
            rows = selected_codec.decode(payload)
            if any(row.split != role for row in rows):
                raise ValueError("label shard contains a row from another physical role")
            state_ids = tuple(sorted({row.state_id for row in rows}))
            for state_id in state_ids:
                previous = state_worker.setdefault(state_id, worker_index)
                if previous != worker_index:
                    raise ValueError("one state is split across execution workers")
            role_state_counts[role] += len(state_ids)
            all_rows.extend(rows)
            descriptors.append(
                LabelShardDescriptor(
                    role=role,
                    execution_worker_index=worker_index,
                    relative_path=relative_path,
                    row_count=len(rows),
                    state_ids=state_ids,
                    byte_count=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
    if any(count == 0 for count in role_state_counts.values()):
        raise ValueError("every formal label role must contain at least one state")
    dataset = validate_cardinality_capped_label_rows(all_rows)
    canonical_descriptors = tuple(descriptors)
    if expected_shards is not None and tuple(expected_shards) != canonical_descriptors:
        raise ValueError("label shard bytes or inventory differ from the frozen descriptors")
    trainer = TrainerLabelInventory(
        worker_count=layout.worker_count,
        shards=tuple(item for item in canonical_descriptors if item.role in TRAINER_ROLES),
    )
    return LabelPartitionPostflight(
        layout=layout,
        shards=canonical_descriptors,
        dataset=dataset,
        trainer_inventory=trainer,
    )


def read_trainer_label_dataset(
    output_root: str | Path,
    inventory: TrainerLabelInventory,
    *,
    codec: LabelShardCodec | None = None,
) -> ValidatedSetUtilityDataset:
    """Read only explicit train/tune descriptors without scanning the label root."""
    if not isinstance(inventory, TrainerLabelInventory):
        raise TypeError("trainer label reader requires TrainerLabelInventory")
    rows = _read_inventory_rows(
        Path(output_root),
        inventory.shards,
        codec=codec or PyArrowLabelShardCodec(),
    )
    dataset = validate_cardinality_capped_label_rows(rows)
    if {table.split for table in dataset.tables} != set(TRAINER_ROLES):
        raise ValueError("trainer dataset must contain train and tune roles only")
    return dataset


def release_evaluation_label_inventory(
    postflight: LabelPartitionPostflight,
    *,
    model_seal_sha256: str,
) -> SealedEvaluationLabelInventory:
    """Expose evaluation shard names only after an explicit model seal exists."""
    seal = _sha256(model_seal_sha256, label="model seal SHA256")
    if not isinstance(postflight, LabelPartitionPostflight):
        raise TypeError("evaluation release requires producer postflight")
    return SealedEvaluationLabelInventory(
        model_seal_sha256=seal,
        worker_count=postflight.layout.worker_count,
        shards=tuple(item for item in postflight.shards if item.role == "evaluation"),
    )


def read_evaluation_label_dataset(
    output_root: str | Path,
    inventory: SealedEvaluationLabelInventory,
    *,
    codec: LabelShardCodec | None = None,
) -> ValidatedSetUtilityDataset:
    if not isinstance(inventory, SealedEvaluationLabelInventory):
        raise TypeError("evaluation reader requires a sealed evaluation inventory")
    rows = _read_inventory_rows(
        Path(output_root),
        inventory.shards,
        codec=codec or PyArrowLabelShardCodec(),
    )
    dataset = validate_cardinality_capped_label_rows(rows)
    if {table.split for table in dataset.tables} != {"evaluation"}:
        raise ValueError("sealed evaluation inventory contains another role")
    return dataset


def _validate_execution_rows(
    value: Sequence[WorkerLabelRow],
    *,
    layout: LabelPartitionLayout,
) -> tuple[WorkerLabelRow, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise TypeError("execution rows must be a non-empty sequence")
    rows = tuple(value)
    if not rows or any(not isinstance(item, WorkerLabelRow) for item in rows):
        raise ValueError("execution rows must contain WorkerLabelRow values")
    state_worker: dict[str, int] = {}
    for item in rows:
        _worker_index(item.execution_worker_index, worker_count=layout.worker_count)
        _role(item.label_row.split)
        previous = state_worker.setdefault(
            item.label_row.state_id,
            item.execution_worker_index,
        )
        if previous != item.execution_worker_index:
            raise ValueError("one state cannot be split across execution workers")
    validate_cardinality_capped_label_rows([item.label_row for item in rows])
    if {item.label_row.split for item in rows} != set(LABEL_ROLES):
        raise ValueError("formal execution rows must cover train, tune, and evaluation")
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                item.label_row.split,
                item.execution_worker_index,
                item.label_row.state_id,
                len(item.label_row.coalition_event_step_ids),
                item.label_row.coalition_event_step_ids,
            ),
        )
    )


def _read_inventory_rows(
    root: Path,
    descriptors: Sequence[LabelShardDescriptor],
    *,
    codec: LabelShardCodec,
) -> tuple[SetUtilityDistanceInputRow, ...]:
    rows: list[SetUtilityDistanceInputRow] = []
    for descriptor in descriptors:
        payload = _read_regular_file(root / descriptor.relative_path)
        if (
            len(payload) != descriptor.byte_count
            or hashlib.sha256(payload).hexdigest() != descriptor.sha256
        ):
            raise ValueError("label shard bytes differ from trainer-facing inventory")
        shard_rows = codec.decode(payload)
        if (
            len(shard_rows) != descriptor.row_count
            or tuple(sorted({row.state_id for row in shard_rows}))
            != descriptor.state_ids
            or any(row.split != descriptor.role for row in shard_rows)
        ):
            raise ValueError("label shard contents differ from trainer-facing inventory")
        rows.extend(shard_rows)
    return tuple(rows)


def _canonical_rows(
    rows: Sequence[SetUtilityDistanceInputRow],
) -> tuple[SetUtilityDistanceInputRow, ...]:
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.state_id,
                len(row.coalition_event_step_ids),
                row.coalition_event_step_ids,
            ),
        )
    )


def _row_payload(row: SetUtilityDistanceInputRow) -> dict[str, Any]:
    if not isinstance(row, SetUtilityDistanceInputRow):
        raise TypeError("label shard row must be SetUtilityDistanceInputRow")
    return {
        "split": row.split,
        "state_id": row.state_id,
        "candidate_event_step_ids": list(row.candidate_event_step_ids),
        "maximum_labeled_cardinality": row.maximum_labeled_cardinality,
        "coalition_event_step_ids": list(row.coalition_event_step_ids),
        "distance": row.distance,
    }


def _row_from_payload(value: Mapping[str, Any]) -> SetUtilityDistanceInputRow:
    if not isinstance(value, Mapping) or tuple(value) != LABEL_ROW_COLUMNS:
        raise ValueError("label shard row fields or ordering drifted")
    return SetUtilityDistanceInputRow(
        split=value["split"],
        state_id=value["state_id"],
        candidate_event_step_ids=tuple(value["candidate_event_step_ids"]),
        maximum_labeled_cardinality=value["maximum_labeled_cardinality"],
        coalition_event_step_ids=tuple(value["coalition_event_step_ids"]),
        distance=value["distance"],
    )


def _write_no_clobber(path: Path, payload: bytes) -> None:
    _require_absolute_path(path, label="label shard path")
    parent_descriptor = _open_absolute_directory(path.parent)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=parent_descriptor,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing label shard")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _read_regular_file(path: Path) -> bytes:
    _require_absolute_path(path, label="label shard path")
    parent_descriptor = _open_absolute_directory(path.parent)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("label shard must be a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 8 * 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    finally:
        os.close(parent_descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size)
    if identity != (after.st_dev, after.st_ino, after.st_size) or identity != (
        current.st_dev,
        current.st_ino,
        current.st_size,
    ):
        raise ValueError("label shard changed while it was being read")
    return b"".join(chunks)


def _validate_exact_tree(root: Path, *, layout: LabelPartitionLayout) -> None:
    _require_absolute_path(root, label="label output root")
    _require_plain_directory_chain(root, label="label output root")
    labels = root / "labels"
    _require_plain_directory_chain(labels, label="labels directory")
    if {item.name for item in labels.iterdir()} != set(LABEL_ROLES):
        raise ValueError("labels directory must contain exactly the three roles")
    for role in LABEL_ROLES:
        role_dir = labels / role
        _require_plain_directory_chain(
            role_dir,
            label=f"{role} label directory",
        )
        expected = {
            PurePosixPath(layout.relative_path(role, worker_index)).name
            for worker_index in range(layout.worker_count)
        }
        if {item.name for item in role_dir.iterdir()} != expected:
            raise ValueError(f"{role} label shard inventory drifted")
        for item in role_dir.iterdir():
            mode = item.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise ValueError("label shard inventory contains a non-regular file")


def _create_label_directories(root: Path) -> None:
    _require_plain_directory_chain(root.parent, label="label output parent")
    parent_descriptor = _open_absolute_directory(root.parent)
    root_descriptor: int | None = None
    labels_descriptor: int | None = None
    try:
        try:
            root_descriptor = os.open(
                root.name,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            os.mkdir(root.name, mode=0o755, dir_fd=parent_descriptor)
            root_descriptor = os.open(
                root.name,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
        try:
            os.stat("labels", dir_fd=root_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"label output subtree already exists: {root / 'labels'}"
            )
        os.mkdir("labels", mode=0o755, dir_fd=root_descriptor)
        labels_descriptor = os.open(
            "labels",
            _directory_open_flags(),
            dir_fd=root_descriptor,
        )
        for role in LABEL_ROLES:
            os.mkdir(role, mode=0o755, dir_fd=labels_descriptor)
            role_descriptor = os.open(
                role,
                _directory_open_flags(),
                dir_fd=labels_descriptor,
            )
            os.close(role_descriptor)
    finally:
        if labels_descriptor is not None:
            os.close(labels_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)
        os.close(parent_descriptor)
    _require_plain_directory_chain(root, label="label output root")
    _require_plain_directory_chain(root / "labels", label="labels directory")
    for role in LABEL_ROLES:
        _require_plain_directory_chain(
            root / "labels" / role,
            label=f"{role} label directory",
        )


def _require_plain_directory_chain(path: Path, *, label: str) -> None:
    _require_absolute_path(path, label=label)
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{label} is missing: {current}") from error
        if not stat.S_ISDIR(mode):
            raise ValueError(f"{label} contains a symlink or non-directory component")


def _open_absolute_directory(path: Path) -> int:
    _require_absolute_path(path, label="directory path")
    descriptor = os.open(path.anchor, _directory_open_flags())
    try:
        for component in path.parts[1:]:
            child = os.open(
                component,
                _directory_open_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _require_absolute_path(path: Path, *, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute canonical path")


def _role(value: Any) -> str:
    if not isinstance(value, str) or value not in LABEL_ROLES:
        raise ValueError(f"label role must be one of {LABEL_ROLES!r}")
    return value


def _worker_index(value: Any, *, worker_count: int) -> int:
    if type(value) is not int or not 0 <= value < worker_count:
        raise ValueError("execution worker index is outside the frozen layout")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


__all__ = [
    "LABEL_ROLES",
    "LABEL_ROW_COLUMNS",
    "SCHEMA_VERSION",
    "TRAINER_ROLES",
    "LabelPartitionLayout",
    "LabelPartitionPostflight",
    "LabelShardCodec",
    "LabelShardDescriptor",
    "PyArrowLabelShardCodec",
    "SealedEvaluationLabelInventory",
    "TrainerLabelInventory",
    "WorkerLabelRow",
    "read_evaluation_label_dataset",
    "read_trainer_label_dataset",
    "release_evaluation_label_inventory",
    "validate_role_partitioned_label_shards",
    "write_role_partitioned_label_shards",
]
