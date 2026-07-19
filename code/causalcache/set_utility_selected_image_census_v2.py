"""Column-projected v2 selected-image format census primitives."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.restoration_v2_text_backend import prepare_image_bytes
from causalcache.set_utility_selected_image_census import (
    SelectedImageCensusRecord,
    canonical_json_bytes,
    record_histograms,
    sha256_bytes,
    sha256_file,
)


SCHEMA_VERSION = "2.0.0"
PROTOCOL_ID = (
    "causalcache_set_utility_selected_image_format_census_v2_"
    "column_projection_repair"
)
WORKER_STATUS = (
    "SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR_WORKER_COMPLETED"
)
FINAL_STATUS = "SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR_COMPLETED"
WORKER_COUNT = 4
EXPECTED_TRAJECTORY_COUNT = 1_200
EXPECTED_OBSERVATION_COUNT = 18_792
PARQUET_PROJECTION = {
    "batch_size": 8,
    "columns": ["images"],
    "expected_batch_schema_names": ["images"],
    "expected_row_keys": ["images"],
}

_SHA256 = re.compile(r"[0-9a-f]{64}")


def parquet_projection_payload() -> dict[str, Any]:
    """Return an unfrozen JSON copy of the exact v2 projection contract."""
    return {
        "batch_size": PARQUET_PROJECTION["batch_size"],
        "columns": list(PARQUET_PROJECTION["columns"]),
        "expected_batch_schema_names": list(
            PARQUET_PROJECTION["expected_batch_schema_names"]
        ),
        "expected_row_keys": list(PARQUET_PROJECTION["expected_row_keys"]),
    }


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def selector_identity_sha256(*, p0_selection_sha256: str, observation_ordinal: int) -> str:
    _sha256(p0_selection_sha256, label="P0 selection SHA256")
    if type(observation_ordinal) is not int or observation_ordinal < 0:
        raise ValueError("observation ordinal must be a non-negative integer")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "observation_ordinal": observation_ordinal,
                "p0_selection_sha256": p0_selection_sha256,
                "protocol_id": PROTOCOL_ID,
            }
        )
    )


def build_selected_image_census_records_from_payloads(
    *,
    p0_selection_sha256: str,
    image_payloads: Sequence[bytes],
) -> tuple[SelectedImageCensusRecord, ...]:
    """Decode one pinned selected row's projected image column only."""
    _sha256(p0_selection_sha256, label="P0 selection SHA256")
    if (
        isinstance(image_payloads, (str, bytes, bytearray, Mapping))
        or not isinstance(image_payloads, Sequence)
        or not image_payloads
    ):
        raise ValueError("selected image payloads must be one non-empty sequence")
    records: list[SelectedImageCensusRecord] = []
    for ordinal, payload in enumerate(image_payloads):
        if not isinstance(payload, bytes) or not payload:
            raise TypeError("selected image payload must be non-empty bytes")
        prepared = prepare_image_bytes(payload)
        records.append(
            SelectedImageCensusRecord(
                selector_identity_sha256=selector_identity_sha256(
                    p0_selection_sha256=p0_selection_sha256,
                    observation_ordinal=ordinal,
                ),
                image_sha256=sha256_bytes(payload),
                decoded_rgb_sha256=prepared.rgb_bytes_sha256,
                format=prepared.source_format,
                mode=prepared.source_mode,
                alpha_extrema=prepared.alpha_extrema,
                exif_present=prepared.exif_present,
                width=prepared.width,
                height=prepared.height,
            )
        )
    return tuple(records)


def _parse_json_line(line: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            line.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def inspect_worker_records(
    path: str | Path,
    *,
    expected_worker_index: int,
    expected_observation_count: int,
) -> dict[str, Any]:
    record_path = Path(path)
    if not record_path.is_file() or record_path.is_symlink():
        raise ValueError("worker record artifact must be one regular file")
    records: list[SelectedImageCensusRecord] = []
    payload = record_path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("worker record artifact must be non-empty canonical JSONL")
    for index, line in enumerate(payload.splitlines()):
        raw = _parse_json_line(line, label=f"worker record {index}")
        record = SelectedImageCensusRecord.from_payload(raw)
        if canonical_json_bytes(record.to_payload()) != line:
            raise ValueError("worker record line is not canonical JSON")
        records.append(record)
    if len(records) != expected_observation_count:
        raise ValueError("worker observation denominator drifted")
    selectors = [record.selector_identity_sha256 for record in records]
    if len(selectors) != len(set(selectors)):
        raise ValueError("worker selector identity is duplicated")
    return {
        "byte_count": len(payload),
        "histograms": record_histograms(records),
        "observation_count": len(records),
        "record_sha256": sha256_bytes(payload),
        "selector_inventory_sha256": sha256_bytes(canonical_json_bytes(selectors)),
        "unique_image_sha256_count": len({record.image_sha256 for record in records}),
        "worker_index": expected_worker_index,
    }


def _worker_paths(staging_root: Path, worker_index: int) -> tuple[Path, Path]:
    if type(worker_index) is not int or not 0 <= worker_index < WORKER_COUNT:
        raise ValueError("worker index must be in [0, 3]")
    return (
        staging_root / "records" / f"worker-{worker_index:02d}.jsonl",
        staging_root / "receipts" / f"worker-{worker_index:02d}.json",
    )


def _ensure_staging_layout(staging: Path) -> None:
    if not staging.is_dir() or staging.is_symlink():
        raise ValueError("staging root must be one existing regular directory")
    for name in ("records", "receipts"):
        path = staging / name
        try:
            path.mkdir()
        except FileExistsError:
            pass
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"staging {name} must be one regular directory")


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    if not temporary.is_file() or temporary.is_symlink():
        raise ValueError("temporary artifact must be one regular file")
    try:
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic_no_clobber(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary artifact already exists: {temporary.name}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_no_clobber(temporary, path)


def _receipt_payload(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact": {
            "byte_count": descriptor["byte_count"],
            "filename": f"records/worker-{descriptor['worker_index']:02d}.jsonl",
            "record_sha256": descriptor["record_sha256"],
        },
        "histograms": descriptor["histograms"],
        "observation_count": descriptor["observation_count"],
        "parquet_projection": parquet_projection_payload(),
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "selector_inventory_sha256": descriptor["selector_inventory_sha256"],
        "status": WORKER_STATUS,
        "unique_image_sha256_count": descriptor["unique_image_sha256_count"],
        "worker_index": descriptor["worker_index"],
    }


def materialize_worker_records(
    staging_root: str | Path,
    *,
    worker_index: int,
    expected_observation_count: int,
    records: Iterable[SelectedImageCensusRecord],
) -> dict[str, Any]:
    staging = Path(staging_root)
    _ensure_staging_layout(staging)
    record_path, receipt_path = _worker_paths(staging, worker_index)
    if record_path.exists() and receipt_path.exists():
        descriptor = inspect_worker_records(
            record_path,
            expected_worker_index=worker_index,
            expected_observation_count=expected_observation_count,
        )
        receipt_payload = receipt_path.read_bytes()
        receipt = _parse_json_line(receipt_payload, label="worker receipt")
        if canonical_json_bytes(receipt, pretty=True) != receipt_payload:
            raise ValueError("worker receipt is not canonical pretty JSON")
        if dict(receipt) != _receipt_payload(descriptor):
            raise ValueError("resumable worker receipt differs from record artifact")
        return descriptor
    if record_path.exists() or receipt_path.exists():
        raise ValueError("incomplete worker record/receipt pair cannot be resumed")
    payload = bytearray()
    for record in records:
        if not isinstance(record, SelectedImageCensusRecord):
            raise TypeError("worker yielded a non-census record")
        payload.extend(canonical_json_bytes(record.to_payload()) + b"\n")
    _write_atomic_no_clobber(record_path, bytes(payload))
    descriptor = inspect_worker_records(
        record_path,
        expected_worker_index=worker_index,
        expected_observation_count=expected_observation_count,
    )
    _write_atomic_no_clobber(
        receipt_path,
        canonical_json_bytes(_receipt_payload(descriptor), pretty=True),
    )
    return descriptor


def aggregate_worker_records(
    staging_root: str | Path,
    *,
    expected_worker_observation_counts: Sequence[int],
) -> dict[str, Any]:
    if tuple(type(value) for value in expected_worker_observation_counts) != (
        int,
    ) * WORKER_COUNT:
        raise ValueError("exactly four integer worker observation counts are required")
    staging = Path(staging_root)
    _ensure_staging_layout(staging)
    all_records: list[SelectedImageCensusRecord] = []
    workers: list[dict[str, Any]] = []
    for worker_index, expected_count in enumerate(expected_worker_observation_counts):
        record_path, receipt_path = _worker_paths(staging, worker_index)
        descriptor = inspect_worker_records(
            record_path,
            expected_worker_index=worker_index,
            expected_observation_count=expected_count,
        )
        receipt_payload = receipt_path.read_bytes()
        receipt = _parse_json_line(receipt_payload, label="worker receipt")
        if canonical_json_bytes(receipt, pretty=True) != receipt_payload:
            raise ValueError("worker receipt is not canonical pretty JSON")
        if dict(receipt) != _receipt_payload(descriptor):
            raise ValueError("worker receipt differs from record artifact")
        for line in record_path.read_bytes().splitlines():
            all_records.append(
                SelectedImageCensusRecord.from_payload(
                    _parse_json_line(line, label="worker record")
                )
            )
        workers.append(
            {
                "observation_count": descriptor["observation_count"],
                "receipt_sha256": sha256_bytes(receipt_payload),
                "record_byte_count": descriptor["byte_count"],
                "record_sha256": descriptor["record_sha256"],
                "selector_inventory_sha256": descriptor[
                    "selector_inventory_sha256"
                ],
                "worker_index": worker_index,
            }
        )
    selectors = [record.selector_identity_sha256 for record in all_records]
    if len(selectors) != len(set(selectors)):
        raise ValueError("global selector identity is duplicated")
    if len(all_records) != sum(expected_worker_observation_counts):
        raise ValueError("global selected-observation denominator drifted")
    return {
        "histograms": record_histograms(all_records),
        "observation_count": len(all_records),
        "record_inventory_sha256": sha256_bytes(
            canonical_json_bytes([worker["record_sha256"] for worker in workers])
        ),
        "selector_inventory_sha256": sha256_bytes(canonical_json_bytes(selectors)),
        "unique_image_sha256_count": len(
            {record.image_sha256 for record in all_records}
        ),
        "workers": workers,
    }


__all__ = [
    "EXPECTED_OBSERVATION_COUNT",
    "EXPECTED_TRAJECTORY_COUNT",
    "FINAL_STATUS",
    "PARQUET_PROJECTION",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SelectedImageCensusRecord",
    "WORKER_COUNT",
    "WORKER_STATUS",
    "aggregate_worker_records",
    "build_selected_image_census_records_from_payloads",
    "canonical_json_bytes",
    "inspect_worker_records",
    "materialize_worker_records",
    "parquet_projection_payload",
    "record_histograms",
    "selector_identity_sha256",
    "sha256_bytes",
    "sha256_file",
]
