"""Pseudonymous CPU-only image-format census for the frozen Freeze-B v2 roster."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.restoration_v2_text_backend import prepare_image_bytes
from causalcache.set_utility_processor_substrate import SelectedPilot


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_selected_image_format_census_v1"
WORKER_STATUS = "SELECTED_IMAGE_FORMAT_CENSUS_WORKER_COMPLETED"
FINAL_STATUS = "SELECTED_IMAGE_FORMAT_CENSUS_COMPLETED"
WORKER_COUNT = 4
EXPECTED_TRAJECTORY_COUNT = 1_200
EXPECTED_OBSERVATION_COUNT = 18_792
RECORD_KEYS = {
    "alpha_extrema",
    "decoded_rgb_sha256",
    "exif_present",
    "format",
    "height",
    "image_sha256",
    "mode",
    "selector_identity_sha256",
    "width",
}

_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
        return (json.dumps(value, **options) + "\n").encode("utf-8")
    options["separators"] = (",", ":")
    return json.dumps(value, **options).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class SelectedImageCensusRecord:
    selector_identity_sha256: str
    image_sha256: str
    decoded_rgb_sha256: str
    format: str
    mode: str
    alpha_extrema: tuple[int, int] | None
    exif_present: bool
    width: int
    height: int

    def __post_init__(self) -> None:
        _sha256(self.selector_identity_sha256, label="selector identity SHA256")
        _sha256(self.image_sha256, label="image SHA256")
        _sha256(self.decoded_rgb_sha256, label="decoded RGB SHA256")
        if not isinstance(self.format, str) or not self.format:
            raise ValueError("image format must be non-empty text")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("image mode must be non-empty text")
        if self.alpha_extrema is not None and (
            len(self.alpha_extrema) != 2
            or any(type(value) is not int or not 0 <= value <= 255 for value in self.alpha_extrema)
            or self.alpha_extrema[0] > self.alpha_extrema[1]
        ):
            raise ValueError("alpha extrema must be null or one ordered uint8 pair")
        if type(self.exif_present) is not bool:
            raise TypeError("EXIF presence must be boolean")
        _positive_int(self.width, label="image width")
        _positive_int(self.height, label="image height")

    def to_payload(self) -> dict[str, Any]:
        return {
            "alpha_extrema": (
                list(self.alpha_extrema) if self.alpha_extrema is not None else None
            ),
            "decoded_rgb_sha256": self.decoded_rgb_sha256,
            "exif_present": self.exif_present,
            "format": self.format,
            "height": self.height,
            "image_sha256": self.image_sha256,
            "mode": self.mode,
            "selector_identity_sha256": self.selector_identity_sha256,
            "width": self.width,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> SelectedImageCensusRecord:
        if not isinstance(value, Mapping) or set(value) != RECORD_KEYS:
            raise ValueError("selected-image census record fields drifted")
        raw_alpha = value["alpha_extrema"]
        if raw_alpha is None:
            alpha = None
        elif (
            isinstance(raw_alpha, Sequence)
            and not isinstance(raw_alpha, (str, bytes, bytearray, Mapping))
            and len(raw_alpha) == 2
        ):
            alpha = (raw_alpha[0], raw_alpha[1])
        else:
            raise ValueError("selected-image census alpha extrema drifted")
        return cls(
            selector_identity_sha256=value["selector_identity_sha256"],
            image_sha256=value["image_sha256"],
            decoded_rgb_sha256=value["decoded_rgb_sha256"],
            format=value["format"],
            mode=value["mode"],
            alpha_extrema=alpha,
            exif_present=value["exif_present"],
            width=value["width"],
            height=value["height"],
        )


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


def build_selected_image_census_records(
    pilot: SelectedPilot,
) -> tuple[SelectedImageCensusRecord, ...]:
    """Decode each selected trajectory observation exactly once in step order."""
    if not isinstance(pilot, SelectedPilot):
        raise TypeError("pilot must be one SelectedPilot")
    trajectory = pilot.manifest.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise ValueError("selected pilot trajectory is missing")
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError("selected pilot steps must be one list")
    expected_count = pilot.assignment.decision_count + 1
    if len(steps) != expected_count:
        raise ValueError("selected pilot observation denominator drifted")
    paths: list[str] = []
    for ordinal, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("observation_index") != ordinal:
            raise ValueError("selected pilot observation order drifted")
        path = step.get("observation_path")
        if not isinstance(path, str) or not path:
            raise ValueError("selected pilot observation path is invalid")
        paths.append(path)
    if len(paths) != len(set(paths)) or set(paths) != set(pilot.image_payloads):
        raise ValueError("selected pilot image inventory drifted")

    return build_selected_image_census_records_from_payloads(
        p0_selection_sha256=pilot.assignment.p0_selection_sha256,
        image_payloads=tuple(pilot.image_payloads[path] for path in paths),
    )


def build_selected_image_census_records_from_payloads(
    *,
    p0_selection_sha256: str,
    image_payloads: Sequence[bytes],
) -> tuple[SelectedImageCensusRecord, ...]:
    """Decode one pinned selected row's ordered image column without semantics."""
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


def _histogram_key(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, tuple):
        return f"{value[0]}:{value[1]}"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def record_histograms(
    records: Iterable[SelectedImageCensusRecord],
) -> dict[str, dict[str, int]]:
    counters = {
        "alpha_extrema": Counter(),
        "dimensions": Counter(),
        "exif_present": Counter(),
        "format": Counter(),
        "format_mode": Counter(),
        "mode": Counter(),
    }
    for record in records:
        if not isinstance(record, SelectedImageCensusRecord):
            raise TypeError("histogram input contains a non-census record")
        counters["alpha_extrema"][_histogram_key(record.alpha_extrema)] += 1
        counters["dimensions"][f"{record.width}x{record.height}"] += 1
        counters["exif_present"][_histogram_key(record.exif_present)] += 1
        counters["format"][record.format] += 1
        counters["format_mode"][f"{record.format}:{record.mode}"] += 1
        counters["mode"][record.mode] += 1
    return {
        name: dict(sorted(counter.items()))
        for name, counter in sorted(counters.items())
    }


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
        "selector_inventory_sha256": sha256_bytes(
            canonical_json_bytes(selectors)
        ),
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
        receipt = _parse_json_line(receipt_path.read_bytes(), label="worker receipt")
        if canonical_json_bytes(receipt, pretty=True) != receipt_path.read_bytes():
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


def _receipt_payload(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact": {
            "byte_count": descriptor["byte_count"],
            "filename": f"records/worker-{descriptor['worker_index']:02d}.jsonl",
            "record_sha256": descriptor["record_sha256"],
        },
        "histograms": descriptor["histograms"],
        "observation_count": descriptor["observation_count"],
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "selector_inventory_sha256": descriptor["selector_inventory_sha256"],
        "status": WORKER_STATUS,
        "unique_image_sha256_count": descriptor["unique_image_sha256_count"],
        "worker_index": descriptor["worker_index"],
    }


def aggregate_worker_records(
    staging_root: str | Path,
    *,
    expected_worker_observation_counts: Sequence[int],
) -> dict[str, Any]:
    if tuple(type(value) for value in expected_worker_observation_counts) != (int,) * WORKER_COUNT:
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
            all_records.append(SelectedImageCensusRecord.from_payload(_parse_json_line(line, label="worker record")))
        workers.append(
            {
                "observation_count": descriptor["observation_count"],
                "receipt_sha256": sha256_bytes(receipt_payload),
                "record_byte_count": descriptor["byte_count"],
                "record_sha256": descriptor["record_sha256"],
                "selector_inventory_sha256": descriptor["selector_inventory_sha256"],
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
        "unique_image_sha256_count": len({record.image_sha256 for record in all_records}),
        "workers": workers,
    }


__all__ = [
    "EXPECTED_OBSERVATION_COUNT",
    "EXPECTED_TRAJECTORY_COUNT",
    "FINAL_STATUS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SelectedImageCensusRecord",
    "aggregate_worker_records",
    "build_selected_image_census_records",
    "build_selected_image_census_records_from_payloads",
    "canonical_json_bytes",
    "inspect_worker_records",
    "materialize_worker_records",
    "record_histograms",
    "selector_identity_sha256",
    "sha256_bytes",
    "sha256_file",
]
