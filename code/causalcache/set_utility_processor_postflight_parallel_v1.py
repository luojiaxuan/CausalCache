"""Versioned four-worker semantic postflight for processor-freeze v2 roots."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.set_utility_processor_artifacts import (
    ProcessorWorkerShard,
    iter_processor_query_artifact_records,
)
from causalcache.set_utility_processor_freeze import WORKER_COUNT
from causalcache.set_utility_processor_image_contract_v2 import (
    PROCESSOR_IMAGE_CONTRACT_V2_ID,
    validate_processor_ocr_record_v2,
)
import causalcache.set_utility_processor_postflight_v2 as _historical
from causalcache.set_utility_processor_postflight_v2 import (
    OUTPUT_BASENAME_PREFIX,
    ProcessorFreezePostflightContextV2,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = (
    "causalcache_set_utility_processor_postflight_v2_parallel_v1"
)
VALIDATION_STATUS = (
    "VALID_COMPLETED_SET_UTILITY_PROCESSOR_FREEZE_V2_"
    "PARALLEL_POSTFLIGHT_V1"
)
HISTORICAL_POSTFLIGHT_SHA256 = (
    "de6eb1a18ea896890efc8361e287733300c1e704887582c5f82342e69382f7cc"
)


@dataclass(frozen=True)
class V2ParallelWorkerSemanticAudit:
    worker_index: int
    terminal_tally: tuple[tuple[str, int], ...]
    fully_validated_paths: tuple[tuple[str, str], ...]
    terminal_paths: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.worker_index) is not int or not 0 <= self.worker_index < WORKER_COUNT:
            raise ValueError("parallel semantic worker index is invalid")
        if (
            not isinstance(self.terminal_tally, tuple)
            or self.terminal_tally != tuple(sorted(self.terminal_tally))
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or type(item[1]) is not int
                or item[1] < 0
                for item in self.terminal_tally
            )
            or len({item[0] for item in self.terminal_tally})
            != len(self.terminal_tally)
        ):
            raise ValueError("parallel semantic worker tally is not canonical")
        for paths, label in (
            (self.fully_validated_paths, "fully validated"),
            (self.terminal_paths, "terminal"),
        ):
            if (
                not isinstance(paths, tuple)
                or paths != tuple(sorted(paths))
                or len(paths) != len(set(paths))
                or any(
                    not isinstance(identity, tuple)
                    or len(identity) != 2
                    or any(
                        not isinstance(value, str) or not value
                        for value in identity
                    )
                    for identity in paths
                )
            ):
                raise ValueError(
                    f"parallel semantic {label} paths are not canonical"
                )


def validate_v2_parallel_worker_artifact_semantics(
    root: Path,
    context: ProcessorFreezePostflightContextV2,
    worker: ProcessorWorkerShard,
) -> V2ParallelWorkerSemanticAudit:
    """Run the historical v2 semantic checks for exactly one bound worker tar."""
    fully_validated_paths: set[tuple[str, str]] = set()
    terminal_paths: set[tuple[str, str]] = set()
    worker_tally: Counter[str] = Counter()
    shard = root / "substrate" / worker.filename
    for query in iter_processor_query_artifact_records(
        shard,
        expected_worker=worker,
    ):
        for path, payload in query.image_payloads.items():
            record = query.ocr_records_by_path.get(path)
            if record is None:
                raise ValueError("stored image is missing its v2 OCR record")
            identity = (query.trajectory_id, path)
            if identity not in fully_validated_paths:
                validate_processor_ocr_record_v2(
                    record,
                    image_bytes=payload,
                    backend_config=context.backend_config,
                    backend_config_sha256=context.backend_config_sha256,
                )
                fully_validated_paths.add(identity)
        if query.query_kind != "terminal":
            continue
        for path, record in query.ocr_records_by_path.items():
            terminal_paths.add((query.trajectory_id, path))
            source_format, source_mode = _historical._validate_record_without_image(
                record,
                expected_path=path,
                backend_config_sha256=context.backend_config_sha256,
            )
            worker_tally[f"{source_format}:{source_mode}"] += 1
    return V2ParallelWorkerSemanticAudit(
        worker_index=worker.worker_index,
        terminal_tally=tuple(sorted(worker_tally.items())),
        fully_validated_paths=tuple(sorted(fully_validated_paths)),
        terminal_paths=tuple(sorted(terminal_paths)),
    )


def validate_v2_artifact_semantics_parallel(
    root: Path,
    context: ProcessorFreezePostflightContextV2,
) -> tuple[dict[str, int], int, int]:
    """Validate four independent tars concurrently and aggregate in worker order."""
    workers = context.structural_context.worker_schedule.workers

    def inspect(worker: ProcessorWorkerShard) -> V2ParallelWorkerSemanticAudit:
        return validate_v2_parallel_worker_artifact_semantics(
            root,
            context,
            worker,
        )

    with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
        worker_audits = tuple(executor.map(inspect, workers))
    expected_indices = tuple(range(WORKER_COUNT))
    if (
        tuple(worker.worker_index for worker in workers) != expected_indices
        or tuple(audit.worker_index for audit in worker_audits) != expected_indices
    ):
        raise ValueError("parallel semantic worker result order drifted")

    terminal_tallies: list[Mapping[str, int]] = []
    fully_validated_paths: set[tuple[str, str]] = set()
    terminal_paths: set[tuple[str, str]] = set()
    for audit in worker_audits:
        worker_fully_validated = set(audit.fully_validated_paths)
        worker_terminal = set(audit.terminal_paths)
        if not worker_fully_validated.issubset(worker_terminal):
            raise ValueError("stored image inventory escaped terminal OCR coverage")
        if fully_validated_paths.intersection(worker_fully_validated):
            raise ValueError("fully validated OCR paths overlap across workers")
        if terminal_paths.intersection(worker_terminal):
            raise ValueError("terminal OCR paths overlap across workers")
        fully_validated_paths.update(worker_fully_validated)
        terminal_paths.update(worker_terminal)
        terminal_tallies.append(dict(audit.terminal_tally))

    metadata_only_paths = terminal_paths - fully_validated_paths
    return (
        _historical._validate_global_format_tally(terminal_tallies),
        len(fully_validated_paths),
        len(metadata_only_paths),
    )


def validate_completed_processor_freeze_root_parallel_v1(
    output_root: str | Path,
    *,
    context: ProcessorFreezePostflightContextV2,
) -> dict[str, Any]:
    """Validate one completed v2 root under a distinct parallel protocol."""
    if not isinstance(context, ProcessorFreezePostflightContextV2):
        raise TypeError("parallel postflight context is invalid")
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("parallel postflight output root must be absolute")
    expected_output_basename = (
        OUTPUT_BASENAME_PREFIX
        + context.structural_context.expected_git_revision[:7]
    )
    if root.name != expected_output_basename:
        raise ValueError("v2 output root differs from its exact Git-bound namespace")

    structural = _historical._validate_completed_processor_freeze_root_v1(
        root,
        context=context.structural_context,
    )
    processor_thread_runtime_evidence = (
        _historical._validate_processor_thread_runtime_logs(root)
    )
    receipt_tally = _historical._validate_global_format_tally(
        _historical._receipt_tallies_and_contract_identity(root, context)
    )
    (
        artifact_tally,
        stored_validation_count,
        metadata_only_validation_count,
    ) = validate_v2_artifact_semantics_parallel(root, context)
    if artifact_tally != receipt_tally:
        raise ValueError("v2 artifact and receipt global format tallies differ")
    return {
        **{key: value for key, value in structural.items() if key != "status"},
        "global_format_mode_tally": artifact_tally,
        "historical_postflight_sha256": HISTORICAL_POSTFLIGHT_SHA256,
        "image_contract_sha256": context.image_contract_sha256,
        "metadata_only_ocr_validation_count": metadata_only_validation_count,
        "parallel_semantic_aggregation_order": list(range(WORKER_COUNT)),
        "parallel_semantic_worker_count": WORKER_COUNT,
        "processor_image_contract_id": PROCESSOR_IMAGE_CONTRACT_V2_ID,
        "processor_thread_runtime_evidence": list(
            processor_thread_runtime_evidence
        ),
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "stored_image_ocr_validation_count": stored_validation_count,
        "structural_validation_status": structural["status"],
        "status": VALIDATION_STATUS,
    }


__all__ = [
    "HISTORICAL_POSTFLIGHT_SHA256",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "VALIDATION_STATUS",
    "V2ParallelWorkerSemanticAudit",
    "validate_completed_processor_freeze_root_parallel_v1",
    "validate_v2_artifact_semantics_parallel",
    "validate_v2_parallel_worker_artifact_semantics",
]
