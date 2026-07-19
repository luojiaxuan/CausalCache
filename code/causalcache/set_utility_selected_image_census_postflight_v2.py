"""Independent read-only postflight for the repaired selected-image census."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.set_utility_processor_freeze import (
    ShardWorkerSchedule,
    build_shard_worker_schedule,
)
from causalcache.set_utility_processor_substrate import build_selected_row_read_plan
from causalcache.set_utility_selected_image_census_v2 import (
    EXPECTED_OBSERVATION_COUNT,
    EXPECTED_TRAJECTORY_COUNT,
    FINAL_STATUS,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    WORKER_COUNT,
    WORKER_STATUS,
    SelectedImageCensusRecord,
    canonical_json_bytes,
    parquet_projection_payload,
    record_histograms,
    selector_identity_sha256,
    sha256_bytes,
)
from causalcache.set_utility_selected_image_census_contract_v2 import (
    CANONICAL_CONFIG_PATH,
    HF_REPO,
    HF_TAG,
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_INPUT_PATHS,
    REQUIRED_PATH_ARGUMENTS,
    SelectedImageCensusContractV2,
)


VALIDATION_STATUS = (
    "VALID_COMPLETED_SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR"
)
EXPECTED_TOP_LEVEL_FILES = frozenset({"manifest.json", "run-identity.json"})
EXPECTED_TOP_LEVEL_DIRECTORIES = frozenset({"receipts", "records"})
INVALID_V1_PROTOCOL_ID = "causalcache_set_utility_selected_image_format_census_v1"
INVALID_V1_FINAL_STATUS = "SELECTED_IMAGE_FORMAT_CENSUS_COMPLETED"
OUTPUT_NAMESPACE_PREFIX = (
    "causalcache-selected-image-format-census-v2-column-projection-repair-"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SelectedImageCensusV2PostflightContext:
    repository_root: Path
    execution_config_path: Path
    execution_config_sha256: str
    expected_git_revision: str
    worker_schedule: ShardWorkerSchedule
    source_audit: Mapping[str, str]
    frozen_input_sha256: Mapping[str, str]
    predecessor_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.repository_root.is_absolute():
            raise ValueError("postflight repository root must be absolute")
        if not self.execution_config_path.is_absolute():
            raise ValueError("postflight execution config must be absolute")
        if self.execution_config_path != (
            self.repository_root / CANONICAL_CONFIG_PATH
        ).resolve():
            raise ValueError("postflight requires the canonical v2 execution config")
        if _SHA256.fullmatch(self.execution_config_sha256) is None:
            raise ValueError("postflight execution-config SHA256 is invalid")
        if _GIT_REVISION.fullmatch(self.expected_git_revision) is None:
            raise ValueError("postflight expected Git revision is invalid")
        if not isinstance(self.worker_schedule, ShardWorkerSchedule):
            raise TypeError("postflight worker schedule is invalid")
        for label, values in (
            ("source audit", self.source_audit),
            ("frozen input", self.frozen_input_sha256),
            ("predecessor", self.predecessor_sha256),
        ):
            if not values or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or _SHA256.fullmatch(value) is None
                for key, value in values.items()
            ):
                raise ValueError(f"postflight {label} hash inventory is invalid")
        if set(self.source_audit) != set(REQUIRED_INPUT_PATHS):
            raise ValueError("postflight source-audit inventory drifted")
        if dict(self.frozen_input_sha256) != dict(self.source_audit):
            raise ValueError("postflight frozen-input/source-audit hashes differ")
        expected_predecessor = {
            "invalid_v1_attempt_summary_sha256": self.source_audit[
                "invalid_v1_attempt_summary"
            ],
            "v1_contract_sha256": self.source_audit["v1_contract"],
            "v1_core_sha256": self.source_audit["v1_core"],
            "v1_execution_config_sha256": self.source_audit[
                "v1_execution_config"
            ],
            "v1_runner_sha256": self.source_audit["v1_runner"],
        }
        if dict(self.predecessor_sha256) != expected_predecessor:
            raise ValueError("postflight predecessor hash chain drifted")


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_selected_image_census_v2_postflight_context(
    contract: SelectedImageCensusContractV2,
    *,
    expected_git_revision: str,
) -> SelectedImageCensusV2PostflightContext:
    """Reconstruct the formal roster solely from the byte-bound v2 contract."""
    if not isinstance(contract, SelectedImageCensusContractV2):
        raise TypeError("postflight requires one validated v2 census contract")
    root = contract.repository_root
    freeze = contract.frozen_inputs["freeze_b_v2_manifest"]
    census = contract.frozen_inputs["p0_census_manifest"]
    inventory = contract.frozen_inputs["full_pool_inventory_manifest"]
    files = tuple(
        SourceFileSpec(item["path"], item["size_bytes"], item["lfs_sha256"])
        for item in inventory["inventory"]["files"]
    )
    plan = build_selected_row_read_plan(
        freeze["assignments"],
        source_files=files,
        source_row_counts=census["source"]["row_counts_by_file"],
    )
    schedule = build_shard_worker_schedule(plan)
    if (
        schedule.selected_trajectory_count != EXPECTED_TRAJECTORY_COUNT
        or schedule.selected_observation_count != EXPECTED_OBSERVATION_COUNT
        or len(schedule.workers) != WORKER_COUNT
    ):
        raise ValueError("v2 postflight roster denominator drifted")
    source_audit = dict(sorted(contract.frozen_input_hashes.items()))
    predecessor_sha256 = {
        "invalid_v1_attempt_summary_sha256": source_audit[
            "invalid_v1_attempt_summary"
        ],
        "v1_contract_sha256": source_audit["v1_contract"],
        "v1_core_sha256": source_audit["v1_core"],
        "v1_execution_config_sha256": source_audit["v1_execution_config"],
        "v1_runner_sha256": source_audit["v1_runner"],
    }
    return SelectedImageCensusV2PostflightContext(
        repository_root=root,
        execution_config_path=(root / CANONICAL_CONFIG_PATH).resolve(),
        execution_config_sha256=contract.config_sha256,
        expected_git_revision=expected_git_revision,
        worker_schedule=schedule,
        source_audit=source_audit,
        frozen_input_sha256=source_audit,
        predecessor_sha256=predecessor_sha256,
    )


def _regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be one real regular file")


def _real_directory(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be one real directory")


def _require_exact_children(
    root: Path,
    *,
    files: set[str] | frozenset[str],
    directories: set[str] | frozenset[str] = frozenset(),
    label: str,
) -> None:
    _real_directory(root, label=label)
    entries = {entry.name: entry for entry in os.scandir(root)}
    expected = set(files) | set(directories)
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise ValueError(f"{label} inventory drifted: missing={missing}, extra={extra}")
    for name in files:
        _regular_file(root / name, label=f"{label}/{name}")
    for name in directories:
        _real_directory(root / name, label=f"{label}/{name}")


def _validate_tree(root: Path) -> None:
    _require_exact_children(
        root,
        files=EXPECTED_TOP_LEVEL_FILES,
        directories=EXPECTED_TOP_LEVEL_DIRECTORIES,
        label="selected-image census v2 output root",
    )
    _require_exact_children(
        root / "records",
        files={f"worker-{index:02d}.jsonl" for index in range(WORKER_COUNT)},
        label="selected-image census v2 records",
    )
    _require_exact_children(
        root / "receipts",
        files={f"worker-{index:02d}.json" for index in range(WORKER_COUNT)},
        label="selected-image census v2 receipts",
    )


def _validate_output_namespace(
    root: Path, context: SelectedImageCensusV2PostflightContext
) -> None:
    expected_name = f"{OUTPUT_NAMESPACE_PREFIX}{context.expected_git_revision[:7]}"
    if root.name != expected_name:
        raise ValueError("v2 output root namespace differs from the producer revision")
    staging = root.parent / f".{root.name}.incomplete"
    if staging.exists() or staging.is_symlink():
        raise ValueError("v2 resumable staging root remains beside completed output")


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, int, str | None], ...]:
    records: list[tuple[str, int, int, str | None]] = []

    def visit(path: Path, relative: str) -> None:
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            records.append((relative, mode, path.stat().st_size, _sha256_file(path)))
            return
        records.append((relative, mode, path.lstat().st_size, None))
        if stat.S_ISDIR(mode) and not path.is_symlink():
            for entry in sorted(os.scandir(path), key=lambda item: item.name):
                child_relative = (
                    entry.name if relative == "." else f"{relative}/{entry.name}"
                )
                visit(path / entry.name, child_relative)

    visit(root, ".")
    return tuple(records)


def _expected_worker_selectors(
    schedule: ShardWorkerSchedule, worker_index: int
) -> tuple[str, ...]:
    worker = schedule.workers[worker_index]
    return tuple(
        selector_identity_sha256(
            p0_selection_sha256=assignment.p0_selection_sha256,
            observation_ordinal=ordinal,
        )
        for shard in worker.shards
        for assignment in sorted(
            shard.assignments, key=lambda item: item.transport_row_index
        )
        for ordinal in range(assignment.decision_count + 1)
    )


def _inspect_worker(
    root: Path,
    *,
    schedule: ShardWorkerSchedule,
    worker_index: int,
) -> tuple[list[SelectedImageCensusRecord], dict[str, Any]]:
    path = root / "records" / f"worker-{worker_index:02d}.jsonl"
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("worker record artifact must be non-empty canonical JSONL")
    records: list[SelectedImageCensusRecord] = []
    for line_index, line in enumerate(payload.splitlines()):
        raw = _strict_json_object(
            line, label=f"worker {worker_index} record {line_index}"
        )
        record = SelectedImageCensusRecord.from_payload(raw)
        if line != canonical_json_bytes(record.to_payload()):
            raise ValueError("worker record line is not canonical compact JSON")
        records.append(record)
    expected_selectors = _expected_worker_selectors(schedule, worker_index)
    selectors = tuple(record.selector_identity_sha256 for record in records)
    if selectors != expected_selectors:
        raise ValueError("worker selector inventory or canonical order drifted")
    descriptor = {
        "byte_count": len(payload),
        "histograms": record_histograms(records),
        "observation_count": len(records),
        "record_sha256": sha256_bytes(payload),
        "selector_inventory_sha256": sha256_bytes(
            canonical_json_bytes(list(selectors))
        ),
        "unique_image_sha256_count": len({record.image_sha256 for record in records}),
        "worker_index": worker_index,
    }
    expected_count = schedule.workers[worker_index].observation_load
    if descriptor["observation_count"] != expected_count:
        raise ValueError("worker observation denominator drifted")
    receipt_path = root / "receipts" / f"worker-{worker_index:02d}.json"
    receipt_payload = receipt_path.read_bytes()
    receipt = _strict_json_object(
        receipt_payload, label=f"worker {worker_index} receipt"
    )
    expected_receipt = {
        "artifact": {
            "byte_count": descriptor["byte_count"],
            "filename": f"records/worker-{worker_index:02d}.jsonl",
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
        "worker_index": worker_index,
    }
    if receipt != expected_receipt:
        raise ValueError("worker receipt differs from record/projection reconstruction")
    if receipt_payload != canonical_json_bytes(expected_receipt, pretty=True):
        raise ValueError("worker receipt is not canonical pretty JSON")
    manifest_descriptor = {
        "observation_count": descriptor["observation_count"],
        "receipt_sha256": sha256_bytes(receipt_payload),
        "record_byte_count": descriptor["byte_count"],
        "record_sha256": descriptor["record_sha256"],
        "selector_inventory_sha256": descriptor["selector_inventory_sha256"],
        "worker_index": worker_index,
    }
    return records, manifest_descriptor


def _aggregate_workers(
    root: Path, context: SelectedImageCensusV2PostflightContext
) -> dict[str, Any]:
    records: list[SelectedImageCensusRecord] = []
    workers: list[dict[str, Any]] = []
    for worker_index in range(WORKER_COUNT):
        worker_records, worker_descriptor = _inspect_worker(
            root,
            schedule=context.worker_schedule,
            worker_index=worker_index,
        )
        records.extend(worker_records)
        workers.append(worker_descriptor)
    selectors = [record.selector_identity_sha256 for record in records]
    if len(selectors) != len(set(selectors)):
        raise ValueError("global selector identity is duplicated")
    if len(records) != context.worker_schedule.selected_observation_count:
        raise ValueError("global selected-observation denominator drifted")
    histograms = record_histograms(records)
    for name, histogram in histograms.items():
        if sum(histogram.values()) != len(records):
            raise ValueError(f"{name} histogram denominator drifted")
    return {
        "histograms": histograms,
        "observation_count": len(records),
        "record_inventory_sha256": sha256_bytes(
            canonical_json_bytes([worker["record_sha256"] for worker in workers])
        ),
        "selector_inventory_sha256": sha256_bytes(
            canonical_json_bytes(selectors)
        ),
        "unique_image_sha256_count": len({record.image_sha256 for record in records}),
        "workers": workers,
    }


def _runtime_cli_keys() -> set[str]:
    return {
        item.removeprefix("--").replace("-", "_")
        for item in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }


def _run_identity(
    root: Path, context: SelectedImageCensusV2PostflightContext
) -> tuple[dict[str, Any], str]:
    path = root / "run-identity.json"
    payload = path.read_bytes()
    value = _strict_json_object(payload, label="v2 run identity")
    if payload != canonical_json_bytes(value, pretty=True):
        raise ValueError("v2 run identity is not canonical pretty JSON")
    expected_keys = {
        "config_sha256",
        "container",
        "git_revision",
        "host",
        "parquet_projection",
        "protocol_id",
        "runtime_cli",
        "schedule",
        "schema_version",
        "source_audit",
        "versions",
    }
    if set(value) != expected_keys:
        raise ValueError("v2 run identity fields drifted")
    if value.get("protocol_id") in {INVALID_V1_PROTOCOL_ID, None}:
        raise ValueError("v1 or missing protocol is forbidden in a v2 output root")
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("v2 run identity protocol or schema drifted")
    if canonical_json_bytes(value.get("parquet_projection")) != canonical_json_bytes(
        parquet_projection_payload()
    ):
        raise ValueError("v2 run identity parquet projection drifted")
    runtime = value.get("runtime_cli")
    if not isinstance(runtime, Mapping) or set(runtime) != _runtime_cli_keys():
        raise ValueError("v2 run identity runtime-CLI fields drifted")
    for key, item in runtime.items():
        if key == "worker_count":
            if type(item) is not int:
                raise ValueError("v2 run identity worker count must be one integer")
        elif (
            not isinstance(item, str)
            or not item
            or item != item.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in item)
        ):
            raise ValueError(f"v2 run identity runtime value {key!r} is invalid")
    path_keys = {
        item.removeprefix("--").replace("-", "_") for item in REQUIRED_PATH_ARGUMENTS
    }
    if any(
        not isinstance(runtime[key], str) or not Path(runtime[key]).is_absolute()
        for key in path_keys
    ):
        raise ValueError("v2 run identity contains a non-absolute runtime path")
    if (
        runtime["repository_root"] != str(context.repository_root.resolve())
        or runtime["execution_config"] != str(context.execution_config_path.resolve())
        or runtime["output_root"] != str(root.resolve())
        or runtime["worker_count"] != WORKER_COUNT
        or runtime["git_revision"] != context.expected_git_revision
    ):
        raise ValueError("v2 run identity fixed runtime values drifted")
    container = value.get("container")
    host = value.get("host")
    versions = value.get("versions")
    if not isinstance(container, Mapping) or set(container) != {"id", "image_digest"}:
        raise ValueError("v2 run identity container fields drifted")
    if (
        not isinstance(container["id"], str)
        or not container["id"]
        or not isinstance(container["image_digest"], str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", container["image_digest"]) is None
        or runtime["container_id"] != container["id"]
        or runtime["container_image_digest"] != container["image_digest"]
    ):
        raise ValueError("v2 run identity container identity drifted")
    if (
        not isinstance(host, Mapping)
        or set(host) != {"alias", "hostname"}
        or any(
            not isinstance(host[key], str) or not host[key]
            for key in ("alias", "hostname")
        )
        or runtime["host_alias"] != host.get("alias")
        or runtime["host_hostname"] != host.get("hostname")
    ):
        raise ValueError("v2 run identity host identity drifted")
    expected_versions = {
        "pillow": runtime["pillow_version"],
        "pyarrow": runtime["pyarrow_version"],
        "python": runtime["python_version"],
    }
    if any(
        not isinstance(version, str) or not version
        for version in expected_versions.values()
    ):
        raise ValueError("v2 run identity runtime versions are invalid")
    expected_schedule = {
        "inventory_sha256": context.worker_schedule.inventory_sha256,
        "selected_observation_count": (
            context.worker_schedule.selected_observation_count
        ),
        "selected_shard_count": context.worker_schedule.selected_shard_count,
        "selected_trajectory_count": context.worker_schedule.selected_trajectory_count,
        "worker_observation_counts": [
            worker.observation_load for worker in context.worker_schedule.workers
        ],
    }
    if (
        value.get("config_sha256") != context.execution_config_sha256
        or value.get("git_revision") != context.expected_git_revision
        or value.get("source_audit") != dict(context.source_audit)
        or versions != expected_versions
        or value.get("schedule") != expected_schedule
    ):
        raise ValueError("v2 run identity fixed fields drifted")
    return value, sha256_bytes(payload)


def _expected_manifest(
    *,
    context: SelectedImageCensusV2PostflightContext,
    aggregate: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    run_identity_sha256: str,
) -> dict[str, Any]:
    bindings = {
        "config_sha256": context.execution_config_sha256,
        "container_image_digest": run_identity["container"]["image_digest"],
        "freeze_b_v2_manifest_sha256": context.frozen_input_sha256[
            "freeze_b_v2_manifest"
        ],
        "git_revision": context.expected_git_revision,
        "run_identity_sha256": run_identity_sha256,
        "schedule_inventory_sha256": context.worker_schedule.inventory_sha256,
        **dict(context.predecessor_sha256),
    }
    return {
        "artifact": {
            "hf_mutation_count": 0,
            "intended_private_hf_repo": HF_REPO,
            "intended_tag": HF_TAG,
            "status": "AWAITING_COMMITTED_POSTFLIGHT",
        },
        "bindings": bindings,
        "counts": {
            "model_or_policy_load_count": 0,
            "ocr_count": 0,
            "selected_observation_count": aggregate["observation_count"],
            "selected_shard_count": context.worker_schedule.selected_shard_count,
            "selected_trajectory_count": (
                context.worker_schedule.selected_trajectory_count
            ),
            "worker_count": WORKER_COUNT,
        },
        "histograms": aggregate["histograms"],
        "protocol_id": PROTOCOL_ID,
        "record_inventory_sha256": aggregate["record_inventory_sha256"],
        "runtime": {
            "device": "cpu",
            "parquet_projection": parquet_projection_payload(),
            "pillow_version": run_identity["versions"]["pillow"],
            "pyarrow_version": run_identity["versions"]["pyarrow"],
            "python_version": run_identity["versions"]["python"],
        },
        "schema_version": SCHEMA_VERSION,
        "selector_inventory_sha256": aggregate["selector_inventory_sha256"],
        "status": FINAL_STATUS,
        "unique_image_sha256_count": aggregate["unique_image_sha256_count"],
        "workers": aggregate["workers"],
    }


def _validate_completed_root(
    root: Path, context: SelectedImageCensusV2PostflightContext
) -> dict[str, Any]:
    _validate_tree(root)
    aggregate = _aggregate_workers(root, context)
    run_identity, run_identity_sha256 = _run_identity(root, context)
    manifest_path = root / "manifest.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = _strict_json_object(manifest_payload, label="v2 completion manifest")
    if manifest.get("protocol_id") == INVALID_V1_PROTOCOL_ID or manifest.get(
        "status"
    ) == INVALID_V1_FINAL_STATUS:
        raise ValueError("invalid v1 output root cannot satisfy v2 postflight")
    expected_manifest = _expected_manifest(
        context=context,
        aggregate=aggregate,
        run_identity=run_identity,
        run_identity_sha256=run_identity_sha256,
    )
    if manifest != expected_manifest:
        raise ValueError(
            "v2 completion manifest differs from whole-root reconstruction"
        )
    if manifest_payload != canonical_json_bytes(expected_manifest, pretty=True):
        raise ValueError("v2 completion manifest is not canonical pretty JSON")
    return {
        "config_sha256": context.execution_config_sha256,
        "git_revision": context.expected_git_revision,
        "manifest_sha256": sha256_bytes(manifest_payload),
        "observation_count": aggregate["observation_count"],
        "parquet_projection": parquet_projection_payload(),
        "protocol_id": PROTOCOL_ID,
        "record_inventory_sha256": aggregate["record_inventory_sha256"],
        "run_identity_sha256": run_identity_sha256,
        "schema_version": SCHEMA_VERSION,
        "selector_inventory_sha256": aggregate["selector_inventory_sha256"],
        "status": VALIDATION_STATUS,
        "unique_image_sha256_count": aggregate["unique_image_sha256_count"],
    }


def validate_completed_selected_image_census_v2_root(
    output_root: str | Path,
    *,
    context: SelectedImageCensusV2PostflightContext,
) -> dict[str, Any]:
    """Validate a completed v2 root and prove that postflight did not mutate it."""
    if not isinstance(context, SelectedImageCensusV2PostflightContext):
        raise TypeError("postflight context is invalid")
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("postflight output root must be absolute")
    _validate_output_namespace(root, context)
    before = _tree_snapshot(root)
    validation_error: BaseException | None = None
    try:
        result = _validate_completed_root(root, context)
    except BaseException as error:
        validation_error = error
        raise
    finally:
        try:
            after = _tree_snapshot(root)
        except BaseException as snapshot_error:
            raise RuntimeError(
                "v2 output tree changed during read-only postflight"
            ) from snapshot_error
        if after != before:
            message = "v2 output tree changed during read-only postflight"
            if validation_error is None:
                raise RuntimeError(message)
            raise RuntimeError(message) from validation_error
    result["output_tree_snapshot_sha256"] = sha256_bytes(
        canonical_json_bytes([list(item) for item in before])
    )
    return result


__all__ = [
    "INVALID_V1_FINAL_STATUS",
    "INVALID_V1_PROTOCOL_ID",
    "OUTPUT_NAMESPACE_PREFIX",
    "SelectedImageCensusV2PostflightContext",
    "VALIDATION_STATUS",
    "build_selected_image_census_v2_postflight_context",
    "validate_completed_selected_image_census_v2_root",
]
