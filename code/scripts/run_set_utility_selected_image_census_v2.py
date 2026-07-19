#!/usr/bin/env python3
"""Run the frozen CPU-only v2 selected-image column-projection repair."""

from __future__ import annotations

import argparse
import ctypes
import errno
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.set_utility_processor_freeze import build_shard_worker_schedule
from causalcache.set_utility_processor_substrate import build_selected_row_read_plan
from causalcache.set_utility_selected_image_census_contract_v2 import (
    HF_REPO,
    HF_TAG,
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_PATH_ARGUMENTS,
    load_execution_contract,
    validate_cpu_only_source,
    validate_runtime_cli_values,
)
from causalcache.set_utility_selected_image_census_v2 import (
    EXPECTED_OBSERVATION_COUNT,
    EXPECTED_TRAJECTORY_COUNT,
    FINAL_STATUS,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    aggregate_worker_records,
    build_selected_image_census_records_from_payloads,
    canonical_json_bytes,
    materialize_worker_records,
    parquet_projection_payload,
    sha256_bytes,
    sha256_file,
)


def _source_path(source_root: Path, transport_file: str) -> Path:
    path = source_root.joinpath(*PurePosixPath(transport_file).parts)
    try:
        path.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise ValueError("source shard escaped the explicit source root") from error
    return path


class ParquetRowFactory:
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root.resolve()

    def __call__(self, spec: SourceFileSpec) -> Iterator[tuple[int, Mapping[str, Any]]]:
        path = _source_path(self.source_root, spec.transport_file)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != spec.size_bytes
        ):
            raise ValueError(f"pinned source shard drifted: {spec.transport_file}")
        if sha256_file(path) != spec.sha256:
            raise ValueError(f"pinned source shard SHA256 drifted: {spec.transport_file}")
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=8, columns=["images"]):
            if list(batch.schema.names) != ["images"]:
                raise ValueError("projected parquet batch schema must be exactly images")
            for row in batch.to_pylist():
                if not isinstance(row, Mapping):
                    raise TypeError("projected GUIOdyssey parquet row must be one mapping")
                if set(row) != {"images"}:
                    raise ValueError("projected parquet row keys must be exactly images")
                yield row_index, row
                row_index += 1
        if row_index != parquet.metadata.num_rows:
            raise ValueError("source shard row count drifted during projected streaming")


def _load_source_plan(contract: Any) -> tuple[Any, Any]:
    freeze = contract.frozen_inputs["freeze_b_v2_manifest"]
    census = contract.frozen_inputs["p0_census_manifest"]
    inventory = contract.frozen_inputs["full_pool_inventory_manifest"]
    files = tuple(
        SourceFileSpec(record["path"], record["size_bytes"], record["lfs_sha256"])
        for record in inventory["inventory"]["files"]
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
    ):
        raise ValueError("selected-image worker schedule denominator drifted")
    return plan, schedule


def _runtime_values(args: argparse.Namespace) -> dict[str, Any]:
    names = {
        name.removeprefix("--").replace("-", "_")
        for name in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }
    return {name: getattr(args, name) for name in names}


def _verify_checkout(root: Path, revision: str) -> None:
    observed = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != revision:
        raise ValueError("formal census v2 Git revision differs from checkout HEAD")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    if status:
        raise ValueError("formal selected-image census v2 requires a clean Git checkout")


def _verify_executable(path: Path, *, expected_version: str) -> None:
    if (
        not path.is_absolute()
        or not path.resolve().is_file()
        or not os.access(path, os.X_OK)
    ):
        raise ValueError("Python executable must be one explicit absolute executable")
    observed = subprocess.check_output(
        [str(path), "-c", "import platform; print(platform.python_version())"],
        text=True,
    ).strip()
    if observed != expected_version:
        raise ValueError("Python executable version differs from explicit CLI value")


def _verify_runtime_versions(args: argparse.Namespace) -> None:
    observed = {
        "python": platform.python_version(),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "pillow": importlib.metadata.version("Pillow"),
    }
    expected = {
        "python": args.python_version,
        "pyarrow": args.pyarrow_version,
        "pillow": args.pillow_version,
    }
    if observed != expected:
        raise ValueError(f"selected-image census v2 runtime versions drifted: {observed!r}")


def _assert_cpu_only_import_state() -> None:
    forbidden = {"onnxruntime", "rapidocr", "torch", "transformers"}
    observed = {name.split(".")[0] for name in sys.modules}
    loaded = sorted(forbidden & observed)
    if loaded:
        raise RuntimeError(f"CPU-only census v2 imported forbidden runtimes: {loaded!r}")


def _worker_records(args: argparse.Namespace, contract: Any) -> None:
    _verify_runtime_versions(args)
    _assert_cpu_only_import_state()
    _, schedule = _load_source_plan(contract)
    worker = schedule.workers[args.worker_index]
    row_factory = ParquetRowFactory(args.source_root)

    def records() -> Iterator[Any]:
        for shard in worker.shards:
            wanted = {
                assignment.transport_row_index: assignment
                for assignment in shard.assignments
            }
            found: set[int] = set()
            expected_row_index = 0
            for row_index, row in row_factory(shard.source_file):
                if row_index != expected_row_index:
                    raise ValueError("selected shard row order drifted")
                assignment = wanted.get(row_index)
                if assignment is not None:
                    images = row.get("images")
                    if not isinstance(images, list):
                        raise ValueError("selected row image column must be one list")
                    if len(images) != assignment.decision_count + 1:
                        raise ValueError("selected row image denominator drifted")
                    payloads: list[bytes] = []
                    for image in images:
                        if not isinstance(image, Mapping):
                            raise ValueError("selected row image entry must be one mapping")
                        payload = image.get("bytes")
                        if not isinstance(payload, bytes) or not payload:
                            raise ValueError("selected row image bytes are missing")
                        payloads.append(payload)
                    yield from build_selected_image_census_records_from_payloads(
                        p0_selection_sha256=assignment.p0_selection_sha256,
                        image_payloads=payloads,
                    )
                    found.add(row_index)
                expected_row_index += 1
            if expected_row_index != shard.source_row_count:
                raise ValueError("selected shard row denominator drifted")
            if found != set(wanted):
                raise ValueError("selected shard is missing one frozen row")

    materialize_worker_records(
        args.staging_root,
        worker_index=args.worker_index,
        expected_observation_count=worker.observation_load,
        records=records(),
    )
    _assert_cpu_only_import_state()


def _write_once_or_verify(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(f"resumable artifact drifted: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def _worker_subprocess_args(
    args: argparse.Namespace, *, worker_index: int, staging: Path
) -> list[str]:
    values: list[str] = [str(Path(__file__).resolve())]
    for name in REQUIRED_PATH_ARGUMENTS:
        key = name.removeprefix("--").replace("-", "_")
        values.extend([name, str(getattr(args, key))])
    for name in REQUIRED_IDENTITY_ARGUMENTS:
        key = name.removeprefix("--").replace("-", "_")
        values.extend([name, str(getattr(args, key))])
    values.extend(
        [
            "--phase",
            "worker",
            "--worker-index",
            str(worker_index),
            "--staging-root",
            str(staging),
        ]
    )
    return values


def _next_log_path(log_dir: Path, worker_index: int) -> Path:
    for attempt in range(1_000):
        path = log_dir / f"worker-{worker_index:02d}-attempt-{attempt:03d}.log"
        if not path.exists() and not path.is_symlink():
            return path
    raise RuntimeError("worker log attempt inventory is exhausted")


def _launch_workers(args: argparse.Namespace, staging: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    for worker_index in range(4):
        log_path = _next_log_path(staging / "logs", worker_index)
        log = log_path.open("xb")
        process = subprocess.Popen(
            [
                str(args.python_executable),
                *_worker_subprocess_args(
                    args, worker_index=worker_index, staging=staging
                ),
            ],
            env=environment,
            stderr=subprocess.STDOUT,
            stdout=log,
        )
        processes.append((worker_index, process, log))
    failures: list[tuple[int, int]] = []
    for worker_index, process, log in processes:
        return_code = process.wait()
        log.close()
        if return_code != 0:
            failures.append((worker_index, return_code))
    if failures:
        raise RuntimeError(f"selected-image census v2 workers failed: {failures!r}")


def _prepare_staging(output_root: Path) -> Path:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("selected-image census v2 output root already exists")
    staging = output_root.parent / f".{output_root.name}.incomplete"
    if staging.exists() or staging.is_symlink():
        if not staging.is_dir() or staging.is_symlink():
            raise ValueError("resumable staging root must be one regular directory")
    else:
        staging.mkdir()
    for name in ("logs", "receipts", "records"):
        path = staging / name
        try:
            path.mkdir()
        except FileExistsError:
            pass
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"staging {name} must be one regular directory")
    allowed = {
        "logs",
        "manifest.json",
        "receipts",
        "records",
        "run-identity.json",
    }
    if {path.name for path in staging.iterdir()} - allowed:
        raise ValueError("staging root contains an unexpected entry")
    return staging


def _remove_success_logs(staging: Path) -> None:
    log_dir = staging / "logs"
    for path in log_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError("worker log inventory contains a non-regular file")
        path.unlink()
    log_dir.rmdir()


def _validate_final_inventory(staging: Path) -> None:
    expected_root = {
        "manifest.json",
        "receipts",
        "records",
        "run-identity.json",
    }
    if {path.name for path in staging.iterdir()} != expected_root:
        raise ValueError("final census v2 root inventory drifted")
    for directory, suffix in (("records", ".jsonl"), ("receipts", ".json")):
        root = staging / directory
        if not root.is_dir() or root.is_symlink():
            raise ValueError("final census v2 directory inventory is invalid")
        expected = {f"worker-{index:02d}{suffix}" for index in range(4)}
        if {path.name for path in root.iterdir()} != expected:
            raise ValueError(f"final census v2 {directory} inventory drifted")
        if any(not path.is_file() or path.is_symlink() for path in root.iterdir()):
            raise ValueError(
                f"final census v2 {directory} contains a non-regular file"
            )
    for filename in ("manifest.json", "run-identity.json"):
        path = staging / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"final census v2 {filename} must be one regular file")


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
            if result == 0:
                return
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(destination)
            raise OSError(error, os.strerror(error), destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    os.rename(source, destination)


def _orchestrate(args: argparse.Namespace, contract: Any) -> None:
    _verify_runtime_versions(args)
    validate_cpu_only_source(repository_root=args.repository_root)
    _assert_cpu_only_import_state()
    _, schedule = _load_source_plan(contract)
    staging = _prepare_staging(args.output_root)
    runtime_cli = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(_runtime_values(args).items())
    }
    run_identity = {
        "config_sha256": contract.config_sha256,
        "container": {
            "id": args.container_id,
            "image_digest": args.container_image_digest,
        },
        "git_revision": args.git_revision,
        "host": {
            "alias": args.host_alias,
            "hostname": args.host_hostname,
        },
        "parquet_projection": parquet_projection_payload(),
        "protocol_id": PROTOCOL_ID,
        "runtime_cli": runtime_cli,
        "schedule": {
            "inventory_sha256": schedule.inventory_sha256,
            "selected_observation_count": schedule.selected_observation_count,
            "selected_shard_count": schedule.selected_shard_count,
            "selected_trajectory_count": schedule.selected_trajectory_count,
            "worker_observation_counts": [
                worker.observation_load for worker in schedule.workers
            ],
        },
        "schema_version": SCHEMA_VERSION,
        "source_audit": dict(sorted(contract.frozen_input_hashes.items())),
        "versions": {
            "pillow": args.pillow_version,
            "pyarrow": args.pyarrow_version,
            "python": args.python_version,
        },
    }
    run_identity_payload = canonical_json_bytes(run_identity, pretty=True)
    _write_once_or_verify(staging / "run-identity.json", run_identity_payload)
    _launch_workers(args, staging)
    worker_counts = tuple(worker.observation_load for worker in schedule.workers)
    aggregate = aggregate_worker_records(
        staging, expected_worker_observation_counts=worker_counts
    )
    if aggregate["observation_count"] != EXPECTED_OBSERVATION_COUNT:
        raise ValueError("final selected-image census v2 denominator drifted")
    hashes = contract.frozen_input_hashes
    manifest = {
        "artifact": {
            "hf_mutation_count": 0,
            "intended_private_hf_repo": HF_REPO,
            "intended_tag": HF_TAG,
            "status": "AWAITING_COMMITTED_POSTFLIGHT",
        },
        "bindings": {
            "config_sha256": contract.config_sha256,
            "container_image_digest": args.container_image_digest,
            "freeze_b_v2_manifest_sha256": hashes["freeze_b_v2_manifest"],
            "git_revision": args.git_revision,
            "invalid_v1_attempt_summary_sha256": hashes[
                "invalid_v1_attempt_summary"
            ],
            "run_identity_sha256": sha256_bytes(run_identity_payload),
            "schedule_inventory_sha256": schedule.inventory_sha256,
            "v1_contract_sha256": hashes["v1_contract"],
            "v1_core_sha256": hashes["v1_core"],
            "v1_execution_config_sha256": hashes["v1_execution_config"],
            "v1_runner_sha256": hashes["v1_runner"],
        },
        "counts": {
            "model_or_policy_load_count": 0,
            "ocr_count": 0,
            "selected_observation_count": aggregate["observation_count"],
            "selected_shard_count": schedule.selected_shard_count,
            "selected_trajectory_count": schedule.selected_trajectory_count,
            "worker_count": 4,
        },
        "histograms": aggregate["histograms"],
        "protocol_id": PROTOCOL_ID,
        "record_inventory_sha256": aggregate["record_inventory_sha256"],
        "runtime": {
            "device": "cpu",
            "parquet_projection": parquet_projection_payload(),
            "pillow_version": args.pillow_version,
            "pyarrow_version": args.pyarrow_version,
            "python_version": args.python_version,
        },
        "schema_version": SCHEMA_VERSION,
        "selector_inventory_sha256": aggregate["selector_inventory_sha256"],
        "status": FINAL_STATUS,
        "unique_image_sha256_count": aggregate["unique_image_sha256_count"],
        "workers": aggregate["workers"],
    }
    _write_once_or_verify(
        staging / "manifest.json", canonical_json_bytes(manifest, pretty=True)
    )
    _remove_success_logs(staging)
    _validate_final_inventory(staging)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("output root appeared before atomic publication")
    _rename_directory_no_replace(staging, args.output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in REQUIRED_PATH_ARGUMENTS:
        parser.add_argument(name, type=Path, required=True)
    for name in REQUIRED_IDENTITY_ARGUMENTS:
        parser.add_argument(
            name,
            required=True,
            type=int if name == "--worker-count" else str,
        )
    parser.add_argument("--phase", choices=("orchestrate", "worker"), default="orchestrate")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--staging-root", type=Path)
    args = parser.parse_args()
    for name in REQUIRED_PATH_ARGUMENTS:
        key = name.removeprefix("--").replace("-", "_")
        if key != "python_executable":
            setattr(args, key, getattr(args, key).resolve())
    if args.phase == "worker":
        if args.worker_index not in range(4) or args.staging_root is None:
            parser.error("worker phase requires --worker-index in [0,3] and --staging-root")
        args.staging_root = args.staging_root.resolve()
        expected = args.output_root.parent / f".{args.output_root.name}.incomplete"
        if args.staging_root != expected:
            parser.error("worker staging root differs from derived resumable path")
    return args


def main() -> None:
    args = parse_args()
    contract = load_execution_contract(
        repository_root=args.repository_root,
        execution_config_path=args.execution_config,
    )
    validate_runtime_cli_values(contract, _runtime_values(args))
    _verify_checkout(args.repository_root, args.git_revision)
    if not args.container_id.startswith(socket.gethostname()):
        raise ValueError("container ID does not match running container hostname")
    _verify_executable(args.python_executable, expected_version=args.python_version)
    if args.phase == "orchestrate":
        _orchestrate(args, contract)
    else:
        _worker_records(args, contract)


if __name__ == "__main__":
    main()
