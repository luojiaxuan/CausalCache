#!/usr/bin/env python3
"""Run the isolated Execution-CF v2 image-contract processor freeze."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import queue
import socket
import subprocess
import sys
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

# note (luojiaxuan): Keep workers self-contained without ambient PYTHONPATH input.
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from causalcache.policy.gui_owl_v2_1 import GUI_OWL_V2_1_MOBILE_USE_TOOL
from causalcache.set_utility_processor_artifacts import (
    build_final_candidate_schedule,
    build_git_safe_processor_freeze_summary,
    build_prefix_safe_processor_artifact_record,
    inspect_processor_artifact_shard,
    iter_processor_query_artifact_records,
    materialize_final_candidate_schedule,
    materialize_processor_artifact_shard,
)
from causalcache.set_utility_processor_freeze import (
    TARGET_PIXELS_PER_IMAGE,
    WORKER_COUNT,
    FrozenQueryCandidateRecord,
    assert_processor_only_import_state,
    canonical_json_bytes,
    freeze_query_candidates,
    sha256_bytes,
    sha256_file,
    verify_processor_only_snapshot,
    write_once_or_verify,
)
from causalcache.set_utility_processor_freeze_contract_v2 import (
    OCR_CONCURRENCY_PER_LOGICAL_WORKER,
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_PATH_ARGUMENTS,
    REQUIRED_VERSION_ARGUMENTS_BY_PHASE,
    load_execution_contract,
    validate_runtime_cli_values,
)
from causalcache.set_utility_processor_image_contract_v2 import (
    PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS,
    build_validated_ocr_batch_v2,
    decode_processor_image_v2,
    prepare_processor_image_v2,
    run_processor_rapidocr_record_v2,
    validate_processor_image_v2,
)
from causalcache.set_utility_processor_postflight_v2 import (
    validate_processor_only_source_v2,
)
from causalcache.set_utility_processor_substrate import (
    SelectedRowReadPlan,
    SourceProvenance,
    build_selected_pilot,
    build_trajectory_processor_substrate,
    load_selected_rows_once,
)


RUNNER_RELATIVE_PATH = "code/scripts/run_set_utility_processor_freeze_v2.py"
V1_RUNNER_RELATIVE_PATH = "code/scripts/run_set_utility_processor_freeze.py"
IMAGE_CONTRACT_RELATIVE_PATH = (
    "code/causalcache/set_utility_processor_image_contract_v2.py"
)
OUTPUT_BASENAME_PREFIX = (
    "causalcache-set-utility-processor-freeze-v2-image-contract-repair-"
)
COMPLETED_STATUS = "PROCESSOR_ONLY_CANDIDATE_FREEZE_COMPLETED"


def _bounded_ordered_parallel_map(
    function: Callable[[Any], Any],
    values: Iterable[Any],
    *,
    max_workers: int,
) -> Iterator[Any]:
    """Execute a bounded number of tasks while yielding source order."""
    if type(max_workers) is not int or max_workers <= 0:
        raise ValueError("parallel-map worker count must be positive")
    iterator = iter(values)
    pending: deque[Future[Any]] = deque()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_workers * 2):
            try:
                value = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(function, value))
        while pending:
            yield pending.popleft().result()
            try:
                value = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(function, value))


def _load_v1_runner_module() -> ModuleType:
    """Load the frozen v1 runner only as a library of unchanged pure helpers."""
    path = _CODE_ROOT / "scripts" / "run_set_utility_processor_freeze.py"
    spec = importlib.util.spec_from_file_location(
        "_causalcache_frozen_processor_runner_v1", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not bind the frozen v1 processor runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_V1 = _load_v1_runner_module()


def _image_contract_sha256(repository_root: Path) -> str:
    return sha256_file(repository_root / IMAGE_CONTRACT_RELATIVE_PATH)


def _runtime_values(args: argparse.Namespace) -> dict[str, Any]:
    names = {
        value.removeprefix("--").replace("-", "_")
        for value in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }
    names.update(
        value.removeprefix("--").replace("-", "_")
        for values in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.values()
        for value in values
    )
    return {name: getattr(args, name) for name in names}


def _validate_resumable_ocr_receipt_v2(
    receipt: Mapping[str, Any],
    *,
    descriptor: Any,
    backend_config_sha256: str,
    ocr_runtime_identity: Mapping[str, Any],
    worker_index: int,
) -> None:
    format_tally = receipt.get("format_tally")
    if (
        set(receipt)
        != {
            "backend_config_sha256",
            "format_tally",
            "ocr_runtime_identity",
            "shard",
            "worker_index",
        }
        or receipt.get("backend_config_sha256") != backend_config_sha256
        or receipt.get("shard") != descriptor.to_payload()
        or receipt.get("ocr_runtime_identity") != ocr_runtime_identity
        or receipt.get("worker_index") != worker_index
        or not isinstance(format_tally, Mapping)
        or not format_tally
        or any(
            key not in PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS
            or type(value) is not int
            or value <= 0
            for key, value in format_tally.items()
        )
        or sum(format_tally.values()) != descriptor.observation_count
    ):
        raise ValueError(
            "resumable v2 OCR receipt differs from current runtime or shard"
        )


def _run_ocr_worker(args: argparse.Namespace, contract: Any, staging: Path) -> None:
    _V1._verify_current_versions(args, phase="ocr")
    base, _, queries, plan, schedule = _V1._load_source_inputs(contract)
    worker = schedule.workers[args.worker_index]
    read_plan = _V1._worker_read_plan(plan, worker.transport_files)
    artifact_path = staging / "substrate" / worker.filename
    receipt_path = staging / "receipts" / f"ocr-worker-{args.worker_index:02d}.json"
    contract_sha = _image_contract_sha256(contract.repository_root)
    raw_ocr_phase = contract.data["phases"]["raw_decode_and_ocr"]
    concurrency = raw_ocr_phase.get("ocr_concurrency_per_logical_worker")
    if concurrency != OCR_CONCURRENCY_PER_LOGICAL_WORKER:
        raise ValueError("OCR execution concurrency differs from the frozen contract")
    engine, backend, backend_sha, identity, _ = _V1._ocr_engine(
        args, contract.repository_root
    )
    identity = {
        **identity,
        "image_contract_sha256": contract_sha,
        "ocr_concurrency_per_logical_worker": concurrency,
    }
    if artifact_path.exists() and receipt_path.exists():
        descriptor = inspect_processor_artifact_shard(
            artifact_path, expected_worker=worker
        )
        receipt = _V1._load_json(receipt_path)
        _validate_resumable_ocr_receipt_v2(
            receipt,
            descriptor=descriptor,
            backend_config_sha256=backend_sha,
            ocr_runtime_identity=identity,
            worker_index=args.worker_index,
        )
        return
    if artifact_path.exists() != receipt_path.exists():
        raise ValueError("incomplete OCR artifact/receipt pair cannot be resumed safely")

    source_pool = base["source_pool"]
    provenance = SourceProvenance(
        upstream_repo=source_pool["upstream_repo"],
        upstream_revision=source_pool["upstream_revision"],
        transport_repo=source_pool["transport_repo"],
        transport_revision=source_pool["transport_revision"],
        derived_repo=contract.freeze_b_v2_manifest["artifacts"]["dataset"][
            "canonical_repo"
        ],
        coordinate_grid_size=int(base["policy"]["coordinate_grid_size"]),
    )
    row_factory = _V1.ParquetRowFactory(args.source_root)
    worker_format_counts: Counter[tuple[str, str]] = Counter()

    from causalcache.restoration_v2_text_backend import create_rapidocr_engine

    engine_pool: queue.Queue[Any] = queue.Queue()
    engine_pool.put(engine)
    for _ in range(concurrency - 1):
        engine_pool.put(create_rapidocr_engine(backend, args.ocr_model_dir))

    def pilots() -> Iterator[Any]:
        for shard in read_plan.shards:
            single = SelectedRowReadPlan(
                shards=(shard,), assignment_count=len(shard.assignments)
            )
            loaded_rows = load_selected_rows_once(
                single,
                row_iterator_factory=row_factory,
                inspection_config=base,
            )
            for loaded in sorted(
                loaded_rows,
                key=lambda item: item.assignment.transport_row_index,
            ):
                yield build_selected_pilot(loaded, provenance=provenance), queries[
                    loaded.assignment.source_id
                ]

    def process_pilot(value: Any) -> tuple[Any, Counter[tuple[str, str]]]:
        pilot, query_states = value
        required = {
            str(event[field])
            for event in pilot.manifest["trajectory"]["events"]
            for field in (
                "observation_before_path",
                "observation_after_path",
            )
        }
        format_counts: Counter[tuple[str, str]] = Counter()
        for path in sorted(required):
            prepared = prepare_processor_image_v2(pilot.image_payloads[path])
            validate_processor_image_v2(prepared)
            format_counts[(prepared.source_format, prepared.source_mode)] += 1
        if sum(format_counts.values()) != len(required):
            raise RuntimeError("pre-OCR v2 image-format tally drifted")

        slot_engine = engine_pool.get()
        try:
            captured_ocr: dict[str, Mapping[str, Any]] = {}

            def batch_builder(images: Mapping[str, bytes]) -> Any:
                batch = build_validated_ocr_batch_v2(
                    images,
                    engine=slot_engine,
                    backend_config=backend,
                    backend_config_sha256=backend_sha,
                    record_runner=run_processor_rapidocr_record_v2,
                )
                captured_ocr.update(batch.records_by_path)
                return batch

            substrate = build_trajectory_processor_substrate(
                pilot,
                query_states=query_states,
                ocr_batch_builder=batch_builder,
            )
            record = build_prefix_safe_processor_artifact_record(
                substrate,
                image_payloads=pilot.image_payloads,
                ocr_records_by_path=captured_ocr,
            )
        finally:
            engine_pool.put(slot_engine)
        return record, format_counts

    def records() -> Iterator[Any]:
        for record, format_counts in _bounded_ordered_parallel_map(
            process_pilot,
            pilots(),
            max_workers=concurrency,
        ):
            worker_format_counts.update(format_counts)
            yield record

    descriptor = materialize_processor_artifact_shard(
        staging / "substrate",
        worker=worker,
        records=records(),
        resume=True,
    )
    receipt = {
        "backend_config_sha256": backend_sha,
        "format_tally": {
            f"{key[0]}:{key[1]}": value
            for key, value in sorted(worker_format_counts.items())
        },
        "ocr_runtime_identity": identity,
        "shard": descriptor.to_payload(),
        "worker_index": args.worker_index,
    }
    write_once_or_verify(receipt_path, canonical_json_bytes(receipt))


class ProcessorLengthRuntimeV2:
    """Processor-only replay whose image decode is owned by the v2 contract."""

    def __init__(self, model_dir: Path) -> None:
        assert_processor_only_import_state()
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            str(model_dir),
            trust_remote_code=False,
            local_files_only=True,
            min_pixels=TARGET_PIXELS_PER_IMAGE,
            max_pixels=TARGET_PIXELS_PER_IMAGE,
        )
        assert_processor_only_import_state()

    def length(self, plan: Any, images: Mapping[str, bytes]) -> int:
        opened: list[Any] = []

        def decode(payload: bytes) -> Any:
            source = decode_processor_image_v2(payload)
            try:
                image = source.convert("RGB")
            finally:
                source.close()
            opened.append(image)
            return image

        try:
            messages = _V1.build_set_utility_gui_owl_v2_1_messages(
                plan,
                image_bytes_loader=lambda reference: images[reference],
                image_decoder=decode,
            )
            encoded = self.processor.apply_chat_template(
                [messages],
                tools=[copy.deepcopy(GUI_OWL_V2_1_MOBILE_USE_TOOL)],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=False,
            )
            if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                raise TypeError("AutoProcessor did not return an input_ids mapping")
            input_ids = encoded["input_ids"]
            if len(input_ids.shape) != 2 or int(input_ids.shape[0]) != 1:
                raise ValueError("AutoProcessor input_ids shape drifted")
            length = int(input_ids.shape[1])
            del input_ids
            del encoded
            return length
        finally:
            for image in opened:
                image.close()
            assert_processor_only_import_state()


def _run_processor_worker(
    args: argparse.Namespace, contract: Any, staging: Path
) -> None:
    _V1._verify_current_versions(args, phase="processor")
    _, _, _, _, schedule = _V1._load_source_inputs(contract)
    worker = schedule.workers[args.worker_index]
    shard = staging / "substrate" / worker.filename
    runtime = ProcessorLengthRuntimeV2(args.model_dir)
    records: list[FrozenQueryCandidateRecord] = []
    for query in iter_processor_query_artifact_records(shard, expected_worker=worker):
        trajectory_payload = {
            "trajectory_id": query.trajectory_id,
            "source_id": query.trajectory_id,
            "role": query.role,
            "task_instruction": query.task_instruction,
            "history_events": [dict(event) for event in query.history_events],
        }
        query_payload = {
            "state_id": query.state_id,
            "query_kind": query.query_kind,
            "decision_step_id": query.decision_step_id,
            "current_equivalent_event_step_id": (
                query.current_equivalent_event_step_id
            ),
            "initial_candidate_event_step_ids": list(
                query.initial_candidate_event_step_ids
            ),
            "maximum_labeled_cardinality": query.maximum_labeled_cardinality,
            "current_observation_ref": query.current_observation_ref,
        }
        records.append(
            freeze_query_candidates(
                trajectory_payload,
                query_payload,
                processor_only_length=lambda plan, images=query.image_payloads: runtime.length(
                    plan, images
                ),
            )
        )
    payload = b"".join(
        canonical_json_bytes(record.to_payload()) + b"\n"
        for record in sorted(records, key=lambda item: item.state_id)
    )
    write_once_or_verify(
        staging / "candidate-parts" / f"worker-{args.worker_index:02d}.jsonl",
        payload,
    )


def _subprocess_args(
    args: argparse.Namespace, *, phase: str, worker: int, staging: Path
) -> list[str]:
    result = [str(args.repository_root / RUNNER_RELATIVE_PATH)]
    names = [
        item.removeprefix("--").replace("-", "_")
        for item in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    ]
    names.extend(
        item.removeprefix("--").replace("-", "_")
        for values in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.values()
        for item in values
    )
    for name in names:
        result.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
    result.extend(
        [
            "--phase",
            phase,
            "--worker-index",
            str(worker),
            "--staging-root",
            str(staging),
        ]
    )
    return result


def _launch_workers(args: argparse.Namespace, *, phase: str, staging: Path) -> None:
    executable = (
        args.ocr_python_executable
        if phase == "ocr-worker"
        else args.processor_python_executable
    )
    processes = []
    for worker in range(WORKER_COUNT):
        log_path = staging / "logs" / f"{phase}-{worker:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("ab")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        process = subprocess.Popen(
            [
                str(executable),
                *_subprocess_args(
                    args, phase=phase, worker=worker, staging=staging
                ),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes.append((process, log, log_path))
    failures = []
    for process, log, log_path in processes:
        code = process.wait()
        log.close()
        if code != 0:
            failures.append(f"{log_path}:{code}")
    if failures:
        raise RuntimeError(f"{phase} failed: {', '.join(failures)}")


def _orchestrate(args: argparse.Namespace, contract: Any) -> None:
    if Path(sys.executable).resolve() != args.processor_python_executable.resolve():
        raise ValueError("orchestrator must run under --processor-python-executable")
    _V1._verify_current_versions(args, phase="processor")
    _, _, _, _, schedule = _V1._load_source_inputs(contract)
    source_audit = validate_processor_only_source_v2(args.repository_root)
    snapshot = verify_processor_only_snapshot(
        model_dir=args.model_dir,
        snapshot_manifest=args.snapshot_manifest,
    )
    staging = args.output_root.parent / f".{args.output_root.name}.incomplete"
    if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
        raise ValueError("resumable v2 staging root must be one real directory")
    staging.mkdir(parents=False, exist_ok=True)
    run_identity = {
        "config_sha256": contract.config_sha256,
        "git_revision": args.git_revision,
        "runtime_cli": {
            key: str(value)
            for key, value in sorted(_runtime_values(args).items())
        },
        "source_audit": source_audit,
        "snapshot": snapshot,
        "worker_execution_sha256": schedule.execution_sha256,
    }
    write_once_or_verify(
        staging / "run-identity.json", canonical_json_bytes(run_identity)
    )
    _launch_workers(args, phase="ocr-worker", staging=staging)
    _launch_workers(args, phase="processor-worker", staging=staging)
    records = _V1._read_candidate_parts(staging)
    candidate_schedule = build_final_candidate_schedule(records)
    candidate_descriptor = materialize_final_candidate_schedule(
        staging, schedule=candidate_schedule, resume=True
    )
    shards = tuple(
        inspect_processor_artifact_shard(
            staging / "substrate" / worker.filename,
            expected_worker=worker,
        )
        for worker in schedule.workers
    )
    summary = build_git_safe_processor_freeze_summary(
        worker_schedule=schedule,
        artifact_shards=shards,
        candidate_schedule=candidate_schedule,
        candidate_artifact=candidate_descriptor,
    )
    manifest = {
        "contains_model_or_policy_output": False,
        "contains_restoration_labels": False,
        "execution_config_sha256": contract.config_sha256,
        "git_revision": args.git_revision,
        "processor_freeze_summary": summary,
        "run_identity_sha256": sha256_bytes(canonical_json_bytes(run_identity)),
        "status": COMPLETED_STATUS,
    }
    write_once_or_verify(
        staging / "manifest.json", canonical_json_bytes(manifest, pretty=True)
    )
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("processor-freeze output appeared before atomic publication")
    os.rename(staging, args.output_root)


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
    for values in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.values():
        for name in values:
            parser.add_argument(name, required=True)
    parser.add_argument(
        "--phase",
        choices=("orchestrate", "ocr-worker", "processor-worker"),
        default="orchestrate",
    )
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--staging-root", type=Path)
    args = parser.parse_args()
    executable_keys = {"ocr_python_executable", "processor_python_executable"}
    for name in REQUIRED_PATH_ARGUMENTS:
        key = name.removeprefix("--").replace("-", "_")
        if key not in executable_keys:
            setattr(args, key, getattr(args, key).resolve())
    expected_output_basename = OUTPUT_BASENAME_PREFIX + args.git_revision[:7]
    if args.output_root.name != expected_output_basename:
        parser.error(
            "v2 output root basename differs from its exact Git-bound namespace"
        )
    if args.phase != "orchestrate":
        if args.worker_index not in range(WORKER_COUNT) or args.staging_root is None:
            parser.error(
                "worker phases require --worker-index in [0,3] and --staging-root"
            )
        args.staging_root = args.staging_root.resolve()
        expected_staging = args.output_root.parent / f".{args.output_root.name}.incomplete"
        if args.staging_root != expected_staging or not args.staging_root.is_dir():
            parser.error("worker --staging-root differs from the derived v2 path")
    return args


def main() -> None:
    args = parse_args()
    if Path(__file__).resolve() != (
        args.repository_root / RUNNER_RELATIVE_PATH
    ).resolve():
        raise ValueError("running v2 source differs from the repository-bound runner")
    contract = load_execution_contract(
        repository_root=args.repository_root,
        execution_config_path=args.execution_config,
    )
    validate_runtime_cli_values(contract, _runtime_values(args))
    _V1._verify_checkout(args.repository_root, args.git_revision)
    if not args.container_id.startswith(socket.gethostname()):
        raise ValueError("--container-id does not match the running container hostname")
    _V1._verify_executable(
        args.ocr_python_executable,
        expected_version=args.ocr_python_version,
        label="OCR Python executable",
    )
    _V1._verify_executable(
        args.processor_python_executable,
        expected_version=args.processor_python_version,
        label="processor Python executable",
    )
    if args.phase == "orchestrate":
        _orchestrate(args, contract)
    elif args.phase == "ocr-worker":
        _run_ocr_worker(args, contract, args.staging_root)
    else:
        _run_processor_worker(args, contract, args.staging_root)


if __name__ == "__main__":
    main()
