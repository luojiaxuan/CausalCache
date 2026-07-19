#!/usr/bin/env python3
"""Run the fail-closed Execution-CF processor-only candidate freeze."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import io
import json
import os
import platform
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

# note (luojiaxuan): Keep workers self-contained without ambient PYTHONPATH input.
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.policy.gui_owl_v2_1 import GUI_OWL_V2_1_MOBILE_USE_TOOL
from causalcache.set_utility_processor_artifacts import (
    build_final_candidate_schedule,
    build_git_safe_processor_freeze_summary,
    build_prefix_safe_processor_artifact_record,
    build_processor_worker_schedule,
    inspect_processor_artifact_shard,
    iter_processor_query_artifact_records,
    materialize_final_candidate_schedule,
    materialize_processor_artifact_shard,
)
from causalcache.set_utility_processor_freeze import (
    TARGET_PIXELS_PER_IMAGE,
    FrozenQueryCandidateRecord,
    assert_processor_only_import_state,
    canonical_json_bytes,
    freeze_query_candidates,
    sha256_bytes,
    sha256_file,
    validate_freeze_query_topology,
    validate_processor_only_source,
    verify_processor_only_snapshot,
    write_once_or_verify,
)
from causalcache.set_utility_processor_freeze_contract import (
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_PATH_ARGUMENTS,
    REQUIRED_VERSION_ARGUMENTS_BY_PHASE,
    load_execution_contract,
    validate_runtime_cli_values,
)
from causalcache.set_utility_full_pool import build_full_pool_inspection_config
from causalcache.set_utility_processor_prompt import (
    build_set_utility_gui_owl_v2_1_messages,
)
from causalcache.set_utility_processor_substrate import (
    SelectedRowReadPlan,
    SourceProvenance,
    build_selected_pilot,
    build_selected_row_read_plan,
    build_trajectory_processor_substrate,
    build_validated_ocr_batch,
    load_selected_rows_once,
)


FULL_POOL_INVENTORY = "data/manifests/set_utility_full_pool_inventory_v1.json"
FULL_POOL_CENSUS = "data/manifests/set_utility_full_pool_census_v2.json"
BASE_CONFIG = "code/configs/independent_reference_gate_v1.json"
OCR_CONFIG = "code/configs/restoration_v2_ocr_backend.json"
RUNNER_RELATIVE_PATH = "code/scripts/run_set_utility_processor_freeze.py"


def _load_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value in {path}: {raw}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


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


def _verify_executable(path: Path, *, expected_version: str, label: str) -> None:
    if (
        not path.is_absolute()
        or not path.resolve().is_file()
        or not os.access(path, os.X_OK)
    ):
        raise ValueError(f"{label} must be an explicit absolute executable")
    observed = subprocess.check_output(
        [str(path), "-c", "import platform; print(platform.python_version())"],
        text=True,
    ).strip()
    if observed != expected_version:
        raise ValueError(f"{label} version differs from its explicit CLI value")


def _verify_checkout(root: Path, revision: str) -> None:
    observed = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != revision:
        raise ValueError("formal runner Git revision differs from checkout HEAD")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    if status:
        raise ValueError("formal processor freeze requires a clean Git checkout")


def _verify_current_versions(args: argparse.Namespace, *, phase: str) -> None:
    if platform.python_version() != getattr(args, f"{phase}_python_version"):
        raise ValueError(f"{phase} Python version drifted")
    if phase == "ocr":
        expected = {
            "rapidocr": args.ocr_runtime_version,
            "pyarrow": args.pyarrow_version,
        }
    else:
        expected = {
            "transformers": args.transformers_version,
            "torch": args.torch_version,
            "Pillow": args.pillow_version,
        }
    observed = {name: importlib.metadata.version(name) for name in expected}
    if observed != expected:
        raise ValueError(f"{phase} package versions drifted: {observed!r}")


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
        if not path.is_file() or path.is_symlink() or path.stat().st_size != spec.size_bytes:
            raise ValueError(f"pinned source shard drifted: {spec.transport_file}")
        if sha256_file(path) != spec.sha256:
            raise ValueError(f"pinned source shard SHA256 drifted: {spec.transport_file}")
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=8):
            for row in batch.to_pylist():
                if not isinstance(row, Mapping):
                    raise TypeError("GUIOdyssey parquet row must be a mapping")
                yield row_index, row
                row_index += 1
        if row_index != parquet.metadata.num_rows:
            raise ValueError("source shard row count drifted during streaming")


def _load_source_inputs(contract: Any) -> tuple[Any, Any, Any, Any, Any]:
    root = contract.repository_root
    freeze = contract.freeze_b_v2_manifest
    census_path = root / FULL_POOL_CENSUS
    inventory_path = root / FULL_POOL_INVENTORY
    base_path = root / BASE_CONFIG
    census = _load_json(census_path)
    inventory = _load_json(inventory_path)
    base = _load_json(base_path)
    source_binding = freeze["source"]
    if (
        source_binding["p0_census_manifest_path"] != FULL_POOL_CENSUS
        or source_binding["p0_census_manifest_sha256"] != sha256_file(census_path)
        or census["source"]["source_manifest_sha256"] != sha256_file(inventory_path)
        or census["source"]["base_config_sha256"] != sha256_file(base_path)
    ):
        raise ValueError("Freeze-B to census/inventory/base source hash chain drifted")
    files = tuple(
        SourceFileSpec(record["path"], record["size_bytes"], record["lfs_sha256"])
        for record in inventory["inventory"]["files"]
    )
    inspection = build_full_pool_inspection_config(
        base,
        transport_files=tuple(file.transport_file for file in files),
        format_safety_maximum_decisions=None,
    )
    assignments, queries = validate_freeze_query_topology(
        freeze["assignments"],
        freeze["query_states"],
        anchor_step_by_stratum=freeze["repair"]["stratum_anchor_decision_step"],
    )
    plan = build_selected_row_read_plan(
        freeze["assignments"],
        source_files=files,
        source_row_counts=census["source"]["row_counts_by_file"],
    )
    schedule = build_processor_worker_schedule(plan)
    return inspection, assignments, queries, plan, schedule


def _worker_read_plan(plan: SelectedRowReadPlan, transport_files: tuple[str, ...]) -> SelectedRowReadPlan:
    allowed = set(transport_files)
    shards = tuple(shard for shard in plan.shards if shard.source_file.transport_file in allowed)
    if {shard.source_file.transport_file for shard in shards} != allowed:
        raise ValueError("worker transport-file schedule differs from selected-row plan")
    return SelectedRowReadPlan(
        shards=shards,
        assignment_count=sum(len(shard.assignments) for shard in shards),
    )


def _ocr_engine(args: argparse.Namespace, root: Path) -> tuple[Any, Any, str, dict[str, Any], Any]:
    from causalcache.restoration_v2_text_backend import (
        create_rapidocr_engine,
        load_backend_config,
        run_rapidocr_record,
        verify_model_files,
        verify_rapidocr_package_files,
        verify_recognizer_character_inventory,
        verify_runtime_packages,
        verify_wheel_files,
    )

    config_path = root / OCR_CONFIG
    config = load_backend_config(config_path)
    config_sha = sha256_file(config_path)
    identity = {
        "runtime_packages": verify_runtime_packages(config),
        "rapidocr_package_file_sha256": verify_rapidocr_package_files(config),
        "wheel_sha256": verify_wheel_files(config, args.ocr_wheel_dir),
        "model_sha256": verify_model_files(config, args.ocr_model_dir),
    }
    engine = create_rapidocr_engine(config, args.ocr_model_dir)
    identity["recognizer_character_inventory"] = verify_recognizer_character_inventory(
        engine, config
    )
    return engine, config, config_sha, identity, run_rapidocr_record


def _run_ocr_worker(args: argparse.Namespace, contract: Any, staging: Path) -> None:
    _verify_current_versions(args, phase="ocr")
    base, _, queries, plan, schedule = _load_source_inputs(contract)
    worker = schedule.workers[args.worker_index]
    read_plan = _worker_read_plan(plan, worker.transport_files)
    artifact_path = staging / "substrate" / worker.filename
    receipt_path = staging / "receipts" / f"ocr-worker-{args.worker_index:02d}.json"
    if artifact_path.exists() and receipt_path.exists():
        descriptor = inspect_processor_artifact_shard(
            artifact_path, expected_worker=worker
        )
        receipt = _load_json(receipt_path)
        if receipt.get("shard") != descriptor.to_payload():
            raise ValueError("resumable OCR receipt differs from its artifact shard")
        return
    if artifact_path.exists() != receipt_path.exists():
        raise ValueError("incomplete OCR artifact/receipt pair cannot be resumed safely")
    engine, backend, backend_sha, identity, record_runner = _ocr_engine(
        args, contract.repository_root
    )
    from causalcache.restoration_v2_text_backend import (
        prepare_selected_guiodyssey_image,
    )

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
    row_factory = ParquetRowFactory(args.source_root)
    worker_format_counts: Counter[tuple[str, str]] = Counter()

    def records() -> Iterator[Any]:
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
                loaded_rows, key=lambda item: item.assignment.transport_row_index
            ):
                pilot = build_selected_pilot(loaded, provenance=provenance)
                required = {
                    str(event[field])
                    for event in pilot.manifest["trajectory"]["events"]
                    for field in ("observation_before_path", "observation_after_path")
                }
                format_counts: Counter[tuple[str, str]] = Counter()
                for path in sorted(required):
                    prepared = prepare_selected_guiodyssey_image(
                        pilot.image_payloads[path], backend
                    )
                    format_counts[(prepared.source_format, prepared.source_mode)] += 1
                worker_format_counts.update(format_counts)
                if sum(format_counts.values()) != len(required):
                    raise RuntimeError("pre-OCR image-format tally drifted")
                captured_ocr: dict[str, Mapping[str, Any]] = {}

                def batch_builder(images: Mapping[str, bytes]) -> Any:
                    batch = build_validated_ocr_batch(
                        images,
                        engine=engine,
                        backend_config=backend,
                        backend_config_sha256=backend_sha,
                        record_runner=record_runner,
                    )
                    captured_ocr.update(batch.records_by_path)
                    return batch

                substrate = build_trajectory_processor_substrate(
                    pilot,
                    query_states=queries[loaded.assignment.source_id],
                    ocr_batch_builder=batch_builder,
                )
                yield build_prefix_safe_processor_artifact_record(
                    substrate,
                    image_payloads=pilot.image_payloads,
                    ocr_records_by_path=captured_ocr,
                )

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
    write_once_or_verify(
        staging / "receipts" / f"ocr-worker-{args.worker_index:02d}.json",
        canonical_json_bytes(receipt),
    )


class ProcessorLengthRuntime:
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
        from PIL import Image

        opened: list[Any] = []

        def decode(payload: bytes) -> Any:
            with Image.open(io.BytesIO(payload)) as source:
                source.load()
                image = source.convert("RGB")
            opened.append(image)
            return image

        try:
            messages = build_set_utility_gui_owl_v2_1_messages(
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


def _run_processor_worker(args: argparse.Namespace, contract: Any, staging: Path) -> None:
    _verify_current_versions(args, phase="processor")
    _, _, _, _, schedule = _load_source_inputs(contract)
    worker = schedule.workers[args.worker_index]
    shard = staging / "substrate" / worker.filename
    runtime = ProcessorLengthRuntime(args.model_dir)
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


def _subprocess_args(args: argparse.Namespace, *, phase: str, worker: int, staging: Path) -> list[str]:
    result = [str(args.repository_root / RUNNER_RELATIVE_PATH)]
    for name in (
        "repository_root",
        "execution_config",
        "source_root",
        "model_dir",
        "snapshot_manifest",
        "ocr_model_dir",
        "ocr_wheel_dir",
        "output_root",
        "ocr_python_executable",
        "processor_python_executable",
        "git_revision",
        "host_alias",
        "host_hostname",
        "container_id",
        "container_image_digest",
        "worker_count",
        "ocr_python_version",
        "ocr_runtime_version",
        "pyarrow_version",
        "processor_python_version",
        "transformers_version",
        "torch_version",
        "pillow_version",
    ):
        result.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
    result.extend(
        ["--phase", phase, "--worker-index", str(worker), "--staging-root", str(staging)]
    )
    return result


def _launch_workers(args: argparse.Namespace, *, phase: str, staging: Path) -> None:
    executable = (
        args.ocr_python_executable
        if phase == "ocr-worker"
        else args.processor_python_executable
    )
    processes = []
    for worker in range(4):
        log_path = staging / "logs" / f"{phase}-{worker:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("ab")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        process = subprocess.Popen(
            [str(executable), *_subprocess_args(args, phase=phase, worker=worker, staging=staging)],
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


def _read_candidate_parts(staging: Path) -> tuple[FrozenQueryCandidateRecord, ...]:
    records = []
    for worker in range(4):
        path = staging / "candidate-parts" / f"worker-{worker:02d}.jsonl"
        for line in path.read_bytes().splitlines():
            value = json.loads(line.decode("utf-8"))
            records.append(FrozenQueryCandidateRecord.from_payload(value))
    ordered = tuple(sorted(records, key=lambda item: item.state_id))
    if len(ordered) != 2400 or len({item.state_id for item in ordered}) != 2400:
        raise ValueError("candidate part denominator differs from frozen 2,400 states")
    if Counter(item.role for item in ordered) != {
        "train": 2000,
        "tune": 200,
        "evaluation": 200,
    }:
        raise ValueError("candidate part role counts drifted")
    return ordered


def _orchestrate(args: argparse.Namespace, contract: Any) -> None:
    if Path(sys.executable).resolve() != args.processor_python_executable.resolve():
        raise ValueError("orchestrator must run under --processor-python-executable")
    _verify_current_versions(args, phase="processor")
    _, _, _, _, schedule = _load_source_inputs(contract)
    source_audit = validate_processor_only_source(
        args.repository_root / RUNNER_RELATIVE_PATH
    )
    snapshot = verify_processor_only_snapshot(
        model_dir=args.model_dir,
        snapshot_manifest=args.snapshot_manifest,
    )
    staging = args.output_root.parent / f".{args.output_root.name}.incomplete"
    if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
        raise ValueError("resumable staging root must be one real directory")
    staging.mkdir(parents=False, exist_ok=True)
    run_identity = {
        "config_sha256": contract.config_sha256,
        "git_revision": args.git_revision,
        "runtime_cli": {key: str(value) for key, value in sorted(_runtime_values(args).items())},
        "source_audit": source_audit,
        "snapshot": snapshot,
        "worker_execution_sha256": schedule.execution_sha256,
    }
    write_once_or_verify(staging / "run-identity.json", canonical_json_bytes(run_identity))
    _launch_workers(args, phase="ocr-worker", staging=staging)
    _launch_workers(args, phase="processor-worker", staging=staging)
    records = _read_candidate_parts(staging)
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
        "status": "PROCESSOR_ONLY_CANDIDATE_FREEZE_COMPLETED",
    }
    write_once_or_verify(staging / "manifest.json", canonical_json_bytes(manifest, pretty=True))
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("processor-freeze output appeared before atomic publication")
    os.rename(staging, args.output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in REQUIRED_PATH_ARGUMENTS:
        parser.add_argument(name, type=Path, required=True)
    for name in REQUIRED_IDENTITY_ARGUMENTS:
        parser.add_argument(name, required=True, type=int if name == "--worker-count" else str)
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
    if args.phase != "orchestrate":
        if args.worker_index not in range(4) or args.staging_root is None:
            parser.error("worker phases require --worker-index in [0,3] and --staging-root")
        args.staging_root = args.staging_root.resolve()
        expected_staging = args.output_root.parent / f".{args.output_root.name}.incomplete"
        if args.staging_root != expected_staging or not args.staging_root.is_dir():
            parser.error("worker --staging-root differs from the derived resumable path")
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
        raise ValueError("--container-id does not match the running container hostname")
    _verify_executable(
        args.ocr_python_executable,
        expected_version=args.ocr_python_version,
        label="OCR Python executable",
    )
    _verify_executable(
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
