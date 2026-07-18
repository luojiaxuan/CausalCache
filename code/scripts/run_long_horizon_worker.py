#!/usr/bin/env python3
"""Run/resume one Hyper00 shard or aggregate all four completed shards."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.long_horizon_contract import (
    CANONICAL_CONFIG_PATH,
    RUNNER_FREEZE_B_PATH,
    SELECTION_MANIFEST_FREEZE_PATH,
    LongHorizonContract,
)
from causalcache.long_horizon_data import (
    IMAGE_TAR_RELATIVE_PATH,
    validate_payload_file_identities,
)
from causalcache.long_horizon_execution import (
    CanonicalLongHorizonRuntimeAdapter,
    aggregate_worker_outputs,
    atomic_write,
    load_development_trajectories,
    load_json_object,
    pretty_json_bytes,
    run_worker_shard,
    sha256_bytes,
    sha256_file,
    validate_runner_config,
)
from scripts.manage_long_horizon_execution import validate_existing


class TarImageLoader:
    """Read one image member at a time without materializing the full tar in RAM."""

    def __init__(self, path: str | Path) -> None:
        self._archive = tarfile.open(path, mode="r:")

    def __call__(self, member_path: str) -> bytes:
        if (
            not isinstance(member_path, str)
            or PurePosixPath(member_path).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(member_path).parts)
        ):
            raise ValueError("image loader requires a canonical relative member path")
        try:
            member = self._archive.getmember(member_path)
        except KeyError as error:
            raise ValueError(f"missing substrate image member: {member_path}") from error
        if not member.isfile():
            raise ValueError("substrate image member must be a regular file")
        handle = self._archive.extractfile(member)
        if handle is None:
            raise ValueError("substrate image member could not be opened")
        with handle:
            return handle.read()

    def close(self) -> None:
        self._archive.close()


def decode_image(payload: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return image.convert("RGB").copy()


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path(CANONICAL_CONFIG_PATH))
    parser.add_argument(
        "--runner-config", type=Path, default=Path(RUNNER_FREEZE_B_PATH)
    )
    parser.add_argument("--expected-execution-b-commit", required=True)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path(SELECTION_MANIFEST_FREEZE_PATH),
    )
    parser.add_argument("--substrate-root", type=Path, required=True)
    parser.add_argument("--selector-preparation-manifest", type=Path, required=True)
    parser.add_argument("--selector-seal", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    worker = commands.add_parser("worker", parents=[common])
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--host-gpu-id", required=True)
    worker.add_argument("--container-cuda-ordinal", type=int, required=True)
    worker.add_argument("--model-dir", type=Path, required=True)
    worker.add_argument("--snapshot-manifest", type=Path, required=True)
    aggregate = commands.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--summary-output", type=Path, required=True)
    return parser


def _load_bound_inputs(args: argparse.Namespace) -> tuple[Any, ...]:
    root = args.repository_root.resolve()
    lifecycle = validate_existing(
        repository_root=root,
        contract_path=args.contract,
        expected_execution_b_commit=args.expected_execution_b_commit,
        substrate_manifest_path=(
            args.substrate_root
            / "derived/long-horizon-development-v1/manifest.json"
        ),
        selector_preparation_manifest_path=args.selector_preparation_manifest,
        selector_seal_path=args.selector_seal,
    )
    contract = LongHorizonContract.load(
        args.contract, repository_root=root, require_runner_absent=False
    )
    runner_payload, runner = load_json_object(
        root / args.runner_config, label="Execution-B runner config"
    )
    if runner_payload != pretty_json_bytes(runner):
        raise ValueError("Execution-B runner config is not canonical pretty JSON")
    selection_payload, selection = load_json_object(
        root / args.selection_manifest, label="selection freeze manifest"
    )
    substrate_manifest_path = (
        args.substrate_root
        / "derived/long-horizon-development-v1/manifest.json"
    )
    substrate_manifest = substrate_manifest_path.read_bytes()
    selector_payload = args.selector_seal.read_bytes()
    validated_runner = validate_runner_config(
        runner,
        contract=contract,
        selection_manifest_payload=selection_payload,
        substrate_manifest_payload=substrate_manifest,
        selector_preparation_manifest_payload=args.selector_preparation_manifest.read_bytes(),
        selector_seal_payload=selector_payload,
    )
    if lifecycle["runner_config_sha256"] != sha256_bytes(runner_payload):
        raise ValueError("lifecycle and loaded runner-config digests differ")
    selector_seal = json.loads(selector_payload)
    _, substrate_manifest_value = load_json_object(
        substrate_manifest_path, label="long-horizon substrate manifest"
    )
    validate_payload_file_identities(args.substrate_root, substrate_manifest_value)
    trajectories = load_development_trajectories(
        args.substrate_root, selection_manifest=selection
    )
    return (
        contract,
        validated_runner,
        sha256_bytes(runner_payload),
        selection,
        selector_seal,
        trajectories,
    )


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    (
        contract,
        runner,
        runner_sha256,
        selection,
        selector_seal,
        trajectories,
    ) = _load_bound_inputs(args)
    output = args.output_root.resolve()
    persistent = Path("/data").resolve()
    if output != persistent and persistent not in output.parents:
        raise ValueError("formal worker output root must remain under /data")
    expected_snapshot = (
        args.repository_root.resolve()
        / contract.data["policy_context_profile"]["snapshot_manifest_path"]
    ).resolve()
    if (
        args.snapshot_manifest.resolve() != expected_snapshot
        or sha256_file(expected_snapshot)
        != contract.data["policy_context_profile"]["snapshot_manifest_sha256"]
    ):
        raise ValueError("worker snapshot manifest differs from frozen Source-A")
    # note (luojiaxuan): Runtime construction happens only inside the explicitly
    # assigned worker process; the manager and aggregator never initialize CUDA.
    from causalcache.long_horizon_runtime import (
        DeterministicReceiptStore,
        GPUFullVocabularyKLBackend,
        build_long_horizon_runtime,
    )

    runtime = build_long_horizon_runtime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=f"cuda:{args.container_cuda_ordinal}",
    )
    image_loader = TarImageLoader(
        args.substrate_root.joinpath(
            *PurePosixPath(IMAGE_TAR_RELATIVE_PATH).parts
        )
    )
    try:
        receipt_store = DeterministicReceiptStore(
            args.output_root / f"worker-{args.worker_index}" / "runtime-receipts",
            run_identity_sha256=runner_sha256,
        )
        adapter = CanonicalLongHorizonRuntimeAdapter(
            contract=contract,
            runtime=runtime,
            distance_backend=GPUFullVocabularyKLBackend(torch_module=runtime.torch),
            image_bytes_loader=image_loader,
            image_decoder=decode_image,
            selection_seal_sha256=runner["selector_seal"]["sha256"],
            receipt_store=receipt_store,
        )
        return run_worker_shard(
            contract=contract,
            runner_config=runner,
            runner_config_sha256=runner_sha256,
            selection_manifest=selection,
            selector_seal=selector_seal,
            trajectories=trajectories,
            worker_index=args.worker_index,
            host_gpu_id=args.host_gpu_id,
            container_cuda_ordinal=args.container_cuda_ordinal,
            output_root=args.output_root,
            adapter=adapter,
        )
    finally:
        image_loader.close()


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    contract, runner, runner_sha256, selection, selector_seal, _ = _load_bound_inputs(args)
    summary = aggregate_worker_outputs(
        contract=contract,
        runner_config=runner,
        runner_config_sha256=runner_sha256,
        selection_manifest=selection,
        selector_seal=selector_seal,
        output_root=args.output_root,
    )
    expected_root = args.output_root.resolve()
    summary_path = args.summary_output.resolve()
    if summary_path != expected_root and expected_root not in summary_path.parents:
        raise ValueError("summary output must remain under the formal /data run root")
    atomic_write(summary_path, pretty_json_bytes(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = _run_worker(args) if args.command == "worker" else _aggregate(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
