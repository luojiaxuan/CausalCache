#!/usr/bin/env python3
"""Run one preregistered single-GPU throughput-pilot worker."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_throughput_pilot_contract_v1 import (
    CANONICAL_CONFIG_PATH,
)
from causalcache.set_utility_throughput_pilot_execution_v1 import (
    ExecutionLaunchEnvelopeV1,
    run_throughput_pilot_worker_v1,
    validated_execution_from_authorized_projection_v1,
)


_SAFE_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-envelope", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, choices=range(4), required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    envelope_loader: Callable[..., Mapping[str, Any]] | None = None,
    worker_runner: Callable[..., Mapping[str, Any]] = run_throughput_pilot_worker_v1,
) -> Mapping[str, Any]:
    args = _parser().parse_args(argv)
    loader = envelope_loader
    if loader is None:
        from causalcache.set_utility_throughput_pilot_envelope_v1 import (
            load_set_utility_throughput_pilot_execution_envelope_v1,
        )

        loader = load_set_utility_throughput_pilot_execution_envelope_v1
    projection = loader(args.execution_envelope)
    root = Path(str(projection["repository_root"]))
    expected_config = root / CANONICAL_CONFIG_PATH
    worker_key = str(args.worker_index)
    if (
        not root.is_absolute()
        or projection.get("config_path") != str(expected_config)
        or projection.get("envelope_path") != str(args.execution_envelope)
        or projection.get("device_by_worker", {}).get(worker_key) != "cuda:0"
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        != projection.get("worker_gpu_uuid", {}).get(worker_key)
    ):
        raise ValueError("execution envelope projection or GPU isolation drifted")
    launch = ExecutionLaunchEnvelopeV1(
        repository_root=root,
        config_path=CANONICAL_CONFIG_PATH,
        processor_root=Path(str(projection["processor_root"])),
        model_dir=Path(str(projection["model_dir"])),
        output_root=Path(str(projection["run_root"])),
        worker_index=args.worker_index,
        device="cuda:0",
    )
    terminal = worker_runner(
        launch,
        envelope_validator=lambda current, contract: (
            validated_execution_from_authorized_projection_v1(
                current,
                contract,
                projection,
            )
        ),
    )
    print(json.dumps(terminal, allow_nan=False, sort_keys=True))
    return terminal


def cli() -> None:
    try:
        main()
    except BaseException as error:
        name = error.__class__.__name__
        failure = name if _SAFE_FAILURE.fullmatch(name) is not None else "UnexpectedException"
        print(
            json.dumps(
                {"failure_class": failure, "status": "FAILED_WORKER_CLI_V1"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
