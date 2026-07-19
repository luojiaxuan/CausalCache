#!/usr/bin/env python3
"""Run one fresh-envelope-authorized D1 profile/worker process."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_action_stability_execution_v1 import (
    PROFILE_ORDER,
    launch_from_fresh_projection_v1,
    run_action_stability_worker_v1,
)


_SAFE_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-envelope", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILE_ORDER, required=True)
    parser.add_argument("--worker-index", type=int, choices=range(4), required=True)
    return parser


def _default_envelope_loader(path: str | Path) -> Mapping[str, Any]:
    from causalcache.set_utility_action_stability_envelope_v1 import (
        load_set_utility_action_stability_envelope_v1,
    )

    return load_set_utility_action_stability_envelope_v1(path)


def main(
    argv: Sequence[str] | None = None,
    *,
    envelope_loader: Callable[[str | Path], Mapping[str, Any]] | None = None,
    worker_runner: Callable[..., Mapping[str, Any]] = (
        run_action_stability_worker_v1
    ),
) -> Mapping[str, Any]:
    args = _parser().parse_args(argv)
    loader = envelope_loader or _default_envelope_loader
    projection = loader(args.execution_envelope)
    launch = launch_from_fresh_projection_v1(
        projection,
        execution_envelope_path=args.execution_envelope,
        profile=args.profile,
        worker_index=args.worker_index,
        visible_gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    terminal = worker_runner(launch)
    print(json.dumps(terminal, allow_nan=False, sort_keys=True))
    return terminal


def cli() -> None:
    try:
        main()
    except Exception as error:
        name = error.__class__.__name__
        failure = name if _SAFE_FAILURE.fullmatch(name) is not None else "UnexpectedException"
        print(
            json.dumps(
                {
                    "failure_class": failure,
                    "status": "FAILED_ACTION_STABILITY_WORKER_CLI_V1",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
