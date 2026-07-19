#!/usr/bin/env python3
"""Aggregate the exact four throughput-pilot worker terminals."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_throughput_pilot_contract_v1 import (
    CANONICAL_CONFIG_PATH,
)
from causalcache.set_utility_throughput_pilot_execution_v1 import (
    aggregate_throughput_pilot_workers_v1,
)


_SAFE_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-envelope", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    envelope_loader: Callable[..., Mapping[str, Any]] | None = None,
    aggregator: Callable[..., Mapping[str, Any]] = (
        aggregate_throughput_pilot_workers_v1
    ),
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
    if (
        not root.is_absolute()
        or projection.get("config_path") != str(root / CANONICAL_CONFIG_PATH)
        or projection.get("envelope_path") != str(args.execution_envelope)
    ):
        raise ValueError("execution envelope aggregate projection drifted")
    aggregate = aggregator(
        repository_root=root,
        output_root=Path(str(projection["run_root"])),
        config_path=CANONICAL_CONFIG_PATH,
    )
    print(json.dumps(aggregate, allow_nan=False, sort_keys=True))
    return aggregate


def cli() -> None:
    try:
        main()
    except BaseException as error:
        name = error.__class__.__name__
        failure = name if _SAFE_FAILURE.fullmatch(name) is not None else "UnexpectedException"
        print(
            json.dumps(
                {"failure_class": failure, "status": "FAILED_AGGREGATE_CLI_V1"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
