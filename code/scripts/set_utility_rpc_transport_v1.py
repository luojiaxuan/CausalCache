#!/usr/bin/env python3
"""Materialize or evaluate selector RPC transport v1 without executing it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from causalcache.set_utility_native_replay import canonical_json_bytes
from causalcache.set_utility_rpc_transport_v1 import (
    evaluate_rpc_transport_readiness,
    materialize_rpc_transport_plan,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_atomic(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    payload = canonical_json_bytes(value, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--h200-host", choices=("hyper00", "hyper01"), required=True)
    materialize.add_argument("--h200-service-port", type=int, required=True)
    materialize.add_argument("--mac-loopback-port", type=int, required=True)
    materialize.add_argument("--taurus-loopback-port", type=int, required=True)
    materialize.add_argument("--aries-loopback-port", type=int, required=True)
    materialize.add_argument("--rtt-p95-ms-maximum", type=float, required=True)
    materialize.add_argument("--rtt-sample-count-minimum", type=int, required=True)
    materialize.add_argument("--taurus-endpoint", default="taurus")
    materialize.add_argument("--aries-host-alias", default="aries.cs.ucsb.edu")
    materialize.add_argument("--aries-user", default="jiaxuanluo")
    materialize.add_argument("--aries-relay-port", type=int, default=20042)
    materialize.add_argument(
        "--container-network-profile", choices=("host_network",), default="host_network"
    )
    materialize.add_argument("--home", type=Path)
    materialize.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--plan", type=Path, required=True)
    evaluate.add_argument("--health-evidence", type=Path, required=True)
    evaluate.add_argument("--authorization-sha256", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "materialize":
        result = materialize_rpc_transport_plan(
            h200_host=args.h200_host,
            h200_service_port=args.h200_service_port,
            mac_loopback_port=args.mac_loopback_port,
            taurus_loopback_port=args.taurus_loopback_port,
            aries_loopback_port=args.aries_loopback_port,
            rtt_p95_ms_maximum=args.rtt_p95_ms_maximum,
            rtt_sample_count_minimum=args.rtt_sample_count_minimum,
            taurus_endpoint=args.taurus_endpoint,
            aries_host_alias=args.aries_host_alias,
            aries_user=args.aries_user,
            aries_relay_port=args.aries_relay_port,
            container_network_profile=args.container_network_profile,
            home=args.home,
        )
    else:
        plan = _read_json(args.plan)
        evidence = _read_json(args.health_evidence)
        samples = evidence.get("samples")
        if not isinstance(samples, list):
            raise ValueError("health evidence must contain a samples array")
        result = evaluate_rpc_transport_readiness(
            plan=plan,
            health_samples=samples,
            expected_authorization_sha256=args.authorization_sha256,
        )
    _write_atomic(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "verdict": result.get("verdict"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
