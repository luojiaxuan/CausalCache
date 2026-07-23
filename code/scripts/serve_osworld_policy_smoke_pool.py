#!/usr/bin/env python3
"""Serve GPU-bound WAIT/DONE replicas for OSWorld transport validation."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from causalcache.osworld import OSWORLD_POLICY_RESPONSE_SCHEMA_VERSION
from causalcache.osworld_benchmark import smoke_policy_action


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _serve(replica_id: int, port: int, require_cuda: bool) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(replica_id)
    device_name = "cpu"
    if require_cuda:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("policy replica requires exactly one visible CUDA device")
        probe = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        device_name = torch.cuda.get_device_name(0)
        del probe

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                _json_response(self, 404, {"error": "not_found"})
                return
            _json_response(
                self,
                200,
                {
                    "status": "ready",
                    "replica_id": replica_id,
                    "device_name": device_name,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/act":
                _json_response(self, 404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                action = smoke_policy_action(request)
                _json_response(
                    self,
                    200,
                    {
                        "schema_version": OSWORLD_POLICY_RESPONSE_SCHEMA_VERSION,
                        "action": action,
                        "replica_id": replica_id,
                        "device_name": device_name,
                    },
                )
            except Exception as error:
                _json_response(
                    self,
                    400,
                    {"error_type": error.__class__.__name__, "error": str(error)},
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"replica={replica_id} {format % args}", flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(
        json.dumps(
            {
                "status": "READY_OSWORLD_POLICY_SMOKE_REPLICA",
                "replica_id": replica_id,
                "port": port,
                "device_name": device_name,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if len(set(args.ports)) != len(args.ports) or any(
        port <= 0 or port >= 65536 for port in args.ports
    ):
        raise ValueError("ports must be unique values within (0, 65536)")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_serve,
            args=(replica_id, port, args.require_cuda),
            name=f"osworld-policy-replica-{replica_id}",
        )
        for replica_id, port in enumerate(args.ports)
    ]
    for process in processes:
        process.start()

    def stop(signum: int, frame: Any) -> None:
        for process in processes:
            if process.is_alive():
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
