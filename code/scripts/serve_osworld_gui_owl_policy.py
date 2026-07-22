#!/usr/bin/env python3
"""Serve one frozen GUI-Owl OSWorld policy replica per visible GPU."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from causalcache.osworld import OSWORLD_POLICY_RESPONSE_SCHEMA_VERSION


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _serve(
    replica_id: int,
    gpu_id: int,
    port: int,
    model_dir: str,
    snapshot_manifest: str,
    visual_tokens: int,
    max_new_tokens: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from causalcache.osworld_gui_owl import GUIOwlOSWorldRuntime

    runtime = GUIOwlOSWorldRuntime(
        model_dir=model_dir,
        expected_snapshot_manifest=snapshot_manifest,
        device="cuda:0",
        effective_visual_tokens_per_image=visual_tokens,
        max_new_tokens=max_new_tokens,
    )
    inference_lock = threading.Lock()
    request_count = 0
    count_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                _json_response(self, 404, {"error": "not_found"})
                return
            with count_lock:
                observed_requests = request_count
            _json_response(
                self,
                200,
                {
                    "status": "ready",
                    "replica_id": replica_id,
                    "gpu_id": gpu_id,
                    "request_count": observed_requests,
                    "runtime": runtime.metadata,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            nonlocal request_count
            if self.path != "/act":
                _json_response(self, 404, {"error": "not_found"})
                return
            arrived = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                with inference_lock:
                    queue_seconds = time.perf_counter() - arrived
                    action, metadata = runtime.generate(request)
                with count_lock:
                    request_count += 1
                    current_count = request_count
                _json_response(
                    self,
                    200,
                    {
                        "schema_version": OSWORLD_POLICY_RESPONSE_SCHEMA_VERSION,
                        "action": action.to_mapping(),
                        "source": "frozen_gui_owl_1_5",
                        "replica_id": replica_id,
                        "gpu_id": gpu_id,
                        "replica_request_count": current_count,
                        "queue_seconds": queue_seconds,
                        "runtime": metadata,
                    },
                )
            except Exception as error:
                _json_response(
                    self,
                    500,
                    {
                        "error_type": error.__class__.__name__,
                        "error": str(error),
                        "replica_id": replica_id,
                    },
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"replica={replica_id} {format % args}", flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(
        json.dumps(
            {
                "status": "READY_OSWORLD_GUI_OWL_REPLICA",
                "replica_id": replica_id,
                "gpu_id": gpu_id,
                "port": port,
                "runtime": runtime.metadata,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+")
    parser.add_argument("--visual-tokens", type=int, default=480)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main() -> None:
    args = _parser().parse_args()
    gpu_ids = args.gpu_ids or list(range(len(args.ports)))
    if len(gpu_ids) != len(args.ports):
        raise ValueError("gpu id count must equal port count")
    if len(set(gpu_ids)) != len(gpu_ids) or len(set(args.ports)) != len(args.ports):
        raise ValueError("gpu ids and ports must be unique")
    if any(port <= 0 or port >= 65536 for port in args.ports):
        raise ValueError("ports must be within (0, 65536)")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_serve,
            args=(
                replica_id,
                gpu_id,
                port,
                str(args.model_dir.resolve()),
                str(args.snapshot_manifest.resolve()),
                args.visual_tokens,
                args.max_new_tokens,
            ),
            name=f"osworld-gui-owl-{replica_id}",
        )
        for replica_id, (gpu_id, port) in enumerate(zip(gpu_ids, args.ports, strict=True))
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
    if any(process.exitcode != 0 for process in processes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
