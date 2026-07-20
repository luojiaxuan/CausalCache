#!/usr/bin/env python3
"""Serve the GO-authorized rich-token selector behind a loopback SSH tunnel."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from causalcache.set_utility_live_controller import (
    canonical_json_bytes,
    load_go_authorized_live_selector_service,
)


def _handler(service: Any, *, maximum_request_bytes: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CausalCacheLiveSelector/1.0"

        def _send(self, status: int, value: Mapping[str, Any]) -> None:
            payload = canonical_json_bytes(value) + b"\n"
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send(404, {"error": "not_found"})
                return
            self._send(200, service.health())

        def do_POST(self) -> None:
            if self.path != "/select":
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send(400, {"error": "invalid_content_length"})
                return
            if length <= 0 or length > maximum_request_bytes:
                self._send(413, {"error": "request_size_outside_contract"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request must be one JSON object")
                response = service.select(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                self._send(
                    400,
                    {
                        "error": error.__class__.__name__,
                        "message": str(error),
                    },
                )
                return
            except Exception as error:
                self._send(
                    500,
                    {
                        "error": error.__class__.__name__,
                        "message": "live selector failed without fallback",
                    },
                )
                return
            self._send(200, response)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--heldout-config", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--heldout-result", type=Path, required=True)
    parser.add_argument("--native-replay-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--maximum-request-bytes", type=int, default=256 * 1024 * 1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bind_host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("live selector must bind loopback and use an SSH tunnel")
    if not 1 <= args.port <= 65535 or args.maximum_request_bytes <= 0:
        raise ValueError("live selector server port or request limit is invalid")
    service = load_go_authorized_live_selector_service(
        repository_root=args.repository_root,
        heldout_config_path=args.heldout_config,
        selections_path=args.selections,
        heldout_result_path=args.heldout_result,
        native_replay_result_path=args.native_replay_result,
        checkpoint_path=args.checkpoint,
        model_dir=args.model_dir,
        device=args.device,
    )
    server = HTTPServer(
        (args.bind_host, args.port),
        _handler(service, maximum_request_bytes=args.maximum_request_bytes),
    )
    print(json.dumps(service.health(), sort_keys=True), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
