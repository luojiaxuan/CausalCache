#!/usr/bin/env python3
"""Serve one shared frozen GUI-Owl policy to concurrent MobileWorld environments."""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from causalcache.mobileworld import (
    MOBILEWORLD_POLICY_RESPONSE_SCHEMA_VERSION,
    build_mobileworld_gui_owl_messages,
    mobileworld_action_from_gui_owl,
)
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime


def _history_image_count(messages: list[dict[str, Any]]) -> int:
    image_count = sum(
        item.get("type") == "image"
        for message in messages
        for item in message.get("content", ())
    )
    if image_count < 1:
        raise ValueError("MobileWorld policy prompt lacks its current image")
    return image_count - 1


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    payload: Any,
) -> None:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--visual-tokens", type=int, default=2560)
    parser.add_argument("--max-history-images", type=int)
    args = parser.parse_args()

    if not 1024 <= args.port <= 65535:
        raise ValueError("MobileWorld policy port must be within [1024, 65535]")
    if args.max_history_images is not None and args.max_history_images < 0:
        raise ValueError("max history images must be non-negative")
    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        target_effective_visual_tokens_per_image=args.visual_tokens,
    )
    inference_lock = threading.Lock()
    counters = {
        "requests": 0,
        "failures": 0,
        "audited_prompts": 0,
        "maximum_history_images": 0,
    }
    counter_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                _json_response(self, 404, {"error": "not_found"})
                return
            with counter_lock:
                observed = dict(counters)
            _json_response(
                self,
                200,
                {
                    "status": "ready",
                    "counters": observed,
                    "max_history_images": args.max_history_images,
                    "runtime": runtime.metadata,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/act":
                _json_response(self, 404, {"error": "not_found"})
                return
            arrived = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = build_mobileworld_gui_owl_messages(request)
                history_image_count = _history_image_count(messages)
                with counter_lock:
                    counters["audited_prompts"] += 1
                    counters["maximum_history_images"] = max(
                        counters["maximum_history_images"],
                        history_image_count,
                    )
                if (
                    args.max_history_images is not None
                    and history_image_count > args.max_history_images
                ):
                    raise RuntimeError(
                        "MobileWorld prompt exceeds the enforced history-image "
                        f"budget: observed={history_image_count}, "
                        f"maximum={args.max_history_images}"
                    )
                queued = time.perf_counter()
                with inference_lock:
                    queue_seconds = time.perf_counter() - queued
                    generated = runtime.generate_native_action(messages)
                screen_size = tuple(request["screen_size"])
                action = mobileworld_action_from_gui_owl(
                    generated.parsed_output.canonical_action,
                    screen_size=screen_size,
                )
                with counter_lock:
                    counters["requests"] += 1
                    request_count = counters["requests"]
                _json_response(
                    self,
                    200,
                    {
                        "schema_version": (
                            MOBILEWORLD_POLICY_RESPONSE_SCHEMA_VERSION
                        ),
                        "action": action,
                        "native_output": generated.output_text,
                        "request_count": request_count,
                        "request_decode_seconds": queued - arrived,
                        "queue_seconds": queue_seconds,
                        "generation": generated.metadata,
                    },
                )
            except Exception as error:  # noqa: BLE001
                with counter_lock:
                    counters["failures"] += 1
                print(
                    json.dumps(
                        {
                            "event": "MOBILEWORLD_POLICY_FAILURE",
                            "error_type": error.__class__.__name__,
                            "error": str(error),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                traceback.print_exc()
                _json_response(
                    self,
                    500,
                    {
                        "error_type": error.__class__.__name__,
                        "error": str(error),
                    },
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(format % args, flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(
        json.dumps(
            {
                "status": "READY_MOBILEWORLD_GUI_OWL_POLICY",
                "port": args.port,
                "max_history_images": args.max_history_images,
                "runtime": runtime.metadata,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
