#!/usr/bin/env python3
"""Serve one shared frozen GUI-Owl policy to concurrent MobileWorld environments.

# note (luojiaxuan): 可选 --adapter-checkpoint 挂 CausalCache HGKV(desktop v4
# 训练, 零样本上 mobile):注入后 lora_b 零初始化外的权重来自 checkpoint,
# 每个请求按官方 mobile 滚动 prompt 的图序契约(历史图 1..K 在前、当前图最后)
# 预编码一次算出历史 image-token mask,generate 的 prefill 在
# history_adapter_scope 里进行——decode 增量步由 HGKV hook 的长度检查自然
# bypass(历史 K/V 已在 cache 中被门控)。K=0(B0)与未挂 adapter 时无 scope,
# 输出与冻结服务 bitwise 相同。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from causalcache.policy.gui_owl_official_runtime import GUIOwlOfficialRuntime


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
    parser.add_argument("--adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--adapter-checkpoint-sha256", default=None)
    parser.add_argument("--adapter-layer-count", type=int, default=8)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-alpha", type=int, default=16)
    args = parser.parse_args()

    if not 1024 <= args.port <= 65535:
        raise ValueError("MobileWorld policy port must be within [1024, 65535]")
    if args.max_history_images is not None and args.max_history_images < 0:
        raise ValueError("max history images must be non-negative")
    base_runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        target_effective_visual_tokens_per_image=args.visual_tokens,
    )
    runtime = GUIOwlOfficialRuntime(base_runtime)

    adapter_meta: dict[str, Any] | None = None
    if args.adapter_checkpoint is not None:
        import torch

        from causalcache.policy.history_gated_lora import (
            inject_history_gated_kv,
            load_history_gated_state_dict,
        )

        if not args.adapter_checkpoint_sha256:
            raise ValueError("adapter checkpoint requires --adapter-checkpoint-sha256")
        digest = hashlib.sha256(args.adapter_checkpoint.read_bytes()).hexdigest()
        if digest != args.adapter_checkpoint_sha256:
            raise ValueError(
                f"adapter checkpoint SHA drifted: {digest} != "
                f"{args.adapter_checkpoint_sha256}"
            )
        wrapped = inject_history_gated_kv(
            base_runtime.model,
            layer_count=args.adapter_layer_count,
            rank=args.adapter_rank,
            alpha=args.adapter_alpha,
        )
        load_history_gated_state_dict(
            wrapped, torch.load(args.adapter_checkpoint, map_location="cpu")
        )
        for lora in wrapped.values():
            lora.lora_a.requires_grad_(False)
            lora.lora_b.requires_grad_(False)
        adapter_meta = {
            "type": "history_gated_kv",
            "checkpoint": str(args.adapter_checkpoint),
            "checkpoint_sha256": digest,
            "layer_count": args.adapter_layer_count,
            "rank": args.adapter_rank,
            "alpha": args.adapter_alpha,
        }

    merge_size = int(base_runtime.processor.image_processor.merge_size)

    def adapter_scope(messages: list[dict[str, Any]], history_image_count: int):
        """HGKV mask scope for one request; nullcontext when adapter off / K=0."""
        if adapter_meta is None or history_image_count < 1:
            return contextlib.nullcontext()
        from causalcache.policy.history_adapter_context import (
            HistoryAdapterContext,
            history_adapter_scope,
        )
        from causalcache.policy.history_token_roles import (
            build_history_token_mask,
        )

        encoded = base_runtime.processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=False,
        )
        mask = build_history_token_mask(
            encoded["input_ids"],
            encoded["mm_token_type_ids"],
            encoded["image_grid_thw"],
            history_image_count,
            merge_size,
        )
        context = HistoryAdapterContext(
            history_token_mask=mask.to(base_runtime.device),
            history_present=bool(mask.any()),
            image_roles=("restored_history",) * history_image_count + ("current",),
        )
        return history_adapter_scope(context)

    inference_lock = threading.Lock()
    counters = {
        "requests": 0,
        "failures": 0,
        "parse_failures": 0,
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
                    "adapter": adapter_meta,
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
                    with adapter_scope(messages, history_image_count):
                        generated = runtime.generate(messages)
                screen_size = tuple(request["screen_size"])
                if generated.canonical_action is None:
                    action = {"action_type": "wait"}
                    with counter_lock:
                        counters["parse_failures"] += 1
                else:
                    action = mobileworld_action_from_gui_owl(
                        generated.canonical_action,
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
                        "action_text": generated.action_text,
                        "full_response": generated.full_response,
                        "policy_parsed": generated.policy_parsed,
                        "parse_error": generated.parse_error,
                        "dropped_arguments": dict(generated.dropped_arguments),
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
                "adapter": adapter_meta,
                "runtime": runtime.metadata,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
