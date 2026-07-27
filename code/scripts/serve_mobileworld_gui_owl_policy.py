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
    parser.add_argument(
        "--selector-bundle", type=Path, default=None,
        help="marginal_scorer.pt;给定即启用 propose-then-select 两遍推理"
             "(要求 runner memory_arm=full 携带全池截图)",
    )
    parser.add_argument("--selector-arch", choices=("two_tower", "concat"),
                        default="two_tower")
    parser.add_argument("--selection-budget", type=int, default=4)
    parser.add_argument("--selector-beam", type=int, default=3)
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

    selector_meta: dict[str, Any] | None = None
    selector_model = None
    selector_norm = None
    if args.selector_bundle is not None:
        import torch

        bundle = torch.load(args.selector_bundle, map_location="cpu")
        s_dim = len(bundle["feature_names"]) + len(bundle["set_feature_names"])
        s_ro = int(bundle.get("readout_dim") or 0)
        s_hidden = int(bundle["hidden"])

        class _TwoTower(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cheap = torch.nn.Sequential(
                    torch.nn.Linear(s_dim, s_hidden), torch.nn.GELU(),
                    torch.nn.Linear(s_hidden, s_hidden), torch.nn.GELU(),
                    torch.nn.Linear(s_hidden, 1),
                )
                if s_ro:
                    self.readout = torch.nn.Sequential(
                        torch.nn.Linear(s_ro, 64), torch.nn.GELU(),
                        torch.nn.Dropout(0.0), torch.nn.Linear(64, 1),
                    )

            def forward(self, x, r=None, rm=None):
                score = self.cheap(x).squeeze(-1)
                # 部署侧不做在线读出:readout 塔以 mask=0 关断(two-tower 结构红利)
                return score

        class _Concat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = torch.nn.Dropout(0.0)
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(s_dim + s_ro, s_hidden), torch.nn.GELU(),
                    torch.nn.Linear(s_hidden, s_hidden), torch.nn.GELU(),
                    torch.nn.Linear(s_hidden, 1),
                )

            def forward(self, x, r=None, rm=None):
                if s_ro:
                    zeros = torch.zeros(x.shape[0], s_ro)
                    x = torch.cat([x, zeros], dim=-1)
                return self.net(x).squeeze(-1)

        selector_model = (_TwoTower() if args.selector_arch == "two_tower"
                          else _Concat())
        selector_model.load_state_dict(bundle["model_state"])
        selector_model.eval()
        s_mean, s_std = bundle["mean"], bundle["std"]
        selector_norm = lambda f: [(v - m) / s for v, m, s in zip(f, s_mean, s_std)]
        selector_meta = {
            "bundle": str(args.selector_bundle),
            "arch": args.selector_arch,
            "budget": args.selection_budget,
            "beam": args.selector_beam,
            "readout_online": False,
            "witness_pseudo_target": "pass1_proposal",
        }

    def run_selection(request, proposal_action):
        """全池特征 + beam 组合 exact-B;返回 (chosen_steps, diagnostics)。"""
        import torch

        from causalcache.mobileworld_selector_features import (
            dedup_pool,
            hash_screenshot,
            mobile_candidate_features,
            mobile_set_context_features,
            parse_history_actions,
        )
        import base64 as _b64

        history = request["history"]
        total = len(history)
        budget = args.selection_budget
        hashes = {}
        for event in history:
            encoded = event.get("restored_observation_screenshot_png_base64")
            if encoded:
                hashes[int(event["step_id"])] = hash_screenshot(
                    _b64.b64decode(encoded))
        pool, alias, counts = dedup_pool(hashes)
        if len(pool) < budget:
            return None, {"reason": "pool_smaller_than_budget", "pool": len(pool)}
        parsed = parse_history_actions(
            [event["full_response"] for event in history])
        prop = None
        if proposal_action is not None:
            prop = {
                "action": proposal_action.action,
                "coordinate": proposal_action.coordinate,
                "coordinate2": proposal_action.coordinate2,
                "text": proposal_action.text,
                "button": proposal_action.button,
            }

        def marginals(selected, candidates):
            xs = []
            for s in candidates:
                f = mobile_candidate_features(
                    step=s, pool=pool, total_steps=total,
                    parsed_actions=parsed, proposal=prop,
                    duplicate_counts=counts,
                ) + mobile_set_context_features(
                    step=s, selected=list(selected), total_steps=total,
                    parsed_actions=parsed, proposal=prop,
                    duplicate_alias=alias,
                )
                xs.append(selector_norm(f))
            with torch.no_grad():
                return selector_model(
                    torch.tensor(xs, dtype=torch.float32)).tolist()

        level = [((), 0.0)]
        for _ in range(budget):
            expanded, seen = [], set()
            for sel, acc in level:
                remaining = [s for s in pool if s not in sel]
                if not remaining:
                    continue
                margs = marginals(sel, remaining)
                for s, m in sorted(zip(remaining, margs), key=lambda t: -t[1])[: args.selector_beam]:
                    child = tuple(sorted((*sel, s)))
                    if child in seen:
                        continue
                    seen.add(child)
                    expanded.append((child, acc + m))
            if not expanded:
                break
            expanded.sort(key=lambda t: -t[1])
            level = expanded[: args.selector_beam]
        chosen = sorted(level[0][0])
        witness_count = sum(
            1 for s in pool
            if parsed[s - 1] is not None and prop is not None
        )
        return chosen, {"pool": len(pool), "witnesses_scanned": witness_count}

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
                    "selector": selector_meta,
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
                selection_info: dict[str, Any] | None = None
                if selector_model is not None:
                    event_ids = [
                        int(e["step_id"]) for e in request.get("history", [])
                    ]
                    tail = event_ids[-args.selection_budget:]
                    pass1 = dict(request)
                    pass1["selected_event_step_ids"] = list(tail)
                    messages = build_mobileworld_gui_owl_messages(pass1)
                else:
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
                    if selector_model is not None:
                        chosen, diag = run_selection(
                            request, generated.canonical_action
                        )
                        tail_ids = [
                            int(e["step_id"]) for e in request["history"]
                        ][-args.selection_budget:]
                        selection_info = {
                            "passes": 1,
                            "recent_tail": tail_ids,
                            "chosen": chosen,
                            "diagnostics": diag,
                        }
                        if chosen is not None and list(chosen) != list(tail_ids):
                            import base64 as _b64

                            from causalcache.mobileworld import _decode_png
                            from causalcache.mobileworld_gapfold import (
                                build_mobile_official_messages_gapfold,
                            )

                            history = request["history"]
                            step_images = {}
                            for event in history:
                                sid = int(event["step_id"])
                                if sid in chosen:
                                    step_images[sid] = _decode_png(
                                        event[
                                            "restored_observation_screenshot_png_base64"
                                        ]
                                    )
                            messages2 = build_mobile_official_messages_gapfold(
                                goal=request["task"]["instruction"],
                                action_texts=[
                                    e["action_text"] for e in history
                                ],
                                full_responses=[
                                    e["full_response"] for e in history
                                ],
                                shown_steps=chosen,
                                step_images=step_images,
                                current_image=_decode_png(
                                    request["current_screenshot_png_base64"]
                                ),
                            )
                            with adapter_scope(messages2, len(chosen)):
                                generated = runtime.generate(messages2)
                            selection_info["passes"] = 2
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
                        "selection": selection_info,
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
