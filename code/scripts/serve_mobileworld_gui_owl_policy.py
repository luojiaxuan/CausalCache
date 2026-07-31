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
import queue
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


def _ocr_image_text(image: Any) -> str:
    # note (luojiaxuan): OCR 消融臂用 tesseract;失败时显式标注而非静默空串。
    try:
        import pytesseract
        text = pytesseract.image_to_string(image)
        text = " ".join(text.split())[:2000]
        return text if text else "(no text detected)"
    except Exception as exc:  # pragma: no cover - 环境缺依赖时显式暴露
        return f"(ocr unavailable: {exc})"


def _apply_history_render(
    messages: list[dict[str, Any]], mode: str
) -> list[dict[str, Any]]:
    """text_only: 删除所有历史图(保留 verbatim 响应);ocr: 历史图→OCR 文本。

    # note (luojiaxuan): 最后一个 image 恒为 current 截图,必须保留;其余
    # image 均属恢复的历史保留轮。变换在 pass-1/pass-2 生成前统一应用。
    """
    if mode == "images":
        return messages
    refs = [
        (mi, ci)
        for mi, message in enumerate(messages)
        for ci, item in enumerate(message.get("content", ()))
        if item.get("type") == "image"
    ]
    if len(refs) <= 1:
        return messages
    out = [dict(m, content=list(m.get("content", ()))) for m in messages]
    for mi, ci in reversed(refs[:-1]):
        if mode == "ocr":
            text = _ocr_image_text(out[mi]["content"][ci].get("image"))
            out[mi]["content"][ci] = {
                "type": "text",
                "text": "[restored screenshot, OCR text] " + text,
            }
        else:
            del out[mi]["content"][ci]
            if not out[mi]["content"]:
                out[mi]["content"].append(
                    {"type": "text", "text": "[screenshot omitted]"}
                )
    return out


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
    parser.add_argument(
        "--model-profile", choices=("8b", "32b"), default="8b",
        help="骨干档案:32b 切换身份钉与视觉输出维度(跨骨干迁移测试)。")
    parser.add_argument(
        "--history-render", choices=("images", "text_only", "ocr"),
        default="images",
        help="消融渲染:text_only 只保留保留轮的 verbatim 响应文本(去历史图);"
             "ocr 用历史截图的 OCR 文本替换图像;current 屏幕恒保留。")
    parser.add_argument("--adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--adapter-checkpoint-sha256", default=None)
    parser.add_argument("--adapter-layer-count", type=int, default=8)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-alpha", type=int, default=16)
    # note (luojiaxuan): 无门控普通 LoRA(训练侧 adapter_type=full_policy_lora)在
    # 部署时没有 history scope 概念,注入路径与 HGKV 不同,必须显式声明,免得把一个
    # 无门控 checkpoint 当成 HGKV 加载(键名对不上会直接报错,但语义混淆更危险)。
    parser.add_argument(
        "--adapter-type", choices=("history_gated_kv", "full_policy_lora"),
        default="history_gated_kv")
    parser.add_argument(
        "--adapter-target-modules", default="q_proj,k_proj,v_proj",
        help="仅 full_policy_lora 使用;逗号分隔,须与训练配置一致")
    # note (luojiaxuan): 微批。实测瓶颈是 inference_lock 的串行化(queue 11.7s vs
    # 前向 2.0s),batch-1 解码是显存带宽瓶颈,批量解码墙钟几乎不变。默认 1 =
    # 逐条路径逐字节不变;>1 需先做动作级一致性验证再用于正式实验。
    parser.add_argument("--batch-max", type=int, default=1)
    parser.add_argument("--batch-wait-ms", type=int, default=25)
    parser.add_argument(
        "--selector-bundle", type=Path, default=None,
        help="marginal_scorer.pt;给定即启用 propose-then-select 两遍推理"
             "(要求 runner memory_arm=full 携带全池截图)",
    )
    parser.add_argument("--selector-arch", choices=("two_tower", "concat"),
                        default="two_tower")
    parser.add_argument("--selection-budget", type=int, default=4)
    parser.add_argument("--selector-beam", type=int, default=3)
    parser.add_argument(
        "--max-replacements", type=int, default=None,
        help="k 上限:beam 组合中非-recent 促升数不得超过该值(k≤1 消融)。")
    # note (luojiaxuan): witness 伪目标来源。proposal = 两遍 propose-then-select
    # (先按 recent 尾出一遍动作再选);last_action = 单遍——witness 对比上一步
    # 已执行动作,选完记忆只生成一次(部署主线)。
    parser.add_argument("--selector-witness", choices=("proposal", "last_action"),
                        default="proposal")
    args = parser.parse_args()
    from causalcache.policy import gui_owl_v2_vision as _vision_profile
    _vision_profile.activate_model_profile(args.model_profile)

    if not 1024 <= args.port <= 65535:
        raise ValueError("MobileWorld policy port must be within [1024, 65535]")
    if args.max_history_images is not None and args.max_history_images < 0:
        raise ValueError("max history images must be non-negative")
    # note (luojiaxuan): torch 2.11 的 cuDNN SDPA 在高并发批形状下报
    # mha_graph.execute is_good()==false(HTTP 500);env 旗标不生效,代码级关闭,
    # 统一回退 flash/mem-efficient 内核。
    import torch as _torch

    _torch.backends.cuda.enable_cudnn_sdp(False)
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
        if args.adapter_type == "full_policy_lora":
            from scripts.train_success_sft_lora import (
                inject_lora,
                load_lora_state_dict,
            )

            targets = tuple(
                m.strip() for m in args.adapter_target_modules.split(",") if m.strip()
            )
            wrapped = inject_lora(
                base_runtime.model,
                rank=args.adapter_rank,
                alpha=args.adapter_alpha,
                target_modules=targets,
                torch=torch,
            )
            load_lora_state_dict(
                wrapped, torch.load(args.adapter_checkpoint, map_location="cpu")
            )
            adapter_meta = {
                "type": "full_policy_lora",
                "checkpoint": str(args.adapter_checkpoint),
                "checkpoint_sha256": digest,
                "target_modules": list(targets),
                "rank": args.adapter_rank,
                "alpha": args.adapter_alpha,
            }
        else:
            wrapped = inject_history_gated_kv(
                base_runtime.model,
                layer_count=args.adapter_layer_count,
                rank=args.adapter_rank,
                alpha=args.adapter_alpha,
            )
            load_history_gated_state_dict(
                wrapped, torch.load(args.adapter_checkpoint, map_location="cpu")
            )
            adapter_meta = {
                "type": "history_gated_kv",
                "checkpoint": str(args.adapter_checkpoint),
                "checkpoint_sha256": digest,
                "layer_count": args.adapter_layer_count,
                "rank": args.adapter_rank,
                "alpha": args.adapter_alpha,
            }
        for lora in wrapped.values():
            lora.lora_a.requires_grad_(False)
            lora.lora_b.requires_grad_(False)

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
            "witness_pseudo_target": (
                "pass1_proposal" if args.selector_witness == "proposal"
                else "last_executed_action"
            ),
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
        if args.selector_witness == "last_action":
            # note (luojiaxuan): 单遍口径——witness 伪目标 = 上一步已执行动作,
            # 与候选同 schema(parse_history_actions 输出),无需 proposal pass。
            prop = parsed[-1] if parsed else None
        elif proposal_action is not None:
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

        event_ids = [int(e["step_id"]) for e in history]
        tail_set = {alias.get(s, s) for s in event_ids[-budget:]}
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
                    if (args.max_replacements is not None
                            and sum(1 for c in child if c not in tail_set)
                            > args.max_replacements):
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

    def gapfold_messages(request, chosen):
        """按 chosen 集合的 gap-fold prompt(单遍与两遍第二遍共用口径)。"""
        import base64 as _b64  # noqa: F401 —— _decode_png 的输入已是 bytes 前置解码

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
                    event["restored_observation_screenshot_png_base64"])
        return build_mobile_official_messages_gapfold(
            goal=request["task"]["instruction"],
            action_texts=[e["action_text"] for e in history],
            full_responses=[e["full_response"] for e in history],
            shown_steps=chosen,
            step_images=step_images,
            current_image=_decode_png(
                request["current_screenshot_png_base64"]),
        )

    def adapter_scope(
        messages: list[dict[str, Any]],
        history_image_count: int,
        encoded: Any = None,
    ):
        """HGKV mask scope for one request; nullcontext when adapter off / K=0.

        # note (luojiaxuan): 传入 encoded 可复用调用方已算好的编码——掩码和 generate
        # 用的是同一份 prompt,重复 apply_chat_template 等于把图像预处理做两遍。
        """
        if adapter_meta is None or history_image_count < 1:
            return contextlib.nullcontext()
        from causalcache.policy.history_adapter_context import (
            HistoryAdapterContext,
            history_adapter_scope,
        )
        from causalcache.policy.history_token_roles import (
            build_history_token_mask,
        )

        if encoded is None:
            encoded = runtime.encode(messages)
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

    class _Job:
        __slots__ = ("messages", "history_images", "done", "result", "error")

        def __init__(self, messages, history_images):
            self.messages = messages
            self.history_images = history_images
            self.done = threading.Event()
            self.result = None
            self.error = None

    job_queue: "queue.Queue[_Job]" = queue.Queue()

    def batched_adapter_scope(batch_masks, seq_len: int):
        """把逐条掩码左填充成 [B, L] 后进入一次 scope。

        # note (luojiaxuan): HGKV hook 按 mask.shape == output.shape[:-1] 校验,
        # 所以批量必须给 [B, L];左填充位一律 False(pad 不是历史 token)。
        """
        if adapter_meta is None or not any(m is not None for m in batch_masks):
            return contextlib.nullcontext()
        import torch as _torch

        from causalcache.policy.history_adapter_context import (
            HistoryAdapterContext,
            history_adapter_scope,
        )

        rows = []
        for mask in batch_masks:
            row = _torch.zeros(seq_len, dtype=_torch.bool)
            if mask is not None:
                flat = mask[0] if mask.dim() == 2 else mask
                row[seq_len - flat.shape[0]:] = flat
            rows.append(row)
        stacked = _torch.stack(rows).to(base_runtime.device)
        return history_adapter_scope(HistoryAdapterContext(
            history_token_mask=stacked,
            history_present=bool(stacked.any()),
            image_roles=(),
        ))

    def build_mask(messages, history_image_count, encoded):
        if adapter_meta is None or history_image_count < 1:
            return None
        from causalcache.policy.history_token_roles import build_history_token_mask
        return build_history_token_mask(
            encoded["input_ids"], encoded["mm_token_type_ids"],
            encoded["image_grid_thw"], history_image_count, merge_size,
        )

    def batch_worker() -> None:
        while True:
            first = job_queue.get()
            jobs = [first]
            deadline = time.perf_counter() + args.batch_wait_ms / 1000.0
            while len(jobs) < args.batch_max:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    jobs.append(job_queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                per_job = [runtime.encode(j.messages) for j in jobs]
                masks = [
                    build_mask(j.messages, j.history_images, e)
                    for j, e in zip(jobs, per_job)
                ]
                encoded = runtime.encode_batch([j.messages for j in jobs])
                seq_len = int(encoded["input_ids"].shape[1])
                with batched_adapter_scope(masks, seq_len):
                    results = runtime.generate_batch(
                        [j.messages for j in jobs], encoded=encoded
                    )
                for job, result in zip(jobs, results):
                    job.result = result
            except Exception as error:  # noqa: BLE001
                for job in jobs:
                    job.error = error
            finally:
                for job in jobs:
                    job.done.set()

    if args.batch_max > 1:
        threading.Thread(target=batch_worker, daemon=True).start()

    def run_generate(messages, history_image_count):
        """batch_max=1 走原逐条路径(逐字节不变);>1 交给微批 worker。"""
        if args.batch_max <= 1:
            encoded = runtime.encode(messages)
            with adapter_scope(messages, history_image_count, encoded):
                return runtime.generate(messages, encoded=encoded)
        job = _Job(messages, history_image_count)
        job_queue.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result

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
                body_bytes = length
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                selection_info: dict[str, Any] | None = None
                presel_seconds = 0.0
                if (selector_model is not None
                        and args.selector_witness == "last_action"):
                    # 单遍主线:选记忆在生成之前完成,只生成一次。
                    presel_started = time.perf_counter()
                    chosen, diag = run_selection(request, None)
                    presel_seconds = time.perf_counter() - presel_started
                    event_ids = [
                        int(e["step_id"]) for e in request.get("history", [])
                    ]
                    tail = event_ids[-args.selection_budget:]
                    selection_info = {
                        "passes": 1,
                        "mode": "last_executed_action",
                        "recent_tail": tail,
                        "chosen": chosen,
                        "diagnostics": diag,
                    }
                    if chosen is not None and list(chosen) != list(tail):
                        messages = gapfold_messages(request, list(chosen))
                    else:
                        pass1 = dict(request)
                        pass1["selected_event_step_ids"] = list(tail)
                        messages = build_mobileworld_gui_owl_messages(pass1)
                elif selector_model is not None:
                    event_ids = [
                        int(e["step_id"]) for e in request.get("history", [])
                    ]
                    tail = event_ids[-args.selection_budget:]
                    pass1 = dict(request)
                    pass1["selected_event_step_ids"] = list(tail)
                    messages = build_mobileworld_gui_owl_messages(pass1)
                else:
                    messages = build_mobileworld_gui_owl_messages(request)
                messages = _apply_history_render(messages, args.history_render)
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
                    pass1_started = time.perf_counter()
                    generated = run_generate(messages, history_image_count)
                    pass1_seconds = time.perf_counter() - pass1_started
                    select_seconds = presel_seconds
                    pass2_seconds = 0.0
                    if (selector_model is not None
                            and args.selector_witness == "proposal"):
                        select_started = time.perf_counter()
                        chosen, diag = run_selection(
                            request, generated.canonical_action
                        )
                        select_seconds = time.perf_counter() - select_started
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
                            messages2 = _apply_history_render(
                                messages2, args.history_render
                            )
                            pass2_started = time.perf_counter()
                            generated = run_generate(
                                messages2, _history_image_count(messages2)
                            )
                            pass2_seconds = time.perf_counter() - pass2_started
                            selection_info["passes"] = 2
                    if selector_model is not None:
                        # note (luojiaxuan): 两遍开销审计——reviewer 关心 pass-2 触发率
                        # 与耗时占比;逐请求落日志,离线聚合出分布。
                        print(json.dumps({
                            "event": "SELECT_AUDIT",
                            "passes": selection_info["passes"],
                            "pool_size": len(request.get("history", [])),
                            "pass1_seconds": round(pass1_seconds, 4),
                            "select_seconds": round(select_seconds, 4),
                            "pass2_seconds": round(pass2_seconds, 4),
                        }), flush=True)
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
                # note (luojiaxuan): 逐请求耗时落到服务日志——闭环慢在哪以前只能靠
                # runner 时间戳反推,拆不出编码/前向/选择各占多少。
                print(
                    json.dumps({
                        "event": "MOBILEWORLD_POLICY_TIMING",
                        "request": request_count,
                        "request_decode_seconds": round(queued - arrived, 3),
                        "body_bytes": body_bytes,
                        "queue_seconds": round(queue_seconds, 3),
                        "pass1_seconds": round(pass1_seconds, 3),
                        "select_seconds": round(select_seconds, 3),
                        "pass2_seconds": round(pass2_seconds, 3),
                        "encode_seconds": round(
                            float(generated.metadata.get("encode_seconds", 0.0)), 3),
                        "prompt_tokens": generated.metadata.get("prompt_tokens"),
                        "generated_tokens": generated.metadata.get("generated_tokens"),
                        "history_images": history_image_count,
                    }, sort_keys=True),
                    flush=True,
                )
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
