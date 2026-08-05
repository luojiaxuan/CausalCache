#!/usr/bin/env python3
"""Serve one GUI-Owl policy for OSWorld — CausalCache-RL 独占版。

# note (luojiaxuan): 2026-08-05 从主仓 serve_osworld_official_policy.py fork
# (基点 = 主仓 5033c5a 时刻的全功能版)。分离原因:RL 的 selector/HGKV 会
# 高频演化,主线(论文/离线评测)的 serve 必须稳定不受扰。
#   * RL 专属能力全在本文件:--selector-temperature(Plackett-Luce 采样)、
#     --rl-audit-dir(逐步审计)、--selector-mode hidden(策略 hidden-state
#     打分头索引遍)、--selector-head/--index-visual-tokens;
#   * 主仓 serve 已回滚到 RL 之前(28094d8^),两边不再共享改动;
#     主仓的 bugfix 需要时手动 cherry-pick 过来,反向永不同步。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--visual-tokens", type=int, default=2560)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--memory-budget", type=int, default=4)
    parser.add_argument("--adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--adapter-checkpoint-sha256", default=None)
    parser.add_argument("--adapter-layer-count", type=int, default=8)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-alpha", type=int, default=16)
    parser.add_argument("--selector-bundle", type=Path, default=None)
    parser.add_argument("--selector-arch", choices=("two_tower", "concat"),
                        default="two_tower")
    parser.add_argument("--selector-beam", type=int, default=3)
    # note (luojiaxuan): None = 原语义(无条件凑满 B);给定 τ 则只提升预测边际 > τ
    # 的候选,不足 B 个用最近帧补齐。τ=0 是模型自身的"这一帧有用"边界,不需调参。
    parser.add_argument("--selector-min-marginal", type=float, default=None,
                        help="弃权阈值:预测边际 <= τ 的候选不提升,槽位留给最近帧")
    # note (luojiaxuan): 校准截距。two-tower 的 readout 塔在部署侧被 mask(它需要挂
    # 探针才能算),而该塔实测近似一个常数偏置(dev 上均值 +0.0625、标准差仅 0.031)。
    # 去掉它使 cheap 塔单独预测在 15,664 条 dev 边上**无一为正**(均值 −0.0778,
    # 而标签 47.7% 为正、均值 +0.0076)。常数偏置不改变 argmax,所以原无阈值臂不受
    # 影响;但它让任何阈值/早停判据全部失效。补上截距后阈值才有意义。
    parser.add_argument("--selector-score-offset", type=float, default=0.0,
                        help="加到 selector 预测边际上的常数;修正 readout 塔缺席造成的偏置")
    # note (luojiaxuan): proposal = CausalCache-P(两遍,先按 recent 尾出拟议动作
    # 作参照,换帧则重组重生成);last_action = CausalCache-LA(单遍)。
    parser.add_argument("--selector-witness",
                        choices=("last_action", "proposal"),
                        default="last_action")
    # note (luojiaxuan): CausalCache-RL(2026-08-04 pivot)。温度 > 0 时选帧从
    # beam-argmax 换成 Plackett-Luce 顺序采样:B 个槽位依次从 softmax(边际/τ)
    # 无放回抽取。这就是 RL 的探索来源;τ→0 退回部署语义。采样轮的候选特征
    # (已归一化)与选中项写进 --rl-audit-dir,trainer 直接在特征上复算
    # log π_sel 做策略梯度——不需要在训练侧重建环境状态。
    # 温度模式下忽略 --selector-min-marginal(弃权交给 RL 自己学)。
    parser.add_argument("--selector-temperature", type=float, default=0.0,
                        help="Plackett-Luce 采样温度;0 = 原 beam-argmax")
    parser.add_argument("--rl-audit-dir", type=Path, default=None,
                        help="RL 轨迹审计目录(每请求一行 jsonl,含采样轮特征)")
    # note (luojiaxuan): selector v2(2026-08-05,用户裁定废除手写特征)。
    # hidden 模式 = "外围视觉"索引遍:全部历史图缩略(--index-visual-tokens/图)
    # + 各步动作行 + 当前屏,过冻结策略一次,各历史图 token 段 mean-pool hidden
    # state,线性头打分 → PL 采样。零手写特征;年龄/新近以位置编码形式留在
    # 模型自己的表征里,用不用由 RL 决定。cheap 模式原样保留(对照用)。
    parser.add_argument("--selector-mode", choices=("cheap", "hidden"),
                        default="cheap")
    parser.add_argument("--selector-head", type=Path, default=None,
                        help="hidden 模式打分头 bundle(make_hidden_head.py 产出)")
    parser.add_argument("--index-visual-tokens", type=int, default=144,
                        help="索引遍每张历史缩略图的视觉 token 预算")
    args = parser.parse_args()

    if args.selector_witness == "last_action":
        # 必须在特征模块导入前生效:witness 伪目标 = 上一步已执行动作(单遍)。
        os.environ["CAUSALCACHE_WITNESS_PSEUDO_TARGET"] = "last_action"

    import hashlib

    import torch

    from causalcache.osworld_gui_owl import (
        GUIOwlOSWorldRuntime,
        parse_gui_owl_osworld_action,
    )
    from causalcache.osworld_official_online import (
        build_official_messages_for_request,
        dedup_alias,
        eligible_pool,
        official_forms_from_history,
        synthetic_selector_record,
    )
    from causalcache.selector_v4_features import (
        candidate_features,
        set_context_features,
    )

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    processor = runtime.processor
    model = runtime.model
    tokens = runtime.generation_tokens
    merge_size = int(processor.image_processor.merge_size)

    adapter_meta: dict[str, Any] | None = None
    if args.adapter_checkpoint is not None:
        from causalcache.policy.history_gated_lora import (
            inject_history_gated_kv,
            load_history_gated_state_dict,
        )

        digest = hashlib.sha256(args.adapter_checkpoint.read_bytes()).hexdigest()
        if (args.adapter_checkpoint_sha256
                and digest != args.adapter_checkpoint_sha256):
            raise SystemExit("adapter checkpoint SHA drifted")
        wrapped = inject_history_gated_kv(
            model,
            layer_count=args.adapter_layer_count,
            rank=args.adapter_rank,
            alpha=args.adapter_alpha,
        )
        load_history_gated_state_dict(
            wrapped, torch.load(args.adapter_checkpoint, map_location="cpu")
        )
        model.eval().requires_grad_(False)
        adapter_meta = {
            "checkpoint": str(args.adapter_checkpoint),
            "checkpoint_sha256": digest,
            "layer_count": args.adapter_layer_count,
            "rank": args.adapter_rank,
            "alpha": args.adapter_alpha,
        }

    selector_meta: dict[str, Any] | None = None
    selector_model = None
    selector_norm = None
    if args.selector_bundle is not None:
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

            def forward(self, x):
                # 部署侧 readout 塔恒 mask(two-tower 结构红利)。
                return self.cheap(x).squeeze(-1)

        class _Concat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(s_dim + s_ro, s_hidden), torch.nn.GELU(),
                    torch.nn.Linear(s_hidden, s_hidden), torch.nn.GELU(),
                    torch.nn.Linear(s_hidden, 1),
                )

            def forward(self, x):
                if s_ro:
                    x = torch.cat([x, torch.zeros(x.shape[0], s_ro)], dim=-1)
                return self.net(x).squeeze(-1)

        selector_model = (_TwoTower() if args.selector_arch == "two_tower"
                          else _Concat())
        selector_model.load_state_dict(bundle["model_state"])
        selector_model.eval()
        s_mean, s_std = bundle["mean"], bundle["std"]
        selector_norm = lambda f: [  # noqa: E731
            (v - m) / s for v, m, s in zip(f, s_mean, s_std)]
        selector_meta = {
            "bundle": str(args.selector_bundle),
            "arch": args.selector_arch,
            "budget": args.memory_budget,
            "beam": args.selector_beam,
            "readout_online": False,
            "witness_pseudo_target": (
                "last_executed_action"
                if args.selector_witness == "last_action"
                else "pass1_proposal"
            ),
            "passes": 1 if args.selector_witness == "last_action" else "1_or_2",
        }

    # ---- hidden 模式装载:索引遍处理器(缩略图预算)+ 线性打分头 ----
    hidden_head = None
    idx_processor = None
    if args.selector_mode == "hidden":
        import copy as _copy

        if args.selector_head is None:
            raise SystemExit("--selector-mode hidden 需要 --selector-head")
        _hb = torch.load(args.selector_head, map_location="cpu",
                         weights_only=False)
        if _hb.get("kind") != "hidden_head":
            raise SystemExit(f"selector-head kind 应为 hidden_head,得到 {_hb.get('kind')}")
        _hsize = int(_hb["hidden_size"])
        _msize = int(getattr(model.config, "text_config", model.config).hidden_size)
        if _hsize != _msize:
            raise SystemExit(f"打分头 hidden_size={_hsize} 与模型 {_msize} 不符")
        hidden_head = torch.nn.Linear(_hsize, 1)
        hidden_head.load_state_dict(_hb["model_state"])
        hidden_head.eval()
        idx_processor = _copy.deepcopy(processor)
        # Qwen 系:merged token ≈ 28×28 px;max_pixels 决定缩略后 token 数
        idx_processor.image_processor.max_pixels = args.index_visual_tokens * 28 * 28
        idx_processor.image_processor.min_pixels = 16 * 28 * 28

    selector_active = (selector_model is not None) or (hidden_head is not None)

    def run_selection_hidden(request):
        """索引遍:全历史缩略 + 动作行 → hidden 池化 → 头打分 → PL 采样。"""
        import base64 as _b64
        import io as _io

        import numpy as _np
        from PIL import Image as _Image

        history = request["history"]
        forms = official_forms_from_history(history)
        pool = eligible_pool(history, forms)
        budget = args.memory_budget
        if len(pool) < budget:
            return None, {"reason": "pool_smaller_than_budget", "pool": len(pool)}
        line_of = {f.step_id: f.action_line for f in forms}

        content = [{"type": "text",
                    "text": "Task: " + str(request["task"]["instruction"])}]
        imaged_events = []
        for event in history:
            sid = int(event["step_id"])
            payload = event.get("restored_post_screenshot_png_base64")
            content.append({"type": "text",
                            "text": f"Step {sid}: {line_of.get(sid, '(action)')}"})
            if payload:
                content.append({"type": "image", "image": _Image.open(
                    _io.BytesIO(_b64.b64decode(payload)))})
                imaged_events.append(sid)
        cur_payload = request.get("current_screenshot_png_base64")
        if cur_payload:
            content.append({"type": "text", "text": "Current screen:"})
            content.append({"type": "image", "image": _Image.open(
                _io.BytesIO(_b64.b64decode(cur_payload)))})
        content.append({"type": "text",
                        "text": "Which past steps matter for the next action?"})
        candidates = [s for s in pool if s in set(imaged_events)]
        if len(candidates) < budget:
            return None, {"reason": "pool_missing_screenshots",
                          "pool": len(pool), "imaged": len(candidates)}

        enc = idx_processor.apply_chat_template(
            [[{"role": "user", "content": content}]],
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**{k: v for k, v in enc.items()},
                        output_hidden_states=True)
            hidden = out.hidden_states[-1][0].float().cpu()
        # 图像 token 段 = mm_token_type_ids 的连续 1 块;顺序 = imaged_events + 当前屏
        types = enc["mm_token_type_ids"][0].tolist()
        spans, start = [], None
        for i, t in enumerate(types + [0]):
            if t == 1 and start is None:
                start = i
            elif t != 1 and start is not None:
                spans.append((start, i))
                start = None
        expect = len(imaged_events) + (1 if cur_payload else 0)
        if len(spans) != expect:
            raise ValueError(
                f"索引遍图像块数 {len(spans)} != 期望 {expect}(fail-closed)")
        vec_of = {}
        for sid, (a, b) in zip(imaged_events, spans[: len(imaged_events)]):
            vec_of[sid] = hidden[a:b].mean(dim=0)
        feats = torch.stack([vec_of[s] for s in candidates])
        with torch.no_grad():
            scores = hidden_head(feats).squeeze(-1)

        diag_common = {
            "pool": len(candidates), "mode": "hidden",
            "index_tokens": args.index_visual_tokens,
        }
        if args.selector_temperature > 0:
            tau_rl = args.selector_temperature
            remaining = list(candidates)
            rem_scores = scores.clone()
            picks, rounds, logprob_sum = [], [], 0.0
            for _ in range(budget):
                logp = torch.log_softmax(rem_scores / tau_rl, dim=0)
                idx = int(torch.multinomial(logp.exp(), 1).item())
                logprob_sum += float(logp[idx])
                rounds.append({"candidates": [int(s) for s in remaining],
                               "chosen": int(remaining[idx]),
                               "logprob": round(float(logp[idx]), 6)})
                picks.append(remaining[idx])
                keep = [i for i in range(len(remaining)) if i != idx]
                remaining = [remaining[i] for i in keep]
                rem_scores = rem_scores[keep]
            hv = {str(s): _b64.b64encode(
                      _np.asarray(vec_of[s], dtype=_np.float16).tobytes()
                  ).decode() for s in candidates}
            return sorted(picks), {
                **diag_common, "promoted": budget,
                "temperature": tau_rl,
                "rl_logprob_sum": round(logprob_sum, 6),
                "rl_rounds": rounds,
                "hidden_vectors": hv,
                "hidden_size": int(feats.shape[1]),
            }
        top = torch.topk(scores, budget).indices.tolist()
        chosen = sorted(candidates[i] for i in top)
        return chosen, {**diag_common, "promoted": budget, "tau": None}

    def run_selection(request, reference_arguments=None):
        if args.selector_mode == "hidden":
            return run_selection_hidden(request)
        """eligible 池上的 beam exact-B;返回 (chosen 或 None, diagnostics)。

        reference_arguments 非空时(proposal 模式)以其为 witness 目标动作
        (computer_use 实参形制,[0,999] 坐标系,与离线 gold 目标同一 matcher);
        为空时按 CAUSALCACHE_WITNESS_PSEUDO_TARGET 环境变量(last_action)。
        """
        history = request["history"]
        forms = official_forms_from_history(history)
        pool = eligible_pool(history, forms)
        budget = args.memory_budget
        if len(pool) < budget:
            return None, {"reason": "pool_smaller_than_budget", "pool": len(pool)}
        record = synthetic_selector_record(request)
        if reference_arguments is not None:
            record["target_tool_call"] = {"arguments": dict(reference_arguments)}
        alias = dedup_alias(history)
        duplicates = {str(src): kept for src, kept in alias.items()}

        def marginals(selected, candidates, return_features=False):
            xs = []
            for event in candidates:
                f = candidate_features(
                    record, candidate_pool=pool, duplicates=duplicates,
                    event=event,
                ) + set_context_features(
                    record, selected=list(selected), event=event,
                    duplicates=duplicates,
                )
                xs.append(selector_norm(f))
            with torch.no_grad():
                raw = selector_model(
                    torch.tensor(xs, dtype=torch.float32)).tolist()
            off = args.selector_score_offset
            margs = [v + off for v in raw] if off else raw
            return (margs, xs) if return_features else margs

        # ---- RL 模式:Plackett-Luce 顺序采样(exact-B,无放回) ----
        # note (luojiaxuan): 每轮把"已选集合"喂回 set_context_features 再算边际,
        # 与 beam 的逐层语义一致;log π_sel(S) = Σ_轮 [m_chosen/τ − logsumexp(m/τ)]。
        # 审计里存**归一化后的特征矩阵**而不是原始状态:trainer 复算 log π 只需
        # 特征 × 当前 selector 参数,彻底避开"训练侧重建 prompt 状态"这类漂移源。
        if args.selector_temperature > 0:
            tau_rl = args.selector_temperature
            sel: tuple = ()
            rounds = []
            logprob_sum = 0.0
            for _ in range(budget):
                remaining = [s for s in pool if s not in sel]
                if not remaining:
                    break
                margs, feats = marginals(sel, remaining, return_features=True)
                scaled = torch.tensor(margs, dtype=torch.float32) / tau_rl
                logp = torch.log_softmax(scaled, dim=0)
                idx = int(torch.multinomial(logp.exp(), 1).item())
                logprob_sum += float(logp[idx])
                rounds.append({
                    "candidates": [int(s) for s in remaining],
                    "features": [[round(v, 6) for v in f] for f in feats],
                    "chosen": int(remaining[idx]),
                    "logprob": round(float(logp[idx]), 6),
                })
                sel = tuple(sorted((*sel, remaining[idx])))
            chosen = sorted(sel)
            return chosen, {
                "pool": len(pool), "promoted": len(chosen),
                "temperature": tau_rl,
                "rl_logprob_sum": round(logprob_sum, 6),
                "rl_rounds": rounds,
            }

        # note (luojiaxuan): --selector-min-marginal 打开弃权。原语义是无条件走满 B 层、
        # 恒取 B 个:池子里没有值得提升的远端帧时,argmax 仍会挑一个出来,挤掉 recent。
        # 同域诊断显示这正是伤害所在——单应用文档类(writer 0 赢 4 输、impress 2:4、
        # thunderbird 0:2)净损失,而跨应用/图形类(multi_apps、gimp、os、vlc)净收益。
        # 打开后:只有预测边际 > τ 的候选才允许进入集合,不足 B 个时用最近帧补齐。
        # 预算仍是 B,只是不强制把槽位交给远端帧。
        tau = args.selector_min_marginal
        level = [((), 0.0)]
        for _ in range(budget):
            expanded, seen = [], set()
            for sel, acc in level:
                remaining = [s for s in pool if s not in sel]
                if not remaining:
                    continue
                margs = marginals(sel, remaining)
                ranked = sorted(zip(remaining, margs), key=lambda t: -t[1])
                if tau is not None:
                    ranked = [(s, m) for s, m in ranked if m > tau]
                for s, m in ranked[: args.selector_beam]:
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
        if len(chosen) < budget:
            if tau is None:
                return None, {"reason": "beam_short", "pool": len(pool)}
            # 弃权模式:剩余槽位交给最近帧(等价于"这些位置维持 Recent 默认")
            promoted = len(chosen)
            for s in sorted(pool, reverse=True):
                if len(chosen) >= budget:
                    break
                if s not in chosen:
                    chosen = sorted((*chosen, s))
            return chosen, {"pool": len(pool), "promoted": promoted,
                            "filled_recent": budget - promoted, "tau": tau}
        return chosen, {"pool": len(pool), "promoted": len(chosen), "tau": tau}

    def adapter_scope(messages, history_image_count: int):
        if adapter_meta is None or history_image_count < 1:
            return contextlib.nullcontext()
        from causalcache.policy.history_adapter_context import (
            HistoryAdapterContext,
            history_adapter_scope,
        )
        from causalcache.policy.history_token_roles import (
            build_history_token_mask,
        )

        encoded = processor.apply_chat_template(
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
            history_token_mask=mask.to(runtime.device),
            history_present=bool(mask.any()),
            image_roles=("restored_history",) * history_image_count
            + ("current",),
        )
        return history_adapter_scope(context)

    def generate_official(messages):
        encoded = processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=False,
        ).to(runtime.device)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokens.tool_call_close_token_id,
                pad_token_id=tokens.pad_token_id,
                suppress_tokens=list(tokens.standard_eos_token_ids),
                num_beams=1,
                num_return_sequences=1,
            )
        seconds = time.perf_counter() - started
        new_tokens = generated[:, prompt_tokens:]
        text = processor.batch_decode(
            new_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        return text, {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(new_tokens.shape[1]),
            "generation_seconds": round(seconds, 4),
        }

    inference_lock = threading.Lock()
    counter_lock = threading.Lock()
    counters = {"requests": 0, "failures": 0, "parse_failures": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                _json_response(self, 404, {"error": "not_found"})
                return
            with counter_lock:
                observed = dict(counters)
            _json_response(self, 200, {
                "status": "ready",
                "counters": observed,
                "memory_budget": args.memory_budget,
                "adapter": adapter_meta,
                "selector": selector_meta,
                "runtime": runtime.metadata,
            })

        def do_POST(self):  # noqa: N802
            if self.path != "/act":
                _json_response(self, 404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                history = request.get("history", [])
                forms = official_forms_from_history(history)
                pool = eligible_pool(history, forms)
                tail = pool[-args.memory_budget:] if args.memory_budget else []
                screen_size = tuple(request["screen_size"])

                def parse_output(text):
                    arguments = None
                    err = None
                    try:
                        matches = re.findall(
                            r"<tool_call>\s*(.*?)\s*</tool_call>",
                            text, re.DOTALL)
                        if len(matches) == 1:
                            payload = json.loads(matches[0])
                            if isinstance(payload, dict):
                                arguments = payload.get("arguments")
                        action = parse_gui_owl_osworld_action(
                            text, screen_size=screen_size)
                        mapping = action.to_mapping()
                    except Exception as error:  # noqa: BLE001
                        err = f"{error.__class__.__name__}: {error}"
                        mapping = {"type": "wait"}
                    return arguments, mapping, err

                selection_info: dict[str, Any] | None = None
                select_seconds = 0.0
                pass2_seconds = 0.0
                shown = list(tail)
                if (selector_active and args.memory_budget > 0
                        and args.selector_witness == "last_action"):
                    select_started = time.perf_counter()
                    chosen, diag = run_selection(request)
                    select_seconds = time.perf_counter() - select_started
                    selection_info = {
                        "passes": 1,
                        "mode": "last_executed_action",
                        "recent_tail": tail,
                        "chosen": chosen,
                        "diagnostics": diag,
                    }
                    if chosen is not None:
                        shown = list(chosen)
                messages = build_official_messages_for_request(request, shown)
                with inference_lock:
                    with adapter_scope(messages, len(shown)):
                        text, generation = generate_official(messages)
                    if (selector_active and args.memory_budget > 0
                            and args.selector_witness == "proposal"):
                        prop_args, _, _ = parse_output(text)
                        select_started = time.perf_counter()
                        chosen, diag = run_selection(
                            request, reference_arguments=prop_args or None)
                        select_seconds = time.perf_counter() - select_started
                        selection_info = {
                            "passes": 1,
                            "mode": "pass1_proposal",
                            "recent_tail": tail,
                            "chosen": chosen,
                            "diagnostics": diag,
                        }
                        if chosen is not None and list(chosen) != list(tail):
                            shown = list(chosen)
                            messages2 = build_official_messages_for_request(
                                request, shown)
                            with adapter_scope(messages2, len(shown)):
                                text2, generation2 = generate_official(
                                    messages2)
                            pass2_seconds = generation2["generation_seconds"]
                            generation = {
                                "prompt_tokens": generation2["prompt_tokens"],
                                "generated_tokens":
                                    generation2["generated_tokens"],
                                "generation_seconds":
                                    generation["generation_seconds"],
                                "pass2_generation_seconds": pass2_seconds,
                            }
                            text = text2
                            selection_info["passes"] = 2
                official_arguments, action_mapping, parse_error = (
                    parse_output(text))
                if parse_error is not None:
                    with counter_lock:
                        counters["parse_failures"] += 1
                if selector_active:
                    # note (luojiaxuan): 记录 promoted / filled_recent / k,
                    # 否则跑完无法判断弃权到底触发了多少、selector 偏离 recent 多远。
                    _si = selection_info or {}
                    _diag = _si.get("diagnostics") or {}
                    _chosen = _si.get("chosen")
                    _tail = set(_si.get("recent_tail") or [])
                    print(json.dumps({
                        "event": "SELECT_AUDIT",
                        "passes": _si.get("passes", 1),
                        "pool_size": len(pool),
                        "promoted": _diag.get("promoted"),
                        "filled_recent": _diag.get("filled_recent"),
                        "k_off_recent": (
                            sum(1 for s in _chosen if s not in _tail)
                            if _chosen else None),
                        "reason": _diag.get("reason"),
                        "pass1_seconds": generation["generation_seconds"],
                        "select_seconds": round(select_seconds, 4),
                        "pass2_seconds": round(pass2_seconds, 4),
                    }), flush=True)
                    # ---- RL 审计:一步一行,自带 join 键与复算所需的全部特征 ----
                    # join = (task_id, step_index, response_sha16):task+步定位到
                    # result.json 的那一步;response 哈希区分同任务重试的多次尝试。
                    if args.rl_audit_dir is not None and _diag.get("rl_rounds"):
                        import hashlib as _hl
                        audit_line = {
                            "task_id": str(request["task"]["task_id"]),
                            "step_index": len(request["history"]) + 1,
                            "response_sha16": _hl.sha256(
                                text.encode("utf-8")).hexdigest()[:16],
                            "shown_events": shown,
                            "temperature": _diag.get("temperature"),
                            "rl_logprob_sum": _diag.get("rl_logprob_sum"),
                            "rounds": _diag.get("rl_rounds"),
                            "mode": _diag.get("mode", "cheap"),
                            "hidden_vectors": _diag.get("hidden_vectors"),
                            "hidden_size": _diag.get("hidden_size"),
                            "full_response": text,
                        }
                        with counter_lock:
                            args.rl_audit_dir.mkdir(parents=True, exist_ok=True)
                            with (args.rl_audit_dir /
                                  f"audit-{args.port}.jsonl").open(
                                      "a", encoding="utf-8") as fh:
                                fh.write(json.dumps(audit_line) + "\n")
                with counter_lock:
                    counters["requests"] += 1
                _json_response(self, 200, {
                    "schema_version": OSWORLD_POLICY_RESPONSE_SCHEMA_VERSION,
                    "action": action_mapping,
                    "official_arguments": official_arguments,
                    "full_response": text,
                    "parse_error": parse_error,
                    "selection": selection_info,
                    "shown_events": shown,
                    "generation": generation,
                    "cost": {
                        "passes": (selection_info or {}).get("passes", 1),
                        "pass1_seconds": generation["generation_seconds"],
                        "select_seconds": round(select_seconds, 4),
                        "pass2_seconds": round(pass2_seconds, 4),
                    },
                })
            except Exception as error:  # noqa: BLE001
                with counter_lock:
                    counters["failures"] += 1
                print(json.dumps({
                    "event": "OSWORLD_POLICY_FAILURE",
                    "error_type": error.__class__.__name__,
                    "error": str(error),
                }), flush=True)
                _json_response(self, 500, {
                    "error": error.__class__.__name__,
                    "message": str(error),
                })

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(json.dumps({
        "status": "READY_OSWORLD_OFFICIAL_POLICY",
        "port": args.port,
        "memory_budget": args.memory_budget,
        "adapter": adapter_meta,
        "selector": selector_meta,
        "runtime": runtime.metadata,
    }, sort_keys=True), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
