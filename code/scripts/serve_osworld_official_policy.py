#!/usr/bin/env python3
"""Serve one GUI-Owl policy for OSWorld under the official desktop protocol.

# note (luojiaxuan): 与 mobile serve 同构的桌面 benchmark 版(旧
# serve_osworld_gui_owl_policy.py 是 witness 语料时代的单轮口径,保留不动):
#   * prompt 走 build_desktop_official_messages(v4 训练同口径,系统提示内嵌
#     computer_use <tools>,编码时**不传** tools 实参);
#   * 可选 --adapter-checkpoint 挂 HGKV(prefill 在 history_adapter_scope 内,
#     decode 步由 hook 长度检查自然 bypass;K=0 与未挂时与冻结 bitwise 相同);
#   * 可选 --selector-bundle 启用单遍 last-action selector:witness 伪目标 =
#     上一步已执行动作(CAUSALCACHE_WITNESS_PSEUDO_TARGET=last_action 走离线
#     判定同一特征分支),beam exact-B、recent 兜底,选完只生成一次;
#   * 无 selector 时 shown = eligible 池的 recent 尾(B0 即空)。
# 请求携带全池截图(runner memory_arm=full;B0 用 summary 不带图)。
"""

from __future__ import annotations

import argparse
import contextlib
import base64
import json
import os
import hashlib
import sys
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
    parser.add_argument("--agentic-selector-ckpt", type=Path, default=None,
                        help="Phase 3/4 的 SubsetSelectorPolicy(RL 学习 selector);"
                             "与旧 --selector-bundle 互斥")
    parser.add_argument("--lora-bundle", type=Path, default=None,
                        help="qkvo LoRA bundle(Phase 2 policy_mem_sft 产物;"
                             "与 --adapter-checkpoint 的 HGKV 互斥)")
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

    # note (luojiaxuan): Phase 5 真实 UI 桥接 —— 加载 agentic 线的 qkvo LoRA
    # (policy_mem_sft)。与上面的 HGKV 是两种不同 adapter,不许同挂:
    # 都改注意力投影,叠加后行为无定义。bundle 自带 rank/alpha/target/层数,
    # 不从 CLI 读,保证与训练时逐字一致。
    if args.lora_bundle is not None:
        if args.adapter_checkpoint is not None:
            raise SystemExit("--lora-bundle 与 --adapter-checkpoint 互斥")
        from train_success_sft_lora import inject_lora, load_lora_state_dict

        lb = torch.load(args.lora_bundle, map_location="cpu",
                        weights_only=False)
        wrapped_lora = inject_lora(
            model, rank=int(lb["rank"]), alpha=int(lb["alpha"]),
            target_modules=tuple(str(lb.get(
                "target_modules", "q_proj,k_proj,v_proj,o_proj")).split(",")),
            torch=torch,
            last_layer_count=(int(lb["last_layers"])
                              if lb.get("last_layers") else None))
        load_lora_state_dict(wrapped_lora, lb["state"])
        model.eval().requires_grad_(False)
        adapter_meta = {
            "kind": "qkvo_lora",
            "checkpoint": str(args.lora_bundle),
            "checkpoint_sha256": hashlib.sha256(
                args.lora_bundle.read_bytes()).hexdigest(),
            "rank": int(lb["rank"]), "alpha": int(lb["alpha"]),
            "last_layers": lb.get("last_layers"),
        }

    # note (luojiaxuan): agentic 线的学习 selector(Phase 3/4 RL 产物)。
    # **编号口径已核对**:agentic 帧索引 ≡ OSWorld 事件 step_id(帧 j = 1-based
    # 动作 j 的 post 帧);当前帧 = len(history);候选 = eligible pool ⊆
    # [1..len(history)-1];动作行 = forms[j-1].action_line。任何一处对不上,
    # subset 就静默指向另一批帧(§4.3 的教训),所以这里逐条写死并断言。
    agentic_sel = None
    agentic_builder = None
    agentic_cache = None
    if args.agentic_selector_ckpt is not None:
        if args.selector_bundle is not None:
            raise SystemExit("--agentic-selector-ckpt 与 --selector-bundle 互斥")
        repo_root = Path(__file__).resolve().parents[2]
        for cand in (repo_root / "rl" / "code", Path("/data/agentic/code")):
            if (cand / "causalcache_agentic").exists():
                sys.path.insert(0, str(cand))
                break
        from causalcache_agentic import features as agf
        from causalcache_agentic.selector_model import (
            FrozenTextEmbedder,
            SelectorStateBuilder,
            SubsetSelectorPolicy,
        )

        agentic_sel = SubsetSelectorPolicy.load(
            str(args.agentic_selector_ckpt), map_location=args.device)
        agentic_sel.to(args.device).eval()
        _emb = (getattr(agentic_sel, "embedder", None)
                or getattr(agentic_sel, "text_embedder", None))
        if _emb is None and callable(getattr(agentic_sel, "make_embedder", None)):
            _emb = agentic_sel.make_embedder()
        if _emb is None:
            _emb = FrozenTextEmbedder.from_model_dir(
                model_dir=str(args.model_dir), device=args.device)
        agentic_builder = SelectorStateBuilder(embedder=_emb)
        agentic_cache = agf.build_cache(argparse.Namespace(
            dry_run=False, model_dir=args.model_dir,
            snapshot_manifest=args.snapshot_manifest, device=args.device,
            visual_tokens=2560, feature_dtype="float16",
            feature_dim=None, feature_tokens=None, dummy_seed=-1,
            feature_cache_dir=None, feature_cache_items=64))
        adapter_meta = dict(adapter_meta or {})
        adapter_meta["agentic_selector"] = {
            "checkpoint": str(args.agentic_selector_ckpt),
            "checkpoint_sha256": hashlib.sha256(
                args.agentic_selector_ckpt.read_bytes()).hexdigest()}

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

    def run_selection(request, reference_arguments=None):
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

        def marginals(selected, candidates):
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
            return [v + off for v in raw] if off else raw

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
                if (agentic_sel is not None and args.memory_budget > 0
                        and pool):
                    select_started = time.perf_counter()
                    # 截图按 sha 落盘一次,FeatureCache 以路径为键做 LRU
                    tmp_root = Path(os.environ.get("TMPDIR", "/tmp")) / "agentic_shots"
                    tmp_root.mkdir(parents=True, exist_ok=True)
                    path_by_sid: dict[int, str] = {}
                    for event in history:
                        sid = int(event["step_id"])
                        if sid not in pool:
                            continue
                        sha = event.get("post_screenshot_sha256") or f"sid{sid}"
                        fp = tmp_root / f"{sha}.png"
                        if not fp.exists():
                            fp.write_bytes(base64.b64decode(
                                event["restored_post_screenshot_png_base64"]))
                        path_by_sid[sid] = str(fp)
                    cur_sha = hashlib.sha256(
                        request["current_screenshot_png_base64"].encode()).hexdigest()[:40]
                    cur_fp = tmp_root / f"cur_{cur_sha}.png"
                    if not cur_fp.exists():
                        cur_fp.write_bytes(base64.b64decode(
                            request["current_screenshot_png_base64"]))
                    forms_lines = [f.action_line for f in forms]
                    cur_step = len(history)          # 当前帧的事件索引 = t-1
                    assert all(1 <= s < cur_step for s in pool), \
                        f"pool 越界: {pool} vs 当前帧 {cur_step}"
                    b_eff = min(args.memory_budget, len(pool))
                    with inference_lock:
                        with torch.inference_mode():
                            state = agentic_builder.build(
                                agentic_sel,
                                task_instruction=str(
                                    request["task"]["instruction"]),
                                history_action_lines=forms_lines,
                                candidates=list(pool),
                                frame_features=(
                                    lambda sid: agentic_cache(
                                        sid, path_by_sid[sid])),
                                current_step=cur_step,
                                current_feature=agentic_cache(
                                    cur_step, str(cur_fp)),
                                budget=b_eff,
                                recent=list(tail))
                            chosen = tuple(int(x) for x in agentic_sel.argmax(
                                state, list(pool), b_eff))
                    select_seconds = time.perf_counter() - select_started
                    shown = sorted(chosen)
                    selection_info = {
                        "passes": 1, "mode": "agentic_subset_selector",
                        "recent_tail": tail, "chosen": shown,
                        "diagnostics": {"pool": list(pool), "b_eff": b_eff}}
                elif (selector_model is not None and args.memory_budget > 0
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
                    if (selector_model is not None and args.memory_budget > 0
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
                if selector_model is not None:
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
