#!/usr/bin/env python3
"""§9 第 6/7 项:HGKV 生成式 parser validity 与完整动作等价率(dev)。

# note (luojiaxuan): s300 冻结前的最后一道闸。对 dev 每组的 R(Recent-B)与
# S(替换后)两个 prompt,分别用 Frozen 与 HGKV checkpoint 做贪心生成:
#   * parser validity:输出可被 parse_gui_owl_osworld_action 解析(恰一个
#     tool_call、schema 合法);
#   * 完整动作等价:生成的 tool_call 与 gold target_tool_call 在 action 名、
#     参数、坐标(τ=25,[0,999] 空间)上全等(type 文本精确、key 序列 casefold、
#     scroll 方向一致)——与语料构造的等价口径同源;
# 生成接线复用移动线闭环验证过的语义(gui_owl_v2_1_runtime._history_generation_scope):
# K = prompt 图数 − 1,prefill 在 history_adapter_scope 内、增量固化进 KV cache,
# decode 步由 hook 长度检查自然 bypass;K=0 用 nullcontext 保 parity。
# 判定(§9):HGKV 的 parser validity 不低于 Frozen;等价率不退化(报告配对差)。
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from causalcache.agentnet_desktop_cr import (
    build_agentnet_cr_request,
    render_agentnet_cr_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tool_call_equivalent(generated: dict, target: dict, *, tolerance: int = 25) -> bool:
    """完整动作等价:action + 参数 + 坐标(τ,[0,999])——语料构造同口径。"""
    g_args = generated.get("arguments") or {}
    t_args = target.get("arguments") or {}
    if generated.get("name") != target.get("name"):
        return False
    g_action, t_action = g_args.get("action"), t_args.get("action")
    click_family = {"left_click", "click"}
    if not (
        g_action == t_action
        or (g_action in click_family and t_action in click_family)
    ):
        return False
    def coord(args, key):
        raw = args.get(key)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None
        try:
            return (round(float(raw[0])), round(float(raw[1])))
        except (TypeError, ValueError):
            return None
    for key in ("coordinate", "coordinate2"):
        t_xy = coord(t_args, key)
        if t_xy is None:
            continue
        g_xy = coord(g_args, key)
        if g_xy is None:
            return False
        if max(abs(g_xy[0] - t_xy[0]), abs(g_xy[1] - t_xy[1])) > tolerance:
            return False
    if "text" in t_args and g_args.get("text") != t_args["text"]:
        return False
    if "keys" in t_args:
        g_keys = g_args.get("keys")
        if not isinstance(g_keys, list):
            return False
        if [str(k).casefold() for k in g_keys] != [
            str(k).casefold() for k in t_args["keys"]
        ]:
            return False
    if "pixels" in t_args:
        g_pixels = g_args.get("pixels")
        if not isinstance(g_pixels, (int, float)):
            return False
        if (float(g_pixels) > 0) != (float(t_args["pixels"]) > 0):
            return False
    if "status" in t_args and g_args.get("status") != t_args["status"]:
        return False
    return True


def main() -> None:
    args = parse_args()
    import re

    import torch
    from causalcache.osworld_gui_owl import (
        _TOOL_SPEC,
        GUIOwlOSWorldRuntime,
        parse_gui_owl_osworld_action,
    )
    from causalcache.policy.history_adapter_context import (
        HistoryAdapterContext,
        history_adapter_scope,
    )
    from causalcache.policy.history_gated_lora import (
        inject_history_gated_kv,
        load_history_gated_state_dict,
    )
    from causalcache.policy.history_token_roles import build_history_token_mask

    records = {
        record["dp_id"]: record
        for record in load_jsonl(args.screening_manifest)
    }
    rows = load_jsonl(args.corpus_root / "samples.jsonl")
    dev_arms = [
        row for row in rows
        if row["split"] == "dev" and row["arm_slot"] in ("R0", "S0")
    ]
    dev_arms = [
        arm for index, arm in enumerate(sorted(dev_arms, key=lambda r: r["sample_id"]))
        if index % args.shard_count == args.shard_index
    ]

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=2560,
        max_new_tokens=args.max_new_tokens,
    )
    model = runtime.model
    merge_size = int(runtime.processor.image_processor.merge_size)
    wrapped = inject_history_gated_kv(model, layer_count=8, rank=8, alpha=16)
    adapter_state = torch.load(args.checkpoint, map_location="cpu")

    def set_adapter(active: bool) -> None:
        # Frozen 生成 = 门控上下文缺席(hook 原样返回),无须卸载权重。
        if active:
            load_history_gated_state_dict(wrapped, adapter_state)

    load_history_gated_state_dict(wrapped, adapter_state)

    def generate(messages, *, use_adapter: bool) -> str:
        encoded = runtime.processor.apply_chat_template(
            messages,
            tools=[_TOOL_SPEC],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(runtime.device)
        image_count = int(encoded["image_grid_thw"].shape[0])
        history_count = image_count - 1
        if use_adapter and history_count > 0:
            mask = build_history_token_mask(
                encoded["input_ids"],
                encoded["mm_token_type_ids"],
                encoded["image_grid_thw"],
                history_count,
                merge_size,
            )
            scope = history_adapter_scope(
                HistoryAdapterContext(
                    history_token_mask=mask,
                    history_present=True,
                    image_roles=("history",) * history_count + ("current",),
                )
            )
        else:
            scope = nullcontext()
        prompt_tokens = int(encoded["input_ids"].shape[1])
        tokens = runtime.generation_tokens
        with scope, torch.inference_mode():
            generated = model.generate(
                **{k: v for k, v in encoded.items()},
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokens.tool_call_close_token_id,
                pad_token_id=tokens.pad_token_id,
                suppress_tokens=list(tokens.standard_eos_token_ids),
                num_beams=1,
                num_return_sequences=1,
            )
        return runtime.processor.batch_decode(
            generated[:, prompt_tokens:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]

    results = []
    for position, arm in enumerate(dev_arms, start=1):
        record = records[arm["source"]["dp_id"]]
        screenshots = [
            (args.corpus_root / relpath).read_bytes()
            for relpath in record["image_relpaths"]
        ]
        request = build_agentnet_cr_request(
            instruction=record["instruction"],
            screenshots=screenshots,
            history_actions=record["history"],
            current_step=int(record["step"]),
            recent_budget=0,
            extra_restored_step_ids=tuple(arm["selected_steps"]),
            screen_size=tuple(record["screen_size"]),
        )
        messages = render_agentnet_cr_messages(request)
        entry = {
            "sample_id": arm["sample_id"],
            "arm_slot": arm["arm_slot"],
            "budget": arm["budget"],
            "episode": arm["episode"],
        }
        for policy, use_adapter in (("frozen", False), ("hgkv", True)):
            text = generate(messages, use_adapter=use_adapter)
            parse_ok = True
            equivalent = False
            try:
                parse_gui_owl_osworld_action(
                    text, screen_size=tuple(record["screen_size"])
                )
            except (ValueError, Exception):
                parse_ok = False
            if parse_ok:
                matches = re.findall(
                    r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL
                )
                try:
                    generated_call = json.loads(matches[0])
                    equivalent = tool_call_equivalent(
                        generated_call, record["target_tool_call"]
                    )
                except (IndexError, json.JSONDecodeError):
                    parse_ok = False
            entry[f"{policy}_parse"] = parse_ok
            entry[f"{policy}_equiv"] = equivalent
        results.append(entry)
        if position % 10 == 0:
            print(f"{position}/{len(dev_arms)}", flush=True)

    args.output.write_text(
        json.dumps(
            {
                "schema_version": "causalcache.desktop_parser_validity_v3.v1",
                "checkpoint": str(args.checkpoint),
                "shard": f"{args.shard_index}/{args.shard_count}",
                "rows": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
