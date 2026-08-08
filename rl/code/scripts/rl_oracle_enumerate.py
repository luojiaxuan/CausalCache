#!/usr/bin/env python3
"""枚举 oracle 选帧:冻结 policy,只换历史帧,量步级动作正确率的头寸。

# note (luojiaxuan): 契约见 rl/docs/oracle_selection_contract.md。回答一个问题:
# **冻结 policy 不动,只把历史帧换成更好的,步级正确率能提多少?**
# 三个产出:① 强基线(选择本身值多少);② 上界标尺(后续方法除以它得捕获率,
# v1 时代捕获了 6.5% 却不自知就是因为没有这把尺);③ selector 的监督标签。
#
# 口径(全部沿用已冻结的组件,不另写一份):
#   * prompt 由 build_desktop_official_messages 构造 —— 与语料、与部署同源;
#   * 历史步形态由 official_step_forms 解析(内部复用 parse_agentnet_step);
#   * 事件 j 的图 = image_relpaths[j],当前屏 = image_relpaths[s-1],
#     候选事件 ∈ [1, s-2](事件 s-1 的 post 帧就是当前屏,builder 禁止重复);
#   * 正确率 = 动作类型匹配 且 坐标 L2 ≤ tolerance(归一 [0,999] 空间)。
#     该指标**不需要可导**——selector 的监督是"哪个子集得分最高",
#     标签在这里算完,容差问题不进损失函数。
#
# 四臂(随机臂是防自欺的硬要求:若 oracle ≈ 随机,说明策略可能根本没在用
# 图像内容 —— 但也可能是它读不懂这种插图 prompt 格式,两者本实验分辨不了,
# 那种情况转联合训练分支,不作终结结论)。
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """从 <tool_call>{...}</tool_call> 里取出 arguments。"""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    args = obj.get("arguments")
    return args if isinstance(args, dict) else None


def _coord(args: dict[str, Any]) -> tuple[float, float] | None:
    for key in ("coordinate", "point", "start_coordinate"):
        v = args.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                return float(v[0]), float(v[1])
            except (TypeError, ValueError):
                return None
    return None


def action_correct(pred: dict[str, Any] | None, gold: dict[str, Any],
                   *, tolerance: float) -> bool:
    """步级正确:动作类型匹配 + 坐标在容差内 + 文本/按键精确匹配。"""
    if pred is None:
        return False
    if str(pred.get("action", "")).strip() != str(gold.get("action", "")).strip():
        return False
    gc, pc = _coord(gold), _coord(pred)
    if gc is not None:
        if pc is None:
            return False
        if ((gc[0] - pc[0]) ** 2 + (gc[1] - pc[1]) ** 2) ** 0.5 > tolerance:
            return False
    for key in ("text", "keys"):
        if key in gold and gold.get(key) != pred.get(key):
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True,
                   help="agentnet_screening_manifest_*.jsonl")
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--tolerance", type=float, default=25.0)
    p.add_argument("--limit-states", type=int, default=30)
    p.add_argument("--max-candidates", type=int, default=30,
                   help="候选事件超过此数的 state 跳过(控成本;会计数披露)")
    p.add_argument("--mode", choices=["full", "pruned"], default="full",
                   help="full=全枚举 C(n,B);pruned=单帧 top-k 后配对。"
                        "校验期用 full——剪枝结果可从全枚举离线推导,零额外成本")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--skip-easy", action="store_true",
                   help="先只跑 B=0;若已正确则记为 easy 并跳过全枚举。"
                        "头寸只存在于 B=0 出错的 state,easy 态上花 190 次前向"
                        "是纯浪费。**被跳过的 state 仍写一行记录**,base rate "
                        "不丢,分母仍然干净")
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260808)
    # 枚举无同步点,按 state 天然可并行:每个分片只处理 行号 % count == index
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()

    import torch

    from causalcache.agentnet_desktop_official import (
        build_desktop_official_messages,
        official_step_forms,
    )
    # note (luojiaxuan): _TOOL_SPEC 是模块私有,但这里必须用同一份 —— 换一份
    # tool spec 就换了 prompt,与语料/部署口径脱钩。下面的 encode+generate 是
    # generate_raw 的逐行镜像(runtime 只暴露 request 级接口,吃不了自建 messages),
    # 解码配置一个不改:do_sample=False、同 eos/pad/suppress、num_beams=1。
    from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    def generate_from_messages(messages: list[dict[str, Any]]) -> str:
        encoded = runtime.processor.apply_chat_template(
            messages, tools=[_TOOL_SPEC], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(runtime.device)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            tok = runtime.generation_tokens
            out = runtime.model.generate(
                **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                eos_token_id=tok.tool_call_close_token_id,
                pad_token_id=tok.pad_token_id,
                suppress_tokens=list(tok.standard_eos_token_ids),
                num_beams=1, num_return_sequences=1,
            )
        return runtime.processor.batch_decode(
            out[:, prompt_tokens:], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)[0]
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():          # 断点续跑
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["dp_id"])

    skipped: dict[str, int] = {}
    processed = 0
    with args.output.open("a", encoding="utf-8") as sink:
        for lineno, line in enumerate(args.manifest.open(encoding="utf-8")):
            if processed >= args.limit_states:
                break
            if lineno % args.shard_count != args.shard_index:
                continue
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["dp_id"] in done:
                continue
            s = int(rec["step"])
            images = rec["image_relpaths"]
            if len(images) != s:
                skipped["image_count_mismatch"] = skipped.get("image_count_mismatch", 0) + 1
                continue
            try:
                screen = tuple(int(x) for x in rec["screen_size"])
                steps = [official_step_forms(h, screen_size=screen)
                         for h in rec["history"]]
            except (ValueError, KeyError):
                skipped["unparseable_history"] = skipped.get("unparseable_history", 0) + 1
                continue
            if len(steps) != s - 1:
                skipped["history_len_mismatch"] = skipped.get("history_len_mismatch", 0) + 1
                continue
            # 候选事件:[1, s-2] 且该步能作保留轮(builder 要求 full_response)。
            # note (luojiaxuan): 事件 j 的保留轮是**步骤 j+1**,其形态是 steps[j]
            # (0-based),不是 steps[j-1]——builder 内部就是这么查的。写错会在
            # 枚举到含该事件的子集时才炸(如 triple_click 无官方形态)。
            cands = [j for j in range(1, s - 1) if steps[j].full_response]
            if len(cands) < args.budget:
                skipped["too_few_candidates"] = skipped.get("too_few_candidates", 0) + 1
                continue
            if len(cands) > args.max_candidates:
                skipped["too_many_candidates"] = skipped.get("too_many_candidates", 0) + 1
                continue

            gold = rec["target_tool_call"].get("arguments", {})
            # note (luojiaxuan): 必须是 str —— processor 的 fetch_images 只吃
            # 字符串路径/URL/PIL,给 Path 会 TypeError。
            root = args.image_root
            # 图片存在性预检:manifest 有 6003 态,但落盘图片只覆盖当初建语料
            # 用到的子集,深处的 state 会缺图。**缺任何一张就整态跳过并计数**——
            # 只丢掉缺的候选会让候选池不完整,从而系统性低估 oracle 头寸。
            needed = [root / images[j] for j in cands] + [root / images[s - 1]]
            if not all(p.exists() for p in needed):
                skipped["missing_images"] = skipped.get("missing_images", 0) + 1
                continue
            event_images = {j: str(root / images[j]) for j in cands}
            current = str(root / images[s - 1])

            def evaluate(subset: tuple[int, ...]) -> dict[str, Any]:
                msgs = build_desktop_official_messages(
                    goal=rec["instruction"],
                    steps=steps,
                    shown_events=list(subset),
                    event_images={j: event_images[j] for j in subset},
                    current_image=current,
                )
                pred = parse_tool_call(generate_from_messages(msgs))
                return {"subset": list(subset),
                        "correct": action_correct(pred, gold,
                                                  tolerance=args.tolerance),
                        "pred_action": (pred or {}).get("action")}

            # 整态兜底:任何一步炸了只记账跳过,不让单态拖垮整批
            try:
                if args.skip_easy:
                    b0_first = evaluate(())
                    if b0_first["correct"]:
                        sink.write(json.dumps({
                            "dp_id": rec["dp_id"], "step": s,
                            "n_candidates": len(cands), "mode": "screen",
                            "budget": args.budget, "gold_action": gold.get("action"),
                            "b0_correct": True, "easy": True,
                        }, ensure_ascii=False) + "\n")
                        sink.flush()
                        processed += 1
                        continue

                if args.mode == "full":
                    subsets = list(itertools.combinations(cands, args.budget))
                else:
                    # note (luojiaxuan): 单帧探针必须用 **B-1 张最近帧** 当搭档,
                    # 保持 prompt 形状与最终子集一致(B 张图)。此前硬编码成
                    # "候选 + 最近 1 帧"的两图 prompt —— B=2 时恰好正确,
                    # 但 B=4 时探针是 2 图、目标是 4 图,形状不一致会让排序失真。
                    partners = tuple(cands[-(args.budget - 1):]) if args.budget > 1 else ()
                    scored = []
                    for c in cands:
                        if c in partners:
                            continue
                        probe = tuple(sorted((c, *partners)))
                        scored.append((int(evaluate(probe)["correct"]), c))
                    scored.sort(key=lambda t: (-t[0], -t[1]))
                    top = sorted({c for _, c in scored[: args.top_k]} | set(partners))
                    subsets = list(itertools.combinations(top, args.budget))

                # 单个 state 出问题不许拖垮整批(跑批贵,续跑成本高)
                try:
                    results = [evaluate(t) for t in subsets]
                except (ValueError, KeyError, OSError) as exc:
                    key = f"eval_error:{type(exc).__name__}"
                    skipped[key] = skipped.get(key, 0) + 1
                    continue
                recent = tuple(cands[-args.budget:])
                rand = tuple(sorted(rng.sample(cands, args.budget)))
                by_subset = {tuple(r["subset"]): r for r in results}
                for extra in (recent, rand):
                    if extra not in by_subset:
                        by_subset[extra] = evaluate(extra)
                b0 = evaluate(())

                sink.write(json.dumps({
                    "dp_id": rec["dp_id"], "step": s, "n_candidates": len(cands),
                    "mode": args.mode, "budget": args.budget,
                    "gold_action": gold.get("action"),
                    "oracle_correct": any(r["correct"] for r in results),
                    "oracle_subsets": [r["subset"] for r in results if r["correct"]][:8],
                    "recent_correct": by_subset[recent]["correct"],
                    "random_correct": by_subset[rand]["correct"],
                    "b0_correct": b0["correct"],
                    "n_subsets_scored": len(by_subset),
                    "n_subsets_correct": sum(1 for r in by_subset.values() if r["correct"]),
                    "all": [{"s": list(k), "c": v["correct"]} for k, v in by_subset.items()],
                }, ensure_ascii=False) + "\n")
                sink.flush()
                processed += 1

            except (ValueError, KeyError, OSError, TypeError) as exc:
                key = f"state_error:{type(exc).__name__}"
                skipped[key] = skipped.get(key, 0) + 1
                continue
    print(json.dumps({"processed": processed, "skipped": skipped},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
