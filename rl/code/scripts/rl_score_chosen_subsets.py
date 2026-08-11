#!/usr/bin/env python3
"""补测:selector 选出但未被枚举覆盖的 (态, 2帧子集),用冻结策略真跑判分。

# note (luojiaxuan): 部署口径评测首版把"选中的对不在枚举表"的态直接丢掉,
# 每折丢 16-44% —— 被丢的恰是 selector 选了单帧探针池外帧的态,系统性有偏。
# 本脚本把这些 (dp, subset) 补上:prompt 构造 / 解码配置 / 判分逐行镜像
# rl_oracle_enumerate(causalcache 包与建表版本逐字节一致已校验,
# action_correct / parse_tool_call 函数级 md5 一致已校验),批 1 贪心解码。
# 输出 {dp_id, s, c, p},归约时并回部署表。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True,
                   help="JSONL {dp_id, subset:[i,j]},重复行自动去重")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--tolerance", type=float, default=25.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()

    import sys
    import torch

    from causalcache.agentnet_desktop_official import (
        build_desktop_official_messages,
        official_step_forms,
    )
    from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rl_oracle_enumerate import action_correct, parse_tool_call

    want: dict[str, set[tuple[int, ...]]] = {}
    for line in args.pairs.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        want.setdefault(d["dp_id"], set()).add(tuple(sorted(d["subset"])))
    total = sum(len(v) for v in want.values())
    print(json.dumps({"states": len(want), "pairs": total}), flush=True)

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=args.max_new_tokens)

    def gen(msgs: list[dict[str, Any]]) -> str:
        enc = runtime.processor.apply_chat_template(
            msgs, tools=[_TOOL_SPEC], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(runtime.device)
        pt = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            gt = runtime.generation_tokens
            out = runtime.model.generate(
                **enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                eos_token_id=gt.tool_call_close_token_id,
                pad_token_id=gt.pad_token_id,
                suppress_tokens=list(gt.standard_eos_token_ids),
                num_beams=1, num_return_sequences=1)
        return runtime.processor.batch_decode(
            out[:, pt:], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)[0]

    done: set[tuple[str, tuple[int, ...]]] = set()
    if args.output.exists():
        for line in args.output.open(encoding="utf-8"):
            line = line.strip()
            if line:
                d = json.loads(line)
                done.add((d["dp_id"], tuple(d["s"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped: dict[str, int] = {}
    with args.output.open("a", encoding="utf-8") as sink:
        for lineno, line in enumerate(args.manifest.open(encoding="utf-8")):
            if lineno % args.shard_count != args.shard_index:
                continue
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            dp = rec["dp_id"]
            todo = [t for t in sorted(want.get(dp, ()))
                    if (dp, t) not in done]
            if not todo:
                continue
            try:
                s = int(rec["step"])
                images = rec["image_relpaths"]
                if len(images) != s:
                    raise ValueError("image count")
                screen = tuple(int(x) for x in rec["screen_size"])
                steps = [official_step_forms(h, screen_size=screen)
                         for h in rec["history"]]
                gold = rec["target_tool_call"].get("arguments", {})
                root = args.image_root
                current = str(root / images[s - 1])
                for t in todo:
                    ev = {j: str(root / images[j]) for j in t}
                    if not all(Path(x).exists() for x in ev.values()):
                        raise ValueError("missing images")
                    pred = parse_tool_call(gen(build_desktop_official_messages(
                        goal=rec["instruction"], steps=steps,
                        shown_events=list(t), event_images=ev,
                        current_image=current)))
                    sink.write(json.dumps({
                        "dp_id": dp, "s": list(t),
                        "c": action_correct(pred, gold, tolerance=args.tolerance),
                        "p": pred}, ensure_ascii=False) + "\n")
                    sink.flush()
                    n += 1
                if n and n % 50 < len(todo):
                    print(json.dumps({"shard": args.shard_index, "scored": n},
                                     ensure_ascii=False), flush=True)
            except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
                k = type(exc).__name__ if not str(exc) else str(exc)[:40]
                skipped[k] = skipped.get(k, 0) + 1
                continue
    print(json.dumps({"shard": args.shard_index, "scored": n,
                      "skipped": skipped, "finished": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
