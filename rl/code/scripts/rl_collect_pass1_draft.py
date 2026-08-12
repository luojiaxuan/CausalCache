#!/usr/bin/env python3
"""采集 pass-1 draft:recent-2 条件下冻结策略一次前向的动作与 token 级分布。

# note (luojiaxuan): selector 设计简报 §2 的"待生成"资产。部署时这一遍
# 本来就要跑(recent-2 是默认输入),其输出对 selector 免费。落盘:
# 贪心动作(parsed)+ 生成 token 序列每步 top-8 (token_id, logprob) ——
# 分布的紧凑形式,下游架构自行取用。prompt/解码与枚举完全同源。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True)
    p.add_argument("--token-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--max-new-tokens", type=int, default=128)
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
    from rl_oracle_enumerate import parse_tool_call

    want: set[str] = set()
    for f in args.labels:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            subs = [bool(a["c"]) for a in d.get("all", []) if len(a["s"]) == 2]
            if subs and any(subs) and not all(subs) \
                    and (args.token_dir / f"{d['dp_id']}.pt").exists():
                want.add(d["dp_id"])
    print(json.dumps({"winnable": len(want)}), flush=True)

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=args.max_new_tokens)
    model, device, proc = runtime.model, runtime.device, runtime.processor

    done = set()
    if args.output.exists():
        for line in args.output.open(encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["dp_id"])
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
            if dp not in want or dp in done:
                continue
            try:
                s = int(rec["step"])
                images = rec["image_relpaths"]
                if len(images) != s:
                    raise ValueError("image count")
                screen = tuple(int(x) for x in rec["screen_size"])
                steps = [official_step_forms(h, screen_size=screen)
                         for h in rec["history"]]
                cands = [j for j in range(1, s - 1) if steps[j].full_response]
                if len(cands) < 2:
                    raise ValueError("too few candidates")
                root = args.image_root
                recent = cands[-2:]
                needed = [root / images[j] for j in recent] + [root / images[s - 1]]
                if not all(x.exists() for x in needed):
                    raise ValueError("missing images")
                msgs = build_desktop_official_messages(
                    goal=rec["instruction"], steps=steps,
                    shown_events=list(recent),
                    event_images={j: str(root / images[j]) for j in recent},
                    current_image=str(root / images[s - 1]))
                enc = proc.apply_chat_template(
                    msgs, tools=[_TOOL_SPEC], tokenize=True,
                    add_generation_prompt=True, return_dict=True,
                    return_tensors="pt").to(device)
                pt = int(enc["input_ids"].shape[1])
                with torch.inference_mode():
                    gt = runtime.generation_tokens
                    out = model.generate(
                        **enc, do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=gt.tool_call_close_token_id,
                        pad_token_id=gt.pad_token_id,
                        suppress_tokens=list(gt.standard_eos_token_ids),
                        num_beams=1, num_return_sequences=1,
                        return_dict_in_generate=True, output_scores=True)
                seq = out.sequences[0, pt:]
                text = proc.batch_decode(seq.unsqueeze(0),
                                         skip_special_tokens=False)[0]
                topk = []
                for step_scores in out.scores:
                    lp = torch.log_softmax(step_scores[0].float(), dim=-1)
                    v, i = lp.topk(8)
                    topk.append([[int(a), round(float(b), 4)]
                                 for a, b in zip(i, v)])
                sink.write(json.dumps({
                    "dp_id": dp, "recent": recent,
                    "pred": parse_tool_call(text), "text": text[:400],
                    "tokens": [int(x) for x in seq],
                    "topk": topk}, ensure_ascii=False) + "\n")
                sink.flush()
                n += 1
                if n % 50 == 0:
                    print(json.dumps({"shard": args.shard_index, "done": n}),
                          flush=True)
            except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
                k = type(exc).__name__ if not str(exc) else str(exc)[:40]
                skipped[k] = skipped.get(k, 0) + 1
                continue
    print(json.dumps({"shard": args.shard_index, "collected": n,
                      "skipped": skipped, "finished": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
