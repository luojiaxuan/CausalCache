#!/usr/bin/env python3
"""同口径判据:训练后的 selector vs 初始头 vs recent-2 vs 随机-2 的步级正确率。

# note (luojiaxuan): v3 训练跑完 3000 态后,位移 1.56、熵 3.94→2.32,
# 但 `mean_reward_logp` 是**累计均值**且跨 state 可比性差,既不能证明学到了、
# 也不能证明没学到。要判定只能这样:**同一批 state 上,用同一个正确率指标
# (动作类型匹配 + 坐标 ≤tolerance)比四个臂**。
#
#   learned   训练后的头,τ=0 取 top-B
#   init      随机初始化的头,τ=0 取 top-B   ← 学习是否有效的直接对照
#   recent    最近 B 帧(部署默认)
#   random    随机 B 帧
#
# **只在 B=0 出错的 state 上比才有意义**(选帧在 easy 态上无杠杆),
# 所以默认 --only-hard:先跑 B=0,已正确则跳过并计数。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--learned-head", type=Path, required=True)
    p.add_argument("--init-head", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--tolerance", type=float, default=25.0)
    p.add_argument("--limit-states", type=int, default=300)
    p.add_argument("--max-candidates", type=int, default=30)
    p.add_argument("--skip-first", type=int, default=4000,
                   help="跳过前 N 行:训练用了 manifest 前 3000 态,评测必须"
                        "落在**没训过**的行上,否则测的是记忆不是泛化")
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--index-visual-tokens", type=int, default=144)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260808)
    args = p.parse_args()

    import torch
    from transformers import AutoProcessor

    from causalcache.agentnet_desktop_official import (
        build_desktop_official_messages,
        official_step_forms,
    )
    from causalcache.osworld_gui_owl import (
        _TOOL_SPEC,
        VISION_PATCH_SIZE,
        VISION_SPATIAL_MERGE_SIZE,
        GUIOwlOSWorldRuntime,
    )
    from causalcache_rl.index_features import index_features
    from rl_oracle_enumerate import action_correct, parse_tool_call

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    model, device = runtime.model, runtime.device
    _px = args.index_visual_tokens * (VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE) ** 2
    index_processor = AutoProcessor.from_pretrained(
        args.model_dir, min_pixels=_px, max_pixels=_px, local_files_only=True)

    def load_head(path: Path):
        b = torch.load(path, map_location="cpu", weights_only=False)
        h = torch.nn.Linear(int(b["hidden_size"]), 1)
        h.load_state_dict(b["model_state"])
        return h.to(device).eval()

    heads = {"learned": load_head(args.learned_head),
             "init": load_head(args.init_head)}
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def gen(messages) -> dict[str, Any] | None:
        enc = runtime.processor.apply_chat_template(
            messages, tools=[_TOOL_SPEC], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(device)
        n = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            tok = runtime.generation_tokens
            out = model.generate(
                **enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                eos_token_id=tok.tool_call_close_token_id,
                pad_token_id=tok.pad_token_id,
                suppress_tokens=list(tok.standard_eos_token_ids),
                num_beams=1, num_return_sequences=1)
        return parse_tool_call(runtime.processor.batch_decode(
            out[:, n:], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)[0])

    tally = {k: 0 for k in ("learned", "init", "recent", "random")}
    skipped: dict[str, int] = {}
    n_eval = 0
    with args.output.open("a", encoding="utf-8") as sink:
        for lineno, line in enumerate(args.manifest.open(encoding="utf-8")):
            if n_eval >= args.limit_states:
                break
            if lineno < args.skip_first or not line.strip():
                continue
            rec = json.loads(line)
            try:
                s = int(rec["step"])
                images = rec["image_relpaths"]
                if len(images) != s:
                    raise ValueError("image count")
                screen = tuple(int(x) for x in rec["screen_size"])
                steps = [official_step_forms(h, screen_size=screen)
                         for h in rec["history"]]
                cands = [j for j in range(1, s - 1) if steps[j].full_response]
                if not (args.budget <= len(cands) <= args.max_candidates):
                    raise ValueError("candidates")
                paths = [args.image_root / images[j] for j in cands]
                cur = args.image_root / images[s - 1]
                if not all(q.exists() for q in paths + [cur]):
                    raise ValueError("missing images")
                ev = {j: str(args.image_root / images[j]) for j in cands}
                gold = rec["target_tool_call"].get("arguments", {})

                def run(subset) -> bool:
                    msgs = build_desktop_official_messages(
                        goal=rec["instruction"], steps=steps,
                        shown_events=sorted(subset),
                        event_images={j: ev[j] for j in sorted(subset)},
                        current_image=str(cur))
                    return action_correct(gen(msgs), gold,
                                          tolerance=args.tolerance)

                if run(()):                       # B=0 已对 → 无杠杆,跳过
                    skipped["easy"] = skipped.get("easy", 0) + 1
                    continue
                feats = index_features(rec, steps, cands, ev, str(cur),
                                       model=model, processor=index_processor,
                                       device=device, torch=torch,
                                       build=build_desktop_official_messages,
                                       tool_spec=_TOOL_SPEC)
                picks: dict[str, tuple[int, ...]] = {}
                with torch.no_grad():
                    for name, head in heads.items():
                        sc = head(feats).squeeze(-1)
                        idx = torch.topk(sc, args.budget).indices.tolist()
                        picks[name] = tuple(sorted(cands[i] for i in idx))
                picks["recent"] = tuple(cands[-args.budget:])
                picks["random"] = tuple(sorted(rng.sample(cands, args.budget)))
                res = {k: run(v) for k, v in picks.items()}
                for k, v in res.items():
                    tally[k] += int(v)
                n_eval += 1
                sink.write(json.dumps({
                    "dp_id": rec["dp_id"], "step": s,
                    "n_candidates": len(cands),
                    "picks": {k: list(v) for k, v in picks.items()},
                    "correct": res,
                }, ensure_ascii=False) + "\n")
                sink.flush()
            except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
                key = f"err:{type(exc).__name__}"
                skipped[key] = skipped.get(key, 0) + 1
                continue

    print(json.dumps({
        "hard_states_evaluated": n_eval, "skipped": skipped,
        "accuracy": {k: (v, round(v / max(n_eval, 1), 4)) for k, v in tally.items()},
        "note": "只在 B=0 出错的 state 上评;learned vs init 是学习是否有效的直接对照",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
