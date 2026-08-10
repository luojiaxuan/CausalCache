#!/usr/bin/env python3
"""#26 评测:各臂在留出 hopeless 态上,{B=0, recent-2} 双 prompt 步级正确率。

# note (luojiaxuan): **B=0 对照臂是本实验的命门。** 在 hopeless 态上做 gold
# 监督,正确率上涨可能全部来自"任务 SFT"(模型更会做这类桌面步骤了),
# 与"历史变得可读"无关。判别式:
#     记忆可读性增益(arm) = [acc_arm(recent2) − acc_frozen(recent2)]
#                          − [acc_arm(B0) − acc_frozen(B0)]
# 只有它显著为正,才说明适配把 recent-2 里**已有但读不出**的信息解锁了。
# 每态两种 prompt 由同一臂同一权重生成,原始预测落盘(容差是分析旋钮)。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None,
                   help="不给 = 冻结基线臂")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--tolerance", type=float, default=25.0)
    p.add_argument("--limit-states", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260809)
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "scripts"))
    from train_success_sft_lora import inject_lora, load_lora_state_dict

    hopeless: list[str] = []
    for f in args.labels:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "oracle_correct" in d and not d["b0_correct"] and not d["oracle_correct"]:
                hopeless.append(d["dp_id"])
    hopeless = sorted(set(hopeless))
    random.Random(args.seed).shuffle(hopeless)
    nh = int(len(hopeless) * args.holdout_frac)
    hold = set(hopeless[:nh])
    print(json.dumps({"holdout": len(hold)}, ensure_ascii=False), flush=True)

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=128)
    model, device, proc = runtime.model, runtime.device, runtime.processor

    arm_name = "frozen"
    if args.adapter is not None:
        bundle = torch.load(args.adapter, map_location="cpu", weights_only=False)
        arm_name = bundle["arm"]
        if arm_name == "ungated_kv":
            wrapped = inject_lora(model, rank=bundle["rank"], alpha=bundle["alpha"],
                                  target_modules=("k_proj", "v_proj"),
                                  torch=torch, last_layer_count=8)
        elif arm_name == "full_lora":
            wrapped = inject_lora(model, rank=bundle["rank"], alpha=bundle["alpha"],
                                  target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                                  torch=torch, last_layer_count=None)
        else:
            raise SystemExit(f"未知 arm {arm_name!r}")
        load_lora_state_dict(wrapped, bundle["state"])
        print(json.dumps({"loaded_arm": arm_name,
                          "modules": len(wrapped)}, ensure_ascii=False), flush=True)

    def gen(msgs):
        enc = proc.apply_chat_template(
            msgs, tools=[_TOOL_SPEC], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(device)
        pt = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            gt = runtime.generation_tokens
            o = model.generate(**enc, do_sample=False, max_new_tokens=128,
                               eos_token_id=gt.tool_call_close_token_id,
                               pad_token_id=gt.pad_token_id,
                               suppress_tokens=list(gt.standard_eos_token_ids),
                               num_beams=1, num_return_sequences=1)
        return parse_tool_call(proc.batch_decode(o[:, pt:], skip_special_tokens=False)[0])

    done = set()
    if args.output.exists():
        for l in args.output.open(encoding="utf-8"):
            l = l.strip()
            if l:
                done.add(json.loads(l)["dp_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped: dict[str, int] = {}
    with args.output.open("a", encoding="utf-8") as sink:
        for line in args.manifest.open(encoding="utf-8"):
            if args.limit_states and n >= args.limit_states:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            dp = rec["dp_id"]
            if dp not in hold or dp in done:
                continue
            try:
                s = int(rec["step"])
                images = rec["image_relpaths"]
                if len(images) != s:
                    raise ValueError("count")
                screen = tuple(int(x) for x in rec["screen_size"])
                steps = [official_step_forms(h, screen_size=screen)
                         for h in rec["history"]]
                cands = [j for j in range(1, s - 1) if steps[j].full_response]
                root = args.image_root
                recent = cands[-2:] if cands else []
                needed = [root / images[j] for j in recent] + [root / images[s - 1]]
                if not all(x.exists() for x in needed):
                    raise ValueError("missing")
                gold = rec["target_tool_call"].get("arguments", {})
                ev = {j: str(root / images[j]) for j in recent}
                cur = str(root / images[s - 1])
                out_row = {"dp_id": dp, "arm": arm_name, "preds": {}}
                for tag, shown in (("b0", []), ("recent2", recent)):
                    pred = gen(build_desktop_official_messages(
                        goal=rec["instruction"], steps=steps,
                        shown_events=list(shown),
                        event_images={j: ev[j] for j in shown},
                        current_image=cur))
                    out_row[f"{tag}_correct"] = action_correct(
                        pred, gold, tolerance=args.tolerance)
                    out_row["preds"][tag] = pred
                out_row["gold"] = gold
                sink.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                sink.flush()
                n += 1
                if n % 50 == 0:
                    print(json.dumps({"arm": arm_name, "done": n},
                                     ensure_ascii=False), flush=True)
            except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
                k = type(exc).__name__ if not str(exc) else str(exc)[:40]
                skipped[k] = skipped.get(k, 0) + 1
                continue
    print(json.dumps({"arm": arm_name, "evaluated": n, "skipped": skipped,
                      "finished": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
