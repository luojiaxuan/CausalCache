#!/usr/bin/env python3
"""Clean four-way comparison on the frozen model (no dangling text references).

# note (luojiaxuan): 上一版 no_history 基线只删图、文本仍写着 "Historical screenshot
# from StepN",给了模型自相矛盾的 prompt,量级读数不可信。此处用官方 builder 生成
# 真正的 K=0 prompt(文本干净),四条一起量:
#   K0(无历史图) / native_recent{K}(连续最近) / sparse_correct / sparse_irrelevant
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repository-root", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--groups", type=int, default=60)
    args = ap.parse_args()
    import torch
    from scripts.run_exploratory_closed_loop_episode import EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
    from scripts.train_success_sft_lora import encode_sample, mean_target_logprob
    from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
    from causalcache.policy.gui_owl_official import build_official_messages

    rt = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json",
        device=args.device, target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE)
    rt.model.eval()
    rows = [json.loads(l) for l in (args.dataset_root / "samples.jsonl").open()]
    groups: dict[str, dict] = {}
    for r in rows:
        groups.setdefault(r["pair_group"], {})[r["variant"]] = r

    def lp(sample):
        enc = encode_sample(rt, sample, dataset_root=args.dataset_root, torch=torch)
        if enc is None: return None
        with torch.no_grad():
            return float(mean_target_logprob(rt.model, enc, torch=torch))

    def k0_sample(base):
        """真正的 K=0:官方 builder,零历史图,文本干净。"""
        msgs = build_official_messages(
            goal=base["instruction"], past_action_texts=base["action_texts"],
            recent_images=[], current_image={"__path__": base["current_image"]})
        out = []
        for m in msgs:
            content = []
            for p in m["content"]:
                if p.get("type") == "image":
                    content.append({"type": "image", "path": p["image"]["__path__"]})
                else:
                    content.append(dict(p))
            out.append({"role": m["role"], "content": content})
        s = dict(base); s["messages"] = out; s["selected_steps"] = []
        return s

    agg = {"K0": [], "native": [], "correct": [], "irrelevant": []}
    used = 0
    for g in groups.values():
        c = g.get("sparse_correct"); ir = g.get("sparse_irrelevant")
        nat = next((v for k, v in g.items() if k.startswith("native_recent")), None)
        if not (c and ir and nat): continue
        vals = {"K0": lp(k0_sample(c)), "native": lp(nat), "correct": lp(c), "irrelevant": lp(ir)}
        if any(v is None for v in vals.values()): continue
        for k, v in vals.items(): agg[k].append(v)
        used += 1
        if used >= args.groups: break
    base = st.mean(agg["K0"])
    print(json.dumps({
        "groups": used,
        "K0_absolute": round(base, 5),
        "native_recentK_minus_K0": round(st.mean(agg["native"]) - base, 5),
        "sparse_correct_minus_K0": round(st.mean(agg["correct"]) - base, 5),
        "sparse_irrelevant_minus_K0": round(st.mean(agg["irrelevant"]) - base, 5),
        "sparse_correct_minus_native": round(st.mean(agg["correct"]) - st.mean(agg["native"]), 5),
        "sparse_correct_minus_irrelevant": round(st.mean(agg["correct"]) - st.mean(agg["irrelevant"]), 5),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
