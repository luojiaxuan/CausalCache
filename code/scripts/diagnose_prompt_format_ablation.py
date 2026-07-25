#!/usr/bin/env python3
"""Isolate selection from prompt format on the frozen model.

# note (luojiaxuan): native_recent{K} 走官方多轮格式,sparse 走单轮格式,两者
# 相差 0.16 nats;这可能全部来自格式而非选点。此处用**同一个 sparse builder**
# 再渲染一遍连续最近 K 步(same_format_recent),把格式变量摁住:
#   sparse_correct - same_format_recent  才是真正的"选点收益"。
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
    from causalcache.policy.gui_owl_sparse_history import build_sparse_history_messages

    rt = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json",
        device=args.device, target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE)
    rt.model.eval()
    rows = [json.loads(l) for l in (args.dataset_root / "samples.jsonl").open()]
    groups: dict[str, dict] = {}
    for r in rows:
        groups.setdefault(r["pair_group"], {})[r["variant"]] = r

    def lp(s):
        enc = encode_sample(rt, s, dataset_root=args.dataset_root, torch=torch)
        if enc is None: return None
        with torch.no_grad():
            return float(mean_target_logprob(rt.model, enc, torch=torch))

    def same_format_recent(base, K):
        cur = base["decision_step"]
        steps = list(range(max(1, cur - K), cur))
        ep = base["episode"]
        msgs = build_sparse_history_messages(
            instruction=base["instruction"], action_texts=base["action_texts"],
            selected_steps=steps,
            selected_images=[{"__path__": f"{base['selected_images'][0].rsplit('/',2)[0]}/{ep}/obs-{s-1:03d}.png"} for s in steps],
            current_step=cur, current_image={"__path__": base["current_image"]})
        out = []
        for m in msgs:
            content = []
            for p in m["content"]:
                content.append({"type": "image", "path": p["image"]["__path__"]}
                               if p.get("type") == "image" else dict(p))
            out.append({"role": m["role"], "content": content})
        s = dict(base); s["messages"] = out; s["selected_steps"] = steps
        return s

    agg = {"sparse": [], "same_fmt_recent": [], "official_recent": []}
    used = 0
    for g in groups.values():
        c = g.get("sparse_correct")
        nat = next((v for k, v in g.items() if k.startswith("native_recent")), None)
        if not (c and nat): continue
        K = len(c["selected_steps"])
        try:
            sfr = same_format_recent(c, K)
        except (ValueError, KeyError, IndexError):
            continue
        vals = {"sparse": lp(c), "same_fmt_recent": lp(sfr), "official_recent": lp(nat)}
        if any(v is None for v in vals.values()): continue
        for k, v in vals.items(): agg[k].append(v)
        used += 1
        if used >= args.groups: break
    print(json.dumps({
        "groups": used,
        "sparse_minus_same_format_recent": round(st.mean(agg["sparse"]) - st.mean(agg["same_fmt_recent"]), 5),
        "sparse_minus_official_recent": round(st.mean(agg["sparse"]) - st.mean(agg["official_recent"]), 5),
        "same_format_recent_minus_official": round(st.mean(agg["same_fmt_recent"]) - st.mean(agg["official_recent"]), 5),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
