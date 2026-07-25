#!/usr/bin/env python3
"""Does the model actually see the history images? Frozen-model ablation.

# note (luojiaxuan): correct/shuffled/irrelevant/duplicate 四变体的**文本完全相同**,
# 只有像素不同。若四者 logprob 也相同,可能是(a)模型对历史内容不敏感(已知性质),
# 也可能是(b)图像根本没进前向(bug)。加一条"完全不给历史图"的对照即可区分:
#   correct ≈ irrelevant 但都 ≫ no_history  -> 图进了,内容没被读(性质,不是 bug)
#   correct ≈ irrelevant ≈ no_history       -> 图没起作用(bug)
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repository-root", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--groups", type=int, default=40)
    args = ap.parse_args()
    import torch
    from scripts.run_exploratory_closed_loop_episode import EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
    from scripts.train_success_sft_lora import encode_sample, mean_target_logprob
    from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime

    rt = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json",
        device=args.device, target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE)
    rt.model.eval()
    rows = [json.loads(l) for l in (args.dataset_root / "samples.jsonl").open()]
    by_group: dict[str, dict] = {}
    for r in rows:
        by_group.setdefault(r["pair_group"], {})[r["variant"]] = r
    groups = [g for g in by_group.values() if "sparse_correct" in g and "sparse_irrelevant" in g][: args.groups]

    def lp(sample):
        enc = encode_sample(rt, sample, dataset_root=args.dataset_root, torch=torch)
        if enc is None: return None
        with torch.no_grad():
            return float(mean_target_logprob(rt.model, enc, torch=torch))

    def strip_history(sample):
        """同一 prompt 但删掉全部历史图(保留全部文本与当前图)。"""
        s = json.loads(json.dumps(sample))
        for m in s["messages"]:
            content = m.get("content", [])
            imgs = [i for i, p in enumerate(content) if p.get("type") == "image"]
            for i in reversed(imgs[:-1]):     # 只留最后一张(当前观测)
                content.pop(i)
        s["selected_steps"] = []
        return s

    agg = {"correct": [], "irrelevant": [], "shuffled": [], "duplicate": [], "no_history": []}
    pixel_same = 0
    for g in groups:
        c, ir = g["sparse_correct"], g["sparse_irrelevant"]
        # 像素是否真的不同
        cp = [p["path"] for m in c["messages"] for p in m["content"] if p.get("type") == "image"][:-1]
        ip = [p["path"] for m in ir["messages"] for p in m["content"] if p.get("type") == "image"][:-1]
        def h(paths):
            return [hashlib.sha256((args.dataset_root / p).read_bytes()).hexdigest()[:12] for p in paths]
        if cp and h(cp) == h(ip): pixel_same += 1
        for key, samp in (("correct", c), ("irrelevant", ir),
                          ("shuffled", g.get("sparse_step_shuffled")),
                          ("duplicate", g.get("sparse_duplicate"))):
            if samp is None: continue
            v = lp(samp)
            if v is not None: agg[key].append(v)
        v0 = lp(strip_history(c))
        if v0 is not None: agg["no_history"].append(v0)
    import statistics as st
    n = len(agg["correct"])
    base = st.mean(agg["no_history"]) if agg["no_history"] else 0.0
    print(json.dumps({
        "groups": n,
        "correct_minus_no_history": round(st.mean(agg["correct"]) - base, 5),
        "irrelevant_minus_no_history": round(st.mean(agg["irrelevant"]) - base, 5),
        "shuffled_minus_no_history": round(st.mean(agg["shuffled"]) - base, 5) if agg["shuffled"] else None,
        "duplicate_minus_no_history": round(st.mean(agg["duplicate"]) - base, 5) if agg["duplicate"] else None,
        "correct_minus_irrelevant": round(st.mean(agg["correct"]) - st.mean(agg["irrelevant"]), 5),
        "groups_with_identical_history_pixels": pixel_same,
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
