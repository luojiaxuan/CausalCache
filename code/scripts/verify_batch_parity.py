#!/usr/bin/env python3
"""微批 vs 逐条:同输入下生成 token 是否一致 + 吞吐对比。

# note (luojiaxuan): 用真实 MobileWorld 截图拼出与线上同形状的 prompt(1-4 张历史
# 图 + 当前图),分别走逐条 generate 与 generate_batch,比对生成 token 与解析动作。
# 判据是动作级一致率——批量 GEMM 归约次序不同,不能期望 bitwise 相同。
"""
import argparse, glob, json, random, time
from pathlib import Path

from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from causalcache.policy.gui_owl_official_runtime import GUIOwlOfficialRuntime


def build_prompt(shots, instruction):
    content = []
    for s in shots[:-1]:
        content.append({"type": "image", "image": str(s)})
    content.append({"type": "text", "text": instruction})
    content.append({"type": "image", "image": str(shots[-1])})
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--snapshot-manifest", required=True)
    ap.add_argument("--screenshot-glob", required=True)
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    shots = sorted(glob.glob(args.screenshot_glob))
    if len(shots) < 40:
        raise SystemExit(f"need more screenshots, got {len(shots)}")
    rng = random.Random(20260731)
    prompts = []
    for i in range(args.count):
        k = 1 + (i % 4)
        picks = rng.sample(shots, k + 1)
        prompts.append(build_prompt(picks, f"Task {i}: complete the pending step."))

    base = GUIOwlV21OfficialToolsRuntime(
        model_dir=Path(args.model_dir),
        expected_snapshot_manifest=Path(args.snapshot_manifest),
        device="cuda:0",
        target_effective_visual_tokens_per_image=2560,
    )
    rt = GUIOwlOfficialRuntime(base)

    t0 = time.perf_counter()
    single = [rt.generate(p) for p in prompts]
    single_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    batched = []
    for i in range(0, len(prompts), args.batch):
        batched.extend(rt.generate_batch(prompts[i:i + args.batch]))
    batch_seconds = time.perf_counter() - t1

    same_tokens = sum(
        a.metadata["generated_token_ids_sha256"] == b.metadata["generated_token_ids_sha256"]
        for a, b in zip(single, batched))
    same_text = sum(a.output_text == b.output_text for a, b in zip(single, batched))
    same_action = sum(
        (a.canonical_action is None) == (b.canonical_action is None)
        and str(a.canonical_action) == str(b.canonical_action)
        for a, b in zip(single, batched))
    report = {
        "count": len(prompts), "batch": args.batch,
        "single_seconds": round(single_seconds, 1),
        "batch_seconds": round(batch_seconds, 1),
        "speedup": round(single_seconds / batch_seconds, 2) if batch_seconds else None,
        "identical_token_ids_pct": round(100 * same_tokens / len(prompts), 1),
        "identical_text_pct": round(100 * same_text / len(prompts), 1),
        "identical_action_pct": round(100 * same_action / len(prompts), 1),
    }
    Path(args.output).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
