#!/usr/bin/env python3
"""P1 部署口径评测:LoRA 选帧器在留出态生成选择 → 查枚举表 vs recent-2。

# note (luojiaxuan): 与 2.6/2.7 部署评测同一把尺:同总体(winnable 1263)、
# 同划分(seed 洗牌 + fold 切片)、同判据(查 B=2 枚举表,选中对不在表内
# 落盘待补测,配对 McNemar + bootstrap CI + MultiApp 分层)。
# 生成解析失败 → 回退 recent-2(部署真实行为)并计数;回退不算 moved。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True)
    p.add_argument("--token-dir", type=Path, required=True)
    p.add_argument("--b1-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--domain-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--limit-states", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260809)
    args = p.parse_args()

    import random
    import sys
    import torch

    from causalcache.agentnet_desktop_official import official_step_forms
    from causalcache.osworld_gui_owl import GUIOwlOSWorldRuntime
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "scripts"))
    except IndexError:
        pass  # 散件部署时靠 PYTHONPATH 提供 code/scripts
    from train_success_sft_lora import inject_lora, load_lora_state_dict
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_frame_selector_p1 import (
        build_selector_messages,
        build_selector_messages_images,
        load_b1,
        load_winnable,
    )

    lab = load_winnable(args.labels, args.token_dir)
    table: dict[str, dict] = {}
    recent_ref: dict[str, bool] = {}
    for f in args.labels:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["dp_id"] not in lab:
                continue
            table[d["dp_id"]] = {frozenset(a["s"]): bool(a["c"])
                                 for a in d.get("all", []) if len(a["s"]) == 2}
            if "recent_correct" in d:
                recent_ref[d["dp_id"]] = bool(d["recent_correct"])

    ids = sorted(lab)
    random.Random(args.seed).shuffle(ids)
    nh = int(len(ids) * args.holdout_frac)
    hold = ids[args.fold * nh:(args.fold + 1) * nh]
    b1 = load_b1(args.b1_dir, set(hold))  # images 模式下仅闲置
    dom = json.loads(args.domain_json.read_text())
    t2d = {}
    for line in args.manifest.open(encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            t2d[d["dp_id"]] = dom.get(d.get("task_id"), "?")
    print(json.dumps({"holdout": len(hold), "fold": args.fold}), flush=True)

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=16)
    model, device = runtime.model, runtime.device
    # note (luojiaxuan): 与训练器同款 —— runtime processor 的 min=max 钉死会把
    # 缩略图放大回 2560 token,选帧遍必须自建 min 放开的 processor。
    from causalcache.osworld_gui_owl import (
        VISION_PATCH_SIZE,
        VISION_SPATIAL_MERGE_SIZE,
    )
    from transformers import AutoProcessor
    unit = (VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE) ** 2
    proc = AutoProcessor.from_pretrained(
        str(args.model_dir), min_pixels=4 * unit,
        max_pixels=args.visual_tokens * unit, local_files_only=True)
    bundle = torch.load(args.adapter, map_location="cpu", weights_only=False)
    wrapped = inject_lora(model, rank=bundle["rank"], alpha=bundle["alpha"],
                          target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                          torch=torch, last_layer_count=None)
    load_lora_state_dict(wrapped, bundle["state"])
    if bundle.get("input_mode", "probe") == "images":
        # note (luojiaxuan): 与训练器同款 —— 36k token 的 eager prefill 会物化
        # L×L 矩阵(72GB)。选帧遍不需要位级对齐,切 flash-attn。
        for mod in model.modules():
            cfg = getattr(mod, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "flash_attention_2"
    print(json.dumps({"adapter": args.adapter.name,
                      "torch_seed": bundle.get("torch_seed"),
                      "modules": len(wrapped)}), flush=True)

    from PIL import Image

    def make_thumb(path: Path, budget: int) -> "Image.Image":
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(1.0, (budget / (w * h)) ** 0.5)
        if scale < 1.0:
            img = img.resize((max(28, int(w * scale)), max(28, int(h * scale))))
        return img

    rows_by_dp = {}
    for line in args.manifest.open(encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            if d["dp_id"] in set(hold):
                rows_by_dp[d["dp_id"]] = d

    done = set()
    if args.output.exists():
        for l in args.output.open(encoding="utf-8"):
            l = l.strip()
            if l:
                done.add(json.loads(l)["dp"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped: dict[str, int] = {}
    thumb_pixels = int(bundle.get("thumb_pixels", 112896))
    with args.output.open("a", encoding="utf-8") as sink:
        for dp in hold:
            if args.limit_states and n >= args.limit_states:
                break
            rec = rows_by_dp.get(dp)
            if rec is None or dp in done:
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
                needed = [root / images[j] for j in cands] + [root / images[s - 1]]
                if not all(x.exists() for x in needed):
                    raise ValueError("missing images")
                if bundle.get("input_mode", "probe") == "images":
                    msgs = build_selector_messages_images(
                        goal=rec["instruction"], cands=cands,
                        frames={j: str(root / images[j]) for j in cands},
                        steps=steps, current_image=str(root / images[s - 1]))
                else:
                    probe = b1.get(dp)
                    if probe is None:
                        raise ValueError("no b1 probe")
                    thumbs = {j: make_thumb(root / images[j], thumb_pixels)
                              for j in cands}
                    msgs = build_selector_messages(
                        goal=rec["instruction"], cands=cands, thumbs=thumbs,
                        probe=probe, current_image=str(root / images[s - 1]))
                enc = proc.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt").to(device)
                pt = int(enc["input_ids"].shape[1])
                with torch.inference_mode():
                    out = model.generate(
                        **enc, do_sample=False, max_new_tokens=16,
                        pad_token_id=proc.tokenizer.eos_token_id,
                        num_beams=1, num_return_sequences=1)
                text = proc.batch_decode(out[:, pt:], skip_special_tokens=True)[0]
                m = re.search(r"SELECT:\s*(\d+)\s*,\s*(\d+)", text)
                recent = frozenset(cands[-2:])
                fallback = False
                if re.search(r"\bKEEP\b", text):
                    chosen = recent
                elif m:
                    a, b = int(m.group(1)), int(m.group(2))
                    if a in cands and b in cands and a != b:
                        chosen = frozenset((a, b))
                    else:
                        chosen, fallback = recent, True
                else:
                    chosen, fallback = recent, True
                subs = table[dp]
                if recent not in subs or (dp in recent_ref
                                          and subs[recent] != recent_ref[dp]):
                    raise SystemExit(f"候选池错位:{dp}")
                sink.write(json.dumps({
                    "dp": dp, "domain": t2d.get(dp, "?"),
                    "chosen": sorted(chosen), "fallback": int(fallback),
                    "raw": text[:60],
                    "c_sel": (int(subs[chosen]) if chosen in subs else None),
                    "c_rec": int(subs[recent]),
                    "moved": int(chosen != recent)}, ensure_ascii=False) + "\n")
                sink.flush()
                n += 1
                if n % 25 == 0:
                    print(json.dumps({"done": n}), flush=True)
            except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
                k = type(exc).__name__ if not str(exc) else str(exc)[:40]
                skipped[k] = skipped.get(k, 0) + 1
                continue
    print(json.dumps({"evaluated": n, "skipped": skipped, "finished": True},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
