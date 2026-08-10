#!/usr/bin/env python3
"""缓存全清(2560-token)逐帧独立视觉特征(selector v2 阶梯 2.5)。

# note (luojiaxuan): 阶梯 1-3 的共同软肋:selector 的特征全部来自 144-token
# 缩略图,而标签由策略在 **2560-token 全清帧**下的行为产生 —— 决定标签的
# 变量(帧上的文字、小控件状态)在缩略图里像素层面就不存在,y 不是 x 的
# 函数,学习问题不适定。本缓存把决定变量放回输入:
#
#   * 每帧**单独**过一次全清编码(单图 prompt),取 LLM hidden_states[0]
#     的图像段 —— 即视觉塔 + 投影后的纯视觉特征,不含跨帧/文本上下文。
#     **逐帧独立是部署故事的关键**:帧在自己当"当前屏"的那一步本来就被
#     策略全清编码过,缓存复用即可,推理期零额外编码成本;
#   * 2560 个 token 保留完整分辨率读入,再**特征层面**自适应池化到
#     10×16=160 个区域向量(保空间结构)。与缩略图的本质区别:缩略图在
#     **像素层面**先毁掉文字再编码;这里是全清读完后在特征层面压缩。
#
# 产物与 token 缓存同构(toks: [160,4096] × n 帧,ctx: [160,4096]),
# 训练器 train_selector_token_xattn.py 原样复用,协议/划分/参照不变。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--visual-tokens", type=int, default=2560,
                   help="全清预算,与策略动作遍一致 —— 这正是要修的错配")
    p.add_argument("--regions", default="10,16",
                   help="自适应池化后的区域网格(行,列);**raw = 不做第二次池化**,"
                        "逐帧保留全部 ~2560 个 token。区域平均是把 16 个 token 混成"
                        "一个向量 —— 文字级细节在特征层被二次毁掉,这正是 2.5 版"
                        "停在 recency 的头号嫌疑。代价:约 230MB/态、全量 ~290GB"
                        "(盘余 2.1T,放得下)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit-states", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()

    import torch
    import torch.nn.functional as F

    from causalcache.osworld_gui_owl import (
        VISION_PATCH_SIZE,
        VISION_SPATIAL_MERGE_SIZE,
        GUIOwlOSWorldRuntime,
    )

    want: set[str] = set()
    for f in args.labels:
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            subs = [bool(a["c"]) for a in d.get("all", [])
                    if len(a["s"]) == args.budget]
            if any(subs) and not all(subs):
                want.add(d["dp_id"])
    print(json.dumps({"winnable": len(want)}, ensure_ascii=False), flush=True)

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=16)
    model, device, proc = runtime.model, runtime.device, runtime.processor
    raw_mode = args.regions.strip() == "raw"
    rows, cols = (0, 0) if raw_mode else tuple(int(x) for x in args.regions.split(","))
    merge = VISION_SPATIAL_MERGE_SIZE

    def encode_frame(path: str) -> "torch.Tensor":
        """单图 prompt → hidden_states[0] 图像段 → [rows*cols, hidden]。"""
        msgs = [{"role": "user",
                 "content": [{"type": "image", "image": path}]}]
        enc = proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[0][0]                     # 嵌入层 = 纯视觉投影
        mm = enc.get("mm_token_type_ids")
        if mm is None:
            raise ValueError("no mm_token_type_ids")
        mask = (mm[0] == 1)
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            raise ValueError("no image tokens")
        seg = hs[idx[0]: idx[-1] + 1]
        # 用 grid_thw 还原二维结构;合并因子后 R×C 应等于段长
        thw = enc.get("image_grid_thw")
        if thw is None:
            raise ValueError("no image_grid_thw")
        t, h, w = (int(x) for x in thw[0])
        R, C = (h // merge) * max(t, 1), w // merge
        if R * C != seg.shape[0]:
            raise ValueError(f"grid {R}x{C} != {seg.shape[0]} tokens")
        if raw_mode:
            return seg.float()                               # [~2560, D],不再池化
        x = seg.float().reshape(R, C, -1).permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(x, (rows, cols))[0]   # [D, rows, cols]
        return pooled.permute(1, 2, 0).reshape(rows * cols, -1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    skipped: dict[str, int] = {}
    done = 0
    for lineno, line in enumerate(args.manifest.open(encoding="utf-8")):
        if args.limit_states and done >= args.limit_states:
            break
        line = line.strip()
        if not line or lineno % args.shard_count != args.shard_index:
            continue
        rec = json.loads(line)
        dp = rec["dp_id"]
        out = args.outdir / f"{dp}.pt"
        if dp not in want or out.exists():
            continue
        try:
            from causalcache.agentnet_desktop_official import official_step_forms
            s = int(rec["step"])
            images = rec["image_relpaths"]
            if len(images) != s:
                raise ValueError("image count")
            screen = tuple(int(x) for x in rec["screen_size"])
            steps = [official_step_forms(h, screen_size=screen) for h in rec["history"]]
            cands = [j for j in range(1, s - 1) if steps[j].full_response]
            cur = args.image_root / images[s - 1]
            if not cands or not cur.exists():
                raise ValueError("no candidates / missing current")
            if not all((args.image_root / images[j]).exists() for j in cands):
                raise ValueError("missing images")
            toks = [encode_frame(str(args.image_root / images[j])).half().cpu()
                    for j in cands]
            ctx = encode_frame(str(cur)).half().cpu()
            tmp = out.with_suffix(".tmp")
            import torch as _t
            _t.save({"cands": cands, "toks": toks, "ctx": ctx}, tmp)
            tmp.rename(out)
            done += 1
            if done % 50 == 0:
                print(json.dumps({"shard": args.shard_index, "done": done,
                                  "skipped": skipped}, ensure_ascii=False), flush=True)
        except (ValueError, KeyError, OSError, TypeError, RuntimeError) as exc:
            k = type(exc).__name__ if not str(exc) else str(exc)[:40]
            skipped[k] = skipped.get(k, 0) + 1
            continue
    print(json.dumps({"shard": args.shard_index, "done": done,
                      "skipped": skipped, "finished": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
