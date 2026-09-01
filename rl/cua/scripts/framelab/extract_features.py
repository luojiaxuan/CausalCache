# note (luojiaxuan): selector 特征抽取(路线 A)。视觉侧 = executor(iter_59,
# Qwen2.5-VL 系)自家视觉塔的 merger 输出做均值池化(外审 F8:避免"编码器 A
# 猜编码器 B 觉得什么有用");文本侧 = 同 checkpoint 的 embed_tokens 均值
# 池化(goal / 折叠历史 / 各候选帧的来历 conclusion)。全部 fp16 落盘。
# 帧特征按截图相对路径为键,覆盖全部成功轨迹帧,一次抽取多轮复用。
import glob, json, os, re, sys

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

CKPT = "/data01/jaxan/sglang-omni-rl/cc_recipe/run_full/hf_p3/iter_59"
OUT_DIR = "/data01/jaxan/selector_feats"
ROOTS = sorted(glob.glob("/data01/jaxan/mw/traj_*"))
DEV = "cuda:0"

os.makedirs(OUT_DIR, exist_ok=True)
proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True,
                                     min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
model = AutoModelForImageTextToText.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval()
visual = getattr(model, "visual", None) or model.model.visual
embed = model.get_input_embeddings()
tok = proc.tokenizer


def frame_feat(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    inputs = proc(images=[img], text=["<|image_pad|>"], return_tensors="pt")
    with torch.no_grad():
        out = visual(inputs["pixel_values"].to(DEV, torch.bfloat16),
                     grid_thw=inputs["image_grid_thw"].to(DEV))
    h = getattr(out, "last_hidden_state", out)
    if h.dim() == 3:
        h = h[0]
    return h.mean(0).float().cpu().numpy().astype(np.float16)


def text_feat(s):
    ids = tok(s or " ", return_tensors="pt", truncation=True,
              max_length=512)["input_ids"].to(DEV)
    with torch.no_grad():
        e = embed(ids).mean(1)[0]
    return e.float().cpu().numpy().astype(np.float16)


def conclusions(traj):
    by_step = {}
    for s in traj:
        by_step[s.get("step", 0)] = (s.get("prediction") or "").split("\n")[0] \
            .replace("Action:", "").strip()
    return by_step


def main():
    # ── 帧视觉特征 ──
    done = {}
    fpath = os.path.join(OUT_DIR, "frames.npz")
    if os.path.exists(fpath):
        done = dict(np.load(fpath))
        print(f"已有帧特征 {len(done)}", flush=True)
    frames = []
    for root in ROOTS:
        for d in glob.glob(os.path.join(root, "*/")):
            rp = os.path.join(d, "result.txt")
            try:
                if float(open(rp).read().split("score:")[1].split()[0]) <= 0:
                    continue
            except Exception:  # noqa: BLE001
                continue
            frames.extend(glob.glob(os.path.join(d, "screenshots", "*.png")))
    todo = [f for f in frames if f.replace("/data01/jaxan/mw/", "") not in done]
    print(f"帧总数 {len(frames)},待抽 {len(todo)}", flush=True)
    for i, f in enumerate(todo):
        try:
            done[f.replace("/data01/jaxan/mw/", "")] = frame_feat(f)
        except Exception as e:  # noqa: BLE001
            print(f"ERR frame {f}: {str(e)[:80]}", flush=True)
        if (i + 1) % 2000 == 0:
            np.savez(fpath, **done)
            print(f"  {i+1}/{len(todo)}", flush=True)
    np.savez(fpath, **done)
    print("帧特征完成", flush=True)

    # ── 状态文本特征(goal / 折叠历史 / 候选 conclusion)──
    keys_needed = set()
    for lf in glob.glob("/data01/jaxan/behav_labels_shard*.jsonl") + \
            glob.glob("/data01/jaxan/behav_pilot_shard*.jsonl"):
        for line in open(lf):
            try:
                r = json.loads(line)
                keys_needed.add((r["dir"], r["step"]))
            except Exception:  # noqa: BLE001
                continue
    print(f"状态数 {len(keys_needed)}", flush=True)
    out = {}
    tpath = os.path.join(OUT_DIR, "state_text.npz")
    if os.path.exists(tpath):
        out = dict(np.load(tpath))
    by_dir = {}
    for d, k in sorted(keys_needed):
        by_dir.setdefault(d, []).append(k)
    n = 0
    for d, ks in by_dir.items():
        try:
            traj = list(json.load(open(os.path.join(d, "traj.json"))).values())[0]["traj"]
        except Exception:  # noqa: BLE001
            continue
        conc = conclusions(traj)
        goal = next((s.get("task_goal", "") for s in traj), "")
        for k in ks:
            key = f"{d.replace('/data01/jaxan/mw/', '')}|{k}"
            if key + "|q" in out:
                continue
            folded = " ; ".join(conc.get(i, "") for i in range(1, k))
            out[key + "|q"] = text_feat(goal + " | " + folded[-2000:])
            # 候选帧 file n 的来历 = 第 n-1 步的 conclusion(n=1 为初始屏)
            for n_file in range(1, k):
                ck = f"{key}|c{n_file}"
                if ck not in out:
                    out[ck] = text_feat(conc.get(n_file - 1, "initial screen"))
            n += 1
            if n % 100 == 0:
                np.savez(tpath, **out)
                print(f"  文本 {n}", flush=True)
    np.savez(tpath, **out)
    print("文本特征完成", flush=True)


if __name__ == "__main__":
    main()
